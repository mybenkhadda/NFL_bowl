"""Step 1 — exhaustive raw data inventory (metadata + bounded samples only).

Row counts are *estimates* (``row_count_kind = "estimated"``) unless the file is small enough
to be counted from the sample, is Parquet/Arrow (exact from metadata), or
:meth:`DataInventory.exact_row_counts` was explicitly run (one streaming pass per file).
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

from .config import Phase01Config
from .reporting import CATEGORICAL, INK, MarkdownReport, add_caption, export_table, new_figure, save_figure
from .sampling import estimate_csv_rows, head_sample, read_header
from .sources import LINK_COLUMNS, SOURCES, match_source
from .utils import PeakMemoryMonitor, get_logger, human_bytes

log = get_logger(__name__)

FORMATS = {
    ".csv": "csv", ".gz": "csv.gz", ".parquet": "parquet", ".pq": "parquet", ".json": "json",
    ".jsonl": "jsonl", ".ndjson": "jsonl", ".arrow": "arrow", ".feather": "feather", ".ipc": "arrow",
    ".zip": "zip", ".md": "doc", ".txt": "doc", ".pdf": "doc",
}
PARTITION_PATTERNS = {
    "season_or_year": r"(?<!\d)(20\d{2})(?!\d)",
    "week": r"week[_-]?(\d{1,2})",
    "part": r"part[_-]?(\d+)",
}


@dataclass
class FileRecord:
    rel_path: str
    file_name: str
    format: str
    size_bytes: int
    size_mb: float
    compression: str
    source: str | None
    domain: str | None
    n_columns: int | None = None
    columns: list[str] = field(default_factory=list)
    declared_dtypes: list[str] = field(default_factory=list)
    sample_inferred_dtypes: list[str] = field(default_factory=list)
    header_matches_spec: bool | None = None
    schema_drift: str = ""
    row_count: int | None = None
    row_count_kind: str = "unknown"          # estimated | exact_sample | exact_metadata | exact_scan
    avg_row_bytes: float | None = None
    partition_info: str = ""
    candidate_keys: list[str] = field(default_factory=list)
    link_columns: list[str] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    sample_rows_read: int | None = None
    sample_read_s: float | None = None
    sample_read_peak_mb: float | None = None
    sample_throughput_mb_s: float | None = None
    est_full_scan_s: float | None = None


class DataInventory:
    def __init__(self, cfg: Phase01Config):
        self.cfg = cfg
        self.root = cfg.paths.data_root

    # ------------------------------------------------------------------ discovery
    def discover_files(self) -> list[Path]:
        if not self.root.exists():
            raise FileNotFoundError(f"DATA_ROOT does not exist: {self.root} (set NFL_BDB_DATA_ROOT)")
        return sorted(p for p in self.root.rglob("*") if p.is_file() and not p.name.startswith("."))

    @staticmethod
    def detect_format(path: Path) -> tuple[str, str]:
        suffixes = [s.lower() for s in path.suffixes]
        if suffixes[-2:] == [".csv", ".gz"]:
            return "csv.gz", "gzip"
        fmt = FORMATS.get(suffixes[-1] if suffixes else "", "other")
        return fmt, ("embedded" if fmt == "parquet" else "none")

    @staticmethod
    def detect_partitions(name: str, spec_value: str | None, spec_col: str | None) -> str:
        if spec_value is not None:
            return f"{spec_col}={spec_value} (from filename, registry rule)"
        hits = [f"{k}={m.group(1)}" for k, rx in PARTITION_PATTERNS.items() if (m := re.search(rx, name, re.I))]
        return "; ".join(hits) + (" (filename pattern — unverified)" if hits else "")

    # ------------------------------------------------------------------ per-file
    def inspect_file(self, path: Path) -> FileRecord:
        rel = path.relative_to(self.root).as_posix()
        fmt, compression = self.detect_format(path)
        size = path.stat().st_size
        spec = match_source(rel)
        rec = FileRecord(
            rel_path=rel, file_name=path.name, format=fmt, size_bytes=size, size_mb=round(size / 1024**2, 2),
            compression=compression, source=spec.name if spec else None, domain=spec.domain if spec else None,
        )
        rec.partition_info = self.detect_partitions(
            path.name, spec.partition_value(path) if spec else None, spec.partition_col if spec else None
        )
        try:
            if fmt in ("csv", "csv.gz"):
                self._inspect_csv(path, rec, spec)
            elif fmt == "parquet":
                self._inspect_parquet(path, rec)
            elif fmt in ("arrow", "feather"):
                self._inspect_ipc(path, rec)
            elif fmt in ("json", "jsonl"):
                self._inspect_json(path, rec, fmt)
        except Exception as exc:  # noqa: BLE001 - inventory must not die on one bad file
            rec.schema_drift = f"INSPECTION ERROR: {type(exc).__name__}: {exc}"
            log.error("inventory failed for %s: %s", rel, exc)
        rec.link_columns = [c for c in rec.columns if c in LINK_COLUMNS or c.endswith("_id")]
        if spec:
            rec.candidate_keys = [" + ".join(k) for k in spec.grain_candidates]
        return rec

    def _inspect_csv(self, path: Path, rec: FileRecord, spec) -> None:
        header = read_header(path)
        rec.columns, rec.n_columns = header, len(header)
        rows, avg, exact = estimate_csv_rows(path, self.cfg.profiling.row_estimate_bytes)
        rec.row_count, rec.avg_row_bytes = rows, round(avg, 1)
        rec.row_count_kind = "exact_sample" if exact else "estimated"

        n = self.cfg.profiling.schema_sample_rows
        t0 = time.perf_counter()
        with PeakMemoryMonitor() as mon:
            sample = head_sample(path, None, n, infer=True)
        dt = time.perf_counter() - t0
        rec.sample_rows_read = sample.height
        rec.sample_read_s = round(dt, 3)
        rec.sample_read_peak_mb = round(mon.peak_mb - mon.start_mb, 1)
        rec.sample_inferred_dtypes = [str(t) for t in sample.dtypes]
        sample_mb = sample.height * avg / 1024**2
        if dt > 0 and sample_mb > 0:
            rec.sample_throughput_mb_s = round(sample_mb / dt, 1)
            rec.est_full_scan_s = round(rec.size_mb / rec.sample_throughput_mb_s, 1)

        if spec:
            rec.declared_dtypes = [str(spec.schema.get(c, "UNDECLARED")) for c in header]
            missing = [c for c in spec.columns if c not in header]
            extra = [c for c in header if c not in spec.schema]
            rec.header_matches_spec = not missing and not extra and header == spec.columns
            if missing or extra:
                rec.schema_drift = f"missing={missing} extra={extra}"
            elif header != spec.columns:
                rec.schema_drift = "column order differs from registry"

    def _inspect_parquet(self, path: Path, rec: FileRecord) -> None:
        import pyarrow.parquet as pq

        md = pq.ParquetFile(path).metadata
        schema = md.schema.to_arrow_schema()
        rec.columns, rec.n_columns = schema.names, len(schema.names)
        rec.sample_inferred_dtypes = [str(t) for t in schema.types]
        rec.row_count, rec.row_count_kind = md.num_rows, "exact_metadata"
        if md.num_row_groups:
            rec.compression = md.row_group(0).column(0).compression

    def _inspect_ipc(self, path: Path, rec: FileRecord) -> None:
        import pyarrow as pa

        with pa.memory_map(str(path)) as src:
            reader = pa.ipc.open_file(src)
            rec.columns = reader.schema.names
            rec.n_columns = len(rec.columns)
            rec.sample_inferred_dtypes = [str(t) for t in reader.schema.types]
            rec.row_count = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
            rec.row_count_kind = "exact_metadata"

    def _inspect_json(self, path: Path, rec: FileRecord, fmt: str) -> None:
        if fmt == "jsonl":
            sample = pl.read_ndjson(path, n_rows=1000) if hasattr(pl, "read_ndjson") else pl.DataFrame()
            rows, avg, exact = estimate_csv_rows(path, self.cfg.profiling.row_estimate_bytes)
            rec.row_count, rec.row_count_kind = rows + (0 if exact else 1), "exact_sample" if exact else "estimated"
        else:
            sample = pl.read_json(path) if rec.size_bytes < 256 * 1024**2 else pl.DataFrame()
            rec.row_count, rec.row_count_kind = sample.height, "exact_scan"
        rec.columns, rec.n_columns = sample.columns, sample.width
        rec.sample_inferred_dtypes = [str(t) for t in sample.dtypes]

    # ------------------------------------------------------------------ orchestration
    def run(self) -> pl.DataFrame:
        records = [self.inspect_file(p) for p in self.discover_files()]
        by_link: dict[str, set[str]] = {}
        for r in records:
            for c in r.link_columns:
                by_link.setdefault(c, set()).add(r.rel_path)
        for r in records:
            related = set()
            for c in r.link_columns:
                related |= {f"{o} (via {c})" for o in by_link[c] if o != r.rel_path}
            r.related_files = sorted(related)
        unmatched = [r.rel_path for r in records if r.source is None and r.format in ("csv", "parquet")]
        if unmatched:
            log.warning("files not in the source registry (add a SourceSpec): %s", unmatched)
        missing_sources = [s for s in SOURCES if not any(r.source == s for r in records)]
        if missing_sources:
            log.warning("registered sources with no file on disk: %s", missing_sources)
        return pl.DataFrame([asdict(r) for r in records], infer_schema_length=None)

    def exact_row_counts(self, inventory: pl.DataFrame, sources: list[str] | None = None) -> pl.DataFrame:
        """One streaming pass per selected CSV (``scan_csv().select(pl.len())``). Expensive on
        large files — check ``est_full_scan_s`` first."""
        out = []
        for row in inventory.filter(pl.col("format") == "csv").iter_rows(named=True):
            if sources and row["source"] not in sources:
                continue
            t0 = time.perf_counter()
            n = pl.scan_csv(self.root / row["rel_path"], infer_schema=False).select(pl.len()).collect().item()
            out.append({"rel_path": row["rel_path"], "exact_rows": n, "count_s": round(time.perf_counter() - t0, 2)})
            log.info("exact count %s = %s rows (%.1fs)", row["rel_path"], f"{n:,}", out[-1]["count_s"])
        if not out:
            return inventory
        counts = pl.DataFrame(out)
        return (
            inventory.join(counts, on="rel_path", how="left")
            .with_columns(
                pl.coalesce("exact_rows", "row_count").alias("row_count"),
                pl.when(pl.col("exact_rows").is_not_null()).then(pl.lit("exact_scan"))
                .otherwise(pl.col("row_count_kind")).alias("row_count_kind"),
            )
            .drop("exact_rows")
        )

    # ------------------------------------------------------------------ outputs
    def export(self, inv: pl.DataFrame) -> list[Path]:
        return export_table(inv, self.cfg.paths.tables, "file_inventory")

    def plot(self, inv: pl.DataFrame) -> list[Path]:
        figs = []
        d = inv.filter(pl.col("format").is_in(["csv", "csv.gz", "parquet"])).sort("size_bytes")
        if d.is_empty():
            return figs
        fig, (ax1, ax2) = new_figure(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        labels = d["file_name"].to_list()
        domain_color = {"combine": CATEGORICAL[0], "nfl": CATEGORICAL[1], "reference": CATEGORICAL[2], "outcome": INK["muted"]}
        colors = [domain_color.get(x, INK["muted"]) for x in d["domain"].to_list()]
        ax1.barh(labels, d["size_mb"].to_list(), color=colors, height=0.6)
        for i, (mb, rows, kind) in enumerate(zip(d["size_mb"], d["row_count"], d["row_count_kind"])):
            tag = "~" if kind == "estimated" else ""
            ax1.text(mb, i, f"  {human_bytes(mb * 1024**2)} · {tag}{(rows or 0):,} rows", va="center", fontsize=9,
                     color=INK["secondary"])
        ax1.set_xscale("log")
        ax1.set_xlabel("Size on disk (MB, log scale)")
        ax1.set_title("Raw files by size (colour = domain)")
        ax1.grid(axis="y", visible=False)
        ax2.barh(labels, d["n_columns"].fill_null(0).to_list(), color=colors, height=0.6)
        ax2.set_xlabel("Number of columns")
        ax2.set_title("Schema width")
        ax2.set_yticklabels([])
        ax2.grid(axis="y", visible=False)
        from matplotlib.patches import Patch

        ax1.legend(handles=[Patch(color=c, label=k) for k, c in domain_color.items()], loc="lower right")
        add_caption(fig, f"Source: DATA_ROOT inventory · {d.height} tabular files · '~' = estimated row count from line-length sampling")
        figs.append(save_figure(fig, self.cfg.paths.figures, "01_inventory_size_and_width"))
        return figs

    def write_report(self, inv: pl.DataFrame, figures: list[Path]) -> Path:
        p = self.cfg.paths
        total = int(inv["size_bytes"].sum())
        rpt = MarkdownReport("01 — Raw data inventory", "Step 1: metadata-only inspection with bounded samples")
        rpt.h2("Summary").kv({
            "Files discovered": inv.height,
            "Total size": human_bytes(total),
            "Tabular files": inv.filter(pl.col("format").is_in(["csv", "csv.gz", "parquet"])).height,
            "Files outside registry": inv.filter(pl.col("source").is_null()).height,
            "Files with schema drift vs registry": inv.filter(pl.col("schema_drift") != "").height,
            "Estimated total rows": f"{int(inv['row_count'].fill_null(0).sum()):,} (mix of estimated/exact — see table)",
        })
        for f in figures:
            rpt.image(f, p.reports_root, "inventory")
        rpt.h2("File inventory").table(inv.select(
            "rel_path", "format", "size_mb", "row_count", "row_count_kind", "n_columns", "source", "domain",
            "partition_info", "header_matches_spec",
        ).sort("size_mb", descending=True))
        rpt.h2("Read performance (bounded head sample) and full-scan estimate").table(inv.select(
            "rel_path", "sample_rows_read", "sample_read_s", "sample_read_peak_mb", "sample_throughput_mb_s", "est_full_scan_s",
        ).sort("est_full_scan_s", descending=True, nulls_last=True))
        rpt.p("`est_full_scan_s` extrapolates the head-sample parse throughput to the whole file. It is a planning "
              "estimate for one single-pass CSV parse, not a benchmark (see 07_performance_benchmarks.md).")
        rpt.h2("Candidate keys and relationships (to be validated in 03/04)").table(inv.select(
            "rel_path", "candidate_keys", "link_columns", "related_files"))
        rpt.h2("Partitioning opportunities")
        parts = inv.filter(pl.col("partition_info") != "")
        rpt.table(parts.select("rel_path", "partition_info")) if parts.height else rpt.p("_No filename partition patterns detected._")
        rpt.bullets([
            "Game tracking is already split by season in the raw files → natural `season=` Hive partition.",
            "Combine tracking carries `draft_year` → candidate `draft_year=` partition (low cardinality).",
            "Do NOT partition by `nfl_id`, `game_id` or `play_id` (high cardinality → many tiny files); "
            "sort by key inside partitions instead so Parquet row-group statistics enable pruning.",
        ])
        drift = inv.filter(pl.col("schema_drift") != "")
        rpt.h2("Schema drift vs registry")
        rpt.table(drift.select("rel_path", "schema_drift")) if drift.height else rpt.p("All headers match `sources.py`.")
        return rpt.save(p.reports_root / "01_data_inventory.md")
