"""Bronze layer — raw data preserved, made columnar.

* The raw CSVs in DATA_ROOT are never modified. Bronze is a *lossless typed mirror*: one
  Parquet file per raw file, same rows, same columns, plus provenance columns
  ``_src_file`` (relative path) and ``_src_row`` (0-based data-row index in the source file,
  which lets later checks reason about original file order).
* No cleaning, no dedup, no unit changes. Strings that look boolean stay strings.
* Partition values that live only in the filename (game tracking season) become a Hive
  directory (``season=2023/``) instead of a duplicated column.
* Incremental: a file is re-ingested only if its fingerprint (size + mtime), the declared
  schema, or :data:`BRONZE_VERSION` changed. State lives in ``catalog/bronze_state.json``.
* Writes are atomic (temp file + ``os.replace``).

Cost: this is the ONE full pass over the raw CSVs. Use :meth:`BronzeIngestor.estimate` first.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from .config import Phase01Config
from .csv_schema import (ColumnResolution, SchemaResolutionError, SchemaResolver, read_csv_strings,
                         resolutions_frame, strict_cast_exprs)
from .reporting import export_table
from .sampling import offset_sample, read_header
from .sources import SOURCES, SourceSpec
from .utils import (atomic_path, atomic_write_json, file_fingerprint, get_logger, human_bytes, read_json,
                    stable_hash, utc_now)

log = get_logger(__name__)

BRONZE_VERSION = "2"  # v2: raw strings + evidence-based strict casting (csv_schema)


class SchemaDriftError(RuntimeError):
    """Raised when a raw file is missing columns declared in the registry."""


class BronzeIngestor:
    def __init__(self, cfg: Phase01Config):
        self.cfg = cfg
        self.state_path = cfg.paths.catalog / "bronze_state.json"
        self.resolver = SchemaResolver(cfg.paths.catalog)
        self.resolutions: list[ColumnResolution] = []

    # ------------------------------------------------------------------ paths
    def output_path(self, spec: SourceSpec, src: Path) -> Path:
        base = self.cfg.paths.bronze / spec.name
        part = spec.partition_value(src)
        if part is not None:
            base = base / f"{spec.partition_col}={part}"
        return base / f"{src.stem}.parquet"

    def source_dir(self, name: str) -> Path:
        return self.cfg.paths.bronze / name

    # ------------------------------------------------------------------ manifest
    def build_manifest(self, with_sha256: bool = True) -> pl.DataFrame:
        """Provenance manifest of every raw file. SHA-256 reads every byte once (IO-bound);
        pass ``with_sha256=False`` for a quick size/mtime-only manifest."""
        rows = []
        for spec in SOURCES.values():
            for f in spec.files(self.cfg.paths.data_root):
                t0 = time.perf_counter()
                fp = file_fingerprint(f, with_sha256=with_sha256)
                rows.append({
                    "source": spec.name, "rel_path": f.relative_to(self.cfg.paths.data_root).as_posix(),
                    **fp, "hash_s": round(time.perf_counter() - t0, 2), "recorded_at": utc_now(),
                })
        df = pl.DataFrame(rows)
        export_table(df, self.cfg.paths.catalog, "bronze_manifest")
        return df

    # ------------------------------------------------------------------ cost estimate
    def estimate(self, inventory: pl.DataFrame | None = None) -> pl.DataFrame:
        """Bytes to convert and an extrapolated time per file (from inventory sample throughput)."""
        rows = []
        inv = {r["rel_path"]: r for r in inventory.iter_rows(named=True)} if inventory is not None else {}
        state = read_json(self.state_path, {})
        for spec in SOURCES.values():
            for f in spec.files(self.cfg.paths.data_root):
                rel = f.relative_to(self.cfg.paths.data_root).as_posix()
                up_to_date = self._is_current(spec, f, state)
                rows.append({
                    "source": spec.name, "rel_path": rel, "size": human_bytes(f.stat().st_size),
                    "est_parse_s": (inv.get(rel) or {}).get("est_full_scan_s"),
                    "action": "skip (up to date)" if up_to_date else "ingest",
                })
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------ ingestion
    def _schema_hash(self, spec: SourceSpec) -> str:
        return stable_hash({c: str(t) for c, t in spec.schema.items()} | {"_v": BRONZE_VERSION})

    def _is_current(self, spec: SourceSpec, src: Path, state: dict) -> bool:
        rel = src.relative_to(self.cfg.paths.data_root).as_posix()
        entry = state.get(rel)
        fp = file_fingerprint(src)
        return bool(
            entry
            and entry.get("size_bytes") == fp["size_bytes"]
            and entry.get("mtime_ns") == fp["mtime_ns"]
            and entry.get("schema_hash") == self._schema_hash(spec)
            and Path(entry.get("output", "")).exists()
        )

    def check_header(self, spec: SourceSpec, src: Path) -> list[str]:
        header = read_header(src)
        missing = [c for c in spec.columns if c not in header]
        if missing:
            raise SchemaDriftError(f"{src.name}: columns missing vs registry: {missing}")
        extra = [c for c in header if c not in spec.schema]
        if extra:
            log.warning("%s: %d undeclared column(s) ingested as String: %s", src.name, len(extra), extra)
        return header

    def resolve_schema(self, spec: SourceSpec, files: list[Path]) -> dict[str, pl.DataType]:
        """Test declared types against the raw strings (csv_schema rules) and persist the result.
        Non-tracking sources: exact full scan. Tracking sources: deterministic byte-offset sample
        across each file; any value the sample missed still fails the strict cast at write time."""
        header = self.check_header(spec, files[0])
        pc = self.cfg.profiling
        if spec.is_tracking:
            frames = [offset_sample(f, spec, pc.offset_sample_chunks, pc.offset_sample_lines, pc.seed) for f in files]
            _, res = self.resolver.resolve_files(spec, files, sample_frames=frames)
        else:
            _, res = self.resolver.resolve_files(spec, files)
        self.resolutions += res
        self.resolver.raise_on_errors(res)
        schema = self.resolver.resolved_schema(spec, res, header)
        self.resolver.save(spec, schema, res)
        for r in res:
            if r.action != "keep":
                log.warning("type resolution %s.%s: declared %s → %s (%s; failing values %s; %s)", r.table, r.column,
                            r.declared, r.resolved, r.action, r.fail_examples, r.evidence)
        return schema

    def ingest_file(self, spec: SourceSpec, src: Path, force: bool = False,
                    schema: dict[str, pl.DataType] | None = None) -> dict:
        state = read_json(self.state_path, {})
        rel = src.relative_to(self.cfg.paths.data_root).as_posix()
        out = self.output_path(spec, src)
        if not force and self._is_current(spec, src, state):
            log.info("bronze skip (up to date): %s", rel)
            return {**state[rel], "action": "skipped"}

        header = self.check_header(spec, src)
        schema = schema or self.resolve_schema(spec, spec.files(self.cfg.paths.data_root))
        t0 = time.perf_counter()
        lf = (
            read_csv_strings(src, null_values=spec.null_values, lazy=True)       # raw strings: never fails
            .with_columns(strict_cast_exprs(schema, header))                      # explicit, strict typing
            .with_row_index("_src_row")
            .with_columns(pl.lit(rel).alias("_src_file"))
        )
        try:
            with atomic_path(out) as tmp:
                lf.sink_parquet(tmp, **self.cfg.parquet.polars_kwargs())
        except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as exc:
            raise SchemaResolutionError(
                f"{rel}: strict cast failed during bronze write — a value not covered by schema resolution "
                f"({'sampled' if spec.is_tracking else 'full scan'}). Original error: {exc}"
            ) from exc
        seconds = round(time.perf_counter() - t0, 2)
        md = pq.ParquetFile(out).metadata
        entry = {
            **file_fingerprint(src), "source": spec.name, "output": str(out),
            "rows": md.num_rows, "row_groups": md.num_row_groups, "out_bytes": out.stat().st_size,
            "schema_hash": self._schema_hash(spec), "seconds": seconds, "written_at": utc_now(),
            "columns": list(header), "schema": {c: str(schema[c]) for c in header},
        }
        state[rel] = entry
        atomic_write_json(state, self.state_path)
        log.info("bronze %s: %s rows, %s → %s in %.1fs", rel, f"{md.num_rows:,}",
                 human_bytes(src.stat().st_size), human_bytes(entry["out_bytes"]), seconds)
        return {**entry, "action": "ingested"}

    def ingest(self, sources: list[str] | None = None, force: bool = False) -> pl.DataFrame:
        results = []
        for spec in SOURCES.values():
            if sources and spec.name not in sources:
                continue
            files = spec.files(self.cfg.paths.data_root)
            if not files:
                log.warning("no raw file for source %s", spec.name)
                continue
            state = read_json(self.state_path, {})
            pending = force or not all(self._is_current(spec, f, state) for f in files)
            schema = self.resolve_schema(spec, files) if pending else None  # one shared schema per source
            for f in files:
                r = self.ingest_file(spec, f, force=force, schema=schema)
                results.append({k: r.get(k) for k in ("source", "output", "rows", "row_groups", "size_bytes",
                                                      "out_bytes", "seconds", "action")}
                               | {"rel_path": f.relative_to(self.cfg.paths.data_root).as_posix()})
        df = pl.DataFrame(results)
        if not df.is_empty():
            df = df.with_columns((pl.col("out_bytes") / pl.col("size_bytes")).round(3).alias("parquet_to_csv_ratio"))
            export_table(df.with_columns(pl.col("output").map_elements(self.cfg.paths.rel, return_dtype=pl.String)),
                         self.cfg.paths.tables, "bronze_ingest_summary")
        if self.resolutions:
            export_table(resolutions_frame(self.resolutions), self.cfg.paths.tables, "schema_resolution")
        return df
