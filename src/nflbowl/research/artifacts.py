"""``ResearchRun`` — the persistent output tree of the unified Milestone 12 notebook and its save / reload helpers.

    RUN_DIR = <drive_root>/unified_research/<experiment_name>/
        config/            unified_config.json (resolved CFG), environment.json, config_history.jsonl
        logs/              run.log (every save, skip, failure), stage tracebacks
        checkpoints/       nflbowl experiments (epoch shards, best.pt, optimizer / scaler / RNG state) — written by nflbowl
        processed_data/    reusable intermediate datasets (Parquet / .npy + provenance sidecar)
        features/          engineered feature tables
        embeddings/        embedding tables (copies of the nflbowl Parquet files) + INDEX.csv
        tables/csv/        every result table as CSV (tables > csv_max_rows rows: Parquet only)
        tables/parquet/    every result table as Parquet
        figures/png|svg|pdf/  every figure; figures/data/ the plotted data (CSV + Parquet)
        reports/markdown/  one report per stage + FINAL_RESEARCH_REPORT.md; reports/html/ the same as HTML
        diagnostics/       validation tables, software checks, data-quality checks
        manifests/         artifacts.jsonl (append-only), artifact_index.json (latest per key), stages.json, research/ (warehouse JSON copies)
        exports/           ZIP archives

Every write is atomic (unique temporary file in the target directory + ``os.replace``, retried on transient Drive /
Windows locks), so a disconnect never leaves a half-written file under its final name. Large files can be written to
local disk first (``local_tmp``) and then copied once to Drive.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..artifacts.store import append_jsonl, atomic_json, atomic_text, atomic_write, sha256_file
from ..utils.reproducibility import code_version, utc_now

LAYOUT = ("config", "logs", "checkpoints", "processed_data", "features", "embeddings", "tables/csv", "tables/parquet", "figures/png",
          "figures/svg", "figures/pdf", "figures/data", "reports/markdown", "reports/html", "diagnostics", "manifests/research", "exports")
STATUS_RUNNING, STATUS_DONE, STATUS_REUSED, STATUS_SKIPPED, STATUS_FAILED = "running", "executed", "reused", "skipped", "failed"
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(name: str) -> str:
    """File-system safe, descriptive artifact name (letters, digits, ``_ . -``)."""
    s = _SAFE.sub("_", str(name)).strip("._")
    if not s:
        raise ValueError(f"empty artifact name from {name!r}")
    return s


def stage_slug(stage: str) -> str:
    """'12.0' → '12_0', '12C' → '12c', 'eda' → 'eda'."""
    return stage.replace(".", "_").lower()


def descriptive_figure_name(name: str, stage: str) -> str:
    """Research figure ids (``AM12_C_03_envelope_vs_quantile_reliability``) → ``12c_envelope_vs_quantile_reliability``;
    any other name gets the stage prefix once."""
    m = re.match(r"^AM12_([0-9A-Z])_\d+_(.+)$", name)
    if m:
        k = m.group(1)
        return safe_name(f"{'12_0' if k == '0' else '12' + k.lower()}_{m.group(2)}")
    slug = stage_slug(stage)
    return safe_name(name if name.lower().startswith(slug) else f"{slug}_{name}")


class ResearchRun:
    """Output tree + artifact manifest of one unified run. ``cfg``: a ``UnifiedConfig`` (or None for defaults)."""

    def __init__(self, root: Path | str, cfg: Any = None, local_tmp: Path | str | None = None):
        self.root = Path(root)
        self.cfg = cfg
        self.local_tmp = Path(local_tmp) if local_tmp else None
        for d in LAYOUT:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "logs" / "run.log"
        self.manifest_path = self.root / "manifests" / "artifacts.jsonl"
        self.index_path = self.root / "manifests" / "artifact_index.json"
        self.stages_path = self.root / "manifests" / "stages.json"
        self._index: dict[str, dict[str, Any]] = self._read_json(self.index_path, {})
        self._stages: dict[str, dict[str, Any]] = self._read_json(self.stages_path, {})
        self.current_stage: str = "run"

    # ================================================================== small helpers
    @staticmethod
    def _read_json(p: Path, default: Any) -> Any:
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else default
        except (OSError, json.JSONDecodeError):
            return default

    def _flag(self, name: str, default: bool = True) -> bool:
        return bool(getattr(self.cfg, name, default)) if self.cfg is not None else default

    def rel(self, p: Path | str) -> str:
        try:
            return Path(p).resolve().relative_to(self.root.resolve()).as_posix()
        except (ValueError, OSError):
            return str(p)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def log(self, msg: str, level: str = "INFO", echo: bool = True) -> None:
        line = f"{utc_now()} [{level}] [{self.current_stage}] {msg}"
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        if echo:
            print(line if level != "INFO" else f"[{self.current_stage}] {msg}", flush=True)

    def _publish_local(self, final: Path, writer: Callable[[Path], None], overwrite: bool = True) -> Path:
        """Write to fast local disk first (when ``local_tmp`` is set), then copy once to the final (Drive) location."""
        if self.local_tmp is None:
            return atomic_write(final, writer, overwrite)
        self.local_tmp.mkdir(parents=True, exist_ok=True)
        tmp_local = self.local_tmp / f".{final.name}.{uuid.uuid4().hex}"
        try:
            writer(tmp_local)
            return atomic_write(final, lambda p: shutil.copyfile(tmp_local, p), overwrite)
        finally:
            tmp_local.unlink(missing_ok=True)

    # ================================================================== manifest
    def register(self, kind: str, name: str, paths: Sequence[Path | str], stage: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        stage = stage or self.current_stage
        rec = {"key": f"{stage}:{kind}:{name}", "kind": kind, "name": name, "stage": stage, "paths": [self.rel(p) for p in paths],
               "created_at": utc_now(), **(meta or {})}
        append_jsonl(rec, self.manifest_path)
        self._index[rec["key"]] = rec
        atomic_json(self._index, self.index_path, overwrite=True)
        return rec

    def find(self, kind: str, name: str, stage: str | None = None) -> dict[str, Any] | None:
        return self._index.get(f"{stage or self.current_stage}:{kind}:{name}")

    def artifacts(self, stage: str | None = None, kind: str | None = None) -> Any:
        import polars as pl

        rows = [{"stage": r["stage"], "kind": r["kind"], "name": r["name"], "paths": "; ".join(r["paths"]), "created_at": r["created_at"],
                 "rows": r.get("rows"), "note": r.get("note")} for r in self._index.values()
                if (stage is None or r["stage"] == stage) and (kind is None or r["kind"] == kind)]
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(
            schema={"stage": pl.String, "kind": pl.String, "name": pl.String, "paths": pl.String, "created_at": pl.String, "rows": pl.Int64, "note": pl.String})

    # ================================================================== stage records
    def stage_begin(self, stage: str, note: str = "") -> None:
        self.current_stage = stage
        rec = self._stages.get(stage, {})
        rec.update({"stage": stage, "status": STATUS_RUNNING, "started_at": utc_now(), "note": note, "error": None})
        self._stages[stage] = rec
        atomic_json(self._stages, self.stages_path, overwrite=True)
        self.log(f"stage started{': ' + note if note else ''}")

    def stage_end(self, stage: str, status: str, note: str = "", **extra: Any) -> None:
        rec = self._stages.get(stage, {"stage": stage})
        rec.update({"status": status, "finished_at": utc_now(), "note": note or rec.get("note", ""), **extra})
        self._stages[stage] = rec
        atomic_json(self._stages, self.stages_path, overwrite=True)
        self.log(f"stage {status}{': ' + note if note else ''}", level="INFO" if status != STATUS_FAILED else "ERROR")
        self.current_stage = "run"

    def stage_fail(self, stage: str, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        p = self.root / "logs" / f"{stage_slug(stage)}_traceback.txt"
        atomic_text(tb, p)
        self.stage_end(stage, STATUS_FAILED, note=f"{type(exc).__name__}: {exc}", error=self.rel(p))

    def stage_status(self, stage: str) -> str | None:
        return self._stages.get(stage, {}).get("status")

    def stage_table(self) -> Any:
        import polars as pl

        rows = [{"stage": k, "status": v.get("status"), "started_at": v.get("started_at"), "finished_at": v.get("finished_at"), "note": v.get("note"),
                 "error": v.get("error")} for k, v in self._stages.items()]
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(
            schema={"stage": pl.String, "status": pl.String, "started_at": pl.String, "finished_at": pl.String, "note": pl.String, "error": pl.String})

    # ================================================================== tables
    def save_table(self, df: Any, name: str, stage: str | None = None, csv: bool | None = None, parquet: bool = True, note: str = "",
                   subdir: str = "tables") -> list[Path]:
        """Result table → ``tables/parquet/<name>.parquet`` + ``tables/csv/<name>.csv`` (nested columns JSON-encoded in the CSV).
        Returns the written paths ([] when saving is disabled or the frame is None)."""
        if df is None or not self._flag("save_tables"):
            return []
        import polars as pl

        stage = stage or self.current_stage
        nm = safe_name(name)
        out: list[Path] = []
        if subdir == "tables":
            pq, cv = self.root / "tables" / "parquet" / f"{nm}.parquet", self.root / "tables" / "csv" / f"{nm}.csv"
        else:
            pq, cv = self.root / subdir / f"{nm}.parquet", self.root / subdir / f"{nm}.csv"
        if parquet:
            out.append(self._publish_local(pq, lambda p: df.write_parquet(p)))
        limit = getattr(self.cfg, "csv_max_rows", 250_000) if self.cfg is not None else 250_000
        write_csv = (df.height <= limit) if csv is None else csv
        if write_csv and df.width:
            flat = df.with_columns([pl.col(c).map_elements(lambda v: json.dumps(v, default=str), return_dtype=pl.String).alias(c)
                                    for c, t in df.schema.items() if t.is_nested() or t == pl.Object])
            out.append(atomic_write(cv, lambda p: flat.write_csv(p)))
        elif not write_csv:
            note = (note + "; " if note else "") + f"CSV omitted ({df.height:,} rows > csv_max_rows)"
        self.register("table" if subdir == "tables" else subdir, nm, out, stage, {"rows": int(df.height), "columns": list(df.columns)[:60], "note": note})
        return out

    def load_table(self, name: str, stage: str | None = None) -> Any:
        """Reload a saved table (Parquet) without recomputation."""
        import polars as pl

        nm = safe_name(name)
        rec = self.find("table", nm, stage) or self.find("processed_data", nm, stage)
        cands = [self.root / p for p in (rec or {}).get("paths", []) if str(p).endswith(".parquet")]
        cands += [self.root / d / f"{nm}.parquet" for d in ("tables/parquet", "processed_data", "diagnostics", "features")]
        for p in cands:
            if p.is_file():
                return pl.read_parquet(p)
        raise FileNotFoundError(f"no saved table {nm!r} under {self.root}")

    def has_table(self, name: str) -> bool:
        nm = safe_name(name)
        return any((self.root / d / f"{nm}.parquet").is_file() for d in ("tables/parquet", "processed_data", "diagnostics", "features"))

    # ================================================================== processed datasets
    def save_processed_dataset(self, obj: Any, name: str, stage: str | None = None, provenance: dict[str, Any] | None = None,
                               overwrite: bool = True) -> list[Path]:
        """Reusable intermediate data: a polars frame → Parquet; a NumPy array → ``.npy``; a dict → JSON. A provenance
        sidecar (``<name>.provenance.json``: stage, code version, time, caller-supplied inputs / hashes) is written next to it."""
        if obj is None or not self._flag("save_processed_data"):
            return []
        import numpy as np

        stage = stage or self.current_stage
        nm = safe_name(name)
        d = self.root / "processed_data"
        if hasattr(obj, "write_parquet"):
            p = d / f"{nm}.parquet"
            paths = [self._publish_local(p, lambda t: obj.write_parquet(t), overwrite)]
            rows = int(obj.height)
        elif isinstance(obj, np.ndarray):
            p = d / f"{nm}.npy"

            def w(t: Path) -> None:
                with open(t, "wb") as f:
                    np.save(f, obj)
            paths = [self._publish_local(p, w, overwrite)]
            rows = int(obj.shape[0]) if obj.ndim else 1
        elif isinstance(obj, dict):
            p = d / f"{nm}.json"
            paths = [atomic_json(obj, p, overwrite)]
            rows = None
        else:
            raise TypeError(f"save_processed_dataset: unsupported type {type(obj).__name__}")
        prov = {"name": nm, "stage": stage, "written_at": utc_now(), "code_version": code_version(), **(provenance or {})}
        paths.append(atomic_json(prov, d / f"{nm}.provenance.json", overwrite=True))
        self.register("processed_data", nm, paths, stage, {"rows": rows})
        return paths

    def load_processed_dataset(self, name: str) -> Any:
        import numpy as np
        import polars as pl

        nm = safe_name(name)
        d = self.root / "processed_data"
        if (d / f"{nm}.parquet").is_file():
            return pl.read_parquet(d / f"{nm}.parquet")
        if (d / f"{nm}.npy").is_file():
            return np.load(d / f"{nm}.npy", mmap_mode="r")
        if (d / f"{nm}.json").is_file():
            return json.loads((d / f"{nm}.json").read_text(encoding="utf-8"))
        raise FileNotFoundError(f"no processed dataset {nm!r} under {d}")

    # ================================================================== figures
    def save_figure(self, fig: Any, name: str, stage: str | None = None, data: dict[str, Any] | None = None, meta: dict[str, Any] | None = None,
                    formats: Sequence[str] | None = None, dpi: int | None = None, close: bool = False) -> dict[str, str]:
        """One figure in every configured format (PNG at ``figure_dpi``, editable SVG, PDF) + its plotted data
        (``figures/data/<name>__<part>.csv|parquet``) + a JSON sidecar. Overwrite protection with
        ``overwrite_figures = False`` (an existing PNG keeps the old files and the call is logged)."""
        if fig is None or not self._flag("save_figures"):
            return {}
        stage = stage or self.current_stage
        nm = safe_name(name)
        formats = tuple(formats or (getattr(self.cfg, "figure_formats", ("png", "svg", "pdf")) if self.cfg is not None else ("png", "svg", "pdf")))
        dpi = int(dpi or (getattr(self.cfg, "figure_dpi", 300) if self.cfg is not None else 300))
        overwrite = self._flag("overwrite_figures")
        if not overwrite and (self.root / "figures" / "png" / f"{nm}.png").is_file():
            self.log(f"figure {nm} exists and overwrite_figures is False: kept the saved version", echo=False)
            return {"png": self.rel(self.root / "figures" / "png" / f"{nm}.png")}
        out: dict[str, str] = {}
        paths: list[Path] = []
        for fmt in formats:
            p = self.root / "figures" / fmt / f"{nm}.{fmt}"
            atomic_write(p, lambda t, _f=fmt: fig.savefig(t, format=_f, dpi=dpi, bbox_inches="tight"))
            out[fmt] = self.rel(p)
            paths.append(p)
        for part, df in (data or {}).items():
            if df is not None and getattr(df, "height", 0):
                base = self.root / "figures" / "data" / f"{nm}__{safe_name(part)}"
                pq = atomic_write(base.with_suffix(".parquet"), lambda t, _d=df: _d.write_parquet(t))
                try:
                    import polars as pl

                    flat = df.select([c for c, t in df.schema.items() if not (t.is_nested() or t == pl.Object)])
                    cv = atomic_write(base.with_suffix(".csv"), lambda t, _d=flat: _d.write_csv(t))
                    paths.append(cv)
                except Exception:                                   # noqa: BLE001 — the Parquet copy is kept
                    pass
                paths.append(pq)
                out[f"data:{part}"] = self.rel(pq)
        atomic_json({"figure": nm, "stage": stage, "formats": list(formats), "dpi": dpi, "saved_at": utc_now(), **(meta or {})},
                    self.root / "figures" / "data" / f"{nm}.meta.json", overwrite=True)
        self.register("figure", nm, paths, stage, {"note": (meta or {}).get("source", "")})
        if close:
            import matplotlib.pyplot as plt

            plt.close(fig)
        return out

    def show(self, fig: Any, name: str, stage: str | None = None, data: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> None:
        """Display inline, save in every format, close."""
        try:
            from IPython.display import display

            display(fig)
        except ImportError:
            pass
        self.save_figure(fig, name, stage, data, meta)
        import matplotlib.pyplot as plt

        plt.close(fig)

    # ================================================================== reports / metadata
    def save_markdown(self, text: str, name: str, stage: str | None = None, html: bool = True) -> list[Path]:
        """``reports/markdown/<name>.md`` (+ ``reports/html/<name>.html``)."""
        if not self._flag("save_reports"):
            return []
        nm = safe_name(name.removesuffix(".md"))
        out = [atomic_text(text, self.root / "reports" / "markdown" / f"{nm}.md")]
        if html:
            out.append(atomic_text(markdown_to_html(text, nm), self.root / "reports" / "html" / f"{nm}.html"))
        self.register("report", nm, out, stage)
        return out

    def save_json(self, obj: Any, name: str, stage: str | None = None, subdir: str = "manifests") -> Path:
        p = atomic_json(obj, self.root / subdir / f"{safe_name(name)}.json", overwrite=True)
        self.register("json", safe_name(name), [p], stage)
        return p

    def save_model_metadata(self, meta: dict[str, Any], name: str, stage: str | None = None) -> Path | None:
        """Model configuration / checkpoint pointers / hashes → ``checkpoints/<name>.model.json`` (weights stay where
        the trainer wrote them; this never moves or overwrites a checkpoint)."""
        if not self._flag("save_models"):
            return None
        p = atomic_json({**meta, "written_at": utc_now()}, self.root / "checkpoints" / f"{safe_name(name)}.model.json", overwrite=True)
        self.register("model_metadata", safe_name(name), [p], stage)
        return p

    # ================================================================== copying existing artifacts in
    def copy_in(self, src: Path, dest_rel: str, stage: str | None = None, kind: str = "copy", overwrite: bool = True) -> Path | None:
        """Copy an existing file (e.g. a warehouse figure or a nflbowl view) into the run tree (atomic)."""
        src = Path(src)
        if not src.is_file():
            return None
        dst = self.root / dest_rel
        if dst.is_file() and not overwrite:
            return dst
        atomic_write(dst, lambda t: shutil.copyfile(src, t))
        self.register(kind, safe_name(Path(dest_rel).name), [dst], stage, {"note": f"copied from {src}"})
        return dst

    # ================================================================== inventory / export
    def inventory(self) -> Any:
        """Every file under RUN_DIR with its size (no hashing)."""
        import polars as pl

        rows = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                rel = p.relative_to(self.root).as_posix()
                rows.append({"path": rel, "area": rel.split("/")[0], "bytes": p.stat().st_size})
        return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"path": pl.String, "area": pl.String, "bytes": pl.Int64})

    def export_zip(self, include: Iterable[str] = ("config", "manifests", "tables", "figures", "reports", "diagnostics", "logs")) -> Path:
        import zipfile

        stamp = utc_now().replace(":", "").replace("-", "")
        out = self.root / "exports" / f"{self.root.name}_{stamp}.zip"

        def w(tmp: Path) -> None:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for d in include:
                    for f in sorted((self.root / d).rglob("*")):
                        if f.is_file() and not f.name.startswith("."):
                            z.write(f, f"{self.root.name}/{f.relative_to(self.root).as_posix()}")

        self._publish_local(out, w, overwrite=False)
        self.register("export", out.name, [out], "run")
        return out


def markdown_to_html(text: str, title: str) -> str:
    """Markdown → standalone HTML (``markdown`` package when installed, else a pre-formatted fallback)."""
    try:
        import markdown

        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    except Exception:                                               # noqa: BLE001
        import html

        body = f"<pre>{html.escape(text)}</pre>"
    style = ("body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
             "table{border-collapse:collapse;font-size:0.9rem}td,th{border:1px solid #ccc;padding:3px 6px}code,pre{background:#f5f5f5}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{style}</style></head><body>{body}</body></html>"


def file_sha256(p: Path) -> str:
    return sha256_file(Path(p))


def wait_for(path: Path, timeout_s: float = 30.0) -> bool:
    """Drive can expose a just-copied file with a delay; wait until it is visible (bounded)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if Path(path).exists():
            return True
        time.sleep(0.5)
    return False


__all__ = ["LAYOUT", "ResearchRun", "STATUS_DONE", "STATUS_FAILED", "STATUS_REUSED", "STATUS_RUNNING", "STATUS_SKIPPED", "descriptive_figure_name",
           "file_sha256", "markdown_to_html", "safe_name", "stage_slug", "wait_for"]
