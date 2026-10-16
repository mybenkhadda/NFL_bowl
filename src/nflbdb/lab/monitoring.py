"""Analytics Lab · engineering-monitoring snapshot ``warehouse/serving/lab_monitoring.sqlite``.

Grafana (frser-sqlite-datasource) reads this file read-only. Every table holds REAL measured
values; nothing is simulated. Snapshot tables are replaced on each export; ``lab_build`` is
append-only (one row per lab build), so its history is genuine.

Tables
    pipeline_step     reports/tables/pipeline_run_log (all steps, incl. lab_* steps)
    storage_layer     bytes / files per warehouse layer, measured now (raw data dir: stat only)
    dataset_file      every bronze/silver/gold Parquet file: rows (footer), bytes, mtime, age
    gold_build        catalog/gold_manifest (rows, bytes, build seconds per gold table)
    benchmark         reports/tables/benchmark_results
    lab_build         append-only lab build log
Time columns are UNIX epoch seconds named ``time`` (plugin default time column).
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from ..data.config import Phase01Config
from ..data.tables import LAYERS
from ..data.utils import get_logger
from .serving import serving_dir

log = get_logger(__name__)

MONITORING_DB_NAME = "lab_monitoring.sqlite"


def _epoch(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        return int(datetime.strptime(iso.replace("Z", ""), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


def _sqlite_type(dt: pl.DataType) -> str:
    if dt.is_integer() or dt == pl.Boolean:
        return "INTEGER"
    if dt.is_float():
        return "REAL"
    return "TEXT"


def write_table(conn: sqlite3.Connection, name: str, df: pl.DataFrame) -> None:
    """Replace ``name`` with ``df`` (lists/structs/dates are stored as text)."""
    df = df.with_columns([pl.col(c).cast(pl.String) for c, dt in df.schema.items()
                          if _sqlite_type(dt) == "TEXT" and dt != pl.String])
    df = df.with_columns([pl.col(c).cast(pl.Int64) for c, dt in df.schema.items() if dt == pl.Boolean])
    conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    cols = ", ".join(f'"{c}" {_sqlite_type(dt)}' for c, dt in df.schema.items())
    conn.execute(f'CREATE TABLE "{name}" ({cols})')
    if df.height:
        marks = ", ".join("?" for _ in df.columns)
        conn.executemany(f'INSERT INTO "{name}" VALUES ({marks})', df.rows())


class MonitoringExporter:
    def __init__(self, cfg: Phase01Config):
        self.cfg = cfg
        self.target = serving_dir(cfg) / MONITORING_DB_NAME

    # ------------------------------------------------------------------ frames
    def _read_report(self, stem: str) -> pl.DataFrame | None:
        for suffix in (".parquet", ".csv"):
            p = self.cfg.paths.tables / f"{stem}{suffix}"
            if p.is_file():
                return pl.read_parquet(p) if suffix == ".parquet" else pl.read_csv(p, infer_schema_length=None)
        return None

    def pipeline_steps(self) -> pl.DataFrame:
        df = self._read_report("pipeline_run_log")
        if df is None or df.is_empty():
            return pl.DataFrame(schema={"time": pl.Int64, "run_id": pl.String, "step": pl.String})
        keep = [c for c in ("run_id", "step", "started_at", "seconds", "rss_start_mb", "rss_peak_mb", "rss_end_mb",
                            "rows_in", "rows_out", "bytes_in", "bytes_out", "status", "error", "warnings") if c in df.columns]
        df = df.select(keep).with_columns(pl.col("started_at").cast(pl.String))
        return df.with_columns(
            pl.col("started_at").map_elements(_epoch, return_dtype=pl.Int64).alias("time"),
            pl.col("step").str.replace(r" \[.*\]$", "").alias("step_name"),
            (pl.col("warnings").fill_null("").str.len_chars() > 0).alias("has_warnings") if "warnings" in keep
            else pl.lit(False).alias("has_warnings"),
        ).drop("warnings", strict=False)

    def _dir_stats(self, root: Path, pattern: str = "*") -> tuple[int, int]:
        n = size = 0
        if root.exists():
            for p in root.rglob(pattern):
                if p.is_file() and not p.name.startswith("."):
                    n += 1
                    size += p.stat().st_size
        return n, size

    def storage_layers(self, now: int) -> pl.DataFrame:
        rows = []
        n, b = self._dir_stats(self.cfg.paths.data_root, "*.csv")   # stat only; raw files are never opened
        rows.append({"layer": "raw (CSV)", "n_files": n, "bytes": b})
        for layer in LAYERS:
            n, b = self._dir_stats(getattr(self.cfg.paths, layer), "*.parquet")
            rows.append({"layer": layer, "n_files": n, "bytes": b})
        n, b = self._dir_stats(serving_dir(self.cfg))
        rows.append({"layer": "serving", "n_files": n, "bytes": b})
        return pl.DataFrame(rows).with_columns(pl.lit(now).alias("time"), (pl.col("bytes") / 1024**2).alias("mb"))

    def dataset_files(self, now: int) -> pl.DataFrame:
        import pyarrow.parquet as pq

        rows = []
        for layer in LAYERS:
            base: Path = getattr(self.cfg.paths, layer)
            if not base.exists():
                continue
            for f in sorted(base.rglob("*.parquet")):
                rel_parts = f.relative_to(base).parts
                if any(p.startswith((".", "_")) for p in rel_parts):
                    continue
                st = f.stat()
                if rel_parts[0].startswith("phase_") and len(rel_parts) > 1:   # gold/phase_NN/<dataset>[/...]
                    dataset = f"{rel_parts[0]}/{rel_parts[1].removesuffix('.parquet')}"
                else:                                                          # <table>.parquet or <table>/<part>/…
                    dataset = rel_parts[0].removesuffix(".parquet")
                rows.append({"layer": layer, "dataset": dataset, "file": self.cfg.paths.rel(f), "rows": pq.ParquetFile(f).metadata.num_rows,
                             "bytes": st.st_size, "modified": int(st.st_mtime),
                             "age_hours": round((now - st.st_mtime) / 3600, 2)})
        schema = {"layer": pl.String, "dataset": pl.String, "file": pl.String, "rows": pl.Int64, "bytes": pl.Int64,
                  "modified": pl.Int64, "age_hours": pl.Float64}
        return pl.DataFrame(rows, schema=schema).with_columns(pl.col("modified").alias("time"))

    def gold_builds(self) -> pl.DataFrame:
        p = self.cfg.paths.catalog / "gold_manifest.parquet"
        if not p.is_file():
            return pl.DataFrame(schema={"time": pl.Int64, "table": pl.String})
        return pl.read_parquet(p).with_columns(
            pl.col("built_at").cast(pl.String).map_elements(_epoch, return_dtype=pl.Int64).alias("time"))

    def benchmarks(self) -> pl.DataFrame:
        df = self._read_report("benchmark_results")
        return df if df is not None else pl.DataFrame(schema={"bench": pl.String})

    # ------------------------------------------------------------------ export
    def export(self, lab_build: dict) -> dict:
        """``lab_build``: {seconds, status, n_built, n_reused, serving_bytes, rss_peak_mb, error}."""
        now = int(time.time())
        self.target.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.target.with_name(f".{self.target.name}.{uuid.uuid4().hex[:8]}.tmp")
        if self.target.exists():
            shutil.copy2(self.target, tmp)      # keep append-only history (lab_build)
        conn = sqlite3.connect(tmp)
        try:
            conn.execute("PRAGMA journal_mode = DELETE")  # no WAL: readers use a read-only mount
            tables = {"pipeline_step": self.pipeline_steps(), "storage_layer": self.storage_layers(now),
                      "dataset_file": self.dataset_files(now), "gold_build": self.gold_builds(),
                      "benchmark": self.benchmarks()}
            for name, df in tables.items():
                write_table(conn, name, df)
            conn.execute("""CREATE TABLE IF NOT EXISTS lab_build (time INTEGER, seconds REAL, status TEXT,
                            n_built INTEGER, n_reused INTEGER, serving_bytes INTEGER, rss_peak_mb REAL, error TEXT)""")
            conn.execute("INSERT INTO lab_build VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (now, lab_build.get("seconds"), lab_build.get("status"), lab_build.get("n_built"),
                          lab_build.get("n_reused"), lab_build.get("serving_bytes"), lab_build.get("rss_peak_mb"),
                          lab_build.get("error", "")))
            write_table(conn, "export_meta", pl.DataFrame([{"time": now, "exported_by": "nflbdb.lab.monitoring"}]))
            conn.commit()
        finally:
            conn.close()
        try:
            os.replace(tmp, self.target)
        except PermissionError as exc:
            tmp.unlink(missing_ok=True)
            raise PermissionError(f"cannot replace {self.cfg.paths.rel(self.target)}: stop Grafana "
                                  "(`docker compose --profile monitoring stop grafana`) and re-run.") from exc
        log.info("monitoring snapshot %s (%d tables)", self.cfg.paths.rel(self.target), len(tables) + 2)
        return {"path": self.cfg.paths.rel(self.target), "bytes": self.target.stat().st_size}
