"""Analytics Lab · serving snapshot ``warehouse/serving/nfl_lab.duckdb``.

Why a materialised snapshot (not views):
* ``catalog/bdb27.duckdb`` embeds absolute *Windows* paths → unusable inside Linux containers.
* Every lab mart is small (tens of MB in total), so copying them into one DuckDB file makes the
  file fully self-contained: Superset opens it read-only with ``enable_external_access = false``
  and never touches the warehouse Parquet files.
* The file is written to a temp name and swapped in atomically, so readers see either the old
  or the new snapshot. Superset uses one connection per query (NullPool), so a swap between
  queries is safe. Streamlit keeps its ATTACH open (cached), so on Windows the swap fails while it
  runs: ``ensure_replaceable`` detects that BEFORE building and raises ServingFileLockedError with
  the exact stop/re-run commands; the existing snapshot is never modified.

Layout: schema ``lab`` = one table per mart (sorted by its key); ``lab.mart_catalog`` documents
grain / key / rows / source path / build time of every table.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import polars as pl

from ..data.config import Phase01Config, duckdb_connect
from ..data.utils import get_logger, sql_path, utc_now
from .marts import LabArtifactStore

log = get_logger(__name__)

SERVING_DB_NAME = "nfl_lab.duckdb"


def serving_dir(cfg: Phase01Config) -> Path:
    return cfg.paths.warehouse_root / "serving"


class ServingFileLockedError(PermissionError):
    """The serving snapshot is held open by another process; the existing file was left untouched."""


LOCK_HELP = (
    "{path} is in use by another process (Windows refuses to replace an open file; WinError 5/32). "
    "The existing snapshot was NOT modified. Release it, then re-run only the serving step:\n"
    "  cd infrastructure\\analytics_lab && docker compose stop superset streamlit grafana\n"
    "  (also close DBeaver / DuckDB CLI / notebooks that opened it)\n"
    "  cd ..\\.. && python -m nflbdb.lab.build --steps serving,monitoring\n"
    "  cd infrastructure\\analytics_lab && docker compose start superset streamlit")


def ensure_replaceable(target: Path) -> None:
    """Fail fast (before building) if ``target`` cannot be replaced.

    Probe = rename the file to a sibling name and straight back: Windows refuses the rename while any
    handle is open, so this detects the lock without opening, truncating or rewriting the data.
    A leftover ``<target>.wal`` would be replayed against the NEW file → refuse instead of risking it.
    """
    wal = target.with_name(target.name + ".wal")
    if wal.exists():
        raise ServingFileLockedError(f"{wal} exists: a process has (or had) {target.name} open read-write. "
                                     + LOCK_HELP.format(path=target))
    if not target.exists():
        return
    probe = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.lockprobe")
    try:
        os.rename(target, probe)
    except PermissionError as exc:
        raise ServingFileLockedError(LOCK_HELP.format(path=target)) from exc
    os.rename(probe, target)


def replace_snapshot(tmp: Path, target: Path, attempts: int = 5, delay_s: float = 1.0) -> None:
    """Atomically swap ``tmp`` in (``os.replace``: old OR new, never partial). Retries short-lived
    locks (antivirus / indexer scanning the fresh file); on final failure deletes ``tmp`` only."""
    for i in range(attempts):
        try:
            os.replace(tmp, target)
            return
        except PermissionError as exc:
            last = exc
            if i < attempts - 1:
                log.warning("serving: %s locked (attempt %d/%d) — retrying in %.1fs", target.name, i + 1, attempts,
                            delay_s)
                time.sleep(delay_s)
    tmp.unlink(missing_ok=True)
    raise ServingFileLockedError(LOCK_HELP.format(path=target)) from last


class ServingSnapshot:
    def __init__(self, cfg: Phase01Config, store: LabArtifactStore | None = None):
        self.cfg = cfg
        self.store = store or LabArtifactStore(cfg)
        self.target = serving_dir(cfg) / SERVING_DB_NAME

    def build(self) -> dict:
        datasets = self.store._manifest().get("datasets", {})
        if not datasets:
            raise FileNotFoundError("no lab marts registered — run the marts step first")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        ensure_replaceable(self.target)
        tmp = self.target.with_name(f".{self.target.name}.{uuid.uuid4().hex[:8]}.tmp")
        con = duckdb_connect(self.cfg, tmp)
        rows = []
        built = False
        try:
            con.execute("CREATE SCHEMA lab")
            for name, entry in sorted(datasets.items()):
                files = self.store.dataset_files(name)
                if not files:
                    log.warning("serving: %s registered but no files — skipped", name)
                    continue
                listing = ", ".join(f"'{sql_path(f)}'" for f in files)
                order = ", ".join(entry.get("key", []))
                con.execute(f"CREATE TABLE lab.{name} AS SELECT * FROM read_parquet([{listing}], hive_partitioning = true)"
                            + (f" ORDER BY {order}" if order else ""))
                n = con.execute(f"SELECT count(*) FROM lab.{name}").fetchone()[0]
                rows.append({"table_name": name, "grain": entry.get("grain", ""), "key": ", ".join(entry.get("key", [])),
                             "rows": n, "version": entry.get("version", ""), "mart_built_at": entry.get("built_at", ""),
                             "source_path": entry.get("path", ""), "snapshot_built_at": utc_now()})
            catalog = pl.DataFrame(rows)
            con.register("mart_catalog_df", catalog.to_arrow())
            con.execute("CREATE TABLE lab.mart_catalog AS SELECT * FROM mart_catalog_df")
            con.unregister("mart_catalog_df")
            con.execute("CHECKPOINT")
            built = True
        finally:
            con.close()
            if not built:                      # failed build: drop the partial temp file, keep the old snapshot
                tmp.unlink(missing_ok=True)
                tmp.with_name(tmp.name + ".wal").unlink(missing_ok=True)
        replace_snapshot(tmp, self.target)
        for leftover in self.target.parent.glob(f".{self.target.name}.*.tmp.wal"):
            leftover.unlink(missing_ok=True)
        size = self.target.stat().st_size
        log.info("serving snapshot %s: %d tables, %.1f MB", self.cfg.paths.rel(self.target), len(rows), size / 1024**2)
        return {"path": self.cfg.paths.rel(self.target), "tables": len(rows), "bytes": size}
