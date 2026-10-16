"""Analytics Lab · build orchestration (marts → serving snapshot → monitoring snapshot).

Run by the USER on the host (conda env ``nfl``)::

    set PYTHONPATH=src
    set NFL_BDB_DUCKDB_MEMORY=3GB
    python -m nflbdb.lab.build                    # incremental: unchanged marts are reused
    python -m nflbdb.lab.build --force            # rebuild every mart (old versions archived)
    python -m nflbdb.lab.build --steps serving    # only re-materialise the serving snapshot

Every step is timed and memory-monitored through :class:`~nflbdb.data.utils.RunLog` and appended
to ``reports/tables/pipeline_run_log`` (steps ``lab_marts``, ``lab_serving``, ``lab_monitoring``).
Inputs are opened read-only; Phase 01 / Phase 02 outputs are never modified.
"""

from __future__ import annotations

import argparse
import time

from ..data.config import Phase01Config, duckdb_connect
from ..data.utils import RunLog, get_logger

log = get_logger(__name__)

STEPS = ("marts", "serving", "monitoring")


def run(cfg: Phase01Config, steps: tuple[str, ...] = STEPS, force: bool = False) -> dict:
    import polars as pl

    from .marts import LabArtifactStore, LabMartBuilder
    from .monitoring import MonitoringExporter
    from .serving import ServingSnapshot

    store = LabArtifactStore(cfg)
    runlog = RunLog()
    out: dict = {"marts": None, "serving": None, "monitoring": None}
    t0 = time.perf_counter()
    status, error, peak = "OK", "", 0.0
    try:
        if "marts" in steps:
            with runlog.step("lab_marts") as m:
                con = duckdb_connect(cfg)
                try:
                    res = LabMartBuilder(cfg, con=con, store=store).build_all(force=force)
                finally:
                    con.close()
                out["marts"] = res
                m.rows_out = res.height
                m.add_artifact(cfg.paths.rel(store.data_dir))
                if res.height and (res["action"] == "built").any():
                    log.info("lab marts built: %s", ", ".join(res.filter(pl.col("action") == "built")["mart"]))
        if "serving" in steps:
            with runlog.step("lab_serving") as m:
                out["serving"] = ServingSnapshot(cfg, store).build()
                m.rows_out = out["serving"]["tables"]
                m.bytes_out = out["serving"]["bytes"]
                m.add_artifact(out["serving"]["path"])
    except Exception as exc:  # noqa: BLE001 — recorded in the monitoring snapshot, then re-raised
        status, error = "FAILED", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        runlog.save(cfg.paths.tables)
        peak = max((s.rss_peak_mb for s in runlog.steps), default=0.0)
        if "monitoring" in steps:
            res = out.get("marts")
            summary = {
                "seconds": round(time.perf_counter() - t0, 2), "status": status, "error": error, "rss_peak_mb": peak,
                "n_built": int((res["action"] == "built").sum()) if res is not None and res.height else None,
                "n_reused": int((res["action"] == "reused").sum()) if res is not None and res.height else None,
                "serving_bytes": (out.get("serving") or {}).get("bytes"),
            }
            with runlog.step("lab_monitoring") as m:
                out["monitoring"] = MonitoringExporter(cfg).export(summary)
                m.add_artifact(out["monitoring"]["path"])
            runlog.save(cfg.paths.tables)
    return out


def main(argv: list[str] | None = None) -> int:
    from ..data.config import configure_runtime

    ap = argparse.ArgumentParser(description="Analytics Lab · build dashboard marts and serving snapshots")
    ap.add_argument("--force", action="store_true", help="rebuild every mart even if current")
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {STEPS}")
    a = ap.parse_args(argv)
    steps = tuple(s.strip() for s in a.steps.split(",") if s.strip())
    unknown = set(steps) - set(STEPS)
    if unknown:
        ap.error(f"unknown steps {sorted(unknown)}")
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    out = run(cfg, steps, force=a.force)
    if out["marts"] is not None:
        print(out["marts"])
    for k in ("serving", "monitoring"):
        if out[k]:
            print(f"{k}: {out[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
