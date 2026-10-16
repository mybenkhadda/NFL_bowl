"""READ-ONLY checks of the analytics lab (safe to run any time; writes nothing).

    python infrastructure/analytics_lab/scripts/validate_connections.py              # files only
    python infrastructure/analytics_lab/scripts/validate_connections.py --http       # + service health URLs

Checks
  1. warehouse/serving/nfl_lab.duckdb opens READ-ONLY with external access disabled (the exact
     Superset connection settings), every table the dashboards need exists and is non-empty.
  2. warehouse/serving/lab_monitoring.sqlite opens read-only and has the Grafana tables.
  3. Silver partitions the trajectory explorer reads exist.
  4. (--http) Superset /health, Streamlit /_stcore/health, Grafana /api/health on localhost.
Exit code 0 = all required checks passed.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVING = ROOT / "warehouse" / "serving"

REQUIRED_TABLES = ["athlete", "combine_attempt", "combine_athlete_drill", "drill_participation", "test_availability",
                   "coverage_matrix", "linkage_funnel", "season_volume", "tracking_partition", "nfl_game", "nfl_play",
                   "nfl_player_play", "nfl_athlete_season", "nfl_speed_distribution", "kpi_snapshot", "availability",
                   "mart_catalog"]
OPTIONAL_TABLES = ["quality_check", "quality_run_history", "tracking_interval", "tracking_consistency",
                   "quality_affected", "research_hypothesis", "research_by_position", "research_drill"]
MONITORING_TABLES = ["pipeline_step", "storage_layer", "dataset_file", "gold_build", "benchmark", "lab_build"]


def report(ok: bool, msg: str, required: bool = True) -> bool:
    print(f"[{'OK  ' if ok else ('FAIL' if required else 'WARN')}] {msg}")
    return ok or not required


def check_duckdb() -> bool:
    path = SERVING / "nfl_lab.duckdb"
    if not path.is_file():
        return report(False, f"{path} missing — run prepare_serving_data.py")
    import duckdb

    con = duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})
    try:
        tables = {r[0]: r[1] for r in con.execute(
            "SELECT table_name, estimated_size FROM duckdb_tables() WHERE schema_name = 'lab'").fetchall()}
        ok = True
        for t in REQUIRED_TABLES + OPTIONAL_TABLES:
            required = t in REQUIRED_TABLES
            if t not in tables:
                ok &= report(False, f"lab.{t} missing", required)
                continue
            n = con.execute(f"SELECT count(*) FROM lab.{t}").fetchone()[0]
            ok &= report(n > 0, f"lab.{t}: {n:,} rows" + ("" if n else " (empty — see lab.availability)"), required)
        kpi = con.execute("SELECT n_athletes, n_athletes_matched, n_player_plays, lab_built_at FROM lab.kpi_snapshot").fetchone()
        print(f"       kpi: athletes={kpi[0]}, matched={kpi[1]}, player-plays={kpi[2]}, built={kpi[3]}")
        return ok
    finally:
        con.close()


def check_sqlite() -> bool:
    path = SERVING / "lab_monitoring.sqlite"
    if not path.is_file():
        return report(False, f"{path} missing (needed only for Grafana)", required=False)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        ok = True
        for t in MONITORING_TABLES:
            n = conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] if t in have else None
            ok &= report(t in have, f"monitoring.{t}: {n if n is not None else 'missing'} rows", required=False)
        return ok
    finally:
        conn.close()


def check_silver() -> bool:
    ok = True
    for table, key in (("game_tracking", "season"), ("combine_tracking", "draft_year")):
        parts = sorted((ROOT / "warehouse" / "silver" / table).glob(f"{key}=*"))
        ok &= report(bool(parts), f"silver/{table}: {len(parts)} partitions ({', '.join(p.name for p in parts)})")
    return ok


def check_http() -> bool:
    ok = True
    for name, url, required in (("Superset", "http://localhost:8088/health", True),
                                ("Streamlit", "http://localhost:8501/_stcore/health", True),
                                ("Grafana", "http://localhost:3000/api/health", False)):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                ok &= report(r.status == 200, f"{name} {url} → HTTP {r.status}", required)
        except Exception as exc:  # noqa: BLE001
            ok &= report(False, f"{name} {url} → {type(exc).__name__}: {exc}", required)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", action="store_true", help="also probe the running services on localhost")
    a = ap.parse_args()
    results = [check_duckdb(), check_sqlite(), check_silver()]
    if a.http:
        results.append(check_http())
    print("ALL REQUIRED CHECKS PASSED" if all(results) else "SOME REQUIRED CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
