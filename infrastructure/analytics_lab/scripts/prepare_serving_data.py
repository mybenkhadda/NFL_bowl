"""Build the lab marts + serving snapshots on the HOST (conda env `nfl`).

Thin wrapper around `python -m nflbdb.lab.build` that works without setting PYTHONPATH:

    conda activate nfl
    set NFL_BDB_DUCKDB_MEMORY=3GB
    python infrastructure/analytics_lab/scripts/prepare_serving_data.py            # incremental
    python infrastructure/analytics_lab/scripts/prepare_serving_data.py --force    # rebuild all marts

Writes (never modifies existing Phase 01/02 outputs):
    warehouse/gold/phase_lab/*.parquet (+ _manifest.json, *.schema.json, _archive/)
    warehouse/serving/nfl_lab.duckdb, warehouse/serving/lab_monitoring.sqlite
    reports/tables/analytics_lab/lab_mart_*.csv, rows appended to reports/tables/pipeline_run_log
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from nflbdb.lab.build import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
