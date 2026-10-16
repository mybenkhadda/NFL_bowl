"""Analytics Lab · marts, serving snapshot, monitoring snapshot, Superset bundle spec, kinematics.

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_analytics_lab.py -q
Uses the synthetic mini-warehouse from conftest.py (no real data, no Docker).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from nflbdb.data.bronze import BronzeIngestor
from nflbdb.data.gold import GoldBuilder
from nflbdb.data.silver import SilverBuilder
from nflbdb.data.validate import DataQualityValidator
from nflbdb.lab.build import run
from nflbdb.lab.marts import LAB_DATASET_DIR, LabArtifactStore, band, pct_rank
from nflbdb.lab.serving import SERVING_DB_NAME, serving_dir

LAB = Path(__file__).resolve().parents[1] / "infrastructure" / "analytics_lab"


def _phase01(cfg) -> None:
    BronzeIngestor(cfg).ingest()
    v = DataQualityValidator(cfg)
    v.export(v.run())
    SilverBuilder(cfg).build_all()
    GoldBuilder(cfg).build_all()


@pytest.fixture()
def lab(synth):
    _phase01(synth)
    out = run(synth)
    return synth, out


def _serving(cfg):
    import duckdb

    # Exactly the Superset connection settings: read-only, no access to any other file.
    return duckdb.connect(str(serving_dir(cfg) / SERVING_DB_NAME), read_only=True,
                          config={"enable_external_access": False})


# ----------------------------------------------------------------------------- marts
def test_marts_persisted_in_gold_phase_lab_with_sidecars(lab):
    cfg, out = lab
    store = LabArtifactStore(cfg)
    assert store.data_dir == cfg.paths.gold / LAB_DATASET_DIR
    names = set(store._manifest()["datasets"])
    assert {"athlete", "combine_attempt", "nfl_player_play", "kpi_snapshot", "availability"} <= names
    for n in names:
        assert (store.data_dir / f"{n}.schema.json").is_file(), n
    assert (cfg.paths.tables / "analytics_lab" / "lab_mart_catalog.csv").is_file()


def test_kpis_reflect_warehouse_contents(lab):
    cfg, _ = lab
    con = _serving(cfg)
    try:
        k = con.execute("SELECT n_athletes, n_player_plays, n_combine_attempts, n_games FROM lab.kpi_snapshot").fetchone()
        n_fact = con.execute("SELECT count(*) FROM lab.nfl_player_play").fetchone()[0]
        n_attempts = con.execute("SELECT count(*) FROM lab.combine_attempt").fetchone()[0]
    finally:
        con.close()
    assert k[0] == 4                       # synthetic players 101–104
    assert k[1] == n_fact == 5             # 3 + 2 tracked athletes on the two plays
    assert k[2] == n_attempts == 4         # 2 athletes × 2 attempts
    assert k[3] == 2


def test_second_run_reuses_unchanged_marts(lab):
    cfg, _ = lab
    out = run(cfg, steps=("marts",))
    actions = dict(zip(out["marts"]["mart"], out["marts"]["action"]))
    assert actions["athlete"] == "reused" and actions["nfl_player_play"] == "reused"
    assert actions["kpi_snapshot"] == "built"          # snapshot tables are always rebuilt


def test_missing_optional_inputs_are_reported_not_invented(lab):
    cfg, _ = lab
    con = _serving(cfg)
    try:
        avail = dict(con.execute("SELECT dataset, available FROM lab.availability").fetchall())
        n_research = con.execute("SELECT count(*) FROM lab.research_hypothesis").fetchone()[0]
        n_affected = con.execute("SELECT count(*) FROM lab.quality_affected").fetchone()[0]
    finally:
        con.close()
    assert avail["reports/tables/research_feasibility_matrix"] is False
    assert n_research == 0                 # no feasibility matrix in the synthetic project → empty, not fabricated
    assert n_affected == 0                 # no Phase 02 flags → no affected counts


def test_percentile_helpers_sql_shape():
    assert "percent_rank()" in pct_rank("forty", "position_group", higher_is_better=False)
    assert "DESC" in pct_rank("forty", "position_group", higher_is_better=False)
    assert "Top 10%" in band("p")


# ----------------------------------------------------------------------------- monitoring
def test_monitoring_snapshot_has_real_tables_and_appends_builds(lab):
    cfg, _ = lab
    path = serving_dir(cfg) / "lab_monitoring.sqlite"
    run(cfg, steps=("monitoring",))
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        n_builds = conn.execute("SELECT count(*) FROM lab_build").fetchone()[0]
        layers = {r[0] for r in conn.execute("SELECT layer FROM storage_layer")}
    finally:
        conn.close()
    assert {"pipeline_step", "storage_layer", "dataset_file", "gold_build", "benchmark", "lab_build"} <= tables
    assert n_builds == 2                   # one row per build (append-only)
    assert {"bronze", "silver", "gold", "serving"} <= layers


# ----------------------------------------------------------------------------- Superset bundle
def _bundle():
    sys.path.insert(0, str(LAB / "superset" / "bootstrap"))
    import build_bundle  # noqa: PLC0415

    return build_bundle


def test_bundle_spec_is_consistent_and_deterministic():
    bb = _bundle()
    assert bb.check() == []
    files = bb.bundle_files()
    for path, text in files.items():
        json.loads(text)                   # JSON == valid YAML for Superset's PyYAML loader
    assert bb.content_hash(files) == bb.content_hash(bb.bundle_files())
    dashboards = [json.loads(t) for p, t in files.items() if "/dashboards/" in p]
    assert {d["slug"] for d in dashboards} >= {"nfl-lab-overview", "nfl-lab-combine", "nfl-lab-nfl",
                                               "nfl-lab-quality", "nfl-lab-research"}
    db = next(json.loads(t) for p, t in files.items() if "/databases/" in p)
    assert db["extra"]["engine_params"]["connect_args"]["read_only"] is True


def test_every_declared_dataset_column_exists_in_the_snapshot(lab):
    cfg, _ = lab
    spec = _bundle().spec
    con = _serving(cfg)
    try:
        missing = []
        for ds in spec.DATASETS:
            cols = {r[0] for r in con.execute(f"DESCRIBE lab.{ds.table}").fetchall()}
            missing += [f"{ds.table}.{c.partition(':')[0]}" for c in ds.columns if c.partition(":")[0] not in cols]
    finally:
        con.close()
    assert missing == []


# ----------------------------------------------------------------------------- Streamlit kinematics
def test_alignment_methods():
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(LAB / "streamlit"))
    from labkit.kinematics import align, summarize  # noqa: PLC0415

    df = pd.DataFrame({"t_s": [0.0, 0.1, 0.2, 0.3, 0.4], "s": [0.0, 0.5, 2.0, 4.0, 3.0],
                       "a": [1.0, 2.0, 3.0, 2.0, 1.0], "dis": [0.0, 0.05, 0.2, 0.4, 0.3],
                       "x": [0, 0.05, 0.25, 0.65, 0.95], "y": [0.0] * 5})
    s = summarize(df)
    assert s["max_s"] == 4.0 and abs(s["time_to_peak_s"] - 0.3) < 1e-9 and abs(s["distance_yds"] - 0.95) < 1e-9
    t, _ = align(df, "peak")
    assert abs(t[3]) < 1e-9
    t, _ = align(df, "onset", onset_threshold=1.0)
    assert abs(t[2]) < 1e-9
    t, note = align(df, "snap", snap_t=None)
    assert t is None and "ball_snap" in note
    t, _ = align(df, "normalized")
    assert t[0] == 0 and abs(t[-1] - 1) < 1e-9
