"""Tests of ``nflbowl.lab`` (comprehensive lab notebook support). Written without being executed in the authoring
session — run with ``pytest tests/nflbowl/test_nflbowl_lab.py -q``. Synthetic data only (conftest fixtures)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflbowl.lab import biomech, registry, retrieval_lab, stats, synthesis, transfer_lab
from nflbowl.lab import config as labconfig
from nflbowl.lab.store import LabRun, StageSkipped


# ---------------------------------------------------------------------- config
def test_config_defaults_and_unknown_keys():
    c = labconfig.resolve({"run_mode": "pilot"})
    assert c.precision == "fp32" and c.folds([0, 1, 2, 3, 4]) == [0] and c.encoders() == ["F4"]
    assert labconfig.resolve({"run_mode": "standard"}).folds([0, 1, 2]) == [0, 1, 2]
    with pytest.raises(KeyError):
        labconfig.resolve({"not_a_key": 1})
    with pytest.raises(ValueError):
        labconfig.resolve({"run_mode": "pilot", "run_ablations": True})
    with pytest.raises(TypeError):
        labconfig.resolve({"resume": "yes"})


def test_science_seed_is_not_the_lab_seed():
    c = labconfig.resolve({"seed": 42})
    assert "science.seed" not in c.experiment_overrides()
    assert labconfig.resolve({"science_seed": 7}).experiment_overrides()["science.seed"] == 7


# ---------------------------------------------------------------------- store / stages / registry
def test_labrun_stage_statuses_and_tables(tmp_path):
    run = LabRun(tmp_path / "lab", labconfig.resolve({}))
    assert run.run_stage("B1", lambda ctx: 1) == 1 and run.stage_status("B1") == "executed"

    def skip(ctx):
        raise StageSkipped("blocked: no warehouse")

    assert run.run_stage("X1", skip) is None and run.stage_status("X1") == "blocked"

    def boom(ctx):
        raise ValueError("bad")

    assert run.run_stage("C1", boom) is None and run.stage_status("C1") == "failed"
    assert (tmp_path / "lab" / "logs" / "c1_traceback.txt").is_file()
    run.run_stage("D2", lambda ctx: 1, enabled=False, reason_disabled="off")
    assert run.stage_status("D2") == "skipped"
    df = pl.DataFrame({"a": [1, 2], "b": [[1], [2, 3]]})
    run.table(df, "t", "retrieval")
    assert run.has("t", "retrieval") and run.frame("t", "retrieval").height == 2
    assert (tmp_path / "lab" / "tables" / "retrieval" / "t.csv").is_file()
    st = registry.status_table(run)
    got = dict(zip(st["id"].to_list(), st["status"].to_list()))
    assert got["B1"] == "completed" and got["C1"] == "failed" and got["D2"] == "skipped" and got["E2"] == "pending"


def test_cache_provenance_mismatch_forces_recompute(tmp_path):
    run = LabRun(tmp_path / "lab", labconfig.resolve({}))
    run.data_file(pl.DataFrame({"x": [1]}), "datasets/cached/a", {"bundle": "abc", "version": 1})
    assert run.load_data("datasets/cached/a", {"bundle": "abc", "version": 1}).height == 1
    assert run.load_data("datasets/cached/a", {"bundle": "zzz", "version": 1}) is None


# ---------------------------------------------------------------------- biomechanics
def _seq(speeds_yds: np.ndarray, an: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    L = 64
    x = np.zeros((1, L, 6), np.float32)
    m = np.zeros((1, L), bool)
    n = speeds_yds.size
    x[0, :n, 0] = speeds_yds / 10.0
    acc = np.r_[0.0, np.diff(speeds_yds) / 0.1]
    x[0, :n, 1] = acc / 5.0
    x[0, :n, 2] = (an if an is not None else np.zeros(n)) / 5.0
    x[0, :n, 4] = 1.0
    x[0, :n, 5] = 1.0
    m[0, :n] = True
    return x, m


def test_sequence_metrics_physical_units():
    v = np.r_[np.linspace(0.0, 9.0, 20), np.full(10, 9.0), np.linspace(9.0, 3.0, 10)]
    x, m = _seq(v)
    r = biomech.sequence_metrics(x, m)
    assert r["peak_speed"][0] == pytest.approx(9.0, rel=1e-5)
    assert r["peak_decel"][0] == pytest.approx(6.0 / 0.9 * 1.0, rel=0.05)          # 6 yd/s lost over 9 steps of 0.1 s
    assert r["duration_s"][0] == pytest.approx(4.0, rel=1e-5)
    assert np.isfinite(r["accel_rate_0_90"][0]) and r["time_to_90"][0] > 0
    assert r["burst_count"][0] >= 1


def test_metrics_undefined_cases_are_nan():
    x, m = _seq(np.full(30, 5.0))                                                 # starts above standstill
    r = biomech.sequence_metrics(x, m)
    assert np.isnan(r["accel_rate_0_90"][0])
    x2, m2 = _seq(np.array([1.0, 1.0]))
    assert np.isnan(biomech.sequence_metrics(x2, m2)["peak_speed"][0])


def test_metric_definitions_cover_analysis_metrics():
    names = set(biomech.metric_definitions()["name"].to_list())
    assert set(biomech.ANALYSIS_METRICS) <= names


# ---------------------------------------------------------------------- stats
def test_chance_baselines():
    n = np.array([4, 4, 4])
    assert stats.chance_recall_at_k(1, n) == pytest.approx(0.2)
    assert stats.chance_recall_at_k(10, n) == pytest.approx(1.0)
    assert stats.chance_mrr(np.array([1])) == pytest.approx((1 + 0.5) / 2)
    assert stats.chance_median_rank(np.array([4])) == pytest.approx(3.0)


def test_cluster_bootstrap_resamples_athletes():
    v = np.r_[np.zeros(50), np.ones(50)]
    c = np.repeat(np.arange(20), 5)
    r = stats.cluster_bootstrap(v, c, np.mean, 200, 0)
    assert r["n_clusters"] == 20 and r["lo"] <= 0.5 <= r["hi"]


# ---------------------------------------------------------------------- retrieval helpers
def _scores() -> pl.DataFrame:
    rows = []
    for rep, shift in (("F0_engineered", 0), ("F4_physics_contrastive", -1)):
        for i in range(30):
            rank = max(1, (i % 5) + 1 + shift)
            rows.append({"representation": rep, "fold": i % 3, "nfl_id": i, "position_group": "WR", "status": "scored", "n_candidates": 9,
                         "own_similarity": 0.1, "auc": 1 - (rank - 1) / 9, "rank": rank, "top1": rank == 1, "top5": rank <= 5,
                         "pool_hash": f"p{i}", "candidate_ids": [0]})
    return pl.DataFrame(rows)


def test_rank_summary_reports_chance_of_same_pools():
    rs = retrieval_lab.rank_summary(_scores())
    assert set(rs["representation"].to_list()) == {"F0_engineered", "F4_physics_contrastive"}
    assert rs["top1_chance"][0] == pytest.approx(0.1)
    rc = retrieval_lab.recall_curve(_scores())
    assert rc.filter(pl.col("k") == 20)["chance"][0] == pytest.approx(1.0)


def test_paired_pairs_direction():
    pp = retrieval_lab.paired_pairs(_scores(), [("F4_physics_contrastive", "F0_engineered")], metrics=("auc",), n_boot=200)
    assert pp["delta"][0] > 0


def test_pool_definitions_do_not_depend_on_representation(arrays):
    X, M, meta, folds, athletes = arrays
    pools = retrieval_lab.pool_definitions(athletes, seed=0)
    same, allc = [1001, 1004, 1007, 1010, 1013, 1016], [1001, 1002, 1004, 1007, 1010, 1013, 1016]
    for name, fn in pools.items():
        assert fn(0, 1000, same, allc) == fn(0, 1000, same, allc)
    assert "stopwatch_twins_3" in pools and len(pools["stopwatch_twins_3"](0, 1000, same, allc)) == 3


def test_pool_scores_identical_hashes_across_representations(arrays):
    X, M, meta, folds, athletes = arrays
    rng = np.random.default_rng(0)
    rows = np.arange(meta.height)
    pools = retrieval_lab.pool_definitions(athletes, seed=0)
    a = retrieval_lab.pool_scores("A", rng.normal(size=(meta.height, 4)), meta, rows, 0, pools)
    b = retrieval_lab.pool_scores("B", rng.normal(size=(meta.height, 4)), meta, rows, 0, pools)
    j = a.join(b, on=["pool", "nfl_id"], suffix="_b")
    assert (j["pool_hash"].fill_null("") == j["pool_hash_b"].fill_null("")).all()


# ---------------------------------------------------------------------- transfer helpers
def _oof(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    y = rng.normal(3, 0.4, n)
    return pl.DataFrame({"nfl_id": np.arange(n), "position_group": ["WR", "CB", "DL"] * (n // 3), "fold": np.arange(n) % 5, "y": y,
                         "pred_M1": y + rng.normal(0, 0.4, n), "pred_M1+F0": y + rng.normal(0, 0.35, n),
                         "pred_M1+F4_physics_contrastive": y + rng.normal(0, 0.2, n), "has_F4_physics_contrastive": [True] * (n - 5) + [False] * 5,
                         "forty": rng.normal(4.6, 0.1, n)})


def test_prediction_metrics_and_comparisons():
    oof = _oof()
    pm = transfer_lab.prediction_metrics(oof)
    assert set(pm["model"].to_list()) == {"M1", "M1+F0", "M1+F4_physics_contrastive"}
    tc = transfer_lab.paired_comparisons(oof, 200, 0)
    assert {"common", "native_coverage"} <= set(tc["cohort"].to_list())
    nat = tc.filter((pl.col("cohort") == "native_coverage") & (pl.col("model") == "M1+F4_physics_contrastive") & (pl.col("reference") == "M1"))
    assert nat["n_athletes"][0] == 55
    best = tc.filter((pl.col("cohort") == "common") & (pl.col("model") == "M1+F4_physics_contrastive") & (pl.col("reference") == "M1"))
    assert best["delta_rmse"][0] > 0


def test_scouting_metrics_shapes():
    sm = transfer_lab.scouting_metrics(_oof(), ["M1", "M1+F4_physics_contrastive"], "M1", 100, 0)
    assert sm["ranking"].height == 2 and sm["calibration"]["bin"].max() == 10


# ---------------------------------------------------------------------- gates
def test_gates_never_pass_on_missing_evidence():
    g = synthesis.evaluate_gates({})
    st = dict(zip(g["gate"].to_list(), g["status"].to_list()))
    assert all(v == synthesis.NOT_EVALUATED for v in st.values())


def test_gate3_pilot_is_inconclusive():
    comp = pl.DataFrame({"metric": ["auc"], "representation": ["F4_physics_contrastive"], "status": ["ok"], "mean": [0.6], "mean_lo": [0.55], "mean_hi": [0.65],
                         "delta": [0.08], "delta_lo": [0.03], "delta_hi": [0.12], "n_athletes": [80]})
    g = synthesis.evaluate_gates({"paired_inductive_comparison": comp, "folds_used": [0]})
    assert g.filter(pl.col("gate") == "G3 inductive retrieval")["status"][0] == synthesis.INCONCLUSIVE
    g5 = synthesis.evaluate_gates({"paired_inductive_comparison": comp, "folds_used": [0, 1, 2, 3, 4]})
    assert g5.filter(pl.col("gate") == "G3 inductive retrieval")["status"][0] == synthesis.PASS


def test_describe_interval_never_claims_no_effect():
    s = synthesis.describe_interval("x", 0.01, -0.02, 0.04)
    assert "inconclusive" in s and "no effect" not in s.replace("not evidence of no effect", "")
