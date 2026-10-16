"""Tests of ``nflbowl.lab`` (comprehensive lab notebook support). Written without being executed in the authoring
session — run with ``pytest tests/nflbowl/test_nflbowl_lab.py -q``. Synthetic data only (conftest fixtures)."""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nflbowl.lab import biomech, geometry, registry, retrieval_lab, stats, synthesis, transfer_lab
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


# ---------------------------------------------------------------------- import shape (C1 / D3 first-run failures)
def test_no_module_style_import_of_a_reexported_function():
    """A subpackage that re-exports a function under its submodule's name (``nflbowl.evaluation.domain_diagnostics``)
    shadows the submodule, so ``from ..evaluation import domain_diagnostics as dd`` binds the function. No package
    module may import such a name module-style."""
    import nflbowl

    root = Path(nflbowl.__file__).parent
    shadowed: list[tuple[str, str]] = []
    for init in root.rglob("__init__.py"):
        pkg_dir = init.parent
        if pkg_dir == root:
            continue
        try:
            mod = importlib.import_module("nflbowl." + ".".join(pkg_dir.relative_to(root).parts))
        except ImportError:                                                      # optional-dependency subpackage: nothing to inspect here
            continue
        shadowed += [(pkg_dir.name, n) for n in vars(mod) if (pkg_dir / f"{n}.py").is_file() and not isinstance(getattr(mod, n), types.ModuleType)]
    assert ("evaluation", "domain_diagnostics") in shadowed                  # the public export shape is unchanged
    assert callable(importlib.import_module("nflbowl.evaluation").domain_diagnostics)
    bad = []
    for py in root.rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        for pkg, name in shadowed:
            if re.search(rf"from\s+[\w.]*\b{pkg}\s+import\s+{name}\s+as\s+\w+", src):
                bad.append(f"{py.relative_to(root)}: {pkg}.{name}")
    assert not bad, bad


# ---------------------------------------------------------------------- geometry (C1 / C4)
def _held_out(meta: pl.DataFrame, folds: pl.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from nflbowl.data.splits import fold_map, split_indices

    return split_indices(meta, fold_map(meta, folds, 5), fold, 0.1, 0)


def _emb(meta: pl.DataFrame, dim: int = 8, seed: int = 0, dtype: type = np.float32) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(meta.height, dim)).astype(dtype)


def test_basic_geometry_on_held_out_fold(arrays):
    _, _, meta, folds, _ = arrays
    _, _, ho = _held_out(meta, folds, 0)
    E = _emb(meta)
    E[ho[0]] = np.nan                                                            # non-finite rows are dropped, not propagated
    g, arr = geometry.basic_geometry(E, meta, np.sort(ho), seed=0)
    assert g["n_rows"] == ho.size - 1 and g["dims"] == 8
    assert {"effective_rank", "pc1_share", "mean_pairwise_cosine", "knn_same_domain_share", "cosine_mean", "norm_median"} <= set(g)
    assert all(np.isfinite(v) for k, v in g.items() if k in ("effective_rank", "cosine_mean", "norm_median"))
    assert arr["norms"].size == ho.size - 1 and arr["spectrum_share"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("n_rows", [0, 2])
def test_basic_geometry_zero_and_low_sample(arrays, n_rows):
    _, _, meta, _, _ = arrays
    g, arr = geometry.basic_geometry(_emb(meta), meta, np.arange(n_rows, dtype=np.int64), seed=0)
    assert g["n_rows"] == n_rows and g["dims"] == 8
    assert all(a.size == 0 for a in arr.values())


def test_domain_geometry_long_table_and_scalars(arrays):
    _, _, meta, _, _ = arrays
    rows = np.arange(meta.height, dtype=np.int64)
    E = _emb(meta, seed=3)
    E[(meta["domain"] == "nfl").to_numpy(), 0] += 4.0                            # planted domain shift
    long, d = geometry.domain_geometry("toy", E, meta, rows, fold=0, seed=0)
    assert long.columns == ["representation", "fold", "position_group", "domain", "metric", "value", "n"]
    assert set(long["representation"].to_list()) == {"toy"} and long["fold"].dtype == pl.Int64
    assert set(d) == {"domain_auc_grouped", "domain_auc_m11", "mmd2", "position_adjusted_domain_auc"}
    assert d["domain_auc_grouped"] > 0.9 and np.isfinite(d["mmd2"])


def test_domain_geometry_low_sample_is_nan_not_error(arrays):
    _, _, meta, _, _ = arrays
    long, d = geometry.domain_geometry("toy", _emb(meta), meta, np.arange(5, dtype=np.int64), fold=0, seed=0)
    assert long.height and np.isnan(d["domain_auc_grouped"]) and np.isnan(d["mmd2"])


def _fake_umap(monkeypatch) -> None:
    """A stand-in ``umap`` module whose UMAP returns float32 like the real one (no numba / umap-learn needed)."""
    class UMAP:
        def __init__(self, **kw):
            self.kw = kw

        def fit_transform(self, Z):
            return np.asarray(Z[:, :2], dtype=np.float32) * 0.5

    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=UMAP))


def _check_projection_schema(co: pl.DataFrame, meta: pl.DataFrame) -> None:
    assert co.columns == [*geometry.PROJECTION_META, "x", "y", "method", "representation", "fold", "fit_scope"]
    assert co["x"].dtype == pl.Float64 and co["y"].dtype == pl.Float64 and co["fold"].dtype == pl.Int64 and co["method"].dtype == pl.String
    for c in geometry.PROJECTION_META:
        assert co[c].dtype == meta[c].dtype, c                                  # metadata dtypes come from the bundle meta unchanged


def test_projection_frame_pca_and_umap_have_one_schema(arrays, monkeypatch):
    _, _, meta, _, _ = arrays
    rows = np.arange(meta.height, dtype=np.int64)
    E = _emb(meta, dtype=np.float32)
    pca = geometry.projection_frame(E, meta, rows, "toy", 0, seed=0, use_umap=False)
    assert set(pca["method"].to_list()) == {"pca"} and pca.height == meta.height
    _check_projection_schema(pca, meta)
    _fake_umap(monkeypatch)
    both = geometry.projection_frame(E, meta, rows, "toy", 0, seed=0, use_umap=True)   # float64 PCA + float32 UMAP: the C4 SchemaError
    assert both["method"].value_counts().sort("method")["count"].to_list() == [meta.height, meta.height]
    _check_projection_schema(both, meta)
    assert pl.concat([pca, both], how="vertical").height == 3 * meta.height     # frames from different calls stack strictly
    bal = geometry.projection_frame(E, meta, rows, "toy", 0, seed=0, max_points=40, use_umap=False)
    assert bal["domain"].value_counts().sort("domain")["count"].to_list() == [20, 20]   # domain-balanced sample


@pytest.mark.parametrize("n_rows", [0, 2])
def test_projection_frame_zero_and_low_sample(arrays, monkeypatch, n_rows):
    _, _, meta, _, _ = arrays
    _fake_umap(monkeypatch)
    co = geometry.projection_frame(_emb(meta), meta, np.arange(n_rows, dtype=np.int64), "toy", 0, seed=0, use_umap=True)
    assert co.height == 0
    _check_projection_schema(co, meta)


def test_projection_frame_small_sample_skips_umap_and_handles_1d(arrays, monkeypatch):
    _, _, meta, _, _ = arrays
    _fake_umap(monkeypatch)
    small = geometry.projection_frame(_emb(meta), meta, np.arange(20, dtype=np.int64), "toy", 0, seed=0, use_umap=True)
    assert set(small["method"].to_list()) == {"pca"} and small.height == 20      # ≤ UMAP_NEIGHBORS rows: PCA only
    one_d = geometry.projection_frame(_emb(meta, dim=1), meta, np.arange(meta.height, dtype=np.int64), "toy", 0, seed=0, use_umap=False)
    assert (one_d["y"] == 0.0).all()
    _check_projection_schema(one_d, meta)


def test_projection_coordinates_reads_held_out_rows_of_the_fold(arrays, monkeypatch):
    _, _, meta, folds, _ = arrays
    E = _emb(meta, dim=17, dtype=np.float64)
    split = {f: _held_out(meta, folds, f) for f in range(5)}
    exp = types.SimpleNamespace(bundle=types.SimpleNamespace(meta=meta), features=lambda: E, splits=lambda f: split[f])
    _fake_umap(monkeypatch)
    co = geometry.projection_coordinates(exp, "F0", 1, seed=0, use_umap=True)
    _check_projection_schema(co, meta)
    assert set(co["idx"].to_list()) <= set(meta["idx"].to_numpy()[split[1][2]].tolist())
    assert co["fit_scope"][0].startswith("fit on this seeded domain-balanced sample of fold-1")


# ---------------------------------------------------------------------- Experiment.compare_inductive (D3), no training, no GPU
@pytest.fixture()
def cpu_experiment(bundle_dir, tmp_path):
    from nflbowl import Experiment

    tiny = {"science.n_boot": 20, "runtime.device": "cpu", "runtime.progress": False}
    exp = Experiment(data_dir=bundle_dir, output_dir=tmp_path, overrides=tiny, name="lab_d3")
    exp.prepare()
    return exp


def test_compare_inductive_all_encoders_missing(cpu_experiment):
    from nflbowl.models import registry as reg

    exp = cpu_experiment
    F0, F2, F4 = reg.resolve("F0"), reg.resolve("F2"), reg.resolve("F4")
    res = exp.compare_inductive(models=["F0", "F2", "F4"], reference="F0", bootstrap=20, folds=[0, 1], reuse=False, save=False)
    assert res.meta["unavailable"] == {F2: [0, 1], F4: [0, 1]}
    auc = res.comparison.filter(pl.col("metric") == "auc")
    st = dict(zip(auc["representation"].to_list(), auc["status"].to_list()))
    assert st[F2] == "missing" and st[F4] == "missing"
    assert set(res.scores["representation"].unique().to_list()) == {F0}
    reps = set(res.domain["representation"].unique().to_list())
    assert reps == {F0, "training_split"}                                       # F0 diagnostics + positive-pair coverage rows
    assert res.domain.filter(pl.col("metric").str.starts_with("positive_pair_coverage_")).height > 0
    with pytest.raises(FileNotFoundError):
        exp.compare_inductive(models=["F0", "F4"], folds=[0], on_missing="raise", reuse=False, save=False)
    with pytest.raises(FileNotFoundError):
        exp.compare_inductive(models=["F4"], reference="F4", folds=[0], reuse=False, save=False)   # reference missing everywhere


def test_compare_inductive_partial_encoder_coverage(cpu_experiment, monkeypatch):
    """A synthetic F4 'encoder' available for fold 0 only (stands in for a trained checkpoint): its diagnostics run,
    fold 1 is reported unavailable, and the paired set keeps fold-0 athletes only."""
    from nflbowl.models import registry as reg

    exp = cpu_experiment
    F0, F4 = reg.resolve("F0"), reg.resolve("F4")
    real = exp._inductive_representation
    E4 = np.random.default_rng(7).normal(size=(exp.bundle.meta.height, 8)).astype(np.float32)

    def fake(name, fold, held_out):
        if name == F4:
            return (E4, f"{F4}:synthetic_f{fold}") if fold == 0 else None
        return real(name, fold, held_out)

    monkeypatch.setattr(exp, "_inductive_representation", fake)
    res = exp.compare_inductive(models=["F0", "F4"], reference="F0", bootstrap=20, folds=[0, 1], reuse=False, save=False)
    assert res.meta["unavailable"] == {F4: [1]}
    f4 = res.scores.filter(pl.col("representation") == F4)
    assert set(f4["fold"].unique().to_list()) == {0}
    row = res.comparison.filter((pl.col("metric") == "auc") & (pl.col("representation") == F4)).row(0, named=True)
    assert row["status"] == "ok" and row["n_folds"] == 1
    dom = res.domain.filter(pl.col("representation") == F4)
    assert set(dom["fold"].unique().to_list()) == {0} and "domain_auc" in set(dom["metric"].to_list())
    cov = res.coverage.filter((pl.col("representation") == F4) & (pl.col("fold") == 1)).row(0, named=True)
    assert cov["available"] is False
