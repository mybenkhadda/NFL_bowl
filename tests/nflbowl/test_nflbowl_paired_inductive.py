"""Paired inductive retrieval comparison and domain diagnostics.

Unit tests run on the synthetic bundle metadata (numpy / polars only); the end-to-end test trains tiny inductive
encoders and needs PyTorch."""

from __future__ import annotations

import importlib

import numpy as np
import polars as pl
import pytest

from nflbowl.data.splits import HELD_OUT, SplitLeakError, fold_map, split_indices, split_roles
from nflbowl.evaluation import paired_inductive as pi

# the submodule: ``from nflbowl.evaluation import domain_diagnostics`` binds the re-exported FUNCTION of that name
dd = importlib.import_module("nflbowl.evaluation.domain_diagnostics")


def _oracle(meta: pl.DataFrame, noise: float = 0.05, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Athlete-specific prototype + small noise: own Combine and NFL means are nearly identical."""
    rng = np.random.default_rng(seed)
    ids = meta["nfl_id"].to_numpy()
    proto = {int(i): rng.normal(size=dim) for i in np.unique(ids)}
    return np.stack([proto[int(i)] + rng.normal(0, noise, dim) for i in ids]).astype(np.float32)


def _random(meta: pl.DataFrame, dim: int = 8, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(meta.height, dim)).astype(np.float32)


def _all_rows(meta: pl.DataFrame) -> np.ndarray:
    return np.arange(meta.height, dtype=np.int64)


def _scores_frame(models_auc: dict[str, np.ndarray], folds: np.ndarray | None = None, positions: list[str] | None = None) -> pl.DataFrame:
    """Hand-built athlete score table (same athletes / pools for every model unless the caller edits it)."""
    n = len(next(iter(models_auc.values())))
    folds = np.zeros(n, int) if folds is None else folds
    positions = positions or ["WR"] * n
    rows = []
    for m, a in models_auc.items():
        for k in range(n):
            rows.append({"representation": m, "fold": int(folds[k]), "nfl_id": 1000 + k, "position_group": positions[k], "status": pi.SCORED,
                         "auc": float(a[k]), "top5": bool(a[k] > 0.5), "pool_hash": f"pool{k}", "n_candidates": 10})
    return pl.DataFrame(rows)


def _same(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    """Frame equality with NaN treated as missing (NaN never equals itself)."""
    nan_free = lambda d: d.with_columns(pl.col(pl.Float64).fill_nan(None))        # noqa: E731
    return nan_free(a).equals(nan_free(b))


# ====================================================================== athlete / fold isolation, no leakage
def test_fold_isolation_and_scores_stay_inside_fold(arrays):
    _, _, meta, folds, _ = arrays
    fm = fold_map(meta, folds, 5)
    E = _oracle(meta)
    for f in range(5):
        tr, va, ho = split_indices(meta, fm, f, 0.1, 0)
        info = pi.assert_fold_isolation(meta, fm, ho, f, tr, va)
        assert info["held_out_athletes"] > 0
        s = pi.athlete_retrieval_scores("oracle", E, meta, ho, f)
        fold_athletes = {i for i, k in fm.items() if k == f}
        assert set(s["nfl_id"].to_list()) == fold_athletes                       # every held-out athlete reported, no other
        for cands in s["candidate_ids"].to_list():
            assert set(cands) <= fold_athletes                                     # candidates never leave the fold
        train_athletes = set(meta["nfl_id"].to_numpy()[np.r_[tr, va]].tolist())
        assert not (fold_athletes & train_athletes)                              # held-out athletes never in training / validation


def test_fold_isolation_detects_leaks(arrays):
    _, _, meta, folds, _ = arrays
    fm = fold_map(meta, folds, 5)
    tr, va, ho = split_indices(meta, fm, 0, 0.1, 0)
    with pytest.raises(SplitLeakError):
        pi.assert_fold_isolation(meta, fm, np.r_[ho, tr[:1]], 0)               # a training-fold row among the held-out rows
    with pytest.raises(SplitLeakError):
        pi.assert_fold_isolation(meta, fm, ho, 0, train=np.r_[tr, ho[:1]])     # a held-out athlete among the training rows


def test_held_out_roles_reject_transductive_or_other_fold_tables(arrays):
    _, _, meta, folds, _ = arrays
    fm = fold_map(meta, folds, 5)
    tr, va, ho = split_indices(meta, fm, 1, 0.1, 0)
    roles = split_roles(meta.height, tr, va, ho)
    pi.verify_held_out_roles(roles, ho, 1)
    tr0, va0, _ = split_indices(meta, fm, None, 0.1, 0)                         # transductive table: nothing held out
    with pytest.raises(SplitLeakError):
        pi.verify_held_out_roles(split_roles(meta.height, tr0, va0, np.empty(0, np.int64)), ho, 1)
    tr2, va2, ho2 = split_indices(meta, fm, 2, 0.1, 0)                          # other fold's table
    with pytest.raises(SplitLeakError):
        pi.verify_held_out_roles(split_roles(meta.height, tr2, va2, ho2), ho, 1)
    assert HELD_OUT == "held_out"


# ====================================================================== identical candidate pools
def test_candidate_pools_identical_across_representations(arrays):
    _, _, meta, _, _ = arrays
    ho = _all_rows(meta)
    a = pi.athlete_retrieval_scores("oracle", _oracle(meta), meta, ho, 0)
    b = pi.athlete_retrieval_scores("random", _random(meta), meta, ho, 0)
    j = a.select("nfl_id", "pool_hash", "candidate_ids", "position_group").join(b.select("nfl_id", "pool_hash", "candidate_ids"), on="nfl_id", suffix="_b")
    assert j.height == a.height == b.height
    assert (j["pool_hash"] == j["pool_hash_b"]).all()
    assert j["candidate_ids"].to_list() == j["candidate_ids_b"].to_list()
    pos = dict(zip(meta["nfl_id"].to_list(), meta["position_group"].to_list()))
    for i, cands, p in zip(j["nfl_id"].to_list(), j["candidate_ids"].to_list(), j["position_group"].to_list()):
        assert i not in cands and all(pos[c] == p for c in cands)               # same position, never itself
    pi.check_identical_pools(pl.concat([a, b]))


def test_mismatched_pools_are_refused():
    s = _scores_frame({"ref": np.linspace(0.4, 0.9, 20), "m": np.linspace(0.5, 1.0, 20)})
    bad = s.with_columns(pl.when((pl.col("representation") == "m") & (pl.col("nfl_id") == 1003)).then(pl.lit("other")).otherwise(pl.col("pool_hash")).alias("pool_hash"))
    with pytest.raises(pi.CandidatePoolMismatch):
        pi.paired_bootstrap(bad, ["ref", "m"], "ref", n_boot=50)


def test_athlete_scores_reproduce_cross_domain_retrieval(arrays):
    from nflbowl.evaluation.retrieval import cross_domain_retrieval

    _, _, meta, _, _ = arrays
    E = _oracle(meta, noise=0.6)
    s = pi.athlete_retrieval_scores("x", E, meta, _all_rows(meta), 0)
    ok = s.filter(pl.col("status") == pi.SCORED)
    ref = cross_domain_retrieval(E, meta, 50, 0)
    assert ok.height == ref["n_athletes"]
    assert ok["auc"].mean() == pytest.approx(ref["auc"], abs=1e-12)
    assert ok["top5"].cast(pl.Float64).mean() == pytest.approx(ref["top5"], abs=1e-12)


# ====================================================================== paired resampling
def test_paired_bootstrap_uses_identical_draws():
    rng = np.random.default_rng(3)
    u = rng.uniform(0.2, 0.9, 60)
    s = _scores_frame({"ref": u, "shifted": u + 0.01, "same": u.copy()})
    r = pi.paired_bootstrap(s, ["ref", "shifted", "same"], "ref", n_boot=500, seed=1)
    sh = r.filter(pl.col("representation") == "shifted").row(0, named=True)
    assert sh["delta"] == pytest.approx(0.01) and sh["delta_lo"] == pytest.approx(0.01) and sh["delta_hi"] == pytest.approx(0.01)
    same = r.filter(pl.col("representation") == "same").row(0, named=True)
    assert same["delta"] == 0 and same["delta_lo"] == 0 and same["delta_hi"] == 0
    assert sh["mean_hi"] > sh["mean_lo"]                                          # marginal means do vary across replicates
    r2 = pi.paired_bootstrap(s, ["ref", "shifted", "same"], "ref", n_boot=500, seed=1)
    assert _same(r, r2)                                                           # deterministic for a seed


def test_bootstrap_is_stratified_by_fold():
    strata = np.array([0] * 3 + [1] * 5 + [2] * 4)
    draws = pi.bootstrap_indices(strata, 200, np.random.default_rng(0))
    assert draws.shape == (200, strata.size)
    for row in draws:
        assert np.bincount(strata[row], minlength=3).tolist() == [3, 5, 4]


# ====================================================================== metric direction
def test_metric_direction_higher_is_better(arrays):
    _, _, meta, _, _ = arrays
    ho = _all_rows(meta)
    s = pl.concat([pi.athlete_retrieval_scores("oracle", _oracle(meta), meta, ho, 0), pi.athlete_retrieval_scores("random", _random(meta), meta, ho, 0)])
    r = pi.paired_bootstrap(s, ["random", "oracle"], "random", n_boot=300, seed=0)
    o = r.filter(pl.col("representation") == "oracle").row(0, named=True)
    assert o["direction"] == pi.HIGHER_IS_BETTER
    assert o["mean"] > 0.9 and o["delta"] > 0 and o["delta_lo"] > 0 and o["ci_excludes_zero"]
    rev = pi.paired_bootstrap(s, ["oracle", "random"], "oracle", n_boot=300, seed=0).filter(pl.col("representation") == "random").row(0, named=True)
    assert rev["delta"] == pytest.approx(-o["delta"]) and rev["delta_hi"] < 0
    perfect = s.filter((pl.col("representation") == "oracle") & (pl.col("status") == pi.SCORED))
    assert (perfect["rank"] >= 1).all() and (perfect.filter(pl.col("auc") == 1.0)["rank"] == 1).all()


# ====================================================================== missing representations
def test_missing_representation_and_incomplete_athletes():
    u = np.linspace(0.3, 0.9, 40)
    folds = np.r_[np.zeros(20, int), np.ones(20, int)]
    s = _scores_frame({"ref": u, "m": u + 0.05}, folds=folds)
    s = s.filter(~((pl.col("representation") == "m") & (pl.col("fold") == 1)))     # m's fold-1 encoder never trained
    r = pi.paired_bootstrap(s, ["ref", "m", "absent"], "ref", n_boot=100)
    m = r.filter(pl.col("representation") == "m").row(0, named=True)
    assert m["n_athletes"] == 20 and m["n_excluded_incomplete"] == 20 and m["n_folds"] == 1
    assert r.filter(pl.col("representation") == "absent")["status"].to_list() == ["missing"]
    with pytest.raises(ValueError):
        pi.paired_bootstrap(s.filter(pl.col("representation") != "ref"), ["ref", "m"], "ref", n_boot=10)
    cov = pi.coverage_table(s, ["ref", "m"], [0, 1], {"m": [1]})
    row = cov.filter((pl.col("representation") == "m") & (pl.col("fold") == 1)).row(0, named=True)
    assert row["available"] is False and row["n_scored"] == 0


# ====================================================================== small subgroups
def test_small_subgroups_are_flagged():
    pos = ["DL"] * 5 + ["CB"] * 15 + ["WR"] * 40
    u = np.random.default_rng(0).uniform(0.3, 0.9, len(pos))
    s = _scores_frame({"ref": u, "m": np.clip(u + 0.02, 0, 1)}, positions=pos)
    r = pi.paired_bootstrap(s, ["ref", "m"], "ref", n_boot=100, by="position_group", min_n=10, adequate_n=30).filter(pl.col("representation") == "m")
    rel = dict(zip(r["position_group"].to_list(), r["reliability"].to_list()))
    assert rel == {"CB": "exploratory", "DL": "insufficient", "WR": "adequate"}
    dl = r.filter(pl.col("position_group") == "DL").row(0, named=True)
    assert np.isnan(dl["delta_lo"]) and np.isnan(dl["delta_hi"]) and dl["n_athletes"] == 5
    assert not np.isnan(r.filter(pl.col("position_group") == "CB")["delta_lo"][0])


def test_tiny_position_pools_are_excluded_not_scored(arrays):
    _, _, meta, folds, _ = arrays
    fm = fold_map(meta, folds, 5)
    _, _, ho = split_indices(meta, fm, 0, 0.1, 0)
    s = pi.athlete_retrieval_scores("x", _oracle(meta), meta, ho, 0)
    small = s.filter(pl.col("status") == "position_pool_too_small")
    assert small["auc"].is_nan().all() and (small["n_candidates"] < 2).all()
    assert (s.filter(pl.col("status") == pi.SCORED)["n_candidates"] >= 2).all()


# ====================================================================== domain diagnostics
def test_grouped_domain_auc_detects_shift(arrays):
    _, _, meta, _, _ = arrays
    rows = _all_rows(meta)
    E = _random(meta, seed=5)
    inv = dd.grouped_domain_auc(E, meta, rows, seed=0)["domain_auc"]
    assert 0.3 < inv < 0.7
    shifted = E.copy()
    shifted[(meta["domain"] == "nfl").to_numpy(), 0] += 4.0
    assert dd.grouped_domain_auc(shifted, meta, rows, seed=0)["domain_auc"] > 0.95
    assert dd.rank_auc(np.ones(10), np.r_[np.zeros(5), np.ones(5)]) == pytest.approx(0.5)


def test_collapse_indicators_and_alignment(arrays):
    _, _, meta, _, _ = arrays
    rows = _all_rows(meta)
    v = np.random.default_rng(0).normal(size=8)
    collapsed = (np.random.default_rng(1).normal(size=(meta.height, 1)) * v).astype(np.float32)
    c = dd.collapse_indicators(collapsed, rows)
    assert c["effective_rank"] == pytest.approx(1.0, abs=1e-6) and c["pc1_share"] == pytest.approx(1.0, abs=1e-9)
    healthy = dd.collapse_indicators(_random(meta), rows)
    assert healthy["effective_rank"] > 5
    al = dd.cross_domain_alignment(_oracle(meta), meta, rows)
    assert al["alignment_gap_raw"] > 0.5 and al["alignment_gap_centered"] > 0.5


def test_positive_pair_coverage_counts_cross_domain_labels(arrays):
    _, _, meta, _, _ = arrays
    rows = _all_rows(meta)
    dom = meta["domain"].to_numpy()
    lab = np.where(dom == "combine", 1, 2)                                       # disjoint labels → no cross-domain positive
    out = dd.positive_pair_coverage(lab, meta, rows, "toy")
    assert all(r["value"] == 0.0 for r in out if r["n"])
    lab2 = np.ones(meta.height, int)                                             # one shared label → full coverage
    assert all(r["value"] == 1.0 for r in dd.positive_pair_coverage(lab2, meta, rows, "toy") if r["n"])
    lab3 = -np.ones(meta.height, int)                                            # unlabeled (< 0) never counts
    assert all(r["value"] == 0.0 for r in dd.positive_pair_coverage(lab3, meta, rows, "toy") if r["n"])


def test_domain_diagnostics_long_table(arrays):
    _, _, meta, _, _ = arrays
    t = dd.domain_diagnostics("oracle", _oracle(meta), meta, _all_rows(meta), fold=0, seed=0)
    assert t.columns == ["representation", "fold", "position_group", "domain", "metric", "value", "n"]
    got = set(t["metric"].to_list())
    assert {"domain_auc", "mean_gap_norm", "alignment_gap_raw", "effective_rank", "norm_mean", "pc1_share"} <= got
    assert {"ALL", "WR", "CB", "DL"} <= set(t["position_group"].to_list())


# ====================================================================== end-to-end: artifacts, reload, missing encoders
def test_compare_inductive_end_to_end(bundle_dir, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from nflbowl import Experiment

    tiny = {"science.epochs": 1, "science.batch": 16, "science.embed_dim": 8, "science.width": 8, "science.max_val_batches": 2, "science.patience": 10,
            "science.n_boot": 20, "runtime.device": "cpu", "runtime.progress": False, "runtime.embed_batch": 64}
    exp = Experiment(data_dir=bundle_dir, output_dir=tmp_path, overrides=tiny, name="paired")
    exp.prepare()
    exp.train("F2", regime="inductive", folds=[0, 1])
    exp.train("F4", regime="transductive")                                      # a transductive F4 must NOT stand in for inductive F4
    with pytest.raises(FileNotFoundError):
        exp.compare_inductive(models=["F0", "F4"], on_missing="raise", save=False)

    res = exp.compare_inductive(models=["F0", "F2", "F4"], reference="F0", bootstrap=50)
    ev, fg = exp.root / "evaluation", exp.root / "figures"
    for p in (ev / "paired_inductive_comparison.csv", ev / "athlete_retrieval_scores.parquet", ev / "position_subgroup_results.csv",
              ev / "domain_diagnostics.csv", fg / "inductive_model_comparison.png", fg / "inductive_delta_auc.png", fg / "domain_separation.png",
              exp.root / "reports" / "INDUCTIVE_RETRIEVAL_REVIEW.md"):
        assert p.is_file(), p
    assert res.meta["science_hash"] == exp.cfg.science_hash() and res.meta["seed"] == exp.cfg.science.seed and res.meta["bundle_fingerprint"]
    assert res.meta["unavailable"]["F4_physics_contrastive"] == [0, 1, 2, 3, 4]
    assert res.meta["unavailable"]["F2_cnn_physics"] == [2, 3, 4]
    auc = res.comparison.filter(pl.col("metric") == "auc")
    assert auc.filter(pl.col("representation") == "F4_physics_contrastive")["status"].to_list() == ["missing"]
    f2 = res.scores.filter(pl.col("representation") == "F2_cnn_physics")
    assert set(f2["fold"].unique().to_list()) == {0, 1}

    # no train/test leakage: every scored / candidate athlete belongs to its fold, and its training record is that fold's encoder
    fm = exp._folds
    for f, ids, cands in zip(res.scores["fold"].to_list(), res.scores["nfl_id"].to_list(), res.scores["candidate_ids"].to_list()):
        assert fm[ids] == f and all(fm[c] == f for c in cands)
    for f in (0, 1):
        rec = exp.manifest.find("training", f"F2_cnn_physics__inductive_f{f}")
        assert rec["fold"] == f and rec["deps"]["regime"] == "inductive"
        t = exp.load_embeddings("F2", regime="inductive", fold=f)
        held = t.filter(pl.col("split_role") == "held_out")
        assert set(held["fold"].unique().to_list()) == {f}

    # reuse: no new record, same numbers; reload in a fresh session without recomputation
    n_rec = len(exp.list_artifacts("inductive_comparison"))
    again = exp.compare_inductive(models=["F0", "F2", "F4"], reference="F0", bootstrap=50, reuse=True)
    assert len(exp.list_artifacts("inductive_comparison")) == n_rec
    assert _same(again.comparison.select("representation", "metric", "delta"), res.comparison.select("representation", "metric", "delta"))
    exp2 = Experiment.open(exp.root, device="cpu")
    loaded = exp2.load_inductive_comparison(models=["F2"], reference="F0")
    assert _same(loaded.scores, res.scores)
    assert loaded.meta["deps"] == res.meta["deps"]
