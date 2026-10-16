"""The fast evaluation paths must reproduce the nflbdb reference functions on fixtures (same estimands, same draws),
plus known-answer fixtures for the retrieval AUC."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest


def _emb(meta: pl.DataFrame, signal: float, seed: int = 1, d: int = 8) -> np.ndarray:
    """Athlete-specific direction + noise (``signal`` = 0 → pure noise)."""
    rng = np.random.default_rng(seed)
    ids = meta["nfl_id"].to_numpy()
    base = {int(i): rng.normal(size=d) for i in np.unique(ids)}
    return (signal * np.stack([base[int(i)] for i in ids]) + rng.normal(size=(ids.size, d))).astype(np.float32)


@pytest.fixture()
def setup(arrays):
    from nflbdb.research.advanced_movement.m11 import representation as rep
    from nflbowl.data.splits import fold_map

    X, M, meta, folds, _ = arrays
    return meta, rep.physical_scalars(X, M), fold_map(meta, folds, 5), rep


@pytest.mark.parametrize("signal", [0.0, 0.7, 3.0])
def test_within_and_cross_retrieval_match_reference(setup, signal):
    from nflbowl.evaluation import retrieval as fr

    meta, _, _, rep = setup
    E = _emb(meta, signal)
    assert fr.within_domain_retrieval(E, meta) == rep.within_domain_retrieval(E, meta)
    a, b = fr.cross_domain_retrieval(E, meta, 50, 11), rep.cross_domain_retrieval(E, meta, 50, 11)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == pytest.approx(b[k], nan_ok=True, abs=1e-12), k
    a, b = fr.cross_domain_retrieval(E, meta, 0, 3, combine_kinds=["forty"]), rep.cross_domain_retrieval(E, meta, 0, 3, combine_kinds=["forty"])
    assert a["auc"] == pytest.approx(b["auc"], nan_ok=True) and a["n_athletes"] == b["n_athletes"]


def test_matched_retrieval_matches_reference(setup):
    from nflbowl.evaluation import retrieval as fr

    meta, phys, _, rep = setup
    E = _emb(meta, 1.0)
    ref = rep.matched_retrieval(E, meta, phys, seed=5)
    for chunk in (7, 64, 10_000):
        got = fr.matched_retrieval(E, meta, phys, seed=5, chunk=chunk)
        for k in ref:
            assert got[k] == pytest.approx(ref[k], nan_ok=True), (chunk, k)
    approx = fr.matched_retrieval(E, meta, phys, seed=5, approx_topk=True)          # ties are the only possible difference
    assert approx["n_queries"] == ref["n_queries"]


def test_probe_r2_matches_reference(setup):
    from nflbdb.research.advanced_movement.m11.config import M11Config
    from nflbowl.evaluation.probes import probe_r2

    meta, phys, folds, rep = setup
    E = _emb(meta, 1.0)
    ref = rep.probe_r2(E, meta, phys, folds, replace(M11Config(), outer_folds=5))
    got = probe_r2(E, meta, phys, folds, 5)
    assert got.select("domain", "probe", "n").equals(ref.select("domain", "probe", "n"))
    assert np.allclose(got["r2_cv"].to_numpy(), ref["r2_cv"].to_numpy(), equal_nan=True, rtol=0, atol=1e-12)


@pytest.mark.parametrize("signal", [0.0, 2.0])
def test_identity_null_matches_reference(setup, signal):
    from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se
    from nflbowl.evaluation.null_tests import shuffled_identity_null

    meta, _, _, _ = setup
    E = _emb(meta, signal)
    ref = se.shuffled_identity_null(E, meta, 6, 21)
    got = shuffled_identity_null(E, meta, 6, 21)
    assert np.allclose(got, ref, equal_nan=True, atol=1e-12)


def test_evaluation_row_fast_equals_reference(setup):
    from nflbowl.config import load_config
    from nflbowl.evaluation.evaluator import evaluation_row

    meta, phys, folds, _ = setup
    E = _emb(meta, 1.5)
    fast = load_config("default", {"science.n_boot": 250}).evaluation
    ref = replace(fast, backend="reference")
    a, pa = evaluation_row("x", "transductive", E, meta, phys, folds, 250, 5, 9, fast)
    b, pb = evaluation_row("x", "transductive", E, meta, phys, folds, 250, 5, 9, ref)
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float):
            assert a[k] == pytest.approx(b[k], nan_ok=True, abs=1e-9), k
        else:
            assert a[k] == b[k], k
    assert np.allclose(pa["r2_cv"].to_numpy(), pb["r2_cv"].to_numpy(), equal_nan=True, atol=1e-12)


# ------------------------------------------------------------------------------ known answers
def test_perfect_identity_embedding_gives_auc_one(setup):
    from nflbowl.evaluation.retrieval import cross_domain_retrieval

    meta, _, _, _ = setup
    ids = meta["nfl_id"].to_numpy()
    uniq = np.unique(ids)
    E = np.eye(uniq.size, dtype=np.float32)[np.searchsorted(uniq, ids)]       # one-hot per athlete: own match is the only similar one
    r = cross_domain_retrieval(E, meta, 0, 0)
    assert r["auc"] == 1.0 and r["top5"] == 1.0 and r["n_athletes"] == uniq.size


def test_constant_embedding_gives_auc_zero(setup):
    """All similarities tie: the reference counts strictly-below others, so AUC = 0 (documents the tie convention)."""
    from nflbowl.evaluation.retrieval import cross_domain_retrieval

    meta, _, _, _ = setup
    r = cross_domain_retrieval(np.ones((meta.height, 4), np.float32), meta, 0, 0)
    assert r["auc"] == 0.0


def test_quick_level_is_flagged_diagnostic(setup):
    from nflbowl.config import load_config
    from nflbowl.evaluation.evaluator import evaluate_representation

    meta, phys, folds, _ = setup
    cfg = load_config("default", {"evaluation.quick_min_athletes": 20, "evaluation.quick_n_boot": 10})
    res = evaluate_representation("x", _emb(meta, 1.0), meta, phys, folds, cfg.science, cfg.evaluation, level="quick")
    assert res.row["diagnostic"] is True and res.row["comparable_to_full"] is False
    assert res.row["quick_athletes"] == 20
