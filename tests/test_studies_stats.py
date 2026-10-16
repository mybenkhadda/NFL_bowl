"""Shared study statistics + results schema (synthetic, deterministic).

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_studies_stats.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from nflbdb.research.contracts import conform
from nflbdb.research.studies.results import RESULTS, ResultCollector
from nflbdb.research.studies.stats import (bh_fdr, bland_altman, cliffs_delta, cluster_bootstrap,
                                           cluster_bootstrap_ratio, cluster_bootstrap_ratios, equivalence_decision,
                                           fisher_ci, percentile_interval, rankdata, spearman, wilcoxon_signed_rank,
                                           wilson_ci)


def test_rankdata_averages_ties():
    assert rankdata(np.array([10, 20, 20, 5])).tolist() == [2.0, 3.5, 3.5, 1.0]
    assert rankdata(np.array([])).size == 0


def test_cliffs_delta_extremes_and_symmetry():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == pytest.approx(0.0)
    assert np.isnan(cliffs_delta([], [1.0]))


def test_spearman_is_monotone_invariant():
    x = np.arange(50, dtype=float)
    assert spearman(x, np.exp(x / 10)) == pytest.approx(1.0)
    assert spearman(x, -x ** 3) == pytest.approx(-1.0)
    lo, hi = fisher_ci(0.5, 100)
    assert lo < 0.5 < hi


def test_wilson_interval_bounds():
    lo, hi = wilson_ci([0, 5, 10], [10, 10, 10])
    assert lo[0] == pytest.approx(0.0, abs=1e-12) and hi[2] == pytest.approx(1.0, abs=1e-12) and lo[1] < 0.5 < hi[1]
    lo, hi = wilson_ci([0], [0])
    assert np.isnan(lo[0]) and np.isnan(hi[0])


def test_bh_fdr_matches_hand_computation_and_ignores_nan():
    adj = bh_fdr([0.01, 0.04, 0.03, np.nan])
    assert adj[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(adj[3])


def test_wilcoxon_detects_shift_and_needs_n():
    rng = np.random.default_rng(0)
    assert wilcoxon_signed_rank(rng.normal(1.0, 1.0, 60))["p_value"] < 0.001
    assert np.isnan(wilcoxon_signed_rank([0.1, -0.2, 0.3])["p_value"])


def test_bland_altman_and_equivalence():
    ba = bland_altman([0.1, 0.1, 0.1, 0.1])
    assert ba["bias"] == pytest.approx(0.1) and ba["loa_lower"] == pytest.approx(0.1)
    assert equivalence_decision(-0.02, 0.03, 0.05) == "equivalent"
    assert equivalence_decision(0.06, 0.2, 0.05) == "not_equivalent"
    assert equivalence_decision(-0.02, 0.08, 0.05) == "inconclusive"


def test_cluster_bootstrap_ratio_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(1)
    num, den = rng.uniform(0, 10, 40), rng.integers(5, 30, 40).astype(float)
    a = cluster_bootstrap_ratio(num, den, 500, seed=7)
    b = cluster_bootstrap_ratio(num, den, 500, seed=7)
    assert a == b and a["ci_lower"] <= a["estimate"] <= a["ci_upper"]
    est, reps = cluster_bootstrap_ratios(np.column_stack([num, num * 2]), den, 500, seed=7)
    assert est[1] == pytest.approx(2 * est[0]) and reps.shape == (500, 2)
    lo, hi, se = percentile_interval(reps)
    assert lo[0] == pytest.approx(a["ci_lower"]) and np.all(se > 0)


def test_cluster_bootstrap_single_cluster_gives_no_interval():
    r = cluster_bootstrap_ratio([3.0], [4.0])
    assert r["estimate"] == 0.75 and np.isnan(r["ci_lower"])


def test_cluster_bootstrap_resamples_whole_clusters():
    """Each athlete contributes 3 identical rows: a resample always contains multiples of 3 rows/athlete."""
    clusters = np.repeat(np.arange(10), 3)
    sizes = []

    def stat(idx):
        sizes.append(np.bincount(clusters[idx], minlength=10))
        return float(len(idx))

    r = cluster_bootstrap(stat, clusters, n_boot=50, seed=3)
    assert r["estimate"] == 30.0 and all((s % 3 == 0).all() for s in sizes)


# ----------------------------------------------------------------------------- results schema
def _add(rc, **kw):
    base = dict(analysis_id="X", grain="trajectory", n_athletes=50, n_trajectories=100, effect_estimate=0.2,
                effect_size_type="d", method="m")
    rc.add(**{**base, **kw})


def test_results_enforce_grain_and_sample_size_rules():
    rc = ResultCollector("A", min_athletes=10)
    with pytest.raises(ValueError):
        _add(rc, grain="frame")                               # frame grain needs n_frames
    with pytest.raises(ValueError):
        _add(rc, n_frames=1000)                               # n_frames only for frame grain
    with pytest.raises(ValueError):
        _add(rc, grain="pixel")
    with pytest.raises(ValueError):
        _add(rc, result_status="confirmed")
    _add(rc, n_athletes=3, feature="small")
    _add(rc, grain="frame", n_frames=10_000, feature="frames")
    df = conform(rc.frame(), RESULTS)
    small = df.filter(df["feature"] == "small").row(0, named=True)
    assert small["result_status"] == "insufficient_data" and "athletes" in small["limitations"]
    assert df.filter(df["feature"] == "frames")["n_frames"][0] == 10_000


def test_results_bh_within_family_only():
    rc = ResultCollector("A")
    for i, p in enumerate([0.01, 0.04, 0.03]):
        _add(rc, feature=f"f{i}", p_value=p, adjustment_family="fam")
    _add(rc, feature="solo", p_value=0.04)
    df = rc.frame()
    adj = dict(zip(df["feature"], df["p_adjusted"]))
    assert adj["f0"] == pytest.approx(0.03) and adj["solo"] == pytest.approx(0.04)
