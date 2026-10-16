"""M1.1 · Study A3 analysis functions on synthetic trajectory tables + study registration.

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_m1_1_signal_study.py -q
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflbdb.research.contracts import conform
from nflbdb.research.studies import signal_reliability as A3
from nflbdb.research.studies.config import SignalStudyConfig, StudiesConfig, threshold_table
from nflbdb.research.studies.results import ResultCollector

CFG = SignalStudyConfig(n_boot=50, min_group_trajectories=5, min_athletes=5)


def table(n_ath: int = 30, per: int = 4, seed: int = 0) -> pl.DataFrame:
    """SIGNALS-shaped frame: vendor_rmse is high exactly where resid_rms_yds is high (planted association)."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_ath):
        for k in range(per):
            noisy = a % 5 == 0 and k in (1, 2)                       # 12 of 120 rows; top 5 % are all noisy
            row = {"domain": "combine", "traj_id": f"E{a}|{a}|{k}", "nfl_id": a, "cohort": 2024 + a % 2,
                   "stratum": "FORTY_YARD_DASH", "position": "WR", "quality_tier": "B" if noisy else "A",
                   "duration_s": 5.0, "resid_rms_yds": 0.2 if noisy else 0.02, "speed_vendor_mad": 0.1,
                   "n_flagged_dims": int(noisy)}
            row |= {c: 30 for c in A3.INT_METRICS} | {"n_gap_breaks": 0, "n_jump_breaks": 0, "n_offgrid": 0}
            row |= {c: 0.1 for c in A3.SIGNAL_METRICS}
            row |= {"vendor_rmse": (0.5 + 0.01 * a + 0.001 * k) if noisy else float(rng.uniform(0.01, 0.05)),
                    "boundary_frac": 0.2,
                    "jerk_rms_interior": float(rng.uniform(1, 2)), "usable_frac": 1.0,
                    "lowspeed_frac": 0.1, "heading_null_frac": 0.1}
            for p in A3.PEAKS:
                inb = bool(k == 0)                                     # first attempt: peak on the boundary
                row |= {p: 5.0, f"{p}_interior": 4.5 if inb else 5.0, f"{p}_in_boundary": inb,
                        f"{p}_boundary_rel": 0.1 if inb else 0.0}
            rows.append(row)
    return pl.DataFrame(rows, schema=A3.SIGNALS.columns)


def test_auc_basics():
    assert A3.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert A3.auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert np.isnan(A3.auc(np.array([]), np.array([1.0])))


def test_planted_association_is_recovered_and_flags_shared_sources():
    t = table()
    rc = ResultCollector("A3", CFG.min_athletes)
    d = conform(A3.tier_discrimination(t, CFG, 1, rc), A3.DISCRIM)
    r = d.filter((pl.col("predictor") == "resid_rms_yds") & (pl.col("target") == "vendor_rmse_high")).row(0, named=True)
    assert r["auc"] > 0.9 and r["ci_lower"] is not None and not r["shares_source"]
    s = d.filter((pl.col("predictor") == "speed_vendor_mad") & (pl.col("target") == "vendor_rmse_high")).row(0, named=True)
    assert s["shares_source"]
    b = d.filter((pl.col("predictor") == "boundary_frac") & (pl.col("target") == "peak_decel_in_boundary")).row(0, named=True)
    assert b["shares_source"] and b["auc"] == pytest.approx(0.5)   # constant predictor → no discrimination
    assert rc.frame().height > 0


def test_degenerate_target_is_insufficient_not_dropped():
    t = table().with_columns(pl.lit(False).alias("peak_a_tan_in_boundary"))
    d = A3.tier_discrimination(t, CFG, 1, ResultCollector("A3", CFG.min_athletes))
    r = d.filter(pl.col("target") == "peak_a_tan_in_boundary")
    assert r.height == len(A3.PREDICTORS) and (r["result_status"] == "insufficient_data").all()


def test_boundary_share_against_expected():
    b = conform(A3.boundary_peak_share(table(), CFG, 1), A3.BOUNDARY)
    r = b.filter(pl.col("peak") == "peak_speed").row(0, named=True)
    assert r["share_in_boundary"] == pytest.approx(0.25)            # 1 of 4 attempts per athlete
    assert r["expected_share"] == pytest.approx(0.2) and r["ratio_observed_expected"] == pytest.approx(1.25)
    assert r["boot_lower"] <= 0.25 <= r["boot_upper"]


def test_sampling_table_counts_heading_inconsistency():
    t = table().with_columns(pl.when(pl.col("nfl_id") == 0).then(0.3).otherwise(pl.col("heading_null_frac"))
                             .alias("heading_null_frac"))
    s = conform(A3.sampling_irregularity(t), A3.SAMPLING)
    assert s["n_heading_inconsistent"].sum() == 4                   # athlete 0's four trajectories
    assert conform(A3.metric_summary(t), A3.SUMMARY).height > 0


def test_studies_registered_with_figure_folders():
    from nflbdb.research.studies.__main__ import STUDIES
    from nflbdb.research.studies.store import STUDY_DIRS

    assert list(STUDIES)[:3] == ["A0", "A", "A3"]
    for _, (cls, contracts, figs) in STUDIES.items():
        assert contracts and all(rec.study in STUDY_DIRS for rec, _ in figs)
    names = threshold_table(StudiesConfig())["study_section"].unique().to_list()
    assert {"quality", "integrity", "signal"} <= set(names)
