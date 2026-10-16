"""Phase 03 · M2 hardening regressions (synthetic, deterministic).

Covers: sprint fit window with a stationary lead-in, vmax extrapolation guard, tracked-vs-official
labelling and same-attempt flags, active-window braking events, reliability guards (attempt cap,
label minimum, degenerate distributions, CV for signed features), cohort-shift interpretation, and
the stable output-path contract.

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_phase03_m2_hardening.py -q
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflbdb.features.braking import braking_features
from nflbdb.features.config import MovementConfig
from nflbdb.features.reliability import reliability_table
from nflbdb.features.sprint import model_distance, sprint_profile
from nflbdb.research.combine_analytics import best_attempt, cohort_shift, timing_validation

MCFG = MovementConfig()
DT = 0.1


def mono(t, vmax=10.0, tau=1.2, t0=0.0):
    return vmax * (1 - np.exp(-np.clip(t - t0, 0, None) / tau))


# ----------------------------------------------------------------------------- sprint
def test_stationary_lead_in_is_excluded_from_fit():
    """v1 fitted from the first frame: a 1.5 s set position pushed t0 off the grid (fit failed)."""
    t0 = 1.5
    t = np.arange(0, 8.1, DT)                      # 6.5 s of running after t0 → > 40 yd covered
    x = model_distance(t, 10.0, 1.2, t0)
    p = sprint_profile(t, x, np.zeros_like(t), mono(t, t0=t0), MCFG)
    assert p["fit_ok"] and p["fit_start_rel_s"] > 1.0
    assert p["model_t0_rel_s"] == pytest.approx(t0, abs=0.05)
    assert p["model_tau_s"] == pytest.approx(1.2, abs=0.03)
    fine = np.arange(0, 8, 1e-4)
    assert p["trk_t40_s"] == pytest.approx(fine[np.argmax(model_distance(fine, 10.0, 1.2, 0.0) >= 40)], abs=0.03)
    assert p["pre_distance_yds"] == 0.0


def test_truncated_acceleration_cannot_support_extrapolated_vmax():
    t = np.arange(0, 1.6, DT)                      # recording stops at 1.5 s, still accelerating
    p = sprint_profile(t, model_distance(t, 10.0, 1.2, 0.0), np.zeros_like(t), mono(t), MCFG)
    assert p["peak_at_end"] and p["vmax_extrapolation_ratio"] > 1.10 and not p["fit_ok"]


# ----------------------------------------------------------------------------- tracked vs official
def test_best_attempt_keeps_flags_of_the_same_attempt():
    att = pl.DataFrame({"nfl_id": [1, 1], "drill": ["SHORT_SHUTTLE"] * 2, "draft_year": [2024] * 2,
                        "attempt_order": [1, 2], "onset_to_end_s": [4.5, 4.3], "end_censored": [False, True]})
    r = best_attempt(att, "SHORT_SHUTTLE", "onset_to_end_s", ("end_censored",)).row(0, named=True)
    assert (r["onset_to_end_s"], r["end_censored"], r["attempt_order"]) == (4.3, True, 2)


def test_timing_validation_labels_and_quantifies_bias():
    n = 30
    off = np.linspace(4.1, 4.8, n)
    att = pl.DataFrame({"nfl_id": list(range(n)), "drill": ["SHORT_SHUTTLE"] * n, "draft_year": [2024] * n,
                        "attempt_order": [1] * n, "onset_to_end_s": off + 0.20 + np.tile([-0.01, 0.01], n // 2),
                        "onset_censored": [False] * n, "end_censored": [True] * n})
    sprint = pl.DataFrame(schema={"nfl_id": pl.Int64, "drill": pl.String, "draft_year": pl.Int64,
                                  "attempt_order": pl.Int64, "fit_ok": pl.Boolean, "run_incomplete": pl.Boolean,
                                  "trk_t40_s": pl.Float64, "trk_t10_s": pl.Float64})
    base = pl.DataFrame({"nfl_id": list(range(n)), "cmb_short_shuttle": off})
    tv = timing_validation(att, sprint, base).filter(
        (pl.col("tracked_measure") == "onset_to_end_s") & (pl.col("cohort") == "all")).row(0, named=True)
    assert tv["official_measure"] == "cmb_short_shuttle" and "official" in tv["official_source"]
    assert tv["tracked_basis"].startswith("tracked")
    assert tv["bias_median_s"] == pytest.approx(0.20, abs=0.011)
    assert tv["agreement_class"] == "interchangeable_after_bias"
    assert tv["share_end_censored"] == 1.0 and tv["n_athletes"] == n


# ----------------------------------------------------------------------------- braking window
def test_braking_events_outside_active_window_are_not_counted():
    keys = ["event_id", "nfl_id", "attempt"]
    act = pl.DataFrame({"event_id": ["E"] * 3, "nfl_id": [1] * 3, "attempt": [1] * 3, "a_tan": [2.0, -3.0, -1.0]})
    ev = pl.DataFrame({"event_id": ["E", "E"], "nfl_id": [1, 1], "attempt": [1, 1], "event_type": ["braking"] * 2,
                       "rule": ["default"] * 2, "t_start_s": [1.0, 9.0], "duration_s": [0.5, 0.5],
                       "distance_yds": [2.0, 1.0], "mean_abs_a_tan": [3.0, 3.0], "peak_abs_a_tan": [4.0, 4.0],
                       "speed_change": [2.0, 1.5], "entry_speed": [5.0, 3.0], "time_to_transition_s": [0.4, None]})
    win = pl.DataFrame({"event_id": ["E"], "nfl_id": [1], "attempt": [1], "t_onset_s": [0.5], "t_end_active_s": [5.0]})
    assert braking_features(act, ev, keys, "default")["n_braking_events"].item() == 2
    r = braking_features(act, ev, keys, "default", win).row(0, named=True)
    assert r["n_braking_events"] == 1 and r["braking_time_s"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- reliability guards
def _long(n_ath: int, k: int, value_fn, feature="model_vmax"):
    return pl.DataFrame([{"drill": "D", "feature": feature, "nfl_id": a, "attempt_order": j + 1, "value": value_fn(a, j)}
                         for a in range(n_ath) for j in range(k)])


def test_reliability_caps_attempts_and_requires_athletes_for_label():
    cfg = MovementConfig(n_boot=200)
    r = reliability_table(_long(12, 6, lambda a, j: 10 + a + 0.01 * j), ["drill", "feature"], cfg).row(0, named=True)
    assert r["n_attempts"] == 12 * cfg.max_attempts            # first 4 of 6 attempts per athlete
    assert r["icc1"] > 0.95 and r["reliability"] == "insufficient_for_label"   # 12 < min_athletes_label


def test_reliability_flags_degenerate_counts_and_signed_cv():
    cfg = MovementConfig(n_boot=200)
    deg = reliability_table(_long(30, 2, lambda a, j: 1.0 if a < 3 else 0.0, "n_braking_events"),
                            ["drill", "feature"], cfg).row(0, named=True)
    assert deg["reliability"] == "degenerate_distribution" and deg["modal_share"] >= 0.8
    signed = reliability_table(_long(30, 2, lambda a, j: (a - 15) * 0.1 + 0.01 * j, "a_tan_mean_late"),
                               ["drill", "feature"], cfg).row(0, named=True)
    assert signed["cv_within_pct"] is None


# ----------------------------------------------------------------------------- cohort shift
def test_cohort_shift_separates_processing_from_athletic_differences():
    rows, base = [], []
    nid = 0
    for drill, feat, feat_shift, off_col, off_shift in (
            ("FORTY_YARD_DASH", "a_tan_max", 2.0, "cmb_forty", 0.0),          # tracking-only shift
            ("SHORT_SHUTTLE", "speed_max", 2.0, "cmb_short_shuttle", 0.4)):   # official time shifts too
        for year, on in ((2023, 0.0), (2025, 1.0)):
            for pos in ("WR", "OL"):
                for i, v in enumerate(np.linspace(0, 1, 20)):
                    rows.append({"nfl_id": nid, "drill": drill, "draft_year": year, "position_group": pos,
                                 "attempt_order": 1, "feature": feat, "value": v + on * feat_shift})
                    base.append({"nfl_id": nid, off_col: 4.2 + 0.4 * v + on * off_shift})
                    nid += 1
    long = pl.DataFrame(rows)
    b = pl.DataFrame(base, infer_schema_length=None)
    cs = cohort_shift(long, b, None, min_n=20, min_stratum=8, flag=0.2, anchor_tol=0.2)
    get = lambda d: cs.filter((pl.col("drill") == d) & (pl.col("draft_year") == 2025)).row(0, named=True)  # noqa: E731
    forty, shuttle = get("FORTY_YARD_DASH"), get("SHORT_SHUTTLE")
    assert forty["noise_sensitive"] and abs(forty["official_anchor_shift"]) < 0.2
    assert forty["interpretation"] == "likely_processing_artifact"
    assert not shuttle["noise_sensitive"] and shuttle["interpretation"] == "athletic_difference_possible"


# ----------------------------------------------------------------------------- output contract
def test_expected_outputs_are_stable_paths(synth):
    from nflbdb.research.combine_analytics import FIGURES, TABLES, CombineMovementAnalytics

    paths = CombineMovementAnalytics(synth).expected_outputs()
    assert len(TABLES) == 9 and len(FIGURES) == 13 and len(paths) == 9 * 2 + 13 * 2
    assert all("phase_03" in str(p) for p in paths)
    assert len(CombineMovementAnalytics(synth).check_outputs()) == len(paths)     # nothing produced yet
