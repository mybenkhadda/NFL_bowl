"""Static tests for the sprint reliability rescue experiment (synthetic data only — no warehouse access).

Authored 2026-10-11, NOT executed by Claude. Run::

    python -m pytest tests/test_sprint_rescue.py -q
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from nflbdb.features.config import MovementConfig
from nflbdb.features.reliability import icc_oneway
from nflbdb.features.sprint import _grid, fit_mono_exponential, sprint_profile
from nflbdb.research.contracts import conform
from nflbdb.research.sprint_rescue import attempts as at
from nflbdb.research.sprint_rescue import censoring as ce
from nflbdb.research.sprint_rescue import cohorts as co
from nflbdb.research.sprint_rescue import redundancy as rd
from nflbdb.research.sprint_rescue import reliability as rl
from nflbdb.research.sprint_rescue import report as rp
from nflbdb.research.sprint_rescue import sprint_model as sm
from nflbdb.research.sprint_rescue.config import VARIANTS, RescueConfig, feature_specs
from nflbdb.research.sprint_rescue.figures import FIGURES
from nflbdb.research.sprint_rescue.pipeline import ALL_CONTRACTS, STAGES
from nflbdb.research.sprint_rescue.stats import (bootstrap_picks, observations_needed, paired_difference, partial_corr,
                                                 r_squared, spearman_brown, stratified_icc, within_athlete_slopes)
from nflbdb.research.sprint_rescue.store import RescueStore

CFG = RescueConfig(n_boot=200, min_athletes=5, min_athletes_label=8, min_athletes_assoc=5)
MCFG = MovementConfig()
DRILL = CFG.drill


# ------------------------------------------------------------------------------ synthetic data
def _curve(n: int = 45, offset: float = 0.2, vmax: float = 10.0, tau: float = 0.9):
    """Sprint recorded from ``offset`` seconds after the true start (offset > 0 = onset-censored)."""
    t = np.arange(n) * 0.1
    u = t + offset
    v = vmax * (1 - np.exp(-u / tau))
    x = vmax * (u - tau * (1 - np.exp(-u / tau)))
    return t, v, vmax / tau * np.exp(-u / tau), x - x[0]


def _frames(attempts: list[tuple[str, int, int, float]]) -> pl.DataFrame:
    rows = []
    for event, nfl_id, attempt, offset in attempts:
        t, v, a, x = _curve(offset=offset)
        for i in range(t.size):
            rows.append({"event_id": event, "nfl_id": nfl_id, "attempt": attempt, "drill": DRILL, "seg_idx": 1, "pos": i,
                         "seg_len": t.size, "t_s": t[i], "x_s": x[i], "y_s": 0.0, "speed": v[i], "a_tan": a[i],
                         "valid_kin": True})
    return pl.DataFrame(rows)


def _attempt_table(n_per_pos: int = 12, seed: int = 0) -> pl.DataFrame:
    """Two attempts per athlete, two position groups. ``p95_speed`` differs between positions only;
    ``p95_a_tan`` differs between athletes; the official forty tracks p95_a_tan."""
    rng = np.random.default_rng(seed)
    rows, nfl_id = [], 0
    for pos, speed in (("DL", 8.0), ("WR", 10.0)):
        for _ in range(n_per_pos):
            nfl_id += 1
            ability = rng.normal(0, 1.0)
            for order in (1, 2):
                r = dict.fromkeys(at.ATTEMPTS.columns)
                start = float(rng.uniform(0.0, 1.5))
                r.update(event_id=f"E{order}", nfl_id=nfl_id, attempt=order, traj_id=f"E{order}|{nfl_id}|{order}",
                         draft_year=2023 + nfl_id % 2, position_group=pos, attempt_order=order, n_attempts=2,
                         quality_tier="A", cmb_forty=5.0 - 0.1 * ability, cmb_ten_yd_split=1.7 - 0.02 * ability,
                         first_speed=start, last_speed=speed, duration_s=4.5 - 0.2 * start, path_smooth_yds=40.0 - start,
                         net_disp_yds=39.9 - start, frames_after_peak=int(rng.integers(0, 5)),
                         last_speed_frac_of_peak=1.0, peak_speed=speed + rng.normal(0, 0.3),
                         peak_speed_interior=speed + rng.normal(0, 0.3), p95_speed=speed + rng.normal(0, 0.3),
                         peak_a_tan=6 + ability + rng.normal(0, 0.2),
                         peak_a_tan_interior=6 + ability + rng.normal(0, 0.2), p95_a_tan=5 + ability + rng.normal(0, 0.2) - 0.5 * start,
                         p95_decel=float(rng.normal(0, 1)), accel_sustained=4 + ability + rng.normal(0, 0.2),
                         accel_initial=4 + rng.normal(0, 1), onset_censored=start > 0.5, end_censored=True,
                         peak_at_end=False, boundary_peak=False, structural_flag=False, signal_flag=False,
                         band_start_censored=False, resid_rms_yds=0.01, smoothing="sg7p2", fit_ok=True,
                         fit_at_grid_edge=False, n_fit_frames=40, fit_start_rel_s=0.0, model_rmse=0.1 + 0.05 * start,
                         model_r2=0.99, vmax_extrapolation_ratio=1.03, model_t0_rel_s=-0.2, pre_distance_yds=0.0,
                         model_vmax=speed + 0.3, model_tau_s=0.9 - 0.05 * ability + rng.normal(0, 0.02),
                         model_a0=10 + ability + rng.normal(0, 0.3), model_a_at_ref=7 + ability + rng.normal(0, 0.3),
                         tau_rel_width=0.2, weakly_identified=False, S0_all=True, S1_structural=True, S2_signal=True,
                         S3_no_boundary=True, **{b: 4 + ability + rng.normal(0, 0.2) for b in at.BANDS})
                rows.append(r)
    return pl.DataFrame(rows, schema=at.ATTEMPTS.columns)


def _fits(att: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for r in att.iter_rows(named=True):
        for var in VARIANTS:
            rows.append({**dict.fromkeys(sm.FITS.columns), "event_id": r["event_id"], "nfl_id": r["nfl_id"],
                         "attempt": r["attempt"], "variant": var, "status": "ok", "fit_ok": True, "n_fit_frames": 40,
                         "vmax": r["model_vmax"], "tau_s": r["model_tau_s"], "a0": r["model_a0"],
                         "a_at_ref": r["model_a_at_ref"]})
    return sm.finalize_fits(rows)


# ------------------------------------------------------------------------------ ICC
def test_stratified_icc_with_one_stratum_equals_icc_oneway():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(30), rng.integers(2, 5, 30))
    values = rng.normal(0, 1, 30)[groups] + rng.normal(0, 0.7, groups.size)
    ref = icc_oneway(values, groups)
    got = stratified_icc(values, groups)
    assert got["icc"] == pytest.approx(ref["icc"], rel=1e-10)
    assert got["var_within"] == pytest.approx(ref["msw"], rel=1e-10) and got["n0"] == pytest.approx(ref["n0"], rel=1e-10)
    assert got["n_athletes"] == 30 and got["n_obs"] == groups.size


def test_hand_computed_icc():
    r = stratified_icc(np.array([1.0, 3.0, 5.0, 7.0]), np.array([1, 1, 2, 2]))       # MSB 16, MSW 2, n0 2
    assert r["icc"] == pytest.approx((16 - 2) / (16 + 2)) and r["var_between"] == pytest.approx(7.0)


def test_position_conditioning_removes_between_position_variance():
    rng = np.random.default_rng(2)
    ath = np.repeat(np.arange(60), 2)
    pos = np.where(ath < 30, "DL", "WR")
    values = np.where(ath < 30, 8.0, 10.0) + rng.normal(0, 0.3, ath.size)             # no athlete effect at all
    pooled, adj = stratified_icc(values, ath), stratified_icc(values, ath, pos)
    assert pooled["icc"] > 0.8 and abs(adj["icc"]) < 0.3
    assert adj["n_strata"] == 2 and adj["var_between"] < 0.1 * pooled["var_between"]


def test_bootstrap_is_deterministic_stratified_and_paired():
    rng = np.random.default_rng(3)
    ath = np.repeat(np.arange(40), 2)
    pos = np.where(ath < 15, "A", "B")
    y = rng.normal(0, 1, 40)[ath] + rng.normal(0, 0.5, ath.size)
    a, b = stratified_icc(y, ath, pos, 300, 7), stratified_icc(y, ath, pos, 300, 7)
    assert (a["ci_lower"], a["ci_upper"]) == (b["ci_lower"], b["ci_upper"]) and a["ci_lower"] < a["icc"] < a["ci_upper"]
    assert stratified_icc(y, ath, pos, 300, 8)["ci_lower"] != a["ci_lower"]
    groups = a["stats"]["groups"]
    pick = bootstrap_picks(groups, 50, 1)
    assert all(np.isin(pick[:, g], g).all() for g in groups)                          # resampled inside each stratum
    assert paired_difference(a, b) == (0.0, 0.0, 0.0)
    worse = stratified_icc(y + rng.normal(0, 2, y.size), ath, pos, 300, 7)
    diff, lo, hi = paired_difference(a, worse)
    assert diff > 0 and lo <= diff <= hi
    other = stratified_icc(y[:40], ath[:40], pos[:40], 300, 7)                        # different athletes → no paired CI
    assert np.isnan(paired_difference(a, other)[1])


def test_partial_association_and_helpers():
    rng = np.random.default_rng(4)
    g = np.repeat(["A", "B", "C"], 60)
    shift = np.repeat([0.0, 5.0, 10.0], 60)
    x, y = shift + rng.normal(0, 1, 180), shift + rng.normal(0, 1, 180)               # related through the group only
    assert partial_corr(y, x)["r"] > 0.8
    adj = partial_corr(y, x, categorical=[g])
    assert abs(adj["r"]) < 0.25 and adj["q"] == 2 and adj["ci_lower"] < adj["r"] < adj["ci_upper"]
    y[:10] = np.nan
    assert partial_corr(y, x)["n"] == 170
    z = rng.normal(0, 1, 180)
    assert r_squared(2 * z + rng.normal(0, 0.1, 180), [z])["r2"] > 0.95
    ath = np.repeat(np.arange(50), 2)
    w = rng.normal(0, 1, 100)
    v = rng.normal(0, 3, 50)[ath] - 0.5 * w                                           # within-athlete slope −0.5
    assert within_athlete_slopes(v, w, ath)[0] == pytest.approx(-0.5, abs=1e-9)
    assert spearman_brown(0.5, 2) == pytest.approx(2 / 3) and observations_needed(0.1, 0.7) == pytest.approx(21.0)


# ------------------------------------------------------------------------------ windows and model
def test_speed_band_acceleration_and_censored_start():
    t = np.arange(0, 5.01, 0.1)
    v = 2.0 * t                                                                       # constant 2 yd/s²
    assert at.band_acceleration(t, v, 2.0, 5.0) == pytest.approx(2.0)
    assert np.isnan(at.first_crossing(t, v + 3.0, 2.0))                               # recording starts above the level
    assert np.isnan(at.band_acceleration(t, np.minimum(v, 4.0), 2.0, 5.0))            # upper level never reached
    f = at.window_features(t, v + 3.0, np.full(t.size, 2.0), v, np.zeros(t.size), np.arange(t.size),
                           np.full(t.size, t.size), 3, CFG)
    assert f["band_start_censored"] and f[at.BANDS[0]] is None and f["first_speed"] == pytest.approx(3.0)
    assert f["frames_after_peak"] == 0 and f["duration_s"] == pytest.approx(5.0)
    assert f["first_a_tan"] == pytest.approx(2.0) and f["first_a_tan_interior"] == pytest.approx(2.0)


def test_base_refit_reproduces_existing_sprint_profile():
    t, v, _, x = _curve()
    prof = sprint_profile(t, x, np.zeros(t.size), v, MCFG)
    row = sm.fit_variant(t, v, "base", MCFG, CFG)
    assert row["vmax"] == pytest.approx(prof["model_vmax"]) and row["tau_s"] == pytest.approx(prof["model_tau_s"])
    assert row["t0_rel_s"] == pytest.approx(prof["model_t0_rel_s"]) and row["fit_ok"] == prof["fit_ok"]
    assert row["n_fit_frames"] == prof["n_fit_frames"] and row["a0"] == pytest.approx(prof["model_a0"])
    assert row["a0_extrapolated"] and row["tau_s"] == pytest.approx(0.9, abs=0.02)   # start 0.2 s before the recording
    assert row["a_at_ref"] == pytest.approx((row["vmax"] - CFG.model_ref_speed) / row["tau_s"])


def test_sse_surface_minimum_is_the_existing_optimum_and_region_is_reported():
    t, v, _, _ = _curve(n=30)
    v = v + np.random.default_rng(5).normal(0, 0.05, v.size)
    tau_grid, t0_grid = _grid(MCFG.tau_grid), t[0] + _grid(MCFG.t0_offset_grid)
    fit = fit_mono_exponential(t, v, tau_grid, t0_grid)
    sse, vmax = sm.sse_surface(t, v, tau_grid, t0_grid)
    i, j = np.unravel_index(int(np.argmin(sse)), sse.shape)
    assert tau_grid[i] == pytest.approx(fit["tau"]) and vmax[i, j] == pytest.approx(fit["vmax"])
    idf = sm.identifiability(sse, vmax, tau_grid, t0_grid, 0.10)
    assert idf["tau_lo"] <= fit["tau"] <= idf["tau_hi"] and idf["a0_lo"] <= fit["vmax"] / fit["tau"] <= idf["a0_hi"]
    wide = sm.identifiability(sse, vmax, tau_grid, t0_grid, 1.0)
    assert wide["tau_hi"] - wide["tau_lo"] >= idf["tau_hi"] - idf["tau_lo"]


def test_fit_failures_keep_every_variant_row():
    short = sm.fit_attempt(np.arange(3) * 0.1, np.array([0.0, 1.0, 2.0]), MCFG, CFG)
    assert [r["variant"] for r in short] == list(VARIANTS) and {r["status"] for r in short} == {"window_too_short"}
    flat = sm.fit_attempt(np.arange(30) * 0.1, np.zeros(30), MCFG, CFG)
    assert all(not r["fit_ok"] for r in flat) and {r["status"] for r in flat} <= {"degenerate", "window_too_short"}
    t, v, _, _ = _curve()
    good = sm.fit_attempt(t, v, MCFG, CFG)
    key = {"event_id": "E", "nfl_id": 1, "attempt": 1}
    df = conform(sm.finalize_fits([{**key, **r} for r in good] + [{**key, "attempt": 2, **r} for r in short]), sm.FITS)
    assert df.height == 2 * len(VARIANTS)
    base = df.filter((pl.col("attempt") == 1) & (pl.col("variant") == "base")).row(0, named=True)
    assert base["d_tau_rel"] == pytest.approx(0.0) and base["status"] == "ok"
    drop = df.filter((pl.col("attempt") == 1) & (pl.col("variant") == "drop3")).row(0, named=True)
    assert drop["n_fit_frames"] == base["n_fit_frames"] - 3 and abs(drop["d_tau_rel"]) < 0.05   # noise-free curve
    assert df.filter(pl.col("attempt") == 2)["d_tau_rel"].null_count() == len(VARIANTS)
    assert conform(sm.finalize_fits([]), sm.FITS).height == 0


# ------------------------------------------------------------------------------ attempt table
def test_attempt_matching_flags_and_missing_inputs():
    frames = _frames([("E1", 1, 1, 0.0), ("E1", 1, 2, 0.5), ("E1", 2, 1, 0.2)])
    feats, fits = at.frame_products(frames, MCFG, CFG, 3)
    assert feats.height == 3 and fits.height == 3 * len(VARIANTS)
    sprint = pl.DataFrame({"event_id": ["E1"] * 3, "nfl_id": [1, 1, 2], "attempt": [1, 2, 1], "drill": [DRILL] * 3,
                           "draft_year": [2023] * 3, "position_group": ["DL", "DL", None], "attempt_order": [1, 2, 1],
                           "quality_tier": ["A"] * 3, "onset_censored": [False, True, False], "end_censored": [True] * 3,
                           "peak_at_end": [True] * 3, "fit_ok": [True] * 3, "fit_at_grid_edge": [False] * 3,
                           "n_fit_frames": [44] * 3, "fit_start_rel_s": [0.0] * 3, "model_rmse": [0.0] * 3,
                           "model_r2": [1.0] * 3, "vmax_extrapolation_ratio": [1.0] * 3, "model_t0_rel_s": [0.0] * 3,
                           "pre_distance_yds": [0.0] * 3, "model_vmax": [10.0] * 3, "model_tau_s": [0.9] * 3,
                           "model_a0": [11.1] * 3, "first_speed": [9.9] * 3, "last_speed": [9.9] * 3})
    af = sprint.select("event_id", "nfl_id", "attempt", pl.lit(5.0).alias("accel_sustained"),
                       pl.lit(4.0).alias("accel_initial"), pl.lit(0.01).alias("resid_rms_yds"))
    ids = ["E1|1|1", "E1|1|2"]                                           # athlete 2 has no Phase 04 row
    p04 = pl.DataFrame([{"domain": "combine", "traj_id": i, "drill": DRILL, "smoothing": s, "feature": f, "value": val}
                        for i in ids for s, k in (("sg7p2", 1.0), ("sg9p2", 99.0)) for f, val in (("p95_a_tan", 5.0 * k),
                                                                                                  ("p95_speed", 9.0 * k))])
    flags = pl.DataFrame({"domain": ["combine"] * 2, "traj_id": ids, "structural_flag": [False, False],
                          "signal_flag": [False, True], "boundary_flag": [False, False]})
    official = pl.DataFrame({"nfl_id": [1], "cmb_forty": [4.9], "cmb_ten_yd_split": [None]},
                            schema={"nfl_id": pl.Int64, "cmb_forty": pl.Float64, "cmb_ten_yd_split": pl.Float64})
    att = conform(at.build_attempt_table(sprint, af, p04, flags, feats, fits, official, CFG), at.ATTEMPTS)
    a11, a12, a21 = (att.filter((pl.col("nfl_id") == i) & (pl.col("attempt") == k)).row(0, named=True)
                     for i, k in ((1, 1), (1, 2), (2, 1)))
    assert att.height == 3 and a11["traj_id"] == "E1|1|1" and a11["n_attempts"] == 2 and a21["n_attempts"] == 1
    assert a11["p95_a_tan"] == 5.0 and a11["p95_speed"] == 9.0           # primary smoothing only
    assert a11["first_speed"] == pytest.approx(0.0) and a12["first_speed"] > 3.0   # recomputed from frames, not 9.9
    assert a11["S2_signal"] and not a12["S2_signal"] and a12["S1_structural"]
    assert a21["p95_a_tan"] is None and not a21["S1_structural"] and a21["position_group"] == "UNKNOWN"
    assert a11["cmb_forty"] == 4.9 and a11["cmb_ten_yd_split"] is None and a21["cmb_forty"] is None
    assert a11["model_a_at_ref"] is not None and a11["tau_rel_width"] is not None
    assert a12["band_start_censored"] and a12[at.BANDS[0]] is None and a11[at.BANDS[0]] is not None

    within = conform(at.within_athlete_variation(att), at.WITHIN)
    w = within.row(0, named=True)
    assert within.height == 1 and w["censor_pattern"] == "discordant" and not w["both_signal_pass"]
    assert w["d_first_speed"] == pytest.approx(a12["first_speed"] - a11["first_speed"]) and w["d_p95_a_tan"] == 0.0
    conform(at.window_distribution(att), at.WINDOWS)
    ex = conform(at.example_signals(frames, fits, at.pick_examples(att), CFG), at.EXAMPLES)
    assert ex["traj_id"].n_unique() >= 1 and ex.filter(pl.col("in_fit_window")).height > 0


# ------------------------------------------------------------------------------ studies
def test_reliability_table_separates_position_from_athlete_signal():
    att = _attempt_table()
    rel = conform(rl.reliability_table(att, feature_specs(CFG), CFG), rl.RELIABILITY)
    get = lambda f, scope, sc="S0_all": rel.filter((pl.col("feature") == f) & (pl.col("scope") == scope)  # noqa: E731
                                                   & (pl.col("scenario") == sc)).row(0, named=True)
    speed_pooled, speed_adj = get("p95_speed", "pooled"), get("p95_speed", "position_adjusted")
    assert speed_pooled["icc1"] > 0.7 and speed_adj["icc1"] < 0.5 and speed_adj["position_dependent"]
    assert speed_adj["position_share_of_between"] > 0.7 and speed_adj["icc_gap_vs_pooled"] > 0.3
    accel = get("accel_sustained", "position_adjusted")
    assert accel["icc1"] > 0.8 and not accel["position_dependent"] and accel["n_athletes"] == 24 and accel["n_attempts"] == 48
    assert get("accel_sustained", "position:DL")["n_athletes"] == 12 and get("accel_sustained", "cohort:2023")["n_athletes"] == 12
    assert get("p95_decel", "pooled")["cv_within_pct"] is None            # signed feature: no CV
    assert get("accel_sustained", "pooled", "S3_no_boundary")["icc1"] == pytest.approx(get("accel_sustained", "pooled")["icc1"])
    few = rl.reliability_table(att.filter(pl.col("nfl_id") <= 3), feature_specs(CFG), CFG)
    assert set(few["reliability"].to_list()) == {"insufficient_athletes"} and few["icc1"].null_count() == few.height


def test_model_repeatability_uses_identical_attempts():
    att = _attempt_table()
    att = att.with_columns(pl.when(pl.col("nfl_id") == 1).then(None).otherwise(pl.col(at.BANDS[0])).alias(at.BANDS[0]))
    m = conform(rl.model_repeatability(att, _fits(att), CFG), rl.MODEL_REPEAT)
    get = lambda f, v, c, s="position_adjusted": m.filter((pl.col("feature") == f) & (pl.col("variant") == v)  # noqa: E731
                                                          & (pl.col("cohort") == c) & (pl.col("scope") == s)).row(0, named=True)
    assert get("a0", "persisted", "own")["n_athletes"] == 24 and get("a0", "persisted", "common")["n_athletes"] == 23
    common = {r["n_athletes"] for r in m.filter(pl.col("cohort") == "common").iter_rows(named=True)}
    assert common == {23}                                                 # one cohort for every representation
    ref = get("p95_a_tan", "direct", "common")
    assert ref["icc_diff_vs_ref"] is None and ref["reference"] == "p95_a_tan"
    a0 = get("a0", "base", "common")
    assert a0["ref_icc_same_cohort"] == pytest.approx(ref["icc1"])
    assert a0["icc_diff_vs_ref"] == pytest.approx(a0["icc1"] - ref["icc1"]) and a0["diff_ci_lower"] < a0["diff_ci_upper"]
    assert get("a0", "base", "common")["icc1"] == pytest.approx(get("a0", "persisted", "common")["icc1"])


def test_censoring_association_detects_a_window_dependent_feature():
    att = _attempt_table(n_per_pos=30)
    within = at.within_athlete_variation(att)
    assoc = conform(ce.association_table(att, within, CFG), ce.ASSOC)
    get = lambda an, out, pred, adj="none", stat="spearman", grp="all": assoc.filter(  # noqa: E731
        (pl.col("analysis") == an) & (pl.col("outcome") == out) & (pl.col("predictor") == pred) & (pl.col("adjustment") == adj)
        & (pl.col("statistic") == stat) & (pl.col("group") == grp)).row(0, named=True)
    dep = get("A2_paired_association", "d_p95_a_tan", "d_first_speed")    # built with slope −0.5 on start speed
    assert dep["estimate"] < -0.4 and dep["ci_upper"] < 0 and dep["n"] == 60
    free = get("A2_paired_association", "d_accel_sustained", "d_first_speed")
    assert abs(free["estimate"]) < 0.4 and free["ci_lower"] < free["ci_upper"]   # built without a window effect
    gain = get("A3_window_adjusted_icc", "p95_a_tan", "start", "position", "icc_gain")
    assert gain["estimate"] > 0 and gain["ci_lower"] > 0
    raw, adj = (get("A3_window_adjusted_icc", "p95_a_tan", "start", "position", s) for s in ("icc_raw", "icc_window_adjusted"))
    assert adj["estimate"] - raw["estimate"] == pytest.approx(gain["estimate"])
    assert assoc.filter(pl.col("analysis") == "A3_icc_by_pattern")["group"].n_unique() >= 5
    assert assoc.filter(pl.col("analysis") == "A2_window_r2").height == len(ce.ACCEL) + len(ce.CONTROL)
    assert ce.association_table(att.head(4), within.head(2), CFG).height == 0


def test_redundancy_and_screening():
    att = _attempt_table(n_per_pos=30)
    att = att.with_columns(pl.when(pl.col("nfl_id") <= 4).then(None).otherwise(pl.col("cmb_forty")).alias("cmb_forty"))
    ath = conform(rd.athlete_table(att, CFG), rd.ATHLETE)
    assert ath.height == 120 and ath.filter(pl.col("aggregation") == "mean")["n_attempts"].to_list() == [2] * 60
    specs = feature_specs(CFG)
    red = conform(rd.redundancy_table(ath, specs, CFG), rd.REDUNDANCY)
    get = lambda f, off, scope="position_adjusted": red.filter((pl.col("feature") == f) & (pl.col("official") == off)  # noqa: E731
                                                              & (pl.col("scope") == scope) & (pl.col("aggregation") == "mean")).row(0, named=True)
    accel = get("accel_sustained", rd.FORTY)
    assert accel["spearman"] < -0.8 and accel["n_athletes"] == 56 and accel["n_missing_official"] == 4
    assert accel["r2"] == pytest.approx(accel["pearson"] ** 2)
    speed = get("p95_speed", rd.FORTY)
    assert abs(speed["spearman"]) < 0.4 and speed["spearman_lower"] < speed["spearman"] < speed["spearman_upper"]
    assert get("accel_sustained", rd.JOINT)["spearman"] is None and get("accel_sustained", rd.JOINT)["r2"] > 0.6
    assert get("accel_sustained", rd.FORTY, "position:DL")["n_athletes"] == 26
    rel = rl.reliability_table(att, specs, CFG)
    screen = conform(rd.screening_summary(specs, rel, red, {"p95_speed": "VALIDATED"}, CFG), rd.SCREEN)
    s = {r["feature"]: r for r in screen.iter_rows(named=True)}
    assert s["accel_sustained"]["screening_status"] == "redundant_with_stopwatch" and s["accel_sustained"]["quadrant"] == "reliable_redundant"
    assert s["p95_speed"]["phase04_status"] == "VALIDATED" and s["accel_sustained"]["phase04_status"] == "not_in_registry"
    assert s["p95_speed"]["position_dependent"] and s["p95_speed"]["screening_status"] == "not_repeatable"
    assert s["accel_sustained"]["reliable_unique_share"] <= s["accel_sustained"]["reliability_of_mean"]
    assert s["accel_sustained"]["redundancy_lower"] <= s["accel_sustained"]["redundancy_abs"] <= s["accel_sustained"]["redundancy_upper"]


def test_screening_label_is_interval_aware():
    lab = lambda *a: rd.screening_label(*a, RescueConfig())  # noqa: E731
    assert lab(0.80, 0.95, 0.10, 0.40) == "strong_candidate" and lab(0.55, 0.80, 0.10, 0.40) == "candidate"
    assert lab(0.10, 0.45, 0.10, 0.40) == "not_repeatable" and lab(0.80, 0.95, 0.75, 0.95) == "redundant_with_stopwatch"
    assert lab(0.40, 0.80, 0.10, 0.40) == "inconclusive" and lab(0.80, 0.95, 0.50, 0.90) == "inconclusive"
    assert lab(None, None, 0.1, 0.2) == "not_estimable" and lab(0.8, 0.9, None, None) == "redundancy_not_estimable"


def test_cohort_reconciliation_names_season_types_of_athletes_without_reg():
    att = _attempt_table(n_per_pos=3)                                     # athletes 1..6, two attempts each
    att = att.with_columns(pl.when(pl.col("nfl_id") == 6).then(False).otherwise(pl.col("S3_no_boundary")).alias("S3_no_boundary"))
    nfl = pl.DataFrame({"nfl_id": [1, 1, 2, 3, 3, 4], "season_type": ["REG", "PRE", "PRE", "PRE", "POST", "REG"],
                        "n_plays": [120, 30, 40, 5, 5, 60]})
    official = pl.DataFrame({"nfl_id": [1, 2, 3, 4, 5, 6, 7], "cmb_forty": [4.5, 4.6, None, 4.7, 4.8, 4.9, 5.0],
                             "cmb_ten_yd_split": [1.5, None, None, 1.6, 1.6, 1.6, 1.7]})
    cfg = RescueConfig(expected_counts=(("combine_results", 7), ("nfl_tracking_any", 9)), nfl_min_plays_grid=(100,))
    t = conform(co.cohort_reconciliation([1, 2, 3, 4, 5, 6, 7], nfl, att, official, cfg), co.RECON)
    n = {r["stage"].split("_", 1)[1]: r for r in t.iter_rows(named=True)}
    assert n["combine_results"]["n_athletes"] == 7 and n["combine_results"]["matches_expected"]
    assert n["nfl_tracking_any"]["n_athletes"] == 4 and n["nfl_tracking_any"]["matches_expected"] is False
    assert n["nfl_tracking_reg"]["n_athletes"] == 2 and n["nfl_tracking_reg"]["expected"] is None
    assert n["nfl_no_reg:PRE"]["n_athletes"] == 1 and n["nfl_no_reg:POST+PRE"]["n_athletes"] == 1
    assert n["nfl_reg_ge_100"]["n_athletes"] == 1 and n["forty_repeat"]["n_athletes"] == 6
    assert n["official_both"]["n_athletes"] == 5 and n["forty_tracked_and_official_both"]["n_athletes"] == 4
    assert n["scenario_S3_no_boundary_repeat"]["n_athletes"] == 5 and n["forty_repeat_and_reg_ge_100"]["n_athletes"] == 1
    assert t["stage"].is_sorted()


def test_endpoint_feasibility_census():
    rng = np.random.default_rng(6)
    rows = []
    for a in range(1, 13):
        skill = rng.normal(1.0, 0.15)
        for g in range(1, 5):
            for p in range(1, 9):
                rows.append({"game_id": 2023090000 + g, "play_id": p, "nfl_id": a, "season": 2023, "cohort": 2023 if a % 2 else 2022,
                             "position_group": "DL", "player_get_off": None if p == 8 else skill + rng.normal(0, 0.1),
                             "lined_up_position": "EDGE" if a % 3 else "INTERIOR_LINE", "pass_result": "C" if p < 8 else None})
    pop = pl.DataFrame(rows, schema_overrides={"player_get_off": pl.Float64})
    flags = pl.DataFrame({"nfl_id": list(range(1, 13)), "has_cmb_forty": [True] * 12, "has_cmb_ten_yd_split": [True] * 6 + [False] * 6,
                          "forty_tracked": [True] * 9 + [False] * 3, "forty_repeat": [True] * 5 + [False] * 7,
                          "model_fit_ok_repeat": [True] * 5 + [False] * 7})
    cfg = RescueConfig(n_boot=200, min_athletes=5, outcome_obs_grid=(20, 50), first_n_grid=(10,), game_min_obs=5,
                       endpoints=(("player_get_off", "DL"), ("separation_at_pass_forward", "WR")))
    t = conform(co.endpoint_table(pop, flags, {"player_get_off": {"meaning": "Timing metric", "unit": "seconds",
                                                                  "evidence": "inferred"}}, cfg), co.ENDPOINT)
    get = lambda sec, strat, met, ep="player_get_off": t.filter((pl.col("endpoint") == ep) & (pl.col("section") == sec)  # noqa: E731
                                                               & (pl.col("stratum") == strat) & (pl.col("metric") == met)).row(0, named=True)
    d = get("definition", "all", "vendor_definition_established")
    assert d["value"] == 0.0 and "inferred" in d["note"] and "unknown" in d["note"]
    assert get("missingness", "all", "share_non_null")["value"] == pytest.approx(7 / 8)
    assert get("missingness", "pass_result:null", "share_non_null")["value"] == 0.0
    assert get("athletes", "min_obs=20", "n_athletes")["value"] == 12 and get("athletes", "min_obs=50", "n_athletes")["value"] == 0
    assert get("intersection", "min_obs=20", "forty_repeat")["value"] == 5
    assert get("intersection", "min_obs=20", "forty_tracked_and_official_both")["value"] == 6
    icc = get("reliability", "play", "icc1")
    assert 0.3 < icc["value"] < 0.95 and icc["n"] == 12
    assert get("reliability", "play", "projected_reliability_n=20")["value"] > icc["value"]
    assert get("reliability", "game", "icc1")["value"] > icc["value"]     # a game mean averages 7 plays
    assert get("first_n", "n=10", "n_athletes_ge_n")["value"] == 12 and get("first_n", "n=10", "n_athletes_ge_n_rookie_season")["value"] == 6
    assert get("first_n", "n=10", "median_games_to_reach_n")["value"] == 2
    assert get("context", "all", "share_non_null:lined_up_position")["value"] == 1.0
    wr = t.filter(pl.col("endpoint") == "separation_at_pass_forward")
    assert wr.height == 2 and wr.filter(pl.col("section") == "missingness")["note"][0].startswith("endpoint column absent")


# ------------------------------------------------------------------------------ persistence, report, registries
def _store(tmp_path) -> RescueStore:
    paths = SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures", gold=tmp_path / "gold",
                            rel=lambda p: str(p))
    return RescueStore(SimpleNamespace(paths=paths, parquet=SimpleNamespace(polars_kwargs=lambda: {})))


def test_store_layout_contract_and_cache_reuse(tmp_path):
    store = _store(tmp_path)
    att = _attempt_table(n_per_pos=3)
    contracts = (at.ATTEMPTS,)
    fp = store.fingerprint("ATT", CFG.params(), {"files": {"a": 1}})
    assert store.is_current("ATT", fp, contracts) == (False, "never run")
    store.write(att, at.ATTEMPTS, "ATT", {"inputs": ["x"]})
    assert store.is_current("ATT", fp, contracts)[0] is False             # written but not recorded = not complete
    store.record("ATT", fp, contracts)
    assert store.is_current("ATT", fp, contracts) == (True, "current")
    assert (tmp_path / "tables" / "sprint_rescue" / "attempt_censoring_diagnostics.csv").is_file()
    assert (tmp_path / "gold" / "sprint_rescue" / "attempt_censoring_diagnostics.schema.json").is_file()
    assert store.read(at.ATTEMPTS).equals(att)
    changed = store.fingerprint("ATT", RescueConfig(n_boot=10).params(), {"files": {"a": 1}})
    assert store.is_current("ATT", changed, contracts)[1] == "parameters or upstream inputs changed"
    store.csv(at.ATTEMPTS).unlink()
    assert store.is_current("ATT", fp, contracts)[0] is False
    with pytest.raises(Exception):
        store.write(att.drop("first_speed"), at.ATTEMPTS, "ATT")           # contract violation is never written
    assert _store(tmp_path / "x").tables_dir != RescueStore(store.cfg, sampled=True).tables_dir


def test_report_never_shows_unmeasured_numbers(tmp_path):
    store = _store(tmp_path)
    tables = rp.Tables(store)
    text = rp.fill("A {{screening}} B {{endpoint_dl}} C {{not_a_token}} D {{run_status}}", tables)
    assert text.count("_Pending:") == 2 and "--only D" in text and "--only F" in text and "Unknown report token" in text
    assert text.count("MISSING") == len(ALL_CONTRACTS)
    att = _attempt_table(n_per_pos=12)
    specs = feature_specs(CFG)
    rel = rl.reliability_table(att, specs, CFG)
    red = rd.redundancy_table(rd.athlete_table(att, CFG), specs, CFG)
    store.write(rd.screening_summary(specs, rel, red, {}, CFG), rd.SCREEN, "D")
    filled = rp.fill("{{screening}}\n{{decision}}", rp.Tables(store))
    assert "_Pending:" not in filled and "Rule-based proposal" in filled and "accel_sustained" in filled
    template = Path(__file__).resolve().parents[1] / "reports" / rp.TEMPLATE
    tokens = set(re.findall(r"\{\{(\w+)\}\}", template.read_text(encoding="utf-8")))
    assert tokens and tokens <= set(rp.SECTIONS)


def test_registries_are_consistent():
    names = [c.name for c in ALL_CONTRACTS]
    assert len(names) == len(set(names))
    for required in ("attempt_censoring_diagnostics", "within_athlete_window_variation", "within_position_reliability",
                     "sprint_model_fit_diagnostics", "sprint_model_repeatability", "combine_timing_redundancy",
                     "feature_screening_summary", "athlete_cohort_reconciliation", "dl_getoff_feasibility"):
        assert required in names and next(c for c in ALL_CONTRACTS if c.name == required).kind == "table"
    assert all(set(deps) <= set(STAGES) for _, deps in STAGES.values())
    figs = [n for n, _, _ in FIGURES]
    assert len(figs) == len(set(figs)) and 6 <= len(figs) <= 10
    assert {n for _, needs, _ in FIGURES for n in needs} <= set(names)
    spec_names = [s.name for s in feature_specs()]
    assert len(spec_names) == len(set(spec_names)) and set(spec_names) <= set(at.ATTEMPTS.columns)
    assert {"peak_speed", "peak_speed_interior", "p95_speed", "peak_a_tan", "p95_a_tan", "p95_decel", "path_smooth_yds",
            "model_vmax", "model_tau_s", "model_a0"} <= set(spec_names)
