"""Phase 03 · M2 — Combine movement features (analytic signals + synthetic warehouse).

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_phase03_m2_combine.py -q
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from nflbdb.features.agility import cod_events, cod_window_measures
from nflbdb.features.config import KinematicsConfig, MovementConfig
from nflbdb.features.efficiency import path_features
from nflbdb.features.events import detect_events
from nflbdb.features.feature_registry import CONTEXT_DOCS, registered_names
from nflbdb.features.kinematics import compute_kinematics
from nflbdb.features.reliability import icc_bootstrap, icc_oneway, reliability_label, reliability_table
from nflbdb.features.sprint import fit_mono_exponential, model_distance, sprint_profile, _grid
from nflbdb.features.temporal import active_frames, movement_window, temporal_features
from nflbdb.features.trajectory import COMBINE, prepare_frames

T0 = datetime(2024, 3, 1, 10, 0, 0)
DT = 0.1
KCFG = KinematicsConfig()
MCFG = MovementConfig()
KEYS = list(COMBINE.keys)


# ----------------------------------------------------------------------------- helpers
def ctraj(xs, ys, nfl=1, attempt=1, drill="FORTY_YARD_DASH", event="E1") -> pl.DataFrame:
    n = len(xs)
    return pl.DataFrame({
        "draft_year": [2024] * n, "event_id": [event] * n, "nfl_id": [nfl] * n, "attempt": [attempt] * n,
        "drill_type": [drill] * n, "drill_name": [drill] * n, "frame_id": list(range(1, n + 1)),
        "ts": [T0 + timedelta(minutes=attempt, milliseconds=100 * k) for k in range(n)],
        "x": [float(v) for v in xs], "y": [float(v) for v in ys],
        "s": pl.Series([None] * n, dtype=pl.Float64), "a": pl.Series([None] * n, dtype=pl.Float64),
        "dir": pl.Series([None] * n, dtype=pl.Float64)})


def kin(df: pl.DataFrame) -> pl.DataFrame:
    frames, _ = prepare_frames(df, COMBINE, KCFG)
    return compute_kinematics(frames, KCFG)


def from_speed(v: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * DT)])


def mono(t, vmax=10.0, tau=1.2, t0=-0.25):
    return vmax * (1 - np.exp(-np.clip(t - t0, 0, None) / tau))


# ----------------------------------------------------------------------------- sprint model
def test_mono_exponential_fit_recovers_parameters():
    t = np.arange(0, 5.0, DT)
    fit = fit_mono_exponential(t, mono(t), _grid(MCFG.tau_grid), t[0] + _grid(MCFG.t0_offset_grid))
    assert fit["tau"] == pytest.approx(1.2, abs=0.02)
    assert fit["vmax"] == pytest.approx(10.0, abs=0.05)
    assert fit["t0"] == pytest.approx(-0.25, abs=0.03)
    assert fit["r2"] > 0.999 and not fit["at_edge"]


def test_sprint_splits_match_analytic_distance_including_censored_start():
    vmax, tau, t0 = 10.0, 1.2, -0.25
    t = np.arange(0, 5.6, DT)                               # starts already moving (censored)
    x = model_distance(t, vmax, tau, t0) - model_distance(0.0, vmax, tau, t0)
    prof = sprint_profile(t, x, np.zeros_like(t), mono(t, vmax, tau, t0), MCFG)
    fine = np.arange(0, 8, 1e-4)
    d = model_distance(fine + t0, vmax, tau, t0)
    for dist, col in ((10, "trk_t10_s"), (40, "trk_t40_s")):
        assert prof[col] == pytest.approx(fine[np.argmax(d >= dist)], abs=0.03)
    assert prof["pre_distance_yds"] == pytest.approx(model_distance(0.0, vmax, tau, t0), abs=0.05)
    assert prof["fit_ok"] and not prof["run_incomplete"]


def test_short_run_marks_split_missing():
    t = np.arange(0, 2.0, DT)
    x = model_distance(t, 10.0, 1.2, -0.25)
    prof = sprint_profile(t, x, np.zeros_like(t), mono(t), MCFG)
    assert prof["run_incomplete"] and np.isnan(prof["trk_t40_s"])


# ----------------------------------------------------------------------------- reliability
def test_icc_oneway_hand_computed():
    vals = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    grp = np.array([0, 0, 1, 1, 2, 2])
    assert icc_oneway(vals, grp)["icc"] == pytest.approx(7.5 / 8.5)


def test_icc_perfect_and_bootstrap_deterministic():
    rng = np.random.default_rng(1)
    means = rng.normal(0, 1, 30)
    vals = np.repeat(means, 2) + rng.normal(0, 0.05, 60)
    grp = np.repeat(np.arange(30), 2)
    icc = icc_oneway(vals, grp)["icc"]
    lo, hi = icc_bootstrap(vals, grp, 500, 7)
    assert icc > 0.98 and lo <= icc <= hi
    assert (lo, hi) == icc_bootstrap(vals, grp, 500, 7)
    assert reliability_label(0.95) == "excellent" and reliability_label(float("nan")) == "undetermined"


def test_reliability_table_end_to_end():
    rng = np.random.default_rng(2)
    rows = []
    for a in range(15):
        mu = rng.normal(10, 1)
        for k in (1, 2, 3):
            rows.append({"drill": "D", "feature": "model_vmax", "nfl_id": a, "attempt_order": k,
                         "value": mu + rng.normal(0, 0.1)})
    rel = reliability_table(pl.DataFrame(rows), ["drill", "feature"], MovementConfig(n_boot=200))
    r = rel.row(0, named=True)
    assert r["n_athletes"] == 15 and r["icc1"] > 0.9 and r["spearman_first_two"] > 0.8
    assert r["median_best_minus_mean"] is None or r["median_best_minus_mean"] >= 0


# ----------------------------------------------------------------------------- window / temporal
def test_onset_and_censoring():
    v_rest = np.concatenate([np.zeros(10), np.linspace(0, 6, 31)])           # 1 s at rest, then accelerates
    v_moving = np.linspace(2, 6, 41)                                          # already moving at frame 0
    k = kin(pl.concat([ctraj(from_speed(v_rest), np.zeros(41), nfl=1),
                       ctraj(from_speed(v_moving), np.zeros(41), nfl=2)]))
    w = movement_window(k, KEYS, MCFG).sort("nfl_id")
    assert w["onset_censored"].to_list() == [False, True]
    # raw speed first reaches 0.6 ≥ onset_speed at frame 13 (t = 1.3 s); SG smoothing may shift it ±1–2 frames
    assert w.filter(pl.col("nfl_id") == 1)["t_onset_s"].item() == pytest.approx(1.3, abs=0.2)


def test_temporal_phases_accelerate_hold_brake():
    v = np.concatenate([np.linspace(0.6, 6, 20), np.full(15, 6.0), np.linspace(6, 0.6, 15)])
    k = kin(ctraj(from_speed(v), np.zeros(v.size)))
    w = movement_window(k, KEYS, MCFG)
    tf = temporal_features(active_frames(k, w, KEYS), KEYS, MCFG).row(0, named=True)
    assert tf["frac_time_accelerating"] > 0.2 and tf["frac_time_decelerating"] > 0.15
    assert tf["frac_time_steady"] > 0.15 and tf["n_phase_transitions"] >= 2
    assert tf["a_tan_mean_early"] > 0 > tf["a_tan_mean_late"]


# ----------------------------------------------------------------------------- COD / geometry
def test_reversal_becomes_cod_with_window_measures():
    v = np.concatenate([np.linspace(0, 5, 15), np.full(10, 5.0), np.linspace(5, -5, 21)[1:], np.full(15, -5.0)])
    k = kin(ctraj(from_speed(v), np.full(v.size, 3.0), drill="SHORT_SHUTTLE"))
    ev = detect_events(k, COMBINE, KCFG, rules=("default",))
    cod = cod_window_measures(k, cod_events(ev, KEYS, MCFG, ["drill"]), KEYS, MCFG)
    assert cod.height == 1
    c = cod.row(0, named=True)
    assert c["cod_type"] == "reversal" and c["cod_angle_deg"] == pytest.approx(180, abs=5)
    assert c["approach_speed"] == pytest.approx(5.0, abs=0.2)
    assert c["approach_decel"] > 2.0 and c["exit_accel"] > 2.0
    assert c["cod_time_s"] is not None and 0.5 < c["cod_time_s"] < 2.5


def test_path_efficiency_straight_vs_out_and_back():
    straight = ctraj(np.linspace(0, 20, 41), np.zeros(41), nfl=1)
    out_back = ctraj(np.concatenate([np.linspace(0, 10, 21), np.linspace(10, 0, 21)[1:]]), np.zeros(40), nfl=2)
    k = kin(pl.concat([straight, out_back]))
    w = movement_window(k, KEYS, MCFG)
    p = path_features(active_frames(k, w, KEYS), KEYS).sort("nfl_id")
    eff = p["path_efficiency"].to_list()
    assert eff[0] == pytest.approx(1.0, abs=0.02) and eff[1] < 0.2
    assert p["lateral_extent_yds"][0] == pytest.approx(0.0, abs=0.05)
    assert p["axis_anisotropy"][0] == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------------- builder (in memory + warehouse)
def _m2_inputs(df: pl.DataFrame):
    k = kin(df)
    ev = detect_events(k, COMBINE, KCFG, rules=("default",))
    usable = k.select(KEYS).unique().with_columns(pl.lit("A").alias("quality_tier"), pl.lit(1, dtype=pl.Int64)
                                                  .alias("n_segments"))
    return k, ev, usable


def test_every_m2_column_is_documented(synth):
    from nflbdb.research.combine_movement import CombineMovementBuilder

    t = np.arange(0, 5.6, DT)
    x = model_distance(t, 10.0, 1.2, -0.25) - model_distance(0.0, 10.0, 1.2, -0.25)
    v = np.concatenate([np.linspace(0, 5, 15), np.full(10, 5.0), np.linspace(5, -5, 21)[1:], np.full(15, -5.0)])
    df = pl.concat([ctraj(x, np.zeros(t.size), nfl=1, attempt=1), ctraj(x * 0.98, np.zeros(t.size), nfl=1, attempt=2),
                    ctraj(from_speed(v), np.full(v.size, 3.0), nfl=2, drill="SHORT_SHUTTLE")])
    out = CombineMovementBuilder(synth).compute_from(*_m2_inputs(df))
    known = registered_names() | set(CONTEXT_DOCS)
    for name, d in out.items():
        missing = [c for c in d.columns if c not in known]
        assert not missing, f"{name}: undocumented columns {missing}"
    att = out["attempts"]
    assert att.filter(pl.col("nfl_id") == 1)["attempt_order"].sort().to_list() == [1, 2]
    assert out["sprint"].height == 2 and out["cod"].height == 1
    ath = out["athlete"].filter((pl.col("nfl_id") == 1) & (pl.col("feature") == "model_vmax"))
    assert ath["n_attempts"].item() == 2 and ath["best"].item() == ath["max"].item()


def test_m2_build_reuse_and_param_invalidation(synth):
    from nflbdb.data.bronze import BronzeIngestor
    from nflbdb.data.silver import SilverBuilder
    from nflbdb.research.combine_movement import CombineMovementBuilder
    from nflbdb.research.kinematic_build import KinematicFeatureBuilder

    BronzeIngestor(synth).ingest()
    SilverBuilder(synth).build_all()
    KinematicFeatureBuilder(synth).build_combine()
    out = CombineMovementBuilder(synth).build()
    assert out["attempts"].select(pl.len()).collect().item() == 4      # 2 athletes × 2 attempts
    reuse = CombineMovementBuilder(synth)
    reuse.compute = lambda: (_ for _ in ()).throw(AssertionError("should reuse persisted data"))
    reuse.build()
    changed = CombineMovementBuilder(synth, mcfg=MovementConfig(onset_speed=0.6))
    current, reason = changed.store.status(changed.specs["attempts"], changed.params())
    assert not current and "parameters" in reason
