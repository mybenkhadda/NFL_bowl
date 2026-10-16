"""Phase 03 · M1 — kinematic foundation (analytic signals + synthetic warehouse).

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_phase03_kinematics.py -q
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from nflbdb.features.config import KinematicsConfig, SmoothingSpec
from nflbdb.features.events import detect_events
from nflbdb.features.feature_registry import CONTEXT_DOCS, registered_names, registry_frame
from nflbdb.features.kinematics import compute_kinematics
from nflbdb.features.signal_processing import savgol_matrix, segmented_savgol, wrap_degrees
from nflbdb.features.trajectory import NFL, prepare_frames
from nflbdb.research.kinematic_build import KinematicFeatureBuilder, run_pipeline
from nflbdb.research.kinematic_diagnostics import vendor_agreement

T0 = datetime(2024, 1, 7, 13, 0, 0)
DT = 0.1
KCFG = KinematicsConfig()


# ----------------------------------------------------------------------------- helpers
def traj(xs, ys, game=1, play=1, nfl=7, offsets_ms=None, s=None, direction=None) -> pl.DataFrame:
    n = len(xs)
    offs = offsets_ms if offsets_ms is not None else [100 * k for k in range(n)]
    return pl.DataFrame({
        "season": [2023] * n, "game_id": [game] * n, "play_id": [play] * n, "nfl_id": [nfl] * n,
        "frame_id": list(range(1, n + 1)), "ts": [T0 + timedelta(milliseconds=o) for o in offs],
        "x": [float(v) for v in xs], "y": [float(v) for v in ys],
        "s": pl.Series(s if s is not None else [None] * n, dtype=pl.Float64),
        "a": pl.Series([None] * n, dtype=pl.Float64),
        "dir": pl.Series(direction if direction is not None else [None] * n, dtype=pl.Float64),
        "o": pl.Series([None] * n, dtype=pl.Float64),
    })


def from_speed(v: np.ndarray) -> np.ndarray:
    """1-D positions from a speed profile (trapezoidal integration, Δt = 0.1 s)."""
    return np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * DT)])


def kin_of(df: pl.DataFrame, kcfg: KinematicsConfig = KCFG) -> pl.DataFrame:
    frames, _ = prepare_frames(df, NFL, kcfg)
    return compute_kinematics(frames, kcfg)


# ----------------------------------------------------------------------------- signal processing
def test_savgol_centre_kernel_matches_published_coefficients():
    m = savgol_matrix(5, 2, 0)
    assert np.allclose(m[2], np.array([-3, 12, 17, 12, -3]) / 35)


def test_savgol_exact_for_quadratic_including_edges():
    t = np.arange(20) * DT
    x = 1 + 2 * t + 3 * t**2
    pos, L = np.arange(20), np.full(20, 20)
    spec = SmoothingSpec(7, 2)
    assert np.allclose(segmented_savgol(x, pos, L, spec, 0, DT), x)
    assert np.allclose(segmented_savgol(x, pos, L, spec, 1, DT), 2 + 6 * t)
    assert np.allclose(segmented_savgol(x, pos, L, spec, 2, DT), 6.0)


def test_segments_never_mix():
    a, b = np.linspace(0, 5, 12), 100 + np.linspace(0, -3, 9) ** 2
    pos = np.concatenate([np.arange(12), np.arange(9)])
    L = np.concatenate([np.full(12, 12), np.full(9, 9)])
    spec = SmoothingSpec(7, 2)
    joint = segmented_savgol(np.concatenate([a, b]), pos, L, spec, 1, DT)
    sep_a = segmented_savgol(a, np.arange(12), np.full(12, 12), spec, 1, DT)
    sep_b = segmented_savgol(b, np.arange(9), np.full(9, 9), spec, 1, DT)
    assert np.allclose(joint, np.concatenate([sep_a, sep_b]))


def test_short_segment_is_undefined_and_causal_ignores_future():
    spec = SmoothingSpec(7, 2)
    assert np.isnan(segmented_savgol(np.arange(5.0), np.arange(5), np.full(5, 5), spec)).all()
    x = np.sin(np.arange(30) * 0.2)
    pos, L = np.arange(30), np.full(30, 30)
    c1 = segmented_savgol(x, pos, L, spec, 1, DT, causal=True)
    x2 = x.copy()
    x2[20:] += 50.0
    c2 = segmented_savgol(x2, pos, L, spec, 1, DT, causal=True)
    assert np.isnan(c1[:6]).all()
    assert np.allclose(c1[:20], c2[:20], equal_nan=True)


def test_wrap_degrees():
    assert wrap_degrees(359.0 - 1.0) == pytest.approx(-2.0)
    assert wrap_degrees(1.0 - 359.0) == pytest.approx(2.0)


# ----------------------------------------------------------------------------- segmentation
def test_prepare_frames_splits_on_dup_gap_jump_and_keeps_trajectories_apart():
    xs = [0.5 * k for k in range(24)]
    offs = [100 * k for k in range(24)]
    xs_bad = list(xs)
    offs_bad = list(offs)
    offs_bad.insert(4, offs_bad[3]); xs_bad.insert(4, xs_bad[3])           # duplicate timestamp
    del offs_bad[12]; del xs_bad[12]                                          # missing frame → gap
    xs_bad = [v + (5.0 if i >= 18 else 0.0) for i, v in enumerate(xs_bad)]  # coordinate jump
    df = pl.concat([traj(xs_bad, [10.0] * len(xs_bad), nfl=7, offsets_ms=offs_bad),
                    traj(xs, [20.0] * 24, nfl=8), traj(xs, [20.0] * 24, play=2, nfl=7)])
    frames, stats = prepare_frames(df, NFL, KCFG)
    s7 = stats.filter((pl.col("nfl_id") == 7) & (pl.col("play_id") == 1)).row(0, named=True)
    assert (s7["n_dup_ts_removed"], s7["n_gap_breaks"], s7["n_jump_breaks"], s7["n_segments"]) == (1, 1, 1, 3)
    assert stats.filter(pl.col("nfl_id") == 8)["n_segments"].item() == 1
    assert stats.filter(pl.col("play_id") == 2)["n_segments"].item() == 1
    reasons = frames.filter((pl.col("nfl_id") == 7) & (pl.col("play_id") == 1))["break_reason"].drop_nulls().to_list()
    assert reasons == ["start", "gap", "jump"]
    assert frames.filter(pl.col("break_reason").is_not_null())["dt_ms"].null_count() == frames.filter(
        pl.col("break_reason").is_not_null()).height


# ----------------------------------------------------------------------------- kinematics
def test_circle_constant_speed_and_heading_wraparound():
    R, v = 10.0, 5.0
    w = v / R
    th = np.arange(160) * w * DT                       # > 1 revolution → heading crosses 360/0
    # interior frames only: edge rows evaluate the fit off-centre, which rotates the (constant) fitted
    # acceleration vector by up to 3·ω·Δt — an expected SG boundary property, not a bug
    k = kin_of(traj(R * np.cos(th), R * np.sin(th))).filter(
        pl.col("valid_kin") & (pl.col("pos") >= 3) & (pl.col("pos") < pl.col("seg_len") - 3))
    assert k["speed"].to_numpy() == pytest.approx(v, rel=0.01)
    assert np.abs(k["a_tan"].to_numpy()).max() < 0.05
    assert k["a_lat"].to_numpy() == pytest.approx(v**2 / R, rel=0.02)          # CCW → positive
    assert k["yaw_rate"].to_numpy() == pytest.approx(math.degrees(w), rel=0.02)
    assert k["curvature"].to_numpy() == pytest.approx(1 / R, rel=0.02)
    assert k["accel_mag"].to_numpy() == pytest.approx(v**2 / R, rel=0.02)
    assert k["heading"].min() < 10 and k["heading"].max() > 350                 # wraps, yaw stays smooth


def test_stationary_jitter_has_no_heading():
    rng = np.random.default_rng(0)
    k = kin_of(traj(50 + rng.normal(0, 0.02, 30), 20 + rng.normal(0, 0.02, 30)))
    assert k["heading"].null_count() == k.height
    assert k["yaw_rate"].null_count() == k.height


# ----------------------------------------------------------------------------- events
def _braking_profile(decel: float) -> np.ndarray:
    up = np.linspace(0, 6, 21)
    hold = np.full(10, 6.0)
    n_down = int(round((6 - 1) / decel / DT))
    down = np.linspace(6, 1, n_down + 1)[1:]
    reacc = np.linspace(1, 5, 15)[1:]
    return np.concatenate([up, hold, down, reacc])


def test_braking_event_and_reacceleration():
    v = _braking_profile(4.0)
    k = kin_of(traj(from_speed(v), np.zeros(v.size)))
    ev = detect_events(k, NFL, KCFG, event_types=("braking",), rules=("default",))
    assert ev.height == 1
    e = ev.row(0, named=True)
    assert e["entry_speed"] > 5.5 and e["speed_change"] > 3.5 and e["peak_abs_a_tan"] > 3.0
    assert e["time_to_transition_s"] is not None and e["time_to_transition_s"] < 1.0


def test_mild_slowdown_is_not_braking():
    v = np.concatenate([np.full(15, 6.0), np.linspace(6, 5, 11)[1:], np.full(15, 5.0)])   # −1 yd/s²
    k = kin_of(traj(from_speed(v), np.zeros(v.size)))
    assert detect_events(k, NFL, KCFG, event_types=("braking",)).height == 0


def test_shuttle_reversal_detected_and_not_a_turn():
    v = np.concatenate([np.full(15, 5.0), np.linspace(5, -5, 21)[1:], np.full(15, -5.0)])  # signed velocity
    k = kin_of(traj(from_speed(v), np.full(v.size, 3.0)))
    ev = detect_events(k, NFL, KCFG, event_types=("reversal", "turn"), rules=("default",))
    rev = ev.filter(pl.col("event_type") == "reversal")
    assert rev.height == 1 and abs(rev["heading_change_deg"].item()) == pytest.approx(180, abs=5)
    assert ev.filter(pl.col("event_type") == "turn").height == 0


def test_turn_event_on_tight_circle():
    R, v = 3.0, 5.0
    th = np.arange(30) * (v / R) * DT
    k = kin_of(traj(R * np.cos(th), R * np.sin(th)))
    turns = detect_events(k, NFL, KCFG, event_types=("turn",), rules=("default",))
    assert turns.height == 1
    t = turns.row(0, named=True)
    assert t["direction"] == "ccw"
    assert t["median_radius_yds"] == pytest.approx(R, rel=0.05)
    assert t["heading_change_deg"] == pytest.approx(math.degrees(v / R) * t["duration_s"], rel=0.05)


# ----------------------------------------------------------------------------- quality tiers
def test_quality_tiers_split_and_implausible():
    xs = [0.5 * k + (3.0 if k >= 10 else 0.0) for k in range(20)]                 # jump → 2 segments
    fast = [1.4 * k for k in range(20)]                                           # 14 yd/s > 12.5
    out = run_pipeline(pl.concat([traj(xs, [0.0] * 20, nfl=1), traj(fast, [0.0] * 20, nfl=2)]), NFL, KCFG)
    tiers = dict(out["trajectories"].select("nfl_id", "quality_tier").iter_rows())
    assert tiers == {1: "B", 2: "reject"}


# ----------------------------------------------------------------------------- registry
def test_every_output_column_is_documented():
    R, v = 3.0, 5.0
    th = np.arange(40) * (v / R) * DT
    out = run_pipeline(traj(R * np.cos(th), R * np.sin(th), s=[v] * 40), NFL, KCFG)
    known = registered_names() | set(CONTEXT_DOCS)
    for name, df in out.items():
        missing = [c for c in df.columns if c not in known]
        assert not missing, f"{name}: undocumented columns {missing}"
    reg = registry_frame()
    assert reg.filter(pl.col("unit") == "").height == 0
    assert set(reg["status"].unique()) <= {"core", "requires_validation", "experimental"}


def test_vendor_dir_convention_is_inferred_not_assumed():
    R, v = 10.0, 5.0
    th = np.arange(80) * (v / R) * DT
    heading_math = (np.degrees(th) + 90) % 360
    compass = (450 - heading_math) % 360
    k = kin_of(traj(R * np.cos(th), R * np.sin(th), s=[v] * 80, direction=list(compass)))
    va = vendor_agreement(k.with_columns(pl.lit(0.0).alias("a")), "test")
    assert va["vendor_dir_convention_best_fit"].item().startswith("compass")
    assert va["median_dir_diff_compass_deg"].item() < 3


# ----------------------------------------------------------------------------- warehouse integration
def _silver(synth):
    from nflbdb.data.bronze import BronzeIngestor
    from nflbdb.data.silver import SilverBuilder

    BronzeIngestor(synth).ingest()
    SilverBuilder(synth).build_all()


def test_combine_build_reuse_and_param_invalidation(synth):
    _silver(synth)
    b = KinematicFeatureBuilder(synth)
    out = b.build_combine()
    frames = out["frames"].collect()
    assert frames.height == 2 * 2 * 8                       # 2 athletes × 2 attempts × 8 frames
    assert frames["speed"].drop_nulls().to_numpy() == pytest.approx(8.0, rel=0.02)
    assert out["trajectories"].collect()["quality_tier"].to_list() == ["A"] * 4

    reuse = KinematicFeatureBuilder(synth)
    reuse.load_combine_raw = lambda: (_ for _ in ()).throw(AssertionError("should reuse persisted data"))
    reuse.build_combine()

    changed = KinematicFeatureBuilder(synth, KinematicsConfig(smoothing=SmoothingSpec(5, 2)))
    from nflbdb.research.kinematic_build import COMBINE_SEGMENTS
    current, reason = changed.store.status(COMBINE_SEGMENTS, changed.params("combine"))
    assert not current and "parameters" in reason


def test_nfl_build_flags_short_gappy_trajectory(synth):
    _silver(synth)
    out = KinematicFeatureBuilder(synth).build_nfl()
    tq = out["trajectories"].collect()
    assert tq.height == 5                                    # player-plays in conftest (football excluded)
    assert tq.filter(pl.col("nfl_id") == 103)["quality_tier"].item() == "reject"   # gap → 5 + 4 frames < window
    assert tq.filter(pl.col("nfl_id") == 103)["n_gap_breaks"].item() == 1
