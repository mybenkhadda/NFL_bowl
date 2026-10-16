"""M1.1 · signal integrity (features.signal_integrity) on synthetic NFL-shaped trajectories.

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_m1_1_signal_integrity.py -q
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from nflbdb.features.config import KinematicsConfig, SmoothingSpec
from nflbdb.features.kinematics import compute_kinematics
from nflbdb.features.signal_integrity import (add_signal_columns, edge_distance, jerk_magnitude,
                                              trajectory_signal_metrics)
from nflbdb.features.trajectory import NFL, prepare_frames

K = KinematicsConfig()
KEYS = list(NFL.keys)
T0 = datetime(2024, 9, 8, 13, 0, 0)


def raw(times_s, xs, ys=None, s=None, play: int = 1, nfl: int = 10) -> pl.DataFrame:
    n = len(times_s)
    ys = np.zeros(n) if ys is None else np.asarray(ys, float)
    xs = np.asarray(xs, float)
    if s is None:   # vendor speed = finite-difference speed (first frame copies the second)
        d = np.r_[np.nan, np.hypot(np.diff(xs), np.diff(ys)) / np.diff(times_s)]
        d[0] = d[1] if n > 1 else 0.0
        s = d
    return pl.DataFrame({"game_id": [1] * n, "play_id": [play] * n, "nfl_id": [nfl] * n, "season": [2024] * n,
                         "frame_id": list(range(1, n + 1)),
                         "ts": [T0 + timedelta(milliseconds=round(t * 1000)) for t in times_s],
                         "x": xs, "y": ys, "s": np.asarray(s, float), "a": np.zeros(n), "dir": np.zeros(n),
                         "o": np.zeros(n)})


def kin(df: pl.DataFrame, k: KinematicsConfig = K) -> pl.DataFrame:
    frames, _ = prepare_frames(df, NFL, k)
    return compute_kinematics(frames, k)


# ----------------------------------------------------------------------------- frame level
def test_edge_distance_and_boundary_flags():
    assert edge_distance(np.arange(5), np.full(5, 5)).tolist() == [0, 1, 2, 1, 0]
    t = np.arange(20) * 0.1
    f = add_signal_columns(kin(raw(t, 5.0 * t)), KEYS, K)          # sg7 → half-window 3
    b = f.sort("pos")["in_filter_boundary"].to_list()
    assert b[:3] == [True] * 3 and b[-3:] == [True] * 3 and not any(b[3:-3])


def test_jerk_never_crosses_a_segment_boundary():
    pos = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    seg_len = np.full(8, 4)
    ax = np.array([0, 0, 0, 0, 100, 100, 100, 100], float)       # a jump in acceleration BETWEEN segments
    j = jerk_magnitude(pos, seg_len, ax, np.zeros(8), 0.1)
    assert np.isnan(j[[0, 3, 4, 7]]).all()                         # segment edges: undefined
    assert np.allclose(j[[1, 2, 5, 6]], 0.0)                       # interior: no spurious jerk from the other segment


def test_constant_velocity_is_smooth_and_agrees_with_vendor():
    t = np.arange(40) * 0.1
    m = trajectory_signal_metrics(kin(raw(t, 6.0 * t, s=np.full(40, 6.0))), KEYS, K).row(0, named=True)
    assert m["jerk_rms_interior"] == pytest.approx(0.0, abs=1e-6)
    assert m["vendor_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["n_missing_frames"] == 0 and m["missing_frame_pct"] == 0.0 and m["n_segments"] == 1
    assert m["usable_frac"] == 1.0


def test_gap_counts_missing_frames_and_splits_segments():
    t = np.r_[np.arange(15) * 0.1, 1.8 + np.arange(15) * 0.1]     # Δt = 0.4 s once → 3 missing frames
    m = trajectory_signal_metrics(kin(raw(t, 5.0 * t)), KEYS, K).row(0, named=True)
    assert m["n_gap_breaks"] == 1 and m["n_segments"] == 2
    assert m["n_missing_frames"] == 3
    assert m["missing_frame_pct"] == pytest.approx(3 / 33)


def test_irregular_sampling_inside_tolerance_is_measured_not_split():
    t = np.cumsum(np.r_[0.0, np.tile([0.1, 0.13], 15)])           # +30 ms jitter, below gap_factor × nominal
    m = trajectory_signal_metrics(kin(raw(t, 5.0 * t)), KEYS, K).row(0, named=True)
    assert m["n_segments"] == 1
    assert m["dt_cv"] > 0 and m["dt_max_dev_ms"] == pytest.approx(30.0, abs=1.0)
    assert m["n_offgrid"] == 15                                    # |Δt − 100| = 30 > 0.25 × 100


def test_low_speed_heading_is_null_exactly_below_threshold():
    t = np.arange(40) * 0.1
    x = np.where(t < 2.0, 0.0, 4.0 * (t - 2.0))                    # stationary 2 s, then 4 yd/s
    f = kin(raw(t, x))
    m = trajectory_signal_metrics(f, KEYS, K).row(0, named=True)
    assert 0 < m["lowspeed_frac"] < 1
    assert m["heading_null_frac"] == pytest.approx(m["lowspeed_frac"])
    v = f.filter(pl.col("valid_kin"))
    assert v.filter(pl.col("speed") < K.min_heading_speed)["heading"].null_count() == \
        v.filter(pl.col("speed") < K.min_heading_speed).height


def test_peak_at_the_recording_edge_is_flagged_and_interior_peak_reported():
    t = np.arange(30) * 0.1
    v = 1.0 + 0.3 * np.arange(30)                                  # accelerating to the last frame (forty-like)
    x = np.r_[0.0, np.cumsum(v[:-1] * 0.1)]
    m = trajectory_signal_metrics(kin(raw(t, x)), KEYS, K).row(0, named=True)
    assert m["peak_speed_in_boundary"] is True
    assert m["peak_speed_interior"] < m["peak_speed"]
    assert m["peak_speed_boundary_rel"] > 0
    assert m["boundary_frac"] == pytest.approx(6 / 30)


def test_short_segment_has_no_derivatives_and_no_peaks():
    t = np.arange(5) * 0.1                                         # shorter than the SG window (7)
    m = trajectory_signal_metrics(kin(raw(t, 5.0 * t)), KEYS, K).row(0, named=True)
    assert m["n_valid_kin"] == 0 and m["usable_frac"] == 0.0
    assert m["peak_speed"] is None and m["peak_speed_in_boundary"] is None


def test_boundary_width_follows_the_smoothing_spec():
    t = np.arange(30) * 0.1
    f = kin(raw(t, 5.0 * t), K.with_smoothing(SmoothingSpec(11, 3)))
    f = add_signal_columns(f, KEYS, K, SmoothingSpec(11, 3))
    assert f["in_filter_boundary"].sum() == 10                     # 5 frames at each edge


def test_unsorted_input_is_restored_before_differencing():
    t = np.arange(20) * 0.1
    f = kin(raw(t, 5.0 * t))
    shuffled = f.sample(fraction=1.0, shuffle=True, seed=3)
    a = trajectory_signal_metrics(f, KEYS, K)
    b = trajectory_signal_metrics(shuffled, KEYS, K)
    assert a.equals(b)
