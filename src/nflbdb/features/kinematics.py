"""Frame-level kinematics from smoothed positions.

Definitions (all in the tracking coordinate frame; yards, seconds):

    x̂, ŷ        SG-smoothed position
    v = (vx, vy)  SG first derivative of x, y                          [yd/s]
    speed = |v|   derived speed (compare with vendor ``s`` for QC)    [yd/s]
    a = (ax, ay)  SG second derivative of x, y — VECTOR acceleration   [yd/s²]
    accel_mag = |a|  magnitude of the vector acceleration; ≥ 0, includes turning
    a_tan = v·a / |v|      = d|v|/dt, the SCALAR speed derivative (signed: < 0 = braking)
    a_lat = (v × a) / |v|  lateral (centripetal) acceleration, signed (+ = counter-clockwise)
    yaw_rate = a_lat / |v| = dθ/dt  heading change rate                [deg/s, signed]
    curvature = yaw_rate / |v|  (path curvature, 1 / turning radius)    [1/yd, signed]
    heading = atan2(vy, vx)  mathematical convention (0° = +x, counter-clockwise) [deg]

Notes
* ``accel_mag`` ≥ |a_tan|; they are different quantities: a sprinter at constant speed around a
  curve has a_tan = 0 but accel_mag > 0. The vendor ``a`` column is a non-negative magnitude
  whose exact definition is unconfirmed — never treat it as braking (sign) information.
* Yaw rate uses the cross product, so heading wraparound (359° → 1°) never occurs.
* ``a_tan``/``a_lat`` are undefined below ``min_tangent_speed``; heading, yaw rate and curvature
  below ``min_heading_speed`` (direction of a nearly stationary athlete is dominated by noise).
* Heading follows the math convention. The vendor ``dir`` convention is NOT assumed; the M1
  diagnostics test both candidate conventions against data (``kinematics_vendor_agreement.csv``).
* No force, power or work is derived: that would require validated mass/drag assumptions.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import KinematicsConfig, SmoothingSpec
from .signal_processing import segment_index, segmented_savgol

DERIVED_COLUMNS = ("x_s", "y_s", "vx", "vy", "speed", "ax", "ay", "accel_mag", "a_tan", "a_lat",
                   "yaw_rate", "curvature", "heading", "step_smooth", "pos_resid", "valid_kin", "heading_valid")


def compute_kinematics(frames: pl.DataFrame, kcfg: KinematicsConfig, spec: SmoothingSpec | None = None,
                       causal: bool = False) -> pl.DataFrame:
    """Append derived kinematic columns to segmented frames from ``trajectory.prepare_frames``.

    Frames must be sorted by (trajectory keys, ts) with contiguous segments; ``pos``/``seg_len``
    come from the segmentation. Sampling inside a segment is treated as uniform at the nominal
    interval (gaps > ``gap_factor`` × nominal already split segments; off-grid jitter is counted
    in the segment quality table).
    """
    spec = spec or kcfg.smoothing
    dt = kcfg.dt_s
    pos = frames["pos"].to_numpy()
    seg_len = frames["seg_len"].to_numpy()
    x = frames["x"].cast(pl.Float64).to_numpy()
    y = frames["y"].cast(pl.Float64).to_numpy()
    index = segment_index(pos, seg_len, spec.window, causal)

    def sg(v: np.ndarray, deriv: int) -> np.ndarray:
        return segmented_savgol(v, pos, seg_len, spec, deriv=deriv, delta=dt, causal=causal, index=index)

    xs, ys = sg(x, 0), sg(y, 0)
    vx, vy = sg(x, 1), sg(y, 1)
    ax, ay = sg(x, 2), sg(y, 2)
    speed = np.hypot(vx, vy)
    accel_mag = np.hypot(ax, ay)
    with np.errstate(divide="ignore", invalid="ignore"):
        moving = speed >= kcfg.min_tangent_speed
        a_tan = np.where(moving, (vx * ax + vy * ay) / speed, np.nan)
        a_lat = np.where(moving, (vx * ay - vy * ax) / speed, np.nan)
        hv = speed >= kcfg.min_heading_speed
        omega = np.where(hv, a_lat / speed, np.nan)                      # rad/s
        curvature = np.where(hv, omega / speed, np.nan)                  # 1/yd
        heading = np.where(hv, np.degrees(np.arctan2(vy, vx)) % 360.0, np.nan)

    # Smoothed path increment from the previous frame of the SAME segment (pos > 0).
    step_smooth = np.full(x.size, np.nan)
    has_prev = np.asarray(pos) > 0
    j = np.nonzero(has_prev)[0]
    step_smooth[j] = np.hypot(xs[j] - xs[j - 1], ys[j] - ys[j - 1])
    pos_resid = np.hypot(x - xs, y - ys)
    valid_kin = ~np.isnan(speed)

    def col(name: str, arr: np.ndarray) -> pl.Series:
        return pl.Series(name, arr, dtype=pl.Float64, nan_to_null=True)

    return frames.with_columns(
        col("x_s", xs), col("y_s", ys), col("vx", vx), col("vy", vy), col("speed", speed),
        col("ax", ax), col("ay", ay), col("accel_mag", accel_mag), col("a_tan", a_tan), col("a_lat", a_lat),
        col("yaw_rate", np.degrees(omega)), col("curvature", curvature), col("heading", heading),
        col("step_smooth", step_smooth), col("pos_resid", pos_resid),
        pl.Series("valid_kin", valid_kin), pl.Series("heading_valid", valid_kin & hv),
    )


def play_kinematics(raw: pl.DataFrame, domain, kcfg: KinematicsConfig, spec: SmoothingSpec | None = None
                    ) -> pl.DataFrame:
    """Convenience for on-demand use (one play / attempt for plotting): segment + kinematics."""
    from .trajectory import prepare_frames

    frames, _ = prepare_frames(raw, domain, kcfg)
    return compute_kinematics(frames, kcfg, spec)
