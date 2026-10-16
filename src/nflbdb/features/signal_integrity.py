"""Derivative-signal reliability metrics (Phase 03 · M1.1, Workstream A2/A3).

M1 quality tiers mix STRUCTURAL validity (is the trajectory intact?) with SIGNAL reliability (can
its derivatives be trusted?). This module measures the second, per frame and per trajectory, on
the frames produced by ``trajectory.prepare_frames`` + ``kinematics.compute_kinematics`` (the
persisted ``combine_frame_kinematics`` or an on-demand NFL batch). Nothing here changes M1 values.

Frame columns added by :func:`add_signal_columns`
    edge_dist           frames to the nearer segment edge, min(pos, seg_len − 1 − pos)
    in_filter_boundary  edge_dist < window // 2: the SG fit is NOT centred on this frame (the
                        polynomial of the first / last full window is evaluated off-centre, as in
                        ``scipy.signal.savgol_filter(mode="interp")``). Off-centre rows of the SG
                        derivative matrix have larger noise gain, most of all for the 2nd derivative.
    jerk_mag            |Δa| / Δt by central difference of the SG acceleration vector, only where
                        both neighbours are in the SAME segment (pos ≥ 1 and pos ≤ seg_len − 2);
                        [yd/s³]. With polyorder 2 the SG 3rd derivative is identically 0, so jerk
                        is a roughness diagnostic of the acceleration signal, not a physical claim.
    dt_dev_ms           |Δt − nominal| inside a segment (null at segment starts)

Trajectory metrics (:func:`trajectory_signal_metrics`), one row per trajectory key
    sampling     dt_cv, dt_max_dev_ms, offgrid_rate, n_missing_frames, missing_frame_pct
    structure    n_segments, n_gap_breaks, n_jump_breaks, usable_frac (frames with derivatives)
    vendor       vendor_rmse, vendor_mae, vendor_bias, vendor_p95_abs (derived − vendor speed)
    roughness    jerk_rms_interior, jerk_rms_all, jerk_p95_interior
    heading      lowspeed_frac (speed < min_heading_speed), heading_null_frac — equal by
                 construction; a difference indicates a heading-validity bug
    boundary     boundary_frac (share of valid frames in the filter boundary), and for every peak
                 feature: ``<peak>``, ``<peak>_interior`` (boundary frames excluded),
                 ``<peak>_in_boundary`` (a frame attaining the peak lies in the boundary; ties →
                 True if any tied frame does) and ``<peak>_boundary_rel`` = (all − interior) / all.

Interpretation (to be verified on data, see Study A3): if peaks fall inside boundary frames much
more often than ``boundary_frac`` predicts, peak features are partly edge artefacts.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import KinematicsConfig, SmoothingSpec

# peak feature → expression on frame columns (higher = more extreme)
PEAK_FEATURES: dict[str, pl.Expr] = {
    "peak_speed": pl.col("speed"),
    "peak_a_tan": pl.col("a_tan"),
    "peak_decel": -pl.col("a_tan"),
    "peak_accel_mag": pl.col("accel_mag"),
    "peak_abs_yaw_rate": pl.col("yaw_rate").abs(),
}
REQUIRED_COLUMNS = ("seg_idx", "pos", "seg_len", "t_s", "dt_ms", "break_reason", "s", "speed", "ax", "ay",
                    "accel_mag", "a_tan", "yaw_rate", "heading", "valid_kin")


def half_window(spec: SmoothingSpec) -> int:
    return spec.window // 2


def edge_distance(pos: np.ndarray, seg_len: np.ndarray) -> np.ndarray:
    pos, seg_len = np.asarray(pos, dtype=np.int64), np.asarray(seg_len, dtype=np.int64)
    return np.minimum(pos, seg_len - 1 - pos)


def jerk_magnitude(pos: np.ndarray, seg_len: np.ndarray, ax: np.ndarray, ay: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference jerk magnitude inside segments; NaN at segment edges or next to NaN.

    Frames must be stored contiguously per segment in time order (the ``prepare_frames`` contract),
    so for an interior frame i the rows i − 1 and i + 1 belong to the same segment.
    """
    pos, seg_len = np.asarray(pos, dtype=np.int64), np.asarray(seg_len, dtype=np.int64)
    ax, ay = np.asarray(ax, dtype=float), np.asarray(ay, dtype=float)
    out = np.full(pos.size, np.nan)
    i = np.nonzero((pos >= 1) & (pos <= seg_len - 2))[0]
    if i.size:
        jx = (ax[i + 1] - ax[i - 1]) / (2.0 * dt)
        jy = (ay[i + 1] - ay[i - 1]) / (2.0 * dt)
        out[i] = np.hypot(jx, jy)
    return out


def sort_frames(frames: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Restore the contiguous (keys, seg_idx, pos) order — persisted, partitioned frames may be re-ordered."""
    return frames.sort([*keys, "seg_idx", "pos"])


def add_signal_columns(frames: pl.DataFrame, keys: list[str], kcfg: KinematicsConfig,
                       spec: SmoothingSpec | None = None) -> pl.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in frames.columns]
    if missing:
        raise KeyError(f"frames lack {missing} — pass M1 frame kinematics")
    spec = spec or kcfg.smoothing
    f = sort_frames(frames, keys)
    pos, seg_len = f["pos"].to_numpy(), f["seg_len"].to_numpy()
    edge = edge_distance(pos, seg_len)
    jerk = jerk_magnitude(pos, seg_len, f["ax"].cast(pl.Float64).to_numpy(), f["ay"].cast(pl.Float64).to_numpy(),
                          kcfg.dt_s)
    return f.with_columns(
        pl.Series("edge_dist", edge, dtype=pl.Int64),
        pl.Series("in_filter_boundary", edge < half_window(spec)),
        pl.Series("jerk_mag", jerk, dtype=pl.Float64, nan_to_null=True),
        (pl.col("dt_ms").cast(pl.Float64) - float(kcfg.nominal_interval_ms)).abs().alias("dt_dev_ms"),
    )


def _peak_aggs() -> list[pl.Expr]:
    aggs = []
    for name in PEAK_FEATURES:
        v, b = pl.col(f"_{name}"), pl.col("in_filter_boundary")
        has = v.is_not_null().any()
        aggs += [
            v.max().cast(pl.Float64).alias(name),
            v.filter(~b).max().cast(pl.Float64).alias(f"{name}_interior"),
            pl.when(has).then((b & (v == v.max())).any()).otherwise(None).alias(f"{name}_in_boundary"),
        ]
    return aggs


def trajectory_signal_metrics(frames: pl.DataFrame, keys: list[str], kcfg: KinematicsConfig,
                              spec: SmoothingSpec | None = None) -> pl.DataFrame:
    """One row per trajectory: sampling, structure, vendor agreement, roughness, heading, boundary."""
    f = frames if "edge_dist" in frames.columns else add_signal_columns(frames, keys, kcfg, spec)
    nom_s = kcfg.dt_s
    tol_ms = kcfg.offgrid_tolerance * kcfg.nominal_interval_ms
    f = f.with_columns(
        # Δt across the whole trajectory (gap breaks included) → missing-frame estimate
        pl.col("t_s").diff().over(keys).alias("_raw_dt_s"),
    ).with_columns(
        ((pl.col("_raw_dt_s") / nom_s).round(0) - 1).clip(lower_bound=0).fill_null(0).alias("_missing"),
        pl.when(pl.col("valid_kin") & pl.col("s").is_not_null()).then(pl.col("speed") - pl.col("s"))
        .otherwise(None).alias("_r"),
        *[expr.alias(f"_{name}") for name, expr in PEAK_FEATURES.items()],
    )
    valid = pl.col("valid_kin")
    interior_j = pl.col("jerk_mag").filter(~pl.col("in_filter_boundary"))
    out = f.group_by(keys).agg(
        pl.len().cast(pl.Int64).alias("n_frames"),
        valid.sum().cast(pl.Int64).alias("n_valid_kin"),
        pl.col("seg_idx").n_unique().cast(pl.Int64).alias("n_segments"),
        (pl.col("break_reason") == "gap").sum().cast(pl.Int64).alias("n_gap_breaks"),
        (pl.col("break_reason") == "jump").sum().cast(pl.Int64).alias("n_jump_breaks"),
        (pl.col("t_s").max() - pl.col("t_s").min()).cast(pl.Float64).alias("duration_s"),
        # sampling
        (pl.col("dt_ms").cast(pl.Float64).std() / pl.col("dt_ms").cast(pl.Float64).mean()).alias("dt_cv"),
        pl.col("dt_dev_ms").max().cast(pl.Float64).alias("dt_max_dev_ms"),
        (pl.col("dt_dev_ms") > tol_ms).sum().cast(pl.Int64).alias("n_offgrid"),
        pl.col("_missing").sum().cast(pl.Int64).alias("n_missing_frames"),
        # vendor agreement
        pl.col("_r").pow(2).mean().sqrt().cast(pl.Float64).alias("vendor_rmse"),
        pl.col("_r").abs().mean().cast(pl.Float64).alias("vendor_mae"),
        pl.col("_r").mean().cast(pl.Float64).alias("vendor_bias"),
        pl.col("_r").abs().quantile(0.95, "linear").cast(pl.Float64).alias("vendor_p95_abs"),
        pl.col("_r").count().cast(pl.Int64).alias("n_vendor_frames"),
        # roughness
        interior_j.pow(2).mean().sqrt().cast(pl.Float64).alias("jerk_rms_interior"),
        pl.col("jerk_mag").pow(2).mean().sqrt().cast(pl.Float64).alias("jerk_rms_all"),
        interior_j.quantile(0.95, "linear").cast(pl.Float64).alias("jerk_p95_interior"),
        # heading
        (pl.col("speed") < kcfg.min_heading_speed).filter(valid).mean().cast(pl.Float64).alias("lowspeed_frac"),
        pl.col("heading").is_null().filter(valid).mean().cast(pl.Float64).alias("heading_null_frac"),
        # boundary
        pl.col("in_filter_boundary").filter(valid).mean().cast(pl.Float64).alias("boundary_frac"),
        *_peak_aggs(),
    )
    rel = [pl.when(pl.col(n).abs() > 1e-9).then((pl.col(n) - pl.col(f"{n}_interior")) / pl.col(n).abs())
           .otherwise(None).cast(pl.Float64).alias(f"{n}_boundary_rel") for n in PEAK_FEATURES]
    return out.with_columns(
        (pl.col("n_valid_kin") / pl.col("n_frames")).cast(pl.Float64).alias("usable_frac"),
        (pl.col("n_offgrid") / pl.col("n_frames")).cast(pl.Float64).alias("offgrid_rate"),
        (pl.col("n_missing_frames") / (pl.col("n_frames") + pl.col("n_missing_frames"))).cast(pl.Float64)
        .alias("missing_frame_pct"),
        *rel,
    ).sort(keys)


SIGNAL_METRIC_DOCS: dict[str, str] = {
    "n_frames": "frames after de-duplication (M1 valid frames)",
    "n_valid_kin": "frames with SG derivatives (segment ≥ window)",
    "usable_frac": "n_valid_kin / n_frames",
    "duration_s": "last − first frame time (s)",
    "dt_cv": "coefficient of variation of within-segment Δt",
    "dt_max_dev_ms": "max |Δt − nominal| within segments (ms)",
    "offgrid_rate": "frames with |Δt − nominal| > offgrid_tolerance × nominal / n_frames",
    "n_missing_frames": "Σ max(round(Δt / nominal) − 1, 0) over the whole trajectory (gaps included)",
    "missing_frame_pct": "n_missing_frames / (n_frames + n_missing_frames)",
    "vendor_rmse": "RMS of derived − vendor speed over valid frames (yd/s)",
    "vendor_mae": "mean |derived − vendor speed| (yd/s)",
    "vendor_bias": "mean derived − vendor speed (yd/s)",
    "vendor_p95_abs": "95th percentile |derived − vendor speed| (yd/s)",
    "jerk_rms_interior": "RMS jerk over frames outside the filter boundary (yd/s³; roughness diagnostic)",
    "jerk_rms_all": "RMS jerk over all frames with a central difference (yd/s³)",
    "jerk_p95_interior": "95th percentile interior jerk (yd/s³)",
    "lowspeed_frac": "valid frames with speed < min_heading_speed (heading undefined by design)",
    "heading_null_frac": "valid frames with null heading (must equal lowspeed_frac)",
    "boundary_frac": "valid frames inside the SG filter boundary (expected share of boundary peaks if "
                     "peaks were located at random)",
    **{f"{n}": f"maximum of {n[5:]} over valid frames" for n in PEAK_FEATURES},
    **{f"{n}_interior": f"maximum of {n[5:]} excluding filter-boundary frames" for n in PEAK_FEATURES},
    **{f"{n}_in_boundary": "a frame attaining the peak lies in the filter boundary (ties → any)" for n in PEAK_FEATURES},
    **{f"{n}_boundary_rel": "(peak − interior peak) / |peak|: how much of the peak depends on boundary frames"
       for n in PEAK_FEATURES},
}
