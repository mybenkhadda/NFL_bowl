"""Speed and acceleration features per trajectory (Family A), computed on ACTIVE frames
(``temporal.active_frames``: onset ≤ t ≤ end of activity, valid derivatives only).

Scalar vs vector acceleration are kept apart:
    a_tan      = d|v|/dt   (speed derivative, signed)   → ``a_tan_*``, ``accel_*`` features
    accel_mag  = |a|       (vector magnitude, ≥ |a_tan|, includes turning) → ``accel_mag_*``

Censoring: a Combine forty trajectory ends at the finish line while the athlete is still at or
near peak speed (M1 figure ``m1_signal_example_forty_yard_dash``). Peak-speed timing is then a
property of the recording window, not of the athlete → ``peak_speed_censored``.
``accel_integral_pos`` = Σ max(a_tan, 0)·Δt equals the speed gained during accelerating frames
(yd/s) — reported because it is noise-robust, not because it is new information.
"""

from __future__ import annotations

import polars as pl

from .config import MovementConfig


def _thr_name(th: float) -> str:
    return f"time_above_{th:g}yds_s".replace(".", "p")


def speed_accel_features(act: pl.DataFrame, keys: list[str], mcfg: MovementConfig, dt: float) -> pl.DataFrame:
    seg = [*keys, "seg_idx"]
    nsus = max(1, int(round(mcfg.sustain_window_s / dt)))
    f = (act.sort([*keys, "t_s"])
         .with_columns(pl.col("a_tan").rolling_mean(window_size=nsus, min_samples=nsus).over(seg).alias("_a_sus"),
                       pl.col("speed").max().over(keys).alias("_vpk"),
                       pl.col("step_smooth").fill_null(0.0).cum_sum().over(keys).alias("_cs"))
         .with_columns((pl.col("_cs") - pl.col("_cs").first().over(keys)).alias("_dist")))
    v, a = pl.col("speed"), pl.col("a_tan")
    out = f.group_by(keys).agg(
        pl.len().alias("n_active_frames"),
        v.mean().alias("speed_mean"), v.median().alias("speed_median"),
        v.quantile(0.90).alias("speed_p90"), v.quantile(0.95).alias("speed_p95"), v.max().alias("speed_max"),
        v.std().alias("speed_sd"),
        *[((v >= th).sum() * dt).alias(_thr_name(th)) for th in mcfg.speed_thresholds],
        ((v >= mcfg.high_speed_frac * pl.col("_vpk")).sum() * dt).alias("time_high_speed_s"),
        pl.col("t_rel").get(v.arg_max()).alias("time_to_peak_speed_s"),
        pl.col("_dist").get(v.arg_max()).alias("dist_to_peak_speed_yds"),
        (pl.len() - 1 - v.arg_max()).alias("_frames_after_peak"),
        pl.col("t_rel").filter(v >= 0.1 * pl.col("_vpk")).min().alias("_t10"),
        pl.col("t_rel").filter(v >= 0.9 * pl.col("_vpk")).min().alias("_t90"),
        a.filter(pl.col("t_rel") < mcfg.initial_window_s).mean().alias("accel_initial"),
        a.max().alias("a_tan_max"), a.quantile(0.90).alias("a_tan_p90"), a.quantile(0.95).alias("a_tan_p95"),
        a.filter(a > 0).mean().alias("a_tan_pos_mean"),
        pl.col("_a_sus").max().alias("accel_sustained"),
        pl.col("t_rel").sort_by("a_tan", descending=True, nulls_last=True).first().alias("time_to_peak_accel_s"),
        ((a >= mcfg.phase_accel).sum() * dt).alias("accel_time_s"),
        (a.clip(lower_bound=0.0) * dt).sum().alias("accel_integral_pos"),
        a.std().alias("a_tan_sd"),
        pl.col("accel_mag").max().alias("accel_mag_max"), pl.col("accel_mag").quantile(0.95).alias("accel_mag_p95"),
    )
    return (out.with_columns(
                pl.when(pl.col("speed_mean") > 0).then(pl.col("speed_sd") / pl.col("speed_mean")).alias("speed_cv"),
                (pl.col("_frames_after_peak") < mcfg.peak_edge_frames).alias("peak_speed_censored"),
                (pl.col("_t90") - pl.col("_t10")).alias("rise_time_10_90_s"))
            .with_columns(pl.when(pl.col("rise_time_10_90_s") > 0)
                          .then(0.8 * pl.col("speed_max") / pl.col("rise_time_10_90_s")).alias("buildup_gradient"))
            .drop("_frames_after_peak", "_t10", "_t90"))
