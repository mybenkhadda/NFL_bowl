"""Segment- and trajectory-level quality metrics with rule-based tiers.

The tier is a *usability* label for kinematic features, decided by explicit rules on measured
numbers (``TierRules``). It never deletes data; downstream feature tables carry the tier so
analyses can filter (e.g. ``quality_tier in ('A', 'B')``) and report sensitivity to the filter.
"""

from __future__ import annotations

import polars as pl

from .config import KinematicsConfig
from .trajectory import TrajectoryDomain


def summarize_segments(kin: pl.DataFrame, domain: TrajectoryDomain, kcfg: KinematicsConfig) -> pl.DataFrame:
    """One row per segment (trajectory keys + seg_idx)."""
    nom = float(kcfg.nominal_interval_ms)
    ctx = [pl.col(c).first() for c in domain.context if c in kin.columns]
    has_vendor_s = "s" in kin.columns
    vendor = [
        (pl.col("speed") - pl.col("s")).abs().median().alias("speed_vendor_mad"),
        (pl.col("speed") - pl.col("s")).abs().quantile(0.95).alias("speed_vendor_p95"),
        pl.col("s").max().alias("max_vendor_s"),
    ] if has_vendor_s else []
    return (
        kin.group_by(domain.seg_keys, maintain_order=True)
        .agg(*ctx,
             pl.col("break_reason").first().alias("break_reason"),
             pl.col("ts").first().alias("start_ts"),
             pl.col("t_s").first().alias("t_start_s"),
             pl.len().cast(pl.Int64).alias("n_frames"),
             (pl.col("t_s").last() - pl.col("t_s").first()).alias("duration_s"),
             (pl.col("seg_len").first() >= kcfg.min_segment_frames).alias("smoothable"),
             ((pl.col("dt_ms") - nom).abs() > kcfg.offgrid_tolerance * nom).sum().cast(pl.Int64).alias("n_offgrid"),
             pl.col("step_yds").sum().alias("path_raw_yds"),
             pl.col("step_smooth").sum().alias("path_smooth_yds"),
             ((pl.col("x").last() - pl.col("x").first()).pow(2)
              + (pl.col("y").last() - pl.col("y").first()).pow(2)).sqrt().alias("net_disp_yds"),
             pl.col("step_yds").max().alias("max_step_raw_yds"),
             pl.col("pos_resid").pow(2).mean().sqrt().alias("resid_rms_yds"),
             pl.col("pos_resid").quantile(0.95).alias("resid_p95_yds"),
             *vendor,
             pl.col("speed").max().alias("max_speed"),
             pl.col("accel_mag").max().alias("max_accel_mag"),
             (pl.col("speed") > kcfg.max_speed_yds).sum().cast(pl.Int64).alias("n_speed_implausible"),
             (pl.col("accel_mag") > kcfg.max_accel_yds2).sum().cast(pl.Int64).alias("n_accel_implausible"),
             pl.col("heading_valid").mean().alias("frac_heading_valid"))
        .sort(domain.seg_keys)
    )


def trajectory_quality(segments: pl.DataFrame, traj_stats: pl.DataFrame, domain: TrajectoryDomain,
                       kcfg: KinematicsConfig) -> pl.DataFrame:
    """One row per trajectory with coverage, aggregated quality metrics and ``quality_tier``."""
    keys = list(domain.keys)
    t = kcfg.tiers
    ctx = [pl.col(c).first() for c in domain.context if c in segments.columns]
    seg = segments.group_by(keys).agg(
        *ctx,
        pl.len().cast(pl.Int64).alias("n_segments_seen"),
        pl.col("smoothable").sum().cast(pl.Int64).alias("n_smoothable_segments"),
        pl.col("n_frames").filter(pl.col("smoothable")).sum().cast(pl.Int64).alias("frames_smoothable"),
        pl.col("n_frames").max().cast(pl.Int64).alias("longest_segment_frames"),
        ((pl.col("resid_rms_yds").pow(2) * pl.col("n_frames")).sum() / pl.col("n_frames").sum()).sqrt()
        .alias("resid_rms_yds"),
        *([pl.col("speed_vendor_mad").median().alias("speed_vendor_mad")] if "speed_vendor_mad" in segments.columns else []),
        pl.col("n_offgrid").sum().alias("n_offgrid"),
        pl.col("max_speed").max().alias("max_speed"),
        pl.col("max_accel_mag").max().alias("max_accel_mag"),
        pl.col("n_speed_implausible").sum().alias("n_speed_implausible"),
        pl.col("n_accel_implausible").sum().alias("n_accel_implausible"),
        pl.col("path_smooth_yds").sum().alias("path_smooth_yds"),
    )
    df = traj_stats.join(seg, on=keys, how="left").with_columns(
        pl.col("frames_smoothable").fill_null(0),
        (pl.col("frames_smoothable").fill_null(0) / pl.col("n_raw_frames")).alias("coverage"),
    )
    mad = pl.col("speed_vendor_mad") if "speed_vendor_mad" in df.columns else pl.lit(None, dtype=pl.Float64)
    rules = {
        "reject": [((pl.col("frames_smoothable") == 0), "no_smoothable_segment"),
                   ((pl.col("n_speed_implausible").fill_null(0) > 0), "speed_implausible"),
                   ((pl.col("n_accel_implausible").fill_null(0) > 0), "accel_implausible")],
        "C": [((pl.col("coverage") < t.c_min_coverage), "low_coverage"),
              ((pl.col("resid_rms_yds") > t.c_max_resid_rms).fill_null(False), "high_residual")],
        "B": [((pl.col("n_segments") > 1), "split_segments"),
              ((pl.col("resid_rms_yds") > t.b_max_resid_rms).fill_null(False), "moderate_residual"),
              ((mad > t.b_max_speed_mad).fill_null(False), "vendor_speed_disagreement")],
    }
    any_ = {k: pl.any_horizontal([c for c, _ in v]) for k, v in rules.items()}
    reasons = [pl.when(c).then(pl.lit(r)) for v in rules.values() for c, r in v]
    return df.with_columns(
        pl.when(any_["reject"]).then(pl.lit("reject")).when(any_["C"]).then(pl.lit("C"))
        .when(any_["B"]).then(pl.lit("B")).otherwise(pl.lit("A")).alias("quality_tier"),
        pl.concat_str(reasons, separator=";", ignore_nulls=True).alias("tier_reasons"),
    ).sort(keys)
