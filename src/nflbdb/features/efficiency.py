"""Path geometry and movement efficiency per trajectory (Family D), on active frames.

    path_len_yds       Σ smoothed step over active frames (first active step excluded)
    net_disp_yds       |p_end − p_onset| (smoothed positions)
    path_efficiency    net / path ∈ (0, 1]; TASK-SPECIFIC — a shuttle or three-cone is designed to
                       return near its start, so efficiency is only comparable within a drill
    excess_path_yds    path − net
Principal-axis decomposition (frame-invariant, no field orientation needed): the axis is the
first eigenvector of the 2×2 covariance of (x̂, ŷ) over active frames; velocity is split into
longitudinal (along) and lateral (across) components.
    lateral_extent_yds / long_extent_yds   range of positions across / along the axis
    lateral_speed_p95                      95th pct |v_lat|
    frac_time_lateral                      share of frames with |v_lat| > |v_long|
For near-isotropic paths (e.g. a box drill) the axis is ill-defined → ``axis_anisotropy``
(λ1 / (λ1 + λ2) ∈ [0.5, 1]) is reported so analyses can exclude low-anisotropy trajectories.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def path_features(act: pl.DataFrame, keys: list[str], min_path_yds: float = 1.0) -> pl.DataFrame:
    f = act.sort([*keys, "t_s"])
    geo = f.group_by(keys).agg(
        pl.col("step_smooth").slice(1).sum().alias("path_len_yds"),
        ((pl.col("x_s").last() - pl.col("x_s").first()).pow(2)
         + (pl.col("y_s").last() - pl.col("y_s").first()).pow(2)).sqrt().alias("net_disp_yds"),
        pl.col("x_s").mean().alias("_mx"), pl.col("y_s").mean().alias("_my"),
        pl.col("x_s").var().alias("_cxx"), pl.col("y_s").var().alias("_cyy"),
        pl.cov("x_s", "y_s").alias("_cxy"),
    )
    # principal-axis angle on the (small, one-row-per-trajectory) table in NumPy
    cxy, cxx, cyy = (geo[c].cast(pl.Float64).fill_null(0.0).to_numpy() for c in ("_cxy", "_cxx", "_cyy"))
    geo = geo.with_columns(pl.Series("_th", 0.5 * np.arctan2(2 * cxy, cxx - cyy))).with_columns(
        ((pl.col("_cxx") - pl.col("_cyy")).pow(2) + 4 * pl.col("_cxy").pow(2)).sqrt().alias("_disc"),
    ).with_columns(
        pl.when((pl.col("_cxx") + pl.col("_cyy")) > 0)
        .then(0.5 + 0.5 * pl.col("_disc") / (pl.col("_cxx") + pl.col("_cyy"))).alias("axis_anisotropy"))
    c, s = pl.col("_th").cos(), pl.col("_th").sin()
    proj = f.join(geo.select(*keys, "_mx", "_my", "_th"), on=keys, how="inner").with_columns(
        ((pl.col("x_s") - pl.col("_mx")) * c + (pl.col("y_s") - pl.col("_my")) * s).alias("_u"),
        (-(pl.col("x_s") - pl.col("_mx")) * s + (pl.col("y_s") - pl.col("_my")) * c).alias("_w"),
        (pl.col("vx") * c + pl.col("vy") * s).abs().alias("_vl"),
        (-pl.col("vx") * s + pl.col("vy") * c).abs().alias("_vw"),
    ).group_by(keys).agg(
        (pl.col("_w").max() - pl.col("_w").min()).alias("lateral_extent_yds"),
        (pl.col("_u").max() - pl.col("_u").min()).alias("long_extent_yds"),
        pl.col("_vw").quantile(0.95).alias("lateral_speed_p95"),
        (pl.col("_vw") > pl.col("_vl")).mean().alias("frac_time_lateral"),
    )
    return (geo.join(proj, on=keys, how="left")
            .with_columns(pl.when(pl.col("path_len_yds") >= min_path_yds)
                          .then(pl.col("net_disp_yds") / pl.col("path_len_yds")).alias("path_efficiency"),
                          (pl.col("path_len_yds") - pl.col("net_disp_yds")).alias("excess_path_yds"))
            .drop("_mx", "_my", "_cxx", "_cyy", "_cxy", "_th", "_disc"))
