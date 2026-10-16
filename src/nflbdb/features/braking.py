"""Deceleration and braking features per trajectory (Family B).

Two sources, deliberately separate:
* frame level (active frames): peak and high-quantile deceleration (−a_tan) — continuous but
  noise-sensitive single-frame statistics;
* event level (M1 ``braking`` events of one threshold set): only sustained, substantial speed
  losses (threshold, duration, minimum speed loss, minimum entry speed) count. Not every negative
  speed derivative is a braking event.

``braking_speed_loss_per_yd`` (speed lost per yard travelled while braking) is the "braking
efficiency" proxy: a higher value = speed shed over a shorter distance. It is a kinematic ratio,
not a force or an energy estimate.
"""

from __future__ import annotations

import polars as pl

COUNT_COLUMNS = ("n_braking_events", "braking_time_s", "braking_distance_yds")


def braking_features(act: pl.DataFrame, events: pl.DataFrame, keys: list[str], rule: str,
                     window: pl.DataFrame | None = None) -> pl.DataFrame:
    """``window`` (from ``temporal.movement_window``): if given, only braking events STARTING inside
    the active window [t_onset, t_end_active] count, consistent with the frame-level features
    (events during pre-start drift or after the drill are excluded)."""
    a = pl.col("a_tan")
    frame = act.group_by(keys).agg(
        (-a).max().alias("decel_max"),
        (-a).filter(a < 0).quantile(0.95).alias("decel_p95"),
        (-a).filter(a < 0).mean().alias("decel_mean"),
    )
    ev = events.filter((pl.col("event_type") == "braking") & (pl.col("rule") == rule))
    if window is not None:
        ev = (ev.join(window.select(*keys, "t_onset_s", "t_end_active_s"), on=keys, how="inner")
              .filter((pl.col("t_start_s") >= pl.col("t_onset_s")) & (pl.col("t_start_s") <= pl.col("t_end_active_s")))
              .drop("t_onset_s", "t_end_active_s"))
    per = ev.group_by(keys).agg(
        pl.len().cast(pl.Int64).alias("n_braking_events"),
        pl.col("duration_s").sum().alias("braking_time_s"),
        pl.col("distance_yds").sum().alias("braking_distance_yds"),
        ((pl.col("mean_abs_a_tan") * pl.col("duration_s")).sum() / pl.col("duration_s").sum()).alias("braking_intensity_mean"),
        pl.col("peak_abs_a_tan").max().alias("braking_peak"),
        pl.col("speed_change").max().alias("braking_speed_drop_max"),
        (pl.col("speed_change") / pl.col("entry_speed")).median().alias("speed_reduction_ratio"),
        pl.when(pl.col("distance_yds") > 0).then(pl.col("speed_change") / pl.col("distance_yds"))
        .median().alias("braking_speed_loss_per_yd"),
        pl.col("time_to_transition_s").median().alias("time_to_reaccel_median_s"),
        pl.col("time_to_transition_s").is_not_null().mean().alias("frac_braking_with_reaccel"),
    )
    return (frame.join(per, on=keys, how="left")
            .with_columns([pl.col(c).fill_null(0) for c in COUNT_COLUMNS]))
