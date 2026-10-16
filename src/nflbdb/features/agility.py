"""Change of direction (COD) events and agility features (Family C).

A COD is a M1 ``turn`` (yaw-rate run at speed) or ``reversal`` (low-speed plant-and-cut) event of
one threshold set with |heading change| ≥ ``cod_min_angle``. Events of the same trajectory
segment closer than ``cod_merge_gap_s`` are merged into one COD (a cut often shows as a short
turn running into a reversal); the merged angle is the sum of |angles|.

Per-COD window measures (from frame kinematics of the same segment only):
    approach_speed    max speed in [t_start − cod_approach_s, t_start]
    approach_decel    mean −a_tan in [t_start − cod_pre_s, t_start)          (+ = slowing)
    approach_decel_peak  max −a_tan in the same window
    exit_accel        mean a_tan in (t_end, t_end + cod_post_s]
    exit_speed_1s     max speed in (t_end, t_end + cod_approach_s]
    speed_retention   min_speed / approach_speed (≈ 0 for reversals by construction)
    cod_time_s        time from the last frame before the COD with speed ≥ recover_frac ×
                      approach_speed to the first frame after it reaching that speed again
                      (null = not recovered within cod_recover_max_s, i.e. censored)
NOTE: M1 ``entry_speed`` of a reversal is the speed just before speed fell below 1 yd/s, NOT the
approach speed — hence the window-based ``approach_speed`` here.
Turn sense (ccw/cw) is not used: Combine coordinate handedness is unverified.
"""

from __future__ import annotations

import polars as pl

from .config import MovementConfig

COD_COLUMNS = ("cod_idx", "cod_type", "n_merged_events", "t_start_s", "t_end_s", "cod_angle_deg", "min_speed",
               "approach_speed", "approach_decel", "approach_decel_peak", "exit_accel", "exit_speed_1s",
               "speed_retention", "cod_time_s", "turn_radius_yds", "touches_segment_edge")


def cod_events(events: pl.DataFrame, keys: list[str], mcfg: MovementConfig, context: list[str] | None = None
               ) -> pl.DataFrame:
    seg = [*keys, "seg_idx"]
    ctx = [c for c in (context or []) if c in events.columns]
    ev = (events.filter((pl.col("rule") == mcfg.event_rule) & pl.col("event_type").is_in(["turn", "reversal"])
                        & (pl.col("heading_change_deg").abs() >= mcfg.cod_min_angle))
          .sort([*seg, "t_start_s"])
          .with_columns((pl.col("t_start_s") - pl.col("t_end_s").shift(1).over(seg)).alias("_gap"))
          .with_columns((pl.col("_gap").is_null() | (pl.col("_gap") > mcfg.cod_merge_gap_s)).cast(pl.Int32)
                        .cum_sum().over(seg).alias("_grp")))
    return (ev.group_by([*seg, "_grp"], maintain_order=True).agg(
                *[pl.col(c).first() for c in ctx],
                pl.when((pl.col("event_type") == "reversal").any()).then(pl.lit("reversal")).otherwise(pl.lit("turn"))
                .alias("cod_type"),
                pl.len().cast(pl.Int64).alias("n_merged_events"),
                pl.col("t_start_s").min(), pl.col("t_end_s").max(),
                pl.col("heading_change_deg").abs().sum().alias("cod_angle_deg"),
                pl.col("min_speed").min(),
                pl.col("median_radius_yds").median().alias("turn_radius_yds"),
                pl.col("touches_segment_edge").any())
            .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).over(keys).alias("cod_idx"))
            .drop("_grp"))


def cod_window_measures(kin: pl.DataFrame, cod: pl.DataFrame, keys: list[str], mcfg: MovementConfig) -> pl.DataFrame:
    """Join each COD with the frames of its segment and measure approach / exit / recovery."""
    seg = [*keys, "seg_idx"]
    fr = kin.filter(pl.col("valid_kin")).select(*seg, pl.col("t_s").alias("_t"), pl.col("speed").alias("_v"),
                                                 pl.col("a_tan").alias("_a"))
    j = cod.select(*seg, "cod_idx", "t_start_s", "t_end_s").join(fr, on=seg, how="inner")
    t, ts, te = pl.col("_t"), pl.col("t_start_s"), pl.col("t_end_s")
    m = j.group_by([*keys, "cod_idx"]).agg(
        pl.col("_v").filter((t >= ts - mcfg.cod_approach_s) & (t <= ts)).max().alias("approach_speed"),
        (-pl.col("_a")).filter((t >= ts - mcfg.cod_pre_s) & (t < ts)).mean().alias("approach_decel"),
        (-pl.col("_a")).filter((t >= ts - mcfg.cod_pre_s) & (t < ts)).max().alias("approach_decel_peak"),
        pl.col("_a").filter((t > te) & (t <= te + mcfg.cod_post_s)).mean().alias("exit_accel"),
        pl.col("_v").filter((t > te) & (t <= te + mcfg.cod_approach_s)).max().alias("exit_speed_1s"),
    )
    # recovery threshold needs approach_speed → second pass
    thr = j.join(m.select(*keys, "cod_idx", "approach_speed"), on=[*keys, "cod_idx"], how="left").with_columns(
        (pl.col("_v") >= mcfg.cod_recover_frac * pl.col("approach_speed")).alias("_fast"))
    rec = thr.group_by([*keys, "cod_idx"]).agg(
        t.filter(pl.col("_fast") & (t < ts) & (t >= ts - mcfg.cod_recover_max_s)).max().alias("_t_loss"),
        t.filter(pl.col("_fast") & (t > te) & (t <= te + mcfg.cod_recover_max_s)).min().alias("_t_rec"))
    m = m.join(rec, on=[*keys, "cod_idx"], how="left").with_columns(
        (pl.col("_t_rec") - pl.col("_t_loss")).alias("cod_time_s")).drop("_t_loss", "_t_rec")
    out = cod.join(m, on=[*keys, "cod_idx"], how="left").with_columns(
        pl.when(pl.col("approach_speed") > 0).then(pl.col("min_speed") / pl.col("approach_speed"))
        .alias("speed_retention"))
    extra = [c for c in out.columns if c not in COD_COLUMNS and c not in seg]
    return out.select(*seg, *extra, *COD_COLUMNS)


def agility_features(act: pl.DataFrame, cod: pl.DataFrame, keys: list[str], mcfg: MovementConfig, dt: float
                     ) -> pl.DataFrame:
    """Per trajectory: COD counts/angles, median COD measures, yaw-based turning load."""
    frame = act.group_by(keys).agg(
        (pl.col("yaw_rate").abs() * dt).sum().alias("yaw_integral_deg"),
        pl.col("yaw_rate").abs().quantile(0.95).alias("abs_yaw_rate_p95"),
        pl.col("a_lat").abs().quantile(0.95).alias("a_lat_abs_p95"),
    )
    per = cod.group_by(keys).agg(
        pl.len().cast(pl.Int64).alias("n_cod"),
        (pl.col("cod_angle_deg") >= mcfg.cod_sharp_angle).sum().cast(pl.Int64).alias("n_cod_sharp"),
        (pl.col("cod_type") == "reversal").sum().cast(pl.Int64).alias("n_cod_reversal"),
        pl.col("cod_angle_deg").sum().alias("cod_angle_total_deg"),
        pl.col("approach_speed").median().alias("cod_approach_speed_median"),
        pl.col("approach_decel").median().alias("cod_approach_decel_median"),
        pl.col("exit_accel").median().alias("cod_exit_accel_median"),
        pl.col("cod_time_s").median().alias("cod_time_median_s"),
        pl.col("speed_retention").filter(pl.col("cod_type") == "turn").median().alias("turn_speed_retention_median"),
        pl.col("turn_radius_yds").median().alias("turn_radius_median_yds"),
    )
    return (frame.join(per, on=keys, how="left")
            # unpacked: Polars 2.0 rejects a list mixed with other positional expressions
            # ("not yet implemented: Nested object types")
            .with_columns(*[pl.col(c).fill_null(0) for c in ("n_cod", "n_cod_sharp", "n_cod_reversal")],
                          pl.col("cod_angle_total_deg").fill_null(0.0)))
