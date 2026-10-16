"""Movement events detected on frame-level kinematics (Polars, vectorised run-length logic).

Event types (each evaluated under every threshold set in ``KinematicsConfig``):

braking       run of frames with a_tan ≤ −threshold, lasting ≥ min_duration_s, losing
              ≥ min_change yd/s, starting from ≥ min_speed yd/s. Not every negative speed
              derivative is a braking event.
acceleration  run of frames with a_tan ≥ +threshold, lasting ≥ min_duration_s, gaining
              ≥ min_change yd/s (no entry-speed requirement: starts from rest count).
turn          run of frames with yaw rate of constant sign and |yaw rate| ≥ threshold (deg/s) while
              speed ≥ min_speed, lasting ≥ min_duration_s with |cumulative heading change| ≥ min_change.
reversal      a low-speed window (speed < min_heading_speed, where heading is undefined) whose
              heading just before and just after differ by ≥ min_change degrees and which lasts
              ≤ max_duration_s — a plant-and-cut such as a shuttle turn, which the yaw-rate
              definition cannot see because the athlete passes through near-zero speed.

Conventions
* An event lives inside one segment (runs never cross a gap / jump / trajectory boundary).
* ``duration_s`` = number of frames in the run × nominal Δt.
* ``entry_speed`` = speed on the frame BEFORE the run (falls back to the first run frame at a
  segment start); ``exit_speed`` = speed on the last run frame (turn/accel/braking) or on the
  frame AFTER the window (reversal).
* ``time_to_transition_s`` = time from the end of a braking (acceleration) event to the next frame
  of the same segment meeting the opposite acceleration threshold of the same rule — e.g.
  braking → re-acceleration. Null when it never happens within the segment.
* Events touching a segment edge may be truncated; ``touches_segment_edge`` flags them.
"""

from __future__ import annotations

import polars as pl

from .config import EventRule, KinematicsConfig
from .trajectory import TrajectoryDomain

EVENT_TYPES = ("braking", "acceleration", "turn", "reversal")

EVENT_SCHEMA: dict[str, pl.DataType] = {
    "event_type": pl.String, "rule": pl.String, "event_idx": pl.Int64,
    "start_frame_id": pl.Int64, "end_frame_id": pl.Int64, "t_start_s": pl.Float64, "t_end_s": pl.Float64,
    "n_frames": pl.Int64, "duration_s": pl.Float64, "entry_speed": pl.Float64, "exit_speed": pl.Float64,
    "min_speed": pl.Float64, "max_speed": pl.Float64, "speed_change": pl.Float64,
    "peak_abs_a_tan": pl.Float64, "mean_abs_a_tan": pl.Float64, "distance_yds": pl.Float64,
    "time_to_transition_s": pl.Float64, "direction": pl.String, "peak_abs_yaw_rate": pl.Float64,
    "heading_change_deg": pl.Float64, "median_radius_yds": pl.Float64, "max_abs_curvature": pl.Float64,
    "touches_segment_edge": pl.Boolean,
}


def _with_neighbours(kin: pl.DataFrame, seg_keys: list[str]) -> pl.DataFrame:
    return kin.with_columns(
        pl.col("speed").shift(1).over(seg_keys).alias("_speed_prev"),
        pl.col("speed").shift(-1).over(seg_keys).alias("_speed_next"),
        pl.col("heading").shift(1).over(seg_keys).alias("_heading_prev"),
        pl.col("heading").shift(-1).over(seg_keys).alias("_heading_next"),
    )


def _finish(ev: pl.DataFrame, seg_keys: list[str], event_type: str, rule: EventRule) -> pl.DataFrame:
    ev = ev.with_columns(pl.lit(event_type).alias("event_type"), pl.lit(rule.name).alias("rule"))
    ev = ev.sort([*seg_keys, "t_start_s"]).with_columns(
        pl.int_range(pl.len(), dtype=pl.Int64).over(seg_keys).alias("event_idx"))
    for c, t in EVENT_SCHEMA.items():
        ev = ev.with_columns((pl.col(c) if c in ev.columns else pl.lit(None)).cast(t).alias(c))
    extra = [c for c in ev.columns if c not in EVENT_SCHEMA and c not in seg_keys]
    return ev.select([*seg_keys, *EVENT_SCHEMA, *extra])


def speed_change_events(kin: pl.DataFrame, seg_keys: list[str], rule: EventRule, kind: str, dt: float,
                        context: list[str] | None = None) -> pl.DataFrame:
    """Braking (``kind='braking'``) or acceleration (``kind='acceleration'``) events for one rule."""
    sign = -1.0 if kind == "braking" else 1.0
    on = ((pl.col("a_tan") * sign) >= rule.threshold).fill_null(False)
    opposite = ((pl.col("a_tan") * -sign) >= rule.threshold).fill_null(False)
    ctx = [pl.col(c).first() for c in (context or []) if c in kin.columns]
    f = _with_neighbours(kin, seg_keys).with_columns(on.alias("_on")).with_columns(
        pl.col("_on").rle_id().over(seg_keys).alias("_run"),
        pl.when(opposite).then(pl.col("t_s")).otherwise(None)
        .fill_null(strategy="backward").over(seg_keys).alias("_next_opp_t"),
    )
    ev = (f.filter(pl.col("_on"))
          .group_by([*seg_keys, "_run"], maintain_order=True)
          .agg(*ctx,
               pl.len().cast(pl.Int64).alias("n_frames"),
               pl.col("frame_id").first().alias("start_frame_id"), pl.col("frame_id").last().alias("end_frame_id"),
               pl.col("t_s").first().alias("t_start_s"), pl.col("t_s").last().alias("t_end_s"),
               pl.coalesce(pl.col("_speed_prev").first(), pl.col("speed").first()).alias("entry_speed"),
               pl.col("speed").last().alias("exit_speed"),
               pl.col("speed").min().alias("min_speed"), pl.col("speed").max().alias("max_speed"),
               pl.col("a_tan").abs().max().alias("peak_abs_a_tan"), pl.col("a_tan").abs().mean().alias("mean_abs_a_tan"),
               pl.col("step_smooth").sum().alias("distance_yds"),
               pl.col("_next_opp_t").last().alias("_next_opp_t"),
               ((pl.col("pos").first() == 0) | (pl.col("pos").last() == pl.col("seg_len").first() - 1))
               .alias("touches_segment_edge"))
          .with_columns((pl.col("n_frames") * dt).alias("duration_s"),
                        ((pl.col("exit_speed") - pl.col("entry_speed")) * sign).alias("speed_change"),
                        (pl.col("_next_opp_t") - pl.col("t_end_s")).alias("time_to_transition_s"))
          .filter((pl.col("duration_s") >= rule.min_duration_s - 1e-9)
                  & (pl.col("speed_change") >= rule.min_change)
                  & (pl.col("entry_speed") >= rule.min_speed))
          .drop("_run", "_next_opp_t"))
    return _finish(ev, seg_keys, kind, rule)


def turn_events(kin: pl.DataFrame, seg_keys: list[str], rule: EventRule, dt: float,
                context: list[str] | None = None) -> pl.DataFrame:
    yaw = pl.col("yaw_rate")
    fast = (pl.col("speed") >= rule.min_speed).fill_null(False)
    state = (pl.when(fast & (yaw >= rule.threshold)).then(1)
             .when(fast & (yaw <= -rule.threshold)).then(-1).otherwise(0))
    ctx = [pl.col(c).first() for c in (context or []) if c in kin.columns]
    f = _with_neighbours(kin, seg_keys).with_columns(state.alias("_state")).with_columns(
        pl.col("_state").rle_id().over(seg_keys).alias("_run"))
    ev = (f.filter(pl.col("_state") != 0)
          .group_by([*seg_keys, "_run"], maintain_order=True)
          .agg(*ctx,
               pl.len().cast(pl.Int64).alias("n_frames"),
               pl.col("_state").first().alias("_dir"),
               pl.col("frame_id").first().alias("start_frame_id"), pl.col("frame_id").last().alias("end_frame_id"),
               pl.col("t_s").first().alias("t_start_s"), pl.col("t_s").last().alias("t_end_s"),
               pl.coalesce(pl.col("_speed_prev").first(), pl.col("speed").first()).alias("entry_speed"),
               pl.col("speed").last().alias("exit_speed"),
               pl.col("speed").min().alias("min_speed"), pl.col("speed").max().alias("max_speed"),
               pl.col("yaw_rate").abs().max().alias("peak_abs_yaw_rate"),
               (pl.col("yaw_rate") * dt).sum().alias("heading_change_deg"),
               (pl.col("speed") / pl.col("yaw_rate").abs().radians()).median().alias("median_radius_yds"),
               pl.col("curvature").abs().max().alias("max_abs_curvature"),
               pl.col("step_smooth").sum().alias("distance_yds"),
               ((pl.col("pos").first() == 0) | (pl.col("pos").last() == pl.col("seg_len").first() - 1))
               .alias("touches_segment_edge"))
          .with_columns((pl.col("n_frames") * dt).alias("duration_s"),
                        (pl.col("min_speed") - pl.col("entry_speed")).alias("speed_change"),
                        pl.when(pl.col("_dir") > 0).then(pl.lit("ccw")).otherwise(pl.lit("cw")).alias("direction"))
          .filter((pl.col("duration_s") >= rule.min_duration_s - 1e-9)
                  & (pl.col("heading_change_deg").abs() >= rule.min_change))
          .drop("_run", "_dir"))
    return _finish(ev, seg_keys, "turn", rule)


def reversal_events(kin: pl.DataFrame, seg_keys: list[str], rule: EventRule, dt: float, min_heading_speed: float,
                    context: list[str] | None = None) -> pl.DataFrame:
    slow = (pl.col("speed") < min_heading_speed).fill_null(False) & pl.col("valid_kin")
    ctx = [pl.col(c).first() for c in (context or []) if c in kin.columns]
    f = _with_neighbours(kin, seg_keys).with_columns(slow.alias("_slow")).with_columns(
        pl.col("_slow").rle_id().over(seg_keys).alias("_run"))
    ev = (f.filter(pl.col("_slow"))
          .group_by([*seg_keys, "_run"], maintain_order=True)
          .agg(*ctx,
               pl.len().cast(pl.Int64).alias("n_frames"),
               pl.col("frame_id").first().alias("start_frame_id"), pl.col("frame_id").last().alias("end_frame_id"),
               pl.col("t_s").first().alias("t_start_s"), pl.col("t_s").last().alias("t_end_s"),
               pl.col("_speed_prev").first().alias("entry_speed"), pl.col("_speed_next").last().alias("exit_speed"),
               pl.col("speed").min().alias("min_speed"), pl.col("speed").max().alias("max_speed"),
               pl.col("_heading_prev").first().alias("_h_in"), pl.col("_heading_next").last().alias("_h_out"),
               pl.col("step_smooth").sum().alias("distance_yds"),
               ((pl.col("pos").first() == 0) | (pl.col("pos").last() == pl.col("seg_len").first() - 1))
               .alias("touches_segment_edge"))
          # signed heading change wrapped to [−180, 180); headings ∈ [0, 360) so the +540 keeps the
          # dividend positive (no dependence on the sign convention of float modulo)
          .with_columns((((pl.col("_h_out") - pl.col("_h_in") + 540.0) % 360.0) - 180.0).alias("heading_change_deg"),
                        (pl.col("n_frames") * dt).alias("duration_s"))
          .with_columns((pl.col("exit_speed") - pl.col("entry_speed")).alias("speed_change"),
                        pl.when(pl.col("heading_change_deg") > 0).then(pl.lit("ccw")).otherwise(pl.lit("cw"))
                        .alias("direction"))
          .filter(pl.col("_h_in").is_not_null() & pl.col("_h_out").is_not_null()
                  & (pl.col("heading_change_deg").abs() >= rule.min_change)
                  & (pl.col("duration_s") <= (rule.max_duration_s if rule.max_duration_s is not None else 1e9) + 1e-9))
          .drop("_run", "_h_in", "_h_out"))
    return _finish(ev, seg_keys, "reversal", rule)


def detect_events(kin: pl.DataFrame, domain: TrajectoryDomain, kcfg: KinematicsConfig,
                  event_types: tuple[str, ...] = EVENT_TYPES, rules: tuple[str, ...] | None = None) -> pl.DataFrame:
    """All event types × threshold sets for frames from ``kinematics.compute_kinematics``.

    ``rules`` restricts the threshold sets by name (e.g. ``("default",)``).
    """
    seg_keys = domain.seg_keys
    ctx = [c for c in domain.context if c in kin.columns]
    dt = kcfg.dt_s
    parts = []
    for et in event_types:
        for rule in kcfg.rules(et):
            if rules is not None and rule.name not in rules:
                continue
            if et in ("braking", "acceleration"):
                parts.append(speed_change_events(kin, seg_keys, rule, et, dt, ctx))
            elif et == "turn":
                parts.append(turn_events(kin, seg_keys, rule, dt, ctx))
            else:
                parts.append(reversal_events(kin, seg_keys, rule, dt, kcfg.min_heading_speed, ctx))
    parts = [p for p in parts if p.height]
    if not parts:
        return empty_events(kin, domain)
    return pl.concat(parts, how="diagonal_relaxed")


def empty_events(kin: pl.DataFrame, domain: TrajectoryDomain) -> pl.DataFrame:
    cols = {c: kin.schema[c] for c in [*domain.keys, *domain.context] if c in kin.columns}
    return pl.DataFrame(schema={**cols, "seg_idx": pl.Int32, **EVENT_SCHEMA})
