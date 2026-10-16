"""Trajectory identity, drill canonicalisation and segmentation (Polars).

Hierarchy
    trajectory  one athlete's continuous recording of one unit of activity
                Combine: (event_id, nfl_id, attempt) — PLAYER rows only
                NFL:     (game_id, play_id, nfl_id) — football rows excluded
    segment     maximal run of frames inside a trajectory with
                  * strictly increasing timestamps (exact duplicate timestamps removed, counted),
                  * Δt ≤ gap_factor × nominal (a missing frame starts a new segment),
                  * per-frame displacement ≤ max_step_yds × max(Δt / nominal, 1) (a coordinate
                    jump starts a new segment — the jump itself is never differentiated),
                  * one drill_name (Combine; a drill change inside an attempt starts a new segment).
    frame       one timestamp of a segment; ``pos`` = 0-based position within the segment.

Every derivative downstream is computed within a segment. Output is sorted by
(trajectory keys, ts) so segments are contiguous — a precondition of
``signal_processing.segmented_savgol``.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .config import KinematicsConfig


@dataclass(frozen=True)
class TrajectoryDomain:
    name: str
    keys: tuple[str, ...]
    context: tuple[str, ...]            # carried unchanged (constant within trajectory, or first value)
    partition: str                      # cohort column used for partitioned outputs
    split_on: tuple[str, ...] = ()      # a change of any of these columns starts a new segment

    @property
    def seg_keys(self) -> list[str]:
        return [*self.keys, "seg_idx"]


COMBINE = TrajectoryDomain(
    name="combine", keys=("event_id", "nfl_id", "attempt"),
    context=("draft_year", "drill_type", "drill_name", "drill"), partition="draft_year", split_on=("drill_name",))
NFL = TrajectoryDomain(
    name="nfl", keys=("game_id", "play_id", "nfl_id"), context=("season",), partition="season")
DOMAINS = {d.name: d for d in (COMBINE, NFL)}

# Raw columns read from silver (kinematic inputs + identifiers). Vendor s/a/dir are kept only
# for agreement diagnostics; derived kinematics come from x, y.
RAW_COLUMNS = {
    "combine": ["draft_year", "event_id", "nfl_id", "attempt", "drill_type", "drill_name", "frame_id", "ts",
                "x", "y", "s", "a", "dir"],
    "nfl": ["season", "game_id", "play_id", "nfl_id", "frame_id", "ts", "x", "y", "s", "a", "dir", "o"],
}

# --------------------------------------------------------------------------------------
# Drill names: the same drill appears under different spellings in different draft years.
# Evidence (reports/tables/combine_drill_counts.csv): each alias occurs ONLY in years where the
# canonical spelling is absent, with comparable athlete and attempt counts → treated as a rename.
# INFERRED, not confirmed by the data provider. Raw drill_name is always kept alongside.
# --------------------------------------------------------------------------------------
DRILL_ALIASES: dict[str, str] = {
    "LINE": "LINE_DRILL",                                   # DB, 2023 only (LINE_DRILL 2024–2025)
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",               # WR, 2024 only
    "PASS_PRO_MIRROR_DRILL": "PASS_PRO-MIRROR_DRILL",       # OL, 2023 only
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",     # TE, 2023 only
}

# Explicit drill families (no regexes — Phase 02 found regex families matching unintended drills).
DRILL_FAMILIES: dict[str, tuple[str, ...]] = {
    "sprint": ("FORTY_YARD_DASH",),
    "agility": ("SHORT_SHUTTLE", "THREE_CONE_DRILL"),
}


def canonical_drill_expr(col: str = "drill_name") -> pl.Expr:
    return pl.col(col).replace(DRILL_ALIASES).alias("drill")


def prepare_frames(raw: pl.DataFrame, domain: TrajectoryDomain, kcfg: KinematicsConfig
                   ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Order, de-duplicate and segment raw frames.

    Returns ``(frames, trajectory_stats)``:

    * ``frames`` — valid frames sorted by (keys, ts) with ``seg_idx`` (1-based within the
      trajectory), ``pos`` (0-based within the segment), ``seg_len``, ``dt_ms`` (null at a
      segment start), ``step_yds`` (raw displacement from the previous frame of the same
      segment; null at a segment start), ``break_reason`` (why the segment starts there),
      ``t_s`` (seconds since the trajectory's first valid frame).
    * ``trajectory_stats`` — one row per trajectory: frames read, removed (null coordinates,
      duplicate timestamps), and the number of gap / jump / drill-change breaks.
    """
    keys = list(domain.keys)
    nom = float(kcfg.nominal_interval_ms)
    if "drill_name" in raw.columns and "drill" not in raw.columns:
        raw = raw.with_columns(canonical_drill_expr())

    df = raw.with_columns(
        (pl.col("x").is_null() | pl.col("y").is_null() | pl.col("ts").is_null()
         | pl.col("x").is_nan() | pl.col("y").is_nan()).alias("_bad_xy"))
    stats_raw = df.group_by(keys).agg(pl.len().alias("n_raw_frames"), pl.col("_bad_xy").sum().alias("n_null_xy_removed"))

    df = df.filter(~pl.col("_bad_xy")).drop("_bad_xy").sort([*keys, "ts"])
    df = df.with_columns((pl.col("ts") == pl.col("ts").shift(1)).over(keys).fill_null(False).alias("_dup"))
    stats_dup = df.group_by(keys).agg(pl.col("_dup").sum().alias("n_dup_ts_removed"))
    df = df.filter(~pl.col("_dup")).drop("_dup")

    df = df.with_columns(
        pl.col("ts").diff().over(keys).dt.total_milliseconds().cast(pl.Float64).alias("_dt"),
        ((pl.col("x") - pl.col("x").shift(1)).pow(2) + (pl.col("y") - pl.col("y").shift(1)).pow(2))
        .sqrt().over(keys).alias("_step"),
    )
    first = pl.col("_dt").is_null()
    gap = pl.col("_dt") > nom * kcfg.gap_factor
    jump = pl.col("_step") > kcfg.max_step_yds * pl.max_horizontal(pl.col("_dt") / nom, pl.lit(1.0))
    split = [(pl.col(c) != pl.col(c).shift(1)).over(keys).fill_null(False) for c in domain.split_on if c in df.columns]
    changed = pl.any_horizontal(split) if split else pl.lit(False)
    df = df.with_columns(
        first.alias("_first"), gap.fill_null(False).alias("_gap"),
        (~gap.fill_null(False) & jump.fill_null(False)).alias("_jump"),
        changed.alias("_split"),
    ).with_columns(
        (pl.col("_first") | pl.col("_gap") | pl.col("_jump") | pl.col("_split")).alias("_brk"),
    ).with_columns(
        pl.when(pl.col("_first")).then(pl.lit("start"))
        .when(pl.col("_gap")).then(pl.lit("gap"))
        .when(pl.col("_jump")).then(pl.lit("jump"))
        .when(pl.col("_split")).then(pl.lit("drill_change"))
        .otherwise(None).alias("break_reason"),
        pl.col("_brk").cast(pl.Int32).cum_sum().over(keys).cast(pl.Int32).alias("seg_idx"),
        ((pl.col("ts") - pl.col("ts").min().over(keys)).dt.total_milliseconds() / 1000.0).alias("t_s"),
    )
    seg_keys = [*keys, "seg_idx"]
    df = df.with_columns(
        pl.int_range(pl.len(), dtype=pl.Int64).over(seg_keys).alias("pos"),
        pl.len().over(seg_keys).cast(pl.Int64).alias("seg_len"),
        pl.when(pl.col("_brk")).then(None).otherwise(pl.col("_dt")).alias("dt_ms"),
        pl.when(pl.col("_brk")).then(None).otherwise(pl.col("_step")).alias("step_yds"),
    )
    stats_brk = df.group_by(keys).agg(
        pl.col("_gap").sum().alias("n_gap_breaks"), pl.col("_jump").sum().alias("n_jump_breaks"),
        pl.col("_split").sum().alias("n_drill_breaks"), pl.col("seg_idx").max().alias("n_segments"),
        pl.len().alias("n_valid_frames"),
    )
    df = df.drop("_dt", "_step", "_first", "_gap", "_jump", "_split", "_brk")
    stats = (stats_raw.join(stats_dup, on=keys, how="left").join(stats_brk, on=keys, how="left")
             .with_columns([pl.col(c).fill_null(0) for c in ("n_dup_ts_removed", "n_gap_breaks", "n_jump_breaks",
                                                             "n_drill_breaks", "n_segments", "n_valid_frames")]))
    return df, stats.sort(keys)
