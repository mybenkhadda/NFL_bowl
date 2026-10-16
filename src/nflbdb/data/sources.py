"""Registry of raw sources.

Column names and order below were CONFIRMED from the header lines of the raw CSVs
(2026-10-08). Everything else in this module is labelled by evidence level:

* ``schema`` — *bronze physical types*. Chosen to be lossless: identifiers/years are Int64,
  every other numeric measure is Float64 (integrality is checked later and cast in silver),
  boolean-looking columns (``TRUE``/``FALSE`` literals) stay Utf8 in bronze and are converted
  in silver only after their distinct values are verified.
* ``grain_candidates`` — HYPOTHESES. :mod:`nflbdb.data.validate` tests them exactly.
* ``FOREIGN_KEYS`` — HYPOTHESES about shared namespaces; tested by validate/relationships.

Nulls are encoded as the literal ``NA`` in the raw files (observed). Some string columns in
``player_play.csv`` also contain empty strings ``""`` (observed in ``receiver_alignment``),
which are distinct from NA and are preserved in bronze.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

I64, F64, STR = pl.Int64, pl.Float64, pl.Utf8


@dataclass(frozen=True)
class SourceSpec:
    name: str
    pattern: str                                   # glob relative to DATA_ROOT
    domain: str                                    # combine | nfl | reference | outcome
    description: str
    schema: dict[str, pl.DataType]                 # bronze physical schema (ordered as header)
    grain_candidates: tuple[tuple[str, ...], ...] = ()
    partition_regex: str | None = None             # extracts a partition value from filename
    partition_col: str | None = None
    is_tracking: bool = False
    time_col: str | None = None
    trajectory_key: tuple[str, ...] = ()
    null_values: tuple[str, ...] = ("NA",)

    @property
    def columns(self) -> list[str]:
        return list(self.schema)

    def matches(self, rel_path: str) -> bool:
        return fnmatch.fnmatch(Path(rel_path).as_posix(), self.pattern)

    def partition_value(self, path: Path | str) -> str | None:
        if not self.partition_regex:
            return None
        m = re.search(self.partition_regex, Path(path).name)
        return m.group(1) if m else None

    def files(self, data_root: Path) -> list[Path]:
        return sorted(p for p in data_root.glob(self.pattern) if p.is_file())


def _schema(ids: list[str], floats: list[str], strs: list[str], order: list[str]) -> dict[str, pl.DataType]:
    """Build an ordered schema and fail loudly if a column was forgotten or duplicated."""
    typed: dict[str, pl.DataType] = {}
    for cols, dt in ((ids, I64), (floats, F64), (strs, STR)):
        for c in cols:
            if c in typed:
                raise ValueError(f"column {c!r} typed twice")
            typed[c] = dt
    missing = [c for c in order if c not in typed]
    extra = [c for c in typed if c not in order]
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={missing} extra={extra}")
    return {c: typed[c] for c in order}


# ----------------------------------------------------------------------------- headers
_PLAYERS = ["nfl_id", "display_name", "draft_year", "nfl_position", "birth_date", "college_name",
            "college_conference", "draft_round", "draft_pick_within_round", "draft_overall_pick"]
_COMBINE_RESULTS = ["draft_year", "nfl_id", "combine_position", "combine_height", "combine_weight",
                    "hand_size", "arm_length", "wing_span", "ten_yd_split", "forty", "vertical",
                    "broad_jump", "three_cone", "short_shuttle", "bench_reps",
                    "ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]
_COMBINE_TRACKING = ["draft_year", "event_id", "nfl_id", "entity_type", "time", "drill_type",
                     "drill_name", "attempt", "x", "y", "s", "a", "dis", "dir"]
_GAME_TRACKING = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a", "dis", "o", "dir", "event"]
_GAMES = ["game_id", "game_key", "season", "season_type", "week", "home_team_abbr", "visitor_team_abbr"]
_CAREER = ["nfl_id", "career_offensive_snaps", "career_defensive_snaps", "career_special_teams_snaps",
           "career_games_active", "career_games_started", "ap_all_pro_1st_team", "ap_all_pro_2nd_team",
           "pro_bowl_original_ballot"]
_PLAYER_PLAY = [
    "game_id", "play_id", "team_abbr", "nfl_id", "lined_up_position", "play_description", "quarter",
    "down", "yards_to_go", "possession_team", "yardline_side", "yardline_number", "game_clock",
    "pre_snap_home_score", "pre_snap_visitor_score", "play_direction",
    "pre_snap_home_team_win_probability", "pre_snap_visitor_team_win_probability",
    "home_team_win_probability_added", "visitor_team_win_probility_added",  # sic: typo in source
    "expected_points_added", "expected_points", "offense_formation", "receiver_alignment",
    "pass_result", "pass_length", "play_nullified_by_penalty", "penalty_yards",
    "pre_penalty_yards_gained", "yards_gained", "team_coverage_man_zone", "team_coverage_type",
    "in_motion_at_ball_snap", "target", "rec_yards", "rush_yards", "yards_after_catch", "fumble",
    "fumble_lost", "route_ran", "separation_at_pass_forward", "cushion", "expected_yards_after_catch",
    "dropback_duration", "extended_sack_allowed", "pass_rushers_encountered",
    "peak_pressure_probability_allowed", "pressure_allowed", "sack_allowed",
    "time_to_pressure_allowed", "quick_pressure", "unblocked_pressure", "blitzing",
    "player_get_off", "sack", "tackle", "assist", "caused_forced_fumble", "recovered_fumble",
    "tackle_for_loss", "time_to_pressure", "time_to_qb_hurry", "coverage_assignment",
    "coverage_assignment_at_snap",
]
# Observed TRUE/FALSE literals (or unknown, all-NA in the inspected rows) -> Utf8 in bronze.
PLAYER_PLAY_BOOLEAN_LIKE = [
    "in_motion_at_ball_snap", "target", "pressure_allowed", "quick_pressure",
    "unblocked_pressure", "blitzing", "tackle_for_loss",
]
_PP_STR = ["team_abbr", "lined_up_position", "play_description", "possession_team", "yardline_side",
           "game_clock", "play_direction", "offense_formation", "receiver_alignment", "pass_result",
           "play_nullified_by_penalty", "team_coverage_man_zone", "team_coverage_type", "route_ran",
           "coverage_assignment", "coverage_assignment_at_snap", *PLAYER_PLAY_BOOLEAN_LIKE]
_PP_IDS = ["game_id", "play_id", "nfl_id"]
_PP_F = [c for c in _PLAYER_PLAY if c not in _PP_STR and c not in _PP_IDS]

# ----------------------------------------------------------------------------- registry
SOURCES: dict[str, SourceSpec] = {
    s.name: s
    for s in [
        SourceSpec(
            name="players", pattern="players.csv", domain="reference",
            description="Player biographical / draft attributes (one row per player — hypothesis).",
            schema=_schema(["nfl_id", "draft_year"],
                           ["draft_round", "draft_pick_within_round", "draft_overall_pick"],
                           ["display_name", "nfl_position", "birth_date", "college_name", "college_conference"],
                           _PLAYERS),
            grain_candidates=(("nfl_id",),),
        ),
        SourceSpec(
            name="combine_results", pattern="combine_results.csv", domain="combine",
            description="Traditional Combine measurements and NGS scores.",
            schema=_schema(["draft_year", "nfl_id"],
                           [c for c in _COMBINE_RESULTS if c not in ("draft_year", "nfl_id", "combine_position")],
                           ["combine_position"], _COMBINE_RESULTS),
            grain_candidates=(("nfl_id",), ("draft_year", "nfl_id")),
        ),
        SourceSpec(
            name="combine_tracking", pattern="combine_tracking.csv", domain="combine",
            description="Combine drill tracking at frame level (10 Hz timestamps observed in sample).",
            schema=_schema(["draft_year", "nfl_id", "attempt"], ["x", "y", "s", "a", "dis", "dir"],
                           ["event_id", "entity_type", "time", "drill_type", "drill_name"], _COMBINE_TRACKING),
            # Verified 2026-10-08: (event_id, nfl_id, attempt, time) has 32,095 duplicate rows = exactly the
            # BALL rows, which carry the athlete's nfl_id. entity_type completes the grain.
            grain_candidates=(("event_id", "nfl_id", "attempt", "time"),
                              ("event_id", "nfl_id", "attempt", "entity_type", "time")),
            is_tracking=True, time_col="time",
            trajectory_key=("event_id", "nfl_id", "attempt"),
        ),
        SourceSpec(
            name="game_tracking", pattern="game_tracking_*.csv", domain="nfl",
            description="NFL game tracking, one file per season (filename suffix). No frame_id column: "
                        "frames must be derived from `time`.",
            schema=_schema(["game_id", "play_id", "nfl_id"], ["x", "y", "s", "a", "dis", "o", "dir"],
                           ["time", "event"], _GAME_TRACKING),
            grain_candidates=(("game_id", "play_id", "nfl_id", "time"),),
            partition_regex=r"game_tracking_(\d{4})", partition_col="season",
            is_tracking=True, time_col="time",
            trajectory_key=("game_id", "play_id", "nfl_id"),
        ),
        SourceSpec(
            name="games", pattern="games.csv", domain="nfl",
            description="Game metadata.",
            schema=_schema(["game_id", "game_key", "season", "week"], [],
                           ["season_type", "home_team_abbr", "visitor_team_abbr"], _GAMES),
            grain_candidates=(("game_id",), ("game_key",)),
        ),
        SourceSpec(
            name="player_play", pattern="player_play.csv", domain="nfl",
            description="Player x play rows; play-level context is denormalised onto every player row "
                        "(verified per column by silver.PlayerPlaySplitter).",
            schema=_schema(_PP_IDS, _PP_F, _PP_STR, _PLAYER_PLAY),
            grain_candidates=(("game_id", "play_id", "nfl_id"),),
        ),
        SourceSpec(
            name="player_career_successes", pattern="player_career_successes.csv", domain="outcome",
            description="Career-level NFL outcomes (snaps, starts, honours). Candidate TARGETS — "
                        "never use as predictors.",
            schema=_schema(["nfl_id"], [c for c in _CAREER if c != "nfl_id"], [], _CAREER),
            grain_candidates=(("nfl_id",),),
        ),
    ]
}


@dataclass(frozen=True)
class ForeignKey:
    child: str
    child_cols: tuple[str, ...]
    parent: str
    parent_cols: tuple[str, ...]
    severity: str = "HIGH"
    child_filter: str = ""          # SQL predicate applied to the child side
    note: str = ""


FOREIGN_KEYS: list[ForeignKey] = [
    ForeignKey("combine_results", ("nfl_id",), "players", ("nfl_id",)),
    ForeignKey("combine_tracking", ("nfl_id",), "players", ("nfl_id",),
               child_filter="entity_type = 'PLAYER' AND nfl_id IS NOT NULL"),
    ForeignKey("combine_tracking", ("nfl_id",), "combine_results", ("nfl_id",), severity="MEDIUM",
               child_filter="entity_type = 'PLAYER' AND nfl_id IS NOT NULL",
               note="Tracked athletes without traditional results limit baseline comparisons."),
    ForeignKey("game_tracking", ("game_id",), "games", ("game_id",), severity="CRITICAL"),
    ForeignKey("game_tracking", ("game_id", "play_id"), "player_play", ("game_id", "play_id"), severity="HIGH"),
    ForeignKey("game_tracking", ("game_id", "play_id", "nfl_id"), "player_play", ("game_id", "play_id", "nfl_id"),
               severity="MEDIUM", child_filter="nfl_id IS NOT NULL",
               note="Tracked players absent from player_play lose play context."),
    ForeignKey("game_tracking", ("nfl_id",), "players", ("nfl_id",), severity="MEDIUM",
               child_filter="nfl_id IS NOT NULL",
               note="Expected to fail partially if players.csv only covers drafted cohorts."),
    ForeignKey("player_play", ("game_id",), "games", ("game_id",), severity="CRITICAL"),
    ForeignKey("player_play", ("nfl_id",), "players", ("nfl_id",), severity="MEDIUM"),
    ForeignKey("player_career_successes", ("nfl_id",), "players", ("nfl_id",)),
]

# Identifier columns that may link tables (used by inventory to propose relationships).
LINK_COLUMNS = ("nfl_id", "game_id", "play_id", "event_id", "game_key", "draft_year", "season")


def match_source(rel_path: str) -> SourceSpec | None:
    for spec in SOURCES.values():
        if spec.matches(rel_path):
            return spec
    return None
