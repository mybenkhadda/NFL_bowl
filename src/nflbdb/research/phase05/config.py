"""Phase 05 · Frontier research parameters. Every field is documented in ``DOCS`` as ``established`` (reused
project value), ``proposed`` (uncalibrated default — the continuous evidence is always exported, so a cut can be
changed without recomputation) or ``execution`` (cost knob, no scientific meaning).

Stage P0 (decision-problem feasibility audit) uses NO Combine feature: it is a census of what the NFL data can
support, so it does not touch the Phase 04 gate ("no Combine-to-NFL statistics before both gates pass").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import polars as pl

ESTABLISHED, PROPOSED, EXECUTION = "established", "proposed", "execution"
PHASE05_VERSION = "1"
PHASE = "phase_05"
VERDICTS = ("FEASIBLE", "LIMITED", "INFEASIBLE")

# vendor player_play fields by decision role (audit §2) — names verified against reports/tables/schema_catalog.csv
STATE_FIELDS = ("down", "yards_to_go", "yardline_number", "quarter", "game_clock", "pre_snap_home_team_win_probability",
                "offense_formation", "receiver_alignment", "team_coverage_man_zone", "team_coverage_type",
                "in_motion_at_ball_snap", "lined_up_position", "cushion")
ACTION_FIELDS = ("route_ran", "coverage_assignment", "coverage_assignment_at_snap", "blitzing")
OUTCOME_FIELDS = ("expected_points_added", "home_team_win_probability_added", "player_get_off", "time_to_pressure",
                  "time_to_pressure_allowed", "pressure_allowed", "peak_pressure_probability_allowed",
                  "separation_at_pass_forward", "target", "rec_yards", "yards_after_catch",
                  "expected_yards_after_catch", "tackle", "sack")


@dataclass(frozen=True)
class DecisionAuditConfig:
    """Stage P0 — data & decision feasibility audit (census; no inference, no Combine features)."""

    seasons: tuple[int, ...] | None = None          # None = all seasons; a subset → _sample namespace
    season_types: tuple[str, ...] = ("REG",)        # analysis population; the entity census reports every type
    state_fields: tuple[str, ...] = STATE_FIELDS
    action_fields: tuple[str, ...] = ACTION_FIELDS
    outcome_fields: tuple[str, ...] = OUTCOME_FIELDS
    min_obs_per_athlete: int = 20                   # athlete "has" a field if ≥ this many non-null player-plays
    exposure_thresholds: tuple[int, ...] = (25, 50, 100, 200, 400)
    min_level_count: int = 20                       # action level counted as "used" by an athlete at ≥ this many plays
    max_players_hist: int = 12                      # players-per-play histogram: last bin = "≥ max"


@dataclass(frozen=True)
class StreamFeasibilityConfig:
    """Rules that turn the P0 census into FEASIBLE / LIMITED / INFEASIBLE per frontier stream."""

    min_plays_envelope: int = 100                   # tracked plays for a per-athlete movement envelope
    min_athletes_twin: int = 150                    # athletes meeting min_plays_envelope (all positions)
    min_athletes_per_position: int = 30             # … per position group
    min_positions_twin: int = 3                     # position groups meeting min_athletes_per_position
    min_share_ball: float = 0.90                    # plays with a tracked football → interaction counterfactuals
    min_share_both_sides: float = 0.25              # plays with ≥ 1 offense AND ≥ 1 defense tracked athlete
    min_action_coverage: float = 0.50               # labelled share of applicable player-plays for an action field
    min_athletes_varying: int = 30                  # athletes using ≥ 2 action levels (within-athlete variation)
    min_pair_plays: int = 500                       # co-tracked plays for an opposing position pair
    min_distinct_pairs: int = 30                    # distinct athlete pairs for that position pair
    full_team_tracked: int = 22                     # median tracked per play needed for full multi-agent models
    min_athletes_outcome: int = 30                  # athletes with ≥ min_obs_per_athlete for an outcome metric
    outcome_positions: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("player_get_off", ("DL", "LB")), ("time_to_pressure", ("DL", "LB")),
        ("separation_at_pass_forward", ("WR", "TE", "RB")), ("cushion", ("WR", "TE", "DB")))
    action_positions: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("route_ran", ("WR", "TE", "RB")), ("coverage_assignment", ("DB", "LB")),
        ("coverage_assignment_at_snap", ("DB", "LB")), ("blitzing", ("DB", "LB", "DL")))


@dataclass(frozen=True)
class SampleFrameConfig:
    """Stage P0b — analysis sample frame (participation counts only; no Combine value, no Combine→NFL statistic)."""

    min_plays_grid: tuple[int, ...] = (0, 50, 100, 200, 400)   # 0 = any play in the population
    outcome_obs_grid: tuple[int, ...] = (20, 50, 100, 200)
    min_attempts_repeat: int = 2                    # Combine attempts needed for a repeatability estimate
    official_results: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("FORTY_YARD_DASH", ("forty", "ten_yd_split")), ("SHORT_SHUTTLE", ("short_shuttle",)),
        ("THREE_CONE_DRILL", ("three_cone",)))
    relevant_pairs: tuple[tuple[str, str], ...] = (("OL", "DL"), ("WR", "DB"), ("TE", "DB"))
    play_type_field: str = "pass_result"
    alpha: float = 0.05
    power: float = 0.80
    figure_min_plays: int = 100                     # display only: threshold shown in figure P0b_01
    figure_max_drills: int = 14


@dataclass(frozen=True)
class Phase05Config:
    version: str = PHASE05_VERSION
    seed: int = 20261011
    audit: DecisionAuditConfig = field(default_factory=DecisionAuditConfig)
    streams: StreamFeasibilityConfig = field(default_factory=StreamFeasibilityConfig)
    frame: SampleFrameConfig = field(default_factory=SampleFrameConfig)

    @property
    def sampled(self) -> bool:
        return self.audit.seasons is not None

    def section(self, *names: str) -> dict[str, Any]:
        return {"version": self.version, "seed": self.seed, **{n: asdict(getattr(self, n)) for n in names}}


_R = "the census value is exported; the cut can be changed without recomputation"
DOCS: dict[str, dict[str, tuple[str, str, str]]] = {
    "audit": {
        "seasons": ("seasons", EXECUTION, "subset = smoke run in the _sample namespace"),
        "season_types": ("games.season_type", ESTABLISHED, "REG only, as in the Phase 03 NFL windows (PlayWindowConfig)"),
        "state_fields": ("columns", ESTABLISHED, "player_play columns (schema_catalog.csv) with a state role (audit §2.1)"),
        "action_fields": ("columns", ESTABLISHED, "player_play columns with an action role (audit §2.2); vendor labels, "
                                                  "never inferred intent"),
        "outcome_fields": ("columns", ESTABLISHED, "player_play columns with an outcome role (audit §2.3)"),
        "min_obs_per_athlete": ("player-plays", PROPOSED, "smallest per-athlete sample for a stable median; " + _R),
        "exposure_thresholds": ("tracked plays", PROPOSED, "grid for the exposure curve; Phase 04 M2 decides the "
                                                           "reliability-based minimum"),
        "min_level_count": ("plays", PROPOSED, "an athlete 'uses' an action level only with repeated evidence"),
        "max_players_hist": ("athletes", EXECUTION, "histogram display cap"),
    },
    "streams": {
        "min_plays_envelope": ("tracked plays", PROPOSED, "placeholder until Phase 04 M2 reliability-vs-plays; " + _R),
        "min_athletes_twin": ("athletes", PROPOSED, "grouped 5-fold CV keeps ≥ 30 test athletes per fold"),
        "min_athletes_per_position": ("athletes", PROPOSED, "clusters needed for a cluster-bootstrap interval"),
        "min_positions_twin": ("position groups", PROPOSED, "a twin covering < 3 groups is position-specific, not general"),
        "min_share_ball": ("share of plays", PROPOSED, "ball geometry needed on almost every play for interaction "
                                                      "counterfactuals and reward shaping"),
        "min_share_both_sides": ("share of plays", PROPOSED, _R),
        "min_action_coverage": ("share", PROPOSED, _R),
        "min_athletes_varying": ("athletes", PROPOSED, "positivity / overlap proxy for any policy evaluation"),
        "min_pair_plays": ("plays", PROPOSED, _R),
        "min_distinct_pairs": ("athlete pairs", PROPOSED, "matchup results must not rest on a handful of pairs"),
        "full_team_tracked": ("athletes", ESTABLISHED, "11 offense + 11 defense"),
        "min_athletes_outcome": ("athletes", PROPOSED, "clusters needed for a cluster-bootstrap interval"),
        "outcome_positions": ("field → positions", PROPOSED, "vendor field applicability (verify in P0 coverage)"),
        "action_positions": ("field → positions", PROPOSED, "vendor field applicability (verify in P0 coverage)"),
    },
    "frame": {
        "min_plays_grid": ("tracked plays", PROPOSED, "grid, not a cut: the minimum comes from the S2 reliability curve"),
        "outcome_obs_grid": ("non-null player-plays", PROPOSED, "grid for per-athlete outcome volume"),
        "min_attempts_repeat": ("attempts", ESTABLISHED, "Phase 04 eligibility: ≥ 2 attempts for repeatability / ICC"),
        "official_results": ("drill → combine_results columns", ESTABLISHED, "official measurements matching a tracked "
                                                                             "drill (baseline model b of E05-02)"),
        "relevant_pairs": ("offense × defense groups", PROPOSED, "pairs that interact by role (P0 review: the P0 rule "
                                                                 "picked OL × DB by count alone)"),
        "play_type_field": ("column", ESTABLISHED, "silver.plays column used for the play-type census"),
        "alpha": ("probability", ESTABLISHED, "two-sided type-I error of the planning calculation"),
        "power": ("probability", ESTABLISHED, "conventional planning power"),
        "figure_min_plays": ("tracked plays", EXECUTION, "threshold displayed in figure P0b_01"),
        "figure_max_drills": ("drills", EXECUTION, "display cap"),
    },
}


def threshold_table(cfg: Phase05Config | None = None) -> pl.DataFrame:
    cfg = cfg or Phase05Config()
    rows = []
    for section, docs in DOCS.items():
        sub = getattr(cfg, section)
        for f in fields(sub):
            unit, source, why = docs[f.name]
            rows.append({"study_section": section, "name": f.name, "value": str(getattr(sub, f.name)), "unit": unit,
                         "provenance": source, "rationale": why})
    return pl.DataFrame(rows, schema={c: pl.String for c in ("study_section", "name", "value", "unit", "provenance",
                                                              "rationale")})
