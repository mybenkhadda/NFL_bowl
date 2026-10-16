"""Stage P0 — data & decision feasibility audit (Phase 05 · experiment E05-00).

Question     Which states, actions, interactions and outcomes are actually OBSERVABLE in the competition
             data, and which frontier streams (A capability twin · B counterfactual engine · C offline RL ·
             D multi-agent · E decision metrics) can they support?
Population   every tracked player-play (one athlete on one play with ≥ 1 tracking frame), all seasons.
             Analysis tables use ``season_types`` (default REG, as in the Phase 03 NFL windows); the entity
             census reports every season type. Grain: player-play; unit: athlete.
Method       A census — counts and shares only. NO inference, NO model and NO Combine feature, so this stage
             does not touch the Phase 04 gate ("no Combine-to-NFL statistics before both gates pass").
Definitions  side            offense if player_play.team_abbr = plays.possession_team, defense if both are
                             known and differ, else unknown (athlete absent from player_play)
             co-tracked      an offensive and a defensive athlete with tracking on the SAME play. This is
                             availability, NOT a matchup: nobody is assumed to cover or block anybody
             action level    a vendor label value (route_ran, coverage_assignment, …). A label is what the
                             vendor recorded, never the athlete's intent or the coach's call
             varying athlete an athlete with ≥ 2 action levels, each on ≥ ``min_level_count`` plays (the
                             within-athlete variation any policy evaluation would need)
Verdicts     ``stream_feasibility`` applies the PROPOSED cuts of ``StreamFeasibilityConfig`` to the census.
             Every criterion is exported with its observed value and threshold, so a cut can be changed
             without recomputation. A verdict is a screening aid for the user's review, not a finding.
Caveat       Shares over plays carry Wilson intervals for scale only: plays are nested in games and
             athletes, so the interval is not an inferential statement.

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.phase05 --only P0
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.config import Phase01Config, duckdb_connect
from ...data.frames import query_df
from ...data.tables import LayerRefs
from ...data.utils import get_logger
from ..contracts import TableContract, cast_keys
from ..studies.stats import wilson_ci
from ..studies.store import StudyStore
from .config import PHASE, VERDICTS, DecisionAuditConfig, Phase05Config

log = get_logger(__name__)
STUDY = "P0"
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
ALL = "ALL"
PLAY = ("game_id", "play_id")
ROLES = ("state", "action", "outcome")

CENSUS = TableContract(
    "P0_entity_census", "Tracking rows, football rows, plays and tracked athletes per season × season type.",
    {"season": I, "season_type": S, "in_population": B, "n_rows": I, "n_football_rows": I, "n_player_rows": I,
     "n_games": I, "n_plays": I, "n_plays_with_football": I, "share_plays_with_football": F,
     "n_tracked_player_plays": I, "n_athletes": I, "n_in_player_play": I, "share_in_player_play": F},
    key=("season", "season_type"), area="descriptive",
    docs={"n_football_rows": "silver is_football = raw nfl_id IS NULL",
          "in_population": "season type used by every other P0 table",
          "n_in_player_play": "tracked player-plays that also have a player_play row (vendor context)"})
PER_PLAY = TableContract(
    "P0_players_per_play", "Distribution of tracked athletes per play (last bin = ≥ max_players_hist).",
    {"season": S, "n_tracked": I, "is_capped": B, "n_plays": I, "share_plays": F},
    key=("season", "n_tracked"), area="descriptive")
SIDE = TableContract(
    "P0_side_coverage", "Plays by which side of the ball has ≥ 1 tracked athlete (metric M13).",
    {"season": S, "n_plays": I, "n_both_sides": I, "n_offense_only": I, "n_defense_only": I, "n_unknown_side": I,
     "share_both_sides": F, "ci_lower": F, "ci_upper": F, "median_tracked": F, "mean_tracked": F},
    key=("season",), area="descriptive",
    docs={"ci_lower": "Wilson 95 % over plays — scale only, plays are not independent"})
PAIRS = TableContract(
    "P0_opponent_pairs", "Co-tracked offensive × defensive athletes by position-group pair (availability, not matchups).",
    {"off_position": S, "def_position": S, "n_plays": I, "n_pair_plays": I, "n_games": I, "n_distinct_pairs": I,
     "n_pairs_min_obs": I, "n_off_athletes": I, "n_def_athletes": I, "median_plays_per_pair": F,
     "max_plays_per_pair": I},
    key=("off_position", "def_position"), area="descriptive",
    docs={"n_pairs_min_obs": "distinct athlete pairs co-tracked on ≥ min_obs_per_athlete plays"})
COVERAGE = TableContract(
    "P0_field_coverage", "Non-null coverage of vendor state / action / outcome fields by position group.",
    {"role": S, "field": S, "source_table": S, "position_group": S, "n_player_plays": I, "n_non_null": I,
     "share_non_null": F, "n_true": I, "n_athletes": I, "n_athletes_any": I, "n_athletes_min_obs": I},
    key=("role", "field", "position_group"), area="descriptive",
    docs={"source_table": "player_play · plays (play-level, silver split) · absent (not in the warehouse)",
          "n_true": "TRUE count for boolean fields (non-null ≠ applicable), else null",
          "n_athletes_min_obs": "athletes with ≥ min_obs_per_athlete non-null player-plays"})
EXPOSURE_DS = TableContract(
    "athlete_exposure", "One row per tracked athlete: NFL tracking exposure in the P0 population.",
    {"nfl_id": I, "position_group": S, "cohort": I, "n_tracked_plays": I, "n_games": I, "n_seasons": I,
     "first_season": I, "last_season": I, "n_frames": I, "n_in_player_play": I, "n_offense_plays": I,
     "n_defense_plays": I},
    key=("nfl_id",), kind="dataset", area="decision_audit",
    docs={"position_group": "gold.dim_athlete (Combine position, else NFL position)",
          "n_tracked_plays": "plays with ≥ 1 tracking frame in the configured season types"})
EXPOSURE = TableContract(
    "P0_exposure_summary", "Athletes with at least N tracked plays, by position group.",
    {"position_group": S, "threshold": I, "n_athletes": I, "n_athletes_ge": I, "share_ge": F, "p25_plays": F,
     "median_plays": F, "p75_plays": F},
    key=("position_group", "threshold"), area="descriptive")
OVERLAP = TableContract(
    "P0_action_overlap", "Vendor action labels: vocabulary, coverage and within-athlete variation by position group.",
    {"field": S, "position_group": S, "n_player_plays": I, "n_labelled": I, "coverage": F, "n_levels": I,
     "n_levels_supported": I, "top_level_share": F, "level_entropy_norm": F, "n_athletes_labelled": I,
     "n_athletes_varying": I, "median_levels_per_athlete": F},
    key=("field", "position_group"), area="descriptive",
    docs={"n_levels_supported": "levels with ≥ min_level_count labelled player-plays",
          "level_entropy_norm": "Shannon entropy of the pooled level distribution ÷ ln(n_levels); 1 = uniform",
          "n_athletes_labelled": "athletes with ≥ min_obs_per_athlete labelled player-plays",
          "n_athletes_varying": "athletes with ≥ 2 levels, each on ≥ min_level_count plays"})
FEAS = TableContract(
    "P0_stream_feasibility", "Rule-based screening verdict per frontier stream (PROPOSED thresholds).",
    {"stream": S, "criterion": S, "observed": F, "threshold": F, "passed": B, "basis": S, "verdict": S, "rule": S},
    key=("stream", "criterion"), area="descriptive",
    docs={"observed": "census value compared with `threshold` (observed ≥ threshold passes)",
          "verdict": "stream-level, repeated on each criterion row", "rule": "how criteria combine into the verdict"})
CONTRACTS = (CENSUS, PER_PLAY, SIDE, PAIRS, COVERAGE, EXPOSURE_DS, EXPOSURE, OVERLAP, FEAS)


def _as(df: pl.DataFrame, c: TableContract) -> pl.DataFrame:
    return df.select([pl.col(k).cast(t) for k, t in c.columns.items()])


def _frame(rows: list[dict], c: TableContract) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=c.columns) if rows else c.empty()


def _positions(df: pl.DataFrame) -> list[str]:
    return sorted(df["position_group"].unique().to_list()) + [ALL]


def audit_fields(cfg: Phase05Config) -> dict[str, tuple[str, ...]]:
    """Vendor fields by role; fields named only in the stream rules are audited too."""
    a, s = cfg.audit, cfg.streams
    extra_out = tuple(f for f, _ in s.outcome_positions if f not in a.outcome_fields and f not in a.state_fields)
    extra_act = tuple(f for f, _ in s.action_positions if f not in a.action_fields)
    return {"state": a.state_fields, "action": a.action_fields + extra_act, "outcome": a.outcome_fields + extra_out}


# ============================================================================== base frame
def assemble_base(fact: pl.DataFrame, player_play: pl.DataFrame, plays: pl.DataFrame, games: pl.DataFrame,
                  athletes: pl.DataFrame, fields: Sequence[str]) -> tuple[pl.DataFrame, dict[str, str]]:
    """One row per tracked player-play with side, position group and every audited vendor field.

    ``fact``: (game_id, play_id, nfl_id, season, n_frames) from gold.fact_nfl_play_participation. Returns the
    frame and ``{field: 'player_play' | 'plays' | 'absent'}``; absent fields become all-null columns so the
    coverage table reports them instead of failing.
    """
    fields = list(dict.fromkeys(fields))
    reserved = {*PLAY, "nfl_id", "team_abbr", "possession_team", "season", "season_type"}
    source = {f: "player_play" if f in player_play.columns else "plays" if f in plays.columns else "absent"
              for f in fields}
    pp = cast_keys(player_play.select(*PLAY, "nfl_id", "team_abbr", pl.lit(True).alias("in_player_play"),
                                      *[f for f in fields if source[f] == "player_play" and f not in reserved]))
    pl_ = cast_keys(plays.select(*PLAY, "possession_team",
                                 *[f for f in fields if source[f] == "plays" and f not in reserved]))
    base = (cast_keys(fact.select(*PLAY, "nfl_id", "season", pl.col("n_frames").cast(pl.Int64)))
            .join(pp, on=[*PLAY, "nfl_id"], how="left")
            .join(pl_, on=list(PLAY), how="left")
            .join(cast_keys(games.select("game_id", pl.col("season_type").cast(pl.String))), on="game_id", how="left")
            .join(cast_keys(athletes.select("nfl_id", pl.col("position_group").cast(pl.String),
                                            pl.col("cohort").cast(pl.Int64))), on="nfl_id", how="left"))
    team, poss = pl.col("team_abbr"), pl.col("possession_team")
    base = base.with_columns(
        pl.col("in_player_play").fill_null(False), pl.col("season_type").fill_null("UNKNOWN"),
        pl.col("position_group").fill_null("UNKNOWN"),
        pl.when(team.is_null() | poss.is_null()).then(pl.lit("unknown")).when(team == poss).then(pl.lit("offense"))
        .otherwise(pl.lit("defense")).alias("side"),
        *[pl.lit(None, dtype=pl.String).alias(f) for f in fields if source[f] == "absent"])
    return base.drop("team_abbr", "possession_team"), source


def frame_census(con, refs: LayerRefs, seasons: tuple[int, ...] | None) -> pl.DataFrame:
    """Exact frame-level counts from silver.game_tracking (the only P0 step that scans frames)."""
    where = f"WHERE season IN ({', '.join(str(int(s)) for s in seasons)})" if seasons else ""
    df = query_df(con, f"""
        WITH p AS (SELECT season, game_id, play_id, count(*) AS n_rows, count_if(is_football) AS n_fb
                   FROM {refs.sql('silver', 'game_tracking')} {where} GROUP BY ALL)
        SELECT CAST(p.season AS BIGINT) AS season, coalesce(g.season_type, 'UNKNOWN') AS season_type,
               CAST(sum(p.n_rows) AS BIGINT) AS n_rows, CAST(sum(p.n_fb) AS BIGINT) AS n_football_rows,
               CAST(count(DISTINCT p.game_id) AS BIGINT) AS n_games, CAST(count(*) AS BIGINT) AS n_plays,
               CAST(count_if(p.n_fb > 0) AS BIGINT) AS n_plays_with_football
        FROM p LEFT JOIN {refs.sql('silver', 'games')} g ON g.game_id = p.game_id
        GROUP BY ALL ORDER BY 1, 2
    """)
    return df.with_columns(pl.col("season_type").cast(pl.String),
                           pl.exclude("season_type").cast(pl.Int64))


# ============================================================================== pure census tables
def entity_census(frames: pl.DataFrame, base: pl.DataFrame, season_types: Sequence[str]) -> pl.DataFrame:
    by = ["season", "season_type"]
    b = base.group_by(by).agg(pl.len().alias("n_tracked_player_plays"), pl.col("nfl_id").n_unique().alias("n_athletes"),
                              pl.col("in_player_play").sum().alias("n_in_player_play"))
    counts = [c for c, t in CENSUS.columns.items() if t == I and c not in ("season", "n_player_rows")]
    out = frames.join(b, on=by, how="full", coalesce=True).with_columns(pl.col(counts).fill_null(0))
    ratio = lambda num, den: pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den))  # noqa: E731
    out = out.with_columns(
        pl.col("season_type").is_in(list(season_types)).alias("in_population"),
        (pl.col("n_rows") - pl.col("n_football_rows")).alias("n_player_rows"),
        ratio("n_plays_with_football", "n_plays").alias("share_plays_with_football"),
        ratio("n_in_player_play", "n_tracked_player_plays").alias("share_in_player_play"))
    return _as(out, CENSUS).sort(by)


def _per_play(pop: pl.DataFrame) -> pl.DataFrame:
    per = pop.group_by("season", *PLAY).agg(
        pl.len().alias("n"), (pl.col("side") == "offense").sum().alias("n_off"),
        (pl.col("side") == "defense").sum().alias("n_def")).with_columns(pl.col("season").cast(pl.String))
    return pl.concat([per, per.with_columns(pl.lit(ALL).alias("season"))])


def players_per_play(pop: pl.DataFrame, cfg: DecisionAuditConfig) -> pl.DataFrame:
    cap = cfg.max_players_hist
    out = (_per_play(pop).with_columns(pl.col("n").clip(upper_bound=cap).alias("n_tracked"))
           .group_by("season", "n_tracked").agg(pl.len().alias("n_plays"))
           .with_columns((pl.col("n_plays") / pl.col("n_plays").sum().over("season")).alias("share_plays"),
                         (pl.col("n_tracked") >= cap).alias("is_capped")))
    return _as(out, PER_PLAY).sort("season", "n_tracked")


def side_coverage(pop: pl.DataFrame) -> pl.DataFrame:
    off, de = pl.col("n_off") > 0, pl.col("n_def") > 0
    out = _per_play(pop).group_by("season").agg(
        pl.len().alias("n_plays"), (off & de).sum().alias("n_both_sides"), (off & ~de).sum().alias("n_offense_only"),
        (~off & de).sum().alias("n_defense_only"), (~off & ~de).sum().alias("n_unknown_side"),
        pl.col("n").median().alias("median_tracked"), pl.col("n").mean().alias("mean_tracked")).sort("season")
    if not out.height:
        return SIDE.empty()
    lo, hi = wilson_ci(out["n_both_sides"].to_numpy(), out["n_plays"].to_numpy())
    out = out.with_columns((pl.col("n_both_sides") / pl.col("n_plays")).alias("share_both_sides"),
                           pl.Series("ci_lower", np.atleast_1d(lo), dtype=pl.Float64),
                           pl.Series("ci_upper", np.atleast_1d(hi), dtype=pl.Float64))
    return _as(out, SIDE)


def opponent_pairs(pop: pl.DataFrame, cfg: DecisionAuditConfig) -> pl.DataFrame:
    def one(side: str, tag: str) -> pl.DataFrame:
        return pop.filter(pl.col("side") == side).select(*PLAY, pl.col("nfl_id").alias(f"{tag}_id"),
                                                         pl.col("position_group").alias(f"{tag}_position"))

    pairs = one("offense", "off").join(one("defense", "def"), on=list(PLAY), how="inner")
    if not pairs.height:
        return PAIRS.empty()
    by = ["off_position", "def_position"]
    plays = pairs.group_by(by).agg(
        pl.struct(*PLAY).n_unique().alias("n_plays"), pl.len().alias("n_pair_plays"),
        pl.col("game_id").n_unique().alias("n_games"), pl.col("off_id").n_unique().alias("n_off_athletes"),
        pl.col("def_id").n_unique().alias("n_def_athletes"))
    per_pair = (pairs.group_by(*by, "off_id", "def_id").agg(pl.len().alias("k")).group_by(by).agg(
        pl.len().alias("n_distinct_pairs"), (pl.col("k") >= cfg.min_obs_per_athlete).sum().alias("n_pairs_min_obs"),
        pl.col("k").median().alias("median_plays_per_pair"), pl.col("k").max().alias("max_plays_per_pair")))
    return _as(plays.join(per_pair, on=by, how="left"), PAIRS).sort(by)


def field_coverage(pop: pl.DataFrame, roles: dict[str, tuple[str, ...]], source: dict[str, str],
                   cfg: DecisionAuditConfig) -> pl.DataFrame:
    fields = list(dict.fromkeys(f for r in ROLES for f in roles.get(r, ())))
    if not pop.height or not fields:
        return COVERAGE.empty()
    bools = [f for f in fields if pop.schema[f] == pl.Boolean]
    per = pop.group_by("position_group", "nfl_id").agg(
        pl.len().alias("_n"), *[pl.col(f).is_not_null().sum().alias(f) for f in fields],
        *[pl.col(f).sum().alias(f"_true_{f}") for f in bools])
    rows = []
    for pos in _positions(pop):
        g = per if pos == ALL else per.filter(pl.col("position_group") == pos)
        n_pp = int(g["_n"].sum())
        for role in ROLES:
            for f in roles.get(role, ()):
                k = g[f].to_numpy()
                rows.append({"role": role, "field": f, "source_table": source.get(f, "absent"), "position_group": pos,
                             "n_player_plays": n_pp, "n_non_null": int(k.sum()),
                             "share_non_null": float(k.sum() / n_pp) if n_pp else None,
                             "n_true": int(g[f"_true_{f}"].sum()) if f in bools else None,
                             "n_athletes": g["nfl_id"].n_unique(), "n_athletes_any": int((k > 0).sum()),
                             "n_athletes_min_obs": int((k >= cfg.min_obs_per_athlete).sum())})
    return _frame(rows, COVERAGE).sort("role", "field", "position_group")


def athlete_exposure(pop: pl.DataFrame) -> pl.DataFrame:
    out = pop.group_by("nfl_id").agg(
        pl.col("position_group").first(), pl.col("cohort").first(), pl.len().alias("n_tracked_plays"),
        pl.col("game_id").n_unique().alias("n_games"), pl.col("season").n_unique().alias("n_seasons"),
        pl.col("season").min().alias("first_season"), pl.col("season").max().alias("last_season"),
        pl.col("n_frames").sum().alias("n_frames"), pl.col("in_player_play").sum().alias("n_in_player_play"),
        (pl.col("side") == "offense").sum().alias("n_offense_plays"),
        (pl.col("side") == "defense").sum().alias("n_defense_plays"))
    return _as(out, EXPOSURE_DS).sort("nfl_id")


def exposure_summary(exposure: pl.DataFrame, cfg: DecisionAuditConfig) -> pl.DataFrame:
    rows = []
    for pos in (_positions(exposure) if exposure.height else []):
        n = (exposure if pos == ALL else exposure.filter(pl.col("position_group") == pos))["n_tracked_plays"].to_numpy()
        q25, q50, q75 = (float(v) for v in np.percentile(n, [25, 50, 75]))
        for thr in cfg.exposure_thresholds:
            ge = int((n >= thr).sum())
            rows.append({"position_group": pos, "threshold": int(thr), "n_athletes": int(n.size), "n_athletes_ge": ge,
                         "share_ge": ge / n.size, "p25_plays": q25, "median_plays": q50, "p75_plays": q75})
    return _frame(rows, EXPOSURE)


def action_overlap(pop: pl.DataFrame, fields: Sequence[str], cfg: DecisionAuditConfig) -> pl.DataFrame:
    rows = []
    for f in (fields if pop.height else ()):
        lab = pop.select("position_group", "nfl_id", pl.col(f).cast(pl.String).alias("_a"))
        for pos in _positions(pop):
            g = lab if pos == ALL else lab.filter(pl.col("position_group") == pos)
            lv = g.drop_nulls("_a")
            k = lv.group_by("_a").agg(pl.len().alias("k"))["k"].to_numpy().astype(float)
            n_lab = int(k.sum())
            p = k / n_lab if n_lab else k
            ath = (lv.group_by("nfl_id", "_a").agg(pl.len().alias("k")).group_by("nfl_id").agg(
                pl.col("k").sum().alias("n"), (pl.col("k") >= cfg.min_level_count).sum().alias("lv")))
            labelled = ath.filter(pl.col("n") >= cfg.min_obs_per_athlete)
            rows.append({
                "field": f, "position_group": pos, "n_player_plays": g.height, "n_labelled": n_lab,
                "coverage": n_lab / g.height if g.height else None, "n_levels": int(k.size),
                "n_levels_supported": int((k >= cfg.min_level_count).sum()),
                "top_level_share": float(p.max()) if n_lab else None,
                "level_entropy_norm": float(-(p * np.log(p)).sum() / np.log(k.size)) if k.size > 1 else None,
                "n_athletes_labelled": labelled.height, "n_athletes_varying": int((ath["lv"] >= 2).sum()),
                "median_levels_per_athlete": float(labelled["lv"].median()) if labelled.height else None})
    return _frame(rows, OVERLAP).sort("field", "position_group")


# ============================================================================== stream verdicts
def _pooled(table: pl.DataFrame, field: str, positions: Sequence[str], cols: Sequence[str]) -> dict[str, float] | None:
    """Sum ``cols`` of a per-position census table over the position groups a field applies to."""
    t = table.filter((pl.col("field") == field) & pl.col("position_group").is_in(list(positions)))
    t = t.unique(subset=["field", "position_group"], keep="first")
    return {c: float(t[c].sum()) for c in cols} if t.height else None


def stream_feasibility(census: pl.DataFrame, side: pl.DataFrame, pairs: pl.DataFrame, coverage: pl.DataFrame,
                       exposure: pl.DataFrame, overlap: pl.DataFrame, cfg: Phase05Config) -> pl.DataFrame:
    """Apply the PROPOSED ``StreamFeasibilityConfig`` cuts to the census (observed ≥ threshold passes)."""
    s, rows = cfg.streams, []

    def crit(stream: str, name: str, observed: float | None, threshold: float, basis: str = "") -> bool:
        ok = observed is not None and bool(np.isfinite(observed)) and observed >= threshold
        rows.append({"stream": stream, "criterion": name, "observed": None if observed is None else float(observed),
                     "threshold": float(threshold), "passed": ok, "basis": basis})
        return ok

    pop = census.filter(pl.col("in_population"))
    n_plays = int(pop["n_plays"].sum()) if pop.height else 0
    share_ball = pop["n_plays_with_football"].sum() / n_plays if n_plays else None
    tot = side.filter(pl.col("season") == ALL)
    share_both = tot["share_both_sides"].item() if tot.height else None
    median_tracked = tot["median_tracked"].item() if tot.height else None
    verdicts: dict[str, tuple[str, str]] = {}

    # A · capability twin — enough athletes (and position groups) with a usable NFL exposure
    ok = exposure.filter(pl.col("n_tracked_plays") >= s.min_plays_envelope)
    by_pos = ok.group_by("position_group").agg(pl.len().alias("n"))
    n_pos = int((by_pos["n"] >= s.min_athletes_per_position).sum()) if by_pos.height else 0
    a1 = crit("A", "athletes_with_min_plays", ok.height, s.min_athletes_twin, f"≥ {s.min_plays_envelope} tracked plays")
    a2 = crit("A", "position_groups_with_min_athletes", n_pos, s.min_positions_twin,
              f"≥ {s.min_athletes_per_position} such athletes per group")
    verdicts["A"] = (VERDICTS[0] if a1 and a2 else VERDICTS[1] if n_pos >= 1 else VERDICTS[2],
                     "FEASIBLE: both · LIMITED: ≥ 1 position group qualifies (position-specific model)")

    # B · counterfactual engine — interaction counterfactuals need the ball and both sides
    b1 = crit("B", "share_plays_with_football", share_ball, s.min_share_ball, "ball geometry")
    b2 = crit("B", "share_plays_both_sides", share_both, s.min_share_both_sides, "opponent state")
    b3 = crit("B", "single_agent_basis", float(verdicts["A"][0] != VERDICTS[2]), 1.0, "stream A not INFEASIBLE")
    verdicts["B"] = (VERDICTS[0] if b1 and b2 and b3 else VERDICTS[1] if b3 else VERDICTS[2],
                     "FEASIBLE: all · LIMITED: single-agent kinematic reachability only (no interaction, no causal claim)")

    # C · offline RL — labelled actions with within-athlete variation, an individual outcome, and interaction state
    cand = []
    for f, positions in s.action_positions:
        v = _pooled(overlap, f, positions, ("n_player_plays", "n_labelled", "n_athletes_varying"))
        if v:
            cov = v["n_labelled"] / v["n_player_plays"] if v["n_player_plays"] else 0.0
            cand.append((cov >= s.min_action_coverage, v["n_athletes_varying"], cov, f))
    best = max(cand) if cand else None
    outs = []
    for f, positions in s.outcome_positions:
        v = _pooled(coverage, f, positions, ("n_athletes_min_obs",))
        outs.append((v["n_athletes_min_obs"] if v else None, f, positions))
    best_out = max((o for o in outs if o[0] is not None), default=None)
    c1 = crit("C", "action_label_coverage", best[2] if best else None, s.min_action_coverage,
              f"best action field: {best[3]}" if best else "no action field audited")
    c2 = crit("C", "athletes_with_action_variation", best[1] if best else None, s.min_athletes_varying,
              f"{best[3]}: ≥ 2 levels per athlete" if best else "")
    c3 = crit("C", "athletes_with_individual_outcome", best_out[0] if best_out else None, s.min_athletes_outcome,
              f"best outcome field: {best_out[1]}" if best_out else "no outcome field audited")
    c4 = crit("C", "share_plays_with_football", share_ball, s.min_share_ball, "attributable reward needs interaction state")
    c5 = crit("C", "share_plays_both_sides", share_both, s.min_share_both_sides, "opponent state")
    verdicts["C"] = (VERDICTS[0] if all((c1, c2, c3, c4, c5)) else VERDICTS[1] if c1 and c2 and c3 else VERDICTS[2],
                     "FEASIBLE: all · LIMITED: contextual-bandit evaluation over vendor labels only (still needs the "
                     "S5 and OPE-ESS gates) · never policy optimisation without interaction state")

    # D · multi-agent — full teams, else co-tracked rookie pairs
    pair = None
    if pairs.height:
        p = pairs.with_columns(((pl.col("n_plays") >= s.min_pair_plays)
                                & (pl.col("n_distinct_pairs") >= s.min_distinct_pairs)).alias("_ok"))
        pair = p.sort("_ok", "n_plays", descending=True).row(0, named=True)
    label = f"best pair: {pair['off_position']} × {pair['def_position']}" if pair else "no co-tracked pair"
    d1 = crit("D", "median_tracked_per_play", median_tracked, s.full_team_tracked, "11 offense + 11 defense")
    d2 = crit("D", "share_plays_both_sides", share_both, s.min_share_both_sides, "opponent state")
    d3 = crit("D", "co_tracked_plays_best_pair", pair["n_plays"] if pair else None, s.min_pair_plays, label)
    d4 = crit("D", "distinct_athlete_pairs_best_pair", pair["n_distinct_pairs"] if pair else None,
              s.min_distinct_pairs, label)
    verdicts["D"] = (VERDICTS[0] if d1 else VERDICTS[1] if d2 and d3 and d4 else VERDICTS[2],
                     "FEASIBLE: full-team tracking · LIMITED: co-tracked rookie pairs (availability, not assignments)")

    # E · decision metrics — individual vendor outcomes with enough athletes
    e = [crit("E", f"athletes_with_{f}", n, s.min_athletes_outcome, "positions: " + ", ".join(positions))
         for n, f, positions in outs]
    verdicts["E"] = (VERDICTS[0] if e and all(e) else VERDICTS[1] if any(e) else VERDICTS[2],
                     "FEASIBLE: every configured outcome · LIMITED: some")
    return _frame([{**r, "verdict": verdicts[r["stream"]][0], "rule": verdicts[r["stream"]][1]} for r in rows], FEAS)


# ============================================================================== orchestration
class DecisionAuditStudy:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, pcfg: Phase05Config | None = None, store: StudyStore | None = None,
                 con: Any = None, **_: Any):
        self.cfg = cfg
        self.pcfg = pcfg or Phase05Config()
        self.a = self.pcfg.audit
        self.store = store or StudyStore(cfg, sampled=self.pcfg.sampled, phase=PHASE)
        self.refs = LayerRefs(cfg)
        self._con = con
        self.roles = audit_fields(self.pcfg)

    @property
    def con(self):
        if self._con is None:
            self._con = duckdb_connect(self.cfg)
        return self._con

    def _scan(self, layer: str, name: str, wanted: Sequence[str]) -> pl.DataFrame:
        lf = self.refs.scan(layer, name)
        have = lf.collect_schema().names()
        return lf.select([c for c in dict.fromkeys(wanted) if c in have]).collect()

    def inputs(self) -> dict[str, pl.DataFrame]:
        fields = [f for r in ROLES for f in self.roles[r]]
        fact = self.refs.scan("gold", "fact_nfl_play_participation").select(*PLAY, "nfl_id", "season", "n_frames")
        if self.a.seasons:
            fact = fact.filter(pl.col("season").cast(pl.Int64).is_in(list(self.a.seasons)))
        return {"fact": fact.collect(),
                "player_play": self._scan("silver", "player_play", [*PLAY, "nfl_id", "team_abbr", *fields]),
                "plays": self._scan("silver", "plays", [*PLAY, "possession_team", *fields]),
                "games": self._scan("silver", "games", ["game_id", "season_type"]),
                "athletes": self._scan("gold", "dim_athlete", ["nfl_id", "position_group", "cohort"])}

    def compute(self) -> dict[str, pl.DataFrame]:
        inp = self.inputs()
        fields = [f for r in ROLES for f in self.roles[r]]
        base, source = assemble_base(inp["fact"], inp["player_play"], inp["plays"], inp["games"], inp["athletes"], fields)
        absent = sorted(f for f, v in source.items() if v == "absent")
        if absent:
            log.warning("[phase05:P0] fields absent from the warehouse (reported with 0 coverage): %s", absent)
        pop = base.filter(pl.col("season_type").is_in(list(self.a.season_types)))
        log.info("[phase05:P0] %s tracked player-plays, %s in population %s", f"{base.height:,}", f"{pop.height:,}",
                 list(self.a.season_types))
        census = entity_census(frame_census(self.con, self.refs, self.a.seasons), base, self.a.season_types)
        side, pairs = side_coverage(pop), opponent_pairs(pop, self.a)
        coverage = field_coverage(pop, self.roles, source, self.a)
        exposure = athlete_exposure(pop)
        overlap = action_overlap(pop, self.roles["action"], self.a)
        return {CENSUS.name: census, PER_PLAY.name: players_per_play(pop, self.a), SIDE.name: side, PAIRS.name: pairs,
                COVERAGE.name: coverage, EXPOSURE_DS.name: exposure, EXPOSURE.name: exposure_summary(exposure, self.a),
                OVERLAP.name: overlap,
                FEAS.name: stream_feasibility(census, side, pairs, coverage, exposure, overlap, self.pcfg)}

    def fingerprint(self) -> str:
        upstream = {f"{layer}.{name}": self.refs.num_rows(layer, name)
                    for layer, name in (("gold", "fact_nfl_play_participation"), ("silver", "game_tracking"),
                                        ("silver", "player_play"), ("silver", "plays"), ("gold", "dim_athlete"))}
        return self.store.fingerprint(STUDY, self.pcfg.section("audit", "streams"), upstream)

    def run(self, force: bool = False, figures: bool = True) -> dict[str, pl.DataFrame]:
        fp = self.fingerprint()
        current, reason = self.store.is_current(STUDY, fp, CONTRACTS)
        if current and not force:
            log.info("[phase05:P0] reuse (%s)", reason)
            out = self.load()
        else:
            log.info("[phase05:P0] compute (%s%s)", "forced; " if force else "", reason)
            res = self.compute()
            out = {c.name: self.store.write(res[c.name], c, STUDY) for c in CONTRACTS}
            self.store.record_run(STUDY, fp, CONTRACTS)
        if figures:
            self.figures(out)
        return out

    def load(self) -> dict[str, pl.DataFrame]:
        return {c.name: self.store.read(c) for c in CONTRACTS}

    def figures(self, out: dict[str, pl.DataFrame]) -> None:
        from .audit_figures import FIGURES

        for rec, fn in FIGURES:
            fn(self.store, rec, out)
