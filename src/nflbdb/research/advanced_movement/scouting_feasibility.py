"""Milestone 08A — football outcome feasibility audit (NumPy / Polars; no PyTorch, no frame scan for the main tables).

Question: which football-specific outcomes of an NFL player-play are actually OBSERVABLE in the delivered data, for
which positions, on how many athletes and games, and with which selection or leakage problem?

    CANDIDATES            pre-declared candidate outcomes (static declarations, written before any value is read)
    observation_base      tracked player-plays of notebook 05A (left) + the vendor fields of silver.player_play
    field_audit           presence, type, non-null share and example values of every required field
    coverage_table        athletes / games / observations / opportunity rate per candidate × position group
    condition_table       share observed by value of a declared conditioning field (selection made visible)
    reliability_table     odd / even-game split-half of the athlete mean, within position (feasibility, descriptive)
    feasibility_registry  static declarations + measured facts → status by an ordered rule on COUNTS and DECLARATIONS
    gate_08a              registry complete; branch = continue | stop_branch

Rules. Nothing is labelled that the vendor did not record: a null stays unobserved (it is never read as 0), an
unparsed flag token stays unobserved, and a candidate that needs a pairing of athletes (who covered whom) is
``unsupported_requires_inference``. No Combine value is read (``assert_no_combine_columns``); no status depends on the
association of a candidate with anything. Vendor definitions are INFERRED from column names (the repository holds no
data dictionary): that limitation is carried in ``definition_status`` of every row.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import get_logger
from ...features.play_windows import PP_KEYS
from ..contracts import TableContract
from .config import NotebookControls
from .nfl_outcomes import assert_no_combine_columns

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
FEASIBILITY_VERSION = "1"
ALL, DECLARED = "ALL", "DECLARED"
KEYS = list(PP_KEYS)
BASE_COLS = ("season", "week", "position_group", "eligible")
SOURCE_TABLES = ("player_play", "plays", "games", "player_career_successes")
ROLES = ("candidate", "context", "career")
KINDS = ("continuous", "flag", "count", "category")
TRUE_TOKENS = ("true", "t", "1", "1.0", "y", "yes")
FALSE_TOKENS = ("false", "f", "0", "0.0", "n", "no")
STATUSES = ("observable", "observable_conditional", "context_only", "not_movement_outcome", "not_interpretable", "blocked_definition",
            "unsupported_missing_field", "unsupported_requires_inference", "insufficient_coverage", "reliability_not_assessable")


# ============================================================================== configuration
@dataclass(frozen=True)
class FeasibilityConfig:
    """Support counts of the audit. All are heuristics declared before running; none is tuned on a result."""

    min_obs: int = 20               # observations for one athlete-level value
    min_games: int = 4              # distinct games behind those observations
    min_half_obs: int = 8           # observations in EACH game half (split-half)
    min_athletes: int = 50          # athletes at (min_obs, min_games), declared positions together
    min_athletes_position: int = 20  # same, inside one declared position
    min_athletes_halves: int = 30   # athletes with both halves supported
    reliability_ref: float = 0.50   # reference line for the Spearman–Brown split-half
    n_boot: int = 1000              # athlete bootstrap of the split-half correlation
    seed: int = 20261011

    def for_mode(self, controls: NotebookControls) -> "FeasibilityConfig":
        if not controls.smoke:
            return replace(self, seed=controls.seed)
        return replace(self, min_obs=5, min_games=2, min_half_obs=2, min_athletes=8, min_athletes_position=3, min_athletes_halves=6,
                       n_boot=100, seed=controls.seed)

    def params(self) -> dict[str, Any]:
        return {**asdict(self), "feasibility_version": FEASIBILITY_VERSION}


@dataclass(frozen=True)
class Candidate:
    outcome_id: str
    label: str
    family: str
    role: str                         # candidate · context · career
    value_field: str                  # "" = no single recorded field holds the outcome
    kind: str                         # continuous · flag · count · category
    required_fields: tuple[str, ...]
    positions: tuple[str, ...]
    interpretability: str             # movement · interaction_outcome · event · context · career · model_output
    interpretation_limit: str
    leakage_risk: str                 # low · medium · high
    leakage_note: str
    unit: str = ""
    source_table: str = "player_play"
    condition_field: str = ""         # field whose values show WHEN the outcome is recorded
    outcome_conditioned: bool = False  # recorded only when the described event happened
    needs_assignment: bool = False    # needs a pairing of athletes that no field records
    definition_status: str = "inferred_from_column_name"


_PASS = "pass_result"
CANDIDATES: tuple[Candidate, ...] = (
    Candidate("FO01.separation_at_pass_forward", "Receiver separation when the pass is released", "receiving", "candidate",
              "separation_at_pass_forward", "continuous", ("separation_at_pass_forward", "route_ran"), ("WR", "TE"), "interaction_outcome",
              "distance to the nearest defender: receiver, defender, coverage call and throw timing together", "medium",
              "recorded on pass plays only; if it exists only for targeted receivers the quarterback's choice selects the plays",
              unit="yd", condition_field="target"),
    Candidate("FO02.yards_after_catch", "Yards after catch", "receiving", "candidate", "yards_after_catch", "continuous",
              ("yards_after_catch", "target"), ("WR", "TE"), "interaction_outcome",
              "depends on the catch point, the pursuit and the blocking; not a movement quantity of the athlete alone", "high",
              "exists only after a completed catch: averaging it conditions on success", unit="yd", condition_field=_PASS,
              outcome_conditioned=True),
    Candidate("FO03.cushion", "Pre-snap cushion", "receiving", "context", "cushion", "continuous", ("cushion",), ("WR", "TE", "DB"),
              "context", "pre-snap spacing chosen by the defence; describes the situation, not the movement", "low",
              "known before the snap", unit="yd"),
    Candidate("FO04.route_ran", "Route label (conditioning of tracked movement)", "receiving", "context", "route_ran", "category",
              ("route_ran",), ("WR", "TE"), "context", "the route is an assignment label; movement on it is already measured by 05A", "low",
              "known from the play design; present on pass plays only", condition_field=_PASS),
    Candidate("FO05.player_get_off", "Get-off", "pass_rush", "candidate", "player_get_off", "continuous", ("player_get_off",), ("DL",),
              "movement", "start and end events of the timer are undocumented", "medium",
              "may be recorded on pass-rush snaps only", unit="s", condition_field=_PASS, definition_status="unresolved"),
    Candidate("FO06.time_to_pressure", "Time to pressure (rusher)", "pass_rush", "candidate", "time_to_pressure", "continuous",
              ("time_to_pressure",), ("DL",), "interaction_outcome", "rusher against a blocker, protection scheme and quarterback", "high",
              "exists only when a pressure was achieved: averaging it conditions on success", unit="s", condition_field="quick_pressure",
              outcome_conditioned=True),
    Candidate("FO07.time_to_qb_hurry", "Time to quarterback hurry", "pass_rush", "candidate", "time_to_qb_hurry", "continuous",
              ("time_to_qb_hurry",), ("DL",), "interaction_outcome", "rusher against a blocker, protection scheme and quarterback", "high",
              "exists only when a hurry was credited", unit="s", condition_field=_PASS, outcome_conditioned=True),
    Candidate("FO08.quick_pressure", "Quick pressure (flag)", "pass_rush", "candidate", "quick_pressure", "flag", ("quick_pressure",),
              ("DL",), "interaction_outcome", "a rate over rush snaps; the snaps on which the athlete rushed are not labelled", "medium",
              "the denominator (rush snaps) is defined by the flag being non-null — verify in the conditioning table",
              condition_field=_PASS),
    Candidate("FO09.pressure_allowed", "Pressure allowed (flag)", "pass_block", "candidate", "pressure_allowed", "flag",
              ("pressure_allowed",), ("OL",), "interaction_outcome", "blocker against a rusher, slide and help unobserved", "medium",
              "the denominator (pass-block snaps) is defined by the flag being non-null", condition_field=_PASS),
    Candidate("FO10.time_to_pressure_allowed", "Time to pressure allowed", "pass_block", "candidate", "time_to_pressure_allowed",
              "continuous", ("time_to_pressure_allowed",), ("OL",), "interaction_outcome", "blocker against a rusher", "high",
              "exists only when the blocker allowed a pressure", unit="s", condition_field="pressure_allowed", outcome_conditioned=True),
    Candidate("FO11.peak_pressure_probability_allowed", "Peak pressure probability allowed", "pass_block", "candidate",
              "peak_pressure_probability_allowed", "continuous", ("peak_pressure_probability_allowed",), ("OL",), "model_output",
              "output of a vendor model whose inputs are unknown (may itself use tracking)", "high",
              "a model output built from tracking would make a tracking-to-tracking comparison circular", condition_field=_PASS),
    Candidate("FO12.tackle", "Tackle credited", "defence_events", "candidate", "tackle", "count", ("tackle",), ("DB", "DL"), "event",
              "credited event: opportunity (plays towards the athlete) is unobserved", "medium",
              "whether null means 'no tackle' or 'not applicable' is unverified — a null is NOT read as 0"),
    Candidate("FO13.sack", "Sack credited", "defence_events", "candidate", "sack", "count", ("sack",), ("DL",), "event",
              "rare credited event", "medium", "whether null means 'no sack' is unverified — a null is NOT read as 0",
              condition_field=_PASS),
    Candidate("FO14.separation_allowed", "Separation allowed in coverage", "coverage", "candidate", "", "continuous",
              ("coverage_assignment", "separation_at_pass_forward"), ("DB",), "interaction_outcome",
              "separation is recorded on the RECEIVER row; no field names the defender responsible for him", "high",
              "pairing defenders to receivers by proximity would manufacture the label", unit="yd", needs_assignment=True),
    Candidate("FO15.coverage_assignment", "Coverage assignment label", "coverage", "context", "coverage_assignment", "category",
              ("coverage_assignment", "coverage_assignment_at_snap"), ("DB",), "context",
              "an assignment label, not an outcome", "low", "present on pass plays only", condition_field=_PASS),
    Candidate("FO16.career_snaps", "Career snaps / games", "career", "career", "career_games_active", "count",
              ("career_offensive_snaps", "career_defensive_snaps", "career_games_active"), ("DB", "DL", "OL", "TE", "WR"), "career",
              "playing time reflects draft capital, depth chart and injuries", "high",
              "draft cohorts have had 1 to 3 seasons: totals are censored by cohort", source_table="player_career_successes"),
)


def registry_hash(cands: Sequence[Candidate] = CANDIDATES) -> str:
    """Hash of the static declarations: part of the cache key, recorded in the manifest."""
    return hashlib.sha256(json.dumps([asdict(c) for c in cands], sort_keys=True).encode()).hexdigest()[:16]


def candidate_table(cands: Sequence[Candidate] = CANDIDATES) -> pl.DataFrame:
    rows = [{**{k: (" · ".join(v) if isinstance(v, tuple) else v) for k, v in asdict(c).items()}} for c in cands]
    return pl.DataFrame(rows)


# ============================================================================== contracts
FIELD_SCHEMA = {"source_table": S, "field": S, "present": B, "dtype": S, "n_rows": I, "n_observed": I, "observed_share": F,
                "n_distinct": I, "examples": S, "used_by": S}
COVERAGE_SCHEMA = {"outcome_id": S, "position_group": S, "declared_position": B, "n_athletes_tracked": I, "n_eligible_plays": I,
                   "n_obs": I, "n_athletes_any": I, "n_athletes_ge_min": I, "n_athlete_games": I, "median_obs": F, "median_games": F,
                   "opportunity_rate": F}
CONDITION_SCHEMA = {"outcome_id": S, "condition_field": S, "condition_value": S, "n_rows": I, "n_obs": I, "obs_share": F}
RELIABILITY_SCHEMA = {"outcome_id": S, "n_athletes_both_halves": I, "n_positions": I, "r_half": F, "rel_sb": F, "rel_lo": F,
                      "rel_hi": F, "note": S}
REGISTRY_SCHEMA = {"outcome_id": S, "label": S, "family": S, "role": S, "kind": S, "unit": S, "source_table": S, "value_field": S,
                   "required_fields": S, "missing_fields": S, "positions": S, "interpretability": S, "interpretation_limit": S,
                   "leakage_risk": S, "leakage_note": S, "outcome_conditioned": B, "needs_assignment": B, "definition_status": S,
                   "n_athletes_ge_min": I, "n_positions_supported": I, "median_games": F, "opportunity_rate": F,
                   "n_athletes_both_halves": I, "rel_sb": F, "rel_lo": F, "rel_hi": F, "reliability_reading": S, "status": S,
                   "reason": S, "defensible": B}
EVENTS_SCHEMA = {"event": S, "n_frames": I, "n_games": I}

FIELD_AUDIT = TableContract("football_field_audit", "Presence, type and observed share of every field a candidate football outcome needs.",
                            FIELD_SCHEMA, key=("source_table", "field"), area="source table × field",
                            docs={"n_observed": "non-null (strings: non-empty) among the tracked player-plays of notebook 05A"})
COVERAGE = TableContract("football_outcome_coverage", "Athletes, games and observations per candidate football outcome and position group.",
                         COVERAGE_SCHEMA, key=("outcome_id", "position_group"), area="candidate × position group",
                         docs={"position_group": "a position group, ALL, or DECLARED = the positions the candidate is declared for",
                               "n_athletes_ge_min": "athletes with ≥ min_obs observations in ≥ min_games games (eligible windows)",
                               "opportunity_rate": "observations / eligible player-plays of the group"})
CONDITIONS = TableContract("football_outcome_conditions", "Share of eligible player-plays on which a candidate is recorded, by value of its conditioning field.",
                           CONDITION_SCHEMA, key=("outcome_id", "condition_value"), area="candidate × conditioning value",
                           docs={"obs_share": "a share near 1 for one value and near 0 elsewhere = the outcome exists only in that situation"})
RELIABILITY = TableContract("football_outcome_split_half", "Odd / even-game split-half of the athlete mean, within declared position.",
                            RELIABILITY_SCHEMA, key=("outcome_id",), area="candidate",
                            docs={"rel_sb": "Spearman–Brown of r_half; descriptive feasibility, not a frozen reliability",
                                  "rel_lo": "athlete bootstrap, percentile 2.5 (position centring kept from the full sample)"})
REGISTRY = TableContract("football_outcome_feasibility_registry", "Feasibility registry of candidate football outcomes: declarations, measured support, status.",
                         REGISTRY_SCHEMA, key=("outcome_id",), area="candidate",
                         docs={"status": " · ".join(STATUSES),
                               "defensible": "status observable AND split-half at or above the reference line with a positive lower bound"})
EVENTS = TableContract("tracking_event_inventory", "Values of the event field of silver.game_tracking.", EVENTS_SCHEMA, key=("event",),
                       area="tracking event")
CONTRACTS = (FIELD_AUDIT, COVERAGE, CONDITIONS, RELIABILITY, REGISTRY)


# ============================================================================== inputs
def source_schemas(refs: Any, tables: Sequence[str] = SOURCE_TABLES, layer: str = "silver") -> dict[str, dict[str, str]]:
    """{table: {column: dtype}} of the delivered tables; an unreadable table is an empty schema (and a warning)."""
    out: dict[str, dict[str, str]] = {}
    for t in tables:
        try:
            out[t] = {k: str(v) for k, v in refs.scan(layer, t).collect_schema().items()}
        except Exception as e:                                   # noqa: BLE001 — absence is a finding of the audit
            log.warning("[scouting_feasibility] %s.%s unreadable (%s: %s) — treated as absent", layer, t, type(e).__name__, e)
            out[t] = {}
    return out


def wanted_fields(cands: Sequence[Candidate] = CANDIDATES, table: str = "player_play") -> list[str]:
    out: list[str] = []
    for c in cands:
        if c.source_table == table:
            out += [*c.required_fields, c.condition_field]
    return [f for f in dict.fromkeys(out) if f]


def observation_base(plays: pl.DataFrame, player_play: pl.LazyFrame | pl.DataFrame, cands: Sequence[Candidate] = CANDIDATES) -> pl.DataFrame:
    """Tracked player-plays of notebook 05A (every row kept) with the vendor fields that exist. A field that is not
    in ``player_play`` is simply not a column of the result — nothing is created in its place."""
    lf = player_play.lazy()
    have = lf.collect_schema().names()
    take = [f for f in wanted_fields(cands) if f in have and f not in KEYS and f not in BASE_COLS]
    pp = (lf.select(*KEYS, *take).collect().with_columns([pl.col(k).cast(pl.Int64) for k in KEYS])
          .unique(subset=KEYS, keep="first", maintain_order=True))
    base = plays.select(*KEYS, *BASE_COLS).join(pp, on=KEYS, how="left")
    assert base.height == plays.height, f"the join changed the number of player-plays ({plays.height:,} → {base.height:,})"
    assert_no_combine_columns(base, "08A observation base")
    return base


def _observed(field: str, dtype: pl.DataType) -> pl.Expr:
    col = pl.col(field)
    if dtype == pl.String:
        return col.is_not_null() & (col.str.strip_chars() != "")
    if dtype.is_float():
        return col.is_not_null() & col.is_not_nan()
    return col.is_not_null()


def value_expr(c: Candidate, dtype: pl.DataType) -> pl.Expr:
    """Float value of one observation, null when unobserved. Flags: recognised tokens only; a category: 1.0 = label present."""
    col, none = pl.col(c.value_field), pl.lit(None, dtype=pl.Float64)
    if c.kind == "category":
        return pl.when(_observed(c.value_field, dtype)).then(pl.lit(1.0)).otherwise(none)
    if c.kind == "flag":
        if dtype == pl.String:
            s = col.str.strip_chars().str.to_lowercase()
            return pl.when(s.is_in(list(TRUE_TOKENS))).then(pl.lit(1.0)).when(s.is_in(list(FALSE_TOKENS))).then(pl.lit(0.0)).otherwise(none)
        if dtype == pl.Boolean:
            return col.cast(pl.Float64)
        return (col.cast(pl.Float64, strict=False).fill_nan(None) > 0).cast(pl.Float64)
    return col.cast(pl.Float64, strict=False).fill_nan(None)


def measurable(c: Candidate, base: pl.DataFrame) -> bool:
    return bool(c.value_field) and c.source_table == "player_play" and c.value_field in base.columns


def observations(base: pl.DataFrame, c: Candidate) -> pl.DataFrame:
    """Eligible player-plays on which the candidate is recorded: nfl_id, position_group, game_id, game_key, value."""
    return (base.filter(pl.col("eligible")).with_columns(value_expr(c, base.schema[c.value_field]).alias("value"))
            .filter(pl.col("value").is_not_null())
            .select("nfl_id", "position_group", "game_id", (pl.col("season") * 100 + pl.col("week")).alias("game_key"), "value"))


# ============================================================================== tables
def _examples(s: pl.Series, k: int = 5) -> str:
    s = s.drop_nulls()
    if s.is_empty():
        return ""
    if s.dtype.is_numeric() and s.n_unique() > 20:
        return f"min {s.min():.3g} · median {s.median():.3g} · max {s.max():.3g}"
    vc = s.cast(pl.String).value_counts(sort=True).head(k)
    return " | ".join(f"{v} ({n:,})" for v, n in vc.iter_rows())


def field_audit(base: pl.DataFrame, schemas: dict[str, dict[str, str]], cands: Sequence[Candidate] = CANDIDATES) -> pl.DataFrame:
    users: dict[tuple[str, str], list[str]] = {}
    for c in cands:
        for f in dict.fromkeys((*c.required_fields, c.condition_field)):
            if f:
                users.setdefault((c.source_table, f), []).append(c.outcome_id.split(".")[0])
    rows = []
    for (table, f), ids in users.items():
        row = {"source_table": table, "field": f, "present": f in schemas.get(table, {}), "dtype": schemas.get(table, {}).get(f, ""),
               "n_rows": None, "n_observed": None, "observed_share": None, "n_distinct": None, "examples": "", "used_by": " ".join(ids)}
        if table == "player_play" and f in base.columns:
            n_obs = int(base.select(_observed(f, base.schema[f]).sum()).item())
            row |= {"n_rows": base.height, "n_observed": n_obs, "observed_share": n_obs / max(base.height, 1),
                    "n_distinct": int(base[f].n_unique()), "examples": _examples(base[f])}
        rows.append(row)
    return pl.DataFrame(rows, schema=FIELD_SCHEMA).sort("source_table", "field")


def _groups(per: pl.DataFrame, positions: Sequence[str]) -> pl.DataFrame:
    """Athlete rows repeated under ALL and DECLARED, so one group-by gives the three kinds of group."""
    return pl.concat([per, per.with_columns(pl.lit(ALL).alias("position_group")),
                      per.filter(pl.col("position_group").is_in(list(positions))).with_columns(pl.lit(DECLARED).alias("position_group"))])


def coverage_table(base: pl.DataFrame, cands: Sequence[Candidate], cfg: FeasibilityConfig) -> pl.DataFrame:
    den_per = base.filter(pl.col("eligible")).group_by("nfl_id", "position_group").agg(pl.len().alias("n_plays"))
    parts = []
    for c in cands:
        if not measurable(c, base):
            continue
        den = (_groups(den_per, c.positions).group_by("position_group")
               .agg(pl.len().alias("n_athletes_tracked"), pl.col("n_plays").sum().alias("n_eligible_plays")))
        per = observations(base, c).group_by("nfl_id", "position_group").agg(pl.len().alias("n"), pl.col("game_id").n_unique().alias("g"))
        num = (_groups(per, c.positions).group_by("position_group")
               .agg(pl.col("n").sum().alias("n_obs"), pl.len().alias("n_athletes_any"),
                    ((pl.col("n") >= cfg.min_obs) & (pl.col("g") >= cfg.min_games)).sum().alias("n_athletes_ge_min"),
                    pl.col("g").sum().alias("n_athlete_games"), pl.col("n").median().alias("median_obs"),
                    pl.col("g").median().alias("median_games")))
        counts = ["n_obs", "n_athletes_any", "n_athletes_ge_min", "n_athlete_games"]
        parts.append(den.join(num, on="position_group", how="left")
                     .with_columns([pl.col(k).fill_null(0).cast(pl.Int64) for k in counts])
                     .with_columns(pl.lit(c.outcome_id).alias("outcome_id"),
                                   pl.col("position_group").is_in([*c.positions, DECLARED]).alias("declared_position"),
                                   (pl.col("n_obs") / pl.col("n_eligible_plays")).alias("opportunity_rate"),
                                   pl.col("n_athletes_tracked").cast(pl.Int64), pl.col("n_eligible_plays").cast(pl.Int64))
                     .select(list(COVERAGE_SCHEMA)))
    if not parts:
        return pl.DataFrame(schema=COVERAGE_SCHEMA)
    return pl.concat(parts).cast(COVERAGE_SCHEMA).sort("outcome_id", "position_group")


def condition_table(base: pl.DataFrame, cands: Sequence[Candidate], top: int = 12) -> pl.DataFrame:
    """Where is the candidate recorded? Share observed by value of its declared conditioning field (declared positions)."""
    parts = []
    for c in cands:
        f = c.condition_field
        if not measurable(c, base) or not f or f == c.value_field or f not in base.columns:
            continue
        t = (base.filter(pl.col("eligible") & pl.col("position_group").is_in(list(c.positions)))
             .select(pl.col(f).cast(pl.String).fill_null("‹null›").alias("condition_value"),
                     value_expr(c, base.schema[c.value_field]).is_not_null().alias("obs"))
             .group_by("condition_value").agg(pl.len().alias("n_rows"), pl.col("obs").sum().alias("n_obs"))
             .sort("n_rows", "condition_value", descending=[True, False]).head(top))
        parts.append(t.select(pl.lit(c.outcome_id).alias("outcome_id"), pl.lit(f).alias("condition_field"), "condition_value",
                              pl.col("n_rows").cast(pl.Int64), pl.col("n_obs").cast(pl.Int64),
                              (pl.col("n_obs") / pl.col("n_rows")).alias("obs_share")))
    return pl.concat(parts).cast(CONDITION_SCHEMA) if parts else pl.DataFrame(schema=CONDITION_SCHEMA)


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two (replicates × athletes) arrays; NaN where a variance is zero."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(-1) / den, np.nan)


def _sb(r: np.ndarray | float) -> np.ndarray:
    r = np.asarray(r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(r > -1, 2 * r / (1 + r), np.nan)


def half_means(obs: pl.DataFrame, min_half_obs: int) -> pl.DataFrame:
    """Athlete mean in odd and even games (chronological rank of the athlete's own observed games), position-centred."""
    a = (obs.with_columns((pl.col("game_key").rank("dense").over("nfl_id") % 2).cast(pl.Int64).alias("half"))
         .group_by("nfl_id", "position_group", "half").agg(pl.len().alias("n"), pl.col("value").mean().alias("m")))
    odd = a.filter(pl.col("half") == 1).select("nfl_id", "position_group", pl.col("n").alias("n_a"), pl.col("m").alias("m_a"))
    even = a.filter(pl.col("half") == 0).select("nfl_id", pl.col("n").alias("n_b"), pl.col("m").alias("m_b"))
    return (odd.join(even, on="nfl_id", how="inner")
            .filter((pl.col("n_a") >= min_half_obs) & (pl.col("n_b") >= min_half_obs))
            .with_columns((pl.col("m_a") - pl.col("m_a").mean().over("position_group")).alias("r_a"),
                          (pl.col("m_b") - pl.col("m_b").mean().over("position_group")).alias("r_b")).sort("nfl_id"))


def reliability_table(base: pl.DataFrame, cands: Sequence[Candidate], cfg: FeasibilityConfig) -> pl.DataFrame:
    rows = []
    for i, c in enumerate(cands):
        if not measurable(c, base) or c.role != "candidate":
            continue
        w = half_means(observations(base, c).filter(pl.col("position_group").is_in(list(c.positions))), cfg.min_half_obs)
        row = {"outcome_id": c.outcome_id, "n_athletes_both_halves": w.height, "n_positions": w["position_group"].n_unique(),
               "r_half": None, "rel_sb": None, "rel_lo": None, "rel_hi": None, "note": ""}
        if w.height < 3:
            row["note"] = f"fewer than 3 athletes with ≥ {cfg.min_half_obs} observations in each game half"
        else:
            ra, rb = w["r_a"].to_numpy(), w["r_b"].to_numpy()
            r = float(_corr(ra[None, :], rb[None, :])[0])
            idx = np.random.default_rng([cfg.seed, i]).integers(0, w.height, (cfg.n_boot, w.height))
            boot = _sb(_corr(ra[idx], rb[idx]))
            if np.isfinite(r) and np.isfinite(boot).any():
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                row |= {"r_half": r, "rel_sb": float(_sb(r)), "rel_lo": float(lo), "rel_hi": float(hi)}
            else:
                row["note"] = "no variation between athletes in one game half"
        rows.append(row)
    return pl.DataFrame(rows, schema=RELIABILITY_SCHEMA)


# ============================================================================== registry and gate
def status_of(c: Candidate, missing: Sequence[str], definition: str, n_ge: int, n_pos: int, n_halves: int,
              cfg: FeasibilityConfig) -> tuple[str, str]:
    """Ordered rule. It reads declarations and COUNTS only — never the value of an outcome or of a reliability."""
    if missing:
        return "unsupported_missing_field", f"absent from {c.source_table}: {', '.join(missing)}"
    if c.needs_assignment:
        return "unsupported_requires_inference", ("needs a pairing of athletes that no field records; pairing them by proximity "
                                                  "would manufacture the label")
    if c.role == "context":
        return "context_only", "describes the situation of a play, not what the athlete achieved; usable as a conditioning label"
    if c.role == "career":
        return "not_movement_outcome", "athlete-level career total: no movement content, exposure censored by draft cohort"
    if c.interpretability == "model_output":
        return "not_interpretable", "output of an undocumented vendor model"
    if definition == "unresolved":
        return "blocked_definition", "vendor definition unresolved: the value cannot be given a football meaning"
    if n_ge < cfg.min_athletes or n_pos < 1:
        return "insufficient_coverage", (f"{n_ge} athletes with ≥ {cfg.min_obs} observations in ≥ {cfg.min_games} games (needed "
                                         f"{cfg.min_athletes}); {n_pos} declared position(s) reach {cfg.min_athletes_position}")
    if n_halves < cfg.min_athletes_halves:
        return "reliability_not_assessable", (f"{n_halves} athletes with ≥ {cfg.min_half_obs} observations in each game half "
                                              f"(needed {cfg.min_athletes_halves})")
    if c.outcome_conditioned:
        return "observable_conditional", ("recorded only when the event it describes happened: the athlete mean is taken over a "
                                          "self-selected subset of plays")
    return "observable", "fields present; coverage and split-half support reach the stated counts"


def _f(x: Any) -> float:
    return float("nan") if x is None else float(x)


def feasibility_registry(cands: Sequence[Candidate], schemas: dict[str, dict[str, str]], coverage: pl.DataFrame,
                         reliability: pl.DataFrame, cfg: FeasibilityConfig, definition_overrides: dict[str, str] | None = None) -> pl.DataFrame:
    over = definition_overrides or {}
    cov = {(r["outcome_id"], r["position_group"]): r for r in coverage.iter_rows(named=True)}
    rel = {r["outcome_id"]: r for r in reliability.iter_rows(named=True)}
    rows = []
    for c in cands:
        missing = [f for f in c.required_fields if f not in schemas.get(c.source_table, {})]
        d, r = cov.get((c.outcome_id, DECLARED), {}), rel.get(c.outcome_id, {})
        n_ge, n_halves = int(d.get("n_athletes_ge_min") or 0), int(r.get("n_athletes_both_halves") or 0)
        n_pos = sum(int(cov.get((c.outcome_id, p), {}).get("n_athletes_ge_min") or 0) >= cfg.min_athletes_position for p in c.positions)
        definition = over.get(c.outcome_id, c.definition_status)
        status, reason = status_of(c, missing, definition, n_ge, n_pos, n_halves, cfg)
        sb, lo = _f(r.get("rel_sb")), _f(r.get("rel_lo"))
        reading = "not_assessed" if not np.isfinite(sb) else ("at_or_above_reference" if sb >= cfg.reliability_ref else "below_reference")
        rows.append({"outcome_id": c.outcome_id, "label": c.label, "family": c.family, "role": c.role, "kind": c.kind, "unit": c.unit,
                     "source_table": c.source_table, "value_field": c.value_field, "required_fields": " · ".join(c.required_fields),
                     "missing_fields": " · ".join(missing), "positions": " · ".join(c.positions), "interpretability": c.interpretability,
                     "interpretation_limit": c.interpretation_limit, "leakage_risk": c.leakage_risk, "leakage_note": c.leakage_note,
                     "outcome_conditioned": c.outcome_conditioned, "needs_assignment": c.needs_assignment, "definition_status": definition,
                     "n_athletes_ge_min": n_ge, "n_positions_supported": n_pos, "median_games": d.get("median_games"),
                     "opportunity_rate": d.get("opportunity_rate"), "n_athletes_both_halves": n_halves, "rel_sb": r.get("rel_sb"),
                     "rel_lo": r.get("rel_lo"), "rel_hi": r.get("rel_hi"), "reliability_reading": reading, "status": status,
                     "reason": reason,
                     "defensible": bool(status == "observable" and reading == "at_or_above_reference" and np.isfinite(lo) and lo > 0)})
    return pl.DataFrame(rows, schema=REGISTRY_SCHEMA)


def run_audit(base: pl.DataFrame, schemas: dict[str, dict[str, str]], cands: Sequence[Candidate], cfg: FeasibilityConfig,
              definition_overrides: dict[str, str] | None = None) -> dict[str, pl.DataFrame]:
    """The five tables of the audit, keyed by contract name."""
    coverage, reliability = coverage_table(base, cands, cfg), reliability_table(base, cands, cfg)
    return {FIELD_AUDIT.name: field_audit(base, schemas, cands), COVERAGE.name: coverage, CONDITIONS.name: condition_table(base, cands),
            RELIABILITY.name: reliability,
            REGISTRY.name: feasibility_registry(cands, schemas, coverage, reliability, cfg, definition_overrides)}


def gate_08a(registry: pl.DataFrame, cands: Sequence[Candidate] = CANDIDATES) -> dict[str, Any]:
    """PASS = the registry is complete (every declared candidate has one valid status and a reason). ``branch`` says
    whether a football-outcome analysis may follow: ``stop_branch`` is a valid scientific result, not a failure."""
    ids = registry["outcome_id"].to_list()
    complete = (sorted(ids) == sorted(c.outcome_id for c in cands) and len(set(ids)) == len(ids)
                and set(registry["status"].to_list()) <= set(STATUSES) and all(registry["reason"].to_list()))
    by_status = {r["status"]: r["n"] for r in registry.group_by("status").agg(pl.len().alias("n")).sort("status").iter_rows(named=True)}
    defensible = registry.filter(pl.col("defensible"))["outcome_id"].to_list()
    conditional = registry.filter(pl.col("status") == "observable_conditional")["outcome_id"].to_list()
    return {"gate": "PASS" if complete else "FAIL", "registry_complete": complete, "n_candidates": len(ids), "by_status": by_status,
            "defensible": defensible, "observable_conditional": conditional, "branch": "continue" if defensible else "stop_branch"}


def conclusions_markdown(registry: pl.DataFrame, gate: dict[str, Any], cfg: FeasibilityConfig) -> str:
    lines = ["### 08A — football outcome feasibility", "",
             f"Gate: **{gate['gate']}** · branch: **{gate['branch']}** · {gate['n_candidates']} candidates · "
             + " · ".join(f"{k}: {v}" for k, v in gate["by_status"].items()), "",
             "| Candidate | Positions | Status | Athletes ≥ min | Split-half [95 %] | Reason |", "|---|---|---|---|---|---|"]
    for r in registry.iter_rows(named=True):
        sb = "—" if r["rel_sb"] is None else f"{r['rel_sb']:.2f} [{r['rel_lo']:.2f}, {r['rel_hi']:.2f}]"
        lines.append(f"| `{r['outcome_id']}` | {r['positions']} | {r['status']} | {r['n_athletes_ge_min']} | {sb} | {r['reason']} |")
    lines += ["", f"Support counts: ≥ {cfg.min_obs} observations in ≥ {cfg.min_games} games per athlete, ≥ {cfg.min_athletes} athletes "
                  f"(≥ {cfg.min_athletes_position} in a declared position), ≥ {cfg.min_athletes_halves} athletes with both game halves; "
                  f"split-half reference line {cfg.reliability_ref:.2f}. All are heuristics declared before running.", ""]
    if gate["defensible"]:
        lines.append("Defensible for a later, separately designed analysis: " + ", ".join(f"`{o}`" for o in gate["defensible"])
                     + ". This is an observability statement: no association with any Combine measure has been looked at.")
    else:
        lines.append("**No candidate is a defensible football-specific outcome under these rules. The football-outcome branch stops "
                     "here**; Milestone 08 continues on descriptive Combine evidence (08B–08D) and the NFL movement targets of Milestone 05.")
    lines += ["", "Limits: vendor definitions are inferred from column names (no data dictionary in the repository); a null is never "
                  "read as zero; assignment, opponent and scheme are unobserved; the split-half is a feasibility screen on one "
                  "odd / even-game split, not a frozen reliability."]
    return "\n".join(lines)


def tracking_event_audit(tracking: pl.LazyFrame) -> pl.DataFrame:
    """Values of ``event`` in the tracking frames (one column scan). Without the column: one row saying so."""
    if "event" not in tracking.collect_schema().names():
        return pl.DataFrame([{"event": "‹column absent from silver.game_tracking›", "n_frames": 0, "n_games": 0}], schema=EVENTS_SCHEMA)
    return (tracking.select("game_id", pl.col("event").cast(pl.String).fill_null("‹null›"))
            .group_by("event").agg(pl.len().alias("n_frames"), pl.col("game_id").n_unique().alias("n_games"))
            .collect().cast(EVENTS_SCHEMA).sort("n_frames", "event", descending=[True, False]))


# ============================================================================== figures
def fig_coverage(coverage: pl.DataFrame, registry: pl.DataFrame, cfg: FeasibilityConfig):
    """AM08A_01 — athletes with enough observations, candidate × position group (declared positions in bold)."""
    import matplotlib.pyplot as plt

    lut = {(r["outcome_id"], r["position_group"]): r for r in coverage.iter_rows(named=True)}
    pos = sorted({p for _, p in lut} - {ALL, DECLARED})
    ids = [o for o in registry["outcome_id"].to_list() if any(k[0] == o for k in lut)]
    fig, ax = plt.subplots(figsize=(1.1 * max(len(pos), 1) + 5.0, 0.42 * max(len(ids), 1) + 1.8))
    if not ids:
        ax.text(0.5, 0.5, "no measurable candidate", ha="center", va="center")
        ax.set_axis_off()
        return fig
    m = np.array([[float(lut.get((o, p), {}).get("n_athletes_ge_min", 0)) for p in pos] for o in ids])
    im = ax.imshow(m, aspect="auto", cmap="Blues", vmin=0)
    for i, o in enumerate(ids):
        for j, p in enumerate(pos):
            dec = bool(lut.get((o, p), {}).get("declared_position", False))
            ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=8, fontweight="bold" if dec else "normal",
                    color="white" if m[i, j] > 0.6 * max(m.max(), 1) else "black")
    ax.set_xticks(range(len(pos)), pos)
    ax.set_yticks(range(len(ids)), ids, fontsize=8)
    ax.set_title(f"Athletes with ≥ {cfg.min_obs} observations in ≥ {cfg.min_games} games (bold = declared position)", fontsize=10)
    fig.colorbar(im, ax=ax, label="athletes")
    fig.tight_layout()
    return fig


def fig_status(registry: pl.DataFrame, cfg: FeasibilityConfig):
    """AM08A_02 — supported athletes per candidate with the registry status written next to each bar."""
    import matplotlib.pyplot as plt

    r = registry.sort("n_athletes_ge_min")
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * max(r.height, 1) + 1.6))
    n = r["n_athletes_ge_min"].to_numpy()
    ax.barh(range(r.height), n, color="#4c78a8")
    ax.axvline(cfg.min_athletes, color="black", linestyle="--", linewidth=1, label=f"support line ({cfg.min_athletes} athletes)")
    for i, (v, s, d) in enumerate(zip(n, r["status"].to_list(), r["defensible"].to_list())):
        ax.text(v, i, f"  {s}" + (" ✓" if d else ""), va="center", fontsize=8)
    ax.set_yticks(range(r.height), r["outcome_id"].to_list(), fontsize=8)
    ax.set_xlabel("athletes with enough observations (declared positions)")
    ax.set_xlim(0, max(float(n.max()) if r.height else 0.0, cfg.min_athletes) * 1.6)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Football outcome feasibility: support and status", fontsize=10)
    fig.tight_layout()
    return fig


__all__ = ["ALL", "CANDIDATES", "CONDITIONS", "CONTRACTS", "COVERAGE", "Candidate", "DECLARED", "EVENTS", "FIELD_AUDIT",
           "FeasibilityConfig", "REGISTRY", "RELIABILITY", "STATUSES", "candidate_table", "conclusions_markdown", "condition_table",
           "coverage_table", "feasibility_registry", "field_audit", "fig_coverage", "fig_status", "gate_08a", "half_means",
           "observation_base", "observations", "registry_hash", "reliability_table", "run_audit", "source_schemas", "status_of",
           "tracking_event_audit", "value_expr", "wanted_fields"]
