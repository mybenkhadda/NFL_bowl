"""Milestone 08A.2 — semantics audit of the candidate football outcomes (NumPy / Polars; no PyTorch, no frame scan).

Question: can ``FO01.separation_at_pass_forward``, ``FO12.tackle`` and ``FO13.sack`` be read as meaningful outcomes of an
NFL player, and what exactly does a recorded value, a recorded zero and a null mean?

    SEMANTICS             static declarations (written before a value is read): field, unit, what is ASSUMED about it
    semantic_base         tracked player-plays of notebook 05A (every row kept) + the vendor fields that exist
    candidates_table      one row per outcome: declaration + measured presence, zero / positive counts, seasons
    zero_missing_table    null vs zero vs positive per field and position (a null is never read as 0)
    structure_table       where is a value recorded? share observed by value of a conditioning field
    per_play_credit       total credit per play over the delivered vendor rows (is the credit a player-level share?)
    fo01_rows / fo01_distribution / route_alignment_cells      separation: values, extremes, context support
    context_table         context fields: availability, timing relative to the outcome, leakage as a predictor
    semantic_checks       rule-based internal-consistency checks (blocking / informative)
    semantic_status       confirmed_source · plausible_inferred · unresolved

Rules. The repository holds no vendor data dictionary, so every definition starts as ``plausible_inferred``. It becomes
``confirmed_source`` only through evidence the analyst records (source + quoted definition), and ``unresolved`` when a
blocking consistency check fails. No Combine value is read (``assert_no_combine_columns``); no check looks at the
association of an outcome with anything outside the NFL player-play table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import polars as pl

from ...features.play_windows import PLAY_KEYS, PP_KEYS
from ..contracts import TableContract
from . import scouting_feasibility as sf
from .nfl_outcomes import assert_no_combine_columns

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
SEMANTICS_VERSION = "1"
KEYS = list(PP_KEYS)
ALL, NULL = "ALL", "‹null›"
FO01, FO12, FO13 = "FO01.separation_at_pass_forward", "FO12.tackle", "FO13.sack"
STATUSES = ("confirmed_source", "plausible_inferred", "unresolved")
PLAY_COLS = ("season", "week", "position_group", "cohort", "side", "lined_up_position", "play_class", "pass_result", "down", "eligible")
AUDIT_FIELDS = ("separation_at_pass_forward", "route_ran", "target", "cushion", "in_motion_at_ball_snap", "receiver_alignment",
                "offense_formation", "pass_length", "team_coverage_man_zone", "team_coverage_type", "tackle", "assist",
                "tackle_for_loss", "caused_forced_fumble", "sack", "blitzing", "unblocked_pressure", "coverage_assignment",
                "quick_pressure", "time_to_pressure", "dropback_duration", "yards_gained")


# ============================================================================== declarations
@dataclass(frozen=True)
class OutcomeSemantics:
    outcome_id: str
    value_field: str
    positions: tuple[str, ...]
    unit: str
    kind: str                       # continuous · count
    declared_definition: str        # what the column name suggests — an ASSUMPTION until evidence is recorded
    event_definition: str
    zero_encoding: str
    missing_meaning: str
    attribution: str
    opportunity: str
    eligibility: str
    limits: str
    allowed_values: tuple[float, ...] = ()
    definition_status: str = "plausible_inferred"
    source_table: str = "silver.player_play"


_ELIGIBLE = "eligible window of notebook 05A (REG season, in-scope position, not nullified by penalty, usable quality tier)"
SEMANTICS: tuple[OutcomeSemantics, ...] = (
    OutcomeSemantics(
        FO01, "separation_at_pass_forward", ("WR", "TE"), "yd", "continuous",
        "distance from the player to the nearest defender at the frame tagged pass_forward",
        "one value per player-play, sampled at the pass release; the release frame itself is not delivered with the value",
        "0 would be a physical distance of zero; no structural zero is expected",
        "null = not recorded (no pass released, or the player was not in a route) — never read as a separation of 0",
        "recorded on the receiver's row; the defender is not named",
        "route snaps on plays with a released pass; whether non-targeted receivers are included is MEASURED, not assumed",
        _ELIGIBLE,
        "an interaction outcome: receiver, nearest defender, coverage call, route depth and throw timing together"),
    OutcomeSemantics(
        FO12, "tackle", ("DB", "DL"), "count per player-play", "count",
        "the player was credited with a (solo) tackle on the play",
        "a credited statistic of the play, not a tracking event; the tracking tag 'tackle' marks a frame of the play",
        "a recorded 0 exists: its meaning (assist only, missed tackle, other credited action) is MEASURED against the companion fields",
        "null = no tackle statistic recorded for the player on the play; 'no credit' is an inference, not a recorded 0",
        "player-level credit; solo versus assisted depends on the companion field 'assist'",
        "no field records a tackle OPPORTUNITY (plays run at the player, tackle attempts); only snaps on the field are observed",
        _ELIGIBLE,
        "credit depends on role, scheme, run direction and playing time; a raw count is not a tackling skill",
        allowed_values=(0.0, 1.0)),
    OutcomeSemantics(
        FO13, "sack", ("DL",), "sacks per player-play (0, 0.5, 1)", "count",
        "sack credit of the player on the play (0.5 = shared sack)",
        "a credited statistic of the play; the tracking tag 'qb_sack' marks a frame of the play",
        "0 is recorded explicitly on a subset of plays; which plays is MEASURED (it is not assumed to be pass-rush snaps)",
        "null = no sack statistic recorded on the play (for instance a run play) — never read as 0",
        "player-level credit share",
        "the recorded rows are a candidate denominator; whether the player actually rushed the passer on them is not recorded",
        _ELIGIBLE,
        "rare event; depends on rush assignment, protection, quarterback and game script",
        allowed_values=(0.0, 0.5, 1.0)),
)
BY_ID = {s.outcome_id: s for s in SEMANTICS}


def definition_hash(sems: Sequence[OutcomeSemantics] = SEMANTICS) -> str:
    return hashlib.sha256(json.dumps([asdict(s) for s in sems], sort_keys=True).encode()).hexdigest()[:16]


# context fields: static declarations of WHEN a field is known and what using it would mean (section 6.6 of the brief)
CONTEXT: tuple[dict[str, Any], ...] = (
    {"field": "position_group", "source": "gold.dim_athlete", "interpretation": "roster position group of the athlete",
     "known_pre_snap": "yes", "observed_after_outcome": False, "available_pre_draft": True,
     "leakage_as_predictor": "none (the Combine position is known at the draft)", "changes_estimand": "no: defines the comparison population"},
    {"field": "lined_up_position", "source": "player_play", "interpretation": "alignment of the player on the play",
     "known_pre_snap": "yes", "observed_after_outcome": False, "available_pre_draft": False,
     "leakage_as_predictor": "future NFL role: forbidden in a pre-draft model",
     "changes_estimand": "yes: an alignment-adjusted value compares players inside an NFL role the team chose"},
    {"field": "route_ran", "source": "player_play", "interpretation": "route label of the player (an assignment label)",
     "known_pre_snap": "partly (play design; labelled after the play)", "observed_after_outcome": True, "available_pre_draft": False,
     "leakage_as_predictor": "future NFL usage: forbidden in a pre-draft model",
     "changes_estimand": "yes: a route-adjusted value removes route mix, which is partly a property of the player"},
    {"field": "target", "source": "player_play", "interpretation": "the pass was thrown to the player",
     "known_pre_snap": "no", "observed_after_outcome": True, "available_pre_draft": False,
     "leakage_as_predictor": "outcome-dependent (the quarterback chooses the open receiver): never a predictor, never a filter of the primary target",
     "changes_estimand": "yes: conditioning on it selects plays on the outcome"},
    {"field": "cushion", "source": "player_play", "interpretation": "pre-snap distance to the defender across",
     "known_pre_snap": "yes", "observed_after_outcome": False, "available_pre_draft": False,
     "leakage_as_predictor": "future NFL context: forbidden in a pre-draft model",
     "changes_estimand": "yes: adjusts for how the defence chose to play the receiver"},
    {"field": "play_class", "source": "nfl_play_movement (from pass_result)", "interpretation": "pass · sack_or_scramble · no_pass (empty code INFERRED)",
     "known_pre_snap": "no", "observed_after_outcome": True, "available_pre_draft": False,
     "leakage_as_predictor": "play result: forbidden", "changes_estimand": "yes when used as a filter: selects plays by their result"},
    {"field": "team_coverage_man_zone", "source": "player_play", "interpretation": "coverage family of the defence",
     "known_pre_snap": "no (labelled after the play)", "observed_after_outcome": True, "available_pre_draft": False,
     "leakage_as_predictor": "future NFL context: forbidden", "changes_estimand": "yes: separation against man and against zone are different quantities"},
    {"field": "coverage_assignment", "source": "player_play", "interpretation": "assignment label of a defender (NO_ASSIGNMENT is a vendor token)",
     "known_pre_snap": "partly", "observed_after_outcome": True, "available_pre_draft": False,
     "leakage_as_predictor": "future NFL role: forbidden", "changes_estimand": "yes: reading NO_ASSIGNMENT as 'rushed the passer' is an inference"},
    {"field": "season", "source": "silver.games", "interpretation": "NFL season of the game",
     "known_pre_snap": "yes", "observed_after_outcome": False, "available_pre_draft": False,
     "leakage_as_predictor": "not a predictor; the draft cohort (known at the draft) is the allowed counterpart",
     "changes_estimand": "no: used for coverage and stability reporting"},
    {"field": "game_id", "source": "silver.games", "interpretation": "game: the unit of every reliability split",
     "known_pre_snap": "yes", "observed_after_outcome": False, "available_pre_draft": False,
     "leakage_as_predictor": "not a predictor", "changes_estimand": "no"},
)


# ============================================================================== contracts
CAND_SCHEMA = {"outcome_id": S, "value_field": S, "source_table": S, "source_files": S, "present": B, "dtype": S, "unit": S, "kind": S,
               "declared_definition": S, "event_definition": S, "zero_encoding": S, "missing_meaning": S, "attribution": S,
               "opportunity": S, "eligibility": S, "positions": S, "seasons": S, "n_rows": I, "n_observed": I, "n_zero": I,
               "n_positive": I, "observed_share": F, "limits": S, "definition_status": S, "status_basis": S}
CHECK_SCHEMA = {"outcome_id": S, "check": S, "observed": S, "passed": B, "blocking": B, "meaning": S}
ZERO_SCHEMA = {"field": S, "position_group": S, "dtype": S, "n_rows": I, "n_null": I, "n_zero": I, "n_positive": I, "n_negative": I,
               "share_null": F, "share_zero": F}
STRUCT_SCHEMA = {"outcome_id": S, "group_field": S, "group_value": S, "n_rows": I, "n_obs": I, "obs_share": F, "n_zero": I,
                 "n_positive": I, "mean_value": F}
CREDIT_SCHEMA = {"field": S, "total_credit": F, "n_plays": I, "median_rows_recorded": F}
DIST_SCHEMA = {"position_group": S, "target": S, "n_obs": I, "n_athletes": I, "mean": F, "sd": F, "p05": F, "p25": F, "median": F,
               "p75": F, "p95": F, "max": F, "share_extreme": F, "share_below_half_yard": F}
CELL_SCHEMA = {"route_ran": S, "lined_up_position": S, "n_obs": I, "n_athletes": I, "mean_value": F, "supported": B}
CONTEXT_SCHEMA = {"field": S, "source": S, "present": B, "observed_share": F, "interpretation": S, "known_pre_snap": S,
                  "observed_after_outcome": B, "available_pre_draft": B, "leakage_as_predictor": S, "changes_estimand": S}

CANDIDATES = TableContract("football_outcome_candidates_v1", "Candidate football outcomes: source, declared semantics, measured presence, semantic status.",
                           CAND_SCHEMA, key=("outcome_id",), area="candidate outcome",
                           docs={"definition_status": " · ".join(STATUSES),
                                 "n_zero": "recorded zeros among eligible player-plays of the declared positions (a null is not a zero)"})
CHECKS = TableContract("football_outcome_semantics_audit", "Rule-based consistency checks of the declared semantics of each candidate outcome.",
                       CHECK_SCHEMA, key=("outcome_id", "check"), area="candidate outcome × check",
                       docs={"blocking": "a failed blocking check makes the definition unresolved",
                             "passed": "null = informative measurement without a pass / fail rule"})
ZERO_MISSING = TableContract("football_outcome_zero_missing", "Null, zero and positive counts of the audited vendor fields by position group.",
                             ZERO_SCHEMA, key=("field", "position_group"), area="field × position group")
STRUCTURE = TableContract("football_outcome_structure", "Share of eligible player-plays on which an outcome is recorded, by value of a conditioning field.",
                          STRUCT_SCHEMA, key=("outcome_id", "group_field", "group_value"), area="candidate outcome × field × value")
CREDIT = TableContract("football_outcome_play_credit", "Total credit per play over the delivered vendor rows.", CREDIT_SCHEMA,
                       key=("field", "total_credit"), area="field × total credit",
                       docs={"total_credit": "sum over the DELIVERED player rows of a play (the delivered rows are not all 22 players)"})
SEP_DIST = TableContract("football_separation_distribution", "Distribution of separation at pass forward by position and target status.",
                         DIST_SCHEMA, key=("position_group", "target"), area="position group × target status")
CELLS = TableContract("football_separation_context_cells", "Route × alignment cells of the recorded separations: support by counts.",
                      CELL_SCHEMA, key=("route_ran", "lined_up_position"), area="route × alignment",
                      docs={"supported": "decided by observation and athlete counts only, never by the values"})
CONTEXT_T = TableContract("football_outcome_context_fields", "Context fields: availability, timing relative to the outcome, leakage as a predictor.",
                          CONTEXT_SCHEMA, key=("field",), area="context field")
CONTRACTS = (CANDIDATES, CHECKS, ZERO_MISSING, STRUCTURE, CREDIT, SEP_DIST, CELLS, CONTEXT_T)


# ============================================================================== expressions
def numeric(field: str) -> pl.Expr:
    return pl.col(field).cast(pl.Float64, strict=False).fill_nan(None)


def observed(df: pl.DataFrame, field: str) -> pl.Expr:
    return sf._observed(field, df.schema[field])


def flag(df: pl.DataFrame, field: str) -> pl.Expr:
    """1.0 / 0.0 / null. Strings: recognised tokens only (anything else stays unobserved)."""
    dt, col, none = df.schema[field], pl.col(field), pl.lit(None, dtype=pl.Float64)
    if dt == pl.Boolean:
        return col.cast(pl.Float64)
    if dt == pl.String:
        s = col.str.strip_chars().str.to_lowercase()
        return pl.when(s.is_in(list(sf.TRUE_TOKENS))).then(pl.lit(1.0)).when(s.is_in(list(sf.FALSE_TOKENS))).then(pl.lit(0.0)).otherwise(none)
    return (numeric(field) > 0).cast(pl.Float64)


def _value(df: pl.DataFrame, field: str) -> pl.Expr:
    """Numeric reading of any audited field: numbers as they are, booleans as 1 / 0, strings as 1 = a label is present."""
    dt = df.schema[field]
    if dt == pl.Boolean:
        return pl.col(field).cast(pl.Float64)
    if dt == pl.String:
        return pl.when(observed(df, field)).then(pl.lit(1.0)).otherwise(pl.lit(None, dtype=pl.Float64))
    return numeric(field)


def _declared(base: pl.DataFrame, positions: Sequence[str]) -> pl.DataFrame:
    return base.filter(pl.col("eligible") & pl.col("position_group").is_in(list(positions)))


# ============================================================================== inputs
def semantic_base(plays: pl.DataFrame, player_play: pl.LazyFrame | pl.DataFrame,
                  fields: Sequence[str] = AUDIT_FIELDS) -> tuple[pl.DataFrame, dict[str, int]]:
    """Tracked player-plays of notebook 05A (every row kept) + the audited vendor fields that exist, and the
    duplication audit of the vendor rows. A field absent from ``player_play`` is not created."""
    lf = player_play.lazy()
    have = lf.collect_schema().names()
    keep = [c for c in PLAY_COLS if c in plays.columns]
    take = [f for f in dict.fromkeys(fields) if f in have and f not in KEYS and f not in keep]
    pp = lf.select(*KEYS, *take).collect().with_columns([pl.col(k).cast(pl.Int64) for k in KEYS])
    uq = pp.unique(subset=KEYS, keep="first", maintain_order=True)
    dup = {"n_vendor_rows": pp.height, "n_player_plays": uq.height, "n_duplicate_rows": pp.height - uq.height,
           "n_conflicting_rows": pp.unique().height - uq.height}
    base = plays.select(*KEYS, *keep).join(uq, on=KEYS, how="left")
    assert base.height == plays.height, f"the join changed the number of player-plays ({plays.height:,} → {base.height:,})"
    assert_no_combine_columns(base, "08A.2 semantic base")
    return base, dup


# ============================================================================== tables
def zero_missing_table(base: pl.DataFrame, fields: Sequence[str] = AUDIT_FIELDS) -> pl.DataFrame:
    el = base.filter(pl.col("eligible"))
    parts = []
    for f in fields:
        if f not in base.columns:
            continue
        v = _value(base, f)
        d = el.select("position_group", v.alias("v"))
        t = (pl.concat([d, d.with_columns(pl.lit(ALL).alias("position_group"))]).group_by("position_group")
             .agg(pl.len().alias("n_rows"), pl.col("v").is_null().sum().alias("n_null"), (pl.col("v") == 0).sum().alias("n_zero"),
                  (pl.col("v") > 0).sum().alias("n_positive"), (pl.col("v") < 0).sum().alias("n_negative")))
        parts.append(t.with_columns(pl.lit(f).alias("field"), pl.lit(str(base.schema[f])).alias("dtype"),
                                    (pl.col("n_null") / pl.col("n_rows")).alias("share_null"),
                                    (pl.col("n_zero") / pl.col("n_rows")).alias("share_zero")).select(list(ZERO_SCHEMA)))
    if not parts:
        return pl.DataFrame(schema=ZERO_SCHEMA)
    return pl.concat(parts).cast(ZERO_SCHEMA).sort("field", "position_group")


def structure_table(base: pl.DataFrame, sem: OutcomeSemantics, by: Sequence[str], top: int = 25) -> pl.DataFrame:
    """Where is the outcome recorded? ``by`` holds field names; ``has:<field>`` groups by recorded / not recorded."""
    if sem.value_field not in base.columns:
        return pl.DataFrame(schema=STRUCT_SCHEMA)
    d, v, parts = _declared(base, sem.positions), numeric(sem.value_field), []
    for f in by:
        name = f.removeprefix("has:")
        if name not in base.columns:
            continue
        g = (pl.when(observed(base, name)).then(pl.lit("recorded")).otherwise(pl.lit(NULL)) if f.startswith("has:")
             else pl.col(name).cast(pl.String).fill_null(NULL))
        t = (d.select(g.alias("group_value"), v.alias("v")).group_by("group_value")
             .agg(pl.len().alias("n_rows"), pl.col("v").is_not_null().sum().alias("n_obs"), (pl.col("v") == 0).sum().alias("n_zero"),
                  (pl.col("v") > 0).sum().alias("n_positive"), pl.col("v").mean().alias("mean_value"))
             .sort("n_rows", "group_value", descending=[True, False]).head(top))
        parts.append(t.with_columns(pl.lit(sem.outcome_id).alias("outcome_id"), pl.lit(f).alias("group_field"),
                                    (pl.col("n_obs") / pl.col("n_rows")).alias("obs_share")).select(list(STRUCT_SCHEMA)))
    return pl.concat(parts).cast(STRUCT_SCHEMA) if parts else pl.DataFrame(schema=STRUCT_SCHEMA)


def per_play_credit(player_play: pl.LazyFrame | pl.DataFrame, fields: Sequence[str]) -> pl.DataFrame:
    """Total credit per play over the delivered vendor rows (duplicated rows counted once). A total above 1 for a
    credit that is supposed to be shared between players contradicts the declared attribution."""
    lf = player_play.lazy()
    have, parts = lf.collect_schema().names(), []
    for f in fields:
        if f not in have:
            continue
        t = (lf.select(*KEYS, numeric(f).alias("v")).unique(subset=KEYS).filter(pl.col("v").is_not_null())
             .group_by(PLAY_KEYS).agg(pl.col("v").sum().alias("total_credit"), pl.len().alias("k"))
             .group_by("total_credit").agg(pl.len().alias("n_plays"), pl.col("k").median().alias("median_rows_recorded")).collect())
        parts.append(t.with_columns(pl.lit(f).alias("field")).select(list(CREDIT_SCHEMA)))
    return pl.concat(parts).cast(CREDIT_SCHEMA).sort("field", "total_credit") if parts else pl.DataFrame(schema=CREDIT_SCHEMA)


def fo01_rows(base: pl.DataFrame) -> pl.DataFrame:
    """Recorded separations of eligible WR / TE player-plays with their context labels (one row per player-play)."""
    sem = BY_ID[FO01]
    if sem.value_field not in base.columns:
        return pl.DataFrame(schema={"nfl_id": I, "position_group": S, "game_id": I, "season": I, "value": F, "target": S,
                                    "route_ran": S, "lined_up_position": S})
    tgt = (pl.when(flag(base, "target") == 1.0).then(pl.lit("targeted")).when(flag(base, "target") == 0.0).then(pl.lit("not_targeted"))
           .otherwise(pl.lit(NULL)) if "target" in base.columns else pl.lit(NULL))
    route = pl.col("route_ran").cast(pl.String) if "route_ran" in base.columns else pl.lit(None, dtype=pl.String)
    return (_declared(base, sem.positions).with_columns(numeric(sem.value_field).alias("value")).filter(pl.col("value").is_not_null())
            .select("nfl_id", "position_group", "game_id", "season", "value", tgt.alias("target"), route.alias("route_ran"),
                    pl.col("lined_up_position").cast(pl.String)))


def fo01_distribution(rows: pl.DataFrame, extreme_yd: float) -> pl.DataFrame:
    if rows.is_empty():
        return pl.DataFrame(schema=DIST_SCHEMA)
    r = rows.select("nfl_id", "position_group", "target", "value")
    grid = pl.concat([r, r.with_columns(pl.lit(ALL).alias("position_group")), r.with_columns(pl.lit(ALL).alias("target")),
                      r.with_columns(pl.lit(ALL).alias("position_group"), pl.lit(ALL).alias("target"))])
    v = pl.col("value")
    return (grid.group_by("position_group", "target")
            .agg(pl.len().alias("n_obs"), pl.col("nfl_id").n_unique().alias("n_athletes"), v.mean().alias("mean"), v.std().alias("sd"),
                 v.quantile(0.05).alias("p05"), v.quantile(0.25).alias("p25"), v.median().alias("median"), v.quantile(0.75).alias("p75"),
                 v.quantile(0.95).alias("p95"), v.max().alias("max"), (v > extreme_yd).mean().alias("share_extreme"),
                 (v < 0.5).mean().alias("share_below_half_yard"))
            .cast(DIST_SCHEMA).sort("position_group", "target"))


def route_alignment_cells(rows: pl.DataFrame, min_obs: int, min_athletes: int) -> pl.DataFrame:
    """Route × alignment cells among recorded separations WITH a route label. Support is decided by counts only."""
    r = rows.filter(pl.col("route_ran").is_not_null() & (pl.col("route_ran").str.strip_chars() != "") & pl.col("lined_up_position").is_not_null())
    if r.is_empty():
        return pl.DataFrame(schema=CELL_SCHEMA)
    return (r.group_by("route_ran", "lined_up_position")
            .agg(pl.len().alias("n_obs"), pl.col("nfl_id").n_unique().alias("n_athletes"), pl.col("value").mean().alias("mean_value"))
            .with_columns(((pl.col("n_obs") >= min_obs) & (pl.col("n_athletes") >= min_athletes)).alias("supported"))
            .cast(CELL_SCHEMA).sort("n_obs", "route_ran", "lined_up_position", descending=[True, False, False]))


def context_table(base: pl.DataFrame) -> pl.DataFrame:
    el = base.filter(pl.col("eligible"))
    rows = []
    for c in CONTEXT:
        f = c["field"]
        present = f in base.columns
        share = float(el.select(observed(base, f).mean()).item()) if present and el.height else None
        rows.append({**c, "present": present, "observed_share": share})
    return pl.DataFrame(rows, schema=CONTEXT_SCHEMA)


def tracked_per_play(plays: pl.DataFrame) -> dict[str, float]:
    """How many tracked athletes one play holds. With a handful per play the nearest defender cannot be recomputed."""
    k = plays.group_by(PLAY_KEYS).agg(pl.len().alias("k"))["k"]
    return {"median": float(k.median()), "p95": float(k.quantile(0.95)), "max": float(k.max())} if k.len() else {"median": 0.0, "p95": 0.0, "max": 0.0}


# ============================================================================== checks and status
def _stats(d: pl.DataFrame, field: str) -> dict[str, Any]:
    v = numeric(field)
    return d.select(pl.len().alias("n"), v.is_not_null().sum().alias("obs"), v.min().alias("min"), v.median().alias("median"),
                    v.max().alias("max"), (v == 0).sum().alias("zero"), (v > 0).sum().alias("pos")).row(0, named=True)


def _share(d: pl.DataFrame, cond: pl.Expr) -> float | None:
    return float(d.select(cond.mean()).item()) if d.height else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f} %"


def semantic_checks(base: pl.DataFrame, dup: dict[str, int], credit: pl.DataFrame, tracked: dict[str, float],
                    sems: Sequence[OutcomeSemantics] = SEMANTICS) -> pl.DataFrame:
    """Internal-consistency checks. Blocking checks decide ``unresolved``; the others are measurements to be read."""
    rows: list[dict[str, Any]] = []

    def add(oid: str, check: str, observed_: str, passed: bool | None, blocking: bool, meaning: str) -> None:
        rows.append({"outcome_id": oid, "check": check, "observed": observed_, "passed": passed, "blocking": blocking, "meaning": meaning})

    for s in sems:
        oid, f = s.outcome_id, s.value_field
        present = f in base.columns
        add(oid, "field_present", f"{f}: {'present' if present else 'ABSENT'}", present, True, "the value field exists in silver.player_play")
        add(oid, "one_row_per_player_play", f"{dup['n_conflicting_rows']:,} conflicting duplicates ({dup['n_duplicate_rows']:,} duplicated rows)",
            dup["n_conflicting_rows"] == 0, True, "duplicated vendor rows carry the same values (the first is kept)")
        if not present:
            continue
        d = _declared(base, s.positions)
        st, v = _stats(d, f), numeric(f)
        add(oid, "recorded_share", f"{st['obs']:,} of {st['n']:,} eligible player-plays ({_pct(st['obs'] / max(st['n'], 1))})", None, False,
            "share of eligible player-plays of the declared positions with a recorded value")
        add(oid, "recorded_zero", f"{st['zero']:,} recorded zeros · {st['pos']:,} positive", None, False,
            "a recorded zero is a statement by the vendor; a null is not")
        if s.allowed_values:
            bad = int(d.select((v.is_not_null() & ~v.is_in(list(s.allowed_values))).sum()).item())
            add(oid, "values_in_declared_set", f"{bad:,} recorded values outside {list(s.allowed_values)}", bad == 0, True,
                "a credit takes the declared values only")
            tot = credit.filter(pl.col("field") == f)
            if tot.height:
                n_all, n_over = int(tot["n_plays"].sum()), int(tot.filter(pl.col("total_credit") > 1 + 1e-9)["n_plays"].sum())
                add(oid, "play_total_at_most_one", f"{n_over:,} of {n_all:,} plays with a total above 1 over the delivered rows",
                    (n_over <= 0.001 * n_all) if oid == FO13 else None, oid == FO13,
                    "a shared credit cannot exceed 1 per play (only the delivered player rows are summed)")
        if oid == FO01 and not st["obs"]:
            add(oid, "non_negative", "no recorded value", False, True, "a distance cannot be negative")
        elif oid == FO01:
            add(oid, "non_negative", f"min {st['min']:.3g}", st["min"] is not None and st["min"] >= 0, True, "a distance cannot be negative")
            add(oid, "unit_plausible_yards", f"median {st['median']:.3g}", st["median"] is not None and 0.5 <= st["median"] <= 10.0, True,
                "a median nearest-defender distance outside 0.5–10 yd contradicts 'yards' (range declared before reading)")
            if "target" in base.columns:
                nt, tg = d.filter(flag(base, "target") == 0.0), d.filter(flag(base, "target") == 1.0)
                a, b = _share(nt, v.is_not_null()), _share(tg, v.is_not_null())
                add(oid, "recorded_for_non_targeted", f"targeted {_pct(b)} · not targeted {_pct(a)}", (a or 0.0) >= 0.5, False,
                    "if the value existed for targeted receivers only, the quarterback's choice would select the plays")
            if "route_ran" in base.columns:
                rec = d.filter(v.is_not_null())
                a, b = _share(rec, observed(base, "route_ran")), _share(d.filter(observed(base, "route_ran")), v.is_not_null())
                add(oid, "route_label_coverage", f"{_pct(a)} of recorded values carry a route label · {_pct(b)} of labelled routes carry a value",
                    None, False, "the route label documents that the player was in a route (denominator S1)")
            add(oid, "nearest_defender_recomputable", f"median {tracked['median']:.0f} tracked athletes per play (max {tracked['max']:.0f})",
                tracked["median"] >= 11, False,
                "with only the project athletes tracked, 'distance to the nearest defender' cannot be recomputed from tracking")
        if oid == FO12:
            for comp in ("assist", "tackle_for_loss", "caused_forced_fumble"):
                if comp in base.columns:
                    z = d.filter(v == 0)
                    add(oid, f"zero_explained_by_{comp}", f"{_pct(_share(z, _value(base, comp) > 0))} of recorded zeros have {comp} > 0", None, False,
                        "what a recorded 0 means: a row created by another credited statistic")
            add(oid, "opportunity_field", "no field records tackle attempts or plays run at the player", False, False,
                "a tackle RATE per opportunity is undefined; only credits per snap on the field can be counted")
        if oid == FO13:
            pos = d.filter(v > 0)
            if "pass_result" in base.columns:
                add(oid, "positive_on_sack_coded_plays",
                    f"{_pct(_share(pos, pl.col('pass_result').cast(pl.String).str.strip_chars() == 'S'))} of positive credits on plays coded 'S'",
                    None, False, "consistency with the play-level result code (its meaning is itself inferred)")
            if "play_class" in base.columns:
                rec = d.filter(v.is_not_null())
                add(oid, "recorded_on_no_pass_plays", f"{_pct(_share(rec, pl.col('play_class') == 'no_pass'))} of recorded rows are 'no_pass' plays",
                    None, False, "if zeros are recorded on run plays, the recorded rows are not pass-rush opportunities")
            if "coverage_assignment" in base.columns:
                rec = d.filter(v.is_not_null())
                add(oid, "recorded_with_coverage_label", f"{_pct(_share(rec, observed(base, 'coverage_assignment')))} of recorded rows carry a coverage label",
                    None, False, "the recorded rows coincide with defensive pass-play rows, whatever the player's job on them")
            add(oid, "rush_participation_field", "no field records that the player rushed the passer", False, False,
                "sacks per PASS-RUSH snap is undefined; sacks per recorded pass-defence row is the supported quantity")
    return pl.DataFrame(rows, schema=CHECK_SCHEMA)


def semantic_status(checks: pl.DataFrame, evidence: dict[str, dict[str, str]] | None = None,
                    sems: Sequence[OutcomeSemantics] = SEMANTICS) -> dict[str, dict[str, str]]:
    """{outcome_id: {status, basis}}. ``evidence`` = analyst record {outcome_id: {status, source, definition}}:
    ``confirmed_source`` needs a named source AND the quoted definition; nothing is confirmed by a column name."""
    out: dict[str, dict[str, str]] = {}
    for s in sems:
        failed = checks.filter((pl.col("outcome_id") == s.outcome_id) & pl.col("blocking") & ~pl.col("passed"))["check"].to_list()
        ev = (evidence or {}).get(s.outcome_id, {})
        if ev.get("status") not in (None, *STATUSES):
            raise ValueError(f"{s.outcome_id}: evidence status must be one of {STATUSES}, got {ev.get('status')!r}")
        if failed:
            out[s.outcome_id] = {"status": "unresolved", "basis": "blocking check(s) failed: " + ", ".join(failed)}
        elif ev.get("status") == "confirmed_source":
            if not (str(ev.get("source", "")).strip() and str(ev.get("definition", "")).strip()):
                raise ValueError(f"{s.outcome_id}: 'confirmed_source' needs a named 'source' and the quoted 'definition'")
            out[s.outcome_id] = {"status": "confirmed_source", "basis": f"{ev['source']}: {ev['definition']}"}
        elif ev.get("status") == "unresolved":
            out[s.outcome_id] = {"status": "unresolved", "basis": "declared unresolved by the analyst: " + str(ev.get("definition", ""))}
        else:
            out[s.outcome_id] = {"status": s.definition_status, "basis": "inferred from the column name and internal consistency; no vendor definition recorded"}
    return out


def candidates_table(base: pl.DataFrame, schemas: dict[str, dict[str, str]], source_files: Sequence[str],
                     status: dict[str, dict[str, str]], sems: Sequence[OutcomeSemantics] = SEMANTICS) -> pl.DataFrame:
    rows = []
    for s in sems:
        present = s.value_field in base.columns
        row = {"outcome_id": s.outcome_id, "value_field": s.value_field, "source_table": s.source_table,
               "source_files": " · ".join(source_files), "present": present, "dtype": schemas.get("player_play", {}).get(s.value_field, ""),
               "unit": s.unit, "kind": s.kind, "declared_definition": s.declared_definition, "event_definition": s.event_definition,
               "zero_encoding": s.zero_encoding, "missing_meaning": s.missing_meaning, "attribution": s.attribution,
               "opportunity": s.opportunity, "eligibility": s.eligibility, "positions": " · ".join(s.positions), "seasons": "",
               "n_rows": None, "n_observed": None, "n_zero": None, "n_positive": None, "observed_share": None, "limits": s.limits,
               "definition_status": status[s.outcome_id]["status"], "status_basis": status[s.outcome_id]["basis"]}
        if present:
            d = _declared(base, s.positions)
            st = _stats(d, s.value_field)
            seasons = d.filter(numeric(s.value_field).is_not_null())["season"].drop_nulls().unique().sort().to_list()
            row |= {"seasons": " · ".join(str(x) for x in seasons), "n_rows": int(st["n"]), "n_observed": int(st["obs"]),
                    "n_zero": int(st["zero"]), "n_positive": int(st["pos"]), "observed_share": float(st["obs"]) / max(int(st["n"]), 1)}
        rows.append(row)
    return pl.DataFrame(rows, schema=CAND_SCHEMA)


STRUCTURE_FIELDS = {FO01: ("target", "has:route_ran", "play_class", "position_group", "season", "lined_up_position", "route_ran",
                           "team_coverage_man_zone"),
                    FO12: ("position_group", "play_class", "has:assist", "assist", "tackle_for_loss", "season", "lined_up_position"),
                    FO13: ("position_group", "play_class", "has:coverage_assignment", "coverage_assignment", "blitzing", "season",
                           "lined_up_position")}


def run_semantics(base: pl.DataFrame, dup: dict[str, int], player_play: pl.LazyFrame | pl.DataFrame, plays: pl.DataFrame,
                  schemas: dict[str, dict[str, str]], source_files: Sequence[str], evidence: dict[str, dict[str, str]] | None,
                  extreme_yd: float, min_cell_obs: int, min_cell_athletes: int) -> dict[str, pl.DataFrame]:
    """The eight tables of the semantics audit, keyed by contract name."""
    credit = per_play_credit(player_play, ("tackle", "assist", "sack"))
    checks = semantic_checks(base, dup, credit, tracked_per_play(plays))
    status = semantic_status(checks, evidence)
    rows = fo01_rows(base)
    structure = [structure_table(base, s, STRUCTURE_FIELDS[s.outcome_id]) for s in SEMANTICS]
    return {CANDIDATES.name: candidates_table(base, schemas, source_files, status), CHECKS.name: checks,
            ZERO_MISSING.name: zero_missing_table(base), STRUCTURE.name: pl.concat(structure).cast(STRUCT_SCHEMA),
            CREDIT.name: credit, SEP_DIST.name: fo01_distribution(rows, extreme_yd),
            CELLS.name: route_alignment_cells(rows, min_cell_obs, min_cell_athletes), CONTEXT_T.name: context_table(base)}


def status_from_table(candidates: pl.DataFrame) -> dict[str, dict[str, str]]:
    return {r["outcome_id"]: {"status": r["definition_status"], "basis": r["status_basis"]} for r in candidates.iter_rows(named=True)}


__all__ = ["ALL", "AUDIT_FIELDS", "BY_ID", "CANDIDATES", "CELLS", "CHECKS", "CONTEXT", "CONTEXT_T", "CONTRACTS", "CREDIT", "FO01",
           "FO12", "FO13", "NULL", "OutcomeSemantics", "SEMANTICS", "SEP_DIST", "STATUSES", "STRUCTURE", "ZERO_MISSING",
           "candidates_table", "context_table", "definition_hash", "flag", "fo01_distribution", "fo01_rows", "numeric", "observed",
           "per_play_credit", "route_alignment_cells", "run_semantics", "semantic_base", "semantic_checks", "semantic_status",
           "status_from_table", "structure_table", "tracked_per_play", "zero_missing_table"]
