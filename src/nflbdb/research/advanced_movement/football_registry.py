"""Milestone 08B.0 — outcome resolution and the immutable football-outcome registry (NumPy / Polars; reads 08A.2 tables).

    LINEAGE               static declarations of every vendor field used by an outcome or an opportunity denominator
    lineage_table         declaration + measured presence / null / zero shares + recorded source evidence
    repository_evidence   non-executing text search of the repository documentation for the field names (to be READ by
                          the analyst; a hit never confirms a definition by itself)
    equivalence           are two definitions operationally identical on the observed data (FO01.S0 versus FO01.S1)?
    definitions_table     one row per opportunity definition: source class and the role it may take
    record_decision       explicit analyst decision → ``football_outcome_registry_<version>.json`` (written once)
    load_registry / branch / require_branch       the gate every later 08B notebook calls

Source classes        CONFIRMED          recorded denominator, vendor definition of every needed field recorded by the analyst
                      DERIVED_VERIFIED   derived from confirmed fields by a documented, checked transformation
                      PROVISIONAL        internally consistent, definition inferred (no authoritative source recorded)
                      BLOCKED            unresolved, unavailable, rejected by the screen, or a sensitivity-only denominator
Decisions             FREEZE        the definitions are locked for Milestone 08B. ``approved_status`` = ``frozen_confirmed`` only when
                                    every target is CONFIRMED / DERIVED_VERIFIED; otherwise ``frozen_exploratory`` (needs the
                                    acknowledgement). A freeze never raises a source class.
                      PROVISIONAL   the definitions are recorded for review and may still change (in a NEW version). With the
                                    acknowledgement: ``provisional_exploratory`` (exploratory analyses allowed, labelled so);
                                    without it: ``provisional_review`` (football transfer blocked).
                      STOP          no football outcome is accepted: football transfer is blocked; the descriptive notebooks
                                    (08B.4–08B.6) continue.
Branches              frozen · exploratory · blocked — computed from the registry, never from a notebook flag.

No Combine value is read here. The 08A.2 registry (``football_outcome_targets_<version>.json``) and the Milestone 05
registry are never written.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import atomic_write_json, get_logger, read_json, utc_now
from ..contracts import TableContract
from . import opportunity_validation as ov
from . import outcome_semantics as osm
from .nfl_outcomes import assert_no_combine_columns

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
REGISTRY_VERSION_CODE = "1"
REGISTRY_FILE = "football_outcome_registry"
CLASSES = ("CONFIRMED", "DERIVED_VERIFIED", "PROVISIONAL", "BLOCKED")
DECISIONS = ("FREEZE", "PROVISIONAL", "STOP")
BRANCHES = ("frozen", "exploratory", "blocked")
ROLES = ("primary_or_secondary", "secondary_only", "sensitivity_only", "redundant", "blocked")
DEFINITION_VERSION = "08B0.1"
# (model, reference): recorded in the registry BEFORE any NFL outcome is related to a Combine value
HYPOTHESES_DEFAULT: dict[str, Any] = {
    "primary": ["M2", "M1"], "secondary": [["M5", "M1"]],
    "controls": [["M3", "M1"], ["M4", "M1"], ["M6", "M1"], ["M5", "M2"], ["M5", "M6"]],
    "primary_metric": "paired Δ RMSE (reference − model) on athlete-held-out predictions, 95 % athlete bootstrap",
    "note": "M2 = M1 + the pre-selected engineered speed-development family (ENG). Chosen first because it is one interpretable, "
            "low-dimensional family — not because it is the most or the least sophisticated representation."}
INTERPRETATION = {
    "FO01.S0": "recorded separation at pass forward, mean over recorded player-plays [yd] — an INTERACTION outcome (receiver, nearest "
               "defender, coverage, route, throw timing); NOT pure receiver ability",
    "FO01.S1": "same as FO01.S0 restricted to rows with a route label",
    "FO01.S2": "separation relative to the mean of the recorded route × alignment cell [yd] — removes route mix and NFL role, which are "
               "partly properties of the player; context SENSITIVITY",
    "FO01.S0T": "separation on targeted plays only — plays selected on the outcome by the quarterback; sensitivity",
    "FO01.S0N": "separation on non-targeted plays only — complement of a selection on the outcome; sensitivity",
    "FO12.T0": "share of credited-statistic rows that are solo tackles — NOT a tackle rate per opportunity",
    "FO12.T1": "credited solo tackles per tracked snap — exposure, NOT tackle success probability",
    "FO12.T2": "PARTICIPATION-CREDIT rate: credited tackles or assists per tracked snap — NOT tackle success probability; snaps on "
               "the field are exposure, not tackle opportunities",
    "FO13.K0": "sack credit per recorded pass-defence row — NOT a pass-rush win rate; that the player rushed the passer is not recorded",
    "FO13.K1": "sack credit per row without a coverage assignment — NO_ASSIGNMENT is NOT verified pass rushing; sensitivity",
    "FO13.K2": "sack credit per tracked snap (run plays in the denominator) — NOT a pass-rush win rate; sensitivity"}
SECONDARY_ONLY = ("FO12.T2", "FO13.K0")          # opportunity-denominator limits: never a primary target
SENSITIVITY_ONLY = ("FO01.S2", "FO13.K2")        # S2 is promoted only by its own reliability (c_reliability of the screen)


class FootballTransferBlocked(RuntimeError):
    """The registry does not allow a football transfer analysis in this branch."""


# ============================================================================== field lineage (static declarations)
@dataclass(frozen=True)
class FieldLineage:
    field: str
    used_by: str                    # outcome or denominator IDs that need the field
    role: str                       # outcome_value · opportunity_filter · context · identifier · eligibility
    source_dataset: str
    units: str
    declared_meaning: str           # what the column name suggests — an ASSUMPTION until evidence is recorded
    timing: str
    attribution: str
    required_observations: str
    missing_rule: str
    zero_rule: str
    allowed_context: str
    limitations: str


_PP = "silver.player_play (vendor player-play table of the competition data)"
LINEAGE: tuple[FieldLineage, ...] = (
    FieldLineage("separation_at_pass_forward", "FO01.*", "outcome_value", _PP, "yd (inferred from the value range)",
                 "distance from the player to the nearest defender at the frame tagged pass_forward",
                 "one value per player-play at the pass release; the release frame is not delivered with the value",
                 "recorded on the receiver's row; the defender is not named", "a recorded (non-null) value on an eligible WR / TE player-play",
                 "null = not recorded; never read as 0", "0 would be a physical distance of zero; no structural zero expected",
                 "position group (comparison population); route / alignment only as a documented sensitivity (changes the estimand)",
                 "nearest defender cannot be recomputed (only project athletes are tracked); interaction outcome"),
    FieldLineage("route_ran", "FO01.S1 · FO01.S2", "context", _PP, "label", "route label of the player on the play",
                 "labelled after the play (an assignment label)", "player-level", "a non-empty label",
                 "null / empty = no route label; a missing route is never inferred", "not applicable",
                 "sensitivity only; forbidden as a pre-draft predictor (future NFL usage)", "route mix is partly a property of the player"),
    FieldLineage("lined_up_position", "FO01.S2", "context", _PP + " via nfl_play_movement", "label", "alignment of the player on the play",
                 "pre-snap", "player-level", "a non-null label", "null = not recorded", "not applicable",
                 "sensitivity only; forbidden as a pre-draft predictor (future NFL role)", "the team chooses the alignment"),
    FieldLineage("cushion", "FO03 (context, blocked as an outcome)", "context", _PP, "yd (inferred)", "pre-snap distance to the defender across",
                 "pre-snap", "player-level", "not used by any approved definition", "null = not recorded", "unknown",
                 "not used in Milestone 08B", "a defensive choice: a contextual variable, not an outcome of the receiver"),
    FieldLineage("target", "FO01.S0T · FO01.S0N", "opportunity_filter", _PP, "flag", "the pass was thrown to the player",
                 "known after the throw", "player-level", "a recognised true / false token", "null = unknown; never read as 'not targeted'",
                 "false is a recorded statement", "sensitivity split only: never a filter of the primary target, never a predictor",
                 "outcome-dependent: the quarterback throws to the open receiver"),
    FieldLineage("tackle", "FO12.*", "outcome_value", _PP, "count per player-play (0 / 1)", "credited solo tackle of the player on the play",
                 "credited statistic of the play (not a tracking event)", "player-level credit", "FO12.T0: a recorded value; T1 / T2: any eligible snap",
                 "null = no tackle statistic recorded; 'no credit' is an INFERENCE of the structural denominators",
                 "a recorded 0 exists on rows created by another credited statistic (assist)", "position group",
                 "no field records a tackle opportunity; credit depends on role, scheme, run direction, playing time"),
    FieldLineage("assist", "FO12.T2", "outcome_value", _PP, "count per player-play", "credited assisted tackle of the player on the play",
                 "credited statistic of the play", "player-level credit", "any eligible snap (null read as no credit — inference)",
                 "null = no statistic recorded", "recorded 0 on credited-tackle rows", "position group", "solo versus assisted is a scorer's judgement"),
    FieldLineage("sack", "FO13.*", "outcome_value", _PP, "sack credit per player-play (0 / 0.5 / 1)", "sack credit of the player on the play (0.5 = shared)",
                 "credited statistic of the play", "player-level credit share", "FO13.K0: a recorded value; K2: any eligible snap",
                 "null = no sack statistic recorded (e.g. a run play); never read as 0 in K0", "0 is recorded explicitly on a subset of plays",
                 "position group", "rare event; rush assignment, protection, quarterback and game script are unobserved"),
    FieldLineage("coverage_assignment", "FO13.K1", "opportunity_filter", _PP, "label", "assignment label of a defender; NO_ASSIGNMENT is a vendor token",
                 "labelled after the play", "player-level", "a non-empty label", "null = not recorded", "not applicable",
                 "sensitivity only", "NO_ASSIGNMENT is NOT verified pass rushing"),
    FieldLineage("game_id · play_id · nfl_id", "every definition", "identifier", _PP + " · silver.games", "identifier",
                 "player-play key; the game is the unit of every reliability split", "—", "—", "one row per player-play (duplicates audited in 08A.2)",
                 "a null key is a defect", "not applicable", "grouping only: never a predictor", "an athlete's games and seasons stay in one fold"),
    FieldLineage("eligible", "every definition", "eligibility", "nfl_play_movement (notebook 05A)", "flag",
                 "eligible window: REG season, in-scope position, not nullified by penalty, usable quality tier", "per player-play", "—",
                 "true", "false = excluded before any denominator", "not applicable", "defines the population", "tracked project athletes only"),
    FieldLineage("position_group", "every definition", "eligibility", "gold.dim_athlete", "label", "roster position group of the athlete",
                 "known at the draft", "athlete-level", "in the declared positions of the definition", "null = out of scope", "not applicable",
                 "comparison population; the only NFL-side label also available pre-draft", "WR and TE are different populations"),
)
LINEAGE_T = TableContract(
    "football_outcome_source_lineage", "Source lineage and semantic confidence of every field used by a football outcome or denominator.",
    {"field": S, "outcome": S, "role": S, "source_dataset": S, "present": B, "dtype": S, "units": S, "semantic_meaning": S,
     "semantic_confidence": S, "timing": S, "player_attribution": S, "required_observations": S, "missing_value_rule": S,
     "zero_value_rule": S, "share_null": F, "share_zero": F, "allowed_context": S, "limitations": S, "source_evidence": S,
     "definition_version": S}, key=("field",), area="field",
    docs={"semantic_confidence": "confirmed = the analyst recorded a named source AND the quoted definition · inferred = from the column "
                                 "name and internal consistency only", "share_null": "among eligible player-plays, all positions (08A.2)"})
REPO_EVIDENCE = TableContract(
    "football_outcome_repository_mentions", "Lines of repository documentation that mention an audited field (to be read by the analyst).",
    {"field": S, "file": S, "line": I, "text": S}, key=("field", "file", "line"), area="field × file × line")
DEFINITIONS = TableContract(
    "football_outcome_definition_classes", "Source class and admissible role of every opportunity definition.",
    {"definition_id": S, "outcome_id": S, "label": S, "formula": S, "unit": S, "legitimacy": S, "semantic_status": S, "source_class": S,
     "class_basis": S, "role_allowed": S, "redundant_with": S, "screen_status": S, "n_supported": I, "n_linked_all_models": I, "rel_sb": F,
     "rel_lo": F, "rel_hi": F, "exposure_rho": F, "interpretation_label": S}, key=("definition_id",), area="definition",
    docs={"source_class": " · ".join(CLASSES), "role_allowed": " · ".join(ROLES)})
EQUIVALENCE = TableContract(
    "football_outcome_definition_equivalence", "Operational comparison of two definitions of one outcome on the observed athletes.",
    {"definition_a": S, "definition_b": S, "n_athletes_a": I, "n_athletes_b": I, "n_common": I, "n_same_observations": I,
     "max_abs_value_difference": F, "spearman": F, "identical": B, "reading": S}, key=("definition_a", "definition_b"), area="definition pair")
GATE_A = TableContract("football_gate_a_semantics", "Gate A — outcome semantics.", {"criterion": S, "passed": B, "detail": S},
                       key=("criterion",), area="criterion")


def lineage_table(zero_missing: pl.DataFrame, evidence: dict[str, dict[str, str]] | None = None) -> pl.DataFrame:
    """Declaration + what 08A.2 measured (dtype, null and zero shares among eligible player-plays) + recorded evidence.
    ``evidence``: {field: {"source": …, "definition": …}} — both are needed for ``confirmed``; nothing is invented."""
    zm = {r["field"]: r for r in zero_missing.filter(pl.col("position_group") == osm.ALL).iter_rows(named=True)}
    rows = []
    for f in LINEAGE:
        ev = (evidence or {}).get(f.field, {})
        ok = bool(str(ev.get("source", "")).strip() and str(ev.get("definition", "")).strip())
        m = zm.get(f.field, {})
        rows.append({"field": f.field, "outcome": f.used_by, "role": f.role, "source_dataset": f.source_dataset,
                     "present": bool(m) or f.role in ("identifier", "eligibility") or f.field == "lined_up_position",
                     "dtype": m.get("dtype", ""), "units": f.units,
                     "semantic_meaning": ev["definition"] if ok else f.declared_meaning + " (INFERRED)",
                     "semantic_confidence": "confirmed" if ok else "inferred", "timing": f.timing, "player_attribution": f.attribution,
                     "required_observations": f.required_observations, "missing_value_rule": f.missing_rule,
                     "zero_value_rule": f.zero_rule, "share_null": m.get("share_null"), "share_zero": m.get("share_zero"),
                     "allowed_context": f.allowed_context, "limitations": f.limitations,
                     "source_evidence": f"{ev['source']}" if ok else "none recorded: no vendor data dictionary found in the repository",
                     "definition_version": DEFINITION_VERSION})
    return pl.DataFrame(rows, schema=LINEAGE_T.columns)


def repository_evidence(files: Sequence[Path], fields: Sequence[str], rel: Any = str, max_bytes: int = 2_000_000,
                        max_hits: int = 20) -> pl.DataFrame:
    """Lines of the given documentation files that mention an audited field. A plain text search: nothing is executed
    and a hit is NOT a definition — the analyst reads it and, if it is authoritative, quotes it in the evidence cell."""
    rows, seen = [], dict.fromkeys(fields, 0)
    for p in files:
        try:
            if not p.is_file() or p.stat().st_size > max_bytes:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for n, line in enumerate(text, 1):
            for f in fields:
                if f in line and seen[f] < max_hits:
                    seen[f] += 1
                    rows.append({"field": f, "file": str(rel(p)), "line": n, "text": line.strip()[:240]})
    return pl.DataFrame(rows, schema=REPO_EVIDENCE.columns).unique(subset=["field", "file", "line"], maintain_order=True)


# ============================================================================== definitions
def equivalence(athletes: pl.DataFrame, a: str, b: str) -> dict[str, Any]:
    """Are definitions ``a`` and ``b`` the same quantity on the observed athletes? Identical = same athletes, same
    observation counts and the same values to 1e-9 (then one of them is redundant, not a second piece of evidence)."""
    ta = athletes.filter(pl.col("denominator_id") == a).select("nfl_id", pl.col("n_obs").alias("na"), pl.col("value").alias("va"))
    tb = athletes.filter(pl.col("denominator_id") == b).select("nfl_id", pl.col("n_obs").alias("nb"), pl.col("value").alias("vb"))
    j = ta.join(tb, on="nfl_id", how="inner")
    same_n = int((j["na"] == j["nb"]).sum()) if j.height else 0
    diff = float((j["va"] - j["vb"]).abs().max()) if j.height else None
    rho = ov.spearman(j["va"].to_numpy(), j["vb"].to_numpy()) if j.height >= 3 else float("nan")
    identical = bool(j.height and j.height == ta.height == tb.height and same_n == j.height and diff is not None and diff <= 1e-9)
    reading = (f"{b} is operationally identical to {a} on the observed data: retain {a}, classify {b} as redundant" if identical
               else f"{a} and {b} differ on the observed data: they are distinct definitions")
    return {"definition_a": a, "definition_b": b, "n_athletes_a": ta.height, "n_athletes_b": tb.height, "n_common": j.height,
            "n_same_observations": same_n, "max_abs_value_difference": diff, "spearman": rho if np.isfinite(rho) else None,
            "identical": identical, "reading": reading}


def equivalence_table(athletes: pl.DataFrame, pairs: Sequence[tuple[str, str]] = (("FO01.S0", "FO01.S1"),)) -> pl.DataFrame:
    return pl.DataFrame([equivalence(athletes, a, b) for a, b in pairs], schema=EQUIVALENCE.columns)


def classify(d: ov.Denominator, screen_row: dict[str, Any], semantic: str, field_confirmed: dict[str, bool]) -> tuple[str, str]:
    """(source class, basis) of one definition. A class is never raised by a decision; ``CONFIRMED`` needs recorded
    vendor evidence for the outcome AND for every field the definition needs."""
    if not screen_row.get("available", False):
        return "BLOCKED", "a needed vendor field is absent: " + ", ".join(d.needs)
    if d.legitimacy not in ov.FREEZABLE:
        return "BLOCKED", f"{d.legitimacy} denominator (sensitivity only): {d.assumption}"
    if semantic == "unresolved":
        return "BLOCKED", "the definition of the outcome is unresolved (a blocking consistency check failed)"
    if screen_row.get("status") == "rejected":
        return "BLOCKED", "rejected by the 08A.2 screen: " + str(screen_row.get("reason", ""))
    missing = [f for f in d.needs if not field_confirmed.get(f, False)]
    if semantic != "confirmed_source" or missing:
        why = "no authoritative definition recorded for " + (", ".join(missing) if missing else "the outcome")
        return "PROVISIONAL", f"internally consistent, semantics inferred ({why})"
    if d.legitimacy == "structural":
        return "PROVISIONAL", "source confirmed, but the denominator reads a null as 'no credit' — an inference that no source can confirm"
    if d.context_adjustment != "none":
        return "DERIVED_VERIFIED", f"derived from confirmed fields by {d.context_adjustment}; the derivation is checked synthetically"
    return "CONFIRMED", "recorded denominator; vendor definition of every needed field recorded by the analyst"


def _role(k: str, cls: str, redundant_with: str | None, screen_row: dict[str, Any]) -> str:
    if cls == "BLOCKED":
        return "blocked"
    if redundant_with:
        return "redundant"
    if k == "FO01.S2":
        return "primary_or_secondary" if screen_row.get("c_reliability") else "sensitivity_only"
    if k in SENSITIVITY_ONLY:
        return "sensitivity_only"
    return "secondary_only" if k in SECONDARY_ONLY else "primary_or_secondary"


def definitions_table(screen: pl.DataFrame, semantic: dict[str, dict[str, str]], lineage: pl.DataFrame, equivalences: pl.DataFrame,
                      dens: Sequence[ov.Denominator] = ov.DENOMINATORS) -> pl.DataFrame:
    assert_no_combine_columns(screen, "definition classes")
    confirmed = {f: c == "confirmed" for f, c in zip(lineage["field"], lineage["semantic_confidence"])}
    redundant = {r["definition_b"]: r["definition_a"] for r in equivalences.iter_rows(named=True) if r["identical"]}
    st = {r["denominator_id"]: r for r in screen.iter_rows(named=True)}
    rows = []
    for d in dens:
        r = st.get(d.denominator_id, {})
        sem = semantic.get(d.outcome_id, {}).get("status", "unresolved")
        cls, basis = classify(d, r, sem, confirmed)
        rows.append({"definition_id": d.denominator_id, "outcome_id": d.outcome_id, "label": d.label, "formula": d.formula, "unit": d.unit,
                     "legitimacy": d.legitimacy, "semantic_status": sem, "source_class": cls, "class_basis": basis,
                     "role_allowed": _role(d.denominator_id, cls, redundant.get(d.denominator_id), r),
                     "redundant_with": redundant.get(d.denominator_id, ""), "screen_status": r.get("status", "unavailable"),
                     "n_supported": int(r.get("n_supported") or 0), "n_linked_all_models": int(r.get("n_linked_all_models") or 0),
                     "rel_sb": r.get("rel_sb"), "rel_lo": r.get("rel_lo"), "rel_hi": r.get("rel_hi"), "exposure_rho": r.get("exposure_rho"),
                     "interpretation_label": INTERPRETATION.get(d.denominator_id, d.label)})
    return pl.DataFrame(rows, schema=DEFINITIONS.columns)


def propose(definitions: pl.DataFrame) -> dict[str, Any]:
    """Rule-based PROPOSAL (never a decision): the first definition allowed as primary in the 08A.2 priority order, then
    one admissible definition per other outcome."""
    ok = {r["definition_id"]: r for r in definitions.iter_rows(named=True)}
    prim = [k for k in ov.PRIORITY if ok.get(k, {}).get("role_allowed") == "primary_or_secondary"]
    if not prim:
        return {"decision": "STOP", "primary": None, "secondary": [], "basis": "no definition may be a primary target"}
    p, sec = prim[0], []
    for k in ov.PRIORITY:
        r = ok.get(k, {})
        if r.get("role_allowed") in ("primary_or_secondary", "secondary_only") and r["outcome_id"] != ok[p]["outcome_id"] \
                and r["outcome_id"] not in {ok[x]["outcome_id"] for x in sec}:
            sec.append(k)
    confirmed = all(ok[k]["source_class"] in ("CONFIRMED", "DERIVED_VERIFIED") for k in (p, *sec))
    return {"decision": "FREEZE", "primary": p, "secondary": sec[:2], "needs_acknowledgement": not confirmed,
            "basis": "first admissible primary in the pre-declared order; one definition per other outcome; "
                     + ("all sources confirmed" if confirmed else "at least one PROVISIONAL source: an exploratory freeze at most")}


# ============================================================================== registry
def registry_path(store: Any, version: str) -> Path:
    return store.data_dir / f"{REGISTRY_FILE}_{version}.json"


def version_hash(doc: dict[str, Any]) -> str:
    return ov._hash({"targets": [(t["target_id"], t["role"], t["definition_hash"], t["cohort_hash"], t["source_confidence"])
                                 for t in doc.get("targets", [])],
                     "decision": doc.get("decision"), "approved_status": doc.get("approved_status"), "hypotheses": doc.get("hypotheses"),
                     "config_hash": doc.get("config_hash"), "version": doc.get("version")})


def _approved_status(decision: str, classes: Sequence[str], acknowledged: bool) -> str:
    if decision == "STOP":
        return "stopped"
    confirmed = all(c in ("CONFIRMED", "DERIVED_VERIFIED") for c in classes)
    if decision == "FREEZE":
        return "frozen_confirmed" if confirmed else "frozen_exploratory"
    return "provisional_exploratory" if acknowledged else "provisional_review"


def record_decision(store: Any, decision: str | None, primary: str | None, secondary: Sequence[str], rationale: str,
                    acknowledge_provisional: bool, definitions: pl.DataFrame, screen: pl.DataFrame, athletes: pl.DataFrame,
                    semantic: dict[str, dict[str, str]], lineage: pl.DataFrame, cfg: ov.ValidationConfig, provenance: dict[str, Any],
                    hypotheses: dict[str, Any] | None = None, version: str = "v1") -> tuple[dict[str, Any] | None, str]:
    """Record the analyst's decision in ``football_outcome_registry_<version>.json``. The file is written ONCE: the same
    decision reproduces it; any other content raises ``FreezeConflictError`` (a change needs a NEW version, and the old
    file stays). ``None`` writes nothing. The source class of a target is copied from ``definitions`` — never upgraded."""
    path = registry_path(store, version)
    if decision is None:
        return (read_json(path, None) if path.is_file() else None), "PENDING — set DECISION to 'FREEZE', 'PROVISIONAL' or 'STOP' and rerun this cell"
    if decision not in DECISIONS:
        raise ValueError(f"DECISION must be one of {DECISIONS} or None, got {decision!r}")
    if not str(rationale).strip():
        raise ValueError("a decision needs a written RATIONALE")
    defs = {r["definition_id"]: r for r in definitions.iter_rows(named=True)}
    st = {r["denominator_id"]: r for r in screen.iter_rows(named=True)}
    secondary, targets = list(secondary or []), []
    if decision != "STOP":
        if primary is None or len(secondary) > 2 or len({primary, *secondary}) != 1 + len(secondary):
            raise ValueError(f"a {decision} needs one PRIMARY and at most two distinct SECONDARY definitions")
        for k, role in ((primary, "primary"), *[(k, "secondary") for k in secondary]):
            r = defs.get(k)
            if r is None:
                raise ValueError(f"unknown definition {k!r}")
            allowed = ("primary_or_secondary",) if role == "primary" else ("primary_or_secondary", "secondary_only")
            if r["role_allowed"] not in allowed:
                raise ValueError(f"{k!r} cannot be a {role} target: role allowed = {r['role_allowed']} ({r['class_basis']}"
                                 + (f"; redundant with {r['redundant_with']}" if r["redundant_with"] else "") + ")")
            if decision == "FREEZE" and r["source_class"] == "PROVISIONAL" and not acknowledge_provisional:
                raise ValueError(f"{k!r} has a PROVISIONAL source ({r['class_basis']}). Freezing it gives an EXPLORATORY freeze and needs "
                                 "ACKNOWLEDGE_PROVISIONAL = True with a RATIONALE; otherwise record 'PROVISIONAL' or 'STOP'")
            t = ov.target_definition(k, role, st[k], semantic, athletes, cfg)
            targets.append({**t, "source_confidence": r["source_class"], "class_basis": r["class_basis"],
                            "interpretation_label": r["interpretation_label"], "definition_version": DEFINITION_VERSION})
        if len({t["outcome_id"] for t in targets}) != len(targets):
            raise ValueError("one definition per outcome: two targets share an outcome")
    status = _approved_status(decision, [t["source_confidence"] for t in targets], bool(acknowledge_provisional))
    excluded = [{"definition_id": k, "role_allowed": r["role_allowed"], "source_class": r["source_class"], "reason": r["class_basis"],
                 "redundant_with": r["redundant_with"]} for k, r in defs.items() if k not in {t["target_id"] for t in targets}]
    doc = {"artifact": REGISTRY_FILE, "version": version, "registry_code_version": REGISTRY_VERSION_CODE, "decision": decision,
           "approved_status": status, "recorded_at": utc_now(), "targets": targets, "not_selected": excluded,
           "hypotheses": hypotheses or HYPOTHESES_DEFAULT, "rationale": rationale, "acknowledge_provisional": bool(acknowledge_provisional),
           "config": cfg.params(), "config_hash": ov._hash(cfg.params()), "denominators_hash": ov.denominators_hash(),
           "semantics_hash": osm.definition_hash(), "lineage_hash": ov._hash([asdict(f) for f in LINEAGE]),
           "lineage": lineage.select("field", "semantic_confidence", "source_evidence").to_dicts(),
           "selection": {"rule_based_proposal": propose(definitions), "combine_values_consulted": False, "combine_link_counts_consulted": True},
           "evidence_family": "NEW exploratory family (Milestone 08B); not a confirmation or a revision of the Milestone 06 result",
           "provenance": provenance, "run_mode": store.controls.run_mode, "seed": cfg.seed}
    doc["version_hash"] = version_hash(doc)
    if path.is_file():
        old = read_json(path, {})
        if old.get("version_hash") == doc["version_hash"]:
            return old, f"already recorded on {old.get('recorded_at')}; the current cell reproduces it"
        raise ov.FreezeConflictError(f"{path.name} exists with another content (recorded {old.get('recorded_at')}, decision "
                                     f"{old.get('decision')!r}, hash {old.get('version_hash')} ≠ {doc['version_hash']}). A registry is "
                                     "immutable: set REGISTRY_VERSION to a new version ('v2'); the old file stays.")
    if not store.controls.save_tables:
        return doc, "NOT WRITTEN (SAVE_TABLES is off)"
    store.data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(doc, path)
    log.info("[football_registry] %s written: %s %s", path.name, decision, [t["target_id"] for t in targets])
    return doc, f"recorded now ({status})"


def load_registry(store: Any, version: str = "v1") -> dict[str, Any] | None:
    """The registry of this run mode, verified (hash, run mode), or None when no decision was recorded."""
    path = registry_path(store, version)
    if not path.is_file():
        return None
    doc = read_json(path, {})
    problems = []
    if doc.get("version_hash") != version_hash(doc):
        problems.append("edited after it was recorded (hash mismatch)")
    if doc.get("run_mode") != store.controls.run_mode:
        problems.append(f"recorded in run mode {doc.get('run_mode')!r}")
    if doc.get("decision") != "STOP" and sum(t.get("role") == "primary" for t in doc.get("targets", [])) != 1:
        problems.append("not exactly one primary target")
    if problems:
        raise FootballTransferBlocked(f"{path.name} is not a valid registry: " + "; ".join(problems))
    return doc


def branch(doc: dict[str, Any] | None) -> dict[str, str]:
    """{name, label, reason, claim} of the registry. ``claim`` is the strongest wording a football-transfer result may carry."""
    if doc is None:
        return {"name": "blocked", "label": "BLOCKED — no decision recorded", "claim": "none",
                "reason": "notebook 08B.0 has no recorded decision: football transfer analyses do not run"}
    status = doc.get("approved_status")
    if status == "frozen_confirmed":
        return {"name": "frozen", "label": "FROZEN — confirmed source definitions", "reason": doc.get("rationale", ""),
                "claim": "held-out predictive evaluation of a validated outcome (single cohort; a new evidence family)"}
    if status in ("frozen_exploratory", "provisional_exploratory"):
        lock = "definitions locked" if status == "frozen_exploratory" else "definitions NOT locked (may change in a new version)"
        return {"name": "exploratory", "label": f"EXPLORATORY — source semantics inferred; {lock}", "reason": doc.get("rationale", ""),
                "claim": "exploratory association only — never a confirmed predictive claim"}
    why = {"stopped": "STOP recorded: no football outcome is accepted", "provisional_review": "PROVISIONAL recorded without the "
           "acknowledgement: definitions kept for review only"}.get(status, f"approved_status {status!r}")
    return {"name": "blocked", "label": "BLOCKED — " + why, "reason": doc.get("rationale", ""), "claim": "none"}


def require_branch(store: Any, version: str = "v1", allow: Sequence[str] = ("frozen", "exploratory")) -> tuple[dict[str, Any], dict[str, str]]:
    """The gate of notebooks 08B.1–08B.3: the registry exists, is intact and its branch is allowed."""
    doc = load_registry(store, version)
    b = branch(doc)
    if b["name"] not in allow:
        raise FootballTransferBlocked(f"football transfer is blocked — {b['label']}. {b['reason']} Record a decision in notebook "
                                      "08b0_outcome_resolution_and_freeze.ipynb (section 7). Notebooks 08B.4–08B.6 do not need it.")
    return doc, b


def targets_of(doc: dict[str, Any] | None, role: str | None = None) -> list[dict[str, Any]]:
    return [t for t in (doc or {}).get("targets", []) if role is None or t["role"] == role]


def gate_a(doc: dict[str, Any] | None, definitions: pl.DataFrame, lineage: pl.DataFrame) -> pl.DataFrame:
    """Gate A — outcome semantics: can the target be interpreted legitimately? ``passed`` on the last row decides."""
    b = branch(doc)
    rows = [{"criterion": "lineage_recorded_for_every_field", "passed": lineage.height == len(LINEAGE),
             "detail": f"{lineage.height} fields; {int((lineage['semantic_confidence'] == 'confirmed').sum())} with recorded source evidence"},
            {"criterion": "unresolved_semantics_stay_visible", "passed": True,
             "detail": "definitions by class: " + ", ".join(f"{c} {n}" for c, n in definitions.group_by("source_class").len().sort("source_class").iter_rows())},
            {"criterion": "decision_recorded", "passed": doc is not None, "detail": "none" if doc is None else f"{doc['decision']} → {doc['approved_status']}"}]
    tg = targets_of(doc)
    rows.append({"criterion": "no_source_class_upgraded", "passed": all(
        t["source_confidence"] == dict(zip(definitions["definition_id"], definitions["source_class"])).get(t["target_id"]) for t in tg),
        "detail": ", ".join(f"{t['target_id']} {t['source_confidence']}" for t in tg) or "no target"})
    rows.append({"criterion": "football_transfer_allowed", "passed": b["name"] != "blocked", "detail": f"{b['label']} — strongest claim: {b['claim']}"})
    return pl.DataFrame(rows, schema=GATE_A.columns)


def conclusions_markdown(definitions: pl.DataFrame, equivalences: pl.DataFrame, doc: dict[str, Any] | None) -> str:
    b = branch(doc)
    f = lambda x: "—" if x is None else f"{x:.2f}"
    lines = ["### 08B.0 — outcome resolution and registry", "", f"Branch: **{b['label']}**. Strongest admissible claim: {b['claim']}.", "",
             "| Definition | Source class | Role allowed | Linked | Reliability [95 %] | May be called |", "|---|---|---|---|---|---|"]
    lines += [f"| `{r['definition_id']}` | {r['source_class']} | {r['role_allowed']}" + (f" (= `{r['redundant_with']}`)" if r["redundant_with"] else "")
              + f" | {r['n_linked_all_models']} | {f(r['rel_sb'])} [{f(r['rel_lo'])}, {f(r['rel_hi'])}] | {r['interpretation_label']} |"
              for r in definitions.iter_rows(named=True)]
    lines += [""] + [f"- {r['reading']} (common athletes {r['n_common']}, same observation counts {r['n_same_observations']})."
                     for r in equivalences.iter_rows(named=True)]
    if doc is not None:
        lines += ["", f"Decision **{doc['decision']}** → `{doc['approved_status']}` (registry `{doc['version']}`, hash `{doc['version_hash']}`): "
                      + (", ".join(f"`{t['target_id']}` ({t['role']}, {t['source_confidence']})" for t in doc["targets"]) or "no target") + ".",
                  f"Rationale: {doc['rationale']}"]
    lines += ["", "A decision never raises a source class. A PROVISIONAL class becomes CONFIRMED only when the analyst records a named, "
                  "authoritative source and the quoted definition, in a NEW registry version."]
    return "\n".join(lines)


__all__ = ["BRANCHES", "CLASSES", "DECISIONS", "DEFINITIONS", "EQUIVALENCE", "FieldLineage", "FootballTransferBlocked", "GATE_A",
           "HYPOTHESES_DEFAULT", "INTERPRETATION", "LINEAGE", "LINEAGE_T", "REGISTRY_FILE", "REPO_EVIDENCE", "branch", "classify",
           "conclusions_markdown", "definitions_table", "equivalence", "equivalence_table", "gate_a", "lineage_table", "load_registry",
           "propose", "record_decision", "registry_path", "repository_evidence", "require_branch", "targets_of", "version_hash"]
