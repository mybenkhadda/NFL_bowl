"""Combine → NFL transfer study (Milestone 06): frozen targets, analytic cohort, folds and leakage audit.
NumPy / Polars only — no PyTorch, no model fitting, no association between a Combine value and an NFL value.

Order of the study (notebook 06A runs this module, 06B and 06C refuse to start without its manifest)
    1. ``audit_freeze``        the NFL targets must be FROZEN by notebook 05B (``nfl_movement_targets_v1.json``): status,
                               formulas, units, context, exposure, hashes, provenance. A pending, absent, stopped or
                               altered registry raises ``TargetFreezeError``. Nothing here can create or default a target.
    2. ``identity_audit``      one canonical athlete id, unique Combine ↔ NFL linkage; conflicts are recorded and the
                               athletes excluded — never resolved silently.
    3. ``cohort_manifest``     one row per Combine athlete with every eligibility flag; the PRIMARY cohort supports every
                               model M0–M7 (identical athletes and labels for all paired comparisons); the WIDE cohort
                               (official results + target, tracking not required) is a labelled secondary analysis.
    4. ``TargetSet``           athlete-level labels from the persisted play table, under a documented exposure rule. A
                               context-adjusted target removes context-cell means estimated on the TRAINING athletes of a
                               fold only; 06A reads exposure counts from it and no label value.
    5. ``assign_folds``        athlete-grouped outer folds, stratified by position × draft cohort. The split registry of
                               notebook 00 is reused when its folds are viable in the linked cohort (the saved encoders are
                               then usable); otherwise new folds are dealt and every saved encoder becomes descriptive-only.
    6. ``compatibility_table`` every saved FPCA / TCN fit against the transfer folds: an encoder is reusable for a fold only
                               if none of its fit, development or normalisation athletes is held out in that fold.

Units: the athlete is the unit everywhere. Player-plays, windows and Combine attempts are never a sample size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import file_fingerprint, get_logger, read_json, stable_hash
from ..combine_nfl_dl.models import make_folds
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from . import nfl_outcomes as no
from .config import NotebookControls
from .fpca_eval import FORTY, HEIGHT, TEN, WEIGHT

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
TRANSFER_VERSION = "1"
TARGETS_FILE = "nfl_movement_targets_v1"
AUDIT_MANIFEST = "transfer_cohort_audit_manifest"
INPUT_CONTRACT = "transfer_input_contract"
ALL = "ALL"
SPEED2_FEATURES = ("F2.forty.p95_speed", "F2.forty.peak_speed_interior")     # notebook 04B: the per-attempt speed control
ENGINEERED_SLOTS = ("C3_acceleration", "C4_sprint_parameter")                # shortlist slots that feed model M2
DEFINITION_FIELDS = ("outcome_id", "formula", "units", "play_measure", "athlete_statistic", "window", "starting_speed_range",
                     "context_definition", "context_adjustment", "exposure_requirement", "inclusion_exclusion", "smoothing",
                     "grouping_grain", "direction_note", "interpretation_limit")
REQUIRED_TARGET_FIELDS = (("outcome_id", "target ID"), ("formula", "exact formula"), ("units", "units"),
                          ("context_definition", "context definition"), ("exposure_requirement", "exposure requirement"),
                          ("inclusion_exclusion", "inclusion / exclusion rules"), ("smoothing", "smoothing rule"),
                          ("definition_hash", "definition hash"), ("reliability", "reliability evidence"))
STUDY_ROLES = {"O1": "secondary_speed", "O3": "secondary_speed_conditioned", "O5": "exploratory_envelope"}
STAGES = (("combine_roster", None, "athlete of the Combine registry (results row or tracked sequence)"),
          ("tracked_forty", "st_tracked", "≥ 1 eligible tracked forty-yard attempt"),
          ("official_core", "st_official", "official forty, 10-yard split and weight recorded"),
          ("engineered_features", "st_engineered", "frozen engineered features and the speed control measured on ≥ 1 attempt"),
          ("fpca_input", "st_fpca", "≥ 1 attempt with a speed-by-distance curve (FPCA input)"),
          ("tcn_input", "st_tcn", "≥ 1 attempt with a speed-by-distance curve (TCN-B input; same input as FPCA)"),
          ("nfl_tracking", "st_nfl", "≥ 1 eligible regular-season NFL window"),
          ("target_exposure", "st_target", "exposure requirement of the frozen PRIMARY target met"),
          ("primary_cohort", "in_primary_cohort", "all of the above and a consistent Combine ↔ NFL identity"))


class TargetFreezeError(RuntimeError):
    """The NFL target registry is absent, pending, stopped, incomplete or altered: no transfer analysis may run."""


class TransferGateError(RuntimeError):
    """A decision gate of Milestone 06 is not passed."""


class FoldError(ValueError):
    """No athlete-grouped fold design is viable for the linked cohort."""


# ============================================================================== configuration
@dataclass(frozen=True)
class TransferConfig:
    """Provenance: established = reused project value · proposed = fixed here, before any NFL label is joined to a
    Combine value · heuristic = reference line, never a sole decision rule · execution = cost knob."""

    version: str = TRANSFER_VERSION
    seed: int = 20261011
    # pre-NFL predictors ---------------------------------------------------------------------------------------
    core_official: tuple[str, ...] = (FORTY, TEN, WEIGHT)       # established: MatchConfig.required (mandatory core)
    height: str = HEIGHT                                        # established: optional in the registry → imputed + flag
    optional_tests: tuple[str, ...] = ("cmb_vertical", "cmb_broad_jump", "cmb_short_shuttle", "cmb_three_cone", "cmb_bench_reps")
    optional_body: tuple[str, ...] = ("cmb_hand_size", "cmb_arm_length", "cmb_wing_span")
    optional_min_coverage: float = 0.50                         # proposed: an optional test enters M0 / M1 above this
    # folds ----------------------------------------------------------------------------------------------------
    outer_folds_grid: tuple[int, ...] = (5, 4, 3)               # proposed: the largest viable K is used
    min_heldout_fold: int = 30                                  # proposed: athletes held out in every fold
    min_position_train: int = 10                                # proposed: athletes of each position in every training set
    inner_folds: int = 5                                        # proposed: athlete-grouped folds of the ridge penalty
    # regularised models ---------------------------------------------------------------------------------------
    alpha_grid: tuple[float, float, int] = (-2.0, 4.0, 13)      # proposed: log10 range and size, fixed
    enet_l1_ratio: float = 0.5                                  # proposed (exploratory learner)
    # representations ------------------------------------------------------------------------------------------
    aggregation: str = "mean"                                   # established: athlete = mean of valid attempts (04B)
    aggregations: tuple[str, ...] = ("mean", "shrunk", "first")
    compact_dims: tuple[int, ...] = (2, 4)                      # brief: M7, exploratory; no dimension is selected
    replicate_seeds: tuple[int, ...] = (20261012, 20261013)     # established: TCN-B seed replicates of notebook 04B
    # uncertainty ----------------------------------------------------------------------------------------------
    n_boot: int = 2000                                          # execution
    n_perm: int = 200                                           # execution
    min_athletes: int = 40                                      # proposed: fewer → no estimate
    min_athletes_group: int = 20                                # proposed: per position / cohort subgroup
    # sensitivity grids (declared, never selected on a result) ---------------------------------------------------
    exposure_min_windows: tuple[int, ...] = (20, 50, 100, 200)  # established: OutcomeConfig.min_windows_grid
    first_n_windows: tuple[int, ...] = (50, 100, 200)           # established: OutcomeConfig.first_n_grid
    min_games: tuple[int, ...] = (4, 8)                         # established: OutcomeConfig.first_games_grid
    horizon_seasons: tuple[int, ...] = (1,)                     # proposed: common post-draft horizon (rookie season)
    play_classes: tuple[str, ...] = ("pass", "no_pass")         # established: nfl_outcomes play_class
    # readiness (heuristic) ------------------------------------------------------------------------------------
    min_relative_gain: float = 0.02                             # heuristic: smaller relative RMSE gains are not practical
    perm_alpha: float = 0.05
    case_studies: int = 6

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def for_mode(self, controls: NotebookControls) -> "TransferConfig":
        """Smoke mode keeps every model and rule; it shortens resampling and lowers the size requirements so about
        sixty athletes can exercise every code path. Smoke numbers are wiring checks, never results."""
        if not controls.smoke:
            return replace(self, seed=controls.seed)
        return replace(self, seed=controls.seed, outer_folds_grid=(3,), min_heldout_fold=4, min_position_train=2, inner_folds=3,
                       n_boot=200, n_perm=20, min_athletes=8, min_athletes_group=4, exposure_min_windows=(5, 10, 20),
                       first_n_windows=(5, 10), min_games=(1, 2), replicate_seeds=self.replicate_seeds[:1], case_studies=3)

    @property
    def alphas(self) -> np.ndarray:
        return np.logspace(self.alpha_grid[0], self.alpha_grid[1], int(self.alpha_grid[2]))


# ============================================================================== contracts
FREEZE_AUDIT = TableContract(
    "transfer_target_freeze_audit", "Checks of the NFL target registry before any transfer analysis.",
    {"check": S, "passed": B, "detail": S}, key=("check",), area="check")
IDENTITY = TableContract(
    "transfer_identity_audit", "Checks of the Combine ↔ NFL athlete linkage.",
    {"check": S, "severity": S, "passed": B, "n_violations": I, "detail": S}, key=("check",), area="check",
    docs={"severity": "critical = the athletes concerned are excluded and listed · info = reported only"})
ELIGIBILITY = TableContract(
    "transfer_target_eligibility", "Exposure of every athlete on every frozen NFL target (no label value).",
    {"nfl_id": I, "target_id": S, "rank": S, "study_role": S, "n_windows": I, "n_games": I, "min_windows": I, "eligible": B},
    key=("nfl_id", "target_id"), kind="dataset", area="athlete × frozen target",
    docs={"n_windows": "eligible regular-season windows on which the target is defined (composites: smallest component)",
          "eligible": "n_windows ≥ the exposure requirement recorded in the frozen registry"})
COHORT = TableContract(
    "transfer_cohort_manifest", "One row per Combine athlete: linkage, eligibility flags and cohort membership.",
    {"nfl_id": I, "display_name": S, "position_group": S, "draft_year": I, "n_attempts": I, "n_curves": I, "n_eng_attempts": I,
     "n_speed_attempts": I, "n_signal_ok": I, "combine_quality": S, "has_official_core": B, "nfl_position_group": S,
     "nfl_cohort": I, "n_nfl_windows": I, "n_nfl_games": I, "n_nfl_seasons": I, "first_season": I, "last_season": I,
     "exposure_band": S, "target_windows": I, "identity_ok": B, "st_tracked": B, "st_official": B, "st_engineered": B,
     "st_fpca": B, "st_tcn": B, "st_nfl": B, "st_target": B, "in_primary_cohort": B, "in_wide_cohort": B,
     "exclusion_reason": S, "manifest_version": S}, key=("nfl_id",), kind="dataset", area="athlete",
    docs={"n_attempts": "eligible tracked forty attempts (sequence manifest of notebook 00)",
          "n_curves": "attempts with a speed-by-distance curve holding data",
          "target_windows": "eligible windows of the frozen PRIMARY target",
          "in_primary_cohort": "supports every model M0–M7: the population of all paired comparisons",
          "in_wide_cohort": "official core + primary-target exposure, tracking not required: secondary analysis of M0 / M1",
          "exclusion_reason": "first rule of the funnel the athlete fails; empty for the primary cohort"})
FUNNEL = TableContract(
    "transfer_cohort_funnel", "Athletes remaining after each rule of the Combine → NFL funnel, and athletes meeting each rule alone.",
    {"stage": S, "stage_order": I, "group_type": S, "group": S, "n_meeting_rule": I, "n_cumulative": I, "definition": S},
    key=("stage", "group_type", "group"), area="funnel stage × group",
    docs={"n_cumulative": "athletes meeting this rule AND every earlier one (unique athletes, never plays or attempts)"})
COUNTS = TableContract(
    "transfer_position_cohort_counts", "Athletes by position group × draft cohort at the main funnel stages.",
    {"group_type": S, "group": S, "position_group": S, "n_roster": I, "n_tracked": I, "n_nfl": I, "n_target": I,
     "n_primary": I, "n_wide": I}, key=("group_type", "group", "position_group"), area="group × position group",
    docs={"group_type": "draft_year · first_season · exposure_band · combine_quality"})
REASONS = TableContract(
    "transfer_eligibility_reasons", "Why Combine athletes are not in the primary cohort (first failing rule).",
    {"exclusion_reason": S, "group_type": S, "group": S, "n_athletes": I}, key=("exclusion_reason", "group_type", "group"),
    area="reason × group")
MISSING = TableContract(
    "transfer_missingness_audit", "Missing pre-NFL measurements, their kind, and how each variable enters the models.",
    {"variable": S, "kind": S, "population": S, "group_type": S, "group": S, "n_athletes": I, "n_missing": I,
     "share_missing": F, "missing_class": S, "in_model": S, "rule": S}, key=("variable", "population", "group_type", "group"),
    area="variable × population × group",
    docs={"population": "roster = Combine registry · target_exposed = athletes meeting the primary-target exposure rule",
          "missing_class": "field_unavailable (column absent, or empty for the whole group) · no_value_recorded (did not "
                           "participate or not reported: the source does not distinguish them) · tracking_insufficient",
          "in_model": "mandatory · imputed_with_indicator · excluded_low_coverage · not_a_predictor"})
FOLDS = TableContract(
    "transfer_outer_folds", "Athlete-grouped outer and inner folds of the transfer study.",
    {"nfl_id": I, "position_group": S, "draft_year": I, "stratum": S, "outer_fold": I, "inner_fold": I, "in_primary_cohort": B,
     "target_eligible": B, "fold_source": S, "n_folds": I, "manifest_version": S}, key=("nfl_id",), area="athlete",
    docs={"outer_fold": "held-out fold: all Combine attempts and all NFL seasons of the athlete follow it",
          "inner_fold": "fold label used INSIDE an outer training set to choose the ridge penalty",
          "fold_source": "registry = outer folds of the notebook-00 split registry (repeat 0) · new_<K> = dealt here"})
FOLD_PLAN = TableContract(
    "transfer_fold_candidates", "Fold designs considered and why one was retained (no NFL value is involved).",
    {"candidate": S, "n_folds": I, "min_heldout": I, "max_heldout": I, "min_position_train": I, "smallest_cell": S, "viable": B,
     "selected": B, "reason": S}, key=("candidate",), area="fold design")
COMPAT = TableContract(
    "representation_checkpoint_compatibility", "Every saved representation fit against the transfer folds.",
    {"artifact": S, "family": S, "method": S, "repeat": I, "fold": I, "n_fit_athletes": I, "n_heldout_transfer": I,
     "n_overlap": I, "status": S, "use": S, "note": S}, key=("artifact", "method", "repeat", "fold"), area="saved fit",
    docs={"n_overlap": "athletes used to fit, early-stop or normalise the saved model that are HELD OUT in the transfer fold",
          "status": "compatible · descriptive_only · combine_only_selection (a choice made on Combine data, no NFL value)"})
GATE = TableContract("transfer_gate", "Decision gate of a Milestone 06 notebook.",
                     {"criterion": S, "passed": B, "detail": S}, key=("criterion",), area="criterion")


# ============================================================================== frozen NFL targets
def _tuplify(x: Any) -> Any:
    return tuple(_tuplify(v) for v in x) if isinstance(x, (list, tuple)) else x


def outcome_config_from_params(params: dict[str, Any]) -> no.OutcomeConfig:
    """``OutcomeConfig`` rebuilt from the ``config`` block of the frozen registry (JSON lists back to tuples)."""
    known = {f.name for f in fields(no.OutcomeConfig)}
    unknown = sorted(set(params) - known)
    if unknown:
        raise TargetFreezeError(f"the frozen registry holds configuration keys this code does not know: {unknown}")
    return no.OutcomeConfig(**{k: _tuplify(v) for k, v in params.items()})


def targets_path(nfl_store: Any) -> Path:
    return Path(nfl_store.data_dir) / f"{TARGETS_FILE}.json"


def audit_freeze(nfl_store: Any, run_mode: str, prior_hash: str | None = None) -> tuple[dict[str, Any] | None, pl.DataFrame]:
    """(registry or None, table of checks). Reads ``nfl_movement_targets_v1.json`` written by notebook 05B and verifies
    it; raises nothing — ``assert_frozen`` does. ``prior_hash``: target hash of an earlier transfer audit, if any."""
    path = targets_path(nfl_store)
    doc = read_json(path, None) if path.is_file() else None
    rows: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"check": check, "passed": bool(ok), "detail": detail})

    add("registry_present", doc is not None,
        f"{path.name} found" if doc is not None else f"{path} does not exist: notebook 05B section 10 has not recorded a "
        "decision (status PENDING REVIEW). Set DECISION / PRIMARY / SECONDARY / RATIONALE there and rerun that cell")
    if doc is None:
        return None, pl.DataFrame(rows, schema=FREEZE_AUDIT.columns)
    targets = list(doc.get("targets") or [])
    add("status_is_freeze", doc.get("decision") == "FREEZE", f"decision = {doc.get('decision')!r} (frozen at {doc.get('frozen_at')})")
    ranks = [t.get("rank") for t in targets]
    add("targets_listed", bool(targets) and ranks.count("primary") == 1 and len({t.get("outcome_id") for t in targets}) == len(targets),
        f"{len(targets)} target(s), ranks {ranks}")
    lacking = [f"{t.get('outcome_id')}: {label}" for t in targets for key, label in REQUIRED_TARGET_FIELDS if not t.get(key)]
    lacking += [f"{t.get('outcome_id')}: minimum windows" for t in targets
                if not isinstance((t.get("exposure_requirement") or {}).get("min_windows"), int)]
    add("target_fields_complete", bool(targets) and not lacking, "every target carries " + ", ".join(label for _, label in REQUIRED_TARGET_FIELDS)
        if not lacking else "missing — " + "; ".join(lacking))
    bad = [t.get("outcome_id") for t in targets
           if stable_hash({k: t.get(k) for k in DEFINITION_FIELDS}) != t.get("definition_hash")]
    add("definition_hashes_reproduce", bool(targets) and not bad,
        "each definition hash equals the hash of the stored definition" if not bad else f"definition altered after the freeze: {bad}")
    version = stable_hash({"decision": doc.get("decision"), "targets": [t.get("definition_hash") for t in targets],
                           "config": doc.get("config")})
    add("version_hash_reproduces", version == doc.get("version_hash"), f"stored {doc.get('version_hash')}, recomputed {version}")
    sel = doc.get("selection") or {}
    justified = bool(sel.get("criteria")) and (bool(str(sel.get("rationale") or "").strip()) or bool(sel.get("follows_proposal")))
    add("selection_justified", justified, "written rationale" if str(sel.get("rationale") or "").strip()
        else "follows the rule-based proposal of notebook 05B" if sel.get("follows_proposal") else "no rationale and the "
        "decision departs from the rule-based proposal")
    add("reliability_evidence", bool(targets) and all((t.get("reliability") or {}).get("estimate") is not None for t in targets),
        "; ".join(f"{t.get('outcome_id')}: {(t.get('reliability') or {}).get('estimate')}" for t in targets))
    add("provenance_recorded", bool(doc.get("provenance")), f"keys {sorted((doc.get('provenance') or {}).keys())}")
    add("frozen_without_combine_data", sel.get("combine_data_consulted") is False, f"combine_data_consulted = {sel.get('combine_data_consulted')!r}")
    add("same_run_mode", doc.get("run_mode") == run_mode, f"registry run mode {doc.get('run_mode')!r}, notebook {run_mode!r}")
    add("no_refreeze_after_transfer", prior_hash in (None, doc.get("version_hash")),
        "no earlier transfer audit" if prior_hash is None else f"earlier transfer audit used targets {prior_hash}; registry is "
        f"{doc.get('version_hash')}" + ("" if prior_hash == doc.get("version_hash") else " — the targets were re-frozen AFTER a "
                                        "transfer run: delete the transfer outputs and record why before continuing"))
    try:
        known = {s.outcome_id: s for s in no.outcome_specs(outcome_config_from_params(doc.get("config") or {}))}
        missing = [t.get("outcome_id") for t in targets if t.get("outcome_id") not in known]
        add("targets_known_to_engine", not missing, "every target is an outcome of the stored definition" if not missing
            else f"not produced by the stored outcome definition: {missing}")
    except (TargetFreezeError, TypeError) as e:
        add("targets_known_to_engine", False, f"{type(e).__name__}: {e}")
    return doc, pl.DataFrame(rows, schema=FREEZE_AUDIT.columns)


def assert_frozen(table: pl.DataFrame) -> None:
    failed = table.filter(~pl.col("passed"))
    if failed.height:
        raise TargetFreezeError(
            "BLOCKER — the NFL targets are not frozen; no Combine ↔ NFL association may be computed.\n"
            + "\n".join(f"  ✗ {r['check']}: {r['detail']}" for r in failed.iter_rows(named=True))
            + "\nNothing was defaulted. Freeze the targets in 05b_nfl_outcome_reliability_and_freeze.ipynb (section 10), then rerun 06A.")


def proposal_for_review(nfl_store: Any) -> dict[str, Any]:
    """Rule-based proposal of notebook 05B, for DISPLAY when the registry is pending. It is not a freeze and this
    module never turns it into one."""
    m = read_json(Path(nfl_store.data_dir) / "nfl_reliability_manifest.json", {}) or {}
    return {"proposal": m.get("proposal"), "recorded_decision": m.get("decision"), "gate_05": m.get("gate_05"),
            "freeze_message": m.get("freeze_message"), "executed": m.get("status"), "written_at": m.get("written_at")}


@dataclass(frozen=True)
class FrozenTarget:
    target_id: str
    rank: str                       # primary · secondary (as frozen)
    study_role: str                 # primary · secondary_speed · secondary_speed_conditioned · exploratory_envelope · secondary
    family: str
    label: str
    unit: str
    stat: str
    components: tuple[str, ...]
    adjustment: str                 # none · cell_residual (as frozen)
    min_windows: int
    definition_hash: str


def frozen_targets(doc: dict[str, Any]) -> tuple[list[FrozenTarget], no.OutcomeConfig]:
    """Targets of a registry that passed ``assert_frozen``, with the outcome configuration they were measured under."""
    ocfg = outcome_config_from_params(doc["config"])
    spec = {s.outcome_id: s for s in no.outcome_specs(ocfg)}
    out = []
    for t in doc["targets"]:
        s = spec[t["outcome_id"]]
        role = "primary" if t["rank"] == "primary" else STUDY_ROLES.get(s.family, "secondary")
        out.append(FrozenTarget(s.outcome_id, t["rank"], role, s.family, s.label, s.unit, s.stat, tuple(s.components),
                                str(t.get("context_adjustment") or "none"), int(t["exposure_requirement"]["min_windows"]),
                                t["definition_hash"]))
    return out, ocfg


def primary_target(targets: Sequence[FrozenTarget]) -> FrozenTarget:
    return next(t for t in targets if t.rank == "primary")


# ============================================================================== athlete-level labels
@dataclass
class _Windows:
    a: np.ndarray                   # (W,) index into TargetSet.ids
    y: np.ndarray                   # (W,) play measure
    cell: np.ndarray                # (W,) context-cell code
    pos: np.ndarray                 # (W,) position-group code
    game: np.ndarray                # (W,) game id
    w_order: np.ndarray             # (W,) 1 = the athlete's first window of this outcome in time
    season_rel: np.ndarray          # (W,) season − draft year (0 = rookie season); NaN when the cohort is unknown
    play_class: np.ndarray          # (W,) str
    stat: str


class TargetSet:
    """Athlete-level values of the frozen targets from the persisted play table (``nfl_play_movement``).

    One value per athlete under a named exposure rule (``variant``):
        all            every eligible regular-season window (the frozen grain)
        first_n:<N>    the athlete's first N windows in time
        horizon:<H>    windows of the first H seasons after the draft year (a common follow-up horizon)
        class:<name>   windows of one play class
    ``adjustment`` 'cell_residual' removes context-cell means (position × alignment × play class) and adds the
    position mean back, exactly as ``OutcomeEngine`` does — but the means come from the windows of ``train_ids`` only,
    so a held-out athlete's label never informs the adjustment applied to the training labels (and vice versa).
    With ``train_ids`` None every athlete is used: descriptive, and equal to ``nfl_athlete_outcomes.value_adj``."""

    def __init__(self, plays: pl.DataFrame, ocfg: no.OutcomeConfig, targets: Sequence[FrozenTarget]):
        self.ocfg = ocfg
        self.eng = no.OutcomeEngine(plays, ocfg)
        self.targets = {t.target_id: t for t in targets}
        self.ids = self.eng.athletes["nfl_id"].to_numpy().astype(np.int64)
        self.position = self.eng.athletes["position_group"].fill_null("?").to_numpy().astype(str)
        self._cohort = self.eng.athletes["cohort"].cast(pl.Float64).fill_null(NAN).to_numpy()
        self._w: dict[str, _Windows] = {}

    def _windows(self, outcome_id: str) -> _Windows:
        if outcome_id not in self._w:
            spec = self.eng.by_id[outcome_id]
            d = self.eng.frame(spec).sort("nfl_id", "w_order")
            a = np.searchsorted(self.ids, d["nfl_id"].to_numpy().astype(np.int64))
            self._w[outcome_id] = _Windows(
                a, d["y"].to_numpy().astype(float), np.unique(d["cell"].to_numpy().astype(str), return_inverse=True)[1],
                np.unique(d["position_group"].fill_null("?").to_numpy().astype(str), return_inverse=True)[1],
                d["game_id"].to_numpy().astype(np.int64), d["w_order"].to_numpy().astype(np.int64),
                d["season"].to_numpy().astype(float) - self._cohort[a], d["play_class"].to_numpy().astype(str), spec.stat)
        return self._w[outcome_id]

    @staticmethod
    def _mask(w: _Windows, variant: str) -> np.ndarray:
        if variant == "all":
            return np.ones(w.y.size, dtype=bool)
        kind, _, arg = variant.partition(":")
        if kind == "first_n":
            return w.w_order <= int(arg)
        if kind == "horizon":
            with np.errstate(invalid="ignore"):
                return (w.season_rel >= 0) & (w.season_rel < int(arg))
        if kind == "class":
            return w.play_class == arg
        raise ValueError(f"unknown exposure variant {variant!r} (all · first_n:<N> · horizon:<H> · class:<name>)")

    def _values(self, w: _Windows, train_ids: Sequence[int] | None, variant: str, adjustment: str
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = self.ids.size
        keep = self._mask(w, variant)
        a, y, game = w.a[keep], w.y[keep].copy(), w.game[keep]
        if adjustment == "cell_residual" and y.size:
            tr = np.ones(A, dtype=bool) if train_ids is None else np.isin(self.ids, np.asarray(list(train_ids), dtype=np.int64))
            wt = tr[a]
            cell, pos = w.cell[keep], w.pos[keep]
            C, P = int(w.cell.max()) + 1, int(w.pos.max()) + 1
            nc, sc = np.bincount(cell[wt], minlength=C), np.bincount(cell[wt], weights=y[wt], minlength=C)
            npos, sp = np.bincount(pos[wt], minlength=P), np.bincount(pos[wt], weights=y[wt], minlength=P)
            grand = float(y[wt].mean()) if wt.any() else NAN
            mp = np.where(npos > 0, sp / np.maximum(npos, 1), grand)
            big = nc >= self.ocfg.min_cell_windows                   # smaller training cells are left unadjusted
            y = np.where(big[cell], y - sc[cell] / np.maximum(nc[cell], 1) + mp[pos], y)
        elif adjustment not in ("none", "cell_residual"):
            raise ValueError(f"unknown context adjustment {adjustment!r}")
        n = np.bincount(a, minlength=A)
        value = np.full(A, NAN)
        if w.stat == "mean":
            value = np.where(n > 0, np.bincount(a, weights=y, minlength=A) / np.maximum(n, 1), NAN)
        else:
            q = {"median": 0.5, "p90": 0.90, "p95": 0.95}[w.stat]
            order = np.lexsort((y, a))
            ys, ends = y[order], np.cumsum(n)
            for i in np.flatnonzero(n > 0):
                value[i] = float(np.quantile(ys[ends[i] - n[i]:ends[i]], q))
        pair = np.unique(np.column_stack([a, game]), axis=0) if a.size else np.empty((0, 2), dtype=np.int64)
        return value, n.astype(np.int64), np.bincount(pair[:, 0], minlength=A).astype(np.int64)

    def athlete_values(self, target_id: str, train_ids: Sequence[int] | None = None, variant: str = "all",
                       adjustment: str | None = None) -> dict[str, np.ndarray]:
        """{ids, value, n_windows, n_games} over ``self.ids``; value is NaN for an athlete without a window (composite:
        without every component)."""
        t = self.targets[target_id]
        adj = t.adjustment if adjustment is None else adjustment
        parts = [self._values(self._windows(o), train_ids, variant, adj) for o in (t.components or (t.target_id,))]
        value = np.mean(np.vstack([p[0] for p in parts]), axis=0)        # NaN when any component is missing
        return {"ids": self.ids, "value": value, "n_windows": np.min(np.vstack([p[1] for p in parts]), axis=0),
                "n_games": np.min(np.vstack([p[2] for p in parts]), axis=0)}

    def eligibility(self, variant: str = "all") -> pl.DataFrame:
        """Exposure of every athlete on every frozen target — counts only, no label value leaves this method."""
        parts = []
        for t in self.targets.values():
            v = self.athlete_values(t.target_id, variant=variant, adjustment="none")
            parts.append(pl.DataFrame({"nfl_id": self.ids, "n_windows": v["n_windows"], "n_games": v["n_games"],
                                       "eligible": v["n_windows"] >= t.min_windows}).with_columns(
                pl.lit(t.target_id).alias("target_id"), pl.lit(t.rank).alias("rank"), pl.lit(t.study_role).alias("study_role"),
                pl.lit(t.min_windows, dtype=pl.Int64).alias("min_windows")).select([pl.col(k).cast(x) for k, x in ELIGIBILITY.columns.items()]))
        return pl.concat(parts).sort("nfl_id", "target_id")

    def exposure(self) -> pl.DataFrame:
        """One row per NFL-tracked athlete: eligible windows, games and seasons (any outcome)."""
        return (self.eng.plays.group_by("nfl_id").agg(
            pl.col("position_group").first().alias("nfl_position_group"), pl.col("cohort").first().cast(pl.Int64).alias("nfl_cohort"),
            pl.len().cast(pl.Int64).alias("n_nfl_windows"), pl.col("game_id").n_unique().cast(pl.Int64).alias("n_nfl_games"),
            pl.col("season").n_unique().cast(pl.Int64).alias("n_nfl_seasons"), pl.col("season").min().cast(pl.Int64).alias("first_season"),
            pl.col("season").max().cast(pl.Int64).alias("last_season")).sort("nfl_id"))


# ============================================================================== identity
def identity_audit(athletes: pl.DataFrame, mart: pl.DataFrame, exposure: pl.DataFrame, manifest: pl.DataFrame,
                   positions: Sequence[str]) -> tuple[pl.DataFrame, dict[int, str]]:
    """(table of checks, {nfl_id: reason} of athletes whose identity is not consistent). Nothing is repaired: an athlete
    with a conflict is excluded from every cohort and listed."""
    rows: list[dict[str, Any]] = []
    bad: dict[int, str] = {}

    def add(check: str, severity: str, ids: Sequence[int], detail: str) -> None:
        ids = sorted({int(i) for i in ids})
        rows.append({"check": check, "severity": severity, "passed": not ids, "n_violations": len(ids),
                     "detail": detail + (f" — nfl_id {ids[:10]}{' …' if len(ids) > 10 else ''}" if ids else "")})
        if severity == "critical":
            for i in ids:
                bad.setdefault(i, check)

    dup = mart.group_by("nfl_id").len().filter(pl.col("len") > 1)["nfl_id"].to_list()
    add("unique_mart_row", "critical", dup, "one row per athlete in gold.mart_athlete_research_base")
    add("unique_registry_row", "critical", athletes.group_by("nfl_id").len().filter(pl.col("len") > 1)["nfl_id"].to_list(),
        "one row per athlete in the athlete registry of notebook 00")
    tracked = athletes.filter(pl.col("n_primary_eligible") >= 1)
    add("position_known", "critical", tracked.filter(pl.col("position_group").is_null())["nfl_id"].to_list(),
        "Combine position group recorded for every tracked athlete")
    add("position_in_scope", "info", tracked.filter(pl.col("position_group").is_not_null()
                                                   & ~pl.col("position_group").is_in(list(positions)))["nfl_id"].to_list(),
        f"Combine position group among {list(positions)} (others cannot have an NFL target)")
    add("draft_year_known", "critical", tracked.filter(pl.col("draft_year").is_null())["nfl_id"].to_list(),
        "draft cohort recorded for every tracked athlete")
    j = exposure.join(athletes.select("nfl_id", "position_group", "draft_year"), on="nfl_id", how="left")
    add("nfl_athlete_in_registry", "info", j.filter(pl.col("position_group").is_null() & pl.col("draft_year").is_null())["nfl_id"].to_list(),
        "every NFL-tracked athlete has a Combine registry row (others cannot be linked and are left out)")
    both = j.filter(pl.col("position_group").is_not_null())
    add("same_position_group", "critical", both.filter(pl.col("position_group") != pl.col("nfl_position_group"))["nfl_id"].to_list(),
        "Combine position group = position group of the NFL play table")
    add("same_draft_cohort", "critical", both.filter(pl.col("nfl_cohort").is_not_null() & (pl.col("draft_year") != pl.col("nfl_cohort")))["nfl_id"].to_list(),
        "draft year of the registry = cohort of the NFL play table")
    add("nfl_after_combine", "critical", both.filter(pl.col("first_season") < pl.col("draft_year"))["nfl_id"].to_list(),
        "no regular-season window before the athlete's draft year (the predictors are pre-NFL)")
    orphan = manifest.filter(pl.col("eligible_primary") & ~pl.col("nfl_id").is_in(athletes["nfl_id"].to_list()))["nfl_id"].to_list()
    add("attempts_have_athlete", "critical", orphan, "every eligible Combine attempt belongs to a registry athlete")
    names = athletes.filter(pl.col("display_name").is_not_null()).group_by("display_name").agg(pl.col("nfl_id").n_unique().alias("k"),
                                                                                                pl.col("nfl_id").first().alias("i"))
    add("display_name_unique", "info", names.filter(pl.col("k") > 1)["i"].to_list(),
        "a display name shared by several ids is reported (ids are the key; names are never used to link)")
    return pl.DataFrame(rows, schema=IDENTITY.columns), bad


# ============================================================================== cohort
def frozen_feature_ids(shortlist: pl.DataFrame) -> dict[str, Any]:
    """Feature families of the transfer models from the FROZEN shortlist of notebook 04B — exact candidate ids, no
    atlas exploration. {'ENG': [feature ids], 'SPEED2': [...], 'FPCA': method or None, 'TCN': method or None,
    'COMPACT': k or None, 'selected': {candidate_id: features}}."""
    sel = {r["candidate_id"]: [f for f in str(r["features"] or "").split(";") if f] for r in shortlist.iter_rows(named=True)
           if r["selected"]}
    one = lambda cid: (sel.get(cid) or [None])[0]                                     # noqa: E731
    compact = one("C7_tcn_compact")
    return {"ENG": [f for cid in ENGINEERED_SLOTS for f in sel.get(cid, [])], "SPEED2": list(SPEED2_FEATURES),
            "SPEED_CONTROL": sel.get("C2_speed_control", []), "FPCA": one("C5_fpca"), "TCN": one("C6_tcn_b_8d"),
            "COMPACT": int(compact.rsplit("pc", 1)[1]) if compact and "|pc" in compact else None, "selected": sel}


def _complete_attempts(attempt_values: pl.DataFrame, features: Sequence[str], name: str) -> pl.DataFrame:
    """Athletes → number of attempts on which EVERY feature of ``features`` has a value."""
    if not features:
        return pl.DataFrame(schema={"nfl_id": I, name: I})
    v = (attempt_values.filter(pl.col("feature_id").is_in(list(features)) & pl.col("value").is_not_null())
         .group_by("seq_id", "nfl_id").agg(pl.col("feature_id").n_unique().alias("k")).filter(pl.col("k") == len(set(features))))
    return v.group_by("nfl_id").agg(pl.len().cast(pl.Int64).alias(name))


def cohort_manifest(athletes: pl.DataFrame, official: pl.DataFrame, attempt_values: pl.DataFrame, curve_rows: pl.DataFrame,
                    windows: pl.DataFrame, exposure: pl.DataFrame, eligibility: pl.DataFrame, primary: FrozenTarget,
                    bad_identity: dict[int, str], families: dict[str, Any], tc: TransferConfig) -> pl.DataFrame:
    """One row per Combine athlete. ``curve_rows``: (seq_id, nfl_id, has_curve) of the representation every encoder
    reads. Membership depends on availability only — no NFL label value and no prediction error enters it."""
    core = pl.all_horizontal([pl.col(c).is_not_null() for c in tc.core_official]).alias("has_official_core")
    curves = curve_rows.filter(pl.col("has_curve")).group_by("nfl_id").agg(pl.len().cast(pl.Int64).alias("n_curves"))
    signal = windows.group_by("nfl_id").agg(pl.col("S2_signal").fill_null(False).sum().cast(pl.Int64).alias("n_signal_ok"))
    tgt = eligibility.filter(pl.col("target_id") == primary.target_id).select(
        "nfl_id", pl.col("n_windows").alias("target_windows"), pl.col("eligible").alias("st_target"))
    m = primary.min_windows
    d = (athletes.select("nfl_id", "display_name", "position_group", "draft_year", pl.col("n_primary_eligible").alias("n_attempts"))
         .join(official.select("nfl_id", core), on="nfl_id", how="left")
         .join(curves, on="nfl_id", how="left").join(signal, on="nfl_id", how="left")
         .join(_complete_attempts(attempt_values, families["ENG"], "n_eng_attempts"), on="nfl_id", how="left")
         .join(_complete_attempts(attempt_values, families["SPEED2"], "n_speed_attempts"), on="nfl_id", how="left")
         .join(exposure, on="nfl_id", how="left").join(tgt, on="nfl_id", how="left"))
    zero = ["n_curves", "n_signal_ok", "n_eng_attempts", "n_speed_attempts", "n_nfl_windows", "n_nfl_games", "n_nfl_seasons", "target_windows"]
    d = d.with_columns(*[pl.col(c).fill_null(0) for c in zero], pl.col("has_official_core").fill_null(False),
                       pl.col("st_target").fill_null(False),
                       (~pl.col("nfl_id").is_in(sorted(bad_identity))).alias("identity_ok"))
    need_eng = pl.lit(True) if not families["ENG"] else pl.col("n_eng_attempts") >= 1
    d = d.with_columns(
        (pl.col("n_attempts") >= 1).alias("st_tracked"), pl.col("has_official_core").alias("st_official"),
        (need_eng & (pl.col("n_speed_attempts") >= 1)).alias("st_engineered"), (pl.col("n_curves") >= 1).alias("st_fpca"),
        (pl.col("n_curves") >= 1).alias("st_tcn"), (pl.col("n_nfl_windows") >= 1).alias("st_nfl"),
        pl.when(pl.col("n_attempts") == 0).then(pl.lit("no eligible attempt"))
        .when(pl.col("n_attempts") == 1).then(pl.lit("single attempt"))
        .when(pl.col("n_signal_ok") >= pl.col("n_attempts")).then(pl.lit("repeat, all signal-clean"))
        .otherwise(pl.lit("repeat, ≥ 1 flagged attempt")).alias("combine_quality"),
        pl.when(pl.col("n_nfl_windows") == 0).then(pl.lit("no NFL window"))
        .when(pl.col("target_windows") < m).then(pl.lit(f"< {m} target windows"))
        .when(pl.col("target_windows") < 4 * m).then(pl.lit(f"{m}–{4 * m - 1} target windows"))
        .otherwise(pl.lit(f"≥ {4 * m} target windows")).alias("exposure_band"))
    flags = [c for _, c, _ in STAGES[1:-1]]
    d = d.with_columns(pl.all_horizontal([pl.col(c) for c in flags] + [pl.col("identity_ok")]).alias("in_primary_cohort"),
                       (pl.col("st_official") & pl.col("st_target") & pl.col("identity_ok")).alias("in_wide_cohort"))
    reason = pl.when(~pl.col("identity_ok")).then(pl.lit("identity_conflict"))
    for stage, col, _ in STAGES[1:-1]:
        reason = reason.when(~pl.col(col)).then(pl.lit(f"no_{stage}"))
    d = d.with_columns(reason.otherwise(pl.lit("")).alias("exclusion_reason"), pl.lit(TRANSFER_VERSION).alias("manifest_version"))
    return d.select([pl.col(k).cast(t) for k, t in COHORT.columns.items()]).sort("nfl_id")


def _groups(d: pl.DataFrame, by: Sequence[str]) -> list[tuple[str, str, pl.DataFrame]]:
    out = [(ALL, ALL, d)]
    for col in by:
        for g in sorted(d[col].drop_nulls().unique().to_list(), key=str):
            out.append((col, str(g), d.filter(pl.col(col) == g)))
    return out


def funnel_table(cohort: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for gt, g, d in _groups(cohort, ("position_group", "draft_year")):
        cum = pl.lit(True)
        for i, (stage, col, definition) in enumerate(STAGES):
            alone = d.height if col is None else int(d[col].sum())
            if col is not None:
                cum = cum & pl.col(col)
            rows.append({"stage": stage, "stage_order": i, "group_type": gt, "group": g, "n_meeting_rule": alone,
                         "n_cumulative": d.height if col is None else d.filter(cum).height, "definition": definition})
    return pl.DataFrame(rows, schema=FUNNEL.columns)


def breakdown_table(cohort: pl.DataFrame, by: Sequence[str] = ("draft_year", "first_season", "exposure_band", "combine_quality")
                    ) -> pl.DataFrame:
    """Athlete counts at the main funnel stages by position group inside every level of ``by`` (+ ALL positions)."""
    rows = []
    for col in by:
        for g in sorted(cohort[col].drop_nulls().unique().to_list(), key=str):
            d = cohort.filter(pl.col(col) == g)
            for p, sub in [(ALL, d), *[(q, d.filter(pl.col("position_group") == q))
                                       for q in sorted(d["position_group"].drop_nulls().unique().to_list())]]:
                rows.append({"group_type": col, "group": str(g), "position_group": p, "n_roster": sub.height,
                             "n_tracked": int(sub["st_tracked"].sum()), "n_nfl": int(sub["st_nfl"].sum()),
                             "n_target": int(sub["st_target"].sum()), "n_primary": int(sub["in_primary_cohort"].sum()),
                             "n_wide": int(sub["in_wide_cohort"].sum())})
    return pl.DataFrame(rows, schema=COUNTS.columns)


def reasons_table(cohort: pl.DataFrame) -> pl.DataFrame:
    rows = [{"exclusion_reason": r or "in_primary_cohort", "group_type": gt, "group": g, "n_athletes": int(n)}
            for gt, g, d in _groups(cohort, ("position_group", "draft_year"))
            for r, n in d.group_by("exclusion_reason").len().sort("exclusion_reason").iter_rows()]
    return pl.DataFrame(rows, schema=REASONS.columns)


def missingness_audit(official: pl.DataFrame, cohort: pl.DataFrame, attempt_values: pl.DataFrame, mart_columns: Sequence[str],
                      families: dict[str, Any], tc: TransferConfig) -> pl.DataFrame:
    """Missing pre-NFL values by variable, population and group, and the pre-declared way each variable enters the
    models. The inclusion of an optional test depends on its COVERAGE among target-exposed athletes — never on a label."""
    kinds = [*[(c, "official_core") for c in tc.core_official if c != WEIGHT], (WEIGHT, "official_core_body"), (tc.height, "body"),
             *[(c, "optional_test") for c in tc.optional_tests], *[(c, "optional_body") for c in tc.optional_body]]
    d = cohort.select("nfl_id", "position_group", "draft_year", "st_target", "st_tracked").join(official, on="nfl_id", how="left")
    eng = (attempt_values.filter(pl.col("feature_id").is_in([*families["ENG"], *families["SPEED2"]]))
           .group_by("feature_id", "nfl_id").agg(pl.col("value").is_not_null().any().alias("has"),
                                                  pl.col("missing_reason").filter(pl.col("value").is_null()).drop_nulls().unique()
                                                  .alias("why")))
    rows = []
    for pop_name, pop in (("roster", d), ("target_exposed", d.filter(pl.col("st_target")))):
        for gt, g, sub in _groups(pop, ("position_group", "draft_year")):
            for var, kind in kinds:
                absent = var not in mart_columns
                miss = sub.height if absent else int(sub[var].is_null().sum())
                share = miss / sub.height if sub.height else NAN
                rows.append({"variable": var, "kind": kind, "population": pop_name, "group_type": gt, "group": g,
                             "n_athletes": sub.height, "n_missing": miss, "share_missing": share,
                             "missing_class": "field_unavailable" if absent or (sub.height and miss == sub.height) else
                             "no_value_recorded" if miss else "complete", "in_model": "", "rule": ""})
            tr = sub.filter(pl.col("st_tracked"))
            for fid in [*families["ENG"], *families["SPEED2"]]:
                e = tr.select("nfl_id").join(eng.filter(pl.col("feature_id") == fid), on="nfl_id", how="left")
                miss = int((~e["has"].fill_null(False)).sum())
                why = " · ".join(sorted({w for lst in e.filter(~pl.col("has").fill_null(False))["why"].to_list() if lst
                                         for w in lst if w}))
                rows.append({"variable": fid, "kind": "engineered", "population": pop_name, "group_type": gt, "group": g,
                             "n_athletes": tr.height, "n_missing": miss, "share_missing": miss / tr.height if tr.height else NAN,
                             "missing_class": f"tracking_insufficient ({why})" if miss and why else
                             "tracking_insufficient" if miss else "complete", "in_model": "", "rule": ""})
    out = pl.DataFrame(rows, schema=MISSING.columns)
    ref = {r["variable"]: r for r in out.filter((pl.col("population") == "target_exposed") & (pl.col("group_type") == ALL)).iter_rows(named=True)}

    def rule(var: str, kind: str) -> tuple[str, str]:
        r = ref.get(var)
        cov = 1.0 - r["share_missing"] if r and r["n_athletes"] else NAN
        if kind.startswith("official_core"):
            return "mandatory", "an athlete without it is not in the primary or wide cohort; never imputed"
        if kind == "engineered":
            return "mandatory", "an athlete without a measured value is not in the primary cohort; never imputed"
        if kind == "body":
            return "imputed_with_indicator", "training-fold median + missingness indicator"
        if np.isfinite(cov) and cov >= tc.optional_min_coverage:
            return "imputed_with_indicator", (f"coverage {cov:.2f} ≥ {tc.optional_min_coverage:g} among target-exposed athletes: "
                                              "training-fold median + missingness indicator")
        return "excluded_low_coverage", f"coverage {cov:.2f} < {tc.optional_min_coverage:g} among target-exposed athletes"
    keys = out.select("variable", "kind").unique().sort("variable", "kind")
    decided = [rule(v, k) for v, k in keys.iter_rows()]
    rules = keys.with_columns(pl.Series("in_model", [d[0] for d in decided], dtype=pl.String),
                              pl.Series("rule", [d[1] for d in decided], dtype=pl.String))
    return out.drop("in_model", "rule").join(rules, on=["variable", "kind"], how="left").select(list(MISSING.columns))


def official_design(missing: pl.DataFrame, tc: TransferConfig) -> dict[str, list[str]]:
    """Columns of the baseline models, fixed by ``missingness_audit``: M0 = position, cohort, body; M1 = M0 + official
    tests. ``indicators``: columns that get a missingness indicator (fitted on training folds)."""
    r = missing.filter((pl.col("population") == "target_exposed") & (pl.col("group_type") == ALL))
    use = {x["variable"]: x for x in r.iter_rows(named=True) if x["in_model"] in ("mandatory", "imputed_with_indicator")}
    body = [c for c in (WEIGHT, tc.height, *tc.optional_body) if c in use]
    tests = [c for c in (*[c for c in tc.core_official if c != WEIGHT], *tc.optional_tests) if c in use]
    return {"m0_numeric": body, "m1_numeric": [*body, *tests],
            "indicators": [c for c in (*body, *tests) if use[c]["in_model"] == "imputed_with_indicator"],
            "excluded": sorted(x["variable"] for x in r.iter_rows(named=True) if x["in_model"] == "excluded_low_coverage")}


# ============================================================================== folds
def _strata(d: pl.DataFrame) -> np.ndarray:
    return (d["position_group"].fill_null("?") + "|" + d["draft_year"].cast(pl.String).fill_null("?")).to_numpy().astype(str)


def fold_viability(position: np.ndarray, strata: np.ndarray, fold: np.ndarray, tc: TransferConfig) -> dict[str, Any]:
    """Size rules of a fold design on the PRIMARY cohort — counts of athletes only."""
    folds = np.unique(fold)
    held = np.array([(fold == f).sum() for f in folds])
    cells = [(int(((fold != f) & (position == p)).sum()), f"{p} outside fold {f}") for f in folds for p in np.unique(position)]
    n_cell, smallest = min(cells) if cells else (0, "no athlete")
    empty = [f"{s} absent from fold {f}" for f in folds for s in np.unique(strata)
             if (strata == s).sum() >= folds.size and not ((fold == f) & (strata == s)).any()]
    ok = bool(held.size and held.min() >= tc.min_heldout_fold and n_cell >= tc.min_position_train)
    why = [] if ok else [x for c, x in ((held.size and held.min() >= tc.min_heldout_fold, f"a fold holds out {int(held.min()) if held.size else 0} "
                                         f"athletes (< {tc.min_heldout_fold})"),
                                        (n_cell >= tc.min_position_train, f"{smallest}: {n_cell} training athletes (< {tc.min_position_train})")) if not c]
    return {"min_heldout": int(held.min()) if held.size else 0, "max_heldout": int(held.max()) if held.size else 0,
            "min_position_train": int(n_cell), "smallest_cell": smallest, "viable": ok,
            "reason": "; ".join(why) if why else "size rules met" + (f" (note: {'; '.join(empty[:3])})" if empty else "")}


def assign_folds(cohort: pl.DataFrame, registry: pl.DataFrame, tc: TransferConfig, source: str = "auto"
                 ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(fold table of the primary ∪ wide cohort, table of the designs considered).

    Designs, in order of preference: ``registry`` (outer folds of the notebook-00 split registry, repeat 0 — the folds
    every saved encoder was fitted under), then new stratified folds with the largest K of ``outer_folds_grid`` that
    meets the size rules. ``source``: 'auto' · 'registry' · 'new'. The choice reads position, cohort and membership
    only. Athletes outside the registry (wide cohort without tracking) are dealt to folds inside their stratum."""
    if source not in ("auto", "registry", "new"):
        raise ValueError(f"FOLD_SOURCE must be 'auto', 'registry' or 'new', got {source!r}")
    pop = cohort.filter(pl.col("in_primary_cohort") | pl.col("in_wide_cohort")).sort("nfl_id")
    ids, prim = pop["nfl_id"].to_numpy(), pop["in_primary_cohort"].to_numpy()
    if prim.sum() < 2 * min(tc.outer_folds_grid):
        raise FoldError(f"only {int(prim.sum())} athletes in the primary cohort: no fold design is possible")
    strata, position = _strata(pop), pop["position_group"].fill_null("?").to_numpy().astype(str)
    reg = dict(registry.filter(pl.col("repeat") == 0).select("nfl_id", "outer_fold").iter_rows())

    def complete(known: dict[int, int], k: int, label: str) -> np.ndarray:
        fold = np.array([known.get(int(i), -1) for i in ids], dtype=np.int64)
        rest = fold < 0
        if rest.any():
            fold[rest] = make_folds(strata[rest], k, 1, seed_for(tc.seed, "transfer_rest", label))[0]
        return fold

    designs: list[tuple[str, int, np.ndarray]] = []
    if all(int(i) in reg for i in ids[prim]) and reg:
        k = len(set(reg.values()))
        designs.append(("registry", k, complete({int(i): int(reg[int(i)]) for i in ids if int(i) in reg}, k, "registry")))
    for k in tc.outer_folds_grid:
        dealt = make_folds(strata[prim], k, 1, seed_for(tc.seed, "transfer_outer", k))[0]
        designs.append((f"new_{k}", k, complete(dict(zip(ids[prim].tolist(), dealt.tolist())), k, f"new_{k}")))
    rows, chosen = [], None
    for name, k, fold in designs:
        v = fold_viability(position[prim], strata[prim], fold[prim], tc)
        allowed = source == "auto" or name.startswith(source)
        pick = chosen is None and v["viable"] and allowed
        if pick:
            chosen = (name, k, fold)
        rows.append({"candidate": name, "n_folds": k, **{x: v[x] for x in ("min_heldout", "max_heldout", "min_position_train", "smallest_cell", "viable")},
                     "selected": pick, "reason": v["reason"] + ("" if allowed else f" · not considered (FOLD_SOURCE = {source!r})")})
    plan = pl.DataFrame(rows, schema=FOLD_PLAN.columns)
    if chosen is None:
        raise FoldError("no viable athlete-grouped fold design for the primary cohort:\n" + "\n".join(
            f"  {r['candidate']}: {r['reason']}" for r in plan.iter_rows(named=True)))
    name, k, fold = chosen
    inner = make_folds(strata, tc.inner_folds, 1, seed_for(tc.seed, "transfer_inner"))[0]
    out = pop.select("nfl_id", "position_group", "draft_year", "in_primary_cohort", pl.col("st_target").alias("target_eligible")).with_columns(
        pl.Series("stratum", strata), pl.Series("outer_fold", fold.astype(np.int64)), pl.Series("inner_fold", inner.astype(np.int64)),
        pl.lit(name).alias("fold_source"), pl.lit(k, dtype=pl.Int64).alias("n_folds"), pl.lit(TRANSFER_VERSION).alias("manifest_version"))
    return out.select([pl.col(c).cast(t) for c, t in FOLDS.columns.items()]).sort("nfl_id"), plan


def folds_hash(folds: pl.DataFrame) -> str:
    """Identity of the fold manifest: athletes and their outer / inner folds."""
    return stable_hash(folds.sort("nfl_id").select("nfl_id", "outer_fold", "inner_fold", "in_primary_cohort").rows())


def encoder_splits(folds: pl.DataFrame, registry: pl.DataFrame) -> pl.DataFrame:
    """Split table (``data.SPLITS`` columns, repeat 0) the representation fits run under.

    registry folds   the split registry unchanged: every tracked athlete — linked to the NFL or not — is held out in his
                     own fold, exactly as when the saved encoders were fitted.
    new folds        transfer athletes take their transfer fold; every other tracked athlete gets outer fold −1: he is
                     never held out and his UNLABELLED Combine attempts take part in every encoder fit.
    The inner (early-stopping) fold is the registry's, taken among the training athletes of each outer fold."""
    reg = registry.filter(pl.col("repeat") == 0)
    if folds["fold_source"][0] == "registry":
        return reg
    m = folds.select("nfl_id", pl.col("outer_fold").alias("_f"))
    return (reg.join(m, on="nfl_id", how="left").with_columns(pl.col("_f").fill_null(-1).alias("outer_fold")).drop("_f")
            .select(reg.columns).sort("nfl_id"))


def assert_fold_integrity(folds: pl.DataFrame, enc_splits: pl.DataFrame, manifest: pl.DataFrame) -> list[str]:
    """Raise on a violated split contract; return the sentences describing what was verified."""
    if folds["nfl_id"].n_unique() != folds.height:
        raise TransferGateError("an athlete appears in more than one outer fold")
    seqs = manifest.filter(pl.col("eligible_primary")).join(enc_splits.select("nfl_id", "outer_fold"), on="nfl_id", how="inner")
    spread = seqs.group_by("nfl_id").agg(pl.col("outer_fold").n_unique().alias("k")).filter(pl.col("k") > 1).height
    if spread:
        raise TransferGateError(f"{spread} athlete(s) have Combine attempts in several encoder folds")
    e = dict(enc_splits.select("nfl_id", "outer_fold").iter_rows())
    moved = [i for i, f in folds.select("nfl_id", "outer_fold").iter_rows() if i in e and e[i] != f]
    if moved:
        raise TransferGateError(f"{len(moved)} athlete(s) are held out in one fold for the encoders and another for the "
                                f"prediction models: {moved[:5]}")
    return [f"{folds.height} athletes, one outer fold each ({folds['n_folds'][0]} folds, source {folds['fold_source'][0]})",
            "all eligible Combine attempts of an athlete share his encoder fold",
            "the encoder fold and the prediction fold of every linked athlete are the same fold",
            "NFL labels are one value per athlete, so all his seasons follow his fold by construction"]


# ============================================================================== representation leakage audit
SELECTIONS = (
    ("TCN-B consistency term (kind, weight)", "tcn", "notebook 03 chose it on the DESIGN athletes' Combine attempts (cross-fitted); "
     "it returned the pre-declared default. No NFL value. Design athletes are held out in some transfer fold: this choice is "
     "transductive in the Combine data only."),
    ("TCN architecture, optimiser and budget", "tcn", "fixed before any fit (TcnConfig); reused unchanged."),
    ("FPCA number of components", "fpca", "smallest K with out-of-fold explained variance ≥ 0.95 on Combine curves (notebook 02). No NFL value."),
    ("Frozen shortlist (engineered feature, speed control, compact dimension)", "shortlist",
     "notebook 04B, by Combine reliability and redundancy on all tracked athletes. No NFL value; frozen before this milestone."),
    ("Matching rule (variables, caliper, k)", "matching", "notebook 04B MatchConfig; scales from design athletes. No NFL value."),
)


def compatibility_table(records: Sequence[dict[str, Any]], folds: pl.DataFrame) -> pl.DataFrame:
    """Saved fits against the transfer folds. ``records``: one dict per saved model with ``artifact``, ``family``,
    ``method``, ``repeat``, ``fold`` and ``athletes`` (everything that touched the fit: weights, early stopping,
    normalisation). A model is reusable for transfer fold f only when it was fitted for split repeat 0, fold f, and
    none of its athletes is held out in transfer fold f. All-athlete fits are descriptive-only by construction."""
    held = {int(f): set(g["nfl_id"].to_list()) for (f,), g in folds.partition_by(["outer_fold"], as_dict=True).items()}
    source = folds["fold_source"][0]
    rows = []
    for r in records:
        used, f, rep = set(int(i) for i in r["athletes"]), int(r["fold"]), int(r["repeat"])
        if rep != 0 or f not in held:
            over = len(used & set().union(*held.values())) if held else 0
            status, use, note = ("descriptive_only", "not used as a predictor",
                                 "all-athlete reference fit" if rep < 0 else "another split repeat: no transfer fold corresponds to it")
            n_held = sum(len(v) for v in held.values())
        else:
            over, n_held = len(used & held[f]), len(held[f])
            ok = over == 0 and source == "registry"
            status = "compatible" if ok else "descriptive_only"
            use = "reusable in 06B for this fold" if ok else "refit on the training athletes of the transfer fold"
            note = ("no held-out transfer athlete took part in the fit" if ok else
                    f"{over} held-out athlete(s) took part in the fit" if over else "transfer folds differ from the registry folds")
        rows.append({"artifact": str(r["artifact"]), "family": str(r["family"]), "method": str(r["method"]), "repeat": rep, "fold": f,
                     "n_fit_athletes": len(used), "n_heldout_transfer": int(n_held), "n_overlap": int(over), "status": status,
                     "use": use, "note": note})
    for i, (what, family, note) in enumerate(SELECTIONS):
        rows.append({"artifact": what, "family": family, "method": "selection", "repeat": -1, "fold": -1 - i, "n_fit_athletes": 0,
                     "n_heldout_transfer": 0, "n_overlap": 0, "status": "combine_only_selection", "use": "kept as frozen", "note": note})
    return pl.DataFrame(rows, schema=COMPAT.columns)


def reusable_checkpoints(compat: pl.DataFrame, method: str) -> dict[int, str]:
    """fold → artifact path of the saved fits of ``method`` that passed the compatibility audit."""
    ok = compat.filter((pl.col("method") == method) & (pl.col("status") == "compatible"))
    return {int(f): str(a) for f, a in ok.select("fold", "artifact").iter_rows()}


# ============================================================================== input contract, gate, hand-over
def describe_input(name: str, path: Path | None, df: pl.DataFrame | None, required: Sequence[str], role: str, rel: Any = str
                   ) -> dict[str, Any]:
    """One entry of ``transfer_input_contract.json``: where an input lives, its size, state and required columns."""
    cols = list(df.columns) if df is not None else []
    return {"name": name, "role": role, "path": rel(path) if path is not None else None, "present": df is not None,
            "rows": None if df is None else df.height, "required_columns": list(required),
            "missing_columns": [c for c in required if c not in cols],
            "fingerprint": file_fingerprint(path) if path is not None and Path(path).is_file() else None}


def assert_inputs(entries: Sequence[dict[str, Any]]) -> None:
    bad = [f"{e['name']}: " + ("absent" if not e["present"] else f"lacks columns {e['missing_columns']}") for e in entries
           if not e["present"] or e["missing_columns"]]
    if bad:
        raise TransferGateError("input contract violated — " + "; ".join(bad))


def gate_06a(freeze: pl.DataFrame, identity: pl.DataFrame, cohort: pl.DataFrame, folds: pl.DataFrame, plan: pl.DataFrame,
             compat: pl.DataFrame, tc: TransferConfig) -> pl.DataFrame:
    """Gate 06A — data integrity. Every row must pass before 06B may fit a model."""
    n = int(cohort["in_primary_cohort"].sum())
    conflicts = int((~cohort["identity_ok"]).sum())
    crit = identity.filter((pl.col("severity") == "critical") & ~pl.col("passed"))
    saved = compat.filter(pl.col("status") != "combine_only_selection")
    unresolved = saved.filter((pl.col("status") == "compatible") & (pl.col("n_overlap") > 0)).height
    rows = [("target_registry_frozen", not freeze.filter(~pl.col("passed")).height, f"{freeze.height} checks of the registry"),
            ("athlete_linkage_valid", not cohort.filter(pl.col("in_primary_cohort") & ~pl.col("identity_ok")).height,
             (f"{crit.height} critical identity check(s) flagged {conflicts} athlete(s): excluded from every cohort and listed, "
              "none repaired") if crit.height else "every critical identity check passed"),
            ("evaluation_folds_viable", bool(plan.filter(pl.col("selected")).height == 1),
             f"{folds['fold_source'][0]}: {folds['n_folds'][0]} folds over {int(folds['in_primary_cohort'].sum())} primary-cohort athletes"),
            ("common_cohort_documented", n >= tc.min_athletes, f"{n} athletes support every model M0–M7 (minimum {tc.min_athletes})"),
            ("representation_leakage_resolved", unresolved == 0,
             f"{saved.filter(pl.col('status') == 'compatible').height} saved fit(s) reusable, "
             f"{saved.filter(pl.col('status') == 'descriptive_only').height} descriptive-only (refitted on training athletes in 06B)")]
    return pl.DataFrame([{"criterion": c, "passed": bool(p), "detail": d} for c, p, d in rows], schema=GATE.columns)


def assert_gate(gate: pl.DataFrame, name: str) -> None:
    failed = gate.filter(~pl.col("passed"))
    if failed.height:
        raise TransferGateError(f"Gate {name} NOT passed — stop here.\n" + "\n".join(
            f"  ✗ {r['criterion']}: {r['detail']}" for r in failed.iter_rows(named=True)))


def require_audit(store: Any, doc: dict[str, Any], folds: pl.DataFrame, notebook: str = "06a_transfer_cohort_and_leakage_audit.ipynb"
                  ) -> dict[str, Any]:
    """Manifest of notebook 06A, verified against the registry and the fold table in hand. 06B and 06C start here."""
    m = store.read_json(AUDIT_MANIFEST, notebook)
    if m.get("gate_06a") != "PASS":
        raise TransferGateError(f"gate 06A is {m.get('gate_06a')!r} in {AUDIT_MANIFEST}.json: rerun {notebook} until it passes")
    if m.get("targets_version_hash") != doc.get("version_hash"):
        raise TransferGateError(f"the cohort audit used targets {m.get('targets_version_hash')}, the registry now holds "
                                f"{doc.get('version_hash')}: rerun {notebook}")
    if m.get("folds_hash") != folds_hash(folds):
        raise TransferGateError(f"the fold table on disk is not the one the cohort audit recorded: rerun {notebook}")
    return m
