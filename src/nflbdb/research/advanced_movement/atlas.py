"""Feature and characteristic atlas of the Combine measurements (notebook 04A). NumPy / Polars only.

Purpose     consolidate what earlier stages already measured into ONE table per feature, with the same columns
            whatever the estimator, and screen it. Nothing is modelled against an outcome and no outcome is read.
Reuse       every value comes from a persisted dataset: official results (research mart), forty-yard-dash attempt
            features and sprint-model parameters (sprint rescue attempt table), recording-window variables
            (notebook 02), functional PCA scores (notebook 02), encoder embeddings (notebook 03), the Phase 04
            feature registry (other drills, events). The only new measurements are four sustained-acceleration
            features (``sustained_table``) read from the persisted sequence frames.
Designs     a feature is characterised by the procedure its measurement design allows — never one estimator for all:
                static_athlete     one official value per athlete: no repeatability can be estimated here
                repeated_attempt   a value per attempt: stratified ICC(1) within position, athlete bootstrap
                model_parameter    as repeated_attempt, on attempts whose fit passed its validity flags
                functional_score   out-of-fold component scores: reliability with the outer fold in the stratum
                                   (reused from notebook 02, not recomputed)
                neural_embedding   a vector: rotation-invariant trace reliability and repeat-attempt retrieval
                                   (reused from notebook 03); a coordinate of an encoder is never a feature
                event_rate         counts and rates: Phase 04 registry evidence only
            The output schema (``CHARACTERISTICS``) is the same for all; ``reliability_method`` says what was used.
Unit        the athlete. Attempts are never counted as athletes: intervals resample athletes inside position.
Sustained   measured intervals only. Every new feature is anchored on a SPEED the recording contains (the moment
acceleration speed first reaches ``anchor_speed``), so it does not depend on where the recording starts. When the anchor,
            the window or the upper threshold is not inside the recording the value is missing, with the reason.
Screening   Strong · Provisional · Exploratory · Reject / not currently usable, from intervals against heuristic
            reference lines. A low correlation with the stopwatch never makes a feature novel: "beyond_stopwatch"
            needs reliable variance the official times do not explain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from ..sprint_rescue.attempts import first_crossing
from ..sprint_rescue.config import RescueConfig, feature_specs
from ..sprint_rescue.stats import design, partial_corr, residualise, seed_for
from . import fpca_eval as fe
from .config import NotebookControls
from .fpca_eval import FORTY, HEIGHT, JOINT_WINDOW, TEN, WEIGHT, athlete_means, first_two
from .repr_eval import _table
from .sequences import SequenceSet

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
ATLAS_VERSION = "1"
FAMILIES = {"F1": "official Combine", "F2": "engineered kinematics", "F3": "maneuvers and events", "F4": "sprint model",
            "F5": "learned representations", "F6": "sustained acceleration", "F7": "acquisition / nuisance"}
GRADES = ("Strong", "Provisional", "Exploratory", "Reject / not currently usable")
ATTEMPT_TABLE, WINDOW_TABLE, SUSTAINED_TABLE = ("sprint_rescue.attempt_censoring_diagnostics",
                                                "advanced_movement.sequence_window_flags", "advanced_movement.atlas_sustained")
MART_TABLE, GATE_TABLE, VECTOR_TABLE, REGISTRY_TABLE = ("gold.mart_athlete_research_base", "advanced_movement.fpca_component_gate",
                                                        "advanced_movement.repr_method_summary",
                                                        "phase_04.V_feature_validation_registry")
ICC_METHOD = "stratified ICC(1) within position, all eligible attempts, athlete bootstrap inside position"
MODEL_COLUMNS = ("model_vmax", "model_tau_s", "model_a0", "model_a_at_ref")
NUISANCE_VARS = ("first_speed", "duration_s", "frames_after_peak", "path_yd", "pos_resid_rms", "a_tan_roughness", "share_boundary")
OFFICIAL_SPECS = (
    (FORTY, "official forty-yard dash", "s", "official Combine forty-yard-dash time", 4.0, 6.5),
    (TEN, "official 10-yard split", "s", "official split of the forty at 10 yards", 1.3, 2.2),
    ("cmb_vertical", "vertical jump", "in", "official vertical jump", 15.0, 50.0),
    ("cmb_broad_jump", "broad jump", "in", "official standing broad jump", 80.0, 150.0),
    ("cmb_short_shuttle", "short shuttle", "s", "official 20-yard short-shuttle time", 3.7, 5.5),
    ("cmb_three_cone", "three-cone drill", "s", "official three-cone-drill time", 6.3, 9.0),
    ("cmb_bench_reps", "bench press", "reps", "official bench-press repetitions", 0.0, 50.0),
    (HEIGHT, "height", "in", "height measured at the Combine", 64.0, 84.0),
    (WEIGHT, "weight", "lb", "weight measured at the Combine", 150.0, 400.0),
    ("cmb_hand_size", "hand size", "in", "hand size measured at the Combine", 7.5, 12.5),
    ("cmb_arm_length", "arm length", "in", "arm length measured at the Combine", 28.0, 38.0),
    ("cmb_wing_span", "wing span", "in", "wing span measured at the Combine", 68.0, 90.0),
)
EXTRA_ATTEMPT = (
    ("peak_a_tan_interior", "F2", "repeated_attempt", "candidate", "yd/s²", "largest tangential acceleration away from the filter edges"),
    ("peak_decel", "F2", "repeated_attempt", "candidate", "yd/s²", "strongest single-frame braking"),
    ("model_rmse", "F4", "model_parameter", "diagnostic", "yd/s", "root-mean-square residual of the fitted sprint curve"),
    ("model_r2", "F4", "model_parameter", "diagnostic", "–",
     "share of the recorded speed variance the fitted curve reproduces — a goodness of fit, not evidence that the parameters are identifiable"),
    ("tau_rel_width", "F4", "model_parameter", "diagnostic", "–", "relative width of the near-optimal τ range (identifiability index)"),
    ("vmax_extrapolation_ratio", "F4", "model_parameter", "diagnostic", "–", "fitted vmax / largest recorded speed"),
)
WINDOW_SPECS = (
    ("first_speed", "yd/s", "speed on the first recorded frame"), ("duration_s", "s", "recorded duration"),
    ("path_yd", "yd", "recorded path length"), ("frames_after_peak", "frames", "frames recorded after the speed peak"),
    ("last_speed_frac_of_peak", "–", "last-frame speed / peak speed"),
    ("pos_resid_rms", "yd", "RMS distance between raw and smoothed position (tracking noise)"),
    ("a_tan_roughness", "yd/s²", "frame-to-frame roughness of the tangential acceleration"),
    ("share_boundary", "–", "share of frames inside a smoothing boundary"),
    ("first_frame_censored", "flag", "already moving on the first recorded frame"),
    ("start_above_ref_speed", "flag", "recording starts at or above the reference speed"),
    ("last_frame_censored", "flag", "recording stops while the athlete is still at top speed"),
    ("ends_at_peak", "flag", "no frame recorded after the speed peak"),
    ("boundary_peak", "flag", "an acceleration extreme lies on a smoothing-boundary frame"),
    ("signal_flag", "flag", "Phase 04 signal-quality flag"),
)
PLAUSIBLE = {"yd/s": (2.0, 14.0), "yd/s²": (-5.0, 25.0), "s": (0.05, 8.0), "yd": (2.0, 60.0)}


@dataclass(frozen=True)
class AtlasConfig:
    version: str = ATLAS_VERSION
    # sustained acceleration (proposed, fixed before any value is read) -------------------------------------
    anchor_speed: float = 4.0                     # yd/s: a level inside the recorded range of nearly every attempt
    upper_speed: float = 8.0                      # yd/s: second threshold of the time-between-speeds feature
    accel_windows_s: tuple[float, ...] = (0.5, 1.0)
    gain_distance_yd: float = 10.0
    # screening reference lines (heuristic; the continuous evidence is exported beside every label) ---------
    icc_reference: float = 0.75                   # established convention (Koo & Li 2016 "good"), FpcaConfig
    icc_floor: float = 0.50                       # established convention ("moderate")
    redundancy_reference: float = 0.70            # proposed (RescueConfig.redundancy_reference)
    window_reference: float = 0.25                # proposed (FpcaConfig.window_reference)
    cohort_reference: float = 0.25                # proposed: share of athlete variance explained by cohort, within position
    cohort_gain_reference: float = 0.10           # proposed: probe accuracy gain over position (vector features)
    unique_margin: float = 0.10                   # proposed: reliable variance not explained by the official times
    min_coverage: float = 0.80                    # proposed: share of tracked athletes with a value
    min_repeat_athletes: int = 50                 # proposed: athletes with a repeat needed to grade a feature
    min_athletes: int = 10                        # established: RescueConfig.min_athletes (below → no estimate)
    cluster_rho: float = 0.70                     # proposed: features joined when their mean |ρ| reaches this
    # execution --------------------------------------------------------------------------------------------
    n_boot: int = 1000
    n_boot_r2: int = 500

    def params(self) -> dict[str, Any]:
        return {**asdict(self), "accel_windows_s": list(self.accel_windows_s)}

    def for_mode(self, controls: NotebookControls) -> "AtlasConfig":
        """Smoke mode keeps every scientific parameter; it only shortens the bootstraps."""
        return replace(self, n_boot=200, n_boot_r2=100) if controls.smoke else self


def _num(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def sustained_specs(c: AtlasConfig) -> tuple[tuple[str, str, str, str], ...]:
    """(name, unit, definition, window) of the sustained-acceleration features added by this milestone."""
    a, u = _num(c.anchor_speed), _num(c.upper_speed)
    out = [(f"accel_{_num(w)}s_from_v{a}", "yd/s²",
            f"mean acceleration over the {w:g} s that follow the moment speed first reaches {c.anchor_speed:g} yd/s "
            "(speed gain / window, both ends inside the recording)", f"{w:g} s after the anchor speed") for w in c.accel_windows_s]
    out.append((f"speed_gain_{_num(c.gain_distance_yd)}yd_from_v{a}", "yd/s",
                f"speed gained over the {c.gain_distance_yd:g} yd of path that follow the moment speed first reaches "
                f"{c.anchor_speed:g} yd/s", f"{c.gain_distance_yd:g} yd after the anchor speed"))
    out.append((f"time_v{a}_to_v{u}_s", "s", f"time between the first crossings of {c.anchor_speed:g} and {c.upper_speed:g} yd/s "
                "(defined only when both crossings are recorded)", f"{c.anchor_speed:g} → {c.upper_speed:g} yd/s"))
    return tuple(out)


# ============================================================================== contracts
DICTIONARY = TableContract(
    "feature_dictionary", "One row per feature definition: what it is, where it comes from, how it may be read.",
    {"feature_id": S, "feature_name": S, "family": S, "drill": S, "position_applicability": S, "grain": S, "units": S,
     "definition": S, "source_module": S, "provenance": S, "smoothing": S, "time_or_distance_window": S,
     "measurement_status": S, "interpretation_status": S, "quality_flags": S, "version": S, "design": S, "role": S,
     "source_table": S, "source_column": S, "plausible_lo": F, "plausible_hi": F}, key=("feature_id",), area="feature",
    docs={"feature_id": "<family>.<drill key>.<name>; the same name on another drill is another feature",
          "design": "static_athlete · repeated_attempt · model_parameter · functional_score · neural_embedding · event_rate",
          "role": "reference (official result) · candidate · diagnostic · nuisance (acquisition variable) · descriptive",
          "measurement_status": "measured · derived · model-based · learned · reused evidence only",
          "plausible_lo": "broad plausibility screen (proposed); a value outside is counted, never removed",
          "units": "official results: the unit of the Combine results file, not re-verified here"})
SUSTAINED = TableContract(
    "atlas_sustained", "Sustained-acceleration features of every eligible forty attempt, with the reason of each absence.",
    {"seq_id": S, "nfl_id": I, "feature": S, "value": F, "missing_reason": S}, key=("seq_id", "feature"), kind="dataset",
    area="sequence × feature",
    docs={"missing_reason": "empty when measured · start_above_anchor (the anchor crossing was not recorded) · "
                            "anchor_not_reached · window_exceeds_recording · upper_threshold_not_reached"})
ATTEMPT_VALUES = TableContract(
    "atlas_attempt_features", "Scalar features of every eligible forty attempt (long form), missing values kept with a reason.",
    {"feature_id": S, "seq_id": S, "nfl_id": I, "value": F, "missing_reason": S}, key=("feature_id", "seq_id"),
    kind="dataset", area="feature × sequence")
ATHLETE_VALUES = TableContract(
    "atlas_athlete_features", "Athlete-level scalar features: official values, and the mean of the athlete's attempts.",
    {"feature_id": S, "nfl_id": I, "position_group": S, "draft_cohort": S, "n_attempts": I, "value": F},
    key=("feature_id", "nfl_id"), kind="dataset", area="feature × athlete",
    docs={"n_attempts": "attempts with a value averaged (null for an official measurement)",
          "value": "attempt features: mean over the athlete's eligible attempts with a value"})
COVERAGE = TableContract(
    "atlas_coverage", "Athletes and attempts with a value, by feature and group.",
    {"feature_id": S, "group_type": S, "group": S, "n_athletes": I, "n_athletes_value": I, "coverage_athletes": F,
     "n_athletes_repeat": I, "n_attempts": I, "n_attempts_value": I, "missing_reasons": S},
    key=("feature_id", "group_type", "group"), area="feature × group",
    docs={"n_athletes": "tracked athletes of the group (≥ 1 eligible forty sequence) — the denominator",
          "n_athletes_repeat": "athletes with ≥ 2 attempts carrying a value"})
CHARACTERISTICS = TableContract(
    "atlas_characteristics", "Standardised characterisation of every feature (all athletes, and within each position).",
    {"feature_id": S, "scope": S, "family": S, "design": S, "source": S, "n_athletes": I, "n_athletes_value": I,
     "n_attempts": I, "n_attempts_value": I, "n_athletes_repeat": I, "coverage_athletes": F, "coverage_attempts": F,
     "mean": F, "sd": F, "p05": F, "p50": F, "p95": F, "implausible_share": F, "sd_within_position": F,
     "cv_within_position_pct": F, "reliability": F, "reliability_lower": F, "reliability_upper": F, "reliability_method": S,
     "sem": F, "reliability_of_mean": F, "repeat_diff_mean": F, "repeat_diff_sd": F, "repeat_spearman": F,
     "reliability_s2": F, "reliability_s3": F, "repeat_retained_s3": F, "reliability_cohort_min": F,
     "reliability_cohort_max": F, "window_r2": F, "window_r2_lower": F, "window_r2_upper": F, "cohort_r2": F,
     "rho_forty": F, "rho_forty_lower": F, "rho_forty_upper": F, "rho_ten": F, "rho_ten_lower": F, "rho_ten_upper": F,
     "redundancy_abs": F, "redundancy_lower": F, "redundancy_upper": F, "r2_stopwatch": F, "r2_stopwatch_lower": F,
     "r2_stopwatch_upper": F, "r2_stopwatch_cohort_adj": F, "r2_body": F, "reliable_unique_share": F, "retrieval_auc": F,
     "retrieval_auc_lower": F, "retrieval_auc_upper": F, "residual_auc": F, "residual_auc_lower": F, "residual_auc_upper": F,
     "forty_from_feature_r2": F, "cohort_probe_gain": F, "note": S}, key=("feature_id", "scope"), area="feature × scope",
    docs={"scope": "all = every tracked athlete, statistics adjusted for position · position:<group> = inside one group",
          "reliability": "single-attempt reliability by the estimator named in reliability_method",
          "sem": "standard error of measurement = √ within-athlete variance, in the unit of the feature",
          "repeat_diff_mean": "attempt 2 − attempt 1 (first two attempts in time)",
          "repeat_spearman": "rank agreement of attempt 1 and attempt 2, within position",
          "reliability_s3": "same ICC on attempts passing the signal and no-boundary-peak quality filters",
          "repeat_retained_s3": "share of the athletes with a repeat who keep one under that filter",
          "window_r2": "share of the attempt-to-attempt change explained by the change of first-frame speed, recorded "
                       "duration and frames after the peak (an athlete does not change between attempts; his recording does)",
          "cohort_r2": "share of the athlete-level variance explained by draft cohort, within position",
          "redundancy_abs": "max(|ρ forty|, |ρ 10-yard|), Spearman within position, athlete level",
          "r2_stopwatch": "share of the athlete-level variance explained by forty + 10-yard together, within position",
          "r2_stopwatch_cohort_adj": "same within position × cohort",
          "reliable_unique_share": "max(reliability of the athlete mean − r2_stopwatch, 0): reproducible variance the "
                                   "official times do not explain (a lower bound, sprint rescue definition)",
          "retrieval_auc": "vector features: confirm-role repeat-attempt discrimination (notebook 03)",
          "forty_from_feature_r2": "vector features: held-out share of the official forty the vector explains beyond "
                                   "position — the reverse direction of r2_stopwatch",
          "cohort_probe_gain": "vector features: held-out cohort accuracy gain over position (notebook 03)"})
RELIABILITY = TableContract(
    "atlas_reliability", "Reliability estimates by feature and scope (position, cohort, quality filter).",
    {"feature_id": S, "scope": S, "n_athletes_repeat": I, "n_attempts": I, "icc": F, "ci_lower": F, "ci_upper": F, "sem": F,
     "reliability_of_mean": F, "method": S}, key=("feature_id", "scope"), area="feature × scope",
    docs={"scope": "position_adjusted · position:<group> · cohort:<year> (within position) · scenario:<quality filter>",
          "ci_lower": "athlete bootstrap, 95 %; position_adjusted and position scopes only"})
NUISANCE = TableContract(
    "atlas_nuisance_diagnostics", "Association of every attempt feature with the recording window, noise and cohort.",
    {"feature_id": S, "variable": S, "level": S, "n_athletes": I, "spearman": F, "spearman_lower": F, "spearman_upper": F,
     "r2": F, "r2_lower": F, "r2_upper": F}, key=("feature_id", "variable", "level"), area="feature × variable × level",
    docs={"level": "within_athlete = attempt 2 − attempt 1 of the feature against attempt 2 − attempt 1 of the variable · "
                   "between_athletes = athlete means, within position",
          "variable": "window_joint = first_speed + duration_s + frames_after_peak together (r2) · draft_cohort (r2)"})
MATRIX = TableContract(
    "atlas_redundancy_matrix", "Within-position Spearman correlation between athlete-level features.",
    {"feature_a": S, "feature_b": S, "n_athletes": I, "rho": F}, key=("feature_a", "feature_b"), area="feature pair")
CLUSTERS = TableContract(
    "atlas_feature_clusters", "Groups of features that carry the same athlete ordering (average linkage on 1 − |ρ|).",
    {"feature_id": S, "cluster": I, "cluster_size": I, "representative": S, "display_order": I, "mean_abs_rho_in_cluster": F},
    key=("feature_id",), area="feature",
    docs={"representative": "member of the cluster with the highest reliability (ties: first by name)"})
STATUS = TableContract(
    "atlas_feature_status", "Screening registry: rule-based labels and grade of every feature, with the evidence used.",
    {"feature_id": S, "family": S, "design": S, "role": S, "definition": S, "support": S, "reliability_label": S,
     "stopwatch_relation": S, "acquisition": S, "grade": S, "transfer_eligibility": S, "reason": S, "n_athletes_repeat": I,
     "coverage_athletes": F, "reliability": F, "reliability_lower": F, "reliability_upper": F, "redundancy_abs": F,
     "r2_stopwatch": F, "reliable_unique_share": F, "window_r2": F, "cohort_r2": F}, key=("feature_id",), area="feature",
    docs={"grade": " · ".join(GRADES) + " — heuristic screening label, for review",
          "stopwatch_relation": "is_reference · redundant · beyond_stopwatch · low_correlation_only (NOT evidence of new "
                                "information) · uncertain · pending_04b",
          "transfer_eligibility": "reference_baseline · candidate · control · candidate_unproven_distinctness · "
                                  "exploratory_only · not_eligible — no NFL value enters this label"})
RETENTION = TableContract(
    "atlas_quality_retention", "Athletes and attempts kept by each nested quality filter.",
    {"scenario": S, "group_type": S, "group": S, "n_attempts": I, "n_athletes": I, "n_athletes_repeat": I,
     "share_athletes": F, "share_repeat": F}, key=("scenario", "group_type", "group"), area="quality filter × group")
CHAR_CONTRACTS = (COVERAGE, CHARACTERISTICS, RELIABILITY, NUISANCE, ATTEMPT_VALUES, ATHLETE_VALUES)


# ============================================================================== sustained acceleration (F6)
def sustained_attempt(t: np.ndarray, v: np.ndarray, d: np.ndarray, c: AtlasConfig) -> dict[str, tuple[float, str]]:
    """name → (value, reason) for one attempt. ``t`` [s], ``v`` [yd/s], ``d`` [yd of path], recorded frames in order.
    Every value is an interpolation BETWEEN recorded frames; nothing before the first or after the last frame is used."""
    names = [s[0] for s in sustained_specs(c)]
    ok = np.isfinite(t) & np.isfinite(v) & np.isfinite(d)
    t, v, d = t[ok], v[ok], d[ok]
    if t.size < 3:
        return {n: (NAN, "too_few_frames") for n in names}
    t0 = first_crossing(t, v, c.anchor_speed)
    if not np.isfinite(t0):
        why = "start_above_anchor" if v[0] >= c.anchor_speed else "anchor_not_reached"
        return {n: (NAN, why) for n in names}
    out: dict[str, tuple[float, str]] = {}
    for name, w in zip(names, c.accel_windows_s):
        end = t0 + w
        out[name] = ((float(np.interp(end, t, v)) - c.anchor_speed) / w, "") if end <= t[-1] + 1e-9 else (NAN, "window_exceeds_recording")
    d_end = float(np.interp(t0, t, d)) + c.gain_distance_yd
    gain = names[len(c.accel_windows_s)]
    after = t >= t0
    if d_end <= d[-1] + 1e-9 and np.all(np.diff(d[after]) >= 0):
        out[gain] = (float(np.interp(d_end, d[after], v[after])) - c.anchor_speed, "")
    else:
        out[gain] = (NAN, "window_exceeds_recording")
    t1 = first_crossing(t, v, c.upper_speed)
    out[names[-1]] = (float(t1 - t0), "") if np.isfinite(t1) and t1 > t0 else (NAN, "upper_threshold_not_reached")
    return out


def sustained_table(ss: SequenceSet, c: AtlasConfig) -> pl.DataFrame:
    rows = []
    for i in range(ss.n):
        got = sustained_attempt(ss.series(i, "t_rel"), ss.series(i, "speed"), ss.series(i, "dist"), c)
        rows += [{"seq_id": str(ss.seq_ids[i]), "nfl_id": int(ss.nfl_ids[i]), "feature": n, "value": x if np.isfinite(x) else None,
                  "missing_reason": why} for n, (x, why) in got.items()]
    return pl.DataFrame(rows, schema=SUSTAINED.columns).sort("seq_id", "feature")


# ============================================================================== persisted upstream evidence (read only)
def read_registry(cfg: Any) -> pl.DataFrame | None:
    """Phase 04 Study V registry (feature × drill status), or None when it was not produced."""
    p = Path(cfg.paths.tables) / "phase_04" / "reliability" / "V_feature_validation_registry.csv"
    if not p.is_file():
        return None
    df = pl.read_csv(p, infer_schema_length=20000)
    return df.select(*[pl.col(c).cast(pl.String) for c in ("domain", "stratum", "drill", "feature", "family", "unit", "status", "status_reason")],
                     pl.col("n_athletes").cast(pl.Int64, strict=False), pl.col("value_repeatability").cast(pl.Float64, strict=False))


def read_catalog(cfg: Any) -> dict[str, str]:
    """Phase 04 feature catalog: feature → definition (empty when absent)."""
    p = Path(cfg.paths.tables) / "phase_04" / "descriptive" / "B_feature_catalog.csv"
    return dict(pl.read_csv(p).select("feature", "definition").iter_rows()) if p.is_file() else {}


def official_table(mart: pl.DataFrame, athletes: pl.DataFrame) -> pl.DataFrame:
    """Official measurements of the registry athletes, one row each, as numbers. Only the columns of
    ``OFFICIAL_SPECS`` are read from the mart; a column that does not exist or does not parse is null."""
    cols = [(pl.col(c).cast(pl.Float64, strict=False) if c in mart.columns else pl.lit(None, dtype=pl.Float64)).alias(c)
            for c, *_ in OFFICIAL_SPECS]
    m = mart.select(pl.col("nfl_id").cast(pl.Int64), *cols).unique(subset=["nfl_id"], keep="first")
    return athletes.select("nfl_id").join(m, on="nfl_id", how="left").sort("nfl_id")


def reuse_inventory(store: Any, cfg: Any) -> pl.DataFrame:
    """Static inventory: which persisted outputs this milestone reuses, whether they are present in this run mode,
    and what is genuinely new. ``store``: ``AtlasStore``."""
    from . import checkpoint as ck
    from . import data as dt
    from . import repr_eval as rv
    from . import sequences as sq

    up, gold, tables = store.upstream, Path(cfg.paths.gold), Path(cfg.paths.tables)
    def have(c: TableContract) -> bool:
        return up.exists(c)
    rows = [
        ("F1 official Combine", "gold.mart_athlete_research_base", True, "reused", "official results (12 columns), cast to numbers"),
        ("F2 / F4 / F6 forty attempt features", "sprint_rescue.attempt_censoring_diagnostics",
         (gold / "sprint_rescue" / "attempt_censoring_diagnostics.parquet").is_file(), "reused",
         "speed, acceleration, speed-band and sprint-model values per attempt; fit validity flags"),
        ("F2 other drills / F3 events", "phase_04 V_feature_validation_registry",
         (tables / "phase_04" / "reliability" / "V_feature_validation_registry.csv").is_file(), "reused (evidence only)",
         "status and repeatability per feature × drill; nothing recomputed"),
        ("F4 sprint model", "phase_03 sprint fit, as carried by the sprint rescue", True, "reused", "no second fitter is written"),
        ("F5 functional PCA", "fpca_component_gate / fpca_athlete_scores", have(fe.GATE), "reused", "component evidence of notebook 02"),
        ("F5 encoders", "repr_method_summary / repr_trace_reliability / *_attempt_embeddings", have(rv.SUMMARY), "reused",
         "TCN-A, TCN-B (8-d) and the 16-d sensitivity of notebook 03"),
        ("F6 sustained acceleration", "accel_sustained, accel_initial, accel_v2_5 / v4_7 / v2_7 (sprint rescue)", True, "reused",
         "best 0.5 s mean, first second after onset, mean acceleration between two speeds"),
        ("F6 sustained acceleration", "atlas_sustained (this notebook)", have(sq.SEQ_FRAMES), "NEW (4 features)",
         "fixed 0.5 s and 1.0 s windows, speed gain over a distance, time between two speeds — all anchored on a recorded speed"),
        ("F7 acquisition", "sequence_window_flags", have(ck.WINDOWS), "reused", "recording window, noise and quality flags of notebook 02"),
        ("cohort and folds", "athlete_registry / split_registry", have(dt.ATHLETES) and have(dt.SPLITS), "reused", "unchanged"),
        ("characterisation engine", "atlas.py (this milestone)", True, "NEW", "one standard table over all designs; screening registry"),
    ]
    return pl.DataFrame([{"family": a, "artifact": b, "present": bool(c), "status": d, "use": e} for a, b, c, d, e in rows])


# ============================================================================== feature dictionary
def _entry(fid: str, name: str, family: str, design_: str, role: str, **kw: Any) -> dict[str, Any]:
    base = {"feature_id": fid, "feature_name": name, "family": family, "design": design_, "role": role,
            "drill": "FORTY_YARD_DASH", "position_applicability": "all tracked position groups", "grain": "attempt",
            "units": "", "definition": "", "source_module": "", "source_table": "", "source_column": name, "provenance": "",
            "smoothing": "", "time_or_distance_window": "whole recording", "measurement_status": "measured",
            "interpretation_status": "interpretable", "quality_flags": "", "plausible_lo": None, "plausible_hi": None,
            "version": f"atlas {ATLAS_VERSION}"}
    unknown = set(kw) - set(base)
    if unknown:
        raise KeyError(f"unknown dictionary field(s) {sorted(unknown)}")
    return {**base, **kw}


def build_dictionary(c: AtlasConfig, rcfg: RescueConfig | None = None, registry: pl.DataFrame | None = None,
                     catalog: dict[str, str] | None = None, fpca_gate: pl.DataFrame | None = None,
                     methods: pl.DataFrame | None = None) -> pl.DataFrame:
    """The dictionary, assembled from the configurations and registries that define the features (never typed twice):
    ``sprint_rescue.config.feature_specs``, the Phase 04 registry and catalog, the FPCA gate and the method table of
    notebook 03. A (drill, feature) of the registry that this atlas characterises itself is not listed a second time."""
    rcfg, catalog = rcfg or RescueConfig(), catalog or {}
    p04 = {}
    if registry is not None:
        own = registry.filter((pl.col("domain") == "combine") & (pl.col("drill") == rcfg.drill) & (pl.col("stratum") == rcfg.drill))
        p04 = dict(own.select("feature", "status").iter_rows())
    rows = []
    for col, name, unit, definition, lo, hi in OFFICIAL_SPECS:
        rows.append(_entry(f"F1.official.{col}", name, "F1", "static_athlete", "reference", drill="official Combine test",
                           grain="athlete", units=unit, definition=definition, source_module="nflbdb.data (research mart)",
                           source_table=MART_TABLE, source_column=col, provenance="official Combine results, one value per athlete",
                           time_or_distance_window="official test", plausible_lo=lo, plausible_hi=hi,
                           quality_flags="single official value: repeatability cannot be estimated here"))
    family_of = {"R1": "F2", "R2": "F6", "R3": "F4", "window": "F7"}
    for s in feature_specs(rcfg):
        fam = family_of[s.representation]
        lo, hi = PLAUSIBLE.get(s.unit, (None, None))
        flags = [f"phase04:{p04[s.name]}"] if s.name in p04 else []
        interp = "interpretable"
        if s.name in MODEL_COLUMNS:
            flags.append("requires fit_ok; a high curve-fit R² does not make the parameter identifiable")
            interp = "extrapolated to the model start (never observed)" if s.name == "model_a0" else "interpretable when identifiable"
        rows.append(_entry(f"{fam}.forty.{s.name}", s.name, fam, "model_parameter" if fam == "F4" else "repeated_attempt",
                           "nuisance" if fam == "F7" else "candidate", units=s.unit,
                           definition=catalog.get(s.name) or s.meaning, source_module="nflbdb.research.sprint_rescue",
                           source_table=ATTEMPT_TABLE, provenance=s.source, smoothing=rcfg.smoothing,
                           time_or_distance_window=("speed band" if s.name.startswith("accel_v") else "fit window" if fam == "F4"
                                                    else "whole recording"),
                           measurement_status="model-based" if fam == "F4" else "derived", interpretation_status=interp,
                           quality_flags="; ".join(flags), plausible_lo=lo, plausible_hi=hi))
    for name, fam, design_, role, unit, definition in EXTRA_ATTEMPT:
        lo, hi = PLAUSIBLE.get(unit, (None, None)) if role == "candidate" else (None, None)
        rows.append(_entry(f"{fam}.forty.{name}", name, fam, design_, role, units=unit, definition=catalog.get(name) or definition,
                           source_module="nflbdb.research.sprint_rescue", source_table=ATTEMPT_TABLE,
                           provenance="sprint rescue attempt table", smoothing=rcfg.smoothing,
                           measurement_status="model-based" if fam == "F4" else "derived",
                           interpretation_status="diagnostic only" if role == "diagnostic" else "interpretable",
                           quality_flags=f"phase04:{p04[name]}" if name in p04 else "", plausible_lo=lo, plausible_hi=hi))
    for name, unit, definition, window in sustained_specs(c):
        lo, hi = PLAUSIBLE.get(unit, (None, None))
        rows.append(_entry(f"F6.forty.{name}", name, "F6", "repeated_attempt", "candidate", units=unit, definition=definition,
                           source_module="nflbdb.research.advanced_movement.atlas", source_table=SUSTAINED_TABLE,
                           provenance="sequence_frames (notebook 01), interpolation between recorded frames", smoothing=rcfg.smoothing,
                           time_or_distance_window=window, measurement_status="derived",
                           quality_flags="missing when the anchor, the window or the upper speed is not inside the recording",
                           plausible_lo=lo, plausible_hi=hi))
    for name, unit, definition in WINDOW_SPECS:
        rows.append(_entry(f"F7.forty.{name}", name, "F7", "repeated_attempt", "nuisance", units=unit, definition=definition,
                           source_module="nflbdb.research.advanced_movement.checkpoint", source_table=WINDOW_TABLE,
                           provenance="sequence_window_flags (notebook 02)", measurement_status="measured",
                           interpretation_status="describes the recording, not the athlete"))
    if fpca_gate is not None:
        for g in fpca_gate.sort("representation", "component").iter_rows(named=True):
            rows.append(_entry(f"F5.fpca.{g['representation']}.pc{g['component']}", f"{g['representation']} component {g['component']}",
                               "F5", "functional_score", "candidate", units="score", grain="attempt (out-of-fold score)",
                               definition=f"functional PCA score {g['component']} of {g['representation']} at K = {g['k']}",
                               source_module="nflbdb.research.advanced_movement.fpca", source_table=GATE_TABLE,
                               source_column=f"{g['representation']}|{g['component']}", provenance="notebook 02, inductive folds",
                               time_or_distance_window="support of the fold", measurement_status="learned",
                               interpretation_status="mode of variation; no physical interpretation established",
                               quality_flags=f"gate:{g['status']}"))
    if methods is not None:
        for m in methods.sort("method").iter_rows(named=True):
            tcn_ = m["family"] == "tcn"
            rows.append(_entry(f"F5.vector.{m['method']}", m["label"], "F5", "neural_embedding" if tcn_ else "functional_score",
                               "candidate", units=f"{m['embedding_dim']}-dimensional vector", grain="attempt (vector)",
                               definition=f"{m['objective']} representation of {m['representation']}; evaluated as a whole — "
                                          "a single coordinate is not a feature",
                               source_module="nflbdb.research.advanced_movement." + ("tcn" if tcn_ else "fpca"),
                               source_table=VECTOR_TABLE, source_column=m["method"], provenance="notebook 03, inductive folds",
                               time_or_distance_window="support of the fold", measurement_status="learned",
                               interpretation_status="no physical interpretation established",
                               quality_flags="" if m["primary"] else "sensitivity run (never selected)"))
    if registry is not None:
        done = {r["feature_name"] for r in rows if r["source_table"] == ATTEMPT_TABLE}
        reg = registry.filter(pl.col("domain") == "combine").sort("stratum", "drill", "feature")
        for r in reg.iter_rows(named=True):
            if r["drill"] == rcfg.drill and r["stratum"] == rcfg.drill and r["feature"] in done:
                continue
            event = (r["family"] or "").startswith("event")
            fam = "F3" if event else "F2"
            rows.append(_entry(f"{fam}.{r['stratum']}:{r['drill']}.{r['feature']}", r["feature"], fam,
                               "event_rate" if event else "repeated_attempt", "descriptive", drill=r["drill"],
                               position_applicability=r["stratum"], grain="athlete × drill (Phase 04)", units=r["unit"] or "",
                               definition=catalog.get(r["feature"], ""), source_module="nflbdb.research.phase04",
                               source_table=REGISTRY_TABLE, source_column=f"{r['stratum']}|{r['drill']}|{r['feature']}",
                               provenance="Phase 04 Study V registry (rule-based status)", smoothing=rcfg.smoothing,
                               measurement_status="reused evidence only",
                               interpretation_status="unvalidated event definition" if event else "interpretable",
                               quality_flags=f"phase04:{r['status']}"))
    return pl.DataFrame(rows, schema=DICTIONARY.columns).sort("feature_id")


def assert_unique_keys(dictionary: pl.DataFrame) -> None:
    dup = dictionary.group_by("feature_id").len().filter(pl.col("len") > 1)
    if dup.height:
        raise ValueError(f"{dup.height} duplicated feature_id(s) in the dictionary: {dup['feature_id'].head(5).to_list()}")
    same = dictionary.filter(pl.col("source_table").is_in([ATTEMPT_TABLE, WINDOW_TABLE, SUSTAINED_TABLE, MART_TABLE]))
    clash = same.group_by("source_table", "source_column").len().filter(pl.col("len") > 1)
    if clash.height:
        raise ValueError(f"{clash.height} source column(s) are listed under two feature ids: {clash.to_dicts()[:3]}")


# ============================================================================== data in array form
@dataclass
class AthleteData:
    """Tracked athletes (≥ 1 eligible forty sequence), one row each."""
    ids: np.ndarray
    position: np.ndarray
    cohort: np.ndarray
    official: dict[str, np.ndarray]
    values: dict[str, np.ndarray]                 # feature_id → official value per athlete

    def rows(self, nfl_ids: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.ids, nfl_ids)
        if np.any(idx >= self.ids.size) or np.any(self.ids[np.minimum(idx, self.ids.size - 1)] != nfl_ids):
            raise KeyError("an attempt belongs to an athlete absent from the athlete table")
        return idx


@dataclass
class AttemptData:
    """Eligible forty attempts, one row each, with their recording-window variables and quality filters."""
    seq_ids: np.ndarray
    nfl_ids: np.ndarray
    position: np.ndarray
    cohort: np.ndarray
    order: np.ndarray
    window: dict[str, np.ndarray]
    scenario: dict[str, np.ndarray]
    values: dict[str, np.ndarray]                 # feature_id → value per attempt (NaN = missing)
    reasons: dict[str, np.ndarray]                # feature_id → reason per attempt ("" = present)


def athlete_data(athletes: pl.DataFrame, official: pl.DataFrame, dictionary: pl.DataFrame) -> AthleteData:
    """``athletes``: ``data.ATHLETES`` · ``official``: :func:`official_table`. Only identity, position, cohort and
    official measurements are read."""
    a = athletes.filter(pl.col("n_primary_eligible") >= 1).select("nfl_id", "position_group", "draft_year").sort("nfl_id")
    a = a.join(official, on="nfl_id", how="left")
    num = lambda col: a[col].cast(pl.Float64, strict=False).fill_null(NAN).to_numpy()            # noqa: E731
    static = dictionary.filter(pl.col("source_table") == MART_TABLE)
    return AthleteData(a["nfl_id"].to_numpy().astype(np.int64), a["position_group"].fill_null("?").to_numpy().astype(str),
                       a["draft_year"].cast(pl.String).fill_null("?").to_numpy().astype(str),
                       {c: num(c) for c in (FORTY, TEN, WEIGHT, HEIGHT)},
                       {r["feature_id"]: num(r["source_column"]) for r in static.iter_rows(named=True)})


def attempt_data(windows: pl.DataFrame, attempts: pl.DataFrame, sustained: pl.DataFrame, dictionary: pl.DataFrame) -> AttemptData:
    """``windows``: ``checkpoint.WINDOWS`` (defines the attempts) · ``attempts``: sprint rescue attempt table ·
    ``sustained``: :func:`sustained_table`. Attempts are those of ``windows``; a value absent upstream stays missing."""
    w = windows.sort("seq_id")
    att = attempts.with_columns((pl.col("event_id") + "|" + pl.col("nfl_id").cast(pl.String) + "|"
                                 + pl.col("attempt").cast(pl.String)).alias("seq_id")).drop("nfl_id")
    j = w.select("seq_id").join(att, on="seq_id", how="left", maintain_order="left")
    sus = w.select("seq_id").join(sustained.pivot(on="feature", index="seq_id", values="value"), on="seq_id", how="left",
                                  maintain_order="left")
    why = w.select("seq_id").join(sustained.pivot(on="feature", index="seq_id", values="missing_reason"), on="seq_id", how="left",
                                  maintain_order="left")
    num = lambda frame, col: frame[col].cast(pl.Float64, strict=False).fill_null(NAN).to_numpy()   # noqa: E731
    in_rescue = j["event_id"].is_not_null().to_numpy()
    fit_ok = j["fit_ok"].fill_null(False).to_numpy()
    band_censored = j["band_start_censored"].fill_null(False).to_numpy()
    values, reasons = {}, {}
    for r in dictionary.filter(pl.col("source_table").is_in([ATTEMPT_TABLE, WINDOW_TABLE, SUSTAINED_TABLE])).iter_rows(named=True):
        fid, col = r["feature_id"], r["source_column"]
        if r["source_table"] == WINDOW_TABLE:
            x, default = num(w, col), np.full(w.height, "missing_upstream")
        elif r["source_table"] == SUSTAINED_TABLE:
            x = num(sus, col) if col in sus.columns else np.full(w.height, NAN)
            default = (why[col].fill_null("sequence_not_processed").to_numpy().astype(str) if col in why.columns
                       else np.full(w.height, "sequence_not_processed"))
        else:
            x = num(j, col)
            if col in MODEL_COLUMNS:
                x = np.where(fit_ok, x, NAN)
            default = (np.where(band_censored, "recording_starts_above_band", "band_not_reached") if col.startswith("accel_v")
                       else np.full(w.height, "sprint_fit_not_ok" if r["design"] == "model_parameter" else "missing_upstream"))
            default = np.where(in_rescue, default, "attempt_not_in_sprint_rescue")
        values[fid] = x
        reasons[fid] = np.where(np.isfinite(x), "", np.where(np.asarray(default).astype(str) == "", "missing_upstream", default)).astype(str)
    return AttemptData(w["seq_id"].to_numpy().astype(str), w["nfl_id"].to_numpy().astype(np.int64),
                       w["position_group"].fill_null("?").to_numpy().astype(str),
                       w["draft_year"].cast(pl.String).fill_null("?").to_numpy().astype(str), w["attempt_order"].fill_null(0).to_numpy(),
                       {c: num(w, c) for c in (*fe.WINDOW_VARS, *fe.FLAG_VARS)},
                       {s: w[s].fill_null(False).to_numpy() for s in ("S2_signal", "S3_no_boundary")}, values, reasons)


# ============================================================================== characterisation engine
def scopes(position: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [("all", np.ones(position.size, dtype=bool))] + [(f"position:{p}", position == p)
                                                             for p in sorted(set(position.tolist())) if p != "?"]


def icc_estimate(x: np.ndarray, athletes: np.ndarray, strata: np.ndarray | None, n_boot: int, seed: int, min_athletes: int
                 ) -> dict[str, Any]:
    """Stratified ICC(1) with an athlete bootstrap (``fpca_eval._icc``). ``strata`` None = one stratum."""
    return fe._icc(np.asarray(x, float), np.asarray(athletes), np.full(len(x), "") if strata is None else np.asarray(strata).astype(str),
                   n_boot, seed, min_athletes)


def _distribution(x: np.ndarray, lo: float | None, hi: float | None) -> dict[str, Any]:
    x = x[np.isfinite(x)]
    if not x.size:
        return {}
    q = np.quantile(x, [0.05, 0.5, 0.95])
    out = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else None, "p05": float(q[0]), "p50": float(q[1]),
           "p95": float(q[2])}
    if lo is not None and hi is not None:
        out["implausible_share"] = float(((x < lo) | (x > hi)).mean())
    return out


def _athlete_level(mean: np.ndarray, ar: np.ndarray, a: AthleteData, full: bool, c: AtlasConfig, seed: int, fid: str,
                   skip: Sequence[str] = ()) -> dict[str, Any]:
    """Athlete-level association of one feature with the official times, body size and cohort. ``full`` = all
    athletes (statistics within position, intervals); otherwise one position (no dummy, no bootstrap)."""
    pos, coh = a.position[ar], a.cohort[ar]
    cats = [pos] if full else []
    out: dict[str, Any] = {}
    ok = np.isfinite(mean)
    if ok.sum() >= 5:
        res = residualise(mean[ok], design(int(ok.sum()), [x[ok] for x in cats]))
        sd = float(res.std(ddof=1))
        out["sd_within_position"] = sd
        out["cv_within_position_pct"] = float(100.0 * sd / abs(mean[ok].mean())) if abs(mean[ok].mean()) > 1e-12 else None
    official = {t: a.official[t][ar] for t in (FORTY, TEN) if t not in skip}
    tops = []
    for t, name in ((FORTY, "rho_forty"), (TEN, "rho_ten")):
        if t in official:
            r = partial_corr(mean, official[t], cats, (), "spearman")
            out.update({name: r["r"], f"{name}_lower": r["ci_lower"], f"{name}_upper": r["ci_upper"]})
            if np.isfinite(r["r"]):
                tops.append(r)
    if tops:
        top = max(tops, key=lambda r: abs(r["r"]))
        lo, hi = fe._abs_interval(top["ci_lower"], top["ci_upper"])
        out.update(redundancy_abs=abs(top["r"]), redundancy_lower=lo, redundancy_upper=hi)
    if official:
        X = list(official.values())
        r2, lo, hi, _ = fe._r2_ci(mean, X, cats, c.n_boot_r2 if full else 0, seed_for(seed, "stopwatch", fid))
        out.update(r2_stopwatch=r2, r2_stopwatch_lower=lo, r2_stopwatch_upper=hi)
        if full:
            out["r2_stopwatch_cohort_adj"] = fe._r2_ci(mean, X, [fe._join(pos, coh)], 0, 0)[0]
    body = [a.official[t][ar] for t in (WEIGHT, HEIGHT) if t not in skip]
    if body:
        out["r2_body"] = fe._r2_ci(mean, body, cats, 0, 0)[0]
    levels = sorted(set(coh.tolist()))
    if len(levels) > 1:
        out["cohort_r2"] = fe._r2_ci(mean, [(coh == y).astype(float) for y in levels[1:]], cats, 0, 0)[0]
    return out


def _repeated(row: dict[str, Any], d: AttemptData, a: AthleteData, c: AtlasConfig, seed: int
              ) -> tuple[list[dict], list[dict], list[dict]]:
    fid, v = row["feature_id"], d.values[row["feature_id"]]
    is_window = row["source_table"] == WINDOW_TABLE and row["source_column"] in JOINT_WINDOW
    chars, rels, nuis = [], [], []
    for scope, keep in scopes(d.position):
        full = scope == "all"
        x, nfl, pos, coh, order = v[keep], d.nfl_ids[keep], d.position[keep], d.cohort[keep], d.order[keep]
        ok = np.isfinite(x)
        ids_all = np.unique(nfl)
        strata = pos if full else None
        icc = icc_estimate(x, nfl, strata, c.n_boot, seed_for(seed, "icc", fid, scope), c.min_athletes)
        rels.append({"feature_id": fid, "scope": "position_adjusted" if full else scope, "n_athletes_repeat": icc["n_athletes_repeat"],
                     "n_attempts": icc["n_sequences"], "icc": icc["icc"], "ci_lower": icc["ci_lower"], "ci_upper": icc["ci_upper"],
                     "sem": icc["typical_error"], "reliability_of_mean": icc["reliability_of_mean"], "method": ICC_METHOD})
        ch: dict[str, Any] = {
            "feature_id": fid, "scope": scope, "family": row["family"], "design": row["design"], "source": "computed (notebook 04A)",
            "n_athletes": int(ids_all.size), "n_athletes_value": int(np.unique(nfl[ok]).size), "n_attempts": int(x.size),
            "n_attempts_value": int(ok.sum()), "n_athletes_repeat": icc["n_athletes_repeat"],
            "coverage_athletes": float(np.unique(nfl[ok]).size / ids_all.size) if ids_all.size else None,
            "coverage_attempts": float(ok.mean()) if x.size else None, **_distribution(x, row["plausible_lo"], row["plausible_hi"]),
            "reliability": icc["icc"], "reliability_lower": icc["ci_lower"], "reliability_upper": icc["ci_upper"],
            "reliability_method": ICC_METHOD if full else "ICC(1) inside the position, athlete bootstrap", "sem": icc["typical_error"],
            "reliability_of_mean": icc["reliability_of_mean"]}
        xo, no, po = x[ok], nfl[ok], pos[ok]
        ids2, i1, i2 = first_two(no, order[ok])
        d21 = xo[i2] - xo[i1]
        if ids2.size >= 5:
            ch.update(repeat_diff_mean=float(d21.mean()), repeat_diff_sd=float(d21.std(ddof=1)),
                      repeat_spearman=partial_corr(xo[i1], xo[i2], [po[i1]] if full else [], (), "spearman")["r"])
        W = {w: d.window[w][keep][ok] for w in NUISANCE_VARS}
        if ids2.size >= 5 and not is_window:
            r2, lo, hi, n = fe._r2_ci(d21, [W[w][i2] - W[w][i1] for w in JOINT_WINDOW], [], c.n_boot_r2 if full else 0,
                                      seed_for(seed, "window", fid))
            ch.update(window_r2=r2, window_r2_lower=lo, window_r2_upper=hi)
            if full:
                nuis.append({"feature_id": fid, "variable": "window_joint", "level": "within_athlete", "n_athletes": n, "r2": r2,
                             "r2_lower": lo, "r2_upper": hi})
        ids, mean, _ = athlete_means(nfl, x)
        mean, ar = mean[:, 0], a.rows(ids)
        ch.update(_athlete_level(mean, ar, a, full, c, seed, fid))
        if icc["reliability_of_mean"] is not None and ch.get("r2_stopwatch") is not None and np.isfinite(ch["r2_stopwatch"]):
            ch["reliable_unique_share"] = max(icc["reliability_of_mean"] - ch["r2_stopwatch"], 0.0)
        if full:
            repeat0 = icc["n_athletes_repeat"]
            for s, flag in d.scenario.items():
                m = flag[keep]
                r = icc_estimate(x[m], nfl[m], pos[m], 0, 0, c.min_athletes)
                rels.append({"feature_id": fid, "scope": f"scenario:{s}", "n_athletes_repeat": r["n_athletes_repeat"],
                             "n_attempts": r["n_sequences"], "icc": r["icc"], "sem": r["typical_error"],
                             "reliability_of_mean": r["reliability_of_mean"], "method": ICC_METHOD})
                ch["reliability_s2" if s == "S2_signal" else "reliability_s3"] = r["icc"]
                if s == "S3_no_boundary" and repeat0:
                    ch["repeat_retained_s3"] = r["n_athletes_repeat"] / repeat0
            by_cohort = []
            for y in sorted(set(coh.tolist())):
                m = coh == y
                r = icc_estimate(x[m], nfl[m], pos[m], 0, 0, c.min_athletes)
                rels.append({"feature_id": fid, "scope": f"cohort:{y}", "n_athletes_repeat": r["n_athletes_repeat"],
                             "n_attempts": r["n_sequences"], "icc": r["icc"], "sem": r["typical_error"],
                             "reliability_of_mean": r["reliability_of_mean"], "method": ICC_METHOD})
                if r["icc"] is not None:
                    by_cohort.append(r["icc"])
            if by_cohort:
                ch.update(reliability_cohort_min=min(by_cohort), reliability_cohort_max=max(by_cohort))
            apos = a.position[ar]
            for w in NUISANCE_VARS:
                if row["source_table"] == WINDOW_TABLE and row["source_column"] == w:
                    continue
                if ids2.size >= 5:
                    r = partial_corr(d21, W[w][i2] - W[w][i1], (), (), "spearman")
                    nuis.append({"feature_id": fid, "variable": w, "level": "within_athlete", "n_athletes": r["n"], "spearman": r["r"],
                                 "spearman_lower": r["ci_lower"], "spearman_upper": r["ci_upper"]})
                _, wm, _ = athlete_means(nfl, d.window[w][keep])
                r = partial_corr(mean, wm[:, 0], [apos], (), "spearman")
                nuis.append({"feature_id": fid, "variable": w, "level": "between_athletes", "n_athletes": r["n"], "spearman": r["r"],
                             "spearman_lower": r["ci_lower"], "spearman_upper": r["ci_upper"]})
            nuis.append({"feature_id": fid, "variable": "draft_cohort", "level": "between_athletes",
                         "n_athletes": int(np.isfinite(mean).sum()), "r2": ch.get("cohort_r2")})
        chars.append(ch)
    return chars, rels, nuis


def _static(row: dict[str, Any], a: AthleteData, c: AtlasConfig, seed: int) -> list[dict]:
    fid, v, col = row["feature_id"], a.values[row["feature_id"]], row["source_column"]
    out = []
    for scope, keep in scopes(a.position):
        full = scope == "all"
        x = v[keep]
        ok = np.isfinite(x)
        ch = {"feature_id": fid, "scope": scope, "family": row["family"], "design": row["design"], "source": "computed (notebook 04A)",
              "n_athletes": int(x.size), "n_athletes_value": int(ok.sum()), "coverage_athletes": float(ok.mean()) if x.size else None,
              **_distribution(x, row["plausible_lo"], row["plausible_hi"]),
              "reliability_method": "not estimable: one official value per athlete in this dataset",
              "note": "reference measurement of the stopwatch comparison" if col in (FORTY, TEN) else None}
        ch.update(_athlete_level(x, np.flatnonzero(keep), a, full, c, seed, fid, skip=(col,)))
        out.append(ch)
    return out


def _reused_rows(dictionary: pl.DataFrame, fpca_gate: pl.DataFrame | None, method_summary: pl.DataFrame | None,
                 trace: pl.DataFrame | None, registry: pl.DataFrame | None) -> tuple[list[dict], list[dict]]:
    """Characteristic rows taken from the tables of notebooks 02 and 03 and from the Phase 04 registry (no recomputation)."""
    chars, rels = [], []
    for r in dictionary.iter_rows(named=True):
        fid, key = r["feature_id"], r["source_column"]
        base = {"feature_id": fid, "scope": "all", "family": r["family"], "design": r["design"]}
        if r["source_table"] == GATE_TABLE and fpca_gate is not None:
            rep, comp = key.split("|")
            g = fe._one(fpca_gate, representation=rep, component=int(comp))
            if not g:
                continue
            method = "stratified ICC(1), position × outer fold, out-of-fold scores (notebook 02)"
            chars.append({**base, "source": "reused: fpca_component_gate (notebook 02)", "n_athletes_repeat": g["n_athletes_repeat"],
                          "reliability": g["icc_adj"], "reliability_lower": g["icc_lower"], "reliability_upper": g["icc_upper"],
                          "reliability_method": method, "reliability_of_mean": g["reliability_of_mean"], "reliability_s3": g["icc_s3"],
                          "reliability_cohort_min": g["icc_cohort_min"], "window_r2": g["window_r2_within"],
                          "window_r2_lower": g["window_r2_lower"], "window_r2_upper": g["window_r2_upper"], "cohort_r2": g["cohort_r2"],
                          "rho_forty": g["rho_forty"], "rho_forty_lower": g["rho_forty_lower"], "rho_forty_upper": g["rho_forty_upper"],
                          "rho_ten": g["rho_ten"], "redundancy_abs": g["redundancy_abs"], "redundancy_lower": g["redundancy_lower"],
                          "redundancy_upper": g["redundancy_upper"], "r2_stopwatch": g["r2_stopwatch"],
                          "r2_stopwatch_lower": g["r2_stopwatch_lower"], "r2_stopwatch_upper": g["r2_stopwatch_upper"],
                          "r2_body": g["r2_body"], "reliable_unique_share": g["reliable_unique_share"],
                          "note": f"notebook 02 gate: {g['status']}"})
            rels.append({"feature_id": fid, "scope": "position_adjusted", "n_athletes_repeat": g["n_athletes_repeat"], "icc": g["icc_adj"],
                         "ci_lower": g["icc_lower"], "ci_upper": g["icc_upper"], "reliability_of_mean": g["reliability_of_mean"],
                         "method": method})
        elif r["source_table"] == VECTOR_TABLE and method_summary is not None:
            s = fe._one(method_summary, method=key)
            t = fe._one(trace, method=key, space="raw") if trace is not None else {}
            if not s:
                continue
            method = "trace ICC (rotation-invariant), position × outer fold; repeat-attempt AUC beside it (notebook 03)"
            chars.append({**base, "source": "reused: repr_method_summary, repr_trace_reliability (notebook 03)",
                          "n_athletes_repeat": t.get("n_athletes_repeat"), "reliability": s["icc_trace"],
                          "reliability_lower": t.get("icc_lower"), "reliability_upper": t.get("icc_upper"), "reliability_method": method,
                          "retrieval_auc": s["auc"], "retrieval_auc_lower": s["auc_lower"], "retrieval_auc_upper": s["auc_upper"],
                          "residual_auc": s["resid_auc"], "residual_auc_lower": s["resid_auc_lower"],
                          "residual_auc_upper": s["resid_auc_upper"], "forty_from_feature_r2": s["forty_partial_r2"],
                          "window_r2": s["window_r2"], "window_r2_lower": s["window_r2_lower"], "window_r2_upper": s["window_r2_upper"],
                          "cohort_probe_gain": s["cohort_gain"],
                          "note": ("flagged collapsed in notebook 03 (effective rank share "
                                   f"{s['rank_share_min']:.2f})" if s["collapsed"] else None)})
            rels.append({"feature_id": fid, "scope": "position_adjusted", "n_athletes_repeat": t.get("n_athletes_repeat"),
                         "n_attempts": t.get("n_sequences"), "icc": s["icc_trace"], "ci_lower": t.get("icc_lower"),
                         "ci_upper": t.get("icc_upper"), "method": method})
        elif r["source_table"] == REGISTRY_TABLE and registry is not None:
            stratum, drill, feature = key.split("|")
            g = fe._one(registry, domain="combine", stratum=stratum, drill=drill, feature=feature)
            if not g:
                continue
            chars.append({**base, "source": "reused: Phase 04 V registry", "n_athletes": g["n_athletes"],
                          "reliability_lower": g["value_repeatability"],
                          "reliability_method": "Phase 04 Study B: smallest ICC(1) interval lower bound over smoothing settings",
                          "note": f"Phase 04 status {g['status']}: {g['status_reason']}"})
    return chars, rels


def coverage_table(dictionary: pl.DataFrame, d: AttemptData, a: AthleteData) -> pl.DataFrame:
    def groups(pos: np.ndarray, coh: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
        return ([("all", "ALL", np.ones(pos.size, dtype=bool))] + [("position", p, pos == p) for p in sorted(set(pos.tolist()))]
                + [("cohort", y, coh == y) for y in sorted(set(coh.tolist()))])
    rows = []
    for r in dictionary.iter_rows(named=True):
        fid = r["feature_id"]
        if fid in d.values:
            v, why = d.values[fid], d.reasons[fid]
            for gt, g, keep in groups(d.position, d.cohort):
                ok, nfl = np.isfinite(v[keep]), d.nfl_ids[keep]
                n_ath = int(np.unique(nfl).size)
                _, cnt = np.unique(nfl[ok], return_counts=True)
                names, k = np.unique(why[keep][~ok], return_counts=True)
                rows.append({"feature_id": fid, "group_type": gt, "group": g, "n_athletes": n_ath, "n_athletes_value": int(cnt.size),
                             "coverage_athletes": cnt.size / n_ath if n_ath else None, "n_athletes_repeat": int((cnt >= 2).sum()),
                             "n_attempts": int(keep.sum()), "n_attempts_value": int(ok.sum()),
                             "missing_reasons": "; ".join(f"{n}: {int(x)}" for n, x in sorted(zip(names.tolist(), k.tolist()),
                                                                                                key=lambda z: -z[1]))})
        elif fid in a.values:
            v = a.values[fid]
            for gt, g, keep in groups(a.position, a.cohort):
                ok = np.isfinite(v[keep])
                rows.append({"feature_id": fid, "group_type": gt, "group": g, "n_athletes": int(keep.sum()),
                             "n_athletes_value": int(ok.sum()), "coverage_athletes": float(ok.mean()) if keep.any() else None,
                             "missing_reasons": f"not_recorded_officially: {int((~ok).sum())}" if (~ok).any() else ""})
    return _table(rows, COVERAGE)


def value_tables(d: AttemptData, a: AthleteData) -> dict[str, pl.DataFrame]:
    """Long-form values. Attempt level keeps the missing rows and their reason; athlete level = mean of the attempts
    with a value (an attempt is never a row of the athlete table)."""
    att, ath = [], []
    for fid, v in d.values.items():
        att.append(pl.DataFrame({"feature_id": np.full(v.size, fid), "seq_id": d.seq_ids, "nfl_id": d.nfl_ids,
                                 "value": v, "missing_reason": d.reasons[fid]}, schema=ATTEMPT_VALUES.columns).fill_nan(None))
        ids, mean, _ = athlete_means(d.nfl_ids, v)
        n_ok = np.array([int(np.isfinite(v[d.nfl_ids == i]).sum()) for i in ids], dtype=np.int64)
        ar, ok = a.rows(ids), np.isfinite(mean[:, 0])
        ath.append(pl.DataFrame({"feature_id": np.full(int(ok.sum()), fid), "nfl_id": ids[ok], "position_group": a.position[ar][ok],
                                 "draft_cohort": a.cohort[ar][ok], "n_attempts": n_ok[ok], "value": mean[ok, 0]},
                                schema=ATHLETE_VALUES.columns))
    for fid, v in a.values.items():
        ok = np.isfinite(v)
        ath.append(pl.DataFrame({"feature_id": np.full(int(ok.sum()), fid), "nfl_id": a.ids[ok], "position_group": a.position[ok],
                                 "draft_cohort": a.cohort[ok], "n_attempts": [None] * int(ok.sum()), "value": v[ok]},
                                schema=ATHLETE_VALUES.columns))
    return {ATTEMPT_VALUES.name: pl.concat(att) if att else ATTEMPT_VALUES.empty(),
            ATHLETE_VALUES.name: pl.concat(ath) if ath else ATHLETE_VALUES.empty()}


def characterise_all(dictionary: pl.DataFrame, d: AttemptData, a: AthleteData, c: AtlasConfig, seed: int,
                     fpca_gate: pl.DataFrame | None = None, method_summary: pl.DataFrame | None = None,
                     trace: pl.DataFrame | None = None, registry: pl.DataFrame | None = None) -> dict[str, pl.DataFrame]:
    """Every table of ``CHAR_CONTRACTS``. Each feature goes through the procedure of its design; rows of learned
    representations and of other drills are taken from their own executed tables."""
    chars, rels, nuis = [], [], []
    for row in dictionary.iter_rows(named=True):
        if row["feature_id"] in d.values:
            x, y, z = _repeated(row, d, a, c, seed)
            chars, rels, nuis = chars + x, rels + y, nuis + z
        elif row["feature_id"] in a.values:
            chars += _static(row, a, c, seed)
    x, y = _reused_rows(dictionary, fpca_gate, method_summary, trace, registry)
    return {COVERAGE.name: coverage_table(dictionary, d, a), CHARACTERISTICS.name: _table(chars + x, CHARACTERISTICS),
            RELIABILITY.name: _table(rels + y, RELIABILITY), NUISANCE.name: _table(nuis, NUISANCE), **value_tables(d, a)}


# ============================================================================== screening
def _known(*xs: Any) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def screen_feature(ch: dict[str, Any], spec: dict[str, Any], c: AtlasConfig) -> dict[str, Any]:
    """Labels and grade of one feature from its ``all``-scope characteristics. Rules (heuristic lines of ``AtlasConfig``):

    support       adequate = coverage ≥ min_coverage and (repeated designs) ≥ min_repeat_athletes athletes with a repeat
    reliability   high = interval above icc_reference · moderate = above icc_floor · low = below icc_floor · else uncertain
    stopwatch     redundant = |ρ| interval above redundancy_reference · beyond_stopwatch = reliability of the athlete
                  mean exceeds the UPPER bound of the stopwatch R² by unique_margin · low_correlation_only otherwise when
                  the |ρ| interval is below the line (never called novel) · vector features: pending notebook 04B
    acquisition   sensitive = window interval above window_reference, or cohort share ≥ cohort_reference · robust = both below
    grade         Reject: nuisance / diagnostic role, rejected or unvalidated upstream, inadequate support, low reliability
                  Strong: high reliability and robust acquisition (learned vectors are capped at Provisional until 04B)
                  Provisional: at least moderate reliability, acquisition not sensitive · Exploratory: the rest"""
    role, design_ = spec["role"], spec["design"]
    vector = ch.get("retrieval_auc") is not None
    reused_registry = spec["source_table"] == REGISTRY_TABLE
    cov, n_rep = ch.get("coverage_athletes"), ch.get("n_athletes_repeat")
    if reused_registry or spec["source_table"] in (GATE_TABLE, VECTOR_TABLE):
        support = "adequate" if (n_rep or ch.get("n_athletes") or 0) >= c.min_repeat_athletes else "inadequate"
    elif cov is None:
        support = "unknown"
    else:
        support = "adequate" if cov >= c.min_coverage and (design_ == "static_athlete" or (n_rep or 0) >= c.min_repeat_athletes) else "inadequate"
    lo, hi = ch.get("reliability_lower"), ch.get("reliability_upper")
    if design_ == "static_athlete":
        rel = "not_estimable"
    elif _known(lo) and lo >= c.icc_reference:
        rel = "high"
    elif _known(lo) and lo >= c.icc_floor:
        rel = "moderate"
    elif _known(hi) and hi < c.icc_floor:
        rel = "low"
    else:
        rel = "uncertain" if _known(lo) or _known(ch.get("reliability")) else "unknown"
    rel_mean, r2_hi = ch.get("reliability_of_mean"), ch.get("r2_stopwatch_upper")
    if role == "reference" and spec["source_column"] in (FORTY, TEN):
        relation = "is_reference"
    elif vector:
        relation = "pending_04b"
    elif _known(ch.get("redundancy_lower")) and ch["redundancy_lower"] >= c.redundancy_reference:
        relation = "redundant"
    elif rel in ("high", "moderate") and _known(rel_mean, r2_hi) and rel_mean - r2_hi >= c.unique_margin:
        relation = "beyond_stopwatch"
    elif _known(ch.get("redundancy_upper")) and ch["redundancy_upper"] < c.redundancy_reference:
        relation = "low_correlation_only"
    else:
        relation = "uncertain"
    w_lo, w_hi = ch.get("window_r2_lower"), ch.get("window_r2_upper")
    cohort = ch.get("cohort_probe_gain") if vector else ch.get("cohort_r2")
    cohort_line = c.cohort_gain_reference if vector else c.cohort_reference
    if role == "nuisance":
        acq = "is_acquisition_variable"
    elif design_ == "static_athlete":
        acq = "not_applicable"
    elif (_known(w_lo) and w_lo >= c.window_reference) or (_known(cohort) and cohort >= cohort_line):
        acq = "sensitive"
    elif _known(w_hi, cohort) and w_hi < c.window_reference and cohort < cohort_line:
        acq = "robust"
    else:
        acq = "uncertain" if _known(ch.get("window_r2")) or _known(cohort) else "unknown"
    upstream = spec["quality_flags"] or ""
    if role == "nuisance":
        grade, reason = GRADES[3], "acquisition variable: a diagnostic of the recording, not an athlete characteristic"
    elif role == "diagnostic":
        grade, reason = GRADES[3], "fit diagnostic: describes the model fit, not the athlete"
    elif "phase04:REJECTED" in upstream or spec["interpretation_status"].startswith("unvalidated"):
        grade, reason = GRADES[3], "rejected or unvalidated upstream (Phase 04 registry): descriptive use only"
    elif support == "inadequate":
        grade, reason = GRADES[3], "inadequate support (coverage or athletes with a repeat below the reference)"
    elif role == "reference":
        grade, reason = GRADES[1], "official measurement kept as the reference; repeatability cannot be estimated in this dataset"
    elif reused_registry:
        grade, reason = GRADES[2], "other drill: smoothing robustness known (Phase 04); stopwatch and acquisition not characterised here"
    elif rel == "low":
        grade, reason = GRADES[3], "not repeatable (reliability interval below the floor)"
    elif rel == "high" and acq == "robust":
        grade, reason = (GRADES[1], "repeatable and robust in notebook 03; capped at Provisional until notebook 04B") if vector \
            else (GRADES[0], "high reliability and no measured acquisition dependence")
    elif rel in ("high", "moderate") and acq != "sensitive":
        grade, reason = GRADES[1], f"reliability {rel}, acquisition {acq}"
    else:
        grade, reason = GRADES[2], f"reliability {rel}, acquisition {acq}"
    usable = grade in GRADES[:2]
    if role == "reference":
        eligible = "reference_baseline" if usable else "not_eligible"
    elif not usable:
        eligible = "exploratory_only" if grade == GRADES[2] and not reused_registry else "not_eligible"
    elif relation == "redundant":
        eligible = "control"
    elif relation in ("beyond_stopwatch", "pending_04b"):
        eligible = "candidate"
    else:
        eligible = "candidate_unproven_distinctness"
    return {"feature_id": ch["feature_id"], "family": spec["family"], "design": design_, "role": role,
            "definition": spec["interpretation_status"], "support": support, "reliability_label": rel, "stopwatch_relation": relation,
            "acquisition": acq, "grade": grade, "transfer_eligibility": eligible, "reason": reason, "n_athletes_repeat": n_rep,
            "coverage_athletes": cov, **{k: ch.get(k) for k in ("reliability", "reliability_lower", "reliability_upper", "redundancy_abs",
                                                                 "r2_stopwatch", "reliable_unique_share", "window_r2", "cohort_r2")}}


def screening_table(characteristics: pl.DataFrame, dictionary: pl.DataFrame, c: AtlasConfig) -> pl.DataFrame:
    spec = {r["feature_id"]: r for r in dictionary.iter_rows(named=True)}
    rows = [screen_feature(ch, spec[ch["feature_id"]], c) for ch in characteristics.filter(pl.col("scope") == "all").iter_rows(named=True)
            if ch["feature_id"] in spec]
    return _table(rows, STATUS).sort("feature_id")


# ============================================================================== redundancy matrix and clusters
def matrix_features(dictionary: pl.DataFrame, status: pl.DataFrame, extra_nuisance: Sequence[str] = ("first_speed", "duration_s", "pos_resid_rms")
                    ) -> list[str]:
    """Scalar forty features and official measurements with adequate support, plus three acquisition variables (so a
    cluster driven by the recording is visible)."""
    ok = set(status.filter(pl.col("support") == "adequate")["feature_id"].to_list())
    d = dictionary.filter(pl.col("source_table").is_in([ATTEMPT_TABLE, SUSTAINED_TABLE, MART_TABLE]) & pl.col("role").is_in(["reference", "candidate"]))
    return [f for f in d["feature_id"].to_list() if f in ok] + [f"F7.forty.{n}" for n in extra_nuisance]


def athlete_matrix(features: Sequence[str], d: AttemptData, a: AthleteData, extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
                   ) -> np.ndarray:
    """(athletes of ``a``, features): official value, or mean of the athlete's attempts. ``extra``: feature_id →
    (athlete ids, values) for features kept elsewhere (functional PCA scores)."""
    M = np.full((a.ids.size, len(features)), np.nan)
    for j, f in enumerate(features):
        if f in a.values:
            M[:, j] = a.values[f]
        elif f in d.values:
            ids, mean, _ = athlete_means(d.nfl_ids, d.values[f])
            M[a.rows(ids), j] = mean[:, 0]
        elif extra and f in extra:
            ids, vals = extra[f]
            keep = np.isin(ids, a.ids)
            M[a.rows(np.asarray(ids)[keep]), j] = np.asarray(vals, float)[keep]
    return M


def redundancy_matrix(M: np.ndarray, position: np.ndarray, min_athletes: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """(ρ, n): Spearman correlation within position between every pair of columns, on the athletes holding both."""
    p = M.shape[1]
    R, N = np.eye(p), np.zeros((p, p), dtype=np.int64)
    for i in range(p):
        N[i, i] = int(np.isfinite(M[:, i]).sum())
        for j in range(i + 1, p):
            r = partial_corr(M[:, i], M[:, j], [position], (), "spearman")
            R[i, j] = R[j, i] = r["r"] if r["n"] >= min_athletes else NAN
            N[i, j] = N[j, i] = r["n"]
    return R, N


def cluster_features(R: np.ndarray, threshold: float) -> tuple[np.ndarray, list[int]]:
    """Average-linkage clustering on 1 − |ρ| (an undefined ρ counts as 0). Two groups are joined while their mean
    |ρ| is at least ``threshold``. Returns (cluster label per feature, display order)."""
    D = 1.0 - np.abs(np.nan_to_num(R, nan=0.0))
    groups = [[i] for i in range(R.shape[0])]
    while len(groups) > 1:
        best, pair = np.inf, None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                dist = float(D[np.ix_(groups[x], groups[y])].mean())
                if dist < best:
                    best, pair = dist, (x, y)
        if pair is None or best > 1.0 - threshold + 1e-12:
            break
        x, y = pair
        groups[x] = groups[x] + groups.pop(y)
    groups.sort(key=lambda g: (-len(g), min(g)))
    labels = np.zeros(R.shape[0], dtype=np.int64)
    for k, g in enumerate(groups):
        labels[g] = k + 1
    return labels, [i for g in groups for i in g]


def redundancy_tables(features: Sequence[str], R: np.ndarray, N: np.ndarray, status: pl.DataFrame, c: AtlasConfig
                      ) -> dict[str, pl.DataFrame]:
    labels, order = cluster_features(R, c.cluster_rho)
    rel = {r["feature_id"]: (r["reliability"] if r["reliability"] is not None else -1.0) for r in status.iter_rows(named=True)}
    mat = [{"feature_a": features[i], "feature_b": features[j], "n_athletes": int(N[i, j]), "rho": R[i, j]}
           for i in range(len(features)) for j in range(len(features))]
    rows = []
    for pos, i in enumerate(order):
        members = np.flatnonzero(labels == labels[i])
        rep = sorted(members, key=lambda m: (-rel.get(features[m], -1.0), features[m]))[0]
        inside = np.abs(R[np.ix_(members, members)])[~np.eye(members.size, dtype=bool)]
        rows.append({"feature_id": features[i], "cluster": int(labels[i]), "cluster_size": int(members.size),
                     "representative": features[rep], "display_order": pos,
                     "mean_abs_rho_in_cluster": float(np.nanmean(inside)) if inside.size and np.isfinite(inside).any() else None})
    return {MATRIX.name: _table(mat, MATRIX), CLUSTERS.name: _table(rows, CLUSTERS)}


def quality_retention(windows: pl.DataFrame) -> pl.DataFrame:
    """Athletes kept by the nested quality filters (S0 = every eligible attempt)."""
    w = windows.with_columns(pl.lit(True).alias("S0_all"), pl.col("draft_year").cast(pl.String).alias("_cohort"))
    groups = ([("all", "ALL", w)] + [("position", p, w.filter(pl.col("position_group") == p))
                                      for p in sorted(w["position_group"].drop_nulls().unique().to_list())]
              + [("cohort", y, w.filter(pl.col("_cohort") == y)) for y in sorted(w["_cohort"].drop_nulls().unique().to_list())])
    rows = []
    for gt, g, df in groups:
        base = None
        for s in ("S0_all", "S2_signal", "S3_no_boundary"):
            k = df.filter(pl.col(s).fill_null(False)).group_by("nfl_id").len()
            n_ath, n_rep = k.height, int((k["len"] >= 2).sum())
            base = base or (max(n_ath, 1), max(n_rep, 1))
            rows.append({"scenario": s, "group_type": gt, "group": g, "n_attempts": int(k["len"].sum()) if k.height else 0,
                         "n_athletes": n_ath, "n_athletes_repeat": n_rep, "share_athletes": n_ath / base[0],
                         "share_repeat": n_rep / base[1]})
    return _table(rows, RETENTION)


# ============================================================================== conclusions from executed tables
def _f(x: Any, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def conclusions_markdown(status: pl.DataFrame, dictionary: pl.DataFrame, c: AtlasConfig, run_mode: str) -> str:
    """Text of the conclusions cell of notebook 04A, written from the saved tables only."""
    lines = [f"_Generated from `feature_dictionary` and `atlas_feature_status` of this run (`RUN_MODE = {run_mode!r}`). Intervals "
             "are 95 %, athlete-level. Grades are heuristic screening labels for review._", ""]
    if run_mode == "smoke":
        lines += ["**Smoke run: a technical check on a few athletes. The counts and grades below must not be interpreted.**", ""]
    forty = status.filter(~pl.col("feature_id").str.contains(":", literal=True))
    names = dict(dictionary.select("feature_id", "feature_name").iter_rows())
    interp = dict(dictionary.select("feature_id", "interpretation_status").iter_rows())
    def count(expr: pl.Expr) -> int:
        return forty.filter(expr).height
    cand = pl.col("role") == "candidate"
    lines += ["**Seven levels of evidence (forty-yard dash and official measurements; candidates only from level 2)**",
              f"1. *Exists*: {dictionary.height} definitions in the dictionary ({forty.height} characterised on the forty or as "
              f"official measurements; {status.height - forty.height} from other drills, evidence reused from Phase 04).",
              f"2. *Measurable*: {count(cand & (pl.col('support') == 'adequate'))} of {count(cand)} candidates have adequate support "
              f"(coverage ≥ {c.min_coverage:g}, ≥ {c.min_repeat_athletes} athletes with a repeat).",
              f"3. *Repeatable*: {count(cand & (pl.col('reliability_label') == 'high'))} candidates have a reliability interval above "
              f"{c.icc_reference:g}; {count(cand & (pl.col('reliability_label') == 'moderate'))} above {c.icc_floor:g}; "
              f"{count(cand & (pl.col('reliability_label') == 'low'))} below it.",
              f"4. *Distinct from the stopwatch*: {count(cand & (pl.col('stopwatch_relation') == 'beyond_stopwatch'))} candidates carry "
              f"reliable variance the official times do not explain (margin {c.unique_margin:g}); "
              f"{count(cand & (pl.col('stopwatch_relation') == 'redundant'))} are redundant; "
              f"{count(cand & (pl.col('stopwatch_relation') == 'low_correlation_only'))} have a low correlation only, which is not "
              "evidence of new information.",
              f"5. *Plausible physical interpretation*: {sum(1 for f in forty.filter(cand)['feature_id'].to_list() if (interp.get(f) or '').startswith('interpretable'))} "
              "candidates have a definition readable in physical terms; learned representations have none established.",
              "6. *Ready for an NFL transfer evaluation*: decided by the frozen shortlist at the end of notebook 04B, not here.",
              "7. *Demonstrated NFL predictive value*: **NOT TESTED in this milestone.** No NFL outcome was read.", ""]
    for grade in GRADES[:2]:
        g = forty.filter(cand & (pl.col("grade") == grade)).sort("reliability", descending=True, nulls_last=True)
        lines.append(f"**{grade} candidates ({g.height})**")
        for r in g.iter_rows(named=True):
            lines.append(f"- `{names.get(r['feature_id'], r['feature_id'])}` ({r['family']}): reliability {_f(r['reliability'])} "
                         f"[{_f(r['reliability_lower'])}, {_f(r['reliability_upper'])}]; |ρ| with the official times "
                         f"{_f(r['redundancy_abs'])}; stopwatch R² {_f(r['r2_stopwatch'])}; reliable share not explained "
                         f"{_f(r['reliable_unique_share'])}; window R² {_f(r['window_r2'])}; cohort R² {_f(r['cohort_r2'])} → "
                         f"{r['stopwatch_relation']}, acquisition {r['acquisition']}.")
        if not g.height:
            lines.append("- none.")
        lines.append("")
    by = forty.filter(cand).group_by("family", "grade").len().sort("family", "grade")
    lines.append("**Candidates by family and grade**: " + ("; ".join(f"{r['family']} {r['grade']}: {r['len']}" for r in by.iter_rows(named=True)) or "none") + ".")
    lines += ["", "A feature with poor repeatability is not a training recommendation, and a grade is not a scouting metric: the "
                  "football meaning and the usefulness of every candidate remain to be established against NFL movement."]
    return "\n".join(lines)
