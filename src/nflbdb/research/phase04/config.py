"""Phase 04 · Milestone 1 parameters. Every field is documented in ``DOCS`` as ``established`` (reused
M1/M2/Study value), ``proposed`` (uncalibrated default — the continuous evidence is always exported, so a
cut-off can be changed without recomputation) or ``execution`` (cost knob, no scientific meaning).

Nothing here changes an M1/M2 feature definition: alternative smoothing settings are the ones
``KinematicsConfig.sensitivity_smoothing`` already declares, and event rules are the existing
lenient / default / strict sets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import polars as pl

ESTABLISHED, PROPOSED, EXECUTION = "established", "proposed", "execution"
PHASE04_VERSION = "1"
PHASE = "phase_04"
STATUSES = ("VALIDATED", "PROVISIONAL", "EXPERIMENTAL", "REJECTED")


@dataclass(frozen=True)
class EligibilityConfig:
    """Study Q — structural vs signal quality and athlete eligibility."""

    signal_quantile: float = 0.95               # signal flag: vendor RMSE or interior jerk above domain × stratum q
    min_attempts_eligible: int = 2              # athlete × drill needs ≥ this many retained attempts for reliability
    min_athletes: int = 10
    n_boot: int = 300


@dataclass(frozen=True)
class SmoothingStudyConfig:
    """Study B — smoothing robustness of trajectory-level kinematic summaries."""

    nfl_games_per_season: int = 8               # same deterministic NFL sample as M1 / Study A2 / A3
    top_fraction: float = 0.10                  # "top decile" overlap
    min_athletes: int = 10
    min_trajectories: int = 20
    n_boot: int = 200                           # athlete-cluster bootstrap for Spearman / overlap
    max_attempts: int = 4                       # ICC: first attempts per athlete × drill (= MovementConfig)
    min_attempts: int = 2
    speed_bands: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 7.0, 9.0)
    example_drills: tuple[str, ...] = ("FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL")
    robustness_settings: tuple[str, ...] = ("sg5p2", "sg11p3")   # extreme windows for cohort / position facets


@dataclass(frozen=True)
class EventStudyConfig:
    """Study C1 — event-detector sensitivity (rule × smoothing), exposure-normalised."""

    match_tolerance_s: float = 0.3              # = C2 onset tolerance (3 frames at 10 Hz)
    min_athletes: int = 10
    n_boot: int = 500


@dataclass(frozen=True)
class AnnotationDesignConfig:
    """Study C2 v2 — reference annotation design (Combine + NFL) and detector evaluation."""

    combine_strata: tuple[str, ...] = ("FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL", "SKILL_DRILLS_WR",
                                       "SKILL_DRILLS_DB", "SKILL_DRILLS_DL", "SKILL_DRILLS_OL", "SKILL_DRILLS_TE")
    nfl_positions: tuple[str, ...] = ("WR", "DB", "DL", "OL", "TE", "LB", "RB")
    per_stratum_random: int = 6                 # unbiased within-stratum estimate of precision / recall
    per_stratum_difficult: int = 2              # stress test (highest interior jerk) — reported separately
    dev_fraction: float = 0.30                  # athlete-level split: dev (detector revision) vs eval (frozen)
    tolerances_s: tuple[float, ...] = (0.2, 0.3, 0.5)
    primary_tolerance_s: float = 0.3
    n_boot: int = 1000


@dataclass(frozen=True)
class ValidationConfig:
    """Study V — evidence-based feature status (VALIDATED / PROVISIONAL / EXPERIMENTAL / REJECTED)."""

    rho_pass: float = 0.90                      # athlete-level Spearman (worst setting, CI lower bound)
    rho_fail: float = 0.80
    overlap_pass: float = 0.70                  # athlete-level top-decile overlap (worst setting)
    overlap_fail: float = 0.50
    rel_diff_pass: float = 0.05                 # median relative change of trajectory values (worst setting)
    rel_diff_fail: float = 0.15
    boundary_ratio_pass: float = 1.5            # A3 observed / expected share of peaks on filter-boundary frames
    boundary_ratio_fail: float = 3.0
    icc_pass: float = 0.75                      # min over settings of the ICC CI lower bound
    icc_fail: float = 0.50
    icc_range_pass: float = 0.05                # max − min ICC across settings
    f1_pass: float = 0.80                       # detector F1 CI lower bound on the EVAL split
    f1_fail: float = 0.60
    vendor_rel_mae_pass: float = 0.05           # A2 relative MAE (speed features only)
    primary_strata: tuple[str, ...] = ("FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL", "nfl_play")
    # severity of a failed criterion: critical → REJECTED, major → at most EXPERIMENTAL, minor → at most PROVISIONAL
    severity: tuple[tuple[str, str], ...] = (
        ("smooth_rank", "critical"), ("boundary", "critical"), ("detector_f1", "critical"),
        ("smooth_overlap", "major"), ("repeatability", "major"), ("event_smoothing", "major"),
        ("smooth_magnitude", "minor"), ("icc_setting_range", "minor"), ("vendor", "minor"))


@dataclass(frozen=True)
class Phase04Config:
    version: str = PHASE04_VERSION
    seed: int = 20261011
    sample_athletes: int | None = None
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    smoothing: SmoothingStudyConfig = field(default_factory=SmoothingStudyConfig)
    events: EventStudyConfig = field(default_factory=EventStudyConfig)
    annotation: AnnotationDesignConfig = field(default_factory=AnnotationDesignConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def section(self, name: str) -> dict[str, Any]:
        return {"version": self.version, "seed": self.seed, "sample_athletes": self.sample_athletes,
                name: asdict(getattr(self, name))}


_R = "the decision uses the exported continuous value; the cut can be changed without recomputation"
DOCS: dict[str, dict[str, tuple[str, str, str]]] = {
    "eligibility": {
        "signal_quantile": ("quantile", PROPOSED, "= Study A3 unreliable_quantile (same 'unreliable signal' target)"),
        "min_attempts_eligible": ("attempts", ESTABLISHED, "MovementConfig.min_attempts — ICC needs ≥ 2 attempts"),
        "min_athletes": ("athletes", PROPOSED, "clusters needed for a cluster-bootstrap interval"),
        "n_boot": ("resamples", EXECUTION, "proportions — cheap"),
    },
    "smoothing": {
        "nfl_games_per_season": ("games", ESTABLISHED, "KinematicDiagnostics / Study A2 / A3 deterministic sample"),
        "top_fraction": ("share", PROPOSED, "decile = scouting-relevant 'top athletes' set"),
        "min_athletes": ("athletes", PROPOSED, "clusters needed for a cluster-bootstrap interval"),
        "min_trajectories": ("trajectories", PROPOSED, "below this a comparison is not interpreted"),
        "n_boot": ("resamples", EXECUTION, "Spearman bootstrap is O(n log n) per resample"),
        "max_attempts": ("attempts", ESTABLISHED, "MovementConfig.max_attempts (position drills must not dominate)"),
        "min_attempts": ("attempts", ESTABLISHED, "MovementConfig.min_attempts"),
        "speed_bands": ("yd/s", ESTABLISHED, "QualityStudyConfig.speed_bands"),
        "example_drills": ("drills", ESTABLISHED, "M1 example drills"),
        "robustness_settings": ("labels", PROPOSED, "the narrowest and widest SG windows bracket the setting effect"),
    },
    "events": {
        "match_tolerance_s": ("s", ESTABLISHED, "AnnotationConfig.tolerance_s (C2)"),
        "min_athletes": ("athletes", PROPOSED, "clusters needed for a cluster-bootstrap interval"),
        "n_boot": ("resamples", EXECUTION, "ratio bootstrap — cheap"),
    },
    "annotation": {
        "combine_strata": ("strata", ESTABLISHED, "Study A drill_type strata with ≥ 40 athletes (LB drills excluded: "
                                                  "4 athletes in A3)"),
        "nfl_positions": ("position groups", ESTABLISHED, "dim_athlete.position_group"),
        "per_stratum_random": ("tasks", PROPOSED, "random tasks give an unbiased within-stratum P/R estimate"),
        "per_stratum_difficult": ("tasks", PROPOSED, "stress test; never pooled with random tasks for the headline"),
        "dev_fraction": ("share of athletes", PROPOSED, "athlete-level split so no athlete informs both detector "
                                                       "revision and evaluation"),
        "tolerances_s": ("s", PROPOSED, "2 / 3 / 5 frames — tolerance sensitivity is reported"),
        "primary_tolerance_s": ("s", ESTABLISHED, "AnnotationConfig.tolerance_s (C2)"),
        "n_boot": ("resamples", EXECUTION, "athlete-cluster bootstrap of TP/FP/FN sums"),
    },
    "validation": {
        **{k: ("Spearman ρ", PROPOSED, _R) for k in ("rho_pass", "rho_fail")},
        **{k: ("share", PROPOSED, _R) for k in ("overlap_pass", "overlap_fail")},
        **{k: ("relative change", PROPOSED, _R) for k in ("rel_diff_pass", "rel_diff_fail")},
        **{k: ("observed / expected", PROPOSED, _R) for k in ("boundary_ratio_pass", "boundary_ratio_fail")},
        "icc_pass": ("ICC", ESTABLISHED, "reliability_label: 'good' = CI lower ≥ 0.75 (Koo & Li 2016)"),
        "icc_fail": ("ICC", ESTABLISHED, "reliability_label: 'poor' below 0.5"),
        "icc_range_pass": ("ICC", PROPOSED, _R),
        **{k: ("F1", PROPOSED, _R) for k in ("f1_pass", "f1_fail")},
        "vendor_rel_mae_pass": ("relative MAE", PROPOSED, "A2: Combine rel MAE ≈ 0.016, NFL ≈ 0.037 (executed)"),
        "primary_strata": ("strata", PROPOSED, "the roll-up status uses the drills that answer research questions"),
        "severity": ("criterion → severity", PROPOSED, "critical = measurement is not trustworthy at all; major = "
                                                       "athlete ranking not trustworthy; minor = magnitude caveat"),
    },
}


def threshold_table(cfg: Phase04Config | None = None) -> pl.DataFrame:
    cfg = cfg or Phase04Config()
    rows = []
    for section, docs in DOCS.items():
        sub = getattr(cfg, section)
        for f in fields(sub):
            unit, source, why = docs[f.name]
            rows.append({"study_section": section, "name": f.name, "value": str(getattr(sub, f.name)), "unit": unit,
                         "provenance": source, "rationale": why})
    return pl.DataFrame(rows, schema={c: pl.String for c in ("study_section", "name", "value", "unit", "provenance",
                                                              "rationale")})
