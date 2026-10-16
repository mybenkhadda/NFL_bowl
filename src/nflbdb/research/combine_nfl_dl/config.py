"""Parameters, model ladder and analysis list of the Combine → NFL defensive-line get-off experiment.

Provenance of every value: ``established`` = reused project value or executed earlier result, ``proposed`` =
pre-specified here before any Combine value was related to any NFL value, ``execution`` = cost knob.
Everything below was fixed before the experiment ran; the primary analysis is the single spec named ``primary``.

Model ladder (identical athletes for every model of one analysis):

    MEAN   training-fold mean (reference for the out-of-sample R²)
    M0     draft cohort + Combine weight and height
    M1     M0 + official forty and official 10-yard split
    M2a    M1 + p95_speed          tracked top-end speed
    M2b    M1 + accel_v4_7         mean acceleration while speed rises from 4 to 7 yd/s
    M2c    M1 + model_tau_s        time constant of the fitted sprint curve
    M2bc   M1 + accel_v4_7 + model_tau_s      EXPLORATORY, never a headline

Alignment (EDGE / INTERIOR_LINE) is observed in the NFL, so it is never a predictor. It enters the outcome only:
the primary target removes alignment × season means, estimated on the training athletes of each fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

VERSION = "1"
NAMESPACE = "combine_nfl_dl"
COHORT = "draft_year"
BODY = ("cmb_combine_weight", "cmb_combine_height")
TRADITIONAL = ("cmb_forty", "cmb_ten_yd_split")
JUMPS = ("cmb_vertical", "cmb_broad_jump")
TRACKING = ("p95_speed", "accel_v4_7", "model_tau_s")
TARGETS = ("adj_mean", "mean", "median", "p25", "p75")
PRIMARY_VARIANT = "mean|S0_all|all"
COMBINE_VARIANTS = (PRIMARY_VARIANT, "first|S0_all|all", "fastest|S0_all|all", "mean|S1_structural|all",
                    "mean|S2_signal|all", "mean|S3_no_boundary|all", "mean|S0_all|fit_ok")
FEATURE_MEANING = {
    "cmb_forty": "official forty time: acceleration and top speed over 40 yd, one number",
    "cmb_ten_yd_split": "official 10-yard split: the start and the first steps",
    "p95_speed": "tracked top-end speed sustained in the run",
    "accel_v4_7": "tracked mean acceleration while speed rises from 4 to 7 yd/s (mid-acceleration phase)",
    "model_tau_s": "time constant of the fitted sprint curve: how quickly the athlete approaches top speed "
                   "(lower = quicker)",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    exploratory: bool = False

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.categorical, *self.numeric)


def model_specs() -> dict[str, ModelSpec]:
    m0, m1 = BODY, (*BODY, *TRADITIONAL)
    specs = (
        ModelSpec("MEAN", "training mean"),
        ModelSpec("M0", "cohort + body", m0, (COHORT,)),
        ModelSpec("M1", "M0 + official forty and split", m1, (COHORT,)),
        ModelSpec("M2a", "M1 + tracked speed", (*m1, "p95_speed"), (COHORT,)),
        ModelSpec("M2b", "M1 + accel 4–7 yd/s", (*m1, "accel_v4_7"), (COHORT,)),
        ModelSpec("M2c", "M1 + sprint time constant", (*m1, "model_tau_s"), (COHORT,)),
        ModelSpec("M2bc", "M1 + both acceleration features", (*m1, "accel_v4_7", "model_tau_s"), (COHORT,), True),
        ModelSpec("M1j", "M1 + vertical and broad jump", (*m1, *JUMPS), (COHORT,), True),
    )
    return {s.name: s for s in specs}


REFERENCE_MODEL = "M1"
PRIMARY_MODELS = ("MEAN", "M0", "M1", "M2a", "M2b", "M2c")
TRACKING_MODELS = ("M2a", "M2b", "M2c")


@dataclass(frozen=True)
class GetOffConfig:
    version: str = VERSION
    endpoint: str = "player_get_off"
    position: str = "DL"
    alignments: tuple[str, ...] = ("EDGE", "INTERIOR_LINE")     # established: 99.6 % of endpoint plays (Study F)
    season_types: tuple[str, ...] = ("REG",)                    # established: DecisionAuditConfig.season_types
    seed: int = 20261011                                        # execution
    n_boot: int = 2000                                          # execution: athlete bootstrap, primary analysis
    n_boot_sensitivity: int = 500                               # execution: athlete bootstrap, each sensitivity
    # outcome ---------------------------------------------------------------------------------------------
    primary_target: str = "adj_mean"                            # proposed (PHASE_05_RESEARCH_DECISION §backup)
    primary_min_obs: int = 50      # proposed: Spearman–Brown plays for 0.7 on the context-centred ICC ≈ 44 (Study F)
    min_obs_grid: tuple[int, ...] = (20, 50, 100, 200)          # established: RescueConfig.outcome_obs_grid
    first_n_grid: tuple[int, ...] = (20, 50, 100)               # established: RescueConfig.first_n_grid
    game_min_obs: int = 5                                       # established: RescueConfig.game_min_obs
    n_game_splits: int = 200                                    # execution: random half-splits of an athlete's games
    low_value_s: float = 0.1                                    # proposed: values below are reported, never dropped
    boundary_tol_s: float = 1e-3                                # proposed: distance to the observed maximum
    # Combine features -------------------------------------------------------------------------------------
    max_attempts: int = 4                                       # established: RescueConfig.max_attempts
    # models -----------------------------------------------------------------------------------------------
    cv_folds: int = 5                                           # proposed
    cv_repeats: int = 20                                        # proposed: repeated K-fold, same folds for all models
    lambda_grid: tuple[float, float, int] = (-2.0, 4.0, 25)     # proposed: log10 range and size of the ridge grid
    huber_k: float = 1.345                                      # established convention (95 % Gaussian efficiency)
    min_athletes_model: int = 30                                # proposed: below → analysis reported as not estimable
    # matched athletes -------------------------------------------------------------------------------------
    match_feature: str = "accel_v4_7"                           # proposed: highest DL ICC among non-redundant features
    match_forty_s: float = 0.03                                 # proposed caliper on the official forty
    match_ten_s: float = 0.02                                   # proposed caliper on the official 10-yard split
    n_case_pairs: int = 6                                       # execution: pairs labelled in the figure

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    description: str
    target: str = "adj_mean"
    window: str = "all"                     # all · first_<n> · no_boundary
    min_obs: int = 50
    alignment: str | None = None            # fit and evaluate inside one NFL alignment group
    cohorts: tuple[int, ...] | None = None  # restrict to these draft cohorts
    combine_variant: str = PRIMARY_VARIANT
    cv: str = "repeated_kfold"              # repeated_kfold · loco (leave one draft cohort out)
    robust: bool = False                    # Huber loss instead of squared loss
    models: tuple[str, ...] = PRIMARY_MODELS

    def params(self) -> dict[str, Any]:
        return asdict(self)


def primary_spec(cfg: GetOffConfig) -> AnalysisSpec:
    """The primary analysis. M2bc rides on the same folds and athletes and is always flagged exploratory."""
    return AnalysisSpec("primary", "pre-specified primary analysis", cfg.primary_target, "all", cfg.primary_min_obs,
                        models=(*PRIMARY_MODELS, "M2bc"))


def sensitivity_specs(cfg: GetOffConfig, cohorts: tuple[int, ...] = (2023, 2024, 2025)) -> tuple[AnalysisSpec, ...]:
    """Every required sensitivity as one spec. None of them can replace the primary analysis."""
    t, n = cfg.primary_target, cfg.primary_min_obs
    out = [AnalysisSpec(f"alignment:{a}", f"athletes whose modal NFL alignment is {a}", t, "all", n, alignment=a)
           for a in cfg.alignments]
    # single cohorts are too small to refit: the stage scores the primary fit on each cohort instead (scored_on:*)
    out += [AnalysisSpec("leave_one_cohort_out", "train on two draft cohorts, predict the third", t, "all", n, cv="loco")]
    out += [AnalysisSpec("exclude_cohort:2024", "2023 and 2025 cohorts (2024 Combine tracking is noisier)", t, "all", n,
                         cohorts=tuple(c for c in cohorts if c != 2024))]
    out += [AnalysisSpec(f"first_{k}", f"first {k} eligible get-off observations of each athlete", t, f"first_{k}", k)
            for k in cfg.first_n_grid]
    out += [AnalysisSpec(f"min_obs:{k}", f"athletes with at least {k} observations", t, "all", k)
            for k in cfg.min_obs_grid if k != n]
    out += [AnalysisSpec(f"target:{x}", f"NFL endpoint = athlete {x}", x, "all", n) for x in TARGETS if x != t]
    out += [AnalysisSpec("no_boundary_values", "endpoint observations at the maximum or below the low bound removed",
                         t, "no_boundary", n)]
    out += [AnalysisSpec(f"combine:{v}", f"Combine tracking features = {v} (aggregation|quality scenario|fit filter)",
                         t, "all", n, combine_variant=v) for v in COMBINE_VARIANTS if v != PRIMARY_VARIANT]
    out += [AnalysisSpec(f"own_sample:{m}", f"largest sample available to {m} (M1 refitted on the same athletes)",
                         t, "all", n, models=("MEAN", "M0", "M1", m)) for m in TRACKING_MODELS]
    out += [AnalysisSpec("robust_huber", "Huber loss (ridge penalty kept)", t, "all", n, robust=True)]
    out += [AnalysisSpec("traditional_plus_jumps", "does adding vertical and broad jump to M1 change the baseline?",
                         t, "all", n, models=("MEAN", "M0", "M1", "M1j"))]
    return tuple(out)
