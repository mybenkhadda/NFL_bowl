"""Parameters and the candidate-feature list of the sprint reliability rescue experiment.

Provenance of every value: ``established`` = reused project value, ``proposed`` = uncalibrated convention (the
continuous evidence is always exported, so a cut can be changed without recomputation), ``execution`` = cost knob.
Units: yd, s, yd/s, yd/s².
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

RESCUE_VERSION = "1"
NAMESPACE = "sprint_rescue"
KEYS = ["event_id", "nfl_id", "attempt"]
SCENARIOS = ("S0_all", "S1_structural", "S2_signal", "S3_no_boundary")   # nested, as Phase 04 Study Q
VARIANTS = ("base", "drop1", "drop2", "drop3", "to_end", "trunc_end", "common_start")
MODEL_PARAMS = ("vmax", "tau_s", "a0", "a_at_ref")


@dataclass(frozen=True)
class RescueConfig:
    version: str = RESCUE_VERSION
    drill: str = "FORTY_YARD_DASH"                  # established: the only sprint drill (MovementConfig.sprint_drills)
    smoothing: str = "sg7p2"                        # established: primary SG setting (KinematicsConfig.smoothing)
    seed: int = 20261011                            # execution
    n_boot: int = 2000                              # execution: athlete bootstrap replicates
    # repeatability (established: MovementConfig reliability guards) --------------------------------------
    min_attempts: int = 2
    max_attempts: int = 4
    min_athletes: int = 10                          # below → no ICC
    min_athletes_label: int = 20                    # below → ICC reported, never labelled
    min_athletes_assoc: int = 20                    # proposed: athletes needed for an association estimate
    # R2 sustained features ------------------------------------------------------------------------------
    # proposed: mean acceleration between two SPEED levels (state-anchored, so it does not depend on where
    # the recording starts in time); null when the recording starts above the lower level.
    speed_bands: tuple[tuple[float, float], ...] = ((2.0, 5.0), (4.0, 7.0), (2.0, 7.0))
    # R3 model audit -------------------------------------------------------------------------------------
    drop_frames: tuple[int, ...] = (1, 2, 3)        # proposed: initial frames removed from the fit window
    truncate_end_s: float = 0.5                     # proposed: fit window shortened at the end
    common_start_speed: float = 2.0                 # established: MovementConfig.onset_speed_fast
    model_ref_speed: float = 3.0                    # proposed: speed at which the model acceleration is read
    sse_tolerance: float = 0.10                     # proposed: (τ, t0) cells with SSE ≤ (1 + tol)·min = "equally good"
    weak_tau_rel_width: float = 0.50                # proposed: τ range / τ above this → weakly identified
    # screening references (heuristic, never acceptance thresholds) --------------------------------------
    icc_reference: float = 0.75                     # established convention: Koo & Li 2016 "good"
    icc_floor: float = 0.50                         # established convention: Koo & Li 2016 "moderate"
    redundancy_reference: float = 0.70              # proposed
    position_dependence_gap: float = 0.10           # proposed: pooled − position-adjusted ICC above this
    # cohorts / NFL endpoints ----------------------------------------------------------------------------
    season_types: tuple[str, ...] = ("REG",)        # established: DecisionAuditConfig.season_types
    nfl_min_plays_grid: tuple[int, ...] = (50, 100, 200)
    outcome_obs_grid: tuple[int, ...] = (20, 50, 100, 200)
    first_n_grid: tuple[int, ...] = (20, 50, 100)
    game_min_obs: int = 5                           # proposed: observations for an athlete-game mean
    endpoints: tuple[tuple[str, str], ...] = (("player_get_off", "DL"), ("separation_at_pass_forward", "WR"))
    context_fields: tuple[str, ...] = ("lined_up_position", "blitzing", "down", "yards_to_go", "offense_formation",
                                       "pass_result", "quarter", "team_coverage_man_zone")
    official: tuple[str, ...] = ("cmb_forty", "cmb_ten_yd_split")
    traditional: tuple[str, ...] = ("cmb_vertical", "cmb_broad_jump", "cmb_combine_weight")
    # counts stated in earlier project reports, reconciled by Study E
    expected_counts: tuple[tuple[str, int], ...] = (("combine_results", 510), ("nfl_tracking_any", 497),
                                                    ("nfl_tracking_reg", 401))

    def params(self) -> dict[str, Any]:
        return asdict(self)


def band_name(lo: float, hi: float) -> str:
    return f"accel_v{lo:g}_{hi:g}".replace(".", "p")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    representation: str        # R1 direct kinematics · R2 sustained / integrated · R3 parametric profile · window
    family: str
    unit: str
    source: str                # persisted dataset the value is read from
    positive: bool             # strictly positive ratio-scale → CV % is meaningful
    meaning: str               # football interpretation, stated before any result


def feature_specs(cfg: RescueConfig | None = None) -> tuple[FeatureSpec, ...]:
    cfg = cfg or RescueConfig()
    p04, p03, here = "phase_04 trajectory_features_by_smoothing", "phase_03 combine_attempt_features", "sprint_rescue"
    bands = tuple(FeatureSpec(band_name(lo, hi), "R2", "acceleration_sustained", "yd/s²", here, True,
                              f"mean acceleration while speed rises from {lo:g} to {hi:g} yd/s")
                  for lo, hi in cfg.speed_bands)
    return (
        FeatureSpec("peak_speed", "R1", "speed", "yd/s", p04, True, "top speed reached in the run"),
        FeatureSpec("peak_speed_interior", "R1", "speed", "yd/s", p04, True, "top speed away from the filter edges"),
        FeatureSpec("p95_speed", "R1", "speed", "yd/s", p04, True, "sustained top-end speed"),
        FeatureSpec("peak_a_tan", "R1", "acceleration_direct", "yd/s²", p04, True, "largest single-frame acceleration"),
        FeatureSpec("p95_a_tan", "R1", "acceleration_direct", "yd/s²", p04, True, "high-end acceleration over the run"),
        FeatureSpec("p95_decel", "R1", "deceleration", "yd/s²", p04, False,
                    "braking — the forty recording ends before the athlete slows, so little is expected"),
        FeatureSpec("path_smooth_yds", "window", "recording_window", "yd", p04, True,
                    "recorded path length: a property of the recording window, not of the athlete"),
        FeatureSpec("accel_sustained", "R2", "acceleration_sustained", "yd/s²", p03, True,
                    "best 0.5 s average acceleration"),
        FeatureSpec("accel_initial", "R2", "acceleration_sustained", "yd/s²", p03, True,
                    "average acceleration in the first second after detected onset"),
        *bands,
        FeatureSpec("model_vmax", "R3", "sprint_profile", "yd/s", "phase_03 combine_sprint_profiles", True,
                    "asymptotic top speed of the fitted sprint curve"),
        FeatureSpec("model_tau_s", "R3", "sprint_profile", "s", "phase_03 combine_sprint_profiles", True,
                    "time constant: how quickly the athlete approaches top speed (lower = quicker)"),
        FeatureSpec("model_a0", "R3", "sprint_profile", "yd/s²", "phase_03 combine_sprint_profiles", True,
                    "model acceleration at zero speed — extrapolated to the model start, never observed"),
        FeatureSpec("model_a_at_ref", "R3", "sprint_profile", "yd/s²", here, True,
                    f"model acceleration at {cfg.model_ref_speed:g} yd/s, inside the recorded speed range"),
    )


FEATURE_NAMES = tuple(f.name for f in feature_specs())
P04_FEATURES = ("peak_speed", "peak_speed_interior", "p95_speed", "peak_a_tan", "peak_a_tan_interior", "p95_a_tan",
                "peak_decel", "p95_decel", "path_smooth_yds", "duration_valid_s")
REFERENCE_FEATURE = "p95_a_tan"      # direct acceleration measurement every other representation is compared with
