"""Milestone 12 configuration: declared tracks, primary outcomes fixed before execution, budgets, movement grids and the
status vocabulary used by every table and report.

Provenance of a value: ``established`` = reused project value (Milestone 11 config / executed result) · ``proposed`` =
fixed in this file before any Milestone 12 number existed · ``execution`` = cost knob.

Nothing in this module reads data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from ....data.utils import stable_hash, utc_now
from ..config import NotebookControls

VERSION = "12.0"

# ------------------------------------------------------------------------------ status vocabulary (brief §17)
PLANNED, AUTHORED, PENDING, EXECUTED, INCOMPLETE = "planned", "authored", "execution_pending", "executed", "incomplete"
INCONCLUSIVE, SUPPORTED, REJECTED = "inconclusive", "supported", "rejected"
STATUSES = (PLANNED, AUTHORED, PENDING, EXECUTED, INCOMPLETE, INCONCLUSIVE, SUPPORTED, REJECTED)
NOT_FEASIBLE, NOT_RUN = "not_feasible", "not_run"

# three evaluation targets (brief §5.1) — a result is always tagged with exactly one
RELIABILITY, PHYSICAL_TRANSFER, FOOTBALL_TRANSFER = "measurement_reliability", "cross_domain_physical_transfer", "football_outcome_transfer"
# training regime of a representation (brief §5.4, §12 F7) — never mixed in one table
INDUCTIVE, TRANSDUCTIVE = "inductive", "transductive"

# feasibility categories of 12.0
FEAS_READY, FEAS_GATED, FEAS_LIMITED, FEAS_BLOCKED = "ready", "gated_on_upstream", "limited_support", "blocked"


@dataclass(frozen=True)
class M12Config:
    version: str = VERSION
    seed: int = 20261015
    dt_s: float = 0.1                               # established: 10 Hz, both domains
    # ---- movement grids (shared with m11_envelope_hist: same edges, so no rescan) --------------------------------
    speed_step: float = 0.5                         # established: M11Config.env_speed_step [yd/s]
    speed_max: float = 11.5                         # established
    accel_step: float = 0.25                        # established [yd/s²]
    accel_lim: float = 8.0                          # established
    time_step: float = 0.1                          # proposed: forty speed–time grid [s]
    time_max: float = 4.0                           # proposed: rising phase of the forty lies within 4 s of onset
    onset_speed: float = 1.0                        # established: sprint onset (M11Config.sprint_identifiable_v0)
    # ---- support rules -------------------------------------------------------------------------------------------
    min_frames_bin_nfl: int = 40                    # established: M11Config.env_min_frames_bin
    min_frames_bin_combine: int = 0                 # D-12-1 (2026-10-15, after 12.0 support counts, no outcome seen): Combine a(v) is
                                                    # interpolated at bin mid-speeds (foundation.CURVE_RULE); support = bracketed by
                                                    # rising-phase frames ≤ 1 yd/s apart. Rule v1 (≥ 2 frames per bin) left 2–6 yd/s empty at 10 Hz
    min_games_athlete: int = 8                      # proposed: NFL games before an athlete distribution is estimated
    shared_support_share: float = 0.6               # proposed: a speed bin is "shared" when ≥ this share of eligible athletes support it in BOTH domains
    shape_vmin: float = 2.0                         # proposed: decreasing-upper-quantile restriction only above this speed
    max_interior_gap_bins: int = 1                  # proposed: one missing interior bin may be linearly bridged (flagged)
    # ---- 12C declared targets -------------------------------------------------------------------------------------
    taus: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95)
    primary_tau: float = 0.90                       # proposed (declared): upper but not extreme; 0.95 secondary
    crps_taus: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    spline_knots: int = 7                           # proposed: interior B-spline knots across the speed range
    spline_lambdas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    qr_iters: int = 60                              # execution: IRLS iterations of the smooth quantile fit
    game_splits: int = 3                            # proposed: random game half-splits (seeds) besides odd / even
    env_boot: int = 100                             # proposed: game-bootstrap replicates for quantile SEs (C3 pooling)
    # ---- shared evaluation (reused Milestone 11 rules) -----------------------------------------------------------
    outer_folds: int = 5                            # established
    inner_folds: int = 4                            # established
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)   # established: five ridge penalties
    n_boot: int = 1000                              # established
    n_perm: int = 200                               # established
    min_athletes: int = 30                          # established: M11Config.min_athletes_cell
    reliability_floor: float = 0.3                  # established
    # ---- 12A -------------------------------------------------------------------------------------------------------
    fpca_var: float = 0.90                          # proposed: smallest K reaching this TRAINING variance share
    fpca_max_k: int = 4                             # proposed
    elastic_grid: int = 41                          # execution: time points of the SRSF alignment
    # ---- 12B -------------------------------------------------------------------------------------------------------
    simex_lambdas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)        # established SIMEX grid (Cook & Stefanski)
    simex_reps: int = 100                           # execution
    em_iters: int = 300                             # execution
    sim_reps: int = 200                             # execution: synthetic replicates per scenario
    # ---- 12D -------------------------------------------------------------------------------------------------------
    fa_max_k: int = 4                               # proposed
    fa_iters: int = 300                             # execution
    fa_boot: int = 100                              # proposed: bootstrap refits for loading stability
    vae_min_athletes: int = 1000                    # proposed: D5 only when complete-view athletes reach this
    # ---- 12E -------------------------------------------------------------------------------------------------------
    ot_eps: float = 0.05                            # proposed: entropic regularisation (relative to the median cost)
    ot_eps_grid: tuple[float, ...] = (0.01, 0.05, 0.2)
    ot_iters: int = 500                             # execution
    ot_projections: int = 64                        # proposed: sliced-Wasserstein directions
    ot_boot: int = 50                               # proposed: game / attempt bootstrap per athlete distance
    ot_accel_on: float = 1.0                        # established: DemandConfig.accel_on — the accelerating state
    unbalanced_rho: float = 1.0                     # proposed: KL marginal relaxation of E6
    # ---- 12F -------------------------------------------------------------------------------------------------------
    seq_len: int = 48                               # established: M11Config.seq_len
    embed_dim: int = 32                             # proposed: small default
    width: int = 48
    epochs: int = 25
    batch: int = 256
    accum_steps: int = 1
    lr: float = 1e-3
    patience: int = 4
    max_minutes: float = 90.0                       # execution: per encoder
    nfl_windows_per_athlete_demand: int = 40        # established
    nfl_windows_max: int = 200_000                  # established
    lambda_mask: float = 1.0
    lambda_phys: float = 1.0
    lambda_temp: float = 0.5
    lambda_align: float = 0.5
    # ---- smoke -------------------------------------------------------------------------------------------------------
    smoke_athletes: int = 60

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def for_mode(self, controls: NotebookControls) -> "M12Config":
        """Smoke mode keeps every definition and lowers budgets and support lines so a few athletes exercise every path.
        Smoke numbers are wiring checks, never results."""
        if not controls.smoke:
            return replace(self, seed=controls.seed)
        return replace(self, seed=controls.seed, n_boot=60, n_perm=10, env_boot=10, game_splits=1, min_games_athlete=2, min_frames_bin_nfl=10,
                       min_athletes=6, simex_reps=10, em_iters=60, sim_reps=8, fa_boot=10, ot_boot=5, ot_iters=100, ot_projections=16,
                       epochs=2, batch=64, max_minutes=5.0, nfl_windows_per_athlete_demand=5, nfl_windows_max=5_000)

    def hash(self) -> str:
        return config_hash(self.params())


def config_hash(obj: Any) -> str:
    return stable_hash(obj)[:16]


# ------------------------------------------------------------------------------ declared tracks (brief §4, §19)
@dataclass(frozen=True)
class TrackSpec:
    track: str
    notebook: str
    question: str
    primary_hypothesis: str
    primary_outcome: str
    primary_comparison: str
    evaluation_target: str
    baselines: str
    negative_controls: str
    budget: str
    depends_on: tuple[str, ...]
    success: str
    failure: str


TRACKS: tuple[TrackSpec, ...] = (
    TrackSpec("12.0", "12_00_advanced_movement_research_foundation.ipynb",
              "Which athletes, observations and speed states support each advanced method?",
              "Not a test: inventory, grids, folds and feasibility.", "—", "—", RELIABILITY, "—", "—", "one pass over persisted tables + Combine frames",
              (), "every track has a declared cohort and grid", "—"),
    TrackSpec("12C", "12c_distributional_nfl_movement_capacity.ipynb",
              "Q3 — Are stable NFL movement distributions better targets than maxima / linear envelopes, and do Combine speed features predict them beyond official tests?",
              "Upper conditional acceleration quantiles (τ = 0.90) over the shared speed range are more game-split reliable than the observed-maximum envelope; "
              "fitted sprint Vmax (M3) lowers held-out RMSE of that target vs M1.",
              "nfl.q90_shared (estimator frozen by the declared reliability rule before any Combine value is joined); secondary nfl.game_top_speed_q75",
              "M3 vs M1 on the primary target; M2 vs M1 on the secondary target (Holm over the two)", PHYSICAL_TRANSFER,
              "C0 observed maximum / linear envelope; M0 position + cohort; M1 + officials", "athlete-label permutation within position; random projection",
              "estimators C0–C4 (C5 declared optional, not implemented); 7 ladder models; ≤ 200 permutations", ("12.0",),
              "primary Δ RMSE CI > 0 after Holm AND beats permutation null", "CI includes 0 → inconclusive at this n"),
    TrackSpec("12B", "12b_measurement_error_and_latent_traits.ipynb",
              "Q2 — Could unreliable Combine features be masking athlete-level relationships?",
              "For preregistered features, corrected latent associations exceed observed ones in proportion to 1/√ICC, but held-out prediction does not improve.",
              "corrected vs naive standardized coefficient of each preregistered feature on its matched NFL target",
              "regression calibration (B1) vs naive (B0) held-out Δ RMSE; SIMEX / EM coefficients descriptive", PHYSICAL_TRANSFER,
              "B0 naive athlete mean", "synthetic null (β = 0) false-positive rate; permutation", "7 features; SIMEX 4 λ × reps; EM; synthetic grid",
              ("12.0",), "corrected estimand interval excludes 0 AND synthetic coverage ≥ 0.9", "corrected interval includes 0 → attenuation does not explain the null"),
    TrackSpec("12A", "12a_functional_movement_transfer.ipynb",
              "Q1 — Does the whole Combine acceleration–speed function retain transferable information lost by scalar features?",
              "Combine functional scores (FPCA, fold-fitted) lower held-out integrated squared error of the NFL τ = 0.90 acceleration–speed function vs scalar band features + M1.",
              "athlete-mean integrated squared error over each athlete's supported shared speed bins", "A2 (M1 + FPCA scores) vs A0 (M1 + scalar bands)",
              PHYSICAL_TRANSFER, "functional M1 (officials only); scalar bands", "shuffled Combine curves within position; random functional scores",
              "spline λ grid × 4; K ≤ 4; elastic alignment exploratory", ("12.0", "12C"), "ΔISE CI > 0", "CI includes 0 → inconclusive"),
    TrackSpec("12D", "12d_multiview_latent_athletic_traits.ipynb",
              "Q4 — Do several drills jointly reveal reproducible physical traits that single drill metrics measure unreliably?",
              "Latent factor scores have higher attempt-split test–retest reliability than the most reliable single tracked feature of their views.",
              "test–retest correlation of factor scores (attempt-split) vs best single feature", "D3 structured multi-view FA vs best single feature",
              RELIABILITY, "D0 officials; D1 concatenated ridge", "factor model on within-position shuffled views", "K ≤ 4; ≤ 100 bootstrap refits",
              ("12.0",), "reliability gain CI > 0", "otherwise single features are as reliable"),
    TrackSpec("12E", "12e_conditional_optimal_transport_matching.ipynb",
              "Q5 — Do distribution-level comparisons recover cross-domain similarity missed by point features?",
              "Conditional OT (speed-bin conditional accelerating-state distributions) gives higher linked-athlete cross-domain retrieval AUC than standardized feature distance.",
              "cross-domain retrieval AUC within position", "E5 conditional OT vs E0 Euclidean features (paired athlete bootstrap)", PHYSICAL_TRANSFER,
              "E0 Euclidean, E1 Mahalanobis", "shuffled athlete labels; coarsened bins; uniform-mass control", "6 distances; ε grid × 3; ≤ 50 bootstrap per athlete",
              ("12.0",), "AUC difference CI > 0 and E5 AUC CI > 0.5", "otherwise not established"),
    TrackSpec("12F", "12f_physics_informed_shared_movement_encoder.ipynb",
              "Q6 — Does an encoder trained on Combine AND NFL sequences carry transferable physical information beyond the Combine-only TCN?",
              "Inductive F4 (physics-informed, matched-state aligned) beats the random encoder and engineered features on cross-domain retrieval AND adds held-out NFL prediction beyond M1.",
              "inductive cross-domain retrieval AUC; Δ RMSE on the 12C primary target", "F4 vs F0 engineered and vs random encoder", PHYSICAL_TRANSFER,
              "F0 engineered; F1 TCN-B (Combine only); random encoder", "random frozen encoder, shuffled identity, shuffled physics, domain-only classifier, no-physics ablation",
              "7 configurations at one size; inductive only for F2 and F4", ("12.0", "12C"), "beats random AND engineered with CIs", "not established"),
    TrackSpec("12G", "12g_advanced_movement_research_synthesis.ipynb",
              "Q7 — Which advanced methods beat simpler physical models under identical cohorts, baselines and protocol?",
              "Not a test: evidence synthesis + one common-cohort tournament (exploratory, Holm-adjusted).",
              "Δ RMSE vs M1 on the 12C primary target, identical athletes", "each available method vs M1", PHYSICAL_TRANSFER, "M1", "permutation of the best block",
              "reads artifacts; one ladder", ("12.0", "12C", "12B", "12A", "12D", "12E", "12F"), "—", "—"),
)
TRACK_IDS = tuple(t.track for t in TRACKS)


def track(tid: str) -> TrackSpec:
    return next(t for t in TRACKS if t.track == tid)


def track_table() -> list[dict[str, Any]]:
    return [{**asdict(t), "depends_on": ", ".join(t.depends_on)} for t in TRACKS]


def provenance(notebook: str, controls: NotebookControls, c: M12Config, **extra: Any) -> dict[str, Any]:
    return {"notebook": notebook, "milestone": "12", "engine_version": VERSION, "run_mode": controls.run_mode, "seed": controls.seed,
            "config_hash": c.hash(), "written_at": utc_now(), **extra}


__all__ = ["AUTHORED", "EXECUTED", "FEAS_BLOCKED", "FEAS_GATED", "FEAS_LIMITED", "FEAS_READY", "FOOTBALL_TRANSFER", "INCOMPLETE", "INCONCLUSIVE",
           "INDUCTIVE", "M12Config", "NOT_FEASIBLE", "NOT_RUN", "PENDING", "PHYSICAL_TRANSFER", "PLANNED", "REJECTED", "RELIABILITY", "STATUSES",
           "SUPPORTED", "TRACKS", "TRACK_IDS", "TRANSDUCTIVE", "TrackSpec", "VERSION", "config_hash", "provenance", "track", "track_table"]
