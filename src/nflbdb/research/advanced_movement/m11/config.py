"""Milestone 11 configuration: thresholds (with provenance), experiment budgets, the declared study registry and the
status vocabularies every table and report uses.

Provenance of a value: ``established`` = reused project value or executed earlier result · ``proposed`` = fixed before
any Milestone 11 distribution was read · ``execution`` = cost knob · ``hypothetical`` = declared assumption that is not
a measurement (drill costs of 11G).

Nothing in this module reads data.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

from ....data.utils import stable_hash, utc_now
from ..config import NotebookControls

VERSION = "11.0"
BANDS: tuple[tuple[float, float], ...] = ((2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0))   # proposed (brief §6 B1)

# ------------------------------------------------------------------------------ vocabularies
SUPPORTED, LIMITED, UNSUPPORTED, UNKNOWN = "SUPPORTED", "LIMITED", "UNSUPPORTED", "UNKNOWN"
FEASIBILITY_STATES = (SUPPORTED, LIMITED, UNSUPPORTED, UNKNOWN)
NOT_FEASIBLE = "NOT_FEASIBLE"

# result status of an experiment / a report section (brief §18)
AUTHORED = "authored_not_executed"
EXECUTED_EXPLORATORY = "executed_exploratory"
COMPLETED_COMPARISON = "completed_statistical_comparison"
SUPPORTED_FINDING = "supported_finding"
INCONCLUSIVE = "inconclusive"
UNSUPPORTED_HYPOTHESIS = "unsupported_hypothesis"
RESULT_STATUSES = (AUTHORED, EXECUTED_EXPLORATORY, COMPLETED_COMPARISON, SUPPORTED_FINDING, INCONCLUSIVE, UNSUPPORTED_HYPOTHESIS)

# cell classes of the 11A matrix (brief §5 signature result) — an untested cell is NEVER a zero effect
ESTABLISHED_OVERLAP = "established_physical_overlap"
EXPLORATORY_RELATION = "exploratory_statistical_relationship"
SUPPORTED_TRANSFER = "supported_held_out_transfer"
NO_ESTABLISHED_TRANSFER = "no_established_transfer"
INSUFFICIENT_DATA = "insufficient_data"
NOT_TESTED = "not_tested"
MATRIX_CLASSES = (SUPPORTED_TRANSFER, EXPLORATORY_RELATION, NO_ESTABLISHED_TRANSFER, ESTABLISHED_OVERLAP, INSUFFICIENT_DATA, NOT_TESTED)

# 11I evidence levels (brief §13): a metric is never promoted automatically; every level has its own rule
EVIDENCE_LEVELS = ((1, "measurable"), (2, "repeatable"), (3, "nonredundant"), (4, "physically_transferable"),
                   (5, "football_outcome_associated"), (6, "predictively_validated"), (7, "operationally_useful"))

# finding types (brief §3): this milestone can establish the first two only
DESCRIPTIVE, PREDICTIVE, CAUSAL = "descriptive_association", "predictive_transfer", "causal_performance_benefit"


@dataclass(frozen=True)
class M11Config:
    version: str = VERSION
    seed: int = 20261011
    dt_s: float = 0.1                          # established: 10 Hz tracking, both domains
    # ---- speed states (11B) -----------------------------------------------------------------------------------
    bands: tuple[tuple[float, float], ...] = BANDS
    band_min_frames: int = 2                   # proposed: frames strictly inside a band before its smoothness is computed
    nfl_band_min_episodes: int = 10            # proposed: athlete-level NFL band value needs ≥ this many crossings
    # ---- envelopes (11B) --------------------------------------------------------------------------------------
    env_speed_step: float = 0.5                # proposed [yd/s]
    env_speed_max: float = 11.5                # established: AdvancedConfig.speed_hi
    env_accel_step: float = 0.25               # proposed [yd/s²]
    env_accel_lim: float = 8.0                 # established: OutcomeConfig.hist_accel_lim
    env_quantiles: tuple[float, ...] = (0.90, 0.95, 0.99)
    env_primary_quantile: float = 0.95         # proposed: upper envelope = 95th percentile of a_tan per speed bin
    env_min_frames_bin: int = 40               # proposed: frames in a speed bin before its quantile is used
    env_min_bins: int = 5                      # proposed: supported speed bins for an envelope fit
    env_fit_vmin: float = 1.0                  # proposed: below it a_tan of a standing player is filter noise
    env_boot: int = 200                        # proposed: game-resampling replicates per athlete
    # ---- sprint physics (11B) ---------------------------------------------------------------------------------
    sprint_tau_grid: tuple[float, float, int] = (0.4, 2.5, 85)     # proposed (s): search range of τ
    sprint_t0_grid: tuple[float, float, int] = (-1.0, 0.6, 33)     # proposed (s): search range of the onset offset
    sprint_identifiable_v0: float = 1.0        # proposed [yd/s]: first-frame speed below it → onset in the recording
    sprint_min_frames: int = 15                # proposed
    # ---- transitions (11C) ------------------------------------------------------------------------------------
    brake_on: float = -1.0                     # established: DemandConfig.accel_on (braking at a_tan ≤ −1 yd/s²)
    accel_on: float = 1.0                      # established
    brake_min_drop: float = 1.5                # proposed [yd/s]
    turn_min_deg: float = 45.0                 # established: DemandConfig.turn_min_deg
    heading_min_speed: float = 1.0             # proposed: velocity direction trusted above this speed (Combine frames)
    turn_window_s: float = 0.4                 # proposed: velocity averaged over this window before / after a transition
    recovery_fraction: float = 0.8             # proposed: Speed Recovery Time = time to regain 80 % of entry speed
    burst_threshold: float = 1.5               # proposed [yd/s²]: Brake-to-Burst latency ends when a_tan first ≥ this
    hmm_iters: int = 30                        # execution
    changepoint_penalty: float = 3.0           # proposed: BIC multiplier of the binary segmentation
    angle_bins: tuple[float, ...] = (45.0, 90.0, 135.0, 181.0)
    entry_speed_bins: tuple[float, ...] = (2.0, 4.0, 6.0, 12.0)
    # ---- cohorts and exposure --------------------------------------------------------------------------------
    min_episodes_athlete: int = 10             # proposed: NFL episodes before an athlete-level value exists
    exposure_grid: tuple[int, ...] = (10, 25, 50)
    min_athletes_cell: int = 30                # proposed: linked athletes for a SUPPORTED transfer cell
    min_athletes_limited: int = 15             # proposed: below it UNSUPPORTED, between → LIMITED
    min_attempts_drill: int = 1
    # ---- evaluation -------------------------------------------------------------------------------------------
    outer_folds: int = 5                       # established: Milestone 06 / AdvancedConfig
    inner_folds: int = 4
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)    # established: five ridge penalties (08B)
    n_boot: int = 1000
    n_perm: int = 200                          # proposed: athlete-label permutations per primary comparison (bounded)
    n_random_projections: int = 50             # proposed
    max_tracking_features: int = 3             # proposed: preselected interpretable tracking features per drill (M2)
    family_components: int = 3                 # proposed: 11G complexity control (training-fold PCA per drill family)
    reliability_floor: float = 0.3             # proposed: below it an attenuation correction is not reported
    # ---- representation learning (11F) -----------------------------------------------------------------------
    seq_len: int = 48                          # proposed: 4.8 s windows
    embed_dim: int = 64
    rep_epochs: int = 30
    rep_batch: int = 256
    rep_lr: float = 1e-3
    rep_patience: int = 5
    nfl_windows_per_athlete_demand: int = 40   # execution: NFL episodes sampled per athlete × demand for 11F
    nfl_windows_max: int = 200_000             # execution
    # ---- development (11D) -----------------------------------------------------------------------------------
    min_seasons: int = 2
    min_episodes_season: int = 10
    # ---- smoke -------------------------------------------------------------------------------------------------
    smoke_games: int = 48                      # established: OutcomeConfig.smoke_games

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def for_mode(self, controls: NotebookControls) -> "M11Config":
        """Smoke mode keeps every definition; it lowers support lines and budgets so a few games exercise every path.
        Smoke numbers are wiring checks, never results."""
        if not controls.smoke:
            return replace(self, seed=controls.seed)
        return replace(self, seed=controls.seed, n_boot=60, n_perm=20, n_random_projections=5, env_boot=20, env_min_frames_bin=10,
                       min_episodes_athlete=2, nfl_band_min_episodes=2, min_athletes_cell=6, min_athletes_limited=3, exposure_grid=(2, 5),
                       rep_epochs=2, rep_batch=64, nfl_windows_per_athlete_demand=5, nfl_windows_max=5_000, min_episodes_season=2)

    def hash(self) -> str:
        return config_hash(self.params())


def config_hash(obj: Any) -> str:
    return stable_hash(obj)[:16]


# ------------------------------------------------------------------------------ declared studies (brief §3 budget)
@dataclass(frozen=True)
class StudySpec:
    study_id: str
    track: str
    notebook: str
    question: str
    hypothesis: str
    baseline: str
    features: str
    metric: str
    negative_controls: str
    inference: str
    budget: str
    feasibility: str
    success: str
    failure: str
    finding_type: str = PREDICTIVE
    primary: bool = True


STUDIES: tuple[StudySpec, ...] = (
    StudySpec("S11_0", "11.0", "11_00_research_setup_and_opportunity_inventory.ipynb",
              "Which Combine drills and NFL movement demands share a measurable physical quantity, and for how many linked athletes?",
              "Not a hypothesis test: an inventory.", "—", "drill coverage, NFL episode counts", "counts, completeness",
              "—", "counts only", "one pass over persisted tables", "state per drill × demand × position by declared rule",
              "—", "—", DESCRIPTIVE, False),
    StudySpec("S11A", "11A", "11a_drill_to_nfl_transfer_matrix.ipynb",
              "Do drill-specific movement features add held-out information on physically comparable NFL movement?",
              "For SUPPORTED drill × demand cells, M2 (official + ≤ 3 preselected tracking features) lowers held-out RMSE vs M1.",
              "M1 = position, cohort, height, weight + official forty, 10-yd split (+ official drill time when the drill has one)",
              "≤ 3 interpretable features per drill, preselected by physics (combine_drills.PRESELECTED)",
              "Δ RMSE (M1 − M2) with paired athlete-bootstrap 95 % CI; Spearman; attenuation-corrected r",
              "wrong-drill features, random projections, athlete-label permutation within position, noncomparable interval",
              "paired athlete bootstrap; permutation p for the primary cell only", f"≤ 5 folds × 5 penalties × (cells ≤ 40); {200} permutations per primary",
              "SUPPORTED cells only; LIMITED cells descriptive; others NOT_FEASIBLE",
              "CI of Δ RMSE excludes 0 AND beats every negative control → supported held-out transfer (exploratory, one cohort)",
              "CI includes 0 → inconclusive at this n (never 'zero effect'); worse than a control → no established transfer"),
    StudySpec("S11B", "11B", "11b_speed_state_transfer_and_envelopes.ipynb",
              "Does Combine acceleration through a speed band predict NFL acceleration through the SAME band better than broad targets?",
              "Held-out Spearman of matched-band transfer exceeds that of the season-average target (O2.accel_sust05) for the same Combine feature.",
              "M1 official; broad target = Milestone 05 primary", "band accelerations 2–4/4–6/6–8(/8–10) yd/s; sprint (Vmax, τ, A0); envelope (A0, V0)",
              "held-out Spearman and Δ RMSE vs M1; partial r given official forty; attenuation-corrected r",
              "noncomparable band pairs (2–4 Combine vs 6–8 NFL), athlete-label permutation", "paired athlete bootstrap",
              "envelopes: game bootstrap ≤ 200 per athlete", "athletes with ≥ nfl_band_min_episodes crossings per band",
              "matched-band r > broad-target r with CI of the difference > 0", "otherwise inconclusive / unsupported"),
    StudySpec("S11C", "11C", "11c_braking_turning_reacceleration.ipynb",
              "Are braking, turning and re-acceleration metrics repeatable, distinct from the stopwatch, and related to NFL transitions?",
              "Transition metrics have ICC ≥ 0.5 across attempts and held-out association with geometry-matched NFL transitions.",
              "M1 official (incl. official shuttle / three-cone time where present)", "braking duration/distance, deceleration, TCI, Brake-to-Burst, recovery time, lateral accel",
              "ICC; stopwatch R²; held-out Δ RMSE; Spearman within geometry strata", "wrong-drill features, permutation, mismatched geometry",
              "athlete bootstrap", "segmenters: thresholds always; changepoint + HMM on a capped attempt sample",
              "drills with turns observed (shuttle, three-cone, transition drills)", "repeatable AND nonredundant AND held-out gain",
              "unstable or redundant → reported as such"),
    StudySpec("S11D", "11D", "11d_nfl_athletic_development.ipynb",
              "Do NFL movement characteristics change measurably across seasons, and are Combine features related to level or change?",
              "Between-athlete variance of season-to-season change exceeds measurement noise.",
              "position-adjusted season means", "per-season band accelerations, envelope A0/V0, braking, burst frequency, diversity",
              "random-slope variance with CI; within-athlete change; Spearman of change with Combine features",
              "shuffled season labels within athlete", "athlete bootstrap; mixed model (EM)", "panels from the 09A episodes and envelope histograms",
              "athletes with ≥ 2 seasons with ≥ min episodes", "slope variance CI excludes 0", "otherwise no detectable development heterogeneity",
              DESCRIPTIVE),
    StudySpec("S11E", "11E", "11e_stopwatch_twin_discordance.ipynb",
              "Within stopwatch twins, do Combine tracking differences point the same way as NFL movement differences?",
              "Sign concordance of ΔCombine and ΔNFL > 0.5 on reliable features.", "random same-position pairs; official-test differences",
              "band accelerations, sprint parameters, envelope parameters, supported non-forty measures",
              "sign concordance, Spearman of Δ, partial association given Δ official", "random pairs, wrong feature, stratum-preserving reshuffle",
              "athlete-cluster bootstrap (pair weight = product of athlete multiplicities); disjoint matching sensitivity",
              "≤ 1000 cluster replicates per feature × target", "pairs where both athletes have the NFL target", "concordance CI > 0.5 and > controls",
              "CI includes 0.5 → inconclusive"),
    StudySpec("S11F", "11F", "11f_shared_combine_nfl_representation.ipynb",
              "Can one self-supervised encoder represent Combine drills and NFL episodes so that athlete identity and physics are retrievable across domains?",
              "Cross-domain athlete retrieval AUC of the trained encoder exceeds the random encoder and the engineered baseline.",
              "random encoder; engineered summary features; original TCN-B (Combine forty only)", "learned embeddings (6 configurations)",
              "retrieval top-k / AUC, probe R², domain AUC, MMD, held-out transfer Δ RMSE", "random encoder, shuffled athlete labels",
              "athlete bootstrap of retrieval AUC", "6 configurations × epoch budget, early stopping, smoke/full",
              "PyTorch present; ≥ 1 athlete in both domains per position", "beats random AND engineered on cross-domain retrieval",
              "alignment that does not beat controls is reported as not established", PREDICTIVE),
    StudySpec("S11G", "11G", "11g_optimal_combine_drill_selection.ipynb",
              "Which subset of drills gives the most reliable incremental information for NFL movement evaluation?",
              "At least one non-forty drill family adds held-out information beyond the official tests on a common cohort.",
              "official tests only", "drill families reduced to ≤ 3 training-fold components",
              "Δ RMSE per added family, repeatability, redundancy, coverage, missingness resilience (reported separately)",
              "random family of equal size", "paired athlete bootstrap on the common cohort", "forward selection over ≤ 6 families",
              "common cohort of athletes with every compared drill", "positive Δ with CI > 0 on the common cohort",
              "complete-case gains are not generalised (participation is nonrandom)"),
    StudySpec("S11H", "11H", "11h_position_specific_football_actions.ipynb",
              "Do drill-specific characteristics correspond to observable position-specific football actions?",
              "For feasible position × action cells, M2 improves on M1 held-out.", "M0 position/cohort, M1 official",
              "relevant drill features per position", "Δ RMSE, Spearman; descriptive tables otherwise",
              "athlete-label permutation", "paired athlete bootstrap", "≤ 15 position × action cells",
              "event semantics supported, linked athletes ≥ min, target split-half reliability ≥ 0.5", "CI > 0", "descriptive only"),
    StudySpec("S11I", "11I", "11i_novel_metric_tournament_and_synthesis.ipynb",
              "Which new metrics are measurable, repeatable, nonredundant, transferable, football-associated, validated, useful?",
              "Not a test: a registry built from the executed tracks with explicit level rules.", "—", "every candidate metric",
              "level rules on persisted evidence", "—", "—", "reads artifacts only", "a metric without evidence stays unsupported", "—", "—",
              DESCRIPTIVE, False),
)


def study(study_id: str) -> StudySpec:
    return next(s for s in STUDIES if s.study_id == study_id)


def study_table() -> list[dict[str, Any]]:
    return [asdict(s) for s in STUDIES]


# ------------------------------------------------------------------------------ progress with ETA
class Progress:
    """Elapsed time and ETA for a loop of known length (no third-party bar; prints at most every ``every_s`` seconds)."""

    def __init__(self, total: int, label: str, every_s: float = 10.0, stream: Any = None):
        self.total, self.label, self.every_s = max(1, int(total)), label, every_s
        self.t0 = time.perf_counter()
        self.last = 0.0
        self.done = 0
        self.stream = stream or sys.stdout

    def step(self, k: int = 1, note: str = "") -> None:
        self.done += k
        now = time.perf_counter()
        if now - self.last >= self.every_s or self.done >= self.total:
            self.last = now
            el = now - self.t0
            eta = el / self.done * (self.total - self.done) if self.done else float("nan")
            print(f"[{self.label}] {self.done}/{self.total} · elapsed {el:,.0f}s · ETA {eta:,.0f}s {note}", file=self.stream, flush=True)

    def summary(self) -> dict[str, Any]:
        return {"label": self.label, "done": self.done, "total": self.total, "elapsed_s": time.perf_counter() - self.t0, "finished_at": utc_now()}


def provenance(notebook: str, controls: NotebookControls, c: M11Config, **extra: Any) -> dict[str, Any]:
    """Provenance block carried by every Milestone 11 result (brief §17)."""
    return {"notebook": notebook, "milestone": "11", "engine_version": VERSION, "run_mode": controls.run_mode, "seed": controls.seed,
            "config_hash": c.hash(), "written_at": utc_now(), **extra}


__all__ = ["AUTHORED", "BANDS", "CAUSAL", "COMPLETED_COMPARISON", "DESCRIPTIVE", "ESTABLISHED_OVERLAP", "EVIDENCE_LEVELS", "EXECUTED_EXPLORATORY",
           "EXPLORATORY_RELATION", "FEASIBILITY_STATES", "INCONCLUSIVE", "INSUFFICIENT_DATA", "LIMITED", "M11Config", "MATRIX_CLASSES",
           "NOT_FEASIBLE", "NOT_TESTED", "NO_ESTABLISHED_TRANSFER", "PREDICTIVE", "Progress", "RESULT_STATUSES", "STUDIES", "SUPPORTED",
           "SUPPORTED_FINDING", "SUPPORTED_TRANSFER", "StudySpec", "UNKNOWN", "UNSUPPORTED", "UNSUPPORTED_HYPOTHESIS", "VERSION", "config_hash",
           "provenance", "study", "study_table"]
