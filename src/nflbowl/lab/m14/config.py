"""Configuration and output store of Milestone 14.

``CFG`` in the notebook is a plain dict; ``resolve(CFG)`` validates it (unknown keys raise) and returns a frozen
``M14Config``. ``M14Config.m13()`` builds the equivalent ``M13Config`` so the M13 hand-off helpers (bundle validation,
experiment opening, read-only checkpoint / embedding adoption, inventory) are reused unchanged.

Execution policy encoded here:

* NO neural training, ever: M14 has no training switch. Encoders missing a fold are excluded.
* M12 / M13 directories are read only; M14 writes only under its own RUN_DIR.
* FP32 everywhere (the reused encoders were evaluated in FP32).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from ...artifacts.store import append_jsonl
from ...research.artifacts import safe_name
from ...utils.reproducibility import utc_now
from ..m13_config import ENCODERS, EXPECTED, M13Run

TRANSFORM_VERSION = "m14_v1"
M14_LAYOUT = ("config", "manifests", "logs", "datasets/cached", "datasets/audits", "features/events", "features/sequence", "features/athlete",
              "features/perturbed", "embeddings/inductive", "checkpoints", "tables/feasibility", "tables/registry", "tables/events", "tables/metrics",
              "tables/reliability", "tables/novelty", "tables/braking", "tables/transfer", "tables/position", "tables/robustness", "tables/scouting",
              "tables/synthesis", "tables/handoff", "figures/png", "figures/svg", "figures/pdf", "figures/data", "reports/markdown", "reports/html",
              "reports/athlete_profiles", "diagnostics", "exports")
M14_FAMILIES = ("handoff", "feasibility", "registry", "events", "metrics", "reliability", "novelty", "braking", "transfer", "position", "robustness",
                "scouting", "synthesis")
STATUSES = ("validated", "exploratory", "inconclusive", "blocked", "invalid")


@dataclass(frozen=True)
class M14Config:
    experiment_name: str = "m14_cross_sport_athletic_intelligence_v1"
    bundle_dir: str = "/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean"
    drive_root: str = "/content/drive/MyDrive/NFLBowl"
    m12_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m12_comprehensive_lab_v1"
    m12_experiment_name: str = "m12_comprehensive_lab_v1"
    m13_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m13_outcome_transfer_v1"
    m13_experiment_name: str = "m13_outcome_transfer_v1"
    stage_dir: str = "/content/nflbowl_stage"
    device: str = "auto"
    precision: str = "fp32"
    seed: int = 42                                     # M14 analysis seed (bootstrap, sampling); the science seed is the preset's
    base_config: str = "colab_gpu"
    expected_fingerprint: str = EXPECTED["fingerprint"]
    expected_science_hash: str = EXPECTED["science_hash"]
    encoders: tuple[str, ...] = ENCODERS
    combo_encoder: str = "F4"                          # PRE-DECLARED encoder for "M1 + reliable metrics + embedding" (M12 primary design, not chosen on outcomes)
    resume: bool = True
    reuse_artifacts: bool = True
    adopt_embeddings: bool = True
    allow_inference: bool = True                       # deterministic embedding of sequences with an ADOPTED checkpoint (no training)
    verify_checkpoint_load: bool = True
    # ---- event detection (Phase 1); every value is recorded in the event-cache signature
    entry_min_speed: float = 3.0                       # yd/s — approach speed below this is not a cut
    drop_min: float = 1.0                              # yd/s — minimum speed drop entry → minimum
    drop_frac: float = 0.25                            # … and at least this share of the entry speed
    rise_min: float = 1.0                              # yd/s — re-acceleration after the minimum that closes an event
    brake_thresh: float = 1.0                          # yd/s² — |a_tan| marking braking / propulsion
    sustain_frames: int = 2                            # frames a braking / propulsion threshold must hold (0.2 s)
    exit_horizon_s: float = 0.5                        # s after the minimum at which exit speed is read
    recovery_frac: float = 0.9                         # recovery = speed back to this share of the entry speed
    max_gap_dt: float = 0.15                           # s — a frame duration above this breaks a segment (missed samples)
    min_segment_frames: int = 7
    trace_half_window: int = 10                        # frames kept on each side of the minimum (aligned trace, ±1 s)
    useful_speed: float = 6.0                          # yd/s — "time to useful speed"
    v_ref: float = 7.0                                 # yd/s — speed at which the acceleration reserve is read
    turn_min_angle: float = 0.35                       # rad (20°) — |turn angle| below this has no turn direction
    # ---- reliability / quality (Phase 3)
    n_boot: int = 2000
    reliability_threshold: float = 0.6                 # ICC(1,1) of a trial for "reliable"
    reliability_marginal: float = 0.4
    min_reliability_athletes: int = 30
    reliability_k_grid: tuple[int, ...] = (2, 3, 4, 6)
    reliability_repeats: int = 20
    noise_sd: tuple[float, ...] = (0.05, 0.10)         # yd/s — Gaussian speed noise (a_tan noise = √2·σ / 0.1 s)
    smooth_windows: tuple[int, ...] = (3,)             # centred moving-average windows (frames)
    robustness_athletes: int = 150                     # athletes re-extracted under each perturbation
    robust_spearman: float = 0.8
    novelty_max_r2: float = 0.5                        # OOF R² of a metric from M1 + position above this = "not novel"
    redundancy_rho: float = 0.8
    min_coverage: float = 0.5
    # ---- transfer (Phase 5)
    single_metric_models: bool = True                  # M1 + one metric for every measurement-eligible metric (secondary, BH)
    n_permutations: int = 20                           # within-position permutation null of M1 + M14_reliable (0 = off)
    min_position_n: int = 20
    # ---- Hidden Braking Advantage (Phase 4)
    twin_covariates: tuple[str, ...] = ("forty", "ten", "weight", "three_cone", "short_shuttle")
    twin_min_coverage: float = 0.7                     # a twin covariate is used when observed for this share of athletes
    twin_caliper: float = 0.75                         # max Euclidean distance on within-position z of the covariates
    n_case_studies: int = 8
    anonymize_ids: bool = True
    # ---- outputs
    figure_formats: tuple[str, ...] = ("png", "svg", "pdf")
    figure_dpi: int = 300
    save_tables: bool = True
    save_figures: bool = True
    save_processed_data: bool = True
    save_reports: bool = True
    save_models: bool = True
    overwrite_figures: bool = True
    csv_max_rows: int = 10_000_000
    git_ref: str = "main"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.drive_root) / "experimental_lab" / self.experiment_name

    @property
    def m12_experiment_dir(self) -> Path:
        return Path(self.m12_run_dir) / "checkpoints" / "experiments" / self.m12_experiment_name

    @property
    def m13_experiment_dir(self) -> Path:
        return Path(self.m13_run_dir) / "checkpoints" / "experiments" / self.m13_experiment_name

    def training_enabled(self) -> bool:
        return False

    def m13(self) -> Any:
        """The equivalent ``M13Config`` (shared keys, M14 experiment name) for the reused hand-off helpers. M12 is the
        primary checkpoint source; the M13 experiment is searched too (it holds copies adopted from M12)."""
        from ..m13_config import resolve as m13_resolve

        extra = tuple(str(p) for p in (self.m13_experiment_dir,) if (p / "config" / "experiment.json").is_file())
        return m13_resolve({"experiment_name": self.experiment_name, "bundle_dir": self.bundle_dir, "drive_root": self.drive_root,
                            "m12_run_dir": self.m12_run_dir, "m12_experiment_name": self.m12_experiment_name, "extra_source_experiments": list(extra),
                            "stage_dir": self.stage_dir, "device": self.device, "precision": self.precision, "seed": self.seed,
                            "base_config": self.base_config, "expected_fingerprint": self.expected_fingerprint,
                            "expected_science_hash": self.expected_science_hash, "encoders": list(self.encoders), "resume": self.resume,
                            "reuse_artifacts": self.reuse_artifacts, "adopt_embeddings": self.adopt_embeddings, "allow_inference": self.allow_inference,
                            "verify_checkpoint_load": self.verify_checkpoint_load, "run_training": False, "allow_new_training": False,
                            "n_boot": self.n_boot, "figure_theme": "light"})

    def lab(self) -> Any:
        return self.m13().lab()

    def event_params(self) -> dict[str, Any]:
        keys = ("entry_min_speed", "drop_min", "drop_frac", "rise_min", "brake_thresh", "sustain_frames", "exit_horizon_s", "recovery_frac",
                "max_gap_dt", "min_segment_frames", "trace_half_window", "useful_speed", "v_ref", "turn_min_angle")
        return {k: getattr(self, k) for k in keys}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def params_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def resolve(cfg: Mapping[str, Any] | None = None) -> M14Config:
    cfg = dict(cfg or {})
    for forbidden in ("run_training", "allow_new_training"):
        if forbidden in cfg:
            raise KeyError(f"CFG[{forbidden!r}]: M14 never trains neural encoders (reuse the M12 inductive checkpoints)")
    known = set(M14Config.__dataclass_fields__)
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise KeyError(f"unknown CFG key(s) {unknown}; allowed: {sorted(known)}")
    base = M14Config()
    vals: dict[str, Any] = {}
    for k, v in cfg.items():
        d = getattr(base, k)
        if isinstance(d, tuple) and isinstance(v, list):
            v = tuple(v)
        if isinstance(d, bool) and not isinstance(v, bool):
            raise TypeError(f"CFG[{k!r}] must be True/False, got {v!r}")
        vals[k] = v
    c = replace(base, **vals)
    if c.precision != "fp32":
        raise ValueError("M14 runs in FP32 only")
    if not 0.0 < c.reliability_marginal < c.reliability_threshold < 1.0:
        raise ValueError("need 0 < reliability_marginal < reliability_threshold < 1")
    if not 0.0 < c.drop_frac < 1.0 or not 0.0 < c.recovery_frac <= 1.0:
        raise ValueError("drop_frac must lie in (0, 1) and recovery_frac in (0, 1]")
    if c.sustain_frames < 1 or c.min_segment_frames < 5:
        raise ValueError("sustain_frames ≥ 1 and min_segment_frames ≥ 5")
    if c.combo_encoder not in c.encoders:
        raise ValueError(f"combo_encoder {c.combo_encoder!r} must be one of encoders {c.encoders}")
    if any(k < 2 for k in c.reliability_k_grid):
        raise ValueError("reliability_k_grid values must be ≥ 2 (ICC needs two trials per athlete)")
    return c


class M14Run(M13Run):
    """RUN_DIR tree of M14 (``M14_LAYOUT``), tables per M14 family, provenance on every table and data file."""

    def __init__(self, root: Path | str, cfg: Any = None, local_tmp: Path | str | None = None):
        root = Path(root)
        for d in M14_LAYOUT:
            (root / d).mkdir(parents=True, exist_ok=True)
        super().__init__(root, cfg, local_tmp)
        self.provenance = {"transform_version": TRANSFORM_VERSION}

    def table(self, df: Any, name: str, family: str, note: str = "", provenance: dict[str, Any] | None = None) -> list[Path]:
        if family not in M14_FAMILIES:
            raise ValueError(f"unknown table family {family!r}; one of {M14_FAMILIES}")
        if df is None or not getattr(df, "width", 0):
            return []
        out = self.save_table(df, name, csv=None, note=note, subdir=f"tables/{family}")
        if out:
            append_jsonl({"table": f"{family}/{safe_name(name)}", "rows": int(getattr(df, "height", 0)), "written_at": utc_now(), **self.provenance,
                          **(provenance or {})}, self.root / "manifests" / "table_provenance.jsonl")
        return out

    def write_status(self, extra: dict[str, Any] | None = None) -> Path:
        """``manifests/m14_status.json``: every stage's status (resumption overview after a Colab disconnect)."""
        rec = {"updated_at": utc_now(), "transform_version": TRANSFORM_VERSION,
               "stages": {k: {kk: v.get(kk) for kk in ("status", "note", "seconds", "cache", "finished_at")} for k, v in self.stage_records().items()},
               **(extra or {})}
        p = self.root / "manifests" / "m14_status.json"
        p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        return p


__all__ = ["M14Config", "M14Run", "M14_FAMILIES", "M14_LAYOUT", "STATUSES", "TRANSFORM_VERSION", "params_hash", "resolve"]
