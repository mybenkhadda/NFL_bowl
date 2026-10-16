"""Notebook-level configuration of the M12 comprehensive experimental lab.

``CFG`` in the notebook is a plain dict. ``resolve(CFG)`` validates it (unknown keys are errors, never silently ignored),
fills run-mode defaults and returns a frozen ``LabConfig``. Nothing here is passed blindly into the nflbowl package:
``LabConfig.experiment_overrides()`` maps only the keys the package actually supports (runtime precision / device /
stage dir) onto ``Experiment`` overrides.

Run modes (expensive work is never switched on implicitly):

``pilot``     one held-out fold (``pilot_fold``), one encoder (``pilot_encoder``) + the F0 reference; ablations, seeds,
              learning curves off; bootstrap sizes reduced.
``standard``  the primary encoders (``primary_encoders``) on all outer folds; secondary encoders only when trained.
``full``      standard + secondary encoders, available F4 ablations, extra seeds, permutation tests, sensitivity runs —
              each still behind its own ``run_*`` flag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

RUN_MODES = ("pilot", "standard", "full")
PRECISIONS = ("fp32", "fp16", "bf16", "auto")
EXPECTED_BUNDLE = {"fingerprint": "4867be08f8191b3a", "n_sequences": 77_125, "frames": 64, "channels": 6, "n_athletes": 510, "folds": 5}


@dataclass(frozen=True)
class LabConfig:
    experiment_name: str = "m12_comprehensive_lab_v1"
    bundle_dir: str = "/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean"
    drive_root: str = "/content/drive/MyDrive/NFLBowl"
    stage_dir: str = "/content/nflbowl_stage"
    device: str = "auto"
    precision: str = "fp32"
    seed: int = 42                        # lab-level seed: sampling for figures, case-study selection, extra bootstraps
    science_seed: int | None = None       # None = keep the science seed of the base config (needed to reuse checkpoints)
    base_config: str = "colab_gpu"        # nflbowl preset; science identical to `default`
    resume: bool = True
    run_mode: str = "pilot"
    run_training: bool = False
    run_inductive: bool = True
    run_transfer: bool = False
    run_ablations: bool = False
    run_full_diagnostics: bool = True
    run_seeds: bool = False
    run_learning_curves: bool = False
    run_permutation_tests: bool = False
    reuse_artifacts: bool = True
    import_checkpoints: bool = True       # adopt compatible inductive best.pt files from earlier experiments (read-only on them)
    save_figures: bool = True
    save_tables: bool = True
    save_processed_data: bool = True
    save_reports: bool = True
    save_checkpoints: bool = True
    save_models: bool = True
    overwrite_figures: bool = True
    figure_formats: tuple[str, ...] = ("png", "svg", "pdf")
    figure_dpi: int = 300
    csv_max_rows: int = 10_000_000        # CSVs are never truncated; above this the CSV is skipped AND reported
    verify_bundle_hashes: bool = False    # True: re-hash every bundle file (slow on Drive; do once)
    expected_fingerprint: str | None = EXPECTED_BUNDLE["fingerprint"]
    pilot_fold: int = 0
    pilot_encoder: str = "F4"
    primary_encoders: tuple[str, ...] = ("F2", "F4")
    secondary_encoders: tuple[str, ...] = ("F3", "F5", "F6")
    reference_experiments: tuple[str, ...] = ("f4_main_clean", "f4_main")
    search_previous_runs: bool = True     # also scan unified_research/*/checkpoints/experiments/* for compatible checkpoints
    extra_seeds: tuple[int, ...] = (1, 2)
    n_boot: int | None = None             # None → run-mode default
    max_points_projection: int = 6000
    use_umap: bool = True                 # UMAP only when umap-learn is importable; PCA always
    min_subgroup: int = 10
    adequate_n: int = 30
    anonymize_ids: bool = True            # figures / reports show salted pseudonyms; Parquet tables keep nfl_id
    n_case_studies: int = 4
    n_profiles: int = 12
    git_ref: str = "main"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ derived settings
    @property
    def boot(self) -> int:
        return int(self.n_boot if self.n_boot is not None else {"pilot": 500, "standard": 2000, "full": 2000}[self.run_mode])

    def folds(self, available: list[int]) -> list[int]:
        return [self.pilot_fold] if self.run_mode == "pilot" else list(available)

    def encoders(self) -> list[str]:
        """Encoders the lab schedules (training only when run_training); F0 is always the reference."""
        if self.run_mode == "pilot":
            return [self.pilot_encoder]
        out = list(self.primary_encoders)
        if self.run_mode == "full":
            out += [m for m in self.secondary_encoders if m not in out]
        return out

    def experiment_overrides(self) -> dict[str, Any]:
        ov: dict[str, Any] = {"runtime.precision": self.precision, "runtime.device": self.device, "runtime.stage_dir": self.stage_dir}
        if self.science_seed is not None:
            ov["science.seed"] = int(self.science_seed)
        return ov

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve(cfg: Mapping[str, Any] | None = None) -> LabConfig:
    """Validate the notebook CFG dict → ``LabConfig`` (unknown keys raise; lists become tuples)."""
    cfg = dict(cfg or {})
    known = {f for f in LabConfig.__dataclass_fields__}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise KeyError(f"unknown CFG key(s) {unknown}; allowed: {sorted(known)}")
    base = LabConfig()
    vals: dict[str, Any] = {}
    for k, v in cfg.items():
        d = getattr(base, k)
        if isinstance(d, tuple) and isinstance(v, list):
            v = tuple(v)
        if isinstance(d, bool) and not isinstance(v, bool):
            raise TypeError(f"CFG[{k!r}] must be True/False, got {v!r}")
        vals[k] = v
    c = replace(base, **vals)
    if c.run_mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {RUN_MODES}, got {c.run_mode!r}")
    if c.precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {c.precision!r}")
    if c.figure_dpi < 72:
        raise ValueError("figure_dpi must be >= 72")
    if c.run_mode == "pilot" and (c.run_ablations or c.run_seeds or c.run_learning_curves):
        raise ValueError("pilot mode never runs ablations / seeds / learning curves: set run_mode='full' (and the flag) explicitly")
    return c


__all__ = ["EXPECTED_BUNDLE", "LabConfig", "PRECISIONS", "RUN_MODES", "resolve"]
