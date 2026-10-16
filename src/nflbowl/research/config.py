"""One configuration for the unified Milestone 12 notebook.

The notebook holds a plain ``CFG`` dict; ``resolve(CFG)`` validates it (unknown keys are errors), fills the run-mode
defaults and returns a frozen ``UnifiedConfig``. Every stage reads its switches from it: nothing is hard-coded in the
notebook cells.

Run modes (they only switch OPTIONAL analyses and the encoder preset; declared estimators, bootstrap sizes, folds and
seeds of the primary analyses never change):

    debug   wiring check. Encoder preset ``fast_iteration`` (smoke budgets: numbers are NOT results). Optional
            warehouse analyses off. Warehouse stages still run in the warehouse run mode given by
            ``warehouse_run_mode`` (``smoke`` needs the ``_sample`` artifacts).
    fast    full protocol of every PRIMARY analysis; expensive secondary analyses off (permutations, the 11B
            replication grid, the 12B synthetic grid, 12D bootstrap stability, 12A elastic alignment, 12E E4 / E6 / ε grid /
            per-athlete bootstrap, 12G cohort-size sensitivity). F4 only (+ F0 / random baselines).
    full    every declared analysis of 12.0–12G; the encoder extras still follow their own switches unless
            ``run_full_research`` is True (then: other families, ablations, inductive F2 / F4, learning curve, transfer).

Explicit per-option overrides (``options``) always win over the run-mode defaults.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

RUN_MODES = ("debug", "fast", "full")
DATA_MODES = ("auto", "bundle", "warehouse")
PRECISIONS = ("fp32", "amp", "fp16", "bf16")
STAGES = ("eda", "12.0", "12C", "12B", "12A", "12D", "12E", "12F", "12G")
RESEARCH_SEED = 20261015                       # M12Config.seed: fold manifest, bootstrap and encoder seeds were declared with it

# optional analyses per stage: (key, default in debug, fast, full)
_OPTIONS: dict[str, dict[str, tuple[bool, bool, bool]]] = {
    "12.0": {"force_rebuild_curves": (False, False, False)},
    "12C": {"permutation": (False, False, True), "replication": (False, False, True)},
    "12B": {"synthetic": (False, False, True), "bayes": (False, False, False)},
    "12A": {"elastic": (False, False, True)},
    "12D": {"bootstrap": (False, False, True)},
    "12E": {"heavy_methods": (False, False, True), "eps_sensitivity": (False, False, True), "consistency": (False, True, True),
            "own_distance_bootstrap": (False, False, True), "twins": (False, True, True), "downstream": (False, True, True),
            "controls": (True, True, True)},
    "12G": {"tournament": (True, True, True), "ot_block": (False, True, True), "cohort_size": (False, False, True)},
}
# numeric knobs that are NOT science (bounded loops whose size the original notebooks also exposed)
_KNOBS: dict[str, Any] = {
    "12.0": {"memory_budget_gb": 6.0},
    "12C": {"memory_budget_gb": 6.0},
    "12B": {"memory_budget_gb": 2.0},
    "12A": {"memory_budget_gb": 3.0, "elastic_max_attempts": 600, "elastic_window_s": 3.0},
    "12D": {"memory_budget_gb": 2.0},
    "12E": {"memory_budget_gb": 3.0, "n_perm_null": 20, "boot_athletes": 60,
            "consistency_methods": ["E5_conditional_w1", "E3_sliced_w2"]},
    "12G": {"size_fractions": [0.5, 0.75], "size_reps": 10},
}


class UnifiedConfigError(ValueError):
    """Invalid unified-notebook configuration."""


DEFAULTS: dict[str, Any] = {
    # ---- identity / mode
    "experiment_name": "m12_unified",
    "seed": RESEARCH_SEED,
    "run_mode": "fast",
    "resume": True,
    "run_full_research": False,
    # ---- locations (Colab defaults; any absolute path works)
    "drive_root": "/content/drive/MyDrive/NFLBowl",
    "bundle_dir": "/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1",
    "bundle_zip": None,                                       # optional: a bundle ZIP, extracted next to it once
    "research_root": "/content/drive/MyDrive/nfl-big-data-bowl-2027/research_root",   # Mode B: holds warehouse/, data/, reports/, artifacts/
    "local_stage_dir": "/content/nflbowl_stage",              # fast local disk for the memory-mapped bundle and temporary writes
    "data_mode": "auto",
    "warehouse_run_mode": "full",                             # nflbdb RUN_MODE of the warehouse stages ("full" | "smoke")
    "build_bundle_from_warehouse": False,                     # Mode B without a bundle: build it with nflbowl.data.builder
    "verify_bundle_sha256": False,                            # re-hash every bundle file (slow on Drive; sizes + small files always checked)
    # ---- device / precision (12F)
    "device": "auto",
    "precision": "fp32",                                      # fp32 default (earlier F4 fp16 overflow); "amp" = fp16 autocast + GradScaler after the fix check
    "embed_batch": 4096,
    "encoder_preset": None,                                   # None → colab_gpu (fast / full) or fast_iteration (debug)
    "encoder_science_overrides": {},                          # e.g. {"science.accum_steps": 2}: changes the science hash → another experiment
    "data_placement": "auto",
    # ---- persistence
    "save_figures": True,
    "save_tables": True,
    "save_models": True,
    "save_processed_data": True,
    "save_reports": True,
    "save_embeddings_copy": True,                            # copy embedding Parquet files into RUN_DIR/embeddings
    "figure_formats": ["png", "svg", "pdf"],
    "figure_dpi": 300,
    "csv_max_rows": 250_000,                                  # larger tables are written as Parquet only (noted in the manifest)
    "overwrite_figures": True,
    # ---- stage switches and behaviour
    "stages": {s: True for s in STAGES},
    "reuse_completed_stages": True,
    "continue_on_error": True,
    "run_gpu_training": True,
    # ---- 12F encoder plan
    "encoder_primary": "F4",
    "encoder_baselines": ["F0", "random"],
    "encoder_families_extra": [],                             # e.g. ["F2", "F3", "F5", "F6"]
    "encoder_ablations": False,
    "encoder_inductive": False,                               # F2 + F4 × outer folds (hours): needed for the transfer ladder and 12G T_encoder
    "encoder_learning_curve": False,
    "encoder_eval_level": "standard",
    "encoder_transfer": True,                                 # run the inductive ladder when inductive encoders + athletes.parquet exist
    "encoder_experiment_dir": None,                           # reopen an existing nflbowl experiment (e.g. .../NFLBowl/experiments/f4_main) instead of a new one
    # ---- per-stage optional analyses / knobs
    "options": {},
    "knobs": {},
}


@dataclass(frozen=True)
class UnifiedConfig:
    experiment_name: str
    seed: int
    run_mode: str
    resume: bool
    run_full_research: bool
    drive_root: str
    bundle_dir: str | None
    bundle_zip: str | None
    research_root: str | None
    local_stage_dir: str
    data_mode: str
    warehouse_run_mode: str
    build_bundle_from_warehouse: bool
    verify_bundle_sha256: bool
    device: str
    precision: str
    embed_batch: int
    encoder_preset: str
    encoder_science_overrides: dict[str, Any]
    data_placement: str
    save_figures: bool
    save_tables: bool
    save_models: bool
    save_processed_data: bool
    save_reports: bool
    save_embeddings_copy: bool
    figure_formats: tuple[str, ...]
    figure_dpi: int
    csv_max_rows: int
    overwrite_figures: bool
    stages: dict[str, bool]
    reuse_completed_stages: bool
    continue_on_error: bool
    run_gpu_training: bool
    encoder_primary: str
    encoder_baselines: tuple[str, ...]
    encoder_families_extra: tuple[str, ...]
    encoder_ablations: bool
    encoder_inductive: bool
    encoder_learning_curve: bool
    encoder_eval_level: str
    encoder_transfer: bool
    encoder_experiment_dir: str | None = None
    options: dict[str, dict[str, bool]] = field(default_factory=dict)
    knobs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------ paths
    @property
    def run_dir(self) -> Path:
        return Path(self.drive_root) / "unified_research" / self.experiment_name

    # ------------------------------------------------------------------ switches
    def enabled(self, stage: str) -> bool:
        return bool(self.stages.get(stage, False))

    def opt(self, stage: str, key: str) -> bool:
        return bool(self.options.get(stage, {}).get(key, False))

    def knob(self, stage: str, key: str) -> Any:
        return self.knobs.get(stage, {}).get(key)

    def runtime_overrides(self) -> dict[str, Any]:
        """``nflbowl.Experiment`` runtime overrides (do NOT change the science hash)."""
        prec = {"fp32": "fp32", "amp": "auto", "fp16": "fp16", "bf16": "bf16"}[self.precision]
        return {"runtime.device": self.device, "runtime.precision": prec, "runtime.embed_batch": int(self.embed_batch),
                "runtime.stage_dir": self.local_stage_dir, "runtime.data_placement": self.data_placement, "runtime.checkpoint_every": 1}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        return d

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path


def _merge_options(run_mode: str, full: bool, user: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    k = RUN_MODES.index(run_mode)
    out: dict[str, dict[str, bool]] = {}
    for st, opts in _OPTIONS.items():
        out[st] = {name: (True if full and name != "bayes" and name != "force_rebuild_curves" else vals[k]) for name, vals in opts.items()}
    for st, opts in (user or {}).items():
        if st not in _OPTIONS:
            raise UnifiedConfigError(f"options: unknown stage {st!r}; stages with options: {sorted(_OPTIONS)}")
        for name, v in opts.items():
            if name not in _OPTIONS[st]:
                raise UnifiedConfigError(f"options[{st!r}]: unknown option {name!r}; allowed: {sorted(_OPTIONS[st])}")
            out[st][name] = bool(v)
    return out


def _merge_knobs(user: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {st: dict(v) for st, v in _KNOBS.items()}
    for st, kn in (user or {}).items():
        if st not in _KNOBS:
            raise UnifiedConfigError(f"knobs: unknown stage {st!r}; stages with knobs: {sorted(_KNOBS)}")
        for name, v in kn.items():
            if name not in _KNOBS[st]:
                raise UnifiedConfigError(f"knobs[{st!r}]: unknown knob {name!r}; allowed: {sorted(_KNOBS[st])}")
            out[st][name] = v
    return out


def resolve(cfg: Mapping[str, Any]) -> UnifiedConfig:
    """Validate the notebook ``CFG`` and return the resolved, frozen configuration."""
    unknown = sorted(set(cfg) - set(DEFAULTS))
    if unknown:
        raise UnifiedConfigError(f"unknown CFG key(s) {unknown}; allowed: {sorted(DEFAULTS)}")
    raw = {**DEFAULTS, **dict(cfg)}
    raw["stages"] = {**DEFAULTS["stages"], **dict(cfg.get("stages") or {})}
    bad = sorted(set(raw["stages"]) - set(STAGES))
    if bad:
        raise UnifiedConfigError(f"stages: unknown stage(s) {bad}; allowed: {STAGES}")
    if raw["run_mode"] not in RUN_MODES:
        raise UnifiedConfigError(f"run_mode must be one of {RUN_MODES}")
    if raw["data_mode"] not in DATA_MODES:
        raise UnifiedConfigError(f"data_mode must be one of {DATA_MODES}")
    if raw["precision"] not in PRECISIONS:
        raise UnifiedConfigError(f"precision must be one of {PRECISIONS}")
    if raw["warehouse_run_mode"] not in ("full", "smoke"):
        raise UnifiedConfigError("warehouse_run_mode must be 'full' or 'smoke'")
    if raw["encoder_eval_level"] not in ("quick", "standard", "full"):
        raise UnifiedConfigError("encoder_eval_level must be quick / standard / full")
    if not str(raw["experiment_name"]).replace("_", "").replace("-", "").replace(".", "").isalnum():
        raise UnifiedConfigError("experiment_name may only contain letters, digits, '_', '-', '.'")
    bad_fmt = sorted(set(raw["figure_formats"]) - {"png", "svg", "pdf"})
    if bad_fmt:
        raise UnifiedConfigError(f"figure_formats: unsupported {bad_fmt}")
    full = bool(raw["run_full_research"])
    preset = raw["encoder_preset"] or ("fast_iteration" if raw["run_mode"] == "debug" else "colab_gpu")
    if full:
        raw["encoder_families_extra"] = list(raw["encoder_families_extra"]) or ["F2", "F3", "F5", "F6"]
        raw["encoder_ablations"] = True
        raw["encoder_inductive"] = True
        raw["encoder_learning_curve"] = True
    return UnifiedConfig(
        experiment_name=str(raw["experiment_name"]), seed=int(raw["seed"]), run_mode=raw["run_mode"], resume=bool(raw["resume"]),
        run_full_research=full, drive_root=str(raw["drive_root"]), bundle_dir=raw["bundle_dir"], bundle_zip=raw["bundle_zip"],
        research_root=raw["research_root"], local_stage_dir=str(raw["local_stage_dir"]), data_mode=raw["data_mode"],
        warehouse_run_mode=raw["warehouse_run_mode"], build_bundle_from_warehouse=bool(raw["build_bundle_from_warehouse"]),
        verify_bundle_sha256=bool(raw["verify_bundle_sha256"]), device=str(raw["device"]), precision=raw["precision"],
        embed_batch=int(raw["embed_batch"]), encoder_preset=preset, encoder_science_overrides=dict(raw["encoder_science_overrides"] or {}),
        data_placement=str(raw["data_placement"]), save_figures=bool(raw["save_figures"]), save_tables=bool(raw["save_tables"]),
        save_models=bool(raw["save_models"]), save_processed_data=bool(raw["save_processed_data"]), save_reports=bool(raw["save_reports"]),
        save_embeddings_copy=bool(raw["save_embeddings_copy"]), figure_formats=tuple(raw["figure_formats"]), figure_dpi=int(raw["figure_dpi"]),
        csv_max_rows=int(raw["csv_max_rows"]), overwrite_figures=bool(raw["overwrite_figures"]), stages=dict(raw["stages"]),
        reuse_completed_stages=bool(raw["reuse_completed_stages"]), continue_on_error=bool(raw["continue_on_error"]),
        run_gpu_training=bool(raw["run_gpu_training"]), encoder_primary=str(raw["encoder_primary"]),
        encoder_baselines=tuple(raw["encoder_baselines"]), encoder_families_extra=tuple(raw["encoder_families_extra"]),
        encoder_ablations=bool(raw["encoder_ablations"]), encoder_inductive=bool(raw["encoder_inductive"]),
        encoder_learning_curve=bool(raw["encoder_learning_curve"]), encoder_eval_level=raw["encoder_eval_level"],
        encoder_transfer=bool(raw["encoder_transfer"]), encoder_experiment_dir=raw["encoder_experiment_dir"],
        options=_merge_options(raw["run_mode"], full, raw["options"]),
        knobs=_merge_knobs(raw["knobs"]))


def describe(c: UnifiedConfig) -> Any:
    """Two-column table of the resolved switches (for display)."""
    import polars as pl

    rows = [{"key": k, "value": json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else str(v)} for k, v in c.to_dict().items()
            if k not in ("options", "knobs")]
    rows += [{"key": f"options.{st}.{k}", "value": str(v)} for st, d in c.options.items() for k, v in d.items()]
    rows += [{"key": f"knobs.{st}.{k}", "value": json.dumps(v)} for st, d in c.knobs.items() for k, v in d.items()]
    return pl.DataFrame(rows)


__all__ = ["DATA_MODES", "DEFAULTS", "PRECISIONS", "RESEARCH_SEED", "RUN_MODES", "STAGES", "UnifiedConfig", "UnifiedConfigError", "describe", "resolve"]
