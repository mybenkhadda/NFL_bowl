"""Experiment configuration, split into three independent sections.

``science``     everything that defines a result (seed, encoder size, losses, batches, budgets, splits, bootstrap sizes).
                Defaults are EXACTLY ``M12Config`` (Milestone 12F full run). Its hash is part of every checkpoint and
                artifact signature, so changing it can never silently reuse an old artifact.
``runtime``     how fast it runs (device, mixed precision, data placement, compile, staging). Runtime keys may change
                between sessions; the ones that change numerics (device type, precision) are still written into the
                checkpoint signature.
``evaluation``  evaluation protocol knobs (levels, opt-in approximations). Default = exact research protocol.

Presets live in ``nflbowl/configs/*.yaml`` (installed with the package): ``default``, ``colab_gpu``, ``research_full``,
``fast_iteration``. A YAML may say ``extends: <preset>``; later keys override earlier ones.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

PRESETS = ("default", "colab_gpu", "research_full", "fast_iteration")
RUN_MODES = ("full", "smoke", "fast_iteration")
PRECISIONS = ("auto", "fp32", "fp16", "bf16")
PLACEMENTS = ("auto", "gpu", "pinned", "cpu")
DEVICES = ("auto", "cpu", "cuda", "mps")
BACKENDS = ("fast", "reference")
LEVELS = ("quick", "standard", "full")


class ConfigError(ValueError):
    """Invalid configuration (unknown key, wrong type, value outside the allowed set)."""


def stable_hash(obj: Any, n: int = 16) -> str:
    """Same construction as ``nflbdb.data.utils.stable_hash`` (sha256 of sorted-key JSON)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


@dataclass(frozen=True)
class ScienceConfig:
    run_mode: str = "full"                  # label stored in checkpoint directories ("full" | "smoke" | "fast_iteration")
    seed: int = 20261015                    # M12Config.seed
    embed_dim: int = 32
    width: int = 48
    epochs: int = 25
    batch: int = 256                        # domain-balanced: half Combine, half NFL (se._balanced_batches)
    accum_steps: int = 1
    lr: float = 1e-3
    patience: int = 4
    max_minutes: float = 90.0               # wall-clock cap per encoder (M12Config); a capped run is flagged
    lambda_mask: float = 1.0
    lambda_phys: float = 1.0
    lambda_temp: float = 0.5
    lambda_align: float = 0.5
    val_share: float = 0.1                  # share of non-held-out ATHLETES used for early stopping (12F notebook)
    outer_folds: int = 5
    n_boot: int = 1000
    n_perm: int = 200
    max_train_batches: int | None = None    # None = full epoch; the 12F smoke mode used 4
    max_val_batches: int = 50               # 12F notebook: 50 (full) / 4 (smoke)
    inductive_configs: tuple[str, ...] = ("F2_cnn_physics", "F4_physics_contrastive")
    learning_curve_fractions: tuple[float, ...] = (0.1, 0.3)

    def validate(self) -> "ScienceConfig":
        if self.run_mode not in RUN_MODES:
            raise ConfigError(f"science.run_mode must be one of {RUN_MODES}, got {self.run_mode!r}")
        for k in ("embed_dim", "width", "epochs", "batch", "accum_steps", "patience", "outer_folds", "max_val_batches"):
            if int(getattr(self, k)) < 1:
                raise ConfigError(f"science.{k} must be >= 1")
        if self.batch < 4 or self.batch % 2:
            raise ConfigError("science.batch must be an even number >= 4 (batches are half Combine, half NFL)")
        if not 0.0 < self.val_share < 0.5:
            raise ConfigError("science.val_share must be in (0, 0.5)")
        if self.lr <= 0 or self.max_minutes <= 0:
            raise ConfigError("science.lr and science.max_minutes must be > 0")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ConfigError("science.max_train_batches must be None or >= 1")
        if any(not 0.0 < f <= 1.0 for f in self.learning_curve_fractions):
            raise ConfigError("science.learning_curve_fractions must lie in (0, 1]")
        return self

    def hash(self) -> str:
        return stable_hash(asdict(self))

    def to_m12config(self) -> Any:
        """The equivalent ``nflbdb`` ``M12Config`` (for ``se.scale_config`` and the ``advanced_evaluation`` ladder)."""
        from nflbdb.research.advanced_movement.m12.config import M12Config

        return replace(M12Config(), seed=self.seed, embed_dim=self.embed_dim, width=self.width, epochs=self.epochs, batch=self.batch,
                       accum_steps=self.accum_steps, lr=self.lr, patience=self.patience, max_minutes=self.max_minutes,
                       lambda_mask=self.lambda_mask, lambda_phys=self.lambda_phys, lambda_temp=self.lambda_temp,
                       lambda_align=self.lambda_align, n_boot=self.n_boot, n_perm=self.n_perm, outer_folds=self.outer_folds)


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"                    # auto → cuda, else mps (embedding / eval only), else cpu
    precision: str = "auto"                 # auto = fp16 autocast + GradScaler on CUDA (exactly the 12F notebook), fp32 elsewhere
    data_placement: str = "auto"            # gpu = whole sequence store resident on the GPU (no per-batch copies)
    gpu_memory_fraction: float = 0.5        # max share of FREE GPU memory the resident data may take in "auto"
    compile: bool = False                   # torch.compile of the encoder (opt-in; not always faster)
    compile_mode: str = "default"
    cudnn_benchmark: bool | None = None     # None = leave the PyTorch default untouched
    allow_tf32: bool | None = None          # None = leave the PyTorch default untouched
    embed_batch: int = 512                  # 12F notebook value; larger is faster on GPU
    embed_precision: str = "fp32"           # embeddings in fp32 (notebook); fp16 / bf16 = opt-in approximation
    checkpoint_every: int = 1               # epochs between checkpoints (the final / early-stopped epoch is always saved)
    stage_dir: str | None = None            # copy the data bundle here before memory-mapping (e.g. Colab local disk)
    num_threads: int | None = None          # torch.set_num_threads on CPU
    progress: bool = True

    def validate(self) -> "RuntimeConfig":
        if self.device not in DEVICES and not str(self.device).startswith("cuda:"):
            raise ConfigError(f"runtime.device must be one of {DEVICES} or 'cuda:<index>', got {self.device!r}")
        if self.precision not in PRECISIONS or self.embed_precision not in PRECISIONS:
            raise ConfigError(f"runtime.precision / embed_precision must be one of {PRECISIONS}")
        if self.data_placement not in PLACEMENTS:
            raise ConfigError(f"runtime.data_placement must be one of {PLACEMENTS}")
        if not 0.0 < self.gpu_memory_fraction <= 0.95:
            raise ConfigError("runtime.gpu_memory_fraction must be in (0, 0.95]")
        if self.embed_batch < 1 or self.checkpoint_every < 1:
            raise ConfigError("runtime.embed_batch and runtime.checkpoint_every must be >= 1")
        return self


@dataclass(frozen=True)
class EvaluationConfig:
    backend: str = "fast"                   # fast = same estimands and random draws, vectorised; reference = call nflbdb as-is
    default_level: str = "standard"
    identity_null_perms: int = 10           # 12F notebook (full run): 10 permutations
    quick_athlete_share: float = 0.25       # quick level: seeded athlete subsample (DIAGNOSTIC, not comparable)
    quick_min_athletes: int = 60
    quick_n_boot: int = 50
    matched_query_chunk: int = 256          # memory bound of matched retrieval (exact; chunking does not change ranks)
    approx_topk: bool = False               # opt-in: argpartition / GPU top-k in matched retrieval (tie order may differ)
    similarity_device: str = "cpu"          # opt-in "auto": GPU cosine similarities for matched retrieval (approximate)

    def validate(self) -> "EvaluationConfig":
        if self.backend not in BACKENDS:
            raise ConfigError(f"evaluation.backend must be one of {BACKENDS}")
        if self.default_level not in LEVELS:
            raise ConfigError(f"evaluation.default_level must be one of {LEVELS}")
        if not 0.0 < self.quick_athlete_share <= 1.0:
            raise ConfigError("evaluation.quick_athlete_share must be in (0, 1]")
        if self.identity_null_perms < 1 or self.quick_n_boot < 0 or self.matched_query_chunk < 1:
            raise ConfigError("evaluation.identity_null_perms / matched_query_chunk must be >= 1, quick_n_boot >= 0")
        if self.similarity_device not in ("cpu", "auto"):
            raise ConfigError("evaluation.similarity_device must be 'cpu' or 'auto'")
        return self

    def is_exact(self) -> bool:
        return not self.approx_topk and self.similarity_device == "cpu"


@dataclass(frozen=True)
class ExperimentConfig:
    science: ScienceConfig = field(default_factory=ScienceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    preset: str = "custom"

    def validate(self) -> "ExperimentConfig":
        self.science.validate()
        self.runtime.validate()
        self.evaluation.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"preset": self.preset, "science": asdict(self.science), "runtime": asdict(self.runtime), "evaluation": asdict(self.evaluation)}

    def science_hash(self) -> str:
        return self.science.hash()

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "ExperimentConfig":
        return self if not overrides else from_dict(_deep_merge(self.to_dict(), _dotted_to_nested(overrides)))


# ------------------------------------------------------------------------------ loading
_SECTIONS = {"science": ScienceConfig, "runtime": RuntimeConfig, "evaluation": EvaluationConfig}


def _coerce(cls: type, values: Mapping[str, Any], section: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{section}]: {unknown}; allowed: {sorted(known)}")
    out = {}
    for k, v in values.items():
        default = getattr(cls(), k)
        if isinstance(default, tuple) and isinstance(v, list):
            v = tuple(v)
        elif isinstance(default, bool) and not isinstance(v, bool) and v is not None:
            raise ConfigError(f"{section}.{k} must be true/false, got {v!r}")
        elif isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        out[k] = v
    return cls(**out)


def from_dict(d: Mapping[str, Any]) -> ExperimentConfig:
    unknown = sorted(set(d) - {"science", "runtime", "evaluation", "preset", "extends", "description"})
    if unknown:
        raise ConfigError(f"unknown top-level key(s) {unknown}; allowed: science, runtime, evaluation, preset, extends, description")
    secs = {name: _coerce(cls, d.get(name) or {}, name) for name, cls in _SECTIONS.items()}
    return ExperimentConfig(**secs, preset=str(d.get("preset", "custom"))).validate()


def _deep_merge(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else v
    return out


def _dotted_to_nested(d: Mapping[str, Any]) -> dict[str, Any]:
    """``{"science.epochs": 3}`` → ``{"science": {"epochs": 3}}``; nested dicts pass through."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        parts = str(k).split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = _dotted_to_nested(v) if isinstance(v, Mapping) else v
    return out


def _read_yaml(text: str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ConfigError("a configuration file must hold a mapping at the top level")
    return data


def preset_text(name: str) -> str:
    if name not in PRESETS:
        raise ConfigError(f"unknown preset {name!r}; available: {PRESETS}")
    return resources.files("nflbowl").joinpath("configs").joinpath(f"{name}.yaml").read_text(encoding="utf-8")   # one arg per call: Python 3.10


def _resolve_raw(raw: dict[str, Any], seen: tuple[str, ...] = ()) -> dict[str, Any]:
    base = raw.get("extends")
    if not base:
        return raw
    if base in seen:
        raise ConfigError(f"circular 'extends' chain: {seen + (base,)}")
    parent = _resolve_raw(_read_yaml(preset_text(base)), seen + (base,))
    merged = _deep_merge(parent, {k: v for k, v in raw.items() if k != "extends"})
    merged.pop("extends", None)
    return merged


def load_config(config: str | Path | Mapping[str, Any] | ExperimentConfig | None = None, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    """Resolve a preset name, a YAML path, a mapping or an ``ExperimentConfig``, then apply ``overrides``
    (nested or dotted keys, e.g. ``{"science.epochs": 5}``)."""
    if isinstance(config, ExperimentConfig):
        cfg = config.validate()
    elif config is None or (isinstance(config, str) and config in PRESETS):
        name = config or "default"
        raw = _resolve_raw(_read_yaml(preset_text(name)))
        raw.setdefault("preset", name)
        cfg = from_dict(raw)
    elif isinstance(config, Mapping):
        cfg = from_dict(_resolve_raw(dict(config)))
    else:
        p = Path(config)
        if not p.is_file():
            raise ConfigError(f"configuration {config!r} is neither a preset {PRESETS} nor an existing YAML file")
        raw = _resolve_raw(_read_yaml(p.read_text(encoding="utf-8")))
        raw.setdefault("preset", p.stem)
        cfg = from_dict(raw)
    return cfg.with_overrides(overrides)


__all__ = ["BACKENDS", "ConfigError", "EvaluationConfig", "ExperimentConfig", "LEVELS", "PRESETS", "RuntimeConfig", "ScienceConfig",
           "from_dict", "load_config", "preset_text", "stable_hash"]
