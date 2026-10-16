"""Configuration of the advanced-movement programme: notebook controls, research parameters, channel registry.

``NotebookControls``  the top-level switches every notebook exposes (run mode, caching, saving, device, budget).
``AdvancedConfig``    research parameters. Provenance of each value: ``established`` = reused project value or
                      executed earlier result, ``proposed`` = starting assumption fixed before any model is fitted,
                      ``execution`` = cost knob.
``CHANNELS``          candidate per-frame signals with unit, kind and the reason each is (not) translation-invariant.

Scale, from executed reports (sprint reliability rescue): 793 forty-yard-dash attempts of 418 athletes, 375 athletes
with a repeat, about 47–58 frames per attempt at 10 Hz — roughly 40,000 frames. The independent unit is the
athlete (418), not the frame.
"""

from __future__ import annotations

import importlib
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, replace
from typing import Any

VERSION = "1"
NAMESPACE = "advanced_movement"
RUN_MODES = ("smoke", "full")
DEVICES = ("auto", "cpu", "cuda")
REQUIRED_PACKAGES = ("numpy", "polars", "duckdb", "pyarrow", "matplotlib", "psutil")
LATER_PACKAGES = (("torch", "notebook 03 onwards (TCN, GRU, LSTM)"),)


class DependencyError(ImportError):
    """A package the notebook needs is not installed in the kernel's environment."""


@dataclass(frozen=True)
class NotebookControls:
    run_mode: str = "smoke"          # "smoke" = small athlete subset, minimal budget, outputs under */_sample
    force_rebuild: bool = False      # True = recompute even when a compatible cached artifact exists
    save_figures: bool = True
    save_tables: bool = True
    save_models: bool = True
    device: str = "auto"             # "auto" | "cpu" | "cuda"
    seed: int = 20261011
    num_workers: int = 0
    batch_size: int = 32
    max_epochs: int = 3

    def validate(self) -> "NotebookControls":
        if self.run_mode not in RUN_MODES:
            raise ValueError(f"RUN_MODE must be one of {RUN_MODES}, got {self.run_mode!r}")
        if self.device not in DEVICES:
            raise ValueError(f"DEVICE must be one of {DEVICES}, got {self.device!r}")
        if self.batch_size < 1 or self.max_epochs < 1 or self.num_workers < 0:
            raise ValueError("BATCH_SIZE and MAX_EPOCHS must be ≥ 1 and NUM_WORKERS ≥ 0")
        return self

    @property
    def smoke(self) -> bool:
        return self.run_mode == "smoke"


@dataclass(frozen=True)
class AdvancedConfig:
    version: str = VERSION
    primary_drill: str = "FORTY_YARD_DASH"        # established: the only sprint drill (MovementConfig.sprint_drills)
    min_frames: int = 20                          # proposed: shortest valid segment accepted as a sequence (2 s)
    frame_dt_s: float = 0.1                       # established: 10 Hz tracking
    max_attempts: int = 4                         # established: RescueConfig.max_attempts
    # athlete split registry ---------------------------------------------------------------------------
    outer_folds: int = 5                          # proposed
    split_repeats: int = 3                        # proposed: persistent repeated outer splits
    inner_folds: int = 4                          # proposed: development folds inside each outer training set
    design_fraction: float = 0.30                 # proposed: athletes on which preprocessing decisions are quantified
    smoke_athletes: int = 60                      # execution: athletes kept in smoke mode (all their attempts)
    # alignment grids ----------------------------------------------------------------------------------
    time_max_s: float = 6.0                       # established: 95th percentile of forty duration is 5.8 s
    norm_points: int = 64                         # proposed
    dist_step_yd: float = 0.5                     # proposed
    dist_max_yd: float = 40.0                     # established: the drill
    speed_lo: float = 1.0                         # proposed [yd/s]
    speed_hi: float = 11.5                        # established: 95th percentile of last-frame speed is 11.4 yd/s
    speed_step: float = 0.25                      # proposed [yd/s]
    phase_ref_speed: float = 2.0                  # established: MovementConfig.onset_speed_fast
    min_athletes_curve: int = 20                  # proposed: repeated athletes needed at a grid point for an ICC
    # NFL outcome census -------------------------------------------------------------------------------
    nfl_min_plays_grid: tuple[int, ...] = (20, 50, 100)     # established: RescueConfig grids
    season_types: tuple[str, ...] = ("REG",)

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def for_mode(self, controls: NotebookControls) -> "AdvancedConfig":
        """Smoke mode keeps every scientific parameter; it only shortens the repeated splits."""
        return replace(self, split_repeats=1) if controls.smoke else self


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    unit: str
    kind: str            # kinematic · geometry · clock · quality
    invariant: bool      # unchanged by translating / rotating the field coordinates
    default: bool        # proposed member of the model input (reviewed in notebook 01)
    meaning: str


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("t_rel", "s", "clock", True, False, "seconds since the first valid frame of the recording"),
    ChannelSpec("dist", "yd", "geometry", True, False, "smoothed path length since the first valid frame"),
    ChannelSpec("along", "yd", "geometry", True, False, "progress along the straight line from first to last frame"),
    ChannelSpec("speed", "yd/s", "kinematic", True, True, "magnitude of the smoothed velocity"),
    ChannelSpec("a_tan", "yd/s²", "kinematic", True, True, "rate of change of speed (signed)"),
    ChannelSpec("a_lat", "yd/s²", "kinematic", True, True, "lateral acceleration (signed; mirror-dependent)"),
    ChannelSpec("yaw_rate", "deg/s", "kinematic", True, False, "rate of change of the direction of motion"),
    ChannelSpec("lat_dev", "yd", "geometry", True, True, "sideways deviation from the first-to-last line"),
    ChannelSpec("head_sin", "–", "geometry", True, False, "sine of the direction of motion relative to the run line"),
    ChannelSpec("head_cos", "–", "geometry", True, False, "cosine of the direction of motion relative to the run line"),
    ChannelSpec("pos_resid", "yd", "quality", True, False, "distance between raw and smoothed position (tracking noise)"),
)
CHANNEL_NAMES = tuple(c.name for c in CHANNELS)
DEFAULT_CHANNELS = tuple(c.name for c in CHANNELS if c.default)
MASKS = ("boundary", "heading_valid")            # per-frame flags carried beside the channels
EXCLUDED_RAW = {"x_s": "absolute field position: encodes lane and recording convention, not the athlete",
                "y_s": "absolute field position: encodes lane and recording convention, not the athlete",
                "heading": "absolute angle: circular and dependent on the run direction; replaced by head_sin / head_cos",
                "s": "vendor speed: superseded by the derived speed (Phase 03 M1)",
                "a": "vendor acceleration: a magnitude, not a signed tangential value (Phase 03 M1)"}


# ============================================================================== environment
def check_dependencies(required: tuple[str, ...] = REQUIRED_PACKAGES) -> list[dict[str, str]]:
    """Version table of the packages the programme uses. A missing REQUIRED package raises with the action to
    take; nothing is ever installed from here. Packages needed only by later notebooks are reported, not raised."""
    rows, missing = [], []
    for name in required:
        try:
            rows.append({"package": name, "version": getattr(importlib.import_module(name), "__version__", "?"),
                         "status": "ok", "needed_for": "all notebooks"})
        except ImportError:
            missing.append(name)
    for name, where in LATER_PACKAGES:
        try:
            rows.append({"package": name, "version": getattr(importlib.import_module(name), "__version__", "?"),
                         "status": "ok", "needed_for": where})
        except ImportError:
            rows.append({"package": name, "version": "", "status": "MISSING", "needed_for": where})
    if missing:
        raise DependencyError(f"missing package(s) {missing} in {sys.executable}. Install them into this environment "
                              "yourself (the notebooks never install anything), then restart the kernel.")
    return rows


def environment_report() -> list[dict[str, str]]:
    """Python, platform, CPU, memory and accelerator facts as (item, value) rows. Imports torch only if present."""
    rows = [{"item": "python", "value": sys.version.split()[0]}, {"item": "executable", "value": sys.executable},
            {"item": "platform", "value": platform.platform()}, {"item": "cpu_count", "value": str(os.cpu_count())}]
    try:
        import psutil

        vm = psutil.virtual_memory()
        rows += [{"item": "ram_total_gb", "value": f"{vm.total / 2 ** 30:.1f}"},
                 {"item": "ram_available_gb", "value": f"{vm.available / 2 ** 30:.1f}"}]
    except ImportError:
        rows.append({"item": "ram", "value": "psutil not installed"})
    try:
        import torch

        rows += [{"item": "torch", "value": torch.__version__},
                 {"item": "cuda_available", "value": str(torch.cuda.is_available())},
                 {"item": "cuda_device", "value": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}]
    except ImportError:
        rows.append({"item": "torch", "value": "NOT INSTALLED — required from notebook 03"})
    return rows


def resolve_device(device: str) -> str:
    """'auto' → 'cuda' when available, else 'cpu'. An explicit 'cuda' without a GPU raises instead of falling back."""
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as e:
        if device == "auto":
            return "cpu"
        raise DependencyError("DEVICE = 'cuda' needs PyTorch, which is not installed in this environment") from e
    if torch.cuda.is_available():
        return "cuda"
    if device == "cuda":
        raise RuntimeError("DEVICE = 'cuda' but torch.cuda.is_available() is False. Set DEVICE = 'cpu' or 'auto'.")
    return "cpu"


def seed_everything(seed: int) -> list[str]:
    """Seed Python, NumPy and (if installed) PyTorch. Returns the known sources of remaining nondeterminism."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    notes = ["Polars group_by output order is not guaranteed: every builder sorts explicitly",
             "NumPy generators are created per task from the seed (no reliance on the global state)"]
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            notes.append("CUDA convolution and scatter kernels can be nondeterministic; bitwise equality across GPU "
                         "runs is not promised")
        notes.append("DataLoader workers > 0 change the order of random draws")
    except ImportError:
        notes.append("PyTorch not installed: nothing to seed there")
    return notes
