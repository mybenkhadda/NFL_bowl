"""Seeding, RNG-state capture / restore (for exact resume) and code-version facts."""

from __future__ import annotations

import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_everything(seed: int) -> None:
    """Python, NumPy global and PyTorch (CPU + CUDA) seeds. The research code also creates explicit generators from the
    seed, so this only covers library-internal draws."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def capture_rng(np_rng: np.random.Generator | None = None, torch_gen: Any = None) -> dict[str, Any]:
    """Every RNG stream the trainer consumes: the NumPy batch generator, the CPU torch generator used for masks / crops,
    and the global torch CPU / CUDA states. Stored in each checkpoint so a resumed run continues the same streams."""
    out: dict[str, Any] = {"python": random.getstate(), "numpy_global": np.random.get_state()}
    if np_rng is not None:
        out["numpy_batches"] = np_rng.bit_generator.state
    try:
        import torch

        out["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            out["torch_cuda"] = torch.cuda.get_rng_state_all()
        if torch_gen is not None:
            out["torch_generator"] = torch_gen.get_state()
    except ImportError:
        pass
    return out


def restore_rng(state: dict[str, Any] | None, np_rng: np.random.Generator | None = None, torch_gen: Any = None) -> bool:
    """Inverse of ``capture_rng``. Returns False when no state was stored (checkpoint written by the original notebook)."""
    if not state:
        return False
    if "python" in state:
        random.setstate(state["python"])
    if "numpy_global" in state:
        np.random.set_state(state["numpy_global"])
    if np_rng is not None and "numpy_batches" in state:
        np_rng.bit_generator.state = state["numpy_batches"]
    try:
        import torch

        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available() and len(state["torch_cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        if torch_gen is not None and "torch_generator" in state:
            torch_gen.set_state(state["torch_generator"])
    except ImportError:
        pass
    return True


def _git_commit(start: Path) -> str | None:
    """Commit of the source tree (editable installs only), read from ``.git`` files — no subprocess."""
    for d in [start, *start.parents]:
        g = d / ".git"
        if g.is_dir():
            head = (g / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref:"):
                ref = g / head.split(" ", 1)[1].strip()
                if ref.is_file():
                    return ref.read_text(encoding="utf-8").strip()
                packed = g / "packed-refs"
                if packed.is_file():
                    name = head.split(" ", 1)[1].strip()
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.endswith(" " + name):
                            return line.split(" ", 1)[0]
                return None
            return head
    return None


def code_version() -> dict[str, Any]:
    from .. import __version__

    out: dict[str, Any] = {"nflbowl": __version__, "python": platform.python_version(), "numpy": np.__version__}
    try:
        import polars as pl

        out["polars"] = pl.__version__
    except ImportError:
        pass
    try:
        import torch

        out["torch"] = torch.__version__
        out["torch_cuda_build"] = getattr(torch.version, "cuda", None)
    except ImportError:
        out["torch"] = None
    try:
        out["git_commit"] = _git_commit(Path(__file__).resolve().parent)
    except OSError:
        out["git_commit"] = None
    return out


__all__ = ["capture_rng", "code_version", "restore_rng", "seed_everything", "utc_now"]
