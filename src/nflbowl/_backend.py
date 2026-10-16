"""Single entry point to the research code. Every scientific definition used by nflbowl comes from these modules
unchanged; nflbowl adds orchestration, device handling, caching and artifacts around them."""

from __future__ import annotations

from typing import Any


def se() -> Any:
    """Milestone 12F shared movement encoder (families, ablations, losses, batches, splits, physics targets)."""
    from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as m

    return m


def rep() -> Any:
    """Milestone 11F representation code (sequence format, store, encoders, label-free evaluation)."""
    from nflbdb.research.advanced_movement.m11 import representation as m

    return m


def ae() -> Any:
    """Milestone 12 evaluation ladder (fold-fitted blocks, paired comparisons)."""
    from nflbdb.research.advanced_movement.m12 import advanced_evaluation as m

    return m


def model_checkpoint() -> Any:
    from nflbdb.research.advanced_movement.m11.checkpoint import ModelCheckpoint

    return ModelCheckpoint


def m11_config() -> Any:
    from nflbdb.research.advanced_movement.m11.config import M11Config

    return M11Config


__all__ = ["ae", "m11_config", "model_checkpoint", "rep", "se"]
