"""Model checkpoints.

Two layers:
* EPOCH checkpoints — ``nflbdb`` ``ModelCheckpoint`` (immutable shards ``shard_epoch=NNNN.pt`` + sha256 sidecar, one
  directory per configuration hash; a shard is never overwritten; a shard without a verified sidecar is ignored).
  ``latest`` = highest verified epoch. nflbowl adds the GradScaler state and every RNG stream to each shard so a resumed
  run continues exactly where it stopped.
* BEST export — ``checkpoints/<model>/<tag>/best.pt``: the early-stopping best weights + ``EncConfig`` + signature, written
  atomically once training finishes. Embedding in a new session loads this file; no optimizer state needed.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .. import _backend
from ..artifacts.store import atomic_json, atomic_write
from ..devices import require_torch


def best_path(root: Path, model: str, tag: str) -> Path:
    return Path(root) / model / tag / "best.pt"


def export_best(path: Path, state_dict: dict[str, Any], enc_config: Any, signature: dict[str, Any]) -> Path:
    torch = require_torch("checkpoint export")
    payload = {"format": "nflbowl.best.v1", "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()}, "enc_config": asdict(enc_config),
               "signature": signature}
    atomic_write(path, lambda p: torch.save(payload, p), overwrite=True)
    atomic_json({k: v for k, v in payload.items() if k != "state_dict"}, path.with_suffix(".json"), overwrite=True)
    return path


def load_best(path: Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Rebuild the ``SharedEncoder`` from ``best.pt`` (weights only; strict key match)."""
    torch = require_torch("checkpoint loading")
    se = _backend.se()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "nflbowl.best.v1":
        raise ValueError(f"{path} is not an nflbowl best-model export")
    e = se.EncConfig(**payload["enc_config"])
    model = se.SharedEncoder(e)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload["signature"]


def epoch_checkpoint(root: Path, name: str, signature_cfg: dict[str, Any], mode: str) -> Any:
    return _backend.model_checkpoint()(root, name, signature_cfg, mode)


__all__ = ["best_path", "epoch_checkpoint", "export_best", "load_best"]
