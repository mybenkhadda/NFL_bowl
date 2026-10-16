"""F0 engineered representation: ``rep.engineered_features`` (per-channel masked mean / SD / max / min of the four
kinematic channels + valid length, 17 dims), computed once per bundle and cached as Parquet."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import _backend
from ..artifacts.manifest import Manifest
from ..data.cache import cached_frame

F0_NAME = "F0_engineered"
F0_VERSION = 1


def engineered_features(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Exactly ``rep.engineered_features`` (float32, one row per sequence)."""
    return _backend.rep().engineered_features(X, M)


def engineered_cached(bundle: Any, manifest: Manifest, directory: Any, reuse: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    import polars as pl

    deps = {"bundle": bundle.fingerprint, "version": F0_VERSION}

    def compute() -> Any:
        E = engineered_features(bundle.X, bundle.M)
        return pl.DataFrame({"idx": np.arange(E.shape[0], dtype=np.int64), **{f"f{j}": E[:, j] for j in range(E.shape[1])}})

    df, path, hit = cached_frame(manifest, "features", F0_NAME, deps, directory, compute, reuse)
    E = df.drop("idx").to_numpy().astype(np.float32)
    return E, {"path": str(path), "from_cache": hit, "deps": deps}


__all__ = ["F0_NAME", "engineered_cached", "engineered_features"]
