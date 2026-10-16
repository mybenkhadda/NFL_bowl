"""Per-sequence derived arrays (12F Part 3), computed once per bundle and cached as Parquet.

    physics targets P (N, 6)   se.physics_targets   — max speed, speed gain, mean a_tan, max |a_n|, |heading change|, duration
    state_id                   se.state_ids         — start-speed bin × accelerating flag (matched-state alignment)
    category_id                se.category_ids      — drill family ↔ NFL demand category
    domain_id                  0 Combine · 1 NFL
    probe scalars              rep.physical_scalars — targets of the physical probes

All values come from the research functions; the cache only avoids recomputing them (Python loops over every sequence)
in each new session. Parquet stores float32 / float64 exactly, so a cached value equals a recomputed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _backend
from ..artifacts.manifest import Manifest
from .cache import cached_frame

PHYS_PREFIX = "pt_"
PROBE_COLS = ("max_speed", "mean_a_tan", "max_abs_a_n", "abs_heading_change_rad")
DERIVED_VERSION = 1


@dataclass
class Derived:
    P: np.ndarray            # (N, 6) float32, NaN where unsupported
    state: np.ndarray        # (N,) int64
    cat: np.ndarray          # (N,) int64
    dom: np.ndarray          # (N,) int64
    phys: Any                # polars DataFrame idx + PROBE_COLS (the `phys` argument of rep.evaluation_row)


def _compute(X: np.ndarray, M: np.ndarray, meta: Any) -> Any:
    import polars as pl

    se, rep = _backend.se(), _backend.rep()
    P = se.physics_targets(X, M)
    state, cat = se.state_ids(X, M), se.category_ids(meta)
    dom = (meta["domain"] == "nfl").to_numpy().astype(np.int64)
    phys = rep.physical_scalars(X, M)
    return pl.DataFrame({"idx": np.arange(X.shape[0], dtype=np.int64), **{f"{PHYS_PREFIX}{k}": P[:, j] for j, k in enumerate(se.PHYS)},
                         "state_id": state, "category_id": cat, "domain_id": dom}).join(phys.with_columns(pl.col("idx").cast(pl.Int64)), on="idx", how="left").sort("idx")


def derive(bundle: Any, manifest: Manifest, directory: Any, reuse: bool = True) -> tuple[Derived, dict[str, Any]]:
    se = _backend.se()
    deps = {"bundle": bundle.fingerprint, "version": DERIVED_VERSION, "phys": list(se.PHYS)}
    df, path, hit = cached_frame(manifest, "processed", "sequence_targets", deps, directory, lambda: _compute(bundle.X, bundle.M, bundle.meta), reuse)
    P = np.column_stack([df[f"{PHYS_PREFIX}{k}"].to_numpy().astype(np.float32) for k in se.PHYS])
    d = Derived(P=P, state=df["state_id"].to_numpy().astype(np.int64), cat=df["category_id"].to_numpy().astype(np.int64),
                dom=df["domain_id"].to_numpy().astype(np.int64), phys=df.select("idx", *PROBE_COLS))
    return d, {"path": str(path), "from_cache": hit, "deps": deps}


__all__ = ["Derived", "PROBE_COLS", "derive"]
