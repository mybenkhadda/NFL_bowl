"""Resolve warehouse tables to DuckDB / Polars references.

Explicit file lists are used instead of globs so hidden staging files (``.name.tmp``) from an
in-flight atomic write are never read.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import Phase01Config
from .utils import sql_path

LAYERS = ("bronze", "silver", "gold")


class LayerRefs:
    def __init__(self, cfg: Phase01Config):
        self.cfg = cfg

    def path(self, layer: str, name: str) -> Path:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}")
        return getattr(self.cfg.paths, layer) / name

    def files(self, layer: str, name: str) -> list[Path]:
        base = self.path(layer, name)
        single = base.with_suffix(".parquet")
        if single.is_file():
            return [single]
        if not base.is_dir():
            return []
        return sorted(
            p for p in base.rglob("*.parquet")
            if not any(part.startswith(".") for part in p.relative_to(base).parts)
        )

    def exists(self, layer: str, name: str) -> bool:
        return bool(self.files(layer, name))

    def sql(self, layer: str, name: str) -> str:
        """``read_parquet([...], hive_partitioning = true)`` expression for DuckDB."""
        files = self.files(layer, name)
        if not files:
            raise FileNotFoundError(f"{layer}.{name} has no parquet files — build it first")
        listing = ", ".join(f"'{sql_path(f)}'" for f in files)
        return f"read_parquet([{listing}], hive_partitioning = true, union_by_name = true)"

    def scan(self, layer: str, name: str) -> pl.LazyFrame:
        files = self.files(layer, name)
        if not files:
            raise FileNotFoundError(f"{layer}.{name} has no parquet files — build it first")
        return pl.scan_parquet(files, hive_partitioning=True)

    def num_rows(self, layer: str, name: str) -> int:
        import pyarrow.parquet as pq

        return sum(pq.ParquetFile(f).metadata.num_rows for f in self.files(layer, name))

    def num_bytes(self, layer: str, name: str) -> int:
        return sum(f.stat().st_size for f in self.files(layer, name))


def sample_clause(pct: float | None, seed: int) -> str:
    """Deterministic Bernoulli row sample (DuckDB). Empty string = full table."""
    if not pct or pct >= 100:
        return ""
    return f" USING SAMPLE {pct}% (bernoulli, {seed})"


def hash_sample_predicate(cols: tuple[str, ...] | list[str], pct: float | None) -> str:
    """Keeps WHOLE trajectories: a stable hash of the trajectory key decides membership, so
    lag/lead-based checks on the sample see complete sequences."""
    if not pct or pct >= 100:
        return "TRUE"
    key = " || '|' || ".join(f"coalesce(CAST({c} AS VARCHAR), '∅')" for c in cols)
    return f"(hash({key}) % 10000) < {int(round(pct * 100))}"
