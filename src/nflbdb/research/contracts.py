"""Strict data contracts for Phase 03 research outputs (shared by M2.1 and the A–I studies).

A :class:`TableContract` declares exact column dtypes and a key. :func:`conform` validates a frame
against it — a missing column, an extra column, a dtype mismatch, a null or duplicated key raises
:class:`ContractError`. Builders cast explicitly; nothing is coerced silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import polars as pl

# normalised identifier dtypes (sources differ: event_id is a string id, years come from hive partitions)
KEY_DTYPES: dict[str, pl.DataType] = {"event_id": pl.String(), "nfl_id": pl.Int64(), "attempt": pl.Int64(),
                                      "draft_year": pl.Int64(), "drill": pl.String(), "game_id": pl.Int64(),
                                      "play_id": pl.Int64(), "season": pl.Int64()}


class ContractError(ValueError):
    """An output violates its declared contract."""


@dataclass(frozen=True)
class TableContract:
    name: str
    description: str
    columns: dict[str, pl.DataType]
    key: tuple[str, ...] = ()
    kind: str = "table"                     # "table" → reports/tables (CSV + Parquet) · "dataset" → warehouse
    docs: dict[str, str] = field(default_factory=dict)
    area: str = ""                          # study stores: sub-folder (tables: category · datasets: warehouse area)

    def __post_init__(self) -> None:
        if self.kind not in ("table", "dataset"):
            raise ValueError(f"{self.name}: kind must be 'table' or 'dataset', got {self.kind!r}")
        missing = [k for k in self.key if k not in self.columns]
        if missing:
            raise ValueError(f"{self.name}: key columns {missing} not declared")

    @property
    def schema(self) -> pl.Schema:
        return pl.Schema(self.columns)

    def empty(self) -> pl.DataFrame:
        return pl.DataFrame(schema=self.schema)


def conform(df: pl.DataFrame, contract: TableContract) -> pl.DataFrame:
    """Validate ``df`` against ``contract`` and return it in contract column order (no casting)."""
    have = df.schema
    missing = [c for c in contract.columns if c not in have]
    extra = [c for c in have if c not in contract.columns]
    wrong = [f"{c}: {have[c]} ≠ {t}" for c, t in contract.columns.items() if c in have and have[c] != t]
    problems = ([f"missing {missing}"] if missing else []) + ([f"extra {extra}"] if extra else []) + wrong
    if problems:
        raise ContractError(f"{contract.name}: " + "; ".join(problems))
    out = df.select(list(contract.columns))
    if contract.key and out.height:
        key = list(contract.key)
        nulls = out.select(pl.any_horizontal([pl.col(c).is_null() for c in key]).sum()).item()
        dups = out.height - out.select(pl.struct(key).n_unique()).item()
        if nulls or dups:
            raise ContractError(f"{contract.name}: key {key} has {nulls} null and {dups} duplicate rows")
    return out


def cast_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Explicit cast of identifier columns to ``KEY_DTYPES`` (strict: an uncastable value raises)."""
    return df.with_columns([pl.col(c).cast(t, strict=True) for c, t in KEY_DTYPES.items() if c in df.columns])


def upstream_state(store: Any, names: Iterable[str]) -> dict[str, Any]:
    """Recorded build state (version, build time, rows) of upstream datasets in an ``ArtifactStore`` —
    a missing mandatory dataset raises, so an analysis never runs on an unknown input."""
    mf = store.manifest_frame()
    out = {}
    for name in names:
        row = mf.filter((pl.col("kind") == "datasets") & (pl.col("name") == name)) if mf.height else mf
        if not row.height:
            raise FileNotFoundError(f"upstream dataset {name!r} not registered in "
                                    f"{store.cfg.paths.rel(store.manifest_path)} — build it first")
        r = row.row(0, named=True)
        out[name] = {"version": r["version"], "built_at": r["built_at"], "rows": r["rows"]}
    return out
