"""Input schema validation of a data bundle. Fails loudly with the column / shape that is wrong."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

CHANNELS = ("speed", "a_tan", "a_n", "yaw", "dt", "heading_valid")      # m11.representation.CHANNELS
META_REQUIRED = ("idx", "seq_key", "domain", "nfl_id", "kind", "position_group")
DOMAINS = ("combine", "nfl")
FOLD_REQUIRED = ("nfl_id", "fold")
M1_COLS = ("draft_year", "height", "weight", "forty", "ten")             # m12.advanced_evaluation.M1_COLS
TRANSFER_TARGET = "nfl.q90_shared"                                       # 12C frozen primary target
ATHLETE_REQUIRED = ("nfl_id", "position_group", "linked", *M1_COLS)


class SchemaError(ValueError):
    """The bundle does not match the expected schema."""


def validate_arrays(X: Any, M: Any, n_meta: int) -> None:
    if getattr(X, "ndim", None) != 3:
        raise SchemaError(f"X must be (N, L, C); got shape {getattr(X, 'shape', None)}")
    if X.shape[2] != len(CHANNELS):
        raise SchemaError(f"X has {X.shape[2]} channels; expected {len(CHANNELS)} {CHANNELS}")
    if X.dtype != np.float32:
        raise SchemaError(f"X must be float32, got {X.dtype}")
    if tuple(M.shape) != tuple(X.shape[:2]):
        raise SchemaError(f"M shape {M.shape} does not match X {X.shape[:2]}")
    if M.dtype != np.bool_:
        raise SchemaError(f"M must be bool, got {M.dtype}")
    if X.shape[0] != n_meta:
        raise SchemaError(f"X holds {X.shape[0]} sequences but meta has {n_meta} rows")


def validate_meta(meta: Any) -> None:
    import polars as pl

    missing = [c for c in META_REQUIRED if c not in meta.columns]
    if missing:
        raise SchemaError(f"meta.parquet lacks columns {missing}")
    if meta["nfl_id"].null_count():
        raise SchemaError("meta.nfl_id has nulls: every sequence must belong to an athlete")
    if not meta["nfl_id"].dtype.is_integer():
        raise SchemaError(f"meta.nfl_id must be an integer column, got {meta['nfl_id'].dtype}")
    bad = sorted(set(meta["domain"].unique().to_list()) - set(DOMAINS))
    if bad:
        raise SchemaError(f"meta.domain has unexpected values {bad}; allowed {DOMAINS}")
    if not meta["idx"].cast(pl.Int64).equals(pl.Series("idx", np.arange(meta.height), dtype=pl.Int64)):
        raise SchemaError("meta.idx must be 0..N-1 in row order (row i describes X[i])")
    # athlete position must be constant per athlete: retrieval candidates are same-position athletes
    per = meta.group_by("nfl_id").agg(pl.col("position_group").n_unique().alias("k")).filter(pl.col("k") > 1)
    if per.height:
        raise SchemaError(f"{per.height} athletes have more than one position_group in meta (e.g. {per['nfl_id'].head(3).to_list()})")


def validate_folds(folds: Any, outer_folds: int | None = None) -> None:
    missing = [c for c in FOLD_REQUIRED if c not in folds.columns]
    if missing:
        raise SchemaError(f"folds.parquet lacks columns {missing}")
    if folds["nfl_id"].n_unique() != folds.height:
        raise SchemaError("folds.parquet must hold one row per athlete (duplicate nfl_id)")
    f = folds["fold"].drop_nulls()
    if outer_folds is not None and f.len() and (int(f.min()) < 0 or int(f.max()) >= outer_folds):
        raise SchemaError(f"folds must lie in [0, {outer_folds}); found [{f.min()}, {f.max()}]")


def validate_athletes(athletes: Any, require_target: bool = False) -> list[str]:
    """Returns the list of missing optional columns (transfer needs them all)."""
    if athletes.select("nfl_id").n_unique() != athletes.height:
        raise SchemaError("athletes.parquet must hold one row per athlete")
    want: Sequence[str] = (*ATHLETE_REQUIRED, TRANSFER_TARGET) if require_target else ATHLETE_REQUIRED
    missing = [c for c in want if c not in athletes.columns]
    if require_target and missing:
        raise SchemaError(f"athletes.parquet lacks {missing} (needed for the NFL-transfer ladder)")
    return missing


__all__ = ["ATHLETE_REQUIRED", "CHANNELS", "DOMAINS", "M1_COLS", "META_REQUIRED", "SchemaError", "TRANSFER_TARGET", "validate_arrays",
           "validate_athletes", "validate_folds", "validate_meta"]
