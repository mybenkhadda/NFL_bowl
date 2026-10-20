"""Unified five-fold evaluation registry (M16, reused by M18): one long table of every (experiment, target, model, cohort,
fold, metric) value, built only from computed metric / fold tables, plus the concatenated OOF predictions."""

from __future__ import annotations

from typing import Any, Sequence


def registry_rows(experiment: str, metrics: Any, folds: Any) -> Any:
    import polars as pl

    parts = []
    if metrics is not None and metrics.height:
        num = [c for c, t in metrics.schema.items() if t.is_numeric() and c not in ("n_folds",)]
        parts.append(metrics.unpivot(index=[c for c in ("target_id", "cohort", "model") if c in metrics.columns], on=num, variable_name="metric",
                                     value_name="value").with_columns(pl.lit("all").alias("fold")))
    if folds is not None and folds.height:
        num = [c for c, t in folds.schema.items() if t.is_numeric() and c not in ("fold",)]
        parts.append(folds.with_columns(pl.col("fold").cast(pl.String)).unpivot(index=[c for c in ("target_id", "model", "fold") if c in folds.columns], on=num,
                                                                                variable_name="metric", value_name="value").with_columns(pl.lit("strict_common").alias("cohort")))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed").with_columns(pl.lit(experiment).alias("experiment"), pl.col("value").cast(pl.Float64))


def evaluation_registry(experiment: str, results: Sequence[dict[str, Any]]) -> tuple[Any, Any]:
    """(registry, OOF long) over every evaluated target."""
    import polars as pl

    reg = [registry_rows(experiment, r.get("metrics"), r.get("folds")) for r in results]
    reg = [x for x in reg if x.height]
    oof = [r["long"] for r in results if r.get("long") is not None]
    return (pl.concat(reg, how="diagonal_relaxed") if reg else pl.DataFrame(), pl.concat(oof, how="diagonal_relaxed") if oof else pl.DataFrame())


__all__ = ["evaluation_registry", "registry_rows"]
