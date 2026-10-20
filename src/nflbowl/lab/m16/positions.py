"""E16.C6 — position-specific investigation (DB, DL, OL, TE, WR) of the ladder predictions.

Three ways to use position, compared on the SAME athletes of one position:

    global     the official ladder (one model, unpenalised position dummies) — predictions from E16.C4
    separate   the same model refitted on that position's athletes only (official engine, same outer folds), only when the
               position has ≥ ``position_min_n`` athletes with the target and ≥ 5 per fold
    pooled     the partially pooled learner (``ladder.pooled_learner``: position-specific slopes shrunk towards the global
               model) — present when ``'pooled'`` is in ``cfg.learners``

Per position: metrics of every model, paired ΔRMSE vs M1 inside the position (``m14.transfer.by_position``), separate /
pooled vs global for the same feature set, and the sample-size reading of ``position_models.position_effects``
(insufficient < 10, exploratory < 30, adequate ≥ 30). The small TE cohort is reported with its interval width; no
subgroup claim is made from a handful of athletes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")


def global_by_position(w: Any, models: Sequence[str], n_boot: int, seed: int, min_n: int = 20) -> tuple[Any, Any]:
    """(per-position metrics, per-position paired Δ vs M1 with BH inside each position) on the strict cohort."""
    from ..m14.transfer import by_position

    return by_position(w, list(models), min_n, n_boot, seed)


def sample_sizes(w: Any) -> Any:
    import polars as pl

    per = w.group_by("position_group", "fold").len().pivot(on="fold", index="position_group", values="len").fill_null(0)
    tot = w.group_by("position_group").len("n_athletes")
    out = tot.join(per.rename({c: f"n_fold_{c}" for c in per.columns if c != "position_group"}), on="position_group", how="left")
    return out.with_columns(pl.when(pl.col("n_athletes") < 10).then(pl.lit("insufficient")).when(pl.col("n_athletes") < 30).then(pl.lit("exploratory"))
                            .otherwise(pl.lit("adequate")).alias("sample_reading")).sort("position_group")


def separate_models(exp: Any, analysis: Any, target_col: str, mods: dict[str, Any], positions: Sequence[str], min_n: int, min_per_fold: int = 5) -> tuple[dict[str, Any], Any]:
    """({position: wide OOF frame with pred_<model>__sep}, eligibility table)."""
    import polars as pl

    from ... import _backend

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    out, elig = {}, []
    for p in positions:
        sub = analysis.filter((pl.col("position_group") == p) & pl.col(target_col).cast(pl.Float64).is_finite())
        per_fold = sub.group_by("fold").len()
        ok = sub.height >= min_n and per_fold.height == sub["fold"].n_unique() and (per_fold["len"].min() or 0) >= min_per_fold
        elig.append({"position_group": p, "n_athletes": sub.height, "min_per_fold": int(per_fold["len"].min() or 0) if per_fold.height else 0,
                     "status": "fitted" if ok else "BLOCKED_INSUFFICIENT_SAMPLE",
                     "reason": "" if ok else f"needs ≥ {min_n} athletes and ≥ {min_per_fold} per fold"})
        if not ok:
            continue
        res = ae.oof_ladder(sub, target_col, mods, C, C.seed)
        w = res.oof_frame()
        out[p] = w.select(pl.col("nfl_id").cast(pl.Int64), *[pl.col(f"pred_{m}").alias(f"pred_{m}__sep") for m in mods])
    return out, pl.DataFrame(elig, infer_schema_length=None)


def compare_strategies(w: Any, sep: dict[str, Any], models: Sequence[str], n_boot: int, seed: int) -> Any:
    """Per position × model: separate vs global and pooled (``<model>_pooled``) vs global — paired Δ RMSE inside the position."""
    import polars as pl

    from .. import transfer_fairness as TF

    rows = []
    for p in sorted(set(w["position_group"].to_list())):
        g = w.filter(pl.col("position_group") == p)
        if p in sep:
            g = g.join(sep[p], on="nfl_id", how="inner")
        pairs = []
        for m in models:
            if f"pred_{m}__sep" in g.columns:
                pairs.append((f"{m}__sep", m))
            if f"pred_{m}_pooled" in g.columns:
                pairs.append((f"{m}_pooled", m))
        if not pairs or g.height < 20:
            continue
        try:
            pc = TF.paired_comparisons(g, pairs, n_boot, seed, f"position:{p}")
        except TF.ComparisonError:
            continue
        rows.append(pc.with_columns(pl.lit(p).alias("position_group"), pl.lit(g.height).alias("n_position")))
    return pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()


def position_effects_all(w: Any, models: Sequence[str], n_boot: int, seed: int) -> Any:
    """``position_models.position_effects`` of every model vs M1 (sample-size labels included)."""
    import polars as pl

    from .. import position_models as PM

    parts = [PM.position_effects(w, m, "M1", n_boot, seed) for m in models if m != "M1" and f"pred_{m}" in w.columns]
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


__all__ = ["compare_strategies", "global_by_position", "position_effects_all", "sample_sizes", "separate_models"]
