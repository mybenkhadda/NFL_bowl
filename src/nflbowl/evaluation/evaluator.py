"""Label-free evaluation of a representation at three levels.

``standard``  the 12F evaluation row exactly (``se.evaluate`` → ``rep.evaluation_row``): within-domain retrieval,
              cross-domain athlete retrieval (bootstrap = max(200, n_boot // 5)), matched retrieval, domain AUC / MMD,
              athlete-grouped physical probes. Full candidate universe, all athletes.
``full``      standard + the shuffled-identity null (``evaluation.identity_null_perms`` replicates, notebook = 10) and its
              summary next to the observed AUC.
``quick``     DIAGNOSTIC: the same metrics on a seeded athlete subsample (``quick_athlete_share``, ≥ ``quick_min_athletes``)
              with ``quick_n_boot`` bootstrap replicates. Smaller candidate sets make retrieval AUCs NOT comparable with
              standard / full; every quick row carries ``diagnostic = True`` and ``comparable_to_full = False``.

``backend = "reference"`` calls the nflbdb functions unchanged (slow, the definitional baseline);
``backend = "fast"`` (default) uses the vectorised equivalents in ``retrieval`` / ``probes`` / ``null_tests``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .. import _backend
from .null_tests import null_summary, shuffled_identity_null
from .probes import domain_shift, probe_r2
from .retrieval import cross_domain_retrieval, matched_retrieval, within_domain_retrieval


@dataclass
class EvalResult:
    row: dict[str, Any]
    probes: Any                     # polars DataFrame
    null: np.ndarray | None = None


def evaluation_row(name: str, regime: str, E: np.ndarray, meta: Any, phys: Any, folds: dict[int, int], n_boot: int, outer_folds: int, seed: int,
                   evaluation: Any, cross_boot: int | None = None) -> tuple[dict[str, Any], Any]:
    """Same keys and values as ``rep.evaluation_row``."""
    import polars as pl

    if evaluation.backend == "reference" and cross_boot is None:
        m11c = replace(_backend.m11_config()(), n_boot=n_boot, outer_folds=outer_folds)
        return _backend.rep().evaluation_row(name, regime, E, meta, phys, folds, m11c, seed)
    w = within_domain_retrieval(E, meta)
    x = cross_domain_retrieval(E, meta, max(200, n_boot // 5) if cross_boot is None else cross_boot, seed)
    mr = matched_retrieval(E, meta, phys, seed=seed, chunk=evaluation.matched_query_chunk, approx_topk=evaluation.approx_topk,
                           similarity_device=evaluation.similarity_device)
    ds = domain_shift(E, meta, seed=seed)
    pr = probe_r2(E, meta, phys, folds, outer_folds)
    row = {"representation": name, "training_mode": regime, "within_top1": w["top1"], "within_auc": w["auc"], "cross_auc": x["auc"], "cross_auc_lo": x["auc_lo"],
           "cross_auc_hi": x["auc_hi"], "cross_top5": x["top5"], "cross_n_athletes": x["n_athletes"], **mr, **ds,
           "probe_r2_median": float(pr["r2_cv"].median()) if pr.height else float("nan")}
    return row, pr.with_columns(pl.lit(name).alias("representation"))


def athlete_subsample(meta: Any, share: float, min_athletes: int, seed: int) -> np.ndarray:
    """Row mask of a seeded athlete subsample (whole athletes: both domains, every sequence)."""
    ath = np.array(sorted(meta["nfl_id"].unique().to_list()))
    k = min(ath.size, max(min_athletes, int(math.ceil(share * ath.size))))
    keep = np.random.default_rng(seed).choice(ath, k, replace=False)
    return np.isin(meta["nfl_id"].to_numpy(), keep)


def evaluate_representation(name: str, E: np.ndarray, meta: Any, phys: Any, folds: dict[int, int], science: Any, evaluation: Any,
                            level: str = "standard", regime: str = "transductive", seed: int | None = None) -> EvalResult:
    """Evaluate one embedding matrix (rows aligned with ``meta``). Low-level entry point; ``Experiment.evaluate`` adds
    caching and artifacts."""
    import polars as pl

    seed = science.seed if seed is None else seed
    t0 = time.perf_counter()
    null = None
    if level == "quick":
        mask = athlete_subsample(meta, evaluation.quick_athlete_share, evaluation.quick_min_athletes, seed)
        idx = np.flatnonzero(mask)
        sub_meta = meta.filter(pl.Series(mask))
        sub_phys = phys.filter(pl.col("idx").is_in(idx.tolist())).sort("idx").with_columns(pl.Series("idx", np.arange(idx.size), dtype=phys["idx"].dtype))
        row, pr = evaluation_row(name, regime, E[idx], sub_meta, sub_phys, folds, science.n_boot, science.outer_folds, seed, evaluation,
                                 cross_boot=evaluation.quick_n_boot)
        row.update({"quick_athletes": int(sub_meta["nfl_id"].n_unique()), "quick_sequences": int(idx.size)})
    elif level in ("standard", "full"):
        row, pr = evaluation_row(name, regime, E, meta, phys, folds, science.n_boot, science.outer_folds, seed, evaluation)
        if level == "full":
            null = shuffled_identity_null(E, meta, evaluation.identity_null_perms, seed, backend=evaluation.backend)
            row.update(null_summary(row["cross_auc"], null))
    else:
        raise ValueError(f"level must be quick / standard / full, got {level!r}")
    row.update({"level": level, "diagnostic": level == "quick", "comparable_to_full": level != "quick", "backend": evaluation.backend,
                "exact": bool(evaluation.is_exact()), "seconds": round(time.perf_counter() - t0, 2)})
    return EvalResult(row, pr.with_columns(pl.lit(level).alias("level")) if pr.height else pr, null)


def inductive_fold_retrieval(name: str, E: np.ndarray, meta: Any, held_out: np.ndarray, fold: int, n_boot: int, seed: int, backend: str = "fast") -> dict[str, Any]:
    """12F Part 7b: cross-domain retrieval among the held-out fold's athletes only, embeddings from the encoder trained
    without them."""
    mh = meta[held_out]
    fn = _backend.rep().cross_domain_retrieval if backend == "reference" else cross_domain_retrieval
    xr = fn(E[held_out], mh, max(200, n_boot // 5), seed)
    return {"representation": name, "training_mode": "inductive", "held_out_fold": int(fold), **xr}


__all__ = ["EvalResult", "athlete_subsample", "evaluate_representation", "evaluation_row", "inductive_fold_retrieval"]
