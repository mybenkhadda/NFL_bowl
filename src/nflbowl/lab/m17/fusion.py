"""E17.C1 — representation fusion (runs only for representations that passed G15 and the M16 leakage audit).

Arms, all built per outer fold from the M15 fold embeddings (same rows, fold-f models):
    single:<rep>        each representation alone
    concat              both blocks standardised with TRAINING rows, concatenated (``m15.transfer.concat_embeddings``)
    reliability_weight  as concat, each block scaled by √w with w = median per-dimension ICC of that block on TRAINING-role
                        Combine attempts (outcome-free; no held-out row involved)
    late_stacking       prediction-level fusion — implemented in M18 (stacked ridge of the B4 / B5 OOF predictions with inner CV)
Scored on held-out athletes: same-position retrieval MRR (``m15.transfer``) and repeat-attempt AUC (``m15.latent``). The
fused tables are written (sequence level) so M18 can build rung B6 from them with ``se.embedding_block``.
Attention fusion is not built: by the adoption rule it is only considered when concatenation underperforms the best single
representation, and it is recorded as NOT_RUN with that reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

NAN = float("nan")


def train_icc_weight(emb: Any, kinds: Sequence[str] = ("forty",), min_athletes: int = 30) -> float:
    """Median per-dimension ICC on TRAINING-role Combine attempts (weight of a block)."""
    import polars as pl

    from ..m15 import latent as LT

    t = emb.filter(pl.col("role") == "train").with_columns(pl.lit("held_out").alias("role"))
    icc = LT.dim_icc(t, {}, kinds=kinds, min_athletes=min_athletes)
    v = icc.filter(pl.col("enough"))["icc1"].to_numpy() if icc.height else np.zeros(0)
    v = v[np.isfinite(v)]
    return float(max(np.median(v), 0.0)) if v.size else NAN


def weighted_concat(a: Any, b: Any, wa: float, wb: float) -> Any:
    import polars as pl

    from ..m15.transfer import concat_embeddings
    from ..m15.latent import zcols

    c = concat_embeddings(a, b)
    za, zb = len(zcols(a)), len(zcols(b))
    sa = np.sqrt(wa) if np.isfinite(wa) else 1.0
    sb = np.sqrt(wb) if np.isfinite(wb) else 1.0
    return c.with_columns([(pl.col(f"z{k}") * (sa if k < za else sb)).alias(f"z{k}") for k in range(za + zb)])


def evaluate_arm(emb: Any, label: dict[str, Any], forty: dict[int, float], seed: int) -> dict[str, Any]:
    import polars as pl

    from ..m15 import latent as LT
    from ..m15 import transfer as T
    from ..status import ResearchBlocked

    try:
        r, _ = T.retrieval_for_embedding(emb, label, forty, arms=("unaligned",), seed=seed)
        sp = [x["rr"] for x in r if x["pool"] == "same_position"]
        mrr = float(np.mean(sp)) if sp else NAN
        nq = len(sp)
    except ResearchBlocked:
        mrr, nq = NAN, 0
    rep = LT.repeat_auc(emb, label)
    fo = rep.filter(pl.col("kind") == "forty")["repeat_auc"] if rep.height else None
    return {**label, "same_position_mrr": mrr, "n_queries": nq, "repeat_auc_forty": float(fo[0]) if fo is not None and fo.len() else NAN}


def run_fusion(m15_run_dir: str, rep_a: tuple[str, int], rep_b: tuple[str, int], athletes: Any, run: Any, folds: Sequence[int], seed: int) -> tuple[Any, dict[str, Path]]:
    """Scores per (arm, fold) and the written fused tables {arm__fold: path}."""
    import polars as pl

    from ..m16.retrieval import embedding_files, forty_map

    fa, fb = embedding_files(m15_run_dir, *rep_a), embedding_files(m15_run_dir, *rep_b)
    forty = forty_map(athletes)
    rows, written = [], {}
    for f in folds:
        if f not in fa or f not in fb:
            continue
        a, b = pl.read_parquet(fa[f]), pl.read_parquet(fb[f])
        wa, wb = train_icc_weight(a), train_icc_weight(b)
        arms = {f"single:{rep_a[0]}_d{rep_a[1]}": a, f"single:{rep_b[0]}_d{rep_b[1]}": b,
                "concat": weighted_concat(a, b, 1.0, 1.0), "reliability_weight": weighted_concat(a, b, wa, wb)}
        for arm, e in arms.items():
            rows.append({**evaluate_arm(e, {"arm": arm, "fold": f}, forty, seed + f), "weight_a": wa, "weight_b": wb})
            if not arm.startswith("single"):
                rel = f"features/fusion/{arm}__fold{f}.parquet"
                p = run.data_file(e, rel, {"rep_a": list(rep_a), "rep_b": list(rep_b), "fold": f})
                if p is not None:
                    written[f"{arm}__fold{f}"] = p
    return pl.DataFrame(rows, infer_schema_length=None), written


def adoption(scores: Any) -> Any:
    """Per fused arm: wins vs the best single arm over folds and mean MRR difference (adoption rule: ≥ 4/5 folds, Δ > 0)."""
    import polars as pl

    if scores is None or not scores.height:
        return pl.DataFrame()
    singles = scores.filter(pl.col("arm").str.starts_with("single")).group_by("fold").agg(pl.col("same_position_mrr").max().alias("best_single"))
    fused = scores.filter(~pl.col("arm").str.starts_with("single")).join(singles, on="fold")
    out = fused.group_by("arm").agg((pl.col("same_position_mrr") > pl.col("best_single")).sum().alias("folds_better"), pl.len().alias("n_folds"),
                                    (pl.col("same_position_mrr") - pl.col("best_single")).mean().alias("mean_delta_mrr"))
    out = out.with_columns(((pl.col("folds_better") >= pl.min_horizontal(pl.lit(4), pl.col("n_folds"))) & (pl.col("mean_delta_mrr") > 0)).alias("adopted"))
    att = pl.DataFrame({"arm": ["attention"], "folds_better": [None], "n_folds": [None], "mean_delta_mrr": [None], "adopted": [False]},
                       schema={"arm": pl.String, "folds_better": pl.UInt32, "n_folds": pl.UInt32, "mean_delta_mrr": pl.Float64, "adopted": pl.Boolean})
    return pl.concat([out.with_columns(pl.lit("").alias("note")),
                      att.with_columns(pl.lit("NOT_RUN: considered only if concatenation underperforms the best single representation").alias("note"))],
                     how="diagonal_relaxed")


__all__ = ["adoption", "evaluate_arm", "run_fusion", "train_icc_weight", "weighted_concat"]
