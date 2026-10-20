"""E17.D1 — context-conditioned cross-domain alignment, simplest maps first (Procrustes → CORAL → OT → InfoNCE).

Every map is FITTED on TRAINING athletes of the fold (athlete means, or sequence pairs) and APPLIED to held-out Combine
means; scoring = same-position retrieval MRR on held-out athletes (``m15.transfer.score_queries``) plus the RBF MMD² between
mapped held-out Combine means and held-out NFL means (median-heuristic bandwidth). Non-orthogonal maps are also checked
for damage (repeat-attempt AUC of mapped held-out Combine attempts, ``m15.transfer.damage_check``):

    unaligned        identity
    procrustes       orthogonal map (``m15.transfer.procrustes_map``)
    coral            whiten with the training Combine covariance, recolour with the training NFL covariance (+ means)
    ot_barycentric   entropic OT (Sinkhorn, uniform weights) between training Combine and NFL means; a new point moves by
                     the mean displacement of its k nearest training Combine means (out-of-sample barycentric map)
    infonce_context  linear heads trained with InfoNCE on SEQUENCE pairs of training athletes that share an entry-speed
                     stratum (context-matched positives), negatives drawn within position — needs both training opt-ins
Adoption: an arm is preferred to Procrustes only if its same-position MRR is higher in ≥ 4/5 folds and it is not destructive.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..status import BLOCKED_MISSING_INPUT, NOT_RUN, ResearchBlocked

NAN = float("nan")


def coral_map(Ct: np.ndarray, Nt: np.ndarray, eps: float = 1e-3) -> Callable[[np.ndarray], np.ndarray]:
    mc, mn = Ct.mean(0), Nt.mean(0)
    Sc = np.cov(Ct, rowvar=False) + eps * np.eye(Ct.shape[1])
    Sn = np.cov(Nt, rowvar=False) + eps * np.eye(Nt.shape[1])

    def msqrt(S: np.ndarray, inv: bool = False) -> np.ndarray:
        w, v = np.linalg.eigh(S)
        w = np.clip(w, eps, None)
        return (v * (w ** (-0.5 if inv else 0.5))) @ v.T

    A = msqrt(Sc, inv=True) @ msqrt(Sn)
    return lambda C: (C - mc) @ A + mn


def sinkhorn(a: np.ndarray, b: np.ndarray, M: np.ndarray, reg: float, n_iter: int = 500) -> np.ndarray:
    K = np.exp(-M / reg)
    u = np.ones(a.size) / a.size
    for _ in range(n_iter):
        v = b / np.maximum(K.T @ u, 1e-300)
        u = a / np.maximum(K @ v, 1e-300)
    return u[:, None] * K * v[None, :]


def ot_map(Ct: np.ndarray, Nt: np.ndarray, k: int = 5, reg_scale: float = 0.05) -> Callable[[np.ndarray], np.ndarray]:
    M = ((Ct[:, None, :] - Nt[None, :, :]) ** 2).sum(-1)
    reg = reg_scale * float(np.median(M)) if np.median(M) > 0 else 1.0
    P = sinkhorn(np.full(Ct.shape[0], 1 / Ct.shape[0]), np.full(Nt.shape[0], 1 / Nt.shape[0]), M, reg)
    bary = (P @ Nt) / np.maximum(P.sum(1, keepdims=True), 1e-300)
    disp = bary - Ct

    def f(C: np.ndarray) -> np.ndarray:
        D = ((C[:, None, :] - Ct[None, :, :]) ** 2).sum(-1)
        nn = np.argsort(D, axis=1)[:, :k]
        return C + disp[nn].mean(1)
    return f


def mmd2(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    h = float(np.median(D[D > 0])) if (D > 0).any() else 1.0
    K = np.exp(-D / h)
    n = X.shape[0]
    kxx, kyy, kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def context_pairs(emb: Any, strata: Any, seed: int, max_pairs: int = 20000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-role (Combine seq, NFL seq) pairs of the same athlete in the same entry stratum; returns (Zc, Zn, position)."""
    import polars as pl

    from ..m15.latent import zcols

    zc = zcols(emb)
    t = emb.filter(pl.col("role") == "train").join(strata.select("idx", "s_entry"), on="idx", how="inner")
    c = t.filter(pl.col("domain") == "combine")
    n = t.filter(pl.col("domain") == "nfl")
    j = c.select("nfl_id", "s_entry", "position_group", *zc).join(n.select("nfl_id", "s_entry", *zc), on=["nfl_id", "s_entry"], suffix="_n")
    if j.height > max_pairs:
        j = j.sample(max_pairs, seed=seed)
    return j.select(zc).to_numpy().astype(float), j.select([f"{x}_n" for x in zc]).to_numpy().astype(float), j["position_group"].fill_null("?").to_numpy()


def infonce_context(emb: Any, strata: Any, seed: int, training_enabled: bool) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """(Combine head, NFL head) into the shared space, trained on context-matched training sequence pairs."""
    if not training_enabled:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "context-matched InfoNCE trains neural heads: set run_training and allow_new_training", missing=["run_training"])
    from ..m15.transfer import infonce_heads

    Zc, Zn, _pos = context_pairs(emb, strata, seed)
    if Zc.shape[0] < 50:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"only {Zc.shape[0]} context-matched training pairs", missing=["context pairs"])
    return infonce_heads(Zc, Zn, Zc.shape[1], seed)


def run_alignment(emb: Any, strata: Any | None, label: dict[str, Any], forty: dict[int, float], seed: int, training_enabled: bool) -> tuple[list[dict], list[dict]]:
    """(per-arm summary rows, damage rows) for one representation × fold."""
    from ..m15 import transfer as T

    ids, pos, C, N = T.domain_pairs(emb, "held_out")
    _, _, Ct, Nt = T.domain_pairs(emb, "train")
    if ids.size < 10 or Ct.shape[0] < Ct.shape[1] + 5:
        return [{**label, "arm": "all", "status": "BLOCKED_INSUFFICIENT_SAMPLE"}], []
    pools = {"same_position": T.pool_members(ids, pos, "same_position", forty)}
    maps: dict[str, Any] = {"unaligned": lambda X: X, "procrustes": T.procrustes_map(Ct, Nt), "coral": coral_map(Ct, Nt), "ot_barycentric": ot_map(Ct, Nt)}
    rows, damage = [], []
    for arm, f in maps.items():
        Q = f(C)
        sc = T.score_queries(Q, N, ids, pos, pools, {**label, "arm": arm})
        rows.append({**label, "arm": arm, "status": "COMPLETED", "same_position_mrr": float(np.mean([s["rr"] for s in sc])) if sc else NAN, "mmd2": mmd2(Q, N), "n": len(sc)})
        if arm in ("coral", "ot_barycentric"):
            damage.append(T.damage_check(emb, {**label, "arm": arm}, f))
    try:
        if strata is None:
            raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no context strata supplied", missing=["strata"])
        fc, fn = infonce_context(emb, strata, seed, training_enabled)
        Qs, Gs = fc(C), fn(N)
        sc = T.score_queries(Qs, Gs, ids, pos, pools, {**label, "arm": "infonce_context"})
        rows.append({**label, "arm": "infonce_context", "status": "COMPLETED", "same_position_mrr": float(np.mean([s["rr"] for s in sc])) if sc else NAN,
                     "mmd2": mmd2(Qs, Gs), "n": len(sc), "note": "both domains mapped by their own head into the shared space"})
        damage.append(T.damage_check(emb, {**label, "arm": "infonce_context"}, fc))
    except ResearchBlocked as e:
        rows.append({**label, "arm": "infonce_context", "status": NOT_RUN if "run_training" in e.reason else e.status, "note": e.reason})
    return rows, damage


def adoption(summary: Any, damage: Any) -> Any:
    """Per arm: folds where MRR beats Procrustes, mean Δ, destructive flag → adopted (≥ 4/5 folds, Δ > 0, not destructive)."""
    import polars as pl

    s = summary.filter(pl.col("status") == "COMPLETED")
    if not s.height:
        return pl.DataFrame()
    pr = s.filter(pl.col("arm") == "procrustes").select("rep", "fold", pl.col("same_position_mrr").alias("mrr_procrustes"))
    j = s.join(pr, on=["rep", "fold"])
    out = j.group_by("rep", "arm").agg((pl.col("same_position_mrr") > pl.col("mrr_procrustes")).sum().alias("folds_better"), pl.len().alias("n_folds"),
                                       (pl.col("same_position_mrr") - pl.col("mrr_procrustes")).mean().alias("mean_delta_vs_procrustes"),
                                       pl.col("mmd2").mean().alias("mean_mmd2"))
    if damage is not None and damage.height:
        dd = damage.group_by("rep", "arm").agg(pl.col("destructive").any().alias("destructive"))
        out = out.join(dd, on=["rep", "arm"], how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Boolean).alias("destructive"))
    return out.with_columns(((pl.col("folds_better") >= pl.min_horizontal(pl.lit(4), pl.col("n_folds"))) & (pl.col("mean_delta_vs_procrustes") > 0) &
                             ~pl.col("destructive").fill_null(False) & (pl.col("arm") != "procrustes")).alias("adopted_over_procrustes"))


__all__ = ["adoption", "context_pairs", "coral_map", "infonce_context", "mmd2", "ot_map", "run_alignment", "sinkhorn"]
