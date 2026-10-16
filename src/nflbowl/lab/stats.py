"""Statistical helpers of the lab (NumPy / polars only).

* athlete-CLUSTER bootstrap for statistics over repeated sequences (resample athletes, keep all their rows) — never an
  ordinary row bootstrap on repeated sequence data;
* chance baselines that depend on the candidate pool size (AUC, recall@K, MRR, median rank);
* paired comparisons inside arbitrary athlete subgroups, built on ``evaluation.paired_inductive.paired_bootstrap``
  (identical resampled athletes for every representation, fold-stratified);
* multiplicity: Benjamini–Hochberg via the research code (``ae.bh``);
* salted pseudonyms for figures / reports.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

import numpy as np

NAN = float("nan")


# ====================================================================== cluster bootstrap
def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int, seed: int,
                      ci: float = 0.95) -> dict[str, float]:
    """Statistic of ``values`` with a percentile interval from resampling CLUSTERS (athletes) with replacement."""
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    out = {"estimate": float(stat(v)) if v.size else NAN, "lo": NAN, "hi": NAN, "n_obs": int(v.size), "n_clusters": int(np.unique(c).size)}
    if out["n_clusters"] < 10 or n_boot <= 0:
        return out
    uniq, inv = np.unique(c, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(uniq.size)]
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        bs[b] = stat(v[np.concatenate([groups[k] for k in pick])])
    a = (1 - ci) / 2
    out["lo"], out["hi"] = (float(x) for x in np.nanquantile(bs, [a, 1 - a]))
    return out


def cluster_bootstrap_diff(va: np.ndarray, ca: np.ndarray, vb: np.ndarray, cb: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int, seed: int,
                           ci: float = 0.95) -> dict[str, float]:
    """stat(a) − stat(b) for two groups of repeated observations; each group's clusters resampled independently
    (athletes appear in both domains, so this interval is conservative relative to a paired design)."""
    va, ca, vb, cb = np.asarray(va, float), np.asarray(ca), np.asarray(vb, float), np.asarray(cb)
    ka, kb = np.isfinite(va), np.isfinite(vb)
    va, ca, vb, cb = va[ka], ca[ka], vb[kb], cb[kb]
    out = {"a": float(stat(va)) if va.size else NAN, "b": float(stat(vb)) if vb.size else NAN, "diff": NAN, "lo": NAN, "hi": NAN,
           "n_a": int(va.size), "n_b": int(vb.size), "clusters_a": int(np.unique(ca).size), "clusters_b": int(np.unique(cb).size)}
    if not (va.size and vb.size):
        return out
    out["diff"] = out["a"] - out["b"]
    if min(out["clusters_a"], out["clusters_b"]) < 10 or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)

    def prep(v: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        u, inv = np.unique(c, return_inverse=True)
        return u, [np.flatnonzero(inv == k) for k in range(u.size)]

    ua, ga = prep(va, ca)
    ub, gb = prep(vb, cb)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ia = np.concatenate([ga[k] for k in rng.integers(0, ua.size, ua.size)])
        ib = np.concatenate([gb[k] for k in rng.integers(0, ub.size, ub.size)])
        d[b] = stat(va[ia]) - stat(vb[ib])
    a = (1 - ci) / 2
    out["lo"], out["hi"] = (float(x) for x in np.nanquantile(d, [a, 1 - a]))
    return out


def athlete_bootstrap_ci(x: np.ndarray, n_boot: int, seed: int, stat: Callable[[np.ndarray], float] = np.mean, ci: float = 0.95) -> tuple[float, float, float]:
    """One value per athlete → (estimate, lo, hi) by the ordinary athlete bootstrap (athletes are the units)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return (float(stat(x)) if x.size else NAN, NAN, NAN)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    bs = np.array([stat(x[i]) for i in idx])
    a = (1 - ci) / 2
    lo, hi = np.quantile(bs, [a, 1 - a])
    return float(stat(x)), float(lo), float(hi)


# ====================================================================== chance baselines
def chance_recall_at_k(k: int, n_candidates: np.ndarray) -> float:
    """Expected recall@K when the own match is uniformly ranked among (n + 1) items, averaged over queries."""
    n = np.asarray(n_candidates, float)
    return float(np.mean(np.minimum(k, n + 1) / (n + 1))) if n.size else NAN


def chance_mrr(n_candidates: np.ndarray) -> float:
    n = np.asarray(n_candidates, int)
    if not n.size:
        return NAN
    h = {m: float(np.sum(1.0 / np.arange(1, m + 2))) for m in np.unique(n)}
    return float(np.mean([h[m] / (m + 1) for m in n]))


def chance_median_rank(n_candidates: np.ndarray) -> float:
    n = np.asarray(n_candidates, float)
    return float(np.median((n + 2) / 2)) if n.size else NAN


# ====================================================================== multiplicity
def bh(p: Sequence[float]) -> list[float]:
    from .. import _backend

    return _backend.ae().bh(list(p))


def reading(lo: float, hi: float) -> str:
    """Interval reading in the research-code vocabulary (never 'no effect' from an interval including 0)."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "insufficient_for_interval"
    if lo > 0:
        return "supported_positive"
    if hi < 0:
        return "supported_negative"
    return "inconclusive"


# ====================================================================== subgroup paired comparisons
def paired_in_subgroups(scores: Any, models: Sequence[str], reference: str, groups: Any, group_col: str, metric: str = "auc", n_boot: int = 1000,
                        seed: int = 0, min_n: int = 10, adequate_n: int = 30) -> Any:
    """``paired_bootstrap`` repeated inside each level of ``group_col`` (a column of ``groups``: nfl_id, fold → level).
    Pools are those of the full analysis (subgrouping selects queries, it never changes candidates). The bootstrap
    share of replicates ≤ 0 is not a p-value, so no multiplicity adjustment is applied; the number of levels compared
    is reported and subgroup results are descriptive."""
    import polars as pl

    from ..evaluation import paired_inductive as pi

    s = scores.join(groups.select("nfl_id", "fold", group_col), on=["nfl_id", "fold"], how="left")
    out = []
    for lvl in sorted(s[group_col].drop_nulls().unique().to_list(), key=str):
        sub = s.filter(pl.col(group_col) == lvl).drop(group_col)
        if not sub.filter(pl.col("status") == pi.SCORED).height:
            continue
        try:
            r = pi.paired_bootstrap(sub, models, reference, metric=metric, n_boot=n_boot, seed=seed, min_n=min_n, adequate_n=adequate_n)
        except ValueError:
            continue
        out.append(r.with_columns(pl.lit(group_col).alias("subgroup_variable"), pl.lit(str(lvl)).alias("subgroup")))
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


# ====================================================================== misc
def pseudonym(nfl_id: Any, salt: str) -> str:
    return "A-" + hashlib.sha256(f"{salt}:{int(nfl_id)}".encode()).hexdigest()[:6].upper()


def effective_rank(Z: np.ndarray) -> dict[str, float]:
    Z = np.asarray(Z, float)
    Z = Z[np.isfinite(Z).all(1)]
    if Z.shape[0] < 3:
        return {"effective_rank": NAN, "participation_ratio": NAN, "pc1_share": NAN}
    lam = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    tot = lam.sum()
    if tot <= 0:
        return {"effective_rank": 0.0, "participation_ratio": 0.0, "pc1_share": NAN}
    p = lam / tot
    p = p[p > 0]
    return {"effective_rank": float(np.exp(-(p * np.log(p)).sum())), "participation_ratio": float(tot ** 2 / (lam ** 2).sum()), "pc1_share": float(lam[0] / tot)}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from .. import _backend  # noqa: F401  (keeps one import path for the research helpers)
    from nflbdb.research.advanced_movement.m11 import evaluation as ev

    return float(ev.spearman(np.asarray(a, float), np.asarray(b, float)))


def partial_spearman(a: np.ndarray, b: np.ndarray, Z: np.ndarray) -> float:
    from nflbdb.research.advanced_movement.m11 import evaluation as ev

    return float(ev.partial_spearman(np.asarray(a, float), np.asarray(b, float), np.asarray(Z, float)))


__all__ = ["athlete_bootstrap_ci", "bh", "chance_median_rank", "chance_mrr", "chance_recall_at_k", "cluster_bootstrap", "cluster_bootstrap_diff",
           "effective_rank", "paired_in_subgroups", "partial_spearman", "pseudonym", "reading", "spearman"]
