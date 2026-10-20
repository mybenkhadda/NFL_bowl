"""E15.E1–E3 — athlete-level cross-domain retrieval with controls (T08; the scoring core is reused by M16 E16.B1).

Unit = held-out athlete of an outer fold with BOTH a Combine and an NFL athlete-mean embedding (from the fold-f model).
Query = the athlete's Combine mean; gallery = NFL means of the held-out athletes in the query's POOL:

    all            every held-out athlete of the fold
    same_position  held-out athletes of the query's position group (the control the prototype lacked, roadmap § 1.2-7)
    twins          same position and |Δforty| ≤ ``twin_forty_s`` (stopwatch twins, M09 rule); queries with no twin skipped

Per query: rank of the true NFL mean (cosine similarity), reciprocal rank, top-1, top-5, AUC (share of impostors scored
below the truth) and pool size → chance top-1 = 1/(n+1), chance MRR from ``stats.chance_mrr``.

Arms (all fold-safe: anything fitted uses TRAINING athletes of the fold only):
    unaligned      raw athlete means
    procrustes     orthogonal map Combine → NFL fitted on training athletes' means
    infonce        linear heads trained with a symmetric InfoNCE loss on training athletes' (Combine, NFL) means (torch)
    random         Gaussian vectors of the same dimension (empirical chance)
    position_only  analytic: position recovered perfectly, random within position → E[top-1] = mean 1/n_pos
Paired comparisons: per-query reciprocal-rank differences, athlete bootstrap (same queries in both arms).
Damage check (E15.E3): an alignment that is not orthogonal can change within-Combine geometry; the repeat-attempt AUC is
recomputed on head-transformed held-out Combine attempts and compared with the raw embedding.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..status import BLOCKED_INSUFFICIENT_SAMPLE, ResearchBlocked
from .latent import athlete_means, repeat_auc, zcols

NAN = float("nan")
POOLS = ("all", "same_position", "twins")


def domain_pairs(emb: Any, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(ids, positions, Combine means, NFL means) of athletes with both domains in ``role`` rows."""
    import polars as pl

    zc = zcols(emb)
    c = athlete_means(emb, "combine", roles=(role,))
    n = athlete_means(emb, "nfl", roles=(role,))
    j = c.select("nfl_id", "position_group", *zc).join(n.select("nfl_id", *zc), on="nfl_id", suffix="_n").sort("nfl_id")
    return (j["nfl_id"].to_numpy(), j["position_group"].fill_null("?").to_numpy(), j.select(zc).to_numpy().astype(float),
            j.select([f"{c}_n" for c in zc]).to_numpy().astype(float))


def _unit(Z: np.ndarray) -> np.ndarray:
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def pool_members(ids: np.ndarray, pos: np.ndarray, pool: str, forty: dict[int, float] | None = None, twin_forty_s: float = 0.03) -> list[np.ndarray]:
    """Candidate indices (excluding the query itself) per query for one pool definition."""
    out = []
    for i in range(ids.size):
        if pool == "all":
            m = np.ones(ids.size, bool)
        elif pool == "same_position":
            m = pos == pos[i]
        elif pool == "twins":
            fi = (forty or {}).get(int(ids[i]))
            if fi is None or not np.isfinite(fi):
                m = np.zeros(ids.size, bool)
            else:
                f = np.array([(forty or {}).get(int(k), np.nan) for k in ids])
                m = (pos == pos[i]) & np.isfinite(f) & (np.abs(f - fi) <= twin_forty_s)
        else:
            raise ValueError(f"unknown pool {pool!r}")
        m[i] = False
        out.append(np.flatnonzero(m))
    return out


def score_queries(Q: np.ndarray, G: np.ndarray, ids: np.ndarray, pos: np.ndarray, pools: dict[str, list[np.ndarray]], label: dict[str, Any]) -> list[dict[str, Any]]:
    """Per (query, pool) retrieval record; queries with an empty pool are skipped (recorded as n_candidates = 0)."""
    S = _unit(Q) @ _unit(G).T
    rows = []
    for pool, members in pools.items():
        for i in range(ids.size):
            cand = members[i]
            if cand.size == 0:
                continue
            true = S[i, i]
            imp = S[i, cand]
            rank = int(1 + (imp > true).sum() + 0.5 * (imp == true).sum())
            rows.append({**label, "pool": pool, "nfl_id": int(ids[i]), "position_group": pos[i], "n_candidates": int(cand.size), "rank": rank, "rr": 1.0 / rank,
                         "top1": float(rank == 1), "top5": float(rank <= 5), "auc": float((imp < true).mean() + 0.5 * (imp == true).mean())})
    return rows


def summarise(scores: Any) -> Any:
    """Mean metrics per (arm, pool) with chance references (``stats.chance_*``)."""
    import polars as pl

    from ..stats import chance_mrr, chance_recall_at_k

    keys = [c for c in ("arm", "rep", "fold", "pool") if c in scores.columns]
    keys = [k for k in keys if k != "fold"]
    out = []
    for k, g in scores.group_by(keys):
        n = g["n_candidates"].to_numpy()
        out.append({**dict(zip(keys, k, strict=True)), "n_queries": g.height, "top1": float(g["top1"].mean()), "top5": float(g["top5"].mean()),
                    "mrr": float(g["rr"].mean()), "auc": float(g["auc"].mean()), "chance_top1": chance_recall_at_k(1, n), "chance_top5": chance_recall_at_k(5, n),
                    "chance_mrr": chance_mrr(n)})
    return pl.DataFrame(out, infer_schema_length=None)


def position_chance(ids: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    """Analytic E[top-1] in the ALL pool if position were recovered perfectly and the within-position pick were random."""
    n_pos = np.array([(pos == p).sum() for p in pos], float)
    return {"position_only_top1_all_pool": float(np.mean(1.0 / n_pos)) if n_pos.size else NAN, "uniform_top1_all_pool": 1.0 / ids.size if ids.size else NAN,
            "n_queries": int(ids.size), "n_positions": int(np.unique(pos).size)}


def binomial_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper–Pearson interval by bisection on the binomial CDF (no SciPy)."""
    from math import comb

    def cdf(x: int, p: float) -> float:
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))

    a = (1 - level) / 2

    def root_decreasing(f: Callable[[float], float]) -> float:
        """p with f(p) = 0 for f decreasing in p."""
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return (lo + hi) / 2

    # lower: P(X ≥ k | p) = a, increasing in p  →  a − P(X ≥ k | p) is decreasing
    lower = 0.0 if k == 0 else root_decreasing(lambda p: a - (1 - cdf(k - 1, p)))
    # upper: P(X ≤ k | p) = a, decreasing in p
    upper = 1.0 if k == n else root_decreasing(lambda p: cdf(k, p) - a)
    return float(lower), float(upper)


def prototype_reading(k: int = 7, n: int = 80, n_positions: int = 5) -> dict[str, Any]:
    """The [EXP] prototype result (7 / 80 top-1) against uniform and position-only chance (E15.E2)."""
    lo, hi = binomial_interval(k, n)
    return {"k": k, "n": n, "top1": k / n, "ci_lo": lo, "ci_hi": hi, "uniform_chance": 1 / n, "position_only_expected_balanced": n_positions / n,
            "note": "position-only expectation assumes equal position groups; the computed value per fold uses the real group sizes"}


# ====================================================================== alignment arms
def procrustes_map(Ct: np.ndarray, Nt: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    from .latent import procrustes

    mc, mn = Ct.mean(0), Nt.mean(0)
    R = procrustes(Nt - mn, Ct - mc)
    return lambda C: (C - mc) @ R + mn


def infonce_heads(Ct: np.ndarray, Nt: np.ndarray, out_dim: int, seed: int, steps: int = 1500, tau: float = 0.1, lr: float = 1e-2,
                  weight_decay: float = 1e-3) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Linear heads (Combine, NFL) → shared space, symmetric InfoNCE on training athletes (full-batch)."""
    from .models import TORCH

    if not TORCH:
        from ..status import BLOCKED_MISSING_INPUT

        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "InfoNCE heads need PyTorch", missing=["torch"])
    import torch

    torch.manual_seed(seed)
    mc, sc = Ct.mean(0), Ct.std(0) + 1e-8
    mn, sn = Nt.mean(0), Nt.std(0) + 1e-8
    a = torch.tensor((Ct - mc) / sc, dtype=torch.float32)
    b = torch.tensor((Nt - mn) / sn, dtype=torch.float32)
    hc, hn = torch.nn.Linear(a.shape[1], out_dim), torch.nn.Linear(b.shape[1], out_dim)
    opt = torch.optim.AdamW([*hc.parameters(), *hn.parameters()], lr=lr, weight_decay=weight_decay)
    lab = torch.arange(a.shape[0])
    for _ in range(steps):
        za, zb = torch.nn.functional.normalize(hc(a), dim=1), torch.nn.functional.normalize(hn(b), dim=1)
        logits = za @ zb.T / tau
        loss = 0.5 * (torch.nn.functional.cross_entropy(logits, lab) + torch.nn.functional.cross_entropy(logits.T, lab))
        opt.zero_grad()
        loss.backward()
        opt.step()
    hc.eval()
    hn.eval()

    def fc(C: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return hc(torch.tensor((C - mc) / sc, dtype=torch.float32)).numpy()

    def fn(N: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return hn(torch.tensor((N - mn) / sn, dtype=torch.float32)).numpy()
    return fc, fn


def retrieval_for_embedding(emb: Any, label: dict[str, Any], forty: dict[int, float] | None, arms: Sequence[str] = ("unaligned", "procrustes", "infonce", "random"),
                            seed: int = 0, twin_forty_s: float = 0.03, min_queries: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every arm on one fold embedding table. Returns (per-query records, extras: position chance, fitted maps)."""
    ids, pos, C, N = domain_pairs(emb, "held_out")
    if ids.size < min_queries:
        raise ResearchBlocked(BLOCKED_INSUFFICIENT_SAMPLE, f"{ids.size} held-out athletes with both domains (< {min_queries})")
    _, _, Ct, Nt = domain_pairs(emb, "train")
    pools = {p: pool_members(ids, pos, p, forty, twin_forty_s) for p in POOLS}
    rows: list[dict[str, Any]] = []
    extras: dict[str, Any] = {"position_chance": position_chance(ids, pos), "maps": {}}
    for arm in arms:
        if arm == "unaligned":
            Q, G = C, N
        elif arm == "procrustes":
            if Ct.shape[0] < C.shape[1] + 5:
                continue
            f = procrustes_map(Ct, Nt)
            extras["maps"]["procrustes"] = f
            Q, G = f(C), N
        elif arm == "infonce":
            if Ct.shape[0] < 30:
                continue
            try:
                fc, fn = infonce_heads(Ct, Nt, C.shape[1], seed)
            except ResearchBlocked:
                continue
            extras["maps"]["infonce"] = fc
            Q, G = fc(C), fn(N)
        elif arm == "random":
            rng = np.random.default_rng(seed + 17)
            Q, G = rng.normal(size=C.shape), rng.normal(size=N.shape)
        else:
            raise ValueError(arm)
        rows += score_queries(Q, G, ids, pos, pools, {**label, "arm": arm})
    return rows, extras


def concat_embeddings(a: Any, b: Any) -> Any:
    """Row-wise concatenation of two fold embedding tables (same rows), each block standardised with its TRAINING rows."""
    import polars as pl

    za, zb = zcols(a), zcols(b)

    def std(t: Any, cols: list[str], pref: str) -> Any:
        tr = t.filter(pl.col("role") == "train")
        mu = {c: float(tr[c].mean()) for c in cols}
        sd = {c: float(tr[c].std()) or 1.0 for c in cols}
        return t.select("idx", *[((pl.col(c) - mu[c]) / sd[c]).alias(f"{pref}{c}") for c in cols])

    j = a.select("idx", "nfl_id", "domain", "kind", "position_group", "fold", "role").join(std(a, za, "a"), on="idx").join(std(b, zb, "b"), on="idx")
    new = [c for c in j.columns if c.startswith(("az", "bz"))]
    return j.rename({c: f"z{k}" for k, c in enumerate(new)})


def sequence_matrix_embedding(E: np.ndarray, index: Any, fold: int) -> Any:
    """A fold 'embedding table' from a fixed per-sequence matrix (e.g. F0 engineered features): role from the fold map."""
    import polars as pl

    role = np.where(index["fold"].to_numpy() == fold, "held_out", "train")
    return index.select("idx", "nfl_id", "domain", "kind", "position_group", "fold").with_columns(pl.Series("role", role),
                                                                                                    *[pl.Series(f"z{j}", E[:, j].astype(float)) for j in range(E.shape[1])])


def paired_rr(scores: Any, arm: str, reference: str, pool: str, n_boot: int, seed: int) -> dict[str, Any]:
    """Mean reciprocal-rank difference (arm − reference) on the SAME queries, athlete bootstrap."""
    import polars as pl

    keys = [c for c in ("rep", "fold", "nfl_id") if c in scores.columns]
    a = scores.filter((pl.col("arm") == arm) & (pl.col("pool") == pool)).select(*keys, pl.col("rr").alias("a"))
    b = scores.filter((pl.col("arm") == reference) & (pl.col("pool") == pool)).select(*keys, pl.col("rr").alias("b"))
    j = a.join(b, on=keys)
    if j.height < 10:
        return {"arm": arm, "reference": reference, "pool": pool, "n": j.height}
    d = (j["a"] - j["b"]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, d.size, (n_boot, d.size))].mean(1)
    return {"arm": arm, "reference": reference, "pool": pool, "n": int(d.size), "delta_mrr": float(d.mean()), "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)), "p_boot_le0": float((bs <= 0).mean())}


def damage_check(emb: Any, label: dict[str, Any], fmap: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    """Repeat-attempt AUC of held-out Combine attempts before vs after the alignment map (E15.E3)."""
    import polars as pl

    zc = zcols(emb)
    raw = repeat_auc(emb, label)
    Z = fmap(emb.select(zc).to_numpy().astype(float))
    mapped = emb.drop(zc).with_columns(*[pl.Series(f"z{k}", Z[:, k]) for k in range(Z.shape[1])])
    al = repeat_auc(mapped, label)
    j = raw.join(al.select("kind", pl.col("repeat_auc").alias("repeat_auc_aligned")), on="kind")
    drop = (j["repeat_auc"] - j["repeat_auc_aligned"]).to_numpy()
    return {**label, "max_repeat_auc_drop": float(np.nanmax(drop)) if drop.size else NAN, "destructive": bool(np.nanmax(drop) > 0.02) if drop.size else None}


__all__ = ["POOLS", "binomial_interval", "concat_embeddings", "damage_check", "domain_pairs", "infonce_heads", "paired_rr", "pool_members", "position_chance",
           "procrustes_map", "prototype_reading", "retrieval_for_embedding", "score_queries", "sequence_matrix_embedding", "summarise"]
