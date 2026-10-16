"""Vectorised retrieval metrics with the EXACT estimands and random draws of ``m11.representation``.

What changed versus the reference, and why the numbers are the same:

``within_domain_retrieval``  per drill family, the Python loop over queries is replaced by matrix operations on the same
    similarity matrix (argmax, masked max, counts). Pure comparisons → identical top-1 / AUC values; queries are
    appended in the same family and row order, so the averages sum in the same order.
``cross_domain_retrieval``   the reference filtered the metadata once per athlete (O(athletes × sequences)). Here the
    row indices of every athlete are grouped once (stable sort → the same ascending rows), and ``E[rows].mean(0)`` is the
    same expression. The per-query AUC / top-5 loop is vectorised per position group (comparisons only) and emitted in the
    reference order; the athlete bootstrap is copied verbatim (same generator, same draws).
``matched_retrieval``        the dense (queries × NFL windows) similarity / argsort is processed in query chunks (bounded
    memory). Rows are sorted independently, so chunking does not change any row's ranking. Opt-in ``approx_topk`` uses
    ``argpartition``: only the SET of k neighbours enters the reported medians / shares, so results differ only when
    similarities tie at the k-th position. Opt-in GPU similarities are flagged approximate (TF32 / reduction order).

Floating-point note: cosine similarities are BLAS products; a different chunk shape can change a value in the last bit.
A metric changes only if two similarities tie to the last bit, which the parity tests check on fixtures.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .. import _backend

NAN = float("nan")
ACCEL_DEMANDS = ["accel_low", "accel_mid", "transition_high", "reaccel_after_brake"]


def cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _backend.rep()._cos(A, B)


def _normalize(A: np.ndarray) -> np.ndarray:
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)      # same expression as rep._cos


def group_rows(ids: np.ndarray, rows: np.ndarray) -> dict[int, np.ndarray]:
    """``{athlete: ascending global row indices}`` restricted to ``rows`` (ascending). Equals the rows returned by
    ``meta.with_row_index('_r').filter(domain …).filter(nfl_id == i)['_r']``."""
    if rows.size == 0:
        return {}
    sub = ids[rows]
    order = np.argsort(sub, kind="stable")
    s = sub[order]
    cut = np.flatnonzero(s[1:] != s[:-1]) + 1
    keys = s[np.r_[0, cut]]
    return {int(k): g for k, g in zip(keys, np.split(rows[order], cut))}


def rank_rows(S: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For query k: AUC = share of same-position other athletes ranked below its own match ``S[k, k]``; top-5 hit.
    Returns (auc, top5, ok) in query order; ``ok`` False where fewer than 2 same-position others exist (reference skip)."""
    n = S.shape[0]
    auc, top5, ok = np.full(n, NAN), np.zeros(n, bool), np.zeros(n, bool)
    groups: dict[Any, list[int]] = {}
    for k, p in enumerate(P.tolist()):
        groups.setdefault(p, []).append(k)
    for g in groups.values():
        if len(g) - 1 < 2:
            continue
        g_ = np.asarray(g)
        sub = S[np.ix_(g_, g_)]
        d = np.diagonal(sub)[:, None]
        auc[g_] = (sub < d).sum(1) / (g_.size - 1)                # self-comparison d < d is False: excluded
        top5[g_] = (sub > d).sum(1) < 5
        ok[g_] = True
    return auc, top5, ok


def within_domain_retrieval(E: np.ndarray, meta: Any) -> dict[str, Any]:
    import polars as pl

    c = meta.with_row_index("_r").filter(pl.col("domain") == "combine")
    top1: list[bool] = []
    aucs: list[float] = []
    for (_fam,), g in c.partition_by(["kind"], as_dict=True).items():
        ids, rows = g["nfl_id"].to_numpy(), g["_r"].to_numpy()
        if rows.size < 5:
            continue
        S = cos(E[rows], E[rows])
        np.fill_diagonal(S, -np.inf)
        same = ids[:, None] == ids[None, :]
        np.fill_diagonal(same, False)
        q = same.any(1)
        best = np.argmax(S, axis=1)
        t1 = same[np.arange(rows.size), best]
        posmax = np.where(same, S, -np.inf).max(1)
        neg = ~same & np.isfinite(S)
        nneg = neg.sum(1)
        cnt = ((S < posmax[:, None]) & neg).sum(1)
        for i in np.flatnonzero(q):
            top1.append(bool(t1[i]))
            if nneg[i]:
                aucs.append(float(cnt[i] / nneg[i]))
    return {"n_queries": len(top1), "top1": float(np.mean(top1)) if top1 else NAN, "auc": float(np.mean(aucs)) if aucs else NAN}


class CrossDomainIndex:
    """Athlete means of both domains + the same-position candidate structure, built once and reused by the identity
    null (where only the NFL labels move)."""

    def __init__(self, E: np.ndarray, meta: Any, combine_kinds: Sequence[str] | None = None):
        dom = meta["domain"].to_numpy()
        self.ids = meta["nfl_id"].to_numpy()
        cmask = dom == "combine"
        if combine_kinds is not None:
            cmask &= meta["kind"].is_in(list(combine_kinds)).to_numpy()
        self.cg = group_rows(self.ids, np.flatnonzero(cmask))
        self.ng = group_rows(self.ids, np.flatnonzero(dom == "nfl"))
        self.both = sorted(set(self.cg) & set(self.ng))
        self.pos = dict(zip(meta["nfl_id"].to_list(), meta["position_group"].to_list()))
        self.E = E

    def combine_means(self) -> np.ndarray:
        return np.stack([self.E[self.cg[i]].mean(0) for i in self.both])

    def nfl_means(self, athletes: Sequence[int]) -> np.ndarray:
        return np.stack([self.E[self.ng[i]].mean(0) for i in athletes])

    def positions(self) -> np.ndarray:
        return np.array([self.pos.get(i) for i in self.both])


def cross_domain_retrieval(E: np.ndarray, meta: Any, n_boot: int, seed: int, combine_kinds: Sequence[str] | None = None,
                           index: CrossDomainIndex | None = None) -> dict[str, Any]:
    ix = index or CrossDomainIndex(E, meta, combine_kinds)
    both = ix.both
    if len(both) < 10:
        return {"n_athletes": len(both), "auc": NAN, "auc_lo": NAN, "auc_hi": NAN, "top5": NAN}
    S = cos(ix.combine_means(), ix.nfl_means(both))
    auc, top5, ok = rank_rows(S, ix.positions())
    a = auc[ok]
    t5 = top5[ok]
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, a.size, a.size)].mean() for _ in range(n_boot)] if a.size >= 10 else []
    return {"n_athletes": int(a.size), "auc": float(a.mean()) if a.size else NAN, "auc_lo": float(np.quantile(bs, 0.025)) if bs else NAN,
            "auc_hi": float(np.quantile(bs, 0.975)) if bs else NAN, "top5": float(np.mean(t5)) if t5.size else NAN}


def _topk_cpu(Qn: np.ndarray, Bn: np.ndarray, k: int, chunk: int, approx: bool) -> np.ndarray:
    nb = np.empty((Qn.shape[0], k), np.int64)
    for b in range(0, Qn.shape[0], chunk):
        S = Qn[b: b + chunk] @ Bn.T
        nb[b: b + chunk] = np.argpartition(-S, k - 1, axis=1)[:, :k] if approx else np.argsort(-S, axis=1)[:, :k]
    return nb


def _topk_gpu(Qn: np.ndarray, Bn: np.ndarray, k: int, chunk: int) -> np.ndarray | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    B = torch.from_numpy(np.ascontiguousarray(Bn, np.float32)).cuda()
    out = []
    for b in range(0, Qn.shape[0], chunk):
        Q = torch.from_numpy(np.ascontiguousarray(Qn[b: b + chunk], np.float32)).cuda()
        out.append(torch.topk(Q @ B.T, k, dim=1).indices.cpu().numpy())
    return np.concatenate(out).astype(np.int64)


def matched_retrieval(E: np.ndarray, meta: Any, phys: Any, k: int = 20, max_queries: int = 2000, seed: int = 0, chunk: int = 256,
                      approx_topk: bool = False, similarity_device: str = "cpu") -> dict[str, Any]:
    import polars as pl

    m = meta.with_row_index("_r").join(phys, left_on="_r", right_on="idx", how="left")
    q = m.filter((pl.col("domain") == "combine") & (pl.col("kind") == "forty"))
    n = m.filter(pl.col("domain") == "nfl")
    if q.height < 5 or n.height < k * 2:
        return {"n_queries": q.height, "neighbour_dspeed": NAN, "random_dspeed": NAN, "neighbour_accel_share": NAN, "base_accel_share": NAN}
    rng = np.random.default_rng(seed)
    q = q.sample(min(max_queries, q.height), seed=seed)
    Qn, Bn = _normalize(E[q["_r"].to_numpy()]), _normalize(E[n["_r"].to_numpy()])
    vq, vn = q["max_speed"].to_numpy(), n["max_speed"].to_numpy()
    acc = n["kind"].is_in(ACCEL_DEMANDS).to_numpy()
    nb = _topk_gpu(Qn, Bn, k, chunk) if similarity_device == "auto" else None
    if nb is None:
        nb = _topk_cpu(Qn, Bn, k, chunk, approx_topk)
    rnd = rng.integers(0, n.height, nb.shape)
    return {"n_queries": q.height, "neighbour_dspeed": float(np.median(np.abs(vn[nb] - vq[:, None]))), "random_dspeed": float(np.median(np.abs(vn[rnd] - vq[:, None]))),
            "neighbour_accel_share": float(acc[nb].mean()), "base_accel_share": float(acc.mean())}


__all__ = ["CrossDomainIndex", "cos", "cross_domain_retrieval", "group_rows", "matched_retrieval", "rank_rows", "within_domain_retrieval"]
