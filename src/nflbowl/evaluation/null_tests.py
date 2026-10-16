"""Shuffled-identity null of cross-domain retrieval (``se.shuffled_identity_null``), without recomputing anything per
permutation that the permutation does not change.

Reference: per replicate, NFL athlete ids are permuted WITHIN position (same generator, same draw order: positions in
first-appearance order of NFL rows, each sorted id list permuted by ``rng.permutation``) and the full cross-domain
retrieval is recomputed (AUC without bootstrap).

Here: the permutation is drawn identically. Relabelling NFL rows only re-assigns which athlete's NFL mean sits in each
column: new athlete ``j`` owns the NFL rows of original athlete ``inv[j]``, whose mean is the same expression over the
same rows. The set of athletes present in both domains and every athlete's position are unchanged by a within-position
relabelling. So the null AUC is ``rank_rows`` on the columns of ONE similarity matrix (Combine means × all NFL means)
re-indexed by the permutation — O(perm × athletes²) instead of O(perm × sequences × athletes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import _backend
from .retrieval import NAN, CrossDomainIndex, cos, rank_rows


def _permutations(meta: Any, n_perm: int, seed: int) -> list[dict[int, int]]:
    rng = np.random.default_rng(seed)
    nfl = meta["domain"].to_numpy() == "nfl"
    ids, pos = meta["nfl_id"].to_numpy().copy(), meta["position_group"].fill_null("?").to_numpy()
    uniq: dict[Any, set[int]] = {}
    for i, p in zip(ids[nfl], pos[nfl]):
        uniq.setdefault(p, set()).add(int(i))
    out = []
    for _ in range(n_perm):
        mp: dict[int, int] = {}
        for _p, s in uniq.items():
            s = sorted(s)
            mp |= dict(zip(s, rng.permutation(s).tolist()))
        out.append(mp)
    return out


def shuffled_identity_null(E: np.ndarray, meta: Any, n_perm: int, seed: int, backend: str = "fast") -> np.ndarray:
    if backend == "reference":
        return _backend.se().shuffled_identity_null(E, meta, n_perm, seed)
    perms = _permutations(meta, n_perm, seed)
    ix = CrossDomainIndex(E, meta)
    both = ix.both
    if len(both) < 10:
        return np.full(n_perm, NAN)
    nfl_all = sorted(ix.ng)
    col = {a: j for j, a in enumerate(nfl_all)}
    S_full = cos(ix.combine_means(), ix.nfl_means(nfl_all))
    P = ix.positions()
    out = []
    for mp in perms:
        inv = {v: k for k, v in mp.items()}
        cols = np.array([col[inv[a]] for a in both])
        auc, _t5, ok = rank_rows(S_full[:, cols], P)
        a = auc[ok]
        out.append(float(a.mean()) if a.size else NAN)
    return np.asarray(out, float)


def null_summary(observed: float, null: np.ndarray) -> dict[str, Any]:
    v = null[np.isfinite(null)]
    return {"null_n": int(v.size), "null_mean": float(v.mean()) if v.size else NAN, "null_max": float(v.max()) if v.size else NAN,
            "null_q95": float(np.quantile(v, 0.95)) if v.size else NAN,
            "null_exceed_share": float((v >= observed).mean()) if (v.size and np.isfinite(observed)) else NAN}


__all__ = ["null_summary", "shuffled_identity_null"]
