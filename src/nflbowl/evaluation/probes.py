"""Physical probes and domain-shift diagnostics.

``probe_r2``      ``rep.probe_r2`` with the per-fold standardisation and Gram matrix computed once per (domain, fold)
                  instead of once per (domain, fold, target). They do not depend on the target, so every prediction is
                  produced by the same arithmetic as the reference.
``domain_shift``  ``rep.domain_shift`` unchanged (already subsampled to ≤ 2000 / 500 rows).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import _backend

NAN = float("nan")
PROBE_TARGETS = ("max_speed", "mean_a_tan", "max_abs_a_n", "abs_heading_change_rad")


def probe_r2(E: np.ndarray, meta: Any, phys: Any, folds_by_athlete: dict[int, int], outer_folds: int) -> Any:
    import polars as pl

    rep = _backend.rep()
    m = meta.with_row_index("_r").join(phys, left_on="_r", right_on="idx", how="left")
    rows = []
    for dom in ("combine", "nfl"):
        d = m.filter(pl.col("domain") == dom)
        if d.height < 50:
            continue
        Z = E[d["_r"].to_numpy()]
        f = np.array([folds_by_athlete.get(i, i % outer_folds) for i in d["nfl_id"].to_list()])
        cache: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        for tgt in PROBE_TARGETS:
            y = d[tgt].to_numpy().astype(float)
            pred = np.full(y.size, NAN)
            for k in np.unique(f):
                tr, te = f != k, f == k
                if k not in cache:
                    Zs = rep._zscore(Z, tr)
                    cache[k] = (Zs, Zs[tr].T @ Zs[tr] + 10.0 * np.eye(Z.shape[1]))
                Zs, A = cache[k]
                beta = np.linalg.solve(A, Zs[tr].T @ (y[tr] - y[tr].mean()))
                pred[te] = y[tr].mean() + Zs[te] @ beta
            ok = np.isfinite(pred) & np.isfinite(y)
            r2 = 1 - np.mean((y[ok] - pred[ok]) ** 2) / np.var(y[ok]) if ok.sum() > 10 else NAN
            rows.append({"domain": dom, "probe": tgt, "n": int(ok.sum()), "r2_cv": float(r2)})
    return pl.DataFrame(rows)


def domain_shift(E: np.ndarray, meta: Any, seed: int = 0) -> dict[str, Any]:
    return _backend.rep().domain_shift(E, meta, seed=seed)


__all__ = ["PROBE_TARGETS", "domain_shift", "probe_r2"]
