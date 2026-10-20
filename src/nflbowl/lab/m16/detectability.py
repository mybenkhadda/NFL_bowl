"""E16.A6 — minimum detectable effect of an added Combine feature on the REAL cohort structure (planted-signal simulation).

Same construction as Milestone 07C (``y* = μ_M1,oof + σ_res,pos · (√(1−f)·ε + √f·z)``), re-implemented here on the M16
analysis frame because the 07C engine is bound to Milestone-06 structures:

* μ_M1,oof — out-of-fold M1 ridge prediction (M1 columns + position dummies, train-fold design via ``learners``);
* σ_res,pos — within-position SD of the M1 OOF residuals; ε ~ N(0, 1); z ~ N(0, 1) the planted athlete trait;
* the analyst observes x = z + noise with measurement reliability ``rel`` (x correlates √rel with z);
* M1 vs M1 + x are refitted out of fold on y* (fixed ridge α on standardised columns — a simulation shortcut, stated);
* detected = fold-stratified paired bootstrap 95 % CI of ΔRMSE (M1 − M1+x) above 0 (``transfer_fairness.bootstrap_index``).

Power(f) = share of replicates detected; MDE = smallest f with power ≥ 0.8 (None if none in the grid). Every replicate is
seeded; results are resumable through the run's provenance cache.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..learners import design_train

NAN = float("nan")


def _ridge(X: np.ndarray, y: np.ndarray, pen: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    P = np.diag(np.where(pen, alpha, 1e-8))
    beta = np.linalg.solve(Xc.T @ Xc + P, Xc.T @ yc)
    return float(ym - xm @ beta), beta


def oof_ridge(df: Any, y: np.ndarray, cols: Sequence[str], folds: np.ndarray, alpha: float, extra: Any = None) -> np.ndarray:
    pred = np.full(y.size, NAN)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        d = design_train(df, cols, extra, tr)
        b0, b = _ridge(d.X[tr], y[tr], d.pen, alpha)
        pred[te] = b0 + d.X[te] @ b
    return pred


def simulate(df: Any, target: str, m1_cols: Sequence[str], fractions: Sequence[float] = (0.01, 0.02, 0.05, 0.1), reliability: float = 0.7,
             n_rep: int = 50, n_boot: int = 200, alpha: float = 1.0, seed: int = 0) -> Any:
    """Power per planted variance fraction f (module docstring). ``df`` = analysis frame (nfl_id, position_group, fold,
    M1 columns, ``target``)."""
    import polars as pl

    from ..transfer_fairness import bootstrap_index

    d = df.filter(pl.col(target).cast(pl.Float64).is_finite())
    y = d[target].cast(pl.Float64).to_numpy()
    folds = d["fold"].to_numpy()
    pos = d["position_group"].fill_null("?").to_numpy()
    mu = oof_ridge(d, y, m1_cols, folds, alpha)
    res = y - mu
    sd_pos = np.zeros(y.size)
    for p in np.unique(pos):
        m = pos == p
        sd_pos[m] = res[m].std() if m.sum() > 2 else res.std()
    idx = bootstrap_index(folds, n_boot, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for f in fractions:
        hits, deltas = [], []
        for r in range(n_rep):
            z = rng.normal(size=y.size)
            eps = rng.normal(size=y.size)
            ystar = mu + sd_pos * (np.sqrt(1 - f) * eps + np.sqrt(f) * z)
            x = np.sqrt(reliability) * z + np.sqrt(1 - reliability) * rng.normal(size=y.size)
            dd = d.with_columns(pl.Series("__planted", x))
            a = oof_ridge(dd, ystar, m1_cols, folds, alpha)
            b = oof_ridge(dd, ystar, [*m1_cols, "__planted"], folds, alpha)
            ea, eb = (ystar - a) ** 2, (ystar - b) ** 2
            bs = np.sqrt(ea[idx].mean(1)) - np.sqrt(eb[idx].mean(1))
            deltas.append(float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())))
            hits.append(float(np.quantile(bs, 0.025)) > 0)
        rows.append({"target": target, "planted_fraction": float(f), "measurement_reliability": reliability, "n_athletes": int(y.size), "n_rep": n_rep,
                     "power": float(np.mean(hits)), "mean_delta_rmse": float(np.mean(deltas)), "delta_pct_of_rmse": float(100 * np.mean(deltas) / np.sqrt((res ** 2).mean())),
                     "m1_oof_rmse": float(np.sqrt((res ** 2).mean()))})
    return pl.DataFrame(rows)


def mde(power: Any, threshold: float = 0.8) -> Any:
    """Smallest planted fraction with power ≥ threshold per (target, reliability); null when none reaches it."""
    import polars as pl

    out = []
    for (t, rel), g in power.group_by("target", "measurement_reliability"):
        ok = g.filter(pl.col("power") >= threshold).sort("planted_fraction")
        out.append({"target": t, "measurement_reliability": rel, "power_threshold": threshold, "mde_fraction": ok["planted_fraction"][0] if ok.height else None,
                    "mde_delta_pct": ok["delta_pct_of_rmse"][0] if ok.height else None, "max_power_in_grid": float(g["power"].max())})
    return pl.DataFrame(out, infer_schema_length=None)


__all__ = ["mde", "oof_ridge", "simulate"]
