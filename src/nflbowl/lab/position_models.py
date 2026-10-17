"""Part F — position-conditioned NFL capacity.

Pooled R² of the frozen target can be dominated by position. Every model here is fitted per OUTER fold on training
athletes only, with the same ridge machinery as the ladder (``m11 ev.design`` train-only design, ``ae._alpha`` inner
CV), on the same analysis rows and folds:

    <set>·pooled        ridge(set) + unpenalized position dummies (= the ladder model for that feature set)
    <set>·centred       target centred on the TRAIN position mean; ridge(set) without dummies; pred = μ_p + f(x)
    <set>·standardised  target z-scored with TRAIN position mean / SD; ridge(set) without dummies; pred = μ_p + σ_p ẑ
    <set>·specific      separate ridge per position (positions with < ``min_train`` training athletes use ·pooled)
    <set>·partial       w_p ·specific + (1 − w_p) ·pooled, w_p = n_p / (n_p + κ); κ by inner CV on TRAINING athletes
                        (α fixed at the outer-fold choices — documented approximation)

Positions unseen in a training fold fall back to the training-fold global mean / SD. Original-scale RMSE and
within-position metrics are both reported; position groups below ``min_position_n`` get no interval.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..data.validation import TRANSFER_TARGET

NAN = float("nan")


def _ae() -> Any:
    from .. import _backend

    return _backend.ae()


def position_params(y: np.ndarray, pos: np.ndarray, train: np.ndarray, min_group: int = 3) -> tuple[dict[str, float], dict[str, float], float, float]:
    """TRAIN-only position means / SDs (+ global train mean / SD fallback). Held-out rows never enter."""
    yt, pt = y[train], pos[train]
    g_mu, g_sd = float(yt.mean()), float(yt.std()) if yt.size > 1 else 1.0
    mu, sd = {}, {}
    for p in np.unique(pt):
        v = yt[pt == p]
        if v.size >= min_group:
            mu[str(p)] = float(v.mean())
            s = float(v.std())
            sd[str(p)] = s if s > 1e-12 else g_sd
    return mu, sd, g_mu, (g_sd if g_sd > 1e-12 else 1.0)


def _lookup(pos: np.ndarray, table: dict[str, float], default: float) -> np.ndarray:
    return np.array([table.get(str(p), default) for p in pos], float)


def _fit(d: Any, y: np.ndarray, cols: Sequence[str], train: np.ndarray, test: np.ndarray, C: Any, seed: int) -> tuple[np.ndarray, float]:
    ae = _ae()
    a = ae._alpha(d, y, cols, None, train, C, seed)
    return ae._fit_predict(d, y, cols, None, train, test, a), a


def oof_position_models(data: Any, cols: Sequence[str], tag: str, C: Any, seed: int, min_train: int = 20, kappas: Sequence[float] = (5.0, 15.0, 40.0, 120.0)
                        ) -> tuple[Any, Any]:
    """(wide OOF frame nfl_id, position_group, fold, y, pred_<tag>·<variant> …; per-fold log incl. κ and train parameters)."""
    import polars as pl

    d = data.filter(pl.col(TRANSFER_TARGET).cast(pl.Float64).is_finite().fill_null(False)).with_columns(pl.col("position_group").fill_null("?"))
    y = d[TRANSFER_TARGET].cast(pl.Float64).to_numpy()
    folds = d["fold"].to_numpy()
    pos = d["position_group"].to_numpy()
    dn = d.drop("position_group")                                   # design without position dummies
    names = [f"{tag}·{v}" for v in ("pooled", "centred", "standardised", "specific", "partial")]
    P = {n: np.full(d.height, NAN) for n in names}
    log = []
    for f in sorted(set(folds.tolist())):
        test, train = folds == f, folds != f
        if test.sum() == 0 or train.sum() < 20:
            continue
        s = seed + 31 * int(f)
        pooled, a_pool = _fit(d, y, cols, train, np.ones(d.height, bool), C, s)          # every row: needed for inner pooling
        P[f"{tag}·pooled"][test] = pooled[test]
        mu, sd, gm, gs = position_params(y, pos, train)
        m_all, s_all = _lookup(pos, mu, gm), _lookup(pos, sd, gs)
        yc = y - m_all
        pc, _ = _fit(dn, yc, cols, train, test, C, s)
        P[f"{tag}·centred"][test] = m_all[test] + pc
        yz = (y - m_all) / s_all
        pz, _ = _fit(dn, yz, cols, train, test, C, s)
        P[f"{tag}·standardised"][test] = m_all[test] + s_all[test] * pz
        spec = pooled.copy()
        n_tr = {}
        for p in np.unique(pos):
            tp, sp_ = train & (pos == p), pos == p
            n_tr[str(p)] = int(tp.sum())
            if tp.sum() >= min_train:
                spec[sp_], _ = _fit(dn, y, cols, tp, sp_, C, s)
        P[f"{tag}·specific"][test] = spec[test]
        kappa = _choose_kappa(d, dn, y, cols, pos, train, C, s, min_train, kappas)
        w = np.array([n_tr.get(str(p), 0) / (n_tr.get(str(p), 0) + kappa) if n_tr.get(str(p), 0) >= min_train else 0.0 for p in pos])
        P[f"{tag}·partial"][test] = (w * spec + (1 - w) * pooled)[test]
        log.append({"tag": tag, "fold": int(f), "alpha_pooled": a_pool, "kappa": kappa, "n_train": int(train.sum()), "n_test": int(test.sum()),
                    "positions_specific": int(sum(v >= min_train for v in n_tr.values())), "train_position_means": str(mu)})
    oof = d.select("nfl_id", "position_group", "fold", pl.col(TRANSFER_TARGET).cast(pl.Float64).alias("y")).with_columns(
        [pl.Series(f"pred_{n}", P[n]) for n in names])
    return oof, pl.DataFrame(log, infer_schema_length=None)


def _choose_kappa(d: Any, dn: Any, y: np.ndarray, cols: Sequence[str], pos: np.ndarray, train: np.ndarray, C: Any, seed: int, min_train: int,
                  kappas: Sequence[float]) -> float:
    """Inner athlete CV among TRAINING athletes: pooled and position-specific inner fits, SSE of the blend per κ."""
    ae = _ae()
    idx = np.flatnonzero(train)
    k_in = int(getattr(C, "inner_folds", 5))
    if idx.size < 2 * k_in:
        return float(max(kappas))
    inner = np.random.default_rng(seed + 7).permutation(idx.size) % k_in
    a_pool = ae._alpha(d, y, cols, None, train, C, seed)
    sse = np.zeros(len(kappas))
    for j in range(k_in):
        tr, va = np.zeros(d.height, bool), np.zeros(d.height, bool)
        tr[idx[inner != j]], va[idx[inner == j]] = True, True
        pooled = ae._fit_predict(d, y, cols, None, tr, va, a_pool)
        spec = pooled.copy()
        pv = pos[va]
        n_tr = {}
        for p in np.unique(pv):
            tp = tr & (pos == p)
            n_tr[str(p)] = int(tp.sum())
            if tp.sum() >= min_train:
                spec[pv == p] = ae._fit_predict(dn, y, cols, None, tp, va & (pos == p), a_pool)
        for i, kappa in enumerate(kappas):
            w = np.array([n_tr.get(str(p), 0) / (n_tr.get(str(p), 0) + kappa) if n_tr.get(str(p), 0) >= min_train else 0.0 for p in pv])
            sse[i] += float(((y[va] - (w * spec + (1 - w) * pooled)) ** 2).sum())
    best = np.flatnonzero(sse <= sse.min() * (1 + 1e-9))
    return float(kappas[int(best.max())])


def within_position_metrics(w: Any, models: Sequence[str]) -> Any:
    """Per model: RMSE in units of the in-cohort position SD and the mean of per-position R² (positions ≥ 10 athletes)."""
    import polars as pl

    y = w["y"].to_numpy().astype(float)
    pos = w["position_group"].to_numpy()
    sd = np.zeros(y.size)
    for p in np.unique(pos):
        m = pos == p
        sd[m] = y[m].std() if m.sum() > 1 else np.nan
    rows = []
    for mname in models:
        e = y - w[f"pred_{mname}"].to_numpy().astype(float)
        ok = np.isfinite(sd) & (sd > 0)
        r2p = []
        for p in np.unique(pos):
            m = pos == p
            if m.sum() >= 10:
                den = ((y[m] - y[m].mean()) ** 2).sum()
                r2p.append(1 - (e[m] ** 2).sum() / den if den > 0 else np.nan)
        rows.append({"model": mname, "rmse_position_sd_units": float(np.sqrt(np.mean((e[ok] / sd[ok]) ** 2))) if ok.any() else NAN,
                     "mean_within_position_r2": float(np.nanmean(r2p)) if r2p else NAN, "n_positions_used": len(r2p)})
    return pl.DataFrame(rows)


def position_effects(w: Any, model: str, reference: str, n_boot: int, seed: int, min_n: int = 10, adequate_n: int = 30) -> Any:
    """Per position: paired Δ RMSE (reference − model) with an athlete bootstrap; < min_n → no interval; < adequate_n →
    'exploratory'. BH over the positions with intervals."""
    import polars as pl

    ae = _ae()
    rows = []
    for p in sorted(w["position_group"].unique().to_list(), key=str):
        g = w.filter(pl.col("position_group") == p)
        n = g.height
        y, a, b = g["y"].to_numpy().astype(float), g[f"pred_{reference}"].to_numpy().astype(float), g[f"pred_{model}"].to_numpy().astype(float)
        r_ref, r_mod = float(np.sqrt(np.mean((y - a) ** 2))), float(np.sqrt(np.mean((y - b) ** 2)))
        rel = "insufficient" if n < min_n else ("exploratory" if n < adequate_n else "adequate")
        lo = hi = pb = NAN
        if rel != "insufficient":
            idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
            d = np.sqrt(((y[idx] - a[idx]) ** 2).mean(1)) - np.sqrt(((y[idx] - b[idx]) ** 2).mean(1))
            lo, hi, pb = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float((d <= 0).mean())
        rows.append({"position_group": p, "model": model, "reference": reference, "n_athletes": n, "reliability": rel, "rmse_ref": r_ref, "rmse_model": r_mod,
                     "delta_rmse": r_ref - r_mod, "delta_lo": lo, "delta_hi": hi, "p_boot_le0": pb,
                     "reading": "suppressed (small group)" if rel == "insufficient" else ("model better" if lo > 0 else ("model worse" if hi < 0 else "inconclusive"))})
    out = pl.DataFrame(rows, infer_schema_length=None)
    if out.height:
        out = out.with_columns(pl.Series("p_bh", ae.bh([1.0 if not np.isfinite(v) else v for v in out["p_boot_le0"].to_list()])))
    return out


__all__ = ["oof_position_models", "position_effects", "position_params", "within_position_metrics"]
