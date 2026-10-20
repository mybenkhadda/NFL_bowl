"""Study C — drill-to-NFL task specificity (E18.C1; Finding 3 hypotheses, [HYP]).

Features of each Combine drill family (M16 ``trk_<scope>__*`` simple statistics and M17 ``sig_*`` agility signatures)
against the NFL task targets (T1 high speed, T2 acceleration, T3 braking, T4 reacceleration, T5 turn retention): partial
Spearman within position CONTROLLING for M1 (residualised on M1 + position dummies, OLS on all athletes — descriptive) with
an athlete bootstrap. Pre-declared physically matched pairs (``DIAGONAL``):

    forty   → T1, T2           agility (three-cone + shuttles, and signatures) → T3, T4, T5

Diagonal contrast per drill family = mean |ρ| on its matched targets − mean |ρ| on its unmatched targets (bootstrap CI).
A positive contrast supports TASK-specific transfer; a contrast ≈ 0 with uniform associations points to general
athleticism or residual position confounding. Effects are also reported in physical units per 1 within-position SD of
the feature (OLS slope on the M1-residualised quantities).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")
DIAGONAL = {"forty": ("T1", "T2"), "agility": ("T3", "T4", "T5")}


def family_of(col: str) -> str | None:
    if col.startswith("trk_forty__"):
        return "forty"
    if col.startswith("trk_agility__") or col.startswith("sig_"):
        return "agility"
    return None


def _resid(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v) & np.isfinite(Z).all(1)
    out = np.full(v.size, NAN)
    if ok.sum() > Z.shape[1] + 5:
        A = np.column_stack([np.ones(ok.sum()), Z[ok]])
        beta, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
        out[ok] = v[ok] - A @ beta
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size)
    r[o] = np.arange(x.size)
    return r


def matrix(data: Any, target_cols: dict[str, str], feature_cols: Sequence[str], m1_cols: Sequence[str], n_boot: int, seed: int) -> tuple[Any, Any]:
    """(long association table, diagonal contrast per drill family)."""
    import polars as pl

    pos = data["position_group"].fill_null("?").to_numpy()
    P = np.stack([(pos == v).astype(float) for v in sorted(set(pos.tolist()))[1:]], 1) if np.unique(pos).size > 1 else np.zeros((data.height, 0))
    M = data.select([pl.col(c).cast(pl.Float64) for c in m1_cols if c in data.columns]).to_numpy()
    M = np.where(np.isfinite(M), M, np.nanmedian(M, 0))
    Z = np.column_stack([M, P])
    rng = np.random.default_rng(seed)
    rows, boots = [], {}
    for f in feature_cols:
        fam = family_of(f)
        if fam is None:
            continue
        x = _resid(data[f].cast(pl.Float64).fill_null(np.nan).to_numpy(), Z)
        for tid, tcol in target_cols.items():
            if tcol not in data.columns:
                continue
            y = _resid(data[tcol].cast(pl.Float64).fill_null(np.nan).to_numpy(), Z)
            ok = np.isfinite(x) & np.isfinite(y)
            n = int(ok.sum())
            if n < 30:
                continue
            xs, ys = x[ok], y[ok]
            rho = float(np.corrcoef(_rank(xs), _rank(ys))[0, 1])
            idx = rng.integers(0, n, (n_boot, n))
            bs = np.array([np.corrcoef(_rank(xs[i]), _rank(ys[i]))[0, 1] for i in idx])
            boots[(f, tid)] = bs
            sd = float(np.nanstd(xs))
            slope = float(np.polyfit(xs / sd, ys, 1)[0]) if sd > 0 else NAN
            rows.append({"feature": f, "drill_family": fam, "target_id": tid, "matched": tid in DIAGONAL.get(fam, ()), "n": n, "partial_spearman": rho,
                         "lo": float(np.nanquantile(bs, 0.025)), "hi": float(np.nanquantile(bs, 0.975)), "slope_per_sd": slope})
    long = pl.DataFrame(rows, infer_schema_length=None)
    contrast = []
    for fam in DIAGONAL:
        sub = [r for r in rows if r["drill_family"] == fam]
        diag = [(r["feature"], r["target_id"]) for r in sub if r["matched"]]
        off = [(r["feature"], r["target_id"]) for r in sub if not r["matched"]]
        if not diag or not off:
            contrast.append({"drill_family": fam, "n_matched": len(diag), "n_unmatched": len(off), "note": "needs matched and unmatched targets"})
            continue
        B = min(len(boots[k]) for k in diag + off)
        cd = np.nanmean(np.abs(np.stack([boots[k][:B] for k in diag])), 0) - np.nanmean(np.abs(np.stack([boots[k][:B] for k in off])), 0)
        est = float(np.mean([abs(r["partial_spearman"]) for r in sub if r["matched"]]) - np.mean([abs(r["partial_spearman"]) for r in sub if not r["matched"]]))
        contrast.append({"drill_family": fam, "n_matched": len(diag), "n_unmatched": len(off), "diagonal_contrast": est, "lo": float(np.nanquantile(cd, 0.025)),
                         "hi": float(np.nanquantile(cd, 0.975)), "reading": "task-specific (CI > 0)" if np.nanquantile(cd, 0.025) > 0 else "not task-specific / inconclusive"})
    return long, pl.DataFrame(contrast, infer_schema_length=None)


__all__ = ["DIAGONAL", "family_of", "matrix"]
