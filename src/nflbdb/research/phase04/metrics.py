"""Pure agreement / stability metrics for Phase 04 validation (NumPy only — no SciPy in this project).

All functions take paired arrays ``v`` (alternative setting) and ``v0`` (reference setting) measured on the
SAME units (trajectories or athletes); non-finite pairs are dropped. They describe *stability*, not
*validity*: two settings can agree perfectly and both be wrong.

* ``percentile_rank``      (average rank − 0.5) / n ∈ (0, 1)
* ``top_fraction_overlap`` |top-k(v) ∩ top-k(v0)| / k, k = ceil(frac · n); ties broken by index (deterministic)
* ``agreement``            Spearman, Pearson, signed / absolute / relative differences, Bland–Altman,
                           top-decile overlap and |percentile-rank change| quantiles in one record
* ``variance_share``       share of total variance due to the setting: mean within-unit variance across
                           settings / (that + variance of unit means). 0 = the setting never matters.
"""

from __future__ import annotations

import math

import numpy as np

from ..studies.stats import bland_altman, pearson, rankdata, spearman

REL_EPS = 1e-6


def _pair(v, v0) -> tuple[np.ndarray, np.ndarray]:
    v, v0 = np.asarray(v, float), np.asarray(v0, float)
    m = np.isfinite(v) & np.isfinite(v0)
    return v[m], v0[m]


def percentile_rank(x) -> np.ndarray:
    x = np.asarray(x, float)
    return (rankdata(x) - 0.5) / x.size if x.size else np.empty(0)


def top_fraction_overlap(v, v0, frac: float = 0.10) -> float:
    v, v0 = _pair(v, v0)
    n = v.size
    if n < 2:
        return float("nan")
    k = max(1, math.ceil(frac * n))
    top = lambda a: set(np.argsort(-a, kind="mergesort")[:k].tolist())  # noqa: E731
    return len(top(v) & top(v0)) / k


def rank_change(v, v0) -> np.ndarray:
    """|percentile rank under v − percentile rank under v0| per unit (0 = same position)."""
    v, v0 = _pair(v, v0)
    return np.abs(percentile_rank(v) - percentile_rank(v0)) if v.size else np.empty(0)


def agreement(v, v0, frac: float = 0.10) -> dict[str, float]:
    v, v0 = _pair(v, v0)
    n = int(v.size)
    out: dict[str, float] = {"n": n}
    if n < 3:
        return {**out, **{k: float("nan") for k in (
            "spearman", "pearson", "bias", "loa_lower", "loa_upper", "median_abs_diff", "p95_abs_diff",
            "median_rel_diff", "p90_rel_diff", "top_overlap", "rank_change_median", "rank_change_p90")}}
    d = v - v0
    ba = bland_altman(d)
    big = np.abs(v0) > REL_EPS
    rel = np.abs(d[big]) / np.abs(v0[big])
    rc = rank_change(v, v0)
    return {**out, "spearman": spearman(v, v0), "pearson": pearson(v, v0), "bias": ba["bias"],
            "loa_lower": ba["loa_lower"], "loa_upper": ba["loa_upper"],
            "median_abs_diff": float(np.median(np.abs(d))), "p95_abs_diff": float(np.quantile(np.abs(d), 0.95)),
            "median_rel_diff": float(np.median(rel)) if rel.size else float("nan"),
            "p90_rel_diff": float(np.quantile(rel, 0.90)) if rel.size else float("nan"),
            "top_overlap": top_fraction_overlap(v, v0, frac),
            "rank_change_median": float(np.median(rc)), "rank_change_p90": float(np.quantile(rc, 0.90))}


def variance_share(m) -> dict[str, float]:
    """``m``: (n_units, n_settings). Units with any non-finite setting are dropped."""
    m = np.asarray(m, float)
    if m.ndim != 2 or m.shape[1] < 2:
        return {"n_units": 0, "var_setting": float("nan"), "var_unit": float("nan"), "share_setting": float("nan")}
    m = m[np.isfinite(m).all(axis=1)]
    if m.shape[0] < 3:
        return {"n_units": int(m.shape[0]), "var_setting": float("nan"), "var_unit": float("nan"),
                "share_setting": float("nan")}
    vs = float(m.var(axis=1, ddof=1).mean())
    vu = float(m.mean(axis=1).var(ddof=1))
    tot = vs + vu
    return {"n_units": int(m.shape[0]), "var_setting": vs, "var_unit": vu,
            "share_setting": vs / tot if tot > 0 else float("nan")}


def cohen_kappa(a, b) -> float:
    """Agreement of two boolean flags beyond chance (used to show structural ≠ signal quality)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = a.size
    if not n:
        return float("nan")
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = float(pa * pb + (1 - pa) * (1 - pb))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
