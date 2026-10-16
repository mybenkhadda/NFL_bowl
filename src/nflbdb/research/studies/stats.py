"""Shared statistics for studies A–I (NumPy only — the project has no SciPy dependency).

Principles
* The athlete is the independent unit. Frames / trajectories / attempts of one athlete are
  correlated, so uncertainty comes from the **athlete-cluster bootstrap**
  (:func:`cluster_bootstrap_ratio` for ratio statistics such as MAE, RMSE², bias and proportions;
  :func:`cluster_bootstrap` for arbitrary statistics).
* Every function is deterministic given ``seed``.
* p-values are provided for exploratory screening only; effect sizes + intervals are the result.

Formulas
* Wilson score interval for a binomial proportion (i.i.d. units only).
* Cliff's δ = P(X > Y) − P(X < Y) = 2·U / (n_x n_y) − 1 (U = Mann–Whitney, average ranks for ties).
* Spearman ρ = Pearson correlation of average ranks; Fisher-z interval (i.i.d. approximation).
* Wilcoxon signed-rank: zero differences discarded, average ranks for |d| ties, normal
  approximation with tie correction (valid for n ≳ 20; reported with n).
* Bland–Altman: bias = mean(d), limits of agreement = bias ± 1.96·SD(d) (describes the unit at
  which d is measured — frame-level LoA describe frames, not athletes).
* Benjamini–Hochberg step-up FDR within a declared family.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

Z95 = 1.959963984540054


# ----------------------------------------------------------------------------- basic
def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties averaged — equivalent to scipy.stats.rankdata(method='average')."""
    a = np.asarray(a, float)
    n = a.size
    if not n:
        return np.empty(0)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    new = np.r_[True, s[1:] != s[:-1]]                 # vectorised: O(n log n), no Python loop
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], n]
    avg = (starts + ends + 1) / 2.0
    ranks = np.empty(n, float)
    ranks[order] = avg[np.cumsum(new) - 1]
    return ranks


def _finite_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def pearson(x, y) -> float:
    x, y = _finite_pair(x, y)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x, y = _finite_pair(x, y)
    if x.size < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def fisher_ci(r: float, n: int, z: float = Z95) -> tuple[float, float]:
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return float("nan"), float("nan")
    zr, se = math.atanh(r), 1.0 / math.sqrt(n - 3)
    return math.tanh(zr - z * se), math.tanh(zr + z * se)


def wilson_ci(k, n, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z ** 2 / n
        mid = (p + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return np.where(n > 0, np.clip(mid - half, 0, 1), np.nan), np.where(n > 0, np.clip(mid + half, 0, 1), np.nan)


# ----------------------------------------------------------------------------- effect sizes / tests
def cliffs_delta(x, y) -> float:
    """P(X > Y) − P(X < Y); positive = x tends to be larger. NaN if a group is empty."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if not x.size or not y.size:
        return float("nan")
    r = rankdata(np.r_[x, y])
    u = r[: x.size].sum() - x.size * (x.size + 1) / 2.0
    return float(2.0 * u / (x.size * y.size) - 1.0)


def wilcoxon_signed_rank(d) -> dict[str, float]:
    """Paired test of median(d) = 0 (normal approximation, tie-corrected)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = d.size
    if n < 10:
        return {"n": n, "w_plus": float("nan"), "z": float("nan"), "p_value": float("nan")}
    r = rankdata(np.abs(d))
    w = float(r[d > 0].sum())
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - (counts ** 3 - counts).sum() / 48.0
    z = (w - mu) / math.sqrt(var) if var > 0 else float("nan")
    return {"n": n, "w_plus": w, "z": z, "p_value": 2 * norm_sf(abs(z)) if np.isfinite(z) else float("nan")}


def bh_fdr(p: Sequence[float]) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values; NaN entries are ignored and stay NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not ok.size:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    m = order.size
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.clip(adj, 0, 1)
    return out


def bland_altman(diff) -> dict[str, float]:
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return {"bias": float("nan"), "sd": float("nan"), "loa_lower": float("nan"), "loa_upper": float("nan")}
    b, sd = float(d.mean()), float(d.std(ddof=1))
    return {"bias": b, "sd": sd, "loa_lower": b - Z95 * sd, "loa_upper": b + Z95 * sd}


def equivalence_decision(ci90_lo: float, ci90_hi: float, margin: float) -> str:
    """TOST via a 90 % interval and a PRE-SPECIFIED margin: 'equivalent' only if the whole interval
    lies within ±margin; 'not_equivalent' if it lies wholly outside; otherwise 'inconclusive'."""
    if not (np.isfinite(ci90_lo) and np.isfinite(ci90_hi)):
        return "inconclusive"
    if -margin < ci90_lo and ci90_hi < margin:
        return "equivalent"
    if ci90_lo >= margin or ci90_hi <= -margin:
        return "not_equivalent"
    return "inconclusive"


# ----------------------------------------------------------------------------- cluster bootstrap
def _resample_counts(n_clusters: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_clusters) multiplicities of a with-replacement cluster resample."""
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    return np.stack([np.bincount(row, minlength=n_clusters) for row in idx]).astype(float)


def cluster_bootstrap_ratio(num, den, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                            transform: Callable[[np.ndarray], np.ndarray] | None = None) -> dict[str, float]:
    """Ratio statistic Σnum / Σden with clusters (athletes) as the resampling unit.

    ``num``/``den`` hold one sufficient statistic per cluster (e.g. Σ|r| and frame count per athlete →
    MAE; Σr² and count → MSE with ``transform=np.sqrt`` → RMSE; flagged count and n → proportion).
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    tr = transform or (lambda v: v)
    if den.sum() <= 0:
        return {"estimate": float("nan"), "se": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_clusters": int(num.size)}
    est = float(tr(np.array([num.sum() / den.sum()]))[0])
    if num.size < 2:
        return {"estimate": est, "se": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_clusters": int(num.size)}
    w = _resample_counts(num.size, n_boot, np.random.default_rng(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = tr((w @ num) / (w @ den))
    b = b[np.isfinite(b)]
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2]) if b.size else (np.nan, np.nan)
    return {"estimate": est, "se": float(b.std(ddof=1)) if b.size > 1 else float("nan"), "ci_lower": float(lo),
            "ci_upper": float(hi), "n_clusters": int(num.size)}


def resample_counts(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """Public, seeded version of the cluster-resample multiplicity matrix (n_boot, n_clusters)."""
    return _resample_counts(n_clusters, n_boot, np.random.default_rng(seed))


def cluster_bootstrap_ratios(num: np.ndarray, den: np.ndarray, n_boot: int = 1000, seed: int = 0
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Several ratio statistics sharing ONE set of cluster resamples (consistent, and K× cheaper).

    ``num``: (n_clusters, K) sufficient statistics; ``den``: (n_clusters,). Returns (estimates (K,),
    bootstrap replicates (n_boot, K)); apply any transform (e.g. sqrt for RMSE) before taking quantiles
    with :func:`percentile_interval`. Fewer than 2 clusters → replicates are all NaN.
    """
    num, den = np.atleast_2d(np.asarray(num, float).T).T, np.asarray(den, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = num.sum(axis=0) / den.sum()
        if den.size < 2:
            return est, np.full((n_boot, num.shape[1]), np.nan)
        w = _resample_counts(den.size, n_boot, np.random.default_rng(seed))
        return est, (w @ num) / (w @ den)[:, None]


def percentile_interval(reps: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise (lower, upper, SE) of bootstrap replicates, ignoring non-finite replicates."""
    reps = np.atleast_2d(np.asarray(reps, float).T).T
    lo, hi, se = (np.full(reps.shape[1], np.nan) for _ in range(3))
    for k in range(reps.shape[1]):
        r = reps[:, k][np.isfinite(reps[:, k])]
        if r.size > 1:
            lo[k], hi[k] = np.quantile(r, [alpha / 2, 1 - alpha / 2])
            se[k] = r.std(ddof=1)
    return lo, hi, se


def cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters, n_boot: int = 500, seed: int = 0,
                      alpha: float = 0.05) -> dict[str, float]:
    """Generic percentile bootstrap. ``stat(idx)`` receives row indices of the resampled data
    (whole clusters drawn with replacement); ``clusters`` gives each row's cluster id."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    est = float(stat(np.arange(clusters.size)))
    if uniq.size < 2:
        return {"estimate": est, "se": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_clusters": int(uniq.size)}
    order = np.argsort(inv, kind="mergesort")
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=uniq.size))]
    members = [order[bounds[i]:bounds[i + 1]] for i in range(uniq.size)]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        out[b] = stat(np.concatenate([members[i] for i in pick]))
    out = out[np.isfinite(out)]
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2]) if out.size else (np.nan, np.nan)
    return {"estimate": est, "se": float(out.std(ddof=1)) if out.size > 1 else float("nan"), "ci_lower": float(lo),
            "ci_upper": float(hi), "n_clusters": int(uniq.size)}
