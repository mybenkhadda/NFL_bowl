"""Statistics of the sprint-rescue experiment (NumPy only; the athlete is the independent unit).

Stratified ICC(1)
    One-way random effects with athletes nested in strata (position group, cohort, or both). With P strata, g
    athletes, N attempts and k_i attempts per athlete:

        SSB = Σ_p Σ_{i∈p} k_i (ȳ_i − ȳ_p)²        df_b = g − P        (athletes around their STRATUM mean)
        SSW = Σ_i Σ_j (y_ij − ȳ_i)²               df_w = N − g
        n0  = (N − Σ_p Σ_{i∈p} k_i² / N_p) / (g − P)
        ICC = (MSB − MSW) / (MSB + (n0 − 1)·MSW)

    One stratum reproduces ``features.reliability.icc_oneway`` exactly (tested). Stratum means are removed from
    the between-athlete variance, so differences between position groups cannot inflate the ICC.
    Variance components: σ²_within = MSW, σ²_between = max((MSB − MSW) / n0, 0).
    Interval: percentile bootstrap resampling athletes WITHIN strata. Two features measured on the same athletes
    share the same resamples (same seed, same athlete order), so their ICC difference has a paired interval.

Partial association
    Both variables are residualised on the covariates by least squares, then correlated. The Spearman version
    rank-transforms the two variables and every continuous covariate first. Interval: Fisher z with
    n − 3 − q degrees of freedom (q = covariate columns), valid because rows are athletes.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Sequence

import numpy as np

from ..studies.stats import Z95, rankdata

NAN = float("nan")


def seed_for(base: int, *parts: Any) -> int:
    return (base + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32)


# ============================================================================== stratified ICC
def athlete_stats(values: np.ndarray, athletes: np.ndarray, strata: np.ndarray | None = None) -> dict[str, Any]:
    """Per-athlete sufficient statistics (k, Σy, Σy²) ordered by stratum, then athlete id."""
    values, athletes = np.asarray(values, float), np.asarray(athletes)
    strata = np.zeros(values.size, dtype=object) if strata is None else np.asarray(strata).astype(str)
    ok = np.isfinite(values)
    values, athletes, strata = values[ok], athletes[ok], strata[ok]
    ids, first, inv = np.unique(athletes, return_index=True, return_inverse=True)
    lab = strata[first].astype(str)                       # an athlete belongs to one stratum (first seen)
    order = np.lexsort((ids, lab))
    rank = np.empty(ids.size, dtype=int)
    rank[order] = np.arange(ids.size)
    inv = rank[inv]
    g = ids.size
    k = np.bincount(inv, minlength=g).astype(float)
    s1 = np.bincount(inv, weights=values, minlength=g)
    s2 = np.bincount(inv, weights=values ** 2, minlength=g)
    lab = lab[order]
    names, starts = np.unique(lab, return_index=True) if g else (np.array([]), np.array([], dtype=int))
    bounds = [*sorted(starts.tolist()), g]
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    return {"k": k, "s1": s1, "s2": s2, "groups": groups, "ids": ids[order], "strata": lab, "n_strata": len(names)}


def icc_core(K: np.ndarray, S1: np.ndarray, S2: np.ndarray, groups: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Vectorised over leading axes; the last axis indexes athletes, ``groups`` the columns of each stratum."""
    N = K.sum(-1)
    g, P = K.shape[-1], len(groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        ssw = (S2 - S1 ** 2 / K).sum(-1)
        ssb, corr = np.zeros_like(N), np.zeros_like(N)
        for idx in groups:
            Kp, S1p = K[..., idx], S1[..., idx]
            Np = Kp.sum(-1)
            ssb = ssb + (S1p ** 2 / Kp).sum(-1) - S1p.sum(-1) ** 2 / Np
            corr = corr + (Kp ** 2).sum(-1) / Np
        dfb, dfw = g - P, N - g
        msb = np.where(dfb > 0, ssb / max(dfb, 1), np.nan)
        msw = np.where(dfw > 0, ssw / np.where(dfw > 0, dfw, 1), np.nan)
        n0 = np.where(dfb > 0, (N - corr) / max(dfb, 1), np.nan)
        den = msb + (n0 - 1) * msw
        icc = np.where(den > 0, (msb - msw) / den, np.nan)
    return {"icc": icc, "msb": msb, "msw": msw, "n0": n0, "N": N}


def bootstrap_picks(groups: Sequence[np.ndarray], n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, g) athlete indices: athletes resampled with replacement inside their own stratum, so every
    replicate keeps the stratum sizes and the column → stratum mapping of ``groups``."""
    rng = np.random.default_rng(seed)
    g = int(sum(len(i) for i in groups))
    pick = np.empty((n_boot, g), dtype=int)
    for idx in groups:
        pick[:, idx] = idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))]
    return pick


def stratified_icc(values: np.ndarray, athletes: np.ndarray, strata: np.ndarray | None = None, n_boot: int = 0,
                   seed: int = 0) -> dict[str, Any]:
    st = athlete_stats(values, athletes, strata)
    g = st["k"].size
    out: dict[str, Any] = {"icc": NAN, "ci_lower": NAN, "ci_upper": NAN, "var_between": NAN, "var_within": NAN,
                           "n0": NAN, "n_athletes": int(g), "n_obs": int(st["k"].sum()), "n_strata": st["n_strata"],
                           "k_mean": float(st["k"].mean()) if g else NAN, "reps": np.empty(0), "stats": st}
    if g - st["n_strata"] < 1 or st["k"].sum() <= g:
        return out
    c = icc_core(st["k"], st["s1"], st["s2"], st["groups"])
    msb, msw, n0 = float(c["msb"]), float(c["msw"]), float(c["n0"])
    out.update(icc=float(c["icc"]), var_within=msw, n0=n0,
               var_between=max((msb - msw) / n0, 0.0) if np.isfinite(n0) and n0 > 0 else NAN)
    if n_boot > 0 and g >= 3:
        pick = bootstrap_picks(st["groups"], n_boot, seed)
        reps = icc_core(st["k"][pick], st["s1"][pick], st["s2"][pick], st["groups"])["icc"]
        out["reps"] = reps
        ok = reps[np.isfinite(reps)]
        if ok.size >= max(20, n_boot // 10):
            out.update(ci_lower=float(np.quantile(ok, 0.025)), ci_upper=float(np.quantile(ok, 0.975)))
    return out


def paired_difference(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float, float]:
    """ICC(a) − ICC(b) with a paired percentile interval. Valid only when both come from ``stratified_icc`` on
    the SAME athletes with the same seed and n_boot; otherwise the interval is NaN."""
    diff = a["icc"] - b["icc"]
    ra, rb = a["reps"], b["reps"]
    same = (ra.size and ra.size == rb.size and a["stats"]["ids"].size == b["stats"]["ids"].size
            and bool((a["stats"]["ids"] == b["stats"]["ids"]).all()))
    if not same:
        return float(diff), NAN, NAN
    d = ra - rb
    d = d[np.isfinite(d)]
    if d.size < 20:
        return float(diff), NAN, NAN
    return float(diff), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# ============================================================================== association
def design(n: int, categorical: Sequence[np.ndarray] = (), continuous: Sequence[np.ndarray] = ()) -> np.ndarray:
    """Intercept + drop-first dummies of every categorical covariate + continuous covariates."""
    cols = [np.ones(n)]
    for c in categorical:
        lev = np.unique(np.asarray(c).astype(str))
        cols += [(np.asarray(c).astype(str) == v).astype(float) for v in lev[1:]]
    cols += [np.asarray(c, float) for c in continuous]
    return np.column_stack(cols)


def residualise(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    return float(np.corrcoef(a, b)[0, 1]) if sa > 1e-12 and sb > 1e-12 else NAN


def partial_corr(y, x, categorical: Sequence = (), continuous: Sequence = (), method: str = "spearman"
                 ) -> dict[str, float]:
    """Association of ``y`` and ``x`` given the covariates; rows with any missing value are dropped."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    cont = [np.asarray(c, float) for c in continuous]
    cat = [np.asarray(c) for c in categorical]
    ok = np.isfinite(y) & np.isfinite(x)
    for c in cont:
        ok &= np.isfinite(c)
    y, x, cont, cat = y[ok], x[ok], [c[ok] for c in cont], [c[ok] for c in cat]
    n = int(y.size)
    out = {"r": NAN, "ci_lower": NAN, "ci_upper": NAN, "n": n, "q": 0}
    if n < 5:
        return out
    if method == "spearman":
        y, x, cont = rankdata(y), rankdata(x), [rankdata(c) for c in cont]
    Z = design(n, cat, cont)
    q = Z.shape[1] - 1
    out["q"] = q
    if n - q < 5:
        return out
    r = _corr(residualise(y, Z), residualise(x, Z))
    out["r"] = r
    if np.isfinite(r) and abs(r) < 1 and n - q > 3:
        z, se = math.atanh(r), 1.0 / math.sqrt(n - q - 3)
        out.update(ci_lower=math.tanh(z - Z95 * se), ci_upper=math.tanh(z + Z95 * se))
    return out


def r_squared(y, X: Sequence, categorical: Sequence = ()) -> dict[str, float]:
    """Share of the variance of ``y`` explained by the columns of ``X`` beyond the categorical covariates
    (partial R²: both sides are residualised on the dummies first). Adjusted for the number of predictors."""
    y = np.asarray(y, float)
    X = [np.asarray(c, float) for c in X]
    cat = [np.asarray(c) for c in categorical]
    ok = np.isfinite(y)
    for c in X:
        ok &= np.isfinite(c)
    y, X, cat = y[ok], [c[ok] for c in X], [c[ok] for c in cat]
    n, p = int(y.size), len(X)
    Z = design(n, cat) if n else np.ones((0, 1))
    q = Z.shape[1] - 1
    if n - q - p < 3:
        return {"r2": NAN, "r2_adj": NAN, "n": n}
    ry = residualise(y, Z)
    rx = np.column_stack([residualise(c, Z) for c in X])
    sst = float((ry ** 2).sum())
    if sst <= 0:
        return {"r2": NAN, "r2_adj": NAN, "n": n}
    beta, *_ = np.linalg.lstsq(rx, ry, rcond=None)
    r2 = 1.0 - float(((ry - rx @ beta) ** 2).sum()) / sst
    return {"r2": r2, "r2_adj": 1.0 - (1.0 - r2) * (n - q - 1) / (n - q - p - 1), "n": n}


def bootstrap_ci(stat, n: int, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile interval of ``stat(row_indices)`` over resampled rows (rows = athletes)."""
    if n < 5 or n_boot <= 0:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    reps = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)], float)
    reps = reps[np.isfinite(reps)]
    return (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))) if reps.size >= 20 else (NAN, NAN)


def within_athlete_slopes(y: np.ndarray, X: np.ndarray, athletes: np.ndarray) -> np.ndarray:
    """Least-squares slopes of y on the columns of X using WITHIN-athlete deviations only (athlete means removed
    from both sides), i.e. how an athlete's value moves with the recording window between own attempts."""
    y, X = np.asarray(y, float), np.atleast_2d(np.asarray(X, float).T).T
    _, inv = np.unique(np.asarray(athletes), return_inverse=True)
    k = np.bincount(inv).astype(float)
    yd = y - (np.bincount(inv, weights=y) / k)[inv]
    Xd = np.column_stack([X[:, j] - (np.bincount(inv, weights=X[:, j]) / k)[inv] for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    return beta


def spearman_brown(icc1: float, k: float) -> float:
    """Reliability of the mean of k observations given the single-observation ICC."""
    if not np.isfinite(icc1) or 1 + (k - 1) * icc1 <= 0:
        return NAN
    return float(k * icc1 / (1 + (k - 1) * icc1))


def observations_needed(icc1: float, target: float) -> float:
    """Observations per athlete for the mean to reach ``target`` reliability (Spearman–Brown inverse)."""
    if not np.isfinite(icc1) or icc1 <= 0 or icc1 >= 1:
        return NAN
    return float(target * (1 - icc1) / (icc1 * (1 - target)))
