"""Measurement uncertainty of scorecard values and percentiles. NumPy only.

Model           an athlete's attempt value = his true value + attempt error. Within a position group:
                    σ²_within   pooled variance of attempts around the athlete mean (athletes with a repeat)
                    σ²_between  variance of athlete means minus the error it contains (method of moments, ≥ 0)
                The athlete's value is the mean of his n valid attempts: standard error σ_within / √n, reliability
                σ²_b / (σ²_b + σ²_w / n). One attempt is trusted less than two.
Estimate        reliability-aware (empirical-Bayes) estimate m = μ + λ (x̄ − μ), λ = the reliability above.
Percentile      the percentile of the OBSERVED value is the point shown. Its interval propagates (a) the athlete's
                measurement error — draws x̄ + SE · N(0, 1) — and (b) the sampling of the reference population — a
                bootstrap of the reference athletes. The reliability-aware percentile places the shrunk estimate among
                the shrunk estimates of the reference. A percentile is NOT multiplied by an ICC: reliability changes
                where the athlete is expected to stand and how wide the interval is, not the scale of the percentile.
Not estimable   an official result is one value per athlete: its measurement error cannot be estimated here, and only
                the reference-sampling part of the interval is given (stated in the table).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .percentiles import NAN, ReferenceSpec, orient, percentile_of


def attempt_components(values: np.ndarray, athlete: np.ndarray, use: np.ndarray | None = None) -> dict[str, Any]:
    """Variance components of repeated attempts for ONE position group. ``values`` / ``athlete``: per attempt;
    ``use``: attempts of the reference athletes (the components describe the reference population only).
    Returns σ²_within, σ²_between, μ, athletes, athletes with a repeat, ICC(1) of a single attempt."""
    ok = np.isfinite(values) & (np.ones(values.size, dtype=bool) if use is None else use)
    v, a = values[ok], athlete[ok]
    ids, inv = np.unique(a, return_inverse=True)
    if ids.size < 3:
        return {"s2_within": NAN, "s2_between": NAN, "mu": NAN, "n_athletes": int(ids.size), "n_repeat": 0, "icc1": NAN}
    n = np.bincount(inv).astype(float)
    mean = np.bincount(inv, weights=v) / n
    df_w = float((n - 1).sum())
    s2w = float(((v - mean[inv]) ** 2).sum() / df_w) if df_w >= 1 else NAN
    s2b = NAN
    if np.isfinite(s2w):
        s2b = max(float(mean.var(ddof=1)) - s2w * float(np.mean(1.0 / n)), 0.0)
    icc = s2b / (s2b + s2w) if np.isfinite(s2b) and s2b + s2w > 0 else NAN
    return {"s2_within": s2w, "s2_between": s2b, "mu": float(mean.mean()), "n_athletes": int(ids.size), "n_repeat": int((n >= 2).sum()),
            "icc1": icc}


def reliability_of_mean(comp: dict[str, Any], n: np.ndarray) -> np.ndarray:
    """σ²_b / (σ²_b + σ²_w / n) for athlete means of n attempts (NaN when the components are not estimable)."""
    s2b, s2w = comp["s2_between"], comp["s2_within"]
    if not (np.isfinite(s2b) and np.isfinite(s2w)):
        return np.full(np.asarray(n).shape, NAN)
    den = s2b + s2w / np.maximum(n, 1)
    return np.where((n > 0) & (den > 0), s2b / np.where(den > 0, den, 1.0), NAN)


def standard_error(comp: dict[str, Any], n: np.ndarray) -> np.ndarray:
    s2w = comp["s2_within"]
    return np.where(n > 0, np.sqrt(s2w / np.maximum(n, 1)), NAN) if np.isfinite(s2w) else np.full(np.asarray(n).shape, NAN)


def shrunk_estimate(mean: np.ndarray, n: np.ndarray, comp: dict[str, Any]) -> np.ndarray:
    """Reliability-aware estimate of the true value: the athlete mean pulled towards the reference mean by its
    reliability. Equal to the mean when the components are not estimable."""
    lam = reliability_of_mean(comp, n)
    return np.where(np.isfinite(lam), comp["mu"] + lam * (mean - comp["mu"]), mean)


def percentile_interval(x: float, se: float, ref: np.ndarray, higher_is_better: bool, rng: np.random.Generator, n_draws: int = 1000,
                        level: float = 0.95) -> tuple[float, float]:
    """Interval of the oriented percentile of ``x`` among ``ref``: every draw perturbs the athlete's value by its
    standard error (0 or NaN → no perturbation) and resamples the reference athletes with replacement."""
    ref = ref[np.isfinite(ref)]
    if not np.isfinite(x) or ref.size < 5:
        return NAN, NAN
    s = float(se) if np.isfinite(se) and se > 0 else 0.0
    xs = (x + s * rng.standard_normal(n_draws))[:, None]
    R = ref[rng.integers(0, ref.size, size=(n_draws, ref.size))]
    out = orient(100.0 * ((R < xs).sum(1) + 0.5 * (R == xs).sum(1)) / ref.size, higher_is_better)
    return float(np.quantile(out, (1 - level) / 2)), float(np.quantile(out, 1 - (1 - level) / 2))


def availability(comp: dict[str, Any], n_reference: int, median_attempts: float, spec: ReferenceSpec, tracked: bool) -> tuple[str, str]:
    """(available · experimental · unavailable, reason) of one feature in one position group."""
    if n_reference < spec.min_reference:
        return "unavailable", f"reference of {n_reference} athletes < {spec.min_reference}"
    if not tracked:
        return "available", "single official value: repeatability not estimable here"
    if comp["n_repeat"] < spec.min_repeat or not np.isfinite(comp["s2_between"]):
        return "experimental", f"only {comp['n_repeat']} athletes with a repeat: measurement error not estimable (< {spec.min_repeat})"
    rel = float(reliability_of_mean(comp, np.array([max(median_attempts, 1.0)]))[0])
    if rel >= spec.available_reliability:
        return "available", f"reliability of the athlete value {rel:.2f}"
    if rel >= spec.experimental_reliability:
        return "experimental", f"reliability of the athlete value {rel:.2f} < {spec.available_reliability:.2f}"
    return "unavailable", f"reliability of the athlete value {rel:.2f} < {spec.experimental_reliability:.2f}"


def cohort_sensitivity(values: np.ndarray, position: np.ndarray, draft_year: np.ndarray, reference: np.ndarray, higher_is_better: bool,
                       min_reference: int) -> list[dict[str, Any]]:
    """How far percentiles move when the reference is ONE draft cohort instead of all: per position × cohort, the mean
    absolute difference and the largest difference from the all-cohort percentile (athletes of the position)."""
    rows = []
    for p in np.unique(position):
        k = (position == p) & np.isfinite(values)
        ref_all = values[k & reference]
        if ref_all.size < min_reference:
            continue
        base = np.array([orient(percentile_of(float(x), ref_all), higher_is_better) for x in values[k]])
        for c in np.unique(draft_year[k & reference]):
            ref = values[k & reference & (draft_year == c)]
            if ref.size < max(8, min_reference // 3):
                continue
            alt = np.array([orient(percentile_of(float(x), ref), higher_is_better) for x in values[k]])
            rows.append({"position_group": str(p), "cohort": int(c), "n_reference_cohort": int(ref.size), "n_scored": int(k.sum()),
                         "mean_abs_shift": float(np.abs(alt - base).mean()), "max_abs_shift": float(np.abs(alt - base).max())})
    return rows


def attempts_wide(values: Sequence[float], athlete: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(athlete ids, mean of valid attempts, number of valid attempts) from per-attempt values."""
    v, a = np.asarray(values, dtype=float), np.asarray(athlete, dtype=np.int64)
    ok = np.isfinite(v)
    ids, inv = np.unique(a[ok], return_inverse=True)
    n = np.bincount(inv, minlength=ids.size)
    return ids, np.bincount(inv, weights=v[ok], minlength=ids.size) / np.maximum(n, 1), n.astype(np.int64)
