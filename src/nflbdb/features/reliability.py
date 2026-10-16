"""Measurement reliability across repeated attempts (Family G).

Unit of analysis: the ATHLETE. Each athlete contributes k_i ≥ ``min_attempts`` repeated attempts
of the same canonical drill. Statistics per (drill, feature[, cohort]):

    ICC(1)     one-way random-effects intraclass correlation for unbalanced groups
               (Shrout & Fleiss 1979, case 1; n0 correction for unequal k_i):
               ICC = (MSB − MSW) / (MSB + (n0 − 1)·MSW),  n0 = (N − Σk_i²/N) / (g − 1)
               Measures how well a SINGLE attempt ranks athletes.
    ICC(1,k̄)   reliability of the athlete MEAN over k̄ attempts (Spearman–Brown):
               k̄·ICC / (1 + (k̄ − 1)·ICC)
    CI         percentile bootstrap over athletes (resample athletes with replacement)
    typical error (within-athlete SD) = √MSW ; CV% = 100 · TE / grand mean
    rank stability   Spearman ρ between each athlete's first two attempts (chronological)
    best_minus_mean  median over athletes of (best − mean of own attempts), direction-aware
Qualitative label (Koo & Li 2016 convention, applied to the CI lower bound so that an
uncertain ICC is never labelled good): <0.5 poor · 0.5–0.75 moderate · 0.75–0.9 good · ≥0.9 excellent.
Guards (MovementConfig): only the first ``max_attempts`` chronological attempts per athlete enter;
``min_athletes`` (≥ 2 attempts each) to compute an ICC; ``min_athletes_label`` to label it
(otherwise ``insufficient_for_label``); ≥ ``max_modal_share`` identical values →
``degenerate_distribution`` (ICC of near-constant counts is not informative). CV % only for
strictly positive features.

Attempts are exchangeable only if conditions are comparable; Combine drill variants (left/right)
are separate canonical drills. Learning/fatigue effects show as a non-zero mean attempt-2 − attempt-1
difference (``mean_diff_2_minus_1``), reported rather than assumed away.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import MovementConfig


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> dict:
    """ICC(1) for unbalanced one-way layouts. ``groups`` = integer codes 0..g−1."""
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    g_ids, groups = np.unique(groups, return_inverse=True)
    g = g_ids.size
    N = values.size
    if g < 2 or N <= g:
        return {"icc": np.nan, "msb": np.nan, "msw": np.nan, "n0": np.nan, "g": g, "N": N, "k_mean": np.nan}
    k = np.bincount(groups, minlength=g).astype(float)
    means = np.bincount(groups, weights=values, minlength=g) / k
    grand = values.mean()
    ssb = float((k * (means - grand) ** 2).sum())
    ssw = float(((values - means[groups]) ** 2).sum())
    msb = ssb / (g - 1)
    msw = ssw / (N - g)
    n0 = (N - (k**2).sum() / N) / (g - 1)
    denom = msb + (n0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else np.nan
    return {"icc": float(icc), "msb": msb, "msw": msw, "n0": float(n0), "g": g, "N": N, "k_mean": float(k.mean())}


def icc_bootstrap(values: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95 % percentile CI by resampling athletes (whole groups) with replacement."""
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    ids, inv = np.unique(groups, return_inverse=True)
    g = ids.size
    if g < 3 or n_boot <= 0:
        return np.nan, np.nan
    # Vectorised over replicates via per-athlete sufficient statistics (k_i, Σx, Σx²).
    k = np.bincount(inv, minlength=g).astype(float)
    s1 = np.bincount(inv, weights=values, minlength=g)
    s2 = np.bincount(inv, weights=values**2, minlength=g)
    pick = np.random.default_rng(seed).integers(0, g, size=(n_boot, g))
    K, S1, S2 = k[pick], s1[pick], s2[pick]
    N = K.sum(1)
    grand = S1.sum(1) / N
    ssb = (S1**2 / K).sum(1) - N * grand**2
    ssw = (S2 - S1**2 / K).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        msb, msw = ssb / (g - 1), ssw / (N - g)
        n0 = (N - (K**2).sum(1) / N) / (g - 1)
        denom = msb + (n0 - 1) * msw
        stats = np.where(denom > 0, (msb - msw) / denom, np.nan)
    stats = stats[np.isfinite(stats)]
    if stats.size < max(20, n_boot // 10):
        return np.nan, np.nan
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def reliability_label(ci_lo: float | None) -> str:
    if ci_lo is None or not np.isfinite(ci_lo):
        return "undetermined"
    return "excellent" if ci_lo >= 0.9 else "good" if ci_lo >= 0.75 else "moderate" if ci_lo >= 0.5 else "poor"


def reliability_table(long: pl.DataFrame, group_cols: list[str], mcfg: MovementConfig,
                      higher_is_better: dict[str, bool] | None = None, athlete: str = "nfl_id",
                      order: str = "attempt_order") -> pl.DataFrame:
    """``long``: one row per athlete × attempt × feature with columns
    ``[*group_cols, athlete, order, 'feature', 'value']`` (``group_cols`` must include 'feature')."""
    hib = higher_is_better or {}          # feature → True (higher better) / False (lower better)
    d = long.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
    ga = [*group_cols, athlete]
    # first max_attempts chronological attempts per athlete (athletes with many repetitions — position
    # drills — must not dominate the variance components), then ≥ min_attempts
    d = (d.filter(pl.col(order).rank("ordinal").over(ga) <= mcfg.max_attempts)
         .with_columns(pl.len().over(ga).alias("_k")).filter(pl.col("_k") >= mcfg.min_attempts))
    rows = []
    for key, part in d.group_by(group_cols, maintain_order=True):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(group_cols, key))
        n_ath = part[athlete].n_unique()
        rec.update(n_athletes=n_ath, n_attempts=part.height)
        if n_ath < mcfg.min_athletes:
            rows.append(rec | {"reliability": "insufficient_athletes"})
            continue
        vals = part["value"].to_numpy().astype(float)
        grp = part[athlete].to_numpy()
        st = icc_oneway(vals, grp)
        lo, hi = icc_bootstrap(vals, grp, mcfg.n_boot, mcfg.seed)
        k = st["k_mean"]
        icc = st["icc"]
        mean = float(np.mean(vals))
        te = float(np.sqrt(st["msw"])) if np.isfinite(st["msw"]) else np.nan
        direction = hib.get(rec.get("feature"))       # None → "best" is undefined for this feature
        sign = -1.0 if direction is False else 1.0
        per = part.sort(athlete, order).group_by(athlete, maintain_order=True).agg(
            pl.col("value").get(0).alias("v1"), pl.col("value").get(1).alias("v2"),
            (pl.col("value") * sign).max().alias("_best"), pl.col("value").mean().alias("_mean"))
        two = per.filter(pl.col("v1").is_not_null() & pl.col("v2").is_not_null())
        rho = (two.select(pl.corr("v1", "v2", method="spearman")).item()
               if two.height >= mcfg.min_athletes else None)
        _, counts = np.unique(vals, return_counts=True)
        modal_share = float(counts.max() / vals.size)
        if modal_share >= mcfg.max_modal_share:
            label = "degenerate_distribution"          # e.g. event counts that are 0 for most attempts
        elif n_ath < mcfg.min_athletes_label:
            label = "insufficient_for_label"           # ICC reported, but too few athletes to call it
        else:
            label = reliability_label(lo)
        rec.update(
            mean=mean, sd_between=float(np.sqrt(max((st["msb"] - st["msw"]) / st["n0"], 0.0))) if np.isfinite(st["n0"]) else None,
            typical_error=te,
            # CV only for strictly positive (ratio-scale) features; meaningless for signed ones
            cv_within_pct=100 * te / mean if (vals > 0).all() and mean > 0 else None,
            modal_share=modal_share,
            icc1=icc, icc1_ci_lo=lo, icc1_ci_hi=hi,
            icc1_mean_k=(k * icc / (1 + (k - 1) * icc)) if np.isfinite(icc) and (1 + (k - 1) * icc) > 0 else None,
            k_mean=k, spearman_first_two=rho,
            mean_diff_2_minus_1=two.select((pl.col("v2") - pl.col("v1")).mean()).item() if two.height else None,
            median_best_minus_mean=(per.select((pl.col("_best") * sign - pl.col("_mean")).abs().median()).item()
                                    if direction is not None else None),
            reliability=label)
        rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
