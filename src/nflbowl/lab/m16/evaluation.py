"""Track C evaluation add-ons on top of ``baselines.evaluate_target`` / ``transfer_fairness`` (which stay the reference).

    cohort_audit        per model: finite predictions, and whether the model's athlete set equals the strict cohort —
                        paired comparisons are only valid on identical evaluation athletes
    calibration         OOF calibration slope / intercept (y on prediction) with a fold-stratified athlete bootstrap, and
                        binned means for plots (slope < 1 = over-dispersed predictions)
    stratified_errors   RMSE per model inside strata (NFL exposure tertiles; Combine-tracking coverage) — the cohort is
                        unchanged, the strata only partition it
    paired_spearman     Δ within-position Spearman (model − reference) on the same athletes, fold-stratified bootstrap
    improvement_table   ΔRMSE, % of M1 RMSE, Δ within-position R² next to the declared status (one row per comparison)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")


def cohort_audit(long: Any, models: Sequence[str]) -> Any:
    import polars as pl

    fin = long.filter(pl.col("pred").is_finite())
    sets = {m: set(int(i) for i in fin.filter(pl.col("model") == m)["nfl_id"].to_list()) for m in models}
    strict = set.intersection(*sets.values()) if sets else set()
    return pl.DataFrame([{"model": m, "n_finite": len(s), "n_strict_cohort": len(strict), "identical_to_strict": s == strict,
                          "athletes_outside_strict": len(s - strict)} for m, s in sets.items()])


def calibration(w: Any, models: Sequence[str], n_boot: int, seed: int, n_bins: int = 10) -> tuple[Any, Any]:
    """(per model slope / intercept with CI, binned means)."""
    import polars as pl

    from ..transfer_fairness import bootstrap_index

    y = w["y"].to_numpy().astype(float)
    idx = bootstrap_index(w["fold"].to_numpy(), n_boot, seed)
    rows, bins = [], []
    for m in models:
        p = w[f"pred_{m}"].to_numpy().astype(float)
        if np.std(p) <= 0:
            continue

        def fit(ii: np.ndarray) -> tuple[float, float]:
            pp, yy = p[ii], y[ii]
            v = np.var(pp)
            b = float(np.cov(pp, yy, ddof=0)[0, 1] / v) if v > 0 else NAN
            return b, float(yy.mean() - b * pp.mean())

        b, a = fit(np.arange(y.size))
        bs = np.array([fit(i)[0] for i in idx])
        rows.append({"model": m, "n": int(y.size), "slope": b, "slope_lo": float(np.nanquantile(bs, 0.025)), "slope_hi": float(np.nanquantile(bs, 0.975)),
                     "intercept": a, "mean_error": float((p - y).mean())})
        q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        code = np.clip(np.searchsorted(q, p, side="right") - 1, 0, n_bins - 1)
        for k in range(n_bins):
            s = code == k
            if s.sum():
                bins.append({"model": m, "bin": k, "n": int(s.sum()), "mean_pred": float(p[s].mean()), "mean_y": float(y[s].mean())})
    return pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(bins, infer_schema_length=None)


def strata_exposure(targets: Any, exposure_col: str, n: int = 3) -> Any:
    """nfl_id → exposure tertile label (descriptive strata)."""
    import polars as pl

    if exposure_col not in targets.columns:
        return pl.DataFrame(schema={"nfl_id": pl.Int64, "stratum": pl.String})
    d = targets.select(pl.col("nfl_id").cast(pl.Int64), pl.col(exposure_col).cast(pl.Float64)).drop_nulls()
    q = np.quantile(d[exposure_col].to_numpy(), np.linspace(0, 1, n + 1)) if d.height else np.zeros(n + 1)
    code = np.clip(np.searchsorted(q, d[exposure_col].to_numpy(), side="right") - 1, 0, n - 1)
    return d.select("nfl_id").with_columns(pl.Series("stratum", [f"exposure T{k + 1}" for k in code]))


def strata_coverage(analysis: Any, cols: Sequence[str], label: str) -> Any:
    """nfl_id → 'has <label>' / 'missing <label>' (any finite value among ``cols``)."""
    import polars as pl

    cols = [c for c in cols if c in analysis.columns]
    if not cols:
        return pl.DataFrame(schema={"nfl_id": pl.Int64, "stratum": pl.String})
    has = pl.any_horizontal([pl.col(c).cast(pl.Float64).is_finite().fill_null(False) for c in cols])
    return analysis.select(pl.col("nfl_id").cast(pl.Int64), pl.when(has).then(pl.lit(f"has {label}")).otherwise(pl.lit(f"missing {label}")).alias("stratum"))


def stratified_errors(w: Any, models: Sequence[str], strata: Any, variable: str) -> Any:
    import polars as pl

    d = w.join(strata, on="nfl_id", how="left").with_columns(pl.col("stratum").fill_null("unknown"))
    rows = []
    y = d["y"].to_numpy().astype(float)
    st = d["stratum"].to_numpy()
    for s in np.unique(st):
        m = st == s
        for mod in models:
            p = d[f"pred_{mod}"].to_numpy().astype(float)
            rows.append({"variable": variable, "stratum": s, "model": mod, "n": int(m.sum()), "rmse": float(np.sqrt(np.mean((y[m] - p[m]) ** 2))),
                         "mean_error": float((p[m] - y[m]).mean())})
    return pl.DataFrame(rows, infer_schema_length=None)


def _within_spearman(y: np.ndarray, p: np.ndarray, pos: np.ndarray, min_n: int = 10) -> float:
    from .target_audit import _spearman

    vals, ns = [], []
    for k in np.unique(pos):
        m = pos == k
        if m.sum() >= min_n:
            r = _spearman(p[m], y[m])
            if np.isfinite(r):
                vals.append(r)
                ns.append(m.sum())
    return float(np.average(vals, weights=ns)) if vals else NAN


def paired_spearman(w: Any, pairs: Sequence[tuple[str, str]], n_boot: int, seed: int) -> Any:
    """Δ of the athlete-weighted mean within-position Spearman (model − reference), fold-stratified bootstrap."""
    import polars as pl

    from ..transfer_fairness import bootstrap_index

    y = w["y"].to_numpy().astype(float)
    pos = w["position_group"].fill_null("?").to_numpy()
    idx = bootstrap_index(w["fold"].to_numpy(), n_boot, seed)
    rows = []
    for mod, ref in pairs:
        if f"pred_{mod}" not in w.columns or f"pred_{ref}" not in w.columns:
            continue
        a, b = w[f"pred_{mod}"].to_numpy().astype(float), w[f"pred_{ref}"].to_numpy().astype(float)
        est = _within_spearman(y, a, pos) - _within_spearman(y, b, pos)
        bs = np.array([_within_spearman(y[i], a[i], pos[i]) - _within_spearman(y[i], b[i], pos[i]) for i in idx])
        bs = bs[np.isfinite(bs)]
        rows.append({"model": mod, "reference": ref, "n": int(y.size), "delta_within_spearman": est, "lo": float(np.quantile(bs, 0.025)) if bs.size else NAN,
                     "hi": float(np.quantile(bs, 0.975)) if bs.size else NAN, "p_boot_le0": float((bs <= 0).mean()) if bs.size else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


def improvement_table(paired: Any, status: Any, within: Any, margin_pct: float) -> Any:
    """One row per comparison vs M1: ΔRMSE (CI), % of M1 RMSE, exceeds the practical margin, declared status."""
    import polars as pl

    if paired is None or not paired.height:
        return pl.DataFrame()
    p = paired.filter(pl.col("reference") == "M1")
    keys = [k for k in ("target_id", "model") if k in p.columns]
    if status is not None and status.height:
        s = status.filter(pl.col("reference") == "M1") if "reference" in status.columns else status
        p = p.join(s.select(*keys, pl.col("status").alias("declared_status")), on=keys, how="left")
    out = p.with_columns((pl.col("delta_pct") >= margin_pct).alias("exceeds_practical_margin"))
    if within is not None and within.height and "model" in within.columns:
        r2 = [c for c in within.columns if "r2" in c]
        if r2:
            out = out.join(within.select(*[k for k in keys if k in within.columns], pl.col(r2[0]).alias("within_position_r2")), on=[k for k in keys if k in within.columns],
                           how="left")
    return out


__all__ = ["calibration", "cohort_audit", "improvement_table", "paired_spearman", "strata_coverage", "strata_exposure", "stratified_errors"]
