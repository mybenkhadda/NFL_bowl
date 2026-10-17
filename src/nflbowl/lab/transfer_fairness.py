"""Parts C + D — the official NFL transfer ladder and the common-cohort fairness system.

Ladder (``Experiment.transfer_predictions``, unchanged): M1 · M1+F0 · M1+F1 TCN (when in the bundle) · M1+<encoder>
for every encoder ready on all folds; athlete outer folds, train-fold-only design (``m11 ev.design``), fold-fitted
embedding PCA (≤ 8 dims), inner-CV ridge α, position as unpenalized dummies in EVERY model.

Baselines added here WITHOUT touching the ladder: ``POS`` (position dummies only = ``ae.Model([])`` through the same
``ae.oof_ladder`` on the SAME analysis frame, same seed) and ``MEAN`` (training-fold mean). The baseline run also
re-fits ``M1``; it must reproduce the ladder's M1 to numerical precision (checked: a reproducibility assertion, not
an assumption).

Fairness rules (Part D). Predictions from any source enter one long table (source, model, nfl_id, fold, y, pred). Before
any metric: (source, model, nfl_id) unique · y identical per athlete across sources · fold identical per athlete across
sources AND equal to ``Experiment.folds_by_athlete`` · each athlete in one fold. Every metric is then computed on an
explicit cohort (set of athletes) shared by all compared models; native-coverage metrics are reported separately and
never compared across models. Δ RMSE = RMSE(reference) − RMSE(model): positive = model better (M11 convention).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..data.validation import M1_COLS, TRANSFER_TARGET

NAN = float("nan")
BASELINES = ("MEAN", "POS")


class ComparisonError(AssertionError):
    """An invalid scientific comparison (misaligned targets / folds, duplicated athletes, unmatched cohorts)."""


# ====================================================================== C — predictions
def ladder_predictions(exp: Any, encoders: Sequence[str], reuse: bool = True) -> Any:
    """Wide OOF frame of the official ladder for the READY encoders only (``[]`` → M1 · M1+F0 · M1+F1 only)."""
    return exp.transfer_predictions(models=list(encoders), reuse=reuse)


def baseline_predictions(exp: Any) -> tuple[Any, Any]:
    """(wide OOF frame with pred_MEAN, pred_POS, pred_M1_check; the analysis frame of the ladder incl. F0 columns)."""
    import polars as pl

    from .. import _backend

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    _names, _byf, _deps, data, _mods = exp._ladder_inputs([], allow_empty=True)
    res = ae.oof_ladder(data, TRANSFER_TARGET, {"POS": ae.Model([]), "M1_check": ae.Model(list(ae.M1_COLS))}, C, C.seed)
    oof = res.oof_frame()
    mean = np.full(oof.height, NAN)
    y, folds = oof["y"].to_numpy().astype(float), oof["fold"].to_numpy()
    for f in np.unique(folds):
        te = folds == f
        mean[te] = float(y[~te].mean())
    oof = oof.with_columns(pl.Series("pred_MEAN", mean))
    return oof, data


def f0_available(data: Any) -> set[int]:
    """Athletes whose Combine-forty F0 features are observed (not imputed) in the ladder frame."""
    import polars as pl

    eng = [c for c in data.columns if c.startswith("eng")]
    if not eng:
        return set()
    return set(int(i) for i in data.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in eng]))["nfl_id"].to_list())


def long_predictions(wide: Any, source: str, models: Iterable[str] | None = None, rename: dict[str, str] | None = None) -> Any:
    """Wide (nfl_id, position_group, fold, y, pred_<m>…) → long (source, model, nfl_id, position_group, fold, y, pred)."""
    import polars as pl

    rename = rename or {}
    ms = [c.removeprefix("pred_") for c in wide.columns if c.startswith("pred_")]
    if models is not None:
        keep = set(models)
        ms = [m for m in ms if m in keep]
    parts = [wide.select(pl.lit(source).alias("source"), pl.lit(rename.get(m, m)).alias("model"), pl.col("nfl_id").cast(pl.Int64),
                         pl.col("position_group").cast(pl.String).fill_null("?"), pl.col("fold").cast(pl.Int64), pl.col("y").cast(pl.Float64),
                         pl.col(f"pred_{m}").cast(pl.Float64).alias("pred")) for m in ms]
    return pl.concat(parts, how="vertical") if parts else pl.DataFrame(schema={"source": pl.String, "model": pl.String, "nfl_id": pl.Int64,
                                                                                 "position_group": pl.String, "fold": pl.Int64, "y": pl.Float64, "pred": pl.Float64})


# ====================================================================== D — alignment validation
def validate_alignment(long: Any, folds_by_athlete: dict[int, int] | None = None, tol: float = 1e-9) -> Any:
    """Raise ``ComparisonError`` on any violation; return the audit rows (check, status, detail, n)."""
    import polars as pl

    rows = []

    def ok(check: str, detail: str, n: int | None = None) -> None:
        rows.append({"check": check, "status": "pass", "detail": detail, "n": n})

    dup = long.group_by("source", "model", "nfl_id").len().filter(pl.col("len") > 1)
    if dup.height:
        raise ComparisonError(f"{dup.height} duplicated (source, model, athlete) prediction rows, e.g. {dup.head(3).to_dicts()}")
    ok("unique (source, model, athlete)", "no duplicates", long.height)
    clash = long.group_by("model", "nfl_id").agg(pl.col("source").n_unique().alias("k")).filter(pl.col("k") > 1)
    if clash.height:
        raise ComparisonError(f"model names occur in several sources ({clash['model'].unique().to_list()}); rename before comparing")
    ok("model names unique across sources", "each model has exactly one source")
    yr = long.group_by("nfl_id").agg((pl.col("y").max() - pl.col("y").min()).alias("range"), pl.col("y").null_count().alias("nulls"))
    bad = yr.filter((pl.col("range") > tol) | (pl.col("nulls") > 0))
    if bad.height:
        raise ComparisonError(f"target differs across prediction sources for {bad.height} athlete(s), e.g. {bad.head(3).to_dicts()}")
    ok("identical frozen target per athlete", f"max |Δy| ≤ {tol}", yr.height)
    fr = long.group_by("nfl_id").agg(pl.col("fold").n_unique().alias("k"), pl.col("fold").first().alias("fold"))
    if fr.filter(pl.col("k") > 1).height:
        raise ComparisonError(f"{fr.filter(pl.col('k') > 1).height} athlete(s) carry different folds in different sources (fold isolation broken)")
    ok("one outer fold per athlete (all sources)", "fold isolation", fr.height)
    if folds_by_athlete is not None:
        mism = [(int(i), int(f), folds_by_athlete.get(int(i))) for i, f in zip(fr["nfl_id"].to_list(), fr["fold"].to_list())
                if folds_by_athlete.get(int(i)) is None or int(folds_by_athlete[int(i)]) != int(f)]
        if mism:
            raise ComparisonError(f"{len(mism)} athlete fold(s) differ from the experiment fold map, e.g. {mism[:3]}")
        ok("folds equal the experiment fold map", "Experiment.folds_by_athlete", fr.height)
    return pl.DataFrame(rows, infer_schema_length=None)


def reproducibility_check(ladder: Any, baseline: Any, tol: float = 1e-8) -> dict[str, Any]:
    """The baseline run's M1 must equal the ladder's M1 on the same athletes (same engine, frame, folds, seed)."""
    import polars as pl

    j = ladder.select("nfl_id", pl.col("pred_M1").alias("a"), pl.col("y").alias("ya")).join(
        baseline.select("nfl_id", pl.col("pred_M1_check").alias("b"), pl.col("y").alias("yb")), on="nfl_id", how="full", coalesce=True)
    n_only = int(j.filter(pl.col("a").is_null() | pl.col("b").is_null()).height)
    d = j.drop_nulls()
    diff = float((d["a"] - d["b"]).abs().max()) if d.height else NAN
    ydiff = float((d["ya"] - d["yb"]).abs().max()) if d.height else NAN
    status = "pass" if (n_only == 0 and np.isfinite(diff) and diff <= tol and ydiff <= tol) else "FAIL"
    return {"check": "baseline-run M1 reproduces ladder M1", "status": status, "max_abs_pred_diff": diff, "max_abs_y_diff": ydiff, "athletes_in_one_only": n_only}


# ====================================================================== D — cohorts
def cohorts(long: Any, oof: Any, primary: Sequence[str], f0_ids: set[int], encoders: Sequence[str]) -> tuple[dict[str, set[int]], Any]:
    """Named cohorts + a description table.

    * ``ladder_all``     every athlete with a finite prediction from EVERY primary model (embedding / M1 / F0 gaps are
                          imputed INSIDE the design with train medians + missing indicators);
    * ``strict_common``  ladder_all ∩ complete M1 ∩ observed F0 ∩ a real (non-imputed) held-out embedding for every
                          primary encoder — the cohort of every performance claim.
    """
    import polars as pl

    fin = long.filter(pl.col("model").is_in(list(primary)) & pl.col("pred").is_finite())
    cnt = fin.group_by("nfl_id").agg(pl.col("model").n_unique().alias("k"))
    ladder_all = set(int(i) for i in cnt.filter(pl.col("k") == len(set(primary)))["nfl_id"].to_list())
    m1c = [c for c in M1_COLS if c in oof.columns]
    complete = set(int(i) for i in oof.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in m1c]))["nfl_id"].to_list()) if m1c else set(ladder_all)
    native = set(ladder_all)
    for e in encoders:
        c = f"has_{e}"
        if c in oof.columns:
            native &= set(int(i) for i in oof.filter(pl.col(c))["nfl_id"].to_list())
    strict = ladder_all & complete & f0_ids & native
    sets = {"ladder_all": ladder_all, "strict_common": strict}
    desc = pl.DataFrame([{"cohort": "ladder_all", "n_athletes": len(ladder_all), "definition": "finite OOF prediction from every primary model (design-internal imputation)"},
                         {"cohort": "strict_common", "n_athletes": len(strict),
                          "definition": "ladder_all ∩ complete M1 ∩ observed F0 ∩ real held-out embedding for every primary encoder"},
                         {"cohort": "excluded from strict (M1 incomplete)", "n_athletes": len(ladder_all - complete), "definition": "diagnostic"},
                         {"cohort": "excluded from strict (no F0)", "n_athletes": len(ladder_all - f0_ids), "definition": "diagnostic"},
                         {"cohort": "excluded from strict (missing embedding)", "n_athletes": len(ladder_all - native), "definition": "diagnostic"}])
    return sets, desc


def native_cohort(model: str, oof: Any, f0_ids: set[int], ladder_ids: set[int]) -> set[int]:
    """The athletes a model predicts with its OWN observed representation (no imputation of that representation)."""
    import polars as pl

    if model.startswith("M1+F") and model != "M1+F0":
        c = f"has_{model.removeprefix('M1+')}"
        if c in oof.columns:
            return set(int(i) for i in oof.filter(pl.col(c))["nfl_id"].to_list()) & ladder_ids
    if model == "M1+F0":
        return f0_ids & ladder_ids
    return set(ladder_ids)


def wide(long: Any, ids: set[int], models: Sequence[str]) -> Any:
    """Cohort-restricted wide frame (nfl_id, position_group, fold, y, pred_<m>) — every model present for every athlete."""
    import polars as pl

    d = long.filter(pl.col("nfl_id").is_in(list(ids)) & pl.col("model").is_in(list(models)))
    if d.group_by("model", "nfl_id").len().filter(pl.col("len") > 1).height:
        raise ComparisonError("duplicated (model, athlete) predictions: run validate_alignment first")
    meta = d.group_by("nfl_id").agg(pl.col("position_group").first(), pl.col("fold").first(), pl.col("y").first())
    p = d.pivot(on="model", index="nfl_id", values="pred")
    missing = [m for m in models if m not in p.columns]
    if missing:
        raise ComparisonError(f"models {missing} have no predictions in this cohort")
    w = meta.join(p.select("nfl_id", *models), on="nfl_id", how="inner").rename({m: f"pred_{m}" for m in models}).sort("nfl_id")
    nulls = {m: int(w[f"pred_{m}"].is_null().sum() + (~w[f"pred_{m}"].is_finite()).sum()) for m in models}
    if any(nulls.values()):
        raise ComparisonError(f"non-finite predictions inside the cohort: { {k: v for k, v in nulls.items() if v} }")
    if w.height != len(ids):
        raise ComparisonError(f"cohort has {len(ids)} athletes but {w.height} aligned rows")
    return w


# ====================================================================== metrics
def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2))) if y.size else NAN


def cohort_metrics(w: Any, models: Sequence[str], cohort: str, position_model: str = "POS") -> Any:
    """Per model on ONE cohort frame: n, RMSE, MAE, R², within-position R² (vs in-cohort position means), skill vs the
    OOF position-only model, mean error (pred − y), error quantiles, Spearman, fold RMSE spread."""
    import polars as pl

    from .stats import spearman

    y = w["y"].to_numpy().astype(float)
    pos = w["position_group"].to_numpy()
    folds = w["fold"].to_numpy()
    pm = np.zeros(y.size)
    for p in np.unique(pos):
        pm[pos == p] = y[pos == p].mean()
    sse_pos_insample = float(((y - pm) ** 2).sum())
    sse_pos_oof = float(((y - w[f"pred_{position_model}"].to_numpy()) ** 2).sum()) if f"pred_{position_model}" in w.columns else NAN
    var = float(((y - y.mean()) ** 2).sum())
    rows = []
    for m in models:
        p = w[f"pred_{m}"].to_numpy().astype(float)
        e = p - y
        sse = float((e ** 2).sum())
        fr = [_rmse(y[folds == f], p[folds == f]) for f in np.unique(folds)]
        q = np.quantile(e, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append({"cohort": cohort, "model": m, "n_athletes": int(y.size), "rmse": _rmse(y, p), "mae": float(np.abs(e).mean()), "r2": 1 - sse / var if var > 0 else NAN,
                     "r2_within_position": 1 - sse / sse_pos_insample if sse_pos_insample > 0 else NAN,
                     "skill_vs_position_oof": 1 - sse / sse_pos_oof if np.isfinite(sse_pos_oof) and sse_pos_oof > 0 else NAN,
                     "mean_error": float(e.mean()), "err_q05": q[0], "err_q25": q[1], "err_q50": q[2], "err_q75": q[3], "err_q95": q[4],
                     "spearman": spearman(p, y), "fold_rmse_mean": float(np.mean(fr)), "fold_rmse_sd": float(np.std(fr, ddof=1)) if len(fr) > 1 else NAN,
                     "fold_rmse_min": float(np.min(fr)), "fold_rmse_max": float(np.max(fr)), "n_folds": len(fr)})
    return pl.DataFrame(rows)


def bootstrap_index(folds: np.ndarray, n_boot: int, seed: int, stratify: bool = True) -> np.ndarray:
    """(n_boot, n) athlete indices; fold-stratified (each fold resampled within itself, sizes kept) by default. ONE matrix
    is shared by every comparison of a table (paired across models)."""
    rng = np.random.default_rng(seed)
    n = folds.size
    if not stratify:
        return rng.integers(0, n, (n_boot, n))
    groups = [np.flatnonzero(folds == f) for f in np.unique(folds)]
    return np.stack([np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups]) for _ in range(n_boot)])


def paired_comparisons(w: Any, pairs: Sequence[tuple[str, str]], n_boot: int, seed: int, cohort: str, stratify: bool = True,
                       primary_family: Sequence[tuple[str, str]] = ()) -> Any:
    """Paired athlete bootstrap of Δ RMSE = RMSE(ref) − RMSE(model) and Δ MAE on one cohort frame. Holm across the
    declared primary family; BH across all rows. ``p_boot_le0`` = bootstrap share of Δ ≤ 0 (one-sided, descriptive)."""
    import polars as pl

    from .. import _backend

    ae = _backend.ae()
    y = w["y"].to_numpy().astype(float)
    folds = w["fold"].to_numpy()
    if y.size < 20:
        raise ComparisonError(f"cohort {cohort} has {y.size} athletes (< 20): no bootstrap interval")
    idx = bootstrap_index(folds, n_boot, seed, stratify)
    rows = []
    for mod, ref in pairs:
        if f"pred_{mod}" not in w.columns or f"pred_{ref}" not in w.columns:
            continue
        a, b = w[f"pred_{ref}"].to_numpy().astype(float), w[f"pred_{mod}"].to_numpy().astype(float)
        ea, eb = (y - a) ** 2, (y - b) ** 2
        d = np.sqrt(ea[idx].mean(1)) - np.sqrt(eb[idx].mean(1))
        dm = np.abs(y - a)[idx].mean(1) - np.abs(y - b)[idx].mean(1)
        r_ref, r_mod = _rmse(y, a), _rmse(y, b)
        lo, hi = np.quantile(d, [0.025, 0.975])
        rows.append({"cohort": cohort, "model": mod, "reference": ref, "n_athletes": int(y.size), "rmse_ref": r_ref, "rmse_model": r_mod,
                     "delta_rmse": r_ref - r_mod, "delta_lo": float(lo), "delta_hi": float(hi), "delta_pct": 100 * (r_ref - r_mod) / r_ref if r_ref > 0 else NAN,
                     "delta_mae": float(np.abs(y - a).mean() - np.abs(y - b).mean()), "delta_mae_lo": float(np.quantile(dm, 0.025)),
                     "delta_mae_hi": float(np.quantile(dm, 0.975)), "p_boot_le0": float((d <= 0).mean()), "n_boot": int(n_boot),
                     "bootstrap": "fold-stratified athlete" if stratify else "athlete", "primary": (mod, ref) in set(primary_family)})
    out = pl.DataFrame(rows, infer_schema_length=None)
    if not out.height:
        return out
    out = out.with_columns(pl.when(pl.col("delta_lo") > 0).then(pl.lit("model_better")).when(pl.col("delta_hi") < 0).then(pl.lit("model_worse"))
                           .otherwise(pl.lit("inconclusive")).alias("verdict"))
    prim = out.filter(pl.col("primary"))
    holm = dict(zip(zip(prim["model"].to_list(), prim["reference"].to_list()), ae.holm(prim["p_boot_le0"].to_list()))) if prim.height else {}
    out = out.with_columns(pl.Series("p_holm_primary", [holm.get((m, r), None) for m, r in zip(out["model"].to_list(), out["reference"].to_list())],
                                     dtype=pl.Float64),
                           pl.Series("p_bh_all", ae.bh(out["p_boot_le0"].to_list())))
    return out


def rmse_intervals(w: Any, models: Sequence[str], n_boot: int, seed: int, cohort: str) -> Any:
    """RMSE of each model with a fold-stratified athlete-bootstrap 95 % interval (same resamples for all models)."""
    import polars as pl

    y = w["y"].to_numpy().astype(float)
    idx = bootstrap_index(w["fold"].to_numpy(), n_boot, seed)
    rows = []
    for m in models:
        e = (y - w[f"pred_{m}"].to_numpy().astype(float)) ** 2
        bs = np.sqrt(e[idx].mean(1))
        rows.append({"cohort": cohort, "model": m, "rmse": float(np.sqrt(e.mean())), "rmse_lo": float(np.quantile(bs, 0.025)), "rmse_hi": float(np.quantile(bs, 0.975))})
    return pl.DataFrame(rows)


def fold_table(w: Any, models: Sequence[str], reference: str = "M1") -> Any:
    import polars as pl

    rows = []
    y, folds = w["y"].to_numpy().astype(float), w["fold"].to_numpy()
    for f in np.unique(folds):
        k = folds == f
        r_ref = _rmse(y[k], w[f"pred_{reference}"].to_numpy()[k]) if f"pred_{reference}" in w.columns else NAN
        for m in models:
            r = _rmse(y[k], w[f"pred_{m}"].to_numpy()[k])
            rows.append({"model": m, "fold": int(f), "n": int(k.sum()), "rmse": r, f"delta_rmse_vs_{reference}": r_ref - r})
    return pl.DataFrame(rows)


def comparison_pairs(models: Sequence[str], encoders_primary: Sequence[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(all pairs, primary family). Every model vs M1 and vs M1+F0; M1 vs POS; primary = each primary encoder vs M1."""
    pairs = [(m, "M1") for m in models if m not in ("M1", *BASELINES)]
    pairs += [(m, "M1+F0") for m in models if m not in ("M1", "M1+F0", *BASELINES)]
    pairs += [("M1", "POS")] if "POS" in models else []
    pairs += [("POS", "MEAN")] if {"POS", "MEAN"} <= set(models) else []
    prim = [(f"M1+{e}", "M1") for e in encoders_primary if f"M1+{e}" in models]
    seen, out = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out, prim


def residual_long(w: Any, models: Sequence[str]) -> Any:
    import polars as pl

    return pl.concat([w.select("nfl_id", "position_group", "fold", "y", pl.lit(m).alias("model"), pl.col(f"pred_{m}").alias("pred")) for m in models],
                     how="vertical").with_columns((pl.col("y") - pl.col("pred")).alias("residual"))


def rank_agreement(w: Any, models: Sequence[str], top_share: float = 0.1) -> Any:
    """Long matrix: Spearman between two models' predictions and the overlap of their predicted top decile."""
    import polars as pl

    from .stats import spearman

    n = w.height
    k = max(1, int(round(top_share * n)))
    P = {m: w[f"pred_{m}"].to_numpy().astype(float) for m in models}
    top = {m: set(np.argsort(-P[m], kind="stable")[:k].tolist()) for m in models}
    return pl.DataFrame([{"model_a": a, "model_b": b, "spearman": spearman(P[a], P[b]), "top_overlap": len(top[a] & top[b]) / k} for a in models for b in models])


def best_worst(w: Any, model: str, reference: str, k: int = 10) -> Any:
    """Athletes where the model most improves / most worsens on the reference (|res_ref| − |res_model|)."""
    import polars as pl

    d = w.select("nfl_id", "position_group", "fold", "y", pl.col(f"pred_{reference}").alias("pred_ref"), pl.col(f"pred_{model}").alias("pred_model")).with_columns(
        ((pl.col("y") - pl.col("pred_ref")).abs() - (pl.col("y") - pl.col("pred_model")).abs()).alias("abs_error_gain"))
    return pl.concat([d.sort("abs_error_gain", descending=True).head(k).with_columns(pl.lit("largest improvement").alias("case")),
                      d.sort("abs_error_gain").head(k).with_columns(pl.lit("largest deterioration").alias("case"))], how="vertical").with_columns(
        pl.lit(model).alias("model"), pl.lit(reference).alias("reference"))


__all__ = ["BASELINES", "ComparisonError", "baseline_predictions", "best_worst", "bootstrap_index", "cohort_metrics", "cohorts", "comparison_pairs", "f0_available",
           "fold_table", "ladder_predictions", "long_predictions", "native_cohort", "paired_comparisons", "rank_agreement", "reproducibility_check",
           "residual_long", "rmse_intervals", "validate_alignment", "wide"]
