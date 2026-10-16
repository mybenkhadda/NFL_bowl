"""Part E — NFL outcome transfer on the frozen 12C target ``nfl.q90_shared``.

Source of every number: the out-of-fold predictions saved by ``Experiment.transfer_predictions`` (the exact ladder of
``Experiment.transfer``: same rows, folds, train-fold-only design, fold-fitted embedding PCA, inner-CV ridge α).
Paired comparisons use the Milestone 11 estimator ``ev.paired_compare`` (Δ RMSE = RMSE(ref) − RMSE(model), positive =
model better, paired athlete bootstrap).

Cohorts. The ladder scores every model on the SAME athletes (common cohort by construction); an athlete without an
embedding in its held-out fold gets the train-median imputation + missing indicator inside the design. The
"native coverage" view repeats each comparison on the athletes that actually have that representation, so the
effect of imputation is visible.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..data.validation import M1_COLS

NAN = float("nan")


def _ev() -> Any:
    from nflbdb.research.advanced_movement.m11 import evaluation as ev

    return ev


def ladder_models(oof: Any) -> list[str]:
    return [c.removeprefix("pred_") for c in oof.columns if c.startswith("pred_")]


def residual_frame(oof: Any) -> Any:
    """Long: nfl_id, position_group, fold, y, model, pred, residual (= observed − predicted)."""
    import polars as pl

    parts = [oof.select("nfl_id", "position_group", "fold", "y", pl.lit(m).alias("model"), pl.col(f"pred_{m}").alias("pred")) for m in ladder_models(oof)]
    return pl.concat(parts, how="vertical").with_columns((pl.col("y") - pl.col("pred")).alias("residual"))


def prediction_metrics(oof: Any) -> Any:
    """Per model: n, RMSE, MAE, R², mean signed error (pred − y), residual quantiles."""
    import polars as pl

    rows = []
    y = oof["y"].cast(float).to_numpy()
    var = float(np.var(y))
    for m in ladder_models(oof):
        p = oof[f"pred_{m}"].cast(float).to_numpy()
        ok = np.isfinite(p) & np.isfinite(y)
        r = y[ok] - p[ok]
        q = np.quantile(r, [0.05, 0.25, 0.5, 0.75, 0.95]) if r.size else [NAN] * 5
        rows.append({"model": m, "n_athletes": int(ok.sum()), "rmse": float(np.sqrt(np.mean(r ** 2))) if r.size else NAN, "mae": float(np.mean(np.abs(r))) if r.size else NAN,
                     "r2": float(1 - np.mean(r ** 2) / var) if var > 0 and r.size else NAN, "mean_signed_error": float(np.mean(-r)) if r.size else NAN,
                     "resid_q05": q[0], "resid_q25": q[1], "resid_q50": q[2], "resid_q75": q[3], "resid_q95": q[4]})
    return pl.DataFrame(rows)


def fold_metrics(oof: Any) -> Any:
    import polars as pl

    rows = []
    for f in sorted(oof["fold"].unique().to_list()):
        g = oof.filter(pl.col("fold") == f)
        y = g["y"].cast(float).to_numpy()
        for m in ladder_models(oof):
            p = g[f"pred_{m}"].cast(float).to_numpy()
            ok = np.isfinite(p)
            rows.append({"model": m, "fold": int(f), "n": int(ok.sum()), "rmse": float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2))) if ok.any() else NAN})
    d = pl.DataFrame(rows)
    ref = d.filter(pl.col("model") == "M1").select("fold", pl.col("rmse").alias("rmse_M1"))
    return d.join(ref, on="fold", how="left").with_columns((pl.col("rmse_M1") - pl.col("rmse")).alias("delta_rmse_vs_M1"))


def comparison_pairs(models: Sequence[str]) -> list[tuple[str, str]]:
    """Declared comparisons: every model vs M1; every tracking model vs M1+F0; encoder vs encoder (F4 vs F2)."""
    out = [(m, "M1") for m in models if m != "M1"]
    out += [(m, "M1+F0") for m in models if m not in ("M1", "M1+F0")]
    enc = [m for m in models if m.startswith("M1+F") and m not in ("M1+F0", "M1+F1_TCN")]
    f4 = [m for m in enc if "F4" in m]
    f2 = [m for m in enc if "F2" in m]
    if f4 and f2:
        out.append((f4[0], f2[0]))
    return out


def paired_comparisons(oof: Any, n_boot: int, seed: int) -> Any:
    """Common cohort (all ladder rows) AND native coverage (rows with a real embedding for every encoder involved)."""
    import polars as pl

    ev = _ev()
    y = oof["y"].cast(float).to_numpy()
    rows = []
    for mod, ref in comparison_pairs(ladder_models(oof)):
        a, b = oof[f"pred_{ref}"].cast(float).to_numpy(), oof[f"pred_{mod}"].cast(float).to_numpy()
        r = ev.paired_compare(y, a, b, n_boot, seed)
        rows.append({"cohort": "common", "model": mod, "reference": ref, **r, "verdict": ev.verdict(r)})
        flags = [f"has_{x.removeprefix('M1+')}" for x in (mod, ref) if f"has_{x.removeprefix('M1+')}" in oof.columns]
        if flags:
            keep = oof.select(pl.all_horizontal([pl.col(c) for c in flags]).alias("_k"))["_k"].to_numpy()
            r2 = ev.paired_compare(y[keep], a[keep], b[keep], n_boot, seed)
            rows.append({"cohort": "native_coverage", "model": mod, "reference": ref, **r2, "verdict": ev.verdict(r2)})
    return pl.DataFrame(rows, infer_schema_length=None)


def position_effects(oof: Any, model: str, reference: str, n_boot: int, seed: int, min_n: int = 10, adequate_n: int = 30) -> Any:
    """Per position: paired Δ RMSE with reliability labels (insufficient < min_n: no interval; exploratory < adequate_n)."""
    import polars as pl

    from .stats import bh

    ev = _ev()
    rows = []
    for p in sorted(oof["position_group"].unique().to_list(), key=str):
        g = oof.filter(pl.col("position_group") == p)
        n = g.height
        rel = "insufficient" if n < min_n else ("exploratory" if n < adequate_n else "adequate")
        r = ev.paired_compare(g["y"].cast(float).to_numpy(), g[f"pred_{reference}"].cast(float).to_numpy(), g[f"pred_{model}"].cast(float).to_numpy(),
                              n_boot if rel != "insufficient" else 0, seed)
        if rel == "insufficient":
            r.update({"delta_lo": NAN, "delta_hi": NAN, "p_boot_le0": NAN})
        rows.append({"position_group": p, "model": model, "reference": reference, "reliability": rel, **r, "verdict": ev.verdict(r)})
    d = pl.DataFrame(rows, infer_schema_length=None)
    if d.height:
        d = d.with_columns(pl.Series("p_boot_le0_bh", bh([1.0 if (v is None or not np.isfinite(v)) else v for v in d["p_boot_le0"].to_list()])))
    return d


def error_analysis(oof: Any, models: Sequence[str]) -> dict[str, Any]:
    """Where models fail: extreme residual athletes, errors by target decile, M1-incomplete athletes, and the residual
    association with each official measurement (Spearman)."""
    import polars as pl

    from .stats import spearman

    res = residual_frame(oof).filter(pl.col("model").is_in(list(models)))
    y = oof["y"].cast(float)
    q10, q90 = float(y.quantile(0.1)), float(y.quantile(0.9))
    res = res.with_columns(pl.when(pl.col("y") >= q90).then(pl.lit("top decile")).when(pl.col("y") <= q10).then(pl.lit("bottom decile"))
                           .otherwise(pl.lit("middle")).alias("capacity_band"))
    band = res.group_by("model", "capacity_band").agg(pl.len().alias("n"), pl.col("residual").mean().alias("mean_residual"),
                                                      (pl.col("residual") ** 2).mean().sqrt().alias("rmse")).sort("model", "capacity_band")
    m1c = [c for c in M1_COLS if c in oof.columns]
    comp = oof.select("nfl_id", pl.all_horizontal([pl.col(c).is_not_null() for c in m1c]).alias("m1_complete")) if m1c else oof.select("nfl_id", pl.lit(True).alias("m1_complete"))
    res = res.join(comp, on="nfl_id", how="left")
    inc = res.group_by("model", "m1_complete").agg(pl.len().alias("n"), (pl.col("residual") ** 2).mean().sqrt().alias("rmse"),
                                                   pl.col("residual").mean().alias("mean_residual")).sort("model", "m1_complete")
    extremes = []
    for m in models:
        g = res.filter(pl.col("model") == m)
        if g.height < 20:
            continue
        lo, hi = float(g["residual"].quantile(0.05)), float(g["residual"].quantile(0.95))
        extremes.append(g.filter((pl.col("residual") <= lo) | (pl.col("residual") >= hi)).with_columns(
            pl.when(pl.col("residual") > 0).then(pl.lit("underprediction")).otherwise(pl.lit("overprediction")).alias("error_type")))
    assoc = []
    for m in models:
        g = res.filter(pl.col("model") == m).join(oof.select("nfl_id", *m1c), on="nfl_id", how="left")
        for c in m1c:
            assoc.append({"model": m, "official": c, "spearman_residual": spearman(g["residual"].to_numpy(), g[c].cast(float).to_numpy())})
    return {"by_capacity": band, "by_m1_completeness": inc, "extremes": pl.concat(extremes, how="vertical") if extremes else pl.DataFrame(),
            "residual_vs_officials": pl.DataFrame(assoc, infer_schema_length=None)}


def scouting_metrics(oof: Any, models: Sequence[str], reference: str, n_boot: int, seed: int, top_share: float = 0.1) -> dict[str, Any]:
    """Decision-oriented, descriptive: rank agreement with the outcome, top-decile precision / recall (athlete
    bootstrap), ranking changes vs the reference, decile calibration, threshold sensitivity."""
    import polars as pl

    from .stats import spearman

    y = oof["y"].cast(float).to_numpy()
    n = y.size
    k = max(1, int(round(top_share * n)))
    true_top = np.zeros(n, bool)
    true_top[np.argsort(-y)[:k]] = True
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n)) if n_boot > 0 and n >= 20 else None
    rank_rows, cal_rows, thr_rows = [], [], []
    pref = oof[f"pred_{reference}"].cast(float).to_numpy()
    for m in models:
        p = oof[f"pred_{m}"].cast(float).to_numpy()
        sel = np.zeros(n, bool)
        sel[np.argsort(-p)[:k]] = True
        prec = float((sel & true_top).sum() / k)
        lo = hi = NAN
        if idx is not None:
            bs = []
            for i in idx:
                yy, pp = y[i], p[i]
                kk = max(1, int(round(top_share * yy.size)))
                tt = np.zeros(yy.size, bool)
                tt[np.argsort(-yy, kind="stable")[:kk]] = True
                ss = np.zeros(yy.size, bool)
                ss[np.argsort(-pp, kind="stable")[:kk]] = True
                bs.append((ss & tt).sum() / kk)
            lo, hi = (float(v) for v in np.quantile(bs, [0.025, 0.975]))
        rk_m = np.argsort(np.argsort(-p))
        rk_r = np.argsort(np.argsort(-pref))
        rank_rows.append({"model": m, "spearman_with_outcome": spearman(p, y), "top_share": top_share, "top_k": k, "top_precision": prec, "top_precision_lo": lo,
                          "top_precision_hi": hi, "top_precision_chance": top_share, "spearman_with_reference_pred": spearman(p, pref),
                          "mean_abs_rank_shift_vs_reference": float(np.mean(np.abs(rk_m - rk_r))),
                          "top_set_overlap_with_reference": float((sel & (np.argsort(np.argsort(-pref)) < k)).sum() / k)})
        qs = np.quantile(p, np.linspace(0, 1, 11))
        for b in range(10):
            mk = (p >= qs[b]) & (p <= qs[b + 1]) if b == 9 else (p >= qs[b]) & (p < qs[b + 1])
            if mk.sum():
                cal_rows.append({"model": m, "bin": b + 1, "n": int(mk.sum()), "pred_mean": float(p[mk].mean()), "obs_mean": float(y[mk].mean())})
        slope = float(np.polyfit(p, y, 1)[0]) if np.std(p) > 0 else NAN
        rank_rows[-1]["calibration_slope"] = slope
        for tq in (0.5, 0.75, 0.9, 0.95):
            ty = np.quantile(y, tq)
            tp = np.quantile(p, tq)
            s_, t_ = p >= tp, y >= ty
            thr_rows.append({"model": m, "threshold_quantile": tq, "selected": int(s_.sum()), "precision": float((s_ & t_).sum() / max(s_.sum(), 1)),
                             "recall": float((s_ & t_).sum() / max(t_.sum(), 1)), "base_rate": float(t_.mean())})
    return {"ranking": pl.DataFrame(rank_rows, infer_schema_length=None), "calibration": pl.DataFrame(cal_rows), "thresholds": pl.DataFrame(thr_rows)}


def permutation_test(exp: Any, model: str, n_perm: int, seed: int) -> dict[str, Any]:
    """Within-position permutation null of Δ RMSE (M1 → M1+<model>) via ``ae.permutation_blocks`` (fold-fitted block
    refitted per fold). Expensive: n_perm full ladders."""
    from .. import _backend
    from ..data.validation import TRANSFER_TARGET
    from ..models import registry as reg

    ae = _backend.ae()
    n = reg.resolve(model)
    names, _byf, _deps, data, mods = exp._ladder_inputs([n])
    C = exp.cfg.science.to_m12config()
    r = ae.permutation_blocks(data, TRANSFER_TARGET, mods["M1"], mods[f"M1+{n}"], C, seed, n_perm=n_perm)
    return {"model": f"M1+{n}", "reference": "M1", **r}


__all__ = ["comparison_pairs", "error_analysis", "fold_metrics", "ladder_models", "paired_comparisons", "permutation_test", "position_effects",
           "prediction_metrics", "residual_frame", "scouting_metrics"]
