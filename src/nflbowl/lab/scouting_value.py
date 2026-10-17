"""Parts H + I — why retrieval does not guarantee transfer, and the scouting value of the validated predictions.

Part H joins per (representation, outer fold) records: inductive retrieval AUC (``Experiment.evaluate(regime=
'inductive')``), effective rank of the held-out embeddings, domain AUC / MMD² (Part G ``original``), M12 physics-probe
summaries when saved, and Δ RMSE of the transfer ladder on the strict common cohort per fold. Associations are read at
two levels only: across MODELS (≤ 7 points, exact permutation p-value, descriptive) and WITHIN models across folds
after removing model and fold means (two-way demeaning) — five folds of one model are never treated as five
architectures.

Part I uses OUT-OF-FOLD predictions only. Prediction intervals: for an athlete of fold f, the empirical residual
quantiles of athletes from the OTHER folds (same position when ≥ 30 such athletes, else pooled) — an approximate
split-conformal construction whose empirical coverage is reported; intervals are shown in profiles only when the
coverage is within 10 points of nominal.

The target is a constructed NFL movement-capacity quantity; nothing here predicts NFL success.
"""

from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np

from ..data.validation import M1_COLS

NAN = float("nan")


# ====================================================================== H — records
def retrieval_records(exp: Any, models: Sequence[str], folds: Sequence[int]) -> Any:
    import polars as pl

    from ..models import registry as reg

    parts = []
    for m in models:
        try:
            r = exp.evaluate(reg.resolve(m), regime="inductive", folds=list(folds))
        except FileNotFoundError:
            continue
        if r is not None and r.height:
            parts.append(r.select(pl.col("representation"), pl.col("held_out_fold").cast(pl.Int64).alias("fold"), pl.col("auc").cast(pl.Float64).alias("retrieval_auc"),
                                  *[pl.col(c) for c in ("n_athletes",) if c in r.columns]))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame(schema={"representation": pl.String, "fold": pl.Int64, "retrieval_auc": pl.Float64})


def geometry_records(exp: Any, models: Sequence[str], folds: Sequence[int]) -> Any:
    import polars as pl

    from .geometry import held_out_rows, inductive_matrix
    from .stats import effective_rank

    rows = []
    for m in models:
        for f in folds:
            E = inductive_matrix(exp, m, f)
            rows.append({"representation": m, "fold": int(f), **effective_rank(E[held_out_rows(exp, f)])})
    return pl.DataFrame(rows, infer_schema_length=None)


def probe_summary(m12_probes: Any) -> Any:
    """Per representation mean of the first available probe score column of the saved M12 probe table (declared)."""
    import polars as pl

    if m12_probes is None or not m12_probes.height or "representation" not in m12_probes.columns:
        return pl.DataFrame(schema={"representation": pl.String, "probe_score": pl.Float64, "probe_column": pl.String})
    col = next((c for c in ("r2", "spearman", "value", "score") if c in m12_probes.columns), None)
    if col is None:
        return pl.DataFrame(schema={"representation": pl.String, "probe_score": pl.Float64, "probe_column": pl.String})
    return m12_probes.group_by("representation").agg(pl.col(col).cast(pl.Float64).mean().alias("probe_score")).with_columns(pl.lit(col).alias("probe_column"))


def transfer_records(fold_tab: Any, f0_model: str = "M1+F0") -> Any:
    """Per (ladder model, fold): Δ RMSE vs M1 and vs M1+F0 (strict cohort fold table)."""
    import polars as pl

    ref = fold_tab.filter(pl.col("model") == f0_model).select("fold", pl.col("rmse").alias("rmse_f0"))
    m1 = fold_tab.filter(pl.col("model") == "M1").select("fold", pl.col("rmse").alias("rmse_m1"))
    d = fold_tab.filter(pl.col("model").str.starts_with("M1+F")).join(ref, on="fold", how="left").join(m1, on="fold", how="left")
    return d.with_columns(pl.col("model").str.replace("M1+", "", literal=True).alias("representation_short"),
                          (pl.col("rmse_m1") - pl.col("rmse")).alias("delta_rmse_vs_M1"), (pl.col("rmse_f0") - pl.col("rmse")).alias("delta_rmse_vs_M1F0"))


def relationship_table(retr: Any, geo: Any, dom: Any, trans: Any, probes: Any) -> Any:
    """(representation, fold) records joined; retrieval gain = AUC − F0 AUC of the same fold."""
    import polars as pl

    d = retr.join(geo, on=["representation", "fold"], how="left")
    if dom is not None and dom.height:
        d = d.join(dom.filter(pl.col("condition") == "original").select("representation", "fold", "domain_auc_grouped", "mmd2"), on=["representation", "fold"], how="left")
    f0 = retr.filter(pl.col("representation") == "F0_engineered").select("fold", pl.col("retrieval_auc").alias("f0_auc"))
    d = d.join(f0, on="fold", how="left").with_columns((pl.col("retrieval_auc") - pl.col("f0_auc")).alias("retrieval_gain_vs_F0"))
    t = trans.select(pl.col("representation_short").alias("representation"), "fold", "delta_rmse_vs_M1", "delta_rmse_vs_M1F0")
    d = d.join(t, on=["representation", "fold"], how="left")
    if probes is not None and probes.height:
        d = d.join(probes.select("representation", "probe_score"), on="representation", how="left")
    return d.sort("representation", "fold")


def _exact_perm_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from .stats import spearman

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3:
        return NAN, NAN
    obs = spearman(x, y)
    if n > 8:
        return obs, NAN
    null = [spearman(x, y[list(p)]) for p in itertools.permutations(range(n))]
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def associations(rel: Any, x: str = "retrieval_gain_vs_F0", ys: Sequence[str] = ("delta_rmse_vs_M1", "delta_rmse_vs_M1F0"),
                 extra_x: Sequence[str] = ("effective_rank", "domain_auc_grouped", "mmd2", "retrieval_auc", "probe_score")) -> Any:
    import polars as pl

    from .stats import spearman

    rows = []
    reps = rel.filter(pl.col("representation") != "F0_engineered")
    model_level = reps.group_by("representation").agg(*[pl.col(c).mean() for c in {x, *ys, *extra_x} if c in reps.columns])
    for xv in [x, *extra_x]:
        if xv not in model_level.columns:
            continue
        for yv in ys:
            if yv not in model_level.columns:
                continue
            r, p = _exact_perm_spearman(model_level[xv].cast(pl.Float64).to_numpy(), model_level[yv].cast(pl.Float64).to_numpy())
            rows.append({"level": "across models (means over folds)", "x": xv, "y": yv, "n_points": model_level.height, "spearman": r, "perm_p_exact": p,
                         "note": "descriptive: very few models"})
    for yv in ys:
        if yv not in reps.columns or x not in reps.columns:
            continue
        d = reps.select("representation", "fold", pl.col(x).cast(pl.Float64), pl.col(yv).cast(pl.Float64)).drop_nulls()
        d = d.filter(pl.col(x).is_finite() & pl.col(yv).is_finite())
        if d.height < 6:
            continue
        dd = d.with_columns(*[(pl.col(c) - pl.col(c).mean().over("representation") - pl.col(c).mean().over("fold") + pl.col(c).mean()).alias(f"{c}__dm") for c in (x, yv)])
        rows.append({"level": "within model across folds (two-way demeaned)", "x": x, "y": yv, "n_points": dd.height,
                     "spearman": spearman(dd[f"{x}__dm"].to_numpy(), dd[f"{yv}__dm"].to_numpy()), "perm_p_exact": NAN,
                     "note": "fold fluctuations; folds are not independent architectures"})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== I — choosing the model to explain
def best_model(paired: Any, cohort: str = "strict_common", reference: str = "M1", exclude: Sequence[str] = ("POS", "MEAN")) -> dict[str, Any]:
    """The model with the largest Δ RMSE vs ``reference`` among those whose interval excludes 0 (validated); otherwise the
    largest point estimate, flagged as NOT validated."""
    import polars as pl

    d = paired.filter((pl.col("cohort") == cohort) & (pl.col("reference") == reference) & ~pl.col("model").is_in(list(exclude)))
    if not d.height:
        return {"model": None, "validated": False, "reason": "no comparison rows"}
    v = d.filter(pl.col("delta_lo") > 0).sort("delta_rmse", descending=True)
    if v.height:
        r = v.row(0, named=True)
        return {"model": r["model"], "validated": True, "delta_rmse": r["delta_rmse"], "delta_lo": r["delta_lo"], "delta_hi": r["delta_hi"],
                "reason": "largest Δ RMSE among intervals excluding 0"}
    r = d.sort("delta_rmse", descending=True).row(0, named=True)
    return {"model": r["model"], "validated": False, "delta_rmse": r["delta_rmse"], "delta_lo": r["delta_lo"], "delta_hi": r["delta_hi"],
            "reason": "no interval excludes 0: largest point estimate shown for exploration only"}


def tracking_gain(w: Any, model: str, reference: str = "M1") -> Any:
    import polars as pl

    return w.select("nfl_id", "position_group", "fold", "y", pl.col(f"pred_{reference}").alias("pred_ref"), pl.col(f"pred_{model}").alias("pred_model")).with_columns(
        ((pl.col("y") - pl.col("pred_ref")).abs() - (pl.col("y") - pl.col("pred_model")).abs()).alias("abs_error_gain"),
        (pl.col("pred_model") - pl.col("pred_ref")).alias("prediction_shift"))


def athlete_context(exp: Any, ath_features: Any | None) -> Any:
    """Label-free athlete descriptors for the help / hurt analysis: M1 completeness, Combine forty attempts, heading
    quality, forty quartile within position."""
    import polars as pl

    b = exp.bundle
    a = b.athletes.select("nfl_id", *[c for c in M1_COLS if c in b.athletes.columns])
    a = a.with_columns(pl.all_horizontal([pl.col(c).is_not_null() for c in M1_COLS if c in a.columns]).alias("m1_complete"))
    meta = b.meta.filter((pl.col("domain") == "combine") & (pl.col("kind") == "forty")).group_by("nfl_id").len().rename({"len": "n_forty_attempts"})
    a = a.join(meta, on="nfl_id", how="left").with_columns(pl.col("n_forty_attempts").fill_null(0))
    if ath_features is not None and "c_forty__n_seq" in ath_features.columns:
        a = a.join(ath_features.select("nfl_id", *[c for c in ("c_forty__peak_speed", "c_agility__n_seq") if c in ath_features.columns]), on="nfl_id", how="left")
    return a


def gain_by_group(gain: Any, context: Any, n_boot: int, seed: int, min_n: int = 10) -> Any:
    """Mean absolute-error gain (positive = tracking helps) with an athlete bootstrap interval per descriptor level."""
    import polars as pl

    from .stats import athlete_bootstrap_ci

    d = gain.join(context, on="nfl_id", how="left")
    if "forty" in d.columns:
        d = d.with_columns(((pl.col("forty").rank("ordinal").over("position_group").cast(pl.Int64) - 1) * 4 // pl.len().over("position_group").cast(pl.Int64) + 1)
                           .cast(pl.String).alias("forty_quartile_within_position"))
    d = d.with_columns(pl.when(pl.col("n_forty_attempts") >= 2).then(pl.lit("≥ 2")).when(pl.col("n_forty_attempts") == 1).then(pl.lit("1"))
                       .otherwise(pl.lit("0")).alias("forty_attempts"),
                       pl.when(pl.col("y") >= pl.col("y").quantile(2 / 3)).then(pl.lit("upper third")).when(pl.col("y") <= pl.col("y").quantile(1 / 3))
                       .then(pl.lit("lower third")).otherwise(pl.lit("middle third")).alias("target_tercile_posthoc"))
    rows = []
    for var in ("position_group", "m1_complete", "forty_attempts", "forty_quartile_within_position", "target_tercile_posthoc"):
        if var not in d.columns:
            continue
        for lvl in sorted(d[var].drop_nulls().unique().to_list(), key=str):
            v = d.filter(pl.col(var) == lvl)["abs_error_gain"].to_numpy().astype(float)
            est, lo, hi = athlete_bootstrap_ci(v, n_boot, seed) if v.size >= min_n else (float(np.mean(v)) if v.size else NAN, NAN, NAN)
            rows.append({"variable": var, "level": str(lvl), "n": int(v.size), "mean_abs_error_gain": est, "lo": lo, "hi": hi,
                         "share_helped": float((v > 0).mean()) if v.size else NAN,
                         "note": "post-hoc descriptive (conditions on the outcome)" if var == "target_tercile_posthoc" else ("small group" if v.size < min_n else "")})
    return pl.DataFrame(rows, infer_schema_length=None)


def forty_twins(w: Any, models: Sequence[str], data: Any, bins: int = 4, min_cell: int = 5) -> tuple[Any, Any]:
    """Athletes with similar forty times: within position × forty-quartile cells, how much of the target variance each
    model explains (pooled within-cell R²) — does tracking separate athletes the stopwatch cannot?"""
    import polars as pl

    d = w.join(data.select("nfl_id", "forty"), on="nfl_id", how="left").filter(pl.col("forty").is_not_null())
    d = d.with_columns(((pl.col("forty").rank("ordinal").over("position_group").cast(pl.Int64) - 1) * bins // pl.len().over("position_group").cast(pl.Int64) + 1)
                       .alias("forty_bin"))
    d = d.with_columns(pl.len().over("position_group", "forty_bin").alias("_n")).filter(pl.col("_n") >= min_cell)
    d = d.with_columns((pl.col("y") - pl.col("y").mean().over("position_group", "forty_bin")).alias("_yc"))
    den = float((d["_yc"] ** 2).sum())
    summary, cells = [], []
    for m in models:
        e = (d["y"] - d[f"pred_{m}"]).to_numpy()
        summary.append({"model": m, "n_athletes": d.height, "n_cells": int(d.select(pl.struct("position_group", "forty_bin").n_unique()).item()),
                        "within_cell_r2": 1 - float((e ** 2).sum()) / den if den > 0 else NAN})
    for (p, b), g in d.group_by(["position_group", "forty_bin"]):
        dc = float((g["_yc"] ** 2).sum())
        row = {"position_group": p, "forty_bin": int(b), "n": g.height, "forty_median": float(g["forty"].median()), "target_sd": float(g["y"].std())}
        for m in models:
            row[f"r2_{m}"] = 1 - float(((g["y"] - g[f"pred_{m}"]) ** 2).sum()) / dc if dc > 0 else NAN
        cells.append(row)
    return pl.DataFrame(summary), pl.DataFrame(cells, infer_schema_length=None).sort("position_group", "forty_bin") if cells else pl.DataFrame()


def unexplained_by_position(w: Any, models: Sequence[str], min_n: int = 10) -> Any:
    import polars as pl

    rows = []
    for p in sorted(w["position_group"].unique().to_list(), key=str):
        g = w.filter(pl.col("position_group") == p)
        y = g["y"].to_numpy().astype(float)
        var = float(y.var()) if y.size > 1 else NAN
        for m in models:
            mse = float(np.mean((y - g[f"pred_{m}"].to_numpy()) ** 2))
            rows.append({"position_group": p, "model": m, "n": int(y.size), "target_sd": float(np.sqrt(var)) if np.isfinite(var) else NAN, "rmse": float(np.sqrt(mse)),
                         "share_unexplained": mse / var if var and var > 0 else NAN, "small_group": y.size < min_n})
    return pl.DataFrame(rows)


def oof_intervals(w: Any, model: str, level: float = 0.8, min_position: int = 30) -> tuple[Any, dict[str, Any]]:
    """Leave-own-fold-out residual-quantile intervals + their empirical coverage (overall and per position)."""
    import polars as pl

    y = w["y"].to_numpy().astype(float)
    p = w[f"pred_{model}"].to_numpy().astype(float)
    r = y - p
    folds, pos = w["fold"].to_numpy(), w["position_group"].to_numpy()
    a = (1 - level) / 2
    lo, hi, scope = np.full(y.size, NAN), np.full(y.size, NAN), []
    for i in range(y.size):
        other = folds != folds[i]
        same = other & (pos == pos[i])
        ref = r[same] if same.sum() >= min_position else r[other]
        scope.append("position" if same.sum() >= min_position else "pooled")
        lo[i], hi[i] = p[i] + np.quantile(ref, a), p[i] + np.quantile(ref, 1 - a)
    cov = (y >= lo) & (y <= hi)
    out = w.select("nfl_id", "position_group", "fold").with_columns(pl.Series("pred", p), pl.Series("lo", lo), pl.Series("hi", hi), pl.Series("covered", cov),
                                                                    pl.Series("interval_scope", scope))
    by = out.group_by("position_group").agg(pl.col("covered").mean().alias("coverage"), pl.len().alias("n")).sort("position_group")
    overall = float(cov.mean())
    return out, {"model": model, "level": level, "coverage": overall, "valid": abs(overall - level) <= 0.10, "by_position": by.to_dicts(),
                 "method": "leave-own-fold-out residual quantiles (approximate split conformal)"}


def select_profiles(gain: Any, n: int, seed: int) -> Any:
    """n // 4 each: largest tracking gain, largest tracking loss, highest augmented prediction, seeded random (unique)."""
    import polars as pl

    q = max(1, n // 4)
    picks = [gain.sort("abs_error_gain", descending=True).head(q).with_columns(pl.lit("tracking helps most").alias("selection")),
             gain.sort("abs_error_gain").head(q).with_columns(pl.lit("tracking hurts most").alias("selection")),
             gain.sort("pred_model", descending=True).head(q).with_columns(pl.lit("highest predicted capacity").alias("selection")),
             gain.sample(min(q, gain.height), seed=seed).with_columns(pl.lit("random (seeded)").alias("selection"))]
    return pl.concat(picks, how="vertical").unique(subset=["nfl_id"], keep="first", maintain_order=True)


def _f(v: Any, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def profile_markdown(label: str, row: dict[str, Any], officials: dict[str, Any], movement: dict[str, tuple[Any, Any]], preds: dict[str, float],
                     interval: tuple[float, float] | None, interval_note: str, disagreement: float, notes: Sequence[str], model: str, reference: str) -> str:
    md = [f"# Athlete profile {label}", "", f"_Selection: **{row.get('selection')}** · position **{row.get('position_group')}** · outer fold {row.get('fold')}. "
          "Exploratory research profile of a constructed NFL movement-capacity quantity — not a personnel recommendation and not a prediction of NFL success._",
          "", "## Official Combine measurements", "", "| measure | value |", "|---|---|"]
    md += [f"| {k} | {_f(v, 2)} |" for k, v in officials.items()]
    md += ["", "## Interpretable movement metrics (Combine tracking)", "", "| metric | value | position percentile |", "|---|---|---|"]
    md += [f"| {k} | {_f(v)} | {_f(pct, 0)} |" for k, (v, pct) in movement.items()]
    md += ["", "## Predicted NFL movement capacity (out-of-fold, yd/s²)", "", "| model | prediction |", "|---|---|"]
    md += [f"| {k} | {_f(v)} |" for k, v in preds.items()]
    md += ["", f"- Augmented ({model}) − baseline ({reference}): **{_f(preds.get(model, NAN) - preds.get(reference, NAN))}**",
           f"- Model disagreement (SD across ladder models): {_f(disagreement)}",
           f"- Observed frozen target (OOF evaluation only): **{_f(row.get('y'))}**",
           f"- Prediction interval: {('[' + _f(interval[0]) + ', ' + _f(interval[1]) + ']') if interval else 'not shown'} — {interval_note}", "",
           "## Coverage and reliability notes", ""]
    md += [f"- {n}" for n in notes]
    return "\n".join(md) + "\n"


def position_percentiles(data: Any, cols: Sequence[str]) -> Any:
    """nfl_id + ``pct_<col>``: within-position percentile (0–100) of each observed metric (nulls stay null)."""
    import polars as pl

    use = [c for c in cols if c in data.columns]
    exprs = [(100.0 * (pl.col(c).rank("average").over("position_group") - 1) / (pl.col(c).count().over("position_group").cast(pl.Int64) - 1).clip(lower_bound=1)).alias(f"pct_{c}")
             for c in use]
    return data.with_columns(pl.col("position_group").fill_null("?")).select("nfl_id", *exprs)


def build_profiles(run: Any, w: Any, selection: Any, model: str, reference: str, bdata: Any, metric_cols: Sequence[str], intervals: Any,
                   interval_info: dict[str, Any], salt: str, ladder_models: Sequence[str], reliability: dict[str, float] | None = None) -> Any:
    """Write ``reports/athlete_profiles/<pseudonym>.md`` for every selected athlete (OOF predictions only); returns the index."""
    import polars as pl

    from .stats import pseudonym

    pct = position_percentiles(bdata, metric_cols)
    rel = reliability or {}
    rows = []
    for s in selection.iter_rows(named=True):
        a = int(s["nfl_id"])
        label = pseudonym(a, salt)
        wr = w.filter(pl.col("nfl_id") == a).row(0, named=True)
        br = bdata.filter(pl.col("nfl_id") == a)
        b = br.row(0, named=True) if br.height else {}
        pr = pct.filter(pl.col("nfl_id") == a)
        p = pr.row(0, named=True) if pr.height else {}
        officials = {c: b.get(c) for c in M1_COLS}
        movement = {c.replace("c_forty__", "forty · ").replace("c_agility__", "agility · "): (b.get(c), p.get(f"pct_{c}")) for c in metric_cols}
        preds = {m: wr.get(f"pred_{m}") for m in ladder_models if f"pred_{m}" in wr}
        vals = np.array([v for v in preds.values() if v is not None and np.isfinite(v)], float)
        iv = None
        if interval_info.get("valid") and intervals is not None:
            ir = intervals.filter(pl.col("nfl_id") == a)
            iv = (float(ir["lo"][0]), float(ir["hi"][0])) if ir.height else None
        note = (f"{interval_info.get('level', 0.8):.0%} {interval_info.get('method')}; empirical coverage {interval_info.get('coverage', float('nan')):.2f}"
                if interval_info.get("valid") else f"not shown — empirical coverage {interval_info.get('coverage', float('nan')):.2f} misses nominal "
                f"{interval_info.get('level', 0.8):.0%} by more than 10 points")
        missing = [c for c in M1_COLS if b.get(c) is None]
        notes = [f"M1 measurements missing: {', '.join(missing)} (imputed inside the design with training medians)" if missing else "all M1 measurements observed",
                 f"movement metrics observed: {sum(1 for c in metric_cols if b.get(c) is not None)} of {len(metric_cols)}"]
        notes += [f"Combine reliability ICC(1,1) of {c}: {rel[c]:.2f}" for c in metric_cols if c in rel and np.isfinite(rel[c])][:6]
        md = profile_markdown(label, {**s, "y": wr.get("y")}, officials, movement, preds, iv, note, float(vals.std(ddof=1)) if vals.size > 1 else NAN, notes,
                              model, reference)
        path = run.write_text(md, f"reports/athlete_profiles/{label}.md")
        rows.append({"athlete": label, "selection": s.get("selection"), "position_group": wr.get("position_group"), "fold": wr.get("fold"), "y": wr.get("y"),
                     "pred_reference": wr.get(f"pred_{reference}"), "pred_model": wr.get(f"pred_{model}"), "abs_error_gain": s.get("abs_error_gain"),
                     "path": run.rel(path)})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["associations", "build_profiles", "position_percentiles", "athlete_context", "best_model", "forty_twins", "gain_by_group", "geometry_records", "oof_intervals", "probe_summary", "profile_markdown",
           "relationship_table", "retrieval_records", "select_profiles", "tracking_gain", "transfer_records", "unexplained_by_position"]
