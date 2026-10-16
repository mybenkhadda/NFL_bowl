"""Part I — interpretability of the transfer signal and research-oriented athlete profiles.

Interpretable model = the Milestone 12 ridge ladder (``m11 ev.design`` train-only design, inner-CV α from
``ae._alpha``, unpenalised position block) on M1 officials + Combine-side movement metrics. Out-of-fold only:

* permutation importance — a column's TEST-fold values are permuted among the test athletes (one row per athlete, so
  the permutation is athlete-grouped); the design statistics come from the unchanged training rows; Δ RMSE averaged
  over repeats and folds, overall and per position;
* partial dependence — test rows set to a grid of TRAINING-fold quantiles (q05…q95: physically observed values);
* one declared interaction test (metric × official forty) as a ladder comparison;
* counterfactual sensitivity — prediction change when one metric moves to its within-position q75 (observed range),
  others fixed.

Importance is predictive, not causal; correlated inputs share or trade importance.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..data.validation import M1_COLS, TRANSFER_TARGET

NAN = float("nan")


def _mods() -> tuple[Any, Any, Any]:
    from nflbdb.research.advanced_movement.m11 import evaluation as ev
    from nflbdb.research.advanced_movement.transfer_models import ridge_solve

    from .. import _backend

    return _backend.ae(), ev, ridge_solve


def choose_metrics(data: Any, candidates: Sequence[str], min_coverage: float = 0.5) -> list[str]:
    return [c for c in candidates if c in data.columns and float(data[c].is_not_null().mean()) >= min_coverage]


def _fit(d: Any, y: np.ndarray, cols: Sequence[str], train: np.ndarray, alpha: float) -> tuple[Any, float, np.ndarray]:
    ae, ev, ridge_solve = _mods()
    des = ev.design(d, list(cols), train)
    b0, beta = ridge_solve(des.X[train], y[train], alpha, des.pen)
    return des, float(b0), beta


def _predict(d: Any, cols: Sequence[str], train: np.ndarray, rows: np.ndarray, b0: float, beta: np.ndarray) -> np.ndarray:
    _ae, ev, _r = _mods()
    des = ev.design(d, list(cols), train)
    return b0 + des.X[rows] @ beta


def interpretable_ladder(data: Any, cols: Sequence[str], exp: Any, seed: int, n_repeats: int = 20, grid: int = 11) -> dict[str, Any]:
    """OOF predictions + permutation importance + partial dependence of ridge(M1 + ``cols``)."""
    import polars as pl

    ae, ev, _ = _mods()
    C = exp.cfg.science.to_m12config()
    feats = [*M1_COLS, *cols]
    d = data.filter(pl.col(TRANSFER_TARGET).cast(pl.Float64).is_finite().fill_null(False))
    y = d[TRANSFER_TARGET].cast(pl.Float64).to_numpy()
    folds = d["fold"].to_numpy()
    pos = d["position_group"].fill_null("?").to_numpy()
    pred = np.full(d.height, NAN)
    imp, pdp, coef = [], [], []
    rng = np.random.default_rng(seed)
    for f in sorted(set(folds.tolist())):
        test, train = folds == f, folds != f
        if test.sum() == 0 or train.sum() < 20:
            continue
        a = ae._alpha(d, y, feats, None, train, C, seed + 31 * int(f))
        des, b0, beta = _fit(d, y, feats, train, a)
        p = b0 + des.X[test] @ beta
        pred[test] = p
        base = float(np.sqrt(np.mean((y[test] - p) ** 2)))
        for name, b in zip(des.names, beta):
            coef.append({"fold": int(f), "term": name, "coefficient_std_units": float(b), "alpha": a})
        ti = np.flatnonzero(test)
        for c in feats:
            if c not in d.columns:
                continue
            col = d[c].cast(pl.Float64).to_numpy().astype(float)        # nulls → NaN; ev.design treats both as missing
            for _ in range(n_repeats):
                perm = col.copy()
                perm[ti] = col[rng.permutation(ti)]
                dp = d.with_columns(pl.Series(c, perm, dtype=pl.Float64))
                pp = _predict(dp, feats, train, test, b0, beta)
                err = (y[test] - pp) ** 2
                imp.append({"fold": int(f), "feature": c, "position_group": "ALL", "delta_rmse": float(np.sqrt(err.mean()) - base)})
                for g in np.unique(pos[test]):
                    mk = pos[test] == g
                    if mk.sum() >= 5:
                        b_g = float(np.sqrt(((y[test][mk] - p[mk]) ** 2).mean()))
                        imp.append({"fold": int(f), "feature": c, "position_group": str(g), "delta_rmse": float(np.sqrt(err[mk].mean()) - b_g)})
            tr_vals = d[c].cast(pl.Float64).to_numpy()[train]
            tr_vals = tr_vals[np.isfinite(tr_vals)]
            if tr_vals.size < 10:
                continue
            for q in np.linspace(0.05, 0.95, grid):
                v = float(np.quantile(tr_vals, q))
                vals = d[c].cast(pl.Float64).to_numpy().copy()
                vals[ti] = v
                dv = d.with_columns(pl.Series(c, vals))
                pp = _predict(dv, feats, train, test, b0, beta)
                pdp.append({"fold": int(f), "feature": c, "quantile": float(q), "value": v, "mean_prediction": float(pp.mean())})
    imp_df = pl.DataFrame(imp, infer_schema_length=None)
    imp_sum = (imp_df.group_by("feature", "position_group").agg(pl.col("delta_rmse").mean().alias("importance_delta_rmse"),
                                                                pl.col("delta_rmse").std().alias("sd_over_repeats_folds"), pl.len().alias("n"))
               .sort("position_group", "importance_delta_rmse", descending=[False, True]) if imp_df.height else imp_df)
    pdp_df = pl.DataFrame(pdp, infer_schema_length=None)
    pdp_sum = (pdp_df.group_by("feature", "quantile").agg(pl.col("value").mean().alias("value"), pl.col("mean_prediction").mean().alias("mean_prediction"),
                                                          pl.col("mean_prediction").min().alias("min_fold"), pl.col("mean_prediction").max().alias("max_fold"))
               .sort("feature", "quantile") if pdp_df.height else pdp_df)
    oof = d.select("nfl_id", "position_group", "fold").with_columns(pl.Series("y", y), pl.Series("pred_interpretable", pred))
    rmse = float(np.sqrt(np.nanmean((y - pred) ** 2)))
    return {"oof": oof, "importance": imp_sum, "importance_raw": imp_df, "pdp": pdp_sum, "coefficients": pl.DataFrame(coef, infer_schema_length=None),
            "features": feats, "rmse": rmse}


def interaction_test(data: Any, metric: str, exp: Any, seed: int) -> Any:
    """Ladder M1 + metric vs M1 + metric + metric × forty (both z-scored on all rows: a fixed, label-free transform)."""
    import polars as pl

    ae, _ev, _ = _mods()
    C = exp.cfg.science.to_m12config()
    if metric not in data.columns or "forty" not in data.columns:
        return pl.DataFrame()
    z = lambda c: (pl.col(c) - pl.col(c).mean()) / pl.col(c).std()   # noqa: E731
    d = data.with_columns((z(metric) * z("forty")).alias(f"{metric}__x__forty"))
    mods = {"add": ae.Model([*M1_COLS, metric]), "int": ae.Model([*M1_COLS, metric, f"{metric}__x__forty"])}
    res = ae.oof_ladder(d, TRANSFER_TARGET, mods, C, seed)
    return ae.compare(res, [("int", "add")], C, seed, metric=metric, interaction="x forty")


def counterfactuals(data: Any, interp: dict[str, Any], athletes: Sequence[int], exp: Any, seed: int, metrics: Sequence[str]) -> Any:
    """For each athlete and metric: prediction change when the metric moves to its within-position q75 (refits the
    athlete's OWN held-out fold model; the athlete never trains it)."""
    import polars as pl

    ae, _ev, _ = _mods()
    C = exp.cfg.science.to_m12config()
    feats = interp["features"]
    d = data.filter(pl.col(TRANSFER_TARGET).cast(pl.Float64).is_finite().fill_null(False))
    y = d[TRANSFER_TARGET].cast(pl.Float64).to_numpy()
    folds = d["fold"].to_numpy()
    ids = d["nfl_id"].to_numpy()
    pos = d["position_group"].fill_null("?").to_numpy()
    out = []
    for a in athletes:
        k = np.flatnonzero(ids == a)
        if not k.size:
            continue
        i = int(k[0])
        f = folds[i]
        train = folds != f
        al = ae._alpha(d, y, feats, None, train, C, seed + 31 * int(f))
        des, b0, beta = _fit(d, y, feats, train, al)
        base = float(b0 + des.X[i] @ beta)
        for m in metrics:
            if m not in d.columns:
                continue
            v0 = d[m][i]
            same = d[m].cast(pl.Float64).to_numpy()[(pos == pos[i]) & train]
            same = same[np.isfinite(same)]
            if v0 is None or same.size < 10:
                continue
            target = float(np.quantile(same, 0.75))
            vals = d[m].cast(pl.Float64).to_numpy().copy()
            vals[i] = target
            pp = _predict(d.with_columns(pl.Series(m, vals)), feats, train, np.array([i]), b0, beta)[0]
            out.append({"nfl_id": int(a), "metric": m, "observed": float(v0), "position_q75": target, "prediction": base, "prediction_at_q75": float(pp),
                        "delta_prediction": float(pp - base)})
    return pl.DataFrame(out, infer_schema_length=None)


def select_profiles(oof: Any, n: int, seed: int) -> Any:
    """Explicit selection: highest predicted capacity, largest model disagreement, largest absolute residual, and a
    seeded random draw (so typical athletes appear too) — n / 4 each, unique athletes."""
    import polars as pl

    pc = [c for c in oof.columns if c.startswith("pred_")]
    d = oof.with_columns(pl.concat_list([pl.col(c) for c in pc]).list.eval(pl.element().std()).list.first().alias("disagreement_sd"))
    main = "pred_M1+F4_physics_contrastive" if "pred_M1+F4_physics_contrastive" in d.columns else pc[-1]
    d = d.with_columns((pl.col("y") - pl.col(main)).abs().alias("abs_residual"))
    q = max(1, n // 4)
    picks = [d.sort(main, descending=True).head(q).with_columns(pl.lit("highest predicted capacity").alias("selection")),
             d.sort("disagreement_sd", descending=True).head(q).with_columns(pl.lit("largest model disagreement").alias("selection")),
             d.sort("abs_residual", descending=True).head(q).with_columns(pl.lit("largest error").alias("selection")),
             d.sample(min(q, d.height), seed=seed).with_columns(pl.lit("random (seeded)").alias("selection"))]
    return pl.concat(picks, how="vertical").unique(subset=["nfl_id"], keep="first", maintain_order=True)


def athlete_profile(a: int, label: str, selection: str, ath_row: dict[str, Any], preds: dict[str, float], y: float | None, interval: tuple[float, float],
                    contributions: Any, neighbours: Any, percentiles: dict[str, float], coverage: dict[str, Any]) -> str:
    """Markdown profile of one athlete (research view — not a personnel recommendation)."""
    f = (lambda v, d=2: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{d}f}" if isinstance(v, float) else str(v)))  # noqa: E731
    md = [f"# Athlete profile {label}", "", f"_Selection rule: **{selection}**. Exploratory research profile: NOT a validated personnel recommendation._", "",
          "## Official Combine measurements", "", "| measure | value |", "|---|---|"]
    md += [f"| {c} | {f(ath_row.get(c))} |" for c in M1_COLS]
    md += ["", "## Tracking-derived movement metrics (position percentile)", "", "| metric | value | position percentile |", "|---|---|---|"]
    md += [f"| {k} | {f(ath_row.get(k), 3)} | {f(v, 0)} |" for k, v in percentiles.items()]
    md += ["", "## Predicted NFL movement capacity (out-of-fold)", "", "| model | prediction |", "|---|---|"]
    md += [f"| {k} | {f(v, 3)} |" for k, v in preds.items()]
    md += ["", f"Observed frozen target (shown for research review only): **{f(y, 3)}**", "",
           f"Approximate 80 % interval for the main model: [{f(interval[0], 3)}, {f(interval[1], 3)}] — empirical OOF residual quantiles of the same "
           "position (other athletes); not a calibrated predictive interval.", "", "## Strongest signals (interpretable model, own held-out fold)", ""]
    md += [("| metric | observed | position q75 | Δ prediction if at q75 |\n|---|---|---|---|\n" + "\n".join(
        f"| {r['metric']} | {f(r['observed'], 3)} | {f(r['position_q75'], 3)} | {f(r['delta_prediction'], 3)} |" for r in contributions.iter_rows(named=True)))
            if contributions is not None and contributions.height else "_not available_"]
    md += ["", "## Comparable athletic profiles (same position, nearest in standardized officials + metrics)", ""]
    md += [("| athlete | distance | observed target |\n|---|---|---|\n" + "\n".join(
        f"| {r['label']} | {f(r['distance'], 2)} | {f(r['y'], 3)} |" for r in neighbours.iter_rows(named=True))) if neighbours is not None and neighbours.height else "_none_"]
    md += ["", "## Data coverage", "", *[f"- {k}: {v}" for k, v in coverage.items()], "", "## Reliability limitations", "",
           "- Predictions are out-of-fold ridge predictions; position-level samples are small.",
           "- Movement metrics are medians over few attempts; see the repeatability table before reading a single value.",
           "- Associations are not causal; no coaching-impact claim is made.", ""]
    return "\n".join(md)


def neighbours(data: Any, a: int, cols: Sequence[str], k: int = 5) -> Any:
    import polars as pl

    d = data.filter(pl.col(TRANSFER_TARGET).is_not_null())
    me = d.filter(pl.col("nfl_id") == a)
    if not me.height:
        return pl.DataFrame()
    p = me["position_group"][0]
    g = d.filter((pl.col("position_group") == p) & (pl.col("nfl_id") != a))
    use = [c for c in cols if c in d.columns]
    Z = g.select(use).to_numpy().astype(float)
    mu, sd = np.nanmean(Z, 0), np.nanstd(Z, 0)
    sd = np.where(sd > 0, sd, 1.0)
    x = (me.select(use).to_numpy().astype(float)[0] - mu) / sd
    Zs = (Z - mu) / sd
    ok = np.isfinite(x)
    dist = np.sqrt(np.nanmean((Zs[:, ok] - x[ok]) ** 2, 1)) if ok.any() else np.full(g.height, np.nan)
    return g.select("nfl_id", pl.col(TRANSFER_TARGET).alias("y")).with_columns(pl.Series("distance", dist)).filter(pl.col("distance").is_finite()).sort("distance").head(k)


__all__ = ["athlete_profile", "choose_metrics", "counterfactuals", "interaction_test", "interpretable_ladder", "neighbours", "select_profiles"]
