"""Stages MOD and SENS — the paired model comparison and its sensitivity analyses.

One analysis = one :class:`AnalysisSpec`: a target, an observation window, an eligibility threshold, a Combine
feature variant, a set of models and a validation scheme. Within an analysis every model is fitted and scored on
IDENTICAL athletes and IDENTICAL folds (complete cases on the official times and on every tracking feature that
any of its models uses; body measurements and jumps are imputed inside the training fold).

Reading the outputs
    difference   model − M1. RMSE / MAE: negative = the model predicts better. R²_oos: positive = better.
    interval     athlete bootstrap of the out-of-fold predictions (2.5 – 97.5 %).
    exploratory  M2bc and M1j never support a headline; they exist to show whether a conclusion is fragile.
    scored_on:*  the PRIMARY fit scored on a subset of athletes (no refit) — a description of where the error sits.
A sensitivity can weaken the primary result. It can never replace it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from ..contracts import TableContract
from ..sprint_rescue.stats import partial_corr, seed_for
from . import models as md
from .cohort import COMPLETE_CASE, OFFICIAL_COLUMNS
from .config import (COHORT, REFERENCE_MODEL, TARGETS, TRACKING, AnalysisSpec, GetOffConfig, ModelSpec, model_specs,
                     primary_spec, sensitivity_specs)
from .targets import build_target

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
FOLDS = TableContract(
    "fold_assignments", "Athlete → fold of the primary analysis (same folds for every model).",
    {"repeat": I, "nfl_id": I, "fold": I}, key=("repeat", "nfl_id"), kind="dataset", area="CV repeat × athlete")
OOF = TableContract(
    "oof_predictions", "Out-of-fold predictions of the primary analysis (one row per model × repeat × athlete).",
    {"model": S, "repeat": I, "nfl_id": I, "fold": I, "y_obs": F, "y_ref": F, "y_pred": F, "ridge_lambda": F},
    key=("model", "repeat", "nfl_id"), kind="dataset", area="model × CV repeat × athlete",
    docs={"y_obs": "target the athlete was scored against [s]; context means from the training athletes of the fold",
          "y_ref": "mean target of the training athletes of the fold (the MEAN model)",
          "ridge_lambda": "penalty chosen inside the training fold that predicted this athlete"})
COMPARISON = TableContract(
    "model_comparison", "Out-of-fold performance of M0 / M1 / M2 on identical athletes (primary analysis).",
    {"model": S, "label": S, "exploratory": B, "n_predictors": I, "metric": S, "estimate": F, "ci_lower": F,
     "ci_upper": F, "n_athletes": I}, key=("model", "metric"), area="model × metric",
    docs={"estimate": "mean over CV repeats of the out-of-fold metric", "ci_lower": "athlete bootstrap, 2.5 %",
          "metric": "mae, rmse [s] · r2_oos (skill vs training mean) · pearson, spearman · calibration slope, intercept"})
BOOT = TableContract(
    "bootstrap_comparisons", "Paired differences against M1 with athlete-bootstrap intervals (primary analysis).",
    {"model": S, "reference": S, "exploratory": B, "metric": S, "estimate": F, "ci_lower": F, "ci_upper": F,
     "share_boot_improved": F, "n_athletes": I}, key=("model", "reference", "metric"), area="model × metric",
    docs={"estimate": "model − reference; negative is better for mae / rmse, positive for r2_oos / pearson",
          "share_boot_improved": "share of bootstrap replicates in which the model beats the reference"})
COEF = TableContract(
    "coefficients", "Standardised ridge coefficients on all athletes (in-sample, descriptive).",
    {"model": S, "term": S, "estimate_std": F, "ci_lower": F, "ci_upper": F, "ridge_lambda": F, "n_athletes": I},
    key=("model", "term"), area="model × term",
    docs={"estimate_std": "seconds of get-off per 1 SD of the predictor (dummy: per level), shrunk by the penalty",
          "ci_lower": "athlete bootstrap at the fitted λ — in-sample, not a held-out result"})
SENSITIVITY = TableContract(
    "sensitivity_analyses", "Model comparison repeated under each sensitivity (one row per analysis × model).",
    {"analysis": S, "description": S, "model": S, "n_athletes": I, "rmse": F, "mae": F, "r2_oos": F, "d_rmse_vs_m1": F,
     "d_rmse_lower": F, "d_rmse_upper": F, "d_r2_vs_m1": F, "d_r2_lower": F, "d_r2_upper": F,
     "share_boot_improved_rmse": F, "note": S}, key=("analysis", "model"), area="analysis × model")
MATCHED = TableContract(
    "matched_athletes", "Pairs of athletes with near-identical official times and different tracked acceleration.",
    {"pair_id": I, "feature": S, "alignment_group": S, "nfl_id_a": I, "name_a": S, "nfl_id_b": I, "name_b": S,
     "forty_a": F, "forty_b": F, "ten_a": F, "ten_b": F, "feature_a": F, "feature_b": F, "d_feature": F,
     "target_a": F, "target_b": F, "d_target": F, "n_obs_a": I, "n_obs_b": I}, key=("pair_id",), area="athlete pair",
    docs={"d_feature": "a − b, always ≥ 0 (a = higher feature value)", "d_target": "a − b in the NFL target [s]",
          "target_a": "descriptive context-adjusted target (all-athlete context means) — in-sample illustration"})


# ============================================================================== analysis frame
@dataclass
class Frame:
    ids: np.ndarray
    names: np.ndarray
    data: dict[str, np.ndarray]
    target: md.Target
    alignment: np.ndarray
    n_obs: np.ndarray

    @property
    def n(self) -> int:
        return int(self.ids.size)


def analysis_frame(base: pl.DataFrame, variants: pl.DataFrame, targets: pl.DataFrame, cells: pl.DataFrame,
                   spec: AnalysisSpec, specs: dict[str, ModelSpec]) -> Frame:
    """Athletes of one analysis: eligible in the window, complete on what the models need, with a finite target."""
    need = sorted({c for m in spec.models for c in specs[m].numeric if c in COMPLETE_CASE})
    trk = variants.filter(pl.col("variant") == spec.combine_variant).select("nfl_id", *TRACKING)
    tg = targets.filter(pl.col("window") == spec.window).select("nfl_id", "n_obs", "alignment_group")
    df = (base.select("nfl_id", "display_name", COHORT, *OFFICIAL_COLUMNS)
          .join(trk, on="nfl_id", how="left").join(tg, on="nfl_id", how="inner")
          .filter((pl.col("n_obs") >= spec.min_obs) & pl.col(COHORT).is_not_null()))
    if need:
        df = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in need]))
    if spec.alignment:
        df = df.filter(pl.col("alignment_group") == spec.alignment)
    if spec.cohorts:
        df = df.filter(pl.col(COHORT).is_in(list(spec.cohorts)))
    df = df.sort("nfl_id")
    target = build_target(targets, cells, df["nfl_id"].to_list(), spec.target, spec.window)
    ok = np.isfinite(target.y(None)) if df.height else np.zeros(0, bool)
    df, target = df.filter(pl.Series(ok)), target.take(np.flatnonzero(ok))
    data = {c: df[c].cast(pl.Float64).fill_null(float("nan")).to_numpy() for c in (*OFFICIAL_COLUMNS, *TRACKING)}
    data[COHORT] = df[COHORT].cast(pl.String).to_numpy()
    return Frame(df["nfl_id"].to_numpy(), df["display_name"].fill_null("").to_numpy(), data, target,
                 df["alignment_group"].fill_null("?").to_numpy(), df["n_obs"].to_numpy())


# ============================================================================== one analysis
def model_designs(spec: AnalysisSpec, specs: dict[str, ModelSpec]) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """name → (numeric, categorical). Leave-one-cohort-out cannot use the cohort as a predictor."""
    return {m: (specs[m].numeric, () if spec.cv == "loco" else specs[m].categorical) for m in spec.models}


def run_analysis(frame: Frame, spec: AnalysisSpec, gcfg: GetOffConfig, n_boot: int) -> dict[str, Any] | None:
    """Cross-validated comparison of the spec's models; None when the analysis is not estimable."""
    cohort = frame.data[COHORT]
    if frame.n < gcfg.min_athletes_model or (spec.cv == "loco" and np.unique(cohort).size < 2):
        return None
    seed = seed_for(gcfg.seed, "cv", spec.name)
    folds = (md.group_folds(cohort) if spec.cv == "loco"
             else md.make_folds(cohort, gcfg.cv_folds, gcfg.cv_repeats, seed))
    cv = md.cross_validate(frame.data, frame.target, model_designs(spec, model_specs()), folds,
                           md.lambda_grid(gcfg.lambda_grid), spec.robust, gcfg.huber_k)
    rows, diffs = md.evaluate(cv, REFERENCE_MODEL, n_boot, seed_for(gcfg.seed, "boot", spec.name))
    return {"cv": cv, "folds": folds, "rows": rows, "diffs": diffs}


def _pick(rows: list[dict[str, Any]], model: str, metric: str, field: str = "estimate") -> float | None:
    v = next((r[field] for r in rows if r["model"] == model and r["metric"] == metric), None)
    return None if v is None or not np.isfinite(v) else float(v)


def summary_rows(name: str, description: str, n: int, rows: list[dict[str, Any]], diffs: list[dict[str, Any]],
                 note: str = "") -> list[dict[str, Any]]:
    out = []
    for m in dict.fromkeys(r["model"] for r in rows):
        out.append({"analysis": name, "description": description, "model": m, "n_athletes": n,
                    "rmse": _pick(rows, m, "rmse"), "mae": _pick(rows, m, "mae"), "r2_oos": _pick(rows, m, "r2_oos"),
                    "d_rmse_vs_m1": _pick(diffs, m, "rmse"), "d_rmse_lower": _pick(diffs, m, "rmse", "ci_lower"),
                    "d_rmse_upper": _pick(diffs, m, "rmse", "ci_upper"), "d_r2_vs_m1": _pick(diffs, m, "r2_oos"),
                    "d_r2_lower": _pick(diffs, m, "r2_oos", "ci_lower"), "d_r2_upper": _pick(diffs, m, "r2_oos", "ci_upper"),
                    "share_boot_improved_rmse": _pick(diffs, m, "rmse", "share_boot_improved"), "note": note})
    return out


def not_estimable(spec: AnalysisSpec, n: int, gcfg: GetOffConfig) -> dict[str, Any]:
    return {**dict.fromkeys(SENSITIVITY.columns), "analysis": spec.name, "description": spec.description, "model": "-",
            "n_athletes": n, "note": f"not estimable: {n} athletes (minimum {gcfg.min_athletes_model}) or a single cohort"}


# ============================================================================== stage MOD
def _clean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]


def n_predictors(spec: ModelSpec, frame: Frame) -> int:
    return len(spec.numeric) + sum(max(np.unique(frame.data[c]).size - 1, 0) for c in spec.categorical)


def matched_pairs(frame: Frame, gcfg: GetOffConfig) -> pl.DataFrame:
    """Greedy, non-overlapping pairs inside the calipers on both official times and inside one alignment group,
    taken in order of decreasing difference in the tracked feature. Deterministic."""
    f = frame.data[gcfg.match_feature]
    forty, ten, y = frame.data["cmb_forty"], frame.data["cmb_ten_yd_split"], frame.target.y(None)
    cand = []
    for i in range(frame.n):
        for j in range(i + 1, frame.n):
            if (frame.alignment[i] == frame.alignment[j] and abs(forty[i] - forty[j]) <= gcfg.match_forty_s
                    and abs(ten[i] - ten[j]) <= gcfg.match_ten_s and np.isfinite(f[i]) and np.isfinite(f[j])):
                cand.append((abs(f[i] - f[j]), i, j))
    used: set[int] = set()
    rows = []
    for _, i, j in sorted(cand, key=lambda c: (-c[0], c[1], c[2])):
        if i in used or j in used:
            continue
        used |= {i, j}
        a, b = (i, j) if f[i] >= f[j] else (j, i)
        rows.append({"pair_id": len(rows) + 1, "feature": gcfg.match_feature, "alignment_group": str(frame.alignment[a]),
                     "nfl_id_a": int(frame.ids[a]), "name_a": str(frame.names[a]), "nfl_id_b": int(frame.ids[b]),
                     "name_b": str(frame.names[b]), "forty_a": float(forty[a]), "forty_b": float(forty[b]),
                     "ten_a": float(ten[a]), "ten_b": float(ten[b]), "feature_a": float(f[a]), "feature_b": float(f[b]),
                     "d_feature": float(f[a] - f[b]), "target_a": float(y[a]), "target_b": float(y[b]),
                     "d_target": float(y[a] - y[b]), "n_obs_a": int(frame.n_obs[a]), "n_obs_b": int(frame.n_obs[b])})
    return pl.DataFrame(rows, schema=MATCHED.columns) if rows else MATCHED.empty()


def matched_summary(pairs: pl.DataFrame) -> dict[str, float | int | None]:
    """Do the athletes with the higher tracked feature differ in the NFL target? (pairs are the unit)"""
    if pairs.height < 5:
        return {"n_pairs": pairs.height, "spearman": None, "ci_lower": None, "ci_upper": None, "mean_d_target": None,
                "share_d_target_negative": None}
    d = pairs["d_target"].to_numpy()
    r = partial_corr(d, pairs["d_feature"].to_numpy())
    return {"n_pairs": pairs.height, "spearman": r["r"], "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
            "mean_d_target": float(d.mean()), "share_d_target_negative": float((d < 0).mean())}


def compute_models(base: pl.DataFrame, variants: pl.DataFrame, targets: pl.DataFrame, cells: pl.DataFrame,
                   gcfg: GetOffConfig) -> dict[str, pl.DataFrame]:
    specs, spec = model_specs(), primary_spec(gcfg)
    frame = analysis_frame(base, variants, targets, cells, spec, specs)
    res = run_analysis(frame, spec, gcfg, gcfg.n_boot)
    if res is None:
        raise RuntimeError(f"primary analysis not estimable: {frame.n} athletes in the modelling population "
                           f"(minimum {gcfg.min_athletes_model}). See cohort_funnel.csv.")
    cv, folds = res["cv"], res["folds"]
    R = folds.shape[0]
    rep = np.repeat(np.arange(R, dtype=np.int64), frame.n)
    ids, fold = np.tile(frame.ids.astype(np.int64), R), folds.ravel().astype(np.int64)
    fold_df = pl.DataFrame({"repeat": rep, "nfl_id": ids, "fold": fold}, schema=FOLDS.columns)
    oof = pl.concat([pl.DataFrame({"model": [m] * rep.size, "repeat": rep, "nfl_id": ids, "fold": fold,
                                   "y_obs": cv["y"].ravel(), "y_ref": cv["y_ref"].ravel(), "y_pred": cv["pred"][m].ravel(),
                                   "ridge_lambda": cv["lam"][m].ravel()}, schema=OOF.columns) for m in spec.models])
    oof = oof.with_columns(pl.col("ridge_lambda").fill_nan(None), pl.col("y_pred").fill_nan(None))
    meta = {m: {"label": specs[m].label, "exploratory": specs[m].exploratory, "n_predictors": n_predictors(specs[m], frame)}
            for m in spec.models}
    comparison = pl.DataFrame(_clean([{**r, **meta[r["model"]]} for r in res["rows"]]), schema_overrides=COMPARISON.columns
                              ).select([pl.col(c).cast(t) for c, t in COMPARISON.columns.items()])
    boot = pl.DataFrame(_clean([{**r, "exploratory": meta[r["model"]]["exploratory"]} for r in res["diffs"]]),
                        schema_overrides=BOOT.columns).select([pl.col(c).cast(t) for c, t in BOOT.columns.items()])
    y_all, lambdas = frame.target.y(None), md.lambda_grid(gcfg.lambda_grid)
    coef = []
    for m in spec.models:
        if not specs[m].columns:
            continue
        for r in md.coefficient_table(frame.data, specs[m].numeric, specs[m].categorical, y_all, lambdas,
                                      gcfg.n_boot_sensitivity, seed_for(gcfg.seed, "coef", m)):
            coef.append({"model": m, "term": r["term"], "estimate_std": r["estimate_std"], "ci_lower": r["ci_lower"],
                         "ci_upper": r["ci_upper"], "ridge_lambda": r["lambda"], "n_athletes": r["n_athletes"]})
    coef_df = pl.DataFrame(_clean(coef), schema=COEF.columns) if coef else COEF.empty()
    return {FOLDS.name: fold_df, OOF.name: oof, COMPARISON.name: comparison, BOOT.name: boot, COEF.name: coef_df,
            MATCHED.name: matched_pairs(frame, gcfg)}


# ============================================================================== stage SENS
def compute_sensitivity(base: pl.DataFrame, variants: pl.DataFrame, targets: pl.DataFrame, cells: pl.DataFrame,
                        gcfg: GetOffConfig) -> pl.DataFrame:
    specs = model_specs()
    rows: list[dict[str, Any]] = []
    cohorts = tuple(int(c) for c in sorted(base[COHORT].drop_nulls().unique().to_list()))
    # ---- the primary fit scored on strata (no refit)
    prim = primary_spec(gcfg)
    frame = analysis_frame(base, variants, targets, cells, prim, specs)
    res = run_analysis(frame, prim, gcfg, gcfg.n_boot_sensitivity)
    if res is not None:
        rows += summary_rows("primary", prim.description, frame.n, res["rows"], res["diffs"])
        strata = [(f"scored_on:alignment={a}", frame.alignment == a) for a in gcfg.alignments]
        strata += [(f"scored_on:cohort={c}", frame.data[COHORT] == str(c)) for c in cohorts]
        for name, mask in strata:
            r, d = md.evaluate(res["cv"], REFERENCE_MODEL, gcfg.n_boot_sensitivity, seed_for(gcfg.seed, "boot", name), mask)
            if r:
                rows += summary_rows(name, "primary fit scored on this subset of athletes (no refit)", int(mask.sum()), r, d,
                                     "R² is skill against the overall training mean, inside the subset")
    # ---- refitted sensitivities
    for spec in sensitivity_specs(gcfg, cohorts):
        frame = analysis_frame(base, variants, targets, cells, spec, specs)
        res = run_analysis(frame, spec, gcfg, gcfg.n_boot_sensitivity)
        rows += [not_estimable(spec, frame.n, gcfg)] if res is None else summary_rows(
            spec.name, spec.description, frame.n, res["rows"], res["diffs"])
    return pl.DataFrame(_clean(rows), schema=SENSITIVITY.columns) if rows else SENSITIVITY.empty()


__all__ = ["FOLDS", "OOF", "COMPARISON", "BOOT", "COEF", "SENSITIVITY", "MATCHED", "Frame", "analysis_frame",
           "run_analysis", "compute_models", "compute_sensitivity", "matched_pairs", "matched_summary", "TARGETS"]
