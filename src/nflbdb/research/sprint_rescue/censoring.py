"""Study A (A2–A4) — does the recording window explain poor acceleration repeatability?

Hypothesis   Attempts of one athlete are recorded over different windows (the recording starts later or earlier
             in the run). If acceleration summaries depend on the window, window differences between an athlete's
             attempts add within-athlete variance and lower the ICC.
Unit         the athlete. Every statistic uses one row per athlete (attempt 2 − attempt 1 differences, or athlete
             means); frames are never treated as independent.
Analyses     A2_paired_association   Spearman of Δacceleration against Δwindow between an athlete's two attempts,
                                     raw and adjusted for position, cohort and positional noise.
             A2_window_r2            share of the variance of Δacceleration explained jointly by the window
                                     differences (partial R² beyond position and cohort).
             A3_abs_discrepancy      |Δacceleration| against |Δwindow|: do larger window differences go with
                                     larger disagreement between attempts?
             A3_fit_quality          sprint-model residual, identifiability width and vmax extrapolation against
                                     the recording window (athlete means).
             A3_icc_by_pattern       ICC among athletes grouped by censoring pattern, by boundary-peak pattern
                                     and by tercile of |Δ first-frame speed|.
             A3_window_adjusted_icc  ICC after removing the within-athlete window slope, with the paired gain.
                                     The slope is fitted on the same data, so the gain is an UPPER bound of what
                                     a window correction could recover.
Reading      These are associations. A positive result says window variation is compatible with the low ICC; it
             does not prove the window causes it. ``p95_speed`` is carried as a negative control.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from .attempts import BANDS
from .config import RescueConfig
from .reliability import ADJUSTED, icc_stats, repeat_frame
from .stats import bootstrap_ci, paired_difference, partial_corr, r_squared, seed_for, within_athlete_slopes

S, I, F = pl.String(), pl.Int64(), pl.Float64()
ACCEL = ("p95_a_tan", "peak_a_tan", "accel_sustained", "accel_initial", *BANDS, "model_a0", "model_tau_s",
         "model_a_at_ref")
CONTROL = ("p95_speed",)
WINDOW = ("first_speed", "duration_s", "path_smooth_yds", "fit_start_rel_s")
JOINT = ("first_speed", "duration_s", "path_smooth_yds")
FIT_QUALITY = ("model_rmse", "tau_rel_width", "vmax_extrapolation_ratio")
FIT_PREDICTORS = ("first_speed", "duration_s", "frames_after_peak")
COVARIATE_SETS = {"start": ("first_speed",), "start+end": ("first_speed", "duration_s", "frames_after_peak")}
ADJ = "position+cohort+noise"

ASSOC = TableContract(
    "censoring_association", "Recording-window variation against acceleration repeatability (athlete-level).",
    {"analysis": S, "outcome": S, "predictor": S, "adjustment": S, "group": S, "statistic": S, "n": I, "estimate": F,
     "ci_lower": F, "ci_upper": F, "note": S},
    key=("analysis", "outcome", "predictor", "adjustment", "group", "statistic"), area="analysis × outcome × predictor",
    docs={"n": "athletes", "ci_lower": "Fisher z (correlations), athlete bootstrap (R², ICC), paired bootstrap (gain)",
          "adjustment": "none, or covariates removed from both sides before correlating"})


def _row(analysis: str, outcome: str, predictor: str, adjustment: str, group: str, statistic: str, n: int,
         est: float | None, lo: float | None = None, hi: float | None = None, note: str = "") -> dict[str, Any]:
    fin = lambda v: float(v) if v is not None and np.isfinite(v) else None  # noqa: E731
    return {"analysis": analysis, "outcome": outcome, "predictor": predictor, "adjustment": adjustment, "group": group,
            "statistic": statistic, "n": int(n), "estimate": fin(est), "ci_lower": fin(lo), "ci_upper": fin(hi),
            "note": note}


def _np(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].cast(pl.Float64).fill_null(float("nan")).to_numpy()


def paired_association(within: pl.DataFrame, rcfg: RescueConfig) -> list[dict[str, Any]]:
    rows = []
    pos, coh, noise = within["position_group"].to_numpy(), within["draft_year"].cast(pl.String).fill_null("?").to_numpy(), \
        _np(within, "mean_resid_rms_yds")
    for out in (*ACCEL, *CONTROL):
        dy = _np(within, f"d_{out}")
        for pred in WINDOW:
            dx = _np(within, f"d_{pred}")
            for adj, kw in (("none", {}), (ADJ, {"categorical": [pos, coh], "continuous": [noise]})):
                r = partial_corr(dy, dx, method="spearman", **kw)
                rows.append(_row("A2_paired_association", f"d_{out}", f"d_{pred}", adj, "all", "spearman", r["n"],
                                 r["r"], r["ci_lower"], r["ci_upper"]))
                a = partial_corr(np.abs(dy), np.abs(dx), method="spearman", **kw)
                rows.append(_row("A3_abs_discrepancy", f"abs_d_{out}", f"abs_d_{pred}", adj, "all", "spearman",
                                 a["n"], a["r"], a["ci_lower"], a["ci_upper"]))
        X = [_np(within, f"d_{p}") for p in JOINT]
        full = r_squared(dy, X, categorical=[pos, coh])
        ok = np.isfinite(dy) & np.all([np.isfinite(c) for c in X], axis=0)
        yk, Xk, pk, ck = dy[ok], [c[ok] for c in X], pos[ok], coh[ok]
        lo, hi = bootstrap_ci(lambda i: r_squared(yk[i], [c[i] for c in Xk], categorical=[pk[i], ck[i]])["r2"],
                              int(ok.sum()), rcfg.n_boot, seed_for(rcfg.seed, "A2r2", out))
        rows.append(_row("A2_window_r2", f"d_{out}", "+".join(f"d_{p}" for p in JOINT), "position+cohort", "all",
                         "partial_r2", full["n"], full["r2"], lo, hi, note=f"adjusted R² = {full['r2_adj']:.4f}"
                         if np.isfinite(full["r2_adj"]) else ""))
    return rows


def fit_quality_association(att: pl.DataFrame) -> list[dict[str, Any]]:
    cols = [*FIT_QUALITY, *FIT_PREDICTORS, "resid_rms_yds"]
    a = att.group_by("nfl_id").agg(pl.col("position_group").first(), pl.col("draft_year").first(),
                                   *[pl.col(c).cast(pl.Float64).mean().alias(c) for c in cols]).sort("nfl_id")
    pos, coh = a["position_group"].to_numpy(), a["draft_year"].cast(pl.String).fill_null("?").to_numpy()
    rows = []
    for out in FIT_QUALITY:
        for pred in FIT_PREDICTORS:
            r = partial_corr(_np(a, out), _np(a, pred), categorical=[pos, coh], continuous=[_np(a, "resid_rms_yds")])
            rows.append(_row("A3_fit_quality", out, pred, ADJ, "all", "spearman", r["n"], r["r"], r["ci_lower"],
                             r["ci_upper"], note="athlete means over attempts"))
    return rows


def pattern_groups(within: pl.DataFrame) -> list[tuple[str, list[int]]]:
    """(group label, athlete ids): censoring pattern, boundary-peak pattern, tercile of |Δ first-frame speed|."""
    out = []
    for p in sorted(within["censor_pattern"].unique().to_list()):
        out.append((f"censor:{p}", within.filter(pl.col("censor_pattern") == p)["nfl_id"].to_list()))
    for flag, lab in ((False, "boundary:none"), (True, "boundary:any")):
        out.append((lab, within.filter(pl.col("any_boundary_peak") == flag)["nfl_id"].to_list()))
    w = within.filter(pl.col("d_first_speed").is_not_null())
    if w.height >= 3:
        x = np.abs(w["d_first_speed"].to_numpy())
        q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
        ids = np.array(w["nfl_id"].to_list())
        for lab, m in (("window_change:low", x <= q1), ("window_change:mid", (x > q1) & (x <= q2)),
                       ("window_change:high", x > q2)):
            out.append((lab, ids[m].tolist()))
    return out


def icc_by_pattern(att: pl.DataFrame, within: pl.DataFrame, rcfg: RescueConfig) -> list[dict[str, Any]]:
    rows = []
    for label, ids in pattern_groups(within):
        if not ids:
            continue
        sub = att.filter(pl.col("nfl_id").is_in(ids))
        for feat in (*ACCEL, *CONTROL):
            d = repeat_frame(sub, [feat], rcfg)
            for scope, by in (("none", ()), ("position", ADJUSTED["position_adjusted"])):
                r, _ = icc_stats(d, feat, by, rcfg, True, seed_for(rcfg.seed, "A3p", label, feat, scope))
                rows.append(_row("A3_icc_by_pattern", feat, "athlete_group", scope, label, "icc1", r["n_athletes"],
                                 r["icc1"], r["ci_lower"], r["ci_upper"], note=r["reliability"] or ""))
    return rows


def window_adjusted_icc(att: pl.DataFrame, rcfg: RescueConfig, covariate_sets: dict[str, Sequence[str]] | None = None
                        ) -> list[dict[str, Any]]:
    rows = []
    by = ADJUSTED["position_adjusted"]
    for feat in (*ACCEL, *CONTROL):
        for name, covs in (covariate_sets or COVARIATE_SETS).items():
            d = repeat_frame(att, [feat, *covs], rcfg)
            if d["nfl_id"].n_unique() < rcfg.min_athletes:
                continue
            y, ath = d[feat].to_numpy().astype(float), d["nfl_id"].to_numpy()
            X = np.column_stack([d[c].to_numpy().astype(float) for c in covs])
            beta = within_athlete_slopes(y, X, ath)
            adj = y - (X - X.mean(0)) @ beta
            seed = seed_for(rcfg.seed, "A3w", feat, name)
            r0, raw = icc_stats(d, feat, by, rcfg, True, seed)
            r1, fit = icc_stats(d, feat, by, rcfg, True, seed, values=adj)
            if r0["icc1"] is None or r1["icc1"] is None:
                continue
            gain, lo, hi = paired_difference(fit, raw)
            slopes = ", ".join(f"{c}: {b:.4g}" for c, b in zip(covs, beta))
            n = r0["n_athletes"]
            rows += [_row("A3_window_adjusted_icc", feat, name, "position", "all", "icc_raw", n, r0["icc1"],
                          r0["ci_lower"], r0["ci_upper"]),
                     _row("A3_window_adjusted_icc", feat, name, "position", "all", "icc_window_adjusted", n, r1["icc1"],
                          r1["ci_lower"], r1["ci_upper"], note=f"within-athlete slopes ({slopes}); in-sample"),
                     _row("A3_window_adjusted_icc", feat, name, "position", "all", "icc_gain", n, gain, lo, hi,
                          note="upper bound: slopes fitted on the same attempts")]
    return rows


def association_table(att: pl.DataFrame, within: pl.DataFrame, rcfg: RescueConfig) -> pl.DataFrame:
    rows = [*paired_association(within, rcfg), *fit_quality_association(att), *icc_by_pattern(att, within, rcfg),
            *window_adjusted_icc(att, rcfg)] if within.height >= rcfg.min_athletes else []
    return (pl.DataFrame(rows, schema=ASSOC.columns).sort(list(ASSOC.key)) if rows else ASSOC.empty())
