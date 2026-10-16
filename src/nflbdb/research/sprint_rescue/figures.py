"""Figures of the sprint-rescue experiment — drawn only from the persisted tables of this namespace.

Ten figures, one question each. Every caption states the unit of analysis; a panel without data says so.
Project style (``research.figkit``): categorical palette for groups, status colours for screening labels.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, STATUS
from ..figkit import ci_errorbar, empty_panel, figure, finish
from .attempts import ATTEMPTS, BANDS, EXAMPLES, WITHIN
from .censoring import ASSOC
from .config import RescueConfig
from .redundancy import SCREEN
from .reliability import MODEL_REPEAT, RELIABILITY
from .sprint_model import FITS, variant_sensitivity
from .store import RescueStore

Tables = dict[str, pl.DataFrame]
STATUS_COLORS = {"strong_candidate": STATUS["good"], "candidate": CATEGORICAL[0], "inconclusive": STATUS["warning"],
                 "redundant_with_stopwatch": STATUS["serious"], "not_repeatable": STATUS["critical"]}
SHOW = ("p95_a_tan", BANDS[0], "model_a0")


def _nice(name: str) -> str:
    return name.replace("_", " ")


def _vals(df: pl.DataFrame, col: str) -> np.ndarray:
    x = df[col].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    return x[np.isfinite(x)]


def fig_first_speed(store: RescueStore, t: Tables) -> None:
    att = t[ATTEMPTS.name]
    fig, ax = figure(1, 2)
    if not att.height:
        empty_panel(ax[0, 0], "no attempts")
        empty_panel(ax[0, 1], "no attempts")
    else:
        hi = float(np.nanquantile(_vals(att, "first_speed"), 0.99)) if _vals(att, "first_speed").size else 1.0
        bins = np.linspace(0, max(hi, 0.5), 40)
        for k, y in enumerate(sorted(att["draft_year"].drop_nulls().unique().to_list())):
            ax[0, 0].hist(_vals(att.filter(pl.col("draft_year") == y), "first_speed"), bins=bins, histtype="step",
                          linewidth=1.8, color=CATEGORICAL[k % 8], label=str(y))
        ax[0, 0].set_xlabel("speed on the first recorded frame (yd/s)")
        ax[0, 0].set_ylabel("attempts")
        ax[0, 0].legend(title="draft cohort", frameon=False)
        pos = sorted(att["position_group"].unique().to_list())
        ax[0, 1].boxplot([_vals(att.filter(pl.col("position_group") == p), "first_speed") for p in pos],
                         showfliers=False)
        ax[0, 1].set_xticks(range(1, len(pos) + 1), pos)
        ax[0, 1].set_ylabel("speed on the first recorded frame (yd/s)")
        ax[0, 1].set_xlabel("position group")
    finish(fig, "Unit: forty-yard-dash attempt. A value above zero means the recording began after the athlete had "
                "started to move. Histogram truncated at the 99th percentile.",
           "Where does the forty recording start?")
    store.save_figure(fig, "SR_01_first_frame_speed", {"attempts": att.select("draft_year", "position_group", "first_speed")},
                      source=ATTEMPTS.name, n=f"{att.height:,} attempts")


def fig_distance_duration(store: RescueStore, t: Tables) -> None:
    att = t[ATTEMPTS.name].filter(pl.col("duration_s").is_not_null() & pl.col("path_smooth_yds").is_not_null())
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    if not att.height:
        empty_panel(a, "no attempts")
    else:
        for k, y in enumerate(sorted(att["draft_year"].drop_nulls().unique().to_list())):
            g = att.filter(pl.col("draft_year") == y)
            a.scatter(g["duration_s"].to_numpy(), g["path_smooth_yds"].to_numpy(), s=14, alpha=0.6,
                      color=CATEGORICAL[k % 8], label=str(y))
        a.set_xlabel("recorded duration (s)")
        a.set_ylabel("recorded path length (yd)")
        a.legend(title="draft cohort", frameon=False)
    finish(fig, "Unit: attempt. A complete forty would sit at 40 yd for every athlete; spread along the vertical axis "
                "is recording-window variation, not athletic difference.", "How much of the run does each recording hold?")
    store.save_figure(fig, "SR_02_distance_vs_duration", {"attempts": att.select("draft_year", "duration_s", "path_smooth_yds")},
                      source=ATTEMPTS.name, n=f"{att.height:,} attempts")


def fig_delta(store: RescueStore, t: Tables) -> None:
    w, assoc = t[WITHIN.name], t[ASSOC.name]
    fig, ax = figure(1, len(SHOW))
    for j, feat in enumerate(SHOW):
        a = ax[0, j]
        g = w.filter(pl.col(f"d_{feat}").is_not_null() & pl.col("d_first_speed").is_not_null())
        if not g.height:
            empty_panel(a, f"no paired attempts for {feat}")
            continue
        a.scatter(g["d_first_speed"].to_numpy(), g[f"d_{feat}"].to_numpy(), s=14, alpha=0.6, color=CATEGORICAL[0])
        a.axhline(0, color=INK["axis"], linewidth=0.8)
        a.axvline(0, color=INK["axis"], linewidth=0.8)
        r = assoc.filter((pl.col("analysis") == "A2_paired_association") & (pl.col("outcome") == f"d_{feat}")
                         & (pl.col("predictor") == "d_first_speed") & (pl.col("adjustment") == "none"))
        if r.height and r["estimate"][0] is not None:
            lo, hi = r["ci_lower"][0], r["ci_upper"][0]
            ci = f" [{lo:.2f}, {hi:.2f}]" if lo is not None and hi is not None else ""
            a.set_title(f"{_nice(feat)}: ρ = {r['estimate'][0]:.2f}{ci}", fontsize=11)
        else:
            a.set_title(_nice(feat), fontsize=11)
        a.set_xlabel("Δ first-frame speed, attempt 2 − 1 (yd/s)")
        a.set_ylabel(f"Δ {_nice(feat)}")
    finish(fig, "Unit: athlete (one point = the difference between an athlete's two attempts). ρ = Spearman with a "
                "Fisher interval. An association, not proof that the window causes the change.",
           "When the recording starts later, does measured acceleration change?")
    store.save_figure(fig, "SR_03_delta_start_vs_delta_accel", {"athletes": w.select("nfl_id", "d_first_speed",
                                                                                      *[f"d_{f}" for f in SHOW])},
                      source=WITHIN.name, n=f"{w.height:,} athletes")


def fig_censoring_icc(store: RescueStore, t: Tables) -> None:
    assoc = t[ASSOC.name]
    fig, ax = figure(1, 2)
    pat = assoc.filter((pl.col("analysis") == "A3_icc_by_pattern") & (pl.col("adjustment") == "position")
                       & pl.col("outcome").is_in(list(SHOW)) & pl.col("estimate").is_not_null())
    a = ax[0, 0]
    groups = sorted(pat["group"].unique().to_list()) if pat.height else []
    if not groups:
        empty_panel(a, "no group with enough athletes")
    else:
        for k, feat in enumerate(SHOW):
            g = {r["group"]: r for r in pat.filter(pl.col("outcome") == feat).iter_rows(named=True)}
            xs = [i + (k - 1) * 0.22 for i, name in enumerate(groups) if name in g]
            rs = [g[name] for name in groups if name in g]
            if rs:
                ci_errorbar(a, xs, [r["estimate"] for r in rs], [r["ci_lower"] or np.nan for r in rs],
                            [r["ci_upper"] or np.nan for r in rs], color=CATEGORICAL[k], label=_nice(feat))
        a.set_xticks(range(len(groups)), [g.replace(":", "\n").replace("_", " ") for g in groups], fontsize=8)
        a.set_ylabel("position-adjusted ICC(1)")
        a.legend(frameon=False, fontsize=8)
    b = ax[0, 1]
    adj = assoc.filter((pl.col("analysis") == "A3_window_adjusted_icc") & (pl.col("predictor") == "start+end")
                       & pl.col("estimate").is_not_null())
    feats = sorted(adj["outcome"].unique().to_list()) if adj.height else []
    if not feats:
        empty_panel(b, "window-adjusted ICC not estimable")
    else:
        for k, (stat, lab) in enumerate((("icc_raw", "as measured"), ("icc_window_adjusted", "window-adjusted (upper bound)"))):
            g = {r["outcome"]: r for r in adj.filter(pl.col("statistic") == stat).iter_rows(named=True)}
            ys = [i + (k - 0.5) * 0.3 for i, f in enumerate(feats) if f in g]
            rs = [g[f] for f in feats if f in g]
            est = np.array([r["estimate"] for r in rs], float)
            lo = np.array([r["ci_lower"] if r["ci_lower"] is not None else np.nan for r in rs], float)
            hi = np.array([r["ci_upper"] if r["ci_upper"] is not None else np.nan for r in rs], float)
            b.errorbar(est, ys, xerr=np.vstack([np.where(np.isfinite(lo), est - lo, 0), np.where(np.isfinite(hi), hi - est, 0)]),
                       fmt="o", capsize=3, color=CATEGORICAL[k], label=lab)
        b.set_yticks(range(len(feats)), [_nice(f) for f in feats], fontsize=8)
        b.set_xlabel("position-adjusted ICC(1)")
        b.legend(frameon=False, fontsize=8)
    finish(fig, "Unit: athlete. Left: ICC among athletes grouped by how their two recordings differ. Right: ICC before "
                "and after removing the within-athlete slope on start speed, duration and frames after peak; the slope "
                "is fitted on the same attempts, so the adjusted value is an upper bound. Bars: 95 % athlete bootstrap.",
           "Is acceleration less repeatable when the recording windows differ?")
    store.save_figure(fig, "SR_04_censoring_vs_repeatability", {"patterns": pat, "window_adjusted": adj},
                      source=ASSOC.name, n=f"{pat.height + adj.height:,} estimates")


def fig_examples(store: RescueStore, t: Tables) -> None:
    ex = t[EXAMPLES.name]
    ids = ex.select("traj_id", "label").unique(maintain_order=True)
    fig, ax = figure(2, 2)
    for i in range(4):
        a = ax[i // 2, i % 2]
        if i >= ids.height:
            empty_panel(a, "no example attempt")
            continue
        tid, label = ids.row(i)
        g = ex.filter(pl.col("traj_id") == tid).sort("t_rel_s")
        a.plot(g["t_rel_s"].to_numpy(), g["speed"].to_numpy(), color=INK["primary"], linewidth=1.6, label="derived speed")
        m = g.filter(pl.col("model_speed").is_not_null())
        if m.height:
            a.plot(m["t_rel_s"].to_numpy(), m["model_speed"].to_numpy(), color=CATEGORICAL[1], linestyle="--",
                   linewidth=1.3, label="existing model")
        w = g.filter(pl.col("in_fit_window"))
        if w.height:
            a.axvspan(float(w["t_rel_s"].min()), float(w["t_rel_s"].max()), color=CATEGORICAL[0], alpha=0.08,
                      label="fit window")
        a.set_title(label, fontsize=11)
        a.set_xlabel("time since first recorded frame (s)")
        a.set_ylabel("speed (yd/s)")
        a.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, "Four attempts chosen on recording-window variables only (earliest and latest start, longest and "
                "shortest recording after peak speed). Illustrations, not evidence of prevalence.",
           "What do complete and truncated forty recordings look like?")
    store.save_figure(fig, "SR_05_example_recordings", {"signals": ex}, source=EXAMPLES.name, n=f"{ids.height} attempts")


def fig_icc_forest(store: RescueStore, t: Tables) -> None:
    rel = t[RELIABILITY.name].filter((pl.col("scenario") == "S0_all") & pl.col("icc1").is_not_null()
                                     & pl.col("scope").is_in(["pooled", "position_adjusted"]))
    fig, ax = figure(1, 1, size="tall")
    a = ax[0, 0]
    feats = rel.filter(pl.col("scope") == "pooled").sort("icc1")["feature"].to_list()
    if not feats:
        empty_panel(a, "no ICC estimable")
    else:
        for k, (scope, lab) in enumerate((("pooled", "pooled over positions"), ("position_adjusted", "within position"))):
            g = {r["feature"]: r for r in rel.filter(pl.col("scope") == scope).iter_rows(named=True)}
            rs = [(i, g[f]) for i, f in enumerate(feats) if f in g]
            est = np.array([r["icc1"] for _, r in rs], float)
            lo = np.array([r["ci_lower"] if r["ci_lower"] is not None else np.nan for _, r in rs], float)
            hi = np.array([r["ci_upper"] if r["ci_upper"] is not None else np.nan for _, r in rs], float)
            a.errorbar(est, [i + (k - 0.5) * 0.3 for i, _ in rs],
                       xerr=np.vstack([np.where(np.isfinite(lo), est - lo, 0), np.where(np.isfinite(hi), hi - est, 0)]),
                       fmt="o", capsize=3, color=CATEGORICAL[k], label=lab)
        a.axvline(RescueConfig().icc_reference, color=INK["muted"], linestyle=":", linewidth=1)
        a.set_yticks(range(len(feats)), [_nice(f) for f in feats])
        a.set_xlabel("ICC(1), single attempt")
        a.legend(frameon=False, loc="lower right")
    finish(fig, "Unit: athlete with ≥ 2 attempts, all structurally valid attempts. The gap between the two markers is "
                "repeatability that comes from differences between position groups. Dotted line: heuristic reference "
                "0.75, not an acceptance threshold. Bars: 95 % athlete bootstrap.",
           "Does repeatability survive inside a position group?")
    store.save_figure(fig, "SR_06_pooled_vs_within_position_icc", {"icc": rel}, source=RELIABILITY.name,
                      n=f"{len(feats)} features")


def fig_variance_slopes(store: RescueStore, t: Tables) -> None:
    rel, att = t[RELIABILITY.name].filter(pl.col("scenario") == "S0_all"), t[ATTEMPTS.name]
    fig, ax = figure(1, 3)
    a = ax[0, 0]
    pooled = {r["feature"]: r for r in rel.filter((pl.col("scope") == "pooled") & pl.col("var_between").is_not_null())
              .iter_rows(named=True)}
    adj = {r["feature"]: r for r in rel.filter((pl.col("scope") == "position_adjusted") & pl.col("var_between").is_not_null())
           .iter_rows(named=True)}
    feats = [f for f in pooled if f in adj]
    if not feats:
        empty_panel(a, "variance components not estimable")
    else:
        y = np.arange(len(feats))
        within = np.array([adj[f]["var_within"] for f in feats])
        between = np.array([adj[f]["var_between"] for f in feats])
        position = np.clip(np.array([pooled[f]["var_between"] for f in feats]) - between, 0, None)
        total = within + between + position
        left = np.zeros(len(feats))
        for part, lab, col in ((between, "between athletes, same position", CATEGORICAL[0]),
                               (position, "between positions", CATEGORICAL[3]), (within, "within athlete", INK["axis"])):
            a.barh(y, part / total, left=left, color=col, label=lab)
            left = left + part / total
        a.set_yticks(y, [_nice(f) for f in feats], fontsize=8)
        a.set_xlabel("share of variance")
        a.legend(frameon=False, fontsize=7, loc="lower right")
    for j, feat in enumerate(("p95_speed", "p95_a_tan"), start=1):
        b = ax[0, j]
        g = (att.filter((pl.col("n_attempts") >= 2) & (pl.col("attempt_order") <= 2) & pl.col(feat).is_not_null())
             .with_columns((pl.col(feat) - pl.col(feat).mean().over("position_group")).alias("_c")).sort("nfl_id", "attempt_order"))
        p = g.group_by("nfl_id", maintain_order=True).agg(pl.col("_c").first().alias("v1"), pl.col("_c").last().alias("v2"),
                                                          pl.len().alias("k")).filter(pl.col("k") == 2)
        if not p.height:
            empty_panel(b, f"no repeated attempts for {feat}")
            continue
        b.plot([1, 2], np.vstack([p["v1"].to_numpy(), p["v2"].to_numpy()]), color=CATEGORICAL[0], alpha=0.15, linewidth=0.8)
        b.set_xticks([1, 2], ["attempt 1", "attempt 2"])
        b.set_ylabel(f"{_nice(feat)}, minus position mean")
        b.set_title(_nice(feat), fontsize=11)
    finish(fig, "Left: variance of single-attempt values split into athlete, position and within-athlete parts. "
                "Right: each line is one athlete; flat, ordered lines mean attempts rank athletes the same way.",
           "What separates athletes: the athlete, the position, or the attempt?")
    store.save_figure(fig, "SR_07_variance_components_and_slopes", {"components": rel}, source=RELIABILITY.name,
                      n=f"{len(feats)} features")


def fig_cohort_sample(store: RescueStore, t: Tables) -> None:
    rel = t[RELIABILITY.name].filter(pl.col("icc1").is_not_null())
    coh = rel.filter((pl.col("scenario") == "S0_all") & pl.col("scope").str.starts_with("cohort:"))
    fig, ax = figure(1, 2)
    a = ax[0, 0]
    feats = sorted(coh["feature"].unique().to_list()) if coh.height else []
    if not feats:
        empty_panel(a, "no cohort with enough repeated athletes")
    else:
        years = sorted(coh["scope"].unique().to_list())
        for k, y in enumerate(years):
            g = {r["feature"]: r for r in coh.filter(pl.col("scope") == y).iter_rows(named=True)}
            rs = [(i, g[f]) for i, f in enumerate(feats) if f in g]
            ci_errorbar(a, [i + (k - (len(years) - 1) / 2) * 0.22 for i, _ in rs], [r["icc1"] for _, r in rs],
                        [r["ci_lower"] if r["ci_lower"] is not None else np.nan for _, r in rs],
                        [r["ci_upper"] if r["ci_upper"] is not None else np.nan for _, r in rs],
                        color=CATEGORICAL[k % 8], label=y.replace("cohort:", ""))
        a.set_xticks(range(len(feats)), [_nice(f) for f in feats], rotation=60, ha="right", fontsize=8)
        a.set_ylabel("ICC(1) within cohort")
        a.legend(title="draft cohort", frameon=False, fontsize=8)
    b = ax[0, 1]
    if not rel.height:
        empty_panel(b, "no ICC estimable")
    else:
        width = (rel["ci_upper"] - rel["ci_lower"]).to_numpy()
        b.scatter(rel["n_athletes"].to_numpy(), width, s=12, alpha=0.5, color=CATEGORICAL[0])
        b.set_xscale("log")
        b.set_xlabel("athletes with repeated attempts")
        b.set_ylabel("width of the 95 % ICC interval")
    finish(fig, "Left: all structurally valid attempts, each draft cohort separately (tracking processing differs by "
                "cohort). Right: every feature × scenario × scope estimate; small groups cannot support a precise ICC.",
           "Is repeatability stable across cohorts, and how much sample does it need?")
    store.save_figure(fig, "SR_08_reliability_by_cohort_and_sample", {"cohort": coh}, source=RELIABILITY.name,
                      n=f"{rel.height:,} estimates")


def fig_model(store: RescueStore, t: Tables) -> None:
    rep = t[MODEL_REPEAT.name].filter((pl.col("cohort") == "common") & (pl.col("scope") == "position_adjusted")
                                      & pl.col("variant").is_in(["direct", "persisted", "base", "common_start"])
                                      & pl.col("icc1").is_not_null())
    fig, ax = figure(1, 2)
    a = ax[0, 0]
    if not rep.height:
        empty_panel(a, "no common-cohort ICC estimable")
    else:
        rep = rep.sort("icc1")
        est, y = rep["icc1"].to_numpy(), np.arange(rep.height)
        lo = rep["ci_lower"].fill_null(float("nan")).to_numpy()
        hi = rep["ci_upper"].fill_null(float("nan")).to_numpy()
        cols = [CATEGORICAL[{"R1": 0, "R2": 2, "R3": 1}.get(r, 4)] for r in rep["representation"].to_list()]
        for i in range(rep.height):
            a.errorbar(est[i], y[i], xerr=[[est[i] - lo[i] if np.isfinite(lo[i]) else 0], [hi[i] - est[i] if np.isfinite(hi[i]) else 0]],
                       fmt="o", capsize=3, color=cols[i])
        a.set_yticks(y, [f"{_nice(f)} ({v})" for f, v in zip(rep["feature"].to_list(), rep["variant"].to_list())], fontsize=7)
        a.set_xlabel("position-adjusted ICC(1), identical attempts")
        for k, lab in ((0, "direct"), (2, "sustained"), (1, "model")):
            a.plot([], [], "o", color=CATEGORICAL[k], label=lab)
        a.legend(frameon=False, fontsize=8, loc="lower right")
    b = ax[0, 1]
    sens = variant_sensitivity(t[FITS.name])
    if not sens.height:
        empty_panel(b, "no refit available")
    else:
        x = np.arange(sens.height)
        for k, (col, lab) in enumerate((("median_abs_d_vmax_rel", "vmax"), ("median_abs_d_tau_rel", "τ"),
                                        ("median_abs_d_a0_rel", "a0"), ("median_abs_d_a_at_ref_rel", "a at reference speed"))):
            b.bar(x + (k - 1.5) * 0.2, sens[col].fill_null(0).to_numpy() * 100, width=0.2, color=CATEGORICAL[k], label=lab)
        b.set_xticks(x, [_nice(v) for v in sens["variant"].to_list()], rotation=30, ha="right")
        b.set_ylabel("median |change| vs existing fit (%)")
        b.legend(frameon=False, fontsize=8)
    finish(fig, "Left: every representation on the SAME attempts (common cohort), so a difference cannot come from a "
                "change of sample; bars: 95 % athlete bootstrap. Right: how far each model parameter moves when the fit "
                "window changes by a few frames.", "Is the sprint model more repeatable than direct acceleration?")
    store.save_figure(fig, "SR_09_model_repeatability_and_sensitivity", {"icc": rep, "sensitivity": sens},
                      source=f"{MODEL_REPEAT.name}, {FITS.name}", n=f"{rep.height} estimates")


def fig_screening(store: RescueStore, t: Tables) -> None:
    rcfg = RescueConfig()
    s = t[SCREEN.name].filter(pl.col("icc_adj").is_not_null() & pl.col("redundancy_abs").is_not_null())
    fig, ax = figure(1, 1, size="square")
    a = ax[0, 0]
    if not s.height:
        empty_panel(a, "no feature with both an ICC and a stopwatch association")
    else:
        for r in s.iter_rows(named=True):
            col = STATUS_COLORS.get(r["screening_status"], INK["muted"])
            x, y = r["redundancy_abs"], r["icc_adj"]
            xe = [[max(x - (r["redundancy_lower"] if r["redundancy_lower"] is not None else x), 0)],
                  [max((r["redundancy_upper"] if r["redundancy_upper"] is not None else x) - x, 0)]]
            ye = [[max(y - (r["icc_adj_lower"] if r["icc_adj_lower"] is not None else y), 0)],
                  [max((r["icc_adj_upper"] if r["icc_adj_upper"] is not None else y) - y, 0)]]
            a.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=col, alpha=0.5, linewidth=1)
            a.scatter([x], [y], s=20 + 0.6 * (r["n_athletes_repeat"] or 0), color=col, edgecolor="white", zorder=3)
            a.annotate(_nice(r["feature"]), (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8,
                       color=INK["secondary"])
        a.axhline(rcfg.icc_reference, color=INK["muted"], linestyle=":", linewidth=1)
        a.axvline(rcfg.redundancy_reference, color=INK["muted"], linestyle=":", linewidth=1)
        a.set_xlim(0, 1.02)
        a.set_xlabel("redundancy with the stopwatch: |position-adjusted Spearman| with official forty or 10-yd split")
        a.set_ylabel("within-position repeatability: ICC(1)")
        for lab, col in STATUS_COLORS.items():
            if lab in s["screening_status"].to_list():
                a.scatter([], [], color=col, label=lab.replace("_", " "))
        a.legend(frameon=False, fontsize=8, loc="lower left", title="screening label (heuristic)")
    finish(fig, f"One point per feature; marker area grows with the number of athletes with repeated attempts. Bars: 95 % "
                f"intervals. Dotted lines at ICC {rcfg.icc_reference:g} and |ρ| {rcfg.redundancy_reference:g} are heuristic "
                "screening references, not acceptance thresholds. Upper left = repeatable and not explained by the "
                "stopwatch; that is a reason to test a feature against NFL play, not evidence that it matters there.",
           "What does the stopwatch miss?")
    store.save_figure(fig, "SR_10_feature_screening", {"features": s}, publication=True, source=SCREEN.name,
                      n=f"{s.height} features")


FIGURES: list[tuple[str, tuple[str, ...], Callable[[RescueStore, Tables], None]]] = [
    ("SR_01_first_frame_speed", (ATTEMPTS.name,), fig_first_speed),
    ("SR_02_distance_vs_duration", (ATTEMPTS.name,), fig_distance_duration),
    ("SR_03_delta_start_vs_delta_accel", (WITHIN.name, ASSOC.name), fig_delta),
    ("SR_04_censoring_vs_repeatability", (ASSOC.name,), fig_censoring_icc),
    ("SR_05_example_recordings", (EXAMPLES.name,), fig_examples),
    ("SR_06_pooled_vs_within_position_icc", (RELIABILITY.name,), fig_icc_forest),
    ("SR_07_variance_components_and_slopes", (RELIABILITY.name, ATTEMPTS.name), fig_variance_slopes),
    ("SR_08_reliability_by_cohort_and_sample", (RELIABILITY.name,), fig_cohort_sample),
    ("SR_09_model_repeatability_and_sensitivity", (MODEL_REPEAT.name, FITS.name), fig_model),
    ("SR_10_feature_screening", (SCREEN.name,), fig_screening),
]
