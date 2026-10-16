"""Figures of notebooks 06A (cohort and leakage audit), 06B (transfer benchmark) and 06C (robustness and scouting
insights). Every function returns a Matplotlib figure drawn ONLY from tables the notebook computed; the notebook passes
it to ``TransferStore.show`` (inline display, PNG 300 DPI + SVG, PDF for publication figures, plot data as CSV).

Conventions: project 16:9 style; one question per figure; the denominator (athletes, never plays or attempts) in the
caption; units on every axis; error axes start at 0; difference axes are centred on an explicit zero line; intervals
are 95 % paired athlete-bootstrap intervals. Colour identifies the MODEL FAMILY in 06B / 06C and the position group or
the status in 06A. A panel without data says so instead of staying blank.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS
from ..figkit import empty_panel, figure, finish
from .transfer import ALL, STAGES
from .transfer_models import PRIMARY, REFERENCE

MODEL_ORDER = ("MEAN", "POS", "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7_PC2", "M7_PC4", "M5S")
MODEL_COLOR = {"MEAN": INK["axis"], "POS": INK["muted"], "M0": INK["secondary"], "M1": CATEGORICAL[0], "M2": CATEGORICAL[2],
               "M3": CATEGORICAL[3], "M4": CATEGORICAL[1], "M5": CATEGORICAL[6], "M6": INK["muted"], "M5S": CATEGORICAL[5]}
MODEL_NAME = {"MEAN": "training mean", "POS": "position mean", "M0": "M0 context", "M1": "M1 official", "M2": "M2 + engineered",
              "M3": "M3 + SPEED2", "M4": "M4 + FPCA", "M5": "M5 + TCN-B", "M6": "M6 + random encoder", "M5S": "M1 + SPEED2 + TCN-B"}
COMPAT_COLOR = {"compatible": STATUS["good"], "descriptive_only": STATUS["warning"], "combine_only_selection": INK["axis"]}
CI_NOTE = "Bars / whiskers: 95 % paired bootstrap of athletes on out-of-fold predictions."


# ============================================================================== helpers
def mcolor(m: str) -> str:
    return MODEL_COLOR.get(m, CATEGORICAL[4] if m.startswith("M7") else INK["secondary"])


def mname(m: str) -> str:
    return MODEL_NAME.get(m, f"M7 + TCN-B {m.rsplit('PC', 1)[-1]} axes" if m.startswith("M7") else m)


def _ordered(models: Sequence[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(set(models), key=lambda m: (rank.get(m, len(rank)), m))


def _a(values: Sequence[Any]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _pos_color(positions: Sequence[str]) -> dict[str, str]:
    return {p: CATEGORICAL[i % len(CATEGORICAL)] for i, p in enumerate(sorted(positions))}


def _whisk(ax, x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, colors: Sequence[str], horizontal: bool = True) -> None:
    """Points with asymmetric intervals; a missing bound draws no whisker."""
    for i in range(len(x)):
        c = x[i] if horizontal else y[i]
        err = np.array([[c - lo[i] if np.isfinite(lo[i]) else 0.0], [hi[i] - c if np.isfinite(hi[i]) else 0.0]]).clip(min=0)
        kw = {"xerr": err} if horizontal else {"yerr": err}
        ax.errorbar(x[i], y[i], fmt="o", color=colors[i], capsize=3, markersize=6, lw=1.6, **kw)


def _forest(ax, labels: Sequence[str], est: np.ndarray, lo: np.ndarray, hi: np.ndarray, colors: Sequence[str], xlabel: str,
            notes: Sequence[str] | None = None) -> None:
    if not len(labels):
        empty_panel(ax, "no comparison available")
        return
    y = np.arange(len(labels))[::-1].astype(float)
    _whisk(ax, est, y, lo, hi, colors)
    ax.axvline(0.0, color=INK["secondary"], lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(list(labels), fontsize=9)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel(xlabel)
    span = np.nanmax(np.abs(np.r_[est, lo, hi, 1e-9]))
    ax.set_xlim(-1.25 * span, 1.25 * span)
    for i, note in enumerate(notes or []):
        ax.text(1.01, y[i], note, transform=ax.get_yaxis_transform(), fontsize=8, va="center", color=INK["muted"])


def _heat(ax, M: np.ndarray, rows: Sequence[str], cols: Sequence[str], text: Sequence[Sequence[str]], vmax: float = 1.0) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cm = LinearSegmentedColormap.from_list("transfer", ["#ffffff", SEQUENTIAL_BLUE[4]])
    cm.set_bad(INK["grid"])
    ax.imshow(np.ma.masked_invalid(np.asarray(M, dtype=float)), cmap=cm, vmin=0.0, vmax=vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8, color=INK["primary"])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(list(cols), fontsize=8, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(list(rows), fontsize=9)
    ax.grid(False)


def _sel(df: pl.DataFrame, **eq: Any) -> pl.DataFrame:
    for k, v in eq.items():
        df = df.filter(pl.col(k).is_in(list(v)) if isinstance(v, (list, tuple, set)) else pl.col(k) == v)
    return df


def _legend(ax, entries: dict[str, str], **kw: Any) -> None:
    from matplotlib.lines import Line2D

    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in entries.items()], frameon=False, fontsize=8, **kw)


# ============================================================================== 06A — cohort and leakage audit
def plot_funnel(funnel: pl.DataFrame) -> Any:
    d = _sel(funnel, group_type=ALL).sort("stage_order")
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    y = np.arange(d.height)[::-1]
    roster = max(int(d["n_cumulative"][0]), 1)
    ax.barh(y, d["n_meeting_rule"].to_numpy(), color=INK["grid"], height=0.7, label="meets this rule alone")
    ax.barh(y, d["n_cumulative"].to_numpy(), color=CATEGORICAL[0], height=0.42, label="meets this rule and every earlier one")
    for yi, n, a in zip(y, d["n_cumulative"].to_list(), d["n_meeting_rule"].to_list()):
        ax.text(max(n, a) + 0.01 * roster, yi, f"{n} ({100 * n / roster:.0f} %) · alone {a}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in d["stage"].to_list()])
    ax.set_xlabel("athletes")
    ax.set_xlim(0, 1.28 * roster)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    finish(fig, f"Unique athletes at each rule of the Combine → NFL funnel (denominator: {roster} athletes of the Combine registry). "
           "The last bar is the primary cohort: the only population of the paired model comparisons. No NFL label value is read.",
           "From the Combine roster to the paired evaluation cohort")
    return fig


def plot_attrition(funnel: pl.DataFrame, group_type: str, title: str) -> Any:
    d = _sel(funnel, group_type=group_type).sort("group", "stage_order")
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    groups = sorted(d["group"].unique().to_list())
    if not groups:
        empty_panel(ax, "no group")
        return fig
    stages = [s for s, _, _ in STAGES]
    M = np.array([[_sel(d, group=g, stage=s)["n_cumulative"].sum() for s in stages] for g in groups], dtype=float)
    share = M / np.maximum(M[:, :1], 1)
    _heat(ax, share, groups, [s.replace("_", " ") for s in stages], [[f"{int(M[i, j])}\n{100 * share[i, j]:.0f} %" for j in range(len(stages))]
                                                                     for i in range(len(groups))])
    finish(fig, "Cell: athletes meeting the rule of the column and every earlier one, and their share of the group's Combine roster "
           "(first column). Shading follows the share.", title)
    return fig


def plot_missingness(missing: pl.DataFrame, population: str = "target_exposed") -> Any:
    d = _sel(missing, population=population).filter(pl.col("group_type").is_in([ALL, "position_group"]) & (pl.col("kind") != "engineered"))
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    variables = d.filter(pl.col("group_type") == ALL).sort("share_missing", descending=True)["variable"].to_list()
    groups = [ALL, *sorted(_sel(d, group_type="position_group")["group"].unique().to_list())]
    if not variables:
        empty_panel(ax, "no official measurement audited")
        return fig
    cell = {(r["variable"], r["group"]): r for r in d.iter_rows(named=True)}
    M = np.array([[cell[(v, g)]["share_missing"] if (v, g) in cell else np.nan for g in groups] for v in variables], dtype=float)
    text = [[f"{100 * cell[(v, g)]['share_missing']:.0f} %\n{cell[(v, g)]['n_missing']}/{cell[(v, g)]['n_athletes']}" if (v, g) in cell else ""
             for g in groups] for v in variables]
    rule = {r["variable"]: r["in_model"] for r in d.iter_rows(named=True)}
    _heat(ax, M, [f"{v.removeprefix('cmb_')}  [{rule[v].replace('_', ' ')}]" for v in variables], groups, text)
    finish(fig, f"Share of athletes without a recorded value (population: {population.replace('_', ' ')}). In brackets: how the variable "
           "enters the models, decided from coverage only. A missing value means 'did not participate or not reported': the source does "
           "not distinguish them.", "Official Combine measurements: what is missing, and for whom")
    return fig


def plot_exposure(cohort: pl.DataFrame, min_windows: int, target_id: str) -> Any:
    d = cohort.filter(pl.col("st_nfl"))
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    if not d.height:
        empty_panel(ax, "no NFL-tracked athlete")
        return fig
    w = np.maximum(d["target_windows"].to_numpy().astype(float), 0.5)
    bins = np.logspace(np.log10(0.5), np.log10(max(w.max(), 2.0)), 28)
    colors = _pos_color(d["position_group"].drop_nulls().unique().to_list())
    for p, c in colors.items():
        x = w[d["position_group"].to_numpy() == p]
        ax.hist(x, bins=bins, histtype="step", lw=1.8, color=c, label=f"{p} (n = {x.size}; ≥ {min_windows}: {int((x >= min_windows).sum())})")
    ax.axvline(min_windows, color=INK["primary"], lw=1.2, ls="--")
    ax.text(min_windows, ax.get_ylim()[1], f" frozen minimum {min_windows}", va="top", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(f"eligible regular-season windows of {target_id} per athlete (log scale; 0 shown at 0.5)")
    ax.set_ylabel("athletes")
    ax.legend(frameon=False, fontsize=9)
    finish(fig, f"{d.height} athletes with at least one eligible NFL window. The dashed line is the exposure requirement stored in the "
           "frozen target registry; it is not chosen here.", "NFL exposure on the primary target")
    return fig


def plot_games(cohort: pl.DataFrame) -> Any:
    d = cohort.filter(pl.col("st_nfl"))
    fig, axes = figure(1, 2)
    years = sorted(d["draft_year"].drop_nulls().unique().to_list())
    if not years:
        empty_panel(axes[0, 0], "no NFL-tracked athlete")
        empty_panel(axes[0, 1], "")
        return fig
    g = d["n_nfl_games"].to_numpy()
    bins = np.arange(0, g.max() + 3, 2)
    for i, y in enumerate(years):
        sub = d.filter(pl.col("draft_year") == y)
        axes[0, 0].hist(sub["n_nfl_games"].to_numpy(), bins=bins, histtype="step", lw=1.8, color=CATEGORICAL[i % 8],
                        label=f"{y} (n = {sub.height}, median {sub['n_nfl_games'].median():.0f} games)")
        axes[0, 1].bar(i, sub["n_nfl_seasons"].mean(), color=CATEGORICAL[i % 8])
        axes[0, 1].text(i, sub["n_nfl_seasons"].mean(), f"{sub['n_nfl_seasons'].mean():.2f}", ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_xlabel("games with an eligible window per athlete")
    axes[0, 0].set_ylabel("athletes")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].set_xticks(range(len(years)))
    axes[0, 1].set_xticklabels([str(y) for y in years])
    axes[0, 1].set_ylabel("mean number of NFL seasons observed")
    axes[0, 1].set_ylim(0, None)
    finish(fig, f"{d.height} NFL-tracked athletes by draft cohort. Older cohorts have a longer follow-up: the outcome population is not "
           "balanced in exposure, which the common-horizon sensitivity of notebook 06C addresses.", "Eligible games and follow-up by draft cohort")
    return fig


def plot_included_vs_excluded(cohort: pl.DataFrame, official: pl.DataFrame, variables: Sequence[tuple[str, str]]) -> Any:
    d = cohort.filter(pl.col("st_tracked")).join(official, on="nfl_id", how="left")
    fig, axes = figure(1, len(variables) + 1)
    inc, exc = d.filter(pl.col("in_primary_cohort")), d.filter(~pl.col("in_primary_cohort"))
    for k, (col, label) in enumerate([*variables, ("n_attempts", "eligible forty attempts")]):
        ax = axes[0, k]
        data = [x[col].drop_nulls().to_numpy().astype(float) for x in (inc, exc)]
        if not all(v.size for v in data):
            empty_panel(ax, f"{label}: a group has no value")
            continue
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], (CATEGORICAL[0], INK["axis"])):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"primary cohort\nn = {data[0].size}", f"excluded\nn = {data[1].size}"], fontsize=9)
        ax.set_ylabel(label)
    finish(fig, f"{d.height} athletes with a tracked forty: those in the primary cohort against those excluded (no NFL window, too little "
           "exposure, a missing mandatory measurement). Pre-NFL measurements only; boxes show quartiles, whiskers 1.5 IQR. A shift "
           "says the evaluated athletes are not a random sample of the Combine.", "Who is evaluated: primary cohort against excluded tracked athletes")
    return fig


def plot_target_availability(eligibility: pl.DataFrame, cohort: pl.DataFrame) -> Any:
    targets = eligibility.select("target_id", "rank").unique().sort("rank", "target_id")["target_id"].to_list()
    fig, axes = figure(1, max(len(targets), 1))
    base = cohort.filter(pl.col("st_nfl")).select("nfl_id", "position_group", "draft_year")
    pos, years = sorted(base["position_group"].drop_nulls().unique().to_list()), sorted(base["draft_year"].drop_nulls().unique().to_list())
    for k, t in enumerate(targets):
        ax = axes[0, k]
        e = base.join(_sel(eligibility, target_id=t).select("nfl_id", "eligible"), on="nfl_id", how="left").with_columns(pl.col("eligible").fill_null(False))
        if not pos or not years:
            empty_panel(ax, "no NFL-tracked athlete")
            continue
        n = np.array([[e.filter((pl.col("position_group") == p) & (pl.col("draft_year") == y)).height for y in years] for p in pos], dtype=float)
        ok = np.array([[int(e.filter((pl.col("position_group") == p) & (pl.col("draft_year") == y))["eligible"].sum()) for y in years] for p in pos], dtype=float)
        _heat(ax, np.where(n > 0, ok / np.maximum(n, 1), np.nan), pos, [str(y) for y in years],
              [[f"{int(ok[i, j])}/{int(n[i, j])}" for j in range(len(years))] for i in range(len(pos))])
        ax.set_title(f"{t}", fontsize=10)
    finish(fig, "Cell: athletes meeting the frozen exposure requirement of the target / NFL-tracked athletes of the position group and "
           "draft cohort. Shading: the share.", "Target availability by position group and draft cohort")
    return fig


def plot_fold_balance(folds: pl.DataFrame) -> Any:
    d = folds.filter(pl.col("in_primary_cohort"))
    fig, axes = figure(1, 2)
    fs = sorted(d["outer_fold"].unique().to_list())
    for k, (col, title) in enumerate((("position_group", "position group"), ("draft_year", "draft cohort"))):
        ax = axes[0, k]
        levels = sorted(d[col].drop_nulls().unique().to_list())
        bottom = np.zeros(len(fs))
        for i, lv in enumerate(levels):
            n = np.array([d.filter((pl.col("outer_fold") == f) & (pl.col(col) == lv)).height for f in fs], dtype=float)
            ax.bar(range(len(fs)), n, bottom=bottom, color=CATEGORICAL[i % 8], label=str(lv))
            for j in range(len(fs)):
                if n[j]:
                    ax.text(j, bottom[j] + n[j] / 2, f"{int(n[j])}", ha="center", va="center", fontsize=8, color="white")
            bottom += n
        ax.set_xticks(range(len(fs)))
        ax.set_xticklabels([f"fold {f}\nn = {int(b)}" for f, b in zip(fs, bottom)])
        ax.set_ylabel("held-out athletes (primary cohort)")
        ax.set_title(f"by {title}", fontsize=11)
        ax.legend(frameon=False, fontsize=8, ncol=len(levels))
    src = d["fold_source"][0] if d.height else "n/a"
    finish(fig, f"{d.height} primary-cohort athletes, one outer fold each (design: {src}). The folds are stratified by position group × "
           "draft cohort and use no measurement and no NFL value.", "Outer-fold balance")
    return fig


def plot_compatibility(compat: pl.DataFrame) -> Any:
    d = compat.filter(pl.col("status") != "combine_only_selection")
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    if not d.height:
        empty_panel(ax, "no saved representation fit was found for this run mode")
        finish(fig, "Nothing to audit: every representation is fitted inside the transfer folds in notebook 06B.", "Saved representation fits against the transfer folds")
        return fig
    d = d.with_columns((pl.col("family") + " · " + pl.col("method").str.split(":").list.first() + pl.when(pl.col("repeat") != 0)
                        .then(pl.lit(" (other repeat / reference)")).otherwise(pl.lit(""))).alias("row"))
    rows, cols = sorted(d["row"].unique().to_list()), sorted(d["fold"].unique().to_list())
    code = {"compatible": 1.0, "descriptive_only": 0.0}
    from matplotlib.colors import ListedColormap

    M, text = np.full((len(rows), len(cols)), np.nan), [["" for _ in cols] for _ in rows]
    for r in d.iter_rows(named=True):
        i, j = rows.index(r["row"]), cols.index(r["fold"])
        M[i, j] = min(code[r["status"]], M[i, j]) if np.isfinite(M[i, j]) else code[r["status"]]
        text[i][j] = f"{r['n_overlap']} shared" if r["n_overlap"] else ("reuse" if r["status"] == "compatible" else "refit")
    cm = ListedColormap([COMPAT_COLOR["descriptive_only"], COMPAT_COLOR["compatible"]])
    cm.set_bad(INK["grid"])
    ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["all athletes" if c < 0 else f"fold {c}" for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.grid(False)
    finish(fig, "Green: no athlete held out in the transfer fold took part in the saved fit (weights, early stopping, normalisation) — the "
           "fit may be reused. Amber: descriptive only; notebook 06B refits on the training athletes of the fold. 'n shared' = athletes "
           "held out in the transfer fold that the saved fit used.", "Saved representation fits against the transfer folds")
    return fig


# ============================================================================== 06B — primary benchmark
def plot_actual_vs_predicted(oof: pl.DataFrame, target_id: str, models: Sequence[str], unit: str, label: str) -> Any:
    d = _sel(oof, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id)
    models = [m for m in models if m in d["model"].unique().to_list()]
    fig, axes = figure(1, max(len(models), 1))
    colors = _pos_color(d["position_group"].unique().to_list())
    for k, m in enumerate(models):
        ax, g = axes[0, k], _sel(d, model=m)
        y, p = g["y"].to_numpy(), g["pred"].to_numpy()
        ax.scatter(p, y, s=16, alpha=0.75, c=[colors[x] for x in g["position_group"].to_list()], linewidths=0)
        lim = [float(min(y.min(), p.min())), float(max(y.max(), p.max()))]
        ax.plot(lim, lim, color=INK["secondary"], lw=1.0, ls="--")
        ax.set_xlabel(f"out-of-fold prediction ({unit})")
        ax.set_ylabel(f"observed {label} ({unit})")
        ax.set_title(f"{mname(m)} · RMSE {np.sqrt(np.mean((y - p) ** 2)):.3f} · r {np.corrcoef(y, p)[0, 1]:.2f}", fontsize=10, color=mcolor(m))
    if models:
        _legend(axes[0, 0], colors, loc="upper left")
    else:
        empty_panel(axes[0, 0], "no out-of-fold prediction")
    n = d["nfl_id"].n_unique()
    finish(fig, f"{n} athletes of the paired cohort, each predicted by a model fitted without him. Dashed: identity. Much of the spread "
           "along the diagonal is the difference between position groups, which every model knows.", f"Observed against predicted NFL movement — {target_id}")
    return fig


def plot_error_metric(perf: pl.DataFrame, target_id: str, metric: str, unit: str) -> Any:
    d = _sel(perf, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id, metric=metric)
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    models = _ordered(d["model"].to_list())
    if not models:
        empty_panel(ax, "no performance row")
        return fig
    r = {x["model"]: x for x in d.iter_rows(named=True)}
    est, lo, hi = (_a([r[m][k] for m in models]) for k in ("estimate", "ci_lower", "ci_upper"))
    y = np.arange(len(models))[::-1].astype(float)
    ax.barh(y, est, color=[mcolor(m) for m in models], height=0.62, alpha=0.85)
    _whisk(ax, est, y, lo, hi, [INK["primary"]] * len(models))
    for ref, ls in (("MEAN", ":"), (REFERENCE, "--")):
        if ref in r and r[ref]["estimate"] is not None:
            ax.axvline(r[ref]["estimate"], color=mcolor(ref) if ref != "MEAN" else INK["secondary"], lw=1.1, ls=ls)
            ax.text(r[ref]["estimate"], len(models) - 0.45, f" {mname(ref)}", fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([mname(m) for m in models])
    ax.set_xlim(0, None)
    ax.set_xlabel(f"held-out {metric.upper()} ({unit}) — lower is better")
    finish(fig, f"{d['n_athletes'][0]} athletes, {d['n_positions'][0]} position groups, {d['n_cohorts'][0]} draft cohorts; identical athletes "
           f"for every model. {CI_NOTE} Overlapping intervals do not decide a comparison: read the paired differences.",
           f"Held-out {metric.upper()} by model — {target_id}")
    return fig


def plot_skill_metric(perf: pl.DataFrame, target_id: str, metric: str, label: str) -> Any:
    d = _sel(perf, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id, metric=metric)
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    models = _ordered(d["model"].to_list())
    if not models:
        empty_panel(ax, "no performance row")
        return fig
    r = {x["model"]: x for x in d.iter_rows(named=True)}
    est, lo, hi = (_a([r[m][k] for m in models]) for k in ("estimate", "ci_lower", "ci_upper"))
    x = np.arange(len(models)).astype(float)
    _whisk(ax, x, est, lo, hi, [mcolor(m) for m in models], horizontal=False)
    ax.axhline(0.0, color=INK["secondary"], lw=1.0, ls=":")
    if REFERENCE in r and r[REFERENCE]["estimate"] is not None:
        ax.axhline(r[REFERENCE]["estimate"], color=mcolor(REFERENCE), lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([mname(m) for m in models], rotation=20, ha="right")
    ax.set_ylabel(label)
    top = np.nanmax(np.r_[est, hi, 0.05])
    ax.set_ylim(min(0.0, np.nanmin(np.r_[est, lo, 0.0])) - 0.03, top + 0.05)
    finish(fig, f"{d['n_athletes'][0]} athletes. Dotted: 0 (no skill). Dashed: official Combine model M1. {CI_NOTE} A pooled value "
           "includes the differences between position groups.", f"{label} by model — {target_id}")
    return fig


def plot_delta(paired: pl.DataFrame, target_id: str, unit: str, reference: str = REFERENCE, metric: str = "delta_rmse",
               analysis: str = PRIMARY, title: str | None = None) -> Any:
    d = _sel(paired, analysis=analysis, cohort=PRIMARY, target_id=target_id, reference=reference, metric=metric)
    fig, axes = figure(1, 1)
    models = _ordered(d["model"].to_list())
    r = {x["model"]: x for x in d.iter_rows(named=True)}
    notes = [("" if r[m]["relative"] is None else f"{100 * r[m]['relative']:+.1f} %  ") + ("established" if r[m]["established"] else
             "worse" if r[m]["worse"] else "includes 0") for m in models]
    _forest(axes[0, 0], [mname(m) for m in models], *(_a([r[m][k] for m in models]) for k in ("estimate", "ci_lower", "ci_upper")),
            [mcolor(m) for m in models], f"Δ RMSE = RMSE({mname(reference)}) − RMSE(model) ({unit}); right of 0 = lower error", notes)
    n = d["n_athletes"][0] if d.height else 0
    finish(fig, f"{n} athletes, the same for both models of every row. Whiskers: 95 % paired bootstrap of athletes. When the interval "
           "includes 0 an incremental predictive benefit is not established.", title or f"Error reduction relative to {mname(reference)} — {target_id}")
    return fig


def plot_speed_vs_acceleration(paired: pl.DataFrame, perf: pl.DataFrame, targets: Sequence[tuple[str, str]], models: Sequence[str]) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    width = 0.8 / max(len(models), 1)
    drawn = False
    for i, (tid, role) in enumerate(targets):
        base = _sel(perf, analysis=PRIMARY, cohort=PRIMARY, target_id=tid, model=REFERENCE, metric="rmse")
        if not base.height or not base["estimate"][0]:
            continue
        b = float(base["estimate"][0])
        for j, m in enumerate(models):
            r = _sel(paired, analysis=PRIMARY, cohort=PRIMARY, target_id=tid, model=m, reference=REFERENCE, metric="delta_rmse")
            if not r.height:
                continue
            x = i - 0.4 + (j + 0.5) * width
            _whisk(ax, np.array([x]), 100 * _a(r["estimate"].to_list()) / b, 100 * _a(r["ci_lower"].to_list()) / b,
                   100 * _a(r["ci_upper"].to_list()) / b, [mcolor(m)], horizontal=False)
            drawn = True
    if not drawn:
        empty_panel(ax, "no frozen target evaluated")
        return fig
    ax.axhline(0.0, color=INK["secondary"], lw=1.0, ls="--")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([f"{t}\n({r.replace('_', ' ')})" for t, r in targets], fontsize=9)
    ax.set_ylabel("Δ RMSE relative to M1 (% of the M1 error); above 0 = lower error")
    _legend(ax, {mname(m): mcolor(m) for m in models}, loc="best")
    finish(fig, "Each frozen target has its own paired cohort (athletes meeting its exposure requirement). Speed transfer and "
           "acceleration transfer are separate questions: a gain on a speed target would say the tracking re-measures speed, not that "
           "it adds an acceleration trait. Secondary and exploratory targets cannot replace the primary conclusion.",
           "Does Combine tracking add to official testing — for NFL speed and for NFL acceleration?")
    return fig


def plot_representation_contrast(paired: pl.DataFrame, target_id: str, unit: str, pairs: Sequence[tuple[str, str]]) -> Any:
    d = _sel(paired, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id, metric="delta_rmse")
    fig, axes = figure(1, 1)
    rows = [(m, r, x) for m, r in pairs for x in d.filter((pl.col("model") == m) & (pl.col("reference") == r)).iter_rows(named=True)]
    _forest(axes[0, 0], [f"{mname(m)}  vs  {mname(r)}" for m, r, _ in rows], *(_a([x[k] for _, _, x in rows]) for k in ("estimate", "ci_lower", "ci_upper")),
            [mcolor(m) for m, _, _ in rows], f"Δ RMSE = RMSE(second model) − RMSE(first model) ({unit}); right of 0 = the first is better",
            ["established" if x["established"] else "worse" if x["worse"] else "includes 0" for _, _, x in rows])
    finish(fig, "The encoder against every simpler or control representation on the same athletes: engineered movement (M2), SPEED2 "
           "(M3), FPCA (M4), an untrained encoder of the same size (M6), and added on top of SPEED2 (M1 + SPEED2 + TCN-B vs M3). If SPEED2 "
           "or the random encoder matches TCN-B, the simpler explanation is preferred.", f"TCN-B against SPEED2, FPCA and controls — {target_id}")
    return fig


def plot_residuals(oof: pl.DataFrame, target_id: str, models: Sequence[str], unit: str) -> Any:
    d = _sel(oof, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id)
    models = [m for m in models if m in d["model"].unique().to_list()]
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    if not models:
        empty_panel(ax, "no out-of-fold prediction")
        return fig
    res = {m: (_sel(d, model=m)["y"] - _sel(d, model=m)["pred"]).to_numpy() for m in models}
    span = max(float(np.abs(np.concatenate(list(res.values()))).max()), 1e-9)
    bins = np.linspace(-span, span, 31)
    for m in models:
        ax.hist(res[m], bins=bins, histtype="step", lw=1.8, color=mcolor(m), label=f"{mname(m)} (SD {res[m].std():.3f}, median {np.median(res[m]):+.3f})")
    ax.axvline(0.0, color=INK["secondary"], lw=1.0, ls="--")
    ax.set_xlabel(f"observed − out-of-fold prediction ({unit})")
    ax.set_ylabel("athletes")
    ax.legend(frameon=False, fontsize=9)
    finish(fig, f"{d['nfl_id'].n_unique()} athletes; one residual per athlete and model. The histograms share their bins.",
           f"Out-of-fold residuals — {target_id}")
    return fig


def plot_calibration(oof: pl.DataFrame, perf: pl.DataFrame, target_id: str, models: Sequence[str], unit: str, n_bins: int = 8) -> Any:
    d = _sel(oof, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id)
    models = [m for m in models if m in d["model"].unique().to_list()]
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    if not models:
        empty_panel(ax, "no out-of-fold prediction")
        return fig
    lim = [float(d["y"].min()), float(d["y"].max())]
    for m in models:
        g = _sel(d, model=m).sort("pred")
        parts = np.array_split(np.arange(g.height), min(n_bins, max(g.height // 5, 1)))
        p, y = g["pred"].to_numpy(), g["y"].to_numpy()
        slope = _sel(perf, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id, model=m, metric="calibration_slope")
        txt = f"{slope['estimate'][0]:.2f}" if slope.height and slope["estimate"][0] is not None else "n/a"
        ax.plot([p[i].mean() for i in parts], [y[i].mean() for i in parts], "o-", color=mcolor(m), lw=1.5, label=f"{mname(m)} (slope {txt})")
    ax.plot(lim, lim, color=INK["secondary"], lw=1.0, ls="--")
    ax.set_xlabel(f"mean out-of-fold prediction of the bin ({unit})")
    ax.set_ylabel(f"mean observed value of the bin ({unit})")
    ax.legend(frameon=False, fontsize=9)
    finish(fig, f"Athletes ordered by prediction and cut into up to {n_bins} equal groups. Dashed: perfect calibration. Slope = regression "
           "of observed on predicted (1 = calibrated; below 1 = predictions too spread out).", f"Calibration of out-of-fold predictions — {target_id}")
    return fig


def plot_fold_performance(fold_perf: pl.DataFrame, target_id: str, models: Sequence[str], unit: str) -> Any:
    d = _sel(fold_perf, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id)
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    fs = sorted(d["fold"].unique().to_list())
    if not fs:
        empty_panel(ax, "no fold result")
        return fig
    for m in [m for m in models if m != REFERENCE]:
        g = _sel(d, model=m).sort("fold")
        if g.height:
            ax.plot(g["fold"].to_numpy(), g["delta_rmse_vs_m1"].to_numpy(), "o-", color=mcolor(m), lw=1.4,
                    label=f"{mname(m)} ({int((g['delta_rmse_vs_m1'] > 0).sum())}/{g.height} folds above 0)")
    ax.axhline(0.0, color=INK["secondary"], lw=1.0, ls="--")
    n = {r["fold"]: r["n_athletes"] for r in _sel(d, model=REFERENCE).iter_rows(named=True)}
    ax.set_xticks(fs)
    ax.set_xticklabels([f"fold {f}\nn = {n.get(f, '?')}" for f in fs])
    ax.set_ylabel(f"Δ RMSE against M1 inside the fold ({unit}); above 0 = lower error")
    ax.legend(frameon=False, fontsize=9)
    finish(fig, "Each point: one outer fold (its held-out athletes only). A gain carried by a single fold is fragile. Fold values are "
           "noisy by construction (a fifth of the athletes): they show consistency, not significance.", f"Consistency across outer folds — {target_id}")
    return fig


def plot_coverage_vs_error(oof: pl.DataFrame, labels: pl.DataFrame, target_id: str, models: Sequence[str], unit: str) -> Any:
    d = (_sel(oof, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id)
         .join(_sel(labels, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id).select("nfl_id", "n_windows"), on="nfl_id", how="inner"))
    models = [m for m in models if m in d["model"].unique().to_list()]
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    if not models:
        empty_panel(ax, "no out-of-fold prediction")
        return fig
    for m in models:
        g = _sel(d, model=m).sort("n_windows")
        w, e = g["n_windows"].to_numpy().astype(float), (g["y"] - g["pred"]).abs().to_numpy()
        ax.scatter(w, e, s=10, alpha=0.25, color=mcolor(m), linewidths=0)
        parts = np.array_split(np.arange(g.height), min(6, max(g.height // 8, 1)))
        ax.plot([np.median(w[i]) for i in parts], [np.median(e[i]) for i in parts], "o-", color=mcolor(m), lw=1.8, label=f"{mname(m)} (median of bins)")
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.set_xlabel("eligible NFL windows behind the athlete's label (log scale)")
    ax.set_ylabel(f"absolute out-of-fold error ({unit})")
    ax.legend(frameon=False, fontsize=9)
    finish(fig, f"{d['nfl_id'].n_unique()} athletes. A label built from few windows is noisier, so part of the error at low exposure is "
           "measurement error of the OUTCOME and no model can remove it.", f"Prediction error against NFL exposure — {target_id}")
    return fig


# ============================================================================== 06C — robustness and scouting insights
def plot_subgroup_gains(sub_paired: pl.DataFrame, paired: pl.DataFrame, target_id: str, group_type: str, models: Sequence[str], unit: str,
                        title: str) -> Any:
    d = _sel(sub_paired, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id, group_type=group_type, reference=REFERENCE, metric="delta_rmse")
    fig, axes = figure(1, max(len(models), 1))
    for k, m in enumerate(models):
        g = _sel(d, model=m).sort("group")
        pooled = _sel(paired, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id, model=m, reference=REFERENCE, metric="delta_rmse")
        rows = [*[(f"{r['group']} (n = {r['n_athletes']})", r) for r in g.iter_rows(named=True)],
                *[(f"pooled (n = {r['n_athletes']})", r) for r in pooled.iter_rows(named=True)]]
        _forest(axes[0, k], [x for x, _ in rows], *(_a([r[c] for _, r in rows]) for c in ("estimate", "ci_lower", "ci_upper")),
                [mcolor(m)] * (len(rows) - pooled.height) + [INK["primary"]] * pooled.height, f"Δ RMSE vs M1 ({unit})")
        axes[0, k].set_title(mname(m), fontsize=10, color=mcolor(m))
    finish(fig, "Pooled models scored inside each subgroup (no subgroup-specific model). A subgroup without a whisker has too few "
           "athletes for an interval. Differences between subgroups are mostly estimation noise unless the heterogeneity test says "
           "otherwise; a subgroup-only gain needs replication.", title)
    return fig


def plot_cohort_performance(sub_perf: pl.DataFrame, loco_paired: pl.DataFrame, target_id: str, models: Sequence[str], unit: str) -> Any:
    fig, axes = figure(1, 2)
    d = _sel(sub_perf, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id, group_type="draft_year", metric="rmse")
    years = sorted(d["group"].unique().to_list())
    ax = axes[0, 0]
    if years:
        width = 0.8 / max(len(models), 1)
        for j, m in enumerate(models):
            g = {r["group"]: r for r in _sel(d, model=m).iter_rows(named=True)}
            x = np.array([i - 0.4 + (j + 0.5) * width for i, y in enumerate(years) if y in g])
            vals = [g[y] for y in years if y in g]
            _whisk(ax, x, _a([v["estimate"] for v in vals]), _a([v["ci_lower"] for v in vals]), _a([v["ci_upper"] for v in vals]),
                   [mcolor(m)] * len(vals), horizontal=False)
        n = {r["group"]: r["n_athletes"] for r in d.iter_rows(named=True)}
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels([f"{y}\nn = {n[y]}" for y in years])
        ax.set_ylim(0, None)
        ax.set_ylabel(f"held-out RMSE ({unit}), stratified folds")
        _legend(ax, {mname(m): mcolor(m) for m in models}, loc="lower right")
    else:
        empty_panel(ax, "no cohort result")
    lo = _sel(loco_paired, target_id=target_id, reference=REFERENCE, metric="delta_rmse") if loco_paired.height else loco_paired
    rows = [r for m in models for r in _sel(lo, model=m).iter_rows(named=True)] if lo.height else []
    _forest(axes[0, 1], [f"{mname(r['model'])} (n = {r['n_athletes']})" for r in rows], *(_a([r[c] for r in rows]) for c in ("estimate", "ci_lower", "ci_upper")),
            [mcolor(r["model"]) for r in rows], f"Δ RMSE vs M1 with each draft cohort left out ({unit})")
    axes[0, 1].set_title("leave-one-draft-cohort-out (encoders refitted without the cohort)", fontsize=10)
    finish(fig, "Left: error by draft cohort under the stratified folds. Right: every athlete predicted by models — representation fits "
           "included — that never saw his draft cohort. Cohorts differ in sensor generation and in NFL follow-up as well as in athletes, "
           "so a cohort difference is not an athlete-development effect.", f"Draft-cohort generalisation — {target_id}")
    return fig


def plot_exposure_sensitivity(sens_paired: pl.DataFrame, plan: Sequence[dict[str, Any]], target_id: str, pairs: Sequence[tuple[str, str]],
                              unit: str, frozen_min: int) -> Any:
    kinds = [("min_windows", "minimum eligible windows"), ("first_n", "first N windows in time"), ("min_games", "minimum games"),
             ("horizon", "seasons after the draft year")]
    fig, axes = figure(1, len(kinds))
    for k, (kind, label) in enumerate(kinds):
        ax = axes[0, k]
        rows = sorted([p for p in plan if p["kind"] == kind], key=lambda p: p["x"])
        drawn = False
        for j, (m, r) in enumerate(pairs):
            pts = [(p["x"], x) for p in rows for x in _sel(sens_paired, analysis=p["analysis"], target_id=target_id, model=m, reference=r,
                                                           metric="delta_rmse").iter_rows(named=True)]
            if not pts:
                continue
            xs = np.arange(len(pts)) + (j - (len(pairs) - 1) / 2) * 0.12
            _whisk(ax, xs, _a([x["estimate"] for _, x in pts]), _a([x["ci_lower"] for _, x in pts]), _a([x["ci_upper"] for _, x in pts]),
                   [mcolor(m)] * len(pts), horizontal=False)
            if not drawn:
                ax.set_xticks(np.arange(len(pts)))
                ax.set_xticklabels([f"{v:g}{' *' if kind == 'min_windows' and int(v) == frozen_min else ''}\nn = {x['n_athletes']}" for v, x in pts], fontsize=8)
            drawn = True
        if not drawn:
            empty_panel(ax, f"{label}: not evaluable")
            continue
        ax.axhline(0.0, color=INK["secondary"], lw=1.0, ls="--")
        ax.set_xlabel(label)
        if k == 0:
            ax.set_ylabel(f"Δ RMSE of the model against its reference ({unit})")
            _legend(ax, {f"{mname(m)} vs {mname(r)}": mcolor(m) for m, r in pairs}, loc="best")
    finish(fig, "Every rule of the declared grids is shown; * marks the frozen exposure requirement. Each rule has its own eligible "
           "athletes (n), identical for both models of a pair. No rule is selected: a conclusion that holds under one threshold only "
           "is not confirmed.", f"Sensitivity to the NFL exposure rule — {target_id}")
    return fig


def plot_context(sens_paired: pl.DataFrame, target_id: str, pairs: Sequence[tuple[str, str]], unit: str, frozen_adjustment: str) -> Any:
    d = sens_paired.filter(pl.col("analysis").str.starts_with("context:") & (pl.col("target_id") == target_id) & (pl.col("metric") == "delta_rmse"))
    fig, axes = figure(1, max(len(pairs), 1))
    mark = {"none": "context:raw", "cell_residual": "context:cell_adjusted"}.get(frozen_adjustment, "")
    for k, (m, r) in enumerate(pairs):
        g = d.filter((pl.col("model") == m) & (pl.col("reference") == r)).sort("analysis")
        rows = list(g.iter_rows(named=True))
        _forest(axes[0, k], [f"{x['analysis'].removeprefix('context:').replace('_', ' ')}{' *' if x['analysis'] == mark else ''} (n = {x['n_athletes']})"
                             for x in rows], *(_a([x[c] for x in rows]) for c in ("estimate", "ci_lower", "ci_upper")), [mcolor(m)] * len(rows),
                f"Δ RMSE ({unit})")
        axes[0, k].set_title(f"{mname(m)} vs {mname(r)}", fontsize=10, color=mcolor(m))
    finish(fig, "raw = label on all windows; cell adjusted = context-cell means (position × alignment × play class) removed with the "
           "training athletes of each fold; class = windows of one play class only. * marks the frozen definition. Context labels are "
           "alignment and play class only: assignment and scheme are unobserved, so a stable label can still reflect deployment.",
           f"Robustness to the NFL context definition — {target_id}")
    return fig


def plot_matched(pair_sum: pl.DataFrame, target_id: str, models: Sequence[str]) -> Any:
    d = _sel(pair_sum, target_id=target_id)
    fig, axes = figure(1, 2)
    r = {x["model"]: x for x in d.iter_rows(named=True)}
    models = [m for m in models if m in r]
    if not models:
        empty_panel(axes[0, 0], "no matched pair inside the caliper")
        empty_panel(axes[0, 1], "")
        return fig
    x = np.arange(len(models)).astype(float)
    ax = axes[0, 0]
    _whisk(ax, x, *(_a([r[m][k] for m in models]) for k in ("concordance", "concordance_lower", "concordance_upper")),
           [mcolor(m) for m in models], horizontal=False)
    ax.axhline(0.5, color=INK["secondary"], lw=1.0, ls="--")
    ax.text(len(models) - 0.5, 0.5, "chance ", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([mname(m) for m in models], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of matched pairs ordered correctly")
    rest = [m for m in models if r[m]["d_concordance"] is not None]
    _forest(axes[0, 1], [mname(m) for m in rest], *(_a([r[m][k] for m in rest]) for k in ("d_concordance", "d_concordance_lower", "d_concordance_upper")),
            [mcolor(m) for m in rest], "concordance − concordance of the official-only model M1")
    one = next(iter(r.values()))
    finish(fig, f"{one['n_pairs']} pairs of {one['n_athletes']} athletes: same position, official forty, 10-yard split and body size inside the "
           f"frozen caliper; an athlete belongs to up to {one['max_reuse']} pairs, so intervals resample ATHLETES. A pair is ordered "
           "correctly when the out-of-fold predicted difference has the sign of the observed NFL difference. Pairs are formed without any "
           "NFL value.", f"Similar stopwatch, different NFL movement: can a model tell matched prospects apart? — {target_id}")
    return fig


def plot_mechanism(x: np.ndarray, y: np.ndarray, position: np.ndarray, feature_label: str, target_label: str, row: dict[str, Any] | None) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        empty_panel(ax, "too few athletes with both values")
        return fig
    colors = _pos_color(np.unique(position[ok]).tolist())
    for p, c in colors.items():
        m = ok & (position == p)
        ax.scatter(x[m] - x[m].mean(), y[m] - y[m].mean(), s=18, alpha=0.75, color=c, linewidths=0, label=f"{p} (n = {int(m.sum())})")
    ax.axhline(0.0, color=INK["grid"], lw=1.0)
    ax.axvline(0.0, color=INK["grid"], lw=1.0)
    ax.set_xlabel(f"{feature_label} — deviation from the position-group mean")
    ax.set_ylabel(f"{target_label} — deviation from the position-group mean")
    ax.legend(frameon=False, fontsize=9)
    txt = "" if row is None else (f" Within-position Spearman {row['spearman_within_position']:+.2f}; after {row['covariates']}: partial r "
                                  f"{row['partial_r']:+.2f} [{row['partial_lower'] if row['partial_lower'] is None else round(row['partial_lower'], 2)}, "
                                  f"{row['partial_upper'] if row['partial_upper'] is None else round(row['partial_upper'], 2)}]. {row['note']}")
    finish(fig, f"{int(ok.sum())} athletes; both axes centred inside the position group, so the picture shows differences between athletes "
           f"of the same position.{txt} In-sample and descriptive: the held-out answer is model M2 against M1.",
           "Combine speed development against NFL acceleration")
    return fig


def plot_rank_shift(shift: pl.DataFrame, a: str, b: str, target_id: str) -> Any:
    fig, axes = figure(1, 2)
    if not shift.height:
        empty_panel(axes[0, 0], "no prediction")
        empty_panel(axes[0, 1], "")
        return fig
    colors = _pos_color(shift["position_group"].unique().to_list())
    n = shift["n_position"].to_numpy().astype(float)
    pa, pb = shift["rank_a"].to_numpy() / n, shift["rank_b"].to_numpy() / n
    axes[0, 0].scatter(pa, pb, s=16, alpha=0.75, c=[colors[p] for p in shift["position_group"].to_list()], linewidths=0)
    axes[0, 0].plot([0, 1], [0, 1], color=INK["secondary"], lw=1.0, ls="--")
    axes[0, 0].set_xlabel(f"rank inside the position group under {mname(a)} (0 = highest predicted)")
    axes[0, 0].set_ylabel(f"rank inside the position group under {mname(b)}")
    _legend(axes[0, 0], colors, loc="upper left")
    g = shift["rank_gain_b"].to_numpy()
    axes[0, 1].hist(g, bins=min(31, max(int(np.ptp(g)) + 1, 5)), color=mcolor(b), alpha=0.8)
    axes[0, 1].axvline(0.0, color=INK["secondary"], lw=1.0, ls="--")
    axes[0, 1].set_xlabel(f"places gained under {mname(b)} relative to {mname(a)} (within position)")
    axes[0, 1].set_ylabel("athletes")
    finish(fig, f"{shift.height} athletes ranked inside their position group by out-of-fold prediction; median absolute change "
           f"{np.median(np.abs(g)):.1f} places. A re-ranking is a difference between models, not evidence that the new order is better: "
           "that is what the error and matched-pair comparisons measure.", f"How much the encoder re-orders prospects — {target_id}")
    return fig


def plot_residuals_by_position(oof: pl.DataFrame, target_id: str, models: Sequence[str], unit: str) -> Any:
    d = _sel(oof, analysis=PRIMARY, cohort=PRIMARY, target_id=target_id).with_columns((pl.col("y") - pl.col("pred")).alias("res"))
    models = [m for m in models if m in d["model"].unique().to_list()]
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    pos = sorted(d["position_group"].unique().to_list())
    if not models or not pos:
        empty_panel(ax, "no out-of-fold prediction")
        return fig
    width = 0.8 / len(models)
    for j, m in enumerate(models):
        data = [d.filter((pl.col("model") == m) & (pl.col("position_group") == p))["res"].to_numpy() for p in pos]
        bp = ax.boxplot(data, positions=[i - 0.4 + (j + 0.5) * width for i in range(len(pos))], widths=0.85 * width, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(mcolor(m))
            patch.set_alpha(0.75)
    ax.axhline(0.0, color=INK["secondary"], lw=1.0, ls="--")
    n = {p: d.filter(pl.col("position_group") == p)["nfl_id"].n_unique() for p in pos}
    ax.set_xticks(range(len(pos)))
    ax.set_xticklabels([f"{p}\nn = {n[p]}" for p in pos])
    ax.set_ylabel(f"observed − out-of-fold prediction ({unit})")
    _legend(ax, {mname(m): mcolor(m) for m in models}, loc="best")
    finish(fig, "Boxes: quartiles of the athletes' residuals; whiskers 1.5 IQR. A box off zero would mean a position group is predicted "
           "with a bias; a wider box means the outcome is harder to predict there. Small groups (TE) are shown with their n.",
           f"Out-of-fold residuals by position group — {target_id}")
    return fig


def plot_ci_stability(stab: pl.DataFrame, unit: str) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    if not stab.height:
        empty_panel(ax, "comparison not available")
        return fig
    budgets = sorted(stab["n_boot"].unique().to_list())
    for s in sorted(stab["boot_seed"].unique().to_list()):
        g = stab.filter(pl.col("boot_seed") == s).sort("n_boot")
        x = [budgets.index(b) + (s - 2) * 0.06 for b in g["n_boot"].to_list()]
        ax.vlines(x, _a(g["ci_lower"].to_list()), _a(g["ci_upper"].to_list()), color=CATEGORICAL[s % 8], lw=2.0, alpha=0.85)
    ax.axhline(float(stab["estimate"][0]), color=INK["primary"], lw=1.2)
    ax.axhline(0.0, color=INK["secondary"], lw=1.0, ls="--")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("bootstrap replicates (each segment: one bootstrap seed)")
    ax.set_ylabel(f"95 % interval of Δ RMSE ({unit})")
    finish(fig, f"{mname(stab['model'][0])} against {mname(stab['reference'][0])} on {stab['target_id'][0]}. Solid line: the point estimate. "
           "If segments of one budget sit on different sides of the dashed zero line, the conclusion depends on resampling noise and "
           "more replicates are needed before any statement.", "Stability of the paired interval")
    return fig


def plot_evidence(evidence: pl.DataFrame, min_relative_gain: float, perm_alpha: float) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    if not evidence.height:
        empty_panel(ax, "no dimension evaluated")
        return fig
    cols = ["gain established\n(interval > 0)", f"relative gain\n≥ {100 * min_relative_gain:g} %", "beats random\nencoder", "cohort left out\n> 0",
            "positions with\na positive estimate", "all training\nseeds > 0", f"permutation\np < {perm_alpha:g}", "matched pairs\nbetter than official"]
    rows, M, text = [], [], []
    for e in evidence.iter_rows(named=True):
        def tri(ok: bool | None) -> float:
            return np.nan if ok is None else (1.0 if ok else 0.0)
        cells = [
            (tri(e["established"]), f"{e['delta_rmse']:+.3f}\n[{'n/a' if e['ci_lower'] is None else format(e['ci_lower'], '+.3f')}, "
                                    f"{'n/a' if e['ci_upper'] is None else format(e['ci_upper'], '+.3f')}]"),
            (tri(None if e["relative"] is None else e["relative"] >= min_relative_gain), "n/a" if e["relative"] is None else f"{100 * e['relative']:+.1f} %"),
            (tri(None if e["vs_random_lower"] is None else e["vs_random_lower"] > 0), "n/a" if e["vs_random_lower"] is None else f"lower {e['vs_random_lower']:+.3f}"),
            (tri(None if e["loco_delta"] is None else e["loco_delta"] > 0), "n/a" if e["loco_delta"] is None else f"{e['loco_delta']:+.3f}"),
            (tri(None if e["positions_positive"] is None else e["positions_positive"] >= 3),
             "n/a" if e["positions_positive"] is None else f"{e['positions_positive']}/{e['positions_total']}"),
            (tri(None if e["seeds_total"] is None else e["seeds_positive"] == e["seeds_total"]),
             "n/a" if e["seeds_total"] is None else f"{e['seeds_positive']}/{e['seeds_total']}"),
            (tri(None if e["perm_p_outcome"] is None else e["perm_p_outcome"] < perm_alpha), "n/a" if e["perm_p_outcome"] is None else f"p {e['perm_p_outcome']:.3f}"),
            (tri(None if e["matched_lower"] is None else e["matched_lower"] > 0),
             "n/a" if e["matched_d_concordance"] is None else f"{e['matched_d_concordance']:+.2f}")]
        rows.append(f"{e['dimension'].replace('_', ' ')}\n{e['model']} vs {e['reference']}")
        M.append([c[0] for c in cells])
        text.append([c[1] for c in cells])
    from matplotlib.colors import ListedColormap

    cm = ListedColormap(["#f3c4c0", "#bfe3c0"])
    cm.set_bad(INK["grid"])
    ax.imshow(np.ma.masked_invalid(np.array(M, dtype=float)), cmap=cm, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.grid(False)
    n = evidence["n_athletes"][0]
    finish(fig, f"Primary frozen target {evidence['target_id'][0]}, {n} athletes. Green: the check holds; red: it does not; grey: not "
           "applicable or not evaluated. First column: Δ RMSE against the dimension's reference with its 95 % interval. Repeatability at "
           "the Combine is not a column: it is not NFL validation. A dimension is 'predictive validated' only when its whole row is green.",
           "Strength of the transfer evidence, dimension by dimension")
    return fig
