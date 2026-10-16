"""Figures of notebooks 07A (context decomposition), 07B (physically aligned transfer) and 07C (detectability).

Every function returns a Matplotlib figure drawn ONLY from tables the notebook computed; the notebook passes it to the
store's ``show`` (inline display, PNG 300 DPI + SVG, PDF for publication figures, plot data as CSV). Conventions of
``transfer_figures``: 16:9 project style, one question per figure, the denominator (athletes) in the caption, units on
every axis, difference axes centred on an explicit zero line, 95 % athlete-bootstrap intervals unless stated.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS
from ..figkit import empty_panel, figure, finish
from .transfer_figures import _a, _forest, _heat, _legend, _pos_color, _whisk

EXPL = "EXPLORATORY — Milestone 07; the frozen Milestone 06 conclusion is unchanged."
DEF_COLOR = {"C0": INK["secondary"], "C1": CATEGORICAL[0], "C2": CATEGORICAL[1], "C3": CATEGORICAL[2], "C4": CATEGORICAL[6]}
COMP_COLOR = {"athlete": CATEGORICAL[0], "game_within_athlete": CATEGORICAL[1], "window_within_game": INK["axis"]}
COMP_NAME = {"athlete": "athlete", "game_within_athlete": "game within athlete", "window_within_game": "window within game"}


def _fit_line(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if x.size >= 5 and x.std() > 1e-12:
        b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, b[0] * xs + b[1], color=color, lw=1.4)


# ============================================================================== 07A
def plot_raw_distributions(adjusted: pl.DataFrame, windows: pl.DataFrame, unit: str, outcome_id: str) -> Any:
    fig, axes = figure(1, 2)
    a = adjusted.filter((pl.col("definition") == "C0") & pl.col("eligible"))
    pos = sorted(a["position_group"].unique().to_list())
    col = _pos_color(pos)
    data = [a.filter(pl.col("position_group") == p)["value"].to_numpy() for p in pos]
    if any(len(d) for d in data):
        bp = axes[0, 0].boxplot(data, patch_artist=True, widths=0.6)
        axes[0, 0].set_xticks(np.arange(1, len(pos) + 1))
        axes[0, 0].set_xticklabels([f"{p}\n(n = {len(d)})" for p, d in zip(pos, data)])
        for patch, p in zip(bp["boxes"], pos):
            patch.set(facecolor=col[p], alpha=0.55)
    else:
        empty_panel(axes[0, 0], "no eligible athlete")
    axes[0, 0].set_ylabel(f"athlete value [{unit}]")
    axes[0, 0].set_title("Athlete values (one point of evidence per athlete)", loc="left", fontsize=11)
    for p in pos:
        v = windows.filter(pl.col("position_group") == p)["y"].to_numpy()
        if v.size:
            axes[0, 1].hist(v, bins=60, histtype="step", density=True, color=col[p], lw=1.4, label=p)
    axes[0, 1].set_xlabel(f"window value [{unit}]")
    axes[0, 1].set_ylabel("density")
    axes[0, 1].set_title("Windows (nested in athletes — not independent)", loc="left", fontsize=11)
    axes[0, 1].legend(frameon=False, fontsize=8)
    finish(fig, f"{outcome_id}: {a.height} eligible athletes, {windows.height:,} windows. Left: the between-athlete spread a comparison rests on. "
                "Right: the window distribution each athlete value averages.", "Raw NFL outcome by position group")
    return fig


def plot_player_game_variation(games: pl.DataFrame, unit: str) -> Any:
    """``games``: nfl_id, position_group, game_id, mean, n (athlete-game means)."""
    fig, axes = figure(1, 2)
    g = games.with_columns((pl.col("mean") - pl.col("mean").mean().over("nfl_id")).alias("dev"))
    pos = sorted(g["position_group"].unique().to_list())
    col = _pos_color(pos)
    for p in pos:
        v = g.filter(pl.col("position_group") == p)["dev"].to_numpy()
        if v.size:
            axes[0, 0].hist(v, bins=50, histtype="step", density=True, color=col[p], lw=1.4, label=f"{p} (SD {v.std():.2f})")
    axes[0, 0].axvline(0, color=INK["secondary"], lw=1, ls="--")
    axes[0, 0].set_xlabel(f"game mean − athlete mean [{unit}]")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(frameon=False, fontsize=8)
    a = g.group_by("nfl_id", "position_group").agg(pl.col("mean").mean().alias("m"), pl.col("mean").std().alias("s"), pl.len().alias("G")).filter(pl.col("G") >= 4)
    for p in pos:
        k = a.filter(pl.col("position_group") == p)
        axes[0, 1].scatter(k["m"], k["s"], s=14, color=col[p], alpha=0.7, label=p)
    axes[0, 1].set_xlabel(f"athlete mean of game means [{unit}]")
    axes[0, 1].set_ylabel(f"SD of the athlete's game means [{unit}]")
    finish(fig, f"{g.select(pl.struct('nfl_id', 'game_id').n_unique()).item():,} athlete-games of {g['nfl_id'].n_unique()} athletes. Left: how far a single game "
                "sits from the athlete's own level. Right: athletes with ≥ 4 games.", "Game-to-game variation of an athlete")
    return fig


def plot_variance_components(variance: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    v = variance.filter(pl.col("component").is_in(list(COMP_COLOR)))
    scopes = ["ALL"] + sorted(s for s in v["scope"].unique().to_list() if s != "ALL")
    left = np.zeros(len(scopes))
    for comp in COMP_COLOR:
        share = np.array([(v.filter((pl.col("scope") == s) & (pl.col("component") == comp))["share_within_position"].to_list() or [np.nan])[0] or 0.0 for s in scopes])
        ax.barh(np.arange(len(scopes)), share, left=left, color=COMP_COLOR[comp], label=COMP_NAME[comp])
        for i, x in enumerate(share):
            if x > 0.04:
                ax.text(left[i] + x / 2, i, f"{x:.0%}", ha="center", va="center", fontsize=9, color="white")
        left += share
    n = {s: (v.filter(pl.col("scope") == s)["n_athletes"].to_list() or [0])[0] for s in scopes}
    ax.set_yticks(np.arange(len(scopes)))
    ax.set_yticklabels([f"{'all positions' if s == 'ALL' else s} (n = {n[s]})" for s in scopes])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the within-position window variance")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    pos = variance.filter(pl.col("component") == "position")
    finish(fig, "Nested method-of-moments components, pooled within position. A single window is mostly window-to-window variation; an athlete VALUE averages "
                "many windows, which is why it can be reliable although the athlete share of one window is small."
                + (f" Position alone carries {pos['share_total'][0]:.0%} of all window variance." if pos.height and pos["share_total"][0] is not None else ""),
           "What one NFL window is made of")
    return fig


def plot_context_association(variance: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    c = variance.filter(pl.col("component").str.starts_with("context:")).sort("share_within_position", descending=True)
    if not c.height:
        empty_panel(ax, "no context factor available")
    else:
        names = [x[8:].replace("_", " ") for x in c["component"]]
        ax.barh(np.arange(c.height), _a(c["share_within_position"]), color=CATEGORICAL[1])
        ax.set_yticks(np.arange(c.height))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        for i, x in enumerate(_a(c["share_within_position"])):
            ax.text(x, i, f" {x:.1%}", va="center", fontsize=9)
        ax.set_xlabel("marginal share of the within-position window variance (η²)")
    finish(fig, "Each factor alone, beyond position; window level, descriptive. The shares are NOT additive with each other or with the nested components.",
           "Observed context and the play measure")
    return fig


def plot_phase_space(phase: pl.DataFrame) -> Any:
    pos = sorted(phase["position_group"].unique().to_list())
    fig, axes = figure(1, max(len(pos), 1))
    for j, p in enumerate(pos):
        g = phase.filter(pl.col("position_group") == p)
        sx, ax_ = np.sort(g["speed_lo"].unique().to_numpy()), np.sort(g["accel_lo"].unique().to_numpy())
        M = np.zeros((ax_.size, sx.size))
        M[np.searchsorted(ax_, g["accel_lo"].to_numpy()), np.searchsorted(sx, g["speed_lo"].to_numpy())] = g["n_frames"].to_numpy()
        M = M / max(M.sum(), 1)
        axes[0, j].imshow(np.log10(np.where(M > 0, M, np.nan)), origin="lower", aspect="auto", cmap="Blues",
                          extent=[sx.min(), sx.max(), ax_.min(), ax_.max()])
        axes[0, j].axhline(0, color=INK["secondary"], lw=0.8, ls="--")
        axes[0, j].set_title(p, loc="left", fontsize=11)
        axes[0, j].set_xlabel("speed [yd/s]")
        axes[0, j].grid(False)
    if pos:
        axes[0, 0].set_ylabel("tangential acceleration [yd/s²]")
    finish(fig, "Share of post-snap frames by speed and tangential acceleration (log colour scale), Milestone 05 phase-space table. Where a position has no frames, "
                "an acceleration at that speed is not observed for it — an opportunity limit, not an ability.", "NFL speed–acceleration phase space")
    return fig


def plot_alignment_comparison(dist: pl.DataFrame, unit: str, top: int = 6) -> Any:
    d = dist.filter((pl.col("group_type") == "alignment") & (pl.col("n_athletes") >= 3))
    pos = sorted(d["position_group"].unique().to_list())
    fig, axes = figure(1, max(len(pos), 1))
    for j, p in enumerate(pos):
        g = d.filter(pl.col("position_group") == p).sort("n_windows", descending=True).head(top)
        y = np.arange(g.height)[::-1].astype(float)
        m, s = _a(g["athlete_mean"]), np.nan_to_num(_a(g["athlete_sd"]))
        _whisk(axes[0, j], m, y, m - s, m + s, [CATEGORICAL[j % len(CATEGORICAL)]] * g.height)
        axes[0, j].set_yticks(y)
        axes[0, j].set_yticklabels([f"{a} (n = {n})" for a, n in zip(g["group"], g["n_athletes"])], fontsize=8)
        axes[0, j].set_title(p, loc="left", fontsize=11)
        axes[0, j].set_xlabel(f"[{unit}]")
    finish(fig, "Mean ± SD of athlete means by lined-up position (the most frequent alignments; n = athletes with ≥ 5 windows there). Alignment is observed, "
                "assignment is not.", "The same position group, different alignments")
    return fig


def plot_start_speed_support(dist: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1)
    d = dist.filter(pl.col("group_type") == "start_band")
    rows, cols = sorted(d["position_group"].unique().to_list()), d["group"].unique().to_list()
    cols = sorted(cols, key=lambda s: (s == "NA", s.startswith("≥"), s))
    if not rows or not cols:
        empty_panel(axes[0, 0], "the play measure has no starting-speed field")
    else:
        get = {(r["position_group"], r["group"]): r for r in d.iter_rows(named=True)}
        tot = {p: sum(get[(p, c)]["n_windows"] for c in cols if (p, c) in get) for p in rows}
        M = np.array([[get[(p, c)]["n_windows"] / max(tot[p], 1) if (p, c) in get else np.nan for c in cols] for p in rows])
        txt = [[f"{M[i, j]:.0%}\n{get[(p, c)]['n_athletes']} ath." if (p, c) in get else "—" for j, c in enumerate(cols)] for i, p in enumerate(rows)]
        _heat(axes[0, 0], M, rows, [f"{c} yd/s" for c in cols], txt, vmax=float(np.nanmax(M)))
    finish(fig, "Share of a position's windows in each starting-speed band of the measured 0.5 s window, with the athletes having ≥ 5 windows there. "
                "A band that is rare for a position cannot support a comparison in it.", "Starting-speed support by position")
    return fig


def plot_raw_vs_adjusted(adjusted: pl.DataFrame, b: str, unit: str, a: str = "C0") -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    x = adjusted.filter((pl.col("definition") == a) & pl.col("eligible")).select("nfl_id", "position_group", pl.col("rank_in_position").alias("x"))
    y = adjusted.filter((pl.col("definition") == b) & pl.col("eligible")).select("nfl_id", pl.col("rank_in_position").alias("y"))
    j = x.join(y, on="nfl_id").drop_nulls()
    col = _pos_color(j["position_group"].unique().to_list())
    for p, c in col.items():
        g = j.filter(pl.col("position_group") == p)
        ax.scatter(g["x"], g["y"], s=16, color=c, alpha=0.75, label=f"{p} (n = {g.height})")
    ax.plot([0, 1], [0, 1], color=INK["secondary"], lw=1, ls="--")
    ax.set_xlabel(f"rank within position, {a} (0 = lowest)")
    ax.set_ylabel(f"rank within position, {b}")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, f"{j.height} athletes eligible under both definitions; ranks among eligible athletes of the position. Distance from the diagonal = how much the "
                f"context definition changes an athlete's standing. Values in {unit}. {EXPL}", f"Raw ({a}) against context-conditioned ({b}) athlete standing")
    return fig


def plot_overlap_retention(overlap: pl.DataFrame) -> Any:
    fig, axes = figure(1, 2)
    o = overlap.filter(pl.col("status") != "unavailable")
    x = np.arange(o.height)
    cols = [DEF_COLOR.get(d, INK["secondary"]) for d in o["definition"]]
    axes[0, 0].bar(x, o["n_athletes_eligible"], color=cols)
    for i, r in enumerate(o.iter_rows(named=True)):
        axes[0, 0].text(i, r["n_athletes_eligible"], f"{r['n_athletes_eligible']}\n{r['n_positions']} pos.", ha="center", va="bottom", fontsize=8)
    axes[0, 0].set_ylabel("eligible athletes")
    axes[0, 1].bar(x, _a(o["share_windows_retained"]), color=cols)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("share of windows retained")
    for ax in axes[0]:
        ax.set_xticks(x)
        ax.set_xticklabels(o["definition"].to_list())
    finish(fig, "Athletes meeting the exposure requirement and windows kept under each definition; 'pos.' = position groups with enough athletes. "
                "A definition that looks better on fewer athletes is not automatically better.", "Context overlap against cohort retention")
    return fig


def plot_reliability_complexity(tradeoff: pl.DataFrame, ref_line: float) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    t = tradeoff.sort("complexity")
    if not t.height:
        empty_panel(ax, "no definition evaluated")
    else:
        x = t["complexity"].to_numpy().astype(float)
        r = _a(t["reliability"])
        _whisk(ax, x, r, _a(t["ci_lower"]), _a(t["ci_upper"]), [DEF_COLOR.get(d, INK["secondary"]) for d in t["definition"]], horizontal=False)
        for xi, ri, row in zip(x, r, t.iter_rows(named=True)):
            ax.annotate(f"{row['definition']}\nn = {row['n_athletes_eligible']} ({row['share_athletes_retained']:.0%})", (xi, ri), textcoords="offset points",
                        xytext=(8, -4), fontsize=8)
        ax.axhline(ref_line, color=INK["secondary"], lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_ylim(min(0.0, np.nanmin(r) - 0.05), 1.0)
    ax.set_xlabel("context complexity (0 = raw aggregate … 4 = hierarchical)")
    ax.set_ylabel("out-of-game reliability (Spearman–Brown)")
    finish(fig, "Random game half-splits, context parameters cross-fitted on the other half; whiskers: 95 % athlete bootstrap of the odd / even split. "
                f"Dashed: reference line {ref_line} (heuristic). {EXPL}", "Reliability against context complexity")
    return fig


def plot_reliability_exposure(reliability: pl.DataFrame, kind: str = "first_games") -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    r = reliability.filter(pl.col("method").str.starts_with(kind + ":")).with_columns(pl.col("method").str.split(":").list.get(1).cast(pl.Int64).alias("N"))
    if not r.height:
        empty_panel(ax, "no exposure row with enough athletes")
    for d in sorted(r["definition"].unique().to_list()):
        g = r.filter(pl.col("definition") == d).sort("N")
        ax.plot(g["N"], g["reliability"], marker="o", color=DEF_COLOR.get(d, INK["secondary"]), label=d)
        ax.fill_between(g["N"].to_numpy(), _a(g["ci_lower"]), _a(g["ci_upper"]), color=DEF_COLOR.get(d, INK["secondary"]), alpha=0.12)
        for n, y, k in zip(g["N"], g["reliability"], g["n_athletes"]):
            ax.annotate(f"n={k}", (n, y), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center", color=INK["muted"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"first N {'games' if kind == 'first_games' else 'windows'}")
    ax.set_ylabel("correlation with the athlete's later games (within position)")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "Value from the first N against the value from later games, each adjusted with parameters from the other part. The cohort changes with N (n above the points).",
           "How much NFL history an athlete value needs")
    return fig


def plot_rank_shift(examples: pl.DataFrame, a: str, b: str, names: dict[int, str] | None = None) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    if not examples.height:
        empty_panel(ax, "no athlete eligible under both definitions")
    for i, r in enumerate(examples.iter_rows(named=True)):
        c = STATUS["good"] if r["rank_shift"] > 0 else STATUS["critical"]
        ax.plot([0, 1], [r[f"rank_{a}"], r[f"rank_{b}"]], color=c, marker="o", lw=1.6)
        ax.text(1.02, r[f"rank_{b}"], f"{(names or {}).get(r['nfl_id'], r['nfl_id'])} · {r['position_group']} · {r['n_windows']} windows", fontsize=8, va="center")
    ax.set_xlim(-0.1, 1.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([a, b])
    ax.set_ylabel("rank within position (0 = lowest, 1 = highest)")
    finish(fig, f"The {examples.height} athletes whose within-position rank moves most between {a} and {b}. An illustration that the definitions disagree about "
                f"these athletes — not evidence about which one is right. {EXPL}", "Athletes re-ranked by the context adjustment")
    return fig


# ============================================================================== 07B
def plot_feature_vs_outcome(x: np.ndarray, y: np.ndarray, position: np.ndarray, xlabel: str, ylabel: str, title: str, note: str = "") -> Any:
    fig, axes = figure(1, 2)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, position = x[ok], y[ok], position[ok]
    col = _pos_color(np.unique(position).tolist())
    xr, yr = x.astype(float).copy(), y.astype(float).copy()
    for p, c in col.items():
        k = position == p
        axes[0, 0].scatter(x[k], y[k], s=16, color=c, alpha=0.7, label=f"{p} (n = {int(k.sum())})")
        _fit_line(axes[0, 0], x[k], y[k], c)
        xr[k] -= x[k].mean()
        yr[k] -= y[k].mean()
        axes[0, 1].scatter(xr[k], yr[k], s=16, color=c, alpha=0.7)
    _fit_line(axes[0, 1], xr, yr, INK["primary"])
    r = float(np.corrcoef(xr, yr)[0, 1]) if x.size > 2 and xr.std() > 0 and yr.std() > 0 else np.nan
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_title("As measured", loc="left", fontsize=11)
    axes[0, 1].set_xlabel(xlabel + " − position mean")
    axes[0, 1].set_ylabel(ylabel + " − position mean")
    axes[0, 1].axhline(0, color=INK["grid"], lw=0.8)
    axes[0, 1].set_title(f"Within position (r = {r:+.2f})", loc="left", fontsize=11)
    finish(fig, f"{x.size} athletes. Left: position differences dominate. Right: position means removed — the association a scout comparing two athletes of the "
                f"same position would use. In-sample and descriptive. {note} {EXPL}", title)
    return fig


def plot_associations(assoc: pl.DataFrame, title: str) -> Any:
    fig, axes = figure(1, 1)
    a = assoc.sort("feature", "target_id", "control")
    labels = [f"{r['feature']} ↔ {r['target_id']} | {r['control']}" for r in a.iter_rows(named=True)]
    cols = [CATEGORICAL[0] if c == "position" else CATEGORICAL[6] for c in a["control"]]
    _forest(axes[0, 0], labels, _a(a["r"]), _a(a["ci_lower"]), _a(a["ci_upper"]), cols, "partial correlation (within position)",
            [f"n = {n} · expected {s}" for n, s in zip(a["n_athletes"], a["expected_sign"])])
    _legend(axes[0, 0], {"position removed": CATEGORICAL[0], "position + official times + weight removed": CATEGORICAL[6]}, loc="lower right")
    finish(fig, "Correlation of residuals, athlete bootstrap that refits the removal. In-sample, descriptive: not a transfer result. "
                f"A time (s) is expected to correlate NEGATIVELY with an acceleration. {EXPL}", title)
    return fig


def plot_paired_forest(paired: pl.DataFrame, title: str, caption: str, label: Sequence[str] = ("target_id", "model", "reference"), relative: bool = False) -> Any:
    """Forest of Δ RMSE rows (``paired`` already filtered and ordered by the notebook)."""
    fig, axes = figure(1, 1, size="wide" if paired.height <= 14 else "tall")
    p = paired.filter(pl.col("metric") == "delta_rmse") if "metric" in paired.columns else paired
    est = _a(p["estimate"] if "estimate" in p.columns else p["delta_rmse"])
    lo, hi = _a(p["ci_lower"]), _a(p["ci_upper"])
    if relative:                                         # same rows on the scale "% of the reference error"
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(np.abs(est) > 0, _a(p["relative"]) * 100.0 / est, np.nan)
        est, lo, hi = est * k, lo * k, hi * k
    labels = [" · ".join(str(r[c]) for c in label if c in r) for r in p.iter_rows(named=True)]
    cols = [STATUS["good"] if e else STATUS["critical"] if w else INK["secondary"]
            for e, w in zip(p["established"], p["worse"] if "worse" in p.columns else [False] * p.height)]
    _forest(axes[0, 0], labels, est, lo, hi, cols,
            "Δ RMSE, % of the reference error" if relative else "Δ RMSE = RMSE(reference) − RMSE(model)  (positive = model better)",
            [f"n = {n}" for n in p["n_athletes"]])
    _legend(axes[0, 0], {"interval above 0": STATUS["good"], "interval includes 0": INK["secondary"], "interval below 0": STATUS["critical"]}, loc="lower right")
    finish(fig, caption + " Whiskers: 95 % paired athlete bootstrap on out-of-fold predictions. " + EXPL, title)
    return fig


def plot_matched(matched: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1, size="wide" if matched.height <= 14 else "tall")
    m = matched.filter(pl.col("delta_vs_reference").is_not_null()).sort("target_id", "model")
    labels = [f"{r['target_id']} · {r['model']}" for r in m.iter_rows(named=True)]
    _forest(axes[0, 0], labels, _a(m["delta_vs_reference"]), _a(m["delta_ci_lower"]), _a(m["delta_ci_upper"]), [CATEGORICAL[6]] * m.height,
            "concordance − concordance of the official model M1", [f"{c:.3f} · {n} pairs" for c, n in zip(m["concordance"], m["n_pairs"])])
    finish(fig, "Matched prospects of 06C (same position, near-identical official forty, 10-yard split and body size; rule fixed on pre-NFL information). "
                "Concordance = share of pairs whose predicted NFL difference has the observed sign; athlete-cluster bootstrap. " + EXPL,
           "Do models order the NFL movement of matched prospects?")
    return fig


# ============================================================================== 07C
def _scen(scen: pl.DataFrame, **eq: Any) -> pl.DataFrame:
    for k, v in eq.items():
        scen = scen.filter(pl.col(k) == v)
    return scen


def plot_power_curve(scen: pl.DataFrame, target_power: float) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    s = _scen(scen, study="power", scope="ALL")
    for i, ((c, m), g) in enumerate(sorted(s.partition_by(["candidate", "mode"], as_dict=True).items())):
        g = g.sort("effect")
        col = CATEGORICAL[(0 if c == "M5" else 2) + (0 if m == "conditional" else 1)]
        ax.plot(g["effect"], g["power"], marker="o", color=col, ls="-" if m == "conditional" else "--", label=f"{c} · {m}")
        ax.fill_between(g["effect"].to_numpy(), _a(g["power_lower"]), _a(g["power_upper"]), color=col, alpha=0.12)
    ax.axhline(target_power, color=INK["secondary"], lw=1, ls=":")
    ax.axhline(0.025, color=INK["grid"], lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("planted effect f = share of the residual outcome variance")
    ax.set_ylabel("detection rate (95 % interval of Δ RMSE above 0)")
    ax.legend(frameon=False, fontsize=8)
    n = s["n_athletes"][0] if s.height else 0
    finish(fig, f"Synthetic targets on the real cohort ({n} athletes, real folds and representations), Milestone 06 pipeline and criterion. Bands: Monte-Carlo 95 % "
                "(Wilson). 'conditional' = signal orthogonal to official tests; 'raw' = the plain representation direction, partly redundant with them.",
           "Power to detect a planted incremental effect")
    return fig


def plot_mde(mde: pl.DataFrame, observed: dict[str, float | None]) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    m = _scen(mde, study="power", scope="ALL").sort("candidate", "mode")
    if not m.height:
        empty_panel(ax, "no minimum detectable effect available")
    else:
        x = np.arange(m.height)
        v = _a(m["mde_delta_rmse"])
        ax.bar(x, np.nan_to_num(v), color=CATEGORICAL[0])
        for i, r in enumerate(m.iter_rows(named=True)):
            ax.text(i, np.nan_to_num(v[i]), "not reached\nin the grid" if r["mde_effect"] is None else f"f = {r['mde_effect']:.3f}\n{100 * (r['mde_relative'] or 0):.1f} %",
                    ha="center", va="bottom", fontsize=8)
            ob = observed.get(r["candidate"])
            if ob is not None:
                ax.plot([i - 0.4, i + 0.4], [ob, ob], color=STATUS["critical"], lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n{md}" for c, md in zip(m["candidate"], m["mode"])])
    ax.set_ylabel("Δ RMSE at the target power [target units]")
    finish(fig, f"Smallest gain detected with {m['target_power'][0] if m.height else 0:.0%} power (interpolated power curve). Red line: the gain observed in Milestone 06. "
                "An observed gain below the bar is inconclusive, not a demonstrated absence.", "Minimum detectable improvement")
    return fig


def plot_null(null: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    n = null.filter(pl.col("scope") == "ALL").sort("study", "candidate", "n_athletes")
    labels = [f"{r['study']} · {r['candidate']} · {r['mode']} · n = {r['n_athletes']}" for r in n.iter_rows(named=True)]
    y = np.arange(n.height)[::-1].astype(float)
    if n.height:
        _whisk(ax, _a(n["false_positive_rate"]), y, _a(n["fpr_lower"]), _a(n["fpr_upper"]), [CATEGORICAL[0]] * n.height)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        empty_panel(ax, "no null scenario")
    ax.axvline(0.025, color=INK["secondary"], lw=1, ls="--")
    ax.set_xlim(0, max(0.15, float(np.nanmax(_a(n["fpr_upper"]))) * 1.1 if n.height else 0.15))
    ax.set_xlabel("false-positive rate with no planted effect")
    finish(fig, "Share of no-signal replicates in which the interval of Δ RMSE lies above 0. Dashed: 0.025, the one-sided nominal rate of a 95 % interval. "
                "Whiskers: Monte-Carlo 95 % (Wilson).", "Does the pipeline invent gains?")
    return fig


def plot_coverage(scen: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    s = _scen(scen, study="power", scope="ALL")
    for i, ((c, m), g) in enumerate(sorted(s.partition_by(["candidate", "mode"], as_dict=True).items())):
        g = g.sort("effect")
        ax.plot(g["effect"], g["ci_coverage"], marker="o", color=CATEGORICAL[i % len(CATEGORICAL)], label=f"{c} · {m}")
    ax.axhline(0.95, color=INK["secondary"], lw=1, ls="--")
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("planted effect f")
    ax.set_ylabel("share of replicate intervals containing the scenario mean Δ RMSE")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "The paired athlete bootstrap does not refit the models: its interval reflects which athletes were sampled. Coverage below 0.95 shows how much "
                "the unmodelled refitting variability matters.", "Coverage of the paired bootstrap interval")
    return fig


def plot_power_by_position(scen: pl.DataFrame, candidate: str) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    s = _scen(scen, study="power", candidate=candidate, mode="conditional")
    scopes = ["ALL"] + sorted(x for x in s["scope"].unique().to_list() if x != "ALL")
    col = {**_pos_color(scopes[1:]), "ALL": INK["primary"]}
    for sc in scopes:
        g = s.filter(pl.col("scope") == sc).sort("effect")
        if g.height:
            ax.plot(g["effect"], g["power"], marker="o", color=col[sc], lw=2.2 if sc == "ALL" else 1.3, label=f"{sc} (n ≈ {g['n_scope'][0]:.0f})")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("planted effect f")
    ax.set_ylabel("detection rate")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, f"{candidate}, signal orthogonal to official tests, the same replicates scored inside each position group. With the athletes of one position, "
                "only very large effects are detectable: a position-level 'finding' from this cohort would mostly be noise.", "Power by position group")
    return fig


def plot_attenuation(grid: pl.DataFrame, marks: dict[str, tuple[float, float]]) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    r0 = grid["r_true"].max()
    g = grid.filter(pl.col("r_true") == r0)
    for i, rx in enumerate(sorted(g["reliability_x"].unique().to_list())):
        k = g.filter(pl.col("reliability_x") == rx).sort("reliability_y")
        ax.plot(k["reliability_y"], k["share_of_r2_kept"], color=SEQUENTIAL_BLUE[min(i + 1, len(SEQUENTIAL_BLUE) - 1)], marker="o", label=f"Combine reliability {rx:.2f}")
    for j, (name, (rx, ry)) in enumerate(marks.items()):
        ax.scatter([ry], [rx * ry], s=90, color=CATEGORICAL[(j + 5) % len(CATEGORICAL)], zorder=5, label=f"{name}: {rx * ry:.2f}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("reliability of the NFL outcome")
    ax.set_ylabel("share of a true R² that survives (rel_x · rel_y)")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "Classical errors-in-variables approximation for a correlation, with the reliabilities measured in this project marked. It is NOT a ceiling on prediction: "
                "a model is scored against the observed outcome, and reliable context variance is not removed by a reliability coefficient.",
           "How much measurement error could hide")
    return fig


def plot_sample_size(scen: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    s = _scen(scen, study="sample_size", scope="ALL")
    for i, e in enumerate(sorted(s["effect"].unique().to_list())):
        g = s.filter(pl.col("effect") == e).sort("n_athletes")
        ax.plot(g["n_athletes"], g["power"], marker="o", color=CATEGORICAL[i % len(CATEGORICAL)], label=f"f = {e:g}" + (" (null)" if e == 0 else ""))
        ax.fill_between(g["n_athletes"].to_numpy(), _a(g["power_lower"]), _a(g["power_upper"]), color=CATEGORICAL[i % len(CATEGORICAL)], alpha=0.10)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("linked athletes (scenario)")
    ax.set_ylabel("detection rate")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "Cohorts resampled with replacement from the observed athletes, folds re-dealt, one fold-independent representation. Scenario analysis of sampling "
                "precision — not a forecast of data that will exist.", "Linked-cohort size against detectable effect")
    return fig


def plot_redundancy(redundancy: pl.DataFrame, scen: pl.DataFrame) -> Any:
    fig, axes = figure(1, 2)
    r = redundancy.filter(pl.col("component") == "direction_1").sort("r2_official")
    axes[0, 0].barh(np.arange(r.height), _a(r["r2_official"]), color=CATEGORICAL[0], label="explained by official design")
    axes[0, 0].barh(np.arange(r.height), _a(r["r2_position"]), color=INK["axis"], height=0.35, label="by position alone")
    axes[0, 0].set_yticks(np.arange(r.height))
    axes[0, 0].set_yticklabels(r["family"].to_list())
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_xlabel("R² of the representation's leading direction")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    s = _scen(scen, study="power", scope="ALL")
    for i, ((c, m), g) in enumerate(sorted(s.partition_by(["candidate", "mode"], as_dict=True).items())):
        g = g.sort("effect")
        axes[0, 1].plot(g["effect"], g["mean_delta_rmse"], marker="o", ls="-" if m == "conditional" else "--", color=CATEGORICAL[i % len(CATEGORICAL)], label=f"{c} · {m}")
    axes[0, 1].axhline(0, color=INK["secondary"], lw=1, ls="--")
    axes[0, 1].set_xlabel("planted effect f")
    axes[0, 1].set_ylabel("mean recovered Δ RMSE")
    axes[0, 1].legend(frameon=False, fontsize=8)
    finish(fig, "Left: how much of each representation official testing already explains. Right: the same planted share of variance recovers a smaller incremental "
                "gain when it lies along the raw (redundant) direction than when it is orthogonal to official tests.",
           "A repeatable representation can add little once official tests are in")
    return fig


def plot_observed_vs_detectable(reps: pl.DataFrame, scen: pl.DataFrame, candidate: str, observed: float | None, lo: float | None, hi: float | None) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    k = (pl.col("study") == "power") & (pl.col("candidate") == candidate) & (pl.col("mode") == "conditional") & (pl.col("scope") == "ALL")
    null = reps.filter(k & (pl.col("effect") == 0.0))["delta_rmse"].to_numpy()
    if null.size:
        ax.hist(null, bins=30, color=INK["axis"], alpha=0.7, label=f"no signal ({null.size} replicates)")
    s = scen.filter(k & (pl.col("effect") > 0)).sort("effect")
    for i, r in enumerate(s.iter_rows(named=True)):
        ax.axvline(r["mean_delta_rmse"], color=SEQUENTIAL_BLUE[min(i + 1, len(SEQUENTIAL_BLUE) - 1)], lw=1.6, label=f"f = {r['effect']:g}: power {r['power']:.2f}")
    if observed is not None:
        ax.axvline(observed, color=STATUS["critical"], lw=2.4, label="observed in Milestone 06")
        if lo is not None and hi is not None:
            ax.axvspan(lo, hi, color=STATUS["critical"], alpha=0.08)
    ax.set_xlabel("Δ RMSE = RMSE(official) − RMSE(candidate)")
    ax.set_ylabel("replicates")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, f"{candidate}. Grey: distribution of the paired gain when nothing is planted. Blue lines: mean gain recovered for each planted effect, with its "
                "detection rate. Red: the real Milestone 06 gain and its interval. Where the red line sits among the blue lines says which effects the study could "
                "and could not have distinguished from none.", "The observed Milestone 06 gain against detectable effects")
    return fig
