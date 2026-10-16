"""Milestone 11 figures (matplotlib through ``figkit``: 16:9, captions that state the denominator, PNG 300 DPI + SVG, PDF
for major figures via ``store.show(..., publication=True)``).

Conventions: categorical colours in the project's fixed order (``CATEGORICAL``; positions always map to the same colour
via ``POSITION_COLORS``), one y-axis per panel (never dual axes), status colours only for evidence classes and always
with a text label, a zero / chance reference line on every effect plot, explicit "no data" panels.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl
from matplotlib.patches import FancyBboxPatch, Rectangle

from ....data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS
from ...figkit import empty_panel, figure, finish
from .config import (ESTABLISHED_OVERLAP, EXPLORATORY_RELATION, INSUFFICIENT_DATA, LIMITED, MATRIX_CLASSES, NO_ESTABLISHED_TRANSFER, NOT_TESTED,
                     SUPPORTED, SUPPORTED_TRANSFER, UNKNOWN, UNSUPPORTED)

POSITIONS = ("DB", "DL", "OL", "TE", "WR", "LB", "RB", "QB", "ST", "ALL")
POSITION_COLORS = {p: CATEGORICAL[i % len(CATEGORICAL)] for i, p in enumerate(POSITIONS)}
DOMAIN_COLORS = {"combine": CATEGORICAL[0], "nfl": CATEGORICAL[1]}
CLASS_COLORS = {SUPPORTED_TRANSFER: STATUS["good"], EXPLORATORY_RELATION: STATUS["warning"], NO_ESTABLISHED_TRANSFER: STATUS["serious"],
                ESTABLISHED_OVERLAP: SEQUENTIAL_BLUE[2], INSUFFICIENT_DATA: INK["axis"], NOT_TESTED: INK["grid"]}
CLASS_LABELS = {SUPPORTED_TRANSFER: "supported held-out transfer", EXPLORATORY_RELATION: "exploratory relationship", NO_ESTABLISHED_TRANSFER: "no established transfer",
                ESTABLISHED_OVERLAP: "physical overlap only", INSUFFICIENT_DATA: "insufficient data", NOT_TESTED: "not tested (no overlap)"}
FEAS_COLORS = {SUPPORTED: STATUS["good"], LIMITED: STATUS["warning"], UNSUPPORTED: INK["axis"], UNKNOWN: INK["grid"]}


def _pos_color(p: Any) -> str:
    return POSITION_COLORS.get(str(p), INK["secondary"])


# ============================================================================== generic
def fig_heatmap(df: pl.DataFrame, row: str, col: str, value: str, title: str, caption: str, fmt: str = "{:.0f}", cmap: str = "Blues", annotate: bool = True):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if df.is_empty():
        empty_panel(ax, "no data")
        finish(fig, caption, title)
        return fig
    rows, cols = sorted(df[row].cast(pl.String).unique().to_list()), sorted(df[col].cast(pl.String).unique().to_list())
    M = np.full((len(rows), len(cols)), np.nan)
    for r in df.iter_rows(named=True):
        v = r[value]
        if v is not None:
            M[rows.index(str(r[row])), cols.index(str(r[col]))] = float(v)
    im = ax.imshow(M, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    if annotate and M.size <= 400:
        hi = np.nanmax(M) if np.isfinite(M).any() else 1.0
        for i in range(len(rows)):
            for j in range(len(cols)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.6 * hi else INK["primary"])
    fig.colorbar(im, ax=ax, shrink=0.8, label=value)
    ax.grid(False)
    finish(fig, caption, title)
    return fig


def fig_state_grid(df: pl.DataFrame, row: str, col: str, state: str, colors: dict[str, str], title: str, caption: str, label: str | None = None):
    """Categorical grid (feasibility state / evidence class) — every cell carries its text label, never colour alone."""
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if df.is_empty():
        empty_panel(ax, "no data")
        finish(fig, caption, title)
        return fig
    rows, cols = sorted(df[row].cast(pl.String).unique().to_list()), sorted(df[col].cast(pl.String).unique().to_list())
    for r in df.iter_rows(named=True):
        i, j = rows.index(str(r[row])), cols.index(str(r[col]))
        s = str(r[state])
        ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, color=colors.get(s, INK["grid"]), lw=0))
        txt = CLASS_LABELS.get(s, s).replace(" ", "\n", 1) if label is None else str(r[label])
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=INK["primary"])
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    finish(fig, caption, title)
    return fig


def fig_funnel(df: pl.DataFrame, title: str, caption: str):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    y = np.arange(df.height)
    ax.barh(y, df["n_athletes"].to_numpy(), color=SEQUENTIAL_BLUE[4], height=0.6)
    for i, (s, n) in enumerate(zip(df["stage"].to_list(), df["n_athletes"].to_list())):
        ax.text(n, i, f"  {n:,}", va="center", fontsize=9, color=INK["primary"])
    ax.set_yticks(y, df["stage"].to_list())
    ax.invert_yaxis()
    ax.set_xlabel("athletes (independent unit)")
    finish(fig, caption, title)
    return fig


def fig_grouped_bars(df: pl.DataFrame, cat: str, group: str, value: str, title: str, caption: str, ylabel: str):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if df.is_empty():
        empty_panel(ax, "no data")
        finish(fig, caption, title)
        return fig
    cats = sorted(df[cat].cast(pl.String).unique().to_list())
    groups = [g for g in POSITIONS if g in set(df[group].cast(pl.String).to_list())] + sorted(set(df[group].cast(pl.String).to_list()) - set(POSITIONS))
    w = 0.8 / max(1, len(groups))
    for k, g in enumerate(groups):
        sub = {str(r[cat]): r[value] for r in df.filter(pl.col(group).cast(pl.String) == g).iter_rows(named=True)}
        ax.bar(np.arange(len(cats)) + k * w - 0.4 + w / 2, [sub.get(c, 0) or 0 for c in cats], width=w * 0.92, color=_pos_color(g), label=g)
    ax.set_xticks(range(len(cats)), cats, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(ncols=min(6, len(groups)), frameon=False, fontsize=8)
    finish(fig, caption, title)
    return fig


def fig_distributions(long: pl.DataFrame, feature_col: str, value_col: str, group_col: str, features: Sequence[str], title: str, caption: str,
                      units: dict[str, str] | None = None):
    n = max(1, len(features))
    ncol = min(3, n)
    fig, axes = figure(int(np.ceil(n / ncol)), ncol, size="wide")
    for k, f in enumerate(features):
        ax = axes.flat[k]
        d = long.filter(pl.col(feature_col) == f)
        groups = sorted(d[group_col].drop_nulls().cast(pl.String).unique().to_list())
        data = [d.filter(pl.col(group_col).cast(pl.String) == g)[value_col].drop_nulls().to_numpy() for g in groups]
        data = [(g, x[np.isfinite(x)]) for g, x in zip(groups, data) if np.isfinite(x).sum() >= 3]
        if not data:
            empty_panel(ax, f"{f}: too few values")
            continue
        bp = ax.boxplot([x for _, x in data], showfliers=False, patch_artist=True, widths=0.55)
        ax.set_xticks(range(1, len(data) + 1), [f"{g}\n(n={x.size})" for g, x in data])
        for patch, (g, _) in zip(bp["boxes"], data):
            patch.set_facecolor(_pos_color(g))
            patch.set_alpha(0.55)
        ax.set_title(f, fontsize=10)
        ax.set_ylabel((units or {}).get(f, ""))
        ax.tick_params(axis="x", labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    finish(fig, caption, title)
    return fig


def fig_forest(rows: pl.DataFrame, label: str, est: str, lo: str, hi: str, title: str, caption: str, xlabel: str, ref: float = 0.0, color_by: str | None = None):
    fig, ax = figure(1, 1, size=(12.8, max(4.0, 0.32 * rows.height + 1.8)))
    ax = ax[0, 0]
    if rows.is_empty():
        empty_panel(ax, "no comparison computed")
        finish(fig, caption, title)
        return fig
    y = np.arange(rows.height)
    cols = [CATEGORICAL[0]] * rows.height if color_by is None else [_pos_color(v) if str(v) in POSITION_COLORS else CATEGORICAL[i % 8]
                                                                    for i, v in enumerate(rows[color_by].cast(pl.String).to_list())]
    for i in range(rows.height):
        e, l, h = (rows[k][i] for k in (est, lo, hi))
        e = np.nan if e is None else float(e)
        l = np.nan if l is None else float(l)
        h = np.nan if h is None else float(h)
        if np.isfinite(l) and np.isfinite(h):
            ax.plot([l, h], [i, i], color=cols[i], lw=2)
        ax.plot([e], [i], "o", color=cols[i], ms=7, mec=INK["surface"], mew=1.5)
    ax.axvline(ref, color=INK["secondary"], lw=1, ls="--")
    ax.set_yticks(y, rows[label].cast(pl.String).to_list(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    finish(fig, caption, title)
    return fig


def fig_null_vs_observed(nulls: dict[str, np.ndarray], observed: dict[str, float], title: str, caption: str, xlabel: str):
    n = max(1, len(nulls))
    fig, axes = figure(1, min(4, n), size="wide")
    for k, (name, arr) in enumerate(list(nulls.items())[:4]):
        ax = axes.flat[k]
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if not arr.size:
            empty_panel(ax, f"{name}: no control run")
            continue
        ax.hist(arr, bins=25, color=INK["axis"])
        ob = observed.get(name, np.nan)
        if np.isfinite(ob):
            ax.axvline(ob, color=CATEGORICAL[1], lw=2, label="observed")
            ax.legend(frameon=False, fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(xlabel)
    finish(fig, caption, title)
    return fig


def fig_scatter(df: pl.DataFrame, x: str, y: str, title: str, caption: str, xlabel: str, ylabel: str, color: str = "position_group", ref_line: str | None = None):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    d = df.select(x, y, color).drop_nulls() if color in df.columns else df.select(x, y).drop_nulls().with_columns(pl.lit("ALL").alias(color))
    if d.is_empty():
        empty_panel(ax, "no athletes with both values")
        finish(fig, caption, title)
        return fig
    for j, g in enumerate(sorted(d[color].cast(pl.String).unique().to_list())):
        s = d.filter(pl.col(color).cast(pl.String) == g)
        col = POSITION_COLORS[g] if g in POSITION_COLORS else CATEGORICAL[j % len(CATEGORICAL)]
        ax.scatter(s[x], s[y], s=22, color=col, alpha=0.75, label=f"{g} (n={s.height})", edgecolors=INK["surface"], linewidths=0.6)
    if ref_line == "zero":
        ax.axhline(0, color=INK["axis"], lw=1)
        ax.axvline(0, color=INK["axis"], lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    finish(fig, caption, title)
    return fig


def fig_traces(traces: Sequence[dict[str, Any]], title: str, caption: str, spans: bool = True):
    """Physical example trajectories: speed (top) and a_tan (bottom) against time; optional shaded phase spans
    (``phases``: list of (t0, t1, name))."""
    n = max(1, len(traces))
    fig, axes = figure(2, n, size="wide", sharex="col")
    phase_col = {"braking": STATUS["serious"], "turning": SEQUENTIAL_BLUE[2], "reacceleration": STATUS["good"], "approach": INK["grid"]}
    for k, tr in enumerate(traces):
        a0, a1 = axes[0, k], axes[1, k]
        t = np.asarray(tr["t"], float)
        a0.plot(t, tr["speed"], color=CATEGORICAL[0], lw=2)
        if "fit" in tr:
            a0.plot(t, tr["fit"], color=CATEGORICAL[1], lw=1.5, ls="--", label=tr.get("fit_label", "model"))
            a0.legend(frameon=False, fontsize=7)
        a1.plot(t, tr["a_tan"], color=CATEGORICAL[2], lw=2)
        a1.axhline(0, color=INK["axis"], lw=1)
        if spans:
            for t0, t1, name in tr.get("phases", ()):
                for ax in (a0, a1):
                    ax.axvspan(t0, t1, color=phase_col.get(name, INK["grid"]), alpha=0.25, lw=0)
        a0.set_title(tr.get("title", ""), fontsize=9)
        a0.set_ylabel("speed [yd/s]")
        a1.set_ylabel("a_tan [yd/s²]")
        a1.set_xlabel("time [s]")
    finish(fig, caption, title)
    return fig


def fig_paths(paths: Sequence[dict[str, Any]], title: str, caption: str):
    """Top-down path (x, y) coloured by speed, one panel per attempt / episode (displacements only)."""
    n = max(1, len(paths))
    fig, axes = figure(1, n, size="wide")
    for k, p in enumerate(paths):
        ax = axes.flat[k]
        x, y, s = (np.asarray(p[c], float) for c in ("x", "y", "speed"))
        sc = ax.scatter(x - x[0], y - y[0], c=s, cmap="Blues", s=14)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(p.get("title", ""), fontsize=9)
        ax.set_xlabel("Δx [yd]")
        ax.set_ylabel("Δy [yd]")
        fig.colorbar(sc, ax=ax, shrink=0.7, label="speed [yd/s]")
    finish(fig, caption, title)
    return fig


# ============================================================================== 11B
def fig_envelopes(pos_points: pl.DataFrame, title: str, caption: str, quantile: float = 0.95):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    d = pos_points.filter(pl.col("quantile") == quantile)
    if d.is_empty():
        empty_panel(ax, "no envelope points")
        finish(fig, caption, title)
        return fig
    for g in [p for p in POSITIONS if p in set(d["position_group"].to_list())]:
        s = d.filter(pl.col("position_group") == g).sort("speed_mid")
        ax.plot(s["speed_mid"], s["accel_q"], color=_pos_color(g), lw=2, marker="o", ms=4, label=g)
    ax.set_xlabel("speed [yd/s]")
    ax.set_ylabel(f"{int(quantile * 100)}th percentile of a_tan [yd/s²]")
    ax.legend(frameon=False)
    finish(fig, caption, title)
    return fig


def fig_athlete_overlay(nfl_pts: pl.DataFrame, combine_v: np.ndarray, combine_a: np.ndarray, fit: dict[str, float], title: str, caption: str):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if nfl_pts.height:
        ax.scatter(nfl_pts["speed_mid"], nfl_pts["accel_q"], s=np.clip(nfl_pts["n_frames"].to_numpy() / 20, 8, 80), color=DOMAIN_COLORS["nfl"],
                   label="NFL games: per-speed-bin upper quantile (size = frames)")
    if combine_v.size:
        ax.plot(combine_v, combine_a, color=DOMAIN_COLORS["combine"], lw=1.5, alpha=0.8, label="Combine forty: a_tan vs speed")
    if np.isfinite(fit.get("A0", np.nan)) and np.isfinite(fit.get("V0", np.nan)):
        v = np.linspace(0, fit["V0"], 50)
        ax.plot(v, fit["A0"] * (1 - v / fit["V0"]), color=INK["secondary"], ls="--", lw=1.5, label="linear envelope fit (NFL)")
    ax.set_xlabel("speed [yd/s]")
    ax.set_ylabel("tangential acceleration [yd/s²]")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    finish(fig, caption, title)
    return fig


# ============================================================================== 11D / 11E / 11F / 11G
def fig_spaghetti(panel: pl.DataFrame, trait: str, title: str, caption: str, max_athletes: int = 60, seed: int = 0):
    d = panel.filter((pl.col("trait") == trait) & pl.col("supported"))
    positions = [p for p in POSITIONS if p in set(d["position_group"].drop_nulls().to_list())]
    fig, axes = figure(1, max(1, len(positions)), size="wide", sharey=True)
    for k, pos in enumerate(positions):
        ax = axes.flat[k]
        s = d.filter(pl.col("position_group") == pos)
        ids = s.group_by("nfl_id").len().filter(pl.col("len") >= 2)["nfl_id"].to_list()
        rng = np.random.default_rng(seed)
        for i in (rng.choice(ids, min(max_athletes, len(ids)), replace=False) if ids else []):
            g = s.filter(pl.col("nfl_id") == i).sort("years_since_draft")
            ax.plot(g["years_since_draft"], g["value"], color=_pos_color(pos), alpha=0.25, lw=1)
        m = s.group_by("years_since_draft").agg(pl.col("value").median(), pl.len()).sort("years_since_draft")
        ax.plot(m["years_since_draft"], m["value"], color=INK["primary"], lw=2.5, marker="o", label="median")
        ax.set_title(f"{pos} ({len(ids)} athletes ≥ 2 seasons)", fontsize=9)
        ax.set_xlabel("years since draft")
    if positions:
        axes.flat[0].set_ylabel(trait)
    else:
        empty_panel(axes.flat[0], "no multi-season athletes")
    finish(fig, caption, title)
    return fig


def fig_embedding_map(E: np.ndarray, meta: pl.DataFrame, title: str, caption: str, max_points: int = 6000, seed: int = 0):
    fig, axes = figure(1, 2, size="wide")
    rng = np.random.default_rng(seed)
    idx = rng.choice(E.shape[0], min(max_points, E.shape[0]), replace=False) if E.shape[0] else np.array([], int)
    if idx.size < 3:
        empty_panel(axes[0, 0], "no embeddings")
        finish(fig, caption, title)
        return fig
    Z = E[idx] - E[idx].mean(0)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    P = Z @ vt[:2].T
    dom = meta["domain"].to_numpy()[idx]
    kind = meta["kind"].to_numpy()[idx]
    for d in ("combine", "nfl"):
        m = dom == d
        axes[0, 0].scatter(P[m, 0], P[m, 1], s=4, alpha=0.4, color=DOMAIN_COLORS[d], label=f"{d} (n={int(m.sum())})")
    axes[0, 0].legend(frameon=False, markerscale=3)
    axes[0, 0].set_title("by domain", fontsize=10)
    kinds = [k for k, _ in sorted(zip(*np.unique(kind, return_counts=True)), key=lambda z: -z[1])][:8]
    for j, k in enumerate(kinds):
        m = kind == k
        axes[0, 1].scatter(P[m, 0], P[m, 1], s=4, alpha=0.4, color=CATEGORICAL[j], label=k)
    axes[0, 1].legend(frameon=False, markerscale=3, fontsize=7)
    axes[0, 1].set_title("by drill family / NFL demand (8 most frequent)", fontsize=10)
    for ax in axes.flat:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    finish(fig, caption, title)
    return fig


def fig_curves(curves: pl.DataFrame, title: str, caption: str):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    for j, (name,) in enumerate(sorted(curves.partition_by(["config"], as_dict=True))):
        s = curves.filter(pl.col("config") == name).sort("epoch")
        ax.plot(s["epoch"], s["val_loss"], color=CATEGORICAL[j % 8], lw=2, label=f"{name} (validation)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (objective-specific scale)")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, caption, title)
    return fig


def fig_frontier(front: pl.DataFrame, title: str, caption: str):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    ax.scatter(front["hypothetical_cost_min"], front["covered_cells"], color=INK["axis"], s=20, label="subset")
    p = front.filter(pl.col("pareto")).sort("hypothetical_cost_min")
    ax.plot(p["hypothetical_cost_min"], p["covered_cells"], color=CATEGORICAL[0], lw=2, marker="o", label="frontier")
    for r in p.iter_rows(named=True):
        ax.annotate(r["subset"], (r["hypothetical_cost_min"], r["covered_cells"]), fontsize=7, xytext=(4, 4), textcoords="offset points", color=INK["secondary"])
    ax.set_xlabel("declared HYPOTHETICAL cost [minutes per athlete]")
    ax.set_ylabel("SUPPORTED demand × position cells covered")
    ax.legend(frameon=False)
    finish(fig, caption, title)
    return fig


def fig_conclusion(rows: pl.DataFrame, title: str, caption: str):
    """Final conclusion figure: one row per hypothesis / comparison with its result status as TEXT on a coloured chip."""
    status_col = {"supported_finding": STATUS["good"], "inconclusive": STATUS["warning"], "unsupported_hypothesis": STATUS["serious"],
                  "executed_exploratory": SEQUENTIAL_BLUE[2], "completed_statistical_comparison": SEQUENTIAL_BLUE[3], "not_feasible": INK["axis"],
                  "authored_not_executed": INK["grid"]}
    fig, ax = figure(1, 1, size=(12.8, max(3.5, 0.45 * rows.height + 1.5)))
    ax = ax[0, 0]
    if rows.is_empty():
        empty_panel(ax, "no result rows")
        finish(fig, caption, title)
        return fig
    for i, r in enumerate(rows.iter_rows(named=True)):
        s = str(r["status"])
        ax.add_patch(FancyBboxPatch((0.0, i - 0.35), 0.22, 0.7, boxstyle="round,pad=0.02", color=status_col.get(s, INK["grid"]), lw=0))
        ax.text(0.11, i, s.replace("_", " "), ha="center", va="center", fontsize=8, color=INK["primary"])
        ax.text(0.25, i, f"{r['item']} — {r.get('evidence', '')}", ha="left", va="center", fontsize=9, color=INK["primary"])
    ax.set_xlim(0, 1.6)
    ax.set_ylim(rows.height - 0.5, -0.5)
    ax.axis("off")
    finish(fig, caption, title)
    return fig


__all__ = ["CLASS_COLORS", "CLASS_LABELS", "DOMAIN_COLORS", "FEAS_COLORS", "POSITION_COLORS", "fig_athlete_overlay", "fig_conclusion", "fig_curves",
           "fig_distributions", "fig_embedding_map", "fig_envelopes", "fig_forest", "fig_frontier", "fig_funnel", "fig_grouped_bars", "fig_heatmap",
           "fig_null_vs_observed", "fig_paths", "fig_scatter", "fig_spaghetti", "fig_state_grid", "fig_traces"]
