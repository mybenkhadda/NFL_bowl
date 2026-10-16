"""Milestone 12 figures (matplotlib through ``figkit``: 16:9, captions that state the denominator; saved by
``store.show`` as PNG 300 DPI + SVG, + PDF for major figures, each with a metadata sidecar and its plot data).

Reused from Milestone 11 (``m11.figures``): funnel, heatmap, state grid, forest, scatter, grouped bars, null-vs-observed,
distributions, embedding map. New here: line / band plots, matrices, athlete speed–acceleration clouds with quantile
curves, calibration, FPCA components, transport plans, training curves, attenuation and SIMEX curves.
Conventions: the project's categorical palette in fixed order, positions always the same colour, one y-axis per panel,
a reference line on every effect plot, explicit "no data" panels, colour never the only carrier of meaning.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ....data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE
from ...figkit import empty_panel, figure, finish
from ..m11 import figures as f11

POSITION_COLORS = f11.POSITION_COLORS
fig_funnel, fig_heatmap, fig_state_grid, fig_forest = f11.fig_funnel, f11.fig_heatmap, f11.fig_state_grid, f11.fig_forest
fig_scatter, fig_grouped_bars, fig_null_vs_observed = f11.fig_scatter, f11.fig_grouped_bars, f11.fig_null_vs_observed
fig_distributions, fig_embedding_map = f11.fig_distributions, f11.fig_embedding_map
_TO_M11 = {"supported": "supported_finding", "rejected": "unsupported_hypothesis", "executed": "executed_exploratory", "inconclusive": "inconclusive",
           "not_feasible": "not_feasible", "execution_pending": "authored_not_executed", "incomplete": "authored_not_executed"}


def fig_conclusion(rows: pl.DataFrame, title: str, caption: str):
    """Milestone 11 conclusion chips with the Milestone 12 status vocabulary mapped onto the Milestone 11 colour labels (supported → supported finding,
    rejected → unsupported hypothesis, pending / incomplete → authored-not-executed grey)."""
    if rows.is_empty():
        return f11.fig_conclusion(rows, title, caption)
    fig = f11.fig_conclusion(rows.with_columns(pl.col("status").replace_strict(_TO_M11, default=pl.col("status"), return_dtype=pl.String)), title, caption)
    return fig
STATUS_COLORS = {"supported": "#0ca30c", "inconclusive": "#fab219", "rejected": "#ec835a", "reversed": "#ec835a", "not_run": INK["grid"],
                 "execution_pending": INK["grid"], "not_tested": INK["grid"], "not_feasible": INK["axis"], "descriptive": SEQUENTIAL_BLUE[2],
                 "insufficient_athletes_for_interval": INK["axis"], "ready": "#0ca30c", "gated_on_upstream": SEQUENTIAL_BLUE[2],
                 "limited_support": "#fab219", "blocked": INK["axis"], "executed": SEQUENTIAL_BLUE[3], "incomplete": INK["axis"]}


def _color(i: int, key: Any = None) -> str:
    if key is not None and str(key) in POSITION_COLORS:
        return POSITION_COLORS[str(key)]
    return CATEGORICAL[i % len(CATEGORICAL)]


def fig_lines(df: pl.DataFrame, x: str, y: str, group: str | None, title: str, caption: str, xlabel: str, ylabel: str, lo: str | None = None,
              hi: str | None = None, ref: float | None = None, shade: tuple[float, float] | None = None, logx: bool = False):
    """One line per group (optional CI band), optional horizontal reference and shaded x-range."""
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if df.is_empty():
        empty_panel(ax, "no data")
        finish(fig, caption, title)
        return fig
    groups = [None] if group is None else sorted(df[group].cast(pl.String).unique().to_list())
    for k, g in enumerate(groups):
        d = (df if g is None else df.filter(pl.col(group).cast(pl.String) == g)).sort(x)
        xs, ys = d[x].cast(pl.Float64).to_numpy(), d[y].cast(pl.Float64).to_numpy()
        col = _color(k, g)
        ax.plot(xs, ys, "-o", color=col, ms=3, lw=1.8, label=str(g) if g is not None else None)
        if lo and hi and lo in d.columns and hi in d.columns:
            ax.fill_between(xs, d[lo].cast(pl.Float64).to_numpy(), d[hi].cast(pl.Float64).to_numpy(), color=col, alpha=0.15, lw=0)
    if ref is not None:
        ax.axhline(ref, color=INK["secondary"], lw=1, ls="--")
    if shade is not None and all(np.isfinite(shade)):
        ax.axvspan(shade[0], shade[1], color=SEQUENTIAL_BLUE[0], alpha=0.5, lw=0, label="shared speed range")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if group is not None or shade is not None:
        ax.legend(frameon=False, fontsize=8, ncols=min(4, len(groups) + 1))
    finish(fig, caption, title)
    return fig


def fig_matrix(M: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str, caption: str, cbar: str, cmap: str = "RdBu_r", center: bool = True,
               annotate: bool = True, fmt: str = "{:.2f}"):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if M.size == 0 or not np.isfinite(M).any():
        empty_panel(ax, "no data")
        finish(fig, caption, title)
        return fig
    v = float(np.nanmax(np.abs(M))) if center else None
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-v if center else None, vmax=v if center else None)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    if annotate and M.size <= 400:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7, color=INK["primary"])
    fig.colorbar(im, ax=ax, shrink=0.8, label=cbar)
    ax.grid(False)
    finish(fig, caption, title)
    return fig


def fig_athlete_cloud(H: np.ndarray, speed_edges: np.ndarray, accel_edges: np.ndarray, curves: dict[str, np.ndarray], title: str, caption: str,
                      shared: tuple[float, float] | None = None):
    """Frame-count image of one athlete's speed × acceleration observations (log colour) with estimator curves."""
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if H.sum() <= 0:
        empty_panel(ax, "no frames")
        finish(fig, caption, title)
        return fig
    im = ax.imshow(np.log1p(H.T), origin="lower", aspect="auto", cmap="Greys",
                   extent=(speed_edges[0], speed_edges[-1], accel_edges[0], accel_edges[-1]))
    mids = 0.5 * (speed_edges[:-1] + speed_edges[1:])
    for k, (name, q) in enumerate(curves.items()):
        ok = np.isfinite(q)
        ax.plot(mids[ok], q[ok], "-o", ms=3, lw=1.8, color=CATEGORICAL[k % 8], label=name)
    if shared is not None and all(np.isfinite(shared)):
        ax.axvline(shared[0], color=INK["secondary"], lw=1, ls=":")
        ax.axvline(shared[1], color=INK["secondary"], lw=1, ls=":")
    ax.set_xlabel("speed [yd/s]")
    ax.set_ylabel("tangential acceleration [yd/s²]")
    ax.set_ylim(-6, 8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log(1 + frames)")
    ax.grid(False)
    finish(fig, caption, title)
    return fig


def fig_calibration(summary: pl.DataFrame, title: str, caption: str):
    """Held-out coverage against the nominal τ per estimator (diagonal = calibrated)."""
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if summary.is_empty():
        empty_panel(ax, "no forecast rows")
        finish(fig, caption, title)
        return fig
    ax.plot([0, 1], [0, 1], color=INK["secondary"], ls="--", lw=1)
    for k, (e,) in enumerate(sorted(summary.select("estimator").unique().rows())):
        d = summary.filter(pl.col("estimator") == e).sort("tau")
        ax.plot(d["tau"].to_numpy(), d["coverage"].to_numpy(), "-o", color=CATEGORICAL[k % 8], label=e)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("nominal quantile level τ")
    ax.set_ylabel("held-out share of frames ≤ estimate")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, caption, title)
    return fig


def fig_components(x: np.ndarray, mean: np.ndarray, phi: np.ndarray, eig: np.ndarray, title: str, caption: str, xlabel: str, ylabel: str):
    """Mean function ± 2√λ_k φ_k for each component (one panel per component)."""
    K = phi.shape[1] if phi.size else 0
    fig, axes = figure(1, max(1, K), size="wide")
    if K == 0:
        empty_panel(axes[0, 0], "FPCA not fitted")
        finish(fig, caption, title)
        return fig
    for k in range(K):
        ax = axes[0, k]
        s = 2 * np.sqrt(max(float(eig[k]), 0.0))
        ax.plot(x, mean, color=INK["primary"], lw=2, label="mean")
        ax.plot(x, mean + s * phi[:, k], color=CATEGORICAL[0], lw=1.5, label="+2√λ φ")
        ax.plot(x, mean - s * phi[:, k], color=CATEGORICAL[1], lw=1.5, label="−2√λ φ")
        ax.set_title(f"component {k + 1}", fontsize=10)
        ax.set_xlabel(xlabel)
        if k == 0:
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False, fontsize=7)
    finish(fig, caption, title)
    return fig


def fig_curve_examples(x: np.ndarray, series: Sequence[dict[str, Any]], title: str, caption: str, xlabel: str, ylabel: str, ncols: int = 3):
    """Small multiples: each item {label, lines: {name: y}} — e.g. observed vs predicted NFL functions."""
    n = max(1, len(series))
    nc = min(ncols, n)
    fig, axes = figure(int(np.ceil(n / nc)), nc, size="wide")
    for k, s in enumerate(series):
        ax = axes.flat[k]
        for j, (name, y) in enumerate(s["lines"].items()):
            y = np.asarray(y, float)
            ok = np.isfinite(y)
            ax.plot(np.asarray(x)[ok], y[ok], "-o" if j == 0 else "-", ms=3, lw=1.6, color=CATEGORICAL[j % 8], label=name)
        ax.set_title(s["label"], fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if k == 0:
            ax.legend(frameon=False, fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    finish(fig, caption, title)
    return fig


def fig_transport_plan(plan: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, title: str, caption: str, top: int = 150):
    """The largest-mass couplings of a transport plan drawn as segments between (speed, acceleration) support points."""
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if plan.size == 0:
        empty_panel(ax, "no plan")
        finish(fig, caption, title)
        return fig
    flat = np.argsort(plan.ravel())[::-1][:top]
    w = plan.ravel()[flat]
    w = w / w.max() if w.max() > 0 else w
    for (i, j), ww in zip(zip(*np.unravel_index(flat, plan.shape)), w):
        ax.plot([src_pts[i, 0], dst_pts[j, 0]], [src_pts[i, 1], dst_pts[j, 1]], color=INK["axis"], lw=0.4 + 2.0 * ww, alpha=0.6)
    ax.scatter(src_pts[:, 0], src_pts[:, 1], s=8, color=CATEGORICAL[0], label="Combine support")
    ax.scatter(dst_pts[:, 0], dst_pts[:, 1], s=8, color=CATEGORICAL[1], label="NFL support")
    ax.set_xlabel("speed [yd/s]")
    ax.set_ylabel("tangential acceleration [yd/s²]")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, caption, title)
    return fig


def fig_training(curves: pl.DataFrame, title: str, caption: str):
    """Validation loss and training loss components per epoch, one panel per encoder."""
    names = sorted(curves["encoder"].unique().to_list()) if curves.height else []
    n = max(1, len(names))
    nc = min(4, n)
    fig, axes = figure(int(np.ceil(n / nc)), nc, size="wide")
    if not names:
        empty_panel(axes[0, 0], "no training run")
    for k, nm in enumerate(names):
        ax = axes.flat[k]
        d = curves.filter(pl.col("encoder") == nm).sort("epoch")
        for j, col in enumerate([c for c in d.columns if c.startswith("train_") or c == "val_loss"]):
            if d[col].is_not_null().any():
                ax.plot(d["epoch"].to_numpy(), d[col].to_numpy(), lw=1.6, color=INK["primary"] if col == "val_loss" else CATEGORICAL[j % 8],
                        ls="-" if col == "val_loss" else "--", label=col)
        ax.set_title(nm, fontsize=9)
        ax.set_xlabel("epoch")
        if k == 0:
            ax.legend(frameon=False, fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    finish(fig, caption, title)
    return fig


def fig_stacked(df: pl.DataFrame, cat: str, parts: Sequence[str], title: str, caption: str, ylabel: str):
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    if df.is_empty():
        empty_panel(ax, "no data")
        finish(fig, caption, title)
        return fig
    x = np.arange(df.height)
    bottom = np.zeros(df.height)
    for k, p in enumerate(parts):
        v = df[p].cast(pl.Float64).fill_null(0).to_numpy()
        ax.bar(x, v, bottom=bottom, color=CATEGORICAL[k % 8], label=p, width=0.7)
        bottom += v
    ax.set_xticks(x, df[cat].cast(pl.String).to_list(), rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    finish(fig, caption, title)
    return fig


__all__ = ["POSITION_COLORS", "STATUS_COLORS", "fig_athlete_cloud", "fig_calibration", "fig_components", "fig_conclusion", "fig_curve_examples",
           "fig_distributions", "fig_embedding_map", "fig_forest", "fig_funnel", "fig_grouped_bars", "fig_heatmap", "fig_lines", "fig_matrix",
           "fig_null_vs_observed", "fig_scatter", "fig_stacked", "fig_state_grid", "fig_training", "fig_transport_plan"]
