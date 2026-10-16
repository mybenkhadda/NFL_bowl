"""Figures. Every function RETURNS a ``matplotlib.figure.Figure`` built without pyplot (no global state, no display side
effect, works headless) and takes plain polars tables, so a figure can be rebuilt from saved CSVs in a new session.
Saving (PNG / SVG / PDF + the plotted data) is ``ExperimentStore.save_figure`` / ``Experiment.plot``.

Style: light surface, recessive grid, thin marks, one axis per panel, text in ink colours (never the series colour),
categorical hues in a fixed order (the validated reference palette of the dataviz method), sequential = one blue ramp.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQ_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
REF = "#898781"


def _figure(w: float, h: float) -> tuple[Any, Any]:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(w, h), facecolor=SURFACE, layout="constrained")
    return fig, fig.add_subplot(1, 1, 1)


def _style(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "", grid_axis: str = "both") -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8, length=3, width=0.6)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)


def _colors(keys: Sequence[Any]) -> dict[Any, str]:
    """Fixed-order assignment by first appearance; beyond eight keys the rest share the muted ink (fold to 'other')."""
    out: dict[Any, str] = {}
    for k in keys:
        if k not in out:
            out[k] = SERIES[len(out)] if len(out) < len(SERIES) else MUTED
    return out


def _empty(title: str, note: str) -> Any:
    fig, ax = _figure(6, 2)
    _style(ax, title, grid_axis="")
    ax.axis("off")
    ax.text(0.0, 0.5, note, color=INK2, fontsize=9, transform=ax.transAxes)
    return fig


def forest(df: Any, label: str, est: str, lo: str | None, hi: str | None, title: str, xlabel: str, ref: float | None = None,
           ref_label: str | None = None, extra_ref: float | None = None, extra_ref_label: str | None = None, color_by: str | None = None) -> Any:
    """Point estimate ± interval per row (one axis; reference line(s) labelled in ink)."""
    if df is None or not df.height:
        return _empty(title, "no rows to plot")
    d = df.filter(df[est].is_not_null())
    labels = d[label].cast(str).to_list()
    y = np.arange(len(labels))[::-1]
    fig, ax = _figure(7.5, max(2.2, 0.32 * len(labels) + 1.2))
    _style(ax, title, xlabel, "", grid_axis="x")
    groups = d[color_by].cast(str).to_list() if color_by else ["all"] * len(labels)
    cmap = _colors(groups)
    v = d[est].to_numpy().astype(float)
    if lo and hi and lo in d.columns and hi in d.columns:
        l, h = d[lo].to_numpy().astype(float), d[hi].to_numpy().astype(float)
        for yi, li, hi_, g in zip(y, l, h, groups):
            if np.isfinite(li) and np.isfinite(hi_):
                ax.plot([li, hi_], [yi, yi], color=cmap[g], linewidth=2, solid_capstyle="round", zorder=2)
    ax.scatter(v, y, s=36, color=[cmap[g] for g in groups], edgecolors=SURFACE, linewidths=1.5, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=REF, linewidth=1, linestyle="--", zorder=1)
        if ref_label:
            ax.annotate(ref_label, (ref, 1.0), xycoords=("data", "axes fraction"), color=INK2, fontsize=7.5, ha="left", va="bottom", xytext=(3, 2),
                        textcoords="offset points")
    if extra_ref is not None and np.isfinite(extra_ref):
        ax.axvline(extra_ref, color=INK2, linewidth=0.8, linestyle=":", zorder=1)
        if extra_ref_label:
            ax.annotate(extra_ref_label, (extra_ref, 0.0), xycoords=("data", "axes fraction"), color=INK2, fontsize=7.5, ha="left", va="bottom",
                        xytext=(3, 2), textcoords="offset points")
    ax.set_yticks(y, labels)
    if color_by and len(cmap) > 1:
        from matplotlib.lines import Line2D

        ax.legend([Line2D([], [], marker="o", linestyle="", color=c) for c in cmap.values()], list(cmap), frameon=False, fontsize=7.5,
                  labelcolor=INK2, loc="lower right")
    return fig


def training_curves(curves: Any, title: str = "Validation loss per epoch") -> Any:
    """Validation loss (held-out athletes) per run; one line per model × tag, direct-labelled at the last epoch."""
    if curves is None or not curves.height:
        return _empty(title, "no training curves yet")
    import polars as pl

    d = curves.with_columns((pl.col("model") + " · " + pl.col("tag")).alias("_run"))
    runs = d["_run"].unique(maintain_order=True).to_list()
    cmap = _colors(runs)
    fig, ax = _figure(7.5, 4.2)
    _style(ax, title, "epoch", "validation loss", grid_axis="y")
    for r in runs:
        g = d.filter(pl.col("_run") == r).sort("epoch")
        ax.plot(g["epoch"].to_numpy() + 1, g["val_loss"].to_numpy(), color=cmap[r], linewidth=2)
        if len(runs) <= 4:
            ax.annotate(r, (g["epoch"][-1] + 1, g["val_loss"][-1]), color=INK2, fontsize=7.5, xytext=(4, 0), textcoords="offset points", va="center")
    if len(runs) > 1:
        ax.legend(runs, frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper right")
    return fig


def heatmap(df: Any, row: str, col: str, value: str, title: str, fmt: str = "{:.2f}", vmin: float | None = None, vmax: float | None = None) -> Any:
    """Single-hue sequential heatmap with the value written in every cell (ink chosen for contrast)."""
    if df is None or not df.height:
        return _empty(title, "no rows to plot")
    from matplotlib.colors import LinearSegmentedColormap

    rows = df[row].cast(str).unique(maintain_order=True).to_list()
    cols = df[col].cast(str).unique(maintain_order=True).to_list()
    Z = np.full((len(rows), len(cols)), np.nan)
    for r, c, v in zip(df[row].cast(str).to_list(), df[col].cast(str).to_list(), df[value].to_list()):
        Z[rows.index(r), cols.index(c)] = np.nan if v is None else float(v)
    fig, ax = _figure(max(5.0, 1.1 * len(cols) + 3), max(2.4, 0.34 * len(rows) + 1.4))
    _style(ax, title, grid_axis="")
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    cmap.set_bad(GRID)
    lo = np.nanmin(Z) if vmin is None and np.isfinite(Z).any() else (vmin or 0.0)
    hi = np.nanmax(Z) if vmax is None and np.isfinite(Z).any() else (vmax or 1.0)
    im = ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1e-9, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(Z[i, j]):
                dark = (Z[i, j] - lo) / ((hi - lo) or 1) > 0.55
                ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center", fontsize=7.5, color=SURFACE if dark else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=7.5)
    return fig


def null_histogram(null: Any, observed: float | None, title: str = "Shuffled-identity null") -> Any:
    """Distribution of null retrieval AUCs with the observed AUC as a labelled reference line."""
    v = np.asarray(null, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return _empty(title, "no null replicates")
    fig, ax = _figure(6.5, 3.4)
    _style(ax, title, "cross-domain retrieval AUC", "replicates", grid_axis="y")
    ax.hist(v, bins=min(30, max(5, v.size // 2)), color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.axvline(0.5, color=REF, linestyle="--", linewidth=1)
    if observed is not None and np.isfinite(observed):
        ax.axvline(observed, color=INK, linewidth=1.5)
        ax.annotate(f"observed {observed:.3f}", (observed, 1.0), xycoords=("data", "axes fraction"), color=INK, fontsize=8, xytext=(4, -10),
                    textcoords="offset points")
    return fig


def embedding_map(E: np.ndarray, meta: Any, title: str = "Embedding map (PCA, 2 components)", max_points: int = 6000, seed: int = 0) -> Any:
    """2-D PCA projection coloured by domain (two hues). A picture is not evidence — see the retrieval tables."""
    dom = meta["domain"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(E.shape[0], min(max_points, E.shape[0]), replace=False))
    Z = E[idx].astype(float)
    Z = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    Y = Z @ vt[:2].T
    fig, ax = _figure(6.5, 5.2)
    _style(ax, title, "PC 1", "PC 2", grid_axis="both")
    for k, (name, c) in enumerate((("combine", SERIES[0]), ("nfl", SERIES[1]))):
        m = dom[idx] == name
        ax.scatter(Y[m, 0], Y[m, 1], s=9, color=c, alpha=0.55, linewidths=0, label=f"{name} (n={int(m.sum())})", zorder=2 + k)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, markerscale=2)
    return fig


def lines(df: Any, x: str, y: str, lo: str | None, hi: str | None, title: str, xlabel: str, ylabel: str, ref: float | None = None) -> Any:
    if df is None or not df.height:
        return _empty(title, "no rows to plot")
    d = df.sort(x)
    fig, ax = _figure(6.5, 3.6)
    _style(ax, title, xlabel, ylabel, grid_axis="y")
    xv, yv = d[x].to_numpy().astype(float), d[y].to_numpy().astype(float)
    if lo and hi and lo in d.columns and hi in d.columns:
        ax.fill_between(xv, d[lo].to_numpy().astype(float), d[hi].to_numpy().astype(float), color=SEQ_BLUE[1], alpha=0.6, linewidth=0)
    ax.plot(xv, yv, color=SERIES[0], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    if ref is not None:
        ax.axhline(ref, color=REF, linestyle="--", linewidth=1)
    return fig


__all__ = ["embedding_map", "forest", "heatmap", "lines", "null_histogram", "training_curves"]
