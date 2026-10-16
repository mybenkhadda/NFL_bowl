"""Lab figure library — one visual identity for every section.

Built on ``nflbowl.visualization.plots`` (same surfaces, inks, recessive grid and the validated reference palette in
fixed order). Every function RETURNS a matplotlib ``Figure`` (no pyplot state) from plain polars tables, so a figure
can be rebuilt from the saved data in ``figures/data``. Wide 16:9 canvases by default; axes never truncated for bars;
intervals drawn whenever the table carries them; text in ink colours, never in series colours.

Colour jobs: identity (models / domains) = categorical slots in FIXED order per entity (``MODEL_COLORS``, Combine =
slot 1, NFL = slot 2); magnitude = the single blue ramp; signed differences = diverging blue ↔ grey ↔ red.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..visualization import plots as pv

SURFACE, INK, INK2, MUTED, GRID, AXIS, REF = pv.SURFACE, pv.INK, pv.INK2, pv.MUTED, pv.GRID, pv.AXIS, pv.REF
S = pv.SERIES
WIDE = (12.0, 6.75)
HALF = (12.0, 4.6)
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")          # reference diverging pair, neutral grey midpoint
DOMAIN_COLORS = {"combine": S[0], "nfl": S[1]}
# identity follows the entity, never its rank: one fixed colour per representation across every figure
MODEL_COLORS = {"M1": MUTED, "F0_engineered": S[0], "M1+F0": S[0], "F4_physics_contrastive": S[1], "M1+F4_physics_contrastive": S[1],
                "F2_cnn_physics": S[2], "M1+F2_cnn_physics": S[2], "F3_masked_ae": S[3], "M1+F3_masked_ae": S[3], "F5_jepa_physics": S[4],
                "M1+F5_jepa_physics": S[4], "F6_transformer": S[5], "M1+F6_transformer": S[5], "F1_TCN": S[6], "M1+F1_TCN": S[6],
                "random_frozen_F4_architecture": S[7]}
SHORT = {"F0_engineered": "F0", "F2_cnn_physics": "F2", "F3_masked_ae": "F3", "F4_physics_contrastive": "F4", "F5_jepa_physics": "F5",
         "F6_transformer": "F6", "F1_TCN": "F1", "random_frozen_F4_architecture": "random"}


def short(name: str) -> str:
    out = str(name)
    for k, v in SHORT.items():
        out = out.replace(k, v)
    return out


def color(name: Any) -> str:
    return MODEL_COLORS.get(str(name), DOMAIN_COLORS.get(str(name), MUTED))


def palette_for(keys: Sequence[Any]) -> dict[str, str]:
    """Entity colours where the key is a model / domain; otherwise categorical slots in first-appearance order
    (beyond eight keys the rest share the muted ink)."""
    out: dict[str, str] = {}
    free = [c for c in S]
    for k in keys:
        k = str(k)
        if k in MODEL_COLORS or k in DOMAIN_COLORS:
            out[k] = color(k)
    used = set(out.values())
    free = [c for c in free if c not in used]
    for k in keys:
        k = str(k)
        if k not in out:
            out[k] = free.pop(0) if free else MUTED
    return out


def _fig(size: tuple[float, float] = WIDE, ncols: int = 1, nrows: int = 1, sharey: bool = False, sharex: bool = False) -> tuple[Any, Any]:
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, sharey=sharey, sharex=sharex, squeeze=False)
    return fig, axes


def _style(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "", grid: str = "y") -> None:
    pv._style(ax, title, xlabel, ylabel, grid_axis=grid)


def _suptitle(fig: Any, title: str, subtitle: str = "") -> None:
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13, fontweight="semibold")
    if subtitle:
        fig.text(0.01, 0.955, subtitle, ha="left", va="top", color=INK2, fontsize=9)


def _legend(ax: Any, loc: str = "best") -> None:
    h, l_ = ax.get_legend_handles_labels()
    if len(h) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc=loc)


def empty(title: str, note: str) -> Any:
    return pv._empty(title, note)


def _vals(df: Any, col: str) -> np.ndarray:
    v = df[col].cast(float).to_numpy() if df is not None and df.height else np.array([])
    return v[np.isfinite(v)]


# ====================================================================== distributions
def ecdf_ax(ax: Any, v: np.ndarray, label: str, c: str) -> None:
    v = np.sort(np.asarray(v, float)[np.isfinite(v)])
    if v.size:
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=c, linewidth=2, label=f"{label} (n={v.size:,})")


def distribution_compare(df: Any, value: str, group: str, title: str, xlabel: str, facet: str | None = None, max_facets: int = 8,
                         subtitle: str = "", order: Sequence[str] | None = None) -> Any:
    """Left: box + violin per facet × group (groups side by side); right: pooled ECDF per group. Same x unit both panels."""
    import polars as pl

    if df is None or not df.height:
        return empty(title, "no rows")
    d = df.filter(pl.col(value).is_not_null() & pl.col(value).cast(float).is_finite())
    groups = list(order) if order else d[group].cast(str).unique(maintain_order=True).to_list()
    facets = d[facet].cast(str).unique().sort().to_list()[:max_facets] if facet else ["all"]
    fig, axes = _fig(WIDE, ncols=2)
    a, b = axes[0, 0], axes[0, 1]
    _style(a, f"by {facet}" if facet else "distribution", xlabel, "", grid="x")
    pal = palette_for(groups)
    color = pal.get                                                     # noqa: F841 — local override of the module colour lookup
    w = 0.8 / max(len(groups), 1)
    yt, yl = [], []
    for i, fc in enumerate(facets):
        for j, g in enumerate(groups):
            sub = d.filter(pl.col(group).cast(str) == g)
            if facet:
                sub = sub.filter(pl.col(facet).cast(str) == fc)
            v = _vals(sub, value)
            y = i + (j - (len(groups) - 1) / 2) * w
            if v.size >= 5:
                try:                                                    # matplotlib ≥ 3.10
                    parts = a.violinplot(v, positions=[y], orientation="horizontal", widths=w * 0.95, showextrema=False)
                except TypeError:
                    parts = a.violinplot(v, positions=[y], vert=False, widths=w * 0.95, showextrema=False)
                for p in parts["bodies"]:
                    p.set_facecolor(color(g))
                    p.set_edgecolor("none")
                    p.set_alpha(0.35)
                q1, q2, q3 = np.quantile(v, [0.25, 0.5, 0.75])
                a.plot([q1, q3], [y, y], color=color(g), linewidth=3, solid_capstyle="round")
                a.scatter([q2], [y], s=28, color=SURFACE, edgecolors=color(g), linewidths=1.5, zorder=3)
        yt.append(i)
        yl.append(fc)
    a.set_yticks(yt, yl)
    _style(b, "empirical CDF (pooled)", xlabel, "cumulative share")
    for g in groups:
        ecdf_ax(b, _vals(d.filter(pl.col(group).cast(str) == g), value), g, color(g))
    _legend(b, "lower right")
    _suptitle(fig, title, subtitle)
    return fig


def histogram_by(df: Any, value: str, group: str, title: str, xlabel: str, bins: int = 40, log: bool = False) -> Any:
    import polars as pl

    if df is None or not df.height:
        return empty(title, "no rows")
    fig, axes = _fig(HALF)
    ax = axes[0, 0]
    _style(ax, "", xlabel, "count")
    allv = _vals(df, value)
    if not allv.size:
        return empty(title, "no finite values")
    edges = np.histogram_bin_edges(allv, bins=bins)
    for g in df[group].cast(str).unique(maintain_order=True).to_list():
        v = _vals(df.filter(pl.col(group).cast(str) == g), value)
        ax.hist(v, bins=edges, histtype="step", linewidth=2, color=color(g), label=f"{g} (n={v.size:,})")
    if log:
        ax.set_yscale("log")
    _legend(ax)
    _suptitle(fig, title)
    return fig


# ====================================================================== matrices
def heatmap(df: Any, row: str, col: str, value: str, title: str, fmt: str = "{:.2f}", vmin: float | None = None, vmax: float | None = None,
            diverging: bool = False, center: float = 0.0, subtitle: str = "") -> Any:
    """Sequential (one blue ramp) or diverging (blue ↔ grey ↔ red, centred on ``center``) annotated heatmap."""
    if df is None or not df.height:
        return empty(title, "no rows")
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    rows = df[row].cast(str).unique(maintain_order=True).to_list()
    cols = df[col].cast(str).unique(maintain_order=True).to_list()
    Z = np.full((len(rows), len(cols)), np.nan)
    for r, c, v in zip(df[row].cast(str).to_list(), df[col].cast(str).to_list(), df[value].to_list()):
        Z[rows.index(r), cols.index(c)] = np.nan if v is None else float(v)
    fig, axes = _fig((min(16.0, max(7.0, 0.9 * len(cols) + 4)), min(12.0, max(3.2, 0.36 * len(rows) + 1.8))))
    ax = axes[0, 0]
    _style(ax, "", grid="")
    fin = Z[np.isfinite(Z)]
    if diverging:
        span = float(np.nanmax(np.abs(fin - center))) if fin.size else 1.0
        span = span or 1.0
        cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
        norm = TwoSlopeNorm(vcenter=center, vmin=center - span if vmin is None else vmin, vmax=center + span if vmax is None else vmax)
        im = ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, norm=norm, aspect="auto")
        lo, hi = norm.vmin, norm.vmax
    else:
        cmap = LinearSegmentedColormap.from_list("seq", pv.SEQ_BLUE)
        lo = float(np.nanmin(fin)) if vmin is None and fin.size else (vmin if vmin is not None else 0.0)
        hi = float(np.nanmax(fin)) if vmax is None and fin.size else (vmax if vmax is not None else 1.0)
        im = ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1e-9, aspect="auto")
    cmap.set_bad(GRID)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(Z[i, j]):
                t = (Z[i, j] - lo) / ((hi - lo) or 1)
                dark = (abs(Z[i, j] - center) / ((hi - lo) / 2 or 1) > 0.6) if diverging else t > 0.55
                ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center", fontsize=7.5, color=SURFACE if dark else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=7.5)
    _suptitle(fig, title, subtitle)
    return fig


def correlation_matrix(C: np.ndarray, labels: Sequence[str], title: str, subtitle: str = "") -> Any:
    import polars as pl

    n = len(labels)
    df = pl.DataFrame({"row": [labels[i] for i in range(n) for _ in range(n)], "col": [labels[j] for _ in range(n) for j in range(n)],
                       "r": [float(C[i, j]) for i in range(n) for j in range(n)]})
    return heatmap(df, "row", "col", "r", title, fmt="{:.2f}", vmin=-1.0, vmax=1.0, diverging=True, subtitle=subtitle)


# ====================================================================== bars / flows
def hbar(df: Any, label: str, value: str, title: str, xlabel: str, color_by: str | None = None, fmt: str = "{:,.0f}", subtitle: str = "",
         ref: float | None = None) -> Any:
    """Horizontal bars from a zero baseline with direct value labels (counts, shares)."""
    if df is None or not df.height:
        return empty(title, "no rows")
    labels = df[label].cast(str).to_list()
    v = df[value].cast(float).to_numpy()
    y = np.arange(len(labels))[::-1]
    fig, axes = _fig((12.0, max(3.0, 0.34 * len(labels) + 1.6)))
    ax = axes[0, 0]
    _style(ax, "", xlabel, "", grid="x")
    cs = [color(c) for c in df[color_by].cast(str).to_list()] if color_by else [S[0]] * len(labels)
    ax.barh(y, np.nan_to_num(v), color=cs, height=0.7, edgecolor=SURFACE, linewidth=2)
    for yi, vi in zip(y, v):
        if np.isfinite(vi):
            ax.annotate(fmt.format(vi), (vi, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    if ref is not None:
        ax.axvline(ref, color=REF, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlim(left=min(0.0, float(np.nanmin(v)) if np.isfinite(v).any() else 0.0))
    _suptitle(fig, title, subtitle)
    return fig


def waterfall(steps: Any, title: str, subtitle: str = "") -> Any:
    """Sample attrition: remaining count after each step (bars from zero) with the removed count annotated."""
    if steps is None or not steps.height:
        return empty(title, "no steps")
    labels = steps["step"].to_list()
    n = steps["n_remaining"].cast(float).to_numpy()
    rem = steps["n_removed"].cast(float).to_numpy() if "n_removed" in steps.columns else np.zeros(n.size)
    x = np.arange(n.size)
    fig, axes = _fig(WIDE)
    ax = axes[0, 0]
    _style(ax, "", "", "athletes remaining")
    ax.bar(x, n, color=S[0], width=0.65, edgecolor=SURFACE, linewidth=2)
    prev = np.r_[n[0], n[:-1]]
    ax.bar(x, rem, bottom=n, color=GRID, width=0.65, edgecolor=SURFACE, linewidth=2)
    for xi, ni, ri in zip(x, n, rem):
        ax.annotate(f"{ni:,.0f}", (xi, ni), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8.5, color=SURFACE, fontweight="semibold")
        if ri > 0:
            ax.annotate(f"−{ri:,.0f}", (xi, ni + ri), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, float(np.nanmax(prev)) * 1.1 if prev.size else 1)
    _suptitle(fig, title, subtitle)
    return fig


def stacked(df: Any, x: str, stack: str, value: str, title: str, ylabel: str, normalize: bool = False, subtitle: str = "") -> Any:
    """Stacked vertical bars (≤ 8 stack levels in fixed order; 2px surface gap between segments)."""
    import polars as pl

    if df is None or not df.height:
        return empty(title, "no rows")
    xs = df[x].cast(str).unique(maintain_order=True).to_list()
    ks = df[stack].cast(str).unique().sort().to_list()
    M = np.zeros((len(ks), len(xs)))
    for a, b, v in zip(df[x].cast(str).to_list(), df[stack].cast(str).to_list(), df[value].cast(float).to_list()):
        M[ks.index(b), xs.index(a)] += 0.0 if v is None else v
    if normalize:
        M = M / np.maximum(M.sum(0, keepdims=True), 1e-12)
    fig, axes = _fig(WIDE)
    ax = axes[0, 0]
    _style(ax, "", "", ylabel)
    bottom = np.zeros(len(xs))
    for i, k in enumerate(ks):
        c = S[i] if i < len(S) else MUTED
        ax.bar(range(len(xs)), M[i], bottom=bottom, color=c, label=k, width=0.7, edgecolor=SURFACE, linewidth=2)
        bottom += M[i]
    ax.set_xticks(range(len(xs)), xs, rotation=0 if len(xs) < 8 else 30, ha="center" if len(xs) < 8 else "right")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncols=min(len(ks), 8), loc="upper left", bbox_to_anchor=(0, 1.08))
    _ = pl
    _suptitle(fig, title, subtitle)
    return fig


# ====================================================================== intervals
def forest(df: Any, label: str, est: str, lo: str | None, hi: str | None, title: str, xlabel: str, ref: float | None = None, ref_label: str | None = None,
           color_by: str | None = None, subtitle: str = "") -> Any:
    """Point ± interval per row, colour = model identity (fixed map), reference line labelled."""
    if df is None or not df.height:
        return empty(title, "no rows")
    import polars as pl

    d = df.filter(pl.col(est).is_not_null() & pl.col(est).cast(float).is_finite())
    if not d.height:
        return empty(title, "no finite estimates")
    labels = [short(x) for x in d[label].cast(str).to_list()]
    y = np.arange(len(labels))[::-1]
    fig, axes = _fig((12.0, max(3.0, 0.36 * len(labels) + 1.8)))
    ax = axes[0, 0]
    _style(ax, "", xlabel, "", grid="x")
    keys = d[color_by].cast(str).to_list() if color_by else [None] * len(labels)
    cs = [color(k) if k is not None else S[0] for k in keys]
    v = d[est].cast(float).to_numpy()
    if lo and hi and lo in d.columns and hi in d.columns:
        L, H = d[lo].cast(float).to_numpy(), d[hi].cast(float).to_numpy()
        for yi, li, hi_, c in zip(y, L, H, cs):
            if np.isfinite(li) and np.isfinite(hi_):
                ax.plot([li, hi_], [yi, yi], color=c, linewidth=2.2, solid_capstyle="round", zorder=2)
    ax.scatter(v, y, s=44, color=cs, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=REF, linewidth=1, linestyle="--", zorder=1)
        if ref_label:
            ax.annotate(ref_label, (ref, 1.0), xycoords=("data", "axes fraction"), color=INK2, fontsize=7.5, xytext=(3, 2), textcoords="offset points")
    ax.set_yticks(y, labels)
    _suptitle(fig, title, subtitle)
    return fig


def fold_points(df: Any, model: str, fold: str, value: str, title: str, ylabel: str, ref: float | None = None, subtitle: str = "") -> Any:
    """Per-fold values (dots) and their mean (bar tick) per model — fold variability at a glance."""
    if df is None or not df.height:
        return empty(title, "no rows")
    import polars as pl

    models = df[model].cast(str).unique(maintain_order=True).to_list()
    fig, axes = _fig(HALF)
    ax = axes[0, 0]
    _style(ax, "", "", ylabel)
    for i, m in enumerate(models):
        g = df.filter(pl.col(model).cast(str) == m)
        v = g[value].cast(float).to_numpy()
        f = g[fold].to_list()
        jit = np.linspace(-0.15, 0.15, max(len(v), 1))
        ax.scatter(i + jit[: len(v)], v, s=40, color=color(m), edgecolors=SURFACE, linewidths=1.5, zorder=3)
        for xj, vj, fj in zip(i + jit[: len(v)], v, f):
            ax.annotate(f"f{fj}", (xj, vj), xytext=(4, 0), textcoords="offset points", fontsize=7, color=MUTED, va="center")
        if np.isfinite(v).any():
            ax.plot([i - 0.25, i + 0.25], [np.nanmean(v)] * 2, color=INK, linewidth=1.5)
    if ref is not None:
        ax.axhline(ref, color=REF, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(models)), [short(m) for m in models])
    _suptitle(fig, title, subtitle)
    return fig


# ====================================================================== lines
def lines(df: Any, x: str, y: str, series: str, title: str, xlabel: str, ylabel: str, lo: str | None = None, hi: str | None = None,
          ref: Any = None, ref_label: str | None = None, logy: bool = False, logx: bool = False, subtitle: str = "", markers: bool = True,
          dashed_series: Sequence[str] = ()) -> Any:
    """One line per series (fixed colours), optional interval band, direct labels at the last point (≤ 4 series).
    ``ref`` may be a number (horizontal line) or the name of a column holding a per-x reference (drawn dashed, e.g. chance)."""
    import polars as pl

    if df is None or not df.height:
        return empty(title, "no rows")
    fig, axes = _fig(HALF)
    ax = axes[0, 0]
    _style(ax, "", xlabel, ylabel)
    keys = df[series].cast(str).unique(maintain_order=True).to_list()
    for k in keys:
        g = df.filter(pl.col(series).cast(str) == k).sort(x)
        xv, yv = g[x].cast(float).to_numpy(), g[y].cast(float).to_numpy()
        c = color(k) if k in MODEL_COLORS or k in DOMAIN_COLORS else S[keys.index(k) % len(S)]
        if lo and hi and lo in g.columns:
            ax.fill_between(xv, g[lo].cast(float).to_numpy(), g[hi].cast(float).to_numpy(), color=c, alpha=0.15, linewidth=0)
        ax.plot(xv, yv, color=c, linewidth=2, linestyle="--" if k in dashed_series else "-", marker="o" if markers and xv.size <= 30 else None,
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=short(k))
        if len(keys) <= 4 and xv.size:
            ax.annotate(short(k), (xv[-1], yv[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    if isinstance(ref, str) and ref in df.columns:
        g = df.unique(subset=[x], keep="first").sort(x)
        ax.plot(g[x].cast(float).to_numpy(), g[ref].cast(float).to_numpy(), color=REF, linestyle=":", linewidth=1.5, label=ref_label or "reference")
    elif ref is not None:
        ax.axhline(float(ref), color=REF, linestyle="--", linewidth=1)
        if ref_label:
            ax.annotate(ref_label, (0.0, float(ref)), xycoords=("axes fraction", "data"), xytext=(3, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    _legend(ax)
    _suptitle(fig, title, subtitle)
    return fig


def small_multiples_lines(df: Any, facet: str, x: str, y: str, series: str, title: str, xlabel: str, ylabel: str, ncols: int = 3,
                          max_facets: int = 9, subtitle: str = "", sharey: bool = False) -> Any:
    import polars as pl

    if df is None or not df.height:
        return empty(title, "no rows")
    facets = df[facet].cast(str).unique(maintain_order=True).to_list()[:max_facets]
    nrows = int(np.ceil(len(facets) / ncols))
    fig, axes = _fig((12.0, max(3.4, 2.6 * nrows + 1.0)), ncols=ncols, nrows=nrows, sharey=sharey)
    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= len(facets):
            ax.axis("off")
            continue
        fc = facets[k]
        _style(ax, fc, xlabel if k // ncols == nrows - 1 else "", ylabel if k % ncols == 0 else "")
        g = df.filter(pl.col(facet).cast(str) == fc)
        for i, s_ in enumerate(g[series].cast(str).unique(maintain_order=True).to_list()):
            h = g.filter(pl.col(series).cast(str) == s_).sort(x)
            c = color(s_) if (s_ in MODEL_COLORS or s_ in DOMAIN_COLORS) else S[i % len(S)]
            ax.plot(h[x].cast(float).to_numpy(), h[y].cast(float).to_numpy(), color=c, linewidth=1.8, label=short(s_))
        if k == 0:
            _legend(ax)
    _suptitle(fig, title, subtitle)
    return fig


# ====================================================================== scatter
def scatter(df: Any, x: str, y: str, title: str, xlabel: str, ylabel: str, group: str | None = None, label: str | None = None, identity: bool = False,
            ref_x: float | None = None, ref_y: float | None = None, subtitle: str = "", annotate_top: int = 0) -> Any:
    import polars as pl

    if df is None or not df.height:
        return empty(title, "no rows")
    d = df.filter(pl.col(x).cast(float).is_finite() & pl.col(y).cast(float).is_finite())
    fig, axes = _fig(WIDE)
    ax = axes[0, 0]
    _style(ax, "", xlabel, ylabel, grid="both")
    keys = d[group].cast(str).unique(maintain_order=True).to_list() if group else [None]
    for i, k in enumerate(keys):
        g = d if k is None else d.filter(pl.col(group).cast(str) == k)
        c = S[0] if k is None else (color(k) if (k in MODEL_COLORS or k in DOMAIN_COLORS) else (S[i] if i < len(S) else MUTED))
        ax.scatter(g[x].cast(float).to_numpy(), g[y].cast(float).to_numpy(), s=26, color=c, alpha=0.75, edgecolors=SURFACE, linewidths=0.8,
                   label=None if k is None else f"{short(k)} (n={g.height})")
    if identity and d.height:
        lo = float(min(d[x].min(), d[y].min()))
        hi = float(max(d[x].max(), d[y].max()))
        ax.plot([lo, hi], [lo, hi], color=REF, linestyle="--", linewidth=1)
    if ref_x is not None:
        ax.axvline(ref_x, color=REF, linestyle="--", linewidth=1)
    if ref_y is not None:
        ax.axhline(ref_y, color=REF, linestyle="--", linewidth=1)
    if label and annotate_top and d.height:
        top = d.with_columns((pl.col(y).cast(float) - pl.col(x).cast(float)).abs().alias("_d")).sort("_d", descending=True).head(annotate_top)
        for r in top.iter_rows(named=True):
            ax.annotate(str(r[label]), (r[x], r[y]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    elif label and d.height <= 25:
        for r in d.iter_rows(named=True):
            ax.annotate(short(str(r[label])), (r[x], r[y]), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK2)
    if group:
        _legend(ax)
    _suptitle(fig, title, subtitle)
    return fig


def pred_vs_actual(oof: Any, models: Sequence[str], title: str, subtitle: str = "") -> Any:
    """Small multiples: OOF prediction vs observed target per ladder model, identity line, RMSE in the panel title."""
    if oof is None or not oof.height:
        return empty(title, "no predictions")
    ncols = min(3, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = _fig((12.0, 3.6 * nrows + 0.8), ncols=ncols, nrows=nrows, sharex=True, sharey=True)
    y = oof["y"].cast(float).to_numpy()
    lo, hi = np.nanmin(y), np.nanmax(y)
    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= len(models):
            ax.axis("off")
            continue
        m = models[k]
        p = oof[f"pred_{m}"].cast(float).to_numpy()
        ok = np.isfinite(p) & np.isfinite(y)
        rmse = float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2))) if ok.any() else float("nan")
        _style(ax, f"{short(m)} · RMSE {rmse:.3f} (n={int(ok.sum())})", "predicted" if k // ncols == nrows - 1 else "", "observed" if k % ncols == 0 else "",
               grid="both")
        ax.scatter(p[ok], y[ok], s=14, color=color(m) if color(m) != MUTED else S[0], alpha=0.6, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=REF, linestyle="--", linewidth=1)
    _suptitle(fig, title, subtitle)
    return fig


def residual_panels(res: Any, models: Sequence[str], title: str, subtitle: str = "") -> Any:
    """Left: residual ECDF per model; right: residual vs observed (binned medians) — where each model errs."""
    import polars as pl

    if res is None or not res.height:
        return empty(title, "no residuals")
    fig, axes = _fig(WIDE, ncols=2)
    a, b = axes[0, 0], axes[0, 1]
    _style(a, "residual distribution (observed − predicted)", "residual", "cumulative share")
    _style(b, "median residual by observed-target decile", "observed target (decile mid)", "median residual")
    for m in models:
        g = res.filter(pl.col("model") == m)
        r = g["residual"].cast(float).to_numpy()
        ecdf_ax(a, r, short(m), color(m) if color(m) != MUTED else INK2)
        yv = g["y"].cast(float).to_numpy()
        if yv.size >= 20:
            qs = np.quantile(yv, np.linspace(0, 1, 11))
            mids, meds = [], []
            for i in range(10):
                mk = (yv >= qs[i]) & (yv <= qs[i + 1])
                if mk.sum() >= 3:
                    mids.append((qs[i] + qs[i + 1]) / 2)
                    meds.append(float(np.median(r[mk])))
            b.plot(mids, meds, color=color(m) if color(m) != MUTED else INK2, linewidth=2, marker="o", markersize=4, label=short(m))
    a.axvline(0, color=REF, linestyle="--", linewidth=1)
    b.axhline(0, color=REF, linestyle="--", linewidth=1)
    _legend(a, "lower right")
    _legend(b)
    _suptitle(fig, title, subtitle)
    return fig


def calibration(df: Any, title: str, subtitle: str = "") -> Any:
    """Binned mean observed vs mean predicted (deciles of the prediction) per model, identity line."""
    import polars as pl

    if df is None or not df.height:
        return empty(title, "no rows")
    fig, axes = _fig(HALF)
    ax = axes[0, 0]
    _style(ax, "", "mean predicted (decile bin)", "mean observed", grid="both")
    for m in df["model"].unique(maintain_order=True).to_list():
        g = df.filter(pl.col("model") == m).sort("pred_mean")
        ax.plot(g["pred_mean"].to_numpy(), g["obs_mean"].to_numpy(), color=color(m) if color(m) != MUTED else INK2, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, label=short(m))
    lo = float(min(df["pred_mean"].min(), df["obs_mean"].min()))
    hi = float(max(df["pred_mean"].max(), df["obs_mean"].max()))
    ax.plot([lo, hi], [lo, hi], color=REF, linestyle="--", linewidth=1)
    _legend(ax)
    _suptitle(fig, title, subtitle)
    return fig


def projection(coords: Any, title: str, subtitle: str = "") -> Any:
    """Two panels of the same 2-D coordinates: coloured by domain, coloured by position group (≤ 8 hues)."""
    import polars as pl

    if coords is None or not coords.height:
        return empty(title, "no coordinates")
    fig, axes = _fig(WIDE, ncols=2, sharex=True, sharey=True)
    a, b = axes[0, 0], axes[0, 1]
    _style(a, "coloured by domain", "component 1", "component 2", grid="both")
    _style(b, "coloured by position group", "component 1", "", grid="both")
    for k, dname in enumerate(("combine", "nfl")):
        g = coords.filter(pl.col("domain") == dname)
        a.scatter(g["x"].to_numpy(), g["y"].to_numpy(), s=7, color=DOMAIN_COLORS[dname], alpha=0.5, linewidths=0, label=f"{dname} (n={g.height:,})",
                  zorder=2 + k)
    pos = coords["position_group"].cast(str).value_counts(sort=True)["position_group"].to_list()
    for i, p in enumerate(pos):
        g = coords.filter(pl.col("position_group").cast(str) == p)
        b.scatter(g["x"].to_numpy(), g["y"].to_numpy(), s=7, color=S[i] if i < len(S) else MUTED, alpha=0.5, linewidths=0,
                  label=p if i < len(S) else None)
    a.legend(frameon=False, fontsize=8, labelcolor=INK2, markerscale=2.5)
    b.legend(frameon=False, fontsize=7.5, labelcolor=INK2, markerscale=2.5, ncols=2)
    _suptitle(fig, title, subtitle or "Exploratory visualization only — not evidence of separation or generalization.")
    return fig


def profile_panels(curves: Any, title: str, subtitle: str = "", ncols: int = 2) -> Any:
    """Movement profiles for case studies: per athlete panel, mean speed over time of Combine forty attempts and of
    NFL windows (± 1 SD band). ``curves``: athlete, domain, t, mean, sd."""
    import polars as pl

    if curves is None or not curves.height:
        return empty(title, "no profiles")
    aths = curves["athlete"].unique(maintain_order=True).to_list()
    nrows = int(np.ceil(len(aths) / ncols))
    fig, axes = _fig((12.0, 2.9 * nrows + 1.0), ncols=ncols, nrows=nrows, sharex=True, sharey=True)
    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= len(aths):
            ax.axis("off")
            continue
        a_ = aths[k]
        g = curves.filter(pl.col("athlete") == a_)
        _style(ax, str(g["label"][0]) if "label" in g.columns else str(a_), "time (s)" if k // ncols == nrows - 1 else "", "speed (yd/s)" if k % ncols == 0 else "")
        for dname in ("combine", "nfl"):
            h = g.filter(pl.col("domain") == dname).sort("t")
            if h.height:
                t, m, s_ = h["t"].to_numpy(), h["mean"].to_numpy(), h["sd"].fill_null(0).to_numpy()
                ax.fill_between(t, m - s_, m + s_, color=DOMAIN_COLORS[dname], alpha=0.15, linewidth=0)
                ax.plot(t, m, color=DOMAIN_COLORS[dname], linewidth=2, label=dname)
        if k == 0:
            _legend(ax)
    _suptitle(fig, title, subtitle)
    return fig


__all__ = ["DIVERGING", "DOMAIN_COLORS", "HALF", "MODEL_COLORS", "WIDE", "calibration", "color", "correlation_matrix", "distribution_compare", "ecdf_ax",
           "empty", "fold_points", "forest", "hbar", "heatmap", "histogram_by", "lines", "pred_vs_actual", "profile_panels", "projection", "residual_panels",
           "scatter", "short", "small_multiples_lines", "stacked", "waterfall"]
