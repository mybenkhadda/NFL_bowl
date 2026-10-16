"""Phase 03 figure kit on top of the project style (``data.reporting``) — shared by M2.1 and studies A–I.

Two recurring export problems are fixed here once:

* captions overlapping axis labels — :func:`finish` reserves a bottom band in the constrained-layout
  rectangle sized to the (wrapped) caption, then writes the caption inside that band;
* ``set_aspect("equal", adjustable="datalim")`` on axes that share BOTH x and y raises in Matplotlib
  (the M2 ``m2_three_cone_path_extremes`` failure) — :func:`equal_aspect_panels` gives unshared
  panels identical limits and uses ``adjustable="box"``, which is always legal.

Colour conventions: categorical = ``CATEGORICAL`` (colour-blind-safe project palette), tiers =
``TIER_COLORS``, domains = ``DOMAIN_COLORS``.
"""

from __future__ import annotations

import textwrap
from typing import Sequence

import numpy as np

from ..data.reporting import CATEGORICAL, FIG_16x9, INK, SEQUENTIAL_BLUE, STATUS, new_figure

SIZES = {"wide": FIG_16x9, "tall": (9.0, 11.0), "square": (8.0, 8.0), "panel": (6.4, 4.8)}
TIER_COLORS = {"A": SEQUENTIAL_BLUE[5], "B": SEQUENTIAL_BLUE[3], "C": SEQUENTIAL_BLUE[1], "reject": STATUS["critical"]}
DOMAIN_COLORS = {"combine": CATEGORICAL[0], "nfl": CATEGORICAL[1], "nfl_sample": CATEGORICAL[1]}
CAPTION_WRAP = 170
_LINE_FRAC = 0.022   # figure-height fraction per caption line at 9 pt on a 7.2-in figure


def figure(nrows: int = 1, ncols: int = 1, size: str | tuple[float, float] = "wide", **kwargs):
    """New figure (constrained layout); ``axes`` is always a 2-D array for uniform indexing."""
    figsize = SIZES[size] if isinstance(size, str) else size
    fig, axes = new_figure(nrows, ncols, figsize=figsize, squeeze=False, **kwargs)
    return fig, axes


def finish(fig, caption: str, title: str | None = None) -> None:
    """Reserve room for a wrapped caption (and an optional suptitle) so nothing overlaps."""
    lines = textwrap.wrap(caption, CAPTION_WRAP) if caption else []
    bottom = min(0.30, (len(lines) * _LINE_FRAC + 0.01) * 7.2 / fig.get_figheight()) if lines else 0.0
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):   # constrained layout (rect = left, bottom, width, height)
        engine.set(rect=(0.0, bottom, 1.0, 1.0 - bottom))
    if title:   # constrained layout reserves suptitle space itself
        fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="semibold")
    if lines:
        fig.text(0.01, 0.005, "\n".join(lines), fontsize=9, color=INK["muted"], ha="left", va="bottom")


def empty_panel(ax, message: str) -> None:
    """Explicit 'no data' panel (never a silently blank axis)."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=INK["muted"], fontsize=11,
            wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def equal_aspect_panels(axes: Sequence, xs: Sequence[np.ndarray], ys: Sequence[np.ndarray], pad: float = 0.05) -> None:
    """Identical x/y limits on every panel and a 1:1 aspect WITHOUT shared axes (export-safe)."""
    xs_all = np.concatenate([np.asarray(x, float) for x in xs if len(x)]) if any(len(x) for x in xs) else np.array([0.0])
    ys_all = np.concatenate([np.asarray(y, float) for y in ys if len(y)]) if any(len(y) for y in ys) else np.array([0.0])
    xs_all, ys_all = xs_all[np.isfinite(xs_all)], ys_all[np.isfinite(ys_all)]
    x0, x1 = (float(xs_all.min()), float(xs_all.max())) if xs_all.size else (0.0, 1.0)
    y0, y1 = (float(ys_all.min()), float(ys_all.max())) if ys_all.size else (0.0, 1.0)
    span = max(x1 - x0, y1 - y0, 1e-6)
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, span * (0.5 + pad)
    for ax in axes:
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal", adjustable="box")


def ci_errorbar(ax, x, est, lo, hi, **kw) -> None:
    """Point estimate with an asymmetric CI; NaN bounds are drawn without whiskers (never clipped)."""
    est, lo, hi = (np.asarray(v, float) for v in (est, lo, hi))
    err = np.vstack([np.where(np.isfinite(lo), est - lo, 0.0), np.where(np.isfinite(hi), hi - est, 0.0)])
    ax.errorbar(x, est, yerr=np.clip(err, 0, None), fmt=kw.pop("fmt", "o"), capsize=kw.pop("capsize", 3), **kw)
