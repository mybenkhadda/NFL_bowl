"""M16 figures (T04 scope). Each returns ``(figure, data)`` for ``M16Run.figure`` (PNG / SVG / PDF + source data)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..m15.plots import INK, INK_2, PALETTE, _style


def reproduction_diffs(diffs: Any, tol: float) -> tuple[Any, dict[str, Any]]:
    """|Δ prediction| per athlete for each comparison (log scale; tolerance line). Athletes present in one source only
    are counted in the title, not plotted (no value to plot)."""
    import matplotlib.pyplot as plt
    import polars as pl

    d = diffs.filter(pl.col("abs_dpred").is_not_null())
    floor = 1e-17                                                     # exact zeros drawn at the floor (stated in the caption)
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    labels = sorted(set(d["comparison"].to_list()))
    for k, lab in enumerate(labels):
        g = d.filter(pl.col("comparison") == lab)
        y = np.maximum(g["abs_dpred"].to_numpy(), floor)
        x = np.full(y.size, k) + np.random.default_rng(k).uniform(-0.18, 0.18, y.size)
        ax.scatter(x, y, s=10, color=PALETTE[k % len(PALETTE)], alpha=0.75, edgecolor="none")
    ax.axhline(tol, color=INK_2, linestyle="--", linewidth=1)
    ax.text(len(labels) - 0.5, tol, f" tolerance {tol:g}", va="bottom", ha="right", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lab.replace(" vs ", "\nvs ") for lab in labels], fontsize=7)
    _style(ax)
    ax.set_ylabel("|Δ prediction| per athlete (yd/s; log)", color=INK)
    n_only = int(diffs.filter(pl.col("abs_dpred").is_null()).height)
    ax.set_title(f"Official M1 reproduction — {d.height:,} athlete comparisons; {n_only} in one source only", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig, {"athlete_diffs": d.select("comparison", "column", "nfl_id", "fold_ref", "abs_dpred", "abs_dy", "mismatch")}


def metrics_by_fold(metrics: Any) -> tuple[Any, dict[str, Any]]:
    """RMSE per fold for each (source, model) — descriptive companion of the prediction-level equality checks."""
    import matplotlib.pyplot as plt
    import polars as pl

    m = metrics.filter(pl.col("fold") != "all")
    keys = m.select("source", "model").unique(maintain_order=True).rows()
    folds = sorted(set(m["fold"].to_list()), key=int)
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    w = 0.8 / max(len(keys), 1)
    for k, (src, mod) in enumerate(keys):
        g = m.filter((pl.col("source") == src) & (pl.col("model") == mod))
        val = dict(zip(g["fold"].to_list(), g["rmse"].to_list(), strict=True))
        xs = np.arange(len(folds)) + (k - (len(keys) - 1) / 2) * w
        ax.bar(xs, [val.get(f, np.nan) for f in folds], width=w * 0.9, color=PALETTE[k % len(PALETTE)], label=f"{mod} · {src}")
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"fold {f}" for f in folds])
    _style(ax)
    ax.set_ylabel("RMSE (yd/s)", color=INK)
    ax.set_title("Out-of-fold RMSE by fold — target nfl.q90_shared", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig, {"rmse_by_fold": m}


__all__ = ["metrics_by_fold", "reproduction_diffs"]
