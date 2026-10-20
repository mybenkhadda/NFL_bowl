"""M18 figures (publication set) — each returns (figure, source data)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..m15.plots import INK, INK_2, PALETTE, _style, gate_matrix
from ..m16.plots import ladder_forest

VERDICT_COLOR = {"scouting_relevant": "#0ca30c", "promising_needs_replication": "#fab219", "precise_null": "#2a78d6", "inconclusive": "#898781", "blocked": "#e1e0d9"}

__all__ = ["controls_panel", "executive_figure", "fold_lines", "gate_matrix", "ladder_forest", "specificity_heatmap", "twin_scatter"]


def fold_lines(folds: Any) -> tuple[Any, dict[str, Any]]:
    """ΔRMSE vs M1 per outer fold for every rung (fold stability)."""
    import matplotlib.pyplot as plt
    import polars as pl

    f = folds.filter(pl.col("model") != "M1").sort("model", "fold")
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    for k, (m,) in enumerate(f.select("model").unique(maintain_order=True).rows()):
        g = f.filter(pl.col("model") == m)
        ax.plot(g["fold"], g["delta_rmse_vs_M1"], marker="o", color=PALETTE[k % len(PALETTE)], linewidth=1.6, label=m)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    _style(ax)
    ax.set_xlabel("outer fold", color=INK)
    ax.set_ylabel("ΔRMSE vs M1 (> 0 = better)", color=INK)
    ax.set_title("Fold-by-fold stability of each rung", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig, {"folds": f}


def specificity_heatmap(long: Any) -> tuple[Any, dict[str, Any]]:
    """Drill feature × NFL task target partial Spearman (M1- and position-controlled); matched cells outlined."""
    import matplotlib.pyplot as plt
    import polars as pl

    feats = list(dict.fromkeys(long.sort("drill_family", "feature")["feature"].to_list()))
    tids = sorted(set(long["target_id"].to_list()))
    M = np.full((len(feats), len(tids)), np.nan)
    match = np.zeros_like(M, bool)
    for r in long.iter_rows(named=True):
        i, j = feats.index(r["feature"]), tids.index(r["target_id"])
        M[i, j], match[i, j] = r["partial_spearman"], r["matched"]
    lim = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.2 * len(tids) + 4, 0.32 * len(feats) + 1.8), facecolor="white")
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(feats)):
        for j in range(len(tids)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color=INK)
            if match[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.2))
    ax.set_xticks(range(len(tids)))
    ax.set_xticklabels(tids)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=6)
    ax.set_title("Drill → NFL task specificity (partial Spearman | M1 + position); outlined = pre-declared match", fontsize=10, color=INK, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig, {"matrix": long.select("feature", "drill_family", "target_id", "matched", "partial_spearman", "lo", "hi", "n")}


def controls_panel(controls: Any, rlabel: Any, perm: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Negative controls and random-label ΔRMSE (should straddle 0) + the permutation reading."""
    import matplotlib.pyplot as plt
    import polars as pl

    parts = [d.with_columns(pl.lit(src).alias("source")) for d, src in ((controls, "control"), (rlabel, "random label")) if d is not None and d.height]
    d = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * d.height + 1.5)), facecolor="white")
    if d.height:
        y = np.arange(d.height)
        ax.hlines(y, d["delta_lo"].to_numpy(), d["delta_hi"].to_numpy(), color=PALETTE[0], linewidth=1.6)
        ax.scatter(d["delta_rmse"].to_numpy(), y, color=PALETTE[0], s=18, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{s}: {m}" for s, m in zip(d["source"].to_list(), d["model"].to_list(), strict=True)], fontsize=7)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    _style(ax)
    ax.set_xlabel("ΔRMSE vs M1 (controls must not be > 0)", color=INK)
    ax.set_title(f"Negative controls and sanity — permutation p = {perm.get('p_value', 'n/a')} ({perm.get('model', '')})", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    return fig, {"controls": d}


def twin_scatter(long: Any) -> tuple[Any, dict[str, Any]]:
    """Per twin pair: Δcomposite vs Δoutcome (higher-composite twin − lower)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="white")
    if long is not None and long.height and "delta_y" in long.columns:
        ax.scatter(long["delta_composite"], long["delta_y"], s=14, color=PALETTE[0], alpha=0.7)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    _style(ax)
    ax.set_xlabel("composite difference (within-position SD)", color=INK)
    ax.set_ylabel("NFL outcome difference (higher − lower composite)", color=INK)
    ax.set_title("Stopwatch twins: does the tracking composite separate same-stopwatch prospects?", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    return fig, {"pairs": long}


def executive_figure(evidence: Any, paired: Any, target_label: str) -> tuple[Any, dict[str, Any]]:
    """One-page figure: ΔRMSE vs M1 per rung with CI, coloured by the G18 verdict (blocked rungs listed)."""
    import matplotlib.pyplot as plt
    import polars as pl

    e = evidence.sort("rung")
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    y = np.arange(e.height)
    for i, r in enumerate(e.iter_rows(named=True)):
        col = VERDICT_COLOR.get(r["verdict"], "#898781")
        if r.get("delta_rmse") is not None:
            ax.hlines(i, r["delta_lo"], r["delta_hi"], color=col, linewidth=4)
            ax.scatter([r["delta_rmse"]], [i], color=col, s=120, zorder=3, edgecolor=INK)
        ax.text(1.01, i, f"{r['verdict']}", transform=ax.get_yaxis_transform(), va="center", fontsize=12, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['rung']}  {r['model']}" for r in e.iter_rows(named=True)], fontsize=12)
    ax.invert_yaxis()
    _style(ax)
    ax.tick_params(labelsize=11)
    ax.set_xlabel(f"Improvement over official Combine testing (ΔRMSE vs M1, {target_label}); right = better", color=INK, fontsize=12)
    ax.set_title("Does Combine movement tracking add NFL information beyond the stopwatch? (out-of-fold, athlete-disjoint, 95 % CI)", fontsize=15,
                 color=INK, loc="left")
    fig.subplots_adjust(left=0.22, right=0.82)
    return fig, {"evidence": e, "paired": paired}
