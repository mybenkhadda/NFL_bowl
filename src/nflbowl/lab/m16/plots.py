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


def target_coverage(cov: Any) -> tuple[Any, dict[str, Any]]:
    """Athletes with a value per target and position (stacked by position; total printed)."""
    import matplotlib.pyplot as plt
    import polars as pl

    c = cov.filter(pl.col("position_group") != "ALL")
    tids = list(dict.fromkeys(c["target_id"].to_list()))
    pos = sorted(set(c["position_group"].to_list()))
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    bottom = np.zeros(len(tids))
    for k, p in enumerate(pos):
        v = np.array([float(c.filter((pl.col("target_id") == t) & (pl.col("position_group") == p))["n_athletes"].sum()) for t in tids])
        ax.bar(range(len(tids)), v, bottom=bottom, color=PALETTE[k % len(PALETTE)], label=p, width=0.7, edgecolor="white", linewidth=1)
        bottom += v
    for i, t in enumerate(bottom):
        ax.text(i, t + 2, f"{int(t)}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(tids)))
    ax.set_xticklabels(tids)
    _style(ax)
    ax.set_ylabel("athletes with the target", color=INK)
    ax.set_title("NFL target coverage by position (minimum-exposure rules applied)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=len(pos))
    fig.tight_layout()
    return fig, {"coverage": c}


def reliability_vs_exposure(curve: Any) -> tuple[Any, dict[str, Any]]:
    """Split-half reliability (Spearman–Brown) vs k observations per half, one line per target; 0.7 reference."""
    import matplotlib.pyplot as plt
    import polars as pl

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    for k, t in enumerate(sorted(set(curve["target_id"].to_list()))):
        g = curve.filter(pl.col("target_id") == t).sort("k_per_half")
        ax.plot(g["k_per_half"], g["reliability"], marker="o", color=PALETTE[k % len(PALETTE)], linewidth=2, label=t)
    ax.axhline(0.7, color=INK_2, linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    _style(ax)
    ax.set_xlabel("observations per half (windows or events)", color=INK)
    ax.set_ylabel("split-half reliability (Spearman–Brown)", color=INK)
    ax.set_title("Target reliability vs exposure (position-residualised)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout()
    return fig, {"curve": curve}


def ladder_forest(paired: Any, reference: str = "M1", margin_pct: float = 2.0) -> tuple[Any, dict[str, Any]]:
    """ΔRMSE (reference − model, positive = model better) with 95 % CI per target × model; shaded ±2 % practical margin."""
    import matplotlib.pyplot as plt
    import polars as pl

    p = paired.filter(pl.col("reference") == reference).sort("target_id", "model")
    labels = [f"{t} · {m}" for t, m in zip(p["target_id"].to_list(), p["model"].to_list(), strict=True)]
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.3 * p.height + 1.4)), facecolor="white")
    y = np.arange(p.height)
    d, lo, hi = (p[c].to_numpy().astype(float) for c in ("delta_rmse", "delta_lo", "delta_hi"))
    marg = margin_pct / 100 * p["rmse_ref"].to_numpy().astype(float)
    for i in range(p.height):
        ax.add_patch(plt.Rectangle((-marg[i], i - 0.4), 2 * marg[i], 0.8, color="#e1e0d9", alpha=0.6, linewidth=0))
    ax.hlines(y, lo, hi, color=PALETTE[0], linewidth=1.6)
    ax.scatter(d, y, color=PALETTE[0], s=18, zorder=3)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    _style(ax)
    ax.set_xlabel(f"ΔRMSE vs {reference} (target units; > 0 = better than {reference})", color=INK)
    ax.set_title(f"Paired out-of-fold comparison vs {reference} (95 % fold-stratified athlete bootstrap; grey = ±{margin_pct:g} % margin)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    return fig, {"paired": p}


def within_position_ranking(rank: Any) -> tuple[Any, dict[str, Any]]:
    """Within-position Spearman (95 % CI) per position × model for one target."""
    import matplotlib.pyplot as plt
    import polars as pl

    r = rank.sort("position_group", "model")
    models = sorted(set(r["model"].to_list()))
    pos = sorted(set(r["position_group"].to_list()))
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    w = 0.8 / max(len(models), 1)
    for k, m in enumerate(models):
        g = r.filter(pl.col("model") == m)
        vals = {p: (s, lo, hi) for p, s, lo, hi in zip(g["position_group"].to_list(), g["spearman"].to_list(), g["spearman_lo"].to_list(), g["spearman_hi"].to_list(),
                                                    strict=True)}
        xs = np.arange(len(pos)) + (k - (len(models) - 1) / 2) * w
        s = np.array([vals.get(p, (np.nan,) * 3)[0] for p in pos], float)
        lo = np.array([vals.get(p, (np.nan,) * 3)[1] for p in pos], float)
        hi = np.array([vals.get(p, (np.nan,) * 3)[2] for p in pos], float)
        ax.errorbar(xs, s, yerr=[np.nan_to_num(s - lo), np.nan_to_num(hi - s)], fmt="o", color=PALETTE[k % len(PALETTE)], capsize=3, label=m)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(pos)))
    ax.set_xticklabels(pos)
    _style(ax)
    ax.set_ylabel("within-position Spearman (prediction, outcome)", color=INK)
    ax.set_title(f"Within-position ranking — target {r['target_id'][0] if 'target_id' in r.columns and r.height else ''}", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"ranking": r}


def power_curves(power: Any) -> tuple[Any, dict[str, Any]]:
    """Power vs planted variance fraction per target (0.8 reference): what this cohort could detect."""
    import matplotlib.pyplot as plt
    import polars as pl

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    for k, t in enumerate(sorted(set(power["target"].to_list()))):
        g = power.filter(pl.col("target") == t).sort("planted_fraction")
        ax.plot(g["planted_fraction"], g["power"], marker="o", color=PALETTE[k % len(PALETTE)], linewidth=2, label=f"{t} (n = {g['n_athletes'][0]})")
    ax.axhline(0.8, color=INK_2, linestyle="--", linewidth=0.8)
    _style(ax)
    ax.set_xlabel("planted share of residual variance (f)", color=INK)
    ax.set_ylabel("power (CI of ΔRMSE above 0)", color=INK)
    ax.set_title("Detectability on the real cohort structure (feature reliability as configured)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig, {"power": power}


__all__ = ["ladder_forest", "metrics_by_fold", "power_curves", "reliability_vs_exposure", "reproduction_diffs", "target_coverage", "within_position_ranking"]
