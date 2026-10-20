"""M16 figures (T04 scope). Each returns ``(figure, data)`` for ``M16Run.figure`` (PNG / SVG / PDF + source data)."""

from __future__ import annotations

from typing import Any, Sequence

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


# ====================================================================== M16 laboratory figures (16:9, white, CIs, n in titles)
W16 = (12.0, 6.75)
STATUS_COLOR = {"PASS": "#0ca30c", "FAIL": "#d03b3b", "EXPLORATORY": "#eda100", "BLOCKED": "#8a8986", "NOT_EVALUATED": "#e1e0d9",
                "VERIFIED_FOLD_SAFE": "#0ca30c", "EXPLORATORY_ONLY": "#eda100", "UNKNOWN_PROVENANCE": "#8a8986", "INVALID_LEAKAGE": "#d03b3b",
                "COMPLETED": "#0ca30c", "BLOCKED_INSUFFICIENT_OVERLAP": "#d03b3b"}


def _fig(nrows: int = 1, ncols: int = 1, **kw: Any) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    return plt.subplots(nrows, ncols, figsize=kw.pop("figsize", W16), facecolor="white", **kw)


def _title(ax: Any, text: str, size: int = 11) -> None:
    ax.set_title(text, fontsize=size, color=INK, loc="left")


def cohort_funnel(funnel: Any) -> tuple[Any, dict[str, Any]]:
    """Athletes remaining at each cohort step (bars, n printed), position split stacked."""
    import polars as pl

    f = funnel.sort("order")
    pos = [c for c in f.columns if c.startswith("n_") and c != "n_athletes"]
    fig, ax = _fig()
    y = np.arange(f.height)
    left = np.zeros(f.height)
    for k, c in enumerate(pos):
        v = f[c].to_numpy().astype(float)
        ax.barh(y, v, left=left, color=PALETTE[k % len(PALETTE)], height=0.6, label=c.removeprefix("n_"), edgecolor="white", linewidth=0.8)
        left += v
    for i, n in enumerate(f["n_athletes"].to_list()):
        ax.text(n + max(f["n_athletes"].to_list()) * 0.01, i, f"{n:,}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(f["step"].to_list(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(f["n_athletes"].to_list()) * 1.12)
    _style(ax)
    ax.set_xlabel("athletes", color=INK)
    _title(ax, "Cohort funnel — from the sequence index to the cohort where prediction and retrieval can both speak")
    ax.legend(frameon=False, fontsize=8, ncol=len(pos), loc="lower right")
    fig.tight_layout()
    return fig, {"funnel": f}


def heat(df: Any, row: str, col: str, val: str, title: str, fmt: str = "{:.0f}", cmap: str = "Blues", vmin: float | None = None, vmax: float | None = None,
         annotate: Any = None) -> tuple[Any, dict[str, Any]]:
    """Generic annotated heatmap (row × col → val); ``annotate`` = optional second value column printed under the value."""
    import polars as pl

    rows = list(dict.fromkeys(df[row].cast(pl.String).to_list()))
    cols = list(dict.fromkeys(df[col].cast(pl.String).to_list()))
    M = np.full((len(rows), len(cols)), np.nan)
    A = np.full((len(rows), len(cols)), np.nan)
    for r in df.iter_rows(named=True):
        i, j = rows.index(str(r[row])), cols.index(str(r[col]))
        M[i, j] = r[val] if r[val] is not None else np.nan
        if annotate:
            A[i, j] = r[annotate] if r[annotate] is not None else np.nan
    fig, ax = _fig(figsize=(max(7.0, 1.0 * len(cols) + 3), max(3.5, 0.45 * len(rows) + 1.8)))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                txt = fmt.format(M[i, j]) + (f"\nn={int(A[i, j])}" if annotate and np.isfinite(A[i, j]) else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    _title(ax, title, 10)
    fig.tight_layout()
    return fig, {"matrix": df}


def exposure_distribution(expo: Any) -> tuple[Any, dict[str, Any]]:
    """NFL windows per athlete by position (log axis; median marked)."""
    import polars as pl

    fig, ax = _fig()
    pos = sorted(set(expo["position_group"].drop_nulls().to_list()))
    bins = np.logspace(0, np.log10(max(2.0, float(expo["n_nfl_windows"].max()))), 30)
    for k, p in enumerate(pos):
        v = expo.filter(pl.col("position_group") == p)["n_nfl_windows"].to_numpy()
        ax.hist(v, bins=bins, histtype="step", linewidth=1.8, color=PALETTE[k % len(PALETTE)], label=f"{p} (n = {v.size}, median {np.median(v):.0f})")
    ax.set_xscale("log")
    _style(ax)
    ax.set_xlabel("NFL tracking windows per athlete (log)", color=INK)
    ax.set_ylabel("athletes", color=INK)
    _title(ax, f"NFL exposure per athlete by position — {expo.height} athletes")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"exposure": expo}


def target_violins(long: Any, max_targets: int = 8) -> tuple[Any, dict[str, Any]]:
    """Athlete-level distribution of every target by position (violin + median; n under each violin)."""
    import matplotlib.pyplot as plt
    import polars as pl

    tids = list(dict.fromkeys(long["target_id"].to_list()))[:max_targets]
    nc = 4
    nr = int(np.ceil(len(tids) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(14, 3.6 * nr), facecolor="white", squeeze=False)
    pos = sorted(set(long["position_group"].drop_nulls().to_list()))
    for a, tid in zip(axes.ravel(), tids):
        d = long.filter(pl.col("target_id") == tid)
        data = [d.filter(pl.col("position_group") == p)["value"].to_numpy() for p in pos]
        ok = [k for k, x in enumerate(data) if x.size >= 3]
        if ok:
            parts = a.violinplot([data[k] for k in ok], positions=ok, showmedians=True, widths=0.8)
            for k, b in zip(ok, parts["bodies"]):
                b.set_facecolor(PALETTE[k % len(PALETTE)])
                b.set_alpha(0.45)
        for k, x in enumerate(data):
            a.text(k, a.get_ylim()[0], f"n={x.size}", ha="center", va="bottom", fontsize=6, color=INK_2)
        a.set_xticks(range(len(pos)))
        a.set_xticklabels(pos, fontsize=7)
        _style(a)
        unit = d["unit"][0] if d.height and "unit" in d.columns else ""
        a.set_ylabel(unit, color=INK, fontsize=8)
        _title(a, f"{tid} ({d.height} athletes)", 9)
    for a in axes.ravel()[len(tids):]:
        a.axis("off")
    fig.suptitle("NFL targets by position — one value per athlete", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig, {"values": long.filter(pl.col("target_id").is_in(tids))}


def target_kde(kde: Any, unit: str) -> tuple[Any, dict[str, Any]]:
    import polars as pl

    fig, ax = _fig()
    for k, p in enumerate(list(dict.fromkeys(kde["position_group"].to_list()))):
        g = kde.filter(pl.col("position_group") == p)
        ax.plot(g["x"], g["density"], linewidth=2.4 if p == "ALL" else 1.6, color=INK if p == "ALL" else PALETTE[k % len(PALETTE)],
                linestyle="--" if p == "ALL" else "-", label=f"{p} (n = {g['n'][0]})")
    _style(ax)
    ax.set_xlabel(unit, color=INK)
    ax.set_ylabel("density", color=INK)
    ax.set_ylim(bottom=0)
    _title(ax, f"{kde['target_id'][0]} — kernel density (Scott bandwidth), overall and by position")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"kde": kde}


def variance_bars(vd: Any) -> tuple[Any, dict[str, Any]]:
    """Between-position share η² of each target (athlete-bootstrap CI): the rest is within-position spread."""
    fig, ax = _fig()
    v = vd.sort("target_id")
    x = np.arange(v.height)
    e = v["eta2_position"].to_numpy()
    ax.bar(x, e, color=PALETTE[0], width=0.6, label="between positions (η²)")
    ax.bar(x, 1 - e, bottom=e, color=GRID_C, width=0.6, label="within positions")
    ax.errorbar(x, e, yerr=[np.nan_to_num(e - v["eta2_lo"].to_numpy()), np.nan_to_num(v["eta2_hi"].to_numpy() - e)], fmt="none", ecolor=INK, capsize=3)
    for i, n in enumerate(v["n_athletes"].to_list()):
        ax.text(i, 1.01, f"n={n}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(v["target_id"].to_list())
    ax.set_ylim(0, 1.08)
    _style(ax)
    ax.set_ylabel("share of athlete-level variance", color=INK)
    _title(ax, "Target variance decomposition — only the within-position share can reward an athlete-specific Combine measurement")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig, {"variance": v}


GRID_C = "#e1e0d9"


def lines_by_group(df: Any, x: str, y: str, group: str, title: str, xlabel: str, ylabel: str, lo: str | None = None, hi: str | None = None,
                   ref: float | None = None, logx: bool = False) -> tuple[Any, dict[str, Any]]:
    """Generic line plot with optional CI band and reference line."""
    import polars as pl

    fig, ax = _fig()
    for k, gname in enumerate(list(dict.fromkeys(df[group].cast(pl.String).to_list()))):
        g = df.filter(pl.col(group).cast(pl.String) == gname).sort(x)
        ax.plot(g[x], g[y], marker="o", linewidth=2, color=PALETTE[k % len(PALETTE)], label=gname)
        if lo and hi and lo in g.columns and hi in g.columns:
            ax.fill_between(g[x].to_numpy().astype(float), g[lo].to_numpy().astype(float), g[hi].to_numpy().astype(float), color=PALETTE[k % len(PALETTE)], alpha=0.15)
    if ref is not None:
        ax.axhline(ref, color=INK_2, linestyle="--", linewidth=0.8)
    if logx:
        ax.set_xscale("log")
    _style(ax)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    _title(ax, title)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout()
    return fig, {"lines": df}


def overlap_profile(uni: Any) -> tuple[Any, dict[str, Any]]:
    """Per drill × covariate: Combine 5–95 % range vs NFL 5–95 % range (bars) with medians; OVL printed."""
    import matplotlib.pyplot as plt
    import polars as pl

    drills = list(dict.fromkeys(uni["drill"].to_list()))
    covs = list(dict.fromkeys(uni["covariate"].to_list()))
    fig, axes = plt.subplots(1, len(covs), figsize=(max(12, 2.6 * len(covs)), 6.75), facecolor="white", squeeze=False)
    for a, c in zip(axes[0], covs):
        g = uni.filter(pl.col("covariate") == c)
        for i, d in enumerate(drills):
            r = g.filter(pl.col("drill") == d)
            if not r.height:
                continue
            r = r.row(0, named=True)
            a.plot([i - 0.12] * 2, [r["combine_q05"], r["combine_q95"]], color=PALETTE[0], linewidth=5, solid_capstyle="butt")
            a.plot([i + 0.12] * 2, [r["nfl_q05"], r["nfl_q95"]], color=PALETTE[1], linewidth=5, solid_capstyle="butt")
            a.scatter([i - 0.12, i + 0.12], [r["combine_median"], r["nfl_median"]], color=INK, s=12, zorder=3)
            a.text(i, max(r["combine_q95"], r["nfl_q95"]), f"OVL {r['ovl']:.2f}" if r["ovl"] == r["ovl"] else "OVL n/a", ha="center", va="bottom", fontsize=7)
        a.set_xticks(range(len(drills)))
        a.set_xticklabels(drills, rotation=20, fontsize=7)
        _style(a)
        _title(a, c, 9)
    axes[0][0].plot([], [], color=PALETTE[0], linewidth=5, label="Combine drill 5–95 %")
    axes[0][0].plot([], [], color=PALETTE[1], linewidth=5, label="NFL windows 5–95 %")
    axes[0][0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Movement-context overlap between Combine drills and NFL windows (same positions; OVL = shared histogram mass)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    return fig, {"overlap": uni}


def status_matrix(df: Any, row: str, col: str, status: str, title: str) -> tuple[Any, dict[str, Any]]:
    """Categorical status cells (text written in every cell; colour is redundant with the text)."""
    import matplotlib.pyplot as plt
    import polars as pl

    rows = list(dict.fromkeys(df[row].cast(pl.String).to_list()))
    cols = list(dict.fromkeys(df[col].cast(pl.String).to_list()))
    fig, ax = _fig(figsize=(max(8.0, 1.6 * len(cols) + 3), max(3.0, 0.42 * len(rows) + 1.6)))
    for r in df.iter_rows(named=True):
        i, j = rows.index(str(r[row])), cols.index(str(r[col]))
        s = str(r[status])
        ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, color=STATUS_COLOR.get(s, GRID_C), alpha=0.35))
        ax.text(j, i, s.replace("_", " ").lower(), ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8, rotation=20)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    _title(ax, title, 10)
    fig.tight_layout()
    return fig, {"status": df}


def delta_forest(df: Any, label_cols: Sequence[str], est: str, lo: str, hi: str, xlabel: str, title: str, ref: float = 0.0, color_by: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Generic forest plot (estimate with 95 % CI per row)."""
    import polars as pl

    d = df
    labels = [" · ".join(str(r[c]) for c in label_cols) for r in d.iter_rows(named=True)]
    fig, ax = _fig(figsize=(12, max(4.0, 0.28 * d.height + 1.6)))
    y = np.arange(d.height)
    groups = list(dict.fromkeys(d[color_by].cast(pl.String).to_list())) if color_by else [None]
    col = [PALETTE[groups.index(str(v)) % len(PALETTE)] for v in d[color_by].cast(pl.String).to_list()] if color_by else [PALETTE[0]] * d.height
    e, l, h = (d[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in (est, lo, hi))
    ax.hlines(y, l, h, color=col, linewidth=1.6)
    ax.scatter(e, y, color=col, s=18, zorder=3)
    ax.axvline(ref, color=INK_2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    _style(ax)
    ax.set_xlabel(xlabel, color=INK)
    _title(ax, title, 10)
    if color_by:
        for k, gname in enumerate(groups):
            ax.scatter([], [], color=PALETTE[k % len(PALETTE)], label=gname)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig, {"forest": d}


def retrieval_vs_chance(summary: Any, scope: str = "all", arm: str = "unaligned") -> tuple[Any, dict[str, Any]]:
    """MRR (95 % CI) per representation in each pool with the pool-specific chance tick."""
    import matplotlib.pyplot as plt
    import polars as pl

    s = summary.filter((pl.col("scope") == scope) & (pl.col("arm").is_in([arm, "analytic"])))
    pools = [p for p in ("all", "same_position", "twins", "physical_twins") if s.filter(pl.col("pool") == p).height]
    fig, axes = plt.subplots(1, max(1, len(pools)), figsize=W16, facecolor="white", sharey=True, squeeze=False)
    reps = sorted(set(s["rep"].to_list()))
    for a, p in zip(axes[0], pools):
        g = s.filter(pl.col("pool") == p)
        for i, r in enumerate(reps):
            x = g.filter(pl.col("rep") == r)
            if not x.height:
                continue
            x = x.row(0, named=True)
            lo = x.get("mrr_lo") if x.get("mrr_lo") is not None and x["mrr_lo"] == x["mrr_lo"] else x["mrr"]
            hi = x.get("mrr_hi") if x.get("mrr_hi") is not None and x["mrr_hi"] == x["mrr_hi"] else x["mrr"]
            a.errorbar(x["mrr"], i, xerr=[[max(0.0, x["mrr"] - lo)], [max(0.0, hi - x["mrr"])]], fmt="o", color=PALETTE[i % len(PALETTE)], capsize=3)
            a.scatter(x["chance_mrr"], i, marker="|", s=220, color=INK)
        a.set_xlim(left=0)
        _style(a)
        n = int(g["n_queries"].median()) if g.height else 0
        _title(a, f"{p} (≈{n} queries / rep; median pool {g['median_pool'].median() if g.height else ''})", 9)
        a.set_xlabel("MRR (| = pool chance)", color=INK)
    axes[0][0].set_yticks(range(len(reps)))
    axes[0][0].set_yticklabels(reps, fontsize=8)
    fig.suptitle(f"Five-fold held-out cross-domain retrieval — scope {scope}, arm {arm}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig, {"summary": s}


def rank_histogram(scores: Any, reps: Sequence[str], pool: str, arm: str = "unaligned", scope: str = "all") -> tuple[Any, dict[str, Any]]:
    """Distribution of the true match's relative rank (rank / pool size + 1): uniform under chance."""
    import polars as pl

    s = scores.filter(pl.col("rep").is_in(list(reps)) & (pl.col("pool") == pool) & (pl.col("arm") == arm) & (pl.col("scope") == scope)).with_columns(
        (pl.col("rank") / (pl.col("n_candidates") + 1)).alias("relative_rank"))
    fig, ax = _fig()
    for k, r in enumerate(reps):
        v = s.filter(pl.col("rep") == r)["relative_rank"].to_numpy()
        if v.size:
            ax.hist(v, bins=np.linspace(0, 1, 21), histtype="step", linewidth=1.8, density=True, color=PALETTE[k % len(PALETTE)], label=f"{r} (n = {v.size})")
    ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=0.8)
    _style(ax)
    ax.set_xlabel("relative rank of the true NFL match (0 = first)", color=INK)
    ax.set_ylabel("density (1 = chance)", color=INK)
    _title(ax, f"True-match rank distribution — pool {pool}, arm {arm}, scope {scope}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"ranks": s.select("rep", "fold", "nfl_id", "position_group", "rank", "n_candidates", "relative_rank")}


def pool_sizes(meta: Any) -> tuple[Any, dict[str, Any]]:
    import polars as pl

    fig, ax = _fig()
    pools = [p for p in ("all", "same_position", "twins", "physical_twins") if meta.filter(pl.col("pool") == p).height]
    data = [meta.filter(pl.col("pool") == p)["n_candidates"].to_numpy() for p in pools]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(pools) + 1))
    ax.set_xticklabels([f"{p}\n(n queries {d.size}; empty {(d == 0).sum()})" for p, d in zip(pools, data)], fontsize=8)
    _style(ax)
    ax.set_ylabel("candidate athletes per query", color=INK)
    ax.set_ylim(bottom=0)
    _title(ax, "Candidate-pool sizes per query (chance = 1 / (n + 1) per query; empty pools are not scored)")
    fig.tight_layout()
    return fig, {"pools": meta}


def similarity_distributions(sims: Any, rep: str, arm: str = "unaligned") -> tuple[Any, dict[str, Any]]:
    import polars as pl

    s = sims.filter((pl.col("rep") == rep) & (pl.col("arm") == arm))
    fig, ax = _fig()
    for k, pair in enumerate(("match", "impostor")):
        v = s.filter(pl.col("pair") == pair)["similarity"].to_numpy()
        if v.size:
            ax.hist(v, bins=40, density=True, alpha=0.5, color=PALETTE[k], label=f"{pair} pairs (n = {v.size})")
    _style(ax)
    ax.set_xlabel("cosine similarity (Combine mean vs NFL mean)", color=INK)
    ax.set_ylabel("density", color=INK)
    _title(ax, f"{rep} · {arm} — true-pair vs same-position impostor similarities (held-out athletes)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"similarities": s}


def alignment_slopes(rows: Any, rep: str) -> tuple[Any, dict[str, Any]]:
    """Repeat AUC per fold: unaligned → each aligned arm (one panel per drill); damage threshold annotated."""
    import matplotlib.pyplot as plt
    import polars as pl

    d = rows.filter((pl.col("rep") == rep) & pl.col("kind").is_not_null())
    kinds = sorted(set(d["kind"].to_list()))
    arms = [a for a in ("unaligned", "procrustes", "infonce") if d.filter(pl.col("arm") == a).height]
    fig, axes = plt.subplots(1, max(1, len(kinds)), figsize=W16, facecolor="white", sharey=True, squeeze=False)
    for a, kd in zip(axes[0], kinds):
        g = d.filter(pl.col("kind") == kd)
        for f in sorted(set(g["fold"].to_list())):
            v = [g.filter((pl.col("fold") == f) & (pl.col("arm") == arm))["repeat_auc"].to_list() for arm in arms]
            v = [x[0] if x else np.nan for x in v]
            a.plot(range(len(arms)), v, marker="o", color=PALETTE[int(f) % len(PALETTE)], label=f"fold {f}")
        a.set_xticks(range(len(arms)))
        a.set_xticklabels(arms, fontsize=8)
        a.axhline(0.5, color=INK_2, linestyle=":", linewidth=0.8)
        _style(a)
        _title(a, kd, 9)
    axes[0][0].set_ylabel("repeated-attempt AUC (held-out athletes)", color=INK)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{rep} — does alignment damage repeatability? (each line = one independent held-out fold)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig, {"repeat_auc": d}


def similarity_heatmap(ids: np.ndarray, S: np.ndarray, title: str) -> tuple[Any, dict[str, Any]]:
    import polars as pl

    fig, ax = _fig(figsize=(8, 7))
    im = ax.imshow(S, cmap="viridis", aspect="equal")
    ax.set_xlabel("NFL athlete-mean (held-out)", color=INK)
    ax.set_ylabel("Combine athlete-mean (held-out)", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="cosine similarity")
    _title(ax, title + f" — diagonal = true pairs (n = {ids.size})", 9)
    fig.tight_layout()
    i, j = np.meshgrid(np.arange(ids.size), np.arange(ids.size), indexing="ij")
    return fig, {"similarity": pl.DataFrame({"combine_id": ids[i.ravel()], "nfl_id": ids[j.ravel()], "similarity": S.ravel()})}


def rmse_dots(intervals: Any, target_id: str) -> tuple[Any, dict[str, Any]]:
    """RMSE with 95 % CI per model on the strict cohort of one target (axis starts at 0: no truncation)."""
    import polars as pl

    d = intervals.filter(pl.col("target_id") == target_id).sort("rmse")
    fig, ax = _fig()
    y = np.arange(d.height)
    ax.errorbar(d["rmse"], y, xerr=[d["rmse"] - d["rmse_lo"], d["rmse_hi"] - d["rmse"]], fmt="o", color=PALETTE[0], capsize=3)
    m1 = d.filter(pl.col("model") == "M1")
    if m1.height:
        ax.axvline(float(m1["rmse"][0]), color=PALETTE[1], linestyle="--", linewidth=1, label="M1")
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"].to_list(), fontsize=8)
    ax.set_xlim(left=0)
    _style(ax)
    ax.set_xlabel("out-of-fold RMSE (target units)", color=INK)
    _title(ax, f"{target_id} — RMSE per model, identical held-out athletes (fold-stratified athlete bootstrap)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"rmse": d}


def calibration_plot(bins: Any, models: Sequence[str], target_id: str) -> tuple[Any, dict[str, Any]]:
    import polars as pl

    b = bins.filter(pl.col("model").is_in(list(models)))
    fig, ax = _fig(figsize=(9, 6.75))
    lo = float(min(b["mean_pred"].min(), b["mean_y"].min())) if b.height else 0.0
    hi = float(max(b["mean_pred"].max(), b["mean_y"].max())) if b.height else 1.0
    ax.plot([lo, hi], [lo, hi], color=INK_2, linestyle="--", linewidth=0.8)
    for k, m in enumerate(models):
        g = b.filter(pl.col("model") == m).sort("bin")
        ax.plot(g["mean_pred"], g["mean_y"], marker="o", color=PALETTE[k % len(PALETTE)], label=m)
    _style(ax)
    ax.set_xlabel("mean prediction in decile", color=INK)
    ax.set_ylabel("mean observed outcome in decile", color=INK)
    _title(ax, f"{target_id} — out-of-fold calibration (dashed = perfect)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"calibration": b}


def null_strip(null_rows: Any, summary: Any, target_id: str) -> tuple[Any, dict[str, Any]]:
    """Shuffled-within-position / random / reduced-dimension ΔRMSE vs M1 per rung with the observed gain (diamond)."""
    import polars as pl

    n = null_rows.filter(pl.col("target_id") == target_id)
    s = summary.filter(pl.col("target_id") == target_id)
    models = list(dict.fromkeys(s["model"].to_list()))
    fig, ax = _fig()
    ctl = list(dict.fromkeys(n["control"].to_list()))
    for i, m in enumerate(models):
        for k, c in enumerate(ctl):
            v = n.filter((pl.col("model") == m) & (pl.col("control") == c))["delta_rmse_vs_M1"].to_numpy()
            ax.scatter(np.full(v.size, i + (k - (len(ctl) - 1) / 2) * 0.15), v, s=12, alpha=0.6, color=PALETTE[k % len(PALETTE)], label=c if i == 0 else None)
        o = s.filter(pl.col("model") == m)
        if o.height:
            ax.scatter([i], [float(o["observed_delta_rmse"][0])], marker="D", s=60, color=INK, zorder=4, label="observed" if i == 0 else None)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    _style(ax)
    ax.set_ylabel("ΔRMSE vs M1 (> 0 = better than M1)", color=INK)
    _title(ax, f"{target_id} — negative controls: a gain inside the shuffled cloud is NOT athlete-specific")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"null": n, "observed": s}


def stratified_bars(strat: Any, models: Sequence[str], title: str) -> tuple[Any, dict[str, Any]]:
    import polars as pl

    d = strat.filter(pl.col("model").is_in(list(models)))
    strata = list(dict.fromkeys(d["stratum"].to_list()))
    fig, ax = _fig()
    w = 0.8 / max(len(models), 1)
    for k, m in enumerate(models):
        g = d.filter(pl.col("model") == m)
        val = dict(zip(g["stratum"].to_list(), g["rmse"].to_list(), strict=True))
        ax.bar(np.arange(len(strata)) + (k - (len(models) - 1) / 2) * w, [val.get(s, np.nan) for s in strata], width=w * 0.9, color=PALETTE[k % len(PALETTE)], label=m)
    ns = {s: int(d.filter(pl.col("stratum") == s)["n"][0]) for s in strata}
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([f"{s}\n(n = {ns[s]})" for s in strata], fontsize=8)
    _style(ax)
    ax.set_ylabel("RMSE (target units)", color=INK)
    _title(ax, title)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    return fig, {"strata": d}


def speed_profiles(prof: Any, title: str) -> tuple[Any, dict[str, Any]]:
    """Speed–time curves (one line per labelled attempt / group mean): scouting-oriented, DESCRIPTIVE only."""
    import polars as pl

    fig, ax = _fig()
    for k, lab in enumerate(list(dict.fromkeys(prof["label"].to_list()))):
        g = prof.filter(pl.col("label") == lab).sort("t")
        ax.plot(g["t"], g["speed"], linewidth=1.8, color=PALETTE[k % len(PALETTE)], label=lab)
        if "lo" in g.columns and "hi" in g.columns:
            ax.fill_between(g["t"].to_numpy(), g["lo"].to_numpy(), g["hi"].to_numpy(), color=PALETTE[k % len(PALETTE)], alpha=0.12)
    _style(ax)
    ax.set_xlabel("time from first valid frame (s)", color=INK)
    ax.set_ylabel("speed (yd/s)", color=INK)
    ax.set_ylim(bottom=0)
    _title(ax, title + " — descriptive; not a scouting recommendation")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig, {"profiles": prof}


def gate_extended_matrix(ext: Any) -> tuple[Any, dict[str, Any]]:
    import polars as pl

    d = ext.filter(~pl.col("criterion").str.contains("detectability")).with_columns(pl.concat_str([pl.col("gate"), pl.col("subject")], separator=" · ").alias("row"),
                                                                                     pl.lit("status").alias("col"))
    return status_matrix(d, "row", "col", "status", "G16 (extended) — one row per gate × subject; state written in each cell")


__all__ = ["GRID_C", "STATUS_COLOR", "W16", "alignment_slopes", "calibration_plot", "cohort_funnel", "delta_forest", "exposure_distribution",
           "gate_extended_matrix", "heat", "ladder_forest", "lines_by_group", "metrics_by_fold", "null_strip", "overlap_profile", "pool_sizes",
           "power_curves", "rank_histogram", "reliability_vs_exposure", "reproduction_diffs", "retrieval_vs_chance", "rmse_dots", "similarity_distributions",
           "similarity_heatmap", "speed_profiles", "status_matrix", "stratified_bars", "target_coverage", "target_kde", "target_violins", "variance_bars",
           "within_position_ranking"]
