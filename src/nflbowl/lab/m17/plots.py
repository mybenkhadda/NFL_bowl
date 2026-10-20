"""M17 figures — each returns (figure, source data) for ``M17Run.figure`` (PNG / SVG / PDF + CSV / Parquet)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..m15.plots import INK, INK_2, PALETTE, _style, gate_matrix

__all__ = ["alignment_tradeoff", "caterpillar", "combine_vs_nfl", "drill_vs_pooled_icc", "failure_bars", "gate_matrix", "reliability_vs_k",
           "stratum_coverage", "variance_stacked"]


def failure_bars(fm: Any) -> tuple[Any, dict[str, Any]]:
    """Share of each detector flag per domain × kind."""
    import matplotlib.pyplot as plt

    cols = [c for c in fm.columns if c.startswith("share_")]
    rows = [f"{d} · {k}" for d, k in zip(fm["domain"].to_list(), fm["kind"].to_list(), strict=True)]
    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.32 * len(rows) + 1.5)), facecolor="white")
    y = np.arange(len(rows))
    h = 0.8 / max(len(cols), 1)
    for k, c in enumerate(cols):
        ax.barh(y + (k - (len(cols) - 1) / 2) * h, fm[c].to_numpy(), height=h * 0.9, color=PALETTE[k % len(PALETTE)], label=c.removeprefix("share_"))
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=7)
    ax.invert_yaxis()
    _style(ax)
    ax.set_xlabel("share of detected events", color=INK)
    ax.set_title("M14 detector flags by domain and kind (why events fail completeness)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    return fig, {"failure_modes": fm}


def drill_vs_pooled_icc(icc: Any) -> tuple[Any, dict[str, Any]]:
    """Per metric: ICC (95 % CI) per drill and pooled agility; lines at 0.4 / 0.6."""
    import matplotlib.pyplot as plt
    import polars as pl

    d = icc.filter(pl.col("enough") & pl.col("icc1").is_finite()).sort("metric", "scope")
    labels = [f"{m} · {s}" for m, s in zip(d["metric"].to_list(), d["scope"].to_list(), strict=True)]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.25 * d.height + 1.5)), facecolor="white")
    y = np.arange(d.height)
    v, lo, hi = (d[c].to_numpy().astype(float) for c in ("icc1", "icc_lo", "icc_hi"))
    col = [PALETTE[1] if s == "agility_pooled" else PALETTE[0] for s in d["scope"].to_list()]
    ax.hlines(y, np.where(np.isfinite(lo), lo, v), np.where(np.isfinite(hi), hi, v), color=col, linewidth=1.5)
    ax.scatter(v, y, color=col, s=16, zorder=3)
    for ref in (0.4, 0.6):
        ax.axvline(ref, color=INK_2, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    _style(ax)
    ax.set_xlabel("ICC(1,1) across Combine attempts (blue: one drill; orange: pooled agility)", color=INK)
    ax.set_title("Event-metric repeatability: per drill vs pooled", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    return fig, {"icc": d}


def stratum_coverage(cov: Any) -> tuple[Any, dict[str, Any]]:
    """Athletes per stratum in each domain; supported strata marked."""
    import matplotlib.pyplot as plt

    c = cov.sort("stratum")
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.3 * c.height + 1.5)), facecolor="white")
    y = np.arange(c.height)
    ax.barh(y - 0.2, c["athletes_combine"].to_numpy(), height=0.38, color=PALETTE[0], label="Combine athletes")
    ax.barh(y + 0.2, c["athletes_nfl"].to_numpy(), height=0.38, color=PALETTE[1], label="NFL athletes")
    for i, s in enumerate(c["supported"].to_list()):
        if s:
            ax.text(0, i, "  supported", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(c["stratum"].to_list(), fontsize=7)
    ax.invert_yaxis()
    _style(ax)
    ax.set_xlabel("athletes with ≥ 2 events in the stratum", color=INK)
    ax.set_title("Context strata coverage (entry speed × turn class)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"coverage": c}


def variance_stacked(summary: Any) -> tuple[Any, dict[str, Any]]:
    """Context / athlete / event-noise shares of total variance per metric × domain."""
    import matplotlib.pyplot as plt
    import polars as pl

    s = summary.with_columns((pl.col("var_context") / pl.col("var_total")).alias("context"), (pl.col("var_u") / pl.col("var_total")).alias("athlete"),
                             (pl.col("var_e") / pl.col("var_total")).alias("noise")).sort("metric", "domain")
    labels = [f"{m} · {d}" for m, d in zip(s["metric"].to_list(), s["domain"].to_list(), strict=True)]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.3 * s.height + 1.5)), facecolor="white")
    left = np.zeros(s.height)
    for k, c in enumerate(("athlete", "context", "noise")):
        v = np.clip(np.nan_to_num(s[c].to_numpy().astype(float)), 0, None)
        ax.barh(range(s.height), v, left=left, color=PALETTE[k], label=c, height=0.7)
        left += v
    ax.set_yticks(range(s.height))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    _style(ax)
    ax.set_xlabel("share of total variance (method per row in the table)", color=INK)
    ax.set_title("Variance components of event metrics (median over folds)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    return fig, {"shares": s.select("metric", "domain", "athlete", "context", "noise", "method")}


def reliability_vs_k(summary: Any) -> tuple[Any, dict[str, Any]]:
    import matplotlib.pyplot as plt
    import polars as pl

    ks = sorted(int(c.removeprefix("reliability_k")) for c in summary.columns if c.startswith("reliability_k"))
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    long = []
    for i, r in enumerate(summary.filter(pl.col("domain") == "combine").iter_rows(named=True)):
        v = [r[f"reliability_k{k}"] for k in ks]
        ax.plot(ks, v, marker="o", color=PALETTE[i % len(PALETTE)], linewidth=1.6, label=r["metric"])
        long += [{"metric": r["metric"], "k": k, "reliability": x} for k, x in zip(ks, v, strict=True)]
    ax.axhline(0.7, color=INK_2, linestyle="--", linewidth=0.8)
    ax.set_xscale("log", base=2)
    _style(ax)
    ax.set_xlabel("events per athlete (k)", color=INK)
    ax.set_ylabel("reliability of the athlete mean", color=INK)
    ax.set_title("How many Combine events are needed for a reliable context-adjusted athlete effect?", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig, {"curve": pl.DataFrame(long)}


def caterpillar(effects: Any, metric: str, domain: str = "combine") -> tuple[Any, dict[str, Any]]:
    """Shrunk athlete effects (BLUP) sorted within position."""
    import matplotlib.pyplot as plt
    import polars as pl

    e = effects.filter((pl.col("metric") == metric) & (pl.col("domain") == domain)).sort("position_group", "blup")
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor="white")
    x0 = 0
    for k, (p,) in enumerate(e.select("position_group").unique(maintain_order=True).rows()):
        g = e.filter(pl.col("position_group") == p)
        ax.scatter(np.arange(x0, x0 + g.height), g["blup"].to_numpy(), s=6, color=PALETTE[k % len(PALETTE)], label=f"{p} (n = {g.height})")
        x0 += g.height + 10
    ax.axhline(0, color=INK_2, linewidth=0.8)
    _style(ax)
    ax.set_xticks([])
    ax.set_ylabel(f"context-adjusted athlete effect: {metric}", color=INK)
    ax.set_title(f"Shrunk athlete effects ({domain}), sorted within position — parameters from other folds", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=5)
    fig.tight_layout()
    return fig, {"effects": e}


def combine_vs_nfl(effects: Any, cross: Any, metric: str) -> tuple[Any, dict[str, Any]]:
    import matplotlib.pyplot as plt
    import polars as pl

    c = effects.filter((pl.col("metric") == metric) & (pl.col("domain") == "combine")).select("nfl_id", "position_group", pl.col("blup").alias("combine"))
    n = effects.filter((pl.col("metric") == metric) & (pl.col("domain") == "nfl")).select("nfl_id", pl.col("blup").alias("nfl"))
    j = c.join(n, on="nfl_id").drop_nulls()
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="white")
    for k, p in enumerate(sorted(set(j["position_group"].to_list()))):
        g = j.filter(pl.col("position_group") == p)
        ax.scatter(g["combine"], g["nfl"], s=10, color=PALETTE[k % len(PALETTE)], alpha=0.6, label=p)
    r = cross.filter(pl.col("metric") == metric) if cross is not None and cross.height else None
    txt = f"r = {r['r'][0]:.2f} [{r['r_lo'][0]:.2f}, {r['r_hi'][0]:.2f}], disattenuated {r['r_disattenuated'][0]:.2f}" if r is not None and r.height and "r" in r.columns else ""
    _style(ax)
    ax.set_xlabel("Combine athlete effect", color=INK)
    ax.set_ylabel("NFL athlete effect", color=INK)
    ax.set_title(f"{metric}: Combine vs NFL (within-position) — {txt}", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig, {"pairs": j}


def alignment_tradeoff(summary: Any, damage: Any) -> tuple[Any, dict[str, Any]]:
    """Same-position MRR vs MMD² per arm (fold means); destructive arms outlined."""
    import matplotlib.pyplot as plt
    import polars as pl

    s = summary.filter(pl.col("status") == "COMPLETED").group_by("rep", "arm").agg(pl.col("same_position_mrr").mean(), pl.col("mmd2").mean())
    if damage is not None and damage.height:
        s = s.join(damage.group_by("rep", "arm").agg(pl.col("destructive").any()), on=["rep", "arm"], how="left")
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")
    arms = sorted(set(s["arm"].to_list()))
    for k, a in enumerate(arms):
        g = s.filter(pl.col("arm") == a)
        ax.scatter(g["mmd2"], g["same_position_mrr"], s=50, color=PALETTE[k % len(PALETTE)], label=a, edgecolor=INK if "destructive" in g.columns and g["destructive"].any() else "none")
    _style(ax)
    ax.set_xlabel("MMD² (held-out mapped Combine vs NFL; lower = closer distributions)", color=INK)
    ax.set_ylabel("same-position MRR (held-out)", color=INK)
    ax.set_title("Alignment trade-off: distribution match vs athlete identification", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"arms": s}
