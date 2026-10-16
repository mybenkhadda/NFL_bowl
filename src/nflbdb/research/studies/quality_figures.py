"""Study A figures — drawn ONLY from persisted Study A tables (``quality.CONTRACTS``).

Every figure is registered with its research question, statistic, grain and filters
(``figure_index.csv``). Distributions are shown as quantile strips (p05–p95 whisker, p25–p75 box,
median dot) so no raw frames are needed to redraw. Intervals are athlete-cluster bootstrap 95 % CIs
unless the panel says otherwise.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK
from ...data.utils import get_logger
from ..figkit import DOMAIN_COLORS, TIER_COLORS, ci_errorbar, empty_panel, equal_aspect_panels, figure, finish
from .config import QualityStudyConfig
from .quality import (A1_EXAMPLES, A1_SIGNALS, A2_EXAMPLES, BY_SPEED, DIMS, EXTREME_ACCEL, FRAME_DIAG, FUNNEL,
                      HIST2D, METRIC_BY_TIER, PROPORTIONS, REASONS, TIERS, VENDOR)
from .store import FigureRecord, StudyStore

log = get_logger(__name__)
SRC = "warehouse/gold/phase_03 M1 datasets → Study A tables"


def _domains(df: pl.DataFrame) -> list[str]:
    order = ["combine", "nfl", "nfl_sample"]
    have = set(df["domain"].unique().to_list()) if df.height else set()
    return [d for d in order if d in have] + sorted(have - set(order))


def _nan(v) -> float:
    return np.nan if v is None else float(v)


def _strip(ax, y: float, r: dict, color: str, height: float = 0.5) -> None:
    from matplotlib.patches import Rectangle

    if r.get("median") is None:
        return
    ax.hlines(y, r["p05"], r["p95"], color=color, lw=1.2)
    ax.add_patch(Rectangle((r["p25"], y - height / 2), r["p75"] - r["p25"], height, color=color, alpha=0.55, lw=0))
    ax.plot(r["median"], y, "o", color=INK["primary"], ms=4)


def _strips_panel(ax, df: pl.DataFrame, label_col: str, color: str | Callable[[dict], str], xlabel: str,
                  center_col: str = "median") -> None:
    rows = list(df.iter_rows(named=True))
    if not rows:
        empty_panel(ax, "no data")
        return
    for i, r in enumerate(rows):
        rr = {**r, "median": r.get(center_col)}
        _strip(ax, i, rr, color(r) if callable(color) else color)
    ax.set_yticks(range(len(rows)), [f"{r[label_col]}  (n={r.get('n_trajectories', r.get('n_frames', 0)):,})"
                                     for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)


def _share_points(ax, d: pl.DataFrame, x_col: str, indicators: list[tuple[str, str, str]]) -> None:
    levels = d[x_col].unique(maintain_order=True).to_list()
    x = np.arange(len(levels))
    for k, (ind, lab, col) in enumerate(indicators):
        s = d.filter(pl.col("indicator") == ind)
        look = {r[x_col]: r for r in s.iter_rows(named=True)}
        est = np.array([look[l]["share"] * 100 if l in look else np.nan for l in levels])
        lo = np.array([_nan(look[l]["boot_lower"]) * 100 if l in look else np.nan for l in levels])
        hi = np.array([_nan(look[l]["boot_upper"]) * 100 if l in look else np.nan for l in levels])
        ci_errorbar(ax, x + (k - (len(indicators) - 1) / 2) * 0.12, est, lo, hi, color=col, label=lab, ms=4)
    ns = {r[x_col]: r["n_trajectories"] for r in d.iter_rows(named=True)}
    ax.set_xticks(x, [f"{l}\nn={ns.get(l, 0):,}" for l in levels], fontsize=8, rotation=0)
    ax.set_ylabel("% of trajectories")
    ax.legend(fontsize=8, frameon=False)


# ============================================================================== A1
def a1_01(o, q):
    p = o[PROPORTIONS.name].filter((pl.col("facet") == "all") & pl.col("indicator").str.starts_with("tier_"))
    fig, ax = figure(1, 2)
    doms = _domains(p)
    for i, dom in enumerate(doms):
        left = 0.0
        for tier in TIERS:
            r = p.filter((pl.col("domain") == dom) & (pl.col("indicator") == f"tier_{tier}"))
            s = float(r["share"][0]) * 100 if r.height else 0.0
            ax[0, 0].barh(i, s, left=left, color=TIER_COLORS[tier], label=tier if i == 0 else None)
            left += s
    ax[0, 0].set_yticks(range(len(doms)), [f"{d}\nn={int(p.filter(pl.col('domain') == d)['n_trajectories'][0]):,}"
                                           for d in doms] if doms else [])
    ax[0, 0].set_xlabel("% of trajectories")
    ax[0, 0].legend(title="tier", frameon=False, fontsize=8)
    ax[0, 0].set_title("Tier composition")
    nona = p.filter(pl.col("indicator") != "tier_A")
    _share_points(ax[0, 1], nona.with_columns(pl.col("domain").alias("lvl")), "lvl",
                  [(f"tier_{t}", f"tier {t}", TIER_COLORS[t]) for t in TIERS[1:]])
    ax[0, 1].set_title("Non-A tiers (zoomed)")
    finish(fig, "Trajectory grain; CIs = athlete-cluster bootstrap. Tiers are PROVISIONAL rule-based labels (M1).",
           title="How are quality tiers distributed?")
    return fig, {"proportions": p}


def a1_02(o, q):
    p = o[PROPORTIONS.name].filter(pl.col("facet") == "cohort")
    doms = _domains(p)
    fig, ax = figure(1, max(1, len(doms)))
    for j, dom in enumerate(doms):
        _share_points(ax[0, j], p.filter(pl.col("domain") == dom).sort("level"), "level",
                      [("tier_B", "tier B", TIER_COLORS["B"]), ("tier_C", "tier C", TIER_COLORS["C"]),
                       ("tier_reject", "reject", TIER_COLORS["reject"]), ("any_flag", "any screening flag", CATEGORICAL[3])])
        ax[0, j].set_title(dom)
    if not doms:
        empty_panel(ax[0, 0], "no data")
    finish(fig, "Cohort = Combine draft year / NFL season. Screening flag = any dimension above its domain × stratum "
                f"p{q.screen_quantile * 100:g} (PROPOSED). CIs: athlete-cluster bootstrap.",
           title="Quality by cohort")
    return fig, {"proportions": p}


def a1_03(o, q):
    p = o[PROPORTIONS.name].filter((pl.col("facet") == "stratum") & (pl.col("domain") == "combine"))
    fig, ax = figure(1, 1, size="tall")
    if p.height:
        _share_points(ax[0, 0], p.sort("level"), "level",
                      [("any_flag", "any screening flag", CATEGORICAL[3]), ("tier_B", "tier B", TIER_COLORS["B"])])
        ax[0, 0].tick_params(axis="x", rotation=60)
    else:
        empty_panel(ax[0, 0], "no Combine trajectories")
    finish(fig, "Combine attempts by drill type. Flags are relative to each drill type's own distribution, so a flag "
                "share near 1 % is expected by construction; departures indicate heavy tails.",
           title="Quality by Combine drill type")
    return fig, {"proportions": p}


def _by_tier_panel(o, metric: str, xlabel: str):
    m = o[METRIC_BY_TIER.name].filter(pl.col("metric") == metric)
    doms = _domains(m)
    fig, ax = figure(1, max(1, len(doms)))
    for j, dom in enumerate(doms):
        d = m.filter(pl.col("domain") == dom).sort(pl.col("quality_tier").replace_strict(
            {t: i for i, t in enumerate(TIERS)}, default=9, return_dtype=pl.Int64))
        _strips_panel(ax[0, j], d, "quality_tier", lambda r: TIER_COLORS.get(r["quality_tier"], INK["muted"]), xlabel)
        ax[0, j].set_title(dom)
    if not doms:
        empty_panel(ax[0, 0], "no data")
    return fig, m


def a1_04(o, q):
    fig, m = _by_tier_panel(o, "speed_vendor_mad", "trajectory median |derived − vendor speed| (yd/s)")
    finish(fig, "Strip: p05–p95 line, p25–p75 box, median dot; trajectory grain. tier B includes the rule "
                "'vendor_speed_disagreement' (MAD > 0.25 yd/s) — separation here is partly by construction.",
           title="Vendor-speed residual by tier")
    return fig, {"metric_by_tier": m}


def a1_05(o, q):
    m = o[METRIC_BY_TIER.name].filter(pl.col("metric") == "max_accel_mag")
    x = o[EXTREME_ACCEL.name]
    fig, ax = figure(1, 2)
    d = m.with_columns((pl.col("domain") + " · " + pl.col("quality_tier")).alias("lab")).sort("domain", "quality_tier")
    _strips_panel(ax[0, 0], d, "lab", lambda r: TIER_COLORS.get(r["quality_tier"], INK["muted"]),
                  "trajectory max |acceleration| (yd/s²)")
    ax[0, 0].axvline(15.0, color=INK["secondary"], ls=":", lw=1)
    ax[0, 0].set_title("Peak acceleration (all trajectories)")
    if x.height:
        xs = x.sort("domain", "quality_tier")
        ci_errorbar(ax[0, 1], np.arange(xs.height), xs["rate"].to_numpy() * 100,
                    xs["boot_lower"].fill_null(np.nan).to_numpy() * 100, xs["boot_upper"].fill_null(np.nan).to_numpy() * 100,
                    color=CATEGORICAL[1])
        ax[0, 1].set_xticks(range(xs.height), [f"{r['domain']}\n{r['quality_tier']}\nn={r['n_trajectories']:,}"
                                               for r in xs.iter_rows(named=True)], fontsize=8)
        ax[0, 1].set_ylabel(f"% of frames with |a| > {q.extreme_accel_yds2:g} yd/s²")
        ax[0, 1].set_title("Extreme-acceleration frame rate (frames available)")
    else:
        empty_panel(ax[0, 1], "no frame data")
    finish(fig, "Dotted line = 15 yd/s² plausibility limit (established; above → reject). Frame rate uses Combine "
                "frames + the deterministic NFL sample; CIs athlete-cluster bootstrap.",
           title="Acceleration outliers by tier")
    return fig, {"metric_by_tier": m, "extreme_accel": x}


def a1_06(o, q):
    p = o[PROPORTIONS.name].filter(pl.col("facet") == "duration_bin")
    doms = _domains(p)
    fig, ax = figure(1, max(1, len(doms)))
    for j, dom in enumerate(doms):
        d = p.filter(pl.col("domain") == dom).sort("level")
        _share_points(ax[0, j], d, "level", [("any_flag", "any screening flag", CATEGORICAL[3]),
                                             ("tier_B", "tier B", TIER_COLORS["B"])])
        ax[0, j].set_xlabel("trajectory duration bin (s)")
        ax[0, j].set_title(dom)
    if not doms:
        empty_panel(ax[0, 0], "no data")
    finish(fig, "Max-type metrics rise mechanically with duration (more opportunity); interpret jointly with the "
                "A1.duration_association Spearman results.", title="Trajectory duration versus quality")
    return fig, {"proportions": p}


def a1_07(o, q):
    r = o[REASONS.name]
    fig, ax = figure(1, 1, size="wide")
    if not r.height:
        empty_panel(ax[0, 0], "no rule or flag fired")
        finish(fig, "", title="Quality failure reasons")
        return fig, {"reasons": r}
    r = r.with_columns((pl.col("domain") + ":" + pl.col("stratum")).alias("col"))
    cols, reasons = sorted(r["col"].unique().to_list()), sorted(r["reason"].unique().to_list())
    mat = np.full((len(reasons), len(cols)), np.nan)
    for row in r.iter_rows(named=True):
        mat[reasons.index(row["reason"]), cols.index(row["col"])] = row["share"] * 100
    im = ax[0, 0].imshow(np.nan_to_num(mat, nan=0.0), aspect="auto", cmap="Blues")
    for i in range(len(reasons)):
        for j in range(len(cols)):
            if np.isfinite(mat[i, j]):
                ax[0, 0].text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7,
                              color="white" if mat[i, j] > np.nanmax(mat) * 0.6 else INK["primary"])
    ax[0, 0].set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7)
    ax[0, 0].set_yticks(range(len(reasons)), reasons, fontsize=8)
    ax[0, 0].grid(False)
    fig.colorbar(im, ax=ax[0, 0], label="% of stratum trajectories", shrink=0.8)
    finish(fig, "Rows: M1 tier rules and A1 screening flags ('flag:'). Cells: % of the stratum's trajectories.",
           title="Which quality checks fire, and where?")
    return fig, {"reasons": r.drop("col")}


def a1_08(o, q):
    d = o[DIMS.name]
    fig, ax = figure(1, 1, size="panel")
    have = d.filter(pl.col("smoothing_rel_change").is_not_null())
    if not have.height:
        empty_panel(ax[0, 0], "Smoothing instability is produced by Study B.\nRe-draw after Study B has run.")
        data = pl.DataFrame({"note": ["pending Study B"]})
    else:
        data = (have.with_columns(pl.col("pct_smoothing_rel_change").cut([0.5, 0.9, 0.99]).cast(pl.String).alias("bin"))
                .group_by("domain", "bin").agg(pl.len().alias("n"), (pl.col("n_flagged_dims") > 1).mean().alias("share_other_flag"))
                .sort("domain", "bin"))
        for k, dom in enumerate(_domains(data)):
            dd = data.filter(pl.col("domain") == dom)
            ax[0, 0].plot(range(dd.height), dd["share_other_flag"] * 100, "o-", color=DOMAIN_COLORS.get(dom, CATEGORICAL[k]),
                          label=dom)
            ax[0, 0].set_xticks(range(dd.height), dd["bin"].to_list())
        ax[0, 0].set_ylabel("% with another flagged dimension")
        ax[0, 0].legend(frameon=False)
    finish(fig, "Do trajectories that are unstable under smoothing also fail other quality dimensions?",
           title="Quality versus smoothing instability")
    return fig, {"binned": data}


def a1_09(o, q):
    ex, sig = o[A1_EXAMPLES.name], o[A1_SIGNALS.name]
    n = max(1, ex.height)
    fig, ax = figure(2, n, size="wide")
    xs, ys, path_axes = [], [], []
    for j, r in enumerate(ex.iter_rows(named=True)):
        s = sig.filter(pl.col("example") == r["example"]).sort("t_s")
        if not s.height:
            empty_panel(ax[0, j], "no frames")
            empty_panel(ax[1, j], "—")
            continue
        ax[0, j].scatter(s["x_rel"], s["y_rel"], c=s["speed"], cmap="Blues", s=6)
        path_axes.append(ax[0, j])
        xs.append(s["x_rel"].to_numpy())
        ys.append(s["y_rel"].to_numpy())
        ax[0, j].set_title(f"{r['example']}\ntier {r['quality_tier']} · flags: {r['flagged_dims'] or 'none'}", fontsize=10)
        ax[1, j].plot(s["t_s"], s["vendor_speed"], color=INK["muted"], lw=1.2, label="vendor s (observed)")
        ax[1, j].plot(s["t_s"], s["speed"], color=CATEGORICAL[0], lw=1.6, label="derived (SG)")
        ax[1, j].set_xlabel("time (s)")
        ax[1, j].legend(fontsize=7, frameon=False)
    if path_axes:
        equal_aspect_panels(path_axes, xs, ys)
        ax[1, 0].set_ylabel("speed (yd/s)")
        ax[0, 0].set_ylabel("y − start (yd)")
    else:
        empty_panel(ax[0, 0], "no examples")
    finish(fig, "Selection rule in A1_examples.csv (deterministic, outcome-blind). NFL examples come from the "
                "deterministic frame sample (full NFL frames are not persisted). Path colour = derived speed.",
           title="Representative good and poor trajectories")
    return fig, {"examples": ex, "signals": sig}


def a1_10(o, q):
    f = o[FUNNEL.name].filter(pl.col("stratum") == "all").sort("domain", "scenario_order")
    doms = _domains(f)
    fig, ax = figure(1, max(1, len(doms)), sharey=True)
    for j, dom in enumerate(doms):
        d = f.filter(pl.col("domain") == dom)
        x = np.arange(d.height)
        ax[0, j].bar(x - 0.2, d["share_trajectories_retained"] * 100, 0.38, color=CATEGORICAL[0], label="trajectories")
        ax[0, j].bar(x + 0.2, d["share_athletes_retained"] * 100, 0.38, color=CATEGORICAL[2], label="athletes")
        for xi, r in zip(x, d.iter_rows(named=True)):
            ax[0, j].text(xi, 2, f"{r['n_trajectories']:,}\n{r['n_athletes']:,} ath.", ha="center", fontsize=7,
                          color="white")
        ax[0, j].set_xticks(x, [s.replace(" + ", "\n+ ") for s in d["scenario"]], fontsize=8)
        ax[0, j].set_title(dom)
        ax[0, j].legend(fontsize=8, frameon=False, loc="lower left")
    ax[0, 0].set_ylabel("% retained")
    finish(fig, "Retention under nested quality filters (A1_filter_funnel.csv). Athlete retention matters more than "
                "trajectory retention for athlete-level research.", title="Quality-filter impact on research sample size")
    return fig, {"funnel": f}


# ============================================================================== A2
def _hist_panel(ax, h: pl.DataFrame, xstep: float, ystep: float, xlabel: str, ylabel: str):
    from matplotlib.colors import LogNorm

    if not h.height:
        empty_panel(ax, "no frames")
        return None
    x, y, c = h["x_lo"].to_numpy(), h["y_lo"].to_numpy(), h["count"].to_numpy().astype(float)
    xe = np.arange(x.min(), x.max() + 2 * xstep, xstep)
    ye = np.arange(y.min(), y.max() + 2 * ystep, ystep)
    grid = np.zeros((xe.size - 1, ye.size - 1))
    grid[np.clip(np.round((x - xe[0]) / xstep).astype(int), 0, xe.size - 2),
         np.clip(np.round((y - ye[0]) / ystep).astype(int), 0, ye.size - 2)] = c
    m = ax.pcolormesh(xe, ye, np.ma.masked_equal(grid.T, 0), cmap="Blues", norm=LogNorm(vmin=1, vmax=max(c.max(), 2)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return m


def a2_01(o, q):
    h = o[HIST2D.name].filter(pl.col("kind") == "vendor_vs_derived")
    doms = _domains(h)
    fig, ax = figure(1, max(1, len(doms)), size="wide")
    for j, dom in enumerate(doms):
        m = _hist_panel(ax[0, j], h.filter(pl.col("domain") == dom), q.hist_bin, q.hist_bin, "vendor s (yd/s)",
                        "derived speed (yd/s)")
        ax[0, j].plot([0, q.hist_speed_max], [0, q.hist_speed_max], color=INK["secondary"], ls="--", lw=1)
        ax[0, j].set_aspect("equal", adjustable="box")
        n = int(h.filter(pl.col("domain") == dom)["count"].sum())
        ax[0, j].set_title(f"{dom} (n = {n:,} frames)")
        if m is not None:
            fig.colorbar(m, ax=ax[0, j], label="frames (log)", shrink=0.8)
    if not doms:
        empty_panel(ax[0, 0], "no frames")
    finish(fig, "Frame grain (frames are not independent observations). Dashed = identity. NFL = deterministic "
                "game sample.", title="Vendor versus derived speed")
    return fig, {"hist2d": h}


def a2_02(o, q):
    b = o[BY_SPEED.name]
    fig, ax = figure(1, 1)
    for k, dom in enumerate(_domains(b)):
        d = b.filter((pl.col("domain") == dom) & (pl.col("n_frames") >= 100)).sort("m_lo")
        c = DOMAIN_COLORS.get(dom, CATEGORICAL[k])
        x = d["m_lo"].to_numpy() + 0.125
        ax[0, 0].fill_between(x, d["p05"], d["p95"], color=c, alpha=0.12, lw=0)
        ax[0, 0].fill_between(x, d["p25"], d["p75"], color=c, alpha=0.3, lw=0)
        ax[0, 0].plot(x, d["median"], color=c, lw=2, label=f"{dom} median (bands p25–p75, p05–p95)")
    ax[0, 0].axhline(0, color=INK["secondary"], lw=1)
    ax[0, 0].set_xlabel("Bland–Altman mean speed (vendor + derived)/2 (yd/s)")
    ax[0, 0].set_ylabel("derived − vendor (yd/s)")
    ax[0, 0].legend(frameon=False, fontsize=9)
    finish(fig, "0.25 yd/s bins with ≥ 100 frames. Binning on the mean (not on either method) avoids regression-to-"
                "the-mean artefacts.", title="Residual versus speed")
    return fig, {"by_speed": b}


def a2_03(o, q):
    v = o[VENDOR.name].filter(pl.col("facet") == "speed_band")
    fig, ax = figure(1, 1)
    doms = _domains(v)
    for k, dom in enumerate(doms):
        d = v.filter(pl.col("domain") == dom).sort("level")
        x = np.arange(d.height) + (k - (len(doms) - 1) / 2) * 0.15
        ci_errorbar(ax[0, 0], x, d["mae"].to_numpy(), d["mae_lower"].fill_null(np.nan).to_numpy(),
                    d["mae_upper"].fill_null(np.nan).to_numpy(), color=DOMAIN_COLORS.get(dom, CATEGORICAL[k]), label=dom)
        ax[0, 0].set_xticks(np.arange(d.height), [f"{l}" for l in d["level"]], fontsize=9)
    ax[0, 0].set_xlabel("speed band of the Bland–Altman mean (yd/s)")
    ax[0, 0].set_ylabel("MAE |derived − vendor| (yd/s)")
    ax[0, 0].legend(frameon=False)
    finish(fig, "Frame-grain MAE with athlete-cluster bootstrap 95 % CI.", title="Absolute error versus speed")
    return fig, {"vendor_error": v}


def _vendor_strips(o, facet: str, title: str, caption: str, domains: tuple[str, ...] | None = None):
    v = o[VENDOR.name].filter(pl.col("facet") == facet)
    if domains:
        v = v.filter(pl.col("domain").is_in(list(domains)))
    doms = _domains(v)
    fig, ax = figure(1, max(1, len(doms)), size="wide")
    for j, dom in enumerate(doms):
        d = v.filter(pl.col("domain") == dom).sort("level")
        _strips_panel(ax[0, j], d, "level", DOMAIN_COLORS.get(dom, CATEGORICAL[j]), "derived − vendor (yd/s)",
                      center_col="median_signed")
        ax[0, j].axvline(0, color=INK["secondary"], lw=1)
        ax[0, j].set_title(dom)
    if not doms:
        empty_panel(ax[0, 0], "no data")
    finish(fig, caption, title=title)
    return fig, {"vendor_error": v}


def a2_04(o, q):
    return _vendor_strips(o, "all", "Residual distribution by domain",
                          "Strip: p05–p95, p25–p75, median signed residual; n = frames.")


def a2_05(o, q):
    return _vendor_strips(o, "position", "Residual distribution by position",
                          "Position = gold.dim_athlete position_group ('UNKNOWN' if unmatched); n = frames.")


def a2_06(o, q):
    return _vendor_strips(o, "stratum", "Residual distribution by Combine drill type",
                          "Combine only. M1: three-cone and shuttle had the widest p95 — check whether the strips "
                          "confirm it.", domains=("combine",))


def a2_07(o, q):
    d = o[FRAME_DIAG.name].filter(pl.col("n_frames") >= q.min_frames_trajectory)
    b = (d.with_columns(pl.col("duration_s").cut(list(q.duration_bins_s[1:]), left_closed=True).cast(pl.String).alias("bin"))
         .group_by("domain", "bin").agg(pl.len().alias("n"), pl.col("mae").median().alias("median"),
                                        pl.col("mae").quantile(0.25).alias("p25"), pl.col("mae").quantile(0.75).alias("p75"),
                                        pl.col("duration_s").median().alias("x"))
         .sort("domain", "x"))
    fig, ax = figure(1, 1)
    for k, dom in enumerate(_domains(b)):
        dd = b.filter(pl.col("domain") == dom)
        c = DOMAIN_COLORS.get(dom, CATEGORICAL[k])
        ax[0, 0].fill_between(dd["x"], dd["p25"], dd["p75"], color=c, alpha=0.25, lw=0)
        ax[0, 0].plot(dd["x"], dd["median"], "o-", color=c, label=f"{dom} (n = {int(dd['n'].sum()):,} trajectories)")
    ax[0, 0].set_xlabel("trajectory duration (s, bin median)")
    ax[0, 0].set_ylabel("trajectory MAE (yd/s)")
    ax[0, 0].legend(frameon=False)
    finish(fig, f"Trajectory grain (≥ {q.min_frames_trajectory} valid frames); band = IQR. Spearman ρ with CI: "
                "results row A2.error_vs_duration.", title="Error versus trajectory duration")
    return fig, {"binned": b}


def a2_08(o, q):
    h = o[HIST2D.name].filter(pl.col("kind") == "bland_altman")
    v = o[VENDOR.name].filter(pl.col("facet") == "all")
    doms = _domains(h)
    fig, ax = figure(1, max(1, len(doms)), size="wide")
    for j, dom in enumerate(doms):
        _hist_panel(ax[0, j], h.filter(pl.col("domain") == dom), q.hist_bin, q.hist_bin / 4,
                    "mean of vendor and derived (yd/s)", "derived − vendor (yd/s)")
        r = v.filter(pl.col("domain") == dom)
        if r.height:
            r = r.row(0, named=True)
            for val, lab, ls in ((r["bias"], "bias", "-"), (r["loa_lower"], "LoA", "--"), (r["loa_upper"], None, "--")):
                if val is not None:
                    ax[0, j].axhline(val, color=INK["primary"], ls=ls, lw=1.2, label=lab)
            ax[0, j].set_title(f"{dom}: bias {r['bias']:+.3f} yd/s, LoA [{r['loa_lower']:+.2f}, {r['loa_upper']:+.2f}]",
                               fontsize=11)
            ax[0, j].legend(fontsize=8, frameon=False)
    if not doms:
        empty_panel(ax[0, 0], "no frames")
    finish(fig, "Frame-level limits of agreement (bias ± 1.96 SD) describe FRAMES; residuals beyond ±1 yd/s are "
                "drawn in the edge bins. Heteroscedasticity → see residual-versus-speed figure.",
           title="Bland–Altman: derived vs vendor speed")
    return fig, {"hist2d": h, "agreement": v}


def a2_09(o, q):
    e = o[A2_EXAMPLES.name]
    exs = e["example"].unique(maintain_order=True).to_list()
    n = max(1, len(exs))
    fig, ax = figure(2 if n > 3 else 1, min(n, 3), size="wide")
    flat = ax.ravel()
    for k, name in enumerate(exs[: flat.size]):
        d = e.filter(pl.col("example") == name).sort("t_s")
        flat[k].plot(d["t_s"], d["vendor_speed"], color=INK["muted"], lw=1.3, label="vendor s")
        flat[k].plot(d["t_s"], d["speed"], color=CATEGORICAL[1], lw=1.6, label="derived")
        flat[k].set_title(f"{name}: MAE {d['trajectory_mae'][0]:.2f} yd/s", fontsize=10)
        flat[k].set_xlabel("time (s)")
        flat[k].legend(fontsize=7, frameon=False)
    for k in range(len(exs), flat.size):
        empty_panel(flat[k], "—")
    finish(fig, f"Top {q.n_examples} trajectories by MAE per domain (≥ {q.min_frames_trajectory} frames; ties by id). "
                "Inspect for segment breaks, vendor lag or smoothing edge effects.",
           title="Highest-disagreement trajectories")
    return fig, {"signals": e}


def a2_10(o, q):
    v = o[VENDOR.name].filter(pl.col("facet").is_in(["cohort", "stratum", "position", "quality_tier"]))
    doms = _domains(v)
    fig, ax = figure(1, max(1, len(doms)), size="tall")
    for j, dom in enumerate(doms):
        d = v.filter(pl.col("domain") == dom).sort("facet", "level")
        y = np.arange(d.height)
        ax[0, j].errorbar(d["mae"], y, xerr=np.clip(np.vstack([d["mae"] - d["mae_lower"].fill_null(d["mae"]),
                                                               d["mae_upper"].fill_null(d["mae"]) - d["mae"]]), 0, None),
                          fmt="o", color=DOMAIN_COLORS.get(dom, CATEGORICAL[j]), capsize=2, ms=3)
        ax[0, j].set_yticks(y, [f"{r['facet']}: {r['level']} (n={r['n_athletes']:,} ath.)" for r in d.iter_rows(named=True)],
                            fontsize=7)
        ax[0, j].invert_yaxis()
        ax[0, j].set_xlabel("MAE (yd/s), 95 % athlete-cluster CI")
        ax[0, j].set_title(dom)
    if not doms:
        empty_panel(ax[0, 0], "no data")
    finish(fig, "Comparative frame-grain MAE; n = athletes (the resampling unit).", title="Error summary with uncertainty")
    return fig, {"vendor_error": v}


def _rec(fid, name, q_, insight, stat, data, grain, filt, method, pub=False):
    return FigureRecord(fid, name, "A", q_, insight, stat, data, grain, filt, method, pub)


FIGURES: list[tuple[FigureRecord, Callable]] = [
    (_rec("A1-01", "A1_01_tier_distribution_by_domain", "How are provisional tiers distributed per domain?",
          "Whether tiers are informative at all (M1: ~99 % tier A)", "share of trajectories per tier",
          PROPORTIONS.name, "trajectory", "all tiers", "proportion + athlete-cluster bootstrap", True), a1_01),
    (_rec("A1-02", "A1_02_quality_by_cohort", "Does quality differ by draft year / season?",
          "Cohort-specific tracking or processing differences", "share non-A / flagged", PROPORTIONS.name,
          "trajectory", "facet=cohort", "proportion + cluster bootstrap"), a1_02),
    (_rec("A1-03", "A1_03_quality_by_drill", "Which Combine drills produce lower-quality trajectories?",
          "Drill-specific measurement problems", "share flagged / tier B", PROPORTIONS.name, "trajectory",
          "Combine, facet=stratum", "proportion + cluster bootstrap"), a1_03),
    (_rec("A1-04", "A1_04_vendor_residual_by_tier", "Do tiers separate vendor-speed disagreement?",
          "Whether tier B/C/reject capture speed inconsistency", "quantiles of speed_vendor_mad", METRIC_BY_TIER.name,
          "trajectory", "all", "quantile strips; Cliff's δ in results"), a1_04),
    (_rec("A1-05", "A1_05_acceleration_outliers_by_tier", "Do tiers separate implausible accelerations?",
          "Derivative reliability per tier", "max |a| quantiles; extreme-frame rate", f"{METRIC_BY_TIER.name}, "
          f"{EXTREME_ACCEL.name}", "trajectory / frame", "all", "quantile strips; ratio + cluster bootstrap"), a1_05),
    (_rec("A1-06", "A1_06_duration_vs_quality", "Is quality associated with trajectory length?",
          "Length-driven flagging (max-type metrics)", "share flagged by duration bin", PROPORTIONS.name,
          "trajectory", "facet=duration_bin", "proportion + cluster bootstrap"), a1_06),
    (_rec("A1-07", "A1_07_failure_reason_heatmap", "Which rules / flags fire, and in which strata?",
          "Dominant failure modes", "% of stratum trajectories", REASONS.name, "trajectory", "all",
          "counts"), a1_07),
    (_rec("A1-08", "A1_08_quality_vs_smoothing_instability", "Are smoothing-unstable trajectories also low quality?",
          "Shared vs independent failure modes", "share with another flag by smoothing-instability bin", DIMS.name,
          "trajectory", "requires Study B", "binned proportion"), a1_08),
    (_rec("A1-09", "A1_09_representative_trajectories", "What do good and poor trajectories look like?",
          "Face validity of the dimensions", "path + speed signals", A1_SIGNALS.name, "frame",
          "deterministic examples", "visual inspection"), a1_09),
    (_rec("A1-10", "A1_10_filter_impact", "How much research sample does each quality filter cost?",
          "Selection cost of filtering", "share trajectories / athletes retained", FUNNEL.name, "trajectory / athlete",
          "nested scenarios", "counts", True), a1_10),
    (_rec("A2-01", "A2_01_vendor_vs_derived_hist", "Do vendor and derived speed agree across the speed range?",
          "Global agreement and outliers", "2-D frame histogram", HIST2D.name, "frame", "valid_kin frames",
          "histogram", True), a2_01),
    (_rec("A2-02", "A2_02_residual_vs_speed", "Is the residual speed-dependent (bias or spread)?",
          "Heteroscedasticity / proportional bias", "residual quantiles by mean-speed bin", BY_SPEED.name, "frame",
          "bins ≥ 100 frames", "binned quantiles"), a2_02),
    (_rec("A2-03", "A2_03_abs_error_vs_speed", "How large is the absolute error per speed band?",
          "Where derived speed is least trustworthy", "MAE with CI per speed band", VENDOR.name, "frame",
          "facet=speed_band", "MAE + athlete-cluster bootstrap"), a2_03),
    (_rec("A2-04", "A2_04_residual_by_domain", "Does agreement differ between Combine and NFL?",
          "Domain-specific tracking differences", "residual quantiles", VENDOR.name, "frame", "facet=all",
          "quantile strips"), a2_04),
    (_rec("A2-05", "A2_05_residual_by_position", "Does agreement differ by position?",
          "Position-specific movement content effects", "residual quantiles", VENDOR.name, "frame",
          "facet=position", "quantile strips"), a2_05),
    (_rec("A2-06", "A2_06_residual_by_drill", "Which Combine drills have the poorest agreement?",
          "COD-heavy drills vs straight-line", "residual quantiles", VENDOR.name, "frame", "Combine, facet=stratum",
          "quantile strips"), a2_06),
    (_rec("A2-07", "A2_07_error_vs_duration", "Is trajectory-level error related to duration?",
          "Edge effects in short trajectories", "median trajectory MAE by duration bin", FRAME_DIAG.name,
          "trajectory", f"≥ {QualityStudyConfig().min_frames_trajectory} frames", "binned median + IQR"), a2_07),
    (_rec("A2-08", "A2_08_bland_altman", "What are bias and limits of agreement?",
          "Interchangeability of derived and vendor speed", "bias, 95 % LoA", f"{HIST2D.name}, {VENDOR.name}",
          "frame", "valid_kin frames", "Bland–Altman", True), a2_08),
    (_rec("A2-09", "A2_09_high_disagreement_trajectories", "What causes the largest disagreements?",
          "Failure-mode case studies", "speed signals", A2_EXAMPLES.name, "frame", "top-MAE trajectories",
          "visual inspection"), a2_09),
    (_rec("A2-10", "A2_10_error_summary_forest", "How does MAE compare across cohorts, drills, positions, tiers?",
          "Comparative agreement with uncertainty", "MAE with CI", VENDOR.name, "frame", "facets", "forest plot",
          True), a2_10),
]


def draw_all(store: StudyStore, out: dict[str, pl.DataFrame], q: QualityStudyConfig) -> list[str]:
    """Draw every Study A figure; a failure is logged with traceback and reported, never hidden."""
    failed = []
    for rec, fn in FIGURES:
        try:
            fig, data = fn(out, q)
            store.save_figure(fig, rec, data)
        except Exception:
            log.exception("[studies:A] figure %s failed", rec.name)
            failed.append(rec.name)
    if failed:
        log.warning("[studies:A] %d figure(s) failed: %s", len(failed), failed)
    return failed
