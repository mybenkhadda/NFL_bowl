"""Study B figures (drawn ONLY from persisted Study B outputs — ``--figures-only`` redraws without recompute).

Every figure states its research question in the figure index (``FigureRecord``), carries sample sizes in
labels, and saves the plotted numbers as plot data next to the table outputs. Captions end with
"PENDING interpretation" — interpretation is written after the user has executed the study.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..studies.store import FigureRecord, StudyStore
from .config import ValidationConfig
from .smoothing import AGREE, BA_FEATURES, EXAMPLES, FEATURES_DS, REPEAT, ROBUST, VARSHARE, VENDOR, athlete_values, paired
from .metrics import rank_change

STUDY = "B"
PRIMARY = ValidationConfig().primary_strata
PAIRS = (("peak_speed", "p95_speed"), ("peak_a_tan", "p95_a_tan"), ("peak_decel", "p95_decel"),
         ("peak_abs_yaw_rate", "p90_abs_yaw_rate"))
FOCUS = [f for pair in PAIRS for f in pair] + ["peak_speed_interior", "peak_decel_interior", "path_smooth_yds",
                                               "rate_braking_per_s", "rate_reversal_per_s"]


def _style():
    from ...data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE

    return CATEGORICAL, INK, SEQUENTIAL_BLUE


def _settings(df: pl.DataFrame) -> list[str]:
    return sorted(df["smoothing"].unique().to_list()) if df.height else []


def _drills(df: pl.DataFrame) -> list[str]:
    have = set(df["drill"].unique().to_list()) if df.height else set()
    return [d for d in PRIMARY if d in have]


def _err(est, lo, hi) -> np.ndarray:
    est, lo, hi = (np.asarray(v, float) for v in (est, lo, hi))
    return np.vstack([np.where(np.isfinite(lo), est - lo, 0), np.where(np.isfinite(hi), hi - est, 0)]).clip(0)


# ----------------------------------------------------------------------------- B-01
def fig_heatmap(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    from ..figkit import empty_panel, figure, finish

    cat, ink, blue = _style()
    a = out[AGREE.name]
    drills = _drills(a)
    fig, ax = figure(1, 2, size="tall")
    plot = a.filter(pl.col("drill").is_in(drills))
    if not plot.height:
        for k in range(2):
            empty_panel(ax[0, k], "no primary-stratum agreement rows")
        finish(fig, "No data.", "B1 · Smoothing sensitivity heatmap")
        store.save_figure(fig, rec, {"agreement": plot}, n="0")
        return
    feats = [f for f in plot["feature"].unique(maintain_order=True).to_list()]
    feats = sorted(feats, key=lambda f: (FOCUS.index(f) if f in FOCUS else 99, f))
    cols = [(d, s) for d in drills for s in _settings(plot)]
    panels = (("athlete", "spearman", "athlete-level Spearman ρ vs primary", (0.8, 1.0), False),
              ("trajectory", "median_rel_diff", "trajectory median |relative change|", (0.0, 0.15), True))
    cmap = LinearSegmentedColormap.from_list("seqblue", list(blue))
    for k, (level, col, label, (vmin, vmax), reverse) in enumerate(panels):
        m = np.full((len(feats), len(cols)), np.nan)
        d = plot.filter(pl.col("level") == level)
        for r in d.iter_rows(named=True):
            if (r["drill"], r["smoothing"]) in cols and r[col] is not None:
                m[feats.index(r["feature"]), cols.index((r["drill"], r["smoothing"]))] = r[col]
        axk = ax[0, k]
        im = axk.imshow(np.clip(m, vmin, vmax), aspect="auto", cmap=cmap.reversed() if reverse else cmap, vmin=vmin,
                        vmax=vmax)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    t = (np.clip(m[i, j], vmin, vmax) - vmin) / (vmax - vmin)
                    dark = t < 0.4 if reverse else t > 0.6           # dark cell → white text
                    axk.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                             color=ink["surface"] if dark else ink["primary"])
        axk.set_yticks(np.arange(len(feats)), feats, fontsize=7)
        axk.set_xticks(np.arange(len(cols)), [f"{dr[:10]}·{s}" for dr, s in cols], rotation=60, ha="right", fontsize=6)
        axk.set_title(label, fontsize=10)
        fig.colorbar(im, ax=axk, shrink=0.6)
    finish(fig, "Left: athlete-level rank agreement with the primary SG setting (sg7p2), colour clipped to [0.8, 1]. "
                "Right: median relative change of trajectory values (clipped at 15 %). High rank agreement with large "
                "relative change = athlete ordering is safe but absolute values (and cross-domain comparisons) are "
                "setting-dependent. PENDING interpretation.", "B1 · Which features are smoothing-sensitive?")
    store.save_figure(fig, rec, {"agreement": plot}, n=f"{len(feats)} features × {len(cols)} drill-settings")


# ----------------------------------------------------------------------------- B-02
def fig_bland_altman(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish
    from ..studies.stats import bland_altman

    cat, ink, _ = _style()
    long = out[FEATURES_DS.name].filter(pl.col("feature").is_in(list(BA_FEATURES)))
    pr = paired(long).with_columns(((pl.col("value") + pl.col("v0")) / 2).alias("mean"),
                                   (pl.col("value") - pl.col("v0")).alias("diff"))
    domains = [d for d in ("combine", "nfl_sample") if d in set(pr["domain"].to_list())] or ["combine"]
    extremes = [s for s in ("sg5p2", "sg11p3") if s in set(pr["smoothing"].to_list())]
    fig, ax = figure(len(domains), len(BA_FEATURES), size="wide")
    for i, dom in enumerate(domains):
        for j, feat in enumerate(BA_FEATURES):
            a = ax[i, j]
            g = pr.filter((pl.col("domain") == dom) & (pl.col("feature") == feat))
            if not g.height:
                empty_panel(a, f"no {dom} {feat}")
                continue
            for k, s in enumerate(extremes):
                h = g.filter(pl.col("smoothing") == s)
                if not h.height:
                    continue
                ba = bland_altman(h["diff"].to_numpy())
                a.scatter(h["mean"].to_numpy(), h["diff"].to_numpy(), s=3, alpha=0.25, color=cat[k],
                          label=f"{s} (n={h.height:,}; bias {ba['bias']:+.3f})", rasterized=True)
                for y, ls in ((ba["bias"], "-"), (ba["loa_lower"], ":"), (ba["loa_upper"], ":")):
                    a.axhline(y, color=cat[k], lw=1, ls=ls)
            a.axhline(0, color=ink["muted"], lw=0.8)
            a.set_title(f"{dom} · {feat}", fontsize=9)
            a.set_xlabel("mean of setting and primary")
            if j == 0:
                a.set_ylabel("setting − primary")
            a.legend(frameon=False, fontsize=6, loc="upper left", markerscale=3)
    finish(fig, "Bland–Altman at trajectory grain (all drills pooled within domain): bias (solid) and 95 % limits of "
                "agreement (dotted) of the narrowest (sg5p2) and widest (sg11p3) SG window against the primary sg7p2. "
                "Units as in B_feature_catalog (yd/s, yd/s², deg/s). Limits describe trajectories, not athletes. "
                "PENDING interpretation.", "B2 · How much do magnitudes move with the smoothing window?")
    store.save_figure(fig, rec, {"pairs": pr.select("domain", "feature", "smoothing", "traj_id", "mean", "diff")},
                      n=f"{pr.height:,} trajectory-setting pairs")


# ----------------------------------------------------------------------------- B-03
def fig_rank_change(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink, _ = _style()
    long = out[FEATURES_DS.name].filter(pl.col("feature").is_in([f for p in PAIRS for f in p]))
    pr = paired(athlete_values(long))
    drills = _drills(pr)
    fig, ax = figure(1, max(1, len(drills)), sharey=True)
    rows = []
    for k, dr in enumerate(drills or ["none"]):
        a = ax[0, k]
        g = pr.filter((pl.col("drill") == dr) & (pl.col("smoothing") == "sg5p2"))
        if not g.height:
            empty_panel(a, f"no {dr}")
            continue
        for j, (peak, robust) in enumerate(PAIRS):
            for feat, ls in ((peak, "-"), (robust, "--")):
                h = g.filter(pl.col("feature") == feat)
                rc = np.sort(rank_change(h["value"].to_numpy(), h["v0"].to_numpy()))
                if not rc.size:
                    continue
                y = np.arange(1, rc.size + 1) / rc.size
                a.plot(rc, y, ls=ls, color=cat[j], lw=1.3, label=feat)
                rows += [{"drill": dr, "feature": feat, "rank_change": float(x), "ecdf": float(p)} for x, p in zip(rc, y)]
        n = g.filter(pl.col("feature") == "peak_speed").height
        a.set_title(f"{dr} (athletes={n})", fontsize=9)
        a.set_xlabel("|Δ percentile rank| (sg5p2 vs sg7p2)")
        a.set_xlim(0, 0.3)
    ax[0, 0].set_ylabel("cumulative share of athletes")
    ax[0, -1].legend(frameon=False, fontsize=6, loc="lower right")
    finish(fig, "Empirical CDF of each athlete's change in percentile rank when the narrowest SG window replaces the "
                "primary one (athlete = median over attempts / plays). Solid = peak (max) feature, dashed = robust "
                "quantile of the same signal. A curve reaching 1.0 quickly = rankings barely move. "
                "PENDING interpretation.", "B3 · Do athlete rankings move under smoothing — peaks vs robust quantiles?")
    store.save_figure(fig, rec, {"ecdf": pl.DataFrame(rows) if rows else pl.DataFrame({"drill": []})},
                      n=f"{pr['nfl_id'].n_unique() if pr.height else 0:,} athletes")


# ----------------------------------------------------------------------------- B-04
def fig_top_overlap(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink, _ = _style()
    a_all = out[AGREE.name].filter(pl.col("feature").is_in(FOCUS))
    drills = _drills(a_all)
    fig, ax = figure(1, max(1, len(drills)), sharey=True)
    for k, dr in enumerate(drills or ["none"]):
        a = ax[0, k]
        g = a_all.filter((pl.col("drill") == dr) & (pl.col("level") == "athlete"))
        if not g.height:
            empty_panel(a, f"no {dr}")
            continue
        for j, s in enumerate(_settings(g)):
            h = pl.DataFrame({"feature": FOCUS}).join(g.filter(pl.col("smoothing") == s), on="feature", how="left")
            y = np.arange(len(FOCUS)) + (j - 1) * 0.2
            est = h["top_overlap"].fill_null(np.nan).to_numpy()
            a.errorbar(est, y, xerr=_err(est, h["top_overlap_lower"].fill_null(np.nan),
                                          h["top_overlap_upper"].fill_null(np.nan)),
                       fmt="o", ms=4, capsize=2, color=cat[j], label=s)
        a.set_yticks(np.arange(len(FOCUS)), FOCUS, fontsize=7)
        a.invert_yaxis()
        a.set_xlim(0, 1.02)
        a.axvline(ValidationConfig().overlap_pass, color=ink["muted"], ls="--", lw=1)
        a.set_title(f"{dr} (athletes={int(g['n_units'].max())})", fontsize=9)
        a.set_xlabel("top-decile overlap with primary")
    ax[0, -1].legend(frameon=False, fontsize=7, title="setting", title_fontsize=7)
    finish(fig, "Share of the primary-setting top-10 % athletes who remain in the top 10 % under each alternative SG "
                "setting (athlete-level medians), athlete bootstrap 95 % CI. Dashed line: proposed pass value 0.70 "
                "(not a validated cut-off). Count features with tied values are tie-dominated (see modal_share). "
                "PENDING interpretation.", "B4 · Would the same athletes be called 'elite' under another smoothing?")
    store.save_figure(fig, rec, {"overlap": a_all.filter(pl.col("level") == "athlete")},
                      n=f"{len(drills)} primary strata")


# ----------------------------------------------------------------------------- B-05
def fig_variance_share(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink, _ = _style()
    v = out[VARSHARE.name].filter(pl.col("feature").is_in(FOCUS))
    drills = _drills(v)
    fig, ax = figure(1, max(1, len(drills)), sharey=True)
    for k, dr in enumerate(drills or ["none"]):
        a = ax[0, k]
        g = v.filter(pl.col("drill") == dr)
        if not g.height:
            empty_panel(a, f"no {dr}")
            continue
        for j, level in enumerate(("trajectory", "athlete")):
            h = pl.DataFrame({"feature": FOCUS}).join(g.filter(pl.col("level") == level), on="feature", how="left")
            y = np.arange(len(FOCUS)) + (j - 0.5) * 0.35
            a.barh(y, np.clip(h["share_setting"].fill_null(np.nan).to_numpy(), 1e-5, None), height=0.35, color=cat[j],
                   label=level)
        a.set_xscale("log")
        a.set_yticks(np.arange(len(FOCUS)), FOCUS, fontsize=7)
        a.invert_yaxis()
        a.set_title(dr, fontsize=9)
        a.set_xlabel("share of variance due to SG setting (log)")
    ax[0, -1].legend(frameon=False, fontsize=7)
    finish(fig, "Variance attributable to the smoothing setting relative to between-unit variance (all four SG settings "
                "jointly). Values ≪ 0.01 mean the setting is negligible against athlete differences. Floored at 1e-5 for "
                "the log axis. PENDING interpretation.", "B5 · Is smoothing variance small relative to athlete variance?")
    store.save_figure(fig, rec, {"variance_share": v}, n=f"{v.height} rows")


# ----------------------------------------------------------------------------- B-06
def fig_icc(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink, _ = _style()
    r = out[REPEAT.name].filter(pl.col("feature").is_in(FOCUS) & pl.col("icc1").is_not_null())
    drills = [d for d in PRIMARY if d in set(r["drill"].to_list())]
    fig, ax = figure(1, max(1, len(drills)), sharey=True)
    for k, dr in enumerate(drills or ["none"]):
        a = ax[0, k]
        g = r.filter(pl.col("drill") == dr)
        if not g.height:
            empty_panel(a, f"no repeated attempts in {dr}")
            continue
        for j, s in enumerate(_settings(g)):
            h = pl.DataFrame({"feature": FOCUS}).join(g.filter(pl.col("smoothing") == s), on="feature", how="left")
            y = np.arange(len(FOCUS)) + (j - 1.5) * 0.15
            est = h["icc1"].fill_null(np.nan).to_numpy()
            a.errorbar(est, y, xerr=_err(est, h["ci_lower"].fill_null(np.nan), h["ci_upper"].fill_null(np.nan)),
                       fmt="o", ms=3.5, capsize=1.5, color=cat[j], label=s)
        a.axvline(0.75, color=ink["muted"], ls="--", lw=1)
        a.set_yticks(np.arange(len(FOCUS)), FOCUS, fontsize=7)
        a.invert_yaxis()
        a.set_xlim(-0.1, 1.0)
        a.set_title(f"{dr} (athletes={int(g['n_athletes'].max())})", fontsize=9)
        a.set_xlabel("ICC(1), first ≤ 4 attempts")
    ax[0, -1].legend(frameon=False, fontsize=7, title="SG setting", title_fontsize=7)
    finish(fig, "Repeatability of each feature across an athlete's repeated Combine attempts, computed separately under "
                "each smoothing setting; athlete bootstrap 95 % CI. Dashed: 0.75 ('good' if the CI lower bound clears "
                "it). A feature whose ICC depends on the setting is not a stable athlete trait. PENDING interpretation.",
           "B6 · Does repeatability depend on the smoothing choice?")
    store.save_figure(fig, rec, {"icc": r}, n=f"{r.height} ICC estimates")


# ----------------------------------------------------------------------------- B-07
def fig_robustness(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink, _ = _style()
    feats = [f for p in PAIRS for f in p]
    r = out[ROBUST.name].filter(pl.col("feature").is_in(feats) & (pl.col("smoothing") == "sg5p2"))
    drills = [d for d in ("FORTY_YARD_DASH", "nfl_play") if d in set(r["drill"].to_list())] or ["FORTY_YARD_DASH"]
    fig, ax = figure(len(drills), 2, sharex=True)
    for i, dr in enumerate(drills):
        for k, facet in enumerate(("cohort", "position")):
            a = ax[i, k]
            g = r.filter((pl.col("drill") == dr) & (pl.col("facet") == facet))
            if not g.height:
                empty_panel(a, f"no {dr} × {facet} with ≥ min athletes")
                continue
            levels = sorted(g["level"].unique().to_list())
            for j, lv in enumerate(levels):
                h = pl.DataFrame({"feature": feats}).join(g.filter(pl.col("level") == lv), on="feature", how="left")
                n = int(h["n_athletes"].max() or 0)
                a.plot(h["spearman"].fill_null(np.nan).to_numpy(), np.arange(len(feats)) + (j - len(levels) / 2) * 0.1,
                       "o", ms=4, color=cat[j % len(cat)], label=f"{lv} (A={n})")
            a.set_yticks(np.arange(len(feats)), feats, fontsize=7)
            a.invert_yaxis()
            a.set_title(f"{dr} · by {facet}", fontsize=9)
            a.set_xlabel("athlete-level Spearman, sg5p2 vs primary")
            a.legend(frameon=False, fontsize=6, loc="lower left")
    finish(fig, "Rank stability under the narrowest SG window within each cohort and position group (groups with "
                "≥ min athletes only; point estimates). A cohort-specific drop would mirror the 2023/24 vs 2025 Combine "
                "noise difference (M1). PENDING interpretation.", "B7 · Is smoothing robustness the same across cohorts "
                                                                   "and positions?")
    store.save_figure(fig, rec, {"robustness": r}, n=f"{r.height} facet rows")


# ----------------------------------------------------------------------------- B-08
def fig_vendor(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink, _ = _style()
    v = out[VENDOR.name]
    domains = [d for d in ("combine", "nfl_sample") if d in set(v["domain"].to_list())] or ["combine"]
    fig, ax = figure(1, len(domains), sharey=True)
    for k, dom in enumerate(domains):
        a = ax[0, k]
        g = v.filter(pl.col("domain") == dom)
        if not g.height:
            empty_panel(a, f"no {dom}")
            continue
        for j, s in enumerate(_settings(g)):
            h = g.filter(pl.col("smoothing") == s).sort("band_lower")
            a.plot(h["band_lower"].to_numpy(), h["mae"].to_numpy(), "o-", color=cat[j], label=f"{s} MAE")
            a.plot(h["band_lower"].to_numpy(), h["bias"].to_numpy(), "x--", color=cat[j], alpha=0.7)
        nf = int(g.filter(pl.col("smoothing") == _settings(g)[0])["n_frames"].sum())
        a.axhline(0, color=ink["muted"], lw=0.8)
        a.set_title(f"{dom} ({nf:,} frames / setting)", fontsize=9)
        a.set_xlabel("vendor speed band, lower edge (yd/s)")
        a.set_ylabel("derived − vendor speed (yd/s): MAE (solid), bias (dashed)")
        a.legend(frameon=False, fontsize=7)
    finish(fig, "Frame-level residual against vendor speed by speed band under each SG setting. Vendor speed is itself "
                "a filtered estimate, so smaller residual = closer to the vendor's filter, NOT necessarily closer to the "
                "truth. Frame grain: no athlete-level inference. PENDING interpretation.",
           "B8 · Residual versus speed under each smoothing setting")
    store.save_figure(fig, rec, {"vendor": v}, n=f"{int(v['n_frames'].sum()):,} frame-settings")


# ----------------------------------------------------------------------------- B-09
def fig_examples(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink, _ = _style()
    e = out[EXAMPLES.name]
    drills = e["drill"].unique(maintain_order=True).to_list() if e.height else []
    fig, ax = figure(2, max(1, len(drills)), sharex="col")
    for k, dr in enumerate(drills or ["none"]):
        g = e.filter(pl.col("drill") == dr)
        if not g.height:
            empty_panel(ax[0, k], "no example")
            empty_panel(ax[1, k], "no example")
            continue
        for j, s in enumerate(_settings(g)):
            h = g.filter(pl.col("smoothing") == s).sort("t_s")
            lw = 2.0 if s == "sg7p2" else 1.0
            ax[0, k].plot(h["t_s"], h["speed"], color=cat[j], lw=lw, label=s)
            ax[1, k].plot(h["t_s"], h["a_tan"], color=cat[j], lw=lw, label=s)
        ax[0, k].set_title(f"{dr} · {g['traj_id'][0]}", fontsize=8)
        ax[0, k].set_ylabel("speed (yd/s)")
        ax[1, k].set_ylabel("a_tan (yd/s²)")
        ax[1, k].set_xlabel("t (s since attempt start)")
        ax[1, k].axhline(0, color=ink["muted"], lw=0.8)
    ax[0, 0].legend(frameon=False, fontsize=7)
    finish(fig, "Deterministic example per drill: the attempt whose primary-setting peak speed is closest to the drill "
                "median. Speed is nearly setting-invariant; tangential acceleration peaks and the first / last half-window "
                "(filter boundary) change most. PENDING interpretation.", "B9 · Smoothing comparison on example attempts")
    store.save_figure(fig, rec, {"signals": e}, n=f"{len(drills)} attempts")


def _rec(fid: str, name: str, q: str, insight: str, stat: str, ds: str, grain: str, method: str,
         publication: bool = False) -> FigureRecord:
    return FigureRecord(figure_id=fid, name=name, study=STUDY, research_question=q, expected_insight=insight,
                        statistic=stat, dataset=ds, grain=grain, filters="none (all tiers)", method=method,
                        publication=publication)


FIGURES = [
    (_rec("B-01", "B_01_sensitivity_heatmap", "Which features keep athlete ranks and values across SG settings?",
          "smoothing-robust vs smoothing-sensitive features", "Spearman ρ, median relative change", AGREE.name,
          "athlete / trajectory", "setting vs primary", publication=True), fig_heatmap),
    (_rec("B-02", "B_02_bland_altman", "How large and biased are magnitude changes under smoothing?",
          "systematic peak attenuation by wider windows", "Bland–Altman bias / LoA", FEATURES_DS.name, "trajectory",
          "difference vs mean"), fig_bland_altman),
    (_rec("B-03", "B_03_rank_change_ecdf", "Do athlete percentile ranks move — peaks vs robust quantiles?",
          "whether quantile features are more rank-stable than maxima", "|Δ percentile rank| ECDF", FEATURES_DS.name,
          "athlete", "athlete medians"), fig_rank_change),
    (_rec("B-04", "B_04_top_decile_overlap", "Would the same athletes be 'elite' under another setting?",
          "scouting-relevant stability of the top decile", "overlap + athlete bootstrap CI", AGREE.name, "athlete",
          "top-decile intersection"), fig_top_overlap),
    (_rec("B-05", "B_05_variance_share", "Is smoothing variance small relative to athlete variance?",
          "setting effect against signal", "variance share", VARSHARE.name, "trajectory / athlete",
          "within-unit across-setting variance"), fig_variance_share),
    (_rec("B-06", "B_06_icc_by_setting", "Does attempt-to-attempt repeatability depend on smoothing?",
          "repeatability that is a property of the athlete, not the filter", "ICC(1) + CI", REPEAT.name, "athlete",
          "one-way ICC per setting", publication=True), fig_icc),
    (_rec("B-07", "B_07_cohort_position_robustness", "Is smoothing robustness uniform across cohorts and positions?",
          "cohort-specific noise effects", "Spearman ρ", ROBUST.name, "athlete", "facet-wise agreement"),
     fig_robustness),
    (_rec("B-08", "B_08_vendor_residual_vs_speed", "How does the residual against vendor speed vary with speed and "
          "setting?", "speed-dependent bias of derived speed", "MAE, bias", VENDOR.name, "frame",
          "binned residuals"), fig_vendor),
    (_rec("B-09", "B_09_example_signals", "What does each smoothing setting do to a real attempt?",
          "visual anchor for the tables", "time series", EXAMPLES.name, "frame", "deterministic examples"),
     fig_examples),
]
