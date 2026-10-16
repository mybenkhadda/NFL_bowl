"""Part 4 — the unified exploratory analysis (one EDA instead of eight partial ones).

Bundle part (Mode A and B): athlete coverage per domain and position, sequences per domain × movement kind, valid
sequence lengths, EXACT channel distributions (speed, tangential / lateral acceleration, sampling interval, heading
validity) accumulated chunk by chunk over every valid frame (the memory-mapped arrays are never copied whole),
per-sequence peak speed, fold balance per position, athletes-table missingness, official-measurement correlations,
the NFL target by position, and a leakage-risk checklist.

Foundation part (Mode B, after 12.0): Combine / NFL / linked coverage per position and NFL exposure (games) from the
cohort registry. The 12.0 eligibility funnel, repeat availability, position support, cross-domain speed support and the
feasibility matrix are drawn by the 12.0 stage itself (one place) — the EDA report lists them.

Colours: fixed categorical order (reference palette, ``dataviz`` skill): Combine = slot 1 (blue), NFL = slot 2 (orange)
in every figure; positions take slots 1–8 in sorted order. Identity is never colour-alone (legends / labels).
"""

from __future__ import annotations

from typing import Any

import numpy as np

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
DOMAIN_COLOR = {"combine": SERIES[0], "nfl": SERIES[1]}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FIGSIZE = (12.8, 7.2)                                   # 16:9
EDGES = {"speed": np.arange(0.0, 12.0001, 0.25), "a_tan": np.arange(-8.0, 8.0001, 0.25), "a_n": np.arange(-8.0, 8.0001, 0.25), "dt": np.arange(0.0, 0.3001, 0.01)}
CH = {"speed": 0, "a_tan": 1, "a_n": 2, "dt": 4, "heading_valid": 5}         # nflbowl.data.validation.CHANNELS order


def style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.labelsize": 11,
                         "legend.frameon": False, "lines.linewidth": 2.0, "font.size": 10.5})


def _fig(title: str, subtitle: str, ncols: int = 1) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, ncols, figsize=FIGSIZE, constrained_layout=True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="semibold")
    fig.text(0.01, 0.945, subtitle, ha="left", va="top", fontsize=10, color=INK2, wrap=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    return fig, ax


def pos_colors(positions: list[str]) -> dict[str, str]:
    return {p: SERIES[k % len(SERIES)] for k, p in enumerate(sorted(positions))}


# ============================================================================== exact chunked statistics of the bundle arrays
def sequence_statistics(b: Any, chunk: int = 32_768) -> tuple[Any, dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """(per-sequence summary frame, {channel: {domain: counts}} histograms, {domain: heading-valid share}) over EVERY valid frame."""
    import polars as pl

    dom = b.meta["domain"].to_numpy()
    n = b.n_sequences
    peak, n_valid, mean_abs_at = np.full(n, np.nan, np.float32), np.zeros(n, np.int32), np.full(n, np.nan, np.float32)
    hist = {c: {d: np.zeros(EDGES[c].size - 1, np.int64) for d in ("combine", "nfl")} for c in EDGES}
    hv_num, hv_den = {"combine": 0.0, "nfl": 0.0}, {"combine": 0.0, "nfl": 0.0}
    for s in range(0, n, chunk):
        X = np.asarray(b.X[s: s + chunk], dtype=np.float32)
        M = np.asarray(b.M[s: s + chunk], dtype=bool)
        d = dom[s: s + chunk]
        sp = np.where(M, X[..., CH["speed"]], np.nan)
        with np.errstate(invalid="ignore", all="ignore"):
            peak[s: s + chunk] = np.nanmax(np.where(np.isfinite(sp), sp, -np.inf), axis=1)
            mean_abs_at[s: s + chunk] = np.nanmean(np.where(M, np.abs(X[..., CH["a_tan"]]), np.nan), axis=1)
        n_valid[s: s + chunk] = M.sum(1)
        for dn in ("combine", "nfl"):
            rows = d == dn
            if not rows.any():
                continue
            Mr = M[rows]
            for c, e in EDGES.items():
                v = X[rows][..., CH[c]][Mr]
                v = v[np.isfinite(v)]
                hist[c][dn] += np.histogram(np.clip(v, e[0], e[-1] - 1e-9), bins=e)[0]
            hvv = X[rows][..., CH["heading_valid"]][Mr]
            hv_num[dn] += float(np.nansum(hvv))
            hv_den[dn] += float(np.isfinite(hvv).sum())
        del X, M
    peak[~np.isfinite(peak)] = np.nan
    seq = b.meta.select("idx", "domain", "nfl_id", "kind", "position_group").with_columns(pl.Series("n_valid", n_valid), pl.Series("peak_speed", peak),
                                                                                         pl.Series("mean_abs_a_tan", mean_abs_at))
    hv = {d: (hv_num[d] / hv_den[d]) if hv_den[d] else float("nan") for d in hv_num}
    return seq, hist, hv


def hist_frame(hist: dict[str, dict[str, np.ndarray]]) -> Any:
    import polars as pl

    rows = []
    for c, by in hist.items():
        e = EDGES[c]
        for d, h in by.items():
            tot = max(int(h.sum()), 1)
            rows += [{"channel": c, "domain": d, "lo": float(e[k]), "hi": float(e[k + 1]), "frames": int(h[k]), "share": float(h[k] / tot)} for k in range(h.size)]
    return pl.DataFrame(rows)


# ============================================================================== tables
def coverage_tables(b: Any) -> dict[str, Any]:
    import polars as pl

    m = b.meta
    per = m.group_by("nfl_id").agg(pl.col("position_group").first(), (pl.col("domain") == "combine").any().alias("combine"), (pl.col("domain") == "nfl").any().alias("nfl"))
    per = per.with_columns(pl.when(pl.col("combine") & pl.col("nfl")).then(pl.lit("both")).when(pl.col("combine")).then(pl.lit("combine_only")).otherwise(pl.lit("nfl_only")).alias("coverage"))
    cov = per.group_by("position_group", "coverage").len().rename({"len": "n_athletes"}).sort("position_group", "coverage")
    kinds = m.group_by("domain", "kind").len().rename({"len": "n_sequences"}).sort("domain", "n_sequences", descending=[False, True])
    out = {"eda_athlete_coverage": cov, "eda_sequences_by_kind": kinds,
           "eda_coverage_summary": per.group_by("coverage").len().rename({"len": "n_athletes"}).sort("coverage")}
    return out


def athletes_tables(b: Any) -> dict[str, Any]:
    import polars as pl

    from ..data.validation import M1_COLS, TRANSFER_TARGET

    a = b.athletes
    if a is None:
        return {}
    cols = [c for c in (*M1_COLS, "three_cone", "short_shuttle", TRANSFER_TARGET) if c in a.columns]
    miss = (a.group_by("position_group").agg([pl.col(c).is_null().mean().alias(c) for c in cols] + [pl.len().alias("n_athletes")]).sort("position_group"))
    lk = a.filter(pl.col("linked").fill_null(False)) if "linked" in a.columns else a
    num = [c for c in cols if lk[c].dtype.is_numeric()]
    cm = []
    if lk.height >= 10 and len(num) >= 2:
        X = lk.select(num).to_numpy().astype(float)
        for i, ci in enumerate(num):
            for j, cj in enumerate(num):
                ok = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
                r = float(np.corrcoef(X[ok, i], X[ok, j])[0, 1]) if ok.sum() >= 10 else float("nan")
                cm.append({"row": ci, "col": cj, "r": r, "n": int(ok.sum())})
    out = {"eda_athletes_missingness": miss, "eda_official_correlations": pl.DataFrame(cm) if cm else pl.DataFrame()}
    if TRANSFER_TARGET in a.columns:
        out["eda_target_by_position"] = (lk.filter(pl.col(TRANSFER_TARGET).is_not_null()).group_by("position_group")
                                         .agg(pl.len().alias("n"), pl.col(TRANSFER_TARGET).mean().alias("mean"), pl.col(TRANSFER_TARGET).std().alias("sd"),
                                              pl.col(TRANSFER_TARGET).quantile(0.1).alias("q10"), pl.col(TRANSFER_TARGET).median().alias("median"),
                                              pl.col(TRANSFER_TARGET).quantile(0.9).alias("q90")).sort("position_group"))
    return out


def leakage_checklist(b: Any, outer_folds: int, freeze: dict[str, Any] | None) -> Any:
    """Data-leakage risks and how the protocol neutralises each (facts from the bundle, not claims)."""
    import polars as pl

    fmap = dict(zip(*b.folds.select("nfl_id", "fold").to_dict(as_series=False).values()))
    ids = set(b.meta["nfl_id"].unique().to_list())
    fallback = sum(1 for i in ids if fmap.get(int(i)) is None)
    linked_nofold = 0
    if b.athletes is not None and "linked" in b.athletes.columns:
        lk = b.athletes.filter(pl.col("linked").fill_null(False))["nfl_id"].to_list()
        linked_nofold = sum(1 for i in lk if fmap.get(int(i)) is None)
    rows = [
        {"risk": "the same athlete in training and test folds", "fact": f"folds.parquet: one fold per athlete ({b.folds.height:,} rows, unique ids)",
         "mitigation": "athlete-grouped splits everywhere; ae.oof_ladder raises LeakageError on a split athlete", "status": "controlled"},
        {"risk": "held-out athlete's sequences of the OTHER domain in encoder training", "fact": "inductive encoders drop held-out athletes from both domains",
         "mitigation": "nflbowl.data.splits.split_indices(held_out) removes every sequence of the fold's athletes", "status": "controlled"},
        {"risk": "athletes without a manifest fold", "fact": f"{fallback:,} athletes use the 12F fallback fold nfl_id % {outer_folds}; {linked_nofold} of them are linked",
         "mitigation": "fallback is deterministic and athlete-level; linked athletes without a fold never enter the transfer ladder", "status": "controlled" if linked_nofold == 0 else "review"},
        {"risk": "target chosen after seeing Combine values", "fact": (f"12C freeze {freeze.get('estimator')} τ={freeze.get('tau')} at {freeze.get('frozen_at')}" if freeze else "freeze record not available here"),
         "mitigation": "dc.freeze_rule uses NFL game-split reliability only; write_freeze refuses a different later freeze", "status": "controlled" if freeze else "unverified here"},
        {"risk": "standardisation / PCA / FPCA / factor models fitted on test athletes", "fact": "every cross-athlete quantity is fitted per training fold",
         "mitigation": "fold-fitted blocks (ae.Model factory); validation check block_sees_training_only", "status": "controlled"},
        {"risk": "position-typical movement inflating retrieval", "fact": "candidate pools are same-position athletes",
         "mitigation": "within-position retrieval + shuffled-identity nulls", "status": "controlled"},
    ]
    return pl.DataFrame(rows)


# ============================================================================== figures
def fig_coverage(cov: Any) -> Any:
    style()
    fig, ax = _fig("Athlete coverage by position and domain", "Athletes (the independent unit) with Combine sequences only, NFL sequences only, or both. Cross-domain analyses use 'both'.")
    pos = sorted(cov["position_group"].unique().to_list())
    order = [("both", SERIES[2]), ("combine_only", SERIES[0]), ("nfl_only", SERIES[1])]
    left = np.zeros(len(pos))
    for lab, col in order:
        v = np.array([int(cov.filter((cov["position_group"] == p) & (cov["coverage"] == lab))["n_athletes"].sum()) for p in pos])
        ax.barh(pos, v, left=left, color=col, edgecolor=SURFACE, linewidth=2, label=lab.replace("_", " "), height=0.6)
        left += v
    for k, t in enumerate(left):
        ax.text(t, k, f"  {int(t)}", va="center", color=INK2)
    ax.set_xlabel("athletes")
    ax.legend(loc="lower right", ncols=3)
    ax.grid(axis="y", visible=False)
    return fig


def fig_kinds(kinds: Any) -> Any:
    style()
    fig, ax = _fig("Sequences per domain and movement kind", "Count of 64-frame sequences in the unified store (Combine drill attempts; NFL movement-demand windows).")
    lab = [f"{d} · {k}" for d, k in zip(kinds["domain"].to_list(), kinds["kind"].to_list())]
    cols = [DOMAIN_COLOR.get(d, INK2) for d in kinds["domain"].to_list()]
    ax.barh(lab[::-1], kinds["n_sequences"].to_list()[::-1], color=cols[::-1], height=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("sequences (log scale)")
    ax.grid(axis="y", visible=False)
    import matplotlib.patches as mp

    ax.legend(handles=[mp.Patch(color=DOMAIN_COLOR["combine"], label="Combine"), mp.Patch(color=DOMAIN_COLOR["nfl"], label="NFL")], loc="lower right")
    return fig


def fig_channels(hf: Any) -> Any:
    style()
    fig, axes = _fig("Movement-signal distributions by domain (exact, every valid frame)",
                     "Share of valid frames per bin. Different tasks: a standing-start drill vs unrestricted game movement — separability is expected, not a defect.", ncols=4)
    names = {"speed": "speed [yd/s]", "a_tan": "tangential acceleration [yd/s²]", "a_n": "lateral acceleration [yd/s²]", "dt": "sampling interval [s]"}
    for ax, c in zip(axes, ("speed", "a_tan", "a_n", "dt")):
        for d in ("combine", "nfl"):
            g = hf.filter((hf["channel"] == c) & (hf["domain"] == d))
            if g.height:
                mid = (g["lo"].to_numpy() + g["hi"].to_numpy()) / 2
                ax.plot(mid, g["share"].to_numpy(), color=DOMAIN_COLOR[d], label="Combine" if d == "combine" else "NFL")
        ax.set_xlabel(names[c])
        ax.set_ylabel("share of frames" if c == "speed" else "")
    axes[0].legend(loc="upper right")
    return fig


def fig_lengths_peak(seq: Any) -> Any:
    style()
    fig, axes = _fig("Sequence quality: valid length and peak speed", "Left: valid frames per 64-frame sequence. Right: per-sequence peak speed. Colour = domain.", ncols=2)
    for d in ("combine", "nfl"):
        g = seq.filter(seq["domain"] == d)
        if g.height:
            axes[0].hist(g["n_valid"].to_numpy(), bins=np.arange(0, 66, 2), histtype="step", linewidth=2, color=DOMAIN_COLOR[d], density=True, label="Combine" if d == "combine" else "NFL")
            pk = g["peak_speed"].drop_nans().drop_nulls().to_numpy()
            axes[1].hist(pk, bins=np.arange(0, 12.25, 0.25), histtype="step", linewidth=2, color=DOMAIN_COLOR[d], density=True, label="Combine" if d == "combine" else "NFL")
    axes[0].set_xlabel("valid frames")
    axes[0].set_ylabel("density")
    axes[1].set_xlabel("peak speed [yd/s]")
    axes[0].legend()
    return fig


def fig_heat(df: Any, row: str, col: str, val: str, title: str, subtitle: str, fmt: str = "{:.0f}", cmap: str = "Blues", vmin: float | None = None, vmax: float | None = None) -> Any:
    style()
    fig, ax = _fig(title, subtitle)
    rows = sorted(df[row].unique().to_list(), key=str)
    cols = sorted(df[col].unique().to_list(), key=str)
    M = np.full((len(rows), len(cols)), np.nan)
    for r in df.iter_rows(named=True):
        M[rows.index(r[row]), cols.index(r[col])] = r[val] if r[val] is not None else np.nan
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)), [str(c) for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [str(r) for r in rows])
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=9, color=INK if (vmax is None or M[i, j] < 0.6 * (vmax or 1)) else SURFACE)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def fig_target(a: Any, target: str) -> Any:
    import polars as pl

    style()
    fig, ax = _fig(f"NFL target {target} by position (linked athletes)", "Box = quartiles, whiskers = 5th–95th percentile; n per position in the label. Observed game movement, not capacity.")
    lk = a.filter(pl.col("linked").fill_null(False) & pl.col(target).is_not_null()) if "linked" in a.columns else a.filter(pl.col(target).is_not_null())
    pos = sorted(lk["position_group"].unique().to_list())
    data = [lk.filter(pl.col("position_group") == p)[target].to_numpy() for p in pos]
    bp = ax.boxplot(data, whis=(5, 95), patch_artist=True, showfliers=False, widths=0.5)
    pc = pos_colors(pos)
    for patch, p in zip(bp["boxes"], pos):
        patch.set_facecolor(pc[p])
        patch.set_alpha(0.55)
        patch.set_edgecolor(INK2)
    ax.set_xticks(range(1, len(pos) + 1), [f"{p} (n={len(d)})" for p, d in zip(pos, data)])
    ax.set_ylabel(f"{target} [yd/s²]")
    return fig


def fig_registry_coverage(reg: Any) -> Any:
    import polars as pl

    style()
    fig, axes = _fig("Combine / NFL / linked coverage and NFL exposure (12.0 cohort registry)",
                     "Left: athletes per position in the Combine registry, with NFL histograms, and linked with a fold. Right: NFL games per linked athlete.", ncols=2)
    pos = sorted([p for p in reg["position_group"].unique().to_list() if p is not None])
    w = 0.26
    for k, (flag, col, lab) in enumerate((("in_combine", SERIES[0], "Combine registry"), ("in_nfl", SERIES[1], "NFL histograms"), ("linked", SERIES[2], "linked + fold"))):
        if flag in reg.columns:
            v = [int(reg.filter((pl.col("position_group") == p) & pl.col(flag))["nfl_id"].n_unique()) for p in pos]
            axes[0].bar(np.arange(len(pos)) + (k - 1) * w, v, width=w, color=col, label=lab)
    axes[0].set_xticks(range(len(pos)), pos)
    axes[0].set_ylabel("athletes")
    axes[0].legend()
    axes[0].grid(axis="x", visible=False)
    if "nfl_games" in reg.columns:
        g = reg.filter(pl.col("linked") & pl.col("nfl_games").is_not_null())["nfl_games"].to_numpy()
        axes[1].hist(g, bins=30, color=SERIES[1], edgecolor=SURFACE)
        axes[1].set_xlabel("NFL games in the histograms")
        axes[1].set_ylabel("linked athletes")
    return fig


# ============================================================================== orchestration
def run_bundle_eda(run: Any, b: Any, outer_folds: int, freeze: dict[str, Any] | None, fold_table: Any, reuse: bool = True) -> list[tuple[str, Any]]:
    """Compute (or reload) every bundle EDA table, show + save every figure. Returns (title, table) pairs for the report."""
    import polars as pl

    from .stages.base import display_df

    st = "eda"
    if reuse and run.has_table("eda_sequence_summary") and run.has_table("eda_channel_histograms"):
        seq, hf = run.load_table("eda_sequence_summary"), run.load_table("eda_channel_histograms")
        hv = run.load_processed_dataset("eda_heading_valid_share") if (run.root / "processed_data" / "eda_heading_valid_share.json").is_file() else {}
        print("EDA: per-sequence summary and channel histograms reloaded from RUN_DIR (exact, computed earlier)")
    else:
        seq, hist, hv = sequence_statistics(b)
        hf = hist_frame(hist)
        run.save_processed_dataset(seq, "eda_sequence_summary", st, {"bundle": b.fingerprint, "rows": b.n_sequences})
        run.save_table(hf, "eda_channel_histograms", st)
        run.save_processed_dataset(hv, "eda_heading_valid_share", st, {"bundle": b.fingerprint})
    tabs = {**coverage_tables(b), **athletes_tables(b)}
    tabs["eda_fold_balance"] = fold_table
    tabs["eda_leakage_checklist"] = leakage_checklist(b, outer_folds, freeze)
    q = (seq.group_by("domain").agg(pl.len().alias("n_sequences"), pl.col("n_valid").mean().alias("mean_valid_frames"), (pl.col("n_valid") < 16).mean().alias("share_lt_16_valid"),
                                    pl.col("peak_speed").median().alias("median_peak_speed"), pl.col("mean_abs_a_tan").median().alias("median_mean_abs_a_tan"))
         .with_columns(pl.col("domain").replace_strict({k: v for k, v in hv.items()} if hv else {}, default=None, return_dtype=pl.Float64).alias("heading_valid_share")).sort("domain"))
    tabs["eda_sequence_quality"] = q
    for name, df in tabs.items():
        if df is not None and getattr(df, "height", 0):
            run.save_table(df, name, st)
    for k in ("eda_coverage_summary", "eda_sequence_quality", "eda_leakage_checklist"):
        display_df(tabs.get(k), strlen=140)
    run.show(fig_coverage(tabs["eda_athlete_coverage"]), "eda_athlete_coverage_by_position", st, {"coverage": tabs["eda_athlete_coverage"]})
    run.show(fig_kinds(tabs["eda_sequences_by_kind"]), "eda_sequences_by_domain_and_kind", st, {"kinds": tabs["eda_sequences_by_kind"]})
    run.show(fig_channels(hf), "eda_movement_signal_distributions_by_domain", st, {"histograms": hf})
    run.show(fig_lengths_peak(seq), "eda_sequence_length_and_peak_speed", st, {"summary": seq.select("domain", "n_valid", "peak_speed")})
    if fold_table is not None and fold_table.height:
        run.show(fig_heat(fold_table, "position_group", "fold", "n_athletes", "Fold balance: athletes per outer fold and position",
                          "Athlete-grouped outer folds of the 12.0 fold manifest (athletes without a manifest fold excluded)."), "eda_fold_balance_by_position", st, {"folds": fold_table})
    if "eda_athletes_missingness" in tabs and tabs["eda_athletes_missingness"].height:
        mm = tabs["eda_athletes_missingness"].drop("n_athletes").unpivot(index="position_group", variable_name="column", value_name="missing_share")
        run.show(fig_heat(mm, "position_group", "column", "missing_share", "Missingness of official measurements and the target (athletes.parquet)",
                          "Share of athletes per position with a missing value. Drill participation follows position and invitation (not at random).", fmt="{:.2f}", cmap="Oranges", vmin=0, vmax=1),
                 "eda_athletes_missingness_by_position", st, {"missingness": mm})
    if "eda_official_correlations" in tabs and tabs["eda_official_correlations"].height:
        run.show(fig_heat(tabs["eda_official_correlations"], "row", "col", "r", "Relationships between physical measurements (linked athletes)",
                          "Pearson r between official Combine measurements, draft year and the NFL target (pairwise complete; pooled over positions — position drives much of it).",
                          fmt="{:.2f}", cmap="RdBu_r", vmin=-1, vmax=1), "eda_official_measurement_correlations", st, {"correlations": tabs["eda_official_correlations"]})
    if b.athletes is not None and "nfl.q90_shared" in b.athletes.columns:
        run.show(fig_target(b.athletes, "nfl.q90_shared"), "eda_nfl_target_by_position", st, {"target": tabs.get("eda_target_by_position")})
    return [("Athlete coverage summary", tabs["eda_coverage_summary"]), ("Sequence quality by domain", q), ("Leakage-risk checklist", tabs["eda_leakage_checklist"]),
            ("Athletes per fold and position", fold_table)] + ([("NFL target by position", tabs["eda_target_by_position"])] if "eda_target_by_position" in tabs else [])


def run_foundation_eda(run: Any, ctx: Any) -> list[tuple[str, Any]]:
    """Registry-based coverage / exposure figure (after 12.0)."""
    import polars as pl

    from nflbdb.research.advanced_movement.m12 import foundation as fd

    reg = ctx.store.read_frame("m12_cohort_registry", fd.NB["12.0"])
    run.show(fig_registry_coverage(reg), "eda_registry_coverage_and_nfl_exposure", "eda", {"registry": reg.select([c for c in ("nfl_id", "position_group", "in_combine", "in_nfl", "linked", "nfl_games") if c in reg.columns])})
    cov = (reg.group_by("position_group").agg(pl.col("in_combine").sum().alias("combine"), pl.col("in_nfl").sum().alias("nfl"), pl.col("linked").sum().alias("linked"),
                                              *([pl.col("nfl_games").median().alias("median_nfl_games")] if "nfl_games" in reg.columns else [])).sort("position_group"))
    run.save_table(cov, "eda_registry_coverage", "eda")
    return [("Cohort registry coverage by position", cov)]


__all__ = ["DOMAIN_COLOR", "EDGES", "FIGSIZE", "SERIES", "coverage_tables", "fig_channels", "fig_coverage", "fig_heat", "hist_frame", "leakage_checklist", "run_bundle_eda",
           "run_foundation_eda", "sequence_statistics", "style"]
