"""Figures of notebook 04A (feature and characteristic atlas). Every function returns a Matplotlib figure; the notebook
passes it to ``AtlasStore.show`` (inline display, PNG 300 DPI + SVG, PDF for publication figures).

Conventions of the programme: project 16:9 style, units on every axis, the unit of analysis and the denominator in the
caption, 95 % athlete-level intervals. Colour identifies the feature FAMILY everywhere; the grade has its own colours.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS
from ..figkit import ci_errorbar, empty_panel, figure, finish
from .atlas import FAMILIES, GRADES, AtlasConfig

FAMILY_COLOR = {"F1": INK["secondary"], "F2": CATEGORICAL[0], "F3": CATEGORICAL[6], "F4": CATEGORICAL[3], "F5": CATEGORICAL[1],
                "F6": CATEGORICAL[2], "F7": INK["muted"]}
GRADE_COLOR = {GRADES[0]: STATUS["good"], GRADES[1]: CATEGORICAL[0], GRADES[2]: STATUS["warning"], GRADES[3]: INK["axis"]}


def _short(fid: str) -> str:
    return fid.split(".", 2)[-1] if fid.count(".") >= 2 else fid


def _num(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _arr(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([_num(r.get(key)) for r in rows])


def _forty(df: pl.DataFrame) -> pl.DataFrame:
    """Rows characterised by this atlas or by notebooks 02–03 (identifiers of other drills contain ':')."""
    return df.filter(~pl.col("feature_id").str.contains(":", literal=True))


def _family_legend(ax, families: Sequence[str], **kw) -> None:
    from matplotlib.patches import Patch

    seen = [f for f in FAMILIES if f in set(families)]
    ax.legend(handles=[Patch(color=FAMILY_COLOR[f], label=f"{f} · {FAMILIES[f]}") for f in seen], frameon=False, fontsize=8, **kw)


def _cmap(colors: Sequence[str]):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("atlas", list(colors))


def _hdots(ax, rows: Sequence[dict[str, Any]], key: str, colors: Sequence[str], ci: bool = True) -> None:
    pos = np.arange(len(rows), dtype=float)
    est, lo, hi = _arr(rows, key), _arr(rows, f"{key}_lower"), _arr(rows, f"{key}_upper")
    for i in range(len(rows)):
        if ci and np.isfinite(lo[i]) and np.isfinite(hi[i]):
            ax.plot([lo[i], hi[i]], [pos[i], pos[i]], color=colors[i], linewidth=1.5)
        if np.isfinite(est[i]):
            ax.plot([est[i]], [pos[i]], "o", color=colors[i], markersize=5)


# ============================================================================== 1 · inventory
def plot_inventory(dictionary: pl.DataFrame, status: pl.DataFrame, coverage: pl.DataFrame, c: AtlasConfig):
    fig, ax = figure(1, 2)
    a, b = ax[0]
    fams = list(FAMILIES)
    counts = dict(((r["family"], r["grade"]), r["len"]) for r in status.group_by("family", "grade").len().iter_rows(named=True))
    left = np.zeros(len(fams))
    for g in GRADES:
        v = np.array([counts.get((f, g), 0) for f in fams], dtype=float)
        a.barh(np.arange(len(fams)), v, left=left, color=GRADE_COLOR[g], label=g)
        left += v
    for i, f in enumerate(fams):
        a.text(left[i] + 0.5, i, f"{int(left[i])} of {dictionary.filter(pl.col('family') == f).height}", va="center", fontsize=8,
               color=INK["secondary"])
    a.set_yticks(np.arange(len(fams)))
    a.set_yticklabels([f"{f} · {FAMILIES[f]}" for f in fams], fontsize=9)
    a.invert_yaxis()
    a.set_xscale("symlog", linthresh=20)
    a.set_xlabel("feature definitions characterised (symmetric-log axis)")
    a.set_title("Definitions by family and screening grade", fontsize=11)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    cov = _forty(coverage).filter(pl.col("group_type") == "position").join(status.select("feature_id", "family"), on="feature_id")
    groups = sorted(cov["group"].unique().to_list())
    M = np.array([[cov.filter((pl.col("family") == f) & (pl.col("group") == g) & (pl.col("coverage_athletes") >= c.min_coverage)).height
                   for g in groups] for f in fams], dtype=float)
    T = np.array([cov.filter(pl.col("family") == f)["feature_id"].n_unique() for f in fams])
    if groups:
        b.imshow(M / np.maximum(T[:, None], 1), cmap=_cmap(["#ffffff", SEQUENTIAL_BLUE[4]]), vmin=0, vmax=1, aspect="auto")
        for i in range(len(fams)):
            for j in range(len(groups)):
                b.text(j, i, f"{int(M[i, j])} / {int(T[i])}" if T[i] else "–", ha="center", va="center", fontsize=9, color=INK["primary"])
        b.set_xticks(np.arange(len(groups)))
        b.set_xticklabels(groups)
        b.set_yticks(np.arange(len(fams)))
        b.set_yticklabels(fams)
        b.grid(False)
    else:
        empty_panel(b, "no scalar feature with a position breakdown")
    b.set_title(f"Scalar features measured on ≥ {100 * c.min_coverage:.0f} % of the athletes of a position", fontsize=11)
    n = coverage.filter((pl.col("group_type") == "all"))["n_athletes"].max() if coverage.height else 0
    finish(fig, f"Left, unit: feature definition (one row of the dictionary). Other-drill and event definitions (F2 / F3 beyond the forty) "
                "carry Phase 04 evidence only. Right, unit: scalar feature of the forty or official measurement; cell = features whose "
                f"share of tracked athletes with a value reaches the reference, of those of the family (denominator: up to {n} tracked "
                "athletes; learned representations have no per-position row). Grades are heuristic screening labels.",
           "Feature inventory by family and position")
    return fig


# ============================================================================== 2 · missingness and eligibility
def plot_missingness(coverage: pl.DataFrame, status: pl.DataFrame, c: AtlasConfig):
    cov = _forty(coverage)
    feats = [f for f in status.sort("family", "feature_id")["feature_id"].to_list() if f in set(cov["feature_id"].to_list())]
    cols = [("all", "ALL")] + [(t, g) for t in ("position", "cohort") for g in sorted(cov.filter(pl.col("group_type") == t)["group"].unique().to_list())]
    fig, ax = figure(1, 2, size=(12.8, max(7.2, 0.19 * len(feats) + 2.0)), gridspec_kw={"width_ratios": [5, 1]})
    a, b = ax[0]
    if not feats:
        empty_panel(a, "no scalar feature")
        empty_panel(b, "")
        finish(fig, "", "Missingness and eligibility map")
        return fig
    got = {(r["feature_id"], r["group_type"], r["group"]): r["coverage_athletes"] for r in cov.iter_rows(named=True)}
    M = np.array([[_num(got.get((f, t, g))) for t, g in cols] for f in feats])
    a.imshow(M, cmap=_cmap([STATUS["critical"], "#ffffff", SEQUENTIAL_BLUE[4]]), vmin=0, vmax=1, aspect="auto")
    for i in range(len(feats)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                a.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=6, color=INK["primary"])
    a.set_xticks(np.arange(len(cols)))
    a.set_xticklabels([g for _, g in cols], fontsize=8)
    a.set_yticks(np.arange(len(feats)))
    a.set_yticklabels([_short(f) for f in feats], fontsize=7)
    a.grid(False)
    a.set_title("Tracked athletes with a value (%)", fontsize=11)
    st = {r["feature_id"]: r for r in status.iter_rows(named=True)}
    for i, f in enumerate(feats):
        b.barh(i, 1.0, color=GRADE_COLOR[st[f]["grade"]])
        b.text(0.5, i, st[f]["transfer_eligibility"], ha="center", va="center", fontsize=6, color=INK["primary"])
    b.set_ylim(len(feats) - 0.5, -0.5)
    b.set_xticks([])
    b.set_yticks([])
    b.grid(False)
    b.set_title("Grade (colour) · eligibility", fontsize=10)
    finish(fig, "Unit: tracked athlete (≥ 1 eligible forty sequence). Cell = share of the athletes of the column with at least one value of "
                "the feature; a missing value is kept with its reason (table atlas_coverage), never filled. Speed-anchored features are "
                "missing when the recording starts above the anchor speed or stops before the window ends. Right: screening grade "
                f"(green Strong, blue Provisional, yellow Exploratory, grey Reject / not usable); reference coverage {c.min_coverage:g}.",
           "Missingness and eligibility map")
    return fig


# ============================================================================== 3 · reliability against redundancy
def _scatter_rel_red(ax, rows: Sequence[dict[str, Any]], c: AtlasConfig, annotate: bool = True, ci: bool = True) -> None:
    for r in rows:
        x, y = _num(r.get("redundancy_abs")), _num(r.get("reliability"))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        col = FAMILY_COLOR.get(r["family"], INK["muted"])
        if ci:
            ci_errorbar(ax, [x], [y], [_num(r.get("reliability_lower"))], [_num(r.get("reliability_upper"))], fmt="o", color=col,
                        capsize=2, markersize=5)
            lo, hi = _num(r.get("redundancy_lower")), _num(r.get("redundancy_upper"))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y, y], color=col, linewidth=0.9, alpha=0.7)
        else:
            ax.plot([x], [y], "o", color=col, markersize=4)
        if annotate:
            ax.annotate(_short(r["feature_id"]), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=6, color=INK["secondary"])
    ax.axhline(c.icc_reference, color=INK["muted"], linestyle="--", linewidth=1)
    ax.axhline(c.icc_floor, color=INK["muted"], linestyle=":", linewidth=1)
    ax.axvline(c.redundancy_reference, color=INK["muted"], linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.02)


def plot_reliability_vs_redundancy(chars: pl.DataFrame, status: pl.DataFrame, c: AtlasConfig):
    rows = (_forty(chars).filter((pl.col("scope") == "all") & pl.col("design").is_in(["repeated_attempt", "model_parameter", "functional_score"])
                                 & pl.col("retrieval_auc").is_null())
            .join(status.select("feature_id", "role"), on="feature_id").filter(pl.col("role") == "candidate").to_dicts())
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    _scatter_rel_red(a, rows, c)
    a.set_xlabel("stopwatch redundancy: max |ρ| with the official forty and 10-yard split (Spearman, within position)")
    a.set_ylabel("single-attempt reliability (ICC, within position)")
    a.text(0.02, 0.97, "repeatable, not explained by the stopwatch", transform=a.transAxes, fontsize=9, va="top", color=INK["secondary"])
    a.text(0.98, 0.97, "repeatable, redundant", transform=a.transAxes, fontsize=9, va="top", ha="right", color=INK["secondary"])
    _family_legend(a, [r["family"] for r in rows], loc="lower left")
    n = max((r.get("n_athletes_repeat") or 0 for r in rows), default=0)
    finish(fig, f"Unit: athlete (up to {n} with a repeated attempt for the vertical axis; athlete means for the horizontal axis). One dot "
                "per candidate scalar feature of the forty, functional PCA components included; whiskers = 95 % intervals (athlete "
                f"bootstrap for the ICC, Fisher z for ρ). Dashed lines: ICC {c.icc_reference:g} and |ρ| {c.redundancy_reference:g}; dotted: "
                f"ICC {c.icc_floor:g} (heuristic references). A low |ρ| alone is not evidence of new information: see the reliable "
                "share not explained by the official times in atlas_characteristics.", "Reliability against stopwatch redundancy")
    return fig


def plot_reliability_by_position(chars: pl.DataFrame, status: pl.DataFrame, c: AtlasConfig):
    d = (_forty(chars).filter(pl.col("scope").str.starts_with("position:") & pl.col("design").is_in(["repeated_attempt", "model_parameter"]))
         .join(status.select("feature_id", "role"), on="feature_id").filter(pl.col("role") == "candidate"))
    groups = sorted(d["scope"].unique().to_list())
    fig, ax = figure(1, max(len(groups), 1))
    for a, g in zip(ax[0], groups):
        rows = d.filter(pl.col("scope") == g).to_dicts()
        _scatter_rel_red(a, rows, c, annotate=False, ci=False)
        n = max((r.get("n_athletes_repeat") or 0 for r in rows), default=0)
        a.set_title(f"{g.split(':')[1]} · up to {n} athletes with a repeat", fontsize=10)
        a.set_xlabel("max |ρ| with official times")
    if groups:
        ax[0, 0].set_ylabel("ICC inside the position")
        _family_legend(ax[0, 0], d["family"].unique().to_list(), loc="lower left")
    else:
        empty_panel(ax[0, 0], "no position-level row")
    finish(fig, "Unit: athlete of one position group. Same axes as the pooled figure, estimated inside each group (no position "
                "adjustment is needed there). Intervals are not drawn: with two attempts and a few dozen athletes they are wide — read "
                "the pattern, not a single dot. Reference lines as in the pooled figure.", "Reliability against redundancy, by position")
    return fig


# ============================================================================== 4 · reliability by family
def plot_reliability_by_family(chars: pl.DataFrame, status: pl.DataFrame, c: AtlasConfig):
    d = (_forty(chars).filter((pl.col("scope") == "all") & pl.col("reliability").is_not_null())
         .join(status.select("feature_id", "role", "grade"), on="feature_id").filter(pl.col("role").is_in(["candidate", "nuisance"])))
    fams = [f for f in FAMILIES if f in set(d["family"].to_list())]
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    for k, f in enumerate(fams):
        rows = d.filter(pl.col("family") == f).sort("reliability").to_dicts()
        xs = k + np.linspace(-0.32, 0.32, len(rows)) if len(rows) > 1 else np.array([float(k)])
        for x, r in zip(xs, rows):
            vector = r.get("retrieval_auc") is not None
            ci_errorbar(a, [x], [r["reliability"]], [_num(r["reliability_lower"])], [_num(r["reliability_upper"])],
                        fmt="s" if vector else "o", color=FAMILY_COLOR[f], capsize=2, markersize=5,
                        markerfacecolor="none" if vector else FAMILY_COLOR[f])
        if rows:
            a.text(k, 1.04, f"median {np.median([r['reliability'] for r in rows]):.2f} · n = {len(rows)}", ha="center", fontsize=8,
                   color=INK["secondary"])
    a.axhline(c.icc_reference, color=INK["muted"], linestyle="--", linewidth=1, label=f"reference {c.icc_reference:g}")
    a.axhline(c.icc_floor, color=INK["muted"], linestyle=":", linewidth=1, label=f"floor {c.icc_floor:g}")
    a.set_xticks(np.arange(len(fams)))
    a.set_xticklabels([f"{f}\n{FAMILIES[f]}" for f in fams], fontsize=9)
    a.set_ylabel("single-attempt reliability, within position")
    a.set_ylim(-0.1, 1.1)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    n = d["n_athletes_repeat"].max() if d.height else 0
    finish(fig, f"Unit: athlete (up to {n} with a repeated attempt). One marker per feature, sorted inside its family; whisker = 95 % "
                "athlete bootstrap. Filled circle = stratified ICC(1) of a scalar; open square = trace reliability of a whole vector "
                "(rotation-invariant; a different estimator, shown for context and never compared coordinate by coordinate). F7 are "
                "acquisition variables: their repeatability describes the recording set-up, not the athlete.",
           "Reliability distribution by feature family")
    return fig


# ============================================================================== 5 · quality filters and retention
def plot_quality_retention(retention: pl.DataFrame, reliability: pl.DataFrame, features: Sequence[tuple[str, str]]):
    """``features``: (feature_id, label) whose ICC is shown under each quality filter."""
    fig, ax = figure(1, 2)
    a, b = ax[0]
    scen = ["S0_all", "S2_signal", "S3_no_boundary"]
    groups = [("all", "ALL")] + [("position", g) for g in sorted(retention.filter(pl.col("group_type") == "position")["group"].unique().to_list())]
    for k, (t, g) in enumerate(groups):
        d = retention.filter((pl.col("group_type") == t) & (pl.col("group") == g))
        got = dict(zip(d["scenario"].to_list(), d["n_athletes_repeat"].to_list()))
        y = [got.get(s, 0) for s in scen]
        a.plot(np.arange(3), y, marker="o", linewidth=2.2 if t == "all" else 1.2, color=INK["primary"] if t == "all" else CATEGORICAL[(k - 1) % 8],
               label=f"{g} ({y[0]} → {y[-1]})")
    a.set_xticks(np.arange(3))
    a.set_xticklabels(["every eligible attempt", "signal-quality filter", "+ no boundary peak"])
    a.set_ylabel("athletes with a repeated attempt")
    a.set_ylim(bottom=0)
    a.set_title("Athletes kept by each nested quality filter", fontsize=11)
    a.legend(frameon=False, fontsize=8)
    scopes = ["position_adjusted", "scenario:S2_signal", "scenario:S3_no_boundary"]
    for k, (fid, label) in enumerate(features):
        d = reliability.filter(pl.col("feature_id") == fid)
        got = dict(zip(d["scope"].to_list(), d["icc"].to_list()))
        b.plot(np.arange(3), [_num(got.get(s)) for s in scopes], marker="o", linewidth=1.4, color=CATEGORICAL[k % 8], label=label)
    b.set_xticks(np.arange(3))
    b.set_xticklabels(["every eligible attempt", "signal-quality filter", "+ no boundary peak"])
    b.set_ylabel("ICC within position")
    b.set_ylim(-0.05, 1.02)
    b.set_title("Reliability of selected features under each filter", fontsize=11)
    b.legend(frameon=False, fontsize=8)
    finish(fig, "Left, unit: athlete; the count is athletes who still have two attempts after the filter (first → last in the legend). "
                "Right, unit: athlete with a repeat under the filter; point estimates only (the athletes differ between filters, so the "
                "three values are not paired). A stricter filter selects athletes by signal quality: a gain in ICC bought with athletes "
                "is reported, not adopted.", "Quality filtering against athlete retention")
    return fig


# ============================================================================== 6 · recording-window sensitivity
def plot_window_sensitivity(chars: pl.DataFrame, status: pl.DataFrame, c: AtlasConfig):
    d = (_forty(chars).filter((pl.col("scope") == "all") & pl.col("window_r2").is_not_null())
         .join(status.select("feature_id", "role"), on="feature_id").filter(pl.col("role") == "candidate").sort("window_r2"))
    rows = d.to_dicts()
    fig, ax = figure(1, 1, size=(12.8, max(7.2, 0.2 * len(rows) + 2.2)))
    a = ax[0, 0]
    if not rows:
        empty_panel(a, "no candidate with a window estimate")
    else:
        _hdots(a, rows, "window_r2", [FAMILY_COLOR.get(r["family"], INK["muted"]) for r in rows])
        coh = np.array([_num(r.get("cohort_probe_gain")) if r.get("retrieval_auc") is not None else _num(r.get("cohort_r2")) for r in rows])
        a.plot(coh, np.arange(len(rows)), "x", color=INK["primary"], markersize=5, label="draft cohort (share of athlete variance; vectors: probe gain)")
        a.axvline(c.window_reference, color=INK["muted"], linestyle="--", linewidth=1, label=f"reference {c.window_reference:g}")
        a.set_yticks(np.arange(len(rows)))
        a.set_yticklabels([_short(r["feature_id"]) for r in rows], fontsize=7)
        a.set_xlim(-0.05, 1.02)
        a.set_xlabel("share of the attempt-to-attempt change explained by the change of the recording window")
        handles, labels = a.get_legend_handles_labels()
        a.legend(handles, labels, frameon=False, fontsize=8, loc="lower right")
    n = max((r.get("n_athletes_repeat") or 0 for r in rows), default=0)
    finish(fig, f"Unit: athlete with two attempts (up to {n}). Dot = R² of (attempt 2 − attempt 1) of the feature on the change of "
                "first-frame speed, recorded duration and frames after the speed peak; bar = 95 % athlete bootstrap. An athlete does "
                "not change between two attempts, his recording does: a large share means the feature follows the recording. Cross = "
                "dependence on the draft cohort (sensor generation). Colour = family; vectors use the trace R² of notebook 03.",
           "Recording-window sensitivity of the candidate features")
    return fig


# ============================================================================== 7 · redundancy matrix and clusters
def plot_redundancy_matrix(matrix: pl.DataFrame, clusters: pl.DataFrame, c: AtlasConfig):
    order = clusters.sort("display_order")
    feats = order["feature_id"].to_list()
    fig, ax = figure(1, 1, size=(12.8, max(9.0, 0.24 * len(feats) + 2.5)))
    a = ax[0, 0]
    if not feats:
        empty_panel(a, "no feature with adequate support")
        finish(fig, "", "Feature redundancy matrix")
        return fig
    idx = {f: i for i, f in enumerate(feats)}
    M = np.full((len(feats), len(feats)), np.nan)
    for r in matrix.iter_rows(named=True):
        if r["feature_a"] in idx and r["feature_b"] in idx and r["rho"] is not None:
            M[idx[r["feature_a"]], idx[r["feature_b"]]] = r["rho"]
    im = a.imshow(M, cmap=_cmap([CATEGORICAL[1], "#ffffff", CATEGORICAL[0]]), vmin=-1, vmax=1, aspect="equal")
    labels = order["cluster"].to_numpy()
    for k in np.flatnonzero(np.diff(labels) != 0):
        a.axhline(k + 0.5, color=INK["primary"], linewidth=0.8)
        a.axvline(k + 0.5, color=INK["primary"], linewidth=0.8)
    reps = set(order["representative"].to_list())
    a.set_xticks(np.arange(len(feats)))
    a.set_xticklabels([_short(f) for f in feats], rotation=90, fontsize=6)
    a.set_yticks(np.arange(len(feats)))
    a.set_yticklabels([("● " if f in reps else "") + _short(f) for f in feats], fontsize=6)
    a.grid(False)
    fig.colorbar(im, ax=a, shrink=0.6, label="Spearman ρ within position")
    n_cl = int(order.filter(pl.col("cluster_size") > 1)["cluster"].n_unique())
    finish(fig, "Unit: tracked athlete (official value, or mean of the athlete's attempts; pairwise-complete, position means removed). "
                f"Black lines separate the groups joined by average linkage while their mean |ρ| is at least {c.cluster_rho:g}: "
                f"{n_cl} group(s) of more than one feature. ● = member with the highest reliability, the natural representative. Three "
                "acquisition variables are included on purpose: a feature that groups with them follows the recording.",
           "Feature redundancy matrix and interpretable clusters")
    return fig


# ============================================================================== 8 · reliable speed against unreliable derivatives
def plot_repeat_examples(attempt_values: pl.DataFrame, windows: pl.DataFrame, reliability: pl.DataFrame,
                         features: Sequence[tuple[str, str, str]]):
    """``features``: (feature_id, label, unit). Attempt 1 against attempt 2 of every athlete with both."""
    fig, ax = figure(2, 2)
    order = windows.select("seq_id", "attempt_order", "position_group")
    positions = sorted(windows["position_group"].drop_nulls().unique().to_list())
    for a, spec in zip(ax.ravel(), list(features)[:4] + [None] * 4):
        if spec is None:
            empty_panel(a, "no feature")
            continue
        fid, label, unit = spec
        d = (attempt_values.filter((pl.col("feature_id") == fid) & pl.col("value").is_not_null()).join(order, on="seq_id")
             .sort("nfl_id", "attempt_order").group_by("nfl_id", maintain_order=True)
             .agg(pl.col("value").get(0).alias("a1"), pl.col("value").get(1, null_on_oob=True).alias("a2"), pl.col("position_group").first())
             .drop_nulls("a2"))
        if not d.height:
            empty_panel(a, f"{label}: no athlete with two values")
            continue
        for k, p in enumerate(positions):
            g = d.filter(pl.col("position_group") == p)
            a.plot(g["a1"], g["a2"], "o", color=CATEGORICAL[k % 8], markersize=3, alpha=0.75, label=p)
        lo, hi = float(min(d["a1"].min(), d["a2"].min())), float(max(d["a1"].max(), d["a2"].max()))
        a.plot([lo, hi], [lo, hi], color=INK["muted"], linewidth=1, linestyle="--")
        icc = reliability.filter((pl.col("feature_id") == fid) & (pl.col("scope") == "position_adjusted"))
        text = f"ICC within position {icc['icc'][0]:.2f} [{_num(icc['ci_lower'][0]):.2f}, {_num(icc['ci_upper'][0]):.2f}]" if icc.height and icc["icc"][0] is not None else ""
        a.set_title(f"{label} · {d.height} athletes · {text}", fontsize=9)
        a.set_xlabel(f"first attempt ({unit})")
        a.set_ylabel(f"second attempt ({unit})")
    ax[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    finish(fig, "Unit: athlete with two attempts carrying a value (first two in time). Dashed line = identical attempts. The pooled cloud "
                "of a speed looks tight partly because position groups differ; the ICC in each title is computed within position. A "
                "derivative taken on a single frame scatters far from the line: its attempt-to-attempt noise is of the size of the "
                "differences between athletes.", "Reliable speed measurements against unreliable derivatives")
    return fig


# ============================================================================== 9 · learned representations in context
def plot_learned_in_context(chars: pl.DataFrame, status: pl.DataFrame, c: AtlasConfig):
    d = _forty(chars).filter(pl.col("scope") == "all").join(status.select("feature_id", "role", "support"), on="feature_id")
    learned = d.filter(pl.col("family") == "F5").sort("feature_id").to_dicts()
    context = []
    for f in ("F2", "F6", "F4"):
        best = d.filter((pl.col("family") == f) & (pl.col("role") == "candidate") & (pl.col("support") == "adequate")
                        & pl.col("reliability").is_not_null()).sort("reliability", descending=True).head(2)
        context += best.to_dicts()
    rows = context + learned
    fig, ax = figure(1, 3, size=(12.8, max(7.2, 0.24 * len(rows) + 2.4)))
    if not rows:
        for a in ax[0]:
            empty_panel(a, "no row")
        finish(fig, "", "Learned representations in the feature-characterisation context")
        return fig
    colors = [FAMILY_COLOR.get(r["family"], INK["muted"]) for r in rows]
    panels = (("reliability", "single-attempt reliability\n(ICC · trace ICC for vectors)", c.icc_reference, True),
              ("_stopwatch", "stopwatch relation\n(scalar: R² from official times · vector ◇: forty explained)", None, False),
              ("window_r2", "attempt change explained\nby the recording window", c.window_reference, True))
    for j, (key, title, line, has_ci) in enumerate(panels):
        a = ax[0, j]
        if key == "_stopwatch":
            for i, r in enumerate(rows):
                vector = r.get("retrieval_auc") is not None
                x = _num(r.get("forty_from_feature_r2")) if vector else _num(r.get("r2_stopwatch"))
                if np.isfinite(x):
                    a.plot([x], [i], "D" if vector else "o", color=colors[i], markersize=5, markerfacecolor="none" if vector else colors[i])
        else:
            _hdots(a, rows, key, colors, ci=has_ci)
        if line is not None:
            a.axvline(line, color=INK["muted"], linestyle="--", linewidth=1)
        a.set_xlim(-0.05, 1.02)
        a.set_ylim(len(rows) - 0.5, -0.5)
        a.set_yticks(np.arange(len(rows)))
        a.set_yticklabels([_short(r["feature_id"]) for r in rows] if j == 0 else [], fontsize=7)
        if context and learned:
            a.axhline(len(context) - 0.5, color=INK["grid"], linewidth=1)
        a.set_title(title, fontsize=9)
    _family_legend(ax[0, 2], [r["family"] for r in rows], loc="lower right")
    finish(fig, "Unit: athlete. Above the line: the two most reliable hand-defined candidates of the kinematic, sustained-acceleration and "
                "sprint-model families, for scale. Below: functional PCA components, and each representation as a whole vector (FPCA "
                "K scores, TCN-A, TCN-B). Estimators differ by design and are named in atlas_characteristics: a scalar has an ICC, a "
                "vector a rotation-invariant trace reliability; the middle panel shows how much of the scalar the official times "
                "explain, and for a vector how much of the official forty the vector explains (held out, beyond position). Bars = 95 %.",
           "Functional PCA and encoders in the feature-characterisation context")
    return fig


# ============================================================================== 10 · leading candidates
def plot_leading_candidates(chars: pl.DataFrame, status: pl.DataFrame, c: AtlasConfig):
    keep = _forty(status).filter((pl.col("role") == "candidate") & pl.col("grade").is_in(list(GRADES[:2])))
    rows = (_forty(chars).filter(pl.col("scope") == "all").join(keep.select("feature_id", "grade", "stopwatch_relation", "acquisition"),
                                                                   on="feature_id").sort("reliability", descending=True, nulls_last=True).to_dicts())
    fig, ax = figure(1, 4, size=(12.8, max(7.2, 0.3 * len(rows) + 2.4)))
    if not rows:
        for a in ax[0]:
            empty_panel(a, "no candidate graded Strong or Provisional")
        finish(fig, "No candidate reaches the Provisional grade in this run. In smoke mode this is expected (too few athletes).",
               "Leading characteristic candidates")
        return fig
    colors = [GRADE_COLOR[r["grade"]] for r in rows]
    panels = (("reliability", "reliability", c.icc_reference, True), ("redundancy_abs", "max |ρ| with official times", c.redundancy_reference, True),
              ("reliable_unique_share", "reliable share not\nexplained by the stopwatch", c.unique_margin, False),
              ("window_r2", "attempt change explained\nby the window", c.window_reference, True))
    for j, (key, title, line, has_ci) in enumerate(panels):
        a = ax[0, j]
        _hdots(a, rows, key, colors, ci=has_ci)
        a.axvline(line, color=INK["muted"], linestyle="--", linewidth=1)
        a.set_xlim(-0.05, 1.02)
        a.set_ylim(len(rows) - 0.5, -0.5)
        a.set_yticks(np.arange(len(rows)))
        a.set_yticklabels([f"{_short(r['feature_id'])}  [{r['stopwatch_relation']}]" for r in rows] if j == 0 else [], fontsize=7)
        a.set_title(title, fontsize=9)
    finish(fig, "Unit: athlete. Candidates graded Strong (green) or Provisional (blue) by the screening rules, most reliable first; the "
                "label in brackets is the relation to the stopwatch. Bars = 95 % intervals; dashed lines = heuristic reference lines. "
                "Vector representations have no |ρ| and no unique share (see notebook 04B). A candidate here is a measurement worth "
                "testing, not a scouting metric: no NFL value was read.", "Leading characteristic candidates")
    return fig
