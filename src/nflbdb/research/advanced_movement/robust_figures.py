"""Figures of notebook 04B (robust validation of the encoder embeddings). Every function returns a Matplotlib figure;
the notebook passes it to ``AtlasStore.show``.

Conventions of the programme. Colour identifies the KIND of representation everywhere: orange = the primary encoder
(TCN-B, pre-declared seed), light orange = its other training seeds, blue = TCN-A, grey = FPCA benchmark, green =
training-fitted projections, dark grey = controls (untrained encoder, tracked top speed). Intervals are 95 %,
athlete-level; for matched retrieval they resample athletes as queries and as impostors.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, STATUS
from ..figkit import ci_errorbar, empty_panel, figure, finish
from .fpca import SPEC

KIND_COLOR = {"primary": CATEGORICAL[1], "seed replicate": "#f4a987", "baseline": INK["secondary"], "encoder (reconstruction only)": CATEGORICAL[0],
              "projection": CATEGORICAL[2], "control": INK["muted"], "sensitivity": CATEGORICAL[3], "other": INK["axis"]}
KIND_ORDER = list(KIND_COLOR)


def _num(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _rows(quality: pl.DataFrame, kinds: Sequence[str] | None = None) -> list[dict[str, Any]]:
    q = quality if kinds is None else quality.filter(pl.col("kind").is_in(list(kinds)))
    return sorted(q.iter_rows(named=True), key=lambda r: (KIND_ORDER.index(r["kind"]) if r["kind"] in KIND_ORDER else 99, r["method"]))


def _color(r: dict[str, Any]) -> str:
    return KIND_COLOR.get(r.get("kind", "other"), INK["axis"])


def _kind_legend(ax, kinds: Sequence[str], **kw) -> None:
    from matplotlib.patches import Patch

    seen = [k for k in KIND_ORDER if k in set(kinds)]
    ax.legend(handles=[Patch(color=KIND_COLOR[k], label=k) for k in seen], frameon=False, fontsize=8, **kw)


def _xaxis(ax, rows: Sequence[dict[str, Any]]) -> np.ndarray:
    pos = np.arange(len(rows), dtype=float)
    ax.set_xticks(pos)
    ax.set_xticklabels([r["method"] for r in rows], rotation=35, ha="right", fontsize=7)
    return pos


# ============================================================================== 1 · training seeds
def plot_seed_distribution(runs: pl.DataFrame, summary: pl.DataFrame, metrics: Sequence[tuple[str, str]], references: dict[str, dict[str, float]]):
    """``metrics``: (metric, axis label). ``references``: label → {metric: value} drawn as horizontal lines (FPCA, control)."""
    fig, ax = figure(2, int(np.ceil(len(metrics) / 2)))
    styles = [("--", INK["secondary"]), (":", INK["muted"]), ("-.", CATEGORICAL[0])]
    for a, (metric, label) in zip(ax.ravel(), metrics):
        d = runs.filter(pl.col("metric") == metric).sort("training_seed")
        if not d.height or d["value"].null_count() == d.height:
            empty_panel(a, f"{label}: not computed")
            continue
        x = np.arange(d.height, dtype=float)
        v, lo, hi = (np.array([_num(t) for t in d[c].to_list()]) for c in ("value", "lower", "upper"))
        s = summary.filter(pl.col("metric") == metric)
        if s.height and s["sd_seed"][0] is not None:
            m, sd = s["mean"][0], s["sd_seed"][0]
            a.axhspan(m - sd, m + sd, color=CATEGORICAL[1], alpha=0.12, lw=0)
            a.axhline(m, color=CATEGORICAL[1], linewidth=1)
            a.set_title(f"{label}\nmean {m:.3f} · SD between seeds {sd:.3f} · athlete SE {_num(s['se_athlete'][0]):.3f}", fontsize=9)
        else:
            a.set_title(label, fontsize=9)
        ci_errorbar(a, x, v, lo, hi, fmt="o", color=CATEGORICAL[1], capsize=3, markersize=7)
        for k, (name, ref) in enumerate(references.items()):
            if metric in ref and ref[metric] is not None:
                a.axhline(ref[metric], linestyle=styles[k % 3][0], color=styles[k % 3][1], linewidth=1, label=name)
        a.set_xticks(x)
        a.set_xticklabels([f"seed {t}" for t in d["training_seed"].to_list()], fontsize=8)
        a.set_xlim(-0.6, d.height - 0.4)
        if a.get_legend_handles_labels()[0]:
            a.legend(frameon=False, fontsize=7)
    for a in ax.ravel()[len(metrics):]:
        a.set_axis_off()
    finish(fig, "Unit of a dot: one training seed of the primary encoder (same architecture, optimiser, budget, input, folds and "
                "augmentations; only initial weights, batch order, masks and noise differ). Whisker = 95 % athlete interval of THAT "
                "seed (athlete sampling). Orange band = mean ± SD between seeds (training variability). The two are different sources "
                "and are reported side by side, never merged. No seed is selected: the first one was declared primary beforehand. "
                "With three seeds the SD is a rough figure.", "Three-seed robustness of the primary encoder")
    return fig


# ============================================================================== 2 · matched against unmatched retrieval
def plot_matched_retrieval(matched_summary: pl.DataFrame, quality: pl.DataFrame, subset: str = "role:confirm"):
    rows = [r for r in _rows(quality) if r["kind"] != "projection"]
    ms = {r["method"]: r for r in matched_summary.filter(pl.col("subset") == subset).iter_rows(named=True)}
    rows = [r for r in rows if r["method"] in ms]
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    if not rows:
        empty_panel(a, "no matched comparison")
    else:
        pos = _xaxis(a, rows)
        for i, r in enumerate(rows):
            m, c = ms[r["method"]], _color(r)
            a.plot([pos[i] - 0.12, pos[i] + 0.12], [_num(m["auc_position"]), _num(m["auc_matched"])], color=c, linewidth=1, alpha=0.6)
            ci_errorbar(a, [pos[i] - 0.12], [_num(m["auc_position"])], [_num(m["auc_position_lower"])], [_num(m["auc_position_upper"])],
                        fmt="o", color=c, capsize=2, markersize=6, markerfacecolor="none")
            ci_errorbar(a, [pos[i] + 0.12], [_num(m["auc_matched"])], [_num(m["auc_matched_lower2"])], [_num(m["auc_matched_upper2"])],
                        fmt="o", color=c, capsize=3, markersize=7)
        a.plot([], [], "o", color=INK["primary"], markerfacecolor="none", label="gallery = every athlete of the position")
        a.plot([], [], "o", color=INK["primary"], label="gallery = matched on forty, 10-yard, weight, height")
        a.axhline(0.5, color=INK["muted"], linestyle="--", linewidth=1, label="chance")
        a.set_ylabel("same-athlete discrimination AUC")
        a.set_ylim(0.3, 1.02)
        a.legend(frameon=False, fontsize=8, loc="lower left")
    one = next(iter(ms.values()), {})
    finish(fig, f"Unit: {subset} held-out athlete with two embedded attempts and at least one matched impostor (n = {one.get('n_athletes', 0)}; "
                f"{one.get('n_comparisons', 0)} query × impostor comparisons from {one.get('n_distinct_impostors', 0)} distinct impostors, "
                f"one athlete serving at most {one.get('impostor_reuse_max', 0)} queries). Each pair of dots is the SAME athletes under two "
                "galleries, joined by a line. Open: whisker = athlete bootstrap. Filled: whisker = athletes resampled as queries and as "
                "impostors together. A representation that only re-encodes the stopwatch and body size falls toward the dashed line "
                "when the gallery is matched.", "Matched-stopwatch retrieval: same athlete against look-alike impostors")
    return fig


# ============================================================================== 3 · raw against residual embeddings
def plot_residual_audit(audit: pl.DataFrame, methods: Sequence[tuple[str, str]], subset: str = "role:confirm"):
    """``methods``: (method, label). One panel per method; x = how the official results were removed."""
    variants = [("none", "raw", "raw"), ("official", "crossfit", "cross-fitted\n(notebook 03)"),
                ("official", "crossfit_permuted", "cross-fitted,\nofficial PERMUTED"), ("official", "train_fit", "fitted on\ntraining athletes"),
                ("official", "heldout_in_sample", "in sample"), ("official_window", "crossfit", "+ window,\ncross-fitted"),
                ("official_window", "train_fit", "+ window,\ntraining fit")]
    marks = {"euclidean": ("o", "-", "Euclidean (standardised embedding)"), "euclidean_restandardised": ("s", "--", "Euclidean, residual re-standardised"),
             "cosine": ("^", ":", "cosine")}
    fig, ax = figure(1, max(len(methods), 1))
    for a, (method, label) in zip(ax[0], methods):
        d = audit.filter((pl.col("method") == method) & (pl.col("subset") == subset))
        if not d.height:
            empty_panel(a, f"{label}: no audit row")
            continue
        got = {(r["covariates"], r["mode"], r["metric"]): r for r in d.iter_rows(named=True)}
        x = np.arange(len(variants), dtype=float)
        for k, (metric, (marker, style, name)) in enumerate(marks.items()):
            rows = [got.get((c, m, metric), {}) for c, m, _ in variants]
            y = np.array([_num(r.get("auc")) for r in rows])
            a.plot(x + 0.08 * (k - 1), y, linestyle=style, color=CATEGORICAL[k], linewidth=1, alpha=0.7)
            ci_errorbar(a, x + 0.08 * (k - 1), y, [_num(r.get("auc_lower")) for r in rows], [_num(r.get("auc_upper")) for r in rows],
                        fmt=marker, color=CATEGORICAL[k], capsize=2, markersize=5, label=name)
        raw = _num(got.get(("none", "raw", "euclidean"), {}).get("auc"))
        if np.isfinite(raw):
            a.axhline(raw, color=INK["muted"], linewidth=1, linestyle="--")
        a.axhline(0.5, color=INK["grid"], linewidth=1)
        a.set_xticks(x)
        a.set_xticklabels([v[2] for v in variants], fontsize=7)
        a.set_ylim(0.3, 1.02)
        a.set_ylabel("repeat-attempt AUC, within position")
        n = max((r["n_athletes"] for r in got.values()), default=0)
        a.set_title(f"{label} · n = {n}", fontsize=10)
    ax[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    finish(fig, f"Unit: {subset} held-out athlete with both official times; the same athletes, positive pair (first two attempts) and "
                "gallery (position, fold) at every x. x = how the linear prediction from official forty, 10-yard, weight and height "
                "(then also the recording window) was removed. Dashed line = raw value. Permuting the official results between "
                "athletes removes nothing real: any gain there is produced by the procedure. One function fitted on training "
                "athletes removes the same vector from both attempts of an athlete. Whisker = 95 % athlete bootstrap.",
           "Raw against residual embeddings: what does removing the stopwatch do?")
    return fig


# ============================================================================== 4 · dimension against stability
def plot_dimension_stability(dims: pl.DataFrame, primary: str):
    panels = (("auc", "repeat-attempt AUC (confirm)", True), ("matched_auc", "matched-stopwatch AUC", True), ("icc_trace", "trace reliability", False),
              ("effective_rank", "effective rank (directions used)", False), ("forty_partial_r2", "official forty explained beyond position", False),
              ("window_r2", "attempt change explained by the window", False))
    fig, ax = figure(2, 3)
    proj = dims.filter(pl.col("method").str.contains("|pc", literal=True))
    refs = dims.filter(~pl.col("method").str.contains("|pc", literal=True))
    bases = sorted(proj["base_method"].unique().to_list(), key=lambda m: (m != primary, m))
    for a, (key, label, has_ci) in zip(ax.ravel(), panels):
        for j, base in enumerate(bases):
            d = proj.filter(pl.col("base_method") == base).sort("k")
            if not d.height:
                continue
            x, y = d["k"].to_numpy().astype(float), np.array([_num(v) for v in d[key].to_list()])
            col = KIND_COLOR["primary"] if base == primary else KIND_COLOR["seed replicate"]
            a.plot(x * (1 + 0.03 * (j - 1)), y, marker="o", markersize=4, linewidth=1.2, color=col, label=f"{base} (seed {d['training_seed'][0]})")
            if has_ci:
                ci_errorbar(a, x * (1 + 0.03 * (j - 1)), y, [_num(v) for v in d[f"{key}_lower"].to_list()],
                            [_num(v) for v in d[f"{key}_upper"].to_list()], fmt="none", color=col, capsize=2)
        for r in refs.iter_rows(named=True):
            if r[key] is not None:
                fp_ = r["method"].startswith("FPCA")
                a.plot([r["k"]], [r[key]], "D" if fp_ else "s", color=KIND_COLOR["baseline"] if fp_ else KIND_COLOR["sensitivity"],
                       markersize=6, markerfacecolor="none", label=r["method"])
        if key == "effective_rank":
            lim = float(dims["k"].max()) if dims.height else 8.0
            a.plot([1, lim], [1, lim], color=INK["grid"], linewidth=1)
        a.set_xscale("log", base=2)
        ks = sorted(set(dims["k"].to_list()))
        a.set_xticks(ks)
        a.set_xticklabels([str(k) for k in ks])
        a.set_xlabel("dimensions kept")
        a.set_title(label, fontsize=10)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    ax[0, 0].legend(handles, labels, frameon=False, fontsize=7, loc="lower right")
    finish(fig, "Unit: athlete (AUC panels: confirm-role athletes; whiskers 95 %). Lines: each training seed of the primary encoder, its "
                "embedding projected on the first k principal axes of the fold's TRAINING embeddings (k = 8 is a rotation of the "
                "original and reproduces it). Axes of different seeds or folds are never matched to each other. Open diamond = FPCA "
                "benchmark at its own K; open square = natively trained 16-dimensional sensitivity model. Grey diagonal on the rank "
                "panel = every kept direction used equally.", "Effective dimension against stability")
    return fig


# ============================================================================== 5 · cohort probe
def plot_cohort_probe(cohort: pl.DataFrame, quality: pl.DataFrame):
    rows = [r for r in _rows(quality) if r["kind"] != "projection"]
    fig, ax = figure(1, 2)
    for a, protocol, title in ((ax[0, 0], "heldout_crossfit", "Fitted and scored on disjoint HELD-OUT athletes"),
                               (ax[0, 1], "train_to_heldout", "Fitted on training athletes, scored on held-out athletes")):
        d = {r["method"]: r for r in cohort.filter(pl.col("protocol") == protocol).iter_rows(named=True)}
        sub = [r for r in rows if r["method"] in d]
        if not sub:
            empty_panel(a, "no probe")
            continue
        pos = _xaxis(a, sub)
        for i, r in enumerate(sub):
            c = d[r["method"]]
            ci_errorbar(a, [pos[i]], [_num(c["balanced_accuracy"])], [_num(c["bal_lower"])], [_num(c["bal_upper"])], fmt="o", color=_color(r),
                        capsize=3, markersize=6)
            a.plot([pos[i]], [_num(c["baseline_balanced_accuracy"])], "_", color=INK["primary"], markersize=16, markeredgewidth=1.6)
            a.plot([pos[i]], [_num(c["accuracy"])], "x", color=INK["muted"], markersize=5)
        one = next(iter(d.values()))
        a.axhline(one["chance"], color=INK["muted"], linestyle="--", linewidth=1, label=f"chance = 1 / {one['n_classes']} cohorts")
        a.axhline(one["majority_share"], color=STATUS["warning"], linestyle=":", linewidth=1, label=f"majority-cohort share {one['majority_share']:.2f}")
        a.plot([], [], "_", color=INK["primary"], markersize=14, markeredgewidth=1.6, label="balanced accuracy, position only")
        a.plot([], [], "x", color=INK["muted"], label="plain accuracy (position + embedding)")
        a.set_ylabel("held-out balanced accuracy (dot)")
        a.set_ylim(0, 1.02)
        a.set_title(f"{title}\ncohort sizes — {one['class_counts']}", fontsize=9)
        a.legend(frameon=False, fontsize=7, loc="upper right")
    n = cohort["n_athletes"].max() if cohort.height else 0
    finish(fig, f"Unit: athlete (n up to {n}; athlete = mean of his held-out attempt embeddings). Ridge probe of the draft cohort. Dot = "
                "balanced accuracy (mean recall over cohorts) of position + embedding, with a 95 % athlete bootstrap; bar = the same "
                "probe with position alone. A dot well above its bar means the embedding carries the recording year (the 2025 cohort "
                "has about a quarter of the positional noise). Plain accuracy is shown only to make the effect of class imbalance "
                "visible: it must be read against the dotted majority line, balanced accuracy against the dashed chance line.",
           "Draft-cohort probe against its baselines")
    return fig


# ============================================================================== 6 · acquisition-window sensitivity
def plot_acquisition(trunc: pl.DataFrame, probes: pl.DataFrame, quality: pl.DataFrame,
                     targets: Sequence[tuple[str, str]] = (("first_speed", "first recorded\nspeed"), ("duration_s", "recording\nduration"),
                                                           ("pos_resid_rms", "tracking noise\n(position residual)"),
                                                           ("a_tan_roughness", "signal\nroughness"), ("observed_fraction", "share of the\nsupport observed"))):
    rows = [r for r in _rows(quality) if r["kind"] != "projection"]
    fig, ax = figure(1, 3)
    a, b, c = ax[0]
    views = sorted(trunc["view"].unique().to_list(), key=lambda v: (v.split("_")[0], int(v.split("_")[1]))) if trunc.height else []
    for r in rows:
        d = {t["view"]: t for t in trunc.filter(pl.col("method") == r["method"]).iter_rows(named=True)}
        if not d:
            continue
        x = np.arange(len(views), dtype=float)
        a.plot(x, [_num(d.get(v, {}).get("relative_shift")) for v in views], marker="o", markersize=4, linewidth=1.2, color=_color(r), alpha=0.9)
        b.plot(x, [_num(d.get(v, {}).get("auc")) - _num(d.get(v, {}).get("auc_full")) for v in views], marker="o", markersize=4,
               linewidth=1.2, color=_color(r), alpha=0.9)
    for axis, label, title in ((a, "embedding shift / median distance between athletes", "How far truncation moves an attempt"),
                               (b, "AUC truncated query − AUC full query", "What truncation costs in discrimination")):
        axis.set_xticks(np.arange(len(views)))
        axis.set_xticklabels([f"{v.split('_')[0]} −{v.split('_')[1]} %" for v in views], fontsize=8)
        axis.set_xlabel("recorded points hidden (head = early frames, tail = late frames)")
        axis.set_ylabel(label)
        axis.set_title(title, fontsize=10)
    b.axhline(0.0, color=INK["muted"], linestyle="--", linewidth=1)
    a.set_ylim(bottom=0)
    _kind_legend(a, [r["kind"] for r in rows], loc="upper left")
    att = probes.filter((pl.col("level") == "attempt") & (pl.col("metric") == "r2"))
    width = 0.8 / max(len(rows), 1)
    for i, r in enumerate(rows):
        d = {t["target"]: t for t in att.filter(pl.col("method") == r["method"]).iter_rows(named=True)}
        x = np.arange(len(targets)) - 0.4 + width * (i + 0.5)
        y = np.array([_num(d.get(t, {}).get("gain")) for t, _ in targets])
        ci_errorbar(c, x, y, [_num(d.get(t, {}).get("gain_lower")) for t, _ in targets], [_num(d.get(t, {}).get("gain_upper")) for t, _ in targets],
                    fmt="o", color=_color(r), capsize=1.5, markersize=4)
    c.axhline(0.0, color=INK["muted"], linestyle="--", linewidth=1)
    c.set_xticks(np.arange(len(targets)))
    c.set_xticklabels([t[1] for t in targets], fontsize=8)
    c.set_ylabel("held-out partial R² beyond position")
    c.set_title("Recording variables readable from ONE attempt's embedding", fontsize=10)
    n = trunc["n_athletes"].max() if trunc.height else 0
    finish(fig, f"Unit: held-out athlete (n up to {n}). Left and middle: the same attempt embedded with its first or last recorded points "
                "hidden — recorded points are only hidden, never invented, and a truncated view covering less than half of the support "
                "is not embedded. Middle: query = truncated first attempt, gallery = full second attempts of the position. Right: "
                "cross-fitted ridge probes from one attempt's embedding to a property of its recording; 0 = the embedding does not "
                "carry it. Whisker = 95 % athlete bootstrap.", "Acquisition-window and tracking-quality sensitivity")
    return fig


# ============================================================================== 7 · same athlete against matched impostor
def plot_matched_distances(matched: pl.DataFrame, methods: Sequence[tuple[str, str, str]]):
    """``methods``: (method, label, kind). One point per athlete: his own repeat distance against his matched impostors'."""
    fig, ax = figure(1, max(len(methods), 1))
    for a, (method, label, kind) in zip(ax[0], methods):
        d = matched.filter((pl.col("method") == method) & (pl.col("n_matched") > 0))
        if not d.height:
            empty_panel(a, f"{label}: no matched athlete")
            continue
        x, y = d["d_same"].to_numpy().astype(float), d["d_matched_median"].to_numpy().astype(float)
        conf = (d["design_role"] == "confirm").to_numpy()
        col = KIND_COLOR.get(kind, INK["axis"])
        a.plot(x[~conf], y[~conf], "o", color=col, markersize=3, alpha=0.35, markerfacecolor="none", label="design-role athlete")
        a.plot(x[conf], y[conf], "o", color=col, markersize=3, alpha=0.8, label="confirm-role athlete")
        lo, hi = float(np.nanmin([x.min(), y.min()])), float(np.nanmax([x.max(), y.max()]))
        lo = max(lo, 1e-3)
        a.plot([lo, hi], [lo, hi], color=INK["muted"], linestyle="--", linewidth=1)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("distance between the athlete's own two attempts")
        a.set_ylabel("median distance to his matched impostors")
        a.set_title(f"{label}\n{d.height} athletes · {100 * (y > x).mean():.0f} % above the line", fontsize=9)
    ax[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    finish(fig, "Unit: held-out athlete with at least one matched impostor (same position and fold, nearest official forty, 10-yard split, "
                "weight and height). x = distance between his two attempts; y = median distance from his attempt to the other attempt "
                "of his impostors; standardised embedding of his fold (distances are never compared between folds; log axes). Above "
                "the dashed line the athlete resembles himself more than his look-alikes. Every athlete is shown, not a summary.",
           "Same athlete against matched-impostor distances")
    return fig


# ============================================================================== 8 · reliability against redundancy
def plot_reliability_redundancy(quality: pl.DataFrame):
    rows = [r for r in _rows(quality) if r["kind"] != "projection"]
    fig, ax = figure(1, 2)
    for a, key, title in ((ax[0, 0], "auc", "Gallery = every athlete of the position"),
                          (ax[0, 1], "matched_auc", "Gallery = matched on forty, 10-yard, weight, height")):
        for r in rows:
            x, y = _num(r.get("forty_partial_r2")), _num(r.get(key))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ci_errorbar(a, [x], [y], [_num(r.get(f"{key}_lower"))], [_num(r.get(f"{key}_upper"))], fmt="o", color=_color(r), capsize=3, markersize=7)
            a.annotate(r["method"], (x, y), textcoords="offset points", xytext=(6, 5), fontsize=7, color=INK["secondary"])
        a.axhline(0.5, color=INK["muted"], linestyle="--", linewidth=1)
        a.set_xlabel("official forty explained by the representation beyond position (held-out partial R²)")
        a.set_ylabel("same-athlete discrimination AUC")
        a.set_xlim(-0.05, 1.02)
        a.set_ylim(0.3, 1.02)
        a.set_title(title, fontsize=10)
    _kind_legend(ax[0, 0], [r["kind"] for r in rows], loc="lower left")
    n = max((r.get("n_confirm") or 0 for r in rows), default=0)
    finish(fig, f"Unit: confirm-role held-out athlete (n up to {n}). x is the same on both panels: how much of the official forty the "
                "representation predicts. Left: repeatability against every athlete of the position (95 % athlete bootstrap). Right: "
                "against look-alike impostors (95 %, athletes resampled as queries and impostors). A representation can sit far "
                "right (it contains the stopwatch) and still be high on the right panel: redundancy and extra repeatable structure "
                "are not exclusive. Neither panel measures NFL value.", "Repeatability against stopwatch redundancy, with uncertainty")
    return fig


# ============================================================================== 9 · matched-stopwatch examples
def plot_matched_examples(examples: Sequence[dict[str, Any]], curves, order: dict[str, int], name: str, label: str):
    """``examples``: ``robust.matched_examples`` · ``curves``: the ``CurveSet`` · ``order``: seq_id → attempt order."""
    spec = SPEC[name]
    shown = list(examples)[:4]
    fig, ax = figure(1, max(len(shown), 1))

    def attempts(nfl_id: int) -> list[int]:
        idx = np.flatnonzero(curves.nfl_ids == nfl_id)
        return sorted(idx.tolist(), key=lambda i: order.get(str(curves.seq_ids[i]), 0))[:2]
    for a, ex in zip(ax[0], shown):
        for who, col, tag, forty in ((ex["nfl_id"], CATEGORICAL[0], "query athlete", ex["forty"]),
                                     (ex["impostor_id"], CATEGORICAL[1], "nearest matched impostor", ex["impostor_forty"])):
            for j, i in enumerate(attempts(who)):
                a.plot(curves.grid, curves.Y[i], color=col, linewidth=1.3, linestyle="-" if j == 0 else "--",
                       label=f"{tag} · forty {_num(forty):.2f} s" if j == 0 else None)
        a.text(0.97, 0.05, f"{label}\nown repeat distance {ex['d_same']:.2f}\ndistance to the impostor {ex['d_embedding']:.2f}\n"
                           f"matching distance {ex['match_distance']:.2f} SD\nmatched AUC of the query {ex['auc_matched']:.2f}",
               transform=a.transAxes, ha="right", va="bottom", fontsize=8, color=INK["secondary"])
        a.set_title(f"{ex['position_group']}", fontsize=10)
        a.set_xlabel(spec.axis_label)
        a.set_ylabel(spec.value_label)
        a.legend(frameon=False, fontsize=7, loc="upper left")
    if not shown:
        empty_panel(ax[0, 0], "no matched pair in this run")
    finish(fig, "Unit: one query athlete and his nearest matched impostor per panel (same position and fold; solid = first attempt, dashed "
                "= second). Queries are those at evenly spaced quantiles of the matched discrimination, from the lowest (left) to the "
                "highest (right) — chosen by rule, not by eye. Distances are in the standardised embedding of their fold. Descriptive "
                "examples: they illustrate what 'the same stopwatch' looks like, they do not test anything.",
           f"Matched-stopwatch athlete comparisons — {spec.label}")
    return fig


# ============================================================================== 10 · final summary
def plot_quality_summary(quality: pl.DataFrame, shortlist: pl.DataFrame | None = None):
    rows = _rows(quality)
    panels = (("auc", "repeat-attempt AUC", 0.5, True), ("matched_auc", "matched-stopwatch AUC", 0.5, True),
              ("resid_auc", "AUC after cross-fitted\nremoval of official results", 0.5, True),
              ("forty_partial_r2", "official forty explained\nbeyond position", None, False),
              ("window_r2", "attempt change explained\nby the window (lower = better)", None, True),
              ("cohort_balanced_gain", "cohort balanced-accuracy\ngain (lower = better)", 0.0, True),
              ("truncation_shift_max", "largest shift under\ntruncation (lower = better)", None, False))
    fig, ax = figure(1, len(panels), size=(12.8, max(7.2, 0.3 * len(rows) + 2.6)))
    pos = np.arange(len(rows), dtype=float)
    for j, (key, title, line, has_ci) in enumerate(panels):
        a = ax[0, j]
        for i, r in enumerate(rows):
            lo, hi = (_num(r.get(f"{key}_lower")), _num(r.get(f"{key}_upper"))) if has_ci else (np.nan, np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                a.plot([lo, hi], [pos[i], pos[i]], color=_color(r), linewidth=1.6)
            if np.isfinite(_num(r.get(key))):
                a.plot([_num(r.get(key))], [pos[i]], "o", color=_color(r), markersize=5)
            if i and r["kind"] != rows[i - 1]["kind"]:
                a.axhline(pos[i] - 0.5, color=INK["grid"], linewidth=1)
        if line is not None:
            a.axvline(line, color=INK["muted"], linestyle="--", linewidth=1)
        a.set_ylim(len(rows) - 0.5, -0.5)
        a.set_yticks(pos)
        a.set_yticklabels([r["method"] + (" *" if r.get("collapsed") else "") for r in rows] if j == 0 else [], fontsize=7)
        a.set_title(title, fontsize=8)
    if rows:
        _kind_legend(ax[0, -1], [r["kind"] for r in rows], loc="lower right")
    chosen = "" if shortlist is None else " Frozen shortlist: " + ", ".join(shortlist.filter(pl.col("selected"))["candidate_id"].to_list()) + "."
    n = max((r.get("n_confirm") or 0 for r in rows), default=0)
    finish(fig, f"Unit: confirm-role held-out athlete for the AUC panels (n up to {n}); all held-out athletes elsewhere. Dot = estimate, bar "
                "= 95 % athlete interval where one is computed (matched AUC: queries and impostors resampled). * = flagged collapsed "
                "(effective rank share below the heuristic line). The third panel is the cross-fitted residual of notebook 03 — read "
                "it with the audit of section 5, not alone." + chosen + " No panel measures NFL value.",
           "Representation quality: repeatability, matched discrimination and nuisance dependence")
    return fig
