"""Figures of notebook 03 (self-supervised encoder against the FPCA benchmark). Every function returns a Matplotlib
figure; the notebook passes it to ``AdvStore.show`` (inline display, PNG 300 DPI + SVG, PDF for publication figures).

Conventions as in ``fpca_figures``: project 16:9 style, units on every axis, the unit of analysis and the sample
size in the caption, 95 % athlete-bootstrap intervals. Colour identifies the METHOD FAMILY everywhere: FPCA grey,
TCN-A blue, TCN-B orange. Methods are always drawn in the same order (input A, B, C; FPCA first).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, STATUS
from ..figkit import ci_errorbar, empty_panel, figure, finish
from . import repr_eval as rv
from .fpca import SPEC
from .tcn import CONSISTENCY, RECON, CurveNormalizer, TcnConfig

OBJECTIVE_COLOR = {"fpca": INK["secondary"], RECON: CATEGORICAL[0], CONSISTENCY: CATEGORICAL[1]}
OBJECTIVE_NAME = {"fpca": "FPCA", RECON: "TCN-A (masked reconstruction)", CONSISTENCY: "TCN-B (+ attempt consistency)"}


def _order(methods: pl.DataFrame, primary_only: bool = True) -> list[dict[str, Any]]:
    m = methods.filter(pl.col("primary")) if primary_only else methods
    rank = {"fpca": 0, RECON: 1, CONSISTENCY: 2}
    return sorted(m.iter_rows(named=True), key=lambda r: (r["representation"], rank.get(r["objective"], 9), r["method"]))


def _num(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _col(df: pl.DataFrame, names: Sequence[str], key: str, col: str) -> np.ndarray:
    got = dict(zip(df[key].to_list(), df[col].to_list()))
    return np.array([_num(got.get(n)) for n in names])


def _legend(ax, objectives: Sequence[str], **kw) -> None:
    from matplotlib.patches import Patch

    seen = [o for o in ("fpca", RECON, CONSISTENCY) if o in objectives]
    ax.legend(handles=[Patch(color=OBJECTIVE_COLOR[o], label=OBJECTIVE_NAME[o]) for o in seen], frameon=False, fontsize=8, **kw)


def _method_axis(ax, rows: Sequence[dict[str, Any]], horizontal: bool = False) -> np.ndarray:
    pos = np.arange(len(rows), dtype=float)
    labels = [r["label"] for r in rows]
    if horizontal:
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    return pos


def _dots(ax, rows, est, lo, hi, horizontal: bool = False) -> None:
    pos = np.arange(len(rows), dtype=float)
    for i, r in enumerate(rows):
        c = OBJECTIVE_COLOR.get(r["objective"], INK["muted"])
        if horizontal:
            if np.isfinite(lo[i]) and np.isfinite(hi[i]):
                ax.plot([lo[i], hi[i]], [pos[i], pos[i]], color=c, linewidth=1.6)
            ax.plot([est[i]], [pos[i]], "o", color=c, markersize=6)
        else:
            ci_errorbar(ax, [pos[i]], [est[i]], [lo[i]], [hi[i]], fmt="o", color=c, capsize=3, markersize=6)


# ============================================================================== 1 · architecture and input
def plot_architecture(tcfg: TcnConfig, norm: CurveNormalizer, example: dict[str, np.ndarray], n_parameters: int, name: str):
    """``example``: grid, y (recorded curve, NaN elsewhere), hidden (bool) — one real held-out attempt."""
    from matplotlib.patches import FancyBboxPatch

    spec = SPEC[name]
    fig, ax = figure(1, 2)
    a, g, y, hidden = ax[0, 0], example["grid"], example["y"], example["hidden"]
    obs = np.isfinite(y) & norm.support
    a.axvspan(norm.grid[norm.support][0], norm.grid[norm.support][-1], color=INK["grid"], alpha=0.35, lw=0, label="support of the training fold")
    a.plot(g[norm.support], norm.mean[norm.support], color=INK["muted"], linestyle="--", linewidth=1, label="training mean function")
    a.plot(g[obs & ~hidden], y[obs & ~hidden], "o", color=CATEGORICAL[0], markersize=3, label="visible to the encoder")
    a.plot(g[obs & hidden], y[obs & hidden], "o", color=STATUS["critical"], markersize=3, label="hidden on purpose = reconstruction targets")
    a.set_xlabel(spec.axis_label)
    a.set_ylabel(spec.value_label)
    a.set_title("Input: one recorded attempt and its masks", fontsize=11)
    a.legend(frameon=False, fontsize=8)
    b = ax[0, 1]
    b.set_axis_off()
    boxes = [("3 channels per grid point", "value (0 where not visible) · visible flag · grid position", INK["grid"]),
             *[(f"residual block, dilation {d}", f"2 × convolution (kernel {tcfg.kernel_size}, {tcfg.hidden} channels), GELU, dropout {tcfg.dropout:g}",
                CATEGORICAL[0]) for d in tcfg.dilations],
             ("pooling over VISIBLE points", "mean and max; padding, unrecorded and hidden points never enter", CATEGORICAL[2]),
             (f"embedding z · {tcfg.embedding_dim} numbers", "the only thing the decoder and every evaluation see", CATEGORICAL[1]),
             ("decoder", f"z → {tcfg.decoder_hidden} → the curve on {norm.grid.size} grid points; loss on hidden points only", INK["grid"])]
    h = 0.92 / len(boxes)
    for i, (title, text, color) in enumerate(boxes):
        y0 = 0.96 - (i + 1) * h
        b.add_patch(FancyBboxPatch((0.04, y0 + 0.012), 0.92, h - 0.03, boxstyle="round,pad=0.008", transform=b.transAxes,
                                   facecolor=color, alpha=0.22, edgecolor=INK["axis"]))
        b.text(0.07, y0 + h * 0.62, title, transform=b.transAxes, fontsize=10, fontweight="bold", color=INK["primary"], va="center")
        b.text(0.07, y0 + h * 0.30, text, transform=b.transAxes, fontsize=8, color=INK["secondary"], va="center")
        if i:
            b.annotate("", xy=(0.5, y0 + h - 0.016), xytext=(0.5, y0 + h + 0.014), xycoords=b.transAxes, textcoords=b.transAxes,
                       arrowprops={"arrowstyle": "->", "color": INK["secondary"]})
    b.set_title(f"Encoder: {n_parameters:,} trainable weights · receptive field {tcfg.receptive_field} grid points", fontsize=11)
    finish(fig, f"Left: one real attempt of a held-out athlete, {spec.label}. Values are never filled: a hidden or unrecorded point "
                "enters the encoder as 0 with the visible flag at 0. Right: the model. TCN-B adds a consistency term between the "
                "embeddings of two attempts of the same TRAINING athlete, encoded on the grid points both recordings hold.",
           "Input representation and encoder")
    return fig


# ============================================================================== 2 · masked reconstruction examples
def plot_masked_examples(examples: Sequence[dict[str, Any]], name: str):
    """``examples``: title, grid, y, hidden, recon = {label: (objective, curve on the grid)} — chosen by rule."""
    spec = SPEC[name]
    fig, ax = figure(2, 2)
    for a, ex in zip(ax.ravel(), list(examples) + [None] * 4):
        if ex is None:
            empty_panel(a, "no example")
            continue
        g, y, hidden = ex["grid"], ex["y"], ex["hidden"]
        obs = np.isfinite(y)
        a.plot(g[obs & ~hidden], y[obs & ~hidden], "o", color=INK["primary"], markersize=2.5, label="visible")
        a.plot(g[hidden], y[hidden], "o", markerfacecolor="none", color=STATUS["critical"], markersize=4, label="hidden (target)")
        for label, (objective, curve) in ex["recon"].items():
            a.plot(g, np.where(obs, curve, np.nan), color=OBJECTIVE_COLOR.get(objective, INK["muted"]), linewidth=1.3, label=label)
        a.set_title(ex["title"], fontsize=10)
        a.set_xlabel(spec.axis_label)
        a.set_ylabel(spec.value_label)
        a.legend(frameon=False, fontsize=7)
    finish(fig, "Unit: one held-out attempt per panel (its athlete was in no fit). Each model reconstructs the hidden points from the "
                "visible ones through its bottleneck: K scores for FPCA, the embedding for the encoders. Attempts are those at the "
                "10th, 50th and 90th percentile and at the maximum of the TCN-A hidden-point error — chosen by rule, not by eye.",
           f"Masked reconstruction on held-out athletes — {spec.label}")
    return fig


# ============================================================================== 3 · learning curves
def plot_learning_curves(history: pl.DataFrame, methods: pl.DataFrame):
    rows = [r for r in _order(methods) if r["family"] == "tcn"]
    reps = sorted({r["representation"] for r in rows})
    fig, ax = figure(2, max(len(reps), 1))
    for i, objective in enumerate((RECON, CONSISTENCY)):
        for j, rep in enumerate(reps):
            a = ax[i, j]
            r = next((x for x in rows if x["objective"] == objective and x["representation"] == rep), None)
            h = history.filter(pl.col("method") == r["method"]) if r else history.clear()
            if not h.height:
                empty_panel(a, "not trained")
                continue
            for k, ((_, fold), d) in enumerate(sorted(h.partition_by(["repeat", "outer_fold"], as_dict=True).items())):
                d = d.sort("epoch")
                e, best = d["epoch"].to_numpy(), d.filter(pl.col("is_best"))
                a.plot(e, d["train_reconstruction"].to_numpy(), color=INK["axis"], linewidth=0.9,
                       label="training athletes (random masks)" if not k else None)
                a.plot(e, d["dev_reconstruction"].to_numpy(), color=OBJECTIVE_COLOR[objective], linewidth=1.1,
                       label="development athletes (fixed mask)" if not k else None)
                a.plot(best["epoch"].to_numpy(), best["dev_reconstruction"].to_numpy(), "v", color=INK["primary"], markersize=5,
                       label="kept weights" if not k else None)
            a.set_yscale("log")
            a.set_title(f"{r['label']} · {h.select(pl.struct('repeat', 'outer_fold').n_unique()).item()} fold models", fontsize=10)
            a.set_xlabel("epoch")
            a.set_ylabel("hidden-point MSE (normalised units)")
            a.legend(frameon=False, fontsize=7)
    finish(fig, "Unit: fold model (one line per outer fold). Development athletes are an inner fold of the training athletes; the "
                "held-out athletes appear nowhere on this figure. Triangle = epoch whose weights were kept (lowest development "
                "objective). Training error uses fresh random masks and input noise, so it is not comparable in level with the "
                "fixed-mask development error.", "Training and development learning curves")
    return fig


# ============================================================================== 4 · reconstruction
def plot_reconstruction(recon: pl.DataFrame, methods: pl.DataFrame):
    rows = _order(methods)
    reps = sorted({r["representation"] for r in rows})
    fig, ax = figure(1, max(len(reps), 1))
    for j, rep in enumerate(reps):
        a, sub = ax[0, j], [r for r in rows if r["representation"] == rep]
        names = [r["method"] for r in sub]
        pos = _method_axis(a, sub)
        for off, key, marker, label in ((-0.12, "rmse", "o", "all recorded points visible"), (0.12, "rmse_masked", "s", "hidden points, predicted")):
            est, lo, hi = (_col(recon, names, "method", f"{key}{s}") for s in ("", "_lower", "_upper"))
            for i, r in enumerate(sub):
                ci_errorbar(a, [pos[i] + off], [est[i]], [lo[i]], [hi[i]], fmt=marker, color=OBJECTIVE_COLOR[r["objective"]], capsize=3,
                            markersize=6, markerfacecolor=OBJECTIVE_COLOR[r["objective"]] if marker == "o" else "none")
            a.plot([], [], marker, color=INK["primary"], markerfacecolor=INK["primary"] if marker == "o" else "none", label=label)
        a.set_ylabel(f"held-out RMSE ({SPEC[rep].value_unit})")
        a.set_title(SPEC[rep].label, fontsize=10)
        a.set_ylim(bottom=0)
        a.legend(frameon=False, fontsize=8)
    n = recon["n_athletes"].max() if recon.height else 0
    finish(fig, f"Unit: athlete (up to n = {n} held out once each). Errors are counted on the grid points shared by the fold's FPCA "
                "fit and the fold's encoder, on the same attempts and with the same hidden points for every method. Whisker = 95 % "
                "athlete bootstrap. A lower error is not by itself evidence of athletic information.",
           "Held-out reconstruction: FPCA against the encoders")
    return fig


# ============================================================================== 5 · same against different athletes
def plot_similarity(pairs: pl.DataFrame, methods: pl.DataFrame, space: str = "standardised", scope: str = "within_position"):
    rows = _order(methods)
    ncols = max(int(np.ceil(len(rows) / 3)), 1)
    fig, ax = figure(3, ncols, size="tall" if ncols <= 3 else "wide")
    bins = np.linspace(0.0, 2.5, 41)
    panels = list(ax.T.ravel())
    for a, r in zip(panels, rows + [None] * (len(panels) - len(rows))):
        if r is None:
            a.set_axis_off()
            continue
        d = pairs.filter((pl.col("method") == r["method"]) & (pl.col("space") == space) & (pl.col("scope") == scope)
                         & (pl.col("n_candidates") > 0))
        if not d.height:
            empty_panel(a, "no repeat pairs")
            continue
        ratio = (d["d_same"] / d["d_diff_median"]).to_numpy()
        ratio = ratio[np.isfinite(ratio)]
        a.hist(np.clip(ratio, 0, bins[-1]), bins=bins, color=OBJECTIVE_COLOR[r["objective"]], alpha=0.85)
        a.axvline(1.0, color=INK["primary"], linewidth=1, linestyle="--")
        a.set_title(f"{r['label']} · {ratio.size} athletes · {100 * (ratio < 1).mean():.0f} % below 1 · median {np.median(ratio):.2f}", fontsize=9)
        a.set_xlabel("own repeat distance / median distance to other athletes")
        a.set_ylabel("athletes")
    finish(fig, "Unit: held-out athlete with two embedded attempts. x = distance between the athlete's two attempts divided by the "
                "median distance from his attempt to the other attempt of the athletes of his position in the same fold (1 = his own "
                f"repeat is as far as a typical other athlete; values above 2.5 are drawn at 2.5). Space: {space}. Distances are "
                "never compared between folds.", "Same-athlete against different-athlete distances")
    return fig


# ============================================================================== 6 · retrieval
def plot_retrieval(repeat: pl.DataFrame, methods: pl.DataFrame, subset: str = "role:confirm", space: str = "standardised"):
    rows = _order(methods)
    names = [r["method"] for r in rows]
    d = repeat.filter((pl.col("scope") == "within_position") & (pl.col("subset") == subset) & (pl.col("space") == space))
    fig, ax = figure(1, 3)
    chance = _col(d, names, "method", "chance_top1")
    for j, (key, label) in enumerate((("auc", "discrimination AUC"), ("top1", "top-1 retrieval accuracy"), ("mrr", "mean reciprocal rank"))):
        a = ax[0, j]
        _method_axis(a, rows)
        _dots(a, rows, *(_col(d, names, "method", f"{key}{s}") for s in ("", "_lower", "_upper")))
        if key == "auc":
            a.axhline(0.5, color=INK["muted"], linestyle="--", linewidth=1, label="chance")
        elif key == "top1" and np.isfinite(chance).any():
            a.axhline(float(np.nanmean(chance)), color=INK["muted"], linestyle="--", linewidth=1, label="chance (1 / gallery)")
        a.set_ylabel(label)
        a.set_ylim(0, 1.02)
        if key == "mrr":
            _legend(a, [r["objective"] for r in rows], loc="lower right")
        elif a.get_legend_handles_labels()[0]:
            a.legend(frameon=False, fontsize=8, loc="lower right")
    n = d["n_athletes"].max() if d.height else 0
    finish(fig, f"Unit: athlete (n up to {n}; subset {subset}; confirm-role athletes took no part in any modelling choice). Query = one "
                "attempt, gallery = the other attempt of every athlete of the same position held out in the same fold. AUC = share of "
                "the gallery farther than the athlete's own other attempt. Whisker = 95 % athlete bootstrap.",
           "Athlete retrieval from a repeated attempt (held-out athletes)")
    return fig


# ============================================================================== 7 · collapse
def plot_collapse(collapse: pl.DataFrame, spectrum: pl.DataFrame, methods: pl.DataFrame, ecfg: rv.EvalConfig):
    rows = _order(methods, primary_only=False)
    fig, ax = figure(1, 3)
    a, b, c = ax[0]
    for r in rows:
        s = spectrum.filter(pl.col("method") == r["method"]).group_by("index").agg(pl.col("eigen_share").mean()).sort("index")
        if s.height:
            a.plot(s["index"], np.maximum(s["eigen_share"].to_numpy().astype(float), 1e-6), marker="o", markersize=3, linewidth=1,
                   color=OBJECTIVE_COLOR[r["objective"]], alpha=0.8)
    a.set_yscale("log")
    a.set_xlabel("covariance eigenvalue (largest first)")
    a.set_ylabel("share of the embedding variance")
    a.set_title("Covariance spectrum (mean over folds)", fontsize=10)
    _legend(a, [r["objective"] for r in rows])
    names = [r["method"] for r in rows]
    for axis, col, title, line in ((b, "rank_share", "effective rank / dimension", ecfg.collapse_rank_share),
                                   (c, "near_identical_share", "share of nearly identical attempt pairs", ecfg.collapse_identical_share)):
        pos = _method_axis(axis, rows)
        for i, n in enumerate(names):
            v = collapse.filter(pl.col("method") == n)[col].to_numpy().astype(float)
            axis.plot(np.full(v.size, pos[i]), v, "o", color=OBJECTIVE_COLOR[rows[i]["objective"]], markersize=4, alpha=0.8)
        axis.axhline(line, color=STATUS["critical"], linestyle="--", linewidth=1, label="collapse line (heuristic)")
        axis.set_title(title, fontsize=10)
        axis.set_ylim(bottom=0)
        axis.legend(frameon=False, fontsize=8)
    finish(fig, "Unit: fold model (one dot per outer fold), on the embeddings of its held-out attempts. Effective rank = exp(entropy "
                "of the normalised covariance spectrum); an FPCA with K components cannot exceed K. Nearly identical = closer than "
                f"{100 * ecfg.near_identical:g} % of the median pair distance. A model below the left line or above the right one is "
                "flagged as collapsed.", "Representation collapse diagnostics")
    return fig


# ============================================================================== 8 · reliability against redundancy
def plot_reliability_vs_redundancy(summary: pl.DataFrame, methods: pl.DataFrame):
    rows = _order(methods)
    names = [r["method"] for r in rows]
    x = _col(summary, names, "method", "forty_partial_r2")
    fig, ax = figure(1, 2)
    for a, key, title in ((ax[0, 0], "auc", "As produced"), (ax[0, 1], "resid_auc", "After removing official forty, 10-yard, weight, height")):
        est, lo, hi = (_col(summary, names, "method", f"{key}{s}") for s in ("", "_lower", "_upper"))
        for i, r in enumerate(rows):
            if not (np.isfinite(x[i]) and np.isfinite(est[i])):
                continue
            ci_errorbar(a, [x[i]], [est[i]], [lo[i]], [hi[i]], fmt="o", color=OBJECTIVE_COLOR[r["objective"]], capsize=3, markersize=7)
            a.annotate(r["method"], (x[i], est[i]), textcoords="offset points", xytext=(6, 5), fontsize=7, color=INK["secondary"])
        a.axhline(0.5, color=INK["muted"], linestyle="--", linewidth=1)
        a.set_xlabel("official forty explained by the embedding beyond position (held-out partial R²)")
        a.set_ylabel("repeat-attempt discrimination AUC")
        a.set_title(title, fontsize=10)
        a.set_xlim(-0.05, 1.02)
        a.set_ylim(0.3, 1.02)
    _legend(ax[0, 0], [r["objective"] for r in rows], loc="lower right")
    n = summary["n_confirm"].max() if summary.height else 0
    finish(fig, f"Unit: confirm-role held-out athlete (n up to {n}), within position. x is the same on both panels: how much of the "
                "official forty the embedding predicts. Left y: can the embedding recognise an athlete from his other attempt? Right "
                "y: can it still do so once the cross-fitted linear prediction from the official results is removed? The target "
                "region is high on the RIGHT panel; dashed line = chance. Whisker = 95 % athlete bootstrap.",
           "Repeatability against stopwatch redundancy")
    return fig


# ============================================================================== 9 · cohort and position probes
def plot_cohort_probe(probes: pl.DataFrame, methods: pl.DataFrame, protocol: str = "heldout_crossfit"):
    rows = _order(methods)
    names = [r["method"] for r in rows]
    fig, ax = figure(1, 2)
    for a, target, title, base in ((ax[0, 0], "draft_cohort", "Draft cohort (sensor generation)", "position only"),
                                   (ax[0, 1], "position_group", "Position group", "majority class")):
        d = probes.filter((pl.col("protocol") == protocol) & (pl.col("target") == target))
        pos = _method_axis(a, rows)
        _dots(a, rows, *(_col(d, names, "method", c) for c in ("value", "value_lower", "value_upper")))
        a.plot(pos, _col(d, names, "method", "baseline"), "_", color=INK["primary"], markersize=16, markeredgewidth=1.6, label=f"baseline: {base}")
        a.set_ylabel("held-out accuracy")
        a.set_ylim(0, 1.02)
        a.set_title(title, fontsize=10)
        a.legend(frameon=False, fontsize=8)
    n = probes.filter(pl.col("target") == "draft_cohort")["n_athletes"].max() if probes.height else 0
    finish(fig, f"Unit: athlete (n up to {n}), athlete = mean of his held-out attempt embeddings. Ridge probes fitted and scored on "
                f"disjoint athletes ({protocol}). Left: dot = position + embedding, bar = position alone; a dot well above its bar "
                "means the embedding carries the recording year. Right: dot = embedding alone. Whisker = 95 % athlete bootstrap.",
           "Can the embedding tell the draft cohort or the position?")
    return fig


# ============================================================================== 10 · recording window
def plot_window_dependence(window: pl.DataFrame, trunc: pl.DataFrame, methods: pl.DataFrame, reference_line: float):
    rows = _order(methods)
    names = [r["method"] for r in rows]
    fig, ax = figure(1, 2)
    a, b = ax[0]
    w = window.filter(pl.col("space") == "standardised")
    _method_axis(a, rows)
    _dots(a, rows, *(_col(w, names, "method", c) for c in ("r2_trace", "r2_lower", "r2_upper")))
    a.axhline(reference_line, color=INK["muted"], linestyle="--", linewidth=1, label=f"reference line of notebook 02 ({reference_line:g})")
    a.set_ylabel("share of the attempt-to-attempt change explained by the window")
    a.set_ylim(0, 1.02)
    a.set_title("Natural variation: two recordings of one athlete", fontsize=10)
    a.legend(frameon=False, fontsize=8)
    views = sorted(trunc["view"].unique().to_list(), key=lambda v: (v.split("_")[0], int(v.split("_")[1]))) if trunc.height else []
    for r in rows:
        y = _col(trunc.filter(pl.col("method") == r["method"]), views, "view", "relative_shift")
        b.plot(np.arange(len(views)), y, marker="o", markersize=4, linewidth=1.2, color=OBJECTIVE_COLOR[r["objective"]],
               linestyle={"A": "-", "B": "--", "C": ":"}.get(r["representation"][0], "-"), alpha=0.9)
    b.set_xticks(np.arange(len(views)))
    b.set_xticklabels([f"{v.split('_')[0]} −{v.split('_')[1]} %" for v in views])
    b.set_xlabel("recorded points hidden (head = start of the recording, tail = end)")
    b.set_ylabel("embedding shift / median distance between athletes")
    b.set_ylim(bottom=0)
    b.set_title("Controlled truncation of the same attempt (line style: input A solid, B dashed, C dotted)", fontsize=10)
    _legend(b, [r["objective"] for r in rows])
    n = window["n_athletes"].max() if window.height else 0
    finish(fig, f"Left, unit: athlete with two attempts (n up to {n}): the change of his embedding between attempts regressed on the "
                "change of first-frame speed, recorded duration and frames after the peak (trace R², in-sample as in notebook 02, "
                "whisker = 95 % athlete bootstrap). An athlete does not change between attempts; his recording does. Right: the same "
                "held-out attempt embedded with part of its recording hidden; a shift near 1 moves it as far as another athlete.",
           "Dependence on the recording window")
    return fig


# ============================================================================== 11 · pairs with the same stopwatch
def pair_examples(pairs: pl.DataFrame, emb: pl.DataFrame, method_names: Sequence[str]) -> list[dict[str, Any]]:
    """For rule-selected pairs of athletes with (nearly) the same official forty: the distance between their first
    attempts in each method's standardised embedding, in units of the fold's median distance between different
    athletes. A distance exists only when both athletes are held out in the same fold."""
    blocks = {b.method: {} for b in rv.emb_blocks(emb)}
    for b in rv.emb_blocks(emb):
        blocks[b.method][b.fold] = b
    out = []
    for p in pairs.iter_rows(named=True):
        dist = {}
        for m in method_names:
            for b in blocks.get(m, {}).values():
                held = {s: i for i, s in enumerate(b.seq_ids) if b.role[i] == "heldout"}
                if p["seq_id_a"] in held and p["seq_id_b"] in held:
                    h = np.flatnonzero(b.held)
                    D = rv.cdist(b.Zs[h], b.Zs[h])
                    scale = float(np.median(D[b.nfl_ids[h][:, None] != b.nfl_ids[h][None, :]]))
                    d = float(np.linalg.norm(b.Zs[held[p["seq_id_a"]]] - b.Zs[held[p["seq_id_b"]]]))
                    dist[m] = d / scale if scale > 0 else float("nan")
        out.append({"seq_id_a": p["seq_id_a"], "seq_id_b": p["seq_id_b"], "forty_a": p.get("forty_a"), "forty_b": p.get("forty_b"),
                    "distance": dist})
    return out


def plot_pair_examples(examples: Sequence[dict[str, Any]], curves, labels: dict[str, str], name: str):
    """``curves``: the ``CurveSet`` of the representation · ``labels``: method → label."""
    spec = SPEC[name]
    shown = list(examples)[:4]
    fig, ax = figure(1, max(len(shown), 1))
    for a, ex in zip(ax[0], shown):
        for k, key in enumerate(("seq_id_a", "seq_id_b")):
            y = curves.Y[curves.index_of(ex[key])]
            forty = ex.get("forty_a" if k == 0 else "forty_b")
            a.plot(curves.grid, y, color=CATEGORICAL[k], linewidth=1.4, label=f"athlete {'AB'[k]} · official forty {_num(forty):.2f} s")
        text = "\n".join(f"{labels.get(m, m)}: {d:.2f}" for m, d in ex["distance"].items()) or "not held out in the same fold"
        a.text(0.97, 0.05, "distance A–B / median between athletes\n" + text, transform=a.transAxes, ha="right", va="bottom", fontsize=8,
               color=INK["secondary"])
        a.set_xlabel(spec.axis_label)
        a.set_ylabel(spec.value_label)
        a.legend(frameon=False, fontsize=8, loc="upper left")
    if not shown:
        empty_panel(ax[0, 0], "no pair of athletes with the same official forty in this run")
    finish(fig, "Unit: pair of athletes of the same position whose official forty differs by at most 0.01 s (first attempts, comparable "
                "recording start; chosen by the rule of notebook 01). The text gives how far apart each method places the two runs, "
                "relative to the median distance between different athletes held out in the same fold. Descriptive examples: they "
                "illustrate, they do not test.", f"Same stopwatch, different run? — {spec.label}")
    return fig


# ============================================================================== 12 · final comparison
def plot_final_comparison(summary: pl.DataFrame, methods: pl.DataFrame, verdict: pl.DataFrame):
    rows = _order(methods)
    names = [r["method"] for r in rows]
    panels = (("auc", "repeat-attempt AUC\n(higher = more repeatable)", 0.5, True),
              ("resid_auc", "AUC after removing\nofficial results", 0.5, True),
              ("forty_partial_r2", "official forty explained\nbeyond position", None, False),
              ("window_r2", "attempt change explained\nby the window (lower = better)", None, True),
              ("cohort_gain", "cohort accuracy gain\nover position (lower = better)", 0.0, True),
              ("truncation_shift_max", "largest shift under\ntruncation (lower = better)", None, False))
    fig, ax = figure(1, len(panels))
    for j, (key, title, line, has_ci) in enumerate(panels):
        a = ax[0, j]
        pos = _method_axis(a, rows, horizontal=True)
        est = _col(summary, names, "method", key)
        lo, hi = ((_col(summary, names, "method", f"{key}_lower"), _col(summary, names, "method", f"{key}_upper")) if has_ci
                  else (np.full(len(rows), np.nan), np.full(len(rows), np.nan)))
        _dots(a, rows, est, lo, hi, horizontal=True)
        if line is not None:
            a.axvline(line, color=INK["muted"], linestyle="--", linewidth=1)
        a.set_title(title, fontsize=9)
        a.set_ylim(len(rows) - 0.5, -0.5)
        if j:
            a.set_yticklabels([])
        for i, r in enumerate(rows):
            if i and r["representation"] != rows[i - 1]["representation"]:
                a.axhline(pos[i] - 0.5, color=INK["grid"], linewidth=1)
    got = dict(zip(verdict["method"].to_list(), verdict["verdict"].to_list()))
    text = " · ".join(f"{n}: {got[n]}" for n in names if n in got) or "no verdict"
    n = summary["n_confirm"].max() if summary.height else 0
    finish(fig, f"Unit: confirm-role held-out athlete (n up to {n}) for the two AUC panels; all held-out athletes elsewhere. Dot = "
                "estimate, bar = 95 % athlete bootstrap where one is computed. Grey = FPCA benchmark of notebook 02, blue = TCN-A, "
                f"orange = TCN-B. Rule-based reading (for review): {text}.", "Evidence by method: repeatability, redundancy and nuisance dependence")
    return fig
