"""Figures of notebook 02 (functional PCA baseline). Every function returns a Matplotlib figure; the notebook
passes it to ``AdvStore.show`` (inline display, PNG 300 DPI + SVG, PDF for publication figures, plot data, sidecar).

Conventions as in ``visualizations``: project 16:9 style, colour-blind-safe palette, units on every axis, the unit
of analysis and the sample size in the caption. Colours identify the representation (A blue, B orange, C aqua) or,
inside one panel, the series named in its legend. Intervals are 95 % and athlete-level. A reconstruction outside
the recorded range of a curve is drawn dotted and named model-implied.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK
from ..figkit import ci_errorbar, empty_panel, figure, finish
from .fpca import SPEC, CurveSet, FpcaConfig, FpcaModel, athlete_weights
from .fpca_eval import FLAG_VARS, WINDOW_VARS, blocks, first_two, fold_tag

REP_COLOR = {"A": CATEGORICAL[0], "B": CATEGORICAL[1], "C": CATEGORICAL[2]}
VAR_LABEL = {"first_speed": "first-frame speed", "duration_s": "recorded duration", "frames_after_peak": "frames after peak",
             "path_yd": "recorded path", "last_speed_frac_of_peak": "last speed / peak", "pos_resid_rms": "position noise",
             "a_tan_roughness": "acceleration roughness", "share_boundary": "share of boundary frames",
             "first_frame_censored": "starts in motion", "start_above_ref_speed": "starts above ref. speed",
             "last_frame_censored": "ends at top speed", "ends_at_peak": "peak on last frame",
             "boundary_peak": "peak on boundary frame", "signal_flag": "signal flag", "observed_fraction": "support covered"}


def _color(k: int) -> str:
    return CATEGORICAL[k % len(CATEGORICAL)]


def _letter(name: str) -> str:
    return name[0] + ("′" if SPEC[name].alternate_of else "")


def _rep_color(name: str) -> str:
    return REP_COLOR.get(name[0], INK["secondary"])


def _num(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _main(names: Sequence[str]) -> list[str]:
    return [n for n in names if n in SPEC and SPEC[n].alternate_of is None]


# ============================================================================== curves and modes
def plot_mean_curves(curves: dict[str, CurveSet], position_of: dict[int, str], min_sequences: int = 10):
    names = _main(list(curves))
    fig, ax = figure(1, max(len(names), 1))
    n_ath = 0
    for j, name in enumerate(names):
        cs, a, spec = curves[name], ax[0, j], SPEC[name]
        pos = np.array([str(position_of.get(int(i), "?")) for i in cs.nfl_ids])
        n_ath = int(np.unique(cs.nfl_ids).size)
        for k, p in enumerate(sorted(set(pos.tolist()))):
            rows = pos == p
            Y, w = cs.Y[rows], athlete_weights(cs.nfl_ids[rows])
            O = np.isfinite(Y)
            wO = w[:, None] * O
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = (wO * np.where(O, Y, 0.0)).sum(0) / wO.sum(0)
            a.plot(cs.grid, np.where(O.sum(0) >= min_sequences, mean, np.nan), color=_color(k),
                   label=f"{p} ({int(np.unique(cs.nfl_ids[rows]).size)})")
        a.set_title(spec.label, fontsize=11)
        a.set_xlabel(spec.axis_label)
        a.set_ylabel(spec.value_label)
        a.legend(title="position (athletes)", frameon=False, fontsize=8)
    finish(fig, f"Unit: athlete (n = {n_ath}); each athlete weighs 1, shared by his attempts. A mean is drawn only where at "
                f"least {min_sequences} sequences of the group are recorded; nothing is extrapolated. Descriptive, all "
                "athletes (no model).", "Mean curves by position group")
    return fig


def plot_modes(model: FpcaModel, name: str, n: int = 4, scale: float = 2.0):
    """Top: mean ± ``scale`` standard deviations of each component. Bottom: the eigenfunction."""
    spec = SPEC[name]
    n = min(n, model.k_max)
    fig, ax = figure(2, n)
    g, ex = model.grid, model.explained()
    for c in range(n):
        d = scale * np.sqrt(model.eigenvalues[c]) * model.phi[:, c]
        top, bot = ax[0, c], ax[1, c]
        top.fill_between(g, model.mean - d, model.mean + d, color=INK["grid"], alpha=0.6, lw=0)
        top.plot(g, model.mean, color=INK["primary"], linewidth=1.5, label="mean")
        top.plot(g, model.mean + d, color=_color(0), label=f"+{scale:g} SD")
        top.plot(g, model.mean - d, color=_color(1), label=f"−{scale:g} SD")
        top.set_title(f"component {c + 1} · {100 * ex[c]:.1f} % of variance", fontsize=11)
        top.set_ylabel(spec.value_label)
        bot.plot(g, model.phi[:, c], color=INK["primary"])
        bot.axhline(0, color=INK["axis"], linewidth=0.8)
        bot.set_xlabel(spec.axis_label)
        bot.set_ylabel("eigenfunction")
        if c == 0:
            top.legend(frameon=False, fontsize=8)
    finish(fig, f"Reference fit on all {len(model.fit_athletes)} athletes ({model.n_fit_sequences} sequences) — TRANSDUCTIVE, "
                "shown to read the modes; every reported statistic uses fold fits that exclude the evaluated athletes. "
                f"Support: {g[0]:g} to {g[-1]:g} ({int(model.support.sum())} grid points observed for most athletes); "
                f"{model.n_basis} cubic B-splines.", f"Principal functional modes — {spec.label}")
    return fig


def plot_explained_variance(variance: pl.DataFrame, names: Sequence[str]):
    names = _main(names)
    fig, ax = figure(1, max(len(names), 1))
    for j, name in enumerate(names):
        v, a = variance.filter(pl.col("representation") == name).sort("component"), ax[0, j]
        if not v.height:
            empty_panel(a, "no fit")
            continue
        x = v["component"].to_numpy()
        est, lo, hi = (v[c].to_numpy().astype(float) for c in ("explained_mean", "explained_min", "explained_max"))
        a.bar(x, est, color=_rep_color(name), alpha=0.85, label="component")
        a.errorbar(x, est, yerr=np.vstack([est - lo, hi - est]), fmt="none", ecolor=INK["primary"], capsize=3)
        a.plot(x, v["cumulative_mean"].to_numpy(), color=INK["primary"], marker="o", markersize=4, label="cumulative")
        a.set_ylim(0, 1.05)
        a.set_xticks(x)
        a.set_xlabel("component")
        a.set_ylabel("share of the training curves' variance")
        a.set_title(f"{SPEC[name].label} ({int(v['n_fits'][0])} fold fits)", fontsize=11)
        a.legend(frameon=False, fontsize=8)
    finish(fig, "Unit: fold fit (training athletes of one outer fold). Bar = mean over fold fits, whisker = smallest and "
                "largest fold value (a range, not a confidence interval). Variance is integrated over the fold's support.",
           "Explained variance per component")
    return fig


# ============================================================================== reconstruction
def reconstruction_examples(recon: pl.DataFrame, representation: str, k: int, repeat: int = 0) -> list[tuple[str, str, str]]:
    """(seq_id, fold tag, note) of the held-out sequences at the 10th, 50th and 90th percentile of the held-out
    error and of the worst one — chosen by rule, not by eye."""
    d = recon.filter((pl.col("representation") == representation) & (pl.col("k") == k) & (pl.col("repeat") == repeat)
                     & pl.col("scored")).sort("rmse", "seq_id")
    out = []
    for q, note in ((0.1, "10th percentile of error"), (0.5, "median error"), (0.9, "90th percentile"), (1.0, "largest error")):
        if d.height:
            r = d.row(int(round(q * (d.height - 1))), named=True)
            out.append((r["seq_id"], fold_tag(repeat, r["outer_fold"]), f"{note} · RMSE {r['rmse']:.3f}"))
    return out


def plot_reconstructions(cs: CurveSet, models: dict[str, FpcaModel], examples: Sequence[tuple[str, str, str]],
                         ks: Sequence[int], names: dict[int, str] | None = None):
    fig, ax = figure(1, max(min(len(examples), 4), 1))
    if not examples:
        empty_panel(ax[0, 0], "no scored held-out sequence")
    for j, (seq_id, tag, note) in enumerate(list(examples)[:4]):
        m, i, a = models[tag], cs.index_of(seq_id), ax[0, j]
        y, g = cs.Y[i][m.support], m.grid
        obs = np.isfinite(y)
        a.plot(g[obs], y[obs], linestyle="none", marker="o", markersize=3, color=INK["secondary"], label="recorded")
        for q, k in enumerate(k for k in ks if k <= m.k_max):
            Sc, _ = m.scores(cs.Y[i:i + 1], k)
            if not np.isfinite(Sc).all():
                continue
            rec = m.reconstruct(Sc, k)[0]
            a.plot(g, np.where(obs, rec, np.nan), color=_color(q), label=f"K = {k}")
            a.plot(g, np.where(obs, np.nan, rec), color=_color(q), linestyle=":", linewidth=2.2,
                   label="model-implied, not recorded" if q == 0 else None)
        who = (names or {}).get(int(cs.nfl_ids[i]), int(cs.nfl_ids[i]))
        a.set_title(f"{who}\n{note}", fontsize=10)
        a.set_xlabel(cs.spec.axis_label)
        a.set_ylabel(cs.spec.value_label)
        a.legend(frameon=False, fontsize=8)
    finish(fig, "Unit: grid point of one HELD-OUT attempt; the model was fitted without this athlete (split repeat 0). Dots: "
                "recorded values. Solid: reconstruction over the recorded range. Dotted: what the model implies where "
                "nothing was recorded — not an observation, and excluded from every error. Examples chosen by rule on the "
                "held-out error.", f"Athlete-level reconstructions — {cs.spec.label}")
    return fig


def plot_heldout_error(recon: pl.DataFrame, names: Sequence[str]):
    names = _main(names)
    fig, ax = figure(1, 2)
    r0 = recon.filter(pl.col("repeat") == 0)
    n = 0
    for name in names:
        d = r0.filter(pl.col("representation") == name).sort("k")
        if not d.height:
            continue
        k = d["k"].to_numpy()
        n = max(n, int(d["n_athletes"].max() or 0))
        col, off = _rep_color(name), 0.08 * (names.index(name) - 1)
        f = lambda c: d[c].cast(pl.Float64).fill_null(float("nan")).to_numpy()     # noqa: E731
        ci_errorbar(ax[0, 0], k + off, f("fve_heldout"), f("fve_lower"), f("fve_upper"), color=col,
                    label=f"{_letter(name)} held-out")
        ax[0, 0].plot(k + off, f("fve_train"), color=col, linestyle="--", linewidth=1.2, label=f"{_letter(name)} training")
        ci_errorbar(ax[0, 1], k + off, f("rmse_heldout"), f("rmse_lower"), f("rmse_upper"), color=col,
                    label=f"{_letter(name)} ({SPEC[name].value_unit})")
        ax[0, 1].plot(k + off, f("rmse_heldout"), color=col, linewidth=1.0)
    for a in ax[0]:
        a.set_xlabel("number of components K")
        a.legend(frameon=False, fontsize=8)
    ax[0, 0].set_ylabel("share of variance reconstructed (1 = perfect)")
    ax[0, 1].set_ylabel("RMSE on recorded points (unit of the curve)")
    ax[0, 1].set_ylim(0, None)
    finish(fig, f"Unit: athlete held out of the fit (up to n = {n}; split repeat 0). Points: held-out value with a 95 % athlete "
                "bootstrap interval; dashed: the same quantity on the training athletes. Errors use recorded grid points "
                "only. The two panels are not comparable across representations with different units (B is an "
                "acceleration).", "Held-out reconstruction error")
    return fig


# ============================================================================== reliability
def plot_attempt_scores(scores: pl.DataFrame, windows: pl.DataFrame, gate: pl.DataFrame, k_by_rep: dict[str, int],
                        n_comp: int = 3):
    names = _main(list(k_by_rep))
    fig, ax = figure(max(len(names), 1), n_comp, size=(12.8, 3.4 * max(len(names), 1) + 0.8))
    order_of = dict(zip(windows["seq_id"].to_list(), windows["attempt_order"].to_list()))
    n_pairs = 0
    for r, name in enumerate(names):
        sub = scores.filter((pl.col("representation") == name) & (pl.col("k") == k_by_rep[name]) & (pl.col("repeat") == 0))
        b = next(blocks(sub), None) if sub.height else None
        for c in range(n_comp):
            a = ax[r, c]
            if b is None or c >= b.k:
                empty_panel(a, "—")
                continue
            _, i1, i2 = first_two(b.nfl_ids, np.array([order_of.get(s, 0) or 0 for s in b.seq_ids]))
            x, y = b.Z[i1, c], b.Z[i2, c]
            n_pairs = max(n_pairs, int(x.size))
            a.scatter(x, y, s=10, alpha=0.5, color=_rep_color(name), linewidths=0)
            lim = [float(np.nanmin(np.r_[x, y, -1])), float(np.nanmax(np.r_[x, y, 1]))]
            a.plot(lim, lim, color=INK["axis"], linewidth=1.0)
            g = gate.filter((pl.col("representation") == name) & (pl.col("component") == c + 1))
            icc = g.row(0, named=True) if g.height else {}
            a.set_title(f"{_letter(name)}{c + 1} · within-position ICC {_num(icc.get('icc_adj')):.2f} "
                        f"[{_num(icc.get('icc_lower')):.2f}, {_num(icc.get('icc_upper')):.2f}]", fontsize=10)
            a.set_xlabel("first attempt (score, training SD units)")
            a.set_ylabel("second attempt")
    finish(fig, f"Unit: athlete with two scored attempts (up to n = {n_pairs}); rows = representations at their benchmark K. "
                "Scores are out-of-fold: the model never saw the athlete. Points on the diagonal = identical scores on both "
                "attempts. The cloud pools all positions, so part of its spread along the diagonal is position; the ICC in "
                "each title removes position and fold means.", "First against second attempt")
    return fig


def plot_variance_components(reliability: pl.DataFrame, k_by_rep: dict[str, int]):
    names = _main(list(k_by_rep))
    fig, ax = figure(1, max(len(names), 1))
    n = 0
    for j, name in enumerate(names):
        a = ax[0, j]
        d = reliability.filter((pl.col("representation") == name) & (pl.col("k") == k_by_rep[name]) & (pl.col("repeat") == 0)
                               & (pl.col("scope") == "position_adjusted") & pl.col("var_between").is_not_null()).sort("component")
        if not d.height:
            empty_panel(a, "not estimable")
            continue
        n = max(n, int(d["n_athletes_repeat"].max()))
        x, vb, vw = d["component"].to_numpy(), d["var_between"].to_numpy(), d["var_within"].to_numpy()
        tot = np.where(vb + vw > 0, vb + vw, np.nan)
        a.bar(x, vb / tot, color=_rep_color(name), label="between athletes (same position)")
        a.bar(x, vw / tot, bottom=vb / tot, color=INK["grid"], label="within athlete (between attempts)")
        a.set_xticks(x)
        a.set_ylim(0, 1)
        a.set_xlabel("component")
        a.set_ylabel("share of the score's variance")
        a.set_title(f"{SPEC[name].label} · K = {k_by_rep[name]}", fontsize=11)
        a.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, f"Unit: athlete with repeated attempts (up to n = {n}); out-of-fold scores, split repeat 0. Variance "
                "components of a one-way model with athletes nested in position × fold. The coloured share is what belongs "
                "to the athlete; the grey share changes between two attempts of the same athlete.",
           "Within-athlete against between-athlete variation")
    return fig


def plot_reliability_vs_redundancy(gate: pl.DataFrame, fcfg: FpcaConfig, rescue: pl.DataFrame | None = None):
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    if rescue is not None and rescue.height:
        r = rescue.filter(pl.col("icc_adj").is_not_null() & pl.col("redundancy_abs").is_not_null())
        a.scatter(r["redundancy_abs"].to_numpy(), r["icc_adj"].to_numpy(), s=28, color=INK["axis"], marker="s",
                  label="handcrafted features (sprint rescue)")
        for row in r.iter_rows(named=True):
            a.annotate(row["feature"], (row["redundancy_abs"], row["icc_adj"]), fontsize=7, color=INK["muted"],
                       xytext=(3, 3), textcoords="offset points")
    g = gate.filter(pl.col("role") != "alternate")
    seen = set()
    for row in g.iter_rows(named=True):
        if row["icc_adj"] is None or row["redundancy_abs"] is None:
            continue
        x, y, name = row["redundancy_abs"], row["icc_adj"], row["representation"]
        xerr = [[max(x - _num(row["redundancy_lower"]), 0)], [max(_num(row["redundancy_upper"]) - x, 0)]]
        yerr = [[max(y - _num(row["icc_lower"]), 0)], [max(_num(row["icc_upper"]) - y, 0)]]
        a.errorbar(x, y, xerr=np.nan_to_num(xerr), yerr=np.nan_to_num(yerr), fmt="o", color=_rep_color(name), capsize=2,
                   markersize=7, label=None if name in seen else SPEC[name].label)
        seen.add(name)
        a.annotate(f"{_letter(name)}{row['component']}", (x, y), fontsize=9, fontweight="semibold", xytext=(5, 5),
                   textcoords="offset points")
    a.axhline(fcfg.icc_floor, color=INK["muted"], linestyle=":", linewidth=1.0)
    a.axvline(fcfg.redundancy_reference, color=INK["muted"], linestyle=":", linewidth=1.0)
    a.set_xlim(0, 1)
    a.set_ylim(min(0.0, a.get_ylim()[0]), 1)
    a.set_xlabel("stopwatch redundancy: larger |ρ| with the official forty or 10-yard split, within position")
    a.set_ylabel("repeat-attempt reliability: within-position ICC")
    if seen or rescue is not None:
        a.legend(frameon=False, fontsize=8, loc="lower left")
    n = int(g["n_athletes_repeat"].max() or 0) if g.height else 0
    finish(fig, f"Unit: athlete (up to n = {n} with a repeat). Each labelled point is one FPCA component at the benchmark K "
                "(letter = representation, number = component), with 95 % intervals. Upper left = repeatable and not "
                "explained by the stopwatch. Dotted lines are heuristic reference lines (ICC "
                f"{fcfg.icc_floor:g}, |ρ| {fcfg.redundancy_reference:g}), not acceptance thresholds. Grey squares: the "
                "handcrafted forty features of the executed sprint rescue, for scale.",
           "Reliability against stopwatch redundancy")
    return fig


# ============================================================================== nuisance dependence
def plot_window_heatmap(window: pl.DataFrame, k_by_rep: dict[str, int]):
    from matplotlib.colors import LinearSegmentedColormap

    names = _main(list(k_by_rep))
    variables = [*WINDOW_VARS, *FLAG_VARS, "observed_fraction"]
    rows = [(n, c) for n in names for c in range(1, k_by_rep[n] + 1)]
    fig, ax = figure(1, 2, size=(12.8, max(4.5, 0.42 * len(rows) + 3.2)))
    cmap = LinearSegmentedColormap.from_list("div", [CATEGORICAL[1], "#ffffff", CATEGORICAL[0]])
    n = 0
    for j, (level, title) in enumerate((("within_athlete", "within athlete: change between two attempts"),
                                        ("between_athletes", "between athletes, within position"))):
        a = ax[0, j]
        z = np.full((len(rows), len(variables)), np.nan)
        for i, (name, c) in enumerate(rows):
            d = window.filter((pl.col("representation") == name) & (pl.col("k") == k_by_rep[name]) & (pl.col("component") == c)
                              & (pl.col("level") == level))
            got = dict(zip(d["variable"].to_list(), d["spearman"].to_list()))
            n = max(n, int(d["n_athletes"].max() or 0)) if d.height else n
            z[i] = [_num(got.get(v)) for v in variables]
        if not rows:
            empty_panel(a, "no component")
            continue
        im = a.imshow(z, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(z.shape[0]):
            for k in range(z.shape[1]):
                if np.isfinite(z[i, k]):
                    a.text(k, i, f"{z[i, k]:.2f}", ha="center", va="center", fontsize=7, color=INK["primary"])
        a.set_xticks(range(len(variables)), [VAR_LABEL.get(v, v) for v in variables], rotation=60, ha="right", fontsize=8)
        a.set_yticks(range(len(rows)), [f"{_letter(nm)}{c}" for nm, c in rows], fontsize=8)
        a.set_title(title, fontsize=11)
        a.grid(False)
        if j == 1:
            fig.colorbar(im, ax=a, label="Spearman correlation with the score")
    finish(fig, f"Unit: athlete (up to n = {n}); out-of-fold scores at the benchmark K, split repeat 0. Left: Spearman "
                "correlation between the attempt-to-attempt change of a score and the attempt-to-attempt change of a "
                "recording variable — the athlete is the same, so a strong cell means the score follows the recording. "
                "Right: athlete means, position removed. Blank = not estimable.",
           "Do the components follow the recording window?")
    return fig


def plot_nuisance_shares(gate: pl.DataFrame, fcfg: FpcaConfig):
    g = gate.filter(pl.col("role") != "alternate").sort("representation", "component")
    fig, ax = figure(1, 2)
    if not g.height:
        empty_panel(ax[0, 0], "no component")
        empty_panel(ax[0, 1], "no component")
    else:
        x = np.arange(g.height)
        labels = [f"{_letter(n)}{c}" for n, c in zip(g["representation"].to_list(), g["component"].to_list())]
        colors = [_rep_color(n) for n in g["representation"].to_list()]
        f = lambda c: g[c].cast(pl.Float64).fill_null(float("nan")).to_numpy()     # noqa: E731
        ax[0, 0].bar(x, f("window_r2_within"), color=colors, alpha=0.85)
        ci_errorbar(ax[0, 0], x, f("window_r2_within"), f("window_r2_lower"), f("window_r2_upper"), fmt="none",
                    ecolor=INK["primary"])
        ax[0, 0].axhline(fcfg.window_reference, color=INK["muted"], linestyle=":", linewidth=1.0)
        ax[0, 0].set_ylabel("share of the attempt-to-attempt change explained by\nfirst-frame speed, duration and frames after peak")
        ax[0, 1].bar(x, f("cohort_r2"), color=colors, alpha=0.85)
        ax[0, 1].set_ylabel("share of the athlete-level score explained by\ndraft cohort, within position")
        for a in ax[0]:
            a.set_xticks(x, labels)
            a.set_ylim(0, 1)
            a.set_xlabel("representation and component")
    finish(fig, "Unit: athlete; out-of-fold scores at the benchmark K, split repeat 0. Left: within-athlete R² with a 95 % "
                f"athlete bootstrap interval; dotted = heuristic reference ({fcfg.window_reference:g}). Right: cohorts differ "
                "in tracking noise (2025 is much smoother), so a large bar means the component separates recording years, "
                "not athletes.", "Component dependence on the recording window and on the cohort")
    return fig


# ============================================================================== comparison, sensitivity, stability
def plot_representation_comparison(summary: pl.DataFrame, gate: pl.DataFrame, recon: pl.DataFrame):
    fig, ax = figure(2, 2, size=(12.8, 9.0))
    names = summary.filter(pl.col("role") != "alternate").sort("representation")["representation"].to_list()
    g = gate.filter(pl.col("representation").is_in(names)).sort("representation", "component")
    f = lambda d, c: d[c].cast(pl.Float64).fill_null(float("nan")).to_numpy()     # noqa: E731
    for name in names:
        d = recon.filter((pl.col("representation") == name) & (pl.col("repeat") == 0)).sort("k")
        if d.height:
            xk = d["k"].to_numpy() + 0.08 * (names.index(name) - 1)
            ax[0, 0].plot(xk, f(d, "fve_heldout"), color=_rep_color(name), linewidth=1.0)
            ci_errorbar(ax[0, 0], xk, f(d, "fve_heldout"), f(d, "fve_lower"), f(d, "fve_upper"), color=_rep_color(name),
                        label=SPEC[name].label)
        k = summary.filter(pl.col("representation") == name)["k_selected"][0]
        ax[0, 0].axvline(k + 0.08 * (names.index(name) - 1), color=_rep_color(name), linestyle=":", linewidth=1.0)
    ax[0, 0].set_xlabel("number of components K (dotted: benchmark K)")
    ax[0, 0].set_ylabel("held-out share of variance reconstructed")
    ax[0, 0].legend(frameon=False, fontsize=8)
    panels = ((ax[0, 1], "icc_adj", "icc_lower", "icc_upper", "within-position ICC (repeat attempts)"),
              (ax[1, 0], "r2_stopwatch", "r2_stopwatch_lower", "r2_stopwatch_upper",
               "share explained by official forty + 10-yard split"),
              (ax[1, 1], "reliable_unique_share", None, None, "reproducible share not explained by the stopwatch\n(lower bound)"))
    if g.height:
        x = np.arange(g.height)
        labels = [f"{_letter(n)}{c}" for n, c in zip(g["representation"].to_list(), g["component"].to_list())]
        colors = [_rep_color(n) for n in g["representation"].to_list()]
        for a, col, lo, hi, ylabel in panels:
            a.bar(x, f(g, col), color=colors, alpha=0.85)
            if lo:
                ci_errorbar(a, x, f(g, col), f(g, lo), f(g, hi), fmt="none", ecolor=INK["primary"])
            a.set_xticks(x, labels)
            a.set_ylim(min(0.0, float(np.nanmin(np.r_[f(g, col), 0.0]))), 1)
            a.set_ylabel(ylabel)
            a.set_xlabel("representation and component, at the benchmark K")
    else:
        for a, *_ in panels:
            empty_panel(a, "no component")
    n = int(summary["n_athletes_repeat"].max() or 0) if summary.height else 0
    finish(fig, f"Unit: athlete (up to n = {n} with a repeat); out-of-fold scores, split repeat 0, 95 % intervals. A blue, B "
                "orange, C aqua. Top left compares reconstruction; the other panels compare what the components carry. A "
                "representation can reconstruct well and still carry only what the stopwatch already measures.",
           "The three representations compared")
    return fig


def plot_sensitivity_and_stability(sensitivity: pl.DataFrame, components: pl.DataFrame, k_by_rep: dict[str, int]):
    names = _main(list(k_by_rep))
    fig, ax = figure(1, 2)
    rows = [(n, c) for n in names for c in range(1, k_by_rep[n] + 1)]
    f = lambda v: float("nan") if v is None else float(v)     # noqa: E731
    est, lo, hi, cong = [], [], [], []
    for name, c in rows:
        s = sensitivity.filter((pl.col("representation") == name) & (pl.col("k") == k_by_rep[name]) & (pl.col("repeat") == 0)
                               & (pl.col("component") == c) & (pl.col("scope") == "position_adjusted"))
        r = s.row(0, named=True) if s.height else {}
        est.append(f(r.get("r2_from_alternate"))), lo.append(f(r.get("r2_lower"))), hi.append(f(r.get("r2_upper")))
        cong.append(components.filter((pl.col("representation") == name) & (pl.col("component") == c)
                                      & (pl.col("repeat") >= 0))["congruence"].to_numpy())
    if not rows:
        empty_panel(ax[0, 0], "no component")
        empty_panel(ax[0, 1], "no component")
    else:
        x = np.arange(len(rows))
        labels = [f"{_letter(n)}{c}" for n, c in rows]
        colors = [_rep_color(n) for n, _ in rows]
        ax[0, 0].bar(x, est, color=colors, alpha=0.85)
        ci_errorbar(ax[0, 0], x, est, lo, hi, fmt="none", ecolor=INK["primary"])
        ax[0, 0].set_ylabel("share of the component reproduced by the\nalternate alignment's scores (within position)")
        for i, v in enumerate(cong):
            if v.size:
                ax[0, 1].scatter(np.full(v.size, x[i]) + np.linspace(-0.18, 0.18, v.size), v, s=14, color=colors[i], alpha=0.7)
        ax[0, 1].set_ylabel("congruence of the fold's eigenfunction\nwith the reference (1 = same shape)")
        for a in ax[0]:
            a.set_xticks(x, labels)
            a.set_ylim(0, 1.02)
            a.set_xlabel("representation and component, at the benchmark K")
    finish(fig, "Left, unit: athlete scored under both alignments (split repeat 0, 95 % bootstrap interval). Alternates: A′ and "
                "C′ move the origin to the moment the reference speed is reached (attempts starting above it are lost); B′ "
                "keeps acceleration on smoothing-boundary frames. Right, unit: fold fit; one point per outer fold and split "
                "repeat. A low congruence means the component is not the same function from one training set to the next.",
           "Sensitivity to the alignment and stability across training folds")
    return fig
