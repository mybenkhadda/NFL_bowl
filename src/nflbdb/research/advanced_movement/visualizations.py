"""Figures of notebooks 00 and 01. Every function returns a Matplotlib figure; the notebook passes it to
``AdvStore.show`` (inline display, PNG 300 DPI + SVG, plot data, metadata sidecar).

Conventions: 16:9 project style (``research.figkit``), colour-blind-safe categorical palette, units on every axis,
sample size and unit of analysis in the caption, attempts never presented as athletes. ``role`` labels an example as
descriptive / training / validation / held-out; in notebooks 00–01 every example is descriptive.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE
from ..figkit import empty_panel, figure, finish
from .alignment import AlignedCurves
from .config import CHANNELS
from .sequences import SequenceSet

UNIT = {c.name: c.unit for c in CHANNELS}
LABEL = {"speed": "speed (yd/s)", "a_tan": "tangential acceleration (yd/s²)", "a_lat": "lateral acceleration (yd/s²)",
         "lat_dev": "sideways deviation (yd)", "dist": "path length (yd)", "along": "progress along the run line (yd)",
         "yaw_rate": "yaw rate (deg/s)", "t_rel": "time since first frame (s)", "pos_resid": "raw − smoothed position (yd)",
         "head_sin": "sin(relative heading)", "head_cos": "cos(relative heading)"}


def _color(k: int) -> str:
    return CATEGORICAL[k % len(CATEGORICAL)]


def _audit(audit: pl.DataFrame, section: str, metric: str | None = None) -> pl.DataFrame:
    a = audit.filter(pl.col("section") == section)
    return a if metric is None else a.filter(pl.col("metric") == metric)


# ============================================================================== notebook 00
def plot_dataset_flow(audit: pl.DataFrame):
    flow, by = _audit(audit, "flow").sort("item"), _audit(audit, "flow_by_position")
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    if not flow.height:
        empty_panel(a, "no flow rows")
    else:
        names = flow["item"].to_list()
        y = np.arange(len(names))[::-1]
        parts = by.with_columns(pl.col("item").str.split("|").list.get(0).alias("stage"),
                                pl.col("item").str.split("|").list.get(1).alias("pos"))
        left = np.zeros(len(names))
        for k, pos in enumerate(sorted(parts["pos"].unique().to_list())):
            n = dict(parts.filter(pl.col("pos") == pos).select("stage", "value").iter_rows())
            w = np.array([n.get(s, 0.0) or 0.0 for s in names])
            a.barh(y, w, left=left, color=_color(k), label=pos)
            left += w
        for yi, tot in zip(y, flow["value"].to_list()):
            a.text(tot, yi, f"  {int(tot)}", va="center", fontsize=10)
        a.set_yticks(y, [f"{s[3:].replace('_', ' ')}" for s in names])
        a.set_xlabel("athletes")
        a.legend(title="position group", frameon=False, loc="lower right")
    finish(fig, "Unit: athlete. Each stage adds its condition to the previous one (definitions in dataset_audit.csv, "
                "section 'flow'). The last stages are the athletes available to a Combine → NFL comparison; the "
                "sequence models themselves see every athlete of stage 02.", "From the Combine to an NFL comparison")
    return fig


def plot_coverage_by_position(audit: pl.DataFrame):
    a_ = _audit(audit, "athletes_by_position_group")
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    pos = sorted(a_["item"].unique().to_list()) if a_.height else []
    metrics = (("n_athletes", "Combine athletes"), ("n_with_primary_sequence", "eligible forty sequence"),
               ("n_with_repeat", "repeated forty"), ("n_nfl_reg", "NFL regular-season tracking"))
    if not pos:
        empty_panel(a, "no athlete rows")
    else:
        w = 0.8 / len(metrics)
        for k, (m, label) in enumerate(metrics):
            v = dict(a_.filter(pl.col("metric") == m).select("item", "value").iter_rows())
            x = np.arange(len(pos)) + (k - (len(metrics) - 1) / 2) * w
            bars = a.bar(x, [v.get(p, 0) or 0 for p in pos], width=w, color=_color(k), label=label)
            a.bar_label(bars, fmt="%d", fontsize=8, padding=2)
        a.set_xticks(range(len(pos)), pos)
        a.set_ylabel("athletes")
        a.set_ylim(0, None)
        a.legend(frameon=False)
    finish(fig, "Unit: athlete. 'Repeated forty' = at least two eligible sequences, the basis of every repeatability "
                "estimate and of the same-athlete consistency objective.", "Athlete coverage by position group")
    return fig


def plot_attempt_counts(athletes: pl.DataFrame):
    t = athletes.filter(pl.col("n_primary_eligible") >= 1)
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    if not t.height:
        empty_panel(a, "no athlete with an eligible sequence")
    else:
        counts = sorted(t["n_primary_eligible"].unique().to_list())
        bottom = np.zeros(len(counts))
        for k, pos in enumerate(sorted(t["position_group"].drop_nulls().unique().to_list())):
            n = dict(t.filter(pl.col("position_group") == pos).group_by("n_primary_eligible").len().iter_rows())
            h = np.array([n.get(c, 0) for c in counts], float)
            a.bar([str(c) for c in counts], h, bottom=bottom, color=_color(k), label=pos)
            bottom += h
        for c, tot in zip(counts, bottom):
            a.text(str(c), tot, f"{int(tot)}", ha="center", va="bottom", fontsize=10)
        a.set_xlabel("eligible forty sequences per athlete")
        a.set_ylabel("athletes")
        a.legend(title="position group", frameon=False)
    finish(fig, f"Unit: athlete (n = {t.height}). Athletes with one sequence contribute to reconstruction objectives "
                "only; they cannot form a same-athlete positive pair.", "How many attempts does an athlete have?")
    return fig


def plot_sequence_lengths(manifest: pl.DataFrame, primary_drill: str):
    m = manifest.filter(pl.col("drill") == primary_drill)
    fig, ax = figure(1, 2)
    if not m.height:
        empty_panel(ax[0, 0], "no primary-drill sequence")
        empty_panel(ax[0, 1], "no primary-drill sequence")
    else:
        hi = int(m["longest_valid"].max() or 1)
        bins = np.arange(0, hi + 3, 2)
        for k, (label, df) in enumerate((("eligible", m.filter(pl.col("eligible_primary"))),
                                         ("excluded", m.filter(~pl.col("eligible_primary"))))):
            x = df["longest_valid"].drop_nulls().to_numpy()
            if x.size:
                ax[0, 0].hist(x, bins=bins, histtype="stepfilled" if k == 0 else "step", alpha=0.6 if k == 0 else 1.0,
                              color=_color(k), label=f"{label} (n = {x.size})")
        ax[0, 0].set_xlabel("frames in the longest valid segment (10 frames = 1 s)")
        ax[0, 0].set_ylabel("sequences")
        ax[0, 0].legend(frameon=False)
        e = m.filter(pl.col("eligible_primary"))
        years = sorted(e["draft_year"].drop_nulls().unique().to_list())
        if years:
            ax[0, 1].boxplot([e.filter(pl.col("draft_year") == y)["longest_valid"].to_numpy() for y in years], showfliers=True)
            ax[0, 1].set_xticks(range(1, len(years) + 1), [f"{y}\n(n = {e.filter(pl.col('draft_year') == y).height})" for y in years])
        ax[0, 1].set_ylabel("frames in the sequence")
        ax[0, 1].set_xlabel("draft cohort")
    finish(fig, "Unit: Combine attempt (sequence). Left: all forty attempts, eligible against excluded. Right: eligible "
                "sequences by cohort. Sequences differ in length, so every model must mask padding.",
           "How long are the sequences?")
    return fig


def plot_cohort_exposure(athletes: pl.DataFrame, thresholds: Sequence[int]):
    t = athletes.filter(pl.col("n_primary_eligible") >= 1)
    fig, ax = figure(1, 2)
    years = sorted(t["draft_year"].drop_nulls().unique().to_list())
    if not years:
        empty_panel(ax[0, 0], "no cohort")
        empty_panel(ax[0, 1], "no cohort")
    else:
        data = [np.log10(t.filter((pl.col("draft_year") == y) & (pl.col("nfl_reg_plays") >= 1))["nfl_reg_plays"].to_numpy() + 0.0)
                for y in years]
        ax[0, 0].boxplot([d if d.size else np.array([np.nan]) for d in data], showfliers=True)
        ax[0, 0].set_xticks(range(1, len(years) + 1), [str(y) for y in years])
        ticks = [1, 10, 100, 1000]
        ax[0, 0].set_yticks(np.log10(ticks), [str(v) for v in ticks])
        ax[0, 0].set_ylabel("tracked regular-season plays (log scale), athletes with ≥ 1")
        ax[0, 0].set_xlabel("draft cohort")
        w = 0.8 / (len(thresholds) + 1)
        for k, n in enumerate([0, *thresholds]):
            h = [t.filter((pl.col("draft_year") == y) & (pl.col("nfl_reg_plays") >= max(n, 1))).height if n else
                 t.filter(pl.col("draft_year") == y).height for y in years]
            bars = ax[0, 1].bar(np.arange(len(years)) + (k - len(thresholds) / 2) * w, h, width=w, color=_color(k),
                                label="with a forty sequence" if n == 0 else f"≥ {n} regular-season plays")
            ax[0, 1].bar_label(bars, fmt="%d", fontsize=8, padding=2)
        ax[0, 1].set_xticks(range(len(years)), [str(y) for y in years])
        ax[0, 1].set_ylabel("athletes")
        ax[0, 1].set_xlabel("draft cohort")
        ax[0, 1].legend(frameon=False, fontsize=9)
    finish(fig, f"Unit: athlete with an eligible forty sequence (n = {t.height}). Later cohorts have had fewer seasons to "
                "accumulate plays: NFL exposure and draft cohort are confounded, so cohort is always a covariate and a "
                "sensitivity.", "NFL exposure by draft cohort")
    return fig


def plot_missingness(missing: pl.DataFrame):
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    if not missing.height:
        empty_panel(a, "no missingness rows")
    else:
        vars_ = list(dict.fromkeys(missing["variable"].to_list()))
        groups = list(dict.fromkeys(missing["group"].to_list()))
        val = {(r["variable"], r["group"]): r for r in missing.iter_rows(named=True)}
        z = np.array([[val[(v, g)]["share_missing"] if (v, g) in val and val[(v, g)]["share_missing"] is not None else np.nan
                       for g in groups] for v in vars_])
        from matplotlib.colors import LinearSegmentedColormap

        im = a.imshow(z, cmap=LinearSegmentedColormap.from_list("miss", ["#ffffff", SEQUENTIAL_BLUE[-1]]), vmin=0, vmax=1,
                      aspect="auto")
        for i in range(len(vars_)):
            for j in range(len(groups)):
                if np.isfinite(z[i, j]):
                    a.text(j, i, f"{100 * z[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                           color="white" if z[i, j] > 0.55 else INK["primary"])
        a.set_xticks(range(len(groups)), [f"{g}\n(n = {val[(vars_[0], g)]['n']})" for g in groups])
        a.set_yticks(range(len(vars_)), [v.replace("_", " ") for v in vars_])
        a.grid(False)
        fig.colorbar(im, ax=a, label="share of athletes without the measurement")
    finish(fig, "Unit: Combine athlete. Rows: official measurements, tracked sequences and NFL outcomes. A missing official "
                "time removes the athlete from every model that uses it; nothing is imputed for the official times.",
           "What is missing, and for whom?")
    return fig


def plot_split_balance(splits: pl.DataFrame, athletes: pl.DataFrame):
    s = splits.filter(pl.col("repeat") == 0).join(athletes.select("nfl_id", "position_group"), on="nfl_id", how="left")
    fig, ax = figure(1, 2)
    if not s.height:
        empty_panel(ax[0, 0], "no split")
        empty_panel(ax[0, 1], "no split")
    else:
        folds = sorted(s["outer_fold"].unique().to_list())
        bottom = np.zeros(len(folds))
        for k, pos in enumerate(sorted(s["position_group"].drop_nulls().unique().to_list())):
            n = dict(s.filter(pl.col("position_group") == pos).group_by("outer_fold").len().iter_rows())
            h = np.array([n.get(f, 0) for f in folds], float)
            ax[0, 0].bar([str(f) for f in folds], h, bottom=bottom, color=_color(k), label=pos)
            bottom += h
        ax[0, 0].set_xlabel("outer fold (repeat 0)")
        ax[0, 0].set_ylabel("held-out athletes")
        ax[0, 0].legend(title="position group", frameon=False, fontsize=9)
        roles = ["design", "confirm"]
        bottom = np.zeros(2)
        for k, pos in enumerate(sorted(s["position_group"].drop_nulls().unique().to_list())):
            h = np.array([s.filter((pl.col("position_group") == pos) & (pl.col("design_role") == r)).height for r in roles], float)
            ax[0, 1].bar(roles, h, bottom=bottom, color=_color(k))
            bottom += h
        for r, tot in zip(roles, bottom):
            ax[0, 1].text(r, tot, f"{int(tot)}", ha="center", va="bottom", fontsize=10)
        ax[0, 1].set_ylabel("athletes")
        ax[0, 1].set_xlabel("role in preprocessing decisions")
    finish(fig, f"Unit: athlete (n = {s.height}). Left: every athlete is held out exactly once per repeat; folds are "
                "stratified by position group × draft cohort. Right: preprocessing choices are quantified on the design "
                "athletes and checked on the others.", "The split registry")
    return fig


# ============================================================================== notebook 01
def plot_attempt_panels(ss: SequenceSet, seq_id: str, role: str = "descriptive", title: str | None = None,
                        channels: Sequence[str] = ("speed", "a_tan", "lat_dev", "dist")):
    """Up to four signal panels of one sequence against time since the first recorded frame."""
    i = ss.index_of(seq_id)
    k = int(ss.lengths[i])
    t = ss.t[i, :k] - ss.t[i, 0]
    m = ss.meta.row(i, named=True)
    channels = [c for c in channels if c in ss.channels][:4]
    fig, ax = figure(2, 2)
    for j in range(4):
        a = ax[j // 2, j % 2]
        if j >= len(channels):
            empty_panel(a, "—")
            continue
        y = ss.series(i, channels[j])
        a.plot(t, y, color=_color(j), marker="o", markersize=3)
        b = ss.boundary[i, :k]
        for lo, hi in _runs(b):
            a.axvspan(t[lo] - 0.05, t[hi] + 0.05, color=INK["grid"], alpha=0.7, lw=0)
        if channels[j] in ("a_tan", "lat_dev", "a_lat", "yaw_rate"):
            a.axhline(0, color=INK["axis"], linewidth=0.8)
        a.set_xlabel("time since first recorded frame (s)")
        a.set_ylabel(LABEL.get(channels[j], channels[j]))
    head = title or f"{m.get('position_group')} · cohort {m.get('draft_year')} · attempt {m.get('attempt_order')}"
    finish(fig, f"Unit: frame of one Combine attempt ({seq_id}, {k} frames at 10 Hz). Shaded: frames inside the smoothing "
                "half-window of the segment edges, where derivatives are less reliable. The recording covers the frames "
                f"shown and nothing before or after. Example role: {role}.", head)
    return fig


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flag) - 1))
    return out


def plot_repeat_attempts(ss: SequenceSet, athlete_ids: Sequence[int], names: dict[int, str] | None = None,
                         role: str = "descriptive"):
    """One column per athlete: speed against time (top) and acceleration against speed (bottom), every attempt."""
    ids = list(athlete_ids)[:4]
    fig, ax = figure(2, max(len(ids), 1))
    if not ids:
        empty_panel(ax[0, 0], "no athlete with repeated attempts")
        empty_panel(ax[1, 0], "no athlete with repeated attempts")
    for c, nfl_id in enumerate(ids):
        rows = np.flatnonzero(ss.nfl_ids == nfl_id)
        order = np.argsort(ss.meta["attempt_order"].to_numpy()[rows])
        for k, i in enumerate(rows[order]):
            n = int(ss.lengths[i])
            t = ss.t[i, :n] - ss.t[i, 0]
            lab = f"attempt {ss.meta['attempt_order'][int(i)]}"
            ax[0, c].plot(t, ss.series(int(i), "speed"), color=_color(k), label=lab)
            ax[1, c].plot(ss.series(int(i), "speed"), ss.series(int(i), "a_tan"), color=_color(k), label=lab)
        pos = ss.meta["position_group"][int(rows[0])]
        ax[0, c].set_title(f"{(names or {}).get(nfl_id, nfl_id)} · {pos}", fontsize=11)
        ax[0, c].set_xlabel("time since first recorded frame (s)")
        ax[0, c].set_ylabel(LABEL["speed"])
        ax[1, c].set_xlabel(LABEL["speed"])
        ax[1, c].set_ylabel(LABEL["a_tan"])
        ax[1, c].axhline(0, color=INK["axis"], linewidth=0.8)
        ax[0, c].legend(frameon=False, fontsize=8)
    finish(fig, "Unit: frame; one column per athlete, one line per attempt. Top: the clock starts at the first RECORDED "
                "frame, so two attempts of an athlete can be shifted against each other. Bottom: acceleration at a given "
                f"speed does not depend on where the recording starts. Example role: {role}.",
           "Do an athlete's attempts look alike?")
    return fig


def plot_matched_forty(ss: SequenceSet, pairs: pl.DataFrame, role: str = "descriptive"):
    """Pairs of athletes with the same official forty: speed against time and acceleration against speed."""
    p = pairs.head(3)
    fig, ax = figure(2, max(p.height, 1))
    if not p.height:
        empty_panel(ax[0, 0], "no pair inside the caliper")
        empty_panel(ax[1, 0], "no pair inside the caliper")
    for c, r in enumerate(p.iter_rows(named=True)):
        for k, side in enumerate(("a", "b")):
            i = ss.index_of(r[f"seq_id_{side}"])
            n = int(ss.lengths[i])
            lab = f"{r[f'name_{side}']} · {r[f'forty_{side}']:.2f} s"
            ax[0, c].plot(ss.t[i, :n] - ss.t[i, 0], ss.series(i, "speed"), color=_color(k), label=lab)
            ax[1, c].plot(ss.series(i, "speed"), ss.series(i, "a_tan"), color=_color(k), label=lab)
        ax[0, c].set_title(str(r["position_group"]), fontsize=11)
        ax[0, c].set_xlabel("time since first recorded frame (s)")
        ax[0, c].set_ylabel(LABEL["speed"])
        ax[1, c].set_xlabel(LABEL["speed"])
        ax[1, c].set_ylabel(LABEL["a_tan"])
        ax[0, c].legend(frameon=False, fontsize=8)
    finish(fig, "Unit: frame; one column per pair of athletes of the same position whose official forty differs by at "
                "most 0.01 s (first attempts, comparable recording start). Pairs are chosen by rule, largest difference "
                "in peak acceleration first. A visible difference on ONE attempt is not yet a repeatable difference: "
                f"repeatability is measured in notebook 05. Example role: {role}.",
           "Same stopwatch, different run?")
    return fig


def plot_group_curves(curves: AlignedCurves, channel: str, groups: np.ndarray, title: str, min_sequences: int = 10):
    """Median and inter-quartile band of one aligned channel per group (position or cohort)."""
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    v = curves.channel(channel)
    drawn = 0
    for k, g in enumerate(sorted({str(x) for x in groups})):
        rows = np.array([str(x) == g for x in groups])
        sub = v[rows]
        enough = np.isfinite(sub).sum(0) >= min_sequences
        if not enough.any():
            continue
        with np.errstate(all="ignore"):
            q25, q50, q75 = np.nanquantile(sub[:, enough], [0.25, 0.5, 0.75], axis=0)
        x = curves.grid[enough]
        a.plot(x, q50, color=_color(k), label=f"{g} (n = {int(rows.sum())} sequences)")
        a.fill_between(x, q25, q75, color=_color(k), alpha=0.18, lw=0)
        drawn += 1
    if not drawn:
        empty_panel(a, "no group with enough sequences")
    else:
        a.set_xlabel(f"{curves.grid_name} ({curves.grid_unit})")
        a.set_ylabel(LABEL.get(channel, channel))
        a.legend(frameon=False, fontsize=9)
    finish(fig, f"Unit: sequence (attempts, not athletes). Line = median, band = inter-quartile range across sequences, "
                f"shown where at least {min_sequences} sequences of the group are observed. Alignment: {curves.method}.", title)
    return fig


def plot_recording_window(summary: pl.DataFrame):
    fig, ax = figure(1, 3)
    if not summary.height:
        for j in range(3):
            empty_panel(ax[0, j], "no sequence")
    else:
        first = summary["first_speed"].to_numpy()
        bins = np.linspace(0, max(float(np.nanquantile(first, 0.99)), 0.5), 40)
        for k, y in enumerate(sorted(summary["draft_year"].drop_nulls().unique().to_list())):
            ax[0, 0].hist(summary.filter(pl.col("draft_year") == y)["first_speed"].to_numpy(), bins=bins, histtype="step",
                          linewidth=1.8, color=_color(k), label=str(y))
        ax[0, 0].set_xlabel("speed on the first recorded frame (yd/s)")
        ax[0, 0].set_ylabel("sequences")
        ax[0, 0].legend(title="draft cohort", frameon=False)
        pos = sorted(summary["position_group"].drop_nulls().unique().to_list())
        ax[0, 1].boxplot([summary.filter(pl.col("position_group") == p)["duration_s"].to_numpy() for p in pos], showfliers=True)
        ax[0, 1].set_xticks(range(1, len(pos) + 1), pos)
        ax[0, 1].set_ylabel("recorded duration (s)")
        ax[0, 1].set_xlabel("position group")
        after = summary["frames_after_peak"].to_numpy()
        ax[0, 2].hist(after, bins=np.arange(-0.5, max(int(after.max()), 1) + 1.5, 1), color=_color(0))
        ax[0, 2].set_xlabel("frames recorded after the speed peak")
        ax[0, 2].set_ylabel("sequences")
    finish(fig, f"Unit: sequence (n = {summary.height}). Left: a first-frame speed above zero means the recording began "
                "after the start — the run is left-censored. Right: zero frames after the peak means the recording ends "
                "while the athlete is still at or near top speed — right-censored. Histogram truncated at the 99th "
                "percentile.", "What part of the run does the recording hold?")
    return fig


def plot_speed_vs_displacement(ss: SequenceSet, max_lines: int = 250, seed: int = 0):
    fig, ax = figure(1, 2)
    idx = np.arange(ss.n)
    if ss.n > max_lines:
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=max_lines, replace=False))
    pos = ss.meta["position_group"].fill_null("?").to_numpy()
    groups = sorted(set(pos.tolist()))
    for k, g in enumerate(groups):
        first = True
        for i in idx[pos[idx] == g]:
            ax[0, 0].plot(ss.series(int(i), "dist"), ss.series(int(i), "speed"), color=_color(k), alpha=0.25, linewidth=0.8,
                          label=g if first else None)
            ax[0, 1].plot(ss.series(int(i), "speed"), ss.series(int(i), "a_tan"), color=_color(k), alpha=0.25, linewidth=0.8)
            first = False
    ax[0, 0].set_xlabel("path length since first recorded frame (yd)")
    ax[0, 0].set_ylabel(LABEL["speed"])
    ax[0, 1].set_xlabel(LABEL["speed"])
    ax[0, 1].set_ylabel(LABEL["a_tan"])
    ax[0, 1].axhline(0, color=INK["axis"], linewidth=0.8)
    leg = ax[0, 0].legend(title="position group", frameon=False, fontsize=9)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    finish(fig, f"Unit: frame; one line per sequence ({idx.size} of {ss.n} sequences drawn, seeded sample). Left: speed "
                "against distance covered in the recording. Right: the phase plane, acceleration against speed, which does "
                "not depend on where the recording starts.", "Speed development across all sequences")
    return fig


def plot_variable_length(ss: SequenceSet):
    fig, ax = figure(1, 2)
    order = np.argsort(ss.lengths)
    ax[0, 0].imshow(ss.mask[order], aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1.4)
    ax[0, 0].set_xlabel("frame index in the padded array")
    ax[0, 0].set_ylabel("sequences, sorted by length")
    ax[0, 0].grid(False)
    miss = ss.channel_missingness()
    y = np.arange(miss.height)[::-1]
    ax[0, 1].barh(y, 100 * miss["share_undefined"].to_numpy(), color=_color(0))
    ax[0, 1].set_yticks(y, miss["channel"].to_list())
    ax[0, 1].set_xlabel("real frames on which the channel is undefined (%)")
    finish(fig, f"Unit: left, cell of the padded array ({ss.n} sequences × {ss.mask.shape[1]} frames; dark = real frame, "
                f"white = padding, {100 * (1 - ss.mask.mean()):.1f} % of cells are padding); right, real frame. Padding and "
                "undefined values are masked in every statistic and loss; they are never treated as observations.",
           "Variable length and per-channel missingness")
    return fig


def plot_noise(summary: pl.DataFrame):
    fig, ax = figure(1, 2)
    years = sorted(summary["draft_year"].drop_nulls().unique().to_list()) if summary.height else []
    if not years:
        empty_panel(ax[0, 0], "no sequence")
        empty_panel(ax[0, 1], "no sequence")
    else:
        for j, (col, label) in enumerate((("pos_resid_rms", "RMS distance raw − smoothed position (yd)"),
                                          ("a_tan_roughness", "SD of frame-to-frame change in acceleration (yd/s²)"))):
            ax[0, j].boxplot([summary.filter(pl.col("draft_year") == y)[col].drop_nulls().to_numpy() for y in years], showfliers=True)
            ax[0, j].set_xticks(range(1, len(years) + 1), [f"{y}\n(n = {summary.filter(pl.col('draft_year') == y).height})" for y in years])
            ax[0, j].set_ylabel(label)
            ax[0, j].set_xlabel("draft cohort")
            ax[0, j].set_ylim(0, None)
    finish(fig, "Unit: sequence. Left: positional tracking noise. Right: roughness of the derived acceleration. A model "
                "can learn to recognise a cohort from its noise level; that is a nuisance to test for (notebook 05), not a "
                "movement trait.", "Measurement noise by draft cohort")
    return fig


def plot_alignment_example(ss: SequenceSet, aligned: dict[str, AlignedCurves], nfl_id: int, channel: str,
                           name: str | None = None, role: str = "descriptive"):
    """The attempts of ONE athlete under each alignment (four panels)."""
    fig, ax = figure(2, 2)
    for j, (method, curves) in enumerate(list(aligned.items())[:4]):
        a = ax[j // 2, j % 2]
        ch = channel if channel in curves.channels else next((c for c in ("a_tan", "speed") if c in curves.channels), None)
        rows = np.flatnonzero(curves.nfl_ids == nfl_id)
        if ch is None or not rows.size:
            empty_panel(a, f"{method}: nothing to show")
            continue
        for k, i in enumerate(rows):
            order = ss.meta["attempt_order"][ss.index_of(str(curves.seq_ids[i]))]
            a.plot(curves.grid, curves.channel(ch)[i], color=_color(k), label=f"attempt {order}")
        a.set_title(method.replace("_", " "), fontsize=11)
        a.set_xlabel(f"{curves.grid_name} ({curves.grid_unit})")
        a.set_ylabel(LABEL.get(ch, ch))
        a.legend(frameon=False, fontsize=8)
    finish(fig, f"Unit: grid point of an aligned curve; athlete {name or nfl_id}, one line per attempt. Lines stop where "
                f"the recording holds no data (no extrapolation). Example role: {role}.",
           "One athlete, four alignments")
    return fig


def plot_alignment_comparison(comparison: pl.DataFrame, scope: str = "position_adjusted"):
    c = comparison.filter((pl.col("scope") == scope) & pl.col("icc_integrated").is_not_null())
    fig, ax = figure(1, 2)
    if not c.height:
        empty_panel(ax[0, 0], "alignment comparison not estimable")
        empty_panel(ax[0, 1], "alignment comparison not estimable")
        finish(fig, "No estimable row.", "Which alignment gives repeatable curves?")
        return fig
    methods = list(dict.fromkeys(c["method"].to_list()))
    channels = list(dict.fromkeys(c["channel"].to_list()))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        for set_name, marker, face in (("design", "o", None), ("confirm", "s", "white")):
            g = {r["channel"]: r for r in c.filter((pl.col("method") == m) & (pl.col("athlete_set") == set_name)).iter_rows(named=True)}
            xs = [i + (k - (len(methods) - 1) / 2) * w for i, ch in enumerate(channels) if ch in g]
            if xs:
                ax[0, 0].plot(xs, [g[ch]["icc_integrated"] for ch in channels if ch in g], linestyle="none", marker=marker,
                              markersize=8, color=_color(k), markerfacecolor=face or _color(k),
                              label=f"{m.replace('_', ' ')} · {set_name}")
        g = {r["channel"]: r for r in c.filter((pl.col("method") == m) & (pl.col("athlete_set") == "all")).iter_rows(named=True)}
        xs = [i + (k - (len(methods) - 1) / 2) * w for i, ch in enumerate(channels) if ch in g]
        if xs:
            ax[0, 1].bar(xs, [g[ch]["coverage"] for ch in channels if ch in g], width=w, color=_color(k), label=m.replace("_", " "))
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xticks(range(len(channels)), [ch.replace("_", " ") for ch in channels])
        a.set_ylim(0, 1)
    ax[0, 0].set_ylabel("integrated ICC of the aligned curve")
    ax[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    ax[0, 1].set_ylabel("share of grid points × sequences observed")
    ax[0, 1].legend(frameon=False, fontsize=8)
    finish(fig, f"Unit: athlete with repeated attempts ({scope.replace('_', ' ')}). Left: share of the curve's variance that "
                "belongs to the athlete rather than to the attempt; filled = design athletes (decision subset), hollow = "
                "the other athletes. Right: how much of each grid the recordings cover. A channel is absent where it is the "
                "alignment's own axis. High repeatability is necessary, not sufficient: an alignment can be stable because "
                "it removed what distinguishes athletes.", "Which alignment gives repeatable curves?")
    return fig
