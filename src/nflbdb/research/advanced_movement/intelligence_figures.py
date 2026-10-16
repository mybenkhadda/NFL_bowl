"""Publication figures of Milestone 09 (notebooks 09A–09G). Matplotlib only; every function returns a figure drawn from
tables the notebook computed (real athletes and real plays, no invented value). An estimate carries its interval and
its sample size; an empty input gives an explicit "no data" panel. Names: ``AM09<notebook>_<nn>_<slug>``.

Colour: athlete A = first categorical hue, athlete B = second (fixed, never by rank); status colours only for verdicts.
"""

from __future__ import annotations

import textwrap
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS
from ..figkit import empty_panel, figure, finish
from . import movement_demands as md
from .scouting_interpretation import DISCLAIMER

A_COL, B_COL = CATEGORICAL[0], CATEGORICAL[1]
VERDICT_COLOR = {"distinguishable": STATUS["good"], "not_distinguishable": INK["axis"], "insufficient_evidence": STATUS["warning"], "not_observed": INK["muted"]}
NAVY, GOLD = "#0b1f3a", "#c9a227"


def traces(frames: pl.DataFrame, seq_ids: Sequence[str], x: str, y: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(x, y)] of each sequence as recorded (``sequence_frames`` columns)."""
    f = frames.filter(pl.col("seq_id").is_in(list(seq_ids))).sort("seq_id", "frame")
    return [(g[x].to_numpy().astype(float), g[y].to_numpy().astype(float)) for _, g in f.partition_by(["seq_id"], as_dict=True, maintain_order=True).items()]


def _positions(df: pl.DataFrame) -> list[str]:
    return [p for p in sorted(df["position_group"].unique().to_list()) if p != md.ALL]


# ============================================================================== 09A — demand atlas
def fig_demand_frequency(freq: pl.DataFrame) -> Any:
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    pos, dem = _positions(freq), [d for d in freq["demand_id"].unique(maintain_order=True).to_list()]
    if not pos:
        empty_panel(ax, "no demand episode")
        return fig
    cell = {(r["position_group"], r["demand_id"]): r for r in freq.iter_rows(named=True)}
    M = np.array([[cell[(p, d)]["episodes_per_100_plays"] if cell[(p, d)]["supported"] else np.nan for p in pos] for d in dem], dtype=float)
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0)
    for i, d in enumerate(dem):
        for j, p in enumerate(pos):
            r = cell[(p, d)]
            txt = f"{r['episodes_per_100_plays']:.0f}\n(n = {r['n_episodes']:,}; {r['n_athletes']} ath.)" if r["supported"] else f"too few\n(n = {r['n_episodes']:,})"
            dark = np.isfinite(M[i, j]) and M[i, j] > 0.6 * np.nanmax(M)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if dark else INK["primary"])
    ax.set_xticks(range(len(pos)), pos)
    ax.set_yticks(range(len(dem)), dem)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="episodes per 100 eligible player-plays")
    finish(fig, "Observed game movement, not a physical requirement: rates follow role, opportunity and the thresholds of the atlas. n = episodes (dependent); "
                "athletes are the sample size.", "NFL movement demands by position — how often each occurs")
    return fig


def fig_demand_profiles(profile: pl.DataFrame, registry: pl.DataFrame) -> Any:
    prim = profile.filter(pl.col("is_primary") & (pl.col("position_group") != md.ALL))
    dem = registry["demand_id"].to_list()
    fig, axes = figure(3, 3, size=(14.0, 10.0))
    label = dict(zip(registry["demand_id"].to_list(), registry["label"].to_list()))
    for ax, d in zip(axes.ravel(), dem):
        g = prim.filter((pl.col("demand_id") == d) & pl.col("supported")).sort("position_group")
        if g.is_empty():
            empty_panel(ax, f"{label[d]}\nno supported position")
            continue
        y = np.arange(g.height)[::-1]
        for yi, r in zip(y, g.iter_rows(named=True)):
            ax.plot([r["q10"], r["q90"]], [yi, yi], color=SEQUENTIAL_BLUE[2], lw=2, solid_capstyle="round")
            ax.plot([r["q25"], r["q75"]], [yi, yi], color=SEQUENTIAL_BLUE[4], lw=6, solid_capstyle="round")
            ax.plot(r["q50"], yi, "o", color="white", mec=SEQUENTIAL_BLUE[5], ms=6)
            ax.annotate(f"{r['n_episodes']:,} ep · {r['n_athletes']} ath", (1.01, yi), xycoords=("axes fraction", "data"), fontsize=7, color=INK["muted"], va="center")
        ax.set_yticks(y, g["position_group"].to_list())
        ax.set_ylim(-0.7, g.height - 0.3)
        ax.set_title(label[d], fontsize=10, loc="left")
        ax.set_xlabel(f"{g['feature'][0]} [{g['unit'][0]}]", fontsize=8)
    for ax in axes.ravel()[len(dem):]:
        ax.axis("off")
    finish(fig, "Thin line = 10th–90th percentile of episodes, thick = 25th–75th, dot = median. Redirection shows the magnitude of the heading change.",
           "Intensity of each movement demand by position")
    return fig


def fig_athlete_share(profile: pl.DataFrame) -> Any:
    g = profile.filter(pl.col("is_primary") & (pl.col("position_group") == md.ALL) & pl.col("athlete_share").is_not_null()).sort("athlete_share")
    fig, ax = figure(1, 1, size="panel")
    ax = ax[0, 0]
    if g.is_empty():
        empty_panel(ax, "variance components not estimable")
        return fig
    ax.barh(g["demand_id"].to_list(), g["athlete_share"].to_numpy(), color=SEQUENTIAL_BLUE[3], height=0.6)
    for i, r in enumerate(g.iter_rows(named=True)):
        ax.text(r["athlete_share"] + 0.01, i, f"{r['athlete_share']:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, max(0.5, float(g["athlete_share"].max()) * 1.25))
    ax.set_xlabel("athlete share of episode variance (one-way method of moments)")
    finish(fig, "Positions pooled without adjustment: part of the athlete share is position. The remainder is play-to-play variation.", "How much of a demand episode is the athlete?")
    return fig


def fig_examples(examples: pl.DataFrame, position: str) -> Any:
    g = examples.filter(pl.col("position_group") == position)
    ids = g["example_id"].unique(maintain_order=True).to_list()
    if not ids:
        fig, ax = figure(1, 1, size="panel")
        empty_panel(ax[0, 0], f"no example episode for {position}")
        return fig
    ncols = 3
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = figure(nrows, ncols, size=(14.0, 3.2 * nrows))
    for ax, eid in zip(axes.ravel(), ids):
        e = g.filter(pl.col("example_id") == eid).sort("frame")
        t, v, inside = e["t_s"].to_numpy(), e["speed"].to_numpy(), e["in_episode"].to_numpy()
        ax.plot(t, v, color=INK["axis"], lw=1.5)
        ax.plot(t[inside], v[inside], color=SEQUENTIAL_BLUE[4], lw=2.5)
        ax.set_title(f"{e['demand_id'][0]} · q{int(round(100 * e['quantile'][0]))} · game {e['game_id'][0]} play {e['play_id'][0]}", fontsize=8, loc="left")
        ax.set_xlabel("time since episode start [s]", fontsize=8)
        ax.set_ylabel("speed [yd/s]", fontsize=8)
    for ax in axes.ravel()[len(ids):]:
        ax.axis("off")
    finish(fig, "Real recorded plays chosen by rule: the episode closest to the stated quantile of its position × demand cell. Blue = the episode; grey = margin.",
           f"Example NFL movement-demand episodes — {position}")
    return fig


# ============================================================================== 09B — matching
def fig_feature_evidence(evidence: pl.DataFrame, icc_ref: float, r2_ref: float) -> Any:
    fig, ax = figure(1, 1, size="panel")
    ax = ax[0, 0]
    g = evidence.filter(pl.col("icc_within_position").is_not_null() & pl.col("r2_official_times").is_not_null())
    if g.is_empty():
        empty_panel(ax, "no feature with both estimates")
        return fig
    ax.axhline(icc_ref, color=INK["axis"], lw=1, ls="--")
    ax.axvline(r2_ref, color=INK["axis"], lw=1, ls="--")
    ax.scatter(g["r2_official_times"].to_numpy(), g["icc_within_position"].to_numpy(), s=60, color=SEQUENTIAL_BLUE[4], zorder=3)
    for r in g.iter_rows(named=True):
        ax.annotate(r["feature"], (r["r2_official_times"], r["icc_within_position"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("R² on official forty + 10-yard split (in-sample, within position)")
    ax.set_ylabel("repeat-attempt ICC (within position)")
    finish(fig, "Upper left = repeatable and not carried by the stopwatch. Repeatability is not football value.", "Combine curve features: repeatable? already in the stopwatch?")
    return fig


def fig_similarity_heatmap(summary: pl.DataFrame) -> Any:
    fig, ax = figure(1, 1)
    ax = ax[0, 0]
    g = summary.filter(pl.col("n_prospects") > 0)
    if g.is_empty():
        empty_panel(ax, "no supported prospect × demand comparison")
        return fig
    cols = sorted({(r["demand_id"], r["dimension"]) for r in g.iter_rows(named=True)})
    pos = sorted(g["position_group"].unique().to_list())
    cell = {(r["position_group"], r["demand_id"], r["dimension"]): r for r in g.iter_rows(named=True)}
    M = np.array([[cell.get((p, *c), {}).get("median_coverage", np.nan) for c in cols] for p in pos], dtype=float)
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    for i, p in enumerate(pos):
        for j, c in enumerate(cols):
            r = cell.get((p, *c))
            if r:
                ax.text(j, i, f"{r['median_coverage']:.0f}\n(n = {r['n_prospects']})", ha="center", va="center", fontsize=8, color="white" if r["median_coverage"] > 60 else INK["primary"])
            else:
                ax.text(j, i, "abstain", ha="center", va="center", fontsize=8, color=INK["muted"])
    ax.set_xticks(range(len(cols)), [f"{d}\n{k}" for d, k in cols], fontsize=8)
    ax.set_yticks(range(len(pos)), pos)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="median coverage of prospects [% of NFL episodes at or below the Combine value]")
    finish(fig, "Coverage = where a maximal-effort Combine value sits inside the observed game demand of the same position. Descriptive physical similarity — "
                "not a prediction. Demands without a forty analogue are absent by design.", "Combine-to-NFL movement matching by position")
    return fig


def fig_alignment(rows: Sequence[dict[str, Any]], name: str) -> Any:
    """One prospect: his Combine value (with interval) inside the NFL demand range of his position, per shared dimension."""
    ok = [r for r in rows if r.get("combine_value") is not None and r.get("nfl_q50") is not None]
    units = list(dict.fromkeys(r["unit"] for r in ok))
    fig, axes = figure(1, max(1, len(units)), size=(13.0, 5.0))
    if not ok:
        empty_panel(axes[0, 0], "no shared dimension recorded for this prospect")
        return fig
    for ax, u in zip(axes[0], units):
        g = [r for r in ok if r["unit"] == u]
        y = np.arange(len(g))[::-1]
        for yi, r in zip(y, g):
            ax.plot([r["nfl_q10"], r["nfl_q90"]], [yi, yi], color=INK["grid"], lw=10, solid_capstyle="butt")
            ax.plot([r["nfl_q50"]] * 2, [yi - 0.18, yi + 0.18], color=INK["secondary"], lw=2)
            se = r.get("standard_error")
            if se:
                ax.plot([r["combine_value"] - 1.96 * se, r["combine_value"] + 1.96 * se], [yi, yi], color=A_COL, lw=2)
            ax.plot(r["combine_value"], yi, "o", color=A_COL, ms=8, mec="white")
            cov = "abstains" if r.get("coverage") is None else f"coverage {r['coverage']:.0f}"
            ax.annotate(f"{cov} · {r['state']}", (1.01, yi), xycoords=("axes fraction", "data"), fontsize=8, color=INK["muted"], va="center")
        ax.set_yticks(y, [f"{r['demand_id']}\n{r['dimension']}" for r in g], fontsize=8)
        ax.set_ylim(-0.7, len(g) - 0.3)
        ax.set_xlabel(f"[{u}]")
    finish(fig, "Grey bar = 10th–90th percentile of NFL episodes of the position, tick = NFL median; dot = the prospect's Combine value with its 95 % repeat-attempt "
                "interval. Maximal-effort test against observed game movement: descriptive, not a prediction.", f"Prospect-to-demand physical alignment — {name}")
    return fig


def fig_curve_alignment(combine: tuple[np.ndarray, np.ndarray], nfl: tuple[np.ndarray, np.ndarray], lo: float, hi: float, labels: tuple[str, str]) -> Any:
    """A Combine trace and an NFL episode through the SAME speed band, both clocks set to 0 at the first crossing of ``lo``."""
    fig, ax = figure(1, 1, size="panel")
    ax = ax[0, 0]
    drawn = 0
    for (t, v), col, lab in zip((combine, nfl), (A_COL, B_COL), labels):
        t0 = md.crossing_time(t, v, lo)
        if not np.isfinite(t0):
            continue
        ax.plot(t - t0, v, color=col, lw=2, label=f"{lab} — {md.band_accel(t, v, lo, hi):.2f} yd/s² through the band")
        drawn += 1
    if not drawn:
        empty_panel(ax, "the band crossing is not recorded in either trace")
        return fig
    ax.axhspan(lo, hi, color=GOLD, alpha=0.12)
    ax.axvline(0, color=INK["axis"], lw=1)
    ax.set_xlabel(f"time since speed first reached {lo:g} yd/s [s]")
    ax.set_ylabel("speed [yd/s]")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, "Same unit, same 10 Hz sampling, same first-crossing rule. The forty is a maximal effort from rest; the NFL episode is whatever the play required.",
           f"Curve alignment through the {lo:g} → {hi:g} yd/s band")
    return fig


# ============================================================================== 09C — stopwatch twins
def fig_twin_summary(summary: pl.DataFrame) -> Any:
    g = summary.filter((pl.col("position_group") == md.ALL) & pl.col("share_distinguishable").is_not_null()).sort("share_distinguishable")
    fig, ax = figure(1, 1, size="panel")
    ax = ax[0, 0]
    if g.is_empty():
        empty_panel(ax, "no twin pair could be tested")
        return fig
    ax.barh(g["feature"].to_list(), 100 * g["share_distinguishable"].to_numpy(), color=SEQUENTIAL_BLUE[3], height=0.6)
    ax.axvline(5, color=STATUS["critical"], lw=1.5, ls="--")
    ax.text(5.5, -0.45, "5 % expected by chance", color=INK["secondary"], fontsize=8, va="top")
    for i, r in enumerate(g.iter_rows(named=True)):
        ax.text(100 * r["share_distinguishable"] + 0.8, i, f"{100 * r['share_distinguishable']:.0f} % of {r['n_tested']:,}", va="center", fontsize=8)
    ax.set_xlabel("stopwatch-twin pairs with a distinguishable difference [%]")
    finish(fig, "Pairs share athletes, so this is a share and not a test. A feature near the dashed line separates twins no more often than noise would.",
           "Same stopwatch — how often is the movement measurably different?")
    return fig


def _forest_z(ax, diffs: Sequence[dict[str, Any]]) -> None:
    g = [d for d in diffs if d.get("z") is not None]
    if not g:
        empty_panel(ax, "no feature with an error estimate")
        return
    y = np.arange(len(g))[::-1]
    ax.axvspan(-1.96, 1.96, color=INK["grid"], alpha=0.8)
    ax.axvline(0, color=INK["axis"], lw=1)
    for yi, d in zip(y, g):
        c = VERDICT_COLOR.get(d["verdict"], INK["muted"])
        ax.plot([d["z"] - 1.96, d["z"] + 1.96], [yi, yi], color=c, lw=2)
        ax.plot(d["z"], yi, "o", color=c, ms=7)
        ax.annotate(f"{d['difference']:+.2f} {d['unit']}", (1.01, yi), xycoords=("axes fraction", "data"), fontsize=8, color=INK["muted"], va="center")
    ax.set_yticks(y, [d["label"] for d in g], fontsize=8)
    ax.set_ylim(-0.7, len(g) - 0.3)
    ax.set_xlabel("difference (B − A) in standard errors of the difference")


def fig_twin_story(twin: dict[str, Any], curves: dict[str, list[tuple[np.ndarray, np.ndarray]]], band: dict[str, np.ndarray] | None,
                   diffs: Sequence[dict[str, Any]]) -> Any:
    """The signature figure: what the stopwatch sees · what tracking sees · whether the difference exceeds its uncertainty."""
    fig, axes = figure(1, 3, size=(16.0, 5.6), gridspec_kw={"width_ratios": [0.7, 1.5, 1.2]})
    a, b = twin["name_a"], twin["name_b"]
    ax = axes[0, 0]
    ax.barh([b, a], [twin["forty_b"], twin["forty_a"]], color=[B_COL, A_COL], height=0.5)
    for i, (v, n) in enumerate(((twin["forty_b"], b), (twin["forty_a"], a))):
        ax.text(v - 0.02, i, f"{v:.2f} s", ha="right", va="center", color="white", fontsize=11, fontweight="semibold")
    ax.set_xlim(min(twin["forty_a"], twin["forty_b"]) - 0.6, max(twin["forty_a"], twin["forty_b"]) + 0.05)
    ax.set_title("The stopwatch", fontsize=11, loc="left")
    ax.set_xlabel("official forty [s]")
    ax = axes[0, 1]
    if band is not None and np.isfinite(band["q50"]).any():
        ax.fill_between(band["grid"], band["q10"], band["q90"], color=INK["grid"], alpha=0.9, label=f"{twin['position_group']} peers, 10–90 % ({band['n']} runs)")
    for name, col in ((a, A_COL), (b, B_COL)):
        for i, (d, s) in enumerate(curves.get(name, [])):
            ax.plot(d, s, color=col, lw=2.2 if i == 0 else 1.2, alpha=1.0 if i == 0 else 0.55, label=name if i == 0 else None)
    ax.set_title("The tracking: speed by distance (every recorded attempt)", fontsize=11, loc="left")
    ax.set_xlabel("distance covered [yd]")
    ax.set_ylabel("tracked speed [yd/s]")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _forest_z(axes[0, 2], diffs)
    axes[0, 2].set_title("Is the difference larger than its uncertainty?", fontsize=11, loc="left")
    finish(fig, "Right: grey band = ±1.96 standard errors of the difference (repeat-attempt variation of both athletes); green = distinguishable. "
                "An observed Combine difference is not demonstrated to cause or predict a football outcome.", f"Same stopwatch, different movement — {a} and {b} ({twin['position_group']})")
    return fig


def fig_accel_speed(traces_by_name: dict[str, list[tuple[np.ndarray, np.ndarray]]], title: str) -> Any:
    """Acceleration against speed (interior frames) — the profile behind a speed curve."""
    fig, ax = figure(1, 1, size="panel")
    ax = ax[0, 0]
    if not any(traces_by_name.values()):
        empty_panel(ax, "no recorded attempt")
        return fig
    for (name, ts), col in zip(traces_by_name.items(), (A_COL, B_COL)):
        for i, (v, acc) in enumerate(ts):
            ax.plot(v, acc, color=col, lw=2 if i == 0 else 1.1, alpha=1.0 if i == 0 else 0.55, label=name if i == 0 else None)
    for lo, hi in md.BANDS:
        ax.axvspan(lo, hi, color=GOLD, alpha=0.06)
    ax.axhline(0, color=INK["axis"], lw=1)
    ax.set_xlabel("tracked speed [yd/s]")
    ax.set_ylabel("tangential acceleration [yd/s²]")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "Shaded = the 2–4, 4–6 and 6–8 yd/s bands. Second derivative of a 10 Hz position: read the level of a band, not single frames.", title)
    return fig


def fig_band_time(diffs: Sequence[dict[str, Any]], names: tuple[str, str]) -> Any:
    """Time spent in each speed band (band width / band acceleration) and the time B gains or loses on A."""
    fig, ax = figure(1, 1, size="panel")
    ax = ax[0, 0]
    rows = []
    for (lo, hi), col in zip(md.BANDS, md.BAND_COLS):
        d = next((x for x in diffs if x["feature"] == col), None)
        if d and d.get("value_a") and d.get("value_b"):
            rows.append((f"{lo:g} → {hi:g} yd/s", (hi - lo) / d["value_a"], (hi - lo) / d["value_b"], d["verdict"]))
    if not rows:
        empty_panel(ax, "no band crossing recorded for both athletes")
        return fig
    x, w = np.arange(len(rows)), 0.36
    ax.bar(x - w / 2 - 0.01, [r[1] for r in rows], w, color=A_COL, label=names[0])
    ax.bar(x + w / 2 + 0.01, [r[2] for r in rows], w, color=B_COL, label=names[1])
    for xi, r in zip(x, rows):
        ax.text(xi, max(r[1], r[2]) + 0.01, f"{r[2] - r[1]:+.3f} s\n{r[3].replace('_', ' ')}", ha="center", va="bottom", fontsize=8, color=INK["secondary"])
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("time in the band [s]")
    ax.set_ylim(0, max(max(r[1], r[2]) for r in rows) * 1.35)
    ax.legend(frameon=False, fontsize=8)
    finish(fig, f"Label = time {names[1]} spends in the band minus {names[0]} (negative = quicker through it), with the verdict on the band acceleration.",
           "Time gained or lost through each speed band")
    return fig


# ============================================================================== 09D / 09G — the one-page card
def fig_intelligence_card(profile: dict[str, Any], metrics: Sequence[dict[str, Any]], twins: Sequence[dict[str, Any]], matrix: Sequence[dict[str, Any]],
                          curves: list[tuple[np.ndarray, np.ndarray]], band: dict[str, np.ndarray] | None, sections: dict[str, list[str]]) -> Any:
    """One-page scout-ready athlete intelligence card (portrait). ``sections``: ``scouting_interpretation.prospect_report``."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 11.0))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.5, 1.5, 1.3, 1.5], hspace=0.42, wspace=0.28, left=0.07, right=0.95, top=0.95, bottom=0.07)
    name, pos = profile.get("display_name") or str(profile.get("nfl_id")), profile.get("position_group")
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.add_patch(plt.Rectangle((0, 0), 1, 1, transform=head.transAxes, color=NAVY))
    head.text(0.02, 0.62, name, color="white", fontsize=20, fontweight="bold", transform=head.transAxes)
    head.text(0.02, 0.22, f"{pos} · draft class {profile.get('draft_year') or 'n/a'} · forty {_v(profile.get('official_forty'), 's')} · 10-yard {_v(profile.get('official_ten'), 's')} · "
              f"{_v(profile.get('weight'), 'lb', 0)} · {_v(profile.get('height'), 'in', 1)}", color="white", fontsize=9, transform=head.transAxes)
    head.text(0.98, 0.62, "COMBINE INTELLIGENCE LAB", color=GOLD, fontsize=9, ha="right", transform=head.transAxes)
    head.text(0.98, 0.22, f"recording evidence: {profile.get('evidence_tier') or 'n/a'}", color="white", fontsize=9, ha="right", transform=head.transAxes)
    ax = fig.add_subplot(gs[1, 0])
    if band is not None and np.isfinite(band["q50"]).any():
        ax.fill_between(band["grid"], band["q10"], band["q90"], color=INK["grid"], alpha=0.9)
    for i, (d, s) in enumerate(curves):
        ax.plot(d, s, color=A_COL, lw=2 if i == 0 else 1.1, alpha=1.0 if i == 0 else 0.55)
    ax.set_title(f"Speed by distance vs {pos} peers (10–90 %)", fontsize=9, loc="left")
    ax.set_xlabel("distance [yd]", fontsize=8)
    ax.set_ylabel("speed [yd/s]", fontsize=8)
    if not curves:
        empty_panel(ax, "no tracked attempt")
    ax = fig.add_subplot(gs[1, 1])
    m = [r for r in metrics if r.get("pct_position") is not None]
    if m:
        y = np.arange(len(m))[::-1]
        ax.barh(y, [r["pct_position"] for r in m], color=SEQUENTIAL_BLUE[3], height=0.55)
        for yi, r in zip(y, m):
            se = "" if r.get("standard_error") is None else f" ± {1.96 * r['standard_error']:.2f}"
            ax.text(101, yi, f"{r['value']:.2f}{se} {r['unit']}", va="center", fontsize=7)
        ax.set_yticks(y, [r["label"] for r in m], fontsize=7)
        ax.set_xlim(0, 100)
        ax.axvline(50, color=INK["axis"], lw=1)
        ax.set_xlabel("raw position percentile of the value (times: lower = quicker)", fontsize=7)
    else:
        empty_panel(ax, "no tracked measure")
    ax.set_title("Tracked measures (value ± 95 % repeat-attempt interval)", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[2, 0])
    s = [r for r in matrix if r.get("state") == "supported"]
    if s:
        y = np.arange(len(s))[::-1]
        ax.barh(y, [r["mean_coverage"] for r in s], color=SEQUENTIAL_BLUE[3], height=0.5)
        for yi, r in zip(y, s):
            ax.plot([r["coverage_lo"], r["coverage_hi"]], [yi, yi], color=INK["primary"], lw=1.5)
        ax.set_yticks(y, [r["demand_id"] for r in s], fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of NFL episodes at or below the Combine value", fontsize=7)
    else:
        empty_panel(ax, "no demand correspondence")
    ax.set_title("NFL movement-demand correspondence (descriptive)", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    ax.set_title("Stopwatch twins", fontsize=9, loc="left")
    lines = [f"• {t['other_name']} — forty {abs(t['d_forty']):.2f} s apart, {t['n_distinguishable']} distinguishable feature(s)" for t in twins[:6]] or ["none within the caliper"]
    ax.text(0, 0.98, "\n".join(textwrap.fill(x, 46) for x in lines), va="top", fontsize=8, transform=ax.transAxes)
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    body = []
    for key in ("Suggested film-review checklist", "Suggested follow-up testing"):
        body.append(key.upper())
        body += [textwrap.fill("• " + x, 118, subsequent_indent="  ") for x in sections.get(key, [])]
        body.append("")
    ax.text(0, 1.0, "\n".join(body), va="top", fontsize=7.5, transform=ax.transAxes)
    fig.text(0.07, 0.025, textwrap.fill(DISCLAIMER, 150), fontsize=6.5, color=INK["secondary"], va="bottom")
    return fig


def _v(x: Any, unit: str, d: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{d}f} {unit}"


__all__ = ["A_COL", "B_COL", "fig_accel_speed", "fig_alignment", "fig_athlete_share", "fig_band_time", "fig_curve_alignment", "fig_demand_frequency",
           "fig_demand_profiles", "fig_examples", "fig_feature_evidence", "fig_intelligence_card", "fig_similarity_heatmap", "fig_twin_story", "fig_twin_summary",
           "traces"]
