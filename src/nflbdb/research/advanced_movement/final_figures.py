"""Milestone 10 — final publication figures (matplotlib). Only the figures Milestone 09 does not already draw are new here;
the submission set reuses ``intelligence_figures`` for the hero twin story, the twin-distinguishability bars, the
reliability-versus-redundancy scatter, the demand frequency map, the matching heatmap and the one-page card.

    fig_phase_decomposition   sprint-phase differences of one pair (z units, native difference annotated)
    fig_same_finish           distance and speed against time for two athletes with the same official forty
    fig_movement_gap          position × demand matrix coloured by what forty tracking can describe
    fig_gap_position          one position's observed demand rates, coloured by coverage class
    fig_decision_workflow     the scouting decision workflow (diagram)
    fig_contribution          the final contribution diagram (what is shown, what is not)

Every figure is drawn from mart rows passed in; an empty input gives an explicit "no data" panel. Names: ``AM10<nb>_<nn>_<slug>``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...data.reporting import CATEGORICAL, INK, STATUS
from ..figkit import empty_panel, figure, finish
from . import movement_gap as mg
from .intelligence_figures import VERDICT_COLOR

A_COL, B_COL = CATEGORICAL[0], CATEGORICAL[1]
FIELD, NAVY = "#1f6f43", "#0b1f3a"
CLASS_COLOR = {mg.DIRECT: FIELD, mg.PARTIAL: "#8cc7a1", mg.NOT_MEASURED: "#d95926", mg.UNAVAILABLE: "#c9ced6"}
CLASS_MARK = {mg.DIRECT: "D", mg.PARTIAL: "P", mg.NOT_MEASURED: "×", mg.UNAVAILABLE: "n/a"}


# ============================================================================== 10B — pair stories
def fig_phase_decomposition(decomp: Sequence[dict[str, Any]], names: tuple[str, str], title: str | None = None) -> Any:
    """One row per feature, grouped by sprint phase; a phase without a recorded feature is shown as 'not identified'."""
    fig, ax = figure(1, 1, size=(11.0, 6.4))
    ax = ax[0, 0]
    rows: list[tuple[str, dict[str, Any] | None, str]] = []
    for p in decomp:
        if not p["features"]:
            rows.append((p["phase"], None, "not identified in both recordings"))
        for r in p["features"]:
            rows.append((p["phase"], r, r["label"]))
    if not rows:
        empty_panel(ax, "no phase recorded for both athletes")
        return fig
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-1.96, 1.96, color=INK["grid"], alpha=0.8)
    ax.axvline(0, color=INK["axis"], lw=1)
    last = None
    for yi, (phase, r, lab) in zip(y, rows):
        if phase != last:
            ax.text(-0.02, yi + 0.42, phase.upper(), transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color=FIELD, fontweight="semibold")
            last = phase
        if r is None or r.get("z") is None:
            ax.text(0, yi, "not identified" if r is None else "no error estimate", ha="center", va="center", fontsize=8, color=INK["muted"], style="italic")
            continue
        c = VERDICT_COLOR.get(r["verdict"], INK["muted"]) if r["usable"] else STATUS["warning"]
        ax.plot([r["z"] - 1.96, r["z"] + 1.96], [yi, yi], color=c, lw=2.2)
        ax.plot(r["z"], yi, "o", color=c, ms=8)
        ax.annotate(f"{r['value_a']:.2f} → {r['value_b']:.2f} {r['unit']}  ({r['difference']:+.2f})", (1.01, yi), xycoords=("axes fraction", "data"), fontsize=8,
                    color=INK["secondary"], va="center")
    ax.set_yticks(y, [lab for _, _, lab in rows], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(f"difference ({names[1]} − {names[0]}) in standard errors of the difference")
    finish(fig, "Grey band = inside ±1.96 standard errors (repeat-attempt variation of both athletes): not a difference. Green = distinguishable; amber = feature "
                "not repeatable enough to describe an athlete. Right: value of A → value of B (difference, native units). No phase is filled from another.",
           title or f"Where the sprints differ — phase by phase ({names[0]} vs {names[1]})")
    return fig


def fig_same_finish(twin: dict[str, Any], curves_t: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> Any:
    """Distance and speed against time from the first recorded frame (every attempt; first in bold). Recordings are drawn as
    recorded: no point is added before the first or after the last frame."""
    fig, axes = figure(1, 2, size=(14.0, 5.4))
    a, b = twin["name_a"], twin["name_b"]
    if not any(curves_t.values()):
        empty_panel(axes[0, 0], "no recorded attempt")
        return fig
    for name, col in ((a, A_COL), (b, B_COL)):
        for i, (t, d, v) in enumerate(curves_t.get(name, [])):
            t0 = t - t[0]
            kw = dict(color=col, lw=2.4 if i == 0 else 1.1, alpha=1.0 if i == 0 else 0.5, label=name if i == 0 else None)
            axes[0, 0].plot(t0, d - d[0], **kw)
            axes[0, 1].plot(t0, v, **kw)
    axes[0, 0].axhline(40, color=INK["axis"], ls="--", lw=1)
    axes[0, 0].text(0.02, 40.6, "40 yd", color=INK["secondary"], fontsize=8, transform=axes[0, 0].get_yaxis_transform())
    axes[0, 0].set_xlabel("time from the first recorded frame [s]")
    axes[0, 0].set_ylabel("distance covered in the recording [yd]")
    axes[0, 0].set_title(f"Official forty: {twin['forty_a']:.2f} s and {twin['forty_b']:.2f} s", loc="left", fontsize=11)
    axes[0, 1].set_xlabel("time from the first recorded frame [s]")
    axes[0, 1].set_ylabel("tracked speed [yd/s]")
    axes[0, 1].set_title("How each built the speed", loc="left", fontsize=11)
    for ax in axes[0]:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, "Recorded 10 Hz frames only; a recording can start after the first movement or end before 40 yd, so the official time — not the recorded "
                "curve — is the timed result. The same final time can be reached through different speed curves.",
           f"Same forty, different curve — {a} and {b} ({twin['position_group']})")
    return fig


# ============================================================================== 10C — movement gap
def fig_movement_gap(matrix: Sequence[dict[str, Any]]) -> Any:
    import matplotlib.patches as mpatches

    fig, ax = figure(1, 1, size=(13.0, 6.6))
    ax = ax[0, 0]
    if not matrix:
        empty_panel(ax, "no movement-gap rows")
        return fig
    pos = sorted({r["position_group"] for r in matrix})
    dem = list(dict.fromkeys(r["demand_id"] for r in matrix))
    lab = {r["demand_id"]: r["label"] for r in matrix}
    cell = {(r["position_group"], r["demand_id"]): r for r in matrix}
    for i, d in enumerate(dem):
        for j, p in enumerate(pos):
            r = cell.get((p, d))
            if r is None:
                continue
            c = r["coverage_class"]
            ax.add_patch(mpatches.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fc=CLASS_COLOR[c], hatch="//" if c == mg.UNAVAILABLE else None, ec="white", lw=1))
            rate = "too few" if not r["supported"] else f"{float(r['episodes_per_100_plays'] or 0):.1f}"
            dark = c in (mg.DIRECT, mg.NOT_MEASURED)
            ax.text(j, i, f"{CLASS_MARK[c]}  {rate}\n{r['n_athletes']} ath.", ha="center", va="center", fontsize=8, color="white" if dark else INK["primary"])
    ax.set_xlim(-0.5, len(pos) - 0.5)
    ax.set_ylim(len(dem) - 0.5, -0.5)
    ax.set_xticks(range(len(pos)), pos)
    ax.set_yticks(range(len(dem)), [lab[d] for d in dem], fontsize=9)
    ax.grid(False)
    handles = [mpatches.Patch(color=CLASS_COLOR[k], label=f"{CLASS_MARK[k]} = {mg.COVERAGE_CLASSES[k]}") for k in mg.CLASS_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    finish(fig, "Cell = episodes per 100 eligible player-plays (05A denominator) and contributing athletes. Class derived from the 09A registry analogue, the 09A support "
                "rule and the 09B matching summary. Categories overlap; frequency is observed exposure, not importance to performance.",
           "The NFL movement gap — what games ask for, and what the forty can describe")
    return fig


def fig_gap_position(matrix: Sequence[dict[str, Any]], position: str) -> Any:
    fig, ax = figure(1, 1, size=(11.0, 5.6))
    ax = ax[0, 0]
    rows = sorted((r for r in matrix if r["position_group"] == position and r["supported"]), key=lambda r: float(r["episodes_per_100_plays"] or 0))
    if not rows:
        empty_panel(ax, f"no supported demand for {position}")
        return fig
    y = np.arange(len(rows))
    ax.barh(y, [float(r["episodes_per_100_plays"] or 0) for r in rows], color=[CLASS_COLOR[r["coverage_class"]] for r in rows], height=0.62)
    for yi, r in zip(y, rows):
        ax.text(float(r["episodes_per_100_plays"] or 0) + 0.8, yi, f"{float(r['episodes_per_100_plays'] or 0):.1f}  ·  {CLASS_MARK[r['coverage_class']]} "
                f"{mg.COVERAGE_CLASSES[r['coverage_class']].lower()}  ·  {r['n_episodes']:,} ep., {r['n_athletes']} ath.", va="center", fontsize=8, color=INK["secondary"])
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=9)
    ax.set_xlabel("episodes per 100 eligible player-plays")
    ax.set_xlim(0, max(float(r["episodes_per_100_plays"] or 0) for r in rows) * 1.75)
    finish(fig, "Observed exposure, not a requirement. Bars in orange are demands the forty-yard dash contains no physical analogue of.",
           f"{position}: observed NFL movement demands and what the forty describes")
    return fig


# ============================================================================== 10D / 10E — diagrams
def _boxes(ax, items: Sequence[tuple[str, str]], y: float, color: str, h: float = 0.30) -> None:
    import matplotlib.patches as mp

    n = len(items)
    w = 0.92 / n
    for k, (head, body) in enumerate(items):
        x = 0.04 + k * w
        ax.add_patch(mp.FancyBboxPatch((x, y), w - 0.02, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=color, ec="none", transform=ax.transAxes))
        ax.text(x + 0.01, y + h - 0.05, head, transform=ax.transAxes, fontsize=10, fontweight="semibold", color="white", va="top")
        ax.text(x + 0.01, y + h - 0.11, body, transform=ax.transAxes, fontsize=8, color="white", va="top", wrap=True)
        if k < n - 1:
            ax.annotate("", xy=(x + w - 0.005, y + h / 2), xytext=(x + w - 0.025, y + h / 2), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=INK["axis"], lw=1.4))


def _wrap(s: str, n: int = 24) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(s, n))


def fig_decision_workflow() -> Any:
    fig, ax = figure(1, 1, size=(15.0, 4.6))
    ax = ax[0, 0]
    ax.axis("off")
    steps = [("1 · Question", "similar forties? repeatable? demand match? not measured? next test?"),
             ("2 · Prospects", "one or two; comparable athletes suggested from the stored twin rule"),
             ("3 · Evidence", "official tests, tracked values with uncertainty, position reference, NFL demand rates"),
             ("4 · Action", "observation · confidence · football relevance · film · testing · cannot conclude"),
             ("5 · Export", "scout report · position-coach report · general-manager summary (HTML / PDF)")]
    _boxes(ax, [(h, _wrap(b)) for h, b in steps], 0.40, NAVY, 0.45)
    labels = ["measured difference", "uncertain difference", "missing observation", "measurement mismatch", "additional data needed"]
    for k, lab in enumerate(labels):
        ax.text(0.04 + k * 0.184, 0.24, f"■ {lab}", transform=ax.transAxes, fontsize=9, color=[FIELD, STATUS["warning"], INK["muted"], "#d95926", A_COL][k])
    ax.text(0.04, 0.10, "Every line comes from a deterministic rule on stored values. No success probability, no ranking, no draft label.",
            transform=ax.transAxes, fontsize=9, color=INK["secondary"])
    finish(fig, "A simulator of the scouting review workflow — not of a draft or a game.", "Scouting Decision Simulator — from a question to an evidence-aware action")
    return fig


def fig_contribution(numbers: dict[str, Any]) -> Any:
    """``numbers``: the mart overview (n_tracked_attempts, n_twin_pairs, n_twin_pairs_distinguishable, n_nfl_episodes)."""
    fig, ax = figure(1, 1, size=(15.0, 6.0))
    ax = ax[0, 0]
    ax.axis("off")
    g = lambda k: numbers.get(k)
    shown = [("Measured", f"{g('n_tracked_attempts') or 0:,} tracked 10 Hz forty attempts of {g('n_tracked_prospects') or 0:,} prospects"),
             ("Repeatable", "top speed and speed development repeat between attempts (08B.6 C01, C02, C17)"),
             ("Beyond the stopwatch", f"{g('n_twin_pairs_distinguishable') or 0:,} of {g('n_twin_pairs') or 0:,} stopwatch twins differ on ≥ 1 feature"),
             ("Football context", f"{g('n_nfl_episodes') or 0:,} NFL movement episodes in {g('n_demands_measurable') or 0} demand categories"),
             ("Scouting action", "film prompts, targeted tests, and what cannot be concluded")]
    _boxes(ax, [(h, _wrap(b)) for h, b in shown], 0.52, FIELD, 0.40)
    _boxes(ax, [("Not established", _wrap("incremental NFL prediction from tracked Combine features (Milestones 06, 08B: intervals include 0)", 60)),
                ("Not claimed", _wrap("grades, rankings, draft value, success probabilities, causal effects of training", 60))], 0.08, "#7a2e14", 0.34)
    finish(fig, "Contribution: an evidence-aware movement intelligence layer on top of the stopwatch — reproducible from the Milestone 09 application mart.",
           "Same stopwatch. Different movement. Better scouting questions.")
    return fig


__all__ = ["CLASS_COLOR", "CLASS_MARK", "fig_contribution", "fig_decision_workflow", "fig_gap_position", "fig_movement_gap", "fig_phase_decomposition",
           "fig_same_finish"]
