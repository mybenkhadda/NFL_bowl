"""M15–M18 research roadmap figure (16:9, publication quality).

Authored statically on 2026-10-20 and NEVER executed by the author: run it in Colab and look at the output
before using it. It needs matplotlib only.

Usage (Colab, from the repository root):

    !python scripts/figures/m15_m18_roadmap_figure.py --out figures/roadmap

or, from a notebook cell (``scripts`` is not a package, so run the file rather than importing it):

    %run scripts/figures/m15_m18_roadmap_figure.py --out figures/roadmap

Outputs:
    figures/roadmap/M15_M18_RESEARCH_ROADMAP.{png,svg,pdf}
    figures/roadmap/M15_M18_RESEARCH_ROADMAP__data.json     (the figure's text content = its source data)

Evidence labels follow docs/research/M15_M18_MASTER_RESEARCH_ROADMAP.md § 0:
[OBS] observed in an executed, protocol-conforming notebook · [EXP] exploratory, needs replication.
Every number on the figure is copied from that document; nothing here is a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# ---------------------------------------------------------------- palette (dataviz reference instance, light mode)
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#c3c2b7"
NEUTRAL = "#f0efec"
# milestone identity = categorical slots 1–4 in fixed order (blue, orange, aqua, yellow); always paired with a text label
MILESTONE_COLOR = {"M15": "#2a78d6", "M16": "#eb6834", "M17": "#1baf7a", "M18": "#eda100"}

FONT = {"family": "sans-serif", "sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]}

# ---------------------------------------------------------------- content (the figure's source data)
TITLE = "M15–M18 research roadmap"
SUBTITLE = ("Does precise Combine movement tracking add reliable, independently predictive information about NFL movement "
            "beyond official Combine testing?")

EVIDENCE = [
    ("OBS", "M12: cross-domain retrieval ≈ chance\n(AUC ≈ 0.50); domain AUC ≈ 0.997"),
    ("OBS", "M06 / M08B: no validated incremental\nNFL prediction beyond M1"),
    ("EXP", "M14: 1 of 30 metric instances passed\nmeasurement validation"),
    ("EXP", "Combine vs NFL windows differ: 62 vs 43\nvalid frames; NFL windows demand-selected"),
    ("EXP", "STFT aligned top-1 8.75 % (7/80, one split)\n≈ position-only chance (~6 %)"),
    ("EXP", "Position-only beat M1 + tracking\n(ΔRMSE −0.020 yd/s, CI includes 0)"),
]

MILESTONES = {
    "M15": "M15  Spectral & temporal\nrepresentation science",
    "M16": "M16  Verified baselines &\ncross-domain validation",
    "M17": "M17  Context-aware\nathletic capability",
    "M18": "M18  NFL transfer &\nscouting value",
}

LANES = {
    "P1": "P1  Representations",
    "P2": "P2  Baselines & targets",
    "P3": "P3  Event measurement",
}

CELLS = {
    ("P1", "M15"): "STFT · LSTM/GRU · TCN vs PCA (equal\nbudget); valid-frame preservation;\nlatent ICC; position-aware chance",
    ("P1", "M16"): "5-fold retrieval: all / same-position /\nstopwatch-twin pools; CEM context\nmatching; leakage audit",
    ("P1", "M17"): "Fusion; context-conditioned alignment\n(Procrustes / CORAL first);\nalignment damage check",
    ("P2", "M15"): "Spectral resolution-artifact test;\nlength- and context-matched\nCombine vs NFL contrasts",
    ("P2", "M16"): "M1 contract, 1e-8 reproducibility;\nNFL targets T0–T6: reliability,\nexposure, MDE; learner ladder",
    ("P2", "M17"): "Mixed-effects variance components;\nshrunk athlete effects;\nCombine ↔ NFL disattenuated r",
    ("P3", "M15"): "Gap / validity diagnostics; M14\ndetector scores event preservation\nof reconstructions",
    ("P3", "M16"): "M1 + M14-reliable rung\n(train-fold quality selection)",
    ("P3", "M17"): "M14 failure diagnosis → drill-specific,\ncensoring-aware events;\nentry-speed / turn strata",
}

M18_TEXT = ("Ladder on one cohort, folds, target\n"
            "B0 position · B1 M1\n"
            "B2 +simple · B3 +biomechanics\n"
            "B4 +STFT · B5 +LSTM\n"
            "B6 +fusion · B7 +signatures\n"
            "\n"
            "Within-position ranking\n"
            "Drill → task specificity matrix\n"
            "Stopwatch-twin cases\n"
            "(selected without outcomes)\n"
            "Negative controls · permutation null")

GATES = {
    "G15": "G15  preservation ≥ PCA · latent\nICC ≥ 0.4 · no collapse · seeds",
    "G16": "G16  target reliable · M1 reproduced ·\nincrement or precise null · D: similarity",
    "G17": "G17  ICC after context · athlete share ·\nnovelty · held-out · NFL relevance",
    "G18": "G18  validated vs M1 (Holm) · ≥ 2 % ·\ncontrols pass · permutation p",
}

ALTERNATIVES = {
    "M15": "PCA · handcrafted Welch features\n(deep model only if better on\npreservation AND repeatability)",
    "M16": "Ridge / Elastic Net · Procrustes\n(trees only if ΔRMSE vs ridge > 0)",
    "M17": "Mixed-effects athlete model\nbefore neural; concat before\nattention fusion",
    "M18": "Partial pooling before separate\nposition models",
}

ARTIFACTS = {
    "M15": "embeddings/inductive · checkpoints\nspectral & preservation tables",
    "M16": "evaluation_registry · oof_long\ntarget_freeze_m16.json · MDE",
    "M17": "athlete_signatures · variance\ncomponents · model cards",
    "M18": "evidence registry · final report\nexecutive summary · figures",
}

SCOUTING = ("Scouting question\n"
            "Should tracking change how we\n"
            "rank two prospects with the\n"
            "same 40 time?\n"
            "\n"
            "Possible verdicts\n"
            "• scouting-relevant (G18 1–6)\n"
            "• promising — replicate\n"
            "• precise null + MDE\n"
            "\n"
            "Always: replicate on new\n"
            "draft classes. No injury,\n"
            "force or causal claims.")

PIVOT = "Any gate fails →\ndocumented negative result;\ndescriptive profiles only"
OUTPUTS = "PNG · SVG · PDF + source data\nParquet · CSV · JSON manifests"

# ---------------------------------------------------------------- layout (data coordinates: 16 × 9 = figure inches)
COLS = {"M15": (3.75, 2.3), "M16": (6.45, 2.3), "M17": (9.15, 2.3), "M18": (11.85, 1.95)}   # x0, width
GATE_X = {"G15": 6.25, "G16": 8.95, "G17": 11.65, "G18": 13.95}
HEADER = (7.3, 0.6)                                                   # y0, height
LANE_Y = {"P1": (6.2, 0.95), "P2": (5.05, 0.95), "P3": (3.9, 0.95)}
GATE_ROW = (3.15, 0.6)
ALT_ROW = (2.1, 0.85)
ART_ROW = (1.05, 0.85)
LEFT = (0.3, 3.1)                                                     # x0, width
RIGHT = (14.25, 1.5)


def _box(ax: Any, x: float, y: float, w: float, h: float, text: str = "", fs: float = 7.0, stripe: str | None = None,
         ls: str = "-", fill: str = SURFACE, edge: str = HAIRLINE, weight: str = "normal", ha: str = "left",
         color: str = INK, lw: float = 0.8) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", facecolor=fill, edgecolor=edge,
                                linewidth=lw, linestyle=ls, zorder=2))
    pad = 0.08
    if stripe is not None:
        ax.add_patch(FancyBboxPatch((x, y), 0.06, h, boxstyle="round,pad=0,rounding_size=0.03", facecolor=stripe,
                                    edgecolor="none", zorder=3))
        pad = 0.14
    if text:
        tx = x + pad if ha == "left" else x + w / 2
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=color, fontweight=weight, linespacing=1.25,
                zorder=4)


def _diamond(ax: Any, cx: float, cy: float, r: float, label: str) -> None:
    ax.add_patch(Polygon([(cx, cy + r), (cx + r * 0.8, cy), (cx, cy - r), (cx - r * 0.8, cy)], closed=True, facecolor=NEUTRAL,
                         edgecolor=INK_2, linewidth=1.0, zorder=5))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=6.5, color=INK, fontweight="bold", zorder=6)


def _arrow(ax: Any, x0: float, y0: float, x1: float, y1: float, color: str = MUTED, lw: float = 1.0, ls: str = "-") -> None:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=color, linewidth=lw,
                                 linestyle=ls, zorder=1, shrinkA=0, shrinkB=0))


def build_figure() -> tuple[Any, dict[str, Any]]:
    """Return (figure, source-data dict). Draws nothing to disk."""
    plt.rc("font", **FONT)
    fig = plt.figure(figsize=(16, 9), facecolor=PAGE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(PAGE)

    # title
    ax.text(0.3, 8.6, TITLE, ha="left", va="center", fontsize=17, color=INK, fontweight="bold")
    ax.text(0.3, 8.2, SUBTITLE, ha="left", va="center", fontsize=9.5, color=INK_2)

    # left column: evidence
    lx, lw = LEFT
    ax.text(lx, HEADER[0] + HEADER[1] / 2, "Evidence from M12–M15", ha="left", va="center", fontsize=10, color=INK, fontweight="bold")
    top, h, gap = 7.2, 0.5, 0.05
    for i, (tag, txt) in enumerate(EVIDENCE):
        y = top - (i + 1) * h - i * gap
        _box(ax, lx, y, lw, h, f"[{tag}]  {txt}", fs=6.6, ls="-" if tag == "OBS" else "--")

    # milestone headers and gate diamonds
    hy, hh = HEADER
    for m, (x, w) in COLS.items():
        _box(ax, x, hy, w, hh, MILESTONES[m], fs=8.2, weight="bold", ha="center")
        ax.add_patch(FancyBboxPatch((x, hy + hh - 0.07), w, 0.07, boxstyle="square,pad=0", facecolor=MILESTONE_COLOR[m],
                                    edgecolor="none", zorder=3))
    for g, gx in GATE_X.items():
        ax.plot([gx, gx], [GATE_ROW[0] + GATE_ROW[1], hy + hh], color=HAIRLINE, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
        _diamond(ax, gx, hy + hh / 2, 0.24, g)

    # lanes
    for lane, (y, h) in LANE_Y.items():
        ax.text(COLS["M15"][0] - 0.12, y + h / 2, LANES[lane], ha="right", va="center", fontsize=6.5, color=INK_2, rotation=90)
        for m in ("M15", "M16", "M17"):
            x, w = COLS[m]
            _box(ax, x, y, w, h, CELLS[(lane, m)], fs=6.6, stripe=MILESTONE_COLOR[m])
        for a, b in (("M15", "M16"), ("M16", "M17")):
            xa = COLS[a][0] + COLS[a][1]
            _arrow(ax, xa + 0.02, y + h / 2, COLS[b][0] - 0.02, y + h / 2)
        xa = COLS["M17"][0] + COLS["M17"][1]
        _arrow(ax, xa + 0.02, y + h / 2, COLS["M18"][0] - 0.02, y + h / 2)

    # M18 spans the three lanes
    x, w = COLS["M18"]
    y_lo = LANE_Y["P3"][0]
    y_hi = LANE_Y["P1"][0] + LANE_Y["P1"][1]
    _box(ax, x, y_lo, w, y_hi - y_lo, M18_TEXT, fs=6.8, stripe=MILESTONE_COLOR["M18"])

    # gate criteria row
    gy, gh = GATE_ROW
    for m, g in zip(("M15", "M16", "M17", "M18"), ("G15", "G16", "G17", "G18"), strict=True):
        x, w = COLS[m]
        _box(ax, x, gy, w, gh, GATES[g], fs=6.2, fill=NEUTRAL, edge=INK_2)

    # alternatives and artifacts rows
    for (row_y, row_h), label, content in ((ALT_ROW, "Simpler alternative first\n(complex model must earn adoption)", ALTERNATIVES),
                                           (ART_ROW, "Primary artifacts\n(Drive, versioned run dirs)", ARTIFACTS)):
        ax.text(lx, row_y + row_h / 2, label, ha="left", va="center", fontsize=8, color=INK, fontweight="bold", linespacing=1.25)
        for m, (x, w) in COLS.items():
            _box(ax, x, row_y, w, row_h, content[m], fs=6.4)
    ax.text(lx, GATE_ROW[0] + GATE_ROW[1] / 2, "Scientific go / no-go gates", ha="left", va="center", fontsize=8, color=INK,
            fontweight="bold")

    # right column: scouting application, pivot, outputs
    rx, rw = RIGHT
    _box(ax, rx, LANE_Y["P3"][0], rw, hy + hh - LANE_Y["P3"][0], SCOUTING, fs=6.8, edge=INK_2, lw=1.0)
    _arrow(ax, COLS["M18"][0] + COLS["M18"][1] + 0.02, 5.5, rx - 0.02, 5.5, color=INK_2, lw=1.2)
    _box(ax, rx, GATE_ROW[0], rw, GATE_ROW[1], "", fill=NEUTRAL, edge=INK_2)
    ax.text(rx + rw / 2, GATE_ROW[0] + GATE_ROW[1] / 2, "Fallback", ha="center", va="center", fontsize=7, color=INK, fontweight="bold")
    _box(ax, rx, ALT_ROW[0], rw, ALT_ROW[1], PIVOT, fs=6.4, ls="--")
    _box(ax, rx, ART_ROW[0], rw, ART_ROW[1], OUTPUTS, fs=6.4)

    # parallel-stream note and legend
    ax.text(0.3, 0.6, "P1, P2 and P3 start in parallel. Critical path: M1 reproducibility → M16 baselines → M18 B0–B3. "
                      "B4–B7 run only if G15 / G17 pass; otherwise they are reported as blocked.",
            ha="left", va="center", fontsize=7.2, color=INK_2)
    ax.text(0.3, 0.3, "[OBS] observed in an executed protocol notebook (solid border) · [EXP] exploratory, needs replication "
                      "(dashed border) · colour bar = milestone; every box is also labelled in text.",
            ha="left", va="center", fontsize=7.2, color=MUTED)

    data = {"title": TITLE, "subtitle": SUBTITLE, "evidence": [{"label": t, "text": s} for t, s in EVIDENCE],
            "milestones": MILESTONES, "lanes": LANES, "cells": {f"{k[0]}|{k[1]}": v for k, v in CELLS.items()}, "m18": M18_TEXT,
            "gates": GATES, "alternatives": ALTERNATIVES, "artifacts": ARTIFACTS, "scouting": SCOUTING, "pivot": PIVOT,
            "source": "docs/research/M15_M18_MASTER_RESEARCH_ROADMAP.md"}
    return fig, data


def main(out_dir: str | Path = "figures/roadmap", show: bool = False) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, data = build_figure()
    stem = out / "M15_M18_RESEARCH_ROADMAP"
    paths = {}
    for ext, kw in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, facecolor=PAGE, **kw)
        paths[ext] = str(p)
    side = out / "M15_M18_RESEARCH_ROADMAP__data.json"
    side.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    paths["data"] = str(side)
    if show:
        plt.show()
    plt.close(fig)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="figures/roadmap")
    args = ap.parse_args()
    for k, v in main(args.out).items():
        print(f"{k}: {v}")
