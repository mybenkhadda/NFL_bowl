"""M15 figures (T01 scope: data-audit figures). Every function returns ``(figure, data)`` where ``data`` maps a part
name to the exact plotted numbers (polars DataFrame) — pass both to ``M15Run.figure`` so PNG / SVG / PDF and the
source-data CSV / Parquet are written together. White background, colour-blind-safe fixed categorical order, physical
units on every axis, sample sizes in every title.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# fixed categorical order (dataviz reference palette, light mode): domain / series identity is ALSO given in text labels
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
DOMAIN_COLOR = {"combine": PALETTE[0], "nfl": PALETTE[1]}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
CHANNEL_LABEL = {"speed": "speed (yd/s)", "a_tan": "tangential acceleration (yd/s²)", "a_n": "normal acceleration (yd/s²)", "yaw": "heading rate (rad/s)"}


def _style(ax: Any) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)


def valid_frames_hist(index: Any, max_frames: int) -> tuple[Any, dict[str, Any]]:
    """E15.A1 preview — distribution of valid frames per sequence by domain (sequence counts; share per domain)."""
    import matplotlib.pyplot as plt
    import polars as pl

    bins = np.arange(0, max_frames + 2) - 0.5
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    parts = []
    for dom in ("combine", "nfl"):
        v = index.filter(pl.col("domain") == dom)["n_valid"].to_numpy()
        if not v.size:
            continue
        h, _ = np.histogram(v, bins=bins)
        share = h / max(v.size, 1)
        ax.step(np.arange(max_frames + 1), share, where="mid", color=DOMAIN_COLOR[dom], linewidth=2, label=f"{dom} (n = {v.size:,}; median {np.median(v):.0f})")
        parts.append(pl.DataFrame({"domain": [dom] * h.size, "valid_frames": np.arange(max_frames + 1), "n_sequences": h, "share": share}))
    _style(ax)
    ax.set_xlabel("valid frames per sequence (10 Hz)", color=INK)
    ax.set_ylabel("share of the domain's sequences", color=INK)
    ax.set_title("Valid frames per sequence by domain", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"valid_frames": pl.concat(parts) if parts else pl.DataFrame()}


def signal_panel(bundle: Any, rows: np.ndarray, labels: list[str], channels: tuple[str, ...] = ("speed", "a_tan", "yaw")) -> tuple[Any, dict[str, Any]]:
    """Physical-unit traces of a few sequences (valid frames only; masked frames are gaps, never drawn as zeros).
    One shared x-axis in seconds from the first frame (Σ dt); identical y-axes per channel across sequences."""
    import matplotlib.pyplot as plt
    import polars as pl

    from .data import physical_rows

    rows = np.asarray(rows, np.int64)
    phys = physical_rows(bundle, rows)
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.4 * len(channels)), sharex=True, facecolor="white")
    axes = np.atleast_1d(axes)
    parts = []
    for k, r in enumerate(rows):
        dt = np.where(np.isfinite(phys["dt"][k]), phys["dt"][k], 0.1)
        t = np.concatenate([[0.0], np.cumsum(dt[:-1])])
        color = PALETTE[k % len(PALETTE)]
        rec = {"row": np.full(t.size, int(r)), "label": [labels[k]] * t.size, "t_s": t, "valid": phys["valid"][k]}
        for ax, ch in zip(axes, channels, strict=True):
            y = phys[ch][k].copy()
            if ch == "yaw":
                y = np.where(phys["heading_valid"][k] == 1.0, y, np.nan)        # heading-invalid yaw is not an observation
            ax.plot(t, y, color=color, linewidth=1.8, label=labels[k])
            rec[ch] = y
        parts.append(pl.DataFrame(rec))
    for ax, ch in zip(axes, channels, strict=True):
        _style(ax)
        ax.set_ylabel(CHANNEL_LABEL.get(ch, ch), color=INK, fontsize=8)
    axes[-1].set_xlabel("time from first frame (s)", color=INK)
    axes[0].set_title(f"Sequences in physical units (n = {rows.size}; gaps = invalid frames)", color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig, {"traces": pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame()}


def audit_status_bar(audit: Any) -> tuple[Any, dict[str, Any]]:
    """Counts of data-contract checks per status (pass / warn / fail) — the one-glance audit summary."""
    import matplotlib.pyplot as plt
    import polars as pl

    order = ("pass", "warn", "fail")
    cnt = audit.group_by("status").len().to_dict(as_series=False)
    d = dict(zip(cnt["status"], cnt["len"], strict=True))
    vals = [int(d.get(s, 0)) for s in order]
    colors = {"pass": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b"}             # status palette, always paired with text labels
    fig, ax = plt.subplots(figsize=(6, 2.6), facecolor="white")
    ax.barh(list(order), vals, color=[colors[s] for s in order], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + 0.2, y, f"{v}", va="center", fontsize=9, color=INK)
    _style(ax)
    ax.set_xlabel("number of checks", color=INK)
    ax.set_title(f"Data-contract audit ({audit.height} checks)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig, {"audit_status": pl.DataFrame({"status": list(order), "n_checks": vals})}


__all__ = ["DOMAIN_COLOR", "PALETTE", "audit_status_bar", "signal_panel", "valid_frames_hist"]
