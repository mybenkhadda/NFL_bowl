"""M14 figure types — white 16:9 canvases, ink text, categorical colours from the shared validated palette
(``lab.figures``). Every builder takes plain polars tables (the saved figure data) and returns a matplotlib Figure;
the ``*_profile`` helpers return the plotted numbers so they are saved next to the figure.

Standard figure types (forest, bars, heatmaps, lines, scatter) are reused from ``lab.figures`` unchanged.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .. import figures as F
from .reporting import STATUS_COLORS

WHITE = "#ffffff"
WIDE = (12.0, 6.75)
NOMINAL_DT = 0.1


def _fig(ncols: int = 1, nrows: int = 1, size: tuple[float, float] = WIDE, sharex: bool = False, sharey: bool = False) -> tuple[Any, Any]:
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, facecolor=WHITE, layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=sharex, sharey=sharey)
    for ax in axes.ravel():
        ax.set_facecolor(WHITE)
    return fig, axes


def _style(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", color=F.INK, fontsize=11, fontweight="semibold")
    ax.set_xlabel(xlabel, color=F.INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=F.INK2, fontsize=9)
    ax.tick_params(colors=F.INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(F.AXIS)
    ax.grid(True, axis="y", color=F.GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def suptitle(fig: Any, title: str, subtitle: str = "") -> None:
    fig.suptitle(title, x=0.01, ha="left", color=F.INK, fontsize=13, fontweight="semibold")
    if subtitle:
        fig.text(0.01, 0.955, subtitle, ha="left", va="top", color=F.INK2, fontsize=9)


# ====================================================================== event anatomy (aligned traces)
def aligned_profile(events: Any, group: str, half: int, min_events: int = 20, only_complete: bool = True) -> Any:
    """Long table (group, t_s, quantity, median, q25, q75, n) of the event traces aligned on the speed minimum."""
    import polars as pl

    ev = events.filter(pl.col("complete")) if only_complete and "complete" in events.columns else events
    rows = []
    t = (np.arange(2 * half + 1) - half) * NOMINAL_DT
    for gval in sorted(set(v for v in ev[group].to_list() if v is not None)):
        sub = ev.filter(pl.col(group) == gval)
        if sub.height < min_events:
            continue
        for q, col in (("speed", "trace_speed"), ("a_tan", "trace_a_tan")):
            A = np.array(sub[col].to_list(), dtype=float)
            if A.ndim != 2 or A.shape[1] != t.size:
                continue
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med, lo, hi = np.nanmedian(A, 0), np.nanquantile(A, 0.25, 0), np.nanquantile(A, 0.75, 0)
            n = np.isfinite(A).sum(0)
            rows += [{"group": str(gval), "t_s": float(t[k]), "quantity": q, "median": float(med[k]), "q25": float(lo[k]), "q75": float(hi[k]), "n": int(n[k])}
                     for k in range(t.size)]
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"group": pl.String, "t_s": pl.Float64, "quantity": pl.String, "median": pl.Float64})


def event_anatomy(profile: Any, title: str, subtitle: str = "") -> Any:
    """Two panels (speed, tangential acceleration) of the median ± IQR aligned event per group."""
    import polars as pl

    if profile is None or not profile.height:
        return F.empty(title, "no group has enough complete events")
    fig, ax = _fig(2, 1)
    groups = profile["group"].unique(maintain_order=True).to_list()
    pal = F.palette_for(groups)
    for j, (q, ylab) in enumerate((("speed", "speed (yd/s)"), ("a_tan", "tangential acceleration (yd/s²)"))):
        a = ax[0, j]
        for gval in groups:
            d = profile.filter((pl.col("group") == gval) & (pl.col("quantity") == q)).sort("t_s")
            if not d.height:
                continue
            a.fill_between(d["t_s"].to_numpy(), d["q25"].to_numpy(), d["q75"].to_numpy(), color=pal[gval], alpha=0.14, linewidth=0)
            a.plot(d["t_s"].to_numpy(), d["median"].to_numpy(), color=pal[gval], linewidth=2, label=f"{gval}")
        a.axvline(0, color=F.REF, linewidth=0.8, linestyle="--")
        if q == "a_tan":
            a.axhline(0, color=F.REF, linewidth=0.8)
        _style(a, "Speed around the minimum" if q == "speed" else "Braking → propulsion", "time from speed minimum (s)", ylab)
        if j == 0:
            F._legend(a)
    suptitle(fig, title, subtitle)
    return fig


def twin_profile(events: Any, ids: Sequence[int], labels: dict[int, str], half: int) -> Any:
    """Aligned profile per athlete of a pair (agility, complete events); ``group`` = anonymised label."""
    import polars as pl

    ev = events.filter(pl.col("nfl_id").is_in(list(ids)) & (pl.col("domain") == "combine"))
    if "kind" in ev.columns:
        ev = ev.filter(pl.col("kind").is_in(["three_cone", "short_shuttle", "long_shuttle"]))
    lab = pl.DataFrame({"nfl_id": [int(i) for i in ids], "athlete": [labels.get(int(i), str(i)) for i in ids]}, schema={"nfl_id": pl.Int64, "athlete": pl.String})
    ev = ev.with_columns(pl.col("nfl_id").cast(pl.Int64)).join(lab, on="nfl_id", how="inner")
    return aligned_profile(ev, "athlete", half, min_events=2)


# ====================================================================== status matrix
def status_matrix(evidence: Any, title: str, cols: Sequence[str] = ("measurement_status", "single_metric_status", "family_status", "final_status"),
                  max_rows: int = 40) -> Any:
    """Metric × evidence-stage grid; each cell = status text on a white cell with a coloured status border."""
    from matplotlib.patches import Rectangle

    if evidence is None or not evidence.height:
        return F.empty(title, "no evidence rows")
    d = evidence.head(max_rows)
    labels = [f"{r['metric']} · {r['scope']}" if r.get("scope") else str(r["metric"]) for r in d.iter_rows(named=True)]
    n = len(labels)
    fig, ax = _fig(1, 1, size=(12.0, max(4.0, min(6.75 * 2, 0.32 * n + 1.6))))
    a = ax[0, 0]
    for i, r in enumerate(d.iter_rows(named=True)):
        for j, c in enumerate(cols):
            v = r.get(c)
            col = STATUS_COLORS.get(str(v), F.GRID)
            a.add_patch(Rectangle((j + 0.04, n - i - 1 + 0.08), 0.92, 0.84, facecolor=WHITE, edgecolor=col, linewidth=1.6))
            a.text(j + 0.5, n - i - 0.5, "—" if v is None else str(v), ha="center", va="center", fontsize=7.5, color=col if v is not None else F.INK2)
    a.set_xlim(0, len(cols))
    a.set_ylim(0, n)
    a.set_xticks(np.arange(len(cols)) + 0.5)
    a.set_xticklabels([c.replace("_", " ") for c in cols], fontsize=8, color=F.INK2)
    a.set_yticks(np.arange(n) + 0.5)
    a.set_yticklabels(labels[::-1], fontsize=7.5, color=F.INK)
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0)
    suptitle(fig, title, "validated · exploratory · inconclusive · blocked · invalid — declared rules, see the report")
    return fig


# ====================================================================== feasibility
def coverage_bars(cov: Any, title: str) -> Any:
    import polars as pl

    if cov is None or not cov.height:
        return F.empty(title, "no coverage rows")
    d = cov.with_columns(pl.col("column").str.replace("m14c_", "").alias("label")).sort("coverage_transfer")
    return F.hbar(d, "label", "coverage_transfer", title, "share of transfer athletes with a value", fmt="{:.2f}", ref=None)


__all__ = ["aligned_profile", "coverage_bars", "event_anatomy", "status_matrix", "suptitle", "twin_profile"]
