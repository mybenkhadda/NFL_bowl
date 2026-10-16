"""Plotly figures of the Combine Intelligence Lab, drawn only from mart rows (real athletes, real plays).

Conventions: athlete A (the selected prospect) = blue, athlete B = orange — fixed by role, never by rank. One y-scale per
chart (no dual axis). Every estimate with an interval carries it. Units on every axis. Hover on every mark.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from components.ui import C, POSITION_COLOR
from nflbdb.research.advanced_movement import movement_gap as mg  # noqa: E402  (src/ is on the path via data_access)
from nflbdb.research.advanced_movement import product_kit as pk  # noqa: E402

A, B = C["a"], C["b"]
CLASS_COLOR = {mg.DIRECT: "#2f9e5b", mg.PARTIAL: "#8cc7a1", mg.NOT_MEASURED: "#d95926", mg.UNAVAILABLE: "#5b6b82"}
CLASS_MARK = {mg.DIRECT: "D", mg.PARTIAL: "P", mg.NOT_MEASURED: "×", mg.UNAVAILABLE: "n/a"}
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def style(fig: go.Figure, height: int = 420, title: str | None = None, present: bool = False) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor=C["panel"], plot_bgcolor=C["panel"], height=int(height * (1.2 if present else 1.0)),
                      font=dict(family=FONT, color=C["text"], size=16 if present else 13), margin=dict(l=60, r=24, t=56 if title else 24, b=52),
                      title=dict(text=title, x=0.01, font=dict(size=18 if present else 15)) if title else None,
                      legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18), hoverlabel=dict(bgcolor=C["panel2"], font=dict(family=FONT)))
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["border"], linecolor=C["border"])
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["border"], linecolor=C["border"])
    return fig


def empty(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=C["muted"], size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height)


# ============================================================================== Combine sprint
def speed_distance(series: Sequence[tuple[str, str, list[dict[str, Any]]]], band: dict[str, Any] | None, position: str, present: bool = False) -> go.Figure:
    """Speed against distance for every recorded attempt of one or two athletes, over the position's 10–90 % band."""
    fig = go.Figure()
    if band:
        fig.add_trace(go.Scatter(x=band["dist_yd"] + band["dist_yd"][::-1], y=band["q90"] + band["q10"][::-1], fill="toself", fillcolor="rgba(143,161,186,0.16)",
                                 line=dict(width=0), hoverinfo="skip", name=f"{position} peers 10–90 % ({band['n']} runs)"))
        fig.add_trace(go.Scatter(x=band["dist_yd"], y=band["q50"], line=dict(color=C["neutral"], width=1, dash="dot"), name=f"{position} median",
                                 hovertemplate="%{x:.1f} yd · median %{y:.2f} yd/s<extra></extra>"))
    for name, color, attempts in series:
        for i, a in enumerate(attempts):
            fig.add_trace(go.Scatter(x=a["dist"], y=a["speed"], mode="lines", line=dict(color=color, width=3 if i == 0 else 1.6), opacity=1.0 if i == 0 else 0.6,
                                     name=f"{name} — attempt {a['attempt']}", hovertemplate=f"{name}<br>%{{x:.1f}} yd · %{{y:.2f}} yd/s<extra>attempt {a['attempt']}</extra>"))
    for lo, hi in ((4, 8),):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=C["gold"], opacity=0.06, line_width=0)
    fig.update_xaxes(title="distance covered [yd]", range=[0, 41])
    fig.update_yaxes(title="tracked speed [yd/s]", rangemode="tozero")
    return style(fig, 440, "Speed by distance — every recorded attempt", present)


def accel_speed(series: Sequence[tuple[str, str, list[dict[str, Any]]]], present: bool = False) -> go.Figure:
    """Tangential acceleration against speed, interior frames only (filter-edge frames removed)."""
    fig = go.Figure()
    for name, color, attempts in series:
        for i, a in enumerate(attempts):
            keep = [k for k, ok in enumerate(a["interior"]) if ok]
            fig.add_trace(go.Scatter(x=[a["speed"][k] for k in keep], y=[a["accel"][k] for k in keep], mode="lines+markers", marker=dict(size=4),
                                     line=dict(color=color, width=2.4 if i == 0 else 1.2), opacity=1.0 if i == 0 else 0.55, name=f"{name} — attempt {a['attempt']}",
                                     hovertemplate=f"{name}<br>%{{x:.2f}} yd/s · %{{y:.2f}} yd/s²<extra></extra>"))
    for lo, hi in ((2, 4), (4, 6), (6, 8)):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=C["gold"], opacity=0.05, line_width=0, annotation_text=f"{lo}–{hi}", annotation_position="top left",
                      annotation_font_color=C["muted"])
    fig.add_hline(y=0, line_color=C["border"])
    fig.update_xaxes(title="tracked speed [yd/s]")
    fig.update_yaxes(title="tangential acceleration [yd/s²]")
    return style(fig, 420, "Acceleration against speed — how speed was built", present)


def _dist_at(t: list[float], d: list[float], grid: np.ndarray) -> np.ndarray:
    t, d = np.asarray(t, float), np.asarray(d, float)
    ok = np.isfinite(t) & np.isfinite(d)
    t, d = t[ok] - t[ok][0], d[ok] - d[ok][0]
    return np.interp(grid, t, d, right=d[-1]) if t.size >= 2 else np.zeros(grid.size)


def sprint_replay(series: Sequence[tuple[str, str, dict[str, Any]]], present: bool = False) -> go.Figure:
    """Time-synchronised replay of the recorded distance of one attempt per athlete (10 Hz frames, linear between them).
    A 1-D replay along the run: no field coordinate is drawn or invented."""
    series = [s for s in series if s[2] and len(s[2]["t"]) >= 2]
    if not series:
        return empty("no recorded attempt to replay")
    tmax = max(float(np.nanmax(np.asarray(a["t"], float)) - np.asarray(a["t"], float)[0]) for _, _, a in series)
    grid = np.round(np.arange(0.0, tmax + 0.05, 0.1), 1)
    pos = [_dist_at(a["t"], a["dist"], grid) for _, _, a in series]
    lanes = list(range(len(series)))[::-1]
    fig = go.Figure()
    for (name, color, _), lane in zip(series, lanes):
        fig.add_trace(go.Scatter(x=[0, 0], y=[lane, lane], mode="lines", line=dict(color=color, width=10), opacity=0.35, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[0], y=[lane], mode="markers+text", marker=dict(size=22, color=color, line=dict(color="white", width=2)), text=[name],
                                 textposition="top center", name=name, hovertemplate=f"{name}<br>%{{x:.1f}} yd<extra></extra>"))
    frames = []
    for k, tk in enumerate(grid):
        data = []
        for p, lane in zip(pos, lanes):
            data += [go.Scatter(x=[0, float(p[k])], y=[lane, lane]), go.Scatter(x=[float(p[k])], y=[lane])]
        frames.append(go.Frame(data=data, name=f"{tk:.1f}"))
    fig.frames = frames
    for x in range(0, 41, 5):
        fig.add_vline(x=x, line_color=C["gold"] if x % 10 == 0 else C["grid"], line_width=2 if x % 10 == 0 else 1, opacity=0.6)
    steps = [dict(method="animate", label=f.name, args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))])
             for f in frames[::2]]
    fig.update_layout(updatemenus=[dict(type="buttons", x=0.0, y=-0.22, xanchor="left", direction="left", bgcolor=C["panel2"], font=dict(color=C["text"]),
                                        buttons=[dict(label="▶ Play (real time)", method="animate", args=[None, dict(frame=dict(duration=100, redraw=False), fromcurrent=True,
                                                                                                                      transition=dict(duration=0))]),
                                                 dict(label="❚❚ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
                      sliders=[dict(active=0, x=0.25, y=-0.12, len=0.75, currentvalue=dict(prefix="t = ", suffix=" s", font=dict(color=C["text"])), steps=steps,
                                    font=dict(color=C["muted"]))])
    fig.update_xaxes(title="distance covered [yd] (from the first recorded frame)", range=[-1, 42])
    fig.update_yaxes(visible=False, range=[-0.8, len(series) - 0.2])
    return style(fig, 300 + 50 * len(series), "Sprint replay — recorded 10 Hz progress, synchronised clocks", present)


# ============================================================================== differences
def forest(diffs: Sequence[dict[str, Any]], names: tuple[str, str], present: bool = False) -> go.Figure:
    """Each difference in standard errors of the difference, with its 95 % interval; grey band = inside the uncertainty."""
    g = [d for d in diffs if d.get("z") is not None]
    if not g:
        return empty("no feature with a repeat-attempt error estimate for both athletes")
    tone = {"distinguishable": C["good"], "not_distinguishable": C["neutral"]}
    fig = go.Figure()
    fig.add_vrect(x0=-1.96, x1=1.96, fillcolor=C["neutral"], opacity=0.12, line_width=0)
    fig.add_vline(x=0, line_color=C["border"])
    for d in g:
        col = tone.get(d["verdict"], C["warning"])
        fig.add_trace(go.Scatter(x=[d["z"] - 1.96, d["z"] + 1.96], y=[d["label"]] * 2, mode="lines", line=dict(color=col, width=3), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[d["z"]], y=[d["label"]], mode="markers", marker=dict(size=12, color=col, line=dict(color="white", width=1)), showlegend=False,
                                 hovertemplate=(f"{d['label']}<br>{names[0]}: {d['value_a']:.3f} {d['unit']}<br>{names[1]}: {d['value_b']:.3f} {d['unit']}"
                                                f"<br>difference {d['difference']:+.3f} [{d['ci_lower']:+.3f}, {d['ci_upper']:+.3f}]<br>{d['verdict'].replace('_', ' ')}<extra></extra>")))
    fig.update_xaxes(title=f"difference ({names[1]} − {names[0]}) in standard errors of the difference")
    fig.update_yaxes(autorange="reversed")
    return style(fig, 120 + 42 * len(g), "Is each difference larger than its measurement uncertainty?", present)


def band_time(diffs: Sequence[dict[str, Any]], names: tuple[str, str], present: bool = False) -> go.Figure:
    rows = []
    for lo, hi in ((2, 4), (4, 6), (6, 8)):
        d = next((x for x in diffs if x["feature"] == f"acc_{lo}_{hi}"), None)
        if d and d.get("value_a") and d.get("value_b"):
            rows.append((f"{lo} → {hi} yd/s", (hi - lo) / d["value_a"], (hi - lo) / d["value_b"], d["verdict"]))
    if not rows:
        return empty("no speed band recorded for both athletes")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[1] for r in rows], name=names[0], marker_color=A, hovertemplate="%{y:.3f} s<extra>" + names[0] + "</extra>"))
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[2] for r in rows], name=names[1], marker_color=B, hovertemplate="%{y:.3f} s<extra>" + names[1] + "</extra>"))
    for r in rows:
        fig.add_annotation(x=r[0], y=max(r[1], r[2]), text=f"{r[2] - r[1]:+.3f} s · {r[3].replace('_', ' ')}", showarrow=False, yshift=16, font=dict(color=C["muted"], size=12))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.06)
    fig.update_yaxes(title="time spent in the band [s]")
    return style(fig, 380, f"Time gained or lost through each speed band ({names[1]} − {names[0]})", present)


def metric_bars(metrics: Sequence[dict[str, Any]], present: bool = False) -> go.Figure:
    m = [r for r in metrics if r.get("pct_position") is not None]
    if not m:
        return empty("no tracked measure with a position reference")
    fig = go.Figure(go.Bar(x=[r["pct_position"] for r in m], y=[r["label"] for r in m], orientation="h", marker_color=A,
                           text=[f"{r['value']:.2f} {r['unit']}" + ("" if r.get("standard_error") is None else f" ± {1.96 * r['standard_error']:.2f}") for r in m],
                           textposition="outside", hovertemplate="%{y}<br>raw percentile %{x:.0f}<extra></extra>"))
    fig.add_vline(x=50, line_color=C["border"], line_dash="dot")
    fig.update_xaxes(title="raw position percentile of the value (for times, LOWER = quicker)", range=[0, 125])
    fig.update_yaxes(autorange="reversed")
    return style(fig, 120 + 40 * len(m), "Tracked measures against the position (value ± 95 % repeat-attempt interval)", present)


# ============================================================================== NFL demands
def demand_heatmap(dem: pl.DataFrame, labels: dict[str, str], present: bool = False) -> go.Figure:
    d = dem.filter(pl.col("position_group") != "ALL")
    if d.is_empty():
        return empty("no NFL demand table")
    pos = sorted(d["position_group"].unique().to_list())
    ids = d["demand_id"].unique(maintain_order=True).to_list()
    cell = {(r["position_group"], r["demand_id"]): r for r in d.iter_rows(named=True)}
    z = [[cell[(p, i)]["episodes_per_100_plays"] if cell.get((p, i), {}).get("supported") else None for p in pos] for i in ids]
    txt = [[(f"{cell[(p, i)]['episodes_per_100_plays']:.0f}<br>n = {cell[(p, i)]['n_episodes']:,}" if cell[(p, i)]["supported"] else "too few") if (p, i) in cell else "—"
            for p in pos] for i in ids]
    fig = go.Figure(go.Heatmap(z=z, x=pos, y=[labels.get(i, i) for i in ids], text=txt, texttemplate="%{text}", colorscale="Blues", hoverongaps=False,
                               colorbar=dict(title="per 100<br>plays"), hovertemplate="%{y} · %{x}<br>%{z:.1f} episodes per 100 eligible player-plays<extra></extra>"))
    return style(fig, 460, "How often each movement demand occurs, by position", present)


def demand_distribution(dist: pl.DataFrame, demand: str, feature: str, positions: Sequence[str], unit: str, present: bool = False) -> go.Figure:
    g = dist.filter((pl.col("demand_id") == demand) & (pl.col("feature") == feature) & pl.col("position_group").is_in(list(positions)))
    if g.is_empty():
        return empty("no distribution for this demand and dimension")
    fig = go.Figure()
    for (p,), s in sorted(g.partition_by(["position_group"], as_dict=True).items()):
        s = s.sort("bin")
        tot = max(1, int(s["n_episodes"].sum()))
        x = [(a + b) / 2 for a, b in zip(s["bin_lo"].to_list(), s["bin_hi"].to_list())]
        fig.add_trace(go.Scatter(x=x, y=[100 * n / tot for n in s["n_episodes"].to_list()], mode="lines", line=dict(color=POSITION_COLOR.get(p, C["neutral"]), width=2.5, shape="hvh"),
                                 name=f"{p} ({tot:,} episodes)", hovertemplate=f"{p}<br>%{{x:.2f}} {unit} · %{{y:.1f}} % of episodes<extra></extra>"))
    fig.update_xaxes(title=f"{feature} [{unit}]")
    fig.update_yaxes(title="share of the position's episodes [%]")
    return style(fig, 400, "Distribution of the demand by position (episodes are dependent; athletes are the sample size)", present)


def episode_replay(ex: pl.DataFrame, present: bool = False) -> go.Figure:
    """A real NFL movement episode: displacement path and speed, animated frame by frame (10 Hz)."""
    e = ex.sort("frame")
    if e.height < 3:
        return empty("no recorded frames for this example")
    x, y, t, v, inside = (e[k].to_list() for k in ("rel_x", "rel_y", "t_s", "speed", "in_episode"))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], subplot_titles=("path (displacement from the episode start, yd)", "speed [yd/s]"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C["neutral"], width=2), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", marker=dict(size=16, color=A, line=dict(color="white", width=2)), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=v, mode="lines", line=dict(color=C["neutral"], width=2), showlegend=False, hoverinfo="skip"), 1, 2)
    fig.add_trace(go.Scatter(x=[t[0]], y=[v[0]], mode="markers", marker=dict(size=14, color=A, line=dict(color="white", width=2)), showlegend=False), 1, 2)
    ins = [k for k, ok in enumerate(inside) if ok]
    if ins:
        fig.add_vrect(x0=t[ins[0]], x1=t[ins[-1]], fillcolor=C["gold"], opacity=0.12, line_width=0, row=1, col=2)
    fig.frames = [go.Frame(data=[go.Scatter(x=[x[k]], y=[y[k]]), go.Scatter(x=[t[k]], y=[v[k]])], traces=[1, 3], name=str(k)) for k in range(len(t))]
    fig.update_layout(updatemenus=[dict(type="buttons", x=0.0, y=-0.18, xanchor="left", direction="left", bgcolor=C["panel2"], font=dict(color=C["text"]),
                                        buttons=[dict(label="▶ Replay", method="animate", args=[None, dict(frame=dict(duration=100, redraw=False), fromcurrent=False,
                                                                                                             transition=dict(duration=0))])])])
    fig.update_xaxes(title_text="t since episode start [s]", row=1, col=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    return style(fig, 420, None, present)


# ============================================================================== matching
def alignment(rows: Sequence[dict[str, Any]], labels: dict[str, str], present: bool = False) -> go.Figure:
    ok = [r for r in rows if r.get("combine_value") is not None and r.get("nfl_q50") is not None]
    units = list(dict.fromkeys(r["unit"] for r in ok))
    if not ok:
        return empty("no shared dimension recorded for this prospect and demand")
    fig = make_subplots(rows=len(units), cols=1, subplot_titles=[f"[{u}]" for u in units], vertical_spacing=0.18)
    for i, u in enumerate(units, 1):
        for r in [r for r in ok if r["unit"] == u]:
            yl = f"{labels.get(r['demand_id'], r['demand_id'])} · {r['dimension']}"
            fig.add_trace(go.Scatter(x=[r["nfl_q10"], r["nfl_q90"]], y=[yl, yl], mode="lines", line=dict(color=C["neutral"], width=14), opacity=0.35, showlegend=False,
                                     hovertemplate=f"NFL episodes 10–90 %: {r['nfl_q10']:.2f}–{r['nfl_q90']:.2f}; median {r['nfl_q50']:.2f} ({r['n_nfl_episodes']:,} episodes)<extra></extra>"), i, 1)
            se = r.get("standard_error")
            if se:
                fig.add_trace(go.Scatter(x=[r["combine_value"] - 1.96 * se, r["combine_value"] + 1.96 * se], y=[yl, yl], mode="lines", line=dict(color=A, width=3),
                                         showlegend=False, hoverinfo="skip"), i, 1)
            cov = "abstains" if r.get("coverage") is None else f"coverage {r['coverage']:.0f} %"
            fig.add_trace(go.Scatter(x=[r["combine_value"]], y=[yl], mode="markers", marker=dict(size=13, color=A, line=dict(color="white", width=1.5)), showlegend=False,
                                     hovertemplate=f"Combine value {r['combine_value']:.2f} {u}<br>{cov} · {r['state'].replace('_', ' ')}<extra></extra>"), i, 1)
    return style(fig, 170 + 110 * len(ok), "Prospect value (blue, 95 % interval) inside the NFL demand range of his position (grey: 10–90 %)", present)


def coverage_bars(matrix: Sequence[dict[str, Any]], labels: dict[str, str], present: bool = False) -> go.Figure:
    s = [r for r in matrix if r.get("state") == "supported"]
    if not s:
        return empty("no supported demand correspondence")
    fig = go.Figure()
    for r in s:
        yl = labels.get(r["demand_id"], r["demand_id"])
        fig.add_trace(go.Bar(x=[r["mean_coverage"]], y=[yl], orientation="h", marker_color=C["c"], showlegend=False,
                             hovertemplate=f"{yl}<br>mean coverage {r['mean_coverage']:.0f} % [{r['coverage_lo']:.0f}, {r['coverage_hi']:.0f}] over {r['n_dimensions']} dimension(s)<extra></extra>"))
        fig.add_trace(go.Scatter(x=[r["coverage_lo"], r["coverage_hi"]], y=[yl, yl], mode="lines", line=dict(color=C["text"], width=2), showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(title="% of the position's NFL episodes at or below the Combine value", range=[0, 100])
    return style(fig, 120 + 50 * len(s), "Descriptive demand correspondence (not a prediction)", present)


def summary_heatmap(summary: pl.DataFrame, present: bool = False) -> go.Figure:
    g = summary.filter(pl.col("n_prospects") > 0)
    if g.is_empty():
        return empty("no supported comparison")
    cols = sorted({f"{r['demand_id']} · {r['dimension']}" for r in g.iter_rows(named=True)})
    pos = sorted(g["position_group"].unique().to_list())
    cell = {(r["position_group"], f"{r['demand_id']} · {r['dimension']}"): r for r in g.iter_rows(named=True)}
    z = [[cell[(p, c)]["median_coverage"] if (p, c) in cell else None for c in cols] for p in pos]
    txt = [[f"{cell[(p, c)]['median_coverage']:.0f}<br>n = {cell[(p, c)]['n_prospects']}" if (p, c) in cell else "abstain" for c in cols] for p in pos]
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=pos, text=txt, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=100, hoverongaps=False,
                               colorbar=dict(title="median<br>coverage %")))
    return style(fig, 380, "Matching by position: median prospect coverage of each demand dimension", present)


def twin_share(summary: pl.DataFrame, labels: dict[str, str], present: bool = False) -> go.Figure:
    g = summary.filter((pl.col("position_group") == "ALL") & pl.col("share_distinguishable").is_not_null()).sort("share_distinguishable")
    if g.is_empty():
        return empty("no tested twin pair")
    fig = go.Figure(go.Bar(x=(100 * g["share_distinguishable"]).to_list(), y=[labels.get(f, f) for f in g["feature"].to_list()], orientation="h", marker_color=A,
                           text=[f"{100 * s:.0f} % of {n:,}" for s, n in zip(g["share_distinguishable"].to_list(), g["n_tested"].to_list())], textposition="outside",
                           hovertemplate="%{y}: %{x:.1f} % of tested pairs distinguishable<extra></extra>"))
    fig.add_vline(x=5, line_color=C["critical"], line_dash="dash", annotation_text="5 % expected by chance", annotation_font_color=C["muted"])
    fig.update_xaxes(title="stopwatch twins with a distinguishable difference [%]", range=[0, max(20.0, 125 * float(g["share_distinguishable"].max()))])
    return style(fig, 140 + 36 * g.height, "Same stopwatch — how often is the movement measurably different?", present)


# ============================================================================== Milestone 10 — synchronized replay
METRIC_LABEL = {"speed": ("tracked speed", "yd/s"), "accel": ("tangential acceleration", "yd/s²")}


def synced_replay(series: Sequence[tuple[str, str, dict[str, Any] | None]], mode: str = "time", metric: str = "speed", playback: float = 1.0,
                  present: bool = False) -> tuple[go.Figure, str]:
    """Two (or more) recorded attempts compared at the same elapsed time (``mode='time'``, 0.1 s grid = the 10 Hz frames) or
    at the same distance (``'distance'``, 0.5 yd grid). Top: progress lanes (distance reached at time t, or time needed to
    reach distance d). Bottom: the selected metric — recorded frames as dots, the cursor moving along them.

    Returns (figure, disclosure). Nothing is drawn beyond a recording; grid points between two frames are linear
    interpolation between those two recorded frames, and the disclosure says how many there are."""
    step = 0.1 if mode == "time" else 0.5
    syn = [(n, c, pk.sync_series(a, mode, step, metric)) for n, c, a in series if a]
    syn = [s for s in syn if s[2]["grid"]]
    if not syn:
        return empty("no recorded attempt to replay"), "No recorded attempt."
    master = max((s[2]["grid"] for s in syn), key=len)
    ylab, yunit = METRIC_LABEL.get(metric, METRIC_LABEL["speed"])
    xlab = "time from the first recorded frame [s]" if mode == "time" else "distance from the first recorded frame [yd]"
    plab = "distance reached [yd]" if mode == "time" else "time needed [s]"
    fig = make_subplots(rows=2, cols=1, row_heights=[0.34, 0.66], vertical_spacing=0.16,
                        subplot_titles=(f"Progress — {plab}", f"{ylab} [{yunit}] — dots are recorded 10 Hz frames"))
    lanes = list(range(len(syn)))[::-1]
    for (name, color, s), lane in zip(syn, lanes):
        last = next((p for p in reversed(s["progress"]) if p is not None), 0.0)
        fig.add_trace(go.Scatter(x=[0, last], y=[lane, lane], mode="lines", line=dict(color=color, width=8), opacity=0.18, showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=s["x_obs"], y=s["y_obs"], mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=4), name=name,
                                 hovertemplate=f"{name}<br>%{{x:.2f}} · %{{y:.2f}} {yunit}<extra>recorded frame</extra>"), 2, 1)
    dyn = []
    for (name, color, s), lane in zip(syn, lanes):
        dyn.append(len(fig.data))
        fig.add_trace(go.Scatter(x=[0], y=[lane], mode="markers+text", marker=dict(size=20, color=color, line=dict(color="white", width=2)), text=[name],
                                 textposition="top center", showlegend=False, hovertemplate=f"{name}<br>%{{x:.2f}}<extra></extra>"), 1, 1)
        dyn.append(len(fig.data))
        fig.add_trace(go.Scatter(x=[s["grid"][0]], y=[s["value"][0]], mode="markers", marker=dict(size=14, color=color, symbol="diamond", line=dict(color="white", width=1.5)),
                                 showlegend=False, hoverinfo="skip"), 2, 1)
    frames = []
    unit = "yd" if mode == "time" else "s"
    for k, g in enumerate(master):
        data = []
        for j, (name, _, s) in enumerate(syn):
            ok = k < len(s["grid"]) and s["progress"][k] is not None
            if ok:
                p, v, gx, label = s["progress"][k], s["value"][k], g, f"{name}: {s['progress'][k]:.1f} {unit}"
            else:                                                  # past the end of this recording: hold the LAST RECORDED point, labelled
                kk = max(i for i, x in enumerate(s["progress"]) if x is not None)
                p, v, gx, label = s["progress"][kk], s["value"][kk], s["grid"][kk], f"{name}: end of recording"
            data.append(go.Scatter(x=[p], y=[lanes[j]], text=[label]))
            data.append(go.Scatter(x=[gx] if v is not None else [], y=[v] if v is not None else []))
        frames.append(go.Frame(data=data, traces=dyn, name=f"{g:.1f}"))
    fig.frames = frames
    dur = int(round((100 if mode == "time" else 60) / max(0.1, playback)))
    steps = [dict(method="animate", label=f.name, args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))]) for f in frames]
    fig.update_layout(updatemenus=[dict(type="buttons", x=0.0, y=-0.12, xanchor="left", direction="left", bgcolor=C["panel2"], font=dict(color=C["text"]),
                                        buttons=[dict(label=f"▶ Play ({playback:g}×)", method="animate",
                                                      args=[None, dict(frame=dict(duration=dur, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
                                                 dict(label="❚❚ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
                      sliders=[dict(active=0, x=0.22, y=-0.06, len=0.78, currentvalue=dict(prefix=("t = " if mode == "time" else "d = "), suffix=(" s" if mode == "time" else " yd"),
                                                                                         font=dict(color=C["text"])), steps=steps, font=dict(color=C["muted"]))])
    fig.update_yaxes(visible=False, range=[-0.8, len(syn) - 0.2], row=1, col=1)
    fig.update_xaxes(title_text=plab, row=1, col=1)
    fig.update_xaxes(title_text=xlab, row=2, col=1)
    fig.update_yaxes(title_text=f"{ylab} [{yunit}]", row=2, col=1)
    n_int = sum(sum(s["interpolated"]) for _, _, s in syn)
    n_pts = sum(len(s["grid"]) for _, _, s in syn)
    note = (f"Replay on a {step:g} {'s' if mode == 'time' else 'yd'} grid. {n_int} of {n_pts} cursor positions fall between two recorded frames and are linear "
            "interpolation between those frames (the recorded series is drawn unchanged); nothing is drawn beyond a recording. Clocks start at each recording's "
            "first frame.")
    return style(fig, 560, f"Synchronized sprint replay — same {'elapsed time' if mode == 'time' else 'distance'}", present), note


def speed_time(series: Sequence[tuple[str, str, list[dict[str, Any]]]], present: bool = False) -> go.Figure:
    fig = go.Figure()
    for name, color, attempts in series:
        for i, a in enumerate(attempts):
            t = np.asarray(a["t"], float)
            if t.size < 2:
                continue
            fig.add_trace(go.Scatter(x=t - t[0], y=a["speed"], mode="lines", line=dict(color=color, width=3 if i == 0 else 1.5), opacity=1.0 if i == 0 else 0.55,
                                     name=f"{name} — attempt {a['attempt']}", hovertemplate=f"{name}<br>%{{x:.1f}} s · %{{y:.2f}} yd/s<extra>attempt {a['attempt']}</extra>"))
    if not fig.data:
        return empty("no recorded attempt")
    fig.update_xaxes(title="time from the first recorded frame [s]")
    fig.update_yaxes(title="tracked speed [yd/s]", rangemode="tozero")
    return style(fig, 400, "Speed against time — every recorded attempt", present)


# ============================================================================== Milestone 10 — phase decomposition
def phase_chart(decomp: Sequence[dict[str, Any]], names: tuple[str, str], present: bool = False) -> go.Figure:
    rows = []
    for p in decomp:
        if not p["features"]:
            rows.append((f"{p['phase']} · not identified", None))
        rows += [(f"{p['phase']} · {r['label']}", r) for r in p["features"]]
    if not rows:
        return empty("no sprint phase recorded for both athletes")
    fig = go.Figure()
    fig.add_vrect(x0=-1.96, x1=1.96, fillcolor=C["neutral"], opacity=0.12, line_width=0)
    fig.add_vline(x=0, line_color=C["border"])
    for lab, r in rows:
        if r is None or r.get("z") is None:
            fig.add_trace(go.Scatter(x=[0], y=[lab], mode="text", text=["not identified" if r is None else "no error estimate"], textfont=dict(color=C["muted"]),
                                     showlegend=False, hoverinfo="skip"))
            continue
        col = (C["good"] if r["verdict"] == "distinguishable" else C["neutral"]) if r["usable"] else C["warning"]
        fig.add_trace(go.Scatter(x=[r["z"] - 1.96, r["z"] + 1.96], y=[lab, lab], mode="lines", line=dict(color=col, width=3), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r["z"]], y=[lab], mode="markers+text", marker=dict(size=12, color=col, line=dict(color="white", width=1)),
                                 text=[f"{r['difference']:+.2f} {r['unit']}"], textposition="middle right", textfont=dict(color=C["muted"], size=11), showlegend=False,
                                 hovertemplate=(f"{lab}<br>{names[0]}: {r['value_a']:.3f} {r['unit']}<br>{names[1]}: {r['value_b']:.3f} {r['unit']}"
                                                f"<br>{r['verdict'].replace('_', ' ')}" + ("" if r["usable"] else " — feature not repeatable enough") + "<extra></extra>")))
    fig.update_xaxes(title=f"difference ({names[1]} − {names[0]}) in standard errors of the difference")
    fig.update_yaxes(autorange="reversed")
    return style(fig, 140 + 40 * len(rows), "Where the sprints differ — phase by phase (amber = feature not repeatable enough)", present)


# ============================================================================== Milestone 10 — movement gap
def gap_matrix(matrix: Sequence[dict[str, Any]], present: bool = False) -> go.Figure:
    if not matrix:
        return empty("no movement-gap rows")
    pos = sorted({r["position_group"] for r in matrix})
    dem = list(dict.fromkeys(r["demand_id"] for r in matrix))
    lab = {r["demand_id"]: r["label"] for r in matrix}
    cell = {(r["position_group"], r["demand_id"]): r for r in matrix}
    idx = {k: i for i, k in enumerate(mg.CLASS_ORDER)}
    z = [[idx[cell[(p, d)]["coverage_class"]] if (p, d) in cell else None for p in pos] for d in dem]
    txt = [[(f"{CLASS_MARK[cell[(p, d)]['coverage_class']]} · " + (f"{float(cell[(p, d)]['episodes_per_100_plays'] or 0):.1f}" if cell[(p, d)]["supported"] else "too few"))
            if (p, d) in cell else "—" for p in pos] for d in dem]
    hov = [[(f"{lab[d]} · {p}<br>{cell[(p, d)]['class_label']}<br>{cell[(p, d)]['n_episodes']:,} episodes · {cell[(p, d)]['n_athletes']} athletes · "
             f"{cell[(p, d)]['n_eligible_plays']:,} eligible player-plays<br>{cell[(p, d)]['reason']}") if (p, d) in cell else "" for p in pos] for d in dem]
    cs, n = [], len(mg.CLASS_ORDER)
    for i, k in enumerate(mg.CLASS_ORDER):
        cs += [[i / n, CLASS_COLOR[k]], [(i + 1) / n, CLASS_COLOR[k]]]
    fig = go.Figure(go.Heatmap(z=z, x=pos, y=[lab[d] for d in dem], text=txt, texttemplate="%{text}", customdata=hov, hovertemplate="%{customdata}<extra></extra>",
                               colorscale=cs, zmin=-0.5, zmax=n - 0.5, showscale=False, xgap=3, ygap=3))
    return style(fig, 470, "The movement gap — what games ask for, and what forty tracking can describe (value = episodes per 100 eligible player-plays)", present)


def gap_bars(matrix: Sequence[dict[str, Any]], position: str, present: bool = False) -> go.Figure:
    rows = sorted((r for r in matrix if r["position_group"] == position and r["supported"]), key=lambda r: float(r["episodes_per_100_plays"] or 0))
    if not rows:
        return empty(f"no supported NFL demand for {position}")
    fig = go.Figure(go.Bar(x=[float(r["episodes_per_100_plays"] or 0) for r in rows], y=[r["label"] for r in rows], orientation="h",
                           marker_color=[CLASS_COLOR[r["coverage_class"]] for r in rows],
                           text=[f"{float(r['episodes_per_100_plays'] or 0):.1f} · {CLASS_MARK[r['coverage_class']]} {r['class_label'].lower()}" for r in rows],
                           textposition="outside", customdata=[[r["n_episodes"], r["n_athletes"], r["n_eligible_plays"]] for r in rows],
                           hovertemplate="%{y}<br>%{x:.1f} per 100 eligible player-plays<br>%{customdata[0]:,} episodes · %{customdata[1]} athletes · "
                                         "%{customdata[2]:,} eligible player-plays<extra></extra>"))
    fig.update_xaxes(title="episodes per 100 eligible player-plays", range=[0, 1.8 * max(float(r["episodes_per_100_plays"] or 0) for r in rows)])
    return style(fig, 140 + 44 * len(rows), f"{position}: observed demands, coloured by what the forty describes", present)


def exposure_compare(matrix: Sequence[dict[str, Any]], a: str, b: str, present: bool = False) -> go.Figure:
    dem = list(dict.fromkeys(r["demand_id"] for r in matrix))
    lab = {r["demand_id"]: r["label"] for r in matrix}
    cell = {(r["position_group"], r["demand_id"]): r for r in matrix}
    fig = go.Figure()
    for p, col in ((a, A), (b, B)):
        rows = [cell.get((p, d)) for d in dem]
        fig.add_trace(go.Bar(y=[lab[d] for d in dem], x=[float(r["episodes_per_100_plays"] or 0) if r and r["supported"] else 0 for r in rows], orientation="h", name=p,
                             marker_color=col, text=[("" if r and r["supported"] else "too few") for r in rows], textposition="outside",
                             customdata=[[r["n_episodes"] if r else 0, r["n_athletes"] if r else 0, r["n_eligible_plays"] if r else 0] for r in rows],
                             hovertemplate=f"{p} · %{{y}}<br>%{{x:.1f}} per 100 eligible player-plays<br>%{{customdata[0]:,}} episodes · %{{customdata[1]}} athletes · "
                                           "%{customdata[2]:,} eligible player-plays<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(title="episodes per 100 eligible player-plays (a rate: compare these, not raw counts)")
    fig.update_yaxes(autorange="reversed")
    return style(fig, 160 + 46 * len(dem), f"{a} versus {b} — standardized exposure rates", present)
