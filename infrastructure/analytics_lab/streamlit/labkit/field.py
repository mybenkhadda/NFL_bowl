"""American-football field + animated trajectories (Plotly).

Coordinates (warehouse convention, yards): x ∈ [0, 120] along the field incl. both end zones
(goal lines at x = 10 and x = 110), y ∈ [0, 53.3] across. Hash marks 70'9" (23.58 yd) from each
sideline. Direction ``dir``: Big Data Bowl convention, 0° = +y, clockwise → unit vector
(sin θ, cos θ). The same convention is ASSUMED for Combine tracking (not verified).
The y-axis is scaled 1:1 with x so distances and angles are not distorted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

FIELD_LEN, FIELD_WID = 120.0, 53.3
HASH_Y = (23.58, FIELD_WID - 23.58)
TURF, ENDZONE, LINE, TEXT = "#173b26", "#12301f", "rgba(255,255,255,0.55)", "rgba(255,255,255,0.75)"
POSITION_COLORS = {"DB": "#2a78d6", "OL": "#eb6834", "DL": "#1baf7a", "WR": "#eda100", "TE": "#e87ba4",
                   "UNKNOWN": "#b9b8b0"}
BALL_COLOR = "#a0522d"
COMPARE_COLORS = ("#2a78d6", "#eb6834")      # A, B


def field_figure(x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None,
                 height: int = 520) -> go.Figure:
    shapes = [dict(type="rect", x0=0, x1=FIELD_LEN, y0=0, y1=FIELD_WID, fillcolor=TURF, line=dict(width=0), layer="below"),
              dict(type="rect", x0=0, x1=10, y0=0, y1=FIELD_WID, fillcolor=ENDZONE, line=dict(width=0), layer="below"),
              dict(type="rect", x0=110, x1=FIELD_LEN, y0=0, y1=FIELD_WID, fillcolor=ENDZONE, line=dict(width=0), layer="below"),
              dict(type="rect", x0=0, x1=FIELD_LEN, y0=0, y1=FIELD_WID, line=dict(color=LINE, width=2), layer="below")]
    for x in range(10, 111, 5):
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=FIELD_WID, layer="below",
                           line=dict(color=LINE, width=2 if x in (10, 110) else (1.2 if x % 10 == 0 else 0.6))))
    for x in range(11, 110):
        if x % 5:
            for hy in HASH_Y:
                shapes.append(dict(type="line", x0=x, x1=x, y0=hy - 0.35, y1=hy + 0.35, line=dict(color=LINE, width=0.6),
                                   layer="below"))
    annotations = []
    for x in range(20, 101, 10):
        label = str(x - 10 if x <= 60 else 110 - x)
        for y, angle in ((12, 0), (FIELD_WID - 12, 180)):
            annotations.append(dict(x=x, y=y, text=label, showarrow=False, textangle=angle,
                                    font=dict(color=TEXT, size=14)))
    fig = go.Figure()
    fig.update_layout(
        shapes=shapes, annotations=annotations, height=height, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#0f1115", plot_bgcolor="#0f1115", font=dict(color="#e8e8e3"),
        xaxis=dict(range=list(x_range or (-2, FIELD_LEN + 2)), showgrid=False, zeroline=False, title="x (yd)"),
        yaxis=dict(range=list(y_range or (-2, FIELD_WID + 2)), showgrid=False, zeroline=False, title="y (yd)",
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def zoom_ranges(xs: np.ndarray, ys: np.ndarray, pad: float = 5.0) -> tuple[tuple[float, float], tuple[float, float]]:
    return (float(np.nanmin(xs) - pad), float(np.nanmax(xs) + pad)), (float(np.nanmin(ys) - pad), float(np.nanmax(ys) + pad))


def _segments(df: pd.DataFrame, lookahead_s: float = 0.5) -> tuple[list, list]:
    """Direction segments: length = distance covered in ``lookahead_s`` at the current speed."""
    xs, ys = [], []
    for x, y, s, d in df[["x", "y", "s", "dir"]].itertuples(index=False):
        if pd.isna(d) or pd.isna(s):
            continue
        r = np.radians(d)
        xs += [x, x + np.sin(r) * s * lookahead_s, None]
        ys += [y, y + np.cos(r) * s * lookahead_s, None]
    return xs, ys


def trajectory_figure(frames: pd.DataFrame, meta: dict, focus=None, step: int = 1, title: str = "",
                      x_range=None, y_range=None, animate: bool = True, frame_index: int | None = None,
                      height: int = 520) -> go.Figure:
    """``frames``: columns entity, frame_id, t_s, x, y, s, a, dir.
    ``meta``: entity → {label, color, size}. ``focus``: entity whose stats are shown per frame.
    ``animate=False`` returns a static figure at ``frame_index`` (used for PNG/SVG export)."""
    fig = field_figure(x_range, y_range, height)
    entities = list(meta)
    for e in entities:                                  # static full paths
        d = frames[frames["entity"] == e]
        fig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="lines", name=meta[e]["label"], hoverinfo="skip",
                                 line=dict(color=meta[e]["color"], width=2 if e == focus else 1.2),
                                 opacity=0.85 if e == focus else 0.45))
    frame_ids = sorted(frames["frame_id"].unique())[::max(1, step)]
    if not frame_ids:
        return fig
    by_frame = {f: g for f, g in frames[frames["frame_id"].isin(frame_ids)].groupby("frame_id")}

    def state(fid):
        g = by_frame.get(fid, frames.iloc[0:0])
        g = g[g["entity"].isin(entities)]
        colors = [meta[e]["color"] for e in g["entity"]]
        sizes = [meta[e].get("size", 10) + (6 if e == focus else 0) for e in g["entity"]]
        hover = [f"{meta[e]['label']}<br>t={t:.2f} s · s={s:.2f} yd/s · a={a:.2f} yd/s² · dir={d:.0f}°"
                 for e, t, s, a, d in g[["entity", "t_s", "s", "a", "dir"]].itertuples(index=False)]
        dots = go.Scatter(x=g["x"], y=g["y"], mode="markers", name="position", showlegend=False, hovertext=hover,
                          hoverinfo="text", marker=dict(color=colors, size=sizes, line=dict(color="#0f1115", width=1)))
        sx, sy = _segments(g)
        arrows = go.Scatter(x=sx, y=sy, mode="lines", name="direction (0.5 s)", showlegend=False, hoverinfo="skip",
                            line=dict(color="#ffffff", width=1.5))
        label = ""
        if focus is not None and (g["entity"] == focus).any():
            r = g[g["entity"] == focus].iloc[0]
            label = (f"{meta[focus]['label']} · t = {r['t_s']:.2f} s · speed {r['s']:.2f} yd/s · "
                     f"accel {r['a']:.2f} yd/s² · dir {r['dir']:.0f}°")
        return dots, arrows, label

    n_static = len(entities)
    start = frame_ids[0] if frame_index is None else frame_ids[min(frame_index, len(frame_ids) - 1)]
    dots, arrows, label = state(start)
    fig.add_trace(dots)
    fig.add_trace(arrows)
    fig.update_layout(title=dict(text=f"{title}<br><sup>{label}</sup>", x=0.01))
    if not animate:
        return fig
    animation = []
    for fid in frame_ids:
        f_dots, f_arrows, f_label = state(fid)
        animation.append(go.Frame(name=str(fid), data=[f_dots, f_arrows], traces=[n_static, n_static + 1],
                                  layout=go.Layout(title=dict(text=f"{title}<br><sup>{f_label}</sup>"))))
    fig.frames = animation
    ms = 100 * max(1, step)
    fig.update_layout(
        updatemenus=[dict(type="buttons", x=0.0, y=-0.02, xanchor="left", yanchor="top", direction="left",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=ms, redraw=False), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.12, y=-0.02, len=0.88, currentvalue=dict(prefix="frame "),
                      steps=[dict(method="animate", label=str(fid),
                                  args=[[str(fid)], dict(mode="immediate", frame=dict(duration=0, redraw=False))])
                             for fid in frame_ids])],
    )
    return fig
