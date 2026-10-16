"""Shared UI pieces: data guard, metric rows, aligned curves, export panel."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from . import data, export
from .kinematics import ALIGNMENTS, align, derived_dsdt


def guard(fn, *args, **kwargs):
    """Run a data call; show a helpful message and stop the page if inputs are missing."""
    try:
        return fn(*args, **kwargs)
    except data.DataUnavailable as exc:
        st.error(f"Data not available: {exc}")
        st.info("Build the serving snapshot on the host: `python -m nflbdb.lab.build` (see the lab README).")
        st.stop()


def sidebar_footer() -> None:
    with st.sidebar:
        st.divider()
        if st.button("Reload data", help="Clear caches after rebuilding the serving snapshot"):
            data.reset_cache()
            st.rerun()


def metric_row(label: str, summary: dict, extra: dict | None = None) -> None:
    st.markdown(f"**{label}**")
    cols = st.columns(6)
    cols[0].metric("Max speed", f"{summary.get('max_s', float('nan')):.2f} yd/s",
                   f"{summary.get('max_s_mph', float('nan')):.1f} mph", delta_color="off")
    cols[1].metric("Time to peak", f"{summary.get('time_to_peak_s', float('nan')):.2f} s")
    cols[2].metric("Max accel", f"{summary.get('max_a', float('nan')):.2f} yd/s²")
    cols[3].metric("Distance", f"{summary.get('distance_yds', float('nan')):.1f} yd")
    cols[4].metric("Duration", f"{summary.get('duration_s', float('nan')):.1f} s")
    cols[5].metric("Frames", f"{summary.get('frames', 0):,}")
    if extra:
        st.caption(" · ".join(f"{k}: {v}" for k, v in extra.items()))


def alignment_picker(key: str, allowed: tuple[str, ...], default: str = "start") -> tuple[str, float]:
    method = st.radio("Time alignment", allowed, index=allowed.index(default), key=f"{key}_align", horizontal=True,
                      format_func=lambda m: ALIGNMENTS[m][0])
    thr = 1.0
    if method == "onset":
        thr = st.slider("Onset threshold (yd/s)", 0.2, 3.0, 1.0, 0.1, key=f"{key}_thr")
    st.caption(f"**Alignment — {ALIGNMENTS[method][0]}:** {ALIGNMENTS[method][1]}")
    return method, thr


def curves_figure(series: list[tuple[str, str, pd.DataFrame, float | None]], method: str, threshold: float,
                  show_dsdt: bool = False) -> tuple[go.Figure, list[str]]:
    """``series``: (label, color, frames, snap_t). Returns the 2-row speed/accel figure and notes."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Speed (yd/s)", "Acceleration (yd/s²)"))
    notes = []
    for label, color, df, snap_t in series:
        t, note = align(df, method, threshold, snap_t)
        if t is None:
            notes.append(f"{label}: alignment not applicable ({note})")
            continue
        fig.add_trace(go.Scatter(x=t, y=df["s"], name=f"{label} · speed", line=dict(color=color, width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=df["a"], name=f"{label} · |a| (vendor)", line=dict(color=color, width=2)), 2, 1)
        if show_dsdt:
            fig.add_trace(go.Scatter(x=t, y=derived_dsdt(df), name=f"{label} · dS/dt (derived, unsmoothed)",
                                     line=dict(color=color, width=1, dash="dot")), 2, 1)
    x_title = "τ (fraction of duration)" if method == "normalized" else f"t (s) — aligned on {ALIGNMENTS[method][0].lower()}"
    fig.update_xaxes(title_text=x_title, row=2, col=1, zeroline=True, zerolinecolor="#52514e")
    fig.update_layout(height=520, paper_bgcolor="#0f1115", plot_bgcolor="#181b22", font=dict(color="#e8e8e3"),
                      legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    fig.update_yaxes(gridcolor="#2a2d35")
    fig.update_xaxes(gridcolor="#2a2d35")
    return fig, notes


def export_panel(key: str, figures: dict[str, go.Figure], tables: dict[str, pd.DataFrame]) -> None:
    """Buttons that write 300-DPI PNG + SVG / CSV into the mounted report folders, plus browser downloads."""
    with st.expander("Export (PNG 300 DPI + SVG → reports/figures/analytics_lab · CSV → reports/tables/analytics_lab)"):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save figures to reports/", key=f"{key}_savefig"):
                try:
                    written = [p.name for name, fig in figures.items() for p in export.save_figure(fig, name)]
                    st.success("Saved: " + ", ".join(written))
                except Exception as exc:  # noqa: BLE001 — kaleido/IO problems are shown, not fatal
                    st.error(f"Figure export failed: {type(exc).__name__}: {exc}")
            if st.toggle("Prepare PNG downloads (renders each figure)", key=f"{key}_prep"):
                for name, fig in figures.items():
                    try:
                        st.download_button(f"Download {name}.png", export.png_bytes(fig), f"{name}.png", "image/png",
                                           key=f"{key}_dl_{name}")
                    except Exception as exc:  # noqa: BLE001
                        st.caption(f"PNG rendering unavailable: {exc}")
                        break
        with c2:
            if st.button("Save tables to reports/", key=f"{key}_savecsv"):
                written = [export.save_csv(df, name).name for name, df in tables.items()]
                st.success("Saved: " + ", ".join(written))
            for name, df in tables.items():
                st.download_button(f"Download {name}.csv", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv",
                                   key=f"{key}_dlcsv_{name}")
