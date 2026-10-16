"""NFL Analytics Lab — trajectory explorer (home page)."""

import streamlit as st

from labkit import data, ui
from labkit.field import POSITION_COLORS

st.set_page_config(page_title="NFL Lab · Trajectories", page_icon="🏈", layout="wide")

st.title("Trajectory explorer")
st.caption("Frame-level movement for one Combine attempt or one NFL play at a time. "
           "Aggregate dashboards live in Superset (http://localhost:8088).")

mt = data.serving_mtime()
kpi = ui.guard(data.kpi, mt)
row = kpi.iloc[0]

c = st.columns(5)
c[0].metric("Athletes", f"{int(row['n_athletes']):,}")
c[1].metric("Matched (Combine + NFL)", f"{int(row['n_athletes_matched']):,}")
c[2].metric("Combine attempts", f"{int(row['n_combine_attempts']):,}")
c[3].metric("Tracked player-plays", f"{int(row['n_player_plays']):,}")
c[4].metric("Seasons", str(row["seasons_label"]))
st.caption(f"Snapshot built {row['lab_built_at']} · gold built {row['gold_built_at']} · "
           f"silver NFL frame rows {int(row['n_silver_nfl_tracking_rows']):,} (incl. football)")

st.subheader("Pages")
st.markdown(
    "- **Combine movement**: athlete → drill → attempt; playback, speed/acceleration curves, A/B comparison.\n"
    "- **NFL play movement**: season → game → play (or athlete → play); playback of every tracked entity, "
    "snap-aligned comparisons.")

st.subheader("Conventions")
st.markdown(
    "- Units: yards, yd/s, yd/s², seconds (1 yd/s ≈ 2.05 mph). Speed `s` and acceleration magnitude `a` are "
    "vendor values; `dS/dt` is derived and unsmoothed.\n"
    "- Field: x along the field incl. end zones (0–120 yd), y across (0–53.3 yd), drawn 1:1. NFL *standardised* "
    "coordinates rotate plays so the offense moves toward +x.\n"
    "- Direction arrows: distance covered in 0.5 s at the current speed; 0° = +y, clockwise (Big Data Bowl "
    "convention, assumed for Combine).\n"
    "- Only one trajectory / one play is loaded at a time from a single silver partition.")
st.markdown(" ".join(f"<span style='color:{c}'>■</span> {p}" for p, c in POSITION_COLORS.items()),
            unsafe_allow_html=True)

with st.expander("Optional inputs at build time"):
    st.dataframe(ui.guard(data.availability, mt), hide_index=True, use_container_width=True)

ui.sidebar_footer()
