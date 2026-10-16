"""PAGE 5 — Movement Match Lab: where a prospect's measured Combine movement sits inside the NFL demands of his position."""

import polars as pl
import streamlit as st

import data_access as da
from components import ui
from visualizations import charts

ui.chrome("Movement Match Lab", "Descriptive physical similarity between a maximal-effort Combine value and observed NFL game demands — not a prediction, "
          "and not a validated ability to meet the demand.")
da.require()
P = bool(st.session_state.get("present"))
nid = ui.athlete_picker("Prospect")
if nid is None:
    st.stop()
p = da.profile(nid)
labels = da.demand_labels()
matrix, sim = da.similarity(nid)
left, right = st.columns([1, 1])
with left:
    st.plotly_chart(charts.coverage_bars(matrix, labels, P), use_container_width=True)
with right:
    unsupported = [r for r in sim if r["state"] == "unsupported_by_forty"]
    ui.card("Not described by the forty-yard dash (abstains)", [f"{labels.get(r['demand_id'], r['demand_id'])}: {r['reason']}" for r in unsupported])
if not matrix:
    ui.empty_state("no_tracked_attempt", "No demand correspondence can be computed without a tracked attempt.")
    st.stop()
mt = [{**r, "demand": labels.get(r["demand_id"], r["demand_id"]), "cov": None if r.get("mean_coverage") is None else f"{r['mean_coverage']:.0f} %",
       "iv": None if r.get("coverage_lo") is None else f"[{r['coverage_lo']:.0f}, {r['coverage_hi']:.0f}]",
       "state_txt": {"supported": "compared", "unsupported_by_forty": "no forty analogue — abstains"}.get(r["state"], str(r["state"]).replace("_", " "))} for r in matrix]
ui.html_table(mt, [("demand", "NFL demand"), ("n_dimensions", "Shared dimensions"), ("cov", "Mean coverage"), ("iv", "Interval"), ("distance", "Distance (RMS robust z)"),
                   ("state_txt", "State")])
ui.note("Distance = root mean square over the shared dimensions of (Combine value − NFL median) / (NFL IQR / 1.349); 0 = at the NFL median of every dimension. "
        "Coverage = share of the position's episodes "
        "at or below it. Both are physical descriptions of one maximal sprint against usually submaximal game movement — not a probability of meeting the demand.")
demand = st.selectbox("NFL movement demand", [r["demand_id"] for r in matrix if r["state"] != "unsupported_by_forty"] or [None], format_func=lambda d: labels.get(d, str(d)))
rows = [r for r in sim if r["demand_id"] == demand]
st.plotly_chart(charts.alignment(rows, labels, P), use_container_width=True)
tab = [{**r, "interval": None if r.get("coverage_lo") is None else f"[{r['coverage_lo']:.0f}, {r['coverage_hi']:.0f}]"} for r in rows]
ui.html_table(tab, [("dimension", "Shared dimension"), ("combine_value", "Combine value"), ("standard_error", "SE"), ("nfl_q10", "NFL 10th"), ("nfl_q50", "NFL median"),
                    ("nfl_q90", "NFL 90th"), ("n_nfl_episodes", "NFL episodes"), ("coverage", "Coverage %"), ("interval", "Interval"), ("state", "State"), ("reason", "Reading")])
ui.note("Coverage = share of this position's NFL episodes of the demand at or below the prospect's Combine value. The forty is a maximal effort from rest; "
        "game movement is context-driven and usually submaximal. A high coverage does not mean the demand is met in pads, in traffic or late in a drive.")

ex = da.table("nfl_demand_examples")
ex = ex.filter((pl.col("demand_id") == demand) & (pl.col("position_group") == p.get("position_group"))) if ex.height and demand else pl.DataFrame()
if ex.height:
    st.subheader("Comparable NFL movement examples (same position, same demand)")
    eid = st.selectbox("Example", ex["example_id"].unique(maintain_order=True).to_list())
    st.plotly_chart(charts.episode_replay(ex.filter(pl.col("example_id") == eid), P), use_container_width=True)
st.subheader("Matching across positions")
st.plotly_chart(charts.summary_heatmap(da.table("movement_matching_summary"), P), use_container_width=True)
with st.expander("Shared feature dictionary — what is compared, and what is not"):
    ui.html_table(da.table("shared_feature_dictionary").to_dicts(), [("dimension", "Dimension"), ("combine_definition", "Combine"), ("nfl_definition", "NFL"),
                                                                    ("comparable", "Comparable"), ("caveat", "Caveat")])
ui.disclaimer()
