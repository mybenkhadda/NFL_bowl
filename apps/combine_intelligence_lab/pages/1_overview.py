"""PAGE 1 — Overview: one signature comparison, the three questions the Lab answers, and what the data cover."""

import polars as pl
import streamlit as st

import data_access as da
from components import ui
from visualizations import charts

ui.chrome("Same stopwatch. Different movement.", "The stopwatch tells you how fast. 10 Hz tracking shows how the athlete got there. "
          "The Lab turns that difference into better scouting questions.", kicker="Combine Intelligence Lab · NFL Big Data Bowl 2027")
m = da.require()
ov = m.get("overview", {})
P = bool(st.session_state.get("present"))

q1, q2, q3 = st.columns(3)
with q1:
    ui.story_card("Question 1", "What does the stopwatch hide?", "Repeatable tracking differences between athletes the official forty cannot separate.")
with q2:
    ui.story_card("Question 2", "Where might it matter in football?", "Observed NFL movement demands of the same position that are physically comparable — and those the forty cannot see.")
with q3:
    ui.story_card("Question 3", "What should the scout do next?", "Specific film situations and targeted tests — with what cannot be concluded stated beside them.")
st.write("")

story = da.stories().get("same_forty")
cases = da.table("scouting_cases")
if story:
    pid = story["provenance"]["pair_id"]
    raw = da.table("stopwatch_twins").filter(pl.col("pair_id") == pid).row(0, named=True)
    tw = da.twin(pid, raw["nfl_id_a"])
else:
    c0 = cases.filter(pl.col("case_type") == "same_forty_different_curve").head(1)
    tw = da.twin(c0["pair_id"][0], c0["nfl_id_a"][0]) if c0.height else {}
if tw:
    diffs = da.differences(tw["pair_id"], tw.get("flip", False))
    names = (tw["name_a"], tw["name_b"])
    st.subheader(f"Signature comparison — {names[0]} and {names[1]} ({tw['position_group']})")
    left, right = st.columns([1, 2])
    with left:
        ui.versus(names[0], names[1], tw["forty_a"], tw["forty_b"])
        lead = [d for d in diffs if d["verdict"] == "distinguishable" and d.get("evidence_status") not in ("unstable", "insufficient_support")]
        ui.card("What 10 Hz tracking adds", [f"{d['label']}: {d['value_a']:.2f} vs {d['value_b']:.2f} {d['unit']} — beyond measurement uncertainty" for d in lead[:3]]
                or ["No difference beyond measurement uncertainty: on the recorded evidence these runs are alike."])
        if st.button("Open Stopwatch Twins with this pair →", type="primary"):
            st.session_state["athlete_id"], st.session_state["pair_id"] = tw["nfl_id_a"], tw["pair_id"]
            st.switch_page("pages/3_stopwatch_twins.py")
    with right:
        series = [(names[0], charts.A, da.curves(tw["nfl_id_a"])), (names[1], charts.B, da.curves(tw["nfl_id_b"]))]
        st.plotly_chart(charts.speed_distance(series, da.band(tw["position_group"]), tw["position_group"], P), use_container_width=True)
else:
    ui.empty_state("no_twin", "No case study of type 'same forty, different curve' in this mart.")

st.subheader("What the data cover")
ui.tiles([("Combine prospects", ov.get("n_prospects"), f"{ov.get('n_tracked_prospects', 0):,} with tracked forty attempts"),
          ("Tracked attempts", ov.get("n_tracked_attempts"), "10 Hz forty-yard dashes"),
          ("Stopwatch-twin pairs", ov.get("n_twin_pairs"), f"{ov.get('n_twin_pairs_distinguishable', 0):,} with ≥ 1 measurable difference"),
          ("NFL movement episodes", ov.get("n_nfl_episodes"), f"{ov.get('n_demands_measurable', 0)} demand categories"),
          ("Positions", " · ".join(ov.get("positions", [])) or "—", "position groups in the warehouse"),
          ("Case studies", ov.get("n_cases"), f"{ov.get('n_case_types', 0)} declared types, chosen without NFL outcomes")])
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(charts.twin_share(da.table("twin_summary"), da.demand_labels() | {r["feature"]: r["label"] for r in da.evidence_list()}, P), use_container_width=True)
with c2:
    reg = da.table("nfl_demand_registry")
    ui.html_table([{"label": r["label"], "analogue": r["combine_analogue"] or "none in the forty", "n": r["n_episodes"]} for r in reg.iter_rows(named=True)],
                  [("label", "NFL movement demand"), ("analogue", "Forty-tracking analogue"), ("n", "Episodes")])

st.subheader("How the evidence reads")
a, b, c3 = st.columns(3)
with a:
    ui.card("Measured and repeatable", ["Tracked top speed and speed development repeat between attempts.",
                                         "Speed development stays repeatable after the official times are accounted for."])
with b:
    ui.card("Not established", ["No tracked Combine feature improved held-out prediction of NFL movement (Milestone 06) or receiver separation (Milestone 08B).",
                                "Similarity to an NFL demand is a physical description, not a forecast."])
with c3:
    ui.card("Use it to", ["Find prospects the stopwatch cannot separate.", "Locate the movement phase in which they differ.",
                          "Decide which film to watch and which test to run next."])
ui.disclaimer()
