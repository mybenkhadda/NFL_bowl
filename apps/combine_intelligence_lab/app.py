"""COMBINE INTELLIGENCE LAB — Same stopwatch. Different movement. Better scouting questions.

Entry point (launched by the user, never by the notebooks):

    cd apps/combine_intelligence_lab
    streamlit run app.py

Reads only the application mart written by notebook 09E (see README.md). Ten pages in five groups, one shared athlete
selection, a presentation / demo mode, and a data-version indicator. Milestone 10 adds the NFL Movement Gap, the Scouting
Decision Simulator and the Signature Stories gallery, all computed from the same mart.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Combine Intelligence Lab", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")

import data_access as da  # noqa: E402  (after set_page_config)
from components import ui  # noqa: E402

PAGES = {
    "Start": [st.Page("pages/1_overview.py", title="Overview", icon=":material/dashboard:", default=True)],
    "Prospects": [st.Page("pages/2_prospect_profile.py", title="Prospect Intelligence", icon=":material/person_search:"),
                  st.Page("pages/3_stopwatch_twins.py", title="Stopwatch Twins", icon=":material/compare_arrows:"),
                  st.Page("pages/8_signature_stories.py", title="Signature Stories", icon=":material/auto_stories:")],
    "NFL movement": [st.Page("pages/4_demand_explorer.py", title="NFL Movement Demands", icon=":material/sports_football:"),
                     st.Page("pages/9_movement_gap.py", title="NFL Movement Gap", icon=":material/grid_view:"),
                     st.Page("pages/5_movement_match_lab.py", title="Movement Match Lab", icon=":material/join_inner:")],
    "Decide": [st.Page("pages/10_decision_simulator.py", title="Scouting Decision Simulator", icon=":material/fact_check:"),
               st.Page("pages/6_report_generator.py", title="Report Generator", icon=":material/description:")],
    "Evidence": [st.Page("pages/7_research_evidence.py", title="Research Evidence", icon=":material/science:")],
}
DEMO_ROUTE = (("pages/1_overview.py", "1 · Same forty — the featured pair"), ("pages/3_stopwatch_twins.py", "2 · Replay and phase differences"),
              ("pages/9_movement_gap.py", "3 · What the forty cannot see"), ("pages/10_decision_simulator.py", "4 · Generate the scouting action"),
              ("pages/8_signature_stories.py", "5 · Three stories"), ("pages/7_research_evidence.py", "6 · The evidence boundary"))

with st.sidebar:
    st.markdown(f"<div style='font-weight:800;font-size:1.2rem;letter-spacing:.02em'>COMBINE INTELLIGENCE LAB</div>"
                f"<div style='color:{ui.C['field']};font-size:.8rem;margin-bottom:12px'>Same stopwatch. Different movement. Better scouting questions.</div>",
                unsafe_allow_html=True)
    st.toggle("Presentation / demo mode", key="present", help="Larger type and charts for a presentation screen, and the demo route below.")
    m = da.manifest()
    positions = sorted(da.profiles()["position_group"].drop_nulls().unique().to_list()) if m else []
    st.selectbox("Position filter", ["All", *positions], key="position_filter", help="Restricts every athlete picker.")
    if st.session_state.get("present"):
        with st.expander("Demo route (≈ 2 minutes)", expanded=True):
            for path, label in DEMO_ROUTE:
                st.page_link(path, label=label)
    if m:
        ui.version_badge()
    else:
        st.caption("No application mart found — see the Overview page.")
    st.caption("Descriptive scouting information. No grade, no ranking, no NFL prediction.")

st.navigation(PAGES).run()
