"""PAGE 6 — Scouting Report Generator: a prospect report or a case report, as HTML (self-contained) or a one-page PDF card."""

import polars as pl
import streamlit as st
import streamlit.components.v1 as components

import data_access as da
import reporting
from components import ui
from visualizations import charts

ui.chrome("Scouting Report Generator", "Self-contained reports: prospect summary, official testing, tracking insights, matched prospects, demand correspondence, "
          "evidence confidence, film checklist, follow-up testing and limitations.")
da.require()
kind = st.radio("Report", ["Prospect report", "Case study (three views)"], horizontal=True)

if kind == "Prospect report":
    nid = ui.athlete_picker("Prospect", tracked_only=False)
    if nid is None:
        st.stop()
    p = da.profile(nid)
    sections = reporting.prospect_sections(nid)
    cols = st.columns(2)
    for k, (sec, lines) in enumerate(sections.items()):
        with cols[k % 2]:
            ui.card(sec, lines)
    att = da.curves(nid)
    figs = [("Speed by distance against the position", charts.speed_distance([(da.name_of(nid), charts.A, att)], da.band(p["position_group"]), p["position_group"]))] if att else []
    matrix, _ = da.similarity(nid)
    figs.append(("Movement-demand correspondence (descriptive)", charts.coverage_bars(matrix, da.demand_labels())))
    html = reporting.prospect_html(nid, figs)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download HTML report", html, file_name=f"prospect_{nid}.html", mime="text/html", type="primary")
    if c3.button("Prepare full PDF report"):
        from nflbdb.research.advanced_movement import decision_simulator as ds

        full = ds.sections_pdf(f"{da.name_of(nid)} — prospect intelligence report", f"{p.get('position_group')} · draft class {p.get('draft_year') or 'n/a'}",
                               sections, reporting._footer())
        if full is None:
            st.warning("PDF export needs matplotlib in this environment; the HTML report is available.")
        else:
            st.download_button("Download full PDF", full, file_name=f"prospect_{nid}.pdf", mime="application/pdf")
    if c2.button("Prepare one-page PDF card"):
        pdf = reporting.card_pdf(nid)
        if pdf is None:
            st.warning("The PDF card needs matplotlib and the project package (`src/nflbdb`) importable in this environment; the HTML report is available.")
        else:
            st.download_button("Download PDF card", pdf, file_name=f"intelligence_card_{nid}.pdf", mime="application/pdf")
    with st.expander("Preview the HTML report"):
        components.html(html, height=900, scrolling=True)
else:
    cases = da.table("scouting_cases")
    if not cases.height:
        st.info("No case study in this mart.")
        st.stop()
    cid = st.selectbox("Case", cases["case_id"].to_list(), format_func=lambda k: (lambda r: f"{r['case_label']} — {r['name_a']} vs {r['name_b']} ({r['position_group']})")(
        cases.filter(pl.col("case_id") == k).row(0, named=True)))
    c = cases.filter(pl.col("case_id") == cid).row(0, named=True)
    tw = da.twin(c["pair_id"], c["nfl_id_a"])
    diffs = da.differences(c["pair_id"], tw.get("flip", False))
    views = da.case_views(cid)
    ui.note(f"Selected by a pre-draft rule — matched on: {c['matched_on']}. Contrast: {c['contrast']}. NFL outcome used for selection: no"
            + ("; NFL exposure count used for eligibility: yes" if c["selected_using_nfl_exposure"] else "") + ".")
    tabs = st.tabs(["Scout view", "Position-coach view", "General-manager view"])
    for tab, key in zip(tabs, ("scout", "coach", "general_manager")):
        with tab:
            for sec, lines in views.get(key, {}).items():
                ui.card(sec, lines)
    obs = da.table("case_nfl_observed").filter(pl.col("case_id") == cid)
    if obs.height:
        st.caption("Observed NFL movement — shown after the selection, never used for it.")
        ui.html_table(obs.to_dicts(), [("name", "Athlete"), ("outcome_id", "NFL movement measure"), ("value", "Value"), ("unit", "Unit"), ("n_windows", "Windows")])
    names = (tw["name_a"], tw["name_b"])
    figs = [("Speed by distance", charts.speed_distance([(names[0], charts.A, da.curves(tw["nfl_id_a"])), (names[1], charts.B, da.curves(tw["nfl_id_b"]))],
                                                       da.band(tw["position_group"]), tw["position_group"])), ("Differences against their uncertainty", charts.forest(diffs, names))]
    st.download_button("Download case report (HTML)", reporting.pair_html(tw, diffs, figs, views), file_name=f"case_{cid}.html", mime="text/html", type="primary")
ui.disclaimer()
