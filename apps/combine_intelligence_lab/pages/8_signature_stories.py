"""PAGE 8 — Signature Stories: three football stories selected by declared rules (notebook 10B rules, same code), plus the 09D case gallery."""

import json

import polars as pl
import streamlit as st

import data_access as da
from components import ui
from visualizations import charts

from nflbdb.research.advanced_movement import product_kit as pk
from nflbdb.research.advanced_movement import signature_stories as ss

ui.chrome("Signature Stories", "Three real comparisons chosen by pre-declared rules — never by how the athletes later performed in the NFL.")
da.require()
P = bool(st.session_state.get("present"))
S = da.stories()
KINDS = (("wr", "Story 1 · Wide receiver", "Same speed, different development"), ("same_forty", "Story 2 · Same forty", "Same forty, different curve"),
         ("gap", "Story 3 · Movement gap", "What the forty cannot see"))

cols = st.columns(3)
for col, (k, kicker, fallback_title) in zip(cols, KINDS):
    with col:
        s = S.get(k)
        ui.story_card(kicker, s["title"] if s else fallback_title, s["summary_100"] if s else f"Not available: {S['selection'].get(k, 'no eligible case')}.")
st.write("")

tabs = st.tabs([kicker for _, kicker, _ in KINDS] + ["All 09D case studies"])
for tab, (k, kicker, _) in zip(tabs, KINDS):
    with tab:
        s = S.get(k)
        if not s:
            ui.empty_state("no_twin" if k != "gap" else "no_nfl_observation", f"Selection: {S['selection'].get(k, '')}.")
            continue
        st.markdown(f"### {s['title']}")
        st.caption(f"Selection: {S['selection'].get(k, '')}")
        if k == "gap":
            pos = s["card"]["position"]
            st.plotly_chart(charts.gap_bars(da.gap(), pos, P), use_container_width=True)
            c1, c2 = st.columns(2)
            for i, (head, lines) in enumerate(s["sections"].items()):
                with (c1 if i % 2 == 0 else c2):
                    ui.card(head, lines)
        else:
            pid = s["provenance"]["pair_id"]
            raw = da.table("stopwatch_twins").filter(pl.col("pair_id") == pid).row(0, named=True)
            tw = da.twin(pid, raw["nfl_id_a"])
            diffs = da.differences(pid, tw["flip"])
            names = (tw["name_a"], tw["name_b"])
            ca, cb = da.curves(tw["nfl_id_a"]), da.curves(tw["nfl_id_b"])
            left, right = st.columns([1, 2])
            with left:
                ui.versus(names[0], names[1], tw["forty_a"], tw["forty_b"])
                card = s["card"]
                ui.html_table([{"k": "10-yard split", "a": tw.get("ten_a"), "b": tw.get("ten_b")}, {"k": "weight [lb]", "a": card["weight"][0], "b": card["weight"][1]},
                               {"k": "tracked attempts", "a": card["tracked_attempts"][0], "b": card["tracked_attempts"][1]},
                               {"k": "recording tier", "a": card["evidence_tier"][0], "b": card["evidence_tier"][1]}],
                              [("k", ""), ("a", names[0]), ("b", names[1])])
                st.markdown(f"**Distinguishable:** {card['distinguishable']}  \n**Largest:** {card['largest'] or '—'}")
            with right:
                st.plotly_chart(charts.speed_distance([(names[0], charts.A, ca), (names[1], charts.B, cb)], da.band(tw["position_group"]), tw["position_group"], P),
                                use_container_width=True)
            st.plotly_chart(charts.phase_chart(pk.phase_decomposition(diffs), names, P), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            ui.card("Scouting follow-up checklist", s["checklist"])
        with c2:
            ui.card("30-second judge narration", [s["narration_30s"]])
        with st.expander("Technical interpretation"):
            for t in s["technical"]:
                st.markdown(f"- {t}")
        with st.expander("Provenance"):
            st.code(json.dumps(s["provenance"], indent=2, default=str), language="json")
        st.download_button("Download this story (Markdown)", ss.story_markdown(s), file_name=f"story_{k}.md", mime="text/markdown", key=f"dl_{k}")

with tabs[-1]:
    cases = da.table("scouting_cases")
    if not cases.height:
        ui.empty_state("no_twin", "No case study in this mart.")
    else:
        st.caption("The 23 cases of notebook 09D, selected by pre-draft rules. Open any of them as a three-view report in the Report Generator.")
        ui.html_table(cases.select("case_label", "name_a", "name_b", "position_group", "matched_on", "contrast").to_dicts(),
                      [("case_label", "Case type"), ("name_a", "Athlete A"), ("name_b", "Athlete B"), ("position_group", "Position"), ("matched_on", "Matched on"),
                       ("contrast", "Contrast")])
        if st.button("Open the Report Generator →"):
            st.switch_page("pages/6_report_generator.py")
ui.disclaimer()
