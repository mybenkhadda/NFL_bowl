"""PAGE 7 — Research Evidence: what is measured, what is repeatable, what is not established, and how to use the tool."""

import streamlit as st

import data_access as da
from components import ui

ui.chrome("Research Evidence", "Every statement on this page is read from an executed artifact of Milestones 04–09. Pending or absent evidence is shown as such.")
da.require()
ev = da.evidence()
claims = {c["claim_id"]: c for c in ev.get("evidence_claims_08b6", [])}


def claim(cid: str) -> str:
    c = claims.get(cid)
    return f"{c['claim']} — {c['value']}" if c and c.get("executed") else f"{cid}: not available in this mart (run 08B.6 in this run mode)"


st.subheader("In plain football language")
TOPICS = (
    ("Reliability", "Does the same athlete produce the same number twice? Measured by comparing repeat forty attempts.", ("C01", "C02", "C03")),
    ("Redundancy", "Does the number say anything the stopwatch did not? Measured by how much of it the official times explain.", ("C04", "C17")),
    ("FPCA", "A compact summary of the shape of the whole speed curve. Two shapes explain almost all of it — statistical axes, not physical traits.", ("C05",)),
    ("TCN-B", "A learned fingerprint of the whole recording. It recognises an athlete's repeat attempt, even among stopwatch twins. It is a similarity, "
               "not a measurement, and repeatable is not predictive.", ("C06", "C07")),
    ("Original NFL transfer (Milestone 06)", "Did tracked Combine features predict NFL sustained acceleration better than the official tests? No improvement was "
                                              "established; the interval includes zero.", ("C08", "C09", "C10")),
    ("Football-outcome transfer (Milestone 08B)", "Did they predict receiver separation better than the official tests? Not established — exploratory, inferred "
                                                   "source semantics.", ("C11", "C12", "C13", "C14", "C15", "C16")),
    ("Movement-demand interpretation", "NFL episodes are observed game movement — usually submaximal and driven by role and context. A Combine value inside "
                                       "that range is a physical description, not an ability to meet it.", ("C09",)),
    ("Measurement limitations", "10 Hz tracking, one maximal sprint from rest without pads; the 2 → 4 yd/s band spans few frames; a recording can end before "
                                "top speed; some vendor definitions are unresolved.", ("C19", "C20")),
    ("Scorecard abstention", "Where evidence is missing or not repeatable, the Lab says nothing rather than guessing.", ("C18",)),
)
for title, plain, ids in TOPICS:
    with st.expander(title):
        st.markdown(plain)
        st.markdown("**Executed evidence**")
        for cid in ids:
            st.markdown(f"- {claim(cid)}")

c1, c2 = st.columns(2)
with c1:
    ui.card("Why the official Combine numbers are incomplete descriptions", [
        "A forty time is one number for a whole run: two athletes can reach the same time through different speed curves.",
        claim("C04"), claim("C02")])
    ui.card("Which tracking measures are repeatable", [claim("C01"), claim("C02"), claim("C03"), claim("C17")])
    ui.card("What FPCA revealed", [claim("C05"), "Its components are statistical axes of the speed curve, not physical quantities."])
with c2:
    ui.card("What TCN-B revealed", [claim("C06"), claim("C07"), "A similarity space: no dimension has a physical meaning, and repeatable is not predictive."])
    ui.card("Why predictive transfer remains unestablished", [claim("C08"), claim("C09"), claim("C10"), claim("C14"), claim("C16")])
    ui.card("How this tool should and should not be used",
            ["Use: " + u for u in ev.get("use", [])] + ["Do not: " + u for u in ev.get("do_not_use", [])])

st.subheader("Scouting evidence registry (Milestone 08B.4)")
reg = ev.get("scouting_evidence_registry_08b4", [])
if reg:
    ui.html_table([{**r, "badge": ui.badge(r.get("status"))} for r in reg], [("label", "Characteristic"), ("raw_repeatability", "Repeatability"),
                  ("residual_repeatability", "Beyond official tests"), ("stopwatch_redundancy", "Stopwatch redundancy"), ("badge", "Status"),
                  ("nfl_predictive_evidence", "NFL predictive evidence")], raw=("badge",))
else:
    st.info("The 08B.4 registry was not available when this mart was built.")
st.subheader("Combine curve features of the Lab (Milestone 09B local screen)")
cur = ev.get("combine_curve_feature_evidence_09b", [])
ui.html_table([{**r, "badge": ui.badge(r.get("evidence_status"))} for r in cur], [("label", "Feature"), ("phase", "Phase"), ("icc_within_position", "ICC"),
              ("r2_official_times", "R² on official times"), ("n_repeat", "Athletes with a repeat"), ("badge", "Status"), ("evidence_source", "Source"),
              ("football_value", "Football value")], raw=("badge",))
with st.expander("Interpretation registry — the seven answers behind every metric"):
    ui.html_table(ev.get("interpretation_registry", []), [("metric", "Metric"), ("what_is_measured", "Measured"), ("physical_meaning", "Physical meaning"),
                  ("difference_from_official_testing", "Beyond official testing"), ("reliability", "Reliability"), ("what_to_look_at_next", "Look at next"),
                  ("what_cannot_be_concluded", "Cannot be concluded")])
with st.expander("All evidence claims (08B.6 registry)"):
    ui.html_table(list(claims.values()), [("claim_id", "ID"), ("claim", "Claim"), ("value", "Value"), ("n", "n"), ("evidence_level", "Level"), ("limitation", "Limitation")])
ui.disclaimer()
