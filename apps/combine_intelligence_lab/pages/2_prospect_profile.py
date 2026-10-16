"""PAGE 2 — Prospect Intelligence: official tests, every recorded 10 Hz attempt, measurement uncertainty and position context."""

import html

import streamlit as st

import data_access as da
from components import ui
from visualizations import charts

from nflbdb.research.advanced_movement import product_kit as pk

ui.chrome("Prospect Intelligence", "Official tests beside the recorded sprint: where speed develops, how repeatable each measure is, and what it may be used for.")
da.require()
P = bool(st.session_state.get("present"))
nid = ui.athlete_picker("Search a prospect", tracked_only=False)
if nid is None:
    st.stop()
p = da.profile(nid)
att = da.curves(nid)
name = da.name_of(nid)
ui.tiles([("Position · class", f"{p.get('position_group')} · {p.get('draft_year') or 'n/a'}", name),
          ("Official forty", ui.fmt(p.get("official_forty"), "s"), "stopwatch"),
          ("10-yard split", ui.fmt(p.get("official_ten"), "s"), "stopwatch" if p.get("official_ten") is not None else "not recorded"),
          ("Weight · height", f"{ui.fmt(p.get('weight'), 'lb', 0)} · {ui.fmt(p.get('height'), 'in', 1)}", "anthropometrics"),
          ("Tracked attempts", len(att), "10 Hz recordings"),
          ("Recording evidence", p.get("evidence_tier") or "n/a", (p.get("evidence_reasons") or "about the measurement, not the athlete")[:60])])
if p.get("official_ten") is None:
    ui.empty_state("missing_split")
if not att:
    ui.empty_state("no_tracked_attempt")
    ui.disclaimer()
    st.stop()
if p.get("evidence_tier") in ("low", "medium"):
    ui.empty_state("low_confidence")

ctx = [("Speed capacity (official forty)", p.get("speed_capacity_pct"), p.get("speed_capacity_lower"), p.get("speed_capacity_upper")),
       ("Tracked top speed", p.get("top_speed_pct"), p.get("top_speed_lower"), p.get("top_speed_upper")),
       ("Speed development 4 → 8 yd/s", p.get("speed_development_pct"), p.get("speed_development_lower"), p.get("speed_development_upper"))]
st.markdown("**Relative position context** — oriented percentile within the tracked prospects of the same position (07D scorecard; 50 = middle)"
            + ui.info("Percentiles are relative to this dataset's tracked prospects of the same position, with the 07D interval. A percentile is a description, "
                      "not a grade."), unsafe_allow_html=True)
ui.tiles([(lab, None if v is None else f"{v:.0f}", "not available" if v is None else ("" if lo is None else f"interval {lo:.0f}–{hi:.0f}")) for lab, v, lo, hi in ctx])

t1, t2, t3, t4, t5 = st.tabs(["Speed vs distance", "Speed vs time", "Acceleration vs speed", "Sprint replay", "Measures, uncertainty & attempts"])
with t1:
    st.plotly_chart(charts.speed_distance([(name, charts.A, att)], da.band(p["position_group"]), p["position_group"], P), use_container_width=True)
    ui.note("Every recorded attempt (first in bold) over the 10–90 % range of the position's recorded runs. The band 4–8 yd/s is where speed development is measured.")
with t2:
    st.plotly_chart(charts.speed_time([(name, charts.A, att)], P), use_container_width=True)
    ui.note("Clock starts at each recording's first frame. A recording can start after the first movement; the official forty, not the curve, is the timed result.")
with t3:
    st.plotly_chart(charts.accel_speed([(name, charts.A, att)], P), use_container_width=True)
    ui.note("Second derivative of a 10 Hz position: read the level within a speed band, not single frames. Filter-edge frames are removed.")
with t4:
    if len(att) >= 2:
        mode = st.radio("Compare attempts at the same", ["time", "distance"], horizontal=True, key="pp_mode")
        fig, disclosure = charts.synced_replay([(f"attempt {a['attempt']}", charts.A if i == 0 else charts.B, a) for i, a in enumerate(att[:2])], mode, "speed", 1.0, P)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(disclosure)
    else:
        ui.empty_state("single_attempt")
with t5:
    ev = da.evidence_rows()
    stab = {s["feature"]: s for s in pk.attempt_stability(da.metrics(nid))}
    rows = []
    for mt in da.metrics(nid):
        lab = pk.evidence_label(ev.get(mt["feature"]))
        rows.append({**mt, "label_html": html.escape(str(mt["label"])) + ui.info(lab["tooltip"]), "evidence": lab["short"],
                     "interval": None if mt.get("standard_error") is None else f"± {1.96 * mt['standard_error']:.2f}", "attempts": stab[mt["feature"]]["reading"]})
    ui.html_table(rows, [("label_html", "Measure"), ("value", "Value"), ("unit", "Unit"), ("interval", "95 % interval"), ("attempts", "Repeat attempts"),
                         ("pct_position", "Raw position percentile"), ("evidence", "Evidence")], raw=("label_html",))
    st.plotly_chart(charts.metric_bars(da.metrics(nid), P), use_container_width=True)
    ui.note("Hover ⓘ for repeatability, redundancy with official testing, NFL transfer evidence and coverage. Percentiles are raw (for times a LOWER value is quicker). "
            "Repeatable does not mean valuable on the field.")
c1, c2 = st.columns(2)
if c1.button("Find stopwatch twins of this prospect →", type="primary"):
    st.switch_page("pages/3_stopwatch_twins.py")
if c2.button("Open in the decision simulator →"):
    st.switch_page("pages/10_decision_simulator.py")
ui.disclaimer()
