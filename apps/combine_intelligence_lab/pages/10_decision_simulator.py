"""PAGE 10 — Scouting Decision Simulator: a scouting-meeting workflow from a question to an evidence-aware action and an exported report.

A simulator of the REVIEW workflow — not of a draft, a game or an outcome. Every line comes from a deterministic rule in
``decision_simulator.simulate`` applied to stored mart values.
"""

import polars as pl
import streamlit as st

import data_access as da
import reporting
from components import ui
from visualizations import charts

from nflbdb.research.advanced_movement import decision_simulator as ds
from nflbdb.research.advanced_movement import product_kit as pk

ui.chrome("Scouting Decision Simulator", "Pick a scouting question and one or two prospects. The Lab reviews the stored evidence and returns an action: "
          "what was measured, how sure it is, where it may matter, what to watch, what to test — and what cannot be concluded.")
da.require()
P = bool(st.session_state.get("present"))

st.markdown("#### Step 1 — The scouting question")
qid = st.radio("Question", list(ds.QUESTIONS), format_func=lambda q: ds.QUESTIONS[q]["text"], label_visibility="collapsed")
need_two = ds.QUESTIONS[qid]["athletes"] == 2

st.markdown("#### Step 2 — The prospect(s)")
c1, c2 = st.columns(2)
with c1:
    nid = ui.athlete_picker("Prospect A", tracked_only=False)
if nid is None:
    st.stop()
a = da.profile(nid)
twins = da.twins_of(nid)
with c2:
    opts = {None: "— none —" if not need_two else "— choose a second prospect —"}
    opts |= {t["other_id"]: f"{t['other_name']} · stopwatch twin, forty {t['d_forty']:+.2f} s · {t['n_distinguishable']} distinguishable" for t in twins[:15]}
    others = da.athlete_labels(tracked_only=False, position=a.get("position_group"))
    opts |= {i: f"{lab} · not a stored twin" for i, lab in others.items() if i not in opts and i != nid}
    keys = list(opts)
    pre = st.session_state.get("sim_b")
    b_id = st.selectbox("Prospect B (suggested: stored stopwatch twins first)", keys, index=keys.index(pre) if pre in keys else (1 if need_two and len(keys) > 1 else 0),
                        format_func=lambda k: opts[k])
    st.session_state["sim_b"] = b_id
b = da.profile(b_id) if b_id is not None else None
twin_row = next((t for t in twins if t["other_id"] == b_id), None) if b_id is not None else None
tw = da.twin(twin_row["pair_id"], nid) if twin_row else None
diffs = da.differences(tw["pair_id"], tw["flip"]) if tw else []

st.markdown("#### Step 3 — Review the evidence")
names = (da.name_of(nid), da.name_of(b_id) if b_id is not None else "")
tile = lambda p: [(f"{da.name_of(p['nfl_id'])} · forty", ui.fmt(p.get("official_forty"), "s"), f"10-yd {ui.fmt(p.get('official_ten'), 's')}"),
                  ("tracked attempts", int(p.get("n_tracked_attempts") or 0), f"recording tier {p.get('evidence_tier') or 'n/a'}")]
ui.tiles(tile(a) + (tile(b) if b else []))
if tw:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(charts.phase_chart(pk.phase_decomposition(diffs), (tw["name_a"], tw["name_b"]), P), use_container_width=True)
    with c2:
        st.plotly_chart(charts.speed_distance([(tw["name_a"], charts.A, da.curves(tw["nfl_id_a"])), (tw["name_b"], charts.B, da.curves(tw["nfl_id_b"]))],
                                              da.band(tw["position_group"]), tw["position_group"], P), use_container_width=True)
elif b is not None:
    ui.empty_state("no_twin", f"{names[0]} and {names[1]} are not stored twins: the Lab shows each profile and computes no difference between them.")
elif int(a.get("n_tracked_attempts") or 0) > 0:
    st.plotly_chart(charts.speed_distance([(names[0], charts.A, da.curves(nid))], da.band(a["position_group"]), a["position_group"], P), use_container_width=True)
else:
    ui.empty_state("no_tracked_attempt")
pos = a.get("position_group")
pm = da.table("prospect_metrics")
ref = pm.filter(pl.col("position_group") == pos)["nfl_id"].n_unique() if pm.height else 0
st.caption(f"Position reference group: {ref} tracked {pos} prospects in this mart. NFL demand rates: {pos} eligible player-plays of the 09A atlas.")

st.markdown("#### Step 4 — The scouting action")
matrix_a, _ = da.similarity(nid)
result = ds.simulate(qid, a, da.metrics(nid), b=b, metrics_b=da.metrics(b_id) if b_id is not None else (), twin=tw, diffs=diffs, matrix_a=matrix_a,
                     demands=da.demands(pos).to_dicts(), registry=da.rows("nfl_demand_registry"), evidence=da.evidence_list(), gap=da.gap())
st.markdown(f"**Decision support:** {result['decision']}")
counts = result["category_counts"]
ui.tiles([(ds.CATEGORIES[k], counts[k], "") for k in ("measured_difference", "uncertain_difference", "missing_observation", "measurement_mismatch", "additional_data_needed")])
ui.action(result)

st.markdown("#### Step 5 — Export")
cols = st.columns(3)
for col, aud in zip(cols, ("scout", "coach", "gm")):
    with col:
        st.markdown(f"**{ds.AUDIENCE_TITLE[aud]}**")
        stem = f"{aud}_{nid}" + (f"_{b_id}" if b_id is not None else "") + f"_{qid}"
        st.download_button("HTML", reporting.decision_html(result, aud), file_name=f"{stem}.html", mime="text/html", key=f"h_{aud}", type="primary")
        if st.button("Prepare PDF", key=f"p_{aud}"):
            pdf = reporting.decision_pdf(result, aud)
            if pdf is None:
                st.warning("PDF export needs matplotlib in this environment; the HTML report is available.")
            else:
                st.download_button("Download PDF", pdf, file_name=f"{stem}.pdf", mime="application/pdf", key=f"d_{aud}")
ui.note("This simulates the scouting review, not a draft or a game. It produces no success probability, no ranking and no draft label.")
ui.disclaimer()
