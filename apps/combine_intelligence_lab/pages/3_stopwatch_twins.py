"""PAGE 3 — Stopwatch Twins (signature view): prospects the official forty cannot separate, and what 10 Hz tracking measures between them."""

import html

import streamlit as st

import data_access as da
import reporting
from components import ui
from visualizations import charts

from nflbdb.research.advanced_movement import product_kit as pk
from nflbdb.research.advanced_movement import scouting_interpretation as sin

ui.chrome("Stopwatch Twins", "Same position, official forty within 0.03 s — and whether tracking separates them beyond its own measurement uncertainty.")
da.require()
P = bool(st.session_state.get("present"))
nid = ui.athlete_picker("Prospect (athlete A)")
if nid is None:
    st.stop()

with st.expander("Matching criteria and filters", expanded=False):
    st.markdown("**Stored scientific definition (09C):** same position group; official forty within **0.03 s**; differences tested against the repeat-attempt "
                "variation of both athletes (distinguishable = 95 % interval of the difference excludes 0). The filters below only **narrow** the stored pairs — "
                "they never widen the caliper or re-match athletes.")
    f1, f2, f3 = st.columns(3)
    forty_tol = f1.slider("Forty-time tolerance [s]", 0.0, da.STORED_FORTY_CALIPER, da.STORED_FORTY_CALIPER, 0.01)
    use_ten = f1.toggle("Require the 10-yard split too", value=False)
    ten_tol = f1.slider("Split tolerance [s]", 0.0, da.STORED_TEN_CALIPER, da.STORED_TEN_CALIPER, 0.01, disabled=not use_ten)
    same_class = f2.toggle("Same draft class only", value=False)
    years = sorted({y for y in da.profiles()["draft_year"].drop_nulls().to_list()})
    classes = f2.multiselect("Draft class of athlete B", years, default=[])
    tier = f3.selectbox("Measurement confidence (both recordings)", ["any", "medium", "high"], format_func=lambda t: {"any": "any tier", "medium": "medium or high",
                                                                                                                      "high": "high only"}[t])
    feats = {r["feature"]: r["label"] for r in da.evidence_list()}
    feature = f3.selectbox("Distinguishable on feature", [None, *feats], format_func=lambda f: "any feature" if f is None else feats[f])
    only_diff = f3.toggle("Only pairs with ≥ 1 distinguishable difference", value=False)

twins = da.twins_filtered(nid, forty_tol, ten_tol if use_ten else None, same_class, classes or None, tier, feature)
if only_diff:
    twins = [t for t in twins if t["n_distinguishable"] >= 1]
if not twins:
    ui.empty_state("no_twin")
    st.stop()

st.markdown(f"**{len(twins)} comparable prospect(s)** — ranked by the largest measured difference relative to its uncertainty (|z|).")
rank = [{"other": t["other_name"], "forty": t["forty_b"], "d_forty": t["d_forty"], "d_ten": t.get("d_ten"), "class": t.get("draft_year_b"),
         "tiers": f"{t.get('tier_a') or 'n/a'} · {t.get('tier_b') or 'n/a'}", "dist": f"{t['n_distinguishable']} of {t.get('n_features_compared') or '—'}",
         "top": (da.evidence_rows().get(t.get("top_feature") or "", {}) or {}).get("label", t.get("top_feature") or "—")} for t in twins[:10]]
ui.html_table(rank, [("other", "Comparable prospect"), ("forty", "Forty [s]"), ("d_forty", "Δ forty (B − A)"), ("d_ten", "Δ 10-yd"), ("class", "Class"),
                     ("tiers", "Recording tiers A · B"), ("dist", "Distinguishable"), ("top", "Largest difference")])
ids = [t["pair_id"] for t in twins]
lab = {t["pair_id"]: f"{t['other_name']} — forty {t['forty_b']:.2f} s ({t['d_forty']:+.2f}) · {t['n_distinguishable']} distinguishable" for t in twins}
cur = st.session_state.get("pair_id")
pair = st.selectbox("Comparable prospect (athlete B)", ids, index=ids.index(cur) if cur in ids else 0, format_func=lambda k: lab[k])
st.session_state["pair_id"] = pair
tw = da.twin(pair, nid)
diffs = da.differences(pair, tw["flip"])
names = (tw["name_a"], tw["name_b"])
ca, cb = da.curves(tw["nfl_id_a"]), da.curves(tw["nfl_id_b"])

left, right = st.columns([1, 2])
with left:
    ui.versus(names[0], names[1], tw["forty_a"], tw["forty_b"])
    ui.tiles([("10-yard split", f"{ui.fmt(tw['ten_a'], 's')} · {ui.fmt(tw['ten_b'], 's')}", "within caliper" if tw.get("ten_matched") else "not matched / not recorded"),
              ("Weight Δ", ui.fmt(tw.get("d_weight"), "lb", 0), "B − A"),
              ("Recording tiers", f"{tw['tier_a'] or 'n/a'} · {tw['tier_b'] or 'n/a'}", "measurement quality"),
              ("TCN-B distance", tw.get("fingerprint_distance"), "mathematical similarity, no unit")])
    if tw.get("ten_a") is None or tw.get("ten_b") is None:
        ui.empty_state("missing_split")
with right:
    c1, c2, c3, c4 = st.columns(4)
    mode = c1.radio("Compare at the same", ["time", "distance"], horizontal=True)
    metric = c2.radio("Metric", ["speed", "accel"], horizontal=True, format_func=lambda m: {"speed": "speed", "accel": "acceleration"}[m])
    ia = c3.selectbox(f"{names[0]} attempt", [a["attempt"] for a in ca] or [None])
    ib = c3.selectbox(f"{names[1]} attempt", [a["attempt"] for a in cb] or [None])
    speed = c4.select_slider("Playback", [0.25, 0.5, 1.0, 2.0], value=1.0, format_func=lambda x: f"{x:g}×")
    pick = lambda att, k: next((a for a in att if a["attempt"] == k), None)
    fig, disclosure = charts.synced_replay([(names[0], charts.A, pick(ca, ia)), (names[1], charts.B, pick(cb, ib))], mode, metric, speed, P)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(disclosure)

st.subheader("Where the sprints differ — phase by phase")
decomp = pk.phase_decomposition(diffs)
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(charts.phase_chart(decomp, names, P), use_container_width=True)
with c2:
    ui.card("Phase reading", [pk.phase_sentence(p, names) for p in decomp])
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(charts.speed_distance([(names[0], charts.A, ca), (names[1], charts.B, cb)], da.band(tw["position_group"]), tw["position_group"], P), use_container_width=True)
with c2:
    st.plotly_chart(charts.band_time(diffs, names, P), use_container_width=True)

st.subheader("Reliable versus uncertain — and what the stopwatch already carries")
ev = da.evidence_rows()
rows = []
for d in diffs:
    lab_e = pk.evidence_label(ev.get(d["feature"]))
    rows.append({**d, "label_html": html.escape(str(d["label"])) + ui.info(lab_e["tooltip"]), "verdict_html": ui.verdict_badge(d["verdict"]),
                 "interval": None if d["ci_lower"] is None else f"[{d['ci_lower']:+.2f}, {d['ci_upper']:+.2f}]",
                 "redundancy": lab_e["redundancy"], "usable": "yes" if pk.usable(d) else "not repeatable enough"})
ui.html_table(rows, [("label_html", "Feature"), ("unit", "Unit"), ("value_a", names[0]), ("value_b", names[1]), ("difference", "B − A"), ("interval", "95 % interval"),
                     ("verdict_html", "Verdict"), ("usable", "Describes an athlete?"), ("redundancy", "Redundancy with official tests")], raw=("label_html", "verdict_html"))
sa, sb = (pk.attempt_stability(da.metrics(i)) for i in (tw["nfl_id_a"], tw["nfl_id_b"]))
c1, c2 = st.columns(2)
for col, n, s in ((c1, names[0], sa), (c2, names[1], sb)):
    with col:
        ui.card(f"Repeat-attempt stability — {n}", [f"{x['label']}: {x['reading']}" for x in s if x["value"] is not None] or [pk.NO_DATA["no_tracked_attempt"]])

st.subheader("NFL movement-demand correspondence (physical, descriptive)")
dem = {r["demand_id"]: r for r in da.demands(tw["position_group"]).iter_rows(named=True)}
phases = list(dict.fromkeys(d["phase"] for d in diffs if d["verdict"] == "distinguishable" and pk.usable(d) and d["phase"] in sin.PHASES and d["phase"] != "whole"))
lines = []
for ph in phases:
    r = dem.get(sin.PHASES[ph]["demand"])
    if r and r["supported"]:
        lines.append(f"{sin.PHASES[ph]['label'].capitalize()} → {r['label']}: {r['episodes_per_100_plays']:.1f} episodes per 100 eligible {tw['position_group']} "
                     f"player-plays ({r['n_episodes']:,} episodes, {r['n_athletes']} athletes).")
    else:
        lines.append(f"{sin.PHASES[ph]['label'].capitalize()}: {pk.NO_DATA['no_nfl_observation']}")
ui.card("Measured difference → closest observed NFL demand", lines + [sin.NOT_ESTABLISHED] if lines else ["No distinguishable difference, so no demand to relate."])

st.subheader("Comparison summary — generated from the measured values")
read = sin.pair_reading(tw, diffs)
cols = st.columns(2)
for k, (sec, ls) in enumerate(read.items()):
    with cols[k % 2]:
        ui.card(sec, ls)
c1, c2 = st.columns(2)
html_doc = reporting.pair_html(tw, diffs, [("Speed by distance", charts.speed_distance([(names[0], charts.A, ca), (names[1], charts.B, cb)], da.band(tw["position_group"]),
                                                                                      tw["position_group"])),
                                           ("Phase decomposition", charts.phase_chart(decomp, names)), ("Differences against their uncertainty", charts.forest(diffs, names))])
c1.download_button("Download this comparison (HTML)", html_doc, file_name=f"stopwatch_twins_{pair}.html", mime="text/html", type="primary")
if c2.button("Take this pair to the decision simulator →"):
    st.session_state["sim_b"] = tw["nfl_id_b"]
    st.switch_page("pages/10_decision_simulator.py")
ui.disclaimer()
