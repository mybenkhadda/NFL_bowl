"""PAGE 9 — NFL Movement Gap: what NFL games ask of each position, and which of those movements forty tracking can physically describe."""

import polars as pl
import streamlit as st

import data_access as da
from components import ui
from visualizations import charts

from nflbdb.research.advanced_movement import movement_gap as mg

ui.chrome("NFL Movement Gap", "The forty is one straight-line acceleration from rest. NFL games add braking, redirection, re-acceleration and repeated speed changes. "
          "Here is the mismatch, position by position, with its denominators.")
da.require()
P = bool(st.session_state.get("present"))
G = da.gap()
if not G:
    ui.empty_state("no_nfl_observation", "No demand table in this mart.")
    st.stop()
SUM = mg.gap_summary(G)
focus = mg.gap_position(SUM)
st.plotly_chart(charts.gap_matrix(G, P), use_container_width=True)
st.markdown("  ·  ".join(f"<span style='color:{charts.CLASS_COLOR[k]};font-weight:700'>{charts.CLASS_MARK[k]}</span> {mg.COVERAGE_CLASSES[k]}" for k in mg.CLASS_ORDER),
            unsafe_allow_html=True)
with st.expander("How each cell is classified (no score, a rule)"):
    st.markdown("- **Directly measured** — the 09A registry names a shared forty dimension AND the episode starts in the same movement state as the forty "
                "(acceleration from low speed; transition into high speed).\n"
                "- **Partially comparable** — a shared dimension exists but the state differs (a standing-start band transit is not a rolling-start "
                "acceleration; the forty can end before top speed and records no holding phase).\n"
                "- **Not directly measured** — the registry names no forty analogue (braking, re-acceleration after braking, redirection, speed maintenance, "
                "repeated speed changes).\n"
                "- **Unavailable** — a shared dimension exists, but this position has too few observed episodes (09A support rule) or no supported prospect "
                "comparison (09B).\n\nHover a cell for its reason, counts and denominator.")

tiles = [(s["position_group"], "—" if s["share_not_measured"] is None else f"{100 * s['share_not_measured']:.0f} %",
          f"of exposure index not described · {s['n_eligible_plays']:,} player-plays") for s in SUM]
st.markdown("**Share of the observed exposure index in demands the forty does not describe**"
            + ui.info("Sum of the supported demand rates (episodes per 100 eligible player-plays) by class. Categories overlap, so this is an exposure index, "
                      "not a share of distinct movements. Frequency is exposure, not importance to performance."), unsafe_allow_html=True)
ui.tiles(tiles)
if focus:
    ui.note(f"Largest gap: {focus['position_group']} ({100 * focus['share_not_measured']:.0f} %) among positions whose supported cells all have ≥ "
            f"{mg.MIN_ATHLETES_COMPARE} athletes.")

st.subheader("One position")
positions = [s["position_group"] for s in SUM]
pos = st.selectbox("Position", positions, index=positions.index(focus["position_group"]) if focus else 0)
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(charts.gap_bars(G, pos, P), use_container_width=True)
with c2:
    props = mg.proposals(G, pos)
    ui.card(f"Candidate measurements for {pos}", [f"{p['test']} — {p['protocol']} (for {p['demand']}, {p['rate']})." for p in props] + [mg.LIMIT])
unav = [r for r in G if r["position_group"] == pos and r["coverage_class"] == mg.UNAVAILABLE]
if unav:
    ui.card("Data availability", [f"{r['label']}: {r['reason']}." for r in unav])

st.subheader("Two positions")
c1, c2, c3 = st.columns(3)
pa = c1.selectbox("Position A", positions, index=0)
pb = c2.selectbox("Position B", positions, index=min(1, len(positions) - 1))
labels = {r["demand_id"]: r["label"] for r in G}
dsel = c3.selectbox("Demand", list(labels), format_func=lambda d: labels[d])
st.plotly_chart(charts.exposure_compare(G, pa, pb, P), use_container_width=True)
cmp = mg.compare_positions(G, pa, pb, dsel)
ratio = cmp["rate_ratio_b_over_a"]
ui.card(f"{labels[dsel]} — {pa} vs {pb}", ([f"Rate ratio {pb} / {pa}: {ratio:.2f}."] if ratio else ["No rate ratio: at least one position is too thinly observed."]) + cmp["flags"])
reg = da.table("nfl_demand_registry").filter(pl.col("demand_id") == dsel)
if reg.height:
    r = reg.row(0, named=True)
    st.plotly_chart(charts.demand_distribution(da.table("nfl_demand_distributions"), dsel, r["primary_feature"], [pa, pb], r["unit"], P), use_container_width=True)
    st.caption(f"Physical intensity of '{r['label']}' ({r['primary_feature']}, {r['unit']}) — share of each position's episodes per bin. {r['physical_definition']}.")

st.subheader("One-page executive summary")
n_ep = int(da.manifest().get("overview", {}).get("n_nfl_episodes") or 0)
summ = mg.executive_summary(SUM, G, focus, n_ep)
ui.card("NFL Movement Gap — summary", summ)
st.download_button("Download the summary (Markdown)", "# NFL Movement Gap — executive summary\n\n" + "\n".join(f"- {s}" for s in summ) + "\n",
                   file_name="movement_gap_summary.md", mime="text/markdown")
ui.disclaimer()
