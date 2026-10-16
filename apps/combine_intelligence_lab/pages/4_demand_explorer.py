"""PAGE 4 — NFL Movement Demand Explorer: which speed-change demands occur in games, how often and how intensely."""

import polars as pl
import streamlit as st

import data_access as da
from components import ui
from visualizations import charts

ui.chrome("NFL Movement Demand Explorer", "Physical speed-change episodes observed in NFL tracking, by position. Observed game movement — not a requirement, "
          "not a maximum, and no route or assignment is inferred from a movement shape.")
da.require()
P = bool(st.session_state.get("present"))
dem = da.table("nfl_movement_demands")
reg = da.table("nfl_demand_registry")
labels = da.demand_labels()
st.plotly_chart(charts.demand_heatmap(dem, labels, P), use_container_width=True)

positions = sorted(p for p in dem["position_group"].unique().to_list() if p != "ALL")
c1, c2, c3 = st.columns([1, 1, 2])
demand = c1.selectbox("Movement demand", reg["demand_id"].to_list(), format_func=lambda d: labels.get(d, d))
r = reg.filter(pl.col("demand_id") == demand).row(0, named=True)
feats = [f.strip() for f in r["features"].split(",")]
feature = c2.selectbox("Dimension", feats, index=feats.index(r["primary_feature"]) if r["primary_feature"] in feats else 0)
chosen = c3.multiselect("Positions", positions, default=positions)
left, right = st.columns([2, 1])
prof = da.table("nfl_demand_profiles").filter((pl.col("demand_id") == demand) & (pl.col("feature") == feature))
unit = prof["unit"][0] if prof.height else ""
with left:
    st.plotly_chart(charts.demand_distribution(da.table("nfl_demand_distributions"), demand, feature, chosen, unit, P), use_container_width=True)
with right:
    ui.card(r["label"], [r["physical_definition"], f"Forty-yard dash analogue: {r['combine_analogue']}.", r["comparability"], "Limitation: " + r["limitation"]])
exp = dem.filter((pl.col("demand_id") == demand) & pl.col("position_group").is_in(chosen + ["ALL"])).sort("position_group").to_dicts()
st.markdown("**Exposure** — the denominator is every eligible player-play of the position (05A rule), so rates compare across positions; raw counts do not.")
ui.html_table([{**r, "rate": f"{r['episodes_per_100_plays']:.1f}", "share": None if r.get("share_plays_with") is None else f"{100 * r['share_plays_with']:.1f} %",
                "supported_txt": "yes" if r["supported"] else "too few"} for r in exp],
              [("position_group", "Position"), ("n_episodes", "Episodes"), ("n_eligible_plays", "Eligible player-plays"), ("rate", "Per 100 player-plays"),
               ("share", "Player-plays with ≥ 1"), ("n_athletes", "Athletes"), ("median_duration_s", "Median duration [s]"),
               ("median_distance_yd", "Median distance [yd]"), ("supported_txt", "Described")])
ui.note("Categories overlap: a re-acceleration is also an acceleration episode, and a repeated-change sequence contains episodes of other categories — rates of "
        "different demands do not add up to a share of plays. Pick the dimension 'v_start' above to see the starting-speed state of the episodes.")
rows = prof.filter(pl.col("position_group").is_in(chosen + ["ALL"])).sort("position_group").to_dicts()
ui.html_table(rows, [("position_group", "Position"), ("n_episodes", "Episodes"), ("n_athletes", "Athletes"), ("q10", "10th"), ("q50", "Median"), ("q90", "90th"),
                     ("within_athlete_sd", "Within-athlete SD"), ("between_athlete_sd", "Between-athlete SD"), ("athlete_share", "Athlete share"), ("supported", "Supported")])
ui.note("Quantiles are over episodes, which are dependent observations; the athlete count is the sample size. 'Athlete share' is how much of an episode value "
        "belongs to the athlete rather than to the play.")

st.subheader("Episode examples — real plays chosen by rule")
ex = da.table("nfl_demand_examples")
ex = ex.filter((pl.col("demand_id") == demand) & pl.col("position_group").is_in(chosen)) if ex.height else ex
if ex.height:
    ids = ex["example_id"].unique(maintain_order=True).to_list()
    eid = st.selectbox("Example episode", ids, help="The episode closest to the stated quantile of its position × demand cell.")
    one = ex.filter(pl.col("example_id") == eid)
    st.caption(f"Game {one['game_id'][0]} · play {one['play_id'][0]} · season {one['season'][0]} · intensity {one['primary_value'][0]:.2f} {r['unit']}")
    st.plotly_chart(charts.episode_replay(one, P), use_container_width=True)
    ui.note("Path = displacement from the start of the episode (no field landmark is reconstructed). Gold band = the episode; the rest is the recorded margin.")
else:
    st.info("No example episode in the mart for this selection (notebook 09A with BUILD_EXAMPLES = True writes them).")

st.subheader("Football context available in the warehouse")
ui.html_table(da.table("nfl_context_availability").to_dicts(), [("context_var", "Context"), ("source", "Source"), ("share_labelled", "Share labelled"), ("status", "Status")])
ctx = da.table("nfl_demand_context").filter((pl.col("demand_id") == demand) & pl.col("position_group").is_in(chosen))
if ctx.height:
    var = st.selectbox("Context variable", sorted(ctx["context_var"].unique().to_list()))
    st.dataframe(ctx.filter(pl.col("context_var") == var).sort("position_group", "n_episodes", descending=[False, True]).select("position_group", "level", "n_episodes",
                 "n_athletes", "median_primary").to_pandas(), use_container_width=True, hide_index=True)
