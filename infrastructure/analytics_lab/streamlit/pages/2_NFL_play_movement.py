"""NFL play explorer: one play at a time, every tracked entity, snap-aligned comparisons."""

import pandas as pd
import streamlit as st

from labkit import data, export, ui
from labkit.field import BALL_COLOR, COMPARE_COLORS, POSITION_COLORS, trajectory_figure, zoom_ranges
from labkit.kinematics import snap_time, summarize

st.set_page_config(page_title="NFL Lab · NFL play movement", page_icon="🏈", layout="wide")
st.title("NFL play movement")
st.caption("Tracking covers the dataset's athletes (plus the football), not all 22 players on a play.")

mt = data.serving_mtime()


def pick_play(slot: str) -> dict | None:
    mode = st.radio(f"Find play ({slot})", ("game", "athlete"), horizontal=True, key=f"{slot}_mode",
                    format_func={"game": "by game", "athlete": "by athlete"}.get)
    if mode == "game":
        seasons = ui.guard(data.nfl_seasons, mt)
        season = st.selectbox(f"Season ({slot})", seasons, index=len(seasons) - 1, key=f"{slot}_season")
        games = ui.guard(data.nfl_games, mt, season)
        g_labels = dict(zip(games["game_id"], games["game_label"]))
        game_id = st.selectbox(f"Game ({slot})", list(g_labels), format_func=g_labels.get, key=f"{slot}_game")
        plays = ui.guard(data.nfl_plays, mt, game_id)
        desc = plays["play_description"] if "play_description" in plays else pd.Series([""] * len(plays))
        p_labels = {pid: f"{lab} · {n} tracked · {str(d)[:70]}" for pid, lab, n, d in
                    zip(plays["play_id"], plays["play_label"], plays["n_athletes_tracked"], desc.fillna(""))}
        play_id = st.selectbox(f"Play ({slot})", list(p_labels), format_func=p_labels.get, key=f"{slot}_play")
        focus_id = None
    else:
        athletes = ui.guard(data.nfl_athletes, mt)
        a_labels = {r.nfl_id: f"{r.display_name} · {r.position_group} · {int(r.nfl_n_plays_tracked):,} plays"
                    for r in athletes.itertuples()}
        focus_id = st.selectbox(f"Athlete ({slot})", list(a_labels), format_func=a_labels.get, key=f"{slot}_ath")
        seasons = [None] + ui.guard(data.nfl_seasons, mt)
        season_f = st.selectbox(f"Season ({slot})", seasons, format_func=lambda s: "all" if s is None else str(s),
                                key=f"{slot}_aseason")
        plays = ui.guard(data.nfl_athlete_plays, mt, focus_id, season_f)
        if plays.empty:
            st.warning(f"No tracked plays for {slot}.")
            return None
        p_labels = {i: f"{r.game_label} · play {int(r.play_id)} · max {r.max_s:.2f} yd/s"
                    for i, r in enumerate(plays.itertuples())}
        i = st.selectbox(f"Play ({slot}) — sorted by max speed", list(p_labels), format_func=p_labels.get,
                         key=f"{slot}_aplay")
        season, game_id, play_id = int(plays.iloc[i]["season"]), int(plays.iloc[i]["game_id"]), int(plays.iloc[i]["play_id"])
    on_play = ui.guard(data.nfl_play_athletes, mt, game_id, play_id)
    if on_play.empty:
        st.warning(f"No tracked athletes on this play ({slot}).")
        return None
    labels = {r.nfl_id: f"{r.display_name} · {r.position_group}" for r in on_play.itertuples()}
    ids = list(labels)
    default = ids.index(focus_id) if focus_id in ids else 0
    nfl_id = st.selectbox(f"Focus athlete ({slot})", ids, index=default, format_func=labels.get, key=f"{slot}_focus")
    return {"slot": slot, "season": int(season), "game_id": int(game_id), "play_id": int(play_id),
            "nfl_id": int(nfl_id), "name": labels[nfl_id].split(" · ")[0], "athletes": on_play,
            "row": on_play[on_play["nfl_id"] == nfl_id].iloc[0]}


with st.sidebar:
    st.header("Play A")
    A = pick_play("A")
    compare = st.toggle("Compare with B", value=False)
    B = None
    if compare:
        st.header("Play / athlete B")
        B = pick_play("B")
    st.header("View")
    standardised = st.toggle("Standardised coordinates (offense → +x)", value=True)
    zoom = st.toggle("Zoom to tracked area", value=False)
    show_ball = st.toggle("Show football", value=True)
    step = st.slider("Playback step (frames)", 1, 5, 1)
ui.sidebar_footer()

if A is None:
    st.stop()


def load(p: dict) -> pd.DataFrame:
    df, truncated = ui.guard(data.nfl_play_frames, p["season"], p["game_id"], p["play_id"])
    if truncated:
        st.warning(f"{p['slot']}: play truncated at {data.MAX_PLAY_ROWS:,} rows.")
    if standardised:
        missing = df["x_std"].isna().mean()
        if missing > 0:
            st.info(f"{p['slot']}: {missing:.0%} of frames lack play_direction → raw coordinates used for those.")
        df = df.assign(x=df["x_std"].fillna(df["x"]), y=df["y_std"].fillna(df["y"]), dir=df["dir_std"].fillna(df["dir"]))
    df["entity"] = df["nfl_id"].map(lambda v: "football" if pd.isna(v) else str(int(v)))
    return df


picked = [p for p in (A, B) if p is not None]
plays = {p["slot"]: load(p) for p in picked}
focus_frames = {}
for p in picked:
    f = plays[p["slot"]]
    focus_frames[p["slot"]] = f[f["entity"] == str(p["nfl_id"])].sort_values("t_s").reset_index(drop=True)
    if focus_frames[p["slot"]].empty:
        st.error(f"{p['slot']}: no frames for the focus athlete on this play.")
        st.stop()

for p in picked:
    r = p["row"]
    extra = {"play": f"{p['game_id']}-{p['play_id']}", "season": p["season"],
             "role": " / ".join(str(r[c]) for c in ("team_abbr", "lined_up_position") if c in r and pd.notna(r[c])) or "n/a",
             "tracked window": "whole play incl. pre-snap"}
    ui.metric_row(f"{p['slot']} — {p['name']}", summarize(focus_frames[p["slot"]]), extra)

# ------------------------------------------------------------------ playback
st.subheader("Play playback")
figs = {}
cols = st.columns(len(picked)) if len(picked) > 1 else [st.container()]
for col, p in zip(cols, picked):
    f = plays[p["slot"]]
    if not show_ball:
        f = f[f["entity"] != "football"]
    meta = {}
    for r in p["athletes"].itertuples():
        e = str(int(r.nfl_id))
        meta[e] = {"label": f"{r.display_name} ({r.position_group})",
                   "color": POSITION_COLORS.get(r.position_group, POSITION_COLORS["UNKNOWN"]), "size": 10}
    if show_ball and (f["entity"] == "football").any():
        meta["football"] = {"label": "football", "color": BALL_COLOR, "size": 6}
    meta = {e: m for e, m in meta.items() if (f["entity"] == e).any()}
    xr, yr = zoom_ranges(f["x"].to_numpy(), f["y"].to_numpy()) if zoom else (None, None)
    fig = trajectory_figure(f, meta, focus=str(p["nfl_id"]), step=step, x_range=xr, y_range=yr,
                            title=f"{p['slot']} · {p['game_id']}-{p['play_id']} · focus {p['name']}")
    figs[export.stem("nfl", p["game_id"], p["play_id"], p["name"], "playback")] = fig
    col.plotly_chart(fig, use_container_width=True)
st.caption("Frames advance by play-aligned frame_id (100 ms). Arrows: distance covered in 0.5 s at the current speed.")

# ------------------------------------------------------------------ curves
st.subheader("Speed and acceleration over time")
method, thr = ui.alignment_picker("nfl", ("snap", "start", "onset", "peak"), default="snap")
show_dsdt = st.toggle("Also show derived dS/dt (signed, unsmoothed)", value=False)
series = [(f"{p['slot']} · {p['name']}", c, focus_frames[p["slot"]], snap_time(plays[p["slot"]]))
          for p, c in zip(picked, COMPARE_COLORS)]
curves, notes = ui.curves_figure(series, method, thr, show_dsdt)
for n in notes:
    st.warning(n)
st.plotly_chart(curves, use_container_width=True)
figs[export.stem("nfl", *[f"{p['game_id']}-{p['play_id']}-{p['name']}" for p in picked], "speed-accel", method)] = curves

# ------------------------------------------------------------------ tables + export
summary = pd.DataFrame([{"slot": p["slot"], "athlete": p["name"], "nfl_id": p["nfl_id"], "season": p["season"],
                         "game_id": p["game_id"], "play_id": p["play_id"], **summarize(focus_frames[p["slot"]])}
                        for p in picked])
st.subheader("Summary")
st.dataframe(summary, hide_index=True, use_container_width=True)
frame_table = pd.concat([focus_frames[p["slot"]].assign(slot=p["slot"], game_id=p["game_id"], play_id=p["play_id"])
                         for p in picked], ignore_index=True)
with st.expander(f"Focus-athlete frames ({len(frame_table):,} rows)"):
    st.dataframe(frame_table, hide_index=True, use_container_width=True)
ui.export_panel("nfl", figs, {export.stem("nfl", *[p["name"] for p in picked], "summary"): summary,
                              export.stem("nfl", *[p["name"] for p in picked], "frames"): frame_table})
