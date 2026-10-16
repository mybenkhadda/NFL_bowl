"""Combine attempt explorer: playback, speed/acceleration curves, A/B comparison."""

import pandas as pd
import streamlit as st

from labkit import data, export, ui
from labkit.field import COMPARE_COLORS, trajectory_figure, zoom_ranges
from labkit.kinematics import summarize

st.set_page_config(page_title="NFL Lab · Combine movement", page_icon="🏈", layout="wide")
st.title("Combine movement")

mt = data.serving_mtime()
drills = ui.guard(data.combine_drills, mt)


def flag_state(value) -> bool | None:
    """Phase 02 exclude flag → True / False / None (flags not built)."""
    return None if pd.isna(value) else bool(value)


def pick_attempt(slot: str, default_drill: str | None = None) -> dict | None:
    names = drills["drill_name"].tolist()
    counts = dict(zip(drills["drill_name"], drills["n_athletes"]))
    drill = st.selectbox(f"Drill ({slot})", names, index=names.index(default_drill) if default_drill in names else 0,
                         format_func=lambda d: f"{d} · {counts[d]} athletes", key=f"{slot}_drill")
    pool = ui.guard(data.combine_athletes, mt, drill, (), ())
    positions = st.multiselect(f"Position ({slot})", sorted(pool["position_group"].dropna().unique()), key=f"{slot}_pos")
    cohorts = st.multiselect(f"Combine year ({slot})", sorted(int(c) for c in pool["cohort"].dropna().unique()),
                             key=f"{slot}_coh")
    athletes = ui.guard(data.combine_athletes, mt, drill, tuple(positions), tuple(cohorts))
    if athletes.empty:
        st.warning(f"No athletes for {slot} with these filters.")
        return None
    labels = {r.nfl_id: f"{r.display_name} · {r.position_group} · {int(r.cohort)} · best {r.best_max_s:.2f} yd/s"
              for r in athletes.itertuples()}
    nfl_id = st.selectbox(f"Athlete ({slot})", list(labels), format_func=labels.get, key=f"{slot}_ath")
    attempts = ui.guard(data.combine_attempts, mt, nfl_id, drill)
    att_labels = {i: (f"#{int(r.attempt_seq)} · attempt {int(r.attempt)} · {r.max_s:.2f} yd/s · {r.duration_s:.1f} s"
                      + (" · ⚑ flagged" if flag_state(r.exclude_kinematics) else ""))
                  for i, r in enumerate(attempts.itertuples())}
    i = st.selectbox(f"Attempt ({slot})", list(att_labels), format_func=att_labels.get, key=f"{slot}_att")
    row = attempts.iloc[i]
    return {"slot": slot, "drill": drill, "nfl_id": int(nfl_id), "name": labels[nfl_id].split(" · ")[0],
            "event_id": row["event_id"], "attempt": int(row["attempt"]), "attempt_seq": int(row["attempt_seq"]),
            "cohort": int(row["cohort"]), "row": row}


with st.sidebar:
    st.header("Attempt A")
    A = pick_attempt("A")
    compare = st.toggle("Compare with attempt B", value=False)
    B = None
    if compare:
        st.header("Attempt B")
        B = pick_attempt("B", default_drill=A["drill"] if A else None)
    st.header("View")
    zoom = st.toggle("Zoom to trajectory", value=True)
    step = st.slider("Playback step (frames)", 1, 5, 1, help="Skip frames for smoother playback on slow machines.")
ui.sidebar_footer()

if A is None:
    st.stop()

picked = [p for p in (A, B) if p is not None]
frames = {}
for p in picked:
    df, truncated = ui.guard(data.combine_frames, p["cohort"], p["nfl_id"], p["event_id"], p["attempt"])
    if truncated:
        st.warning(f"{p['slot']}: trajectory truncated at {data.MAX_FRAMES:,} frames.")
    if df.empty:
        st.error(f"{p['slot']}: no PLAYER frames found for this attempt.")
        st.stop()
    frames[p["slot"]] = df

for p in picked:
    flag = flag_state(p["row"].get("exclude_kinematics"))
    extra = {"drill": p["drill"], "event": p["event_id"], "attempt": p["attempt"],
             "percentile in drill": f"{p['row']['max_s_pct_in_drill']:.0%}" if pd.notna(p["row"]["max_s_pct_in_drill"]) else "n/a",
             "quality flag": ("excluded: " + str(p["row"].get("exclusion_reasons"))) if flag
             else ("ok" if flag is False else "flags not built")}
    ui.metric_row(f"{p['slot']} — {p['name']}", summarize(frames[p["slot"]]), extra)

if B is not None and A["drill"] != B["drill"]:
    st.info("A and B are different drills: trajectories and curves are shown together, but the movements are "
            "not the same task — compare with care.")

# ------------------------------------------------------------------ trajectories
st.subheader("Trajectory playback")
layout = "side"
if B is not None:
    layout = st.radio("Layout", ("side", "overlay"), horizontal=True,
                      format_func={"side": "Side by side (field coordinates)",
                                   "overlay": "Overlay — both translated to start at (0, 0)"}.get)
figs = {}
if layout == "side":
    cols = st.columns(len(picked))
    for col, p, color in zip(cols, picked, COMPARE_COLORS):
        df = frames[p["slot"]].assign(entity=p["slot"])
        xr, yr = zoom_ranges(df["x"].to_numpy(), df["y"].to_numpy()) if zoom else (None, None)
        fig = trajectory_figure(df, {p["slot"]: {"label": f"{p['slot']} · {p['name']}", "color": color, "size": 12}},
                                focus=p["slot"], step=step, title=f"{p['slot']} · {p['drill']}", x_range=xr, y_range=yr)
        figs[export.stem("combine", p["name"], p["drill"], f"a{p['attempt']}", "trajectory")] = fig
        col.plotly_chart(fig, use_container_width=True)
else:
    parts = []
    for p in picked:
        df = frames[p["slot"]].copy()
        df["x"], df["y"] = df["x"] - df["x"].iloc[0], df["y"] - df["y"].iloc[0]
        parts.append(df.assign(entity=p["slot"]))
    both = pd.concat(parts, ignore_index=True)
    xr, yr = zoom_ranges(both["x"].to_numpy(), both["y"].to_numpy(), pad=3)
    meta = {p["slot"]: {"label": f"{p['slot']} · {p['name']}", "color": c, "size": 12} for p, c in zip(picked, COMPARE_COLORS)}
    fig = trajectory_figure(both, meta, focus="A", step=step, title="A vs B — displacement from start", x_range=xr,
                            y_range=yr)
    fig.update_layout(shapes=[], annotations=[], plot_bgcolor="#181b22", xaxis_title="Δx from start (yd)",
                      yaxis_title="Δy from start (yd)")
    st.caption("Overlay: positions are translated so each attempt starts at (0, 0); frames advance together by "
               "elapsed time since each attempt's start (frame_id is attempt-relative). Orientation is unchanged.")
    figs[export.stem("combine", A["name"], "vs", B["name"], "overlay")] = fig
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ curves
st.subheader("Speed and acceleration over time")
allowed = ("start", "onset", "peak") + (("normalized",) if B is None or A["drill"] == B["drill"] else ())
method, thr = ui.alignment_picker("cmb", allowed)
show_dsdt = st.toggle("Also show derived dS/dt (signed, unsmoothed)", value=False)
curves, notes = ui.curves_figure([(f"{p['slot']} · {p['name']}", c, frames[p["slot"]], None)
                                  for p, c in zip(picked, COMPARE_COLORS)], method, thr, show_dsdt)
for n in notes:
    st.warning(n)
st.plotly_chart(curves, use_container_width=True)
figs[export.stem("combine", *[p["name"] for p in picked], A["drill"], "speed-accel", method)] = curves

# ------------------------------------------------------------------ tables + export
summary = pd.DataFrame([{"slot": p["slot"], "athlete": p["name"], "nfl_id": p["nfl_id"], "drill": p["drill"],
                         "event_id": p["event_id"], "attempt": p["attempt"], **summarize(frames[p["slot"]])}
                        for p in picked])
st.subheader("Summary")
st.dataframe(summary, hide_index=True, use_container_width=True)
frame_table = pd.concat([frames[p["slot"]].assign(slot=p["slot"], nfl_id=p["nfl_id"], event_id=p["event_id"],
                                                  attempt=p["attempt"]) for p in picked], ignore_index=True)
with st.expander(f"Frames ({len(frame_table):,} rows)"):
    st.dataframe(frame_table, hide_index=True, use_container_width=True)
ui.export_panel("cmb", figs, {export.stem("combine", *[p["name"] for p in picked], A["drill"], "summary"): summary,
                              export.stem("combine", *[p["name"] for p in picked], A["drill"], "frames"): frame_table})
