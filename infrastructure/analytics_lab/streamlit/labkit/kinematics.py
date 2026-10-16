"""Trajectory summaries and EXPLICIT time-alignment methods.

Vendor columns: ``s`` speed (yd/s), ``a`` acceleration MAGNITUDE (yd/s², unsigned), ``dis`` distance
since the previous frame (yd), ``dir`` direction of motion (deg). ``dsdt`` below is a derived,
UNSMOOTHED signed longitudinal acceleration (np.gradient of s over t) shown for comparison only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YDS_TO_MPH = 2.04545

ALIGNMENTS = {
    "start": ("Trajectory start",
              "t = 0 at each trajectory's first tracked frame (Combine: attempt start; NFL: first tracked frame of "
              "the play). Absolute timing is preserved."),
    "onset": ("Movement onset",
              "t = 0 at the first frame with speed ≥ the onset threshold. Approximates when the athlete starts "
              "moving; sensitive to the threshold and to pre-movement drift."),
    "peak": ("Peak speed",
             "t = 0 at each trajectory's maximum speed. Compares curve SHAPE around the peak; discards when the "
             "peak happened."),
    "snap": ("Ball snap (NFL)",
             "t = 0 at the play's first 'ball_snap' event. The football-meaningful reference for in-game "
             "comparisons; unavailable for Combine attempts or plays without the event."),
    "normalized": ("Normalised duration",
                   "τ = (t − t_start) / duration ∈ [0, 1]. Only meaningful when both trajectories perform the SAME "
                   "drill/task; it stretches time, so rates (yd/s²) are no longer comparable visually."),
}


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    t = df["t_s"].to_numpy(dtype=float)
    s = df["s"].to_numpy(dtype=float)
    i_peak = int(np.nanargmax(s)) if np.isfinite(s).any() else 0
    dx = float(df["x"].iloc[-1] - df["x"].iloc[0])
    dy = float(df["y"].iloc[-1] - df["y"].iloc[0])
    return {
        "frames": int(len(df)),
        "duration_s": float(t[-1] - t[0]),
        "distance_yds": float(np.nansum(df["dis"].to_numpy(dtype=float))),
        "net_displacement_yds": float(np.hypot(dx, dy)),
        "max_s": float(np.nanmax(s)),
        "max_s_mph": float(np.nanmax(s) * YDS_TO_MPH),
        "mean_s": float(np.nanmean(s)),
        "time_to_peak_s": float(t[i_peak] - t[0]),
        "max_a": float(np.nanmax(df["a"].to_numpy(dtype=float))),
    }


def derived_dsdt(df: pd.DataFrame) -> np.ndarray:
    t = df["t_s"].to_numpy(dtype=float)
    s = df["s"].to_numpy(dtype=float)
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        return np.full(len(t), np.nan)
    return np.gradient(s, t)


def align(df: pd.DataFrame, method: str, onset_threshold: float = 1.0,
          snap_t: float | None = None) -> tuple[np.ndarray | None, str]:
    """Return (aligned time axis, note). ``None`` when the method is not applicable."""
    t = df["t_s"].to_numpy(dtype=float)
    if len(t) == 0:
        return None, "no frames"
    if method == "start":
        return t - t[0], ""
    if method == "peak":
        return t - t[int(np.nanargmax(df["s"].to_numpy(dtype=float)))], ""
    if method == "onset":
        idx = np.flatnonzero(df["s"].to_numpy(dtype=float) >= onset_threshold)
        if idx.size == 0:
            return None, f"speed never reaches {onset_threshold:.2f} yd/s"
        return t - t[idx[0]], ""
    if method == "snap":
        if snap_t is None:
            return None, "no 'ball_snap' event on this play"
        return t - snap_t, ""
    if method == "normalized":
        dur = t[-1] - t[0]
        if dur <= 0:
            return None, "zero duration"
        return (t - t[0]) / dur, ""
    raise ValueError(f"unknown alignment {method!r}")


def snap_time(play_frames: pd.DataFrame) -> float | None:
    ev = play_frames.loc[play_frames["event"].fillna("").str.lower() == "ball_snap", "t_s"]
    return float(ev.min()) if len(ev) else None
