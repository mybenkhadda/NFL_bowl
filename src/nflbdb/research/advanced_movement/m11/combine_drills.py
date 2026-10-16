"""Drill-specific Combine feature families (11A · A1) computed from the persisted Phase 03 frames of every tracked drill.

Each attempt is reduced to its longest segment of valid kinematics (frames in time order). Only quantities supported by
the observed geometry are built; anything whose end point is outside the recording is NaN (``…_censored`` flags kept).

    forty                 speed-band passages (2→4, 4→6, 6→8, 8→10 yd/s), sprint model (time form + offset-free phase
                          form), early acceleration, observed top speed, phase timing (onset → 4 / 6 / 8 yd/s), speed at
                          5 / 10 yd, model 40-yd time (plausibility against the official time only)
    short_shuttle, three_cone, long_shuttle, backpedal_transition
                          approach speed, braking profile, minimum speed, turn geometry, exit acceleration, Brake-to-Burst,
                          Speed Recovery Time, Transition Cost Index, lateral acceleration, curved-running speed (three-cone),
                          movement smoothness (state changes per second), tracked drill duration
    pass_rush             launch acceleration (first 0.5 s after onset), speed at 2 / 5 yd, lateral extent, accelerating phases
    gauntlet              sustained speed, speed variation in the sustained phase, top speed
    other                 generic: top speed, 95th percentile a_tan, transitions

Feature names are ``<family>.<metric>``. ``PRESELECTED`` declares, BEFORE any NFL value is read, the ≤ 3 interpretable
features per family that enter model M2 of 11A (physics-motivated, not searched).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from . import physics as ph
from . import segmentation as sg
from .config import M11Config

NAN = float("nan")
TRANSITION_FAMILIES = ("short_shuttle", "three_cone", "long_shuttle", "backpedal_transition")
PRESELECTED: dict[str, tuple[str, ...]] = {
    "forty": ("forty.band_4_6_mean_accel", "forty.band_6_8_mean_accel", "forty.phase_A0"),
    "short_shuttle": ("short_shuttle.decel_mean", "short_shuttle.brake_to_burst_s", "short_shuttle.exit_accel"),
    "three_cone": ("three_cone.transition_cost_index", "three_cone.curved_speed", "three_cone.exit_accel"),
    "long_shuttle": ("long_shuttle.decel_mean", "long_shuttle.exit_accel", "long_shuttle.speed_recovery_time_s"),
    "backpedal_transition": ("backpedal_transition.brake_to_burst_s", "backpedal_transition.exit_accel", "backpedal_transition.transition_cost_index"),
    "pass_rush": ("pass_rush.launch_accel", "pass_rush.speed_at_5yd", "pass_rush.lateral_extent_yd"),
    "gauntlet": ("gauntlet.sustained_speed", "gauntlet.speed_cv_sustained", "gauntlet.v_max"),
    "other": ("other.v_max", "other.a_tan_p95", "other.n_transitions"),
}
# lower is quicker / better-looking physically, used only to orient figures (never to rank athletes)
LOWER_IS_QUICKER = {"time_s", "brake_to_burst_s", "speed_recovery_time_s", "transition_cost_index", "braking_duration_s", "brake_time_5_2_s",
                    "transition_duration_s", "tau", "duration_s", "t_to_4", "t_to_6", "t_to_8", "model_t40", "speed_cv_sustained"}


def longest_segment(f: pl.DataFrame) -> pl.DataFrame:
    """Frames of the longest segment with valid kinematics (``valid_kin`` when present), time order."""
    if "valid_kin" in f.columns:
        f = f.filter(pl.col("valid_kin").fill_null(False))
    if f.is_empty():
        return f
    if "seg_idx" in f.columns:
        best = f.group_by("seg_idx").len().sort(["len", "seg_idx"], descending=[True, False])["seg_idx"][0]
        f = f.filter(pl.col("seg_idx") == best)
    return f.sort("t_s" if "t_s" in f.columns else "ts")


def _arr(f: pl.DataFrame, k: str) -> np.ndarray | None:
    return f[k].cast(pl.Float64).fill_null(NAN).to_numpy().astype(float) if k in f.columns else None


def _onset(t: np.ndarray, v: np.ndarray, level: float = 1.0) -> int | None:
    """Index of the first frame at or above ``level`` when the recording starts below it (else None: onset not recorded)."""
    if v.size == 0 or not np.isfinite(v[0]) or v[0] >= level:
        return None
    idx = np.flatnonzero(v >= level)
    return int(idx[0]) if idx.size else None


def _path(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = np.zeros_like(v)
    d[1:] = np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))
    return d


def _speed_at_distance(t: np.ndarray, v: np.ndarray, start: int, dist: float) -> float:
    d = _path(t[start:], v[start:])
    return float(np.interp(dist, d, v[start:])) if d.size and d[-1] >= dist else NAN


def forty_features(t: np.ndarray, v: np.ndarray, a: np.ndarray, c: M11Config) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update({k: float(x) for k, x in ph.band_profile(t, v, a, c.bands, c.band_min_frames).items()})
    ident = bool(np.isfinite(v[0]) and v[0] < c.sprint_identifiable_v0)
    tf = ph.fit_sprint_time(t, v, c.sprint_tau_grid, c.sprint_t0_grid, fit_offset=ident, min_frames=c.sprint_min_frames)
    pf = ph.fit_sprint_phase(v, a, vmin=1.0)
    out |= {"sprint_vmax": tf["vmax"], "sprint_tau": tf["tau"], "sprint_a0": tf["a0"], "sprint_r2": tf["r2"], "sprint_rmse": tf["rmse"],
            "sprint_offset_identifiable": float(ident), "sprint_grid_edge": float(tf["grid_edge"]), "phase_A0": pf["A0"], "phase_V0": pf["V0"],
            "phase_tau": pf["tau"], "phase_rmse": pf["rmse"], "v_max_obs": float(np.nanmax(v)) if np.isfinite(v).any() else NAN,
            "model_t40": ph.time_to_distance(40.0, tf["vmax"], tf["tau"]) if tf["ok"] else NAN}
    on = _onset(t, v)
    if on is not None:
        early = (t >= t[on]) & (t <= t[on] + 1.0) & np.isfinite(a)
        out["early_accel"] = float(np.mean(a[early])) if early.sum() >= 3 else NAN
        for lvl in (4.0, 6.0, 8.0):
            tc = ph.crossing_time(t[on:], v[on:], lvl) if v[on] < lvl else NAN
            out[f"t_to_{lvl:g}"] = float(tc - t[on]) if np.isfinite(tc) else NAN
        out["speed_at_5yd"], out["speed_at_10yd"] = _speed_at_distance(t, v, on, 5.0), _speed_at_distance(t, v, on, 10.0)
    else:
        out |= {"early_accel": NAN, "t_to_4": NAN, "t_to_6": NAN, "t_to_8": NAN, "speed_at_5yd": NAN, "speed_at_10yd": NAN}
    out["onset_censored"] = float(on is None)
    return out


def transition_features(f: pl.DataFrame, family: str, c: M11Config) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float]]:
    """(attempt features, transition rows, segmenter agreement) of one transition-drill attempt."""
    t, v, a = _arr(f, "t_s"), _arr(f, "speed"), _arr(f, "a_tan")
    vx, vy, ax, ay = (_arr(f, k) for k in ("vx", "vy", "ax", "ay"))
    tr = sg.transitions(t, v, a, vx, vy, ax, ay, brake_on=c.brake_on, accel_on=c.accel_on, brake_min_drop=c.brake_min_drop, burst_threshold=c.burst_threshold,
                        recovery_fraction=c.recovery_fraction, turn_window_s=c.turn_window_s, heading_min_speed=c.heading_min_speed)
    med = lambda k: float(np.nanmedian([d[k] for d in tr])) if tr and np.isfinite([d[k] for d in tr]).any() else NAN
    out = {m: med(m) for m in sg.METRICS}
    out["n_transitions"] = float(len(tr))
    out["approach_speed"] = float(np.nanmax(v[: tr[0]["i_brake"] + 1])) if tr else NAN
    out["decel_peak_max"] = float(np.nanmax([d["decel_peak"] for d in tr])) if tr else NAN
    out["censored_share"] = float(np.mean([d["censored"] for d in tr])) if tr else NAN
    on = _onset(t, v)
    out["duration_s"] = float(t[-1] - t[on]) if on is not None else NAN
    yaw = _arr(f, "yaw_rate")
    if yaw is not None:
        curved = np.isfinite(yaw) & (np.abs(yaw) >= 60.0) & (v >= 3.0)
        out["curved_speed"] = float(np.nanmedian(v[curved])) if curved.sum() >= 3 else NAN
    else:
        out["curved_speed"] = NAN
    lab = sg.threshold_labels(a, c.brake_on, c.accel_on)
    dur = float(t[-1] - t[0]) if t.size > 1 else NAN
    out["state_changes_per_s"] = float((np.diff(lab) != 0).sum() / dur) if dur and dur > 0 else NAN
    for side, sign in (("ccw", 1.0), ("cw", -1.0)):
        vals = [d["transition_cost_index"] for d in tr if np.isfinite(d["turn_angle_deg"]) and np.sign(d["turn_angle_deg"]) == sign]
        out[f"tci_{side}"] = float(np.nanmedian(vals)) if vals and np.isfinite(vals).any() else NAN
    agree = {"agree_changepoint": sg.agreement(lab, sg.changepoint_labels(a, c.changepoint_penalty, c.brake_on, c.accel_on)),
             "agree_hmm": sg.agreement(lab, sg.hmm_labels(a, c.hmm_iters)), "n_frames": float(t.size)}
    rows = [{k: (float(x) if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) else x) for k, x in d.items()} | {"transition": i}
            for i, d in enumerate(tr)]
    return out, rows, agree


def pass_rush_features(t: np.ndarray, v: np.ndarray, a: np.ndarray, x: np.ndarray | None, y: np.ndarray | None) -> dict[str, float]:
    on = _onset(t, v)
    out = {"launch_accel": NAN, "speed_at_2yd": NAN, "speed_at_5yd": NAN, "v_max": float(np.nanmax(v)) if np.isfinite(v).any() else NAN}
    if on is not None:
        w = (t >= t[on]) & (t <= t[on] + 0.5) & np.isfinite(a)
        out["launch_accel"] = float(np.mean(a[w])) if w.sum() >= 3 else NAN
        out["speed_at_2yd"], out["speed_at_5yd"] = _speed_at_distance(t, v, on, 2.0), _speed_at_distance(t, v, on, 5.0)
    out["n_accel_phases"] = float(sum(1 for i, j in sg.runs(np.isfinite(a) & (a >= 1.0)) if j - i >= 3))
    if x is not None and y is not None and np.isfinite(x).sum() >= 5:
        P = np.c_[x, y][np.isfinite(x) & np.isfinite(y)]
        P = P - P.mean(0)
        _, _, vt = np.linalg.svd(P, full_matrices=False)
        w2 = P @ vt[1]
        out["lateral_extent_yd"] = float(w2.max() - w2.min())
    else:
        out["lateral_extent_yd"] = NAN
    return out


def gauntlet_features(t: np.ndarray, v: np.ndarray) -> dict[str, float]:
    vmax = float(np.nanmax(v)) if np.isfinite(v).any() else NAN
    sus = np.isfinite(v) & (v >= 0.8 * vmax) if np.isfinite(vmax) else np.zeros(v.size, bool)
    mean = float(v[sus].mean()) if sus.sum() >= 3 else NAN
    return {"v_max": vmax, "sustained_speed": mean, "speed_cv_sustained": float(v[sus].std() / mean) if sus.sum() >= 3 and mean > 0 else NAN,
            "sustained_time_s": float(sus.sum() * float(np.nanmedian(np.diff(t)))) if t.size > 1 else NAN}


def generic_features(t: np.ndarray, v: np.ndarray, a: np.ndarray, c: M11Config) -> dict[str, float]:
    tr = sg.transitions(t, v, a, brake_on=c.brake_on, accel_on=c.accel_on, brake_min_drop=c.brake_min_drop)
    return {"v_max": float(np.nanmax(v)) if np.isfinite(v).any() else NAN, "a_tan_p95": float(np.nanquantile(a, 0.95)) if np.isfinite(a).any() else NAN,
            "n_transitions": float(len(tr))}


def attempt_tables(frames: pl.DataFrame, c: M11Config, families: Sequence[str] | None = None) -> dict[str, pl.DataFrame]:
    """{attempt_features (long), transitions, segmenter_agreement, attempt_index} for every attempt of ``frames``
    (``sources.combine_frames`` output)."""
    feats, trans, agree, index = [], [], [], []
    fams = None if families is None else set(families)
    order = (frames.group_by("seq_id").agg(pl.col("nfl_id").first(), pl.col("family").first(), pl.col("drill").first(), pl.col("ts").min().alias("_start"))
             .with_columns(pl.col("_start").rank("ordinal").over("nfl_id", "drill").cast(pl.Int64).alias("attempt_order")))
    meta = {r["seq_id"]: r for r in order.iter_rows(named=True)}
    for (sid,), g in frames.partition_by(["seq_id"], as_dict=True, maintain_order=True).items():
        m = meta[sid]
        fam = m["family"]
        if fams is not None and fam not in fams:
            continue
        f = longest_segment(g)
        base = {"seq_id": sid, "nfl_id": int(m["nfl_id"]), "family": fam, "drill": m["drill"], "attempt_order": int(m["attempt_order"])}
        index.append({**base, "n_frames": f.height, "first_speed": float(f["speed"][0]) if f.height else NAN})
        if f.height < 6:
            continue
        t, v, a = _arr(f, "t_s"), _arr(f, "speed"), _arr(f, "a_tan")
        if fam == "forty":
            vals = forty_features(t, v, a, c)
        elif fam in TRANSITION_FAMILIES:
            vals, rows, ag = transition_features(f, fam, c)
            trans += [{**base, **r} for r in rows]
            agree.append({**base, **ag})
        elif fam == "pass_rush":
            vals = pass_rush_features(t, v, a, _arr(f, "x_s"), _arr(f, "y_s"))
        elif fam == "gauntlet":
            vals = gauntlet_features(t, v)
        else:
            vals = generic_features(t, v, a, c)
        feats += [{**base, "feature": f"{fam}.{k}", "value": (float(x) if x is not None and np.isfinite(x) else None)} for k, x in vals.items()]
    long = pl.DataFrame(feats, schema={"seq_id": pl.String, "nfl_id": pl.Int64, "family": pl.String, "drill": pl.String, "attempt_order": pl.Int64,
                                       "feature": pl.String, "value": pl.Float64}) if feats else pl.DataFrame()
    return {"m11_attempt_features": long, "m11_combine_transitions": pl.DataFrame(trans, infer_schema_length=None) if trans else pl.DataFrame(),
            "m11_segmenter_agreement": pl.DataFrame(agree) if agree else pl.DataFrame(), "m11_attempt_index": pl.DataFrame(index)}


def athlete_features(long: pl.DataFrame) -> pl.DataFrame:
    """Athlete value of each feature = mean of the athlete's attempts (the frozen rule of Milestone 04), with SD and count."""
    return (long.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .group_by("nfl_id", "family", "feature").agg(pl.col("value").mean().alias("value"), pl.col("value").std().alias("sd_attempts"),
                                                         pl.len().cast(pl.Int64).alias("n_attempts")).sort("nfl_id", "feature"))


def wide(athlete: pl.DataFrame, features: Sequence[str] | None = None) -> pl.DataFrame:
    a = athlete if features is None else athlete.filter(pl.col("feature").is_in(list(features)))
    return a.pivot(on="feature", index="nfl_id", values="value") if a.height else pl.DataFrame({"nfl_id": pl.Series([], dtype=pl.Int64)})


__all__ = ["LOWER_IS_QUICKER", "PRESELECTED", "TRANSITION_FAMILIES", "athlete_features", "attempt_tables", "forty_features", "gauntlet_features",
           "generic_features", "longest_segment", "pass_rush_features", "transition_features", "wide"]
