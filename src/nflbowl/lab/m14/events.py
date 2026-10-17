"""Phase 1 — movement-event detection on the 10 Hz sequence channels.

Signals (physical units after ``m11.representation.SCALE``): speed v (yd/s), a_tan (yd/s², + accelerating / − braking),
a_n (yd/s², signed centripetal), yaw (rad/s), dt (s), heading_valid. Only frames inside the mask with finite speed and
a_tan are used.

Time convention (same as ``lab.biomech``): ``dt[k]`` is the duration of frame k, so t[k+1] − t[k] = dt[k]. A frame whose
duration exceeds ``max_gap_dt`` (missed samples) or an invalid frame BREAKS the signal into segments; an event never
crosses a segment boundary. Non-positive / non-finite dt on a valid frame is replaced by the nominal 0.1 s.

Cut event (all indices inside one segment):

    entry peak e   highest speed between the previous accepted minimum and this one, before speed first drops below
                   the minimum (topographic left prominence)
    braking onset b  first frame in [e, m] that starts ``sustain_frames`` consecutive frames with a_tan ≤ −brake_thresh
                   (fallback: first single such frame, flagged ``brake_unsustained``; none: b = e, ``weak_braking``)
    turn phase     frames [b, x]; turn angle = Σ yaw·dt over heading-valid frames, direction = sign when |angle| ≥
                   ``turn_min_angle`` and heading coverage ≥ 0.5
    minimum m      local speed minimum (v[m] ≤ v[m−1] and v[m] < v[m+1]); never the first / last frame of a segment
    propulsion p   first frame in [m, x] that starts ``sustain_frames`` frames with a_tan ≥ +brake_thresh
                   (none: ``propulsion_censored``)
    exit x         highest speed after m before speed falls to ≤ v[m] again (right prominence); the event is accepted
                   when the rise ≥ ``rise_min`` or the segment ends first (``exit_censored``); a smaller rise followed
                   by a deeper dip is a double dip and is merged into the next valley
    recovery r     first time after m with v ≥ recovery_frac · v_entry (crossing interpolated; else censored)

Accepted when v_entry ≥ ``entry_min_speed`` and v_entry − v_min ≥ max(drop_min, drop_frac · v_entry).
``complete`` = sustained or single braking frame present, propulsion observed and exit not censored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

NAN = float("nan")
SPEED, ATAN, AN, YAW, DT, HV = 0, 1, 2, 3, 4, 5
SC = {"speed": 10.0, "a_tan": 5.0, "a_n": 5.0, "yaw": 3.0, "dt": 0.1}
NOMINAL_DT = 0.1
EVENT_VERSION = 1


@dataclass(frozen=True)
class EventParams:
    entry_min_speed: float = 3.0
    drop_min: float = 1.0
    drop_frac: float = 0.25
    rise_min: float = 1.0
    brake_thresh: float = 1.0
    sustain_frames: int = 2
    exit_horizon_s: float = 0.5
    recovery_frac: float = 0.9
    max_gap_dt: float = 0.15
    min_segment_frames: int = 7
    trace_half_window: int = 10
    useful_speed: float = 6.0
    v_ref: float = 7.0
    turn_min_angle: float = 0.35

    @classmethod
    def from_config(cls, cfg: Any) -> "EventParams":
        d = cfg.event_params() if hasattr(cfg, "event_params") else dict(cfg)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_version": EVENT_VERSION}


# ====================================================================== signals
def physical(x: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
    """(B, L, 6) scaled channels + (B, L) mask → physical arrays; invalid frames are NaN (dt: nominal on valid frames)."""
    x = np.asarray(x, np.float64)
    m = np.asarray(m, bool)
    if x.ndim == 2:
        x, m = x[None], m[None]
    sp = x[..., SPEED] * SC["speed"]
    at = x[..., ATAN] * SC["a_tan"]
    valid = m & np.isfinite(sp) & np.isfinite(at)
    dt = x[..., DT] * SC["dt"]
    dt = np.where(valid & np.isfinite(dt) & (dt > 0), dt, np.where(valid, NOMINAL_DT, NAN))
    return {"speed": np.where(valid, sp, NAN), "a_tan": np.where(valid, at, NAN), "a_n": np.where(valid, x[..., AN] * SC["a_n"], NAN),
            "yaw": np.where(valid, x[..., YAW] * SC["yaw"], NAN), "dt": dt, "heading_valid": np.where(valid, x[..., HV], NAN), "valid": valid}


def segments(valid: np.ndarray, dt: np.ndarray, max_gap_dt: float, min_len: int) -> list[tuple[int, int]]:
    """Maximal gap-free runs [s, e) of valid frames; a frame k with dt[k] > max_gap_dt ends its run after k."""
    out: list[tuple[int, int]] = []
    n = valid.size
    s = None
    for k in range(n):
        if valid[k]:
            if s is None:
                s = k
            if dt[k] > max_gap_dt or k == n - 1:
                if k + 1 - s >= min_len:
                    out.append((s, k + 1))
                s = None
        elif s is not None:
            if k - s >= min_len:
                out.append((s, k))
            s = None
    return out


def seg_time(dt: np.ndarray) -> np.ndarray:
    """Time at the start of each frame of a segment (t[0] = 0)."""
    return np.concatenate([[0.0], np.cumsum(dt[:-1])]) if dt.size else dt


def _cross(t: np.ndarray, y: np.ndarray, k0: int, k1: int, level: float) -> float:
    """Time at which y crosses ``level`` between frames k0 and k1 (linear interpolation, clipped to the interval)."""
    a, b = y[k0], y[k1]
    if not (np.isfinite(a) and np.isfinite(b)) or b == a:
        return float(t[k1])
    f = min(1.0, max(0.0, (level - a) / (b - a)))
    return float(t[k0] + f * (t[k1] - t[k0]))


def _value_at(t: np.ndarray, y: np.ndarray, start: int, horizon: float) -> float:
    """y interpolated at t[start] + horizon (NaN when the segment ends earlier)."""
    target = t[start] + horizon
    k = int(np.searchsorted(t, target - 1e-9, side="left"))
    if k >= t.size:
        return NAN
    if k == start or t[k] == t[k - 1]:
        return float(y[k])
    f = (target - t[k - 1]) / (t[k] - t[k - 1])
    return float(y[k - 1] + f * (y[k] - y[k - 1]))


def trapezoid(y: np.ndarray, t: np.ndarray) -> float:
    """∫ y dt by the trapezoid rule (explicit: ``np.trapz`` is deprecated in NumPy 2)."""
    y, t = np.asarray(y, float), np.asarray(t, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t))) if y.size >= 2 else NAN


def _first_sustained(cond: np.ndarray, lo: int, hi: int, length: int, limit: int) -> int | None:
    """First k in [lo, hi] such that cond[k : k + length] is all True with k + length − 1 ≤ limit."""
    for k in range(lo, hi + 1):
        end = k + length
        if end - 1 > limit:
            break
        if cond[k:end].all():
            return k
    return None


# ====================================================================== valleys
def valleys(v: np.ndarray) -> list[int]:
    """Interior local minima (v[i] ≤ v[i−1] and v[i] < v[i+1]); plateaus resolve to their last frame."""
    return [i for i in range(1, v.size - 1) if v[i] <= v[i - 1] and v[i] < v[i + 1]]


def _left_peak(v: np.ndarray, i: int, bound: int) -> tuple[int, float]:
    j, best, bi = i - 1, v[i], i
    while j >= bound and v[j] >= v[i]:
        if v[j] > best:
            best, bi = v[j], j
        j -= 1
    return bi, float(best)


def _right_peak(v: np.ndarray, i: int) -> tuple[int, float, bool]:
    j, best, bi = i + 1, v[i], i
    while j < v.size and v[j] > v[i]:
        if v[j] > best:
            best, bi = v[j], j
        j += 1
    return bi, float(best), j == v.size


# ====================================================================== event detection on one segment
def detect_segment(v: np.ndarray, at: np.ndarray, an: np.ndarray, yw: np.ndarray, dt: np.ndarray, hv: np.ndarray, P: EventParams) -> list[dict[str, Any]]:
    """Cut events of ONE gap-free segment (all arrays finite on speed / a_tan); indices are segment-local."""
    n = v.size
    if n < 3:
        return []
    t = seg_time(dt)
    thr = P.brake_thresh
    brk, prp = at <= -thr, at >= thr
    out: list[dict[str, Any]] = []
    bound = 0
    for i in valleys(v):
        if i < bound:
            continue
        e, v_entry = _left_peak(v, i, bound)
        v_min = float(v[i])
        drop = v_entry - v_min
        if v_entry < P.entry_min_speed or drop < max(P.drop_min, P.drop_frac * v_entry):
            continue
        x, v_exit, rcens = _right_peak(v, i)
        rise = v_exit - v_min
        if rise >= P.rise_min:
            exit_censored = False
        elif rcens:
            exit_censored = True
        else:
            continue                                           # double dip: the deeper valley that follows takes this approach
        # ---- braking
        b = _first_sustained(brk, e, i, P.sustain_frames, i)
        flags = {"brake_unsustained": False, "weak_braking": False}
        if b is None:
            single = np.flatnonzero(brk[e: i + 1])
            if single.size:
                b, flags["brake_unsustained"] = e + int(single[0]), True
            else:
                b, flags["weak_braking"] = e, True
        # ---- propulsion
        p = _first_sustained(prp, i, x, P.sustain_frames, n - 1)
        prop_censored = p is None
        hi_b = p if p is not None else x
        be_c = np.flatnonzero(brk[b: hi_b + 1])
        be = b + int(be_c[-1]) if be_c.size else b
        complete = not (flags["weak_braking"] or prop_censored or exit_censored)
        # ---- braking phase [b, i]
        dec = -at[b: i + 1]
        n_brake = dec.size
        peak_dec = float(np.max(dec)) if n_brake else NAN
        mean_dec = float(np.mean(dec)) if n_brake else NAN
        d_brake = trapezoid(v[b: i + 1], t[b: i + 1]) if n_brake >= 2 else NAN
        v_b = float(v[b])
        rec: dict[str, Any] = {"i_entry": e, "i_brake": b, "i_min": i, "i_prop": -1 if p is None else p, "i_exit": x, "i_brake_end": be,
                               "t_entry": float(t[e]), "t_brake": float(t[b]), "t_min": float(t[i]), "t_exit": float(t[x]),
                               "v_entry": v_entry, "v_brake": v_b, "v_min": v_min, "v_exit_peak": v_exit, "speed_drop": drop,
                               "braking_duration": float(t[i] - t[b]) if i > b else NAN, "braking_distance": d_brake,
                               "peak_decel": peak_dec, "mean_decel": mean_dec, "n_brake_frames": int(n_brake),
                               "exit_censored": exit_censored, "propulsion_censored": prop_censored, "complete": complete, **flags}
        ok_brake = not flags["weak_braking"]
        # motorsport
        rec["braking_point_efficiency"] = ((v_b ** 2 - v_min ** 2) / (2 * d_brake)) if (ok_brake and np.isfinite(d_brake) and d_brake >= 0.05) else NAN
        rec["braking_budget_utilization"] = (mean_dec / peak_dec) if (ok_brake and n_brake >= 2 and peak_dec >= thr) else NAN
        v_h = _value_at(t, v, i, P.exit_horizon_s)
        rec["v_exit_horizon"] = v_h
        rec["corner_exit_efficiency"] = v_h / v_entry if np.isfinite(v_h) and v_entry > 0 else NAN
        wb = [k for k in range(e, i + 1) if t[k] >= t[i] - P.exit_horizon_s - 1e-9]
        wp = [k for k in range(i + 1, n) if t[k] - t[i] <= P.exit_horizon_s + 1e-9]
        full_p = (t[n - 1] - t[i]) >= P.exit_horizon_s - 1e-9
        bm = float(np.mean(-at[wb])) if len(wb) >= 3 else NAN
        pm = float(np.mean(at[wp])) if len(wp) >= 3 and full_p else NAN
        rec["stop_go_elasticity"] = pm / bm if np.isfinite(bm) and np.isfinite(pm) and bm >= 0.5 else NAN
        # alpine
        rec["turn_speed_retention"] = v_min / v_entry if v_entry > 0 else NAN
        # court
        trans = NAN
        if ok_brake and p is not None and be + 1 < n and p >= 1:
            t_rel = _cross(t, at, be, be + 1, -thr)
            t_go = _cross(t, at, p - 1, p, thr) if p - 1 >= be else float(t[p])
            trans = t_go - t_rel if t_go >= t_rel else NAN
        rec["brake_propulsion_transition"] = trans
        rec["first_step_explosiveness"] = float(np.mean(at[p: p + 3])) if p is not None and p + 2 <= n - 1 else NAN
        thr_v = P.recovery_frac * v_entry
        rk = next((k for k in range(i + 1, n) if v[k] >= thr_v), None)
        rec["recovery_censored"] = rk is None
        rec["cut_recovery_time"] = (_cross(t, v, rk - 1, rk, thr_v) - t[i]) if rk is not None else NAN
        # rugby / soccer
        if complete and n_brake >= 2:
            rec["max_controlled_decel"] = float(np.max(0.5 * (dec[1:] + dec[:-1])))
        else:
            rec["max_controlled_decel"] = NAN
        # turn phase [b, x]
        seg = slice(b, x + 1)
        h = np.asarray(hv[seg], float)
        hok = np.isfinite(h) & (h > 0.5) & np.isfinite(yw[seg])
        cov = float(hok.mean()) if hok.size else 0.0
        ang = float(np.sum(np.where(hok, yw[seg], 0.0) * dt[seg]))
        rec["turn_heading_coverage"] = cov
        rec["turn_angle"] = ang if cov > 0 else NAN
        rec["turn_dir"] = int(np.sign(ang)) if (cov >= 0.5 and abs(ang) >= P.turn_min_angle) else 0
        an_s = np.asarray(an[seg], float)
        an_sum = float(np.nansum(an_s * dt[seg]))
        rec["an_turn_sign"] = int(np.sign(an_sum)) if np.isfinite(an_s).any() else 0
        rec["max_abs_a_n"] = float(np.nanmax(np.abs(an_s))) if np.isfinite(an_s).any() else NAN
        vs = v[seg]
        mv = np.isfinite(an_s) & (vs >= 2.0)
        rec["max_curvature"] = float(np.max(np.abs(an_s[mv]) / vs[mv] ** 2)) if mv.any() else NAN
        out.append(rec)
        bound = i + 1
    return out


def detect_sequence(arrs: dict[str, np.ndarray], P: EventParams) -> tuple[list[dict[str, Any]], dict[str, Any], list[tuple[int, int]]]:
    """Events of one sequence (1-D physical arrays) with sequence-level indices + the sequence quality record + segments."""
    valid = arrs["valid"]
    dt = arrs["dt"]
    segs = segments(valid, np.where(np.isfinite(dt), dt, 0.0), P.max_gap_dt, P.min_segment_frames)
    events: list[dict[str, Any]] = []
    H = P.trace_half_window
    L = valid.size
    for s, e in segs:
        sl = slice(s, e)
        evs = detect_segment(arrs["speed"][sl], arrs["a_tan"][sl], arrs["a_n"][sl], arrs["yaw"][sl], dt[sl], arrs["heading_valid"][sl], P)
        for ev in evs:
            for k in ("i_entry", "i_brake", "i_min", "i_exit", "i_brake_end"):
                ev[k] += s
            if ev["i_prop"] >= 0:
                ev["i_prop"] += s
            ev["seg_start"], ev["seg_end"] = s, e
            c = ev["i_min"]
            idx = np.arange(c - H, c + H + 1)
            inside = (idx >= s) & (idx < e)
            sp = np.full(idx.size, NAN)
            ac = np.full(idx.size, NAN)
            sp[inside] = arrs["speed"][idx[inside]]
            ac[inside] = arrs["a_tan"][idx[inside]]
            ev["trace_speed"] = sp.astype(np.float32).tolist()
            ev["trace_a_tan"] = ac.astype(np.float32).tolist()
            events.append(ev)
    gaps = int(np.sum(valid & np.isfinite(dt) & (dt > P.max_gap_dt)))
    hv = arrs["heading_valid"]
    q = {"n_valid": int(valid.sum()), "n_frames": int(L), "n_segments": len(segs), "longest_segment": max((e - s for s, e in segs), default=0),
         "n_dt_gaps": gaps, "n_internal_invalid": int(_internal_invalid(valid)),
         "heading_valid_share": float(np.nanmean(np.where(valid, hv, np.nan))) if valid.any() else NAN,
         "n_events": len(events), "n_complete_events": int(sum(1 for ev in events if ev["complete"]))}
    return events, q, segs


def _internal_invalid(valid: np.ndarray) -> int:
    """Invalid frames strictly between the first and last valid frame (mask holes)."""
    v = np.flatnonzero(valid)
    if v.size < 2:
        return 0
    return int((~valid[v[0]: v[-1] + 1]).sum())


EVENT_METRIC_COLUMNS = ("corner_exit_efficiency", "braking_point_efficiency", "braking_budget_utilization", "stop_go_elasticity", "turn_speed_retention",
                        "brake_propulsion_transition", "first_step_explosiveness", "cut_recovery_time", "max_controlled_decel")
EVENT_CONTEXT_COLUMNS = ("v_entry", "v_brake", "v_min", "v_exit_peak", "v_exit_horizon", "speed_drop", "braking_duration", "braking_distance",
                         "peak_decel", "mean_decel", "n_brake_frames", "turn_angle", "turn_dir", "an_turn_sign", "turn_heading_coverage", "max_abs_a_n",
                         "max_curvature")
EVENT_FLAG_COLUMNS = ("complete", "exit_censored", "propulsion_censored", "recovery_censored", "weak_braking", "brake_unsustained")


__all__ = ["EVENT_CONTEXT_COLUMNS", "EVENT_FLAG_COLUMNS", "EVENT_METRIC_COLUMNS", "EVENT_VERSION", "EventParams", "NOMINAL_DT", "SC", "detect_segment",
           "detect_sequence", "physical", "seg_time", "segments", "trapezoid", "valleys"]
