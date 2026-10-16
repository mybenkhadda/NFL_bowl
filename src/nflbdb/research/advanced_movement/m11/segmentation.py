"""Movement-phase segmentation and transition metrics (11C · C1–C2). Pure NumPy; the same functions run on a Combine
attempt (shuttle, three-cone, transition drills) and on an NFL trace.

Three segmenters, compared — never stacked:
    threshold     physically defined: braking = a_tan ≤ brake_on, acceleration = a_tan ≥ accel_on (the 09A rule values)
    changepoint   binary segmentation of a_tan into piecewise-constant means with a BIC-type penalty
    hmm           3-state Gaussian HMM on a_tan (decelerate · steady · accelerate), Baum–Welch then Viterbi

A *transition* (threshold segmenter) is a braking run with a speed drop ≥ ``brake_min_drop`` followed — inside the
recording — by an accelerating run. Its phases, in order:

    approach        ``turn_window_s`` before braking onset
    braking         a_tan ≤ brake_on
    turning         braking end → minimum-speed frame (may be empty for a turn at speed)
    exit            minimum-speed frame → first frame with a_tan ≥ burst_threshold
    reacceleration  the accelerating run after the minimum
    recovery        minimum → speed back to ``recovery_fraction`` × entry speed

Metrics (units in ``METRIC_UNITS``): braking duration and distance, mean / robust peak deceleration, entry speed, exit speed
(speed ``exit_horizon_s`` after the minimum), Transition Cost Index (v_entry − v_exit)/v_entry, Brake-to-Burst latency,
Speed Recovery Time, peak lateral acceleration, signed turn angle. A quantity whose end point is not in the recording is
NaN and flagged ``censored`` — never extrapolated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import physics as ph

NAN = float("nan")
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")       # NumPy 2 renamed trapz
EXIT_HORIZON_S = 0.5          # proposed: exit speed is read this long after the minimum-speed frame
METRIC_UNITS = {"entry_speed": "yd/s", "v_min": "yd/s", "exit_speed": "yd/s", "braking_duration_s": "s", "braking_distance_yd": "yd",
                "decel_mean": "yd/s²", "decel_peak": "yd/s²", "brake_time_5_2_s": "s", "transition_cost_index": "fraction", "brake_to_burst_s": "s",
                "speed_recovery_time_s": "s", "lat_accel_peak": "yd/s²", "turn_angle_deg": "deg", "abs_turn_angle_deg": "deg",
                "exit_accel": "yd/s²", "transition_duration_s": "s"}
METRICS = tuple(METRIC_UNITS)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.size:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def falling_crossing(t: np.ndarray, v: np.ndarray, level: float, start: int = 0) -> float:
    """Time at which ``v`` first falls through ``level`` at or after index ``start`` (interpolated); NaN if not recorded."""
    for j in range(max(1, start + 1), v.size):
        if np.isfinite(v[j - 1]) and np.isfinite(v[j]) and v[j - 1] > level >= v[j]:
            return float(t[j - 1] + (v[j - 1] - level) / (v[j - 1] - v[j]) * (t[j] - t[j - 1]))
    return NAN


def first_time(t: np.ndarray, cond: np.ndarray, start: int) -> float:
    idx = np.flatnonzero(cond[start:])
    return float(t[start + idx[0]]) if idx.size else NAN


# ============================================================================== threshold segmenter + metrics
def transitions(t: np.ndarray, v: np.ndarray, a: np.ndarray, vx: np.ndarray | None = None, vy: np.ndarray | None = None,
                ax: np.ndarray | None = None, ay: np.ndarray | None = None, *, brake_on: float = -1.0, accel_on: float = 1.0,
                brake_min_drop: float = 1.5, burst_threshold: float = 1.5, recovery_fraction: float = 0.8, turn_window_s: float = 0.4,
                heading_min_speed: float = 1.0, min_brake_frames: int = 3, exit_horizon_s: float = EXIT_HORIZON_S) -> list[dict[str, Any]]:
    """Every transition of ONE contiguous trace, with its phase boundaries (indices) and metrics."""
    t, v, a = (np.asarray(z, float) for z in (t, v, a))
    n = v.size
    if n < 6:
        return []
    dt = float(np.nanmedian(np.diff(t))) if n > 1 else 0.1
    w = max(1, int(round(turn_window_s / dt)))
    fin = np.isfinite(v) & np.isfinite(a)
    out = []
    brakes = [(i, j) for i, j in runs(fin & (a <= brake_on)) if j - i >= min_brake_frames]
    accels = runs(fin & (a >= accel_on))
    for k, (i, j) in enumerate(brakes):
        nxt_brake = brakes[k + 1][0] if k + 1 < len(brakes) else n
        acc = next(((p, q) for p, q in accels if p >= j - 1 and p < nxt_brake), None)
        lo_i = max(0, i - w)
        entry = float(np.nanmax(v[lo_i:i + 1]))
        if not np.isfinite(entry) or entry - float(np.nanmin(v[i:j])) < brake_min_drop:
            continue
        search_end = acc[1] if acc else min(n, j + w)
        m = i + int(np.nanargmin(v[i:search_end]))
        d = {"i_approach": lo_i, "i_brake": i, "i_brake_end": j, "i_min": m, "i_accel": acc[0] if acc else -1, "i_accel_end": acc[1] if acc else -1,
             "entry_speed": entry, "v_min": float(v[m]), "t_min": float(t[m]), "censored": False}
        d["braking_duration_s"] = float(t[j - 1] - t[i])
        d["braking_distance_yd"] = float(_trapz(v[i:j], t[i:j])) if j - i >= 2 else NAN
        dec = -a[i:j][np.isfinite(a[i:j])]
        d["decel_mean"] = float(dec.mean()) if dec.size else NAN
        d["decel_peak"] = float(np.quantile(dec, 0.9)) if dec.size else NAN
        t5 = falling_crossing(t, v, 5.0, lo_i)
        t2 = falling_crossing(t, v, 2.0, lo_i)
        d["brake_time_5_2_s"] = float(t2 - t5) if np.isfinite(t5) and np.isfinite(t2) and t2 > t5 else NAN
        t_exit = t[m] + exit_horizon_s
        d["exit_speed"] = float(np.interp(t_exit, t, v)) if t_exit <= t[-1] else NAN
        d["transition_cost_index"] = float((entry - d["exit_speed"]) / entry) if np.isfinite(d["exit_speed"]) and entry > 0 else NAN
        t_burst = first_time(t, fin & (a >= burst_threshold), j)
        d["brake_to_burst_s"] = float(t_burst - t[j - 1]) if np.isfinite(t_burst) else NAN
        t_rec = first_time(t, np.isfinite(v) & (v >= recovery_fraction * entry), m)
        d["speed_recovery_time_s"] = float(t_rec - t[m]) if np.isfinite(t_rec) else NAN
        d["exit_accel"] = float(np.nanmean(a[acc[0]:acc[1]])) if acc else NAN
        d["transition_duration_s"] = float((t_rec if np.isfinite(t_rec) else t[min(n - 1, search_end - 1)]) - t[i])
        d["censored"] = bool(not np.isfinite(d["exit_speed"]) or not np.isfinite(d["speed_recovery_time_s"]) or acc is None)
        if vx is not None and vy is not None:
            ang = ph.turn_angle(vx, vy, slice(lo_i, i + 1), slice(m + 1, min(n, m + 1 + 2 * w)), heading_min_speed)
            d["turn_angle_deg"], d["abs_turn_angle_deg"] = ang, abs(ang) if np.isfinite(ang) else NAN
            if ax is not None and ay is not None:
                an = ph.lateral_accel(vx[lo_i:min(n, m + 2 * w)], vy[lo_i:min(n, m + 2 * w)], ax[lo_i:min(n, m + 2 * w)], ay[lo_i:min(n, m + 2 * w)],
                                      heading_min_speed)
                d["lat_accel_peak"] = float(np.nanmax(np.abs(an))) if np.isfinite(an).any() else NAN
            else:
                d["lat_accel_peak"] = NAN
        else:
            d["turn_angle_deg"] = d["abs_turn_angle_deg"] = d["lat_accel_peak"] = NAN
        out.append(d)
    return out


def threshold_labels(a: np.ndarray, brake_on: float = -1.0, accel_on: float = 1.0) -> np.ndarray:
    """Per-frame state 0 = decelerate, 1 = steady, 2 = accelerate (NaN frames → 1)."""
    a = np.asarray(a, float)
    lab = np.ones(a.size, dtype=np.int64)
    lab[np.isfinite(a) & (a <= brake_on)] = 0
    lab[np.isfinite(a) & (a >= accel_on)] = 2
    return lab


# ============================================================================== changepoints
def binary_segmentation(y: np.ndarray, penalty: float = 3.0, min_size: int = 3, max_cp: int = 20) -> list[int]:
    """Changepoints of a piecewise-constant mean (squared-error cost, penalty = ``penalty`` × σ̂² × log n)."""
    y = np.asarray(y, float)
    y = np.where(np.isfinite(y), y, np.nanmedian(y) if np.isfinite(y).any() else 0.0)
    n = y.size
    if n < 2 * min_size:
        return []
    s = float(np.median(np.abs(np.diff(y)))) / 0.6745                  # robust noise scale from first differences
    sig2 = s * s / 2.0 if s > 0 else 1e-3 * (float(np.var(y)) or 1.0)   # noise-free trace: only real level changes are split
    pen = penalty * sig2 * np.log(n)
    cps: list[int] = []
    segs = [(0, n)]
    while segs and len(cps) < max_cp:
        best = (0.0, None, None)
        for s, e in segs:
            seg = y[s:e]
            if seg.size < 2 * min_size:
                continue
            cs, cs2 = np.cumsum(seg), np.cumsum(seg ** 2)
            tot = cs2[-1] - cs[-1] ** 2 / seg.size
            for k in range(min_size, seg.size - min_size + 1):
                left = cs2[k - 1] - cs[k - 1] ** 2 / k
                right = (cs2[-1] - cs2[k - 1]) - (cs[-1] - cs[k - 1]) ** 2 / (seg.size - k)
                gain = tot - left - right
                if gain > best[0]:
                    best = (gain, s + k, (s, e))
        if best[1] is None or best[0] <= pen:
            break
        cps.append(best[1])
        s, e = best[2]
        segs.remove((s, e))
        segs += [(s, best[1]), (best[1], e)]
    return sorted(cps)


def changepoint_labels(a: np.ndarray, penalty: float = 3.0, brake_on: float = -1.0, accel_on: float = 1.0) -> np.ndarray:
    """State of each changepoint segment from its mean a_tan (same physical thresholds)."""
    a = np.asarray(a, float)
    cps = [0, *binary_segmentation(a, penalty), a.size]
    lab = np.ones(a.size, dtype=np.int64)
    for s, e in zip(cps[:-1], cps[1:]):
        m = np.nanmean(a[s:e]) if np.isfinite(a[s:e]).any() else 0.0
        lab[s:e] = 0 if m <= brake_on else (2 if m >= accel_on else 1)
    return lab


# ============================================================================== 3-state Gaussian HMM
def hmm_labels(a: np.ndarray, iters: int = 30, init_means: tuple[float, float, float] = (-2.0, 0.0, 2.0), stay: float = 0.9) -> np.ndarray:
    """Viterbi states of a 3-state Gaussian HMM fitted by Baum–Welch to ONE trace (state order kept by sorting the means).
    A small model by design: it tests whether a data-driven segmentation agrees with the physical thresholds."""
    y = np.asarray(a, float)
    y = np.where(np.isfinite(y), y, 0.0)
    n, K = y.size, 3
    if n < 6:
        return np.ones(n, dtype=np.int64)
    mu = np.array(init_means, float)
    var = np.full(K, max(float(np.var(y)), 0.25))
    A = np.full((K, K), (1 - stay) / (K - 1))
    np.fill_diagonal(A, stay)
    pi = np.full(K, 1.0 / K)

    def logb() -> np.ndarray:
        return -0.5 * (np.log(2 * np.pi * var)[None, :] + (y[:, None] - mu[None, :]) ** 2 / var[None, :])

    for _ in range(iters):
        B = np.exp(logb() - logb().max(1, keepdims=True))
        alpha, c = np.zeros((n, K)), np.zeros(n)
        alpha[0] = pi * B[0]
        c[0] = alpha[0].sum() or 1e-300
        alpha[0] /= c[0]
        for k in range(1, n):
            alpha[k] = (alpha[k - 1] @ A) * B[k]
            c[k] = alpha[k].sum() or 1e-300
            alpha[k] /= c[k]
        beta = np.ones((n, K))
        for k in range(n - 2, -1, -1):
            beta[k] = (A @ (B[k + 1] * beta[k + 1])) / c[k + 1]
        g = alpha * beta
        g /= g.sum(1, keepdims=True)
        xi = np.zeros((K, K))
        for k in range(n - 1):
            x = alpha[k][:, None] * A * (B[k + 1] * beta[k + 1])[None, :]
            s = x.sum()
            if s > 0:
                xi += x / s
        A = xi / np.maximum(xi.sum(1, keepdims=True), 1e-12)
        pi = g[0]
        w = g.sum(0)
        mu = (g * y[:, None]).sum(0) / np.maximum(w, 1e-12)
        var = np.maximum((g * (y[:, None] - mu[None, :]) ** 2).sum(0) / np.maximum(w, 1e-12), 0.05)
        order = np.argsort(mu)
        mu, var, A, pi = mu[order], var[order], A[np.ix_(order, order)], pi[order]
    lb = logb()
    logA = np.log(np.maximum(A, 1e-300))
    delta = np.log(np.maximum(pi, 1e-300)) + lb[0]
    back = np.zeros((n, K), dtype=np.int64)
    for k in range(1, n):
        m = delta[:, None] + logA
        back[k] = m.argmax(0)
        delta = m.max(0) + lb[k]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for k in range(n - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    return path


def agreement(l1: np.ndarray, l2: np.ndarray) -> float:
    """Share of frames with the same state."""
    l1, l2 = np.asarray(l1), np.asarray(l2)
    return float((l1 == l2).mean()) if l1.size and l1.size == l2.size else NAN


__all__ = ["EXIT_HORIZON_S", "METRICS", "METRIC_UNITS", "agreement", "binary_segmentation", "changepoint_labels", "falling_crossing", "first_time",
           "hmm_labels", "runs", "threshold_labels", "transitions"]
