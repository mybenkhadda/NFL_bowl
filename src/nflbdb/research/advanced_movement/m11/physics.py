"""Shared physical definitions of Milestone 11 (pure NumPy; identical code on Combine and NFL traces).

Units: yd, s, yd/s, yd/s², degrees. Frames are samples of ONE contiguous segment in time order.

Speed bands (11B · B1)
    The first-crossing rule of the 09A atlas (``movement_demands.crossing_time``): a band lo → hi is entered when speed
    first rises through lo and left when it first rises through hi. A crossing that is not in the recording (trace starts
    at or above lo, or never reaches hi) gives NO value — censoring is explicit, never extrapolated.

Sprint model (11B · B3)
    v(t) = Vmax (1 − exp(−(t − t0)/τ)) for t ≥ t0. Equivalent phase-plane form a(v) = (Vmax − v)/τ, i.e. a LINEAR
    acceleration–speed relation with intercept A0 = Vmax/τ and speed intercept V0 = Vmax. The time form needs the onset
    t0 (identifiable only when the start is in the recording); the phase-plane form is offset-free. Both are fitted;
    neither yields a force without a body-mass and drag assumption (none is made here).

Envelopes (11B · B2)
    The OBSERVED upper envelope of an athlete's game a(v): per speed bin the q-quantile of tangential acceleration
    (frames are dependent; the quantile is descriptive). Candidate shapes: linear A0 (1 − v/V0) by weighted least squares
    and by a Huber fit, a power family A0 (1 − (v/V0)^k), and a nonparametric decreasing curve (PAVA). An observed envelope
    is what the athlete DID under game opportunity, never a physiological ceiling.

Geometry (11C)
    Turn angle = signed angle between the mean velocity vectors of the windows before and after a transition (+ = counter-
    clockwise in the tracking frame; the handedness of the Combine frame is unverified, so asymmetry uses magnitudes per
    side only where both sides are observed). Lateral acceleration a_n = (vx·ay − vy·ax)/|v|.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")


# ============================================================================== derivatives (estimation checks)
def central_accel(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """dv/dt by second-order central differences on an irregular clock (one-sided at the ends)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    if t.size < 3:
        return np.full(t.size, NAN)
    return np.gradient(v, t, edge_order=2)


def speed_from_xy(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vx, vy, speed) from positions by central differences (raw; the warehouse frames are already smoothed)."""
    vx, vy = np.gradient(np.asarray(x, float), t, edge_order=2), np.gradient(np.asarray(y, float), t, edge_order=2)
    return vx, vy, np.hypot(vx, vy)


# ============================================================================== speed bands
def crossing_time(t: np.ndarray, v: np.ndarray, level: float) -> float:
    """Time at which ``v`` FIRST rises through ``level`` (linear interpolation). NaN when not recorded. Same rule as
    ``movement_demands.crossing_time`` (kept identical on purpose; checked by ``validation``)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    if v.size < 2 or not np.isfinite(v[0]) or v[0] >= level:
        return NAN
    idx = np.flatnonzero(v >= level)
    if not idx.size:
        return NAN
    j = int(idx[0])
    v0, v1 = v[j - 1], v[j]
    if not np.isfinite(v0) or v1 <= v0:
        return NAN
    return float(t[j - 1] + (level - v0) / (v1 - v0) * (t[j] - t[j - 1]))


def band_metrics(t: np.ndarray, v: np.ndarray, a: np.ndarray | None, lo: float, hi: float, min_frames: int = 2) -> dict[str, Any]:
    """Physical description of ONE passage through the band lo → hi.

    time_s       time between the first crossings of lo and hi [s]
    mean_accel   (hi − lo) / time_s [yd/s²]
    speed_gain   hi − lo when both crossings are recorded [yd/s]
    smoothness   coefficient of variation of a_tan on frames strictly inside the band (lower = smoother); NaN with
                 fewer than ``min_frames`` interior frames
    start_censored / end_censored   the crossing of lo / hi is not in the recording"""
    t, v = np.asarray(t, float), np.asarray(v, float)
    a_ = central_accel(t, v) if a is None else np.asarray(a, float)
    t_lo, t_hi = crossing_time(t, v, lo), crossing_time(t, v, hi)
    out = {"t_lo": t_lo, "t_hi": t_hi, "time_s": NAN, "mean_accel": NAN, "speed_gain": NAN, "smoothness": NAN, "n_inside": 0,
           "start_censored": bool(v.size == 0 or not np.isfinite(v[0]) or v[0] >= lo), "end_censored": bool(not np.isfinite(t_hi))}
    if np.isfinite(t_lo) and np.isfinite(t_hi) and t_hi > t_lo:
        dt = t_hi - t_lo
        inside = (t > t_lo) & (t < t_hi) & np.isfinite(a_)
        out |= {"time_s": float(dt), "mean_accel": float((hi - lo) / dt), "speed_gain": float(hi - lo), "n_inside": int(inside.sum())}
        if inside.sum() >= min_frames:
            m = float(np.mean(a_[inside]))
            out["smoothness"] = float(np.std(a_[inside]) / m) if m > 1e-9 else NAN
    return out


def band_profile(t: np.ndarray, v: np.ndarray, a: np.ndarray | None, bands: Sequence[tuple[float, float]], min_frames: int = 2) -> dict[str, float]:
    """Flat {band_<lo>_<hi>_<metric>: value} for every band (Combine attempt or NFL episode)."""
    out: dict[str, float] = {}
    for lo, hi in bands:
        m = band_metrics(t, v, a, lo, hi, min_frames)
        tag = f"band_{lo:g}_{hi:g}"
        for k in ("time_s", "mean_accel", "smoothness"):
            out[f"{tag}_{k}"] = m[k]
        out[f"{tag}_censored"] = float(m["start_censored"] or m["end_censored"])
    return out


# ============================================================================== sprint model
def sprint_speed(t: np.ndarray, vmax: float, tau: float, t0: float = 0.0) -> np.ndarray:
    s = np.clip(np.asarray(t, float) - t0, 0.0, None)
    return vmax * (1.0 - np.exp(-s / tau))


def sprint_distance(t: np.ndarray, vmax: float, tau: float, t0: float = 0.0) -> np.ndarray:
    s = np.clip(np.asarray(t, float) - t0, 0.0, None)
    return vmax * (s - tau * (1.0 - np.exp(-s / tau)))


def time_to_distance(d: float, vmax: float, tau: float) -> float:
    """Model time (from onset) to cover ``d`` yards — a plausibility check against an official time, which also contains
    the reaction and start-detection conventions of the timing system."""
    if not (np.isfinite(vmax) and np.isfinite(tau)) or vmax <= 0 or tau <= 0:
        return NAN
    lo, hi = 0.0, 60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sprint_distance(np.array([mid]), vmax, tau)[0] < d:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def fit_sprint_time(t: np.ndarray, v: np.ndarray, tau_grid: tuple[float, float, int], t0_grid: tuple[float, float, int],
                    fit_offset: bool, min_frames: int = 15) -> dict[str, Any]:
    """Least-squares fit of v(t) on the rising phase (frames up to the speed peak). For each (τ, t0) on the grid Vmax has a
    closed form. ``fit_offset = False`` fixes t0 = first frame (used when the onset is NOT in the recording, where t0 is
    not identifiable — the phase-plane fit is then the primary estimate)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    out = {"vmax": NAN, "tau": NAN, "t0": NAN, "a0": NAN, "rmse": NAN, "r2": NAN, "n_frames": int(t.size), "grid_edge": False,
           "offset_fitted": bool(fit_offset), "ok": False}
    if t.size < min_frames:
        return out
    k = int(np.argmax(v)) + 1
    t, v = t[:k], v[:k]
    if t.size < max(5, min_frames // 2):
        return out
    taus = np.linspace(*tau_grid)
    t0s = np.linspace(t[0] + t0_grid[0], t[0] + t0_grid[1], t0_grid[2]) if fit_offset else np.array([t[0]])
    best = (np.inf, NAN, NAN, NAN)
    for t0 in t0s:
        s = np.clip(t - t0, 0.0, None)
        for tau in taus:
            g = 1.0 - np.exp(-s / tau)
            gg = float(g @ g)
            if gg <= 0:
                continue
            vm = float(g @ v) / gg
            sse = float(((v - vm * g) ** 2).sum())
            if sse < best[0]:
                best = (sse, vm, tau, t0)
    sse, vm, tau, t0 = best
    if not np.isfinite(sse):
        return out
    sst = float(((v - v.mean()) ** 2).sum())
    out |= {"vmax": vm, "tau": tau, "t0": t0, "a0": vm / tau, "rmse": float(np.sqrt(sse / v.size)), "r2": 1.0 - sse / sst if sst > 0 else NAN,
            "grid_edge": bool(np.isclose(tau, taus[0]) or np.isclose(tau, taus[-1]) or (fit_offset and (np.isclose(t0, t0s[0]) or np.isclose(t0, t0s[-1])))),
            "ok": True}
    return out


def fit_sprint_phase(v: np.ndarray, a: np.ndarray, vmin: float = 1.0) -> dict[str, Any]:
    """Offset-free phase-plane fit a = A0 − (A0/V0)·v on rising-phase frames (up to the speed peak, speed ≥ vmin).
    Returns A0 [yd/s²], V0 [yd/s] and τ = V0/A0 [s]; NaN when the slope is not negative."""
    v, a = np.asarray(v, float), np.asarray(a, float)
    out = {"A0": NAN, "V0": NAN, "tau": NAN, "rmse": NAN, "n_frames": 0, "ok": False}
    if v.size < 5:
        return out
    k = int(np.nanargmax(v)) + 1
    v, a = v[:k], a[:k]
    ok = np.isfinite(v) & np.isfinite(a) & (v >= vmin)
    if ok.sum() < 5:
        return out
    b, c = np.polyfit(v[ok], a[ok], 1)
    out["n_frames"] = int(ok.sum())
    if b < 0 and c > 0:
        V0, A0 = -c / b, c
        out |= {"A0": float(A0), "V0": float(V0), "tau": float(V0 / A0), "rmse": float(np.sqrt(np.mean((a[ok] - (c + b * v[ok])) ** 2))), "ok": True}
    return out


# ============================================================================== envelopes
def hist_quantile(edges: np.ndarray, counts: np.ndarray, q: float) -> float:
    """q-quantile of a histogram (linear interpolation inside the bin)."""
    counts = np.asarray(counts, float)
    n = counts.sum()
    if n <= 0:
        return NAN
    cdf = np.cumsum(counts) / n
    j = int(np.searchsorted(cdf, q, side="left"))
    j = min(j, counts.size - 1)
    prev = cdf[j - 1] if j > 0 else 0.0
    frac = (q - prev) / (cdf[j] - prev) if cdf[j] > prev else 0.5
    return float(edges[j] + frac * (edges[j + 1] - edges[j]))


def envelope_points(H: np.ndarray, speed_edges: np.ndarray, accel_edges: np.ndarray, q: float, min_frames: int, vmin: float = 0.0
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(speed bin mids, q-quantile of a_tan, frames) for speed bins with ≥ ``min_frames`` frames. ``H``: (speed bins, accel bins)."""
    mids = 0.5 * (speed_edges[:-1] + speed_edges[1:])
    n = H.sum(1)
    keep = (n >= min_frames) & (mids >= vmin)
    aq = np.array([hist_quantile(accel_edges, H[i], q) if keep[i] else NAN for i in range(H.shape[0])])
    return mids[keep], aq[keep], n[keep].astype(float)


def fit_linear_envelope(v: np.ndarray, a: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    """Weighted least squares a = A0 + b v. V0 = −A0/b when b < 0 (else NaN: no speed-limited decline observed)."""
    v, a = np.asarray(v, float), np.asarray(a, float)
    w = np.ones_like(v) if w is None else np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(a) & (w > 0)
    if ok.sum() < 3:
        return {"A0": NAN, "V0": NAN, "slope": NAN, "rmse": NAN}
    W = np.sqrt(w[ok])
    X = np.c_[np.ones(ok.sum()), v[ok]] * W[:, None]
    beta = np.linalg.lstsq(X, a[ok] * W, rcond=None)[0]
    A0, b = float(beta[0]), float(beta[1])
    res = a[ok] - (A0 + b * v[ok])
    return {"A0": A0, "V0": float(-A0 / b) if b < 0 else NAN, "slope": b, "rmse": float(np.sqrt(np.average(res ** 2, weights=w[ok])))}


def fit_huber_envelope(v: np.ndarray, a: np.ndarray, w: np.ndarray | None = None, k: float = 1.345, iters: int = 50) -> dict[str, float]:
    """Robust linear envelope (Huber IRLS on the bin quantiles)."""
    v, a = np.asarray(v, float), np.asarray(a, float)
    w0 = np.ones_like(v) if w is None else np.asarray(w, float)
    fit = fit_linear_envelope(v, a, w0)
    if not np.isfinite(fit["A0"]):
        return fit
    for _ in range(iters):
        r = a - (fit["A0"] + fit["slope"] * v)
        s = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r))) or 1e-6
        u = np.abs(r) / (k * s)
        new = fit_linear_envelope(v, a, w0 * np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12)))
        if abs(new["A0"] - fit["A0"]) < 1e-8 and abs(new["slope"] - fit["slope"]) < 1e-10:
            fit = new
            break
        fit = new
    return fit


def fit_power_envelope(v: np.ndarray, a: np.ndarray, w: np.ndarray | None = None, ks: Sequence[float] = tuple(np.linspace(0.5, 3.0, 26)),
                       v0_grid: int = 60) -> dict[str, float]:
    """Nonlinear family a = A0 (1 − (v/V0)^k); A0 closed form for each (k, V0) on a grid. k = 1 is the linear model."""
    v, a = np.asarray(v, float), np.asarray(a, float)
    w = np.ones_like(v) if w is None else np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(a) & (w > 0)
    out = {"A0": NAN, "V0": NAN, "k": NAN, "rmse": NAN}
    if ok.sum() < 4:
        return out
    v, a, w = v[ok], a[ok], w[ok]
    best = np.inf
    for V0 in np.linspace(v.max() * 1.01, max(v.max() * 2.0, 12.0), v0_grid):
        for k in ks:
            g = 1.0 - (v / V0) ** k
            gg = float((w * g * g).sum())
            if gg <= 0:
                continue
            A0 = float((w * g * a).sum()) / gg
            sse = float((w * (a - A0 * g) ** 2).sum())
            if sse < best and A0 > 0:
                best, out = sse, {"A0": A0, "V0": float(V0), "k": float(k), "rmse": float(np.sqrt(sse / w.sum()))}
    return out


def isotonic_decreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pool-adjacent-violators: the weighted least-squares non-increasing fit of ``y`` (nonparametric envelope)."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            tot = wts[-2] + wts[-1]
            m = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tot
            c = cnt[-2] + cnt[-1]
            vals, wts, cnt = vals[:-2] + [m], wts[:-2] + [tot], cnt[:-2] + [c]
    return np.repeat(vals, cnt)


def quantile_regression(x: np.ndarray, y: np.ndarray, q: float, w: np.ndarray | None = None, iters: int = 200, eps: float = 1e-6
                        ) -> tuple[float, float]:
    """Linear q-quantile regression (intercept, slope) by iteratively reweighted least squares on the pinball loss.
    Used on (speed, a_tan) histogram cells weighted by their frame counts."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w0 = np.ones_like(x) if w is None else np.asarray(w, float)
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.lstsq(X * np.sqrt(w0)[:, None], y * np.sqrt(w0), rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        ww = w0 * np.where(r >= 0, q, 1 - q) / np.maximum(np.abs(r), eps)
        new = np.linalg.lstsq(X * np.sqrt(ww)[:, None], y * np.sqrt(ww), rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    return float(beta[0]), float(beta[1])


def envelope_summary(H: np.ndarray, speed_edges: np.ndarray, accel_edges: np.ndarray, q: float, min_frames: int, min_bins: int, vmin: float,
                     bands: Sequence[tuple[float, float]] = ()) -> dict[str, Any]:
    """Every envelope estimator on one athlete histogram, plus upper-tail acceleration and the envelope at band mid-speeds."""
    v, aq, n = envelope_points(H, speed_edges, accel_edges, q, min_frames, vmin)
    out: dict[str, Any] = {"n_frames": float(H.sum()), "n_bins": int(v.size), "v_lo": float(v.min()) if v.size else NAN,
                           "v_hi": float(v.max()) if v.size else NAN, "upper_tail_accel": float(np.nanmax(aq)) if v.size else NAN, "supported": v.size >= min_bins}
    for m in ("lin", "hub", "pow", "qr"):
        for p in ("A0", "V0"):
            out[f"{m}_{p}"] = NAN
    out["pow_k"] = NAN
    for lo, hi in bands:
        out[f"env_{lo:g}_{hi:g}"] = NAN
    if v.size < min_bins:
        return out
    lin, hub, pw = fit_linear_envelope(v, aq, n), fit_huber_envelope(v, aq, n), fit_power_envelope(v, aq, n)
    out |= {"lin_A0": lin["A0"], "lin_V0": lin["V0"], "lin_rmse": lin["rmse"], "hub_A0": hub["A0"], "hub_V0": hub["V0"], "pow_A0": pw["A0"],
            "pow_V0": pw["V0"], "pow_k": pw["k"], "pow_rmse": pw["rmse"]}
    # frame-level linear quantile regression on the histogram cells of the supported speed range
    mids_v = 0.5 * (speed_edges[:-1] + speed_edges[1:])
    mids_a = 0.5 * (accel_edges[:-1] + accel_edges[1:])
    rows = (H.sum(1) >= min_frames) & (mids_v >= vmin)
    if rows.sum() >= min_bins:
        ii, jj = np.nonzero(H[rows] > 0)
        xv, ya, ww = mids_v[rows][ii], mids_a[jj], H[rows][ii, jj].astype(float)
        b0, b1 = quantile_regression(xv, ya, q, ww)
        out |= {"qr_A0": b0, "qr_V0": -b0 / b1 if b1 < 0 else NAN}
    iso = isotonic_decreasing(aq, n)
    for lo, hi in bands:
        mid = 0.5 * (lo + hi)
        out[f"env_{lo:g}_{hi:g}"] = float(np.interp(mid, v, iso)) if v.min() <= mid <= v.max() else NAN
    return out


# ============================================================================== geometry
def heading_deg(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(np.asarray(vy, float), np.asarray(vx, float)))


def signed_angle_deg(u: np.ndarray, w: np.ndarray) -> float:
    """Signed angle from vector u to vector w (+ = counter-clockwise), in (−180, 180]."""
    cross = u[0] * w[1] - u[1] * w[0]
    dot = u[0] * w[0] + u[1] * w[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def turn_angle(vx: np.ndarray, vy: np.ndarray, before: slice, after: slice, min_speed: float = 1.0) -> float:
    """Signed angle between the mean velocity vectors of two windows (frames slower than ``min_speed`` are ignored because
    their direction is noise). NaN when a window has no fast-enough frame. Valid through a stop (shuttle reversal)."""
    vx, vy = np.asarray(vx, float), np.asarray(vy, float)
    sp = np.hypot(vx, vy)

    def mean_dir(s: slice) -> np.ndarray | None:
        m = (sp[s] >= min_speed) & np.isfinite(vx[s]) & np.isfinite(vy[s])
        if not m.any():
            return None
        u = np.array([vx[s][m].mean(), vy[s][m].mean()])
        n = np.hypot(*u)
        return u / n if n > 0 else None

    u, w = mean_dir(before), mean_dir(after)
    return NAN if u is None or w is None else signed_angle_deg(u, w)


def lateral_accel(vx: np.ndarray, vy: np.ndarray, ax: np.ndarray, ay: np.ndarray, min_speed: float = 1.0) -> np.ndarray:
    """Signed normal (centripetal) acceleration (vx·ay − vy·ax)/|v|; NaN below ``min_speed``."""
    vx, vy, ax, ay = (np.asarray(z, float) for z in (vx, vy, ax, ay))
    sp = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        an = (vx * ay - vy * ax) / sp
    an[sp < min_speed] = NAN
    return an


def tangential_accel(vx: np.ndarray, vy: np.ndarray, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
    vx, vy, ax, ay = (np.asarray(z, float) for z in (vx, vy, ax, ay))
    sp = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sp > 0, (vx * ax + vy * ay) / sp, NAN)


# ============================================================================== measurement error
def attenuation_correct(r: float, rel_x: float, rel_y: float, floor: float = 0.3) -> float:
    """Spearman's correction r / sqrt(rel_x · rel_y); NaN when either reliability is below ``floor`` (the correction then
    amplifies noise more than it removes bias)."""
    if not all(np.isfinite(z) for z in (r, rel_x, rel_y)) or rel_x < floor or rel_y < floor:
        return NAN
    return float(np.clip(r / np.sqrt(rel_x * rel_y), -1.0, 1.0))


__all__ = ["attenuation_correct", "band_metrics", "band_profile", "central_accel", "crossing_time", "envelope_points", "envelope_summary",
           "fit_huber_envelope", "fit_linear_envelope", "fit_power_envelope", "fit_sprint_phase", "fit_sprint_time", "heading_deg", "hist_quantile",
           "isotonic_decreasing", "lateral_accel", "quantile_regression", "signed_angle_deg", "speed_from_xy", "sprint_distance", "sprint_speed",
           "tangential_accel", "time_to_distance", "turn_angle"]
