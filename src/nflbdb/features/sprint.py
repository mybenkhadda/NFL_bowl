"""Forty-yard-dash decomposition (Analysis 1): mono-exponential sprint model + tracked splits.

Model (Furusawa et al. 1927; Chelly & Denis 2001; widely used in sprint profiling)::

    v(t) = vmax · (1 − exp(−(t − t0)/τ))   for t ≥ t0,  0 before
    d(t) = vmax · (u − τ(1 − exp(−u/τ))),   u = max(t − t0, 0)

fitted by least squares on derived speed from the first valid frame up to the peak-speed frame.
vmax is solved in closed form for each (τ, t0) on a grid, so the fit is deterministic and needs no
optimiser (no SciPy). ``a0 = vmax / τ`` is the model's initial acceleration (yd/s²). No force or
power is derived: that requires body-mass and drag assumptions this project has not validated.

Splits: distance along the run axis (unit vector first → last smoothed position) from the model
start t0. When the recording starts after the athlete is already moving (``onset_censored``; M1
evidence: the forty example starts at ≈ 1 yd/s), the distance covered before the first frame is
taken from the fitted model (``pre_distance_yds``) — an extrapolation, flagged as such.
Times are TRACKED times (prefix ``trk_``) measured from the MODEL start t0, which is not the
official timing convention (official electronic timing starts at the athlete's first movement /
hand release). Agreement with official ``cmb_forty`` / ``cmb_ten_yd_split`` is an empirical
question answered in ``timing_validation.csv`` — no tracked split is presented as an official split.
Flying intervals (``trk_t10_20_s``, ``trk_t20_40_s``) do not depend on the start convention.

Fit window: from one frame before speed first reaches ``onset_speed`` to the peak-speed frame
(a stationary lead-in is excluded). ``fit_ok`` additionally requires an interior grid optimum,
``0 < vmax ≤ max_model_vmax`` and ``vmax ≤ 1.10 × observed peak`` — a forty recording that stops
while the athlete still accelerates cannot support a much higher asymptote.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .config import MovementConfig

SPRINT_COLUMNS = (
    "n_frames", "n_fit_frames", "first_speed", "last_speed", "speed_max_obs", "peak_at_end", "run_distance_yds",
    "track_distance_yds", "pre_distance_yds", "model_vmax", "model_tau_s", "model_t0_rel_s", "model_a0",
    "model_rmse", "model_r2", "fit_at_grid_edge", "fit_ok", "fit_start_rel_s", "vmax_extrapolation_ratio",
    "trk_t10_s", "trk_t20_s", "trk_t40_s", "trk_t10_20_s", "trk_t20_40_s", "speed_at_10", "speed_at_20", "speed_at_40",
    "dist_to_90pct_vmax_yds", "split_extrapolated", "run_incomplete")
# ``trk_*`` = TRACKED times (model-start convention). Official Combine times live only in
# gold.mart_athlete_research_base as ``cmb_*``; the two are never mixed in one column.
MAX_VMAX_EXTRAPOLATION = 1.10   # model vmax > 110 % of observed peak → asymptote not supported by data


def _grid(spec: tuple[float, float, float]) -> np.ndarray:
    lo, hi, step = spec
    return np.arange(lo, hi + step / 2, step)


def fit_mono_exponential(t: np.ndarray, v: np.ndarray, tau_grid: np.ndarray, t0_grid: np.ndarray) -> dict:
    """Grid least-squares fit; returns vmax, tau, t0, rmse, r2 and whether the optimum hit a grid edge."""
    T = t[None, None, :]
    g = 1.0 - np.exp(-np.clip(T - t0_grid[None, :, None], 0.0, None) / tau_grid[:, None, None])
    sgv = (g * v).sum(-1)
    sgg = (g * g).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vmax = np.where(sgg > 0, sgv / sgg, np.nan)
        sse = np.where(sgg > 0, (v * v).sum() - sgv**2 / sgg, np.inf)
    i, j = np.unravel_index(int(np.argmin(sse)), sse.shape)
    sst = float(((v - v.mean()) ** 2).sum())
    best = float(max(sse[i, j], 0.0))
    return {"vmax": float(vmax[i, j]), "tau": float(tau_grid[i]), "t0": float(t0_grid[j]),
            "rmse": math.sqrt(best / v.size), "r2": (1.0 - best / sst) if sst > 0 else float("nan"),
            "at_edge": i in (0, tau_grid.size - 1) or j in (0, t0_grid.size - 1)}


def model_distance(t: np.ndarray | float, vmax: float, tau: float, t0: float) -> np.ndarray:
    u = np.clip(np.asarray(t, dtype=float) - t0, 0.0, None)
    return vmax * (u - tau * (1.0 - np.exp(-u / tau)))


def _crossing(total: np.ndarray, t: np.ndarray, v: np.ndarray, target: float, tol: float) -> tuple[float, bool]:
    """Time at which ``total`` first reaches ``target`` (linear interpolation); within ``tol`` of
    the end it is extrapolated at the last speed (flagged)."""
    idx = np.nonzero(total >= target)[0]
    if idx.size:
        k = int(idx[0])
        if k == 0:
            return float(t[0]), False
        f = (target - total[k - 1]) / (total[k] - total[k - 1])
        return float(t[k - 1] + f * (t[k] - t[k - 1])), False
    if target - total[-1] <= tol and v[-1] > 0:
        return float(t[-1] + (target - total[-1]) / v[-1]), True
    return float("nan"), False


def sprint_profile(t: np.ndarray, x: np.ndarray, y: np.ndarray, v: np.ndarray, mcfg: MovementConfig) -> dict:
    """Profile of one sprint attempt from sorted, valid frames of a single segment."""
    nan = float("nan")
    n = t.size
    ip = int(np.argmax(v))
    out = {c: None for c in SPRINT_COLUMNS}
    out.update(n_frames=n, n_fit_frames=0, first_speed=float(v[0]), last_speed=float(v[-1]),
               speed_max_obs=float(v[ip]), peak_at_end=(n - 1 - ip) < mcfg.peak_edge_frames)
    axis = np.array([x[-1] - x[0], y[-1] - y[0]])
    norm = float(np.hypot(*axis))
    if n < 5 or norm <= 0:
        out.update(fit_ok=False, run_incomplete=True)
        return out
    u = axis / norm
    d_track = (x - x[0]) * u[0] + (y - y[0]) * u[1]
    # Fit window starts one frame before speed first reaches onset_speed: a stationary lead-in
    # (athlete set at the line while recording runs) would otherwise push t0 beyond the t0 grid.
    moving = np.nonzero(v >= mcfg.onset_speed)[0]
    i0 = max(0, int(moving[0]) - 1) if moving.size else 0
    i0 = min(i0, ip)
    tf, vf = t[i0: ip + 1], v[i0: ip + 1]
    fit = fit_mono_exponential(tf, vf, _grid(mcfg.tau_grid), tf[0] + _grid(mcfg.t0_offset_grid))
    ratio = fit["vmax"] / float(v[ip]) if v[ip] > 0 and np.isfinite(fit["vmax"]) else nan
    fit_ok = (tf.size >= mcfg.min_fit_frames and not fit["at_edge"] and np.isfinite(fit["vmax"])
              and 0 < fit["vmax"] <= mcfg.max_model_vmax and np.isfinite(ratio) and ratio <= MAX_VMAX_EXTRAPOLATION)
    out.update(n_fit_frames=int(tf.size), fit_start_rel_s=float(tf[0] - t[0]), vmax_extrapolation_ratio=ratio)
    t0 = fit["t0"]
    if t0 < t[0]:
        pre = float(model_distance(t[0], fit["vmax"], fit["tau"], t0))
        total = d_track + pre
    else:  # athlete still at the start on the first frame(s): measure from the position at t0
        pre = 0.0
        total = d_track - float(np.interp(t0, t, d_track))
    splits, extrap = {}, False
    for dist in mcfg.split_distances:
        tc, ex = _crossing(total, t, v, dist, mcfg.split_tolerance_yds)
        splits[dist] = (tc - t0) if np.isfinite(tc) else nan
        extrap |= ex
        out[f"speed_at_{dist:g}"] = float(np.interp(tc, t, v)) if np.isfinite(tc) and tc <= t[-1] else (
            float(v[-1]) if np.isfinite(tc) else nan)
    t10, t20, t40 = splits.get(10.0, nan), splits.get(20.0, nan), splits.get(40.0, nan)
    i90 = np.nonzero(v >= 0.9 * fit["vmax"])[0] if fit_ok else np.array([], dtype=int)
    out.update(
        run_distance_yds=float(total[-1]), track_distance_yds=float(d_track[-1]), pre_distance_yds=pre,
        model_vmax=fit["vmax"], model_tau_s=fit["tau"], model_t0_rel_s=t0 - float(t[0]),
        model_a0=fit["vmax"] / fit["tau"], model_rmse=fit["rmse"], model_r2=fit["r2"],
        fit_at_grid_edge=fit["at_edge"], fit_ok=bool(fit_ok),
        trk_t10_s=t10, trk_t20_s=t20, trk_t40_s=t40, trk_t10_20_s=t20 - t10, trk_t20_40_s=t40 - t20,
        dist_to_90pct_vmax_yds=float(total[i90[0]]) if i90.size else nan,
        split_extrapolated=bool(extrap),
        run_incomplete=bool(total[-1] < max(mcfg.split_distances) - mcfg.split_tolerance_yds),
    )
    return out


def sprint_profiles(kin: pl.DataFrame, keys: list[str], mcfg: MovementConfig, context: list[str] | None = None
                    ) -> pl.DataFrame:
    """One row per sprint attempt (drill ∈ ``mcfg.sprint_drills``); longest segment only."""
    ctx = [c for c in (context or []) if c in kin.columns]
    f = kin.filter(pl.col("valid_kin") & pl.col("drill").is_in(list(mcfg.sprint_drills)))
    if f.is_empty():
        return pl.DataFrame(schema={**{c: kin.schema[c] for c in [*keys, *ctx]}, "n_segments_used": pl.Int64})
    longest = (f.group_by([*keys, "seg_idx"]).agg(pl.len().alias("_n"))
               .sort([*keys, "_n", "seg_idx"], descending=[*[False] * len(keys), True, False])
               .group_by(keys, maintain_order=True).first().select(*keys, "seg_idx"))
    f = f.join(longest, on=[*keys, "seg_idx"], how="inner").sort([*keys, "t_s"])
    rows = []
    for part in f.partition_by(keys, maintain_order=True):
        r = {k: part[k][0] for k in [*keys, *ctx]}
        r.update(sprint_profile(part["t_s"].to_numpy().astype(float), part["x_s"].to_numpy().astype(float),
                                part["y_s"].to_numpy().astype(float), part["speed"].to_numpy().astype(float), mcfg))
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        [pl.col(c).cast(pl.Float64) for c in SPRINT_COLUMNS
         if c not in ("n_frames", "n_fit_frames", "peak_at_end", "fit_at_grid_edge", "fit_ok", "split_extrapolated",
                      "run_incomplete")])
