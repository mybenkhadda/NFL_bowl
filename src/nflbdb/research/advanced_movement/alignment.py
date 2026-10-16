"""Four ways to put sequences of different length and start state on a common axis (notebook 01).

    elapsed_time      value at t seconds after the first recorded frame.            axis: s
    normalized_time   value at a fraction u of the recorded duration.               axis: 0 … 1
    distance          value after d yards of path since the first recorded frame.   axis: yd
    speed_phase       value at the moment speed first reaches v (up to the peak).   axis: yd/s

None is physically neutral, and the notebook must say what each one removes:
    elapsed_time      keeps the clock, but frame 0 is where the RECORDING starts — 58.5 % of forty attempts start
                      with the athlete already moving (sprint rescue), so two attempts of one athlete are shifted.
    normalized_time   removes the duration, i.e. most of what the stopwatch measures; a late start still shifts it.
    distance          the origin is again the first recorded frame, so a late start shifts the curve along the axis.
    speed_phase       anchored on the athlete's STATE, so it does not depend on where the recording starts; it
                      discards how long the athlete took to reach each speed unless ``t_rel`` (re-zeroed at the
                      reference speed) is kept as a channel, and it is undefined after the speed peak.

Every method is a linear interpolation INSIDE one sequence, between real frames only. Grid points outside the
recorded range are NaN with ``mask`` False — never extrapolated, never padded with a value. ``t_orig`` keeps the
original clock at every grid point, so an aligned curve can always be traced back to its frames.

Comparison criterion (no NFL value involved): how repeatable is a channel's aligned curve between attempts of the
same athlete? Pointwise one-way variance components (``sprint_rescue.stats.stratified_icc``) are summed over the
grid:  integrated ICC = Σ σ²_between / Σ (σ²_between + σ²_within).  A high value is necessary, not sufficient: an
alignment can be repeatable because it erased the differences between athletes that matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from ..sprint_rescue.stats import stratified_icc
from .config import AdvancedConfig
from .sequences import SequenceError, SequenceSet

S, I, F = pl.String(), pl.Int64(), pl.Float64()
CUMULATIVE = ("t_rel", "dist", "along")          # channels re-zeroed at the reference speed by speed_phase
COMPARISON = TableContract(
    "alignment_comparison", "Repeatability and coverage of aligned curves, by alignment method and channel.",
    {"method": S, "channel": S, "athlete_set": S, "scope": S, "n_sequences": I, "n_athletes_repeat": I, "grid_points": I,
     "grid_points_estimable": I, "coverage": F, "icc_integrated": F, "icc_pointwise_median": F, "var_between": F,
     "var_within": F}, key=("method", "channel", "athlete_set", "scope"), area="method × channel × athlete set × scope",
    docs={"athlete_set": "design = decision subset · confirm = the other athletes · all = both (transductive view)",
          "scope": "pooled, or position_adjusted (athletes compared within their position group)",
          "coverage": "share of grid points × sequences with an observed value",
          "icc_integrated": "Σ σ²_between / Σ (σ²_between + σ²_within) over estimable grid points",
          "grid_points_estimable": "grid points with enough repeated athletes"})


@dataclass
class AlignedCurves:
    method: str
    grid: np.ndarray                 # (G,)
    grid_name: str
    grid_unit: str
    channels: tuple[str, ...]
    values: np.ndarray               # (n, C, G) float32, NaN where not observed
    t_orig: np.ndarray               # (n, G) original clock [s] at each grid point, NaN where not observed
    seq_ids: np.ndarray
    nfl_ids: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        """(n, C, G) True where a value was observed."""
        return np.isfinite(self.values)

    def channel(self, name: str) -> np.ndarray:
        return self.values[:, self.channels.index(name), :]

    def subset(self, idx: np.ndarray) -> "AlignedCurves":
        return AlignedCurves(self.method, self.grid, self.grid_name, self.grid_unit, self.channels, self.values[idx],
                             self.t_orig[idx], self.seq_ids[idx], self.nfl_ids[idx])

    def to_long(self, channel: str) -> pl.DataFrame:
        """(seq_id, nfl_id, grid, value, t_orig) for the observed grid points of one channel."""
        v = self.channel(channel)
        i, j = np.nonzero(np.isfinite(v))
        return pl.DataFrame({"seq_id": self.seq_ids[i].astype(str), "nfl_id": self.nfl_ids[i].astype(np.int64),
                             "grid": self.grid[j].astype(float), "value": v[i, j].astype(float),
                             "t_orig": self.t_orig[i, j].astype(float)})


# ============================================================================== interpolation
def increasing(x: np.ndarray) -> np.ndarray:
    """Indices of the frames at which ``x`` reaches a new maximum (a strictly increasing subsequence)."""
    x = np.asarray(x, float)
    if not x.size:
        return np.empty(0, dtype=int)
    run = np.maximum.accumulate(np.where(np.isfinite(x), x, -np.inf))
    keep = np.r_[np.isfinite(x[0]), run[1:] > run[:-1] + 1e-9]
    return np.flatnonzero(keep)


def interp_inside(xq: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation of y(x) at ``xq``; NaN outside [x[0], x[-1]] and where fewer than two points exist.
    ``x`` must be strictly increasing. NaN in ``y`` propagates to the neighbouring grid points (no bridging)."""
    out = np.full(xq.shape, np.nan)
    if x.size < 2:
        return out
    tol = 1e-5          # channels are stored as float32: a grid point equal to the last frame must count as inside
    inside = (xq >= x[0] - tol) & (xq <= x[-1] + tol)
    out[inside] = np.interp(xq[inside], x, y)
    return out


def _align(ss: SequenceSet, channels: Sequence[str], grid: np.ndarray, axis: Callable[[int], tuple[np.ndarray, np.ndarray]],
           method: str, name: str, unit: str, rezero: float | None = None) -> AlignedCurves:
    """``axis(i)`` returns (frame indices kept, strictly increasing axis value at those frames) for sequence i."""
    values = np.full((ss.n, len(channels), grid.size), np.nan, dtype=np.float32)
    t_orig = np.full((ss.n, grid.size), np.nan, dtype=np.float32)
    for i in range(ss.n):
        idx, x = axis(i)
        if idx.size < 2:
            continue
        k = int(ss.lengths[i])
        t_orig[i] = interp_inside(grid, x, ss.t[i, :k][idx].astype(float))
        for c, ch in enumerate(channels):
            y = ss.series(i, ch)[idx]
            v = interp_inside(grid, x, y)
            if rezero is not None and ch in CUMULATIVE:
                ref = interp_inside(np.array([rezero]), x, y)[0]
                v = v - ref                                  # NaN when the reference state was not recorded
            values[i, c] = v
    return AlignedCurves(method, grid, name, unit, tuple(channels), values, t_orig, ss.seq_ids, ss.nfl_ids)


def _require(ss: SequenceSet, names: Sequence[str]) -> None:
    missing = [c for c in names if c not in ss.channels]
    if missing:
        raise SequenceError(f"alignment needs channel(s) {missing}, absent from the sequence set {ss.channels}")


# ============================================================================== the four alignments
def align_elapsed_time(ss: SequenceSet, acfg: AdvancedConfig, channels: Sequence[str]) -> AlignedCurves:
    _require(ss, ["t_rel", *channels])
    grid = np.arange(0.0, acfg.time_max_s + 1e-9, acfg.frame_dt_s)

    def axis(i: int):
        t = ss.series(i, "t_rel")
        idx = increasing(t)
        return idx, t[idx]
    return _align(ss, channels, grid, axis, "elapsed_time", "time since the first recorded frame", "s")


def align_normalized_time(ss: SequenceSet, acfg: AdvancedConfig, channels: Sequence[str]) -> AlignedCurves:
    _require(ss, ["t_rel", *channels])
    grid = np.linspace(0.0, 1.0, acfg.norm_points)

    def axis(i: int):
        t = ss.series(i, "t_rel")
        idx = increasing(t)
        span = t[idx][-1] - t[idx][0] if idx.size > 1 else 0.0
        return (idx, (t[idx] - t[idx][0]) / span) if span > 0 else (np.empty(0, int), np.empty(0))
    return _align(ss, channels, grid, axis, "normalized_time", "fraction of the recorded duration", "–")


def align_distance(ss: SequenceSet, acfg: AdvancedConfig, channels: Sequence[str]) -> AlignedCurves:
    _require(ss, ["dist", *channels])
    grid = np.arange(0.0, acfg.dist_max_yd + 1e-9, acfg.dist_step_yd)

    def axis(i: int):
        d = ss.series(i, "dist")
        idx = increasing(d)
        return idx, d[idx]
    return _align(ss, channels, grid, axis, "distance", "path length since the first recorded frame", "yd")


def align_speed_phase(ss: SequenceSet, acfg: AdvancedConfig, channels: Sequence[str]) -> AlignedCurves:
    """Rising phase only: frames up to the speed peak, at the first time each speed level is reached. Cumulative
    channels (t_rel, dist, along) are re-zeroed at ``phase_ref_speed``; they are NaN for a sequence whose recording
    starts above that speed."""
    _require(ss, ["speed", *channels])
    grid = np.arange(acfg.speed_lo, acfg.speed_hi + 1e-9, acfg.speed_step)

    def axis(i: int):
        v = ss.series(i, "speed")
        if not np.isfinite(v).any():
            return np.empty(0, int), np.empty(0)
        idx = increasing(v[: int(np.nanargmax(v)) + 1])
        return idx, v[idx]
    return _align(ss, channels, grid, axis, "speed_phase", "speed level reached (rising phase)", "yd/s",
                  rezero=acfg.phase_ref_speed)


ALIGNERS: dict[str, Callable[[SequenceSet, AdvancedConfig, Sequence[str]], AlignedCurves]] = {
    "elapsed_time": align_elapsed_time, "normalized_time": align_normalized_time, "distance": align_distance,
    "speed_phase": align_speed_phase}
AXIS_CHANNEL = {"elapsed_time": "t_rel", "normalized_time": None, "distance": "dist", "speed_phase": "speed"}


def align_all(ss: SequenceSet, acfg: AdvancedConfig, channels: Sequence[str]) -> dict[str, AlignedCurves]:
    """Every alignment, each without the channel that is its own axis (it would be the identity)."""
    return {m: fn(ss, acfg, [c for c in channels if c != AXIS_CHANNEL[m]]) for m, fn in ALIGNERS.items()}


# ============================================================================== repeatability of aligned curves
def curve_reliability(curves: AlignedCurves, channel: str, strata: np.ndarray | None, min_athletes: int) -> dict[str, Any]:
    """Variance components of one channel's aligned curve along the grid (athletes with ≥ 2 attempts carry the
    within-athlete variance; a grid point is estimable when ≥ ``min_athletes`` of them are observed there)."""
    v = curves.channel(channel)
    vb, vw, icc, n_rep = [], [], [], 0
    for j in range(curves.grid.size):
        ok = np.isfinite(v[:, j])
        if ok.sum() < 3:
            continue
        ath = curves.nfl_ids[ok]
        _, counts = np.unique(ath, return_counts=True)
        if (counts >= 2).sum() < min_athletes:
            continue
        r = stratified_icc(v[ok, j].astype(float), ath, None if strata is None else strata[ok])
        if np.isfinite(r["var_between"]) and np.isfinite(r["var_within"]):
            vb.append(r["var_between"]), vw.append(r["var_within"])
            if np.isfinite(r["icc"]):        # undefined where a cumulative channel is constant (e.g. dist at its origin)
                icc.append(r["icc"])
            n_rep = max(n_rep, int((counts >= 2).sum()))
    tot = float(np.sum(vb) + np.sum(vw)) if vb else 0.0
    return {"n_sequences": int(np.isfinite(v).any(1).sum()), "n_athletes_repeat": n_rep, "grid_points": int(curves.grid.size),
            "grid_points_estimable": len(vb), "coverage": float(np.isfinite(v).mean()) if v.size else None,
            "icc_integrated": float(np.sum(vb) / tot) if tot > 0 else None,
            "icc_pointwise_median": float(np.median(icc)) if icc else None,
            "var_between": float(np.mean(vb)) if vb else None, "var_within": float(np.mean(vw)) if vw else None}


def compare_alignments(aligned: dict[str, AlignedCurves], positions: dict[int, str], athlete_sets: dict[str, Sequence[int]],
                       acfg: AdvancedConfig) -> pl.DataFrame:
    """``positions``: nfl_id → position group. ``athlete_sets``: name → athlete ids (e.g. design / confirm / all)."""
    rows = []
    for method, curves in aligned.items():
        pos = np.array([positions.get(int(i), "?") for i in curves.nfl_ids])
        for set_name, ids in athlete_sets.items():
            keep = np.isin(curves.nfl_ids, np.asarray(list(ids), dtype=np.int64))
            if keep.sum() < 3:
                continue
            sub = curves.subset(keep)
            for ch in curves.channels:
                for scope, strata in (("pooled", None), ("position_adjusted", pos[keep])):
                    rows.append({"method": method, "channel": ch, "athlete_set": set_name, "scope": scope,
                                 **curve_reliability(sub, ch, strata, acfg.min_athletes_curve)})
    return pl.DataFrame(rows, schema=COMPARISON.columns) if rows else COMPARISON.empty()
