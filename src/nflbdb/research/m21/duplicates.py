"""M2.1 · Workstream A — are repeated Combine attempts independent observations?

Question: does an extraordinarily stable athlete (e.g. the reported SD(vmax) ≈ 0) reflect genuine
consistency, ties on a coarse/integer feature grid, re-used tracking trajectories or duplicated
source recordings?

Evidence per pair of attempts of the SAME athlete × drill (never a pairwise scan of the whole
dataset; ≤ ``max_attempts_per_group`` attempts per group):

    exact_duplicate     identical raw (x, y) sequence, or ≥ 95 % of frames identical in (ts, x, y)
    translated_copy     identical raw displacement sequence (same path shifted in space and/or time)
    reused_sequence     ≥ 50 % of the moving 1-s displacement windows shared, or ≥ 20 % identical frames
    time_overlap        the two recordings overlap in wall-clock time (physically impossible for 1 athlete)
    near_duplicate      after onset alignment (+ ≤ 1 s lag search): rigid-aligned raw position RMSE AND
                        smoothed-speed RMSE both below measurement-noise thresholds
    consistent          comparable and smoothed-speed RMSE ≤ ``consistent_speed_rmse`` (natural)
    distinct            comparable, otherwise
    not_comparable      no common aligned window ≥ ``min_overlap_frac`` of the shorter attempt

Across groups (hash joins only): identical relative paths shared by different athletes
(``cross_athlete_copy``) and metadata conflicts (same athlete/drill/start time under different
ids; same id pair used by several athletes; repeated attempt numbers; athletes in > 1 cohort).

Groups: ``integrity_failure`` (exact / translated / reused / time overlap / cross-athlete),
``suspicious`` (near duplicate), ``natural`` (consistent / distinct), ``unresolved``
(not comparable), ``unpaired`` (single attempt).

Nothing is deleted. Two exclusion scenarios are flagged for sensitivity analyses:

    exclude_dedup   non-representative members of within-athlete integrity-failure clusters
                    (representative = earliest attempt)
    exclude_strict  as above with near-duplicate edges added, plus every cross-athlete copy

Both raw (x, y, vendor s) and smoothed (x_s, y_s, speed, a_tan) signals are compared, so similarity
created by smoothing can be separated from similarity present in the source data.

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.m21.duplicates                 # --force, --sample-athletes N, --no-figures
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl

from ...data.config import Phase01Config
from ...data.reporting import CATEGORICAL, INK
from ...data.utils import get_logger
from ..combine_movement import ID_COLS, NOT_FEATURES, SPRINT_FEATURES, CombineMovementBuilder
from ..kinematic_build import COMBINE_FRAMES, COMBINE_TRAJ
from . import plotting as P
from .config import DuplicateAuditConfig, M21Config, threshold_table
from .io import FigureSpec, M21Store, TableContract, cast_keys, upstream_state

log = get_logger(__name__)

WORKSTREAM = "A_duplicates"
KEYS = ["event_id", "nfl_id", "attempt"]
FORTY, SHUTTLE, CONE = "FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL"
KEY_DRILLS = (FORTY, SHUTTLE, CONE)
FRAME_INPUT = ["event_id", "nfl_id", "attempt", "drill", "draft_year", "ts_ms", "x", "y", "s", "x_s", "y_s", "speed",
               "a_tan", "valid_kin"]

PAIR_CLASSES = ("exact_duplicate", "translated_copy", "reused_sequence", "time_overlap", "near_duplicate",
                "consistent", "distinct", "not_comparable")
SEVERITY = {"cross_athlete_copy": 7, "exact_duplicate": 7, "translated_copy": 6, "reused_sequence": 5,
            "time_overlap": 4, "near_duplicate": 3, "consistent": 2, "distinct": 1, "not_comparable": 0, "unpaired": -1}
GROUP_OF = {"exact_duplicate": "integrity_failure", "translated_copy": "integrity_failure",
            "reused_sequence": "integrity_failure", "time_overlap": "integrity_failure",
            "cross_athlete_copy": "integrity_failure", "near_duplicate": "suspicious", "consistent": "natural",
            "distinct": "natural", "not_comparable": "unresolved", "unpaired": "unpaired"}
FLAG_CLASSES = ("exact_duplicate", "translated_copy", "reused_sequence", "time_overlap", "near_duplicate")
GROUPS = ("integrity_failure", "suspicious", "natural", "unresolved", "unpaired")

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()

# ============================================================================== contracts
ATTEMPT_AUDIT = TableContract(
    "attempt_duplicate_audit", "One row per Combine attempt: content hashes, pair evidence, metadata conflicts.",
    {"event_id": S, "nfl_id": I, "attempt": I, "drill": S, "draft_year": I, "quality_tier": S, "in_m2": B,
     "attempt_order": I, "group_n_attempts": I, "group_truncated": B, "n_frames": I, "ts_start_ms": I,
     "ts_end_ms": I, "duration_s": F, "path_len_raw_yds": F, "path_len_smooth_yds": F, "peak_speed": F,
     "n_moving_kgrams": I, "raw_xy_hash": S, "rel_xy_hash": S, "n_pairs": I, "n_pairs_comparable": I,
     "worst_pair_class": S, "partners": S, "n_athletes_same_rel_hash": I, "cross_athlete_copy": B,
     "conflicting_ids_same_start": B, "shared_event_attempt_id": B, "repeated_attempt_number": B,
     "multi_cohort_athlete": B},
    key=tuple(KEYS),
    docs={"attempt_order": "chronological rank within athlete × drill over ALL tiers (≠ M2 attempt_order, which "
                           "ranks usable attempts only)",
          "raw_xy_hash": "blake2b-64 of raw x, y on the coord_quantum grid (stable across library versions)",
          "rel_xy_hash": "same after subtracting the first frame (translation invariant)",
          "partners": "flagged partner attempts '<event_id>:<attempt>=<class>' (integrity failures and near duplicates)",
          "n_athletes_same_rel_hash": "athletes sharing this relative path (null = attempt too short/static to judge)"})

_PAIR_COLS: dict[str, pl.DataType] = {
    "pair_id": S, "nfl_id": I, "drill": S, "draft_year": I, "event_id_a": S, "attempt_a": I, "event_id_b": S,
    "attempt_b": I, "attempt_order_a": I, "attempt_order_b": I, "n_frames_a": I, "n_frames_b": I,
    "ts_overlap_s": F, "ts_overlap_frac": F, "shared_ts_frac": F, "shared_frame_frac": F, "shared_kgram_frac": F,
    "raw_xy_hash_equal": B, "rel_xy_hash_equal": B, "comparable": B, "anchor": S, "lag_s": F, "overlap_s": F,
    "n_grid": I, "speed_rmse": F, "vendor_speed_rmse": F, "accel_rmse": F, "speed_corr": F,
    "max_abs_speed_diff": F, "raw_pos_rmse_yds": F, "raw_pos_rmse_rigid_yds": F, "smooth_pos_rmse_rigid_yds": F,
    "path_len_diff_yds": F, "duration_diff_s": F, "peak_speed_diff": F,
    **{f"is_{c}": B for c in FLAG_CLASSES}, "pair_class": S, "integrity_group": S, "evidence": S}
PAIRS = TableContract(
    "attempt_similarity_pairs", "One row per within-athlete × drill attempt pair: similarity metrics + class.",
    {**_PAIR_COLS, "speed_rmse_pct_in_drill": F}, key=("pair_id",),
    docs={"lag_s": "attempt b is shifted by lag_s after onset alignment (b at aligned t = b(t + lag))",
          "speed_rmse": "smoothed speed, aligned grid (yd/s)", "vendor_speed_rmse": "vendor s (raw), aligned grid",
          "raw_pos_rmse_yds": "raw x, y after translation to the first common grid point",
          "raw_pos_rmse_rigid_yds": "raw x, y after optimal rotation + translation (Kabsch)",
          "speed_rmse_pct_in_drill": "percentile rank of speed_rmse among comparable pairs of the drill"})
FLAGS = TableContract(
    "attempt_integrity_flags", "Attempt-level integrity flags + exclusion scenarios for sensitivity analyses.",
    {"event_id": S, "nfl_id": I, "attempt": I, "drill": S, "draft_year": I,
     **{f"flag_{c}": B for c in FLAG_CLASSES}, "flag_cross_athlete_copy": B, "flag_conflicting_ids": B,
     "flag_group_truncated": B, "info_shared_event_attempt_id": B, "info_repeated_attempt_number": B,
     "info_multi_cohort_athlete": B, "integrity_class": S, "integrity_group": S, "dedup_cluster": S,
     "strict_cluster": S, "is_dedup_representative": B, "exclude_dedup": B, "exclude_strict": B, "m21_version": S},
    key=tuple(KEYS), kind="dataset",
    docs={"dedup_cluster": "within-athlete cluster joined by integrity-failure edges (null = singleton)",
          "exclude_dedup": "True → drop in the 'dedup' sensitivity scenario (redundant copy)",
          "exclude_strict": "True → drop in the 'strict' scenario (also near duplicates and cross-athlete copies)"})
SUMMARY = TableContract(
    "duplicate_flag_summary", "Flag prevalence by drill × cohort (cohort 'all' included), Wilson 95 % CI.",
    {"drill": S, "cohort": S, "flag": S, "n_attempts": I, "n_flagged": I, "share": F, "ci_low": F, "ci_high": F,
     "n_athletes": I, "n_athletes_flagged": I}, key=("drill", "cohort", "flag"))
SENSITIVITY = TableContract(
    "duplicate_sensitivity", "Within-athlete variation of every attempt feature under the 3 exclusion scenarios.",
    {"drill": S, "feature": S, "scenario": S, "n_attempts": I, "n_athletes_repeated": I, "median_within_sd": F,
     "pooled_within_sd": F, "share_zero_sd": F, "pooled_within_sd_rel_change": F},
    key=("drill", "feature", "scenario"),
    docs={"pooled_within_sd_rel_change": "pooled_within_sd / pooled_within_sd(all) − 1"})
ZERO_VAR = TableContract(
    "zero_variance_athletes", "Athlete × drill × feature with ≥ 2 attempts and (numerically) zero within-athlete SD.",
    {"nfl_id": I, "drill": S, "draft_year": I, "feature": S, "n_attempts": I, "sd": F, "median": F,
     "integer_valued": B, "quantized": B, "worst_integrity_class": S, "worst_integrity_group": S, "explanation": S},
    key=("nfl_id", "drill", "feature"))
EXAMPLES = TableContract(
    "duplicate_examples", "Deterministically selected example pairs for the duplicate figure (+ selection rule).",
    {"example": S, "pair_id": S, "nfl_id": I, "drill": S, "event_id_a": S, "attempt_a": I, "event_id_b": S,
     "attempt_b": I, "attempt_order_a": I, "attempt_order_b": I, "pair_class": S, "speed_rmse": F,
     "raw_pos_rmse_rigid_yds": F, "n_class_pairs": I, "rule": S}, key=("example",))
SIGNALS = TableContract(
    "duplicate_example_signals", "Aligned signals of the example pairs (figures redraw without raw tracking).",
    {"example": S, "pair_id": S, "t_aligned_s": F, "speed_a": F, "speed_b": F, "vendor_speed_a": F,
     "vendor_speed_b": F, "x_a": F, "y_a": F, "x_b": F, "y_b": F}, key=("example", "t_aligned_s"))
THRESHOLDS = TableContract(
    "m21_thresholds", "All M2.1 thresholds with unit, provenance (established / proposed) and rationale.",
    {"workstream": S, "name": S, "value": S, "unit": S, "provenance": S, "rationale": S}, key=("workstream", "name"))
CONTRACTS = (ATTEMPT_AUDIT, PAIRS, FLAGS, SUMMARY, SENSITIVITY, ZERO_VAR, EXAMPLES, SIGNALS)
FIGURES = ("m21_A_duplicate_examples", "m21_A_pair_similarity", "m21_A_flag_prevalence")


# ============================================================================== per-attempt signals
@dataclass
class Track:
    event_id: str
    nfl_id: int
    attempt: int
    drill: str
    draft_year: int
    ts_ms: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    xq: np.ndarray
    yq: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    speed: np.ndarray
    a_tan: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.event_id}:{self.attempt}"

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    @property
    def v_align(self) -> np.ndarray:
        """Smoothed speed where valid, vendor speed elsewhere (alignment only)."""
        return np.where(np.isfinite(self.speed), self.speed, self.s)


def make_track(g: pl.DataFrame, cfg: DuplicateAuditConfig) -> Track:
    """``g``: frames of ONE attempt with ``FRAME_INPUT`` columns (``valid_kin`` optional)."""
    g = g.sort("ts_ms")
    valid = (g.get_column("valid_kin").fill_null(False).to_numpy() if "valid_kin" in g.columns
             else np.ones(g.height, dtype=bool))

    def f(c: str, smoothed: bool = False) -> np.ndarray:
        a = g.get_column(c).cast(pl.Float64).to_numpy().astype(float)
        return np.where(valid, a, np.nan) if smoothed else a

    x, y = f("x"), f("y")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"attempt {g['event_id'][0]}:{g['attempt'][0]} has null raw x/y — M1 should have removed them")
    ts = g.get_column("ts_ms").cast(pl.Int64).to_numpy()
    return Track(str(g["event_id"][0]), int(g["nfl_id"][0]), int(g["attempt"][0]), str(g["drill"][0]),
                 int(g["draft_year"][0]), ts, (ts - ts[0]) / 1000.0, x, y, f("s"),
                 np.rint(x / cfg.coord_quantum_yds).astype(np.int64), np.rint(y / cfg.coord_quantum_yds).astype(np.int64),
                 f("x_s", True), f("y_s", True), f("speed", True), f("a_tan", True))


def _hash(*arrays: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=8)
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype="<i8").tobytes())
    return h.hexdigest()


def kgrams(tr: Track, cfg: DuplicateAuditConfig) -> set[bytes]:
    """Unique raw-displacement windows of ``kgram_frames`` steps that cover ≥ ``kgram_min_motion_yds``."""
    k = cfg.kgram_frames
    d = np.column_stack([np.diff(tr.xq), np.diff(tr.yq)]).astype("<i8")
    if d.shape[0] < k:
        return set()
    w = np.lib.stride_tricks.sliding_window_view(d, (k, 2)).reshape(-1, k, 2)
    motion = np.hypot(w[..., 0], w[..., 1]).sum(axis=1) * cfg.coord_quantum_yds
    return {row.tobytes() for row in np.ascontiguousarray(w[motion >= cfg.kgram_min_motion_yds])}


def signature(tr: Track, cfg: DuplicateAuditConfig) -> dict[str, Any]:
    steps_raw = np.hypot(np.diff(tr.x), np.diff(tr.y))
    steps_s = np.hypot(np.diff(tr.xs), np.diff(tr.ys))
    finite_speed = tr.speed[np.isfinite(tr.speed)]
    return {"n_frames": int(tr.ts_ms.size), "ts_start_ms": int(tr.ts_ms[0]), "ts_end_ms": int(tr.ts_ms[-1]),
            "duration_s": tr.duration, "path_len_raw_yds": float(steps_raw.sum()),
            "path_len_smooth_yds": float(np.nansum(steps_s)) if np.isfinite(steps_s).any() else None,
            "peak_speed": float(finite_speed.max()) if finite_speed.size else None,
            "raw_xy_hash": _hash(tr.xq, tr.yq), "rel_xy_hash": _hash(tr.xq - tr.xq[0], tr.yq - tr.yq[0])}


# ============================================================================== alignment + metrics
def onset_index(v: np.ndarray, threshold: float, frames: int) -> int | None:
    ok = (np.nan_to_num(v, nan=-np.inf) >= threshold).astype(np.int64)
    if ok.size < frames:
        return None
    run = np.convolve(ok, np.ones(frames, dtype=np.int64), "valid")
    idx = np.flatnonzero(run == frames)
    return int(idx[0]) if idx.size else None


def _interp(t: np.ndarray, v: np.ndarray, g: np.ndarray) -> np.ndarray:
    m = np.isfinite(v)
    if m.sum() < 2:
        return np.full(g.shape, np.nan)
    return np.interp(g, t[m], v[m], left=np.nan, right=np.nan)


def _rmse(d: np.ndarray) -> float | None:
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else None


def rigid_rmse(p: np.ndarray, q: np.ndarray) -> float | None:
    """Point RMSE after the optimal proper rotation + translation of p onto q (2-D Kabsch)."""
    m = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    p, q = p[m], q[m]
    if p.shape[0] < 3:
        return None
    p, q = p - p.mean(axis=0), q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = float(np.sign(np.linalg.det(vt.T @ u.T))) or 1.0
    r = vt.T @ np.diag([1.0, d]) @ u.T
    return float(np.sqrt(np.mean(np.sum((p @ r.T - q) ** 2, axis=1))))


def _translated_rmse(p: np.ndarray, q: np.ndarray) -> float | None:
    m = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    if m.sum() < 2:
        return None
    p, q = p[m], q[m]
    return _rmse(np.hypot(*((p - p[0]) - (q - q[0])).T))


@dataclass(frozen=True)
class Alignment:
    anchor: str          # "onset" (both attempts reach align_speed) or "start" (first frame)
    lag_s: float
    grid: np.ndarray     # aligned time; attempt a at g, attempt b at g + lag
    ta: np.ndarray
    tb: np.ndarray


def align(a: Track, b: Track, cfg: DuplicateAuditConfig) -> Alignment | None:
    va, vb = a.v_align, b.v_align
    ia, ib = onset_index(va, cfg.align_speed, cfg.align_frames), onset_index(vb, cfg.align_speed, cfg.align_frames)
    anchor = "onset" if ia is not None and ib is not None else "start"
    ta = a.t - (a.t[ia] if anchor == "onset" else a.t[0])
    tb = b.t - (b.t[ib] if anchor == "onset" else b.t[0])
    shorter = min(a.duration, b.duration)
    if shorter <= 0:
        return None
    dt = cfg.grid_dt_s
    lags = np.round(np.arange(-cfg.max_lag_s, cfg.max_lag_s + dt / 2, dt), 6)
    lags = lags[np.argsort(np.abs(lags), kind="stable")]     # ties → smallest |lag| (deterministic)
    best: tuple[float, float, np.ndarray] | None = None
    for lag in lags:
        lo, hi = max(ta[0], tb[0] - lag), min(ta[-1], tb[-1] - lag)
        if hi - lo < cfg.min_overlap_frac * shorter:
            continue
        g = lo + dt * np.arange(int(np.floor((hi - lo) / dt + 1e-9)) + 1)
        r = _rmse(_interp(ta, va, g) - _interp(tb, vb, g + lag))
        if r is not None and (best is None or r < best[0] - 1e-12):
            best = (r, float(lag), g)
    if best is None:
        return None
    return Alignment(anchor, best[1], best[2], ta, tb)


def aligned_signals(a: Track, b: Track, al: Alignment) -> dict[str, np.ndarray]:
    g, gb = al.grid, al.grid + al.lag_s
    out = {"t": g}
    for name in ("speed", "s", "a_tan", "x", "y", "xs", "ys"):
        out[f"{name}_a"] = _interp(al.ta, getattr(a, name), g)
        out[f"{name}_b"] = _interp(al.tb, getattr(b, name), gb)
    return out


def _frac(n: int, d: int) -> float | None:
    return n / d if d else None


def compare_pair(a: Track, b: Track, ka: set[bytes], kb: set[bytes], sa: dict, sb: dict,
                 cfg: DuplicateAuditConfig) -> dict[str, Any]:
    n_min = min(a.ts_ms.size, b.ts_ms.size)
    overlap_ms = min(a.ts_ms[-1], b.ts_ms[-1]) - max(a.ts_ms[0], b.ts_ms[0])
    shared_frames = len(set(zip(a.ts_ms.tolist(), a.xq.tolist(), a.yq.tolist()))
                        & set(zip(b.ts_ms.tolist(), b.xq.tolist(), b.yq.tolist())))
    m: dict[str, Any] = {
        "ts_overlap_s": max(0.0, overlap_ms / 1000.0),
        "ts_overlap_frac": _frac(max(0, int(overlap_ms)), int(min(a.ts_ms[-1] - a.ts_ms[0], b.ts_ms[-1] - b.ts_ms[0]))),
        "shared_ts_frac": _frac(np.intersect1d(a.ts_ms, b.ts_ms).size, n_min),
        "shared_frame_frac": _frac(shared_frames, n_min),
        "shared_kgram_frac": _frac(len(ka & kb), min(len(ka), len(kb))),
        "raw_xy_hash_equal": sa["raw_xy_hash"] == sb["raw_xy_hash"],
        "rel_xy_hash_equal": sa["rel_xy_hash"] == sb["rel_xy_hash"],
        "path_len_diff_yds": sb["path_len_raw_yds"] - sa["path_len_raw_yds"],
        "duration_diff_s": sb["duration_s"] - sa["duration_s"],
        "peak_speed_diff": (sb["peak_speed"] - sa["peak_speed"]) if None not in (sa["peak_speed"], sb["peak_speed"])
        else None,
    }
    al = align(a, b, cfg)
    metrics = dict.fromkeys(("lag_s", "overlap_s", "speed_rmse", "vendor_speed_rmse", "accel_rmse", "speed_corr",
                             "max_abs_speed_diff", "raw_pos_rmse_yds", "raw_pos_rmse_rigid_yds",
                             "smooth_pos_rmse_rigid_yds"))
    metrics.update(comparable=al is not None, anchor=al.anchor if al else None, n_grid=int(al.grid.size) if al else 0)
    if al is not None:
        z = aligned_signals(a, b, al)
        dv = z["speed_b"] - z["speed_a"]
        both = np.isfinite(z["speed_a"]) & np.isfinite(z["speed_b"])
        corr = None
        if both.sum() >= 3 and np.std(z["speed_a"][both]) > 0 and np.std(z["speed_b"][both]) > 0:
            corr = float(np.corrcoef(z["speed_a"][both], z["speed_b"][both])[0, 1])
        pa, pb = np.column_stack([z["x_a"], z["y_a"]]), np.column_stack([z["x_b"], z["y_b"]])
        metrics.update(
            lag_s=al.lag_s, overlap_s=float(al.grid[-1] - al.grid[0]), speed_rmse=_rmse(dv),
            vendor_speed_rmse=_rmse(z["s_b"] - z["s_a"]), accel_rmse=_rmse(z["a_tan_b"] - z["a_tan_a"]),
            speed_corr=corr, max_abs_speed_diff=float(np.nanmax(np.abs(dv))) if np.isfinite(dv).any() else None,
            raw_pos_rmse_yds=_translated_rmse(pa, pb), raw_pos_rmse_rigid_yds=rigid_rmse(pa, pb),
            smooth_pos_rmse_rigid_yds=rigid_rmse(np.column_stack([z["xs_a"], z["ys_a"]]),
                                                 np.column_stack([z["xs_b"], z["ys_b"]])))
    m.update(metrics)
    cls, flags, evidence = classify_pair(m, cfg)
    m.update({f"is_{c}": flags[c] for c in FLAG_CLASSES})
    m.update(pair_class=cls, integrity_group=GROUP_OF[cls], evidence=evidence)
    return m


def _ge(v: float | None, thr: float) -> bool:
    return bool(v is not None and np.isfinite(v) and v >= thr)


def _le(v: float | None, thr: float) -> bool:
    return bool(v is not None and np.isfinite(v) and v <= thr)


def classify_pair(m: dict[str, Any], cfg: DuplicateAuditConfig) -> tuple[str, dict[str, bool], str]:
    """Pure rule set → (most severe class, every criterion, human-readable evidence)."""
    speed = m.get("speed_rmse") if m.get("speed_rmse") is not None else m.get("vendor_speed_rmse")
    flags = {k: bool(v) for k, v in {
        "exact_duplicate": bool(m["raw_xy_hash_equal"]) or _ge(m["shared_frame_frac"], cfg.exact_shared_frac),
        "translated_copy": bool(m["rel_xy_hash_equal"]),
        "reused_sequence": _ge(m["shared_kgram_frac"], cfg.reuse_kgram_frac)
                           or _ge(m["shared_frame_frac"], cfg.reuse_shared_frame_frac),
        "time_overlap": (m["ts_overlap_s"] or 0.0) > 0.0,
        "near_duplicate": bool(m["comparable"]) and _le(m.get("raw_pos_rmse_rigid_yds"), cfg.near_dup_pos_rmse_yds)
                          and _le(speed, cfg.near_dup_speed_rmse),
    }.items()}
    cls = next((c for c in FLAG_CLASSES if flags[c]), None)
    if cls is None:
        cls = ("not_comparable" if not m["comparable"]
               else "consistent" if _le(speed, cfg.consistent_speed_rmse) else "distinct")

    def fmt(v):
        return "na" if v is None else f"{v:.3g}"

    ev = [f"{c}" for c in FLAG_CLASSES if flags[c]]
    ev.append(f"shared_frames={fmt(m['shared_frame_frac'])}, kgrams={fmt(m['shared_kgram_frac'])}, "
              f"ts_overlap={fmt(m['ts_overlap_s'])}s, speed_rmse={fmt(speed)}, "
              f"pos_rigid={fmt(m.get('raw_pos_rmse_rigid_yds'))}")
    return cls, flags, "; ".join(ev)


# ============================================================================== group / partition audit
def _clusters(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)     # root = earliest attempt (deterministic)
    return [find(i) for i in range(n)]


def audit_group(tracks: list[Track], cfg: DuplicateAuditConfig) -> tuple[list[dict], list[dict]]:
    """All pairs of ONE athlete × drill (bounded by ``max_attempts_per_group``)."""
    tracks = sorted(tracks, key=lambda t: (int(t.ts_ms[0]), t.event_id, t.attempt))
    n, k = len(tracks), min(len(tracks), cfg.max_attempts_per_group)
    sigs = [signature(t, cfg) for t in tracks]
    grams = [kgrams(t, cfg) for t in tracks[:k]]
    pairs, by_attempt = [], {i: [] for i in range(n)}
    for i, j in combinations(range(k), 2):
        a, b = tracks[i], tracks[j]
        m = compare_pair(a, b, grams[i], grams[j], sigs[i], sigs[j], cfg)
        m.update(pair_id=f"{a.nfl_id}|{a.drill}|{a.label}|{b.label}", nfl_id=a.nfl_id, drill=a.drill,
                 draft_year=a.draft_year, event_id_a=a.event_id, attempt_a=a.attempt, event_id_b=b.event_id,
                 attempt_b=b.attempt, attempt_order_a=i + 1, attempt_order_b=j + 1, n_frames_a=int(a.ts_ms.size),
                 n_frames_b=int(b.ts_ms.size))
        pairs.append(m)
        by_attempt[i].append((j, m))
        by_attempt[j].append((i, m))

    dedup = _clusters(n, [(p["attempt_order_a"] - 1, p["attempt_order_b"] - 1) for p in pairs
                          if p["integrity_group"] == "integrity_failure"])
    strict = _clusters(n, [(p["attempt_order_a"] - 1, p["attempt_order_b"] - 1) for p in pairs
                           if p["integrity_group"] in ("integrity_failure", "suspicious")])

    def cluster_id(roots: list[int], i: int) -> str | None:
        r = roots[i]
        return f"{tracks[r].nfl_id}|{tracks[r].drill}|{tracks[r].label}" if roots.count(r) > 1 else None

    rows = []
    for i, tr in enumerate(tracks):
        mine = by_attempt[i]
        classes = [m["pair_class"] for _, m in mine]
        rows.append({
            "event_id": tr.event_id, "nfl_id": tr.nfl_id, "attempt": tr.attempt, "drill": tr.drill,
            "draft_year": tr.draft_year, "attempt_order": i + 1, "group_n_attempts": n, "group_truncated": n > k,
            **sigs[i], "n_moving_kgrams": len(grams[i]) if i < k else None,
            "n_pairs": len(mine), "n_pairs_comparable": sum(bool(m["comparable"]) for _, m in mine),
            "worst_pair_class": max(classes, key=SEVERITY.__getitem__) if classes else "unpaired",
            "partners": "; ".join(f"{tracks[j].label}={m['pair_class']}" for j, m in mine
                                  if m["integrity_group"] in ("integrity_failure", "suspicious")),
            **{f"flag_{c}": any(m[f"is_{c}"] for _, m in mine) for c in FLAG_CLASSES},
            "dedup_cluster": cluster_id(dedup, i), "strict_cluster": cluster_id(strict, i),
            "is_dedup_representative": dedup[i] == i, "_strict_rep": strict[i] == i})
    return rows, pairs


_ATT_RAW: dict[str, pl.DataType] = {
    "event_id": S, "nfl_id": I, "attempt": I, "drill": S, "draft_year": I, "attempt_order": I,
    "group_n_attempts": I, "group_truncated": B, "n_frames": I, "ts_start_ms": I, "ts_end_ms": I, "duration_s": F,
    "path_len_raw_yds": F, "path_len_smooth_yds": F, "peak_speed": F, "raw_xy_hash": S, "rel_xy_hash": S,
    "n_moving_kgrams": I, "n_pairs": I, "n_pairs_comparable": I, "worst_pair_class": S, "partners": S,
    **{f"flag_{c}": B for c in FLAG_CLASSES}, "dedup_cluster": S, "strict_cluster": S,
    "is_dedup_representative": B, "_strict_rep": B}


def audit_frames(frames: pl.DataFrame, cfg: DuplicateAuditConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Pure partition-level audit. ``frames`` must contain ``FRAME_INPUT`` (``valid_kin`` optional) and
    every athlete × drill group completely (partition by cohort, never split a group)."""
    missing = [c for c in FRAME_INPUT if c not in frames.columns and c != "valid_kin"]
    if missing:
        raise KeyError(f"frames lack required columns {missing}")
    att_rows, pair_rows = [], []
    for grp in cast_keys(frames).sort("nfl_id", "drill", "ts_ms").partition_by("nfl_id", "drill", maintain_order=True):
        tracks = [make_track(g, cfg) for g in grp.partition_by(KEYS, maintain_order=True)]
        a, p = audit_group(tracks, cfg)
        att_rows.extend(a)
        pair_rows.extend(p)
    return (pl.DataFrame(att_rows, schema=_ATT_RAW) if att_rows else pl.DataFrame(schema=_ATT_RAW),
            pl.DataFrame(pair_rows, schema=_PAIR_COLS) if pair_rows else pl.DataFrame(schema=_PAIR_COLS))


def _bool_flag(df: pl.DataFrame, by: list[str], expr: pl.Expr, name: str) -> pl.DataFrame:
    return df.group_by(by).agg((expr > 1).alias(name))


def finalize(att: pl.DataFrame, pairs: pl.DataFrame, cfg: DuplicateAuditConfig, version: str,
             tiers: pl.DataFrame | None = None, m2_keys: pl.DataFrame | None = None
             ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Cross-group evidence (hash joins only) → (attempt audit, pairs, integrity flags)."""
    eligible = att.filter((pl.col("n_frames") >= cfg.cross_athlete_min_frames)
                          & (pl.col("path_len_raw_yds") >= cfg.cross_athlete_min_path_yds))
    xa = (eligible.group_by("rel_xy_hash").agg(pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes_same_rel_hash"))
          .join(eligible.select(*KEYS, "rel_xy_hash"), on="rel_xy_hash").drop("rel_xy_hash"))
    conflicts = [
        _bool_flag(att, ["nfl_id", "drill", "ts_start_ms"], pl.struct("event_id", "attempt").n_unique(),
                   "conflicting_ids_same_start"),
        _bool_flag(att, ["event_id", "attempt"], pl.col("nfl_id").n_unique(), "shared_event_attempt_id"),
        _bool_flag(att, ["nfl_id", "drill", "attempt"], pl.col("event_id").n_unique(), "repeated_attempt_number"),
        _bool_flag(att, ["nfl_id"], pl.col("draft_year").n_unique(), "multi_cohort_athlete"),
    ]
    out = att.join(xa, on=KEYS, how="left").with_columns(
        (pl.col("n_athletes_same_rel_hash").fill_null(1) > 1).alias("cross_athlete_copy"))
    for c, on in zip(conflicts, (["nfl_id", "drill", "ts_start_ms"], ["event_id", "attempt"],
                                 ["nfl_id", "drill", "attempt"], ["nfl_id"])):
        out = out.join(c, on=on, how="left")
    tiers = tiers if tiers is not None else pl.DataFrame(schema={**{k: ATTEMPT_AUDIT.columns[k] for k in KEYS},
                                                                 "quality_tier": S})
    out = out.join(cast_keys(tiers).select(*KEYS, pl.col("quality_tier").cast(pl.String)), on=KEYS, how="left")
    if m2_keys is not None:
        out = out.join(cast_keys(m2_keys).select(KEYS).unique().with_columns(pl.lit(True).alias("in_m2")), on=KEYS,
                       how="left").with_columns(pl.col("in_m2").fill_null(False))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Boolean).alias("in_m2"))

    worst = (pl.when(pl.col("cross_athlete_copy")
                     & (pl.col("worst_pair_class").replace_strict(SEVERITY, return_dtype=pl.Int64)
                        < SEVERITY["cross_athlete_copy"]))
             .then(pl.lit("cross_athlete_copy")).otherwise(pl.col("worst_pair_class")))
    flags = out.with_columns(worst.alias("integrity_class")).with_columns(
        pl.col("integrity_class").replace_strict(GROUP_OF, return_dtype=pl.String).alias("integrity_group"),
        pl.col("cross_athlete_copy").alias("flag_cross_athlete_copy"),
        pl.col("conflicting_ids_same_start").alias("flag_conflicting_ids"),
        pl.col("group_truncated").alias("flag_group_truncated"),
        pl.col("shared_event_attempt_id").alias("info_shared_event_attempt_id"),
        pl.col("repeated_attempt_number").alias("info_repeated_attempt_number"),
        pl.col("multi_cohort_athlete").alias("info_multi_cohort_athlete"),
        (pl.col("dedup_cluster").is_not_null() & ~pl.col("is_dedup_representative")).alias("exclude_dedup"),
        ((pl.col("strict_cluster").is_not_null() & ~pl.col("_strict_rep")) | pl.col("cross_athlete_copy"))
        .alias("exclude_strict"),
        pl.lit(version).alias("m21_version"))

    pairs = pairs.with_columns(
        (pl.col("speed_rmse").rank("average").over("drill") / pl.col("speed_rmse").count().over("drill"))
        .cast(pl.Float64).alias("speed_rmse_pct_in_drill"))
    return (out.select(list(ATTEMPT_AUDIT.columns)), pairs.select(list(PAIRS.columns)),
            flags.select(list(FLAGS.columns)))


# ============================================================================== summaries / sensitivity
def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z ** 2 / n
        mid = (p + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return np.where(n > 0, mid - half, np.nan), np.where(n > 0, mid + half, np.nan)


def flag_summary(flags: pl.DataFrame) -> pl.DataFrame:
    bool_cols = [c for c in FLAGS.columns if c.startswith(("flag_", "info_")) or c in ("exclude_dedup", "exclude_strict")]
    f = flags.with_columns(
        *[(pl.col("integrity_group") == g).alias(f"group:{g}") for g in GROUPS],
        pl.col("draft_year").cast(pl.String).alias("cohort"))
    f = pl.concat([f, f.with_columns(pl.lit("all").alias("cohort"))])
    long = f.unpivot(index=["drill", "cohort", "nfl_id"], on=[*bool_cols, *[f"group:{g}" for g in GROUPS]],
                     variable_name="flag", value_name="v")
    out = (long.group_by("drill", "cohort", "flag").agg(
               pl.len().cast(pl.Int64).alias("n_attempts"), pl.col("v").fill_null(False).sum().cast(pl.Int64).alias("n_flagged"),
               pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
               pl.col("nfl_id").filter(pl.col("v").fill_null(False)).n_unique().cast(pl.Int64).alias("n_athletes_flagged"))
           .with_columns((pl.col("n_flagged") / pl.col("n_attempts")).cast(pl.Float64).alias("share"))
           .sort("drill", "cohort", "flag"))
    lo, hi = wilson(out["n_flagged"].to_numpy(), out["n_attempts"].to_numpy())
    return out.with_columns(pl.Series("ci_low", lo, dtype=pl.Float64), pl.Series("ci_high", hi, dtype=pl.Float64))


def feature_values(att: pl.DataFrame, sprint: pl.DataFrame | None) -> pl.DataFrame:
    """Attempt-level M2 feature values in long form (event_id, nfl_id, attempt, drill, feature, value)."""
    idx = [*KEYS, "drill"]
    feats = [c for c, t in att.schema.items() if c not in ID_COLS and c not in NOT_FEATURES and t.is_numeric()]
    parts = [att.select(*idx, *[pl.col(c).cast(pl.Float64) for c in feats])
             .unpivot(index=idx, on=feats, variable_name="feature", value_name="value")]
    if sprint is not None and sprint.height:
        sf = [c for c in SPRINT_FEATURES if c in sprint.columns and c not in feats]
        if sf:
            parts.append(sprint.select(*idx, *[pl.col(c).cast(pl.Float64) for c in sf])
                         .unpivot(index=idx, on=sf, variable_name="feature", value_name="value"))
    return (pl.concat([cast_keys(p) for p in parts], how="vertical")
            .filter(pl.col("value").is_not_null() & pl.col("value").is_finite()))


def _athlete_stats(v: pl.DataFrame) -> pl.DataFrame:
    return v.group_by("nfl_id", "drill", "feature").agg(
        pl.len().cast(pl.Int64).alias("n"), pl.col("value").std(ddof=1).alias("sd"),
        pl.col("value").var(ddof=1).alias("var"), pl.col("value").median().alias("median"))


def _is_zero(cfg: DuplicateAuditConfig) -> pl.Expr:
    return pl.col("sd") <= pl.max_horizontal(pl.lit(cfg.zero_sd_abs), cfg.zero_sd_rel * pl.col("median").abs())


def duplicate_sensitivity(values: pl.DataFrame, flags: pl.DataFrame, cfg: DuplicateAuditConfig) -> pl.DataFrame:
    v = values.join(flags.select(*KEYS, "exclude_dedup", "exclude_strict"), on=KEYS, how="inner")
    scen = {"all": pl.lit(True), "dedup": ~pl.col("exclude_dedup"), "strict": ~pl.col("exclude_strict")}
    parts = []
    for name, keep in scen.items():
        st = _athlete_stats(v.filter(keep))
        rep = st.filter(pl.col("n") >= 2)
        agg = (st.group_by("drill", "feature").agg(pl.col("n").sum().cast(pl.Int64).alias("n_attempts"))
               .join(rep.group_by("drill", "feature").agg(
                         pl.len().cast(pl.Int64).alias("n_athletes_repeated"),
                         pl.col("sd").median().alias("median_within_sd"),
                         (((pl.col("n") - 1) * pl.col("var")).sum() / (pl.col("n") - 1).sum()).sqrt()
                         .alias("pooled_within_sd"),
                         _is_zero(cfg).mean().cast(pl.Float64).alias("share_zero_sd")),
                     on=["drill", "feature"], how="left")
               .with_columns(pl.lit(name).alias("scenario"), pl.col("n_athletes_repeated").fill_null(0)))
        parts.append(agg)
    out = pl.concat(parts, how="vertical")
    base = out.filter(pl.col("scenario") == "all").select("drill", "feature", pl.col("pooled_within_sd").alias("_b"))
    return (out.join(base, on=["drill", "feature"], how="left")
            .with_columns(pl.when(pl.col("_b") > 0).then(pl.col("pooled_within_sd") / pl.col("_b") - 1)
                          .cast(pl.Float64).alias("pooled_within_sd_rel_change"))
            .select(list(SENSITIVITY.columns)).sort("drill", "feature", "scenario"))


def zero_variance_athletes(values: pl.DataFrame, flags: pl.DataFrame, cfg: DuplicateAuditConfig) -> pl.DataFrame:
    """Athletes with repeated attempts but zero SD, with the most likely explanation."""
    feat = values.group_by("drill", "feature").agg(
        (pl.col("value") == pl.col("value").round(0)).all().alias("integer_valued"),
        (pl.col("value").n_unique() / pl.len() < cfg.quantized_unique_frac).alias("quantized"))
    sev = pl.col("integrity_class").replace_strict(SEVERITY, return_dtype=pl.Int64)
    worst = flags.group_by("nfl_id", "drill").agg(
        pl.col("draft_year").first(),
        pl.col("integrity_class").sort_by(sev, descending=True).first().alias("worst_integrity_class"))
    z = (_athlete_stats(values.join(flags.select(KEYS), on=KEYS, how="semi"))
         .filter((pl.col("n") >= 2) & _is_zero(cfg))
         .join(feat, on=["drill", "feature"], how="left").join(worst, on=["nfl_id", "drill"], how="left")
         .with_columns(pl.col("worst_integrity_class").replace_strict(GROUP_OF, return_dtype=pl.String)
                       .alias("worst_integrity_group")))
    g = pl.col("worst_integrity_group")
    expl = (pl.when(g == "integrity_failure").then(pl.lit("duplicated / overlapping source data"))
            .when(g == "suspicious").then(pl.lit("near-duplicate trajectories"))
            .when(pl.col("integer_valued")).then(pl.lit("integer-valued feature — ties expected"))
            .when(pl.col("quantized")).then(pl.lit("coarse value grid (e.g. fit grid) — ties expected"))
            .otherwise(pl.lit("UNEXPLAINED — inspect attempts and feature definition")))
    return (z.with_columns(expl.alias("explanation"), pl.col("n").alias("n_attempts"))
            .select(list(ZERO_VAR.columns)).sort("drill", "feature", "nfl_id"))


# ============================================================================== examples
def select_examples(pairs: pl.DataFrame, cfg: DuplicateAuditConfig) -> pl.DataFrame:
    sev = pl.col("pair_class").replace_strict(SEVERITY, return_dtype=pl.Int64)
    tie = ["nfl_id", "attempt_order_a", "attempt_order_b"]
    raw = pairs.filter(pl.col("integrity_group") == "integrity_failure")
    near = pairs.filter(pl.col("pair_class") == "near_duplicate")
    cons = pairs.filter(pl.col("pair_class") == "consistent")
    if cons.filter(pl.col("drill") == FORTY).height:
        cons = cons.filter(pl.col("drill") == FORTY)
    med = cons["speed_rmse"].median() if cons.height else None
    picks = [
        ("raw_duplicate", raw.sort([sev, pl.col("drill") != FORTY, *tie], descending=[True, False, False, False, False]),
         raw.height, "most severe integrity-failure class; forty first; then lowest nfl_id / attempt order"),
        ("near_duplicate", near.sort(["speed_rmse", *tie]), near.height,
         "near-duplicate pair with the smallest aligned speed RMSE; ties by nfl_id / attempt order"),
        ("consistent", cons.sort([(pl.col("speed_rmse") - (med if med is not None else 0.0)).abs(), *tie]),
         cons.height, f"forty 'consistent' pair (speed RMSE ≤ {cfg.consistent_speed_rmse} yd/s) closest to the "
                      f"class median speed RMSE; ties by nfl_id / attempt order"),
    ]
    rows = []
    for name, df, n, rule in picks:
        if df.height:
            rows.append(df.head(1).select(
                pl.lit(name).alias("example"), "pair_id", "nfl_id", "drill", "event_id_a", "attempt_a", "event_id_b",
                "attempt_b", "attempt_order_a", "attempt_order_b", "pair_class", "speed_rmse",
                "raw_pos_rmse_rigid_yds", pl.lit(n, dtype=pl.Int64).alias("n_class_pairs"), pl.lit(rule).alias("rule")))
    return pl.concat(rows) if rows else EXAMPLES.empty()


def example_signals(frames: pl.DataFrame, examples: pl.DataFrame, cfg: DuplicateAuditConfig) -> pl.DataFrame:
    fr = cast_keys(frames)
    out = []
    for r in examples.iter_rows(named=True):
        def track(side: str) -> Track | None:
            g = fr.filter((pl.col("event_id") == r[f"event_id_{side}"]) & (pl.col("nfl_id") == r["nfl_id"])
                          & (pl.col("attempt") == r[f"attempt_{side}"]))
            return make_track(g, cfg) if g.height else None

        a, b = track("a"), track("b")
        al = align(a, b, cfg) if a is not None and b is not None else None
        if al is None:
            log.warning("[m21:A] example %s (%s) could not be aligned — omitted", r["example"], r["pair_id"])
            continue
        z = aligned_signals(a, b, al)

        def rel(v: np.ndarray) -> np.ndarray:
            f = v[np.isfinite(v)]
            return v - f[0] if f.size else v

        n = z["t"].size
        out.append(pl.DataFrame({
            "example": [r["example"]] * n, "pair_id": [r["pair_id"]] * n, "t_aligned_s": np.round(z["t"], 6),
            "speed_a": z["speed_a"], "speed_b": z["speed_b"], "vendor_speed_a": z["s_a"], "vendor_speed_b": z["s_b"],
            "x_a": rel(z["x_a"]), "y_a": rel(z["y_a"]), "x_b": rel(z["x_b"]), "y_b": rel(z["y_b"])},
            schema=SIGNALS.columns).with_columns(*[pl.col(c).fill_nan(None) for c in list(SIGNALS.columns)[3:]]))
    return pl.concat(out) if out else SIGNALS.empty()


# ============================================================================== figures (from persisted tables only)
EXAMPLE_PANELS = (("raw_duplicate", "Raw-data duplicate / integrity failure"),
                  ("near_duplicate", "Suspicious near-duplicate"),
                  ("consistent", "Genuinely consistent attempts"))


def plot_duplicate_examples(signals: pl.DataFrame, examples: pl.DataFrame, pairs: pl.DataFrame,
                            cfg: DuplicateAuditConfig):
    fig, axes = P.figure(2, 3, size="wide")
    counts = {e: 0 for e, _ in EXAMPLE_PANELS}
    for r in examples.iter_rows(named=True):
        counts[r["example"]] = r["n_class_pairs"]
    for col, (ex, label) in enumerate(EXAMPLE_PANELS):
        top, bot = axes[0, col], axes[1, col]
        r = examples.filter(pl.col("example") == ex)
        s = signals.filter(pl.col("example") == ex).sort("t_aligned_s")
        if r.is_empty() or s.is_empty():
            P.empty_panel(top, f"{label}\nno pair in this class (n = 0)")
            P.empty_panel(bot, "—")
            continue
        r = r.row(0, named=True)
        t = s["t_aligned_s"].to_numpy()
        color = P.CLASS_COLORS.get(r["pair_class"], CATEGORICAL[0])
        top.plot(t, s["speed_a"].to_numpy(), color=color, lw=2.2, label=f"attempt {r['attempt_order_a']} (observed)")
        top.plot(t, s["speed_b"].to_numpy(), color=INK["primary"], lw=1.4, ls="--",
                 label=f"attempt {r['attempt_order_b']} (observed)")
        top.set_title(f"{label}\n{r['drill'].replace('_', ' ').lower()} · id {r['nfl_id']} · {r['pair_class']}",
                      fontsize=11)
        top.legend(fontsize=8, loc="lower right", frameon=False)
        d = (s["speed_b"] - s["speed_a"]).to_numpy()
        dv = (s["vendor_speed_b"] - s["vendor_speed_a"]).to_numpy()
        thr = cfg.near_dup_speed_rmse
        bot.axhspan(-thr, thr, color=P.CLASS_COLORS["near_duplicate"], alpha=0.18, lw=0,
                    label=f"± {thr} yd/s near-duplicate scale")
        bot.axhline(0, color=INK["axis"], lw=1)
        bot.plot(t, d, color=color, lw=1.8, label="smoothed speed b − a")
        bot.plot(t, dv, color=INK["muted"], lw=1.0, alpha=0.8, label="vendor speed b − a (raw)")
        lim = max(float(np.nanmax(np.abs(np.r_[d, dv, 0.0]))) * 1.15, 2.5 * thr)
        bot.set_ylim(-lim, lim)
        bot.set_xlabel("time aligned to movement onset (s)")
        bot.legend(fontsize=7.5, loc="upper right", frameon=False)
        rm, pr = r["speed_rmse"], r["raw_pos_rmse_rigid_yds"]
        bot.text(0.02, 0.04, f"speed RMSE {('%.3f' % rm) if rm is not None else 'na'} yd/s · rigid position RMSE "
                             f"{('%.3f' % pr) if pr is not None else 'na'} yd",
                 transform=bot.transAxes, fontsize=8.5, color=INK["secondary"])
    axes[0, 0].set_ylabel("speed (yd/s)")
    axes[1, 0].set_ylabel("difference (yd/s)")
    n_pairs = pairs.height
    P.finish(fig, (f"Within-athlete attempt pairs (n = {n_pairs:,}; integrity-failure {counts['raw_duplicate']:,}, "
                   f"near-duplicate {counts['near_duplicate']:,}, consistent {counts['consistent']:,} in the panel "
                   f"subset). Selection: duplicate_examples.csv (rule column). Attempt b is lag-aligned to a after "
                   f"onset anchoring; both curves are observations (no model). Source: combine_frame_kinematics."),
             title="Are repeated attempts independent? Exact duplicate vs near-duplicate vs genuinely consistent")
    return fig


def similarity_histogram(pairs: pl.DataFrame, cfg: DuplicateAuditConfig, floor: float = 1e-3) -> pl.DataFrame:
    """Binned log10(speed RMSE) counts per drill panel × integrity group (plot data)."""
    edges = np.linspace(-3.0, 1.0, 41)
    p = pairs.filter(pl.col("comparable") & pl.col("speed_rmse").is_not_null()).with_columns(
        pl.when(pl.col("drill").is_in(list(KEY_DRILLS))).then(pl.col("drill")).otherwise(pl.lit("OTHER_DRILLS"))
        .alias("panel"))
    rows = []
    for (panel, grp), d in p.group_by("panel", "integrity_group", maintain_order=True):
        v = np.log10(np.clip(d["speed_rmse"].to_numpy(), floor, 10.0 - 1e-9))
        c, _ = np.histogram(v, bins=edges)
        rows += [{"panel": panel, "integrity_group": grp, "bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                  "count": int(c[i])} for i in range(len(c))]
    return pl.DataFrame(rows, schema={"panel": S, "integrity_group": S, "bin_lo": F, "bin_hi": F, "count": I})


def plot_similarity(hist: pl.DataFrame, pairs: pl.DataFrame, cfg: DuplicateAuditConfig):
    panels = [d for d in (*KEY_DRILLS, "OTHER_DRILLS") if d in set(hist["panel"].to_list())] or ["(none)"]
    fig, axes = P.figure(1, len(panels), size="wide")
    order = ("natural", "suspicious", "integrity_failure")
    colors = {"natural": P.CLASS_COLORS["consistent"], "suspicious": P.CLASS_COLORS["near_duplicate"],
              "integrity_failure": P.CLASS_COLORS["exact_duplicate"]}
    for ax, panel in zip(axes[0], panels):
        h = hist.filter(pl.col("panel") == panel)
        if h.is_empty():
            P.empty_panel(ax, "no comparable pairs")
            continue
        bottom = None
        for g in order:
            hg = h.filter(pl.col("integrity_group") == g).sort("bin_lo")
            if hg.is_empty():
                continue
            c = hg["count"].to_numpy()
            ax.bar(hg["bin_lo"].to_numpy(), c, width=(hg["bin_hi"] - hg["bin_lo"]).to_numpy(), align="edge",
                   bottom=bottom, color=colors[g], label=f"{g} (n = {int(c.sum()):,})", lw=0)
            bottom = c if bottom is None else bottom + c
        for thr, lab in ((cfg.near_dup_speed_rmse, "near-dup"), (cfg.consistent_speed_rmse, "consistent")):
            ax.axvline(np.log10(thr), color=INK["secondary"], ls=":", lw=1.2)
            ax.text(np.log10(thr), ax.get_ylim()[1] * 0.97, f" {lab}", fontsize=8, va="top", color=INK["secondary"])
        ax.set_title(f"{panel.replace('_', ' ').lower()} (n = {int(h['count'].sum()):,} pairs)", fontsize=12)
        ax.set_xlabel("log10 aligned speed RMSE (yd/s)")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    axes[0, 0].set_ylabel("attempt pairs")
    n_nc = pairs.filter(~pl.col("comparable")).height
    P.finish(fig, (f"Within-athlete × drill attempt pairs; not comparable (no ≥ {cfg.min_overlap_frac:.0%} aligned "
                   f"overlap) excluded: n = {n_nc:,}. Exact duplicates (RMSE 0) are drawn at the 1e-3 floor. "
                   f"Thresholds are PROPOSED (m21_thresholds.csv); the class also requires position agreement. "
                   f"Source: attempt_similarity_pairs."),
             title="How similar are an athlete's repeated attempts?")
    return fig


def plot_flag_prevalence(summary: pl.DataFrame):
    flags = (("group:integrity_failure", "integrity failure"), ("group:suspicious", "suspicious near-duplicate"))
    d = summary.filter(pl.col("drill").is_in(list(KEY_DRILLS)) & pl.col("flag").is_in([f for f, _ in flags]))
    drills = [x for x in KEY_DRILLS if x in set(d["drill"].to_list())] or ["(none)"]
    fig, axes = P.figure(1, len(drills), size="wide", sharey=True)
    for ax, drill in zip(axes[0], drills):
        dd = d.filter(pl.col("drill") == drill)
        if dd.is_empty():
            P.empty_panel(ax, "no attempts")
            continue
        cohorts = sorted(set(dd["cohort"].to_list()), key=lambda c: (c == "all", c))
        x = np.arange(len(cohorts))
        for k, (flag, lab) in enumerate(flags):
            f = dd.filter(pl.col("flag") == flag)
            look = {r["cohort"]: r for r in f.iter_rows(named=True)}
            share = np.array([look[c]["share"] if c in look else np.nan for c in cohorts]) * 100
            lo = np.array([look[c]["ci_low"] if c in look else np.nan for c in cohorts]) * 100
            hi = np.array([look[c]["ci_high"] if c in look else np.nan for c in cohorts]) * 100
            ax.bar(x + (k - 0.5) * 0.38, share, width=0.36,
                   color=P.CLASS_COLORS["exact_duplicate" if k == 0 else "near_duplicate"],
                   label=lab, yerr=np.vstack([share - lo, hi - share]), capsize=3, error_kw={"lw": 1})
        ns = {r["cohort"]: r["n_attempts"] for r in dd.filter(pl.col("flag") == flags[0][0]).iter_rows(named=True)}
        ax.set_xticks(x, [f"{c}\nn={ns.get(c, 0):,}" for c in cohorts])
        ax.set_title(drill.replace("_", " ").lower(), fontsize=12)
        ax.legend(fontsize=8, frameon=False)
    axes[0, 0].set_ylabel("% of attempts (Wilson 95 % CI)")
    P.finish(fig, "Share of attempts in each integrity group by cohort. n = attempts (all quality tiers). "
                  "Source: duplicate_flag_summary.", title="Attempt-integrity flags by drill and cohort")
    return fig


# ============================================================================== orchestration
class DuplicateAudit:
    UPSTREAM = (COMBINE_FRAMES.name, COMBINE_TRAJ.name, "combine_attempt_features", "combine_sprint_profiles")

    def __init__(self, cfg: Phase01Config, m21: M21Config | None = None,
                 builder: CombineMovementBuilder | None = None, store: M21Store | None = None):
        self.cfg = cfg
        self.m21 = m21 or M21Config()
        self.dcfg = self.m21.duplicates
        self.b = builder or CombineMovementBuilder(cfg)
        self.src = self.b.store                 # M1/M2 store (read only)
        self.store = store or M21Store(cfg, sampled=self.m21.sample_athletes is not None)
        if self.store.sampled != (self.m21.sample_athletes is not None):
            raise ValueError("sampled runs must use M21Store(sampled=True) and full runs sampled=False")

    def fingerprint(self) -> str:
        return self.store.fingerprint(WORKSTREAM, self.m21.section("duplicates"), upstream_state(self.src, self.UPSTREAM))

    # ------------------------------------------------------------------ inputs
    def athlete_sample(self) -> list[int] | None:
        if self.m21.sample_athletes is None:
            return None
        ids = (self.src.load(COMBINE_TRAJ.name).select(pl.col("nfl_id").cast(pl.Int64)).unique().collect()
               .get_column("nfl_id").sort().to_numpy())
        k = min(self.m21.sample_athletes, ids.size)
        return sorted(np.random.default_rng(self.m21.seed).choice(ids, size=k, replace=False).tolist())

    def _frames_lf(self, nfl_ids: list[int] | None) -> pl.LazyFrame:
        lf = self.src.load(COMBINE_FRAMES.name)
        if nfl_ids is not None:
            lf = lf.filter(pl.col("nfl_id").is_in(nfl_ids))
        return lf

    def frames(self, draft_year: int, nfl_ids: list[int] | None) -> pl.DataFrame:
        return (self._frames_lf(nfl_ids).filter(pl.col("draft_year") == draft_year)
                .select("event_id", "nfl_id", "attempt", "drill", "draft_year",
                        pl.col("ts").dt.epoch("ms").alias("ts_ms"), "x", "y", "s", "x_s", "y_s", "speed", "a_tan",
                        "valid_kin")
                .collect())

    # ------------------------------------------------------------------ compute
    def compute(self) -> dict[str, pl.DataFrame]:
        ids = self.athlete_sample()
        years = sorted(self._frames_lf(None).select(pl.col("draft_year").unique()).collect().get_column("draft_year")
                       .to_list())
        atts, pairs = [], []
        for y in years:   # cohort partitions keep every athlete × drill group whole
            fr = self.frames(y, ids)
            a, p = audit_frames(fr, self.dcfg)
            log.info("[m21:A] cohort %s: %s frames, %s attempts, %s pairs", y, f"{fr.height:,}", f"{a.height:,}",
                     f"{p.height:,}")
            atts.append(a)
            pairs.append(p)
            del fr
        tiers = self.src.load(COMBINE_TRAJ.name).select(*KEYS, "quality_tier").collect()
        att_m2 = self.b.load("attempts").collect()
        sprint = self.b.load("sprint").collect()
        audit, pr, flags = finalize(pl.concat(atts), pl.concat(pairs), self.dcfg, self.m21.version, tiers,
                                    att_m2.select(KEYS))
        values = feature_values(att_m2, sprint)
        examples = select_examples(pr, self.dcfg)
        ex_ids = examples["nfl_id"].unique().to_list()
        signals = (example_signals(self._frames_lf(None).filter(pl.col("nfl_id").is_in(ex_ids))
                                   .select("event_id", "nfl_id", "attempt", "drill", "draft_year",
                                           pl.col("ts").dt.epoch("ms").alias("ts_ms"), "x", "y", "s", "x_s", "y_s",
                                           "speed", "a_tan", "valid_kin").collect(), examples, self.dcfg)
                   if ex_ids else SIGNALS.empty())
        return {"attempt_duplicate_audit": audit, "attempt_similarity_pairs": pr, "attempt_integrity_flags": flags,
                "duplicate_flag_summary": flag_summary(flags),
                "duplicate_sensitivity": duplicate_sensitivity(values, flags, self.dcfg),
                "zero_variance_athletes": zero_variance_athletes(values, flags, self.dcfg),
                "duplicate_examples": examples, "duplicate_example_signals": signals}

    def run(self, force: bool = False, figures: bool = True) -> dict[str, pl.DataFrame]:
        fp = self.fingerprint()
        self.store.write(threshold_table(self.m21), THRESHOLDS)
        current, reason = self.store.is_current(WORKSTREAM, fp, CONTRACTS)
        if current and not force:
            log.info("[m21:A] reuse (%s)", reason)
            out = self.load()
        else:
            log.info("[m21:A] compute (%s%s)", "forced; " if force else "", reason)
            res = self.compute()
            out = {c.name: self.store.write(res[c.name], c) for c in CONTRACTS}
            self.store.record_run(WORKSTREAM, fp, CONTRACTS, {"sample_athletes": self.m21.sample_athletes})
        if figures:
            self.figures(out)
        return out

    def load(self) -> dict[str, pl.DataFrame]:
        return {c.name: self.store.read(c) for c in CONTRACTS}

    def figures(self, out: dict[str, pl.DataFrame]) -> None:
        """Redraw every Workstream A figure from persisted tables (no tracking reads)."""
        pairs, d = out["attempt_similarity_pairs"], self.dcfg
        hist = similarity_histogram(pairs, d)
        steps = [
            (lambda: plot_duplicate_examples(out["duplicate_example_signals"], out["duplicate_examples"], pairs, d),
             FigureSpec(FIGURES[0], "Speed overlays and differences for one exact/raw duplicate, one near-duplicate "
                                    "and one genuinely consistent attempt pair.",
                        ("duplicate_example_signals", "duplicate_examples", "attempt_similarity_pairs"),
                        "within-athlete × drill pairs; all quality tiers", "see duplicate_examples.rule",
                        ("speed", "s", "speed_rmse", "raw_pos_rmse_rigid_yds"), WORKSTREAM),
             {"signals": out["duplicate_example_signals"], "examples": out["duplicate_examples"]}),
            (lambda: plot_similarity(hist, pairs, d),
             FigureSpec(FIGURES[1], "Distribution of aligned speed RMSE between an athlete's attempts, stacked by "
                                    "integrity group, per drill.", ("attempt_similarity_pairs",),
                        "comparable pairs only", "all pairs", ("speed_rmse",), WORKSTREAM),
             {"histogram": hist}),
            (lambda: plot_flag_prevalence(out["duplicate_flag_summary"]),
             FigureSpec(FIGURES[2], "Share of attempts flagged as integrity failure / suspicious per drill × cohort "
                                    "with Wilson 95 % CIs.", ("duplicate_flag_summary",), "key drills",
                        "all attempts", ("integrity_group",), WORKSTREAM),
             {"summary": out["duplicate_flag_summary"].filter(pl.col("flag").str.starts_with("group:"))}),
        ]
        for draw, spec, data in steps:
            try:
                fig = draw()
                self.store.save_figure(fig, spec, data, n=pairs.height)
            except Exception:  # one failed figure must not lose the tables; logged with traceback, re-checkable
                log.exception("[m21:A] figure %s failed", spec.name)


def main(argv: list[str] | None = None) -> int:
    from ...data.config import configure_runtime

    ap = argparse.ArgumentParser(description="M2.1 Workstream A — duplicate / near-duplicate attempt audit")
    ap.add_argument("--force", action="store_true", help="recompute even if current")
    ap.add_argument("--sample-athletes", type=int, default=None, help="deterministic athlete sample (bounded run)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from persisted tables")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    audit = DuplicateAudit(cfg, M21Config(sample_athletes=a.sample_athletes))
    if a.figures_only:
        audit.figures(audit.load())
        return 0
    out = audit.run(force=a.force, figures=not a.no_figures)
    with pl.Config(tbl_rows=40, tbl_cols=12):
        print(out["duplicate_flag_summary"].filter((pl.col("cohort") == "all") & pl.col("flag").str.starts_with("group:")))
        print(out["zero_variance_athletes"].group_by("drill", "explanation").len().sort("drill", "len"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
