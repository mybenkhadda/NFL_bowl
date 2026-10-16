"""Synthetic validation of Milestone 05 (NFL movement outcomes, game-grouped reliability, target freeze) — callable from
a notebook cell, no subprocess, no warehouse access.

    from nflbdb.research.advanced_movement import nfl_outcome_validation
    report = nfl_outcome_validation.run_all()          # table: check, status, detail
    nfl_outcome_validation.assert_all_passed(report)

Two kinds of fixture, both with a known answer:

``kin_play``         frames of ONE player-play at 10 Hz with an analytic speed profile (constant speed, constant
                     acceleration, a braking and re-acceleration sequence), so every window measure has an exact value.
``synthetic_plays``  a play table: play measure = position offset + athlete effect + athlete-game effect + window noise
                     (+ an alignment effect), so the reliability across games has a closed form:
                     σ²_a / (σ²_a + σ²_g / G + σ²_w / (G·W)) for an athlete mean over G games of W windows.

Every check writes only inside a temporary directory that it removes and raises ``AssertionError`` on failure.
``tests/test_advanced_movement_nfl_outcomes.py`` calls the same functions.
"""

from __future__ import annotations

import inspect
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ...features.config import KinematicsConfig, SmoothingSpec
from ...features.kinematics import compute_kinematics
from ...features.play_windows import PP_KEYS
from ...features.trajectory import NFL, prepare_frames
from ..contracts import conform
from ..studies.stats import pearson
from . import nfl_outcomes as no
from . import outcome_reliability as orl
from .artifacts import MissingArtifactError, NflOutcomeStore
from .config import NotebookControls
from .validation import _fake_cfg, _raises, failure_location

DT, HALF = 0.1, 3                                   # 10 Hz; half-width of the primary 7-frame filter
T0 = datetime(2024, 9, 8, 13, 0, 0)
POSITIONS = ("DB", "DL", "WR")
C = no.OutcomeConfig()                              # measure definitions exactly as the notebooks use them
CFG = no.OutcomeConfig(n_boot=60, n_game_splits=8, min_athletes=10, min_athletes_position=5, min_windows_grid=(10, 20),
                       primary_min_windows=20, min_half_windows=3, first_n_grid=(10, 20), first_games_grid=(2, 4),
                       holdout_min_windows=10, min_game_windows=2, min_cell_windows=10)
MEAN_ID = "O1.speed_p90_mean"                       # athlete MEAN of the play measure speed_p90: closed-form reliability
SCANNED = ("nfl_outcomes.py", "outcome_reliability.py", "nfl_outcome_figures.py")
COMBINE_SOURCES = ("mart_athlete_research_base", "mart_athlete_combine_profile", "fact_combine_attempt", "combine_tracking",
                   "combine_frame_kinematics", "combine_attempt_features", "combine_sprint_profiles",
                   "attempt_censoring_diagnostics", "transfer_candidates", "fpca_athlete_scores", "attempt_embeddings")


# ============================================================================== fixtures
def kin_play(speed: Sequence[float], a_tan: Sequence[float] | None = None, game_id: int = 1, play_id: int = 1, nfl_id: int = 10,
             seg_breaks: Sequence[int] = ()) -> pl.DataFrame:
    """Kinematic frames of one player-play (the columns ``start_table`` reads). ``seg_breaks``: frame indices at which a
    new tracking segment starts. ``step_smooth`` = speed × Δt, missing on the first frame of a segment."""
    v = np.asarray(speed, dtype=float)
    n = v.size
    a = np.gradient(v, DT) if a_tan is None else np.asarray(a_tan, dtype=float)
    bounds = [0, *seg_breaks, n]
    seg, pos, length = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
    step = v * DT
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        seg[lo:hi], pos[lo:hi], length[lo:hi] = k + 1, np.arange(hi - lo), hi - lo
        step[lo] = np.nan
    return pl.DataFrame({"game_id": np.full(n, game_id, np.int64), "play_id": np.full(n, play_id, np.int64),
                         "nfl_id": np.full(n, nfl_id, np.int64), "seg_idx": seg, "pos": pos, "seg_len": length,
                         "ts": [T0 + timedelta(milliseconds=100 * i) for i in range(n)], "speed": v, "a_tan": a,
                         "step_smooth": pl.Series(step, dtype=pl.Float64, nan_to_null=True)})


def snaps_at(frame: int = 0, game_id: int = 1, play_id: int = 1) -> pl.DataFrame:
    return pl.DataFrame({"game_id": [game_id], "play_id": [play_id], "snap_ts": [T0 + timedelta(milliseconds=100 * frame)]},
                        schema_overrides={"game_id": pl.Int64, "play_id": pl.Int64})


def measures_of(kin: pl.DataFrame, snaps: pl.DataFrame, c: no.OutcomeConfig = C, exclude_boundary: bool | None = None) -> dict[str, Any]:
    out = no.play_table(no.start_table(kin, snaps, c, HALF, DT), c, exclude_boundary)
    assert out.height == 1, f"expected one player-play, got {out.height}"
    return out.row(0, named=True)


def synthetic_plays(c: no.OutcomeConfig, n_athletes: int = 90, n_games: int = 8, n_windows: int = 8, seed: int = 0,
                    sd: tuple[float, float, float] = (1.0, 0.3, 1.0), pos_offset: tuple[float, float, float] = (0.0, 3.0, 6.0),
                    align_effect: float = 0.0) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Play table in the contract of ``nfl_play_movement``: every athlete has ``n_games`` games of ``n_windows`` eligible
    windows, in time order. Every play measure = signal + small independent noise (SD 0.2), where
    signal = 6 + position offset + athlete (sd[0]) + athlete-game (sd[1]) + window (sd[2]) + align_effect·[alignment A1];
    the 1.0 s sustained acceleration is pure noise. Returns (plays, truth) with the athlete effects and the athlete's
    probability of alignment A1 (0.2 or 0.8, balanced over positions)."""
    rng = np.random.default_rng(seed)
    A, G, W = n_athletes, n_games, n_windows
    ai, gi, wi = np.repeat(np.arange(A), G * W), np.tile(np.repeat(np.arange(G), W), A), np.tile(np.arange(W), A * G)
    n = ai.size
    code = ai % len(POSITIONS)
    mu, gam = rng.normal(0.0, sd[0], A), rng.normal(0.0, sd[1], (A, G))
    p_align = np.where((np.arange(A) // len(POSITIONS)) % 2 == 0, 0.2, 0.8)
    a1 = rng.random(n) < p_align[ai]
    signal = 6.0 + np.asarray(pos_offset)[code] + mu[ai] + gam[ai, gi] + rng.normal(0.0, sd[2], n) + align_effect * a1
    half = max(G // 2, 1)
    season, week = 2023 + (gi >= half).astype(np.int64), gi % half + 1
    passing = rng.random(n) < 0.6
    names = np.array(POSITIONS)[code]
    data: dict[str, Any] = {
        "game_id": season * 1_000_000 + week * 1_000 + ai % 16, "play_id": wi + 1, "nfl_id": 1000 + ai, "season": season, "week": week,
        "position_group": names.tolist(), "cohort": 2023 + ai % 3, "side": np.where(code == 2, "offense", "defense").tolist(),
        "lined_up_position": [f"{p}_{'A1' if x else 'A2'}" for p, x in zip(names, a1)],
        "play_class": np.where(passing, "pass", "no_pass").tolist(), "pass_result": np.where(passing, "C", "").tolist(),
        "down": rng.integers(1, 5, n), "quality_tier": np.where(rng.random(n) < 0.7, "A", "B").tolist(),
        "in_player_play": np.ones(n, bool), "eligible": np.ones(n, bool), "exclusion_reason": [""] * n, "game_order": gi + 1,
        "play_order": gi * W + wi + 1, "post_s": np.full(n, 4.0), "boundary_frac": np.full(n, 0.1), "brake_valid": np.ones(n, bool)}
    for k in no.count_columns(c):
        data[k] = np.full(n, 40 if k == "n_post_frames" else 10, np.int64)
    for k in no.measure_columns(c):
        data[k] = signal + rng.normal(0.0, 0.2, n)
    data[f"accel_sust{no.wlab(c.sustain_s[-1])}"] = rng.normal(3.0, 1.0, n)
    plays = pl.DataFrame(data).select([pl.col(k).cast(t) for k, t in no.plays_contract(c).columns.items()])
    return plays, {"nfl_id": 1000 + np.arange(A), "mu": mu, "p_align": p_align}


def synthetic_variants(plays: pl.DataFrame, c: no.OutcomeConfig, seed: int = 1) -> pl.DataFrame:
    """Two measure variants on the same player-plays: the primary values plus a small independent perturbation."""
    rng = np.random.default_rng(seed)
    m = no.measure_columns(c)
    parts = [plays.select(pl.lit(v).alias("variant"), *PP_KEYS,
                          *[(pl.col(k) + pl.lit(pl.Series(rng.normal(0.0, s, plays.height)))).cast(pl.Float32).alias(k) for k in m])
             for v, s in (("sg9p2", 0.05), ("boundary_incl", 0.02))]
    return pl.concat(parts, how="vertical").select(list(no.variants_contract(c).columns))


def theory(sd: tuple[float, float, float], games: float, windows_per_game: float, extra: float = 0.04) -> float:
    """Reliability of an athlete mean over ``games`` games of ``windows_per_game`` windows (extra = measure noise variance)."""
    va, vg, vw = sd[0] ** 2, sd[1] ** 2, sd[2] ** 2 + extra
    return va / (va + vg / games + vw / (games * windows_per_game))


def _row(rel: pl.DataFrame, method: str, c: no.OutcomeConfig = CFG, oid: str = MEAN_ID, adj: str = "raw", group: str = orl.WITHIN) -> dict[str, Any]:
    r = rel.filter((pl.col("outcome_id") == oid) & (pl.col("method") == method) & (pl.col("adjustment") == adj)
                   & (pl.col("group") == group) & (pl.col("min_windows") == c.primary_min_windows))
    assert r.height == 1, f"{r.height} rows for {oid} / {method} / {adj} / {group}"
    return r.row(0, named=True)


def _close(what: str, observed: float, expected: float, tol: float) -> None:
    assert observed is not None and abs(float(observed) - expected) <= tol, f"{what}: observed {observed}, expected {expected} ± {tol}"


# ============================================================================== 1 · windows and units
def check_window_construction_and_units() -> str:
    v = 1.0 + 2.0 * DT * np.arange(60)                              # constant acceleration 2 yd/s² from 1 yd/s
    kin = kin_play(v, a_tan=np.full(60, 2.0))
    starts = no.start_table(kin, snaps_at(0), C, HALF, DT)
    assert starts.height == 60 and list(starts.columns) == list(no.start_schema(C))
    m = measures_of(kin, snaps_at(0))
    _close("sustained acceleration over 0.5 s [yd/s²]", m["accel_sust05"], 2.0, 1e-9)
    _close("sustained acceleration over 1.0 s [yd/s²]", m["accel_sust10"], 2.0, 1e-9)
    _close("speed gain over 0.5 s [yd/s]", m["gain05"], 1.0, 1e-9)
    _close("speed gain over 1.0 s [yd/s]", m["gain10"], 2.0, 1e-9)
    _close("gain / window = mean acceleration", m["gain05"] / 0.5, m["accel_sust05"], 1e-9)
    _close("largest 0.5 s mean speed (start frame 51, edge windows excluded) [yd/s]", m["speed_sust05"], 1.0 + 0.2 * 53.5, 1e-9)
    _close("single-frame peak speed [yd/s]", m["speed_max"], float(v[-1]), 1e-12)
    _close("play p90 speed [yd/s]", m["speed_p90"], 1.0 + 0.2 * 0.9 * 59, 1e-9)
    assert m["n_post_frames"] == 60 and abs(m["post_s"] - 5.9) < 1e-9
    flat = measures_of(kin_play(np.full(60, 5.0), a_tan=np.zeros(60)), snaps_at(0))      # 0.5 yd per frame
    _close("speed gain over 5 yd at constant speed [yd/s]", flat["gain_d"], 0.0, 1e-12)
    _close("time to cover 5 yd at 5 yd/s [s]", flat["gain_d_time"], 1.0, 1e-9)
    athletes = pl.DataFrame({"nfl_id": [10], "position_group": ["WR"]})
    hist = conform(no.finish_hist([no.phase_hist(starts, athletes, C)], C), no.PHASE_SPACE)
    assert int(hist["n_frames"].sum()) == starts.filter(pl.col("a0").is_not_null()).height == 60
    assert hist["accel_lo"].unique().to_list() == [2.0]
    return (f"constant 2 yd/s²: sustained acceleration {m['accel_sust05']:.3f} yd/s², gain {m['gain05']:.2f} yd/s per 0.5 s and "
            f"{m['gain10']:.2f} per 1.0 s; 5 yd at 5 yd/s takes {flat['gain_d_time']:.1f} s; the phase-space histogram keeps all 60 frames")


# ============================================================================== 2 · snap anchor
def check_snap_anchor_excludes_pre_snap() -> str:
    v = np.r_[np.full(20, 11.0), np.full(40, 3.0)]                  # fast motion BEFORE the snap must not count
    a = np.zeros(60)
    kin = pl.concat([kin_play(v, a), kin_play(v, a, play_id=2)])
    out = no.play_table(no.start_table(kin, snaps_at(20), C, HALF, DT), C)
    assert out.height == 1 and out["play_id"].to_list() == [1], "a play without a snap anchor must give no row"
    m = out.row(0, named=True)
    assert m["n_post_frames"] == 40 and m["speed_max"] == 3.0 and m["gain05"] == 0.0
    _close("post-snap duration [s]", m["post_s"], 3.9, 1e-9)
    return "40 post-snap frames kept, 20 pre-snap frames at 11 yd/s ignored (peak 3.0 yd/s); the play without a snap tag has no row"


# ============================================================================== 3 · segments and boundaries
def check_segment_boundaries() -> str:
    v = np.r_[np.full(30, 2.0), np.full(30, 9.0)]                   # two tracking segments: a window across them would gain 7 yd/s
    kin = kin_play(v, np.zeros(60), seg_breaks=(30,))
    ex, inc = measures_of(kin, snaps_at(0)), measures_of(kin, snaps_at(0), exclude_boundary=False)
    assert ex["gain05"] == 0.0 and inc["gain05"] == 0.0 and ex["gain10"] == 0.0, "a window crossed a segment break"
    b = f"band_n_{no.blab(2.0, 4.0)}"
    assert ex[b] == 30 - 11, f"start frames away from filter edges in a 30-frame segment: {ex[b]} (expected 19)"
    assert inc[b] == 30 - 5, f"start frames with a full 0.5 s window in a 30-frame segment: {inc[b]} (expected 25)"
    _close("share of filter-boundary frames", ex["boundary_frac"], 12 / 60, 1e-12)
    short = no.play_table(no.start_table(kin_play(np.full(8, 4.0), np.zeros(8)), snaps_at(0), C, HALF, DT), C).row(0, named=True)
    assert short["gain05"] is None and short["accel_sust05"] is None and short["speed_max"] == 4.0
    return ("no window spans a segment break; a 30-frame segment gives 19 start frames with edge windows excluded and 25 with them "
            "kept; an 8-frame segment has no sustained measure (kept missing)")


# ============================================================================== 4 · speed bands
def check_speed_band_definitions() -> str:
    v = 0.05 + 0.1 * np.arange(100)                                 # 1 yd/s² ramp; no value on a band edge
    m = measures_of(kin_play(v, np.full(100, 1.0)), snaps_at(0))
    for lo, hi in C.speed_bands:
        b = no.blab(lo, hi)
        expected = 17 if lo == 0.0 else 20                          # the first 3 frames are filter-boundary frames
        assert m[f"band_n_{b}"] == expected and m[f"band_npos_{b}"] == expected, (b, m[f"band_n_{b}"], m[f"band_npos_{b}"])
        _close(f"band acceleration {b} [yd/s²]", m[f"band_acc_{b}"], 1.0, 1e-9)
        _close(f"band envelope point {b} [yd/s]", m[f"band_gain_{b}"], 0.5, 1e-9)
    w = np.full(80, 5.0)
    w[40:42] = 3.0                                                  # two frames only in the 2–4 band
    few = measures_of(kin_play(w, np.zeros(80)), snaps_at(0))
    b = no.blab(2.0, 4.0)
    assert few[f"band_n_{b}"] == 2 and few[f"band_gain_{b}"] is None and few[f"band_acc_{b}"] is None
    assert [no.blab(lo, hi) for lo, hi in C.speed_bands] == ["v0_2", "v2_4", "v4_6", "v6_8"]
    return (f"bands are half-open [lo, hi) on the STARTING speed: 20 start frames per 2 yd/s band on a 1 yd/s² ramp, band acceleration "
            f"1.0 yd/s²; a band visited on fewer than {C.min_band_starts} frames stays missing")


# ============================================================================== 5 · braking and re-acceleration
def check_braking_reacceleration_rule() -> str:
    v = np.r_[np.full(20, 8.0), 8.0 - 0.4 * np.arange(10), 4.0 + 0.3 * np.arange(15), np.full(25, 8.5)]
    a = np.r_[np.zeros(20), np.full(10, -4.0), np.full(15, 3.0), np.zeros(25)]
    m = measures_of(kin_play(v, a), snaps_at(0))
    _close("sustained braking [yd/s²]", m["decel_sust"], 4.0, 1e-9)
    assert m["brake_valid"] and m["brake_v0"] >= C.brake_min_speed
    _close("re-acceleration gain over 0.5 s [yd/s]", m["reaccel_gain"], 1.5, 1e-9)
    calm = measures_of(kin_play(np.full(60, 6.0), np.zeros(60)), snaps_at(0))
    assert not calm["brake_valid"] and calm["reaccel_gain"] is None and calm["decel_sust"] == 0.0
    return ("a 4 yd/s² braking from 8 yd/s followed by 3 yd/s² gives decel 4.0 yd/s² and a re-acceleration gain of 1.5 yd/s; a "
            "play without a qualifying braking has no re-acceleration value")


# ============================================================================== 6 · smoothing
def _raw_play(noise: float, seed: int = 0, n: int = 50) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = DT * np.arange(n)
    x = 10.0 + 1.0 * t + 0.5 * 2.0 * t ** 2 + rng.normal(0.0, noise, n) if noise else 10.0 + 1.0 * t + 0.5 * 2.0 * t ** 2
    y = 20.0 + (rng.normal(0.0, noise, n) if noise else np.zeros(n))
    return pl.DataFrame({"season": np.full(n, 2024, np.int64), "game_id": np.full(n, 1, np.int64), "play_id": np.full(n, 1, np.int64),
                         "nfl_id": np.full(n, 10, np.int64), "frame_id": np.arange(n, dtype=np.int64),
                         "ts": [T0 + timedelta(milliseconds=100 * i) for i in range(n)], "x": x, "y": y})


def check_smoothing_behaviour() -> str:
    kcfg = KinematicsConfig()
    assert abs(kcfg.dt_s - DT) < 1e-12 and SmoothingSpec(*C.primary_smoothing).window // 2 == HALF
    frames, _ = prepare_frames(_raw_play(0.0), NFL, kcfg)
    kin = compute_kinematics(frames, kcfg, SmoothingSpec(*C.primary_smoothing))
    m = measures_of(kin, snaps_at(0))
    _close("sustained acceleration from positions, quadratic motion [yd/s²]", m["accel_sust05"], 2.0, 1e-6)
    _close("speed gain over 1.0 s from positions [yd/s]", m["gain10"], 2.0, 1e-6)
    noisy, _ = prepare_frames(_raw_play(0.02, seed=5), NFL, kcfg)
    sd = {}
    for w in (5, 9):
        k = compute_kinematics(noisy, kcfg, SmoothingSpec(w, 2))
        inner = k.filter((pl.col("pos") >= 5) & (pl.col("pos") < pl.col("seg_len") - 5))
        sd[w] = float((inner["a_tan"] - 2.0).std())
    assert sd[9] < sd[5], f"a wider filter must carry less acceleration noise: {sd}"
    peak = {w: measures_of(compute_kinematics(noisy, kcfg, SmoothingSpec(w, 2)), snaps_at(0), C)["accel_sust05"] for w in (5, 9)}
    return (f"the real segmentation + Savitzky–Golay chain returns 2.000 yd/s² on noise-free quadratic motion; with 0.02 yd position "
            f"noise the acceleration error SD is {sd[5]:.2f} (5 frames) against {sd[9]:.2f} yd/s² (9 frames), and the play maximum of the "
            f"0.5 s mean reads {peak[5]:.2f} against {peak[9]:.2f} (a maximum rises with noise)")


# ============================================================================== 7 · deduplication, eligibility, order
def _measure_rows(c: no.OutcomeConfig, rows: Sequence[dict[str, Any]]) -> pl.DataFrame:
    sch = no.measure_schema(c)
    return pl.DataFrame([{k: r.get(k) for k in sch} for r in rows], schema=sch)


def check_player_play_deduplication_and_order() -> str:
    g23b, g23a, g24 = 2023000200, 2023000100, 2024000100               # week 2, week 1, next season
    rows = [{"game_id": g, "play_id": p, "nfl_id": 10, "n_post_frames": 30, "speed_max": 5.0} for g in (g23b, g23a, g24) for p in (1, 2)]
    rows[0]["n_post_frames"] = 5                                        # too short
    rows += [{"game_id": g23a, "play_id": 1, "nfl_id": 11, "n_post_frames": 30},     # position out of scope
             {"game_id": g23a, "play_id": 1, "nfl_id": 12, "n_post_frames": 30},     # nullified by penalty
             {"game_id": g23a, "play_id": 2, "nfl_id": 12, "n_post_frames": 30},     # no quality record
             {"game_id": g23b, "play_id": 1, "nfl_id": 12, "n_post_frames": 30}]     # quality tier C
    measures = _measure_rows(C, rows)
    keys = measures.select(PP_KEYS)
    labels = keys.with_columns(pl.lit("defense").alias("side"), pl.lit("LCB").alias("lined_up_position"),
                               pl.when((pl.col("nfl_id") == 12) & (pl.col("game_id") == g23a) & (pl.col("play_id") == 1))
                               .then(pl.lit("Y")).otherwise(pl.lit("N")).alias("play_nullified_by_penalty"),
                               pl.lit(True).alias("in_player_play"))
    labels = pl.concat([labels, labels.head(3)])                        # duplicated label rows must not fan out
    quality = (keys.filter(~((pl.col("nfl_id") == 12) & (pl.col("game_id") == g23a) & (pl.col("play_id") == 2)))
               .with_columns(pl.when((pl.col("nfl_id") == 12) & (pl.col("game_id") == g23b)).then(pl.lit("C")).otherwise(pl.lit("A"))
                             .alias("quality_tier")))
    context = keys.with_columns(pl.lit(1, dtype=pl.Int64).alias("down"),
                                pl.when(pl.col("play_id") == 1).then(pl.lit("C")).otherwise(pl.lit(None, dtype=pl.String)).alias("pass_result"))
    athletes = pl.DataFrame({"nfl_id": [10, 11, 12], "position_group": ["DB", "QB", "DL"], "cohort": [2023, 2023, 2024]})
    games = pl.DataFrame({"game_id": [g23a, g23b, g24], "season": [2023, 2023, 2024], "week": [1, 2, 1]})
    plays = conform(no.assemble_plays(measures, labels, quality, context, athletes, games, C), no.plays_contract(C))
    assert plays.height == measures.height, "one row per player-play: a duplicated label row changed the row count"
    why = {(r["nfl_id"], r["game_id"], r["play_id"]): r["exclusion_reason"] for r in plays.iter_rows(named=True)}
    assert why[(10, g23b, 1)] == "short_post_snap_window" and why[(11, g23a, 1)] == "position_out_of_scope"
    assert why[(12, g23a, 1)] == "nullified_by_penalty" and why[(12, g23a, 2)] == "no_quality_record"
    assert why[(12, g23b, 1)] == "quality_tier_not_usable"
    mine = plays.filter((pl.col("nfl_id") == 10) & pl.col("eligible")).sort("play_order")
    assert mine["game_id"].to_list() == [g23a, g23a, g23b, g24, g24], "eligible windows are not in (season, week, game, play) order"
    assert mine["game_order"].to_list() == [1, 1, 2, 3, 3] and mine["play_order"].to_list() == [1, 2, 3, 4, 5]
    assert plays.filter(~pl.col("eligible"))["play_order"].null_count() == plays.filter(~pl.col("eligible")).height
    assert set(plays.filter(pl.col("nfl_id") == 10)["play_class"].to_list()) == {"pass", "no_pass"}
    funnel = conform(no.funnel_table(plays, C), no.FUNNEL).filter(pl.col("position_group") == no.ALL).sort("stage_order")
    n = funnel["n_player_plays"].to_list()
    assert n == sorted(n, reverse=True) and n[0] == plays.height and n[-1] == plays.filter(pl.col("eligible")).height == 5
    return (f"{plays.height} player-plays stay {plays.height} rows with duplicated labels; the first failing rule is the exclusion "
            "reason; eligible windows and games are numbered in time order across seasons; the funnel is monotone")


# ============================================================================== 8 · reliability across games
def check_game_grouped_reliability() -> str:
    sd = (0.6, 0.5, 2.0)
    plays, _ = synthetic_plays(CFG, n_athletes=300, n_games=8, n_windows=8, seed=11, sd=sd)
    eng = no.OutcomeEngine(plays, CFG)
    spec = eng.by_id[MEAN_ID]
    rel = conform(orl.reliability_table(eng, CFG, [spec]), orl.RELIABILITY)
    truth = theory(sd, 8, 8)
    oe, rh = _row(rel, "game_odd_even_sb"), _row(rel, "game_random_halves_sb")
    _close("odd / even games, stepped up", oe["estimate"], truth, 0.08)
    _close("random game halves, stepped up", rh["estimate"], truth, 0.08)
    assert oe["ci_lower"] <= oe["estimate"] <= oe["ci_upper"] and oe["ci_upper"] - oe["ci_lower"] < 0.25
    assert oe["n_athletes"] == 300, f"the unit is the athlete (300), not the window ({plays.height}): {oe['n_athletes']}"
    _close("single-window ICC", _row(rel, "play_icc1")["estimate"], sd[0] ** 2 / (sd[0] ** 2 + sd[1] ** 2 + sd[2] ** 2 + 0.04), 0.04)
    # a strong game effect: halves that share games look far more reliable than halves of independent games
    heavy = (0.6, 1.0, 1.0)
    p2, _ = synthetic_plays(CFG, n_athletes=300, n_games=8, n_windows=8, seed=12, sd=heavy)
    e2 = no.OutcomeEngine(p2, CFG)
    pair = e2.pairs(e2.by_id[MEAN_ID])
    codes = orl._codes(pair["position_group"].to_list())
    by_game = orl.sb2(orl.within_r(pair["value_0"].to_numpy(), pair["value_1"].to_numpy(), codes))
    w = (e2.frame(e2.by_id[MEAN_ID]).with_columns((pl.col("w_order") % 2).alias("h")).group_by("nfl_id", "h")
         .agg(pl.col("y").mean().alias("v"), pl.col("position_group").first()))
    j = w.filter(pl.col("h") == 0).join(w.filter(pl.col("h") == 1), on="nfl_id", suffix="_b").sort("nfl_id")
    by_window = orl.sb2(orl.within_r(j["v"].to_numpy(), j["v_b"].to_numpy(), orl._codes(j["position_group"].to_list())))
    _close("game split under a strong game effect", by_game, theory(heavy, 8, 8), 0.08)
    assert by_window > by_game + 0.10, f"splitting windows inside games must overstate reliability: {by_window:.3f} vs {by_game:.3f}"
    assert abs(orl.sb2(0.5) - 2 / 3) < 1e-12 and np.isnan(orl.sb2(float("nan")))
    return (f"closed-form reliability {truth:.3f}: odd / even games {oe['estimate']:.3f} [{oe['ci_lower']:.3f}, {oe['ci_upper']:.3f}], random "
            f"halves {rh['estimate']:.3f}, on 300 athletes; with a strong game effect a window split reads {by_window:.3f} where the game "
            f"split reads {by_game:.3f} (truth {theory(heavy, 8, 8):.3f})")


# ============================================================================== 9 · exposure
def check_exposure_thresholds_keep_time_order() -> str:
    plays, _ = synthetic_plays(CFG, n_athletes=30, n_games=8, n_windows=8, seed=2)
    plays = plays.with_columns(pl.col("play_order").cast(pl.Float64).alias("speed_p90"))      # value = position in time
    plays = plays.filter((pl.col("nfl_id") != 1000) | (pl.col("play_order") <= 6))            # one athlete with 6 windows only
    eng = no.OutcomeEngine(plays, CFG)
    spec = eng.by_id[MEAN_ID]
    p = orl.budget_pairs(eng, spec, "windows", 10)
    assert 1000 not in p["nfl_id"].to_list() and p.height == 29, "an athlete without the budget must be absent"
    assert p["n_0"].unique().to_list() == [10] and p["value_0"].unique().to_list() == [5.5], "first 10 windows = orders 1..10"
    assert p["value_1"].unique().to_list() == [(17 + 64) / 2], "later games start AFTER the game holding the 10th window"
    assert p["n_1"].unique().to_list() == [48]
    g = orl.budget_pairs(eng, spec, "games", 2)
    assert g["n_0"].unique().to_list() == [16] and g["value_0"].unique().to_list() == [8.5] and g["value_1"].unique().to_list() == [40.5]
    proj = [orl.projected_reliability(1.0, 0.25, 4.0, n, 8.0) for n in (20, 50, 200)]
    assert proj[0] < proj[1] < proj[2] < 1.0
    _close("projection at 16 windows over 2 games", orl.projected_reliability(1.0, 0.25, 4.0, 16, 8.0), 1 / (1 + 0.125 + 0.25), 1e-12)
    smoke = no.OutcomeConfig().for_mode(NotebookControls(run_mode="smoke"))
    for cfg in (no.OutcomeConfig(), smoke, CFG):
        assert cfg.primary_min_windows in cfg.min_windows_grid and cfg.primary_min_windows in cfg.first_n_grid
    assert (smoke.sustain_s, smoke.speed_bands, smoke.primary_smoothing, smoke.min_band_starts) == (C.sustain_s, C.speed_bands, C.primary_smoothing, C.min_band_starts)
    return ("the first-N value uses the athlete's first N windows in time; its reference uses only games played after them (no shared "
            f"game); athletes without the budget are absent; projected reliability rises with exposure ({proj[0]:.2f} → {proj[2]:.2f}); "
            "smoke mode changes budgets, never a measure definition")


# ============================================================================== 10 · position and season grouping
def check_season_and_position_grouping() -> str:
    plays, _ = synthetic_plays(CFG, n_athletes=150, n_games=8, n_windows=8, seed=4, sd=(0.0, 0.3, 1.0))   # NO athlete effect
    eng = no.OutcomeEngine(plays, CFG)
    spec = eng.by_id[MEAN_ID]
    pair = eng.pairs(spec)
    pooled = pearson(pair["value_0"].to_numpy(), pair["value_1"].to_numpy())
    rel = orl.reliability_table(eng, CFG, [spec])
    within = _row(rel, "game_odd_even")["estimate"]
    assert pooled > 0.9 and abs(within) < 0.3, f"position means must not count as athlete reliability: pooled {pooled:.2f}, within {within:.2f}"
    groups = set(rel.filter(pl.col("method") == "game_odd_even")["group"].to_list())
    assert groups == {orl.WITHIN, *POSITIONS}, groups
    seasons = rel.filter(pl.col("method").str.starts_with("season_"))
    assert seasons["method"].unique().to_list() == ["season_2023_to_2024"] and seasons["n_athletes"].max() == 150
    assert abs(seasons.filter(pl.col("adjustment") == "raw")["estimate"][0]) < 0.3
    return (f"three positions with different means and NO athlete effect: pooled half-split correlation {pooled:.2f}, within position "
            f"{within:+.2f}; estimates exist per position group and between consecutive seasons")


# ============================================================================== 11 · context
def check_context_comparability() -> str:
    plays, truth = synthetic_plays(CFG, n_athletes=150, n_games=8, n_windows=8, seed=6, sd=(0.0, 0.0, 1.0), pos_offset=(0.0, 0.0, 0.0),
                                   align_effect=2.0)
    rare = (pl.col("nfl_id") == 1000) & (pl.col("play_order") <= 5)
    plays = plays.with_columns(pl.when(rare).then(pl.lit("RARE")).otherwise(pl.col("lined_up_position")).alias("lined_up_position"))
    eng = no.OutcomeEngine(plays, CFG)
    spec = eng.by_id[MEAN_ID]
    v = eng.values(spec).join(pl.DataFrame({"nfl_id": truth["nfl_id"], "p": truth["p_align"]}), on="nfl_id").sort("nfl_id")
    raw, adj = pearson(v["value"].to_numpy(), v["p"].to_numpy()), pearson(v["value_adj"].to_numpy(), v["p"].to_numpy())
    assert raw > 0.8 and abs(adj) < 0.3, f"alignment mix must drive the raw value ({raw:.2f}) and not the adjusted one ({adj:.2f})"
    f = eng.frame(spec)
    small = f.filter(pl.col("lined_up_position") == "RARE")
    assert small.height == 5 and not small["cell_adjusted"].any() and (small["y"] == small["r"]).all(), "a small cell must stay unadjusted"
    assert f.filter(pl.col("lined_up_position") != "RARE")["cell_adjusted"].all()
    ctx = conform(orl.context_audit(eng, no.athlete_outcomes(eng), CFG), orl.CONTEXT)
    eta = ctx.filter((pl.col("outcome_id") == MEAN_ID) & (pl.col("position_group") == "DB") & (pl.col("variable") == "lined_up_position")
                     & (pl.col("level") == "__ALL__"))["eta2"][0]
    assert eta > 0.3, f"alignment explains about half of the play-measure variance here: {eta}"
    return (f"an alignment effect of 2 yd/s with athletes differing in alignment mix: athlete value follows the mix (r = {raw:.2f}) until "
            f"cell means are removed (r = {adj:+.2f}); η² of alignment {eta:.2f}; a cell with fewer than {CFG.min_cell_windows} windows is "
            "left unadjusted and flagged")


# ============================================================================== 12 · missingness
def check_missingness_and_composites() -> str:
    plays, _ = synthetic_plays(CFG, n_athletes=30, n_games=8, n_windows=8, seed=8)
    band = f"band_gain_{no.blab(*CFG.speed_bands[-1])}"
    plays = plays.with_columns(
        pl.when(pl.col("play_id") == 1).then(None).otherwise(pl.col("speed_p90")).alias("speed_p90"),
        pl.when(pl.col("position_group") == "DL").then(None).otherwise(pl.col(band)).alias(band))
    eng = no.OutcomeEngine(plays, CFG)
    outcomes = conform(no.athlete_outcomes(eng), no.ATHLETE_OUTCOMES)
    mine = outcomes.filter(pl.col("outcome_id") == MEAN_ID)
    assert mine["n_windows"].unique().to_list() == [56], "an undefined window must not be counted"
    one = plays.filter((pl.col("nfl_id") == 1003) & pl.col("speed_p90").is_not_null())["speed_p90"].mean()
    _close("athlete mean over DEFINED windows only", mine.filter(pl.col("nfl_id") == 1003)["value"][0], one, 1e-9)
    cov = conform(no.coverage_table(eng, outcomes), no.coverage_contract(CFG))
    row = cov.filter((pl.col("outcome_id") == MEAN_ID) & (pl.col("position_group") == no.ALL)).row(0, named=True)
    assert row["n_plays_undefined"] == 30 * 8 and abs(row["opportunity_rate"] - 7 / 8) < 1e-12 and row["n_athletes_tracked"] == 30
    env = outcomes.filter(pl.col("outcome_id") == "O5.envelope_level")
    assert "DL" not in env["position_group"].to_list() and env.height == 20, "a composite needs every component: DL athletes lack a band"
    assert cov.filter((pl.col("outcome_id") == "O5.envelope_level") & (pl.col("position_group") == "DL"))["n_athletes_any"][0] == 0
    return ("undefined windows are excluded from the statistic and counted as excluded (240 of 1,920, opportunity rate 0.875); an "
            "athlete lacking one speed band has no envelope value instead of an average")


# ============================================================================== 13 · target freeze
def _screen_fixture(c: no.OutcomeConfig, statuses: dict[str, str]) -> pl.DataFrame:
    specs = {s.outcome_id: s for s in no.outcome_specs(c)}
    rows = [{**{k: None for k in orl.SCREEN.columns}, "outcome_id": o, "family": specs[o].family, "role": specs[o].role, "n_athletes": 80,
             "n_positions": 3, "reliability": 0.85, "reliability_lower": 0.70, "reliability_upper": 0.92, "pass_coverage": True,
             "pass_reliability": st != "not_eligible", "pass_smoothing": st == "eligible", "pass_exposure": True, "pass_context": True,
             "n_passed": 5 if st == "eligible" else 4, "status": st, "reasons": ""} for o, st in statuses.items()]
    return pl.DataFrame(rows, schema=orl.SCREEN.columns)


def _coverage_fixture(c: no.OutcomeConfig, ids: Sequence[str]) -> pl.DataFrame:
    contract = no.coverage_contract(c)
    rows = [{**{k: None for k in contract.columns}, "outcome_id": o, "position_group": g, "n_athletes_tracked": 90, "n_athletes_any": 90,
             f"n_athletes_ge_{c.primary_min_windows}": 80, "n_windows": 5000, "n_athlete_games": 700, "n_eligible_plays": 5200}
            for o in ids for g in (no.ALL, "DB")]
    return pl.DataFrame(rows, schema=contract.columns)


def check_target_freeze() -> str:
    c = CFG
    good, good2, prov, bad, ref = "O1.speed_sust10_p90", "O4.gain10_p90", "O2.accel_sust05_mean", "O3.band_acc_v2_4_mean", "O1.speed_max_p95"
    screen = _screen_fixture(c, {good: "eligible", good2: "eligible", prov: "provisional", bad: "not_eligible", ref: "reference_only"})
    dictionary, coverage = no.outcome_dictionary(c), _coverage_fixture(c, [good, good2, prov, bad, ref])
    proposal = {"decision": "FREEZE", "primary": good, "secondary": []}
    with tempfile.TemporaryDirectory() as tmp:
        store = SimpleNamespace(data_dir=Path(tmp) / "gold", controls=NotebookControls(run_mode="smoke"))
        path = store.data_dir / f"{orl.TARGETS_FILE}.json"

        def freeze(decision, primary=None, secondary=(), rationale="", refreeze=False, st=store):
            return orl.freeze_targets(st, decision, primary, secondary, rationale, proposal, screen, dictionary, coverage, c,
                                      {"source": "synthetic"}, refreeze)

        doc, msg = freeze(None)
        assert doc is None and msg.startswith("PENDING") and not path.exists(), "an unrecorded decision must write nothing"
        assert _raises(ValueError, lambda: freeze("MAYBE", good))
        assert _raises(ValueError, lambda: freeze("FREEZE", bad, rationale="because"))
        assert _raises(ValueError, lambda: freeze("FREEZE", ref, rationale="because")), "a reference outcome can never be a target"
        assert _raises(ValueError, lambda: freeze("FREEZE", prov)), "a provisional outcome needs a written rationale"
        assert _raises(ValueError, lambda: freeze("FREEZE", good, [good2, prov, bad], "three secondaries"))
        assert _raises(ValueError, lambda: freeze("FREEZE", good, [good], "duplicate"))
        assert not path.exists()
        dry = SimpleNamespace(data_dir=store.data_dir, controls=NotebookControls(run_mode="smoke", save_tables=False))
        assert freeze("FREEZE", good, st=dry)[1].startswith("NOT WRITTEN") and not path.exists()
        doc, msg = freeze("FREEZE", good, [prov], "provisional secondary kept for sensitivity")
        assert msg.startswith("frozen now") and path.is_file() and orl.gate_05(doc)[0] == "PASS"
        t = doc["targets"]
        assert [x["rank"] for x in t] == ["primary", "secondary"] and [x["outcome_id"] for x in t] == [good, prov]
        for field in ("formula", "units", "context_definition", "exposure_requirement", "inclusion_exclusion", "smoothing",
                      "grouping_grain", "reliability", "eligibility_counts", "definition_hash"):
            assert t[0].get(field), f"frozen target lacks {field}"
        assert doc["selection"]["combine_data_consulted"] is False and len(doc["version_hash"]) == 16
        assert t[0]["definition_hash"] == orl.target_definition(good, "primary", dictionary, screen, coverage, c)["definition_hash"]
        kept, msg = freeze("FREEZE", good2)
        assert kept["targets"][0]["outcome_id"] == good and "DIFFERENT" in msg, "a second freeze must not replace the first silently"
        assert "reproduces" in freeze("FREEZE", good, [prov], "provisional secondary kept for sensitivity")[1]
        stop, msg = freeze("STOP", rationale="no outcome accepted", refreeze=True)
        assert stop["decision"] == "STOP" and not stop["targets"] and orl.gate_05(stop)[0] == "STOP" and "REFREEZE" in msg
        assert orl.gate_05(None)[0] == "PENDING"
    red = pl.DataFrame({"outcome_a": [good], "outcome_b": [good2], "n_athletes": [80], "spearman": [0.95]}, schema=orl.REDUNDANCY.columns)
    prop = orl.propose_targets(screen, red, c)
    assert prop["decision"] == "FREEZE" and prop["primary"] in (good, good2) and prop["secondary"] == [], prop
    assert orl.propose_targets(screen.filter(pl.col("status") != "eligible"), red, c)["decision"] == "STOP"
    return ("nothing is written while the decision is pending; reference, not-eligible and unexplained provisional outcomes are "
            "refused; the artifact carries formula, units, context, exposure, reliability, counts and hashes; a second freeze is kept "
            "apart unless REFREEZE is set; a near-duplicate outcome is not proposed as a secondary; no eligible outcome → STOP")


# ============================================================================== 14 · no Combine access
def check_no_combine_access() -> str:
    assert _raises(no.CombineLeakError, lambda: no.assert_no_combine_columns(pl.DataFrame({"nfl_id": [1], "cmb_forty": [4.4]})))
    plays, _ = synthetic_plays(CFG, n_athletes=12, n_games=4, n_windows=4, seed=1)
    assert _raises(no.CombineLeakError, lambda: no.OutcomeEngine(plays.with_columns(pl.lit(4.4).alias("cmb_forty")), CFG))
    screen = _screen_fixture(CFG, {"O1.speed_sust10_p90": "eligible"}).with_columns(pl.lit(0.3).alias("forty_rho"))
    store = SimpleNamespace(data_dir=Path("unused"), controls=NotebookControls(run_mode="smoke"))
    assert _raises(no.CombineLeakError, lambda: orl.freeze_targets(
        store, "FREEZE", "O1.speed_sust10_p90", [], "", {}, screen, no.outcome_dictionary(CFG),
        _coverage_fixture(CFG, ["O1.speed_sust10_p90"]), CFG, {}))
    here = Path(__file__).resolve().parent
    hits = [f"{name}: {s}" for name in SCANNED for s in COMBINE_SOURCES if s in (here / name).read_text(encoding="utf-8")]
    assert not hits, f"a Milestone 05 module names a Combine source: {hits}"
    for fn in (orl.screen_table, orl.propose_targets, orl.freeze_targets, orl.reliability_table, orl.exposure_table):
        bad = [p for p in inspect.signature(fn).parameters if any(k in p.lower() for k in ("combine", "forty", "fpca", "tcn", "official"))]
        assert not bad, f"{fn.__name__} takes a Combine argument: {bad}"
    assert set(plays.columns) == set(no.plays_contract(CFG).columns)
    return (f"{len(SCANNED)} module files name no Combine table or representation; a Combine-prefixed column stops the engine and the "
            "freeze; the screening, proposal and freeze functions take NFL evidence only")


# ============================================================================== 15 · artifacts
def check_artifact_schema_and_isolation() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fake_cfg(Path(tmp))
        smoke, full = NflOutcomeStore(cfg, NotebookControls(run_mode="smoke")), NflOutcomeStore(cfg, NotebookControls(run_mode="full"))
        for s in (smoke, full):
            assert s.data_dir.name == s.tables_dir.name == s.figures_dir.name == "nfl_outcomes" and s.manifest_path.parent == s.data_dir
            assert s.data_dir != s.upstream.data_dir and s.models_dir.name == "nfl_outcomes"
        assert "_sample" in smoke.data_dir.parts and "_sample" in smoke.tables_dir.parts and "_sample" not in full.data_dir.parts
        calls = []

        def builder() -> dict[str, pl.DataFrame]:
            calls.append(1)
            return {no.DICTIONARY.name: no.outcome_dictionary(CFG)}

        out = smoke.build("NFL_DICTIONARY", [no.DICTIONARY], builder, params=CFG.params())[no.DICTIONARY.name]
        smoke.build("NFL_DICTIONARY", [no.DICTIONARY], builder, params=CFG.params())
        assert len(calls) == 1, "an unchanged build must be restored from cache"
        smoke.build("NFL_DICTIONARY", [no.DICTIONARY], builder, params=C.params())
        assert len(calls) == 2, "a changed configuration must recompute"
        assert out["outcome_id"].n_unique() == out.height and set(out["role"].to_list()) <= set(no.ROLES)
        assert smoke.csv(no.DICTIONARY).is_file() and smoke.parquet(no.DICTIONARY).is_file()
        from ...data.utils import read_json

        side = read_json(smoke.parquet(no.DICTIONARY).with_suffix(".schema.json"), {})
        assert side["provenance"]["run_mode"] == "smoke" and len(side["columns"]) == len(no.DICTIONARY.columns) and side["key"] == ["outcome_id"]
        assert not full.exists(no.DICTIONARY) and _raises(MissingArtifactError, lambda: full.require(no.DICTIONARY, "05a"))
        assert not list(smoke.upstream.data_dir.glob("*.parquet")), "Milestone 05 wrote into the area of notebooks 00–03"
    units = {r["outcome_id"]: r["unit"] for r in no.outcome_dictionary(C).iter_rows(named=True)}
    assert all(u == ("yd/s²" if o.startswith(("O2", "O3", "O6.decel")) else "yd/s") for o, u in units.items()), units
    roles = {s.outcome_id: s.role for s in no.outcome_specs(C)}
    assert roles["O1.speed_max_p95"] == "reference" and all(r == "exploratory" for o, r in roles.items() if o.startswith("O6"))
    return (f"outputs live under …/advanced_movement/nfl_outcomes (smoke under _sample) with their own run cache; the dictionary has "
            f"{len(units)} uniquely keyed outcomes with units; the single-frame peak is a reference and O6 is exploratory by declaration")


# ============================================================================== 16 · end to end
def check_engine_end_to_end() -> str:
    c = CFG
    plays, _ = synthetic_plays(c, n_athletes=90, n_games=8, n_windows=8, seed=3)
    rng = np.random.default_rng(3)
    keep = pl.DataFrame({"nfl_id": 1000 + np.arange(90), "_keep": rng.integers(5, 9, 90)})       # 5–8 games: exposure varies
    plays = plays.join(keep, on="nfl_id").filter(pl.col("game_order") <= pl.col("_keep")).drop("_keep")
    variants = conform(synthetic_variants(plays, c), no.variants_contract(c))
    eng = no.OutcomeEngine(plays, c)
    outcomes = conform(no.athlete_outcomes(eng), no.ATHLETE_OUTCOMES)
    games = conform(no.athlete_game_outcomes(eng), no.ATHLETE_GAMES)
    dictionary = conform(no.outcome_dictionary(c), no.DICTIONARY)
    coverage = conform(no.coverage_table(eng, outcomes), no.coverage_contract(c))
    rel = conform(orl.reliability_table(eng, c), orl.RELIABILITY)
    exp = conform(orl.exposure_table(eng, c, rel), orl.EXPOSURE)
    ctx = conform(orl.context_audit(eng, outcomes, c), orl.CONTEXT)
    sens = conform(orl.sensitivity_table(eng, plays, variants, c), orl.SENSITIVITY)
    red = conform(orl.redundancy_table(outcomes, c), orl.REDUNDANCY)
    screen = conform(orl.screen_table(eng.specs, coverage, rel, exp, sens, ctx, c), orl.SCREEN)
    assert screen.height == dictionary.height == len(eng.specs) and games.height > 0
    st = {r["outcome_id"]: r for r in screen.iter_rows(named=True)}
    signal, noise = st[MEAN_ID], st[f"O2.accel_sust{no.wlab(c.sustain_s[-1])}_p90"]
    assert signal["pass_coverage"] and signal["pass_reliability"] and signal["reliability"] > 0.8, signal
    assert not noise["pass_reliability"] and noise["status"] == "not_eligible", noise
    assert st["O1.speed_max_p95"]["status"] == "reference_only" and st["O6.decel_sust_p90"]["status"] == "exploratory"
    assert st[f"O5.band_gain_{no.blab(*c.speed_bands[0])}_p90"]["status"] == "component"
    assert set(sens["kind"].unique().to_list()) == {"smoothing", "boundary", "quality_filter"}
    assert {"first_vs_later_games", "within_budget_game_split_sb", "model_projection"} <= set(exp["method"].unique().to_list())
    proposal = orl.propose_targets(screen, red, c)
    assert proposal["decision"] in orl.DECISIONS
    if proposal["decision"] == "FREEZE":
        chosen = [proposal["primary"], *proposal["secondary"]]
        assert len(chosen) <= 3 and all(st[o]["status"] == "eligible" and st[o]["role"] == "candidate" for o in chosen), chosen
        assert f"O2.accel_sust{no.wlab(c.sustain_s[-1])}_p90" not in chosen
    text = orl.conclusions_markdown(screen, proposal, None, c)
    assert "PENDING" in text and "Rule-based proposal" in text
    return (f"{len(eng.specs)} outcomes on 90 synthetic athletes pass every table contract; planted repeatable outcome reliability "
            f"{signal['reliability']:.2f} ({signal['status']}), planted noise {noise['reliability']:.2f} (not eligible); rule-based proposal "
            f"{proposal['decision']} (primary {proposal['primary']})")


CHECKS_05A: tuple[Callable[[], str], ...] = (
    check_window_construction_and_units, check_snap_anchor_excludes_pre_snap, check_segment_boundaries, check_speed_band_definitions,
    check_braking_reacceleration_rule, check_smoothing_behaviour, check_player_play_deduplication_and_order,
    check_missingness_and_composites, check_no_combine_access, check_artifact_schema_and_isolation)
CHECKS_05B: tuple[Callable[[], str], ...] = (
    check_game_grouped_reliability, check_exposure_thresholds_keep_time_order, check_season_and_position_grouping,
    check_context_comparability, check_missingness_and_composites, check_target_freeze, check_no_combine_access,
    check_engine_end_to_end)
ALL_CHECKS = tuple(dict.fromkeys((*CHECKS_05A, *CHECKS_05B)))


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    """Run every synthetic check in-process. A failing check is reported, not raised, so one table shows them all."""
    rows = []
    for fn in checks:
        try:
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "PASS", "detail": fn()})
        except Exception as e:                                   # noqa: BLE001 — the table must list every failure
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "FAIL",
                         "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


def assert_all_passed(report: pl.DataFrame) -> None:
    failed = report.filter(pl.col("status") != "PASS")
    if failed.height:
        raise AssertionError(f"{failed.height} synthetic validation check(s) failed: "
                             + "; ".join(f"{r['check']} → {r['detail']}" for r in failed.iter_rows(named=True)))
