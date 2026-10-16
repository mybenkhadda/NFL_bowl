"""Milestone 09A — NFL movement-demand atlas: what speed-change demands occur on the field, by position.

A *movement demand* is a physically defined episode inside the post-snap tracking of one player-play. Categories are
CANDIDATE PHYSICAL CATEGORIES, not football roles: no route, assignment, coverage technique or rush responsibility is
inferred from a movement shape. Where the warehouse holds a vendor label it is carried as context; otherwise the
situation is described in physical terms only.

Units of analysis
    frame      one 100 ms sample — never a sample size.
    episode    one maximal run of frames meeting the rule of a demand, inside ONE tracking segment of one player-play.
    athlete    the independent unit of every count that is called a sample size.

What is reused (nothing of the warehouse is rebuilt, no Milestone 00–08 output is rewritten)
    ``NflOutcomeBuilder``            game batches, snap anchors, segmentation and Savitzky–Golay kinematics (Milestone 05)
    ``nfl_play_movement``            the ELIGIBLE player-plays of Milestone 05A and their observed context labels
    ``crossing_time`` / ``band_accel``   the same first-crossing convention as the atlas feature ``time_v4_to_v8_s``;
                                     ``movement_matching`` applies the SAME functions to the Combine forty.

Demand rules (``DemandConfig``; yd, s, yd/s, yd/s², deg)
    accel_low            accelerating run (a_tan ≥ accel_on for ≥ min_run_s) starting below 2 yd/s, gain ≥ min_gain
    accel_mid            the same, starting at 2–5 yd/s
    transition_high      the same, starting at ≥ 5 yd/s and reaching the high-speed line
    sustained_high       speed ≥ high_speed for ≥ sustain_s
    speed_maintenance    |a_tan| < accel_on at speed ≥ maintain_min_speed for ≥ sustain_s
    braking              decelerating run (a_tan ≤ −accel_on for ≥ brake_min_s) from ≥ brake_min_v0, drop ≥ brake_drop
    reaccel_after_brake  an accelerating run that starts within reaccel_gap_s after a braking episode
    redirection          heading change ≥ turn_min_deg at |yaw rate| ≥ turn_rate and speed ≥ turn_min_speed, only where
                         every frame has a valid heading (magnitude only: the handedness convention is unverified)
    repeated_change      a player-play with ≥ 2 accelerating episodes and ≥ 1 braking episode

Episodes of different demands may overlap in time (a high-speed run contains a maintenance run). Thresholds are
proposed values fixed before any distribution was read; they are heuristics of a descriptive atlas.
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import get_logger
from ...features.config import SmoothingSpec
from ...features.kinematics import compute_kinematics
from ...features.play_windows import PLAY_KEYS, PP_KEYS
from ...features.trajectory import NFL, prepare_frames
from ..contracts import TableContract
from .config import NotebookControls
from .nfl_outcomes import NflOutcomeBuilder, OutcomeConfig

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
DEMANDS_VERSION = "1"
ALL = "ALL"
BANDS: tuple[tuple[float, float], ...] = ((2.0, 4.0), (4.0, 6.0), (6.0, 8.0))


def band_col(lo: float, hi: float) -> str:
    return f"acc_{lo:g}_{hi:g}"


BAND_COLS = tuple(band_col(lo, hi) for lo, hi in BANDS)
ACCEL_DEMANDS = ("accel_low", "accel_mid", "transition_high", "reaccel_after_brake")
CONTEXT_VARS = ("side", "lined_up_position", "play_class", "down", "season", "route_label", "coverage_label")


@dataclass(frozen=True)
class DemandConfig:
    """Provenance: proposed = fixed before any distribution was read · established = reused project value."""

    version: str = DEMANDS_VERSION
    dt_s: float = 0.1                    # established: 10 Hz tracking
    accel_on: float = 1.0                # proposed [yd/s²]: a frame belongs to an accelerating / braking run beyond this
    min_run_s: float = 0.5               # proposed: shortest accelerating run (established sustain window of Milestone 05)
    min_gain: float = 1.5                # proposed [yd/s]: smallest speed gain of an accelerating episode
    low_speed: float = 2.0               # established: MovementConfig.onset_speed_fast
    mid_speed: float = 5.0               # proposed
    high_speed: float = 7.5              # proposed [yd/s] ≈ 15.3 mph: the high-speed running line
    sustain_s: float = 1.0               # proposed: shortest sustained / steady episode
    maintain_min_speed: float = 5.0      # proposed
    brake_min_s: float = 0.4             # proposed
    brake_min_v0: float = 4.0            # proposed
    brake_drop: float = 2.0              # proposed [yd/s]
    reaccel_gap_s: float = 1.0           # proposed
    turn_rate: float = 60.0              # proposed [deg/s]
    turn_min_speed: float = 3.0          # proposed: below it the heading of a 10 Hz track is not trusted
    turn_min_deg: float = 45.0           # proposed
    turn_min_s: float = 0.3              # proposed
    frame_tolerance_s: float = 0.05      # established: PlayWindowConfig.frame_tolerance_s
    min_episodes: int = 30               # proposed: episodes of a position × demand cell before a distribution is shown
    min_athletes: int = 10               # proposed: athletes of the cell
    example_quantiles: tuple[float, ...] = (0.5, 0.75, 0.9)
    example_margin_s: float = 0.5        # frames kept before / after an example episode
    hist_bins: int = 24

    def params(self) -> dict[str, Any]:
        return {**asdict(self), "bands": [list(b) for b in BANDS]}

    def for_mode(self, controls: NotebookControls) -> "DemandConfig":
        """Smoke mode keeps every rule; it lowers the support lines so a few games exercise every code path."""
        return replace(self, min_episodes=5, min_athletes=2, example_quantiles=(0.5,)) if controls.smoke else self


@dataclass(frozen=True)
class DemandSpec:
    demand_id: str
    label: str
    definition: str
    primary: str                 # the intensity feature of the demand
    unit: str
    features: tuple[str, ...]    # the dimensions reported for it
    combine_analogue: str        # shared dimensions measurable in the forty-yard dash ("" = none)
    comparability: str
    limitation: str


def demand_specs(c: DemandConfig) -> tuple[DemandSpec, ...]:
    acc = f"a run of frames with tangential acceleration ≥ {c.accel_on:g} yd/s² lasting ≥ {c.min_run_s:g} s with a speed gain ≥ {c.min_gain:g} yd/s"
    game = "game movement is rarely maximal: it follows role, opportunity and play context"
    return (
        DemandSpec("accel_low", "acceleration from low speed", f"{acc}, starting below {c.low_speed:g} yd/s", "mean_accel", "yd/s²",
                   ("v_start", "v_max", "speed_change", "duration_s", "distance_yd", "mean_accel", "peak_rate_05", BAND_COLS[0], BAND_COLS[1]),
                   f"{BAND_COLS[0]}, {BAND_COLS[1]}", "same speed bands, same 10 Hz first-crossing rule; the forty starts from rest at maximal effort",
                   game + "; the 2 → 4 yd/s band spans few frames at 10 Hz"),
        DemandSpec("accel_mid", "acceleration from intermediate speed", f"{acc}, starting at {c.low_speed:g}–{c.mid_speed:g} yd/s", "mean_accel", "yd/s²",
                   ("v_start", "v_max", "speed_change", "duration_s", "distance_yd", "mean_accel", "peak_rate_05", BAND_COLS[1], BAND_COLS[2]),
                   f"{BAND_COLS[1]}, {BAND_COLS[2]}", "same speed bands; in the forty the athlete passes through the band once, already accelerating",
                   game + "; a standing-start band transit is not a rolling-start acceleration"),
        DemandSpec("transition_high", "transition into high-speed running", f"{acc}, starting at ≥ {c.mid_speed:g} yd/s and reaching {c.high_speed:g} yd/s",
                   "mean_accel", "yd/s²", ("v_start", "v_max", "speed_change", "duration_s", "distance_yd", "mean_accel", BAND_COLS[2]),
                   BAND_COLS[2], "same speed band (6 → 8 yd/s)", game + "; positions that rarely reach the line have few episodes"),
        DemandSpec("sustained_high", "sustained high-speed running", f"speed ≥ {c.high_speed:g} yd/s for ≥ {c.sustain_s:g} s", "v_max", "yd/s",
                   ("v_max", "v_mean", "duration_s", "distance_yd"), "top_speed",
                   "largest speed reached: tracked top speed of the forty against the peak speed of the episode",
                   game + "; an observed game speed is not a physical maximum"),
        DemandSpec("speed_maintenance", "speed maintenance", f"|tangential acceleration| < {c.accel_on:g} yd/s² at ≥ {c.maintain_min_speed:g} yd/s for ≥ {c.sustain_s:g} s",
                   "v_mean", "yd/s", ("v_mean", "speed_change", "duration_s", "distance_yd"), "",
                   "none: forty recordings usually end at or before the speed peak (last-frame censoring)", "no maintenance phase is recorded in the forty"),
        DemandSpec("braking", "braking and deceleration", f"a run with tangential acceleration ≤ −{c.accel_on:g} yd/s² for ≥ {c.brake_min_s:g} s from ≥ "
                   f"{c.brake_min_v0:g} yd/s with a speed drop ≥ {c.brake_drop:g} yd/s", "mean_accel", "yd/s²",
                   ("v_start", "speed_change", "duration_s", "distance_yd", "mean_accel", "peak_rate_05"), "",
                   "none: the forty contains no braking phase inside the timed run", "contact and the end of a play also produce decelerations that are not braking"),
        DemandSpec("reaccel_after_brake", "acceleration following deceleration", f"{acc}, starting within {c.reaccel_gap_s:g} s after a braking episode",
                   "mean_accel", "yd/s²", ("v_start", "v_max", "speed_change", "duration_s", "mean_accel", "peak_rate_05"), "",
                   "none: the forty contains no braking, so no re-acceleration", "no validated manoeuvre label; the braking rule is a provisional threshold"),
        DemandSpec("redirection", "directional redirection", f"|heading change| ≥ {c.turn_min_deg:g}° at |yaw rate| ≥ {c.turn_rate:g} deg/s and speed ≥ "
                   f"{c.turn_min_speed:g} yd/s, every frame with a valid heading", "heading_change_deg", "deg",
                   ("heading_change_deg", "v_start", "v_mean", "speed_change", "duration_s"), "",
                   "none: the forty is a straight run", "magnitude only (handedness of the frame unverified); heading is the direction of MOTION, not body orientation"),
        DemandSpec("repeated_change", "repeated speed-change sequence", "a player-play with ≥ 2 accelerating episodes and ≥ 1 braking episode", "n_changes", "episodes",
                   ("n_changes", "duration_s", "distance_yd", "v_max"), "", "none: a single maximal sprint has one acceleration phase",
                   "a count of episodes of this atlas, sensitive to its thresholds"),
    )


EPISODE_FLOATS = ("t_start", "duration_s", "v_start", "v_end", "v_max", "v_mean", "speed_change", "distance_yd", "mean_accel", "peak_rate_05",
                  "heading_change_deg", *BAND_COLS)
EPISODES = TableContract(
    "nfl_movement_demand_episodes", "One row per movement-demand episode of an eligible NFL player-play (post-snap, primary smoothing).",
    {"game_id": I, "play_id": I, "nfl_id": I, "episode": I, "demand_id": S, "season": I, "week": I, "position_group": S, "side": S, "lined_up_position": S,
     "play_class": S, "down": I, "route_label": S, "coverage_label": S, **{k: F for k in EPISODE_FLOATS}, "n_changes": I, "edge": B},
    key=("game_id", "play_id", "nfl_id", "episode"), kind="dataset", area="NFL player-play × episode",
    docs={"t_start": "seconds after the snap", "speed_change": "v_end − v_start [yd/s]", "mean_accel": "speed_change / duration_s [yd/s²]",
          "peak_rate_05": "largest (accelerating) or most negative (braking) speed change over 0.5 s, per second [yd/s²]",
          "heading_change_deg": "signed integral of the yaw rate; null unless every frame has a valid heading — use the magnitude",
          BAND_COLS[1]: "(hi − lo) / time between the FIRST crossings of lo and hi inside the episode; null unless both are recorded",
          "edge": "the episode touches a Savitzky–Golay boundary frame of its segment (excluded from every profile)",
          "route_label": "vendor label when the warehouse holds one (source column in the manifest); never inferred from movement"})
REGISTRY = TableContract(
    "nfl_movement_demand_registry", "Definition of every movement-demand category and of what the forty can and cannot say about it.",
    {"demand_id": S, "label": S, "physical_definition": S, "primary_feature": S, "unit": S, "features": S, "combine_analogue": S, "comparability": S,
     "limitation": S, "measurable": B, "n_episodes": I}, key=("demand_id",), area="demand")
FREQUENCY = TableContract(
    "nfl_movement_demand_frequency", "How often each demand occurs, by position group: exposure counts beside every rate.",
    {"position_group": S, "demand_id": S, "n_episodes": I, "n_athletes": I, "n_plays_with": I, "n_eligible_plays": I, "episodes_per_100_plays": F,
     "share_plays_with": F, "median_duration_s": F, "median_distance_yd": F, "supported": B}, key=("position_group", "demand_id"), area="position group × demand",
    docs={"supported": "enough episodes and athletes for a distribution to be shown (DemandConfig.min_episodes / min_athletes)"})
PROFILE = TableContract(
    "nfl_movement_demand_profiles", "Distribution of each demand dimension by position group, with within- and across-athlete variability.",
    {"position_group": S, "demand_id": S, "feature": S, "unit": S, "is_primary": B, "n_episodes": I, "n_athletes": I, "mean": F, "q10": F, "q25": F, "q50": F,
     "q75": F, "q90": F, "within_athlete_sd": F, "between_athlete_sd": F, "athlete_share": F, "supported": B},
    key=("position_group", "demand_id", "feature"), area="position group × demand × dimension",
    docs={"athlete_share": "between-athlete variance / (between + within), one-way method of moments: how much of an episode value is the athlete",
          "q50": "quantiles over EPISODES (dependent observations); n_athletes is the sample size"})
CONTEXT = TableContract(
    "nfl_movement_demand_context", "Demand counts and median intensity by observed context level.",
    {"context_var": S, "level": S, "position_group": S, "demand_id": S, "n_episodes": I, "n_athletes": I, "median_primary": F},
    key=("context_var", "level", "position_group", "demand_id"), area="context × position group × demand")
AVAILABILITY = TableContract(
    "nfl_movement_demand_context_availability", "Which football-context labels the episode table actually carries.",
    {"context_var": S, "source": S, "n_episodes_labelled": I, "share_labelled": F, "n_levels": I, "status": S}, key=("context_var",), area="context variable")
ATHLETE = TableContract(
    "nfl_movement_demand_athletes", "Athlete-level summary of each demand (observed game movement; not a capability).",
    {"nfl_id": I, "position_group": S, "demand_id": S, "n_episodes": I, "n_games": I, "median_primary": F, "p90_primary": F},
    key=("nfl_id", "demand_id"), kind="dataset", area="athlete × demand")
DISTRIBUTION = TableContract(
    "nfl_movement_demand_distributions", "Histogram of each demand dimension by position group (application figures).",
    {"position_group": S, "demand_id": S, "feature": S, "bin": I, "bin_lo": F, "bin_hi": F, "n_episodes": I},
    key=("position_group", "demand_id", "feature", "bin"), kind="dataset", area="position group × demand × dimension × bin")
EXAMPLES = TableContract(
    "nfl_movement_demand_examples", "Recorded frames of rule-chosen example episodes (real plays; no hand picking).",
    {"example_id": S, "position_group": S, "demand_id": S, "quantile": F, "game_id": I, "play_id": I, "nfl_id": I, "season": I, "frame": I, "t_s": F,
     "in_episode": B, "speed": F, "a_tan": F, "rel_x": F, "rel_y": F, "primary_value": F}, key=("example_id", "frame"), kind="dataset",
    area="example episode × frame",
    docs={"t_s": "seconds since the start of the episode (negative = margin before it)",
          "rel_x": "smoothed field position minus the position at the start of the episode [yd] — a displacement, no field landmark",
          "quantile": "the episode whose primary intensity is closest to this quantile of its position × demand cell"})
CONTRACTS = (EPISODES, REGISTRY, FREQUENCY, PROFILE, CONTEXT, AVAILABILITY, ATHLETE, DISTRIBUTION)
UNITS = {"v_start": "yd/s", "v_end": "yd/s", "v_max": "yd/s", "v_mean": "yd/s", "speed_change": "yd/s", "duration_s": "s", "distance_yd": "yd",
         "mean_accel": "yd/s²", "peak_rate_05": "yd/s²", "heading_change_deg": "deg", "n_changes": "episodes", "t_start": "s", **{k: "yd/s²" for k in BAND_COLS}}


# ============================================================================== pure trace functions (shared with the Combine side)
def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of the maximal True runs of ``mask``."""
    if not mask.size:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def crossing_time(t: np.ndarray, v: np.ndarray, level: float) -> float:
    """Time at which ``v`` FIRST rises through ``level`` (linear interpolation between the two frames). NaN when the
    trace starts at or above the level (the crossing was not recorded), never reaches it, or the frame before is missing."""
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


def band_accel(t: np.ndarray, v: np.ndarray, lo: float, hi: float) -> float:
    """Mean acceleration through a speed band: (hi − lo) / time between the first crossings of lo and hi [yd/s²]."""
    a, b = crossing_time(t, v, lo), crossing_time(t, v, hi)
    return float((hi - lo) / (b - a)) if np.isfinite(a) and np.isfinite(b) and b > a else NAN


def _episode(demand: str, i: int, j: int, t: np.ndarray, v: np.ndarray, yaw: np.ndarray, step: np.ndarray, hv: np.ndarray, edge: np.ndarray,
             c: DemandConfig) -> dict[str, Any]:
    sv = v[i:j]
    dur = float(t[j - 1] - t[i])
    d = {"demand_id": demand, "_i": i, "_j": j, "t_start": float(t[i]), "duration_s": dur, "v_start": float(sv[0]), "v_end": float(sv[-1]),
         "v_max": float(np.max(sv)), "v_mean": float(np.mean(sv)), "speed_change": float(sv[-1] - sv[0]),
         "distance_yd": float(np.nansum(step[i + 1:j])), "mean_accel": float((sv[-1] - sv[0]) / dur) if dur > 0 else NAN, "peak_rate_05": NAN,
         "heading_change_deg": float(np.sum(yaw[i:j]) * c.dt_s) if bool(hv[i:j].all()) and np.isfinite(yaw[i:j]).all() else NAN,
         **{k: NAN for k in BAND_COLS}, "n_changes": None, "edge": bool(edge[i:j].any())}
    k = int(round(0.5 / c.dt_s))
    if sv.size > k and demand in (*ACCEL_DEMANDS, "braking"):
        g = (sv[k:] - sv[:-k]) / 0.5
        d["peak_rate_05"] = float(g.min() if demand == "braking" else g.max())
    if demand in ACCEL_DEMANDS:
        for (lo, hi), name in zip(BANDS, BAND_COLS):
            d[name] = band_accel(t[i:j], sv, lo, hi)
    return d


def segment_trace(t: np.ndarray, v: np.ndarray, a: np.ndarray, yaw: np.ndarray, step: np.ndarray, hv: np.ndarray, edge: np.ndarray,
                  c: DemandConfig) -> list[dict[str, Any]]:
    """Episodes of ONE contiguous trace (frames in time order, one tracking segment). Pure NumPy; no table access."""
    n = v.size
    if n < 3:
        return []
    fin = np.isfinite(v) & np.isfinite(a)
    need = lambda s: int(round(s / c.dt_s)) + 1                     # frames spanning s seconds
    args = (t, v, yaw, step, hv, edge, c)
    brakes = []
    for i, j in runs(fin & (a <= -c.accel_on)):
        if j - i >= need(c.brake_min_s) and v[i] >= c.brake_min_v0 and v[i] - v[j - 1] >= c.brake_drop:
            brakes.append(_episode("braking", i, j, *args))
    brake_ends = np.array([t[e["_j"] - 1] for e in brakes], float)
    accels = []
    for i, j in runs(fin & (a >= c.accel_on)):
        if j - i < need(c.min_run_s) or v[j - 1] - v[i] < c.min_gain:
            continue
        gap = t[i] - brake_ends if brake_ends.size else np.array([np.inf])
        if bool(((gap >= -1e-9) & (gap <= c.reaccel_gap_s + 1e-9)).any()):
            demand = "reaccel_after_brake"
        elif v[i] < c.low_speed:
            demand = "accel_low"
        elif v[i] < c.mid_speed:
            demand = "accel_mid"
        elif np.max(v[i:j]) >= c.high_speed:
            demand = "transition_high"
        else:
            continue
        accels.append(_episode(demand, i, j, *args))
    out = brakes + accels
    for i, j in runs(np.isfinite(v) & (v >= c.high_speed)):
        if j - i >= need(c.sustain_s):
            out.append(_episode("sustained_high", i, j, *args))
    for i, j in runs(fin & (np.abs(a) < c.accel_on) & (v >= c.maintain_min_speed)):
        if j - i >= need(c.sustain_s):
            out.append(_episode("speed_maintenance", i, j, *args))
    turning = hv & fin & np.isfinite(yaw) & (v >= c.turn_min_speed)
    for sign in (1.0, -1.0):
        for i, j in runs(turning & (sign * yaw >= c.turn_rate)):
            if j - i >= need(c.turn_min_s) and abs(float(np.sum(yaw[i:j]) * c.dt_s)) >= c.turn_min_deg:
                out.append(_episode("redirection", i, j, *args))
    if len(accels) >= 2 and brakes:
        seq = sorted(accels + brakes, key=lambda e: e["_i"])
        e = _episode("repeated_change", seq[0]["_i"], max(x["_j"] for x in seq), *args)
        e["n_changes"] = len(seq)
        out.append(e)
    out.sort(key=lambda e: (e["_i"], e["demand_id"]))
    return out


RAW_SCHEMA = {"game_id": I, "play_id": I, "nfl_id": I, "episode": I, "demand_id": S, **{k: F for k in EPISODE_FLOATS}, "n_changes": I, "edge": B}


def batch_episodes(kin: pl.DataFrame, snaps: pl.DataFrame, eligible: pl.DataFrame, c: DemandConfig, half_window: int) -> pl.DataFrame:
    """Episodes of every eligible player-play of one batch of frame kinematics (``compute_kinematics`` output)."""
    f = (kin.join(eligible.select(PP_KEYS).unique(), on=PP_KEYS, how="semi")
         .join(snaps.select(*PLAY_KEYS, "snap_ts").unique(subset=PLAY_KEYS), on=PLAY_KEYS, how="inner")
         .with_columns(((pl.col("ts") - pl.col("snap_ts")).dt.total_microseconds().cast(pl.Float64) / 1e6).alias("t_rel"))
         .filter(pl.col("t_rel") >= -c.frame_tolerance_s).sort([*PP_KEYS, "seg_idx", "ts"]))
    if f.is_empty():
        return pl.DataFrame(schema=RAW_SCHEMA)
    num = lambda k: f[k].cast(pl.Float64).fill_null(NAN).to_numpy().astype(float)
    t, v, a, yaw, step = num("t_rel"), num("speed"), num("a_tan"), num("yaw_rate"), num("step_smooth")
    hv = f["heading_valid"].fill_null(False).to_numpy().astype(bool)
    pos, left = f["pos"].to_numpy().astype(np.int64), f["seg_len"].to_numpy().astype(np.int64) - 1 - f["pos"].to_numpy().astype(np.int64)
    edge = (pos < half_window) | (left < half_window)
    keys = [f[k].to_numpy().astype(np.int64) for k in (*PP_KEYS, "seg_idx")]
    change = np.ones(f.height, bool)
    change[1:] = np.any([k[1:] != k[:-1] for k in keys], axis=0)
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], f.height)
    g, p, i_ = keys[0], keys[1], keys[2]
    rows, counter = [], {}
    for s, e in zip(starts.tolist(), ends.tolist()):
        eps = segment_trace(t[s:e], v[s:e], a[s:e], yaw[s:e], step[s:e], hv[s:e], edge[s:e], c)
        if not eps:
            continue
        key = (int(g[s]), int(p[s]), int(i_[s]))
        for ep in eps:
            counter[key] = counter.get(key, 0) + 1
            rows.append({"game_id": key[0], "play_id": key[1], "nfl_id": key[2], "episode": counter[key], **{k: ep[k] for k in RAW_SCHEMA if k in ep}})
    if not rows:
        return pl.DataFrame(schema=RAW_SCHEMA)
    out = pl.DataFrame(rows, schema=RAW_SCHEMA, infer_schema_length=None)
    return out.with_columns([pl.col(k).fill_nan(None) for k in EPISODE_FLOATS])


# ============================================================================== builder (reads the warehouse through Milestone 05 machinery)
class DemandBuilder:
    """Scans silver.game_tracking once, in the game batches of ``NflOutcomeBuilder``, and reduces the eligible
    player-plays of Milestone 05A to movement-demand episodes. Nothing else of the warehouse is read or rebuilt."""

    def __init__(self, cfg: Any, c: DemandConfig, oc: OutcomeConfig, limit_games: int | None = None, athlete_ids: Sequence[int] | None = None):
        self.c, self.oc = c, oc
        self.nfl = NflOutcomeBuilder(cfg, oc, limit_games=limit_games, athlete_ids=athlete_ids)
        self.spec = SmoothingSpec(*oc.primary_smoothing)

    def params(self) -> dict[str, Any]:
        return {"demands": self.c.params(), "nfl": self.nfl.params()}

    def input_files(self) -> list[Any]:
        return self.nfl.input_files()

    def context_labels(self) -> tuple[pl.DataFrame, dict[str, str]]:
        """(game_id, play_id, nfl_id, route_label, coverage_label) from whichever silver column holds such a vendor
        label, and the source of each. A label that the warehouse does not hold stays null: it is never inferred."""
        refs = self.nfl.refs
        pp_cols = refs.scan("silver", "player_play").collect_schema().names()
        play_cols = refs.scan("silver", "plays").collect_schema().names()
        pick = lambda cols, word: next((k for k in sorted(cols) if word in k.lower()), None)
        route, cov_pp, cov_play = pick(pp_cols, "route"), pick(pp_cols, "coverage"), pick(play_cols, "coverage")
        d = (refs.scan("silver", "player_play").select(*PP_KEYS, *[k for k in (route, cov_pp) if k]).collect()
             .with_columns([pl.col(k).cast(pl.Int64) for k in PP_KEYS]).unique(subset=PP_KEYS))
        source = {"route_label": f"silver.player_play.{route}" if route else "", "coverage_label": ""}
        if cov_pp:
            source["coverage_label"] = f"silver.player_play.{cov_pp}"
        elif cov_play:
            pl_ = (refs.scan("silver", "plays").select(*PLAY_KEYS, cov_play).collect().with_columns([pl.col(k).cast(pl.Int64) for k in PLAY_KEYS])
                   .unique(subset=PLAY_KEYS))
            d = d.join(pl_, on=PLAY_KEYS, how="left")
            source["coverage_label"] = f"silver.plays.{cov_play}"
        cov = cov_pp or cov_play
        lab = lambda k: pl.col(k).cast(pl.String).str.strip_chars().replace("", None) if k else pl.lit(None, dtype=pl.String)
        return d.select(*PP_KEYS, lab(route).alias("route_label"), lab(cov).alias("coverage_label")), source

    def compute(self, plays: pl.DataFrame) -> pl.DataFrame:
        """Raw episodes (``RAW_SCHEMA``) of the eligible rows of ``plays`` (the Milestone 05A play table)."""
        t0 = time.perf_counter()
        nb, c = self.nfl, self.c
        anchors = nb.anchors()
        eligible = plays.filter(pl.col("eligible")).select(PP_KEYS)
        parts: list[pl.DataFrame] = []
        for season, file in nb.kin.nfl_season_files().items():
            batches = nb.kin.game_batches(file)
            for bi, (lo, hi) in enumerate(batches, 1):
                snaps = anchors.filter(pl.col("game_id").is_between(lo, hi))
                want = eligible.filter(pl.col("game_id").is_between(lo, hi))
                if snaps.is_empty() or want.is_empty():
                    continue
                raw = (nb.kin.load_nfl_raw(file, season, lo, hi).drop("s", "a", "dir", "o").with_columns([pl.col(k).cast(pl.Int64) for k in PP_KEYS])
                       .join(want, on=PP_KEYS, how="semi"))
                if raw.is_empty():
                    continue
                frames, _ = prepare_frames(raw, NFL, nb.kcfg)
                del raw
                parts.append(batch_episodes(compute_kinematics(frames, nb.kcfg, self.spec), snaps, want, c, self.spec.window // 2))
                log.info("[movement_demands] season %s batch %d/%d (games %s–%s): %s episodes (%.0fs)", season, bi, len(batches), lo, hi,
                         f"{parts[-1].height:,}", time.perf_counter() - t0)
                del frames
                gc.collect()
        if not parts:
            raise RuntimeError("no eligible NFL player-play matched a tracking frame — check that notebook 05A ran in this RUN_MODE")
        return pl.concat(parts, how="vertical")

    def example_frames(self, chosen: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
        """Recorded frames of the chosen example episodes, recomputed from silver one play at a time. Returns the
        frames and the ids of the examples that could not be read (reported, never replaced by another play)."""
        anchors, c = self.nfl.anchors(), self.c
        parts, failed = [], []
        for r in chosen.iter_rows(named=True):
            try:
                kin = self.nfl.kin.nfl_play_kinematics(int(r["game_id"]), int(r["play_id"]), int(r["nfl_id"]), self.spec)
                kin = kin.with_columns([pl.col(k).cast(pl.Int64) for k in PP_KEYS])
                snap = anchors.filter((pl.col("game_id") == r["game_id"]) & (pl.col("play_id") == r["play_id"]))
                d = (kin.join(snap.select(*PLAY_KEYS, "snap_ts"), on=PLAY_KEYS, how="inner")
                     .with_columns(((pl.col("ts") - pl.col("snap_ts")).dt.total_microseconds().cast(pl.Float64) / 1e6 - r["t_start"]).alias("t_s"))
                     .filter((pl.col("t_s") >= -c.example_margin_s - 1e-6) & (pl.col("t_s") <= r["duration_s"] + c.example_margin_s + 1e-6)).sort("ts"))
                if d.height < 3:
                    raise ValueError("fewer than 3 frames in the episode window")
                at = d.filter(pl.col("t_s").abs() == pl.col("t_s").abs().min()).row(0, named=True)
                parts.append(d.select(
                    pl.lit(r["example_id"]).alias("example_id"), pl.lit(r["position_group"]).alias("position_group"), pl.lit(r["demand_id"]).alias("demand_id"),
                    pl.lit(float(r["quantile"])).alias("quantile"), pl.col("game_id"), pl.col("play_id"), pl.col("nfl_id"),
                    pl.lit(r["season"], dtype=pl.Int64).alias("season"), pl.int_range(pl.len(), dtype=pl.Int64).alias("frame"), pl.col("t_s"),
                    ((pl.col("t_s") >= -1e-6) & (pl.col("t_s") <= r["duration_s"] + 1e-6)).alias("in_episode"), pl.col("speed").cast(pl.Float64),
                    pl.col("a_tan").cast(pl.Float64), (pl.col("x_s") - at["x_s"]).cast(pl.Float64).alias("rel_x"),
                    (pl.col("y_s") - at["y_s"]).cast(pl.Float64).alias("rel_y"), pl.lit(float(r["primary_value"])).alias("primary_value")))
            except Exception as e:                                # noqa: BLE001 — one unreadable play must not hide the others
                log.warning("[movement_demands] example %s not read: %s: %s", r["example_id"], type(e).__name__, e)
                failed.append(r["example_id"])
        out = pl.concat(parts, how="vertical") if parts else EXAMPLES.empty()
        return out.select([pl.col(k).cast(t) for k, t in EXAMPLES.columns.items()]), failed


# ============================================================================== tables
def attach_context(raw: pl.DataFrame, plays: pl.DataFrame, labels: pl.DataFrame | None = None) -> pl.DataFrame:
    """Episodes with the observed context of their player-play (Milestone 05A labels + optional vendor labels)."""
    ctx = plays.select(*PP_KEYS, "season", "week", "position_group", "side", "lined_up_position", "play_class", "down").unique(subset=PP_KEYS)
    d = raw.join(ctx, on=PP_KEYS, how="inner")
    if labels is not None:
        d = d.join(labels.unique(subset=PP_KEYS), on=PP_KEYS, how="left")
    else:
        d = d.with_columns(pl.lit(None, dtype=pl.String).alias("route_label"), pl.lit(None, dtype=pl.String).alias("coverage_label"))
    return d.select([pl.col(k).cast(t) for k, t in EPISODES.columns.items()]).sort(list(EPISODES.key))


def _primary(spec: DemandSpec) -> pl.Expr:
    e = pl.col(spec.primary).cast(pl.Float64)
    return e.abs() if spec.primary == "heading_change_deg" else e


def _with_all(d: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([d, d.with_columns(pl.lit(ALL).alias("position_group"))], how="vertical")


def frequency_table(ep: pl.DataFrame, plays: pl.DataFrame, c: DemandConfig) -> pl.DataFrame:
    el = _with_all(plays.filter(pl.col("eligible")).select(*PP_KEYS, "position_group"))
    n_plays = dict(el.group_by("position_group").len().iter_rows())
    d = _with_all(ep.filter(~pl.col("edge")))
    g = d.group_by("position_group", "demand_id").agg(
        pl.len().cast(pl.Int64).alias("n_episodes"), pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
        pl.struct(PP_KEYS).n_unique().cast(pl.Int64).alias("n_plays_with"), pl.col("duration_s").median().alias("median_duration_s"),
        pl.col("distance_yd").median().alias("median_distance_yd"))
    grid = pl.DataFrame([{"position_group": p, "demand_id": s.demand_id} for p in sorted(n_plays) for s in demand_specs(c)], schema={"position_group": S, "demand_id": S})
    out = grid.join(g, on=["position_group", "demand_id"], how="left").with_columns(
        pl.col("n_episodes").fill_null(0), pl.col("n_athletes").fill_null(0), pl.col("n_plays_with").fill_null(0),
        pl.col("position_group").replace_strict(n_plays, return_dtype=pl.Int64).alias("n_eligible_plays"))
    out = out.with_columns((100.0 * pl.col("n_episodes") / pl.col("n_eligible_plays")).alias("episodes_per_100_plays"),
                           (pl.col("n_plays_with") / pl.col("n_eligible_plays")).alias("share_plays_with"),
                           ((pl.col("n_episodes") >= c.min_episodes) & (pl.col("n_athletes") >= c.min_athletes)).alias("supported"))
    return out.select([pl.col(k).cast(t) for k, t in FREQUENCY.columns.items()]).sort("position_group", "demand_id")


def variance_components(y: np.ndarray, ids: np.ndarray) -> tuple[float, float, float]:
    """(within-athlete SD, between-athlete SD, athlete share) by the one-way method of moments. NaN when not estimable."""
    codes, counts = np.unique(ids, return_inverse=True)[1], None
    counts = np.bincount(codes).astype(float)
    k, n = counts.size, float(y.size)
    if k < 2 or n - k < 1:
        return NAN, NAN, NAN
    means = np.bincount(codes, weights=y) / counts
    msw = float(((y - means[codes]) ** 2).sum() / (n - k))
    msb = float((counts * (means - y.mean()) ** 2).sum() / (k - 1))
    n0 = (n - float((counts ** 2).sum()) / n) / (k - 1)
    s2b = max(0.0, (msb - msw) / n0) if n0 > 0 else NAN
    tot = s2b + msw
    return float(np.sqrt(msw)), float(np.sqrt(s2b)) if np.isfinite(s2b) else NAN, float(s2b / tot) if tot > 0 and np.isfinite(s2b) else NAN


def profile_table(ep: pl.DataFrame, c: DemandConfig) -> pl.DataFrame:
    specs = {s.demand_id: s for s in demand_specs(c)}
    rows = []
    for (pos, dem), g in sorted(_with_all(ep.filter(~pl.col("edge"))).partition_by(["position_group", "demand_id"], as_dict=True).items()):
        spec = specs[dem]
        ids_all = g["nfl_id"].to_numpy()
        for feat in spec.features:
            y = g[feat].cast(pl.Float64).fill_null(NAN).to_numpy().astype(float)
            y = np.abs(y) if feat == "heading_change_deg" else y
            ok = np.isfinite(y)
            y, ids = y[ok], ids_all[ok]
            n_ath = int(np.unique(ids).size)
            row = {"position_group": pos, "demand_id": dem, "feature": feat, "unit": UNITS.get(feat, ""), "is_primary": feat == spec.primary,
                   "n_episodes": int(y.size), "n_athletes": n_ath, "supported": bool(y.size >= c.min_episodes and n_ath >= c.min_athletes)}
            if y.size:
                q = np.quantile(y, [0.1, 0.25, 0.5, 0.75, 0.9])
                w, b, share = variance_components(y, ids)
                row |= {"mean": float(y.mean()), "q10": float(q[0]), "q25": float(q[1]), "q50": float(q[2]), "q75": float(q[3]), "q90": float(q[4]),
                        "within_athlete_sd": w, "between_athlete_sd": b, "athlete_share": share}
            rows.append(row)
    out = pl.DataFrame(rows, schema=PROFILE.columns) if rows else PROFILE.empty()
    return out.with_columns([pl.col(k).fill_nan(None) for k in ("within_athlete_sd", "between_athlete_sd", "athlete_share")])


def context_table(ep: pl.DataFrame, c: DemandConfig) -> pl.DataFrame:
    parts = []
    d = ep.filter(~pl.col("edge"))
    for spec in demand_specs(c):
        g = d.filter(pl.col("demand_id") == spec.demand_id).with_columns(_primary(spec).alias("_y"))
        for var in CONTEXT_VARS:
            parts.append(g.filter(pl.col(var).is_not_null()).group_by(pl.col(var).cast(pl.String).alias("level"), "position_group", "demand_id").agg(
                pl.len().cast(pl.Int64).alias("n_episodes"), pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
                pl.col("_y").median().alias("median_primary")).with_columns(pl.lit(var).alias("context_var")))
    out = pl.concat(parts, how="vertical") if parts else CONTEXT.empty()
    return out.select([pl.col(k).cast(t) for k, t in CONTEXT.columns.items()]).sort(list(CONTEXT.key))


def availability_table(ep: pl.DataFrame, sources: dict[str, str] | None = None) -> pl.DataFrame:
    src = {"side": "nfl_play_movement (Phase 03 windows)", "lined_up_position": "nfl_play_movement (Phase 03 windows)",
           "play_class": "nfl_play_movement (grouping of the vendor pass_result; empty code INFERRED as no_pass)", "down": "nfl_play_movement",
           "season": "silver.games", **(sources or {})}
    rows = []
    for var in CONTEXT_VARS:
        n = int(ep[var].is_not_null().sum()) if ep.height else 0
        rows.append({"context_var": var, "source": src.get(var) or "not held by the warehouse", "n_episodes_labelled": n,
                     "share_labelled": n / ep.height if ep.height else 0.0, "n_levels": int(ep[var].drop_nulls().n_unique()) if n else 0,
                     "status": "observed" if n else "unavailable — the situation is described in physical terms only"})
    return pl.DataFrame(rows, schema=AVAILABILITY.columns)


def athlete_table(ep: pl.DataFrame, c: DemandConfig) -> pl.DataFrame:
    parts = []
    d = ep.filter(~pl.col("edge"))
    for spec in demand_specs(c):
        parts.append(d.filter(pl.col("demand_id") == spec.demand_id).with_columns(_primary(spec).alias("_y")).filter(pl.col("_y").is_not_null())
                     .group_by("nfl_id", "demand_id").agg(pl.col("position_group").first(), pl.len().cast(pl.Int64).alias("n_episodes"),
                                                          pl.col("game_id").n_unique().cast(pl.Int64).alias("n_games"), pl.col("_y").median().alias("median_primary"),
                                                          pl.col("_y").quantile(0.9, "linear").alias("p90_primary")))
    out = pl.concat(parts, how="vertical") if parts else ATHLETE.empty()
    return out.select([pl.col(k).cast(t) for k, t in ATHLETE.columns.items()]).sort("nfl_id", "demand_id")


def distribution_table(ep: pl.DataFrame, c: DemandConfig) -> pl.DataFrame:
    """Histograms with edges shared by every position of a demand × dimension (1st–99th percentile of all episodes)."""
    specs = {s.demand_id: s for s in demand_specs(c)}
    rows = []
    d = _with_all(ep.filter(~pl.col("edge")))
    for (dem,), g in sorted(d.partition_by(["demand_id"], as_dict=True).items()):
        for feat in specs[dem].features:
            y = g[feat].cast(pl.Float64).fill_null(NAN).to_numpy().astype(float)
            y = np.abs(y) if feat == "heading_change_deg" else y
            ok = np.isfinite(y)
            if ok.sum() < 2:
                continue
            lo, hi = np.quantile(y[ok], [0.01, 0.99])
            if hi <= lo:
                continue
            edges = np.linspace(lo, hi, c.hist_bins + 1)
            pos = g["position_group"].to_numpy()
            for p in np.unique(pos):
                h = np.histogram(np.clip(y[ok & (pos == p)], lo, hi), bins=edges)[0]
                rows += [{"position_group": str(p), "demand_id": dem, "feature": feat, "bin": b, "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                          "n_episodes": int(h[b])} for b in range(c.hist_bins)]
    return pl.DataFrame(rows, schema=DISTRIBUTION.columns) if rows else DISTRIBUTION.empty()


def registry_table(c: DemandConfig, frequency: pl.DataFrame) -> pl.DataFrame:
    n = dict(frequency.filter(pl.col("position_group") == ALL).select("demand_id", "n_episodes").iter_rows()) if frequency.height else {}
    rows = [{"demand_id": s.demand_id, "label": s.label, "physical_definition": s.definition, "primary_feature": s.primary, "unit": s.unit,
             "features": ", ".join(s.features), "combine_analogue": s.combine_analogue or "none", "comparability": s.comparability, "limitation": s.limitation,
             "measurable": int(n.get(s.demand_id, 0)) > 0, "n_episodes": int(n.get(s.demand_id, 0))} for s in demand_specs(c)]
    return pl.DataFrame(rows, schema=REGISTRY.columns)


def choose_examples(ep: pl.DataFrame, frequency: pl.DataFrame, c: DemandConfig) -> pl.DataFrame:
    """Rule-chosen example episodes: for each SUPPORTED position × demand cell, the episode whose primary intensity is
    closest to each quantile of ``example_quantiles`` (ties: smallest game, play, athlete, episode). No hand picking."""
    ok = {(r["position_group"], r["demand_id"]) for r in frequency.filter(pl.col("supported") & (pl.col("position_group") != ALL)).iter_rows(named=True)}
    rows = []
    d = ep.filter(~pl.col("edge"))
    for spec in demand_specs(c):
        g = d.filter(pl.col("demand_id") == spec.demand_id).with_columns(_primary(spec).alias("_y")).filter(pl.col("_y").is_not_null())
        for (pos,), sub in sorted(g.partition_by(["position_group"], as_dict=True).items()):
            if (pos, spec.demand_id) not in ok:
                continue
            used: set[tuple[int, int, int, int]] = set()
            for q in c.example_quantiles:
                target = sub["_y"].quantile(q, "linear")
                cand = sub.with_columns((pl.col("_y") - target).abs().alias("_d")).sort("_d", *EPISODES.key)
                r = next((x for x in cand.head(len(used) + 1).iter_rows(named=True) if tuple(x[k] for k in EPISODES.key) not in used), None)
                if r is None:
                    continue
                used.add(tuple(r[k] for k in EPISODES.key))
                rows.append({"example_id": f"{pos}-{spec.demand_id}-q{int(round(q * 100)):02d}", "position_group": pos, "demand_id": spec.demand_id,
                             "quantile": float(q), **{k: r[k] for k in (*EPISODES.key, "season", "t_start", "duration_s")}, "primary_value": float(r["_y"])})
    return pl.DataFrame(rows, schema={"example_id": S, "position_group": S, "demand_id": S, "quantile": F, "game_id": I, "play_id": I, "nfl_id": I,
                                      "episode": I, "season": I, "t_start": F, "duration_s": F, "primary_value": F})


def build_tables(raw: pl.DataFrame, plays: pl.DataFrame, c: DemandConfig, labels: pl.DataFrame | None = None, sources: dict[str, str] | None = None
                 ) -> dict[str, pl.DataFrame]:
    """Every table of the atlas from the raw episodes (one call, used by the notebook builder and by the checks)."""
    ep = attach_context(raw, plays, labels)
    freq = frequency_table(ep, plays, c)
    return {EPISODES.name: ep, REGISTRY.name: registry_table(c, freq), FREQUENCY.name: freq, PROFILE.name: profile_table(ep, c),
            CONTEXT.name: context_table(ep, c), AVAILABILITY.name: availability_table(ep, sources), ATHLETE.name: athlete_table(ep, c),
            DISTRIBUTION.name: distribution_table(ep, c)}


def conclusions(frequency: pl.DataFrame, profile: pl.DataFrame, availability: pl.DataFrame) -> list[str]:
    """Plain-language reading of the executed tables (numbers are read here, never typed)."""
    out = []
    pos = [p for p in sorted(frequency["position_group"].unique().to_list()) if p != ALL]
    out.append(f"Positions supported by the warehouse: {', '.join(pos) or 'none'}. A position absent from this list has no observation and is not shown.")
    for r in frequency.filter(pl.col("position_group") == ALL).sort("episodes_per_100_plays", descending=True).iter_rows(named=True):
        out.append(f"{r['demand_id']}: {r['n_episodes']:,} episodes of {r['n_athletes']} athletes, {r['episodes_per_100_plays']:.1f} per 100 eligible player-plays.")
    prim = profile.filter(pl.col("is_primary") & (pl.col("position_group") == ALL) & pl.col("athlete_share").is_not_null())
    for r in prim.iter_rows(named=True):
        out.append(f"{r['demand_id']} — {r['feature']}: athlete share of episode variance {r['athlete_share']:.2f} (the rest is play-to-play).")
    miss = availability.filter(pl.col("n_episodes_labelled") == 0)["context_var"].to_list()
    out.append("Context labels not held by the warehouse (physical description only): " + (", ".join(miss) or "none") + ".")
    return out


__all__ = ["ACCEL_DEMANDS", "ALL", "ATHLETE", "AVAILABILITY", "BANDS", "BAND_COLS", "CONTEXT", "CONTEXT_VARS", "CONTRACTS", "DISTRIBUTION", "DemandBuilder",
           "DemandConfig", "DemandSpec", "EPISODES", "EXAMPLES", "FREQUENCY", "PROFILE", "RAW_SCHEMA", "REGISTRY", "UNITS", "athlete_table", "attach_context",
           "availability_table", "band_accel", "band_col", "batch_episodes", "build_tables", "choose_examples", "conclusions", "context_table",
           "crossing_time", "demand_specs", "distribution_table", "frequency_table", "profile_table", "registry_table", "runs", "segment_trace",
           "variance_components"]
