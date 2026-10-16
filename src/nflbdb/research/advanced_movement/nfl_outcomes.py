"""NFL movement outcomes (Milestone 05): tracking-derived play measures and athlete-level candidate outcomes.

No Combine value is read here. The only athlete attributes used are the position group and the draft cohort
(``gold.dim_athlete``), for stratification. ``assert_no_combine_columns`` guards every table that reaches the freeze.

Units of analysis
    frame          one 100 ms sample — never a sample size.
    start frame    a post-snap frame from which a fixed window (0.5 s, 1.0 s, 5 yd) fits inside its tracking segment.
                   Overlapping start frames of one play are strongly dependent: they are reduced to ONE row per play.
    window         one eligible player-play post-snap window on which an outcome is defined = "eligible window".
    athlete-game   all eligible windows of an athlete in one game — the unit of the reliability splits.
    athlete        the independent unit of every interval and count.

What is reused (nothing of the warehouse is rebuilt)
    silver.game_tracking            player frames, read in contiguous game batches (``KinematicFeatureBuilder``)
    gold/phase_03/nfl_play_anchors  snap timestamp per play (executed Phase 03 M1.1 windows build)
    gold/phase_03/nfl_player_play_windows   side, alignment, penalty flag per player-play (window ``post_snap_all``)
    gold/phase_03/nfl_trajectory_quality    quality tier per player-play
    features.trajectory / features.kinematics   segmentation and Savitzky–Golay kinematics (unchanged)

What is new
    Play measures over fixed post-snap windows (per play, per smoothing variant), their athlete-level statistics
    (``OutcomeEngine``), the outcome dictionary, coverage and the context cells.

Play measures (yd, s, yd/s, yd/s²; all from derived kinematics, never from vendor ``s`` / ``a`` / ``player_get_off``)
    speed_max            largest single-frame speed (REFERENCE only: one frame)
    speed_p90            90th percentile of the play's post-snap frame speeds
    speed_sust<w>        largest mean speed over a w-second window
    accel_sust<w>        largest mean of a_tan = d|v|/dt over a w-second window (second derivative of position)
    gain<w>              largest speed CHANGE over w seconds: speed(t + w) − speed(t) (first derivative only);
                         gain<w>_v0 = speed at the start of that window
    gain_d               largest speed change over the next ``gain_distance_yd`` of smoothed path
    band_acc_<band>      mean of gain<w0> / w0 over start frames whose speed is in the band and whose gain is positive
    band_gain_<band>     largest gain<w0> among start frames whose speed is in the band (envelope point)
    decel_sust           largest mean of −a_tan over w0; reaccel_gain = largest gain<w0> starting within
                         ``reaccel_search_s`` after that braking window (only when the braking is marked valid)

``accel_sust`` and ``gain`` measure nearly the same physical quantity by construction (the mean of d|v|/dt over a
window equals the speed change divided by the window); they differ in how much filter noise they carry.
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.tables import LayerRefs
from ...data.utils import get_logger, stable_hash
from ...features.config import KinematicsConfig, SmoothingSpec
from ...features.kinematics import compute_kinematics
from ...features.play_windows import PLAY_KEYS, PP_KEYS
from ...features.trajectory import NFL, prepare_frames
from ..artifacts import ArtifactStore
from ..contracts import TableContract
from ..kinematic_build import KinematicFeatureBuilder
from ..studies.stats import spearman
from .config import NotebookControls

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
OUTCOMES_VERSION = "1"
ALL = "ALL"
PRIMARY = "primary"
CONTEXT_VARS = ("lined_up_position", "play_class", "side", "down")
FORBIDDEN_PREFIXES = ("cmb_", "trk_", "fpca", "tcn", "official_", "forty", "combine_")
ROLES = ("candidate", "component", "reference", "exploratory")


class CombineLeakError(RuntimeError):
    """A Combine-derived column reached the NFL outcome tables."""


def assert_no_combine_columns(df: pl.DataFrame, what: str = "table") -> None:
    bad = [c for c in df.columns if c.lower().startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise CombineLeakError(f"{what} carries Combine-derived columns {bad}: NFL targets are defined and selected "
                               "without any Combine measurement (safeguard B of the Milestone 05 brief)")


# ============================================================================== configuration
@dataclass(frozen=True)
class OutcomeConfig:
    """Provenance: established = reused project value · proposed = fixed before any reliability is read ·
    heuristic = reference line of the screening, never a sole decision rule · execution = cost knob."""

    version: str = OUTCOMES_VERSION
    seed: int = 20261011
    # population ------------------------------------------------------------------------------------------
    season_types: tuple[str, ...] = ("REG",)                    # established: PlayWindowConfig.season_types
    positions: tuple[str, ...] = ("DB", "DL", "OL", "TE", "WR")  # established: Phase 05 P0 tracked position groups
    usable_tiers: tuple[str, ...] = ("A", "B")                  # established: MovementConfig.usable_tiers
    min_post_frames: int = 15                                   # proposed: 1.5 s of valid post-snap frames
    # windows ---------------------------------------------------------------------------------------------
    sustain_s: tuple[float, ...] = (0.5, 1.0)                   # established 0.5 s (sustain_window_s); proposed 1.0 s
    gain_distance_yd: float = 5.0                               # proposed
    speed_bands: tuple[tuple[float, float], ...] = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0))   # proposed
    min_band_starts: int = 3                                    # proposed: start frames in a band for a play value
    brake_threshold: float = 2.5                                # established: KinematicsConfig default braking rule
    brake_min_speed: float = 2.0                                # established: same rule (speed before onset)
    reaccel_search_s: float = 1.5                               # proposed
    exclude_boundary: bool = True                               # proposed: windows touching filter-edge frames dropped
    frame_tolerance_s: float = 0.05                             # established: PlayWindowConfig.frame_tolerance_s
    # smoothing variants ----------------------------------------------------------------------------------
    primary_smoothing: tuple[int, int] = (7, 2)                 # established: KinematicsConfig v1
    sensitivity_smoothing: tuple[tuple[int, int], ...] = ((5, 2), (9, 2), (11, 3))   # established sensitivity set
    sensitivity_game_stride: int = 2                            # execution: alternates on every k-th game of a batch
    # phase-space histogram (figures only) ----------------------------------------------------------------
    hist_speed_step: float = 0.5
    hist_speed_max: float = 13.0
    hist_accel_step: float = 0.5
    hist_accel_lim: float = 8.0
    # context ---------------------------------------------------------------------------------------------
    min_cell_windows: int = 30                                  # proposed: smaller context cells are not adjusted
    # athlete level ---------------------------------------------------------------------------------------
    min_windows_grid: tuple[int, ...] = (20, 50, 100, 200)      # established: RescueConfig / AdvancedConfig grids
    primary_min_windows: int = 50                               # proposed
    min_half_windows: int = 5                                   # proposed: windows in each half of a split
    first_n_grid: tuple[int, ...] = (20, 50, 100, 200)          # brief §6
    first_games_grid: tuple[int, ...] = (2, 4, 8, 16)           # proposed
    holdout_min_windows: int = 50                               # proposed: later-games reference of a first-N estimate
    min_game_windows: int = 3                                   # proposed: windows for an athlete-game mean
    n_boot: int = 1000
    n_game_splits: int = 100
    min_athletes: int = 20                                      # proposed: fewer → no estimate
    min_athletes_position: int = 15                             # proposed: per position group
    # screening reference lines (heuristic) ---------------------------------------------------------------
    ref_reliability: float = 0.70
    ref_reliability_lower: float = 0.50
    ref_smoothing_rank: float = 0.90
    ref_exposure_rho: float = 0.30
    ref_context_share: float = 0.25
    ref_positions: int = 3
    ref_redundancy: float = 0.90
    # smoke -----------------------------------------------------------------------------------------------
    smoke_games: int = 48                                       # execution: first games per season in smoke mode

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def for_mode(self, controls: NotebookControls) -> "OutcomeConfig":
        """Smoke mode keeps every measure definition; it shortens resampling and lowers exposure grids so the few
        games it reads can exercise every code path. Smoke numbers are wiring checks, never results."""
        if not controls.smoke:
            return replace(self, seed=controls.seed)
        return replace(self, seed=controls.seed, sensitivity_smoothing=self.sensitivity_smoothing[:1], n_boot=100,
                       n_game_splits=10, min_windows_grid=(5, 10, 20), primary_min_windows=10, min_half_windows=2,
                       first_n_grid=(5, 10, 20), first_games_grid=(1, 2), holdout_min_windows=5, min_game_windows=2,
                       min_athletes=6, min_athletes_position=3, min_cell_windows=10)

    def smoothing_specs(self) -> list[SmoothingSpec]:
        return [SmoothingSpec(*self.primary_smoothing), *[SmoothingSpec(*s) for s in self.sensitivity_smoothing]]


DEFINITION_KEYS = ("version", "season_types", "positions", "usable_tiers", "min_post_frames", "sustain_s", "gain_distance_yd",
                   "speed_bands", "min_band_starts", "brake_threshold", "brake_min_speed", "reaccel_search_s", "exclude_boundary",
                   "frame_tolerance_s", "primary_smoothing", "sensitivity_smoothing", "sensitivity_game_stride")


def definition_hash(c: OutcomeConfig) -> str:
    """Hash of the values that change a play measure or the eligible windows (resampling budgets and reference lines
    are excluded). Notebook 05B refuses play tables built under another definition."""
    return stable_hash({k: getattr(c, k) for k in DEFINITION_KEYS})


def wlab(w: float) -> str:
    return f"{int(round(w * 10)):02d}"


def blab(lo: float, hi: float) -> str:
    return f"v{lo:g}_{hi:g}".replace(".", "p")


def measure_columns(c: OutcomeConfig) -> list[str]:
    """Float play measures, in table order."""
    cols = ["speed_max", "speed_p90", "speed_mean"]
    for w in c.sustain_s:
        cols += [f"speed_sust{wlab(w)}", f"accel_sust{wlab(w)}", f"gain{wlab(w)}", f"gain{wlab(w)}_v0"]
    cols += ["decel_sust", "brake_v0", "reaccel_gain", "gain_d", "gain_d_time"]
    for lo, hi in c.speed_bands:
        cols += [f"band_gain_{blab(lo, hi)}", f"band_acc_{blab(lo, hi)}"]
    return cols


def count_columns(c: OutcomeConfig) -> list[str]:
    return ["n_post_frames", *[f"band_n_{blab(lo, hi)}" for lo, hi in c.speed_bands],
            *[f"band_npos_{blab(lo, hi)}" for lo, hi in c.speed_bands]]


# ============================================================================== contracts
def plays_contract(c: OutcomeConfig) -> TableContract:
    return TableContract(
        "nfl_play_movement", "Post-snap movement measures per tracked player-play (primary smoothing).",
        {"game_id": I, "play_id": I, "nfl_id": I, "season": I, "week": I, "position_group": S, "cohort": I, "side": S,
         "lined_up_position": S, "play_class": S, "pass_result": S, "down": I, "quality_tier": S, "in_player_play": B,
         "eligible": B, "exclusion_reason": S, "game_order": I, "play_order": I, "post_s": F, "boundary_frac": F,
         "brake_valid": B, **{k: I for k in count_columns(c)}, **{k: F for k in measure_columns(c)}},
        key=tuple(PP_KEYS), kind="dataset", area="NFL player-play (post-snap window)",
        docs={"eligible": "in-scope position, not nullified by penalty, usable quality tier, enough post-snap frames",
              "play_class": "grouping of the vendor pass_result code: pass (C / I / IN), sack_or_scramble (S / R), "
                            "no_pass (empty) — the meaning of the empty code is INFERRED, not confirmed",
              "game_order": "1 = the athlete's first game with an eligible window (season, week, game_id)",
              "play_order": "1 = the athlete's first eligible window in time",
              "boundary_frac": "share of post-snap frames inside the filter boundary of their segment"})


def variants_contract(c: OutcomeConfig) -> TableContract:
    return TableContract(
        "nfl_play_movement_variants", "Play measures under alternative smoothing or boundary handling (subset of games).",
        {"variant": S, "game_id": I, "play_id": I, "nfl_id": I, **{k: pl.Float32() for k in measure_columns(c)}},
        key=("variant", *PP_KEYS), kind="dataset", area="variant × NFL player-play",
        docs={"variant": "sg<window>p<order> = alternative Savitzky–Golay filter · boundary_incl = primary filter, "
                         "windows touching filter-edge frames kept"})


PHASE_SPACE = TableContract(
    "nfl_phase_space", "Post-snap frames by position group × speed bin × tangential-acceleration bin (primary smoothing).",
    {"position_group": S, "speed_bin": I, "accel_bin": I, "speed_lo": F, "accel_lo": F, "n_frames": I},
    key=("position_group", "speed_bin", "accel_bin"), kind="dataset", area="position group × speed bin × acceleration bin",
    docs={"n_frames": "frames, NOT independent observations; accelerations beyond ± hist_accel_lim are in the end bins"})
FUNNEL = TableContract(
    "nfl_window_funnel", "Player-plays, athletes and athlete-games remaining after each eligibility rule.",
    {"stage": S, "stage_order": I, "position_group": S, "n_player_plays": I, "n_athletes": I, "n_athlete_games": I},
    key=("stage", "position_group"), area="eligibility stage × position group")
DICTIONARY = TableContract(
    "nfl_outcome_dictionary", "Definition of every candidate NFL movement outcome.",
    {"outcome_id": S, "family": S, "role": S, "label": S, "unit": S, "play_measure": S, "athlete_statistic": S,
     "formula": S, "window_definition": S, "starting_speed_range": S, "applicable_context": S, "positions": S,
     "min_exposure": S, "quality_requirements": S, "smoothing": S, "direction_note": S, "interpretation_limit": S},
    key=("outcome_id",), area="outcome")
ATHLETE_OUTCOMES = TableContract(
    "nfl_athlete_outcomes", "Athlete-level value of every candidate outcome on all eligible windows.",
    {"nfl_id": I, "outcome_id": S, "position_group": S, "cohort": I, "n_windows": I, "n_games": I, "value": F,
     "value_adj": F}, key=("nfl_id", "outcome_id"), kind="dataset", area="athlete × outcome",
    docs={"value_adj": "same statistic after removing context-cell means of ALL athletes (descriptive; a predictive "
                       "evaluation must rebuild the cell means on training athletes)",
          "n_windows": "eligible windows on which the outcome is defined (composites: smallest component count)"})
ATHLETE_GAMES = TableContract(
    "nfl_athlete_game_outcomes", "Athlete-game mean and maximum of each play measure used by an outcome.",
    {"nfl_id": I, "game_id": I, "season": I, "game_order": I, "outcome_id": S, "n_windows": I, "mean": F, "max": F},
    key=("nfl_id", "game_id", "outcome_id"), kind="dataset", area="athlete × game × outcome")


def coverage_contract(c: OutcomeConfig) -> TableContract:
    return TableContract(
        "nfl_outcome_coverage", "Athletes, games and eligible windows per outcome and position group.",
        {"outcome_id": S, "position_group": S, "n_athletes_tracked": I, "n_athletes_any": I,
         **{f"n_athletes_ge_{t}": I for t in c.min_windows_grid}, "n_windows": I, "n_athlete_games": I,
         "median_windows": F, "median_games": F, "n_eligible_plays": I, "n_plays_undefined": I, "opportunity_rate": F,
         "rho_value_log_windows": F, "n_rho": I},
        key=("outcome_id", "position_group"), area="outcome × position group",
        docs={"n_athletes_tracked": "denominator: athletes of the group with ≥ 1 eligible player-play",
              "n_plays_undefined": "eligible player-plays on which the outcome is not defined (excluded observations)",
              "opportunity_rate": "eligible windows of the outcome / eligible player-plays (frequency of opportunity)",
              "rho_value_log_windows": "Spearman of athlete value with log window count, athletes at the primary "
                                       "exposure threshold: a value that follows usage, not movement, shows here"})


# ============================================================================== frame windows (pure, NumPy)
def _window_mean(x: np.ndarray, idx: np.ndarray, j: np.ndarray, k: int, ok: np.ndarray) -> np.ndarray:
    """Mean of x over frames i … i + k (k + 1 samples spanning k intervals); NaN when a sample is missing."""
    bad = ~np.isfinite(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(bad, 0.0, x))])
    cb = np.concatenate([[0], np.cumsum(bad)])
    full = ok & ((cb[j + 1] - cb[idx]) == 0)
    return np.where(full, (cs[j + 1] - cs[idx]) / (k + 1), np.nan)


def start_schema(c: OutcomeConfig) -> dict[str, pl.DataType]:
    sch: dict[str, pl.DataType] = {"game_id": I, "play_id": I, "nfl_id": I, "seg_idx": I, "t_rel": F, "v0": F, "a0": F,
                                   "boundary": B}
    for w in c.sustain_s:
        sch |= {f"sp{wlab(w)}": F, f"acc{wlab(w)}": F, f"gain{wlab(w)}": F, f"edge{wlab(w)}": B}
    return sch | {"gain_d": F, "time_d": F, "edge_d": B}


def start_table(kin: pl.DataFrame, snaps: pl.DataFrame, c: OutcomeConfig, half_window: int, dt: float) -> pl.DataFrame:
    """One row per post-snap frame with a defined speed, carrying the fixed windows that START on it.

    ``kin``: frame kinematics of whole player trajectories (``compute_kinematics`` output: PP_KEYS, seg_idx, pos,
    seg_len, ts, speed, a_tan, step_smooth). ``snaps``: (game_id, play_id, snap_ts). A window is defined only when it
    lies entirely inside one tracking segment; ``edge*`` marks windows that touch a frame within ``half_window`` of
    a segment edge (Savitzky–Golay boundary frames). Plays without a snap anchor give no row."""
    f = (kin.join(snaps.select(*PLAY_KEYS, "snap_ts").unique(subset=PLAY_KEYS), on=PLAY_KEYS, how="left")
         .filter(pl.col("snap_ts").is_not_null())
         .with_columns(((pl.col("ts") - pl.col("snap_ts")).dt.total_microseconds().cast(pl.Float64) / 1e6).alias("t_rel"))
         .sort([*PP_KEYS, "ts"]))
    n = f.height
    if not n:
        return pl.DataFrame(schema=start_schema(c))
    t_rel = f["t_rel"].to_numpy().astype(float)
    speed = f["speed"].cast(pl.Float64).fill_null(float("nan")).to_numpy().astype(float)
    a_tan = f["a_tan"].cast(pl.Float64).fill_null(float("nan")).to_numpy().astype(float)
    step = f["step_smooth"].cast(pl.Float64).fill_null(float("nan")).to_numpy().astype(float)
    pos = f["pos"].to_numpy().astype(np.int64)
    left = f["seg_len"].to_numpy().astype(np.int64) - 1 - pos            # frames after this one in its segment
    idx = np.arange(n)
    boundary = (pos < half_window) | (left < half_window)
    cols: dict[str, np.ndarray] = {"t_rel": t_rel, "v0": speed, "a0": a_tan}
    flags: dict[str, np.ndarray] = {"boundary": boundary}
    for w in c.sustain_s:
        k = int(round(w / dt))
        j, ok = np.minimum(idx + k, n - 1), left >= k
        cols[f"sp{wlab(w)}"] = _window_mean(speed, idx, j, k, ok)
        cols[f"acc{wlab(w)}"] = _window_mean(a_tan, idx, j, k, ok)
        cols[f"gain{wlab(w)}"] = np.where(ok, speed[j] - speed, np.nan)
        flags[f"edge{wlab(w)}"] = boundary | (left - k < half_window)
    cum = np.cumsum(np.where(np.isfinite(step), step, 0.0))               # non-decreasing over the whole batch
    jd = np.searchsorted(cum, cum + c.gain_distance_yd, side="left")
    span = jd - idx
    okd = span <= left                                                    # the distance is covered inside the segment
    jc = np.minimum(jd, n - 1)
    cols["gain_d"] = np.where(okd, speed[jc] - speed, np.nan)
    cols["time_d"] = np.where(okd, span * dt, np.nan)
    flags["edge_d"] = boundary | (left - span < half_window)
    out = (f.select(*[pl.col(k).cast(pl.Int64) for k in PP_KEYS], pl.col("seg_idx").cast(pl.Int64))
           .with_columns(*[pl.Series(k, v, dtype=pl.Float64, nan_to_null=True) for k, v in cols.items()],
                         *[pl.Series(k, v, dtype=pl.Boolean) for k, v in flags.items()]))
    post = np.isfinite(t_rel) & (t_rel >= -c.frame_tolerance_s) & np.isfinite(speed)
    return out.filter(pl.Series("_post", post, dtype=pl.Boolean)).select(list(start_schema(c)))


def measure_schema(c: OutcomeConfig) -> dict[str, pl.DataType]:
    return {"game_id": I, "play_id": I, "nfl_id": I, "post_s": F, "boundary_frac": F, "brake_valid": B,
            **{k: I for k in count_columns(c)}, **{k: F for k in measure_columns(c)}}


def play_table(starts: pl.DataFrame, c: OutcomeConfig, exclude_boundary: bool | None = None) -> pl.DataFrame:
    """One row per player-play from its start frames (see the module docstring for every measure)."""
    if starts.is_empty():
        return pl.DataFrame(schema=measure_schema(c))
    exclude = c.exclude_boundary if exclude_boundary is None else exclude_boundary
    labs = [wlab(w) for w in c.sustain_s]
    l0, w0 = labs[0], float(c.sustain_s[0])
    s = starts
    if exclude:
        s = s.with_columns(
            *[pl.when(pl.col(f"edge{lb}")).then(None).otherwise(pl.col(f"{x}{lb}")).alias(f"{x}{lb}")
              for lb in labs for x in ("sp", "acc", "gain")],
            *[pl.when(pl.col("edge_d")).then(None).otherwise(pl.col(x)).alias(x) for x in ("gain_d", "time_d")])
    g0, dec, gd = pl.col(f"gain{l0}"), -pl.col(f"acc{l0}"), pl.col("gain_d")
    aggs: list[pl.Expr] = [
        pl.len().cast(pl.Int64).alias("n_post_frames"), pl.col("t_rel").max().alias("post_s"),
        pl.col("boundary").cast(pl.Float64).mean().alias("boundary_frac"),
        pl.col("v0").max().alias("speed_max"), pl.col("v0").quantile(0.9, "linear").alias("speed_p90"),
        pl.col("v0").mean().alias("speed_mean")]
    for lb in labs:
        g = pl.col(f"gain{lb}")
        aggs += [pl.col(f"sp{lb}").max().alias(f"speed_sust{lb}"), pl.col(f"acc{lb}").max().alias(f"accel_sust{lb}"),
                 g.max().alias(f"gain{lb}"), pl.col("v0").filter(g == g.max()).first().alias(f"gain{lb}_v0")]
    aggs += [dec.max().alias("decel_sust"), pl.col("v0").filter(dec == dec.max()).first().alias("brake_v0"),
             pl.col("t_rel").filter(dec == dec.max()).first().alias("_t_brake"),
             gd.max().alias("gain_d"), pl.col("time_d").filter(gd == gd.max()).first().alias("gain_d_time")]
    for lo, hi in c.speed_bands:
        b = blab(lo, hi)
        m = (pl.col("v0") >= lo) & (pl.col("v0") < hi) & g0.is_not_null()
        up = m & (g0 > 0)
        aggs += [m.sum().cast(pl.Int64).alias(f"band_n_{b}"), up.sum().cast(pl.Int64).alias(f"band_npos_{b}"),
                 g0.filter(m).max().alias(f"band_gain_{b}"), (g0.filter(up).mean() / w0).alias(f"band_acc_{b}")]
    out = s.group_by(PP_KEYS).agg(aggs)
    after = (s.join(out.select(*PP_KEYS, "_t_brake"), on=PP_KEYS, how="inner")
             .filter((pl.col("t_rel") >= pl.col("_t_brake") + w0 - 1e-6)
                     & (pl.col("t_rel") <= pl.col("_t_brake") + w0 + c.reaccel_search_s + 1e-6))
             .group_by(PP_KEYS).agg(g0.max().alias("reaccel_gain")))
    out = out.join(after, on=PP_KEYS, how="left").with_columns(
        ((pl.col("decel_sust") >= c.brake_threshold) & (pl.col("brake_v0") >= c.brake_min_speed)).fill_null(False)
        .alias("brake_valid"))
    out = out.with_columns(
        pl.when(pl.col("brake_valid")).then(pl.col("reaccel_gain")).otherwise(None).alias("reaccel_gain"),
        *[pl.when(pl.col(f"band_n_{blab(lo, hi)}") >= c.min_band_starts).then(pl.col(f"band_gain_{blab(lo, hi)}"))
          .otherwise(None).alias(f"band_gain_{blab(lo, hi)}") for lo, hi in c.speed_bands],
        *[pl.when(pl.col(f"band_npos_{blab(lo, hi)}") >= c.min_band_starts).then(pl.col(f"band_acc_{blab(lo, hi)}"))
          .otherwise(None).alias(f"band_acc_{blab(lo, hi)}") for lo, hi in c.speed_bands])
    sch = measure_schema(c)
    return out.select([pl.col(k).cast(t) for k, t in sch.items()]).sort(PP_KEYS)


def phase_hist(starts: pl.DataFrame, athletes: pl.DataFrame, c: OutcomeConfig) -> pl.DataFrame:
    """Frame counts by position group × speed bin × a_tan bin (mergeable across batches)."""
    nb = int(round(c.hist_speed_max / c.hist_speed_step))
    lim = c.hist_accel_lim
    return (starts.filter(pl.col("a0").is_not_null())
            .join(athletes.select("nfl_id", "position_group"), on="nfl_id", how="inner")
            .with_columns((pl.col("v0") / c.hist_speed_step).floor().clip(0, nb - 1).cast(pl.Int64).alias("speed_bin"),
                          ((pl.col("a0").clip(-lim, lim - 1e-9) + lim) / c.hist_accel_step).floor().cast(pl.Int64)
                          .alias("accel_bin"))
            .group_by("position_group", "speed_bin", "accel_bin").agg(pl.len().cast(pl.Int64).alias("n_frames")))


def finish_hist(parts: Sequence[pl.DataFrame], c: OutcomeConfig) -> pl.DataFrame:
    if not parts:
        return PHASE_SPACE.empty()
    h = (pl.concat(list(parts), how="vertical").group_by("position_group", "speed_bin", "accel_bin")
         .agg(pl.col("n_frames").sum()))
    return (h.with_columns((pl.col("speed_bin") * c.hist_speed_step).cast(pl.Float64).alias("speed_lo"),
                           (pl.col("accel_bin") * c.hist_accel_step - c.hist_accel_lim).cast(pl.Float64).alias("accel_lo"))
            .select([pl.col(k).cast(t) for k, t in PHASE_SPACE.columns.items()])
            .sort("position_group", "speed_bin", "accel_bin"))


# ============================================================================== eligibility, order, funnel
STAGES = ("tracked_with_snap", "position_in_scope", "not_nullified_by_penalty", "usable_quality_tier", "eligible")


def assemble_plays(measures: pl.DataFrame, labels: pl.DataFrame, quality: pl.DataFrame, context: pl.DataFrame,
                   athletes: pl.DataFrame, games: pl.DataFrame, c: OutcomeConfig) -> pl.DataFrame:
    """Join play measures with the persisted labels, apply the eligibility rules IN ORDER (the first failing rule
    is the exclusion reason) and number each athlete's eligible windows and games chronologically.

    Nothing is imputed: a player-play without a quality record or with too few frames is excluded with a reason."""
    key = list(PP_KEYS)
    d = (measures
         .join(labels.select(*key, "side", "lined_up_position", "play_nullified_by_penalty", "in_player_play")
               .unique(subset=key), on=key, how="left")
         .join(quality.select(*key, "quality_tier").unique(subset=key), on=key, how="left")
         .join(context.select(*key, "down", "pass_result").unique(subset=key), on=key, how="left")
         .join(athletes.select("nfl_id", "position_group", "cohort").unique(subset="nfl_id"), on="nfl_id", how="left")
         .join(games.select("game_id", "season", "week").unique(subset="game_id"), on="game_id", how="left"))
    pr = pl.col("pass_result").cast(pl.String).str.strip_chars()
    d = d.with_columns(
        pl.col("in_player_play").fill_null(False), pl.col("side").fill_null("unknown"),
        pl.col("week").fill_null(0).cast(pl.Int64), pl.col("season").cast(pl.Int64), pl.col("cohort").cast(pl.Int64),
        pl.col("down").cast(pl.Int64),
        pl.when(pr.is_in(["C", "I", "IN"])).then(pl.lit("pass")).when(pr.is_in(["S", "R"])).then(pl.lit("sack_or_scramble"))
        .when(pr.is_null() | (pr == "")).then(pl.lit("no_pass")).otherwise(pl.lit("other")).alias("play_class"))
    reason = (pl.when(pl.col("position_group").is_null() | ~pl.col("position_group").is_in(list(c.positions)))
              .then(pl.lit("position_out_of_scope"))
              .when(pl.col("play_nullified_by_penalty").cast(pl.String) == "Y").then(pl.lit("nullified_by_penalty"))
              .when(pl.col("quality_tier").is_null()).then(pl.lit("no_quality_record"))
              .when(~pl.col("quality_tier").is_in(list(c.usable_tiers))).then(pl.lit("quality_tier_not_usable"))
              .when(pl.col("n_post_frames") < c.min_post_frames).then(pl.lit("short_post_snap_window"))
              .otherwise(pl.lit("")))
    d = d.with_columns(reason.alias("exclusion_reason")).with_columns((pl.col("exclusion_reason") == "").alias("eligible"))
    order = ["nfl_id", "season", "week", "game_id", "play_id"]
    el = (d.filter(pl.col("eligible")).sort(order)
          .with_columns((pl.int_range(pl.len(), dtype=pl.Int64).over("nfl_id") + 1).alias("play_order"),
                        (pl.col("game_id") != pl.col("game_id").shift(1).over("nfl_id")).fill_null(True)
                        .cast(pl.Int64).cum_sum().over("nfl_id").alias("game_order"))
          .select(*key, "play_order", "game_order"))
    d = d.join(el, on=key, how="left")
    contract = plays_contract(c)
    return d.select([pl.col(k).cast(t) for k, t in contract.columns.items()]).sort(key)


def funnel_table(plays: pl.DataFrame, c: OutcomeConfig) -> pl.DataFrame:
    scope = pl.col("position_group").is_not_null() & pl.col("position_group").is_in(list(c.positions))
    r = pl.col("exclusion_reason")
    masks = {"tracked_with_snap": pl.lit(True), "position_in_scope": scope,
             "not_nullified_by_penalty": scope & (r != "nullified_by_penalty"),
             "usable_quality_tier": scope & ~r.is_in(["nullified_by_penalty", "no_quality_record", "quality_tier_not_usable"]),
             "eligible": pl.col("eligible")}
    rows = []
    for i, st in enumerate(STAGES):
        d = plays.filter(masks[st]).with_columns(pl.col("position_group").fill_null("UNKNOWN"))
        for g, sub in [(ALL, d), *[(p, d.filter(pl.col("position_group") == p)) for p in sorted(d["position_group"].unique().to_list())]]:
            rows.append({"stage": st, "stage_order": i, "position_group": g, "n_player_plays": sub.height,
                         "n_athletes": sub["nfl_id"].n_unique() if sub.height else 0,
                         "n_athlete_games": sub.select(pl.struct("nfl_id", "game_id").n_unique()).item() if sub.height else 0})
    return pl.DataFrame(rows, schema=FUNNEL.columns)


# ============================================================================== outcome registry
@dataclass(frozen=True)
class OutcomeSpec:
    outcome_id: str
    family: str
    role: str                       # candidate · component · reference · exploratory
    label: str
    column: str                     # play measure ("" for a composite)
    stat: str                       # mean · median · p90 · p95 over the athlete's eligible windows
    unit: str
    formula: str
    window: str
    start_speed: str = "any"
    components: tuple[str, ...] = ()
    limit: str = ""


def _stat(e: pl.Expr, stat: str) -> pl.Expr:
    if stat == "mean":
        return e.mean()
    if stat == "median":
        return e.median()
    if stat in ("p90", "p95"):
        return e.quantile(int(stat[1:]) / 100.0, "linear")
    raise ValueError(f"unknown athlete statistic {stat!r}")


def outcome_specs(c: OutcomeConfig) -> list[OutcomeSpec]:
    labs = [(wlab(w), w) for w in c.sustain_s]
    l0, w0 = labs[0]
    usage = "observed movement depends on role and opportunity; it is not a physical maximum"
    out: list[OutcomeSpec] = []
    for lb, w in labs:
        out.append(OutcomeSpec(f"O1.speed_sust{lb}_p90", "O1", "candidate", f"sustained speed, {w:g} s (p90 of windows)",
                               f"speed_sust{lb}", "p90", "yd/s",
                               f"p90 over eligible windows of max_t mean(speed[t, t + {w:g} s])", f"{w:g} s, post-snap", limit=usage))
    out.append(OutcomeSpec("O1.speed_p90_mean", "O1", "candidate", "play p90 speed (mean of windows)", "speed_p90", "mean",
                           "yd/s", "mean over eligible windows of the 90th percentile of post-snap frame speeds",
                           "whole post-snap window", limit=usage + "; follows play length and assignment"))
    out.append(OutcomeSpec("O1.speed_max_p95", "O1", "reference", "single-frame peak speed (p95 of windows)", "speed_max",
                           "p95", "yd/s", "p95 over eligible windows of max_t speed[t]", "one frame, post-snap",
                           limit="single-frame maximum: shown for comparison, never a target"))
    for lb, w in labs:
        out.append(OutcomeSpec(f"O2.accel_sust{lb}_p90", "O2", "candidate", f"sustained acceleration, {w:g} s (p90 of windows)",
                               f"accel_sust{lb}", "p90", "yd/s²",
                               f"p90 over eligible windows of max_t mean(a_tan[t, t + {w:g} s]), a_tan = d|v|/dt",
                               f"{w:g} s, post-snap", limit=usage + "; second derivative: most sensitive to the filter"))
    out.append(OutcomeSpec(f"O2.accel_sust{l0}_mean", "O2", "candidate", f"sustained acceleration, {w0:g} s (mean of windows)",
                           f"accel_sust{l0}", "mean", "yd/s²", f"mean over eligible windows of max_t mean(a_tan[t, t + {w0:g} s])",
                           f"{w0:g} s, post-snap", limit=usage))
    for lo, hi in c.speed_bands:
        b = blab(lo, hi)
        out.append(OutcomeSpec(f"O3.band_acc_{b}_mean", "O3", "candidate", f"acceleration from {lo:g}–{hi:g} yd/s (mean of windows)",
                               f"band_acc_{b}", "mean", "yd/s²",
                               f"mean over eligible windows of mean_t [(speed(t + {w0:g} s) − speed(t)) / {w0:g} s] over start "
                               f"frames with speed in [{lo:g}, {hi:g}) and a positive change (≥ {c.min_band_starts} such frames)",
                               f"{w0:g} s, post-snap", f"[{lo:g}, {hi:g}) yd/s",
                               limit=usage + "; conditioning on a positive change selects accelerating moments"))
    for lb, w in labs:
        out.append(OutcomeSpec(f"O4.gain{lb}_p90", "O4", "candidate", f"speed gain over {w:g} s (p90 of windows)", f"gain{lb}",
                               "p90", "yd/s", f"p90 over eligible windows of max_t [speed(t + {w:g} s) − speed(t)]",
                               f"{w:g} s, post-snap", limit=usage + "; an average speed change, not an instantaneous acceleration"))
    out.append(OutcomeSpec("O4.gain_d_p90", "O4", "candidate", f"speed gain over {c.gain_distance_yd:g} yd (p90 of windows)",
                           "gain_d", "p90", "yd/s",
                           f"p90 over eligible windows of max_t [speed at path(t) + {c.gain_distance_yd:g} yd − speed(t)]",
                           f"{c.gain_distance_yd:g} yd of smoothed path, post-snap", limit=usage))
    comp = []
    for lo, hi in c.speed_bands:
        b = blab(lo, hi)
        comp.append(f"O5.band_gain_{b}_p90")
        out.append(OutcomeSpec(comp[-1], "O5", "component", f"envelope point {lo:g}–{hi:g} yd/s (p90 of windows)",
                               f"band_gain_{b}", "p90", "yd/s",
                               f"p90 over eligible windows of max_t [speed(t + {w0:g} s) − speed(t)] over start frames with "
                               f"speed in [{lo:g}, {hi:g}) (≥ {c.min_band_starts} such frames)", f"{w0:g} s, post-snap",
                               f"[{lo:g}, {hi:g}) yd/s", limit=usage))
    out.append(OutcomeSpec("O5.envelope_level", "O5", "candidate", "speed–acceleration envelope level", "", "mean", "yd/s",
                           "mean of the envelope points over all speed bands (an athlete lacking a band has no value)",
                           f"{w0:g} s, post-snap", "all bands", tuple(comp),
                           limit="observed conditional distribution, not the athlete's maximum capability; bands that a "
                                 "position rarely reaches remove its athletes"))
    out.append(OutcomeSpec("O6.decel_sust_p90", "O6", "exploratory", f"sustained braking, {w0:g} s (p90 of windows)", "decel_sust",
                           "p90", "yd/s²", f"p90 over eligible windows of max_t mean(−a_tan[t, t + {w0:g} s])",
                           f"{w0:g} s, post-snap", limit="exploratory: contact and play end produce decelerations that are not braking"))
    out.append(OutcomeSpec("O6.reaccel_gain_mean", "O6", "exploratory", "re-acceleration after braking (mean of windows)",
                           "reaccel_gain", "mean", "yd/s",
                           f"mean over windows with a valid braking (mean −a_tan ≥ {c.brake_threshold:g} yd/s² from ≥ "
                           f"{c.brake_min_speed:g} yd/s) of the largest {w0:g} s speed gain starting within {c.reaccel_search_s:g} s after it",
                           f"{w0:g} s after a braking window", f"≥ {c.brake_min_speed:g} yd/s before braking",
                           limit="exploratory: no validated manoeuvre label; the braking rule is a provisional threshold"))
    return out


def outcome_dictionary(c: OutcomeConfig) -> pl.DataFrame:
    sm = f"Savitzky–Golay window {c.primary_smoothing[0]} frames, order {c.primary_smoothing[1]} (KinematicsConfig v1)"
    edge = "windows touching filter-edge frames excluded" if c.exclude_boundary else "filter-edge frames kept"
    rows = [{"outcome_id": s.outcome_id, "family": s.family, "role": s.role, "label": s.label, "unit": s.unit,
             "play_measure": s.column or " + ".join(s.components), "athlete_statistic": s.stat, "formula": s.formula,
             "window_definition": s.window, "starting_speed_range": s.start_speed,
             "applicable_context": f"season types {', '.join(c.season_types)}; post-snap frames of plays with a snap tag; "
                                   "context cell = position group × alignment × play class (adjusted variant)",
             "positions": ", ".join(c.positions),
             "min_exposure": f"≥ {c.primary_min_windows} eligible windows (primary threshold; grid {list(c.min_windows_grid)})",
             "quality_requirements": f"quality tier in {list(c.usable_tiers)}; ≥ {c.min_post_frames} valid post-snap frames; "
                                     f"not nullified by penalty; {edge}",
             "smoothing": sm, "direction_note": "larger = more of the measured movement was observed; no claim that larger is better",
             "interpretation_limit": s.limit} for s in outcome_specs(c)]
    return pl.DataFrame(rows, schema=DICTIONARY.columns)


# ============================================================================== athlete-level engine
class OutcomeEngine:
    """Athlete-level values of the candidate outcomes from the play table, for any subset or split of windows.

    ``frame(spec)``: the eligible windows on which the outcome is defined, with ``y`` (play measure), ``r``
    (y − context-cell mean + position-group mean; cells with < ``min_cell_windows`` windows are left unadjusted),
    ``w_order`` (1 = first such window of the athlete in time) and ``g_par`` (parity of the athlete's game order,
    common to every outcome so that composite halves never share a game)."""

    def __init__(self, plays: pl.DataFrame, c: OutcomeConfig, specs: Sequence[OutcomeSpec] | None = None):
        assert_no_combine_columns(plays, "play table")
        self.c = c
        self.plays = plays.filter(pl.col("eligible"))
        self.specs = list(specs) if specs is not None else outcome_specs(c)
        self.by_id = {s.outcome_id: s for s in self.specs}
        self.athletes = (self.plays.select("nfl_id", "position_group", "cohort").unique(subset="nfl_id").sort("nfl_id"))
        self._frames: dict[str, pl.DataFrame] = {}

    def base_specs(self) -> list[OutcomeSpec]:
        return [s for s in self.specs if not s.components]

    def frame(self, spec: OutcomeSpec) -> pl.DataFrame:
        if spec.components:
            raise ValueError(f"{spec.outcome_id} is a composite: it has no play measure of its own")
        if spec.outcome_id not in self._frames:
            cell = pl.concat_str([pl.col("position_group"), pl.col("lined_up_position").fill_null("NA"), pl.col("play_class")],
                                 separator="|")
            d = (self.plays.filter(pl.col(spec.column).is_not_null())
                 .select(*PP_KEYS, "season", "position_group", "game_order", "play_order", "quality_tier", "in_player_play",
                         "n_post_frames", "boundary_frac", *CONTEXT_VARS, pl.col(spec.column).cast(pl.Float64).alias("y"))
                 .sort("nfl_id", "play_order")
                 .with_columns(cell.alias("cell"), (pl.int_range(pl.len(), dtype=pl.Int64).over("nfl_id") + 1).alias("w_order"),
                               (pl.col("game_order") % 2).alias("g_par"))
                 .with_columns(pl.len().over("cell").alias("_nc"), pl.col("y").mean().over("cell").alias("_mc"),
                               pl.col("y").mean().over("position_group").alias("_mp"))
                 .with_columns(pl.when(pl.col("_nc") >= self.c.min_cell_windows).then(pl.col("y") - pl.col("_mc") + pl.col("_mp"))
                               .otherwise(pl.col("y")).alias("r"), (pl.col("_nc") >= self.c.min_cell_windows).alias("cell_adjusted"))
                 .drop("_nc", "_mc", "_mp"))
            self._frames[spec.outcome_id] = d
        return self._frames[spec.outcome_id]

    def values(self, spec: OutcomeSpec, by: Sequence[str] = ("nfl_id",), where: pl.Expr | None = None,
               assign: pl.DataFrame | None = None) -> pl.DataFrame:
        """[*by, n, n_games, value, value_adj]. ``where`` filters windows; ``assign`` (nfl_id, game_id, extra columns)
        is inner-joined so a game-level split can be one of the ``by`` columns. A composite needs every component."""
        by = list(by)
        if spec.components:
            parts = [self.values(self.by_id[k], by, where, assign) for k in spec.components]
            out = parts[0].rename({x: f"{x}_0" for x in ("n", "n_games", "value", "value_adj")})
            for i, p in enumerate(parts[1:], 1):
                out = out.join(p.rename({x: f"{x}_{i}" for x in ("n", "n_games", "value", "value_adj")}), on=by, how="inner")
            k = range(len(parts))
            return out.select(*by, pl.min_horizontal([pl.col(f"n_{i}") for i in k]).alias("n"),
                              pl.min_horizontal([pl.col(f"n_games_{i}") for i in k]).alias("n_games"),
                              pl.mean_horizontal([pl.col(f"value_{i}") for i in k]).alias("value"),
                              pl.mean_horizontal([pl.col(f"value_adj_{i}") for i in k]).alias("value_adj"))
        d = self.frame(spec)
        if where is not None:
            d = d.filter(where)
        if assign is not None:
            d = d.join(assign, on=["nfl_id", "game_id"], how="inner")
        return d.group_by(by).agg(pl.len().cast(pl.Int64).alias("n"), pl.col("game_id").n_unique().cast(pl.Int64).alias("n_games"),
                                  _stat(pl.col("y"), spec.stat).alias("value"), _stat(pl.col("r"), spec.stat).alias("value_adj"))

    def pairs(self, spec: OutcomeSpec, half: str = "g_par", where: pl.Expr | None = None,
              assign: pl.DataFrame | None = None) -> pl.DataFrame:
        """One row per athlete with both halves: n_0, value_0, value_adj_0, n_1, value_1, value_adj_1, position_group.
        ``half`` is a 0 / 1 column of the frame (``g_par``) or of ``assign``. Halves are sets of whole games."""
        v = self.values(spec, ("nfl_id", half), where, assign)
        cols = ("n", "n_games", "value", "value_adj")
        a = v.filter(pl.col(half) == 0).drop(half).rename({x: f"{x}_0" for x in cols})
        b = v.filter(pl.col(half) == 1).drop(half).rename({x: f"{x}_1" for x in cols})
        return a.join(b, on="nfl_id", how="inner").join(self.athletes, on="nfl_id", how="left").sort("nfl_id")

    def games(self) -> pl.DataFrame:
        """(nfl_id, game_id, game_order): every athlete-game with an eligible window, in time order."""
        return self.plays.select("nfl_id", "game_id", "game_order").unique().sort("nfl_id", "game_order")


def athlete_outcomes(eng: OutcomeEngine) -> pl.DataFrame:
    parts = [eng.values(s).join(eng.athletes, on="nfl_id", how="left")
             .select("nfl_id", pl.lit(s.outcome_id).alias("outcome_id"), "position_group", "cohort",
                     pl.col("n").alias("n_windows"), "n_games", "value", "value_adj") for s in eng.specs]
    out = pl.concat(parts, how="vertical") if parts else ATHLETE_OUTCOMES.empty()
    return out.select([pl.col(k).cast(t) for k, t in ATHLETE_OUTCOMES.columns.items()]).sort("nfl_id", "outcome_id")


def athlete_game_outcomes(eng: OutcomeEngine) -> pl.DataFrame:
    parts = [eng.frame(s).group_by("nfl_id", "game_id").agg(
        pl.col("season").first(), pl.col("game_order").first(), pl.len().cast(pl.Int64).alias("n_windows"),
        pl.col("y").mean().alias("mean"), pl.col("y").max().alias("max")).with_columns(pl.lit(s.outcome_id).alias("outcome_id"))
        for s in eng.base_specs()]
    out = pl.concat(parts, how="vertical") if parts else ATHLETE_GAMES.empty()
    return out.select([pl.col(k).cast(t) for k, t in ATHLETE_GAMES.columns.items()]).sort("nfl_id", "game_id", "outcome_id")


def coverage_table(eng: OutcomeEngine, outcomes: pl.DataFrame) -> pl.DataFrame:
    """Coverage by outcome × position group (+ ALL). Counts of athletes, never of frames."""
    c, contract = eng.c, coverage_contract(eng.c)
    tracked = dict(eng.athletes.group_by("position_group").len().iter_rows())
    plays_n = dict(eng.plays.group_by("position_group").len().iter_rows())
    rows = []
    for s in eng.specs:
        o = outcomes.filter(pl.col("outcome_id") == s.outcome_id)
        for g in [ALL, *sorted(tracked)]:
            sub = o if g == ALL else o.filter(pl.col("position_group") == g)
            n_tr = sum(tracked.values()) if g == ALL else tracked[g]
            n_pl = sum(plays_n.values()) if g == ALL else plays_n.get(g, 0)
            big = sub.filter(pl.col("n_windows") >= c.primary_min_windows)
            rho = float("nan")
            if big.height >= c.min_athletes_position:
                v, lw = big["value"].to_numpy().astype(float), np.log(big["n_windows"].to_numpy().astype(float))
                if g == ALL:                                   # within position: remove the group means first
                    codes = np.unique(big["position_group"].to_numpy(), return_inverse=True)[1]
                    v, lw = v - _group_mean(v, codes), lw - _group_mean(lw, codes)
                rho = spearman(v, lw)
            n_w = int(sub["n_windows"].sum()) if sub.height else 0
            rows.append({"outcome_id": s.outcome_id, "position_group": g, "n_athletes_tracked": n_tr, "n_athletes_any": sub.height,
                         **{f"n_athletes_ge_{t}": sub.filter(pl.col("n_windows") >= t).height for t in c.min_windows_grid},
                         "n_windows": n_w, "n_athlete_games": int(sub["n_games"].sum()) if sub.height else 0,
                         "median_windows": float(sub["n_windows"].median()) if sub.height else None,
                         "median_games": float(sub["n_games"].median()) if sub.height else None,
                         "n_eligible_plays": n_pl, "n_plays_undefined": None if s.components else n_pl - n_w,
                         "opportunity_rate": None if s.components or not n_pl else n_w / n_pl,
                         "rho_value_log_windows": rho if np.isfinite(rho) else None, "n_rho": big.height})
    return pl.DataFrame(rows, schema=contract.columns)


def _group_mean(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    k = np.bincount(codes).astype(float)
    return (np.bincount(codes, weights=x) / k)[codes]


def variant_engine(plays: pl.DataFrame, variants: pl.DataFrame, variant: str, c: OutcomeConfig) -> tuple[OutcomeEngine, OutcomeEngine]:
    """(primary engine, variant engine) on the SAME player-plays: those of the variant table. The eligibility and
    the time order of the primary table are kept, only the play measures change."""
    m = measure_columns(c)
    v = variants.filter(pl.col("variant") == variant).drop("variant").with_columns([pl.col(k).cast(pl.Float64) for k in m])
    matched = plays.join(v.select(PP_KEYS), on=PP_KEYS, how="inner")
    swapped = matched.drop(m).join(v, on=PP_KEYS, how="inner").select(plays.columns)
    return OutcomeEngine(matched, c), OutcomeEngine(swapped, c)


# ============================================================================== builder (reads the warehouse)
class NflOutcomeBuilder:
    """Scans silver.game_tracking once, in contiguous game batches, and reduces it to play measures.

    ``limit_games``: first N games per season (smoke). ``athlete_ids``: optional athlete subset (smoke)."""

    def __init__(self, cfg: Any, c: OutcomeConfig, limit_games: int | None = None, athlete_ids: Sequence[int] | None = None,
                 kcfg: KinematicsConfig | None = None):
        self.cfg, self.c = cfg, c
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        self.p03 = ArtifactStore(cfg, phase="phase_03")
        self.kin = KinematicFeatureBuilder(cfg, self.kcfg, store=self.p03, limit_games=limit_games)
        self.refs = LayerRefs(cfg)
        self.limit_games = limit_games
        self.athlete_ids = None if athlete_ids is None else sorted(int(i) for i in athlete_ids)

    # ------------------------------------------------------------------ persisted inputs
    def _p03(self, name: str) -> pl.LazyFrame:
        if not self.p03.dataset_files(name):
            raise FileNotFoundError(f"gold/phase_03/{name} is missing. It is written by the Phase 03 build "
                                    "(`nflbdb.research.nfl_windows` / `kinematic_build`, executed 2026-10-09); Milestone 05 "
                                    "reuses it and does not rebuild it.")
        return self.p03.load(name)

    def anchors(self) -> pl.DataFrame:
        return (self._p03("nfl_play_anchors")
                .filter(pl.col("has_snap") & pl.col("season_type").is_in(list(self.c.season_types)))
                .select(*[pl.col(k).cast(pl.Int64) for k in PLAY_KEYS], "snap_ts").collect())

    def labels(self) -> pl.DataFrame:
        return (self._p03("nfl_player_play_windows").filter(pl.col("window") == "post_snap_all")
                .select(*[pl.col(k).cast(pl.Int64) for k in PP_KEYS], pl.col("side").cast(pl.String),
                        pl.col("lined_up_position").cast(pl.String), pl.col("play_nullified_by_penalty").cast(pl.String),
                        pl.col("in_player_play").cast(pl.Boolean)).collect())

    def quality(self) -> pl.DataFrame:
        return (self._p03("nfl_trajectory_quality")
                .select(*[pl.col(k).cast(pl.Int64) for k in PP_KEYS], pl.col("quality_tier").cast(pl.String)).collect())

    def athletes(self) -> pl.DataFrame:
        return (self.refs.scan("gold", "dim_athlete")
                .select(pl.col("nfl_id").cast(pl.Int64), pl.col("position_group").cast(pl.String),
                        pl.col("cohort").cast(pl.Int64)).collect().unique(subset="nfl_id"))

    def games(self) -> pl.DataFrame:
        return (self.refs.scan("silver", "games")
                .select(pl.col("game_id").cast(pl.Int64), pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)).collect())

    def play_context(self) -> pl.DataFrame:
        """(game_id, play_id, nfl_id) → down, pass_result, each taken from whichever silver table holds it."""
        wanted = ("down", "pass_result")
        pp_cols = self.refs.scan("silver", "player_play").collect_schema().names()
        play_cols = self.refs.scan("silver", "plays").collect_schema().names()
        from_pp = [k for k in wanted if k in pp_cols]
        from_play = [k for k in wanted if k in play_cols and k not in from_pp]
        ctx = (self.refs.scan("silver", "player_play").select(*PP_KEYS, *from_pp).collect()
               .with_columns([pl.col(k).cast(pl.Int64) for k in PP_KEYS]))
        if from_play:
            plays = (self.refs.scan("silver", "plays").select(*PLAY_KEYS, *from_play).collect()
                     .with_columns([pl.col(k).cast(pl.Int64) for k in PLAY_KEYS]).unique(subset=PLAY_KEYS))
            ctx = ctx.join(plays, on=PLAY_KEYS, how="left")
        for k in wanted:
            if k not in ctx.columns:
                log.warning("[nfl_outcomes] context column %s unavailable — filled with null", k)
                ctx = ctx.with_columns(pl.lit(None, dtype=pl.String).alias(k))
        return ctx.select(*PP_KEYS, pl.col("down").cast(pl.Float64, strict=False).round(0).cast(pl.Int64),
                          pl.col("pass_result").cast(pl.String))

    def input_files(self) -> list[Path]:
        files = [*self.refs.files("silver", "game_tracking")]
        for name in ("nfl_play_anchors", "nfl_player_play_windows", "nfl_trajectory_quality"):
            files += self.p03.dataset_files(name)
        for layer, name in (("gold", "dim_athlete"), ("silver", "games"), ("silver", "player_play"), ("silver", "plays")):
            files += self.refs.files(layer, name)
        return [Path(f) for f in files]

    def params(self) -> dict[str, Any]:
        return {"outcomes": self.c.params(), "kinematics": self.kcfg.params(), "limit_games": self.limit_games,
                "athletes": self.athlete_ids}

    # ------------------------------------------------------------------ compute
    def compute(self) -> dict[str, pl.DataFrame]:
        c, t0 = self.c, time.perf_counter()
        anchors, athletes = self.anchors(), self.athletes()
        specs = c.smoothing_specs()
        primary, alternates = specs[0], specs[1:]
        acc: dict[str, list[pl.DataFrame]] = {"plays": [], "variants": [], "hist": []}
        m = measure_columns(c)
        for season, file in self.kin.nfl_season_files().items():
            batches = self.kin.game_batches(file)
            for bi, (lo, hi) in enumerate(batches, 1):
                snaps = anchors.filter(pl.col("game_id").is_between(lo, hi))
                raw = (self.kin.load_nfl_raw(file, season, lo, hi).drop("s", "a", "dir", "o")
                       .with_columns([pl.col(k).cast(pl.Int64) for k in PP_KEYS])
                       .join(snaps.select(PLAY_KEYS), on=PLAY_KEYS, how="semi"))
                if self.athlete_ids is not None:
                    raw = raw.filter(pl.col("nfl_id").is_in(self.athlete_ids))
                if raw.is_empty():
                    continue
                frames, _ = prepare_frames(raw, NFL, self.kcfg)
                del raw
                starts = start_table(compute_kinematics(frames, self.kcfg, primary), snaps, c, primary.window // 2, self.kcfg.dt_s)
                acc["plays"].append(play_table(starts, c))
                acc["hist"].append(phase_hist(starts, athletes, c))
                ids = sorted(frames["game_id"].unique().to_list())[:: max(1, c.sensitivity_game_stride)]
                sub = frames.filter(pl.col("game_id").is_in(ids))
                other = play_table(starts.filter(pl.col("game_id").is_in(ids)), c, exclude_boundary=not c.exclude_boundary)
                acc["variants"].append(other.select(pl.lit("boundary_incl" if c.exclude_boundary else "boundary_excl").alias("variant"),
                                                    *PP_KEYS, *[pl.col(k).cast(pl.Float32) for k in m]))
                del starts
                for sp in alternates:
                    st = start_table(compute_kinematics(sub, self.kcfg, sp), snaps, c, sp.window // 2, self.kcfg.dt_s)
                    acc["variants"].append(play_table(st, c).select(pl.lit(sp.label).alias("variant"), *PP_KEYS,
                                                                    *[pl.col(k).cast(pl.Float32) for k in m]))
                    del st
                log.info("[nfl_outcomes] season %s batch %d/%d (games %s–%s): %s frames, %s player-plays (%.0fs)", season, bi,
                         len(batches), lo, hi, f"{frames.height:,}", f"{acc['plays'][-1].height:,}", time.perf_counter() - t0)
                del frames, sub
                gc.collect()
        if not acc["plays"]:
            raise RuntimeError("no NFL tracking frame matched a snap-anchored play — check gold/phase_03/nfl_play_anchors")
        measures = pl.concat(acc["plays"], how="vertical")
        plays = assemble_plays(measures, self.labels(), self.quality(), self.play_context(), athletes, self.games(), c)
        assert_no_combine_columns(plays, "play table")
        vc = variants_contract(c)
        variants = pl.concat(acc["variants"], how="vertical") if acc["variants"] else vc.empty()
        return {plays_contract(c).name: plays, vc.name: variants.select(list(vc.columns)).sort("variant", *PP_KEYS),
                PHASE_SPACE.name: finish_hist(acc["hist"], c), FUNNEL.name: funnel_table(plays, c)}

    # ------------------------------------------------------------------ on demand (figures)
    def example_frames(self, game_id: int, play_id: int, nfl_id: int) -> pl.DataFrame:
        """Recorded frames of ONE player-play (t_rel, speed, a_tan, boundary), recomputed from silver on demand."""
        sp = SmoothingSpec(*self.c.primary_smoothing)
        kin = self.kin.nfl_play_kinematics(int(game_id), int(play_id), int(nfl_id), sp)
        kin = kin.with_columns([pl.col(k).cast(pl.Int64) for k in PP_KEYS])
        snaps = self.anchors().filter((pl.col("game_id") == int(game_id)) & (pl.col("play_id") == int(play_id)))
        return start_table(kin, snaps, self.c, sp.window // 2, self.kcfg.dt_s)


def example_plays(plays: pl.DataFrame, column: str, positions: Sequence[str], quantiles: Sequence[float] = (0.5, 0.9)) -> pl.DataFrame:
    """Rule-chosen example windows: for each position the eligible window whose ``column`` is closest to the given
    quantiles of that position (ties: smallest game_id, play_id, nfl_id). No hand picking."""
    rows = []
    for p in positions:
        d = plays.filter(pl.col("eligible") & (pl.col("position_group") == p) & pl.col(column).is_not_null())
        if not d.height:
            continue
        for q in quantiles:
            target = d[column].quantile(q, "linear")
            r = d.with_columns((pl.col(column) - target).abs().alias("_d")).sort("_d", *PP_KEYS).row(0, named=True)
            rows.append({"position_group": p, "quantile": float(q), "game_id": r["game_id"], "play_id": r["play_id"],
                         "nfl_id": r["nfl_id"], "value": float(r[column])})
    return pl.DataFrame(rows, schema={"position_group": S, "quantile": F, "game_id": I, "play_id": I, "nfl_id": I, "value": F})
