"""Movement window (onset / end) and temporal phase structure (Family E).

Input everywhere: frame kinematics from ``kinematics.compute_kinematics`` (persisted for Combine as
``combine_frame_kinematics``), sorted by (trajectory keys, ts). ``keys`` = trajectory keys
(Combine ``event_id, nfl_id, attempt``; NFL ``game_id, play_id, nfl_id``).

Movement window
    onset      first frame of the first run of ``onset_frames`` consecutive frames (within one
               segment) with speed ≥ ``onset_speed``. ``onset_censored`` = the trajectory already
               moves on its first valid frame, i.e. the true start lies before the recording.
    onset_fast same with ``onset_speed_fast``: insensitive to slow pre-start drift (M1 evidence:
               the shuttle example drifts at < 1 yd/s for ~1 s before the first leg).
    end        last frame with speed ≥ ``onset_speed``.
    end_censored  last-frame speed ≥ ``end_censor_speed``: the recording stopped while the
               athlete was still moving fast (forty: by design at the finish line), so
               ``active_duration_s`` and time-to-peak are truncated, not observed.
    active     onset ≤ t ≤ end. All window-based features use active frames only.

Phases (frame label, PROVISIONAL thresholds)
    stationary  speed < stationary_speed
    accelerating a_tan ≥ +phase_accel     decelerating a_tan ≤ −phase_accel     steady otherwise
Phase runs shorter than ``min_phase_frames`` are ignored when counting transitions (noise).
"""

from __future__ import annotations

import polars as pl

from .config import MovementConfig

PHASES = ("stationary", "accelerating", "steady", "decelerating")


def _sustained(threshold: float, n: int, seg: list[str]) -> pl.Expr:
    on = (pl.col("speed") >= threshold).fill_null(False)
    return pl.all_horizontal([on.shift(-k).over(seg).fill_null(False) for k in range(n)])


def movement_window(kin: pl.DataFrame, keys: list[str], mcfg: MovementConfig) -> pl.DataFrame:
    """One row per trajectory: first/last valid frame, onsets, end of activity, censoring flags."""
    seg = [*keys, "seg_idx"]
    f = kin.filter(pl.col("valid_kin")).sort([*keys, "t_s"]).with_columns(
        _sustained(mcfg.onset_speed, mcfg.onset_frames, seg).alias("_on"),
        _sustained(mcfg.onset_speed_fast, mcfg.onset_frames, seg).alias("_onf"))
    return (f.group_by(keys).agg(
                pl.col("t_s").min().alias("t_first_s"), pl.col("t_s").max().alias("t_last_s"),
                pl.col("t_s").filter(pl.col("_on")).min().alias("t_onset_s"),
                pl.col("t_s").filter(pl.col("_onf")).min().alias("t_onset_fast_s"),
                pl.col("t_s").filter(pl.col("speed") >= mcfg.onset_speed).max().alias("t_end_active_s"),
                pl.col("speed").first().alias("first_speed"), pl.col("speed").last().alias("last_speed"),
                pl.col("speed").max().alias("_vmax"))
            .with_columns(
                (pl.col("t_onset_s") <= pl.col("t_first_s") + 1e-6).alias("onset_censored"),
                # recording stops while the athlete still moves fast → durations / peaks truncated
                (pl.col("last_speed") >= mcfg.end_censor_speed).alias("end_censored"),
                (pl.col("t_end_active_s") - pl.col("t_onset_s")).alias("active_duration_s"),
                (pl.col("t_last_s") - pl.col("t_onset_s")).alias("onset_to_end_s"),
                (pl.col("t_last_s") - pl.col("t_onset_fast_s")).alias("fast_onset_to_end_s"),
                (pl.col("last_speed") / pl.col("_vmax")).alias("last_speed_frac_of_max"))
            .drop("_vmax"))


def active_frames(kin: pl.DataFrame, window: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Valid frames inside the movement window, with ``t_rel`` = seconds since onset."""
    return (kin.filter(pl.col("valid_kin"))
            .join(window.select(*keys, "t_onset_s", "t_end_active_s", "active_duration_s"), on=keys, how="inner")
            .filter(pl.col("t_onset_s").is_not_null()
                    & (pl.col("t_s") >= pl.col("t_onset_s")) & (pl.col("t_s") <= pl.col("t_end_active_s")))
            .with_columns((pl.col("t_s") - pl.col("t_onset_s")).alias("t_rel"))
            .sort([*keys, "t_s"]))


def phase_expr(mcfg: MovementConfig) -> pl.Expr:
    return (pl.when(pl.col("speed") < mcfg.stationary_speed).then(pl.lit("stationary"))
            .when(pl.col("a_tan") >= mcfg.phase_accel).then(pl.lit("accelerating"))
            .when(pl.col("a_tan") <= -mcfg.phase_accel).then(pl.lit("decelerating"))
            .otherwise(pl.lit("steady")))


def temporal_features(act: pl.DataFrame, keys: list[str], mcfg: MovementConfig) -> pl.DataFrame:
    """Phase fractions, transition counts, timing of peak acceleration vs peak speed, and
    early / middle / late thirds of the active window (``act`` from :func:`active_frames`)."""
    seg = [*keys, "seg_idx"]
    f = act.with_columns(phase_expr(mcfg).alias("phase")).with_columns(
        pl.col("phase").rle_id().over(seg).alias("_prun"),
        (3.0 * pl.col("t_rel") / pl.col("active_duration_s")).floor().clip(0, 2).alias("_third"))
    runs = (f.group_by([*seg, "_prun"]).agg(pl.col("phase").first(), pl.len().alias("_n"), pl.col("t_s").min().alias("_t"))
            .filter(pl.col("_n") >= mcfg.min_phase_frames).sort([*seg, "_t"]))
    trans = (runs.with_columns((pl.col("phase") != pl.col("phase").shift(1).over(seg)).fill_null(False).alias("_chg"))
             .group_by(keys).agg(pl.col("_chg").sum().cast(pl.Int64).alias("n_phase_transitions")))
    third = lambda k: pl.col("_third") == k  # noqa: E731
    out = f.group_by(keys).agg(
        *[(pl.col("phase") == p).mean().alias(f"frac_time_{p}") for p in PHASES],
        pl.col("t_rel").sort_by("speed", descending=True, nulls_last=True).first().alias("_t_vpk"),
        pl.col("t_rel").sort_by("a_tan", descending=True, nulls_last=True).first().alias("_t_apk"),
        *[pl.col("a_tan").filter(third(k)).mean().alias(f"a_tan_mean_{n}") for k, n in enumerate(("early", "mid", "late"))],
        pl.col("speed").filter(third(2)).mean().alias("_v_late"), pl.col("speed").max().alias("_vmax"),
        pl.col("active_duration_s").first().alias("_dur"),
    )
    return (out.join(trans, on=keys, how="left")
            .with_columns(pl.col("n_phase_transitions").fill_null(0),
                          (pl.col("_t_vpk") - pl.col("_t_apk")).alias("peak_accel_lead_s"),
                          (pl.col("_v_late") / pl.col("_vmax")).alias("late_speed_maintenance"),
                          pl.when(pl.col("_dur") > 0).then(pl.col("n_phase_transitions") / pl.col("_dur"))
                          .alias("phase_transition_rate"))
            .drop("_t_vpk", "_t_apk", "_v_late", "_vmax", "_dur"))
