"""Snap-anchored NFL play windows (Phase 03 · M1.1 — Workstream C foundation).

Why
    M1 summarised each NFL player-play over the WHOLE tracked window, pre-snap included, so a
    "peak speed" could come from pre-snap motion and a "first-second acceleration" had no anchor.
    Combine drills and game movements are only comparable inside football-defined windows.

Anchors (one row per play, :func:`play_anchors`)
    play_start / play_end   first / last tracked timestamp of the play (any entity, football included)
    snap                    earliest timestamp tagged with the highest-priority snap event present
                            (``PlayWindowConfig.snap_events``); null → no window is defined on the play
    release                 earliest release-event timestamp at or after the snap (pass / handoff / run
                            / sack …); null → ``snap_to_release`` is undefined, never approximated
    flags                   multiple_snap_tags (the chosen snap event occurs at > 1 timestamp — the
                            earliest is used), snap_at_play_start (no pre-snap frames: pre-snap window
                            and snap-relative latencies are not observable)

Windows (``PlayWindowConfig.windows``) are CLOSED intervals in seconds relative to the snap,
[lo, hi], with membership tolerance ``frame_tolerance_s`` (half a frame). ``pre_snap`` ends one
nominal frame before the snap, so pre- and post-snap frames never overlap.

Kinematics are computed on the WHOLE player trajectory first and windowed afterwards. The snap is
not a data boundary, so windowing never creates new filter edges. Caveat: a centred SG window of
``w`` frames lets frames up to ``w // 2`` × 0.1 s before the snap influence derived values in the
first post-snap frames (sg7p2: 0.3 s). Snap-relative TIMING features therefore use raw
(unsmoothed) positions: ``time_to_disp_s`` = first post-snap time at which the raw displacement
from the player's snap position reaches ``getoff_disp_yds``.

Coordinates: the NFL builder feeds STANDARDISED coordinates (silver ``x_std``/``y_std``: offense
always moves toward +x). A 180° rotation leaves speed, accelerations, yaw rate and curvature
unchanged, and ``disp_x`` > 0 always means "toward the offense's goal line".
"""

from __future__ import annotations

import polars as pl

from .config import KinematicsConfig, PlayWindowConfig, SmoothingSpec, WindowSpec

PLAY_KEYS = ["game_id", "play_id"]
PP_KEYS = ["game_id", "play_id", "nfl_id"]


def _secs(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """(a − b) in seconds for two datetime expressions."""
    return (a - b).dt.total_microseconds().cast(pl.Float64) / 1e6


# ============================================================================== anchors
def play_anchors(rows: pl.DataFrame, cfg: PlayWindowConfig) -> pl.DataFrame:
    """One row per play from tracking rows with (game_id, play_id, ts, event). Include football rows:
    an event tag may be attached to any entity of the frame."""
    bounds = rows.group_by(PLAY_KEYS).agg(pl.col("ts").min().alias("play_start_ts"),
                                          pl.col("ts").max().alias("play_end_ts"))
    tagged = (rows.filter(pl.col("event").is_not_null())
              .select(*PLAY_KEYS, "ts", pl.col("event").cast(pl.String).str.to_lowercase().alias("event")).unique())
    prio = {e: i for i, e in enumerate(cfg.snap_events)}
    snap_c = (tagged.filter(pl.col("event").is_in(list(cfg.snap_events)))
              .with_columns(pl.col("event").replace_strict(prio, return_dtype=pl.Int64).alias("_prio")))
    snaps = (snap_c.filter(pl.col("_prio") == pl.col("_prio").min().over(PLAY_KEYS))
             .group_by(PLAY_KEYS).agg(pl.col("ts").min().alias("snap_ts"), pl.col("event").first().alias("snap_event"),
                                      pl.col("ts").n_unique().cast(pl.Int64).alias("n_snap_tag_times")))
    rel = (tagged.filter(pl.col("event").is_in(list(cfg.release_events)))
           .join(snaps.select(*PLAY_KEYS, "snap_ts"), on=PLAY_KEYS, how="inner")
           .filter(pl.col("ts") >= pl.col("snap_ts"))
           .group_by(PLAY_KEYS).agg(pl.col("ts").min().alias("release_ts"),
                                    pl.col("event").sort_by("ts").first().alias("release_event")))
    a = bounds.join(snaps, on=PLAY_KEYS, how="left").join(rel, on=PLAY_KEYS, how="left")
    tol = cfg.frame_tolerance_s
    return a.with_columns(
        pl.col("snap_ts").is_not_null().alias("has_snap"),
        (pl.col("n_snap_tag_times").fill_null(0) > 1).alias("multiple_snap_tags"),
        (_secs(pl.col("snap_ts"), pl.col("play_start_ts")) <= tol).fill_null(False).alias("snap_at_play_start"),
        pl.col("release_ts").is_not_null().alias("has_release"),
        _secs(pl.col("snap_ts"), pl.col("play_start_ts")).alias("pre_snap_s"),
        _secs(pl.col("play_end_ts"), pl.col("snap_ts")).alias("post_snap_s"),
        _secs(pl.col("release_ts"), pl.col("snap_ts")).alias("snap_to_release_s"),
        pl.col("n_snap_tag_times").fill_null(0),
    ).sort(PLAY_KEYS)


ANCHOR_DOCS: dict[str, str] = {
    "play_start_ts": "first tracked timestamp of the play (any entity)",
    "play_end_ts": "last tracked timestamp of the play (any entity)",
    "snap_ts": "earliest timestamp tagged with the chosen snap event (null = no snap tag)",
    "snap_event": "snap event name used (highest priority present in PlayWindowConfig.snap_events)",
    "n_snap_tag_times": "distinct timestamps carrying the chosen snap event (> 1 → multiple_snap_tags)",
    "release_ts": "earliest release-event timestamp at/after the snap (null = none)",
    "release_event": "release event name (pass_forward, handoff, run, qb_sack, …)",
    "has_snap": "snap anchor defined", "has_release": "release anchor defined",
    "multiple_snap_tags": "the chosen snap event occurs at more than one timestamp (earliest used)",
    "snap_at_play_start": "snap within half a frame of the first tracked frame (no pre-snap frames)",
    "pre_snap_s": "snap − play start (s)", "post_snap_s": "play end − snap (s)",
    "snap_to_release_s": "release − snap (s)",
}


# ============================================================================== window bounds
def _ref(ref: str) -> pl.Expr:
    return pl.col({"play_start": "play_start_ts", "snap": "snap_ts", "release": "release_ts",
                   "play_end": "play_end_ts"}[ref])


def window_bounds(anchors: pl.DataFrame, spec: WindowSpec, dt_s: float) -> pl.DataFrame:
    """Per play: [lo_rel, hi_rel] in seconds relative to the snap, the nominal expected frame count and
    whether the nominal window runs past the end of tracking. Plays where the window is undefined
    (missing reference, or hi < lo) are dropped here and therefore get no window row."""
    snap = pl.col("snap_ts")
    b = anchors.select(
        *PLAY_KEYS, "snap_ts", "play_end_ts",
        (_secs(_ref(spec.start_ref), snap) + spec.start_offset_s).alias("lo_rel"),
        (_secs(_ref(spec.end_ref), snap) + spec.end_offset_s).alias("hi_rel"),
        _secs(pl.col("play_end_ts"), snap).alias("_end_rel"),
    ).filter(pl.col("lo_rel").is_not_null() & pl.col("hi_rel").is_not_null() & (pl.col("hi_rel") >= pl.col("lo_rel")))
    return b.with_columns(
        (((pl.col("hi_rel") - pl.col("lo_rel")) / dt_s).round(0) + 1).cast(pl.Int64).alias("expected_frames"),
        (pl.col("hi_rel") > pl.col("_end_rel") + dt_s / 2).alias("truncated_by_play_end"),
    ).drop("_end_rel", "play_end_ts")


# ============================================================================== window features
def _num_features(cfg: PlayWindowConfig) -> list[str]:
    return ["speed_start", "speed_end", "speed_max", "speed_mean", "speed_p90", "disp_x", "disp_y", "net_disp",
            "path_len", *[f"hs_dist_{_thr_label(t)}" for t in cfg.speed_thresholds], "a_tan_sust_max",
            "decel_sust_max", "accel_mag_p95", "time_to_disp_s"]


def _thr_label(t: float) -> str:
    return f"{t:g}".replace(".", "p")


def window_features(kin: pl.DataFrame, anchors: pl.DataFrame, spec: WindowSpec, cfg: PlayWindowConfig,
                    kcfg: KinematicsConfig, smoothing: SmoothingSpec | None = None) -> pl.DataFrame:
    """Player-play features inside one window. ``kin``: player frame kinematics (whole trajectories,
    standardised coordinates) with PP_KEYS, seg_idx, pos, seg_len, ts, x, y, speed, a_tan, accel_mag,
    step_smooth, valid_kin. Returns one row per player-play observed in the window."""
    smoothing = smoothing or kcfg.smoothing
    dt, tol = kcfg.dt_s, cfg.frame_tolerance_s
    n_sust = max(1, round(cfg.sustain_window_s / dt))
    b = window_bounds(anchors, spec, dt)
    f = (kin.join(b, on=PLAY_KEYS, how="inner")
         .with_columns(_secs(pl.col("ts"), pl.col("snap_ts")).alias("t_rel"))
         .filter((pl.col("t_rel") >= pl.col("lo_rel") - tol) & (pl.col("t_rel") <= pl.col("hi_rel") + tol))
         .sort([*PP_KEYS, "ts"]))
    if f.is_empty():
        return _empty_features(cfg)
    seg = [*PP_KEYS, "seg_idx"]
    anchored_at_snap = spec.start_ref == "snap" and abs(spec.start_offset_s) < 1e-9
    f = f.with_columns(
        (pl.min_horizontal(pl.col("pos"), pl.col("seg_len") - 1 - pl.col("pos")) < smoothing.window // 2)
        .alias("_boundary"),
        pl.int_range(pl.len()).over(PP_KEYS).alias("_k"),       # 0 = first frame of the window
        pl.col("a_tan").rolling_mean(window_size=n_sust, min_samples=n_sust).over(seg).alias("_a_sust"),
        # raw position at the snap frame (|t_rel| ≤ tol) → displacement for the first-step time
        pl.col("x").filter(pl.col("t_rel").abs() <= tol).first().over(PP_KEYS).alias("_x0"),
        pl.col("y").filter(pl.col("t_rel").abs() <= tol).first().over(PP_KEYS).alias("_y0"),
    ).with_columns(
        ((pl.col("x") - pl.col("_x0")).pow(2) + (pl.col("y") - pl.col("_y0")).pow(2)).sqrt().alias("_disp0"),
        pl.when(pl.col("_k") > 0).then(pl.col("step_smooth")).otherwise(None).alias("_step"),   # in-window steps only
    )
    v = pl.col("valid_kin")
    getoff = (pl.col("t_rel").filter((pl.col("_disp0") >= cfg.getoff_disp_yds) & (pl.col("t_rel") >= -tol)
                                     & (pl.col("t_rel") <= cfg.getoff_max_s + tol)).min()
              if anchored_at_snap else pl.lit(None, dtype=pl.Float64))
    out = f.group_by(PP_KEYS).agg(
        pl.len().cast(pl.Int64).alias("n_frames"),
        v.sum().cast(pl.Int64).alias("n_valid_kin"),
        pl.col("expected_frames").first(),
        pl.col("truncated_by_play_end").first(),
        pl.col("t_rel").min().alias("t_first_rel"), pl.col("t_rel").max().alias("t_last_rel"),
        pl.col("lo_rel").first(), pl.col("hi_rel").first(),
        pl.col("speed").filter(v).first().alias("speed_start"),
        pl.col("speed").filter(v).last().alias("speed_end"),
        pl.col("speed").filter(v).max().alias("speed_max"),
        pl.col("speed").filter(v).mean().alias("speed_mean"),
        pl.col("speed").filter(v).quantile(0.9, "linear").alias("speed_p90"),
        (pl.col("x").last() - pl.col("x").first()).alias("disp_x"),
        (pl.col("y").last() - pl.col("y").first()).alias("disp_y"),
        ((pl.col("x").last() - pl.col("x").first()).pow(2) + (pl.col("y").last() - pl.col("y").first()).pow(2))
        .sqrt().alias("net_disp"),
        pl.col("_step").sum().alias("path_len"),
        *[pl.col("_step").filter(pl.col("speed") >= t).sum().alias(f"hs_dist_{_thr_label(t)}")
          for t in cfg.speed_thresholds],
        pl.col("_a_sust").max().alias("a_tan_sust_max"),
        (-pl.col("_a_sust")).max().alias("decel_sust_max"),
        pl.col("accel_mag").filter(v).quantile(0.95, "linear").alias("accel_mag_p95"),
        getoff.alias("time_to_disp_s"),
        pl.col("_boundary").filter(v).mean().alias("boundary_frac"),
        (pl.col("_boundary") & (pl.col("speed") == pl.col("speed").filter(v).max())).any().alias("speed_max_in_boundary"),
    )
    ok = pl.col("n_valid_kin") >= cfg.min_window_frames
    num = _num_features(cfg)
    out = out.with_columns(
        pl.lit(spec.name).alias("window"), pl.lit(spec.phase).alias("phase"),
        (pl.col("n_frames") / pl.col("expected_frames")).cast(pl.Float64).alias("coverage"),
        (pl.col("t_first_rel") > pl.col("lo_rel") + tol).alias("censored_start"),
        (pl.col("t_last_rel") < pl.col("hi_rel") - tol).alias("censored_end"),
        ok.alias("features_valid"),
    ).with_columns(
        *[pl.when(ok).then(pl.col(c)).otherwise(None).cast(pl.Float64).alias(c) for c in num],
        (pl.col("coverage") >= cfg.min_coverage).alias("coverage_ok"),
    )
    return out.select(list(_feature_schema(cfg))).sort(PP_KEYS)


def _feature_schema(cfg: PlayWindowConfig) -> dict[str, pl.DataType]:
    I, F, B, S = pl.Int64(), pl.Float64(), pl.Boolean(), pl.String()
    return {"game_id": I, "play_id": I, "nfl_id": I, "window": S, "phase": S, "n_frames": I, "n_valid_kin": I,
            "expected_frames": I, "coverage": F, "coverage_ok": B, "censored_start": B, "censored_end": B,
            "truncated_by_play_end": B, "features_valid": B, "lo_rel": F, "hi_rel": F, "t_first_rel": F,
            "t_last_rel": F, **{c: F for c in _num_features(cfg)}, "boundary_frac": F, "speed_max_in_boundary": B}


def _empty_features(cfg: PlayWindowConfig) -> pl.DataFrame:
    return pl.DataFrame(schema=_feature_schema(cfg))


def all_window_features(kin: pl.DataFrame, anchors: pl.DataFrame, cfg: PlayWindowConfig, kcfg: KinematicsConfig,
                        smoothing: SmoothingSpec | None = None) -> pl.DataFrame:
    parts = [window_features(kin, anchors, w, cfg, kcfg, smoothing) for w in cfg.windows]
    schema = _feature_schema(cfg)
    return pl.concat([p.cast(schema) for p in parts], how="vertical") if parts else _empty_features(cfg)


def feature_docs(cfg: PlayWindowConfig) -> dict[str, str]:
    d = {
        "window": "window name (PlayWindowConfig.windows)", "phase": "pre_snap | post_snap",
        "n_frames": "player frames inside the window", "n_valid_kin": "of which with SG derivatives",
        "expected_frames": "nominal frames of the window at 10 Hz (not truncated at play end)",
        "coverage": "n_frames / expected_frames", "coverage_ok": "coverage ≥ min_coverage",
        "censored_start": "player's first window frame later than the window start",
        "censored_end": "player's last window frame earlier than the window end",
        "truncated_by_play_end": "the nominal window extends past the play's last tracked frame",
        "features_valid": "n_valid_kin ≥ min_window_frames (else numeric features are null)",
        "lo_rel": "window start relative to the snap (s)", "hi_rel": "window end relative to the snap (s)",
        "t_first_rel": "first player frame in the window, relative to the snap (s)",
        "t_last_rel": "last player frame in the window, relative to the snap (s)",
        "speed_start": "derived speed at the first valid window frame (yd/s)",
        "speed_end": "derived speed at the last valid window frame (yd/s)",
        "speed_max": "maximum derived speed in the window (yd/s) — opportunity-dependent, not a capacity",
        "speed_mean": "mean derived speed (yd/s)", "speed_p90": "90th percentile derived speed (yd/s)",
        "disp_x": "raw standardised x(last) − x(first) (yd; + = toward the offense's goal line)",
        "disp_y": "raw standardised y(last) − y(first) (yd)", "net_disp": "straight-line displacement (yd)",
        "path_len": "Σ smoothed path increments inside the window (yd)",
        "a_tan_sust_max": "max rolling mean of a_tan over sustain_window_s within a segment (yd/s²)",
        "decel_sust_max": "max rolling mean of −a_tan over sustain_window_s (yd/s²; braking capability proxy)",
        "accel_mag_p95": "95th percentile |acceleration vector| (yd/s²)",
        "time_to_disp_s": "post-snap windows anchored at the snap: first time the RAW displacement from the "
                          "player's snap position reaches getoff_disp_yds (s; null if not reached within "
                          "getoff_max_s or no frame at the snap)",
        "boundary_frac": "valid frames inside the SG filter boundary of their segment",
        "speed_max_in_boundary": "the window's speed maximum lies on a filter-boundary frame",
    }
    for t in cfg.speed_thresholds:
        d[f"hs_dist_{_thr_label(t)}"] = f"Σ smoothed path increments with speed ≥ {t:g} yd/s (yd)"
    return d
