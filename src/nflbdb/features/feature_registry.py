"""Feature registry — the single source of truth for feature definitions.

Every persisted feature column must have a :class:`FeatureDef`. The registry is exported to
``reports/tables/phase_03/feature_dictionary.csv`` and used to document dataset columns in the
Parquet sidecars (``*.schema.json``). ``tests/test_phase03_kinematics.py`` asserts that every
feature column produced by the M1 builders is registered.

``status``:
    core                  physically interpretable, computed with explicit validity rules
    requires_validation   definition sound, but thresholds / conventions not yet calibrated on data
    experimental          noise-sensitive or convention-dependent; never use without sensitivity checks
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from . import FEATURES_VERSION


@dataclass(frozen=True)
class FeatureDef:
    feature_name: str
    family: str                   # A speed/accel · B braking · C agility · D efficiency · E temporal
                                  # F signal · G reliability · H context · I normalized · Q quality
    level: str                    # frame | segment | trajectory | event
    definition: str
    formula: str
    unit: str
    source_columns: str
    grain: str
    applicable_context: str
    missing_behavior: str
    quality_checks: str
    limitations: str
    cost: str                     # O(frames) vectorised, etc.
    status: str = "core"
    version: str = FEATURES_VERSION


_F = "frame (trajectory keys + ts)"
_S = "segment (trajectory keys + seg_idx)"
_T = "trajectory (Combine attempt / NFL player-play)"
_E = "event (trajectory keys + seg_idx + event_type + rule + event_idx)"
_BOTH = "Combine (all drills) and NFL (all plays)"
_VEC = "O(frames), vectorised NumPy"
_POL = "O(frames), Polars group/run-length"

REGISTRY: tuple[FeatureDef, ...] = (
    # ------------------------------------------------------------------ frame level
    FeatureDef("x_s", "F", "frame", "Savitzky–Golay smoothed x position", "SG(x; window, polyorder), segment-local",
               "yd", "x", _F, _BOTH, "null if segment shorter than window", "pos_resid distribution",
               "zero-phase (uses future frames) unless causal=True", _VEC),
    FeatureDef("y_s", "F", "frame", "Savitzky–Golay smoothed y position", "SG(y), segment-local", "yd", "y", _F, _BOTH,
               "null if segment shorter than window", "pos_resid distribution", "as x_s", _VEC),
    FeatureDef("vx", "A", "frame", "velocity component along x", "SG first derivative of x / Δt", "yd/s", "x", _F, _BOTH,
               "null if segment shorter than window", "speed vs vendor s", "coordinate-frame dependent", _VEC),
    FeatureDef("vy", "A", "frame", "velocity component along y", "SG first derivative of y / Δt", "yd/s", "y", _F, _BOTH,
               "null if segment shorter than window", "speed vs vendor s", "coordinate-frame dependent", _VEC),
    FeatureDef("speed", "A", "frame", "derived speed (magnitude of smoothed velocity)", "sqrt(vx² + vy²)", "yd/s", "x, y",
               _F, _BOTH, "null if segment shorter than window",
               "agreement with vendor s (corr, MAD); ≤ max_speed_yds", "smoothing attenuates short peaks", _VEC),
    FeatureDef("ax", "A", "frame", "vector acceleration component along x", "SG second derivative of x / Δt²", "yd/s²",
               "x", _F, _BOTH, "null if segment shorter than window", "smoothing sensitivity",
               "second derivative: noise-sensitive", _VEC),
    FeatureDef("ay", "A", "frame", "vector acceleration component along y", "SG second derivative of y / Δt²", "yd/s²",
               "y", _F, _BOTH, "null if segment shorter than window", "smoothing sensitivity",
               "second derivative: noise-sensitive", _VEC),
    FeatureDef("accel_mag", "A", "frame", "magnitude of the vector acceleration (includes turning)", "sqrt(ax² + ay²)",
               "yd/s²", "x, y", _F, _BOTH, "null if segment shorter than window", "≤ max_accel_yds2; vs vendor a",
               "NOT the speed derivative; always ≥ |a_tan|", _VEC),
    FeatureDef("a_tan", "A", "frame", "tangential acceleration = scalar speed derivative d|v|/dt (signed; < 0 braking)",
               "(vx·ax + vy·ay) / speed", "yd/s²", "x, y", _F, _BOTH,
               "null if speed < min_tangent_speed or segment too short", "smoothing sensitivity",
               "noisy near min_tangent_speed", _VEC),
    FeatureDef("a_lat", "C", "frame", "lateral (centripetal) acceleration, signed (+ counter-clockwise)",
               "(vx·ay − vy·ax) / speed", "yd/s²", "x, y", _F, _BOTH, "null if speed < min_tangent_speed",
               "smoothing sensitivity", "sign depends on coordinate handedness (Combine frame unverified)", _VEC),
    FeatureDef("yaw_rate", "C", "frame", "heading change rate (wraparound-free), signed", "a_lat / speed (rad/s → deg/s)",
               "deg/s", "x, y", _F, _BOTH, "null if speed < min_heading_speed", "smoothing sensitivity",
               "undefined at low speed; sign convention as a_lat", _VEC),
    FeatureDef("curvature", "C", "frame", "path curvature (1 / turning radius), signed", "yaw_rate[rad/s] / speed",
               "1/yd", "x, y", _F, _BOTH, "null if speed < min_heading_speed",
               "smoothing sensitivity; extreme values near threshold speed", "unstable at low speed", _VEC,
               status="requires_validation"),
    FeatureDef("heading", "C", "frame", "direction of motion, math convention (0° = +x, counter-clockwise)",
               "atan2(vy, vx) mod 360", "deg", "x, y", _F, _BOTH, "null if speed < min_heading_speed",
               "circular agreement with vendor dir under both candidate conventions",
               "absolute value meaningful only in a verified coordinate frame", _VEC, status="requires_validation"),
    FeatureDef("step_smooth", "D", "frame", "smoothed path increment from the previous frame of the same segment",
               "|(x̂, ŷ)_t − (x̂, ŷ)_{t−1}|", "yd", "x, y", _F, _BOTH, "null at segment start", "sum vs raw path",
               "smoothing shortens jittery paths", _VEC),
    FeatureDef("pos_resid", "Q", "frame", "distance between raw and smoothed position", "|(x, y) − (x̂, ŷ)|", "yd",
               "x, y", _F, _BOTH, "null if not smoothable", "tier rules", "mixes noise and real high-frequency motion",
               _VEC),
    # ------------------------------------------------------------------ segment level
    FeatureDef("n_frames", "Q", "segment", "frames in the segment", "count", "frames", "ts", _S, _BOTH, "—", "—", "—", _POL),
    FeatureDef("duration_s", "Q", "segment", "segment duration (first → last frame)", "t_last − t_first", "s", "ts",
               _S, _BOTH, "—", "—", "—", _POL),
    FeatureDef("smoothable", "Q", "segment", "segment long enough for the SG window", "n_frames ≥ window", "bool",
               "—", _S, _BOTH, "—", "—", "short segments carry no derivatives", _POL),
    FeatureDef("n_offgrid", "Q", "segment", "frames whose Δt deviates from nominal by > tolerance",
               "Σ |Δt − 100 ms| > 25 ms", "frames", "ts", _S, _BOTH, "—", "should be 0 (Phase 01 sample: 0)",
               "uniform-sampling assumption of SG", _POL),
    FeatureDef("path_raw_yds", "D", "segment", "raw path length", "Σ raw step", "yd", "x, y", _S, _BOTH, "—",
               "≥ net_disp_yds", "inflated by positional jitter", _POL),
    FeatureDef("path_smooth_yds", "D", "segment", "smoothed path length", "Σ step_smooth", "yd", "x, y", _S, _BOTH,
               "null if not smoothable", "≤ path_raw_yds typically", "smoothing-dependent", _POL),
    FeatureDef("net_disp_yds", "D", "segment", "straight-line displacement first → last raw frame",
               "|p_last − p_first|", "yd", "x, y", _S, _BOTH, "—", "≤ path_raw_yds", "—", _POL),
    FeatureDef("max_step_raw_yds", "Q", "segment", "largest raw per-frame displacement", "max raw step", "yd", "x, y",
               _S, _BOTH, "—", "≤ max_step_yds by construction", "—", _POL),
    FeatureDef("resid_rms_yds", "Q", "segment", "RMS raw-vs-smoothed position residual (noise proxy)",
               "sqrt(mean pos_resid²)", "yd", "x, y", _S + " / " + _T, _BOTH, "null if not smoothable", "tier rules",
               "also captures genuine rapid motion (e.g. foot plant)", _POL, status="requires_validation"),
    FeatureDef("resid_p95_yds", "Q", "segment", "95th percentile of position residual", "quantile(pos_resid, .95)",
               "yd", "x, y", _S, _BOTH, "null if not smoothable", "—", "as resid_rms_yds", _POL),
    FeatureDef("speed_vendor_mad", "Q", "segment", "median |derived speed − vendor s|", "median |speed − s|", "yd/s",
               "x, y, s", _S + " / " + _T, _BOTH, "null if not smoothable", "tier rules",
               "vendor s may itself be smoothed/lagged", _POL),
    FeatureDef("speed_vendor_p95", "Q", "segment", "95th pct |derived speed − vendor s|", "quantile(|speed − s|, .95)",
               "yd/s", "x, y, s", _S, _BOTH, "null if not smoothable", "—", "as speed_vendor_mad", _POL),
    FeatureDef("max_vendor_s", "Q", "segment", "maximum vendor speed", "max s", "yd/s", "s", _S, _BOTH, "—", "—",
               "vendor definition unconfirmed", _POL),
    FeatureDef("max_speed", "A", "segment", "maximum derived speed", "max speed", "yd/s", "x, y", _S + " / " + _T + " / " + _E,
               _BOTH, "null if not smoothable", "≤ max_speed_yds", "single-frame statistic; prefer percentiles in M2/M3",
               _POL),
    FeatureDef("max_accel_mag", "A", "segment", "maximum vector-acceleration magnitude", "max accel_mag", "yd/s²",
               "x, y", _S + " / " + _T, _BOTH, "null if not smoothable", "≤ max_accel_yds2",
               "single-frame, noise-sensitive", _POL),
    FeatureDef("n_speed_implausible", "Q", "segment", "frames with derived speed > max_speed_yds", "count", "frames",
               "x, y", _S + " / " + _T, _BOTH, "—", "reject tier if > 0", "threshold 12.5 yd/s", _POL),
    FeatureDef("n_accel_implausible", "Q", "segment", "frames with accel_mag > max_accel_yds2", "count", "frames",
               "x, y", _S + " / " + _T, _BOTH, "—", "reject tier if > 0", "threshold 15 yd/s²", _POL),
    FeatureDef("frac_heading_valid", "Q", "segment", "share of frames with speed ≥ min_heading_speed", "mean(heading_valid)",
               "fraction", "x, y", _S, _BOTH, "—", "—", "low for stationary roles (OL pre-snap)", _POL),
    # ------------------------------------------------------------------ trajectory level
    FeatureDef("n_raw_frames", "Q", "trajectory", "PLAYER frames read from silver", "count", "frames", "—", _T, _BOTH,
               "—", "—", "—", _POL),
    FeatureDef("n_null_xy_removed", "Q", "trajectory", "frames dropped for null/NaN coordinates or timestamp", "count",
               "frames", "x, y, ts", _T, _BOTH, "—", "expected 0", "—", _POL),
    FeatureDef("n_dup_ts_removed", "Q", "trajectory", "frames dropped as exact duplicate timestamps (first kept)",
               "count", "frames", "ts", _T, _BOTH, "—", "expected 0", "keeps first; values of duplicates not compared",
               _POL),
    FeatureDef("n_gap_breaks", "Q", "trajectory", "segment breaks caused by Δt > gap_factor × nominal", "count",
               "breaks", "ts", _T, _BOTH, "—", "—", "—", _POL),
    FeatureDef("n_jump_breaks", "Q", "trajectory", "segment breaks caused by implausible displacement", "count",
               "breaks", "x, y", _T, _BOTH, "—", "—", "—", _POL),
    FeatureDef("n_drill_breaks", "Q", "trajectory", "segment breaks caused by a drill_name change", "count", "breaks",
               "drill_name", _T, "Combine", "0 for NFL", "expected 0", "—", _POL),
    FeatureDef("n_segments", "Q", "trajectory", "segments in the trajectory", "max seg_idx", "count", "—", _T, _BOTH,
               "—", "1 expected", "—", _POL),
    FeatureDef("n_valid_frames", "Q", "trajectory", "frames after removals", "count", "frames", "—", _T, _BOTH, "—",
               "—", "—", _POL),
    FeatureDef("coverage", "Q", "trajectory", "share of raw frames inside smoothable segments",
               "frames_smoothable / n_raw_frames", "fraction", "—", _T, _BOTH, "—", "tier C if < c_min_coverage", "—",
               _POL),
    FeatureDef("quality_tier", "Q", "trajectory", "usability tier for kinematic features (A best)",
               "rule cascade reject → C → B → A (TierRules)", "label", "—", _T, _BOTH, "—", "—",
               "PROVISIONAL thresholds", _POL, status="requires_validation"),
    FeatureDef("tier_reasons", "Q", "trajectory", "all triggered tier rules", "';'-joined rule names", "text", "—", _T,
               _BOTH, "—", "—", "—", _POL),
    # ------------------------------------------------------------------ event level
    FeatureDef("duration_s", "B", "event", "event duration", "n_frames × Δt", "s", "a_tan / yaw_rate / speed", _E, _BOTH,
               "—", "≥ rule.min_duration_s", "quantised to 0.1 s", _POL),
    FeatureDef("entry_speed", "B", "event", "speed on the frame before the event", "speed[t_start − 1]", "yd/s",
               "speed", _E, _BOTH, "first run frame at a segment start", "—", "—", _POL),
    FeatureDef("exit_speed", "B", "event", "speed at event end (reversal: frame after the window)", "speed[t_end]",
               "yd/s", "speed", _E, _BOTH, "null if window ends the segment (reversal)", "—", "—", _POL),
    FeatureDef("speed_change", "B", "event",
               "braking: speed lost; acceleration: speed gained; turn: min − entry (≤ 0 = loss); reversal: exit − entry",
               "see definition", "yd/s", "speed", _E, _BOTH, "—", "≥ rule.min_change (braking/accel)",
               "sign convention differs by event type — read with event_type", _POL),
    FeatureDef("peak_abs_a_tan", "B", "event", "peak |tangential acceleration| in the event", "max |a_tan|", "yd/s²",
               "a_tan", _E, "braking, acceleration", "null for turn/reversal", "smoothing sensitivity",
               "noise-sensitive peak", _POL),
    FeatureDef("mean_abs_a_tan", "B", "event", "mean |tangential acceleration| in the event", "mean |a_tan|", "yd/s²",
               "a_tan", _E, "braking, acceleration", "null for turn/reversal", "—", "—", _POL),
    FeatureDef("distance_yds", "B", "event", "smoothed distance travelled during the event", "Σ step_smooth", "yd",
               "x, y", _E, _BOTH, "—", "—", "—", _POL),
    FeatureDef("time_to_transition_s", "B", "event",
               "braking → time to re-acceleration; acceleration → time to next braking (same rule threshold)",
               "t(next opposite-threshold frame) − t_end", "s", "a_tan", _E, "braking, acceleration",
               "null if no transition in the segment", "—", "censored by segment end", _POL,
               status="requires_validation"),
    FeatureDef("peak_abs_yaw_rate", "C", "event", "peak |yaw rate| during a turn", "max |yaw_rate|", "deg/s",
               "yaw_rate", _E, "turn", "null for other types", "smoothing sensitivity", "—", _POL),
    FeatureDef("heading_change_deg", "C", "event",
               "turn: Σ yaw_rate·Δt (signed); reversal: wrapped heading_out − heading_in", "see definition", "deg",
               "yaw_rate / heading", _E, "turn, reversal", "null for other types", "|value| ≥ rule.min_change",
               "reversal angle is a two-point estimate", _POL),
    FeatureDef("median_radius_yds", "C", "event", "median turning radius during a turn", "median(speed / |ω|)", "yd",
               "speed, yaw_rate", _E, "turn", "null for other types", "—", "—", _POL, status="requires_validation"),
    FeatureDef("max_abs_curvature", "C", "event", "maximum |curvature| during a turn", "max |curvature|", "1/yd",
               "curvature", _E, "turn", "null for other types", "—", "unstable near min speed", _POL,
               status="experimental"),
    FeatureDef("direction", "C", "event", "rotation sense in the tracking frame (ccw / cw)", "sign(yaw_rate)", "label",
               "yaw_rate", _E, "turn, reversal", "null for braking/acceleration", "—",
               "handedness of Combine coordinates unverified", _POL, status="requires_validation"),
    FeatureDef("touches_segment_edge", "Q", "event", "event starts or ends on a segment boundary (may be truncated)",
               "pos = 0 or pos = seg_len − 1", "bool", "—", _E, _BOTH, "—", "—", "—", _POL),
)

# ======================================================================================
# Phase 03 · M2 — attempt / COD / sprint features (``features/{temporal,acceleration,braking,
# agility,efficiency,sprint}.py``). Version = M2_VERSION. Window-based features use ACTIVE frames
# (movement onset → end of activity, see temporal.py).
# ======================================================================================
from .config import M2_VERSION  # noqa: E402

_A = "attempt (Combine event_id, nfl_id, attempt) — also NFL player-play in M3"
_C = "COD event (trajectory keys + seg_idx + cod_idx)"
_SP = "sprint attempt (FORTY_YARD_DASH attempt)"
_ALL = "all drills (comparable WITHIN a drill only)"
_AG = "drills with direction changes (shuttle, three-cone, position drills)"
_ACT = "active frames (onset → end of activity)"


def _m2(name: str, family: str, level: str, definition: str, formula: str, unit: str, src: str, ctx: str,
        missing: str, qc: str, lim: str, status: str = "core", grain: str = _A,
        cost: str = "O(frames), Polars group_by") -> FeatureDef:
    return FeatureDef(name, family, level, definition, formula, unit, src, grain, ctx, missing, qc, lim, cost,
                      status, M2_VERSION)


_W = "null if no onset (athlete never reaches onset_speed for onset_frames)"
M2_REGISTRY: tuple[FeatureDef, ...] = (
    # ---------------------------------------------------------------- movement window (E)
    _m2("t_first_s", "E", "attempt", "time of first valid frame", "min t_s (valid_kin)", "s", "t_s", _ALL, "—", "—", "—"),
    _m2("t_last_s", "E", "attempt", "time of last valid frame", "max t_s (valid_kin)", "s", "t_s", _ALL, "—", "—", "—"),
    _m2("t_onset_s", "E", "attempt", "movement onset", "first frame starting onset_frames consecutive frames with speed ≥ onset_speed",
        "s", "speed", _ALL, _W, "onset_censored share by drill", "threshold-defined; not the official start"),
    _m2("t_onset_fast_s", "E", "attempt", "fast onset (robust to slow pre-start drift)",
        "as t_onset_s with onset_speed_fast", "s", "speed", _ALL, _W, "compare with t_onset_s", "later than true start"),
    _m2("t_end_active_s", "E", "attempt", "end of activity", "last frame with speed ≥ onset_speed", "s", "speed", _ALL,
        _W, "—", "recording may stop before the athlete stops"),
    _m2("first_speed", "Q", "attempt", "derived speed on the first valid frame", "speed[first]", "yd/s", "x, y", _ALL,
        "—", "> onset_speed ⇒ start not recorded", "—"),
    _m2("last_speed", "Q", "attempt", "derived speed on the last valid frame", "speed[last]", "yd/s", "x, y", _ALL,
        "—", "high ⇒ recording cut while moving", "—"),
    _m2("onset_censored", "Q", "attempt", "athlete already moving on the first valid frame",
        "t_onset_s ≤ t_first_s", "bool", "speed", _ALL, "—", "share by drill/cohort", "—"),
    _m2("end_censored", "Q", "attempt", "recording stops while the athlete still moves fast (truncated end)",
        "last_speed ≥ end_censor_speed (2 yd/s)", "bool", "speed", _ALL, "—",
        "expected ≈ 1 for the forty (stops at the finish)", "durations / time-to-peak are lower bounds"),
    _m2("active_duration_s", "E", "attempt", "duration of the movement window", "t_end_active − t_onset", "s", "speed",
        _ALL, _W, "vs official drill time", "window-definition dependent", "requires_validation"),
    _m2("onset_to_end_s", "E", "attempt", "onset to end of recording", "t_last − t_onset", "s", "speed", _ALL, _W,
        "vs official time (forty_timing_validation.csv)", "start/stop convention ≠ official timing", "requires_validation"),
    _m2("fast_onset_to_end_s", "E", "attempt", "fast onset to end of recording", "t_last − t_onset_fast", "s", "speed",
        _ALL, _W, "vs official time", "systematically shorter than true duration", "requires_validation"),
    _m2("last_speed_frac_of_max", "Q", "attempt", "last-frame speed relative to peak speed", "last_speed / max speed",
        "fraction", "speed", _ALL, "—", "≈1 ⇒ window cut at peak (forty)", "—"),
    # ---------------------------------------------------------------- temporal phases (E)
    *[_m2(f"frac_time_{p}", "E", "attempt", f"share of active frames labelled {p}", f"mean(phase == '{p}')", "fraction",
          "speed, a_tan", _ALL, _W, "phase_accel sensitivity", "PROVISIONAL phase thresholds", "requires_validation")
      for p in ("stationary", "accelerating", "steady", "decelerating")],
    *[_m2(f"a_tan_mean_{n}", "E", "attempt", f"mean tangential acceleration in the {n} third of the active window",
          "mean a_tan over t_rel ∈ third", "yd/s²", "a_tan", _ALL, _W, "—", "thirds are time-relative, not distance-relative")
      for n in ("early", "mid", "late")],
    _m2("n_phase_transitions", "E", "attempt", "changes between stable movement phases",
        "count of phase changes among runs ≥ min_phase_frames", "count", "speed, a_tan", _ALL, _W, "—",
        "depends on phase thresholds", "requires_validation"),
    _m2("phase_transition_rate", "D", "attempt", "phase transitions per second (smoothness proxy; lower = smoother)",
        "n_phase_transitions / active_duration_s", "1/s", "speed, a_tan", _ALL, _W, "—",
        "proxy only; not a validated smoothness metric (no jerk used)", "experimental"),
    _m2("peak_accel_lead_s", "E", "attempt", "time from peak a_tan to peak speed", "t(peak speed) − t(peak a_tan)", "s",
        "speed, a_tan", _ALL, _W, "—", "censored if peak speed at window end"),
    _m2("late_speed_maintenance", "E", "attempt", "mean speed in last third relative to peak speed",
        "mean speed (late third) / max speed", "fraction", "speed", _ALL, _W, "—", "window-definition dependent"),
    # ---------------------------------------------------------------- speed (A)
    _m2("n_active_frames", "Q", "attempt", "frames in the movement window", "count", "frames", "speed", _ALL, "—", "—", "—"),
    _m2("speed_mean", "A", "attempt", "mean derived speed", "mean speed", "yd/s", "x, y", _ALL, _W, "—", _ACT),
    _m2("speed_median", "A", "attempt", "median derived speed", "median speed", "yd/s", "x, y", _ALL, _W, "—", _ACT),
    _m2("speed_p90", "A", "attempt", "90th percentile speed", "quantile(speed, .9)", "yd/s", "x, y", _ALL, _W, "—", _ACT),
    _m2("speed_p95", "A", "attempt", "95th percentile speed", "quantile(speed, .95)", "yd/s", "x, y", _ALL, _W,
        "smoothing Spearman ≥ 0.999 (M1)", _ACT),
    _m2("speed_max", "A", "attempt", "maximum derived speed in the active window", "max speed", "yd/s", "x, y", _ALL,
        _W, "≤ max_speed_yds", "single frame; forty: often at window end (see peak_speed_censored)"),
    _m2("speed_sd", "A", "attempt", "speed standard deviation", "sd speed", "yd/s", "x, y", _ALL, _W, "—", _ACT),
    _m2("speed_cv", "A", "attempt", "speed coefficient of variation", "speed_sd / speed_mean", "fraction", "x, y",
        _ALL, _W, "—", "drill-design dependent"),
    *[_m2(f"time_above_{th}yds_s", "A", "attempt", f"time with speed ≥ {th} yd/s", f"Σ(speed ≥ {th})·Δt", "s",
          "x, y", _ALL, _W, "—", "absolute threshold; position-dependent relevance") for th in (4, 6, 8)],
    _m2("time_high_speed_s", "A", "attempt", "sustained high-speed time (≥ 90 % of own peak)",
        "Σ(speed ≥ 0.9·max)·Δt", "s", "x, y", _ALL, _W, "—", "relative threshold; window-cut forty underestimates"),
    _m2("time_to_peak_speed_s", "A", "attempt", "time from onset to peak speed", "t_rel at max speed", "s", "x, y",
        _ALL, _W, "use only if NOT peak_speed_censored", "censored by recording end", "requires_validation"),
    _m2("dist_to_peak_speed_yds", "A", "attempt", "smoothed distance from onset to peak speed",
        "Σ step_smooth until max speed", "yd", "x, y", _ALL, _W, "use only if NOT peak_speed_censored",
        "censored by recording end", "requires_validation"),
    _m2("peak_speed_censored", "Q", "attempt", "peak speed within the last peak_edge_frames frames",
        "frames after peak < peak_edge_frames", "bool", "speed", _ALL, "—", "—", "—"),
    _m2("rise_time_10_90_s", "A", "attempt", "time from 10 % to 90 % of own peak speed", "t(0.9·max) − t(0.1·max)",
        "s", "speed", _ALL, _W, "—", "relative to own peak (slower athletes reach a lower target)"),
    _m2("buildup_gradient", "A", "attempt", "speed build-up gradient", "0.8·speed_max / rise_time_10_90_s", "yd/s²",
        "speed", _ALL, _W, "—", "average, not instantaneous, acceleration"),
    # ---------------------------------------------------------------- acceleration (A)
    _m2("accel_initial", "A", "attempt", "mean tangential acceleration over the first initial_window_s after onset",
        "mean a_tan, t_rel < 1.0 s", "yd/s²", "a_tan", _ALL, _W, "onset_censored attempts start mid-acceleration",
        "depends on onset definition", "requires_validation"),
    _m2("a_tan_max", "A", "attempt", "peak tangential acceleration", "max a_tan", "yd/s²", "a_tan", _ALL, _W,
        "smoothing Spearman 0.95–0.99 (M1)", "single frame; SG edge frames can bias it"),
    _m2("a_tan_p90", "A", "attempt", "90th percentile tangential acceleration", "quantile(a_tan, .9)", "yd/s²", "a_tan",
        _ALL, _W, "—", "—"),
    _m2("a_tan_p95", "A", "attempt", "95th percentile tangential acceleration", "quantile(a_tan, .95)", "yd/s²",
        "a_tan", _ALL, _W, "—", "—"),
    _m2("a_tan_pos_mean", "A", "attempt", "mean positive tangential acceleration", "mean a_tan | a_tan > 0", "yd/s²",
        "a_tan", _ALL, _W, "—", "—"),
    _m2("accel_sustained", "A", "attempt", "best rolling-mean tangential acceleration over sustain_window_s",
        "max rolling_mean(a_tan, 0.5 s)", "yd/s²", "a_tan", _ALL, _W, "—", "more robust than a_tan_max"),
    _m2("time_to_peak_accel_s", "A", "attempt", "time from onset to peak a_tan", "t_rel at max a_tan", "s", "a_tan",
        _ALL, _W, "—", "onset-definition dependent"),
    _m2("accel_time_s", "A", "attempt", "time with a_tan ≥ phase_accel", "Σ(a_tan ≥ 1.0)·Δt", "s", "a_tan", _ALL, _W,
        "—", "PROVISIONAL threshold"),
    _m2("accel_integral_pos", "A", "attempt", "integrated positive tangential acceleration (= speed gained)",
        "Σ max(a_tan, 0)·Δt", "yd/s", "a_tan", _ALL, _W, "—", "redundant with speed range in straight sprints"),
    _m2("a_tan_sd", "A", "attempt", "standard deviation of a_tan", "sd a_tan", "yd/s²", "a_tan", _ALL, _W, "—",
        "noise-inflated"),
    _m2("accel_mag_max", "A", "attempt", "peak vector-acceleration magnitude", "max |a|", "yd/s²", "x, y", _ALL, _W,
        "—", "includes turning; NOT the speed derivative"),
    _m2("accel_mag_p95", "A", "attempt", "95th percentile vector-acceleration magnitude", "quantile(|a|, .95)", "yd/s²",
        "x, y", _ALL, _W, "—", "includes turning"),
    # ---------------------------------------------------------------- braking (B)
    _m2("decel_max", "B", "attempt", "peak deceleration (frame level)", "max(−a_tan)", "yd/s²", "a_tan", _ALL, _W,
        "smoothing rel. change 6–14 % (M1)", "single frame; noise-sensitive"),
    _m2("decel_p95", "B", "attempt", "95th percentile deceleration among decelerating frames",
        "quantile(−a_tan | a_tan < 0, .95)", "yd/s²", "a_tan", _ALL, _W, "—", "—"),
    _m2("decel_mean", "B", "attempt", "mean deceleration among decelerating frames", "mean(−a_tan | a_tan < 0)",
        "yd/s²", "a_tan", _ALL, _W, "—", "includes mild slowing"),
    _m2("n_braking_events", "B", "attempt", "braking events (M1 rule)", "count", "count", "events", _ALL, "0 if none",
        "rule sensitivity (event_threshold_sensitivity.csv)", "threshold-defined"),
    _m2("braking_time_s", "B", "attempt", "total braking-event duration", "Σ duration", "s", "events", _ALL, "0 if none",
        "—", "—"),
    _m2("braking_distance_yds", "B", "attempt", "total distance travelled while braking", "Σ distance", "yd", "events",
        _ALL, "0 if none", "—", "—"),
    _m2("braking_intensity_mean", "B", "attempt", "duration-weighted mean |a_tan| over braking events",
        "Σ(mean|a_tan|·dur)/Σdur", "yd/s²", "events", _ALL, "null if no event", "—", "—"),
    _m2("braking_peak", "B", "attempt", "largest event peak deceleration", "max peak_abs_a_tan", "yd/s²", "events",
        _ALL, "null if no event", "—", "single frame"),
    _m2("braking_speed_drop_max", "B", "attempt", "largest speed loss in one braking event", "max speed_change", "yd/s",
        "events", _ALL, "null if no event", "—", "—"),
    _m2("speed_reduction_ratio", "B", "attempt", "median relative speed loss per braking event",
        "median(speed_change / entry_speed)", "fraction", "events", _ALL, "null if no event", "—", "—"),
    _m2("braking_speed_loss_per_yd", "B", "attempt", "braking efficiency proxy: speed lost per yard while braking",
        "median(speed_change / distance)", "(yd/s)/yd", "events", _ALL, "null if no event", "—",
        "kinematic ratio, not force/energy", "requires_validation"),
    _m2("time_to_reaccel_median_s", "B", "attempt", "median braking → re-acceleration time",
        "median time_to_transition_s", "s", "events", _ALL, "null if never", "—", "censored by window end",
        "requires_validation"),
    _m2("frac_braking_with_reaccel", "B", "attempt", "share of braking events followed by re-acceleration",
        "mean(time_to_transition_s not null)", "fraction", "events", _ALL, "null if no event", "—", "—"),
    # ---------------------------------------------------------------- agility (C)
    _m2("yaw_integral_deg", "C", "attempt", "cumulative absolute heading change while moving ≥ min_heading_speed",
        "Σ|yaw_rate|·Δt", "deg", "x, y", _AG, _W, "smoothing sensitivity", "excludes low-speed reversals (see n_cod)"),
    _m2("abs_yaw_rate_p95", "C", "attempt", "95th percentile |yaw rate|", "quantile(|yaw|, .95)", "deg/s", "x, y", _AG,
        _W, "—", "undefined below 1 yd/s"),
    _m2("a_lat_abs_p95", "C", "attempt", "95th percentile |lateral acceleration|", "quantile(|a_lat|, .95)", "yd/s²",
        "x, y", _AG, _W, "—", "sign unused (handedness unverified)"),
    _m2("n_cod", "C", "attempt", "changes of direction ≥ cod_min_angle (merged turn/reversal events)", "count", "count",
        "events", _AG, "0 if none", "vs drill design (shuttle: 2 reversals expected)", "PROVISIONAL angle", "requires_validation"),
    _m2("n_cod_sharp", "C", "attempt", "CODs ≥ cod_sharp_angle", "count", "count", "events", _AG, "0", "—", "—"),
    _m2("n_cod_reversal", "C", "attempt", "CODs containing a low-speed reversal", "count", "count", "events", _AG, "0",
        "—", "—"),
    _m2("cod_angle_total_deg", "C", "attempt", "summed |angle| of CODs", "Σ cod_angle_deg", "deg", "events", _AG, "0",
        "—", "—"),
    _m2("cod_approach_speed_median", "C", "attempt", "median COD approach speed", "median approach_speed", "yd/s",
        "speed", _AG, "null if no COD", "—", "—"),
    _m2("cod_approach_decel_median", "C", "attempt", "median COD approach deceleration", "median approach_decel",
        "yd/s²", "a_tan", _AG, "null if no COD", "—", "—"),
    _m2("cod_exit_accel_median", "C", "attempt", "median COD exit acceleration", "median exit_accel", "yd/s²", "a_tan",
        _AG, "null if no COD", "—", "SG edge effects near reversal minimum"),
    _m2("cod_time_median_s", "C", "attempt", "median COD time (below 80 % of approach speed)", "median cod_time_s", "s",
        "speed", _AG, "null if no recovered COD", "—", "censored CODs excluded", "requires_validation"),
    _m2("turn_speed_retention_median", "C", "attempt", "median min/approach speed over turn-type CODs",
        "median speed_retention | cod_type = turn", "fraction", "speed", _AG, "null if no turn COD", "—",
        "reversals excluded (≈ 0 by construction)"),
    _m2("turn_radius_median_yds", "C", "attempt", "median turning radius over CODs", "median turn_radius_yds", "yd",
        "speed, yaw_rate", _AG, "null if no turn", "—", "unstable at low speed", "requires_validation"),
    # ---------------------------------------------------------------- efficiency (D)
    _m2("path_len_yds", "D", "attempt", "smoothed path length in the active window", "Σ step_smooth", "yd", "x, y",
        _ALL, _W, "smoothing Spearman ≈ 1 (M1)", "—"),
    _m2("net_disp_yds", "D", "attempt", "net displacement onset → end", "|p_end − p_onset|", "yd", "x, y", _ALL, _W,
        "—", "—"),
    _m2("path_efficiency", "D", "attempt", "net / path (directness)", "net_disp / path_len", "fraction", "x, y", _ALL,
        "null if path < 1 yd", "—", "TASK-SPECIFIC: compare within a drill only"),
    _m2("excess_path_yds", "D", "attempt", "path length beyond the straight line", "path − net", "yd", "x, y", _ALL, _W,
        "—", "task-specific"),
    _m2("axis_anisotropy", "Q", "attempt", "elongation of the path (principal-axis validity)", "λ1 / (λ1 + λ2)",
        "fraction", "x, y", _ALL, _W, "exclude < 0.7 from lateral features", "—"),
    _m2("lateral_extent_yds", "D", "attempt", "range of positions across the principal axis", "max w − min w", "yd",
        "x, y", _ALL, _W, "—", "axis ill-defined when anisotropy low", "requires_validation"),
    _m2("long_extent_yds", "D", "attempt", "range of positions along the principal axis", "max u − min u", "yd", "x, y",
        _ALL, _W, "—", "—"),
    _m2("lateral_speed_p95", "C", "attempt", "95th percentile speed across the principal axis", "quantile(|v_lat|, .95)",
        "yd/s", "x, y", _ALL, _W, "—", "axis ill-defined when anisotropy low", "requires_validation"),
    _m2("frac_time_lateral", "C", "attempt", "share of frames moving more across than along the axis",
        "mean(|v_lat| > |v_long|)", "fraction", "x, y", _ALL, _W, "—", "—", "requires_validation"),
    # ---------------------------------------------------------------- COD event level (C)
    _m2("cod_idx", "C", "cod", "COD order within the trajectory", "rank by t_start", "index", "events", _AG, "—", "—",
        "—", grain=_C),
    _m2("cod_type", "C", "cod", "reversal if any merged event is a reversal, else turn", "label", "label", "events",
        _AG, "—", "—", "—", grain=_C),
    _m2("n_merged_events", "Q", "cod", "M1 events merged into the COD", "count", "count", "events", _AG, "—", "—", "—",
        grain=_C),
    _m2("cod_angle_deg", "C", "cod", "COD angle (sum of |heading changes| of merged events)", "Σ|Δheading|", "deg",
        "events", _AG, "—", "≥ cod_min_angle", "reversal angle is a two-point estimate", grain=_C),
    _m2("approach_speed", "C", "cod", "max speed in the approach window", "max speed, [t_start − 1 s, t_start]", "yd/s",
        "speed", _AG, "null at segment start", "—", "—", grain=_C),
    _m2("approach_decel", "B", "cod", "mean deceleration before the COD", "mean −a_tan, [t_start − 0.5 s, t_start)",
        "yd/s²", "a_tan", _AG, "null at segment start", "—", "—", grain=_C),
    _m2("approach_decel_peak", "B", "cod", "peak deceleration before the COD", "max −a_tan, same window", "yd/s²",
        "a_tan", _AG, "null at segment start", "—", "single frame", grain=_C),
    _m2("exit_accel", "A", "cod", "mean acceleration after the COD", "mean a_tan, (t_end, t_end + 0.5 s]", "yd/s²",
        "a_tan", _AG, "null at segment end", "—", "—", grain=_C),
    _m2("exit_speed_1s", "A", "cod", "max speed within 1 s after the COD", "max speed, (t_end, t_end + 1 s]", "yd/s",
        "speed", _AG, "null at segment end", "—", "recording may stop", grain=_C),
    _m2("speed_retention", "C", "cod", "min / approach speed", "min_speed / approach_speed", "fraction", "speed", _AG,
        "null if approach speed 0", "—", "≈ 0 for reversals by construction", grain=_C),
    _m2("cod_time_s", "C", "cod", "time below 80 % of approach speed around the COD", "t_recover − t_loss", "s",
        "speed", _AG, "null if not recovered within 2 s (censored)", "—", "PROVISIONAL fraction", "requires_validation",
        grain=_C),
    _m2("turn_radius_yds", "C", "cod", "median turning radius of merged turn events", "median(speed/|ω|)", "yd",
        "speed, yaw_rate", _AG, "null for pure reversals", "—", "—", "requires_validation", grain=_C),
    _m2("touches_segment_edge", "Q", "cod", "COD touches a segment boundary", "any merged event at edge", "bool", "—",
        _AG, "—", "—", "—", grain=_C),
    # ---------------------------------------------------------------- sprint (A/E)
    _m2("n_frames", "Q", "sprint", "frames in the longest segment", "count", "frames", "—", "forty", "—", "—", "—", grain=_SP),
    _m2("n_fit_frames", "Q", "sprint", "frames used in the model fit (one frame before onset → peak speed)", "count", "frames", "—",
        "forty", "—", "≥ min_fit_frames", "—", grain=_SP),
    _m2("first_speed", "Q", "sprint", "speed on the first valid frame", "speed[0]", "yd/s", "x, y", "forty", "—",
        "> 0.5 ⇒ start not recorded", "—", grain=_SP),
    _m2("last_speed", "Q", "sprint", "speed on the last valid frame", "speed[-1]", "yd/s", "x, y", "forty", "—", "—",
        "—", grain=_SP),
    _m2("speed_max_obs", "A", "sprint", "observed peak derived speed", "max speed", "yd/s", "x, y", "forty", "—", "—",
        "censored if peak_at_end", grain=_SP),
    _m2("peak_at_end", "Q", "sprint", "peak within the last frames (window cut while accelerating)",
        "frames after peak < peak_edge_frames", "bool", "speed", "forty", "—", "—", "—", grain=_SP),
    _m2("run_distance_yds", "Q", "sprint", "distance along the run axis from model start to last frame",
        "track_distance + pre_distance", "yd", "x, y", "forty", "—", "≈ 40 if window = timed run", "—", grain=_SP),
    _m2("track_distance_yds", "Q", "sprint", "tracked distance along the run axis", "(p_last − p_first)·u", "yd",
        "x, y", "forty", "—", "—", "—", grain=_SP),
    _m2("pre_distance_yds", "Q", "sprint", "model-extrapolated distance before the first frame", "d_model(t_first)",
        "yd", "model", "forty", "0 if not censored", "should be small (< 1 yd)", "extrapolation", grain=_SP),
    _m2("model_vmax", "A", "sprint", "mono-exponential asymptotic maximal speed", "LS fit v = vmax(1 − e^{−(t−t0)/τ})",
        "yd/s", "speed", "forty", "null if fit fails", "model_r2, rmse", "biased low if window cut early", grain=_SP),
    _m2("model_tau_s", "A", "sprint", "acceleration time constant (lower = quicker approach to vmax)", "fitted τ", "s",
        "speed", "forty", "null if fit fails", "fit_at_grid_edge", "model misfit near the start from stance", grain=_SP),
    _m2("model_t0_rel_s", "E", "sprint", "model start relative to the first frame (negative = before recording)",
        "t0 − t_first", "s", "speed", "forty", "—", "—", "≠ official start", "requires_validation", grain=_SP),
    _m2("model_a0", "A", "sprint", "model initial acceleration", "vmax / τ", "yd/s²", "speed", "forty",
        "null if fit fails", "—", "theoretical value at t0; no force/power implied", grain=_SP),
    _m2("model_rmse", "Q", "sprint", "fit RMSE", "sqrt(SSE/n)", "yd/s", "speed", "forty", "—", "—", "—", grain=_SP),
    _m2("model_r2", "Q", "sprint", "fit R²", "1 − SSE/SST", "fraction", "speed", "forty", "—", "—", "—", grain=_SP),
    _m2("fit_at_grid_edge", "Q", "sprint", "optimum on the τ/t0 grid boundary", "edge test", "bool", "—", "forty", "—",
        "exclude if true", "—", grain=_SP),
    _m2("fit_ok", "Q", "sprint", "fit usable (enough frames, interior optimum, vmax > 0)", "rule", "bool", "—", "forty",
        "—", "—", "—", grain=_SP),
    *[_m2(f"trk_t{d}_s", "A", "sprint",
          f"TRACKED time from model start to {d} yd along the run axis (not an official Combine time)",
          "linear interpolation", "s", "x, y", "forty", "null if run shorter than split − tol",
          f"bias vs official {'cmb_ten_yd_split' if d == 10 else 'cmb_forty' if d == 40 else '— (no official 20-yd)'} "
          "in timing_validation.csv", "model-start convention ≠ official timing", "requires_validation", grain=_SP)
      for d in (10, 20, 40)],
    _m2("trk_t10_20_s", "A", "sprint", "TRACKED flying 10→20 yd interval", "trk_t20 − trk_t10", "s", "x, y", "forty",
        "null if split missing", "—", "start-convention free", grain=_SP),
    _m2("trk_t20_40_s", "A", "sprint", "TRACKED flying 20→40 yd interval", "trk_t40 − trk_t20", "s", "x, y", "forty",
        "null if split missing", "—", "start-convention free; may be extrapolated ≤ 1 yd (split_extrapolated)",
        grain=_SP),
    _m2("fit_start_rel_s", "Q", "sprint", "start of the model-fit window relative to the first frame",
        "t(one frame before speed ≥ onset_speed) − t_first", "s", "speed", "forty", "—",
        "> 0 ⇒ stationary lead-in excluded from the fit", "—", grain=_SP),
    _m2("vmax_extrapolation_ratio", "Q", "sprint", "model vmax relative to the observed peak speed",
        "model_vmax / speed_max_obs", "fraction", "speed", "forty", "null if fit undefined",
        "fit_ok requires ≤ 1.10", "> 1 when the recording stops before the plateau", grain=_SP),
    *[_m2(f"speed_at_{d}", "A", "sprint", f"derived speed when crossing {d} yd", "interp speed", "yd/s", "x, y", "forty",
          "null if split missing", "—", "—", grain=_SP) for d in (10, 20, 40)],
    _m2("dist_to_90pct_vmax_yds", "A", "sprint", "distance at which 90 % of model vmax is first reached",
        "total distance at first speed ≥ 0.9·vmax", "yd", "x, y", "forty", "null if never / fit not ok", "—", "—",
        grain=_SP),
    _m2("split_extrapolated", "Q", "sprint", "a split was extrapolated past the last frame (≤ tolerance)", "flag",
        "bool", "—", "forty", "—", "—", "—", grain=_SP),
    _m2("run_incomplete", "Q", "sprint", "run shorter than 40 yd − tolerance", "flag", "bool", "—", "forty", "—", "—",
        "—", grain=_SP),
)

REGISTRY = REGISTRY + M2_REGISTRY

# Direction for "best attempt" aggregation and reliability (feature → higher is better).
# Features absent here have no defensible "better" direction; their "best" is left null.
HIGHER_IS_BETTER: dict[str, bool] = {
    **{c: True for c in ("speed_max", "speed_p95", "speed_p90", "speed_mean", "accel_initial", "a_tan_max", "a_tan_p95",
                         "a_tan_p90", "accel_sustained", "buildup_gradient", "model_vmax", "model_a0", "speed_max_obs",
                         "speed_at_10", "speed_at_20", "speed_at_40", "time_high_speed_s", "cod_exit_accel_median",
                         "cod_approach_speed_median", "turn_speed_retention_median", "decel_max", "decel_p95",
                         "braking_peak", "braking_intensity_mean", "braking_speed_loss_per_yd")},
    **{c: False for c in ("trk_t10_s", "trk_t20_s", "trk_t40_s", "trk_t10_20_s", "trk_t20_40_s", "model_tau_s",
                          "rise_time_10_90_s",
                          "onset_to_end_s", "fast_onset_to_end_s", "cod_time_median_s", "time_to_reaccel_median_s",
                          "dist_to_90pct_vmax_yds")},
}

# Keys / context / bookkeeping columns documented for dataset sidecars (not features).
CONTEXT_DOCS: dict[str, str] = {
    "event_id": "Combine event (session) id", "nfl_id": "athlete id", "attempt": "attempt number",
    "draft_year": "Combine / draft cohort", "drill_type": "drill family (raw)", "drill_name": "drill (raw spelling)",
    "drill": "canonical drill name (DRILL_ALIASES applied; inferred renames)",
    "season": "NFL season", "game_id": "game id", "play_id": "play id",
    "seg_idx": "segment number within trajectory (1-based)", "pos": "frame position within segment (0-based)",
    "seg_len": "frames in the segment", "frame_id": "silver frame_id", "ts": "frame timestamp",
    "t_s": "seconds since the trajectory's first valid frame (s)", "dt_ms": "Δt to previous frame of the segment (ms)",
    "step_yds": "raw displacement from previous frame of the segment (yd)",
    "break_reason": "why the segment starts here: start | gap | jump | drill_change",
    "x": "raw x (yd)", "y": "raw y (yd)", "s": "vendor speed (yd/s)", "a": "vendor acceleration magnitude (yd/s²)",
    "dir": "vendor direction of motion (deg, convention unverified)", "o": "vendor orientation (deg)",
    "valid_kin": "derivatives defined on this frame", "heading_valid": "heading/yaw/curvature defined on this frame",
    "start_ts": "segment start timestamp", "t_start_s": "start time since trajectory start (s)",
    "t_end_s": "end time since trajectory start (s)", "event_type": "braking | acceleration | turn | reversal",
    "rule": "threshold set (lenient | default | strict)", "event_idx": "event order within segment and rule",
    "start_frame_id": "first frame_id of the event", "end_frame_id": "last frame_id of the event",
    "min_speed": "minimum speed in the event (yd/s)", "max_speed": "maximum derived speed (yd/s)",
    "n_segments_seen": "segments summarised", "n_smoothable_segments": "segments ≥ SG window",
    "frames_smoothable": "frames inside smoothable segments", "longest_segment_frames": "frames in longest segment",
    "smoothing": "SG setting label", "n_frames": "frames",
    # Phase 03 · M2
    "attempt_order": "chronological order of the attempt within athlete × canonical drill (1 = first)",
    "n_attempts_drill": "usable attempts of this athlete in this canonical drill",
    "position_group": "position group (combine_position, else nfl_position) from gold.dim_athlete",
    "quality_tier": "M1 trajectory quality tier", "n_segments": "segments in the trajectory",
    "n_segments_used": "segments used (longest only)", "feature": "feature name (see feature_dictionary.csv)",
    "n_attempts": "usable attempts aggregated", "median": "median over attempts", "mean": "mean over attempts",
    "sd": "SD over attempts (null if 1 attempt)", "min": "minimum over attempts", "max": "maximum over attempts",
    "best": "best attempt value (direction from HIGHER_IS_BETTER; null if no defensible direction)",
    "first": "value of the chronologically first attempt",
    "resid_rms_yds": "M1 position-residual RMS of the attempt (yd) — tracking-noise proxy (QC, not a feature)",
}


def registry_frame() -> pl.DataFrame:
    return pl.DataFrame([asdict(f) for f in REGISTRY]).sort("level", "family", "feature_name")


def registered_names(level: str | None = None) -> set[str]:
    return {f.feature_name for f in REGISTRY if level is None or f.level == level}


def column_docs(columns: list[str] | tuple[str, ...], level: str | None = None) -> dict[str, str]:
    """Column → description (with unit) for dataset sidecars; unknown columns get ''."""
    by_name: dict[str, FeatureDef] = {}
    for f in REGISTRY:
        if level is None or f.level == level:
            by_name.setdefault(f.feature_name, f)
    out = {}
    for c in columns:
        if c in by_name:
            f = by_name[c]
            out[c] = f"{f.definition} [{f.unit}] ({f.status})"
        else:
            out[c] = CONTEXT_DOCS.get(c, "")
    return out
