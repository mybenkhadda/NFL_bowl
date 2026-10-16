"""Stage ATT — one analytical row per forty-yard-dash attempt, built from persisted datasets only.

Sources (nothing is recomputed upstream):
    phase_03 combine_sprint_profiles      keys, cohort, position, attempt order, existing model fit + flags
    phase_03 combine_attempt_features     accel_sustained, accel_initial, resid_rms_yds
    phase_03 combine_frame_kinematics     recording-window descriptors, speed-band accelerations, model refits
    phase_04 trajectory_features_by_smoothing (primary SG setting)   peak / p95 speed and acceleration, path length
    phase_04 trajectory_quality_flags     structural / signal / boundary-peak flags (all kept, none applied)
    gold mart_athlete_research_base       official forty and 10-yard split

Recording-window variables describe the RECORDING (where it starts and stops), not the athlete. They are the
candidate explanation of poor acceleration repeatability that Study A tests.

Note on ``end_censored``: the Phase 03 flag (last-frame speed ≥ 2 yd/s) is true for every forty attempt
(``forty_window_diagnostics.csv``: share 1.0), so it cannot explain differences between attempts. The continuous
``last_speed_frac_of_peak`` and ``frames_after_peak`` carry the end-of-recording information instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ...features.config import MovementConfig
from ..contracts import TableContract, cast_keys
from ..studies.quality import traj_id_expr
from .config import KEYS, P04_FEATURES, RescueConfig, band_name
from .sprint_model import finalize_fits, fit_attempt

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
BANDS = tuple(band_name(lo, hi) for lo, hi in RescueConfig().speed_bands)
FRAME_COLUMNS = ["event_id", "nfl_id", "attempt", "drill", "seg_idx", "pos", "seg_len", "t_s", "x_s", "y_s", "speed",
                 "a_tan", "valid_kin"]
WINDOW_VARS = ("first_speed", "last_speed", "duration_s", "path_smooth_yds", "net_disp_yds", "last_speed_frac_of_peak",
               "fit_start_rel_s")
FLAG_VARS = ("onset_censored", "end_censored", "peak_at_end", "boundary_peak", "structural_flag", "signal_flag",
             "fit_ok", "weakly_identified", "band_start_censored")
DIFF_VARS = ("first_speed", "last_speed", "duration_s", "path_smooth_yds", "net_disp_yds", "fit_start_rel_s",
             "p95_a_tan", "peak_a_tan", "p95_speed", "accel_sustained", "accel_initial", *BANDS, "model_a0",
             "model_tau_s", "model_vmax", "model_a_at_ref", "model_rmse")

ATTEMPTS = TableContract(
    "attempt_censoring_diagnostics", "Forty-yard-dash attempts: recording window, kinematic summaries, model fit, flags.",
    {"event_id": S, "nfl_id": I, "attempt": I, "traj_id": S, "draft_year": I, "position_group": S, "attempt_order": I,
     "n_attempts": I, "quality_tier": S, "cmb_forty": F, "cmb_ten_yd_split": F,
     "first_speed": F, "last_speed": F, "duration_s": F, "path_smooth_yds": F, "net_disp_yds": F, "first_a_tan": F,
     "first_a_tan_interior": F, "frames_after_peak": I, "last_speed_frac_of_peak": F,
     "peak_speed": F, "peak_speed_interior": F, "p95_speed": F, "peak_a_tan": F, "peak_a_tan_interior": F,
     "p95_a_tan": F, "peak_decel": F, "p95_decel": F, "accel_sustained": F, "accel_initial": F,
     **{b: F for b in BANDS},
     "onset_censored": B, "end_censored": B, "peak_at_end": B, "boundary_peak": B, "structural_flag": B,
     "signal_flag": B, "band_start_censored": B, "resid_rms_yds": F, "smoothing": S,
     "fit_ok": B, "fit_at_grid_edge": B, "n_fit_frames": I, "fit_start_rel_s": F, "model_rmse": F, "model_r2": F,
     "vmax_extrapolation_ratio": F, "model_t0_rel_s": F, "pre_distance_yds": F, "model_vmax": F, "model_tau_s": F,
     "model_a0": F, "model_a_at_ref": F, "tau_rel_width": F, "weakly_identified": B,
     "S0_all": B, "S1_structural": B, "S2_signal": B, "S3_no_boundary": B},
    key=("event_id", "nfl_id", "attempt"), area="attempt (event_id, nfl_id, attempt), forty-yard dash",
    docs={"attempt_order": "chronological order within the athlete (1 = first), from Phase 03",
          "first_speed": "derived speed on the first valid frame of the longest segment [yd/s]",
          "duration_s": "last − first valid frame of that segment [s]",
          "path_smooth_yds": "smoothed path length of the recording (Phase 04, primary SG) [yd]",
          "net_disp_yds": "straight-line distance first → last smoothed position [yd]",
          "first_a_tan": "tangential acceleration on the first valid frame (an SG filter-boundary frame) [yd/s²]",
          "first_a_tan_interior": "tangential acceleration on the first filter-interior frame [yd/s²]",
          "frames_after_peak": "frames recorded after the peak-speed frame (0 = recording ends at the peak)",
          "onset_censored": "Phase 03: athlete already moving on the first valid frame",
          "end_censored": "Phase 03: last-frame speed ≥ end_censor_speed (true for every forty attempt)",
          "peak_at_end": "Phase 03: peak speed within the last peak_edge_frames frames",
          "boundary_peak": "Phase 04 Q: peak a_tan or peak deceleration on an SG filter-boundary frame",
          "band_start_censored": "recording starts above the lowest speed-band level (band accelerations null)",
          "model_vmax": "existing Phase 03 fit; null unless fit_ok (as persisted)",
          "model_a_at_ref": "base refit: (vmax − model_ref_speed) / τ, null unless the base refit is fit_ok",
          "tau_rel_width": "base refit: relative width of the near-optimal τ range (identifiability index)",
          "S1_structural": "scenario flags are nested as in Phase 04 Study Q: S1 ⊂ S0, S2 ⊂ S1, S3 ⊂ S2",
          **{b: "mean acceleration between the two speed levels in the name [yd/s²]; null when not observed"
             for b in BANDS}})
WINDOWS = TableContract(
    "recording_window_distribution", "Distribution of recording-window variables and flags by position and cohort.",
    {"group_type": S, "group": S, "variable": S, "kind": S, "n": I, "mean": F, "sd": F, "p05": F, "p25": F, "p50": F,
     "p75": F, "p95": F}, key=("group_type", "group", "variable"), area="stratum × variable",
    docs={"kind": "continuous, or flag (mean = share of attempts with the flag set)"})
WITHIN = TableContract(
    "within_athlete_window_variation", "Attempt 2 − attempt 1 differences per athlete (first two chronological attempts).",
    {"nfl_id": I, "position_group": S, "draft_year": I, "n_attempts": I, "censor_pattern": S,
     "any_boundary_peak": B, "both_signal_pass": B, "mean_resid_rms_yds": F, "first_speed_1": F, "first_speed_2": F,
     **{f"d_{v}": F for v in DIFF_VARS}}, key=("nfl_id",), area="athlete with ≥ 2 forty attempts",
    docs={"censor_pattern": "onset_censored on the two attempts: both_uncensored / both_censored / discordant",
          "d_first_speed": "every d_* column is attempt 2 minus attempt 1"})
EXAMPLES = TableContract(
    "example_signals", "Speed and acceleration of deterministic example attempts with the base model curve.",
    {"traj_id": S, "label": S, "t_rel_s": F, "speed": F, "a_tan": F, "model_speed": F, "in_fit_window": B},
    key=("traj_id", "t_rel_s"), kind="dataset", area="frame of an example attempt")
CONTRACTS = (ATTEMPTS, WINDOWS, WITHIN, EXAMPLES)


# ============================================================================== frames
def longest_segments(frames: pl.DataFrame, drill: str) -> pl.DataFrame:
    """Valid frames of the longest segment of every attempt, as ``features.sprint.sprint_profiles`` selects them."""
    f = cast_keys(frames).filter(pl.col("valid_kin") & (pl.col("drill") == drill))
    if f.is_empty():
        return f
    longest = (f.group_by(*KEYS, "seg_idx").agg(pl.len().alias("_n"))
               .sort([*KEYS, "_n", "seg_idx"], descending=[False, False, False, True, False])
               .group_by(KEYS, maintain_order=True).first().select(*KEYS, "seg_idx"))
    return f.join(longest, on=[*KEYS, "seg_idx"], how="inner").sort([*KEYS, "t_s"])


def first_crossing(t: np.ndarray, v: np.ndarray, level: float) -> float:
    """Time at which speed first reaches ``level`` (linear interpolation). NaN when the series starts at or
    above the level (the crossing was not recorded) or never reaches it."""
    idx = np.nonzero(v >= level)[0]
    if not idx.size or idx[0] == 0:
        return NAN
    k = int(idx[0])
    return float(t[k - 1] + (level - v[k - 1]) / (v[k] - v[k - 1]) * (t[k] - t[k - 1]))


def band_acceleration(t: np.ndarray, v: np.ndarray, lo: float, hi: float) -> float:
    """(hi − lo) / (time between the first crossings of lo and hi): mean acceleration between two speeds."""
    t_lo, t_hi = first_crossing(t, v, lo), first_crossing(t, v, hi)
    if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or t_hi <= t_lo:
        return NAN
    return float((hi - lo) / (t_hi - t_lo))


def window_features(t: np.ndarray, v: np.ndarray, a: np.ndarray, x: np.ndarray, y: np.ndarray, pos: np.ndarray,
                    seg_len: np.ndarray, half: int, rcfg: RescueConfig) -> dict[str, Any]:
    """Recording-window descriptors and speed-band accelerations of one attempt (sorted valid frames)."""
    ip = int(np.argmax(v))
    interior = np.nonzero((pos >= half) & (seg_len - 1 - pos >= half) & np.isfinite(a))[0]
    out = {"first_speed": float(v[0]), "last_speed": float(v[-1]), "duration_s": float(t[-1] - t[0]),
           "net_disp_yds": float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
           "first_a_tan": float(a[0]) if np.isfinite(a[0]) else None,
           "first_a_tan_interior": float(a[interior[0]]) if interior.size else None,
           "frames_after_peak": int(v.size - 1 - ip),
           "last_speed_frac_of_peak": float(v[-1] / v[ip]) if v[ip] > 0 else None,
           "band_start_censored": bool(v[0] >= min(lo for lo, _ in rcfg.speed_bands))}
    for lo, hi in rcfg.speed_bands:
        acc = band_acceleration(t, v, lo, hi)
        out[band_name(lo, hi)] = acc if np.isfinite(acc) else None
    return out


def frame_products(frames: pl.DataFrame, mcfg: MovementConfig, rcfg: RescueConfig, half: int
                   ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(window features per attempt, model fits per attempt × variant) in ONE pass over the forty frames."""
    f = longest_segments(frames, rcfg.drill)
    feats, fits = [], []
    for part in f.partition_by(KEYS, maintain_order=True):
        key = {k: part[k][0] for k in KEYS}
        arr = {c: part[c].to_numpy().astype(float) for c in ("t_s", "speed", "a_tan", "x_s", "y_s")}
        pos, seg_len = part["pos"].to_numpy().astype(int), part["seg_len"].to_numpy().astype(int)
        feats.append({**key, **window_features(arr["t_s"], arr["speed"], arr["a_tan"], arr["x_s"], arr["y_s"], pos,
                                               seg_len, half, rcfg)})
        fits += [{**key, **r} for r in fit_attempt(arr["t_s"], arr["speed"], mcfg, rcfg)]
    schema = {"event_id": S, "nfl_id": I, "attempt": I, "first_speed": F, "last_speed": F, "duration_s": F,
              "net_disp_yds": F, "first_a_tan": F, "first_a_tan_interior": F, "frames_after_peak": I,
              "last_speed_frac_of_peak": F, "band_start_censored": B, **{b: F for b in BANDS}}
    return pl.DataFrame(feats, schema=schema), finalize_fits(fits)


# ============================================================================== attempt table
def build_attempt_table(sprint: pl.DataFrame, attempt_feats: pl.DataFrame, p04_long: pl.DataFrame,
                        p04_flags: pl.DataFrame, frame_feats: pl.DataFrame, fits: pl.DataFrame,
                        official: pl.DataFrame, rcfg: RescueConfig) -> pl.DataFrame:
    """Left joins on the attempt key: every forty attempt of ``sprint`` is kept, whatever is missing elsewhere."""
    s = cast_keys(sprint).filter(pl.col("drill") == rcfg.drill).with_columns(traj_id_expr(KEYS))
    p04 = (p04_long.filter((pl.col("domain") == "combine") & (pl.col("drill") == rcfg.drill)
                           & (pl.col("smoothing") == rcfg.smoothing))
           .group_by("traj_id").agg(*[pl.col("value").filter(pl.col("feature") == f).first().alias(f)
                                      for f in P04_FEATURES]))
    flags = (p04_flags.filter(pl.col("domain") == "combine")
             .select("traj_id", "structural_flag", "signal_flag", pl.col("boundary_flag").alias("boundary_peak")))
    af = cast_keys(attempt_feats).select(*KEYS, "accel_sustained", "accel_initial", "resid_rms_yds")
    base = fits.filter(pl.col("variant") == "base").select(
        *KEYS, pl.when(pl.col("fit_ok")).then(pl.col("a_at_ref")).alias("model_a_at_ref"), "tau_rel_width",
        "weakly_identified")
    # window descriptors are recomputed from frames; the sprint table's own first/last speed are not reused
    s = s.drop([c for c in ("first_speed", "last_speed") if c in s.columns])
    d = (s.join(cast_keys(frame_feats), on=KEYS, how="left").join(af, on=KEYS, how="left")
         .join(p04, on="traj_id", how="left").join(flags, on="traj_id", how="left")
         .join(base, on=KEYS, how="left")
         .join(cast_keys(official).select("nfl_id", "cmb_forty", "cmb_ten_yd_split"), on="nfl_id", how="left"))
    s1 = pl.col("structural_flag").fill_null(True).not_()      # a missing flag never counts as a pass
    s2 = s1 & pl.col("signal_flag").fill_null(True).not_()
    d = d.with_columns(
        pl.col("position_group").fill_null("UNKNOWN"), pl.len().over("nfl_id").cast(pl.Int64).alias("n_attempts"),
        pl.lit(rcfg.smoothing).alias("smoothing"), pl.lit(True).alias("S0_all"), s1.alias("S1_structural"),
        s2.alias("S2_signal"), (s2 & pl.col("boundary_peak").fill_null(True).not_()).alias("S3_no_boundary"))
    return d.select([pl.col(c).cast(t) for c, t in ATTEMPTS.columns.items()]).sort("nfl_id", "attempt_order", "event_id",
                                                                                 "attempt")


# ============================================================================== A1 / A2 descriptive tables
def _quantile_row(x: np.ndarray) -> dict[str, Any]:
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "mean": None, "sd": None, "p05": None, "p25": None, "p50": None, "p75": None, "p95": None}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else None,
            "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}


def strata(att: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    """(group_type, group, rows): all attempts, each position group, each cohort, each position × cohort."""
    out = [("all", "ALL", att)]
    for p in sorted(att["position_group"].unique().to_list()):
        out.append(("position", p, att.filter(pl.col("position_group") == p)))
    for y in sorted(att["draft_year"].drop_nulls().unique().to_list()):
        out.append(("cohort", str(y), att.filter(pl.col("draft_year") == y)))
    for (p, y), g in att.filter(pl.col("draft_year").is_not_null()).group_by("position_group", "draft_year",
                                                                             maintain_order=True):
        out.append(("position_cohort", f"{p}|{y}", g))
    return out


def window_distribution(att: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for gtype, name, g in strata(att):
        for var in WINDOW_VARS:
            rows.append({"group_type": gtype, "group": name, "variable": var, "kind": "continuous",
                         **_quantile_row(g[var].cast(pl.Float64).fill_null(NAN).to_numpy())})
        for var in FLAG_VARS:
            b = g[var].drop_nulls()
            rows.append({"group_type": gtype, "group": name, "variable": var, "kind": "flag", "n": b.len(),
                         "mean": float(b.mean()) if b.len() else None, "sd": None, "p05": None, "p25": None,
                         "p50": None, "p75": None, "p95": None})
    return pl.DataFrame(rows, schema=WINDOWS.columns).sort("group_type", "group", "variable")


def within_athlete_variation(att: pl.DataFrame) -> pl.DataFrame:
    """Attempt 2 − attempt 1 per athlete. Paired by construction: one row per athlete, never per frame."""
    r = att.filter(pl.col("n_attempts") >= 2).sort("nfl_id", "attempt_order")
    if not r.height:
        return WITHIN.empty()
    c1, c2 = pl.col("onset_censored").get(0), pl.col("onset_censored").get(1)
    out = r.group_by("nfl_id", maintain_order=True).agg(
        pl.col("position_group").first(), pl.col("draft_year").first(), pl.col("n_attempts").first(),
        c1.alias("_c1"), c2.alias("_c2"),
        (pl.col("boundary_peak").get(0).fill_null(False) | pl.col("boundary_peak").get(1).fill_null(False))
        .alias("any_boundary_peak"),
        (pl.col("S2_signal").get(0) & pl.col("S2_signal").get(1)).alias("both_signal_pass"),
        pl.col("resid_rms_yds").head(2).mean().alias("mean_resid_rms_yds"),
        pl.col("first_speed").get(0).alias("first_speed_1"), pl.col("first_speed").get(1).alias("first_speed_2"),
        *[(pl.col(v).get(1) - pl.col(v).get(0)).alias(f"d_{v}") for v in DIFF_VARS])
    pattern = (pl.when(pl.col("_c1").is_null() | pl.col("_c2").is_null()).then(pl.lit("unknown"))
               .when(pl.col("_c1") & pl.col("_c2")).then(pl.lit("both_censored"))
               .when(pl.col("_c1").not_() & pl.col("_c2").not_()).then(pl.lit("both_uncensored"))
               .otherwise(pl.lit("discordant")).alias("censor_pattern"))
    return out.with_columns(pattern).select([pl.col(c).cast(t) for c, t in WITHIN.columns.items()])


# ============================================================================== example recordings (figure 5)
def pick_examples(att: pl.DataFrame) -> list[tuple[str, str]]:
    """Deterministic (traj_id, label) examples: earliest and latest recording start, a recording that continues
    past the peak and one that ends on it. Chosen on window variables only — never on a feature value."""
    ok = att.filter(pl.col("S2_signal") & pl.col("fit_ok").fill_null(False) & pl.col("first_speed").is_not_null())
    if not ok.height:
        return []
    picks = [("recording starts at rest", ok.sort("first_speed", "traj_id")),
             ("recording starts in motion", ok.sort(["first_speed", "traj_id"], descending=[True, False])),
             ("recording continues past peak speed", ok.sort(["frames_after_peak", "traj_id"], descending=[True, False])),
             ("recording ends at peak speed", ok.filter(pl.col("peak_at_end").fill_null(False))
              .sort(["vmax_extrapolation_ratio", "traj_id"], descending=[True, False]))]
    out, seen = [], set()
    for label, df in picks:
        for tid in df["traj_id"].head(4).to_list():
            if tid not in seen:
                seen.add(tid)
                out.append((tid, label))
                break
    return out


def example_signals(frames: pl.DataFrame, fits: pl.DataFrame, examples: list[tuple[str, str]], rcfg: RescueConfig
                    ) -> pl.DataFrame:
    if not examples:
        return EXAMPLES.empty()
    f = longest_segments(frames, rcfg.drill).with_columns(traj_id_expr(KEYS))
    base = fits.filter(pl.col("variant") == "base").with_columns(traj_id_expr(KEYS))
    rows = []
    for tid, label in examples:
        p, b = f.filter(pl.col("traj_id") == tid), base.filter(pl.col("traj_id") == tid)
        if not p.height or not b.height:
            continue
        m = b.row(0, named=True)
        t = p["t_s"].to_numpy().astype(float)
        t = t - t[0]
        model = np.full(t.size, np.nan)
        if m["vmax"] is not None:
            model = m["vmax"] * (1.0 - np.exp(-np.clip(t - m["t0_rel_s"], 0.0, None) / m["tau_s"]))
        lo, hi = m["fit_start_rel_s"], m["fit_end_rel_s"]
        for i in range(t.size):
            rows.append({"traj_id": tid, "label": label, "t_rel_s": float(t[i]), "speed": float(p["speed"][i]),
                         "a_tan": float(p["a_tan"][i]) if p["a_tan"][i] is not None else None,
                         "model_speed": float(model[i]) if np.isfinite(model[i]) else None,
                         "in_fit_window": bool(lo is not None and lo - 1e-9 <= t[i] <= hi + 1e-9)})
    return pl.DataFrame(rows, schema=EXAMPLES.columns) if rows else EXAMPLES.empty()
