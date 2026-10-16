"""Preprocessing checkpoint closing notebook 01 (run at the top of notebook 02).

Three outputs:
    sequence_window_flags          one row per sequence: original clock of its first and last frame, first-frame
                                   and last-frame censoring indicators, real / boundary frame counts, noise
    preprocessing_audit            executable checks of the properties every later model relies on
    preprocessing_decisions_final  the decisions, each with its justification and the evidence it rests on

Censoring indicators describe the RECORDING, not the athlete. They are diagnostics: no sequence is filtered on
them and they are never a model input.
    first_frame_censored    the athlete is already moving on the first valid frame (Phase 03 ``onset_censored``,
                            carried by the sprint rescue) — the start of the run is not in the recording
    start_above_ref_speed   the recording starts at or above the reference speed, so the moment that speed is
                            reached is unknown (alternate origins and re-zeroed channels are undefined)
    last_frame_censored     the last-frame speed is at least ``last_frame_speed_fraction`` of the peak — the
                            recording stops while the athlete is still at top speed
    ends_at_peak            the speed peak is the last frame (no frame recorded after it)

About the "sprint-rescue attempt table missing" warning of notebook 00: ``data.sequence_manifest`` falls back to
the quality tier when it is given no sprint-rescue table. The synthetic fixtures of ``validation`` call it that
way on purpose, once per fixture, which is where every occurrence in the executed notebook 00 comes from (the
validation cell). The production census received the table: all eligible sequences carry the flags. The check
``sprint_rescue_flags_present`` below fails if production ever runs on the fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..contracts import TableContract
from . import data as dt
from .config import AdvancedConfig
from .fpca import CurveSet, FpcaConfig
from .sequences import SequenceError, SequenceSet

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
FPCA_AXES = ("t_rel", "dist", "speed")          # channels used as an axis or a value by every representation
WINDOWS = TableContract(
    "sequence_window_flags", "Recording window, censoring indicators and frame counts of every eligible sequence.",
    {"seq_id": S, "nfl_id": I, "position_group": S, "draft_year": I, "attempt_order": I, "t_first_s": F, "t_last_s": F,
     "n_frames": I, "n_boundary_frames": I, "duration_s": F, "path_yd": F, "first_speed": F, "last_speed": F,
     "peak_speed": F, "last_speed_frac_of_peak": F, "frames_after_peak": I, "first_frame_censored": B,
     "start_above_ref_speed": B, "last_frame_censored": B, "ends_at_peak": B, "pos_resid_rms": F, "a_tan_roughness": F,
     "share_boundary": F, "signal_flag": B, "structural_flag": B, "boundary_peak": B, "S2_signal": B, "S3_no_boundary": B},
    key=("seq_id",), area="sequence",
    docs={"t_first_s": "ORIGINAL clock of the first frame of the sequence [s since the trajectory's first valid frame]",
          "t_last_s": "ORIGINAL clock of the last frame", "n_frames": "real frames (padding excluded)",
          "n_boundary_frames": "real frames inside the smoothing half-window of a segment edge",
          "first_frame_censored": "Phase 03 onset_censored: already moving on the first valid frame",
          "start_above_ref_speed": "first-frame speed ≥ AdvancedConfig.phase_ref_speed",
          "last_frame_censored": "last-frame speed ≥ FpcaConfig.last_frame_speed_fraction × peak speed",
          "ends_at_peak": "frames_after_peak = 0"})
AUDIT = TableContract(
    "preprocessing_audit", "Executable checks of the preprocessing the sequence models rely on.",
    {"check": S, "passed": B, "blocking": B, "n_violations": I, "detail": S}, key=("check",), area="check",
    docs={"blocking": "a failed blocking check stops notebook 02; a failed non-blocking check is a reported finding"})
DECISIONS = TableContract(
    "preprocessing_decisions_final", "Preprocessing decisions used from notebook 02 on, with justification and evidence.",
    {"decision": S, "choice": S, "justification": S, "evidence": S, "basis": S, "status": S}, key=("decision",),
    area="decision",
    docs={"basis": "milestone brief · executed evidence · proposed (starting assumption, open to review)",
          "evidence": "numbers computed in the executing notebook, in its own run mode"})


class PreprocessingError(RuntimeError):
    """A blocking preprocessing check failed."""


# ============================================================================== per-sequence window table
def sequence_windows(ss: SequenceSet, summary: pl.DataFrame, manifest: pl.DataFrame, acfg: AdvancedConfig,
                     fcfg: FpcaConfig) -> pl.DataFrame:
    """``summary``: ``eda.SUMMARY`` · ``manifest``: ``data.MANIFEST``. One row per sequence of ``ss``."""
    rows = []
    for i in range(ss.n):
        k = int(ss.lengths[i])
        v = ss.series(i, "speed")
        peak = float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")
        rows.append({"seq_id": str(ss.seq_ids[i]), "t_first_s": float(ss.t[i, 0]), "t_last_s": float(ss.t[i, k - 1]),
                     "n_frames": k, "n_boundary_frames": int(ss.boundary[i, :k].sum()), "last_speed": float(v[k - 1]),
                     "last_speed_frac_of_peak": float(v[k - 1] / peak) if peak > 0 else None})
    own = pl.DataFrame(rows, schema={"seq_id": S, "t_first_s": F, "t_last_s": F, "n_frames": I, "n_boundary_frames": I,
                                     "last_speed": F, "last_speed_frac_of_peak": F})
    out = (own.join(summary.select("seq_id", "nfl_id", "position_group", "draft_year", "attempt_order", "duration_s",
                                   "path_yd", "first_speed", "peak_speed", "frames_after_peak", "pos_resid_rms",
                                   "a_tan_roughness", "share_boundary"), on="seq_id", how="left")
           .join(manifest.select("seq_id", pl.col("onset_censored").alias("first_frame_censored"), "signal_flag",
                                 "structural_flag", "boundary_peak", "S2_signal", "S3_no_boundary"), on="seq_id", how="left")
           .with_columns((pl.col("first_speed") >= acfg.phase_ref_speed).alias("start_above_ref_speed"),
                         (pl.col("last_speed_frac_of_peak") >= fcfg.last_frame_speed_fraction).alias("last_frame_censored"),
                         (pl.col("frames_after_peak") == 0).alias("ends_at_peak")))
    return out.select([pl.col(c).cast(t) for c, t in WINDOWS.columns.items()]).sort("seq_id")


def window_summary(windows: pl.DataFrame) -> pl.DataFrame:
    """Share of sequences with each recording-window flag, overall and by draft cohort (sequences, not athletes)."""
    flags = ("first_frame_censored", "start_above_ref_speed", "last_frame_censored", "ends_at_peak", "signal_flag",
             "boundary_peak")
    groups = [("ALL", windows)] + [(str(y), windows.filter(pl.col("draft_year") == y))
                                   for y in sorted(windows["draft_year"].drop_nulls().unique().to_list())]
    return pl.DataFrame([{"cohort": name, "sequences": g.height, "athletes": g["nfl_id"].n_unique(),
                          **{f: (float(g[f].drop_nulls().mean()) if g[f].drop_nulls().len() else None) for f in flags},
                          "median_first_speed": g["first_speed"].median(), "median_duration_s": g["duration_s"].median(),
                          "median_frames_after_peak": g["frames_after_peak"].median()} for name, g in groups])


def extreme_recordings(windows: pl.DataFrame, splits: pl.DataFrame, k: int = 5) -> pl.DataFrame:
    """The ``k`` longest recordings and the ``k`` latest starts, with the athlete's role in the design subset —
    a few such recordings can dominate a within-athlete variance on a small subset."""
    role = splits.filter(pl.col("repeat") == 0).select("nfl_id", "design_role")
    cols = ["seq_id", "position_group", "draft_year", "attempt_order", "n_frames", "duration_s", "first_speed",
            "frames_after_peak", "path_yd"]
    longest = windows.sort(["duration_s", "seq_id"], descending=[True, False]).head(k).with_columns(pl.lit("longest recording").alias("why"))
    latest = windows.sort(["first_speed", "seq_id"], descending=[True, False]).head(k).with_columns(pl.lit("latest start").alias("why"))
    return pl.concat([longest, latest]).join(role, on="nfl_id", how="left").select("why", *cols, "design_role")


# ============================================================================== audit
def audit_preprocessing(ss: SequenceSet, seq_frames: pl.DataFrame, manifest: pl.DataFrame, splits: pl.DataFrame,
                        windows: pl.DataFrame, curves: dict[str, CurveSet], acfg: AdvancedConfig, half_window: int
                        ) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, violations: int, detail: str, blocking: bool = True) -> None:
        rows.append({"check": check, "passed": int(violations) == 0, "blocking": blocking, "n_violations": int(violations),
                     "detail": detail})

    # ---- original timestamps
    f = seq_frames.sort("seq_id", "frame").with_columns(
        pl.col("t_s").diff().over("seq_id").alias("_dt"),
        (pl.col("t_rel") - (pl.col("t_s") - pl.col("t_s").first().over("seq_id"))).abs().alias("_drift"))
    dt_ = f["_dt"].drop_nulls()
    add("original_clock_strictly_increasing", int((dt_ <= 0).sum()),
        "t_s (seconds since the trajectory's first valid frame, Phase 03) increases on every frame of every sequence")
    add("t_rel_is_original_clock_minus_first_frame", int((f["_drift"] > 1e-6).sum()),
        "t_rel = t_s − t_s of frame 0; the original clock is kept beside it in sequence_frames and on every aligned grid")
    irregular = int(((dt_ - acfg.frame_dt_s).abs() > 0.011).sum())
    add("regular_sampling_inside_sequences", irregular,
        f"{irregular} of {dt_.len()} frame steps differ from {acfg.frame_dt_s:g} s; every alignment interpolates on the "
        "recorded clock, so an irregular step is used as recorded, not assumed regular", blocking=False)
    # ---- real versus padded frames
    try:
        ss.validate()
        bad = 0
    except SequenceError:
        bad = 1
    padded = int((~ss.mask).sum())
    add("padding_is_zero_and_masked", bad + int(np.count_nonzero(ss.X[~ss.mask])),
        f"{int(ss.mask.sum()):,} real frames and {padded:,} padded cells in {ss.n} sequences; padded cells are exactly "
        "zero, flagged by `mask`, and aligned curves hold NaN (not zero) outside the recording")
    # ---- channel-specific missingness
    miss = {r["channel"]: r["share_undefined"] or 0.0 for r in ss.channel_missingness().iter_rows(named=True)}
    axis_bad = sum(int((~ss.valid[:, :, ss.channel(c)] & ss.mask).sum()) for c in FPCA_AXES if c in ss.channels)
    add("fpca_axis_channels_always_defined", axis_bad,
        "undefined real frames of t_rel / dist / speed; shares of undefined real frames per channel: "
        + ", ".join(f"{c} {100 * v:.2f} %" for c, v in miss.items()))
    # ---- derivative boundaries
    nb = windows["n_boundary_frames"].to_numpy()
    add("boundary_frames_flagged_on_both_edges", int((nb != 2 * half_window).sum()),
        f"sequences whose flagged boundary frames differ from 2 × {half_window} (the Savitzky–Golay half-window at each "
        f"edge); median share of boundary frames per sequence {float(windows['share_boundary'].median()):.3f}", blocking=False)
    # ---- censoring indicators (and the production guard against the sprint-rescue fallback)
    elig = manifest.filter(pl.col("eligible_primary") & pl.col("seq_id").is_in(windows["seq_id"].to_list()))
    add("sprint_rescue_flags_present", int(elig["onset_censored"].null_count() + elig["S2_signal"].null_count()),
        f"eligible sequences without sprint-rescue flags ({elig.height} checked). Non-zero means the manifest was built "
        "on the quality-tier fallback: rerun the sprint rescue, then notebook 00")
    add("censoring_indicators_complete",
        int(windows.select(pl.any_horizontal(pl.col("first_frame_censored", "last_frame_censored", "ends_at_peak",
                                                    "start_above_ref_speed").is_null()).sum()).item()),
        "sequences with an undefined first-frame or last-frame censoring indicator")
    # ---- alignment coverage masks: an observed grid value always maps to a time inside the recording
    lo = dict(zip(windows["seq_id"].to_list(), windows["t_first_s"].to_list()))
    hi = dict(zip(windows["seq_id"].to_list(), windows["t_last_s"].to_list()))
    for name, cs in curves.items():
        t0 = np.array([lo[str(s)] for s in cs.seq_ids])[:, None]
        t1 = np.array([hi[str(s)] for s in cs.seq_ids])[:, None]
        seen = np.isfinite(cs.Y)
        with np.errstate(invalid="ignore"):
            outside = seen & ~(np.isfinite(cs.t_orig) & (cs.t_orig >= t0 - 1e-3) & (cs.t_orig <= t1 + 1e-3))
        add(f"no_value_outside_the_recording:{name}", int(outside.sum()),
            "grid values whose original time is unknown or outside [first frame, last frame] — any would be an extrapolation")
        empty = int((~seen.any(1)).sum())
        add(f"coverage:{name}", 0, f"{100 * seen.mean():.1f} % of grid points × sequences observed; {empty} of {cs.n} "
            "sequences have no value on the grid and cannot be scored", blocking=False)
    # ---- athlete-grouped split integrity
    leak = dt.leakage_audit(splits, manifest, acfg)
    add("split_registry_passes_the_leakage_audit", int((~leak["passed"]).sum()),
        "the seven checks of notebook 00 re-run on the registry as loaded: " + "; ".join(
            f"{r['check']} = {'ok' if r['passed'] else 'FAILED'}" for r in leak.iter_rows(named=True)))
    known = set(splits["nfl_id"].to_list())
    add("every_sequence_athlete_has_a_fold", int(sum(int(i) not in known for i in ss.nfl_ids)),
        "sequences of an athlete absent from the split registry")
    return pl.DataFrame(rows, schema=AUDIT.columns)


def assert_blocking_passed(audit: pl.DataFrame) -> None:
    failed = audit.filter(pl.col("blocking") & ~pl.col("passed"))
    if failed.height:
        raise PreprocessingError(f"{failed.height} blocking preprocessing check(s) failed: " + "; ".join(
            f"{r['check']} ({r['n_violations']} violation(s)) — {r['detail']}" for r in failed.iter_rows(named=True)))


# ============================================================================== decisions
def final_decisions(evidence: dict[str, str]) -> pl.DataFrame:
    """The preprocessing decisions of Milestone 2. ``evidence``: decision → text computed by the executing
    notebook (so the numbers quoted belong to its run mode). Choices and justifications are fixed here."""
    BRIEF, EXEC, PROP = "milestone brief", "executed evidence", "proposed"
    rows = [
        ("representations",
         "A speed against recorded distance (primary) · B tangential acceleration against speed, rising phase "
         "(secondary) · C speed against elapsed time (control)",
         "Fixed by the Milestone 2 brief; it supersedes the single `alignment` choice left pending in notebook 01. "
         "C is kept as the control because the recording clock starts at an arbitrary moment of the run.", BRIEF),
        ("input_channels",
         "speed and a_tan as values; dist, speed and t_rel as axes; no lateral channel",
         "FPCA is fitted one channel at a time. a_lat, lat_dev, yaw_rate and head_sin change sign under a mirror "
         "image of the run and the handedness of the Combine frame is unverified, so they are left out.", PROP),
        ("original_timestamps",
         "kept: t_s per frame in sequence_frames, t_orig per grid point on every aligned curve",
         "An aligned value can always be traced back to its frames; no representation replaces the clock.", EXEC),
        ("censoring_indicators",
         "carried per sequence as diagnostics (sequence_window_flags); never a filter, never a model input",
         "Phase 03 `end_censored` is true for every forty attempt, so the continuous window variables and four "
         "explicit flags are used instead. Filtering on them would select athletes by how they were recorded.", EXEC),
        ("real_versus_padded_frames",
         "padded cells are zero and masked in the sequence array; aligned curves are NaN outside the recording",
         "A padded or unrecorded cell is never an observation: means, covariances and scores use observed points only.", EXEC),
        ("channel_missingness",
         "no imputation; an undefined frame leaves the neighbouring grid points undefined",
         "The axis channels are always defined; a_tan is undefined on a handful of frames, which interpolation does not bridge.", EXEC),
        ("derivative_boundaries",
         "a_tan is masked on smoothing-boundary frames in B (kept in the alternate B′); speed keeps every frame",
         "Acceleration is a second derivative of a locally fitted polynomial and is least reliable at the segment "
         "edges (Phase 03 A3: peaks concentrate on boundary frames). Dropping the same frames for speed would remove "
         "the start and the end of every run, which is where athletes differ.", PROP),
        ("partial_coverage",
         "support = longest run of grid points observed for ≥ 80 % of the training weight; a curve is scored when it "
         "covers ≥ 50 % of the support; scores are best linear predictions from its observed points",
         "Nothing is extrapolated to fit or to score. A reconstruction outside a curve's recorded range is "
         "model-implied, labelled as such, and excluded from every error.", PROP),
        ("quality_scenario",
         "S0_all for fitting and for the primary evaluation; S2_signal and S3_no_boundary as reliability sensitivities",
         "Stricter scenarios cost repeated athletes and select on signal quality; they are reported, not applied.", PROP),
        ("normalisation",
         "training-fold mean function; scores divided by the square root of the training eigenvalue; one unit of "
         "weight per athlete; no per-curve scaling",
         "Per-curve scaling would remove speed level, the main difference between athletes. Every learned quantity "
         "(support, mean, covariance, basis, eigenfunctions, noise variance) comes from the training athletes of the fold.", PROP),
        ("split_integrity",
         "split registry of notebook 00, unchanged; outer folds only",
         "Athletes, never attempts, are assigned to folds. The number of components is compared, not tuned, so the "
         "inner folds are not used in notebook 02.", EXEC),
        ("sprint_rescue_fallback_warning",
         "synthetic fixtures only; production data unaffected; the warning is now silenced in fixtures and guarded in production",
         "Every occurrence in the executed notebook 00 is in the synthetic-validation cell. The production manifest "
         "carries the sprint-rescue flags on every eligible sequence.", EXEC),
    ]
    status = {BRIEF: "DECIDED (brief)", EXEC: "DECIDED (verified)", PROP: "PROPOSED — open to review"}
    return pl.DataFrame([{"decision": d, "choice": c, "justification": j, "evidence": evidence.get(d, ""), "basis": b,
                          "status": status[b]} for d, c, j, b in rows], schema=DECISIONS.columns)
