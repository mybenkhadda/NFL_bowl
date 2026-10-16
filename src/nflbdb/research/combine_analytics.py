"""Phase 03 · M2 — Combine movement analytics: validation, reliability, EDA tables and figures.

Reads ONLY persisted datasets (M2 ``combine_*`` features, M1 ``combine_frame_kinematics`` for the
handful of example trajectories, ``gold.mart_athlete_research_base`` for official Combine results).

Tables (``reports/tables/phase_03/``)
    forty_window_diagnostics.csv    is the forty recording = the timed run? (onset censoring, distance, cut at peak)
    timing_validation.csv           tracked vs official forty / 10-yd / shuttle / three-cone (bias, r, ρ)
    agility_cod_structure.csv       COD counts per drill vs drill design
    combine_feature_summary.csv     feature distributions by drill × cohort (attempt level)
    cohort_shift.csv                standardised cohort differences per drill × feature (M1: 2025 noise differs)
    feature_reliability.csv         ICC(1) + bootstrap CI, typical error, CV, rank stability (drill × feature × cohort)
    feature_redundancy.csv          athlete-level Spearman |ρ| ≥ 0.9 pairs within forty / shuttle / three-cone
    position_comparisons.csv        athlete-median features by position group (key drills)
    m2_example_selection.csv        deterministic example athletes + selection rule
Figures (``reports/figures/phase_03/``, PNG 300 DPI + SVG): see ``FIGURES``.

Every statistic here is DESCRIPTIVE / exploratory. The independent unit is the athlete.

Command for the USER (after ``combine_movement``)::

    python -m nflbdb.research.combine_analytics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..data.config import Phase01Config
from ..data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, add_caption, new_figure
from ..data.utils import get_logger
from ..features.config import MovementConfig
from ..features.feature_registry import HIGHER_IS_BETTER, registry_frame
from ..features.reliability import reliability_table
from .combine_movement import KEYS, NOT_FEATURES, SPRINT_FEATURES, ID_COLS, CombineMovementBuilder
from .kinematic_build import COMBINE_FRAMES

log = get_logger(__name__)

FORTY, SHUTTLE, CONE = "FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL"
KEY_DRILLS = (FORTY, SHUTTLE, CONE)
POSITIONS = ("DB", "DL", "OL", "TE", "WR")
MIN_DRILL_ATTEMPTS = 30
KEY_FEATURES = {
    FORTY: ["model_vmax", "model_tau_s", "model_a0", "trk_t10_s", "trk_t10_20_s", "trk_t20_40_s", "speed_at_10",
            "accel_initial",
            "accel_sustained", "a_tan_max", "speed_max", "rise_time_10_90_s", "onset_to_end_s", "path_efficiency"],
    SHUTTLE: ["onset_to_end_s", "fast_onset_to_end_s", "n_cod", "cod_time_median_s", "cod_approach_decel_median",
              "cod_exit_accel_median", "cod_approach_speed_median", "decel_max", "a_tan_max", "speed_max",
              "path_len_yds", "frac_time_decelerating"],
    CONE: ["onset_to_end_s", "fast_onset_to_end_s", "n_cod", "cod_time_median_s", "cod_approach_decel_median",
           "cod_exit_accel_median", "turn_speed_retention_median", "turn_radius_median_yds", "yaw_integral_deg",
           "decel_max", "speed_max", "path_len_yds"],
}
OFFICIAL = {  # OFFICIAL Combine result (cmb_*, mart_athlete_research_base) → (drill, TRACKED candidates)
    "cmb_forty": (FORTY, ("trk_t40_s", "onset_to_end_s", "fast_onset_to_end_s")),
    "cmb_ten_yd_split": (FORTY, ("trk_t10_s",)),
    "cmb_short_shuttle": (SHUTTLE, ("onset_to_end_s", "fast_onset_to_end_s", "active_duration_s")),
    "cmb_three_cone": (CONE, ("onset_to_end_s", "fast_onset_to_end_s", "active_duration_s")),
}
OFFICIAL_SOURCE = "official Combine result (gold.mart_athlete_research_base)"
TRACKED_BASIS = {
    "trk_t40_s": "tracked: model start t0 → 40 yd along run axis",
    "trk_t10_s": "tracked: model start t0 → 10 yd along run axis",
    "onset_to_end_s": "tracked: onset (≥ 0.5 yd/s, 3 frames) → last recorded frame",
    "fast_onset_to_end_s": "tracked: fast onset (≥ 2.0 yd/s, 3 frames) → last recorded frame",
    "active_duration_s": "tracked: onset → last frame with speed ≥ 0.5 yd/s",
}
TRUNCATION_FLAGS = ("onset_censored", "end_censored", "split_extrapolated", "peak_at_end")
DRILL_OFFICIAL = {FORTY: "cmb_forty", SHUTTLE: "cmb_short_shuttle", CONE: "cmb_three_cone"}
# Single-frame peaks / derivative-based features whose LEVEL moved 5–14 % with the SG window in M1
# (smoothing_sensitivity.csv) — candidates for processing-driven cohort differences.
NOISE_SENSITIVE = frozenset({
    "a_tan_max", "a_tan_p95", "a_tan_p90", "a_tan_sd", "accel_mag_max", "accel_mag_p95", "accel_initial",
    "decel_max", "decel_p95", "braking_peak", "approach_decel_peak", "abs_yaw_rate_p95", "a_lat_abs_p95",
    "n_phase_transitions", "phase_transition_rate", "n_braking_events", "cod_approach_decel_median",
    "cod_exit_accel_median", "turn_radius_median_yds", "speed_cv"})
TABLES = ("forty_window_diagnostics", "timing_validation", "agility_cod_structure", "combine_feature_summary",
          "cohort_shift", "feature_reliability", "feature_redundancy", "position_comparisons", "m2_example_selection")
FIGURES = ("m2_forty_timing_validation", "m2_agility_timing_validation", "m2_forty_speed_profiles_by_position",
           "m2_sprint_model_quality", "m2_forty_similar_time_different_profile", "m2_forty_acceleration_extremes",
           "m2_forty_attempt_stability", "m2_shuttle_braking_extremes", "m2_three_cone_path_extremes",
           "m2_feature_reliability", "m2_cohort_shift", "m2_position_distributions", "m2_forty_feature_correlation")


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bdb_div", [CATEGORICAL[1], "#ecebe6", CATEGORICAL[0]])


def _with_all(df: pl.DataFrame, col: str = "draft_year", name: str = "cohort") -> pl.DataFrame:
    """Rows per cohort plus pooled rows labelled 'all' (cohort column as string)."""
    d = df.with_columns(pl.col(col).cast(pl.String).alias(name))
    return pl.concat([d, d.with_columns(pl.lit("all").alias(name))], how="vertical_relaxed")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    ra = pl.Series(a[ok]).rank("average").to_numpy()
    rb = pl.Series(b[ok]).rank("average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ======================================================================================
# Pure table functions (unit-testable)
# ======================================================================================
def agreement(tracked: np.ndarray, official: np.ndarray) -> dict:
    """Agreement of a tracked measure with an official one (seconds). diff = tracked − official."""
    ok = np.isfinite(tracked) & np.isfinite(official)
    t, o = tracked[ok], official[ok]
    if t.size < 5:
        return {"n_athletes": int(t.size)}
    d = t - o
    resid = d - np.median(d)
    slope, intercept = np.polyfit(o, t, 1)
    r, rho = float(np.corrcoef(t, o)[0, 1]), _spearman(t, o)
    within = float(np.mean(np.abs(resid) <= 0.05))
    # heuristic agreement class (documented, not a validated standard)
    cls = ("interchangeable_after_bias" if r >= 0.9 and within >= 0.8
           else "rank_consistent_only" if np.isfinite(rho) and rho >= 0.8 else "not_interchangeable")
    return {"n_athletes": int(t.size), "bias_median_s": float(np.median(d)), "bias_mean_s": float(np.mean(d)),
            "diff_p05_s": float(np.quantile(d, 0.05)), "diff_p95_s": float(np.quantile(d, 0.95)),
            "mad_diff_s": float(np.median(np.abs(resid))), "mean_abs_diff_s": float(np.mean(np.abs(d))),
            "mae_after_bias_s": float(np.mean(np.abs(resid))), "pearson_r": r, "spearman_rho": rho,
            "slope": float(slope), "intercept": float(intercept), "share_within_0p05s_after_bias": within,
            "agreement_class": cls}


def best_attempt(df: pl.DataFrame, drill: str, col: str, carry: tuple[str, ...] = ()) -> pl.DataFrame:
    """Per athlete, the WHOLE attempt row with the smallest (fastest) tracked ``col`` — flags carried
    from the SAME attempt (never a column-wise minimum that mixes attempts)."""
    keep = [c for c in ("nfl_id", "draft_year", "position_group", "attempt_order", col, *carry) if c in df.columns]
    return (df.filter((pl.col("drill") == drill) & pl.col(col).is_not_null() & pl.col(col).is_finite())
            .sort(["nfl_id", col, "attempt_order"]).group_by("nfl_id", maintain_order=True).first().select(keep))


def athlete_best(df: pl.DataFrame, drill: str, cols: tuple[str, ...]) -> pl.DataFrame:
    """Back-compatible wrapper: best attempt for a single tracked column."""
    return best_attempt(df, drill, cols[0])


def timing_validation(att: pl.DataFrame, sprint: pl.DataFrame, base: pl.DataFrame) -> pl.DataFrame:
    """TRACKED vs OFFICIAL timing. One row per official measure × tracked candidate × cohort.
    diff = tracked − official (s). Athlete = unit; tracked value = athlete's fastest usable attempt
    (official Combine results are best attempts too). Truncation shares describe the attempts used."""
    sp_ok = sprint.filter(pl.col("fit_ok").fill_null(False) & ~pl.col("run_incomplete").fill_null(True))
    rows = []
    for off, (drill, cands) in OFFICIAL.items():
        if off not in base.columns:
            continue
        ref = base.select("nfl_id", off).filter(pl.col(off).is_not_null())
        for c in cands:
            src = sp_ok if c in sp_ok.columns else att
            if c not in src.columns:
                continue
            best = best_attempt(src, drill, c, TRUNCATION_FLAGS)
            n_tracked = best.height
            j = best.join(ref, on="nfl_id", how="inner")
            for (cohort,), g in _with_all(j).group_by(["cohort"], maintain_order=True):
                rec = {"official_measure": off, "official_source": OFFICIAL_SOURCE, "tracked_measure": c,
                       "tracked_basis": TRACKED_BASIS.get(c, "tracked"), "drill": drill, "cohort": cohort,
                       "n_athletes_tracked_all_cohorts": n_tracked,
                       **agreement(g[c].cast(pl.Float64).to_numpy(), g[off].cast(pl.Float64).to_numpy())}
                for fl in TRUNCATION_FLAGS:
                    if fl in g.columns:
                        rec[f"share_{fl}"] = g[fl].cast(pl.Float64).mean()
                rows.append(rec)
    return (pl.DataFrame(rows, infer_schema_length=None).sort("official_measure", "tracked_measure", "cohort")
            if rows else pl.DataFrame())


def forty_window_diagnostics(sprint: pl.DataFrame) -> pl.DataFrame:
    return (_with_all(sprint).group_by("cohort").agg(
                pl.len().alias("n_attempts"), pl.col("nfl_id").n_unique().alias("n_athletes"),
                pl.col("onset_censored").mean().alias("share_onset_censored"),
                pl.col("end_censored").mean().alias("share_end_censored"),
                (pl.col("fit_start_rel_s") > 0).mean().alias("share_stationary_lead_in"),
                pl.col("vmax_extrapolation_ratio").median().alias("median_vmax_extrapolation_ratio"),
                (pl.col("vmax_extrapolation_ratio") > 1.10).mean().alias("share_vmax_extrapolated_gt10pct"),
                pl.col("first_speed").median().alias("median_first_speed"),
                pl.col("pre_distance_yds").median().alias("median_pre_distance_yds"),
                pl.col("run_distance_yds").median().alias("median_run_distance_yds"),
                pl.col("run_distance_yds").quantile(0.05).alias("p05_run_distance_yds"),
                pl.col("run_distance_yds").quantile(0.95).alias("p95_run_distance_yds"),
                pl.col("run_incomplete").mean().alias("share_run_incomplete"),
                pl.col("peak_at_end").mean().alias("share_peak_at_end"),
                pl.col("split_extrapolated").mean().alias("share_split_extrapolated"),
                pl.col("fit_ok").mean().alias("share_fit_ok"),
                pl.col("model_r2").median().alias("median_model_r2"),
                pl.col("model_rmse").median().alias("median_model_rmse"))
            .sort("cohort"))


def agility_cod_structure(att: pl.DataFrame, drills=(SHUTTLE, CONE)) -> pl.DataFrame:
    d = _with_all(att.filter(pl.col("drill").is_in(list(drills))))
    tot = d.group_by("drill", "cohort").agg(pl.len().alias("n_attempts"))
    return (d.group_by("drill", "cohort", "n_cod").agg(pl.len().alias("n"),
                                                      pl.col("n_cod_reversal").median().alias("median_n_reversal"),
                                                      pl.col("cod_angle_total_deg").median().alias("median_total_angle"))
            .join(tot, on=["drill", "cohort"]).with_columns((pl.col("n") / pl.col("n_attempts")).alias("share"))
            .sort("drill", "cohort", "n_cod"))


def attempt_long(att: pl.DataFrame, sprint: pl.DataFrame) -> pl.DataFrame:
    """One row per athlete × attempt × feature (usable attempts; sprint features only for fit-ok rows)."""
    idx = ["nfl_id", "drill", "draft_year", "position_group", "attempt_order"]
    feats = [c for c, t in att.schema.items() if c not in ID_COLS and c not in NOT_FEATURES and t.is_numeric()]
    parts = [att.select(*idx, *[pl.col(c).cast(pl.Float64) for c in feats])
             .unpivot(index=idx, on=feats, variable_name="feature", value_name="value")]
    sf = [c for c in SPRINT_FEATURES if c in sprint.columns]
    if sprint.height and sf:
        parts.append(sprint.select(*idx, *[pl.col(c).cast(pl.Float64) for c in sf])
                     .unpivot(index=idx, on=sf, variable_name="feature", value_name="value"))
    return pl.concat(parts, how="vertical_relaxed").filter(pl.col("value").is_not_null() & pl.col("value").is_finite())


def feature_summary(long: pl.DataFrame, min_attempts: int = MIN_DRILL_ATTEMPTS) -> pl.DataFrame:
    big = long.group_by("drill").agg(pl.struct("nfl_id", "attempt_order").n_unique().alias("_n")).filter(
        pl.col("_n") >= min_attempts)["drill"]
    v = pl.col("value")
    return (_with_all(long.filter(pl.col("drill").is_in(big)))
            .group_by("drill", "cohort", "feature").agg(
                pl.len().alias("n_attempts"), pl.col("nfl_id").n_unique().alias("n_athletes"),
                v.median().alias("median"), v.quantile(0.10).alias("p10"), v.quantile(0.25).alias("p25"),
                v.quantile(0.75).alias("p75"), v.quantile(0.90).alias("p90"), v.mean().alias("mean"), v.std().alias("sd"))
            .sort("drill", "feature", "cohort"))


def _stratified_shift(df: pl.DataFrame, by: list[str], min_stratum: int) -> pl.DataFrame:
    """Within-position standardised cohort shift, n-weighted over positions.
    ``df``: columns [*by, position_group, draft_year, value]. Returns [*by, draft_year, n, n_positions, std_shift]."""
    d = df.with_columns(pl.col("position_group").fill_null("UNKNOWN"))
    pooled = d.group_by(*by, "position_group").agg(
        pl.col("value").median().alias("_pm"),
        (pl.col("value").quantile(0.75) - pl.col("value").quantile(0.25)).alias("_iqr"))
    per = d.group_by(*by, "position_group", "draft_year").agg(pl.len().alias("_n"), pl.col("value").median().alias("_m"))
    s = (per.filter(pl.col("_n") >= min_stratum).join(pooled, on=[*by, "position_group"])
         .filter(pl.col("_iqr") > 0).with_columns(((pl.col("_m") - pl.col("_pm")) / pl.col("_iqr")).alias("_s")))
    return s.group_by(*by, "draft_year").agg(
        pl.col("_n").sum().alias("n"), pl.len().alias("n_positions"),
        ((pl.col("_s") * pl.col("_n")).sum() / pl.col("_n").sum()).alias("std_shift"))


def cohort_shift(long: pl.DataFrame, base: pl.DataFrame | None = None, noise: pl.DataFrame | None = None,
                 min_n: int = 20, min_stratum: int = 8, flag: float = 0.5, anchor_tol: float = 0.25) -> pl.DataFrame:
    """Cohort shift screen that separates likely PROCESSING artifacts from ATHLETIC differences.

    std_shift            within-position (cohort median − position median) / position IQR, n-weighted
                         over positions — removes position-mix differences between cohorts.
    official_anchor_shift  the same statistic for the drill's OFFICIAL Combine time (cmb_*) on the same
                         athletes. Official times are not affected by tracking processing, so they
                         anchor how different the cohorts are athletically.
    noise_ratio          cohort median M1 residual RMS / pooled median (≈ 0.25–0.4 for 2025 per M1).
    noise_sensitive      feature is a single-frame peak / derivative statistic (``NOISE_SENSITIVE``).
    interpretation (HEURISTIC screen, not a test):
      no_material_shift           |std_shift| < flag
      likely_processing_artifact  noise-sensitive, shifted, while the official anchor is ~unchanged
                                  (|anchor| < anchor_tol) or absent and the cohort's noise_ratio ≠ 1 by > 25 %
      athletic_difference_possible  the official anchor itself shifts (|anchor| ≥ anchor_tol)
      unresolved                  everything else
    Recommended confirmation (not run here): recompute the flagged features for 2023/24 with a wider SG
    window that matches 2025 residual noise and check whether the shift disappears."""
    shift = _stratified_shift(long, ["drill", "feature"], min_stratum).filter(pl.col("n") >= min_n)
    out = shift.with_columns(pl.col("feature").is_in(list(NOISE_SENSITIVE)).alias("noise_sensitive"))
    if base is not None:
        anchors = []
        for drill, off in DRILL_OFFICIAL.items():
            if off not in base.columns:
                continue
            ath = (long.filter(pl.col("drill") == drill).select("nfl_id", "draft_year", "position_group").unique("nfl_id")
                   .join(base.select("nfl_id", pl.col(off).alias("value")).filter(pl.col("value").is_not_null()),
                         on="nfl_id"))
            if ath.height:
                anchors.append(_stratified_shift(ath.with_columns(pl.lit(drill).alias("drill")), ["drill"], min_stratum)
                               .select("drill", "draft_year", pl.col("std_shift").alias("official_anchor_shift")))
        if anchors:
            out = out.join(pl.concat(anchors), on=["drill", "draft_year"], how="left")
    if "official_anchor_shift" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("official_anchor_shift"))
    if noise is not None and noise.height:
        out = out.join(noise.select("drill", "draft_year", "noise_ratio", "median_resid_rms_yds"),
                       on=["drill", "draft_year"], how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("noise_ratio"),
                               pl.lit(None, dtype=pl.Float64).alias("median_resid_rms_yds"))
    shifted = pl.col("std_shift").abs() >= flag
    anchor_flat = pl.col("official_anchor_shift").abs() < anchor_tol
    noisy = (pl.col("noise_ratio") - 1.0).abs() > 0.25
    return (out.with_columns(
                pl.col("std_shift").abs().max().over("drill", "feature").alias("max_abs_shift"),
                pl.when(~shifted.fill_null(False)).then(pl.lit("no_material_shift"))
                .when(pl.col("noise_sensitive") & (anchor_flat.fill_null(False)
                                                   | (pl.col("official_anchor_shift").is_null() & noisy.fill_null(False))))
                .then(pl.lit("likely_processing_artifact"))
                .when((pl.col("official_anchor_shift").abs() >= anchor_tol).fill_null(False))
                .then(pl.lit("athletic_difference_possible"))
                .otherwise(pl.lit("unresolved")).alias("interpretation"))
            .with_columns((pl.col("max_abs_shift") >= flag).alias("flag_cohort_shift"))
            .sort("drill", "feature", "draft_year"))


def cohort_noise(att: pl.DataFrame) -> pl.DataFrame:
    """Per drill × cohort median M1 residual RMS and its ratio to the drill's pooled median."""
    if "resid_rms_yds" not in att.columns:
        return pl.DataFrame()
    return (att.group_by("drill", "draft_year").agg(pl.col("resid_rms_yds").median().alias("median_resid_rms_yds"))
            .join(att.group_by("drill").agg(pl.col("resid_rms_yds").median().alias("_pooled")), on="drill")
            .with_columns((pl.col("median_resid_rms_yds") / pl.col("_pooled")).alias("noise_ratio")).drop("_pooled"))


def reliability_all(long: pl.DataFrame, mcfg: MovementConfig) -> pl.DataFrame:
    pooled = reliability_table(long, ["drill", "feature"], mcfg, HIGHER_IS_BETTER)
    if pooled.height:
        pooled = pooled.with_columns(pl.lit("all").alias("cohort"))
    by = reliability_table(long.with_columns(pl.col("draft_year").cast(pl.String).alias("cohort")),
                           ["drill", "feature", "cohort"], mcfg, HIGHER_IS_BETTER)
    parts = [t for t in (pooled, by) if t.height]
    return pl.concat(parts, how="diagonal_relaxed").sort("drill", "feature", "cohort") if parts else pl.DataFrame()


def athlete_wide(athlete: pl.DataFrame, drill: str, stat: str = "median") -> pl.DataFrame:
    return athlete.filter(pl.col("drill") == drill).pivot(on="feature", index="nfl_id", values=stat)


def feature_redundancy(athlete: pl.DataFrame, drills=KEY_DRILLS, threshold: float = 0.9) -> pl.DataFrame:
    rows = []
    for drill in drills:
        w = athlete_wide(athlete, drill)
        feats = [c for c in w.columns if c != "nfl_id"]
        arr = {c: w[c].cast(pl.Float64).to_numpy() for c in feats}
        for i, a in enumerate(feats):
            for b in feats[i + 1:]:
                r = _spearman(arr[a], arr[b])
                if np.isfinite(r) and abs(r) >= threshold:
                    rows.append({"drill": drill, "feature_a": a, "feature_b": b, "spearman_rho": r,
                                 "n_athletes": int((np.isfinite(arr[a]) & np.isfinite(arr[b])).sum())})
    return pl.DataFrame(rows).sort("drill", "spearman_rho", descending=[False, True]) if rows else pl.DataFrame()


def key_feature_frame() -> pl.DataFrame:
    return pl.DataFrame([{"drill": d, "feature": f} for d, fs in KEY_FEATURES.items() for f in fs])


def position_comparisons(athlete: pl.DataFrame) -> pl.DataFrame:
    d = athlete.join(key_feature_frame(), on=["drill", "feature"], how="semi")
    return (d.group_by("drill", "feature", "position_group").agg(
                pl.len().alias("n_athletes"), pl.col("median").median().alias("median_of_athlete_medians"),
                pl.col("median").quantile(0.25).alias("q25"), pl.col("median").quantile(0.75).alias("q75"))
            .sort("drill", "feature", "position_group"))


def _closest(df: pl.DataFrame, col: str, target: float) -> dict:
    return (df.with_columns((pl.col(col) - target).abs().alias("_d")).sort("_d", "nfl_id").row(0, named=True))


def select_examples(athlete: pl.DataFrame, base: pl.DataFrame) -> pl.DataFrame:
    """Deterministic, NFL-outcome-blind example athletes. Each row states its rule."""
    rows = []

    def add(kind, rec, value_col, rule, drill):
        rows.append({"example": kind, "drill": drill, "nfl_id": rec["nfl_id"], "value_feature": value_col,
                     "value": rec.get(value_col), "rule": rule})

    fw = athlete_wide(athlete, FORTY)
    if {"model_a0", "model_tau_s"} <= set(fw.columns):
        a0 = fw.filter(pl.col("model_a0").is_not_null())
        if a0.height >= 10:
            for q, kind in ((0.9, "forty_high_acceleration"), (0.1, "forty_low_acceleration")):
                add(kind, _closest(a0, "model_a0", a0["model_a0"].quantile(q)), "model_a0",
                    f"athlete median model_a0 closest to its p{int(q*100)}; ties → lowest nfl_id", FORTY)
        if "cmb_forty" in base.columns:
            j = (fw.filter(pl.col("model_tau_s").is_not_null())
                 .join(base.select("nfl_id", "cmb_forty").filter(pl.col("cmb_forty").is_not_null()), on="nfl_id")
                 .sort("cmb_forty", "nfl_id"))
            best, med = None, j["cmb_forty"].median() if j.height else None
            for r1, r2 in zip(j.iter_rows(named=True), j.slice(1).iter_rows(named=True)):
                # adjacent athletes by official time; same official time to 0.02 s
                if abs(r2["cmb_forty"] - r1["cmb_forty"]) <= 0.02 + 1e-9:
                    score = (abs(r2["model_tau_s"] - r1["model_tau_s"]), -abs((r1["cmb_forty"] or 0) - med))
                    if best is None or score > best[0]:
                        best = (score, r1, r2)
            if best:
                for r in best[1:]:
                    add("forty_similar_time_different_profile", r, "model_tau_s",
                        "adjacent athletes with |Δ official forty| ≤ 0.02 s maximising |Δ τ|; ties → closer to median "
                        "forty", FORTY)
    stab = athlete.filter((pl.col("drill") == FORTY) & (pl.col("feature") == "model_vmax") & (pl.col("n_attempts") >= 2)
                          & pl.col("sd").is_not_null())
    if stab.height >= 5:
        for kind, desc in (("forty_stable_attempts", False), ("forty_variable_attempts", True)):
            add(kind, stab.sort(["sd", "nfl_id"], descending=[desc, False]).row(0, named=True), "sd",
                f"{'max' if desc else 'min'} within-athlete SD of model_vmax (≥ 2 fit-ok attempts); ties → lowest nfl_id",
                FORTY)
    sw = athlete_wide(athlete, SHUTTLE)
    if "cod_approach_decel_median" in sw.columns:
        s = sw.filter(pl.col("cod_approach_decel_median").is_not_null())
        if s.height >= 10:
            for q, kind in ((0.9, "shuttle_high_braking"), (0.1, "shuttle_low_braking")):
                add(kind, _closest(s, "cod_approach_decel_median", s["cod_approach_decel_median"].quantile(q)),
                    "cod_approach_decel_median", f"athlete median closest to p{int(q*100)}; ties → lowest nfl_id", SHUTTLE)
    cw = athlete_wide(athlete, CONE)
    if "path_len_yds" in cw.columns:
        c = cw.filter(pl.col("path_len_yds").is_not_null())
        if c.height >= 10:
            for q, kind in ((0.1, "three_cone_short_path"), (0.9, "three_cone_long_path")):
                add(kind, _closest(c, "path_len_yds", c["path_len_yds"].quantile(q)), "path_len_yds",
                    f"athlete median path length closest to p{int(q*100)} (fixed course → shorter = tighter)", CONE)
    return pl.DataFrame(rows, infer_schema_length=None)


# ======================================================================================
class CombineMovementAnalytics:
    def __init__(self, cfg: Phase01Config, builder: CombineMovementBuilder | None = None):
        self.cfg = cfg
        self.b = builder or CombineMovementBuilder(cfg)
        self.store, self.mcfg, self.kcfg = self.b.store, self.b.mcfg, self.b.kcfg

    # ------------------------------------------------------------------ inputs
    def official(self) -> pl.DataFrame:
        cols = ["nfl_id", "position_group", "cohort", *OFFICIAL, "cmb_combine_height", "cmb_combine_weight"]
        lf = self.store.refs.scan("gold", "mart_athlete_research_base")
        have = lf.collect_schema().names()
        return lf.select([c for c in cols if c in have]).collect()

    def frames_for(self, keys: pl.DataFrame) -> pl.DataFrame:
        return (self.store.load(COMBINE_FRAMES.name).join(keys.lazy().select(KEYS).unique(), on=KEYS, how="semi")
                .filter(pl.col("valid_kin")).collect().sort([*KEYS, "t_s"]))

    def best_attempt_keys(self, sprint_or_att: pl.DataFrame, nfl_ids: list[int], drill: str, by: str,
                          lowest: bool = True) -> pl.DataFrame:
        return (sprint_or_att.filter((pl.col("drill") == drill) & pl.col("nfl_id").is_in(nfl_ids) & pl.col(by).is_not_null())
                .sort(["nfl_id", by, "attempt_order"], descending=[False, not lowest, False])
                .group_by("nfl_id", maintain_order=True).first())

    # ------------------------------------------------------------------ figures
    def _save(self, fig, name: str, source: str, n=None) -> Path:
        return self.store.save_fig(fig, name, source=source, n=n)

    def fig_timing(self, att, sprint, base, name: str, pairs: list[tuple[str, str, str]]) -> Path | None:
        fig, axes = new_figure(1, len(pairs))
        axes = np.atleast_1d(axes)
        sp_ok = sprint.filter(pl.col("fit_ok").fill_null(False) & ~pl.col("run_incomplete").fill_null(True))
        for ax, (off, drill, tracked) in zip(axes, pairs):
            src = sp_ok if tracked in sp_ok.columns else att
            d = athlete_best(src, drill, (tracked,)).join(base.select("nfl_id", off), on="nfl_id", how="inner").drop_nulls(
                [tracked, off])
            ag = agreement(d[tracked].cast(pl.Float64).to_numpy(), d[off].cast(pl.Float64).to_numpy())
            ax.scatter(d[off], d[tracked], s=40, color=CATEGORICAL[0], edgecolor=INK["surface"], linewidth=1.5, zorder=3)
            if d.height:
                lo = float(min(d[off].min(), d[tracked].min()))
                hi = float(max(d[off].max(), d[tracked].max()))
                ax.plot([lo, hi], [lo, hi], color=INK["secondary"], lw=1, ls="--", label="identity (y = x)")
            ax.set_xlabel(f"OFFICIAL Combine {off.removeprefix('cmb_')} (s)")
            ax.set_ylabel(f"TRACKED {tracked}, fastest usable attempt (s)")
            r = ag.get("pearson_r")
            ax.set_title(f"{drill}: tracked {tracked} vs official (n = {ag['n_athletes']}"
                         + (f", r = {r:.2f}, bias = {ag['bias_median_s']:+.2f} s)" if r is not None else ")"),
                         fontsize=11)
            ax.legend(loc="upper left", fontsize=9)
        add_caption(fig, "Athlete = unit. Tracked = fastest usable attempt; model-start / onset conventions differ from "
                         "official timing — see timing_validation.csv")
        return self._save(fig, name, "combine_sprint_profiles / combine_attempt_features + mart_athlete_research_base")

    def _model_time(self, fr: pl.DataFrame, sprint: pl.DataFrame) -> pl.DataFrame:
        """Frames with t_model = t − (t_first + model_t0_rel_s) (time since the model start)."""
        return (fr.with_columns(pl.col("t_s").min().over(KEYS).alias("_tf"))
                .join(sprint.select(*KEYS, "model_t0_rel_s", "model_vmax", "model_tau_s", "attempt_order"), on=KEYS,
                      how="inner")
                .with_columns((pl.col("t_s") - pl.col("_tf") - pl.col("model_t0_rel_s")).alias("t_model")))

    def fig_speed_profiles(self, sprint: pl.DataFrame) -> Path | None:
        best = (sprint.filter(pl.col("fit_ok").fill_null(False)).sort("nfl_id", "model_vmax", descending=[False, True])
                .group_by("nfl_id", maintain_order=True).first())
        if best.is_empty():
            return None
        fr = self._model_time(self.frames_for(best), best)
        grid = np.round(np.arange(0.0, 5.01, 0.1), 2)
        pos_of = dict(best.select("nfl_id", "position_group").iter_rows())
        curves = [(pos_of.get(part["nfl_id"][0]),
                   np.interp(grid, part["t_model"].to_numpy(), part["speed"].to_numpy(), left=np.nan, right=np.nan))
                  for part in fr.partition_by(KEYS, maintain_order=True)]
        allc = np.array([c for _, c in curves])
        pooled = np.nanmedian(allc, axis=0)
        fig, axes = new_figure(1, len(POSITIONS), sharey=True)
        for ax, pos in zip(axes, POSITIONS):
            arr = np.array([c for p, c in curves if p == pos])
            ax.plot(grid, pooled, color=INK["muted"], lw=1.2, ls="--", label="all positions (median)")
            if arr.size:
                with np.errstate(all="ignore"):
                    q25, q50, q75 = (np.nanquantile(arr, q, axis=0) for q in (0.25, 0.5, 0.75))
                ax.fill_between(grid, q25, q75, color=SEQUENTIAL_BLUE[1], lw=0, label="IQR")
                ax.plot(grid, q50, color=CATEGORICAL[0], label="median")
            ax.set_title(f"{pos} (n = {len(arr)})", fontsize=12)
            ax.set_xlabel("time since model start (s)")
        axes[0].set_ylabel("derived speed (yd/s)")
        axes[0].legend(loc="lower right", fontsize=8)
        fig.suptitle("Forty-yard dash: speed–time profiles by position (best fit-ok attempt per athlete)",
                     x=0.01, ha="left", fontsize=15, fontweight="semibold")
        add_caption(fig, "Curves end where recordings end (window cut at the finish). Source: combine_frame_kinematics + "
                         "combine_sprint_profiles")
        return self._save(fig, "m2_forty_speed_profiles_by_position", "combine_sprint_profiles", n=len(curves))

    def fig_model_quality(self, sprint: pl.DataFrame) -> Path:
        ok = sprint.filter(pl.col("fit_ok").fill_null(False))
        fig, axes = new_figure(1, 2)
        axes[0].hist(ok["model_r2"].drop_nulls().to_numpy(), bins=40, color=CATEGORICAL[0], edgecolor=INK["surface"])
        axes[0].set_xlabel("model R²")
        axes[0].set_ylabel("attempts")
        axes[0].set_title(f"Fit quality (n = {ok.height:,} fit-ok of {sprint.height:,} attempts)", fontsize=12)
        axes[1].scatter(ok["model_vmax"], ok["model_tau_s"], s=24, color=CATEGORICAL[0], alpha=0.7,
                        edgecolor=INK["surface"], linewidth=1)
        axes[1].set_xlabel("model vmax (yd/s)")
        axes[1].set_ylabel("model τ (s)")
        axes[1].set_title("Asymptotic speed vs acceleration time constant", fontsize=12)
        add_caption(fig, "v(t) = vmax(1 − e^{−(t − t0)/τ}); grid least squares. Source: combine_sprint_profiles")
        return self._save(fig, "m2_sprint_model_quality", "combine_sprint_profiles", n=ok.height)

    def fig_two_athletes(self, ex: pl.DataFrame, kinds: tuple[str, ...], sprint: pl.DataFrame, base: pl.DataFrame,
                         name: str, title: str) -> Path | None:
        rows = ex.filter(pl.col("example").is_in(list(kinds)))
        kind_prefix = "forty_"
        if rows.height < 2:
            return None
        ids = rows["nfl_id"].to_list()[:2]
        keys = self.best_attempt_keys(sprint.filter(pl.col("fit_ok").fill_null(False)), ids, FORTY, "model_vmax",
                                      lowest=False)
        fr = self._model_time(self.frames_for(keys), keys)
        fig, ax = new_figure()
        off = dict(base.select("nfl_id", "cmb_forty").iter_rows()) if "cmb_forty" in base.columns else {}
        for i, (nid, kind) in enumerate(zip(ids, rows["example"].to_list()[:2])):
            f = fr.filter(pl.col("nfl_id") == nid)
            if f.is_empty():
                continue
            k = keys.filter(pl.col("nfl_id") == nid).row(0, named=True)
            t = f["t_model"].to_numpy()
            model = k["model_vmax"] * (1 - np.exp(-np.clip(t, 0, None) / k["model_tau_s"]))
            lbl = (f"{kind.replace(kind_prefix, '').strip('_') or 'athlete'} · id {nid} · τ = {k['model_tau_s']:.2f} s, "
                   f"vmax = {k['model_vmax']:.1f} yd/s" + (f", official forty {off[nid]:.2f} s" if off.get(nid) else ""))
            ax.plot(t, f["speed"].to_numpy(), "o", ms=4, color=CATEGORICAL[i], label=f"observed — {lbl}")
            ax.plot(t, model, color=CATEGORICAL[i], lw=2, alpha=0.8, label="model fit")
        ax.set_xlabel("time since model start (s)")
        ax.set_ylabel("derived speed (yd/s)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        add_caption(fig, "Selection rule: m2_example_selection.csv (deterministic, outcome-blind). Best fit-ok attempt shown.")
        return self._save(fig, name, "combine_frame_kinematics + combine_sprint_profiles", n=2)

    def fig_stability(self, ex: pl.DataFrame, sprint: pl.DataFrame) -> Path | None:
        rows = ex.filter(pl.col("example").is_in(["forty_stable_attempts", "forty_variable_attempts"]))
        if rows.height < 2:
            return None
        sp = sprint.filter(pl.col("fit_ok").fill_null(False) & pl.col("nfl_id").is_in(rows["nfl_id"].to_list()))
        fr = self._model_time(self.frames_for(sp), sp)
        fig, axes = new_figure(1, 2, sharey=True)
        for ax, r in zip(axes, rows.iter_rows(named=True)):
            f = fr.filter(pl.col("nfl_id") == r["nfl_id"]).sort("attempt_order", "t_s")
            for i, part in enumerate(f.partition_by("attempt_order", maintain_order=True)[:4]):
                ax.plot(part["t_model"], part["speed"], color=CATEGORICAL[i],
                        label=f"attempt {part['attempt_order'][0]}")
            ax.set_title(f"{r['example'].replace('forty_', '').replace('_', ' ')} · id {r['nfl_id']} "
                         f"(SD of vmax = {r['value']:.2f} yd/s)", fontsize=12)
            ax.set_xlabel("time since model start (s)")
            ax.legend(loc="lower right", fontsize=9)
        axes[0].set_ylabel("derived speed (yd/s)")
        add_caption(fig, "Repeated forty attempts of the most and least consistent athletes (within-athlete SD of model "
                         "vmax). Source: combine_sprint_profiles")
        return self._save(fig, "m2_forty_attempt_stability", "combine_sprint_profiles", n=sp.height)

    def fig_shuttle(self, ex: pl.DataFrame, att: pl.DataFrame) -> Path | None:
        rows = ex.filter(pl.col("example").is_in(["shuttle_high_braking", "shuttle_low_braking"]))
        if rows.height < 2:
            return None
        keys = self.best_attempt_keys(att, rows["nfl_id"].to_list(), SHUTTLE, "onset_to_end_s")
        fr = self.frames_for(keys).join(keys.select(*KEYS, "t_onset_s"), on=KEYS).with_columns(
            (pl.col("t_s") - pl.col("t_onset_s")).alias("t_rel"))
        fig, axes = new_figure(2, 1, sharex=True)
        for i, r in enumerate(rows.iter_rows(named=True)):
            f = fr.filter(pl.col("nfl_id") == r["nfl_id"])
            lbl = f"{r['example'].replace('shuttle_', '').replace('_', ' ')} · id {r['nfl_id']}"
            axes[0].plot(f["t_rel"], f["speed"], color=CATEGORICAL[i], label=lbl)
            axes[1].plot(f["t_rel"], f["a_tan"], color=CATEGORICAL[i], label=lbl)
        axes[0].set_ylabel("derived speed (yd/s)")
        axes[1].axhline(0, color=INK["axis"], lw=1)
        axes[1].set_ylabel("a_tan (yd/s²)")
        axes[1].set_xlabel("time since movement onset (s)")
        axes[0].set_title("Short shuttle: high vs low COD approach deceleration (fastest usable attempt)")
        for ax in axes:
            ax.legend(loc="upper right", fontsize=9)
        add_caption(fig, "a_tan jumps at a reversal (speed passes ≈ 0) — the scalar speed derivative is not the vector "
                         "acceleration. Selection: m2_example_selection.csv")
        return self._save(fig, "m2_shuttle_braking_extremes", "combine_frame_kinematics", n=2)

    def fig_three_cone(self, ex: pl.DataFrame, att: pl.DataFrame) -> Path | None:
        from matplotlib.collections import LineCollection

        rows = ex.filter(pl.col("example").is_in(["three_cone_short_path", "three_cone_long_path"]))
        if rows.height < 2:
            return None
        keys = self.best_attempt_keys(att, rows["nfl_id"].to_list(), CONE, "onset_to_end_s")
        fr = self.frames_for(keys)
        # M2.1 fix: shared x AND y + adjustable="datalim" is rejected by Matplotlib at draw/export time →
        # unshared axes with identical limits and adjustable="box" (m21.plotting.equal_aspect_panels)
        from .figkit import equal_aspect_panels

        fig, axes = new_figure(1, 2)
        vmax = float(fr["speed"].max() or 1.0)
        lc, xs, ys = None, [], []
        for ax, r in zip(axes, rows.iter_rows(named=True)):
            f = fr.filter(pl.col("nfl_id") == r["nfl_id"])
            x = f["x_s"].to_numpy() - f["x_s"][0]
            y = f["y_s"].to_numpy() - f["y_s"][0]
            xs.append(x)
            ys.append(y)
            segs = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
            lc = LineCollection(segs, cmap="Blues", linewidths=3)
            lc.set_array(f["speed"].to_numpy()[:-1])
            lc.set_clim(0, vmax)
            ax.add_collection(lc)
            ax.set_title(f"{r['example'].replace('three_cone_', '').replace('_', ' ')} · id {r['nfl_id']} "
                         f"(median path {r['value']:.1f} yd)", fontsize=12)
            ax.set_xlabel("x − start (yd, orientation unverified)")
        axes[0].set_ylabel("y − start (yd)")
        equal_aspect_panels(axes, xs, ys)
        if lc is not None:
            fig.colorbar(lc, ax=list(axes), label="derived speed (yd/s)", shrink=0.8)
        add_caption(fig, "Fixed course → shorter path = tighter cone rounding. Selection: m2_example_selection.csv")
        return self._save(fig, "m2_three_cone_path_extremes", "combine_frame_kinematics", n=2)

    def fig_reliability(self, rel: pl.DataFrame) -> Path | None:
        d = rel.filter((pl.col("cohort") == "all") & pl.col("icc1").is_not_null())
        if d.is_empty():
            return None
        fig, axes = new_figure(1, len(KEY_DRILLS), sharex=True)
        for ax, drill in zip(axes, KEY_DRILLS):
            r = d.filter((pl.col("drill") == drill) & pl.col("feature").is_in(KEY_FEATURES[drill])).sort("icc1")
            yy = np.arange(r.height)
            # a percentile CI need not contain the point estimate → clip error lengths at 0
            lo = np.clip((r["icc1"] - r["icc1_ci_lo"].fill_null(r["icc1"])).to_numpy(), 0, None)
            hi = np.clip((r["icc1_ci_hi"].fill_null(r["icc1"]) - r["icc1"]).to_numpy(), 0, None)
            ax.errorbar(r["icc1"].to_numpy(), yy, xerr=[lo, hi], fmt="o", ms=7, color=CATEGORICAL[0],
                        ecolor=SEQUENTIAL_BLUE[2], elinewidth=2, capsize=0)
            ax.set_yticks(yy, r["feature"].to_list(), fontsize=9)
            for ref in (0.5, 0.75, 0.9):
                ax.axvline(ref, color=INK["muted"], lw=1, ls=":")
            n = int(r["n_athletes"].max() or 0) if r.height else 0
            ax.set_title(f"{drill} (≤ {n} athletes)", fontsize=12)
            ax.set_xlabel("ICC(1), single attempt — 95 % bootstrap CI")
            ax.set_xlim(-0.2, 1.0)
        add_caption(fig, "Athletes with ≥ 2 usable attempts. Dotted: 0.5 / 0.75 / 0.9 (Koo & Li 2016 conventions). "
                         "Source: feature_reliability.csv")
        return self._save(fig, "m2_feature_reliability", "feature_reliability.csv")

    def fig_cohort_shift(self, cs: pl.DataFrame) -> Path | None:
        rows = [(d, f) for d in KEY_DRILLS for f in KEY_FEATURES[d]]
        d = cs.join(key_feature_frame(), on=["drill", "feature"], how="semi")
        if d.is_empty():
            return None
        years = sorted(d["draft_year"].unique().to_list())
        labels = [f"{a} · {b}" for a, b in rows]
        mat = np.full((len(rows), len(years)), np.nan)
        lut = {(r["drill"], r["feature"], r["draft_year"]): r["std_shift"] for r in d.iter_rows(named=True)}
        for i, (a, b) in enumerate(rows):
            for j, y in enumerate(years):
                v = lut.get((a, b, y))
                mat[i, j] = np.nan if v is None else v
        fig, ax = new_figure(figsize=(9.6, 12.8))
        im = ax.imshow(mat, cmap=_diverging_cmap(), vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(years)), [str(y) for y in years])
        ax.set_yticks(range(len(rows)), labels, fontsize=8)
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7, color=INK["primary"])
        fig.colorbar(im, ax=ax, label="(cohort median − pooled median) / pooled IQR", shrink=0.6)
        ax.set_title("Cohort shifts in Combine features (descriptive screen)")
        add_caption(fig, "|shift| ≥ 0.5 flagged in cohort_shift.csv. Cohorts differ in athletes AND in tracking noise "
                         "(M1) — a shift is not attributable to either without further checks.")
        return self._save(fig, "m2_cohort_shift", "cohort_shift.csv")

    def fig_positions(self, athlete: pl.DataFrame) -> Path | None:
        panels = [(FORTY, "model_vmax", "model vmax (yd/s)"), (FORTY, "model_a0", "model a0 (yd/s²)"),
                  (FORTY, "trk_t10_20_s", "tracked flying 10→20 yd (s)"), (SHUTTLE, "cod_time_median_s", "shuttle COD time (s)")]
        fig, axes = new_figure(2, 2)
        for ax, (drill, feat, lab) in zip(np.ravel(axes), panels):
            d = athlete.filter((pl.col("drill") == drill) & (pl.col("feature") == feat))
            data = [d.filter(pl.col("position_group") == p)["median"].drop_nulls().to_numpy() for p in POSITIONS]
            ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                       boxprops={"facecolor": SEQUENTIAL_BLUE[1], "edgecolor": INK["secondary"]},
                       medianprops={"color": CATEGORICAL[0], "linewidth": 2},
                       whiskerprops={"color": INK["secondary"]}, capprops={"color": INK["secondary"]},
                       flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK["muted"], "markeredgecolor": "none"})
            ax.set_xticks(range(1, len(POSITIONS) + 1), [f"{p}\n(n = {len(x)})" for p, x in zip(POSITIONS, data)])
            ax.set_ylabel(lab)
            ax.set_title(f"{drill}: {feat}", fontsize=12)
        add_caption(fig, "One value per athlete (median over usable attempts). Position differences reflect body mass "
                         "and role, not quality. Source: combine_athlete_drill_features")
        return self._save(fig, "m2_position_distributions", "combine_athlete_drill_features")

    def fig_forty_corr(self, athlete: pl.DataFrame) -> Path | None:
        w = athlete_wide(athlete, FORTY)
        feats = [f for f in KEY_FEATURES[FORTY] if f in w.columns]
        if len(feats) < 3:
            return None
        arr = {c: w[c].cast(pl.Float64).to_numpy() for c in feats}
        mat = np.array([[_spearman(arr[a], arr[b]) if a != b else 1.0 for b in feats] for a in feats])
        fig, ax = new_figure(figsize=(11.2, 9.6))
        im = ax.imshow(mat, cmap=_diverging_cmap(), vmin=-1, vmax=1)
        ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right", fontsize=9)
        ax.set_yticks(range(len(feats)), feats, fontsize=9)
        ax.grid(False)
        for i in range(len(feats)):
            for j in range(len(feats)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK["primary"])
        fig.colorbar(im, ax=ax, label="Spearman ρ (athlete medians)", shrink=0.7)
        ax.set_title(f"Forty-yard dash: feature correlation structure (n ≤ {w.height} athletes)")
        add_caption(fig, "Pairwise-complete. |ρ| ≥ 0.9 pairs listed in feature_redundancy.csv.")
        return self._save(fig, "m2_forty_feature_correlation", "combine_athlete_drill_features", n=w.height)

    # ------------------------------------------------------------------ orchestration
    def run(self, figures: bool = True) -> dict[str, pl.DataFrame]:
        att = self.b.load("attempts").collect()
        sprint = self.b.load("sprint").collect()
        athlete = self.b.load("athlete").collect()
        base = self.official()
        long = attempt_long(att, sprint)
        out = {
            "feature_dictionary": registry_frame(),
            "forty_window_diagnostics": forty_window_diagnostics(sprint) if sprint.height else pl.DataFrame(),
            "timing_validation": timing_validation(att, sprint, base),
            "agility_cod_structure": agility_cod_structure(att),
            "combine_feature_summary": feature_summary(long),
            "cohort_shift": cohort_shift(long, base, cohort_noise(att)),
            "feature_reliability": reliability_all(long, self.mcfg),
            "feature_redundancy": feature_redundancy(athlete),
            "position_comparisons": position_comparisons(athlete),
            "m2_example_selection": select_examples(athlete, base),
        }
        for name, df in out.items():
            if not df.height:   # keep a stable file with an explicit note instead of silently skipping
                log.warning("[m2] %s is empty — writing a placeholder with a note", name)
                df = pl.DataFrame({"note": [f"{name}: no rows (insufficient data for this table)"]})
            # CSV for humans + Parquet twin for typed reuse (reload: CombineMovementAnalytics.load_table)
            self.store.export_csv(df, name, f"Phase 03 M2: {name}", parquet=True)
        if figures:
            ex = out["m2_example_selection"]
            steps = [
                lambda: self.fig_timing(att, sprint, base, "m2_forty_timing_validation",
                                        [("cmb_forty", FORTY, "trk_t40_s"), ("cmb_ten_yd_split", FORTY, "trk_t10_s")]),
                lambda: self.fig_timing(att, sprint, base, "m2_agility_timing_validation",
                                        [("cmb_short_shuttle", SHUTTLE, "fast_onset_to_end_s"),
                                         ("cmb_three_cone", CONE, "fast_onset_to_end_s")]),
                lambda: self.fig_speed_profiles(sprint),
                lambda: self.fig_model_quality(sprint),
                lambda: self.fig_two_athletes(ex, ("forty_similar_time_different_profile",), sprint, base,
                                              "m2_forty_similar_time_different_profile",
                                              "Forty: similar official times, different acceleration profiles"),
                lambda: self.fig_two_athletes(ex, ("forty_high_acceleration", "forty_low_acceleration"), sprint, base,
                                              "m2_forty_acceleration_extremes",
                                              "Forty: high vs low model initial acceleration (a0)"),
                lambda: self.fig_stability(ex, sprint),
                lambda: self.fig_shuttle(ex, att),
                lambda: self.fig_three_cone(ex, att),
                lambda: self.fig_reliability(out["feature_reliability"]),
                lambda: self.fig_cohort_shift(out["cohort_shift"]),
                lambda: self.fig_positions(athlete),
                lambda: self.fig_forty_corr(athlete),
            ]
            for step in steps:
                try:
                    step()
                except Exception as exc:  # one failed figure must not lose the CSVs / other figures
                    log.exception("[m2] figure failed: %s", exc)
            missing = self.check_outputs()
            if missing:
                log.warning("[m2] missing outputs after run: %s", missing)
        return out

    # ------------------------------------------------------------------ outputs / reload
    def expected_outputs(self) -> list[Path]:
        """Stable paths of every M2 analytics artifact (9 tables × CSV + Parquet, 13 figures × PNG + SVG)."""
        t, f = self.store.tables_dir, self.store.figures_dir
        return ([t / f"{n}.{ext}" for n in TABLES for ext in ("csv", "parquet")]
                + [f / f"{n}.{ext}" for n in FIGURES for ext in ("png", "svg")])

    def check_outputs(self) -> list[str]:
        return [self.cfg.paths.rel(p) for p in self.expected_outputs() if not p.is_file()]

    def load_table(self, name: str) -> pl.DataFrame:
        """Reload a persisted M2 table with its dtypes (Parquet twin of the CSV)."""
        path = self.store.tables_dir / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing — run `python -m nflbdb.research.combine_analytics`")
        return pl.read_parquet(path)


def main(argv: list[str] | None = None) -> int:
    from ..data.config import configure_runtime

    ap = argparse.ArgumentParser(description="Phase 03 · M2 Combine movement analytics (tables + figures)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--check", action="store_true", help="only list missing expected outputs (no computation)")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    if a.check:
        missing = CombineMovementAnalytics(cfg).check_outputs()
        print("all M2 outputs present" if not missing else "MISSING:\n  " + "\n  ".join(missing))
        return 1 if missing else 0
    out = CombineMovementAnalytics(cfg).run(figures=not a.no_figures)
    with pl.Config(tbl_rows=40, tbl_cols=14):
        print(out["forty_window_diagnostics"])
        print(out["timing_validation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
