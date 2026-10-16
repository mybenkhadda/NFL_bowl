"""Study A — data quality and measurement validity (Phase 03 · M1.1).

A1 · Quality-tier calibration
    Question     Do the provisional tiers (or continuous quality dimensions) distinguish reliable from
                 unreliable trajectories? M1 evidence: Combine 6,309 / 6,310 tier A; NFL ≈ 99 % tier A —
                 the tiers barely discriminate, so the study measures continuous DIMENSIONS.
    Population   every persisted Combine attempt and NFL player-play (all tiers; nothing pre-filtered).
    Grain        trajectory (Combine attempt / NFL player-play); uncertainty clustered by athlete.
    Dimensions   structural integrity · temporal continuity · coordinate validity · speed consistency ·
                 derivative reliability · heading reliability · event stability · smoothing sensitivity
                 (the last joins from Study B; null until B has run). Higher metric = worse.
    Methods      per-dimension screening flag above the domain × stratum quantile (PROPOSED, 0.99) +
                 percentile rank; proportions with Wilson AND athlete-cluster bootstrap intervals;
                 Cliff's δ (non-A vs A) and Spearman ρ (metric vs duration) with athlete-cluster
                 bootstrap CIs; retention funnel under four filter scenarios.
    Caveat       heading validity depends on movement CONTENT (standing still ⇒ undefined heading by
                 design), so it is content-confounded, not a pure quality measure.

A2 · Vendor vs derived speed
    Question     Where and by how much does coordinate-derived speed disagree with vendor ``s``?
    Population   valid-kinematics frames: all Combine frames + the deterministic NFL game sample
                 (NFL frames are not persisted; ``nfl_games_per_season`` evenly spaced games per season).
    Grain        frame (statistics) · trajectory (error vs duration) · athlete (bootstrap unit).
    Methods      r = derived − vendor; MAE, RMSE, bias, SD, Bland–Altman 95 % LoA, median / p95 |r|,
                 relative error for s ≥ 1 yd/s; groups: domain, cohort, stratum (drill type), position,
                 tier, speed band of the Bland–Altman mean (s + derived)/2. Correlation is NOT used to
                 claim agreement. Frame-level LoA describe frames; CIs come from resampling athletes.

Outputs (``StudyStore``): see ``CONTRACTS``; figures in ``quality_figures``.

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.studies --only A [--sample-athletes 300] [--skip-nfl-sample] [--force]
"""

from __future__ import annotations

import dataclasses
import zlib
from typing import Any

import numpy as np
import polars as pl

from ...data.config import Phase01Config
from ...data.utils import get_logger
from ...features.config import KinematicsConfig
from ..contracts import TableContract, cast_keys, upstream_state
from ..kinematic_build import (COMBINE_EVENTS, COMBINE_FRAMES, COMBINE_SEGMENTS, COMBINE_TRAJ, NFL_SEGMENTS, NFL_TRAJ,
                               KinematicFeatureBuilder)
from .config import QualityStudyConfig, StudiesConfig, threshold_table
from .results import RESULTS, ResultCollector
from .stats import (cliffs_delta, cluster_bootstrap, cluster_bootstrap_ratios, percentile_interval, spearman,
                    wilson_ci)
from .store import StudyStore

log = get_logger(__name__)

STUDY = "A"
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
TIERS = ("A", "B", "C", "reject")
# dimension → (metric column, description); higher metric = worse quality
DIMENSIONS: dict[str, tuple[str, str]] = {
    "structural_integrity": ("structural_breaks", "gap + jump breaks + null-xy + duplicate-ts removals + extra segments"),
    "temporal_continuity": ("offgrid_rate", "off-grid sampling intervals / valid frames"),
    "coordinate_validity": ("resid_rms_yds", "RMS raw − smoothed position residual (yd)"),
    "speed_consistency": ("speed_vendor_mad", "median |derived − vendor speed| (yd/s)"),
    "derivative_reliability": ("max_accel_mag", "maximum |acceleration| (yd/s²)"),
    "heading_reliability": ("heading_invalid_frac", "share of frames without a defined heading (content-confounded)"),
    "event_stability": ("event_count_rel_range", "(lenient − strict event count) / lenient count (Combine only)"),
    "smoothing_sensitivity": ("smoothing_rel_change", "relative feature change across SG settings (Study B; null)"),
}
METRICS = [m for m, _ in DIMENSIONS.values()]
FACETS = ("all", "cohort", "stratum", "position", "duration_bin")
VENDOR_FACETS = ("all", "cohort", "stratum", "position", "quality_tier", "speed_band")
SCENARIOS = (("all trajectories", pl.lit(True)), ("exclude reject", pl.col("quality_tier") != "reject"),
             ("tier A only", pl.col("quality_tier") == "A"),
             ("tier A + no screening flag", (pl.col("quality_tier") == "A") & (pl.col("n_flagged_dims") == 0)))

# ============================================================================== contracts
DIMS = TableContract(
    "trajectory_quality_dimensions", "One row per trajectory: quality dimensions, percentile ranks, screening flags.",
    {"domain": S, "traj_id": S, "nfl_id": I, "cohort": I, "stratum": S, "position": S, "quality_tier": S,
     "tier_reasons": S, "duration_s": F, "duration_bin": S, "n_valid_frames": I,
     **{m: F for m in METRICS}, **{f"pct_{m}": F for m in METRICS}, **{f"flag_{d}": B for d in DIMENSIONS},
     "n_flagged_dims": I, "flagged_dims": S},
    key=("domain", "traj_id"), kind="dataset", area="quality",
    docs={"traj_id": "Combine '<event_id>|<nfl_id>|<attempt>' · NFL '<game_id>|<play_id>|<nfl_id>'",
          "cohort": "Combine draft_year · NFL season", "stratum": "Combine drill_type · NFL 'nfl_play'",
          "duration_s": "sum of segment durations (s)",
          **{f"pct_{m}": "percentile rank within domain × stratum (0–1, higher = worse)" for m in METRICS}})
PROPORTIONS = TableContract(
    "A1_proportions", "Tier and screening-flag proportions by facet with Wilson and athlete-cluster bootstrap CIs.",
    {"domain": S, "facet": S, "level": S, "indicator": S, "n_trajectories": I, "n_athletes": I, "n_flagged": I,
     "share": F, "wilson_lower": F, "wilson_upper": F, "boot_lower": F, "boot_upper": F},
    key=("domain", "facet", "level", "indicator"), area="descriptive",
    docs={"wilson_lower": "assumes independent trajectories (optimistic)",
          "boot_lower": "athlete-cluster bootstrap (respects repeated trajectories per athlete)"})
METRIC_BY_TIER = TableContract(
    "A1_metric_by_tier", "Distribution of every quality metric (and duration) by domain × tier.",
    {"domain": S, "quality_tier": S, "metric": S, "n_trajectories": I, "n_athletes": I, "mean": F, "p05": F,
     "p25": F, "median": F, "p75": F, "p95": F, "p99": F, "max": F},
    key=("domain", "quality_tier", "metric"), area="descriptive")
REASONS = TableContract(
    "A1_reason_matrix", "How often each tier rule / screening flag fires per domain × stratum.",
    {"domain": S, "stratum": S, "reason": S, "source": S, "n_trajectories": I, "n_stratum": I, "share": F},
    key=("domain", "stratum", "reason"), area="descriptive")
FUNNEL = TableContract(
    "A1_filter_funnel", "Research sample retained under each quality-filter scenario.",
    {"domain": S, "stratum": S, "scenario": S, "scenario_order": I, "n_trajectories": I, "n_athletes": I,
     "share_trajectories_retained": F, "share_athletes_retained": F},
    key=("domain", "stratum", "scenario"), area="descriptive")
THRESHOLDS_TBL = TableContract(
    "A1_screen_thresholds", "Empirical screening thresholds per domain × stratum × metric.",
    {"domain": S, "stratum": S, "metric": S, "quantile": F, "threshold": F, "n_trajectories": I, "share_zero": F},
    key=("domain", "stratum", "metric"), area="descriptive",
    docs={"share_zero": "share of trajectories with metric = 0 (degenerate distributions → threshold 0)"})
EXTREME_ACCEL = TableContract(
    "A1_extreme_accel_by_tier", "Frame share with |a| above extreme_accel_yds2 by domain × tier (frames available).",
    {"domain": S, "quality_tier": S, "n_trajectories": I, "n_athletes": I, "n_frames": I, "n_extreme_frames": I,
     "rate": F, "boot_lower": F, "boot_upper": F}, key=("domain", "quality_tier"), area="descriptive")
A1_EXAMPLES = TableContract(
    "A1_examples", "Deterministic representative good / poor trajectories (+ selection rule).",
    {"example": S, "domain": S, "traj_id": S, "quality_tier": S, "n_flagged_dims": I, "flagged_dims": S,
     "resid_rms_yds": F, "rule": S}, key=("example",), area="descriptive")
A1_SIGNALS = TableContract(
    "A1_example_signals", "Signals of the A1 example trajectories (figure redraws without tracking reads).",
    {"example": S, "t_s": F, "x_rel": F, "y_rel": F, "speed": F, "vendor_speed": F},
    key=("example", "t_s"), area="descriptive")
VENDOR = TableContract(
    "A2_vendor_error_summary", "Derived − vendor speed error by domain × facet × level (frame grain, athlete CIs).",
    {"domain": S, "facet": S, "level": S, "n_frames": I, "n_trajectories": I, "n_athletes": I,
     "mae": F, "mae_lower": F, "mae_upper": F, "rmse": F, "rmse_lower": F, "rmse_upper": F,
     "bias": F, "bias_lower": F, "bias_upper": F, "sd": F, "loa_lower": F, "loa_upper": F, "median_signed": F,
     "median_abs": F, "p05": F, "p25": F, "p75": F, "p95": F, "p95_abs": F, "rel_mae": F, "n_frames_rel": I},
    key=("domain", "facet", "level"), area="descriptive",
    docs={"mae_lower": "athlete-cluster bootstrap percentile interval",
          "loa_lower": "Bland–Altman bias − 1.96 SD of FRAME residuals (describes frames)",
          "rel_mae": "mean |r| / vendor s over frames with s ≥ rel_error_min_speed"})
FRAME_DIAG = TableContract(
    "trajectory_frame_diagnostics", "Per-trajectory vendor error + acceleration outliers (frames available only).",
    {"domain": S, "traj_id": S, "nfl_id": I, "cohort": I, "stratum": S, "position": S, "quality_tier": S,
     "n_frames": I, "duration_s": F, "mae": F, "bias": F, "p95_abs": F, "max_abs": F, "n_extreme_accel": I,
     "extreme_accel_rate": F}, key=("domain", "traj_id"), kind="dataset", area="quality")
HIST2D = TableContract(
    "A2_hist2d", "2-D histogram counts (non-zero cells): vendor vs derived speed, and Bland–Altman (mean, diff).",
    {"domain": S, "kind": S, "x_lo": F, "y_lo": F, "count": I}, key=("domain", "kind", "x_lo", "y_lo"),
    area="descriptive")
BY_SPEED = TableContract(
    "A2_residual_by_speed", "Residual quantiles by Bland–Altman mean-speed bin (0.25 yd/s).",
    {"domain": S, "m_lo": F, "n_frames": I, "mae": F, "median": F, "p05": F, "p25": F, "p75": F, "p95": F},
    key=("domain", "m_lo"), area="descriptive")
A2_EXAMPLES = TableContract(
    "A2_example_signals", "Highest-disagreement trajectories: vendor vs derived speed over time.",
    {"example": S, "domain": S, "traj_id": S, "trajectory_mae": F, "t_s": F, "vendor_speed": F, "speed": F},
    key=("example", "t_s"), area="descriptive")
STUDY_RESULTS = dataclasses.replace(RESULTS, name="A_quality_results")
THRESHOLD_DOC = TableContract(
    "studies_thresholds", "Study parameters with provenance (established / proposed / execution).",
    {"study_section": S, "name": S, "value": S, "unit": S, "provenance": S, "rationale": S},
    key=("study_section", "name"), area="descriptive")
CONTRACTS = (DIMS, PROPORTIONS, METRIC_BY_TIER, REASONS, FUNNEL, THRESHOLDS_TBL, EXTREME_ACCEL, A1_EXAMPLES,
             A1_SIGNALS, VENDOR, FRAME_DIAG, HIST2D, BY_SPEED, A2_EXAMPLES, STUDY_RESULTS)


def _seed(base: int, *parts: Any) -> int:
    return (base + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32)


# ============================================================================== A1: normalisation
def traj_id_expr(keys: list[str]) -> pl.Expr:
    return pl.concat_str([pl.col(k).cast(pl.String) for k in keys], separator="|").alias("traj_id")


def event_stability(events: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Per trajectory: lenient − strict event count (all event types), absolute and relative."""
    c = events.group_by(keys).agg((pl.col("rule") == "lenient").sum().cast(pl.Int64).alias("n_lenient"),
                                  (pl.col("rule") == "strict").sum().cast(pl.Int64).alias("n_strict"))
    return c.select(*keys, (pl.col("n_lenient") - pl.col("n_strict")).cast(pl.Float64).alias("event_count_range"),
                    ((pl.col("n_lenient") - pl.col("n_strict")) / pl.max_horizontal(pl.col("n_lenient"), pl.lit(1)))
                    .cast(pl.Float64).alias("event_count_rel_range"))


def normalize_trajectories(traj: pl.DataFrame, seg: pl.DataFrame, events: pl.DataFrame | None, domain: str,
                           keys: list[str], cohort_col: str, stratum_col: str | None,
                           positions: pl.DataFrame) -> pl.DataFrame:
    """Common trajectory schema for both domains (raw M1 metrics + segment aggregates + event stability)."""
    segs = seg.group_by(keys).agg(
        pl.col("duration_s").sum().cast(pl.Float64).alias("duration_s"),
        (1 - (pl.col("frac_heading_valid") * pl.col("n_frames")).sum() / pl.col("n_frames").sum())
        .cast(pl.Float64).alias("heading_invalid_frac"))
    t = traj.join(segs, on=keys, how="left")
    if events is not None and set(events["rule"].unique().to_list()) >= {"lenient", "strict"}:
        t = (t.join(event_stability(events, keys), on=keys, how="left")
             .with_columns(pl.col("event_count_rel_range").fill_null(0.0)))   # no event under any rule → stable
    else:
        t = t.with_columns(pl.lit(None, dtype=pl.Float64).alias("event_count_rel_range"))
    nfl_pos = positions.select(pl.col("nfl_id").cast(pl.Int64), pl.col("position_group").cast(pl.String))
    t = cast_keys(t).join(nfl_pos, on="nfl_id", how="left")
    ints = ("n_segments", "n_gap_breaks", "n_jump_breaks", "n_null_xy_removed", "n_dup_ts_removed", "n_offgrid",
            "n_valid_frames")
    return t.select(
        pl.lit(domain).alias("domain"), traj_id_expr(keys), pl.col("nfl_id").cast(pl.Int64),
        pl.col(cohort_col).cast(pl.Int64).alias("cohort"),
        (pl.col(stratum_col).cast(pl.String) if stratum_col else pl.lit("nfl_play")).alias("stratum"),
        pl.col("position_group").fill_null("UNKNOWN").alias("position"),
        pl.col("quality_tier").cast(pl.String), pl.col("tier_reasons").cast(pl.String).fill_null(""),
        *[pl.col(c).cast(pl.Int64).fill_null(0) for c in ints],
        *[pl.col(c).cast(pl.Float64) for c in ("resid_rms_yds", "speed_vendor_mad", "max_accel_mag", "duration_s",
                                               "heading_invalid_frac", "event_count_rel_range")])


def quality_dimensions(t: pl.DataFrame, cfg: QualityStudyConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add dimension metrics, percentile ranks, screening flags. Returns (dimensions, thresholds)."""
    t = t.with_columns(
        (pl.col("n_gap_breaks") + pl.col("n_jump_breaks") + pl.col("n_null_xy_removed") + pl.col("n_dup_ts_removed")
         + (pl.col("n_segments") - 1).clip(0)).cast(pl.Float64).alias("structural_breaks"),
        (pl.col("n_offgrid") / pl.col("n_valid_frames")).cast(pl.Float64).alias("offgrid_rate"),
        pl.lit(None, dtype=pl.Float64).alias("smoothing_rel_change"),
        pl.col("duration_s").cut(list(cfg.duration_bins_s[1:]), left_closed=True).cast(pl.String)
        .fill_null("unknown").alias("duration_bin"))
    grp = ["domain", "stratum"]
    thr = (t.unpivot(index=grp, on=METRICS, variable_name="metric", value_name="v").filter(pl.col("v").is_not_null())
           .group_by(*grp, "metric").agg(pl.col("v").quantile(cfg.screen_quantile, "linear").alias("threshold"),
                                         pl.len().cast(pl.Int64).alias("n_trajectories"),
                                         (pl.col("v") == 0).mean().cast(pl.Float64).alias("share_zero"))
           .with_columns(pl.lit(cfg.screen_quantile).alias("quantile")).sort(*grp, "metric"))
    wide = thr.pivot(on="metric", index=grp, values="threshold")
    wide = wide.rename({m: f"_thr_{m}" for m in METRICS if m in wide.columns})
    t = t.join(wide, on=grp, how="left")
    flags, pcts = [], []
    for d, (m, _) in DIMENSIONS.items():
        thr_col = pl.col(f"_thr_{m}") if f"_thr_{m}" in t.columns else pl.lit(None, dtype=pl.Float64)
        flags.append((pl.col(m) > thr_col).fill_null(False).alias(f"flag_{d}"))
        pcts.append((pl.col(m).rank("average").over(grp) / pl.col(m).count().over(grp)).cast(pl.Float64)
                    .alias(f"pct_{m}"))
    t = t.with_columns(*flags, *pcts)
    t = t.with_columns(
        pl.sum_horizontal([pl.col(f"flag_{d}").cast(pl.Int64) for d in DIMENSIONS]).cast(pl.Int64).alias("n_flagged_dims"),
        pl.concat_str([pl.when(pl.col(f"flag_{d}")).then(pl.lit(d)) for d in DIMENSIONS], separator=";",
                      ignore_nulls=True).fill_null("").alias("flagged_dims"))
    return t.select(list(DIMS.columns)), thr.select(list(THRESHOLDS_TBL.columns))


# ============================================================================== A1: tables
def _indicators(t: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    inds = [*(f"tier_{x}" for x in TIERS), *(f"flag_{d}" for d in DIMENSIONS), "any_flag"]
    t = t.with_columns(*[(pl.col("quality_tier") == x).alias(f"tier_{x}") for x in TIERS],
                       (pl.col("n_flagged_dims") > 0).alias("any_flag"), pl.lit("all").alias("all"))
    return t, inds


def proportions(t: pl.DataFrame, cfg: QualityStudyConfig, seed: int, facets=FACETS) -> pl.DataFrame:
    t, inds = _indicators(t)
    rows = []
    for facet in facets:
        for (domain, level), g in t.group_by("domain", facet, maintain_order=True):
            per = (g.group_by("nfl_id").agg(pl.len().cast(pl.Int64).alias("_n"),
                                            *[pl.col(c).cast(pl.Int64).sum().alias(c) for c in inds]).sort("nfl_id"))
            num, den = per.select(inds).to_numpy().astype(float), per["_n"].to_numpy().astype(float)
            est, reps = cluster_bootstrap_ratios(num, den, cfg.n_boot, _seed(seed, domain, facet, level))
            blo, bhi, _ = percentile_interval(reps)
            k, n = num.sum(axis=0), den.sum()
            wlo, whi = wilson_ci(k, np.full(k.shape, n))
            for j, ind in enumerate(inds):
                rows.append({"domain": domain, "facet": facet, "level": str(level), "indicator": ind,
                             "n_trajectories": int(n), "n_athletes": int(per.height), "n_flagged": int(k[j]),
                             "share": float(est[j]), "wilson_lower": float(wlo[j]), "wilson_upper": float(whi[j]),
                             "boot_lower": float(blo[j]), "boot_upper": float(bhi[j])})
    return pl.DataFrame(rows, schema=PROPORTIONS.columns).with_columns(
        *[pl.col(c).fill_nan(None) for c in ("boot_lower", "boot_upper", "wilson_lower", "wilson_upper")])


def metric_by_tier(t: pl.DataFrame) -> pl.DataFrame:
    long = t.unpivot(index=["domain", "quality_tier", "nfl_id"], on=[*METRICS, "duration_s"], variable_name="metric",
                     value_name="v").filter(pl.col("v").is_not_null())
    q = lambda p: pl.col("v").quantile(p, "linear").cast(pl.Float64)  # noqa: E731
    return (long.group_by("domain", "quality_tier", "metric").agg(
                pl.len().cast(pl.Int64).alias("n_trajectories"), pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
                pl.col("v").mean().alias("mean"), q(0.05).alias("p05"), q(0.25).alias("p25"), q(0.5).alias("median"),
                q(0.75).alias("p75"), q(0.95).alias("p95"), q(0.99).alias("p99"), pl.col("v").max().alias("max"))
            .select(list(METRIC_BY_TIER.columns)).sort("domain", "metric", "quality_tier"))


def reason_matrix(t: pl.DataFrame) -> pl.DataFrame:
    n = t.group_by("domain", "stratum").agg(pl.len().cast(pl.Int64).alias("n_stratum"))
    tier = (t.select("domain", "stratum", pl.col("tier_reasons").str.split(";").alias("reason")).explode("reason")
            .filter(pl.col("reason").is_not_null() & (pl.col("reason") != "")).with_columns(pl.lit("tier_rule").alias("source")))
    flag = (t.select("domain", "stratum", pl.col("flagged_dims").str.split(";").alias("reason")).explode("reason")
            .filter(pl.col("reason").is_not_null() & (pl.col("reason") != ""))
            .with_columns(("flag:" + pl.col("reason")).alias("reason"), pl.lit("screen_flag").alias("source")))
    both = pl.concat([tier, flag])
    if not both.height:
        return REASONS.empty()
    return (both.group_by("domain", "stratum", "reason", "source").agg(pl.len().cast(pl.Int64).alias("n_trajectories"))
            .join(n, on=["domain", "stratum"], how="left")
            .with_columns((pl.col("n_trajectories") / pl.col("n_stratum")).cast(pl.Float64).alias("share"))
            .select(list(REASONS.columns)).sort("domain", "stratum", "reason"))


def filter_funnel(t: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for strat_all in (False, True):
        d = t.with_columns(pl.lit("all").alias("stratum")) if strat_all else t
        base = d.group_by("domain", "stratum").agg(pl.len().alias("_n0"), pl.col("nfl_id").n_unique().alias("_a0"))
        for i, (name, keep) in enumerate(SCENARIOS):
            r = (d.filter(keep).group_by("domain", "stratum")
                 .agg(pl.len().cast(pl.Int64).alias("n_trajectories"), pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes")))
            rows.append(base.join(r, on=["domain", "stratum"], how="left").with_columns(
                pl.col("n_trajectories").fill_null(0), pl.col("n_athletes").fill_null(0),
                pl.lit(name).alias("scenario"), pl.lit(i, dtype=pl.Int64).alias("scenario_order")))
    out = pl.concat(rows, how="vertical_relaxed").with_columns(
        (pl.col("n_trajectories") / pl.col("_n0")).cast(pl.Float64).alias("share_trajectories_retained"),
        (pl.col("n_athletes") / pl.col("_a0")).cast(pl.Float64).alias("share_athletes_retained"))
    return out.select(list(FUNNEL.columns)).sort("domain", "stratum", "scenario_order")


def a1_results(t: pl.DataFrame, cfg: QualityStudyConfig, seed: int, rc: ResultCollector) -> None:
    """Cliff's δ (non-A vs A) per metric (+ duration) and Spearman ρ (metric vs duration), athlete-clustered."""
    method_b = f"athlete-cluster bootstrap (B={cfg.n_boot_generic}, percentile)"
    for (domain,), d in t.group_by("domain", maintain_order=True):
        cl = d["nfl_id"].to_numpy()
        non_a = (d["quality_tier"] != "A").to_numpy()
        dur = d["duration_s"].to_numpy().astype(float)
        for m in [*METRICS, "duration_s"]:
            v = d[m].to_numpy().astype(float)
            if not np.isfinite(v).any():
                continue
            n_non_a = int((non_a & np.isfinite(v)).sum())
            enough = n_non_a >= cfg.min_group_trajectories
            b = (cluster_bootstrap(lambda i: cliffs_delta(v[i][non_a[i]], v[i][~non_a[i]]), cl, cfg.n_boot_generic,
                                   _seed(seed, domain, m, "delta")) if enough
                 else {"estimate": cliffs_delta(v[non_a], v[~non_a]), "se": None, "ci_lower": None, "ci_upper": None})
            rc.add(analysis_id="A1.tier_contrast", grain="trajectory", n_athletes=int(np.unique(cl).size),
                   n_trajectories=int(np.isfinite(v).sum()), effect_estimate=b["estimate"],
                   effect_size_type="cliffs_delta_nonA_vs_A", method="Cliff's δ", domain=domain, feature=m,
                   outcome="quality_tier != A", standard_error=b["se"], ci_lower=b["ci_lower"],
                   ci_upper=b["ci_upper"], ci_method=method_b if enough else "",
                   quality_filter="none (all tiers)",
                   result_status="descriptive" if enough else "insufficient_data",
                   limitations="" if enough else f"only {n_non_a} non-A trajectories")
            if m == "duration_s":
                continue
            ok = np.isfinite(v) & np.isfinite(dur)
            if ok.sum() < cfg.min_group_trajectories:
                continue
            b = cluster_bootstrap(lambda i: spearman(v[i], dur[i]), cl, cfg.n_boot_generic, _seed(seed, domain, m, "rho"))
            rc.add(analysis_id="A1.duration_association", grain="trajectory", n_athletes=int(np.unique(cl[ok]).size),
                   n_trajectories=int(ok.sum()), effect_estimate=b["estimate"], effect_size_type="spearman_rho",
                   method="Spearman ρ (metric vs trajectory duration)", domain=domain, feature=m, outcome="duration_s",
                   standard_error=b["se"], ci_lower=b["ci_lower"], ci_upper=b["ci_upper"], ci_method=method_b,
                   quality_filter="none (all tiers)",
                   limitations="longer trajectories have more opportunity for extreme values (max-type metrics)")


def select_a1_examples(t: pl.DataFrame, available: dict[str, set[str]]) -> pl.DataFrame:
    """Per domain: 'good' = tier A, no flag, resid_rms closest to the domain median; 'poor' = most flagged
    dimensions, then largest resid_rms. Only trajectories whose frames are available are eligible."""
    rows = []
    for (domain,), d in t.group_by("domain", maintain_order=True):
        d = d.filter(pl.col("traj_id").is_in(list(available.get(domain, set()))) & pl.col("resid_rms_yds").is_not_null())
        if not d.height:
            continue
        med = d["resid_rms_yds"].median()
        good = (d.filter((pl.col("quality_tier") == "A") & (pl.col("n_flagged_dims") == 0))
                .sort((pl.col("resid_rms_yds") - med).abs(), "traj_id").head(1))
        poor = d.sort(["n_flagged_dims", "resid_rms_yds", "traj_id"], descending=[True, True, False]).head(1)
        for name, r, rule in ((f"{domain}_good", good, "tier A, no screening flag, resid_rms closest to domain median"),
                              (f"{domain}_poor", poor, "most flagged dimensions, then largest resid_rms")):
            if r.height:
                rows.append(r.select(pl.lit(name).alias("example"), "domain", "traj_id", "quality_tier",
                                     "n_flagged_dims", "flagged_dims", "resid_rms_yds", pl.lit(rule).alias("rule")))
    return pl.concat(rows) if rows else A1_EXAMPLES.empty()


# ============================================================================== A2: vendor agreement
def vendor_frames(frames: pl.DataFrame, domain: str, keys: list[str], t: pl.DataFrame,
                  cfg: QualityStudyConfig) -> pl.DataFrame:
    """Valid frames with residual r = derived − vendor and the trajectory context from ``t``."""
    bands = list(cfg.speed_bands[1:])
    ctx = t.filter(pl.col("domain") == domain).select("traj_id", "cohort", "stratum", "position", "quality_tier",
                                                      "duration_s")
    f = (cast_keys(frames).filter(pl.col("valid_kin") & pl.col("s").is_not_null() & pl.col("speed").is_not_null())
         .select(traj_id_expr(keys), pl.col("nfl_id").cast(pl.Int64), pl.col("t_s").cast(pl.Float64),
                 pl.col("x_s").cast(pl.Float64), pl.col("y_s").cast(pl.Float64), pl.col("s").cast(pl.Float64),
                 pl.col("speed").cast(pl.Float64), pl.col("accel_mag").cast(pl.Float64)))
    return (f.join(ctx, on="traj_id", how="inner")
            .with_columns(pl.lit(domain).alias("domain"), (pl.col("speed") - pl.col("s")).alias("r"),
                          ((pl.col("speed") + pl.col("s")) / 2).alias("m"), pl.lit("all").alias("all"))
            .with_columns(pl.col("m").cut(bands, left_closed=True).cast(pl.String).alias("speed_band")))


def vendor_error_summary(fr: pl.DataFrame, cfg: QualityStudyConfig, seed: int, rc: ResultCollector,
                         facets=VENDOR_FACETS) -> pl.DataFrame:
    rows = []
    for facet in facets:
        for (domain, level), g in fr.group_by("domain", facet, maintain_order=True):
            a = g.select(
                pl.len().cast(pl.Int64).alias("n_frames"), pl.col("traj_id").n_unique().cast(pl.Int64).alias("n_trajectories"),
                pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"), pl.col("r").std().alias("sd"),
                pl.col("r").median().alias("median_signed"), pl.col("r").abs().median().alias("median_abs"),
                *[pl.col("r").quantile(q, "linear").alias(n) for q, n in ((0.05, "p05"), (0.25, "p25"),
                                                                          (0.75, "p75"), (0.95, "p95"))],
                pl.col("r").abs().quantile(0.95, "linear").alias("p95_abs"),
                (pl.col("r").abs() / pl.col("s")).filter(pl.col("s") >= cfg.rel_error_min_speed).mean().alias("rel_mae"),
                (pl.col("s") >= cfg.rel_error_min_speed).sum().cast(pl.Int64).alias("n_frames_rel")).row(0, named=True)
            per = (g.group_by("nfl_id").agg(pl.len().alias("n"), pl.col("r").abs().sum().alias("sa"),
                                            (pl.col("r") ** 2).sum().alias("s2"), pl.col("r").sum().alias("s1"))
                   .sort("nfl_id"))
            est, reps = cluster_bootstrap_ratios(per.select("sa", "s2", "s1").to_numpy(), per["n"].to_numpy(),
                                                 cfg.n_boot, _seed(seed, domain, facet, level, "vendor"))
            est = np.array([est[0], np.sqrt(est[1]), est[2]])
            reps = np.column_stack([reps[:, 0], np.sqrt(reps[:, 1]), reps[:, 2]])
            lo, hi, se = percentile_interval(reps)
            sd = a["sd"]
            row = {"domain": domain, "facet": facet, "level": str(level), **a,
                   "mae": est[0], "mae_lower": lo[0], "mae_upper": hi[0], "rmse": est[1], "rmse_lower": lo[1],
                   "rmse_upper": hi[1], "bias": est[2], "bias_lower": lo[2], "bias_upper": hi[2],
                   "loa_lower": est[2] - 1.959964 * sd if sd is not None else None,
                   "loa_upper": est[2] + 1.959964 * sd if sd is not None else None}
            rows.append(row)
            for k, (name, unit) in enumerate((("MAE", "yd/s"), ("RMSE", "yd/s"), ("bias_derived_minus_vendor", "yd/s"))):
                rc.add(analysis_id="A2.vendor_agreement", grain="frame", n_athletes=a["n_athletes"],
                       n_trajectories=a["n_trajectories"], n_frames=a["n_frames"], effect_estimate=est[k],
                       effect_size_type=name, unit=unit, method="derived − vendor speed, frame residuals",
                       domain=domain, stratum=f"{facet}={level}", feature="speed", outcome="vendor s",
                       standard_error=se[k], ci_lower=lo[k], ci_upper=hi[k],
                       ci_method=f"athlete-cluster bootstrap (B={cfg.n_boot}, percentile)",
                       quality_filter="valid_kin frames, all tiers",
                       limitations="NFL = deterministic game sample" if domain == "nfl_sample" else "")
    out = pl.DataFrame(rows, schema={c: VENDOR.columns[c] for c in VENDOR.columns}, strict=False)
    return out.with_columns(*[pl.col(c).cast(t).fill_nan(None) if t == F else pl.col(c).cast(t)
                              for c, t in VENDOR.columns.items()]).select(list(VENDOR.columns))


def frame_diagnostics(fr: pl.DataFrame, cfg: QualityStudyConfig) -> pl.DataFrame:
    return (fr.group_by("domain", "traj_id").agg(
                pl.col("nfl_id").first(), pl.col("cohort").first(), pl.col("stratum").first(), pl.col("position").first(),
                pl.col("quality_tier").first(), pl.len().cast(pl.Int64).alias("n_frames"),
                pl.col("duration_s").first(), pl.col("r").abs().mean().alias("mae"), pl.col("r").mean().alias("bias"),
                pl.col("r").abs().quantile(0.95, "linear").alias("p95_abs"), pl.col("r").abs().max().alias("max_abs"),
                (pl.col("accel_mag") > cfg.extreme_accel_yds2).sum().cast(pl.Int64).alias("n_extreme_accel"))
            .with_columns((pl.col("n_extreme_accel") / pl.col("n_frames")).cast(pl.Float64).alias("extreme_accel_rate"))
            .select(list(FRAME_DIAG.columns)).sort("domain", "traj_id"))


def extreme_accel_by_tier(diag: pl.DataFrame, cfg: QualityStudyConfig, seed: int) -> pl.DataFrame:
    rows = []
    for (domain, tier), g in diag.group_by("domain", "quality_tier", maintain_order=True):
        per = g.group_by("nfl_id").agg(pl.col("n_frames").sum().alias("n"), pl.col("n_extreme_accel").sum().alias("k")).sort("nfl_id")
        est, reps = cluster_bootstrap_ratios(per["k"].to_numpy(), per["n"].to_numpy(), cfg.n_boot,
                                             _seed(seed, domain, tier, "xacc"))
        lo, hi, _ = percentile_interval(reps)
        rows.append({"domain": domain, "quality_tier": tier, "n_trajectories": g.height, "n_athletes": per.height,
                     "n_frames": int(per["n"].sum()), "n_extreme_frames": int(per["k"].sum()), "rate": float(est[0]),
                     "boot_lower": float(lo[0]), "boot_upper": float(hi[0])})
    return pl.DataFrame(rows, schema=EXTREME_ACCEL.columns).with_columns(
        pl.col("boot_lower").fill_nan(None), pl.col("boot_upper").fill_nan(None))


def hist2d(fr: pl.DataFrame, cfg: QualityStudyConfig) -> pl.DataFrame:
    out = []
    for (domain,), g in fr.group_by("domain", maintain_order=True):
        s, v, m, r = (g[c].to_numpy() for c in ("s", "speed", "m", "r"))
        e = np.arange(0, cfg.hist_speed_max + cfg.hist_bin / 2, cfg.hist_bin)
        er = np.arange(-cfg.resid_range, cfg.resid_range + 1e-9, cfg.hist_bin / 4)
        for kind, (x, y, ex, ey) in {"vendor_vs_derived": (s, v, e, e),
                                     "bland_altman": (m, np.clip(r, er[0], er[-1] - 1e-9), e, er)}.items():
            h, _, _ = np.histogram2d(np.clip(x, ex[0], ex[-1] - 1e-9), np.clip(y, ey[0], ey[-1] - 1e-9), bins=[ex, ey])
            ix, iy = np.nonzero(h)
            out.append(pl.DataFrame({"domain": [domain] * ix.size, "kind": [kind] * ix.size,
                                     "x_lo": np.round(ex[ix], 6), "y_lo": np.round(ey[iy], 6),
                                     "count": h[ix, iy].astype(np.int64)}, schema=HIST2D.columns))
    return pl.concat(out) if out else HIST2D.empty()


def residual_by_speed(fr: pl.DataFrame) -> pl.DataFrame:
    q = lambda p: pl.col("r").quantile(p, "linear")  # noqa: E731
    return (fr.with_columns(((pl.col("m") / 0.25).floor() * 0.25).round(4).alias("m_lo"))
            .group_by("domain", "m_lo").agg(pl.len().cast(pl.Int64).alias("n_frames"), pl.col("r").abs().mean().alias("mae"),
                                            q(0.5).alias("median"), q(0.05).alias("p05"), q(0.25).alias("p25"),
                                            q(0.75).alias("p75"), q(0.95).alias("p95"))
            .select(list(BY_SPEED.columns)).sort("domain", "m_lo"))


def a2_trajectory_results(diag: pl.DataFrame, cfg: QualityStudyConfig, seed: int, rc: ResultCollector) -> None:
    for (domain,), d in diag.filter(pl.col("n_frames") >= cfg.min_frames_trajectory).group_by("domain", maintain_order=True):
        cl, v, dur = d["nfl_id"].to_numpy(), d["mae"].to_numpy().astype(float), d["duration_s"].to_numpy().astype(float)
        b = cluster_bootstrap(lambda i: spearman(v[i], dur[i]), cl, cfg.n_boot_generic, _seed(seed, domain, "mae_dur"))
        rc.add(analysis_id="A2.error_vs_duration", grain="trajectory", n_athletes=int(np.unique(cl).size),
               n_trajectories=d.height, effect_estimate=b["estimate"], effect_size_type="spearman_rho",
               method="Spearman ρ (trajectory MAE vs duration)", domain=domain, feature="trajectory_mae",
               outcome="duration_s", standard_error=b["se"], ci_lower=b["ci_lower"], ci_upper=b["ci_upper"],
               ci_method=f"athlete-cluster bootstrap (B={cfg.n_boot_generic}, percentile)",
               quality_filter=f"≥ {cfg.min_frames_trajectory} valid frames")


def a2_examples(fr: pl.DataFrame, diag: pl.DataFrame, cfg: QualityStudyConfig) -> pl.DataFrame:
    top = (diag.filter(pl.col("n_frames") >= cfg.min_frames_trajectory)
           .sort(["domain", "mae", "traj_id"], descending=[False, True, False])
           .group_by("domain", maintain_order=True).head(cfg.n_examples)
           .with_columns((pl.col("domain") + "_" + (pl.int_range(pl.len()).over("domain") + 1).cast(pl.String)).alias("example")))
    if not top.height:
        return A2_EXAMPLES.empty()
    return (fr.join(top.select("domain", "traj_id", "example", pl.col("mae").alias("trajectory_mae")),
                    on=["domain", "traj_id"], how="inner")
            .select("example", "domain", "traj_id", "trajectory_mae", pl.col("t_s").round(4),
                    pl.col("s").alias("vendor_speed"), "speed")
            .unique(["example", "t_s"], keep="first").sort("example", "t_s"))


def a1_signals(fr: pl.DataFrame, ex: pl.DataFrame) -> pl.DataFrame:
    if not ex.height:
        return A1_SIGNALS.empty()
    f = fr.join(ex.select("domain", "traj_id", "example"), on=["domain", "traj_id"], how="inner").sort("example", "t_s")
    return (f.with_columns((pl.col("x_s") - pl.col("x_s").first().over("example")).alias("x_rel"),
                           (pl.col("y_s") - pl.col("y_s").first().over("example")).alias("y_rel"))
            .select("example", pl.col("t_s").round(4), "x_rel", "y_rel", "speed", pl.col("s").alias("vendor_speed"))
            .unique(["example", "t_s"], keep="first").sort("example", "t_s"))


# ============================================================================== orchestration
class QualityStudy:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, scfg: StudiesConfig | None = None, store: StudyStore | None = None,
                 kcfg: KinematicsConfig | None = None, skip_nfl_sample: bool = False):
        self.cfg = cfg
        self.scfg = scfg or StudiesConfig()
        self.q = self.scfg.quality
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        self.kin = KinematicFeatureBuilder(cfg, self.kcfg)
        self.src = self.kin.store                      # M1 store (read only)
        self.store = store or StudyStore(cfg, sampled=self.scfg.sample_athletes is not None)
        self.skip_nfl_sample = skip_nfl_sample

    # ------------------------------------------------------------------ inputs
    def _has(self, spec) -> bool:
        return bool(self.src.dataset_files(spec.name))

    def upstream(self) -> list[str]:
        mandatory = [COMBINE_TRAJ, COMBINE_SEGMENTS, COMBINE_EVENTS, COMBINE_FRAMES]
        missing = [s.name for s in mandatory if not self._has(s)]
        if missing:
            raise FileNotFoundError(f"mandatory M1 datasets missing {missing} — run kinematic_build --domain combine")
        return [s.name for s in (*mandatory, NFL_TRAJ, NFL_SEGMENTS) if self._has(s)]

    def positions(self) -> pl.DataFrame:
        if not self.src.refs.files("gold", "dim_athlete"):
            log.warning("[studies:A] gold.dim_athlete missing — every position is 'UNKNOWN'")
            return pl.DataFrame(schema={"nfl_id": pl.Int64, "position_group": pl.String})
        return self.src.refs.scan("gold", "dim_athlete").select("nfl_id", "position_group").collect()

    def sample_ids(self, ids: np.ndarray) -> list[int] | None:
        if self.scfg.sample_athletes is None:
            return None
        ids = np.sort(np.unique(ids))
        k = min(self.scfg.sample_athletes, ids.size)
        return sorted(np.random.default_rng(self.scfg.seed).choice(ids, size=k, replace=False).tolist())

    def _load(self, spec, ids: list[int] | None) -> pl.DataFrame:
        lf = self.src.load(spec.name)
        return (lf.filter(pl.col("nfl_id").is_in(ids)) if ids is not None else lf).collect()

    def trajectories(self, pos: pl.DataFrame) -> tuple[pl.DataFrame, list[int] | None]:
        ckeys, nkeys = ["event_id", "nfl_id", "attempt"], ["game_id", "play_id", "nfl_id"]
        ids = self.sample_ids(self.src.load(COMBINE_TRAJ.name).select("nfl_id").collect()["nfl_id"].to_numpy()
                              if self.scfg.sample_athletes is not None else np.array([]))
        parts = [normalize_trajectories(self._load(COMBINE_TRAJ, ids), self._load(COMBINE_SEGMENTS, ids),
                                        self._load(COMBINE_EVENTS, ids), "combine", ckeys, "draft_year", "drill_type",
                                        pos)]
        if self._has(NFL_TRAJ) and self._has(NFL_SEGMENTS):
            parts.append(normalize_trajectories(self._load(NFL_TRAJ, ids), self._load(NFL_SEGMENTS, ids), None, "nfl",
                                                nkeys, "season", None, pos))
        else:
            log.warning("[studies:A] NFL trajectory quality not built — A1 covers Combine only")
        return pl.concat(parts, how="vertical"), ids

    def frames(self, t: pl.DataFrame, ids: list[int] | None, pos: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """(frame residual table, quality dimensions of the NFL-sample trajectories — examples only; the
        sample is screened against its own quantiles and is NOT mixed into the full-NFL A1 tables)."""
        cols = ["event_id", "nfl_id", "attempt", "t_s", "x_s", "y_s", "s", "speed", "accel_mag", "valid_kin"]
        cf = self._load(COMBINE_FRAMES, ids).select(cols)
        parts = [vendor_frames(cf, "combine", ["event_id", "nfl_id", "attempt"], t, self.q)]
        st = DIMS.empty()
        if not self.skip_nfl_sample:
            from ..kinematic_diagnostics import KinematicDiagnostics

            s = KinematicDiagnostics(self.cfg, self.kcfg, self.src,
                                     nfl_games_per_season=self.q.nfl_games_per_season).nfl_sample()
            nk = ["game_id", "play_id", "nfl_id"]
            st, _ = quality_dimensions(normalize_trajectories(s["trajectories"], s["segments"], None, "nfl_sample", nk,
                                                              "season", None, pos), self.q)
            nf = s["frames"] if ids is None else s["frames"].filter(pl.col("nfl_id").is_in(ids))
            parts.append(vendor_frames(nf.select(*nk, *cols[3:]), "nfl_sample", nk, st, self.q))
        return pl.concat(parts, how="vertical"), st

    # ------------------------------------------------------------------ compute
    def compute(self) -> dict[str, pl.DataFrame]:
        seed, q = self.scfg.seed, self.q
        pos = self.positions()
        t_raw, ids = self.trajectories(pos)
        t, thr = quality_dimensions(t_raw, q)
        rc = ResultCollector(STUDY, q.min_athletes)
        log.info("[studies:A] %s trajectories (%s)", f"{t.height:,}", t.group_by("domain").len().rows())
        props = proportions(t, q, seed)
        for r in props.filter(pl.col("indicator").is_in(["tier_A", "any_flag"]) & pl.col("facet").is_in(["all", "cohort"])).iter_rows(named=True):
            rc.add(analysis_id="A1.proportion", grain="trajectory", n_athletes=r["n_athletes"],
                   n_trajectories=r["n_trajectories"], effect_estimate=r["share"], effect_size_type="proportion",
                   method="share of trajectories", domain=r["domain"],
                   cohort=r["level"] if r["facet"] == "cohort" else "all", feature=r["indicator"],
                   ci_lower=r["boot_lower"], ci_upper=r["boot_upper"],
                   ci_method=f"athlete-cluster bootstrap (B={q.n_boot}, percentile)", quality_filter="none (all tiers)")
        a1_results(t, q, seed, rc)

        fr, sample_dims = self.frames(t, ids, pos)
        diag = frame_diagnostics(fr, q)
        a2_trajectory_results(diag, q, seed, rc)
        vendor = vendor_error_summary(fr, q, seed, rc)
        # examples need frames: Combine (persisted) and the NFL sample (full NFL frames are not persisted)
        avail = {d: set(fr.filter(pl.col("domain") == d)["traj_id"].unique().to_list()) for d in ("combine", "nfl_sample")}
        ex1 = select_a1_examples(pl.concat([t.filter(pl.col("domain") == "combine"), sample_dims]), avail)
        return {DIMS.name: t, THRESHOLDS_TBL.name: thr, PROPORTIONS.name: props,
                METRIC_BY_TIER.name: metric_by_tier(t), REASONS.name: reason_matrix(t), FUNNEL.name: filter_funnel(t),
                EXTREME_ACCEL.name: extreme_accel_by_tier(diag, q, seed), A1_EXAMPLES.name: ex1,
                A1_SIGNALS.name: a1_signals(fr, ex1), VENDOR.name: vendor, FRAME_DIAG.name: diag,
                HIST2D.name: hist2d(fr, q), BY_SPEED.name: residual_by_speed(fr),
                A2_EXAMPLES.name: a2_examples(fr, diag, q), STUDY_RESULTS.name: rc.frame()}

    # ------------------------------------------------------------------ run
    def fingerprint(self) -> str:
        return self.store.fingerprint(STUDY, {**self.scfg.section("quality"), "skip_nfl_sample": self.skip_nfl_sample},
                                      upstream_state(self.src, self.upstream()))

    def run(self, force: bool = False, figures: bool = True) -> dict[str, pl.DataFrame]:
        self.store.write(threshold_table(self.scfg), THRESHOLD_DOC, STUDY)
        fp = self.fingerprint()
        current, reason = self.store.is_current(STUDY, fp, CONTRACTS)
        if current and not force:
            log.info("[studies:A] reuse (%s)", reason)
            out = self.load()
        else:
            log.info("[studies:A] compute (%s%s)", "forced; " if force else "", reason)
            res = self.compute()
            out = {c.name: self.store.write(res[c.name], c, STUDY) for c in CONTRACTS}
            self.store.record_run(STUDY, fp, CONTRACTS, {"skip_nfl_sample": self.skip_nfl_sample})
        if figures:
            self.figures(out)
        return out

    def load(self) -> dict[str, pl.DataFrame]:
        return {c.name: self.store.read(c) for c in CONTRACTS}

    def figures(self, out: dict[str, pl.DataFrame]) -> None:
        from .quality_figures import draw_all

        draw_all(self.store, out, self.q)
