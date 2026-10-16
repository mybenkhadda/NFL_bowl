"""Study B — smoothing robustness of trajectory-level kinematic summaries (Phase 04 · M1 · Study 1.2).

Question     Which movement summaries keep their VALUE and their athlete ORDERING when the Savitzky–Golay
             setting changes within the range M1 already declared (sg5p2, sg7p2 = primary, sg9p2, sg11p3)?
             M1 evidence (executed): trajectory Spearman ≥ 0.92 for every feature, but median relative change
             up to 14 % (Combine peak_decel, sg5p2) — rank stability ≠ magnitude stability.
Population   all Combine PLAYER attempts (raw silver frames, ~0.46 M) + the deterministic NFL game sample
             (8 games / season, same selection as M1 / Study A2 / A3). Bounded recomputation only.
Grain        frame (vendor residual), trajectory (feature values), athlete (median over the athlete's
             trajectories in the drill). Unit of independence: athlete — trajectory CIs are athlete-cluster
             bootstrap intervals.
Features     peaks (all frames and filter-INTERIOR frames, A3 primitive), robust quantiles (p95 / p90),
             path length, and primary-rule event counts and rates per valid second. The interior and quantile
             variants exist because A3 showed peaks concentrate on SG edge frames (e.g. Combine three-cone
             peak speed 11× the expected share) — they are candidate replacements, not post-hoc fixes.
Statistics   per domain × drill × feature × setting (vs primary), at trajectory AND athlete level:
             Spearman (cluster-bootstrap CI), Pearson, Bland–Altman bias / LoA, median |Δ| and relative |Δ|,
             top-decile overlap (CI at athlete level), |percentile-rank change| quantiles;
             variance share of the setting (all four settings jointly);
             ICC(1) of repeated Combine attempts under EACH setting (does repeatability depend on smoothing?);
             cohort / position robustness of the athlete-level Spearman at the extreme windows;
             frame-level vendor residual by speed band under each setting (vendor speed is NOT ground truth).
Persistence  gold/phase_04/quality/trajectory_features_by_smoothing (long),
             gold/phase_04/movement_events/events_by_smoothing (all rules × settings; read by Study C1),
             tables/phase_04/{descriptive,statistical,reliability}/B_*.
Grain        agreement / variance share / ICC tables: domain × stratum (drill_type) × drill × feature [× setting
             × level]. Five canonical skill drills are run by several position groups, so ``drill`` alone is
             not a key. ``B_robustness_facets`` pools drill_types within a drill (the facet is the subgroup).
Run cache    two records in the studies manifest: ``B.frames`` (frame-derived outputs, written first) and ``B``
             (complete study). A failure in the analysis stage leaves ``B.frames`` valid: the next
             ``--only B`` reads the long table from disk instead of recomputing frames.

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.phase04 --only B --sample-athletes 120     # smoke → */phase_04/_sample
    python -m nflbdb.research.phase04 --only B
"""

from __future__ import annotations

import dataclasses
import gc
import zlib
from typing import Any

import numpy as np
import polars as pl

from ...data.config import Phase01Config
from ...data.utils import get_logger
from ...features.config import KinematicsConfig
from ...features.events import EVENT_TYPES, detect_events
from ...features.kinematics import compute_kinematics
from ...features.reliability import icc_bootstrap, icc_oneway, reliability_label
from ...features.signal_integrity import add_signal_columns
from ...features.trajectory import COMBINE, NFL, TrajectoryDomain, prepare_frames
from ..contracts import TableContract, cast_keys, upstream_state
from ..kinematic_build import COMBINE_TRAJ, KinematicFeatureBuilder
from ..studies.quality import traj_id_expr
from ..studies.results import RESULTS, ResultCollector
from ..studies.stats import cluster_bootstrap, spearman
from ..studies.store import FigureRecord, StudyStore
from .config import Phase04Config, SmoothingStudyConfig
from .metrics import agreement, percentile_rank, top_fraction_overlap, variance_share

log = get_logger(__name__)
STUDY = "B"
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
CKEYS, NKEYS = ["event_id", "nfl_id", "attempt"], ["game_id", "play_id", "nfl_id"]

# feature → (unit, family). Families drive the registry criteria (Study V).
FEATURES: dict[str, tuple[str, str]] = {
    "peak_speed": ("yd/s", "speed_peak"), "peak_speed_interior": ("yd/s", "speed_interior"),
    "p95_speed": ("yd/s", "speed_quantile"), "path_smooth_yds": ("yd", "path"),
    "peak_a_tan": ("yd/s²", "accel_peak"), "peak_a_tan_interior": ("yd/s²", "accel_interior"),
    "p95_a_tan": ("yd/s²", "accel_quantile"),
    "peak_decel": ("yd/s²", "decel_peak"), "peak_decel_interior": ("yd/s²", "decel_interior"),
    "p95_decel": ("yd/s²", "decel_quantile"),
    "peak_accel_mag": ("yd/s²", "accel_mag_peak"), "peak_accel_mag_interior": ("yd/s²", "accel_mag_interior"),
    "peak_abs_yaw_rate": ("deg/s", "yaw_peak"), "peak_abs_yaw_rate_interior": ("deg/s", "yaw_interior"),
    "p90_abs_yaw_rate": ("deg/s", "yaw_quantile"),
    **{f"n_{e}": ("events", "event_count") for e in EVENT_TYPES},
    **{f"rate_{e}_per_s": ("events/s", "event_rate") for e in EVENT_TYPES},
    "duration_valid_s": ("s", "exposure"),      # exposure for rates (Study C1); not a movement feature
}
FEATURE_NAMES = list(FEATURES)
# peak features whose A3 boundary evidence applies (feature → A3 peak name)
BOUNDARY_PEAK = {"peak_speed": "peak_speed", "peak_a_tan": "peak_a_tan", "peak_decel": "peak_decel",
                 "peak_accel_mag": "peak_accel_mag", "peak_abs_yaw_rate": "peak_abs_yaw_rate"}
BA_FEATURES = ("peak_speed", "peak_a_tan", "peak_decel", "peak_abs_yaw_rate")
LEVELS = ("trajectory", "athlete")
G = ["domain", "stratum", "drill", "feature"]

# ============================================================================== contracts
FEATURES_DS = TableContract(
    "trajectory_features_by_smoothing", "Long table: trajectory × SG setting × feature value (Study B).",
    {"domain": S, "traj_id": S, "nfl_id": I, "cohort": I, "stratum": S, "drill": S, "position": S,
     "attempt_order": I, "smoothing": S, "is_primary": B, "feature": S, "value": F},
    key=("domain", "traj_id", "smoothing", "feature"), kind="dataset", area="quality",
    docs={"attempt_order": "Combine: chronological order of the athlete's attempts within the drill (event_id, "
                           "attempt); NFL: null", "stratum": "Combine drill_type · NFL 'nfl_play'",
          "drill": "canonical Combine drill · NFL 'nfl_play'", "value": "unit per feature: see B_feature_catalog"})
EVENTS_DS = TableContract(
    "events_by_smoothing", "Detected events for every SG setting × threshold rule (Study B → read by C1).",
    {"domain": S, "traj_id": S, "nfl_id": I, "stratum": S, "drill": S, "smoothing": S, "rule": S, "event_type": S,
     "seg_idx": I, "event_idx": I, "t_start_s": F, "t_end_s": F, "duration_s": F, "peak_abs_a_tan": F,
     "peak_abs_yaw_rate": F, "heading_change_deg": F, "touches_segment_edge": B},
    key=("domain", "traj_id", "smoothing", "rule", "event_type", "seg_idx", "event_idx"), kind="dataset",
    area="movement_events")
CATALOG = TableContract(
    "B_feature_catalog", "Feature definitions used by Phase 04 validation.",
    {"feature": S, "unit": S, "family": S, "definition": S}, key=("feature",), area="descriptive")
AGREE = TableContract(
    "B_smoothing_agreement", "Each SG setting vs the primary: agreement at trajectory and athlete level.",
    {"domain": S, "stratum": S, "drill": S, "feature": S, "smoothing": S, "level": S, "n_units": I, "n_athletes": I,
     "spearman": F, "spearman_lower": F, "spearman_upper": F, "pearson": F, "bias": F, "loa_lower": F,
     "loa_upper": F, "median_abs_diff": F, "p95_abs_diff": F, "median_rel_diff": F, "p90_rel_diff": F,
     "top_overlap": F, "top_overlap_lower": F, "top_overlap_upper": F, "rank_change_median": F,
     "rank_change_p90": F, "modal_share": F, "result_status": S},
    key=("domain", "stratum", "drill", "feature", "smoothing", "level"), area="statistical",
    docs={"stratum": "Combine drill_type · NFL 'nfl_play'. Part of the grain: a canonical drill run by several "
                     "position groups (GAUNTLET_DRILL, OVER_SHOULDER_ADJUST, PASS_RUSH_DRILL, BACK_PEDAL_AND_REACT, "
                     "SHORT_ZONE_BREAKS) has one row per drill_type",
          "level": "trajectory = one value per trajectory; athlete = median over the athlete's trajectories",
          "top_overlap": "share of the primary top decile that stays in the top decile",
          "modal_share": "share of units at the modal primary value (≥ 0.5 → ties dominate ranks / overlap)",
          "spearman_lower": "athlete-cluster bootstrap percentile interval"})
VARSHARE = TableContract(
    "B_variance_share", "Share of variance due to the SG setting (all settings jointly).",
    {"domain": S, "stratum": S, "drill": S, "feature": S, "level": S, "n_units": I, "var_setting": F, "var_unit": F,
     "share_setting": F}, key=("domain", "stratum", "drill", "feature", "level"), area="statistical")
REPEAT = TableContract(
    "B_repeatability_by_setting", "ICC(1) of repeated Combine attempts under each SG setting.",
    {"domain": S, "stratum": S, "drill": S, "feature": S, "smoothing": S, "n_athletes": I, "n_attempts": I,
     "icc1": F, "ci_lower": F, "ci_upper": F, "reliability": S},
    key=("domain", "stratum", "drill", "feature", "smoothing"), area="reliability",
    docs={"reliability": "features.reliability.reliability_label on the CI lower bound (excellent ≥ .9, good ≥ .75, "
                         "moderate ≥ .5)"})
ROBUST = TableContract(
    "B_robustness_facets", "Athlete-level Spearman vs primary by cohort / position (extreme SG windows).",
    {"domain": S, "drill": S, "feature": S, "smoothing": S, "facet": S, "level": S, "n_athletes": I, "spearman": F,
     "median_rel_diff": F}, key=("domain", "drill", "feature", "smoothing", "facet", "level"), area="statistical",
    docs={"drill": "canonical drill POOLED over drill_types (no stratum column): the facet is the subgroup here"})
VENDOR = TableContract(
    "B_vendor_residual_by_setting", "Frame-level derived − vendor speed by speed band under each SG setting.",
    {"domain": S, "smoothing": S, "speed_band": S, "band_lower": F, "n_frames": I, "n_trajectories": I,
     "n_athletes": I, "mae": F, "bias": F, "rmse": F, "p95_abs": F},
    key=("domain", "smoothing", "speed_band"), area="descriptive",
    docs={"mae": "frame grain — describes frames, not athletes; vendor speed is a reference, not ground truth"})
EXAMPLES = TableContract(
    "B_example_signals", "Speed and tangential acceleration of deterministic example attempts under each setting.",
    {"domain": S, "drill": S, "traj_id": S, "smoothing": S, "t_s": F, "speed": F, "a_tan": F},
    key=("traj_id", "smoothing", "t_s"), area="descriptive")
STUDY_RESULTS = dataclasses.replace(RESULTS, name="B_results")
# Two stages with their own run record. FRAME_CONTRACTS need the raw-frame recomputation (heavy) and are written
# and recorded FIRST; ANALYSIS_CONTRACTS are derived from the persisted long table only, so a failure there never
# costs the frame pass again.
FRAME_STAGE = "B.frames"
FRAME_CONTRACTS = (FEATURES_DS, EVENTS_DS, CATALOG, VENDOR, EXAMPLES)
ANALYSIS_CONTRACTS = (AGREE, VARSHARE, REPEAT, ROBUST, STUDY_RESULTS)
CONTRACTS = (*FRAME_CONTRACTS, *ANALYSIS_CONTRACTS)
FRAME_PARAMS = ("nfl_games_per_season", "speed_bands", "example_drills")   # SmoothingStudyConfig fields used by frames

DEFINITIONS = {
    "peak_speed": "max derived speed over valid frames", "peak_speed_interior": "max speed over valid frames NOT in "
    "the SG filter boundary (edge distance < window // 2)", "p95_speed": "95th percentile of speed (valid frames)",
    "path_smooth_yds": "Σ smoothed step length", "peak_a_tan": "max signed tangential acceleration d|v|/dt",
    "peak_a_tan_interior": "max a_tan over interior frames", "p95_a_tan": "95th percentile of a_tan",
    "peak_decel": "max of −a_tan (strongest braking)", "peak_decel_interior": "max −a_tan over interior frames",
    "p95_decel": "95th percentile of −a_tan", "peak_accel_mag": "max |vector acceleration|",
    "peak_accel_mag_interior": "max |a| over interior frames", "peak_abs_yaw_rate": "max |yaw rate| (speed ≥ 1 yd/s)",
    "peak_abs_yaw_rate_interior": "max |yaw rate| over interior frames", "p90_abs_yaw_rate": "90th percentile of |yaw rate|",
    **{f"n_{e}": f"number of primary-rule {e} events" for e in EVENT_TYPES},
    **{f"rate_{e}_per_s": f"primary-rule {e} events per valid tracked second" for e in EVENT_TYPES},
    "duration_valid_s": "valid-kinematics frames × nominal Δt (exposure; segments shorter than the SG window "
                        "have no derivatives, so it can differ slightly between settings)",
}


def _seed(base: int, *parts: Any) -> int:
    return (base + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32)


def stratum_label(stratum: str, drill: str) -> str:
    """Value of the single ``stratum`` field of RESULTS for a stratum × drill cell: the drill when it names its
    own stratum (measured drills, NFL), else ``drill_type:drill`` (skill drills shared by position groups)."""
    return drill if stratum == drill else f"{stratum}:{drill}"


# ============================================================================== pure feature extraction
def trajectory_features(kin: pl.DataFrame, events: pl.DataFrame, domain: TrajectoryDomain, kcfg: KinematicsConfig
                        ) -> pl.DataFrame:
    """One row per trajectory: FEATURES (+ context). ``kin`` must already carry ``in_filter_boundary``
    (``add_signal_columns`` with the SAME smoothing spec); ``events``: ``detect_events`` output (any rules)."""
    keys = list(domain.keys)
    v, b = pl.col("valid_kin"), pl.col("in_filter_boundary")
    a_tan, yaw = pl.col("a_tan"), pl.col("yaw_rate").abs()
    ctx = [pl.col(c).first() for c in domain.context if c in kin.columns]
    wide = kin.group_by(keys).agg(
        *ctx,
        (v.cast(pl.Int64).sum() * kcfg.dt_s).alias("duration_valid_s"),
        pl.col("speed").filter(v).max().alias("peak_speed"),
        pl.col("speed").filter(v & ~b).max().alias("peak_speed_interior"),
        pl.col("speed").filter(v).quantile(0.95, "linear").alias("p95_speed"),
        pl.col("step_smooth").sum().alias("path_smooth_yds"),
        a_tan.filter(v).max().alias("peak_a_tan"), a_tan.filter(v & ~b).max().alias("peak_a_tan_interior"),
        a_tan.filter(v).quantile(0.95, "linear").alias("p95_a_tan"),
        (-a_tan).filter(v).max().alias("peak_decel"), (-a_tan).filter(v & ~b).max().alias("peak_decel_interior"),
        (-a_tan).filter(v).quantile(0.95, "linear").alias("p95_decel"),
        pl.col("accel_mag").filter(v).max().alias("peak_accel_mag"),
        pl.col("accel_mag").filter(v & ~b).max().alias("peak_accel_mag_interior"),
        yaw.filter(v).max().alias("peak_abs_yaw_rate"), yaw.filter(v & ~b).max().alias("peak_abs_yaw_rate_interior"),
        yaw.filter(v).quantile(0.90, "linear").alias("p90_abs_yaw_rate"),
    )
    prim = events.filter(pl.col("rule") == kcfg.primary_rule) if events.height else events
    if prim.height:
        cnt = prim.group_by(keys).agg(*[(pl.col("event_type") == e).sum().cast(pl.Float64).alias(f"n_{e}")
                                        for e in EVENT_TYPES])
        wide = wide.join(cnt, on=keys, how="left")
    else:
        wide = wide.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(f"n_{e}") for e in EVENT_TYPES])
    has = pl.col("duration_valid_s") > 0
    return wide.with_columns(
        *[pl.when(has).then(pl.col(f"n_{e}").fill_null(0.0)).alias(f"n_{e}") for e in EVENT_TYPES]).with_columns(
        *[pl.when(has).then(pl.col(f"n_{e}") / pl.col("duration_valid_s")).alias(f"rate_{e}_per_s")
          for e in EVENT_TYPES]).with_columns([pl.col(c).cast(pl.Float64) for c in FEATURE_NAMES])


def band_expr(col: str, edges: tuple[float, ...]) -> tuple[pl.Expr, pl.Expr]:
    """(label, lower edge) of the speed band containing ``col`` (bands are [e_k, e_{k+1}), last open)."""
    lab, low = pl.lit(None, dtype=pl.String), pl.lit(None, dtype=pl.Float64)
    for k, e in enumerate(edges):
        hi = edges[k + 1] if k + 1 < len(edges) else None
        name = f"[{e:g}, {hi:g})" if hi is not None else f"≥ {e:g}"
        cond = pl.col(col) >= e
        lab = pl.when(cond).then(pl.lit(name)).otherwise(lab)
        low = pl.when(cond).then(pl.lit(float(e))).otherwise(low)
    return lab.alias("speed_band"), low.alias("band_lower")


def vendor_residual(kin: pl.DataFrame, keys: list[str], edges: tuple[float, ...]) -> pl.DataFrame:
    lab, low = band_expr("s", edges)
    f = (kin.filter(pl.col("valid_kin") & pl.col("s").is_not_null() & pl.col("speed").is_not_null())
         .select(traj_id_expr(keys), pl.col("nfl_id").cast(pl.Int64),
                 (pl.col("speed").cast(pl.Float64) - pl.col("s").cast(pl.Float64)).alias("r"), lab, low)
         .filter(pl.col("speed_band").is_not_null()))      # vendor s below the first edge (negative) — none expected
    return f.group_by("speed_band", "band_lower").agg(
        pl.len().cast(pl.Int64).alias("n_frames"), pl.col("traj_id").n_unique().cast(pl.Int64).alias("n_trajectories"),
        pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"), pl.col("r").abs().mean().alias("mae"),
        pl.col("r").mean().alias("bias"), (pl.col("r") ** 2).mean().sqrt().alias("rmse"),
        pl.col("r").abs().quantile(0.95, "linear").alias("p95_abs"))


def normalize_features(wide: pl.DataFrame, domain: TrajectoryDomain, positions: pl.DataFrame, label: str,
                       primary: str) -> pl.DataFrame:
    keys = list(domain.keys)
    w = cast_keys(wide).with_columns(traj_id_expr(keys)).join(positions, on="nfl_id", how="left")
    if domain.name == "combine":
        w = w.sort("nfl_id", "drill", "event_id", "attempt").with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("nfl_id", "drill").alias("attempt_order"),
            pl.col("draft_year").cast(pl.Int64).alias("cohort"), pl.col("drill_type").cast(pl.String).alias("stratum"),
            pl.col("drill").cast(pl.String))
        dom = "combine"
    else:
        w = w.with_columns(pl.lit(None, dtype=pl.Int64).alias("attempt_order"),
                           pl.col("season").cast(pl.Int64).alias("cohort"), pl.lit("nfl_play").alias("stratum"),
                           pl.lit("nfl_play").alias("drill"))
        dom = "nfl_sample"
    return (w.select(pl.lit(dom).alias("domain"), "traj_id", pl.col("nfl_id").cast(pl.Int64), "cohort", "stratum",
                     "drill", pl.col("position_group").fill_null("UNKNOWN").alias("position"), "attempt_order",
                     pl.lit(label).alias("smoothing"), pl.lit(label == primary).alias("is_primary"), *FEATURE_NAMES)
            .unpivot(index=["domain", "traj_id", "nfl_id", "cohort", "stratum", "drill", "position", "attempt_order",
                            "smoothing", "is_primary"], on=FEATURE_NAMES, variable_name="feature", value_name="value")
            .with_columns(pl.col("value").cast(pl.Float64)))


def normalize_events(ev: pl.DataFrame, domain: TrajectoryDomain, label: str) -> pl.DataFrame:
    keys = list(domain.keys)
    if not ev.height:
        return EVENTS_DS.empty()
    e = cast_keys(ev).with_columns(traj_id_expr(keys))
    if domain.name == "combine":
        e = e.with_columns(pl.lit("combine").alias("domain"), pl.col("drill_type").cast(pl.String).alias("stratum"),
                           pl.col("drill").cast(pl.String))
    else:
        e = e.with_columns(pl.lit("nfl_sample").alias("domain"), pl.lit("nfl_play").alias("stratum"),
                           pl.lit("nfl_play").alias("drill"))
    return e.select("domain", "traj_id", pl.col("nfl_id").cast(pl.Int64), "stratum", "drill",
                    pl.lit(label).alias("smoothing"), pl.col("rule").cast(pl.String),
                    pl.col("event_type").cast(pl.String), pl.col("seg_idx").cast(pl.Int64),
                    pl.col("event_idx").cast(pl.Int64), *[pl.col(c).cast(pl.Float64) for c in (
                        "t_start_s", "t_end_s", "duration_s", "peak_abs_a_tan", "peak_abs_yaw_rate",
                        "heading_change_deg")], pl.col("touches_segment_edge").cast(pl.Boolean))


# ============================================================================== pure analysis
def athlete_values(long: pl.DataFrame) -> pl.DataFrame:
    """Median value per athlete × drill × feature × setting (athlete-level measurement)."""
    return (long.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .group_by("domain", "stratum", "drill", "feature", "smoothing", "is_primary", "nfl_id")
            .agg(pl.col("value").median().alias("value"), pl.col("cohort").first(), pl.col("position").first())
            .with_columns(pl.col("nfl_id").cast(pl.String).alias("traj_id")))


def paired(long: pl.DataFrame) -> pl.DataFrame:
    """Alternative-setting values joined to the primary value of the same unit (``traj_id`` = unit id)."""
    d = long.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
    ref = d.filter(pl.col("is_primary")).select(*G, "traj_id", pl.col("value").alias("v0"))
    return d.filter(~pl.col("is_primary")).join(ref, on=[*G, "traj_id"], how="inner")


def _modal_share(x: np.ndarray) -> float:
    if not x.size:
        return float("nan")
    _, c = np.unique(np.round(x, 9), return_counts=True)
    return float(c.max() / x.size)


def agreement_table(long: pl.DataFrame, cfg: SmoothingStudyConfig, seed: int) -> pl.DataFrame:
    rows = []
    for level in LEVELS:
        base = long if level == "trajectory" else athlete_values(long)
        pr = paired(base)
        for (domain, stratum, drill, feature, smoothing), g in pr.group_by(*G, "smoothing", maintain_order=True):
            v, v0, cl = g["value"].to_numpy(), g["v0"].to_numpy(), g["nfl_id"].to_numpy()
            n_ath = int(np.unique(cl).size)
            a = agreement(v, v0, cfg.top_fraction)
            row = {"domain": domain, "stratum": stratum, "drill": drill, "feature": feature, "smoothing": smoothing,
                   "level": level, "n_units": int(a["n"]), "n_athletes": n_ath, "spearman_lower": None,
                   "spearman_upper": None, "top_overlap_lower": None, "top_overlap_upper": None,
                   "modal_share": _modal_share(v0),
                   **{k: a[k] for k in ("spearman", "pearson", "bias", "loa_lower", "loa_upper", "median_abs_diff",
                                        "p95_abs_diff", "median_rel_diff", "p90_rel_diff", "top_overlap",
                                        "rank_change_median", "rank_change_p90")}}
            ok = n_ath >= cfg.min_athletes and a["n"] >= (cfg.min_trajectories if level == "trajectory" else
                                                          cfg.min_athletes)
            row["result_status"] = "exploratory" if ok else "insufficient_data"
            if ok:
                s0 = _seed(seed, level, domain, stratum, drill, feature, smoothing)
                bs = cluster_bootstrap(lambda idx: spearman(v[idx], v0[idx]), cl, cfg.n_boot, s0)
                row["spearman_lower"], row["spearman_upper"] = bs["ci_lower"], bs["ci_upper"]
                if level == "athlete":
                    bo = cluster_bootstrap(lambda idx: top_fraction_overlap(v[idx], v0[idx], cfg.top_fraction), cl,
                                           cfg.n_boot, s0 + 1)
                    row["top_overlap_lower"], row["top_overlap_upper"] = bo["ci_lower"], bo["ci_upper"]
            rows.append(row)
    return (pl.DataFrame(rows, schema=AGREE.columns, strict=False) if rows else AGREE.empty()).sort(
        "domain", "stratum", "drill", "feature", "smoothing", "level")


def variance_table(long: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for level in LEVELS:
        base = long if level == "trajectory" else athlete_values(long)
        wide = (base.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
                .select(*G, "traj_id", "smoothing", "value").pivot(on="smoothing", index=[*G, "traj_id"], values="value"))
        settings = [c for c in wide.columns if c not in (*G, "traj_id")]
        for (domain, stratum, drill, feature), g in wide.group_by(*G, maintain_order=True):
            vs = variance_share(g.select(settings).to_numpy())
            rows.append({"domain": domain, "stratum": stratum, "drill": drill, "feature": feature, "level": level,
                         "n_units": vs["n_units"], "var_setting": vs["var_setting"], "var_unit": vs["var_unit"],
                         "share_setting": vs["share_setting"]})
    return pl.DataFrame(rows, schema=VARSHARE.columns, strict=False) if rows else VARSHARE.empty()


def repeatability_table(long: pl.DataFrame, cfg: SmoothingStudyConfig, seed: int) -> pl.DataFrame:
    d = (long.filter((pl.col("domain") == "combine") & pl.col("value").is_not_null() & pl.col("value").is_finite()
                     & (pl.col("attempt_order") <= cfg.max_attempts))
         .with_columns(pl.len().over("stratum", "drill", "feature", "smoothing", "nfl_id").alias("_k"))
         .filter(pl.col("_k") >= cfg.min_attempts))
    rows = []
    for (stratum, drill, feature, smoothing), g in d.group_by("stratum", "drill", "feature", "smoothing",
                                                             maintain_order=True):
        groups = g["nfl_id"].to_numpy()
        vals = g["value"].to_numpy().astype(float)
        n_ath = int(np.unique(groups).size)
        if n_ath < cfg.min_athletes:
            icc, lo, hi = float("nan"), float("nan"), float("nan")
        else:
            icc = icc_oneway(vals, groups)["icc"]
            lo, hi = icc_bootstrap(vals, groups, cfg.n_boot, _seed(seed, "icc", stratum, drill, feature, smoothing))
        rows.append({"domain": "combine", "stratum": stratum, "drill": drill, "feature": feature,
                     "smoothing": smoothing, "n_athletes": n_ath, "n_attempts": g.height,
                     "icc1": icc if np.isfinite(icc) else None, "ci_lower": lo if np.isfinite(lo) else None,
                     "ci_upper": hi if np.isfinite(hi) else None,
                     "reliability": reliability_label(lo) if n_ath >= cfg.min_athletes else "insufficient_athletes"})
    return pl.DataFrame(rows, schema=REPEAT.columns, strict=False) if rows else REPEAT.empty()


def robustness_table(long: pl.DataFrame, cfg: SmoothingStudyConfig) -> pl.DataFrame:
    d = long.filter(pl.col("smoothing").is_in(list(cfg.robustness_settings)) | pl.col("is_primary"))
    rows = []
    for facet in ("cohort", "position"):
        av = (d.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
              .group_by("domain", "drill", "feature", "smoothing", "is_primary", facet, "nfl_id")
              .agg(pl.col("value").median()))
        ref = av.filter(pl.col("is_primary")).select("domain", "drill", "feature", facet, "nfl_id",
                                                      pl.col("value").alias("v0"))
        pr = av.filter(~pl.col("is_primary")).join(ref, on=["domain", "drill", "feature", facet, "nfl_id"], how="inner")
        for (domain, drill, feature, smoothing, level), g in pr.group_by("domain", "drill", "feature", "smoothing",
                                                                         facet, maintain_order=True):
            if g.height < cfg.min_athletes:
                continue
            a = agreement(g["value"].to_numpy(), g["v0"].to_numpy(), cfg.top_fraction)
            rows.append({"domain": domain, "drill": drill, "feature": feature, "smoothing": smoothing, "facet": facet,
                         "level": str(level), "n_athletes": g.height, "spearman": a["spearman"],
                         "median_rel_diff": a["median_rel_diff"]})
    return pl.DataFrame(rows, schema=ROBUST.columns, strict=False) if rows else ROBUST.empty()


def b_results(agree: pl.DataFrame, var: pl.DataFrame, rep: pl.DataFrame, rc: ResultCollector) -> None:
    for r in agree.iter_rows(named=True):
        rc.add(analysis_id=f"B.agreement.{r['level']}", grain=r["level"], n_athletes=r["n_athletes"],
               n_trajectories=r["n_units"] if r["level"] == "trajectory" else None, effect_estimate=r["spearman"],
               effect_size_type="Spearman rho", method=f"{r['smoothing']} vs primary SG setting",
               domain=r["domain"], stratum=stratum_label(r["stratum"], r["drill"]), feature=r["feature"],
               outcome=r["smoothing"], ci_lower=r["spearman_lower"], ci_upper=r["spearman_upper"],
               ci_method="athlete-cluster bootstrap (percentile)", result_status=r["result_status"],
               limitations="stability, not validity" + ("; ties dominate (modal share ≥ 0.5)"
                                                        if (r["modal_share"] or 0) >= 0.5 else ""))
    # athlete level only: at trajectory level the table has no athlete count (see B_variance_share)
    for r in var.filter(pl.col("share_setting").is_not_null() & (pl.col("level") == "athlete")).iter_rows(named=True):
        rc.add(analysis_id="B.variance_share.athlete", grain="athlete", n_athletes=r["n_units"], n_trajectories=None,
               effect_estimate=r["share_setting"], effect_size_type="variance share",
               method="mean within-athlete variance across settings / total", domain=r["domain"],
               stratum=stratum_label(r["stratum"], r["drill"]), feature=r["feature"], result_status="descriptive")
    for r in rep.filter(pl.col("icc1").is_not_null()).iter_rows(named=True):
        rc.add(analysis_id="B.repeatability", grain="athlete", n_athletes=r["n_athletes"], n_trajectories=r["n_attempts"],
               effect_estimate=r["icc1"], effect_size_type="ICC(1)", method="one-way random effects, first attempts",
               domain=r["domain"], stratum=stratum_label(r["stratum"], r["drill"]), feature=r["feature"],
               outcome=r["smoothing"], ci_lower=r["ci_lower"], ci_upper=r["ci_upper"],
               ci_method="athlete bootstrap (percentile)",
               result_status="exploratory")


# ============================================================================== orchestration
def nfl_sample_raw(builder: KinematicFeatureBuilder, games_per_season: int) -> pl.DataFrame:
    """Same deterministic selection as ``KinematicDiagnostics.nfl_sample`` (evenly spaced sorted game_ids)."""
    raws = []
    for season, file in builder.nfl_season_files().items():
        games = pl.scan_parquet(file).select("game_id").unique().sort("game_id").collect()["game_id"].to_list()
        step = max(1, len(games) // max(1, games_per_season))
        for g in games[::step][:games_per_season]:
            raws.append(builder.load_nfl_raw(file, season, g, g))
    return pl.concat(raws, how="vertical_relaxed")


class SmoothingStudy:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, pcfg: Phase04Config | None = None, store: StudyStore | None = None,
                 kcfg: KinematicsConfig | None = None, skip_nfl_sample: bool = False, **_: Any):
        self.cfg = cfg
        self.pcfg = pcfg or Phase04Config()
        self.s = self.pcfg.smoothing
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        self.kin = KinematicFeatureBuilder(cfg, self.kcfg)
        self.store = store or StudyStore(cfg, sampled=self.pcfg.sample_athletes is not None, phase="phase_04")
        self.skip_nfl_sample = skip_nfl_sample

    def positions(self) -> pl.DataFrame:
        refs = self.kin.store.refs
        if not refs.files("gold", "dim_athlete"):
            return pl.DataFrame(schema={"nfl_id": pl.Int64, "position_group": pl.String})
        return refs.scan("gold", "dim_athlete").select(pl.col("nfl_id").cast(pl.Int64),
                                                      pl.col("position_group").cast(pl.String)).collect()

    def sample_ids(self) -> list[int] | None:
        if self.pcfg.sample_athletes is None:
            return None
        ids = np.sort(self.kin.store.load(COMBINE_TRAJ.name).select("nfl_id").unique().collect()["nfl_id"].to_numpy())
        k = min(self.pcfg.sample_athletes, ids.size)
        return sorted(np.random.default_rng(self.pcfg.seed).choice(ids, size=k, replace=False).tolist())

    def _domain(self, raw: pl.DataFrame, domain: TrajectoryDomain, pos: pl.DataFrame
                ) -> tuple[list[pl.DataFrame], list[pl.DataFrame], list[pl.DataFrame], list[pl.DataFrame]]:
        keys = list(domain.keys)
        feats, evs, vend, ex = [], [], [], []
        for spec in self.kcfg.sensitivity_smoothing:
            k = self.kcfg.with_smoothing(spec)
            frames, _ = prepare_frames(raw, domain, k)
            kin = add_signal_columns(compute_kinematics(frames, k), keys, k, spec)
            ev = detect_events(kin, domain, k)
            feats.append(normalize_features(trajectory_features(kin, ev, domain, k), domain, pos, spec.label,
                                            self.kcfg.smoothing.label))
            evs.append(normalize_events(ev, domain, spec.label))
            vend.append(vendor_residual(kin, keys, self.s.speed_bands).with_columns(
                pl.lit("combine" if domain.name == "combine" else "nfl_sample").alias("domain"),
                pl.lit(spec.label).alias("smoothing")))
            if domain.name == "combine":
                ex.append(cast_keys(kin.filter(pl.col("drill").is_in(list(self.s.example_drills))))
                          .select(traj_id_expr(keys), pl.col("drill").cast(pl.String), pl.col("t_s").cast(pl.Float64),
                                  pl.col("speed").cast(pl.Float64), pl.col("a_tan").cast(pl.Float64))
                          .with_columns(pl.lit(spec.label).alias("smoothing")))
            log.info("[phase04:B] %s %s: %s trajectories, %s events", domain.name, spec.label,
                     f"{feats[-1]['traj_id'].n_unique():,}", f"{ev.height:,}")
            del frames, kin, ev
            gc.collect()
        return feats, evs, vend, ex

    def examples(self, long: pl.DataFrame, sig: list[pl.DataFrame]) -> pl.DataFrame:
        """Per example drill: the primary-setting attempt whose peak speed is closest to the drill median
        (ties → traj_id). Outcome-blind, deterministic."""
        if not sig:
            return EXAMPLES.empty()
        p = long.filter(pl.col("is_primary") & (pl.col("feature") == "peak_speed") & pl.col("value").is_not_null()
                        & pl.col("drill").is_in(list(self.s.example_drills)))
        pick = (p.with_columns((pl.col("value") - pl.col("value").median().over("drill")).abs().alias("_d"))
                .sort("drill", "_d", "traj_id").group_by("drill", maintain_order=True).first()["traj_id"].to_list())
        return (pl.concat(sig).filter(pl.col("traj_id").is_in(pick))
                .select(pl.lit("combine").alias("domain"), "drill", "traj_id", "smoothing", "t_s", "speed", "a_tan")
                .unique(["traj_id", "smoothing", "t_s"], keep="first").sort("traj_id", "smoothing", "t_s"))

    def compute_frames(self) -> dict[str, pl.DataFrame]:
        """Heavy stage: everything that needs raw frames (FRAME_CONTRACTS)."""
        pos, ids = self.positions(), self.sample_ids()
        raw = self.kin.load_combine_raw()
        if ids is not None:
            raw = raw.filter(pl.col("nfl_id").is_in(ids))
        feats, evs, vend, ex = self._domain(raw, COMBINE, pos)
        del raw
        if not self.skip_nfl_sample:
            nraw = nfl_sample_raw(self.kin, self.s.nfl_games_per_season)
            if ids is not None:
                nraw = nraw.filter(pl.col("nfl_id").is_in(ids))
            if nraw.height:
                f2, e2, v2, _ = self._domain(nraw, NFL, pos)
                feats, evs, vend = feats + f2, evs + e2, vend + v2
            del nraw
        long = pl.concat(feats, how="vertical")
        catalog = pl.DataFrame([{"feature": f, "unit": u, "family": fam, "definition": DEFINITIONS[f]}
                                for f, (u, fam) in FEATURES.items()], schema=CATALOG.columns)
        vendor = pl.concat(vend, how="vertical").select(list(VENDOR.columns)).sort("domain", "smoothing", "band_lower")
        return {FEATURES_DS.name: long, EVENTS_DS.name: pl.concat(evs, how="vertical"), CATALOG.name: catalog,
                VENDOR.name: vendor, EXAMPLES.name: self.examples(long, ex)}

    def analyse(self, long: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Light stage: ANALYSIS_CONTRACTS from the long feature table only (no frames)."""
        seed = self.pcfg.seed
        rc = ResultCollector(STUDY, self.s.min_athletes)
        agree = agreement_table(long, self.s, seed)
        var = variance_table(long)
        rep = repeatability_table(long, self.s, seed)
        b_results(agree, var, rep, rc)
        return {AGREE.name: agree, VARSHARE.name: var, REPEAT.name: rep, ROBUST.name: robustness_table(long, self.s),
                STUDY_RESULTS.name: rc.frame()}

    def compute(self) -> dict[str, pl.DataFrame]:
        frames = self.compute_frames()
        return {**frames, **self.analyse(frames[FEATURES_DS.name])}

    def frames_fingerprint(self) -> str:
        """Only what the frame stage depends on, so changing an analysis parameter (n_boot, min_athletes, …)
        never invalidates the heavy outputs."""
        up = upstream_state(self.kin.store, [COMBINE_TRAJ.name])
        sec = self.pcfg.section("smoothing")
        params = {**{k: v for k, v in sec.items() if k != "smoothing"},
                  "smoothing": {k: sec["smoothing"][k] for k in FRAME_PARAMS}, "kinematics": self.kcfg.params(),
                  "skip_nfl_sample": self.skip_nfl_sample}
        return self.store.fingerprint(FRAME_STAGE, params, up)

    def fingerprint(self) -> str:
        up = upstream_state(self.kin.store, [COMBINE_TRAJ.name])
        return self.store.fingerprint(STUDY, {**self.pcfg.section("smoothing"), "kinematics": self.kcfg.params(),
                                              "skip_nfl_sample": self.skip_nfl_sample}, up)

    def frames(self, force: bool = False) -> dict[str, pl.DataFrame]:
        """FRAME_CONTRACTS, reused when their own run record matches ``frames_fingerprint`` and every file is
        present; otherwise recomputed, written and recorded BEFORE any analysis runs. Outputs on disk without a
        matching record (e.g. left by a run that failed before this staging existed) are never trusted."""
        fp = self.frames_fingerprint()
        current, reason = self.store.is_current(FRAME_STAGE, fp, FRAME_CONTRACTS)
        if current and not force:
            log.info("[phase04:B] frame stage: reuse (%s)", reason)
            return {c.name: self.store.read(c) for c in FRAME_CONTRACTS}
        log.info("[phase04:B] frame stage: compute (%s%s)", "forced; " if force else "", reason)
        res = self.compute_frames()
        out = {c.name: self.store.write(res[c.name], c, STUDY) for c in FRAME_CONTRACTS}
        self.store.record_run(FRAME_STAGE, fp, FRAME_CONTRACTS, {"skip_nfl_sample": self.skip_nfl_sample})
        return out

    def run(self, force: bool = False, figures: bool = True) -> dict[str, pl.DataFrame]:
        fp = self.fingerprint()
        current, reason = self.store.is_current(STUDY, fp, CONTRACTS)
        if current and not force:
            log.info("[phase04:B] reuse (%s)", reason)
            out = self.load()
        else:
            log.info("[phase04:B] compute (%s%s)", "forced; " if force else "", reason)
            out = self.frames(force)
            res = self.analyse(out[FEATURES_DS.name])
            out = {**out, **{c.name: self.store.write(res[c.name], c, STUDY) for c in ANALYSIS_CONTRACTS}}
            self.store.record_run(STUDY, fp, CONTRACTS, {"skip_nfl_sample": self.skip_nfl_sample,
                                                         "frames_fingerprint": self.frames_fingerprint()})
        if figures:
            from .smoothing_figures import FIGURES

            for rec, fn in FIGURES:
                fn(self.store, rec, out)
        return out

    def load(self) -> dict[str, pl.DataFrame]:
        return {c.name: self.store.read(c) for c in CONTRACTS}

    def figures(self, out: dict[str, pl.DataFrame]) -> None:
        from .smoothing_figures import FIGURES

        for rec, fn in FIGURES:
            fn(self.store, rec, out)


__all__ = ["SmoothingStudy", "CONTRACTS", "FRAME_CONTRACTS", "ANALYSIS_CONTRACTS", "FRAME_STAGE", "stratum_label",
           "FEATURES", "FEATURE_NAMES", "BOUNDARY_PEAK", "BA_FEATURES",
           "trajectory_features", "agreement_table", "variance_table", "repeatability_table", "robustness_table",
           "athlete_values", "paired", "band_expr", "percentile_rank", "FigureRecord"]
