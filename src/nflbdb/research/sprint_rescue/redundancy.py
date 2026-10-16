"""Study D — does tracking carry athlete information the stopwatch does not?

Unit         the athlete. A feature value is the MEAN of the athlete's attempts (``aggregation = mean``); the
             attempt with the highest p95 speed is a sensitivity (``fastest_attempt``), because the official forty
             is usually the athlete's best run.
Association  Spearman and Pearson of each feature with the official forty and the official 10-yard split:
                 pooled, within each position group, and adjusted for position, cohort, or both (partial
                 correlation: stratum means removed from both sides);
                 ``a|b``  association with official a after also conditioning on official b;
                 ``a+b``  share of the feature's variance explained by both officials jointly (partial R²).
             Intervals are Fisher z with n − 3 − q degrees of freedom. Nothing is assumed about their size.
Screening    one row per feature combining Study B (within-position repeatability) and this study. With
             ``reliability_of_mean`` the reliability of the athlete mean (Spearman–Brown) and ``r2_stopwatch_adj``
             the share explained by the two official times within position,
                 reliable_unique_share = max(reliability_of_mean − r2_stopwatch_adj, 0)
             is a LOWER bound of the reproducible variance the stopwatch does not explain (it assumes the
             stopwatch explains reproducible variance only).
             ``quadrant`` and ``screening_status`` use two heuristic reference lines (ICC and |ρ|); they are
             screening labels with the continuous evidence exported beside them, not acceptance thresholds.
             High repeatability with low redundancy is interesting; it does not show relevance to NFL play.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from .config import FEATURE_NAMES, FeatureSpec, RescueConfig
from .stats import partial_corr, r_squared, spearman_brown

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
AGGREGATIONS = ("mean", "fastest_attempt")
FORTY, TEN = "cmb_forty", "cmb_ten_yd_split"
OFFICIALS = ((FORTY, FORTY, None), (TEN, TEN, None), (f"{TEN}|{FORTY}", TEN, FORTY), (f"{FORTY}|{TEN}", FORTY, TEN))
JOINT = f"{FORTY}+{TEN}"
SCOPE_COVARIATES = {"pooled": (), "position_adjusted": ("position_group",), "cohort_adjusted": ("draft_year",),
                    "position_cohort_adjusted": ("position_group", "draft_year")}

ATHLETE = TableContract(
    "athlete_sprint_features", "Athlete-level forty features (mean of attempts and fastest attempt) with official times.",
    {"nfl_id": I, "aggregation": S, "position_group": S, "draft_year": I, "n_attempts": I, FORTY: F, TEN: F,
     **{f: F for f in FEATURE_NAMES}}, key=("nfl_id", "aggregation"), kind="dataset", area="athlete × aggregation",
    docs={"aggregation": "mean = mean over the athlete's first max_attempts attempts · fastest_attempt = values of "
                         "the attempt with the highest p95 speed"})
REDUNDANCY = TableContract(
    "combine_timing_redundancy", "Association of forty tracking features with official Combine times.",
    {"feature": S, "representation": S, "aggregation": S, "official": S, "scope": S, "n_athletes": I,
     "n_missing_official": I, "n_missing_feature": I, "spearman": F, "spearman_lower": F, "spearman_upper": F,
     "pearson": F, "pearson_lower": F, "pearson_upper": F, "r2": F},
    key=("feature", "aggregation", "official", "scope"), area="feature × official time × scope",
    docs={"official": "a|b = official a after conditioning on official b · a+b = both jointly (only r2 is filled)",
          "n_missing_official": "athletes of the scope with the feature but without the official result(s)",
          "spearman_lower": "Fisher z, n − 3 − q", "r2": "squared Pearson; for a+b the partial R² of both officials"})
SCREEN = TableContract(
    "feature_screening_summary", "Per-feature evidence: within-position repeatability against stopwatch redundancy.",
    {"feature": S, "representation": S, "family": S, "unit": S, "meaning": S, "phase04_status": S,
     "n_athletes_repeat": I, "icc_pooled": F, "icc_adj": F, "icc_adj_lower": F, "icc_adj_upper": F,
     "position_share_of_between": F, "position_dependent": B, "icc_cohort_min": F, "icc_cohort_max": F,
     "icc_adj_s3": F, "typical_error": F, "sd_between_adj": F, "signal_to_noise": F, "n_athletes_official": I,
     "rho_forty_adj": F, "rho_forty_lower": F, "rho_forty_upper": F, "rho_ten_adj": F, "rho_ten_lower": F,
     "rho_ten_upper": F, "redundancy_abs": F, "redundancy_lower": F, "redundancy_upper": F, "r2_stopwatch_adj": F,
     "reliability_of_mean": F, "reliable_unique_share": F, "quadrant": S, "screening_status": S},
    key=("feature",), area="feature",
    docs={"icc_adj": "position-adjusted ICC(1), scenario S0_all", "icc_adj_s3": "same, scenario S3_no_boundary",
          "icc_cohort_min": "smallest single-cohort ICC (cohort robustness)",
          "signal_to_noise": "sd_between_adj / typical_error", "redundancy_abs": "max(|rho_forty_adj|, |rho_ten_adj|)",
          "rho_forty_adj": "position-adjusted Spearman with the official forty (athlete means)",
          "reliable_unique_share": "max(reliability_of_mean − r2_stopwatch_adj, 0): lower bound, see module docstring",
          "quadrant": "point estimates against the two heuristic reference lines",
          "screening_status": "interval-aware heuristic label; requires review",
          "phase04_status": "Phase 04 V registry status of the feature for the forty, when it is registered"})


# ============================================================================== athlete table
def athlete_table(att: pl.DataFrame, rcfg: RescueConfig) -> pl.DataFrame:
    feats = [f for f in FEATURE_NAMES if f in att.columns]
    a = att.filter(pl.col("attempt_order") <= rcfg.max_attempts)
    head = [pl.col("position_group").first(), pl.col("draft_year").first(), pl.len().cast(pl.Int64).alias("n_attempts"),
            pl.col(FORTY).first(), pl.col(TEN).first()]
    mean = a.group_by("nfl_id").agg(*head, *[pl.col(f).cast(pl.Float64).mean().alias(f) for f in feats])
    fast = (a.sort(["nfl_id", "p95_speed"], descending=[False, True], nulls_last=True)
            .group_by("nfl_id", maintain_order=True).agg(*head, *[pl.col(f).cast(pl.Float64).first().alias(f)
                                                                  for f in feats]))
    out = pl.concat([mean.with_columns(pl.lit("mean").alias("aggregation")),
                     fast.with_columns(pl.lit("fastest_attempt").alias("aggregation"))], how="diagonal_relaxed")
    missing = [pl.lit(None, dtype=pl.Float64).alias(f) for f in FEATURE_NAMES if f not in feats]
    return (out.with_columns(missing) if missing else out).select(
        [pl.col(c).cast(t) for c, t in ATHLETE.columns.items()]).sort("nfl_id", "aggregation")


# ============================================================================== redundancy
def _f(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].cast(pl.Float64).fill_null(float("nan")).to_numpy()


def _cats(df: pl.DataFrame, cols: Sequence[str]) -> list[np.ndarray]:
    return [df[c].cast(pl.String).fill_null("?").to_numpy() for c in cols]


def redundancy_rows(a: pl.DataFrame, feature: str, representation: str, aggregation: str, scope: str,
                    covariates: Sequence[str]) -> list[dict[str, Any]]:
    y, cats = _f(a, feature), _cats(a, covariates)
    have = np.isfinite(y)
    base = {"feature": feature, "representation": representation, "aggregation": aggregation, "scope": scope}
    rows = []
    for name, target, given in OFFICIALS:
        x = _f(a, target)
        off = np.isfinite(x) & (np.isfinite(_f(a, given)) if given else True)
        cont = [_f(a, given)] if given else []
        sp = partial_corr(y, x, cats, cont, "spearman")
        pe = partial_corr(y, x, cats, cont, "pearson")
        rows.append({**base, "official": name, "n_athletes": sp["n"], "n_missing_official": int((have & ~off).sum()),
                     "n_missing_feature": int((~have & off).sum()), "spearman": sp["r"], "spearman_lower": sp["ci_lower"],
                     "spearman_upper": sp["ci_upper"], "pearson": pe["r"], "pearson_lower": pe["ci_lower"],
                     "pearson_upper": pe["ci_upper"], "r2": pe["r"] ** 2 if np.isfinite(pe["r"]) else None})
    both = np.isfinite(_f(a, FORTY)) & np.isfinite(_f(a, TEN))
    j = r_squared(y, [_f(a, FORTY), _f(a, TEN)], categorical=cats)
    rows.append({**base, "official": JOINT, "n_athletes": j["n"], "n_missing_official": int((have & ~both).sum()),
                 "n_missing_feature": int((~have & both).sum()), "spearman": None, "spearman_lower": None,
                 "spearman_upper": None, "pearson": None, "pearson_lower": None, "pearson_upper": None, "r2": j["r2"]})
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]


def redundancy_table(athletes: pl.DataFrame, specs: Sequence[FeatureSpec], rcfg: RescueConfig) -> pl.DataFrame:
    rows = []
    for agg in AGGREGATIONS:
        a = athletes.filter(pl.col("aggregation") == agg)
        if not a.height:
            continue
        positions = sorted(a["position_group"].unique().to_list())
        for spec in specs:
            for scope, cov in SCOPE_COVARIATES.items():
                rows += redundancy_rows(a, spec.name, spec.representation, agg, scope, cov)
            for p in positions:
                sub = a.filter(pl.col("position_group") == p)
                if sub.filter(pl.col(spec.name).is_not_null()).height >= rcfg.min_athletes_assoc:
                    rows += redundancy_rows(sub, spec.name, spec.representation, agg, f"position:{p}", ())
    return pl.DataFrame(rows, schema=REDUNDANCY.columns) if rows else REDUNDANCY.empty()


# ============================================================================== screening
def _abs_interval(r: float | None, lo: float | None, hi: float | None) -> tuple[float | None, float | None]:
    if r is None or lo is None or hi is None:
        return None, None
    return (0.0 if lo < 0 < hi else min(abs(lo), abs(hi))), max(abs(lo), abs(hi))


def screening_label(icc_lo: float | None, icc_hi: float | None, red_lo: float | None, red_hi: float | None,
                    rcfg: RescueConfig) -> str:
    """Interval-aware heuristic label (never a decision on its own):
    not_repeatable             the whole ICC interval lies below ``icc_floor``
    redundant_with_stopwatch   the whole |ρ| interval lies above ``redundancy_reference``
    strong_candidate           ICC interval above ``icc_reference`` and |ρ| interval below the reference
    candidate                  ICC interval above ``icc_floor`` and |ρ| interval below the reference
    inconclusive               anything else (intervals straddle a reference line)"""
    if icc_lo is None or icc_hi is None:
        return "not_estimable"
    if icc_hi < rcfg.icc_floor:
        return "not_repeatable"
    if red_lo is None or red_hi is None:
        return "redundancy_not_estimable"
    if red_lo >= rcfg.redundancy_reference:
        return "redundant_with_stopwatch"
    if red_hi < rcfg.redundancy_reference and icc_lo >= rcfg.icc_reference:
        return "strong_candidate"
    if red_hi < rcfg.redundancy_reference and icc_lo >= rcfg.icc_floor:
        return "candidate"
    return "inconclusive"


def screening_summary(specs: Sequence[FeatureSpec], reliability: pl.DataFrame, redundancy: pl.DataFrame,
                      registry: dict[str, str], rcfg: RescueConfig) -> pl.DataFrame:
    """``registry``: feature → Phase 04 V status for the forty (empty when the registry was not found)."""
    def rel(feature: str, scenario: str, scope: str) -> dict[str, Any]:
        r = reliability.filter((pl.col("feature") == feature) & (pl.col("scenario") == scenario)
                               & (pl.col("scope") == scope))
        return r.row(0, named=True) if r.height else {}

    def red(feature: str, official: str) -> dict[str, Any]:
        r = redundancy.filter((pl.col("feature") == feature) & (pl.col("aggregation") == "mean")
                              & (pl.col("official") == official) & (pl.col("scope") == "position_adjusted"))
        return r.row(0, named=True) if r.height else {}

    rows = []
    for spec in specs:
        pooled, adj = rel(spec.name, "S0_all", "pooled"), rel(spec.name, "S0_all", "position_adjusted")
        s3 = rel(spec.name, "S3_no_boundary", "position_adjusted")
        coh = reliability.filter((pl.col("feature") == spec.name) & (pl.col("scenario") == "S0_all")
                                 & pl.col("scope").str.starts_with("cohort:") & pl.col("icc1").is_not_null())
        forty, ten, joint = red(spec.name, FORTY), red(spec.name, TEN), red(spec.name, JOINT)
        icc, te, vb = adj.get("icc1"), adj.get("typical_error"), adj.get("var_between")
        cand = [(abs(r["spearman"]), r) for r in (forty, ten) if r.get("spearman") is not None]
        top = max(cand, key=lambda c: c[0])[1] if cand else {}
        red_lo, red_hi = _abs_interval(top.get("spearman"), top.get("spearman_lower"), top.get("spearman_upper"))
        red_abs = abs(top["spearman"]) if top else None
        k = adj["n_attempts"] / adj["n_athletes"] if adj.get("n_athletes") else None
        rel_mean = spearman_brown(icc, k) if icc is not None and k else None
        rel_mean = rel_mean if rel_mean is not None and np.isfinite(rel_mean) else None
        r2 = joint.get("r2")
        quadrant = None
        if icc is not None and red_abs is not None:
            quadrant = (("reliable" if icc >= rcfg.icc_reference else "unreliable") + "_"
                        + ("redundant" if red_abs >= rcfg.redundancy_reference else "nonredundant"))
        sd_b = float(np.sqrt(vb)) if vb is not None else None
        rows.append({
            "feature": spec.name, "representation": spec.representation, "family": spec.family, "unit": spec.unit,
            "meaning": spec.meaning, "phase04_status": registry.get(spec.name, "not_in_registry"),
            "n_athletes_repeat": adj.get("n_athletes"), "icc_pooled": pooled.get("icc1"), "icc_adj": icc,
            "icc_adj_lower": adj.get("ci_lower"), "icc_adj_upper": adj.get("ci_upper"),
            "position_share_of_between": adj.get("position_share_of_between"),
            "position_dependent": adj.get("position_dependent"),
            "icc_cohort_min": coh["icc1"].min() if coh.height else None,
            "icc_cohort_max": coh["icc1"].max() if coh.height else None, "icc_adj_s3": s3.get("icc1"),
            "typical_error": te, "sd_between_adj": sd_b,
            "signal_to_noise": sd_b / te if sd_b is not None and te else None,
            "n_athletes_official": forty.get("n_athletes"), "rho_forty_adj": forty.get("spearman"),
            "rho_forty_lower": forty.get("spearman_lower"), "rho_forty_upper": forty.get("spearman_upper"),
            "rho_ten_adj": ten.get("spearman"), "rho_ten_lower": ten.get("spearman_lower"),
            "rho_ten_upper": ten.get("spearman_upper"), "redundancy_abs": red_abs, "redundancy_lower": red_lo,
            "redundancy_upper": red_hi, "r2_stopwatch_adj": r2, "reliability_of_mean": rel_mean,
            "reliable_unique_share": max(rel_mean - r2, 0.0) if rel_mean is not None and r2 is not None else None,
            "quadrant": quadrant,
            "screening_status": screening_label(adj.get("ci_lower"), adj.get("ci_upper"), red_lo, red_hi, rcfg)})
    return pl.DataFrame(rows, schema=SCREEN.columns) if rows else SCREEN.empty()
