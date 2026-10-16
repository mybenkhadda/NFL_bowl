"""Stage COH — the defensive-line research base, its eligibility funnel and the predictors' redundancy.

Research base   one row per defensive-line athlete with a Combine results row. Combine columns are built for every
                such athlete, whether or not he later played: Combine eligibility never looks at an NFL outcome.
                NFL columns (observation counts, alignment group, descriptive targets) are joined afterwards.
Combine variants  athlete-level forty features under each aggregation | quality scenario | fit filter:
                    mean | first | fastest   mean of the attempts · first attempt in time · attempt with the highest
                                             p95 speed (the official forty is usually the better run)
                    S0_all … S3_no_boundary  nested Phase 04 quality scenarios
                    all | fit_ok             every usable attempt · only attempts with a usable sprint-model fit
                  ``model_tau_s`` exists only on fit_ok attempts in every variant (as persisted by Phase 03).
Funnel          athlete counts from Combine to the primary modelling population, in total and per draft cohort.
Redundancy      Spearman of each tracking candidate with the official tests among the modelling population
                (cohort-adjusted), and the share of its variance explained by all M1 predictors together.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from ..sprint_rescue.stats import partial_corr, r_squared
from .config import BODY, COHORT, COMBINE_VARIANTS, JUMPS, PRIMARY_VARIANT, TARGETS, TRACKING, TRADITIONAL, GetOffConfig

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
OFFICIAL_COLUMNS = (*BODY, *TRADITIONAL, *JUMPS)
COMPLETE_CASE = (*TRADITIONAL, *TRACKING)          # columns a model may require; body and jumps are imputed in-fold

VARIANTS = TableContract(
    "combine_feature_variants", "Athlete-level tracked forty features per aggregation | quality scenario | fit filter.",
    {"nfl_id": I, "variant": S, "n_attempts_used": I, **{f: F for f in TRACKING}}, key=("nfl_id", "variant"),
    kind="dataset", area="athlete × Combine feature variant")
BASE = TableContract(
    "athlete_research_base", "One row per defensive-line Combine athlete: pre-NFL predictors and NFL get-off summary.",
    {"nfl_id": I, "display_name": S, COHORT: I, **{c: F for c in OFFICIAL_COLUMNS}, "n_forty_attempts": I,
     **{f: F for f in TRACKING}, "n_obs": I, "n_games": I, "alignment_group": S, "share_first_alignment": F,
     **{f"nfl_{t}": F for t in TARGETS}, "has_officials": B, "has_tracking_all": B, "eligible_primary": B},
    key=("nfl_id",), kind="dataset", area="athlete (defensive line, Combine results present)",
    docs={"n_obs": "valid regular-season get-off observations (0 when none)",
          "alignment_group": "modal NFL alignment — an NFL observation, used for the outcome and for strata only",
          "nfl_adj_mean": "descriptive context-adjusted mean (context means of all athletes); models rebuild it per fold",
          "eligible_primary": "n_obs ≥ primary_min_obs AND official forty + split AND all three tracking features",
          "p95_speed": f"Combine tracking features of the primary variant {PRIMARY_VARIANT}"})
FUNNEL = TableContract(
    "cohort_funnel", "Athlete counts from the Combine to the primary modelling population.",
    {"stage": S, "definition": S, "cohort": S, "n_athletes": I}, key=("stage", "cohort"), area="funnel stage × draft cohort")
MISSING = TableContract(
    "cohort_missingness", "Missing predictors among Combine DL athletes and among target-eligible athletes.",
    {"population": S, "variable": S, "n_athletes": I, "n_missing": I, "share_missing": F},
    key=("population", "variable"), area="population × variable")
CORRELATION = TableContract(
    "predictor_correlations", "Tracking candidates against official Combine measurements (modelling population).",
    {"feature": S, "other": S, "n_athletes": I, "spearman": F, "ci_lower": F, "ci_upper": F, "r2": F, "residual_sd": F},
    key=("feature", "other"), area="tracking feature × official measurement",
    docs={"spearman": "cohort-adjusted partial Spearman, Fisher interval",
          "r2": "other = M1_all: share of the feature's variance explained by body + official times, within cohort",
          "residual_sd": "other = M1_all: SD of the feature left after M1 predictors — what M2 can add, in feature units"})


# ============================================================================== Combine features
def combine_variants(att: pl.DataFrame, gcfg: GetOffConfig, variants: Sequence[str] = COMBINE_VARIANTS) -> pl.DataFrame:
    """``att``: sprint_rescue ``attempt_censoring_diagnostics`` (one row per forty attempt)."""
    parts = []
    feats = [pl.col(f).cast(pl.Float64) for f in TRACKING]
    for variant in variants:
        agg, scenario, fit = variant.split("|")
        a = att.filter(pl.col(scenario) & (pl.col("attempt_order") <= gcfg.max_attempts))
        if fit == "fit_ok":
            a = a.filter(pl.col("fit_ok").fill_null(False))
        if not a.height:
            continue
        n = pl.len().cast(pl.Int64).alias("n_attempts_used")
        if agg == "mean":
            g = a.group_by("nfl_id").agg(n, *[f.mean() for f in feats])
        else:
            by, desc = (["nfl_id", "attempt_order"], [False, False]) if agg == "first" else (["nfl_id", "p95_speed"], [False, True])
            g = a.sort(by, descending=desc, nulls_last=True).group_by("nfl_id", maintain_order=True).agg(
                n, *[f.first() for f in feats])
        parts.append(g.with_columns(pl.lit(variant).alias("variant")))
    if not parts:
        return VARIANTS.empty()
    return pl.concat(parts, how="vertical_relaxed").select(
        [pl.col(c).cast(t) for c, t in VARIANTS.columns.items()]).sort("nfl_id", "variant")


# ============================================================================== research base
def research_base(mart: pl.DataFrame, variants: pl.DataFrame, targets: pl.DataFrame, gcfg: GetOffConfig) -> pl.DataFrame:
    """``mart``: gold.mart_athlete_research_base. ``targets``: athlete_targets (window ``all`` is used)."""
    m = mart.filter((pl.col("position_group") == gcfg.position) & pl.col("has_combine_results").fill_null(False))
    officials = [(pl.col(c).cast(pl.Float64, strict=False) if c in m.columns else pl.lit(None, dtype=pl.Float64)).alias(c)
                 for c in OFFICIAL_COLUMNS]
    m = m.select(pl.col("nfl_id").cast(pl.Int64), pl.col("display_name").cast(pl.String),
                 pl.col("cohort").cast(pl.Int64).alias(COHORT), *officials)
    trk = variants.filter(pl.col("variant") == PRIMARY_VARIANT).select(
        "nfl_id", pl.col("n_attempts_used").alias("n_forty_attempts"), *TRACKING)
    nfl = targets.filter(pl.col("window") == "all").select(
        "nfl_id", "n_obs", "n_games", "alignment_group", "share_first_alignment", *[pl.col(t).alias(f"nfl_{t}") for t in TARGETS])
    out = m.join(trk, on="nfl_id", how="left").join(nfl, on="nfl_id", how="left").with_columns(
        pl.col("n_obs").fill_null(0), pl.col("n_games").fill_null(0), pl.col("n_forty_attempts").fill_null(0))
    out = out.with_columns(
        pl.all_horizontal([pl.col(c).is_not_null() for c in TRADITIONAL]).alias("has_officials"),
        pl.all_horizontal([pl.col(c).is_not_null() for c in TRACKING]).alias("has_tracking_all"))
    out = out.with_columns((pl.col("has_officials") & pl.col("has_tracking_all")
                            & (pl.col("n_obs") >= gcfg.primary_min_obs)).alias("eligible_primary"))
    return out.select([pl.col(c).cast(t) for c, t in BASE.columns.items()]).sort("nfl_id")


# ============================================================================== funnel and missingness
def cohort_funnel(mart: pl.DataFrame, base: pl.DataFrame, gcfg: GetOffConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    cohorts = sorted(base[COHORT].drop_nulls().unique().to_list())

    def add(stage: str, definition: str, df: pl.DataFrame) -> None:
        k = len({r["stage"] for r in rows}) + 1
        name = f"{k:02d}_{stage}"
        rows.append({"stage": name, "definition": definition, "cohort": "ALL", "n_athletes": df.height})
        rows.extend({"stage": name, "definition": definition, "cohort": str(c),
                     "n_athletes": df.filter(pl.col(COHORT) == c).height} for c in cohorts)

    add("combine_dl", f"{gcfg.position} athletes with a Combine results row", base)
    if "has_nfl_tracking" in mart.columns:
        tracked = mart.filter(pl.col("has_nfl_tracking").fill_null(False))["nfl_id"].cast(pl.Int64).to_list()
        add("nfl_tracking_any", "… with ≥ 1 tracked NFL play (any season type)", base.filter(pl.col("nfl_id").is_in(tracked)))
    add("getoff_any", "… with ≥ 1 valid regular-season get-off observation", base.filter(pl.col("n_obs") >= 1))
    for thr in gcfg.min_obs_grid:
        add(f"getoff_ge_{thr}", f"… with ≥ {thr} valid observations", base.filter(pl.col("n_obs") >= thr))
    e = base.filter(pl.col("n_obs") >= gcfg.primary_min_obs)
    add("officials", f"≥ {gcfg.primary_min_obs} observations AND official forty + 10-yard split", e.filter(pl.col("has_officials")))
    e = e.filter(pl.col("has_officials"))
    for f in TRACKING:
        e = e.filter(pl.col(f).is_not_null())
        add(f"tracking_{f}", f"… AND tracked `{f}` (cumulative)", e)
    add("primary_population", "primary modelling population (identical athletes for every model)",
        base.filter(pl.col("eligible_primary")))
    for thr in gcfg.min_obs_grid:
        if thr != gcfg.primary_min_obs:
            add(f"population_at_{thr}", f"same predictors, ≥ {thr} observations (sensitivity)",
                base.filter(pl.col("has_officials") & pl.col("has_tracking_all") & (pl.col("n_obs") >= thr)))
    return pl.DataFrame(rows, schema=FUNNEL.columns)


def cohort_missingness(base: pl.DataFrame, gcfg: GetOffConfig) -> pl.DataFrame:
    rows = []
    pops = [("combine_dl", base)] + [(f"getoff_ge_{thr}", base.filter(pl.col("n_obs") >= thr)) for thr in gcfg.min_obs_grid]
    for name, df in pops:
        for c in (*OFFICIAL_COLUMNS, *TRACKING):
            k = df[c].null_count()
            rows.append({"population": name, "variable": c, "n_athletes": df.height, "n_missing": k,
                         "share_missing": k / df.height if df.height else None})
    return pl.DataFrame(rows, schema=MISSING.columns)


# ============================================================================== redundancy in the modelling population
def predictor_correlations(base: pl.DataFrame) -> pl.DataFrame:
    e = base.filter(pl.col("eligible_primary"))
    rows: list[dict[str, Any]] = []
    if e.height < 10:
        return CORRELATION.empty()

    def col(c: str) -> np.ndarray:
        return e[c].cast(pl.Float64).fill_null(float("nan")).to_numpy()

    cohort = [e[COHORT].cast(pl.String).fill_null("?").to_numpy()]
    for f in TRACKING:
        for other in (*TRADITIONAL, BODY[0], *[t for t in TRACKING if t != f]):
            r = partial_corr(col(f), col(other), categorical=cohort)
            rows.append({"feature": f, "other": other, "n_athletes": r["n"], "spearman": r["r"], "ci_lower": r["ci_lower"],
                         "ci_upper": r["ci_upper"], "r2": None, "residual_sd": None})
        X = [col(c) for c in (BODY[0], *TRADITIONAL)]
        j = r_squared(col(f), X, categorical=cohort)
        y = col(f)
        sd = float(np.nanstd(y, ddof=1) * np.sqrt(max(1.0 - j["r2"], 0.0))) if np.isfinite(j["r2"]) else None
        rows.append({"feature": f, "other": "M1_all", "n_athletes": j["n"], "spearman": None, "ci_lower": None,
                     "ci_upper": None, "r2": j["r2"], "residual_sd": sd})
    rows = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]
    return pl.DataFrame(rows, schema=CORRELATION.columns)
