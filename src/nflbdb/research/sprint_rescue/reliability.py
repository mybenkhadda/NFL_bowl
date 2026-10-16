"""Studies B and C — repeatability of forty-yard-dash measurements within position groups.

Study B   Every candidate feature, in every quality scenario (S0 … S3), under five scopes:
              pooled                        all athletes, no adjustment (the number reported so far)
              position:<P>                  athletes of one position group only
              position_adjusted             all athletes, position means removed (stratified ICC)
              cohort:<Y>                    one draft cohort only
              position_cohort_adjusted      position × cohort means removed
          A pooled ICC can be high only because position groups differ (linemen are slower than receivers);
          ``position_share_of_between`` = 1 − σ²_between(position-adjusted) / σ²_between(pooled) says how much of
          the between-athlete variance is position, and ``icc_gap_vs_pooled`` how much the ICC falls without it.

Study C   Central comparison: is the sprint model more repeatable than direct acceleration ON THE SAME ATTEMPTS?
          Each feature is evaluated on its own largest sample (``own``) and on a common cohort (``common``:
          attempts where every core feature is available). The difference against the reference feature is
          computed on identical athletes and attempts with shared bootstrap resamples (paired interval), so a
          gain can never come from a change of sample.

Population: athletes with ≥ ``min_attempts`` usable attempts; only the first ``max_attempts`` chronological ones.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...features.reliability import reliability_label
from ..contracts import TableContract
from ..studies.stats import spearman
from .attempts import BANDS
from .config import KEYS, REFERENCE_FEATURE, SCENARIOS, VARIANTS, FeatureSpec, RescueConfig
from .stats import paired_difference, seed_for, stratified_icc

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
ID = ["nfl_id", "position_group", "draft_year", "attempt_order"]
ADJUSTED = {"position_adjusted": ("position_group",), "position_cohort_adjusted": ("position_group", "draft_year")}
CORE = (REFERENCE_FEATURE, "peak_a_tan", "accel_sustained", *BANDS, "model_a0")     # defines the common cohort
DIRECT = (("R1", "p95_a_tan"), ("R1", "peak_a_tan"), ("R1", "peak_a_tan_interior"), ("R2", "accel_sustained"),
          ("R2", "accel_initial"), *[("R2", b) for b in BANDS])
PERSISTED = {"vmax": "model_vmax", "tau_s": "model_tau_s", "a0": "model_a0"}
STAT_COLS = {"n_athletes": I, "n_attempts": I, "mean": F, "icc1": F, "ci_lower": F, "ci_upper": F, "var_between": F,
             "var_within": F, "typical_error": F, "cv_within_pct": F, "spearman_first_two": F,
             "mean_diff_2_minus_1": F, "reliability": S}

RELIABILITY = TableContract(
    "within_position_reliability", "ICC(1) of forty features: pooled, within position, position- and cohort-adjusted.",
    {"feature": S, "representation": S, "family": S, "scenario": S, "scope": S, **STAT_COLS, "icc_gap_vs_pooled": F,
     "position_share_of_between": F, "position_dependent": B},
    key=("feature", "scenario", "scope"), area="feature × quality scenario × scope",
    docs={"scenario": "S0_all ⊃ S1_structural ⊃ S2_signal ⊃ S3_no_boundary (Phase 04 Study Q flags)",
          "ci_lower": "percentile bootstrap over athletes, resampled within strata",
          "var_between": "σ² between athletes (within strata for adjusted scopes)", "var_within": "σ² within athlete",
          "typical_error": "√σ²_within", "cv_within_pct": "100 · typical error / mean, strictly positive features only",
          "spearman_first_two": "attempt 1 vs attempt 2; stratum-centred for adjusted scopes",
          "icc_gap_vs_pooled": "pooled ICC − this ICC (adjusted scopes only)",
          "position_share_of_between": "1 − σ²_between(adjusted) / σ²_between(pooled) (adjusted scopes only)",
          "position_dependent": "icc_gap_vs_pooled > position_dependence_gap (heuristic flag)",
          "reliability": "features.reliability.reliability_label on the CI lower bound; insufficient_* otherwise"})
MODEL_REPEAT = TableContract(
    "sprint_model_repeatability", "Direct, sustained and model acceleration compared on identical attempts.",
    {"representation": S, "feature": S, "variant": S, "cohort": S, "scope": S, "n_athletes": I, "n_attempts": I,
     "icc1": F, "ci_lower": F, "ci_upper": F, "typical_error": F, "cv_within_pct": F, "spearman_first_two": F,
     "reference": S, "ref_icc_same_cohort": F, "icc_diff_vs_ref": F, "diff_ci_lower": F, "diff_ci_upper": F,
     "reliability": S}, key=("feature", "variant", "cohort", "scope"),
    area="feature × fit-window variant × cohort × scope",
    docs={"variant": "direct = not a model quantity · persisted = existing Phase 03 fit · other = refit window",
          "cohort": "own = every attempt with the feature · common = attempts with every core feature",
          "ref_icc_same_cohort": "ICC of the reference feature on the attempts where this feature AND the reference "
                                 "exist (for cohort = common these are exactly the attempts of the row)",
          "diff_ci_lower": "paired bootstrap interval of icc1 − ref_icc_same_cohort (shared athlete resamples)"})


# ============================================================================== shared helpers
def repeat_frame(att: pl.DataFrame, columns: Sequence[str], rcfg: RescueConfig, scenario: str = "S0_all"
                 ) -> pl.DataFrame:
    """Attempts with every one of ``columns`` finite, capped at the first ``max_attempts`` per athlete, from
    athletes with ≥ ``min_attempts`` such attempts. Sorted by athlete, then chronological attempt order."""
    d = att.filter(pl.col(scenario)) if scenario in att.columns else att
    d = d.select(*ID, *[pl.col(c).cast(pl.Float64) for c in dict.fromkeys(columns)])
    d = d.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in dict.fromkeys(columns)]))
    return (d.filter(pl.col("attempt_order").rank("ordinal").over("nfl_id") <= rcfg.max_attempts)
            .with_columns(pl.len().over("nfl_id").alias("_k")).filter(pl.col("_k") >= rcfg.min_attempts).drop("_k")
            .sort("nfl_id", "attempt_order"))


def strata_of(d: pl.DataFrame, by: Sequence[str]) -> np.ndarray | None:
    if not by:
        return None
    return d.select(pl.concat_str([pl.col(c).cast(pl.String).fill_null("?") for c in by], separator="|")
                    ).to_series().to_numpy()


def icc_stats(d: pl.DataFrame, column: str, by: Sequence[str], rcfg: RescueConfig, positive: bool, seed: int,
              values: np.ndarray | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(STAT_COLS row, raw ``stratified_icc`` result). ``d`` must come from :func:`repeat_frame`."""
    row: dict[str, Any] = {c: None for c in STAT_COLS}
    y = d[column].to_numpy().astype(float) if values is None else values
    ath, strata = d["nfl_id"].to_numpy(), strata_of(d, by)
    res = stratified_icc(y, ath, strata, rcfg.n_boot, seed)
    n_ath = res["n_athletes"]
    row.update(n_athletes=int(n_ath), n_attempts=int(d.height))
    if n_ath < rcfg.min_athletes or n_ath - res["n_strata"] < 2 or not np.isfinite(res["icc"]):
        row["reliability"] = "insufficient_athletes"
        return row, res
    first = np.unique(ath, return_index=True)[1]            # rows are sorted by athlete, attempt order
    v1, v2 = y[first], y[first + 1]
    c1, c2 = v1, v2
    if strata is not None:                                  # remove stratum means before ranking athletes
        lab, inv = np.unique(strata, return_inverse=True)
        centre = (np.bincount(inv, weights=y, minlength=lab.size) / np.bincount(inv, minlength=lab.size))[inv]
        c1, c2 = v1 - centre[first], v2 - centre[first + 1]
    te, mean = float(np.sqrt(res["var_within"])), float(y.mean())
    row.update(mean=mean, icc1=res["icc"], ci_lower=res["ci_lower"], ci_upper=res["ci_upper"],
               var_between=res["var_between"], var_within=res["var_within"], typical_error=te,
               cv_within_pct=100 * te / mean if positive and (y > 0).all() and mean > 0 else None,
               spearman_first_two=spearman(c1, c2) if first.size >= rcfg.min_athletes else None,
               mean_diff_2_minus_1=float((v2 - v1).mean()),
               reliability=(reliability_label(res["ci_lower"]) if n_ath >= rcfg.min_athletes_label
                            else "insufficient_for_label"))
    return row, res


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}


# ============================================================================== Study B
def scopes(d: pl.DataFrame) -> list[tuple[str, pl.DataFrame, tuple[str, ...]]]:
    out: list[tuple[str, pl.DataFrame, tuple[str, ...]]] = [("pooled", d, ())]
    for p in sorted(d["position_group"].unique().to_list()):
        out.append((f"position:{p}", d.filter(pl.col("position_group") == p), ()))
    out.append(("position_adjusted", d, ADJUSTED["position_adjusted"]))
    for y in sorted(d["draft_year"].drop_nulls().unique().to_list()):
        out.append((f"cohort:{y}", d.filter(pl.col("draft_year") == y), ()))
    out.append(("position_cohort_adjusted", d, ADJUSTED["position_cohort_adjusted"]))
    return out


def reliability_table(att: pl.DataFrame, specs: Sequence[FeatureSpec], rcfg: RescueConfig) -> pl.DataFrame:
    rows = []
    for spec in specs:
        if spec.name not in att.columns:
            continue
        for scenario in SCENARIOS:
            d = repeat_frame(att, [spec.name], rcfg, scenario)
            pooled: dict[str, Any] | None = None
            for scope, sub, by in (scopes(d) if d.height else []):
                row, _ = icc_stats(sub, spec.name, by, rcfg, spec.positive, seed_for(rcfg.seed, "B", spec.name, scenario, scope))
                row.update(feature=spec.name, representation=spec.representation, family=spec.family,
                           scenario=scenario, scope=scope, icc_gap_vs_pooled=None, position_share_of_between=None,
                           position_dependent=None)
                if scope == "pooled":
                    pooled = row
                elif scope in ADJUSTED and pooled and row["icc1"] is not None and pooled["icc1"] is not None:
                    gap = pooled["icc1"] - row["icc1"]
                    vb = pooled["var_between"]
                    row.update(icc_gap_vs_pooled=gap, position_dependent=bool(gap > rcfg.position_dependence_gap),
                               position_share_of_between=1.0 - row["var_between"] / vb if vb and vb > 0 else None)
                rows.append(_clean(row))
    return pl.DataFrame(rows, schema=RELIABILITY.columns) if rows else RELIABILITY.empty()


# ============================================================================== Study C
def model_wide(att: pl.DataFrame, fits: pl.DataFrame) -> tuple[pl.DataFrame, list[tuple[str, str, str, str]]]:
    """Attempt table with one column per (model parameter, fit-window variant); values only where that variant's
    fit is usable (``fit_ok``). Returns the frame and the entries (representation, feature, variant, column)."""
    w = att
    entries = [(rep, f, "direct", f) for rep, f in DIRECT if f in att.columns]
    entries += [("R3", p, "persisted", col) for p, col in PERSISTED.items() if col in att.columns]
    for var in VARIANTS:
        sub = fits.filter((pl.col("variant") == var) & pl.col("fit_ok")).select(
            *KEYS, *[pl.col(p).alias(f"{p}@{var}") for p in ("vmax", "tau_s", "a0", "a_at_ref")])
        w = w.join(sub, on=KEYS, how="left")
        entries += [("R3", p, var, f"{p}@{var}") for p in ("vmax", "tau_s", "a0", "a_at_ref")]
    return w, entries


def model_repeatability(att: pl.DataFrame, fits: pl.DataFrame, rcfg: RescueConfig) -> pl.DataFrame:
    w, entries = model_wide(att, fits)
    core = [c for c in CORE if c in w.columns]
    ref = REFERENCE_FEATURE
    rows = []
    for rep, feature, variant, col in entries:
        for cohort in ("own", "common"):
            need = [col, ref] if cohort == "own" else [col, *core]
            # "own" keeps every attempt with the feature; the reference is added only for the paired difference
            d_own = repeat_frame(w, [col], rcfg)
            d_pair = repeat_frame(w, need, rcfg)
            d = d_own if cohort == "own" else d_pair
            for scope, by in (("pooled", ()), *ADJUSTED.items()):
                if scope == "position_cohort_adjusted":
                    continue
                seed = seed_for(rcfg.seed, "C", col, cohort, scope)
                row, _ = icc_stats(d, col, by, rcfg, True, seed)
                out = {"representation": rep, "feature": feature, "variant": variant, "cohort": cohort, "scope": scope,
                       **{k: row[k] for k in MODEL_REPEAT.columns if k in row}, "reference": ref,
                       "ref_icc_same_cohort": None, "icc_diff_vs_ref": None, "diff_ci_lower": None,
                       "diff_ci_upper": None}
                if col != ref and d_pair.height:
                    # identical athletes and attempts for both features, identical seed → paired resamples
                    _, a = icc_stats(d_pair, col, by, rcfg, True, seed)
                    _, b = icc_stats(d_pair, ref, by, rcfg, True, seed)
                    if np.isfinite(a["icc"]) and np.isfinite(b["icc"]) and a["n_athletes"] >= rcfg.min_athletes:
                        diff, lo, hi = paired_difference(a, b)
                        out.update(ref_icc_same_cohort=b["icc"], icc_diff_vs_ref=diff, diff_ci_lower=lo,
                                   diff_ci_upper=hi)
                rows.append(_clean(out))
    return pl.DataFrame(rows, schema=MODEL_REPEAT.columns) if rows else MODEL_REPEAT.empty()
