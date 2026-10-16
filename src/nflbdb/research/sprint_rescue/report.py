"""Fills ``reports/SPRINT_RELIABILITY_RESCUE_TEMPLATE.md`` → ``reports/SPRINT_RELIABILITY_RESCUE_RESULTS.md``.

Every ``{{token}}`` of the template maps to a function below that reads the persisted tables of this namespace
and returns Markdown. A token whose table is missing renders an explicit "pending" line with the command to run,
so the report never shows a number that was not measured. The decision section is a rule-based PROPOSAL from the
screening table; it is labelled as such and needs review.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import polars as pl

from ...data.reporting import df_to_markdown
from ...data.utils import atomic_write_text, utc_now
from . import attempts as at
from . import censoring as ce
from . import cohorts as co
from . import redundancy as rd
from . import reliability as rl
from .config import REFERENCE_FEATURE, RescueConfig
from .pipeline import ALL_CONTRACTS, STAGES
from .sprint_model import FITS, variant_sensitivity
from .store import RescueStore

TEMPLATE = "SPRINT_RELIABILITY_RESCUE_TEMPLATE.md"
OUTPUT = "SPRINT_RELIABILITY_RESCUE_RESULTS.md"
ACCEL_FAMILIES = ("acceleration_direct", "acceleration_sustained", "sprint_profile")
CANDIDATE = ("strong_candidate", "candidate")


class Tables:
    def __init__(self, store: RescueStore):
        self.store = store
        self._cache: dict[str, pl.DataFrame | None] = {}

    def get(self, c: Any) -> pl.DataFrame | None:
        if c.name not in self._cache:
            self._cache[c.name] = self.store.read(c) if self.store.exists(c) else None
        return self._cache[c.name]


def pending(c: Any) -> str:
    stage = next(k for k, (contracts, _) in STAGES.items() if c in contracts)
    return f"_Pending: `{c.name}` not found. Run `python -m nflbdb.research.sprint_rescue --only {stage}`._\n"


def _md(df: pl.DataFrame, max_rows: int = 40) -> str:
    return df_to_markdown(df, max_rows=max_rows, float_digits=3)


def run_status(t: Tables) -> str:
    rows = [{"output": c.name, "rows": (t.get(c).height if t.get(c) is not None else None),
             "status": "present" if t.get(c) is not None else "MISSING"} for c in ALL_CONTRACTS]
    return f"Generated {utc_now()} from persisted outputs.\n\n" + _md(pl.DataFrame(rows))


def window_distribution(t: Tables) -> str:
    w = t.get(at.WINDOWS)
    if w is None:
        return pending(at.WINDOWS)
    cont = w.filter((pl.col("kind") == "continuous") & pl.col("group_type").is_in(["all", "position", "cohort"])
                    & pl.col("variable").is_in(["first_speed", "last_speed", "duration_s", "path_smooth_yds"]))
    flags = w.filter((pl.col("kind") == "flag") & pl.col("group_type").is_in(["all", "position", "cohort"])).select(
        "group_type", "group", "variable", "n", pl.col("mean").alias("share"))
    return ("**Recording-window variables (attempts)**\n\n"
            + _md(cont.select("group_type", "group", "variable", "n", "p05", "p50", "p95", "sd"), 60)
            + "\n**Flag shares (attempts)**\n\n" + _md(flags, 90))


def paired_association(t: Tables) -> str:
    a = t.get(ce.ASSOC)
    if a is None:
        return pending(ce.ASSOC)
    pa = a.filter((pl.col("analysis") == "A2_paired_association") & (pl.col("predictor") == "d_first_speed"))
    r2 = a.filter(pl.col("analysis") == "A2_window_r2")
    cols = ["outcome", "adjustment", "n", "estimate", "ci_lower", "ci_upper"]
    return ("**Δ feature against Δ first-frame speed between an athlete's two attempts (Spearman)**\n\n" + _md(pa.select(cols))
            + "\n**Share of the variance of Δ feature explained jointly by Δ start speed, Δ duration and Δ path "
              "(partial R² beyond position and cohort)**\n\n" + _md(r2.select("outcome", "n", "estimate", "ci_lower", "ci_upper", "note")))


def censoring_association(t: Tables) -> str:
    a = t.get(ce.ASSOC)
    if a is None:
        return pending(ce.ASSOC)
    cols = ["outcome", "predictor", "group", "statistic", "n", "estimate", "ci_lower", "ci_upper"]
    gain = a.filter((pl.col("analysis") == "A3_window_adjusted_icc") & (pl.col("predictor") == "start+end"))
    pat = a.filter((pl.col("analysis") == "A3_icc_by_pattern") & (pl.col("adjustment") == "position")
                   & pl.col("outcome").is_in([REFERENCE_FEATURE, "model_a0", "p95_speed"]))
    fit = a.filter(pl.col("analysis") == "A3_fit_quality")
    disc = a.filter((pl.col("analysis") == "A3_abs_discrepancy") & (pl.col("adjustment") != "none")
                    & (pl.col("predictor") == "abs_d_first_speed"))
    return ("**ICC before and after removing the within-athlete window slope (upper bound of a window correction)**\n\n"
            + _md(gain.select(cols), 60)
            + "\n**ICC by how the two recordings of an athlete differ (position-adjusted)**\n\n" + _md(pat.select(cols), 60)
            + "\n**|Δ feature| against |Δ first-frame speed|, adjusted for position, cohort and positional noise**\n\n"
            + _md(disc.select(cols)) + "\n**Model fit quality against the recording window (athlete means)**\n\n" + _md(fit.select(cols)))


def reliability_scopes(t: Tables) -> str:
    r = t.get(rl.RELIABILITY)
    if r is None:
        return pending(rl.RELIABILITY)
    s0 = r.filter((pl.col("scenario") == "S0_all") & pl.col("scope").is_in(["pooled", "position_adjusted", "position_cohort_adjusted"]))
    wide = (s0.filter(pl.col("scope") == "pooled").select("feature", "family", "n_athletes", pl.col("icc1").alias("icc_pooled"))
            .join(s0.filter(pl.col("scope") == "position_adjusted").select(
                "feature", pl.col("icc1").alias("icc_within_position"), "ci_lower", "ci_upper", "icc_gap_vs_pooled",
                "position_share_of_between", "position_dependent", "typical_error", "cv_within_pct", "spearman_first_two",
                "mean_diff_2_minus_1"), on="feature", how="left"))
    dep = wide.filter(pl.col("position_dependent").fill_null(False))["feature"].to_list()
    by_pos = r.filter((pl.col("scenario") == "S0_all") & pl.col("scope").str.starts_with("position:")).select(
        "feature", "scope", "n_athletes", "icc1", "ci_lower", "ci_upper", "reliability")
    by_coh = r.filter((pl.col("scenario") == "S0_all") & pl.col("scope").str.starts_with("cohort:")).select(
        "feature", "scope", "n_athletes", "icc1", "ci_lower", "ci_upper")
    return ("**Pooled against within-position ICC(1), all structurally valid attempts**\n\n" + _md(wide)
            + f"\nFeatures whose pooled ICC exceeds the within-position ICC by more than "
              f"{RescueConfig().position_dependence_gap:g}: " + (", ".join(f"`{f}`" for f in dep) if dep else "none") + ".\n"
            + "\n**Within each position group**\n\n" + _md(by_pos, 120) + "\n**Within each draft cohort**\n\n" + _md(by_coh, 80))


def reliability_scenarios(t: Tables) -> str:
    r = t.get(rl.RELIABILITY)
    if r is None:
        return pending(rl.RELIABILITY)
    s = r.filter(pl.col("scope") == "position_adjusted").select("feature", "scenario", "n_athletes", "icc1", "ci_lower", "ci_upper")
    return "**Position-adjusted ICC(1) by quality scenario**\n\n" + _md(s.sort("feature", "scenario"), 120)


def model_identifiability(t: Tables) -> str:
    f = t.get(FITS)
    if f is None:
        return pending(FITS)
    base = f.filter(pl.col("variant") == "base")
    summary = base.select(
        pl.len().alias("n_attempts"), pl.col("fit_ok").mean().alias("share_fit_ok"),
        pl.col("at_grid_edge").mean().alias("share_optimum_at_grid_edge"),
        pl.col("a0_extrapolated").mean().alias("share_model_start_before_recording"),
        pl.col("extrap_gap_s").median().alias("median_gap_model_start_to_first_fitted_frame_s"),
        pl.col("weakly_identified").mean().alias("share_weakly_identified"),
        pl.col("tau_rel_width").median().alias("median_tau_rel_width"), pl.col("a0_rel_width").median().alias("median_a0_rel_width"),
        pl.col("vmax_rel_width").median().alias("median_vmax_rel_width"), pl.col("tau_t0_corr").median().alias("median_tau_t0_corr"),
        pl.col("r2").median().alias("median_r2"), pl.col("rmse").median().alias("median_rmse"))
    status = f.group_by("variant", "status").len().sort("variant", "status")
    return ("**Existing fit window (`base`), all forty attempts**\n\n"
            + _md(summary.unpivot(variable_name="quantity", value_name="value"))
            + "\n**Parameter movement when the fit window changes (median |relative change| against `base`)**\n\n"
            + _md(variant_sensitivity(f)) + "\n**Fit status by variant**\n\n" + _md(status))


def model_repeatability(t: Tables) -> str:
    m = t.get(rl.MODEL_REPEAT)
    if m is None:
        return pending(rl.MODEL_REPEAT)
    cols = ["representation", "feature", "variant", "n_athletes", "icc1", "ci_lower", "ci_upper", "ref_icc_same_cohort",
            "icc_diff_vs_ref", "diff_ci_lower", "diff_ci_upper"]
    common = m.filter((pl.col("cohort") == "common") & (pl.col("scope") == "position_adjusted")).sort("icc1", descending=True, nulls_last=True)
    own = m.filter((pl.col("cohort") == "own") & (pl.col("scope") == "position_adjusted")
                   & pl.col("variant").is_in(["direct", "persisted", "base"])).sort("icc1", descending=True, nulls_last=True)
    better = common.filter(pl.col("diff_ci_lower").fill_null(-1.0) > 0)
    names = [f"`{r['feature']}` ({r['variant']})" for r in better.iter_rows(named=True)]
    return (f"**Common cohort, position-adjusted; reference = `{REFERENCE_FEATURE}` on identical attempts**\n\n" + _md(common.select(cols), 60)
            + "\nRows whose paired interval for the ICC difference lies wholly above zero: " + (", ".join(names) if names else "none") + ".\n"
            + "\n**Own largest sample (sample differs between rows — not a like-for-like comparison)**\n\n" + _md(own.select(cols), 40))


def redundancy(t: Tables) -> str:
    r = t.get(rd.REDUNDANCY)
    if r is None:
        return pending(rd.REDUNDANCY)
    m = r.filter((pl.col("aggregation") == "mean") & (pl.col("scope") == "position_adjusted"))
    single = m.filter(pl.col("official").is_in([rd.FORTY, rd.TEN])).select(
        "feature", "official", "n_athletes", "n_missing_official", "spearman", "spearman_lower", "spearman_upper", "pearson")
    cond = m.filter(pl.col("official").str.contains("|", literal=True)).select(
        "feature", "official", "n_athletes", "spearman", "spearman_lower", "spearman_upper")
    joint = m.filter(pl.col("official") == rd.JOINT).select("feature", "n_athletes", "r2")
    pooled = r.filter((pl.col("aggregation") == "mean") & (pl.col("scope") == "pooled") & (pl.col("official") == rd.FORTY)).select(
        "feature", "n_athletes", "spearman", "spearman_lower", "spearman_upper")
    fast = r.filter((pl.col("aggregation") == "fastest_attempt") & (pl.col("scope") == "position_adjusted")
                    & (pl.col("official") == rd.FORTY)).select("feature", "n_athletes", "spearman", "spearman_lower", "spearman_upper")
    return ("**Position-adjusted association with each official time (athlete means)**\n\n" + _md(single, 60)
            + "\n**After also conditioning on the other official time**\n\n" + _md(cond, 60)
            + "\n**Share of feature variance explained by both official times, within position (partial R²)**\n\n" + _md(joint)
            + "\n**Pooled over positions, official forty (for comparison only)**\n\n" + _md(pooled)
            + "\n**Sensitivity: fastest attempt instead of the mean, official forty**\n\n" + _md(fast))


def screening(t: Tables) -> str:
    s = t.get(rd.SCREEN)
    if s is None:
        return pending(rd.SCREEN)
    cols = ["feature", "family", "phase04_status", "n_athletes_repeat", "icc_adj", "icc_adj_lower", "icc_adj_upper",
            "icc_cohort_min", "icc_adj_s3", "signal_to_noise", "redundancy_abs", "redundancy_lower", "redundancy_upper",
            "r2_stopwatch_adj", "reliable_unique_share", "quadrant", "screening_status"]
    return _md(s.select(cols).sort("reliable_unique_share", descending=True, nulls_last=True), 40)


def reconciliation(t: Tables) -> str:
    r = t.get(co.RECON)
    if r is None:
        return pending(co.RECON)
    bad = r.filter(pl.col("matches_expected") == False)  # noqa: E712
    note = ("All counts quoted in earlier reports are reproduced." if not bad.height else
            "Counts that do NOT match earlier reports: " + ", ".join(f"`{x}`" for x in bad["stage"].to_list()) + ".")
    return _md(r.select("stage", "definition", "n_athletes", "expected", "matches_expected"), 60) + "\n" + note + "\n"


def _endpoint(t: Tables, endpoint: str) -> str:
    e = t.get(co.ENDPOINT)
    if e is None:
        return pending(co.ENDPOINT)
    e = e.filter(pl.col("endpoint") == endpoint)
    if not e.height:
        return "_No row for this endpoint._\n"
    out = []
    d = e.filter(pl.col("section") == "definition")
    out.append("**Definition status:** " + ("established" if d["value"][0] == 1.0 else "**NOT established**") + ". " + d["note"][0] + "\n")
    for section, title in (("missingness", "Missingness (player-plays)"), ("distribution", "Distribution of non-null values"),
                           ("athletes", "Athletes and observations per athlete"), ("intersection", "Overlap with Combine measurements (athletes)"),
                           ("context", "Play context among endpoint plays"), ("reliability", "Repeatability"),
                           ("first_n", "First-N-play eligibility")):
        g = e.filter(pl.col("section") == section).select("stratum", "metric", "value", "n", "note")
        out.append(f"**{title}**\n\n" + _md(g, 80))
    return "\n".join(out)


def decision(t: Tables) -> str:
    s = t.get(rd.SCREEN)
    if s is None:
        return pending(rd.SCREEN)
    fam = (s.filter(pl.col("screening_status").is_in(list(CANDIDATE)) & pl.col("reliable_unique_share").is_not_null())
           .group_by("family").agg(pl.col("reliable_unique_share").max().alias("best_reliable_unique_share"),
                                   pl.col("feature").sort_by("reliable_unique_share", descending=True).first().alias("best_feature"),
                                   pl.len().alias("n_candidate_features"))
           .sort("best_reliable_unique_share", descending=True))
    lines = ["_Rule-based proposal from `feature_screening_summary` (heuristic labels, proposed references). It requires review "
             "and is not a result._\n"]
    if not fam.height:
        lines.append("- **No feature family reaches the candidate label.** No tracking feature is both repeatable within position "
                     "and distinguishable from the stopwatch at the heuristic references. A null incremental result is the "
                     "supported reading; the next experiment should test the official times alone against the NFL endpoint, "
                     "with tracking speed as a pre-registered secondary comparison.")
    else:
        lines.append(_md(fam))
        top = fam.row(0, named=True)
        lines.append(f"- **Primary family (proposal):** `{top['family']}`, lead feature `{top['best_feature']}`.")
        lines.append(f"- **Backup family (proposal):** `{fam.row(1, named=True)['family']}`." if fam.height > 1 else
                     "- **Backup family:** none reaches the candidate label.")
        if not set(fam["family"].to_list()) & set(ACCEL_FAMILIES):
            lines.append("- **No acceleration family is among the candidates.** Fallback: use the repeatable speed and "
                         "sprint-profile measurements and test whether they explain the NFL endpoint beyond the official times.")
    red = s.filter(pl.col("screening_status") == "redundant_with_stopwatch")["feature"].to_list()
    bad = s.filter(pl.col("screening_status") == "not_repeatable")["feature"].to_list()
    lines.append("- Redundant with the stopwatch: " + (", ".join(f"`{f}`" for f in red) if red else "none") + ".")
    lines.append("- Not repeatable within position: " + (", ".join(f"`{f}`" for f in bad) if bad else "none") + ".")
    return "\n".join(lines) + "\n"


SECTIONS: dict[str, Callable[[Tables], str]] = {
    "run_status": run_status, "window_distribution": window_distribution, "paired_association": paired_association,
    "censoring_association": censoring_association, "reliability_scopes": reliability_scopes,
    "reliability_scenarios": reliability_scenarios, "model_identifiability": model_identifiability,
    "model_repeatability": model_repeatability, "redundancy": redundancy, "screening": screening,
    "reconciliation": reconciliation, "endpoint_dl": lambda t: _endpoint(t, "player_get_off"),
    "endpoint_wr": lambda t: _endpoint(t, "separation_at_pass_forward"), "decision": decision,
}


def fill(template: str, t: Tables) -> str:
    def sub(m: re.Match) -> str:
        fn = SECTIONS.get(m.group(1))
        return fn(t) if fn else f"_Unknown report token `{m.group(1)}`._"
    return re.sub(r"\{\{(\w+)\}\}", sub, template)


def write_report(cfg: Any, store: RescueStore) -> str:
    root = cfg.paths.reports_root
    text = fill((root / TEMPLATE).read_text(encoding="utf-8"), Tables(store))
    atomic_write_text(text, root / OUTPUT)
    return cfg.paths.rel(root / OUTPUT)
