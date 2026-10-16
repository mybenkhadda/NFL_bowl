"""Fills ``reports/COMBINE_TO_NFL_DL_GETOFF_TEMPLATE.md`` → ``reports/COMBINE_TO_NFL_DL_GETOFF_RESULTS.md``.

Every ``{{token}}`` maps to a function that reads persisted tables and returns Markdown. A missing table renders
an explicit "pending" line with the command to run, so the report never shows a number that was not measured.
Sentences produced by ``reading`` and ``next_decision`` are rule-based readings of the tables; they are labelled
as such and need review. While the endpoint definition is unresolved, ``interpretation`` prints the blocker.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import polars as pl

from ...data.reporting import df_to_markdown
from ...data.utils import atomic_write_text, utc_now
from ..sprint_rescue import redundancy as rescue_rd
from ..sprint_rescue import reliability as rescue_rl
from ..sprint_rescue.store import RescueStore
from . import cohort as co
from . import endpoint as ep
from . import evaluate as ev
from . import targets as tg
from .config import FEATURE_MEANING, TRACKING, TRACKING_MODELS, GetOffConfig, model_specs
from .pipeline import ALL_CONTRACTS, STAGES

TEMPLATE = "COMBINE_TO_NFL_DL_GETOFF_TEMPLATE.md"
OUTPUT = "COMBINE_TO_NFL_DL_GETOFF_RESULTS.md"
GCFG = GetOffConfig()
MODULE = "python -m nflbdb.research.combine_nfl_dl"


class Tables:
    def __init__(self, store: RescueStore):
        self.store = store
        self.definition = ep.load_definition(store.cfg)
        self.confirmed, self.missing = ep.definition_status(self.definition)
        self._cache: dict[str, pl.DataFrame | None] = {}

    def get(self, c: Any) -> pl.DataFrame | None:
        if c.name not in self._cache:
            self._cache[c.name] = self.store.read(c) if self.store.exists(c) else None
        return self._cache[c.name]


def pending(c: Any) -> str:
    stage = next(k for k, (contracts, _) in STAGES.items() if c in contracts)
    return f"_Pending: `{c.name}` not found. Run `{MODULE} --only {stage}`._\n"


def _md(df: pl.DataFrame, max_rows: int = 60) -> str:
    return df_to_markdown(df, max_rows=max_rows, float_digits=3)


def _section(t: Tables, section: str, max_rows: int = 80) -> str:
    a = t.get(ep.AUDIT)
    if a is None:
        return pending(ep.AUDIT)
    return _md(a.filter(pl.col("section") == section).select("stratum", "metric", "value", "n", "note"), max_rows)


# ------------------------------------------------------------------------------ status and gate
def run_status(t: Tables) -> str:
    rows = [{"output": c.name, "rows": (t.get(c).height if t.get(c) is not None else None),
             "status": "present" if t.get(c) is not None else "MISSING"} for c in ALL_CONTRACTS]
    return f"Generated {utc_now()} from persisted outputs.\n\n" + _md(pl.DataFrame(rows))


def gate(t: Tables) -> str:
    if t.confirmed:
        return ("**Endpoint definition: CONFIRMED** (source: " + str(t.definition.get("source")) + "). "
                "The football interpretation in §10 applies.\n")
    return "> **" + ep.blocker_text(t.definition) + "**\n"


def definition(t: Tables) -> str:
    d = t.definition
    rows = [{"field": k, "value": "" if d.get(k) is None else str(d.get(k))} for k in
            ("status", "source", "source_quote", "timing_origin", "timing_end", "unit", "valid_play_contexts",
             "lower_is_better", "censoring_or_cap")]
    out = "**Recorded definition (`configs/player_get_off_definition.json`)**\n\n" + _md(pl.DataFrame(rows))
    if not t.confirmed and d.get("open_questions"):
        out += "\n**Open questions**\n\n" + "\n".join(f"- {q}" for q in d["open_questions"]) + "\n"
    return out + "\n**Repository evidence**\n\n" + _section(t, "definition")


# ------------------------------------------------------------------------------ cohort and outcome
def funnel(t: Tables) -> str:
    f = t.get(co.FUNNEL)
    if f is None:
        return pending(co.FUNNEL)
    wide = f.pivot(on="cohort", index=["stage", "definition"], values="n_athletes").sort("stage")
    return _md(wide)


def missingness(t: Tables) -> str:
    m = t.get(co.MISSING)
    if m is None:
        return pending(co.MISSING)
    keep = m.filter(pl.col("population").is_in(["combine_dl", f"getoff_ge_{GCFG.primary_min_obs}"]))
    return _md(keep, 40)


def reliability(t: Tables) -> str:
    r = t.get(tg.RELIABILITY)
    if r is None:
        return pending(tg.RELIABILITY)
    cols = ["target", "window", "min_obs", "method", "n_athletes", "estimate", "ci_lower", "ci_upper"]
    games = r.filter((pl.col("window") == "all") & pl.col("method").is_in(["game_random_halves_sb", "game_odd_even", "game_icc1"]))
    first = r.filter(pl.col("window").str.starts_with("first_") & (pl.col("method") == "game_random_halves_sb"))
    play = r.filter((pl.col("min_obs") == min(GCFG.min_obs_grid)) & (pl.col("method").str.starts_with("sb_projected")
                                                                     | (pl.col("method") == "play_icc1")))
    ceiling = tg.primary_reliability(r, GCFG)
    line = (f"\nReliability of the primary target (`{GCFG.primary_target}`, ≥ {GCFG.primary_min_obs} observations, random game "
            f"halves stepped up): **{ceiling:.3f}**. No predictor can explain more than about this share of its variance.\n"
            if ceiling is not None else "")
    return ("**Across independent games, every observation, by eligibility threshold**\n\n" + _md(games.select(cols), 80) + line
            + "\n**First n observations only**\n\n" + _md(first.select(cols), 40)
            + "\n**Single observations and the Spearman–Brown projection**\n\n" + _md(play.select(cols), 40))


# ------------------------------------------------------------------------------ predictors
def combine_reliability(t: Tables) -> str:
    rescue = RescueStore(t.store.cfg)
    if not (rescue.exists(rescue_rl.RELIABILITY) and rescue.exists(rescue_rd.REDUNDANCY)):
        return "_Pending: sprint rescue tables not found. Run `python -m nflbdb.research.sprint_rescue`._\n"
    scope = f"position:{GCFG.position}"
    icc = rescue.read(rescue_rl.RELIABILITY).filter((pl.col("scenario") == "S0_all") & (pl.col("scope") == scope)
                                                    & pl.col("feature").is_in(list(TRACKING))).select(
        "feature", pl.col("n_athletes").alias("n_repeat"), "icc1", "ci_lower", "ci_upper")
    red = rescue.read(rescue_rd.REDUNDANCY).filter((pl.col("aggregation") == "mean") & (pl.col("scope") == scope)
                                                   & pl.col("feature").is_in(list(TRACKING))
                                                   & pl.col("official").is_in([rescue_rd.FORTY, rescue_rd.TEN])).select(
        "feature", "official", "n_athletes", "spearman", "spearman_lower", "spearman_upper")
    return ("**Repeatability between attempts, defensive line (sprint rescue, executed)**\n\n" + _md(icc)
            + "\n**Association with official times, defensive line (sprint rescue, executed)**\n\n" + _md(red))


def correlations(t: Tables) -> str:
    c = t.get(co.CORRELATION)
    if c is None:
        return pending(co.CORRELATION)
    pair = c.filter(pl.col("other") != "M1_all").select("feature", "other", "n_athletes", "spearman", "ci_lower", "ci_upper")
    joint = c.filter(pl.col("other") == "M1_all").select("feature", "n_athletes", "r2", "residual_sd")
    return ("**Cohort-adjusted Spearman in the modelling population**\n\n" + _md(pair)
            + "\n**Share of each candidate explained by all M1 predictors, and what is left**\n\n" + _md(joint))


# ------------------------------------------------------------------------------ models
def comparison(t: Tables) -> str:
    c = t.get(ev.COMPARISON)
    if c is None:
        return pending(ev.COMPARISON)
    out = []
    for metric in ("rmse", "mae", "r2_oos", "pearson", "spearman", "calibration_slope", "calibration_intercept"):
        g = c.filter(pl.col("metric") == metric).select("model", "label", "exploratory", "n_predictors", "n_athletes",
                                                        "estimate", "ci_lower", "ci_upper")
        out.append(f"**{metric}**\n\n" + _md(g))
    return "\n".join(out)


def paired(t: Tables) -> str:
    b = t.get(ev.BOOT)
    if b is None:
        return pending(ev.BOOT)
    return _md(b.sort("metric", "model").select("metric", "model", "reference", "exploratory", "n_athletes", "estimate",
                                                "ci_lower", "ci_upper", "share_boot_improved"), 60)


def coefficients(t: Tables) -> str:
    c = t.get(ev.COEF)
    return pending(ev.COEF) if c is None else _md(c, 80)


def sensitivity(t: Tables) -> str:
    s = t.get(ev.SENSITIVITY)
    if s is None:
        return pending(ev.SENSITIVITY)
    cols = ["analysis", "model", "n_athletes", "rmse", "r2_oos", "d_rmse_vs_m1", "d_rmse_lower", "d_rmse_upper",
            "share_boot_improved_rmse"]
    trk = s.filter(pl.col("model").is_in(list(TRACKING_MODELS)))
    base = s.filter(pl.col("model").is_in(["M0", "M1", "M1j"])).select("analysis", "model", "n_athletes", "rmse", "r2_oos",
                                                                      "d_rmse_vs_m1", "d_rmse_lower", "d_rmse_upper")
    skipped = s.filter(pl.col("model") == "-").select("analysis", "description", "n_athletes", "note")
    desc = s.select("analysis", "description").unique(maintain_order=True)
    return ("**What each analysis changes**\n\n" + _md(desc, 60)
            + "\n**Tracking-augmented models against M1 (difference in RMSE, seconds; below 0 = better)**\n\n" + _md(trk.select(cols), 120)
            + "\n**Baselines under each analysis**\n\n" + _md(base, 120)
            + ("\n**Not estimable**\n\n" + _md(skipped) if skipped.height else ""))


def matched(t: Tables) -> str:
    p = t.get(ev.MATCHED)
    if p is None:
        return pending(ev.MATCHED)
    s = ev.matched_summary(p)
    head = _md(pl.DataFrame([{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in s.items()}]))
    top = p.sort("d_feature", descending=True).head(GCFG.n_case_pairs).select(
        "pair_id", "alignment_group", "name_a", "name_b", "forty_a", "forty_b", "feature_a", "feature_b", "target_a",
        "target_b", "d_target")
    return ("**All pairs: difference in the NFL target against difference in the tracked feature**\n\n" + head
            + f"\n**The {GCFG.n_case_pairs} pairs with the largest tracked difference**\n\n" + _md(top))


# ------------------------------------------------------------------------------ readings (rule-based)
def _row(df: pl.DataFrame, **eq) -> dict[str, Any] | None:
    for k, v in eq.items():
        df = df.filter(pl.col(k) == v)
    return df.row(0, named=True) if df.height else None


def _ms(v: float | None) -> str:
    return "n/a" if v is None else f"{1000 * v:+.1f} ms"


def _f3(v: float | None, signed: bool = False) -> str:
    return "n/a" if v is None else (f"{v:+.3f}" if signed else f"{v:.3f}")


def reading(t: Tables) -> str:
    c, b = t.get(ev.COMPARISON), t.get(ev.BOOT)
    if c is None or b is None:
        return pending(ev.COMPARISON)
    specs = model_specs()
    lines = ["_Rule-based reading of the tables above. It needs review and is not a conclusion._\n"]
    m1 = _row(c, model="M1", metric="r2_oos")
    m1_rmse, ref = _row(c, model="M1", metric="rmse"), _row(c, model="MEAN", metric="rmse")
    if m1 and m1["estimate"] is not None:
        verdict = ("interval not estimable" if m1["ci_lower"] is None else "its interval lies above 0" if m1["ci_lower"] > 0
                   else "its interval reaches 0 or below, so the official tests are not shown to predict the endpoint "
                        "out of sample")
        lines.append(f"- **Official tests (M1):** out-of-fold R² = {_f3(m1['estimate'])} "
                     f"[{_f3(m1['ci_lower'])}, {_f3(m1['ci_upper'])}]; {verdict}.")
    if m1_rmse and ref and m1_rmse["estimate"] is not None and ref["estimate"] is not None:
        lines.append(f"- **Practical scale:** predicting every athlete at the training mean has an RMSE of "
                     f"{1000 * ref['estimate']:.1f} ms; M1 has {1000 * m1_rmse['estimate']:.1f} ms.")
    for m in (*TRACKING_MODELS, "M2bc"):
        d, r2 = _row(b, model=m, metric="rmse"), _row(b, model=m, metric="r2_oos")
        if not d or d["estimate"] is None:
            continue
        lo, hi = d["ci_lower"], d["ci_upper"]
        if lo is None or hi is None:
            verdict = "interval not estimable"
        elif hi < 0:
            verdict = "the whole interval is below 0: the model predicted held-out athletes better than M1"
        elif lo > 0:
            verdict = "the whole interval is above 0: the model predicted held-out athletes WORSE than M1"
        else:
            verdict = "the interval includes 0: no demonstrated gain over M1"
        tag = " (EXPLORATORY)" if specs[m].exploratory else ""
        lines.append(f"- **{m}{tag}, {specs[m].label}:** RMSE − RMSE(M1) = {_ms(d['estimate'])} [{_ms(lo)}, {_ms(hi)}]; "
                     f"ΔR² = {_f3(r2['estimate'] if r2 else None, True)}; improved in {100 * (d['share_boot_improved'] or 0):.0f} % of bootstrap "
                     f"replicates; {verdict}.")
    s = t.get(ev.SENSITIVITY)
    if s is not None:
        for m in TRACKING_MODELS:
            g = s.filter((pl.col("model") == m) & pl.col("d_rmse_vs_m1").is_not_null() & (pl.col("analysis") != "primary"))
            if g.height:
                neg = g.filter(pl.col("d_rmse_vs_m1") < 0).height
                clear = g.filter(pl.col("d_rmse_upper").fill_null(1.0) < 0).height
                lines.append(f"- **{m} across sensitivities:** RMSE lower than M1 in {neg} of {g.height} analyses; interval "
                             f"wholly below 0 in {clear}.")
    return "\n".join(lines) + "\n"


def interpretation(t: Tables) -> str:
    feats = "\n".join(f"- `{k}` — {v}." for k, v in FEATURE_MEANING.items())
    head = "**What each Combine predictor represents (stated before any result)**\n\n" + feats + "\n\n"
    if not t.confirmed:
        return (head + "> **" + ep.blocker_text(t.definition) + "**\n\n"
                "What can be said without the definition: whether a model predicts the delivered number for held-out "
                "athletes, by how many milliseconds, and how stable that is. What cannot be said: that a lower or higher "
                "value is a quicker first step, that the endpoint measures acceleration, or that a coach should act on it. "
                "The direction evidence of §3.5 is co-variation with sacks and pressures; it is not a definition.\n")
    d = t.definition
    direction = "smaller values are better" if d.get("lower_is_better") else "larger values are better"
    return (head + f"**What the NFL endpoint captures (vendor definition, source: {d.get('source')}).** Clock starts at: "
            f"{d.get('timing_origin')}. Clock stops at: {d.get('timing_end')}. Valid contexts: {d.get('valid_play_contexts')}. "
            f"Direction: {direction}. Cap or censoring: {d.get('censoring_or_cap')}.\n\n"
            "The interpretation below must be written by the reviewer from §7–§9 with these limits: the association is "
            "predictive, not causal transfer; get-off also reflects snap anticipation, alignment, stance, assignment and "
            "the opponent, not only acceleration; the sample is one position group of a few dozen athletes.\n")


def next_decision(t: Tables) -> str:
    b, c = t.get(ev.BOOT), t.get(ev.COMPARISON)
    lines = ["_Rule-based proposal. It needs review._\n"]
    if not t.confirmed:
        lines.append("1. **Resolve the endpoint definition** (`configs/player_get_off_definition.json`). Nothing below is "
                     "interpretable in football terms before that.")
    if b is None or c is None:
        lines.append(f"- Model comparison not computed yet (`{MODULE} --only MOD,SENS`).")
        return "\n".join(lines) + "\n"
    wins = [m for m in TRACKING_MODELS if (r := _row(b, model=m, metric="rmse")) and r["ci_upper"] is not None and r["ci_upper"] < 0]
    m1 = _row(c, model="M1", metric="r2_oos")
    if wins:
        lines.append(f"- Pre-specified tracking model(s) with an RMSE interval wholly below M1: {', '.join(wins)}. Before any "
                     "claim: check that the sign holds in the sensitivity table (cohorts, alignment, first-n, Combine "
                     "variants), then test the same feature on a second endpoint or position group.")
    else:
        lines.append("- No pre-specified tracking model shows a demonstrated gain over the official tests. With this sample "
                     "the supported statement is 'not shown', not 'no effect': the interval in §8 states which gains remain "
                     "compatible with the data.")
    if m1 and m1["ci_lower"] is not None and m1["ci_lower"] <= 0:
        lines.append("- The official tests themselves are not shown to predict this endpoint. A tracking gain over a baseline "
                     "without skill is weak evidence; consider whether get-off is the right first endpoint.")
    return "\n".join(lines) + "\n"


SECTIONS: dict[str, Callable[[Tables], str]] = {
    "run_status": run_status, "gate": gate, "definition": definition,
    "availability": lambda t: _section(t, "availability", 120), "validity": lambda t: _section(t, "validity"),
    "distribution": lambda t: _section(t, "distribution", 200), "boundary": lambda t: _section(t, "boundary"),
    "alignment": lambda t: _section(t, "alignment"), "direction": lambda t: _section(t, "direction_evidence"),
    "funnel": funnel, "missingness": missingness, "reliability": reliability, "combine_reliability": combine_reliability,
    "correlations": correlations, "comparison": comparison, "paired": paired, "coefficients": coefficients,
    "sensitivity": sensitivity, "matched": matched, "reading": reading, "interpretation": interpretation,
    "next_decision": next_decision,
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
