"""Reports of the unified notebook: one Markdown (+ HTML) report per stage, the decision matrix, the final research
report and the artifact inventory. Every number comes from an executed, persisted table or from the stage's
``rp.conclusion_markdown`` facts; nothing is typed by hand. A stage that did not run says so and why.
"""

from __future__ import annotations

from typing import Any

from ..utils.reproducibility import code_version, utc_now
from .data_access import REQUIREMENTS

VALIDATION_PROTOCOLS = (
    "Athlete is the independent unit: athlete-grouped outer folds (12.0 manifest = Milestone 11 folds), paired athlete bootstrap intervals.",
    "Every cross-athlete quantity (standardisation, FPCA, factor models, RC parameters, OT references, encoder PCA) is fitted on training folds only.",
    "Identical athletes for every model of a ladder (common cohort per target); Holm over declared primaries, BH for exploratory families.",
    "12C target frozen by a declared reliability rule on NFL data only, before any Combine value is joined (immutable freeze).",
    "Transductive (label-free, all sequences) and inductive (held-out athletes removed from both domains) encoder results are separate tables.",
    "Negative controls: within-position permutations / shuffled identities, shuffled Combine curves, random scores / encoders, coarsened bins, uniform mass, shuffled physics.",
    "Quick-level and debug-preset numbers are wiring checks (flagged), never results.",
)

# declared BEFORE results: what each track could inform, per role (used only when its primary evidence is supported)
DECISIONS: dict[str, dict[str, str]] = {
    "12C": {"scout": "compare a prospect's Combine speed profile with stable NFL acceleration quantiles (not single-game maxima)",
            "general_manager": "treat fitted sprint Vmax as information beyond the stopwatch for NFL upper-tail acceleration",
            "coach": "describe a player's game acceleration capacity with game-split-reliable quantiles over the shared speed range",
            "player_development": "monitor the frozen upper-quantile trait across seasons instead of noisy maxima"},
    "12B": {"scout": "read corrected (latent) Combine–NFL relations as the ceiling of what better measurement could reveal",
            "general_manager": "decide whether more repeat attempts per drill are worth collecting",
            "coach": "—", "player_development": "use regression-calibrated athlete values when only 1–2 attempts exist"},
    "12A": {"scout": "look at where along the speed axis a prospect's acceleration curve departs from position peers",
            "general_manager": "consider the whole forty acceleration profile, not only splits, as transfer information",
            "coach": "target the speed band where Combine and game acceleration correspond", "player_development": "track acceleration-curve shape (FPC scores) over time"},
    "12D": {"scout": "use multi-drill factor scores as more reproducible physical profiles than single drill numbers",
            "general_manager": "combine drills into one profile only where the factor is more reliable than its best drill",
            "coach": "—", "player_development": "use the factor profile descriptively; posterior SD shows how certain each athlete's score is"},
    "12E": {"scout": "find NFL players whose game movement distribution resembles a prospect's Combine distribution (descriptive comparables)",
            "general_manager": "use distribution matching for comparables, not for prediction", "coach": "—",
            "player_development": "compare a player's accelerating-state distribution with position references"},
    "12F": {"scout": "use the shared encoder for cross-domain comparables only if inductive retrieval beats engineered features",
            "general_manager": "fund representation learning only if it adds held-out NFL prediction beyond M1",
            "coach": "—", "player_development": "—"},
}


def _md_table(df: Any, cols: Any = None, max_rows: int = 40) -> str:
    try:
        from nflbdb.research.advanced_movement.m12 import reporting as rp

        return rp.md_table(df, list(cols) if cols else None, max_rows=max_rows)
    except Exception:                                               # noqa: BLE001
        return "_table unavailable_"


def _track_spec(track: str) -> Any:
    try:
        from nflbdb.research.advanced_movement.m12 import config as m12c

        return m12c.track(track)
    except Exception:                                               # noqa: BLE001
        return None


def _demote(md: str) -> str:
    return "\n".join(("#" + line) if line.startswith("#") else line for line in md.splitlines())


def stage_report(stage: Any, outcome: str) -> str:
    """Research question · data used · methodology · results · interpretation · limitations · artifact locations."""
    run, track = stage.run, stage.track
    t = _track_spec(track)
    req = REQUIREMENTS.get(track)
    L = [f"# {track} — {stage.title}", "",
         f"_Unified run `{run.root.name}` · outcome **{outcome}** · Milestone 12 status `{stage.m12_status or '—'}` · generated {utc_now()}_", ""]
    if outcome not in ("executed", "reused"):
        L += [f"> **Not executed in this run:** {stage.reason or outcome}.", "> Its methodology is kept below; the stage runs once its inputs exist and it is enabled.", ""]
    L += ["## Research question", t.question if t else "—", ""]
    if t:
        L += ["## Declared hypothesis and primary comparison", t.primary_hypothesis, "", f"- Primary outcome: {t.primary_outcome}", f"- Primary comparison: {t.primary_comparison}",
              f"- Evaluation target: `{t.evaluation_target}` · baselines: {t.baselines} · negative controls: {t.negative_controls}", f"- Success rule: {t.success} · failure rule: {t.failure}", ""]
    L += ["## Data used"]
    if req:
        L += [f"- Input mode: **{req.mode}**"] + [f"- required: `{a}/{n}.{k}`" for (a, n, k) in req.required] + [f"- optional: `{a}/{n}.{k}`" for (a, n, k) in req.optional]
        if req.note:
            L.append(f"- note: {req.note}")
    if getattr(stage, "missing_optional", None):
        L.append(f"- optional inputs missing in this run: {', '.join(stage.missing_optional)}")
    if stage.facts and stage.facts.get("sample"):
        L.append(f"- sample: {stage.facts['sample']}")
    L += ["", "## Methodology", stage.methodology or "—", ""]
    L += ["## Results and interpretation (generated from the persisted tables)", ""]
    L += [_demote(stage.conclusion_md), ""] if stage.conclusion_md else ["_No results: the stage did not produce a conclusion in this run._", ""]
    if stage.report_tables:
        L += ["## Key result tables", ""]
        for title, df, cols in stage.report_tables:
            L += [f"### {title}", "", _md_table(df, cols), ""]
    if stage.facts and stage.facts.get("limitations") and not stage.conclusion_md:
        L += ["## Limitations"] + [f"- {x}" for x in stage.facts["limitations"]] + [""]
    arts = run.artifacts(stage=track)
    L += ["## Artifact locations (relative to RUN_DIR)", "", _md_table(arts.select("kind", "name", "paths") if arts.height else arts, max_rows=120), ""]
    return "\n".join(L)


def eda_report(run: Any, tables: list[tuple[str, Any]], notes: list[str]) -> str:
    L = ["# Unified exploratory data analysis", "", f"_Unified run `{run.root.name}` · generated {utc_now()}_", "",
         "## Research question", "Who and what is observed — athletes, domains, positions, movement signals, sequence quality, folds, missingness and leakage risks — before any method runs?", "",
         "## Data used", *[f"- {n}" for n in notes], "",
         "## Methodology", "Exact statistics: channel histograms and per-sequence summaries are accumulated chunk by chunk over every valid frame of the memory-mapped sequence store "
         "(no sampling). Athlete-level counts (the independent unit). Method-specific diagnostics stay in their stage (12C exposure support, 12D view availability, 12E density maps).", "",
         "## Results", ""]
    for title, df in tables:
        L += [f"### {title}", "", _md_table(df), ""]
    L += ["## Interpretation", "Domain differences in the signal distributions are expected (a standing-start drill vs unrestricted game movement); "
          "they motivate comparable-state definitions (12E) and domain diagnostics (12F), not a correction of the data.", "",
          "## Limitations", "- Coverage reflects who was invited / tracked and who played, not prospects in general.", "- Histograms are frame-weighted; athlete-level analyses re-weight per athlete.", "",
          "## Artifact locations (relative to RUN_DIR)", "", _md_table(run.artifacts(stage="eda").select("kind", "name", "paths"), max_rows=80), ""]
    return "\n".join(L)


def evaluation_report(run: Any, tabs: dict[str, Any]) -> str:
    """Part 11 report: one section per evidence class / validation family, each table in its own estimand."""
    titles = (("model_comparison", "Declared primary comparisons (all classes)"), ("unified_inductive", "Inductive (held-out, identical athletes per comparison)"),
              ("unified_transductive", "Transductive (label-free diagnostics)"), ("unified_measurement", "Measurement reliability"),
              ("unified_in_sample_estimand", "In-sample estimands (12B coefficients; no held-out claim)"), ("unified_outcome_linked", "Outcome-linked rows (NFL targets)"),
              ("cross_domain_retrieval", "Cross-domain retrieval (12E distributions · 12F embeddings)"), ("athlete_identity_controls", "Athlete identity controls / nulls"),
              ("domain_shift_diagnostics", "Domain-shift diagnostics"), ("physical_property_probes", "Physical property prediction (12F probes)"),
              ("latent_trait_consistency", "Latent-trait and measurement consistency"), ("distributional_calibration", "Distributional / interval calibration"),
              ("model_ablations", "Model ablations (12F)"), ("cross_validation_folds", "Cross-validation folds (athletes per fold × position)"),
              ("unified_software_checks", "Software checks (diagnostic, synthetic data only)"))
    L = ["# Unified scientific evaluation (Part 11)", "", f"_Unified run `{run.root.name}` · generated {utc_now()}_", "",
         "Rows are classified, never pooled: each keeps its own estimand, cohort and units. Evidence classes: diagnostic · measurement · transductive · "
         "in_sample_estimand · inductive; `outcome_linked` marks rows involving an NFL target. Intervals: paired / athlete bootstrap 95 % percentile intervals "
         "(each track's own resampling unit; games for 12C reliability splits).", ""]
    for k, title in titles:
        df = tabs.get(k)
        L += [f"## {title}", "", _md_table(df, max_rows=60) if df is not None and getattr(df, "height", 0) else "_no rows in this run_", ""]
    L += ["## Artifact locations (relative to RUN_DIR)", "", _md_table(run.artifacts(stage="evaluation").select("kind", "name", "paths"), max_rows=80), ""]
    return "\n".join(L)


def decision_matrix(run: Any, comps: Any) -> Any:
    """Per track × role: the declared candidate use and whether this run's PRIMARY evidence supports it (text, never a score)."""
    import polars as pl

    rows = []
    for tr, uses in DECISIONS.items():
        sub = comps.filter((pl.col("track") == tr) & pl.col("primary")) if comps is not None and comps.height else None
        st = run.stage_status(tr)
        if sub is None or not sub.height:
            ev, verdict = "no primary evidence in this run", "not_tested"
        else:
            v = sub["verdict"].to_list()
            verdict = "supported" if all(x == "supported" for x in v) else ("reversed" if any(x == "reversed" for x in v) else "inconclusive")
            ev = f"{v.count('supported')} supported / {v.count('inconclusive')} inconclusive / {v.count('reversed')} reversed of {len(v)} primary comparison(s)"
        for role, use in uses.items():
            if use == "—":
                continue
            stance = {"supported": "candidate use — exploratory, one cohort; confirm on an independent draft class before relying on it",
                      "inconclusive": "NOT supported at this sample size — descriptive context only",
                      "reversed": "do not use — the evidence points the other way",
                      "not_tested": "no evidence from this run"}[verdict]
            rows.append({"track": tr, "role": role, "candidate_use": use, "primary_evidence": ev, "stage_status": st, "decision_stance": stance})
    return pl.DataFrame(rows)


def final_report(run: Any, ucfg: Any, sources: Any, eval_tabs: dict[str, Any], decision: Any, stages: list[Any]) -> str:
    import polars as pl

    st = run.stage_table()
    inv = run.inventory()
    by_area = inv.group_by("area").agg(pl.len().alias("files"), (pl.col("bytes").sum() / 2 ** 20).round(1).alias("MB")).sort("area") if inv.height else inv
    mc = eval_tabs.get("model_comparison")
    sup = mc.filter(pl.col("verdict") == "supported") if mc is not None and mc.height else None
    nsup = mc.filter(pl.col("verdict") != "supported") if mc is not None and mc.height else None
    tour = run.load_table("m12g_tournament") if run.has_table("m12g_tournament") else None
    rec = run.load_table("m12g_recommendations") if run.has_table("m12g_recommendations") else None
    lim, nxt = [], []
    for s in stages:
        f = getattr(s, "facts", None) or {}
        lim += [f"{s.track}: {x}" for x in f.get("limitations", [])]
        nxt += [f"{s.track}: {x}" for x in f.get("next", [])]
    L = ["# Milestone 12 — unified research report", "",
         f"_Run `{ucfg.experiment_name}` · run mode `{ucfg.run_mode}` · warehouse run mode `{ucfg.warehouse_run_mode}` · seed {ucfg.seed} · data mode `{sources.mode}` · generated {utc_now()}_", "",
         f"Code: `{code_version()}`. Configuration: `config/unified_config.json`. Every number below was produced by this notebook run from persisted tables.", "",
         "## 1. Stages: executed, reused, skipped, failed", "", _md_table(st, ["stage", "status", "finished_at", "note", "error"], max_rows=40), "",
         "## 2. Data sources", "", _md_table(sources.table()), "",
         "## 3. Model configurations", ""]
    for s in stages:
        if getattr(s, "exp", None) is not None:
            e = s.exp
            L += [f"- 12F nflbowl experiment `{e.id}` · preset `{ucfg.encoder_preset}` · science hash `{e.cfg.science_hash()}` · precision `{e.cfg.runtime.precision}` · "
                  f"device `{e.cfg.runtime.device}` · batch {e.cfg.science.batch} × accum {e.cfg.science.accum_steps} · epochs ≤ {e.cfg.science.epochs} · seed {e.cfg.science.seed}"]
        elif getattr(s, "ctx", None) is not None and s.track == "12.0":
            L += [f"- M12Config hash `{s.ctx.C.hash()}` (warehouse run mode `{s.ctx.controls.run_mode}`, seed {s.ctx.controls.seed})"]
    L += ["", "## 4. Validation protocols", "", *[f"- {p}" for p in VALIDATION_PROTOCOLS], "",
          "## 5. Metrics — declared primary comparisons", "", _md_table(mc, max_rows=60) if mc is not None else "_none_", ""]
    rr = eval_tabs.get("cross_domain_retrieval")
    if rr is not None and rr.height:
        L += ["### Cross-domain retrieval (same estimand family, different query units)", "", _md_table(rr), ""]
    if tour is not None and tour.height:
        L += ["### Common-cohort tournament (12G; the only identical-athlete method comparison)", "",
              _md_table(tour, ["target", "model", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "p_boot_le0_holm", "verdict"]), ""]
    L += ["## 6. What the data support", "", "### Supported primary comparisons (interval excludes the null in the declared direction)", "",
          _md_table(sup) if sup is not None and sup.height else "_none in this run_", "",
          "### Not supported (inconclusive, reversed or without an interval)", "", _md_table(nsup) if nsup is not None and nsup.height else "_none_", "",
          "No result here is causal: Combine and NFL movement share causes (position, body size, role, opportunity). Supported comparisons are associations or held-out "
          "predictive gains on ONE draft cohort.", "",
          "## 7. Decisions for scouts, general managers, coaches and player development", "", _md_table(decision, max_rows=60), "",
          "## 8. Saved artifacts", "", _md_table(by_area), "", "Full list: `manifests/artifact_index.json`; stage records: `manifests/stages.json`.", "",
          "## 9. Limitations", "", *[f"- {x}" for x in (lim or ["—"])], "- Static code review only before this run; this report is the runtime record.", "",
          "## 10. Next research directions", ""]
    if rec is not None and rec.height:
        L += [_md_table(rec), ""]
    L += [*[f"- {x}" for x in (nxt or ["—"])], ""]
    return "\n".join(L)


__all__ = ["DECISIONS", "VALIDATION_PROTOCOLS", "decision_matrix", "eda_report", "evaluation_report", "final_report", "stage_report"]
