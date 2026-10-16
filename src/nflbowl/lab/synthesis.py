"""Parts J and K — integrated comparison, research gates and the final report.

Gates are evaluated by DECLARED rules (``GATE_RULES``) from saved tables only. Missing evidence is reported as
``not_evaluated`` and never counted as a pass. The integrated comparison lists objective measurements per
representation; the separate ``researcher_judgment`` column is produced by stated rules and labelled as judgment.
No weighted ranking is computed.
"""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np

PASS, FAIL, INCONCLUSIVE, NOT_EVALUATED = "pass", "fail", "inconclusive", "not_evaluated"
GATE_RULES = {
    "G1 dataset & leakage": "pass: every bundle check and every leakage check has status pass/warn/info; fail: any check 'fail'.",
    "G2 training stability": "pass: every primary-encoder run evaluated has finite losses and a finite best validation loss on ALL outer folds; "
                             "fail: any non-finite loss; inconclusive: finite but not all folds (e.g. pilot).",
    "G3 inductive retrieval": "pass: F4 mean athlete AUC interval > 0.5 AND paired Δ AUC (F4 − F0) interval > 0, on all folds; fail: Δ interval < 0; "
                              "inconclusive otherwise (incl. pilot = one fold).",
    "G4 NFL transfer": "pass: common-cohort Δ RMSE interval > 0 for M1+F4 vs M1 AND vs M1+F0; fail: either interval < 0; inconclusive otherwise.",
    "G5 robustness": "pass: G3 pass and the F4 − F0 Δ AUC keeps its sign under every quality filter, pool definition and extra seed with no interval "
                     "supporting a reversal; fail: an interval supports a reversal; inconclusive otherwise.",
    "G6 interpretability": "pass: interpretable ladder evaluated and ≥ 1 of its top-5 movement metrics has split-half reliability ≥ 0.5 in some scope; "
                           "fail: no movement metric reaches 0.5; inconclusive otherwise.",
    "G7 scouting-facing demonstration": "pass: G1, G3, G4, G5 pass and G6 not fail; fail: G1 or G4 fail; not_evaluated if G3/G4 not evaluated; inconclusive otherwise.",
}


def _f(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def md_table(df: Any, cols: Sequence[str] | None = None, digits: int = 3, max_rows: int = 60) -> str:
    if df is None or not getattr(df, "height", 0):
        return "_none_\n"
    cols = [c for c in (cols or df.columns) if c in df.columns]

    def f(v: Any) -> str:
        if isinstance(v, float):
            return "—" if not math.isfinite(v) else f"{v:.{digits}f}"
        if isinstance(v, (list, dict)):
            return json.dumps(v)[:60]
        return "—" if v is None else str(v).replace("|", "/")

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(r[c]) for c in cols) + " |" for r in df.select(cols).head(max_rows).iter_rows(named=True)]
    if df.height > max_rows:
        lines.append(f"\n_{df.height - max_rows} more rows in the saved table._")
    return "\n".join(lines) + "\n"


# ====================================================================== saved-result registry
RESULT_TABLES = {
    "bundle_checks": ("dataset", "bundle_checks"), "dataset_overview": ("dataset", "overview"), "leakage_checks": ("dataset", "leakage_checks"),
    "shortcuts": ("dataset", "leakage_shortcuts"), "domain_composition": ("dataset", "domain_composition"),
    "geometry_summary": ("geometry", "geometry_summary"), "geometry_table": ("geometry", "geometry_overview"), "collapse_flags": ("geometry", "collapse_flags"),
    "paired_inductive_comparison": ("retrieval", "paired_inductive_comparison"), "rank_summary": ("retrieval", "rank_summary"),
    "paired_pairs": ("retrieval", "paired_pairs"), "retrieval_by_fold": ("retrieval", "retrieval_by_fold"),
    "transductive_reference_table": ("retrieval", "transductive_reference"), "folds_used_table": ("retrieval", "folds_used"),
    "prediction_metrics": ("prediction", "prediction_metrics"), "transfer_comparisons": ("prediction", "paired_comparisons"),
    "position_effects": ("prediction", "position_effects"), "scouting_ranking": ("prediction", "scouting_ranking"),
    "training_health": ("ablations", "training_health"), "probes": ("ablations", "inductive_probes"), "complexity": ("ablations", "complexity"),
    "ablation_summary": ("ablations", "ablation_summary"), "pool_comparison": ("robustness", "pool_comparison"),
    "quality_comparison": ("robustness", "quality_comparison"), "seed_comparison": ("robustness", "seed_comparison"),
    "learning_curve": ("robustness", "learning_curve"), "repeatability": ("biomechanics", "repeatability"),
    "cross_domain_stability": ("biomechanics", "cross_domain_stability"), "incremental_value": ("biomechanics", "incremental_value"),
    "hypothesis_tests": ("biomechanics", "hypothesis_tests"), "importance": ("interpretability", "permutation_importance"),
}


def collect(run: Any) -> dict[str, Any]:
    """Every result table this run has SAVED (a new session rebuilds the synthesis without recomputation)."""
    out: dict[str, Any] = {}
    for key, (fam, name) in RESULT_TABLES.items():
        if run.has(name, fam):
            out[key] = run.frame(name, fam)
    f = out.get("folds_used_table")
    out["folds_used"] = sorted(int(x) for x in f["fold"].to_list()) if f is not None and f.height else []
    return out


# ====================================================================== gates
def evaluate_gates(r: dict[str, Any], primary: str = "F4_physics_contrastive", reference: str = "F0_engineered", all_folds: int = 5) -> Any:
    """``r``: saved tables by name (missing keys = not produced). Returns gate · status · evidence · rule."""
    import polars as pl

    out = []

    def add(g: str, status: str, evidence: str) -> None:
        out.append({"gate": g, "status": status, "evidence": evidence, "rule": GATE_RULES[g]})

    # G1
    bc, lk = r.get("bundle_checks"), r.get("leakage_checks")
    if bc is None or lk is None:
        add("G1 dataset & leakage", NOT_EVALUATED, "bundle checks or leakage audit missing")
    else:
        nf = bc.filter(pl.col("status") == "fail").height + lk.filter(pl.col("status") == "fail").height
        add("G1 dataset & leakage", FAIL if nf else PASS, f"{nf} failing check(s); {lk.filter(pl.col('status') == 'warn').height} warning(s)")
    g1 = out[-1]["status"]
    # G2
    th = r.get("training_health")
    if th is None or not th.height:
        add("G2 training stability", NOT_EVALUATED, "no training curves in this run (none trained or adopted)")
    else:
        t = th.filter(pl.col("model") == primary) if "model" in th.columns else th
        if not t.height:
            add("G2 training stability", NOT_EVALUATED, f"no {primary} curves")
        else:
            bad = t.filter(~pl.col("all_losses_finite") | ~pl.col("best_val_loss").is_finite()).height
            ind = t.filter(pl.col("tag").str.starts_with("inductive")).height
            st = FAIL if bad else (PASS if ind >= all_folds else INCONCLUSIVE)
            add("G2 training stability", st, f"{t.height} run(s), {ind} inductive fold(s), {bad} with non-finite values")
    # G3
    comp = r.get("paired_inductive_comparison")
    folds_used = r.get("folds_used") or []
    if comp is None or not comp.height:
        add("G3 inductive retrieval", NOT_EVALUATED, "no paired inductive comparison")
    else:
        c = comp.filter((pl.col("metric") == "auc") & (pl.col("representation") == primary) & (pl.col("status") == "ok"))
        if not c.height:
            add("G3 inductive retrieval", NOT_EVALUATED, f"{primary} not scored")
        else:
            x = c.row(0, named=True)
            lo_m, dlo, dhi = _f(x.get("mean_lo")), _f(x.get("delta_lo")), _f(x.get("delta_hi"))
            ev_ = f"{primary} AUC {_f(x.get('mean')):.3f} [{lo_m:.3f}, {_f(x.get('mean_hi')):.3f}], Δ vs {reference} {_f(x.get('delta')):.3f} [{dlo:.3f}, {dhi:.3f}], n={x.get('n_athletes')}, folds={folds_used}"
            if np.isfinite(dhi) and dhi < 0:
                st = FAIL
            elif np.isfinite(dlo) and dlo > 0 and np.isfinite(lo_m) and lo_m > 0.5 and len(folds_used) >= all_folds:
                st = PASS
            else:
                st = INCONCLUSIVE
            add("G3 inductive retrieval", st, ev_)
    g3 = out[-1]["status"]
    # G4
    tc = r.get("transfer_comparisons")
    if tc is None or not tc.height:
        add("G4 NFL transfer", NOT_EVALUATED, "transfer ladder not run")
    else:
        m = f"M1+{primary}"
        c = tc.filter((pl.col("cohort") == "common") & (pl.col("model") == m))
        a = c.filter(pl.col("reference") == "M1")
        b = c.filter(pl.col("reference") == "M1+F0")
        if not (a.height and b.height):
            add("G4 NFL transfer", NOT_EVALUATED, f"{m} comparisons missing")
        else:
            ra, rb = a.row(0, named=True), b.row(0, named=True)
            ev_ = (f"Δ RMSE vs M1 {_f(ra['delta_rmse']):.4f} [{_f(ra['delta_lo']):.4f}, {_f(ra['delta_hi']):.4f}]; vs M1+F0 {_f(rb['delta_rmse']):.4f} "
                   f"[{_f(rb['delta_lo']):.4f}, {_f(rb['delta_hi']):.4f}]; n={ra['n_athletes']}")
            if (_f(ra["delta_hi"]) < 0) or (_f(rb["delta_hi"]) < 0):
                st = FAIL
            elif _f(ra["delta_lo"]) > 0 and _f(rb["delta_lo"]) > 0:
                st = PASS
            else:
                st = INCONCLUSIVE
            add("G4 NFL transfer", st, ev_)
    g4 = out[-1]["status"]
    # G5
    sens = []
    for key in ("quality_comparison", "pool_comparison"):                  # F4 − F0 under alternative filters / pools
        t = r.get(key)
        if t is not None and t.height and "representation" in t.columns:
            rows = t.filter((pl.col("representation") == primary) & pl.col("delta").is_not_null())
            if "metric" in rows.columns:
                rows = rows.filter(pl.col("metric") == "auc")
            sens.append(rows.select(pl.lit(key).alias("source"), "delta", "delta_lo", "delta_hi"))
    sc = r.get("seed_comparison")                                          # extra seed vs main run of the same model
    seed_unstable = 0
    if sc is not None and sc.height and "ci_excludes_zero" in sc.columns:
        seed_unstable = sc.filter((pl.col("representation") != primary) & pl.col("ci_excludes_zero").fill_null(False)).height
    if not sens:
        add("G5 robustness", NOT_EVALUATED, "no sensitivity analysis available")
    else:
        s = pl.concat(sens, how="diagonal_relaxed")
        rev = s.filter(pl.col("delta_hi") < 0).height
        flips = s.filter(pl.col("delta") <= 0).height
        if rev:
            st = FAIL
        elif g3 == PASS and flips == 0 and seed_unstable == 0:
            st = PASS
        else:
            st = INCONCLUSIVE
        add("G5 robustness", st, f"{s.height} sensitivity estimate(s): {flips} with Δ ≤ 0, {rev} with interval supporting a reversal; "
                                 f"{seed_unstable} extra-seed run(s) differing from the main run (interval)")
    g5 = out[-1]["status"]
    # G6
    imp, rel = r.get("importance"), r.get("repeatability")
    if imp is None or not imp.height or rel is None or not rel.height:
        add("G6 interpretability", NOT_EVALUATED, "interpretable ladder or repeatability missing")
    else:
        top = imp.filter((pl.col("position_group") == "ALL") & pl.col("feature").str.starts_with("c_")).sort("importance_delta_rmse", descending=True).head(5)
        base = [f.split("__", 1)[1] for f in top["feature"].to_list() if "__" in f]
        good = rel.filter(pl.col("metric").is_in(base) & (pl.col("reliability") >= 0.5)).height if base else 0
        anyrel = rel.filter(pl.col("reliability") >= 0.5).height
        st = PASS if good else (FAIL if anyrel == 0 else INCONCLUSIVE)
        add("G6 interpretability", st, f"top movement metrics {base}; {good} reliable (≥ 0.5) scope(s); {anyrel} reliable metric scope(s) overall")
    g6 = out[-1]["status"]
    # G7
    if g1 == FAIL or g4 == FAIL:
        st = FAIL
    elif NOT_EVALUATED in (g3, g4):
        st = NOT_EVALUATED
    elif g1 == PASS and g3 == PASS and g4 == PASS and g5 == PASS and g6 != FAIL:
        st = PASS
    else:
        st = INCONCLUSIVE
    add("G7 scouting-facing demonstration", st, f"G1 {g1} · G3 {g3} · G4 {g4} · G5 {g5} · G6 {g6}")
    return pl.DataFrame(out)


# ====================================================================== integrated comparison
def integrated_comparison(r: dict[str, Any], gates: Any) -> Any:
    """One row per representation with objective columns from saved tables + a rule-based judgment column."""
    import polars as pl

    names: list[str] = []
    for key, col in (("paired_inductive_comparison", "representation"), ("geometry_summary", "representation"), ("complexity", "representation")):
        t = r.get(key)
        if t is not None and t.height and col in t.columns:
            names += [x for x in t[col].unique(maintain_order=True).to_list() if x not in names]
    rows = []
    comp, geo, prb, cx, bf, tc = (r.get(k) for k in ("paired_inductive_comparison", "geometry_summary", "probes", "complexity", "retrieval_by_fold",
                                                       "transfer_comparisons"))
    g4 = dict(zip(gates["gate"].to_list(), gates["status"].to_list())) if gates is not None else {}
    for n in names:
        row: dict[str, Any] = {"representation": n}
        if comp is not None and comp.height:
            c = comp.filter((pl.col("representation") == n) & (pl.col("metric") == "auc") & (pl.col("status") == "ok"))
            if c.height:
                x = c.row(0, named=True)
                row.update({"inductive_auc": x.get("mean"), "auc_lo": x.get("mean_lo"), "auc_hi": x.get("mean_hi"), "delta_auc_vs_ref": x.get("delta"),
                            "delta_lo": x.get("delta_lo"), "delta_hi": x.get("delta_hi"), "n_athletes_retrieval": x.get("n_athletes")})
        if bf is not None and bf.height:
            b = bf.filter(pl.col("representation") == n)
            row["auc_fold_sd"] = float(b["auc"].std()) if b.height > 1 else None
        if geo is not None and geo.height:
            g = geo.filter(pl.col("representation") == n)
            for met in ("domain_auc_grouped", "position_adjusted_domain_auc", "mmd2", "effective_rank"):
                v = g.filter(pl.col("metric") == met)["mean"]
                row[met] = float(v[0]) if v.len() else None
        if prb is not None and prb.height:
            p = prb.filter(pl.col("representation") == n)["r2_cv"]
            row["probe_r2_median"] = float(p.median()) if p.len() else None
        if tc is not None and tc.height:
            t = tc.filter((pl.col("cohort") == "common") & (pl.col("model") == f"M1+{n}") & (pl.col("reference") == "M1"))
            if t.height:
                x = t.row(0, named=True)
                row.update({"delta_rmse_vs_M1": x["delta_rmse"], "delta_rmse_lo": x["delta_lo"], "delta_rmse_hi": x["delta_hi"], "n_athletes_transfer": x["n_athletes"]})
        if cx is not None and cx.height:
            k = cx.filter(pl.col("representation") == n)
            if k.height:
                row.update({"parameters_total": k["parameters_total"][0], "training_seconds_saved": k["training_seconds_saved"][0]})
        # rule-based judgment (labelled as such)
        notes = []
        dlo = _f(row.get("delta_lo"))
        if np.isfinite(dlo) and dlo > 0:
            notes.append("retrieval above reference (interval)")
        elif row.get("inductive_auc") is not None:
            notes.append("retrieval not distinguishable from reference")
        if _f(row.get("domain_auc_grouped")) >= 0.9:
            notes.append("domains remain linearly separable")
        if np.isfinite(_f(row.get("delta_rmse_lo"))) and _f(row.get("delta_rmse_lo")) > 0:
            notes.append("adds out-of-fold transfer value over M1 (interval)")
        if g4.get("G7 scouting-facing demonstration") != PASS:
            notes.append("not yet evidence for scouting use")
        row["researcher_judgment (rule-based)"] = "; ".join(notes) or "insufficient evidence"
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# ====================================================================== programmatic interpretation + section reports
def describe_interval(label: str, est: Any, lo: Any, hi: Any, unit: str = "", positive_means: str = "better", null: float = 0.0) -> str:
    """One sentence generated from a measured estimate and its interval (never 'no effect' from an interval covering the null)."""
    e, l_, h = _f(est), _f(lo), _f(hi)
    if not np.isfinite(e):
        return f"- {label}: not estimable from the available data."
    if not (np.isfinite(l_) and np.isfinite(h)):
        return f"- {label}: {e:.4g}{unit} (no interval: too few athletes) — descriptive only."
    if l_ > null:
        verdict = f"interval above {null:g}: supported ({positive_means})"
    elif h < null:
        verdict = f"interval below {null:g}: supported in the opposite direction"
    else:
        verdict = f"interval includes {null:g}: inconclusive (not evidence of no effect)"
    return f"- {label}: {e:.4g}{unit} [{l_:.4g}, {h:.4g}] — {verdict}."


def section_report(run: Any, name: str, title: str, question: str, rationale: str, methods: Sequence[str], tables: dict[str, Any],
                   interpretation: Sequence[str], limitations: Sequence[str], status: str = "") -> list[Any]:
    """One Markdown + HTML report per experiment family (reports/markdown/<name>.md): research question → rationale →
    methods → results (saved tables) → generated interpretation → limitations → saved artifacts."""
    figs = [f for f in run.results.get("_figures", []) if f.get("figure", "").startswith(name.split("_")[0].lower())]
    md = [f"# {title}", "", f"_Stage status: {status or 'see manifests/stages.json'}. Generated from saved results; interpretation sentences are "
                            "produced from the measured intervals by fixed rules._", "",
          "## Research question", "", question, "", "## Scientific rationale", "", rationale, "", "## Data and methods", "", *[f"- {m}" for m in methods], "",
          "## Results", ""]
    for k, df in tables.items():
        md += [f"### {k}", "", md_table(df), ""]
    md += ["## Interpretation (generated)", "", *(interpretation or ["- No measured result to interpret."]), "", "## Limitations", "",
           *[f"- {x}" for x in limitations], "", "## Saved artifacts", "", *[f"- {f.get('png', f.get('figure'))}" for f in figs], ""]
    return run.save_markdown("\n".join(md), name)


# ====================================================================== final report
def _figs(run: Any, prefix: str = "") -> list[str]:
    return sorted(f"figures/png/{p.name}" for p in (run.root / "figures" / "png").glob(f"{prefix}*.png"))


def final_report(run: Any, r: dict[str, Any], gates: Any, integrated: Any, status: Any, cfg: Any, env: dict[str, Any], exp: Any) -> str:
    """FINAL_COMPREHENSIVE_RESEARCH_REPORT.md from saved / loaded results only; every section states its evidence class."""
    import polars as pl

    st = dict(zip(status["id"].to_list(), status["status"].to_list())) if status is not None else {}

    def cls(ids: Sequence[str]) -> str:
        s = [st.get(i, "pending") for i in ids]
        if all(x in ("completed", "cached") for x in s):
            return "**Measured** (executed or reloaded in this run)"
        if any(x in ("completed", "cached") for x in s):
            return f"**Partially measured** (stage status: {', '.join(f'{i}={st.get(i)}' for i in ids)})"
        return f"**Not yet executed** (stage status: {', '.join(f'{i}={st.get(i, 'pending')}' for i in ids)}) — design only"

    T = lambda k, cols=None, d=3: md_table(r.get(k), cols, d)    # noqa: E731
    gate_line = "; ".join(f"{g}: **{s}**" for g, s in zip(gates["gate"].to_list(), gates["status"].to_list())) if gates is not None else "—"
    comp = r.get("paired_inductive_comparison")
    md = [f"# Final comprehensive research report — {cfg.experiment_name}", "",
          f"_Generated {env.get('recorded_at')} from saved artifacts of this run. Nothing in this file is typed by hand; sections without executed "
          "results say so. Prior exploratory numbers (F0 0.4968 / random 0.5002 / F4 0.5241 transductive AUC) are NOT used as results here._", "",
          "## 1. Executive summary", "", f"Research gates: {gate_line}.", "",
          md_table(gates, ["gate", "status", "evidence"]),
          "Reading rule: a gate passes only on interval evidence from this run; missing evidence is 'not_evaluated'.", "",
          "## 2. Research objectives", "",
          "Test whether 10 Hz Combine movement carries athlete-specific, domain-robust information that (i) retrieves the same athlete's NFL movement "
          "for athletes unseen in training and (ii) improves out-of-fold prediction of the frozen NFL movement-capacity target beyond official Combine "
          "measurements (M1) and engineered summaries (F0).", "",
          "## 3. Data", "", cls(["B1"]), "", T("dataset_overview"), T("bundle_checks", ["check", "expected", "observed", "status"]),
          "## 4. Methodology", "",
          "- Athlete-level outer folds from the 12.0 fold manifest; inductive encoders trained without any held-out athlete window (both domains).",
          "- Retrieval: athlete Combine-mean query vs same-fold, same-position NFL-mean candidates; paired athlete bootstrap stratified by fold; identical pools.",
          "- Transfer: Milestone 12 ridge ladder, train-fold-only design and embedding PCA, inner-CV α, paired athlete bootstrap of Δ RMSE.",
          "- Biomechanics: deterministic per-sequence metrics; reliability by split halves; incremental value via the same ladder; BH within families.", "",
          "## 5. Experiment matrix", "", md_table(status, ["id", "title", "status", "compute", "feasibility", "note"]),
          "## 6. Baselines", "", "F0 engineered (17 fixed statistics), random frozen F4 architecture, M1 officials, M1+F0. "
          "F1 TCN enters the transfer ladder only (external athlete-level representation).", "",
          "## 7. Representation-learning results (geometry, collapse, domain)", "", cls(["C1", "C3"]), "", T("geometry_table"), T("collapse_flags"), "",
          "## 8. Inductive retrieval", "", cls(["D3", "D4"]), "",
          md_table(comp.filter(pl.col("metric") == "auc") if comp is not None and comp.height else None,
                   ["representation", "n_athletes", "n_folds", "mean", "mean_lo", "mean_hi", "delta", "delta_lo", "delta_hi", "reliability"]),
          T("rank_summary"), T("paired_pairs", ["representation", "reference", "metric", "n_athletes", "delta", "delta_lo", "delta_hi", "reliability"]),
          "### Transductive reference (separate regime, never pooled with inductive)", "", cls(["D1"]), "", T("transductive_reference_table"),
          "## 9. NFL outcome transfer", "", cls(["E2"]), "", T("prediction_metrics"),
          T("transfer_comparisons", ["cohort", "model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict"], 4),
          T("position_effects", ["position_group", "reliability", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "p_boot_le0_bh"], 4),
          T("scouting_ranking"), "",
          "## 10. Domain-shift findings", "", cls(["B2", "C3"]), "", T("domain_composition"), T("shortcuts"), "",
          "Domain separability is not failure on its own (different tasks). Lower MMD² with high domain AUC means distribution alignment without invariance.", "",
          "## 11. Ablations", "", cls(["F1"]), "", T("ablation_summary"), "",
          "## 12. Robustness", "", cls(["G3", "G4", "G5"]), "", T("pool_comparison", ["pool", "representation", "metric", "n_athletes", "delta", "delta_lo", "delta_hi", "chance"]),
          T("quality_comparison", ["filter", "representation", "n_athletes", "delta", "delta_lo", "delta_hi"]), T("seed_comparison"), T("learning_curve"), "",
          "## 13. Biomechanics interpretation", "", cls(["H1"]), "", T("repeatability", ["metric", "domain", "scope", "n_athletes", "reliability", "rel_lo", "rel_hi"]),
          T("cross_domain_stability"), T("incremental_value", ["metric", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "p_boot_le0_bh", "verdict"], 4),
          T("hypothesis_tests", ["hypothesis", "metric", "controls", "n_athletes", "partial_spearman", "lo", "hi", "p_bh", "status"]), "",
          "## 14. Athlete case studies", "", cls(["D5", "I1"]), "",
          "Retrieval case studies and research profiles are exported under figures/ and reports/athlete_profiles/ with salted pseudonyms. "
          "They illustrate model behaviour and are not personnel recommendations.", "",
          "## 15. Statistical limitations", "",
          "- Candidate pools within fold × position are small: athlete AUCs are coarse and the bootstrap holds pools fixed.",
          "- Five encoders per family (one per fold) are pooled after per-fold scoring.",
          "- The 12C target estimator was selected on NFL data of all athletes before any Combine join (definition choice, not a predictor).",
          "- The ridge α choice uses a fold-fitted block not refitted per inner fold (mildly optimistic for both arms of a comparison).",
          "- Subgroup analyses are descriptive; no multiplicity-adjusted subgroup claim is made unless a BH column is shown.",
          "- turn_asymmetry relies on an unverified sign convention of a_n across sources.", "",
          "## 16. Evidence-supported conclusions", "", *conclusions(gates), "",
          "## 17. Practical implications", "", *implications(gates), "",
          "## 18. Next research priorities", "", *priorities(gates, st), "",
          "## 19. Reproducibility appendix", "",
          f"- Bundle `{exp.bundle.fingerprint if exp is not None else '—'}` · science hash `{exp.cfg.science_hash() if exp is not None else '—'}` · "
          f"science seed {exp.cfg.science.seed if exp is not None else '—'} · lab seed {cfg.seed} · precision {cfg.precision} · run mode {cfg.run_mode}",
          f"- Code `{json.dumps(env.get('code_version'))}` · GPU {env.get('gpu', {}).get('device')}",
          "- Exact configuration: config/run_config.json; environment: config/environment.json; stage ledger: manifests/stages.json; "
          "artifact log: manifests/artifacts.jsonl.", "",
          "## 20. Saved figures and tables", "", "### Figures", "", *[f"- [{p}](../../{p})" for p in _figs(run)], "", "### Tables", "",
          *[f"- [{p.relative_to(run.root).as_posix()}](../../{p.relative_to(run.root).as_posix()})" for p in sorted((run.root / "tables").rglob("*.csv"))], ""]
    return "\n".join(md)


def conclusions(gates: Any) -> list[str]:
    if gates is None:
        return ["- No gate evaluated."]
    s = dict(zip(gates["gate"].to_list(), gates["status"].to_list()))
    e = dict(zip(gates["gate"].to_list(), gates["evidence"].to_list()))
    out = []
    for g in ("G3 inductive retrieval", "G4 NFL transfer", "G5 robustness", "G6 interpretability"):
        v = s.get(g)
        if v == PASS:
            out.append(f"- **Supported** — {g}: {e[g]}")
        elif v == FAIL:
            out.append(f"- **Contradicted** — {g}: {e[g]}")
        elif v == INCONCLUSIVE:
            out.append(f"- **Inconclusive** — {g}: {e[g]}")
        else:
            out.append(f"- **Not evaluated** — {g}.")
    return out


def implications(gates: Any) -> list[str]:
    s = dict(zip(gates["gate"].to_list(), gates["status"].to_list())) if gates is not None else {}
    if s.get("G7 scouting-facing demonstration") == PASS:
        return ["- The evidence gates support a scouting-facing DEMONSTRATION of the learned representation (not a validated decision tool)."]
    return ["- Evidence is not yet sufficient for a scouting-facing claim about the learned encoders; interpretable movement metrics that pass their "
            "own reliability and incremental-value checks (Part H tables) are the candidates to present, with their intervals.",
            "- No statement about real NFL coaching impact is supported by these analyses."]


def priorities(gates: Any, st: dict[str, str]) -> list[str]:
    s = dict(zip(gates["gate"].to_list(), gates["status"].to_list())) if gates is not None else {}
    out = []
    if s.get("G3 inductive retrieval") in (NOT_EVALUATED, INCONCLUSIVE):
        out.append("- Complete inductive training / evaluation of F2 and F4 on all five folds (run_mode='standard', run_training=True).")
    if s.get("G4 NFL transfer") == NOT_EVALUATED:
        out.append("- Run the transfer ladder (run_transfer=True) once all folds have inductive encoders.")
    if st.get("F1") != "completed":
        out.append("- Mechanism ablations (run_mode='full', run_ablations=True) once the primary comparison is stable.")
    out.append("- Warehouse-dependent 12A–12G studies: run M12_UNIFIED_COLAB_RESEARCH.ipynb with the research root on Drive.")
    return out


__all__ = ["FAIL", "GATE_RULES", "INCONCLUSIVE", "NOT_EVALUATED", "PASS", "RESULT_TABLES", "collect", "conclusions", "describe_interval", "section_report", "evaluate_gates", "final_report", "integrated_comparison", "md_table"]
