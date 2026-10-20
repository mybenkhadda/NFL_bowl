"""E18.F1 — final report and executive summary, written ONLY from computed tables (every sentence is chosen by a computed
status; no number is typed by hand). Missing inputs produce an explicit "not evaluated" line, never an interpretation."""

from __future__ import annotations

from typing import Any

EXEC_TEMPLATES = {
    "scouting_relevant": "Combine tracking ({models}) improved out-of-fold prediction of the NFL target beyond official M1 testing, passed every "
                         "negative control and the permutation test. Treat as a candidate scouting input pending replication on new draft classes.",
    "promising_needs_replication": "Combine tracking ({models}) improved prediction beyond M1 with a validated paired comparison, but not every "
                                   "robustness criterion passed. Do not use for ranking decisions before replication.",
    "precise_null": "No tracked Combine family improved prediction beyond M1, and the intervals are narrow enough to exclude gains larger than "
                    "the practical margin. Use tracking as a descriptive profile, not as a predictor.",
    "inconclusive": "No tracked Combine family showed a validated improvement beyond M1; the intervals are too wide to exclude a modest effect "
                    "(see the detectable-effect table). Use tracking descriptively; a larger linked cohort is needed for a decision.",
    "blocked": "No tracking family reached the prediction stage: every rung beyond M1 was blocked by an upstream measurement gate.",
}


def _tbl(df: Any, cols: list[str], digits: int = 3) -> list[str]:
    if df is None or not getattr(df, "height", 0):
        return ["(not evaluated)"]
    cols = [c for c in cols if c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.select(cols).iter_rows():
        out.append("| " + " | ".join("" if v is None else (f"{v:.{digits}g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return out


def executive_summary(g18: Any, evidence: Any, mde: Any, primary_target: str) -> str:
    import polars as pl

    ov = g18.filter(pl.col("criterion") == "G18 overall")["status"][0] if g18 is not None and g18.height else "not_evaluated"
    best = evidence.filter(pl.col("verdict") == ov)["model"].to_list() if evidence is not None and evidence.height else []
    msg = EXEC_TEMPLATES.get(ov, "The scouting-value analysis did not produce a verdict in this session (see the status ledger).").format(models=", ".join(best) or "—")
    lines = ["# Executive summary — Combine movement tracking and NFL scouting value", "", f"**Verdict ({primary_target}): {ov}.** {msg}", "",
             "**What was tested.** Each tracking family was added to the official Combine baseline (M1: draft year, height, weight, 40-yard and 10-yard times) "
             "and evaluated on athletes never used to fit the model (5 athlete-disjoint folds), with the same athletes, folds and target for every model.", "",
             "**What this does not say.** Nothing here concerns injury risk, force, reaction time or causal development; every result requires replication on new draft classes."]
    if mde is not None and getattr(mde, "height", 0):
        lines += ["", "**Detectable effects with this cohort:**", "", *_tbl(mde, ["target", "mde_fraction", "mde_delta_pct", "max_power_in_grid"])]
    return "\n".join(lines)


def final_report(cfg: Any, prereg: dict[str, Any], rung_frame: Any, A: dict[str, Any], positions: Any, spec: Any, contrast: Any, twins: Any, controls: Any,
                 rlabel: Any, perm: dict[str, Any], sens: Any, evidence: Any, g18: Any) -> str:
    lines = ["# M18 final report — NFL performance transfer and scouting value", "",
             f"Primary target {prereg.get('primary_target')} · pre-registered family {prereg.get('primary_family')} · pre-registration conflict: {prereg.get('conflict')}", "",
             "## 1. Eligible rungs (from upstream gates only)", "", *_tbl(rung_frame, ["rung", "name", "status", "reason"]), "",
             "## 2. Ladder on the primary target (strict cohort)", "", *_tbl(A.get("metrics"), ["model", "n_athletes", "rmse", "mae", "r2", "spearman", "skill_vs_position_oof"]), "",
             "### Paired comparisons vs M1 (95 % fold-stratified athlete bootstrap)", "",
             *_tbl(A.get("paired"), ["model", "reference", "delta_rmse", "delta_lo", "delta_hi", "delta_pct", "p_holm_primary", "p_bh_all", "precise_null", "verdict"]), "",
             "### Declared status", "", *_tbl(A.get("status"), ["model", "reference", "status", "p_adjusted", "folds_better"]), "",
             "## 3. Position-specific models", "", *_tbl(positions, ["position_group", "model", "n_athletes", "delta_pooled_minus_separate", "lo", "hi", "concordance_pooled", "concordance_separate"]), "",
             "## 4. Drill → task specificity", "", *_tbl(contrast, ["drill_family", "diagonal_contrast", "lo", "hi", "reading"]), "",
             "## 5. Stopwatch twins (pooled contrast; case cards in reports/athlete_profiles)", "", *_tbl(twins, ["outcome", "n_pairs", "mean_diff_hi_minus_lo", "ci_lo", "ci_hi", "sign_test_p", "spearman_dcomposite_doutcome"]), "",
             "## 6. Robustness", "", "### Negative controls", "", *_tbl(controls, ["model", "delta_rmse", "delta_lo", "delta_hi", "control_status"]), "",
             "### Random-label sanity", "", *_tbl(rlabel, ["model", "delta_rmse", "delta_lo", "delta_hi"]), "",
             f"### Refit permutation null: {perm}", "", "### Subgroup sensitivity", "", *_tbl(sens, ["subset", "model", "n_athletes", "delta_rmse", "delta_lo", "delta_hi"]), "",
             "## 7. Evidence registry and G18", "", *_tbl(evidence, ["rung", "model", "c1 measurement", "c2 predictive validated", "c3 practical margin", "c4 negative controls",
                                                                 "c5 permutation", "c6 consistency", "verdict"]), "", *_tbl(g18, ["criterion", "status", "note"]), "",
             "## 8. Limitations", "", "- One linked cohort; replication on new draft classes is required for any scouting use.",
             "- The primary NFL target is a constructed movement quantity (12C), not NFL success; task targets were built and frozen in M16.",
             "- 10 Hz tracking without positions, orientation, opponents or ball; Combine and NFL contexts differ (M15 E15.A2).",
             "- B2 simple statistics have no measurement gate in this programme, so they cannot be declared scouting-relevant here."]
    return "\n".join(lines)


__all__ = ["EXEC_TEMPLATES", "executive_summary", "final_report"]
