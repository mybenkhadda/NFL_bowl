"""Formal experiment registry of the comprehensive lab (one row per experiment / notebook section).

Each ``Spec`` declares the hypothesis, inputs, representation, training regime, held-out protocol, baseline, metrics,
statistical comparison, outputs, compute class, the run modes / flag that enable it, the decision rule used to read it
and whether the portable bundle suffices. ``status_table(run)`` joins the specs with the stage ledger of a ``LabRun`` —
a status is never inferred: an experiment without a stage record is ``pending``.

Status vocabulary: completed (executed in this run) · cached (reused saved artifacts) · pending (not yet run) ·
skipped (disabled by configuration or prerequisite missing) · blocked (needs unavailable data) · failed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

STATUS_MAP = {"executed": "completed", "reused": "cached", "skipped": "skipped", "blocked": "blocked", "failed": "failed", "running": "running"}


@dataclass(frozen=True)
class Spec:
    id: str
    part: str
    title: str
    hypothesis: str
    inputs: str
    representation: str
    training_regime: str
    held_out_protocol: str
    baseline: str
    metrics: str
    statistics: str
    outputs: str
    compute: str               # cpu-light · cpu-heavy · gpu-light · gpu-heavy
    modes: str                 # run modes in which it runs (subject to its flag)
    flag: str                  # CFG flag(s) that gate it ('' = always)
    interpretation_rule: str
    feasibility: str           # portable_bundle · portable_bundle+training · requires_warehouse · api_limited


SPECS: tuple[Spec, ...] = (
    Spec("A2", "A", "Reproducibility record", "n/a (provenance)", "bundle.json, run config, environment", "—", "—", "—", "—",
         "bundle fingerprint, science hash, versions", "—", "config/run_config.json, environment.json", "cpu-light", "all", "",
         "Every later table is read only together with this record.", "portable_bundle"),
    Spec("B1", "B", "Dataset audit", "The cleaned bundle matches its declared provenance (77,125 × 64 × 6, 510 athletes, 5 folds).",
         "bundle", "—", "—", "—", "declared provenance", "counts, coverage, valid frames, missingness, fold balance", "exact counts (no sampling)",
         "tables/dataset/*", "cpu-light", "all", "", "A mismatch with the declared provenance fails Gate 1.", "portable_bundle"),
    Spec("B2", "B", "Domain composition", "Combine drills and NFL windows differ in speed / acceleration / heading / duration even within position.",
         "bundle, derived physics targets", "per-sequence physical summaries", "—", "—", "Combine distribution", "medians, quantiles, standardized differences, position-adjusted gaps",
         "athlete-cluster bootstrap of the position-adjusted median difference", "tables/dataset/domain_*", "cpu-light", "all", "",
         "Large shifts are expected (different tasks); they bound what domain invariance can mean.", "portable_bundle"),
    Spec("B3", "B", "Sample selection and attrition", "Each analysis uses a documented subset; attrition is not random across positions.",
         "bundle, athletes.parquet", "—", "—", "—", "all athletes", "attrition counts per step and position", "descriptive",
         "tables/dataset/attrition*", "cpu-light", "all", "", "Positions with < min_subgroup eligible athletes are flagged insufficient.", "portable_bundle"),
    Spec("B4", "B", "Leakage audit", "Folds are athlete-disjoint; no fitted transform or outcome crosses folds; no domain shortcut is trivially available.",
         "bundle, splits, embedding tables, training records", "—", "inductive", "athlete-level outer folds", "—",
         "split integrity, role checks, duplicate sequences, shortcut-feature domain AUC, preprocessing scope", "exact checks",
         "diagnostics/leakage_*", "cpu-light", "all", "", "Any failed check fails Gate 1; checks verified only by code inspection are labelled so.",
         "portable_bundle"),
    Spec("C1", "C", "Representation geometry and collapse", "Learned encoders are not collapsed (effective rank ≫ 1, no dead dims).",
         "inductive / F0 embeddings", "F0, random, F2, F4 (+ secondary)", "inductive", "held-out rows of each fold", "F0, random",
         "effective rank, participation ratio, PC1 share, norms, cosine / NN distance distributions", "per-fold values, min–max over folds",
         "tables/geometry/*", "cpu-light", "all", "run_full_diagnostics", "Effective rank < 2 or > 20% dead dims = collapse flag.", "portable_bundle"),
    Spec("C3", "C", "Domain geometry", "F4 reduces domain mean discrepancy (MMD²) but stays linearly domain-separable.",
         "inductive embeddings", "F0, random, F2, F4", "inductive", "held-out rows", "F0", "domain AUC (athlete-grouped), position-adjusted domain AUC, MMD², centroid gap",
         "fold variability", "tables/geometry/domain_*", "cpu-light", "all", "run_full_diagnostics",
         "Lower MMD² with domain AUC still ≥ 0.9 = distribution alignment without invariance.", "portable_bundle"),
    Spec("C4", "C", "Exploratory projections", "n/a (visualization only)", "inductive embeddings", "F0, F2, F4", "inductive", "held-out rows of one fold",
         "—", "PCA (+ UMAP if installed) coordinates", "none — exploratory", "figures/*projection*", "cpu-light", "all", "run_full_diagnostics",
         "Never used as evidence of separation or generalization.", "portable_bundle"),
    Spec("D1", "D", "Transductive reference", "Earlier transductive F4 > F0 retrieval (≈0.52 vs ≈0.50) is reproducible from saved records.",
         "saved metrics.csv of reference experiments", "F0, random, F4", "transductive", "none (all athletes seen)", "F0",
         "cross-domain AUC, domain AUC, MMD²", "bootstrap CI from the record", "tables/retrieval/transductive_reference", "cpu-light", "all", "",
         "Shown only when bundle fingerprint and evaluation level match; never compared with inductive numbers.", "portable_bundle"),
    Spec("D2", "D", "Inductive encoder training", "Encoders can be trained without any held-out athlete window.", "bundle", "F2, F4 (+ secondary)",
         "inductive", "5 athlete-level outer folds", "—", "validation loss, epochs, time", "—", "checkpoints/, tables/ablations/training_*",
         "gpu-heavy", "standard, full (pilot: one fold)", "run_training", "Non-finite losses or capped runs fail Gate 2.", "portable_bundle+training"),
    Spec("D3", "D", "Inductive retrieval", "F4 retrieves an unseen athlete's NFL movement from Combine movement above chance and above F0.",
         "inductive embeddings", "F0, F2, F4", "inductive", "held-out athletes vs same-fold same-position candidates", "F0 and chance",
         "athlete AUC, top-1, recall@K, MRR, median rank, own-vs-negative similarity", "paired athlete bootstrap stratified by fold, identical pools",
         "tables/retrieval/*", "cpu-light", "all", "run_inductive", "Supported only if the paired Δ AUC interval excludes 0 and AUC > chance.",
         "portable_bundle"),
    Spec("D4", "D", "Paired retrieval comparisons", "F4 > F0, F4 > F2, F2 > F0 on the same held-out athletes.", "athlete retrieval scores",
         "F0, F2, F4, random", "inductive", "identical candidate pools", "F0 / F2", "Δ mean AUC, Δ MRR, Δ top-5", "paired bootstrap (fold-stratified)",
         "tables/retrieval/paired_*", "cpu-light", "all", "run_inductive", "Δ interval excluding 0 = supported; otherwise inconclusive (never 'no effect').",
         "portable_bundle"),
    Spec("D5", "D", "Retrieval case studies", "n/a (descriptive)", "athlete scores, sequences", "F4 vs F0", "inductive", "held-out athletes", "—",
         "own rank, similarity margins, speed profiles", "none — illustrative", "figures/retrieval_case_*", "cpu-light", "all", "run_inductive",
         "Illustrations, selected by explicit rules; not evidence.", "portable_bundle"),
    Spec("E2", "E", "NFL outcome-transfer ladder", "Inductive tracking embeddings add out-of-fold predictive value for nfl.q90_shared beyond M1 and M1+F0.",
         "athletes.parquet (frozen target), inductive embeddings", "M1, M1+F0, M1+F1, M1+F2, M1+F4", "inductive", "athlete-level outer folds (ladder)",
         "M1, M1+F0", "RMSE, MAE, R², bias, residual quantiles, Δ RMSE", "paired athlete bootstrap; per-fold variability", "tables/prediction/*",
         "cpu-light", "standard, full", "run_transfer", "Δ RMSE interval > 0 vs M1 AND vs M1+F0 = Gate 4 pass.", "portable_bundle"),
    Spec("E5", "E", "Position-level transfer", "Gains (if any) concentrate in specific position groups.", "OOF predictions", "M1+F4 vs M1", "inductive",
         "same folds", "M1", "Δ RMSE per position", "paired bootstrap per position, BH-adjusted", "tables/prediction/position_*", "cpu-light",
         "standard, full", "run_transfer", "Positions with n < 30 are exploratory; < 10 not interpreted.", "portable_bundle"),
    Spec("E7", "E", "Scouting value", "Tracking features change top-decile identification and rankings relative to M1.", "OOF predictions", "M1 vs M1+X",
         "inductive", "same folds", "M1", "Spearman, top-decile precision / recall, calibration, threshold sensitivity", "athlete bootstrap",
         "tables/prediction/scouting_*", "cpu-light", "standard, full", "run_transfer", "Decision metrics are descriptive; no coaching-impact claim.",
         "portable_bundle"),
    Spec("F1", "F", "Objective ablations", "Removing alignment / physics / shuffling physics changes retrieval and domain separation in predicted directions.",
         "inductive ablation encoders", "F4 + registry ablations", "inductive", "same folds as F4", "F4", "paired Δ AUC, domain AUC, MMD², effective rank, probes",
         "paired bootstrap vs F4", "tables/ablations/*", "gpu-heavy", "full", "run_ablations + run_training",
         "Each ablation reported with benefits AND regressions.", "portable_bundle+training"),
    Spec("G1", "G", "Seed sensitivity", "F4 retrieval is stable across training seeds.", "extra-seed experiments", "F4", "inductive", "pilot fold",
         "seed of the main run", "Δ AUC between seeds", "paired bootstrap", "tables/robustness/seeds*", "gpu-heavy", "full", "run_seeds + run_training",
         "Seed spread comparable to model differences = differences not interpretable.", "portable_bundle+training"),
    Spec("G2", "G", "NFL-data fraction learning curve", "More unlabeled NFL windows improve alignment.", "transductive lc runs", "F4", "transductive",
         "none (transductive diagnostic)", "F4 @100%", "cross-domain AUC vs fraction", "bootstrap CI", "tables/robustness/learning_curve",
         "gpu-heavy", "full", "run_learning_curves + run_training", "API supports transductive fractions only — labelled diagnostic, not data efficiency.",
         "api_limited"),
    Spec("G3", "G", "Candidate-pool difficulty", "Retrieval advantage persists against smaller pools and stopwatch-twin hard negatives.",
         "inductive embeddings, M1 officials", "F0, F2, F4", "inductive", "fixed sub-pools shared by all representations", "chance for each pool",
         "AUC / top-1 vs pool size and hard-negative pools", "paired bootstrap", "tables/robustness/pool_*", "cpu-light", "all", "run_inductive",
         "Chance baseline reported per pool definition.", "portable_bundle"),
    Spec("G4", "G", "Sequence-quality sensitivity", "Results do not depend on short / extreme / low-heading-quality sequences.", "inductive embeddings, bundle",
         "F0, F4", "inductive", "quality-filtered held-out rows", "unfiltered", "AUC under each filter", "paired bootstrap", "tables/robustness/quality_*",
         "cpu-light", "standard, full", "run_full_diagnostics", "A sign flip under a filter = fragile.", "portable_bundle"),
    Spec("G5", "G", "Population heterogeneity", "Retrieval and transfer effects are consistent across draft years, combine completeness and NFL coverage.",
         "scores + athletes", "F0, F4", "inductive", "same folds", "F0", "subgroup Δ", "per-subgroup paired bootstrap, BH", "tables/robustness/heterogeneity_*",
         "cpu-light", "standard, full", "run_full_diagnostics", "Subgroups below min_subgroup are not interpreted.", "portable_bundle"),
    Spec("H1", "H", "Biomechanical movement metrics", "Interpretable Combine movement metrics are repeatable, transfer across domains and add value beyond M1.",
         "bundle sequences, athletes.parquet", "engineered biomechanics (no fitting)", "none (deterministic features)", "athlete-level ladder folds",
         "M1", "split-half reliability, Combine↔NFL Spearman, Δ RMSE", "athlete bootstrap; BH across metrics", "tables/biomechanics/*", "cpu-heavy",
         "all (ladder: standard, full)", "run_full_diagnostics", "A metric is 'actionable candidate' only if reliable (≥0.5), defined in both domains and Δ RMSE CI > 0.",
         "portable_bundle"),
    Spec("I1", "I", "Interpretability and profiles", "The ladder's predictive signal can be attributed to physically meaningful inputs.",
         "OOF ladder on M1 + biomechanics", "ridge ladder", "inductive", "athlete-grouped permutation within test folds", "—",
         "permutation importance, partial dependence, counterfactual sensitivity", "fold-level spread", "tables/interpretability/*, reports/athlete_profiles/*",
         "cpu-light", "standard, full", "run_transfer", "Importance is predictive, not causal; correlated inputs share importance.", "portable_bundle"),
    Spec("J1", "J", "Integrated comparison and research gates", "n/a (synthesis)", "all saved tables", "all", "—", "—", "—", "gate status per criterion",
         "rules declared in synthesis.GATE_RULES", "tables/synthesis/*", "cpu-light", "all", "", "Missing evidence = 'not evaluated', never pass.",
         "portable_bundle"),
    Spec("X1", "—", "12A–12G warehouse studies", "Functional / latent-trait / distributional / OT studies of Milestone 12.", "research warehouse",
         "—", "—", "—", "—", "—", "—", "unified notebook outputs", "cpu-heavy", "—", "", "Run with M12_UNIFIED_COLAB_RESEARCH.ipynb when the warehouse is on Drive.",
         "requires_warehouse"),
)


def specs_frame() -> Any:
    import polars as pl

    return pl.DataFrame([asdict(s) for s in SPECS])


def status_table(run: Any) -> Any:
    """Specs joined with the stage ledger (``manifests/stages.json``) of ``run``; missing stage record → pending."""
    import polars as pl

    st = run.stage_records() if run is not None else {}
    rows = []
    for s in SPECS:
        rec = st.get(s.id, {})
        status = STATUS_MAP.get(rec.get("status"), "pending") if rec else ("blocked" if s.feasibility == "requires_warehouse" else "pending")
        rows.append({"id": s.id, "part": s.part, "title": s.title, "status": status, "compute": s.compute, "modes": s.modes, "flag": s.flag,
                     "feasibility": s.feasibility, "started_at": rec.get("started_at"), "finished_at": rec.get("finished_at"), "seconds": rec.get("seconds"),
                     "note": rec.get("note"), "error": rec.get("error")})
    return pl.DataFrame(rows, infer_schema_length=None)


def matrix_markdown(status: Any | None = None) -> str:
    """The experiment matrix as Markdown (spec columns; plus status when given)."""
    st = {} if status is None else dict(zip(status["id"].to_list(), status["status"].to_list()))
    cols = ["id", "title", "hypothesis", "representation", "training_regime", "held_out_protocol", "baseline", "metrics", "statistics", "compute",
            "modes", "flag", "interpretation_rule", "feasibility"]
    head = cols + (["status"] if st else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in SPECS:
        d = asdict(s)
        vals = [str(d[c]).replace("|", "/") for c in cols] + ([st.get(s.id, "pending")] if st else [])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


__all__ = ["SPECS", "STATUS_MAP", "Spec", "matrix_markdown", "specs_frame", "status_table"]
