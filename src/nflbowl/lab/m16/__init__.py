"""``nflbowl.lab.m16`` — Milestone 16: verified baselines & cross-domain validation.

Notebooks: ``notebooks/research/M16_VERIFIED_BASELINES_CROSS_DOMAIN_VALIDATION_COLAB.ipynb`` (laboratory, sections 00–11;
supersedes ``notebooks/nflbowl/M16_…`` which keeps the session-4 core steps). Specification: ``docs/research/M15_M18_*.md``
and ``docs/research/M16_EXPERIMENTS_AND_VALIDATION.md``. Authored, not executed.

Core (session 4)
    config              M16Config / resolve / M16Run                                                                       [T04]
    m1_verification     M1 contract, official-ladder recomputation, internal + persisted reproducibility at 1e-8, G16-B   [T04]
    targets             target registry T0 / T0x / T1–T6, NFL window table, M14 detector on NFL rows, athlete targets      [T04, T06]
    target_reliability  split-half reliability (season / game / random), exposure, context share, G16-A, immutable freeze [T06]
    detectability       planted-signal power and minimum detectable effect on the real cohort structure                   [T06]
    baselines           MEAN · POS · M1 · M1+F0 · M1+simple · M1+M14_reliable (+ ENet / trees), evaluation, ranking        [T09]
    registry            unified five-fold evaluation registry + OOF long table                                            [T09]
    retrieval           confirmatory retrieval of G15-eligible representations, Holm vs F0, position-only test, CEM       [T10]
    leakage             leakage audit → combine_only_inductive / diagnostic_only                                           [T10]
    gates               G16 A–D from the saved tables (+ ``g16_extended``: evidence-rich rows, five gate states)           [T10]

Laboratory (session 5)
    cohort              E16.A9  dual-domain reconciliation (399 observed vs 401 flagged), fold integrity, funnel, inventory
    target_registry     A2      typed, versioned target definitions + within-athlete bootstrap uncertainty
    target_audit        E16.A7  distributions, variance structure, exposure / context / cohort relations, sensitivity, selection
    context_matching    E16.A8  Combine-drill vs NFL common support (BLOCKED_INSUFFICIENT_OVERLAP) and context-matched outcomes
    representation_audit E16.B4 four-state provenance registry of every M15 artifact (+ G15 decision, E15.C evidence)
    controlled_retrieval E16.B5 controlled retrieval of every verified representation vs F0 / kinematic / position-only controls
    repeatability       E16.B6  alignment damage, effective rank, ICC, probes, M1 re-encoding, duration sensitivity
    features            E16.C0  fold-specific Combine-only athlete features (five aggregations, drill scopes, fusion)
    ladder              E16.C4–C5 ladder B0–B7, pooled learner, negative controls (shuffled / random / reduced)
    evaluation          C4      cohort audit, calibration, stratified errors, paired Spearman, improvement table
    positions           E16.C6  global vs separate vs partially pooled position models, sample-size readings
    hypotheses          E16-C1–C3 drill → NFL hypotheses with prerequisites
    profiles            descriptive forty speed profiles (outcome quartiles, stopwatch twins)
    reporting           evidence-only research report generator
    pipeline            notebook orchestration (one guarded ``run_*`` per step; signature-validated caches)
    plots               figures with source data (PNG / SVG / PDF + CSV through ``M16Run.figure``)
"""

__all__ = ["baselines", "cohort", "config", "context_matching", "controlled_retrieval", "detectability", "evaluation", "features", "gates", "hypotheses", "ladder",
           "leakage", "m1_verification", "pipeline", "plots", "positions", "profiles", "registry", "repeatability", "reporting", "representation_audit",
           "retrieval", "target_audit", "target_registry", "target_reliability", "targets"]
