"""``nflbowl.lab`` — the M12 comprehensive experimental lab (notebook ``M12_COMPREHENSIVE_EXPERIMENTAL_LAB_COLAB.ipynb``).

Orchestration and analysis on top of ``nflbowl.Experiment`` and the unchanged research code:

    config        CFG dict → LabConfig (run modes pilot / standard / full; unknown keys rejected)
    store         LabRun: Drive layout, tables per family, figures (PNG 300 dpi / SVG / PDF + data), stage ledger
    registry      formal experiment matrix + status from the stage ledger
    environment   environment record, bundle validation, lab Experiment, precision lock, checkpoint adoption, training plan
    audit         Part B   dataset audit, domain composition, attrition, leakage audit
    geometry      Part C   geometry, collapse, domain geometry, projections
    retrieval_lab Part D/G retrieval metrics, paired comparisons, case studies, pool difficulty, quality, heterogeneity
    transfer_lab  Part E   OOF metrics, paired Δ RMSE (common / native cohorts), positions, errors, scouting metrics
    ablations     Part F   ablations, probes, training curves, complexity
    robustness    Part G   seeds, learning curves
    biomech       Part H   interpretable movement metrics, reliability, transfer, incremental value, hypotheses
    interpret     Part I   permutation importance, partial dependence, counterfactuals, athlete profiles
    synthesis     Part J/K gates, integrated comparison, final report
    figures       the visual identity shared by every section

Milestone 13 (notebook ``M13_NFL_OUTCOME_TRANSFER_RESEARCH_COLAB.ipynb``):

    m13_config        M13Config (resolve), M13Run (M13 layout, provenance, signature-checked cache)
    handoff           Part A   environment, read-only adoption of M12 checkpoints / embeddings, inventory, reconciliation, F1 validity
    target_audit      Part B   frozen-target provenance, attrition flow, overlap matrix, position structure
    transfer_fairness Part C/D ladder + POS / MEAN baselines, alignment validation, cohorts, metrics, paired bootstrap
    reliability       Part E   ICC audit, equal-observation curves, train-fold-only selection ladders, stability, interactions
    position_models   Part F   centred / standardised / position-specific / partially pooled models
    domain_matching   Part G   coarsened exact matching of frozen embeddings, domain AUC / MMD² / retrieval sensitivity
    scouting_value    Part H/I retrieval-vs-transfer records, help / hurt analysis, intervals, athlete profiles
    m13_reporting     Part J/K unified comparison, declared outcome rule, evidence matrix, caveats, reports
    m13_figures       navy / cyan theme + M13 figure types

Milestone 14 (notebook ``M14_CROSS_SPORT_ATHLETIC_INTELLIGENCE_COLAB.ipynb``): subpackage ``nflbowl.lab.m14`` — metric
registry, event detection, cross-sport metrics, reliability / novelty, Hidden Braking Advantage, transfer, reports.

Modules are imported explicitly (``from nflbowl.lab import audit``); importing the package loads nothing heavy.
"""

__all__ = ["ablations", "audit", "biomech", "config", "domain_matching", "environment", "figures", "geometry", "handoff", "interpret", "m13_config", "m13_figures", "m14",
           "m13_reporting", "position_models", "registry", "reliability", "retrieval_lab", "robustness", "scouting_value", "stats", "store", "synthesis",
           "target_audit", "transfer_fairness", "transfer_lab"]
