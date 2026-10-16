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

Modules are imported explicitly (``from nflbowl.lab import audit``); importing the package loads nothing heavy.
"""

__all__ = ["ablations", "audit", "biomech", "config", "environment", "figures", "geometry", "interpret", "registry", "retrieval_lab", "robustness", "stats",
           "store", "synthesis", "transfer_lab"]
