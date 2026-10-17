"""``nflbowl.lab.m14`` — Milestone 14: cross-sport movement intelligence and the Hidden Braking Advantage
(notebook ``M14_CROSS_SPORT_ATHLETIC_INTELLIGENCE_COLAB.ipynb``).

    config      M14Config (resolve), M14Run (M14 layout, provenance, signature-checked cache, stage status export)
    registry    Phase 1  formal metric specifications (origin sport, formula, channels, units, failure cases,
                         positions, reliability requirement, NFL hypothesis) + availability / blocking
    events      Phase 1  10 Hz signal segmentation and cut-event detection (approach → braking onset → turn →
                         minimum speed → propulsion onset → exit → recovery), per-event metrics
    metrics     Phase 2  sequence-level sprint / envelope / smoothness metrics, one-pass extraction driver,
                         sequence table, athlete aggregation, perturbation re-extraction
    reliability Phase 3  ICC(1,1) / ICC(1,k), split-half + Spearman–Brown, matched-k curves, perturbation
                         sensitivity, coverage, measurement status
    novelty     Phase 3  out-of-fold explainability by M1, redundancy clusters, train-fold-only residual blocks
    transfer    Phase 5  M14 ladder (same engine / frame / folds / seed as the official ladder), outcome-free
                         train-fold quality selection, cohorts, paired comparisons, positions, permutation null
    scouting    Phase 4/6 braking signature, stopwatch twins, twin contrasts, case studies, evidence synthesis
    reporting   white-theme HTML helpers, Markdown / HTML reports
    figures     M14 figure types on the shared light figure library

Nothing here trains a neural encoder. Every predictor is Combine-side; NFL-window metrics (``m14n__*``) are
outcome-side only.
"""

__all__ = ["config", "events", "figures", "metrics", "novelty", "registry", "reliability", "reporting", "scouting", "transfer"]
