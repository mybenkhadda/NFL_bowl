"""``nflbowl.lab.m16`` — Milestone 16: verified baselines & cross-domain validation.

Notebook: ``notebooks/nflbowl/M16_VERIFIED_BASELINES_CROSS_DOMAIN_VALIDATION_COLAB.ipynb``.
Specification: ``docs/research/M15_M18_*.md`` (experiment IDs E16.A0–E16.B3, gate G16).

Implemented (task T04; authored, not executed):

    config            M16Config / resolve / M16Run (layout, read-only guards, fingerprint, status ledger, hand-off)
    targets           target registry (T0 nfl.q90_shared official · T0x nfl_speed_q90_median exploratory), fingerprints,
                      the different-target comparison guard
    m1_verification   M1 contract audit, recomputation of the official ladder, internal + persisted reproducibility at
                      1e-8, fold / target checks, athlete-level diffs, G16-B rows, M1 audit report, exploratory-target audit
    plots             reproduction_diffs, metrics_by_fold (figure + source data)

Planned next (T06/T09/T10): NFL task targets T1–T6, target reliability, baseline learner ladder, registry, retrieval.
"""

__all__ = ["config", "m1_verification", "plots", "targets"]
