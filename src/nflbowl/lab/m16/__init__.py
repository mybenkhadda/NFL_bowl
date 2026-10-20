"""``nflbowl.lab.m16`` — Milestone 16: verified baselines & cross-domain validation.

Notebook: ``notebooks/nflbowl/M16_VERIFIED_BASELINES_CROSS_DOMAIN_VALIDATION_COLAB.ipynb``.
Specification: ``docs/research/M15_M18_*.md`` (experiment IDs E16.A0–E16.B3, gate G16). Authored, not executed.

    config              M16Config / resolve / M16Run                                                                       [T04]
    m1_verification     M1 contract, official-ladder recomputation, internal + persisted reproducibility at 1e-8, G16-B   [T04]
    targets             target registry T0 / T0x / T1–T6, NFL window table, M14 detector on NFL rows, athlete targets      [T04, T06]
    target_reliability  split-half reliability (season / game / random), exposure, context share, G16-A, immutable freeze [T06]
    detectability       planted-signal power and minimum detectable effect on the real cohort structure                   [T06]
    baselines           MEAN · POS · M1 · M1+F0 · M1+simple · M1+M14_reliable (+ ENet / trees), evaluation, ranking        [T09]
    registry            unified five-fold evaluation registry + OOF long table                                            [T09]
    retrieval           confirmatory retrieval of G15-eligible representations, Holm vs F0, position-only test, CEM       [T10]
    leakage             leakage audit → combine_only_inductive / diagnostic_only                                           [T10]
    gates               G16 A–D from the saved tables                                                                      [T10]
    pipeline            notebook orchestration; plots: figures with source data
"""

__all__ = ["baselines", "config", "detectability", "gates", "leakage", "m1_verification", "pipeline", "plots", "registry", "retrieval", "target_reliability",
           "targets"]
