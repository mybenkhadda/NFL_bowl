"""``nflbowl.lab.m17`` — Milestone 17: context-aware athletic capability learning.

Notebook: ``notebooks/nflbowl/M17_CONTEXT_AWARE_ATHLETIC_SIGNATURES_COLAB.ipynb``. Spec: ``docs/research/M15_M18_*.md`` (E17.A1–E17.E1, G17).
Authored, not executed.

    config        M17Config / resolve / M17Run
    events        E17.A1 failure modes, per-drill vs pooled ICC, one-factor threshold grid (M14 detector reused), diagnosis
    contexts      E17.A2 a-priori strata (entry speed × turn class, deceleration demand) and cross-domain coverage
    censoring     E17.A3 Kaplan–Meier recovery time, censoring-aware vs naive reliability
    hierarchical  E17.B1/B2 fold-safe context adjustment, MixedLM / method-of-moments variance components, BLUPs, Combine↔NFL r
    fusion        E17.C1 concat / reliability-weighted fusion of G15-eligible representations (late stacking → M18; attention NOT_RUN)
    alignment     E17.D1 Procrustes, CORAL, entropic OT, context-matched InfoNCE (opt-in), MMD², damage check, adoption
    signatures    E17.E1 + G17 criteria a–f, status, labels, validated signature columns for M18 B7
    pipeline      notebook orchestration; plots: figures with source data
"""

__all__ = ["alignment", "censoring", "config", "contexts", "events", "fusion", "hierarchical", "pipeline", "plots", "signatures"]
