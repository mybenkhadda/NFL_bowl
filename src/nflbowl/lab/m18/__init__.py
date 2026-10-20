"""``nflbowl.lab.m18`` — Milestone 18: NFL performance transfer and scouting value.

Notebook: ``notebooks/nflbowl/M18_NFL_SCOUTING_VALUE_COLAB.ipynb``. Spec: ``docs/research/M15_M18_*.md`` (E18.A1–E18.F1, G18).
Authored, not executed.

    config       M18Config / resolve / M18Run (immutable pre-registration)
    ladder       Study A: rung eligibility from gate tables, B0–B7 with the official engine and ``se.embedding_block``, evaluation
    position     Study B: pooled vs separate within-position models
    specificity  Study C: drill × task partial-Spearman matrix and diagonal contrast
    scouting     Study D: outcome-free composite, stopwatch twins, pooled contrast, pseudonymised case cards
    robustness   Study E: negative controls, random label, refit permutation null, subgroup sensitivity
    evidence     evidence registry and G18 verdicts; reporting: final report + executive summary from tables only
    pipeline     notebook orchestration; plots: publication figures with source data
"""

__all__ = ["config", "evidence", "ladder", "pipeline", "plots", "position", "reporting", "robustness", "scouting", "specificity"]
