"""Phase 04 — scientific validation → athlete feature store → statistical discovery → candidate selection.

Milestone 1 (scientific validation, the P0 gate) — authored 2026-10-11, NOT executed:

    quality_eligibility   Study Q   structural vs signal quality; athlete eligibility under quality filters
    smoothing             Study B   smoothing robustness (trajectory + athlete level, ICC per setting)
    smoothing_figures               Study B figures B_01…B_09
    events                Study C1  event-detector sensitivity (rule × smoothing, per-second rates, event matching)
    annotation_design     Study C2  v2 reference annotation (Combine + NFL, dev/eval split) and detector accuracy
    validation_registry   Study V   VALIDATED / PROVISIONAL / EXPERIMENTAL / REJECTED per feature × drill
    metrics                         pure agreement metrics (top-decile overlap, rank change, variance share, κ)
    config                          parameters with provenance (established / proposed / execution)
    report                          fills reports/PHASE_04_SCIENTIFIC_RESEARCH_REPORT.md from saved tables

Persistence (``StudyStore(phase="phase_04")``): ``reports/tables/phase_04/<area>/``,
``reports/figures/phase_04/<study dir>/`` and ``warehouse/gold/phase_04/<area>/``. Bounded runs
(``--sample-athletes``) nest under ``…/phase_04/_sample/``. Phase 03 outputs are read, never written.

Plan, reuse map and gates: ``docs/PHASE_04_PLAN.md``.
"""
