"""Phase 05 — frontier research: capability model, counterfactuals, offline-RL feasibility, decision metrics.

Stage P0 (data & decision feasibility audit) — authored 2026-10-11, NOT executed:

    decision_audit        Stage P0  census of observable states / actions / outcomes / interactions and the
                                    rule-based feasibility verdict per frontier stream (no Combine feature)
    audit_figures                   Stage P0 figures P0_01…P0_05 (drawn from persisted tables only)
    sample_frame          Stage P0b analysis sample frame: Combine drill ∩ NFL exposure, cohort, outcome volume,
                                    role-relevant pairs by season, play type (participation counts only)
    config                          parameters with provenance (established / proposed / execution)

Later stages (S2 envelope … S8 dashboard) are gated: see ``reports/PHASE_05_FRONTIER_RESEARCH_PLAN.md`` §5 and
``reports/PHASE_05_EXPERIMENT_REGISTRY.md``. No Combine-to-NFL statistic is computed before the Phase 04 gates.

Persistence (``StudyStore(phase="phase_05")``): ``reports/tables/phase_05/<area>/``,
``reports/figures/phase_05/<study dir>/`` and ``warehouse/gold/phase_05/<area>/``. Bounded runs (``--seasons``)
nest under ``…/phase_05/_sample/``. Earlier phases are read, never written.
"""
