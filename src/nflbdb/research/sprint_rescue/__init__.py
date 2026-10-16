"""Sprint reliability rescue — "What does the stopwatch miss?" (one experiment, forty-yard dash only).

Question     Is there a Combine sprint measurement that is repeatable WITHIN position groups and carries
             information beyond the official forty / 10-yard split?
Inputs       persisted datasets only: Phase 03 sprint profiles, attempt features and frame kinematics, Phase 04
             smoothing features and quality flags, gold research base, Phase 05 population. Nothing is rebuilt.
Stages       ATT attempt table + sprint-model refits · A censoring association · B within-position reliability ·
             C model repeatability · D stopwatch redundancy + screening · E cohort reconciliation ·
             F NFL endpoint feasibility (no Combine-to-NFL statistic).
Outputs      reports/tables/sprint_rescue/*.csv · warehouse/gold/sprint_rescue/*.parquet (+ .schema.json) ·
             reports/figures/sprint_rescue/* · reports/SPRINT_RELIABILITY_RESCUE_RESULTS.md

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.sprint_rescue
"""
