# Football-Context Transfer Diagnosis and Athletic Fingerprint Scorecard (Milestone 07)

**Status: TEMPLATE — no Milestone 07 result has been produced.** Notebooks 07A–07D were authored on 2026-10-13 and have
not been executed. Every `‹…›` is a placeholder to be filled from the persisted table named beside it, after the FULL
run. Smoke-run numbers are wiring checks and never go here. Section 2 is the only section holding results: they are the
executed Milestone 06 numbers, copied from its saved tables.

Tables: `reports/tables/advanced_movement/{context_transfer,scoring}/` · datasets:
`warehouse/gold/advanced_movement/{context_transfer,scoring}/` · figures:
`reports/figures/advanced_movement/{context_transfer,scoring}/`.

Three statements this report keeps apart: a characteristic is **measurable**; it is **repeatable**; it **predicts**
football-relevant outcomes. The first two are established for selected Combine characteristics. The third is not.

---

## 1. Executive summary

‹Five to eight sentences, written last, each traceable to a table below: what the frozen NFL target is made of (07A);
whether any physically aligned comparison was established and its status as exploratory (07B); what effect size the
linked cohort could have detected and where the Milestone 06 gain sits against it (07C); what the scorecard shows and
what it does not claim (07D); the decision of section 14.›

## 2. Milestone 06 — negative / inconclusive transfer findings (executed, unchanged)

Source: `transfer/transfer_paired_comparisons.csv`, `transfer_permutation.csv`, `transfer_matched_pair_summary.csv`,
`scoring_readiness_registry.csv`, `transfer_robustness_conclusions.md`.

- Primary frozen target `O2.accel_sust05_mean` (yd/s², frozen with a context-cell residual); paired cohort 284 athletes
  (DB, DL, OL, TE, WR); folds, models and comparisons declared before any label was joined.
- Official Combine model M1: out-of-fold RMSE 0.1794, R² 0.947 — almost all of it position (POS alone: RMSE 0.1815).
- TCN-B (M5) vs M1: Δ RMSE +0.0003 (primary analysis; +0.0002 to +0.0003 across aggregation rules), about 0.16 % of
  the official error, 95 % paired athlete-bootstrap interval [−0.001, +0.002]; outcome-permutation p 0.169,
  linkage-permutation p 0.234.
- Speed development (M2), tracked top speed (M3), FPCA (M4), random encoder (M6), compact TCN (M7): every interval
  includes 0. No feature family has an established incremental gain.
- Matched prospects (833 pairs, 275 athletes): no model orders NFL movement better than the official model
  (M5 concordance 0.496 [0.425, 0.568]).
- Scoring-readiness registry: no `predictive_validated`, no `predictive_candidate`. Decision-tree case **E** — report
  the negative finding, keep Combine profiles as descriptions, build no Athletic Transfer Score.

This is "not established", not "no relationship exists". Sections 3–10 ask why, and whether the study could have seen one.

## 3. NFL context decomposition (07A)

Source: `nfl_context_distribution.csv`, `nfl_context_variance_decomposition.csv`, `nfl_context_scale_audit.csv`.

- Play measure, unit, plausibility: ‹share of implausible windows; floor / ceiling shares›
- Distribution by position / alignment / cohort / season / exposure: ‹reading of the tables and of AM07A_01, 06, 07›
- Position share of all window variance: ‹share_total, component = position›

## 4. Athlete-versus-context variance (07A)

| Component (within position) | Share | 95 % athlete bootstrap | Truncated? |
|---|---|---|---|
| athlete | ‹…› | ‹…› | ‹…› |
| game within athlete | ‹…› | ‹…› | ‹…› |
| window within game | ‹…› | ‹…› | — |

- Marginal context shares (η², not additive): alignment ‹…›, alignment × play class ‹…›, starting-speed band ‹…›, season ‹…›, down ‹…›
- Scale: between-athlete SD ‹…› yd/s² against a standard error of ‹…› yd/s² at the median exposure (ratio ‹…›)
- Model limitations to state: ‹truncated components; C4 backfitting convergence›

## 5. Outcome reliability and coverage (07A)

Source: `nfl_context_overlap.csv`, `nfl_context_outcome_reliability.csv`, `nfl_context_coverage_tradeoff.csv`,
`nfl_context_target_registry.json` (exploratory).

| Definition | Athletes | Kept vs C0 | Positions | Reliability (cross-fitted) [95 %] | ρ with C0 | Feasible |
|---|---|---|---|---|---|---|
| C0 raw (reference) | ‹…› | 1.00 | ‹…› | ‹…› | 1.00 | — |
| C1 position-standardised | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| C2 starting-speed-conditioned | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| C3 position × alignment | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| C4 hierarchical | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |

- First N games / windows against later games: ‹…›
- Gate 07A: ‹PASS / FAIL›; definitions meeting the reference lines: ‹…›

## 6. Physically aligned transfer results (07B — exploratory)

Source: `aligned_transfer_hypotheses.csv` (written before evaluation), `aligned_transfer_outcome_coverage.csv`,
`aligned_transfer_paired_uncertainty.csv`, `aligned_transfer_unit_audit.csv`, `aligned_transfer_interpretation.md`.

| Hypothesis (primary) | Outcome | Model vs reference | n | Δ RMSE [95 %] | Relative | Holm p | Reading |
|---|---|---|---|---|---|---|---|
| H1 | `C0:O1.speed_p90_mean` | M1 vs M0 | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| H2 | `C0:O3.band_acc_v4_6_mean` | M2 vs M1 | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| H3 | `C4:O2.accel_sust05_mean` | M5 vs M1 | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |

- Secondary comparisons: ‹from the table; none is promoted›
- H4 by position: ‹estimates and intervals; heterogeneity reading; no "best position" claim›
- Unit audit: ‹passed checks; stated definitional mismatches›

## 7. NFL speed positive-control results (07B, H1)

‹Does the pipeline detect a conventional relationship (Combine speed → NFL speed)? M1 vs M0, S0 vs M0, M3 vs M1 on NFL
p90 speed; within-position partial correlations of `aligned_transfer_physical_association.csv`; AM07B_01. If the
positive control is not established either, say what that implies for reading H2 and H3.›

## 8. TCN incremental value (07B, H3)

‹M5 vs M1, M2, M3, M4, M6 on the context-conditioned outcome; AM07B_10. Is the fingerprint associated with physical NFL
movement after official information, or primarily useful for identifying athletes (repeat-attempt AUC 0.873)?›

## 9. Matched-stopwatch results (07B)

Source: `aligned_transfer_matched_pairs.csv` — ‹concordance and difference from the official model per outcome, pairs,
distinct athletes, pair reuse›.

## 10. Detectability and power (07C)

Source: `transfer_power_scenarios.csv`, `transfer_null_calibration.csv`, `transfer_mde_estimates.csv`,
`transfer_information_limits.csv`, `transfer_power_report.md`. Simulation hash ‹…›, ‹…› replicates, complete: ‹…›.

- False-positive rate with no planted effect: ‹…› [‹…›, ‹…›] (nominal 0.025)
- Interval coverage: ‹…›
- Minimum detectable effect at 80 % power (TCN-B, conditional): f = ‹…›, Δ RMSE ‹…› (‹…› % of the official error)
- The Milestone 06 gain (+0.0003) against it: ‹smaller / comparable / larger; in null SDs ‹…››
- Share of the TCN-B direction already explained by official tests: ‹…›
- By position: ‹…› · linked-cohort scenarios: ‹athletes needed for f = ‹…››
- Attenuation (classical approximation, not a ceiling): ‹factor for the measured reliabilities›

## 11. Scoring methodology (07D)

Source: `combine_score_reference_manifest.json`, `combine_speed_percentiles.csv`,
`combine_speed_development_percentiles.csv`, `combine_measurement_confidence.csv`.

Fixed by design (not results): four sections never combined; oriented mid-rank percentiles within position;
leave-one-out for reference members and fold-isolated scoring as the generalisation check; intervals from repeat-attempt
measurement error and reference sampling (never percentile × ICC); a missing or unsupported measurement abstains; one
headline for Speed Capacity (official forty); the fingerprint is similarity only; evidence confidence is about the
recording; no NFL input.

To fill: reference sizes by position ‹…›; availability of speed development by position ‹…›; leave-one-out against
fold-isolated percentiles ‹mean and max difference›; cohort sensitivity ‹…›; evidence tiers ‹counts›; fingerprint repeat
stability ‹mean self-AUC›, seed overlap ‹…› against the untrained encoder ‹…›; gate 07D ‹…›.

## 12. Example player profiles (07D)

‹Cards AM07D_01_* and the cases of `combine_scorecard_examples.csv`, each with the Combine-side rule that selected it.
No case was selected with an NFL outcome; a case is an illustration, not evidence.›

## 13. Scientific limitations

- NFL labels are observed movement shaped by role, scheme, opponent and usage; context definitions remove observed
  labels only, and role is partly chosen because of ability.
- Milestone 07B comparisons were declared after the Milestone 06 result was known: exploratory whatever they show.
- Intervals cover athlete sampling, not model refitting (07C quantifies what that omits).
- Power is conditional on a linear, one-dimensional planted signal — an upper reading.
- Three draft cohorts, different sensor generations and NFL follow-up lengths; TE is small in every analysis.
- Variance components are method-of-moments estimates of an unbalanced nested design.
- ‹limitations found during execution›

## 14. Final decision

‹One of: (a) a physically aligned relationship is a credible hypothesis for an independent cohort — still not a
predictive score; (b) inconclusive: the cohort cannot distinguish practically relevant effects from none; (c) negative
within the detectable range: effects of at least ‹…› are unlikely. In every case: the scorecard stays descriptive.›

## 15. Recommended competition narrative

‹To write from the decision. Skeleton: what tracking makes measurable → what is repeatable (and what is only the
stopwatch again) → the pre-registered transfer test and its honest null → what the NFL target is made of → what this
cohort could and could not have detected → a descriptive, uncertainty-aware scouting profile that says what it is not.›
