# Combine-to-NFL Athletic Transfer Validation

**Status: TEMPLATE — no result has been produced.** Notebooks 06A, 06B and 06C were authored on 2026-10-13 and have not
been executed. Every `‹…›` below is a placeholder to be filled from the saved table named beside it, after the full run.
Nothing in this document may be quoted as a finding until then. Smoke-run numbers are wiring checks and never go here.

Tables live under `reports/tables/advanced_movement/transfer/`, datasets under
`warehouse/gold/advanced_movement/transfer/`, figures under `reports/figures/advanced_movement/transfer/`.

---

## 1. Research motivation

When two prospects have similar official Combine results, does high-frequency Combine tracking reveal a repeatable
movement characteristic that helps predict how they will move during NFL games? The aim is a reproducible answer —
positive or negative — that a scouting scoring system could be built on, not the best possible prediction score.

## 2. Prior Combine discoveries (executed, Milestones 01–04)

Read from the saved tables of those milestones; none of them is NFL evidence.

- 793 eligible tracked forty attempts of 418 athletes, 375 with a repeat (notebook 00).
- Functional PCA of speed by distance: K = 2 reconstructs ‹fve› of held-out variance; the dominant, reliable component is
  largely the official forty (notebook 02).
- TCN-B (speed by distance, 8-d): repeat-attempt AUC 0.873 [0.856, 0.890], 0.872 ± 0.001 over three seeds;
  matched-stopwatch AUC 0.796; trace reliability 0.925; official forty explained beyond position 0.89 (notebooks 03, 04B).
- Frozen shortlist `transfer_candidates_v1`: speed control `F2.forty.p95_speed`, acceleration candidate
  `F6.forty.time_v4_to_v8_s`, FPCA K2, TCN-B 8-d, compact projection on 4 axes; the sprint-parameter slot is empty.

## 3. NFL target validation (Milestone 05)

- 401 NFL-tracked athletes, 242,043 eligible player-plays, 21 candidate outcomes; 14 met all five screening lines.
- Rule-based proposal of notebook 05B: primary `O2.accel_sust05_mean`, secondary `O5.envelope_level`,
  `O3.band_acc_v0_2_mean`.
- **Freeze recorded:** ‹decision, frozen_at, version_hash — `nfl_movement_targets_v1.json`›. At authoring time the
  recorded decision was PENDING: notebook 06A stops until it is recorded.
- Frozen targets, formulas, units, exposure requirement, context adjustment: ‹06A section 2 tables›.

## 4. Linked evaluation population

| Stage | Athletes |
|---|---|
| Combine roster | ‹transfer_cohort_funnel› |
| Eligible tracked forty | ‹…› |
| Official core (forty, 10-yard, weight) | ‹…› |
| Frozen engineered features | ‹…› |
| FPCA / TCN input | ‹…› |
| NFL regular-season tracking | ‹…› |
| Frozen exposure requirement | ‹…› |
| **Primary cohort (06A)** | ‹…› |
| **Paired cohort (06B, every representation in every fold)** | ‹transfer_common_support› |

- By position group and draft cohort: ‹transfer_position_cohort_counts›.
- Exclusion reasons: ‹transfer_eligibility_reasons›. Identity conflicts: ‹transfer_identity_audit›.
- Primary cohort against excluded tracked athletes (pre-NFL measurements): ‹06A section 8 table, figure AM06A_07›.
- Follow-up by draft cohort: ‹06A section 8›. Missing-data rules and the predictors of M0 / M1:
  ‹transfer_missingness_audit›.

## 5. Pre-registered hypotheses

Primary: a repeatable Combine movement representation contains information about realized NFL movement that official
Combine testing and pre-draft anthropometrics do not capture.

| | Hypothesis | Comparison | Status |
|---|---|---|---|
| H1 | interpretable speed-development measures predict NFL sustained acceleration beyond official testing | M2 vs M1 | secondary |
| H2 | TCN-B embeddings improve NFL acceleration prediction over official tests alone | **M5 vs M1, primary target** | **primary** |
| H3 | any TCN-B improvement remains after a simple speed representation | M1 + SPEED2 + TCN-B vs M3 | secondary |
| H4 | transfer differs by position and NFL context | subgroup scoring, heterogeneity test | exploratory |
| H5 | direction holds under exposure and measurement changes | declared sensitivity grids, cohort left out | secondary |
| H6 | among similarly timed prospects, learned differences predict NFL differences | matched pairs | secondary |

## 6. Feature families

| Model | Added to the previous block | Source |
|---|---|---|
| M0 | position (unpenalised), draft cohort, ‹body predictors› | registry, research mart |
| M1 | ‹official tests kept by the coverage rule›, missingness indicators | research mart |
| M2 | ‹frozen engineered ids› | atlas attempt features |
| M3 | SPEED2: p95 speed, interior peak speed | atlas attempt features |
| M4 | FPCA scores, K = ‹K›, refitted per fold | notebook 02 configuration |
| M5 | TCN-B 8-d, fold encoders | notebook 03 configuration |
| M6 | untrained encoder, same processing | — |
| M7 | TCN-B on 2 / 4 training axes (exploratory) | — |

## 7. Fold and leakage design

- Fold design retained: ‹fold_source, n_folds — transfer_fold_candidates›; balance: ‹figure AM06A_09›.
- Saved encoders: ‹n compatible / n descriptive-only — representation_checkpoint_compatibility›.
- Fold encoders in 06B: ‹reused / trained — transfer_encoder_fits_outer›; athletes shared between a fit and its held-out
  fold: ‹must be 0›.
- Per fold, from training athletes only: label context means, encoders, imputation, scaling, dummy levels, ridge penalty
  (inner athlete-grouped folds).
- Gates: 06A ‹PASS/FAIL›, 06B ‹…›, 06C ‹…›. Synthetic checks: ‹21 rows PASS›.

## 8. Official baseline results

Primary target ‹id›, ‹n› athletes (`transfer_model_performance`):

| Model | RMSE [95 %] | MAE | R² (oos) | Spearman | Calibration slope |
|---|---|---|---|---|---|
| training mean | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| position mean | ‹…› | | | | |
| M0 | ‹…› | | | | |
| **M1** | ‹…› | | | | |

M1 against M0: Δ RMSE ‹…› [‹…›, ‹…›]. Wide cohort (official results, tracking not required, ‹n› athletes): ‹…›.

## 9. Engineered movement results

M2 vs M1: Δ RMSE ‹…› [‹…›, ‹…›], ‹…› % — ‹established / not established / worse›.
In-sample association after official times: ‹transfer_physical_mechanism›.

## 10. FPCA results

M4 vs M1: Δ RMSE ‹…› [‹…›, ‹…›] — ‹…›.

## 11. TCN results

**M5 vs M1 (primary comparison): Δ RMSE ‹…› [‹…›, ‹…›], ‹…› % — ‹established / not established / worse›.**
By training seed: ‹transfer_paired_comparisons_seeds›. Compact projections (exploratory): ‹…›.

## 12. Simple and random controls

| Comparison | Δ RMSE [95 %] | Reading |
|---|---|---|
| M3 (SPEED2) vs M1 | ‹…› | |
| M5 vs M3 | ‹…› | |
| M1 + SPEED2 + TCN-B vs M3 | ‹…› | |
| M6 (random encoder) vs M1 | ‹…› | |
| M5 vs M6 | ‹…› | |
| M5 vs M2 / M5 vs M4 | ‹…› | |

Permutation checks (`transfer_permutation`): outcome p ‹…›, linkage p ‹…›; positive control M1 vs M0 p ‹…›.

## 13. Paired uncertainty

Paired bootstrap of athletes, ‹n_boot› replicates, on out-of-fold predictions. Interval stability across budgets and
seeds: ‹figure AM06C_09›. Fold variability: ‹transfer_fold_performance, figure AM06B_11›. The intervals do not cover the
refitting of models or encoders.

## 14. Position effects

‹transfer_subgroup_comparisons, group_type = position_group; figure AM06C_01›. Heterogeneity test p ‹…›.
Sample sizes per position: ‹…›. TE and other small groups: ‹n, interval or none›.

## 15. Cohort generalisation

Stratified folds by cohort: ‹…›. Leave-one-draft-cohort-out (encoders refitted): M5 vs M1 Δ RMSE ‹…› [‹…›, ‹…›].
Cohorts differ in sensor generation and NFL follow-up: a cohort difference is not an athlete-development effect.

## 16. Exposure and context sensitivity

‹transfer_paired_comparisons_sens; figures AM06C_03, AM06C_04›. Rules evaluated ‹k of K›; rules with a positive estimate
‹…›; with an established gain ‹…›. No rule was selected.

## 17. Matched-prospect results

‹n› pairs of ‹n› athletes, an athlete in up to ‹…› pairs (`transfer_matched_pair_summary`):

| Model | Concordance [95 %] | vs official-only [95 %] |
|---|---|---|
| M1 | ‹…› | — |
| M2 | ‹…› | ‹…› |
| M3 | ‹…› | ‹…› |
| M4 | ‹…› | ‹…› |
| M5 | ‹…› | ‹…› |

Case studies are illustrative only and are not reported as evidence.

## 18. Scientific limitations

- The labels are observed NFL movement: role, scheme and usage are stable too and are part of what is predicted.
- Context labels are alignment and play class only; assignment and opponent are unobserved.
- The forty recording starts after the first movement; engineered features describe the recorded part of the run.
- Draft cohorts differ in tracking noise and in follow-up; the linked cohort is not a random sample of the Combine
  (‹section 4 comparison›).
- A few hundred athletes: only ‹minimum detectable gain, from the M1 interval width› can be detected.
- Choices made on Combine data alone before this milestone (consistency term, K, shortlist) are transductive in the
  Combine data; no NFL value was involved.
- `player_get_off` definition, handedness of the Combine frame and the units of official measurements remain open from
  earlier milestones (not used here / not verified).
- ‹anything the run revealed›

## 19. Scoring-system readiness

`scoring_readiness_registry.csv`:

| Dimension | Status | Δ RMSE vs its reference [95 %] | Reason |
|---|---|---|---|
| official_combine_tests | ‹…› | ‹…› | ‹…› |
| tracked_top_speed | ‹…› | ‹…› | ‹…› |
| sustained_acceleration | ‹…› | ‹…› | ‹…› |
| fpca_speed_curve | ‹…› | ‹…› | ‹…› |
| tcn_b_movement_fingerprint | ‹…› | ‹…› | ‹…› |
| tcn_b_compact | ‹…› | ‹…› | ‹…› |

No athlete score was built in this milestone.

## 20. Next decision

Decision-tree case returned by notebook 06C: **‹A / B / C / D / E›** — ‹sentence›.

| Case | Meaning | Next step |
|---|---|---|
| A | TCN-B adds a meaningful NFL prediction gain | interpretability and calibration; validated predictive scorecard candidate |
| B | engineered movement helps, TCN-B does not | scorecards around reliable speed-development measures |
| C | SPEED2 matches TCN-B | prefer the simpler representation |
| D | gain only in certain positions | investigate, require replication, no universal claim |
| E | nothing improves on official testing | report the negative finding; descriptive Combine profiles only; no Athletic Transfer Score |

Conditions for a predictive score: ‹transfer_robustness_conclusions.md, "Conditions for the scoring milestone"›.
