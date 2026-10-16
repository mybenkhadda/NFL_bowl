# Advanced movement — Feature & Characteristic Atlas and robust TCN validation (Milestone 04)

**Status: TEMPLATE — NOT EXECUTED.** Authored 2026-10-12. Notebooks `04a_feature_characteristic_atlas.ipynb` and
`04b_tcn_robustness_and_shortlist.ipynb` have not been run. Every `‹…›` below is a value to copy from a persisted
table of the FULL run; nothing in this file is a result yet. Smoke-run values must never be copied here.

How to fill it: each section names the table (CSV under `reports/tables/advanced_movement/atlas/`, Parquet under
`warehouse/gold/advanced_movement/atlas/`) and the columns to read. The two generated texts
`atlas_conclusions.md` and `robust_conclusions.md` in the tables folder already contain most sentences.

---

## 1. Research objective

Which reproducible movement characteristics exist beyond traditional Combine testing, and which are credible enough
to test against NFL performance?

Six questions: RQ1 measurement inventory · RQ2 repeatability · RQ3 stopwatch redundancy · RQ4 acquisition artifacts ·
RQ5 robustness of the learned embedding · RQ6 transfer readiness. No NFL outcome is read in this milestone and no
candidate is selected with one.

Run identification: run mode ‹full›, seed ‹…›, atlas manifest `atlas_manifest.json` written ‹…›, robust manifest
`robust_manifest.json` written ‹…›, shortlist configuration hash ‹…›.

## 2. Existing results reused

Source: inventory table of notebook 04A, section 1 (`reuse_inventory`).

| Family | Persisted output reused | Present | New in this milestone |
|---|---|---|---|
| F1 official Combine | `gold.mart_athlete_research_base` | ‹…› | nothing |
| F2 / F4 / F6 forty attempt features | `sprint_rescue.attempt_censoring_diagnostics` | ‹…› | nothing |
| F2 other drills, F3 events | Phase 04 `V_feature_validation_registry` | ‹…› | nothing (evidence only) |
| F5 functional PCA | `fpca_component_gate`, `fpca_athlete_scores` | ‹…› | nothing |
| F5 encoders | `repr_*`, `tcn_*_attempt_embeddings`, checkpoints | ‹…› | two seed replicates of TCN-B (04B) |
| F6 sustained acceleration | speed-band and best-window features of the sprint rescue | ‹…› | four speed-anchored features (`atlas_sustained`) |
| F7 acquisition | `sequence_window_flags` | ‹…› | nothing |

Not reused / not rebuilt: the raw tracking warehouse, the frame kinematics, the sprint fitter (no second fitter was written).

## 3. Feature dictionary

Source: `feature_dictionary.csv` (key `feature_id`).

- Definitions: ‹…›, of which ‹…› characterised on the forty or as official measurements and ‹…› from other drills (Phase 04 evidence).
- By family (count): F1 ‹…› · F2 ‹…› · F3 ‹…› · F4 ‹…› · F5 ‹…› · F6 ‹…› · F7 ‹…›.
- Duplicate keys: ‹0 expected — the notebook stops otherwise›.
- Units of official measurements were taken from the Combine results file and not re-verified: implausible shares ‹…› (table `atlas_characteristics`, column `implausible_share`).

## 4. Coverage

Source: `atlas_coverage.csv` (`group_type = all`), figure `AM04A_02`.

- Denominator: ‹…› tracked athletes (≥ 1 eligible forty sequence), ‹…› eligible attempts, ‹…› athletes with a repeat.
- Official measurements with coverage below the reference (‹0.80›): ‹…›.
- New sustained features — attempts measured / missing by reason (`atlas_sustained`): ‹feature: measured …, start_above_anchor …, window_exceeds_recording …, upper_threshold_not_reached …›.
- Athletes kept by the quality filters (`atlas_quality_retention`): S0 ‹…› → S2 ‹…› → S3 ‹…› with a repeat.

## 5. Reliability

Source: `atlas_characteristics.csv` (`scope = all`), `atlas_reliability.csv`, figures `AM04A_04`, `AM04A_08`, `AM04A_05`.

| Feature | Athletes with a repeat | ICC within position [95 %] | SEM (unit) | Rank agreement | ICC range over cohorts | ICC under S3 |
|---|---|---|---|---|---|---|
| ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |

- Candidates with a reliability interval above 0.75: ‹…›; above 0.50: ‹…›; below 0.50: ‹…›.
- Position-level pattern (`scope = position:*`): ‹…›.
- Whole-vector representations (trace reliability, reused from notebook 03): ‹…›.

## 6. Stopwatch redundancy

Source: `atlas_characteristics.csv`, figures `AM04A_03`, `AM04A_03b`.

| Feature | ρ forty | ρ 10-yard | Stopwatch R² [95 %] | Cohort-adjusted R² | Body R² | Reliable share not explained | Label |
|---|---|---|---|---|---|---|---|
| ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |

- `beyond_stopwatch`: ‹…› · `redundant`: ‹…› · `low_correlation_only` (not evidence of new information): ‹…›.
- Feature groups (`atlas_feature_clusters`, figure `AM04A_07`): ‹number of groups, representatives›.

## 7. Acquisition sensitivity

Source: `atlas_characteristics.csv` (`window_r2*`, `cohort_r2`), `atlas_nuisance_diagnostics.csv`, figure `AM04A_06`.

- Candidates whose window interval lies above 0.25: ‹…›.
- Candidates with a cohort share ≥ 0.25: ‹…›.
- Strongest single-variable within-athlete associations: ‹…›.

Screening outcome (`atlas_feature_status.csv`, figures `AM04A_01`, `AM04A_10`): Strong ‹…› · Provisional ‹…› · Exploratory ‹…› · Reject / not currently usable ‹…›.

## 8. TCN seed robustness

Source: `robust_seed_runs.csv`, `robust_seed_summary.csv`, `robust_seed_geometry.csv`, figure `AM04B_01`.

- Encoder configuration source: ‹read from the checkpoints of notebook 03 / reconstructed›. Primary seed reused from notebook 03: ‹yes / no›.

| Metric | Seed 1 | Seed 2 | Seed 3 | Mean | SD between seeds | Athlete SE |
|---|---|---|---|---|---|---|
| repeat-attempt AUC (confirm) | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| matched-stopwatch AUC | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| trace reliability | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| effective rank / dimension | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | n/a |
| cross-fitted residual AUC | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| window share | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| cohort balanced-accuracy gain | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |

- Seeds flagged collapsed: ‹…› of ‹…›. Geometry agreement between seeds (CKA, distance rank correlation, neighbour overlap): ‹…›.

## 9. Matched-athlete retrieval

Source: `robust_matched_summary.csv` (`subset = role:confirm`), `robust_matched_paired.csv`, `robust_matched_pairs.parquet`,
figures `AM04B_02`, `AM04B_07`, `AM04B_09`.

- Matching rule: ‹k_nearest›, caliper ‹…› SD, variables ‹…›; scales from ‹…› design-role athletes.
- Eligible athletes ‹…›, unmatched ‹…›, comparisons ‹…›, distinct impostors ‹…›, largest reuse ‹…›; median |Δ forty| ‹…› s.

| Representation | Matched AUC [two-way 95 %] | Position AUC | Matching penalty [95 %] |
|---|---|---|---|
| TCN-B (primary seed) | ‹…› | ‹…› | ‹…› |
| TCN-A | ‹…› | ‹…› | ‹…› |
| FPCA benchmark | ‹…› | ‹…› | ‹…› |
| tracked top speed | ‹…› | ‹…› | ‹…› |
| untrained encoder | ‹…› | ‹…› | ‹…› |

- Paired differences, TCN-B − reference: FPCA ‹…› · TCN-A ‹…› · untrained ‹…› · top speed ‹…›.
- Official results as a representation: not evaluable by repeat-attempt retrieval (one value per athlete; AUC 1 by construction).

Residualisation audit (`robust_residualisation_audit.csv`, figure `AM04B_03`): raw ‹…› · cross-fitted ‹…› · cross-fitted with permuted
official results ‹…› · fitted on training athletes ‹…› · in sample ‹…›. Reading: ‹copy the rule-based reading; state whether the
notebook-03 increase is a cross-fitting artifact, a property of the metric, or absent›.

## 10. Effective-dimensionality analysis

Source: `robust_dimension_stability.csv`, `robust_dims_paired_comparisons.csv`, figure `AM04B_04`.

| k | AUC | Matched AUC | Trace reliability | Effective rank | Forty explained | Window share |
|---|---|---|---|---|---|---|
| 2 | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| 4 | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| 8 | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |

- Compact projection by the pre-declared rule: ‹k or none›.

Acquisition and cohort (`robust_cohort_probe.csv`, `robust_truncation_stability.csv`, `robust_loco.csv`, figures `AM04B_05`, `AM04B_06`):
cohort balanced accuracy ‹…› against position alone ‹…› (chance ‹…›, majority share ‹…›); largest truncation shift ‹…›;
leave-one-cohort-out AUC by cohort ‹…› against within-fold ‹…›.

## 11. Leading candidate characteristics

Source: `atlas_feature_status.csv`, `robust_representation_quality.csv`, figures `AM04A_10`, `AM04B_10`.

For each, state separately: exists · measurable · repeatable · distinct from the stopwatch · plausible physical interpretation ·
ready for an NFL transfer evaluation · demonstrated NFL predictive value (**untested**).

| Candidate | Measurable | Repeatable | Distinct from stopwatch | Interpretation | Ready for transfer test |
|---|---|---|---|---|---|
| ‹…› | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |

## 12. Known limitations

Fixed by design (true whatever the numbers):

- Two attempts per athlete, one drill, three cohorts with different sensor noise; intervals are wide in small position groups.
- The recording starts after the first movement and ends near peak speed; anchored features avoid the missing start, they do not recover it.
- Associations with official times are linear; a nonlinear dependence can survive residualisation and matching on four measurements.
- Three training seeds give a rough spread. Leave-one-cohort-out rests on small cohorts.
- Coordinates of an encoder are not comparable across folds or seeds; the embedding is one vector, not eight measurements.
- Screening lines (ICC 0.75 / 0.50, |ρ| 0.70, window 0.25) are heuristics. Grades are for review.
- Other-drill and event features carry Phase 04 evidence only.

To add after execution: ‹anything the run revealed — failed checks, unexpected missingness, unit problems›.

## 13. Frozen shortlist

Source: `transfer_candidates_v1.csv` (written once; hash and date in `transfer_candidates_v1_freeze.json`).

| Slot | Selected | Readiness | Features | Rationale |
|---|---|---|---|---|
| C1 traditional baseline | ‹…› | reference | ‹…› | ‹…› |
| C2 speed control | ‹…› | ‹…› | ‹…› | ‹…› |
| C3 acceleration | ‹…› | ‹…› | ‹…› | ‹…› |
| C4 sprint-curve parameter | ‹…› | ‹…› | ‹…› | ‹…› |
| C5 functional PCA | ‹…› | ‹…› | ‹…› | ‹…› |
| C6 TCN-B 8-d (primary seed) | ‹…› | ‹…› | ‹…› | ‹…› |
| C7 compact projection | ‹…› | ‹…› | ‹…› | ‹…› |

Exploratory registry (`transfer_exploratory_registry.csv`): ‹…› candidates, not part of the confirmatory comparison.

## 14. NFL transfer readiness

‹State, from section 13, which families are `ready_for_transfer_test`, which are carried with a caveat and why, which slots are empty.›

Readiness is Combine-side evidence that a family is worth testing. It is not evidence that the family predicts NFL
movement. **NFL predictive value: not tested in this milestone.**

## 15. Next decision

Decision recorded in notebook 04B section 16 (`robust_next_stage_decision.json`): ‹PROCEED / REVISE / STOP› — ‹rationale›.

Next milestone: **NFL movement outcome reliability** — show that the NFL target is repeatable for an athlete across games
before any Combine representation is compared with it. GRU / LSTM training and the transfer comparison start only after that.
