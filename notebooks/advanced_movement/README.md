# Advanced movement representation learning — notebook guide

Status (2026-10-13): **Milestones 1–3 (notebooks 00–03) executed by the user, smoke and full. Milestone 4
(notebooks 04A and 04B) and Milestone 5 (notebooks 05A and 05B) executed by the user; the 05B freeze decision is still
`PENDING`. Milestone 6 (notebooks 06A, 06B, 06C: Combine → NFL transfer validation) AUTHORED 2026-10-13 — AWAITING USER
EXECUTION.** Nothing here has been run by its author.

Every experiment is run from these notebooks. No terminal command is needed. Reusable code lives in
`src/nflbdb/research/advanced_movement/`; the notebooks configure, run, plot, interpret and save.

## Before you start

- Kernel: the project environment (`nfl`), with `nflbdb` installed in editable mode as for the earlier notebooks.
- PyTorch is not needed for notebooks 00–02 and is REQUIRED from notebook 03. Notebook 00 reports whether it is
  installed; nothing is installed for you.
- Earlier executed work the notebooks read: Phase 03 frame kinematics, the sprint reliability rescue, the DL get-off study.
  Notebook 00 lists any dataset that is missing.

## Order

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 00 | `00_research_setup_and_data_audit.ipynb` | warehouse (read only) | `schema_inventory`, `sequence_manifest`, `athlete_registry`, `dataset_audit`, `missingness_matrix`, `split_registry`, `leakage_audit`, `research_metadata.json`, figures `AM00_01`–`AM00_07` |
| 01 | `01_sequence_eda_and_alignment.ipynb` | outputs of 00, frame kinematics | `sequence_frames`, `sequence_summary`, `alignment_comparison`, `preprocessing_decisions`, figures `AM01_01`–`AM01_12` |
| 02 | `02_functional_pca_baseline.ipynb` | outputs of 00 and 01 | `sequence_window_flags`, `preprocessing_audit`, `preprocessing_decisions_final`, 16 `fpca_*` tables and datasets, `fpca_config_and_basis.json`, `fpca_experiment_manifest.json`, model files `fpca/*.npz`, figures `AM02_01`–`AM02_12` |
| 03 | `03_self_supervised_tcn.ipynb` | outputs of 00, 01 and 02 (tables and the FPCA fold models) | `fpca_ref_*`, `tcn_a_*`, `tcn_b_*`, `tcn_e16_*` embeddings and reconstruction sums, training histories and fold fits, `tcn_design_sweep`, 13 `repr_*` tables and datasets, `tcn_experiment_manifest.json`, checkpoints `tcn/*.pt`, figures `AM03_01`–`AM03_12` |
| 04A | `04a_feature_characteristic_atlas.ipynb` | outputs of 00–03, sprint rescue attempt table, research mart, Phase 04 registry (all read only) | under `…/advanced_movement/atlas/`: `feature_dictionary`, `atlas_sustained`, `atlas_coverage`, `atlas_characteristics`, `atlas_reliability`, `atlas_nuisance_diagnostics`, `atlas_attempt_features`, `atlas_athlete_features`, `atlas_feature_status`, `atlas_quality_retention`, `atlas_redundancy_matrix`, `atlas_feature_clusters`, `atlas_manifest.json`, `atlas_conclusions.md`, figures `AM04A_01`–`AM04A_10` |
| 04B | `04b_tcn_robustness_and_shortlist.ipynb` | outputs of 00–03 and 04A, checkpoints of 03 | under `…/atlas/`: `tcn_b_s<i>_*` (seed replicates), `rand_*`, 13 `robust_*` evaluation tables, `robust_matched_*`, `robust_residualisation_audit`, `robust_cohort_probe`, `robust_seed_*`, `robust_dimension_stability`, `robust_loco`, `robust_representation_quality`, `transfer_candidates_v1` (frozen), `transfer_exploratory_registry`, `robust_manifest.json`, checkpoints `atlas/tcn/*.pt`, figures `AM04B_01`–`AM04B_10` |
| 05A | `05a_nfl_movement_outcome_eda.ipynb` | `silver.game_tracking`, Phase 03 `nfl_play_anchors` / `nfl_player_play_windows` / `nfl_trajectory_quality`, `dim_athlete`, silver `games` / `player_play` / `plays` (all read only; **no Combine table**) | under `…/advanced_movement/nfl_outcomes/`: `nfl_play_movement`, `nfl_play_movement_variants`, `nfl_phase_space`, `nfl_window_funnel`, `nfl_outcome_dictionary`, `nfl_athlete_outcomes`, `nfl_athlete_game_outcomes`, `nfl_outcome_coverage`, `nfl_play_measure_summary.csv`, `nfl_play_variant_agreement.csv`, `nfl_exclusion_reasons.csv`, `nfl_outcome_manifest.json`, figures `AM05A_01`–`AM05A_12` |
| 05B | `05b_nfl_outcome_reliability_and_freeze.ipynb` | outputs of 05A (same `RUN_MODE`) | under `…/nfl_outcomes/`: `nfl_outcome_reliability`, `nfl_reliability_vs_exposure`, `nfl_outcome_context_audit`, `nfl_outcome_quality_sensitivity`, `nfl_outcome_redundancy`, `nfl_outcome_screen`, `nfl_movement_targets_v1.json` (frozen once `DECISION` is set), `nfl_reliability_manifest.json`, `nfl_outcome_conclusions.md`, figures `AM05B_01`–`AM05B_10` |
| 06A | `06a_transfer_cohort_and_leakage_audit.ipynb` | outputs of 00–05B (same `RUN_MODE`), saved encoders (opened, not trained), research mart | under `…/advanced_movement/transfer/`: `transfer_identity_audit`, `transfer_target_eligibility`, `transfer_cohort_manifest`, `transfer_cohort_funnel`, `transfer_position_cohort_counts`, `transfer_eligibility_reasons`, `transfer_missingness_audit`, `transfer_outer_folds` (+ CSV), `transfer_fold_candidates`, `representation_checkpoint_compatibility`, `transfer_gate`, `transfer_input_contract.json`, `transfer_cohort_audit_manifest.json`, figures `AM06A_01`–`AM06A_10` |
| 06B | `06b_combine_to_nfl_transfer_benchmark.ipynb` | outputs of 06A, curves, atlas attempt features, `nfl_play_movement` | `transfer_attempt_representations_outer`, `transfer_encoder_fits_outer`, fold encoders `transfer/tcn/*.pt`, `transfer_common_support`, `transfer_oof_predictions`, `transfer_oof_targets`, `transfer_fold_preprocessing`, `transfer_model_performance`, `transfer_paired_comparisons`, `transfer_fold_performance`, `transfer_gate_06b`, `transfer_benchmark_conclusions.md`, `transfer_benchmark_manifest.json`, figures `AM06B_01`–`AM06B_12` |
| 06C | `06c_transfer_robustness_and_scouting_insights.ipynb` | outputs of 06A and 06B | `transfer_subgroup_*`, `transfer_gain_heterogeneity`, `transfer_*_loco`, `transfer_*_sens`, `transfer_*_seeds`, `transfer_matched_pair*`, `transfer_physical_mechanism`, `transfer_permutation`, `transfer_ci_stability`, `transfer_evidence_summary`, `scoring_readiness_registry`, `transfer_gate_06c`, `transfer_robustness_conclusions.md`, `transfer_robustness_manifest.json`, figures `AM06C_01`–`AM06C_10` |
| 07A | `07a_nfl_outcome_context_decomposition.ipynb` | `nfl_play_movement`, `nfl_phase_space`, frozen `nfl_movement_targets_v1.json` (read only; **no Combine table**) | under `…/advanced_movement/context_transfer/`: `nfl_context_distribution`, `nfl_context_variance_decomposition`, `nfl_context_scale_audit`, `nfl_context_adjusted_outcomes`, `nfl_context_overlap`, `nfl_context_outcome_reliability`, `nfl_context_coverage_tradeoff`, `nfl_context_target_registry.json` (exploratory), `nfl_context_conclusions.md`, `context_decomposition_manifest.json`, figures `AM07A_01`–`AM07A_12` |
| 07B | `07b_physics_aligned_transfer_hypotheses.ipynb` | outputs of 06A–06C and 07A, fold representations of 06B (read by name), atlas attempt features, `nfl_play_movement` | `aligned_transfer_hypotheses` (written BEFORE evaluation), `aligned_transfer_unit_audit`, `aligned_transfer_outcome_coverage`, `aligned_transfer_oof_predictions`, `aligned_transfer_model_comparisons`, `aligned_transfer_paired_uncertainty`, `aligned_transfer_physical_association`, `aligned_transfer_matched_pairs`, `aligned_transfer_evidence_vs_m06`, `aligned_transfer_interpretation.md`, `aligned_transfer_manifest.json`, figures `AM07B_01`–`AM07B_11` |
| 07C | `07c_detectability_and_information_ceiling.ipynb` | outputs of 06A–06B (out-of-fold table, folds, fold representations); 07A reliability if present | `transfer_planted_signal_recovery`, `transfer_power_scenarios`, `transfer_null_calibration`, `transfer_mde_estimates`, `transfer_information_limits`, `transfer_power_report.md`, `transfer_power_manifest.json`, checkpoint `checkpoints/transfer_power_replicates.parquet`, figures `AM07C_01`–`AM07C_09` |
| 07D | `07d_athletic_fingerprint_scorecard.ipynb` | outputs of 00–02, 04A, 06A–06B (06C seed replicates and 07A are optional) — **no NFL input** | under `…/advanced_movement/scoring/`: `combine_speed_percentiles`, `combine_speed_development_percentiles`, `combine_fingerprint_neighbors`, `combine_fingerprint_stability`, `combine_measurement_confidence`, `combine_athletic_scorecard_profiles`, `combine_quality_ranking_stability`, `combine_scorecard_examples`, `combine_score_reference_manifest.json`, figures `AM07D_01`–`AM07D_07` |
| 08 | not authored (scouting decision experiments wait for the executed Milestone 07) | | |

## How to run a notebook

1. Open it in Jupyter and select the project kernel.
2. In the **Configuration** cell keep `RUN_MODE = 'smoke'` for the first pass.
3. Run all cells from top to bottom. A cell that fails stops with a message that says what to do.
4. Read the outputs. Then set `RUN_MODE = 'full'`, restart the kernel and run all cells again.
5. Use the same `RUN_MODE` and `SEED` in notebook 01 as in notebook 00.

Smoke mode uses about 60 athletes (all their attempts) and writes under `…/advanced_movement/_sample`.
A full run never reads smoke artifacts; if a prerequisite is missing for the current mode the notebook says which
notebook to run first.

Rerunning is safe: unchanged steps are restored from cache. Set `FORCE_REBUILD = True` to recompute.

## Where outputs go

| What | Full run | Smoke run |
|---|---|---|
| Tables (CSV) | `reports/tables/advanced_movement/` | `reports/tables/advanced_movement/_sample/` |
| Datasets (Parquet + `.schema.json`) | `warehouse/gold/advanced_movement/` | `warehouse/gold/advanced_movement/_sample/` |
| Figures (PNG 300 DPI, SVG, PDF for main figures, `.meta.json`) | `reports/figures/advanced_movement/` | `reports/figures/advanced_movement/_sample/` |
| Models and fitted preprocessing (from notebook 02) | `artifacts/models/advanced_movement/` | `artifacts/models/advanced_movement/_sample/` |

## Notebooks 09A–09G — Combine Intelligence Lab (Milestone 09, AUTHORED 2026-10-14, not executed)

Run in Jupyter, in this order, with the same `RUN_MODE` (smoke first if the smoke outputs of 00–07D exist, then full). Each notebook starts with its synthetic checks; a failing check stops the notebook — send the table back.

| Notebook | Reads | Writes (`…/intelligence_v1/`) | Cost |
|---|---|---|---|
| `09a_nfl_movement_demand_atlas` | 05A play table, silver tracking (one cached scan) | demand episodes, registry, frequency, profiles, context, examples | one tracking pass (like 05A); `BUILD_EXAMPLES` reads one play at a time |
| `09b_cross_domain_movement_matching` | 01, 02, 04A, 07D, 09A (+ 08B.4) | curve features, evidence, shared dictionary, similarity | about a minute |
| `09c_stopwatch_twins_discovery` | 01, 02, 07D, 09B (+ 06B for TCN-B distance) | twins, differences, summary | about a minute |
| `09d_football_interpretation_and_case_studies` | 09A–09C (+ 05B athlete outcomes) | cases, three views, interpretation registry, case HTML reports | about a minute |
| `09e_application_data_mart` | 07D, 09A–09D (+ 08B.4, 08B.6) | `app_mart/v1/` + validation | seconds |
| `09f_intelligence_lab_demonstration` | the mart only | demo report and card in `reports/milestone09/demo` | seconds |
| `09g_competition_presentation_package` | the mart only | `AM09G_*`, intelligence cards, `reports/milestone09/EXECUTED_NUMBERS.md`, `CASE_STUDY_GALLERY_EXECUTED.md` | seconds |

Then launch the application yourself: `cd apps/combine_intelligence_lab; streamlit run app.py` (details in `apps/combine_intelligence_lab/README.md`). Tests: `tests/test_advanced_movement_intelligence.py` or `iv.run_all()` in a cell.

## Notebooks 08B.0–08B.6 — running them (Milestone 08B, AUTHORED 2026-10-13, not executed)

Football performance transfer, outcome validation and scouting decision intelligence. Prerequisites: Milestones 00–07, 08A and 08A.2 executed in the same `RUN_MODE`.

| Order | Notebook | What it does | Gate |
|---|---|---|---|
| 1 | `08b0_outcome_resolution_and_freeze` | field lineage, source classes, **your decision** → `football_outcome_registry_<version>.json` | A semantics |
| 2 | `08b1_football_outcome_reliability` | game-split reliability, context conditioning, secondary-outcome diagnostics | B measurement |
| 3 | `08b2_football_transfer_benchmark` | M0–M6 on the linked football cohort; paired bootstrap; resumable refit permutation | C transfer |
| 4 | `08b3_transfer_robustness_and_detectability` | declared sensitivities; planted-signal simulation; robustness reading | — |
| 5 | `08b4_incremental_movement_information` | what stays repeatable after official tests; scouting evidence registry | D information |
| 6 | `08b5_scouting_decision_intelligence` | abstention, pairwise uncertainty, comparison cases, scorecards | E scorecard |
| 7 | `08b6_final_research_evidence_package` | evidence registry, report tokens, figure index, manifest | F package |

**Step by step.**

1. Restart the kernel. Open 08B.0, leave `RUN_MODE = 'full'` (a smoke run needs smoke outputs of every earlier notebook). Run all cells.
2. Read sections 2–6. Fill `DEFINITION_EVIDENCE` / `FIELD_EVIDENCE` **only** from an authoritative source (the competition data dictionary); otherwise leave them empty.
3. In section 7 set `DECISION`, `PRIMARY`, `SECONDARY`, `RATIONALE`, `ACKNOWLEDGE_PROVISIONAL`, then rerun the two last code cells. The registry is written once; a change needs `REGISTRY_VERSION = 'v2'`.
4. Branch `frozen` or `exploratory`: run 08B.1 → 08B.2 → 08B.3. Branch `blocked` (no decision, `STOP`, or `PROVISIONAL` without acknowledgement): these three raise `FootballTransferBlocked` — skip them.
5. Run 08B.4 → 08B.5 → 08B.6 in every branch.
6. Long cells (permutations in 08B.2, simulation in 08B.3) print a time estimate and progress, and write checkpoints. If you interrupt one, **rerun the cell**: it resumes. `MAX_SECONDS` stops a run cleanly.
7. Rerun 08B.6 after any other notebook: it refreshes the `‹Cnn: …›` tokens of the three `reports/NFL_BDB_2027_*.md` files from the saved tables. A token stays `pending` until its notebook has run.

**Outputs.** `{reports/tables, reports/figures, warehouse/gold}/advanced_movement/football_v1/` (smoke: `…/_sample/football_v1/`); checkpoints in `warehouse/gold/advanced_movement/football_v1/checkpoints/`; reports in `reports/`.

**Send back** the items listed in the last cell of each notebook (check tables, gate tables, the named figures) and any traceback.

## Notebooks 07A–07D — running them (Milestone 07, AUTHORED 2026-10-13, not executed)

Why: Milestone 06 was executed and found no established gain over official Combine testing (decision case E). Milestone
07 asks what the frozen NFL target is made of (07A), tests a small pre-declared family of physically aligned
comparisons — exploratory (07B), measures what the 284-athlete cohort could have detected (07C), and builds a
descriptive, uncertainty-aware pre-draft scorecard (07D). No encoder is trained; PyTorch is not needed.

1. Restart the kernel. Run `07a_…ipynb` with `RUN_MODE = 'smoke'`, then 07B, 07C, 07D in smoke mode (a smoke run needs
   the smoke-mode outputs of 05A/05B and 06A–06C; if they do not exist, go straight to the full run).
2. Restart the kernel. Run 07A → 07B → 07C → 07D with `RUN_MODE = 'full'`.
3. Each notebook starts with a synthetic validation cell (planted, known answers). A failure stops it: send the table.
4. 07B: the hypothesis registry is written before any model is fitted and may not be edited afterwards.
5. 07C: the simulation prints a time estimate from its first replicate, then progress every `PROGRESS_INTERVAL`
   replicates, and checkpoints. If the kernel is interrupted, rerun the same cell — it resumes. Full budget: about
   6,800 replicates; expect tens of minutes (unmeasured — the first printed estimate is the reference). Lower
   `SIMULATION_REPLICATES` for a first pass (a different budget is a different configuration and starts again).
6. 07D: `PROFILE_ATHLETES = 'auto'` draws cards for the athletes of the pre-declared cases; give a list of `nfl_id`
   for specific players.
7. Fill `reports/FOOTBALL_CONTEXT_TRANSFER_AND_SCORING_RESULTS.md` from the persisted tables.

Each notebook ends with a "What to send back" cell. Tests wrapper: `tests/test_advanced_movement_context.py`.

## Notebooks 06A, 06B and 06C — running them (Milestone 06, executed by the user in full mode 2026-10-13)

Question: when two prospects have similar official Combine results, does Combine tracking help predict how they move in NFL
games? Models M0–M7 are compared on identical held-out athletes against the official-Combine model M1. PyTorch is required.
Restart the kernel first (new modules: `transfer`, `transfer_repr`, `transfer_models`, `transfer_robust`,
`transfer_figures`, `transfer_validation`; `TransferStore` in `artifacts`). Everything is written under
`…/advanced_movement/transfer/` (smoke: `…/_sample/transfer/`), with its own run cache and model directory.

**Before anything: freeze the NFL targets.** At authoring time `nfl_movement_targets_v1.json` did not exist (05B decision
`PENDING`). Open 05B, review the full run, set `DECISION`, `PRIMARY`, `SECONDARY`, `RATIONALE` in section 10 and rerun that
cell — in the run mode you are about to use (a smoke run of 06 needs a smoke freeze). Until then 06A stops in section 2 with
`TargetFreezeError`; that is the designed behaviour. Do not re-freeze after 06B has run.

Order: 06A smoke → 06B smoke → 06C smoke → restart → 06A full → 06B full → 06C full.

**06A** (no model; minutes)
1. **Section 1**: seven synthetic checks must read `PASS`. **Section 2**: the freeze audit (stops here when not frozen).
2. **Sections 3–8**: inputs, identity, exposure counts, funnel, missing-data rules, follow-up. No NFL label value is used.
3. **Section 9**: folds. `FOLD_SOURCE = 'auto'` reuses the notebook-00 folds when viable — the saved encoders are then usable.
4. **Section 10**: every saved FPCA / TCN fit against the folds. **Section 11**: gate 06A and the audit manifest.

**06B** (trains one TCN-B per outer fold only where no saved encoder was cleared)
1. **Section 1**: eleven synthetic checks (three train tiny encoders). **Section 4**: fold-safe representations (cached as
   `REP_OUTER`). **Section 5**: the paired cohort. **Section 6**: out-of-fold predictions (cached as `OOF_PRIMARY`).
2. **Sections 7–8**: tables and twelve figures. **Section 9**: gate 06B and the reading generated from the tables.

**06C** (`RUN_LOCO` trains one encoder per draft cohort; `RUN_SEEDS` reuses or trains the seed replicates;
`RUN_PERMUTATIONS` refits two ridge models `n_perm` times per comparison)
1. **Section 1**: three synthetic checks. **Sections 2–8**: positions, cohorts, exposure / context, matched prospects,
   mechanism, seeds and re-ranking, null controls. **Section 9**: evidence table, `scoring_readiness_registry`, decision case.
2. Turning a `RUN_*` switch off leaves that evidence null, and no dimension can then be `predictive_validated`.

## What to send back after Milestone 6 (full run)

- 06A: the validation table; section 2 audit table; section 4 count line; section 6 funnel, reasons and position × cohort
  tables; section 7 table and the predictor line; section 8 follow-up and exposure-rule tables; section 9 design table and
  the retained-design line; section 10 table; section 11 gate; figures `AM06A_01`, `AM06A_04`, `AM06A_07`, `AM06A_10`.
- 06B: the validation table; section 4 fit table; section 5 line; section 7 both tables for every target; section 7 wide-cohort
  and sensitivity tables; section 9 gate and reading; figures `AM06B_02`, `AM06B_03`, `AM06B_07`, `AM06B_08`, `AM06B_11`.
- 06C: the validation table; section 2 both tables; section 3 leave-one-cohort-out table; section 4 table and the direction
  table; section 5 summary table; section 6 table; section 7 seed table; section 8 permutation table; section 9 evidence
  table, registry and decision case; section 10 gate; figures `AM06C_01`, `AM06C_03`, `AM06C_05`, `AM06C_10`.
- Any traceback, with the cell it came from. Then fill `reports/COMBINE_TO_NFL_TRANSFER_VALIDATION.md`.

## Notebooks 05A and 05B — running them (Milestone 05, executed by the user; freeze decision still to record)

Question: which NFL movement outcomes are reliable enough across independent games to be targets of the transfer study? No
Combine value is read. Notebooks 00–04 are NOT prerequisites; the Phase 03 NFL datasets are. PyTorch is not needed. Restart
the kernel first (new modules: `nfl_outcomes`, `outcome_reliability`, `nfl_outcome_figures`, `nfl_outcome_validation`).
Order: 05A smoke → 05B smoke → restart → 05A full → 05B full → set `DECISION`. Everything is written under
`…/advanced_movement/nfl_outcomes/` (smoke: `…/_sample/nfl_outcomes/`), with its own run cache.

**05A** (smoke reads the first 48 games of each season; the full run scans all NFL tracking frames once under four filters —
expect a run of the order of the Phase 04 smoothing study, then it is cached)
1. **Configuration**, **setup**: prints the definition hash. **Section 1**: a missing input stops with the build that produces it.
2. **Section 2**: ten synthetic checks must read `PASS`. Send the table if one fails; nothing was run by the author.
3. **Section 3**: outcome dictionary. **Section 4**: the frame scan (cached as `NFL_PLAYS`).
4. **Sections 5–8**: tables and figures. **Section 9**: athlete-level outcomes and coverage (cached as `NFL_ATHLETE`).
5. **Sections 10–12**: game variation, filter sensitivity, real example plays. **Section 13**: reading generated from the tables.

**05B** (no frame is read; bootstraps and 100 game half-splits)
1. **Setup**: refuses a play table built under another definition hash or seed. **Section 2**: eight synthetic checks.
2. **Sections 4–8**: reliability, exposure, context, sensitivity, redundancy (each cached).
3. **Section 9**: screening table and the rule-based proposal (`FREEZE` or `STOP`).
4. **Section 10**: after the FULL run, set `DECISION`, `PRIMARY`, `SECONDARY`, `RATIONALE` and rerun from that cell.
   `nfl_movement_targets_v1.json` is written once; `REFREEZE_TARGETS = True` replaces it. Do not look at any Combine
   correlation before this cell is final.

## What to send back after Milestone 5 (full run)

- 05A: the validation table; the run-mode line; section 4 count line; section 5 funnel and exposure tables; section 6 summary
  table; section 8 band-coverage table; section 9 both coverage tables; section 10 table; section 11 agreement table; section 13
  reading; figures `AM05A_02`, `AM05A_05`, `AM05A_06`, `AM05A_09`, `AM05A_12`.
- 05B: the validation table; section 4 the first three tables and the season table; section 5 the `available` and `common`
  tables for windows; section 6 both tables; section 7 the rank-agreement table; section 8 table; section 9 screening table,
  the reasons table and the proposal line; section 10 output; section 12 conclusions; figures `AM05B_01`, `AM05B_03`,
  `AM05B_06`, `AM05B_08`, `AM05B_09`.
- Any traceback, with the cell it came from. Then fill `reports/NFL_MOVEMENT_OUTCOME_RESULTS.md`.

## Notebooks 04A and 04B — running them

Prerequisites: notebooks 00–03 run in the same `RUN_MODE`; PyTorch for 04B only. Restart the kernel first (new modules:
`atlas`, `atlas_figures`, `atlas_validation`, `robust`, `robust_train`, `robust_figures`). Order: 04A smoke → 04B smoke →
restart → 04A full → 04B full. Milestone 04 writes only under `…/advanced_movement/atlas/` (smoke: `…/_sample/atlas/`), with
its own run cache and model directory, so no artifact of notebooks 00–03 can be replaced.

**04A** (nothing is trained; a few minutes in full mode, mostly bootstraps)
1. **Configuration**, **setup**: a missing prerequisite names the notebook to run.
2. **Section 1**: reuse inventory. **Section 2**: ten synthetic checks must read `PASS`.
3. **Sections 3–5**: dictionary, the four new sustained-acceleration features, the characterisation tables (cached).
4. **Sections 6–11**: tables and figures only. **Section 12**: output listing. **Section 13**: generated conclusions.

**04B** (trains 2 seeds × 5 folds of TCN-B on input A, plus 3 leave-one-cohort-out encoders if `RUN_LEAVE_COHORT_OUT`)
1. **Configuration**: leave `TRAINING_SEEDS[0] = SEED` and `MAX_EPOCHS = None` so the executed notebook-03 run is reused as the primary seed.
2. **Section 2**: ten synthetic checks. **Section 3**: check the line *Encoder configuration: read from the checkpoints of notebook 03*.
3. **Section 4**: seed replicates (cached, checkpoints under `atlas/tcn`). **Sections 5–6**: controls, assembly, every statistic (cached).
4. **Sections 7–12**: tables and figures only. **Section 13**: the shortlist is frozen (written once in full mode).
5. **Section 16**: after the FULL run, set `DECISION` / `RATIONALE`.

Manual fix (the executed files are too large for the authoring tools): in the first cell of notebooks 02 and 03 replace
`AUTHORED — AWAITING USER EXECUTION` by `EXECUTED BY THE USER (full run 2026-10-11)`. No number is affected.

## What to send back after Milestone 4 (full run)

- 04A: the validation table; section 1 inventory; section 4 missing-reason table; the section 5 summary table; section 7 first
  table; section 8 first table; section 9 first table; section 11 all three tables; section 13 conclusions; figures
  `AM04A_03`, `AM04A_06`, `AM04A_07`, `AM04A_10`.
- 04B: the validation table; section 3 configuration line and seed table; section 4 fold-model summary; section 7 the four
  tables; section 8 the first and the paired tables; section 9 the audit table of the primary seed and the rule-based reading;
  section 10 all tables; section 11 the cohort and leave-one-cohort-out tables; section 13 readiness line and the three
  shortlist tables; section 15 conclusions; figures `AM04B_01`, `AM04B_02`, `AM04B_03`, `AM04B_04`, `AM04B_07`, `AM04B_10`.
- Any traceback, with the cell it came from. Then fill `reports/ADVANCED_MOVEMENT_FEATURE_ATLAS_RESULTS.md`.

## Notebook 03 — running it cell by cell

Prerequisites: notebooks 00, 01 and 02 already run in the same `RUN_MODE`; **PyTorch installed in the kernel's
environment** (a CPU build is enough; the notebook stops with a message if it is missing and installs nothing).
Restart the kernel first, because the package gained new modules (`tcn`, `repr_eval`, `tcn_figures`, `tcn_validation`).

1. **Configuration**: `RUN_MODE = 'smoke'` first; same `SEED` as notebook 00. `MAX_EPOCHS` is an upper bound (smoke caps it at 12).
2. **Setup cell**: dependency table, then the prerequisite artifacts. A missing one names the notebook to run.
3. **Sections 2–4**: FPCA reference tables, configuration tables, input checks (FPCA fold models loaded or refitted).
4. **Section 5**: fifteen synthetic checks must all read `PASS`; the cell stops otherwise. Send the table if one fails.
5. **Section 6**: fold preprocessing table (`same_support_as_fpca` should be true everywhere), then the FPCA benchmark as embeddings.
6. **Section 7**: model size table and figure `AM03_01`.
7. **Section 8**: trains TCN-A (5 folds × 3 inputs). **Section 9**: reconstruction against FPCA, figure `AM03_02`.
8. **Section 10**: design-athlete sweep of the consistency term, then TCN-B (5 × 3) and the 16-dimensional sensitivity (5 × 2).
9. **Section 11**: assembles all methods, cohort check, learning curves; then the evaluation cell (`REPR_EVAL`, `REPR_GATE`).
10. **Sections 12–17**: tables and figures only. **Section 18**: output listing and manifest. **Section 19**: generated conclusions.
11. **Section 20**: after the FULL run, set `DECISION` / `RATIONALE` and rerun that cell.

Then set `RUN_MODE = 'full'`, restart the kernel, run all cells again. Training is cached: a rerun restores tables and
reloads checkpoints unless a parameter or an input file changed (`FORCE_REBUILD = True` retrains). A smoke run is a
technical check on about 60 athletes and a few epochs; its numbers are not results.

## What to send back after Milestone 3 (full run)

- The dependency table and the run-mode line of the setup cell; the synthetic validation table (section 5).
- Section 6: the fold preprocessing table. Section 8 and 10: the fold-model summaries, the design sweep table and the selected line.
- Section 11: the methods table, the cohort table and the reconstruction table.
- Section 12: both collapse tables. Section 13: the first table (confirm / design / all) and the truncation table.
- Section 14: the three tables. Section 15: the cohort table and the window table.
- Section 16: both summary tables. Section 17: the paired tables and the 8-against-16 table.
- Section 19: the generated conclusions; section 20: the verdict table.
- Figures `AM03_03`, `AM03_04`, `AM03_06`, `AM03_07`, `AM03_08`, `AM03_10`, `AM03_12`.
- Any traceback, with the cell it came from.

## Notebook 02 — running it cell by cell

Prerequisite: notebooks 00 and 01 already run in the same `RUN_MODE` (both modes exist on disk). Restart the kernel
first, because the package gained new modules.

1. **Configuration**: keep `RUN_MODE = 'smoke'` for the first pass; same `SEED` as notebook 00.
2. **Setup cell**: loads the six prerequisite artifacts. A missing one names the notebook to run.
3. **Synthetic validation**: nine checks must all read `PASS`; the cell stops otherwise. Send the table if one fails.
4. **Section 1**: the audit table must show no failed *blocking* check (the cell stops if one fails). Read the window
   summary, the extreme recordings and the two decision tables.
5. **Sections 2–3**: representation table, coverage table, mean curves.
6. **Section 4**: two cells. The first fits (6 representations × (1 + repeats × 5) fits); the second computes the
   evaluation and gate tables with bootstraps. Expect a few minutes in full mode, well under a minute in smoke.
7. **Sections 5–11**: tables and figures only; nothing is fitted there.
8. **Section 12**: conclusions generated from the saved tables. A smoke run states that it must not be interpreted.
9. **Section 13**: output listing, model files, the two JSON files.

Then set `RUN_MODE = 'full'`, restart the kernel, run all cells again.

What notebook 02 saves (full run; smoke adds `_sample`):

| Kind | Names |
|---|---|
| Tables, CSV + Parquet (`reports/tables/…`, `warehouse/gold/…`) | `sequence_window_flags`, `preprocessing_audit`, `preprocessing_decisions_final`, `fpca_fold_fits`, `fpca_fold_components`, `fpca_explained_variance`, `fpca_reconstruction_error`, `fpca_reliability`, `fpca_stopwatch_redundancy`, `fpca_body_association`, `fpca_window_dependence`, `fpca_alignment_sensitivity`, `fpca_score_stability`, `fpca_component_gate`, `fpca_representation_summary` |
| Datasets, Parquet only | `fpca_attempt_scores`, `fpca_athlete_scores`, `fpca_sequence_reconstruction`, `fpca_functions` |
| JSON (`warehouse/gold/…`) | `fpca_config_and_basis.json`, `fpca_experiment_manifest.json` |
| Models (`artifacts/models/advanced_movement/fpca/`) | one `.npz` per representation (reference fit + every fold fit) and `_models.json` |
| Figures | `AM02_01`–`AM02_12`, PNG 300 DPI + SVG; PDF for `AM02_02` (representation A), `AM02_04`, `AM02_06`, `AM02_08`, `AM02_12` |

## What to send back after Milestone 2 (full run)

- The synthetic validation table and the audit table (section 1), with the window summary and the extreme recordings.
- Section 4: the fold-fit table and the benchmark-K table.
- Section 6: the reconstruction table. Section 7: the first reliability table and the per-split-repeat table.
- Section 8: the within-position official-times table. Section 9: the shares-explained table.
- Section 11: both tables. Section 12: the generated conclusions.
- Figures `AM02_02` (A), `AM02_04`, `AM02_06`, `AM02_08`, `AM02_09`, `AM02_12`.
- Any traceback, with the cell it came from.

## What to send back after Milestone 1

- Notebook 00: the validation table, the missing-datasets line, the flow table, the reconciliation table (full mode),
  the leakage audit, figures `AM00_01`, `AM00_05`, `AM00_07`.
- Notebook 01: the three alignment tables of section 8, the scenario counts of section 7, figures `AM01_03`, `AM01_04`,
  `AM01_12`, and your `CHOICES` / `RATIONALE` from section 9.
- Any traceback, with the cell it came from.
