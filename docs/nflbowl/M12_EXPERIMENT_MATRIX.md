# M12 comprehensive lab — experiment matrix

The machine-readable source is `src/nflbowl/lab/registry.py` (`SPECS`). At run time the notebook exports it with live statuses to `config/experiment_registry.csv|md`. Statuses come from the stage ledger: completed · cached · pending · skipped · blocked · failed.

## Representation families

| Role | Models (registry names) |
|---|---|
| Primary comparison | F0_engineered, F2_cnn_physics, F4_physics_contrastive |
| Secondary | random_frozen_F4_architecture, F3_masked_ae, F5_jepa_physics, F6_transformer |
| Transfer only | F1_TCN (external athlete-level representation) |
| Mechanism (full mode) | `reg.ablations()`: F4_nophys, F4_noalign, F4_adv, F4_mmd, F4_category, F4_shuffled_phys |

## Matrix

| ID | Experiment | Hypothesis | Regime / held-out | Baseline | Metrics | Statistics | Modes / flag | Decision rule |
|---|---|---|---|---|---|---|---|---|
| A2 | Reproducibility record | — | — | — | hashes, versions | — | all | provenance for every table |
| B1 | Dataset audit | the bundle matches its provenance | — | declared counts | counts, coverage, missingness, fold balance | exact | all | a mismatch fails G1 |
| B2 | Domain composition | domains differ even within position | — | Combine | medians, SMD, position-adjusted gap | athlete-cluster bootstrap | all | describes the gap, not a failure |
| B3 | Attrition | eligibility is not random across positions | — | all athletes | step counts | descriptive | all | n < min_subgroup → insufficient |
| B4 | Leakage audit | no cross-fold information flow | inductive folds | — | split, role, duplicate and shortcut checks | exact | all | any fail → G1 fail |
| C1/C3 | Geometry and domain | F4 lowers MMD² but stays separable | held-out rows | F0, random | effective rank, domain AUC, MMD², probes | fold range | all · run_full_diagnostics | collapse rule; AUC ≥ 0.9 = separable |
| C4 | Projections | — | held-out rows, one fold | — | PCA / UMAP coordinates | none | all · run_full_diagnostics | exploratory only |
| D1 | Transductive reference | earlier ≈0.52 vs ≈0.50 reproducible | transductive | F0 | cross AUC | record CI | all | same bundle only; never pooled |
| D2 | Inductive training | stable training without held-out athletes | 5 athlete folds | — | losses, epochs | — | pilot (1 fold) / standard / full · run_training | non-finite → G2 fail |
| D3/D4 | Inductive retrieval, paired | F4 > chance, F4 > F0, F4 > F2 | held-out, identical pools | F0, chance | AUC, top-1, recall@K, MRR, median rank, margins | fold-stratified paired bootstrap | all · run_inductive | Δ CI > 0 and mean CI > 0.5 |
| D5 | Case studies | — | held-out | F0 | rank, margin, speed profiles | none | all | illustrative |
| E2–E4 | Transfer ladder | M1+F4 beats M1 and M1+F0 | OOF athlete folds | M1, M1+F0 | RMSE, MAE, R², bias, residual quantiles, ΔRMSE | paired athlete bootstrap; common + native cohorts | standard / full · run_transfer | both Δ CIs > 0 → G4 pass |
| E5 | Position effects | gains concentrate in positions | OOF | M1 | ΔRMSE by position | bootstrap + BH | standard / full · run_transfer | n < 30 exploratory |
| E7 | Scouting value | rankings and top decile change | OOF | M1 | Spearman, top-decile precision, calibration, thresholds | athlete bootstrap | standard / full · run_transfer | descriptive |
| F1 | Ablations | loss terms change retrieval / domain geometry | held-out, same pools | F4 | ΔAUC, domain AUC, MMD², rank, probes | paired bootstrap | full · run_ablations + run_training | benefits and regressions listed |
| G1 | Seeds | F4 stable across seeds | held-out | main seed | ΔAUC | paired bootstrap | full · run_seeds | seed spread vs model gaps |
| G2 | NFL-fraction curve | more NFL data helps alignment | transductive | 100 % | cross AUC | CI | full · run_learning_curves | diagnostic only |
| G3 | Pool difficulty | the advantage survives stopwatch twins | fixed sub-pools | chance per pool | AUC, top-1 | paired bootstrap | all · run_inductive | sign kept → robustness |
| G4 | Sequence quality | not driven by artefacts | filtered held-out rows | unfiltered | ΔAUC | paired bootstrap | standard / full | a sign flip = fragile |
| G5 | Heterogeneity | consistent across subgroups | held-out | F0 | subgroup ΔAUC | paired bootstrap | standard / full | descriptive |
| H1 | Biomechanics | metrics are repeatable, transfer and add value | deterministic + OOF ladder | M1 | reliability, Combine↔NFL Spearman, ΔRMSE, partial Spearman | bootstrap + BH | all | "actionable candidate": reliability ≥ 0.5, CI > 0 |
| I1 | Interpretability | signal is carried by physical metrics | OOF | — | permutation importance, PDP, counterfactuals | fold spread | standard / full (or pilot + run_transfer) | predictive, not causal |
| J1 | Gates and integrated comparison | — | saved tables | — | gate status | declared rules | all | missing evidence = not_evaluated |
| X1 | 12A–12G warehouse studies | — | — | — | — | — | — | **blocked** on the portable bundle |

## Research gates (`synthesis.GATE_RULES`)

| Gate | Pass rule |
|---|---|
| G1 dataset & leakage | no failing bundle or leakage check |
| G2 training stability | finite losses and best validation loss for F4 on all folds |
| G3 inductive retrieval | F4 AUC CI > 0.5 and Δ(F4 − F0) CI > 0, all folds |
| G4 NFL transfer | ΔRMSE CI > 0 vs M1 and vs M1+F0 (common cohort) |
| G5 robustness | G3 pass; no sign flip under filters / pools; seeds stable |
| G6 interpretability | ≥ 1 top-5 movement metric with reliability ≥ 0.5 |
| G7 scouting demonstration | G1, G3, G4, G5 pass and G6 not fail |
