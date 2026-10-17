# M13 experiment matrix

Stage id → module → inputs → outputs (under `RUN_DIR = MyDrive/NFLBowl/experimental_lab/m13_outcome_transfer_v1/`). Status per run: `manifests/stages.json`.

| Stage | Part | Module / API | Needs | Main outputs (`tables/…` unless stated) | Cost | Cache |
|---|---|---|---|---|---|---|
| A1 | A | `handoff.environment`, `environment.validate_bundle`, `handoff.open_experiment` | bundle, Drive | `handoff/bundle_checks`, `config/*.json`, `manifests/provenance.json` | s | — |
| A2 | A | `handoff.discover / adopt / adopt_embeddings / inventory / ready_models / f1_validity` | M12 experiment dir | `handoff/checkpoint_inventory`, `adoption`, `embedding_adoption`, `f1_validity_by_fold`, fig `a2_readiness_matrix` | min (copies + CPU load checks) | manifest |
| A3 | A | `handoff.load_m12_tables / reconcile` | M12 run dir | `handoff/m12_tables_found`, `m12_reconciliation` | s | — |
| A4 | A | `handoff.prerequisites / handoff_record` | A2, A3 | `handoff/prerequisites`, `provenance`, `manifests/handoff.json` | s | — |
| B1 | B | `target_audit.target_provenance` | bundle (+ optional freeze JSON) | `target/target_provenance` | s | — |
| B2 | B | `target_audit.athlete_sets / attrition_flow / overlap_matrix` | bundle | `target/attrition_flow_pre_ladder`, `overlap_matrix_pre_ladder`, figs `b2_*` | s | — |
| B3 | B | `target_audit.analysis_frame / variance_decomposition / position_table / outliers` | bundle | `target/variance_decomposition`, `position_table`, `target_outliers`, figs `b3_*` | s | — |
| C1 | C | `Experiment.transfer_predictions`, `TF.baseline_predictions / reproducibility_check` | ready encoders | `datasets/cached/ladder_oof`, `baseline_oof`, `transfer/reproducibility_check` | GPU min (embedding if not adopted) + CPU ridge | experiment manifest (`transfer_oof`) |
| C2 | C/D | `TF.long_predictions / validate_alignment / cohorts` | C1 | in memory (audited in C3) | s | — |
| C3 | C/D | `TF.wide / cohort_metrics / paired_comparisons / rmse_intervals / fold_table / native_cohort` | C2 | **`transfer/common_cohort_audit.csv`**, **`common_cohort_predictions.parquet`**, **`paired_transfer_comparisons.csv`**, `metrics_*`, `fold_metrics_*`, `rmse_intervals` | min | — |
| C4 | C | `lab.figures`, `M13F.case_dumbbell`, `TF.rank_agreement / best_worst` | C3 | figs `c4_*`, `transfer/rank_agreement`, `best_worst_cases` | s | — |
| D1–D3 | D | `TF`, `target_audit` | C3 | `transfer/native_vs_common`, `target/attrition_flow`, `overlap_matrix`, figs `d_*` | s | — |
| E0 | E | `handoff.sequence_metrics`, `biomech.athlete_features / ladder_frame`, `REL.candidate_columns` | bundle | `datasets/cached/sequence_metrics`, `features/athlete/athlete_movement_features`, `reliability/metric_definitions` | min | provenance-matched (M13, else M12 copy) |
| E1 | E | `REL.reliability_audit / explain_m12` | E0 | `reliability/reliability_audit`, `m12_negative_reliability_explained`, figs `e1_*` | min | signature |
| E2 | E | `REL.equal_observation_curves / projected_curve` | E0 | `reliability/reliability_curves`, `reliability_curve_differences`, `reliability_domain_difference_summary`, fig `e2_reliability_curves` | min | signature |
| E3 | E | `REL.selection_ladder` (train-fold blocks), `TF.paired_comparisons` | E0, C2 | `reliability/selection_log`, `selection_paired_comparisons`, `datasets/cached/biomech_selection_oof`, figs `e3_*` | min | — |
| E4 | E | `REL.feature_stability`, `biomech.incremental_value` | E0 | `reliability/feature_stability`, `incremental_value`, fig `e4_added_feature_importance` | min | — |
| E5 | E | `REL.interaction_tests` | E0 | `reliability/interaction_tests`, fig `e5_interactions` | min | — |
| F1 | F | `PM.oof_position_models / within_position_metrics / position_effects` | C1, E0 | `position/position_model_metrics`, `position_model_paired`, `position_effects`, `position_model_log`, figs `f_*` | min | — |
| G1 | G | `DM.run / conditional_distributions` (frozen embeddings, CEM) | E0, embeddings | `domain/domain_matching_long`, `domain_matching_summary`, figs `g_*` | GPU-free, 10–30 min | signature |
| H1 | H | `SV.retrieval_records / geometry_records / probe_summary / transfer_records / relationship_table / associations` | C3, G1 | `retrieval_transfer/*`, figs `h_*` | min | experiment records |
| I1 | I | `SV.best_model / tracking_gain / gain_by_group / forty_twins / unexplained_by_position / oof_intervals / build_profiles`, `interpret.interpretable_ladder` | C3, E, F | `scouting/*`, `reports/athlete_profiles/*.md`, figs `i_*` | min | — |
| J1 | J | `TF` (all sources), `REP.unified_comparison / determine_outcome / evidence_rows` | all | `synthesis/final_*`, `unified_comparison`, `evidence_matrix`, `manifests/outcome.json`, `transfer/all_predictions_long`, fig `j_evidence_matrix` | min | — |
| K1 | K | `M13F.dashboard`, `REP.final_report / write_reports`, `RUN.inventory / export_zip` | J1 | **`reports/markdown/M13_FINAL_RESEARCH_REPORT.md`**, **`reports/html/M13_FINAL_RESEARCH_REPORT.html`**, `synthesis/figure_index`, `artifact_inventory`, `exports/*.zip` | s | — |

## Model inventory

| Name | Definition | Source |
|---|---|---|
| MEAN | training-fold mean | `TF.baseline_predictions` |
| POS | position dummies only (`ae.Model([])`) | same engine, same frame |
| M1 | draft year, height, weight, forty, ten + position | ladder |
| M1+F0 | + athlete-mean Combine-forty engineered features | ladder |
| M1+F1_TCN | + Milestone 06 TCN-B block (primary only if `f1_validity` = valid) | ladder |
| M1+F2…F6 | + fold-f inductive embedding, PCA ≤ 8 fitted on training athletes | ladder |
| M1+BIO_all / reliable / trainsel / poscond | + Combine movement metrics (all / train ICC ≥ threshold / train top-k partial association / train position-standardised) | `REL.selection_ladder` |
| ⟨set⟩·pooled / centred / standardised / specific / partial | position-conditioned variants for M1, M1+F0, M1+BIO_all | `PM.oof_position_models` |

## Figures (PNG 300 dpi + SVG + PDF; data in `figures/data/`)

| Figure group | Names |
|---|---|
| Cohort selection flow | `b2_cohort_flow`, `d_cohort_flow` |
| Overlap | `b2_overlap_matrix`, `d_overlap_matrix` |
| Target | `b3_target_by_position`, `b3_position_counts`, `c4_target_distribution` |
| Ladder | `c4_pred_vs_actual`, `c4_rmse_comparison`, `c4_delta_rmse_forest_vs_m1`, `c4_delta_rmse_forest_vs_f0`, `c4_fold_rmse_heatmap`, `c4_residuals`, `c4_residual_by_position`, `c4_rank_agreement`, `c4_best_worst_cases` |
| Reliability | `e1_sb_vs_icc`, `e1_icc_forest`, `e2_reliability_curves`, `e3_biomech_forest`, `e3_selection_heatmap`, `e4_added_feature_importance`, `e5_interactions` |
| Position | `f_position_effects`, `f_skill_beyond_position` |
| Domain | `g_domain_auc_heatmap`, `g_mmd2_heatmap`, `g_norm_ratio`, `g_retrieval_sensitivity`, `g_conditional_distributions` |
| Retrieval vs transfer | `h_retrieval_vs_transfer`, `h_model_level` |
| Scouting | `i_help_hurt_by_position`, `i_forty_twins`, `i_interpretable_importance`, `i_case_errors`, `i_case_profiles` |
| Synthesis | `a2_readiness_matrix`, `j_evidence_matrix`, `k_executive_dashboard` |
