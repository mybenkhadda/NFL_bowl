# Figure index

## Core visual story

| Step | Story | Saved figure(s) |
|---|---|---|
| 1 | Feature reliability versus redundancy | AM04A_03_reliability_vs_stopwatch_redundancy · AM04A_03b_reliability_vs_redundancy_by_position · AM04A_04_reliability_by_family · AM04A_08_reliable_speed_vs_derivatives |
| 2 | FPCA curve structure | AM03_04_reconstruction_fpca_tcn_a_tcn_b · AM04A_09_fpca_and_tcn_in_context · AM04B_02_matched_retrieval_tcn_vs_fpca · AM06B_08_tcn_vs_speed2_vs_fpca |
| 3 | TCN matched-stopwatch retrieval | AM04B_02_matched_retrieval_tcn_vs_fpca |
| 4 | Original NFL Δ RMSE (Milestone 06) | AM06B_03_delta_rmse_relative_to_m1 |
| 5 | NFL context decomposition | AM07A_03_variance_components |
| 6 | Statistical detectability | AM07C_01_power_vs_effect · AM07C_05_power_by_position |
| 7 | Football-outcome availability | AM08A_02_outcome_feasibility_status |
| 8 | Separation semantics and reliability | AM08A2_09_outcome_reliability · AM08A2_09b_odd_even_game_values |
| 9 | Football transfer performance | AM08B2_02_paired_delta_rmse_forest |
| 10 | Matched-prospect comparison | AM08B2_07_matched_prospect_concordance |
| 11 | Scorecard uncertainty | AM08B5_04_pairwise_uncertainty |
| 12 | Final scouting intelligence diagram | _missing — not saved yet_ |

## All figures

| Area | Figure | File | Source | n |
|---|---|---|---|---|
|  | AM00_01_dataset_flow | `reports/figures/advanced_movement/AM00_01_dataset_flow.png` | dataset_audit | 510 athletes |
|  | AM00_02_athlete_coverage_by_position | `reports/figures/advanced_movement/AM00_02_athlete_coverage_by_position.png` | dataset_audit | 510 athletes |
|  | AM00_03_attempt_count_distribution | `reports/figures/advanced_movement/AM00_03_attempt_count_distribution.png` | athlete_registry | 418 athletes |
|  | AM00_04_sequence_length_distribution | `reports/figures/advanced_movement/AM00_04_sequence_length_distribution.png` | sequence_manifest | 793 forty attempts |
|  | AM00_05_cohort_exposure | `reports/figures/advanced_movement/AM00_05_cohort_exposure.png` | athlete_registry | 510 athletes |
|  | AM00_06_missingness_heatmap | `reports/figures/advanced_movement/AM00_06_missingness_heatmap.png` | missingness_matrix | 510 athletes |
|  | AM00_07_split_registry | `reports/figures/advanced_movement/AM00_07_split_registry.png` | split_registry | 418 athletes |
|  | AM01_01_variable_length_and_missingness | `reports/figures/advanced_movement/AM01_01_variable_length_and_missingness.png` | sequence_frames | 793 sequences |
|  | AM01_02_attempt_p10_peak_speed | `reports/figures/advanced_movement/AM01_02_attempt_p10_peak_speed.png` | sequence_frames | 59 frames |
|  | AM01_02_attempt_p50_peak_speed | `reports/figures/advanced_movement/AM01_02_attempt_p50_peak_speed.png` | sequence_frames | 54 frames |
|  | AM01_02_attempt_p90_peak_speed | `reports/figures/advanced_movement/AM01_02_attempt_p90_peak_speed.png` | sequence_frames | 51 frames |
|  | AM01_03_same_athlete_repeated_attempts | `reports/figures/advanced_movement/AM01_03_same_athlete_repeated_attempts.png` | sequence_frames | 4 athletes |
|  | AM01_04_same_official_forty_different_run | `reports/figures/advanced_movement/AM01_04_same_official_forty_different_run.png` | sequence_frames + athlete_registry | 3 pairs |
|  | AM01_05_speed_by_position_elapsed_time | `reports/figures/advanced_movement/AM01_05_speed_by_position_elapsed_time.png` | sequence_frames | 793 sequences |
|  | AM01_06_acceleration_by_position_speed_phase | `reports/figures/advanced_movement/AM01_06_acceleration_by_position_speed_phase.png` | sequence_frames | 793 sequences |
|  | AM01_07_acceleration_by_cohort_speed_phase | `reports/figures/advanced_movement/AM01_07_acceleration_by_cohort_speed_phase.png` | sequence_frames | 793 sequences |
|  | AM01_08_recording_window_censoring | `reports/figures/advanced_movement/AM01_08_recording_window_censoring.png` | sequence_summary | 793 sequences |
|  | AM01_09_speed_vs_displacement | `reports/figures/advanced_movement/AM01_09_speed_vs_displacement.png` | sequence_frames | 793 sequences |
|  | AM01_10_measurement_noise_by_cohort | `reports/figures/advanced_movement/AM01_10_measurement_noise_by_cohort.png` | sequence_summary | 793 sequences |
|  | AM01_11_one_athlete_four_alignments | `reports/figures/advanced_movement/AM01_11_one_athlete_four_alignments.png` | sequence_frames | 1 athlete |
|  | AM01_12_alignment_comparison | `reports/figures/advanced_movement/AM01_12_alignment_comparison.png` | alignment_comparison | 125 design + 293 confirm athletes |
|  | AM02_01_mean_curves_by_position | `reports/figures/advanced_movement/AM02_01_mean_curves_by_position.png` | sequence_frames | 793 sequences of 418 athletes |
|  | AM02_02_modes_A_speed_by_distance | `reports/figures/advanced_movement/AM02_02_modes_A_speed_by_distance.png` | fpca_functions (reference fit, transductive) | 418 athletes |
|  | AM02_02_modes_B_accel_by_speed | `reports/figures/advanced_movement/AM02_02_modes_B_accel_by_speed.png` | fpca_functions (reference fit, transductive) | 418 athletes |
|  | AM02_02_modes_C_speed_by_time | `reports/figures/advanced_movement/AM02_02_modes_C_speed_by_time.png` | fpca_functions (reference fit, transductive) | 418 athletes |
|  | AM02_03_explained_variance | `reports/figures/advanced_movement/AM02_03_explained_variance.png` | fpca_explained_variance | 15 fold fits per representation |
|  | AM02_04_heldout_reconstruction_error | `reports/figures/advanced_movement/AM02_04_heldout_reconstruction_error.png` | fpca_reconstruction_error | 418 athletes, each held out once |
|  | AM02_05_reconstructions_A_speed_by_distance | `reports/figures/advanced_movement/AM02_05_reconstructions_A_speed_by_distance.png` | fpca_sequence_reconstruction (held-out) | 4 held-out attempts |
|  | AM02_05_reconstructions_B_accel_by_speed | `reports/figures/advanced_movement/AM02_05_reconstructions_B_accel_by_speed.png` | fpca_sequence_reconstruction (held-out) | 4 held-out attempts |
|  | AM02_05_reconstructions_C_speed_by_time | `reports/figures/advanced_movement/AM02_05_reconstructions_C_speed_by_time.png` | fpca_sequence_reconstruction (held-out) | 4 held-out attempts |
|  | AM02_06_first_vs_second_attempt | `reports/figures/advanced_movement/AM02_06_first_vs_second_attempt.png` | fpca_attempt_scores (out-of-fold) | athletes with two scored attempts |
|  | AM02_07_within_vs_between_variation | `reports/figures/advanced_movement/AM02_07_within_vs_between_variation.png` | fpca_reliability | athletes with a repeat |
|  | AM02_08_reliability_vs_stopwatch_redundancy | `reports/figures/advanced_movement/AM02_08_reliability_vs_stopwatch_redundancy.png` | fpca_component_gate | athletes with a repeat and official times |
|  | AM02_09_window_dependence_heatmap | `reports/figures/advanced_movement/AM02_09_window_dependence_heatmap.png` | fpca_window_dependence | athletes (within-athlete panel: athletes with two scored attempts) |
|  | AM02_10_window_and_cohort_shares | `reports/figures/advanced_movement/AM02_10_window_and_cohort_shares.png` | fpca_component_gate | athletes |
|  | AM02_11_alignment_sensitivity_and_fold_stability | `reports/figures/advanced_movement/AM02_11_alignment_sensitivity_and_fold_stability.png` | fpca_alignment_sensitivity, fpca_fold_components | athletes scored under both alignments; 15 fold fits |
|  | AM02_12_representation_comparison | `reports/figures/advanced_movement/AM02_12_representation_comparison.png` | fpca_representation_summary, fpca_component_gate | athletes with a repeat |
|  | AM03_01_architecture_and_input | `reports/figures/advanced_movement/AM03_01_architecture_and_input.png` | sequence_frames, split_registry | 1 held-out attempt (illustration) |
|  | AM03_02_masked_reconstruction_examples | `reports/figures/advanced_movement/AM03_02_masked_reconstruction_examples.png` | sequence_frames, fold models | 4 held-out attempts (rule-selected) |
|  | AM03_02_masked_reconstruction_examples_tcn_a | `reports/figures/advanced_movement/AM03_02_masked_reconstruction_examples_tcn_a.png` | sequence_frames, fold models | 4 held-out attempts (rule-selected) |
|  | AM03_03_learning_curves | `reports/figures/advanced_movement/AM03_03_learning_curves.png` | tcn_a_training_history, tcn_b_training_history | 30 fold models |
|  | AM03_04_reconstruction_fpca_tcn_a_tcn_b | `reports/figures/advanced_movement/AM03_04_reconstruction_fpca_tcn_a_tcn_b.png` | repr_reconstruction | 418 athletes |
|  | AM03_05_same_vs_different_athlete_distances | `reports/figures/advanced_movement/AM03_05_same_vs_different_athlete_distances.png` | repr_pair_statistics | 375 athletes with two attempts |
|  | AM03_06_retrieval_performance | `reports/figures/advanced_movement/AM03_06_retrieval_performance.png` | repr_repeatability | 265 confirm athletes |
|  | AM03_07_collapse_diagnostics | `reports/figures/advanced_movement/AM03_07_collapse_diagnostics.png` | repr_collapse, repr_covariance_spectrum | 55 fold models |
|  | AM03_08_reliability_vs_stopwatch_redundancy | `reports/figures/advanced_movement/AM03_08_reliability_vs_stopwatch_redundancy.png` | repr_method_summary | 265 confirm athletes |
|  | AM03_09_cohort_and_position_probes | `reports/figures/advanced_movement/AM03_09_cohort_and_position_probes.png` | repr_probes | 418 athletes |
|  | AM03_10_recording_window_dependence | `reports/figures/advanced_movement/AM03_10_recording_window_dependence.png` | repr_window_dependence, repr_truncation_stability | 375 athletes with two attempts |
|  | AM03_11_same_forty_athlete_pairs | `reports/figures/advanced_movement/AM03_11_same_forty_athlete_pairs.png` | sequence_summary, athlete_registry, embeddings | 4 pairs of athletes (illustration) |
|  | AM03_12_evidence_by_method | `reports/figures/advanced_movement/AM03_12_evidence_by_method.png` | repr_method_summary, repr_verdict | 265 confirm athletes |
| atlas | AM04A_01_feature_inventory | `reports/figures/advanced_movement/atlas/AM04A_01_feature_inventory.png` | feature_dictionary, atlas_feature_status, atlas_coverage | 1409 definitions; 418 tracked athletes |
| atlas | AM04A_02_missingness_eligibility_map | `reports/figures/advanced_movement/atlas/AM04A_02_missingness_eligibility_map.png` | atlas_coverage, atlas_feature_status | 418 tracked athletes |
| atlas | AM04A_03_reliability_vs_stopwatch_redundancy | `reports/figures/advanced_movement/atlas/AM04A_03_reliability_vs_stopwatch_redundancy.png` | atlas_characteristics | 375 athletes with a repeat (at most) |
| atlas | AM04A_03b_reliability_vs_redundancy_by_position | `reports/figures/advanced_movement/atlas/AM04A_03b_reliability_vs_redundancy_by_position.png` | atlas_characteristics (position scopes) | 375 athletes with a repeat (at most) |
| atlas | AM04A_04_reliability_by_family | `reports/figures/advanced_movement/atlas/AM04A_04_reliability_by_family.png` | atlas_characteristics | 375 athletes with a repeat (at most) |
| atlas | AM04A_05_quality_filter_vs_retention | `reports/figures/advanced_movement/atlas/AM04A_05_quality_filter_vs_retention.png` | atlas_quality_retention, atlas_reliability | 418 tracked athletes |
| atlas | AM04A_06_recording_window_sensitivity | `reports/figures/advanced_movement/atlas/AM04A_06_recording_window_sensitivity.png` | atlas_characteristics | 375 athletes with a repeat (at most) |
| atlas | AM04A_07_feature_redundancy_matrix | `reports/figures/advanced_movement/atlas/AM04A_07_feature_redundancy_matrix.png` | atlas_redundancy_matrix, atlas_feature_clusters | 418 tracked athletes |
| atlas | AM04A_08_reliable_speed_vs_derivatives | `reports/figures/advanced_movement/atlas/AM04A_08_reliable_speed_vs_derivatives.png` | atlas_attempt_features, atlas_reliability | 375 athletes with a repeat (at most) |
| atlas | AM04A_09_fpca_and_tcn_in_context | `reports/figures/advanced_movement/atlas/AM04A_09_fpca_and_tcn_in_context.png` | atlas_characteristics (rows reused from notebooks 02 and 03) | 375 athletes with a repeat (at most) |
| atlas | AM04A_10_leading_characteristic_candidates | `reports/figures/advanced_movement/atlas/AM04A_10_leading_characteristic_candidates.png` | atlas_feature_status | 375 athletes with a repeat (at most) |
| atlas | AM04B_01_three_seed_performance | `reports/figures/advanced_movement/atlas/AM04B_01_three_seed_performance.png` | robust_seed_runs, robust_seed_summary | 3 training seeds; 265 confirm athletes (at most) |
| atlas | AM04B_02_matched_retrieval_tcn_vs_fpca | `reports/figures/advanced_movement/atlas/AM04B_02_matched_retrieval_tcn_vs_fpca.png` | robust_matched_summary | 234 confirm athletes, 865 matched comparisons |
| atlas | AM04B_03_raw_vs_residual_embedding_retrieval | `reports/figures/advanced_movement/atlas/AM04B_03_raw_vs_residual_embedding_retrieval.png` | robust_residualisation_audit | 265 confirm athletes (at most) |
| atlas | AM04B_04_effective_dimension_vs_stability | `reports/figures/advanced_movement/atlas/AM04B_04_effective_dimension_vs_stability.png` | robust_dimension_stability | 265 confirm athletes (at most) |
| atlas | AM04B_05_cohort_probe_vs_baseline | `reports/figures/advanced_movement/atlas/AM04B_05_cohort_probe_vs_baseline.png` | robust_cohort_probe | 418 athletes (at most) |
| atlas | AM04B_06_acquisition_window_sensitivity | `reports/figures/advanced_movement/atlas/AM04B_06_acquisition_window_sensitivity.png` | robust_truncation_stability, robust_probes | 418 athletes (at most) |
| atlas | AM04B_08_reliability_vs_redundancy_with_uncertainty | `reports/figures/advanced_movement/atlas/AM04B_08_reliability_vs_redundancy_with_uncertainty.png` | robust_representation_quality | 265 confirm athletes (at most) |
| atlas | AM04B_10_representation_quality_summary | `reports/figures/advanced_movement/atlas/AM04B_10_representation_quality_summary.png` | robust_representation_quality, transfer_candidates_v1 | 265 confirm athletes (at most) |
| context_transfer | AM07A_01_raw_distributions | `reports/figures/advanced_movement/context_transfer/AM07A_01_raw_distributions.png` | nfl_play_movement (05A) | 401 athletes |
| context_transfer | AM07A_02_player_game_variation | `reports/figures/advanced_movement/context_transfer/AM07A_02_player_game_variation.png` | nfl_play_movement (05A) | 7,421 athlete-games |
| context_transfer | AM07A_03_variance_components | `reports/figures/advanced_movement/context_transfer/AM07A_03_variance_components.png` | nfl_context_variance_decomposition | 401 athletes |
| context_transfer | AM07A_04_context_association | `reports/figures/advanced_movement/context_transfer/AM07A_04_context_association.png` | nfl_context_variance_decomposition | 241,760 windows |
| context_transfer | AM07A_05_phase_space | `reports/figures/advanced_movement/context_transfer/AM07A_05_phase_space.png` | nfl_phase_space (05A) | 15,592,605 frames |
| context_transfer | AM07A_06_alignment_comparison | `reports/figures/advanced_movement/context_transfer/AM07A_06_alignment_comparison.png` | nfl_context_distribution | 401 athletes |
| context_transfer | AM07A_07_start_speed_support | `reports/figures/advanced_movement/context_transfer/AM07A_07_start_speed_support.png` | nfl_context_distribution | 241,760 windows |
| context_transfer | AM07A_08_raw_vs_C2 | `reports/figures/advanced_movement/context_transfer/AM07A_08_raw_vs_C2.png` | nfl_context_adjusted_outcomes | 401 athletes |
| context_transfer | AM07A_08_raw_vs_C4 | `reports/figures/advanced_movement/context_transfer/AM07A_08_raw_vs_C4.png` | nfl_context_adjusted_outcomes | 401 athletes |
| context_transfer | AM07A_09_overlap_retention | `reports/figures/advanced_movement/context_transfer/AM07A_09_overlap_retention.png` | nfl_context_overlap | 401 athletes |
| context_transfer | AM07A_10_reliability_vs_complexity | `reports/figures/advanced_movement/context_transfer/AM07A_10_reliability_vs_complexity.png` | nfl_context_coverage_tradeoff | 401 athletes |
| context_transfer | AM07A_11_reliability_vs_games | `reports/figures/advanced_movement/context_transfer/AM07A_11_reliability_vs_games.png` | nfl_context_outcome_reliability | athletes with enough later games (see points) |
| context_transfer | AM07A_12_rank_shift_examples | `reports/figures/advanced_movement/context_transfer/AM07A_12_rank_shift_examples.png` | nfl_context_adjusted_outcomes | 8 athletes (illustration) |
| context_transfer | AM07B_01_combine_speed_vs_nfl_speed | `reports/figures/advanced_movement/context_transfer/AM07B_01_combine_speed_vs_nfl_speed.png` | atlas_attempt_features + nfl_play_movement | paired Milestone 06 cohort (284 athletes); each outcome on its eligible athletes |
| context_transfer | AM07B_02_combine_accel_vs_nfl_accel | `reports/figures/advanced_movement/context_transfer/AM07B_02_combine_accel_vs_nfl_accel.png` | atlas_attempt_features + nfl_play_movement | paired Milestone 06 cohort (284 athletes); each outcome on its eligible athletes |
| context_transfer | AM07B_03_speed_band_alignment | `reports/figures/advanced_movement/context_transfer/AM07B_03_speed_band_alignment.png` | aligned_transfer_physical_association | paired Milestone 06 cohort (284 athletes); each outcome on its eligible athletes |
| context_transfer | AM07B_04_context_adjusted_vs_raw | `reports/figures/advanced_movement/context_transfer/AM07B_04_context_adjusted_vs_raw.png` | nfl_context_adjusted_outcomes (07A) | 401 NFL athletes |
| context_transfer | AM07B_05_model_gains | `reports/figures/advanced_movement/context_transfer/AM07B_05_model_gains.png` | aligned_transfer_paired_uncertainty | paired Milestone 06 cohort (284 athletes); each outcome on its eligible athletes |
| context_transfer | AM07B_06_gain_by_position | `reports/figures/advanced_movement/context_transfer/AM07B_06_gain_by_position.png` | aligned_transfer_paired_uncertainty | paired Milestone 06 cohort (284 athletes); each outcome on its eligible athletes |
| context_transfer | AM07B_07_gain_by_cohort | `reports/figures/advanced_movement/context_transfer/AM07B_07_gain_by_cohort.png` | aligned_transfer_paired_uncertainty | paired Milestone 06 cohort (284 athletes); each outcome on its eligible athletes |
| context_transfer | AM07B_08_gain_by_exposure | `reports/figures/advanced_movement/context_transfer/AM07B_08_gain_by_exposure.png` | aligned_transfer_paired_uncertainty | paired Milestone 06 cohort (284 athletes); each outcome on its eligible athletes |
| context_transfer | AM07B_09_matched_prospects | `reports/figures/advanced_movement/context_transfer/AM07B_09_matched_prospects.png` | aligned_transfer_matched_pairs | 833 frozen pairs |
| context_transfer | AM07B_10_tcn_vs_physical | `reports/figures/advanced_movement/context_transfer/AM07B_10_tcn_vs_physical.png` | aligned_transfer_paired_uncertainty | paired Milestone 06 cohort (284 athletes); each outcome on its eligible athletes |
| context_transfer | AM07B_11_m06_vs_m07_evidence | `reports/figures/advanced_movement/context_transfer/AM07B_11_m06_vs_m07_evidence.png` | transfer_paired_comparisons (06B) + aligned_transfer_paired_uncertainty | paired Milestone 06 cohort (284 athletes); each outcome on its eligible athletes |
| context_transfer | AM07C_01_power_vs_effect | `reports/figures/advanced_movement/context_transfer/AM07C_01_power_vs_effect.png` | transfer_power_scenarios | 284 athletes · 6800 synthetic replicates |
| context_transfer | AM07C_02_minimum_detectable_delta_rmse | `reports/figures/advanced_movement/context_transfer/AM07C_02_minimum_detectable_delta_rmse.png` | transfer_mde_estimates | 284 athletes · 6800 synthetic replicates |
| context_transfer | AM07C_03_false_positive_rate | `reports/figures/advanced_movement/context_transfer/AM07C_03_false_positive_rate.png` | transfer_null_calibration | 284 athletes · 6800 synthetic replicates |
| context_transfer | AM07C_04_interval_coverage | `reports/figures/advanced_movement/context_transfer/AM07C_04_interval_coverage.png` | transfer_power_scenarios | 284 athletes · 6800 synthetic replicates |
| context_transfer | AM07C_05_power_by_position | `reports/figures/advanced_movement/context_transfer/AM07C_05_power_by_position.png` | transfer_power_scenarios | 284 athletes · 6800 synthetic replicates |
| context_transfer | AM07C_06_reliability_attenuation | `reports/figures/advanced_movement/context_transfer/AM07C_06_reliability_attenuation.png` | attenuation approximation + persisted reliabilities | analytic |
| context_transfer | AM07C_07_sample_size_vs_detectable | `reports/figures/advanced_movement/context_transfer/AM07C_07_sample_size_vs_detectable.png` | transfer_power_scenarios | resampled cohorts (scenario analysis) |
| context_transfer | AM07C_08_collinearity_vs_incremental_signal | `reports/figures/advanced_movement/context_transfer/AM07C_08_collinearity_vs_incremental_signal.png` | transfer_power_scenarios + representation blocks (06B) | 284 athletes · 6800 synthetic replicates |
| context_transfer | AM07C_09_observed_vs_detectable_M2 | `reports/figures/advanced_movement/context_transfer/AM07C_09_observed_vs_detectable_M2.png` | transfer_planted_signal_recovery + transfer_paired_comparisons (06B) | 284 athletes · 6800 synthetic replicates |
| context_transfer | AM07C_09_observed_vs_detectable_M5 | `reports/figures/advanced_movement/context_transfer/AM07C_09_observed_vs_detectable_M5.png` | transfer_planted_signal_recovery + transfer_paired_comparisons (06B) | 284 athletes · 6800 synthetic replicates |
| football_v1 | AM08B0_01_definition_source_classes | `reports/figures/advanced_movement/football_v1/AM08B0_01_definition_source_classes.png` | football_outcome_definition_classes | linked athletes per definition |
| football_v1 | AM08B0_02_registry_decision | `reports/figures/advanced_movement/football_v1/AM08B0_02_registry_decision.png` | football_outcome_registry | definitions |
| football_v1 | AM08B1_01_outcome_distribution_by_position | `reports/figures/advanced_movement/football_v1/AM08B1_01_outcome_distribution_by_position.png` | football_outcome_rows | recorded player-plays |
| football_v1 | AM08B1_02_game_to_game_variability | `reports/figures/advanced_movement/football_v1/AM08B1_02_game_to_game_variability.png` | football_athlete_outcomes | supported athletes |
| football_v1 | AM08B1_03_targeted_versus_not_targeted | `reports/figures/advanced_movement/football_v1/AM08B1_03_targeted_versus_not_targeted.png` | football_outcome_rows | recorded player-plays with a target flag |
| football_v1 | AM08B1_04_separation_by_route_alignment | `reports/figures/advanced_movement/football_v1/AM08B1_04_separation_by_route_alignment.png` | football_context_cells | rows and athletes per cell |
| football_v1 | AM08B1_05_reliability_versus_opportunities | `reports/figures/advanced_movement/football_v1/AM08B1_05_reliability_versus_opportunities.png` | football_reliability_vs_opportunities | athletes kept at each threshold |
| football_v1 | AM08B1_06_reliability_by_position | `reports/figures/advanced_movement/football_v1/AM08B1_06_reliability_by_position.png` | football_reliability_game_splits | athletes with enough observations in each half |
| football_v1 | AM08B1_07_raw_versus_context_ranking | `reports/figures/advanced_movement/football_v1/AM08B1_07_raw_versus_context_ranking.png` | football_context_conditioning | supported athletes |
| football_v1 | AM08B1_08_tackle_assist_attribution | `reports/figures/advanced_movement/football_v1/AM08B1_08_tackle_assist_attribution.png` | football_tackle_attribution | eligible defensive snaps |
| football_v1 | AM08B1_09_sack_exposure_sensitivity | `reports/figures/advanced_movement/football_v1/AM08B1_09_sack_exposure_sensitivity.png` | football_sack_exposure | supported athletes |
| football_v1 | AM08B1_10_linked_cohort_funnel | `reports/figures/advanced_movement/football_v1/AM08B1_10_linked_cohort_funnel.png` | football_outcome_linked_cohort (08A.2) | athletes per stage |
| football_v1 | AM08B2_01_model_rmse_comparison | `reports/figures/advanced_movement/football_v1/AM08B2_01_model_rmse_comparison.png` | football_transfer_* tables | 84 athletes (TE 20 · WR 64) |
| football_v1 | AM08B2_02_paired_delta_rmse_forest | `reports/figures/advanced_movement/football_v1/AM08B2_02_paired_delta_rmse_forest.png` | football_transfer_* tables | 84 athletes (TE 20 · WR 64) |
| football_v1 | AM08B2_03_predicted_versus_observed | `reports/figures/advanced_movement/football_v1/AM08B2_03_predicted_versus_observed.png` | football_transfer_* tables | 84 athletes (TE 20 · WR 64) |
| football_v1 | AM08B2_04_calibration | `reports/figures/advanced_movement/football_v1/AM08B2_04_calibration.png` | football_transfer_* tables | 84 athletes (TE 20 · WR 64) |
| football_v1 | AM08B2_05_position_specific_performance | `reports/figures/advanced_movement/football_v1/AM08B2_05_position_specific_performance.png` | football_transfer_* tables | 84 athletes (TE 20 · WR 64) |
| football_v1 | AM08B2_06_cohort_sensitivity | `reports/figures/advanced_movement/football_v1/AM08B2_06_cohort_sensitivity.png` | football_transfer_* tables | 84 athletes (TE 20 · WR 64) |
| football_v1 | AM08B2_07_matched_prospect_concordance | `reports/figures/advanced_movement/football_v1/AM08B2_07_matched_prospect_concordance.png` | football_transfer_* tables | 624 pairs, 210 athletes |
| football_v1 | AM08B2_08_tcn_versus_physical_features | `reports/figures/advanced_movement/football_v1/AM08B2_08_tcn_versus_physical_features.png` | football_transfer_* tables | 84 athletes (TE 20 · WR 64) |
| football_v1 | AM08B2_09_uncertainty_bootstrap_versus_permutation | `reports/figures/advanced_movement/football_v1/AM08B2_09_uncertainty_bootstrap_versus_permutation.png` | football_transfer_* tables | 84 athletes (TE 20 · WR 64) |
| football_v1 | AM08B2_10_baseline_versus_tracking_added_value | `reports/figures/advanced_movement/football_v1/AM08B2_10_baseline_versus_tracking_added_value.png` | football_transfer_* tables | 84 athletes (TE 20 · WR 64) |
| football_v1 | AM08B3_01_sensitivity_forest | `reports/figures/advanced_movement/football_v1/AM08B3_01_sensitivity_forest.png` | football_transfer_sensitivity_comparisons | athletes per analysis (annotated) |
| football_v1 | AM08B3_02_detectability_power_and_sample_size | `reports/figures/advanced_movement/football_v1/AM08B3_02_detectability_power_and_sample_size.png` | transfer_power_scenarios (football cohort) | 84 athletes; 6800 synthetic replicates |
| football_v1 | AM08B3_03_permutation_diagnostics | `reports/figures/advanced_movement/football_v1/AM08B3_03_permutation_diagnostics.png` | football_transfer_permutation_draws | 84 athletes |
| football_v1 | AM08B3_04_robustness_reading | `reports/figures/advanced_movement/football_v1/AM08B3_04_robustness_reading.png` | football_transfer_robustness_reading | 84 athletes |
| football_v1 | AM08B4_01_reliability_versus_redundancy | `reports/figures/advanced_movement/football_v1/AM08B4_01_reliability_versus_redundancy.png` | movement_incremental_information | 510 Combine athletes; athletes with a repeat per candidate in the table |
| football_v1 | AM08B4_02_residual_reliability | `reports/figures/advanced_movement/football_v1/AM08B4_02_residual_reliability.png` | movement_incremental_information | 510 Combine athletes; athletes with a repeat per candidate in the table |
| football_v1 | AM08B4_03_speed_development_curves | `reports/figures/advanced_movement/football_v1/AM08B4_03_speed_development_curves.png` | sequence_frames + movement_official_adjusted_values | 102 DB athletes |
| football_v1 | AM08B4_04_matched_stopwatch_athletes | `reports/figures/advanced_movement/football_v1/AM08B4_04_matched_stopwatch_athletes.png` | movement_matched_distinguishability | matched pairs per characteristic (annotated) |
| football_v1 | AM08B4_05_tcn_residual_discrimination | `reports/figures/advanced_movement/football_v1/AM08B4_05_tcn_residual_discrimination.png` | movement_incremental_information | 510 Combine athletes; athletes with a repeat per candidate in the table |
| football_v1 | AM08B4_06_position_specific_evidence | `reports/figures/advanced_movement/football_v1/AM08B4_06_position_specific_evidence.png` | movement_information_sensitivity | athletes with a repeat (annotated) |
| football_v1 | AM08B4_07_acquisition_robustness | `reports/figures/advanced_movement/football_v1/AM08B4_07_acquisition_robustness.png` | movement_information_sensitivity | athletes with a repeat (annotated) |
| football_v1 | AM08B4_08_additional_information_matrix | `reports/figures/advanced_movement/football_v1/AM08B4_08_additional_information_matrix.png` | movement_incremental_information | 510 Combine athletes; athletes with a repeat per candidate in the table |
| football_v1 | AM08B5_01_scouting_scorecard_56096 | `reports/figures/advanced_movement/football_v1/AM08B5_01_scouting_scorecard_56096.png` | 07D scorecard tables + scouting_evidence_states | one athlete; reference sizes in the table |
| football_v1 | AM08B5_01_scouting_scorecard_57226 | `reports/figures/advanced_movement/football_v1/AM08B5_01_scouting_scorecard_57226.png` | 07D scorecard tables + scouting_evidence_states | one athlete; reference sizes in the table |
| football_v1 | AM08B5_01_scouting_scorecard_57334 | `reports/figures/advanced_movement/football_v1/AM08B5_01_scouting_scorecard_57334.png` | 07D scorecard tables + scouting_evidence_states | one athlete; reference sizes in the table |
| football_v1 | AM08B5_01_scouting_scorecard_58330 | `reports/figures/advanced_movement/football_v1/AM08B5_01_scouting_scorecard_58330.png` | 07D scorecard tables + scouting_evidence_states | one athlete; reference sizes in the table |
| football_v1 | AM08B5_02_two_player_comparison_high_versus_low_confidence | `reports/figures/advanced_movement/football_v1/AM08B5_02_two_player_comparison_high_versus_low_confidence.png` | scouting_pairwise_uncertainty | two athletes |
| football_v1 | AM08B5_02_two_player_comparison_similar_body_different_fingerprint | `reports/figures/advanced_movement/football_v1/AM08B5_02_two_player_comparison_similar_body_different_fingerprint.png` | scouting_pairwise_uncertainty | two athletes |
| football_v1 | AM08B5_02_two_player_comparison_similar_forty_different_development | `reports/figures/advanced_movement/football_v1/AM08B5_02_two_player_comparison_similar_forty_different_development.png` | scouting_pairwise_uncertainty | two athletes |
| football_v1 | AM08B5_02_two_player_comparison_similar_top_speed_different_curve | `reports/figures/advanced_movement/football_v1/AM08B5_02_two_player_comparison_similar_top_speed_different_curve.png` | scouting_pairwise_uncertainty | two athletes |
| football_v1 | AM08B5_02_two_player_comparison_stable_versus_unstable_attempts | `reports/figures/advanced_movement/football_v1/AM08B5_02_two_player_comparison_stable_versus_unstable_attempts.png` | scouting_pairwise_uncertainty | two athletes |
| football_v1 | AM08B5_03_observed_sprint_curve_overlay | `reports/figures/advanced_movement/football_v1/AM08B5_03_observed_sprint_curve_overlay.png` | sequence_frames | recorded attempts of two athletes |
| football_v1 | AM08B5_04_pairwise_uncertainty | `reports/figures/advanced_movement/football_v1/AM08B5_04_pairwise_uncertainty.png` | scouting_pairwise_uncertainty | 10 comparison cases |
| football_v1 | AM08B5_05_evidence_confidence | `reports/figures/advanced_movement/football_v1/AM08B5_05_evidence_confidence.png` | combine_measurement_confidence + scouting_evidence_states | 510 athletes |
| football_v1 | AM08B5_06_reference_population_sensitivity | `reports/figures/advanced_movement/football_v1/AM08B5_06_reference_population_sensitivity.png` | scouting_reference_sensitivity | athletes scored (annotated) |
| football_v1 | AM08B5_07_repeat_attempt_stability | `reports/figures/advanced_movement/football_v1/AM08B5_07_repeat_attempt_stability.png` | scouting_attempt_percentile_shifts | athletes with two valid attempts |
| football_v1 | AM08B5_08_supported_unsupported_information | `reports/figures/advanced_movement/football_v1/AM08B5_08_supported_unsupported_information.png` | scouting_information_summary | characteristics of the 08B.4 registry |
| nfl_outcomes | AM05A_01_nfl_tracking_cohort | `reports/figures/advanced_movement/nfl_outcomes/AM05A_01_nfl_tracking_cohort.png` | nfl_play_movement | 401 athletes |
| nfl_outcomes | AM05A_02_eligible_movement_windows | `reports/figures/advanced_movement/nfl_outcomes/AM05A_02_eligible_movement_windows.png` | nfl_window_funnel, nfl_play_movement | 242043 windows, 401 athletes |
| nfl_outcomes | AM05A_03_speed_distributions_by_position | `reports/figures/advanced_movement/nfl_outcomes/AM05A_03_speed_distributions_by_position.png` | nfl_play_movement | 242043 windows, 401 athletes |
| nfl_outcomes | AM05A_04_acceleration_distributions_by_position | `reports/figures/advanced_movement/nfl_outcomes/AM05A_04_acceleration_distributions_by_position.png` | nfl_play_movement | 242043 windows, 401 athletes |
| nfl_outcomes | AM05A_05_speed_acceleration_phase_space | `reports/figures/advanced_movement/nfl_outcomes/AM05A_05_speed_acceleration_phase_space.png` | nfl_phase_space | 15592605 frames, 401 athletes |
| nfl_outcomes | AM05A_06_starting_speed_coverage | `reports/figures/advanced_movement/nfl_outcomes/AM05A_06_starting_speed_coverage.png` | nfl_play_movement | 242043 windows, 401 athletes |
| nfl_outcomes | AM05A_07_movement_opportunities_by_alignment | `reports/figures/advanced_movement/nfl_outcomes/AM05A_07_movement_opportunities_by_alignment.png` | nfl_play_movement | 242043 windows, 401 athletes |
| nfl_outcomes | AM05A_08a_game_vs_athlete_variability | `reports/figures/advanced_movement/nfl_outcomes/AM05A_08a_game_vs_athlete_variability.png` | nfl_athlete_game_outcomes | 401 athletes |
| nfl_outcomes | AM05A_08b_game_vs_athlete_variability | `reports/figures/advanced_movement/nfl_outcomes/AM05A_08b_game_vs_athlete_variability.png` | nfl_athlete_game_outcomes | 401 athletes |
| nfl_outcomes | AM05A_09_smoothing_and_boundary_sensitivity | `reports/figures/advanced_movement/nfl_outcomes/AM05A_09_smoothing_and_boundary_sensitivity.png` | nfl_play_movement, nfl_play_movement_variants | 125419 windows |
| nfl_outcomes | AM05A_10_exclusions_and_missingness | `reports/figures/advanced_movement/nfl_outcomes/AM05A_10_exclusions_and_missingness.png` | nfl_play_movement | 251089 player-plays |
| nfl_outcomes | AM05A_11_example_movement_sequences | `reports/figures/advanced_movement/nfl_outcomes/AM05A_11_example_movement_sequences.png` | silver.game_tracking (recomputed), nfl_play_movement | 10 recorded player-plays |
| nfl_outcomes | AM05A_12_tracked_exposure_vs_opportunity | `reports/figures/advanced_movement/nfl_outcomes/AM05A_12_tracked_exposure_vs_opportunity.png` | nfl_athlete_outcomes, nfl_outcome_coverage | 401 athletes |
| nfl_outcomes | AM05B_01_reliability_across_games | `reports/figures/advanced_movement/nfl_outcomes/AM05B_01_reliability_across_games.png` | nfl_outcome_reliability, nfl_outcome_screen | up to 336 athletes |
| nfl_outcomes | AM05B_02_reliability_by_position_group | `reports/figures/advanced_movement/nfl_outcomes/AM05B_02_reliability_by_position_group.png` | nfl_outcome_reliability | up to 336 athletes |
| nfl_outcomes | AM05B_03_reliability_vs_exposure | `reports/figures/advanced_movement/nfl_outcomes/AM05B_03_reliability_vs_exposure.png` | nfl_reliability_vs_exposure | up to 336 athletes |
| nfl_outcomes | AM05B_04_variance_components | `reports/figures/advanced_movement/nfl_outcomes/AM05B_04_variance_components.png` | nfl_outcome_reliability | up to 336 athletes |
| nfl_outcomes | AM05B_05_context_share | `reports/figures/advanced_movement/nfl_outcomes/AM05B_05_context_share.png` | nfl_outcome_context_audit | 242043 windows, 401 athletes |
| nfl_outcomes | AM05B_06_outcome_sensitivity | `reports/figures/advanced_movement/nfl_outcomes/AM05B_06_outcome_sensitivity.png` | nfl_outcome_quality_sensitivity | up to 344 athletes |
| nfl_outcomes | AM05B_07_outcome_redundancy | `reports/figures/advanced_movement/nfl_outcomes/AM05B_07_outcome_redundancy.png` | nfl_outcome_redundancy | up to 336 athletes |
| nfl_outcomes | AM05B_08_outcome_screen | `reports/figures/advanced_movement/nfl_outcomes/AM05B_08_outcome_screen.png` | nfl_outcome_screen | 21 outcomes |
| nfl_outcomes | AM05B_09_first_windows_vs_later_games | `reports/figures/advanced_movement/nfl_outcomes/AM05B_09_first_windows_vs_later_games.png` | nfl_play_movement | 324 athletes |
| nfl_outcomes | AM05B_10_season_stability | `reports/figures/advanced_movement/nfl_outcomes/AM05B_10_season_stability.png` | nfl_outcome_reliability | up to 176 athletes |
| scoring | AM07D_01_scorecard_55891 | `reports/figures/advanced_movement/scoring/AM07D_01_scorecard_55891.png` | combine scorecard tables | 1 athlete; position reference 94 athletes |
| scoring | AM07D_01_scorecard_56250 | `reports/figures/advanced_movement/scoring/AM07D_01_scorecard_56250.png` | combine scorecard tables | 1 athlete; position reference 103 athletes |
| scoring | AM07D_01_scorecard_57141 | `reports/figures/advanced_movement/scoring/AM07D_01_scorecard_57141.png` | combine scorecard tables | 1 athlete; position reference 94 athletes |
| scoring | AM07D_01_scorecard_57355 | `reports/figures/advanced_movement/scoring/AM07D_01_scorecard_57355.png` | combine scorecard tables | 1 athlete; position reference 96 athletes |
| scoring | AM07D_01_scorecard_58239 | `reports/figures/advanced_movement/scoring/AM07D_01_scorecard_58239.png` | combine scorecard tables | 1 athlete; position reference 96 athletes |
| scoring | AM07D_01_scorecard_58303 | `reports/figures/advanced_movement/scoring/AM07D_01_scorecard_58303.png` | combine scorecard tables | 1 athlete; position reference 94 athletes |
| scoring | AM07D_02_matched_prospects_01_similar_forty_different_development | `reports/figures/advanced_movement/scoring/AM07D_02_matched_prospects_01_similar_forty_different_development.png` | combine percentile tables + sequence_frames | 2 athletes (illustration) |
| scoring | AM07D_02_matched_prospects_02_similar_top_speed_different_curve | `reports/figures/advanced_movement/scoring/AM07D_02_matched_prospects_02_similar_top_speed_different_curve.png` | combine percentile tables + sequence_frames | 2 athletes (illustration) |
| scoring | AM07D_02_matched_prospects_03_repeatable_fingerprint_difference | `reports/figures/advanced_movement/scoring/AM07D_02_matched_prospects_03_repeatable_fingerprint_difference.png` | combine percentile tables + sequence_frames | 2 athletes (illustration) |
| scoring | AM07D_02_matched_prospects_04_physical_agrees_fingerprint_differs | `reports/figures/advanced_movement/scoring/AM07D_02_matched_prospects_04_physical_agrees_fingerprint_differs.png` | combine percentile tables + sequence_frames | 2 athletes (illustration) |
| scoring | AM07D_02_matched_prospects_05_high_versus_low_recording_confidence | `reports/figures/advanced_movement/scoring/AM07D_02_matched_prospects_05_high_versus_low_recording_confidence.png` | combine percentile tables + sequence_frames | 2 athletes (illustration) |
| scoring | AM07D_03_position_percentile_distributions | `reports/figures/advanced_movement/scoring/AM07D_03_position_percentile_distributions.png` | combine percentile tables | 510 Combine athletes |
| scoring | AM07D_04_speed_development_curves | `reports/figures/advanced_movement/scoring/AM07D_04_speed_development_curves.png` | sequence_frames + combine_speed_development_percentiles | 95 DB athletes |
| scoring | AM07D_05_uncertainty_p95_speed | `reports/figures/advanced_movement/scoring/AM07D_05_uncertainty_p95_speed.png` | combine percentile tables | athletes with a percentile |
| scoring | AM07D_05_uncertainty_time_v4_to_v8_s | `reports/figures/advanced_movement/scoring/AM07D_05_uncertainty_time_v4_to_v8_s.png` | combine percentile tables | athletes with a percentile |
| scoring | AM07D_06_fingerprint_similarity | `reports/figures/advanced_movement/scoring/AM07D_06_fingerprint_similarity.png` | transfer_attempt_representations_outer (06B) | 418 athletes |
| scoring | AM07D_07_quality_vs_ranking_stability | `reports/figures/advanced_movement/scoring/AM07D_07_quality_vs_ranking_stability.png` | atlas_attempt_features + combine_measurement_confidence | athletes with a repeat |
| scouting_v1 | AM08A2_01_outcome_semantic_status | `reports/figures/advanced_movement/scouting_v1/AM08A2_01_outcome_semantic_status.png` | football_outcome_semantics_audit | 3 outcomes |
| scouting_v1 | AM08A2_02_missing_versus_zero | `reports/figures/advanced_movement/scouting_v1/AM08A2_02_missing_versus_zero.png` | football_outcome_zero_missing | 242,043 eligible player-plays |
| scouting_v1 | AM08A2_03_separation_distribution | `reports/figures/advanced_movement/scouting_v1/AM08A2_03_separation_distribution.png` | silver.player_play.separation_at_pass_forward (eligible WR / TE player-plays) | 37,398 recorded separations · 116 athletes |
| scouting_v1 | AM08A2_04_separation_by_target | `reports/figures/advanced_movement/scouting_v1/AM08A2_04_separation_by_target.png` | football_separation_distribution | 37,398 recorded separations · 116 athletes |
| scouting_v1 | AM08A2_05_separation_by_position | `reports/figures/advanced_movement/scouting_v1/AM08A2_05_separation_by_position.png` | football_separation_distribution | 37,398 recorded separations · 116 athletes |
| scouting_v1 | AM08A2_06_route_alignment_coverage | `reports/figures/advanced_movement/scouting_v1/AM08A2_06_route_alignment_coverage.png` | football_separation_context_cells | 37,398 recorded separations with a route label |
| scouting_v1 | AM08A2_07_tackle_opportunity_structure | `reports/figures/advanced_movement/scouting_v1/AM08A2_07_tackle_opportunity_structure.png` | football_outcome_structure | eligible DB / DL player-plays |
| scouting_v1 | AM08A2_08_sack_opportunity_structure | `reports/figures/advanced_movement/scouting_v1/AM08A2_08_sack_opportunity_structure.png` | football_outcome_structure | eligible DL player-plays |
| scouting_v1 | AM08A2_09_outcome_reliability | `reports/figures/advanced_movement/scouting_v1/AM08A2_09_outcome_reliability.png` | football_outcome_reliability | 273 supported athletes over all denominators |
| scouting_v1 | AM08A2_09b_odd_even_game_values | `reports/figures/advanced_movement/scouting_v1/AM08A2_09b_odd_even_game_values.png` | football_outcome_athlete_values | 273 supported athletes over all denominators |
| scouting_v1 | AM08A2_10_reliability_versus_exposure | `reports/figures/advanced_movement/scouting_v1/AM08A2_10_reliability_versus_exposure.png` | football_outcome_reliability_exposure | 273 supported athletes over all denominators |
| scouting_v1 | AM08A2_11_linked_cohort_funnel | `reports/figures/advanced_movement/scouting_v1/AM08A2_11_linked_cohort_funnel.png` | football_outcome_linked_cohort | 510 Combine athletes in the Milestone 06 manifest |
| scouting_v1 | AM08A2_12_position_cohort_support | `reports/figures/advanced_movement/scouting_v1/AM08A2_12_position_cohort_support.png` | football_outcome_cohort_support | linked athletes with NFL support |
| scouting_v1 | AM08A2_13_definition_ranking_sensitivity | `reports/figures/advanced_movement/scouting_v1/AM08A2_13_definition_ranking_sensitivity.png` | football_outcome_ranking_sensitivity | 273 supported athletes over all denominators |
| scouting_v1 | AM08A2_14_outcome_freeze_decision | `reports/figures/advanced_movement/scouting_v1/AM08A2_14_outcome_freeze_decision.png` | football_outcome_screen | 11 denominators |
| scouting_v1 | AM08A_01_outcome_coverage_by_position | `reports/figures/advanced_movement/scouting_v1/AM08A_01_outcome_coverage_by_position.png` | football_outcome_coverage | 401 athletes |
| scouting_v1 | AM08A_02_outcome_feasibility_status | `reports/figures/advanced_movement/scouting_v1/AM08A_02_outcome_feasibility_status.png` | football_outcome_feasibility_registry | 16 candidates |
| transfer | AM06A_01_combine_to_nfl_cohort_funnel | `reports/figures/advanced_movement/transfer/AM06A_01_combine_to_nfl_cohort_funnel.png` | transfer_cohort_manifest (notebook 06A) | 510 Combine athletes, 284 in the primary cohort |
| transfer | AM06A_02_attrition_by_position | `reports/figures/advanced_movement/transfer/AM06A_02_attrition_by_position.png` | transfer_cohort_manifest (notebook 06A) | 510 Combine athletes |
| transfer | AM06A_03_attrition_by_draft_year | `reports/figures/advanced_movement/transfer/AM06A_03_attrition_by_draft_year.png` | transfer_cohort_manifest (notebook 06A) | 510 Combine athletes |
| transfer | AM06A_04_official_test_missingness | `reports/figures/advanced_movement/transfer/AM06A_04_official_test_missingness.png` | transfer_missingness_audit | 336 target-exposed athletes |
| transfer | AM06A_05_nfl_outcome_exposure_distribution | `reports/figures/advanced_movement/transfer/AM06A_05_nfl_outcome_exposure_distribution.png` | transfer_cohort_manifest (notebook 06A) | 401 NFL-tracked athletes |
| transfer | AM06A_06_eligible_game_count_distribution | `reports/figures/advanced_movement/transfer/AM06A_06_eligible_game_count_distribution.png` | transfer_cohort_manifest (notebook 06A) | 401 NFL-tracked athletes |
| transfer | AM06A_07_analytical_cohort_vs_excluded | `reports/figures/advanced_movement/transfer/AM06A_07_analytical_cohort_vs_excluded.png` | transfer_cohort_manifest (notebook 06A) + official Combine measurements | 418 athletes with a tracked forty |
| transfer | AM06A_08_target_availability_by_position_and_cohort | `reports/figures/advanced_movement/transfer/AM06A_08_target_availability_by_position_and_cohort.png` | transfer_target_eligibility | 401 NFL-tracked athletes |
| transfer | AM06A_09_fold_population_balance | `reports/figures/advanced_movement/transfer/AM06A_09_fold_population_balance.png` | transfer_outer_folds | 284 primary-cohort athletes |
| transfer | AM06A_10_representation_checkpoint_compatibility | `reports/figures/advanced_movement/transfer/AM06A_10_representation_checkpoint_compatibility.png` | representation_checkpoint_compatibility | 46 saved fits, 5 transfer folds |
| transfer | AM06B_01_actual_vs_predicted_nfl_acceleration | `reports/figures/advanced_movement/transfer/AM06B_01_actual_vs_predicted_nfl_acceleration.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_02_model_rmse_comparison | `reports/figures/advanced_movement/transfer/AM06B_02_model_rmse_comparison.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_03_delta_rmse_relative_to_m1 | `reports/figures/advanced_movement/transfer/AM06B_03_delta_rmse_relative_to_m1.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_04_model_mae_comparison | `reports/figures/advanced_movement/transfer/AM06B_04_model_mae_comparison.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_05_out_of_sample_r2 | `reports/figures/advanced_movement/transfer/AM06B_05_out_of_sample_r2.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_06_spearman_ranking_comparison | `reports/figures/advanced_movement/transfer/AM06B_06_spearman_ranking_comparison.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_07_nfl_speed_vs_acceleration_transfer | `reports/figures/advanced_movement/transfer/AM06B_07_nfl_speed_vs_acceleration_transfer.png` | transfer_oof_predictions (notebook 06B) | each frozen target on its own paired cohort |
| transfer | AM06B_08_tcn_vs_speed2_vs_fpca | `reports/figures/advanced_movement/transfer/AM06B_08_tcn_vs_speed2_vs_fpca.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_09_residual_distributions | `reports/figures/advanced_movement/transfer/AM06B_09_residual_distributions.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_10_calibration | `reports/figures/advanced_movement/transfer/AM06B_10_calibration.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_11_performance_by_fold | `reports/figures/advanced_movement/transfer/AM06B_11_performance_by_fold.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06B_12_coverage_vs_prediction_error | `reports/figures/advanced_movement/transfer/AM06B_12_coverage_vs_prediction_error.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06C_01_position_specific_model_gains | `reports/figures/advanced_movement/transfer/AM06C_01_position_specific_model_gains.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06C_02_cohort_specific_performance | `reports/figures/advanced_movement/transfer/AM06C_02_cohort_specific_performance.png` | transfer_oof_predictions (notebook 06B) + leave-one-cohort-out refits | 284 athletes of the paired cohort |
| transfer | AM06C_03_exposure_sensitivity | `reports/figures/advanced_movement/transfer/AM06C_03_exposure_sensitivity.png` | transfer_paired_comparisons_sens | each rule on its own eligible athletes |
| transfer | AM06C_04_nfl_context_robustness | `reports/figures/advanced_movement/transfer/AM06C_04_nfl_context_robustness.png` | transfer_paired_comparisons_sens | each rule on its own eligible athletes |
| transfer | AM06C_05_matched_stopwatch_player_comparisons | `reports/figures/advanced_movement/transfer/AM06C_05_matched_stopwatch_player_comparisons.png` | transfer_matched_pair_predictions | 833 matched pairs |
| transfer | AM06C_06_combine_acceleration_vs_nfl_acceleration | `reports/figures/advanced_movement/transfer/AM06C_06_combine_acceleration_vs_nfl_acceleration.png` | atlas_attempt_features + transfer_oof_targets | 284 athletes |
| transfer | AM06C_07_model_predicted_athlete_ranking_differences | `reports/figures/advanced_movement/transfer/AM06C_07_model_predicted_athlete_ranking_differences.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06C_08_nfl_outcome_residuals_by_position | `reports/figures/advanced_movement/transfer/AM06C_08_nfl_outcome_residuals_by_position.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06C_09_confidence_interval_stability | `reports/figures/advanced_movement/transfer/AM06C_09_confidence_interval_stability.png` | transfer_oof_predictions (notebook 06B) | 284 athletes of the paired cohort |
| transfer | AM06C_10_final_evidence_strength_summary | `reports/figures/advanced_movement/transfer/AM06C_10_final_evidence_strength_summary.png` | transfer_evidence_summary | 284 athletes of the paired cohort |