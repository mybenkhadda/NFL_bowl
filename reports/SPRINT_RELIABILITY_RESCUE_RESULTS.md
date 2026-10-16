# Sprint reliability rescue — what does the stopwatch miss?

_Template: `reports/SPRINT_RELIABILITY_RESCUE_TEMPLATE.md`. Filled by `python -m nflbdb.research.sprint_rescue --report` into `reports/SPRINT_RELIABILITY_RESCUE_RESULTS.md`. Every table below is read from the persisted outputs of this experiment; text outside the tables states the design and facts read from source code or from earlier executed outputs. Interpretation sentences are left to the review._

Generated 2026-10-11T12:21:44Z from persisted outputs.

| output | rows | status |
|---|---|---|
| attempt_censoring_diagnostics | 793 | present |
| sprint_model_fit_diagnostics | 5,551 | present |
| recording_window_distribution | 384 | present |
| within_athlete_window_variation | 375 | present |
| example_signals | 212 | present |
| censoring_association | 438 | present |
| within_position_reliability | 704 | present |
| sprint_model_repeatability | 156 | present |
| athlete_sprint_features | 836 | present |
| combine_timing_redundancy | 1,440 | present |
| feature_screening_summary | 16 | present |
| athlete_cohort_reconciliation | 30 | present |
| dl_getoff_feasibility | 259 | present |


## 1. Research hypothesis

**Question.** Is there a Combine sprint measurement that (a) is repeatable within a position group, (b) separates athletes of the same position, and (c) carries information that the official forty and 10-yard split do not?

**Hypothesis under test (Study A).** Variation in the recorded sprint window between an athlete's attempts contributes to the poor repeatability of acceleration measurements.

Three things are measured rather than assumed: whether the low acceleration ICC is related to the recording window; whether sprint-model parameters are more repeatable than direct acceleration; whether speed is redundant with the official forty.

## 2. Existing evidence

Executed before this experiment (Phase 04 Study B, `reports/tables/phase_04/reliability/B_repeatability_by_setting.csv`, forty-yard dash, primary smoothing sg7p2, 375 athletes with two attempts each, pooled over positions):

| Feature | ICC(1) | 95 % interval |
|---|---|---|
| `p95_speed` | 0.987 | 0.984 – 0.989 |
| `p95_a_tan` | 0.502 | 0.356 – 0.617 |
| `peak_decel` | 0.175 | 0.001 – 0.386 |
| `path_smooth_yds` | 0.306 | 0.200 – 0.381 |

From `reports/tables/phase_03/forty_window_diagnostics.csv` (793 attempts, 418 athletes): 58.5 % of attempts start with the athlete already moving; 28.1 % have peak speed within the last three frames; **the `end_censored` flag is set on 100 % of attempts**, so it cannot distinguish attempts and is replaced here by continuous end-of-recording variables; 95.6 % of attempts have a usable model fit. Phase 04 Study Q: the strictest quality scenario keeps 254 of 375 athletes with repeated attempts.

## 3. Censoring diagnostics

### 3.1 Recording-window distribution (A1)

**Recording-window variables (attempts)**

| group_type | group | variable | n | p05 | p50 | p95 | sd |
|---|---|---|---|---|---|---|---|
| all | ALL | duration_s | 793 | 4.7 | 5 | 5.8 | 0.359 |
| all | ALL | first_speed | 793 | 0.229 | 0.562 | 1.15 | 0.311 |
| all | ALL | last_speed | 793 | 8.88 | 10.5 | 11.4 | 0.829 |
| all | ALL | path_smooth_yds | 793 | 39.6 | 40.5 | 41.3 | 0.547 |
| cohort | 2023 | duration_s | 262 | 4.7 | 5 | 5.7 | 0.39 |
| cohort | 2023 | first_speed | 262 | 0.246 | 0.515 | 0.852 | 0.195 |
| cohort | 2023 | last_speed | 262 | 8.88 | 10.6 | 11.4 | 0.836 |
| cohort | 2023 | path_smooth_yds | 262 | 39.7 | 40.4 | 41 | 0.442 |
| cohort | 2024 | duration_s | 269 | 4.7 | 5 | 5.8 | 0.356 |
| cohort | 2024 | first_speed | 269 | 0.272 | 0.657 | 1.26 | 0.351 |
| cohort | 2024 | last_speed | 269 | 8.82 | 10.4 | 11.3 | 0.821 |
| cohort | 2024 | path_smooth_yds | 269 | 39.4 | 40.3 | 41.5 | 0.651 |
| cohort | 2025 | duration_s | 262 | 4.7 | 5.1 | 5.8 | 0.322 |
| cohort | 2025 | first_speed | 262 | 0.16 | 0.525 | 1.19 | 0.336 |
| cohort | 2025 | last_speed | 262 | 8.99 | 10.5 | 11.5 | 0.827 |
| cohort | 2025 | path_smooth_yds | 262 | 40 | 40.7 | 41.4 | 0.45 |
| position | DB | duration_s | 195 | 4.6 | 4.9 | 5.2 | 0.355 |
| position | DB | first_speed | 195 | 0.263 | 0.578 | 1.2 | 0.298 |
| position | DB | last_speed | 195 | 10.4 | 10.9 | 11.5 | 0.324 |
| position | DB | path_smooth_yds | 195 | 39.5 | 40.4 | 41.2 | 0.594 |
| position | DL | duration_s | 177 | 4.88 | 5.2 | 5.7 | 0.267 |
| position | DL | first_speed | 177 | 0.3 | 0.597 | 1.02 | 0.231 |
| position | DL | last_speed | 177 | 8.79 | 10 | 10.9 | 0.659 |
| position | DL | path_smooth_yds | 177 | 39.6 | 40.4 | 41.2 | 0.492 |
| position | OL | duration_s | 179 | 5.1 | 5.5 | 5.9 | 0.248 |
| position | OL | first_speed | 179 | 0.358 | 0.589 | 1.22 | 0.317 |
| position | OL | last_speed | 179 | 8.6 | 9.24 | 9.88 | 0.387 |
| position | OL | path_smooth_yds | 179 | 39.8 | 40.5 | 41.7 | 0.597 |
| position | TE | duration_s | 66 | 4.8 | 5 | 5.3 | 0.17 |
| position | TE | first_speed | 66 | 0.193 | 0.553 | 1.26 | 0.469 |
| position | TE | last_speed | 66 | 9.81 | 10.4 | 11 | 0.34 |
| position | TE | path_smooth_yds | 66 | 39.5 | 40.4 | 41.4 | 0.554 |
| position | WR | duration_s | 176 | 4.7 | 4.9 | 5.3 | 0.192 |
| position | WR | first_speed | 176 | 0.11 | 0.43 | 1.06 | 0.295 |
| position | WR | last_speed | 176 | 10.5 | 11 | 11.5 | 0.324 |
| position | WR | path_smooth_yds | 176 | 40.1 | 40.7 | 41.2 | 0.407 |

**Flag shares (attempts)**

| group_type | group | variable | n | share |
|---|---|---|---|---|
| all | ALL | band_start_censored | 793 | 0.00378 |
| all | ALL | boundary_peak | 793 | 0.154 |
| all | ALL | end_censored | 793 | 1 |
| all | ALL | fit_ok | 793 | 0.956 |
| all | ALL | onset_censored | 793 | 0.585 |
| all | ALL | peak_at_end | 793 | 0.281 |
| all | ALL | signal_flag | 793 | 0.0731 |
| all | ALL | structural_flag | 793 | 0.0101 |
| all | ALL | weakly_identified | 793 | 0.0164 |
| cohort | 2023 | band_start_censored | 262 | 0 |
| cohort | 2023 | boundary_peak | 262 | 0.0725 |
| cohort | 2023 | end_censored | 262 | 1 |
| cohort | 2023 | fit_ok | 262 | 0.977 |
| cohort | 2023 | onset_censored | 262 | 0.534 |
| cohort | 2023 | peak_at_end | 262 | 0.336 |
| cohort | 2023 | signal_flag | 262 | 0 |
| cohort | 2023 | structural_flag | 262 | 0 |
| cohort | 2023 | weakly_identified | 262 | 0.00763 |
| cohort | 2024 | band_start_censored | 269 | 0.00743 |
| cohort | 2024 | boundary_peak | 269 | 0.19 |
| cohort | 2024 | end_censored | 269 | 1 |
| cohort | 2024 | fit_ok | 269 | 0.929 |
| cohort | 2024 | onset_censored | 269 | 0.717 |
| cohort | 2024 | peak_at_end | 269 | 0.216 |
| cohort | 2024 | signal_flag | 269 | 0.171 |
| cohort | 2024 | structural_flag | 269 | 0.0186 |
| cohort | 2024 | weakly_identified | 269 | 0.0223 |
| cohort | 2025 | band_start_censored | 262 | 0.00382 |
| cohort | 2025 | boundary_peak | 262 | 0.198 |
| cohort | 2025 | end_censored | 262 | 1 |
| cohort | 2025 | fit_ok | 262 | 0.962 |
| cohort | 2025 | onset_censored | 262 | 0.5 |
| cohort | 2025 | peak_at_end | 262 | 0.294 |
| cohort | 2025 | signal_flag | 262 | 0.0458 |
| cohort | 2025 | structural_flag | 262 | 0.0115 |
| cohort | 2025 | weakly_identified | 262 | 0.0191 |
| position | DB | band_start_censored | 195 | 0 |
| position | DB | boundary_peak | 195 | 0.138 |
| position | DB | end_censored | 195 | 1 |
| position | DB | fit_ok | 195 | 0.938 |
| position | DB | onset_censored | 195 | 0.631 |
| position | DB | peak_at_end | 195 | 0.4 |
| position | DB | signal_flag | 195 | 0.133 |
| position | DB | structural_flag | 195 | 0.0256 |
| position | DB | weakly_identified | 195 | 0.0308 |
| position | DL | band_start_censored | 177 | 0 |
| position | DL | boundary_peak | 177 | 0.102 |
| position | DL | end_censored | 177 | 1 |
| position | DL | fit_ok | 177 | 0.977 |
| position | DL | onset_censored | 177 | 0.667 |
| position | DL | peak_at_end | 177 | 0.164 |
| position | DL | signal_flag | 177 | 0.0113 |
| position | DL | structural_flag | 177 | 0 |
| position | DL | weakly_identified | 177 | 0 |
| position | OL | band_start_censored | 179 | 0.0112 |
| position | OL | boundary_peak | 179 | 0.112 |
| position | OL | end_censored | 179 | 1 |
| position | OL | fit_ok | 179 | 0.966 |
| position | OL | onset_censored | 179 | 0.693 |
| position | OL | peak_at_end | 179 | 0.162 |
| position | OL | signal_flag | 179 | 0.0447 |
| position | OL | structural_flag | 179 | 0 |
| position | OL | weakly_identified | 179 | 0.0112 |
| position | TE | band_start_censored | 66 | 0.0152 |
| position | TE | boundary_peak | 66 | 0.136 |
| position | TE | end_censored | 66 | 1 |
| position | TE | fit_ok | 66 | 1 |
| position | TE | onset_censored | 66 | 0.576 |
| position | TE | peak_at_end | 66 | 0.333 |
| position | TE | signal_flag | 66 | 0.0152 |
| position | TE | structural_flag | 66 | 0 |
| position | TE | weakly_identified | 66 | 0 |
| position | WR | band_start_censored | 176 | 0 |
| position | WR | boundary_peak | 176 | 0.273 |
| position | WR | end_censored | 176 | 1 |
| position | WR | fit_ok | 176 | 0.926 |
| position | WR | onset_censored | 176 | 0.347 |
| position | WR | peak_at_end | 176 | 0.369 |
| position | WR | signal_flag | 176 | 0.119 |
| position | WR | structural_flag | 176 | 0.017 |
| position | WR | weakly_identified | 176 | 0.0284 |


Figures: `SR_01_first_frame_speed`, `SR_02_distance_vs_duration`, `SR_05_example_recordings`.

### 3.2 Within-athlete comparison (A2)

One row per athlete: attempt 2 minus attempt 1. `p95_speed` is a negative control (no window effect expected).

**Δ feature against Δ first-frame speed between an athlete's two attempts (Spearman)**

| outcome | adjustment | n | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|
| d_accel_initial | none | 375 | -0.213 | -0.308 | -0.114 |
| d_accel_initial | position+cohort+noise | 375 | -0.214 | -0.31 | -0.115 |
| d_accel_sustained | none | 375 | 0.00838 | -0.093 | 0.11 |
| d_accel_sustained | position+cohort+noise | 375 | 0.0167 | -0.0856 | 0.119 |
| d_accel_v2_5 | none | 372 | 0.00538 | -0.0964 | 0.107 |
| d_accel_v2_5 | position+cohort+noise | 372 | 0.0132 | -0.0895 | 0.116 |
| d_accel_v2_7 | none | 372 | 0.00248 | -0.0992 | 0.104 |
| d_accel_v2_7 | position+cohort+noise | 372 | 0.0153 | -0.0875 | 0.118 |
| d_accel_v4_7 | none | 375 | 0.0277 | -0.0737 | 0.129 |
| d_accel_v4_7 | position+cohort+noise | 375 | 0.0428 | -0.0597 | 0.144 |
| d_model_a0 | none | 345 | 0.0294 | -0.0764 | 0.135 |
| d_model_a0 | position+cohort+noise | 345 | 0.0458 | -0.0612 | 0.152 |
| d_model_a_at_ref | none | 345 | 0.0313 | -0.0745 | 0.136 |
| d_model_a_at_ref | position+cohort+noise | 345 | 0.0481 | -0.0589 | 0.154 |
| d_model_tau_s | none | 345 | -0.0212 | -0.126 | 0.0846 |
| d_model_tau_s | position+cohort+noise | 345 | -0.0388 | -0.145 | 0.0681 |
| d_p95_a_tan | none | 375 | 0.00244 | -0.0989 | 0.104 |
| d_p95_a_tan | position+cohort+noise | 375 | 0.0132 | -0.0892 | 0.115 |
| d_p95_speed | none | 375 | -0.00186 | -0.103 | 0.0994 |
| d_p95_speed | position+cohort+noise | 375 | 0.0084 | -0.0939 | 0.111 |
| d_peak_a_tan | none | 375 | 0.0232 | -0.0783 | 0.124 |
| d_peak_a_tan | position+cohort+noise | 375 | 0.0278 | -0.0747 | 0.13 |

**Share of the variance of Δ feature explained jointly by Δ start speed, Δ duration and Δ path (partial R² beyond position and cohort)**

| outcome | n | estimate | ci_lower | ci_upper | note |
|---|---|---|---|---|---|
| d_accel_initial | 375 | 0.61 | 0.543 | 0.866 | adjusted R² = 0.6073 |
| d_accel_sustained | 375 | 0.16 | 0.082 | 0.477 | adjusted R² = 0.1528 |
| d_accel_v2_5 | 372 | 0.249 | 0.104 | 0.636 | adjusted R² = 0.2425 |
| d_accel_v2_7 | 372 | 0.219 | 0.0823 | 0.6 | adjusted R² = 0.2121 |
| d_accel_v4_7 | 375 | 0.0921 | 0.0368 | 0.351 | adjusted R² = 0.0846 |
| d_model_a0 | 345 | 0.437 | 0.35 | 0.551 | adjusted R² = 0.4319 |
| d_model_a_at_ref | 345 | 0.434 | 0.353 | 0.549 | adjusted R² = 0.4292 |
| d_model_tau_s | 345 | 0.483 | 0.376 | 0.587 | adjusted R² = 0.4785 |
| d_p95_a_tan | 375 | 0.184 | 0.106 | 0.486 | adjusted R² = 0.1778 |
| d_p95_speed | 375 | 0.0382 | 0.0164 | 0.0818 | adjusted R² = 0.0303 |
| d_peak_a_tan | 375 | 0.128 | 0.0604 | 0.41 | adjusted R² = 0.1211 |


Figure: `SR_03_delta_start_vs_delta_accel`.

### 3.3 Association with repeatability and fit quality (A3)

**ICC before and after removing the within-athlete window slope (upper bound of a window correction)**

| outcome | predictor | group | statistic | n | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| accel_initial | start+end | all | icc_gain | 375 | 0.000214 | -0.0886 | 0.0975 |
| accel_initial | start+end | all | icc_raw | 375 | 0.298 | 0.177 | 0.423 |
| accel_initial | start+end | all | icc_window_adjusted | 375 | 0.298 | 0.182 | 0.429 |
| accel_sustained | start+end | all | icc_gain | 375 | -0.145 | -0.22 | -0.0866 |
| accel_sustained | start+end | all | icc_raw | 375 | 0.331 | 0.191 | 0.468 |
| accel_sustained | start+end | all | icc_window_adjusted | 375 | 0.186 | 0.0584 | 0.316 |
| accel_v2_5 | start+end | all | icc_gain | 372 | -0.145 | -0.238 | -0.0648 |
| accel_v2_5 | start+end | all | icc_raw | 372 | 0.333 | 0.199 | 0.458 |
| accel_v2_5 | start+end | all | icc_window_adjusted | 372 | 0.189 | 0.0669 | 0.323 |
| accel_v2_7 | start+end | all | icc_gain | 372 | -0.166 | -0.288 | -0.0773 |
| accel_v2_7 | start+end | all | icc_raw | 372 | 0.436 | 0.286 | 0.576 |
| accel_v2_7 | start+end | all | icc_window_adjusted | 372 | 0.27 | 0.116 | 0.438 |
| accel_v4_7 | start+end | all | icc_gain | 375 | -0.0794 | -0.161 | -0.0318 |
| accel_v4_7 | start+end | all | icc_raw | 375 | 0.539 | 0.394 | 0.674 |
| accel_v4_7 | start+end | all | icc_window_adjusted | 375 | 0.46 | 0.311 | 0.601 |
| model_a0 | start+end | all | icc_gain | 345 | 0.0905 | -0.0135 | 0.195 |
| model_a0 | start+end | all | icc_raw | 345 | 0.345 | 0.168 | 0.563 |
| model_a0 | start+end | all | icc_window_adjusted | 345 | 0.436 | 0.267 | 0.608 |
| model_a_at_ref | start+end | all | icc_gain | 345 | -0.0493 | -0.155 | 0.0545 |
| model_a_at_ref | start+end | all | icc_raw | 345 | 0.376 | 0.194 | 0.599 |
| model_a_at_ref | start+end | all | icc_window_adjusted | 345 | 0.327 | 0.17 | 0.503 |
| model_tau_s | start+end | all | icc_gain | 345 | 0.263 | 0.171 | 0.365 |
| model_tau_s | start+end | all | icc_raw | 345 | 0.472 | 0.343 | 0.574 |
| model_tau_s | start+end | all | icc_window_adjusted | 345 | 0.735 | 0.673 | 0.785 |
| p95_a_tan | start+end | all | icc_gain | 375 | -0.154 | -0.225 | -0.0958 |
| p95_a_tan | start+end | all | icc_raw | 375 | 0.323 | 0.175 | 0.46 |
| p95_a_tan | start+end | all | icc_window_adjusted | 375 | 0.169 | 0.0405 | 0.3 |
| p95_speed | start+end | all | icc_gain | 375 | -0.00127 | -0.00377 | 0.00144 |
| p95_speed | start+end | all | icc_raw | 375 | 0.948 | 0.934 | 0.957 |
| p95_speed | start+end | all | icc_window_adjusted | 375 | 0.947 | 0.933 | 0.956 |
| peak_a_tan | start+end | all | icc_gain | 375 | -0.123 | -0.2 | -0.0642 |
| peak_a_tan | start+end | all | icc_raw | 375 | 0.335 | 0.184 | 0.465 |
| peak_a_tan | start+end | all | icc_window_adjusted | 375 | 0.212 | 0.0721 | 0.343 |

**ICC by how the two recordings of an athlete differ (position-adjusted)**

| outcome | predictor | group | statistic | n | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| model_a0 | athlete_group | boundary:any | icc1 | 91 | 0.369 | 0.0922 | 0.659 |
| model_a0 | athlete_group | boundary:none | icc1 | 254 | 0.339 | 0.143 | 0.585 |
| model_a0 | athlete_group | censor:both_censored | icc1 | 127 | 0.425 | 0.145 | 0.664 |
| model_a0 | athlete_group | censor:both_uncensored | icc1 | 73 | 0.11 | -0.0269 | 0.579 |
| model_a0 | athlete_group | censor:discordant | icc1 | 145 | 0.492 | 0.334 | 0.611 |
| model_a0 | athlete_group | window_change:high | icc1 | 110 | 0.354 | 0.116 | 0.577 |
| model_a0 | athlete_group | window_change:low | icc1 | 116 | 0.17 | -0.0421 | 0.604 |
| model_a0 | athlete_group | window_change:mid | icc1 | 119 | 0.6 | 0.411 | 0.697 |
| p95_a_tan | athlete_group | boundary:any | icc1 | 103 | 0.3 | 0.00937 | 0.549 |
| p95_a_tan | athlete_group | boundary:none | icc1 | 272 | 0.345 | 0.19 | 0.492 |
| p95_a_tan | athlete_group | censor:both_censored | icc1 | 138 | 0.379 | 0.18 | 0.571 |
| p95_a_tan | athlete_group | censor:both_uncensored | icc1 | 79 | 0.156 | -0.1 | 0.526 |
| p95_a_tan | athlete_group | censor:discordant | icc1 | 158 | 0.322 | 0.0921 | 0.509 |
| p95_a_tan | athlete_group | window_change:high | icc1 | 125 | 0.229 | 0.012 | 0.43 |
| p95_a_tan | athlete_group | window_change:low | icc1 | 125 | 0.315 | 0.0305 | 0.58 |
| p95_a_tan | athlete_group | window_change:mid | icc1 | 125 | 0.523 | 0.345 | 0.638 |
| p95_speed | athlete_group | boundary:any | icc1 | 103 | 0.933 | 0.891 | 0.955 |
| p95_speed | athlete_group | boundary:none | icc1 | 272 | 0.952 | 0.937 | 0.962 |
| p95_speed | athlete_group | censor:both_censored | icc1 | 138 | 0.947 | 0.924 | 0.961 |
| p95_speed | athlete_group | censor:both_uncensored | icc1 | 79 | 0.937 | 0.889 | 0.958 |
| p95_speed | athlete_group | censor:discordant | icc1 | 158 | 0.95 | 0.931 | 0.962 |
| p95_speed | athlete_group | window_change:high | icc1 | 125 | 0.937 | 0.907 | 0.954 |
| p95_speed | athlete_group | window_change:low | icc1 | 125 | 0.955 | 0.932 | 0.967 |
| p95_speed | athlete_group | window_change:mid | icc1 | 125 | 0.948 | 0.924 | 0.963 |

**|Δ feature| against |Δ first-frame speed|, adjusted for position, cohort and positional noise**

| outcome | predictor | group | statistic | n | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| abs_d_accel_initial | abs_d_first_speed | all | spearman | 375 | 0.114 | 0.0121 | 0.214 |
| abs_d_accel_sustained | abs_d_first_speed | all | spearman | 375 | 0.04 | -0.0624 | 0.142 |
| abs_d_accel_v2_5 | abs_d_first_speed | all | spearman | 372 | 0.0797 | -0.0232 | 0.181 |
| abs_d_accel_v2_7 | abs_d_first_speed | all | spearman | 372 | 0.015 | -0.0878 | 0.117 |
| abs_d_accel_v4_7 | abs_d_first_speed | all | spearman | 375 | -0.0233 | -0.125 | 0.0791 |
| abs_d_model_a0 | abs_d_first_speed | all | spearman | 345 | 0.0269 | -0.08 | 0.133 |
| abs_d_model_a_at_ref | abs_d_first_speed | all | spearman | 345 | 0.0164 | -0.0905 | 0.123 |
| abs_d_model_tau_s | abs_d_first_speed | all | spearman | 345 | 0.0527 | -0.0542 | 0.159 |
| abs_d_p95_a_tan | abs_d_first_speed | all | spearman | 375 | 0.0481 | -0.0544 | 0.15 |
| abs_d_p95_speed | abs_d_first_speed | all | spearman | 375 | -0.0245 | -0.126 | 0.078 |
| abs_d_peak_a_tan | abs_d_first_speed | all | spearman | 375 | 0.0288 | -0.0736 | 0.131 |

**Model fit quality against the recording window (athlete means)**

| outcome | predictor | group | statistic | n | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| model_rmse | duration_s | all | spearman | 418 | 0.188 | 0.0931 | 0.28 |
| model_rmse | first_speed | all | spearman | 418 | 0.148 | 0.0516 | 0.241 |
| model_rmse | frames_after_peak | all | spearman | 418 | 0.0688 | -0.0281 | 0.164 |
| tau_rel_width | duration_s | all | spearman | 418 | 0.25 | 0.157 | 0.339 |
| tau_rel_width | first_speed | all | spearman | 418 | 0.112 | 0.0153 | 0.206 |
| tau_rel_width | frames_after_peak | all | spearman | 418 | 0.14 | 0.0441 | 0.234 |
| vmax_extrapolation_ratio | duration_s | all | spearman | 418 | 0.197 | 0.102 | 0.288 |
| vmax_extrapolation_ratio | first_speed | all | spearman | 418 | -0.0426 | -0.139 | 0.0543 |
| vmax_extrapolation_ratio | frames_after_peak | all | spearman | 418 | 0.322 | 0.232 | 0.406 |


Figure: `SR_04_censoring_vs_repeatability`.

**How to read.** These are associations. The window-adjusted ICC is an upper bound because the slope is fitted on the same attempts. A gain whose paired interval excludes zero means window variation is compatible with part of the low ICC; it does not show that the window causes it.

### 3.4 Sensitivity to quality scenarios (A4)

**Position-adjusted ICC(1) by quality scenario**

| feature | scenario | n_athletes | icc1 | ci_lower | ci_upper |
|---|---|---|---|---|---|
| accel_initial | S0_all | 375 | 0.298 | 0.171 | 0.421 |
| accel_initial | S1_structural | 369 | 0.314 | 0.201 | 0.422 |
| accel_initial | S2_signal | 333 | 0.342 | 0.231 | 0.452 |
| accel_initial | S3_no_boundary | 254 | 0.428 | 0.29 | 0.558 |
| accel_sustained | S0_all | 375 | 0.331 | 0.192 | 0.471 |
| accel_sustained | S1_structural | 369 | 0.348 | 0.207 | 0.472 |
| accel_sustained | S2_signal | 333 | 0.467 | 0.362 | 0.549 |
| accel_sustained | S3_no_boundary | 254 | 0.478 | 0.359 | 0.571 |
| accel_v2_5 | S0_all | 372 | 0.333 | 0.201 | 0.456 |
| accel_v2_5 | S1_structural | 366 | 0.378 | 0.242 | 0.496 |
| accel_v2_5 | S2_signal | 332 | 0.487 | 0.379 | 0.578 |
| accel_v2_5 | S3_no_boundary | 253 | 0.509 | 0.391 | 0.605 |
| accel_v2_7 | S0_all | 372 | 0.436 | 0.286 | 0.575 |
| accel_v2_7 | S1_structural | 366 | 0.466 | 0.316 | 0.616 |
| accel_v2_7 | S2_signal | 332 | 0.633 | 0.549 | 0.697 |
| accel_v2_7 | S3_no_boundary | 253 | 0.64 | 0.541 | 0.714 |
| accel_v4_7 | S0_all | 375 | 0.539 | 0.387 | 0.674 |
| accel_v4_7 | S1_structural | 369 | 0.541 | 0.392 | 0.674 |
| accel_v4_7 | S2_signal | 333 | 0.685 | 0.608 | 0.741 |
| accel_v4_7 | S3_no_boundary | 254 | 0.686 | 0.598 | 0.753 |
| model_a0 | S0_all | 345 | 0.345 | 0.171 | 0.565 |
| model_a0 | S1_structural | 341 | 0.351 | 0.161 | 0.582 |
| model_a0 | S2_signal | 315 | 0.552 | 0.463 | 0.62 |
| model_a0 | S3_no_boundary | 240 | 0.546 | 0.442 | 0.619 |
| model_a_at_ref | S0_all | 345 | 0.376 | 0.198 | 0.604 |
| model_a_at_ref | S1_structural | 341 | 0.384 | 0.184 | 0.621 |
| model_a_at_ref | S2_signal | 315 | 0.616 | 0.54 | 0.677 |
| model_a_at_ref | S3_no_boundary | 240 | 0.618 | 0.532 | 0.685 |
| model_tau_s | S0_all | 345 | 0.472 | 0.341 | 0.571 |
| model_tau_s | S1_structural | 341 | 0.472 | 0.343 | 0.578 |
| model_tau_s | S2_signal | 315 | 0.514 | 0.418 | 0.589 |
| model_tau_s | S3_no_boundary | 240 | 0.496 | 0.382 | 0.588 |
| model_vmax | S0_all | 345 | 0.871 | 0.834 | 0.898 |
| model_vmax | S1_structural | 341 | 0.872 | 0.838 | 0.898 |
| model_vmax | S2_signal | 315 | 0.895 | 0.867 | 0.914 |
| model_vmax | S3_no_boundary | 240 | 0.899 | 0.87 | 0.92 |
| p95_a_tan | S0_all | 375 | 0.323 | 0.185 | 0.461 |
| p95_a_tan | S1_structural | 369 | 0.35 | 0.215 | 0.478 |
| p95_a_tan | S2_signal | 333 | 0.468 | 0.361 | 0.554 |
| p95_a_tan | S3_no_boundary | 254 | 0.484 | 0.373 | 0.581 |
| p95_decel | S0_all | 375 | 0.266 | 0.166 | 0.373 |
| p95_decel | S1_structural | 369 | 0.265 | 0.164 | 0.378 |
| p95_decel | S2_signal | 333 | 0.331 | 0.218 | 0.419 |
| p95_decel | S3_no_boundary | 254 | 0.345 | 0.207 | 0.455 |
| p95_speed | S0_all | 375 | 0.948 | 0.935 | 0.958 |
| p95_speed | S1_structural | 369 | 0.948 | 0.936 | 0.957 |
| p95_speed | S2_signal | 333 | 0.949 | 0.935 | 0.959 |
| p95_speed | S3_no_boundary | 254 | 0.953 | 0.938 | 0.963 |
| path_smooth_yds | S0_all | 375 | 0.267 | 0.161 | 0.359 |
| path_smooth_yds | S1_structural | 369 | 0.26 | 0.151 | 0.354 |
| path_smooth_yds | S2_signal | 333 | 0.301 | 0.197 | 0.391 |
| path_smooth_yds | S3_no_boundary | 254 | 0.292 | 0.178 | 0.385 |
| peak_a_tan | S0_all | 375 | 0.335 | 0.194 | 0.473 |
| peak_a_tan | S1_structural | 369 | 0.326 | 0.18 | 0.462 |
| peak_a_tan | S2_signal | 333 | 0.457 | 0.355 | 0.54 |
| peak_a_tan | S3_no_boundary | 254 | 0.47 | 0.361 | 0.558 |
| peak_speed | S0_all | 375 | 0.942 | 0.928 | 0.953 |
| peak_speed | S1_structural | 369 | 0.942 | 0.928 | 0.952 |
| peak_speed | S2_signal | 333 | 0.943 | 0.93 | 0.954 |
| peak_speed | S3_no_boundary | 254 | 0.947 | 0.931 | 0.958 |
| peak_speed_interior | S0_all | 375 | 0.945 | 0.932 | 0.956 |
| peak_speed_interior | S1_structural | 369 | 0.946 | 0.933 | 0.956 |
| peak_speed_interior | S2_signal | 333 | 0.947 | 0.933 | 0.957 |
| peak_speed_interior | S3_no_boundary | 254 | 0.951 | 0.935 | 0.961 |


No scenario is applied by default. A feature that needs S3 to become repeatable loses about a third of the athletes with repeated attempts (Phase 04 Q).

## 4. Within-position ICC

**Pooled against within-position ICC(1), all structurally valid attempts**

| feature | family | n_athletes | icc_pooled | icc_within_position | ci_lower | ci_upper | icc_gap_vs_pooled | position_share_of_between | position_dependent | typical_error | cv_within_pct | spearman_first_two | mean_diff_2_minus_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| peak_speed | speed | 375 | 0.985 | 0.942 | 0.928 | 0.953 | 0.0432 | 0.754 | False | 0.0953 | 0.918 | 0.928 | -0.021 |
| peak_speed_interior | speed | 375 | 0.986 | 0.945 | 0.932 | 0.956 | 0.0407 | 0.756 | False | 0.0906 | 0.874 | 0.933 | -0.0162 |
| p95_speed | speed | 375 | 0.987 | 0.948 | 0.935 | 0.958 | 0.0388 | 0.755 | False | 0.0891 | 0.86 | 0.935 | -0.0181 |
| peak_a_tan | acceleration_direct | 375 | 0.507 | 0.335 | 0.194 | 0.473 | 0.171 | 0.509 | True | 0.539 | 9.54 | 0.42 | 0.0231 |
| p95_a_tan | acceleration_direct | 375 | 0.502 | 0.323 | 0.185 | 0.461 | 0.179 | 0.528 | True | 0.505 | 9.49 | 0.423 | 0.00976 |
| p95_decel | deceleration | 375 | 0.313 | 0.266 | 0.166 | 0.373 | 0.0477 | 0.207 | False | 0.255 |  | 0.343 | 0.0433 |
| path_smooth_yds | recording_window | 375 | 0.306 | 0.267 | 0.161 | 0.359 | 0.0389 | 0.173 | False | 0.453 | 1.12 | 0.289 | -0.0477 |
| accel_sustained | acceleration_sustained | 375 | 0.505 | 0.331 | 0.192 | 0.471 | 0.175 | 0.516 | True | 0.496 | 9.23 | 0.428 | 0.0118 |
| accel_initial | acceleration_sustained | 375 | 0.354 | 0.298 | 0.171 | 0.421 | 0.0554 | 0.223 | False | 0.731 |  | 0.432 | -0.0265 |
| accel_v2_5 | acceleration_sustained | 372 | 0.505 | 0.333 | 0.201 | 0.456 | 0.172 | 0.51 | True | 0.515 | 9.94 | 0.436 | 0.0263 |
| accel_v4_7 | acceleration_sustained | 375 | 0.78 | 0.539 | 0.387 | 0.674 | 0.241 | 0.671 | True | 0.418 | 9.42 | 0.657 | 0.00362 |
| accel_v2_7 | acceleration_sustained | 372 | 0.707 | 0.436 | 0.286 | 0.575 | 0.271 | 0.68 | True | 0.428 | 9.14 | 0.57 | 0.0171 |
| model_vmax | sprint_profile | 345 | 0.962 | 0.871 | 0.834 | 0.898 | 0.0906 | 0.732 | False | 0.17 | 1.57 | 0.844 | -0.0235 |
| model_tau_s | sprint_profile | 345 | 0.471 | 0.472 | 0.341 | 0.571 | -0.000874 | -0.00351 | False | 0.114 | 8.63 | 0.485 | -0.00809 |
| model_a0 | sprint_profile | 345 | 0.513 | 0.345 | 0.171 | 0.565 | 0.167 | 0.499 | True | 0.833 | 10.1 | 0.513 | 0.0339 |
| model_a_at_ref | sprint_profile | 345 | 0.623 | 0.376 | 0.198 | 0.604 | 0.247 | 0.636 | True | 0.577 | 9.69 | 0.568 | 0.0203 |

Features whose pooled ICC exceeds the within-position ICC by more than 0.1: `peak_a_tan`, `p95_a_tan`, `accel_sustained`, `accel_v2_5`, `accel_v4_7`, `accel_v2_7`, `model_a0`, `model_a_at_ref`.

**Within each position group**

| feature | scope | n_athletes | icc1 | ci_lower | ci_upper | reliability |
|---|---|---|---|---|---|---|
| peak_speed | position:DB | 93 | 0.904 | 0.859 | 0.932 | good |
| peak_speed | position:DL | 84 | 0.973 | 0.959 | 0.983 | excellent |
| peak_speed | position:OL | 84 | 0.925 | 0.887 | 0.948 | good |
| peak_speed | position:TE | 32 | 0.911 | 0.808 | 0.954 | good |
| peak_speed | position:WR | 82 | 0.891 | 0.823 | 0.932 | good |
| peak_speed_interior | position:DB | 93 | 0.912 | 0.869 | 0.939 | good |
| peak_speed_interior | position:DL | 84 | 0.974 | 0.96 | 0.984 | excellent |
| peak_speed_interior | position:OL | 84 | 0.926 | 0.892 | 0.947 | good |
| peak_speed_interior | position:TE | 32 | 0.922 | 0.826 | 0.962 | good |
| peak_speed_interior | position:WR | 82 | 0.894 | 0.829 | 0.935 | good |
| p95_speed | position:DB | 93 | 0.916 | 0.875 | 0.941 | good |
| p95_speed | position:DL | 84 | 0.976 | 0.962 | 0.985 | excellent |
| p95_speed | position:OL | 84 | 0.933 | 0.902 | 0.953 | excellent |
| p95_speed | position:TE | 32 | 0.923 | 0.835 | 0.962 | good |
| p95_speed | position:WR | 82 | 0.894 | 0.831 | 0.936 | good |
| peak_a_tan | position:DB | 93 | 0.409 | 0.164 | 0.589 | poor |
| peak_a_tan | position:DL | 84 | 0.559 | 0.399 | 0.662 | poor |
| peak_a_tan | position:OL | 84 | 0.498 | 0.34 | 0.664 | poor |
| peak_a_tan | position:TE | 32 | 0.217 | -0.105 | 0.492 | poor |
| peak_a_tan | position:WR | 82 | -0.0133 | -0.206 | 0.341 | poor |
| p95_a_tan | position:DB | 93 | 0.388 | 0.156 | 0.569 | poor |
| p95_a_tan | position:DL | 84 | 0.632 | 0.485 | 0.741 | poor |
| p95_a_tan | position:OL | 84 | 0.419 | 0.235 | 0.655 | poor |
| p95_a_tan | position:TE | 32 | 0.242 | -0.11 | 0.522 | poor |
| p95_a_tan | position:WR | 82 | -0.052 | -0.228 | 0.308 | poor |
| p95_decel | position:DB | 93 | 0.33 | 0.127 | 0.546 | poor |
| p95_decel | position:DL | 84 | 0.329 | 0.105 | 0.501 | poor |
| p95_decel | position:OL | 84 | 0.178 | 0.0271 | 0.427 | poor |
| p95_decel | position:TE | 32 | 0.388 | 0.0613 | 0.637 | poor |
| p95_decel | position:WR | 82 | 0.236 | -0.0118 | 0.465 | poor |
| path_smooth_yds | position:DB | 93 | 0.282 | 0.106 | 0.489 | poor |
| path_smooth_yds | position:DL | 84 | 0.388 | 0.203 | 0.54 | poor |
| path_smooth_yds | position:OL | 84 | 0.253 | 0.0214 | 0.443 | poor |
| path_smooth_yds | position:TE | 32 | 0.325 | 0.0081 | 0.568 | poor |
| path_smooth_yds | position:WR | 82 | 0.0337 | -0.219 | 0.268 | poor |
| accel_sustained | position:DB | 93 | 0.416 | 0.203 | 0.582 | poor |
| accel_sustained | position:DL | 84 | 0.61 | 0.461 | 0.72 | poor |
| accel_sustained | position:OL | 84 | 0.434 | 0.263 | 0.658 | poor |
| accel_sustained | position:TE | 32 | 0.24 | -0.109 | 0.541 | poor |
| accel_sustained | position:WR | 82 | -0.0485 | -0.227 | 0.335 | poor |
| accel_initial | position:DB | 93 | 0.26 | 0.0511 | 0.502 | poor |
| accel_initial | position:DL | 84 | 0.613 | 0.459 | 0.731 | poor |
| accel_initial | position:OL | 84 | 0.432 | 0.174 | 0.626 | poor |
| accel_initial | position:TE | 32 | 0.142 | -0.181 | 0.46 | poor |
| accel_initial | position:WR | 82 | 0.116 | -0.095 | 0.376 | poor |
| accel_v2_5 | position:DB | 93 | 0.219 | 0.0111 | 0.438 | poor |
| accel_v2_5 | position:DL | 84 | 0.646 | 0.51 | 0.748 | moderate |
| accel_v2_5 | position:OL | 82 | 0.586 | 0.364 | 0.74 | poor |
| accel_v2_5 | position:TE | 31 | 0.155 | -0.206 | 0.458 | poor |
| accel_v2_5 | position:WR | 82 | 0.00288 | -0.213 | 0.353 | poor |
| accel_v4_7 | position:DB | 93 | 0.552 | 0.343 | 0.687 | poor |
| accel_v4_7 | position:DL | 84 | 0.832 | 0.724 | 0.905 | moderate |
| accel_v4_7 | position:OL | 84 | 0.67 | 0.573 | 0.817 | moderate |
| accel_v4_7 | position:TE | 32 | 0.759 | 0.596 | 0.854 | moderate |
| accel_v4_7 | position:WR | 82 | 0.0894 | -0.0639 | 0.451 | poor |
| accel_v2_7 | position:DB | 93 | 0.258 | 0.0125 | 0.48 | poor |
| accel_v2_7 | position:DL | 84 | 0.8 | 0.695 | 0.876 | moderate |
| accel_v2_7 | position:OL | 82 | 0.741 | 0.596 | 0.821 | moderate |
| accel_v2_7 | position:TE | 31 | 0.502 | 0.255 | 0.69 | poor |
| accel_v2_7 | position:WR | 82 | 0.0414 | -0.132 | 0.433 | poor |
| model_vmax | position:DB | 83 | 0.772 | 0.671 | 0.836 | moderate |
| model_vmax | position:DL | 80 | 0.956 | 0.934 | 0.972 | excellent |
| model_vmax | position:OL | 79 | 0.874 | 0.807 | 0.914 | good |
| model_vmax | position:TE | 32 | 0.767 | 0.556 | 0.885 | moderate |
| model_vmax | position:WR | 71 | 0.723 | 0.546 | 0.846 | moderate |
| model_tau_s | position:DB | 83 | 0.426 | 0.213 | 0.587 | poor |
| model_tau_s | position:DL | 80 | 0.676 | 0.522 | 0.788 | moderate |
| model_tau_s | position:OL | 79 | 0.621 | 0.436 | 0.74 | poor |
| model_tau_s | position:TE | 32 | 0.242 | -0.0315 | 0.479 | poor |
| model_tau_s | position:WR | 71 | 0.136 | -0.129 | 0.49 | poor |
| model_a0 | position:DB | 83 | 0.369 | 0.204 | 0.553 | poor |
| model_a0 | position:DL | 80 | 0.705 | 0.574 | 0.8 | moderate |
| model_a0 | position:OL | 79 | 0.685 | 0.476 | 0.783 | poor |
| model_a0 | position:TE | 32 | 0.315 | 0.00285 | 0.556 | poor |
| model_a0 | position:WR | 71 | -0.00177 | -0.129 | 0.437 | poor |
| model_a_at_ref | position:DB | 83 | 0.382 | 0.216 | 0.567 | poor |
| model_a_at_ref | position:DL | 80 | 0.779 | 0.68 | 0.851 | moderate |
| model_a_at_ref | position:OL | 79 | 0.717 | 0.566 | 0.8 | moderate |
| model_a_at_ref | position:TE | 32 | 0.385 | 0.0698 | 0.61 | poor |
| model_a_at_ref | position:WR | 71 | 0.0075 | -0.11 | 0.468 | poor |

**Within each draft cohort**

| feature | scope | n_athletes | icc1 | ci_lower | ci_upper |
|---|---|---|---|---|---|
| peak_speed | cohort:2023 | 124 | 0.984 | 0.978 | 0.989 |
| peak_speed | cohort:2024 | 122 | 0.983 | 0.978 | 0.988 |
| peak_speed | cohort:2025 | 129 | 0.987 | 0.983 | 0.991 |
| peak_speed_interior | cohort:2023 | 124 | 0.986 | 0.979 | 0.99 |
| peak_speed_interior | cohort:2024 | 122 | 0.984 | 0.979 | 0.988 |
| peak_speed_interior | cohort:2025 | 129 | 0.988 | 0.984 | 0.991 |
| p95_speed | cohort:2023 | 124 | 0.986 | 0.98 | 0.99 |
| p95_speed | cohort:2024 | 122 | 0.985 | 0.98 | 0.989 |
| p95_speed | cohort:2025 | 129 | 0.988 | 0.984 | 0.992 |
| peak_a_tan | cohort:2023 | 124 | 0.665 | 0.549 | 0.757 |
| peak_a_tan | cohort:2024 | 122 | 0.329 | 0.152 | 0.515 |
| peak_a_tan | cohort:2025 | 129 | 0.618 | 0.504 | 0.71 |
| p95_a_tan | cohort:2023 | 124 | 0.726 | 0.616 | 0.808 |
| p95_a_tan | cohort:2024 | 122 | 0.314 | 0.128 | 0.512 |
| p95_a_tan | cohort:2025 | 129 | 0.622 | 0.511 | 0.707 |
| p95_decel | cohort:2023 | 124 | 0.302 | 0.132 | 0.454 |
| p95_decel | cohort:2024 | 122 | 0.178 | 0.0444 | 0.368 |
| p95_decel | cohort:2025 | 129 | 0.573 | 0.426 | 0.684 |
| path_smooth_yds | cohort:2023 | 124 | 0.17 | 0.022 | 0.311 |
| path_smooth_yds | cohort:2024 | 122 | 0.288 | 0.124 | 0.46 |
| path_smooth_yds | cohort:2025 | 129 | 0.215 | 0.0414 | 0.357 |
| accel_sustained | cohort:2023 | 124 | 0.721 | 0.608 | 0.806 |
| accel_sustained | cohort:2024 | 122 | 0.319 | 0.147 | 0.514 |
| accel_sustained | cohort:2025 | 129 | 0.62 | 0.507 | 0.712 |
| accel_initial | cohort:2023 | 124 | 0.457 | 0.25 | 0.726 |
| accel_initial | cohort:2024 | 122 | 0.252 | 0.0353 | 0.439 |
| accel_initial | cohort:2025 | 129 | 0.308 | 0.112 | 0.539 |
| accel_v2_5 | cohort:2023 | 124 | 0.772 | 0.665 | 0.847 |
| accel_v2_5 | cohort:2024 | 120 | 0.304 | 0.145 | 0.48 |
| accel_v2_5 | cohort:2025 | 128 | 0.638 | 0.528 | 0.728 |
| accel_v4_7 | cohort:2023 | 124 | 0.905 | 0.862 | 0.935 |
| accel_v4_7 | cohort:2024 | 122 | 0.648 | 0.49 | 0.814 |
| accel_v4_7 | cohort:2025 | 129 | 0.895 | 0.867 | 0.92 |
| accel_v2_7 | cohort:2023 | 124 | 0.89 | 0.831 | 0.928 |
| accel_v2_7 | cohort:2024 | 120 | 0.527 | 0.358 | 0.688 |
| accel_v2_7 | cohort:2025 | 128 | 0.839 | 0.79 | 0.878 |
| model_vmax | cohort:2023 | 118 | 0.975 | 0.963 | 0.983 |
| model_vmax | cohort:2024 | 107 | 0.938 | 0.909 | 0.958 |
| model_vmax | cohort:2025 | 120 | 0.969 | 0.958 | 0.977 |
| model_tau_s | cohort:2023 | 118 | 0.552 | 0.394 | 0.672 |
| model_tau_s | cohort:2024 | 107 | 0.361 | 0.138 | 0.565 |
| model_tau_s | cohort:2025 | 120 | 0.431 | 0.254 | 0.58 |
| model_a0 | cohort:2023 | 118 | 0.751 | 0.658 | 0.82 |
| model_a0 | cohort:2024 | 107 | 0.354 | 0.163 | 0.643 |
| model_a0 | cohort:2025 | 120 | 0.699 | 0.598 | 0.78 |
| model_a_at_ref | cohort:2023 | 118 | 0.851 | 0.799 | 0.891 |
| model_a_at_ref | cohort:2024 | 107 | 0.451 | 0.255 | 0.72 |
| model_a_at_ref | cohort:2025 | 120 | 0.816 | 0.752 | 0.865 |


Figures: `SR_06_pooled_vs_within_position_icc`, `SR_07_variance_components_and_slopes`, `SR_08_reliability_by_cohort_and_sample`.

## 5. Sprint-model identifiability

### 5.1 What the existing model does (read from `src/nflbdb/features/sprint.py`)

| Question | Answer from the source |
|---|---|
| Model | v(t) = vmax·(1 − exp(−(t − t0)/τ)), zero before t0 |
| Data entering the fit | derived speed on valid frames of the longest segment, from one frame before speed first reaches 0.5 yd/s to the peak-speed frame |
| Timestamps | real time `t_s` (seconds from the first valid frame), not frame index |
| Estimation | grid search over τ and t0; vmax in closed form for each cell; the grid minimum is returned |
| Bounds | τ 0.30–2.50 s (step 0.01); t0 −1.50 to +0.50 s around the first fitted frame (step 0.02); `fit_ok` needs ≥ 15 frames, an interior optimum, 0 < vmax ≤ 12.5 yd/s and vmax ≤ 1.10 × observed peak |
| Uncertainty | none reported |
| Censoring | not represented in the model. A late recording start is absorbed by a t0 before the first frame; the unobserved plateau is limited only by the 1.10 guard |
| vmax, τ, t0 separately estimable? | not guaranteed: without a plateau, larger vmax with larger τ fits the same rising curve; with a late start, t0 and τ trade off. Section 5.2 measures how flat the fit surface is |
| `a0` | vmax/τ, the model acceleration at zero speed at the model start. When t0 precedes the recording it is extrapolated. It is a curve parameter, not a measured acceleration |
| Fit quality | residuals of smoothed speed are autocorrelated, so a high R² is expected and does not show identifiability |
| Splits | measured from the model start; never official Combine splits |

### 5.2 Measured identifiability and window sensitivity

`tau_rel_width` = relative width of the τ range among grid cells whose SSE is within 10 % of the minimum. It is an index of flatness, not a confidence interval.

**Existing fit window (`base`), all forty attempts**

| quantity | value |
|---|---|
| n_attempts | 793 |
| share_fit_ok | 0.956 |
| share_optimum_at_grid_edge | 0.0101 |
| share_model_start_before_recording | 0.253 |
| median_gap_model_start_to_first_fitted_frame_s | 0 |
| share_weakly_identified | 0.0164 |
| median_tau_rel_width | 0.106 |
| median_a0_rel_width | 0.0812 |
| median_vmax_rel_width | 0.0254 |
| median_tau_t0_corr | -0.728 |
| median_r2 | 0.996 |
| median_rmse | 0.174 |

**Parameter movement when the fit window changes (median |relative change| against `base`)**

| variant | n_attempts | share_fit_ok | median_abs_d_vmax_rel | median_abs_d_tau_rel | median_abs_d_a0_rel | median_abs_d_a_at_ref_rel |
|---|---|---|---|---|---|---|
| common_start | 793 | 0.991 | 0.015 | 0.0976 | 0.0907 | 0.0839 |
| drop1 | 793 | 0.961 | 2.71e-10 | 0 | 2.71e-10 | 3.72e-10 |
| drop2 | 793 | 0.968 | 0.00587 | 0.0392 | 0.0356 | 0.0336 |
| drop3 | 793 | 0.98 | 0.0105 | 0.0696 | 0.0628 | 0.058 |
| to_end | 793 | 0.984 | 0.00645 | 0.0149 | 0.00866 | 0.00622 |
| trunc_end | 793 | 0.914 | 0.0066 | 0.0146 | 0.00776 | 0.0053 |

**Fit status by variant**

| variant | status | len |
|---|---|---|
| base | ok | 793 |
| common_start | ok | 793 |
| drop1 | ok | 793 |
| drop2 | ok | 793 |
| drop3 | ok | 793 |
| to_end | ok | 793 |
| trunc_end | ok | 793 |


## 6. Model repeatability

Central comparison on identical attempts. `model_a_at_ref` / `a_at_ref` is the model acceleration at 3 yd/s, inside the recorded speed range.

**Common cohort, position-adjusted; reference = `p95_a_tan` on identical attempts**

| representation | feature | variant | n_athletes | icc1 | ci_lower | ci_upper | ref_icc_same_cohort | icc_diff_vs_ref | diff_ci_lower | diff_ci_upper |
|---|---|---|---|---|---|---|---|---|---|---|
| R3 | vmax | common_start | 344 | 0.91 | 0.888 | 0.927 | 0.406 | 0.504 | 0.365 | 0.647 |
| R3 | vmax | to_end | 344 | 0.909 | 0.883 | 0.927 | 0.406 | 0.503 | 0.376 | 0.648 |
| R3 | vmax | drop3 | 344 | 0.881 | 0.847 | 0.905 | 0.406 | 0.475 | 0.343 | 0.619 |
| R3 | vmax | drop2 | 344 | 0.873 | 0.838 | 0.899 | 0.406 | 0.468 | 0.332 | 0.606 |
| R3 | vmax | persisted | 344 | 0.872 | 0.837 | 0.899 | 0.406 | 0.466 | 0.331 | 0.606 |
| R3 | vmax | base | 344 | 0.872 | 0.835 | 0.898 | 0.406 | 0.466 | 0.341 | 0.602 |
| R3 | vmax | drop1 | 344 | 0.87 | 0.835 | 0.897 | 0.406 | 0.465 | 0.335 | 0.615 |
| R3 | vmax | trunc_end | 315 | 0.85 | 0.804 | 0.88 | 0.472 | 0.378 | 0.252 | 0.527 |
| R2 | accel_v4_7 | direct | 344 | 0.552 | 0.371 | 0.718 | 0.406 | 0.146 | 0.0846 | 0.223 |
| R3 | tau_s | common_start | 344 | 0.518 | 0.414 | 0.607 | 0.406 | 0.112 | 0.0127 | 0.216 |
| R2 | accel_v2_7 | direct | 344 | 0.514 | 0.349 | 0.671 | 0.406 | 0.108 | 0.0397 | 0.171 |
| R3 | tau_s | to_end | 344 | 0.505 | 0.382 | 0.61 | 0.406 | 0.0996 | 0.00813 | 0.183 |
| R3 | tau_s | drop3 | 344 | 0.486 | 0.357 | 0.59 | 0.406 | 0.0806 | -0.012 | 0.167 |
| R3 | tau_s | drop2 | 344 | 0.481 | 0.35 | 0.581 | 0.406 | 0.0754 | -0.0125 | 0.162 |
| R3 | tau_s | trunc_end | 315 | 0.476 | 0.352 | 0.571 | 0.472 | 0.00412 | -0.0975 | 0.105 |
| R3 | tau_s | persisted | 344 | 0.472 | 0.345 | 0.583 | 0.406 | 0.0662 | -0.0269 | 0.155 |
| R3 | tau_s | base | 344 | 0.472 | 0.355 | 0.571 | 0.406 | 0.0662 | -0.0239 | 0.157 |
| R3 | tau_s | drop1 | 344 | 0.468 | 0.341 | 0.573 | 0.406 | 0.0624 | -0.0351 | 0.151 |
| R3 | a_at_ref | common_start | 344 | 0.435 | 0.275 | 0.612 | 0.406 | 0.0291 | -0.0339 | 0.121 |
| R2 | accel_v2_5 | direct | 344 | 0.423 | 0.263 | 0.555 | 0.406 | 0.0171 | -0.0676 | 0.0825 |
| R3 | a_at_ref | trunc_end | 315 | 0.409 | 0.223 | 0.63 | 0.472 | -0.0627 | -0.135 | 0.079 |
| R3 | a0 | common_start | 344 | 0.409 | 0.244 | 0.591 | 0.406 | 0.00333 | -0.0588 | 0.0896 |
| R1 | p95_a_tan | direct | 344 | 0.406 | 0.249 | 0.554 |  |  |  |  |
| R3 | a_at_ref | drop3 | 344 | 0.402 | 0.218 | 0.591 | 0.406 | -0.00339 | -0.0613 | 0.0867 |
| R2 | accel_sustained | direct | 344 | 0.396 | 0.235 | 0.543 | 0.406 | -0.00955 | -0.0245 | 0.00719 |
| R3 | a_at_ref | drop2 | 344 | 0.389 | 0.195 | 0.599 | 0.406 | -0.0169 | -0.0828 | 0.0874 |
| R3 | a_at_ref | to_end | 344 | 0.385 | 0.195 | 0.613 | 0.406 | -0.0206 | -0.101 | 0.111 |
| R3 | a0 | drop3 | 344 | 0.381 | 0.216 | 0.566 | 0.406 | -0.0247 | -0.0786 | 0.0625 |
| R3 | a0 | trunc_end | 315 | 0.376 | 0.185 | 0.584 | 0.472 | -0.0957 | -0.163 | 0.0298 |
| R3 | a_at_ref | drop1 | 344 | 0.374 | 0.194 | 0.589 | 0.406 | -0.0313 | -0.102 | 0.0834 |
| R3 | a_at_ref | base | 344 | 0.373 | 0.19 | 0.608 | 0.406 | -0.0322 | -0.101 | 0.0941 |
| R1 | peak_a_tan | direct | 344 | 0.37 | 0.198 | 0.53 | 0.406 | -0.0359 | -0.0796 | 0.00858 |
| R1 | peak_a_tan_interior | direct | 344 | 0.37 | 0.196 | 0.528 | 0.406 | -0.0359 | -0.0796 | 0.00784 |
| R3 | a0 | drop2 | 344 | 0.364 | 0.176 | 0.568 | 0.406 | -0.0413 | -0.104 | 0.0615 |
| R3 | a0 | to_end | 344 | 0.359 | 0.174 | 0.582 | 0.406 | -0.0471 | -0.128 | 0.0791 |
| R3 | a0 | drop1 | 344 | 0.346 | 0.169 | 0.56 | 0.406 | -0.0594 | -0.122 | 0.0484 |
| R3 | a0 | base | 344 | 0.344 | 0.172 | 0.564 | 0.406 | -0.0622 | -0.13 | 0.0497 |
| R3 | a0 | persisted | 344 | 0.344 | 0.167 | 0.56 | 0.406 | -0.0622 | -0.127 | 0.0545 |
| R2 | accel_initial | direct | 344 | 0.312 | 0.186 | 0.432 | 0.406 | -0.0937 | -0.222 | 0.0315 |

Rows whose paired interval for the ICC difference lies wholly above zero: `vmax` (common_start), `vmax` (to_end), `vmax` (drop3), `vmax` (drop2), `vmax` (persisted), `vmax` (base), `vmax` (drop1), `vmax` (trunc_end), `accel_v4_7` (direct), `tau_s` (common_start), `accel_v2_7` (direct), `tau_s` (to_end).

**Own largest sample (sample differs between rows — not a like-for-like comparison)**

| representation | feature | variant | n_athletes | icc1 | ci_lower | ci_upper | ref_icc_same_cohort | icc_diff_vs_ref | diff_ci_lower | diff_ci_upper |
|---|---|---|---|---|---|---|---|---|---|---|
| R3 | vmax | persisted | 345 | 0.871 | 0.835 | 0.897 | 0.409 | 0.462 | 0.332 | 0.615 |
| R3 | vmax | base | 345 | 0.871 | 0.836 | 0.897 | 0.409 | 0.462 | 0.325 | 0.608 |
| R2 | accel_v4_7 | direct | 375 | 0.539 | 0.391 | 0.682 | 0.323 | 0.216 | 0.132 | 0.31 |
| R3 | tau_s | persisted | 345 | 0.472 | 0.348 | 0.576 | 0.409 | 0.0623 | -0.0281 | 0.147 |
| R3 | tau_s | base | 345 | 0.472 | 0.348 | 0.576 | 0.409 | 0.0623 | -0.0292 | 0.144 |
| R2 | accel_v2_7 | direct | 372 | 0.436 | 0.289 | 0.581 | 0.335 | 0.101 | -0.000655 | 0.193 |
| R3 | a_at_ref | base | 345 | 0.376 | 0.189 | 0.6 | 0.409 | -0.0333 | -0.104 | 0.0955 |
| R3 | a0 | base | 345 | 0.345 | 0.173 | 0.567 | 0.409 | -0.0638 | -0.128 | 0.0546 |
| R3 | a0 | persisted | 345 | 0.345 | 0.161 | 0.561 | 0.409 | -0.0638 | -0.129 | 0.0494 |
| R1 | peak_a_tan | direct | 375 | 0.335 | 0.188 | 0.47 | 0.323 | 0.0124 | -0.0488 | 0.0799 |
| R1 | peak_a_tan_interior | direct | 375 | 0.335 | 0.191 | 0.471 | 0.323 | 0.0124 | -0.053 | 0.0834 |
| R2 | accel_v2_5 | direct | 372 | 0.333 | 0.189 | 0.461 | 0.335 | -0.0014 | -0.0993 | 0.073 |
| R2 | accel_sustained | direct | 375 | 0.331 | 0.186 | 0.46 | 0.323 | 0.00799 | -0.0159 | 0.0317 |
| R1 | p95_a_tan | direct | 375 | 0.323 | 0.18 | 0.462 |  |  |  |  |
| R2 | accel_initial | direct | 375 | 0.298 | 0.172 | 0.419 | 0.323 | -0.0245 | -0.161 | 0.111 |


Figure: `SR_09_model_repeatability_and_sensitivity`.

## 7. Stopwatch redundancy

**Position-adjusted association with each official time (athlete means)**

| feature | official | n_athletes | n_missing_official | spearman | spearman_lower | spearman_upper | pearson |
|---|---|---|---|---|---|---|---|
| peak_speed | cmb_forty | 418 | 0 | -0.883 | -0.902 | -0.86 | -0.921 |
| peak_speed | cmb_ten_yd_split | 418 | 0 | -0.549 | -0.613 | -0.477 | -0.693 |
| peak_speed_interior | cmb_forty | 418 | 0 | -0.89 | -0.908 | -0.868 | -0.928 |
| peak_speed_interior | cmb_ten_yd_split | 418 | 0 | -0.555 | -0.618 | -0.485 | -0.698 |
| p95_speed | cmb_forty | 418 | 0 | -0.887 | -0.906 | -0.865 | -0.926 |
| p95_speed | cmb_ten_yd_split | 418 | 0 | -0.554 | -0.617 | -0.483 | -0.697 |
| peak_a_tan | cmb_forty | 418 | 0 | -0.297 | -0.382 | -0.206 | -0.343 |
| peak_a_tan | cmb_ten_yd_split | 418 | 0 | -0.27 | -0.357 | -0.178 | -0.324 |
| p95_a_tan | cmb_forty | 418 | 0 | -0.328 | -0.411 | -0.239 | -0.379 |
| p95_a_tan | cmb_ten_yd_split | 418 | 0 | -0.281 | -0.368 | -0.19 | -0.343 |
| p95_decel | cmb_forty | 418 | 0 | 0.15 | 0.0541 | 0.243 | 0.151 |
| p95_decel | cmb_ten_yd_split | 418 | 0 | 0.0861 | -0.0104 | 0.181 | 0.0779 |
| path_smooth_yds | cmb_forty | 418 | 0 | -0.0667 | -0.162 | 0.0298 | -0.0241 |
| path_smooth_yds | cmb_ten_yd_split | 418 | 0 | -0.024 | -0.12 | 0.0725 | 0.00654 |
| accel_sustained | cmb_forty | 418 | 0 | -0.311 | -0.395 | -0.221 | -0.37 |
| accel_sustained | cmb_ten_yd_split | 418 | 0 | -0.277 | -0.364 | -0.186 | -0.341 |
| accel_initial | cmb_forty | 418 | 0 | -0.173 | -0.265 | -0.0775 | -0.205 |
| accel_initial | cmb_ten_yd_split | 418 | 0 | -0.174 | -0.266 | -0.0792 | -0.182 |
| accel_v2_5 | cmb_forty | 418 | 0 | -0.28 | -0.367 | -0.189 | -0.338 |
| accel_v2_5 | cmb_ten_yd_split | 418 | 0 | -0.253 | -0.341 | -0.161 | -0.331 |
| accel_v4_7 | cmb_forty | 418 | 0 | -0.523 | -0.589 | -0.449 | -0.589 |
| accel_v4_7 | cmb_ten_yd_split | 418 | 0 | -0.4 | -0.478 | -0.315 | -0.52 |
| accel_v2_7 | cmb_forty | 418 | 0 | -0.437 | -0.512 | -0.356 | -0.491 |
| accel_v2_7 | cmb_ten_yd_split | 418 | 0 | -0.352 | -0.434 | -0.265 | -0.45 |
| model_vmax | cmb_forty | 413 | 0 | -0.83 | -0.858 | -0.797 | -0.868 |
| model_vmax | cmb_ten_yd_split | 413 | 0 | -0.522 | -0.589 | -0.447 | -0.638 |
| model_tau_s | cmb_forty | 413 | 0 | -0.082 | -0.178 | 0.015 | -0.0507 |
| model_tau_s | cmb_ten_yd_split | 413 | 0 | 0.018 | -0.0791 | 0.115 | 0.0521 |
| model_a0 | cmb_forty | 413 | 0 | -0.307 | -0.392 | -0.217 | -0.308 |
| model_a0 | cmb_ten_yd_split | 413 | 0 | -0.29 | -0.376 | -0.199 | -0.303 |
| model_a_at_ref | cmb_forty | 413 | 0 | -0.447 | -0.521 | -0.366 | -0.442 |
| model_a_at_ref | cmb_ten_yd_split | 413 | 0 | -0.379 | -0.459 | -0.293 | -0.399 |

**After also conditioning on the other official time**

| feature | official | n_athletes | spearman | spearman_lower | spearman_upper |
|---|---|---|---|---|---|
| peak_speed | cmb_ten_yd_split\|cmb_forty | 418 | 0.104 | 0.00778 | 0.199 |
| peak_speed | cmb_forty\|cmb_ten_yd_split | 418 | -0.829 | -0.857 | -0.797 |
| peak_speed_interior | cmb_ten_yd_split\|cmb_forty | 418 | 0.101 | 0.00464 | 0.196 |
| peak_speed_interior | cmb_forty\|cmb_ten_yd_split | 418 | -0.838 | -0.864 | -0.806 |
| p95_speed | cmb_ten_yd_split\|cmb_forty | 418 | 0.1 | 0.00358 | 0.195 |
| p95_speed | cmb_forty\|cmb_ten_yd_split | 418 | -0.834 | -0.862 | -0.803 |
| peak_a_tan | cmb_ten_yd_split\|cmb_forty | 418 | -0.103 | -0.197 | -0.00608 |
| peak_a_tan | cmb_forty\|cmb_ten_yd_split | 418 | -0.163 | -0.256 | -0.0678 |
| p95_a_tan | cmb_ten_yd_split\|cmb_forty | 418 | -0.0905 | -0.185 | 0.00606 |
| p95_a_tan | cmb_forty\|cmb_ten_yd_split | 418 | -0.197 | -0.288 | -0.102 |
| p95_decel | cmb_ten_yd_split\|cmb_forty | 418 | -0.0178 | -0.114 | 0.0788 |
| p95_decel | cmb_forty\|cmb_ten_yd_split | 418 | 0.124 | 0.0281 | 0.218 |
| path_smooth_yds | cmb_ten_yd_split\|cmb_forty | 418 | 0.0271 | -0.0696 | 0.123 |
| path_smooth_yds | cmb_forty\|cmb_ten_yd_split | 418 | -0.0679 | -0.163 | 0.0288 |
| accel_sustained | cmb_ten_yd_split\|cmb_forty | 418 | -0.1 | -0.195 | -0.0039 |
| accel_sustained | cmb_forty\|cmb_ten_yd_split | 418 | -0.176 | -0.268 | -0.0813 |
| accel_initial | cmb_ten_yd_split\|cmb_forty | 418 | -0.0812 | -0.176 | 0.0155 |
| accel_initial | cmb_forty\|cmb_ten_yd_split | 418 | -0.0774 | -0.173 | 0.0193 |
| accel_v2_5 | cmb_ten_yd_split\|cmb_forty | 418 | -0.0937 | -0.188 | 0.00281 |
| accel_v2_5 | cmb_forty\|cmb_ten_yd_split | 418 | -0.155 | -0.248 | -0.0595 |
| accel_v4_7 | cmb_ten_yd_split\|cmb_forty | 418 | -0.0834 | -0.178 | 0.0132 |
| accel_v4_7 | cmb_forty\|cmb_ten_yd_split | 418 | -0.375 | -0.455 | -0.289 |
| accel_v2_7 | cmb_ten_yd_split\|cmb_forty | 418 | -0.093 | -0.188 | 0.00352 |
| accel_v2_7 | cmb_forty\|cmb_ten_yd_split | 418 | -0.29 | -0.376 | -0.2 |
| model_vmax | cmb_ten_yd_split\|cmb_forty | 413 | 0.0675 | -0.0298 | 0.164 |
| model_vmax | cmb_forty\|cmb_ten_yd_split | 413 | -0.757 | -0.796 | -0.713 |
| model_tau_s | cmb_ten_yd_split\|cmb_forty | 413 | 0.097 | -9.7e-05 | 0.192 |
| model_tau_s | cmb_forty\|cmb_ten_yd_split | 413 | -0.126 | -0.22 | -0.0288 |
| model_a0 | cmb_ten_yd_split\|cmb_forty | 413 | -0.121 | -0.216 | -0.0245 |
| model_a0 | cmb_forty\|cmb_ten_yd_split | 413 | -0.16 | -0.253 | -0.0642 |
| model_a_at_ref | cmb_ten_yd_split\|cmb_forty | 413 | -0.123 | -0.218 | -0.0266 |
| model_a_at_ref | cmb_forty\|cmb_ten_yd_split | 413 | -0.283 | -0.369 | -0.191 |

**Share of feature variance explained by both official times, within position (partial R²)**

| feature | n_athletes | r2 |
|---|---|---|
| peak_speed | 418 | 0.859 |
| peak_speed_interior | 418 | 0.872 |
| p95_speed | 418 | 0.869 |
| peak_a_tan | 418 | 0.123 |
| p95_a_tan | 418 | 0.147 |
| p95_decel | 418 | 0.029 |
| path_smooth_yds | 418 | 0.00267 |
| accel_sustained | 418 | 0.141 |
| accel_initial | 418 | 0.0428 |
| accel_v2_5 | 418 | 0.123 |
| accel_v4_7 | 418 | 0.351 |
| accel_v2_7 | 418 | 0.248 |
| model_vmax | 413 | 0.77 |
| model_tau_s | 413 | 0.0292 |
| model_a0 | 413 | 0.103 |
| model_a_at_ref | 413 | 0.199 |

**Pooled over positions, official forty (for comparison only)**

| feature | n_athletes | spearman | spearman_lower | spearman_upper |
|---|---|---|---|---|
| peak_speed | 418 | -0.969 | -0.974 | -0.962 |
| peak_speed_interior | 418 | -0.97 | -0.976 | -0.964 |
| p95_speed | 418 | -0.97 | -0.975 | -0.964 |
| peak_a_tan | 418 | -0.643 | -0.696 | -0.583 |
| p95_a_tan | 418 | -0.654 | -0.706 | -0.596 |
| p95_decel | 418 | 0.344 | 0.257 | 0.426 |
| path_smooth_yds | 418 | -0.0968 | -0.191 | -0.000877 |
| accel_sustained | 418 | -0.645 | -0.698 | -0.585 |
| accel_initial | 418 | -0.421 | -0.497 | -0.339 |
| accel_v2_5 | 418 | -0.616 | -0.672 | -0.552 |
| accel_v4_7 | 418 | -0.836 | -0.862 | -0.804 |
| accel_v2_7 | 418 | -0.782 | -0.816 | -0.741 |
| model_vmax | 413 | -0.948 | -0.957 | -0.938 |
| model_tau_s | 413 | -0.013 | -0.109 | 0.0836 |
| model_a0 | 413 | -0.659 | -0.71 | -0.6 |
| model_a_at_ref | 413 | -0.78 | -0.815 | -0.739 |

**Sensitivity: fastest attempt instead of the mean, official forty**

| feature | n_athletes | spearman | spearman_lower | spearman_upper |
|---|---|---|---|---|
| peak_speed | 418 | -0.875 | -0.895 | -0.85 |
| peak_speed_interior | 418 | -0.882 | -0.902 | -0.858 |
| p95_speed | 418 | -0.881 | -0.901 | -0.857 |
| peak_a_tan | 418 | -0.274 | -0.361 | -0.182 |
| p95_a_tan | 418 | -0.286 | -0.372 | -0.195 |
| p95_decel | 418 | 0.131 | 0.0354 | 0.225 |
| path_smooth_yds | 418 | -0.049 | -0.145 | 0.0476 |
| accel_sustained | 418 | -0.276 | -0.363 | -0.184 |
| accel_initial | 418 | -0.179 | -0.271 | -0.0845 |
| accel_v2_5 | 417 | -0.248 | -0.337 | -0.155 |
| accel_v4_7 | 418 | -0.499 | -0.568 | -0.423 |
| accel_v2_7 | 417 | -0.404 | -0.482 | -0.32 |
| model_vmax | 399 | -0.81 | -0.841 | -0.773 |
| model_tau_s | 399 | -0.0286 | -0.127 | 0.0702 |
| model_a0 | 399 | -0.323 | -0.408 | -0.231 |
| model_a_at_ref | 399 | -0.451 | -0.527 | -0.369 |


### Feature screening

| feature | family | phase04_status | n_athletes_repeat | icc_adj | icc_adj_lower | icc_adj_upper | icc_cohort_min | icc_adj_s3 | signal_to_noise | redundancy_abs | redundancy_lower | redundancy_upper | r2_stopwatch_adj | reliable_unique_share | quadrant | screening_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_tau_s | sprint_profile | not_in_registry | 345 | 0.472 | 0.341 | 0.571 | 0.361 | 0.496 | 0.945 | 0.082 | 0 | 0.178 | 0.0292 | 0.612 | unreliable_nonredundant | inconclusive |
| path_smooth_yds | recording_window | EXPERIMENTAL | 375 | 0.267 | 0.161 | 0.359 | 0.17 | 0.292 | 0.604 | 0.0667 | 0 | 0.162 | 0.00267 | 0.419 | unreliable_nonredundant | not_repeatable |
| accel_initial | acceleration_sustained | not_in_registry | 375 | 0.298 | 0.171 | 0.421 | 0.252 | 0.428 | 0.652 | 0.174 | 0.0792 | 0.266 | 0.0428 | 0.417 | unreliable_nonredundant | not_repeatable |
| model_a0 | sprint_profile | not_in_registry | 345 | 0.345 | 0.171 | 0.565 | 0.354 | 0.546 | 0.726 | 0.307 | 0.217 | 0.392 | 0.103 | 0.411 | unreliable_nonredundant | inconclusive |
| p95_decel | deceleration | REJECTED | 375 | 0.266 | 0.166 | 0.373 | 0.178 | 0.345 | 0.602 | 0.15 | 0.0541 | 0.243 | 0.029 | 0.391 | unreliable_nonredundant | not_repeatable |
| peak_a_tan | acceleration_direct | EXPERIMENTAL | 375 | 0.335 | 0.194 | 0.473 | 0.329 | 0.47 | 0.71 | 0.297 | 0.206 | 0.382 | 0.123 | 0.379 | unreliable_nonredundant | not_repeatable |
| accel_v2_5 | acceleration_sustained | not_in_registry | 372 | 0.333 | 0.201 | 0.456 | 0.304 | 0.509 | 0.707 | 0.28 | 0.189 | 0.367 | 0.123 | 0.377 | unreliable_nonredundant | not_repeatable |
| accel_v2_7 | acceleration_sustained | not_in_registry | 372 | 0.436 | 0.286 | 0.575 | 0.527 | 0.64 | 0.879 | 0.437 | 0.356 | 0.512 | 0.248 | 0.359 | unreliable_nonredundant | inconclusive |
| accel_sustained | acceleration_sustained | not_in_registry | 375 | 0.331 | 0.192 | 0.471 | 0.319 | 0.478 | 0.703 | 0.311 | 0.221 | 0.395 | 0.141 | 0.356 | unreliable_nonredundant | not_repeatable |
| accel_v4_7 | acceleration_sustained | not_in_registry | 375 | 0.539 | 0.387 | 0.674 | 0.648 | 0.686 | 1.08 | 0.523 | 0.449 | 0.589 | 0.351 | 0.349 | unreliable_nonredundant | inconclusive |
| model_a_at_ref | sprint_profile | not_in_registry | 345 | 0.376 | 0.198 | 0.604 | 0.451 | 0.618 | 0.776 | 0.447 | 0.366 | 0.521 | 0.199 | 0.347 | unreliable_nonredundant | inconclusive |
| p95_a_tan | acceleration_direct | EXPERIMENTAL | 375 | 0.323 | 0.185 | 0.461 | 0.314 | 0.484 | 0.69 | 0.328 | 0.239 | 0.411 | 0.147 | 0.341 | unreliable_nonredundant | not_repeatable |
| model_vmax | sprint_profile | not_in_registry | 345 | 0.871 | 0.834 | 0.898 | 0.938 | 0.899 | 2.6 | 0.83 | 0.797 | 0.858 | 0.77 | 0.162 | reliable_redundant | redundant_with_stopwatch |
| peak_speed | speed | PROVISIONAL | 375 | 0.942 | 0.928 | 0.953 | 0.983 | 0.947 | 4.02 | 0.883 | 0.86 | 0.902 | 0.859 | 0.111 | reliable_redundant | redundant_with_stopwatch |
| p95_speed | speed | VALIDATED | 375 | 0.948 | 0.935 | 0.958 | 0.985 | 0.953 | 4.27 | 0.887 | 0.865 | 0.906 | 0.869 | 0.105 | reliable_redundant | redundant_with_stopwatch |
| peak_speed_interior | speed | VALIDATED | 375 | 0.945 | 0.932 | 0.956 | 0.984 | 0.951 | 4.16 | 0.89 | 0.868 | 0.908 | 0.872 | 0.1 | reliable_redundant | redundant_with_stopwatch |


Figure: `SR_10_feature_screening` (PNG, SVG, PDF). The reference lines at ICC 0.75 and |ρ| 0.70 are heuristic screening references, not acceptance thresholds. The upper-left region (repeatable, not explained by the stopwatch) is a reason to test a feature against NFL play; it is not evidence that the feature matters there.

`reliable_unique_share` is a lower bound of the reproducible athlete variance not explained by the two official times. It assumes the stopwatch explains reproducible variance only, and it treats official times as error-free.

## 8. Data eligibility

| stage | definition | n_athletes | expected | matches_expected |
|---|---|---|---|---|
| 01_combine_results | athletes with a Combine results row | 510 | 510 | True |
| 02_nfl_tracking_any | … with ≥ 1 tracked NFL play in any season type | 497 | 497 | True |
| 03_nfl_tracking_reg | … with ≥ 1 tracked regular-season play | 401 | 401 | True |
| 04_nfl_no_reg:POST+PRE | NFL tracking but no regular-season play; season types present: POST+PRE | 1 |  |  |
| 05_nfl_no_reg:PRE | NFL tracking but no regular-season play; season types present: PRE | 95 |  |  |
| 06_nfl_reg_ge_50 | ≥ 50 tracked regular-season plays | 337 |  |  |
| 07_nfl_reg_ge_100 | ≥ 100 tracked regular-season plays | 303 |  |  |
| 08_nfl_reg_ge_200 | ≥ 200 tracked regular-season plays | 265 |  |  |
| 09_forty_tracked | ≥ 1 tracked forty attempt (usable tier) | 418 |  |  |
| 10_forty_repeat | ≥ 2 tracked forty attempts | 375 |  |  |
| 11_official_forty | official forty time present | 424 |  |  |
| 12_official_ten_split | official 10-yard split present | 425 |  |  |
| 13_official_both | official forty AND 10-yard split present | 424 |  |  |
| 14_forty_tracked_and_official_both | tracked forty AND both official times | 418 |  |  |
| 15_scenario_S0_all_any | ≥ 1 forty attempt retained by S0_all | 418 |  |  |
| 16_scenario_S0_all_repeat | ≥ 2 forty attempts retained by S0_all | 375 |  |  |
| 17_scenario_S1_structural_any | ≥ 1 forty attempt retained by S1_structural | 416 |  |  |
| 18_scenario_S1_structural_repeat | ≥ 2 forty attempts retained by S1_structural | 369 |  |  |
| 19_scenario_S2_signal_any | ≥ 1 forty attempt retained by S2_signal | 401 |  |  |
| 20_scenario_S2_signal_repeat | ≥ 2 forty attempts retained by S2_signal | 333 |  |  |
| 21_scenario_S3_no_boundary_any | ≥ 1 forty attempt retained by S3_no_boundary | 378 |  |  |
| 22_scenario_S3_no_boundary_repeat | ≥ 2 forty attempts retained by S3_no_boundary | 254 |  |  |
| 23_model_fit_ok_any | ≥ 1 forty attempt with a usable sprint-model fit | 413 |  |  |
| 24_model_fit_ok_repeat | ≥ 2 forty attempts with a usable fit | 345 |  |  |
| 25_forty_repeat_and_reg_ge_50 | repeated forty AND ≥ 50 regular-season plays | 252 |  |  |
| 26_forty_official_both_and_reg_ge_50 | tracked forty, both official times AND ≥ 50 regular-season plays | 285 |  |  |
| 27_forty_repeat_and_reg_ge_100 | repeated forty AND ≥ 100 regular-season plays | 222 |  |  |
| 28_forty_official_both_and_reg_ge_100 | tracked forty, both official times AND ≥ 100 regular-season plays | 253 |  |  |
| 29_forty_repeat_and_reg_ge_200 | repeated forty AND ≥ 200 regular-season plays | 189 |  |  |
| 30_forty_official_both_and_reg_ge_200 | tracked forty, both official times AND ≥ 200 regular-season plays | 219 |  |  |

All counts quoted in earlier reports are reproduced.


**Data-quality status (from `reports/tables/phase_03/descriptive/A0_dq_failures.csv`, executed).** All seven non-PASS Phase 01 checks are classified and none is blocking: the CRITICAL and MEDIUM Combine key failures and both range warnings are check-definition errors caused by BALL rows (PLAYER rows are unaffected); the two `player_play` orphan checks are coverage gaps (202 plays, 2,497 player-plays) that are excluded from context-aware features; the jump warning is handled by segmentation. The orphan gap affects Study F only: tracked plays without a `player_play` row have no endpoint value and count as missing there.

## 9. NFL endpoint feasibility

No Combine value is related to any NFL value in this section, and no model is fitted.

### 9.1 Primary: defensive line `player_get_off`

**Definition status:** **NOT established**. schema_catalog: meaning = 'Timing metric: player get off', unit = 'seconds', evidence = 'inferred'. No vendor data dictionary in the repository: start / end events and sign convention are unknown.

**Missingness (player-plays)**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | n_non_null | 3.34e+04 | 52,591 | player-plays |
| all | share_non_null | 0.635 | 52,591 | player-plays |
| season:2023 | n_non_null | 4.86e+03 | 7,413 | player-plays |
| season:2023 | share_non_null | 0.656 | 7,413 | player-plays |
| season:2024 | n_non_null | 1.12e+04 | 17,402 | player-plays |
| season:2024 | share_non_null | 0.642 | 17,402 | player-plays |
| season:2025 | n_non_null | 1.74e+04 | 27,776 | player-plays |
| season:2025 | share_non_null | 0.626 | 27,776 | player-plays |
| pass_result:labelled | n_non_null | 3.34e+04 | 52,591 | player-plays |
| pass_result:labelled | share_non_null | 0.635 | 52,591 | player-plays |
| pass_result:null | n_non_null | 0 | 0 | player-plays |
| pass_result:null | share_non_null |  | 0 | player-plays |
| lined_up_position:EDGE | n_non_null | 2.02e+04 | 28,882 | player-plays |
| lined_up_position:EDGE | share_non_null | 0.7 | 28,882 | player-plays |
| lined_up_position:INTERIOR_LINE | n_non_null | 1.31e+04 | 23,028 | player-plays |
| lined_up_position:INTERIOR_LINE | share_non_null | 0.568 | 23,028 | player-plays |
| lined_up_position:SLOT_CB | n_non_null | 114 | 503 | player-plays |
| lined_up_position:SLOT_CB | share_non_null | 0.227 | 503 | player-plays |
| lined_up_position:MLB | n_non_null | 0 | 85 | player-plays |
| lined_up_position:MLB | share_non_null | 0 | 85 | player-plays |
| lined_up_position:OLB | n_non_null | 0 | 51 | player-plays |
| lined_up_position:OLB | share_non_null | 0 | 51 | player-plays |
| lined_up_position:CB | n_non_null | 11 | 33 | player-plays |
| lined_up_position:CB | share_non_null | 0.333 | 33 | player-plays |
| blitzing:labelled | n_non_null | 2.53e+04 | 29,000 | player-plays |
| blitzing:labelled | share_non_null | 0.873 | 29,000 | player-plays |
| blitzing:null | n_non_null | 8.1e+03 | 23,591 | player-plays |
| blitzing:null | share_non_null | 0.344 | 23,591 | player-plays |

**Distribution of non-null values**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | mean | 1.01 | 33,417 | non-null player-plays |
| all | sd | 0.293 | 33,417 | non-null player-plays |
| all | min | 0.004 | 33,417 | non-null player-plays |
| all | p01 | 0.506 | 33,417 | non-null player-plays |
| all | p05 | 0.628 | 33,417 | non-null player-plays |
| all | p25 | 0.805 | 33,417 | non-null player-plays |
| all | p50 | 0.96 | 33,417 | non-null player-plays |
| all | p75 | 1.16 | 33,417 | non-null player-plays |
| all | p95 | 1.61 | 33,417 | non-null player-plays |
| all | p99 | 1.88 | 33,417 | non-null player-plays |
| all | max | 2 | 33,417 | non-null player-plays |
| all | share_le_zero | 0 | 33,417 | non-null player-plays |

**Athletes and observations per athlete**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | n_athletes_position | 98 | 98 | athletes of the position group with tracked plays |
| all | n_athletes_any_obs | 98 | 98 | athletes |
| all | obs_per_athlete_p10 | 24.4 | 98 | athletes |
| all | obs_per_athlete_p25 | 74.5 | 98 | athletes |
| all | obs_per_athlete_p50 | 195 | 98 | athletes |
| all | obs_per_athlete_p75 | 460 | 98 | athletes |
| all | obs_per_athlete_p90 | 937 | 98 | athletes |
| min_obs=20 | n_athletes | 93 | 98 | athletes with at least this many observations |
| min_obs=50 | n_athletes | 82 | 98 | athletes with at least this many observations |
| min_obs=100 | n_athletes | 69 | 98 | athletes with at least this many observations |
| min_obs=200 | n_athletes | 49 | 98 | athletes with at least this many observations |

**Overlap with Combine measurements (athletes)**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| min_obs=20 | has_cmb_forty | 75 | 93 | athletes also meeting the Combine flag |
| min_obs=20 | has_cmb_ten_yd_split | 75 | 93 | athletes also meeting the Combine flag |
| min_obs=20 | has_cmb_vertical | 82 | 93 | athletes also meeting the Combine flag |
| min_obs=20 | has_cmb_broad_jump | 80 | 93 | athletes also meeting the Combine flag |
| min_obs=20 | has_cmb_combine_weight | 93 | 93 | athletes also meeting the Combine flag |
| min_obs=20 | forty_tracked | 74 | 93 | athletes also meeting the Combine flag |
| min_obs=20 | forty_repeat | 66 | 93 | athletes also meeting the Combine flag |
| min_obs=20 | model_fit_ok_repeat | 63 | 93 | athletes also meeting the Combine flag |
| min_obs=20 | forty_tracked_and_official_both | 74 | 93 | athletes |
| min_obs=50 | has_cmb_forty | 66 | 82 | athletes also meeting the Combine flag |
| min_obs=50 | has_cmb_ten_yd_split | 66 | 82 | athletes also meeting the Combine flag |
| min_obs=50 | has_cmb_vertical | 72 | 82 | athletes also meeting the Combine flag |
| min_obs=50 | has_cmb_broad_jump | 70 | 82 | athletes also meeting the Combine flag |
| min_obs=50 | has_cmb_combine_weight | 82 | 82 | athletes also meeting the Combine flag |
| min_obs=50 | forty_tracked | 65 | 82 | athletes also meeting the Combine flag |
| min_obs=50 | forty_repeat | 57 | 82 | athletes also meeting the Combine flag |
| min_obs=50 | model_fit_ok_repeat | 54 | 82 | athletes also meeting the Combine flag |
| min_obs=50 | forty_tracked_and_official_both | 65 | 82 | athletes |
| min_obs=100 | has_cmb_forty | 54 | 69 | athletes also meeting the Combine flag |
| min_obs=100 | has_cmb_ten_yd_split | 54 | 69 | athletes also meeting the Combine flag |
| min_obs=100 | has_cmb_vertical | 63 | 69 | athletes also meeting the Combine flag |
| min_obs=100 | has_cmb_broad_jump | 61 | 69 | athletes also meeting the Combine flag |
| min_obs=100 | has_cmb_combine_weight | 69 | 69 | athletes also meeting the Combine flag |
| min_obs=100 | forty_tracked | 53 | 69 | athletes also meeting the Combine flag |
| min_obs=100 | forty_repeat | 45 | 69 | athletes also meeting the Combine flag |
| min_obs=100 | model_fit_ok_repeat | 43 | 69 | athletes also meeting the Combine flag |
| min_obs=100 | forty_tracked_and_official_both | 53 | 69 | athletes |
| min_obs=200 | has_cmb_forty | 37 | 49 | athletes also meeting the Combine flag |
| min_obs=200 | has_cmb_ten_yd_split | 37 | 49 | athletes also meeting the Combine flag |
| min_obs=200 | has_cmb_vertical | 44 | 49 | athletes also meeting the Combine flag |
| min_obs=200 | has_cmb_broad_jump | 43 | 49 | athletes also meeting the Combine flag |
| min_obs=200 | has_cmb_combine_weight | 49 | 49 | athletes also meeting the Combine flag |
| min_obs=200 | forty_tracked | 37 | 49 | athletes also meeting the Combine flag |
| min_obs=200 | forty_repeat | 30 | 49 | athletes also meeting the Combine flag |
| min_obs=200 | model_fit_ok_repeat | 28 | 49 | athletes also meeting the Combine flag |
| min_obs=200 | forty_tracked_and_official_both | 37 | 49 | athletes |

**Play context among endpoint plays**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | share_non_null:lined_up_position | 1 | 33,417 | non-null endpoint player-plays |
| all | share_non_null:blitzing | 0.757 | 33,417 | non-null endpoint player-plays |
| all | share_non_null:down | 1 | 33,417 | non-null endpoint player-plays |
| all | share_non_null:yards_to_go | 1 | 33,417 | non-null endpoint player-plays |
| all | share_non_null:offense_formation | 0.998 | 33,417 | non-null endpoint player-plays |
| all | share_non_null:pass_result | 1 | 33,417 | non-null endpoint player-plays |
| all | share_non_null:quarter | 1 | 33,417 | non-null endpoint player-plays |
| all | share_non_null:team_coverage_man_zone | 0.997 | 33,417 | non-null endpoint player-plays |
| lined_up_position:EDGE | share_of_endpoint_plays | 0.605 | 33,417 | player-plays |
| lined_up_position:INTERIOR_LINE | share_of_endpoint_plays | 0.391 | 33,417 | player-plays |
| lined_up_position:SLOT_CB | share_of_endpoint_plays | 0.00341 | 33,417 | player-plays |
| lined_up_position:CB | share_of_endpoint_plays | 0.000329 | 33,417 | player-plays |

**Repeatability**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| play | icc1 | 0.0967 | 93 | single plays, athletes with ≥ 20 observations; athlete bootstrap |
| play | icc1_ci_lower | 0.0673 | 93 | single plays, athletes with ≥ 20 observations; athlete bootstrap |
| play | icc1_ci_upper | 0.128 | 93 | single plays, athletes with ≥ 20 observations; athlete bootstrap |
| play | projected_reliability_n=20 | 0.682 | 93 | Spearman–Brown from the single-play ICC |
| play | projected_reliability_n=50 | 0.843 | 93 | Spearman–Brown from the single-play ICC |
| play | projected_reliability_n=100 | 0.915 | 93 | Spearman–Brown from the single-play ICC |
| play | projected_reliability_n=200 | 0.955 | 93 | Spearman–Brown from the single-play ICC |
| play | plays_needed_for_0.7 | 21.8 | 93 | Spearman–Brown inverse |
| play | plays_needed_for_0.8 | 37.3 | 93 | Spearman–Brown inverse |
| play_context_centred | icc1 | 0.0509 | 93 | alignment × season means removed before the ICC |
| play_context_centred | icc1_ci_lower | 0.0328 | 93 |  |
| play_context_centred | icc1_ci_upper | 0.0717 | 93 |  |
| game | icc1 | 0.41 | 89 | athlete-game means, games with ≥ 5 observations, athletes with ≥ 2 such games |
| game | icc1_ci_lower | 0.325 | 89 | athlete-game means, games with ≥ 5 observations, athletes with ≥ 2 such games |
| game | icc1_ci_upper | 0.488 | 89 | athlete-game means, games with ≥ 5 observations, athletes with ≥ 2 such games |
| game | games_per_athlete_median | 16 | 89 | athletes |
| split_half | spearman_odd_even | 0.903 | 93 | odd vs even chronological observations |
| split_half | spearman_brown_full_length | 0.949 | 93 | reliability of the athlete mean over all own observations |

**First-N-play eligibility**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| n=20 | n_athletes_ge_n | 93 | 98 | athletes with ≥ n observations overall |
| n=20 | n_athletes_ge_n_rookie_season | 81 | 98 | observations in the season equal to the draft cohort year |
| n=20 | median_games_to_reach_n | 3 | 93 | athletes reaching n observations |
| n=20 | spearman_first_n_vs_next_n | 0.602 | 85 | mean of the first n observations vs mean of the next n (athletes with ≥ 2n) |
| n=50 | n_athletes_ge_n | 82 | 98 | athletes with ≥ n observations overall |
| n=50 | n_athletes_ge_n_rookie_season | 70 | 98 | observations in the season equal to the draft cohort year |
| n=50 | median_games_to_reach_n | 5 | 82 | athletes reaching n observations |
| n=50 | spearman_first_n_vs_next_n | 0.788 | 69 | mean of the first n observations vs mean of the next n (athletes with ≥ 2n) |
| n=100 | n_athletes_ge_n | 69 | 98 | athletes with ≥ n observations overall |
| n=100 | n_athletes_ge_n_rookie_season | 51 | 98 | observations in the season equal to the draft cohort year |
| n=100 | median_games_to_reach_n | 7 | 69 | athletes reaching n observations |
| n=100 | spearman_first_n_vs_next_n | 0.739 | 49 | mean of the first n observations vs mean of the next n (athletes with ≥ 2n) |


### 9.2 Secondary: wide receiver `separation_at_pass_forward`

**Definition status:** **NOT established**. schema_catalog: meaning = 'Distance metric: separation at pass forward', unit = 'yards', evidence = 'inferred'. No vendor data dictionary in the repository: start / end events and sign convention are unknown.

**Missingness (player-plays)**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | n_non_null | 3.1e+04 | 36,944 | player-plays |
| all | share_non_null | 0.838 | 36,944 | player-plays |
| season:2023 | n_non_null | 6.6e+03 | 7,828 | player-plays |
| season:2023 | share_non_null | 0.843 | 7,828 | player-plays |
| season:2024 | n_non_null | 1.09e+04 | 13,075 | player-plays |
| season:2024 | share_non_null | 0.832 | 13,075 | player-plays |
| season:2025 | n_non_null | 1.35e+04 | 16,041 | player-plays |
| season:2025 | share_non_null | 0.84 | 16,041 | player-plays |
| pass_result:labelled | n_non_null | 3.1e+04 | 36,789 | player-plays |
| pass_result:labelled | share_non_null | 0.841 | 36,789 | player-plays |
| pass_result:null | n_non_null | 0 | 155 | player-plays |
| pass_result:null | share_non_null | 0 | 155 | player-plays |
| lined_up_position:WR | n_non_null | 1.89e+04 | 21,584 | player-plays |
| lined_up_position:WR | share_non_null | 0.876 | 21,584 | player-plays |
| lined_up_position:SLOT_WR | n_non_null | 1.08e+04 | 12,413 | player-plays |
| lined_up_position:SLOT_WR | share_non_null | 0.873 | 12,413 | player-plays |
| lined_up_position:TE | n_non_null | 1e+03 | 1,127 | player-plays |
| lined_up_position:TE | share_non_null | 0.89 | 1,127 | player-plays |
| lined_up_position:RB | n_non_null | 163 | 179 | player-plays |
| lined_up_position:RB | share_non_null | 0.911 | 179 | player-plays |
| lined_up_position:FB | n_non_null | 37 | 40 | player-plays |
| lined_up_position:FB | share_non_null | 0.925 | 40 | player-plays |
| blitzing:labelled | n_non_null | 3.09e+04 | 35,318 | player-plays |
| blitzing:labelled | share_non_null | 0.876 | 35,318 | player-plays |
| blitzing:null | n_non_null | 21 | 1,626 | player-plays |
| blitzing:null | share_non_null | 0.0129 | 1,626 | player-plays |

**Distribution of non-null values**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | mean | 3.11 | 30,956 | non-null player-plays |
| all | sd | 2.12 | 30,956 | non-null player-plays |
| all | min | 0.15 | 30,956 | non-null player-plays |
| all | p01 | 0.533 | 30,956 | non-null player-plays |
| all | p05 | 0.741 | 30,956 | non-null player-plays |
| all | p25 | 1.4 | 30,956 | non-null player-plays |
| all | p50 | 2.63 | 30,956 | non-null player-plays |
| all | p75 | 4.32 | 30,956 | non-null player-plays |
| all | p95 | 7.11 | 30,956 | non-null player-plays |
| all | p99 | 9.77 | 30,956 | non-null player-plays |
| all | max | 17.7 | 30,956 | non-null player-plays |
| all | share_le_zero | 0 | 30,956 | non-null player-plays |

**Athletes and observations per athlete**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | n_athletes_position | 83 | 83 | athletes of the position group with tracked plays |
| all | n_athletes_any_obs | 83 | 83 | athletes |
| all | obs_per_athlete_p10 | 25.4 | 83 | athletes |
| all | obs_per_athlete_p25 | 120 | 83 | athletes |
| all | obs_per_athlete_p50 | 250 | 83 | athletes |
| all | obs_per_athlete_p75 | 576 | 83 | athletes |
| all | obs_per_athlete_p90 | 840 | 83 | athletes |
| min_obs=20 | n_athletes | 77 | 83 | athletes with at least this many observations |
| min_obs=50 | n_athletes | 69 | 83 | athletes with at least this many observations |
| min_obs=100 | n_athletes | 63 | 83 | athletes with at least this many observations |
| min_obs=200 | n_athletes | 45 | 83 | athletes with at least this many observations |

**Overlap with Combine measurements (athletes)**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| min_obs=20 | has_cmb_forty | 71 | 77 | athletes also meeting the Combine flag |
| min_obs=20 | has_cmb_ten_yd_split | 71 | 77 | athletes also meeting the Combine flag |
| min_obs=20 | has_cmb_vertical | 68 | 77 | athletes also meeting the Combine flag |
| min_obs=20 | has_cmb_broad_jump | 69 | 77 | athletes also meeting the Combine flag |
| min_obs=20 | has_cmb_combine_weight | 77 | 77 | athletes also meeting the Combine flag |
| min_obs=20 | forty_tracked | 69 | 77 | athletes also meeting the Combine flag |
| min_obs=20 | forty_repeat | 59 | 77 | athletes also meeting the Combine flag |
| min_obs=20 | model_fit_ok_repeat | 50 | 77 | athletes also meeting the Combine flag |
| min_obs=20 | forty_tracked_and_official_both | 69 | 77 | athletes |
| min_obs=50 | has_cmb_forty | 64 | 69 | athletes also meeting the Combine flag |
| min_obs=50 | has_cmb_ten_yd_split | 64 | 69 | athletes also meeting the Combine flag |
| min_obs=50 | has_cmb_vertical | 61 | 69 | athletes also meeting the Combine flag |
| min_obs=50 | has_cmb_broad_jump | 62 | 69 | athletes also meeting the Combine flag |
| min_obs=50 | has_cmb_combine_weight | 69 | 69 | athletes also meeting the Combine flag |
| min_obs=50 | forty_tracked | 63 | 69 | athletes also meeting the Combine flag |
| min_obs=50 | forty_repeat | 53 | 69 | athletes also meeting the Combine flag |
| min_obs=50 | model_fit_ok_repeat | 45 | 69 | athletes also meeting the Combine flag |
| min_obs=50 | forty_tracked_and_official_both | 63 | 69 | athletes |
| min_obs=100 | has_cmb_forty | 58 | 63 | athletes also meeting the Combine flag |
| min_obs=100 | has_cmb_ten_yd_split | 58 | 63 | athletes also meeting the Combine flag |
| min_obs=100 | has_cmb_vertical | 55 | 63 | athletes also meeting the Combine flag |
| min_obs=100 | has_cmb_broad_jump | 56 | 63 | athletes also meeting the Combine flag |
| min_obs=100 | has_cmb_combine_weight | 63 | 63 | athletes also meeting the Combine flag |
| min_obs=100 | forty_tracked | 57 | 63 | athletes also meeting the Combine flag |
| min_obs=100 | forty_repeat | 47 | 63 | athletes also meeting the Combine flag |
| min_obs=100 | model_fit_ok_repeat | 41 | 63 | athletes also meeting the Combine flag |
| min_obs=100 | forty_tracked_and_official_both | 57 | 63 | athletes |
| min_obs=200 | has_cmb_forty | 41 | 45 | athletes also meeting the Combine flag |
| min_obs=200 | has_cmb_ten_yd_split | 41 | 45 | athletes also meeting the Combine flag |
| min_obs=200 | has_cmb_vertical | 39 | 45 | athletes also meeting the Combine flag |
| min_obs=200 | has_cmb_broad_jump | 40 | 45 | athletes also meeting the Combine flag |
| min_obs=200 | has_cmb_combine_weight | 45 | 45 | athletes also meeting the Combine flag |
| min_obs=200 | forty_tracked | 40 | 45 | athletes also meeting the Combine flag |
| min_obs=200 | forty_repeat | 31 | 45 | athletes also meeting the Combine flag |
| min_obs=200 | model_fit_ok_repeat | 27 | 45 | athletes also meeting the Combine flag |
| min_obs=200 | forty_tracked_and_official_both | 40 | 45 | athletes |

**Play context among endpoint plays**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | share_non_null:lined_up_position | 1 | 30,956 | non-null endpoint player-plays |
| all | share_non_null:blitzing | 0.999 | 30,956 | non-null endpoint player-plays |
| all | share_non_null:down | 1 | 30,956 | non-null endpoint player-plays |
| all | share_non_null:yards_to_go | 1 | 30,956 | non-null endpoint player-plays |
| all | share_non_null:offense_formation | 1 | 30,956 | non-null endpoint player-plays |
| all | share_non_null:pass_result | 1 | 30,956 | non-null endpoint player-plays |
| all | share_non_null:quarter | 1 | 30,956 | non-null endpoint player-plays |
| all | share_non_null:team_coverage_man_zone | 1 | 30,956 | non-null endpoint player-plays |
| lined_up_position:WR | share_of_endpoint_plays | 0.611 | 30,956 | player-plays |
| lined_up_position:SLOT_WR | share_of_endpoint_plays | 0.35 | 30,956 | player-plays |
| lined_up_position:TE | share_of_endpoint_plays | 0.0324 | 30,956 | player-plays |
| lined_up_position:RB | share_of_endpoint_plays | 0.00527 | 30,956 | player-plays |
| lined_up_position:FB | share_of_endpoint_plays | 0.0012 | 30,956 | player-plays |

**Repeatability**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| play | icc1 | 0.0239 | 77 | single plays, athletes with ≥ 20 observations; athlete bootstrap |
| play | icc1_ci_lower | 0.0144 | 77 | single plays, athletes with ≥ 20 observations; athlete bootstrap |
| play | icc1_ci_upper | 0.0357 | 77 | single plays, athletes with ≥ 20 observations; athlete bootstrap |
| play | projected_reliability_n=20 | 0.329 | 77 | Spearman–Brown from the single-play ICC |
| play | projected_reliability_n=50 | 0.551 | 77 | Spearman–Brown from the single-play ICC |
| play | projected_reliability_n=100 | 0.71 | 77 | Spearman–Brown from the single-play ICC |
| play | projected_reliability_n=200 | 0.831 | 77 | Spearman–Brown from the single-play ICC |
| play | plays_needed_for_0.7 | 95.2 | 77 | Spearman–Brown inverse |
| play | plays_needed_for_0.8 | 163 | 77 | Spearman–Brown inverse |
| play_context_centred | icc1 | 0.0188 | 77 | alignment × season means removed before the ICC |
| play_context_centred | icc1_ci_lower | 0.011 | 77 |  |
| play_context_centred | icc1_ci_upper | 0.0285 | 77 |  |
| game | icc1 | 0.225 | 74 | athlete-game means, games with ≥ 5 observations, athletes with ≥ 2 such games |
| game | icc1_ci_lower | 0.136 | 74 | athlete-game means, games with ≥ 5 observations, athletes with ≥ 2 such games |
| game | icc1_ci_upper | 0.295 | 74 | athlete-game means, games with ≥ 5 observations, athletes with ≥ 2 such games |
| game | games_per_athlete_median | 16.5 | 74 | athletes |
| split_half | spearman_odd_even | 0.656 | 77 | odd vs even chronological observations |
| split_half | spearman_brown_full_length | 0.792 | 77 | reliability of the athlete mean over all own observations |

**First-N-play eligibility**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| n=20 | n_athletes_ge_n | 77 | 83 | athletes with ≥ n observations overall |
| n=20 | n_athletes_ge_n_rookie_season | 70 | 83 | observations in the season equal to the draft cohort year |
| n=20 | median_games_to_reach_n | 3 | 77 | athletes reaching n observations |
| n=20 | spearman_first_n_vs_next_n | 0.4 | 70 | mean of the first n observations vs mean of the next n (athletes with ≥ 2n) |
| n=50 | n_athletes_ge_n | 69 | 83 | athletes with ≥ n observations overall |
| n=50 | n_athletes_ge_n_rookie_season | 63 | 83 | observations in the season equal to the draft cohort year |
| n=50 | median_games_to_reach_n | 5 | 69 | athletes reaching n observations |
| n=50 | spearman_first_n_vs_next_n | 0.403 | 63 | mean of the first n observations vs mean of the next n (athletes with ≥ 2n) |
| n=100 | n_athletes_ge_n | 63 | 83 | athletes with ≥ n observations overall |
| n=100 | n_athletes_ge_n_rookie_season | 52 | 83 | observations in the season equal to the draft cohort year |
| n=100 | median_games_to_reach_n | 8 | 63 | athletes reaching n observations |
| n=100 | spearman_first_n_vs_next_n | 0.493 | 45 | mean of the first n observations vs mean of the next n (athletes with ≥ 2n) |


## 10. Decision for the next modeling experiment

Each candidate is judged on measurement validity (Phase 04 status), within-position repeatability and its interval, between-athlete discrimination (signal-to-noise), incremental information beyond official tests, cohort robustness, eligible sample and football interpretability. No single threshold decides.

_Rule-based proposal from `feature_screening_summary` (heuristic labels, proposed references). It requires review and is not a result._

- **No feature family reaches the candidate label.** No tracking feature is both repeatable within position and distinguishable from the stopwatch at the heuristic references. A null incremental result is the supported reading; the next experiment should test the official times alone against the NFL endpoint, with tracking speed as a pre-registered secondary comparison.
- Redundant with the stopwatch: `peak_speed`, `peak_speed_interior`, `p95_speed`, `model_vmax`.
- Not repeatable within position: `peak_a_tan`, `p95_a_tan`, `p95_decel`, `path_smooth_yds`, `accel_sustained`, `accel_initial`, `accel_v2_5`.


The main NFL models are not started until this report has been reviewed.

## 11. Limitations and failure cases

- Two attempts per athlete: within-athlete variance rests on one difference per athlete, and intervals are wide inside small position groups.
- Repeating the forty is the athlete's choice, and the official time is usually the better run; athlete means are compared with it, with the fastest attempt as a sensitivity.
- Tracking processing differs by draft cohort (2025 has about a quarter of the positional noise of 2023/24), so cohort-specific and cohort-adjusted estimates are reported beside pooled ones.
- Censoring analyses are observational. Window variables may also proxy for how the athlete started.
- Speed-band accelerations are null when the recording starts above the lower speed level, which removes the most censored attempts from those features.
- The identifiability index and the window-adjusted ICC are diagnostics, not inferential statistics.
- Official times are treated as error-free in the redundancy analysis.
- The vendor definition of `player_get_off` is not established from repository metadata.
- Physical handedness of the Combine coordinate frame is unverified; no feature here depends on turn direction.
