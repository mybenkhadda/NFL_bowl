# Sprint reliability rescue — what does the stopwatch miss?

_Template: `reports/SPRINT_RELIABILITY_RESCUE_TEMPLATE.md`. Filled by `python -m nflbdb.research.sprint_rescue --report` into `reports/SPRINT_RELIABILITY_RESCUE_RESULTS.md`. Every table below is read from the persisted outputs of this experiment; text outside the tables states the design and facts read from source code or from earlier executed outputs. Interpretation sentences are left to the review._

{{run_status}}

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

{{window_distribution}}

Figures: `SR_01_first_frame_speed`, `SR_02_distance_vs_duration`, `SR_05_example_recordings`.

### 3.2 Within-athlete comparison (A2)

One row per athlete: attempt 2 minus attempt 1. `p95_speed` is a negative control (no window effect expected).

{{paired_association}}

Figure: `SR_03_delta_start_vs_delta_accel`.

### 3.3 Association with repeatability and fit quality (A3)

{{censoring_association}}

Figure: `SR_04_censoring_vs_repeatability`.

**How to read.** These are associations. The window-adjusted ICC is an upper bound because the slope is fitted on the same attempts. A gain whose paired interval excludes zero means window variation is compatible with part of the low ICC; it does not show that the window causes it.

### 3.4 Sensitivity to quality scenarios (A4)

{{reliability_scenarios}}

No scenario is applied by default. A feature that needs S3 to become repeatable loses about a third of the athletes with repeated attempts (Phase 04 Q).

## 4. Within-position ICC

{{reliability_scopes}}

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

{{model_identifiability}}

## 6. Model repeatability

Central comparison on identical attempts. `model_a_at_ref` / `a_at_ref` is the model acceleration at 3 yd/s, inside the recorded speed range.

{{model_repeatability}}

Figure: `SR_09_model_repeatability_and_sensitivity`.

## 7. Stopwatch redundancy

{{redundancy}}

### Feature screening

{{screening}}

Figure: `SR_10_feature_screening` (PNG, SVG, PDF). The reference lines at ICC 0.75 and |ρ| 0.70 are heuristic screening references, not acceptance thresholds. The upper-left region (repeatable, not explained by the stopwatch) is a reason to test a feature against NFL play; it is not evidence that the feature matters there.

`reliable_unique_share` is a lower bound of the reproducible athlete variance not explained by the two official times. It assumes the stopwatch explains reproducible variance only, and it treats official times as error-free.

## 8. Data eligibility

{{reconciliation}}

**Data-quality status (from `reports/tables/phase_03/descriptive/A0_dq_failures.csv`, executed).** All seven non-PASS Phase 01 checks are classified and none is blocking: the CRITICAL and MEDIUM Combine key failures and both range warnings are check-definition errors caused by BALL rows (PLAYER rows are unaffected); the two `player_play` orphan checks are coverage gaps (202 plays, 2,497 player-plays) that are excluded from context-aware features; the jump warning is handled by segmentation. The orphan gap affects Study F only: tracked plays without a `player_play` row have no endpoint value and count as missing there.

## 9. NFL endpoint feasibility

No Combine value is related to any NFL value in this section, and no model is fitted.

### 9.1 Primary: defensive line `player_get_off`

{{endpoint_dl}}

### 9.2 Secondary: wide receiver `separation_at_pass_forward`

{{endpoint_wr}}

## 10. Decision for the next modeling experiment

Each candidate is judged on measurement validity (Phase 04 status), within-position repeatability and its interval, between-athlete discrimination (signal-to-noise), incremental information beyond official tests, cohort robustness, eligible sample and football interpretability. No single threshold decides.

{{decision}}

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
