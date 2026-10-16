# Combine to NFL — does tracked sprint movement predict defensive-line get-off beyond the official tests?

_Template: `reports/COMBINE_TO_NFL_DL_GETOFF_TEMPLATE.md`. Filled by `python -m nflbdb.research.combine_nfl_dl --report` into `reports/COMBINE_TO_NFL_DL_GETOFF_RESULTS.md`. Every table is read from the persisted outputs of this experiment. Text outside the tables states the design, which was fixed before any Combine value was related to any NFL value. Sentences marked "rule-based" are mechanical readings of the tables and need review._

{{run_status}}

{{gate}}

## 1. Research question

Among defensive linemen, do tracking-derived features of the Combine forty-yard dash improve the out-of-sample prediction of NFL `player_get_off` beyond the official Combine measurements?

Five things are kept apart: whether a Combine feature is repeatable (§4.2), whether it repeats the stopwatch (§5), whether the NFL outcome is a stable athlete property (§4.1), whether the feature improves prediction of held-out athletes (§7–§8), and what that means in football terms (§10). A positive answer is not assumed.

## 2. Cohort definition

Population: athletes of the defensive-line position group with a Combine results row. Combine features are computed for all of them, whether or not they later played. NFL observations: regular season, `player_get_off` present, alignment EDGE or INTERIOR_LINE, play not nullified by penalty.

{{funnel}}

**Missing predictors** (Combine DL athletes, and athletes at the primary eligibility threshold):

{{missingness}}

Body measurements and jumps are imputed with the training-fold median. The official times and the tracking features are never imputed: an athlete without them is outside the modelling population. Figure: `CN_01_cohort_funnel`.

## 3. Endpoint validity

### 3.1 Definition

{{definition}}

### 3.2 Where the value exists

{{availability}}

**Validity filter (player-plays)**

{{validity}}

### 3.3 Distribution

{{distribution}}

### 3.4 Boundary values and resolution

{{boundary}}

A spike at the maximum would mean a cap or a censoring window: observations there are then "at least this long", not a measured time. Values below the low bound are reported, not removed; the sensitivity `no_boundary_values` removes both.

### 3.5 EDGE and INTERIOR_LINE

{{alignment}}

Alignment is an NFL observation. It is never a predictor. It enters the outcome (the primary target removes alignment × season means) and the strata of §9.

### 3.6 Direction evidence (not a definition)

{{direction}}

These rows show how the value co-varies with sacks and pressures. They can suggest which direction goes with rush success. They cannot say what the clock measures.

Figure: `CN_02_getoff_distribution`.

## 4. Measurement reliability

### 4.1 NFL outcome

Reliability is estimated across independent games: an athlete's games are split into halves, the target is computed on each half, and the correlation between halves is stepped up to the full number of games. Random play splits are not used, because plays of one game share opponent, scheme and game state.

{{reliability}}

Figure: `CN_03_getoff_reliability`.

### 4.2 Combine features

{{combine_reliability}}

The window-corrected ICC of `model_tau_s` reported in the sprint rescue is an in-sample upper bound and is not used here. Figure: `CN_04_combine_reliability_vs_redundancy`.

## 5. Predictors

| Model | Predictors | Role |
|---|---|---|
| MEAN | none (training-fold mean) | reference for the out-of-sample R² |
| M0 | draft cohort, Combine weight, Combine height | context available before the draft |
| M1 | M0 + official forty + official 10-yard split | traditional testing — the model to beat |
| M2a | M1 + `p95_speed` | tracked top-end speed (expected to repeat the forty) |
| M2b | M1 + `accel_v4_7` | tracked mid-phase acceleration |
| M2c | M1 + `model_tau_s` | sprint-curve time constant |
| M2bc | M1 + `accel_v4_7` + `model_tau_s` | exploratory only |

Tracking features are the mean of an athlete's usable forty attempts (primary variant `mean|S0_all|all`), as persisted by the sprint reliability rescue. No feature search is run: the three candidates were named before this experiment.

**How much of each candidate is new information in the modelling population**

{{correlations}}

## 6. Modelling methodology

- **Unit.** One row per athlete. The target is an athlete-level summary of repeated plays; no play-level row enters a model, so play counts never stand in for sample size.
- **Primary target.** Mean `player_get_off` after removing alignment × season means, athletes with at least the primary number of valid observations. Inside each fold the context means come from the training athletes only.
- **Estimator.** Ridge regression, one penalty for all non-intercept terms. The penalty is chosen by exact leave-one-out error inside the training fold.
- **Validation.** Repeated K-fold over athletes, stratified by draft cohort; the same folds for every model. Imputation, scaling and dummy levels are fitted on the training fold.
- **Metrics.** MAE, RMSE, out-of-sample R² (skill against the training-fold mean), Pearson and Spearman between predicted and observed, calibration slope and intercept.
- **Comparison.** Paired difference against M1 on identical athletes. Interval: athlete bootstrap of the out-of-fold predictions. It reflects which athletes were sampled; it does not include the variability of refitting.
- **Not done.** No mixed-effects model on plays (the athlete-level target already aggregates them; the first-n and median targets cover the repeated-play sensitivity), no deep learning, no feature selection on any fold.

## 7. Out-of-sample results

{{comparison}}

Figures: `CN_05_forty_vs_getoff` (in-sample description), `CN_06_model_performance`.

## 8. Statistical uncertainty

Difference = model − M1. For MAE and RMSE a negative value means the model predicts held-out athletes better; for R² and Pearson a positive value does.

{{paired}}

**Standardised coefficients on all athletes (in-sample, descriptive)**

{{coefficients}}

Figure: `CN_07_incremental_gain`.

## 9. Sensitivity analyses

Each analysis changes one condition and refits every model on its own athletes; `scored_on:*` rows score the primary fit on a subset without refitting. Subgroups are small: their estimates describe stability, they are not separate findings.

{{sensitivity}}

### Matched athletes

Pairs of athletes of the same alignment group with near-identical official forty and 10-yard split, ordered by how much their tracked acceleration differs. An in-sample illustration of the question, not a held-out result.

{{matched}}

Figure: `CN_08_matched_athletes`.

## 10. Football interpretation

{{interpretation}}

**Reading of the results**

{{reading}}

## 11. Limitations

- The vendor definition of `player_get_off` is read from a file the user fills; the repository itself holds no data dictionary.
- One position group, a few dozen athletes: only large incremental effects can be detected, and an interval that includes zero does not show absence of an effect.
- Athletes reach the eligibility threshold only if they were drafted, kept and played. Selection on NFL playing time can attenuate or distort Combine–NFL associations; it is not corrected.
- Get-off depends on snap anticipation, alignment, stance, assignment, the opposing lineman and game state. It is not a pure measure of acceleration.
- Two forty attempts per athlete, with recording windows that differ between attempts and tracking noise that differs by draft cohort (2024 noisier).
- The outcome is measured with error; its reliability (§4.1) caps the variance any predictor can explain.
- The bootstrap interval conditions on the fitted out-of-fold predictions.
- Associations are predictive. Nothing here shows that training a Combine quality changes NFL get-off.

## 12. Submission relevance

This is the first test of the project's central claim: that detailed Combine movement holds information the stopwatch does not, and that the information matters on the field. A demonstrated gain would be the anchor result. A null result at this sample size is also reportable: it bounds how much a scout could gain from tracked sprint features for defensive-line get-off.

## 13. Next decision

{{next_decision}}
