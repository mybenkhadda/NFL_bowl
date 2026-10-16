# Combine to NFL — does tracked sprint movement predict defensive-line get-off beyond the official tests?

_Template: `reports/COMBINE_TO_NFL_DL_GETOFF_TEMPLATE.md`. Filled by `python -m nflbdb.research.combine_nfl_dl --report` into `reports/COMBINE_TO_NFL_DL_GETOFF_RESULTS.md`. Every table is read from the persisted outputs of this experiment. Text outside the tables states the design, which was fixed before any Combine value was related to any NFL value. Sentences marked "rule-based" are mechanical readings of the tables and need review._

Generated 2026-10-11T12:51:35Z from persisted outputs.

| output | rows | status |
|---|---|---|
| endpoint_audit | 325 | present |
| endpoint_distribution | 76 | present |
| dl_getoff_plays | 33,292 | present |
| athlete_targets | 490 | present |
| athlete_context_cells | 1,143 | present |
| outcome_reliability | 153 | present |
| combine_feature_variants | 2,862 | present |
| athlete_research_base | 118 | present |
| cohort_funnel | 60 | present |
| cohort_missingness | 45 | present |
| predictor_correlations | 18 | present |
| fold_assignments | 1,280 | present |
| oof_predictions | 8,960 | present |
| model_comparison | 49 | present |
| bootstrap_comparisons | 24 | present |
| coefficients | 39 | present |
| matched_athletes | 18 | present |
| sensitivity_analyses | 185 | present |


> **BLOCKER — the vendor definition of `player_get_off` is unresolved (status = 'unresolved'; missing: source, timing_origin, timing_end, lower_is_better). Fill `configs/player_get_off_definition.json` from the competition data dictionary. Until then the endpoint is reported as an opaque vendor number: no direction ('faster', 'better'), no physical meaning and no coaching conclusion may be attached to any estimate.**


## 1. Research question

Among defensive linemen, do tracking-derived features of the Combine forty-yard dash improve the out-of-sample prediction of NFL `player_get_off` beyond the official Combine measurements?

Five things are kept apart: whether a Combine feature is repeatable (§4.2), whether it repeats the stopwatch (§5), whether the NFL outcome is a stable athlete property (§4.1), whether the feature improves prediction of held-out athletes (§7–§8), and what that means in football terms (§10). A positive answer is not assumed.

## 2. Cohort definition

Population: athletes of the defensive-line position group with a Combine results row. Combine features are computed for all of them, whether or not they later played. NFL observations: regular season, `player_get_off` present, alignment EDGE or INTERIOR_LINE, play not nullified by penalty.

| stage | definition | ALL | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| 01_combine_dl | DL athletes with a Combine results row | 118 | 37 | 37 | 44 |
| 02_nfl_tracking_any | … with ≥ 1 tracked NFL play (any season type) | 115 | 36 | 36 | 43 |
| 03_getoff_any | … with ≥ 1 valid regular-season get-off observation | 98 | 32 | 29 | 37 |
| 04_getoff_ge_20 | … with ≥ 20 valid observations | 93 | 32 | 26 | 35 |
| 05_getoff_ge_50 | … with ≥ 50 valid observations | 81 | 29 | 25 | 27 |
| 06_getoff_ge_100 | … with ≥ 100 valid observations | 69 | 25 | 23 | 21 |
| 07_getoff_ge_200 | … with ≥ 200 valid observations | 49 | 19 | 20 | 10 |
| 08_officials | ≥ 50 observations AND official forty + 10-yard split | 65 | 25 | 24 | 16 |
| 09_tracking_p95_speed | … AND tracked `p95_speed` (cumulative) | 64 | 25 | 23 | 16 |
| 10_tracking_accel_v4_7 | … AND tracked `accel_v4_7` (cumulative) | 64 | 25 | 23 | 16 |
| 11_tracking_model_tau_s | … AND tracked `model_tau_s` (cumulative) | 64 | 25 | 23 | 16 |
| 12_primary_population | primary modelling population (identical athletes for every model) | 64 | 25 | 23 | 16 |
| 13_population_at_20 | same predictors, ≥ 20 observations (sensitivity) | 74 | 28 | 24 | 22 |
| 14_population_at_100 | same predictors, ≥ 100 observations (sensitivity) | 53 | 21 | 21 | 11 |
| 15_population_at_200 | same predictors, ≥ 200 observations (sensitivity) | 37 | 15 | 19 | 3 |


**Missing predictors** (Combine DL athletes, and athletes at the primary eligibility threshold):

| population | variable | n_athletes | n_missing | share_missing |
|---|---|---|---|---|
| combine_dl | cmb_combine_weight | 118 | 0 | 0 |
| combine_dl | cmb_combine_height | 118 | 0 | 0 |
| combine_dl | cmb_forty | 118 | 24 | 0.203 |
| combine_dl | cmb_ten_yd_split | 118 | 24 | 0.203 |
| combine_dl | cmb_vertical | 118 | 16 | 0.136 |
| combine_dl | cmb_broad_jump | 118 | 21 | 0.178 |
| combine_dl | p95_speed | 118 | 25 | 0.212 |
| combine_dl | accel_v4_7 | 118 | 25 | 0.212 |
| combine_dl | model_tau_s | 118 | 25 | 0.212 |
| getoff_ge_50 | cmb_combine_weight | 81 | 0 | 0 |
| getoff_ge_50 | cmb_combine_height | 81 | 0 | 0 |
| getoff_ge_50 | cmb_forty | 81 | 16 | 0.198 |
| getoff_ge_50 | cmb_ten_yd_split | 81 | 16 | 0.198 |
| getoff_ge_50 | cmb_vertical | 81 | 9 | 0.111 |
| getoff_ge_50 | cmb_broad_jump | 81 | 11 | 0.136 |
| getoff_ge_50 | p95_speed | 81 | 17 | 0.21 |
| getoff_ge_50 | accel_v4_7 | 81 | 17 | 0.21 |
| getoff_ge_50 | model_tau_s | 81 | 17 | 0.21 |


Body measurements and jumps are imputed with the training-fold median. The official times and the tracking features are never imputed: an athlete without them is outside the modelling population. Figure: `CN_01_cohort_funnel`.

## 3. Endpoint validity

### 3.1 Definition

**Recorded definition (`configs/player_get_off_definition.json`)**

| field | value |
|---|---|
| status | unresolved |
| source |  |
| source_quote |  |
| timing_origin |  |
| timing_end |  |
| unit | seconds (schema_catalog, inferred) |
| valid_play_contexts |  |
| lower_is_better |  |
| censoring_or_cap |  |

**Open questions**

- Which event starts the clock (ball snap, first offensive movement, other)?
- Which event stops it (first movement of the defender, crossing the line of scrimmage, a distance travelled, contact)?
- Is the value computed on pass plays only, on pass-rush snaps only, or on every defensive-line snap?
- Is 2.0 s (the exact observed maximum) a cap, a censoring window or a coincidence?
- What do values below 0.1 s mean (anticipated snap, offside, measurement artefact)?
- Is the measurement identical for EDGE and INTERIOR_LINE alignments?

**Repository evidence**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | vendor_definition_confirmed | 0 |  | configs/player_get_off_definition.json: status = 'unresolved'; schema_catalog: meaning = 'Timing metric: player get off', unit = 'seconds', evidence = 'inferred' |
| field | source | 0 |  | unknown |
| field | timing_origin | 0 |  | unknown |
| field | timing_end | 0 |  | unknown |
| field | lower_is_better | 0 |  | unknown |
| field | valid_play_contexts | 0 |  | unknown |
| field | censoring_or_cap | 0 |  | unknown |


### 3.2 Where the value exists

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
| lined_up_position:<null> | n_non_null | 0 | 8 | player-plays |
| lined_up_position:<null> | share_non_null | 0 | 8 | player-plays |
| lined_up_position:TE | n_non_null | 0 | 1 | player-plays |
| lined_up_position:TE | share_non_null | 0 | 1 | player-plays |
| pass_result: | n_non_null | 8.13e+03 | 21,788 | player-plays |
| pass_result: | share_non_null | 0.373 | 21,788 | player-plays |
| pass_result:C | n_non_null | 1.44e+04 | 17,174 | player-plays |
| pass_result:C | share_non_null | 0.836 | 17,174 | player-plays |
| pass_result:I | n_non_null | 7.44e+03 | 9,427 | player-plays |
| pass_result:I | share_non_null | 0.79 | 9,427 | player-plays |
| pass_result:S | n_non_null | 1.62e+03 | 1,932 | player-plays |
| pass_result:S | share_non_null | 0.84 | 1,932 | player-plays |
| pass_result:R | n_non_null | 1.35e+03 | 1,655 | player-plays |
| pass_result:R | share_non_null | 0.818 | 1,655 | player-plays |
| pass_result:IN | n_non_null | 510 | 615 | player-plays |
| pass_result:IN | share_non_null | 0.829 | 615 | player-plays |
| blitzing:true | n_non_null | 2.53e+04 | 27,460 | player-plays |
| blitzing:true | share_non_null | 0.921 | 27,460 | player-plays |
| blitzing:<null> | n_non_null | 8.1e+03 | 23,591 | player-plays |
| blitzing:<null> | share_non_null | 0.344 | 23,591 | player-plays |
| blitzing:false | n_non_null | 19 | 1,540 | player-plays |
| blitzing:false | share_non_null | 0.0123 | 1,540 | player-plays |
| down:1 | n_non_null | 1.39e+04 | 23,730 | player-plays |
| down:1 | share_non_null | 0.584 | 23,730 | player-plays |
| down:2 | n_non_null | 1.13e+04 | 17,209 | player-plays |
| down:2 | share_non_null | 0.654 | 17,209 | player-plays |
| down:3 | n_non_null | 7.47e+03 | 10,175 | player-plays |
| down:3 | share_non_null | 0.734 | 10,175 | player-plays |
| down:4 | n_non_null | 828 | 1,268 | player-plays |
| down:4 | share_non_null | 0.653 | 1,268 | player-plays |
| down:0 | n_non_null | 0 | 209 | player-plays |
| down:0 | share_non_null | 0 | 209 | player-plays |
| offense_formation:SHOTGUN | n_non_null | 2.02e+04 | 27,646 | player-plays |
| offense_formation:SHOTGUN | share_non_null | 0.732 | 27,646 | player-plays |
| offense_formation:SINGLEBACK | n_non_null | 6.66e+03 | 12,988 | player-plays |
| offense_formation:SINGLEBACK | share_non_null | 0.513 | 12,988 | player-plays |
| offense_formation:EMPTY | n_non_null | 2.84e+03 | 3,441 | player-plays |
| offense_formation:EMPTY | share_non_null | 0.827 | 3,441 | player-plays |
| offense_formation:I_FORM | n_non_null | 1.63e+03 | 3,427 | player-plays |
| offense_formation:I_FORM | share_non_null | 0.477 | 3,427 | player-plays |
| offense_formation:PISTOL | n_non_null | 1.65e+03 | 3,203 | player-plays |
| offense_formation:PISTOL | share_non_null | 0.514 | 3,203 | player-plays |
| offense_formation:<null> | n_non_null | 56 | 903 | player-plays |
| offense_formation:<null> | share_non_null | 0.062 | 903 | player-plays |
| offense_formation:JUMBO | n_non_null | 295 | 777 | player-plays |
| offense_formation:JUMBO | share_non_null | 0.38 | 777 | player-plays |
| offense_formation:WILDCAT | n_non_null | 59 | 206 | player-plays |
| offense_formation:WILDCAT | share_non_null | 0.286 | 206 | player-plays |
| sack:recorded | n_non_null | 2.53e+04 | 28,970 | player-plays |
| sack:recorded | share_non_null | 0.873 | 28,970 | player-plays |
| sack:not_recorded | n_non_null | 8.12e+03 | 23,621 | player-plays |
| sack:not_recorded | share_non_null | 0.344 | 23,621 | player-plays |
| coverage_assignment:recorded | n_non_null | 2.53e+04 | 29,052 | player-plays |
| coverage_assignment:recorded | share_non_null | 0.871 | 29,052 | player-plays |
| coverage_assignment:not_recorded | n_non_null | 8.12e+03 | 23,539 | player-plays |
| coverage_assignment:not_recorded | share_non_null | 0.345 | 23,539 | player-plays |
| time_to_pressure:recorded | n_non_null | 2.61e+03 | 2,704 | player-plays |
| time_to_pressure:recorded | share_non_null | 0.964 | 2,704 | player-plays |
| time_to_pressure:not_recorded | n_non_null | 3.08e+04 | 49,887 | player-plays |
| time_to_pressure:not_recorded | share_non_null | 0.618 | 49,887 | player-plays |
| nullified_by_penalty | n_non_null | 0 | 2,184 | player-plays |
| nullified_by_penalty | share_non_null | 0 | 2,184 | player-plays |


**Validity filter (player-plays)**

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | n_endpoint_non_null | 3.34e+04 | 52,591 | player-plays |
| all | n_other_alignment | 125 | 33,417 | endpoint present, alignment not EDGE / INTERIOR_LINE (excluded) |
| all | n_nullified | 0 | 33,417 | endpoint present, play nullified by penalty (excluded) |
| all | n_valid | 3.33e+04 | 33,417 | valid observations kept |
| all | n_athletes_valid | 98 |  | athletes |


### 3.3 Distribution

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | mean | 1.01 | 33,292 | valid player-plays |
| all | sd | 0.293 | 33,292 | valid player-plays |
| all | min | 0.004 | 33,292 | valid player-plays |
| all | p01 | 0.506 | 33,292 | valid player-plays |
| all | p05 | 0.628 | 33,292 | valid player-plays |
| all | p25 | 0.805 | 33,292 | valid player-plays |
| all | p50 | 0.959 | 33,292 | valid player-plays |
| all | p75 | 1.16 | 33,292 | valid player-plays |
| all | p95 | 1.61 | 33,292 | valid player-plays |
| all | p99 | 1.88 | 33,292 | valid player-plays |
| all | max | 2 | 33,292 | valid player-plays |
| alignment:EDGE | mean | 0.966 | 20,219 | valid player-plays |
| alignment:EDGE | sd | 0.29 | 20,219 | valid player-plays |
| alignment:EDGE | min | 0.004 | 20,219 | valid player-plays |
| alignment:EDGE | p01 | 0.477 | 20,219 | valid player-plays |
| alignment:EDGE | p05 | 0.595 | 20,219 | valid player-plays |
| alignment:EDGE | p25 | 0.761 | 20,219 | valid player-plays |
| alignment:EDGE | p50 | 0.907 | 20,219 | valid player-plays |
| alignment:EDGE | p75 | 1.12 | 20,219 | valid player-plays |
| alignment:EDGE | p95 | 1.55 | 20,219 | valid player-plays |
| alignment:EDGE | p99 | 1.85 | 20,219 | valid player-plays |
| alignment:EDGE | max | 2 | 20,219 | valid player-plays |
| alignment:INTERIOR_LINE | mean | 1.08 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | sd | 0.284 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | min | 0.024 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | p01 | 0.602 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | p05 | 0.717 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | p25 | 0.886 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | p50 | 1.02 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | p75 | 1.22 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | p95 | 1.68 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | p99 | 1.92 | 13,073 | valid player-plays |
| alignment:INTERIOR_LINE | max | 2 | 13,073 | valid player-plays |
| season:2023 | mean | 0.993 | 4,848 | valid player-plays |
| season:2023 | sd | 0.286 | 4,848 | valid player-plays |
| season:2023 | min | 0.329 | 4,848 | valid player-plays |
| season:2023 | p01 | 0.52 | 4,848 | valid player-plays |
| season:2023 | p05 | 0.627 | 4,848 | valid player-plays |
| season:2023 | p25 | 0.789 | 4,848 | valid player-plays |
| season:2023 | p50 | 0.941 | 4,848 | valid player-plays |
| season:2023 | p75 | 1.14 | 4,848 | valid player-plays |
| season:2023 | p95 | 1.57 | 4,848 | valid player-plays |
| season:2023 | p99 | 1.85 | 4,848 | valid player-plays |
| season:2023 | max | 2 | 4,848 | valid player-plays |
| season:2024 | mean | 0.994 | 11,123 | valid player-plays |
| season:2024 | sd | 0.297 | 11,123 | valid player-plays |
| season:2024 | min | 0.135 | 11,123 | valid player-plays |
| season:2024 | p01 | 0.479 | 11,123 | valid player-plays |
| season:2024 | p05 | 0.603 | 11,123 | valid player-plays |
| season:2024 | p25 | 0.784 | 11,123 | valid player-plays |
| season:2024 | p50 | 0.944 | 11,123 | valid player-plays |
| season:2024 | p75 | 1.15 | 11,123 | valid player-plays |
| season:2024 | p95 | 1.6 | 11,123 | valid player-plays |
| season:2024 | p99 | 1.88 | 11,123 | valid player-plays |
| season:2024 | max | 2 | 11,123 | valid player-plays |
| season:2025 | mean | 1.03 | 17,321 | valid player-plays |
| season:2025 | sd | 0.291 | 17,321 | valid player-plays |
| season:2025 | min | 0.004 | 17,321 | valid player-plays |
| season:2025 | p01 | 0.528 | 17,321 | valid player-plays |
| season:2025 | p05 | 0.646 | 17,321 | valid player-plays |
| season:2025 | p25 | 0.823 | 17,321 | valid player-plays |
| season:2025 | p50 | 0.974 | 17,321 | valid player-plays |
| season:2025 | p75 | 1.18 | 17,321 | valid player-plays |
| season:2025 | p95 | 1.62 | 17,321 | valid player-plays |
| season:2025 | p99 | 1.89 | 17,321 | valid player-plays |
| season:2025 | max | 2 | 17,321 | valid player-plays |
| pass_result: | mean | 1.17 | 8,049 | valid player-plays |
| pass_result: | sd | 0.313 | 8,049 | valid player-plays |
| pass_result: | min | 0.135 | 8,049 | valid player-plays |
| pass_result: | p01 | 0.575 | 8,049 | valid player-plays |
| pass_result: | p05 | 0.718 | 8,049 | valid player-plays |
| pass_result: | p25 | 0.94 | 8,049 | valid player-plays |
| pass_result: | p50 | 1.13 | 8,049 | valid player-plays |
| pass_result: | p75 | 1.36 | 8,049 | valid player-plays |
| pass_result: | p95 | 1.77 | 8,049 | valid player-plays |
| pass_result: | p99 | 1.95 | 8,049 | valid player-plays |
| pass_result: | max | 2 | 8,049 | valid player-plays |
| pass_result:C | mean | 0.969 | 14,348 | valid player-plays |
| pass_result:C | sd | 0.269 | 14,348 | valid player-plays |
| pass_result:C | min | 0.004 | 14,348 | valid player-plays |
| pass_result:C | p01 | 0.505 | 14,348 | valid player-plays |
| pass_result:C | p05 | 0.618 | 14,348 | valid player-plays |
| pass_result:C | p25 | 0.784 | 14,348 | valid player-plays |
| pass_result:C | p50 | 0.923 | 14,348 | valid player-plays |
| pass_result:C | p75 | 1.1 | 14,348 | valid player-plays |
| pass_result:C | p95 | 1.51 | 14,348 | valid player-plays |
| pass_result:C | p99 | 1.84 | 14,348 | valid player-plays |
| pass_result:C | max | 2 | 14,348 | valid player-plays |
| pass_result:I | mean | 0.954 | 7,435 | valid player-plays |
| pass_result:I | sd | 0.265 | 7,435 | valid player-plays |
| pass_result:I | min | 0.024 | 7,435 | valid player-plays |
| pass_result:I | p01 | 0.488 | 7,435 | valid player-plays |
| pass_result:I | p05 | 0.608 | 7,435 | valid player-plays |
| pass_result:I | p25 | 0.775 | 7,435 | valid player-plays |
| pass_result:I | p50 | 0.911 | 7,435 | valid player-plays |
| pass_result:I | p75 | 1.08 | 7,435 | valid player-plays |
| pass_result:I | p95 | 1.48 | 7,435 | valid player-plays |
| pass_result:I | p99 | 1.83 | 7,435 | valid player-plays |
| pass_result:I | max | 2 | 7,435 | valid player-plays |
| pass_result:IN | mean | 0.965 | 510 | valid player-plays |
| pass_result:IN | sd | 0.267 | 510 | valid player-plays |
| pass_result:IN | min | 0.327 | 510 | valid player-plays |
| pass_result:IN | p01 | 0.478 | 510 | valid player-plays |
| pass_result:IN | p05 | 0.633 | 510 | valid player-plays |
| pass_result:IN | p25 | 0.78 | 510 | valid player-plays |
| pass_result:IN | p50 | 0.923 | 510 | valid player-plays |
| pass_result:IN | p75 | 1.11 | 510 | valid player-plays |
| pass_result:IN | p95 | 1.48 | 510 | valid player-plays |
| pass_result:IN | p99 | 1.81 | 510 | valid player-plays |
| pass_result:IN | max | 1.94 | 510 | valid player-plays |
| pass_result:R | mean | 0.966 | 1,328 | valid player-plays |
| pass_result:R | sd | 0.277 | 1,328 | valid player-plays |
| pass_result:R | min | 0.344 | 1,328 | valid player-plays |
| pass_result:R | p01 | 0.465 | 1,328 | valid player-plays |
| pass_result:R | p05 | 0.605 | 1,328 | valid player-plays |
| pass_result:R | p25 | 0.774 | 1,328 | valid player-plays |
| pass_result:R | p50 | 0.919 | 1,328 | valid player-plays |
| pass_result:R | p75 | 1.1 | 1,328 | valid player-plays |
| pass_result:R | p95 | 1.51 | 1,328 | valid player-plays |
| pass_result:R | p99 | 1.85 | 1,328 | valid player-plays |
| pass_result:R | max | 1.99 | 1,328 | valid player-plays |
| pass_result:S | mean | 0.931 | 1,622 | valid player-plays |
| pass_result:S | sd | 0.263 | 1,622 | valid player-plays |
| pass_result:S | min | 0.309 | 1,622 | valid player-plays |
| pass_result:S | p01 | 0.481 | 1,622 | valid player-plays |
| pass_result:S | p05 | 0.588 | 1,622 | valid player-plays |
| pass_result:S | p25 | 0.754 | 1,622 | valid player-plays |
| pass_result:S | p50 | 0.897 | 1,622 | valid player-plays |
| pass_result:S | p75 | 1.05 | 1,622 | valid player-plays |
| pass_result:S | p95 | 1.43 | 1,622 | valid player-plays |
| pass_result:S | p99 | 1.8 | 1,622 | valid player-plays |
| pass_result:S | max | 1.99 | 1,622 | valid player-plays |
| blitzing:<null> | mean | 1.17 | 8,019 | valid player-plays |
| blitzing:<null> | sd | 0.313 | 8,019 | valid player-plays |
| blitzing:<null> | min | 0.135 | 8,019 | valid player-plays |
| blitzing:<null> | p01 | 0.576 | 8,019 | valid player-plays |
| blitzing:<null> | p05 | 0.718 | 8,019 | valid player-plays |
| blitzing:<null> | p25 | 0.94 | 8,019 | valid player-plays |
| blitzing:<null> | p50 | 1.13 | 8,019 | valid player-plays |
| blitzing:<null> | p75 | 1.36 | 8,019 | valid player-plays |
| blitzing:<null> | p95 | 1.77 | 8,019 | valid player-plays |
| blitzing:<null> | p99 | 1.95 | 8,019 | valid player-plays |
| blitzing:<null> | max | 2 | 8,019 | valid player-plays |
| blitzing:true | mean | 0.962 | 25,255 | valid player-plays |
| blitzing:true | sd | 0.268 | 25,255 | valid player-plays |
| blitzing:true | min | 0.004 | 25,255 | valid player-plays |
| blitzing:true | p01 | 0.496 | 25,255 | valid player-plays |
| blitzing:true | p05 | 0.613 | 25,255 | valid player-plays |
| blitzing:true | p25 | 0.779 | 25,255 | valid player-plays |
| blitzing:true | p50 | 0.917 | 25,255 | valid player-plays |
| blitzing:true | p75 | 1.09 | 25,255 | valid player-plays |
| blitzing:true | p95 | 1.5 | 25,255 | valid player-plays |
| blitzing:true | p99 | 1.83 | 25,255 | valid player-plays |
| blitzing:true | max | 2 | 25,255 | valid player-plays |
| down:1 | mean | 1.07 | 13,805 | valid player-plays |
| down:1 | sd | 0.307 | 13,805 | valid player-plays |
| down:1 | min | 0.216 | 13,805 | valid player-plays |
| down:1 | p01 | 0.527 | 13,805 | valid player-plays |
| down:1 | p05 | 0.658 | 13,805 | valid player-plays |
| down:1 | p25 | 0.855 | 13,805 | valid player-plays |
| down:1 | p50 | 1.02 | 13,805 | valid player-plays |
| down:1 | p75 | 1.25 | 13,805 | valid player-plays |
| down:1 | p95 | 1.69 | 13,805 | valid player-plays |
| down:1 | p99 | 1.92 | 13,805 | valid player-plays |
| down:1 | max | 2 | 13,805 | valid player-plays |
| down:2 | mean | 1 | 11,222 | valid player-plays |
| down:2 | sd | 0.282 | 11,222 | valid player-plays |
| down:2 | min | 0.004 | 11,222 | valid player-plays |
| down:2 | p01 | 0.515 | 11,222 | valid player-plays |
| down:2 | p05 | 0.633 | 11,222 | valid player-plays |
| down:2 | p25 | 0.809 | 11,222 | valid player-plays |
| down:2 | p50 | 0.956 | 11,222 | valid player-plays |
| down:2 | p75 | 1.15 | 11,222 | valid player-plays |
| down:2 | p95 | 1.58 | 11,222 | valid player-plays |
| down:2 | p99 | 1.87 | 11,222 | valid player-plays |
| down:2 | max | 2 | 11,222 | valid player-plays |
| down:3 | mean | 0.909 | 7,440 | valid player-plays |
| down:3 | sd | 0.25 | 7,440 | valid player-plays |
| down:3 | min | 0.024 | 7,440 | valid player-plays |
| down:3 | p01 | 0.474 | 7,440 | valid player-plays |
| down:3 | p05 | 0.584 | 7,440 | valid player-plays |
| down:3 | p25 | 0.745 | 7,440 | valid player-plays |
| down:3 | p50 | 0.869 | 7,440 | valid player-plays |
| down:3 | p75 | 1.02 | 7,440 | valid player-plays |
| down:3 | p95 | 1.39 | 7,440 | valid player-plays |
| down:3 | p99 | 1.78 | 7,440 | valid player-plays |
| down:3 | max | 2 | 7,440 | valid player-plays |
| down:4 | mean | 0.962 | 825 | valid player-plays |
| down:4 | sd | 0.272 | 825 | valid player-plays |
| down:4 | min | 0.397 | 825 | valid player-plays |
| down:4 | p01 | 0.504 | 825 | valid player-plays |
| down:4 | p05 | 0.626 | 825 | valid player-plays |
| down:4 | p25 | 0.77 | 825 | valid player-plays |
| down:4 | p50 | 0.911 | 825 | valid player-plays |
| down:4 | p75 | 1.1 | 825 | valid player-plays |
| down:4 | p95 | 1.5 | 825 | valid player-plays |
| down:4 | p99 | 1.81 | 825 | valid player-plays |
| down:4 | max | 1.99 | 825 | valid player-plays |


### 3.4 Boundary values and resolution

| stratum | metric | value | n | note |
|---|---|---|---|---|
| all | max_value | 2 | 33,292 | largest value observed [s] |
| all | share_at_max | 6.01e-05 | 33,292 | within 0.001 s of the maximum; a spike here means a cap or a censoring window |
| all | share_top_5pct_of_range | 0.00868 | 33,292 | value ≥ 0.95 · max |
| all | share_below_low | 6.01e-05 | 33,292 | value < 0.1 s |
| all | n_distinct_values | 1.63e+03 | 33,292 |  |
| all | share_on_1_decimal_grid | 0.0112 | 33,292 | numeric resolution of the delivered value |
| all | share_on_2_decimal_grid | 0.101 | 33,292 | numeric resolution of the delivered value |
| all | share_on_3_decimal_grid | 1 | 33,292 | numeric resolution of the delivered value |
| alignment:EDGE | share_at_max | 0 | 20,219 |  |
| alignment:EDGE | share_below_low | 4.95e-05 | 20,219 |  |
| alignment:INTERIOR_LINE | share_at_max | 0.000153 | 13,073 |  |
| alignment:INTERIOR_LINE | share_below_low | 7.65e-05 | 13,073 |  |


A spike at the maximum would mean a cap or a censoring window: observations there are then "at least this long", not a measured time. Values below the low bound are reported, not removed; the sensitivity `no_boundary_values` removes both.

### 3.5 EDGE and INTERIOR_LINE

| stratum | metric | value | n | note |
|---|---|---|---|---|
| athletes | n_mostly_EDGE | 39 | 98 | ≥ 90 % of observations |
| athletes | n_mostly_INTERIOR_LINE | 36 | 98 | ≥ 90 % of observations |
| athletes | n_mixed | 23 | 98 |  |
| within_athlete | mean_difference_s | -0.0473 | 44 | EDGE − INTERIOR_LINE, athletes with ≥ 10 observations in each alignment; athlete bootstrap |
| within_athlete | mean_difference_ci_lower | -0.0787 | 44 | EDGE − INTERIOR_LINE, athletes with ≥ 10 observations in each alignment; athlete bootstrap |
| within_athlete | mean_difference_ci_upper | -0.0151 | 44 | EDGE − INTERIOR_LINE, athletes with ≥ 10 observations in each alignment; athlete bootstrap |


Alignment is an NFL observation. It is never a predictor. It enters the outcome (the primary target removes alignment × season means) and the strata of §9.

### 3.6 Direction evidence (not a definition)

| stratum | metric | value | n | note |
|---|---|---|---|---|
| sack | mean_value_when_true | 0.896 | 380 | plays with sack > 0 |
| sack | mean_value_when_false | 1.01 | 32,912 | plays without sack > 0 |
| sack | athlete_partial_spearman | -0.421 | 81 | athlete mean get-off vs athlete rate of 'sack > 0', given the athlete's alignment share; athletes with ≥ 50 observations; Fisher interval. Evidence about direction only |
| sack | athlete_partial_spearman_ci_lower | -0.587 | 81 | athlete mean get-off vs athlete rate of 'sack > 0', given the athlete's alignment share; athletes with ≥ 50 observations; Fisher interval. Evidence about direction only |
| sack | athlete_partial_spearman_ci_upper | -0.222 | 81 | athlete mean get-off vs athlete rate of 'sack > 0', given the athlete's alignment share; athletes with ≥ 50 observations; Fisher interval. Evidence about direction only |
| time_to_pressure | mean_value_when_true | 0.888 | 2,604 | plays with pressure timed |
| time_to_pressure | mean_value_when_false | 1.02 | 30,688 | plays without pressure timed |
| time_to_pressure | athlete_partial_spearman | -0.481 | 81 | athlete mean get-off vs athlete rate of 'pressure timed', given the athlete's alignment share; athletes with ≥ 50 observations; Fisher interval. Evidence about direction only |
| time_to_pressure | athlete_partial_spearman_ci_lower | -0.634 | 81 | athlete mean get-off vs athlete rate of 'pressure timed', given the athlete's alignment share; athletes with ≥ 50 observations; Fisher interval. Evidence about direction only |
| time_to_pressure | athlete_partial_spearman_ci_upper | -0.292 | 81 | athlete mean get-off vs athlete rate of 'pressure timed', given the athlete's alignment share; athletes with ≥ 50 observations; Fisher interval. Evidence about direction only |
| unblocked_pressure | mean_value_when_true | 1.01 | 232 | plays with unblocked pressure |
| unblocked_pressure | mean_value_when_false | 1.01 | 33,060 | plays without unblocked pressure |
| unblocked_pressure | athlete_partial_spearman | -0.163 | 81 | athlete mean get-off vs athlete rate of 'unblocked pressure', given the athlete's alignment share; athletes with ≥ 50 observations; Fisher interval. Evidence about direction only |
| unblocked_pressure | athlete_partial_spearman_ci_lower | -0.369 | 81 | athlete mean get-off vs athlete rate of 'unblocked pressure', given the athlete's alignment share; athletes with ≥ 50 observations; Fisher interval. Evidence about direction only |
| unblocked_pressure | athlete_partial_spearman_ci_upper | 0.0588 | 81 | athlete mean get-off vs athlete rate of 'unblocked pressure', given the athlete's alignment share; athletes with ≥ 50 observations; Fisher interval. Evidence about direction only |


These rows show how the value co-varies with sacks and pressures. They can suggest which direction goes with rush success. They cannot say what the clock measures.

Figure: `CN_02_getoff_distribution`.

## 4. Measurement reliability

### 4.1 NFL outcome

Reliability is estimated across independent games: an athlete's games are split into halves, the target is computed on each half, and the correlation between halves is stepped up to the full number of games. Random play splits are not used, because plays of one game share opponent, scheme and game state.

**Across independent games, every observation, by eligibility threshold**

| target | window | min_obs | method | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| adj_mean | all | 20 | game_odd_even | 93 | 0.733 | 0.582 | 0.839 |
| mean | all | 20 | game_odd_even | 93 | 0.828 | 0.736 | 0.897 |
| median | all | 20 | game_odd_even | 93 | 0.785 | 0.638 | 0.887 |
| p25 | all | 20 | game_odd_even | 93 | 0.857 | 0.784 | 0.914 |
| p75 | all | 20 | game_odd_even | 93 | 0.754 | 0.634 | 0.849 |
| adj_mean | all | 20 | game_random_halves_sb | 93 | 0.811 | 0.727 | 0.878 |
| mean | all | 20 | game_random_halves_sb | 93 | 0.888 | 0.842 | 0.93 |
| median | all | 20 | game_random_halves_sb | 93 | 0.873 | 0.83 | 0.911 |
| p25 | all | 20 | game_random_halves_sb | 93 | 0.898 | 0.872 | 0.922 |
| p75 | all | 20 | game_random_halves_sb | 93 | 0.828 | 0.771 | 0.884 |
| mean | all | 20 | game_icc1 | 89 | 0.411 | 0.324 | 0.487 |
| adj_mean | all | 20 | game_icc1 | 89 | 0.243 | 0.162 | 0.315 |
| adj_mean | all | 50 | game_odd_even | 81 | 0.746 | 0.628 | 0.847 |
| mean | all | 50 | game_odd_even | 81 | 0.844 | 0.767 | 0.906 |
| median | all | 50 | game_odd_even | 81 | 0.831 | 0.746 | 0.902 |
| p25 | all | 50 | game_odd_even | 81 | 0.879 | 0.826 | 0.926 |
| p75 | all | 50 | game_odd_even | 81 | 0.771 | 0.66 | 0.86 |
| adj_mean | all | 50 | game_random_halves_sb | 81 | 0.835 | 0.773 | 0.885 |
| mean | all | 50 | game_random_halves_sb | 81 | 0.906 | 0.871 | 0.934 |
| median | all | 50 | game_random_halves_sb | 81 | 0.897 | 0.863 | 0.924 |
| p25 | all | 50 | game_random_halves_sb | 81 | 0.923 | 0.898 | 0.945 |
| p75 | all | 50 | game_random_halves_sb | 81 | 0.846 | 0.787 | 0.892 |
| mean | all | 50 | game_icc1 | 81 | 0.413 | 0.324 | 0.49 |
| adj_mean | all | 50 | game_icc1 | 81 | 0.247 | 0.168 | 0.319 |
| adj_mean | all | 100 | game_odd_even | 69 | 0.857 | 0.79 | 0.906 |
| mean | all | 100 | game_odd_even | 69 | 0.912 | 0.874 | 0.941 |
| median | all | 100 | game_odd_even | 69 | 0.911 | 0.866 | 0.944 |
| p25 | all | 100 | game_odd_even | 69 | 0.914 | 0.877 | 0.944 |
| p75 | all | 100 | game_odd_even | 69 | 0.85 | 0.783 | 0.904 |
| adj_mean | all | 100 | game_random_halves_sb | 69 | 0.871 | 0.817 | 0.914 |
| mean | all | 100 | game_random_halves_sb | 69 | 0.93 | 0.902 | 0.953 |
| median | all | 100 | game_random_halves_sb | 69 | 0.927 | 0.9 | 0.951 |
| p25 | all | 100 | game_random_halves_sb | 69 | 0.935 | 0.908 | 0.955 |
| p75 | all | 100 | game_random_halves_sb | 69 | 0.881 | 0.834 | 0.917 |
| mean | all | 100 | game_icc1 | 69 | 0.403 | 0.308 | 0.482 |
| adj_mean | all | 100 | game_icc1 | 69 | 0.234 | 0.154 | 0.311 |
| adj_mean | all | 200 | game_odd_even | 49 | 0.852 | 0.761 | 0.914 |
| mean | all | 200 | game_odd_even | 49 | 0.92 | 0.876 | 0.953 |
| median | all | 200 | game_odd_even | 49 | 0.906 | 0.848 | 0.949 |
| p25 | all | 200 | game_odd_even | 49 | 0.91 | 0.859 | 0.952 |
| p75 | all | 200 | game_odd_even | 49 | 0.856 | 0.779 | 0.917 |
| adj_mean | all | 200 | game_random_halves_sb | 49 | 0.89 | 0.835 | 0.926 |
| mean | all | 200 | game_random_halves_sb | 49 | 0.943 | 0.917 | 0.962 |
| median | all | 200 | game_random_halves_sb | 49 | 0.934 | 0.909 | 0.957 |
| p25 | all | 200 | game_random_halves_sb | 49 | 0.946 | 0.922 | 0.964 |
| p75 | all | 200 | game_random_halves_sb | 49 | 0.907 | 0.857 | 0.941 |
| mean | all | 200 | game_icc1 | 49 | 0.4 | 0.297 | 0.487 |
| adj_mean | all | 200 | game_icc1 | 49 | 0.231 | 0.143 | 0.314 |

Reliability of the primary target (`adj_mean`, ≥ 50 observations, random game halves stepped up): **0.835**. No predictor can explain more than about this share of its variance.

**First n observations only**

| target | window | min_obs | method | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| adj_mean | first_20 | 20 | game_random_halves_sb | 82 | 0.304 | 0.198 | 0.417 |
| mean | first_20 | 20 | game_random_halves_sb | 82 | 0.541 | 0.481 | 0.61 |
| median | first_20 | 20 | game_random_halves_sb | 82 | 0.493 | 0.42 | 0.567 |
| p25 | first_20 | 20 | game_random_halves_sb | 82 | 0.363 | 0.275 | 0.452 |
| p75 | first_20 | 20 | game_random_halves_sb | 82 | 0.481 | 0.4 | 0.555 |
| adj_mean | first_50 | 50 | game_random_halves_sb | 81 | 0.467 | 0.339 | 0.582 |
| mean | first_50 | 50 | game_random_halves_sb | 81 | 0.704 | 0.638 | 0.769 |
| median | first_50 | 50 | game_random_halves_sb | 81 | 0.692 | 0.616 | 0.755 |
| p25 | first_50 | 50 | game_random_halves_sb | 81 | 0.729 | 0.673 | 0.778 |
| p75 | first_50 | 50 | game_random_halves_sb | 81 | 0.601 | 0.482 | 0.706 |
| adj_mean | first_100 | 100 | game_random_halves_sb | 69 | 0.66 | 0.548 | 0.759 |
| mean | first_100 | 100 | game_random_halves_sb | 69 | 0.826 | 0.771 | 0.875 |
| median | first_100 | 100 | game_random_halves_sb | 69 | 0.817 | 0.764 | 0.868 |
| p25 | first_100 | 100 | game_random_halves_sb | 69 | 0.838 | 0.788 | 0.884 |
| p75 | first_100 | 100 | game_random_halves_sb | 69 | 0.742 | 0.664 | 0.812 |

**Single observations and the Spearman–Brown projection**

| target | window | min_obs | method | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| mean | all | 20 | play_icc1 | 93 | 0.0972 | 0.0657 | 0.129 |
| mean | all | 20 | sb_projected:20 | 93 | 0.683 | 0.585 | 0.747 |
| mean | all | 20 | sb_projected:50 | 93 | 0.843 | 0.779 | 0.881 |
| mean | all | 20 | sb_projected:100 | 93 | 0.915 | 0.876 | 0.937 |
| mean | all | 20 | sb_projected:200 | 93 | 0.956 | 0.934 | 0.967 |
| adj_mean | all | 20 | play_icc1 | 93 | 0.0508 | 0.0317 | 0.0703 |
| adj_mean | all | 20 | sb_projected:20 | 93 | 0.517 | 0.396 | 0.602 |
| adj_mean | all | 20 | sb_projected:50 | 93 | 0.728 | 0.621 | 0.791 |
| adj_mean | all | 20 | sb_projected:100 | 93 | 0.843 | 0.766 | 0.883 |
| adj_mean | all | 20 | sb_projected:200 | 93 | 0.915 | 0.867 | 0.938 |


Figure: `CN_03_getoff_reliability`.

### 4.2 Combine features

**Repeatability between attempts, defensive line (sprint rescue, executed)**

| feature | n_repeat | icc1 | ci_lower | ci_upper |
|---|---|---|---|---|
| p95_speed | 84 | 0.976 | 0.962 | 0.985 |
| accel_v4_7 | 84 | 0.832 | 0.724 | 0.905 |
| model_tau_s | 80 | 0.676 | 0.522 | 0.788 |

**Association with official times, defensive line (sprint rescue, executed)**

| feature | official | n_athletes | spearman | spearman_lower | spearman_upper |
|---|---|---|---|---|---|
| p95_speed | cmb_forty | 93 | -0.977 | -0.985 | -0.966 |
| p95_speed | cmb_ten_yd_split | 93 | -0.879 | -0.918 | -0.822 |
| accel_v4_7 | cmb_forty | 93 | -0.824 | -0.88 | -0.745 |
| accel_v4_7 | cmb_ten_yd_split | 93 | -0.783 | -0.851 | -0.69 |
| model_tau_s | cmb_forty | 93 | -0.151 | -0.344 | 0.0544 |
| model_tau_s | cmb_ten_yd_split | 93 | -0.0893 | -0.288 | 0.117 |


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

**Cohort-adjusted Spearman in the modelling population**

| feature | other | n_athletes | spearman | ci_lower | ci_upper |
|---|---|---|---|---|---|
| p95_speed | cmb_forty | 64 | -0.98 | -0.988 | -0.967 |
| p95_speed | cmb_ten_yd_split | 64 | -0.887 | -0.931 | -0.819 |
| p95_speed | cmb_combine_weight | 64 | -0.812 | -0.883 | -0.705 |
| p95_speed | accel_v4_7 | 64 | 0.852 | 0.766 | 0.909 |
| p95_speed | model_tau_s | 64 | 0.291 | 0.0447 | 0.504 |
| accel_v4_7 | cmb_forty | 64 | -0.878 | -0.925 | -0.805 |
| accel_v4_7 | cmb_ten_yd_split | 64 | -0.838 | -0.899 | -0.743 |
| accel_v4_7 | cmb_combine_weight | 64 | -0.788 | -0.867 | -0.67 |
| accel_v4_7 | p95_speed | 64 | 0.852 | 0.766 | 0.909 |
| accel_v4_7 | model_tau_s | 64 | -0.134 | -0.371 | 0.12 |
| model_tau_s | cmb_forty | 64 | -0.244 | -0.466 | 0.00563 |
| model_tau_s | cmb_ten_yd_split | 64 | -0.166 | -0.399 | 0.0873 |
| model_tau_s | cmb_combine_weight | 64 | -0.134 | -0.372 | 0.119 |
| model_tau_s | p95_speed | 64 | 0.291 | 0.0447 | 0.504 |
| model_tau_s | accel_v4_7 | 64 | -0.134 | -0.371 | 0.12 |

**Share of each candidate explained by all M1 predictors, and what is left**

| feature | n_athletes | r2 | residual_sd |
|---|---|---|---|
| p95_speed | 64 | 0.969 | 0.107 |
| accel_v4_7 | 64 | 0.791 | 0.299 |
| model_tau_s | 64 | 0.135 | 0.132 |


## 6. Modelling methodology

- **Unit.** One row per athlete. The target is an athlete-level summary of repeated plays; no play-level row enters a model, so play counts never stand in for sample size.
- **Primary target.** Mean `player_get_off` after removing alignment × season means, athletes with at least the primary number of valid observations. Inside each fold the context means come from the training athletes only.
- **Estimator.** Ridge regression, one penalty for all non-intercept terms. The penalty is chosen by exact leave-one-out error inside the training fold.
- **Validation.** Repeated K-fold over athletes, stratified by draft cohort; the same folds for every model. Imputation, scaling and dummy levels are fitted on the training fold.
- **Metrics.** MAE, RMSE, out-of-sample R² (skill against the training-fold mean), Pearson and Spearman between predicted and observed, calibration slope and intercept.
- **Comparison.** Paired difference against M1 on identical athletes. Interval: athlete bootstrap of the out-of-fold predictions. It reflects which athletes were sampled; it does not include the variability of refitting.
- **Not done.** No mixed-effects model on plays (the athlete-level target already aggregates them; the first-n and median targets cover the repeated-play sensitivity), no deep learning, no feature selection on any fold.

## 7. Out-of-sample results

**rmse**

| model | label | exploratory | n_predictors | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| MEAN | training mean | False | 0 | 64 | 0.0826 | 0.0702 | 0.0952 |
| M0 | cohort + body | False | 4 | 64 | 0.0814 | 0.0686 | 0.0943 |
| M1 | M0 + official forty and split | False | 6 | 64 | 0.0801 | 0.0671 | 0.0935 |
| M2a | M1 + tracked speed | False | 7 | 64 | 0.0803 | 0.0671 | 0.0937 |
| M2b | M1 + accel 4–7 yd/s | False | 7 | 64 | 0.0794 | 0.0664 | 0.0926 |
| M2c | M1 + sprint time constant | False | 7 | 64 | 0.0797 | 0.0668 | 0.0926 |
| M2bc | M1 + both acceleration features | True | 8 | 64 | 0.0793 | 0.0666 | 0.0922 |

**mae**

| model | label | exploratory | n_predictors | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| MEAN | training mean | False | 0 | 64 | 0.0676 | 0.0566 | 0.0804 |
| M0 | cohort + body | False | 4 | 64 | 0.0656 | 0.0543 | 0.0781 |
| M1 | M0 + official forty and split | False | 6 | 64 | 0.0636 | 0.0523 | 0.0764 |
| M2a | M1 + tracked speed | False | 7 | 64 | 0.0637 | 0.0524 | 0.0766 |
| M2b | M1 + accel 4–7 yd/s | False | 7 | 64 | 0.0629 | 0.0514 | 0.0759 |
| M2c | M1 + sprint time constant | False | 7 | 64 | 0.0633 | 0.052 | 0.0763 |
| M2bc | M1 + both acceleration features | True | 8 | 64 | 0.0631 | 0.0516 | 0.076 |

**r2_oos**

| model | label | exploratory | n_predictors | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| MEAN | training mean | False | 0 | 64 | 0 | 0 | 0 |
| M0 | cohort + body | False | 4 | 64 | 0.0301 | -0.0552 | 0.106 |
| M1 | M0 + official forty and split | False | 6 | 64 | 0.0613 | -0.0438 | 0.161 |
| M2a | M1 + tracked speed | False | 7 | 64 | 0.0564 | -0.0496 | 0.158 |
| M2b | M1 + accel 4–7 yd/s | False | 7 | 64 | 0.076 | -0.0371 | 0.187 |
| M2c | M1 + sprint time constant | False | 7 | 64 | 0.0707 | -0.0399 | 0.176 |
| M2bc | M1 + both acceleration features | True | 8 | 64 | 0.0789 | -0.0379 | 0.188 |

**pearson**

| model | label | exploratory | n_predictors | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| MEAN | training mean | False | 0 | 64 | -0.174 | -0.24 | -0.104 |
| M0 | cohort + body | False | 4 | 64 | 0.0759 | -0.133 | 0.272 |
| M1 | M0 + official forty and split | False | 6 | 64 | 0.164 | -0.0621 | 0.384 |
| M2a | M1 + tracked speed | False | 7 | 64 | 0.155 | -0.0736 | 0.376 |
| M2b | M1 + accel 4–7 yd/s | False | 7 | 64 | 0.201 | -0.0341 | 0.421 |
| M2c | M1 + sprint time constant | False | 7 | 64 | 0.191 | -0.0368 | 0.408 |
| M2bc | M1 + both acceleration features | True | 8 | 64 | 0.208 | -0.0171 | 0.427 |

**spearman**

| model | label | exploratory | n_predictors | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| MEAN | training mean | False | 0 | 64 | -0.165 |  |  |
| M0 | cohort + body | False | 4 | 64 | 0.125 |  |  |
| M1 | M0 + official forty and split | False | 6 | 64 | 0.194 |  |  |
| M2a | M1 + tracked speed | False | 7 | 64 | 0.188 |  |  |
| M2b | M1 + accel 4–7 yd/s | False | 7 | 64 | 0.215 |  |  |
| M2c | M1 + sprint time constant | False | 7 | 64 | 0.209 |  |  |
| M2bc | M1 + both acceleration features | True | 8 | 64 | 0.222 |  |  |

**calibration_slope**

| model | label | exploratory | n_predictors | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| MEAN | training mean | False | 0 | 64 | -1.86 | -3.04 | -0.842 |
| M0 | cohort + body | False | 4 | 64 | 0.378 | -0.588 | 1.44 |
| M1 | M0 + official forty and split | False | 6 | 64 | 0.693 | -0.242 | 1.65 |
| M2a | M1 + tracked speed | False | 7 | 64 | 0.652 | -0.297 | 1.64 |
| M2b | M1 + accel 4–7 yd/s | False | 7 | 64 | 0.778 | -0.123 | 1.67 |
| M2c | M1 + sprint time constant | False | 7 | 64 | 0.764 | -0.128 | 1.63 |
| M2bc | M1 + both acceleration features | True | 8 | 64 | 0.798 | -0.0555 | 1.65 |

**calibration_intercept**

| model | label | exploratory | n_predictors | n_athletes | estimate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| MEAN | training mean | False | 0 | 64 | 2.95 | 1.9 | 4.16 |
| M0 | cohort + body | False | 4 | 64 | 0.643 | -0.46 | 1.64 |
| M1 | M0 + official forty and split | False | 6 | 64 | 0.318 | -0.677 | 1.28 |
| M2a | M1 + tracked speed | False | 7 | 64 | 0.36 | -0.661 | 1.34 |
| M2b | M1 + accel 4–7 yd/s | False | 7 | 64 | 0.23 | -0.698 | 1.16 |
| M2c | M1 + sprint time constant | False | 7 | 64 | 0.244 | -0.648 | 1.17 |
| M2bc | M1 + both acceleration features | True | 8 | 64 | 0.21 | -0.674 | 1.1 |


Figures: `CN_05_forty_vs_getoff` (in-sample description), `CN_06_model_performance`.

## 8. Statistical uncertainty

Difference = model − M1. For MAE and RMSE a negative value means the model predicts held-out athletes better; for R² and Pearson a positive value does.

| metric | model | reference | exploratory | n_athletes | estimate | ci_lower | ci_upper | share_boot_improved |
|---|---|---|---|---|---|---|---|---|
| mae | M0 | M1 | False | 64 | 0.00208 | 0.000454 | 0.00369 | 0.0075 |
| mae | M2a | M1 | False | 64 | 0.000153 | -0.000153 | 0.00046 | 0.163 |
| mae | M2b | M1 | False | 64 | -0.000632 | -0.00151 | 0.00036 | 0.925 |
| mae | M2bc | M1 | True | 64 | -0.000522 | -0.00213 | 0.00109 | 0.757 |
| mae | M2c | M1 | False | 64 | -0.000233 | -0.00158 | 0.00114 | 0.634 |
| mae | MEAN | M1 | False | 64 | 0.00407 | -7.34e-05 | 0.00826 | 0.029 |
| pearson | M0 | M1 | False | 64 | -0.0883 | -0.168 | -0.00862 | 0.0145 |
| pearson | M2a | M1 | False | 64 | -0.00974 | -0.0252 | 0.00522 | 0.096 |
| pearson | M2b | M1 | False | 64 | 0.0365 | -0.0133 | 0.0777 | 0.932 |
| pearson | M2bc | M1 | True | 64 | 0.044 | -0.043 | 0.121 | 0.854 |
| pearson | M2c | M1 | False | 64 | 0.0268 | -0.0446 | 0.0948 | 0.772 |
| pearson | MEAN | M1 | False | 64 | -0.338 | -0.566 | -0.112 | 0.0015 |
| r2_oos | M0 | M1 | False | 64 | -0.0312 | -0.0744 | 0.00945 | 0.0675 |
| r2_oos | M2a | M1 | False | 64 | -0.00483 | -0.0118 | 0.00213 | 0.0765 |
| r2_oos | M2b | M1 | False | 64 | 0.0147 | -0.0143 | 0.0358 | 0.885 |
| r2_oos | M2bc | M1 | True | 64 | 0.0176 | -0.0299 | 0.055 | 0.805 |
| r2_oos | M2c | M1 | False | 64 | 0.00941 | -0.0284 | 0.0419 | 0.713 |
| r2_oos | MEAN | M1 | False | 64 | -0.0613 | -0.161 | 0.0438 | 0.121 |
| rmse | M0 | M1 | False | 64 | 0.00132 | -0.000395 | 0.00312 | 0.068 |
| rmse | M2a | M1 | False | 64 | 0.000206 | -9.29e-05 | 0.000497 | 0.0765 |
| rmse | M2b | M1 | False | 64 | -0.000628 | -0.00159 | 0.00055 | 0.884 |
| rmse | M2bc | M1 | True | 64 | -0.000753 | -0.00243 | 0.00124 | 0.805 |
| rmse | M2c | M1 | False | 64 | -0.000401 | -0.00182 | 0.00116 | 0.712 |
| rmse | MEAN | M1 | False | 64 | 0.00258 | -0.00182 | 0.00687 | 0.119 |


**Standardised coefficients on all athletes (in-sample, descriptive)**

| model | term | estimate_std | ci_lower | ci_upper | ridge_lambda | n_athletes |
|---|---|---|---|---|---|---|
| M0 | cmb_combine_weight | 0.0125 | 0.00332 | 0.0203 | 56.2 | 64 |
| M0 | cmb_combine_height | 0.00483 | -0.00552 | 0.0162 | 56.2 | 64 |
| M0 | draft_year=2024 | -0.00258 | -0.0104 | 0.00529 | 56.2 | 64 |
| M0 | draft_year=2025 | 0.00214 | -0.00522 | 0.0104 | 56.2 | 64 |
| M1 | cmb_combine_weight | 0.00497 | 0.000792 | 0.00967 | 100 | 64 |
| M1 | cmb_combine_height | 0.00316 | -0.00391 | 0.0105 | 100 | 64 |
| M1 | cmb_forty | 0.0066 | 0.00228 | 0.0106 | 100 | 64 |
| M1 | cmb_ten_yd_split | 0.00632 | 0.00145 | 0.0101 | 100 | 64 |
| M1 | draft_year=2024 | -0.00148 | -0.00563 | 0.00331 | 100 | 64 |
| M1 | draft_year=2025 | 0.00124 | -0.00258 | 0.00552 | 100 | 64 |
| M2a | cmb_combine_weight | 0.00432 | -0.000367 | 0.00856 | 100 | 64 |
| M2a | cmb_combine_height | 0.00317 | -0.00447 | 0.0107 | 100 | 64 |
| M2a | cmb_forty | 0.00566 | 0.0019 | 0.00915 | 100 | 64 |
| M2a | cmb_ten_yd_split | 0.00559 | 0.000678 | 0.00959 | 100 | 64 |
| M2a | p95_speed | -0.00373 | -0.00798 | 0.0012 | 100 | 64 |
| M2a | draft_year=2024 | -0.00153 | -0.00645 | 0.00355 | 100 | 64 |
| M2a | draft_year=2025 | 0.00119 | -0.00319 | 0.00638 | 100 | 64 |
| M2b | cmb_combine_weight | 0.00375 | -0.000187 | 0.00846 | 100 | 64 |
| M2b | cmb_combine_height | 0.00288 | -0.00452 | 0.01 | 100 | 64 |
| M2b | cmb_forty | 0.00508 | 0.00123 | 0.00888 | 100 | 64 |
| M2b | cmb_ten_yd_split | 0.00492 | 0.000492 | 0.00878 | 100 | 64 |
| M2b | accel_v4_7 | -0.00743 | -0.0132 | -0.00188 | 100 | 64 |
| M2b | draft_year=2024 | -0.001 | -0.00509 | 0.00359 | 100 | 64 |
| M2b | draft_year=2025 | 0.000711 | -0.00341 | 0.00549 | 100 | 64 |
| M2c | cmb_combine_weight | 0.00492 | -0.000382 | 0.00939 | 100 | 64 |
| M2c | cmb_combine_height | 0.00271 | -0.00532 | 0.00978 | 100 | 64 |
| M2c | cmb_forty | 0.00694 | 0.00215 | 0.0107 | 100 | 64 |
| M2c | cmb_ten_yd_split | 0.0064 | 0.00145 | 0.0104 | 100 | 64 |
| M2c | model_tau_s | 0.0057 | -0.00136 | 0.0133 | 100 | 64 |
| M2c | draft_year=2024 | -0.000952 | -0.00533 | 0.00359 | 100 | 64 |
| M2c | draft_year=2025 | 0.000541 | -0.00336 | 0.00482 | 100 | 64 |
| M2bc | cmb_combine_weight | 0.00383 | -0.000427 | 0.00821 | 100 | 64 |
| M2bc | cmb_combine_height | 0.00252 | -0.00538 | 0.0103 | 100 | 64 |
| M2bc | cmb_forty | 0.00552 | 0.00175 | 0.00885 | 100 | 64 |
| M2bc | cmb_ten_yd_split | 0.00513 | 0.001 | 0.00871 | 100 | 64 |
| M2bc | accel_v4_7 | -0.00667 | -0.0112 | -0.00178 | 100 | 64 |
| M2bc | model_tau_s | 0.00482 | -0.00199 | 0.0117 | 100 | 64 |
| M2bc | draft_year=2024 | -0.000604 | -0.00495 | 0.00411 | 100 | 64 |
| M2bc | draft_year=2025 | 0.000176 | -0.0037 | 0.00453 | 100 | 64 |


Figure: `CN_07_incremental_gain`.

## 9. Sensitivity analyses

Each analysis changes one condition and refits every model on its own athletes; `scored_on:*` rows score the primary fit on a subset without refitting. Subgroups are small: their estimates describe stability, they are not separate findings.

**What each analysis changes**

| analysis | description |
|---|---|
| primary | pre-specified primary analysis |
| scored_on:alignment=EDGE | primary fit scored on this subset of athletes (no refit) |
| scored_on:alignment=INTERIOR_LINE | primary fit scored on this subset of athletes (no refit) |
| scored_on:cohort=2023 | primary fit scored on this subset of athletes (no refit) |
| scored_on:cohort=2024 | primary fit scored on this subset of athletes (no refit) |
| scored_on:cohort=2025 | primary fit scored on this subset of athletes (no refit) |
| alignment:EDGE | athletes whose modal NFL alignment is EDGE |
| alignment:INTERIOR_LINE | athletes whose modal NFL alignment is INTERIOR_LINE |
| leave_one_cohort_out | train on two draft cohorts, predict the third |
| exclude_cohort:2024 | 2023 and 2025 cohorts (2024 Combine tracking is noisier) |
| first_20 | first 20 eligible get-off observations of each athlete |
| first_50 | first 50 eligible get-off observations of each athlete |
| first_100 | first 100 eligible get-off observations of each athlete |
| min_obs:20 | athletes with at least 20 observations |
| min_obs:100 | athletes with at least 100 observations |
| min_obs:200 | athletes with at least 200 observations |
| target:mean | NFL endpoint = athlete mean |
| target:median | NFL endpoint = athlete median |
| target:p25 | NFL endpoint = athlete p25 |
| target:p75 | NFL endpoint = athlete p75 |
| no_boundary_values | endpoint observations at the maximum or below the low bound removed |
| combine:first\|S0_all\|all | Combine tracking features = first\|S0_all\|all (aggregation\|quality scenario\|fit filter) |
| combine:fastest\|S0_all\|all | Combine tracking features = fastest\|S0_all\|all (aggregation\|quality scenario\|fit filter) |
| combine:mean\|S1_structural\|all | Combine tracking features = mean\|S1_structural\|all (aggregation\|quality scenario\|fit filter) |
| combine:mean\|S2_signal\|all | Combine tracking features = mean\|S2_signal\|all (aggregation\|quality scenario\|fit filter) |
| combine:mean\|S3_no_boundary\|all | Combine tracking features = mean\|S3_no_boundary\|all (aggregation\|quality scenario\|fit filter) |
| combine:mean\|S0_all\|fit_ok | Combine tracking features = mean\|S0_all\|fit_ok (aggregation\|quality scenario\|fit filter) |
| own_sample:M2a | largest sample available to M2a (M1 refitted on the same athletes) |
| own_sample:M2b | largest sample available to M2b (M1 refitted on the same athletes) |
| own_sample:M2c | largest sample available to M2c (M1 refitted on the same athletes) |
| robust_huber | Huber loss (ridge penalty kept) |
| traditional_plus_jumps | does adding vertical and broad jump to M1 change the baseline? |

**Tracking-augmented models against M1 (difference in RMSE, seconds; below 0 = better)**

| analysis | model | n_athletes | rmse | r2_oos | d_rmse_vs_m1 | d_rmse_lower | d_rmse_upper | share_boot_improved_rmse |
|---|---|---|---|---|---|---|---|---|
| primary | M2a | 64 | 0.0803 | 0.0564 | 0.000206 | -9.63e-05 | 0.000485 | 0.08 |
| primary | M2b | 64 | 0.0794 | 0.076 | -0.000628 | -0.0016 | 0.000624 | 0.886 |
| primary | M2c | 64 | 0.0797 | 0.0707 | -0.000401 | -0.00179 | 0.00126 | 0.722 |
| scored_on:alignment=EDGE | M2a | 29 | 0.0876 | 0.028 | 0.000377 | -0.000109 | 0.000834 | 0.05 |
| scored_on:alignment=EDGE | M2b | 29 | 0.0871 | 0.0404 | -0.000174 | -0.00164 | 0.00186 | 0.622 |
| scored_on:alignment=EDGE | M2c | 29 | 0.0868 | 0.0455 | -0.000404 | -0.00261 | 0.00228 | 0.618 |
| scored_on:alignment=INTERIOR_LINE | M2a | 35 | 0.0736 | 0.0884 | 3.75e-05 | -0.000375 | 0.000336 | 0.428 |
| scored_on:alignment=INTERIOR_LINE | M2b | 35 | 0.0724 | 0.116 | -0.00109 | -0.00219 | -1.24e-05 | 0.978 |
| scored_on:alignment=INTERIOR_LINE | M2c | 35 | 0.0731 | 0.0991 | -0.000401 | -0.00175 | 0.00105 | 0.67 |
| scored_on:cohort=2023 | M2a | 25 | 0.0763 | 0.0852 | -0.000123 | -0.000582 | 0.000335 | 0.714 |
| scored_on:cohort=2023 | M2b | 25 | 0.0746 | 0.126 | -0.00184 | -0.00303 | -0.000454 | 0.998 |
| scored_on:cohort=2023 | M2c | 25 | 0.0758 | 0.0981 | -0.000648 | -0.00241 | 0.00155 | 0.722 |
| scored_on:cohort=2024 | M2a | 23 | 0.0786 | 0.0706 | 0.000436 | -0.000137 | 0.000891 | 0.054 |
| scored_on:cohort=2024 | M2b | 23 | 0.0789 | 0.0636 | 0.000731 | -0.00116 | 0.00368 | 0.312 |
| scored_on:cohort=2024 | M2c | 23 | 0.0791 | 0.0596 | 0.000904 | -0.00158 | 0.00419 | 0.28 |
| scored_on:cohort=2025 | M2a | 16 | 0.0882 | 0.0017 | 0.000354 | -0.000109 | 0.000882 | 0.066 |
| scored_on:cohort=2025 | M2b | 16 | 0.0871 | 0.0268 | -0.000762 | -0.00195 | 0.000713 | 0.852 |
| scored_on:cohort=2025 | M2c | 16 | 0.0861 | 0.0491 | -0.00177 | -0.00427 | 0.00103 | 0.91 |
| alignment:INTERIOR_LINE | M2a | 35 | 0.0707 | 0.207 | -0.000678 | -0.00194 | 0.00046 | 0.874 |
| alignment:INTERIOR_LINE | M2b | 35 | 0.0697 | 0.229 | -0.00171 | -0.00383 | 0.000607 | 0.912 |
| alignment:INTERIOR_LINE | M2c | 35 | 0.0715 | 0.189 | 0.000119 | -0.00337 | 0.00363 | 0.544 |
| leave_one_cohort_out | M2a | 64 | 0.0787 | 0.14 | 6.71e-05 | -0.000426 | 0.000512 | 0.39 |
| leave_one_cohort_out | M2b | 64 | 0.0782 | 0.149 | -0.000344 | -0.00248 | 0.00216 | 0.664 |
| leave_one_cohort_out | M2c | 64 | 0.0778 | 0.158 | -0.00076 | -0.00266 | 0.000993 | 0.812 |
| exclude_cohort:2024 | M2a | 41 | 0.0844 | 0.0216 | -0.000374 | -0.000714 | 4.85e-06 | 0.972 |
| exclude_cohort:2024 | M2b | 41 | 0.0863 | -0.0223 | 0.00146 | -0.00318 | 0.0063 | 0.254 |
| exclude_cohort:2024 | M2c | 41 | 0.0883 | -0.0704 | 0.00347 | -0.00266 | 0.0117 | 0.214 |
| first_20 | M2a | 74 | 0.118 | -0.0136 | 0.0003 | 0.000107 | 0.000544 | 0 |
| first_20 | M2b | 74 | 0.118 | -0.00979 | 6.98e-05 | -0.000105 | 0.000239 | 0.234 |
| first_20 | M2c | 74 | 0.118 | -0.00963 | 6.32e-05 | -0.000335 | 0.000424 | 0.358 |
| first_50 | M2a | 64 | 0.0932 | 0.045 | 8.45e-06 | -0.000296 | 0.000353 | 0.456 |
| first_50 | M2b | 64 | 0.0934 | 0.0401 | 0.000205 | -0.00115 | 0.0017 | 0.388 |
| first_50 | M2c | 64 | 0.0939 | 0.0306 | 0.000676 | -0.000432 | 0.00197 | 0.15 |
| first_100 | M2a | 53 | 0.0808 | 0.0544 | 0.0013 | 0.000124 | 0.00263 | 0.02 |
| first_100 | M2b | 53 | 0.0797 | 0.0803 | 0.000191 | -0.00133 | 0.00218 | 0.47 |
| first_100 | M2c | 53 | 0.0805 | 0.0623 | 0.000981 | 3.44e-05 | 0.0023 | 0.016 |
| min_obs:20 | M2a | 74 | 0.0877 | 0.102 | 0.000679 | 0.000353 | 0.00105 | 0 |
| min_obs:20 | M2b | 74 | 0.0868 | 0.121 | -0.000247 | -0.00121 | 0.000786 | 0.674 |
| min_obs:20 | M2c | 74 | 0.0867 | 0.123 | -0.000345 | -0.00198 | 0.00148 | 0.7 |
| min_obs:100 | M2a | 53 | 0.0773 | 0.0231 | 0.00203 | 0.000192 | 0.0042 | 0.006 |
| min_obs:100 | M2b | 53 | 0.0752 | 0.0754 | -4.8e-05 | -0.00121 | 0.00164 | 0.51 |
| min_obs:100 | M2c | 53 | 0.0767 | 0.037 | 0.00149 | 0.000159 | 0.00301 | 0.012 |
| min_obs:200 | M2a | 37 | 0.0713 | -0.0407 | 0.00232 | 0.00107 | 0.00402 | 0 |
| min_obs:200 | M2b | 37 | 0.0693 | 0.0209 | 0.000314 | -0.000318 | 0.00119 | 0.244 |
| min_obs:200 | M2c | 37 | 0.0707 | -0.0205 | 0.00173 | 0.000609 | 0.00289 | 0.002 |
| target:mean | M2a | 64 | 0.0773 | 0.434 | 0.000241 | -0.000467 | 0.000955 | 0.288 |
| target:mean | M2b | 64 | 0.0759 | 0.454 | -0.0011 | -0.00273 | 0.000795 | 0.874 |
| target:mean | M2c | 64 | 0.0768 | 0.441 | -0.00023 | -0.00192 | 0.00187 | 0.596 |
| target:median | M2a | 64 | 0.0807 | 0.389 | -3.66e-05 | -0.000834 | 0.000712 | 0.592 |
| target:median | M2b | 64 | 0.0802 | 0.397 | -0.000513 | -0.00204 | 0.00129 | 0.73 |
| target:median | M2c | 64 | 0.0806 | 0.39 | -6.8e-05 | -0.00138 | 0.00143 | 0.498 |
| target:p25 | M2a | 64 | 0.0673 | 0.498 | 0.000665 | -0.000124 | 0.00135 | 0.046 |
| target:p25 | M2b | 64 | 0.0671 | 0.501 | 0.000424 | -0.00124 | 0.00204 | 0.344 |
| target:p25 | M2c | 64 | 0.0668 | 0.506 | 8.28e-05 | -0.00189 | 0.002 | 0.514 |
| target:p75 | M2a | 64 | 0.096 | 0.331 | 0.000253 | -0.000618 | 0.0011 | 0.266 |
| target:p75 | M2b | 64 | 0.0948 | 0.348 | -0.000997 | -0.00295 | 0.00113 | 0.828 |
| target:p75 | M2c | 64 | 0.0956 | 0.336 | -0.000125 | -0.00237 | 0.00227 | 0.55 |
| no_boundary_values | M2a | 64 | 0.0796 | 0.0742 | 9.09e-06 | -0.000333 | 0.000322 | 0.45 |
| no_boundary_values | M2b | 64 | 0.079 | 0.0893 | -0.000644 | -0.00156 | 0.000525 | 0.87 |
| no_boundary_values | M2c | 64 | 0.0795 | 0.0774 | -0.000133 | -0.00173 | 0.00175 | 0.566 |
| combine:first\|S0_all\|all | M2a | 63 | 0.0808 | 0.0632 | 0.000178 | -0.000123 | 0.000466 | 0.13 |
| combine:first\|S0_all\|all | M2b | 63 | 0.0795 | 0.0917 | -0.00106 | -0.00273 | 0.000482 | 0.904 |
| combine:first\|S0_all\|all | M2c | 63 | 0.0803 | 0.0753 | -0.000349 | -0.00197 | 0.00122 | 0.65 |
| combine:fastest\|S0_all\|all | M2a | 63 | 0.0811 | 0.0532 | 3.23e-05 | -0.000259 | 0.000328 | 0.416 |
| combine:fastest\|S0_all\|all | M2b | 63 | 0.0803 | 0.0704 | -0.000729 | -0.00244 | 0.000956 | 0.82 |
| combine:fastest\|S0_all\|all | M2c | 63 | 0.0803 | 0.0706 | -0.000731 | -0.00252 | 0.00107 | 0.758 |
| combine:mean\|S1_structural\|all | M2a | 64 | 0.0799 | 0.0701 | 1.8e-05 | -0.000286 | 0.000341 | 0.436 |
| combine:mean\|S1_structural\|all | M2b | 64 | 0.0793 | 0.0846 | -0.000609 | -0.00175 | 0.0008 | 0.822 |
| combine:mean\|S1_structural\|all | M2c | 64 | 0.0797 | 0.0744 | -0.000167 | -0.00178 | 0.00174 | 0.572 |
| combine:mean\|S2_signal\|all | M2a | 64 | 0.0805 | 0.0583 | 3.62e-05 | -0.000269 | 0.000353 | 0.414 |
| combine:mean\|S2_signal\|all | M2b | 64 | 0.0797 | 0.0754 | -0.000698 | -0.00176 | 0.00057 | 0.888 |
| combine:mean\|S2_signal\|all | M2c | 64 | 0.08 | 0.0702 | -0.000482 | -0.00209 | 0.00127 | 0.7 |
| combine:mean\|S3_no_boundary\|all | M2a | 63 | 0.0803 | 0.0596 | 0.000601 | 0.000248 | 0.00109 | 0 |
| combine:mean\|S3_no_boundary\|all | M2b | 63 | 0.0788 | 0.093 | -0.000827 | -0.002 | 0.000523 | 0.902 |
| combine:mean\|S3_no_boundary\|all | M2c | 63 | 0.0789 | 0.0924 | -0.000804 | -0.00287 | 0.00155 | 0.788 |
| combine:mean\|S0_all\|fit_ok | M2a | 64 | 0.0798 | 0.0626 | -7.27e-06 | -0.00032 | 0.000308 | 0.494 |
| combine:mean\|S0_all\|fit_ok | M2b | 64 | 0.0793 | 0.0725 | -0.000439 | -0.00174 | 0.0014 | 0.672 |
| combine:mean\|S0_all\|fit_ok | M2c | 64 | 0.0793 | 0.0727 | -0.000435 | -0.00187 | 0.00155 | 0.66 |
| own_sample:M2a | M2a | 64 | 0.0801 | 0.0551 | 0.00018 | -0.000146 | 0.000483 | 0.134 |
| own_sample:M2b | M2b | 64 | 0.08 | 0.0697 | -0.000517 | -0.00168 | 0.000954 | 0.796 |
| own_sample:M2c | M2c | 64 | 0.0798 | 0.0645 | -0.00017 | -0.0015 | 0.00124 | 0.596 |
| robust_huber | M2a | 64 | 0.0803 | 0.0647 | 4.83e-05 | -0.000301 | 0.000344 | 0.394 |
| robust_huber | M2b | 64 | 0.0795 | 0.0821 | -0.000706 | -0.00178 | 0.000666 | 0.898 |
| robust_huber | M2c | 64 | 0.08 | 0.0706 | -0.000207 | -0.00146 | 0.00131 | 0.638 |

**Baselines under each analysis**

| analysis | model | n_athletes | rmse | r2_oos | d_rmse_vs_m1 | d_rmse_lower | d_rmse_upper |
|---|---|---|---|---|---|---|---|
| primary | M0 | 64 | 0.0814 | 0.0301 | 0.00132 | -0.000337 | 0.00309 |
| primary | M1 | 64 | 0.0801 | 0.0613 |  |  |  |
| scored_on:alignment=EDGE | M0 | 29 | 0.0866 | 0.0507 | -0.000651 | -0.00269 | 0.00189 |
| scored_on:alignment=EDGE | M1 | 29 | 0.0872 | 0.0364 |  |  |  |
| scored_on:alignment=INTERIOR_LINE | M0 | 35 | 0.0767 | 0.0081 | 0.00321 | 0.00125 | 0.00533 |
| scored_on:alignment=INTERIOR_LINE | M1 | 35 | 0.0735 | 0.0893 |  |  |  |
| scored_on:cohort=2023 | M0 | 25 | 0.0778 | 0.0483 | 0.00141 | -0.00201 | 0.00491 |
| scored_on:cohort=2023 | M1 | 25 | 0.0764 | 0.0824 |  |  |  |
| scored_on:cohort=2024 | M0 | 23 | 0.0795 | 0.0498 | 0.0013 | -0.00151 | 0.00423 |
| scored_on:cohort=2024 | M1 | 23 | 0.0782 | 0.0809 |  |  |  |
| scored_on:cohort=2025 | M0 | 16 | 0.089 | -0.0182 | 0.00121 | -0.00123 | 0.00332 |
| scored_on:cohort=2025 | M1 | 16 | 0.0878 | 0.00967 |  |  |  |
| alignment:INTERIOR_LINE | M0 | 35 | 0.0763 | 0.0783 | 0.00489 | -0.00273 | 0.0124 |
| alignment:INTERIOR_LINE | M1 | 35 | 0.0714 | 0.191 |  |  |  |
| leave_one_cohort_out | M0 | 64 | 0.0805 | 0.0986 | 0.00192 | 9.88e-05 | 0.0036 |
| leave_one_cohort_out | M1 | 64 | 0.0786 | 0.141 |  |  |  |
| exclude_cohort:2024 | M0 | 41 | 0.0861 | -0.017 | 0.00128 | -0.000557 | 0.00311 |
| exclude_cohort:2024 | M1 | 41 | 0.0848 | 0.013 |  |  |  |
| first_20 | M0 | 74 | 0.118 | -0.00655 | -0.000112 | -0.000587 | 0.000395 |
| first_20 | M1 | 74 | 0.118 | -0.00852 |  |  |  |
| first_50 | M0 | 64 | 0.0944 | 0.0204 | 0.0012 | 1.56e-06 | 0.00247 |
| first_50 | M1 | 64 | 0.0932 | 0.0452 |  |  |  |
| first_100 | M0 | 53 | 0.0817 | 0.0335 | 0.00221 | -0.000249 | 0.00493 |
| first_100 | M1 | 53 | 0.0795 | 0.0852 |  |  |  |
| min_obs:20 | M0 | 74 | 0.0874 | 0.11 | 0.000315 | -0.00155 | 0.00213 |
| min_obs:20 | M1 | 74 | 0.087 | 0.116 |  |  |  |
| min_obs:100 | M0 | 53 | 0.0783 | -0.0017 | 0.00302 | -0.000357 | 0.0071 |
| min_obs:100 | M1 | 53 | 0.0752 | 0.0742 |  |  |  |
| min_obs:200 | M0 | 37 | 0.0712 | -0.0354 | 0.00227 | -0.000426 | 0.00466 |
| min_obs:200 | M1 | 37 | 0.0689 | 0.0298 |  |  |  |
| target:mean | M0 | 64 | 0.08 | 0.394 | 0.00295 | -0.00154 | 0.00755 |
| target:mean | M1 | 64 | 0.077 | 0.438 |  |  |  |
| target:median | M0 | 64 | 0.0831 | 0.351 | 0.00244 | -0.00181 | 0.00682 |
| target:median | M1 | 64 | 0.0807 | 0.389 |  |  |  |
| target:p25 | M0 | 64 | 0.0699 | 0.459 | 0.00321 | -0.00176 | 0.00786 |
| target:p25 | M1 | 64 | 0.0667 | 0.508 |  |  |  |
| target:p75 | M0 | 64 | 0.0974 | 0.312 | 0.0016 | -0.00261 | 0.00547 |
| target:p75 | M1 | 64 | 0.0958 | 0.334 |  |  |  |
| no_boundary_values | M0 | 64 | 0.0812 | 0.0378 | 0.00155 | -0.000183 | 0.00337 |
| no_boundary_values | M1 | 64 | 0.0796 | 0.0744 |  |  |  |
| combine:first\|S0_all\|all | M0 | 63 | 0.0821 | 0.0326 | 0.00149 | -0.000374 | 0.00347 |
| combine:first\|S0_all\|all | M1 | 63 | 0.0806 | 0.0673 |  |  |  |
| combine:fastest\|S0_all\|all | M0 | 63 | 0.0825 | 0.0203 | 0.00143 | -0.000423 | 0.00337 |
| combine:fastest\|S0_all\|all | M1 | 63 | 0.081 | 0.0539 |  |  |  |
| combine:mean\|S1_structural\|all | M0 | 64 | 0.0812 | 0.0397 | 0.00131 | -0.000354 | 0.00296 |
| combine:mean\|S1_structural\|all | M1 | 64 | 0.0799 | 0.0705 |  |  |  |
| combine:mean\|S2_signal\|all | M0 | 64 | 0.0816 | 0.0313 | 0.00119 | -0.000615 | 0.00298 |
| combine:mean\|S2_signal\|all | M1 | 64 | 0.0804 | 0.0592 |  |  |  |
| combine:mean\|S3_no_boundary\|all | M0 | 63 | 0.0812 | 0.0369 | 0.00158 | -9.56e-05 | 0.00351 |
| combine:mean\|S3_no_boundary\|all | M1 | 63 | 0.0797 | 0.0739 |  |  |  |
| combine:mean\|S0_all\|fit_ok | M0 | 64 | 0.0814 | 0.0239 | 0.00162 | -7.39e-05 | 0.0032 |
| combine:mean\|S0_all\|fit_ok | M1 | 64 | 0.0798 | 0.0624 |  |  |  |
| own_sample:M2a | M0 | 64 | 0.0813 | 0.027 | 0.00137 | -0.000347 | 0.00311 |
| own_sample:M2a | M1 | 64 | 0.0799 | 0.0594 |  |  |  |
| own_sample:M2b | M0 | 64 | 0.0816 | 0.0318 | 0.0011 | -0.000587 | 0.0027 |
| own_sample:M2b | M1 | 64 | 0.0805 | 0.0578 |  |  |  |
| own_sample:M2c | M0 | 64 | 0.0814 | 0.0266 | 0.00143 | -0.000231 | 0.00323 |
| own_sample:M2c | M1 | 64 | 0.08 | 0.0605 |  |  |  |
| robust_huber | M0 | 64 | 0.0817 | 0.031 | 0.00148 | -0.000269 | 0.00352 |
| robust_huber | M1 | 64 | 0.0802 | 0.0659 |  |  |  |
| traditional_plus_jumps | M0 | 65 | 0.0808 | 0.0322 | 0.00166 | -0.000111 | 0.00351 |
| traditional_plus_jumps | M1 | 65 | 0.0791 | 0.0717 |  |  |  |
| traditional_plus_jumps | M1j | 65 | 0.0785 | 0.086 | -0.00062 | -0.00239 | 0.00126 |

**Not estimable**

| analysis | description | n_athletes | note |
|---|---|---|---|
| alignment:EDGE | athletes whose modal NFL alignment is EDGE | 29 | not estimable: 29 athletes (minimum 30) or a single cohort |


### Matched athletes

Pairs of athletes of the same alignment group with near-identical official forty and 10-yard split, ordered by how much their tracked acceleration differs. An in-sample illustration of the question, not a held-out result.

**All pairs: difference in the NFL target against difference in the tracked feature**

| n_pairs | spearman | ci_lower | ci_upper | mean_d_target | share_d_target_negative |
|---|---|---|---|---|---|
| 18 | -0.0464 | -0.502 | 0.43 | -0.0353 | 0.667 |

**The 6 pairs with the largest tracked difference**

| pair_id | alignment_group | name_a | name_b | forty_a | forty_b | feature_a | feature_b | target_a | target_b | d_target |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | EDGE | Eric Watts | Jordan Burch | 4.67 | 4.67 | 5.43 | 4.08 | 1.14 | 1.11 | 0.0267 |
| 2 | EDGE | Cedric Johnson | Tyler Baron | 4.63 | 4.62 | 5.35 | 4.19 | 0.985 | 1.06 | -0.0752 |
| 3 | INTERIOR_LINE | Justin Eboigbe | Nazir Stackhouse | 5.18 | 5.15 | 4.14 | 3.29 | 1.06 | 1.08 | -0.0194 |
| 4 | INTERIOR_LINE | Darius Robinson | Darius Alexander | 4.95 | 4.95 | 4.11 | 3.51 | 0.996 | 1.04 | -0.043 |
| 5 | EDGE | Laiatu Latu | Tavius Robinson | 4.64 | 4.66 | 5.24 | 4.67 | 0.985 | 1.13 | -0.147 |
| 6 | INTERIOR_LINE | Leonard Taylor | Eric Gregory | 5.12 | 5.14 | 3.68 | 3.19 | 1.06 | 1.21 | -0.148 |


Figure: `CN_08_matched_athletes`.

## 10. Football interpretation

**What each Combine predictor represents (stated before any result)**

- `cmb_forty` — official forty time: acceleration and top speed over 40 yd, one number.
- `cmb_ten_yd_split` — official 10-yard split: the start and the first steps.
- `p95_speed` — tracked top-end speed sustained in the run.
- `accel_v4_7` — tracked mean acceleration while speed rises from 4 to 7 yd/s (mid-acceleration phase).
- `model_tau_s` — time constant of the fitted sprint curve: how quickly the athlete approaches top speed (lower = quicker).

> **BLOCKER — the vendor definition of `player_get_off` is unresolved (status = 'unresolved'; missing: source, timing_origin, timing_end, lower_is_better). Fill `configs/player_get_off_definition.json` from the competition data dictionary. Until then the endpoint is reported as an opaque vendor number: no direction ('faster', 'better'), no physical meaning and no coaching conclusion may be attached to any estimate.**

What can be said without the definition: whether a model predicts the delivered number for held-out athletes, by how many milliseconds, and how stable that is. What cannot be said: that a lower or higher value is a quicker first step, that the endpoint measures acceleration, or that a coach should act on it. The direction evidence of §3.5 is co-variation with sacks and pressures; it is not a definition.


**Reading of the results**

_Rule-based reading of the tables above. It needs review and is not a conclusion._

- **Official tests (M1):** out-of-fold R² = 0.061 [-0.044, 0.161]; its interval reaches 0 or below, so the official tests are not shown to predict the endpoint out of sample.
- **Practical scale:** predicting every athlete at the training mean has an RMSE of 82.6 ms; M1 has 80.1 ms.
- **M2a, M1 + tracked speed:** RMSE − RMSE(M1) = +0.2 ms [-0.1 ms, +0.5 ms]; ΔR² = -0.005; improved in 8 % of bootstrap replicates; the interval includes 0: no demonstrated gain over M1.
- **M2b, M1 + accel 4–7 yd/s:** RMSE − RMSE(M1) = -0.6 ms [-1.6 ms, +0.6 ms]; ΔR² = +0.015; improved in 88 % of bootstrap replicates; the interval includes 0: no demonstrated gain over M1.
- **M2c, M1 + sprint time constant:** RMSE − RMSE(M1) = -0.4 ms [-1.8 ms, +1.2 ms]; ΔR² = +0.009; improved in 71 % of bootstrap replicates; the interval includes 0: no demonstrated gain over M1.
- **M2bc (EXPLORATORY), M1 + both acceleration features:** RMSE − RMSE(M1) = -0.8 ms [-2.4 ms, +1.2 ms]; ΔR² = +0.018; improved in 80 % of bootstrap replicates; the interval includes 0: no demonstrated gain over M1.
- **M2a across sensitivities:** RMSE lower than M1 in 5 of 27 analyses; interval wholly below 0 in 0.
- **M2b across sensitivities:** RMSE lower than M1 in 20 of 27 analyses; interval wholly below 0 in 2.
- **M2c across sensitivities:** RMSE lower than M1 in 18 of 27 analyses; interval wholly below 0 in 0.


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

_Rule-based proposal. It needs review._

1. **Resolve the endpoint definition** (`configs/player_get_off_definition.json`). Nothing below is interpretable in football terms before that.
- No pre-specified tracking model shows a demonstrated gain over the official tests. With this sample the supported statement is 'not shown', not 'no effect': the interval in §8 states which gains remain compatible with the data.
- The official tests themselves are not shown to predict this endpoint. A tracking gain over a baseline without skill is weak evidence; consider whether get-off is the right first endpoint.

