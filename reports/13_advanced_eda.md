# 13 — Advanced EDA: Combine movement (Phase 03 · M2)

_Status: **code authored, NOT executed — user validation required.** No M2 result is reported here until the user has run the commands below. M1 facts quoted in this document come from executed M1 outputs (report 14 §6)._

## 1. Questions
1. Is each tracked Combine recording the timed drill? Check whether the forty is ≈ 40 yd long, whether the start is recorded, and whether the recording ends at the finish.
2. Which tracked timing conventions agree with official `forty`, `ten_yd_split`, `short_shuttle` and `three_cone`? A tracked time may be interpreted as a split estimate **only if** agreement is high.
3. Within similar official forty times, how different are acceleration profiles (τ, a0, flying intervals)?
4. Do shuttle and three-cone COD counts match drill design? What distinguishes high from low braking and re-acceleration?
5. Do features differ by cohort beyond what athlete differences explain? M1 shows that 2025 tracking is about 4× smoother.
6. Which features are redundant (|ρ| ≥ 0.9) and should be dropped before modelling?

## 2. Commands (USER)
```
python -m pytest tests/test_phase03_m2_combine.py -q
python -m nflbdb.research.combine_movement
python -m nflbdb.research.combine_analytics
```
Then open `notebooks/03_combine_advanced_eda.ipynb` and `06_movement_reliability.ipynb`; both reuse the persisted outputs.

## 3. Outputs
| Kind | Path | Answers |
|---|---|---|
| CSV | `tables/phase_03/forty_window_diagnostics.csv` | Q1 |
| CSV | `tables/phase_03/timing_validation.csv` | Q2. Each row names `official_measure` (`cmb_*`, official Combine result) and `tracked_measure` (`trk_*` / onset-based, with `tracked_basis`). Diff = tracked − official, using each athlete's fastest whole attempt. Columns: bias (median/mean), diff p05/p95, MAD, MAE after bias, r, ρ, slope, share within ±0.05 s after bias, heuristic `agreement_class`, and truncation shares (`share_onset_censored`, `share_end_censored`, `share_split_extrapolated`, `share_peak_at_end`) |
| CSV | `tables/phase_03/agility_cod_structure.csv` | Q4 |
| CSV | `tables/phase_03/combine_feature_summary.csv` | distributions by drill × cohort |
| CSV | `tables/phase_03/cohort_shift.csv` | Q5. Within-position standardised shift, an official-time anchor shift, the M1 noise ratio and a noise-sensitivity flag, combined into a heuristic `interpretation`: no_material_shift / likely_processing_artifact / athletic_difference_possible / unresolved |
| CSV | `tables/phase_03/feature_redundancy.csv` | Q6 |
| CSV | `tables/phase_03/position_comparisons.csv` | position profiles (athlete medians) |
| CSV | `tables/phase_03/m2_example_selection.csv` | deterministic examples + rule |
| Figures | `m2_forty_timing_validation`, `m2_agility_timing_validation`, `m2_forty_speed_profiles_by_position`, `m2_sprint_model_quality`, `m2_forty_similar_time_different_profile`, `m2_forty_acceleration_extremes`, `m2_forty_attempt_stability`, `m2_shuttle_braking_extremes`, `m2_three_cone_path_extremes`, `m2_cohort_shift`, `m2_position_distributions`, `m2_forty_feature_correlation` | all |

Every table also has a Parquet twin (`<name>.parquet`) for typed reuse, loaded with `CombineMovementAnalytics(cfg).load_table("<name>")`. `python -m nflbdb.research.combine_analytics --check` lists any of the 44 expected files (9 tables × CSV + Parquet, 13 figures × PNG + SVG) that is missing. An empty table is written as a one-row `note` placeholder, never silently skipped.

### Acceptance criteria (M2)
1. All tests pass: `test_phase03_kinematics`, `test_phase03_m2_combine`, `test_phase03_m2_hardening`, `test_polars2_compat`.
2. `--check` reports all 44 outputs present. A missing figure is acceptable only with a logged reason, such as too few examples.
3. In `forty_window_diagnostics`, the median run distance is about 40 yd and `share_fit_ok` is high. Each truncation share is reported and explained.
4. `timing_validation`: a tracked measure may be called an estimate of an official time only if `agreement_class = interchangeable_after_bias` in every cohort. Otherwise it stays a tracked feature.
5. `feature_reliability`: a feature counts as reliable only if it is labelled good or excellent (≥ 20 athletes with ≥ 2 attempts, CI lower bound ≥ 0.75, non-degenerate distribution).
6. `cohort_shift`: every `likely_processing_artifact` row is either excluded or carried with a cohort adjustment into M4. Nothing is interpreted as an athletic difference without an official-time anchor.

## 4. Example selection (deterministic, NFL-outcome-blind)
| Example | Rule |
|---|---|
| High / low acceleration | Athlete median `model_a0` closest to its p90 / p10 |
| Similar time, different profile | Adjacent athletes by official forty (\|Δ\| ≤ 0.02 s) that maximise \|Δτ\|; ties go to the pair closer to the median forty |
| Stable / variable attempts | Min / max within-athlete SD of `model_vmax` (≥ 2 fit-ok attempts) |
| High / low braking | Shuttle athlete median `cod_approach_decel_median` closest to p90 / p10 |
| Short / long path | Three-cone athlete median path length closest to p10 / p90 (fixed course) |

Remaining ties go to the lowest `nfl_id`.

## 5. Results
_Pending user execution. Fill in from the CSVs above; record n per cohort; do not round away uncertainty._

## 6. Limitations
- Tracked timing conventions (model start, threshold onset) differ from official electronic or hand timing.
- The mono-exponential model under-fits the first steps from a stance. `model_a0` is a model parameter.
- Cohort differences mix athlete populations and tracking processing.
- Position-specific drills are analysed within drill only. Generic features computed on them are not comparable across drills.
