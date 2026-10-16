# Combine Intelligence Lab — technical documentation (Milestone 09)

_Authored 2026-10-14. Nothing has been executed._

## Modules (`src/nflbdb/research/advanced_movement/`)

| Module | Content |
|---|---|
| `movement_demands.py` | `DemandConfig`, `demand_specs` (9 demands), pure trace functions `runs` · `crossing_time` · `band_accel` · `segment_trace`, `batch_episodes`, `DemandBuilder` (reuses `NflOutcomeBuilder` batches, anchors, kinematics; `context_labels` reads vendor route / coverage columns only if present; `example_frames`), tables EPISODES · REGISTRY · FREQUENCY · PROFILE · CONTEXT · AVAILABILITY · ATHLETE · DISTRIBUTION · EXAMPLES, `variance_components`, `choose_examples`, `conclusions` |
| `movement_matching.py` | `MatchingConfig`, `combine_features` (8), `trace_features` (same band rule), `attempt_table`, `prospect_tables` (SE from pooled within-athlete SD; local ICC / in-sample R² screen; 08B.4 status when available), `dictionary_table`, `similarity_table` (coverage, interval, robust z, abstention states), `matrix_table`, `summary_table` |
| `stopwatch_twins.py` | `TwinConfig`, `twin_index` (sorted scan), `difference_table` (SE of the difference via `scouting_intelligence.pairwise_difference`), `rank_twins`, `twin_summary`, `select_cases` (10 types), `missing_case_types` |
| `scouting_interpretation.py` | standard library only: `INTERPRETATIONS` (7 answers per metric), `pair_reading`, `case_views`, `views_rows`, `prospect_report`, `render_html`, `assert_language` |
| `milestone09_artifacts.py` | mart layout `MART_TABLES`, `curves_table`, `bands_table`, `profiles_table`, `evidence_document`, `write_mart`, `load_mart`, `overview`, `validate_mart`, contracts VIEWS · INTERPRETATION · CASE_NFL |
| `intelligence_figures.py` | matplotlib figures `AM09*` incl. the twin story and the one-page intelligence card |
| `intelligence_validation.py` | 10 synthetic checks in 5 groups |
| `artifacts.py` | + `IntelligenceStore` (area `intelligence_v1`, `.football`, `mart_dir`, `reports_dir`) |

## Notebooks (`notebooks/advanced_movement/`)

09a → 09g, each with parameter cell, `RUN_MODE` smoke / full, synthetic checks first, explicit output contract, reading, "send back" list. 09A caches its tracking scan (`store.build('DEMAND_ATLAS')`); the others recompute in seconds to minutes.

## Data contracts (warehouse `gold/advanced_movement/intelligence_v1/`)

All tables are `TableContract`s validated on write (exact dtypes, unique non-null keys) with a schema sidecar. The application mart (`app_mart/v1/`) is plain Parquet + `mart_manifest.json` (rows, columns, fingerprint per file) + `scouting_evidence_registry.json`.

## Safeguards (local to the features that need them)

- Pre-draft tables refuse NFL-named columns (`assert_no_nfl_columns`).
- Case selection never reads an NFL outcome value; one type reads an exposure count and is flagged.
- Abstention states instead of imputation everywhere; abstentions carry no interval (validated).
- Generated text passes the wording guard.
- Every mart row-count and fingerprint is checked against the manifest; grade-like columns are refused.

## Assumptions not verified by execution

- `compute_kinematics` output carries `yaw_rate`, `heading_valid`, `step_smooth`, `pos`, `seg_len`, `seg_idx`, `ts`, `x_s`, `y_s` (read from the module source).
- `nfl_play_kinematics` returns the same columns for one play.
- Combine sequence frames and NFL frames use comparable Savitzky–Golay settings (stated as a caveat in the dictionary).
- `silver.player_play` / `silver.plays` may or may not hold a route or coverage column; the first column whose name contains "route" / "coverage" is used as recorded.
- `sequence_frames` has `t_rel`, `dist`, `speed`, `a_tan`, `boundary`; `athlete outcomes` hold `O2.accel_sust05_mean`.
- Streamlit ≥ 1.39 (`st.navigation`, `st.Page`, `st.switch_page`, material icons); Plotly ≥ 5.24.

## Application

`apps/combine_intelligence_lab/` — see its README. Reads the mart only; the interpretation module is imported from `src/`.
