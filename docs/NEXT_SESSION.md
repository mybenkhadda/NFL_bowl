# Next session: M14 cross-sport movement intelligence (start here)

**Rules:** code authoring only. Never execute Python, tests, shell or notebooks; the user runs everything in Colab. Read this file and `docs/nflbowl/M14_RESEARCH_SPEC.md`, then read only the module that failed.

## State (2026-10-17, end of session 7)

- Package version is 0.6.0.
- Nothing from M14 has been executed.

**M14 is fully authored:**

| Kind | Path |
|---|---|
| Library | `src/nflbowl/lab/m14/`: `config`, `registry`, `events`, `metrics`, `reliability`, `novelty`, `transfer`, `scouting`, `reporting`, `figures` |
| Notebook | `notebooks/nflbowl/M14_CROSS_SPORT_ATHLETIC_INTELLIGENCE_COLAB.ipynb` (Setup 1–4, Parts A–O) |
| Tests | `tests/nflbowl/test_nflbowl_m14.py` (synthetic signals; unexecuted) |
| Spec | `docs/nflbowl/M14_RESEARCH_SPEC.md` (definitions, status rules, safeguards) |

**Not changed:** `api.py`, the ladder engine, and every M12 / M13 module. Only the package `__init__` docstring and the version were edited.

## Module map (notebook part → APIs)

- **A — environment and hand-off.**
  - `cfg.m13()` builds an `M13Config` for the M14 experiment.
  - It drives `lab.handoff.environment / open_experiment / discover / adopt / adopt_embeddings / inventory / ready_models`.
  - Checkpoint sources: the M12 experiment, plus the M13 experiment when present.
- **B — feasibility and extraction.**
  - `registry.availability`.
  - `metrics.cached_extract` → `features/events/events.parquet` and `features/sequence/sequence_records.parquet`, keyed by (bundle, event-params hash).
- **C — definitions.** `registry.definitions_frame / definitions_json` → `tables/registry/`.
- **D — event tables.**
  - `metrics.sequence_table`, then `metrics.athlete_table` → cached as `features/sequence/sequence_table` and `features/athlete/athlete_features`.
  - Figures: `figures.aligned_profile / event_anatomy`.
- **E–H — family analyses.** Notebook-level summaries plus `novelty.residual_specs / oof_residual`.
- **I — reliability.**
  - `reliability.trial_cache` → `reliability_table / matched_k_curves / projected_curve / athlete_level_split_half` (cached, signature `reliability/core`).
  - `metrics.robustness_rows` + `cached_perturbed` → `reliability.sensitivity`.
  - `coverage` → `measurement_status`.
- **J — novelty.**
  - `transfer.analysis_frame` (the official frame + `m14c_*`).
  - `novelty.novelty_table / spearman_matrix / redundancy_clusters` (existing features: biomech `c_*` via `handoff.sequence_metrics`, F0 `eng*`).
- **K — Hidden Braking Advantage.** `scouting.signature_columns → braking_signature → twin_covariates → stopwatch_twins → twin_contrast / hidden_variation`.
- **L — transfer.**
  - `TF.ladder_predictions` and `TF.baseline_predictions`.
  - `transfer.build_models` → `run_ladder` (cached `datasets/cached/m14_ladder_oof`).
  - `integrate` (reproducibility checks + alignment + cohorts) → `evaluate` → `by_position`.
- **M — status and robustness.** `transfer.fold_stability / transfer_status / selection_frequency / permutation_null / permutation_p`.
- **N — case studies.** `scouting.select_case_studies / case_study_markdown`, `reporting.write_profile`, `figures.twin_profile`.
- **O — synthesis and report.** `scouting.evidence_table / research_answers`, `figures.status_matrix`, `reporting.final_report / write_report`, `RUN.write_status`, `RUN.export_zip`.

## Paths

| What | Path |
|---|---|
| Bundle | `/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean` |
| M12 (read only) | `/content/drive/MyDrive/NFLBowl/experimental_lab/m12_comprehensive_lab_v1` |
| M13 (read only) | `/content/drive/MyDrive/NFLBowl/experimental_lab/m13_outcome_transfer_v1` |
| Output | `/content/drive/MyDrive/NFLBowl/experimental_lab/m14_cross_sport_athletic_intelligence_v1` |
| Stage ledger | `manifests/stages.json`, `manifests/m14_status.json` |
| Cache signatures | `manifests/stage_signatures.json` |
| Tracebacks | `logs/<stage>_traceback.txt` |

## Waiting on the user (Colab)

1. Setup 2 prints `nflbowl 0.6.0`.
2. Run all cells with the defaults.
3. Send back:
   - `manifests/stages.json`
   - every `logs/*_traceback.txt`
   - `tables/feasibility/signal_quality_by_kind.csv`
   - `tables/events/event_completeness.csv`
   - `tables/reliability/measurement_status.csv`
   - `tables/transfer/m14_reproducibility.csv`
   - `reports/markdown/m14_final_report.md`
4. Locally: `pytest tests/nflbowl/test_nflbowl_m14.py -q`.

## Known limitations / likely first failures (unverified API details)

**Polars:**
- `DataFrame.unpivot` (≥ 1.0; older versions use `melt`).
- `pivot(on=…)` and `with_row_index`.
- `str.extract` with a raw-string regex.
- `list[f32]` trace columns in `write_parquet` / `pl.DataFrame(rows)`.
- `fill_nan` on Float columns only.

**pandas:** `Styler.map` (pandas ≥ 2.1).

**Ladder engine:** `ae.oof_ladder` must accept a factory block that concatenates the embedding block with M14 columns (`transfer.combine_factories`, columns prefixed `p<k>_`).

**Runtime:**
- B1 is the expensive pass: a Python loop over about 77k sequences, estimated at minutes.
- I1 runs an ICC bootstrap with `NBOOT_FAST` × about 40 columns.
- M2 runs 20 ladder refits.

**Scientific:**
- The yaw / a_n sign agreement (D1) checks the turn-direction convention.
- The cohort is small (about 246 athletes): expect wide CIs.
- Position explains about 94 % of the target variance (M13 context, not an M14 result).

## Open scientific points to check once results exist

- Does the detector fire on agility / NFL braking windows and stay near zero on the forty (B1 figure)?
- How many metrics are measurement-`validated`? Is the braking signature non-empty? (If it is empty, K and N are blocked.)
- Does `M14Run` reproduce the official ladder (`tables/transfer/m14_reproducibility.csv`, all rows `pass`)?
- What are the primary statuses, the permutation p-value, and the twin contrast on `nfl.q90_shared`?

## Exact next action

User: run the M14 notebook in Colab. Claude: fix only the reported failures (start from the failing stage's traceback and module).
