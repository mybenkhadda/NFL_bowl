# M13 execution guide (Colab)

## 1. Configuration

- **Runtime:** GPU (T4 or better). The GPU is needed only when an M12 embedding file cannot be adopted and must be recomputed from the adopted checkpoint. Everything else runs on CPU.
- **Precision:** FP32 (enforced by `m13_config.resolve`).
- **Drive inputs:**
  - Bundle: `/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean`, fingerprint `4867be08f8191b3a`.
  - M12 lab: `/content/drive/MyDrive/NFLBowl/experimental_lab/m12_comprehensive_lab_v1`, with its experiment under `checkpoints/experiments/m12_comprehensive_lab_v1`.
  - Optional: the path of `m12c_target_freeze.json` as `CFG['extra']['freeze_json']`. This documents the frozen estimator in B1.
- **Output:** `/content/drive/MyDrive/NFLBowl/experimental_lab/m13_outcome_transfer_v1/`. Nothing under the M12 directory is ever written.

## 2. Run order

1. **Setup 1** installs from `@main` (force-reinstall, no deps) plus `markdown`. Then restart the session if `nflbowl` was imported earlier.
2. **Setup 2** checks the version: it must print `nflbowl 0.5.0`.
3. **Setup 3** sets `CFG`. Keep `run_training=False` and `allow_new_training=False`.
4. **Setup 4** loads the imports, the navy theme and the run directory.
5. Run Parts A → K in order.

   **Hard stops:**
   - A1: bundle checks
   - A2: adoption
   - A4: blocking prerequisites
   - C1: ladder and reproducibility assertion
   - C2/C3: fairness assertions
   - J1: synthesis

   Other stages fail softly (traceback in `logs/<stage>_traceback.txt`), and later independent stages still run.
6. **Collect** these and send them back:
   - `manifests/stages.json`
   - `logs/*_traceback.txt`
   - `tables/handoff/checkpoint_inventory.csv`
   - `tables/transfer/common_cohort_audit.csv`
   - `reports/markdown/M13_FINAL_RESEARCH_REPORT.md`

Local tests (no data needed): `pytest tests/nflbowl/test_nflbowl_m13.py -q`.

## 3. Resuming after a disconnect

Rerun Setup 2 → 4 and every Part with the same `experiment_name`. The following are reused:

- adopted checkpoints and embeddings (experiment manifest)
- the ladder OOF record (`transfer_oof` with exact deps)
- the cached sequence metrics (provenance-matched)
- signature-checked tables for E1, E2 and G1 (inputs unchanged)

Cheap stages recompute.

## 4. Stage statuses

| Status | Meaning |
|---|---|
| `executed` | Ran now. |
| `reused` | Signature-checked cache hit. |
| `skipped` | Disabled by a flag (`reason_disabled`). |
| `blocked` | A prerequisite is missing (e.g. no model to explain). |
| `failed` | See the traceback. |

The K1 stage table shows these statuses for every stage.

## 5. Expected artifacts

- `tables/transfer/common_cohort_audit.csv`
- `tables/transfer/common_cohort_predictions.parquet`
- `tables/transfer/paired_transfer_comparisons.csv`
- `tables/<family>/*.parquet|csv` for the families `handoff`, `target`, `transfer`, `reliability`, `position`, `domain`, `retrieval_transfer`, `scouting` and `synthesis`
- `figures/png|svg|pdf/<name>.*`, plus `figures/data/<name>__<part>.parquet|csv` and `<name>.meta.json`
- `reports/markdown/M13_FINAL_RESEARCH_REPORT.md` and `reports/html/M13_FINAL_RESEARCH_REPORT.html`
- `reports/athlete_profiles/A-XXXXXX.md`
- `datasets/cached/{ladder_oof, baseline_oof, sequence_metrics, biomech_selection_oof}.parquet` (each with a `.provenance.json` sidecar)
- `manifests/{provenance, handoff, outcome, interval_check, f1_validity}.json`
- `manifests/table_provenance.jsonl`
- `manifests/stage_signatures.json`
- `exports/*.zip`

## 6. Unverified API details (likely first failures)

None of the M13 code has been executed. If something fails, check these first:

- **Polars:**
  - `pivot(on=, index=, values=)` without an aggregate in `TF.wide`
  - `int_range(pl.len()).over(a, b)` in `REL.trials`
  - `.list.len()`
  - `unpivot`
  - `str.replace(..., literal=True)`
  - `rank('ordinal')` cast to Int64
  - `replace({...})` on strings
- **pandas Styler:** `hide(axis='index')` and `map` vs `applymap`, in `M13F.styled`.
- **matplotlib:** `Figure.add_artist(Line2D)` with `transFigure`, and `FancyBboxPatch` with `rounding_size`.
- **Private engine helpers used:** `ae._alpha` and `ae._fit_predict` (`position_models`), and `Experiment._ladder_inputs` (`TF.baseline_predictions`).
- **Runtime:**
  - E1 bootstrap over NFL windows with `NBOOT_FAST` (up to 1000) can take several minutes. Lower `n_boot` to speed it up.
  - G1 runs every fold × model × condition.
- **Palette:** the navy theme reuses the palette's dark-mode steps, which were validated against `#1a1a19`, not against the navy surface `#0b1626`. Validate before publication, or set `figure_theme='light'`.

## 7. Fix strategy

Each Part is isolated by `run_stage`. To fix a failure:

1. Fix the module.
2. Push and reinstall from `@main`.
3. Restart the session.
4. Rerun from Setup 2 with the same `experiment_name`.
