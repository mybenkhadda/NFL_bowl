# Running `M12_UNIFIED_COLAB_RESEARCH.ipynb` on Google Colab

The notebook `notebooks/nflbowl/M12_UNIFIED_COLAB_RESEARCH.ipynb` runs all of Milestone 12 (12.0, 12A–12G) from one configuration and saves every output to Google Drive. What it merged and why: `docs/nflbowl/M12_MERGE_MATRIX.md`.

> This notebook was written and reviewed by reading the code only. It has not been run yet. The first Colab run is its first real test. Section 9 lists what to watch.

## 1. What to put on Drive

| Input | Needed for | Default location (`CFG`) |
|---|---|---|
| **Mode A**: portable M12F bundle (a folder, or `m12f_v1.zip`) | 12F (GPU encoder) and the bundle part of the EDA | `bundle_dir = /content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1` (`bundle_zip` for an archive) |
| **Mode B**: research root, i.e. a copy of the research machine's repository root with at least `warehouse/gold/advanced_movement/` (optionally `data/`, `reports/milestone12/`, `artifacts/`) | 12.0, 12C, 12B, 12A, 12D, 12E, 12G | `research_root = /content/drive/MyDrive/nfl-big-data-bowl-2027/research_root` |

- Build the bundle on the research machine: `python -m nflbowl.data.builder --repo-root . --output bundles/m12f_v1 --zip` (see `BUNDLE_BUILD_GUIDE.md`).
- For Mode B, upload `warehouse/gold/advanced_movement/` with its sub-areas `discovery_v1/`, `intelligence_v1/`, `advanced_v1/` and `transfer/`. A full run reads the non-`_sample` paths. The notebook lists every expected input file with its size and row count (Part 3.3, `diagnostics/warehouse_inventory.csv`) and checks each stage's inputs when the stage starts.
- Rebuilding the 12.0 Combine curve tables from raw frames also needs the Milestone 00 inputs. This happens only when the tables are missing or `options['12.0']['force_rebuild_curves']` is set. Normally the persisted `m12_combine_*` tables are reused.
- With Mode A only, every warehouse stage is skipped with its reason, and 12F, the EDA, Part 11 and the final report still run.
- With Mode B only, set `build_bundle_from_warehouse = True` to build the bundle in Colab (`nflbowl.data.builder`). Otherwise 12F is skipped.

## 2. Steps

1. *Runtime → Change runtime type → GPU* (a T4 is enough).
2. **Part 1.1**: installs the package from `@main` with `--no-deps` and installs its pure-Python dependencies separately. Colab's CUDA PyTorch is never touched. Optional: `INSTALL_POT` (exact W2 for 12E-E2) and `INSTALL_PYMC` (12B-B4).
3. If `nflbowl` had already been imported in this runtime: *Runtime → Restart session*, then continue at Part 1.2.
4. **Part 1.2** must print `nflbowl 0.3.0` and `fp32 … regions present: True`, plus the GPU, CUDA, memory and disk facts.
5. **Part 1.3** mounts Drive.
6. **Part 2**: edit `CFG`, then run the next cell. It validates the config (unknown keys are errors), creates `RUN_DIR` and saves `config/unified_config.json` and `config/environment.json`.
7. Run all remaining cells in order (*Runtime → Run all* works). Each stage prints one line saying whether it will run, be reused, be skipped (with the reason) or is disabled.

## 3. `CFG`: what to set

| Key | Default | Meaning |
|---|---|---|
| `experiment_name` | `m12_unified` | `RUN_DIR = <drive_root>/unified_research/<experiment_name>`. The same name reopens everything after a disconnect. |
| `run_mode` | `fast` | `debug`: wiring check with the encoder smoke preset (numbers are not results). `fast`: every primary analysis; expensive secondary analyses off. `full`: every declared analysis of 12.0–12G. |
| `run_full_research` | `False` | `True` also trains the other encoder families, the ablations, the inductive F2/F4 encoders and the learning curve. That takes hours. |
| `seed` | `20261015` | The declared Milestone 12 seed. Another seed means another experiment. |
| `precision` | `fp32` | `amp` = fp16 autocast + GradScaler on a T4, allowed only after Part 1.2 shows the fix. `bf16` is for A100 / L4 only and does not match the notebook's numerics. Precision is part of the checkpoint signature. |
| `encoder_science_overrides` | `{}` | Batch / gradient accumulation (`{'science.accum_steps': 2}`) and similar. These change the science hash, so the run gets its own experiment directory (`<name>_<hash8>`). |
| `encoder_experiment_dir` | `None` | Reopen an existing nflbowl experiment, e.g. `/content/drive/MyDrive/NFLBowl/experiments/f4_main`, to continue its checkpoints. |
| `encoder_inductive` | `False` | F2 + F4 × 5 folds. Needed for the inductive retrieval, the NFL-transfer ladder and the 12G encoder block. |
| `stages` | all `True` | Switch any section off. A disabled section touches none of its inputs. |
| `options` / `knobs` | run-mode defaults | Per-stage optional analyses (see below) and loop sizes. Explicit values win. |
| `reuse_completed_stages` | `True` | A warehouse stage already executed in the same run mode with the same `M12Config` hash is not recomputed: its tables, figures and conclusion are copied into `RUN_DIR`. |
| `continue_on_error` | `True` | A failing stage is recorded (`logs/<stage>_traceback.txt`, `manifests/stages.json`). Stages that depend on it skip themselves and the rest continues. Set to `False` to stop at the first error. |

### Optional analyses (`options`), with their defaults per mode

| Stage | Option | debug | fast | full |
|---|---|---|---|---|
| 12.0 | `force_rebuild_curves` | – | – | – |
| 12C | `permutation`, `replication` (C7) | – | – | ✔ |
| 12B | `synthetic` (checkpointed grid) | – | – | ✔ |
| 12B | `bayes` (PyMC) | – | – | – (never implicit) |
| 12A | `elastic` (SRSF alignment) | – | – | ✔ |
| 12D | `bootstrap` (loading stability) | – | – | ✔ |
| 12E | `heavy_methods` (E4, E6), `eps_sensitivity`, `own_distance_bootstrap` | – | – | ✔ |
| 12E | `consistency`, `twins`, `downstream` | – | ✔ | ✔ |
| 12E | `controls` | ✔ | ✔ | ✔ |
| 12G | `tournament` | ✔ | ✔ | ✔ |
| 12G | `ot_block` | – | ✔ | ✔ |
| 12G | `cohort_size` | – | – | ✔ |

An analysis that is off is reported as "not run" in its stage report, never as a negative result.

## 4. Notebook outline

| Part | Content |
|---|---|
| 0 | Research overview, questions Q1–Q7, architecture diagram (Mermaid) |
| 1 | Install, versions and fix check, GPU / CUDA / memory, Drive |
| 2 | `CFG` → `UnifiedConfig`, `RUN_DIR`, environment record, seed |
| 3 | Input modes and stage plan · bundle staging + validation · warehouse context + inventory + software checks · **12.0 foundation** |
| 4 | Unified EDA (bundle + foundation) |
| 5 | **12C** distributional NFL movement capacity: H1 reliability, held-out forecasts, **target freeze**, H2 transfer, controls, C7 replication |
| 6 | **12B** measurement error: ICC, RC / SIMEX / EM estimands, held-out B0 / B1, synthetic grid |
| 7 | **12A** functional transfer: FPCA / splines / hierarchical vs scalar bands, pointwise, elastic alignment |
| 8 | **12D** multi-view latent traits: factor models, test–retest (primary), stability, transfer |
| 9 | **12E** conditional optimal transport: E0–E6 retrieval, controls, consistency, bootstrap, twins, downstream |
| 10 | **12F** shared encoder on GPU: open / prepare → train F4 → (opt-in extras, inductive, learning curve) → embed + evaluate → transfer → collect / publish |
| 11 | Unified evaluation: diagnostic · measurement · transductive · in-sample · inductive · outcome-linked, retrieval side by side, controls, domain shift, probes, consistency, calibration, ablations, folds |
| 12 | **12G** synthesis + tournament, decision matrix for scouts, GMs, coaches and player development |
| 13 | Final report, stage records, artifact inventory, ZIP export |

The methods run in dependency order: 12C freezes the NFL target that 12A, 12F and 12G use.

## 5. Drive artifact structure

```text
/content/drive/MyDrive/NFLBowl/unified_research/<experiment_name>/
├── config/            unified_config.json, environment.json
├── logs/              run.log, <stage>_traceback.txt
├── checkpoints/       experiments/<name>_<hash8>/ (nflbowl: epoch shards, best.pt, optimizer/scheduler/GradScaler/RNG state,
│                      embeddings, evaluation records) + *.model.json (checkpoint pointers, signatures)
├── processed_data/    reusable datasets (m12_combine_*, m12_cohort_registry, m12c_targets, eda_sequence_summary, …) + *.provenance.json
├── features/
├── embeddings/        copies of the embedding Parquet files + INDEX.csv/.parquet
├── tables/
│   ├── csv/           every result table (tables > csv_max_rows: Parquet only, noted in the manifest)
│   └── parquet/
├── figures/
│   ├── png/  svg/  pdf/   e.g. 12c_envelope_vs_quantile_reliability.png, eda_movement_signal_distributions_by_domain.svg, 12f_training_curves.pdf
│   └── data/          plotted data (CSV + Parquet) and a JSON sidecar per figure
├── reports/
│   ├── markdown/      eda_unified.md, 12_0_foundation.md, 12a_functional_transfer.md, 12b_measurement_error.md,
│   │                  12c_distributional_capacity.md, 12d_multiview_traits.md, 12e_optimal_transport.md, 12f_shared_encoder.md,
│   │                  12g_research_synthesis.md, unified_evaluation.md, FINAL_RESEARCH_REPORT.md
│   └── html/          the same reports as HTML
├── diagnostics/       stage_plan, bundle_validation, warehouse_inventory, software_checks, memory plans
├── manifests/         artifacts.jsonl (append-only), artifact_index.json, stages.json, artifact_inventory, research/ (warehouse JSON copies:
│                      m12c_target_freeze.json, m12d_selected_representation.json, m12_run_*.json, …)
└── exports/           <experiment_name>_<timestamp>.zip
```

The warehouse stages also keep writing their own outputs into the research root's warehouse, as the original notebooks did (same paths, same immutable 12C freeze). The copies in `RUN_DIR` are what you share.

## 6. After a disconnect

1. Reconnect, then run Parts 1.1 (restart if prompted), 1.2, 1.3 and 2 with the same `experiment_name`.
2. Run all. Finished warehouse stages are reused. The EDA reloads its exact statistics. `ENC.train_primary()` resumes F4 from the last epoch checkpoint, or restores a finished run without retraining. `ENC.resume_unfinished()` continues any other job left running. Embeddings and evaluation records are reused when their inputs are unchanged.
3. The 12B synthetic grid resumes per scenario (immutable checkpoint shards).

## 7. Performance notes

- The bundle is staged once to `/content/nflbowl_stage` and memory-mapped. Random batch reads straight from Drive would be very slow.
- GPU training (Part 10) and the CPU-heavy evaluation are separate phases. Evaluation is exact by default: no approximate top-k and no GPU similarities unless you opt in through the nflbowl evaluation settings.
- The EDA streams the arrays in 32k-sequence chunks (about 50 MB at a time).
- 12E distances are computed only between same-position pairs, which is what retrieval uses. E4 and E6 are the expensive methods and are opt-in.
- Large writes can go to local disk first and then be copied once to Drive (`ResearchRun(local_tmp=...)`).

## 8. Precision / AMP

Start with `precision = 'fp32'`. Switch to `'amp'` only after an fp32 F4 run has finished and Part 1.2 shows `FIX['ok'] == True`. AMP checkpoints get their own signature, so they never mix with fp32 ones. If an AMP run raises `FloatingPointError` (the trainer names the loss term, epoch and batch), go back to fp32. That separates numeric overflow from data problems.

## 9. Not verified yet (static authoring only)

- None of the new code has run: `nflbowl.research.*`, the notebook and `tests/nflbowl/test_nflbowl_unified_research.py`.
- The stage runners copy the source notebooks' cells line for line. Mistakes are most likely in the new glue (mirroring, reuse, reports) and in **12E**, which was re-authored on the module API: consistency, the twins ranking, the uniform-mass control, the ε grid and the downstream ladder are new orchestration of existing module functions.
- Matplotlib layout details of the EDA / evaluation figures (`constrained_layout` with a suptitle band).
- 12G with the encoder block taken directly from `Experiment.inductive_athlete_embeddings` rather than from warehouse tables.
