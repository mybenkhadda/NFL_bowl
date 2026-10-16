# M12 experimental lab — API audit

_Static audit (2026-10-16), done by reading the code only. Nothing was executed._

## 1. What exists and is reused

| Capability | Existing API | Notes |
|---|---|---|
| Bundle open, staging and validation | `nflbowl.data.loading.open_bundle`, `SequenceBundle.summary()` | read-only; the fingerprint is the sha256 map of the bundle files |
| Experiment directory, science hash, manifest | `nflbowl.Experiment`, `artifacts.Manifest` | an existing name is reopened; a science mismatch raises |
| Inductive training and resume | `Experiment.train(model, regime='inductive', folds=[f], resume=True)` | epoch shards with RNG and scaler state; precision is part of the checkpoint signature |
| Embeddings | `Experiment.embed(..., regime='inductive')`, `load_embeddings` | Parquet with `split_role`; reuse is keyed by the weights' sha256 |
| Transductive evaluation | `Experiment.evaluate`, `metrics.csv` | diagnostic only |
| Paired inductive retrieval | `Experiment.compare_inductive` → `InductiveComparison` | identical pools (`pool_hash`), fold-stratified paired bootstrap, position subgroups, domain diagnostics |
| Domain / collapse diagnostics | `evaluation.domain_diagnostics.*` | athlete-grouped domain AUC, mean gap, alignment, collapse, pair coverage |
| Transfer ladder | `Experiment.transfer` (`ae.oof_ladder`, `ae.compare`) | train-fold design, fold-fitted embedding PCA, inner α. **It saved only the comparison rows, not the predictions.** |
| Model registry | `models.registry` (`se.FAMILIES`, `se.ABLATIONS`) | F0, random, F1 (external), F2–F6, and the ablations F4_nophys / noalign / adv / mmd / category / shuffled_phys |
| Run tree and atomic writers | `research.artifacts.ResearchRun` | subclassed by `lab.store.LabRun` |
| Colab helpers | `research.colab` (Drive, versions, `fixed_losses_check`, GPU memory) | |
| Plot primitives | `visualization.plots` (palette, `forest`, `heatmap`) | extended by `lab.figures` |
| Reliability, BH, permutation | `ae.split_reliability`, `ae.bh`, `ae.add_adjusted`, `ae.permutation_blocks` | |

## 2. Added (backward compatible)

- **`Experiment.derived`**, **`Experiment.folds_by_athlete`** and **`Experiment.splits(f)`** are public accessors over existing internals.
- **`Experiment.transfer_predictions(models, reuse)`** saves the out-of-fold predictions of exactly the `transfer()` ladder as an immutable record, `transfer_oof__<hash>.parquet`. It also adds `has_<encoder>` coverage flags and the M1 columns. The construction is shared through `_ladder_inputs`; `transfer()` behaves as before.
- **`nflbowl.lab`**: `config`, `store`, `registry`, `environment`, `audit`, `geometry`, `retrieval_lab`, `transfer_lab`, `ablations`, `robustness`, `biomech`, `interpret`, `synthesis`, `figures`, `stats`.
- Version 0.4.0; optional extra `lab = [umap-learn, markdown]`.

## 3. Per experiment

Classes: **E** = existing and directly usable · **A** = existing, needs adaptation · **N** = new module · **B** = blocked by unavailable data.

| Exp | API used | Class | Inputs | Portable bundle? | Compute | Leakage risk and control | Outputs |
|---|---|---|---|---|---|---|---|
| A2 | `Experiment.cfg`, `research.colab` | E | bundle | yes | trivial | — | config/*.json |
| B1 | `lab.audit.dataset_overview`, `biomech.sequence_table` | N | bundle | yes | CPU, ~1–2 min (vectorised metrics) | none (descriptive) | tables/dataset |
| B2 | `audit.domain_composition` | N | sequence metrics | yes | CPU | cluster bootstrap avoids row pseudo-replication | tables/dataset, figures b2_* |
| B3 | `audit.attrition` | N | athletes.parquet | yes | trivial | — | attrition tables |
| B4 | `audit.leakage_audit` (+ `pi.assert_fold_isolation`, `verify_held_out_roles`) | N/A | splits, manifests | yes | CPU (hashing 77k sequences) | checks are labelled runtime vs code inspection | diagnostics, tables |
| C1/C3 | `geometry.run_geometry` (+ `domain_diagnostics`, `rep.domain_shift`) | A | inductive embeddings | yes | CPU, plus embedding of the random encoder | held-out rows only | tables/geometry |
| C4 | `geometry.projection_coordinates` | N | embeddings | yes | CPU (UMAP optional) | fit scope recorded | figures c4_* |
| D1 | `environment.transductive_reference` | A | metrics.csv of earlier experiments | yes, if present | trivial | shown only for the same bundle; never pooled | tables/retrieval |
| D2 | `Experiment.train` via `environment.train_missing` | E | bundle | yes | **GPU-heavy** (≈ minutes to an hour per fold on a T4, unverified) | held-out athletes absent, checked by the split integrity checks | checkpoints |
| D3/D4 | `compare_inductive` + `retrieval_lab` | E + N | embeddings | yes | CPU | identical pools; complete cases | tables/retrieval |
| D5 | `retrieval_lab.select_case_studies`, `speed_profiles` | N | scores, X | yes | CPU | pseudonyms | figures d5_* |
| E2–E7 | `transfer_predictions` + `transfer_lab` | A + N | target, embeddings for all folds | yes | CPU | train-fold design; NFL metrics never used as predictors | tables/prediction |
| E (perm) | `ae.permutation_blocks` | E | as E2 | yes | CPU-heavy (n_perm ladders) | — | permutation_test |
| F1 | `compare_inductive(reference=F4)`, `ablations.*` | E + N | ablation encoders | yes | **GPU-heavy** (6 ablations × folds) | same pools as F4 | tables/ablations |
| G1 | `robustness.seed_scores` (separate experiments) | N | bundle | yes | **GPU-heavy** | seeds never share artifacts | tables/robustness |
| G2 | `Experiment.train(nfl_fraction)` + `evaluate` | A | bundle | yes | GPU | **API-limited: transductive only**, labelled as a diagnostic | learning_curve |
| G3 | `retrieval_lab.pool_difficulty` | N | embeddings, M1 | yes | CPU | pools defined from metadata / officials only | pool_comparison |
| G4 | `retrieval_lab.quality_sensitivity` | N | embeddings, sequence metrics | yes | CPU | pools identical across representations within a filter | quality_comparison |
| G5 | `stats.paired_in_subgroups` | N | scores, athletes | yes | CPU | descriptive | heterogeneity |
| H1 | `lab.biomech` | N | bundle, target | yes | CPU | deterministic features; `nfl__*` columns excluded from predictors | tables/biomechanics |
| I1 | `lab.interpret` | N | ladder frame, OOF | yes | CPU | permutation inside test folds; train-fold statistics | tables/interpretability, athlete_profiles |
| J/K | `lab.synthesis` | N | saved tables | yes | trivial | missing evidence = not_evaluated | synthesis, final report |
| 12A–12G studies | `M12_UNIFIED_COLAB_RESEARCH.ipynb` | **B** | research warehouse | **no** | CPU-heavy | — | out of scope here; registry row X1 = blocked |

## 4. Gaps and decisions

1. **Transfer predictions were not persisted.** Added `transfer_predictions` rather than recomputing metrics from aggregate rows.
2. **Inductive learning curves are not supported by the API.** G2 stays transductive and is labelled accordingly. It is not presented as data efficiency.
3. **The F1 TCN has no sequence embeddings.** It enters only the transfer ladder, as `has_F1_TCN` coverage, and never retrieval or geometry.
4. **`paired_bootstrap(by=...)` only accepts key columns.** Arbitrary subgroups are handled by `stats.paired_in_subgroups`, which filters athletes and keeps the candidate pools.
5. **The target-freeze provenance (12C) is not in the bundle.** The audit documents it and does not verify it.
6. **Science seed vs lab seed.** CFG `seed = 42` would change the science hash and prevent checkpoint reuse, so it is used as the lab seed only. `science_seed` is a separate, explicit key.
7. **Earlier experiments are read-only.** Compatible checkpoints are adopted by copying them in, with the bundle and science hashes checked.
8. **The unified-notebook names `M12_UNIFIED_COLAB_RESEARCH_CLEAN_GPU.ipynb` and `M12_NEXT_STEP_INDUCTIVE_GPU_COLAB.ipynb` were not found in the repository** (only `M12_UNIFIED_COLAB_RESEARCH.ipynb` and `00–05` exist). Their Drive outputs are discovered generically under `experiments/*`, `unified_research/*/checkpoints/experiments/*` and `experimental_lab/*`.
