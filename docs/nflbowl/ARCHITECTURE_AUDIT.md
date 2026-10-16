# Architecture audit — Milestone 12F encoder pipeline (2026-10-16)

Static review only; nothing was executed.

## 1. What the slow notebook actually runs

`notebooks/advanced_movement/12f_physics_informed_shared_movement_encoder.ipynb` drives two modules:

| Alias | Module | Role |
|---|---|---|
| `se` | `src/nflbdb/research/advanced_movement/m12/shared_movement_encoder.py` | `EncConfig`, `FAMILIES` (F2–F6), `ABLATIONS` (6), `PRIMARY`, `scale_config`, `physics_targets`, `state_ids`, `category_ids`, `split_indices`, `SharedEncoder`, `batch_loss`, `_balanced_batches`, `train`, `embed`, `random_frozen`, `evaluate`, `shuffled_identity_null`, `athlete_mean_embedding`, `embedding_block`, `tcn_by_fold` |
| `rep` | `src/nflbdb/research/advanced_movement/m11/representation.py` | sequence format and scaling, `SequenceStore`, `ConvEncoder`, `TransformerEncoder`, `_span_mask`, `_crop`, `engineered_features`, `physical_scalars`, the label-free evaluation (`within_domain_retrieval`, `cross_domain_retrieval`, `matched_retrieval`, `probe_r2`, `domain_shift`, `evaluation_row`), `save_embeddings` |
| — | `m11/checkpoint.py` | `ShardCheckpoint` / `ModelCheckpoint`: immutable, sha256-verified epoch shards |
| `ae` | `m12/advanced_evaluation.py` | `Model`, `M1_COLS`, `oof_ladder`, `compare` (NFL-transfer ladder) |
| — | `m12/config.py` | `M12Config` (all 12F budgets), `config_hash` |

Upstream data (only on the research machine): `ResearchStore` (12.0 outputs, `m12_fold_manifest`, `m12_cohort_registry`), `M12Data.officials()`, 09A episodes via `M11Sources`, NFL windows from one resumable tracking scan (`m11.nfl_physics.NflWindowExtractor`), 12C targets.

## 2. Reusable as is

- Scientific core: losses, encoders, batch construction, physics targets, splits, metrics, ladder. All are pure functions of their arguments. **nflbowl imports them unchanged** (`nflbowl/_backend.py`).
- `ModelCheckpoint`: already atomic and immutable (unique temp file, `os.replace` with retries, sidecar-committed shards, config-hash directories). Reused.
- `SequenceStore`: an immutable memory-mapped `.npy` store. It is the natural unit to ship to a GPU session.

## 3. Notebook-specific logic, now moved into the package

| Notebook logic | Now |
|---|---|
| `FOLD` construction with `nfl_id % outer_folds` fallback | `nflbowl.data.splits.fold_map` |
| P / STATE / CAT / DOM / PHYS derivation (Part 3) | `nflbowl.data.preprocessing.derive` (cached Parquet) |
| F0 + random baselines, transductive loop, learning curve, inductive loop | `Experiment.train / embed / evaluate` |
| NFL-transfer ladder (cell 12) | `Experiment.transfer` |
| Figures (`fg.*` + `store.show`) | `nflbowl.visualization.plots` + `ExperimentStore.save_figure` |

## 4. Hardcoded paths, hidden state

- Paths come from `Phase01Config` / `ResearchStore` (`cfg.paths.project_root / artifacts/models/...`, `reports/tables/...`). There are no absolute paths in the modules, but the stores assume the whole warehouse layout. **Colab cannot use them** → portable bundle (`nflbowl.data.loading`).
- Notebook globals: `X, M, meta, P, STATE, CAT, DOM, PHYS, FOLD, C, DEV, store`. These are now `Experiment` attributes, rebuilt or reloaded per session.
- `nflbdb.data.config.configure_runtime` must run before Polars is imported (thread count). nflbowl does not need it. The export notebook still calls it.
- **Critical packaging defect found:** `.gitignore` had an unanchored `data/` pattern, which also ignores the packages `src/nflbdb/data/` (and the new `src/nflbowl/data/`). A `pip install git+…` would ship without `nflbdb.data`, and every module would fail to import. Fixed: `/data/`.
- The project folder has no `.git` of its own; the enclosing repo is the home directory. A GitHub repo for this project must be created before the Colab install command works.

## 5. CPU / GPU issues and expensive operations

| Where | Problem | Change (science unchanged) |
|---|---|---|
| `se.train → to(ix)` | every batch fancy-indexes the **memory-mapped** store and copies host→device. On Drive-mounted files, every batch is random disk I/O. | `DeviceData`: the store is placed **once per session** on the GPU (≈ 0.3 GB for 200k × 64 × 6 float32), else in pinned memory with non-blocking copies. The bundle is staged to local disk first. |
| `se.train` | autocast is hard-wired to fp16 on CUDA; GradScaler state is not checkpointed; the NumPy batch generator restarts from the seed after an interruption | explicit precision (`auto` = same fp16), scaler state and all RNG streams in every shard, so resume continues the same streams |
| `se.batch_loss` | `float(v.detach())` per loss part forces up to 5 GPU syncs per batch | **kept** (redefining the loss is out of scope); documented in PERFORMANCE.md |
| `se.embed` | per-batch memmap read + `.cpu()` per batch | GPU-resident slices, one device buffer, one host copy, configurable batch |
| `rep.cross_domain_retrieval` | `meta.filter(nfl_id == i)` per athlete → O(athletes × sequences) | rows grouped once, same `E[rows].mean(0)` |
| `se.shuffled_identity_null` | recomputes the full retrieval (including all per-athlete filters) per permutation | one similarity matrix; each permutation re-indexes its columns (same draws) |
| `rep.matched_retrieval` | dense 2000 × ~200k similarity + full argsort → several GB | query chunks (rows sorted independently → same ranks); opt-in argpartition / GPU top-k |
| `rep.within_domain_retrieval` | Python loop per query | vectorised per drill family, same order |
| `rep.probe_r2` | standardisation + Gram matrix recomputed per target | computed once per (domain, fold) |
| `physics_targets`, `state_ids`, `physical_scalars`, `engineered_features` | Python loops over every sequence, recomputed every session | computed once per bundle, cached Parquet |

## 6. Leakage review

- Physics-target standardisation uses training-sequence constants only (kept).
- Inductive split removes held-out athletes of **both** domains before training. Validation = held-out **athletes** (kept; nflbowl adds `check_split_integrity`, which raises on any athlete in two roles).
- Engineered features use fixed physical scaling and no fitted statistics (kept).
- Probe CV and domain classifier are athlete-grouped (kept).
- Transfer ladder: fold-fitted PCA on training rows only (`se.embedding_block`, kept). Transductive embeddings never enter the ladder (enforced: `transfer()` reads inductive embeddings only).
- Possible concern (unchanged, flagged): `domain_shift` groups its CV by `nfl_id % 5`, not by the fold manifest. That is grouped, so no leakage, but not the manifest folds.

## 7. Refactoring risks

- Floating point: BLAS products in chunked or permuted form can differ in the last bit. A metric changes only on exact ties. Parity tests cover this on fixtures (`tests/nflbowl/test_nflbowl_metric_parity.py`).
- Precision `bf16`, `torch.compile`, larger embedding batches, `approx_topk`, and GPU similarities can change numbers. All are opt-in and recorded.
- Checkpoint directories of nflbowl differ from the notebook's (the signature adds `numerics`), so notebook checkpoints are not reused. That is deliberate: device type and precision change results.
- Polars API drift (`partition_by` keys, `replace_strict`, `unpivot`): same calls as the research code, which targets Polars ≥ 1.10.
