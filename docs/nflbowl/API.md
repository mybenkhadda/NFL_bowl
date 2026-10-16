# API reference

## `nflbowl.Experiment`

| Call | What it does | Saves |
|---|---|---|
| `Experiment(data_dir, output_dir=".", device=None, config="default", seed=None, name=None, overrides=None)` | Creates `<output_dir>/experiments/<name or experiment_NNN>`. An existing name is **reopened**; its science hash must match. | `config/experiment.json`, `config/runtime_history.jsonl` |
| `Experiment.open(path, data_dir=None, device=None, runtime=None, evaluation=None)` | Reopens with the experiment's own science config. The other arguments override session settings only. | — |
| `Experiment.device_info()` | Python, PyTorch, CUDA, GPU and memory facts; prints a table. | — |
| `prepare(verify=False, reuse=True)` | Opens and stages the bundle, validates schemas, binds the experiment to the bundle fingerprint, and builds the derived targets and F0 features. | `data/processed/*`, `data/features/*` |
| `features()` | F0 engineered features (N × 17). | cached |
| `list_models()` | Registry, plus `trained / embedded / evaluated` flags. | — |
| `train(model, regime="transductive", resume=True, nfl_fraction=1.0, folds=None)` | Trains, resumes or restores one encoder. `regime="inductive"` trains one encoder per outer fold. `nfl_fraction<1` is a learning-curve run. `resume=False` refuses to touch existing checkpoints. | epoch shards, `checkpoints/<model>/<tag>/best.pt`, `curve.parquet`, `evaluation/training_curves.csv` |
| `train_all(families=True, ablations=False, regime="transductive", resume=True, models=None)` | Sequential training. Ablations are opt-in. Inductive mode defaults to `science.inductive_configs`. | as above |
| `resume()` | Continues every job left `running`. | as above |
| `embed(model, regime="transductive", reuse=True, nfl_fraction=1.0, folds=None)` | Embeddings from `best.pt` (or a baseline). Returns `ndarray`, or `{fold: ndarray}` for inductive. | `embeddings/<model>__<tag>__<hash>.parquet` + `.json` |
| `load_embeddings(model, regime="transductive", fold=None, nfl_fraction=1.0)` | Loads the saved table without computing anything. | — |
| `evaluate(model, level=None, regime="transductive", reuse=True, nfl_fraction=1.0, folds=None)` | `quick` / `standard` / `full` (transductive) or held-out retrieval per fold (inductive). Returns polars rows. | `evaluation/records/*.json`, `metrics.csv`, `probes.csv`, `identity_null.csv`, `inductive_retrieval.csv`, `retrieval_results.parquet` |
| `transfer(models=None, reuse=True)` | Inductive NFL-transfer ladder on the 12C target (M1, M1+F0, M1+F1 TCN if bundled, M1+encoders). | `evaluation/transfer.csv` + record |
| `compare_inductive(models=("F0","F2","F4"), reference="F0", bootstrap=None, reuse=True, folds=None, stratify="fold", min_subgroup=10, adequate_n=30, ci=0.95, on_missing="skip", diagnostics=True, save=True)` | Paired, athlete-level held-out retrieval comparison (inductive only), with position subgroups, domain diagnostics and positive-pair coverage. `bootstrap=None` uses `science.n_boot`. Returns an `InductiveComparison` (`comparison`, `subgroups`, `scores`, `domain`, `coverage`, `meta`, `paths`, `summary()`). See [Paired inductive comparison](#paired-inductive-comparison). | `evaluation/records/inductive_comparison__<hash>.json` + `__scores.parquet` (immutable), `evaluation/paired_inductive_comparison.csv`, `athlete_retrieval_scores.parquet`, `position_subgroup_results.csv`, `domain_diagnostics.csv`, `inductive_coverage.csv`, `figures/{inductive_model_comparison,inductive_delta_auc,domain_separation}.png` (also under `figures/{png,svg,pdf}/` + `figures/source/`), `reports/INDUCTIVE_RETRIEVAL_REVIEW.md` |
| `load_inductive_comparison(models=None, reference=None)` | Latest saved comparison for the current bundle and science config. Computes nothing. | — |
| `metrics(include_quick=True)` · `compare(models=None, level=None)` | Transductive metric tables, one level at a time. | — |
| `Experiment.compare_experiments(paths, level="standard")` | Stacks the metrics of several experiments. | — |
| `plot(kind="all", save=True, formats=("png","svg","pdf"))` | Builds figures from the saved tables. Returns `{name: Figure}`. | `figures/{png,svg,pdf}/`, `figures/source/*.csv`, `figures/index.csv` |
| `report()` | `reports/REPORT.md` from saved tables; quick results go in a separate, labelled section. | report |
| `export(dest=None, include_embeddings=True, include_checkpoints=False)` | Zip archive. | `exports/*.zip`, `manifests/artifacts.csv` |
| `list_artifacts(kind=None)` | Manifest table. | — |

Model names: research names (`F4_physics_contrastive`, …) or aliases `F0 F1 F2 F3 F4 F5 F6 random`. Representation names in tables follow the notebook (`F4_physics_contrastive@nfl10%` for learning-curve runs; `random_frozen_F4_architecture`).

Unsupported, and it fails clearly: training `F0` / `random` / `F1_TCN`; transductive embeddings of `F1_TCN` (it is athlete-level and external); multi-GPU; a transfer ladder without `athletes.parquet` + target.

## Paired inductive comparison

`compare_inductive` answers one question: on athletes that no encoder ever trained on, does F2 or F4 match an athlete's Combine
movement to their own NFL movement more often than F0 does?

| Element | Definition |
|---|---|
| Population | For outer fold *f*: the athletes of fold *f*. They are held out of encoder *f* in both domains, and none of them appears among its training or validation rows (checked by `assert_fold_isolation`). |
| Representation | F0: fixed per-sequence statistics with nothing fitted. F2 / F4: the fold-*f* inductive checkpoint only. A transductive or other-fold checkpoint is refused: the training record's regime, tag and fold are checked, and so are the `split_role` values in the embedding table. |
| Query | The athlete's mean Combine embedding (all drill kinds). |
| Candidate pool | NFL mean embeddings of the *other* held-out athletes of fold *f* with the same `position_group` and both domains. Pools depend only on metadata and the split, so every representation is ranked against identical pools. Each row stores `pool_hash` and `candidate_ids`, and a mismatch raises `CandidatePoolMismatch`. |
| Athlete AUC | Share of candidates strictly below the own match. 0.5 = chance, **higher is better**. The mean over athletes equals `cross_domain_retrieval` on the same rows. Also `rank`, `top1`, `top5`. |
| Exclusions | `missing_combine`, `missing_nfl`, `position_pool_too_small` (< 2 candidates), `nonfinite_embedding`, and athletes not scored by *every* representation (complete cases, `n_excluded_incomplete`). Nothing is imputed. |
| Paired Δ | `mean AUC(model) − mean AUC(reference)` over the same athletes. |
| What is resampled | Athletes (query units), with replacement, **within each outer fold** (fold sizes kept). The same index draws are used for every representation, so each replicate is a paired difference. Candidate pools and per-athlete AUCs stay fixed, so the interval is conditional on the realised pools. |
| Interval | Percentile interval (`ci`, default 95 %) of the `bootstrap` replicate Δs. `delta_se` is the bootstrap SD. `p_boot_delta_le_0` is the share of replicates with Δ ≤ 0 (descriptive, not a p-value). |
| Subgroups | The same analysis inside each `position_group`. Below `min_subgroup` athletes there is no interval (`insufficient`); below `adequate_n` the row is `exploratory`. There is no multiplicity correction, so treat subgroups as descriptive. |
| Leakage | No NFL outcome (`athletes.parquet`) is read. Retrieval has no fitted or tuned parameter. Domain diagnostics are computed on held-out rows, with athlete-grouped inner CV and standardisation fitted on inner-training rows. |

`domain_diagnostics.csv` is a long table (`representation · fold · position_group · domain · metric · value · n`) with these metrics:
- `domain_auc` (overall and per position)
- `mean_gap_norm`, `mean_cosine`, `standardized_mean_gap_rms`, `norm_ratio_nfl_over_combine`
- `own_cosine_*`, `other_same_position_cosine_*`, `alignment_gap_{raw,centered}`
- per-domain `norm_*`, `dim_sd_mean`, `nonfinite_share`
- `effective_rank`, `participation_ratio`, `pc1_share`, `n_dead_dims`, `mean_pairwise_cosine`
- `positive_pair_coverage_{state,category}` (rows with `representation = training_split`; the share of training sequences that have an other-domain training sequence with the same alignment label)

Caching: the result is keyed by the bundle, the science hash, the models, the per-fold embedding ids, the bootstrap settings and the seed. `reuse=True` with
unchanged inputs reloads the record and rewrites the views.

```python
from nflbowl.evaluation import athlete_retrieval_scores, paired_bootstrap, coverage_table, domain_diagnostics, \
    grouped_domain_auc, collapse_indicators, cross_domain_alignment, positive_pair_coverage
```

## Low-level modules

```python
from nflbowl.config import load_config                      # presets, YAML, overrides
from nflbowl.models import list_models, get_model, enc_config
from nflbowl.data.loading import open_bundle
from nflbowl.data.splits import fold_map, split_indices     # integrity-checked
from nflbowl.training import DeviceData, train_encoder, embed, load_best
from nflbowl.evaluation import evaluate_representation, cross_domain_retrieval, within_domain_retrieval, \
    matched_retrieval, probe_r2, domain_shift, shuffled_identity_null
from nflbowl.visualization import forest, heatmap, training_curves, null_histogram, embedding_map, lines
from nflbowl.artifacts import ExperimentStore, Manifest
```

The research functions themselves stay reachable through `nflbowl._backend.se()`, `.rep()` and `.ae()`, or directly from `nflbdb`.
