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
| `metrics(include_quick=True)` · `compare(models=None, level=None)` | Transductive metric tables, one level at a time. | — |
| `Experiment.compare_experiments(paths, level="standard")` | Stacks the metrics of several experiments. | — |
| `plot(kind="all", save=True, formats=("png","svg","pdf"))` | Builds figures from the saved tables. Returns `{name: Figure}`. | `figures/{png,svg,pdf}/`, `figures/source/*.csv`, `figures/index.csv` |
| `report()` | `reports/REPORT.md` from saved tables; quick results go in a separate, labelled section. | report |
| `export(dest=None, include_embeddings=True, include_checkpoints=False)` | Zip archive. | `exports/*.zip`, `manifests/artifacts.csv` |
| `list_artifacts(kind=None)` | Manifest table. | — |

Model names: research names (`F4_physics_contrastive`, …) or aliases `F0 F1 F2 F3 F4 F5 F6 random`. Representation names in tables follow the notebook (`F4_physics_contrastive@nfl10%` for learning-curve runs; `random_frozen_F4_architecture`).

Unsupported, and it fails clearly: training `F0` / `random` / `F1_TCN`; transductive embeddings of `F1_TCN` (it is athlete-level and external); multi-GPU; a transfer ladder without `athletes.parquet` + target.

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
