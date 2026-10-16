# Next session — nflbowl package (start here)

**Rules:** code authoring only; the user runs everything. Read this file, then only the module you need to fix.

**State (2026-10-16):** the `nflbowl` package is fully authored but **never executed**: no tests, notebooks or imports have run. Details in `docs/IMPLEMENTATION_STATE.md`.

**Map** (all under `src/nflbowl/`):
- `api.py`: `Experiment` (prepare / train / embed / evaluate / transfer / plot / report / export / resume / compare / list_*).
- `config.py` + `configs/*.yaml`: science / runtime / evaluation; science hash.
- `data/`: bundle format (`loading.py`), export from the research store (`export.py`), schema checks, folds and splits (`splits.py` → `se.split_indices` + integrity), cached derived arrays.
- `training/trainer.py`: `DeviceData` (GPU-resident store) and `train_encoder`, which mirrors `se.train`. `checkpoints.py` handles `best.pt` and the nflbdb `ModelCheckpoint` shards.
- `evaluation/`: fast exact retrieval / probes / null; `evaluator.py` levels; `backend="reference"` calls nflbdb.
- `artifacts/`: atomic writers, layout, dependency-hashed manifest.
- `visualization/plots.py`: Figure builders.
- Research code it wraps: `nflbdb/research/advanced_movement/m12/shared_movement_encoder.py` (`se`) and `m11/representation.py` (`rep`).

**Waiting on the user:**
1. `pytest tests/nflbowl -q` output.
2. GitHub repo created and pushed (the `.gitignore` fix makes `src/nflbdb/data` trackable).
3. Bundle exported via `notebooks/nflbowl/01_data_preparation.ipynb`.
4. Colab run of `00_colab_quickstart.ipynb`: `device_info`, epoch time, tracebacks.

**Likely first failures:** Polars API details (`partition_by` keys, `replace_strict`, joins on UInt32 `_r` vs Int64 `idx`: the same as the research code), PyTorch version APIs (`torch.amp.GradScaler`, `nn.Module.compile`), and the test tolerance in `test_trainer_matches_se_train` (exact CPU equality is asserted).
