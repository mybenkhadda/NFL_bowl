# Configuration

Three sections. Only `science` defines results, and its hash is part of every artifact and checkpoint signature.

## Presets (`src/nflbowl/configs/*.yaml`, installed with the package)

| Preset | Science | Runtime / evaluation |
|---|---|---|
| `default` | M12Config / 12F full protocol | auto device, `precision: auto`, exact evaluation, level `standard` |
| `colab_gpu` | **identical to default** (same hash) | stage to `/content/nflbowl_stage`, data resident on GPU, `embed_batch: 4096` |
| `research_full` | identical to default | evaluation level `full` by default |
| `fast_iteration` | 12F **smoke** budgets (2 epochs, batch 64, 4 train / 4 val batches, n_boot 60, 5-minute cap), `run_mode: fast_iteration` | level `quick`. **Not results** |

Custom YAML:

```yaml
extends: colab_gpu
science:
  seed: 7          # new science hash → new experiment
runtime:
  precision: bf16  # A100 / L4 only; changes numerics; recorded
```

`load_config("my.yaml")`, `Experiment(config="my.yaml")`, or overrides `Experiment(..., overrides={"science.epochs": 10})`.

## `science`

`run_mode, seed, embed_dim, width, epochs, batch (even, half Combine / half NFL), accum_steps, lr, patience, max_minutes, lambda_mask, lambda_phys, lambda_temp, lambda_align, val_share, outer_folds, n_boot, n_perm, max_train_batches, max_val_batches, inductive_configs, learning_curve_fractions`

Changing batch size or gradient accumulation changes the science (batch composition enters the supervised-contrastive and MMD losses); the hash changes accordingly.

## `runtime`

| Key | Default | Notes |
|---|---|---|
| `device` | `auto` | `cpu`, `cuda`, `cuda:1`, `mps`. An explicit GPU without one available raises. |
| `precision` | `auto` | CUDA: fp16 autocast + GradScaler (the notebook's numerics); CPU: fp32. `bf16` / `fp32` are opt-in. Part of the checkpoint signature. |
| `data_placement` | `auto` | `gpu` when the store fits in `gpu_memory_fraction` of free memory, else `pinned`, else `cpu`. Placement does not change any value. |
| `compile` | `false` | `nn.Module.compile` on the encoder, in place. Not always faster; first epoch slower. |
| `cudnn_benchmark`, `allow_tf32` | `null` | `null` leaves the PyTorch defaults untouched. |
| `embed_batch`, `embed_precision` | 512, fp32 | larger batches are faster; non-fp32 embedding is an approximation |
| `checkpoint_every` | 1 | the final epoch is always checkpointed |
| `stage_dir` | null | local copy of the bundle |
| `num_threads`, `progress` | null, true | |

## `evaluation`

| Key | Default | Notes |
|---|---|---|
| `backend` | `fast` | `reference` calls the nflbdb functions unchanged (slow, definitional) |
| `default_level` | `standard` | |
| `identity_null_perms` | 10 | 12F notebook value |
| `quick_athlete_share`, `quick_min_athletes`, `quick_n_boot` | 0.25, 60, 50 | quick = diagnostic subsample |
| `matched_query_chunk` | 256 | memory bound; exact |
| `approx_topk` | false | argpartition top-k (differs only on ties) |
| `similarity_device` | `cpu` | `auto` = GPU similarities for matched retrieval (approximate) |
