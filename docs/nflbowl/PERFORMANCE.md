# Performance tuning

No speed was measured: nothing was executed while writing the package. The statements below come from code inspection.

## What is faster by construction (default, results unchanged)

1. **Data residency.** The sequence store (≈ N × 64 × 6 × 4 bytes; ≈ 0.3 GB at 200k sequences) is copied to the GPU once per session and shared by every model trained in it. The notebook read a memory-mapped file and copied host→device on every batch.
2. **Staging.** The bundle is read once from Drive and then copied to local disk.
3. **Cached derivations.** Physics targets, state and category ids, probe scalars and F0 features are computed once per bundle (Python loops over every sequence) and cached as Parquet.
4. **Embedding.** GPU-resident slices, one device buffer, one host copy.
5. **Evaluation.** Athlete grouping replaces per-athlete filters (cross-domain retrieval). The identity null re-indexes one similarity matrix instead of recomputing retrieval per permutation. Matched retrieval is chunked, which bounds memory. Probes reuse per-fold standardisation.
6. **No retraining.** Finished runs are restored, embeddings and evaluations are reused by dependency hash, and phases are independent.

## Opt-in knobs (may change numbers; always recorded)

| Knob | When | Effect on results |
|---|---|---|
| `runtime.precision: bf16` | A100 / L4 / H100 | different rounding than the notebook's fp16 |
| `runtime.compile: true` | long runs, PyTorch ≥ 2.2 | fused kernels can change rounding; first epoch slower |
| `runtime.cudnn_benchmark: true` | fixed shapes | algorithm choice can be nondeterministic |
| `runtime.embed_batch` larger | always on GPU | kernel choice may change the last bits |
| `evaluation.approx_topk: true` | huge NFL samples | differs only on ties at rank k |
| `evaluation.similarity_device: auto` | matched retrieval on GPU | TF32 / reduction-order differences |
| `science.batch`, `accum_steps` | memory pressure | **science change** (new hash) |

## Known remaining costs

- `se.batch_loss` converts every loss part with `float(...)`. That is up to 5 GPU synchronisations per batch. It belongs to the research definition and is left as is.
- `_span_mask` / `_crop` draw random numbers on the CPU generator and move them to the GPU. Moving them would change the random streams.
- The `matched_retrieval` argsort is O(n log n) per query row. `approx_topk` makes it O(n).
- Training remains sequential per model. Parallel training of several small encoders on one GPU is not implemented.

## Memory

- GPU: resident data at about 0.3 GB, plus the model and activations at batch 256 × 64 frames, which is small. `gpu_memory_fraction` caps residency.
- Host: matched retrieval chunk = `matched_query_chunk × n_NFL × 12` bytes (similarity + argsort); lower the chunk if RAM is tight.
