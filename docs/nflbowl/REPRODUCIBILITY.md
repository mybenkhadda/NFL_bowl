# Reproducibility and leakage prevention

## Scientific invariants (must never change silently)

1. **Definitions come from nflbdb.** Losses (`se.batch_loss`), encoders (`SharedEncoder`, `ConvEncoder`, `TransformerEncoder`), batches (`se._balanced_batches`), physics targets, state and category ids, splits (`se.split_indices`), evaluation estimands (`rep.*`) and the transfer ladder (`ae.*`) are imported, never re-implemented. The exception is the fast evaluation paths, which are parity-tested against the originals.
2. **Athlete grouping everywhere.** Validation is a seeded share of athletes. Inductive training removes the held-out fold's athletes from both domains. `check_split_integrity` raises on any athlete in two roles.
3. **Regimes are never mixed.** Transductive encoders (all sequences, no labels) feed label-free diagnostics only. `transfer()` reads inductive fold embeddings only. Reports keep separate tables.
4. **Training-only statistics.** Physics-target standardisation uses training sequences, and ladder PCA blocks are fitted on training rows.
5. **Candidate universe.** `standard` and `full` use every athlete and the same-position candidate sets of the reference. `quick` subsamples athletes and is flagged `diagnostic`, `comparable_to_full = False`, and reported separately.
6. **Domain separability is not a success criterion.** Combine drills and NFL play differ.

## Exactness of the fast backend

| Function | Equivalence |
|---|---|
| within-domain retrieval | identical comparisons; same query order |
| cross-domain retrieval | same rows per athlete mean (stable grouping), same comparisons, same bootstrap generator and draws |
| identity null | same permutation draws; the null AUC is computed on re-indexed columns of one similarity matrix |
| matched retrieval | per-row argsort on chunked rows (rows are independent); same random-baseline draws |
| probe R² | identical arithmetic (standardisation cached per fold) |
| domain shift | the reference function itself |

Cosine similarities are BLAS products. A different chunk shape, or a column permutation instead of a recomputation, can change a similarity in its last bit. A reported metric can change only when two similarities tie to the last bit. `evaluation.backend: reference` reproduces the nflbdb call path exactly.

## Training equivalence

`train_encoder` follows `se.train` step by step: seeding order, generators, batches, loss, accumulation rule, EMA, validation, early stopping, cap, best-state restore. The test `test_trainer_matches_se_train` asserts identical CPU weights. Deliberate differences:
- Precision is explicit and part of the signature (`auto` = the notebook's fp16 on CUDA).
- Resume restores all RNG streams. The notebook re-seeded its batch generator after an interruption, so the notebook's interrupted runs were not equal to uninterrupted ones; nflbowl's are.
- `max_val_batches` is a config key: 50 = notebook full, 4 = smoke.

## What is recorded

- `config/experiment.json`: resolved config, science hash, bundle fingerprint (sha256 of every bundle file), code versions (nflbowl, Python, NumPy, Polars, PyTorch, CUDA build, git commit when available).
- `config/runtime_history.jsonl`: every session's runtime settings, the device, and the effective cuDNN / TF32 flags.
- Checkpoint `config.json`: full training signature. Evaluation records: their dependency mapping.

## Nondeterminism that remains

CUDA convolution kernels can be nondeterministic, so bitwise-equal GPU runs are not promised (as in the notebook). The wall-clock cap makes training length depend on hardware; capped runs are flagged.
