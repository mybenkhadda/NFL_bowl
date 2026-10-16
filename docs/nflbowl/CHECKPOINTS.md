# Checkpoints and recovery

## Layout

```
experiments/<id>/checkpoints/
  enc_<model>_<tag>/<run_mode>_<signature hash>/     nflbdb ModelCheckpoint (immutable)
      config.json                                     full signature: EncConfig, training-index hash, epochs, batch, lr, seed,
                                                      nfl_fraction, numerics (device type, precision, patience, caps, val hash)
      shard_epoch__0000.pt + .pt.meta.json            model, optimizer, GradScaler, curve, best state, patience counter,
      …                                               signature, RNG streams (NumPy batch generator, torch CPU generator,
                                                      global torch CPU / CUDA, Python, NumPy global)
  <model>/<tag>/best.pt + best.json                   best weights + EncConfig + signature (for embedding)
  <model>/<tag>/curve.parquet                         training curve
```

Tags: `transductive`, `lc10` / `lc30` (learning curve), `inductive_f<k>`.

## Guarantees

- A shard is committed only once its sha256 sidecar is written. A shard without a valid sidecar is ignored and recomputed. Nothing is overwritten (Windows-safe publication with retries).
- A different signature, for example another precision or seed, gets a different directory. Checkpoints of another configuration are never loaded.
- `resume=True` restores a finished run without retraining, or continues after the last verified epoch with the same random streams. `resume=False` raises if checkpoints of the same configuration exist.
- Jobs are recorded in `manifests/manifest.json`. `exp.resume()` restarts everything still marked `running`.
- Embeddings depend on the sha256 of `best.pt`. Evaluations depend on the embedding artifact id. Changing an upstream input creates new files, so stale results are never reused.

## Recovery recipes

| Situation | Action |
|---|---|
| Colab disconnected mid-epoch | Reopen (`Experiment.open(...)` or the same `name=`), `exp.prepare()`, `exp.resume()`. At most one epoch is lost. |
| Drive sync left a `.tmp` file | Ignore it. Temporary names never collide with final names. |
| Want a fresh training run with the same settings | New experiment `name`. Checkpoints are intentionally immutable. |
| Moved the bundle | `Experiment.open(path, data_dir=new_location)`. The fingerprint check accepts it if the content is the same. |
| Notebook-era checkpoints (12F) | Not reused: their signature has no numerics block. Retrain in nflbowl, or embed with `nflbdb` directly. |
