# Migration from the 12F notebook

The research notebooks and `nflbdb` are unchanged and keep working. nflbowl is a second path to the same science.

| 12F notebook | nflbowl |
|---|---|
| Part 3: build / open the sequence store, `P`, `STATE`, `CAT`, `DOM`, `PHYS`, `FOLD` | `export_research_bundle(store, D, out)` once (locally), then `exp.prepare()` |
| `E0 = rep.engineered_features(X, M)`; `se.evaluate('F0_engineered', …)` | `exp.evaluate("F0")` |
| `se.random_frozen(se.scale_config(se.by_name(se.PRIMARY), C), DEV, C.seed)` | `exp.evaluate("random")` |
| `for e in CONFIGS: se.train(...)` (transductive) | `exp.train_all(families=True, ablations=RUN_ABLATIONS)` |
| `rep.save_embeddings(...)` | `exp.embed(name)` (Parquet + sidecar, dependency-hashed) |
| `se.shuffled_identity_null(E, meta, 10, seed)` | `exp.evaluate(name, level="full")` |
| learning curve `nfl_fraction ∈ {0.1, 0.3}` | `exp.train("F4", nfl_fraction=0.1)` + `exp.evaluate("F4", nfl_fraction=0.1)` |
| inductive loop (`f'inductive_f{f}'`) + `rep.cross_domain_retrieval(E[ho], meta[ho], …)` | `exp.train_all(regime="inductive")`, `exp.evaluate(name, regime="inductive")` |
| transfer ladder (`ae.oof_ladder`, `ae.compare`) | `exp.transfer()` |
| `fg.fig_*` + `store.show` | `exp.plot()` (PNG / SVG / PDF + source CSV) |
| conclusion / manifest (`rp.*`) | `exp.report()`, `exp.export()` |

## Equivalence of settings

- `RUN_MODE='full'` corresponds to the `default` or `colab_gpu` preset. `RUN_MODE='smoke'` corresponds to `fast_iteration` (same budgets, separate run-mode label).
- `DEVICE=None` corresponds to `device="auto"`. The notebook's CUDA AMP corresponds to `precision: auto`.
- Table names: the notebook's `m12f_eval_transductive` becomes `evaluation/metrics.csv` (rows tagged `level`). `m12f_eval_inductive` becomes `evaluation/inductive_retrieval.csv`, `m12f_transfer` becomes `evaluation/transfer.csv`, and `m12f_training_curves` becomes `evaluation/training_curves.csv`. The column names of the evaluation rows are the same.

## Not migrated (stays notebook-only)

12G comparison and evidence rows (`sy.comparison_row`, `ae.evidence_row`), report-document updates (`rp.update_doc`), and the validation checks `v12.CHECKS_ENCODER`. They feed the milestone documents, not the experiment workflow. If you want nflbowl results in 12G, read `evaluation/*.csv`.
