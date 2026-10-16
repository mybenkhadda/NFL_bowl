# Copy-paste examples

All examples assume:

```python
from nflbowl import Experiment
exp = Experiment(data_dir="/content/drive/MyDrive/NFLBowl/bundles/m12f_v1",
                 output_dir="/content/drive/MyDrive/NFLBowl", config="colab_gpu", name="f4_main")
exp.prepare()
```

## Engineered-features baseline (F0)

```python
E0 = exp.features()                       # (N, 17) — cached
exp.evaluate("F0", level="standard")
```

## Random frozen encoder

```python
exp.evaluate("random", level="standard")  # untrained F4 architecture, seeded
```

## Primary encoder

```python
exp.train("F4", resume=True)
exp.embed("F4")
exp.evaluate("F4", level="full")          # includes the shuffled-identity null
```

## Several models compared

```python
for m in ["F2", "F3", "F4", "F5", "F6"]:
    exp.train(m)
    exp.evaluate(m, level="standard")
exp.compare(level="standard")
```

## Ablation study

```python
exp.train_all(families=False, ablations=True)
for m in ["F4_nophys", "F4_noalign", "F4_adv", "F4_mmd", "F4_category", "F4_shuffled_phys"]:
    exp.evaluate(m, level="standard")
exp.compare(models=["F4", "F4_nophys", "F4_noalign", "F4_shuffled_phys"], level="standard")
```

## Cross-domain retrieval (inductive, held-out athletes)

```python
exp.train_all(regime="inductive")                  # F2 and F4 × folds
exp.evaluate("F0", regime="inductive")
exp.evaluate("F4", regime="inductive")
exp.transfer()                                     # NFL ladder on the 12C target
```

## Paired F0 / F2 / F4 comparison on held-out athletes (Colab)

```python
# Session 1 (GPU): one inductive encoder per outer fold; finished folds are restored, not retrained.
exp.train("F2", regime="inductive")
exp.train("F4", regime="inductive")

res = exp.compare_inductive(models=["F0", "F2", "F4"], reference="F0", bootstrap=2000, reuse=True)
res.summary()            # mean athlete AUC ± CI and paired Δ AUC vs F0 (positive = better than F0)
res.coverage             # held-out / scored / excluded athletes per model × fold
res.subgroups            # per position; 'insufficient' / 'exploratory' rows are not findings
res.paths                # evaluation/*.csv, figures/*.png, reports/INDUCTIVE_RETRIEVAL_REVIEW.md

# Session 2 (CPU is enough): reopen and read the saved result, nothing is recomputed.
from nflbowl import Experiment
exp = Experiment.open("/content/drive/MyDrive/NFLBowl/experiments/f4_main")
res = exp.load_inductive_comparison(models=["F2", "F4"], reference="F0")

from IPython.display import Image, Markdown
Image(str(exp.root / "figures" / "inductive_delta_auc.png"))
Markdown((exp.root / "reports" / "INDUCTIVE_RETRIEVAL_REVIEW.md").read_text())
```

Only some folds trained so far? `compare_inductive(on_missing="skip")` (the default) keeps going. Folds without an encoder show up in
`res.coverage` and `folds_unavailable`, and their athletes leave the paired set. Use `on_missing="raise"` to stop instead.

## Identity-null experiment

```python
row = exp.evaluate("F4", level="full")
row.select("cross_auc", "null_mean", "null_q95", "null_exceed_share")
import polars as pl; pl.read_csv(exp.root / "evaluation" / "identity_null.csv")
```

## Plot generation

```python
figs = exp.plot()                          # all kinds, saved as PNG/SVG/PDF + source CSV
figs = exp.plot("retrieval", save=False)   # just build
figs["retrieval_transductive"]
```

## Loading saved embeddings (new session, no GPU needed)

```python
exp = Experiment.open("/content/drive/MyDrive/NFLBowl/experiments/f4_main")
df = exp.load_embeddings("F4")             # ids, domain, kind, position, fold, split_role, e0…
```

## Comparing completed experiments

```python
Experiment.compare_experiments(["/content/drive/MyDrive/NFLBowl/experiments/f4_main",
                                "/content/drive/MyDrive/NFLBowl/experiments/f4_seed7"], level="full")
```

## Second seed as its own experiment

```python
exp7 = Experiment(data_dir=..., output_dir=..., config="colab_gpu", seed=7, name="f4_seed7")
```
