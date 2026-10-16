# Installation, data bundle, local CPU / GPU

## 1. Package

| Environment | Command | PyTorch |
|---|---|---|
| Colab, Kaggle | `pip install "git+https://github.com/YOUR_USERNAME/YOUR_REPO.git"` | preinstalled CUDA build, left untouched |
| Local, PyTorch already installed | `pip install -e ".[dev]"` | yours |
| Local CPU without PyTorch | `pip install -e ".[dev,torch]"` | default PyPI wheel |
| Local NVIDIA GPU | install the CUDA build from https://pytorch.org/get-started/locally/ first, then `pip install -e ".[dev]"` | yours |

Core dependencies: numpy, polars ≥ 1.10, pyarrow, psutil, pyyaml, matplotlib, duckdb (the last two are for `nflbdb`'s own layers). Python ≥ 3.10.

The extras are `torch`, `dev` (pytest, ipykernel) and `all`. **Never install the `torch` extra on Colab or Kaggle.** pip would leave a satisfying build alone, but a version pin elsewhere could replace the CUDA build.

Private repository on Colab: `pip install "git+https://<TOKEN>@github.com/YOUR_USERNAME/YOUR_REPO.git"`, with the token read from Colab *Secrets* (`from google.colab import userdata; userdata.get("GH_TOKEN")`). Never paste it into a cell.

Prerequisite: this project folder must be its own git repository pushed to GitHub (today it sits inside a home-directory repo).

## 2. Data bundle

A GPU session needs only the bundle, never the warehouse:

```
bundle/
  bundle.json                    format version, sha256 + size of every file
  sequence_store/X.npy M.npy meta.parquet COMPLETE.json
  folds.parquet                  12.0 fold manifest
  athletes.parquet               optional: officials, linked flag, 12C target nfl.q90_shared
  tcn_representations.parquet    optional: Milestone 06 TCN-B fold representations
```

Build it on the research machine: run `notebooks/nflbowl/01_data_preparation.ipynb`, which calls `nflbowl.data.export.export_research_bundle(store, D, out_dir)`. Prerequisites: 12.0 executed and the 12F sequence store built in FULL mode (12F Part 3); 12C as well, if you want the transfer ladder. Upload the folder to Drive or a Kaggle dataset.

From other data: `nflbowl.data.export.write_bundle(out_dir, X, M, meta, folds, athletes=None)`, where `meta` row *i* must describe `X[i]`.

## 3. Local runs

```python
from nflbowl import Experiment
exp = Experiment(data_dir="bundles/m12f_v1", output_dir=".", config="default", device="auto", name="local_f4")
exp.prepare(); exp.train("F4"); exp.evaluate("F4")
```

CPU: precision is fp32, and the data stays in host memory. On CPU, prefer `config="fast_iteration"` for wiring checks; its numbers are not results.

## 4. Tests (written, not executed by the author)

```bash
pytest tests/nflbowl -q          # torch-dependent tests are skipped without PyTorch
```
