# NFL Big Data Bowl 2027 — research code + `nflbowl` GPU experiment API

One installable distribution (`nflbdb`, version 0.2.0) with two import packages:

- **`nflbdb`**: the research code (warehouse, Milestones 00–12). It is unchanged, and the existing notebooks keep importing it.
- **`nflbowl`**: a high-level, GPU-ready, resumable API for the Milestone 12F shared Combine–NFL movement encoders. It covers training, embeddings, evaluation, the NFL-transfer ladder, figures, reports and export.

```python
from nflbowl import Experiment

exp = Experiment(data_dir="/content/drive/MyDrive/NFLBowl/bundles/m12f_v1",
                 output_dir="/content/drive/MyDrive/NFLBowl", config="colab_gpu", name="f4_main")
exp.prepare()
exp.train("F4", resume=True)
exp.embed("F4")
exp.evaluate("F4", level="standard")
exp.plot(); exp.report(); exp.export()
```

## Install

```bash
# Colab / Kaggle (keeps the preinstalled PyTorch/CUDA build)
pip install "git+https://github.com/YOUR_USERNAME/YOUR_REPO.git"
# local development
pip install -e ".[dev]"            # add [torch] only if PyTorch is not installed yet
```

## Documentation

| Topic | File |
|---|---|
| Install, data bundle, local CPU/GPU | [docs/nflbowl/INSTALL.md](docs/nflbowl/INSTALL.md) |
| Google Colab GPU quickstart | [docs/nflbowl/COLAB.md](docs/nflbowl/COLAB.md) · [notebooks/nflbowl/00_colab_quickstart.ipynb](notebooks/nflbowl/00_colab_quickstart.ipynb) |
| Kaggle GPU quickstart | [docs/nflbowl/KAGGLE.md](docs/nflbowl/KAGGLE.md) |
| API reference | [docs/nflbowl/API.md](docs/nflbowl/API.md) |
| Configuration | [docs/nflbowl/CONFIG.md](docs/nflbowl/CONFIG.md) |
| Checkpoints and recovery | [docs/nflbowl/CHECKPOINTS.md](docs/nflbowl/CHECKPOINTS.md) |
| Performance tuning | [docs/nflbowl/PERFORMANCE.md](docs/nflbowl/PERFORMANCE.md) |
| Reproducibility and leakage prevention | [docs/nflbowl/REPRODUCIBILITY.md](docs/nflbowl/REPRODUCIBILITY.md) |
| Migration from the 12F notebook | [docs/nflbowl/MIGRATION.md](docs/nflbowl/MIGRATION.md) |
| Copy-paste examples | [docs/nflbowl/EXAMPLES.md](docs/nflbowl/EXAMPLES.md) |
| Architecture audit | [docs/nflbowl/ARCHITECTURE_AUDIT.md](docs/nflbowl/ARCHITECTURE_AUDIT.md) |

Raw competition data, warehouses, bundles, experiment outputs and checkpoints are never packaged or committed (`.gitignore`).
