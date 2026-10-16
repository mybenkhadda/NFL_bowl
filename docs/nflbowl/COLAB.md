# Google Colab GPU quickstart

Notebook: `notebooks/nflbowl/00_colab_quickstart.ipynb`. Upload it via *File → Upload notebook*, or open it from GitHub.

1. *Runtime → Change runtime type → GPU*.
2. Copy the data bundle to `MyDrive/NFLBowl/bundles/m12f_v1/` (see INSTALL.md § 2).
3. Run the cells:

```python
%pip install -q "git+https://github.com/YOUR_USERNAME/YOUR_REPO.git"
from nflbowl import Experiment
Experiment.device_info()
from google.colab import drive; drive.mount("/content/drive")
exp = Experiment(data_dir="/content/drive/MyDrive/NFLBowl/bundles/m12f_v1",
                 output_dir="/content/drive/MyDrive/NFLBowl", config="colab_gpu", device="auto", name="f4_main")
exp.prepare()
exp.train("F4", resume=True)
exp.evaluate("F4", level="standard")
exp.export()
```

## Storage layout on Colab

- **Bundle**: read from Drive once, then copied to `/content/nflbowl_stage/bundle_<fingerprint>` (`runtime.stage_dir` in the `colab_gpu` preset). Training reads local disk or GPU memory, never Drive.
- **Artifacts**: checkpoints, embeddings, tables, figures and reports are written directly to `output_dir` on Drive, so they survive a runtime reset. Every write goes through a temporary file and an atomic rename, so a disconnect never leaves a half-written file under its final name.

## Reconnect after a runtime reset

```python
%pip install -q "git+https://github.com/YOUR_USERNAME/YOUR_REPO.git"
from google.colab import drive; drive.mount("/content/drive")
from nflbowl import Experiment
exp = Experiment.open("/content/drive/MyDrive/NFLBowl/experiments/f4_main",
                      runtime={"stage_dir": "/content/nflbowl_stage"})
exp.prepare()          # re-stage (local disk was wiped); derived tables come from Drive
exp.resume()           # continue every training job left "running"
exp.embed("F4", reuse=True); exp.evaluate("F4", reuse=True)
```

Re-running the original `Experiment(..., name="f4_main")` cell also reopens the experiment, provided the science configuration is the same. A different science configuration raises an error instead of mixing results.

## Colab-specific notes

- A T4 has no bf16: keep `precision: auto` (fp16 autocast + GradScaler, the notebook's numerics).
- Session limits: training checkpoints after every epoch (`checkpoint_every: 1`). In the worst case a disconnect loses one epoch.
- The `max_minutes` wall-clock cap (90 min per encoder) counts from the start of the current session. A capped run is flagged `time_capped` in its signature.
