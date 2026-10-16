# Kaggle GPU quickstart

1. Upload the data bundle as a private Kaggle **Dataset** (e.g. `nflbowl-m12f-v1`).
2. Create a notebook. Under *Settings*, set Accelerator to *GPU T4 x2* or *P100* and turn *Internet* on. Under *Add data*, attach the dataset.
3. Run:

```python
%pip install -q "git+https://github.com/YOUR_USERNAME/YOUR_REPO.git"
from nflbowl import Experiment
Experiment.device_info()

exp = Experiment(
    data_dir="/kaggle/input/nflbowl-m12f-v1",       # read-only input
    output_dir="/kaggle/working/NFLBowl",            # persisted as notebook output
    config="colab_gpu",                              # same science; runtime tuned for one GPU
    overrides={"runtime.stage_dir": None},           # /kaggle/input is already local disk
    name="f4_main",
)
exp.prepare()
exp.train("F4", resume=True)
exp.evaluate("F4", level="standard")
exp.export()
```

## Persistence between sessions

`/kaggle/working` is saved only when you *Save Version*. To continue in a later session, add the saved notebook output as an input and copy the experiment back into `/kaggle/working/NFLBowl/experiments/` before calling `Experiment.open(...)`. Its checkpoints are then resumed.

Only one GPU is used. With two T4s, nflbowl uses `cuda:0`; multi-GPU training is not implemented.
