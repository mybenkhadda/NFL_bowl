"""nflbowl — GPU-ready experiment API over the NFL Big Data Bowl 2027 research code (``nflbdb``).

Scope: the Milestone 12F shared Combine–NFL movement encoders (families F2–F6, ablations, baselines F0 engineered and
random frozen), their training, embedding extraction, label-free evaluation (retrieval, probes, domain shift,
identity null), the inductive NFL-transfer ladder, figures and reports — as independent, resumable, artifact-backed
phases.

    from nflbowl import Experiment
    exp = Experiment(data_dir="/content/data", output_dir="/content/drive/MyDrive/NFLBowl", config="colab_gpu", name="f4_main")
    exp.prepare(); exp.train("F4", resume=True); exp.embed("F4"); exp.evaluate("F4"); exp.report()

The scientific definitions (losses, batches, splits, metrics) are imported from ``nflbdb.research.advanced_movement``
and never redefined here; see ``docs/nflbowl/REPRODUCIBILITY.md``.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.2.1"
__all__ = ["Experiment", "__version__", "device_info", "load_config"]


def __getattr__(name: str) -> Any:              # lazy: `import nflbowl` must not import torch / polars
    if name == "Experiment":
        from .api import Experiment

        return Experiment
    if name == "device_info":
        from .devices import device_info

        return device_info
    if name == "load_config":
        from .config import load_config

        return load_config
    raise AttributeError(f"module 'nflbowl' has no attribute {name!r}")
