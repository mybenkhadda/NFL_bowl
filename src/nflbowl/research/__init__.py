"""nflbowl.research — orchestration of the unified Milestone 12 research notebook.

``notebooks/nflbowl/M12_UNIFIED_COLAB_RESEARCH.ipynb`` drives the eight Milestone 12 tracks (12.0, 12A–12G) from one
configuration. This subpackage owns the infrastructure the eight notebooks used to repeat, and nothing scientific:

    config        one validated configuration (``resolve``) with run-mode defaults and per-stage switches
    artifacts     ``ResearchRun``: the Drive output tree, save_figure / save_table / save_markdown /
                  save_processed_dataset / save_model_metadata, an artifact manifest and stage status records
    colab         runtime facts (GPU, CUDA, memory), Drive mounting, local staging, the mixed-precision fix check
    data_access   bundle / warehouse discovery, validation tables and the per-stage input requirements (gates)
    warehouse     ``WarehouseContext`` (cfg, controls, store, M12Config, sources, data accessor; built once) and
                  ``MirroredResearchStore`` (a ``ResearchStore`` that also copies its outputs into the run directory)
    eda           the unified exploratory analysis (bundle + foundation)
    stages        the 12.0 / 12C / 12B / 12A / 12D / 12E / 12G stage runners (orchestration ported from the notebooks)
    encoder       the 12F section on ``nflbowl.Experiment``
    evaluation    Part 11: the comparison rows of every stage, classified (diagnostic / transductive / inductive / outcome-linked)
    reporting     per-stage Markdown + HTML reports, decision matrix, final report, artifact inventory

The algorithms (estimators, losses, splits, bootstrap, ladders) are imported from ``nflbdb.research.advanced_movement``
unchanged. Every module imports polars / matplotlib / torch lazily where possible.
"""

from __future__ import annotations

__all__ = ["artifacts", "colab", "config", "data_access", "eda", "encoder", "evaluation", "reporting", "stages", "warehouse"]
