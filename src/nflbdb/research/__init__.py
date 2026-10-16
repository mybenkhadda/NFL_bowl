"""Phase 02 research layer: quality remediation, coverage marts, movement EDA, outcomes, feasibility.

Builds on ``nflbdb.data`` (config, LayerRefs, frames, reporting). Import-light on purpose:
call ``nflbdb.data.config.configure_runtime(cfg)`` before importing modules that import Polars.

Module map
    artifacts   versioned Parquet datasets + CSV/figure registry (manifest, reuse, archive)
    quality     data-quality remediation: failure triage, exact diagnostics, exclusion flags
"""
