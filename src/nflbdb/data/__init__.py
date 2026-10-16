"""Phase 01 data layer: discovery, profiling, validation and a Bronze/Silver/Gold lakehouse.

Kept import-free on purpose: ``nflbdb.data.config.configure_runtime`` must run before Polars is
imported (POLARS_MAX_THREADS is read at import time). Import submodules explicitly:

    from nflbdb.data.config import Phase01Config, configure_runtime
    cfg = Phase01Config.load(); configure_runtime(cfg)
    from nflbdb.data.pipeline import Phase01Pipeline
    from nflbdb.data.access import ResearchDataAccess

Module map
    config        settings, DuckDB connection factory
    sources       raw source registry (schemas confirmed from CSV headers), FK hypotheses
    sampling      bounded raw-file access (row estimates, byte-offset samples)
    environment   Step 0  — 00_environment_report.md
    discover      Step 1  — 01_data_inventory.md, file_inventory.parquet
    schema        Step 2  — 02_schema_dictionary.md, schema_catalog.parquet
    bronze        lossless typed Parquet mirror + manifest
    profile       Step 3A — column profiles, Combine participation, drill coverage
    tracking_qc   Step 3A — tracking integrity (intervals, gaps, jumps, kinematic consistency)
    validate      Step 3B — 03_data_quality.md, data_quality_results.parquet
    relationships Step 4  — 04_relationship_analysis.md, linkage, join cardinality
    silver        validated / standardised tables
    gold          dims, facts, marts
    catalog       DuckDB views over all layers
    access        research access layer (lazy, pushdown)
    benchmark     Step 8  — 07_performance_benchmarks.md
    feasibility   Step 10 — 08_research_feasibility.md
    pipeline      orchestrator + CLI
"""
