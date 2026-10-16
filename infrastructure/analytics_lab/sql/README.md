# SQL

The dashboard marts themselves are defined in Python + SQL in
[`src/nflbdb/lab/marts.py`](../../../src/nflbdb/lab/marts.py) (one `build_*` method per mart, persisted
through `ArtifactStore` with manifest + schema sidecars). This folder holds **read-only** reference
queries for manual use:

| File | Where to run | Purpose |
|---|---|---|
| `serving_checks.sql` | Superset SQL Lab (database *NFL Lab (DuckDB)*) or `duckdb -readonly warehouse/serving/nfl_lab.duckdb` | Sanity checks and reconciliation against Phase 01 numbers |
| `bounded_frame_queries.sql` | DuckDB CLI / Python on the **host** (not Superset: frames are deliberately not exposed there) | Single-play / single-attempt frame pulls with partition + row-group pruning |
