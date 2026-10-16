# Milestone 12 — Reproducibility

**Status:** authored 2026-10-15 · execution pending.

## Environment and controls
Windows Conda / Jupyter; Polars (lazy scans for `m11_envelope_hist`), NumPy; PyTorch only for 12F; POT and PyMC optional. Every notebook starts with `RUN_MODE`, `SEED = 20261015`, `FORCE_REBUILD`, `SAVE_TABLES`, `SAVE_FIGURES`, `NUM_WORKERS = 0` and a displayed memory plan. Smoke outputs live under `…/_sample/advanced_v1` and `reports/milestone12/_sample` and can never replace full outputs.

## Artifacts
- Tables: `reports/tables/advanced_movement/advanced_v1/*.csv`; datasets: `warehouse/gold/advanced_movement/advanced_v1/*.parquet` (atomic writes, unique temporary names).
- Figures: `reports/figures/advanced_movement/advanced_v1/AM12_*` (PNG 300 DPI + SVG, PDF for major figures, metadata sidecar with source and n).
- Manifests: `m12_foundation_manifest.json` (config, cohort, fold and grid signatures; shared speed range), `m12_dependency_manifest.json`, `m12c_target_freeze.json` (immutable), `m12d_selected_representation.json`, `m12_run_<track>.json`.
- Checkpoints: immutable shards keyed by configuration + cohort (`advanced_v1/checkpoints`; 12B synthetic grid, NFL window scan) and encoder epoch shards (`artifacts/models/advanced_movement/advanced_v1/m12`). A checkpoint of another definition is refused; incomplete runs never publish final inference tables.

## Software checks
`m12/validation.py` (synthetic, notebook-callable; `tests/test_milestone12_advanced.py` wraps them for pytest). They verify software behaviour only.

<!-- M12:RESULTS:START -->
_execution pending — written by 12G._
<!-- M12:RESULTS:END -->
