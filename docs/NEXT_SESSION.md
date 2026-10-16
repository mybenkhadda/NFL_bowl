# Next session — nflbowl package + unified Milestone 12 notebook (start here)

**Rules:** code authoring only; the user runs everything. Read this file, then only the module you need to fix.

**State (2026-10-16, end of session 4):**
- Package 0.3.0. New: `src/nflbowl/research/` and `notebooks/nflbowl/M12_UNIFIED_COLAB_RESEARCH.ipynb`, the one Colab notebook for 12.0 and 12A–12G.
- Docs: `docs/nflbowl/M12_MERGE_MATRIX.md` (what was merged and why) and `docs/nflbowl/M12_COLAB_EXECUTION_GUIDE.md` (how to run it, the CFG keys, the Drive layout).
- Session 3's fp16 fix (`se._fp32`) is unchanged. The notebook checks it statically (`colab.fixed_losses_check`) and defaults to fp32.
- **Never executed:** anything from sessions 1–4. Details: `docs/IMPLEMENTATION_STATE.md` § Session 4 and § Known limitations.

**Map** (all under `src/nflbowl/research/`):
- `config.py`: `CFG` → `UnifiedConfig`.
- `artifacts.py`: `ResearchRun`, the RUN_DIR writers.
- `data_access.py`: input gates (`REQUIREMENTS`), bundle checks, the stage plan.
- `warehouse.py`: `WarehouseContext`, `MirroredResearchStore`.
- `stages/base.py`: gate, reuse, `@step`, finish.
- `stages/s12_0 | s12c | s12b | s12a | s12d | s12e | s12g.py`: track runners ported from the notebooks. `s12e.py` is re-authored on the module API.
- `encoder.py`: 12F through `nflbowl.Experiment`.
- `eda.py`, `evaluation.py` (Part 11), `reporting.py` (reports, decision matrix).
- Wrapped research code: `nflbdb/research/advanced_movement/m12/*`, unchanged.

**Waiting on the user (Colab):**
1. Part 1.1 install → restart → Part 1.2 must print `nflbowl 0.3.0` and the fix check `True`.
2. First pass: `run_mode = 'debug'` with the bundle only (warehouse stages skip themselves). Then `fast` with `research_root` on Drive.
3. Send back the failing cell, `logs/<stage>_traceback.txt`, `manifests/stages.json` and `diagnostics/stage_plan.csv`.
4. Locally: `pytest tests/nflbowl -q`.

**Likely next failures:**
- Polars API details in the new glue (`map_elements` in `ResearchRun.save_table`, `replace_strict` in the EDA, `diagonal_relaxed` concats of empty frames).
- Matplotlib `get_layout_engine().set(rect=…)` on old versions.
- Mismatched column names in the 12E orchestration (`mt.own_distance_ci` output, `m12_combine_quality.attempt_order`).
- `Phase01Config.load(project_root=research_root)` with a `configs/phase01.toml` that points elsewhere.
- Drive FUSE renames inside `atomic_write`.

**Fix strategy:** a failing stage is isolated (`continue_on_error`). Fix the module, reinstall from `@main`, and rerun with the same `experiment_name`. Finished stages are reused.
