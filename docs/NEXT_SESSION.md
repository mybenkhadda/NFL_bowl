# Next session: M12 comprehensive experimental lab (start here)

**Rules:** code authoring only; the user runs everything. Read this file, then only the module you need to fix.

**State (2026-10-16, end of session 5):**
- Package 0.4.0. New: `src/nflbowl/lab/` and `notebooks/nflbowl/M12_COMPREHENSIVE_EXPERIMENTAL_LAB_COLAB.ipynb` (Parts A–K).
- API additions: `Experiment.derived`, `folds_by_athlete`, `splits(f)`, `transfer_predictions()`.
- Docs: `docs/nflbowl/M12_EXPERIMENTAL_LAB_API_AUDIT.md`, `M12_EXPERIMENT_MATRIX.md`, `M12_EXPERIMENTAL_LAB_EXECUTION_GUIDE.md`.
- **Never executed:** anything from sessions 1–5.

**Map** (`src/nflbowl/lab/`):
- `config`: CFG → LabConfig.
- `store`: LabRun, `run_stage`, `table`, `figure`, `data_file`.
- `registry`: SPECS and status.
- `environment`: setup, adoption, training plan.
- `audit` (B), `geometry` (C), `retrieval_lab` (D/G3–G5), `transfer_lab` (E), `ablations` (F), `robustness` (G1/G2), `biomech` (H), `interpret` (I), `synthesis` (J/K).
- `figures`, `stats`.

**Notebook stage ids:** SETUP, A2, P2, B1–B4, D2, C1/C3, C4, D1, D3/D4, D5, E2, E5/E7, E_perm, F1, G3, G4, G5, G1, G2, H1, I1, J1, K.

**Waiting on the user (Colab):**
1. Setup 2 must print `nflbowl 0.4.0`.
2. First pass: `run_mode='pilot'`, `run_training=False`.
3. Send back the failing cells, `logs/<stage>_traceback.txt`, `manifests/stages.json`, and the P2 adoption table, to confirm whether `f4_main_clean` shares the science hash.
4. Locally: `pytest tests/nflbowl/test_nflbowl_lab.py -q`.

**Likely next failures:**
- Polars details: `unpivot` / `pivot`, `replace_strict(default=)`, `rank('ordinal')` UInt arithmetic, `concat_list().list.eval`.
- Frames built from lists of dicts with missing keys.
- The `violinplot` orientation fallback.
- `ae._alpha` private-call signature in `interpret`.
- Ladder runtime when `transfer_predictions(models=[])`.

**Fix strategy:** each section is isolated by `run_stage`. Fix the module, reinstall from `@main`, restart, and rerun with the same `experiment_name`. Saved artifacts are reused.
