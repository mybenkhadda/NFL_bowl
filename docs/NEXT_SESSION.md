# Next session: M13 NFL outcome transfer (start here)

**Rules:** code authoring only; the user runs everything in Colab. Read this file, then only the module that failed.

**State (2026-10-17, end of session 6):**
- Package 0.5.0.
- New notebook `notebooks/nflbowl/M13_NFL_OUTCOME_TRANSFER_RESEARCH_COLAB.ipynb` (Parts A–K).
- New modules in `src/nflbowl/lab/`: `m13_config`, `handoff`, `target_audit`, `transfer_fairness`, `reliability`, `position_models`, `domain_matching`, `scouting_value`, `m13_reporting`, `m13_figures`.
- Tests: `tests/nflbowl/test_nflbowl_m13.py`.
- Docs: `docs/nflbowl/M13_RESEARCH_PLAN.md`, `M13_EXPERIMENT_MATRIX.md`, `M13_EXECUTION_GUIDE.md`.
- **Never executed:** anything from sessions 1–6.
- No change to `api.py` or the ladder. The POS / MEAN baselines are built on `Experiment._ladder_inputs` plus `ae.oof_ladder`.

**Map:**
- **A** — `handoff`: adopt M12 checkpoints and embeddings read-only, build the inventory, run F1 validity, reconcile.
- **B** — `target_audit`.
- **C/D** — `transfer_fairness`: long table → `validate_alignment` → `cohorts` (strict_common = claims) → `wide` / `paired_comparisons`.
- **E** — `reliability`: ICC, matched-k curves, train-fold selection blocks, interactions.
- **F** — `position_models`.
- **G** — `domain_matching` (CEM on frozen embeddings).
- **H/I** — `scouting_value`.
- **J/K** — `m13_reporting` (declared outcome rule), `m13_figures` (navy theme, HTML helpers).

**Waiting on the user (Colab):**
1. Setup 2 prints `nflbowl 0.5.0`.
2. Run all with the defaults (`run_training=False`).
3. Send back `manifests/stages.json`, every `logs/*_traceback.txt`, `tables/handoff/checkpoint_inventory.csv`, `tables/transfer/common_cohort_audit.csv` and the final report.
4. Locally: `pytest tests/nflbowl/test_nflbowl_m13.py -q`.

**Likely next failures:** see `docs/nflbowl/M13_EXECUTION_GUIDE.md` § 6 (polars pivot / over / list APIs, Styler `map`, private `ae._alpha` / `_fit_predict`, runtime of E1 / G1).

**Open scientific points to check once results exist:**
- Is the strict cohort large enough? Compare `COH_DESC` with the ladder rows.
- F1 validity verdict.
- Which estimator is frozen. Supply `freeze_json` to B1.
- Whether E2 curves are estimable (Combine attempts per drill).
