# M15–M18 session handoff (start here in a fresh session)

**Updated:** 2026-10-20, session 2 (T01 + T04 implemented). Nothing has been executed by Claude.

## Rules (binding)

- **Code-authoring only.** Claude never runs Python, notebooks, tests, linters, formatters, shell commands or installs. The user runs everything in Colab.
- **Read-only sources.** M12 / M13 / M14 run directories and the exploratory M15 folders are read only. The code enforces this in `resolve()` and `Run.guard()`.
- **Context discipline.** Read this file, then only the module the current task touches. Never open executed `.ipynb` files.
- **Numbers.** Numbers come from saved tables only. Keep [OBS] / [EXP] / [HYP] / [PLAN] distinct. Gates default to `NOT_EVALUATED`.

## Objective

Do 10 Hz Combine movement measurements give reliable, independently predictive information about NFL movement beyond official testing? The official baseline is M1 = `draft_year, height, weight, forty, ten`. The official target is `nfl.q90_shared`. A rigorous negative answer is acceptable.

## Task status

| Task | Status | Entry point |
|---|---|---|
| T01 M15 config, sequence-store adapter, legacy provenance | **authored, not executed** | `src/nflbowl/lab/m15/{config,data,legacy,optional,plots}.py`, notebook sections B–G |
| T04 official M1 verification | **authored, not executed** | `src/nflbowl/lab/m16/{config,targets,m1_verification,plots}.py`, M16 notebook |
| Shared status vocabulary | authored | `src/nflbowl/lab/status.py` |
| T02, T03, T05–T17 | not started | `M15_M18_IMPLEMENTATION_PLAN.md` § 3 |

## Files created or changed in session 2

**New:**
- `src/nflbowl/lab/status.py`
- `src/nflbowl/lab/m15/__init__.py`, `config.py`, `optional.py`, `data.py`, `legacy.py`, `plots.py`
- `src/nflbowl/lab/m16/__init__.py`, `config.py`, `targets.py`, `m1_verification.py`, `plots.py`
- `tests/nflbowl/test_nflbowl_m15.py`, `tests/nflbowl/test_nflbowl_m16.py` (synthetic, unexecuted)
- `notebooks/nflbowl/M15_SPECTRAL_TEMPORAL_REPRESENTATION_LAB_COLAB.ipynb`
- `notebooks/nflbowl/M16_VERIFIED_BASELINES_CROSS_DOMAIN_VALIDATION_COLAB.ipynb`

**Changed:**
- `src/nflbowl/__init__.py` and `pyproject.toml`: version 0.7.0, plus a new optional extra `m15`.
- `src/nflbowl/lab/__init__.py`: docstring and `__all__`.

## Key APIs (as implemented)

### Status (`nflbowl.lab.status`)

- Step statuses: `READY`, `COMPLETED`, `CACHED`, `BLOCKED_MISSING_INPUT`, `BLOCKED_PROVENANCE`, `BLOCKED_INSUFFICIENT_SAMPLE`, `FAILED_VALIDATION`, `FAILED`, `NOT_RUN`.
- Gate statuses: `NOT_EVALUATED`, `PASS`, `FAIL`.
- `run_step(run, ledger, step, fn, requires=...)` wraps `LabRun.run_stage` and records the status in `manifests/<m1X>_status.json`.
- Exceptions: `ResearchBlocked(status, reason, missing)` and `ValidationFailed(reason, detail)`.
- `gate_row(...)` builds one gate row.

### M15 configuration (`nflbowl.lab.m15.config`)

- `resolve(CFG) -> M15Config`. It rejects:
  - unknown keys
  - read-only prefixes or overlapping directories
  - a single training opt-in
  - a scale ≠ `[10,5,5,3,0.1,1]`
  - bands above Nyquist
- `M15Config` provides `.fingerprint()` (science keys only), `.m13()` and `.selected_folds()`.
- `M15Run(root, cfg)` provides `.table`, `.guard`, `.snapshot_config`, `.write_status`, `.handoff_markdown` and `.ledger`.

### M15 data adapter (`nflbowl.lab.m15.data`)

- `prepare_m15_data(cfg, run, strict) -> M15Data(bundle, audit, folds, index, fold_of, fingerprint, game_key_status)`. `M15Data.rows(domain, folds, role, held_out)` selects rows.
- `audit_bundle` returns about 30 checks with status pass / warn / fail.
- Also: `fold_assignments`, `sequence_index` (vectorised segment rule), `counts_tables`, `game_key` (heuristic), `physical_rows`, `athlete_weighted_rows`.

### M15 legacy audit (`nflbowl.lab.m15.legacy`)

- `audit(dirs, bundle, outer_folds, split_paths, fold_of) -> (table, records)` assigns `verified_fold_safe`, `verified_exploratory_only` or `unknown_provenance`.
- `assert_fold_safe(record)` raises `BLOCKED_PROVENANCE` unless the record is `verified_fold_safe`.
- `adopt(record, run)` copies the artifact and writes a sidecar.
- `inspect_checkpoint(path)` loads on CPU with `weights_only=True`. A load failure is recorded, never raised.

### M16 (`nflbowl.lab.m16`)

- `config.resolve` rejects any training key, a changed official target and a tolerance above 1e-4.
- `targets.TARGETS` holds `{T0 nfl.q90_shared, T0x nfl_speed_q90_median}`. Also `target_frame`, `target_summary` (with fingerprint) and `assert_same_target`.
- `m1_verification.verify_m1(cfg, run, exp, ctx)` reuses:
  - `Experiment.transfer_predictions(models=[])`
  - `transfer_fairness.baseline_predictions`
  - `transfer_fairness.reproducibility_check`

  It persisted-compares against `evaluation/records/transfer_oof__*.parquet` in the M12 / M13 / M14 experiments. Verdicts are `REPRODUCED`, `REPRODUCED_INTERNAL_ONLY` or `MISMATCH`. `MISMATCH` raises `FAILED_VALIDATION` after writing the tables and the report.
- Also: `m1_contract`, `discover_persisted_oof`, `compare_predictions`, `fold_checks`, `target_alignment`, `gate_g16b`, `exploratory_target_audit`.

## Colab entry points

- **M15 notebook:** Setup 1–3 → E15.00 (bundle audit) → explorer → E15.00-L (legacy audit) → G15 rows (all `NOT_EVALUATED`) → hand-off.
- **M16 notebook:** Setup 1–4 (opens the experiment) → E16.A1 (`verify_m1`) → figures → E16.A0-x (T0x audit) → G16-B → hand-off.
- **Inputs from Drive:**
  - the bundle `…/bundles/m12f_v1_clean`
  - the M12 / M13 / M14 experiment folders (read only)
  - optional: `legacy_m15_dirs`, `exploratory_target_path`

## Pending manual validation (user, in Colab)

1. `pytest tests/nflbowl/test_nflbowl_m15.py tests/nflbowl/test_nflbowl_m16.py -q`.
2. Run the M15 notebook. Send back:
   - `manifests/stages.json`, `manifests/m15_status.json`
   - `tables/dataset/data_contract_audit.csv`, `trials_per_drill.csv`, `linked_cohort.csv`
   - every traceback
3. Run the M16 notebook. Send back:
   - `manifests/m1_verification.json`
   - `tables/baselines/m1_reproducibility.csv`, `m1_checks.csv`, `m1_persisted_records.csv`
   - `tables/synthesis/g16b_gate.csv`
   - `reports/markdown/m1_audit_report.md`
   - every traceback
4. Provide the exploratory M15 Drive folder(s) and the `nfl_speed_q90_median` export, if available.

## Known risks / assumptions (unverified by execution)

- `ae.oof_ladder` refits are assumed deterministic, so persisted and recomputed M1 should match bit-for-bit. A mismatch near 1e-12 points to BLAS/thread nondeterminism. Report it; do not relax the tolerance.
- Persisted records are found only when they keep the `transfer_oof__<hash>.parquet` + `.json` layout in `evaluation/records`. M13/M14 *lab tables* are not compared.
- Some masked frames inside NFL windows carry recorded values (the builder only maps NaN → 0). This is a `warn`, and every loss must use `M`.
- The game-key parse (10-digit `20YYMMDDgg` inside `seq_key`) is heuristic. If it fails, the status says so and season splits are used.
- Units of the M1 columns (inches, pounds, seconds) are assumed. `implausible_share` tests the assumption.
- Polars APIs assumed: `replace_strict`, `read_parquet_schema`, `write_csv(float_precision=…)`, `join(how="full", coalesce=True)`.
- **Spec discrepancy:** the brief suggested `notebooks/research/`. The planning specs (authoritative) place notebooks in `notebooks/nflbowl/`; the specs were followed.

## Next task IDs

- **T02:** M15 E15.A1–A2 signal analytics, `m15/signals.py`.
- **T03:** E15.A3–A4 spectral features and the resolution-artifact test, `m15/spectral.py`.
- **T06:** M16 NFL task targets T1–T6, reliability, MDE.
- **T05:** M17 event diagnosis (needs the M14 outputs).
- **T07–T08:** encoders and evaluation. They need a GPU session after T02.
