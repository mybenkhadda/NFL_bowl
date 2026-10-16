# nflbowl — implementation state

_Updated 2026-10-16 (second session: data-bundle builder). Authored by static analysis only: in this session no script, test, notebook or shell command was executed._

## Existing architecture (unchanged)

- `src/nflbdb/` is the research package (warehouse, Milestones 00–12). The 12F notebook uses `m12/shared_movement_encoder.py` (`se`), `m11/representation.py` (`rep`), `m11/checkpoint.py`, `m12/advanced_evaluation.py` (`ae`) and `m12/config.py` (`M12Config`).
- Research artifact layout (`nflbdb.research.advanced_movement.artifacts.ResearchStore`): `warehouse/gold/advanced_movement/[_sample/]advanced_v1/`. It holds the sequence store `sequence_shards/<key>/`, `m12_fold_manifest.parquet` and the `m12*.json` records. The 11.0 data is in `discovery_v1/`, Milestone 06 in `transfer/`.
- Full audit: `docs/nflbowl/ARCHITECTURE_AUDIT.md`.

## Decisions

1. **Wrap, don't move.** `nflbdb` is untouched. The new package `src/nflbowl/` imports the scientific definitions (`nflbowl/_backend.py`).
2. **One distribution, two import packages** (`nflbdb`, version 0.2.0).
3. **Portable data bundle** on GPU machines (format version 1, unchanged).
   - Session 2: it is built from the warehouse **files** by `nflbowl.data.builder`. This needs no nflbdb config stack and no `ResearchStore` object.
   - `export_research_bundle(store, D, out)` still works. Both writers share `export.athletes_table` and `export._finish`.
4. **Science / runtime / evaluation config split**; the science hash goes into every artifact.
5. **Trainer** = `se.train` step for step, plus GPU-resident data, explicit precision, RNG and scaler state, and opt-in compile.
6. **Fast evaluation** with the same estimands; the `reference` backend calls nflbdb unchanged.
7. **Artifacts**: immutable, dependency-hashed, manifest-based.
8. **Builder rules (session 2):**
   - Never recompute, never fabricate.
   - Choose a store explicitly when there are several FULL stores; never select smoke stores automatically.
   - Copy store files byte for byte and compare sha256 (source vs bundle).
   - Assemble in `.<name>.building`, then rename into place. Never delete an existing bundle: with `--overwrite` it is moved aside.
   - Reports go to `<name>.reports/`, outside the bundle.
   - The build signature (input sha256 + options) is stored in `bundle.json`; an identical rebuild is `up_to_date`.

## Files

Session 1 (package): see git history / `docs/nflbowl/ARCHITECTURE_AUDIT.md`. Created `README.md`, `src/nflbowl/**`, `notebooks/nflbowl/00–05`, `tests/nflbowl/*`, `docs/nflbowl/*`; modified `pyproject.toml` and `.gitignore`.

Session 2 (bundle builder):

- Created:
  - `src/nflbowl/data/discovery.py` (`discover_bundle_sources`, `BundleSources.summary()/rows()/to_dict()`, `npy_header`)
  - `src/nflbowl/data/builder.py` (`build_bundle_from_repo`, `validate_bundle`, `zip_bundle`, `extract_bundle_zip`, CLI `python -m nflbowl.data.builder`)
  - `tests/nflbowl/test_nflbowl_bundle_builder.py` (22 tests on a synthetic research layout)
  - `docs/nflbowl/BUNDLE_BUILD_GUIDE.md`
  - `docs/nflbowl/BUNDLE_TROUBLESHOOTING.md`
- Modified:
  - `src/nflbowl/data/export.py`: `athletes_table()` shared by both writers; `_finish` logs each file it hashes.
  - `src/nflbowl/data/loading.py`: actionable "bundle.json not found" hints (ZIP, missing directory, raw store, nested folder, incomplete upload); `SequenceBundle.verified`; richer `summary()` (athletes per domain, valid-frame lengths, fold coverage, transfer target, store key, run mode, total bytes, integrity). All old summary keys are kept.
  - `notebooks/nflbowl/00_colab_quickstart.ipynb`: repo URL `mybenkhadda/NFL_bowl`; new Cell 4 finds the bundle, extracts the ZIP and validates; reuse flags on embed/evaluate.
  - `notebooks/nflbowl/01_data_preparation.ipynb`: discovery → dry run → build → validate → zip.
  - `notebooks/nflbowl/02–04`: repo URL; `Experiment.open(..., data_dir=...)`.
  - `docs/nflbowl/INSTALL.md` § 2.

## Installation

- Colab / Kaggle: `pip install "git+https://github.com/mybenkhadda/NFL_bowl.git"`.
- Local: `pip install -e ".[dev]"`.
- Bundle: `python -m nflbowl.data.builder --repo-root . --output bundles/m12f_v1 --zip`.

## Local research artifacts (seen by static inspection, 2026-10-16)

- FULL store `warehouse/gold/advanced_movement/advanced_v1/sequence_shards/b69677c48cd9f6a9/`: all 4 files present; `COMPLETE.json` records n = 77,126, L = 64, the 6 channels and `sha_meta`. The 12F pointer `m12f_sequence_store.json` names it (run_mode full). No other store was found by glob.
- Present: `m12_fold_manifest.parquet` (columns `nfl_id, fold, fold_source, fold_signature, config_hash`), `m12_foundation_manifest.json` (fold_signature 6064fcbb0863bb23, config_hash c2903a12f8449fa2, run_mode full, n_linked 401), `m12_cohort_registry.parquet`, `m12c_targets.parquet`, `m12c_target_freeze.json` (config_hash c2903a12f8449fa2), `discovery_v1/m11_officials.parquet`, `transfer/transfer_attempt_representations_outer.parquet`.
- So every bundle prerequisite exists. Expected result: a bundle with all optional parts, ≈ 120 MB plus the TCN table.

## Scientific invariants

`docs/nflbowl/REPRODUCIBILITY.md` § 1.

The bundle builder never changes athlete ids, fold assignments (it copies `nfl_id, fold` exactly; athletes missing from the manifest keep the 12F `nfl_id % outer_folds` fallback at training time), sequence order, channels or values.

## Known limitations / unverified

- **Nothing executed** in either session: imports, tests, notebooks, CLI, GPU code.
- Builder-specific points that are unverified:
  - Whether the recomputed fold signature equals the stored one (it is only a warning).
  - Polars details: `read_csv(n_rows=0)`, `scan_parquet().collect_schema()`.
  - Windows directory renames while OneDrive or an indexer holds handles (retried 12×; on failure the bundle stays in `.m12f_v1.building/`).
- An existing bundle written by `export_research_bundle` has no `build_signature`, so the builder refuses to treat it as up to date; use `--overwrite`.
- The session-1 limitations remain (PyTorch / Polars API details, single GPU, `se.batch_loss` sync points).

## Pending work / exact next steps

1. User: `python -m nflbowl.data.builder --repo-root . --discover-only`, then `python -m nflbowl.data.builder --repo-root . --output bundles/m12f_v1 --verify --zip`. Send back the output, or `bundles/m12f_v1.reports/BUILD_REPORT.md` on failure.
2. User: `pytest tests/nflbowl -q`.
3. User: upload `bundles/m12f_v1.zip` to `MyDrive/NFLBowl/`, then run `00_colab_quickstart.ipynb`.
4. Claude: fix the failures that come back.
