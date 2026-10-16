# nflbowl — implementation state

_Updated 2026-10-16 (third session: fp16 overflow fix in the 12F contrastive loss). Authored by static analysis only: no script, test, notebook or training was executed._

## Existing architecture (unchanged)

- `src/nflbdb/` is the research package (warehouse, Milestones 00–12). The 12F notebook uses `m12/shared_movement_encoder.py` (`se`), `m11/representation.py` (`rep`), `m11/checkpoint.py`, `m12/advanced_evaluation.py` (`ae`) and `m12/config.py` (`M12Config`).
- Research artifact layout (`nflbdb.research.advanced_movement.artifacts.ResearchStore`): `warehouse/gold/advanced_movement/[_sample/]advanced_v1/`. It holds the sequence store `sequence_shards/<key>/`, `m12_fold_manifest.parquet` and the `m12*.json` records. The 11.0 data is in `discovery_v1/`, Milestone 06 in `transfer/`.
- Full audit: `docs/nflbowl/ARCHITECTURE_AUDIT.md`.

## Decisions

1. **Wrap, don't move.** `nflbdb` is untouched. The new package `src/nflbowl/` imports the scientific definitions (`nflbowl/_backend.py`).
2. **One distribution, two import packages** (`nflbdb`, `nflbowl`; version 0.2.1).
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

Session 3 (GPU mixed-precision fix, package 0.2.1):

- **Bug:** first Colab GPU run of `exp.train("F4")` crashed in `se._supcon` with `RuntimeError: value cannot be converted to type c10::Half without overflow`. Under fp16 autocast the projection output is fp16, so the similarity matrix was fp16 and `masked_fill(eye, -1e9)` cannot represent -1e9 (fp16 max ≈ 65504). Every config with `align ∈ {matched_state, category}` (F4, F6, most F4 ablations) failed on its first batch.
- **Fix (`m12/shared_movement_encoder.py`):** new `_fp32(t)` = autocast-disabled region on the tensor's device. Inside it, with `.float()` inputs: `_supcon` (whole loss), new `_infonce` (the temporal InfoNCE, previously inline in `batch_loss`), `_mmd`, and the JEPA cosine. The encoder / heads stay under autocast; fp16 autocast + GradScaler remain the default. In fp32 every term is the original formulation operation for operation (same -1e9 mask, positives, anchor rule, temperature, normalisation). Physics / masked-reconstruction MSE already promote to fp32 (fp16 head output − fp32 target); BCE-with-logits is fp32 under autocast. No objective, weight, architecture or config change.
- **Trainer (`nflbowl/training/trainer.py`):** `check_finite_loss` raises `FloatingPointError` naming the term(s), model, tag, epoch, batch and precision for a nonfinite train or validation loss (uses the already-synchronised `parts` floats: no extra sync). Nonfinite gradients with a finite loss stay the GradScaler's business.
- **Tests:** `tests/nflbowl/test_nflbowl_amp_losses.py` — fp32 parity with the pre-fix formulas, CPU bf16, CUDA fp16 / bf16 (explicit skips), GradScaler gradients, empty-positive / single-domain / invalid / repeated labels, temperature 0.01, batches 256 / 1024, full `batch_loss` for F4/F5/F6/F4_mmd/F4_adv/F4_category, and a test documenting the original overflow.
- **Notebooks:** `00_colab_quickstart` and `02_train_gpu`: force-reinstall from `@main`, version / fix check, GPU check, bundle candidates incl. `MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1`, `training_table = exp.train(...)`. No monkey-patching of `_supcon` needed.
- **Version:** 0.2.0 → 0.2.1 (`pyproject.toml`, `nflbowl.__version__`; recorded in `code_version` only, not in any hash). Science config and hashes unchanged.
- **Checkpoint compatibility:** checkpoint signatures do not include code, so existing epoch shards / `best.pt` are still found and load (state-dict keys unchanged). fp32 / CPU checkpoints: identical numerics. fp16 F4-type checkpoints cannot exist (the crash hit batch 0). Configs that did train under fp16 before (F5, F4_noalign, F4_mmd, F4_adv) resume with slightly different (more precise) loss numerics after the resume point; same objective. Use a new experiment `name` if a run must have uniform numerics. F2 / F3 are unaffected.

## Installation

- Colab / Kaggle: `%pip install -q "git+https://github.com/mybenkhadda/NFL_bowl.git@main"` then `%pip install -q --upgrade --force-reinstall --no-deps "git+https://github.com/mybenkhadda/NFL_bowl.git@main"` (restart the session if nflbowl was already imported). Torch is never installed.
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

- **Nothing executed** in any session: imports, tests, notebooks, CLI, GPU code. Session 3 specifically unverified: the fp16 F4 run end to end, whether other fp16 overflows exist inside the encoders (none found statically: pooling divides in fp32, the transformer uses PyTorch's own padding mask), the CUDA tests in `test_nflbowl_amp_losses.py`, and the `torch.autocast(..., enabled=False)` fallback on MPS.
- Builder-specific points that are unverified:
  - Whether the recomputed fold signature equals the stored one (it is only a warning).
  - Polars details: `read_csv(n_rows=0)`, `scan_parquet().collect_schema()`.
  - Windows directory renames while OneDrive or an indexer holds handles (retried 12×; on failure the bundle stays in `.m12f_v1.building/`).
- An existing bundle written by `export_research_bundle` has no `build_signature`, so the builder refuses to treat it as up to date; use `--overwrite`.
- The session-1 limitations remain (PyTorch / Polars API details, single GPU, `se.batch_loss` sync points).

## Pending work / exact next steps

1. Done by the user: bundle built and validated; Colab training starts (then hit the fp16 overflow fixed in session 3).
2. User (Colab): restart the runtime, reinstall from `@main` (Cell 1), check `nflbowl.__version__ == "0.2.1"`, rerun with the same `name="f4_main"` and `exp.train("F4", resume=True)`.
3. User: `pytest tests/nflbowl -q` locally, and `pytest tests/nflbowl/test_nflbowl_amp_losses.py -q` on the Colab GPU (the CUDA cases skip elsewhere).
4. Claude: fix the failures that come back.
