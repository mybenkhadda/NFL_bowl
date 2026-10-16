# nflbowl — implementation state

_Updated 2026-10-16. Authored by static analysis only: no script, test, notebook or shell command was executed (one read-only shell listing slipped in at the very start of the session; nothing else)._

## Existing architecture (unchanged)

- `src/nflbdb/` is the research package (warehouse, Milestones 00–12). The 12F notebook uses `m12/shared_movement_encoder.py` (`se`), `m11/representation.py` (`rep`), `m11/checkpoint.py`, `m12/advanced_evaluation.py` (`ae`) and `m12/config.py` (`M12Config`).
- Full audit: `docs/nflbowl/ARCHITECTURE_AUDIT.md`.

## Decisions

1. **Wrap, don't move.** `nflbdb` is untouched. The new package `src/nflbowl/` imports the scientific definitions (`nflbowl/_backend.py`). Notebooks keep working.
2. **One distribution, two import packages.** The distribution name stays `nflbdb` (existing editable installs keep working); version is 0.2.0.
3. **Portable data bundle** instead of the warehouse on GPU machines: sequence store + fold manifest (+ athletes / 12C target, TCN-B), built locally by `export_research_bundle`.
4. **Science / runtime / evaluation config split.** The science defaults equal `M12Config`. The science hash goes into every artifact. `colab_gpu` has the same science hash as `default`.
5. **Trainer** = `se.train` step for step, plus GPU-resident data, explicit precision, RNG and scaler state in checkpoints, and opt-in compile.
6. **Fast evaluation** with the same estimands and draws; the `reference` backend calls nflbdb unchanged. Approximations are opt-in only.
7. **Artifacts**: immutable files named by dependency hash; manifest-based discovery; CSV views rebuilt from JSON records.
8. **Bug fix (packaging):** `.gitignore` `data/` → `/data/`. The unanchored pattern hid `src/nflbdb/data/` from git.

## Files

Created: `README.md`; `src/nflbowl/{__init__,api,config,devices,_backend,py.typed}`; `src/nflbowl/configs/{default,colab_gpu,research_full,fast_iteration}.yaml`; `src/nflbowl/data/{__init__,loading,validation,export,splits,preprocessing,cache}.py`; `src/nflbowl/features/{__init__,engineering}.py`; `src/nflbowl/models/{__init__,registry}.py`; `src/nflbowl/training/{__init__,trainer,checkpoints}.py`; `src/nflbowl/evaluation/{__init__,evaluator,retrieval,probes,null_tests}.py`; `src/nflbowl/artifacts/{__init__,store,manifest}.py`; `src/nflbowl/visualization/{__init__,plots}.py`; `src/nflbowl/utils/{__init__,logging,reproducibility}.py`; `notebooks/nflbowl/00–05*.ipynb`; `tests/nflbowl/{conftest,test_nflbowl_imports_config,test_nflbowl_data_artifacts,test_nflbowl_metric_parity,test_nflbowl_training_experiment}.py`; `docs/nflbowl/{ARCHITECTURE_AUDIT,INSTALL,COLAB,KAGGLE,API,CONFIG,CHECKPOINTS,PERFORMANCE,REPRODUCIBILITY,MIGRATION,EXAMPLES}.md`; this file; `docs/NEXT_SESSION.md`.

Modified: `pyproject.toml` (packages nflbdb + nflbowl, pyyaml, extras `torch` / `dev` / `all`, package data); `.gitignore`.

Not created on purpose: `LICENSE` (a legal choice for the owner); `models/architectures.py`, `models/encoders.py`, `training/callbacks.py` (the encoders live in nflbdb; no callback system was needed).

## Installation

Colab / Kaggle: `pip install "git+https://github.com/YOUR_USERNAME/YOUR_REPO.git"`. Local: `pip install -e ".[dev]"`. Details: `docs/nflbowl/INSTALL.md`.

## Scientific invariants

See `docs/nflbowl/REPRODUCIBILITY.md` § 1: definitions from nflbdb; athlete-grouped splits; transductive ≠ inductive; training-only statistics; full candidate universe outside `quick`; separability is not success.

## Known limitations / unverified

- **Nothing executed.** Unverified: every import, the tests, the notebooks, the GPU speed-ups, PyTorch-version-specific APIs (`torch.amp.GradScaler("cuda")`, `nn.Module.compile`, `torch.load(weights_only=False)`), and Polars API details (`partition_by(as_dict=True)` tuple keys, `replace_strict`, `unpivot`, `DataType.is_nested`).
- The palette validator of the dataviz method was not run (no execution); the documented, already-validated reference palette is used unchanged.
- The research parity claims rest on the parity tests, which have not been run.
- `se.batch_loss` sync points remain (5 `float()` calls per batch).
- Single-GPU only; sequential training.
- 12F checkpoints made by the notebook are not reused (different signature).
- 12G evidence and comparison rows are not produced by nflbowl.
- The project needs its own GitHub repository before the install command works.

## Pending work / exact next steps

1. User: create the GitHub repo for this folder (check `git status` shows `src/nflbdb/data/*.py` tracked after the `.gitignore` fix) and push.
2. User: `pytest tests/nflbowl -q` locally (conda env `nfl`; PyTorch needed for the training tests). Send back the failures.
3. User: run `notebooks/nflbowl/01_data_preparation.ipynb` locally (needs the 12F sequence store built in full mode) and upload the bundle.
4. User: run `00_colab_quickstart.ipynb` on Colab; send back the `device_info` output, the epoch timings and any traceback.
5. Claude: fix failures; optionally add multi-encoder scheduling or sync-free loss accounting after measurement.
