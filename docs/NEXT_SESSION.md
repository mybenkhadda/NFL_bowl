# Next session — nflbowl package (start here)

**Rules:** code authoring only; the user runs everything. Read this file, then only the module you need to fix.

**State (2026-10-16, end of session 2):**
- The `nflbowl` package and the new **bundle builder** are authored but **never executed**.
- Details: `docs/IMPLEMENTATION_STATE.md`.
- Bundle docs: `docs/nflbowl/BUNDLE_BUILD_GUIDE.md` and `docs/nflbowl/BUNDLE_TROUBLESHOOTING.md`.

**Map** (all under `src/nflbowl/`):
- `api.py`: `Experiment`.
- `config.py` + `configs/*.yaml`.
- `data/loading.py`: bundle format v1, `open_bundle`, `summary()`, missing-bundle hints.
- `data/discovery.py`: finds the 12F store and its companion files from warehouse files; metadata only.
- `data/builder.py`: checks → byte copy → `bundle.json` via `export._finish` → `open_bundle`; plus reports, zip / extract and the CLI `python -m nflbowl.data.builder`.
- `data/export.py`: in-session exporter and the shared `athletes_table`.
- `data/splits.py`, `training/`, `evaluation/`, `artifacts/`, `visualization/`.
- Research code it wraps: `nflbdb/research/advanced_movement/m12/shared_movement_encoder.py` (`se`) and `m11/representation.py` (`rep`).

**Local data facts:** the FULL store `warehouse/gold/advanced_movement/advanced_v1/sequence_shards/b69677c48cd9f6a9` (77,126 × 64 × 6) and every companion file exist, so the build should succeed.

**Waiting on the user:**
1. Output of `python -m nflbowl.data.builder --repo-root . --output bundles/m12f_v1 --verify --zip` (or `bundles/m12f_v1.reports/BUILD_REPORT.md`).
2. `pytest tests/nflbowl -q` output (new: `test_nflbowl_bundle_builder.py`).
3. Colab run of `00_colab_quickstart.ipynb` after uploading `m12f_v1.zip` to `MyDrive/NFLBowl/`.

**Likely first failures:**
- Builder: the `folds.signature_recomputed` warning (harmless); Polars `read_csv(n_rows=0)` / `collect_schema`; a Windows rename blocked by OneDrive.
- Package: Polars API details and PyTorch-version APIs (see IMPLEMENTATION_STATE).
