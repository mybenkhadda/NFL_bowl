# Building the M12F data bundle

A GPU session (Colab, Kaggle, another machine) reads only a **data bundle**, never the warehouse. The bundle is built on the research machine from files that already exist; nothing is recomputed.

## 1. One command

From the repository root, in the environment where `nflbowl` is installed (`pip install -e .`):

```bash
python -m nflbowl.data.builder --repo-root . --output bundles/m12f_v1 --zip
```

The command:

1. discovers the sources (§3),
2. runs the compatibility checks (§5),
3. copies the files into `bundles/.m12f_v1.building/` and writes `bundle.json`,
4. renames that folder to `bundles/m12f_v1/`,
5. validates the result with `open_bundle`,
6. writes `bundles/m12f_v1.zip`, which is CRC-tested.

Exit codes: `0` ok, `2` missing sources, `3` incompatible sources, `4` output exists, `1` other error.

Useful variants:

| Command | Effect |
|---|---|
| `--discover-only` | list stores, inputs and missing prerequisites; writes nothing |
| `--dry-run` | run every check and print the copy plan; writes nothing |
| `--verify` | before the build, scan `X` for NaN/inf, non-zero padding and a non-binary heading flag; after it, re-hash every file |
| `--source-sequence-store PATH` | choose the store explicitly (required when several FULL stores exist) |
| `--fold-manifest PATH` | override the 12.0 fold manifest (Parquet, or the CSV view `reports/tables/advanced_movement/advanced_v1/m12_fold_manifest.csv`) |
| `--no-include-transfer` / `--no-include-tcn` | leave out the optional `athletes.parquet` or `tcn_representations.parquet` |
| `--overwrite` | rebuild an existing bundle; the old folder is moved to `m12f_v1.replaced-<timestamp>`, never deleted |
| `--no-reuse` | refuse an existing identical bundle instead of reporting it as up to date |
| `--validate bundles/m12f_v1 [--verify]` | only open the bundle and print its summary as JSON |
| `--zip-existing bundles/m12f_v1` | only write the ZIP |
| `--json`, `--no-progress`, `--report-dir PATH` | output options |

The same steps in Python:

```python
from nflbowl.data.discovery import discover_bundle_sources
from nflbowl.data.builder import build_bundle_from_repo
from nflbowl.data.loading import open_bundle

print(discover_bundle_sources(".").summary())
res = build_bundle_from_repo(repo_root=".", output_dir="bundles/m12f_v1", include_transfer=True, include_tcn=True,
                             reuse_existing=True, overwrite=False, make_zip=True)
print(res.text())
print(open_bundle("bundles/m12f_v1", verify=True).summary())
```

`notebooks/nflbowl/01_data_preparation.ipynb` runs the same calls cell by cell.

## 2. What is mandatory

| Bundle file | Source (FULL run mode) | Written by | Required |
|---|---|---|---|
| `sequence_store/X.npy`, `M.npy`, `meta.parquet`, `COMPLETE.json` | `warehouse/gold/advanced_movement/advanced_v1/sequence_shards/<key>/` (byte copy) | 12F Part 3 | **yes** |
| `folds.parquet` (`nfl_id`, `fold`) | `…/advanced_v1/m12_fold_manifest.parquet` | 12.0 | **yes** |
| `athletes.parquet` | `…/discovery_v1/m11_officials.parquet` + `…/advanced_v1/m12_cohort_registry.parquet` (`linked`) + `…/advanced_v1/m12c_targets.parquet` (`nfl.q90_shared`) | 11.0, 12.0, 12C | optional (needed by `exp.transfer()`) |
| `tcn_representations.parquet` | `…/transfer/transfer_attempt_representations_outer.parquet` (byte copy) | Milestone 06 (06b) | optional (F1_TCN baseline) |
| `bundle.json` | generated: format version, size + sha256 of every file, channels, store key, run mode, input hashes, build signature | builder | **yes** |

These files are read only for the checks and are not copied into the bundle: `m12_foundation_manifest.json` (fold signature, config hash), `m12f_sequence_store.json` (the 12F pointer) and `m12c_target_freeze.json` (the 12C config hash).

`athletes.parquet` is built by `nflbowl.data.export.athletes_table`, the same function `export_research_bundle` uses, so both writers produce the same table.

On this machine (state of 2026-10-16), discovery should find the FULL store `b69677c48cd9f6a9` (77,126 sequences, L = 64, 6 channels), and the pointer names that same store. The fold manifest, cohort registry, 12C targets, officials and TCN-B representations are all present. This has not been run yet: confirm it with `--discover-only`.

## 3. How discovery works

- **Warehouse root:** `$NFL_BDB_WAREHOUSE`, else `warehouse_root` in `configs/phase01.toml`, else `warehouse/`. The research layout is `<warehouse>/gold/advanced_movement/[_sample/]advanced_v1/…`.
- **Stores:** every `sequence_shards/<key>/` in the declared layout, plus a bounded walk (depth ≤ 8, ≤ 50k directories) of `warehouse/`, `data/`, `artifacts/`, `.tmp/`, `notebooks/` and `outputs/`. The walk skips the `bronze`, `silver` and `catalog` layers, dot-directories and existing bundles.
- **Per store:** facts come from `COMPLETE.json` and the `.npy` headers only (shape, dtype, Fortran order). No array is loaded.
- **Selection:** a store is auto-selected only if it is the single complete FULL store in an `advanced_v1` area and the 12F pointer does not name another one. Otherwise the build stops and asks for `--source-sequence-store`. Smoke stores (`_sample/`) are never selected automatically; selecting one explicitly also needs `--allow-smoke`, and is for wiring tests only.
- **Companions:** every other input is taken from the same area and run mode as the store. A smoke file next to a full store is rejected.

## 4. Output

```
bundles/
├── m12f_v1/
│   ├── bundle.json
│   ├── folds.parquet
│   ├── athletes.parquet                 (optional)
│   ├── tcn_representations.parquet      (optional)
│   └── sequence_store/
│       ├── X.npy        (77126, 64, 6) float32  ≈ 113 MB
│       ├── M.npy        (77126, 64) bool        ≈ 4.7 MB
│       ├── meta.parquet
│       └── COMPLETE.json
├── m12f_v1.zip                          (--zip; contains m12f_v1/… only)
└── m12f_v1.reports/                     (never inside the bundle or the ZIP)
    ├── BUILD_REPORT.md
    ├── build_report.json
    ├── checks.csv
    ├── plan.csv
    ├── discovery.csv
    └── discovery.json
```

`bundles/` and `*.zip` are git-ignored.

## 5. Checks

A `fail` stops the build before anything is written. A `warn` is recorded in `checks.csv`.

| Check | Fails when |
|---|---|
| `store.complete`, `store.channels`, `store.arrays`, `store.count_vs_complete` | a store file is missing; the channels are not `speed, a_tan, a_n, yaw, dt, heading_valid`; `X` is not `(N, L, 6)` float32; `M` is not `(N, L)` bool; the rows of `X`, `M` and `meta` differ |
| `store.meta_schema` | `meta` columns are missing, `idx ≠ 0..N-1`, the domain is not combine/nfl, or an athlete has two positions |
| `store.meta_sha256` | `meta.parquet` differs from the hash in `COMPLETE.json` |
| `store.mask_padding` | a valid-mask frame lies beyond `n_frames` |
| `folds.schema`, `folds.ids_integer` | a column is missing, an `nfl_id` is duplicated, or a fold is outside `[0, outer_folds)` |
| `folds.matches_foundation`, `folds.config_hash`, `foundation.run_mode` | the fold manifest and the 12.0 foundation manifest come from different runs |
| `targets.freeze_config` | the 12C freeze has another config hash than 12.0 |
| `bundle.byte_identical` | a copied file does not hash identically to its source |
| warnings | 12F pointer mismatch, recomputed fold signature, empty sequences, position disagreement meta vs officials, linked counts, TCN schema/folds |

Athlete ids, fold assignments, sequence order, channels and values are never modified.

## 6. Upload to Colab

1. Upload `bundles/m12f_v1.zip` to `MyDrive/NFLBowl/`. A single file uploads faster than many.
2. Cell 4 of `00_colab_quickstart.ipynb` finds it and runs `extract_bundle_zip`. That function guards against path traversal, extracts to a temporary folder first, is idempotent and validates the result, so you end up with `MyDrive/NFLBowl/bundles/m12f_v1/`.
3. Alternative: upload the whole `m12f_v1/` folder to `MyDrive/NFLBowl/bundles/`.

Then:

```python
from nflbowl import Experiment
exp = Experiment(data_dir="/content/drive/MyDrive/NFLBowl/bundles/m12f_v1", output_dir="/content/drive/MyDrive/NFLBowl",
                 config="colab_gpu", device="auto", name="f4_main")
exp.prepare()
exp.train("F4", resume=True)
exp.embed("F4", reuse=True)
exp.evaluate("F4", level="standard", reuse=True)
exp.export()
```

The `colab_gpu` preset copies the bundle to `/content/nflbowl_stage` before memory-mapping it, because reading from Drive is slow. Artifacts stay on Drive.

## 7. Avoiding expensive reruns

- The builder never calls a research pipeline. When an input is missing, it names the notebook that writes it (see `BUNDLE_TROUBLESHOOTING.md`).
- Running the builder again with the same inputs and options returns `up_to_date`. It compares the build signature (sha256 of every input plus the options) and then re-validates the bundle.
- An experiment is bound to the bundle fingerprint. A rebuilt bundle with identical contents keeps the same fingerprint, because the fingerprint hashes the files, not `created_at`. Different contents require a new experiment name.
