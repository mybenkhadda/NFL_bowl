# Bundle troubleshooting

Start with `python -m nflbowl.data.builder --repo-root . --discover-only`. Every failed build also writes `bundles/<name>.reports/BUILD_REPORT.md` and `checks.csv`.

## Colab: `bundle.json not found: data_dir must be an nflbowl data bundle`

The error message now names the cause:

| Hint in the message | Fix |
|---|---|
| "is a ZIP archive" / "but …m12f_v1.zip does" | `from nflbowl.data.builder import extract_bundle_zip; extract_bundle_zip(zip, "/content/drive/MyDrive/NFLBowl/bundles")`, or run Cell 4 of the quickstart |
| "directory … does not exist" | Drive is not mounted, or the bundle was uploaded somewhere else. Mount Drive and check the path |
| "A bundle exists one level down" | use the path printed in the message as `data_dir` |
| "raw nflbdb SequenceStore directory" | you uploaded `sequence_shards/<key>/` itself. Build a bundle locally instead |
| "sequence_store/ is present but bundle.json is not" | the upload is incomplete. Upload the folder again, or use the ZIP |

If the message is `bundle file … has N bytes; bundle.json says M`, Drive is still syncing or the copy was truncated. Wait, or upload again.

## Builder exit code 2: missing sources

| Message | Research step to rerun (same `RUN_MODE='full'`) |
|---|---|
| no complete FULL 12F sequence store | `12f_physics_informed_shared_movement_encoder.ipynb`, Part 3 (store build only) |
| store `missing COMPLETE.json (build interrupted)` | same: the store build was interrupted. The research code writes `COMPLETE.json` last |
| fold manifest … is missing | `12_00_advanced_movement_research_foundation.ipynb` (12.0). If only the CSV view exists, pass `--fold-manifest reports/tables/advanced_movement/advanced_v1/m12_fold_manifest.csv` |
| N candidate FULL sequence stores | no rerun. Choose one with `--source-sequence-store warehouse/gold/advanced_movement/advanced_v1/sequence_shards/<key>`. The message says which store the 12F pointer names |
| smoke-mode store / smoke artifact | rerun the step in FULL mode. Smoke and full outputs are never mixed |

Optional parts are skipped with a warning; they do not block the build:

- `athletes.parquet` needs 11.0 officials and the 12.0 cohort registry.
- The 12C target needs `m12c_targets.parquet` (12C). Without it, `exp.transfer()` is unavailable.
- `tcn_representations.parquet` needs Milestone 06 (06b). Without it, the F1_TCN baseline is unavailable.

## Builder exit code 3: incompatible sources

| Failed check | Meaning / action |
|---|---|
| `store.meta_sha256` | `meta.parquet` was modified after the store was written. Rebuild the store (12F Part 3) |
| `folds.matches_foundation`, `folds.config_hash`, `foundation.run_mode` | the fold manifest and the foundation manifest come from different 12.0 runs. Rerun 12.0, then rerun 12F Part 3 if its cohort changed |
| `targets.freeze_config` | 12C ran under another configuration. Rerun 12C, or build with `--no-include-transfer` |
| `store.mask_padding`, `store.arrays` | the store is damaged. Rebuild it |
| `athletes.build` | officials, registry or targets are unreadable, or an athlete appears twice. Use `--no-include-transfer` or fix the source |

## Builder exit code 4: output exists

- An identical bundle is reported `up_to_date`; this is not an error.
- If the existing bundle came from other inputs or options, or from `export_research_bundle` (no build signature), use `--overwrite` (the old folder is moved aside) or another `--output`.
- If the output folder is a non-bundle folder, the builder never touches it. Choose another `--output`.
- A leftover `bundles/.m12f_v1.building/` from an interrupted build is cleaned up automatically, but only when it contains the builder's marker file.

## Windows: `could not rename … (a program holds a file open)`

OneDrive, an antivirus or Explorer is holding a file open. The complete bundle stays in `.m12f_v1.building/`. Close the program and run the builder again.

## `Experiment … is bound to bundle X, but … is bundle Y`

The experiment was prepared with other bundle contents. Use a new `name=`. A rebuild with identical inputs keeps the same fingerprint.
