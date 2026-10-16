# M12 comprehensive lab — Colab execution guide

Notebook: `notebooks/nflbowl/M12_COMPREHENSIVE_EXPERIMENTAL_LAB_COLAB.ipynb`. Package: `nflbowl >= 0.4.0` (`nflbowl.lab`).

**Status: authored only. The notebook, the new modules and the tests have never been executed.** Expect a first round of API-detail fixes (see § 6).

## 1. Prerequisites

- Colab with a GPU runtime (a T4 is enough). The notebook keeps Colab's own PyTorch; torch is not a dependency.
- The cleaned bundle at `MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean/`, with fingerprint `4867be08f8191b3a`, 77,125 sequences, 64 frames, 6 channels, 510 athletes and 5 folds. It is opened read-only and staged to `/content/nflbowl_stage`; it is never rebuilt.
- Optional: earlier experiments under `MyDrive/NFLBowl/experiments/` (`f4_main_clean`, …) or `unified_research/*`. Their compatible inductive checkpoints are adopted.

## 2. Recommended order

| Step | CFG | Purpose | Expected cost (unverified) |
|---|---|---|---|
| 1 | `run_mode='pilot'`, `run_training=False` | Setup, A, B (full audit), D1, D3/G3 with F0 + random and any adopted F4 fold | CPU, ~10–20 min (first run computes the sequence metrics and hashes) |
| 2 | `pilot`, `run_training=True` | trains F4 on `pilot_fold` and runs C, D on it | GPU, one encoder |
| 3 | `run_mode='standard'`, `run_training=True`, `run_transfer=True` | F2 + F4 on 5 folds, E, G4, G5, H, I, J | GPU, 10 encoder runs |
| 4 | `run_mode='full'` + `run_ablations` / `run_seeds` / `run_learning_curves` / `run_permutation_tests` (each opt-in) | mechanism and robustness | GPU-heavy |

Before step 3, pin `GIT_REF` to the tested commit SHA.

## 3. CFG keys

The dict in Setup 2 is validated by `nflbowl.lab.config.resolve`; an unknown key raises.

- `seed` is the **lab** seed (figure sampling, case studies). The science seed stays that of the `colab_gpu` preset, so compatible checkpoints can be reused. `science_seed` changes the science hash, which means a new experiment.
- `precision` defaults to `fp32`. It is locked in `config/precision_lock.json` at the first session; a later session with another precision is refused while training records exist.
- Optional keys: `pilot_fold`, `pilot_encoder`, `primary_encoders`, `secondary_encoders`, `reference_experiments`, `import_checkpoints`, `search_previous_runs`, `n_boot`, `min_subgroup`, `adequate_n`, `anonymize_ids`, `n_case_studies`, `n_profiles`, `use_umap`, `verify_bundle_hashes`, `extra_seeds`.

## 4. Drive output

```text
MyDrive/NFLBowl/experimental_lab/m12_comprehensive_lab_v1/
├── config/            run_config.json · environment.json · precision_lock.json · experiment_registry.csv|md
├── datasets/          audits/ · cached/ (sequence_metrics, athlete_table + provenance)
├── features/          athlete/ (athlete_movement_features)
├── embeddings/        inductive/ (athlete scores, pool / quality scores, projections, norms)
├── checkpoints/       experiments/m12_comprehensive_lab_v1/ (nflbowl experiment: best.pt, epoch shards, embeddings, records)
│                      experiments/m12_comprehensive_lab_v1_seed<k>/ (G1 only)
├── tables/            dataset · retrieval · prediction · ablations · robustness · biomechanics · interpretability · geometry · synthesis (CSV + Parquet)
├── figures/           png (300 dpi) · svg · pdf · data/ (plotted numbers + .meta.json with the research question)
├── reports/           markdown/ · html/ (one per part + FINAL_COMPREHENSIVE_RESEARCH_REPORT) · athlete_profiles/
├── diagnostics/       checkpoint sources / adoption, duplicate sequence groups
├── logs/              run.log · <stage>_traceback.txt
├── manifests/         stages.json (stage ledger) · artifacts.jsonl · artifact_index.json
└── exports/           ZIP archives
```

## 5. Resume and partial execution

- Rerun Setup 1–3 with the same `experiment_name`, then any section. Each section is independent and guarded, and records `executed / reused / skipped / blocked / failed`.
- Training resumes from the last verified epoch shard. Embeddings, comparisons and the OOF ladder are reused when their dependency hashes match.
- Part J/K rebuilds from saved tables (`synthesis.collect`), so it runs in a fresh session.

## 6. Likely first-run issues (static review)

- Polars API details: `unpivot`, `pivot(on=…)`, `replace_strict(default=…)`, `rank('ordinal')` arithmetic, `concat_list(...).list.eval(...)`.
- Matplotlib `violinplot(orientation=…)` (falls back to `vert=`), and the subtitle placement under constrained layout.
- Polars column dtypes in `pl.DataFrame(list_of_dicts)` where some rows lack keys (e.g. `split_reliability` with n = 0).
- `biomech.sequence_metrics` memory: chunk 4096 × 64 × 6, fine on Colab.
- Send back `logs/<stage>_traceback.txt` and `manifests/stages.json`.

## 7. Local tests (not executed during authoring)

```bash
pytest tests/nflbowl/test_nflbowl_lab.py -q
pytest tests/nflbowl -q
```
