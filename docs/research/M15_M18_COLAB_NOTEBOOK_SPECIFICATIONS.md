# M15–M18 Colab notebook specifications

**Status:** these are specifications. No notebook has been authored or executed yet.

## Common skeleton

All four notebooks share the M14 skeleton (`notebooks/nflbowl/M14_CROSS_SPORT_ATHLETIC_INTELLIGENCE_COLAB.ipynb`).

| Cell | Content |
|---|---|
| **Title (md)** | Question, evidence labels ([OBS]/[EXP]/[HYP]/[PLAN]), execution policy, gate summary, the "send back" list |
| **Setup 1** | Mount Drive; GPU check (`nflbowl.devices`) |
| **Setup 2** | `pip install` (repo + optional extras); print `nflbowl.__version__`; import the milestone package; print optional-package availability |
| **Setup 3** | `CFG = {...}` (the only edited cell) → `cfg = resolve(CFG)` → `run = M1XRun(cfg.run_dir, cfg)` → `run.write_config(cfg, handoff.environment(cfg.m13()))` |
| **Setup 4** | `exp = handoff.open_experiment(cfg.m13(), run)`; `bundle = exp.bundle`; fingerprint check against `EXPECTED`; fold map persisted |
| **Sections** | One `run.run_stage("<ID>", fn, inputs=[...])` per experiment ID. Each section ends with a markdown "Reading" cell that states how to interpret pass / fail, with numbers drawn from tables |
| **Synthesis** | Gate table, report rendering (token fill + audit), `run.write_status()`, `run.export_zip()` |
| **Send back (md)** | Exact files to return |

**Stage IDs** equal the experiment IDs (`E15.A3`, …), so `manifests/stages.json` lines up with the experiment matrix.

**Cell conventions:**
- No analysis number is typed by hand.
- Every figure is saved with `run.figure(fig, name, data=..., question=..., caption=...)`.
- Large intermediate data go to `run.data_file(...)` with a provenance dict, so a re-run reuses them.

---

## M15 — `M15_SPECTRAL_TEMPORAL_REPRESENTATION_LAB_COLAB.ipynb`

| § | Stage | Content | Tables | Figures | Checkpoints / cache |
|---|---|---|---|---|---|
| 0 | Setup 1–4 | as above; `cfg.architectures`, `latent_dims` and `seeds` printed with the grid size and the projected fit count | `dataset/run_plan` | — | — |
| A0 | E15.00 | counts; trials per drill; linked cohort; game-key parse; prototype `legacy.find/audit` | `dataset/counts_by_domain_kind_position`, `dataset/trials_per_drill`, `dataset/linked_cohort`, `legacy/prototype_audit` | `M15_A0_01_counts`, `M15_A0_02_trials_per_drill` | `manifests/dataset_fingerprint.json`, `fold_assignments.parquet` |
| A1 | E15.A1 | validity and gaps (athlete-weighted sampling for plots) | `signal/validity_by_domain_kind`, `signal/validity_sequences` (sampled) | `M15_A1_01_valid_frames_hist`, `_02_gap_share`, `_03_heading_valid_share` | `features/signal/validity.parquet` |
| A2 | E15.A2 | domain contrast waterfall over 5 matching steps | `signal/domain_contrast_steps` | `M15_A2_01_contrast_waterfall`, `_02_matched_distributions` | `features/signal/sequence_time_features.parquet` |
| A3 | E15.A3 | Welch (n16/n32 × detrend), STFT summary, CWT summary; surrogates; artifact test; reliability; perturbation | `spectral/features_summary`, `spectral/artifact_test`, `spectral/reliability`, `spectral/sensitivity` | `M15_A3_01_spectrogram_gallery` (per drill/demand), `_02_psd_by_domain`, `_03_centroid_vs_length`, `_04_surrogate_auc`, `_05_spectral_icc_forest` | `features/spectral/{welch_n16, welch_n32, stft_summary, cwt_summary}.parquet` |
| A4 | E15.A4 | equal-observation curves; NFL split reliability | `reliability/spectral_k_curves` | `M15_A4_01_k_curves` | — |
| A5 | E15.A5 (opt.) | coupling and jerk | `signal/coupling_reliability` | `M15_A5_01_phase_portrait` | `features/signal/coupling.parquet` |
| B | E15.B1 | **requires two opt-ins.** `training.run_grid` (resumable); prints the param count per arch × d (band check) and a time projection after the first fit | `preservation/grid_status`, `latent/param_counts` | `M15_B_01_training_curves` | `checkpoints/<arch>_d<d>_s<seed>/fold<k>/{best.pt, config.json, history.parquet}`, `embeddings/inductive/*.parquet` |
| C1 | E15.C1 | valid-frame errors, peaks, event preservation, bands, derivatives; paired vs PCA | `preservation/metrics`, `preservation/paired_vs_pca` | `M15_C1_01_error_heatmap` (arch × d), `_02_event_f1`, `_03_overlays_best_median_worst`, `_04_band_error` | `predictions/reconstruction/*` (sampled) |
| C2 | E15.C2 | dropout and truncation; embedding drift | `preservation/robustness` | `M15_C2_01_degradation_curves` | — |
| D1 | E15.D1 | geometry, probes, projections (UMAP when available; captioned "descriptive") | `latent/geometry`, `latent/probes` | `M15_D1_01_effective_rank`, `_02_umap_domain`, `_03_umap_position`, `_04_umap_drill`, `_05_umap_athlete_subset` | `features/projections/*.parquet` |
| D2 | E15.D2 | per-dimension ICC, split-half, repeat AUC, seed stability | `reliability/latent_icc`, `latent/seed_stability` | `M15_D2_01_dim_icc_strip`, `_02_repeat_auc` | — |
| D3 | E15.D3 | interpretation, nearest-neighbour galleries, decoder perturbations | `latent/interpretation` | `M15_D3_01_corr_matrix`, `_02_nn_gallery`, `_03_perturbation_traces` | — |
| E1 | E15.E1/E2 | retrieval arms × pools; position chance | `retrieval/scores`, `retrieval/paired`, `retrieval/by_fold`, `retrieval/position_chance` | `M15_E1_01_topk_curves` (chance + position-only bands), `_02_mrr_forest` | Procrustes / InfoNCE heads under `checkpoints/align/` |
| E3 | E15.E3 | damage check | `retrieval/damage` | `M15_E3_01_tradeoff` | — |
| S | G15 | gate table; model cards; report | `synthesis/m15_gate` | `M15_S_01_gate_matrix` | `model_cards/*.md` |

**Send back:**
- `stages.json` and tracebacks
- `tables/dataset/counts_by_domain_kind_position.csv`
- `tables/legacy/prototype_audit.csv`
- `tables/spectral/artifact_test.csv`
- `tables/preservation/paired_vs_pca.csv`
- `tables/reliability/latent_icc.csv`
- `tables/retrieval/paired.csv`
- `tables/synthesis/m15_gate.csv`
- `reports/markdown/m15_report.md`

---

## M16 — `M16_VERIFIED_BASELINES_CROSS_DOMAIN_VALIDATION_COLAB.ipynb`

| § | Stage | Content | Tables | Figures |
|---|---|---|---|---|
| 0 | Setup | read-only sources M12–M15; `preregistration.json` written here (primary family, primary target) **before** § A4 | `handoff/sources` | — |
| A0 | E16.A0 | M1 contract, official tests present, missingness, units, funnel (incl. exploratory 393 reconciliation if an ID list is supplied) | `cohort/funnel`, `cohort/missingness`, `cohort/units` | `M16_A0_01_funnel`, `_02_missingness_heatmap` |
| A1 | E16.A1 | official ladder OOF; M1 refit; POS/MEAN; reproducibility | `baselines/m1_reproducibility` | — |
| A2 | E16.A2 | NFL window and event tables; T0–T6; coverage; redundancy | `target/definitions`, `target/coverage`, `target/redundancy` | `M16_A2_01_target_distributions_by_position`, `_02_kind_mix` |
| A3 | E16.A3 | split reliability, exposure, context share; **freeze** | `target/reliability`, `target/exposure`, `target/context_share` | `M16_A3_01_reliability_vs_exposure`, `_02_position_variance_share` |
| A6 | E16.A6 | MDE simulation (resumable) | `detectability/mde` | `M16_A6_01_power_curves` |
| A4 | E16.A4 | learner ladder per frozen target; registry | `registry/evaluation_registry`, `baselines/paired`, `baselines/fold_table` | `M16_A4_01_ladder_forest`, `_02_fold_lines` |
| A5 | E16.A5 | within-position | `baselines/within_position` | `M16_A5_01_within_position_spearman` |
| B1 | E16.B1 | retrieval pools | `retrieval/m16_scores`, `retrieval/m16_paired` | `M16_B1_01_pool_curves` |
| B2 | E16.B2 | CEM, domain AUC | `retrieval/cem`, `latent/domain` | `M16_B2_01_domain_auc_before_after` |
| B3 | E16.B3 | leakage audit | `diagnostics/leakage_audit` | — |
| S | G16 | gates A–D; report | `synthesis/m16_gate` | `M16_S_01_gate_matrix` |

**Checkpoints / cache:**
- `features/targets/athlete_targets.parquet`
- `features/targets/window_table.parquet`
- `predictions/oof_long.parquet`
- `manifests/target_freeze_m16.json` (immutable)

**Send back:**
- `m1_contract.json`
- `baselines/m1_reproducibility.csv`
- `target/reliability.csv`
- `detectability/mde.csv`
- `registry/evaluation_registry.csv`
- `synthesis/m16_gate.csv`
- the report

---

## M17 — `M17_CONTEXT_AWARE_ATHLETIC_SIGNATURES_COLAB.ipynb`

| § | Stage | Content | Tables | Figures |
|---|---|---|---|---|
| 0 | Setup | sources: M14 events (read-only), M15 embeddings, M16 targets/freeze | — | — |
| A1 | E17.A1 | failure modes, threshold grid (cached per params hash), per-drill ICC | `events_v2/failure_modes`, `events_v2/threshold_grid`, `events_v2/per_drill_icc` | `M17_A1_01_failure_bars`, `_02_threshold_heatmap` |
| A2 | E17.A2 | strata + coverage | `context/coverage` | `M17_A2_01_stratum_coverage_matrix` |
| A3 | E17.A3 | censoring-aware estimates | `events_v2/censoring` | `M17_A3_01_km_examples` |
| B1 | E17.B1 | mixed models / backfit; variance components | `variance/components` | `M17_B1_01_variance_stacked`, `_02_reliability_vs_k` |
| B2 | E17.B2 | BLUPs; cross-domain r | `signatures/athlete_effects`, `variance/cross_domain_r` | `M17_B2_01_caterpillar_by_position`, `_02_combine_vs_nfl_re` |
| C1 | E17.C1 (cond.) | fusion arms | `fusion/*` | `M17_C1_01_fusion_forest` |
| D1 | E17.D1 (cond.; two opt-ins) | Procrustes, CORAL → InfoNCE-context, MMD, OT, DANN, MIL | `alignment/*` | `M17_D1_01_alignment_tradeoff` |
| E1 | E17.E1 | signature status + labels; model cards | `signatures/status`, `signatures/labels` | `M17_E1_01_signature_physics_corr` |
| S | G17 | gate; report | `synthesis/m17_gate` | `M17_S_01_gate_matrix` |

**Checkpoints / cache:**
- `features/events_v2/<params_hash>/events.parquet`
- `checkpoints/align/<arm>/fold<k>/`
- `tables/signatures/athlete_signatures.parquet` (fold-tagged)

**Send back:**
- `events_v2/failure_modes.csv`
- `events_v2/per_drill_icc.csv`
- `variance/components.csv`
- `variance/cross_domain_r.csv`
- `synthesis/m17_gate.csv`
- the report

---

## M18 — `M18_NFL_SCOUTING_VALUE_COLAB.ipynb`

| § | Stage | Content | Tables | Figures |
|---|---|---|---|---|
| 0 | Setup | reads the gates of M15–M17 and the M16 freeze; computes `eligible_rungs`; prints the pre-registered primary family (from `preregistration.json`, written in M16 § 0 or updated here **before** § A) | `ladder/eligibility` | — |
| A | E18.A1 | ladder B0–B7 on the primary target, then secondary targets | `ladder/oof_long`, `ladder/paired`, `ladder/status`, `ladder/fold_table` | `M18_A_01_ladder_forest` (margin band), `_02_fold_lines`, `_03_mde_annotated` |
| B | E18.B1 | position arms | `position/*` | `M18_B_01_within_position_concordance` |
| C | E18.C1 | specificity matrix | `specificity/matrix`, `specificity/diagonal_contrast` | `M18_C_01_specificity_heatmap` |
| D | E18.D1 | twins, cases (selected without outcomes; outcomes revealed last) | `scouting/twins`, `scouting/twin_contrast` | `M18_D_01_twin_cards` |
| E | E18.E1 | robustness, negative controls, random label, permutation null | `robustness/*`, `controls/*` | `M18_E_01_controls_panel`, `_02_permutation_null` |
| F | E18.F1 / G18 | evidence registry; report; executive summary; publication figure set | `synthesis/evidence`, `synthesis/m18_gate` | `M18_F_01_executive_figure` + `figures/publication/*` |

**Send back:**
- `ladder/status.csv`
- `controls/*.csv`
- `specificity/diagonal_contrast.csv`
- `synthesis/evidence.csv`
- `synthesis/m18_gate.csv`
- `reports/markdown/{m18_final_report, executive_summary}.md`
