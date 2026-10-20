# M16 — experiments and validation (laboratory, session 5)

**Status:** authored 2026-10-20, never executed by the author. All results come from the user's Colab run. Gates default to `NOT_EVALUATED`.

Notebook: `notebooks/research/M16_VERIFIED_BASELINES_CROSS_DOMAIN_VALIDATION_COLAB.ipynb` (sections 00–11). It supersedes `notebooks/nflbowl/M16_…`, which keeps only the session-4 core steps. Code: `src/nflbowl/lab/m16/`. Tests: `tests/nflbowl/test_nflbowl_m16.py` and `test_nflbowl_m16_lab.py`.

## 1. What M16 must answer

Does Combine tracking carry generalizable, reproducible, athlete-specific information? Does it improve held-out NFL movement prediction beyond M1 (`draft_year, height, weight, forty, ten`)? Retrieval (G16-D) and prediction (G16-C) are evaluated independently.

M16 keeps four kinds of evidence apart:

- **Observed M15 results:** hand-off numbers. These are context only and are never recomputed here.
- **Unverified saved artifacts:** items in the E16.B4 provenance registry.
- **New hypotheses:** E16-C1 to C3.
- **M16 outcomes:** the tables of the run.

## 2. Dependency map (reused, not cloned)

| Need | Reused API |
|---|---|
| Bundle, folds, experiment | `handoff.open_experiment`, `Experiment.folds_by_athlete`, `exp.features()` (F0), `m16.pipeline.data_from_experiment` → `M15Data` |
| Per-window kinematics | `m15.signals.feature_table` (read from the M15 run when present) |
| Official engine | `ae.oof_ladder`, `ae.Model(factory=…)`, `ev.design` (train-fold medians, missing indicators, scaling, unpenalised position), `se.embedding_block` |
| Blocks, status | `m14.transfer.coverage_factory`, `combine_factories`, `transfer_status`, `fold_stability`, `by_position` |
| Paired statistics | `transfer_fairness.{long_predictions, validate_alignment, wide, paired_comparisons, rmse_intervals, fold_table, rank_agreement, bootstrap_index}`, `position_models.{within_position_metrics, position_effects}`, `m16.baselines.{evaluate_target, within_position_ranking, precise_null}` |
| Learners | `learners.{LEARNERS, oof_generic, design_train}` |
| M1 / targets | `m16.m1_verification.verify_m1`, `m16.targets`, `m16.target_reliability`, `m16.detectability` |
| Representations | `m15.transfer.{pool_members, score_queries, procrustes_map, infonce_heads, concat_embeddings, sequence_matrix_embedding}`, `m15.latent.{athlete_means, repeat_auc, geometry, probes, dim_icc, icc_summary, projection}`, `m15.legacy.audit`, `m15.training.athlete_hash`, `m15.gates` decision rows |
| Matching | `domain_matching.{quantile_bins, strata_codes, smd}` |
| Persistence / status | `M16Run.{table, data_file, load_data, figure, save_markdown, set_signature}`, `status.run_step`, `gate_row` |

## 3. Experiments

Each experiment card answers why, what, how, and what if it fails. The generated report repeats these cards with a scouting rationale.

### Track A — NFL targets

| ID | Module | What | Key outputs |
|---|---|---|---|
| E16.A0–A1 | `m1_verification` | M1 contract and reproduction at 1e-8 (internal and persisted) | `tables/baselines/m1_*`, `manifests/m1_verification.json`, G16-B rows |
| E16.A2 | `targets` | T0, T0x, T1–T6 from NFL data only | `features/targets/{window_table, nfl_events, athlete_targets}` |
| A2 registry | `target_registry` | Typed, versioned definitions; within-athlete bootstrap SE; measurement reliability (1 − mean SE² / within-position variance) | `manifests/target_definitions_m16.json`, `tables/exploration/{definitions, measurement_reliability}` |
| E16.A3 | `target_reliability` | Split-half reliability, exposure, context share, G16-A, one-time freeze | `tables/target/*`, `manifests/target_freeze_m16.json` |
| **E16.A7** | `target_audit` | Distributions/KDE, η² with CI, position means, within-position relations (exposure, duration, kind mix, draft year), threshold, aggregation and subsample sensitivity, correlations with common n, selection SMDs, censoring, scorecard | `tables/exploration/*`, `features/targets/{target_long, athlete_uncertainty}` |
| **E16.A8** | `context_matching` | Drill vs NFL common support on entry speed, peak a_tan, duration, braking share and peak \|a_n\| (exact position). Below `min_support_share` or `min_support_athletes` the result is `BLOCKED_INSUFFICIENT_OVERLAP`. Context-matched outcomes `t16_ctx_<drill>__<T>` are outcomes only. | `tables/context/*`, `features/context/matched_outcomes` |
| **E16.A9** | `cohort` | 401 linked-flag vs 399 observed-dual (both kept), fold integrity (hard), funnel, composition, exposure, inventory | `tables/cohort/*`, `manifests/dual_domain.json` |

### Track B — Representation transfer

| ID | Module | What |
|---|---|---|
| E16.B1, B3 | `retrieval`, `leakage` | Session 4: confirmatory retrieval of G15-eligible representations, leakage verdict (read by M17 / M18) |
| **E16.B4** | `representation_audit` | Every M15 embedding and checkpoint is classified `VERIFIED_FOLD_SAFE` / `EXPLORATORY_ONLY` / `UNKNOWN_PROVENANCE` / `INVALID_LEAKAGE`. Checks: held-out athletes absent from training, roles match official folds, hash and bundle consistent, channel contract, schema, checksum. The G15 decision and E15.C evidence are attached. Legacy folders are audited descriptively. |
| **E16.B5** | `controlled_retrieval` | Verified representations at the pre-declared `ladder_dim`, STFT ⊕ LSTM fusion, and controls (F0, train-z-scored kinematics, analytic position-only, random). Pools: all, same position, stopwatch twins, physical twins. Scopes: all, forty, three_cone, short_shuttle. Arms: unaligned, Procrustes, InfoNCE. Chance is computed per query pool. The decisive test is paired ΔMRR against the **strongest** control (Holm). Also reported: error gallery and similarity distributions. |
| **E16.B6** | `repeatability` | Repeat AUC unaligned vs aligned for each fold and drill (t-interval over 5 folds; destructive if the mean drop is > 0.02 or ≥ 3 folds drop). Also: effective rank, ICC, domain/position probes, M1 re-encoding R², duration sensitivity. |

### Track C — Incremental prediction

The laboratory ladder uses M16 numbering:

| Rung | Model | Inputs |
|---|---|---|
| B0 | POS | position only |
| B1 | M1 | official Combine M1 |
| B2 | M1+simple | `trk_*`, the same columns as G16-C |
| B3 | M1+PCA | PCA embedding block |
| B4 | M1+STFT | STFT-AE embedding block |
| B5 | M1+LSTM | LSTM-AE embedding block |
| B6 | M1+STFT+LSTM | fused block |
| B7 | M1+M14_reliable | blocked when no M14 metric is `validated`; never relaxed |

M18 numbers its confirmatory ladder differently. M18 owns the confirmatory claim.

| ID | Module | What |
|---|---|---|
| **E16.C0** | `features` | Fold-f encoder, Combine attempts only. Aggregations: mean (primary), median, trimmed mean, q25/q75, ICC-weighted (training-athlete ICC). Scopes: all, forty, agility, three_cone, short_shuttle. Fusion is standardised on training athletes. Drill kinematics are `kin_*`. Coverage and M1 re-encoding are reported. |
| **E16.C4** | `ladder`, `evaluation` | Official engine on identical athletes. Learners: ENet, LightGBM, CatBoost (optional) and `pooled` (global ridge plus shrunken per-position slopes, inner CV). Holm runs on the screen family {B2…B7 vs M1}. Reported: calibration, exposure- and coverage-stratified errors, paired within-position Spearman, improvement table, and the aggregation/scope sensitivity ladder on T0. A representation without G15 GO is labelled `exploratory_screen`. |
| **E16.C5** | `ladder.controls` | `n_shuffle` within-position shuffles (null q95, permutation p), 5 random blocks, reduced k ∈ {2, 4}. A gain inside the null is "not athlete-specific". |
| **E16.C6** | `positions` | Global vs separate (n ≥ `position_min_n`, ≥ 5 per fold) vs pooled, compared inside each position. Sample-size readings are insufficient / exploratory / adequate; TE is reported with interval width. |
| **E16-C1–C3** | `hypotheses` | C1: forty → T1, T0. C2: short shuttle → T3, T4. C3: three-cone → T5 and `t16_ctx_three_cone__T5`. Each runs M1 against drill PCA / STFT / LSTM / kinematics (+ F0 for C1), with Holm within the hypothesis. Prerequisites: ≥ `hypothesis_min_athletes`, G16-A not FAIL, censoring ≤ `max_censored_share`, overlap COMPLETED. |

### Track D — Gates and report

- `gates.g16`: unchanged table `m16_gate`, read by M17 / M18.
- `gates.g16_extended`: table `m16_gate_extended`.
- `reporting.build`: writes `reports/markdown/m16_research_report.md`.

## 4. Gate rules of the extended table (pre-specified before execution)

States are PASS · FAIL · EXPLORATORY · BLOCKED · NOT_EVALUATED. Document IDs are kept: **G16-C = incremental information**, **G16-D = cross-domain similarity**. The laboratory brief swaps these two letters; the content is the same.

| Gate | Subject | PASS | EXPLORATORY | FAIL / BLOCKED |
|---|---|---|---|---|
| G16-A | target | `gate_a` PASS and no laboratory flag | `gate_a` PASS with a flag (exposure, threshold, aggregation, selection, censoring) | `gate_a` FAIL. T0 is NOT_EVALUATED (frozen upstream). |
| G16-B | M1 | persisted and internal reproduction pass; every ladder cohort is identical | internal reproduction only (documented controlled refit) | any mismatch: hard stop |
| G16-C | target × rung | `validated` (CI > 0, adjusted p ≤ 0.05, ≥ 4/5 folds) ∧ Δ ≥ 2 % ∧ above the shuffled null (embedding rungs) ∧ G15 GO (representations) | validated or exploratory, but a condition is missing | inconclusive, precise null, or reliably worse = FAIL; rung not fittable = BLOCKED |
| G16-D | representation | same-position vs strongest control (CI > 0, Holm ≤ 0.05) ∧ twin pool vs strongest control (CI > 0) ∧ all-pool vs position-only (CI > 0) ∧ CEM persistence ∧ non-destructive arm ∧ G15 GO | core criteria pass, but CEM is missing or G15 is not GO | a core criterion fails = FAIL; provenance not verified = BLOCKED |

Overall rows:

- **G16-A / G16-D:** PASS if any subject passes.
- **G16-B:** PASS only if every row passes.
- **G16-C:** decided on T0 only.

The MDE is reported next to G16-C.

## 5. Persistence, caching and status

Run directory: `<drive_root>/experimental_lab/m16_verified_baselines_v1/`. Subfolders: `manifests, config, tables/<family>, features/{targets,context,athlete}, predictions, models, retrieval, figures/{png,svg,pdf,data}, reports, checkpoints, logs`. M12–M15 directories are read only (`resolve` and `M16Run.guard`).

Caching:

- A step reloads only when the stored `{inputs, tables}` signature matches. The signature covers bundle and config fingerprints, the target frame fingerprint, and the size + mtime of every upstream embedding.
- `force_steps` forces a recomputation.
- Data files use provenance sidecars.

Every step ends in exactly one status, recorded in `manifests/m16_status.json` and `manifests/stages.json`: READY · COMPLETED · CACHED · BLOCKED_MISSING_INPUT · BLOCKED_PROVENANCE · BLOCKED_INSUFFICIENT_SAMPLE · **BLOCKED_INSUFFICIENT_OVERLAP** (new, shared in `lab/status.py`) · FAILED_VALIDATION · FAILED · NOT_RUN.

## 6. Known risks (unverified by execution)

- **Polars APIs assumed beyond the session-4 list:**
  - `list.join`
  - `Series.skew` inside `agg`
  - `group_by` iteration yielding tuple keys
  - `pl.any_horizontal` / `all_horizontal`
  - `DataFrame.sample(n, seed=)`
- **External engine:** `se.embedding_block` expects tables with `nfl_id` + numeric columns only. `features.by_fold` enforces this.
- **Cost:**
  - E16.C5: 27 ladder refits per embedding rung and target. T0 plus frozen task targets only.
  - E16.B5: about 6 representations × 5 folds × 4 scopes × 3 arms.
  - Both are cached.
- **Stated simplifications:**
  - The SE for T1 uses an unweighted q90 (not the demand-mix weights).
  - The pooled learner's design scaling uses outer-training rows during inner CV, which is unsupervised and mirrors `ae._alpha`.
  - Context bins use all athletes' windows (outcome-free).
  - Matched outcomes use `target_min_events` for windows as well as events.
- **Config fingerprint:** the new science keys change `cfg.fingerprint()`. A pre-existing M16 v1 run recomputes its cached steps once.

## 7. Scientific blockers

1. **E15.C2 robustness:** `m15.preservation.robustness` reads `cfg.dropout_rates`, which `M15Config` lacks. Adding the key would change the M15 fingerprint and invalidate M15 caches. This is the user's decision; M16 only reports it.
2. **G15 decisions:** if M15 produced no GO, every representation rung is `exploratory_screen` and G16-C / G16-D cannot PASS for it.
3. **Persisted records:** without a compatible persisted official OOF record, G16-B is EXPLORATORY (controlled refit), not PASS.
