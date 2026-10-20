# M15–M18 master research roadmap

**Status:** planning document. It was authored statically on 2026-10-20. No code, notebook, test or experiment was executed to write it.
**Scope:** Milestones 15–18 of the NFL Big Data Bowl 2027 project, built on `nflbowl` 0.6.0 (`src/nflbowl/`) and the `nflbdb` research code (`src/nflbdb/`).
**Companion files:** `M15_M18_EXPERIMENT_MATRIX.md` (experiment register), `M15_M18_IMPLEMENTATION_PLAN.md` (tasks and interfaces), `M15_M18_SCIENTIFIC_GATES.md` (go/no-go rules), `M15_M18_COLAB_NOTEBOOK_SPECIFICATIONS.md` (notebook layouts), `M15_M18_SESSION_HANDOFF.md` (fresh-session entry point). The roadmap figure is drawn by `scripts/figures/m15_m18_roadmap_figure.py`.

---

## 0. The question and how evidence is labelled

**Ultimate question.** Do precise 10 Hz Combine movement measurements give reliable, interpretable information about future NFL movement that is independently predictive and that official Combine testing does not already capture?

A rigorous negative answer is an acceptable outcome. The roadmap is built so that each milestone can end in a documented negative result without wasting the next one.

Every claim in this document and in every M15–M18 report carries one of four labels.

| Label | Meaning | Example source |
|---|---|---|
| **[OBS]** observed | Measured in an executed, protocol-conforming notebook whose saved artifact can be re-read | M06 `transfer_paired_comparisons`, M12 lab tables |
| **[EXP]** exploratory | Measured once, outside the frozen protocol, or on one split. It needs replication. | The independent M15 STFT/LSTM notebook and the 393-athlete RMSE study |
| **[HYP]** hypothesis | Not yet measured | "short shuttle → NFL reacceleration" |
| **[PLAN]** planned | Code or experiments to be written | everything in §§ 3–6 |

Numbers quoted from the user's brief (Findings 1–7) are tagged **[EXP]** unless a saved, protocol-conforming artifact in the repository or on Drive confirms them. The M15 notebooks must re-read the saved tables and reconcile them (E15.00, E16.A0). They never type a published number into an analysis. This is the same rule as `M12_REPORTED` in `src/nflbowl/lab/m13_config.py`.

---

## 1. Repository audit: what exists and what M15–M18 must reuse

### 1.1 Two packages, one data contract

| Layer | Path | Role for M15–M18 |
|---|---|---|
| Portable bundle | `nflbowl.data.loading.open_bundle` → `SequenceBundle(X, M, meta, folds, athletes, tcn)` | **The only data source.** `X` is (N, 64, 6) float32 holding the channels `speed, a_tan, a_n, yaw, dt, heading_valid`, scaled by `[10, 5, 5, 3, 0.1, 1]` (`m11.representation.SCALE`). `M` is the validity mask. `meta` = `idx, seq_key, domain, nfl_id, kind, position_group` (+ `season`, `n_frames`). `folds` = `nfl_id, fold` (5 outer, athlete-level). `athletes` = `nfl_id, position_group, linked, draft_year, height, weight, forty, ten, nfl.q90_shared, …` |
| Schema | `nflbowl.data.validation` | `CHANNELS`, `M1_COLS = (draft_year, height, weight, forty, ten)`, `TRANSFER_TARGET = "nfl.q90_shared"`, `validate_athletes` |
| Splits | `nflbowl.data.splits` | `fold_map`, `split_indices`, `check_split_integrity`, `SplitLeakError` |
| Experiment | `nflbowl.api.Experiment` | `_ladder_inputs`, `transfer_predictions` (official OOF ladder: M1 · M1+F0 · M1+F1_TCN · M1+F2…F6), `inductive_athlete_embeddings`, `compare_inductive` |
| Lab store | `nflbowl.lab.store.LabRun` → `m13_config.M13Run` → `m14.config.M14Run` | Drive layout, `table / frame / data_file / load_data (provenance-keyed cache) / figure (PNG+SVG+PDF+data) / run_stage (ledger, tracebacks)` |
| Hand-off | `nflbowl.lab.handoff` | `environment, open_experiment, discover, adopt, adopt_embeddings, inventory, ready_models, sequence_metrics, f1_validity` |
| Statistics | `nflbowl.lab.stats` | `cluster_bootstrap(_diff)`, `athlete_bootstrap_ci`, `chance_recall_at_k`, `chance_mrr`, `bh`, `effective_rank`, `spearman`, `partial_spearman`, `pseudonym` |
| Fair comparison | `nflbowl.lab.transfer_fairness` | `long_predictions, validate_alignment, reproducibility_check, cohorts (ladder_all / strict_common), wide, bootstrap_index, paired_comparisons, fold_table, rank_agreement` |
| Retrieval | `nflbowl.lab.retrieval_lab`, `nflbowl.evaluation.retrieval`, `nflbowl.evaluation.paired_inductive` | `pool_definitions` (k-nearest, stopwatch-twin pools), `pool_scores`, `similarity_margins`, `paired_pairs`, `quality_sensitivity` |
| Geometry / domain | `nflbowl.lab.geometry`, `nflbowl.lab.domain_matching`, `nflbowl.evaluation.domain_diagnostics` | `basic_geometry, collapse_flags, projection_frame (UMAP optional)`, `cem` (coarsened exact matching), `condition_rows`, `smd`, `conditional_distributions` |
| Reliability | `nflbowl.lab.reliability`, `nflbowl.lab.m14.reliability` | `icc_components, icc_bootstrap, equal_observation_curves, projected_curve`, train-fold `reliable_block`, `measurement_status` |
| Position models | `nflbowl.lab.position_models` | `oof_position_models` (partial pooling, κ grid), `within_position_metrics` |
| M14 biomechanics | `nflbowl.lab.m14.{events, metrics, registry, novelty, transfer, scouting}` | Event detector (`detect_sequence`, `EventParams`), `cached_extract`, `sequence_table`, `athlete_table`, `perturbation`, `novelty_table`, `quality_factory`, `run_ladder`, `transfer_status`, `permutation_null`, `stopwatch_twins`, `twin_contrast` |
| Encoders | `nflbdb.research.advanced_movement.m11.representation` | `ConvEncoder` (dilated residual 1-D conv), `TransformerEncoder`, masked / contrastive / JEPA objectives = F2–F6 |
| Earlier programme | `nflbdb.research.advanced_movement.{context_outcomes, context_reliability, transfer_power, transfer_models, robust, outcome_reliability}` | Variance components and backfit (07A), planted-signal power / MDE (07C), ridge/ENet ladders (06B), matched retrieval (04B), game-split outcome reliability (05B) |

### 1.2 What the audit found that changes the plan

1. **The M15 prototype is not in the repository.** No file holds `stft`, `lstm`, `spectral` or `M15` (search on 2026-10-20). Its numbers (Finding 4) are therefore **[EXP]** with no code provenance. M15 must (a) adopt any saved embeddings through a provenance-checked adapter (`m15/legacy.py`) and (b) re-implement both models inside the fold-safe protocol. Unchecked prototype artifacts never enter a claim.
2. **"Official M1 loading is blocked" is a notebook problem, not a data problem.** M1 is already part of the bundle contract: `athletes.parquet` must hold `M1_COLS`, `validate_athletes` checks them, and `Experiment._ladder_inputs` builds the M1 rung. M16 therefore *reconciles* M1 (E16.A0/A1) rather than recovering it. Optional tests (`three_cone`, `short_shuttle`, vertical, broad, bench) are used only when present as columns, which is checked at run time as in M14.
3. **Two different NFL targets exist.** The protocol target is `nfl.q90_shared` (12C frozen; M13 and M14 use it). The exploratory study used `nfl_speed_q90_median` (393 athletes, position-only RMSE 0.4108). These are not interchangeable. M16 reports both side by side and keeps `nfl.q90_shared` as the primary target unless gate G16-A demotes it. For reference, a different frozen target in M06 (`O2.accel_sust05_mean`) had M1 RMSE 0.1794 and R² 0.947, mostly from position **[OBS, repo docs]**. Scales differ by target, so RMSEs are never compared across targets.
4. **NFL windows are demand-selected.** `meta.kind` for NFL rows is the 09A `demand_id` (`nfl_sequence_arrays`). Finding 2 (Combine vs NFL medians) is therefore partly a *selection* effect by construction. Every M15 domain contrast is reported within `kind` and after length and entry-speed matching.
5. **Masked frames are zeros, not observations.** `_channels` writes `nan → 0`, and `yaw = 0` where heading is invalid. Every reconstruction loss and every spectral estimate must therefore be computed on `M` (and `heading_valid` for yaw) within gap-free segments, using the M14 rule `dt > 0.15 s` breaks a segment.
6. **Spectral resolution is very coarse.** At 10 Hz with ≤ 64 frames (NFL median 43), the frequency resolution is 0.16–0.23 Hz, and 0.3125 Hz with a 32-sample Welch segment. The reported speed spectral centroid of about 0.312 Hz in Combine and 0.319 Hz in NFL **[EXP]** is numerically equal to the first non-zero bin of a 32-sample segment (10/32 Hz). **[HYP]** Spectral summaries of short windows may mostly be trend and leakage artifacts. E15.A3 tests this directly before any spectral feature is promoted.
7. **Retrieval chance must be position-aware.** Among 80 held-out linked athletes in 5 position groups (≈ 16 per group), a model that only recovers position and guesses within it has an expected top-1 of about 1/16 ≈ 6 %. The reported aligned STFT top-1 of 8.75 % (7/80) **[EXP]** is therefore not yet distinguishable from position-only retrieval. Same-position pools and an analytic position-only baseline are mandatory (E15.E2, E16.B1).
8. **History sets the prior.** Repo docs record **[OBS]**:
   - Within-Combine repeat-attempt fingerprints are repeatable (M04: TCN-B within-position AUC 0.873).
   - Cross-domain inductive retrieval was near chance (M12: AUC ≈ 0.50–0.51 for F0 and F2–F6).
   - F4 domain AUC was ≈ 0.997.
   - No tracked Combine feature has shown incremental NFL prediction (M06 case E; M08B C08/C14).
   - M14 measurement validation passed 1 of 30 metric instances (Finding 1, user-reported).

   M15–M18 are designed to test whether *better measurement* changes that conclusion. They do not assume it will.

---

## 2. Cross-milestone architecture

### 2.1 Research dependency graph

```
                       ┌─────────────────────────── PARALLEL STREAM P1 (measurement) ───────────────────────────┐
E15.00 bundle/provenance ─► M15-A signal analytics ─► M15-C preservation ─► M15-D latent stability ─► G15 ─┐
          │                         │                                                                      │
          │                         └─► E15.A3 spectral-resolution test ─► (spectral features or drop) ────┤
          │                                                                                                ▼
          ├──────────── PARALLEL STREAM P2 (baselines/targets; no neural work) ───────────► M16-A ─► G16-A/B ─┐
          │   E16.A0 M1 reconciliation ─► E16.A1 M1 reproducibility ─► E16.A2/3 NFL target family+reliability │
          │                                                                                                  │
          └──────────── PARALLEL STREAM P3 (event repair; reuses M14) ──► E17.A1 diagnosis ─► E17.A2/A3 ──────┤
                                                                                                             ▼
                        G15 + G16-A/B ─► M16-B representation transfer (5-fold) ─► G16-C/D ─► M17-B/C/D ─► G17 ─► M18 ─► G18
                        G16-A/B ─────────────────────────────────────────────────────────────────────────────► M18-A (B0–B3 rungs)
```

| Chain | Steps | Can start |
|---|---|---|
| **P2 (critical path)** | M1 reconciliation → M16 baseline evaluation → M18 incremental prediction | Immediately, CPU only |
| **P1** | M15 latent diagnostics → M16 representation reliability → M17 context-conditioned fusion | Immediately, GPU for B1 |
| **P3** | M14 event reliability repair → M17 contextual signatures → M18 task-matched transfer | Immediately, CPU only |

M18 rungs B0–B3 need only P2 and P3. Rungs B4–B7 need P1 to pass its gates. If P1 fails, M18 still runs and simply reports B4–B7 as `blocked`.

### 2.2 Shared conventions (all four milestones)

- **Unit of inference = athlete.** Every bootstrap is an athlete (cluster) bootstrap, fold-stratified, with one resample matrix shared across a table (`transfer_fairness.bootstrap_index`). Sequence- and event-level statistics are descriptive only.
- **Folds = `bundle.folds`.** These are the M12/M13 five athlete-disjoint outer folds. Inner folds for tuning are drawn inside training athletes only. A new fold system is never created. A milestone that cannot use them records why in `manifests/fold_exceptions.json`.
- **Same-cohort comparisons.** Every model comparison runs on one aligned cohort (`transfer_fairness.validate_alignment` + `wide`). Claims use `strict_common`. Native-coverage results are descriptive.
- **Status vocabulary** = M14 `STATUSES`: `validated · exploratory · inconclusive · blocked · invalid`. Measurement and predictive rules are inherited from `docs/nflbowl/M14_RESEARCH_SPEC.md` § 5. M15–M18 add representation rules (see `M15_M18_SCIENTIFIC_GATES.md`).
- **Multiplicity.** Holm within each pre-declared primary family. BH within exploratory families. Primary families are written to `manifests/preregistration.json` *before* the first outcome-bearing cell runs, and the file is immutable once written.
- **Practical margin.** An incremental gain must exceed 2 % of the reference RMSE (the M06 heuristic) to count as practically relevant, in addition to statistical validation. **Precise null** (M13 H-D) = every CI includes 0 **and** every CI half-width ≤ 2 % of RMSE(reference).
- **Never overwrite M12/M13/M14.** Their run directories are read-only sources. Each new milestone writes only under `/content/drive/MyDrive/NFLBowl/experimental_lab/m1X_<name>_v1/`.
- **Every figure has source data.** `LabRun.figure(fig, name, data={...}, question=..., caption=...)` writes PNG (300 dpi), SVG, PDF and `figures/data/<name>__<part>.{csv,parquet}`.
- **Leakage invariants** (unit-tested with synthetic data, executed by the user):
  1. No held-out athlete's NFL sequence enters any fitted object used to build a Combine-side predictor. This covers scalers, PCA, encoders, alignment heads, spectral normalisers, context bins and selection.
  2. No outcome enters feature selection, case-study selection or twin matching.
  3. `m14n__*`, `nfl__*`, `t16_*` (M16 NFL targets) and NFL-derived embeddings never enter a predictor frame (`m14.transfer.analysis_frame` already raises; the M16–M18 frames extend the guard).
  4. Transductive embeddings never enter a predictive comparison.

---

## 3. Milestone 15 — spectral and temporal representation science

### 3.1 Scientific motivation

- Finding 4 **[EXP]**: aligned STFT top-1 was 8.75 % on one split of 80 athletes. It may be above chance, but it is not above position-only chance (§ 1.2-7). The STFT model reconstructs magnitude only.
- Finding 5 **[EXP]**: domain differences in spectral entropy and power are uncontrolled for length, context and athlete weighting, and may be resolution artifacts (§ 1.2-6).
- Finding 2 **[EXP]**: Combine and NFL windows differ in duration (62 vs 43 valid frames) and demand mix. Maximum-type statistics are length-biased by construction.
- History **[OBS]**: within-Combine fingerprints are repeatable (M04). Cross-domain retrieval was near chance (M12). Neural reconstruction never beat FPCA on Combine curves (M03/M04).

M15 therefore asks whether spectral or recurrent representations are *measurement instruments*: reliable, information-preserving and stable under comparable conditions. It does not ask whether they predict the NFL. That question belongs to M16–M18.

### 3.2 Hypotheses

| ID | Hypothesis | Falsified by |
|---|---|---|
| H15.1 | Spectral summaries of speed/a_tan carry athlete-level information that survives length matching, detrending and fixed-resolution estimation | ICC(1,1) < 0.4 across repeated Combine trials of the same drill, or domain differences vanishing after matching |
| H15.2 | A learned encoder (STFT-AE, LSTM/GRU-AE, dilated TCN) preserves physically meaningful structure better than PCA at equal latent size | Physical-unit errors (peak speed, peak acceleration, braking-onset timing) not better than PCA in ≥ 4/5 folds |
| H15.3 | Latent dimensions are repeatable within athlete across trials of the same drill | Per-dimension trial ICC below 0.4 for the majority of dimensions; repeat-attempt AUC not above the F0 reference |
| H15.4 | Cross-domain athlete retrieval exceeds position-only retrieval | Same-position-pool top-1/MRR CI including the within-position chance |
| H15.5 | STFT and LSTM representations are complementary | Concatenation not better than the better single model on preservation and repeatability; high CCA redundancy |

### 3.3 Research questions

1. Which descriptive Combine–NFL differences survive matching on duration, entry speed, demand kind, position and athlete weight?
2. At 10 Hz and ≤ 64 frames, which spectral quantities are estimable at all? What is their bias from window length and detrending?
3. At equal latent size and training budget, which architecture best preserves peaks, braking onset and derivative structure on *valid* frames?
4. Are latent dimensions repeatable, non-collapsed and not merely position or domain indicators?
5. Does any representation's cross-domain retrieval beat position-only, random and F0 baselines in athlete-disjoint 5-fold evaluation?

### 3.4 Reusable components

`open_bundle`, `splits.split_indices`, `m14.events.physical / segments / detect_sequence` (event-preservation scoring on reconstructions), `m14.metrics.perturbation` (noise and smoothing), `reliability.icc_components / icc_bootstrap / equal_observation_curves`, `stats.effective_rank / chance_*`, `geometry.basic_geometry / collapse_flags / projection_frame`, `domain_matching.cem / smd`, `retrieval_lab.pool_definitions / pool_scores`, `m11.representation.ConvEncoder` (the dilated TCN arm — not reimplemented), `handoff.adopt_embeddings` (pattern for the legacy adapter), `LabRun` store.

### 3.5 Data requirements and availability

| Need | Status |
|---|---|
| Six channels, mask, `kind`, `position_group`, `season` | available in the bundle |
| Repeated Combine trials of the same drill per athlete | available (forty ≈ 2 per athlete; agility fewer; short-shuttle coverage limited — measured in E15.00) |
| NFL game identifier for game-split reliability | **unverified**. `seq_key` = 09A `window_id`; E15.00 tests whether a game key can be parsed. If not, fall back to season split, then random window halves (weaker, labelled). |
| x/y, orientation, opponents, ball | **not available**; nothing requiring them is planned |
| Prototype M15 embeddings and checkpoints | **outside the repo**; adopted only if found on Drive with readable provenance |

### 3.6 Candidate methods

- **Track A (signal analytics).**
  - Gap and validity diagnostics.
  - Length- and context-matched descriptive contrasts.
  - Welch PSD at fixed `nperseg ∈ {16, 32}` after linear detrending, Hann window, on valid segments ≥ `nperseg`.
  - STFT magnitudes.
  - Morlet CWT (PyWavelets when installed, otherwise a NumPy implementation).
  - Spectral entropy, centroid, roll-off (85 %), flatness and band energy in bands {0–0.5, 0.5–1.5, 1.5–3, 3–5 Hz}.
  - Autocorrelation and speed–a_tan cross-correlation lag.
  - Jerk from Δa_tan/dt within segments.
- **Track B (architectures).**
  - PCA / truncated SVD on masked, length-padded sequences (fit on training athletes).
  - STFT-CNN-AE: magnitude, plus a sin/cos phase variant as an ablation.
  - LSTM-AE and GRU-AE: bidirectional encoder, masked loss.
  - Temporal CNN and dilated TCN (`ConvEncoder`).
  - Optional: masked temporal AE (F3-style), wavelet-CNN.
  - Latent sizes d ∈ {8, 16, 32, 64}.
- **Track C (preservation).** All errors computed on valid frames only. Event preservation = the M14 detector applied to the original and to the reconstruction, compared by event-matching F1 and onset-timing error.
- **Track D (latent analytics).** PCA/UMAP (descriptive), effective rank, collapse flags, linear domain and position probes (fold-safe), per-dimension trial ICC, athlete split-half, latent ↔ M14/F0 partial Spearman, nearest-neighbour galleries, decoder perturbation along dimensions.
- **Track E (preliminary transfer).** Compares:
  - no alignment
  - InfoNCE athlete alignment (training athletes only)
  - orthogonal Procrustes on athlete means (training athletes only)
  - random embeddings
  - F0 (17 engineered statistics)
  - an analytic position-only model
  - STFT ⊕ LSTM concatenation

### 3.7 Experiments and ablations

See `M15_M18_EXPERIMENT_MATRIX.md`, IDs E15.00–E15.E3. Key ablations:
- magnitude vs magnitude+phase
- masked vs unmasked loss
- d sweep
- equal parameter count vs equal step budget
- detrend on/off and `nperseg` 16/32 for spectral features
- Combine-only vs Combine+NFL training (training athletes only)

### 3.8 Visual analyses (each with source data)

1. Valid-frame and gap histograms by domain × kind.
2. Athlete-weighted domain contrasts before and after matching (paired dot plots).
3. Spectrogram galleries per drill and demand.
4. Welch PSD median ± IQR by domain at fixed resolution.
5. Centroid vs window length (the resolution artifact test).
6. Spectral feature ICC forest plot.
7. Architecture × d heatmaps of physical-unit errors.
8. Event-preservation F1 bars.
9. Reconstruction overlays (best, median and worst decile).
10. Effective-rank and collapse panels.
11. Per-dimension ICC strip plot.
12. UMAP coloured by domain, position, drill and athlete. Descriptive only, captioned as such.
13. Retrieval top-k curves with position-only and chance bands.
14. Nearest-neighbour signal galleries.

### 3.9 Evaluation metrics

- **Reconstruction:** MSE and MAE (scaled), RMSE in yd/s and yd/s², peak-speed and peak-acceleration absolute error, braking-onset timing error (s), event-matching F1, band-energy relative error, derivative RMSE.
- **Robustness:** degradation under 10/20/30 % frame dropout and under truncation to the NFL length distribution.
- **Latent:** effective rank, domain AUC, position balanced accuracy, trial ICC, split-half r, repeat-attempt AUC.
- **Retrieval:** top-1, top-5, MRR, pairwise AUC (all-position and same-position pools).

### 3.10 Statistics and uncertainty

- Athlete cluster bootstrap (2,000; 500 for model-grid screening) of every fold-pooled metric.
- Paired differences against PCA (preservation) and against F0 / position-only (retrieval).
- Fold-wise win counts.
- ICC(1,1) with bootstrap CI.
- Equal-observation curves k = 2…6 where trials allow.
- Model-grid comparisons are BH-adjusted. The single pre-declared M15 primary comparison is the best d=32 learned model vs PCA d=32 on physical-unit preservation.

### 3.11 Leakage prevention

- Encoders, PCA, scalers and spectral normalisers are fitted per outer fold on training athletes only (inductive regime).
- Early stopping uses an inner validation split of training athletes.
- Alignment pairs come only from training athletes' Combine and NFL sequences.
- Held-out athletes are embedded with frozen weights.
- Reliability uses trials of held-out athletes only for held-out reporting.
- The legacy adapter refuses prototype embeddings whose training-athlete list overlaps the held-out fold. When the list is unknown, the embeddings are labelled `transductive_unknown` and excluded from claims.

### 3.12 Expected artifacts

`RUN_DIR = /content/drive/MyDrive/NFLBowl/experimental_lab/m15_spectral_temporal_v1/`:

```
config/ manifests/{stages.json, m15_status.json, preregistration.json, fold_assignments.parquet, dataset_fingerprint.json}
features/signal/{sequence_signal_features, athlete_signal_features}.parquet
features/spectral/{welch_n16, welch_n32, stft_summary, cwt_summary}.parquet
checkpoints/<arch>_d<d>_s<seed>/fold<k>/{best.pt, config.json, history.parquet}
embeddings/inductive/<arch>_d<d>_s<seed>__fold<k>.parquet       (seq-level: idx, nfl_id, domain, kind, fold, role, z0..z{d-1})
predictions/reconstruction/<arch>_d<d>__fold<k>.parquet          (sampled rows; valid-frame errors per channel)
tables/{dataset, signal, spectral, preservation, latent, reliability, retrieval, legacy, synthesis}/*.parquet|csv
figures/{png,svg,pdf,data}/M15_*
reports/markdown/m15_report.md · reports/html/m15_report.html · model_cards/<arch>.md
```

### 3.13 Computational considerations

- The data are small: 77,125 × 64 × 6 float32 ≈ 118 MB.
- Spectral features: vectorised NumPy over valid segments. CPU only.
- Neural grid: 6 architectures × 4 d × 5 folds = 120 fits per seed. This is the expensive pass.
- Budget matching: a fixed number of optimiser steps per fit, a parameter band per d (± 25 %) and identical batch size.
- Screening runs on 1 seed. Finalists run on 3 seeds.
- Every fit is resumable (`checkpoints/.../best.pt` + `history.parquet`). The cache key = (bundle fingerprint, arch config hash, fold, seed).
- Run time is unmeasured. The notebook prints a projection after the first fit.

### 3.14 Risks and negative-result readings

| Risk | Reading if it happens |
|---|---|
| Spectral features are length artifacts | H15.1 is weakened. Spectral features are dropped and the summary features become simple time-domain ones. Report the artifact as a methods contribution. |
| No learned model beats PCA | H15.2 is weakened. PCA becomes the representation carried into M16, and neural tracks stop. |
| Latent dimensions are not repeatable | H15.3 is weakened. Representations become descriptive only. M17 uses hierarchical models on events instead. |
| Same-position retrieval is at chance | H15.4 is weakened. The prototype's alignment result was position-driven. Alignment work stops unless M17-A context matching changes it. |

### 3.15 Go/no-go gate (summary; full rule in `M15_M18_SCIENTIFIC_GATES.md` G15)

A representation advances to M16 only if it meets all of the following:
- **(a)** physical-unit preservation no worse than PCA at the same d (CI-based)
- **(b)** event-preservation F1 ≥ the PCA value minus 0.05
- **(c)** median per-dimension trial ICC ≥ 0.4, with ≥ 25 % of dimensions ≥ 0.6
- **(d)** not collapsed (effective rank ≥ 0.25·d)
- **(e)** results stable across 3 seeds

Spectral features advance only if they pass the resolution-artifact test (E15.A3) and the ICC rule.

### 3.16 Dependencies

None upstream beyond the bundle. Feeds M16-B (representations), M17-C (fusion) and M18 B4–B6.

### 3.17 Minimum viable experiments (MVE)

E15.00, E15.A1, E15.A2, E15.A3, E15.B1 restricted to {PCA, STFT-AE, LSTM-AE} at d=32 with 1 seed, E15.C1, E15.D2, E15.E1–E15.E2. These alone answer whether the prototype result survives the protocol.

### 3.18 Optional advanced tracks

Masked temporal AE, wavelet-CNN, magnitude+phase STFT, GRU, d=8/64, CWT scalogram features, multi-seed ensembles.

### 3.19 Scouting implications (if validated)

Validated representations would give a repeatable "movement texture" descriptor of Combine drills, usable for similarity search ("which past prospects moved like this one in the three-cone?"). They would carry no predictive claim at this stage.

### 3.20 Definition of done

- `M15_SPECTRAL_TEMPORAL_REPRESENTATION_LAB_COLAB.ipynb` and `src/nflbowl/lab/m15/` are authored.
- The user has executed them in Colab.
- `m15_status.json` shows every MVE stage `done` / `reused`.
- `tables/synthesis/m15_gate.csv` gives a status per representation.
- The report states which representations advance and why, with all figures carrying source data.

---

## 4. Milestone 16 — verified baselines and cross-domain validation

### 4.1 Scientific motivation

- Finding 6 **[EXP]**: position alone predicted `nfl_speed_q90_median` better than M1 + engineered tracking (ΔRMSE −0.020 [−0.041, +0.002]). Within-position RMSE was worse in every group.
- Finding 7: M1 loading, M14 integration and fold-safe embedding evaluation were all missing from that study.
- History **[OBS]**: position explains most of the target variance (M13 context: ≈ 94 % for `nfl.q90_shared`).

Before any complex model, M16 must establish four things:
1. the targets are reliable
2. M1 is reproducible
3. the evaluation is fold-safe and same-cohort
4. the detectable effect size is known

### 4.2 Hypotheses

| ID | Hypothesis |
|---|---|
| H16.1 | At least one NFL movement target beyond `nfl.q90_shared` is reliable across independent games or windows (ICC ≥ 0.6) after position adjustment |
| H16.2 | M1 is exactly reproducible from the bundle (official-ladder equality ≤ 1e-8) |
| H16.3 | M1 improves on position-only for at least one target (paired ΔRMSE CI > 0) |
| H16.4 | Simple tracking statistics or reliable M14 metrics improve on M1 (paired ΔRMSE CI > 0, Holm) |
| H16.5 | Cross-domain retrieval of some representation exceeds same-position chance and F0 in 5-fold athlete-disjoint evaluation, and persists after CEM context matching |

### 4.3 Workstream A — official baseline and target validation

**Targets.** Each is defined from bundle NFL sequences on valid segments. Each target has physical units, an aggregation rule (window → athlete), a minimum exposure, a censoring rule and a selection audit by `kind`.

| ID | Target | Unit | Definition sketch | Censoring / exposure |
|---|---|---|---|---|
| T0 | `nfl.q90_shared` | yd/s | frozen 12C target (read from `athletes`) | as frozen |
| T0x | `nfl_speed_q90_median` | yd/s | exploratory study definition, re-derived from its saved code or table if available | reconciliation only |
| T1 | NFL high-speed capacity | yd/s | athlete q90 over windows of window-max valid speed, within `kind` strata, demand-mix reweighted | ≥ 20 windows; windows ending at max speed flagged right-censored |
| T2 | Acceleration at matched entry speed | yd/s² | mean of a_tan over 0.5 s after entry-speed bins 2–4 / 4–6 yd/s (M05 `band_acc` logic) | ≥ 3 start frames per bin; reported per bin |
| T3 | Braking capacity | yd/s² | athlete q10 of event `max_controlled_decel` (M14 detector, `m14n__`) on complete events | complete events only; exposure curve |
| T4 | Reacceleration after braking | yd/s² or s | M14 `brake_propulsion_transition` / `cut_recovery_time` on NFL events | recovery-censored handled by an interval / Kaplan–Meier median |
| T5 | Speed retention through redirection | ratio | M14 `turn_speed_retention` on events with \|turn\| ≥ 20° and heading coverage ≥ 0.5 | turn-direction coverage audit |
| T6 | Movement consistency | yd/s | within-athlete SD of window-max speed within `kind` (lower = more consistent) | ≥ 30 windows; exposure-corrected |

**Learners.** MEAN · POS · M1-ridge (official, `ae.oof_ladder`) · M1-ElasticNet · M1-LightGBM · M1-CatBoost (optional imports, small trees, monotone-free, inner-CV depth ≤ 3) · M1+F0 · M1+simple tracking · M1+reliable M14 (`m14.transfer.quality_factory`).

All learners use the same folds, cohort and target. Tree learners are adopted only if they beat ridge in ≥ 4/5 folds with a CI > 0 (§ 7).

### 4.4 Workstream B — representation transfer validation

- Five-fold athlete-disjoint retrieval for the representations that pass G15: STFT, LSTM, F0, STFT ⊕ LSTM, aligned variants, Procrustes, random.
- **Pools** (all from `retrieval_lab.pool_definitions` or extensions):
  - all-position
  - same-position
  - stopwatch twins (same position, |Δforty| ≤ 0.03 s, the M09 rule)
- **Metrics:** top-1, top-5, MRR, pairwise AUC, latent ICC, domain AUC, embedding stability across seeds, and sensitivity to CEM context matching (`domain_matching.cem` on length, entry speed, kind, position).
- **The "Combine-only" rule:** representations used later as predictors must be computable for a new prospect from Combine data alone. Alignment models may use NFL data from **training** athletes, never held-out athletes. M16 records for each representation whether it is `combine_only_inductive` (eligible for M18) or `diagnostic_only`.

### 4.5 Visual analyses

1. M1 reconciliation table.
2. Missingness heatmap of official tests by position.
3. Target exposure histograms.
4. Reliability vs exposure curves.
5. Position variance-share bars.
6. `kind`-mix by position.
7. Learner ladder ΔRMSE forest (vs POS and vs M1).
8. Within-position Spearman with CIs.
9. Retrieval curves per pool with chance and position-only bands.
10. Domain AUC before/after CEM.
11. Detectability (MDE) curves.

### 4.6 Statistics

- Paired fold-stratified athlete bootstrap (2,000).
- Holm over the primary family {M1 vs POS on T0; M1+simple vs M1 on T0; M1+M14_reliable vs M1 on T0}.
- BH over the secondary targets T1–T6 and learners.
- Fold win counts.
- Within-position Spearman with bootstrap CIs and BH.
- Detectability is computed with the 07C planted-signal machinery (`nflbdb…transfer_power`), giving the effect size detectable at 80 % power for this cohort.
- The precise-null rule follows M13 H-D.

### 4.7 Leakage prevention

- Target context adjustments (position means, `kind` reweighting) are fit on training athletes per fold.
- Learner hyperparameters are tuned in inner folds.
- M14 selection uses `quality_factory` (train-only ICC, coverage, novelty, redundancy).
- Retrieval representations are inductive.
- Target construction provenance is recorded. If a target uses all-athlete constants (as in 12C), it is labelled "not strictly inductive end-to-end" (M13 decision 7).

### 4.8 Artifacts

`RUN_DIR = …/m16_verified_baselines_v1/`. Main outputs:
- `tables/registry/evaluation_registry.parquet`: one row per (experiment, model, target, cohort, fold, metric)
- `predictions/oof_long.parquet`: `nfl_id, fold, target, model, y, pred, cohort flags`
- `tables/target/{definitions, coverage, reliability, exposure, selection_audit}`
- `tables/baselines/{ladder, paired, by_position, within_position}`
- `tables/retrieval/{pools, paired, cem}`
- `tables/detectability/mde`
- `manifests/m1_contract.json`
- `manifests/target_freeze_m16.json` (immutable once written)

### 4.9 Compute

- CPU for Workstream A. Tree learners are small (≤ 400 athletes).
- Retrieval uses embeddings already produced by M15. Only Procrustes and the alignment heads are trained (GPU optional).

### 4.10 Risks and negative readings

| Outcome | Reading |
|---|---|
| Targets unreliable (ICC < 0.4) or dominated by `kind` selection | Revise definitions (T1–T6 only) and report. T0 stays primary, as frozen. |
| M1 not reproducible | Stop M18 B1+ until fixed. This is a bug, not a finding. |
| M1 ≤ POS | Official testing adds nothing beyond position for this target. M18's question then becomes ranking *within* position. |
| No representation beats same-position chance | Cross-domain similarity claims are dropped. M17-D is downgraded to optional. |

### 4.11 Gate (G16; full rule in the gates file)

- **A:** a target is reliable.
- **B:** M1 is reproducible.
- **C:** incremental information is stable (or a precise null is documented).
- **D:** cross-domain similarity survives same-position and CEM controls.

Failing C or D does **not** stop M18. It changes M18's primary family (§ 6) and puts the negative result on record.

### 4.12 Dependencies, MVE, optional tracks, scouting, done

- **Depends on:** the bundle (Workstream A). G15 (Workstream B).
- **MVE:** E16.A0, A1, A2 (T0, T1, T3), A3, A4 (MEAN, POS, M1-ridge, M1+F0, M1+simple, M1+M14_reliable), A6, B1 (same-position pool), B3.
- **Optional:** CatBoost/LightGBM, T4–T6, season-split reliability.
- **Scouting:** M16 answers "does anything tracked add to the stopwatch for *this* outcome, and how big an effect could we even see?" That is the honest prior any scout should get before reading a tracking metric.
- **Done:** registry and OOF table saved; gates evaluated; report written.

---

## 5. Milestone 17 — context-aware athletic capability learning

### 5.1 Motivation

- Finding 1 **[EXP]**: braking and transition metrics had near-zero or negative ICC.
- Finding 3 **[EXP]**: drills map to different movement behaviours, but short-shuttle coverage is thin.
- Finding 2: domain contexts differ.

Low ICC can come from three sources: real within-athlete variability, poor event definitions (thresholds, double dips, censoring), or mixing incomparable contexts. M17 separates these explanations before learning "traits".

### 5.2 Hypotheses

| ID | Hypothesis |
|---|---|
| H17.1 | Part of the M14 unreliability is definitional: drill-specific, context-stratified event metrics have higher ICC than the M14 pooled versions |
| H17.2 | After context adjustment, between-athlete variance exceeds measurement noise (athlete variance share ≥ 0.4) for at least one event family |
| H17.3 | Athlete random effects estimated in Combine correlate with the same athlete's NFL random effects (disattenuated r > 0, CI-based) for physically matched families |
| H17.4 | Context-conditioned alignment improves same-position retrieval over unconditioned alignment without lowering M15 preservation metrics |

### 5.3 Track A — event-based context matching

- **E17.A1** diagnoses M14 failures per metric:
  - events per trial
  - share of `weak_braking`, `propulsion_censored`, `exit_censored`, double-dip merges
  - threshold sensitivity (brake_thresh 0.5/1.0/1.5; sustain 1/2/3)
  - per-drill ICC vs pooled ICC
- Repairs only what the diagnosis implicates. Candidates:
  - drill-specific phase templates (short shuttle has a known 5-10-5 structure, so the turn count is fixed and expected events can be matched by order)
  - entry-speed-stratified metrics
  - censoring-aware estimators
- **Strata:** entry speed (2–4, 4–6, 6–8, ≥ 8 yd/s), deceleration demand (drop / entry), turn magnitude (< 20°, 20–90°, > 90° where heading is valid), phase (accelerating / braking), duration, position, completeness.
- Coverage per stratum × domain is published before any modelling.

### 5.4 Track B — hierarchical athlete trait models (before any neural model)

- Linear mixed model per event metric: `y_ij = μ + context_ij·β + u_athlete + e_ij`. Use `statsmodels` MixedLM when available; otherwise the 07A method-of-moments backfit (`context_outcomes.backfit`).
- Variance components with athlete-bootstrap CIs.
- Reliability of the athlete mean at k events: σ²_u / (σ²_u + σ²_e/k).
- Empirical-Bayes shrunk athlete effects.
- Context-adjusted residual profiles.
- Reliability-weighted aggregation across drills.
- Fit per fold on training athletes; held-out athletes get BLUPs from fixed training parameters.

### 5.5 Track C — representation fusion

Runs only for representations passing G15 and G16-D. Arms:
- STFT
- LSTM
- concatenation
- late fusion: ridge stacking with inner folds
- attention fusion: a single head over ≤ 2 views, heavily regularised
- reliability-weighted fusion: weights ∝ train-fold ICC

The cohort is small (≤ ~400 linked athletes), so every learned fusion weight is fitted on training athletes and compared against plain concatenation. Attention fusion is only kept if it beats concatenation in ≥ 4/5 folds.

### 5.6 Track D — context-conditioned cross-domain learning

- InfoNCE with context-matched positives: a Combine event and an NFL event of the same athlete in the same entry-speed × turn stratum.
- Negatives restricted to same-position athletes.
- Alternatives: CORAL, MMD, entropic optimal transport (POT if installed, otherwise Sinkhorn in NumPy/torch), DANN, and multiple-instance pooling over an athlete's events.
- **Simple baselines first:** orthogonal Procrustes and CORAL on PCA features.
- **Damage check:** each alignment is re-scored on M15 preservation and repeatability metrics. An alignment that improves retrieval but drops repeat-attempt AUC below the unaligned model is labelled `destructive`.

### 5.7 Track E — interpretable signatures

- Candidate signatures: acceleration profile, braking-to-propulsion transition, turn-speed retention, high-speed dynamics, smoothness (SPARC, with the M14 caveats) and context-adjusted consistency.
- A label is assigned only when the signature's correlation with its named physical quantity is ≥ 0.5 (|ρ|, CI-based) on held-out athletes and its ICC passes.
- Otherwise the signature is reported as `unlabelled latent k`.

### 5.8 Visuals

1. Failure-mode bars per metric.
2. Threshold-sensitivity heatmaps.
3. Stratum coverage matrix by domain.
4. Variance-component stacked bars.
5. Reliability vs k events.
6. Caterpillar plots of shrunk athlete effects by position.
7. Combine-RE vs NFL-RE scatter with disattenuated r.
8. Fusion ablation forest.
9. Alignment trade-off plot (retrieval gain vs preservation loss).
10. Signature–physics correlation matrix.

### 5.9 Statistics

- Variance components via REML, or MoM with bootstrap.
- Disattenuated correlation r / √(rel_C·rel_N) with bootstrap CI. Values above 1 are reported, never clipped, with a note.
- Fusion and alignment comparisons are paired by fold and athlete-bootstrapped. BH within track. Holm for the single pre-declared primary H17.2 family.

### 5.10 Leakage

- Strata boundaries are fixed a priori, not data-tuned.
- Mixed-model fixed effects are fitted on training athletes.
- Context-conditioned positives use training athletes only.
- No NFL outcome is used in signature definition.

### 5.11 Artifacts

`RUN_DIR = …/m17_context_signatures_v1/`:
- `features/events_v2/*` (repaired events plus diagnostics)
- `tables/variance/*`
- `tables/signatures/athlete_signatures.parquet` (fold-tagged, BLUPs + SE)
- `tables/fusion/*`, `tables/alignment/*`
- `model_cards/*.md`

### 5.12 Compute

- Mixed models: seconds to minutes on CPU.
- Event re-extraction reuses the `cached_extract` pattern (a Python loop of about 77k sequences, a few minutes).
- Alignment models are small. GPU is optional.

### 5.13 Risks and readings

| Outcome | Reading |
|---|---|
| Repaired events still unreliable | Measurement at 10 Hz with ≤ 2 trials cannot resolve braking traits. This is a strong, publishable negative. Signatures are restricted to speed/acceleration families. |
| Athlete variance < noise | No trait claim. M18 B7 is blocked. |
| Combine–NFL RE correlation ≈ 0 | Transfer of that family is unsupported. Drill-to-NFL hypotheses (M18-C) are restricted to families with positive r. |
| Alignment is destructive | Keep unaligned representations. |

### 5.14 Gate G17 (summary)

A signature passes only if:
- ICC ≥ 0.6 (lower CI ≥ 0.4) **after** context adjustment
- athlete variance share ≥ 0.4
- novelty: OOF R² from M1 + position ≤ 0.5
- held-out generalisation: athlete split-half r ≥ 0.5 on held-out athletes
- an association with a reliable M16 target, |ρ| CI excluding 0 within position, BH

### 5.15 Dependencies, MVE, optional, scouting, done

- **Depends on:** M14 events (P3, from day 1), G16-A (targets for the association test), and G15 / G16-D for Tracks C and D only.
- **MVE:** E17.A1, A2, B1, B2, E1.
- **Optional:** all of Track D except Procrustes/CORAL; attention fusion; MIL.
- **Scouting:** a validated context-adjusted braking or reacceleration signature would let scouts compare change-of-direction quality *at equal entry speed*, which the stopwatch cannot do.
- **Done:** the signature table carries a status per signature and model cards exist for any learned component.

---

## 6. Milestone 18 — NFL performance transfer and scouting value

### 6.1 Motivation

M18 is the decisive test. It is run once on the frozen definitions from M16/M17, with a pre-registered primary family.

### 6.2 Study A — augmentation ladder

| Rung | Predictors | Source | Eligible if |
|---|---|---|---|
| B0 | position | `POS` | always |
| B1 | M1 | `ae.M1_COLS` | G16-B |
| B2 | M1 + simple tracking | F0 / M15 time-domain athlete features | always (descriptive if G16-C fails) |
| B3 | M1 + reliable biomechanics | `quality_factory` over M14 / M17-A metrics | ≥ 1 metric validated |
| B4 | M1 + STFT | M15 inductive athlete embeddings (≤ 8 train-fold PCA dims, as in the official ladder) | G15 + `combine_only_inductive` |
| B5 | M1 + LSTM | same | same |
| B6 | M1 + fusion | M17-C | G17 fusion arm |
| B7 | M1 + validated signatures | M17-E | G17 |

- **Primary family (Holm):** {B2, B3, B4…B7 that are eligible} vs B1 on the primary target. The primary target is T0 unless G16-A promoted a task target, in which case that decision is recorded in `manifests/preregistration.json` before M18 runs.
- **Secondary:** B1 vs B0, every rung on T1–T6 (BH), and learners (ridge primary; trees secondary).

### 6.3 Study B — position-specific modelling

- Arms: pooled-with-position-dummies, partial pooling (`position_models.oof_position_models`, κ grid) and separate within-position models (WR, DB, DL, OL; TE only when n ≥ 30).
- **Metrics:** within-position RMSE, within-position Spearman, pairwise concordance (probability that a model correctly orders two same-position athletes), top-decile hit rate.

### 6.4 Study C — drill-to-NFL movement transfer (task matching)

**Pre-declared pairs:**
- forty features → T1 (high speed)
- short shuttle → T4 (reacceleration)
- three-cone → T5 (redirection)

**Specificity matrix.** Every drill × every target. Physically meaningful transfer predicts a *diagonal* pattern: the forty predicts T1 better than T4, and so on. A uniform pattern points to general athleticism or position confounding, not task transfer.

Effects are reported in physical units per 1 SD within position. Short-shuttle analyses report n and are `exploratory` when n < 100.

### 6.5 Study D — scouting decision research

- Case studies use stopwatch twins (`m14.scouting.stopwatch_twins`: same position, z-distance on official tests ≤ 0.75), selected **without outcomes** (largest Δ in a validated feature among close pairs).
- Each case shows:
  - official results
  - tracking features with percentiles
  - model prediction ± interval
  - observed NFL outcome when available, revealed after the prediction
  - an interpretation and the limitations
- Athletes are pseudonymised (`stats.pseudonym`).
- No injury, force, reaction-time or causal-development claims.

### 6.6 Study E — robustness and negative results

- athlete bootstrap
- fold stability
- position subgroups
- season/cohort sensitivity (`meta.season`, `draft_year`)
- model-complexity ablation
- feature-family ablation
- missingness (complete-case vs imputed-with-indicator)
- exposure (T restricted to athletes above the median exposure)
- negative controls: random embeddings, shuffled athlete linkage of Combine features, and features from a different drill
- random-label sanity: permuted target within position; every model must show ΔRMSE ≈ 0
- refit permutation null (`m14.transfer.permutation_null`, ≥ 200 permutations for the primary)

### 6.7 Visuals (publication set)

1. Ladder forest (ΔRMSE vs B1 with CIs and practical margin).
2. Fold-wise lines.
3. Within-position concordance bars.
4. Drill × target specificity heatmap.
5. Negative-control panel.
6. Permutation null histogram.
7. MDE-annotated result plot ("what we could have detected").
8. Twin case-study cards.
9. Executive one-page figure.

### 6.8 Success criterion

A metric or representation earns scouting relevance only if:
- it is `validated` under the M14 predictive rule (CI lower > 0, Holm p ≤ 0.05, better in ≥ 4/5 folds)
- it exceeds the 2 % practical margin
- it passes the negative controls
- it has passed its measurement gate (G15 or G17)

Anything less is reported with its exact status. A precise null is reported as a finding.

### 6.9 Artifacts

`RUN_DIR = …/m18_scouting_value_v1/`:
- `predictions/oof_long.parquet`
- `tables/{ladder, position, specificity, robustness, controls, scouting}`
- `reports/{m18_final_report.md|html, executive_summary.md}`
- `reports/athlete_profiles/*.md`
- `model_cards/*`
- `figures/publication/*` (PNG/SVG/PDF + data)

### 6.10 Compute, risks, dependencies, MVE, done

- **Compute:** CPU except embedding inference. The permutation null is the costliest step: 200 refits × ladder, minutes on CPU.
- **Risks:** small cohort, which gives wide CIs. The M16 MDE gives the reading.
- **Depends on:** G16-A/B (always), G15/G17 (rungs B4–B7).
- **MVE:** Study A B0–B3 on T0 plus the Study E controls.
- **Done:** final report, executive summary, evidence registry, and publication figures with source data.

**Scouting implication.** M18 answers the GM-level question: "Should tracking data change how I rank two prospects with the same 40 time?" Possible answers are *yes, for this outcome and position, by this much*, *no detectable difference at this sample size*, or *no — and here is the precise null*.

---

## 7. Alternative routes and the evidence that justifies complexity

| Primary path | Simpler alternative (always run) | Adopt the complex path only if |
|---|---|---|
| Deep AE (STFT/LSTM/TCN) | PCA / truncated SVD at the same d | better physical-unit preservation (paired CI > 0, ≥ 4/5 folds) **and** latent ICC ≥ PCA's |
| Spectral CNN | Handcrafted Welch/band features (E15.A3) | the CNN beats handcrafted features on repeatability **and** retrieval, and the handcrafted features pass the resolution test |
| Contrastive alignment (InfoNCE) | Orthogonal Procrustes / CORAL on athlete means | same-position MRR gain over Procrustes with CI > 0, and no preservation damage |
| Context-conditioned neural embedding | Mixed-effects athlete model on events (M17-B) | the neural embedding beats BLUPs on held-out ICC and on the M16 target association |
| Attention fusion | Concatenation / reliability-weighted average | beats concatenation in ≥ 4/5 folds with CI > 0 |
| CatBoost / LightGBM | Ridge / Elastic Net (official engine) | ΔRMSE vs ridge CI > 0 on the same cohort; otherwise trees are reported as a sensitivity |
| Optimal transport / DANN | CORAL | as for alignment |
| Separate within-position models | Pooled + position dummies, partial pooling | within-position concordance gain CI > 0 in ≥ 3 positions |

Transformers and large architectures get no priority. F6 (transformer) already exists from M12 and is used only as a reference.

---

## 8. Failure and pivot logic

| Failure | Detected by | Pivot |
|---|---|---|
| Embeddings not repeatable | G15 (c) | Representations become descriptive only. M17-B hierarchical events become the main path. B4–B6 are blocked. |
| Same-position retrieval near chance | E15.E2 / E16.B1 | Drop cross-domain similarity claims. Keep Combine-only representations for prediction. M17-D becomes optional. |
| M1 hard to reproduce | E16.A1 | Stop M18. Fix the data contract (bug). Never substitute an approximate M1. |
| NFL targets dominated by selection (`kind` share > 0.25 or ρ(value, log exposure) > 0.3) | E16.A3 | Use within-`kind` targets with demand-mix reweighting. If still failing, T0 stays the only target. |
| Tracking fails to improve baseline | G16-C / M18 | Report a precise null with MDE. Shift the product to descriptive profiles (as in M07D/M09). |
| Alignment damages prediction | E17.D damage check, M18 B4 vs B4-unaligned | Use unaligned representations. Report the trade-off. |
| Position cohorts too small (< 30) | M18-B | Pool with partial pooling. Report positions as exploratory. TE is never a primary. |
| Latent interpretations unstable across seeds | E15.D2, E17.E1 | Label `unlabelled latent k`. No scouting language. |
| Short-shuttle coverage too low | E15.00 | Study C shuttle arm becomes exploratory. Three-cone + shuttle are pooled as `agility` (M14 scope). |

---

## 9. Experiment prioritisation (summary; full matrix in the experiment register)

Scores are 1–5. "Info" = expected diagnostic value. "P(evid.)" = probability of producing interpretable evidence. Priority = (Importance + Info + P(evid.) + Scouting) − (Difficulty + Compute) + Availability, so that cheap, high-information experiments come first.

| Rank | Experiment | Imp. | Avail. | Diff. | Compute | Info | Dependency | P(evid.) | Scouting |
|---|---|---|---|---|---|---|---|---|---|
| 1 | E16.A0/A1 M1 reconciliation + reproducibility | 5 | 5 | 1 | 1 | 5 | none | 5 | 4 |
| 2 | E15.00 bundle and provenance reconciliation | 5 | 5 | 1 | 1 | 4 | none | 5 | 2 |
| 3 | E15.E2 position-aware retrieval chance (analytic) | 5 | 5 | 1 | 1 | 5 | E15.00 | 5 | 3 |
| 4 | E16.A3 target reliability and selection audit | 5 | 4 | 2 | 1 | 5 | E16.A2 | 4 | 4 |
| 5 | E16.A6 detectability / MDE | 4 | 5 | 2 | 2 | 5 | E16.A1 | 5 | 4 |
| 6 | E15.A2 length/context-matched contrasts | 4 | 5 | 2 | 1 | 4 | E15.00 | 4 | 3 |
| 7 | E15.A3 spectral resolution-artifact test | 4 | 5 | 2 | 1 | 5 | E15.00 | 5 | 2 |
| 8 | E17.A1 M14 failure diagnosis | 5 | 4 | 2 | 2 | 5 | M14 events | 4 | 4 |
| 9 | E16.A4 baseline learner ladder | 5 | 4 | 2 | 1 | 4 | A1, A2 | 4 | 5 |
| 10 | E17.B1 mixed-effects variance components | 5 | 4 | 3 | 1 | 5 | A1 | 4 | 4 |
| 11 | E15.B1 MVE architectures (PCA/STFT/LSTM, d=32) | 4 | 5 | 3 | 3 | 4 | E15.00 | 3 | 2 |
| 12 | E15.D2 latent repeatability | 5 | 4 | 2 | 1 | 5 | B1 | 4 | 3 |
| 13 | E16.B1 five-fold retrieval with pools | 4 | 4 | 2 | 2 | 4 | G15 | 4 | 3 |
| 14 | M18 Study A B0–B3 | 5 | 4 | 2 | 2 | 5 | G16-A/B | 4 | 5 |
| 15 | M18 Study C specificity | 4 | 3 | 3 | 2 | 4 | G16-A, G17 | 3 | 5 |
| 16 | E17.D context-conditioned alignment | 3 | 3 | 4 | 4 | 3 | G15, G16-D | 2 | 3 |
| 17 | E15.B1 full grid (all arch × d × 3 seeds) | 3 | 5 | 3 | 5 | 3 | MVE | 3 | 2 |
| 18 | E17.C attention fusion | 2 | 3 | 4 | 3 | 2 | G17 | 2 | 2 |

---

## 10. Execution policy (binding for every M15–M18 session)

- Claude authors code and documents only. Claude never executes Python, notebooks, tests, shell commands or installs.
- The user executes in Colab and sends back: `manifests/stages.json`, `logs/*_traceback.txt`, and the "send back" tables listed at the end of each notebook.
- Numbers in reports are read from saved tables (the `‹M1X:…›` token pattern of earlier milestones), never typed in by hand.
- Optional dependencies (`scipy`, `scikit-learn`, `statsmodels`, `lightgbm`, `catboost`, `PyWavelets`, `umap-learn`, `POT`) are imported lazily. A missing package marks the dependent stage `blocked`, never `failed`. A future `pyproject` extra `m15` lists them. Core dependencies are unchanged.
