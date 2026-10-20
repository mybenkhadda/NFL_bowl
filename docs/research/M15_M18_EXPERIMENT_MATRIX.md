# M15–M18 experiment matrix

**Status:** a register of planned experiments. Nothing in this file has been executed.

Each entry answers five questions:
- **Why:** the finding it addresses.
- **What:** the data, method and repository component.
- **Works if:** the metrics, comparison and acceptance rule.
- **If it fails:** which hypothesis is weakened, and the fallback.
- **Scouting:** the question it could answer.

**Labels and conventions:**
- Evidence labels: [OBS], [EXP], [HYP], [PLAN], as defined in the master roadmap § 0.
- "Folds" always means `bundle.folds`: 5 athlete-disjoint outer folds.
- "Bootstrap" means a fold-stratified athlete cluster bootstrap with 2,000 replicates unless stated.
- Priority: **P0** = MVE, needed before anything downstream. **P1** = core. **P2** = optional or advanced.

**Shorthand used in the entries:**

| Name | Meaning |
|---|---|
| `B` | bundle |
| `S` | `nflbowl.lab.stats` |
| `TF` | `nflbowl.lab.transfer_fairness` |
| `RL` | `nflbowl.lab.retrieval_lab` |
| `REL` | `nflbowl.lab.reliability` |
| `M14E` | `nflbowl.lab.m14.events` |
| `M14M` | `nflbowl.lab.m14.metrics` |
| `M14R` | `nflbowl.lab.m14.reliability` |
| `M14T` | `nflbowl.lab.m14.transfer` |
| `DM` | `nflbowl.lab.domain_matching` |
| `GEO` | `nflbowl.lab.geometry` |
| `PM` | `nflbowl.lab.position_models` |
| `TP` | `nflbdb.research.advanced_movement.transfer_power` |
| `CO` | `nflbdb.research.advanced_movement.context_outcomes` |

---

## M15 — spectral and temporal representation science

### E15.00 — Bundle, cohort and provenance reconciliation (P0)

- **Why:** The counts in the brief are [EXP] (77,125 / 6,310 / 70,815; 510 / 399 / 393). The M15 prototype is not in the repository (master roadmap § 1.2-1).
- **What:**
  - Open the bundle with `open_bundle(verify=True)` and check `EXPECTED` from `m13_config`.
  - Count sequences by domain × kind × position.
  - Count athletes with ≥ 2 trials per drill.
  - Count linked athletes.
  - Try to parse a game key from NFL `seq_key`.
  - Search Drive for the prototype artifacts. If found, run `m15.legacy.audit()`: schema, row count, training-athlete list, fold overlap.
- **Baseline:** the brief's counts and `EXPECTED`.
- **Variables:** none.
- **Validation:** exact equality for counts. Each prototype artifact is classified `adoptable_inductive`, `transductive_unknown` or `unreadable`.
- **Metrics:** count deltas; prototype status.
- **Output:**
  - `tables/dataset/{counts_by_domain_kind_position, trials_per_drill, linked_cohort}`
  - `tables/legacy/prototype_audit`
  - `manifests/dataset_fingerprint.json`
- **Acceptance:** fingerprint equals `4867be08f8191b3a`. Every mismatch with the brief is listed and explained.
- **If it fails:** the bundle differs from the M12–M14 bundle. Stop and decide on a re-bundle before any comparison.
- **Scouting:** none directly. This is the provenance basis for every later claim.

### E15.A1 — Signal validity and gap diagnostics (P0)

- **Why:** Masked frames are stored as zeros, yaw is 0 when heading is invalid (`_channels`), and NFL windows are shorter (Finding 2).
- **What:**
  - Per sequence: valid-frame count, internal gaps (`dt > 0.15 s`), number of segments (`M14E.segments`), heading-valid share, longest segment, leading/trailing invalid frames.
  - Sampling is athlete-weighted: at most 20 NFL windows per athlete for plots; all rows for tables.
- **Baseline:** none (descriptive).
- **Variables:** domain, kind, position, season.
- **Validation:** descriptive. Athlete-weighted medians with bootstrap CIs.
- **Metrics:** medians and IQRs; share of sequences with ≥ 1 gap.
- **Output:** `tables/signal/validity_*`, figures `M15_A1_*`.
- **Acceptance:** none (descriptive). The table drives the masking rules used later.
- **If it fails:** not applicable.
- **Scouting:** sets which Combine drills give complete enough data to profile at all.

### E15.A2 — Length-, context- and athlete-matched domain contrasts (P0)

- **Why:** Finding 2 medians [EXP] mix drill composition, demand selection and window length.
- **What:** Recompute the Finding 2 statistics (max speed, q90 speed, max/min a_tan, braking-frame fraction, valid frames) under five conditions:
  1. raw
  2. athlete-weighted (equal weight per athlete)
  3. Combine cropped to the NFL valid-length distribution (random crop with a fixed seed, valid frames only)
  4. entry-speed matched via `DM.cem` on (first-second mean speed, length, position)
  5. within `kind`
- **Baseline:** raw medians.
- **Variables:** matching step.
- **Validation:** bootstrap CI of the domain difference at each step.
- **Metrics:** the domain Δ and the share of the raw Δ that remains after each step.
- **Output:** `tables/signal/domain_contrast_steps`, figure `M15_A2_contrast_waterfall`.
- **Acceptance:** none. The reading is how much of the gap is context. Feeds M17-A strata.
- **If it fails:** if the differences persist after matching, that is consistent with real demand differences (still not athlete change).
- **Scouting:** a scout should not read a Combine max speed as an NFL speed; this quantifies by how much and why.

### E15.A3 — Spectral feature family with a resolution-artifact test (P0)

- **Why:** Finding 5 [EXP]. A centroid of ≈ 0.312 Hz equals 10/32 Hz, the first non-zero bin of a 32-sample segment (§ 1.2-6).
- **What:**
  - Welch PSD on valid segments ≥ `nperseg`, with `nperseg ∈ {16, 32}`, Hann window, 50 % overlap, linear detrend on/off.
  - STFT magnitude on padded valid frames (used only for the AE input).
  - Morlet CWT (PyWavelets, otherwise NumPy).
  - Features: entropy, centroid, roll-off (85 %), flatness, band energy {0–0.5, 0.5–1.5, 1.5–3, 3–5 Hz}, autocorrelation at lags 1–5, speed–a_tan cross-correlation peak lag, jerk RMS.
  - **Artifact test:** regress each feature on segment length and nperseg; compare with synthetic signals (pure linear ramps, the bundle's own sequences time-reversed, phase-randomised surrogates with the same PSD). Run the Combine-vs-NFL comparison at equal length (cropped).
- **Baseline:** time-domain summary F0.
- **Variables:** nperseg, detrend, length crop.
- **Validation:**
  - The feature is "estimable" if |ρ(feature, length)| ≤ 0.3 within domain × kind and its value on ramps differs from real data (AUC ≥ 0.7).
  - Reliability: ICC(1,1) across repeated Combine trials of the same drill (`REL.icc_components`, `icc_bootstrap`).
- **Metrics:** ρ with length, surrogate AUC, ICC [CI], perturbation Spearman (`M14M.perturbation` noise 0.05/0.10, smoothing 3).
- **Output:** `features/spectral/*`, `tables/spectral/{artifact_test, reliability, sensitivity}`, figures `M15_A3_*`.
- **Acceptance:** a spectral feature advances only if it is estimable, has ICC ≥ 0.6 with lower CI ≥ 0.4, and perturbation Spearman ≥ 0.8 (the M14 `validated` measurement rule).
- **If it fails:** H15.1 is weakened. Drop spectral summaries. The STFT-AE is still tested in B1, but its input is acknowledged to be resolution-limited.
- **Scouting:** whether "movement rhythm" descriptors are real athlete traits or artifacts of clip length.

### E15.A4 — Within-athlete spectral variability and smoothing sensitivity (P1)

- **Why:** Finding 5 requires reliability evidence before any trait claim.
- **What:**
  - Equal-observation curves (`REL.equal_observation_curves`, k = 2…6 where trials exist).
  - NFL: game-split if a game key exists (E15.00), otherwise season-split, otherwise random halves (labelled).
  - Sensitivity to `M14M.perturbation` grids.
- **Baseline:** time-domain features.
- **Variables:** k, perturbation.
- **Validation:** ICC with CIs.
- **Metrics:** ICC(k), Spearman–Brown projection (`REL.projected_curve`).
- **Output:** `tables/reliability/spectral_k_curves`.
- **Acceptance:** none on its own; it informs G15.
- **If it fails:** spectral features need more trials than the Combine provides. State the required k.
- **Scouting:** how many attempts a club would need to record for a stable descriptor.

### E15.A5 — Speed–acceleration coupling and jerk dynamics (P2)

- **Why:** Track A of the brief; candidate complements to event metrics.
- **What:** Within segments:
  - jerk = Δa_tan/Δt
  - the speed–a_tan phase portrait (binned a_tan by speed band, the M05 `band_acc` logic)
  - cross-correlation peak lag
- **Baseline:** F0.
- **Validation:** ICC.
- **Metrics:** ICC, novelty (`m14.novelty.oof_explained` vs M1 + position).
- **Output:** `features/signal/coupling`.
- **Acceptance:** as for E15.A3.
- **If it fails:** drop.
- **Scouting:** whether "acceleration sustained at speed" is distinct from the forty.

### E15.B1 — Architecture benchmark under equal budget (P0 MVE subset; P2 full grid)

- **Why:** Finding 4 [EXP] covers one split and two models. History [OBS]: no neural encoder beat FPCA on reconstruction (M03/M04).
- **What:** Models, all inductive per outer fold:
  - PCA / TruncatedSVD on masked, flattened sequences (masked mean-imputation fitted on training athletes)
  - STFT-CNN-AE (magnitude; ablation magnitude + sin/cos phase)
  - BiLSTM-AE
  - BiGRU-AE
  - temporal CNN AE
  - dilated TCN-AE (`m11.representation.ConvEncoder`, reused)
  - optional masked temporal AE
  - optional wavelet-CNN

  Training rules:
  - d ∈ {8, 16, 32, 64}.
  - Loss is masked (valid frames; yaw only where heading is valid).
  - Equal budget: fixed optimiser steps, batch 256, AdamW, cosine schedule, parameter band ± 25 % per d.
  - Early stopping on an inner validation split of training athletes.
  - Training data: Combine + NFL sequences of training athletes. Ablation: Combine only.
- **Baseline:** PCA at the same d.
- **Variables:** architecture, d, seed, training domain.
- **Validation:**
  - Screening: 5 folds × 1 seed.
  - Finalists (best 2 learned + PCA at d=32, plus each finalist's best d): 3 seeds.
- **Metrics:** see E15.C1 and E15.D1/D2. Also training time and parameter count.
- **Output:** `checkpoints/…`, `embeddings/inductive/…`, `tables/preservation/grid`, `tables/latent/grid`.
- **Acceptance:** this feeds G15. It is not a pass/fail of its own.
- **If it fails:** if PCA wins, H15.2 is weakened and PCA is carried forward.
- **Scouting:** none directly.

### E15.C1 — Physical information preservation (P0)

- **Why:** Track C. The STFT prototype reconstructs magnitude only (Finding 4).
- **What:** For held-out athletes, compute on valid frames only:
  - reconstruction MSE and MAE (scaled)
  - RMSE in yd/s and yd/s²
  - peak speed and peak a_tan absolute error
  - min a_tan error
  - braking-onset timing error and event-matching F1: run `M14E.detect_sequence` on the original and on the reconstruction; an event matches when its minimum lies within ±0.3 s
  - speed-minimum and propulsion-transition timing
  - band-energy relative error (E15.A3 bands)
  - derivative RMSE (Δspeed/Δt vs a_tan consistency)

  STFT-AE: for the magnitude-only arm, the time-domain reconstruction is not defined. That arm is scored on spectrogram error only and is flagged `no_time_domain`. The phase arm is scored fully.
- **Baseline:** PCA at the same d.
- **Variables:** architecture × d.
- **Validation:** paired athlete bootstrap Δ vs PCA; fold wins.
- **Output:** `tables/preservation/{metrics, paired_vs_pca}`, `predictions/reconstruction/*` (sampled), figures `M15_C1_*`.
- **Acceptance:** for G15 (a) and (b).
- **If it fails:** learned models destroy event structure → not suitable for biomechanical interpretation.
- **Scouting:** whether a model summary can be trusted to keep the braking moment a scout cares about.

### E15.C2 — Robustness to missing frames and short NFL windows (P1)

- **Why:** Finding 2 (43 vs 62 frames). Gaps (E15.A1).
- **What:**
  - Random frame dropout of 10/20/30 % (contiguous blocks of 1–3 frames).
  - Truncation of Combine sequences to the NFL length quantiles (q25/q50/q75).
  - Errors are measured only on frames observed in the *original*, never on dropped frames treated as truth.
  - Embedding drift: cosine distance between the clean and degraded embedding of the same sequence.
- **Baseline:** PCA.
- **Validation:** paired bootstrap.
- **Metrics:** Δerror; median drift; retrieval of a sequence's own clean embedding (top-1).
- **Output:** `tables/preservation/robustness`.
- **Acceptance:** drift top-1 ≥ 0.9 at 10 % dropout (heuristic; reported, not gating).
- **If it fails:** the representation is unstable for NFL-length windows. Cross-domain use needs length-matched training.
- **Scouting:** none directly.

### E15.D1 — Latent geometry and collapse (P0)

- **Why:** Track D. M12 [OBS]: F4 domain AUC ≈ 0.997.
- **What:**
  - Effective rank (`S.effective_rank`) and collapse flags (`GEO.collapse_flags`).
  - Covariance spectrum.
  - Domain AUC and position balanced accuracy from a train-fold linear probe.
  - Drill probe.
  - PCA/UMAP projections (`GEO.projection_frame`) coloured by domain, position, drill and athlete. Descriptive only.
- **Baseline:** PCA, F0, random.
- **Variables:** architecture × d.
- **Validation:** fold-wise probes.
- **Metrics:** effective rank / d; domain AUC; position balanced accuracy.
- **Output:** `tables/latent/geometry`, figures `M15_D1_*`.
- **Acceptance:** G15 (d): effective rank ≥ 0.25·d.
- **If it fails:** collapsed → excluded.
- **Scouting:** none.

### E15.D2 — Latent repeatability and athlete stability (P0)

- **Why:** H15.3. History [OBS]: within-position repeat-attempt AUC 0.873 (M04 TCN-B) is a historical reference, not a target.
- **What:**
  - Per-dimension trial ICC across repeated attempts of the same drill (held-out athletes).
  - Athlete split-half r on athlete-mean embeddings.
  - Within-position repeat-attempt AUC (`nflbowl.evaluation.retrieval` within-Combine: attempt 1 vs attempt 2).
  - Seed stability: Procrustes-aligned correlation of embeddings across seeds.
- **Baseline:** PCA, F0.
- **Variables:** architecture × d × drill.
- **Validation:** bootstrap CIs.
- **Metrics:** median per-dimension ICC, share of dimensions with ICC ≥ 0.6, repeat AUC, seed r.
- **Output:** `tables/reliability/latent_icc`, `tables/latent/seed_stability`.
- **Acceptance:** G15 (c) and (e).
- **If it fails:** H15.3 is weakened → the representation is descriptive only.
- **Scouting:** whether "moves like" comparisons are stable from attempt to attempt.

### E15.D3 — Latent interpretation (P1)

- **Why:** Track D. Labels must follow evidence.
- **What:**
  - Partial Spearman (`S.partial_spearman`, controlling for position) between each latent dimension and F0 / M14 / E15.A3 features.
  - CCA redundancy with F0.
  - Nearest-neighbour signal galleries.
  - Decoder perturbation (±1 SD along a dimension) for generative models, plotted in physical units.
- **Baseline:** none.
- **Validation:** held-out athletes; BH.
- **Metrics:** |ρ|, CCA canonical correlations.
- **Output:** `tables/latent/interpretation`, figures `M15_D3_*`.
- **Acceptance:** a dimension is described in words only if |ρ| ≥ 0.5 with a CI excluding 0.3, stable across seeds.
- **If it fails:** unlabelled.
- **Scouting:** the language allowed in a profile.

### E15.E1 — Five-fold cross-domain retrieval with controls (P0)

- **Why:** Finding 4 [EXP] (one split, 80 athletes, all-position pool).
- **What:** Athlete-level retrieval, with Combine as query and NFL as gallery, in each held-out fold. Arms:
  - unaligned STFT, LSTM, PCA
  - InfoNCE-aligned (heads trained on training athletes)
  - Procrustes-aligned (training athletes)
  - STFT ⊕ LSTM
  - F0
  - random Gaussian embedding with the same d
  - prototype embeddings if `adoptable_inductive`

  Pools (`RL.pool_definitions`): all-position and same-position.
- **Baseline:** F0, random, position-only (E15.E2).
- **Variables:** arm × pool.
- **Validation:** fold-pooled paired bootstrap (`RL.paired_pairs`); fold wins.
- **Metrics:** top-1, top-5, MRR, pairwise AUC, with chance (`S.chance_recall_at_k`, `S.chance_mrr`).
- **Output:** `tables/retrieval/{scores, paired, by_fold}`, figure `M15_E1_topk_curves`.
- **Acceptance:** exploratory in M15. The confirmatory test is E16.B1.
- **If it fails:** at chance → H15.4 is weakened.
- **Scouting:** whether a Combine drill identifies how the same player moves on Sundays.

### E15.E2 — Analytic position-only retrieval baseline (P0)

- **Why:** § 1.2-7. Perfect position recovery gives about 6 % top-1 among 80 athletes.
- **What:**
  - Closed-form expected top-1 / MRR when position is identified perfectly and the within-position pick is random: E[top-1] = mean over queries of 1/n_pos(query).
  - Simulated "position + noise" retrievers with position accuracy 0.6–1.0.
  - Re-express the prototype's 7/80 against these baselines, with exact binomial intervals.
- **Baseline:** uniform chance.
- **Validation:** analytic.
- **Output:** `tables/retrieval/position_chance`.
- **Acceptance:** a reference line on every retrieval figure.
- **If it fails:** not applicable.
- **Scouting:** prevents over-reading of position-driven similarity.

### E15.E3 — Alignment damage check (P1)

- **Why:** Track E plus the M17-D concern.
- **What:** For each aligned arm, recompute E15.C1 and E15.D2 metrics vs its unaligned counterpart.
- **Validation:** paired bootstrap.
- **Acceptance:** an arm is labelled `destructive` if repeat AUC falls by more than 0.02 or preservation worsens with CI > 0.
- **If it fails:** keep unaligned representations.
- **Scouting:** none.

---

## M16 — verified baselines and cross-domain validation

### E16.A0 — M1 contract and official-test audit (P0)

- **Why:** Finding 7: "M1 loading blocked" [EXP]. M1 is in the bundle contract (`M1_COLS`).
- **What:**
  - Read `B.athletes`; run `validate_athletes(require_target=True)`.
  - List every official-test column actually present (e.g. `three_cone`, `short_shuttle`, `vertical`, `broad_jump`, `bench`, only when present).
  - Missingness by position.
  - Units audit (implausible-value share).
  - Linked cohort: 510 → linked → with T0 → with M1 complete.
  - Reconcile with the exploratory 393.
- **Baseline:** the brief's counts.
- **Validation:** exact.
- **Output:** `manifests/m1_contract.json`, `tables/cohort/{funnel, missingness, units}`.
- **Acceptance:** M1 columns present with ≥ 0.95 coverage in the linked cohort, or the shortfall is explained.
- **If it fails:** the bundle is incomplete → re-export with `include_transfer=True` (`_ladder_inputs` message).
- **Scouting:** which official numbers a model can actually lean on.

### E16.A1 — M1 reproducibility (P0)

- **Why:** G16-B.
- **What:**
  - `Experiment.transfer_predictions(models=[], …)` gives the official M1 / M1+F0 OOF.
  - The M16 baseline engine refits M1 on the same frame.
  - `TF.reproducibility_check(tol=1e-8)`.
  - Add POS and MEAN through the same `ae.oof_ladder` (the M13 pattern).
- **Baseline:** official ladder.
- **Validation:** exact equality.
- **Output:** `tables/baselines/m1_reproducibility`.
- **Acceptance:** all rows `pass`.
- **If it fails:** stop. Fix the bug.
- **Scouting:** trust in every later comparison.

### E16.A2 — NFL target family construction (P0 for T0/T1/T3; P1 for T4–T6)

- **Why:** Finding 6 used one speed target. Finding 3 suggests task-specific targets.
- **What:** Targets T0–T6 (master roadmap § 4.3), built from bundle NFL sequences via `m15.signals` and `M14M.cached_extract` (for event-based T3–T5).
  - Each target has a unit, aggregation, minimum exposure and censoring rule.
  - Window → athlete aggregation within `kind`, with demand-mix reweighting to the position's pooled `kind` mix (weights computed on training athletes).
  - T0x is reproduced if the exploratory code or table is available.
- **Baseline:** T0.
- **Variables:** target, minimum exposure.
- **Validation:** coverage per position; Spearman between targets (redundancy).
- **Output:** `features/targets/athlete_targets.parquet` (fold-tagged), `tables/target/{definitions, coverage, redundancy}`.
- **Acceptance:** ≥ 30 athletes per position for at least 4 positions.
- **If it fails:** the target is not estimable → dropped.
- **Scouting:** what "NFL speed / braking / reacceleration" can mean with these data.

### E16.A3 — Target reliability, exposure and selection audit (P0)

- **Why:** Targets must be reliable before transfer. History [OBS]: M05 reliability methodology.
- **What:**
  - Split reliability in priority order: game, then season, then random window halves.
  - ICC and Spearman–Brown, within position.
  - Exposure curves at 20/50/100/200 windows.
  - ρ(value, log exposure) within position.
  - Kind-share of variance (`kind` as context).
  - Censoring share.
- **Baseline:** T0.
- **Validation:** athlete bootstrap.
- **Metrics:** reliability [CI], exposure ρ, context share.
- **Output:** `tables/target/{reliability, exposure, context_share}`, `manifests/target_freeze_m16.json` (immutable).
- **Acceptance (G16-A):** reliability ≥ 0.7 with lower CI ≥ 0.5; |ρ exposure| ≤ 0.3; context share ≤ 0.25 or adjusted reliability ≥ 0.7. These are the M05 screening lines.
- **If it fails:** the target is selection-dominated → revise the definition or keep only T0.
- **Scouting:** which NFL outcomes are measurable traits vs usage artifacts.

### E16.A4 — Baseline learner ladder (P0 core; P1 trees)

- **Why:** Finding 6 [EXP] (position beat M1+tracking).
- **What:** MEAN, POS, M1-ridge, M1-ENet, M1+F0, M1+simple tracking (M15 time-domain athlete features), M1+M14_reliable (`M14T.quality_factory`), and optionally M1-LightGBM and M1-CatBoost (depth ≤ 3, inner-CV early stopping), for each reliable target.
- **Baseline:** POS and M1.
- **Variables:** learner × feature family × target.
- **Validation:** same cohort (`TF.cohorts`, `strict_common`); paired bootstrap (`TF.paired_comparisons`); fold table.
- **Metrics:** RMSE, MAE, R², within-position RMSE, Spearman.
- **Output:** `predictions/oof_long.parquet`, `tables/registry/evaluation_registry.parquet`, `tables/baselines/*`.
- **Acceptance:** status by the M14 predictive rule. Holm over the primary family (roadmap § 4.6).
- **If it fails:** a precise null if half-widths ≤ 2 % of RMSE; otherwise inconclusive.
- **Scouting:** "Does anything beat the 40 + position for this outcome?"

### E16.A5 — Within-position ranking (P1)

- **Why:** Finding 6 [EXP] within-position ρ: DB −0.086, DL 0.267, OL 0.155, TE −0.078, WR 0.166. With n ≈ 33–98 per position the SE is ≈ 0.10–0.18.
- **What:**
  - Within-position Spearman and pairwise concordance for each learner.
  - Partial pooling (`PM.oof_position_models`).
  - Per-position paired Δ (`PM.position_effects`).
- **Baseline:** M1 within position.
- **Validation:** bootstrap; BH across positions.
- **Output:** `tables/baselines/within_position`.
- **Acceptance:** exploratory unless CI > 0 after BH.
- **If it fails:** no within-position ranking signal.
- **Scouting:** ranking players at one position, the real draft-board use.

### E16.A6 — Detectability and minimum detectable effect (P0)

- **Why:** The cohort is small (about 250–400). A null result is uninterpretable without power.
- **What:**
  - 07C machinery (`TP.synthetic_target`, `TP.run_simulation`, `TP.mde_table`) on each reliable target's real structure (M1 OOF residual SD by position).
  - Planted signal fraction f ∈ {0.01, 0.02, 0.05, 0.1}.
  - Resumable by configuration hash.
- **Validation:** simulation replicates.
- **Output:** `tables/detectability/mde`.
- **Acceptance:** the MDE is reported next to every M16/M18 result.
- **If it fails:** not applicable.
- **Scouting:** tells clubs how many linked athletes a confirmatory study would need.

### E16.B1 — Five-fold representation transfer retrieval (P0)

- **Why:** G16-D; confirmatory version of E15.E1.
- **What:**
  - Representations passing G15, plus F0, random and Procrustes.
  - Pools: all-position, same-position, stopwatch twins (same position, |Δforty| ≤ 0.03 s), via `RL.pool_definitions` (`twin_k`) or an extension.
  - Pre-declared primary: the best G15 representation vs F0, same-position pool, MRR.
- **Baseline:** F0, position-only.
- **Validation:** paired bootstrap; Holm (primary), BH (others).
- **Metrics:** top-1, top-5, MRR, pairwise AUC.
- **Output:** `tables/retrieval/m16_*`.
- **Acceptance:** G16-D (gates file).
- **If it fails:** similarity is position-driven → drop the cross-domain claims.
- **Scouting:** comparables ("NFL players whose movement resembles this prospect's Combine").

### E16.B2 — Latent ICC, domain AUC and context-matched retrieval (P1)

- **Why:** History [OBS]: F4 domain AUC 0.997. Finding 2.
- **What:**
  - `DM.run` / `DM.condition_rows` (CEM on length, entry speed, kind, position).
  - Domain AUC and MMD² before/after CEM.
  - Retrieval recomputed on CEM-matched rows.
  - Latent ICC for NFL windows (game or season split).
- **Baseline:** unmatched.
- **Validation:** bootstrap.
- **Output:** `tables/retrieval/cem`, `tables/latent/domain`.
- **Acceptance:** retrieval gain persists after CEM (CI > 0).
- **If it fails:** similarity is explained by context.
- **Scouting:** none directly.

### E16.B3 — Leakage audit (P0)

- **Why:** A representation used as a Combine-only predictor must never see held-out NFL data.
- **What:** For each representation, and per fold:
  - Assert the training-athlete set ∩ held-out = ∅ (`splits.check_split_integrity`).
  - Assert that alignment pairs come from training athletes only.
  - Record `combine_only_inductive` vs `diagnostic_only`.
  - Synthetic test: perturbing held-out NFL sequences must leave held-out Combine embeddings bit-identical.
- **Validation:** assertions.
- **Output:** `diagnostics/leakage_audit.parquet`.
- **Acceptance:** all pass. Otherwise the representation is excluded from M18.
- **If it fails:** exclusion.
- **Scouting:** credibility of any later claim.

---

## M17 — context-aware athletic capability learning

### E17.A1 — M14 event-metric failure diagnosis (P0)

- **Why:** Finding 1 [EXP]: 1/30 passed. ICCs include braking efficiency 0.223, budget −0.014, stop-go 0.082, turn retention −0.106.
- **What:** Per metric × drill:
  - events per trial
  - flag shares (`weak_braking`, `brake_unsustained`, `propulsion_censored`, `exit_censored`, `recovery_censored`, double-dip merges)
  - per-drill vs pooled ICC
  - threshold sensitivity grid: `brake_thresh` 0.5/1.0/1.5, `sustain_frames` 1/2/3, `drop_frac` 0.15/0.25/0.35, each via `M14M.cached_extract` with modified `EventParams` (cache keyed by params hash)
  - event-order consistency in the shuttle (expected 2 turns)
- **Baseline:** M14 defaults.
- **Variables:** event parameters, drill.
- **Validation:** ICC with CIs; BH.
- **Output:** `tables/events_v2/{failure_modes, threshold_grid, per_drill_icc}`.
- **Acceptance:** identifies the dominant failure mode per metric (descriptive).
- **If it fails:** not applicable.
- **Scouting:** which change-of-direction numbers are measurable at all.

### E17.A2 — Context strata and coverage (P0)

- **Why:** Findings 2 and 3.
- **What:** Assign each event an entry-speed bin, deceleration demand, turn-magnitude class (heading-valid only), duration and completeness. Coverage matrix: stratum × domain × position. Boundaries are fixed a priori in `m17/config.py`.
- **Validation:** counts.
- **Output:** `tables/context/coverage`.
- **Acceptance:** a stratum is "supported" if ≥ 30 athletes have ≥ 2 events in both domains.
- **If it fails:** unsupported strata are excluded from cross-domain analysis.
- **Scouting:** where Combine and NFL are comparable.

### E17.A3 — Censoring-aware event measurement (P1)

- **Why:** M14 recovery and exit censoring.
- **What:**
  - Recovery time and propulsion delay as interval or right-censored quantities.
  - Athlete-level Kaplan–Meier median, and a Tobit-style mixed model when `statsmodels` is available.
  - Compare with naive complete-case medians.
- **Validation:** ICC of the censoring-aware vs naive estimates.
- **Output:** `tables/events_v2/censoring`.
- **Acceptance:** the censoring-aware version is adopted if its ICC exceeds naive (paired bootstrap).
- **If it fails:** keep naive with a censoring flag.
- **Scouting:** fair comparison of players whose clips end early.

### E17.B1 — Mixed-effects variance components (P0)

- **Why:** H17.2.
- **What:**
  - Per metric (repaired) and per domain: `y ~ context + (1|athlete)`. Use statsmodels MixedLM (REML); fallback: `CO.backfit` / `CO.components`.
  - σ²_athlete, σ²_residual, athlete share, reliability at k events.
  - Fitted per outer fold on training athletes; pooled read-out on all athletes, labelled descriptive.
- **Baseline:** M14 ICC.
- **Variables:** context adjustment on/off.
- **Validation:** athlete bootstrap.
- **Output:** `tables/variance/components`.
- **Acceptance:** H17.2 holds for a metric if the athlete share is ≥ 0.4 with a CI lower bound ≥ 0.25.
- **If it fails:** noise dominates → no trait.
- **Scouting:** whether a "braking style" exists as an athlete property.

### E17.B2 — Shrunk athlete profiles and the cross-domain RE correlation (P0)

- **Why:** H17.3.
- **What:**
  - BLUP / empirical-Bayes athlete effects (training-fit parameters).
  - Context-adjusted residual profiles.
  - Disattenuated Combine-RE ↔ NFL-RE correlation per family, within position.
- **Validation:** bootstrap; BH.
- **Output:** `tables/signatures/athlete_effects`, `tables/variance/cross_domain_r`.
- **Acceptance:** r CI > 0 for the physically matched family.
- **If it fails:** no transfer for that family.
- **Scouting:** whether Combine braking style is the same thing as NFL braking style.

### E17.C1 — Representation fusion (P2)

- **Why:** H15.5. Track C.
- **What:** Arms: STFT, LSTM, concat, ridge-stacked late fusion, single-head attention, and reliability-weighted fusion. Only for representations that pass G15 and G16-D.
- **Baseline:** concat.
- **Validation:** paired bootstrap; fold wins.
- **Output:** `tables/fusion/*`.
- **Acceptance:** the § 7 adoption rule (master roadmap).
- **If it fails:** keep concat or a single model.
- **Scouting:** none directly.

### E17.D1 — Context-conditioned alignment (P2; Procrustes/CORAL P1)

- **Why:** H17.4.
- **What:**
  - Procrustes and CORAL first.
  - Then InfoNCE with context-matched positives (same stratum, E17.A2) and same-position negatives.
  - Optional: MMD, Sinkhorn OT, DANN (gradient reversal), MIL pooling over events.
- **Baseline:** unaligned; Procrustes.
- **Validation:** E16.B1 metrics plus E15.E3 damage check.
- **Output:** `tables/alignment/*`, checkpoints.
- **Acceptance:** same-position MRR gain over Procrustes CI > 0 and not `destructive`.
- **If it fails:** keep Procrustes or unaligned.
- **Scouting:** none directly.

### E17.E1 — Signature validation and labelling (P0)

- **Why:** Track E.
- **What:** Candidate signatures from B2, C1 or D1. For each:
  - ICC after context adjustment
  - athlete share
  - novelty (`m14.novelty.oof_explained` from M1 + position)
  - held-out split-half
  - association with reliable M16 targets within position
  - correlation with the named physical quantity
- **Validation:** BH.
- **Output:** `tables/signatures/{status, labels}`, `model_cards/*`.
- **Acceptance:** G17.
- **If it fails:** the signature is unvalidated → descriptive or dropped.
- **Scouting:** the vocabulary of the final profiles.

---

## M18 — NFL performance transfer and scouting value

### E18.A1 — Augmentation ladder B0–B7 (P0 for B0–B3)

- **Why:** The ultimate question.
- **What:**
  - Rungs from master roadmap § 6.2, on the same cohort, folds and target, through the official ladder engine (`ae.oof_ladder` + factories, the `M14T.combine_factories` pattern).
  - Embeddings enter as ≤ 8 train-fold PCA dimensions (official ladder convention).
- **Baseline:** B1 (primary), B0 (secondary).
- **Variables:** rung, target, learner.
- **Validation:** paired bootstrap; Holm over the primary family; fold stability; `M14T.transfer_status`.
- **Metrics:** RMSE, ΔRMSE, R², MAE.
- **Output:** `predictions/oof_long`, `tables/ladder/*`.
- **Acceptance:** the M18 success criterion (roadmap § 6.8).
- **If it fails:** precise null or inconclusive (with the MDE).
- **Scouting:** "Should tracking change our ranking beyond the 40?"

### E18.B1 — Position-specific models (P1)

- **Why:** Finding 6 within-position results.
- **What:** Pooled vs partial pooling vs separate (n ≥ 30) per rung.
- **Baseline:** pooled B1.
- **Validation:** within-position concordance; BH.
- **Output:** `tables/position/*`.
- **Acceptance:** CI > 0 in ≥ 3 positions to claim a position-specific benefit.
- **If it fails:** pooled only.
- **Scouting:** position-specific draft boards.

### E18.C1 — Drill-to-task specificity matrix (P1)

- **Why:** Finding 3 hypotheses.
- **What:**
  - Drill features (forty, three-cone, shuttle; M15 time-domain + M17 signatures) × targets T1, T3, T4, T5.
  - Partial association within position, controlling for M1.
  - Pre-declared diagonal contrast: the diagonal effect minus the mean off-diagonal effect.
- **Baseline:** off-diagonal.
- **Validation:** bootstrap of the contrast; BH.
- **Output:** `tables/specificity/*`, heatmap.
- **Acceptance:** the diagonal contrast CI > 0 supports task-specific transfer.
- **If it fails:** general athleticism or position confounding.
- **Scouting:** which drill to weigh for which role.

### E18.D1 — Stopwatch-twin case studies (P1)

- **Why:** Study D.
- **What:**
  - `m14.scouting.stopwatch_twins` on official tests.
  - Select by the largest Δ in a validated feature among pairs closer than the median distance, without outcomes.
  - Profile cards; the outcome is revealed after the prediction.
- **Validation:** sign test and pair bootstrap across all twins (`twin_contrast`).
- **Output:** `reports/athlete_profiles/*`, `tables/scouting/twins`.
- **Acceptance:** cases are illustrative. Only the pooled contrast carries evidence.
- **If it fails:** no differentiation within twins.
- **Scouting:** "Two players with the same 40: does tracking separate them?"

### E18.E1 — Robustness, negative controls and sanity tests (P0)

- **Why:** Study E.
- **What:**
  - Fold stability; season and draft-year sensitivity; complexity ablation; family ablation; missingness; exposure.
  - Negative controls: random embeddings, shuffled Combine→athlete linkage, wrong-drill features.
  - Random-label test (target permuted within position).
  - Refit permutation null with ≥ 200 permutations (`M14T.permutation_null`).
- **Validation:** each control must show ΔRMSE CI including 0.
- **Output:** `tables/robustness/*`, `tables/controls/*`.
- **Acceptance:** a failing control invalidates the corresponding positive result.
- **If it fails:** the result is invalid.
- **Scouting:** trustworthiness.

### E18.F1 — Evidence registry, report and executive summary (P0)

- **Why:** Final deliverable.
- **What:**
  - An evidence table per feature family: measurement status, predictive status, novelty, controls, MDE.
  - Report and executive summary with `‹M18:…›` tokens filled from tables.
  - Publication figures.
- **Validation:** token audit (no unfilled tokens).
- **Output:** `reports/*`.
- **Acceptance:** definition of done (roadmap § 6.10).
- **If it fails:** not applicable.
- **Scouting:** the deliverable for scouts and GMs.
