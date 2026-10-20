# M15–M18 implementation plan

> **Implementation status (2026-10-20, session 4).** Tasks T01–T17 are authored and have not been executed. The current module and notebook map is in `M15_M18_SESSION_HANDOFF.md`, and it supersedes the per-module file lists in § 1–2 where they differ. The differences:
> - M17 uses `events.py` / `contexts.py` / `censoring.py`, where the plan names `event_diagnostics.py` / `context.py`.
> - M18 uses `ladder.py` / `evidence.py`, and `robustness.py` holds the controls.
> - The shared `nflbowl/lab/learners.py` holds the non-ridge learners.
>
> **Earlier note (session 2).** T01 and T04 are authored and have not been executed. Session 2 also added three things this plan did not list:
> - a shared `nflbowl/lab/status.py` (explicit step and gate statuses)
> - `m15/plots.py` and `m16/plots.py`
> - `m16/targets.py`, which holds the T0 and T0x target registry and was brought forward from T06
>
> Two signatures in § 2 changed during implementation:
> - `m15.data.prepare_m15_data(cfg, run, strict)` returns an `M15Data`.
> - `m16.m1_verification.verify_m1(cfg, run, exp, ctx)` replaces the `m16/cohort.py` + `baselines(min)` split of T04.
>
> See `M15_M18_SESSION_HANDOFF.md` for the current API list.

**Status:** plan only. No module listed below exists yet, unless it is marked *existing*. Every interface was written by reading the source. None was verified by execution.

**Policy:** Claude authors and the user executes in Colab. Each task ends with:
- an updated `docs/research/M15_M18_SESSION_HANDOFF.md`
- a "send back" list in the notebook's last cell

---

## 0. Ground rules for the code

1. **Reuse, do not re-implement.** The table below maps each need to the existing API. A new function is written only when no existing API covers the need.

   | Need | Use |
   |---|---|
   | bundle | `nflbowl.data.loading.open_bundle` |
   | folds and splits | `nflbowl.data.splits` |
   | run store | `M14Run` (subclassed) |
   | stage ledger | `LabRun.run_stage` |
   | provenance cache | `LabRun.data_file` / `load_data` |
   | figures | `LabRun.figure` |
   | statistics | `nflbowl.lab.stats` |
   | fair comparisons | `nflbowl.lab.transfer_fairness` |
   | retrieval pools | `nflbowl.lab.retrieval_lab` |
   | geometry | `nflbowl.lab.geometry` |
   | CEM | `nflbowl.lab.domain_matching` |
   | ICC | `nflbowl.lab.reliability` |
   | events and metrics | `nflbowl.lab.m14.*` |
   | dilated TCN | `nflbdb.research.advanced_movement.m11.representation.ConvEncoder` |
   | power / MDE | `nflbdb.research.advanced_movement.transfer_power` |
   | variance components | `nflbdb.research.advanced_movement.context_outcomes` |
   | official ladder | `Experiment._ladder_inputs` / `transfer_predictions` and `ae.oof_ladder` |

2. **Configuration pattern** (as in M13/M14): a frozen dataclass `M1XConfig` plus `resolve(CFG)`. Unknown keys raise. `m13()` returns the equivalent `M13Config` so `nflbowl.lab.handoff` works unchanged.
3. **Run directory pattern:** `M1XRun(M14Run)` with its own `M1X_LAYOUT` and `M1X_FAMILIES`, plus `write_status()`.
4. **Training pattern:** neural training is allowed in M15 and M17 only, behind **two** opt-ins (`run_training=True` **and** `allow_new_training=True`), as in M13. M16 and M18 never train encoders. They adopt checkpoints read-only.
5. **Optional imports** live in one helper, `nflbowl/lab/m15/optional.py`, with `require(name) -> module | None`. A missing module turns the dependent stage into `StageSkipped("blocked: <pkg> not installed")`.
6. **Masking invariant:** every function that consumes `X` takes `M` and returns quantities computed on valid frames only, inside gap-free segments (`m14.events.segments`, `max_gap_dt = 0.15`).
7. **Tests:** `tests/nflbowl/test_nflbowl_m15.py` … `m18.py` use synthetic signals only. The user runs them. They must cover the leakage invariants (roadmap § 2.2).
8. **Version:** bump `nflbdb` to 0.7.0 when M15 code lands. Add a `pyproject` extra `m15 = ["scipy", "scikit-learn", "statsmodels", "PyWavelets", "umap-learn", "lightgbm", "catboost", "POT"]`. Core dependencies stay unchanged.

---

## 1. Package layout

```
src/nflbowl/lab/m15/
  __init__.py        public API listing
  config.py          M15Config, resolve, M15Run, M15_LAYOUT, M15_FAMILIES
  optional.py        lazy optional imports
  data.py            bundle access in physical units, valid segments, athlete-weighted sampling, length-matched cropping
  signals.py         time-domain features, gap diagnostics, coupling, jerk
  spectral.py        Welch / STFT / CWT, spectral features, surrogate + resolution-artifact test
  models.py          PCA/SVD wrapper, STFTCNNAE, RNNAE (LSTM/GRU), TCNAE (reuses ConvEncoder), MaskedTAE, WaveletCNN
  training.py        budget-matched, fold-safe trainer; checkpoint/resume; embedding export
  preservation.py    Track C metrics (valid-frame only), event preservation via m14 detector, robustness
  latent.py          Track D geometry, per-dimension ICC, seed stability, interpretation
  transfer.py        Track E retrieval arms, Procrustes, InfoNCE heads, position-chance baseline, damage check
  legacy.py          audit/adopt prototype STFT/LSTM artifacts with provenance
  gates.py           G15
  figures.py         M15_* figures (each returns fig + data dict)
  reporting.py       report tokens, model cards
src/nflbowl/lab/m16/
  config.py · cohort.py · targets.py · target_reliability.py · baselines.py · registry.py · retrieval.py · leakage.py · detectability.py · gates.py · figures.py · reporting.py
src/nflbowl/lab/m17/
  config.py · event_diagnostics.py · context.py · censoring.py · hierarchical.py · fusion.py · alignment.py · signatures.py · gates.py · figures.py · reporting.py
src/nflbowl/lab/m18/
  config.py · ladder.py · position.py · specificity.py · scouting.py · robustness.py · evidence.py · gates.py · figures.py · reporting.py
scripts/figures/m15_m18_roadmap_figure.py      (authored in this session)
notebooks/nflbowl/M15_SPECTRAL_TEMPORAL_REPRESENTATION_LAB_COLAB.ipynb
notebooks/nflbowl/M16_VERIFIED_BASELINES_CROSS_DOMAIN_VALIDATION_COLAB.ipynb
notebooks/nflbowl/M17_CONTEXT_AWARE_ATHLETIC_SIGNATURES_COLAB.ipynb
notebooks/nflbowl/M18_NFL_SCOUTING_VALUE_COLAB.ipynb
tests/nflbowl/test_nflbowl_m15.py … test_nflbowl_m18.py
```

---

## 2. Interfaces (signatures to implement)

### 2.1 `m15.config`

```python
TRANSFORM_VERSION = "m15_v1"
M15_LAYOUT = ("config", "manifests", "logs", "datasets/cached", "features/signal", "features/spectral", "checkpoints",
              "embeddings/inductive", "predictions/reconstruction", "tables/dataset", "tables/signal", "tables/spectral",
              "tables/preservation", "tables/latent", "tables/reliability", "tables/retrieval", "tables/legacy", "tables/synthesis",
              "figures/png", "figures/svg", "figures/pdf", "figures/data", "reports/markdown", "reports/html", "model_cards",
              "diagnostics", "exports")
M15_FAMILIES = ("dataset", "signal", "spectral", "preservation", "latent", "reliability", "retrieval", "legacy", "synthesis")

@dataclass(frozen=True)
class M15Config:
    experiment_name: str = "m15_spectral_temporal_v1"
    bundle_dir: str = "/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean"
    drive_root: str = "/content/drive/MyDrive/NFLBowl"
    m12_run_dir / m13_run_dir / m14_run_dir: str          # read-only sources
    legacy_m15_dirs: tuple[str, ...] = ()                 # Drive folders of the exploratory STFT/LSTM notebooks
    seed: int = 42
    run_training: bool = False; allow_new_training: bool = False
    architectures: tuple[str, ...] = ("pca", "stft_ae", "lstm_ae")          # MVE; full: + gru_ae, tcn_ae, dtcn_ae, masked_tae, wavelet_cnn
    latent_dims: tuple[int, ...] = (32,)                                     # full: (8, 16, 32, 64)
    seeds: tuple[int, ...] = (20261020,)                                     # finalists: 3 seeds
    train_steps: int = 6000; batch_size: int = 256; lr: float = 2e-3; weight_decay: float = 1e-4
    param_band: float = 0.25
    inner_val_share: float = 0.1
    train_domains: tuple[str, ...] = ("combine", "nfl")                      # ablation: ("combine",)
    stft_nperseg: int = 16; stft_hop: int = 4; stft_phase: bool = False
    welch_nperseg: tuple[int, ...] = (16, 32); detrend: tuple[str, ...] = ("linear", "none")
    bands_hz: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 5.0))
    max_gap_dt: float = 0.15; min_segment_frames: int = 7
    nfl_windows_per_athlete_plot: int = 20
    dropout_rates: tuple[float, ...] = (0.1, 0.2, 0.3)
    event_match_tol_s: float = 0.3
    n_boot: int = 2000; n_boot_screen: int = 500
    gate: dict = {...}                                  # G15 thresholds (gates file)
    figure_formats, figure_dpi, save_* …                # as M14
    def m13(self) -> M13Config; def run_dir(self) -> Path; def training_enabled(self) -> bool
def resolve(cfg: Mapping | None) -> M15Config
class M15Run(M14Run): table(df, name, family), write_status()
```

### 2.2 `m15.data`

```python
def physical(bundle, rows) -> dict[str, np.ndarray]           # wraps m14.events.physical over rows → yd/s, yd/s², rad/s, s, masks
def valid_segments(arrs, max_gap_dt, min_len) -> list[list[tuple[int, int]]]   # per row, via m14.events.segments
def athlete_weighted_rows(meta, per_athlete: int, seed: int, domain: str | None = None) -> np.ndarray
def length_matched_crop(arrs, rows_src, target_lengths: np.ndarray, seed: int) -> dict[str, np.ndarray]
def fold_assignments(bundle) -> pl.DataFrame                  # nfl_id, fold (from bundle.folds); persisted to manifests/fold_assignments.parquet
def game_key(meta) -> pl.Series | None                        # parse from NFL seq_key if possible; None + reason otherwise
```

### 2.3 `m15.signals` and `m15.spectral`

```python
# signals
def validity_table(arrs, meta, P) -> pl.DataFrame            # E15.A1
def time_domain_features(arrs, meta, P) -> pl.DataFrame       # max/q90 speed, max/min a_tan, braking fraction, jerk RMS, …
def domain_contrast_steps(feat, meta, steps=("raw", "athlete_weighted", "length_matched", "cem", "within_kind"), seed, n_boot) -> pl.DataFrame
def coupling_features(arrs, P) -> pl.DataFrame                # E15.A5
# spectral
def welch_psd(x, valid, fs=10.0, nperseg=32, detrend="linear") -> tuple[np.ndarray, np.ndarray] | None
def stft_mag(x, valid, nperseg, hop, phase=False) -> np.ndarray
def cwt_morlet(x, valid, scales) -> np.ndarray
def spectral_features(freqs, psd, bands) -> dict[str, float]  # entropy, centroid, rolloff85, flatness, band energies
def sequence_spectral_table(arrs, meta, cfg) -> pl.DataFrame
def surrogates(x, valid, kind: Literal["ramp", "phase_random", "reversed"], seed) -> np.ndarray
def resolution_artifact_test(spec_tab, surrogate_tab, meta) -> pl.DataFrame   # ρ with length, surrogate AUC, per feature × domain × kind
```

### 2.4 `m15.models` and `m15.training` (torch; imported lazily)

```python
class STFTCNNAE(nn.Module): forward(spec, mask) -> (recon, z)     # magnitude (+ sin/cos phase when stft_phase)
class RNNAE(nn.Module): __init__(cell: Literal["lstm", "gru"], c_in, hidden, d, bidirectional=True)
class TCNAE(nn.Module): wraps m11.representation.ConvEncoder + a light conv decoder
class MaskedTAE(nn.Module)                                           # optional
def build(arch: str, d: int, cfg) -> nn.Module                       # enforces the param band; raises when out of band
def masked_loss(recon, x, mask, heading_valid) -> torch.Tensor       # yaw counted only where heading_valid
class PCAEncoder: fit(X_train, M_train) / transform(X, M) / inverse(Z)   # NumPy; masked mean imputation from training rows
def fit_fold(arch, d, seed, fold, bundle, cfg, run) -> dict           # inner-val early stopping; resume from checkpoints/<…>/fold<k>/
def embed_fold(model, bundle, rows, batch) -> np.ndarray
def export_embeddings(run, arch, d, seed, fold, E, meta, roles) -> Path   # embeddings/inductive/<arch>_d<d>_s<seed>__fold<k>.parquet
def run_grid(cfg, bundle, run) -> pl.DataFrame                       # resumable; one row per (arch, d, seed, fold) with status, steps, params, seconds
```

### 2.5 `m15.preservation`, `m15.latent`, `m15.transfer`, `m15.legacy`, `m15.gates`

```python
# preservation
def valid_frame_errors(x, xhat, mask, hv) -> dict[str, float]
def peak_errors(x, xhat, mask) -> dict[str, float]
def event_preservation(arrs, arrs_hat, P: EventParams, tol_s) -> dict[str, float]   # m14.events.detect_sequence on both
def band_errors(x, xhat, mask, bands) -> dict[str, float]
def degrade(arrs, rate, seed) -> dict[str, np.ndarray]; def truncate(arrs, q, nfl_lengths) -> dict
def preservation_table(run, arch, d, seed, fold, ...) -> pl.DataFrame
def paired_vs_reference(tab, reference="pca", n_boot, seed) -> pl.DataFrame          # uses stats.cluster_bootstrap_diff
# latent
def geometry_table(E, meta, rows, seed) -> pl.DataFrame        # geometry.basic_geometry + stats.effective_rank
def probe_table(E, meta, folds) -> pl.DataFrame                # train-fold logistic/ridge probes: domain AUC, position bal. acc., drill
def dim_icc(E, meta, drill_kinds, n_boot, seed) -> pl.DataFrame  # reliability.icc_components per dimension, held-out athletes
def repeat_attempt_auc(E, meta, kinds) -> pl.DataFrame
def seed_stability(E_by_seed) -> pl.DataFrame                  # orthogonal Procrustes then r
def interpretation(E, feats, meta) -> pl.DataFrame             # stats.partial_spearman vs F0/M14/spectral, BH
# transfer
def athlete_means(E, meta, domain) -> pl.DataFrame
def procrustes_fit(Zc_train, Zn_train) -> np.ndarray; def infonce_head(...)       # training athletes only
def retrieval_scores(arm, Zc, Zn, meta, fold, pools) -> pl.DataFrame              # retrieval_lab.pool_scores
def position_chance(meta_heldout) -> pl.DataFrame                                 # E15.E2 analytic + simulated
def damage_check(aligned, unaligned) -> pl.DataFrame
# legacy
def find(cfg) -> list[Path]; def audit(path, bundle) -> dict   # schema, n rows, athlete list, fold overlap → adoptable_inductive | transductive_unknown | unreadable
# gates
def g15(pres, latent, rel, stab, robust, cfg) -> pl.DataFrame   # one row per (representation, criterion)
```

### 2.6 `m16` (no neural training)

```python
# cohort
def m1_contract(bundle) -> dict                                  # columns present, coverage, units audit
def official_tests_present(athletes) -> list[str]
def cohort_funnel(bundle, exploratory_ids: set[int] | None) -> pl.DataFrame
# targets
TARGETS: dict[str, TargetSpec]                                   # T0..T6 (unit, aggregation, min_exposure, censoring, source)
def window_table(bundle, cfg) -> pl.DataFrame                    # per NFL window: athlete, kind, season, game_key?, max speed, band accels, …
def nfl_event_table(run, exp, P) -> pl.DataFrame                 # m14.metrics.cached_extract restricted to domain == "nfl"
def athlete_targets(windows, events, folds, cfg) -> pl.DataFrame # demand-mix weights fitted per fold on training athletes
# target_reliability
def split_reliability(windows, target, split: Literal["game", "season", "random"], n_boot, seed) -> pl.DataFrame
def exposure_curves(...); def context_share(...); def freeze(run, rows) -> Path   # immutable manifests/target_freeze_m16.json
# baselines
def learner_factories(cfg) -> dict[str, Callable]                # ridge (official), enet, lgbm, catboost (optional)
def run_ladder(exp, data, target, models, cfg) -> pl.DataFrame   # ae.oof_ladder + factories; POS/MEAN as in M13
# registry
def to_long(oof, experiment_id, target) -> pl.DataFrame          # transfer_fairness.long_predictions
def evaluation_registry(long, cohorts, pairs, n_boot, seed) -> pl.DataFrame
# retrieval / leakage / detectability
def pools(athletes, seed) -> dict                                 # retrieval_lab.pool_definitions + stopwatch-twin pool
def run_retrieval(representations, bundle, pools, n_boot, seed) -> pl.DataFrame
def leakage_audit(representations, bundle) -> pl.DataFrame
def mde(target_frame, oof_m1, fractions, cfg) -> pl.DataFrame     # transfer_power
def g16(...) -> pl.DataFrame
```

### 2.7 `m17` and `m18`

```python
# m17
def failure_modes(events, seqtab) -> pl.DataFrame
def threshold_grid(run, exp, base: EventParams, grid) -> pl.DataFrame          # cached_extract per params hash
def strata(events, cfg) -> pl.DataFrame; def coverage(strata) -> pl.DataFrame
def censored_median(times, censored) -> float                                  # Kaplan–Meier
def mixed_model(df, metric, context_cols, train_ids) -> dict                   # statsmodels MixedLM | context_outcomes.backfit
def blups(fit, df_heldout) -> pl.DataFrame
def cross_domain_r(eff_c, eff_n, rel_c, rel_n, n_boot, seed) -> pl.DataFrame
def fusion_arms(views, folds, cfg) -> dict; def alignment_arms(...) -> dict
def signature_status(sig, targets, cfg) -> pl.DataFrame; def g17(...)
# m18
RUNGS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")
def eligible_rungs(g15, g16, g17) -> dict[str, str]                            # rung → eligible | blocked(reason)
def ladder(exp, data, rungs, target, cfg) -> pl.DataFrame
def position_arms(...); def specificity_matrix(...); def twin_cases(...)
def controls(...) -> pl.DataFrame                                              # random emb., shuffled linkage, wrong drill, random label
def evidence_table(...) -> pl.DataFrame; def g18(...)
```

---

## 3. Task list (ordered; each task is small enough for one session)

Legend: **[C]** CPU only · **[G]** GPU recommended. "Send back" is the minimum the user returns.

| # | Task | Files | Depends on | Send back |
|---|---|---|---|---|
| T01 | M15 config, store, optional imports, legacy audit, data helpers | `m15/{config, optional, data, legacy}.py`, tests | — | — |
| T02 | M15 notebook § 0–A (setup, E15.00, A1, A2) **[C]** | notebook, `m15/signals.py`, `m15/figures.py` | T01 | `stages.json`, `tables/dataset/*`, `tables/legacy/prototype_audit.csv`, `tables/signal/domain_contrast_steps.csv` |
| T03 | Spectral module and artifact test (E15.A3/A4) **[C]** | `m15/spectral.py`, notebook § A3–A4 | T02 | `tables/spectral/artifact_test.csv`, `reliability.csv` |
| T04 | M16 cohort + M1 reproducibility + E15.E2 (P2 critical path; parallel with T02) **[C]** | `m16/{config, cohort, baselines(min), registry}.py`, `m15/transfer.position_chance`, M16 notebook § 0–A1 | T01 | `manifests/m1_contract.json`, `tables/baselines/m1_reproducibility.csv`, `tables/retrieval/position_chance.csv` |
| T05 | M17 event diagnosis (E17.A1/A2) (P3; parallel) **[C]** | `m17/{config, event_diagnostics, context}.py`, M17 notebook § 0–A | M14 artifacts | `tables/events_v2/*`, `tables/context/coverage.csv` |
| T06 | M16 targets + reliability + MDE (E16.A2/A3/A6) **[C]** | `m16/{targets, target_reliability, detectability}.py` | T04 | `tables/target/*`, `tables/detectability/mde.csv`, `target_freeze_m16.json` |
| T07 | M15 models + trainer (MVE: PCA, STFT-AE, LSTM-AE, d=32, 1 seed) **[G]** | `m15/{models, training}.py`, notebook § B | T02 | `tables/preservation/grid.csv`, training log, timing |
| T08 | M15 preservation, latent, retrieval (C1, C2, D1, D2, E1, E3) + G15 **[G/C]** | `m15/{preservation, latent, transfer, gates}.py`, notebook § C–E, synthesis | T07 | `tables/synthesis/m15_gate.csv`, `tables/retrieval/paired.csv`, report |
| T09 | M16 baseline ladder + within-position (E16.A4/A5) **[C]** | `m16/baselines.py` (full), `m16/registry.py` | T06 | `tables/registry/evaluation_registry.csv` (+ G16-A/B/C) |
| T10 | M16 retrieval + leakage (E16.B1–B3) + G16 **[C]** | `m16/{retrieval, leakage, gates}.py` | T08, T09 | `tables/synthesis/m16_gate.csv` |
| T11 | M15 full grid (optional; all arch × d × 3 seeds) **[G]** | config only | T08 | updated G15 |
| T12 | M17 censoring + hierarchical + cross-domain r (A3, B1, B2) **[C]** | `m17/{censoring, hierarchical}.py` | T05, T06 | `tables/variance/*` |
| T13 | M17 fusion / alignment (C1, D1; conditional on G15, G16-D) **[G]** | `m17/{fusion, alignment}.py` | T10 | `tables/fusion/*`, `tables/alignment/*` |
| T14 | M17 signatures + G17 **[C]** | `m17/{signatures, gates}.py` | T12 (T13) | `tables/synthesis/m17_gate.csv` |
| T15 | M18 ladder B0–B3 + controls (MVE) **[C]** | `m18/{ladder, robustness, gates}.py` | T09 | `tables/ladder/*`, `tables/controls/*` |
| T16 | M18 B4–B7, position, specificity, cases **[C]** | `m18/{position, specificity, scouting}.py` | T10, T14, T15 | tables + profiles |
| T17 | M18 evidence, report, executive summary, publication figures **[C]** | `m18/{evidence, reporting, figures}.py` | T16 | final report, token audit |

**Parallelism.** T02/T03, T04/T06/T09 and T05/T12 can proceed in any interleaving. T07–T08 need a GPU session. M18 MVE (T15) needs only T09.

---

## 4. Colab execution instructions (for every M15–M18 notebook)

1. **Runtime.** GPU (T4 or better) for M15 § B and M17 § D. CPU is enough elsewhere. Mount Drive.
2. **Install:**
   ```
   !pip -q install "git+https://github.com/mybenkhadda/NFL_bowl.git@main"
   !pip -q install scipy scikit-learn statsmodels PyWavelets umap-learn lightgbm catboost POT
   ```
   The second line is optional; missing packages block only their stages. Confirm `nflbowl.__version__` in Setup 2.
3. **Configure.** Edit the `CFG` dict in Setup 3 only. Unknown keys raise. Leave `run_training=False` unless running M15 § B or M17 § D, and then also set `allow_new_training=True`.
4. **Run all.** Every section is a `run.run_stage(...)`. A failure writes `logs/<stage>_traceback.txt` and later independent stages still run.
5. **Resume after a disconnect.** Re-run Setup 1–4, then *Run all*. Cached stages report `reused`. This relies on provenance-matched `load_data`, `manifests/stages.json` and per-fold checkpoints with `history.parquet`. To force recomputation of one stage, set `CFG["extra"]["force"] = ["<stage>"]`.
6. **Reload cached datasets elsewhere:**
   - `M15Run(cfg.run_dir).load_data("features/spectral/welch_n32", provenance={...})`
   - `run.frame(name, family)`
   - embeddings: `pl.read_parquet(run_dir/"embeddings/inductive/<arch>_d<d>_s<seed>__fold<k>.parquet")`
7. **Send back:** the list in the notebook's last cell, plus `manifests/stages.json` and every `logs/*_traceback.txt`.
8. **Never** point `experiment_name` at an M12/M13/M14 directory. `resolve` refuses names that start with `m12_`, `m13_` or `m14_`.

---

## 5. Output conventions

- **Tables:** `tables/<family>/<name>.parquet` + `.csv`, with a provenance line in `manifests/table_provenance.jsonl`.
- **Figures:**
  - `figures/{png,svg,pdf}/M1X_<section>_<nn>_<slug>` and `figures/data/<same>__<part>.{csv,parquet}`.
  - Every figure has a `question` and a `caption`.
  - Colour-blind-safe palette, consistent across milestones.
- **Fingerprints:**
  - `manifests/dataset_fingerprint.json` (bundle fingerprint, science hash).
  - Per model, `model_cards/<name>.md` + `checkpoints/.../config.json` (with the architecture hash and the training-athlete hash per fold).
- **Reports:** `reports/markdown/m1X_report.md` + HTML. Numbers are filled from tables via `‹M1X:…›` tokens, and a token audit runs at the end.
- **Pseudonyms:** athlete IDs are pseudonymised in profiles (`stats.pseudonym`, salt in config, not in reports).
