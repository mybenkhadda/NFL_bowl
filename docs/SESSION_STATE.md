# Session state — NFL Big Data Bowl 2027

## ▶ COMPACT CHECKPOINT (2026-10-15, Milestone 12 authored) — read ONLY this section, then `docs/ACTIVE_TASK.md`

- **Rules.** Code authoring only: Claude runs nothing (no shell, no Python); the user runs notebooks. Numbers: read saved tables under `reports/tables/advanced_movement/advanced_v1/`, never this file.
- **Executed by the user:** Milestones 00–11 (11F / 11H / 11I not executed). 11B candidate findings (`forty.sprint_vmax → env.lin_V0`, `forty.v_max_obs → nfl.vmax_p90`, Δ CI > 0) remain exploratory. **12.0 executed once (2026-10-15T00:32Z):** 401 linked athletes, but 0 shared speed bins → fixed (D-12-1, see below); 3 synthetic checks failed → fixed (two test bugs; one real 12B standardization bug in RC / SIMEX).
- **Milestone 12 — AUTHORED 2026-10-15.** Package `src/nflbdb/research/advanced_movement/m12/` (config, advanced_evaluation, foundation, smoothing, distributional_capacity, measurement_error, functional_movement, multiview_traits, movement_transport, shared_movement_encoder, synthesis, figures, reporting, validation); `artifacts.ResearchStore` (area `advanced_v1`, reports `reports/milestone12`); notebooks `12_00`, `12c`, `12b`, `12a`, `12d`, `12e`, `12f`, `12g` in `notebooks/advanced_movement/`; 12 docs in `reports/milestone12/` with generated `M12:RESULTS` blocks; `tests/test_milestone12_advanced.py`.
- **Design fixed before results:** NFL observations = `m11_envelope_hist` (no tracking rescan except 12F windows); 12C target estimator frozen by a reliability rule on NFL data only (`m12c_target_freeze.json`, immutable); ladders score every model on identical athletes with fold-fitted blocks (`Model.factory`); Holm for primaries, BH exploratory; inductive vs transductive kept apart; 12D representation chosen by likelihood + reliability (`m12d_selected_representation.json`).
- **Next action:** USER re-runs 12.0 (rebuilds Combine curves under `rising_phase_interp_v2` automatically), then 12C → 12B → 12A → 12D → 12E → (12F optional) → 12G; sends back each notebook's "Send back" items.

---

## Previous checkpoint (2026-10-14, Milestone 10 authored) — history

- **Rules.** Code authoring only: Claude runs nothing (no shell, no Python); the user runs notebooks and launches the app. Numbers: read saved tables / `reports/milestone09/EXECUTED_NUMBERS.md`, never this file.
- **Executed by the user:** Milestones 00–08B and **09A–09G (full; mart built 2026-10-14T13:13Z, 510 prospects, 3,266 twin pairs, 23 cases; `EXECUTED_NUMBERS.md`, `CASE_STUDY_GALLERY_EXECUTED.md` written)**. Frozen: no tracked Combine feature established incremental NFL prediction (C08, C14).
- **Milestone 10 — final product + submission — AUTHORED 2026-10-14, NOT EXECUTED.** Notebooks `10a`…`10e` in `notebooks/advanced_movement/`; modules `product_kit`, `movement_gap`, `decision_simulator`, `signature_stories` (stdlib, shared with app), `submission_builder`, `final_figures`, `milestone10_validation` (11 checks); `artifacts.ProductStore` (area `product_v1`, reports `reports/milestone10`). App: 10 pages in 5 groups (new: Signature Stories, NFL Movement Gap, Decision Simulator); no new mart table. Docs in `reports/milestone10/` carry `‹M10:…›` tokens rendered by 10E (templates kept under `templates/`). Tests `tests/test_milestone10_product.py`, `tests/test_cil_app_wiring.py`.
- **Design fixed before running:** filters only narrow the stored 0.03 s twin rule; 5 presentation sprint phases over the 8 09B features; coverage class by registry analogue + declared start-state table + 09A support + 09B comparison; gap focus = largest not-measured exposure share among positions with ≥ 30 athletes per supported cell; WR story = 6 criteria, McConkey/Bond only if all pass; decision simulator deterministic, 6 sections × 5 evidence categories.
- **Next action:** USER runs 10A → 10E (full), launches the app, walks `reports/milestone10/APP_BROWSER_CHECKLIST.md`; sends back failures, the 10B selection lines, the 10E token audit. Claude fixes and finalizes the documents.

---

## Previous checkpoint (2026-10-14, Milestone 09 authored) — history

- **Rules.** Code authoring only: Claude runs nothing (no shell, no Python); the user runs notebooks in Jupyter and launches the app. Executed `.ipynb` are large — grep them. Numbers: read saved tables, never this file.
- **Executed by the user:** Milestones 00–08B (08B.0–08B.6 outputs present 2026-10-14: registry `frozen_exploratory` (FO01.S0 primary), Gate E PASS, Gate F PASS, 20/20 claims executed — `reports/tables/advanced_movement/football_v1/final_evidence_registry.csv`). Frozen: no tracked Combine feature established incremental NFL prediction (C08, C14); speed development and TCN-B are repeatable beyond official tests (C17).
- **Milestone 09 — Combine Intelligence Lab — AUTHORED 2026-10-14, NOT EXECUTED.** Notebooks `09a`…`09g` in `notebooks/advanced_movement/`; modules `movement_demands.py`, `movement_matching.py`, `stopwatch_twins.py`, `scouting_interpretation.py` (stdlib only, shared with the app), `milestone09_artifacts.py` (mart + `validate_mart`), `intelligence_figures.py`, `intelligence_validation.py` (10 checks); `artifacts.IntelligenceStore` (area `intelligence_v1`, `mart_dir`, `reports_dir = reports/milestone09`). App `apps/combine_intelligence_lab/` (Streamlit + Plotly, 7 pages, reads `…/intelligence_v1/app_mart/v1` or `CIL_MART_DIR`). Reports `reports/milestone09/` (storyboard, narrative, added value, technical doc, gallery design). Test wrapper `tests/test_advanced_movement_intelligence.py`.
- **Design fixed before running:** 9 NFL demand rules (`DemandConfig`) on 05A eligible plays, edge episodes excluded from profiles; shared dims = band accelerations 2–4/4–6/6–8 yd/s (same `band_accel` both sides) + top speed; coverage = share of position NFL episodes ≤ Combine value with SE interval; non-forty demands abstain; twins = same position, forty within 0.03 s; verdict on SE of the difference; 10 case types, pre-draft, type 10 uses NFL exposure COUNT only (flagged).
- **Assumptions not verified:** kinematics columns (`yaw_rate`, `heading_valid`, `x_s`…); `nfl_play_kinematics` columns; vendor route/coverage column names (auto-detected by substring); Streamlit ≥ 1.39 APIs. Full list: `reports/milestone09/TECHNICAL_DOCUMENTATION.md`.
- **Next action:** USER runs 09A → 09G (same `RUN_MODE`), launches the app; sends back each notebook's "send back" items + `reports/milestone09/EXECUTED_NUMBERS.md`. Claude fixes failures and fills `RESEARCH_NARRATIVE.md` § 5 from executed numbers only.

---

## Previous checkpoint (2026-10-13, Milestone 08B authored) — history

- **Rules.** Code authoring only: Claude runs nothing; the user runs notebooks in Jupyter. Executed `.ipynb` files are large — grep them. Smoke and full outputs are separate. Numbers: read the saved tables, never this file.
- **Executed by the user:** Milestones 00–07, 08A, 08A.2 (full). Frozen conclusions: tracked speed is repeatable and largely redundant with the stopwatch; TCN-B is a repeatable fingerprint; **no feature has established incremental NFL movement prediction** (Milestone 06, case E); NFL movement is dominated by context; the 07D scorecard is descriptive.
- **08A.2 result (verified in `reports/tables/advanced_movement/scouting_v1/football_outcome_screen.csv`):** `FO01.S0` separation provisional (reliability 0.74, 91 linked athletes, semantics `plausible_inferred`); S1 same numbers as S0; S2 0.52; `FO12.T1` rejected; `FO12.T2`, `FO13.K0`, `FO13.K2` provisional; no decision was recorded in 08A.2 (its registry `football_outcome_targets_v1.json` does not exist and is **superseded** by the 08B.0 registry).
- **Milestone 08B — AUTHORED 2026-10-13, NOT EXECUTED** (no check, no smoke run, no full run, no figure). Seven notebooks in `notebooks/advanced_movement/`:

  | Notebook | Objective | Gate |
  |---|---|---|
  | `08b0_outcome_resolution_and_freeze` | lineage, source classes, **manual decision** → `football_outcome_registry_<version>.json` | A |
  | `08b1_football_outcome_reliability` | game-split reliability, context, persists `football_outcome_rows` | B |
  | `08b2_football_transfer_benchmark` | M0–M6 on the linked football cohort, resumable refit permutation | C |
  | `08b3_transfer_robustness_and_detectability` | declared sensitivities, planted-signal simulation (07C machinery) | — |
  | `08b4_incremental_movement_information` | cross-fitted stopwatch adjustment, residual repeatability, evidence registry | D |
  | `08b5_scouting_decision_intelligence` | abstention, pairwise uncertainty, cases, football panel | E |
  | `08b6_final_research_evidence_package` | evidence registry, report tokens, figure index, manifest | F |

- **Modules (`src/nflbdb/research/advanced_movement/`):** `football_registry.py`, `football_reliability.py`, `football_transfer.py`, `movement_information.py`, `scouting_intelligence.py`, `evidence_package.py`, `football_figures.py`, `football_validation.py` (21 checks in four groups); `artifacts.FootballStore` (area `football_v1`, `.scouting` reads 08A / 08A.2). Test wrapper `tests/test_advanced_movement_football.py`.
- **Branches (computed from the registry, never from a flag):** `frozen` (confirmed sources) · `exploratory` (inferred semantics + `ACKNOWLEDGE_PROVISIONAL`; every result labelled exploratory) · `blocked` (no decision, `STOP`, or `PROVISIONAL` without acknowledgement). 08B.1–08B.3 raise `FootballTransferBlocked` when blocked; 08B.2 also needs Gate B = PASS for the primary; 08B.4–08B.6 run in every branch.
- **Design fixed before running:** cohort = Milestone 06 paired cohort ∩ athletes supporting the outcome (≈ 91, not 284), Milestone 06 folds and fold-fitted representations reused (no encoder trained); primary comparison `M2` vs `M1` and secondary `M5` vs `M1` stored in the registry; ridge with 5 penalties; permutation modes `linkage` (primary, 500 refits) and `outcome` (leak check); sensitivities never replace the primary.
- **Reports (authored):** `reports/NFL_BDB_2027_{FOOTBALL_TRANSFER_RESEARCH, SCOUTING_VALUE, METHODS_AND_LIMITATIONS}.md` hold `‹Cnn: …›` claim tokens: verified numbers for Milestones 04–08A.2, `pending` for 08B. 08B.6 refreshes them in place and writes the executive summary, evidence registry, figure index, gallery, decision log.
- **Assumptions not verified by execution:** `athlete_registry` holds `cmb_forty`, `cmb_ten_yd_split`, `cmb_combine_weight`, `cmb_combine_height` (fallback: `atlas.official_table`); the 06B representation table has `scheme · family · fold · seq_id · nfl_id · dim · value · role` with `role = 'train'` for training athletes; 07D percentile tables are `combine_speed_percentiles` and `combine_speed_development_percentiles`; `store.write` returns the frame; Polars calls (`replace_strict`, `with_row_index`, full join); check tolerances (set by reasoning; `[draw]` assertions rest on a fixed seed).
- **Open scientific points:** vendor definition of `separation_at_pass_forward` (no data dictionary in the repository); `player_get_off`; handedness of the Combine frame; units of official measurements.
- **Next action:** USER runs 08B.0 (fill evidence only from an authoritative source; record `DECISION`), then 08B.1 → 08B.6 in order; send back the items listed in the last cell of each notebook. Claude fixes failures and transcribes the decision to `docs/RESEARCH_DECISIONS.md` (D-08B0-1).
- **Housekeeping pending:** everything below this section is legacy history (≈ 40k tokens). Move it to `docs/archive/` by hand so this file is ≤ 800 words.

---

## Previous checkpoint (08A.2) — history, do not read by default
- **Current milestone: 08 (football-relevant transfer + scouting decision framework), phases A–E, stop after each.** 08A EXECUTED by the user (full; gate PASS, branch continue; defensible by the screen: FO01 separation, FO12 tackle, FO13 sack — tables in `reports/tables/advanced_movement/scouting_v1/`). **Phase A = notebook 08A.2 (outcome semantics + opportunity validation) AUTHORED 2026-10-13, NOT executed.** Phases B–E (08B transfer, 08C incremental information, 08D scorecard decisions, 08E package) not started; each starts only on the user's instruction. 08B is gated on `opportunity_validation.require_frozen` (registry `football_outcome_targets_<version>.json`, written only by an explicit `DECISION = 'FREEZE'`).
- **08A.2 files:** `outcome_semantics.py`, `opportunity_validation.py`, `outcome_validation_figures.py`, `scouting_validation.CHECKS_08A2` (10 checks), notebook `08a2_outcome_semantics_and_opportunity_validation.ipynb`. Facts read from the 08A tables that shaped the design: `tackle` is recorded on ~4 % of rows with values 0 / 1 (credited rows only); `sack` is recorded with explicit zeros on ~57 % of DL rows (same rows as the coverage label); separation is recorded for 85 % of non-targeted and 100 % of targeted receiver rows; tracking and `player_play` hold the project athletes only (nearest defender not recomputable); `pass_result` is absent from silver `player_play` (taken from the 05A play table).
- **Milestones 00–07: executed by the user** (stated by the user 2026-10-13; the "07 NOT executed" lines below are history). Conclusions: tracked speed is repeatable and largely redundant with the stopwatch; TCN-B repeat-attempt retrieval is strong; **no feature has established incremental NFL predictive validity** (Milestone 06 case E); athlete share of NFL window variance is small; scorecard 07D is descriptive only. Exact numbers: read the saved tables, never this file.
- **Frozen IDs:** NFL targets `nfl_movement_targets_v1` (`8ff9fc191616711a`): `O2.accel_sust05_mean` (primary), `O3.band_acc_v0_2_mean`, `O1.speed_p90_mean`. Combine candidates `transfer_candidates_v1`: `F2.forty.p95_speed`, `F6.forty.time_v4_to_v8_s`, `FPCA_A_K2`, `TCNB_A`, `TCNB_A|pc4`. Decisions: `docs/RESEARCH_DECISIONS.md`.
- **08A files (unchanged):** `scouting_feasibility.py`, `ScoutingStore` (area `scouting_v1`) in `artifacts.py`, notebook `08a_football_outcome_feasibility.ipynb`, `tests/test_advanced_movement_scouting.py` (runs `ALL_CHECKS` = 08A + 08A.2).
- **Rules:** code authoring only (Claude runs nothing; the user runs notebooks in Jupyter); executed `.ipynb` files are large — grep them, do not open them; smoke and full outputs are separate.
- **Next action:** USER runs 08A.2 (full; smoke only if smoke outputs of 05–06 and 08A exist), reviews sections 1–10, records `DECISION` in section 11, sends back the items listed in the last cell. Then Phase B on instruction, on the frozen registry only.
- **Housekeeping pending:** everything below this section is legacy history (≈ 40k tokens). Move it to `docs/archive/` by hand; Phase D rewrites this file to ≤ 800 words.

---

# LEGACY HISTORY — do not read by default

_Last updated: 2026-10-13 · **Milestone 06 (notebooks 06A + 06B + 06C: Combine → NFL transfer validation) AUTHORED — AWAITING USER EXECUTION; blocked at 06A until the NFL targets are frozen in 05B (decision still PENDING). Milestone 05 executed by the user. Milestones 07–08 not started** · Advanced movement programme: Milestones 1–3 (notebooks 00–03) executed by the user (smoke + full); Milestone 4 (notebooks 04A + 04B: feature atlas, robust TCN validation, frozen shortlist) AUTHORED — AWAITING USER EXECUTION** · Combine → NFL DL get-off executed by the user (report generated 2026-10-11T12:51Z, definition still unresolved) · sprint reliability rescue executed · Phase 04 Q / B / C1 / V executed · Phase 05 P0 + P0b executed. Code-authoring only: Claude executes nothing._

## ▶ Advanced movement — Milestone 07 checkpoint: football-context transfer diagnosis + scorecard (2026-10-13) — read this first
- **Status: notebooks 07A, 07B, 07C, 07D AUTHORED, NOT executed** (no synthetic check, no smoke run, no full run, no figure). `reports/FOOTBALL_CONTEXT_TRANSFER_AND_SCORING_RESULTS.md` is a template with `‹…›` placeholders (its section 2 holds the executed Milestone 06 numbers). **The status line at the top of this file and the "NOT executed / BLOCKER" lines of the Milestone 06 section below are history.**
- **Milestones 00–06 are executed by the user (full mode).** Milestone 05: targets FROZEN 2026-10-13T11:09Z (`nfl_movement_targets_v1.json`, version hash `8ff9fc191616711a`): primary `O2.accel_sust05_mean` (cell_residual, reliability 0.932), secondaries `O3.band_acc_v0_2_mean`, `O1.speed_p90_mean`. Frozen Combine candidates (`transfer_candidates_v1`): C2 `F2.forty.p95_speed`, C3 `F6.forty.time_v4_to_v8_s`, C5 `FPCA_A_K2`, C6 `TCNB_A`, C7 `TCNB_A|pc4`.
- **Actual Milestone 06 findings (read from `reports/tables/advanced_movement/transfer/`):** paired cohort 284 athletes; M1 official RMSE 0.1794 (R² 0.947, almost all position: POS 0.1815); M5 TCN-B vs M1 Δ RMSE +0.0003 [−0.001, +0.002] (0.16 %), outcome-permutation p 0.169, linkage p 0.234; M2 / M3 / M4 / M6 / M7 intervals all include 0; matched prospects (833 pairs, 275 athletes) no model beats official; readiness registry: no `predictive_validated`, no `predictive_candidate`; decision case **E**. **No validated incremental NFL prediction exists.** These conclusions are frozen; Milestone 07 does not replace them.
- **Milestone 07 design (fixed before running):**
  - 07A — nested variance components (athlete / game / window, method of moments, athlete bootstrap), context definitions C0 raw · C1 position z · C2 starting-speed band · C3 position × alignment · C4 hierarchical backfit + shrunk athlete effect; every context parameter from the `fit` windows only (training athletes of a fold; the other game half in a reliability split); support of a cell decided by counts, never by values. C1–C4 are EXPLORATORY (`nfl_context_target_registry.json`).
  - 07B — EXPLORATORY registry `physical_transfer.HYPOTHESES` (18 pooled + 3 by-position rows; primaries H1 M1 vs M0 on NFL p90 speed · H2 M2 vs M1 on `band_acc_v4_6` · H3 M5 vs M1 on C4 of `accel_sust05`), persisted before evaluation; Milestone 06 folds, fold representations (read by name, no encoder trained), ridge ladder + `S0` (M0 + SPEED2) and `E0` (M0 + ENG); Holm over primaries; matched pairs = the frozen 06C pairs.
  - 07C — synthetic targets on the real structure (`y* = μ_M1,oof + σ_res,pos (√(1−f) ε + √f z)`, z = leading direction of the candidate block, conditional or raw), Milestone 06 pipeline and criterion, checkpointed by configuration hash (`RESUME_MODE`), power / null / coverage / MDE / by position / cohort-size scenarios.
  - 07D — descriptive scorecard, four sections never combined: A speed capacity (headline official forty), B speed development (`time_v4_to_v8_s`), C fingerprint similarity (06B fold encoders, inductive query fold), D evidence confidence; oriented mid-rank percentiles, leave-one-out + fold-isolated, intervals = repeat-attempt SE + reference bootstrap; missing → abstain; no NFL input (`assert_no_nfl_columns`).
- **New files in `src/nflbdb/research/advanced_movement/`:**

  | File | Content |
  |---|---|
  | `context_outcomes.py` | `ContextConfig`, `Progress`, `Windows` (`from_engine`, `make`), `athlete_stats` / `components` / `shrink_factor`, `backfit`, `ContextLabels` (`adjust`, `values`, `label_fn`), tables DIST · VARIANCE · SCALE · OVERLAP · ADJUSTED, `rank_shift_examples`, `target_registry` |
  | `context_reliability.py` | `game_halves`, `pair_stats`, `reliability_table` (cross-fitted), `exposure_table`, `tradeoff_table`, `gate_07a`, `assert_gate`, `conclusions_07a`, shared `GATE` contract |
  | `physical_transfer.py` | `OUTCOMES`, `HYPOTHESES`, `hypothesis_table`, `models_for`, `run_outcome`, `evaluate` (+ `holm`), `subgroup_eval`, `pair_concordance`, `association_table`, `unit_audit`, `evidence_vs_m06`, `gate_07b`, `interpretation_markdown` |
  | `transfer_power.py` | `PowerConfig`, `build_structure`, `planted_direction`, `synthetic_target`, `replicate_rows`, `resampled_cohort`, `run_simulation` (resumable), `scenario_table`, `null_table`, `mde_table`, `attenuation`, `redundancy_table`, `limits_table`, `gate_07c`, `report_markdown` |
  | `context_figures.py` | 12 + 11 + 9 figures (`AM07A_*`, `AM07B_*`, `AM07C_*`) |
  | `context_validation.py` | 21 synthetic checks (`CHECKS_07A` 9, `CHECKS_07B` 6, `CHECKS_07C` 3, `CHECKS_07D` 5; two shared) |
  | `scoring/` | `percentiles.py`, `uncertainty.py`, `fingerprint.py`, `scorecards.py`, `figures.py` (7 figures `AM07D_*`) |
  | `artifacts.py` | + `ContextTransferStore` (sub-area `context_transfer`), `ScoringStore` (`scoring`), `AdvStore.require_named` |

  Also new: the four notebooks, `tests/test_advanced_movement_context.py`, the report template; README updated.
- **Reusable artifacts (read only):** 05A `nfl_play_movement`, `nfl_phase_space` · 05B frozen registry · 06A `transfer_outer_folds`, `transfer_cohort_manifest`, `transfer_input_contract.json` · 06B `transfer_attempt_representations_outer`, `transfer_common_support`, `transfer_oof_predictions`, `transfer_paired_comparisons` · 06C `transfer_matched_pairs`, `transfer_attempt_representations_seeds` · 04A `atlas_attempt_features`, `feature_dictionary`, `atlas_feature_status` · 00–02 `athlete_registry`, `sequence_frames`, `sequence_window_flags`.
- **Execution (USER, in Jupyter; no terminal command):** restart → 07A → 07B → 07C → 07D (smoke only if smoke outputs of 05–06 exist) → restart → same order in `'full'` → fill the report template. Steps: `notebooks/advanced_movement/README.md`.
- **Outputs:** `{reports/tables, reports/figures, warehouse/gold}/advanced_movement/{context_transfer, scoring}/` (smoke: `…/_sample/…`).
- **Blockers / open points:**
  - Nothing was executed. First failures are most likely in the synthetic checks (tolerances set by reasoning) or in a Polars call; send the table and tracebacks.
  - Assumptions made by source reading, not verified by execution: `S2_signal = True` means "no signal-quality flag"; `gain05_v0` is the starting speed of the measured 0.5 s window; representation family names are `ENG · SPEED2 · FPCA · TCN · RAND`; seed-replicate embeddings (if any) share the layout of the outer table; NFL sampling interval passed as 100 ms.
  - 07C run time is unmeasured (about 6,800 replicates at the full budget); the notebook prints an estimate from the first replicate.
  - C4 is a method-of-moments / backfitting approximation of a mixed model (no scikit-learn / statsmodels in the project); truncation and non-convergence are reported, not hidden. Alignment is athlete-level for many athletes, so C3 / C4 can remove real athlete differences.
  - 07B comparisons were declared after the Milestone 06 result: exploratory whatever they show.
  - Still open from before: `player_get_off` definition, handedness of the Combine frame, units of official measurements.
- **Next research decision (after the user's executed Milestone 07):** (a) a physically aligned relationship is a credible hypothesis → plan an independent-cohort confirmation, scorecard stays descriptive; (b) inconclusive at the detectable effect size → state the cohort size needed, keep descriptive profiles; (c) negative within the detectable range → competition narrative = measurable, repeatable, honestly not predictive + descriptive scouting profile. Milestone 08 (scouting decision experiments) is NOT started.

## Advanced movement — Milestone 06 checkpoint: Combine → NFL transfer validation (2026-10-13) — EXECUTED by the user; "NOT executed" and "BLOCKER" below are history
- **Status: notebooks 06A, 06B, 06C AUTHORED, NOT executed** (no synthetic check, no smoke run, no full run, no figure). `reports/COMBINE_TO_NFL_TRANSFER_VALIDATION.md` is a template with `‹…›` placeholders. No scoring system, no new architecture.
- **Milestone 05 actual status (read from disk, 2026-10-13):** 05A and 05B were EXECUTED by the user in full mode (`nfl_reliability_manifest.json` written 2026-10-13T09:14Z; 401 athletes, 242,043 eligible windows, 8 / 8 checks PASS). 21 outcomes: 14 eligible, 4 components, 1 reference, 2 exploratory. Rule-based proposal: primary `O2.accel_sust05_mean` (reliability 0.932, n = 336 at ≥ 50 windows), secondary `O5.envelope_level` (0.937, n = 207) and `O3.band_acc_v0_2_mean` (0.922). Milestone 04 was executed too (`transfer_candidates_v1` frozen 2026-10-12T01:36Z: C2 `F2.forty.p95_speed`, C3 `F6.forty.time_v4_to_v8_s`, C5 `FPCA_A_K2`, C6 `TCNB_A`, C7 `TCNB_A|pc4`; C4 empty).
- **BLOCKER — targets NOT frozen.** `nfl_movement_targets_v1.json` does not exist; the recorded decision is `PENDING` (`gate_05: PENDING`). 06A section 2 raises `TargetFreezeError` until the user sets `DECISION` / `PRIMARY` / `SECONDARY` / `RATIONALE` in 05B section 10. Nothing in Milestone 06 defaults to the proposal. Note: `O2.accel_sust05_mean` has context share 0.338 > 0.25, so its frozen `context_adjustment` will be `cell_residual` — the transfer code rebuilds the context means on the training athletes of each fold.
- **Why Milestone 06:** repeatable Combine representations exist (TCN-B AUC 0.873, 3-seed 0.872 ± 0.001, matched-stopwatch 0.796) and reliable NFL outcomes exist; whether one predicts the other beyond official testing is untested.
- **New files in `src/nflbdb/research/advanced_movement/`:**

  | File | Content |
  |---|---|
  | `transfer.py` | NumPy / Polars. `TransferConfig`, freeze audit (`audit_freeze`, `assert_frozen`, `frozen_targets` — status, fields, definition and version hashes recomputed, run mode, no re-freeze after a transfer run), `TargetSet` (athlete labels from `nfl_play_movement`; context means from TRAINING athletes; variants `all` · `first_n:N` · `horizon:H` · `class:c`), `identity_audit`, `frozen_feature_ids`, `cohort_manifest`, funnel / breakdown / reasons / missingness tables, `official_design`, `assign_folds` (registry folds when viable, else new K), `encoder_splits`, `assert_fold_integrity`, `compatibility_table`, `gate_06a`, `require_audit` |
  | `transfer_repr.py` | PyTorch. `fold_representations` / `build_representations` (per fold, fitted on training athletes: ENG, SPEED2, FPCA, TCN, RAND, TCN_PC<k>, seed replicates; a cleared checkpoint is re-asserted disjoint or refused), `checkpoint_records`, `scalar_matrix`, `occlusion_profile` |
  | `transfer_models.py` | NumPy. Ladder MEAN · POS · M0 · M1 · M2 · M3 · M4 · M5 · M6 · M7_PC<k> · M5S, `Design`, `aggregate_attempts` (mean · first · shrunk), `RepBlocks`, `common_support`, `design_matrix` (train-only), `ridge_solve` (position unpenalised), `enet_solve`, `select_alpha` (inner athlete-grouped folds), `run_oof`, `run_analysis`, `evaluate` (paired athlete bootstrap; Δ = reference − model), `fold_performance`, `permutation_null` (outcome · linkage), `gate_06b` |
  | `transfer_robust.py` | subgroup tables, heterogeneity test, `sensitivity_plan`, `loco_design`, matched pairs (rule of 04B, athlete-cluster bootstrap), `mechanism_table`, `ci_stability`, `rank_shift`, `evidence_table`, `readiness_status` / `readiness_registry`, `decision_case` (A–E), `gate_06c` |
  | `transfer_figures.py` | 10 + 12 + 10 figures (`AM06A_*`, `AM06B_*`, `AM06C_*`) |
  | `transfer_validation.py` | 21 synthetic checks (`CHECKS_06A` 7, `CHECKS_06B` 11 of which 3 need torch, `CHECKS_06C` 3) |
  | `artifacts.py` | + `TransferStore`: sub-area `transfer`, own manifest and model directory; `.atlas`, `.nfl`, `.upstream` for reading |

  Also new: `tests/test_advanced_movement_transfer.py` (wrapper), the three notebooks, the report template; README updated.
- **Design fixed before running:**
  - Unit = athlete; one label per athlete (frozen statistic over all eligible REG windows, frozen minimum windows). Primary comparison M5 vs M1 on the frozen primary target; M5 also vs M2, M3, M4, M6; M5S vs M3.
  - Cohorts: primary (every model) → paired (06B: every representation under every fold model; no embedding imputed); wide (official + target, M0 / M1 only, labelled secondary).
  - Official core mandatory (forty, 10-yard, weight); height imputed + indicator; optional tests enter only at ≥ 0.50 coverage among target-exposed athletes.
  - Folds: `FOLD_SOURCE = 'auto'`; size rules min 30 held out per fold, min 10 training athletes per position. Ridge grid 13 points 1e-2…1e4, 5 inner folds; 2000 bootstrap replicates; 200 permutations; practical margin 2 % of the M1 error (heuristic).
  - Readiness: `predictive_validated` only when the gain is established AND ≥ margin AND positive with each cohort left out AND in ≥ 3 positions AND permutation p < 0.05 (+ beats the random encoder and all seeds positive for TCN-B).
- **Inputs (read only, same run mode):** 00 `athlete_registry`, `sequence_manifest`, `split_registry` · 01 `sequence_frames` · 02 `sequence_window_flags`, `fpca_representation_summary`, `fpca/*.npz` · 03 `tcn/*.pt`, `tcn_design_sweep` · 04A `feature_dictionary`, `atlas_attempt_features` · 04B `transfer_candidates_v1`, `atlas/tcn/*.pt` · 05A `nfl_play_movement` · 05B `nfl_movement_targets_v1.json` · `gold.mart_athlete_research_base`.
- **Execution (USER, in Jupyter; no terminal command):** record the freeze in 05B (in the run mode to be used) → restart the kernel → 06A smoke → 06B smoke → 06C smoke → restart → 06A full → 06B full → 06C full → fill the report template. Steps and the share-back list: `notebooks/advanced_movement/README.md`.
- **Outputs:** `{reports/tables, reports/figures, warehouse/gold, artifacts/models}/advanced_movement/transfer/` (smoke: `…/_sample/transfer/`). Names in the README table.
- **Blockers / open points:**
  - Nothing was executed. First failures are most likely in the synthetic checks (tolerances set by reasoning; two assertions rest on a fixed-seed draw and have a few-percent chance of a spurious failure: the no-trait rows of `delta_rmse_sign` and `grouped_permutation`) or in a Polars call; send the table and tracebacks.
  - A smoke run needs a smoke-mode freeze; smoke leave-one-cohort-out may be too small to train (set `RUN_LOCO = False` there).
  - FPCA fold models are refitted (deterministic) and not written to disk; TCN fold encoders are saved.
  - Saved-checkpoint paths are rebuilt as `project_root / relative path`; if `cfg.paths.rel` is not relative to the project root, 06B reuse fails loudly → set `ENCODER_SOURCE = 'retrain'`.
  - The matching rule is `MatchConfig()` defaults (the 04B values), not re-read from the 04B manifest.
  - Gradient-boosted trees and hierarchical models were not authored (no scikit-learn in the project); elastic net is the only alternative learner.
  - Still open from before: `player_get_off` definition, handedness of the Combine frame, units of official measurements.
- **Next milestone: scoring system — conditional.** Designed only after the user's executed Milestone 06, along the decision-tree case of 06C (A validated predictive scorecard candidate · B scorecards on speed-development measures · C prefer SPEED2 · D replicate by position · E descriptive profiles only, no Athletic Transfer Score).

## Advanced movement — Milestone 05 checkpoint: NFL movement outcome lab (2026-10-13) — executed by the user; "NOT executed" below is history
- **Programme (Milestones 05–08):** NFL movement outcomes → Combine-to-NFL transfer → athletic scorecards → scouting decision experiments. Checkpointed: each milestone is authored only after the user's executed results of the previous one. Two products kept apart: pre-draft fingerprint (Combine only, never an NFL input) and post-draft movement transfer.
- **Question (05):** which NFL movement characteristics are measured reliably enough, across independent games, to be targets of the transfer study? No Combine value is read; the target is frozen before any Combine association is looked at.
- **Status: notebooks 05A and 05B AUTHORED, NOT executed** (no synthetic check, no smoke run, no full run, no figure). `reports/NFL_MOVEMENT_OUTCOME_RESULTS.md` is a template with `‹…›` placeholders. `nfl_outcomes.py`, `outcome_reliability.py` and `NflOutcomeStore` were on disk from an earlier, unrecorded session; they were read line by line against their dependencies, not run.
- **Observed on disk, not reviewed:** notebook 04A holds full-run outputs (793 attempts, 418 athletes, synthetic checks PASS). Milestone 04 results and the 04B `DECISION` have not been reported back; Milestone 05 does not depend on them.
- **Files in `src/nflbdb/research/advanced_movement/`:**

  | File | Content |
  |---|---|
  | `nfl_outcomes.py` | `OutcomeConfig` (+ `definition_hash`), `start_table` (fixed 0.5 s / 1.0 s / 5 yd windows inside one tracking segment, filter-edge flags), `play_table` (one row per player-play), `phase_hist`, `assemble_plays` (ordered eligibility rules, chronological window and game order), `outcome_specs` / `outcome_dictionary` (O1–O6, roles candidate · component · reference · exploratory), `OutcomeEngine` (athlete values, context-cell adjustment, game halves), `coverage_table`, `NflOutcomeBuilder` (one scan of `silver.game_tracking`, alternative filters on every 2nd game), `example_plays`, `assert_no_combine_columns` |
  | `outcome_reliability.py` | `reliability_table` (odd / even games, random game halves, Spearman–Brown, ICCs, variance components, seasons), `exposure_table` (first N windows / G games vs later games, common cohort, projection), `context_audit`, `sensitivity_table`, `redundancy_table`, `screen_table`, `propose_targets`, `freeze_targets` (`nfl_movement_targets_v1.json`, written once), `gate_05`, `conclusions_markdown` |
  | `nfl_outcome_figures.py` (new) | 12 figure functions of 05A, 10 of 05B, `measure_summary`, `variant_play_agreement` |
  | `nfl_outcome_validation.py` (new) | 16 synthetic checks (`CHECKS_05A` 10, `CHECKS_05B` 8, `run_all`, `assert_all_passed`); fixtures `kin_play` (analytic frames) and `synthetic_plays` (closed-form reliability) |
  | `artifacts.py` | `NflOutcomeStore`: sub-area `nfl_outcomes`, own manifest |

  Also new: `tests/test_advanced_movement_nfl_outcomes.py` (wrapper), the two notebooks, the report template; README updated.
- **Design fixed before running:**
  - Population REG, positions DB / DL / OL / TE / WR, quality tier A or B, ≥ 15 post-snap frames, not nullified by penalty. Primary filter sg7p2; sensitivity sg5p2 / sg9p2 / sg11p3 and a boundary-inclusive variant.
  - Play measure = maximum over fixed windows (never a single frame; the single-frame peak is a reference row). Athlete statistic = mean or p90 over eligible windows. Speed bands 0–2, 2–4, 4–6, 6–8 yd/s, ≥ 3 start frames.
  - Reliability: halves are whole games; athlete bootstrap (1000); 100 random game half-splits; within position. Primary exposure threshold 50 windows; grids 20 / 50 / 100 / 200 windows and 2 / 4 / 8 / 16 games; later-games reference ≥ 50 windows.
  - Screening reference lines (heuristics): reliability ≥ 0.70 with lower bound ≥ 0.50; ≥ 20 athletes at the threshold in ≥ 3 position groups; filter rank agreement ≥ 0.90; |ρ(value, log windows)| ≤ 0.30; context-cell share ≤ 0.25 or adjusted reliability ≥ 0.70. Eligible = all five; the freeze is recorded by the analyst (`DECISION`, `PRIMARY`, `SECONDARY`, `RATIONALE`), `STOP` allowed.
- **Dependencies (read only):** `silver.game_tracking`, silver `games` / `player_play` / `plays`, `gold.dim_athlete`, `gold/phase_03/{nfl_play_anchors, nfl_player_play_windows, nfl_trajectory_quality}`. Notebooks 00–04 are not prerequisites. No PyTorch.
- **Execution (USER, in Jupyter; no terminal command):** restart the kernel → 05A `RUN_MODE = 'smoke'` → 05B smoke → restart → 05A `'full'` → 05B `'full'` → set `DECISION` in 05B section 10 → fill the report template. Steps and the share-back list: `notebooks/advanced_movement/README.md`.
- **Outputs:** `{reports/tables, reports/figures, warehouse/gold}/advanced_movement/nfl_outcomes/` (smoke: `…/_sample/nfl_outcomes/`). Required tables: `nfl_outcome_dictionary`, `nfl_outcome_coverage`, `nfl_outcome_reliability`, `nfl_reliability_vs_exposure`, `nfl_outcome_context_audit`, `nfl_outcome_quality_sensitivity` (+ `nfl_outcome_redundancy`, `nfl_outcome_screen`); datasets `nfl_play_movement`, `nfl_play_movement_variants`, `nfl_phase_space`, `nfl_athlete_outcomes`, `nfl_athlete_game_outcomes`; `nfl_movement_targets_v1.json`; figures `AM05A_01`–`AM05A_12`, `AM05B_01`–`AM05B_10`.
- **Blockers / open points:**
  - Nothing was executed. First failures are most likely in the synthetic checks (tolerances set by reasoning) or in a Polars call of the frame scan; send the table and tracebacks.
  - Full 05A run time is unknown (one scan of about 22.6 M frames, plus three alternative filters on half of the games).
  - `play_class` is inferred from the vendor `pass_result` code; the meaning of the empty code is unconfirmed.
  - Context labels are alignment, play class, side and down only; assignment, opponent and scheme are unobserved.
  - `coverage_table` computes ρ(value, log windows) on within-position residuals: when every athlete of a run has the same window count it is rounding noise (synthetic fixtures only; not asserted).
  - Still open: `player_get_off` definition (not used), handedness of the Combine frame, PyTorch not a declared dependency.
- **Decision gate 05:** PASS when ≥ 1 outcome is precisely defined, physically interpretable, adequately observed, reproducible across independent games, evaluated for exposure and context sensitivity, and frozen without consulting Combine associations; else STOP and revise the definitions.
- **Next notebook:** `06a_combine_to_nfl_transfer.ipynb` / `06b_transfer_uncertainty_and_subgroups.ipynb` (models M0–M7, inductive folds, paired athlete-level comparison) — NOT started; authored only after the user's executed Milestone 05 results and freeze decision. Then 07 (scorecards, `scoring/` package) and 08 (scouting decision experiments).

## Advanced movement — Milestone 4 checkpoint: feature atlas + robust TCN validation (2026-10-12)
- **Question:** which reproducible movement characteristics exist beyond traditional Combine testing, and which are credible enough to test against NFL performance? No NFL outcome is read; nothing is selected with one.
- **Status: notebooks 04A and 04B authored, NOT executed** (no synthetic check, no smoke run, no full run, no figure). Report `reports/ADVANCED_MOVEMENT_FEATURE_ATLAS_RESULTS.md` is a template with `‹…›` placeholders.
- **Notebook 03 was executed by the user** (full run, manifest `tcn_experiment_manifest.json` 2026-10-11T23:50Z; 793 sequences, 418 athletes, 375 with a repeat, 265 confirm-role athletes in the read-out). Read from the saved tables, input A (speed by distance), confirm athletes, within position:

  | Method | AUC [95 %] | Trace ICC | Resid. AUC | Forty partial R² | Window R² | Cohort gain | Rank share min | Rule label |
  |---|---|---|---|---|---|---|---|---|
  | FPCA_A_K2 | 0.776 [0.745, 0.806] | 0.852 | 0.695 | 0.909 | 0.671 | 0.100 | 0.582 | reference |
  | TCNA_A | 0.832 [0.806, 0.858] | 0.739 | 0.794 | 0.903 | 0.595 | 0.244 | 0.161 | C (collapsed) |
  | TCNB_A (primary) | 0.873 [0.856, 0.890] | 0.925 | 0.917 | 0.890 | 0.102 | 0.077 | 0.255 | A |
  | TCNB_A_E16 | 0.880 [0.864, 0.897] | 0.928 | 0.935 | 0.900 | 0.094 | 0.127 | 0.193 | C (collapsed) |

  **Limitations of that result (why it is not accepted yet):** one training seed; gallery = whole position; the residual AUC EXCEEDS the raw AUC (unexplained — and the FPCA residual goes down); TCNB_A sits 0.005 above the collapse line (0.25) while every other TCN is flagged; every candidate of the design sweep was flagged collapsed, so the consistency term is the pre-declared default (`vicreg`, λ = 0.3), not a ranked choice; reconstruction is worse than FPCA for every encoder; `DECISION` in notebook 03 section 20 is still `None`; NFL value untested.
- **Static inventory (what is reused, what is new):** F1 official ← research mart (12 `cmb_*` columns) · F2 / F4 / F6 forty attempt features and sprint-model parameters ← sprint rescue `attempt_censoring_diagnostics` (no second fitter) · F2 other drills / F3 events ← Phase 04 V registry, evidence only · F5 ← `fpca_component_gate`, `repr_method_summary`, `repr_trace_reliability`, `*_attempt_embeddings`, checkpoints · F7 ← `sequence_window_flags`. **New:** four speed-anchored sustained-acceleration features (anchor 4 yd/s: 0.5 s and 1.0 s mean acceleration, speed gain over 10 yd, time 4 → 8 yd/s; missing with a reason when the interval is not recorded), the characterisation engine, two TCN-B seed replicates, the shortlist.
- **New files in `src/nflbdb/research/advanced_movement/`:**

  | File | Content |
  |---|---|
  | `atlas.py` | NumPy / Polars. `AtlasConfig`, dictionary (`build_dictionary`, 22 columns, key `feature_id`), `sustained_table`, `AttemptData` / `AthleteData`, engine by design (`characterise_all`: static / repeated attempt / model parameter computed; functional score / neural embedding / other drills reused), `screen_feature` (Strong · Provisional · Exploratory · Reject), redundancy matrix + average-linkage clusters, `quality_retention`, `conclusions_markdown` |
  | `atlas_figures.py` | the 10 (+1 by position) figures of notebook 04A |
  | `robust.py` | NumPy only. `MatchConfig`, `matched_retrieval` (k nearest inside a caliper, scales from design athletes, two-way athlete bootstrap), `residual_audit` (raw · crossfit · crossfit_permuted · train_fit · in-sample × 3 distances) + `audit_reading`, `project_embeddings` (training-fitted axes), `cohort_probe` (balanced accuracy), `seed_runs` / `aggregate_seeds`, `cross_geometry`, `cohort_splits` / `loco_table`, `quality_table`, `compact_dimension`, `encoder_readiness`, `build_shortlist`, `freeze_shortlist`, `conclusions_markdown` |
  | `robust_train.py` | PyTorch. `executed_config` (configuration from a notebook-03 checkpoint), `fit_seeded` / `run_seed_stage` (training seed separate from evaluation seed), `untrained_outputs`, `loco_outputs` |
  | `robust_figures.py` | the 10 figures of notebook 04B |
  | `atlas_validation.py` | 23 synthetic checks, no torch (`ATLAS_CHECKS` for 04A, `ROBUST_CHECKS` for 04B, `PROJECTION_CHECKS` = the six parts of `embedding_projection`, `run_all`, `assert_all_passed`) |

- **Patch 2026-10-12 — `embedding_projection` synthetic check failed on the user's first 04B run (AUTHORED, NOT re-run):**
  - Cause (by source reading, not confirmed by execution): the fixture compared `rv.cdist(P, P)` with `rv.cdist(Z, Z)` at `atol = 1e-8` INCLUDING the diagonal. `cdist` expands ‖a‖² + ‖b‖² − 2a·b and takes a root, so a self-distance is √(rounding ≈ 1e-14) ≈ 1e-7 or 0, differently in the two matrices. The projection algorithm is unchanged; no FPCA / TCN output is affected (production never takes a self-distance: attempt 1 is compared with attempt 2), nothing upstream needs a rerun.
  - `robust.py`: `training_axes` (mean, orthonormal axes, variances, rank — the eigen-decomposition `project_embeddings` already did), `projection_status`, `projection_report` (fold × k: training rows, rank, variance share, status), `ProjectionWarning`. `project_embeddings` projects exactly as before but now WARNS when a k is left out (k > dimension, < 3 training rows) or exceeds the training rank; no k is ever replaced by another.
  - `atlas_validation.py`: `check_embedding_projection` now runs `PROJECTION_CHECKS` (dimensionality · orthogonality · determinism · rank-deficient · train/test isolation · scalar embedding) on a 3-fold fixture; separate fits are compared through Gram matrices / squared distances only (axis signs are arbitrary); every assertion reports observed, expected, tolerance, shape, fold and k. `validation.run_all` adds file:line and source text of the failing statement to `detail` (all notebooks).
  - Rerun: restart the kernel, 04B from the top (imports → synthetic validation cell). If a row still fails, send its `detail`.

  Changed: `artifacts.py` (+ `AtlasStore`: sub-area `atlas`, own manifest and model directory, `upstream` for reading), `__init__.py` docstring, README. New `tests/test_advanced_movement_atlas.py` (wrapper). `tcn.py`, `repr_eval.py`, `fpca*.py` are unchanged.
- **Design fixed before running:**
  - Denominator = tracked athletes (≥ 1 eligible forty sequence). Reliability = stratified ICC(1) within position, athlete bootstrap. `beyond_stopwatch` needs reliability of the athlete mean − UPPER bound of the stopwatch R² ≥ 0.10; a low |ρ| alone is labelled `low_correlation_only`. Reference lines ICC 0.75 / 0.50, |ρ| 0.70, window 0.25, cohort 0.25, coverage 0.80, ≥ 50 athletes with a repeat (heuristics).
  - TCN-B seeds `(20261011, 20261012, 20261013)`; the first is the PRIMARY seed = the executed notebook-03 run, reused from disk when `MAX_EPOCHS = None`; replicates `TCNB_A_s1`, `TCNB_A_s2`; configuration read from the notebook-03 checkpoints. No seed is selected.
  - Matching: forty, 10-yard, weight (required), height; k = 5, caliper 0.75 design-athlete SD, same position and fold. Official results cannot be a retrieval representation (one value per athlete → AUC 1 by construction): the per-attempt stand-in is `SPEED2` (p95 + interior peak speed). Controls: untrained encoder `TCNB_A_RAND`, FPCA, TCN-A.
  - Projections on 2 / 4 / 8 training axes per seed (`<method>|pc<k>`); compact candidate = smallest k within `margin_auc` of the 8-d AUC for every seed.
  - Shortlist = 7 pre-declared slots (C1 traditional baseline · C2 speed control · C3 acceleration · C4 sprint parameter · C5 FPCA · C6 TCN-B primary seed · C7 compact projection), filled by rule; `transfer_candidates_v1` is written once per run mode (`REFREEZE_SHORTLIST` to replace).
- **Required artifacts (same RUN_MODE):** 00: `sequence_manifest`, `athlete_registry`, `split_registry`, `research_metadata.json` · 01: `sequence_frames` · 02: `sequence_window_flags`, `fpca_component_gate`, `fpca_athlete_scores`, `fpca_representation_summary`, `fpca/*.npz` · 03: `repr_methods`, `repr_method_summary`, `repr_trace_reliability`, `repr_verdict`, `tcn_design_sweep`, `fpca_ref_*`, `tcn_a_*`, `tcn_b_*` (+ `tcn_e16_*`) embeddings and reconstruction sums, `tcn/TCNB_A__r0_f*.pt` · full-mode only sources: sprint rescue `attempt_censoring_diagnostics`, research mart, Phase 04 V registry and B catalog CSV. 04B also needs 04A's `feature_dictionary`, `atlas_characteristics`, `atlas_feature_status`.
- **Execution (USER, in Jupyter; no terminal command):** restart the kernel → `04a_…ipynb` with `RUN_MODE = 'smoke'` → `04b_…ipynb` smoke → restart → 04A `'full'` → 04B `'full'` → set `DECISION` / `RATIONALE` in 04B section 16 → fill the report template. Steps and the share-back list: `notebooks/advanced_movement/README.md`.
- **Outputs:** everything under `{reports/tables, reports/figures, warehouse/gold, artifacts/models}/advanced_movement/atlas/` (smoke: `…/_sample/atlas/`). Names in the README table. Figures `AM04A_01`–`AM04A_10`, `AM04B_01`–`AM04B_10`.
- **Scientific blockers / open points:**
  - Nothing was executed. First failures are most likely in the synthetic checks (tolerances were set by reasoning, not by running) or in a Polars call; send the table and tracebacks.
  - Units of the official measurements (inches / pounds assumed) are not verified; the `implausible_share` column will show it.
  - Phase 04 registry is read from its CSV path; if absent, F3 and other-drill rows are simply missing.
  - If no notebook-03 checkpoint exists (SAVE_MODELS was off), the encoder configuration is reconstructed and the primary seed is only equal to notebook 03 when BATCH_SIZE and MAX_EPOCHS match.
  - The residual-AUC explanation (cross-fitting artifact vs metric geometry) is a hypothesis until the audit table exists.
  - The status line in the first cell of the executed notebooks 02 and 03 still reads `AUTHORED — AWAITING USER EXECUTION` (files too large for the authoring tools): replace it by hand.
  - Still open from before: handedness of the Combine frame, `player_get_off` definition, PyTorch not a declared dependency.
- **Next milestone: NFL movement outcome reliability** — show that an NFL movement target is repeatable for an athlete across games before any Combine representation is compared with it. GRU / LSTM and the transfer comparison are not started before the user's Milestone 4 results and that reliability study.

## Advanced movement — Milestone 3 checkpoint: self-supervised encoder (2026-10-11) — executed by the user; "NOT executed" below is history
- **Question:** can a neural network discover a stable, athlete-specific movement signature beyond what official Combine results and recording artifacts already explain? Not: does it reconstruct curves better than PCA.
- **Status: notebook 03 authored, NOT executed** (no synthetic check, no smoke run, no full run, no figure).
- **Notebook 02 full execution reviewed** (manifest `fpca_experiment_manifest.json`, 2026-10-11T15:20Z; 793 sequences, 418 athletes, 375 with a repeat; 9 / 9 checks; no blocking failure). Within position, out of fold:

  | Representation | K | Held-out FVE | RMSE | Finding |
  |---|---|---|---|---|
  | A speed by distance | 2 | 0.975 | 0.101 yd/s | PC1 ICC 0.95 [0.94, 0.96] but \|ρ\| 0.93 with the official times (redundant); PC2 ICC 0.36, window share 0.59 |
  | B a_tan by speed | 5 | 0.964 | 0.162 yd/s² | PC1 ICC 0.53 [0.35, 0.70], \|ρ\| 0.61, window 0.15 (inconclusive); PC2–5 not repeatable |
  | C speed by time | 2 | 0.962 | 0.152 yd/s | PC1 ICC 0.42, window share 0.95 (recording artifact) |

  No component combines repeatability, low stopwatch redundancy and low window dependence. Rule-based benchmark: A, K = 2.
- **Design of notebook 03 (fixed before running):**
  - Inputs = the notebook-02 curves (`fpca.build_curves`), one single-input encoder each (A primary, B secondary, C control); support, mean function and eligibility (≥ 50 % of the support) are those of the FPCA fit of the same fold.
  - Model `tcn.CurveTCN`: 3 channels (value, visible flag, grid position), 3 dilated residual blocks (32 channels, kernel 5), pooling over visible points, 8-d embedding, decoder from the embedding only (about 32k weights on input A — to be confirmed by the size table).
  - TCN-A = masked reconstruction (loss on recorded points hidden on purpose). TCN-B = TCN-A + λ · consistency between two attempts of a training athlete on their shared recorded range (`vicreg` or `mse`).
  - Outer folds = inductive evaluation (split repeat 0); inner fold 0 = early stopping; **consistency kind and λ chosen on design athletes only** (cross-fitted inside them, input A, applied unchanged to B and C); **confirm-role held-out athletes = read-out**. 16-d embedding = declared sensitivity, never selected.
  - Evaluation identical for FPCA and encoders (`repr_eval`): within-position repeat-attempt AUC / retrieval, trace reliability, collapse, probes (official results, body, position, cohort, window), residual repeatability after removing official results, window regression, truncation stability, reconstruction on the common support with one fixed mask. Rule-based A / B / C / D verdict.
- **New files in `src/nflbdb/research/advanced_movement/`:**

  | File | Content |
  |---|---|
  | `repr_eval.py` | NumPy only. `EvalConfig`, `MethodSpec`, masks (`span_mask`, `evaluation_mask`, `truncation_mask`), embedding / reconstruction frames, `pair_stats`, repeatability and paired tables, `trace_icc`, collapse metrics, ridge probes, `residualise_embeddings`, window and truncation tables, `fpca_fold_outputs`, `evaluate`, `gate`, `conclusions_markdown` |
  | `tcn.py` | PyTorch. `TcnConfig`, `CurveNormalizer`, `CurveTCN`, `masked_mse`, `consistency_loss`, `AttemptPairSampler`, `train_encoder`, `embed`, checkpoints, `train_fold`, `fit_methods`, `fpca_reference`, `run_stage` (cached), `design_sweep`, `select_consistency`, `masked_examples`, `AUGMENTATIONS` |
  | `tcn_figures.py` | the 12 figure functions of notebook 03 |
  | `tcn_validation.py` | 15 synthetic checks (`run_all`, `assert_all_passed`) |

  Also `tests/test_advanced_movement_tcn.py` (wraps the checks, skipped without torch), `notebooks/advanced_movement/03_self_supervised_tcn.ipynb`, README updated. No existing module was changed.
- **Required artifacts (same RUN_MODE):** `sequence_manifest`, `athlete_registry`, `split_registry`, `research_metadata.json` (00); `sequence_frames`, `sequence_summary` (01); `sequence_window_flags`, `preprocessing_audit`, `fpca_representation_summary`, `fpca_component_gate`, `artifacts/models/advanced_movement/fpca/*.npz` (02; refitted in memory if the key differs).
- **Execution (USER, in Jupyter):** install PyTorch in env `nfl` if absent; restart the kernel; run `03_self_supervised_tcn.ipynb` top to bottom with `RUN_MODE = 'smoke'`, then `'full'` after a kernel restart; after the full run set `DECISION` / `RATIONALE` in section 20. Cell-by-cell steps and the share-back list: `notebooks/advanced_movement/README.md`.
- **Outputs:** `reports/tables/advanced_movement/{tcn_*,repr_*}.csv`, `warehouse/gold/advanced_movement/{fpca_ref_*,tcn_*,repr_*}.parquet`, `tcn_experiment_manifest.json`, `artifacts/models/advanced_movement/tcn/*.pt`, figures `AM03_01`–`AM03_12`.
- **Scientific blockers / open points:**
  - PyTorch is not a declared project dependency; version compatibility is unverified until the checks run (`torch.load(weights_only=True)` needs ≥ 1.13).
  - Nothing was executed: first failures are most likely in the synthetic checks; send the table and tracebacks.
  - The design subset (about 125 athletes) ranks the consistency term noisily; it is used once and not revisited.
  - Intervals cover athlete sampling only, not training seeds or split repeats (repeat 0).
  - Residualisation is linear; a nonlinear function of the forty can survive it.
  - The stale status line in the first cell of notebook 02 could NOT be corrected by Claude (the executed file is too large for the non-executing edit tool). Replace `AUTHORED — AWAITING USER EXECUTION` there by `EXECUTED BY THE USER (full run 2026-10-11)` by hand; no number is affected.
  - Still open from before: handedness of the Combine frame, `player_get_off` definition.
- **Next research milestone:** read the executed notebook 03 and classify A / B / C / D on input A. A → GRU / LSTM comparison and NFL transfer; B → keep as experimental, no claim; C → stop adding complexity, choose an NFL target aligned with validated signals; D → state what a further experiment can resolve. Notebook 04 is not started before that review.

## Advanced movement — Milestone 2 checkpoint: functional PCA (2026-10-11) — executed by the user; "NOT executed" below is history
- **Question:** can functional representations of Combine sprint trajectories capture repeatable athlete characteristics beyond official Combine measurements? FPCA is the low-complexity benchmark for the TCN and GRU/LSTM; it is not assumed to beat the official forty.
- **Status: authored, NOT executed.** Neither the synthetic checks, nor the smoke run, nor the full run of notebook 02 has happened.
- **Milestone 1 is executed** (the "NOT executed" lines of the Milestone 1 section below are history). Full run: 793 eligible forty sequences, 418 athletes, 375 with a repeat, 41,500 frames; leakage audit 7 / 7; channel missingness 0 for `t_rel` / `dist` / `speed`, 0.03 % for `a_tan`, 3.5 % for heading channels. Integrated ICC, position-adjusted, design / confirm: speed by distance 0.754 / 0.837; `a_tan` by speed phase 0.364 / 0.416; speed by elapsed time 0.230 / 0.575 (within-athlete variance about 4× larger on the design subset for the clock-based alignments). Notebook 01's `CHOICES` were left `PENDING REVIEW`; the Milestone 2 brief fixes the representations and supersedes them.
- **Sprint-rescue fallback warning (investigated):** synthetic fixtures only. All 11 occurrences in the executed notebook 00 are in the output of its synthetic-validation cell; the production census cell printed none, and every eligible sequence carries the flags. Corrections: `data.sequence_manifest(..., rescue_expected=False)` in fixtures (no warning), and a blocking check `sprint_rescue_flags_present` in notebook 02.
- **Representations:** A speed against recorded distance (primary) · B `a_tan` against speed, rising phase, boundary frames masked (secondary) · C speed against elapsed time (control). Alternates for sensitivity: A′ and C′ with the origin at the moment speed reaches 2 yd/s, B′ with boundary frames kept.
- **New files in `src/nflbdb/research/advanced_movement/`:**

  | File | Content |
  |---|---|
  | `fpca.py` | `FpcaConfig`, `RepresentationSpec` / `REPRESENTATIONS`, `CurveSet`, `build_curves`, B-spline basis, `fit_fpca` (athlete-weighted, pairwise-complete covariance, support = grid observed for ≥ 80 % of training weight), `FpcaModel` (BLUP scores from observed points, reconstruction, residuals on observed points), `align_to_reference`, `save_models` / `load_models` |
  | `checkpoint.py` | `sequence_windows` (original clock, first / last-frame censoring indicators, frame counts), `audit_preprocessing`, `assert_blocking_passed`, `final_decisions` |
  | `fpca_eval.py` | fold loop (`fit_representation`, `fit_all`), reliability (fold inside every stratum), stopwatch and body association, window dependence (between and within athlete), alignment sensitivity, score stability, `select_k`, `gate_tables`, `conclusions_markdown` |
  | `fpca_figures.py` | the 12 figure functions of notebook 02 |
  | `fpca_validation.py` | 9 synthetic checks (`run_all`, `assert_all_passed`) |

- **Changed files:** `data.py` (`rescue_expected` flag), `validation.py` (fixture passes it), `alignment.py` (`icc_pointwise_median` ignores undefined points; the cached `alignment_comparison` keeps its NaN until notebook 01 is rebuilt, the integrated ICC is unaffected). New `tests/test_advanced_movement_fpca.py`. `notebooks/advanced_movement/README.md` updated.
- **Design fixed before running:** only K ∈ {2, 3, 5, 8} is compared; basis size (16 / 10 / 16), 80 % support rule, 50 % scoring rule and BLUP scoring are fixed. Evaluation is inductive on the split registry of notebook 00 (outer folds; inner folds unused). The reference fit on all athletes is transductive and fixes sign and order only. Intervals: athlete bootstrap on split repeat 0. Benchmark K = smallest K with held-out FVE ≥ 0.95. Gate reference lines: ICC 0.50, |ρ| 0.70, window share 0.25 (heuristic). No NFL value is read.
- **Execution (USER, in Jupyter):** restart the kernel; run `02_functional_pca_baseline.ipynb` top to bottom with `RUN_MODE = 'smoke'`, then `'full'` after a kernel restart. Steps and outputs: `notebooks/advanced_movement/README.md`.
- **Share back:** listed in the README ("What to send back after Milestone 2").
- **Open points:** the proposed choices in `preprocessing_decisions_final` (channels, boundary masking, coverage rules, quality scenario, normalisation) need review; missing high speeds in B are informative; the design / confirm gap of notebook 01 on clock-based alignments is unexplained (see the extreme-recordings table of notebook 02 section 1); PyTorch is still not a project dependency (needed from notebook 03).
- **Next:** interpret the executed notebook 02, settle the benchmark representation and K, then Milestone 3 (self-supervised TCN, notebook 03). No NFL outcome modelling before that review.

## Advanced movement representation learning — Milestone 1 checkpoint (2026-10-11) — executed by the user; "NOT executed" below is history
- **Programme:** notebook-first study of whether sequence representations of Combine tracking (FPCA, self-supervised TCN, GRU/LSTM) hold repeatable information beyond official tests and handcrafted features, and whether it transfers to NFL movement. Notebooks are the only execution entry points; no terminal command.
- **Status: Milestone 1 authored, NOT executed.** Neither smoke nor full run has happened.
- **Notebooks (`notebooks/advanced_movement/`, guide in its `README.md`):** `00_research_setup_and_data_audit.ipynb`, `01_sequence_eda_and_alignment.ipynb`. Notebooks 02–08 are not authored.
- **Package `src/nflbdb/research/advanced_movement/`:**

  | File | Content |
  |---|---|
  | `config.py` | `NotebookControls` (RUN_MODE, FORCE_REBUILD, SAVE_*, DEVICE, SEED, …), `AdvancedConfig` (values with provenance), channel registry, dependency / environment / device / seed helpers |
  | `artifacts.py` | `AdvStore` (on `RescueStore`): smoke outputs under `_sample`, cached builds, `require` with an actionable error, inline + saved figures with metadata |
  | `data.py` | read-only inputs, schema inventory, sequence manifest, athlete registry, census, split registry (outer / inner folds, design subset), leakage audit |
  | `sequences.py` | invariant per-frame channels of the longest valid segment, `SequenceSet` (padded arrays + masks, validated) |
  | `alignment.py` | elapsed time, normalised time, distance, speed phase; integrated curve ICC; comparison table |
  | `preprocessing.py` | `ChannelScaler` fitted on training athletes only |
  | `eda.py`, `visualizations.py` | per-sequence summary, rule-based examples, decisions table; figures of notebooks 00–01 |
  | `validation.py` | 13 synthetic checks runnable from a notebook cell (`run_all`, `assert_all_passed`) |

- **Other files:** `tests/test_advanced_movement.py` (wraps the same checks), `tests/test_polars2_compat.py` (guard extended). No existing pipeline was changed.
- **Rules fixed here:** athlete is the unit; splits use position × cohort only; primary evaluation is INDUCTIVE (encoder refitted inside each outer training fold, held-out athletes' Combine sequences excluded); anything else is labelled TRANSDUCTIVE; preprocessing choices are quantified on a fixed ~30 % design subset; absolute x / y / heading are not channels; padding is masked everywhere.
- **Execution (USER, in Jupyter):** run 00 then 01, top to bottom, first with `RUN_MODE = 'smoke'`, then `'full'` after a kernel restart; same RUN_MODE and SEED in both. Then record `CHOICES` / `RATIONALE` in notebook 01 section 9 and rerun that cell.
- **Share back:** see `notebooks/advanced_movement/README.md` (validation table, flow and reconciliation tables, leakage audit, alignment tables, figures AM00_01 / 05 / 07 and AM01_03 / 04 / 12, the recorded choices, any traceback).
- **Blockers / open points:** PyTorch is not a project dependency (needed from notebook 03; notebook 00 reports it); `player_get_off` definition unresolved, so get-off stays an opaque endpoint and the NFL movement envelope (notebook 07) is the alternative target; handedness of the Combine frame unverified, so signs of lateral channels are not comparable across cohorts; `reports/ADVANCED_MOVEMENT_RESULTS.md` template not written yet (Milestone 8).
- **Next milestone (2):** FPCA utilities and notebook 02, after the user's Milestone 1 outputs and alignment choice. A fresh session or `/compact` is sensible before starting it.

## Combine → NFL DL get-off checkpoint (2026-10-11) — executed by the user; "NOT executed" below is history
- **Question:** among defensive linemen, do tracked forty features (`p95_speed`, `accel_v4_7`, `model_tau_s`) improve out-of-fold prediction of NFL `player_get_off` beyond the official forty and 10-yard split? One experiment, athlete-level.
- **Status: authored, NOT executed.** No test, stage or figure has been run.
- **BLOCKER (open): the vendor definition of `player_get_off` is unresolved.** The repository holds no data dictionary (`schema_catalog.csv` evidence = "inferred"). The definition is read from `configs/player_get_off_definition.json` (shipped with status `unresolved`). Until the user fills it from the competition data dictionary: stages DEF / TGT / COH run (they use no Combine–NFL relation); MOD / SENS run only with `--allow-unresolved-definition`, are stamped `definition_status = unresolved`, and the report prints the blocker instead of a football interpretation.
- **New package `src/nflbdb/research/combine_nfl_dl/`:**

  | File | Content |
  |---|---|
  | `config.py` | `GetOffConfig` (values with provenance), model ladder MEAN / M0 / M1 / M2a / M2b / M2c (+ exploratory M2bc, M1j), `AnalysisSpec`, the sensitivity list |
  | `endpoint.py` | stage DEF: definition gate, endpoint audit, valid observations |
  | `targets.py` | stage TGT: athlete targets (adj_mean, mean, median, p25, p75; windows all / first_n / no_boundary), context cells, reliability across independent games |
  | `cohort.py` | stage COH: Combine feature variants, research base, funnel, missingness, predictor redundancy |
  | `models.py` | NumPy ridge with in-fold leave-one-out penalty, Huber, athlete K-fold, fold-aware target, paired athlete bootstrap |
  | `evaluate.py` | stages MOD and SENS: primary comparison, coefficients, matched pairs, sensitivities |
  | `pipeline.py`, `figures.py`, `report.py`, `__main__.py` | stages, 8 figures, report filler, CLI |

- **Other files:** `configs/player_get_off_definition.json` (new), `reports/COMBINE_TO_NFL_DL_GETOFF_TEMPLATE.md` (new), `tests/test_combine_nfl_dl.py` (new, 28 tests, synthetic), `tests/test_polars2_compat.py` (guard extended), `sprint_rescue/store.py` (`RescueStore` takes an optional `namespace`; default unchanged).
- **Design fixed before running:** primary target = mean get-off after removing alignment × season means (context means from training athletes inside each fold), athletes with ≥ 50 valid observations; identical athletes and folds for every model; repeated 5-fold × 20 stratified by cohort; reference model M1; alignment is an NFL observation and never a predictor. No scikit-learn dependency.
- **Inputs (read only):** Phase 05 P0 player-play base (`fact_nfl_play_participation`, silver `player_play` / `plays` / `games`, `dim_athlete`), `gold.mart_athlete_research_base`, sprint rescue `attempt_censoring_diagnostics`, `within_position_reliability`, `combine_timing_redundancy`. No frame scan.

### Manual execution (USER) — Combine → NFL DL get-off
```
python -m pytest tests/test_combine_nfl_dl.py tests/test_polars2_compat.py -q
python -m nflbdb.research.combine_nfl_dl --only DEF,TGT,COH                        # audit, targets, cohort (not gated)
python -m nflbdb.research.combine_nfl_dl --smoke --allow-unresolved-definition     # wiring check → */combine_nfl_dl/_sample
python -m nflbdb.research.combine_nfl_dl --allow-unresolved-definition             # MOD + SENS, labelled unresolved
python -m nflbdb.research.combine_nfl_dl --check
```
After the definition is filled and set to `confirmed`, rerun without the flag: MOD and SENS recompute (the definition is in their fingerprint). `--figures-only` and `--report` redraw from saved tables.

- **Outputs:** `reports/tables/combine_nfl_dl/*.csv` (11 tables), `warehouse/gold/combine_nfl_dl/*.parquet` + `.schema.json` (18 outputs incl. `athlete_research_base`, `dl_getoff_plays`, `athlete_targets`, `athlete_context_cells`, `combine_feature_variants`, `fold_assignments`, `oof_predictions`), `reports/figures/combine_nfl_dl/CN_01 … CN_08` (PNG + SVG; CN_05–07 also PDF), `reports/COMBINE_TO_NFL_DL_GETOFF_RESULTS.md`.
- **Share back:** test output and any traceback; the generated report; `endpoint_audit.csv` (sections boundary, alignment, direction_evidence); `cohort_funnel.csv`; `outcome_reliability.csv`; `model_comparison.csv`; `bootstrap_comparisons.csv`; `sensitivity_analyses.csv`; figures CN_03, CN_06, CN_07; and the vendor text for `player_get_off` if available.
- **Known before running:** about 54–65 athletes are expected in the primary population (Study F overlap counts), so only large gains are detectable; `cmb_combine_height` is cast to a number and imputed in-fold if it does not parse; single cohorts are scored from the primary fit (`scored_on:*`), not refitted.

## Sprint reliability rescue checkpoint (2026-10-11) — executed by the user; "NOT executed" below is history
- **Question:** is there a forty-yard-dash measurement that is repeatable within position and not explained by the official forty / 10-yard split? One experiment, forty only. Main NFL models stay unstarted until its report is reviewed.
- **Status: authored, NOT executed.** No test, stage or figure has been run.
- **Correction of older notes:** Phase 04 is no longer "not run". `warehouse/gold/phase_04/research_base/_studies_manifest.json` records completed Q, B, B.frames and C1 runs (2026-10-11 02:07–02:15) and the V registry is on disk. The Phase 04 and Phase 05 "NOT run" lines further down are history. Report headers 14 and 15 were corrected from the dataset sidecars.
- **New package `src/nflbdb/research/sprint_rescue/`:**

  | File | Content |
  |---|---|
  | `config.py` | `RescueConfig` (every value with provenance), candidate feature list |
  | `store.py` | flat namespace store: CSV + Parquet + schema sidecar, stage fingerprints |
  | `stats.py` | stratified ICC(1) with paired athlete bootstrap, partial correlation, partial R² |
  | `attempts.py` | stage ATT: attempt table, window features, speed-band accelerations, within-athlete differences |
  | `sprint_model.py` | Study C: refits of the existing model on altered windows, identifiability index |
  | `censoring.py` | Study A: paired association, window-adjusted ICC, ICC by censoring pattern |
  | `reliability.py` | Studies B and C: pooled / within-position / adjusted ICC; model vs direct on identical attempts |
  | `redundancy.py` | Study D: association with official times, feature screening |
  | `cohorts.py` | Studies E and F: cohort reconciliation, DL get-off and WR separation feasibility |
  | `pipeline.py`, `figures.py`, `report.py`, `__main__.py` | stages, 10 figures, report filler, CLI |

- **Other files:** `reports/SPRINT_RELIABILITY_RESCUE_TEMPLATE.md` (new), `tests/test_sprint_rescue.py` (new, 20 tests, synthetic), `tests/test_polars2_compat.py` (guard extended to the new package), `reports/14_feature_engineering.md` and `reports/15_movement_reliability.md` (stale headers corrected). No existing source module was changed.
- **Inputs (read only):** Phase 03 `combine_sprint_profiles`, `combine_attempt_features`, `combine_frame_kinematics`; Phase 04 `trajectory_features_by_smoothing`, `trajectory_quality_flags`, V registry; `gold.mart_athlete_research_base`; the Phase 05 P0 population.

### Manual execution (USER) — sprint rescue
```
python -m pytest tests/test_sprint_rescue.py tests/test_polars2_compat.py -q
python -m nflbdb.research.sprint_rescue --sample-athletes 80      # smoke → */sprint_rescue/_sample
python -m nflbdb.research.sprint_rescue                           # stages ATT, A, B, C, D, E, F + figures + report
python -m nflbdb.research.sprint_rescue --check
```
Single stages: `--only ATT` (then any of `A,B,C,D,E,F`; D needs B). `--figures-only` and `--report` redraw from saved tables.

- **Outputs:** `reports/tables/sprint_rescue/*.csv` (11 tables), `warehouse/gold/sprint_rescue/*.parquet` + `.schema.json` (13 outputs incl. `athlete_sprint_features`, `example_signals`), `reports/figures/sprint_rescue/SR_01 … SR_10` (PNG + SVG; SR_10 also PDF), `reports/SPRINT_RELIABILITY_RESCUE_RESULTS.md`.
- **Share back:** test output and any traceback; the generated report; `feature_screening_summary.csv`; `sprint_model_repeatability.csv` (cohort = common); `censoring_association.csv` (analysis = A3_window_adjusted_icc); `athlete_cohort_reconciliation.csv`; `dl_getoff_feasibility.csv` (sections definition, reliability); figures SR_04, SR_06, SR_09, SR_10.
- **Scientific blockers / open points:**
  - The vendor definition of `player_get_off` is not in the repository (`schema_catalog.csv` evidence = "inferred"). Study F flags it; it must be settled before the endpoint is modelled.
  - `end_censored` is true for every forty attempt, so it carries no information; continuous window variables are used instead.
  - Two attempts per athlete: within-position intervals will be wide in small groups (TE).
  - The window-adjusted ICC is an in-sample upper bound; the identifiability index is not a confidence interval.
- **Decisions waiting for the results:** primary and backup feature family (rule-based proposal in report §10, needs review); whether any acceleration representation is kept; whether DL get-off has enough repeatability and overlap to be the first endpoint.

## ▶ Phase 04 repair checkpoint (2026-10-11) — read this first
- **Failures:** `Q_athlete_eligibility` 80 duplicate keys; `B_smoothing_agreement` 864 duplicate keys; C1 / V / report (83 placeholders) failed downstream.
- **Root cause (both):** five canonical skill drills exist under several drill_types (`stratum`): GAUNTLET_DRILL (DB / WR / TE), OVER_SHOULDER_ADJUST (WR / TE), PASS_RUSH_DRILL, BACK_PEDAL_AND_REACT, SHORT_ZONE_BREAKS (DL / LB). Producers group by domain × stratum × drill; the keys omitted `stratum`. Counts match `reports/tables/combine_drill_counts.csv`: B 6 extra cells × 24 features × 3 settings × 2 levels = 864; Q also joined its S0 reference on (domain, drill), a real fan-out: 4 × Σ(m² − 1) = 80.
- **Fix = grain, no deduplication.** Keys now include `stratum`:

  | Table | Key |
  |---|---|
  | `Q_athlete_eligibility` | domain, stratum, drill, scenario (S0 join fixed to the same grain) |
  | `B_smoothing_agreement` | domain, stratum, drill, feature, smoothing, level |
  | `B_variance_share` | domain, stratum, drill, feature, level |
  | `B_repeatability_by_setting` | domain, stratum, drill, feature, smoothing (attempt count now within stratum) |
  | `V_feature_validation_registry` | domain, stratum, drill, feature |
  | `B_results`, `V_results` | `stratum` field = drill, or `drill_type:drill` for shared drills |

  No column added or removed. `B_robustness_facets` and all C1 tables stay pooled per canonical drill (documented in their contracts).
- **Other changes:** Study B has two run records, `B.frames` (written first) and `B`; `StudyStore.require_run`; C1 and V refuse to run without a completed B; `--check` reports missing / invalid / incomplete / blocked / stale; the report cites a Phase 04 study only after its run completed and lists a source-dependency table; `configure_runtime` runs before `import polars` in `phase04/__main__.py`; template tokens select by full key.
- **Files:** `phase04/{quality_eligibility,smoothing,events,validation_registry,report,__main__}.py`, `studies/store.py`, `reports/PHASE_04_SCIENTIFIC_RESEARCH_REPORT_TEMPLATE.md`, new `tests/test_phase04_grain_regression.py` (17 tests).
- **Intermediates on disk** (`trajectory_features_by_smoothing` 1,322,496 rows, `events_by_smoothing` 427,901 rows): schema unchanged, but the failed run recorded no fingerprint (`runs` is empty) and never wrote `B_vendor_residual_by_setting` / `B_example_signals`, which need frames. One frame pass is therefore required (previous full B run ≈ 18 min by manifest timestamps); after it, `B.frames` is reused.

### Manual execution (USER) — Phase 04 repair
```
python -m pytest tests/test_phase04_grain_regression.py tests/test_phase04_m1.py tests/test_polars2_compat.py -q
python -m nflbdb.research.phase04 --only Q
python -m nflbdb.research.phase04 --only B        # one frame pass, then B.frames is cached
python -m nflbdb.research.phase04 --only C1
python -m nflbdb.research.phase04 --only V
python -m nflbdb.research.phase04 --check
python -m nflbdb.research.phase04 --report
```
**Share back:** test output; any traceback; `--check` output; the "Source dependencies" table at the end of the report.

### Open gates (unchanged, not satisfied)
- Review Q / B / C1 / V outputs before any interpretation; C2 human labels (15 report tokens stay pending until then); Milestone 2 NFL repeatability.
- Decision pending: should C1 and `B_robustness_facets` also be stratified by drill_type for the five shared drills (today pooled; V uses the pooled C1 value for those rows)?
- The stale `_sample` namespace was written by the failed code; rerun the smoke command before reading it.

## Phase 05 checkpoint after the P0 review (2026-10-11)
- **Read:** `reports/PHASE_05_RESEARCH_DECISION.md` (decision, gates, commands), then `PHASE_05_P0_RESULTS_REVIEW.md` §10 (evidence table) and `PHASE_05_UPDATED_FEASIBILITY.md`.
- **P0 executed by the user (REG):** 0 football rows in 22,654,290; median 2 tracked athletes per play (mean 2.72; 1.55 → 3.63 by season); both sides tracked on 58.9 % of plays (29.5 % → 83.0 %); 401 athletes, 303 with ≥ 100 plays (DB 73, DL 77, OL 69, WR 63, TE 21); OL × DL 28,760 and WR × DB 15,710 co-tracked plays; dense outcomes = DL get-off, WR separation, OL pressure allowed.
- **Reviewed verdicts:** A supported on the NFL side (Combine side unmeasured) · B single-athlete reachability only · C not pursued as RL (E05-07 closed) · D pairwise only, unverified matchups · E partly feasible.
- **Decision:** primary = capability transfer (E05-01 then E05-02), positions DB / DL / OL / WR, role from `lined_up_position`. Backup = DL get-off transfer (E05-B1). Stretch = pairwise S7.
- **Phase 04 gates: NOT satisfied.** On disk: Q without `Q_athlete_eligibility` / `Q_results`, B without its agreement tables, no C1 / C2 / V. Both runs stopped part-way on contract key errors; fix authored, see the repair checkpoint above.

| Status | Item |
|---|---|
| Executed + reviewed | Stage P0 (`reports/tables/phase_05/descriptive/P0_*`, figures P0_01–05) |
| Authored, NOT run | Stage P0b: `phase05/sample_frame.py`, `SampleFrameConfig`, CLI key `P0B`, `tests/test_phase05_p0b_sample_frame.py` (7 tests); `STUDY_DIRS["P0B"]`; one-line change in `tests/test_phase05_p0_audit.py` |
| Next to author | Stage S2 `phase05/envelope.py` (NFL only, not gated); config fixed after the P0b tables |
| Blocked | S3, S4, backup E05-B1: Phase 04 gates |

### Manual execution (USER)
```
python -m pytest tests/test_phase05_p0_audit.py tests/test_phase05_p0b_sample_frame.py tests/test_polars2_compat.py -q
python -m nflbdb.research.phase05 --only P0B
python -m nflbdb.research.phase05 --only P0B --check
python -m nflbdb.research.phase04 --only Q        # send the traceback if it fails
python -m nflbdb.research.phase04 --only B
```
**Share back:** test output; `P0b_combine_nfl_sample`, `P0b_exposure_by_cohort`, `P0b_outcome_exposure`, `P0b_pairs_by_season`, `P0b_play_type`; figures P0b_01–02; Phase 04 console output.

### Exact next step
1. Read the P0b tables: apply gate G-0b, settle the play-minimum grid, the role taxonomy and whether WR / TE plays are route plays only.
2. Author S2 (envelope and outcome reliability vs plays; robust quantiles on interior frames; cohort and season strata).
3. Fix Phase 04 Q / B from the traceback so the V registry can be produced.

## Phase 05 checkpoint before P0 ran (2026-10-11) — history; "NOT run" below is superseded for P0
- **Documents (complete, no results):** `reports/PHASE_05_{FRONTIER_RESEARCH_PLAN,DECISION_PROBLEM_AUDIT,NOVELTY_MATRIX,EXPERIMENT_REGISTRY,METRIC_REGISTRY,RESEARCH_RESULTS}.md`. The plan holds the stages S2–S8 and their gates.
- **Primary question (RQ-P):** does a tracking-derived Combine capacity profile predict the athlete's realized, context-matched NFL movement envelope better than official Combine results, under athlete-grouped evaluation?
- **Pre-P0 stream verdicts (hypotheses):** A capability twin = primary · B counterfactuals = single-agent reachability only · C offline RL = not identifiable as policy optimisation · D multi-agent = co-tracked rookie pairs at most · E metrics = feasible.
- **Why:** NFL tracking covers only the Combine cohort (≈ 3.6 tracked athletes per play, executed report 04) and a 120k-row sample showed no football rows. P0 measures both exactly.

| Status | Item |
|---|---|
| Authored, NOT run | `src/nflbdb/research/phase05/` — `config.py`, `decision_audit.py` (Stage P0), `audit_figures.py` (P0_01–05), `__main__.py`; `tests/test_phase05_p0_audit.py` (16 tests, synthetic) |
| Validated | nothing in Phase 05 |
| Blocked | S2–S8: every Combine→NFL statistic waits for the Phase 04 gates (V registry + M2 reliability) |
| Hypothetical | every Phase 05 innovation |

- **P0 design:** a census, no inference and no Combine feature, so it does not touch the Phase 04 gate. Base = one row per tracked player-play (`gold.fact_nfl_play_participation` + silver `player_play` / `plays` / `games` + `dim_athlete`), in Polars. Only the entity census scans frames (DuckDB over `silver.game_tracking`). Population = REG; the census reports every season type.
- **Outputs:** `reports/tables/phase_05/descriptive/P0_*.csv` (9 tables incl. `P05_thresholds`), `warehouse/gold/phase_05/decision_audit/athlete_exposure.parquet`, `reports/figures/phase_05/P0_feasibility/`.
- **Known before running:** position groups in the warehouse are DB, DL, OL, TE, WR only, so the `LB` / `RB` entries of `StreamFeasibilityConfig` match nothing (harmless). Side = `team_abbr` vs `possession_team`. Co-tracking is availability, not a matchup.
- **Not done:** no Phase 05 viewer notebook; no literature search (plan in the novelty matrix §2, no citation invented); `CLAUDE.md` and `NFL_BDB_2027_IMPLEMENTATION_AND_RESULTS_REPORT.md` do not exist in the repo.

### Manual execution (USER) — Phase 05 P0
```
python -m pytest tests/test_phase05_p0_audit.py tests/test_polars2_compat.py -q
python -m nflbdb.research.phase05 --only P0 --seasons 2023      # smoke → */phase_05/_sample
python -m nflbdb.research.phase05 --only P0
python -m nflbdb.research.phase05 --check
```
**Share back:** test output; `P0_entity_census`, `P0_side_coverage`, `P0_opponent_pairs`, `P0_stream_feasibility`, `P0_action_overlap`, `P0_exposure_summary`; figures P0_01, P0_02, P0_05.

### Exact next step (Phase 05)
1. Review the P0 outputs, re-rate the stream verdicts, fill `PHASE_05_RESEARCH_RESULTS.md` §P0 and log any threshold change in the experiment registry.
2. Phase 04 M1 then M2 must pass before Stage S2 (realized NFL envelope, `phase05/envelope.py`) is run. S2 code may be authored earlier.

## ▶ Phase 04 checkpoint (2026-10-11) — read this first
- **Plan / reuse map / gates:** `docs/PHASE_04_PLAN.md`. Annotation rules: `docs/PHASE_04_ANNOTATION_PROTOCOL.md`.
- **Executed Phase 03 evidence, first-read (not yet user-reviewed):**
  - **A0.** The CRITICAL Combine key failure is a rule error (BALL rows). The s/a WARNs are BALL-only. Orphans → excluded from context features. Nothing is blocking.
  - **A3, tiers.** AUC ≈ 0.50 everywhere for `tier_not_A`, so tiers are non-informative. `resid_rms_yds` / `speed_vendor_mad` do discriminate (AUC 0.65–0.92).
  - **A3, boundary peaks.** Combine raw `peak_speed` sits on SG-boundary frames at 2.7× (forty) to 11.4× (three-cone) the expected share. NFL ≤ 0.7×.
  - **NFL windows.** Snap tag present in ≥ 99.99 % of REG plays.
- **New package `src/nflbdb/research/phase04/`** (authored, not run):

  | Module | Content |
  |---|---|
  | `config.py` | `Phase04Config`; every threshold with provenance |
  | `metrics.py` | Pure metrics: top-decile overlap, rank change, variance share, κ |
  | `quality_eligibility.py` | Study Q: structural vs signal flags, κ, athlete eligibility |
  | `smoothing.py` + `smoothing_figures.py` | Study B: recompute under sg5/7/9/11 → `gold/phase_04/quality/trajectory_features_by_smoothing` + `movement_events/events_by_smoothing`; trajectory and athlete agreement, BA, ICC per setting; 9 figures |
  | `events.py` | Study C1: rates per valid second, rule/SG count stability, event-level matching |
  | `annotation_design.py` | Study C2 v2: Combine + NFL tasks, random vs difficult, athlete dev/eval split, eval per SG × rule × tolerance |
  | `validation_registry.py` | Study V: VALIDATED / PROVISIONAL / EXPERIMENTAL / REJECTED per feature × drill |
  | `report.py`, `__main__.py` | Report filler and CLI |

- **Infrastructure change:** `StudyStore(phase=…)`, defaulting to `phase_03`, so Phase 04 writes to `{tables,figures,gold}/phase_04/` with its own manifest and `_sample`. New STUDY_DIRS are Q, C1, C2 and V. The Polars guard now covers `phase04/`.
- **Tests (not run):** `tests/test_phase04_m1.py` (32 tests, synthetic data).
- **Notebook:** `notebooks/09_phase04_scientific_validation.ipynb` (viewer only).
- **Report:** template at `reports/PHASE_04_SCIENTIFIC_RESEARCH_REPORT_TEMPLATE.md`; output at `reports/PHASE_04_SCIENTIFIC_RESEARCH_REPORT.md` (written by `--report`).
- **Design decisions:**
  - Interior peaks and robust quantiles (p95/p90) are tested alongside raw peaks. This is motivated by A3 and is not post-hoc.
  - Event features cannot be VALIDATED without eval-split labels.
  - NFL features stay ≤ PROVISIONAL until NFL repeatability exists (Milestone 2).
  - C2 v2 task selection is detector-blind.
  - No threshold is tuned to rebalance tiers or statuses.

### Manual execution (USER) — Phase 04 M1
```
python -m pytest tests/test_phase04_m1.py tests/test_polars2_compat.py -q
python -m nflbdb.research.phase04 --only Q
python -m nflbdb.research.phase04 --only B --sample-athletes 120        # smoke → */phase_04/_sample
python -m nflbdb.research.phase04 --only B                              # heavy: 4 SG settings × (0.46M Combine + NFL sample)
python -m nflbdb.research.phase04 --only C1
python -m nflbdb.research.phase04 --only V
python -m nflbdb.research.phase04 --check
python -m nflbdb.research.phase04 --report
python -m nflbdb.research.phase04 --annotation-tasks                    # then label (protocol), then:
python -m nflbdb.research.phase04 --evaluate-annotations reports/annotation/phase_04/annotation_labels.csv
python -m nflbdb.research.phase04 --only V                              # re-run once C2 exists
```
**Share back:**
- test output;
- `Q_flag_agreement`, `Q_athlete_eligibility`;
- `B_smoothing_agreement` (athlete level, primary strata), `B_repeatability_by_setting`, `B_variance_share`;
- `C1_event_rates` (drill = ALL), `C1_count_stability` (athlete);
- `V_feature_status_summary`;
- figures B_01, B_06, V_01 and C1_03.

### Exact next step
1. Review the Phase 04 M1 outputs, write the interpretation and decision cells in notebook 09 and the report, and freeze the Milestone 1 feature list.
2. Then Milestone 2: the athlete feature store (levels B–F: maneuvers, Combine attempt, NFL player-play windows, athlete-season, research base), restricted to admitted features, plus NFL reliability versus the number of plays.

---

_Phase 03 history below (2026-10-09 and earlier)._

## Project objective
Do Combine movement signatures carry information about NFL performance? First, build a small, reliable, interpretable movement-feature library with defensible diagnostics.

## Verified evidence (executed by the user earlier, not re-computed)
- M1 kinematics, M2 Combine features, and Study A (A1/A2) are all executed; Study A tables exist in `reports/tables/phase_03/{descriptive,statistical}/`.
- Phase 01 non-PASS checks, from `data_quality_results.csv`:
  - CRITICAL: Combine key `(event_id, nfl_id, attempt, time)`, 16 dup rows in a 2 % sample. Hypothesis: BALL rows; `sources.py` notes 32,095 = all BALL rows.
  - MEDIUM: the same key without `attempt`.
  - WARN: Combine `s` range (177 rows in the sample, max 25.6) and Combine `a` range (3 rows).
  - HIGH: game_tracking → player_play play orphans (150 of 104,545 keys, sampled).
  - MEDIUM: player-level orphans (1,879 of 237,177, sampled).
  - WARN: game jumps (4).
- Event-tag vocabulary (profile, sampled): `ball_snap`, `pass_forward`, `handoff`, `run`, `qb_sack`, `play_action`, `man_in_motion`, …
- `games.season_type`: REG / PRE / POST.

## Milestone 1 — authored 2026-10-09 (NOT run)
| File | Purpose |
|---|---|
| `features/config.py` | **+** `PLAY_WINDOWS_VERSION = "1"`, `WindowSpec`, `PlayWindowConfig` (separate dataclass, so M1/M2 caches are untouched). Windows: `pre_snap`, `post_0_1p5`, `post_0_3`, `snap_to_release`, `post_snap_all`. |
| `features/signal_integrity.py` | Filter-boundary flags, within-segment jerk, Δt irregularity, missing frames across gaps, vendor RMSE, heading/low-speed consistency, and interior vs all peaks with `*_in_boundary`. |
| `features/play_windows.py` | `play_anchors` (snap/release from event tags, flags), `window_bounds`, `window_features` (speed, displacement in standardised x, path, high-speed distance, sustained accel/decel, raw first-step time, censoring/coverage). |
| `research/nfl_windows.py` | `NflWindowBuilder`. Datasets `nfl_play_anchors` and `nfl_player_play_windows` (gold/phase_03, partitioned by season). Five `m11_nfl_*` tables and three figures. Bounded runs go to `phase_03/_sample`. CLI with `--check`. |
| `research/studies/dq_failures.py` | **Study A0**: exact bronze drill-down of every non-PASS check (`uniq` / `range` / `fk` / `trk`). Classification functions are pure and tested. Outputs `A0_dq_failures`, `A0_dq_failure_breakdown`, `A0_dq_failure_examples`, plus figure A0_01. |
| `research/studies/signal_reliability.py` | **Study A3**: signal metrics (Combine plus the NFL sample), boundary-peak share vs expected, tier/metric discrimination AUC (cluster bootstrap, `shares_source` flag), sampling table, three figures. |
| `research/studies/{config,store,__main__}.py` | Adds `IntegrityConfig` and `SignalStudyConfig` with provenance docs, registers STUDY_DIRS `A0` and `A3`, and registers both studies (order A0 → A → A3). |
| `tests/test_m1_1_{signal_integrity,play_windows,dq_failures,signal_study}.py` | Static tests covering boundaries, duplicate timestamps, irregular sampling, low speed, smoothing edges, snap exclusion, censoring, play isolation, standardised coordinates, cache reuse/invalidation, DQ classification, and an end-to-end synthetic drill-down. |
| `notebooks/08_m1_1_foundation.ipynb` | Thin loader/viewer with interpretation placeholders. |
| `docs/M1_1_RESEARCH_PLAN.md` §3b | Maps the brief → studies, plus the A3 code audit table. |

**Path mapping** (brief → actual convention):

| Brief path | Actual path |
|---|---|
| `reports/tables/m1_1/dq_failures.csv` | `reports/tables/phase_03/descriptive/A0_dq_failures.csv` |
| NFL player-play features | `warehouse/gold/phase_03/nfl_player_play_windows/` |
| NFL tables | `reports/tables/phase_03/m11_nfl_*.csv` |

**Design decisions:**
- Kinematics are computed on whole trajectories, then windowed, so the snap never creates a filter edge.
- The first-step time uses raw displacement, because SG smears up to 0.3 s across the snap.
- NFL windows use `x_std` / `y_std`.
- REG season only by default.
- Diagnostic figures keep the project style; the navy 16:9 identity is reserved for the publication layer (Milestone 4).

## Outstanding validation questions
- Is the BALL-row hypothesis behind the CRITICAL Combine key failure confirmed (A0 `rule_error` + silver resolved)?
- Are the `s` / `a` WARN rows BALL-only?
- How many linked athletes lose play context through the orphans?
- Do peaks concentrate on filter-boundary frames, and do the tiers discriminate (AUC)?
- What are snap-tag coverage and the pre-snap peak share?
- Combine coordinate handedness is still unverified.
- `PlayWindowConfig` fallback snap names (`snap_direct`, `autoevent_ballsnap`) were not observed in the sample.

## Manual execution (USER) — M1.1 foundation
```
python -m pytest tests/test_m1_1_signal_integrity.py tests/test_m1_1_play_windows.py tests/test_m1_1_dq_failures.py tests/test_m1_1_signal_study.py tests/test_polars2_compat.py -q
python -m nflbdb.research.studies --only A0
python -m nflbdb.research.studies --only A3 --sample-athletes 300 --skip-nfl-sample     # smoke → phase_03/_sample
python -m nflbdb.research.studies --only A3
python -m nflbdb.research.nfl_windows --seasons 2023 --limit-games 10                  # pilot → phase_03/_sample
python -m nflbdb.research.nfl_windows
python -m nflbdb.research.nfl_windows --check
```
Then run notebook 08 top to bottom.

**Share back for review:**
- `A0_dq_failures.csv` and `A0_dq_failure_breakdown.csv`.
- `A3_boundary_peak_share.csv`, `A3_tier_discrimination.csv` and `A3_sampling_irregularity.csv`.
- `m11_nfl_anchor_coverage.csv`, `m11_nfl_window_coverage.csv` and `m11_nfl_presnap_peak_share.csv`.
- Figures A0_01, A3_01–03 and the `m11_nfl_*` figures.
- Any test failures.

## Exact next step
1. Interpret the user's A0 / A3 / window outputs, then fill notebook 08.
2. Next, author Study B (`studies/smoothing.py`) and C1 (`studies/events.py`).
3. Then Milestone 2: maneuver segmentation (C3), sprint robustness (D) and agility (E).
4. Milestone 3 (role-specific NFL windows: DL get-off, DB transition, WR release, OL set) builds on `nfl_player_play_windows`.

## Earlier state (2026-10-08) — kept for history
_M1.1 + M2 research studies A–I. Shared utilities, Study A and the C2 annotation framework were authored then; Study A has since been run. M2.1 Workstreams B–G are folded into Studies D–G (see the plan). M2.1 Workstream A (duplicate audit) remains and feeds Studies A, D, E and F._

## Studies A–I (current) — plan: `docs/M1_1_RESEARCH_PLAN.md`
- **Shared code (authored, not run):**
  - `research/contracts.py`: `TableContract`, `conform`, `cast_keys`, `upstream_state`. Moved out of `m21/io.py`, which re-exports them.
  - `research/figkit.py`: `figure`, `finish`, `empty_panel`, `equal_aspect_panels`, `ci_errorbar`. Moved out of `m21/plotting.py`, which re-exports them.
  - `research/studies/`:
    - `stats.py`: cluster bootstrap, Wilson, Cliff's δ, Spearman, Wilcoxon, BH, Bland–Altman, TOST decision.
    - `results.py`: §15 schema plus a `grain` rule — `n_frames` only at frame grain, and fewer than `min_athletes` athletes → `insufficient_data`.
    - `store.py`: study folders, run cache, `figure_index.csv`, `table_index.csv`.
    - `config.py`: parameters with provenance.
    - `report.py`: template population from saved tables.
    - `__main__.py`: orchestrator.
- **Study A** (`studies/quality.py` + `quality_figures.py`):
  - A1: 8 quality dimensions, p99 screening flags, proportions with Wilson and cluster CIs, Cliff's δ, Spearman vs duration, funnel, reason heatmap.
  - A2: vendor vs derived MAE / RMSE / bias / LoA with cluster CIs, by domain, cohort, drill, position, tier and speed band.
  - 20 figures, each indexed with its research question.
- **C2 framework** (`studies/annotation.py`): blind stratified tasks, packets, label schema and validation, one-to-one matching within tolerance, error taxonomy, P/R/F1 with cluster CIs, with and without uncertain labels.
- **Report template:** `reports/phase_03_integrated_report_TEMPLATE.md`, using `{{tbl:…}}` tokens.
- **Notebook:** `notebooks/07_m1_1_scientific_validation.ipynb`. Numbering starts at 07 because 06 is taken.
- **Tests (not run):** `tests/test_studies_stats.py`, `tests/test_study_a_quality.py`, `tests/test_event_annotation.py`. The `test_polars2_compat.py` guard now also covers `research/studies/`.
- **M1 evidence already persisted:**
  - Combine is 6,309 / 6,310 tier A; NFL is ≈ 99 % A.
  - Vendor speed median |Δ|: 0.030 (Combine) and 0.058 (NFL sample) yd/s.
  - Smoothing: min Spearman 0.92 (NFL braking count).
  - Implication: the tiers barely discriminate, so A1 studies continuous dimensions.
- **Assumptions:**
  - The athlete is the independent unit.
  - Screening thresholds are proposed, not proof.
  - Heading validity is content-confounded.
  - Vendor speed is not ground truth.
- **Open issues:**
  - The NFL frame sample is not persisted (A2 NFL is a sample).
  - The smoothing-instability dimension is null until Study B runs.
  - Study A cluster-bootstrap runtime on 315k NFL trajectories is unknown. If it is too slow, lower `n_boot_generic`.
- **Next implementation step:** Study B (`studies/smoothing.py`, reusing `kinematic_diagnostics.smoothing_sensitivity` and `trajectory_peaks`), then C1 (`studies/events.py`).

### Studies — manual execution (USER)
```
python -m pytest tests/test_studies_stats.py tests/test_study_a_quality.py tests/test_event_annotation.py tests/test_m21_infra.py tests/test_polars2_compat.py -q
python -m nflbdb.research.studies --only A --sample-athletes 300 --skip-nfl-sample   # smoke → */phase_03/_sample
python -m nflbdb.research.studies --only A                                           # full (NFL sample recomputed, 8 games/season)
python -m nflbdb.research.studies --check                                            # exit 0 = Study A complete
python -m nflbdb.research.studies --annotation-tasks                                 # then label → --evaluate-annotations <csv>
python -m nflbdb.research.studies --report                                           # → reports/phase_03_integrated_report.md
```

## M2.1 (current milestone) — see `docs/M21_PROGRESS.md` for the full log
- **Authored (not run):**
  - `src/nflbdb/research/m21/`: `config.py`, `io.py`, `plotting.py`, `duplicates.py` (WS A), `__main__.py`.
  - `tests/test_m21_duplicates.py`, `tests/test_m21_infra.py`.
  - Polars AST guard extended to `research/m21/`.
  - `combine_analytics.fig_three_cone` export fix.
- **Namespace:** `reports/{tables,figures}/phase_03/m2_1/`, `warehouse/gold/phase_03/m2_1/`. Sampled runs go to `…/m2_1_sample/`.
- **User-reported M2 observations (NOT re-verified by Claude):**
  - Forty: 793 attempts, 758 fit-ok; r(official) ≈ 0.97 with +0.25 s bias; 10-yd r ≈ 0.77.
  - ICC: `trk_t20_40_s` 0.992, `speed_max` 0.985.
  - 2025 is ~4× smoother.
  - One athlete has SD(vmax) ≈ 0. WS A's `zero_variance_athletes.csv` explains such cases.
- **Next:**
  - User runs the M2.1 A commands below.
  - Then: Workstream B (sprint robustness) in `research/m21/sprint.py`, then C–G, the report template, fingerprint contracts and the registry update.

### M2.1 manual execution (USER)
```
python -m pytest tests/test_m21_infra.py tests/test_m21_duplicates.py tests/test_polars2_compat.py -q
python -m nflbdb.research.m21 --only A --sample-athletes 200     # smoke → phase_03/m2_1_sample
python -m nflbdb.research.m21 --only A                           # full run
python -m nflbdb.research.m21 --only A --check                   # exit 0 = all A outputs present
python -m nflbdb.research.combine_analytics                      # re-renders m2_three_cone_path_extremes (fix)
```

## Current objective
Close M2 (Combine movement analytics + reliability) with defensible outputs. Claude works in code-authoring-only mode; the user runs everything.

## Environment (user runs)
Conda env `nfl`: Python 3.10, Polars 2.0.0, DuckDB 1.5.6. `pip install -e .`. Set `NFL_BDB_DUCKDB_MEMORY=3GB`. Restart notebook kernels after code edits.

## M1 (executed; facts in report 14 §6)
- Vendor speed r = 0.9996 / 0.998. `dir` = compass convention. `a` = magnitude.
- Combine 6,309 / 6,310 tier A. The forty recording starts in motion and ends at peak speed. The shuttle has pre-start drift.
- 2025 Combine residual noise ≈ ¼ of 2023/24.
- `KinematicsConfig` v1 kept, and M1 artifacts are untouched.

## M2 hardening (this session), by file
- **`features/agility.py`:** Polars 2.0 fix. A list was mixed with a positional expression in `.with_columns`; it is now unpacked with `*`.
- **`features/config.py`:**
  - `M2_VERSION = "2"`. This invalidates only v1 M2 outputs. They were never built successfully, or were built just once after the fix, so this is a needed rebuild, not a forced one.
  - New parameters: `max_model_vmax = 12.5`, `end_censor_speed = 2.0`, `max_attempts = 4`, `min_athletes_label = 20`, `max_modal_share = 0.8`.
- **`features/sprint.py`:**
  - The fit window starts one frame before speed first reaches `onset_speed`, so a stationary lead-in no longer pushes t0 off the grid.
  - `fit_ok` also requires `vmax ≤ 12.5` and `vmax ≤ 1.10 × observed peak`.
  - New columns: `fit_start_rel_s`, `vmax_extrapolation_ratio`.
  - Tracked splits are renamed `trk_t10_s`, `trk_t20_s`, `trk_t40_s`, `trk_t10_20_s`, `trk_t20_40_s`. Official times exist only as `cmb_*`.
- **`features/temporal.py`:** `end_censored` (last-frame speed ≥ 2 yd/s).
- **`features/braking.py`:** optional `window`. Only braking events starting inside [onset, end of activity] count, and the builder passes it.
- **`features/reliability.py`:**
  - Attempts are capped at the first 4 per athlete.
  - Labels: `insufficient_for_label` below 20 athletes; `degenerate_distribution` when the modal share is ≥ 0.8.
  - CV only for strictly positive features; rank ρ needs ≥ 10 athletes; `modal_share` column.
- **`features/feature_registry.py`:** `trk_*` entries, `fit_start_rel_s`, `vmax_extrapolation_ratio`, `end_censored`, `HIGHER_IS_BETTER` renamed, `resid_rms_yds` documented.
- **`research/combine_movement.py`:** carries `resid_rms_yds` from M1 quality (a QC column, excluded from features). Passes the window to braking. Uses `trk_*` sprint features. Carries `end_censored` into the sprint table.
- **`research/combine_analytics.py`:**
  - `best_attempt` takes each athlete's whole fastest attempt, so flags and values come from the same attempt.
  - `timing_validation` labels `official_measure` / `official_source` / `tracked_measure` / `tracked_basis`. It reports bias median/mean, p05/p95, MAE after bias, a heuristic `agreement_class`, and truncation shares.
  - `cohort_shift` is within-position stratified, with an official-time anchor (`cmb_*`), the M1 `noise_ratio`, a `noise_sensitive` flag and a heuristic `interpretation`.
  - `forty_window_diagnostics` adds end-censoring, stationary lead-in and vmax-extrapolation shares.
  - Every table is written as CSV + Parquet. An empty table becomes a placeholder note.
  - New `TABLES`, `expected_outputs()`, `check_outputs()`, `load_table()`, and a CLI `--check`.
- **Tests (new, not run):**
  - `tests/test_polars2_compat.py`: an AST guard over `features/` and `research/`, plus `agility_features` zero/empty cases.
  - `tests/test_phase03_m2_hardening.py`: stationary lead-in, vmax-extrapolation guard, same-attempt flags, timing labels and bias, braking window, reliability cap/label/degenerate/CV, cohort-shift interpretation, output-path contract.
  - `tests/test_phase03_m2_combine.py`: updated to the `trk_*` names.
- **Notebook 06:** Parquet reload / export, label legend, `trk_*` names. Notebook 03 is unchanged.
- **Reports:** 13 (output semantics + acceptance criteria) and 15 (guards).

## Audit notes (no change needed)
- Single-list `.with_columns([...])` / `.select([...])` calls are valid and already executed in M1.
- `rolling_mean(min_samples=)`, `Expr.get(arg_max)` and `pivot(on=, index=, values=)` use the Polars ≥ 1.21 / 2.0 API.
- The COD merge (≤ 0.3 s gap) never mixes segments or rules. Reversal ≠ turn by speed regime. Attempts without any onset keep null COD/braking counts, because they are undefined rather than zero.
- Leakage: M2 is descriptive. Official times are used only for validation, cohort anchors and example selection. Nothing uses NFL outcomes. No standardisation is fitted.

## Manual execution order (USER)
```
python -m pytest tests/test_polars2_compat.py tests/test_phase03_m2_hardening.py tests/test_phase03_m2_combine.py -q
python -m pytest tests -q
python -m nflbdb.research.combine_movement          # builds M2 v2 once; later runs reuse
python -m nflbdb.research.combine_movement          # must log "reuse … (current)" ×4
python -m nflbdb.research.combine_analytics         # 9 tables (CSV+Parquet) + 13 figures
python -m nflbdb.research.combine_analytics --check # exit 0 = all 44 outputs present
```
Then restart the kernel and run notebooks 03 and 06 top to bottom.

## Expected outputs / acceptance
- **`warehouse/gold/phase_03/`:** `combine_attempt_features`, `combine_cod_events`, `combine_sprint_profiles`, `combine_athlete_drill_features` (+ `.schema.json`).
- **`reports/tables/phase_03/`:** `forty_window_diagnostics`, `timing_validation`, `agility_cod_structure`, `combine_feature_summary`, `cohort_shift`, `feature_reliability`, `feature_redundancy`, `position_comparisons`, `m2_example_selection` (each `.csv` + `.parquet`) and `feature_dictionary`.
- **`reports/figures/phase_03/m2_*`:** 13 figures × PNG/SVG.
- **Acceptance criteria:** report 13 §3. In short:
  - all tests pass;
  - `--check` is clean;
  - tracked times are called official estimates only if `interchangeable_after_bias` holds in every cohort;
  - "reliable" means a good/excellent label;
  - processing-artifact rows are excluded or cohort-adjusted.

## Remaining risks
- Polars 2.0 behaviour is unverified until the tests run (first suspects: `rank().over`, `pivot`, list-of-dicts DataFrame construction).
- The agreement-class and cohort-interpretation thresholds are heuristics.
- The forty may still fail `fit_ok` often if recordings stop well before the plateau (watch `share_vmax_extrapolated_gt10pct`).
- Physical coordinate handedness is unverified.

## Exact next task (M2.1)
Get the WS A test output and the run's results from the user: `duplicate_flag_summary` (cohort = all, `group:*`), `zero_variance_athletes` explanation counts, `duplicate_sensitivity` for the forty, and the 3 `m21_A_*` figures. Interpret them, then author WS B.

## Earlier next task (M2, superseded by the user's report that M2 ran)
Get the test output and the `--check` result from the user. Review `forty_window_diagnostics`, `timing_validation`, `cohort_shift` (interpretation counts) and `feature_reliability` (label counts). Fill in report 13 §5 and report 15. Only after acceptance: fingerprints, then M3.

## Handoff (fresh session)
> **M1.1 foundation (current, 2026-10-09):** Read only the top of this file and `docs/M1_1_RESEARCH_PLAN.md` §3b. If the user sends A0 / A3 / NFL-window outputs or test failures, fix and interpret them, and fill notebook 08. Otherwise author Study B (`studies/smoothing.py`), following `signal_reliability.py` (contracts → `ResultCollector` → `StudyStore` → `FigureRecord`), and register it in `studies/__main__.py`. Execute nothing.
>
> **Studies A–I (2026-10-08):** Read this file and `docs/M1_1_RESEARCH_PLAN.md` only. If the user sends Study A outputs, interpret them and fill notebook 07 sections K and N. Otherwise author Study B in `research/studies/smoothing.py`. Use the same pattern as `quality.py`: contracts with `area`, `ResultCollector`, `StudyStore`, figures registered as `FigureRecord`. Then register it in `studies/__main__.py` `STUDIES`. Execute nothing.
>
> **Older handoff (M2.1, superseded for B–G):** Phase 03 M2.1. Read `docs/SESSION_STATE.md` and `docs/M21_PROGRESS.md` only. Infrastructure and WS A are authored, not run. If the user sends A results, interpret them. Otherwise continue with WS B, reusing `m21/io.py` contracts, the `m21/config.py` threshold docs and `m21/plotting.finish`. Register each new workstream in `m21/__main__.py`. Do not rebuild M1/M2. Execute nothing.
