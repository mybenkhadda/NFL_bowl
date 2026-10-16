# Phase 04 — implementation map, reuse and validation gates

_Authored 2026-10-11 in code-authoring-only mode. Nothing in this document was executed by Claude. Numbers marked **[executed]** come from CSVs that the user's runs already saved; everything else is design._

## 1. Context summary (what exists before Phase 04)

| Area | State | Evidence |
|---|---|---|
| Lakehouse (bronze / silver / gold, DuckDB catalog) | executed + verified | Phase 01; 27 tests passed |
| M1 kinematics (segmentation, NumPy SG, events × 3 rules, tiers) | executed | `gold/phase_03/combine_*`, `nfl_*`; report 14 |
| M2 Combine attempt features, sprint model, COD, ICC | executed (v2) | `tables/phase_03/feature_reliability.csv`, … |
| Study A (A1 quality dimensions, A2 vendor agreement) | executed | `tables/phase_03/{descriptive,statistical}/A*` |
| Study A0 (exact drill-down of Phase 01 failures) | executed, **not yet reviewed** | `A0_dq_failures.csv` |
| Study A3 (signal reliability, boundary peaks, tier AUC) | executed, **not yet reviewed** | `A3_*.csv` |
| Snap-anchored NFL windows | executed, **not yet reviewed** | `m11_nfl_*.csv` |
| C2 annotation framework (Combine only) | authored; never labelled | `studies/annotation.py` |

**Grains in use:** frame → segment → trajectory (Combine attempt `(event_id, nfl_id, attempt)`; NFL player-play `(game_id, play_id, nfl_id)`) → athlete × drill → athlete. The athlete is the independent unit everywhere.

**First reading of the executed Phase 03 outputs (descriptive, for the user's review):**
- **A0 [executed].** The CRITICAL Combine key failure is a `rule_error`: 32,095 duplicate keys, all of them BALL rows. The duplicates vanish once `entity_type` joins the key, and silver is unique. The `s` / `a` range WARNs are BALL-only, with 0 athlete rows. Orphan play contexts are a `coverage_gap`: 202 play keys and 2,497 player-play keys, affecting 124 linked athletes. They stay in role-agnostic kinematics but are excluded from context features. No blocking defect remains.
- **A3 [executed], tier discrimination.** `tier_not_A` has AUC 0.50 for every target in the Combine data. The continuous metrics do discriminate: `speed_vendor_mad` 0.73 (vendor target, shares a source) and 0.82 (jerk). `resid_rms_yds` reaches 0.92 for the vendor target in the NFL sample. **The tier label carries no information; continuous metrics do.**
- **A3 [executed], boundary peaks.** Combine `peak_speed` lands on SG-boundary frames 2.7× (forty) to 11.4× (three-cone) more often than chance, with OL drills at 7.6× and the shuttle at 6.7×. Forty `peak_abs_yaw_rate` reaches 6.5× (low-speed start). Peak `a_tan` is ≤ 0.65× everywhere. Peak deceleration ranges from 0.06× to 1.8× (highest: OL 1.7×, LB 1.8×, where LB has only 4 athletes). NFL peaks are rarely on boundary frames (≤ 0.7×). Reading: in Combine drills the raw peak speed is frequently an **edge value** (the recording ends at or near peak speed), so raw `peak_speed` is suspect there. Study B tests the interior and quantile alternatives.
- **M1 smoothing [executed].** Spearman ≥ 0.92 everywhere, but the median relative change reaches 14 % (Combine `peak_decel`, sg5p2) and 3–9 % for the other acceleration and yaw peaks (Combine and NFL). Ranks are stable; magnitudes are not.
- **NFL windows [executed].** A snap tag is present in ≥ 99.99 % of REG plays and a release tag in 98.7–99.7 %.

## 2. Phase 04 implementation map

| Brief item | Implemented as | Module | Status |
|---|---|---|---|
| M1 · 1.1 quality tiers, structural vs signal, eligibility | Study A/A3 (executed) + **Study Q** | `research/phase04/quality_eligibility.py` | CODE AUTHORED |
| M1 · 1.2 signal agreement + smoothing (traj / athlete, BA, top-decile, rank change, variance, repeatability) | **Study B** | `phase04/smoothing.py`, `smoothing_figures.py` | CODE AUTHORED |
| M1 · 1.3 event detectors: rule × smoothing sensitivity, exposure-normalised | **Study C1** | `phase04/events.py` | CODE AUTHORED |
| M1 · 1.3 annotation dataset, matching, P/R/F1, timing, taxonomy, dev/eval | **Study C2 v2** (reuses C2 matcher / evaluator) | `phase04/annotation_design.py`, `docs/PHASE_04_ANNOTATION_PROTOCOL.md` | CODE AUTHORED; labels needed |
| M1 decision VALIDATED / PROVISIONAL / EXPERIMENTAL / REJECTED | **Study V** | `phase04/validation_registry.py` | CODE AUTHORED |
| Report | template + filler (Phase 04 + Phase 03 manifests) | `reports/PHASE_04_SCIENTIFIC_RESEARCH_REPORT_TEMPLATE.md`, `phase04/report.py` | CODE AUTHORED |
| M2 athlete feature store (levels B–F) | maneuvers (C3), Combine attempt, NFL player-play windows, athlete-season, research base | planned: `phase04/feature_store/` | NOT STARTED |
| M3 Combine analytics · M4 NFL analytics · M5 transfer · M6 registry | Studies D/E/F/G, NFL windows (role-specific), H, I | planned | NOT STARTED |

**Brief → repository path mapping:**
- `src/features/reliability.py` exists already as `features/reliability.py` (M2 ICC), and Study B reuses it.
- `src/evaluation/bootstrap.py` is `research/studies/stats.py`: the athlete-cluster bootstrap, BH, Bland–Altman and Cliff's δ.
- `src/reporting/*` is `research/figkit.py`, `studies/store.py` (figure and table indexes) and `phase04/report.py`.
- The brief's notebook numbering (`06_…`) clashes with existing notebooks 05–08, so Phase 04 notebooks start at `09_phase04_scientific_validation.ipynb`.

## 3. Reused components (no duplication)
- **`features/`:** `compute_kinematics`, `prepare_frames`, `detect_events`, `add_signal_columns` (boundary flags, interior peaks), `icc_oneway`, `icc_bootstrap` and `reliability_label`.
- **`research/kinematic_build`:** `load_combine_raw`, `load_nfl_raw`, `nfl_season_files`, and the deterministic NFL game sample (the same selection as M1, A2 and A3).
- **`research/studies`:**
  - `StudyStore`, which gained a `phase` argument so Phase 04 writes under `phase_04/`.
  - The `TableContract` / `conform` contracts and `ResultCollector`, including the grain and status rules and BH.
  - From `stats`: `cluster_bootstrap`, `cluster_bootstrap_ratio`, `spearman` and `bland_altman`.
  - From `annotation`: `match_events`, `match_all`, `evaluate`, `export_packets` and `validate_labels`.
  - The report token engine.
- **Persisted Phase 03 inputs:** `trajectory_quality_dimensions` (A1), `trajectory_signal_reliability` (A3), `A3_boundary_peak_share`, `A2_vendor_error_summary` and `combine_trajectory_quality`.

## 4. Outstanding measurement-validity questions (Milestone 1)
1. Is raw `peak_speed` in Combine agility and position drills an edge artefact? If so, do `peak_speed_interior` / `p95_speed` replace it without losing repeatability? (B + A3 → V)
2. Do acceleration and deceleration **magnitudes** move enough under smoothing to forbid absolute thresholds and Combine-vs-NFL magnitude comparisons, even though ranks are stable? (B, Bland–Altman + relative change)
3. Is athlete-level repeatability (ICC) a property of the athlete or of the filter setting? (B, ICC per setting)
4. Are event rates usable as athlete traits? This needs both stable ordering across rules and smoothing (C1) and agreement with human labels (C2). Turns under the strict rule are a known detector artefact.
5. Are structural and signal quality distinct, and how many athletes survive each filter? (Q)
6. Are 2023/24 versus 2025 Combine differences a noise artefact? (B7 cohort robustness + A3 roughness by cohort)
7. Still open from M1:
   - Combine coordinate handedness, which blocks left/right features.
   - NFL within-athlete reliability, which needs context-matched plays (Milestone 2).

## 5. Gates
- **Milestone 1 gate (P0):** `V_feature_status_summary` has been reviewed. Only VALIDATED or PROVISIONAL features (with the open criterion stated) enter the feature store. Event features stay ≤ PROVISIONAL until C2 eval-split labels exist. Thresholds are never tuned to rebalance tiers or statuses.
- **Milestone 2 gate (P1):** athlete-level ICC per feature, with NFL reliability as a function of the number of plays.
- No Combine-to-NFL statistics (M5) before both gates pass.

## 6. Status (code / execution / validation)
| Study | Code | Execution | Validation |
|---|---|---|---|
| Q | CODE AUTHORED | AWAITING USER EXECUTION | NOT STARTED |
| B | CODE AUTHORED | AWAITING USER EXECUTION | NOT STARTED |
| C1 | CODE AUTHORED | AWAITING USER EXECUTION (after B) | NOT STARTED |
| C2 v2 | CODE AUTHORED | AWAITING USER EXECUTION, then manual labelling | BLOCKED on labels |
| V | CODE AUTHORED | AWAITING USER EXECUTION (after Q, B, C1) | NOT STARTED |
