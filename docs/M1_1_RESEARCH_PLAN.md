# Phase 03 · M1.1 + M2 research plan — Studies A–I

_Authored 2026-10-08. Code-authoring only: every number in this document comes from outputs that M1/M2 already executed, and each such number is labelled "M1 evidence". Nothing here is a new result._

## 0. Starting evidence (M1 evidence, persisted CSVs)
| Fact | Source |
|---|---|
| Combine: 6,310 attempts, 6,309 tier A, 1 tier B (vendor-speed disagreement) | `tier_reason_counts.csv`, notebook 05 |
| NFL: 314,944 player-plays, ≈ 99 % tier A, 2,654 B, 306 reject | notebook 05 |
| Combine vendor vs derived speed: median \|Δ\| 0.030 yd/s, p95 0.163, 431,094 frames | `kinematics_vendor_agreement.csv` |
| NFL sample (24 games): median \|Δ\| 0.058 yd/s, p95 0.269, 543,964 frames | same |
| Three-cone / shuttle agreement is the weakest (p95 0.37 / 0.28 yd/s) | same |
| Smoothing: lowest Spearman vs the primary setting ≈ 0.92 (NFL `n_braking_events`), 0.95 (Combine `peak_a_tan`, `peak_abs_yaw_rate`) | `smoothing_sensitivity.csv` |

**What this already implies:**
- The provisional tiers hardly discriminate. Study A1 must therefore study continuous quality *dimensions*, not tier labels.
- Vendor agreement is high in aggregate, and the open question is *where* it fails (drill, speed band, cohort).

## 1. Shared design rules (apply to every study)
- **Template.** Every analysis follows the A–N template of the brief. The study module docstring holds A–F (question … methodology). The notebook holds G–N.
- **Grain is explicit.** Every result row carries `grain` ∈ {frame, segment, trajectory, attempt, event, maneuver, athlete}, plus `n_athletes` and `n_trajectories` (and `n_frames` only for frame statistics). Frames and trajectories are never treated as independent athletes. Uncertainty is computed by **athlete-cluster bootstrap**.
- **Results schema.** Results are stored in one long table per study (`studies/results.py`, brief §15 columns). Exploratory test families get Benjamini–Hochberg FDR. Every result is reported, not only significant ones.
- **Thresholds.** No threshold is evidence of correctness. Screening thresholds are empirical percentiles, documented as *proposed*.
- **Persistence.**
  - Datasets: `warehouse/gold/phase_03/<area>/` (quality, movement_events, maneuvers, athlete_profiles, research_base).
  - Tables: `reports/tables/phase_03/<category>/` (descriptive, statistical, reliability, hypothesis_screening).
  - Figures: `reports/figures/phase_03/<study>/`.
  - Indexes: `figure_index.csv`, `table_index.csv`.
- **Read-only inputs.** Heavy work reads persisted M1/M2 data. The only recomputation is bounded and deterministic (NFL game sample, alternative smoothing settings).

## 2. Studies, data, reuse
| Study | Question | Grain | Inputs | Reused code | New module | Milestone |
|---|---|---|---|---|---|---|
| A1 Quality calibration | Do tiers / quality dimensions separate reliable from unreliable trajectories? | trajectory | `*_trajectory_quality`, `*_segment_quality`, `combine_kinematic_events` (all rules), `dim_athlete` | `segment_quality`, `kinematic_diagnostics.quality_summary` | `studies/quality.py` | 1 |
| A2 Vendor vs derived | Where and how much does derived speed disagree with vendor speed? | frame (+ trajectory, athlete for CIs) | `combine_frame_kinematics`, NFL deterministic sample | `KinematicDiagnostics.nfl_sample`, `vendor_agreement` | `studies/quality.py` | 1 |
| B Smoothing robustness | Which features keep values / athlete ranks across SG settings? | trajectory (numeric) · athlete (rank) | raw Combine + NFL sample re-smoothed | `smoothing_sensitivity`, `compute_kinematics`, `prepare_frames` | `studies/smoothing.py` | 1 (next) |
| C1 Threshold sensitivity | How do event counts / durations respond to rule choice? | event · trajectory | `*_kinematic_events` (3 rules) | `event_threshold_sensitivity`, `detect_events` | `studies/events.py` | 1 (next) |
| C2 Reference annotation | Precision / recall / timing error of the detectors vs human labels | event, clustered by athlete | annotation CSVs (user-labelled) | — | `studies/annotation.py` (**authored**) | 1 |
| C3 Maneuvers | Approach → brake → transition → reaccelerate → recover segments | maneuver | frames + events | `features.agility` (COD), `braking` | `studies/maneuvers.py` | 2 |
| D Forty decomposition | Same official time, different signatures? | attempt → athlete | `combine_sprint_profiles`, frames | `features.sprint`, M2.1 WS A flags | `studies/sprint.py` (absorbs M2.1 B + C) | 2 |
| E Agility / COD | Speed preservation, braking efficiency, reacceleration | maneuver → athlete | C3 maneuvers, `combine_cod_events` | `features.agility` | `studies/agility.py` (absorbs M2.1 D) | 2 |
| F Reliability | Variance from athlete vs attempt vs smoothing | attempt → athlete | M2 attempts, B outputs, WS A flags | `features.reliability` | `studies/reliability.py` (absorbs M2.1 E) | 3 |
| G Positions | Within-position profiles, PCA, cluster stability | athlete | F-validated features | `combine_analytics.position_comparisons` | `studies/positions.py` (absorbs M2.1 F/G) | 3 |
| H Transfer | Combine ↔ NFL movement features (exploratory) | athlete | athlete profiles + NFL player-play features | — | `studies/transfer.py` | 4 |
| I Selection | Missingness, participation, quality-filter selection | athlete | all of the above + `mart_athlete_research_base` | — | `studies/selection.py` | 5 |

The M2.1 Workstream A duplicate audit (`research/m21/duplicates.py`) stays as is and feeds A (integrity), D, E and F.

## 3. Study A detail (implemented this session)
**A1 — quality dimensions.** One row per trajectory, saved as `quality/trajectory_quality_dimensions`:

| dimension | metric(s) | direction |
|---|---|---|
| structural integrity | n_segments, n_gap_breaks + n_jump_breaks, n_null_xy_removed, n_dup_ts_removed | higher = worse |
| temporal continuity | offgrid_rate = n_offgrid / n_valid_frames | worse ↑ |
| coordinate validity | resid_rms_yds | worse ↑ |
| speed consistency | speed_vendor_mad | worse ↑ |
| derivative reliability | max_accel_mag, accel-implausible count | worse ↑ |
| heading reliability | 1 − frac_heading_valid (frame-weighted) | worse ↑ |
| event stability | lenient − strict event count, per event type (Combine) | worse ↑ |

- **Screening flag.** A dimension is flagged when it exceeds the domain × drill-type p99 (*proposed*, and the percentile is configurable). The composite is the number of flagged dimensions.
- **Smoothing instability** joins in from Study B once B has run. The column is reserved and stays null until then.

**Statistics:**
- Tier and flag proportions by domain, cohort, drill, position and duration bin, each with a Wilson CI and an athlete-cluster bootstrap CI.
- Metric distributions by tier: Cliff's δ (A vs non-A), with an athlete-cluster bootstrap CI.
- Spearman ρ between each quality metric and duration.
- A retention funnel under four filter scenarios.

**A2 — vendor agreement.** The frame residual is r = derived speed − vendor `s` (valid frames). For each group (domain, cohort, drill, position, tier, speed band):
- MAE, RMSE, mean / median signed error, median |r| and p95 |r|.
- Relative error for s ≥ 1 yd/s.
- Bland–Altman bias and 95 % limits of agreement.
- Cluster-bootstrap CIs (athletes) for MAE, RMSE and bias.

Trajectory-level MAE is also computed against duration. Correlation alone is never used to claim agreement.

## 3b. Brief of 2026-10-09 ("Scientific validation, advanced features & exploratory analytics") → studies
The new brief's Workstreams A–E map onto this plan; nothing is duplicated.

| Brief item | Implemented as | Status |
|---|---|---|
| A1 quality-check remediation (`dq_failures.csv`) | **Study A0** `studies/dq_failures.py` → `tables/phase_03/descriptive/A0_dq_failures.csv` (+ breakdown, examples, figure A0_01) | authored, not run |
| A2 trajectory reliability (structural vs signal) | Study A1/A2 (done) + **Study A3** `studies/signal_reliability.py`, primitives `features/signal_integrity.py` | authored, not run |
| A3 boundary-safe kinematics | audit below; boundary flags, interior peaks, jerk, missing frames (A3); snap-anchored windows never add filter edges | authored, not run |
| A4 event calibration / annotation | C1 (next) + C2 (authored) + C3 maneuvers (M2) | C2 authored |
| B Combine features (sprint model, COD stages, position drills) | D / E / C3; M2 `features/sprint.py` already fits v(t) = vmax(1 − e^(−t/τ)) | M2 ran; D/E pending |
| C NFL context windows | **`features/play_windows.py` + `research/nfl_windows.py`** (anchors + 5 windows); role-specific features are Milestone 3 | authored, not run |
| D validation registry / reliability | F (+ `features/feature_registry.py`) | pending |
| E Combine ↔ NFL linkage | H (+ I selection) | pending |
| §4 figure framework (navy 16:9 identity) | deferred to the publication layer (R / final_story); diagnostic figures keep the project style used by the executed M1/M2 figures | decision |

### A3 code audit (2026-10-09, by reading — no execution)
| Item | Finding | Action |
|---|---|---|
| Derivatives | SG on x, y within segments; v, a analytic; `a_tan = v·a/|v|` signed; yaw via cross product (no wraparound) | none — correct |
| SG edges | `interp`-style: frames within `window // 2` of a segment edge use the off-centre row of the edge window (higher noise gain, worst for the 2nd derivative). No per-frame flag existed | **added** `in_filter_boundary`, interior peaks, `*_boundary_rel`, boundary-peak share (A3) |
| Segmentation | gap > 1.5 × nominal, jump > 1.5 yd per interval, drill change → new segment; dt_ms / step null at breaks | none; **added** missing-frame count across gaps (raw Δt over the trajectory) |
| Sampling | inside a segment Δt is assumed nominal (jitter ≤ 50 ms not split) | **added** `dt_cv`, `dt_max_dev_ms`, off-grid rate per trajectory |
| Low-speed heading | heading / yaw / curvature null below `min_heading_speed` (1 yd/s); a_tan / a_lat null below 0.1 yd/s | none — preserved as null; **added** consistency check (`n_heading_inconsistent` must be 0) |
| Duplicate timestamps | removed after sorting (first kept); counted per trajectory | none |
| Orientation | M1 NFL kinematics use RAW x, y (direction-dependent features impossible). Silver already holds `x_std/y_std/dir_std` (offense → +x, 180° rotation, handedness preserved) | **NFL windows use standardised coordinates**; magnitudes unchanged |
| Pre-snap | M1 NFL summaries span the whole tracked window | **snap-anchored windows** with a separate pre-snap window |
| Snap-relative timing | centred SG smears ≤ `window // 2` frames (0.3 s) across the snap | first-step time uses RAW displacement from the snap position |
| Combine handedness | still unverified (needed before any left/right symmetry feature) | open |

## 4. Milestones & status
| Milestone | Studies | Status |
|---|---|---|
| shared utilities | contracts, figkit, stats, results, study store, report template | **authored, not run** |
| 1 Scientific validation | A (**ran**), A0 + A3 + NFL windows (authored 2026-10-09), C2 framework (authored), B, C1 (next) | in progress |
| 2 Combine biomechanics | C3, D, E | pending |
| 3 Reliability & profiles | F, G | pending |
| 4 Transfer | H | pending |
| 5 Selection | I | pending |
| 6 Integrated report | template authored; populated from saved tables | pending |
