# 14 — Feature engineering (Phase 03 · M1: kinematic foundation)

_Status: **M1 executed by the user (2026-10-08)** — datasets current, diagnostics CSVs/figures on disk, §6 filled from them. **M2 (§8) executed by the user**: the four M2 datasets exist with `version` 2 and `built_at` 2026-10-08T14:59:15Z (`warehouse/gold/phase_03/combine_*_features.schema.json`, `combine_sprint_profiles.schema.json`: 6,310 attempts, 793 forty attempts). §8 describes the design; its results have not been written into this report. Whether the test suites were run is not recorded. (Header corrected 2026-10-11 from those sidecar files.)_

## 1. Evidence used (saved Phase 01 outputs)

| Evidence | Source | Consequence for M1 |
|---|---|---|
| Δt = 100 ms on every sampled frame (game 446k, Combine 9.5k frames, 2 % sample) | `tables/tracking_qc_intervals.csv` | Uniform-sampling SG is appropriate inside a segment; off-grid frames are still counted (`n_offgrid`) |
| Vendor `s` vs derived speed: r = 0.997 (game), 0.998 (Combine) | `tables/tracking_qc_consistency.csv` | Derived speed is consistent with vendor; agreement stays a per-segment QC metric |
| Combine `dis` vs derived step: r = 0.948, median abs diff 0.21 yd | same | Combine `dis` is **not** used; path lengths come from x, y |
| Max Combine `s` 25.6 yd/s; Phase 02 found BALL rows carry athlete ids | `03_data_quality.md`, `09_quality_remediation.md` | Only `entity_type = 'PLAYER'`; plausibility limits → `reject` tier |
| Drill spelling variants occur in disjoint years (`LINE`/`LINE_DRILL`, `SPEED_OUT_LEFT`/`SPEED_OUT_ROUTE_LEFT`, `PASS_PRO_MIRROR_DRILL`/`PASS_PRO-MIRROR_DRILL`, `OVER_THE_SHOULDER_ADJUST`/`OVER_SHOULDER_ADJUST`) | `tables/combine_drill_counts.csv` | Canonical `drill` column (inferred rename, raw name kept); explicit drill lists, no regexes |
| Phase 02 flag datasets not yet built | no `gold/phase_02/_manifest.json` | M1 computes its own integrity metrics; Phase 02 linkage flags are joined in M3 |

## 2. Trajectory model

* **Trajectory**: Combine `(event_id, nfl_id, attempt)` PLAYER frames; NFL `(game_id, play_id, nfl_id)` without the football.
* **Segment**: maximal run inside a trajectory with strictly increasing timestamps (exact duplicates removed and counted), Δt ≤ 1.5 × nominal, displacement ≤ 1.5 yd × max(Δt/nominal, 1), constant `drill_name`. Derivatives never cross a segment boundary.
* Segments shorter than the SG window (7 frames) are kept and flagged (`smoothable = false`); they carry no derivatives.

## 3. Kinematics (`src/nflbdb/features/kinematics.py`)

| Quantity | Definition | Unit | Note |
|---|---|---|---|
| speed | \|SG′(x, y)\| | yd/s | |
| accel_mag | \|SG″(x, y)\| — vector acceleration | yd/s² | includes turning; ≥ \|a_tan\| |
| a_tan | v·a/\|v\| = d\|v\|/dt | yd/s² | signed; braking < 0; undefined below 0.1 yd/s |
| a_lat | (v×a)/\|v\| | yd/s² | signed (+ counter-clockwise) |
| yaw_rate | a_lat/\|v\| | deg/s | wraparound-free; undefined below 1.0 yd/s |
| curvature | yaw_rate/\|v\| | 1/yd | requires validation near the speed floor |
| heading | atan2(vy, vx) | deg | math convention; vendor `dir` convention tested, not assumed |

* SG(7, 2) is the primary setting. Sensitivity settings are SG(5, 2), SG(9, 2) and SG(11, 3), compared using Spearman rank stability across trajectories.
* Edges are handled like SciPy `mode="interp"`. Centred SG is zero-phase; `causal=True` is available.
* Not computed: jerk, force, power and work. These need a validated model.

## 4. Events (`src/nflbdb/features/events.py`)

Every event type uses three threshold sets: `lenient`, `default` and `strict`. All three are **provisional**.

* **Braking.** a_tan ≤ −2.5 yd/s² for ≥ 0.3 s, losing ≥ 1 yd/s, from ≥ 2 yd/s. The event records the time to re-acceleration.
* **Acceleration.** a_tan ≥ 2.5 yd/s² for ≥ 0.3 s, gaining ≥ 1 yd/s.
* **Turn.** Constant-sign yaw rate with \|yaw\| ≥ 75°/s while speed ≥ 2 yd/s, for ≥ 0.3 s and ≥ 30° in total.
* **Reversal.** A low-speed window (< 1 yd/s, ≤ 1.5 s) whose heading in and heading out differ by ≥ 90°. This captures shuttle-type plant-and-cut moves, which yaw rate cannot see.

Coverage of threshold sets:
- **Combine:** all three threshold sets are persisted.
- **NFL:** the full build persists `default` only. `lenient` and `strict` are evaluated on a deterministic game sample to bound memory.

## 5. Quality tiers (`src/nflbdb/features/segment_quality.py`)

The tiers form a cascade; each trajectory gets the first tier whose rule it triggers.

1. `reject`: no smoothable frame, or any smoothed speed > 12.5 yd/s, or any smoothed acceleration > 15 yd/s².
2. `C`: coverage < 0.80, or residual RMS > 0.15 yd.
3. `B`: more than one segment, or residual RMS > 0.06 yd, or median \|speed − s\| > 0.25 yd/s.
4. `A`: none of the above.

The residual and vendor thresholds are **provisional**. Calibrate them with `segment_quality_summary.csv` and `tier_reason_counts.csv`.

## 6. Results (executed M1 outputs)

**Built datasets** (notebook 05 output): `combine_frame_kinematics` 431,094 frames · 6,310 attempts · 59,922 Combine events (all 3 rules). NFL: 314,944 player-plays · 315,343 segments · 619,409 events (default rule).
**Combine integrity:** 0 null coordinates, 0 duplicate timestamps, 0 gap / jump / drill-change breaks.

| Output | Result |
|---|---|
| Vendor speed (`kinematics_vendor_agreement.csv`) | Combine r = 0.9996, median \|Δ\| = 0.030 yd/s, p95 = 0.163. NFL sample (24 games, 544k frames): r = 0.998, median \|Δ\| = 0.058. Shuttle (0.996) and three-cone (0.995) are lowest; at reversals, speed passes through 0. |
| Vendor `a` | A **magnitude**: 0 % negative values. It correlates with `accel_mag` (r = 0.976 Combine, 0.887 NFL) more than with \|a_tan\| (0.633 / 0.627). It carries no braking sign. |
| Vendor `dir` convention | **Compass (0° = +y, clockwise)** in both domains. Median circular difference: 0.33° (Combine), 0.95° (NFL), vs ≈ 90° under the math convention; three-cone 1.34°. The vendor direction is therefore internally consistent with x, y. The physical handedness of the Combine frame is still unverified, so turn sense stays unused. |
| Smoothing sensitivity | Rank stability vs SG(7,2): path length and p95 speed have ρ ≥ 0.9997. Peak a_tan / decel / yaw / \|a\| have ρ = 0.949–0.993. Braking-event counts have ρ = 0.92–0.97. Levels shift with the window: median relative change in peak decel is 6–14 %. **Conclusion:** ranks are stable, but levels depend on smoothing, so compare only features built with the same SG setting. |
| Event thresholds (`event_threshold_sensitivity.csv`) | Combine, default rule, events per attempt: acceleration 1.71, braking 0.62, reversal 0.45, turn 0.34. Shuttle reversals: **1.93 per attempt (design: 2)**. Forty braking: 0.001 per attempt, because the recording ends at the finish before deceleration; 65 % of forty acceleration events touch the segment edge. The strict turn rule almost never fires (0.02 per attempt Combine, 0.004 NFL), so it does not bracket the default from above. NFL: 14.7 % of braking events touch the segment edge, because plays end mid-braking. |
| Quality tiers | Combine: 6,309 / 6,310 tier A (1 B: a 2025 shuttle). NFL tier A: 99.6 % (2023), 99.1 % (2024), 98.9 % (2025). Rejects: 0.09–0.11 %. Leading NFL reasons: vendor-speed disagreement (0.75 %), split segments (0.12 %), implausible acceleration (0.10 %). |
| **Cohort noise difference** | Median position-residual RMS is 0.011–0.014 yd for Combine 2023/24 vs 0.002–0.009 yd for 2025 (NFL ≈ 0.005). The 2025 Combine data are markedly smoother, so tracking processing probably differs by cohort. Noise-sensitive features (peaks) need a cohort check, done in M2 by `cohort_shift.csv`. |
| Examples | The forty example (nfl_id 58326) starts at ≈ 1 yd/s and ends at peak speed (10.6 yd/s), so the recording is the run window with the start already under way. The shuttle example shows ≈ 1 s of slow drift (< 1 yd/s) before the first leg. Both facts shape the M2 onset and censoring design. |

### Calibration decision (M1 thresholds)
`KinematicsConfig` stays at **version 1, unchanged**. Default rules produce plausible counts (e.g. 1.93 shuttle reversals vs 2 by design), and tiers are not limiting. The only identified defect is the non-informative strict turn set. Changing it would invalidate all M1 datasets, including the full NFL build, for a sensitivity-only rule, so it is documented here instead. Revisit it if M3 turn analyses need an upper bracket; then change it together with any M3-driven change and bump `FEATURES_VERSION`.

## 7. Datasets (`warehouse/gold/phase_03/`, catalog views `p03_*`)

| Dataset | Grain / key | Partition |
|---|---|---|
| combine_frame_kinematics | event_id, nfl_id, attempt, ts | draft_year |
| combine_segment_quality | + seg_idx | — |
| combine_trajectory_quality | event_id, nfl_id, attempt | — |
| combine_kinematic_events | + seg_idx, event_type, rule, event_idx | — |
| nfl_segment_quality | game_id, play_id, nfl_id, seg_idx | season |
| nfl_trajectory_quality | game_id, play_id, nfl_id | season |
| nfl_kinematic_events | + seg_idx, event_type, rule, event_idx | season |

A dataset is reused only when its version, its input fingerprint (silver files, every `KinematicsConfig` parameter and the build scope) and its on-disk schema all match. Stale builds are archived to `_archive/`. NFL frame-level kinematics are recomputed on demand per play with `KinematicFeatureBuilder.nfl_play_kinematics`.

## 8. M2 — attempt-level movement features (executed 2026-10-08; design only below)

All parameters live in `features/config.py::MovementConfig`, version `M2_VERSION = 2` (the persisted datasets carry version 2). This config is separate from `KinematicsConfig`, so an M2 change never invalidates the M1 datasets. Every feature is registered in `feature_registry.M2_REGISTRY` with its formula, unit, limitations and status.

| Module | Family | Content |
|---|---|---|
| `temporal.py` | E | Movement window: onset with ≥ 3 frames ≥ 0.5 yd/s, a "fast" onset at ≥ 2.0 yd/s that is robust to pre-start drift, and end of activity. Also censoring flags, phase labels (stationary / accelerating / steady / decelerating), transition counts, early/mid/late thirds, and peak-acceleration lead time. |
| `acceleration.py` | A | Speed mean/median/p90/p95/max/sd/CV, time above 4/6/8 yd/s, sustained high-speed time, time and distance to peak (censoring-flagged), and 10→90 % rise time. Acceleration: initial (first 1 s), max/p90/p95, 0.5 s rolling sustained value, positive integral, and vector \|a\| max/p95. |
| `braking.py` | B | Frame-level peak, p95 and mean deceleration. Event-level (M1 default rule): count, time, distance, intensity, speed-reduction ratio, speed lost per yard, and time to re-acceleration. |
| `agility.py` | C | Turn and reversal events of ≥ 60° are merged into a COD when less than 0.3 s apart. Per-COD measures: approach speed, approach deceleration, exit acceleration, speed retention, and COD time (time below 80 % of approach speed; censored if not recovered within 2 s). Per attempt: COD counts and medians, yaw integral, \|a_lat\| p95. |
| `efficiency.py` | D | Path length, net displacement, path efficiency, and excess path (all task-specific). Principal-axis lateral extent, lateral speed and lateral time share, plus an anisotropy validity flag. |
| `sprint.py` | A/E | Mono-exponential model v = vmax(1 − e^{−(t−t0)/τ}), fitted by grid least squares (deterministic). It gives a0 = vmax/τ, model-start splits at 10/20/40 yd (extrapolating pre-recording distance when onset is censored) and flying 10→20 / 20→40 intervals. No force or power is derived. |
| `reliability.py` | G | Unbalanced one-way ICC(1) with n0 correction and a vectorised athlete bootstrap CI. Also ICC(1,k̄), typical error, CV %, first-two-attempt Spearman, attempt-2 − attempt-1 drift, and best-vs-mean. The label is taken from the CI lower bound. |

Datasets (`warehouse/gold/phase_03/`):

| Dataset | Grain / key |
|---|---|
| `combine_attempt_features` | `event_id, nfl_id, attempt` (tiers A/B) |
| `combine_cod_events` | `event_id, nfl_id, attempt, cod_idx` |
| `combine_sprint_profiles` | `event_id, nfl_id, attempt` (forty) |
| `combine_athlete_drill_features` | `nfl_id, drill, feature`: median / mean / sd / min / max / best / first |

Design responses to the M1 evidence:
1. The forty recording starts in motion and ends at the finish. Hence the `onset_censored`, `peak_speed_censored` and `peak_at_end` flags, model-based pre-distance, and splits measured from the model start, never claimed to be official.
2. Shuttle drift → a second onset definition (`fast_onset_to_end_s`); the analytics test both against official times.
3. Reversal `entry_speed` from M1 ≈ 1 yd/s by construction → window-based `approach_speed`.
4. Cohort noise difference → `cohort_shift.csv` and per-cohort reliability.
5. Ranks are smoothing-robust but levels are not → all M2 features use the single primary SG(7,2) setting.
