# NFL Big Data Bowl 2027 — Phase 04 Scientific Research Report

> **Generated from saved tables only.** `python -m nflbdb.research.phase04 --report` fills every `{{…}}` token from the Phase 04 manifest, falling back to the Phase 03 manifest. A token that cannot be resolved stays visible as `⟨pending: …⟩` and is listed at the end. The template contains no hand-typed result. Interpretation paragraphs marked _[to write after review]_ are written by the analyst once the numbers exist.

Each analysis follows: **Question → Population → Method → Descriptive results → Statistical evidence → Figures → Football interpretation → Limitations → Decision.**

---

## 1. Research objectives
- **A — Measurement validity:** which movement features are physically meaningful, numerically stable and repeatable?
- **B — Athletic characteristics:** which reliable features distinguish athletes, positions and drill performances?
- **C — NFL transfer:** which Combine movement characteristics relate to NFL movement beyond traditional Combine results?

Phase 04 Milestone 1 (this report's first complete part) answers A. B and C are gated on it (`docs/PHASE_04_PLAN.md` §5).

## 2. Existing data foundation
- Phase 01 lakehouse executed and verified. Phase 01 non-PASS checks drilled down exactly: 7 checks (Study A0, `A0_dq_failures.csv`).
- Combine vendor-speed agreement, frame grain: MAE 0.050 yd/s over 431,094 frames from 510 athletes.
- NFL sample, frame grain: MAE 0.088 yd/s over 543,964 frames.

## 3. Study populations
| Population | Trajectories | Athletes | Source |
|---|---:|---:|---|
| Combine, all attempts (Study Q) | 6,310 | 510 | `Q_flag_rates` |
| NFL deterministic sample (Study Q) | 7,466 | 173 | `Q_flag_rates` |

## 4. Quality validation (Study 1.1 — A1, A3, Q)
**Question.** Do tiers or continuous quality metrics separate reliable from unreliable signals? Are structural and signal quality distinct?

**Statistical evidence.**
- Tier ≠ A as a predictor of high vendor disagreement, Combine: AUC 0.502 (95 % CI 0.500–0.505).
- Positional residual RMS as a predictor of high jerk, Combine: AUC 0.725.
- Structural vs signal flags, Cohen's κ: Combine 0.17 (CI 0.13–0.21), NFL sample 0.20.
- Raw peak speed on SG-boundary frames, observed / expected: forty 2.7×, shuttle 6.7×, three-cone 11.4×, NFL 0.4×.

**Athlete eligibility** (Combine forty, athletes with ≥ 2 retained attempts relative to all trajectories; table grain = domain × stratum × drill × scenario):
- S1 structural pass: 98.4%
- S2 + signal pass: 88.8%
- S3 + no boundary peak: 67.7%

**Figures:**
- `figures/phase_03/A3_signal/A3_01…03`
- `figures/phase_04/Q_quality_eligibility/Q_01…Q_04`

**Football interpretation:** _[to write after review]_
**Limitations:** quantile flags are ≈ 5 % per stratum by construction; vendor speed is not ground truth.
**Decision:** _[to write after review]_

## 5. Feature methodology — smoothing robustness (Study 1.2 — B)
**Question.** Which summaries keep values and athlete ranks across SG settings sg5p2 / sg9p2 / sg11p3 versus the primary sg7p2?

| Feature (forty) | Athlete ρ, sg5p2 [CI lower] | Athlete top-decile overlap, sg5p2 | Trajectory median rel. change, sg5p2 | ICC sg7p2 [CI] |
|---|---|---|---|---|
| peak_speed | 0.999 [0.998] | 0.93 | 0.1% | 0.985 [0.982] |
| p95_speed | 1.000 [1.000] | 0.95 | 0.1% | 0.987 [0.984] |
| peak_a_tan | 0.923 [0.902] | 0.74 | 6.5% | 0.507 [0.397] |
| p95_a_tan | 0.968 [0.956] | 0.86 | 2.3% | 0.502 [0.356] |

**Shuttle deceleration.** Athlete-level ρ for sg11p3:
- `peak_decel`: 0.957
- `p95_decel`: 0.969

**Figures:** `figures/phase_04/B_smoothing/B_01…B_09` (B_01 and B_06 are publication-grade, PNG + SVG + PDF).
**Football interpretation:** _[to write after review]_
**Limitations:** this is stability, not validity; the setting range is the one M1 declared, not an optimum.
**Decision:** _[to write after review]_

## 6. Event detector validation (Study 1.3 — C1, C2)
**C1 — consistency.** Combine turn rate (all drills, events per valid second):

| Rule | Rate (events/s) | 95 % CI |
|---|---:|---|
| lenient | 0.157 | |
| default | 0.050 | 0.048–0.053 |
| strict | 0.003 | |

Shuttle reversal-rate athlete ordering, sg11p3 vs sg7p2: ρ = 0.974.

**C2 — accuracy against human labels** (eval split, random tasks, ± 0.3 s, default rule, sg7p2):

| Event class | Precision | Recall | F1 [CI] | Onset MAE (s) |
|---|---:|---:|---|---:|
| braking | ⟨pending: C2_detector_performance⟩ | ⟨pending: C2_detector_performance⟩ | ⟨pending: C2_detector_performance⟩ [⟨pending: C2_detector_performance⟩] | ⟨pending: C2_detector_performance⟩ |
| reversal | ⟨pending: C2_detector_performance⟩ | ⟨pending: C2_detector_performance⟩ | ⟨pending: C2_detector_performance⟩ [⟨pending: C2_detector_performance⟩] | ⟨pending: C2_detector_performance⟩ |
| turn | ⟨pending: C2_detector_performance⟩ | ⟨pending: C2_detector_performance⟩ | ⟨pending: C2_detector_performance⟩ [⟨pending: C2_detector_performance⟩] | ⟨pending: C2_detector_performance⟩ |

**Figures:**
- `figures/phase_04/C1_event_sensitivity/C1_01…05`
- `figures/phase_04/C2_annotation/C2_01…05`

**Decision:** event features remain ≤ PROVISIONAL until this table is populated from eval-split labels.

## 7. Athlete reliability
Milestone 2 (P1): NOT STARTED. Combine ICC per SG setting is in §5 and `B_repeatability_by_setting`.

## 8. Combine performance
Milestone 3: NOT STARTED.

## 9. NFL movement profiles
Milestone 4: NOT STARTED. Snap-anchored windows already exist (`nfl_player_play_windows`, Phase 03).

## 10. Combine-to-NFL relationships
Milestone 5: NOT STARTED. Gated on §11 below and on Milestone 2.

## 11. Statistical analyses — Milestone 1 decision (Study V)
| Feature | Combine status | NFL status | Limiting criteria (Combine) |
|---|---|---|---|
| peak_speed | REJECTED | PROVISIONAL | boundary, icc_setting_range, repeatability, vendor |
| peak_speed_interior | EXPERIMENTAL | PROVISIONAL | repeatability, smooth_overlap, vendor |
| p95_speed | EXPERIMENTAL | PROVISIONAL | icc_setting_range, repeatability, vendor |
| path_smooth_yds | EXPERIMENTAL | PROVISIONAL | repeatability |
| peak_a_tan | REJECTED | PROVISIONAL | icc_setting_range, repeatability, smooth_magnitude, smooth_overlap, smooth_rank |
| p95_a_tan | REJECTED | PROVISIONAL | icc_setting_range, repeatability, smooth_magnitude, smooth_overlap, smooth_rank |
| peak_decel | REJECTED | PROVISIONAL | icc_setting_range, repeatability, smooth_magnitude, smooth_overlap, smooth_rank |
| p95_decel | REJECTED | PROVISIONAL | icc_setting_range, repeatability, smooth_magnitude, smooth_overlap, smooth_rank |
| peak_abs_yaw_rate | REJECTED | PROVISIONAL | boundary, icc_setting_range, repeatability, smooth_magnitude, smooth_overlap, smooth_rank |
| rate_braking_per_s | REJECTED | PROVISIONAL | detector_f1, event_smoothing, icc_setting_range, repeatability, smooth_overlap, smooth_rank |
| rate_reversal_per_s | REJECTED | PROVISIONAL | detector_f1, event_smoothing, icc_setting_range, repeatability, smooth_rank |

The full registry is in `reports/tables/phase_04/reliability/V_feature_validation_registry.csv`, and the figures are `figures/phase_04/V_feature_validation/V_01`, `V_02`. Statuses are rule-based **proposals** (cut-offs in `P04_thresholds.csv`) pending scientific review.

## 12. Robustness checks
- Cohort and position robustness of smoothing: `B_robustness_facets` (figure B_07).
- Tolerance sensitivity of detector accuracy: figure C2_04.
- _[to write after review]_

## 13. Football interpretation
_[to write after review — only for features that are VALIDATED or PROVISIONAL]_

## 14. Study limitations
- The NFL sample is 8 games per season (deterministic). Full-NFL signal metrics were not recomputed.
- Vendor speed is a filtered reference, not ground truth.
- Combine coordinate handedness is unverified.
- NFL within-athlete reliability is not yet estimated, so NFL features stay ≤ PROVISIONAL by rule.
- Every threshold in `ValidationConfig` is proposed, not calibrated.

## 15. Candidate discoveries and next research phase
No discovery is claimed in Milestone 1. Next steps:
- Milestone 2: the athlete feature store (levels B–F), restricted to features that passed §11.
- NFL reliability as a function of the number of plays.
- After that, Combine and NFL analytics, then the transfer studies.


---
## Unresolved placeholders
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|precision|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|recall|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|f1|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|f1_lower|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|onset_mae_s|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|precision|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|recall|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|f1|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|f1_lower|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|onset_mae_s|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|precision|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|recall|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|f1|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|f1_lower|.2f}}`
- `{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|onset_mae_s|.2f}}`

### Source dependencies
| Table | Produced by | State | Command | Tokens |
|---|---|---|---|---:|
| `C2_detector_performance` | Phase 04 Study C2 (human annotation) | annotation labels not evaluated yet | `python -m nflbdb.research.phase04 --annotation-tasks, label the tasks, then --evaluate-annotations reports/annotation/phase_04/annotation_labels.csv` | 15 |
