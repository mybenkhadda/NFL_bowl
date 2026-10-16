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
- Phase 01 lakehouse executed and verified. Phase 01 non-PASS checks drilled down exactly: {{rows:A0_dq_failures}} checks (Study A0, `A0_dq_failures.csv`).
- Combine vendor-speed agreement, frame grain: MAE {{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|mae|.3f}} yd/s over {{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|n_frames|,}} frames from {{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|n_athletes|,}} athletes.
- NFL sample, frame grain: MAE {{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|mae|.3f}} yd/s over {{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|n_frames|,}} frames.

## 3. Study populations
| Population | Trajectories | Athletes | Source |
|---|---:|---:|---|
| Combine, all attempts (Study Q) | {{tbl:Q_flag_rates|domain=combine,facet=all,level=all,flag=any_flag|n_trajectories|,}} | {{tbl:Q_flag_rates|domain=combine,facet=all,level=all,flag=any_flag|n_athletes|,}} | `Q_flag_rates` |
| NFL deterministic sample (Study Q) | {{tbl:Q_flag_rates|domain=nfl_sample,facet=all,level=all,flag=any_flag|n_trajectories|,}} | {{tbl:Q_flag_rates|domain=nfl_sample,facet=all,level=all,flag=any_flag|n_athletes|,}} | `Q_flag_rates` |

## 4. Quality validation (Study 1.1 — A1, A3, Q)
**Question.** Do tiers or continuous quality metrics separate reliable from unreliable signals? Are structural and signal quality distinct?

**Statistical evidence.**
- Tier ≠ A as a predictor of high vendor disagreement, Combine: AUC {{tbl:A3_tier_discrimination|domain=combine,predictor=tier_not_A,target=vendor_rmse_high|auc|.3f}} (95 % CI {{tbl:A3_tier_discrimination|domain=combine,predictor=tier_not_A,target=vendor_rmse_high|ci_lower|.3f}}–{{tbl:A3_tier_discrimination|domain=combine,predictor=tier_not_A,target=vendor_rmse_high|ci_upper|.3f}}).
- Positional residual RMS as a predictor of high jerk, Combine: AUC {{tbl:A3_tier_discrimination|domain=combine,predictor=resid_rms_yds,target=jerk_high|auc|.3f}}.
- Structural vs signal flags, Cohen's κ: Combine {{tbl:Q_flag_agreement|domain=combine,stratum=all|kappa|.2f}} (CI {{tbl:Q_flag_agreement|domain=combine,stratum=all|ci_lower|.2f}}–{{tbl:Q_flag_agreement|domain=combine,stratum=all|ci_upper|.2f}}), NFL sample {{tbl:Q_flag_agreement|domain=nfl_sample,stratum=all|kappa|.2f}}.
- Raw peak speed on SG-boundary frames, observed / expected: forty {{tbl:A3_boundary_peak_share|domain=combine,stratum=FORTY_YARD_DASH,peak=peak_speed|ratio_observed_expected|.1f}}×, shuttle {{tbl:A3_boundary_peak_share|domain=combine,stratum=SHORT_SHUTTLE,peak=peak_speed|ratio_observed_expected|.1f}}×, three-cone {{tbl:A3_boundary_peak_share|domain=combine,stratum=THREE_CONE_DRILL,peak=peak_speed|ratio_observed_expected|.1f}}×, NFL {{tbl:A3_boundary_peak_share|domain=nfl_sample,stratum=nfl_play,peak=peak_speed|ratio_observed_expected|.1f}}×.

**Athlete eligibility** (Combine forty, athletes with ≥ 2 retained attempts relative to all trajectories; table grain = domain × stratum × drill × scenario):
- S1 structural pass: {{tbl:Q_athlete_eligibility|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,scenario=S1 structural pass|share_athletes_repeat|.1%}}
- S2 + signal pass: {{tbl:Q_athlete_eligibility|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,scenario=S2 + signal pass|share_athletes_repeat|.1%}}
- S3 + no boundary peak: {{tbl:Q_athlete_eligibility|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,scenario=S3 + no boundary peak|share_athletes_repeat|.1%}}

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
| peak_speed | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_speed,smoothing=sg5p2,level=athlete|spearman|.3f}} [{{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_speed,smoothing=sg5p2,level=athlete|spearman_lower|.3f}}] | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_speed,smoothing=sg5p2,level=athlete|top_overlap|.2f}} | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_speed,smoothing=sg5p2,level=trajectory|median_rel_diff|.1%}} | {{tbl:B_repeatability_by_setting|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_speed,smoothing=sg7p2|icc1|.3f}} [{{tbl:B_repeatability_by_setting|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_speed,smoothing=sg7p2|ci_lower|.3f}}] |
| p95_speed | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_speed,smoothing=sg5p2,level=athlete|spearman|.3f}} [{{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_speed,smoothing=sg5p2,level=athlete|spearman_lower|.3f}}] | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_speed,smoothing=sg5p2,level=athlete|top_overlap|.2f}} | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_speed,smoothing=sg5p2,level=trajectory|median_rel_diff|.1%}} | {{tbl:B_repeatability_by_setting|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_speed,smoothing=sg7p2|icc1|.3f}} [{{tbl:B_repeatability_by_setting|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_speed,smoothing=sg7p2|ci_lower|.3f}}] |
| peak_a_tan | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_a_tan,smoothing=sg5p2,level=athlete|spearman|.3f}} [{{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_a_tan,smoothing=sg5p2,level=athlete|spearman_lower|.3f}}] | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_a_tan,smoothing=sg5p2,level=athlete|top_overlap|.2f}} | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_a_tan,smoothing=sg5p2,level=trajectory|median_rel_diff|.1%}} | {{tbl:B_repeatability_by_setting|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_a_tan,smoothing=sg7p2|icc1|.3f}} [{{tbl:B_repeatability_by_setting|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=peak_a_tan,smoothing=sg7p2|ci_lower|.3f}}] |
| p95_a_tan | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_a_tan,smoothing=sg5p2,level=athlete|spearman|.3f}} [{{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_a_tan,smoothing=sg5p2,level=athlete|spearman_lower|.3f}}] | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_a_tan,smoothing=sg5p2,level=athlete|top_overlap|.2f}} | {{tbl:B_smoothing_agreement|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_a_tan,smoothing=sg5p2,level=trajectory|median_rel_diff|.1%}} | {{tbl:B_repeatability_by_setting|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_a_tan,smoothing=sg7p2|icc1|.3f}} [{{tbl:B_repeatability_by_setting|domain=combine,stratum=FORTY_YARD_DASH,drill=FORTY_YARD_DASH,feature=p95_a_tan,smoothing=sg7p2|ci_lower|.3f}}] |

**Shuttle deceleration.** Athlete-level ρ for sg11p3:
- `peak_decel`: {{tbl:B_smoothing_agreement|domain=combine,stratum=SHORT_SHUTTLE,drill=SHORT_SHUTTLE,feature=peak_decel,smoothing=sg11p3,level=athlete|spearman|.3f}}
- `p95_decel`: {{tbl:B_smoothing_agreement|domain=combine,stratum=SHORT_SHUTTLE,drill=SHORT_SHUTTLE,feature=p95_decel,smoothing=sg11p3,level=athlete|spearman|.3f}}

**Figures:** `figures/phase_04/B_smoothing/B_01…B_09` (B_01 and B_06 are publication-grade, PNG + SVG + PDF).
**Football interpretation:** _[to write after review]_
**Limitations:** this is stability, not validity; the setting range is the one M1 declared, not an optimum.
**Decision:** _[to write after review]_

## 6. Event detector validation (Study 1.3 — C1, C2)
**C1 — consistency.** Combine turn rate (all drills, events per valid second):

| Rule | Rate (events/s) | 95 % CI |
|---|---:|---|
| lenient | {{tbl:C1_event_rates|domain=combine,drill=ALL,event_type=turn,rule=lenient,smoothing=sg7p2|rate_per_s|.3f}} | |
| default | {{tbl:C1_event_rates|domain=combine,drill=ALL,event_type=turn,rule=default,smoothing=sg7p2|rate_per_s|.3f}} | {{tbl:C1_event_rates|domain=combine,drill=ALL,event_type=turn,rule=default,smoothing=sg7p2|rate_lower|.3f}}–{{tbl:C1_event_rates|domain=combine,drill=ALL,event_type=turn,rule=default,smoothing=sg7p2|rate_upper|.3f}} |
| strict | {{tbl:C1_event_rates|domain=combine,drill=ALL,event_type=turn,rule=strict,smoothing=sg7p2|rate_per_s|.3f}} | |

Shuttle reversal-rate athlete ordering, sg11p3 vs sg7p2: ρ = {{tbl:C1_count_stability|domain=combine,drill=SHORT_SHUTTLE,event_type=reversal,comparison=sg:sg11p3,level=athlete|spearman_rate|.3f}}.

**C2 — accuracy against human labels** (eval split, random tasks, ± 0.3 s, default rule, sg7p2):

| Event class | Precision | Recall | F1 [CI] | Onset MAE (s) |
|---|---:|---:|---|---:|
| braking | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|precision|.2f}} | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|recall|.2f}} | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|f1|.2f}} [{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|f1_lower|.2f}}] | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=braking,facet=all,level=all,label_set=all_labels|onset_mae_s|.2f}} |
| reversal | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|precision|.2f}} | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|recall|.2f}} | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|f1|.2f}} [{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|f1_lower|.2f}}] | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=reversal,facet=all,level=all,label_set=all_labels|onset_mae_s|.2f}} |
| turn | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|precision|.2f}} | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|recall|.2f}} | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|f1|.2f}} [{{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|f1_lower|.2f}}] | {{tbl:C2_detector_performance|split=eval,selection_set=random,smoothing=sg7p2,detector_rule=default,tolerance_s=0.3,event_class=turn,facet=all,level=all,label_set=all_labels|onset_mae_s|.2f}} |

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
| peak_speed | {{tbl:V_feature_status_summary|domain=combine,feature=peak_speed|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=peak_speed|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=peak_speed|limiting_criteria}} |
| peak_speed_interior | {{tbl:V_feature_status_summary|domain=combine,feature=peak_speed_interior|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=peak_speed_interior|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=peak_speed_interior|limiting_criteria}} |
| p95_speed | {{tbl:V_feature_status_summary|domain=combine,feature=p95_speed|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=p95_speed|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=p95_speed|limiting_criteria}} |
| path_smooth_yds | {{tbl:V_feature_status_summary|domain=combine,feature=path_smooth_yds|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=path_smooth_yds|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=path_smooth_yds|limiting_criteria}} |
| peak_a_tan | {{tbl:V_feature_status_summary|domain=combine,feature=peak_a_tan|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=peak_a_tan|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=peak_a_tan|limiting_criteria}} |
| p95_a_tan | {{tbl:V_feature_status_summary|domain=combine,feature=p95_a_tan|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=p95_a_tan|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=p95_a_tan|limiting_criteria}} |
| peak_decel | {{tbl:V_feature_status_summary|domain=combine,feature=peak_decel|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=peak_decel|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=peak_decel|limiting_criteria}} |
| p95_decel | {{tbl:V_feature_status_summary|domain=combine,feature=p95_decel|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=p95_decel|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=p95_decel|limiting_criteria}} |
| peak_abs_yaw_rate | {{tbl:V_feature_status_summary|domain=combine,feature=peak_abs_yaw_rate|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=peak_abs_yaw_rate|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=peak_abs_yaw_rate|limiting_criteria}} |
| rate_braking_per_s | {{tbl:V_feature_status_summary|domain=combine,feature=rate_braking_per_s|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=rate_braking_per_s|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=rate_braking_per_s|limiting_criteria}} |
| rate_reversal_per_s | {{tbl:V_feature_status_summary|domain=combine,feature=rate_reversal_per_s|status}} | {{tbl:V_feature_status_summary|domain=nfl_sample,feature=rate_reversal_per_s|status}} | {{tbl:V_feature_status_summary|domain=combine,feature=rate_reversal_per_s|limiting_criteria}} |

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
