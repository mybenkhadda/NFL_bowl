_Generated from the `robust_*` tables and `transfer_candidates_v1` of this run (`RUN_MODE = 'full'`). Read-out: confirm-role held-out athletes, within position, 95 % athlete intervals. Labels are heuristic._

**RQ5 — does the TCN-B result survive training seeds and matched impostors?**
- repeat-attempt AUC: mean 0.872 over 3 seed(s), SD between seeds 0.001, range [0.872, 0.873]; athlete standard error 0.009.
- matched-stopwatch AUC: mean 0.802 over 3 seed(s), SD between seeds 0.007, range [0.796, 0.810]; athlete standard error 0.022.
- trace reliability: mean 0.935 over 3 seed(s), SD between seeds 0.009, range [0.925, 0.942]; athlete standard error 0.007.
- effective rank / dimension (smallest fold): mean 0.257 over 3 seed(s), SD between seeds 0.002, range [0.255, 0.259]; athlete standard error n/a.
- cross-fitted residual AUC: mean 0.924 over 3 seed(s), SD between seeds 0.006, range [0.917, 0.930]; athlete standard error 0.008.
- attempt change explained by the window: mean 0.112 over 3 seed(s), SD between seeds 0.009, range [0.102, 0.118]; athlete standard error 0.031.
- cohort balanced-accuracy gain: mean 0.063 over 3 seed(s), SD between seeds 0.025, range [0.034, 0.079]; athlete standard error 0.022.
- Matched discrimination, `TCNB_A` − FPCA benchmark: +0.067 [0.024, 0.111] on 234 confirm athletes with a matched impostor.
- Matched discrimination, `TCNB_A` − TCN-A: +0.037 [-0.001, 0.077] on 234 confirm athletes with a matched impostor.
- Matched discrimination, `TCNB_A` − untrained encoder: +0.056 [0.016, 0.095] on 234 confirm athletes with a matched impostor.
- Matched discrimination, `TCNB_A` − tracked top speed: +0.027 [-0.006, 0.059] on 234 confirm athletes with a matched impostor.
- Collapse flag (effective rank share below 0.25): 0 of 3 seed(s).

**Why did the residual AUC of notebook 03 exceed the raw AUC?**
- `TCNB_A` (role:confirm, n = 265): raw AUC 0.873; cross-fitted residual 0.917 (Δ 0.043 [0.035, 0.052]); cross-fitted with PERMUTED official results 0.926 (Δ 0.052); fitted on training athletes 0.903 (Δ 0.030 [0.021, 0.039]); in sample 0.901.
  - Own-repeat distance relative to raw: cross-fitted 1.000, training fit 1.000 (1 = unchanged, as expected when one vector is removed from both attempts); cross-fitted gallery under the same fitted function 0.902 against another function 0.919; residual variance share 0.422; effective rank raw 2.423 → 1.842.
  - Reading (rule-based, for review): the increase also appears when the official results are permuted, so it is produced by the cross-fitting procedure (different fitted functions for different athletes), not by removing stopwatch information.

**Effective dimension.** k = 4: paired AUC difference from the 8-dimensional embedding has a lower bound above −0.02 for all 3 seed(s).

**RQ6 — frozen shortlist** (frozen now)
- `C1_traditional_baseline` · traditional Combine baseline: **selected** (reference) — F1.official.cmb_forty;F1.official.cmb_ten_yd_split;F1.official.cmb_combine_weight;F1.official.cmb_combine_height. comparator of the transfer study; carried by definition, not by a screening rule
- `C2_speed_control` · reliable speed control: **selected** (ready_for_transfer_test) — F2.forty.p95_speed. tracked speed feature with the highest reliability lower bound among those graded Strong or Provisional; grade Strong (high reliability and no measured acquisition dependence)
- `C3_acceleration` · physically interpretable acceleration: **selected** (carry_with_caveat) — F6.forty.time_v4_to_v8_s. interpretable acceleration feature (sustained or direct) with the highest reliability lower bound, graded Strong or Provisional; grade Provisional (reliability high, acquisition uncertain)
- `C4_sprint_parameter` · sprint-curve parameter: **not selected** (not_ready) — no feature. no feature met the rule (fitted parameter of the existing exponential sprint model, not extrapolated and not redundant, with the highest reliability lower bound); best exploratory alternative: F4.forty.model_vmax (Provisional)
- `C5_fpca` · functional PCA benchmark: **selected** (carry_with_caveat) — FPCA_A_K2. low-complexity learned benchmark every encoder is compared with; carried by design
- `C6_tcn_b_8d` · TCN-B 8-d embedding: **selected** (ready_for_transfer_test) — TCNB_A. primary representation of notebook 03, frozen at its PRE-DECLARED first training seed; the other seeds are robustness replicates on which the transfer result must also be reported
- `C7_tcn_compact` · compact TCN projection: **selected** (ready_for_transfer_test) — TCNB_A|pc4. k = 4: paired AUC difference from the 8-dimensional embedding has a lower bound above −0.02 for all 3 seed(s)

**What is and is not established.** A selected family exists, is measurable on the stated athletes, and has the repeatability, stopwatch relation and nuisance dependence written in its row. `ready_for_transfer_test` means the Combine-side evidence justifies TESTING it against NFL movement; it does not mean it predicts anything. **Demonstrated NFL predictive value: NOT TESTED in this milestone** — no NFL outcome was read, and a same-athlete retrieval metric does not measure football ability. The eight coordinates of the encoder are one vector, not eight validated physiological measurements.