### What the tables show (primary target `O2.accel_sust05_mean`)

- **official_combine_tests** (M1 vs M0, 284 athletes): Δ RMSE +0.000 [-0.002, +0.003]; cohort left out +0.006; positions with a positive estimate 4/5; permutation p +0.219.
- **tracked_top_speed** (M3 vs M1, 284 athletes): Δ RMSE -0.000 [-0.001, +0.000]; cohort left out +0.000; positions with a positive estimate 1/5; permutation p +0.965.
- **sustained_acceleration** (M2 vs M1, 284 athletes): Δ RMSE +0.000 [-0.000, +0.000]; cohort left out -0.000; positions with a positive estimate 3/5; permutation p +0.104.
- **fpca_speed_curve** (M4 vs M1, 284 athletes): Δ RMSE -0.000 [-0.001, +0.000]; cohort left out -0.002; positions with a positive estimate 1/5; permutation p +0.846.
- **tcn_b_movement_fingerprint** (M5 vs M1, 284 athletes): Δ RMSE +0.000 [-0.001, +0.002]; cohort left out +0.000; positions with a positive estimate 3/5; permutation p +0.169.
- **tcn_b_compact** (M7_PC4 vs M1, 284 athletes): Δ RMSE +0.000 [-0.001, +0.001]; cohort left out n/a; positions with a positive estimate 4/5; permutation p n/a.

Matched prospects (833 pairs, 275 athletes, an athlete in up to 14 pairs):
- M0: concordance 0.521 [0.449, 0.594]; vs official +0.020 [-0.025, +0.067]
- M1: concordance 0.501 [0.430, 0.573]
- M2: concordance 0.505 [0.436, 0.577]; vs official +0.005 [-0.014, +0.025]
- M3: concordance 0.510 [0.443, 0.583]; vs official +0.010 [-0.010, +0.031]
- M4: concordance 0.510 [0.442, 0.580]; vs official +0.010 [-0.013, +0.035]
- M5: concordance 0.496 [0.425, 0.568]; vs official -0.005 [-0.035, +0.028]
- M5S: concordance 0.507 [0.437, 0.577]; vs official +0.006 [-0.021, +0.036]
- M6: concordance 0.499 [0.433, 0.569]; vs official -0.001 [-0.031, +0.031]
- M7_PC2: concordance 0.505 [0.435, 0.576]; vs official +0.005 [-0.012, +0.023]
- M7_PC4: concordance 0.505 [0.436, 0.574]; vs official +0.005 [-0.038, +0.050]
- MEAN: concordance 0.379 [0.310, 0.447]; vs official -0.121 [-0.220, -0.019]
- POS: concordance 0.313 [0.249, 0.380]; vs official -0.187 [-0.269, -0.105]

Scoring readiness: `official_combine_tests` → **exploratory**; `tracked_top_speed` → **descriptive_ready**; `sustained_acceleration` → **exploratory**; `fpca_speed_curve` → **descriptive_ready**; `tcn_b_movement_fingerprint` → **exploratory**; `tcn_b_compact` → **exploratory**.

Decision-tree case **E** — no method is shown to improve on official Combine testing for the frozen primary NFL target: report the negative finding, keep the Combine profiles as descriptions, build no Athletic Transfer Score.

Limits: the labels are observed NFL movement, shaped by role, scheme and usage as well as ability; draft cohorts differ in sensor generation and in NFL follow-up; intervals cover athlete sampling, not the refitting of the models; subgroup and matched-pair results are exploratory unless the pooled comparison is established.

### Conditions for the scoring milestone

A dimension may enter a PREDICTIVE score only when, on the frozen primary NFL target and the paired cohort:
1. its model lowers held-out RMSE against its reference with a 95 % athlete-bootstrap interval above 0;
2. the relative gain is at least 2 % of the reference error;
3. the gain stays positive with each draft cohort left out, and in at least three position groups;
4. the outcome-permutation check gives p < 0.05;
5. for the encoder: it beats the random encoder and the gain is positive for every training seed.
A dimension with a reliable Combine measurement and no established gain may enter a DESCRIPTIVE profile only, labelled as such.

- Meeting 1–5 in this run (`predictive_validated`): none
- Meeting 1 only (`predictive_candidate`, needs replication or the missing checks): none
- Decision-tree case: **E**