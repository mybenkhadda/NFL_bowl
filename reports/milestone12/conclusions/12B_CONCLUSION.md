## 12B — conclusion (inconclusive; run mode full; generated 2026-10-15T01:18:06Z)

### Research question
Q2 — Could unreliable Combine features be masking athlete-level relationships?

### Declared primary hypothesis
For preregistered features, corrected latent associations exceed observed ones in proportion to 1/√ICC, but held-out prediction does not improve.

### Primary outcome / comparison
corrected vs naive standardized coefficient of each preregistered feature on its matched NFL target — regression calibration (B1) vs naive (B0) held-out Δ RMSE; SIMEX / EM coefficients descriptive (evaluation target: `cross_domain_physical_transfer`)

### Sample
460 athletes with preregistered attempt values; 395 with a repeat attempt of ≥ 1 feature.

### Main evidence (from persisted tables)
- forty.band_2_4_mean_accel: ICC 0.457 [0.342, 0.561] (n=418, repeats 372)
- forty.band_4_6_mean_accel: ICC 0.716 [0.619, 0.816] (n=418, repeats 375)
- forty.band_6_8_mean_accel: ICC 0.851 [0.782, 0.918] (n=418, repeats 375)
- forty.time_4_8: ICC 0.943 [0.928, 0.957] (n=408, repeats 320)
- forty.v_max_obs: ICC 0.985 [0.982, 0.987] (n=418, repeats 375)
- short_shuttle.decel_mean: ICC 0.320 [0.068, 0.540] (n=169, repeats 40)
- three_cone.transition_cost_index: ICC 0.000 [0.000, 0.404] (n=230, repeats 73)
- forty.band_2_4_mean_accel → nfl.band_2_4: β_obs -0.018 [-0.119, 0.109], β_RC -0.026 [-0.160, 0.177], compatible latent r [-0.168, 0.155], detectable latent r ≥ 0.229
- forty.band_4_6_mean_accel → nfl.band_4_6: β_obs -0.037 [-0.157, 0.108], β_RC -0.058 [-0.269, 0.197], compatible latent r [-0.249, 0.171], detectable latent r ≥ 0.293
- forty.band_6_8_mean_accel → nfl.band_6_8: β_obs -0.079 [-0.247, 0.118], β_RC -0.096 [-0.290, 0.160], compatible latent r [-0.337, 0.161], detectable latent r ≥ 0.358
- forty.time_4_8 → nfl.band_4_6: β_obs 0.044 [-0.087, 0.178], β_RC 0.059 [-0.114, 0.245], compatible latent r [-0.118, 0.243], detectable latent r ≥ 0.255
- forty.v_max_obs → nfl.vmax_p90: β_obs 0.272 [0.110, 0.425], β_RC 0.306 [0.124, 0.480], compatible latent r [0.137, 0.529], detectable latent r ≥ 0.290
- short_shuttle.decel_mean → nfl.brake_decel: β_obs -0.006 [-0.143, 0.152], β_RC -0.188 [-4.514, 4.799], compatible latent r [-1.000, 1.000], detectable latent r ≥ 1.000
- three_cone.transition_cost_index → nfl.redirect_retention: β_obs 0.034 [-0.170, 0.290], β_RC 1.074 [-5.388, 9.182], compatible latent r [-1.000, 1.000], detectable latent r ≥ 1.000
- forty.band_2_4_mean_accel → nfl.q90_shared: β_obs — [—, —], β_RC — [—, —], compatible latent r [—, —], detectable latent r ≥ —
- forty.band_4_6_mean_accel → nfl.q90_shared: β_obs — [—, —], β_RC — [—, —], compatible latent r [—, —], detectable latent r ≥ —
- forty.band_6_8_mean_accel → nfl.q90_shared: β_obs — [—, —], β_RC — [—, —], compatible latent r [—, —], detectable latent r ≥ —
- forty.time_4_8 → nfl.q90_shared: β_obs — [—, —], β_RC — [—, —], compatible latent r [—, —], detectable latent r ≥ —
- forty.v_max_obs → nfl.q90_shared: β_obs — [—, —], β_RC — [—, —], compatible latent r [—, —], detectable latent r ≥ —
- held-out B1 vs B0 (forty.band_2_4_mean_accel → nfl.band_2_4): Δ RMSE 0.0001 [-0.0007, 0.0008] — inconclusive
- held-out B1 vs B0 (forty.band_4_6_mean_accel → nfl.band_4_6): Δ RMSE -0.0011 [-0.0025, 0.0003] — inconclusive
- held-out B1 vs B0 (forty.band_6_8_mean_accel → nfl.band_6_8): Δ RMSE 0.0011 [-0.0005, 0.0027] — inconclusive
- held-out B1 vs B0 (forty.time_4_8 → nfl.band_4_6): Δ RMSE -0.0001 [-0.0021, 0.0019] — inconclusive
- held-out B1 vs B0 (forty.v_max_obs → nfl.vmax_p90): Δ RMSE -0.0003 [-0.0006, 0.0000] — inconclusive
- held-out B1 vs B0 (short_shuttle.decel_mean → nfl.brake_decel): Δ RMSE 0.0000 [-0.0006, 0.0007] — inconclusive
- held-out B1 vs B0 (three_cone.transition_cost_index → nfl.redirect_retention): Δ RMSE -0.0000 [-0.0004, 0.0003] — inconclusive
- held-out B1 vs B0 (forty.band_2_4_mean_accel → nfl.q90_shared): Δ RMSE -0.0001 [-0.0006, 0.0004] — inconclusive
- held-out B1 vs B0 (forty.band_4_6_mean_accel → nfl.q90_shared): Δ RMSE -0.0004 [-0.0010, 0.0001] — inconclusive
- held-out B1 vs B0 (forty.band_6_8_mean_accel → nfl.q90_shared): Δ RMSE -0.0005 [-0.0014, 0.0001] — inconclusive
- held-out B1 vs B0 (forty.time_4_8 → nfl.q90_shared): Δ RMSE -0.0001 [-0.0005, 0.0003] — inconclusive
- held-out B1 vs B0 (forty.v_max_obs → nfl.q90_shared): Δ RMSE -0.0001 [-0.0002, 0.0001] — inconclusive

### Alternative explanations
- Errors may be differential (a nervous athlete is slow in every attempt and in games): RC / SIMEX then over-correct (see the synthetic correlated-error scenario).
- Repeat attempts are not missing at random (fatigue, injury, invitation): σ²_ε may be underestimated (see the MNAR scenario).

### Practical interpretation
Distinguishes “the Combine feature is too noisy to show a relation” from “there is no relation of a detectable size”.

### Limitations
- 1–3 attempts per athlete: within-athlete variance is imprecise.
- Corrections assume classical nondifferential error.
- One cohort.

### Next research questions
- If compatible latent ranges are narrow and near 0 → stop correction work; otherwise collect more repeats per athlete.
