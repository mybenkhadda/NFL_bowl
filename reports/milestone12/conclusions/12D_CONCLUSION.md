## 12D — conclusion (rejected; run mode full; generated 2026-10-15T02:03:00Z)

### Research question
Q4 — Do several drills jointly reveal reproducible physical traits that single drill metrics measure unreliably?

### Declared primary hypothesis
Latent factor scores have higher attempt-split test–retest reliability than the most reliable single tracked feature of their views.

### Primary outcome / comparison
test–retest correlation of factor scores (attempt-split) vs best single feature — D3 structured multi-view FA vs best single feature (evaluation target: `measurement_reliability`)

### Sample
510 athletes, 20 features in 6 views; 401 linked for transfer.

### Main evidence (from persisted tables)
- K by held-out likelihood: {'D2': 4, 'D3': 2, 'D4': 1}; selected D3 (K=2)
- only 0 athletes have every view (< 1000): D5 is not run — a declared rule, not a result
- D2 shared0: retest 0.642 vs forty.v_max_obs 0.985, Δ -0.344 [-0.423, -0.279] (n=375)
- D2 shared1: retest 0.434 vs forty.v_max_obs 0.985, Δ -0.551 [-0.651, -0.449] (n=375)
- D2 shared2: retest 0.704 vs gauntlet.sustained_speed 0.741, Δ -0.037 [-0.070, -0.003] (n=137)
- D2 shared3: retest 0.318 vs forty.band_6_8_mean_accel 0.850, Δ -0.532 [-0.608, -0.438] (n=375)
- D3 shared0: retest 0.813 vs forty.v_max_obs 0.985, Δ -0.172 [-0.225, -0.129] (n=375)
- D3 shared1: retest 0.608 vs forty.v_max_obs 0.985, Δ -0.378 [-0.466, -0.271] (n=375)
- D4 shared0: retest 0.620 vs forty.v_max_obs 0.985, Δ -0.365 [-0.457, -0.278] (n=375)
- D1 vs D0 → nfl.q90_shared: Δ RMSE -0.0034 [-0.0120, 0.0051] — inconclusive
- D2 vs D0 → nfl.q90_shared: Δ RMSE -0.0011 [-0.0029, 0.0007] — inconclusive
- D3 vs D0 → nfl.q90_shared: Δ RMSE -0.0004 [-0.0017, 0.0009] — inconclusive
- D4 vs D0 → nfl.q90_shared: Δ RMSE -0.0004 [-0.0011, 0.0004] — inconclusive
- D2 vs D1 → nfl.q90_shared: Δ RMSE 0.0023 [-0.0057, 0.0106] — inconclusive
- D3 vs D1 → nfl.q90_shared: Δ RMSE 0.0031 [-0.0051, 0.0112] — inconclusive
- D4 vs D1 → nfl.q90_shared: Δ RMSE 0.0031 [-0.0053, 0.0115] — inconclusive
- D1 vs D0 → nfl.band_4_6: Δ RMSE -0.0026 [-0.0093, 0.0049] — inconclusive
- D2 vs D0 → nfl.band_4_6: Δ RMSE 0.0013 [-0.0020, 0.0048] — inconclusive
- D3 vs D0 → nfl.band_4_6: Δ RMSE 0.0024 [-0.0016, 0.0061] — inconclusive
- D4 vs D0 → nfl.band_4_6: Δ RMSE -0.0001 [-0.0005, 0.0004] — inconclusive
- D2 vs D1 → nfl.band_4_6: Δ RMSE 0.0039 [-0.0023, 0.0099] — inconclusive
- D3 vs D1 → nfl.band_4_6: Δ RMSE 0.0050 [-0.0023, 0.0117] — inconclusive
- D4 vs D1 → nfl.band_4_6: Δ RMSE 0.0025 [-0.0050, 0.0092] — inconclusive
- D1 vs D0 → env.lin_A0: Δ RMSE -0.0267 [-0.0428, -0.0092] — model_worse
- D2 vs D0 → env.lin_A0: Δ RMSE -0.0043 [-0.0082, -0.0008] — model_worse
- D3 vs D0 → env.lin_A0: Δ RMSE -0.0009 [-0.0041, 0.0023] — inconclusive
- D4 vs D0 → env.lin_A0: Δ RMSE -0.0016 [-0.0034, 0.0001] — inconclusive
- D2 vs D1 → env.lin_A0: Δ RMSE 0.0224 [0.0049, 0.0379] — model_better
- D3 vs D1 → env.lin_A0: Δ RMSE 0.0258 [0.0084, 0.0415] — model_better
- D4 vs D1 → env.lin_A0: Δ RMSE 0.0251 [0.0076, 0.0411] — model_better
- D1 vs D0 → nfl.brake_decel: Δ RMSE -0.0055 [-0.0110, -0.0005] — model_worse
- D2 vs D0 → nfl.brake_decel: Δ RMSE -0.0017 [-0.0035, 0.0001] — inconclusive
- D3 vs D0 → nfl.brake_decel: Δ RMSE -0.0014 [-0.0033, 0.0004] — inconclusive
- D4 vs D0 → nfl.brake_decel: Δ RMSE -0.0000 [-0.0004, 0.0003] — inconclusive
- D2 vs D1 → nfl.brake_decel: Δ RMSE 0.0038 [-0.0013, 0.0092] — inconclusive
- D3 vs D1 → nfl.brake_decel: Δ RMSE 0.0040 [-0.0007, 0.0092] — inconclusive
- D4 vs D1 → nfl.brake_decel: Δ RMSE 0.0054 [0.0004, 0.0109] — model_better

### Alternative explanations
- A higher factor reliability can come from averaging several noisy features (Spearman–Brown), not from a new trait.
- Shared factors may encode position / body size rather than movement mechanics (compare D3 and D4).

### Practical interpretation
Tells whether several imperfect drills can be combined into one more stable, interpretable physical measurement.

### Limitations
- Views are missing not at random.
- Few repeats per drill.
- Loadings rotate within the shared block; interpretation only with stability support.

### Next research questions
- If a factor is more reliable but not predictive, use it descriptively (scouting profile), not as an NFL forecast.
