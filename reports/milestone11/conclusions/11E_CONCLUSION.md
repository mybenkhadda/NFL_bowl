## 11E — conclusion (inconclusive; run mode full; generated 2026-10-14T23:21:59Z)

### Research question
Within stopwatch twins, do Combine tracking differences point the same way as NFL movement differences?

### Hypothesis
Sign concordance of ΔCombine and ΔNFL > 0.5 on reliable features.

### Sample
3,266 09C pairs in the pool; eligible pairs per feature × target in the eligibility table (max 1937).

### Main evidence
- forty.band_4_6_mean_accel → nfl.band_4_6 [matched]: concordance 0.524 [0.455, 0.594], disjoint 0.413 [0.325, 0.507], 1341 pairs — inconclusive
- forty.band_6_8_mean_accel → nfl.band_6_8 [matched]: concordance 0.505 [0.413, 0.594], disjoint 0.472 [0.344, 0.603], 767 pairs — inconclusive
- forty.band_2_4_mean_accel → nfl.band_2_4 [matched]: concordance 0.557 [0.495, 0.615], disjoint 0.448 [0.368, 0.529], 1814 pairs — inconclusive
- forty.phase_A0 → env.lin_A0 [matched]: concordance 0.517 [0.462, 0.577], disjoint 0.440 [0.363, 0.520], 1937 pairs — inconclusive
- forty.phase_V0 → env.lin_V0 [matched]: concordance 0.515 [0.457, 0.576], disjoint 0.460 [0.382, 0.540], 1937 pairs — inconclusive
- forty.v_max_obs → nfl.vmax_p90 [matched]: concordance 0.579 [0.492, 0.665], disjoint 0.544 [0.427, 0.657], 900 pairs — inconclusive
- forty.early_accel → nfl.accel_low_mean [matched]: concordance 0.533 [0.479, 0.594], disjoint 0.444 [0.366, 0.526], 1800 pairs — inconclusive
- short_shuttle.decel_mean → nfl.brake_decel [matched]: concordance 0.488 [0.348, 0.642], disjoint 0.405 [0.270, 0.555], 172 pairs — inconclusive
- forty.band_2_4_mean_accel → nfl.band_6_8 [control]: concordance 0.512 [0.436, 0.595], disjoint 0.396 [0.276, 0.531], 767 pairs — inconclusive
- forty.band_6_8_mean_accel → nfl.brake_decel [control]: concordance 0.481 [0.427, 0.546], disjoint 0.485 [0.401, 0.569], 1558 pairs — inconclusive
- controls nfl.band_4_6: p(random pairs) 0.990, p(reshuffle) 0.229
- controls nfl.band_6_8: p(random pairs) 1.000, p(reshuffle) 0.483
- controls nfl.band_2_4: p(random pairs) 0.806, p(reshuffle) 0.015
- controls env.lin_A0: p(random pairs) 0.602, p(reshuffle) 0.224
- controls env.lin_V0: p(random pairs) 0.995, p(reshuffle) 0.284
- controls nfl.vmax_p90: p(random pairs) 1.000, p(reshuffle) 0.020
- controls nfl.accel_low_mean: p(random pairs) 0.930, p(reshuffle) 0.095
- controls nfl.brake_decel: p(random pairs) 0.577, p(reshuffle) 0.607

### Alternative explanations
- Twins still differ in size and role; Δweight is not in the concordance statistic (it is in the conditional slope only through Δforty / Δten).
- Selection into the NFL tracking cohort can differ within a pair (exposure).

### Practical scouting relevance
Among prospects the stopwatch cannot separate, a concordant tracking difference would be the most direct scouting use of tracking — or its absence a caution.

### Limitations
- Few eligible pairs per non-forty feature.
- Pairs share athletes: the disjoint matching has far fewer pairs but independent ones.
- Association between differences, not causation.

### Next experiment
- Re-run on the next draft class with the frozen features.
- Add reliability-weighted concordance as the primary if attempt SEs are available for most athletes.

