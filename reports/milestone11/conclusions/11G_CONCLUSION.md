## 11G — conclusion (inconclusive; run mode full; generated 2026-10-14T23:27:46Z)

### Research question
Which subset of drills gives the most reliable incremental information for NFL movement evaluation?

### Hypothesis
At least one non-forty drill family adds held-out information beyond the official tests on a common cohort.

### Sample
nfl.band_4_6: 43 athletes with forty, three_cone, gauntlet; nfl.brake_decel: 43 athletes with forty, three_cone, gauntlet; nfl.reaccel_mean: 39 athletes with forty, three_cone, gauntlet; nfl.accel_low_mean: 45 athletes with forty, three_cone, gauntlet

### Main evidence
- nfl.band_4_6 + forty: Δ RMSE -0.0017 [-0.0067, 0.0025] (inconclusive)
- nfl.band_4_6 + three_cone: Δ RMSE -0.0007 [-0.0087, 0.0090] (inconclusive)
- nfl.band_4_6 + gauntlet: Δ RMSE 0.0040 [-0.0016, 0.0096] (inconclusive)
- nfl.band_4_6 + _random: Δ RMSE -0.0032 [-0.0111, 0.0064] (inconclusive)
- nfl.brake_decel + forty: Δ RMSE 0.0032 [-0.0050, 0.0141] (inconclusive)
- nfl.brake_decel + three_cone: Δ RMSE 0.0001 [-0.0013, 0.0018] (inconclusive)
- nfl.brake_decel + gauntlet: Δ RMSE 0.0004 [-0.0012, 0.0021] (inconclusive)
- nfl.brake_decel + _random: Δ RMSE 0.0003 [-0.0013, 0.0017] (inconclusive)
- nfl.reaccel_mean + forty: Δ RMSE -0.0033 [-0.0059, -0.0010] (model_worse)
- nfl.reaccel_mean + three_cone: Δ RMSE 0.0000 [-0.0041, 0.0053] (inconclusive)
- nfl.reaccel_mean + gauntlet: Δ RMSE -0.0000 [-0.0005, 0.0004] (inconclusive)
- nfl.reaccel_mean + _random: Δ RMSE -0.0020 [-0.0051, 0.0008] (inconclusive)
- nfl.accel_low_mean + forty: Δ RMSE -0.0005 [-0.0022, 0.0011] (inconclusive)
- nfl.accel_low_mean + three_cone: Δ RMSE -0.0030 [-0.0080, 0.0002] (inconclusive)
- nfl.accel_low_mean + gauntlet: Δ RMSE 0.0013 [-0.0017, 0.0056] (inconclusive)
- nfl.accel_low_mean + _random: Δ RMSE 0.0015 [-0.0014, 0.0058] (inconclusive)
- frontier: forty covers 4 cells at 2 hypothetical min
- frontier: forty + three_cone covers 10 cells at 5 hypothetical min
- frontier: forty + three_cone + gauntlet covers 14 cells at 8 hypothetical min
- frontier: forty + three_cone + gauntlet + backpedal_transition covers 17 cells at 12 hypothetical min

### Alternative explanations
- Common cohorts are small and select athletes who did every drill.
- A drill can be valuable for a demand that has no reliable NFL target yet (coverage without prediction).

### Practical scouting relevance
Separates drills that add held-out information, drills that only add physical coverage, and drills that duplicate the stopwatch — per objective, without a combined score.

### Limitations
- Hypothetical costs.
- Participation is nonrandom; position-specific orderings rest on few athletes.

### Next experiment
- Collect the most informative non-forty drills for more athletes before deciding.
- Re-run on a later class with drills frozen.

