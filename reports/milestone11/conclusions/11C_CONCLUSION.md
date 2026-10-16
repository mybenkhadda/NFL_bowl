## 11C — conclusion (inconclusive; run mode full; generated 2026-10-14T22:49:26Z)

### Research question
Are braking, turning and re-acceleration metrics repeatable, distinct from the stopwatch, and related to NFL transitions?

### Hypothesis
Transition metrics have ICC ≥ 0.5 across attempts and held-out association with geometry-matched NFL transitions.

### Sample
367 Combine athletes with transitions (1,633 transitions); 387 linked athletes with ≥ 1 supported NFL target.

### Main evidence
- repeatable (ICC lower bound ≥ 0.5): short_shuttle.approach_speed, short_shuttle.curved_speed, three_cone.brake_time_5_2_s, three_cone.tci_cw
- short_shuttle.decel_mean → nfl.brake_decel: ρ 0.608 [0.474, 0.709], n=101 — association (interval excludes 0) — descriptive
- short_shuttle.decel_peak → nfl.brake_peak05: ρ 0.623 [0.495, 0.733], n=101 — association (interval excludes 0) — descriptive
- short_shuttle.braking_distance_yd → nfl.brake_distance_matched: ρ 0.070 [-0.149, 0.278], n=100 — inconclusive (|r| below ≈ 0.28 is not detectable at this n)
- short_shuttle.brake_to_burst_s → nfl.brake_to_burst: ρ 0.137 [-0.074, 0.340], n=72 — inconclusive (|r| below ≈ 0.33 is not detectable at this n)
- short_shuttle.exit_accel → nfl.reaccel_mean: ρ 0.606 [0.409, 0.752], n=72 — association (interval excludes 0) — descriptive
- short_shuttle.transition_cost_index → nfl.pair_tci: ρ 0.246 [0.015, 0.459], n=72 — association (interval excludes 0) — descriptive
- three_cone.transition_cost_index → nfl.redirect_retention_90: ρ -0.379 [—, —], n=18 — too few athletes for an interval
- three_cone.exit_accel → nfl.reaccel_mean: ρ 0.142 [-0.091, 0.350], n=87 — inconclusive (|r| below ≈ 0.30 is not detectable at this n)
- backpedal_transition.exit_accel → nfl.reaccel_mean: ρ -0.438 [-0.592, -0.274], n=92 — association (interval excludes 0) — descriptive
- backpedal_transition.brake_to_burst_s → nfl.brake_to_burst: ρ 0.067 [-0.163, 0.278], n=92 — inconclusive (|r| below ≈ 0.29 is not detectable at this n)
- geom.tci_turn90 → nfl.pair_tci_turn90: ρ — [—, —], n=0 — too few athletes for an interval
- held-out short_shuttle.decel_mean → nfl.brake_decel: Δ RMSE 0.0000 [-0.0010, 0.0012] (inconclusive)
- held-out short_shuttle.decel_peak → nfl.brake_peak05: Δ RMSE -0.0007 [-0.0033, 0.0018] (inconclusive)
- held-out short_shuttle.braking_distance_yd → nfl.brake_distance_matched: Δ RMSE 0.0051 [-0.0033, 0.0185] (inconclusive)
- held-out short_shuttle.brake_to_burst_s → nfl.brake_to_burst: Δ RMSE 0.0002 [-0.0001, 0.0005] (inconclusive)
- held-out short_shuttle.exit_accel → nfl.reaccel_mean: Δ RMSE -0.0007 [-0.0035, 0.0020] (inconclusive)
- held-out short_shuttle.transition_cost_index → nfl.pair_tci: Δ RMSE 0.0000 [-0.0004, 0.0003] (inconclusive)
- held-out three_cone.transition_cost_index → nfl.redirect_retention_90: Δ RMSE 0.0009 [-0.0009, 0.0029] (inconclusive)
- held-out three_cone.curved_speed → nfl.redirect_retention: Δ RMSE 0.0002 [-0.0018, 0.0022] (inconclusive)
- segmenter agreement (median): backpedal_transition cp 0.91 / hmm 0.83, short_shuttle cp 0.91 / hmm 0.86, three_cone cp 0.90 / hmm 0.83

### Alternative explanations
- Drill technique (footwork coaching, plant foot) changes transition metrics without a capacity difference.
- NFL transitions follow routes and coverage; their geometry is chosen, not imposed.
- Non-forty participation is selected (see 11G).

### Practical scouting relevance
Braking and re-acceleration are what the forty cannot see; a repeatable, nonredundant transition metric adds a measurable dimension even if NFL transfer is unproven.

### Limitations
- 10 Hz sampling coarsens braking onset and latency (± 0.1 s).
- Combine reversals pass through a stop, NFL turns mostly do not: few shared geometry cells.
- Burst thresholds differ by domain (declared).

### Next experiment
- Collect tracked deceleration and cutting tests with controlled entry speed (10C proposals).
- Repeat the geometry-matched comparison when more NFL turns with valid headings are available.

