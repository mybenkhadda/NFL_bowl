## 11D — conclusion (supported_finding; run mode full; generated 2026-10-14T23:19:11Z)

### Research question
Do NFL movement characteristics change measurably across seasons, and are Combine features related to level or change?

### Hypothesis
Between-athlete variance of season-to-season change exceeds measurement noise.

### Sample
seasons [2023, 2024, 2025]; 401 athletes in the panel; multi-season traits: env.lin_A0, env.lin_V0, nfl.accel_low_mean, nfl.band_4_6, nfl.band_6_8, nfl.brake_decel, nfl.burst_per_100_plays, nfl.high_speed_per_100_plays, nfl.movement_diversity, nfl.reaccel_mean, nfl.vmax_p90.

### Main evidence
- env.lin_A0: slope SD 0.060 [0.034, 0.060], fixed slope -0.024 per year, n=207
- env.lin_V0: slope SD 0.032 [0.032, 0.032], fixed slope 3.960 per year, n=206
- nfl.accel_low_mean: slope SD 0.035 [0.025, 0.052], fixed slope 0.013 per year, n=207
- nfl.band_4_6: slope SD 0.025 [0.023, 0.040], fixed slope -0.012 per year, n=136
- nfl.band_6_8: slope SD 0.022 [0.020, 0.041], fixed slope -0.026 per year, n=65
- nfl.brake_decel: slope SD 0.039 [0.028, 0.049], fixed slope -0.007 per year, n=179
- nfl.burst_per_100_plays: slope SD 0.032 [0.032, 0.032], fixed slope 0.415 per year, n=180
- nfl.high_speed_per_100_plays: slope SD 0.033 [0.032, 0.033], fixed slope -0.283 per year, n=180
- nfl.movement_diversity: slope SD 0.012 [0.014, 0.025], fixed slope -0.002 per year, n=187
- nfl.reaccel_mean: slope SD 0.037 [0.028, 0.047], fixed slope -0.009 per year, n=120
- nfl.vmax_p90: slope SD 0.059 [0.030, 0.069], fixed slope 0.038 per year, n=90
- first → later season env.lin_A0: ρ 0.946 [0.923, 0.960] (n=207)
- first → later season env.lin_V0: ρ 0.674 [0.579, 0.763] (n=206)
- first → later season nfl.accel_low_mean: ρ 0.946 [0.927, 0.959] (n=207)
- first → later season nfl.band_4_6: ρ 0.931 [0.899, 0.952] (n=136)
- first → later season nfl.band_6_8: ρ 0.618 [0.448, 0.768] (n=65)
- first → later season nfl.brake_decel: ρ 0.781 [0.702, 0.844] (n=179)
- first → later season nfl.burst_per_100_plays: ρ 0.858 [0.805, 0.904] (n=180)
- first → later season nfl.high_speed_per_100_plays: ρ 0.942 [0.920, 0.958] (n=180)
- first → later season nfl.movement_diversity: ρ 0.938 [0.909, 0.958] (n=187)
- first → later season nfl.reaccel_mean: ρ 0.634 [0.494, 0.740] (n=120)
- first → later season nfl.vmax_p90: ρ 0.674 [0.525, 0.791] (n=90)

### Alternative explanations
- Role, scheme and snap share change across seasons; an envelope can move with opportunity, not ability.
- Athletes who stay in the league are selected (see the selection table).
- Tracking or filtering changes between seasons shift all athletes alike (removed by the position × season adjustment).

### Practical scouting relevance
Tells whether an NFL movement profile measured in year 1 is stable enough to describe a player, and whether athletes differ in how it develops.

### Limitations
- Short panels (few seasons per athlete).
- Random slopes on 2–3 points are weakly identified; intervals are wide.
- No age variable.

### Next experiment
- Extend the panel as new seasons are tracked.
- Model development within role (alignment) to separate usage from capacity.

