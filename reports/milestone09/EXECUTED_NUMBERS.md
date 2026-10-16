# Milestone 09 — executed numbers

_Generated 2026-10-14T13:20:58Z from the application mart `warehouse/gold/advanced_movement/intelligence_v1/app_mart/v1` (run mode full, built 2026-10-14T13:13:05Z)._

## Coverage

- Combine prospects: 510 (418 with a tracked forty); positions DB, DL, OL, TE, WR
- Tracked forty attempts: 793
- NFL movement-demand episodes: 416,507 (9 measurable categories)
- Stopwatch-twin pairs: 3,266, of which 2,097 have at least one distinguishable curve feature
- Case studies: 23 of 10 types

## Hero pair

- Anthony Bradford and Mason McCormick (OL): official forty 5.08 s and 5.08 s
- band acceleration 2 → 4 yd/s: 4.598 vs 9.427 yd/s², difference +4.829 [+3.770, +5.889]
- band acceleration 4 → 6 yd/s: 3.726 vs 7.171 yd/s², difference +3.445 [+2.580, +4.310]
- band acceleration 6 → 8 yd/s: 1.897 vs 6.048 yd/s², difference +4.151 [+3.389, +4.914]
- speed at 10 yd: 7.593 vs 9.103 yd/s, difference +1.510 [+1.232, +1.787]
- speed at 20 yd: 9.113 vs 8.766 yd/s, difference -0.347 [-0.507, -0.187]
- speed development 4 → 8 yd/s: 1.592 vs 0.683 s, difference -0.909 [-1.127, -0.691]
- tracked top speed (p95): 9.337 vs 9.557 yd/s, difference +0.220 [+0.043, +0.397]

## How often stopwatch twins differ (5 % expected by chance)

| Feature | Tested pairs | Distinguishable |
|---|---|---|
| acc_2_4 | 3,128 | 14.5 % |
| acc_4_6 | 3,128 | 16.4 % |
| acc_6_8 | 3,128 | 13.9 % |
| speed_10yd | 3,128 | 16.7 % |
| speed_20yd | 3,128 | 26.0 % |
| speed_30yd | 3,128 | 38.0 % |
| time_4_8 | 3,128 | 20.6 % |
| top_speed | 3,128 | 40.0 % |

## NFL movement demands (all positions)

| Demand | Episodes | Athletes | Per 100 eligible player-plays |
|---|---|---|---|
| acceleration from low speed | 228,227 | 398 | 94.3 |
| braking and deceleration | 67,836 | 390 | 28.0 |
| speed maintenance | 28,602 | 387 | 11.8 |
| acceleration from intermediate speed | 22,875 | 384 | 9.5 |
| repeated speed-change sequence | 19,799 | 378 | 8.2 |
| acceleration following deceleration | 18,001 | 370 | 7.4 |
| sustained high-speed running | 15,659 | 300 | 6.5 |
| directional redirection | 14,311 | 358 | 5.9 |
| transition into high-speed running | 1,197 | 181 | 0.5 |

## Earlier milestones (08B.6 evidence claims)

- C01: Tracked top speed (p95) of the forty is highly repeatable between attempts — ICC 0.95 [0.94, 0.96]; atlas grade Strong
- C02: Speed development (time from 4 to 8 yd/s) is repeatable and only partly explained by the stopwatch — ICC 0.82 [0.78, 0.86]; R² on the stopwatch 0.65; atlas grade Provisional
- C03: Peak tangential acceleration is NOT repeatable enough to describe an athlete — ICC 0.35 [0.20, 0.49]; atlas grade Reject / not currently usable
- C04: Tracked top speed is largely the same information as official testing — R² on the stopwatch 0.87
- C05: The speed-by-distance curve of a forty is low-dimensional — two components explain 97.7 % of held-out curve variance (mean over 15 fold fits)
- C06: TCN-B recognises an athlete's repeat attempt: a repeatable movement fingerprint — repeat-attempt retrieval AUC 0.873 [0.856, 0.890]
- C07: The fingerprint still recognises the athlete among prospects with near-identical official results — matched-stopwatch retrieval AUC 0.796 [0.768, 0.824]
- C08: TCN-B did NOT establish an improvement of held-out NFL sustained-acceleration prediction beyond official testing — Δ RMSE +0.0003 [-0.0008, +0.0015] (+0.16 % of the official-model error); interval includes 0
- C09: NFL movement windows are dominated by context and opportunity, not by the athlete — athlete share of within-position NFL window variance 6.3 % [5.0, 7.2]
- C10: Only large incremental effects were detectable in the Milestone 06 design — smallest effect detected with 80 % power — M2: planted share 0.23 of residual variance (Δ RMSE 6.7 %); M5: planted share 0.19 of residual variance (Δ RMSE 6.4 %)
- C11: Recorded receiver separation is a reliable athlete-level quantity with usable linked support — odd / even game reliability 0.74 [0.52, 0.87]; semantic status plausible_inferred; screen status provisional
- C12: Football-outcome registry decision — FREEZE → frozen_exploratory: FO01.S0 (primary, PROVISIONAL), FO12.T2 (secondary, PROVISIONAL), FO13.K0 (secondary, PROVISIONAL)
- C13: Gate B of the primary football outcome (game-split reliability and support) — FO01.S0: odd / even Spearman–Brown 0.742 (lower bound 0.516); reference 0.7, floor 0.5 — met
- C14: Primary football transfer comparison (engineered speed development beyond official testing) — M2 versus M1 on FO01.S0: Δ RMSE -0.0007 [-0.0093, +0.0082], linkage permutation p 0.345 — not established (interval includes 0) (EXPLORATORY — source semantics inferred; definitions locked)
- C15: Secondary football transfer comparison (TCN-B beyond official testing) — M5 versus M1 on FO01.S0: Δ RMSE -0.0108 [-0.0244, -0.0001], linkage permutation p 0.886 — the tracking model is worse (interval below 0) (EXPLORATORY — source semantics inferred; definitions locked)
- C16: Detectability of an incremental effect in the football cohort — smallest effect detected with 80 % power — M2: 80 % power not reached in the grid (max 0.60); M5: 80 % power not reached in the grid (max 0.29)
- C17: Combine tracking characteristics that stay repeatable after official tests are accounted for — repeatable beyond official tests: tracked top speed (p95), speed development 4 → 8 yd/s, FPCA speed-curve scores (K = 2), TCN-B movement fingerprint (8-d). Others: acceleration over 1 s from 4 yd/s (unstable); speed gain over 10 yd from 4 yd/s (weak_residual_signal)
- C18: The scouting scorecard is descriptive, uncertainty-aware and abstains where evidence is missing — every criterion met
- C19: Get-off cannot be used as an outcome — player_get_off: blocked_definition (source semantics unresolved)
- C20: Cohort size a future study needs — power to detect a planted share 0.05 of residual variance by cohort size — n = 84: 0.12, n = 150: 0.24, n = 284: 0.48, n = 500: 0.78, n = 1000: 1.00
