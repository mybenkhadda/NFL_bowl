# Physically aligned transfer — interpretation (Milestone 07B)

**Status: EXPLORATORY (Milestone 07B).** Milestone 06 remains the confirmatory study (no method shown to improve on official Combine testing). Nothing below is a confirmatory finding; an interval that excludes 0 here is a hypothesis for an independent cohort.

## H1 — Combine speed → NFL speed (positive control)

| Tier | Outcome | Model vs reference | n | Δ RMSE [95 %] | Relative | Holm p (primaries) | Reading |
|---|---|---|---|---|---|---|---|
| primary | `C0:O1.speed_p90_mean` | M1 vs M0 | 284 | -0.0041 [-0.0098, +0.0013] | -1.5 % | 0.453 | not established (interval includes 0) |
| secondary | `C0:O1.speed_p90_mean` | M3 vs M1 | 284 | +0.0028 [-0.0011, +0.0068] | +1.0 % | n/a | not established (interval includes 0) |
| secondary | `C0:O1.speed_p90_mean` | S0 vs M0 | 284 | +0.0032 [-0.0019, +0.0084] | +1.2 % | n/a | not established (interval includes 0) |

## H2 — Combine speed development → NFL acceleration

| Tier | Outcome | Model vs reference | n | Δ RMSE [95 %] | Relative | Holm p (primaries) | Reading |
|---|---|---|---|---|---|---|---|
| primary | `C0:O3.band_acc_v4_6_mean` | M2 vs M1 | 251 | +0.0013 [-0.0005, +0.0029] | +0.8 % | 0.453 | not established (interval includes 0) |
| secondary | `C0:O3.band_acc_v0_2_mean` | M2 vs M1 | 284 | -0.0005 [-0.0016, +0.0006] | -0.2 % | n/a | not established (interval includes 0) |
| secondary | `C0:O3.band_acc_v4_6_mean` | E0 vs M0 | 251 | -0.0005 [-0.0016, +0.0005] | -0.4 % | n/a | not established (interval includes 0) |
| secondary | `C0:O3.band_acc_v6_8_mean` | M2 vs M1 | 155 | +0.0002 [-0.0002, +0.0006] | +0.1 % | n/a | not established (interval includes 0) |
| secondary | `C4:O2.accel_sust05_mean` | M2 vs M1 | 284 | -0.0002 [-0.0007, +0.0004] | -0.2 % | n/a | not established (interval includes 0) |

## H3 — TCN-B beyond official measurements

| Tier | Outcome | Model vs reference | n | Δ RMSE [95 %] | Relative | Holm p (primaries) | Reading |
|---|---|---|---|---|---|---|---|
| primary | `C4:O2.accel_sust05_mean` | M5 vs M1 | 284 | -0.0003 [-0.0010, +0.0004] | -0.3 % | 0.521 | not established (interval includes 0) |
| secondary | `C0:O1.speed_p90_mean` | M5 vs M1 | 284 | +0.0014 [-0.0049, +0.0077] | +0.5 % | n/a | not established (interval includes 0) |
| secondary | `C0:O3.band_acc_v4_6_mean` | M5 vs M1 | 251 | +0.0040 [-0.0006, +0.0090] | +2.6 % | n/a | not established (interval includes 0) |
| secondary | `C2:O2.accel_sust05_mean` | M5 vs M1 | 284 | +0.0019 [-0.0033, +0.0075] | +0.9 % | n/a | not established (interval includes 0) |
| secondary | `C4:O2.accel_sust05_mean` | M3 vs M1 | 284 | -0.0001 [-0.0007, +0.0006] | -0.1 % | n/a | not established (interval includes 0) |
| secondary | `C4:O2.accel_sust05_mean` | M4 vs M1 | 284 | -0.0001 [-0.0007, +0.0006] | -0.1 % | n/a | not established (interval includes 0) |
| secondary | `C4:O2.accel_sust05_mean` | M5 vs M3 | 284 | -0.0002 [-0.0004, +0.0001] | -0.2 % | n/a | not established (interval includes 0) |
| secondary | `C4:O2.accel_sust05_mean` | M5 vs M2 | 284 | -0.0001 [-0.0006, +0.0004] | -0.1 % | n/a | not established (interval includes 0) |
| secondary | `C4:O2.accel_sust05_mean` | M5 vs M4 | 284 | -0.0002 [-0.0006, +0.0002] | -0.2 % | n/a | not established (interval includes 0) |
| secondary | `C4:O2.accel_sust05_mean` | M5 vs M6 | 284 | +0.0000 [-0.0002, +0.0003] | +0.0 % | n/a | not established (interval includes 0) |

## H4 — position-conditioned transfer

- `C0:O1.speed_p90_mean` M1 vs M0, position:DB (n = 67): -0.0022 [-0.0135, +0.0090]
- `C0:O1.speed_p90_mean` M1 vs M0, position:DL (n = 70): -0.0060 [-0.0173, +0.0046]
- `C0:O1.speed_p90_mean` M1 vs M0, position:OL (n = 63): -0.0080 [-0.0239, +0.0086]
- `C0:O1.speed_p90_mean` M1 vs M0, position:TE (n = 20): -0.0028 [-0.0247, +0.0296]
- `C0:O1.speed_p90_mean` M1 vs M0, position:WR (n = 64): -0.0020 [-0.0114, +0.0061]
- `C0:O3.band_acc_v4_6_mean` M2 vs M1, position:DB (n = 64): +0.0021 [-0.0003, +0.0045]
- `C0:O3.band_acc_v4_6_mean` M2 vs M1, position:DL (n = 57): -0.0013 [-0.0056, +0.0027]
- `C0:O3.band_acc_v4_6_mean` M2 vs M1, position:OL (n = 48): +0.0011 [-0.0050, +0.0069]
- `C0:O3.band_acc_v4_6_mean` M2 vs M1, position:TE (n = 19): -0.0011 [n/a, n/a]
- `C0:O3.band_acc_v4_6_mean` M2 vs M1, position:WR (n = 63): +0.0030 [-0.0001, +0.0060]
- `C4:O2.accel_sust05_mean` M5 vs M1, position:DB (n = 67): -0.0003 [-0.0012, +0.0006]
- `C4:O2.accel_sust05_mean` M5 vs M1, position:DL (n = 70): -0.0004 [-0.0019, +0.0011]
- `C4:O2.accel_sust05_mean` M5 vs M1, position:OL (n = 63): -0.0009 [-0.0024, +0.0006]
- `C4:O2.accel_sust05_mean` M5 vs M1, position:TE (n = 20): -0.0004 [-0.0034, +0.0031]
- `C4:O2.accel_sust05_mean` M5 vs M1, position:WR (n = 64): +0.0005 [-0.0010, +0.0023]

Subgroups are scored from the pooled fit and are exploratory: the group with the largest estimate is not a discovery.

## Matched prospects

- `C0:O1.speed_p90_mean` E0: concordance 0.513 [+0.446, +0.576], vs official +0.023 [-0.024, +0.068] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` M0: concordance 0.522 [+0.455, +0.587], vs official +0.032 [-0.012, +0.078] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` M1: concordance 0.490 [+0.425, +0.554], vs official n/a [n/a, n/a] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` M2: concordance 0.491 [+0.427, +0.556], vs official +0.001 [-0.019, +0.024] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` M3: concordance 0.511 [+0.449, +0.572], vs official +0.022 [-0.021, +0.064] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` M4: concordance 0.498 [+0.435, +0.558], vs official +0.008 [-0.025, +0.047] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` M5: concordance 0.502 [+0.438, +0.564], vs official +0.012 [-0.043, +0.067] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` M6: concordance 0.493 [+0.425, +0.559], vs official +0.004 [-0.048, +0.058] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` MEAN: concordance 0.454 [+0.387, +0.515], vs official -0.036 [-0.126, +0.051] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` POS: concordance 0.444 [+0.380, +0.506], vs official -0.046 [-0.122, +0.032] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O1.speed_p90_mean` S0: concordance 0.519 [+0.454, +0.584], vs official +0.029 [-0.017, +0.077] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` E0: concordance 0.546 [+0.478, +0.617], vs official +0.023 [-0.027, +0.075] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` M0: concordance 0.535 [+0.468, +0.604], vs official +0.012 [-0.040, +0.068] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` M1: concordance 0.523 [+0.454, +0.592], vs official n/a [n/a, n/a] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` M2: concordance 0.533 [+0.462, +0.603], vs official +0.010 [-0.007, +0.027] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` M3: concordance 0.527 [+0.458, +0.596], vs official +0.004 [-0.021, +0.028] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` M4: concordance 0.517 [+0.450, +0.584], vs official -0.006 [-0.032, +0.018] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` M5: concordance 0.514 [+0.441, +0.579], vs official -0.010 [-0.061, +0.040] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` M6: concordance 0.517 [+0.449, +0.584], vs official -0.006 [-0.052, +0.040] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` MEAN: concordance 0.490 [+0.423, +0.550], vs official -0.034 [-0.127, +0.061] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` POS: concordance 0.435 [+0.372, +0.501], vs official -0.089 [-0.180, +0.003] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O2.accel_sust05_mean` S0: concordance 0.535 [+0.468, +0.601], vs official +0.012 [-0.032, +0.061] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` E0: concordance 0.544 [+0.480, +0.605], vs official +0.048 [+0.007, +0.091] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` M0: concordance 0.510 [+0.447, +0.574], vs official +0.014 [-0.037, +0.064] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` M1: concordance 0.496 [+0.430, +0.561], vs official n/a [n/a, n/a] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` M2: concordance 0.492 [+0.427, +0.555], vs official -0.004 [-0.033, +0.023] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` M3: concordance 0.508 [+0.439, +0.574], vs official +0.012 [-0.019, +0.043] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` M4: concordance 0.507 [+0.436, +0.575], vs official +0.011 [-0.032, +0.052] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` M5: concordance 0.462 [+0.397, +0.530], vs official -0.034 [-0.084, +0.017] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` M6: concordance 0.489 [+0.422, +0.552], vs official -0.007 [-0.060, +0.049] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` MEAN: concordance 0.485 [+0.423, +0.546], vs official -0.011 [-0.098, +0.074] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` POS: concordance 0.447 [+0.386, +0.510], vs official -0.049 [-0.133, +0.038] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v0_2_mean` S0: concordance 0.504 [+0.439, +0.567], vs official +0.008 [-0.034, +0.052] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` E0: concordance 0.516 [+0.443, +0.595], vs official +0.025 [-0.039, +0.090] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` M0: concordance 0.511 [+0.437, +0.593], vs official +0.021 [-0.042, +0.089] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` M1: concordance 0.490 [+0.421, +0.569], vs official n/a [n/a, n/a] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` M2: concordance 0.504 [+0.429, +0.581], vs official +0.013 [-0.033, +0.055] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` M3: concordance 0.505 [+0.437, +0.580], vs official +0.015 [-0.048, +0.076] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` M4: concordance 0.501 [+0.432, +0.575], vs official +0.010 [-0.039, +0.056] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` M5: concordance 0.526 [+0.455, +0.600], vs official +0.036 [-0.035, +0.110] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` M6: concordance 0.505 [+0.436, +0.576], vs official +0.015 [-0.066, +0.092] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` MEAN: concordance 0.466 [+0.397, +0.537], vs official -0.024 [-0.110, +0.065] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` POS: concordance 0.423 [+0.356, +0.491], vs official -0.067 [-0.168, +0.029] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v4_6_mean` S0: concordance 0.487 [+0.409, +0.563], vs official -0.003 [-0.083, +0.074] (669 pairs, 241 athletes, an athlete in up to 14 pairs)
- `C0:O3.band_acc_v6_8_mean` E0: concordance 0.472 [+0.381, +0.566], vs official -0.024 [-0.132, +0.090] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` M0: concordance 0.491 [+0.408, +0.585], vs official -0.005 [-0.110, +0.104] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` M1: concordance 0.496 [+0.401, +0.597], vs official n/a [n/a, n/a] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` M2: concordance 0.494 [+0.402, +0.596], vs official -0.002 [-0.030, +0.023] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` M3: concordance 0.511 [+0.416, +0.605], vs official +0.015 [-0.063, +0.091] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` M4: concordance 0.504 [+0.402, +0.602], vs official +0.007 [-0.059, +0.074] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` M5: concordance 0.521 [+0.424, +0.619], vs official +0.024 [-0.053, +0.111] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` M6: concordance 0.513 [+0.410, +0.613], vs official +0.017 [-0.063, +0.104] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` MEAN: concordance 0.510 [+0.418, +0.604], vs official +0.013 [-0.130, +0.162] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` POS: concordance 0.425 [+0.338, +0.519], vs official -0.072 [-0.195, +0.051] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C0:O3.band_acc_v6_8_mean` S0: concordance 0.557 [+0.455, +0.651], vs official +0.061 [-0.040, +0.160] (411 pairs, 147 athletes, an athlete in up to 13 pairs)
- `C2:O2.accel_sust05_mean` E0: concordance 0.549 [+0.477, +0.617], vs official +0.007 [-0.044, +0.059] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` M0: concordance 0.527 [+0.460, +0.597], vs official -0.014 [-0.073, +0.038] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` M1: concordance 0.541 [+0.475, +0.609], vs official n/a [n/a, n/a] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` M2: concordance 0.544 [+0.475, +0.614], vs official +0.002 [-0.022, +0.029] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` M3: concordance 0.533 [+0.464, +0.600], vs official -0.008 [-0.038, +0.020] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` M4: concordance 0.538 [+0.468, +0.608], vs official -0.004 [-0.029, +0.020] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` M5: concordance 0.515 [+0.446, +0.585], vs official -0.026 [-0.073, +0.018] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` M6: concordance 0.521 [+0.454, +0.586], vs official -0.020 [-0.070, +0.030] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` MEAN: concordance 0.496 [+0.432, +0.558], vs official -0.046 [-0.134, +0.043] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` POS: concordance 0.448 [+0.385, +0.511], vs official -0.094 [-0.188, -0.003] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C2:O2.accel_sust05_mean` S0: concordance 0.537 [+0.474, +0.603], vs official -0.005 [-0.056, +0.042] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` E0: concordance 0.430 [+0.359, +0.504], vs official -0.060 [-0.125, +0.008] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` M0: concordance 0.461 [+0.387, +0.535], vs official -0.029 [-0.104, +0.044] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` M1: concordance 0.490 [+0.421, +0.561], vs official n/a [n/a, n/a] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` M2: concordance 0.492 [+0.421, +0.566], vs official +0.002 [-0.031, +0.035] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` M3: concordance 0.489 [+0.418, +0.561], vs official -0.001 [-0.034, +0.033] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` M4: concordance 0.483 [+0.412, +0.553], vs official -0.007 [-0.041, +0.028] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` M5: concordance 0.471 [+0.403, +0.543], vs official -0.019 [-0.051, +0.014] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` M6: concordance 0.471 [+0.402, +0.541], vs official -0.019 [-0.053, +0.015] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` MEAN: concordance 0.474 [+0.412, +0.536], vs official -0.016 [-0.101, +0.070] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` POS: concordance 0.456 [+0.395, +0.523], vs official -0.034 [-0.091, +0.023] (833 pairs, 275 athletes, an athlete in up to 14 pairs)
- `C4:O2.accel_sust05_mean` S0: concordance 0.432 [+0.362, +0.507], vs official -0.058 [-0.124, +0.015] (833 pairs, 275 athletes, an athlete in up to 14 pairs)

## Descriptive physical associations (in-sample)

- tracked top speed p95 [yd/s] ↔ `C0:O1.speed_p90_mean` after position (n = 284): r = +0.455 [+0.371, +0.535], expected sign +
- tracked top speed p95 [yd/s] ↔ `C0:O1.speed_p90_mean` after official (n = 284): r = +0.147 [+0.036, +0.261], expected sign +
- official forty [s] ↔ `C0:O1.speed_p90_mean` after position (n = 284): r = -0.452 [-0.537, -0.356], expected sign −
- official forty [s] ↔ `C0:O1.speed_p90_mean` after official (n = 284): r = n/a [n/a, n/a], expected sign −
- Combine accel 4→8 yd/s (4 / time) [yd/s²] ↔ `C0:O3.band_acc_v4_6_mean` after position (n = 251): r = +0.194 [+0.087, +0.297], expected sign +
- Combine accel 4→8 yd/s (4 / time) [yd/s²] ↔ `C0:O3.band_acc_v4_6_mean` after official (n = 251): r = -0.078 [-0.180, +0.038], expected sign +
- Combine accel 4→8 yd/s (4 / time) [yd/s²] ↔ `C0:O3.band_acc_v6_8_mean` after position (n = 155): r = +0.181 [+0.052, +0.315], expected sign +
- Combine accel 4→8 yd/s (4 / time) [yd/s²] ↔ `C0:O3.band_acc_v6_8_mean` after official (n = 155): r = -0.051 [-0.200, +0.112], expected sign +
- Combine accel 4→8 yd/s (4 / time) [yd/s²] ↔ `C0:O3.band_acc_v0_2_mean` after position (n = 284): r = +0.300 [+0.171, +0.430], expected sign +
- Combine accel 4→8 yd/s (4 / time) [yd/s²] ↔ `C0:O3.band_acc_v0_2_mean` after official (n = 284): r = -0.083 [-0.209, +0.051], expected sign +
- Combine accel 4→8 yd/s (4 / time) [yd/s²] ↔ `C4:O2.accel_sust05_mean` after position (n = 284): r = +0.101 [-0.016, +0.216], expected sign +
- Combine accel 4→8 yd/s (4 / time) [yd/s²] ↔ `C4:O2.accel_sust05_mean` after official (n = 284): r = -0.054 [-0.171, +0.062], expected sign +

## Coverage

- Y_BAND02 `C0:O3.band_acc_v0_2_mean`: 284 of 284 athletes (DB 67 · DL 70 · OL 63 · TE 20 · WR 64)
- Y_BAND46 `C0:O3.band_acc_v4_6_mean`: 251 of 284 athletes (DB 64 · DL 57 · OL 48 · TE 19 · WR 63)
- Y_BAND68 `C0:O3.band_acc_v6_8_mean`: 155 of 284 athletes (DB 55 · DL 19 · OL 1 · TE 18 · WR 62)
- Y_CTX `C4:O2.accel_sust05_mean`: 284 of 284 athletes (DB 67 · DL 70 · OL 63 · TE 20 · WR 64)
- Y_RAW `C0:O2.accel_sust05_mean`: 284 of 284 athletes (DB 67 · DL 70 · OL 63 · TE 20 · WR 64)
- Y_SPEED `C0:O1.speed_p90_mean`: 284 of 284 athletes (DB 67 · DL 70 · OL 63 · TE 20 · WR 64)
- Y_START `C2:O2.accel_sust05_mean`: 284 of 284 athletes (DB 67 · DL 70 · OL 63 · TE 20 · WR 64)

Limits: outcomes are observed NFL movement shaped by role, scheme and usage; the context-conditioned outcomes remove observed labels only; intervals cover athlete sampling, not model refitting; outcomes have different cohorts and are not compared with each other; the comparisons of this notebook were declared after the Milestone 06 result was known.