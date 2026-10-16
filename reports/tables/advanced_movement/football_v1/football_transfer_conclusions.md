### 08B.2 — football transfer benchmark

Evidence label: **EXPLORATORY — source semantics inferred; definitions locked** — exploratory association only — never a confirmed predictive claim.
Gate C: **PASS** (a valid experiment, whatever the sign of the result).

| Target | Role | Model vs reference | n | Δ RMSE [95 %] | Relative | p linkage | p outcome | Holm | Reading |
|---|---|---|---|---|---|---|---|---|---|
| `FO01.S0` | primary | M2 vs M1 | 84 | -0.0007 [-0.0093, +0.0082] | -0.18 % | 0.345 | 0.821 | 0.691 | not established (interval includes 0) |
| `FO01.S0` | secondary | M5 vs M1 | 84 | -0.0108 [-0.0244, -0.0001] | -2.85 % | 0.886 | n/a | 0.886 | the tracking model is worse (interval below 0) |
| `FO01.S0` | control | M3 vs M1 | 84 | -0.0014 [-0.0098, +0.0078] | -0.36 % | n/a | n/a | n/a | not established (interval includes 0) |
| `FO01.S0` | control | M4 vs M1 | 84 | -0.0016 [-0.0102, +0.0073] | -0.43 % | n/a | n/a | n/a | not established (interval includes 0) |
| `FO01.S0` | control | M6 vs M1 | 84 | +0.0021 [-0.0080, +0.0120] | +0.55 % | n/a | n/a | n/a | not established (interval includes 0) |
| `FO01.S0` | control | M5 vs M2 | 84 | -0.0101 [-0.0253, +0.0041] | -2.67 % | n/a | n/a | n/a | not established (interval includes 0) |
| `FO01.S0` | control | M5 vs M6 | 84 | -0.0129 [-0.0269, +0.0007] | -3.42 % | n/a | n/a | n/a | not established (interval includes 0) |
| `FO12.T2` | primary | M2 vs M1 | 133 | +0.0001 [-0.0003, +0.0005] | +0.38 % | n/a | n/a | 0.640 | not established (interval includes 0) |
| `FO12.T2` | secondary | M5 vs M1 | 133 | +0.0003 [-0.0003, +0.0009] | +1.13 % | n/a | n/a | 0.598 | not established (interval includes 0) |
| `FO12.T2` | control | M3 vs M1 | 133 | +0.0001 [-0.0003, +0.0006] | +0.47 % | n/a | n/a | n/a | not established (interval includes 0) |
| `FO12.T2` | control | M4 vs M1 | 133 | +0.0001 [-0.0002, +0.0006] | +0.57 % | n/a | n/a | n/a | not established (interval includes 0) |
| `FO12.T2` | control | M6 vs M1 | 133 | +0.0002 [-0.0003, +0.0007] | +0.74 % | n/a | n/a | n/a | not established (interval includes 0) |
| `FO12.T2` | control | M5 vs M2 | 133 | +0.0002 [-0.0003, +0.0006] | +0.75 % | n/a | n/a | n/a | not established (interval includes 0) |
| `FO12.T2` | control | M5 vs M6 | 133 | +0.0001 [-0.0002, +0.0004] | +0.39 % | n/a | n/a | n/a | not established (interval includes 0) |

Baselines: `FO01.S0` M0 r2_oos 0.175; `FO01.S0` M0 rmse 0.378; `FO01.S0` M0 spearman 0.420; `FO01.S0` M1 r2_oos 0.170; `FO01.S0` M1 rmse 0.379; `FO01.S0` M1 spearman 0.421; `FO01.S0` POS r2_oos 0.054; `FO01.S0` POS rmse 0.405; `FO01.S0` POS spearman 0.135; `FO12.T2` M0 r2_oos 0.148; `FO12.T2` M0 rmse 0.026; `FO12.T2` M0 spearman 0.306; `FO12.T2` M1 r2_oos 0.137; `FO12.T2` M1 rmse 0.026; `FO12.T2` M1 spearman 0.340; `FO12.T2` POS r2_oos 0.167; `FO12.T2` POS rmse 0.025; `FO12.T2` POS spearman 0.305

Matched stopwatch prospects: 624 of 833 frozen 06C pairs lie inside the football cohort (210 athletes).
- `FO01.S0` M0: concordance 0.478 [0.354, 0.607], versus official +0.040 [-0.077, +0.159]
- `FO01.S0` M1: concordance 0.438 [0.316, 0.558], versus official n/a [n/a, n/a]
- `FO01.S0` M2: concordance 0.462 [0.342, 0.581], versus official +0.024 [-0.070, +0.125]
- `FO01.S0` M3: concordance 0.450 [0.332, 0.565], versus official +0.012 [-0.094, +0.112]
- `FO01.S0` M4: concordance 0.450 [0.330, 0.564], versus official +0.012 [-0.091, +0.113]
- `FO01.S0` M5: concordance 0.462 [0.337, 0.588], versus official +0.024 [-0.037, +0.086]
- `FO01.S0` M6: concordance 0.478 [0.376, 0.583], versus official +0.040 [-0.066, +0.137]
- `FO01.S0` MEAN: concordance 0.454 [0.339, 0.575], versus official +0.016 [-0.138, +0.185]
- `FO01.S0` POS: concordance 0.458 [0.337, 0.573], versus official +0.020 [-0.145, +0.188]
- `FO12.T2` M0: concordance 0.515 [0.413, 0.611], versus official +0.038 [-0.039, +0.121]
- `FO12.T2` M1: concordance 0.477 [0.378, 0.573], versus official n/a [n/a, n/a]
- `FO12.T2` M2: concordance 0.477 [0.376, 0.576], versus official +0.000 [-0.051, +0.053]
- `FO12.T2` M3: concordance 0.466 [0.367, 0.564], versus official -0.011 [-0.067, +0.041]
- `FO12.T2` M4: concordance 0.461 [0.357, 0.559], versus official -0.016 [-0.073, +0.036]
- `FO12.T2` M5: concordance 0.448 [0.352, 0.549], versus official -0.029 [-0.116, +0.049]
- `FO12.T2` M6: concordance 0.483 [0.383, 0.580], versus official +0.005 [-0.062, +0.076]
- `FO12.T2` MEAN: concordance 0.504 [0.417, 0.591], versus official +0.027 [-0.092, +0.143]
- `FO12.T2` POS: concordance 0.501 [0.401, 0.596], versus official +0.024 [-0.093, +0.138]

Limits: one linked cohort of about 90 receivers; an interaction outcome with inferred semantics; intervals cover athlete sampling, not model refitting; the practical margin (2%) is a heuristic; the linkage permutation also breaks the tracking–official link of the same athlete. Milestone 06 (no validated incremental prediction of NFL movement) is unchanged.