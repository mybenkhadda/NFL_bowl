# Detectability of incremental Combine → NFL prediction (Milestone 07C)

Simulation `f7d3e88b6cd36566` — COMPLETE (6800 replicates).

## Assumptions

Synthetic target y* = μ + σ_res,pos · (√(1 − f) · ε + √f · z). μ = the athlete's real Milestone 06 out-of-fold prediction of the official model; σ_res,pos = SD of the real residuals of the position; ε = real standardised residuals resampled within position; z = unit-variance leading direction of the candidate representation (fold-0 encoder), either orthogonal to the official design ('conditional': f is the partial R² given official testing) or as is ('raw'). Folds, athletes, covariates, missingness and fold-fitted representations are the real ones; the pipeline, bootstrap and detection criterion are those of Milestone 06. The signal is linear and one-dimensional — favourable to a ridge read-out: real effects that are non-linear, spread over many weak directions or specific to a position are harder to detect than the planted ones, so the power shown is an upper reading.

## Power on the real cohort

| Candidate | Mode | f | Power [MC 95 %] | Mean Δ RMSE | Relative | Interval coverage |
|---|---|---|---|---|---|---|
| M2 | conditional | 0 | 0.040 [0.020, 0.077] | -0.0000 | -0.01 % | 0.825 |
| M2 | conditional | 0.02 | 0.045 [0.024, 0.083] | -0.0000 | -0.00 % | 0.820 |
| M2 | conditional | 0.05 | 0.065 [0.038, 0.108] | 0.0004 | 0.20 % | 0.700 |
| M2 | conditional | 0.1 | 0.185 [0.137, 0.245] | 0.0021 | 1.18 % | 0.730 |
| M2 | conditional | 0.2 | 0.740 [0.675, 0.796] | 0.0092 | 5.05 % | 0.830 |
| M2 | conditional | 0.35 | 1.000 [0.981, 1.000] | 0.0220 | 12.35 % | 0.965 |
| M2 | raw | 0 | 0.025 [0.011, 0.057] | -0.0000 | -0.02 % | 0.755 |
| M2 | raw | 0.02 | 0.060 [0.035, 0.102] | 0.0000 | 0.01 % | 0.760 |
| M2 | raw | 0.05 | 0.075 [0.046, 0.120] | 0.0002 | 0.09 % | 0.580 |
| M2 | raw | 0.1 | 0.160 [0.116, 0.217] | 0.0003 | 0.17 % | 0.675 |
| M2 | raw | 0.2 | 0.385 [0.320, 0.454] | 0.0010 | 0.60 % | 0.740 |
| M2 | raw | 0.35 | 0.760 [0.696, 0.814] | 0.0025 | 1.62 % | 0.835 |
| M5 | conditional | 0 | 0.020 [0.008, 0.050] | -0.0003 | -0.15 % | 0.815 |
| M5 | conditional | 0.02 | 0.100 [0.066, 0.149] | 0.0006 | 0.33 % | 0.795 |
| M5 | conditional | 0.05 | 0.240 [0.186, 0.304] | 0.0022 | 1.16 % | 0.800 |
| M5 | conditional | 0.1 | 0.485 [0.417, 0.554] | 0.0052 | 2.77 % | 0.830 |
| M5 | conditional | 0.2 | 0.830 [0.772, 0.876] | 0.0127 | 6.78 % | 0.900 |
| M5 | conditional | 0.35 | 1.000 [0.981, 1.000] | 0.0279 | 14.78 % | 0.920 |
| M5 | raw | 0 | 0.025 [0.011, 0.057] | -0.0002 | -0.09 % | 0.775 |
| M5 | raw | 0.02 | 0.045 [0.024, 0.083] | -0.0000 | -0.00 % | 0.875 |
| M5 | raw | 0.05 | 0.095 [0.062, 0.144] | 0.0004 | 0.23 % | 0.855 |
| M5 | raw | 0.1 | 0.195 [0.146, 0.255] | 0.0011 | 0.59 % | 0.780 |
| M5 | raw | 0.2 | 0.370 [0.306, 0.439] | 0.0020 | 1.16 % | 0.795 |
| M5 | raw | 0.35 | 0.635 [0.566, 0.699] | 0.0039 | 2.54 % | 0.845 |

## No-signal calibration

- power · M2 · conditional · n = 284 · ALL: false-positive rate 0.040 [0.020, 0.077] (nominal 0.025); mean Δ RMSE -0.00002, SD 0.00032
- power · M2 · raw · n = 284 · ALL: false-positive rate 0.025 [0.011, 0.057] (nominal 0.025); mean Δ RMSE -0.00004, SD 0.00038
- power · M5 · conditional · n = 284 · ALL: false-positive rate 0.020 [0.008, 0.050] (nominal 0.025); mean Δ RMSE -0.00027, SD 0.00096
- power · M5 · raw · n = 284 · ALL: false-positive rate 0.025 [0.011, 0.057] (nominal 0.025); mean Δ RMSE -0.00016, SD 0.00084
- sample_size · M5 · conditional · n = 150 · ALL: false-positive rate 0.010 [0.002, 0.054] (nominal 0.025); mean Δ RMSE -0.00031, SD 0.00206
- sample_size · M5 · conditional · n = 284 · ALL: false-positive rate 0.050 [0.022, 0.112] (nominal 0.025); mean Δ RMSE -0.00002, SD 0.00085
- sample_size · M5 · conditional · n = 500 · ALL: false-positive rate 0.020 [0.006, 0.070] (nominal 0.025); mean Δ RMSE -0.00010, SD 0.00053
- sample_size · M5 · conditional · n = 1000 · ALL: false-positive rate 0.010 [0.002, 0.054] (nominal 0.025); mean Δ RMSE -0.00014, SD 0.00040
- sample_size · M5 · conditional · n = 2000 · ALL: false-positive rate 0.010 [0.002, 0.054] (nominal 0.025); mean Δ RMSE -0.00006, SD 0.00016

## Minimum detectable effects

- power · M2 · conditional · n = 284 · ALL: f = 0.235, Δ RMSE 0.0122 (6.73 % of the official error)
- power · M2 · conditional · n = 284 · DB: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M2 · conditional · n = 284 · DL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M2 · conditional · n = 284 · OL: f = 0.343, Δ RMSE 0.0404 (26.52 % of the official error)
- power · M2 · conditional · n = 284 · TE: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M2 · conditional · n = 284 · WR: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M2 · raw · n = 284 · ALL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M2 · raw · n = 284 · DB: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M2 · raw · n = 284 · DL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M2 · raw · n = 284 · OL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M2 · raw · n = 284 · TE: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M2 · raw · n = 284 · WR: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · conditional · n = 284 · ALL: f = 0.191, Δ RMSE 0.0121 (6.43 % of the official error)
- power · M5 · conditional · n = 284 · DB: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · conditional · n = 284 · DL: f = 0.313, Δ RMSE 0.0335 (18.65 % of the official error)
- power · M5 · conditional · n = 284 · OL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · conditional · n = 284 · TE: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · conditional · n = 284 · WR: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · raw · n = 284 · ALL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · raw · n = 284 · DB: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · raw · n = 284 · DL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · raw · n = 284 · OL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · raw · n = 284 · TE: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- power · M5 · raw · n = 284 · WR: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.35: the effect grid does not reach it
- sample_size · M5 · conditional · n = 150 · ALL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 150 · DB: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 150 · DL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 150 · OL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 150 · TE: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 150 · WR: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 284 · ALL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 284 · DB: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 284 · DL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 284 · OL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 284 · TE: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 284 · WR: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 500 · ALL: f = 0.096, Δ RMSE 0.0064 (3.50 % of the official error)
- sample_size · M5 · conditional · n = 500 · DB: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 500 · DL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 500 · OL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 500 · TE: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 500 · WR: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 1000 · ALL: f = 0.049, Δ RMSE 0.0033 (1.79 % of the official error)
- sample_size · M5 · conditional · n = 1000 · DB: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 1000 · DL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 1000 · OL: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 1000 · TE: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 1000 · WR: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 2000 · ALL: f = 0.026, Δ RMSE 0.0018 (1.00 % of the official error)
- sample_size · M5 · conditional · n = 2000 · DB: f = n/a, Δ RMSE n/a (n/a % of the official error) — power stays below 80% up to f = 0.1: the effect grid does not reach it
- sample_size · M5 · conditional · n = 2000 · DL: f = 0.047, Δ RMSE 0.0053 (3.28 % of the official error)
- sample_size · M5 · conditional · n = 2000 · OL: f = 0.088, Δ RMSE 0.0062 (5.08 % of the official error)
- sample_size · M5 · conditional · n = 2000 · TE: f = 0.090, Δ RMSE 0.0159 (6.81 % of the official error)
- sample_size · M5 · conditional · n = 2000 · WR: f = 0.077, Δ RMSE 0.0057 (2.23 % of the official error)

## Information limits

- [cohort] athletes: 284.0000 athletes — paired cohort of Milestone 06
- [cohort] rmse_official_model: 0.1794 target units — real M1 out-of-fold RMSE: the residual a candidate can still explain
- [redundancy] M5: R² of the planted direction on the official design: 0.9461 share — share of the representation's leading direction already explained by official testing
- [null] M5: false-positive rate: 0.0200 [0.0078, 0.0503] rate — nominal 0.025 (one-sided)
- [null] M5: SD of Δ RMSE with no signal: 0.0010 target units — sampling noise of the paired gain
- [null] M5: mean Δ RMSE with no signal: -0.0003 target units — cost of fitting useless predictors
- [mde] M5 (conditional): effect f at 80% power: 0.1913 share of residual variance
- [mde] M5 (conditional): Δ RMSE at 80% power: 0.0121 target units
- [mde] M5 (conditional): relative Δ RMSE at 80% power: 0.0643 share of M1 RMSE
- [mde] M5 (raw): effect f at 80% power: n/a share of residual variance — power stays below 80% up to f = 0.35: the effect grid does not reach it
- [mde] M5 (raw): Δ RMSE at 80% power: n/a target units
- [mde] M5 (raw): relative Δ RMSE at 80% power: n/a share of M1 RMSE
- [observed_m06] M5: Δ RMSE: 0.0003 [-0.0008, 0.0015] target units — real Milestone 06 comparison (unchanged)
- [observed_m06] M5: Δ RMSE in null SDs: 0.5739 SD — distance of the observed gain from the simulated no-signal mean
- [redundancy] M2: R² of the planted direction on the official design: 0.8915 share — share of the representation's leading direction already explained by official testing
- [null] M2: false-positive rate: 0.0400 [0.0204, 0.0769] rate — nominal 0.025 (one-sided)
- [null] M2: SD of Δ RMSE with no signal: 0.0003 target units — sampling noise of the paired gain
- [null] M2: mean Δ RMSE with no signal: -0.0000 target units — cost of fitting useless predictors
- [mde] M2 (conditional): effect f at 80% power: 0.2346 share of residual variance
- [mde] M2 (conditional): Δ RMSE at 80% power: 0.0122 target units
- [mde] M2 (conditional): relative Δ RMSE at 80% power: 0.0673 share of M1 RMSE
- [mde] M2 (raw): effect f at 80% power: n/a share of residual variance — power stays below 80% up to f = 0.35: the effect grid does not reach it
- [mde] M2 (raw): Δ RMSE at 80% power: n/a target units
- [mde] M2 (raw): relative Δ RMSE at 80% power: n/a share of M1 RMSE
- [observed_m06] M2: Δ RMSE: 0.0001 [-0.0002, 0.0004] target units — real Milestone 06 comparison (unchanged)
- [observed_m06] M2: Δ RMSE in null SDs: 0.3605 SD — distance of the observed gain from the simulated no-signal mean
- [reliability] combine: speed development 4 → 8 yd/s (attempt ICC, 04A): 0.8241 reliability — read from persisted Milestone 04–07A tables
- [reliability] combine: tracked top speed p95 (attempt ICC, 04A): 0.9485 reliability — read from persisted Milestone 04–07A tables
- [reliability] nfl: O2.accel_sust05_mean (game halves, 05B): 0.9316 reliability — read from persisted Milestone 04–07A tables
- [reliability] nfl: context-conditioned C4 (cross-fitted game halves, 07A): 0.8293 reliability — read from persisted Milestone 04–07A tables
- [attenuation] combine: speed development 4 → 8 yd/s (attempt ICC, 04A) × nfl: O2.accel_sust05_mean (game halves, 05B): √(rel_x · rel_y): 0.8762 factor on a correlation — classical approximation, NOT a ceiling on prediction (see transfer_power.attenuation)
- [attenuation] combine: speed development 4 → 8 yd/s (attempt ICC, 04A) × nfl: context-conditioned C4 (cross-fitted game halves, 07A): √(rel_x · rel_y): 0.8267 factor on a correlation — classical approximation, NOT a ceiling on prediction (see transfer_power.attenuation)
- [attenuation] combine: tracked top speed p95 (attempt ICC, 04A) × nfl: O2.accel_sust05_mean (game halves, 05B): √(rel_x · rel_y): 0.9400 factor on a correlation — classical approximation, NOT a ceiling on prediction (see transfer_power.attenuation)
- [attenuation] combine: tracked top speed p95 (attempt ICC, 04A) × nfl: context-conditioned C4 (cross-fitted game halves, 07A): √(rel_x · rel_y): 0.8869 factor on a correlation — classical approximation, NOT a ceiling on prediction (see transfer_power.attenuation)

Reading rules: (1) an observed gain smaller than the minimum detectable gain is INCONCLUSIVE, not a demonstrated absence; (2) the attenuation factor is a classical approximation for a correlation and is not a ceiling on prediction; (3) sample-size rows resample the observed athletes — scenario analysis, not a claim about data that will exist; (4) position-level rows have few athletes and wide Monte-Carlo and sampling error.