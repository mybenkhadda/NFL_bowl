# 12A — Functional movement transfer

**Status:** authored 2026-10-15 · execution pending. Module `m12/functional_movement.py`; notebook `12a_functional_movement_transfer.ipynb`.

## Functions
- Combine a_C(v): mean tangential acceleration per 0.5 yd/s bin on the rising phase of the forty (onset → first frame of peak speed), frame-weighted over attempts (`m12_combine_curves`).
- NFL a_N(v): the 12C frozen estimator at τ = 0.90 per speed bin.
- Only the 12.0 shared speed range; each athlete only on its supported bins; one interior gap ≤ 1 bin bridged (counted); no extrapolation.

## Representations and models
M1 officials · A0 + band accelerations · A1 + P-spline coefficients (λ by repeat-attempt agreement, Combine only) · A2 + fold-fitted FPCA scores (K by ≥ 90 % training variance, ≤ 4; partial curves by conditional projection) · A4 + FPCA of hierarchical curves (position mean + shrunk athlete deviation; parameters from training athletes) · controls: curves shuffled within position, random scores. NFL side: function-on-function regression through a training-fold NFL FPCA (pointwise ridge = sensitivity).

## Evaluation
Per-athlete mean squared error over its own supported bins (equal athlete weight), paired athlete bootstrap; pointwise Δ per speed bin (where transfer sits); positions; hierarchical repeat prediction (attempt 1 → attempt 2); FPC test–retest. Elastic analysis (exploratory): landmark registration (4 / 6 / 8 yd/s crossing times) and SRSF dynamic-programming alignment of speed–time curves; phase and amplitude ICC, stopwatch redundancy, stopwatch-twin decomposition. Phase is reported, never registered away.

<!-- M12:RESULTS:START -->
_Generated 2026-10-15T01:20:29Z from executed artifacts (run mode full)._

## 12A — conclusion (inconclusive; run mode full; generated 2026-10-15T01:20:29Z)

### Research question
Q1 — Does the whole Combine acceleration–speed function retain transferable information lost by scalar features?

### Declared primary hypothesis
Combine functional scores (FPCA, fold-fitted) lower held-out integrated squared error of the NFL τ = 0.90 acceleration–speed function vs scalar band features + M1.

### Primary outcome / comparison
athlete-mean integrated squared error over each athlete's supported shared speed bins — A2 (M1 + FPCA scores) vs A0 (M1 + scalar bands) (evaluation target: `cross_domain_physical_transfer`)

### Sample
310 linked athletes with the shared speed range in both domains; 586 forty attempts; 12 shared bins.

### Main evidence (from persisted tables)
- fof A2 vs A0: Δ MSE -0.0002 [-0.0007, 0.0003] (n=310) — inconclusive
- fof A0 vs M1: Δ MSE -0.0001 [-0.0007, 0.0005] (n=310) — inconclusive
- fof A1 vs M1: Δ MSE 0.0002 [-0.0007, 0.0010] (n=310) — inconclusive
- fof A2 vs M1: Δ MSE -0.0003 [-0.0007, 0.0002] (n=310) — inconclusive
- fof A4 vs M1: Δ MSE -0.0005 [-0.0009, -0.0002] (n=310) — reversed
- fof A2 vs A1: Δ MSE -0.0004 [-0.0012, 0.0003] (n=310) — inconclusive
- fof CTRL_shuffled vs M1: Δ MSE -0.0001 [-0.0010, 0.0009] (n=310) — inconclusive
- fof CTRL_random vs M1: Δ MSE -0.0007 [-0.0012, -0.0003] (n=310) — reversed
- pointwise A2 vs A0: Δ MSE -0.0004 [-0.0010, 0.0002] (n=310) — inconclusive
- pointwise A0 vs M1: Δ MSE -0.0002 [-0.0009, 0.0003] (n=310) — inconclusive
- pointwise A2 vs M1: Δ MSE -0.0006 [-0.0012, -0.0001] (n=310) — reversed
- FPC1 attempt test–retest 0.763 [0.643, 0.858]
- FPC2 attempt test–retest 0.569 [0.438, 0.701]
- FPC3 attempt test–retest -0.011 [-0.379, 0.309]
- elastic amplitude_distance: ICC 0.149 [0.000, 0.955]
- elastic phase_distance: ICC 0.350 [0.186, 0.497]
- elastic t_to_4: ICC 0.599 [0.383, 0.756]
- elastic t_to_6: ICC 0.712 [0.600, 0.798]
- elastic t_to_8: ICC 0.917 [0.887, 0.939]
- elastic warp_mid_shift_s: ICC 0.361 [0.086, 0.594]

### Alternative explanations
- The a(v) shape of a standing-start sprint and of game acceleration differ by task; a null functional transfer may reflect task, not athlete.
- FPCA scores largely re-express the band accelerations: similar performance of A0 and A2 is expected if information is low-dimensional.

### Practical interpretation
Shows whether the whole Combine acceleration profile — not just its summaries — carries information about how the athlete accelerates in games, and at which speeds.

### Limitations
- 10 Hz sampling leaves few frames per 0.5 yd/s bin at low speed.
- The NFL function is an upper quantile of observed game movement.
- One cohort.

### Next research questions
- If transfer concentrates in a speed band, test that band alone on an independent cohort.

### Comparisons (identical athletes)

| mode | model | reference | n_athletes | delta_mse | delta_lo | delta_hi | verdict |
|---|---|---|---|---|---|---|---|
| fof | A2 | A0 | 310 | -0.000 | -0.001 | 0.000 | inconclusive |
| fof | A0 | M1 | 310 | -0.000 | -0.001 | 0.001 | inconclusive |
| fof | A1 | M1 | 310 | 0.000 | -0.001 | 0.001 | inconclusive |
| fof | A2 | M1 | 310 | -0.000 | -0.001 | 0.000 | inconclusive |
| fof | A4 | M1 | 310 | -0.001 | -0.001 | -0.000 | reversed |
| fof | A2 | A1 | 310 | -0.000 | -0.001 | 0.000 | inconclusive |
| fof | CTRL_shuffled | M1 | 310 | -0.000 | -0.001 | 0.001 | inconclusive |
| fof | CTRL_random | M1 | 310 | -0.001 | -0.001 | -0.000 | reversed |
| pointwise | A2 | A0 | 310 | -0.000 | -0.001 | 0.000 | inconclusive |
| pointwise | A0 | M1 | 310 | -0.000 | -0.001 | 0.000 | inconclusive |
| pointwise | A2 | M1 | 310 | -0.001 | -0.001 | -0.000 | reversed |
<!-- M12:RESULTS:END -->
