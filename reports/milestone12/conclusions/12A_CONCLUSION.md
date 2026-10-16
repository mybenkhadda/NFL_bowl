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
