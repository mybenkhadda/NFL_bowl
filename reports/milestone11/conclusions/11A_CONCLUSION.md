## 11A — conclusion (inconclusive; run mode full; generated 2026-10-14T22:40:51Z)

### Research question
Do drill-specific movement features add held-out information on physically comparable NFL movement?

### Hypothesis
For SUPPORTED drill × demand cells, M2 (official + ≤ 3 preselected tracking features) lowers held-out RMSE vs M1.

### Sample
387 linked athletes; 18 cells computed; positions DB, DL, OL, TE, WR.

### Main evidence
- not_tested: 45 cells
- exploratory_statistical_relationship: 11 cells
- no_established_transfer: 6 cells
- established_physical_overlap: 1 cells
- forty → accel_low (nfl.accel_low_mean): Δ RMSE -0.0009 [-0.0023, 0.0006], n=322, controls beaten: False, permutation p 0.657
- pass_rush → accel_low (nfl.accel_low_mean): Δ RMSE -0.0007 [-0.0029, 0.0015], n=184, controls beaten: False, permutation p 0.413
- short_shuttle → braking (nfl.brake_decel): Δ RMSE 0.0001 [-0.0011, 0.0014], n=111, controls beaten: False, permutation p 0.483
- short_shuttle → reaccel_after_brake (nfl.reaccel_mean): Δ RMSE -0.0014 [-0.0039, 0.0006], n=77, controls beaten: False, permutation p 0.811
- three_cone → redirection (nfl.redirect_retention): Δ RMSE -0.0003 [-0.0017, 0.0010], n=96, controls beaten: False, permutation p 0.458

### Alternative explanations
- Drill participation is selected; complete-case cells describe tracked athletes only.
- Official drill times absorb much of a drill’s information; M2 measures what remains.
- Targets mix athlete and role: a gain may reflect position usage rather than capacity.

### Practical scouting relevance
Shows which drills add measurable, held-out information about physically matched game movement — and which only overlap physically.

### Limitations
- Many cells: classes other than the declared primary cells are exploratory.
- Small linked cohorts give wide intervals: inconclusive ≠ no effect (see mde in 11B / 11C).

### Next experiment
- Freeze any supported cell as a confirmatory hypothesis for a later draft class.
- Use 11G to decide which drills to keep tracking.

