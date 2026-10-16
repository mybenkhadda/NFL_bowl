### 08A — football outcome feasibility

Gate: **PASS** · branch: **continue** · 16 candidates · blocked_definition: 1 · context_only: 3 · insufficient_coverage: 3 · not_interpretable: 1 · not_movement_outcome: 1 · observable: 4 · observable_conditional: 2 · unsupported_requires_inference: 1

| Candidate | Positions | Status | Athletes ≥ min | Split-half [95 %] | Reason |
|---|---|---|---|---|---|
| `FO01.separation_at_pass_forward` | WR · TE | observable | 102 | 0.74 [0.52, 0.87] | fields present; coverage and split-half support reach the stated counts |
| `FO02.yards_after_catch` | WR · TE | observable_conditional | 59 | 0.68 [0.36, 0.83] | recorded only when the event it describes happened: the athlete mean is taken over a self-selected subset of plays |
| `FO03.cushion` | WR · TE · DB | context_only | 174 | — | describes the situation of a play, not what the athlete achieved; usable as a conditioning label |
| `FO04.route_ran` | WR · TE | context_only | 104 | — | describes the situation of a play, not what the athlete achieved; usable as a conditioning label |
| `FO05.player_get_off` | DL | blocked_definition | 90 | 0.89 [0.83, 0.94] | vendor definition unresolved: the value cannot be given a football meaning |
| `FO06.time_to_pressure` | DL | insufficient_coverage | 33 | 0.59 [0.29, 0.77] | 33 athletes with ≥ 20 observations in ≥ 4 games (needed 50); 1 declared position(s) reach 20 |
| `FO07.time_to_qb_hurry` | DL | insufficient_coverage | 34 | 0.61 [0.28, 0.80] | 34 athletes with ≥ 20 observations in ≥ 4 games (needed 50); 1 declared position(s) reach 20 |
| `FO08.quick_pressure` | DL | insufficient_coverage | 33 | 0.42 [-0.15, 0.70] | 33 athletes with ≥ 20 observations in ≥ 4 games (needed 50); 1 declared position(s) reach 20 |
| `FO09.pressure_allowed` | OL | observable | 71 | 0.46 [0.11, 0.69] | fields present; coverage and split-half support reach the stated counts |
| `FO10.time_to_pressure_allowed` | OL | observable_conditional | 60 | 0.67 [0.42, 0.80] | recorded only when the event it describes happened: the athlete mean is taken over a self-selected subset of plays |
| `FO11.peak_pressure_probability_allowed` | OL | not_interpretable | 71 | 0.76 [0.62, 0.87] | output of an undocumented vendor model |
| `FO12.tackle` | DB · DL | observable | 114 | 0.61 [0.43, 0.73] | fields present; coverage and split-half support reach the stated counts |
| `FO13.sack` | DL | observable | 87 | 0.64 [0.44, 0.78] | fields present; coverage and split-half support reach the stated counts |
| `FO14.separation_allowed` | DB | unsupported_requires_inference | 0 | — | needs a pairing of athletes that no field records; pairing them by proximity would manufacture the label |
| `FO15.coverage_assignment` | DB | context_only | 77 | — | describes the situation of a play, not what the athlete achieved; usable as a conditioning label |
| `FO16.career_snaps` | DB · DL · OL · TE · WR | not_movement_outcome | 0 | — | athlete-level career total: no movement content, exposure censored by draft cohort |

Support counts: ≥ 20 observations in ≥ 4 games per athlete, ≥ 50 athletes (≥ 20 in a declared position), ≥ 30 athletes with both game halves; split-half reference line 0.50. All are heuristics declared before running.

Defensible for a later, separately designed analysis: `FO01.separation_at_pass_forward`, `FO12.tackle`, `FO13.sack`. This is an observability statement: no association with any Combine measure has been looked at.

Limits: vendor definitions are inferred from column names (no data dictionary in the repository); a null is never read as zero; assignment, opponent and scheme are unobserved; the split-half is a feasibility screen on one odd / even-game split, not a frozen reliability.