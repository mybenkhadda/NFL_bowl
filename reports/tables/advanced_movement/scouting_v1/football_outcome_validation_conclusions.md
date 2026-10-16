### 08A.2 — outcome semantics and opportunity validation

Gate 08A.2: **PENDING** — no decision recorded yet

| Outcome | Definition status | Basis |
|---|---|---|
| `FO01.separation_at_pass_forward` | plausible_inferred | inferred from the column name and internal consistency; no vendor definition recorded |
| `FO12.tackle` | plausible_inferred | inferred from the column name and internal consistency; no vendor definition recorded |
| `FO13.sack` | plausible_inferred | inferred from the column name and internal consistency; no vendor definition recorded |

| Denominator | Legitimacy | Linked (all models) | Odd / even reliability [95 %] | Random halves | ρ exposure | Status | Reason |
|---|---|---|---|---|---|---|---|
| `FO01.S0` | recorded | 91 | 0.74 [0.52, 0.87] | 0.65 | 0.09 | provisional | unmet: vendor definition not confirmed |
| `FO01.S1` | recorded | 91 | 0.74 [0.50, 0.87] | 0.65 | 0.09 | provisional | unmet: vendor definition not confirmed |
| `FO01.S2` | recorded | 90 | 0.52 [0.11, 0.76] | 0.43 | 0.05 | provisional | unmet: vendor definition not confirmed; reliability below 0.70 / lower bound below 0.50 |
| `FO01.S0T` | selected | 68 | 0.68 [0.45, 0.83] | 0.74 | 0.10 | sensitivity_only | selected denominator: the quarterback chooses the target: the plays are selected on the outcome |
| `FO01.S0N` | selected | 90 | 0.62 [0.30, 0.79] | 0.57 | -0.00 | sensitivity_only | selected denominator: complement of a selection on the outcome |
| `FO12.T0` | credited_only | 93 | 0.61 [0.41, 0.73] | 0.55 | 0.18 | sensitivity_only | credited_only denominator: the rows exist only when a tackle statistic was credited: non-successes are not observed |
| `FO12.T1` | structural | 141 | 0.32 [-0.13, 0.62] | 0.36 | 0.14 | rejected | game-split reliability is not assessable, below the floor, or its lower bound is not positive |
| `FO12.T2` | structural | 141 | 0.59 [0.36, 0.75] | 0.55 | 0.03 | provisional | unmet: vendor definition not confirmed; denominator is structural; reliability below 0.70 / lower bound below 0.50 |
| `FO13.K0` | recorded | 70 | 0.64 [0.44, 0.78] | 0.54 | 0.60 | provisional | unmet: vendor definition not confirmed; reliability below 0.70 / lower bound below 0.50; |ρ(value, log exposure)| above 0.30 |
| `FO13.K1` | inferred | 70 | 0.63 [0.40, 0.78] | 0.53 | 0.61 | sensitivity_only | inferred denominator: NO_ASSIGNMENT is read as 'rushed the passer': an inference about the assignment |
| `FO13.K2` | structural | 73 | 0.58 [0.35, 0.74] | 0.59 | 0.61 | provisional | unmet: vendor definition not confirmed; denominator is structural; reliability below 0.70 / lower bound below 0.50; |ρ(value, log exposure)| above 0.30 |

Rule-based proposal (a recommendation, not a freeze): **PROVISIONAL** · primary `FO01.S0` · secondary ['FO13.K0', 'FO12.T2'] — first provisional denominator in the pre-declared order ['FO01.S0', 'FO01.S1', 'FO01.S2', 'FO13.K0', 'FO12.T1', 'FO12.T2', 'FO13.K2']; one denominator per other outcome.

Reference lines (heuristics declared before running): ≥ 20 rows in ≥ 4 games per athlete; ≥ 60 linked athletes supporting every model family; odd / even Spearman–Brown ≥ 0.70 with a lower bound ≥ 0.50; |ρ(value, log exposure)| ≤ 0.30.

Limits: no vendor data dictionary in the repository (a definition is confirmed only by evidence the analyst records); the delivered tracking holds the project athletes only, so nearest-defender distance cannot be recomputed; assignment, opponent and scheme are unobserved; no Combine measurement was read and no outcome was related to one.