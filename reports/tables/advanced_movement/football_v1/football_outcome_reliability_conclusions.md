### 08B.1 — outcome reliability and context

| Target | Gate B | Position | Odd / even reliability [95 %] | Random halves | n |
|---|---|---|---|---|---|
| `FO01.S0` | PASS | ALL | 0.74 [0.52, 0.87] | 0.65 | 100 |
| `FO01.S0` | PASS | WR | 0.78 [0.45, 0.90] | 0.70 | 74 |
| `FO01.S0` | PASS | TE | 0.62 [0.38, 0.85] | 0.51 | 26 |
| `FO12.T2` | WEAK | ALL | 0.59 [0.36, 0.75] | 0.55 | 173 |
| `FO12.T2` | WEAK | DB | 0.70 [0.43, 0.86] | 0.65 | 81 |
| `FO12.T2` | WEAK | DL | 0.43 [0.07, 0.70] | 0.41 | 92 |
| `FO13.K0` | FAIL | ALL | 0.64 [0.44, 0.78] | 0.56 | 89 |

| Definition | Context | Reliability | Rank agreement with raw | Row variance removed | Estimand |
|---|---|---|---|---|---|
| `FO01.S0` | raw (no conditioning) | 0.74 | 1.00 | 0.00 | the approved estimand |
| `FO01.S0` | route | 0.53 | 0.92 | 0.12 | removes route mix — partly a property of the player and of his NFL usage |
| `FO01.S0` | route × alignment | 0.51 | 0.90 | 0.12 | also removes the NFL alignment the team chose |
| `FO01.S0` | target status | 0.74 | 0.98 | 0.02 | conditions on the quarterback's choice, which depends on the outcome: SENSITIVITY ONLY |

- `FO12.T2` fails **player_level_stability**: odd / even Spearman–Brown 0.59 (lower bound 0.362); reference 0.7, floor 0.5
- `FO13.K0` fails **player_level_stability**: odd / even Spearman–Brown 0.642 (lower bound 0.437); reference 0.7, floor 0.5
- `FO13.K0` fails **exposure_independence**: ρ(value, log observations) = 0.601 (|ρ| ≤ 0.3)

Reference lines (heuristics declared in 08A.2): reliability ≥ 0.70 with a lower bound ≥ 0.50; ≥ 60 linked athletes. Reliability says athletes differ stably in the RECORDED outcome; it does not say the outcome is a skill of the athlete alone. Position-level rows (about 24 linked TE) are descriptive, not independent evidence.