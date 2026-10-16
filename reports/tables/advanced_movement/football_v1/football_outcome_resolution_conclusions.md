### 08B.0 — outcome resolution and registry

Branch: **EXPLORATORY — source semantics inferred; definitions locked**. Strongest admissible claim: exploratory association only — never a confirmed predictive claim.

| Definition | Source class | Role allowed | Linked | Reliability [95 %] | May be called |
|---|---|---|---|---|---|
| `FO01.S0` | PROVISIONAL | primary_or_secondary | 91 | 0.74 [0.52, 0.87] | recorded separation at pass forward, mean over recorded player-plays [yd] — an INTERACTION outcome (receiver, nearest defender, coverage, route, throw timing); NOT pure receiver ability |
| `FO01.S1` | PROVISIONAL | redundant (= `FO01.S0`) | 91 | 0.74 [0.50, 0.87] | same as FO01.S0 restricted to rows with a route label |
| `FO01.S2` | PROVISIONAL | sensitivity_only | 90 | 0.52 [0.11, 0.76] | separation relative to the mean of the recorded route × alignment cell [yd] — removes route mix and NFL role, which are partly properties of the player; context SENSITIVITY |
| `FO01.S0T` | BLOCKED | blocked | 68 | 0.68 [0.45, 0.83] | separation on targeted plays only — plays selected on the outcome by the quarterback; sensitivity |
| `FO01.S0N` | BLOCKED | blocked | 90 | 0.62 [0.30, 0.79] | separation on non-targeted plays only — complement of a selection on the outcome; sensitivity |
| `FO12.T0` | BLOCKED | blocked | 93 | 0.61 [0.41, 0.73] | share of credited-statistic rows that are solo tackles — NOT a tackle rate per opportunity |
| `FO12.T1` | BLOCKED | blocked | 141 | 0.32 [-0.13, 0.62] | credited solo tackles per tracked snap — exposure, NOT tackle success probability |
| `FO12.T2` | PROVISIONAL | secondary_only | 141 | 0.59 [0.36, 0.75] | PARTICIPATION-CREDIT rate: credited tackles or assists per tracked snap — NOT tackle success probability; snaps on the field are exposure, not tackle opportunities |
| `FO13.K0` | PROVISIONAL | secondary_only | 70 | 0.64 [0.44, 0.78] | sack credit per recorded pass-defence row — NOT a pass-rush win rate; that the player rushed the passer is not recorded |
| `FO13.K1` | BLOCKED | blocked | 70 | 0.63 [0.40, 0.78] | sack credit per row without a coverage assignment — NO_ASSIGNMENT is NOT verified pass rushing; sensitivity |
| `FO13.K2` | PROVISIONAL | sensitivity_only | 73 | 0.58 [0.35, 0.74] | sack credit per tracked snap (run plays in the denominator) — NOT a pass-rush win rate; sensitivity |

- FO01.S1 is operationally identical to FO01.S0 on the observed data: retain FO01.S0, classify FO01.S1 as redundant (common athletes 116, same observation counts 116).

Decision **FREEZE** → `frozen_exploratory` (registry `v1`, hash `475f557b0f0a0685`): `FO01.S0` (primary, PROVISIONAL), `FO12.T2` (secondary, PROVISIONAL), `FO13.K0` (secondary, PROVISIONAL).
Rationale: We freeze FO01.S0 as the primary EXPLORATORY football outcome because it has the strongest combination of football relevance, observed athlete coverage and preliminary across-game reliability among the candidates evaluated without consulting Combine-to-NFL predictive correlations. FO01.S0 uses the recorded separation_at_pass_forward field, whose precise provider semantics remain plausible but unconfirmed. FO01.S1 is operationally redundant with S0 in the observed population, while FO01.S2 shows lower preliminary reliability and is retained for sensitivity analysis only. FO12.T2 is retained as a secondary tackle-or-assist credit rate per eligible defensive snap, not as tackling success probability. FO13.K0 is retained as a secondary recorded sack outcome, not as a pass-rush win rate or opportunity-adjusted sack rate. The source definitions and opportunity denominators remain subject to their documented limitations. All subsequent football-transfer analyses are EXPLORATORY, and these outcomes must not be described as independently validated measures of player skill. This selection was made from semantic plausibility, coverage, reliability and physical interpretability rather than favorable Combine prediction results.

A decision never raises a source class. A PROVISIONAL class becomes CONFIRMED only when the analyst records a named, authoritative source and the quoted definition, in a NEW registry version.