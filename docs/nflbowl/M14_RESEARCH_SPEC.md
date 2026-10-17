# M14 research specification: cross-sport movement intelligence and the Hidden Braking Advantage

Notebook: `notebooks/nflbowl/M14_CROSS_SPORT_ATHLETIC_INTELLIGENCE_COLAB.ipynb`
Code: `src/nflbowl/lab/m14/`
Output: `/content/drive/MyDrive/NFLBowl/experimental_lab/m14_cross_sport_athletic_intelligence_v1`

## 1. Question and primary criterion

Can reliable, interpretable movement features inspired by other sports reveal hidden athletic capabilities that predict NFL movement performance (`nfl.q90_shared`) beyond M1 (draft year, height, weight, forty, ten)?

**Primary criterion:** held-out Δ RMSE vs M1 on the M13 `strict_common` cohort. The cohort is athlete-disjoint over 5 outer folds, and every model is evaluated on the same athletes.

**Primary family** (Holm-adjusted):
- `M1+M14_reliable` vs M1
- `M1+M14_braking` vs M1
- `M1+M14_reliable+<combo encoder>` vs M1

The combo encoder is F4 by default. It was pre-declared as the M12 primary design; it was not chosen on outcomes.

## 2. Data limits (decide what is blocked)

**Channels at 10 Hz:**
- speed (yd/s)
- a_tan (yd/s², signed)
- a_n (yd/s², signed centripetal)
- yaw (rad/s)
- dt (s)
- heading_valid
- plus a validity mask

**Not in the bundle:**
- x / y positions
- body orientation
- synchronised opponents
- ball events
- stimulus timestamps

**Blocked metrics:**
- lateral recovery efficiency
- pursuit interception efficiency
- separation creation efficiency
- defensive recovery efficiency
- the COD-deficit residual, when the athletes table has no `three_cone` / `short_shuttle` column (checked at run time)

## 3. Event definition (`events.py`)

**Segments.**
- Gap-free runs of valid frames. A frame with `dt > max_gap_dt` (0.15 s) ends its run.
- Time convention: `t[k+1] − t[k] = dt[k]`.
- Events never cross a segment boundary.

**Phases.**

| Phase | Definition |
|---|---|
| minimum m | interior local speed minimum |
| entry e | left prominence peak, bounded by the previous accepted minimum |
| braking onset b | first run of 2 frames with a_tan ≤ −1 yd/s² in [e, m]. Fallbacks: a single frame (`brake_unsustained`), or b = e (`weak_braking`) |
| propulsion p | first run of 2 frames with a_tan ≥ +1 in [m, x] (otherwise `propulsion_censored`) |
| exit x | right prominence peak |
| recovery | first time v ≥ 0.9·v_entry, interpolated (otherwise `recovery_censored`) |
| turn phase | [b, x]: angle = Σ yaw·dt over heading-valid frames; direction when \|angle\| ≥ 20° and coverage ≥ 0.5 |

**Acceptance.**
- v_entry ≥ 3 yd/s.
- drop ≥ max(1, 0.25·v_entry).
- The minimum must be closed by a rise ≥ 1 yd/s. If the segment ends first, the event is kept and flagged `exit_censored`.
- A smaller rise followed by a deeper dip is a double dip. It is merged into the deeper valley.
- `complete` = braking present, propulsion observed and exit not censored.

**Aggregation.**
- event → sequence: median over the sequence's events (`max_controlled_decel`: max over complete events).
- sequence → athlete: median per scope (`forty`, `agility` = three-cone + shuttles, `drill` = position drills).
- Predictor columns are named `m14c_<scope>__<metric>`.
- NFL windows produce `m14n__<metric>` columns. These are outcomes only.

## 4. Metrics (registry = `registry.METRICS`, full formulas there and in `tables/registry/metric_definitions.json`)

| Family | Metrics |
|---|---|
| motorsport | corner_exit_efficiency\*, braking_point_efficiency\*, braking_budget_utilization\*, stop_go_elasticity |
| alpine | turn_speed_retention\*, curvature_speed_envelope, multi_cut_consistency, directional_asymmetry |
| court | brake_propulsion_transition\*, first_step_explosiveness, cut_recovery_time\*, sparc_smoothness, lateral_recovery_efficiency (blocked) |
| rugby_soccer | cod_deficit_residual (train-fit), speed_adjusted_cutting (train-fit), max_controlled_decel\*, cut_recovery_asymmetry |
| sprint | accel_envelope_a0, accel_envelope_v0, accel_reserve_at_vref, time_to_useful_speed, high_speed_accel |
| nfl_context | pursuit / separation / defensive recovery (all blocked) |

\* = braking core (the Hidden Braking Advantage signature).

## 5. Status rules

### Measurement status (`reliability.measurement_status`)

| Status | Rule |
|---|---|
| invalid | coverage < 0.5, or < 30 athletes with ≥ 2 trials, or ICC(1,1) < 0.4 |
| validated | ICC(1,1) ≥ 0.6, ICC lower CI ≥ 0.4, and minimum perturbation Spearman ≥ 0.8 |
| exploratory | otherwise |
| blocked | inputs absent |

### Predictive status (`transfer.transfer_status`), for each model vs its reference

| Status | Rule |
|---|---|
| validated | CI lower > 0, adjusted p ≤ 0.05 (Holm for the primary family, BH otherwise), and better in ≥ 4 of 5 folds |
| exploratory | Δ > 0 and the bootstrap share of Δ ≤ 0 is ≤ 0.2 |
| invalid | CI upper < 0 |
| inconclusive | otherwise |

### Final status per metric (`scouting.evidence_table`)

| Status | Rule |
|---|---|
| invalid | measurement status is invalid |
| validated | measurement validated, AND the single-metric model is validated, AND the metric is novel (OOF R² from M1 + position ≤ 0.5) |
| exploratory | the single-metric or family model is validated / exploratory |
| inconclusive | otherwise |
| blocked | inputs absent |

## 6. Transfer design (`transfer.py`)

**Engine.** The official ladder engine (`exp._ladder_inputs` + `ae.oof_ladder`): same frame, folds, seed, inner-CV α and position dummies. M1, M1+F0 and M1+combo are refitted and must reproduce the official ladder; the run aborts otherwise.

**Selection inside a fold.** Outcome-free and fitted on training athletes only, in this order:
1. train coverage
2. train-athlete ICC
3. train R² from M1 ≤ 0.5
4. greedy redundancy pruning (|ρ_train| < 0.8)

The residual metrics use train-fold OLS.

**Models:**
- M1
- M1+F0
- M1+`<combo>`
- M1+M14_`<family>`
- M1+M14_all
- M1+M14_reliable
- M1+M14_braking
- M1+M14_reliable+`<combo>`
- M1+`<metric>` for each measurement-eligible metric (secondary, BH)

**Cohorts:**
- `strict_common` (claims)
- `m14_observed` = strict ∩ every braking-signature metric observed
- `ladder_all`

**Robustness checks:**
- fold stability
- the secondary cohorts
- an unstratified bootstrap
- selection frequency across folds
- a within-position permutation null (20 permutations of the M14 columns, selection unchanged)

## 7. Hidden Braking Advantage (`scouting.py`)

**Steps:**
1. **Signature.** Braking-core agility metrics that are not invalid; within-position signed z; composite = mean over the available metrics (≥ half).
2. **Twins.** Same position, distance on within-position z of the official tests (forty, ten, weight, three_cone, short_shuttle where coverage ≥ 0.7) ≤ 0.75. Greedy one-to-one matching. No outcome is used.
3. **Contrast.** The higher-composite twin minus the lower one, on `nfl.q90_shared` and the `m14n__` braking metrics. Reported as a pair-bootstrap CI, an exact sign test, and the Spearman of Δcomposite with Δoutcome.
4. **Hidden variation.** Within-twin |Δz| divided by random same-position |Δz|.
5. **Case studies.** The largest Δcomposite among pairs closer than the median distance (no outcome used). Athletes are pseudonymised.

## 8. Safeguards (enforced in code or reports)

- No neural training: there is no config switch, and `resolve` rejects `run_training`.
- `m14n__` and `nfl__` columns never enter a predictor frame (`transfer.analysis_frame` raises).
- No held-out outcome, held-out trial or training outcome enters any feature selection (unit-tested).
- Comparisons are made only on one aligned cohort (`TF.validate_alignment`, `TF.wide`).
- No inference of force, power, reaction time, injury risk or causal smoothness. The caveats are printed in every report.
- `validated` ≠ scouting guidance: replication on independent athletes is always listed as required.
