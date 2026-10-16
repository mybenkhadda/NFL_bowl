# Methods and limitations

**NFL Big Data Bowl 2027 — football performance transfer, outcome validation and scouting decision intelligence**

> Numbers are tokens `‹Cnn: …›` read from persisted tables by notebook `08b6_final_research_evidence_package`. A *pending* token has no executed artifact yet.

## 1 · Data

- Combine tracking at 10 Hz: forty-yard dash attempts of the tracked athletes; athletes with a repeat attempt support every repeatability estimate (375 for the scalar features of the atlas).
- NFL tracking and vendor player-play fields for the **project athletes only**. Other players on the field are not tracked. Consequences: distance to the nearest defender cannot be recomputed, and who rushed the passer is not observed.
- No vendor data dictionary is in the repository. A field definition is *confirmed* only when the analyst records a named source and the quoted text; otherwise it is *inferred* and labelled so everywhere.

## 2 · Five objectives, kept apart

| Objective | Question | Notebook |
|---|---|---|
| O1 semantic credibility | what does an outcome mean; which opportunity denominator is justified | 08B.0 |
| O2 measurement validity | are athlete-level differences stable across games | 08B.1 |
| O3 prospective transfer | does tracking improve held-out prediction beyond official tests | 08B.2, 08B.3 |
| O4 additional information | what stays repeatable after official tests are accounted for | 08B.4 |
| O5 scouting decision value | what can be compared responsibly, and when to abstain | 08B.5 |

## 3 · Outcome definitions and the registry

Each opportunity definition gets a source class — `CONFIRMED`, `DERIVED_VERIFIED`, `PROVISIONAL`, `BLOCKED` — from recorded evidence and internal-consistency checks. The analyst then records `FREEZE`, `PROVISIONAL` or `STOP` in an immutable, versioned, hashed registry. A decision never raises a source class. The registry yields one of three branches: *frozen* (transfer runs), *exploratory* (transfer runs, every result labelled exploratory) or *blocked* (no football transfer; the descriptive analyses continue).

Current decision: ‹C12: FREEZE → frozen_exploratory: FO01.S0 (primary, PROVISIONAL), FO12.T2 (secondary, PROVISIONAL), FO13.K0 (secondary, PROVISIONAL)›.

Rules applied: a null is never read as zero in a recorded denominator; target status is never a filter or a predictor (the quarterback throws to the open receiver); tackles or assists per snap are a participation-credit rate; `NO_ASSIGNMENT` is not verified pass rushing; a definition operationally identical to another is redundant, not a second piece of evidence.

## 4 · Reliability

Reliability is estimated between **whole games**: chronological odd / even games with an athlete bootstrap, random game halves, first-N versus later games, and reliability against the observations required per half. Plays of one game never sit in both halves. Screened value for recorded separation: ‹C11: odd / even game reliability 0.74 [0.52, 0.87]; semantic status plausible_inferred; screen status provisional›. Gate B on the approved definition: ‹C13: FO01.S0: odd / even Spearman–Brown 0.742 (lower bound 0.516); reference 0.7, floor 0.5 — met›.

Context conditioning (route, alignment, target status) uses parameters fitted on other athletes. It changes the estimand — it removes route mix and NFL role, which are partly properties of the player — so it is a sensitivity, never a replacement of the approved outcome.

## 5 · Transfer benchmark

- **Unit:** the athlete. All Combine attempts and all NFL games and seasons of an athlete are in one fold.
- **Folds and representations:** those of Milestone 06. A fold's FPCA and TCN models were fitted without that fold's athletes, which remains true for any subset of the cohort; no encoder is trained on an NFL outcome.
- **Models:** M0 position, cohort, body · M1 + official tests · M2 + engineered speed development · M3 + tracked speed · M4 + FPCA · M5 + TCN-B · M6 + untrained encoder. Ridge, five penalties, inner athlete-grouped tuning, all preprocessing from training folds.
- **Primary comparison:** M2 versus M1, recorded in the registry before any football outcome was related to a Combine value. Secondary: M5 versus M1. Holm adjustment over the two.
- **Uncertainty:** paired athlete bootstrap of Δ RMSE; a permutation test in which the tracking rows are shuffled between athletes within position and both models are **refitted** in every fold. Out-of-fold predictions are never shuffled. Runs are checkpointed and resumable with per-iteration seeds.
- **Label kinds:** an *observed* label is the athlete's own recorded outcome. A *cross-fitted* label subtracts context means estimated on the training athletes of the fold.

Results: ‹C14: M2 versus M1 on FO01.S0: Δ RMSE -0.0007 [-0.0093, +0.0082], linkage permutation p 0.345 — not established (interval includes 0) (EXPLORATORY — source semantics inferred; definitions locked)› · ‹C15: M5 versus M1 on FO01.S0: Δ RMSE -0.0108 [-0.0244, -0.0001], linkage permutation p 0.886 — the tracking model is worse (interval below 0) (EXPLORATORY — source semantics inferred; definitions locked)›.

## 6 · Detectability

Synthetic targets are built on the real cohort — athletes, positions, folds, collinearity of each tracking block with official testing, residual noise — with a planted share of the residual variance. The same pipeline and criterion are applied. This gives the false-positive rate, the power curve, the smallest detectable effect and cohort-size scenarios.

- Milestone 06 design: ‹C10: smallest effect detected with 80 % power — M2: planted share 0.23 of residual variance (Δ RMSE 6.7 %); M5: planted share 0.19 of residual variance (Δ RMSE 6.4 %)›.
- Football cohort: ‹C16: smallest effect detected with 80 % power — M2: 80 % power not reached in the grid (max 0.60); M5: 80 % power not reached in the grid (max 0.29)›.

## 7 · Incremental information

The athlete's expected value from position, forty, 10-yard split, weight and height is fitted on athletes of other folds. Repeatability is then measured on the residual: ICC(1) for scalars, repeat-attempt retrieval AUC on inductive embeddings for FPCA and TCN-B. Status follows an ordered rule on support, raw and residual repeatability. There is no weighted novelty score. Result: ‹C17: repeatable beyond official tests: tracked top speed (p95), speed development 4 → 8 yd/s, FPCA speed-curve scores (K = 2), TCN-B movement fingerprint (8-d). Others: acceleration over 1 s from 4 yd/s (unstable); speed gain over 10 yd from 4 yd/s (weak_residual_signal)›.

## 8 · Scorecard

Position-relative percentiles with intervals from repeat-attempt error and reference sampling; pairwise verdicts from the standard error of the difference; explicit states `supported`, `experimental`, `insufficient_evidence`, `not_observed`; cases selected by pre-draft rules. No table that feeds a pre-draft score may contain an NFL-named column. Gate: ‹C18: every criterion met›.

## 9 · Limitations

1. **Small linked cohort.** About 91 receivers with a supported separation value and every Combine representation, about 24 of them tight ends. Subgroup results are descriptive.
2. **Inferred semantics.** The definition of separation at pass forward is not confirmed by a source. Every football-transfer result is exploratory until it is.
3. **Interaction outcomes.** Separation depends on the defender, the coverage, the route and the throw. It is not a property of the receiver alone.
4. **No opportunity denominators** for tackles and sacks; the sack rate follows playing time.
5. **Selection.** Athletes with NFL snaps are not a random sample of Combine athletes, and playing time depends on ability.
6. **One drill.** The fingerprint and the speed-development measures come from the forty only.
7. **Heuristic reference lines.** Reliability lines, the 2 % practical margin and the status thresholds were declared before running and are not optimised, but they are conventions.
8. **Linkage permutation.** Shuffling the tracking block also breaks its link with the official tests of the same athlete.
9. **Intervals** cover athlete sampling, not model refitting.
10. **Not a decision study.** Nothing here tests whether a scouting or draft decision improves.

## 10 · Reproducibility

Every analysis runs from a Jupyter notebook, in order, with a smoke and a full mode kept in separate directories. Each artifact carries its cohort, definition, fold and configuration hashes, seed and run mode. The reproducibility manifest (`final_reproducibility_manifest.json`) lists package versions, registry hashes, the notebook order and artifact fingerprints. Synthetic checks with planted answers (`football_validation`) cover semantics, reliability, prediction and scouting logic and are run at the top of each notebook.

Unresolved from earlier milestones and unchanged here: the definition of get-off (‹C19: player_get_off: blocked_definition (source semantics unresolved)›), the handedness of the Combine frame, and the units of official measurements.
