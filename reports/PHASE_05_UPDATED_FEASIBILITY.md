# Phase 05 · Updated Feasibility (after P0)

*Written 2026-10-11. Supersedes §1 of `PHASE_05_FRONTIER_RESEARCH_PLAN.md` and the `verdict` column of
`P0_stream_feasibility.csv`. Evidence and row numbers (E1–E17) refer to the evidence table in
`PHASE_05_P0_RESULTS_REVIEW.md` §10. Nothing was executed for this document. Markers: **[O]** observed,
**[C]** calculated by hand, **[A]** assumption.*

---

## 1 · Reviewed verdicts

| Stream | Pre-P0 | P0 screening | **Reviewed verdict** | Change |
|---|---|---|---|---|
| A · Capability model | feasible, primary | FEASIBLE | **SUPPORTED on the NFL side; Combine side unmeasured** | narrowed to DB, DL, OL, WR |
| B · Counterfactual engine | single-agent only | LIMITED | **LIMITED: single-athlete reachability only** | unchanged |
| C · Offline RL | not identifiable | LIMITED | **NOT PURSUED as RL**; route / coverage labels kept as context | downgraded from the screening |
| D · Multi-agent | co-tracked pairs at most | LIMITED | **PAIRWISE ONLY, promising but unverified** (OL × DL, WR × DB) | upgraded in substance |
| E · Decision metrics | feasible | FEASIBLE | **PARTLY FEASIBLE**: three solid outcomes, the rest thin or not attributable | downgraded from the screening |

---

## 2 · Stream by stream

### A · Capability model (Combine-conditioned NFL movement envelope)

Evidence:
- 303 athletes with ≥ 100 REG plays; DB 73, DL 77, OL 69, WR 63, TE 21. [O, E4]
- The verdict does not depend on the play cut between 25 and 400 plays. [O, E5]
- Own motion is fully observed at 10 Hz, about 72 frames per player-play. [C]

Limitations:
- **Sample on the Combine side is unknown.** P0 counted NFL exposure only. P0b measures the intersection.
- **Power.** Per position, only r ≈ 0.3 or larger is detectable; pooled, about 0.16; an incremental gain over official results needs more. [C]
- **Cohort confounding.** Exposure (1 to 3 seasons), co-tracking density and Combine noise all change with cohort. [O for exposure by season, A for the consequence]
- **A realized envelope is opportunity, not capacity.** Athletes rarely move at their maximum in games, and role decides which movements occur. [A]
- **WR / TE plays may be route plays only.** [E15]
- **TE has too few athletes** for its own model.
- **Phase 04 gates are open**, so the Combine half cannot be run yet.

Decision: stream A stays the primary line. Scope: DB, DL, OL, WR; TE reported descriptively. Role is defined from `lined_up_position`.

### B · Counterfactual movement engine

Evidence: no football in any season [E1]; opponents only when they are cohort athletes [E2, E3].

What remains possible: a single-athlete statement of the form "from this state, with this capability profile, this region was reachable within t seconds". What is not possible: any statement about what would have happened to separation, pressure or the play.

Decision: unchanged. Reachability only, labelled as feasibility, after the movement model passes its calibration gate (E05-05). Not on the critical path of the submission.

### C · Offline reinforcement learning

The screening passed four of five criteria. The review rejects that reading for three reasons:
1. **The action is not the athlete's.** A route or a coverage assignment is set by the play call. Evaluating a "policy" over routes would evaluate the coordinator, confounded by down, distance, coverage and personnel. [A, football knowledge]
2. **The reward is not attributable.** EPA is a team outcome. Separation depends on the defender and the quarterback, who are usually not observed. [E1, E2]
3. **The outcome criterion passed on `cushion`**, a pre-snap state. [E14]

The only continuous action the athlete controls is his own movement, and its consequences cannot be scored without the ball and the opponents.

Decision: no offline RL, no off-policy evaluation, no "recommended action" in the submission. The labels are still valuable: `route_ran` (WR 95.7 %, TE 92.2 %) and `coverage_assignment` (DB 59.2 %) become **conditioning context** for movement and outcome analysis [E13]. Experiment E05-07 is closed as "not identifiable" with this evidence.

### D · Multi-agent and matchups

Evidence:
- Both sides tracked on 58.9 % of plays, 83.0 % in 2025. [O, E3]
- OL × DL: 28,760 co-tracked plays, 994 pairs with ≥ 20 plays. WR × DB: 15,710 plays, 551 such pairs. [O, E6, E7]

Limitations:
- Median 2 tracked athletes per play, so this is pairwise at best, never team-level. [E2]
- Co-tracking is availability. Which co-tracked athletes face each other is unknown.
- Pairs share few plays (median 22 and 12), so named-pair estimates are noise.
- The volume leans on 2025. [A until P0b]

Why it is still interesting: on these plays **both** athletes have Combine tracking. A relation between a capability difference and a pairwise outcome (separation, pressure allowed) would be a direct Combine-to-NFL statement about matchups, which is rare.

How a matchup could be established without inventing assignments [A, to be tested in S7]: compare the tracked distance between a co-tracked WR and DB at the pass-forward frame with the vendor `separation_at_pass_forward`. If they agree within a tolerance, that DB was the nearest defender on that play. This rests on the vendor field meaning "distance to the nearest defender", which is not documented in the repo.

Decision: stretch stage S7, after the primary experiment. Not the backup, because its identification step is unproven.

### E · Decision metrics

Solid [O]: DL get-off (93 athletes), WR separation (77), OL pressure allowed and peak pressure probability (80 and 79). Adequate volume but few athletes: TE separation (27). Thin: time to pressure, yards after catch, tackles. Not attributable: EPA and win probability.

Decision: metrics are built only on the solid outcomes and on tracking-derived quantities whose reliability has been measured (S2).

---

## 3 · Which practical NFL metrics can genuinely be constructed

"Constructible" means the inputs exist. None is validated: reliability and incremental value are still to be shown.

| Candidate metric (brief §8) | Inputs observed? | Verdict | What it needs first |
|---|---|---|---|
| Capability Utilization Score | own tracking, context | **constructible**, single athlete | S2 envelope with measured reliability |
| Athletic Transfer Stability | Combine features, NFL envelope | **constructible**; this is the primary experiment | Phase 04 gates, S2, S3 |
| Movement State Deviation | own tracking over time | **constructible** for historical replay | S2 envelope; never read as fatigue or injury |
| Role-Capability Fit | `lined_up_position` (99.1 %), envelope | **constructible**, descriptive | S2; role taxonomy from alignment |
| Recovery Efficiency | braking / direction-change events | **blocked** | validated event detector (Phase 04 C1, C2) |
| Matchup Exploitability Index | co-tracked pairs | **exploratory only**, position-pair or profile level | S7 identification; not per named pair |
| Movement Opportunity Value | outcome of an alternative movement | **not constructible** | ball and opponents, which are absent |
| Counterfactual Decision Regret | preferred action and its value | **not constructible** | same; a reachability gap can be reported, never "regret" |

Vendor-based metrics:

| Metric | Position | Verdict | Main limitation |
|---|---|---|---|
| Get-off (`player_get_off`) | DL | **usable** | definition not documented; pass-rush plays; mixed DL roles |
| Separation at pass forward | WR (TE with caution) | **usable with route adjustment** | defender and quarterback unobserved |
| Pressure allowed, peak pressure probability | OL | **usable**, new since P0 | depends on the rusher; 8.4 % base rate for the binary |
| Cushion | DB, WR | **context, not an outcome** | pre-snap alignment |
| Time to pressure (and allowed) | DL, OL | **do not headline** | exists only when pressure occurs |
| Yards after catch vs expected | WR | **thin** | 46 athletes with ≥ 20 |
| EPA, win probability | all | **context only** | team outcome |

Registry consequences (applied in `PHASE_05_METRIC_REGISTRY.md` at the next edit, not yet done): M06 becomes secondary; an OL pressure metric is added; M09 keeps its "feasibility, not outcome" label; M12 is retired with stream C.

---

## 4 · Rule changes recorded for the screening

The P0 outputs are kept as executed. These corrections apply to any re-run and are logged in the experiment registry.

| Rule | Was | Should be | Reason |
|---|---|---|---|
| D · best pair | largest play count over all pairs | only pairs in `SampleFrameConfig.relevant_pairs` | OL × DB was selected [E8] |
| C · outcome | best of any configured outcome field | an outcome field of the same position as the action; `cushion` excluded | `cushion` is a state [E14] |
| C · verdict | counts only | counts **and** an argument that the athlete controls the action | identification, not volume, decides |
| A · sample | NFL exposure only | intersection with Combine drill, repeat attempts and official result | the usable sample was not measured |
| E · verdict | all configured outcomes pass | report per outcome; `time_to_pressure` marked marginal | 33 athletes against a cut of 30 |
| all · minimum plays / observations | fixed placeholders (100, 20) | taken from the S2 reliability curves | no reliability evidence yet |
