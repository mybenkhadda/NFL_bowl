# Phase 05 · P0 Results Review

*Written 2026-10-11 from the P0 tables the user executed (full run, `reports/tables/phase_05/descriptive/`).
Nothing was executed for this review. Three kinds of statement are kept apart:*

- **[O]** observation: a number read from an executed table, with its file;
- **[C]** calculation: arithmetic on those numbers, or a textbook planning formula, done by hand in this review;
- **[A]** assumption or interpretation: not established by the data.

Population of every table except the entity census: regular season (REG), tracked player-plays.
Companion files: `PHASE_05_UPDATED_FEASIBILITY.md` (streams and metrics), `PHASE_05_RESEARCH_DECISION.md`.

---

## 1 · Findings in brief

1. **No football is tracked.** 0 football rows in 22,654,290 tracking rows, every season and season type. [O]
2. **A play is seen through about two athletes, not 3.6.** REG median 2, mean 2.72; the mean rises from 1.55 (2023) to 3.63 (2025) as cohorts accrue. [O]
3. **Both sides of the ball are tracked far more often than assumed.** 58.9 % of REG plays, from 29.5 % (2023) to 83.0 % (2025). [O]
4. **401 athletes have a regular-season play, not 497.** 303 have ≥ 100 plays. Four position groups (DB, DL, OL, WR) keep 63–77 athletes at that cut; TE keeps 21. [O]
5. **Individual vendor outcomes are dense for three roles:** DL get-off, WR/TE separation, OL pressure allowed. Time-to-pressure is thin. [O]
6. **The P0 screening verdicts are partly misleading** for streams C, D and E (§7). The stream A verdict is robust to the exposure cut but ignores the Combine side.
7. **The Phase 04 gates are not satisfied.** The Phase 04 outputs on disk are incomplete (§9).

---

## 2 · Entity census and tracking coverage by season

Source: `P0_entity_census.csv`, `P0_players_per_play.csv`, `P0_side_coverage.csv`.

| Season (REG) | Games | Plays with ≥ 1 tracked athlete | Plays per game [C] | Tracked athletes | Tracked player-plays | Mean tracked per play | Both sides tracked |
|---|---|---|---|---|---|---|---|
| 2023 | 271 | 25,195 | 93 | 122 | 39,146 | 1.55 | 29.5 % |
| 2024 | 272 | 32,638 | 120 | 247 | 86,219 | 2.64 | 56.0 % |
| 2025 | 272 | 34,617 | 127 | 353 | 125,742 | 3.63 | 83.0 % |
| All REG | 815 | 92,450 | 113 | 401 | 251,107 | 2.72 | 58.9 % |

Checks [C]:
- Rows sum to 22,654,290 over all nine season × type cells, the total in executed report 04. REG holds 18,108,111 rows (79.9 %).
- REG plays (92,450) and player-plays (251,107) agree across the census, side-coverage and field-coverage tables.
- All season types together hold 314,944 tracked player-plays, in line with the ≈ 315k NFL trajectories of Phase 03 Study A.
- About 72 frames per tracked player-play (18,108,111 ÷ 251,107), i.e. about 7 s at 10 Hz.

Observations [O]:
- **Football rows: 0** in every cell. Ball position, ball flight and the quarterback's release point are not in the tracking.
- **Tracked athletes per play:** 26.3 % of plays have one athlete, 51.6 % have at most two, 27.9 % have four or more, the maximum is 11. In 2023, 58.1 % of plays have a single athlete.
- **Side coverage:** 40.9 % of plays have athletes on one side only; 188 plays (0.2 %) have no known side.
- **Vendor context:** 99.1 % of tracked REG player-plays have a `player_play` row. All 2,233 that lack one belong to WR (1,601) and TE (632). [C from `P0_field_coverage.csv`]
- **Preseason and postseason** hold 50,792 and 13,045 tracked player-plays. Preseason 2025 alone has 420 tracked athletes, more than any regular season.

Interpretation [A]: the growth by season is cohort accrual. The 2023 cohort is alone in 2023; by 2025 three cohorts play. Season, cohort, exposure and co-tracking density are therefore tied together, and 2025 Combine data also have about a quarter of the positional noise of 2023/24 (Phase 03). Any pooled analysis must stratify by cohort.

---

## 3 · Athlete exposure and effective sample size

Source: `P0_exposure_summary.csv`, `P0_field_coverage.csv`.

| Position group | Athletes with a REG play | Player-plays | p25 / median / p75 plays | ≥ 100 plays | ≥ 200 | ≥ 400 | Smallest detectable r at ≥ 100 [C] |
|---|---|---|---|---|---|---|---|
| DB | 96 | 66,887 | 102 / 437 / 1,127 | 73 | 65 | 51 | 0.32 |
| DL | 98 | 52,591 | 136 / 305 / 712 | 77 | 66 | 42 | 0.31 |
| OL | 91 | 86,543 | 152 / 774 / 1,333 | 69 | 65 | 56 | 0.33 |
| WR | 83 | 36,944 | 151 / 292 / 706 | 63 | 53 | 33 | 0.35 |
| TE | 33 | 8,142 | 36 / 184 / 385 | 21 | 16 | 7 | 0.58 |
| All | 401 | 251,107 | 104 / 360 / 944 | 303 | 265 | 189 | 0.16 |

"Smallest detectable r" is a planning value: the smallest correlation detectable with 80 % power at two-sided α = 0.05 for that many athletes (Fisher z). It is not a result.

What the numbers support:
- **The athlete is the unit, and there are few of them.** Within one position group only a correlation of about 0.3 or larger is detectable. Pooled across groups the figure is about 0.16.
- **An incremental gain is harder to detect than a correlation.** The primary comparison (tracking-derived profile versus official results) is a difference between two models, so its power is lower than the table suggests.
- **TE cannot carry its own model.** 21 athletes at ≥ 100 plays.
- **These are upper bounds.** They count the NFL side only. The usable sample is the intersection with athletes who ran the relevant Combine drill, have an official result, and pass the Phase 04 feature status. P0 did not measure that intersection (§8, gap 1).
- **Exposure is very unequal.** The interquartile range spans roughly a factor of nine (104 to 944 plays). Exposure reflects role, depth chart and cohort, so a play minimum is also a selection on who the athlete is.

Not comparable [C]: `reports/tables/coverage_by_position.csv` (Phase 01) reports medians over all season types and all tracked athletes, so its values differ from the REG-only medians here.

---

## 4 · Offense–defense co-tracking and matchup feasibility

Source: `P0_opponent_pairs.csv`. Position group is the Combine position from `gold.dim_athlete`.

| Offense × defense | Co-tracked plays | Pair-plays | Distinct athlete pairs | Pairs with ≥ 20 plays | Median plays per pair | Athletes (off / def) | Roles interact? |
|---|---|---|---|---|---|---|---|
| OL × DL | 28,760 | 51,539 | 1,845 | 994 | 22 | 89 / 98 | yes |
| WR × DB | 15,710 | 27,555 | 1,666 | 551 | 12 | 82 / 89 | yes |
| TE × DB | 4,158 | 5,971 | 562 | 104 | 6 | 30 / 83 | yes |
| OL × DB | 33,837 | 65,453 | 1,542 | 1,007 | 37 | 90 / 94 | no |
| WR × DL | 12,682 | 19,487 | 1,756 | 278 | 7 | 82 / 97 | no |
| TE × DL | 3,536 | 4,669 | 642 | 48 | 4 | 32 / 92 | partly |

What this supports:
- **Volume exists for pairwise analysis** of OL × DL and WR × DB. This overturns the pre-P0 expectation that opponents are rarely both tracked.
- **Co-tracking is not a matchup.** Two athletes on the same play need not face each other. The table gives an upper bound on matchup plays, and the true share is unknown.
- **Pair-level estimates are thin.** A typical WR × DB pair shares 12 plays. Analysis has to be at the level of position pairs or capability profiles, with uncertainty clustered on both athletes, not at the level of named pairs.
- **The volume is concentrated in 2025** [A, from the side-coverage trend; the per-season pair count is not in P0, see §8 gap 4].

Two small rows show label coarseness [O, interpretation A]:
- One athlete with Combine position DB appears on offense on 23 plays and has 9 route labels. This is consistent with a two-way player; identity not checked.
- 26 DL athletes have `cushion` values and DL shows 10 coverage-assignment levels. This is consistent with edge / off-ball hybrids inside the Combine "DL" group. `lined_up_position` (99.1 % coverage) should define role, not Combine position.

---

## 5 · Action and state coverage; support for counterfactual modelling

Source: `P0_field_coverage.csv`, `P0_action_overlap.csv`.

**State.** Play-level context (down, distance, yard line, clock, win probability, formation, coverage shell) is present on ≥ 95 % of player-plays for every position. `lined_up_position` is present on 99.1 %.

**Action labels.**

| Field | Position | Labelled | Levels | Athletes with ≥ 2 repeated levels | Reading |
|---|---|---|---|---|---|
| `route_ran` | WR | 95.7 % | 12 | 63 of 78 | well spread (normalised entropy 0.88, top route 19 %) |
| `route_ran` | TE | 92.2 % | 12 | 21 of 28 | as WR |
| `coverage_assignment` | DB | 59.2 % | 20 | 66 of 86 | well spread (entropy 0.90) |
| `coverage_assignment` | DL | 55.2 % | 10 | 9 of 89 | one level holds 93 % |
| `blitzing` | DL | 55.1 % | 2 | 19 | TRUE on 94.7 % of labelled plays |
| `blitzing` | DB | 59.1 % | 2 | 15 | TRUE on 2.5 % |
| `blitzing` | OL, TE, WR | 56–96 % | 1 | 0 | constant: non-null but not applicable |

**Individual outcomes.**

| Outcome | Position | Non-null player-plays | Athletes with ≥ 20 | Mean per athlete [C] | Reading |
|---|---|---|---|---|---|
| `player_get_off` | DL | 33,417 (63.5 %) | 93 | 341 | dense |
| `separation_at_pass_forward` | WR | 30,956 (83.8 %) | 77 | 373 | dense |
| `separation_at_pass_forward` | TE | 6,563 (80.6 %) | 27 | 199 | adequate volume, few athletes |
| `pressure_allowed` | OL | 48,698 (56.3 %), 4,104 TRUE | 80 | 547 | dense; pressure rate 8.4 % |
| `peak_pressure_probability_allowed` | OL | 46,493 (53.7 %) | 79 | 522 | dense, continuous |
| `time_to_pressure` | DL | 2,704 (5.1 %) | 33 | 30 | thin; defined only when pressure occurs |
| `time_to_pressure_allowed` | OL | 4,101 (4.7 %) | 60 | 47 | thin; same selection |
| `yards_after_catch` | WR | 4,098 (11.1 %) | 46 | 52 | thin |
| `tackle` | DB / DL | 5,831 / 3,455 | 64 / 50 | 66 / 38 | sparse events |
| `expected_points_added` | all | ≈ 100 % | 376 | — | team outcome, not attributable to one athlete |

**Support for counterfactual modelling.**
- *Own motion:* fully observed at 10 Hz. A single-athlete model of short-horizon movement is supported.
- *Interaction:* the ball is never observed and opponents are observed only when they are cohort athletes. An outcome model for "what would have happened" cannot be built from tracking.
- *Action labels:* routes and coverage assignments vary within athlete, but they are assigned by the play call and the defensive scheme. They are not the athlete's decision. [A, football knowledge]
- *Selection in WR / TE plays:* every WR and TE `player_play` row carries a route label (35,343 and 7,510, identical to their `lined_up_position` counts). Either the vendor labels a route on every play, or the WR / TE subset contains route plays only. OL, DL and DB show pass-specific fields on 55–64 % of plays, so their subsets do include runs. [O for the counts, A for the reading; §8 gap 5]

---

## 6 · P0 results against the pre-P0 assumptions

| Pre-P0 statement (source) | P0 result | Status |
|---|---|---|
| No football trajectory (audit §1, from a 120k-row sample) | 0 of 22,654,290 rows | confirmed exactly |
| ≈ 3.64 tracked athletes per play (report 04, all season types) | REG mean 2.72, median 2; 3.63 only in 2025 | too high for the pooled population |
| "Frequently all on one side of the ball" | one side only on 40.9 % of plays; both sides on 58.9 % | wrong for 2024–25, right for 2023 |
| "Opponents rarely both tracked" (plan, stream D) | OL × DL 28,760 plays; WR × DB 15,710 plays | refuted in volume; matchups still unverified |
| 497 athletes with NFL tracking (report 08, any season type) | 401 with a REG play; 303 with ≥ 100 | sample about 20 % smaller in REG, 39 % smaller at ≥ 100 |
| Position groups DB, DL, OL, TE, WR only | same five; no UNKNOWN group | confirmed |
| H01 sample: DB 99 + WR 93 = 192 (report 08) | at ≥ 100 REG plays: DB 73 + WR 63 = 136, before the Combine intersection | smaller |
| H03 sample: DL 113 (report 08) | 93 DL with ≥ 20 get-off observations; 77 with ≥ 100 plays | smaller |
| Vendor outcomes exist (schema catalog) | confirmed, with very unequal density (§5) | refined |
| OL not named among outcome positions | OL has two dense individual outcomes | new |

---

## 7 · Are the feasibility thresholds arbitrary or misleading?

Source: `P0_stream_feasibility.csv`, `P05_thresholds.csv`. All cuts were labelled *proposed* before the run.

| Stream | P0 verdict | Problem with the rule | Does the verdict survive? |
|---|---|---|---|
| A | FEASIBLE (303 ≥ 150 athletes; 4 ≥ 3 groups) | The 100-play minimum is a placeholder with no reliability evidence. The rule ignores the Combine side entirely. | Robust to the play cut: at 25, 50, 200 and 400 plays there are still ≥ 189 athletes and four groups with ≥ 30. Not established for the Combine intersection. |
| B | LIMITED | `min_share_ball = 0.90` is moot: the observed value is exactly 0. | Yes. |
| C | LIMITED | (1) The "individual outcome" that passed is `cushion` (184 athletes), which is a pre-snap alignment, not an outcome. (2) The action (`route_ran`, WR/TE) and the outcome (pooled WR/TE/DB) are not required to belong to the same position. (3) Label coverage says nothing about who chose the action. | The pass is not meaningful. With the coherent pairing (`route_ran` × `separation_at_pass_forward`, 104 athletes) the counts still pass, but the stream fails on identification, not on counts (see the feasibility update). |
| D | LIMITED | The "best pair" is OL × DB, chosen by play count; those roles do not interact. `min_distinct_pairs = 30` is far below what was observed and hides that pairs share few plays. The pooled 0.589 both-sides share hides a range of 0.29 to 0.83 by season. | Yes for OL × DL and WR × DB, which pass the same cuts by a wide margin. |
| E | FEASIBLE | `time_to_pressure` passes at 33 athletes against 30 on 5 % of DL plays. The 20-observation minimum is not based on reliability. Availability is not validity. | Overstated. Three outcomes are solid, one is marginal. |
| all | — | Wilson intervals on play shares are about ± 0.3 points. Plays are nested in games and athletes, so these intervals mean nothing inferentially (already stated in the table docs). | — |

Decision: the P0 code and outputs stay as executed. The reviewed verdicts in `PHASE_05_UPDATED_FEASIBILITY.md` supersede the screening column, and the rule changes are logged in the experiment registry.

---

## 8 · What P0 cannot answer

| Gap | Why it matters | Closed by |
|---|---|---|
| 1 · Athletes with both NFL exposure and the Combine drill (≥ 2 attempts, official result) | It is the real sample size of the primary experiment | Stage P0b, `P0b_combine_nfl_sample` |
| 2 · Exposure by draft cohort | Cohort confounds exposure, co-tracking and Combine noise | P0b, `P0b_exposure_by_cohort` |
| 3 · Outcome volume per athlete beyond one cut | Sets which outcomes can be reliable | P0b, `P0b_outcome_exposure`, `athlete_outcome_counts` |
| 4 · Role-relevant co-tracking by season | Tells whether pairwise analysis is a 2025-only study | P0b, `P0b_pairs_by_season` |
| 5 · Play type of tracked player-plays | Tells whether WR / TE envelopes describe route plays only | P0b, `P0b_play_type` |
| 6 · Whether co-tracked athletes actually face each other | Needed before any matchup claim | Stage S7 (geometry), not authored |
| 7 · Reliability of any NFL movement quantity | Needed before any Combine → NFL statistic | Stage S2 (E05-01), Phase 04 M2 |

P0b is authored (not run): `src/nflbdb/research/phase05/sample_frame.py`. It reads Combine participation only, no Combine value.

---

## 9 · Phase 04 feature-validation gates

Read from `reports/tables/phase_04/table_index.csv` and the folder listing.

| Study | Expected | Found on disk | Status |
|---|---|---|---|
| Q · quality and eligibility | 5 outputs | `Q_flag_rates`, `Q_flag_agreement`, `trajectory_quality_flags` | **incomplete**: `Q_athlete_eligibility`, `Q_results` missing |
| B · smoothing robustness | agreement, repeatability, variance-share tables and 9 figures | `B_feature_catalog`, datasets `trajectory_features_by_smoothing` (1,322,496 rows) and `events_by_smoothing` (427,901 rows) | **incomplete**: no agreement or repeatability table |
| C1 · event sensitivity | 5 figures and tables | none | not run |
| C2 · annotation | labels and evaluation | none | not run |
| V · feature status registry (**Milestone 1 gate**) | `V_*` | none | **not produced** |
| M2 · athlete ICC and NFL reliability vs plays (**Milestone 2 gate**) | — | not authored | **not started** |

**Verdict: neither gate is satisfied.** No Combine feature has a VALIDATED status, so no Combine → NFL statistic may be computed yet.

Both Q and B stopped part-way through writing their outputs. I cannot see why. [A] Outputs are written in a fixed order, so Q most likely failed while writing `Q_athlete_eligibility`; one candidate cause is a duplicate key when the same drill appears in two strata. This is a guess until I see the traceback.

What exists is usable as context only [O, `Q_flag_agreement.csv`]: Cohen's κ between structural and signal flags is 0.17 (95 % CI 0.13–0.21) for Combine and 0.20 (0.14–0.26) for the NFL sample. The two flag families mostly pick different trajectories.

---

## 10 · Evidence table

Paths are relative to `reports/tables/phase_05/descriptive/` unless stated.

| # | Observed result | Sample size | Data limitation | Proposed decision | Artifact |
|---|---|---|---|---|---|
| 1 | 0 football rows | 22,654,290 rows, 9 season × type cells | none: exact count | Drop every method that needs ball or quarterback geometry | `P0_entity_census.csv` |
| 2 | Median 2, mean 2.72 tracked athletes per play; 26.3 % of plays have one | 92,450 REG plays | cohort-limited by design; rises by season | No team-level or 22-player model | `P0_players_per_play.csv`, `P0_side_coverage.csv`, fig. `P0_01` |
| 3 | Both sides tracked on 58.9 % of plays (29.5 % → 83.0 %) | 54,448 plays | side unknown on 188 plays; strongly season-dependent | Pairwise analysis is possible; stratify by season | `P0_side_coverage.csv` |
| 4 | 401 athletes with a REG play; 303 with ≥ 100; DB 73, DL 77, OL 69, WR 63, TE 21 | 401 athletes, 251,107 player-plays | NFL side only; exposure is a selection; cohort-confounded | Primary experiment on DB, DL, OL, WR; TE descriptive only | `P0_exposure_summary.csv`, `warehouse/gold/phase_05/decision_audit/athlete_exposure.parquet`, fig. `P0_04` |
| 5 | Stream A verdict unchanged for play cuts from 25 to 400 | 189–372 athletes | Combine intersection unknown | Run P0b before fixing any sample | `P0_exposure_summary.csv`, `P0_stream_feasibility.csv` |
| 6 | OL × DL: 28,760 co-tracked plays, 1,845 pairs, 994 with ≥ 20 plays | 89 OL, 98 DL | availability, not matchups; median 22 plays per pair | Stretch stage S7 after the primary; profile-level only | `P0_opponent_pairs.csv`, fig. `P0_02` |
| 7 | WR × DB: 15,710 co-tracked plays, 1,666 pairs, 551 with ≥ 20 plays | 82 WR, 89 DB | as above; median 12 plays per pair | as above | `P0_opponent_pairs.csv` |
| 8 | "Best pair" of the stream D rule is OL × DB | 33,837 plays | roles do not interact | Restrict the rule to role-relevant pairs | `P0_stream_feasibility.csv` |
| 9 | DL get-off: 33,417 observations, 93 athletes with ≥ 20 | 98 DL | vendor definition not documented; pass-rush plays; DL group mixes roles | Backup experiment outcome | `P0_field_coverage.csv`, fig. `P0_03` |
| 10 | WR separation: 30,956 observations, 77 athletes with ≥ 20; TE 6,563, 27 athletes | 83 WR, 33 TE | confounded by route, coverage, quarterback; pass plays only | Secondary outcome, route-adjusted | `P0_field_coverage.csv` |
| 11 | OL pressure allowed: 48,698 observations, 8.4 % TRUE, 80 athletes; peak pressure probability 46,493, 79 athletes | 91 OL | depends on the rusher faced | New candidate metric for OL | `P0_field_coverage.csv` |
| 12 | Time to pressure: DL 2,704 observations, 33 athletes; OL allowed 4,101, 60 athletes | — | defined only when pressure occurs | Do not use as a primary outcome | `P0_field_coverage.csv` |
| 13 | Route labels: WR 95.7 %, 12 levels, 63 athletes with variation; DB coverage assignment 59.2 %, 20 levels, 66 athletes | 78 WR, 86 DB labelled | label is a play call, not the athlete's choice | Use as conditioning context, not as an action for policy evaluation | `P0_action_overlap.csv` |
| 14 | Stream C passed its outcome criterion through `cushion` | 184 athletes | `cushion` is a pre-snap state | Reclassify; stream C not pursued as RL | `P0_stream_feasibility.csv` |
| 15 | Every WR / TE `player_play` row has a route label; all 2,233 tracked player-plays without a `player_play` row are WR / TE | 36,944 WR, 8,142 TE player-plays | play type not in P0 | Check play type in P0b before interpreting WR / TE movement | `P0_field_coverage.csv` |
| 16 | 26 DL athletes have cushion values; one DB athlete plays offense | — | Combine position is a coarse role label | Define role from `lined_up_position` | `P0_field_coverage.csv`, `P0_opponent_pairs.csv` |
| 17 | Phase 04: Q and B incomplete, C1 / C2 / V absent | — | cause of the stops unknown | Gates not satisfied; Combine → NFL statistics stay blocked | `reports/tables/phase_04/table_index.csv` |
