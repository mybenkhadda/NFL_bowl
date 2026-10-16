# Phase 05 · Research Decision (after P0)

*Written 2026-10-11. Based on `PHASE_05_P0_RESULTS_REVIEW.md` (evidence rows E1–E17) and
`PHASE_05_UPDATED_FEASIBILITY.md`. Nothing was executed for this document. Every threshold below is
**proposed** and must be fixed before the corresponding stage is run.*

---

## 1 · Decision

| | Experiment | Status |
|---|---|---|
| **Primary** | **Capability transfer** (E05-01 then E05-02): does a tracking-derived Combine profile predict the realized NFL movement envelope better than official Combine results? Positions DB, DL, OL, WR. | **Supported on the NFL side. Authoring of S2 may start.** The Combine half stays blocked by the Phase 04 gates. |
| **Backup** | **DL get-off transfer** (new, E05-B1): does Combine tracking burst predict the vendor `player_get_off` of defensive linemen better than the 10-yard split? | Supported on the NFL side (93 DL athletes, 33,417 observations). Activated by the triggers in §4. |
| Stretch | Pairwise capability difference (S7): OL × DL and WR × DB co-tracked plays | Only after the primary; needs a matchup-identification step that is unproven. |
| Closed | Offline RL and off-policy evaluation (E05-07) | Not identifiable; see the feasibility update §2-C. |

The review identifies a supported primary experiment, so the condition for starting S2 is met. S2 is NFL-only and computes no Combine-to-NFL statistic.

---

## 2 · Why this primary

| Requirement | Evidence | Row |
|---|---|---|
| The outcome is observable for every athlete | own motion at 10 Hz, about 72 frames per player-play | E2 |
| Enough athletes in several positions | 303 with ≥ 100 plays; DB 73, DL 77, OL 69, WR 63 | E4 |
| Not sensitive to the play minimum | same verdict from 25 to 400 plays | E5 |
| No need for the ball or opponents | none tracked reliably | E1, E2 |
| Direct Combine-to-NFL statement | Combine profile → NFL envelope, per position | — |
| A team can act on it | which Combine movement traits carry into games, by role, beyond the stopwatch | — |

What it does not have yet:
- the sample after intersecting with Combine drills (P0b);
- any reliability evidence for the envelope (S2);
- validated Combine features (Phase 04).

Honest expectation [C]: with about 63–77 athletes per position, only relations of r ≈ 0.3 or more are detectable within a position, and about 0.16 pooled. An incremental gain over official results is harder still. A null result with a confidence interval is a likely outcome and remains a reportable finding.

---

## 3 · Primary experiment, staged

| Stage | Content | Needs Phase 04? | Gate to pass (proposed) |
|---|---|---|---|
| **P0b** · sample frame *(authored, not run)* | Athletes with NFL exposure and the Combine drill; exposure by cohort; outcome volume; relevant pairs by season; play type | no (participation counts only) | **G-0b:** at the play minimum chosen in S2, pooled athletes with drill and official result ≥ 200, and ≥ 50 in a position group for that group to be analysed alone |
| **S2 · E05-01** · realized NFL envelope *(next to author)* | Per athlete: upper-quantile acceleration and deceleration by speed band in snap-anchored windows, by role; split-half reliability by game as a function of plays; the same reliability for get-off, separation and pressure allowed | no (NFL only) | **G2:** split-half reliability, lower confidence bound ≥ 0.75 at the chosen play minimum, in at least three of the four position groups |
| **S3 · E05-02** · transfer | Athlete-grouped cross-validation: (a) position mean, (b) official results, (c) b + tracking-derived profile; cohort as a stratum | **yes, both gates** | **G3:** skill gain (c − b), lower confidence bound > 0 |
| S4 · E05-04 | Envelope utilization against get-off, separation, pressure allowed | yes | direction fixed in advance; association only |

Design rules carried into S2 from the review:
- Role from `lined_up_position`, not Combine position [E16].
- Robust quantiles on interior frames, not raw peaks (Phase 03 A3 finding on filter-edge peaks).
- Cohort and season reported as strata in every table [E3, E4].
- The play minimum is read from the reliability curve, not set in advance.
- WR / TE envelopes are labelled "route plays" if P0b confirms that reading [E15].
- S2 results stay PROVISIONAL until Phase 04 Study B confirms smoothing robustness for NFL features.

---

## 4 · Backup experiment: DL get-off transfer (E05-B1)

| | |
|---|---|
| Question | Among defensive linemen, does a Combine tracking burst profile predict mean NFL `player_get_off` better than the official 10-yard split? |
| Outcome | vendor `player_get_off`, athlete mean on pass-rush plays, adjusted for alignment |
| Sample | 93 DL athletes with ≥ 20 observations; 33,417 observations; about 341 per athlete [E9] |
| Design | same grouped cross-validation and baselines as E05-02, one position |
| Strength | individual outcome defined outside this project; dense; no envelope construction needed |
| Limitations | vendor definition not documented; one position, so r ≈ 0.29 or more [C]; the DL group mixes interior and edge roles [E16]; snap anticipation is not athletic capacity |
| Needs Phase 04 | yes, for the Combine side |

Triggers for switching the headline to the backup:
1. G2 fails: the envelope is not reliable enough at any usable play minimum.
2. G-0b fails for the pooled sample but DL alone keeps ≥ 50 athletes.
3. The envelope turns out to be dominated by role and opportunity (S2 shows almost all variance between roles, almost none between athletes within a role).

The get-off reliability is measured inside S2 either way, so the backup costs no extra stage before S3.

---

## 5 · What is blocked, and by what

| Blocker | Blocks | Unblocked by |
|---|---|---|
| Phase 04 Milestone 1 gate (V registry) not produced; Q and B stopped part-way | S3, S4, backup E05-B1 | user reruns Phase 04 Q, B, C1, V and sends any traceback |
| Phase 04 Milestone 2 (athlete ICC, NFL reliability) not authored | S3, S4, backup | S2 delivers the NFL-reliability half; the Combine ICC half is still to author |
| P0b not run | fixing the sample and play minimum | user runs P0b |
| Matchup identification unproven | stretch S7 | S7 design after the primary |

---

## 6 · Manual execution (user)

```powershell
conda activate nfl
$env:NFL_BDB_DUCKDB_MEMORY = "3GB"

# 1 · tests (the P0 test file changed by one line; the P0b file is new)
python -m pytest tests/test_phase05_p0_audit.py tests/test_phase05_p0b_sample_frame.py tests/test_polars2_compat.py -q

# 2 · sample frame
python -m nflbdb.research.phase05 --only P0B
python -m nflbdb.research.phase05 --only P0B --check

# 3 · Phase 04, to find out why Q and B stopped (send me the traceback if either fails)
python -m nflbdb.research.phase04 --only Q
python -m nflbdb.research.phase04 --only B
```

Send back: the test output; `P0b_combine_nfl_sample.csv`, `P0b_exposure_by_cohort.csv`, `P0b_outcome_exposure.csv`, `P0b_pairs_by_season.csv`, `P0b_play_type.csv` (all in `reports/tables/phase_05/descriptive/`); figures `P0b_01`, `P0b_02`; and the Phase 04 console output.

New outputs when run: the five tables above (CSV + Parquet), `warehouse/gold/phase_05/decision_audit/athlete_outcome_counts.parquet`, and two figures in `reports/figures/phase_05/P0b_sample_frame/`.

---

## 7 · Next authoring step

Stage S2 (`src/nflbdb/research/phase05/envelope.py`), using the design rules of §3. I will fix its configuration after the P0b tables are back, because the play-minimum grid, the role taxonomy and the WR / TE labelling depend on them.
