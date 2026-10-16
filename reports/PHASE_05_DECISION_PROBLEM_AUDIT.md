# Phase 05 · Decision-Problem Audit

*Authored 2026-10-11 from saved reports and source code only. **Nothing was executed.** Every quantitative
statement below is either quoted from an executed, saved report (marked **[E]**, with its source) or is a
hypothesis to be measured by Stage P0 (marked **[H → P0]**). No number in this file was computed in this session.*

---

## 0 · Compact initial state report

| Status | Item |
|---|---|
| **Exists & executed** | Phase 01 lakehouse (bronze → silver → gold, DuckDB catalog), 27 tests passing (2026-10-08). Phase 03 M1 kinematics; M2 v2 Combine features; Study A, A0, A3; snap-anchored NFL windows (`gold/phase_03/nfl_player_play_windows`). Tables in `reports/tables/phase_03/`. |
| **Coded, NOT executed** | Phase 04 M1 (`research/phase04/`: Q, B, C1, C2 v2, V) — no `reports/tables/phase_04/` output exists. Analytics Lab (Docker). |
| **Validated (scientific gate passed)** | None of the Phase 04 gates. Executed *findings* exist (see §1), but no feature has a VALIDATED status yet. |
| **Hypothetical** | Every Phase 05 innovation (digital twin, counterfactual engine, offline RL, multi-agent, new metrics). |
| **Missing files** | `CLAUDE.md` and `NFL_BDB_2027_IMPLEMENTATION_AND_RESULTS_REPORT.md` do not exist in the repo. |

**Binding gate inherited from Phase 04** (`docs/PHASE_04_PLAN.md`): *"No Combine-to-NFL statistics (M5) before
both gates pass"* (M1 gate = V status registry; M2 gate = athlete ICC + NFL reliability vs number of plays).
Phase 05 respects it: Stage P0 uses **no Combine feature**; every Combine→NFL experiment is gated on it.

---

## 1 · Evidence base (executed, saved)

| # | Fact | Source |
|---|---|---|
| E1 | 510 Combine athletes; 497 have NFL tracking. | **[E]** `reports/08_research_feasibility.md` |
| E2 | `game_tracking` 22,654,290 rows; every sampled row's `nfl_id` joins `players` (510 athletes), match rate 1.0. | **[E]** `reports/04_relationship_analysis.md` |
| E3 | Frame ↔ play join multiplicity ≈ 3.64 tracked players per play. | **[E]** `reports/04_relationship_analysis.md` |
| E4 | `game_tracking.nfl_id` null share 0.0 % in a 120,000-row profiling sample (the schema expected null-id football rows). | **[E]** `reports/tables/schema_catalog.csv` (sample — cannot prove absence) |
| E5 | `player_play`: 64 columns, 314,197 rows; vendor context + outcomes incl. `expected_points_added`, `expected_points`, win-probability fields, `player_get_off`, `time_to_pressure(_allowed)`, `separation_at_pass_forward`, `cushion`, `route_ran`, `coverage_assignment(_at_snap)`, `team_coverage_man_zone/type`, `pressure_allowed`, `peak_pressure_probability_allowed`, `blitzing`, `target`, `rec_yards`, `yards_after_catch`, `expected_yards_after_catch`. | **[E]** `reports/02_schema_dictionary.md`, `schema_catalog.csv` |
| E6 | Matched athletes for the H01–H05 feasibility hypotheses: H01 192 (DB 99, WR 93); H02 65 (agility drill missingness); H03 113 (DL); H04/H05 510. | **[E]** `reports/08_research_feasibility.md` |
| E7 | Snap tag present in ≥ 99.99 % of REG plays; vendor `dir` is compass convention; vendor `a` is a magnitude; Combine `dis` unreliable; 2025 Combine noise ≈ ¼ of 2023/24; forty starts in motion and ends at peak speed. | **[E]** Phase 03 M1, A3, NFL windows |
| E8 | Provisional quality tiers are non-informative (AUC ≈ 0.50); Combine raw peaks sit on SG-boundary frames. | **[E]** Study A3 |

**Decisive implication (to be confirmed exactly by P0):** NFL tracking covers only the Combine-linked rookies
(draft cohorts 2023–2025), **not 22 players**, with **no football trajectory** [H → P0: E4 is a sample].
On a typical play we observe ≈ 3–4 athletes, frequently all on one side of the ball [H → P0].
Opponents, ball flight and teammates outside the cohort are visible **only** through vendor `player_play` fields.

---

## 2 · State / action / reward / context inventory

Legend — **Obs**: observed in tracking (T), vendor field (V), derived (D), not available (✗).
Coverage column is a **P0 measurement target**, never a number here.

### 2.1 State (what the athlete's situation is at decision time)

| Element | Obs | Source | Caveat | P0 measures |
|---|---|---|---|---|
| Own position, speed, accel, heading | T | `silver_game_tracking` x_std, y_std, s, a, dir_std, o_std | 10 Hz; derivatives via SG (Phase 03) | frames per play |
| Own pre-snap alignment | V/T | `lined_up_position`, pre-snap window | | coverage by position |
| Motion at snap | V | `in_motion_at_ball_snap` | | non-null share |
| Teammate / opponent positions | T (partial) | only if also a cohort rookie | ≈ 3.6 per play **[E3]** | co-tracking census |
| Ball position / flight | ✗ | — | no ball rows **[H → P0]** | football row count |
| Opponent proximity summaries | V | `cushion`, `separation_at_pass_forward` | vendor-derived; definition opaque | coverage |
| Game state | V | down, yards_to_go, yardline, quarter, clock, score, pre-snap WP | | coverage |
| Formation / coverage shell | V | `offense_formation`, `receiver_alignment`, `team_coverage_man_zone/type` | team-level labels | coverage |

### 2.2 Action (what the athlete or coach chose)

| Element | Obs | Source | Granularity | Caveat |
|---|---|---|---|---|
| Continuous movement | T | trajectory | 10 Hz | the only *fully observed* action |
| Route | V | `route_ran` | categorical, per play | label, not intent — we never infer *intended* routes |
| Coverage responsibility | V | `coverage_assignment(_at_snap)` | categorical | vendor label; never invented |
| Blitz / rush | V | `blitzing`, `pass_rushers_encountered` | binary / count | |
| Coach play-call | ✗ | — | | not observed → no coaching-policy RL |

### 2.3 Reward / outcome

| Element | Obs | Source | Credit-assignment problem |
|---|---|---|---|
| EPA / EP | V | `expected_points_added`, `expected_points` | team-level; one rookie among 22 |
| Win-prob added | V | `home/visitor_team_win_probability_added` | team-level, noisy |
| Get-off | V | `player_get_off` | individual — best-aligned with movement capacity |
| Time to pressure / pressure allowed | V | `time_to_pressure`, `time_to_pressure_allowed`, `pressure_allowed`, `peak_pressure_probability_allowed` | individual / unit |
| Separation, cushion | V | `separation_at_pass_forward`, `cushion` | individual receiver/DB (vendor) |
| Targets, yards, YAC vs xYAC | V | `target`, `rec_yards`, `yards_after_catch`, `expected_yards_after_catch` | individual + QB/scheme confounded |
| Tackles, sacks, TFL | V | `tackle`, `sack`, `tackle_for_loss` | sparse events |
| Career outcomes | V | `career` (gold `has_career`) | athlete-level, never a predictor |

### 2.4 Context / confounders

Scheme and team (`team_abbr`, `possession_team`), opponent quality (not modelled), snap share / role
(selection into plays), season and draft cohort (E7: cohort noise shift), position switches, game script.
**Not inferred, ever:** fatigue, injury, intent.

---

## 3 · Decision horizons

| | A · Pregame | B · In-game (historical replay only) | C · Postgame | D · Development |
|---|---|---|---|---|
| **Persona** | Scouting / pro-personnel analyst | Coaching analyst replaying past games | Position coach, film staff | Strength & performance staff, player-development |
| **Decision** | Which rookie traits to expect / plan around; draft & roster evaluation | Retrospective: was a situation within the athlete's physical envelope? | Which plays showed under- or over-use of capacity | Which capacity (accel, decel, COD, top speed) limits NFL realization |
| **Information** | Combine tracking + results; prior NFL envelope | Tracking replay + vendor context of a finished game | Game tracking + vendor outcomes | Combine profile + season envelope trend |
| **Metric** | Predicted envelope ± interval (M02) | Envelope utilization (M04) per situation | Utilization by context; get-off / separation (M05–M07) | Transfer gap (M03), reliability-qualified trend |
| **Frequency** | Annual (draft) / weekly | Post-hoc only | Weekly | Season / off-season |
| **Data availability** | Good: 497 athletes **[E1]** | Single-agent only **[E3]** | Good for vendor outcomes; opponents vendor-only | 3 cohorts, ≤ 3 seasons per athlete |
| **Limitations** | Small n per position; cohort shift | **No real-time claim**; no opponents/ball | Credit assignment; scheme confounding | Short panels; no fatigue/injury inference |
| **Validation** | Athlete-grouped holdout vs official-Combine baseline | Held-out-athlete calibration of envelope; expert face validity | Association with vendor outcomes, cluster CI | Within-athlete reliability (Phase 04 M2 gate) |

**Primary horizon for the Kaggle submission: A (pregame) + D (development)** — they need only the
Combine→NFL link and single-agent NFL tracking, which the data supports. B and C are secondary use cases
for the dashboard.

---

## 4 · Open questions P0 must answer (decides stream feasibility)

1. Exact football-row count and null-`nfl_id` count per season (E4 is only a sample).
2. Distribution of tracked athletes per play; share of plays with ≥ 1 offensive **and** ≥ 1 defensive tracked athlete.
3. Opponent co-tracking matrix: plays on which rookie position *a* and opposing rookie position *b* are both tracked (WR × DB, OL × DL, …) and number of distinct athlete pairs.
4. Non-null coverage of every vendor state/action/outcome field by position group.
5. Per-athlete exposure (tracked plays, plays with each outcome) and athletes above exposure thresholds.
6. Action-vocabulary size and per-athlete action coverage (route / coverage labels) — the overlap condition for any offline policy evaluation.

The answers are written to `reports/tables/phase_05/` and summarised in `reports/PHASE_05_RESEARCH_RESULTS.md`
**after the user runs P0**.
