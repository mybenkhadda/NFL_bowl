# Phase 05 · Experiment Registry

*Pre-registration of every Phase 05 experiment. Authored 2026-10-11, **none executed**. Status values:
`AUTHORED` (code exists, not run) · `PLANNED` (no code) · `BLOCKED` (gate not passed) · `RUN` (executed by the
user, not yet reviewed) · `REVIEWED`. Thresholds marked *proposed* may be changed only **before** the stage
is run, with the change logged here.*

## Common protocol

- **Unit of analysis:** athlete (`nfl_id`). Plays are nested; uncertainty = athlete-cluster bootstrap (`studies/stats.py`).
- **Splits:** athlete-grouped K-fold (K = 5), stratified by position group × cohort; fold assignment persisted once in `warehouse/gold/phase_05/research_base/athlete_folds.parquet` (S3) and reused by every later stage.
- **No leakage:** every fitted transform (scaling, imputation, feature selection, hyper-parameters) is fit inside the training fold; NFL outcomes are never used to choose Combine features; `career_*` is never a predictor.
- **Multiplicity:** Benjamini–Hochberg within each stage's pre-registered family.
- **Gate inheritance:** no Combine→NFL statistic before Phase 04 G-P04 (V registry + M2 reliability) passes.
- **Persistence:** CSV + Parquet tables, PNG 300 DPI + SVG figures with plot data, models with config + provenance + reload instructions.

## Registry

| ID | Stage | Question | Population | Design / comparator | Primary metric | Pre-registered decision rule | Status |
|---|---|---|---|---|---|---|---|
| E05-00 | P0 | What decision problems does the data support? | all tracked plays and athletes | census (no inference) | M13, field coverage, exposure | Stream verdict by `StreamFeasibilityConfig` rules (proposed) | **REVIEWED** (2026-10-11; verdicts superseded by `PHASE_05_UPDATED_FEASIBILITY.md`) |
| E05-00b | P0b | How many athletes can a Combine → NFL analysis use? | P0 population + all Combine athletes | census of participation (no Combine value) | athletes with drill ∩ exposure; planning r | G-0b: pooled ≥ 200 and ≥ 50 per analysed position group at the S2 play minimum (proposed) | **AUTHORED** |
| E05-01 | S2 | Is an athlete's realized NFL envelope reliable? (also: get-off, separation, pressure allowed) | athletes of DB, DL, OL, WR, role from `lined_up_position` | split-half by game, Spearman–Brown, as a function of plays | M08 | CI lower ≥ 0.75 at the play minimum read from the curve, in ≥ 3 of 4 position groups | **PLANNED — primary, next to author** (NFL only, not gated by G-P04) |
| E05-B1 | S3 (backup) | Does Combine tracking burst predict DL `player_get_off` beyond the 10-yard split? | DL with ≥ min observations (93 at ≥ 20, P0) | grouped CV; (a) mean, (b) official results, (c) b + tracking | M05, M10 | ΔSkill(c − b) CI lower > 0; headline only if a trigger of the research decision §4 fires | BLOCKED (G-P04) |
| E05-02 | S3 | Does tracked Combine capacity predict the NFL envelope beyond official results? | athletes with VALIDATED features + envelope | grouped CV; (a) position mean, (b) official results, (c) b + tracking capacity | M10 | ΔSkill(c − b) cluster-bootstrap CI lower > 0 | BLOCKED (G-P04, E05-01) |
| E05-03 | S3 | Robustness | as E05-02 | leave-one-cohort-out; per position group; 2025-only noise facet | M10 | sign stable in ≥ 2 of 3 cohorts | BLOCKED |
| E05-04 | S4 | Is utilization associated with individual vendor outcomes? | DL / WR / DB with outcome coverage (P0) | athlete-level Spearman, cluster CI, BH | M04 vs M05–M07 | direction pre-registered (higher utilization ↔ faster get-off / larger separation); CI excludes 0 after BH | BLOCKED |
| E05-05 | S5 | Is a capacity-conditioned motion model calibrated on held-out athletes? | snap windows 0–1.5 s | grouped CV; baseline = constant-velocity extrapolation | M11 | interval coverage within ±5 pp of nominal and better log-score than baseline | PLANNED |
| E05-06 | S5 | Reachable-set counterfactual | E05-05 passed | swap envelope quantile (e.g. p25 ↔ p75 of position) | M09 | reported as feasibility only; no causal claim | PLANNED |
| E05-07 | S6 | Is OPE over vendor action labels identifiable? | plays with action label + outcome | behaviour-policy estimate, overlap, ESS | M12 | ESS / n ≥ proposed min and positivity on ≥ 90 % of contexts, else publish non-identifiability | **CLOSED — not identifiable** (labels are play calls, no ball, reward not attributable; P0 review) |
| E05-08 | S7 | Rookie-vs-rookie matchups | co-tracked WR × DB plays | relative kinematics vs vendor separation | M07, M13 | ≥ proposed min plays and distinct pairs (P0) | PLANNED |
| E05-09 | S8 | Dashboard | persisted tables | use cases A–C | — | every panel traceable to a table | PLANNED |

## Change log

| Date | Experiment | Change | Reason |
|---|---|---|---|
| 2026-10-11 | all | created | Phase 05 start |
| 2026-10-11 | E05-00 | P0 executed by the user and reviewed; screening verdicts for C, D, E superseded | rule flaws: D picked OL × DB by count; C passed on `cushion` (a state); E passed `time_to_pressure` at 33 vs 30 |
| 2026-10-11 | E05-00b | added | P0 counted NFL exposure only; the Combine intersection, cohort split, outcome volume, pairs by season and play type were missing |
| 2026-10-11 | E05-01 | scope set to DB, DL, OL, WR; role from `lined_up_position`; play minimum read from the reliability curve; vendor outcome reliability added; status no longer blocked | TE has 21 athletes at ≥ 100 plays; S2 is NFL only and computes no Combine → NFL statistic |
| 2026-10-11 | E05-B1 | added as backup | DL get-off is dense (33,417 observations, 93 athletes) and defined outside the project |
| 2026-10-11 | E05-07 | closed | not identifiable |
| 2026-10-11 | E05-08 | restricted to OL × DL and WR × DB; needs a matchup-identification step first | co-tracking is availability, not a matchup |
