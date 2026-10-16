# Phase 05 · Research Results

*Each section is filled only from persisted tables after the user runs the corresponding stage; every number
cites its table. Interpretation lives in the review documents, not here.*

## P0 · Data & decision feasibility audit (E05-00) — EXECUTED by the user, REVIEWED 2026-10-11

Command: `python -m nflbdb.research.phase05 --only P0`. Population: regular season unless stated.
Full review: `PHASE_05_P0_RESULTS_REVIEW.md`; reviewed verdicts: `PHASE_05_UPDATED_FEASIBILITY.md`.

| Question | Table (`reports/tables/phase_05/descriptive/`) | Result |
|---|---|---|
| Football / null-id rows per season | `P0_entity_census.csv` | 0 football rows in 22,654,290 rows (all seasons and season types) |
| Tracked athletes per play | `P0_players_per_play.csv` | median 2, mean 2.72; one athlete on 26.3 % of 92,450 plays; maximum 11 |
| Plays with both sides tracked (M13) | `P0_side_coverage.csv` | 58.9 % overall; 29.5 % (2023), 56.0 % (2024), 83.0 % (2025) |
| Opponent co-tracking by position pair | `P0_opponent_pairs.csv` | OL × DL 28,760 plays, 1,845 pairs; WR × DB 15,710 plays, 1,666 pairs |
| Vendor field coverage by position | `P0_field_coverage.csv` | dense: DL get-off (33,417), WR separation (30,956), OL pressure allowed (48,698); thin: time to pressure (DL 2,704) |
| Athlete exposure | `P0_exposure_summary.csv` | 401 athletes; 303 with ≥ 100 plays (DB 73, DL 77, OL 69, WR 63, TE 21) |
| Action vocabulary & overlap | `P0_action_overlap.csv` | `route_ran` WR 95.7 % labelled, 12 levels; `coverage_assignment` DB 59.2 %, 20 levels |
| Stream feasibility (screening) | `P0_stream_feasibility.csv` | A FEASIBLE; B, C, D LIMITED; E FEASIBLE — superseded for C, D, E by the review |

Figures: `reports/figures/phase_05/P0_feasibility/P0_01` … `P0_05`.

## P0b · Analysis sample frame (E05-00b) — AUTHORED, PENDING EXECUTION

Command: `python -m nflbdb.research.phase05 --only P0B`. Tables `P0b_*` — _pending_.

## S2 · Realized NFL envelope (E05-01) — PLANNED, next to author

NFL only; not blocked by the Phase 04 gate.

## S3 – S8 — BLOCKED

Blocked by the Phase 04 gate (G-P04) and/or earlier Phase 05 gates; see `PHASE_05_EXPERIMENT_REGISTRY.md`.
