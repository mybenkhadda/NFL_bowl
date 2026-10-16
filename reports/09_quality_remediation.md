# 09 — Data-quality remediation

_Pre-execution evidence review, written 2026-10-08 from **saved** artifacts only (`reports/tables/data_quality_results.csv`, `reports/tables/pipeline_run_log.csv`). Nothing in this file was newly computed. Running `python -m nflbdb.research.quality` **overwrites** this file with measured diagnostics, flag counts and a remediation table._

## Saved failures (from `data_quality_results.csv`, run `20261008_003258`, 00:35 UTC)

| Check | Table | Severity | Result | Saved detail | Mode |
|---|---|---|---|---|---|
| Key uniqueness `(event_id, nfl_id, attempt, time)` | combine_tracking | CRITICAL | FAIL | 16 duplicate-key rows of 9,317 | 2% sample |
| Key uniqueness `(event_id, nfl_id, time)` | combine_tracking | MEDIUM | FAIL | 16 duplicate-key rows of 9,317 | 2% sample |
| Referential integrity `(game_id, play_id)` → player_play | game_tracking | HIGH | FAIL | 150 of 104,545 distinct sampled plays have no parent | 2% sample |
| Referential integrity `(game_id, play_id, nfl_id)` → player_play | game_tracking | MEDIUM | FAIL | 1,879 of 237,177 distinct sampled player-plays have no parent | 2% sample |
| Value range `s` ∈ [0, 12.5] | combine_tracking | MEDIUM | WARN | 177 of 9,317 sampled rows; max 25.56 yd/s | 2% sample |
| Value range `a` ∈ [0, 15] | combine_tracking | MEDIUM | WARN | 3 of 9,317 sampled rows; max 23.99 yd/s² | 2% sample |
| Implausible jumps (> 1.5 yd/frame) | game_tracking | MEDIUM | WARN | 4 of 452,714 frames over 6,307 sampled trajectories | 2% sample |

## Finding 1 — the CRITICAL failure comes from an outdated check definition (evidence)

- The CRITICAL row tests the grain `(event_id, nfl_id, attempt, time)` at CRITICAL severity. The current registry (`src/nflbdb/data/sources.py`) defines the grain as `(event_id, nfl_id, attempt, entity_type, time)`. Under that registry, the 4-column key is a MEDIUM alternative.
- `pipeline_run_log.csv` shows the sequence:
  - Run `20261008_002524` (00:25): validate **FAILED** on this check.
  - Run `20261008_002652` (00:27): after the registry fix, validate finished with **0 CRITICAL failures**.
  - Run `20261008_003258` (00:32–00:36): this run reported the **old** check list again (`event_id, nfl_id, time` + CRITICAL 4-column key) and overwrote the saved results.
- Interpretation (inferred, not proven): run `003258` came from a notebook kernel that had imported `nflbdb.data.sources` before the fix. Restarting the kernel and re-running `validate` should restore the current definitions.
- The Phase 01 tracker records the root cause: Combine `BALL` rows carry the athlete's `nfl_id`, so `entity_type` is part of the grain. **Diagnostic D1** re-verifies this exactly. It checks whether every duplicate key differs only by `entity_type`, and whether the 5-column grain has zero duplicates.

## Finding 2 — tracked plays / player-plays without `player_play` rows (unresolved cause)

- The counts above are from a 2% Bernoulli **row** sample, so they are distinct keys seen in the sample, not table totals.
- **D2** lists every tracked play missing from `player_play` (season, week, season type, frames, players). **D3** separates "whole play missing" from "play present, player row missing".
- No explanation is asserted. The module flags these units (`exclude_context`) and keeps the tracking data.

## Finding 3 — Combine speed/acceleration above human thresholds (hypothesis to test)

- The range checks ran over **all entity types**, including `BALL`. That a thrown ball accounts for the extreme values is a **hypothesis**. **D4** splits the outliers by `entity_type` and drill.
- The remediation rules then decide on the measured numbers:
  - all outliers are BALL rows → `expected_behaviour`;
  - any PLAYER rows exceed the thresholds → `tracking_artifact`, with attempt-level exclusion flags.

## Finding 4 — NFL per-frame jumps (sampled)

- The 4 sampled frames are re-counted exactly in `tracking_quality_flags_nfl` (`n_jumps`). Affected player-plays get `exclude_kinematics = TRUE`.

## What the module produces when executed

| Artifact | Content |
|---|---|
| `reports/tables/phase_02/quality_failures.csv` | Triage: category, definition status, sampled vs exact, affected-row estimates |
| `reports/tables/phase_02/quality_remediation_summary.csv` | Status, root cause, evidence, affected athletes, remediation, exclusion flag |
| `reports/tables/phase_02/quality_diag_D1…D4_*.csv` | Exact diagnostic outputs |
| `reports/tables/phase_02/quality_flags_summary.csv` | Flag counts by domain and cohort |
| `reports/figures/phase_02/quality_failure_breakdown.png` / `.svg` | Failures by severity × status; affected rows |
| `warehouse/gold/phase_02/tracking_quality_flags_nfl.parquet` | Flags per player × play |
| `warehouse/gold/phase_02/tracking_quality_flags_combine.parquet` | Flags per attempt |
| `reports/sql/phase_02/quality/D1…D4.sql` | Same diagnostics, for manual DuckDB CLI runs |
