# Lab marts — data contracts

Built by `python -m nflbdb.lab.build` → `warehouse/gold/phase_lab/` (one Parquet per mart, sidecar
`<name>.schema.json` with column docs + dtypes, `_manifest.json` with fingerprints). The Phase 01
catalog exposes them as `plab_<name>` views; the serving snapshot copies them to `lab.<name>`.

**Refresh strategy.** Incremental: a mart is rebuilt only when its version, input fingerprint
(size + mtime of every source file) or on-disk schema changed; otherwise reused. Rebuilt marts
archive the previous file to `phase_lab/_archive/` (prune manually when large). `kpi_snapshot`
and `availability` are rebuilt every run. The serving snapshot is re-materialised every run.

**Reload.** `ArtifactStore`-style: `LabArtifactStore(cfg).load("<name>")` (Polars LazyFrame), or
`SELECT * FROM plab_<name>` in the Phase 01 catalog, or `lab.<name>` in the serving snapshot.

| Mart | Grain / key | Sources | Notes |
|---|---|---|---|
| `athlete` | athlete · `nfl_id` | gold.dim_athlete, silver.combine_results, gold.fact_combine_attempt, gold.mart_athlete_nfl_profile, silver.player_career_successes | traditional tests (latest Combine), tracked exposure, `career_*` = OUTCOMES, 40-yd percentile within position |
| `combine_attempt` | attempt · `event_id, nfl_id, attempt` | gold.fact_combine_attempt, athlete, *Phase 02 Combine flags (optional)* | attempt_seq, within-drill percentile + band, mph |
| `combine_athlete_drill` | athlete × drill · `nfl_id, drill_type, drill_name` | combine_attempt | best/mean/SD/CV of max speed; flagged attempts counted, not removed |
| `drill_participation` | drill × position × cohort | combine_attempt | distinct athletes, attempts |
| `test_availability` | position × cohort × test | athlete | share of non-null traditional results |
| `coverage_matrix` | position × cohort × source | athlete | source presence shares |
| `linkage_funnel` | position × cohort × stage | athlete | nested stages, label `'<n>. <stage>'` |
| `season_volume` | season × season_type | gold.dim_game, gold.fact_nfl_play_participation | athlete frames exclude football |
| `tracking_partition` | silver tracking file · `file` | silver Parquet **footers** | frame rows incl. football (no scan) |
| `nfl_game` | game · `game_id` | gold.dim_game, fact | `game_label` |
| `nfl_play` | play · `game_id, play_id` | fact, gold.dim_play, nfl_game | picker source, optional play context |
| `nfl_player_play` | player-play · `game_id, play_id, nfl_id` (partition `season`) | fact, dim_play, athlete, nfl_game, *Phase 02 NFL flags (optional)* | whole tracked window incl. pre-snap |
| `nfl_athlete_season` | athlete × season · `nfl_id, season` | gold.mart_athlete_nfl_profile, athlete | `seasons_since_draft` |
| `nfl_speed_distribution` | season × position × metric × 0.25 bin | nfl_player_play | pre-binned histograms |
| `quality_check` | saved check · `check_seq` | reports/tables/data_quality_results, *phase_02 remediation (optional)* | category, severity rank, `results_saved_at` |
| `quality_run_history` | step execution · `run_id, step, started_at` | reports/tables/pipeline_run_log | real history only |
| `tracking_interval`, `tracking_consistency` | domain (× Δt) | saved tracking-QC tables | sampled |
| `quality_affected` | domain | Phase 02 flag datasets | **empty until flags exist** |
| `research_hypothesis` | hypothesis | reports/tables/research_feasibility_matrix.csv | rule-based `readiness_status`; `statistical_support = 'not assessed'` |
| `research_by_position` | hypothesis × position | research_feasibility_by_position.csv | match / missing shares |
| `research_drill` | hypothesis × drill | feasibility matrix + gold.dim_combine_drill | Phase 01 regex matches (H02/H03 over-match) |
| `kpi_snapshot` | single row | gold + silver footers + gold_manifest | unfiltered KPIs, build timestamps |
| `availability` | optional input | — | what was missing at build time |

Units: yards, yd/s, yd/s², seconds; `*_mph` = yd/s × 2.04545. Key columns are never NULL
(unknown cohort = −1, unknown position = `UNKNOWN`).

**Readiness rule (`research_hypothesis.readiness_status`)**, first match wins:
`insufficient sample` (< 30 matched athletes with baseline) → `needs drill-definition review`
(drill regex ≠ `.*`) → `exploratory only` (Phase 01 feasibility = Low) → `data available`.
It is a data-availability statement, never evidence for the hypothesis.
