# 06 — Warehouse design

_Status: **DESIGN — implemented in code (`src/nflbdb/data/`), not yet executed.** Cardinalities are filled in by `warehouse/catalog/gold_manifest.csv` and `reports/tables/table_catalog.csv` after a build. Grains and keys marked "hypothesis" are tested by `validate.py` and asserted by `gold.py` (the build fails if a primary key is not unique)._

Diagrams: [warehouse_er.mmd](diagrams/warehouse_er.mmd) · [data_pipeline.mmd](diagrams/data_pipeline.mmd) · [source_relationships.mmd](diagrams/source_relationships.mmd)

## Layout

```text
warehouse/
  bronze/<source>/[season=YYYY/]<raw_stem>.parquet     lossless typed mirror + _src_file, _src_row
  silver/<table>.parquet                               reference tables
  silver/game_tracking/season=YYYY/part-0.parquet      sorted by game_id, play_id, nfl_id, ts
  silver/combine_tracking/draft_year=YYYY/part-0.parquet  sorted by nfl_id, drill_name, event_id, attempt, ts
  silver/_quarantine/                                  rows removed by documented dedup rules
  gold/<table>.parquet, gold/fact_nfl_play_participation/season=YYYY/
  catalog/bdb27.duckdb (views), bronze_manifest, bronze_state.json, silver_state.json,
          type_plans.json, player_play_levels.json, gold_manifest, table_catalog
```

## Bronze

| Rule | Implementation |
|---|---|
| Raw files untouched | Read-only; outputs only under `warehouse/` |
| Provenance | `bronze_manifest` (size, mtime, SHA-256), `_src_file`, `_src_row` |
| No destructive cleaning | Same rows/columns; nulls from literal `NA` only |
| Schema-change detection | Missing registry column → `SchemaDriftError`; extra column → ingested as Utf8 + warning |
| Incremental | Skip when fingerprint + schema hash + version unchanged |

## Silver

| Table | Grain (PK) | Transformations |
|---|---|---|
| `players` | nfl_id | type plan; `birth_date_parsed` |
| `games` | game_id | type plan; `game_date` from game_id |
| `combine_results` | draft_year, nfl_id | type plan |
| `player_career_successes` | nfl_id | type plan (outcomes) |
| `plays` | game_id, play_id | play-constant columns from player_play (measured) + `game_clock_seconds`, season/week |
| `player_play` | game_id, play_id, nfl_id | team- and player-level columns; boolean casts |
| `game_tracking` | game_id, play_id, nfl_id, ts | strict ts parse; derived `frame_id` (play-aligned), `frame_offset_ms`; `is_football`; `x_std, y_std, dir_std, o_std` (offense → +x); exact-dup removal |
| `combine_tracking` | event_id, nfl_id, attempt, ts (+entity_type) | strict ts; attempt-relative `frame_id` |

Coordinate convention: raw `x` ∈ [0, 120] yd along the field, `y` ∈ [0, 53.3] yd across (inferred from NFL convention; checked by range rules). `*_std` rotate by 180° when `play_direction = 'left'`. Units unchanged: yards, yd/s, yd/s², degrees.

## Gold

| Table | Grain / PK | Sources | Partition | Typical query |
|---|---|---|---|---|
| `dim_athlete` | nfl_id | all player-bearing silver tables | — | athletes by position/cohort with source flags |
| `dim_game` | game_id | silver.games | — | game calendar |
| `dim_play` | game_id, play_id | silver.plays | — | play context for facts |
| `dim_season` | season, season_type | silver.games | — | season ranges |
| `dim_position` | source, position | combine/players/player_play | — | position vocabulary crosswalk |
| `dim_combine_drill` | drill_type, drill_name | fact_combine_attempt | — | which drills exist, coverage |
| `fact_combine_attempt` | event_id, nfl_id, attempt | silver.combine_tracking | — | attempt kinematics (max_s, time-to-max) |
| `fact_nfl_play_participation` | game_id, play_id, nfl_id | silver.game_tracking ⨝ player_play | season | per-play movement + role |
| `fact_nfl_tracking` / `fact_combine_tracking` | frame grain | silver (VIEW) | season / draft_year | trajectories |
| `mart_athlete_combine_profile` | nfl_id, drill_type, drill_name | fact_combine_attempt | — | best/median per drill |
| `mart_athlete_nfl_profile` | nfl_id, season | fact_nfl_play_participation | — | season exposure + movement distribution |
| `mart_combine_nfl_coverage` | position_group, cohort | dim_athlete | — | linkage coverage |
| `mart_athlete_research_base` | nfl_id | dims + marts + career | — | modelling base (`career_*` = outcomes) |

Expected cardinality ordering (to verify): frames ≫ player×play ≫ attempts ≫ athlete×season ≈ athletes.

## Data-quality constraints

- PK uniqueness asserted on every gold build (`gold_manifest.pk_unique`).
- Silver aborts on unparseable timestamps and on conflicting duplicate keys above `max_key_conflict_fraction`.
- Exact duplicates removed with the lowest `_src_row` kept; dropped rows quarantined.
- No frame-level cross-domain joins (cardinality analysis in `04` quantifies the explosion).
- `career_*` outcomes never used as predictors; NFL-derived features must not predict the same NFL follow-up.
