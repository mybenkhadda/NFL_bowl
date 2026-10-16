-- Read-only sanity checks on the serving snapshot (schema `lab`).
-- Superset SQL Lab → database "NFL Lab (DuckDB)", or:  duckdb -readonly warehouse/serving/nfl_lab.duckdb

-- 1. What is in the snapshot (grain, key, rows, build time)
SELECT table_name, grain, key, rows, mart_built_at FROM lab.mart_catalog ORDER BY table_name;

-- 2. Headline KPIs (compare with docs/SESSION_STATE.md: 510 athletes, 497 with NFL tracking,
--    6,310 Combine attempts, 314,944 player-plays, 22.65M silver NFL rows)
SELECT n_athletes, n_athletes_nfl_tracking, n_athletes_matched, n_combine_attempts, n_player_plays,
       n_silver_nfl_tracking_rows, seasons_label, gold_built_at, lab_built_at
FROM lab.kpi_snapshot;

-- 3. Reconciliation: player-plays by season must sum to the KPI
SELECT (SELECT sum(n_player_plays) FROM lab.season_volume) AS by_season,
       (SELECT n_player_plays FROM lab.kpi_snapshot)        AS kpi;

-- 4. Key uniqueness spot-check (expect 0)
SELECT count(*) - count(DISTINCT (game_id, play_id, nfl_id)) AS duplicate_keys FROM lab.nfl_player_play;

-- 5. Which optional inputs were missing at build time
SELECT * FROM lab.availability WHERE NOT available;

-- 6. Linkage funnel overall
SELECT stage, sum(n_athletes) AS athletes FROM lab.linkage_funnel GROUP BY stage ORDER BY stage;
