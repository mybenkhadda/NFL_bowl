-- Bounded frame-level queries (run on the HOST from the project root, e.g. `duckdb` CLI, read-only).
-- Pattern used by the Streamlit explorer:
--   * read ONE partition file (path = partition pruning),
--   * filter on the sort keys (game_id, play_id) / (nfl_id, drill_name, event_id, attempt) so DuckDB skips
--     row groups via Parquet min/max statistics,
--   * project only needed columns, always LIMIT.
-- Replace the literals; pick game/play ids from lab.nfl_play and attempts from lab.combine_attempt.

-- One NFL play (all tracked entities incl. football)
SELECT nfl_id, is_football, frame_id, frame_offset_ms / 1000.0 AS t_s, x_std, y_std, s, a, dis, dir_std, event
FROM read_parquet('warehouse/silver/game_tracking/season=2024/part-0.parquet')
WHERE game_id = 2024010707 AND play_id = 570
ORDER BY is_football, nfl_id, ts
LIMIT 150000;

-- One Combine attempt (PLAYER frames)
SELECT frame_id, frame_offset_ms / 1000.0 AS t_s, x, y, s, a, dis, dir
FROM read_parquet('warehouse/silver/combine_tracking/draft_year=2025/part-0.parquet')
WHERE nfl_id = 58359 AND event_id = '20250302101000_1740947407678_000' AND attempt = 1 AND entity_type = 'PLAYER'
ORDER BY ts
LIMIT 6000;

-- Inspect pruning: EXPLAIN ANALYZE shows how many row groups were actually scanned.
EXPLAIN ANALYZE
SELECT count(*) FROM read_parquet('warehouse/silver/game_tracking/season=2024/part-0.parquet')
WHERE game_id = 2024010707 AND play_id = 570;
