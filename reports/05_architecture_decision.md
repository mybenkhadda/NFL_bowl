# 05 — Architecture decision record

_Status: **PROPOSED — pending measurements.** Written 2026-10-08 from the confirmed raw schema (CSV headers of all 9 files). Nothing below has been executed yet. Re-validate this decision once `00_environment_report.md`, `01_data_inventory.md` and `07_performance_benchmarks.md` exist._

## Context (confirmed facts)

- 9 raw CSV files: `players`, `combine_results`, `combine_tracking`, `game_tracking_2023/2024/2025`, `games`, `player_play`, `player_career_successes`.
- Two frame-level tracking domains (Combine drills; NFL games), 10 Hz timestamps observed in sampled rows; **no `frame_id` column** in game tracking.
- Game tracking is already split by season in the filenames → a natural partition.
- `player_play` (66 columns) denormalises play context onto each player row.
- `nfl_id` is present in every player-bearing table (shared namespace is a hypothesis, tested in `04`).
- Nulls are encoded as the literal `NA`; booleans as `TRUE`/`FALSE` strings.

## Measurements this decision depends on (PENDING)

| Quantity | Source | Value |
|---|---|---|
| Raw size, rows per file | `01_data_inventory.md` | _pending_ |
| RAM / cores / disk | `00_environment_report.md` | _pending_ |
| Parquet/CSV size ratio | `tables/bronze_ingest_summary.csv` | _pending_ |
| CSV vs Parquet query times, peak memory | `07_performance_benchmarks.md` | _pending_ |

## Options

| Criterion | A: Raw CSV + Polars lazy | B: Parquet + DuckDB | C: Bronze/Silver/Gold Parquet + DuckDB catalog + Polars | D: Spark / Dask / Ray |
|---|---|---|---|---|
| Repeated-query efficiency | ✗ every query re-parses CSV | ✓ columnar, pushdown | ✓ plus compact gold tables (no frame scans for athlete work) | ✓ but overhead |
| Fit to dataset size on one workstation | ✓ | ✓ | ✓ | ✗ unnecessary unless data ≫ disk/RAM |
| Data-quality controls | ad hoc | ad hoc views | ✓ explicit layer contracts, quarantine, PK assertions | possible, heavy |
| Reproducibility / provenance | weak | medium | ✓ manifest, checksums, `_src_row`, state files, idempotent builds | medium |
| Complexity | lowest | low | moderate (plain files + Python, no services) | high |
| Incremental rebuilds | n/a | manual | ✓ fingerprint-based per file/partition | ✓ |
| ML interoperability | ✓ Polars | ✓ Arrow/Polars | ✓ Arrow/Polars/DuckDB | ✓ |
| Local / Kaggle portability | ✓ | ✓ | ✓ (no server; paths from config) | ✗ |

## Decision

**Option C — Polars + DuckDB + Parquet, Bronze/Silver/Gold**, which validates the initial preference, with three refinements driven by the observed schema:

1. **Partition only by low-cardinality keys already present** (`season` for game tracking, `draft_year` for Combine tracking). No partitioning by `nfl_id`/`game_id`/`play_id`; instead **sort within partitions** so Parquet row-group statistics prune selective lookups (benchmark Q4 tests this).
2. **Gold never copies frames.** Frame tables are exposed as views (`fact_nfl_tracking`, `fact_combine_tracking`); gold materialises only aggregates at their natural parent grain (attempt, player×play, athlete×season, athlete).
3. **Silver splits `player_play`** into `plays` and `player_play` using a measured constancy test per column, not assumptions.

Option D is rejected unless benchmarks show DuckDB/Polars exceeding memory with spill enabled, or runtimes unacceptable for the iteration loop.

## Technology rationale

| Decision | Why |
|---|---|
| Polars `scan_csv → sink_parquet` for bronze | Streaming, adds `_src_row` cheaply, explicit typed schema; fails loudly on unparseable values. |
| DuckDB for silver/gold/QC | Out-of-core window functions, joins and `COPY … TO parquet` with spill to disk and a hard memory limit. |
| Parquet zstd-3, 500k-row groups | Good compression/speed balance; row groups large enough for scan speed, small enough for pruning. |
| DuckDB file of views as catalog | Zero data duplication; rebuilt per machine in seconds. |
| Bronze types: ids Int64, measures Float64, boolean-like Utf8 | Lossless; integer/boolean casts happen in silver only after evidence. |
| Exact `nfl_id` linkage only | Names are not identifiers; name matching risks silent false links. |

## Revisit triggers

- Raw data > ~50% of free disk → move warehouse/spill via `NFL_BDB_WAREHOUSE` / `NFL_BDB_TMP`.
- Benchmarks show Parquet not faster than CSV for repeated queries → simplify to Option B.
- Peak memory near the DuckDB limit during silver builds → per-season or per-week processing.
