# 00 — Environment report

_Step 0: resources, versions and recommended engine settings_

_Generated 2026-10-08T00:32:58Z by `nflbdb.data` — numbers below come from executed code only._

## Execution environment

| item | value |
|---|---|
| Platform | local |
| OS | Windows 10 (10.0.26300) |
| Python | 3.10.18 |
| CPU (logical / physical) | 16 / 8 |
| RAM total / available | 15.4 GB / 706.3 MB |
| DATA_ROOT | data (exists: True) |
| Raw data on disk | 2.1 GB |
| Disk free (data) | 200.0 GB |
| Disk free (warehouse) | 200.0 GB |
| POLARS_MAX_THREADS / pool | 15 / 15 |


## Library versions

| item | value |
|---|---|
| polars | 2.0.0 |
| duckdb | 1.5.6 |
| pyarrow | 25.0.1 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| matplotlib | 3.10.7 |
| psutil | 7.1.0 |
| pytest | 9.1.1 |


## Recommended settings

| item | value |
|---|---|
| polars_threads | POLARS_MAX_THREADS=15 (logical cores - 1; leaves one core for OS/IO) |
| duckdb_threads | SET threads=15 |
| duckdb_memory_limit | SET memory_limit='7.7GB' (50% of 15.4 GB; never 100%) |
| duckdb_temp_directory | .tmp/duckdb_spill (git-ignored; same disk as warehouse) |
| preserve_insertion_order | false for COPY/aggregates (lower memory); ordering is made explicit with ORDER BY / _src_row |
| parquet | zstd level 3, row groups of 500,000 rows; one file per partition |
| streaming | Polars: scan_csv/scan_parquet + sink_parquet (streaming); collect only aggregates or head(). DuckDB: COPY (SELECT …) TO … for out-of-core materialisation. |
| raw_vs_ram | raw data 2.1 GB vs RAM 15.4 GB (ratio 0.14) — fits comfortably; out-of-core still used for safety |
| disk_headroom | estimated need ≈ 4.3 GB for warehouse + spill; free 200.0 GB — OK |
| distributed | Not recommended unless measured peak memory or runtimes exceed limits (see 07_performance_benchmarks.md). |


## Configuration

All paths derive from the project root or from `configs/phase01.toml`; override with `NFL_BDB_DATA_ROOT`, `NFL_BDB_WAREHOUSE`, `NFL_BDB_REPORTS`, `NFL_BDB_TMP`, `NFL_BDB_THREADS`, `NFL_BDB_DUCKDB_MEMORY`. Call `configure_runtime(cfg)` before importing Polars.
