# Analytics Lab — architecture decision

_Status: authored 2026-10-08, **not executed**. Resource numbers are estimates._

## 1. Decision

| Need | Choice | Why |
|---|---|---|
| Interactive BI dashboards (overview, Combine, NFL, quality, research) | **Apache Superset 5.0.0** (custom image) | Native filters, cross-filters, SQL Lab, CSV export, dashboards as importable YAML bundles |
| Analytical serving layer | **Self-contained DuckDB snapshot** `warehouse/serving/nfl_lab.duckdb` | Gold marts are tiny (tens of MB); copying them into one file makes Superset independent of host paths and of the 22.7M-frame tables |
| Frame-level trajectories, playback, speed/accel curves | **Streamlit 1.39 + Plotly 5.24** | Superset has no field/trajectory playback; a small app reads ONE play / ONE attempt at a time |
| Engineering observability | **Grafana 12** (optional `monitoring` profile) | Pipeline durations, memory, storage, freshness, Superset query durations |
| Superset metadata | **PostgreSQL 16** (metadata only, ~MB) | Robust multi-worker metadata store; **no warehouse data is copied into it** |

Rejected: PostgreSQL as the *analytical* store (would duplicate data and add a load step; DuckDB
already reads Parquet natively); Superset → DuckDB views over host Parquet (the existing
`catalog/bdb27.duckdb` embeds absolute Windows paths, unusable in Linux containers); Grafana's
community DuckDB plugin (unsigned, glibc-only image, version-fragile) — Grafana reads a SQLite
snapshot through the stable `frser-sqlite-datasource` plugin instead.

### Connector verification
* Superset ↔ DuckDB: SQLAlchemy dialect **`duckdb-engine`** (`duckdb:////abs/path.duckdb`, four
  slashes for absolute paths; `connect_args` accepts `read_only` and a DuckDB `config` dict).
  Lean Superset images contain **no** drivers; the docs prescribe installing them into
  `/app/.venv` with `uv pip install` in a derived image → `superset/Dockerfile`.
* Grafana ↔ SQLite: `frser-sqlite-datasource` (installed by `GF_PLUGINS_PREINSTALL`), provisioned
  with `jsonData.path`; time columns as UNIX seconds named `time`.
* Grafana ↔ Superset metadata: built-in PostgreSQL datasource with a read-only role.

## 2. Data flow

```text
 data/*.csv (raw, read-only)
      │  Phase 01 pipeline (host, conda env `nfl`)
      ▼
 warehouse/bronze ─► silver (22.65M NFL frames, 463k Combine frames) ─► gold (dims, facts, marts)
      │                     │                                             │
      │                     │                  python -m nflbdb.lab.build │ (host; reads gold + saved reports)
      │                     │                                             ▼
      │                     │                    warehouse/gold/phase_lab/*.parquet  (23 marts, manifest,
      │                     │                    schema sidecars, archive; catalogued as plab_* views)
      │                     │                                             │
      │                     │                    ┌────────────────────────┴───────────────┐
      │                     │                    ▼                                        ▼
      │                     │    warehouse/serving/nfl_lab.duckdb          warehouse/serving/lab_monitoring.sqlite
      │                     │    (schema lab, self-contained)              (run log, storage, freshness, builds)
      │                     │           │ ro              │ ro                            │ ro
      │                     │           ▼                 ▼                               ▼
      │                     └─ro──►  Streamlit ◄──── pickers           Superset        Grafana ◄── Superset logs
      │                       (one play / one attempt                 (5 dashboards)      (postgres, read-only role)
      │                        per query, LIMITed)                         │
      │                                                                    ▼
      │                                                     PostgreSQL: Superset metadata only
      ▼
 reports/figures/analytics_lab, reports/tables/analytics_lab  ◄── Streamlit exports (PNG 300 DPI, SVG, CSV)
```

Grains served: athlete · athlete×drill · Combine attempt · athlete×season · player-play · play ·
game · season. Frames are **never** exposed to Superset.

## 3. Services

| Service | Image | Port (127.0.0.1) | Mounts | Limit |
|---|---|---|---|---|
| superset-db | postgres:16.4-alpine | — (internal network) | volume `superset_db` | 512 MB |
| superset-init | nfl-lab/superset (one-shot) | — | serving ro, bootstrap ro, `superset_home` | 1.5 GB |
| superset | nfl-lab/superset:5.0.0 | 8088 | serving ro | 2 GB |
| streamlit | nfl-lab/streamlit:1 (python:3.10-slim) | 8501 | serving ro, silver ro, 2 export folders rw | 1.5 GB |
| grafana *(profile monitoring)* | grafana/grafana:12.0.2 | 3000 | serving ro, provisioning ro, volume `grafana_data` | 512 MB |

Networks: `lab-backend` is `internal` (metadata DB unreachable from outside); web services also
join `lab-frontend` to publish their port on localhost only.

## 4. Resource estimates (ESTIMATES — not measured)

| Item | Estimate |
|---|---|
| Image disk | Superset ≈ 1.3–1.7 GB · Streamlit ≈ 0.6–0.8 GB · PostgreSQL ≈ 0.25 GB · Grafana ≈ 0.5 GB → **≈ 3–3.5 GB** |
| Serving snapshot | ≈ 20–60 MB (dominated by `nfl_player_play`, ~315k rows) · SQLite ≈ 1–5 MB |
| `phase_lab` marts | ≈ 15–40 MB Parquet |
| RAM, idle stack (no Grafana) | Superset 2 workers ≈ 0.6–1.0 GB · Postgres ≈ 0.05–0.1 GB · Streamlit ≈ 0.2–0.4 GB → **≈ 1–1.5 GB** |
| RAM, active use | + DuckDB per query ≤ 1 GB (Superset cap) / ≤ 768 MB (Streamlit cap) → **≈ 2–3 GB**; hard caps sum to 4.5 GB (+0.5 GB Grafana) |
| Host lab build | DuckDB capped by `NFL_BDB_DUCKDB_MEMORY` (3 GB recommended); reads only gold (16.5 MB) + footers; expected < 1 min |
| CPU | 2 cores per web service cap; idle ≈ 0 |

This machine has 15.4 GB RAM but only ~2 GB free during earlier runs (`reports/00_environment_report.md`).
Recommended: give WSL2 a fixed budget (`%UserProfile%\.wslconfig` → `[wsl2] memory=6GB`), run the
host lab build **before** `docker compose up`, close heavy apps, and enable Grafana only when needed.

## 5. Compatibility issues to watch

1. **DuckDB storage version.** The snapshot is written by the host DuckDB (1.5.6) and read by the
   container DuckDB (pinned 1.5.6). A *newer* reader is fine; an *older* reader cannot open newer files.
   Keep `superset/requirements.txt` and `streamlit/requirements.txt` ≥ host version.
2. **Lean Superset images have no drivers**; `uv` lives in `/app/.venv`. If a future tag changes this,
   the build fails at the `python -c "import duckdb_engine"` sanity check (by design).
3. **Chart params drift between Superset versions.** The generator sets only core params of
   `echarts_timeseries_bar/line`, `pie`, `table`, `big_number_total`, `heatmap_v2`, `histogram_v2`,
   `box_plot`, `bubble_v2` (all present in 5.0). A chart that renders oddly can be fixed in the UI and
   exported (`scripts/export_dashboards.py`).
4. **Dark mode**: Superset 5.0 has no built-in dark theme; Streamlit and Grafana run dark. Colours are
   consistent across all three (position palette from `src/nflbdb/data/reporting.py`).
5. **Windows file replacement**: a DuckDB/SQLite file that is open cannot be replaced on NTFS. Superset
   opens a connection per query, so swaps normally succeed; if not, stop readers and rebuild.
6. **Windows bind mounts** are slower than WSL-native paths; acceptable here because Superset reads one
   small file and Streamlit reads one partition per request.
7. **CRLF**: shell scripts must keep LF endings (`.gitattributes` enforces this on checkout).
8. **`enable_external_access = false`** is set on Superset's DuckDB connection. If a future
   duckdb-engine tries to auto-load extensions on connect, remove that key from the database
   *Advanced → Other → Engine parameters* (the snapshot needs no extensions).
9. **Grafana plugin** downloads on first start (needs internet once; cached in `grafana_data`).
   The PostgreSQL datasource type id is `grafana-postgresql-datasource` (Grafana ≥ 10.x).
10. **kaleido 0.2.1 ↔ plotly 5.24.1** are pinned as a pair; kaleido ≥ 1.0 needs Chrome + plotly ≥ 6.1.
11. **Native filters** apply to every chart whose dataset has a column of the same name. Marts share
    `position_group`, `cohort`, `season`, `drill_name`, `display_name` for this reason; charts without
    the column show the filter as "not applicable".
