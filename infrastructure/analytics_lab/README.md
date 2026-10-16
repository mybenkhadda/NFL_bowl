# NFL Big Data Bowl 2027 — Analytics Lab

Local, Dockerised exploration and decision-support environment over the existing Bronze/Silver/Gold
warehouse. **Not a competition submission.** Authored 2026-10-08; **nothing has been executed yet**.

| Component | What it shows | URL |
|---|---|---|
| Superset · *NFL Data Overview* | KPIs, coverage, linkage, drill & test availability, volume by season | http://localhost:8088 |
| Superset · *Combine Explorer* | distributions, peak speed vs 40 time, variability, athlete comparison | ″ |
| Superset · *NFL Game Tracking Explorer* | speed/accel distributions, exposure, season → game → play drill-down | ″ |
| Superset · *Data Quality Monitor* | checks by severity/status/table/category, run history, CSV export | ″ |
| Superset · *Research Feasibility* | H01–H05 candidates, eligibility, coverage, constraints (no statistical claims) | ″ |
| Streamlit · trajectory explorer | field playback, speed/accel-time curves, A/B comparison, 300-DPI export | http://localhost:8501 |
| Grafana · *Data Engineering* (optional) | storage, freshness, step durations, memory, benchmarks, query durations | http://localhost:3000 |

## Quick start

```bat
rem 1) host, project root, conda env nfl — build marts + serving snapshots
set PYTHONPATH=src
set NFL_BDB_DUCKDB_MEMORY=3GB
python -m nflbdb.lab.build

rem 2) stack
cd infrastructure\analytics_lab
copy .env.example .env   &  rem then fill every CHANGE_ME
docker compose build
docker compose up -d                              &  rem + --profile monitoring for Grafana
docker compose logs -f superset-init
```

Full guide (safe vs state-changing commands, refresh, export, shutdown): [docs/setup.md](docs/setup.md).
Design + estimates + compatibility: [docs/architecture.md](docs/architecture.md).
Mart schemas/grains/refresh: [docs/data_contracts.md](docs/data_contracts.md).
Problems: [docs/troubleshooting.md](docs/troubleshooting.md).

## Layout

```text
infrastructure/analytics_lab/
  compose.yaml  .env.example  .gitignore  .gitattributes
  postgres/initdb/10_grafana_reader.sh       read-only role for Grafana
  superset/
    Dockerfile  requirements.txt  superset_config.py
    bootstrap/  init.sh  post_import.py  bundle_lib.py  dashboards_spec.py  build_bundle.py
    dashboards/ README.md  generated/  exports/<timestamp>/
  streamlit/
    Dockerfile  requirements.txt  .streamlit/config.toml  app.py
    pages/1_Combine_movement.py  pages/2_NFL_play_movement.py
    labkit/  data.py  field.py  kinematics.py  export.py  ui.py
  grafana/
    provisioning/datasources/datasources.yaml  provisioning/dashboards/provider.yaml
    dashboards/engineering.json
  sql/  serving_checks.sql  bounded_frame_queries.sql
  scripts/  prepare_serving_data.py  validate_connections.py  export_dashboards.py
  docs/  architecture.md  setup.md  data_contracts.md  troubleshooting.md
src/nflbdb/lab/   marts.py  serving.py  monitoring.py  build.py      (host-side builder)
tests/test_analytics_lab.py
```

## Guarantees by design (to be verified by the user's run)

- Raw data and Phase 01/02 outputs are mounted **read-only**; the lab writes only to
  `warehouse/gold/phase_lab`, `warehouse/serving`, `reports/{figures,tables}/analytics_lab`.
- Dashboards query compact marts; frame tables are never exposed to Superset. Streamlit reads one
  play / one attempt per request from a single partition with a row cap.
- All KPIs are computed from warehouse contents at build time; missing optional inputs produce empty
  charts plus a row in `lab.availability`, never placeholder numbers.
- Dashboard, datasource and panel definitions are version-controlled; metadata lives in named volumes.
- Secrets live only in the git-ignored `.env`; all ports bind to 127.0.0.1.
