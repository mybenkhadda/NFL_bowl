# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker compose` → "Set SUPERSET_SECRET_KEY in .env" | `.env` missing / incomplete | `copy .env.example .env`, fill every `CHANGE_ME` |
| `superset-init` exits non-zero at `db upgrade` | Postgres not ready or wrong password | `docker compose logs superset-db`; if you changed `POSTGRES_PASSWORD` after first start, the volume keeps the old one → 🔴 `docker compose down --volumes` (lab metadata only) |
| Image build fails at `uv: command not found` | Base tag without uv in `/app/.venv` | In `superset/Dockerfile` replace the `uv pip install` line with `/app/.venv/bin/python -m ensurepip && /app/.venv/bin/python -m pip install -r /tmp/lab-requirements.txt` |
| Build fails at `import duckdb_engine` | Driver not installed into Superset's venv | Same as above; check the tag's Python version supports the pinned wheels |
| `init.sh: $'\r': command not found` | CRLF line endings after checkout | `git config core.autocrlf false`, re-checkout (`.gitattributes` forces LF), or convert the file to LF in VS Code |
| Charts: "IO Error: Cannot open file … nfl_lab.duckdb" | Snapshot not built / wrong project root | Run `python -m nflbdb.lab.build`; check `NFL_BDB_PROJECT_ROOT` in `.env`; `docker compose exec superset ls -l /srv/nfl-lab/serving` |
| "Serialization Error … storage version" / "file was written by a newer DuckDB" | Host DuckDB newer than container DuckDB | Bump `duckdb==` in `superset/requirements.txt` and `streamlit/requirements.txt` to ≥ host version, `docker compose build` |
| Connection test fails mentioning `enable_external_access` / extensions | Driver tries to auto-load an extension | Database → Edit → Advanced → Other → Engine parameters: delete `"enable_external_access": false` |
| Chart: "Column … does not exist" | Mart changed, dataset metadata stale | 🟡 `docker compose run --rm superset-init` (column sync) or Dataset → *Sync columns from source* |
| A chart renders oddly after import | Param differences across Superset versions | Fix in Explore, save, run `scripts/export_dashboards.py`; port the params to `dashboards_spec.py` |
| Filter shows "not applicable" on some charts | Chart's dataset lacks that column | Expected (e.g. *NFL season* on Combine charts) |
| Charts empty on Quality / Research dashboards | Optional input missing | Overview → *Optional inputs* table (`lab.availability`); build the input (e.g. `python -m nflbdb.research.quality`) and rebuild the lab |
| Numbers unchanged after rebuild | Cached results | Dashboard ⋯ → Force refresh, or `docker compose restart superset` |
| Lab build: `ServingFileLockedError: … nfl_lab.duckdb is in use` (was `WinError 5`) | Streamlit keeps the snapshot ATTACHed; Windows refuses to replace an open file. Detected **before** building; the old file is untouched | `docker compose stop superset streamlit grafana` → `python -m nflbdb.lab.build --steps serving,monitoring` → `docker compose start superset streamlit` |
| Lab build: `… nfl_lab.duckdb.wal exists` | A tool opened the snapshot read-write | Close it; if nothing is open, delete only `warehouse\serving\nfl_lab.duckdb.wal` and re-run the serving step |
| `superset-init` / `streamlit` never start; `serving-check` exited 1 | Snapshot missing / empty / wrong mount | `docker compose logs serving-check`; build the snapshot; set an absolute `NFL_BDB_PROJECT_ROOT` (forward slashes) in `.env` |
| `superset-init`: `KeyError: 'allow_csv_upload'` | Bundle built before the Superset 5.0 database-key fix | `docker compose build` not needed (bootstrap is mounted): `docker compose run --rm superset-init` |
| Superset container `Exited (137)` | OOM kill (container or WSL VM) | `SUPERSET_WORKERS=1` in `.env`; `docker stats`; raise `.wslconfig` memory before `SUPERSET_MEM_LIMIT` |
| Streamlit "Data not available: no silver partition" | Silver not mounted / not built | Check `warehouse/silver/game_tracking/season=*`; `docker compose exec streamlit ls /data/warehouse/silver` |
| Streamlit export: "Figure export failed … kaleido" | kaleido Chromium missing libraries | `docker compose exec streamlit python -c "import plotly.io as pio; pio.kaleido.scope"`; keep kaleido 0.2.1 + plotly 5.24.1 pinned together |
| Grafana panels "plugin not found" | Plugin download blocked (no internet on first start) | Ensure internet once; `docker compose --profile monitoring restart grafana`; or pin `GRAFANA_SQLITE_PLUGIN=frser-sqlite-datasource@<version>` |
| Grafana SQLite "database is locked" / "unable to open" | Read-only mount + journal | Re-run the lab build (it writes `journal_mode=DELETE`); optionally add `pathOptions: "mode=ro&immutable=1"` under `jsonData` in `grafana/provisioning/datasources/datasources.yaml` |
| Grafana Superset panels: "permission denied" | Role created before tables existed on an old volume | `docker compose exec superset-db psql -U superset -d superset -c "GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_reader;"` |
| Containers OOM-killed / very slow | Not enough memory for WSL2 | Raise `.wslconfig` memory, lower `SUPERSET_WORKERS=1`, skip the monitoring profile, close other apps |
| Port already in use | Another local service | Change `SUPERSET_PORT` / `STREAMLIT_PORT` / `GRAFANA_PORT` in `.env` |
| Bind mount empty inside container | Drive not shared / path with unusual characters | Docker Desktop → Resources → File sharing; set `NFL_BDB_PROJECT_ROOT` to an absolute forward-slash path |
