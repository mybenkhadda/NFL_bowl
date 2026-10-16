# User execution guide

Nothing here has been executed by Claude. Labels:
🟢 **read-only** (inspects, changes nothing) · 🟡 **writes lab outputs only** (`warehouse/gold/phase_lab`,
`warehouse/serving`, `reports/*/analytics_lab`, Docker images/volumes) · 🔴 **destructive** (deletes lab
state; never touches raw data or Phase 01/02 outputs).

Commands use **cmd.exe** syntax for host Python (as in `docs/SESSION_STATE.md`); PowerShell variants are
given where they differ. Docker commands are identical in both shells.

## 0. Prerequisites (once)

- Docker Desktop for Windows with the **WSL 2 backend**. Give it a memory budget suited to this
  machine (~2 GB free in earlier runs), e.g. `%UserProfile%\.wslconfig`:
  ```ini
  [wsl2]
  memory=6GB
  processors=6
  ```
  then 🟡 `wsl --shutdown` and restart Docker Desktop.
- Docker Desktop → Settings → Resources → File sharing: the drive holding the project (C:) must be shared.
- The Phase 01 warehouse exists (gold + silver). Phase 02 quality flags are optional.

🟢 Check versions:
```bat
docker version
docker compose version
```

## 0b. Windows PowerShell quick path (editable install → host build → Docker Desktop → Compose)

Equivalent to §§0–4 for PowerShell. The package is installed editable (`pip install -e .`), so
`PYTHONPATH` is not needed. The lab build **reuses** current marts (manifest fingerprint); never pass
`--force` unless you want every mart rebuilt.

🟡 Host Python (project root):
```powershell
conda activate nfl
python -m pip install -e .                       # once; editable — code edits apply without reinstall
$env:NFL_BDB_DUCKDB_MEMORY = "3GB"
python -m pytest tests/test_lab_sql_regressions.py tests/test_analytics_lab.py -q
python -m nflbdb.lab.build                       # reuses current marts, builds only missing/stale ones
python infrastructure\analytics_lab\scripts\validate_connections.py
python infrastructure\analytics_lab\superset\bootstrap\build_bundle.py --check
```

🟡 Docker Desktop (once; elevated PowerShell, then reboot if asked):
```powershell
wsl --install --no-distribution                  # or: wsl --update  (WSL 2 already present)
winget install -e --id Docker.DockerDesktop
```
Start Docker Desktop, accept the terms, keep "Use the WSL 2 based engine" on, then (normal PowerShell):
```powershell
docker version
docker compose version
```

🟡 Compose (from `infrastructure\analytics_lab`):
```powershell
Set-Location infrastructure\analytics_lab
if (-not (Test-Path .env)) { Copy-Item .env.example .env }; notepad .env   # replace every CHANGE_ME
docker compose config --quiet; if ($?) { "compose OK" }
docker compose build
docker compose up -d                             # add --profile monitoring for Grafana
docker compose logs -f superset-init             # Ctrl+C once it exits (0)
docker compose ps
```

## 0c. Startup order and the serving-snapshot rule (cmd.exe)

`serving-check` (one-shot) → `superset-init` (one-shot) → `superset`; `serving-check` → `streamlit`.
Each waits for the previous to **exit 0**, so a missing or unreadable `warehouse\serving\nfl_lab.duckdb`
stops the stack with an explicit message (`docker compose logs serving-check`) instead of empty charts.

**Rebuilding the snapshot while the stack runs fails on Windows** (Streamlit keeps it open). The build
detects this *before* doing any work and leaves the old file untouched. Always:
```bat
cd infrastructure\analytics_lab
docker compose stop superset streamlit grafana
cd ..\..
python -m nflbdb.lab.build --steps serving,monitoring
cd infrastructure\analytics_lab
docker compose up -d
```
(`up -d` re-runs `serving-check` and `superset-init`. Both are idempotent, and the dashboard import is
skipped when the bundle hash is unchanged.)

## 1. Prepare environment variables

🟡 From the project root:
```bat
cd infrastructure\analytics_lab
copy .env.example .env
notepad .env
```
Replace every `CHANGE_ME`. Generate a secret (PowerShell):
```powershell
[Convert]::ToBase64String((1..42 | ForEach-Object { Get-Random -Maximum 256 }))
```
`.env` is git-ignored. 🟢 Validate the resolved configuration (prints it, starts nothing):
```bat
docker compose config --quiet && echo compose OK
```

## 2. Build the analytical marts and serving snapshots (host)

🟡 From the project root, conda env `nfl` (restart any notebook kernel first):
```bat
conda activate nfl
set PYTHONPATH=src
set NFL_BDB_DUCKDB_MEMORY=3GB
python -m pytest tests/test_analytics_lab.py -q
python -m nflbdb.lab.build
```
PowerShell: `$env:PYTHONPATH="src"; $env:NFL_BDB_DUCKDB_MEMORY="3GB"` (or `pip install -e .` once and skip
`PYTHONPATH` — see §0b).

Outputs: `warehouse/gold/phase_lab/` (marts), `warehouse/serving/nfl_lab.duckdb`,
`warehouse/serving/lab_monitoring.sqlite`, `reports/tables/analytics_lab/lab_mart_*.csv`.
🟢 Then:
```bat
python infrastructure\analytics_lab\scripts\validate_connections.py
python infrastructure\analytics_lab\superset\bootstrap\build_bundle.py --check
```

## 3. Build the containers

🟡 From `infrastructure\analytics_lab`:
```bat
docker compose build
```
(First build downloads ≈ 3 GB of base images.)

## 4. Start the stack

🟡
```bat
docker compose up -d
docker compose --profile monitoring up -d        & rem optional: also Grafana
```
🟢 Follow initialisation (migrations, admin user, dashboard import, column sync):
```bat
docker compose logs -f superset-init
docker compose ps
```
`superset-init` must show `Exited (0)`; `superset` and `streamlit` become `healthy` within ~1–2 min.

## 5. Open the applications

| App | URL | Login |
|---|---|---|
| Superset dashboards | http://localhost:8088 | `SUPERSET_ADMIN_USERNAME` / `SUPERSET_ADMIN_PASSWORD` |
| Streamlit trajectory explorer | http://localhost:8501 | none (localhost only) |
| Grafana engineering monitor | http://localhost:3000 | `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` |

All ports are bound to 127.0.0.1 and are not reachable from other machines.

## 6. Application metadata initialisation

Done automatically by `superset-init` (idempotent). To re-run it manually 🟡:
```bat
docker compose run --rm superset-init
```

## 7. Connect analytical data

Automatic: the bundle creates the database **NFL Lab (DuckDB)** →
`duckdb:////srv/nfl-lab/serving/nfl_lab.duckdb` (read-only, external file access disabled).
🟢 Verify in Superset: *Settings → Database Connections → NFL Lab (DuckDB) → Test connection*, or
SQL Lab → run `infrastructure/analytics_lab/sql/serving_checks.sql`.

## 8. Import dashboard definitions

Automatic on first start (`LAB_IMPORT_MODE=on_change`). After editing
`superset/bootstrap/dashboards_spec.py`, 🟡 `docker compose run --rm superset-init` re-imports
(only if the bundle hash changed). To force: set `LAB_IMPORT_MODE=always` in `.env` for one run
(**overwrites UI edits** of the bundled dashboards). Manual import of any bundle 🟡:
```bat
docker compose exec superset superset import-dashboards --path /app/lab/dashboards/exports/<ts>/dashboards_<ts>.zip --username admin
```

## 9. Load and refresh analytical marts

After new Phase 01/02 outputs:
```bat
rem host, project root, conda env nfl
python -m nflbdb.lab.build
```
Then refresh caches:
- Superset: dashboard ⋯ → **Force refresh** (or 🟡 `docker compose restart superset` to clear the in-process cache).
- Streamlit: sidebar **Reload data** (it also reconnects automatically when the snapshot file changes).
- Grafana: panels re-query on load.

If the build reports the snapshot file is locked: 🟡 `docker compose stop superset streamlit` (and
`grafana`), rebuild, 🟡 `docker compose start superset streamlit`.

Full rebuild of all marts (archives previous versions): `python -m nflbdb.lab.build --force`.
Serving snapshot only: `python -m nflbdb.lab.build --steps serving,monitoring`.

## 10. Test the dashboards (manual acceptance)

🟢 `python infrastructure\analytics_lab\scripts\validate_connections.py --http`

| Check | Expectation |
|---|---|
| NFL Data Overview KPIs | athletes / matched / player-plays equal `lab.kpi_snapshot` (SQL Lab check 2) |
| Position filter | every chart on athlete-based datasets narrows; funnel sums change |
| Season filter | tracking volume + NFL charts narrow; Combine charts show "not applicable" |
| Combine Explorer | pick 2–3 athletes in *Athlete* → comparison table shows only them |
| NFL Explorer | Season → Game → Play cascade; detail table ≤ 500 rows |
| Data Quality | failed/warn counts match `reports/tables/data_quality_results.csv`; detail table downloads as CSV |
| Research | readiness column populated; no "supported" wording anywhere |
| Streamlit | a Combine attempt and an NFL play play back; A/B curves align per the chosen method |
| Persistence | `docker compose restart` → dashboards, charts and users still present |

## 11. Export figures and tables

- Streamlit: **Export** panel → *Save figures to reports/* writes `reports/figures/analytics_lab/<name>__<timestamp>.png`
  (300 DPI) + `.svg`; *Save tables* writes `reports/tables/analytics_lab/<name>__<timestamp>.csv`.
- Superset: any chart ⋯ → *Download* → CSV / image; dashboard ⋯ → *Download as image/PDF*.
- Dashboard definitions 🟡: `python scripts\export_dashboards.py` (from `infrastructure\analytics_lab`)
  → `superset/dashboards/exports/<timestamp>/` — commit it.

## 12. Stop, restart, remove

| Action | Command | Effect |
|---|---|---|
| 🟡 Stop (keep everything) | `docker compose --profile monitoring stop` | containers stopped; volumes kept |
| 🟡 Start again | `docker compose --profile monitoring start` | metadata, dashboards, users intact |
| 🟡 Remove containers + network | `docker compose --profile monitoring down` | **volumes kept** → nothing lost |
| 🔴 Remove lab metadata too | `docker compose --profile monitoring down --volumes` | deletes Superset metadata DB, Superset home, Grafana state (dashboards re-import from code on next start) |
| 🔴 Remove lab images | `docker image rm nfl-lab/superset:5.0.0 nfl-lab/streamlit:1` | frees disk |

None of these touch `data/`, `warehouse/bronze|silver|gold` (Phase 01/02), or `reports/` — the warehouse
is mounted read-only into every container. Lab-only host outputs can be removed manually 🔴 if desired:
`warehouse\serving\`, `warehouse\gold\phase_lab\`, `reports\figures\analytics_lab\`, `reports\tables\analytics_lab\`.
