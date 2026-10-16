# Superset dashboard definitions

| Folder | Content | Source of truth? |
|---|---|---|
| `../bootstrap/dashboards_spec.py` | Declarative definition of the database, 23 datasets, ~70 charts and 5 dashboards | **Yes** — edit here |
| `generated/` | Unzipped bundle written by `build_bundle.py --out-dir generated` (for review / diffs) | Derived |
| `exports/<timestamp>/` | Bundles exported from the running Superset by `scripts/export_dashboards.py` (captures UI edits) | Snapshot of UI state |

Round trip:
1. Change `dashboards_spec.py` → restart `superset-init` → dashboards re-import (only when the bundle hash changed).
2. Polish a chart in the UI → run `scripts/export_dashboards.py` → commit the export. To make a UI change
   permanent in code, port it back into `dashboards_spec.py` (the export shows the exact `params`).
3. To import an export instead of the generated bundle:
   `docker compose exec superset superset import-dashboards --path /app/lab/dashboards/exports/<ts>/<file>.zip --username admin`
