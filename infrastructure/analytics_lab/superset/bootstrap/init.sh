#!/bin/bash
# One-shot Superset initialisation (service `superset-init`). Idempotent: safe on every start.
#   1. migrate the metadata DB            4. build the dashboard bundle from dashboards_spec.py
#   2. create the admin user (once)       5. import it (LAB_IMPORT_MODE: on_change | always | never)
#   3. superset init (roles/permissions)  6. sync dataset columns from the DuckDB snapshot
set -euo pipefail
cd /app

echo "[lab-init] 1/6 superset db upgrade"
superset db upgrade

echo "[lab-init] 2/6 admin user"
# Capture first: `… | grep -q` under pipefail can fail on SIGPIPE and wrongly re-create the admin.
USERS="$(superset fab list-users 2>/dev/null || true)"
if grep -q "username:${ADMIN_USERNAME}\b" <<<"${USERS}"; then
  echo "[lab-init] admin '${ADMIN_USERNAME}' already exists"
else
  superset fab create-admin --username "${ADMIN_USERNAME}" --firstname Lab --lastname Admin \
    --email "${ADMIN_EMAIL}" --password "${ADMIN_PASSWORD}"
fi

echo "[lab-init] 3/6 superset init"
superset init

# Serving snapshot must be readable by THIS image's DuckDB (catches missing mount, empty file,
# storage-version mismatch). Failing here keeps `superset` from starting (service_completed_successfully).
if ! python - <<'PY'
import os, sys
import duckdb
p = os.environ["LAB_SERVING_DB"]
if not os.path.isfile(p) or not os.access(p, os.R_OK) or os.path.getsize(p) == 0:
    sys.exit(f"[lab-init] ERROR: {p} missing, unreadable or empty")
con = duckdb.connect(p, read_only=True, config={"enable_external_access": False})
n = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'lab'").fetchone()[0]
con.close()
if n == 0:
    sys.exit(f"[lab-init] ERROR: {p} has no tables in schema 'lab'")
print(f"[lab-init] serving snapshot OK: {n} lab tables (duckdb {duckdb.__version__})")
PY
then
  echo "[lab-init] Build it on the host: python -m nflbdb.lab.build --steps serving,monitoring"
  echo "[lab-init] and check NFL_BDB_PROJECT_ROOT in .env (host folder mounted at /srv/nfl-lab/serving)."
  exit 1
fi

MODE="${LAB_IMPORT_MODE:-on_change}"
BUNDLE=/tmp/nfl_lab_bundle.zip
MARKER_DIR=/app/superset_home/lab
mkdir -p "${MARKER_DIR}"

echo "[lab-init] 4/6 build dashboard bundle (mode=${MODE})"
python /app/lab/bootstrap/build_bundle.py --zip "${BUNDLE}"   # exits 2 on spec / reference problems
HASH=$(python /app/lab/bootstrap/build_bundle.py --print-hash)

if [ "${MODE}" = "never" ]; then
  echo "[lab-init] 5/6 import skipped (LAB_IMPORT_MODE=never)"
elif [ "${MODE}" = "on_change" ] && [ -f "${MARKER_DIR}/imported_${HASH}" ]; then
  echo "[lab-init] 5/6 bundle ${HASH} already imported — skipped (UI edits preserved)"
else
  echo "[lab-init] 5/6 import bundle ${HASH}"
  superset import-dashboards --path "${BUNDLE}" --username "${ADMIN_USERNAME}"
  rm -f "${MARKER_DIR}"/imported_*
  touch "${MARKER_DIR}/imported_${HASH}"
fi

echo "[lab-init] 6/6 sync dataset columns"
python /app/lab/bootstrap/post_import.py || echo "[lab-init] column sync reported problems (non-fatal; see above)"

echo "[lab-init] done"
