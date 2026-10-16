"""Superset configuration for the NFL Analytics Lab (local, single user).

Secrets come from environment variables injected by compose.yaml from the git-ignored `.env`.
"""

import os

# ---- Security -------------------------------------------------------------------------
SECRET_KEY = os.environ["SUPERSET_SECRET_KEY"]          # fail fast when missing
WTF_CSRF_ENABLED = True
PREVENT_UNSAFE_DB_CONNECTIONS = True
APP_NAME = "NFL Analytics Lab"

# ---- Metadata database (PostgreSQL service `superset-db`) ------------------------------
SQLALCHEMY_DATABASE_URI = (
    "postgresql+psycopg2://{u}:{p}@{h}:5432/{d}".format(
        u=os.environ["POSTGRES_USER"], p=os.environ["POSTGRES_PASSWORD"],
        h=os.environ.get("POSTGRES_HOST", "superset-db"), d=os.environ["POSTGRES_DB"])
)

# ---- Caching ---------------------------------------------------------------------------
# In-process caches: a restart (or "Force refresh" on a dashboard) invalidates them, which is
# exactly the refresh procedure after rebuilding the serving snapshot.
CACHE_CONFIG = {"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300, "CACHE_KEY_PREFIX": "nfl_lab_"}
DATA_CACHE_CONFIG = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": int(os.environ.get("LAB_DATA_CACHE_SECONDS", "3600")),
    "CACHE_KEY_PREFIX": "nfl_lab_data_",
}

# ---- Query guards (dashboards never pull unbounded rows) -------------------------------
ROW_LIMIT = 10_000
SAMPLES_ROW_LIMIT = 1_000
SQL_MAX_ROW = 50_000
DISPLAY_MAX_ROW = 10_000
SQLLAB_TIMEOUT = 120
SUPERSET_WEBSERVER_TIMEOUT = 120
CSV_EXPORT = {"encoding": "utf-8"}

# ---- Features --------------------------------------------------------------------------
FEATURE_FLAGS = {
    "DASHBOARD_CROSS_FILTERS": True,
    "ENABLE_TEMPLATE_PROCESSING": False,
    "ALERT_REPORTS": False,
}

# ---- Colours: consistent with src/nflbdb/data/reporting.py CATEGORICAL ------------------
EXTRA_CATEGORICAL_COLOR_SCHEMES = [{
    "id": "nflLab",
    "description": "NFL Analytics Lab — position colours match the Phase 01/02 figures",
    "label": "NFL Lab",
    "isDefault": True,
    "colors": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
}]
