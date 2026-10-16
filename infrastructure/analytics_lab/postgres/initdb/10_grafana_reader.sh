#!/bin/bash
# Runs ONCE, when the superset_db volume is first initialised (official postgres entrypoint).
# Creates a read-only role so Grafana can chart Superset query/log durations.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v reader_pw="$GRAFANA_READER_PASSWORD" -v owner="$POSTGRES_USER" -v dbname="$POSTGRES_DB" <<'SQL'
CREATE ROLE grafana_reader LOGIN PASSWORD :'reader_pw';
GRANT CONNECT ON DATABASE :"dbname" TO grafana_reader;
GRANT USAGE ON SCHEMA public TO grafana_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_reader;
-- Superset creates its tables later (as the owner role): grant SELECT on those too.
ALTER DEFAULT PRIVILEGES FOR ROLE :"owner" IN SCHEMA public GRANT SELECT ON TABLES TO grafana_reader;
SQL
