# Superset + DuckDB driver + PostgreSQL metadata driver.
# Official "lean" images ship WITHOUT database drivers; drivers are added into the image's
# virtualenv (/app/.venv) with uv, as documented in the Superset docker-builds guide.
ARG SUPERSET_VERSION=5.0.0
FROM apache/superset:${SUPERSET_VERSION}

USER root
COPY requirements.txt /tmp/lab-requirements.txt
RUN . /app/.venv/bin/activate \
 && uv pip install --no-cache -r /tmp/lab-requirements.txt \
 && python -c "import duckdb, duckdb_engine, psycopg2; print('duckdb', duckdb.__version__)" \
 && rm /tmp/lab-requirements.txt

USER superset
