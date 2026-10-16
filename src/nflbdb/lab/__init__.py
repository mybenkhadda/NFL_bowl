"""Analytics Lab — compact dashboard marts, a self-contained DuckDB serving snapshot and a
SQLite monitoring snapshot for the Dockerised Superset / Streamlit / Grafana stack.

Entry point (run by the USER on the host, conda env ``nfl``)::

    set PYTHONPATH=src & set NFL_BDB_DUCKDB_MEMORY=3GB
    python -m nflbdb.lab.build            # --force rebuilds every mart

See ``infrastructure/analytics_lab/README.md``.
"""
