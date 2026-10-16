"""Refresh column metadata of every lab dataset from the DuckDB snapshot (runs in superset-init).

The bundle declares the columns charts use; this adds any other mart columns (so they appear in
Explore) and drops columns that no longer exist. Failures are reported, never fatal.
"""

from __future__ import annotations

import sys

DATABASE_NAME = "NFL Lab (DuckDB)"


def main() -> int:
    from superset.app import create_app

    app = create_app()
    with app.app_context():
        from superset import db
        from superset.connectors.sqla.models import SqlaTable
        from superset.models.core import Database

        database = db.session.query(Database).filter_by(database_name=DATABASE_NAME).one_or_none()
        if database is None:
            print(f"[post-import] database {DATABASE_NAME!r} not found — was the bundle imported?")
            return 1
        failures = 0
        for table in db.session.query(SqlaTable).filter_by(database_id=database.id).all():
            try:
                table.fetch_metadata()
                db.session.commit()
                print(f"[post-import] synced {table.schema}.{table.table_name} ({len(table.columns)} columns)")
            except Exception as exc:  # noqa: BLE001 — keep going, report at the end
                db.session.rollback()
                failures += 1
                print(f"[post-import] could not sync {table.table_name}: {type(exc).__name__}: {exc}")
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
