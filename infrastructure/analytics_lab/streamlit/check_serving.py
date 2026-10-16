"""Startup gate (compose service `serving-check`): the serving snapshot exists and opens READ-ONLY.

    python check_serving.py [path]          # default: $LAB_SERVING_DB

Exit 0 = OK; exit 1 = actionable message on stderr. `superset-init` and `streamlit` depend on this
service (`service_completed_successfully`), so neither starts against a missing or broken file.
Never writes: opens with read_only=True and enable_external_access=false.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT = "/srv/nfl-lab/serving/nfl_lab.duckdb"
HINT = ("Build it on the host (project root): python -m nflbdb.lab.build --steps serving,monitoring\n"
        "Then check NFL_BDB_PROJECT_ROOT in infrastructure/analytics_lab/.env: its warehouse/serving folder is "
        "mounted at /srv/nfl-lab/serving.")


def check(path: Path) -> list[str]:
    """Problems found ([] = OK)."""
    if not path.parent.is_dir():
        return [f"{path.parent} is not mounted (no such directory)"]
    if not path.is_file():
        present = sorted(p.name for p in path.parent.iterdir())
        return [f"{path} not found (folder contains: {present or 'nothing'})"]
    if not os.access(path, os.R_OK):
        return [f"{path} is not readable by uid {os.getuid() if hasattr(os, 'getuid') else '?'}"]
    if path.stat().st_size == 0:
        return [f"{path} is empty (interrupted build?)"]
    with path.open("rb") as fh:
        header = fh.read(12)
    if header[8:12] != b"DUCK":
        return [f"{path} is not a DuckDB database file"]
    import duckdb  # noqa: PLC0415  (only after the cheap checks)

    try:
        con = duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})
        try:
            n = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'lab'").fetchone()[0]
            has_catalog = con.execute("SELECT count(*) FROM information_schema.tables "
                                      "WHERE table_schema = 'lab' AND table_name = 'mart_catalog'").fetchone()[0]
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001  (storage-version mismatch, lock, corruption → all actionable)
        return [f"{path} cannot be opened read-only by duckdb {duckdb.__version__}: {exc}"]
    problems = []
    if n == 0:
        problems.append(f"{path} has no tables in schema 'lab'")
    elif not has_catalog:
        problems.append(f"{path} has no lab.mart_catalog (incomplete snapshot)")
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0] if argv else os.environ.get("LAB_SERVING_DB", DEFAULT))
    problems = check(path)
    if problems:
        print("[serving-check] FAILED:\n  " + "\n  ".join(problems) + "\n" + HINT, file=sys.stderr)
        return 1
    print(f"[serving-check] OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
