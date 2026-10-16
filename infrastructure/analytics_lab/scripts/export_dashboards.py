"""Export the CURRENT Superset dashboards (including UI edits) into version control.

Run from infrastructure/analytics_lab with the stack up:

    python scripts/export_dashboards.py

Steps (all via `docker compose`, nothing else is touched):
  1. `superset export-dashboards` inside the running `superset` container → /tmp/<ts>.zip
  2. `docker compose cp` the zip to superset/dashboards/exports/<ts>/dashboards_<ts>.zip
  3. unzip next to it so the YAML diff is reviewable in git
Database passwords are never included in Superset exports (and the DuckDB connection has none).
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("$", " ".join(args))
    subprocess.run(args, cwd=HERE, check=True)


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    remote = f"/tmp/dashboards_{ts}.zip"
    out_dir = HERE / "superset" / "dashboards" / "exports" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / f"dashboards_{ts}.zip"
    run("docker", "compose", "exec", "-T", "superset", "superset", "export-dashboards", "--dashboard-file", remote)
    run("docker", "compose", "cp", f"superset:{remote}", str(local))
    run("docker", "compose", "exec", "-T", "superset", "rm", "-f", remote)
    with zipfile.ZipFile(local) as zf:
        zf.extractall(out_dir / "unzipped")
    print(f"exported → {local.relative_to(HERE)} (+ unzipped/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
