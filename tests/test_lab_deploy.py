"""Analytics Lab · deployment regressions (Superset 5.0 import keys, bundle references, compose gating,
serving-snapshot startup check, Windows lock-safe snapshot replacement).

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_lab_deploy.py -q
No Docker, no warehouse: pure files + in-memory / tmp DuckDB.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import pytest

from nflbdb.lab import serving
from nflbdb.lab.serving import ServingFileLockedError, ensure_replaceable, replace_snapshot

LAB = Path(__file__).resolve().parents[1] / "infrastructure" / "analytics_lab"
MOUNT = "/srv/nfl-lab/serving"


def _bundle():
    sys.path.insert(0, str(LAB / "superset" / "bootstrap"))
    import build_bundle  # noqa: PLC0415

    return build_bundle


def _check_serving():
    sys.path.insert(0, str(LAB / "streamlit"))
    import check_serving  # noqa: PLC0415

    return check_serving


# ----------------------------------------------------------------------------- Superset bundle
def test_database_yaml_has_superset5_import_keys():
    bb = _bundle()
    files = bb.bundle_files()
    db = next(json.loads(t) for p, t in files.items() if "/databases/" in p)
    assert db["allow_csv_upload"] is False                 # KeyError: 'allow_csv_upload' regression
    assert all(k in db for k in bb.DATABASE_REQUIRED_KEYS)
    assert db["sqlalchemy_uri"] == f"duckdb:///{MOUNT}/nfl_lab.duckdb"
    cfg = db["extra"]["engine_params"]["connect_args"]
    assert cfg["read_only"] is True and cfg["config"]["enable_external_access"] is False


def test_generated_bundle_references_resolve():
    bb = _bundle()
    assert bb.check() == []
    assert bb.check_bundle(bb.bundle_files()) == []


def test_check_bundle_detects_broken_references():
    bb = _bundle()
    files = bb.bundle_files()
    db_path = next(p for p in files if "/databases/" in p)
    chart_path = next(p for p in files if "/charts/" in p)
    db = json.loads(files[db_path])
    del db["allow_csv_upload"]
    chart = json.loads(files[chart_path])
    chart["dataset_uuid"] = "00000000-0000-0000-0000-000000000000"
    broken = {**files, db_path: json.dumps(db), chart_path: json.dumps(chart)}
    problems = bb.check_bundle(broken)
    assert any("allow_csv_upload" in p for p in problems)
    assert any("dataset_uuid not in bundle" in p for p in problems)


def test_bundle_cli_check_exits_zero():
    assert _bundle().main(["--check"]) == 0


# ----------------------------------------------------------------------------- compose + init
def test_compose_gates_services_on_serving_check():
    yaml = pytest.importorskip("yaml")
    c = yaml.safe_load((LAB / "compose.yaml").read_text(encoding="utf-8"))
    s = c["services"]
    gate = {"condition": "service_completed_successfully"}
    assert s["serving-check"]["command"][-1] == "/app/check_serving.py"
    assert s["serving-check"]["restart"] == "no"
    assert s["superset-init"]["depends_on"]["serving-check"] == gate
    assert s["streamlit"]["depends_on"]["serving-check"] == gate
    assert s["superset"]["depends_on"]["superset-init"] == gate
    for name in ("serving-check", "superset-init", "superset", "streamlit"):
        mounts = [v for v in s[name]["volumes"] if isinstance(v, str) and v.endswith(f":{MOUNT}:ro")]
        assert len(mounts) == 1 and mounts[0].startswith("${NFL_BDB_PROJECT_ROOT"), name
    env = s["superset"]["environment"]
    assert env["LAB_SERVING_DB"] == f"{MOUNT}/nfl_lab.duckdb" == s["streamlit"]["environment"]["LAB_SERVING_DB"]
    assert env["SERVER_WORKER_AMOUNT"] == "${SUPERSET_WORKERS:-1}"


def test_init_script_is_strict_and_idempotent():
    text = (LAB / "superset" / "bootstrap" / "init.sh").read_text(encoding="utf-8")
    assert "list-users 2>/dev/null | grep" not in text      # SIGPIPE + pipefail → duplicate admin attempt
    assert "read_only=True" in text and "exit 1" in text     # snapshot must open before import / start
    # marker written only after a successful import (set -e aborts before `touch` on failure)
    assert text.index('touch "${MARKER_DIR}/imported_') > text.index("superset import-dashboards")


# ----------------------------------------------------------------------------- serving-check
def _snapshot(path: Path, with_catalog: bool = True) -> Path:
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA lab")
    con.execute("CREATE TABLE lab.athlete AS SELECT 1 AS nfl_id")
    if with_catalog:
        con.execute("CREATE TABLE lab.mart_catalog AS SELECT 'athlete' AS table_name")
    con.close()
    return path


def test_check_serving_ok(tmp_path):
    cs = _check_serving()
    assert cs.check(_snapshot(tmp_path / "nfl_lab.duckdb")) == []
    assert cs.main([str(tmp_path / "nfl_lab.duckdb")]) == 0


@pytest.mark.parametrize("case", ["missing", "empty", "not_duckdb", "no_catalog", "no_mount"])
def test_check_serving_reports_problems(tmp_path, case):
    cs = _check_serving()
    p = tmp_path / "nfl_lab.duckdb"
    if case == "empty":
        p.write_bytes(b"")
    elif case == "not_duckdb":
        p.write_bytes(b"x" * 64)
    elif case == "no_catalog":
        _snapshot(p, with_catalog=False)
    elif case == "no_mount":
        p = tmp_path / "absent_dir" / "nfl_lab.duckdb"
    assert cs.check(p), case
    assert cs.main([str(p)]) == 1


# ----------------------------------------------------------------------------- lock-safe replacement
def test_replace_snapshot_keeps_old_file_when_locked(tmp_path, monkeypatch):
    target, tmp = tmp_path / "nfl_lab.duckdb", tmp_path / ".nfl_lab.duckdb.abc.tmp"
    target.write_bytes(b"OLD")
    tmp.write_bytes(b"NEW")
    calls = []

    def locked(src, dst):
        calls.append(1)
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(serving.os, "replace", locked)
    with pytest.raises(ServingFileLockedError, match="docker compose stop"):
        replace_snapshot(tmp, target, attempts=3, delay_s=0)
    assert len(calls) == 3
    assert target.read_bytes() == b"OLD" and not tmp.exists()


def test_replace_snapshot_swaps_when_free(tmp_path):
    target, tmp = tmp_path / "nfl_lab.duckdb", tmp_path / ".nfl_lab.duckdb.abc.tmp"
    target.write_bytes(b"OLD")
    tmp.write_bytes(b"NEW")
    replace_snapshot(tmp, target, attempts=1, delay_s=0)
    assert target.read_bytes() == b"NEW" and not tmp.exists()


def test_ensure_replaceable_detects_lock_without_touching_file(tmp_path, monkeypatch):
    target = tmp_path / "nfl_lab.duckdb"
    target.write_bytes(b"OLD")
    ensure_replaceable(target)                           # free → no error, file intact
    assert target.read_bytes() == b"OLD" and sorted(os.listdir(tmp_path)) == ["nfl_lab.duckdb"]

    def locked(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(serving.os, "rename", locked)
    with pytest.raises(ServingFileLockedError):
        ensure_replaceable(target)
    assert target.read_bytes() == b"OLD"


def test_ensure_replaceable_refuses_stale_wal(tmp_path):
    target = tmp_path / "nfl_lab.duckdb"
    target.write_bytes(b"OLD")
    (tmp_path / "nfl_lab.duckdb.wal").write_bytes(b"W")
    with pytest.raises(ServingFileLockedError, match=r"\.wal exists"):
        ensure_replaceable(target)
    ensure_replaceable(tmp_path / "absent.duckdb")       # no file, no wal → nothing to check
