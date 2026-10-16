"""Generate the Superset import bundle from dashboards_spec.py (stdlib only).

    python build_bundle.py --zip out.zip            # bundle for `superset import-dashboards`
    python build_bundle.py --out-dir DIR            # unzipped, for review / diffs
    python build_bundle.py --print-hash             # content hash (drives LAB_IMPORT_MODE=on_change)
    python build_bundle.py --check                  # static validation only (no files written)

Runs inside the Superset container (superset-init) or on the host with any Python ≥ 3.9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dashboards_spec as spec  # noqa: E402
from bundle_lib import (BUNDLE_VERSION, DATABASE_REQUIRED_KEYS, chart_yaml, dashboard_yaml,  # noqa: E402
                        database_yaml, dataset_yaml, slug)

ROOT = "nfl_lab_bundle"


def bundle_files() -> dict[str, str]:
    """Relative path → file content (JSON, valid YAML)."""
    dump = lambda obj: json.dumps(obj, indent=2) + "\n"  # noqa: E731
    db = database_yaml(spec.DATABASE_NAME, spec.DATABASE_URI)
    files = {f"{ROOT}/metadata.yaml": dump({"version": BUNDLE_VERSION, "type": "Dashboard",
                                            "timestamp": "2026-10-08T00:00:00+00:00"}),
             f"{ROOT}/databases/{slug(spec.DATABASE_NAME)}.yaml": dump(db)}
    for ds in spec.DATASETS:
        files[f"{ROOT}/datasets/{slug(spec.DATABASE_NAME)}/{ds.table}.yaml"] = dump(dataset_yaml(ds, db["uuid"]))
    charts = {}
    for d in spec.DASHBOARDS:
        for c in d.charts():
            charts[c.key] = c
        files[f"{ROOT}/dashboards/{d.slug}.yaml"] = dump(dashboard_yaml(d))
    for key, c in sorted(charts.items()):
        files[f"{ROOT}/charts/{key}_{c.uuid[:8]}.yaml"] = dump(chart_yaml(c))
    return files


def content_hash(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.encode())
        h.update(files[path].encode())
    return h.hexdigest()[:12]


def check() -> list[str]:
    """Static checks: unique keys/titles, row widths ≤ 12, filter/chart datasets declared, columns declared."""
    problems = []
    declared = {ds.key: {c.partition(":")[0] for c in ds.columns} for ds in spec.DATASETS}
    seen_keys: dict[str, str] = {}
    for d in spec.DASHBOARDS:
        for r, row in enumerate(d.rows):
            width = sum(item[2] for item in row)
            if width > 12:
                problems.append(f"{d.key} row {r}: width {width} > 12")
        for c in d.charts():
            if c.dataset.key not in declared:
                problems.append(f"chart {c.key}: dataset {c.dataset.key} not in DATASETS")
            if seen_keys.setdefault(c.key, c.title) != c.title:
                problems.append(f"chart key {c.key} reused with a different title")
            for col_param in ("x_axis", "column", "entity", "series"):
                col = c.params.get(col_param)
                if isinstance(col, str) and col not in declared.get(c.dataset.key, set()):
                    problems.append(f"chart {c.key}: {col_param}={col!r} not declared on {c.dataset.key}")
            as_list = lambda v: [v] if isinstance(v, str) else list(v or [])  # noqa: E731 (heatmap groupby is a str)
            for col in as_list(c.params.get("groupby")) + as_list(c.params.get("all_columns")) + \
                    as_list(c.params.get("columns")):
                if col not in declared.get(c.dataset.key, set()):
                    problems.append(f"chart {c.key}: column {col!r} not declared on {c.dataset.key}")
        for f in d.filters:
            if f.column not in declared.get(f.dataset.key, set()):
                problems.append(f"{d.key} filter {f.name}: {f.column!r} not declared on {f.dataset.key}")
    return problems


def check_bundle(files: dict[str, str]) -> list[str]:
    """Import compatibility of the GENERATED files: required database keys (Superset 5.0) and every
    uuid reference (dataset → database, chart → dataset, dashboard → chart / filter dataset) resolves."""
    problems = []
    docs = {p: json.loads(t) for p, t in files.items()}
    kind = lambda k: {p: d for p, d in docs.items() if f"/{k}/" in p}  # noqa: E731
    dbs, dss, charts, dashes = kind("databases"), kind("datasets"), kind("charts"), kind("dashboards")
    if len(dbs) != 1:
        problems.append(f"expected exactly 1 database file, found {len(dbs)}")
    for p, d in dbs.items():
        problems += [f"{p}: missing key {k!r}" for k in DATABASE_REQUIRED_KEYS if k not in d]
        if not isinstance(d.get("extra"), dict):
            problems.append(f"{p}: 'extra' must be a mapping")
    db_uuids = {d["uuid"] for d in dbs.values()}
    ds_uuids = {d["uuid"] for d in dss.values()}
    chart_uuids = {d["uuid"] for d in charts.values()}
    all_uuids = [d["uuid"] for d in docs.values() if "uuid" in d]
    if len(all_uuids) != len(set(all_uuids)):
        problems.append("duplicate uuids across bundle files")
    problems += [f"{p}: database_uuid not in bundle" for p, d in dss.items() if d.get("database_uuid") not in db_uuids]
    problems += [f"{p}: dataset_uuid not in bundle" for p, d in charts.items() if d.get("dataset_uuid") not in ds_uuids]
    for p, d in dashes.items():
        for node in d["position"].values():
            if isinstance(node, dict) and node.get("type") == "CHART" and node["meta"].get("uuid") not in chart_uuids:
                problems.append(f"{p}: chart {node['id']} references unknown chart uuid")
        for f in d["metadata"].get("native_filter_configuration", []):
            problems += [f"{p}: filter {f['name']} targets unknown dataset" for t in f["targets"]
                         if t.get("datasetUuid") not in ds_uuids]
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--print-hash", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    problems = check()
    if problems:
        print("bundle spec problems:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    files = bundle_files()
    problems = check_bundle(files)
    if problems:
        print("bundle import-compatibility problems:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    if a.print_hash:
        print(content_hash(files))
        return 0
    if a.check:
        print(f"ok: {len(spec.DASHBOARDS)} dashboards, {len(spec.DATASETS)} datasets, "
              f"{sum(1 for p in files if '/charts/' in p)} charts, hash {content_hash(files)}")
        return 0
    if a.zip:
        a.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for path, text in sorted(files.items()):
                zf.writestr(path, text)
        print(f"wrote {a.zip} ({len(files)} files, hash {content_hash(files)})")
    if a.out_dir:
        for path, text in files.items():
            target = a.out_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        print(f"wrote {len(files)} files under {a.out_dir / ROOT}")
    if not (a.zip or a.out_dir):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
