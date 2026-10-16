"""Bundle discovery / builder / zip on a synthetic research layout (no real data). Authored, not executed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_spec = importlib.util.spec_from_file_location("_nflbowl_test_data", Path(__file__).with_name("conftest.py"))   # tests/ is not a package
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_arrays = _mod.make_arrays

KEY = "abc123def4567890"
GOLD = ("warehouse", "gold", "advanced_movement")


def _sha(p):
    from nflbowl.artifacts.store import sha256_file

    return sha256_file(p)


def write_store(area, key, X, M, meta):
    d = area / "sequence_shards" / key
    d.mkdir(parents=True)
    for name, arr in (("X.npy", X), ("M.npy", M)):
        with open(d / name, "wb") as f:
            np.save(f, arr)
    meta.write_parquet(d / "meta.parquet")
    (d / "COMPLETE.json").write_text(json.dumps({"key": key, "n": int(X.shape[0]), "L": int(X.shape[1]),
                                                 "channels": ["speed", "a_tan", "a_n", "yaw", "dt", "heading_valid"],
                                                 "sha_meta": _sha(d / "meta.parquet")}))
    return d


@pytest.fixture()
def fake_repo(tmp_path, monkeypatch):
    """Research layout of ResearchStore (full mode) with every input of a bundle."""
    from nflbowl.config import stable_hash

    monkeypatch.delenv("NFL_BDB_WAREHOUSE", raising=False)
    X, M, meta, folds, athletes = make_arrays()
    gold = tmp_path.joinpath(*GOLD)
    area = gold / "advanced_v1"
    write_store(area, KEY, X, M, meta)
    (area / "m12f_sequence_store.json").write_text(json.dumps({"key": KEY, "run_mode": "full"}))
    sig = stable_hash(folds.sort("nfl_id").select("nfl_id", "fold").rows(), 16)
    folds.with_columns(pl.lit("m11").alias("fold_source"), pl.lit(sig).alias("fold_signature"), pl.lit("cfg0").alias("config_hash")
                       ).write_parquet(area / "m12_fold_manifest.parquet")
    (area / "m12_foundation_manifest.json").write_text(json.dumps({"fold_signature": sig, "config_hash": "cfg0", "run_mode": "full",
                                                                    "n_linked": int(athletes.height)}))
    athletes.select("nfl_id", "linked").write_parquet(area / "m12_cohort_registry.parquet")
    athletes.select("nfl_id", "position_group", "nfl.q90_shared").write_parquet(area / "m12c_targets.parquet")
    (area / "m12c_target_freeze.json").write_text(json.dumps({"config_hash": "cfg0"}))
    (gold / "discovery_v1").mkdir(parents=True)
    athletes.select("nfl_id", "position_group", "draft_year", "forty", "ten", "weight", "height").write_parquet(
        gold / "discovery_v1" / "m11_officials.parquet")
    (gold / "transfer").mkdir(parents=True)
    ids = athletes["nfl_id"].to_list()
    pl.DataFrame({"scheme": ["outer"] * (2 * len(ids)), "family": ["TCN"] * (2 * len(ids)), "fold": [i % 5 for i in range(len(ids))] * 2,
                  "seq_id": [f"s{i}" for i in range(2 * len(ids))], "nfl_id": ids * 2, "dim": [0] * len(ids) + [1] * len(ids),
                  "value": np.linspace(0, 1, 2 * len(ids)), "role": ["test"] * (2 * len(ids))}).write_parquet(
        gold / "transfer" / "transfer_attempt_representations_outer.parquet")
    return tmp_path


# ------------------------------------------------------------------------------ discovery
def test_discovery_finds_every_source(fake_repo):
    from nflbowl.data.discovery import discover_bundle_sources

    s = discover_bundle_sources(fake_repo)
    assert s.can_build and s.selected.key == KEY and s.selected.run_mode == "full" and s.selected.pointer_selected
    assert s.selected.x_shape == [280, 16, 6] and s.selected.m_shape == [280, 16]
    assert all(s.records[r].available for r in ("fold_manifest", "officials", "cohort_registry", "transfer_targets", "tcn_representations"))
    assert "bundle can be built  : YES" in s.summary()
    assert any(r["role"] == "sequence_store" and r["selected"] for r in s.rows())


def test_npy_header_reads_shape_without_loading(tmp_path):
    from nflbowl.data.discovery import npy_header

    with open(tmp_path / "a.npy", "wb") as f:
        np.save(f, np.zeros((3, 4, 6), np.float32))
    assert npy_header(tmp_path / "a.npy") == ((3, 4, 6), np.dtype("float32"), False)


def test_missing_fold_manifest_names_the_research_step(fake_repo):
    from nflbowl.data.builder import MissingSourcesError, build_bundle_from_repo
    from nflbowl.data.discovery import discover_bundle_sources

    fake_repo.joinpath(*GOLD, "advanced_v1", "m12_fold_manifest.parquet").unlink()
    s = discover_bundle_sources(fake_repo)
    assert not s.can_build and any("12.0" in m for m in s.missing)
    with pytest.raises(MissingSourcesError, match="fold manifest"):
        build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)
    assert not fake_repo.joinpath("bundles", "m12f_v1", "bundle.json").exists()


def test_two_full_stores_require_explicit_choice(fake_repo):
    from nflbowl.data.builder import MissingSourcesError, build_bundle_from_repo
    from nflbowl.data.discovery import discover_bundle_sources

    X, M, meta, _, _ = make_arrays(seed=1)
    other = write_store(fake_repo.joinpath(*GOLD, "advanced_v1"), "ffff000011112222", X, M, meta)
    s = discover_bundle_sources(fake_repo)
    assert s.selected is None and "ambiguous" in s.selection
    with pytest.raises(MissingSourcesError, match="--source-sequence-store"):
        build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)
    s2 = discover_bundle_sources(fake_repo, sequence_store=other)
    assert s2.selected is not None and s2.selected.key == "ffff000011112222"
    assert any("pointer" in w for w in s2.warnings)


def test_smoke_store_is_never_auto_selected(tmp_path, monkeypatch):
    from nflbowl.data.discovery import discover_bundle_sources

    monkeypatch.delenv("NFL_BDB_WAREHOUSE", raising=False)
    X, M, meta, _, _ = make_arrays()
    write_store(tmp_path.joinpath(*GOLD, "_sample", "advanced_v1"), KEY, X, M, meta)
    s = discover_bundle_sources(tmp_path)
    assert s.selected is None and any("Smoke" in m for m in s.missing)


def test_incomplete_store_is_reported(fake_repo):
    from nflbowl.data.discovery import discover_bundle_sources

    fake_repo.joinpath(*GOLD, "advanced_v1", "sequence_shards", KEY, "COMPLETE.json").unlink()
    s = discover_bundle_sources(fake_repo)
    assert not s.can_build and any("INCOMPLETE" in line or "interrupted" in line for line in s.summary().splitlines())


# ------------------------------------------------------------------------------ build
def test_build_copies_byte_for_byte_and_opens(fake_repo):
    from nflbowl.data.builder import build_bundle_from_repo
    from nflbowl.data.loading import open_bundle

    res = build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)
    out = fake_repo / "bundles" / "m12f_v1"
    assert res.status == "built" and res.ok and res.bundle_json == out / "bundle.json"
    src = fake_repo.joinpath(*GOLD, "advanced_v1", "sequence_shards", KEY)
    for n in ("X.npy", "M.npy", "meta.parquet", "COMPLETE.json"):
        assert _sha(out / "sequence_store" / n) == _sha(src / n)
    b = open_bundle(out, verify=True, outer_folds=5)
    X, M, meta, folds, _ = make_arrays()
    assert np.array_equal(np.asarray(b.X), X) and np.array_equal(np.asarray(b.M), M) and b.meta.equals(meta)
    assert b.folds.equals(folds) and b.athletes is not None and "nfl.q90_shared" in b.athletes.columns and b.tcn is not None
    info = json.loads((out / "bundle.json").read_text())
    assert info["store_key"] == KEY and info["run_mode"] == "full" and info["transfer_target"] == "nfl.q90_shared"
    assert set(info["files"]) == {"sequence_store/X.npy", "sequence_store/M.npy", "sequence_store/meta.parquet", "sequence_store/COMPLETE.json",
                                  "folds.parquet", "athletes.parquet", "tcn_representations.parquet"}
    assert info["source"]["inputs"]["store/X.npy"]["sha256"] == info["files"]["sequence_store/X.npy"]["sha256"]
    assert not any(p.name.startswith(".") for p in out.rglob("*"))                  # no marker / temporaries left
    assert not (fake_repo / "bundles" / ".m12f_v1.building").exists()
    s = res.summary
    assert s["n_sequences"] == 280 and s["n_combine"] == 120 and s["n_nfl"] == 160 and s["folds"]["athletes_fallback_fold"] == 2
    rd = fake_repo / "bundles" / "m12f_v1.reports"
    assert all((rd / f).is_file() for f in ("build_report.json", "checks.csv", "plan.csv", "discovery.json", "discovery.csv", "BUILD_REPORT.md"))
    assert {c.name for c in res.checks} >= {"store.meta_sha256", "store.mask_padding", "folds.matches_foundation", "bundle.byte_identical"}


def test_builder_matches_write_bundle_content(fake_repo, tmp_path):
    """Same arrays / folds / athletes as the in-memory writer (one construction for both writers)."""
    from nflbowl.data.builder import build_bundle_from_repo
    from nflbowl.data.loading import open_bundle

    build_bundle_from_repo(fake_repo, "bundles/m12f_v1", include_tcn=False, progress=False)
    b = open_bundle(fake_repo / "bundles" / "m12f_v1")
    _, _, _, folds, athletes = make_arrays()
    cols = ["nfl_id", "position_group", "draft_year", "height", "weight", "forty", "ten", "linked", "nfl.q90_shared"]
    assert b.athletes.select(cols).sort("nfl_id").equals(athletes.select(cols).sort("nfl_id"))
    assert b.tcn is None


def test_deep_verify_build(fake_repo):
    from nflbowl.data.builder import build_bundle_from_repo

    res = build_bundle_from_repo(fake_repo, "bundles/m12f_v1", verify=True, progress=False)
    names = {c.name: c.status for c in res.checks}
    assert names["store.values_finite"] == "pass" and names["store.padding_zero"] == "pass" and res.summary["integrity"].startswith("sha256 verified")


def test_dry_run_writes_nothing(fake_repo):
    from nflbowl.data.builder import build_bundle_from_repo

    res = build_bundle_from_repo(fake_repo, "bundles/m12f_v1", dry_run=True, progress=False)
    assert res.status == "dry_run" and [p.dest for p in res.plan][:4] == ["sequence_store/X.npy", "sequence_store/M.npy",
                                                                          "sequence_store/meta.parquet", "sequence_store/COMPLETE.json"]
    assert not (fake_repo / "bundles").exists()


def test_reuse_overwrite_and_refusal(fake_repo):
    from nflbowl.data.builder import build_bundle_from_repo

    first = build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)
    again = build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)
    assert again.status == "up_to_date" and again.fingerprint == first.fingerprint
    with pytest.raises(FileExistsError):                                            # other options → other signature
        build_bundle_from_repo(fake_repo, "bundles/m12f_v1", include_tcn=False, progress=False)
    with pytest.raises(FileExistsError):
        build_bundle_from_repo(fake_repo, "bundles/m12f_v1", reuse_existing=False, progress=False)
    third = build_bundle_from_repo(fake_repo, "bundles/m12f_v1", include_tcn=False, overwrite=True, progress=False)
    assert third.status == "built" and third.replaced is not None and (third.replaced / "bundle.json").is_file()   # old bundle kept


def test_refuses_foreign_non_empty_output(fake_repo):
    from nflbowl.data.builder import build_bundle_from_repo

    d = fake_repo / "bundles" / "m12f_v1"
    d.mkdir(parents=True)
    (d / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError, match="not a bundle"):
        build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)
    assert (d / "notes.txt").read_text() == "mine"


def test_fold_signature_mismatch_is_incompatible(fake_repo):
    from nflbowl.data.builder import IncompatibleSourcesError, build_bundle_from_repo

    p = fake_repo.joinpath(*GOLD, "advanced_v1", "m12_foundation_manifest.json")
    p.write_text(json.dumps({**json.loads(p.read_text()), "fold_signature": "0000000000000000"}))
    with pytest.raises(IncompatibleSourcesError, match="folds.matches_foundation"):
        build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)
    assert not (fake_repo / "bundles" / "m12f_v1" / "bundle.json").exists()
    assert (fake_repo / "bundles" / "m12f_v1.reports" / "checks.csv").is_file()


def test_modified_meta_fails_sha_check(fake_repo):
    from nflbowl.data.builder import IncompatibleSourcesError, build_bundle_from_repo

    p = fake_repo.joinpath(*GOLD, "advanced_v1", "sequence_shards", KEY, "meta.parquet")
    pl.read_parquet(p).with_columns(pl.col("kind").str.to_uppercase()).write_parquet(p)
    with pytest.raises(IncompatibleSourcesError, match="store.meta_sha256"):
        build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)


def test_transfer_parts_are_optional(fake_repo):
    from nflbowl.data.builder import build_bundle_from_repo
    from nflbowl.data.loading import open_bundle

    fake_repo.joinpath(*GOLD, "discovery_v1", "m11_officials.parquet").unlink()
    res = build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)
    b = open_bundle(fake_repo / "bundles" / "m12f_v1")
    assert res.status == "built" and b.athletes is None and b.tcn is not None
    assert any(c.name == "athletes" and c.status == "skip" for c in res.checks)


def test_csv_fold_manifest_fallback(fake_repo):
    from nflbowl.data.builder import build_bundle_from_repo
    from nflbowl.data.loading import open_bundle

    area = fake_repo.joinpath(*GOLD, "advanced_v1")
    csv = fake_repo / "folds_view.csv"
    pl.read_parquet(area / "m12_fold_manifest.parquet").write_csv(csv)
    (area / "m12_fold_manifest.parquet").unlink()
    build_bundle_from_repo(fake_repo, "bundles/m12f_v1", fold_manifest=csv, progress=False)
    b = open_bundle(fake_repo / "bundles" / "m12f_v1")
    assert b.folds["nfl_id"].dtype == pl.Int64 and b.folds.equals(make_arrays()[3])


# ------------------------------------------------------------------------------ zip / CLI / hints
def test_zip_and_extract_roundtrip(fake_repo, tmp_path):
    from nflbowl.data.builder import build_bundle_from_repo, extract_bundle_zip, zip_bundle

    res = build_bundle_from_repo(fake_repo, "bundles/m12f_v1", progress=False)
    zp = zip_bundle(fake_repo / "bundles" / "m12f_v1", progress=False)
    assert zp.name == "m12f_v1.zip"
    import zipfile

    with zipfile.ZipFile(zp) as z:
        assert "m12f_v1/bundle.json" in z.namelist() and all(n.startswith("m12f_v1/") for n in z.namelist())
    root = extract_bundle_zip(zp, tmp_path / "drive" / "bundles", verify=True)
    assert root == tmp_path / "drive" / "bundles" / "m12f_v1"
    from nflbowl.data.loading import open_bundle

    assert open_bundle(root).fingerprint == res.fingerprint
    assert extract_bundle_zip(zp, tmp_path / "drive" / "bundles") == root                # idempotent


def test_extract_rejects_path_traversal(tmp_path):
    import zipfile

    from nflbowl.data.builder import extract_bundle_zip
    from nflbowl.data.validation import SchemaError

    zp = tmp_path / "evil.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("../escape.txt", "x")
        z.writestr("bundle.json", "{}")
    with pytest.raises(SchemaError, match="unsafe"):
        extract_bundle_zip(zp, tmp_path / "dest")


def test_cli_discover_dry_run_build_validate(fake_repo, capsys):
    from nflbowl.data.builder import main

    assert main(["--repo-root", str(fake_repo), "--discover-only"]) == 0
    assert "bundle can be built  : YES" in capsys.readouterr().out
    assert main(["--repo-root", str(fake_repo), "--output", "bundles/m12f_v1", "--dry-run", "--no-progress"]) == 0
    assert not (fake_repo / "bundles").exists()
    assert main(["--repo-root", str(fake_repo), "--output", "bundles/m12f_v1", "--no-progress", "--zip"]) == 0
    assert (fake_repo / "bundles" / "m12f_v1.zip").is_file()
    capsys.readouterr()
    assert main(["--validate", str(fake_repo / "bundles" / "m12f_v1"), "--verify"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["n_sequences"] == 280 and s["integrity"].startswith("sha256 verified")


def test_cli_missing_sources_exit_code(tmp_path, monkeypatch):
    from nflbowl.data.builder import main

    monkeypatch.delenv("NFL_BDB_WAREHOUSE", raising=False)
    assert main(["--repo-root", str(tmp_path), "--discover-only"]) == 2
    assert main(["--repo-root", str(tmp_path), "--output", "bundles/x", "--no-progress"]) == 2


def test_missing_bundle_hints(tmp_path):
    from nflbowl.data.loading import read_info

    (tmp_path / "m12f_v1.zip").write_bytes(b"PK")
    with pytest.raises(FileNotFoundError, match="extract_bundle_zip"):
        read_info(tmp_path / "m12f_v1")
    nested = tmp_path / "parent" / "m12f_v1"
    nested.mkdir(parents=True)
    (nested / "bundle.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="one level down"):
        read_info(tmp_path / "parent")
