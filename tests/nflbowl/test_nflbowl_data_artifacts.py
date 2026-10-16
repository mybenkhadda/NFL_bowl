"""Bundle schema validation, round trip, staging, fingerprints, manifest cache invalidation, splits, registry, devices."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest


# ------------------------------------------------------------------------------ schema validation
def test_validate_meta_errors(arrays):
    from nflbowl.data.validation import SchemaError, validate_meta

    _, _, meta, _, _ = arrays
    validate_meta(meta)
    with pytest.raises(SchemaError):
        validate_meta(meta.drop("kind"))
    with pytest.raises(SchemaError):
        validate_meta(meta.with_columns(pl.when(pl.col("idx") == 0).then(pl.lit("college")).otherwise(pl.col("domain")).alias("domain")))
    with pytest.raises(SchemaError):
        validate_meta(meta.reverse())                                   # idx not in row order
    with pytest.raises(SchemaError):                                    # one athlete with two positions
        validate_meta(meta.with_columns(pl.when(pl.col("idx") == 0).then(pl.lit("QB")).otherwise(pl.col("position_group")).alias("position_group")))


def test_validate_arrays_and_folds(arrays):
    from nflbowl.data.validation import SchemaError, validate_arrays, validate_folds

    X, M, meta, folds, _ = arrays
    validate_arrays(X, M, meta.height)
    with pytest.raises(SchemaError):
        validate_arrays(X[..., :5], M, meta.height)
    with pytest.raises(SchemaError):
        validate_arrays(X.astype(np.float64), M, meta.height)
    with pytest.raises(SchemaError):
        validate_arrays(X, M[:, :-1], meta.height)
    with pytest.raises(SchemaError):
        validate_arrays(X, M, meta.height + 1)
    with pytest.raises(SchemaError):
        validate_folds(pl.concat([folds, folds.head(1)]))
    with pytest.raises(SchemaError):
        validate_folds(folds.with_columns(pl.lit(7).alias("fold")), outer_folds=5)


# ------------------------------------------------------------------------------ bundle
def test_bundle_roundtrip_and_fingerprint(bundle_dir, arrays):
    from nflbowl.data.loading import open_bundle

    X, M, meta, _, _ = arrays
    b = open_bundle(bundle_dir, verify=True, outer_folds=5)
    assert np.array_equal(np.asarray(b.X), X) and np.array_equal(np.asarray(b.M), M)
    assert b.meta.equals(meta)
    assert b.athletes is not None and b.tcn is None
    assert open_bundle(bundle_dir).fingerprint == b.fingerprint
    s = b.summary()
    assert s["n_combine"] == 120 and s["n_nfl"] == 160 and s["n_athletes"] == 40


def test_bundle_staging(bundle_dir, tmp_path):
    from nflbowl.data.loading import open_bundle

    b = open_bundle(bundle_dir, stage_dir=tmp_path / "stage")
    assert b.staged_from == bundle_dir and b.root != bundle_dir
    assert (b.root / ".staged.json").is_file()
    assert open_bundle(bundle_dir, stage_dir=tmp_path / "stage").root == b.root          # second call reuses the staged copy


def test_bundle_detects_truncation_and_tampering(bundle_dir):
    from nflbowl.data.loading import open_bundle
    from nflbowl.data.validation import SchemaError

    p = bundle_dir / "folds.parquet"
    df = pl.read_parquet(p).with_columns(pl.col("fold").reverse())
    df.write_parquet(p)
    with pytest.raises(SchemaError):
        open_bundle(bundle_dir)


def test_raw_store_directory_gets_a_hint(tmp_path):
    from nflbowl.data.loading import open_bundle

    (tmp_path / "COMPLETE.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="export_research_bundle"):
        open_bundle(tmp_path)


# ------------------------------------------------------------------------------ manifest / cache
def test_manifest_find_respects_deps(tmp_path):
    from nflbowl.artifacts.manifest import Manifest
    from nflbowl.artifacts.store import atomic_json

    m = Manifest(tmp_path, "exp")
    f = atomic_json({"x": 1}, tmp_path / "a.json")
    m.register("embeddings", "F4__transductive", f, {"bundle": "b1", "science": "s1"})
    assert m.find("embeddings", "F4__transductive", {"bundle": "b1", "science": "s1"}) is not None
    assert m.find("embeddings", "F4__transductive", {"bundle": "b2", "science": "s1"}) is None        # other inputs → never reused
    assert m.find("embeddings", "F4__transductive") is not None
    f.unlink()
    assert m.find("embeddings", "F4__transductive") is None                                           # missing file → not found
    m2 = Manifest(tmp_path, "exp")                                                                    # persisted
    assert len(m2.artifacts("embeddings")) == 1


def test_manifest_verify_detects_modified_file(tmp_path):
    from nflbowl.artifacts.manifest import Manifest
    from nflbowl.artifacts.store import atomic_json

    m = Manifest(tmp_path, "exp")
    f = atomic_json({"x": 1}, tmp_path / "a.json")
    m.register("k", "a", f, {"d": 1})
    f.write_text(json.dumps({"x": 2}))
    assert m.find("k", "a", {"d": 1}, verify=True) is None


def test_jobs(tmp_path):
    from nflbowl.artifacts.manifest import Manifest

    m = Manifest(tmp_path, "exp")
    m.set_job("train:F4:transductive", status="running", model="F4_physics_contrastive")
    assert [j["id"] for j in m.jobs("running")] == ["train:F4:transductive"]
    m.set_job("train:F4:transductive", status="finished")
    assert not m.jobs("running") and m.job("train:F4:transductive")["model"] == "F4_physics_contrastive"


def test_atomic_write_never_leaves_partial_file(tmp_path):
    from nflbowl.artifacts.store import atomic_write

    target = tmp_path / "x.bin"

    def boom(p):
        p.write_bytes(b"half")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        atomic_write(target, boom)
    assert not target.exists() and not list(tmp_path.glob(".*tmp"))
    atomic_write(target, lambda p: p.write_bytes(b"one"))
    atomic_write(target, lambda p: p.write_bytes(b"two"), overwrite=False)
    assert target.read_bytes() == b"one"


# ------------------------------------------------------------------------------ splits
def test_fold_map_fallback(arrays):
    from nflbowl.data.splits import fold_map

    _, _, meta, folds, _ = arrays
    fm = fold_map(meta, folds, 5)
    missing = set(meta["nfl_id"].unique().to_list()) - set(folds["nfl_id"].to_list())
    assert missing and all(fm[i] == i % 5 for i in missing)
    assert all(fm[int(r["nfl_id"])] == r["fold"] for r in folds.iter_rows(named=True))


@pytest.mark.parametrize("held", [None, 0, 3])
def test_split_integrity(arrays, held):
    from nflbowl.data.splits import fold_map, split_indices

    _, _, meta, folds, _ = arrays
    fm = fold_map(meta, folds, 5)
    tr, va, ho = split_indices(meta, fm, held, 0.1, 7)
    ids = meta["nfl_id"].to_numpy()
    assert not (set(ids[tr]) & set(ids[va])) and not (set(ids[tr]) & set(ids[ho]))
    if held is None:
        assert ho.size == 0
    else:
        assert ho.size and all(fm[int(i)] == held for i in ids[ho])
        # both domains of a held-out athlete are held out
        dom = meta["domain"].to_numpy()
        for i in set(ids[ho].tolist()):
            assert set(dom[ids == i]) <= set(dom[ho][ids[ho] == i])


def test_split_leak_is_detected(arrays):
    from nflbowl.data.splits import SplitLeakError, check_split_integrity, fold_map

    _, _, meta, folds, _ = arrays
    fm = fold_map(meta, folds, 5)
    n = meta.height
    tr = np.arange(0, n - 1)
    va = np.array([n - 1])
    ids = meta["nfl_id"].to_numpy()
    if ids[n - 1] in set(ids[tr]):
        with pytest.raises(SplitLeakError):
            check_split_integrity(meta, fm, tr, va, np.array([], int), None)


# ------------------------------------------------------------------------------ registry / devices
def test_registry_discovers_research_configs():
    from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se
    from nflbowl.models import registry as reg

    assert reg.families() == [e.name for e in se.FAMILIES]
    assert reg.ablations() == [e.name for e in se.ABLATIONS]
    assert reg.resolve("F4") == se.PRIMARY
    assert reg.spec("F4").primary and reg.spec("F4").trainable
    assert not reg.spec("F0").trainable and not reg.spec("TCN").trainable
    with pytest.raises(KeyError):
        reg.resolve("F9")
    t = reg.list_models()
    assert {"name", "kind", "group", "trainable"} <= set(t.columns)


def test_enc_config_scaling_matches_scale_config():
    from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se
    from nflbdb.research.advanced_movement.m12.config import M12Config
    from nflbowl.config import load_config
    from nflbowl.models import registry as reg

    sci = load_config("default").science
    assert reg.enc_config("F4", sci) == se.scale_config(se.by_name(se.PRIMARY), M12Config())


def test_device_selection_cpu():
    from nflbowl.devices import resolve_device, resolve_precision
    from nflbowl.training.trainer import resolve_placement

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")
    assert resolve_precision("auto", "cpu").name == "fp32"
    assert resolve_placement("auto", "cpu", 10, 0.5) == "cpu" and resolve_placement("gpu", "cpu", 10, 0.5) == "cpu"


def test_cpu_fallback_rules():
    pytest.importorskip("torch")
    import torch

    from nflbowl.devices import resolve_device, resolve_precision

    with pytest.raises(RuntimeError):
        resolve_precision("fp16", "cpu")
    if not torch.cuda.is_available():
        assert resolve_device("auto") == "cpu"
        with pytest.raises(RuntimeError):
            resolve_device("cuda")                  # explicit GPU request never silently falls back
    else:
        assert resolve_precision("auto", "cuda").name == "fp16"     # the 12F notebook numerics
