"""M15 task T01 — configuration, sequence-store adapter, legacy provenance audit, status runner.

Synthetic data only (``conftest.make_arrays``: 40 athletes, L = 16, 2 athletes on the fallback fold). AUTHORED, NOT
EXECUTED by the author: run with ``pytest tests/nflbowl/test_nflbowl_m15.py -q``.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from nflbowl.lab import status as S
from nflbowl.lab.m15 import config as C
from nflbowl.lab.m15 import data as D
from nflbowl.lab.m15 import legacy as LG


def _cfg(tmp_path, bundle_dir=None, **kw):
    base = {"drive_root": str(tmp_path / "drive"), "bundle_dir": str(bundle_dir or tmp_path / "missing_bundle"), "stage_dir": "",
            "expected_fingerprint": "", "max_frames": 16, "m12_run_dir": str(tmp_path / "ro" / "m12"), "m13_run_dir": str(tmp_path / "ro" / "m13"),
            "m14_run_dir": str(tmp_path / "ro" / "m14")}
    base.update(kw)
    return C.resolve(base)


# ====================================================================== config
def test_resolve_rejects_unknown_and_unsafe(tmp_path):
    with pytest.raises(KeyError):
        C.resolve({"not_a_key": 1})
    with pytest.raises(ValueError):
        _cfg(tmp_path, experiment_name="m14_reuse")
    with pytest.raises(ValueError):
        _cfg(tmp_path, run_training=True)                                  # second opt-in missing
    with pytest.raises(ValueError):
        _cfg(tmp_path, scale=[10.0, 5.0, 5.0, 3.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        _cfg(tmp_path, bands_hz=[[0.0, 6.0]])                              # above Nyquist (5 Hz)
    with pytest.raises(TypeError):
        _cfg(tmp_path, resume="yes")
    ro = tmp_path / "drive" / "experimental_lab"
    with pytest.raises(ValueError):                                        # run dir inside a read-only directory
        _cfg(tmp_path, m14_run_dir=str(ro))


def test_fingerprint_tracks_science_not_io(tmp_path):
    a = _cfg(tmp_path)
    assert a.fingerprint() == _cfg(tmp_path, notes="other", device="cpu", experiment_name="m15_other").fingerprint()
    assert a.fingerprint() != _cfg(tmp_path, latent_dims=[16]).fingerprint()
    assert a.training_enabled() is False
    assert _cfg(tmp_path, run_training=True, allow_new_training=True).training_enabled() is True
    assert a.selected_folds() == (0, 1, 2, 3, 4)


def test_run_guard_and_snapshot(tmp_path):
    cfg = _cfg(tmp_path)
    run = C.M15Run(cfg.run_dir, cfg)
    assert run.guard(run.root / "tables" / "x.parquet")
    with pytest.raises(PermissionError):
        run.guard(tmp_path / "elsewhere.txt")
    paths = run.snapshot_config(cfg, {"env": "test"})
    assert (run.root / "config" / f"config_{cfg.fingerprint()}.json").is_file()
    assert any(p.name == "run_config.json" for p in paths)
    with pytest.raises(ValueError):
        run.table(pl.DataFrame({"a": [1]}), "x", "not_a_family")


# ====================================================================== status runner
def test_run_step_statuses(tmp_path):
    cfg = _cfg(tmp_path)
    run = C.M15Run(cfg.run_dir, cfg)
    led = run.ledger

    def ok(ctx):
        ctx["outputs"].append("a.parquet")
        return 1

    def blocked(ctx):
        raise S.ResearchBlocked(S.BLOCKED_MISSING_INPUT, "no file", missing=["x.parquet"])

    def invalid(ctx):
        raise S.ValidationFailed("contract broken", {"n": 3})

    def cached(ctx):
        ctx["cache"] = "hit"
        return 2

    assert S.run_step(run, led, "A", ok) == 1 and led.get("A") == S.COMPLETED
    assert S.run_step(run, led, "B", blocked) is None and led.get("B") == S.BLOCKED_MISSING_INPUT
    assert led.records["B"]["detail"]["missing"] == ["x.parquet"]
    assert S.run_step(run, led, "C", invalid) is None and led.get("C") == S.FAILED_VALIDATION
    assert S.run_step(run, led, "D", cached) == 2 and led.get("D") == S.CACHED
    assert S.run_step(run, led, "E", ok, requires=["B"]) is None and led.get("E") == S.NOT_RUN
    assert S.run_step(run, led, "F", ok, enabled=False, reason_disabled="off") is None and led.records["F"]["reason"] == "off"
    reopened = S.StatusLedger.open(run.root, "m15")
    assert reopened.get("D") == S.CACHED                                   # persisted for the next session
    assert S.gate_row("g", None, 0.6, None)["status"] == S.NOT_EVALUATED    # no evidence → never PASS


# ====================================================================== adapter
def test_segment_stats_rule():
    M = np.array([[1, 1, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1]], bool)
    dt = np.full(M.shape, 0.1)
    dt[1, 3] = 0.3                                                          # frame 3 too long → a new segment starts at frame 4
    nv, ns, lead, trail = D._segment_stats(M, dt, 0.15)
    assert nv.tolist() == [5, 7] and ns.tolist() == [2, 2] and lead.tolist() == [0, 1] and trail.tolist() == [2, 0]


def test_prepare_m15_data_on_synthetic_bundle(tmp_path, bundle_dir, arrays):
    X, M, meta, folds, athletes = arrays
    cfg = _cfg(tmp_path, bundle_dir)
    run = C.M15Run(cfg.run_dir, cfg)
    d = D.prepare_m15_data(cfg, run)
    assert d.audit.filter(pl.col("status") == "fail").height == 0, d.audit.filter(pl.col("status") == "fail").to_dicts()
    assert d.index.height == X.shape[0]
    assert d.index["n_valid"].to_list() == M.sum(1).tolist()
    fb = d.folds.filter(pl.col("fold_source") == "fallback_mod")
    assert fb.height == 2                                                   # conftest leaves 2 athletes out of the manifest
    assert set(d.folds["fold"].to_list()) == set(range(5))
    tr = d.rows(role="train", held_out=0)
    ho = d.rows(role="held_out", held_out=0)
    assert not set(d.index[tr]["nfl_id"].to_list()) & set(d.index[ho]["nfl_id"].to_list())
    assert (run.root / "manifests" / "dataset_fingerprint.json").is_file()
    assert (run.root / "tables" / "dataset" / "data_contract_audit.parquet").is_file()
    d2 = D.prepare_m15_data(cfg, run)                                      # second call reuses the cached index
    assert d2.index.equals(d.index)


def test_audit_flags_wrong_fingerprint_and_missing_bundle(tmp_path, bundle_dir):
    cfg = _cfg(tmp_path, bundle_dir, expected_fingerprint="0000000000000000")
    with pytest.raises(S.ValidationFailed):
        D.prepare_m15_data(cfg, None)
    with pytest.raises(S.ResearchBlocked) as e:
        D.open_m15_bundle(_cfg(tmp_path))
    assert e.value.status == S.BLOCKED_MISSING_INPUT


def test_game_key_heuristic():
    m = pl.DataFrame({"domain": ["nfl", "nfl", "combine"], "seq_key": ["2023091000_123_45", "w_2024120801_9", "c1"]})
    s, st = D.game_key(m)
    assert st["status"] == "parsed" and s.to_list() == ["2023091000", "2024120801"]
    s2, st2 = D.game_key(pl.DataFrame({"domain": ["nfl"], "seq_key": ["n1000_0"]}))
    assert s2 is None and st2["status"] == "not established"


# ====================================================================== legacy provenance
def _official(bundle_dir):
    from nflbowl.data.loading import open_bundle
    from nflbowl.data.splits import fold_map

    b = open_bundle(bundle_dir)
    return b, fold_map(b.meta, b.folds, 5)


def _emb(fold_of, leak=False):
    rows = []
    for i, f in fold_of.items():
        for k in range(5):
            role = "held_out" if f == k else "train"
            rows.append({"nfl_id": i, "fold": k, "role": role, "z0": 0.1 * k, "z1": float(i % 7)})
    if leak:
        i0 = next(iter(fold_of))
        rows.append({"nfl_id": i0, "fold": fold_of[i0], "role": "train", "z0": 0.0, "z1": 0.0})
    return pl.DataFrame(rows)


def test_legacy_fold_safe_exploratory_unknown(tmp_path, bundle_dir):
    b, fold_of = _official(bundle_dir)
    safe = tmp_path / "legacy" / "safe"
    safe.mkdir(parents=True)
    _emb(fold_of).write_parquet(safe / "stft_embeddings.parquet")
    (safe / "config.json").write_text(json.dumps({"bundle_fingerprint": b.fingerprint, "seed": 1}))
    leak = tmp_path / "legacy" / "leak"
    leak.mkdir()
    _emb(fold_of, leak=True).write_parquet(leak / "lstm_embeddings.parquet")
    (leak / "config.json").write_text(json.dumps({"bundle_fingerprint": b.fingerprint}))
    single = tmp_path / "legacy" / "single"
    single.mkdir()
    ids = sorted(fold_of)
    pl.DataFrame({"nfl_id": ids, "z0": np.arange(len(ids), dtype=float)}).write_parquet(single / "athlete_emb.parquet")
    (single / "split.json").write_text(json.dumps({"train_ids": ids[:30], "test_ids": ids[30:]}))
    np.save(tmp_path / "legacy" / "bare.npy", np.zeros((4, 3)))
    before = {p: p.read_bytes() for p in (tmp_path / "legacy").rglob("*") if p.is_file()}

    tab, recs = LG.audit([tmp_path / "legacy"], b, 5)
    prov = {r.path.split("legacy")[-1]: r.provenance for r in recs}
    assert prov[next(k for k in prov if k.endswith("stft_embeddings.parquet"))] == LG.VERIFIED_FOLD_SAFE
    assert prov[next(k for k in prov if k.endswith("lstm_embeddings.parquet"))] == LG.UNKNOWN_PROVENANCE
    assert prov[next(k for k in prov if k.endswith("athlete_emb.parquet"))] == LG.VERIFIED_EXPLORATORY_ONLY
    assert prov[next(k for k in prov if k.endswith("bare.npy"))] == LG.UNKNOWN_PROVENANCE
    assert {p: p.read_bytes() for p in before} == before                     # sources never modified
    bad = next(r for r in recs if r.path.endswith("athlete_emb.parquet"))
    with pytest.raises(S.ResearchBlocked) as e:
        LG.assert_fold_safe(bad)
    assert e.value.status == S.BLOCKED_PROVENANCE
    LG.assert_fold_safe(next(r for r in recs if r.path.endswith("stft_embeddings.parquet")))
    assert tab.height == len(recs) and "provenance" in tab.columns


def test_legacy_fingerprint_mismatch_is_unknown(tmp_path, bundle_dir):
    b, fold_of = _official(bundle_dir)
    d = tmp_path / "legacy_old"
    d.mkdir()
    _emb(fold_of).write_parquet(d / "emb.parquet")
    (d / "config.json").write_text(json.dumps({"bundle_fingerprint": "ffffffffffffffff"}))
    _tab, recs = LG.audit([d], b, 5)
    r = next(r for r in recs if r.path.endswith("emb.parquet"))
    assert r.provenance == LG.UNKNOWN_PROVENANCE and any("recorded bundle" in x for x in r.reasons)


def test_legacy_adopt_copies(tmp_path, bundle_dir):
    b, fold_of = _official(bundle_dir)
    d = tmp_path / "legacy_adopt"
    d.mkdir()
    _emb(fold_of).write_parquet(d / "emb.parquet")
    (d / "config.json").write_text(json.dumps({"bundle_fingerprint": b.fingerprint}))
    _tab, recs = LG.audit([d], b, 5)
    cfg = _cfg(tmp_path, bundle_dir)
    run = C.M15Run(cfg.run_dir, cfg)
    dst = LG.adopt(next(r for r in recs if r.path.endswith("emb.parquet")), run)
    assert dst.is_file() and dst.read_bytes() == (d / "emb.parquet").read_bytes()
    assert (dst.parent / (dst.name + ".provenance.json")).is_file()
    assert (d / "emb.parquet").is_file()
