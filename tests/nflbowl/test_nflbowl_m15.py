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


# ====================================================================== T02 — signals
def _scaled(speed, a_tan, valid=None, dt=0.1):
    """(1, L, 6) scaled array + (1, L) mask from physical speed / a_tan."""
    L = len(speed)
    x = np.zeros((1, L, 6), np.float32)
    x[0, :, 0] = np.asarray(speed) / 10.0
    x[0, :, 1] = np.asarray(a_tan) / 5.0
    x[0, :, 4] = dt / 0.1
    x[0, :, 5] = 1.0
    m = np.ones((1, L), bool) if valid is None else np.asarray(valid, bool)[None]
    return x, m


def test_features_from_arrays_physics():
    from nflbowl.lab.m15 import signals as SG

    t = np.arange(30) * 0.1
    sp = 2.0 + 2.0 * t                                                        # constant +2 yd/s² acceleration
    x, m = _scaled(sp, np.full(30, 2.0))
    f = SG.features_from_arrays(x, m, 0.15)
    assert f["max_speed"][0] == pytest.approx(sp.max(), rel=1e-5)
    assert f["braking_frac"][0] == 0.0 and f["accel_frac"][0] == 1.0
    assert f["jerk_rms"][0] == pytest.approx(0.0, abs=1e-4)                  # constant a_tan
    assert f["deriv_rmse"][0] == pytest.approx(0.0, abs=1e-3)                # Δv/Δt agrees with a_tan
    assert f["entry_speed"][0] == pytest.approx(sp[:10].mean(), rel=1e-5)
    valid = np.ones(30, bool)
    valid[10:15] = False
    x2, m2 = _scaled(sp, np.full(30, -2.0), valid)
    f2 = SG.features_from_arrays(x2, m2, 0.15)
    assert f2["n_valid"][0] == 25 and f2["braking_frac"][0] == 1.0
    assert f2["max_speed"][0] == pytest.approx(sp[valid].max(), rel=1e-5)    # invalid frames never enter a statistic
    empty = SG.features_from_arrays(x2, np.zeros((1, 30), bool), 0.15)
    assert np.isnan(empty["max_speed"][0]) and empty["n_valid"][0] == 0


def test_signals_steps_on_synthetic_bundle(tmp_path, bundle_dir):
    from nflbowl.lab.m15 import signals as SG

    cfg = _cfg(tmp_path, bundle_dir, n_boot_screen=100)
    run = C.M15Run(cfg.run_dir, cfg)
    d = D.prepare_m15_data(cfg, run)
    ctx = {"outputs": []}
    feat = SG.run_a1(d, cfg, run, ctx)
    assert feat.height == d.index.height and {"max_speed", "braking_frac", "entry_speed"} <= set(feat.columns)
    crop, audit = SG.length_matched_combine(d, feat, cfg, 0)
    assert (audit["kept_valid"] <= audit["original_valid"]).all()
    steps, bal = SG.domain_contrast_steps(feat, crop, n_boot=100, seed=0)
    assert set(steps["step"].to_list()) == set(SG.STEPS)
    raw = steps.filter((pl.col("metric") == "max_speed") & (pl.col("step") == "raw"))
    assert raw["share_of_raw_remaining"][0] == pytest.approx(1.0)
    assert "smd_entry_speed_after" in bal
    ctx2 = {"outputs": []}
    SG.run_a1(d, cfg, run, ctx2)
    assert ctx2.get("cache") == "hit"                                          # cached on the second call


# ====================================================================== T03 — spectral
def test_welch_peak_and_features():
    from nflbowl.lab.m15 import spectral as SP

    fs, n = 10.0, 64
    t = np.arange(n) / fs
    seg = 3.0 + np.sin(2 * np.pi * 2.5 * t)                                   # 2.5 Hz sits on a bin for nperseg 32 (Δf 0.3125)
    freqs, psd, nw = SP.welch_psd(seg, fs, 32, "linear")
    assert nw == 3 and freqs[1] == pytest.approx(10 / 32)
    feats = SP.spectral_features(freqs, psd, ((0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 5.0)))
    assert feats["peak_freq"] == pytest.approx(2.5)
    assert feats["band_2"] > 0.9                                              # power in 1.5–3 Hz
    assert SP.welch_psd(seg[:20], fs, 32) is None                             # too short → no estimate, never padded
    rng = np.random.default_rng(0)
    white = SP.spectral_features(*SP.welch_psd(rng.normal(size=640), fs, 32, "linear")[:2], ((0, 5.0),))
    assert white["entropy"] > 0.9 and white["flatness"] > 0.5
    assert SP._auc(np.arange(50.0), np.arange(50.0) + 100) == pytest.approx(1.0)
    assert SP._auc(np.arange(50.0), np.arange(50.0)) == pytest.approx(0.5)


def test_ramp_surrogate_is_flat_after_linear_detrend():
    from nflbowl.lab.m15 import spectral as SP

    rng = np.random.default_rng(1)
    ramp = SP.ramp_surrogate(np.linspace(1, 8, 40) + rng.normal(0, 0.1, 40), rng)
    _, psd, _ = SP.welch_psd(ramp, 10.0, 16, "linear")
    assert psd.max() < 1e-20


def test_measurement_status_rule():
    from nflbowl.lab.m15 import spectral as SP

    keys = {"channel": ["speed"] * 4, "nperseg": [32] * 4, "detrend": ["linear"] * 4, "feature": ["a", "b", "c", "d"]}
    art = pl.DataFrame({**keys, "domain": ["combine"] * 4, "estimable": [True, True, False, True], "rho_len": [0.1] * 4, "pinned_share": [None] * 4},
                       schema_overrides={"pinned_share": pl.Float64})
    rel = pl.DataFrame({**keys, "domain": ["combine"] * 4, "scope": ["forty"] * 4, "icc1": [0.7, 0.5, 0.9, 0.3], "icc_lo": [0.5, 0.3, 0.8, 0.1],
                        "icc_hi": [0.8, 0.6, 0.95, 0.5], "n_athletes": [100] * 4})
    sens = pl.DataFrame({**keys, "perturbation": ["noise_0.05"] * 4, "spearman": [0.9, 0.9, 0.9, 0.9]})
    st = SP.measurement_status(art, rel, sens, None)
    got = dict(zip(st["feature"].to_list(), st["measurement_status"].to_list(), strict=True))
    assert got == {"a": "validated", "b": "exploratory", "c": "invalid", "d": "invalid"}


def test_spectral_steps_on_synthetic_bundle(tmp_path, bundle_dir):
    from nflbowl.lab.m15 import spectral as SP

    cfg = _cfg(tmp_path, bundle_dir, welch_nperseg=[8], stft_nperseg=8, stft_hop=2, n_boot_screen=100)
    run = C.M15Run(cfg.run_dir, cfg)
    d = D.prepare_m15_data(cfg, run)
    tab = SP.spectral_table(d.bundle, d.index["idx"].to_numpy(), cfg)
    assert set(tab["channel"].unique().to_list()) == {"speed", "a_tan"} and (tab["seg_len"] >= 1).all()
    dom = dict(zip(d.index["idx"].to_list(), d.index["domain"].to_list(), strict=True))
    art = SP.artifact_test(tab, None, None, dom, cfg)
    assert {"rho_len", "estimable"} <= set(art.columns)
    rel = SP.reliability(tab.join(d.index.select("idx", "domain", "kind", "nfl_id"), on="idx"), cfg, 100, 0, min_athletes=10)
    assert rel.height and "icc1" in rel.columns


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
