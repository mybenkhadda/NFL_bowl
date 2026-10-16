"""Synthetic validation of Milestone 3 (self-supervised encoder) — callable from a notebook cell, no subprocess,
no warehouse. Needs PyTorch.

    from nflbdb.research.advanced_movement import tcn_validation
    report = tcn_validation.run_all()          # table: check, status, detail
    tcn_validation.assert_all_passed(report)

Data: the analytic sprint fixture of ``validation`` / ``fpca_validation`` (40 athletes, two attempts, recorded
from a late start) and the two-mode curves of ``fpca_validation``. Encoders are tiny and trained for a few epochs:
the checks test the WIRING (who is seen by what, what a loss counts, what a metric returns on a known answer),
never the quality of a trained model. ``tests/test_advanced_movement_tcn.py`` calls the same functions.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch

from ..contracts import conform
from . import data as dt
from . import fpca as fp
from . import fpca_eval as fe
from . import fpca_validation as fv
from . import repr_eval as rv
from . import tcn
from . import validation as base

ACFG = base.ACFG
NAME = "A_speed_by_distance"
TINY = tcn.TcnConfig(embedding_dim=4, hidden=8, kernel_size=3, dilations=(1, 2), decoder_hidden=16, batch_athletes=8,
                     max_epochs=6, min_epochs=2, patience=3)
TINY_B = replace(TINY, objective=tcn.CONSISTENCY)
ECFG = rv.EvalConfig(n_boot=40, n_boot_probe=20, min_athletes=5, probe_folds=3)
_raises = base._raises


# ============================================================================== fixtures
def sprint_fixture(n_athletes: int = 40, seed: int = 0) -> dict:
    """Sprint bundle + the curves of representation A + a ``Meta`` of its windows and athletes."""
    b = fv.synthetic_fpca_bundle(n_athletes, seed)
    return {**b, "curves": fp.build_curves(b["ss"], ACFG, fp.SPEC[NAME]), "meta": fe.Meta(b["windows"], b["athletes"])}


def _state_equal(a: tcn.Trained, b: tcn.Trained) -> bool:
    sa, sb = a.model.state_dict(), b.model.state_dict()
    return all(torch.allclose(sa[k], sb[k], atol=1e-6) for k in sa)


def _synthetic_embeddings(b: dict, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    """One fold model's worth of embeddings with a known structure: dimension 1 = official forty, dimension 2 = an
    athlete trait unrelated to it, dimension 3 = attempt noise. Half of the athletes are 'heldout', half 'fit'."""
    meta, w = b["meta"], b["windows"]
    seq, nfl = w["seq_id"].to_numpy().astype(str), w["nfl_id"].to_numpy()
    rng = np.random.default_rng(seed)
    ids = np.unique(nfl)
    trait = dict(zip(ids.tolist(), rng.normal(size=ids.size)))
    forty = meta.official[fe.FORTY][meta.athlete_rows(nfl)]
    z = (forty - np.nanmean(forty)) / np.nanstd(forty)
    Z = np.column_stack([z + 0.05 * rng.normal(size=nfl.size), np.array([trait[int(i)] for i in nfl]) + 0.05 * rng.normal(size=nfl.size),
                         0.3 * rng.normal(size=nfl.size)])
    role = np.where(np.isin(nfl, ids[::2]), "heldout", "fit")
    return rv.embedding_frame("SYN", 0, 0, seq, nfl, role, "full", Z, Z), Z


# ============================================================================== checks
def check_athlete_group_split_separation() -> str:
    b = sprint_fixture()
    cs, splits = b["curves"], b["splits"]
    spec = tcn.tcn_plan(TINY, None, [NAME], {NAME: 2})[0][0]
    for f in range(ACFG.outer_folds):
        train, test = dt.fold_athletes(splits, 0, f)
        fit, dev = dt.inner_athletes(splits, 0, f, TINY.inner_dev_fold)
        assert not set(fit) & set(dev) and set(fit) | set(dev) == set(train) and not set(train) & set(test)
    t = tcn.train_fold(cs, splits, 0, 0, spec, TINY, seed=1)
    train, test = dt.fold_athletes(splits, 0, 0)
    assert set(t.normalizer.fit_athletes) == set(train) and not set(t.fit_athletes) & set(test) and not set(t.dev_athletes) & set(test)
    assert _raises(dt.LeakageError, lambda: t.normalizer.assert_unseen(train[:1]))
    emb, recon, counts = tcn.tcn_fold_outputs(t, cs, splits, None, ECFG, seed=1)
    held = emb.filter(pl.col("role") == "heldout")
    assert set(held["nfl_id"].unique().to_list()) <= set(test) and held.height > 0
    assert set(emb.filter(pl.col("role") != "heldout")["nfl_id"].unique().to_list()) <= set(train)
    assert set(recon["nfl_id"].unique().to_list()) <= set(test) and counts["n_heldout_athletes"] == len(test)
    by_seq = emb.filter(pl.col("view") == "full").join(b["windows"].select("seq_id", pl.col("nfl_id").alias("_a")), on="seq_id")
    assert (by_seq["nfl_id"] == by_seq["_a"]).all() and by_seq.group_by("nfl_id").agg(pl.col("role").n_unique())["role"].max() == 1
    return f"every attempt of an athlete has one role; the {len(test)} held-out athletes are in no normaliser, fit or early-stopping set"


def check_masked_reconstruction_loss() -> str:
    pred = torch.zeros(2, 6, requires_grad=True)
    target = torch.arange(12, dtype=torch.float32).view(2, 6)
    hidden = torch.tensor([[False, True, True, False, False, False], [False, False, False, False, True, False]])
    loss = tcn.masked_mse(pred, target, hidden)
    assert abs(float(loss) - (1 + 4 + 100) / 3) < 1e-5                       # mean over the three hidden points only
    loss.backward()
    assert float(pred.grad[~hidden].abs().max()) == 0.0 and (pred.grad[hidden] != 0).all()
    moved = pred.detach().clone()
    moved[~hidden] = 50.0                                                   # predictions on visible points do not count
    assert abs(float(tcn.masked_mse(moved, target, hidden)) - float(loss)) < 1e-5
    none = tcn.masked_mse(pred, target, torch.zeros_like(hidden))
    assert float(none) == 0.0 and none.requires_grad
    obs = np.zeros((3, 20), dtype=bool)
    obs[0, 2:18], obs[1, :9], obs[2, 5:10] = True, True, True
    h = rv.span_mask(obs, np.random.default_rng(0))
    assert not (h & ~obs).any() and ((obs & ~h).sum(1) >= 4).all() and h[0].sum() <= round(0.3 * 16)
    big, ids = np.ones((20, 40), dtype=bool), np.array([f"s{i}" for i in range(20)])
    m1, m2 = rv.evaluation_mask(ids, big, 7), rv.evaluation_mask(ids[::-1], big[::-1], 7)
    assert np.array_equal(m1, m2[::-1]) and not np.array_equal(m1, rv.evaluation_mask(ids, big, 8))   # per sequence, seeded
    return "the loss counts recorded points hidden on purpose and nothing else; masks never touch an unrecorded point"


def check_augmentation_physical_validity() -> str:
    cs, _ = fv.synthetic_curves(noise=0.02)
    norm = tcn.CurveNormalizer.fit(cs, np.unique(cs.nfl_ids), TINY.min_coverage, TINY.min_observed_fraction)
    assert 0.5 * 0.02 < norm.noise_sd < 2.0 * 0.02, norm.noise_sd            # the measured noise level, not a guess
    assert norm.noise_sd / norm.scale < 0.1                                  # far below the spread between athletes
    X, obs = norm.transform(cs.Y)
    assert np.allclose(norm.inverse(X.astype(float))[obs], cs.Y[obs], atol=1e-4)      # a reversible change of units
    rng = np.random.default_rng(1)
    hidden = rv.span_mask(obs, rng, TINY.mask_ratio, TINY.mask_spans, TINY.p_edge, TINY.min_visible)
    assert not (hidden & ~obs).any() and 0.1 < hidden.sum() / obs.sum() <= TINY.mask_ratio + 1e-9
    edge = (hidden[:, 0] | hidden[:, -1]).mean()
    assert 0.1 < edge < 0.85                                                 # some masks shorten the recording, not all
    head, tail = rv.truncation_mask(obs, "head", 0.3), rv.truncation_mask(obs, "tail", 0.3)
    n = obs.sum(1)
    assert (head.sum(1) == n - np.ceil(0.3 * n)).all() and not head[:, 0].any() and head[:, -1].all() and not tail[:, -1].any()
    assert _raises(ValueError, lambda: rv.truncation_mask(obs, "middle", 0.3))
    used = [a for a in tcn.AUGMENTATIONS if a[1].startswith("used")]
    excluded = " ".join(a[0] for a in tcn.AUGMENTATIONS if a[1] == "excluded")
    assert len(used) == 4 and all(w in excluded for w in ("warping", "scaling", "acceleration", "mixing", "simulated"))
    return f"noise SD {norm.noise_sd:.3f} recovered from the curves (true 0.02); masks only hide, values are never altered"


def check_variable_length_padding() -> str:
    cs, _ = fv.synthetic_curves()
    Y = cs.Y.copy()
    Y[::3, 36:] = np.nan                                                    # a third of the recordings stop early
    Y[5, 12:] = np.nan                                                      # one keeps a quarter of the axis
    part = fp.CurveSet(cs.spec, cs.grid, Y, cs.t_orig, cs.seq_ids, cs.nfl_ids)
    norm = tcn.CurveNormalizer.fit(part, np.unique(cs.nfl_ids), 0.60, 0.50)
    X, obs = norm.transform(Y)
    assert (X[~obs] == 0).all() and not obs[::3, 36:].any() and norm.support.all()
    model = tcn.CurveTCN(TINY, cs.grid.size).eval()
    x, v, dom = torch.from_numpy(X[:6]), torch.from_numpy(obs[:6]), torch.from_numpy(norm.support)
    z = model.encode(x, v, dom)
    junk = x.clone()
    junk[~v] = 99.0                                                         # whatever sits in an unrecorded cell
    assert torch.allclose(z, model.encode(junk, v, dom), atol=1e-6)
    short = norm.support.copy()
    short[40:] = False
    outside = x.clone()
    outside[:, 40:] = -7.0                                                  # values outside the model's support
    d = torch.from_numpy(short)
    assert torch.allclose(model.encode(x, v, d), model.encode(outside, v, d), atol=1e-6)
    Z, R, frac = tcn.embed(model, norm, Y)
    assert np.isnan(Z[5]).all() and frac[5] < 0.5 and np.isfinite(Z[0]).all() and np.isnan(R[5]).all()
    assert _raises(tcn.TcnError, lambda: norm.transform(Y[:, :10]))
    return "an unrecorded or out-of-support cell cannot change an embedding; a recording that is too short is not embedded"


def check_channel_availability() -> str:
    b = fv.synthetic_fpca_bundle()
    cs = fp.build_curves(b["ss"], ACFG, fp.SPEC["B_accel_by_speed"])        # boundary frames carry no acceleration
    ids = np.unique(cs.nfl_ids)
    norm = tcn.CurveNormalizer.fit(cs, ids, TINY.min_coverage, TINY.min_observed_fraction)
    X, obs = norm.transform(cs.Y)
    assert not obs[~np.isfinite(cs.Y)].any() and np.isfinite(X).all() and not obs[:, ~norm.support].any()
    m = fp.fit_fpca(cs, ids, fv.FCFG)
    assert np.array_equal(norm.support, m.support)                           # the support rule of notebook 02
    assert np.allclose(norm.mean[norm.support], m.mean)                      # and its mean function
    Y = cs.Y.copy()
    Y[0] = np.nan                                                           # a sequence whose signal is missing
    model = tcn.CurveTCN(TINY, cs.grid.size)
    Z, _, _ = tcn.embed(model, norm, Y)
    assert np.isnan(Z[0]).all() and tcn.N_INPUT == 3
    empty = fp.CurveSet(cs.spec, cs.grid, np.full_like(cs.Y, np.nan), cs.t_orig, cs.seq_ids, cs.nfl_ids)
    assert _raises(tcn.TcnError, lambda: tcn.CurveNormalizer.fit(empty, ids, 0.8, 0.5))
    assert _raises(tcn.TcnError, lambda: tcn.TcnConfig(kernel_size=4)) and _raises(tcn.TcnError, lambda: tcn.TcnConfig(objective="x"))
    return "a missing signal stays missing; the normaliser reproduces the FPCA support and mean of the same athletes"


def check_attempt_pairing() -> str:
    nfl = np.array([1, 1, 2, 3, 3, 3, 4, 4, 9, 9])
    rows = np.arange(8)                                                      # athlete 9 is not among the rows
    s = tcn.AttemptPairSampler(nfl, rows, batch_athletes=2)
    rng = np.random.default_rng(0)
    seen = []
    for batch, pairs in s.epoch(rng):
        tcn.assert_pairs_same_athlete(nfl, batch, pairs)
        assert set(batch.tolist()) <= set(rows.tolist()) and len(set(batch.tolist())) == batch.size
        per = np.unique(nfl[batch], return_counts=True)[1]
        assert per.max() <= 2 and len(pairs) == int((per == 2).sum())        # at most two attempts; a single attempt has no pair
        seen += np.unique(nfl[batch]).tolist()
    assert sorted(seen) == [1, 2, 3, 4]
    batch, pairs = s.all_pairs()
    assert batch.tolist() == [0, 1, 2, 3, 4, 6, 7] and pairs.tolist() == [[0, 1], [3, 4], [5, 6]]
    assert _raises(tcn.TcnError, lambda: tcn.assert_pairs_same_athlete(nfl, batch, np.array([[1, 2]])))
    return "positive pairs are two attempts of one athlete; a third attempt is dropped and a lone attempt forms no pair"


def check_no_cross_fold_pair_leakage() -> str:
    b = sprint_fixture()
    cs, splits = b["curves"], b["splits"]
    train, test = dt.fold_athletes(splits, 0, 1)
    fit, dev = dt.inner_athletes(splits, 0, 1, 0)
    norm = tcn.CurveNormalizer.fit(cs, train, TINY.min_coverage, TINY.min_observed_fraction)
    rows = np.flatnonzero(np.isin(cs.nfl_ids, fit))
    for batch, pairs in tcn.AttemptPairSampler(cs.nfl_ids, rows, 8).epoch(np.random.default_rng(0)):
        assert set(cs.nfl_ids[batch].tolist()) <= set(fit)                   # no development or held-out athlete in a batch
        tcn.assert_pairs_same_athlete(cs.nfl_ids, batch, pairs)
    assert _raises(dt.LeakageError, lambda: tcn.train_encoder(cs, train, fit, [*dev, fit[0]], TINY_B, seed=3))
    a = tcn.train_encoder(cs, train, fit, dev, TINY_B, seed=3, tag="leak")
    moved = cs.Y.copy()
    out = np.isin(cs.nfl_ids, test)
    moved[out] = moved[out] * 3.0 + 1.0                                      # change the held-out athletes only
    c = tcn.train_encoder(fp.CurveSet(cs.spec, cs.grid, moved, cs.t_orig, cs.seq_ids, cs.nfl_ids), train, fit, dev, TINY_B, seed=3, tag="leak")
    assert _state_equal(a, c) and np.allclose(a.normalizer.mean, c.normalizer.mean) and a.normalizer.scale == c.normalizer.scale
    assert np.array_equal(norm.support, a.normalizer.support) and a.info["n_fit_athletes"] <= len(fit)
    return "held-out athletes cannot move a weight, a normaliser or a stopping epoch; fit and early-stopping athletes cannot overlap"


def check_embedding_extraction() -> str:
    b = sprint_fixture()
    cs = b["curves"]
    ids = np.unique(cs.nfl_ids)
    t = tcn.train_encoder(cs, ids, ids[:28], ids[28:], TINY, seed=2, tag="extract")
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    Z1, R1, frac = tcn.embed(t.model, t.normalizer, cs.Y)
    Z2, _, _ = tcn.embed(t.model, t.normalizer, cs.Y)
    assert Z1.shape == (cs.n, TINY.embedding_dim) and np.array_equal(Z1, Z2, equal_nan=True)     # dropout is off, nothing random
    assert all(torch.equal(before[k], v) for k, v in t.model.state_dict().items()) and not t.model.training
    ok = np.isfinite(Z1).all(1)
    assert ok.any() and (frac[ok] >= t.normalizer.min_observed_fraction).all() and np.isnan(R1[:, ~t.normalizer.support]).all()
    hidden = rv.evaluation_mask(cs.seq_ids, t.normalizer.observed(cs.Y), 5)
    Zm, _, fm = tcn.embed(t.model, t.normalizer, cs.Y, visible=~hidden)
    assert (fm <= frac + 1e-12).all() and not np.allclose(Zm[ok & np.isfinite(Zm).all(1)], Z1[ok & np.isfinite(Zm).all(1)])
    Zs = Z1.copy()
    Zs[0] = np.nan
    frame = conform(rv.embedding_frame("M", 0, 0, cs.seq_ids, cs.nfl_ids, "heldout", "full", Zs, Zs), rv.EMBEDDINGS)
    blocks = list(rv.emb_blocks(frame))
    keep = np.isfinite(Zs).all(1)
    order = np.argsort(cs.seq_ids[keep].astype(str))
    assert len(blocks) == 1 and blocks[0].Z.shape == (int(keep.sum()), TINY.embedding_dim) and np.allclose(blocks[0].Z, Zs[keep][order])
    assert _raises(ValueError, lambda: blocks[0].space("whitened"))
    return f"extraction is deterministic and leaves the weights untouched; {int(ok.sum())} of {cs.n} attempts embedded, the rest left out"


def check_representation_collapse_detection() -> str:
    rng = np.random.default_rng(0)
    healthy = rv.collapse_metrics(rng.normal(size=(300, 8)))
    assert healthy["rank_share"] > 0.9 and not rv.is_collapsed(healthy, ECFG) and healthy["near_identical_share"] == 0.0
    line = rv.collapse_metrics(np.outer(rng.normal(size=300), np.ones(8)) + 1e-9 * rng.normal(size=(300, 8)))
    assert line["effective_rank"] < 1.1 and rv.is_collapsed(line, ECFG)         # every athlete on one direction
    point = rv.collapse_metrics(np.ones((50, 8)))
    assert point["effective_rank"] == 0.0 and rv.is_collapsed(point, ECFG)      # every athlete the same vector
    clumps = np.repeat(rng.normal(size=(3, 8)), 40, axis=0) + 1e-6 * rng.normal(size=(120, 8))
    assert rv.is_collapsed(rv.collapse_metrics(clumps), ECFG)                   # a few distinct vectors only
    assert not rv.is_collapsed(rv.collapse_metrics(np.ones((2, 8))), ECFG)      # too few rows to judge
    torch.manual_seed(0)
    spread = torch.randn(64, 4)
    dead = 1e-3 * torch.randn(64, 4)
    vic = replace(TINY_B, consistency="vicreg")
    assert float(tcn.consistency_loss(dead, dead, vic)) > float(tcn.consistency_loss(spread, spread, vic)) + 0.3
    mse = replace(TINY_B, consistency="mse")
    za, zb = torch.randn(32, 4), torch.randn(32, 4)
    assert abs(float(tcn.consistency_loss(za, zb, mse)) - float(tcn.consistency_loss(5 * za, 5 * zb, mse))) < 1e-4
    assert float(tcn.consistency_loss(za, za, mse)) == 0.0
    return "one direction, one point and a few clumps are flagged; the variance-preserving loss penalises a shrunken embedding"


def check_checkpoint_reload_compatibility() -> str:
    b = sprint_fixture()
    cs, splits = b["curves"], b["splits"]
    plan = tcn.tcn_plan(TINY, None, [NAME], {NAME: 2})
    with tempfile.TemporaryDirectory() as tmp:
        tables, models = tcn.fit_methods({NAME: cs}, splits, plan, [0], ECFG, seed=4, models_dir=Path(tmp))
        tcn.record_models(Path(tmp), "TCN_A", "key-1", [s.method for s, _ in plan])
        t = models[("TCNA_A", 0, 0)]
        back = tcn.load_checkpoint(tcn.checkpoint_path(Path(tmp), "TCNA_A", 0, 0))
        assert _state_equal(t, back) and back.tcfg == t.tcfg and back.fit_athletes == t.fit_athletes
        assert (back.method, back.repeat, back.outer_fold, back.representation) == ("TCNA_A", 0, 0, NAME)
        assert np.array_equal(back.normalizer.support, t.normalizer.support) and back.normalizer.fit_athletes == t.normalizer.fit_athletes
        Z, _, _ = tcn.embed(t.model, t.normalizer, cs.Y)
        Zb, _, _ = tcn.embed(back.model, back.normalizer, cs.Y)
        assert np.allclose(Z, Zb, atol=1e-6, equal_nan=True)
        loaded = tcn.load_models(Path(tmp), "TCN_A", "key-1", plan, splits, [0])
        assert set(loaded) == set(models) and len(loaded) == ACFG.outer_folds
        assert tcn.load_models(Path(tmp), "TCN_A", "another-key", plan, splits, [0]) == {}      # a stale checkpoint is never used
        assert tcn.load_models(Path(tmp), "TCN_B", "key-1", plan, splits, [0]) == {}
    for c in (rv.EMBEDDINGS, rv.RECON_SEQ, tcn.HISTORY, tcn.FOLDS):
        conform(tables[c.name], c)
    renamed = tcn.stage_tables("tcn_a", tables)
    assert sorted(renamed) == sorted(c.name for c in tcn.stage_contracts("tcn_a")) and "tcn_a_attempt_embeddings" in renamed
    return "a reloaded fold model gives the same embeddings, normalisation and provenance; a checkpoint of another configuration is refused"


def check_retrieval_calculations() -> str:
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    Z1, Z2 = centres + 0.1, centres - 0.1
    same = np.array(["g"] * 4)
    st = rv.pair_stats(Z1, Z2, same, top_k=2)
    assert np.allclose(st["auc"], 1) and np.allclose(st["top1"], 1) and np.allclose(st["rr"], 1) and st["n_candidates"].tolist() == [3] * 4
    assert np.allclose(st["d_same"], np.sqrt(0.08)) and (st["d_diff_median"] > 9).all() and (st["cos_same"] > st["cos_diff_mean"])[1:].all()
    swapped = rv.pair_stats(Z1, Z2[[1, 0, 3, 2]], same, top_k=2)             # the own attempt is never the nearest
    assert np.allclose(swapped["top1"], 0) and (swapped["auc"] < 1).all() and (swapped["rr"] <= 0.5).all()
    split = rv.pair_stats(Z1, Z2, np.array(["a", "a", "b", "c"]))
    assert split["n_candidates"].tolist() == [1, 1, 0, 0] and np.isnan(split["auc"][2:]).all() and np.allclose(split["auc"][:2], 1)
    ties = rv.pair_stats(np.zeros((3, 2)), np.zeros((3, 2)), same[:3])
    assert np.allclose(ties["auc"], 0.5) and np.allclose(ties["top1"], 0)     # a tie is half a win, never a hit
    rng = np.random.default_rng(0)
    noise = rv.pair_stats(rng.normal(size=(300, 4)), rng.normal(size=(300, 4)), np.array(["g"] * 300))
    assert abs(noise["auc"].mean() - 0.5) < 0.05 and noise["top1"].mean() < 0.05
    one = rv.pair_stats(Z1, Z2, same, symmetric=False)
    assert np.allclose(one["auc"], 1) and abs(rv.auc_score(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) - 0.75) < 1e-12
    return "discrimination, rank and gallery follow hand-computed answers; unrelated attempts score at chance"


def check_bootstrap_sampling() -> str:
    x = np.random.default_rng(0).normal(1.0, 1.0, size=(200, 2))
    est, lo, hi = rv.boot_mean(x, 400, 11)
    again = rv.boot_mean(x, 400, 11)
    assert np.allclose(est, x.mean(0)) and (lo < est).all() and (est < hi).all() and np.allclose(lo, again[1])
    assert 0.15 < (hi - lo)[0] < 0.45                                        # about 4 standard errors of a mean of 200
    assert np.isnan(rv.boot_mean(x[:4], 400, 11)[1]).all() and np.isnan(rv.boot_mean(x, 0, 11)[1]).all()
    f = rv.athlete_folds(np.array([5, 5, 6, 7, 7, 8, 9, 9]), 3, 1)
    assert f[0] == f[1] and f[3] == f[4] and f[6] == f[7] and set(f.tolist()) <= {0, 1, 2}
    rng = np.random.default_rng(1)
    ath = np.repeat(np.arange(60), 2)
    strata = np.repeat(np.array(["p", "q"]), 60)
    trait = np.repeat(rng.normal(size=(60, 3)), 2, axis=0)
    Z = trait + 0.2 * rng.normal(size=(120, 3))
    r = rv.trace_icc(Z, ath, strata, 200, 3)
    assert r["icc_trace"] > 0.9 and r["icc_lower"] < r["icc_upper"] and r["n_athletes_repeat"] == 60
    assert r["icc_lower"] - 0.03 <= r["icc_trace"] <= r["icc_upper"] + 0.03
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert abs(rv.trace_icc(Z @ Q, ath, strata)["icc_trace"] - r["icc_trace"]) < 1e-9          # invariant to a rotation
    assert rv.trace_icc(Z, rng.permutation(ath), strata)["icc_trace"] < 0.3                   # labels shuffled: no athlete effect
    assert rv.trace_icc(Z[::2], ath[::2], strata[::2])["icc_trace"] is None                    # one attempt each: undefined
    return "intervals resample athletes, reproducibly; the trace reliability is rotation-invariant and vanishes when labels are shuffled"


def check_nuisance_probe_training_separation() -> str:
    b = sprint_fixture()
    meta = b["meta"]
    emb, _ = _synthetic_embeddings(b)
    block = next(rv.emb_blocks(emb))
    for level in ("athlete", "attempt"):
        u = rv._units(block, meta, level)
        for protocol in ("heldout_crossfit", "train_to_heldout"):
            for fit, ev in rv.probe_splits(u["athlete"], u["held"], protocol, ECFG.probe_folds, 1):
                assert not set(u["athlete"][fit].tolist()) & set(u["athlete"][ev].tolist()) and u["held"][ev].all()
                assert (u["held"][fit]).all() == (protocol == "heldout_crossfit")
    assert _raises(ValueError, lambda: rv.probe_splits(block.nfl_ids, block.held, "all", 3, 1))
    args = ([block], meta, "athlete", fe.FORTY, "continuous")
    for protocol in ("heldout_crossfit", "train_to_heldout"):
        r = rv.probe(*args, protocol, ECFG, 2)
        assert r["value"] > 0.8 and r["gain"] > 0.5 and r["metric"] == "r2", r      # a planted official time is found
    w = rv.probe([block], meta, "athlete", fe.WEIGHT, "continuous", "heldout_crossfit", ECFG, 2)
    assert w["gain"] < 0.4, w                                                     # an unrelated measurement is not
    pos = rv.probe([block], meta, "athlete", "position_group", "categorical", "heldout_crossfit", ECFG, 2)
    assert pos["metric"] == "accuracy" and 0 <= pos["value"] <= 1 and 0 <= pos["baseline"] <= 1
    flag = rv.probe([block], meta, "attempt", "first_frame_censored", "binary", "heldout_crossfit", ECFG, 2)
    assert flag is None or (flag["metric"] == "auc" and 0 <= flag["value"] <= 1)
    m = rv.ridge_fit(np.arange(10.0)[:, None], 2 * np.arange(10.0) + 1, 1e-9)
    assert np.allclose(rv.ridge_predict(m, np.array([[20.0]])), 41.0, atol=1e-5)
    assert np.allclose(rv.ridge_predict(rv.ridge_fit(np.zeros((5, 0)), np.arange(5.0), 1.0), np.zeros((2, 0))), 2.0)
    resid = rv.residualise_embeddings(emb, meta, "official", ECFG, 2)
    assert resid["method"].unique().to_list() == ["SYN|resid_official"] and (resid["role"] == "heldout").all()
    rb = next(rv.emb_blocks(resid))
    left = rv.probe([rb], meta, "athlete", fe.FORTY, "continuous", "heldout_crossfit", ECFG, 2)
    assert left["gain"] < 0.4, left                                               # the official time is gone from the residual
    pairs = rv.pair_statistics(pl.concat([emb, resid]), meta, ECFG, spaces=("standardised",))
    auc = dict(pairs.filter((pl.col("scope") == "pooled")).group_by("method").agg(pl.col("auc").mean()).iter_rows())
    assert auc["SYN"] > 0.85 and auc["SYN|resid_official"] > 0.75, auc            # the athlete trait is still repeatable
    assert _raises(ValueError, lambda: rv.residualise_embeddings(emb, meta, "everything", ECFG, 2))
    return "probes never score an athlete they were fitted on; removing official results leaves the planted athlete trait repeatable"


def check_matching_fpca_and_tcn_cohorts() -> str:
    b = sprint_fixture(n_athletes=80)
    cs, splits = b["curves"], b["splits"]
    fpca_models = {NAME: fe.fit_all({NAME: cs}, splits, fv.FCFG)[NAME].models}
    ref = rv.fpca_method(NAME, 2)
    plan = tcn.tcn_plan(TINY, TINY_B, [NAME], {NAME: 2})
    supports = tcn.evaluation_supports({NAME: cs}, splits, fpca_models, [0], TINY)
    for (_, r, f), sup in supports.items():
        assert np.array_equal(sup, fpca_models[NAME][fe.fold_tag(r, f)].support)      # one support for every method
    fp_tables = tcn.fpca_reference({NAME: cs}, splits, [ref], fpca_models, supports, [0], ECFG, seed=6)
    tc_tables, models = tcn.fit_methods({NAME: cs}, splits, plan, [0], ECFG, seed=6, supports=supports)
    emb = pl.concat([fp_tables[rv.EMBEDDINGS.name], tc_tables[rv.EMBEDDINGS.name]])
    recon = pl.concat([fp_tables[rv.RECON_SEQ.name], tc_tables[rv.RECON_SEQ.name]])
    match = rv.cohort_match(emb)
    assert match["n_sequences"].n_unique() == 1 and (match["n_sequences"] == match["n_sequences_common_to_all"]).all()
    assert rv.restrict_to_common(emb, [[ref.method, "TCNA_A", "TCNB_A"]]).height == emb.height
    per = recon.group_by("seq_id").agg(pl.len().alias("methods"), pl.col("n_points").n_unique().alias("points"),
                                       pl.col("n_masked").n_unique().alias("masked"))
    assert (per["methods"] == 3).all() and (per["points"] == 1).all() and (per["masked"] == 1).all()   # same points, same mask
    old = fe.fit_tables(fe.fit_all({NAME: cs}, splits, fv.FCFG))[fe.SEQ_RECON.name].filter((pl.col("k") == 2) & (pl.col("repeat") == 0))
    j = recon.filter(pl.col("method") == ref.method).join(old.select("seq_id", pl.col("sse").alias("_sse")), on="seq_id")
    assert j.height == old.filter(pl.col("scored")).height and np.allclose(j["sse"].to_numpy(), j["_sse"].to_numpy(), rtol=1e-6)
    methods = tcn.methods_table([ref, *[s for s, _ in plan]], tc_tables[tcn.FOLDS.name], fpca_models)
    conform(methods, rv.METHODS)
    assert methods.filter(pl.col("family") == "tcn")["n_parameters"].min() > 0 and rv.reference_map(methods)["TCNB_A|resid_official"] == f"{ref.method}|resid_official"
    ev = rv.evaluate(emb, recon, methods, b["windows"], b["athletes"], splits, ECFG, seed=6)
    for c in rv.EVAL_CONTRACTS:
        conform(ev[c.name], c)
    assert set(ev[rv.REPEAT.name]["method"].unique().to_list()) >= {ref.method, "TCNA_A", "TCNB_A", "TCNA_A|resid_official"}
    assert ev[rv.RECON.name].filter(pl.col("method") == "TCNA_A")["rmse_diff"].null_count() == 0
    assert ev[rv.TRUNC.name]["view"].n_unique() == len(ECFG.truncations)
    gated = rv.gate(ev, methods, ECFG)
    s, v = conform(gated[rv.SUMMARY.name], rv.SUMMARY), conform(gated[rv.VERDICT.name], rv.VERDICT)
    assert s.height == 3 and v["method"].to_list() == ["TCNA_A", "TCNB_A"] and all(x[0] in "ABCD" for x in v["verdict"].to_list())
    text = rv.conclusions_markdown(s, v, ECFG, "smoke")
    assert "must not be interpreted" in text and "Go / no-go" in text
    assert _raises(ValueError, lambda: rv.single_repeat(pl.concat([emb, emb.with_columns(pl.lit(1, dtype=pl.Int64).alias("repeat"))])))
    assert rv.gain_label(0.01, 0.05, 0.02) == "better" and rv.gain_label(-0.05, 0.01, 0.02) == "not_better"
    assert rv.gain_label(-0.05, 0.05, 0.02) == "uncertain" and rv.gain_label(None, 0.1, 0.02) == "uncertain"
    return (f"FPCA and both encoders are scored on the same {match['n_sequences'][0]} held-out attempts, grid points and hidden "
            "points; the FPCA sums equal those of notebook 02; every evaluation table satisfies its contract")


def check_design_only_selection() -> str:
    b = sprint_fixture()
    cs, meta = b["curves"], b["meta"]
    half = b["splits"].with_columns(pl.when(pl.col("nfl_id") % 2 == 0).then(pl.lit("design")).otherwise(pl.lit("confirm")).alias("design_role"))
    grid = [("vicreg", 0.3)]
    sweep = conform(tcn.design_sweep(cs, half, meta, TINY, grid, ECFG, seed=8), tcn.DESIGN)
    assert sweep["consistency"].to_list() == ["none", "vicreg"] and (sweep["n_folds"] == ACFG.inner_folds).all()
    moved = cs.Y.copy()
    confirm = np.isin(cs.nfl_ids, half.filter(pl.col("design_role") == "confirm")["nfl_id"].unique().to_list())
    moved[confirm] = moved[confirm] * 2.0 - 3.0                              # change the confirm athletes only
    again = tcn.design_sweep(fp.CurveSet(cs.spec, cs.grid, moved, cs.t_orig, cs.seq_ids, cs.nfl_ids), half, meta, TINY, grid, ECFG, seed=8)
    for col in ("auc", "rmse_masked", "rank_share_min"):
        assert np.allclose(sweep[col].fill_null(-1).to_numpy(), again[col].fill_null(-1).to_numpy(), atol=1e-6), col
    table = pl.DataFrame({"consistency": ["none", "mse", "vicreg", "vicreg", "vicreg"], "consistency_weight": [0.0, 0.3, 0.1, 0.3, 1.0],
                          "n_folds": [4] * 5, "n_athletes_repeat": [90] * 5, "auc": [0.99, 0.80, 0.700, 0.703, 0.90],
                          "auc_fold_min": [0.5] * 5, "auc_fold_max": [0.9] * 5, "top1": [0.2] * 5, "rmse_masked": [0.1] * 5,
                          "rank_share_min": [0.5] * 5, "near_identical_max": [0.0] * 5, "collapsed": [False, False, False, False, True],
                          "selected": [False] * 5}, schema=tcn.DESIGN.columns)
    marked, kind, weight, _ = tcn.select_consistency(table, ("vicreg", 0.3))
    assert (kind, weight) == ("mse", 0.3) and marked["selected"].sum() == 1     # never the reference row, never a collapsed one
    _, kind, weight, _ = tcn.select_consistency(table.filter(pl.col("consistency") != "mse"), ("mse", 9.0))
    assert (kind, weight) == ("vicreg", 0.1)                                   # a tie goes to the smaller weight
    _, kind, weight, reason = tcn.select_consistency(table.filter(pl.col("consistency") == "none"), ("vicreg", 0.3))
    assert (kind, weight) == ("vicreg", 0.3) and "default" in reason
    return "the consistency weight is ranked on design athletes; changing every confirm athlete does not change the ranking table"


TCN_CHECKS: tuple[Callable[[], str], ...] = (
    check_athlete_group_split_separation, check_masked_reconstruction_loss, check_augmentation_physical_validity,
    check_variable_length_padding, check_channel_availability, check_attempt_pairing, check_no_cross_fold_pair_leakage,
    check_embedding_extraction, check_representation_collapse_detection, check_checkpoint_reload_compatibility,
    check_retrieval_calculations, check_bootstrap_sampling, check_nuisance_probe_training_separation,
    check_matching_fpca_and_tcn_cohorts, check_design_only_selection)


def run_all(checks: tuple[Callable[[], str], ...] = TCN_CHECKS) -> pl.DataFrame:
    """Run every check in-process. A failing check is reported, not raised, so one table shows them all."""
    return base.run_all(checks)


assert_all_passed = base.assert_all_passed
