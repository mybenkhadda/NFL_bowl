"""Milestone 12 notebook-callable software checks on SYNTHETIC data. They verify that code does what its docstring says;
they are never evidence about athletes or about physical transfer. Tolerances were set by reasoning, not tuned on runs;
checks marked ``[draw]`` rest on a fixed seed.

Groups: CHECKS_GENERAL · CHECKS_FUNCTIONAL · CHECKS_MEASUREMENT · CHECKS_DISTRIBUTIONAL · CHECKS_MULTIVIEW · CHECKS_TRANSPORT ·
CHECKS_ENCODER (needs PyTorch; reported as SKIP without it).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from ..m11 import physics as ph
from ..m11.checkpoint import CheckpointMismatch, ShardCheckpoint
from ..m11.evaluation import LeakageError
from ..validation import failure_location
from . import advanced_evaluation as ae
from . import distributional_capacity as dc
from . import functional_movement as fm
from . import measurement_error as me
from . import movement_transport as mt
from . import multiview_traits as mv
from . import smoothing as sm
from .config import M12Config

C = M12Config()


class Skip(Exception):
    """Check not applicable in this environment (e.g. no PyTorch)."""


def _close(a: float, b: float, tol: float, what: str) -> None:
    if not (np.isfinite(a) and abs(a - b) <= tol):
        raise AssertionError(f"{what}: {a} vs expected {b} (tol {tol})")


def _frame(n: int = 120, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    pos = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    x = rng.standard_normal(n)
    return pl.DataFrame({"nfl_id": np.arange(n), "position_group": pos, "draft_year": rng.integers(2018, 2023, n).astype(float),
                         "height": rng.normal(74, 2, n), "weight": rng.normal(220, 20, n), "forty": rng.normal(4.6, 0.1, n), "ten": rng.normal(1.6, 0.05, n),
                         "x": x, "y": 0.8 * x + rng.standard_normal(n) * 0.5, "fold": np.arange(n) % 5})


# ============================================================================== general inference
def check_grouped_folds_and_leakage() -> str:
    d = _frame()
    d = d.with_columns(pl.col("fold").cast(pl.Int64))
    bad = pl.concat([d, d.filter(pl.col("nfl_id") == 3).with_columns(pl.lit(1, dtype=pl.Int64).alias("fold"))])
    bad = bad.with_columns(pl.when((pl.col("nfl_id") == 3) & (pl.col("fold") != 1)).then(pl.lit(4, dtype=pl.Int64)).otherwise(pl.col("fold")).alias("fold"))
    try:
        ae.oof_ladder(bad, "y", {"M": ae.Model(["x"])}, C, 0)
    except LeakageError:
        pass
    else:
        raise AssertionError("an athlete in two folds must raise LeakageError")
    return "athlete-grouped folds enforced (an athlete in two folds raises)"


def check_block_sees_training_only() -> str:
    d = _frame()
    seen = []

    def block(train: np.ndarray) -> pl.DataFrame:
        seen.append(train.copy())
        mu = d["x"].to_numpy()[train].mean()
        return pl.DataFrame({"xc": d["x"].to_numpy() - mu})
    res = ae.oof_ladder(d, "y", {"B": ae.Model([], block)}, C, 0)
    for tr in seen:
        held = set(np.unique(d["fold"].to_numpy()[~tr]).tolist())
        assert len(held) == 1, "a block must be called with exactly one held-out fold"
    d2 = d.with_columns(pl.when(pl.col("fold") == 0).then(pl.col("y") + 100).otherwise(pl.col("y")).alias("y"))
    r2 = ae.oof_ladder(d2, "y", {"B": ae.Model([], block)}, C, 0)
    assert np.allclose(res.preds["B"][d["fold"].to_numpy() == 0], r2.preds["B"][d["fold"].to_numpy() == 0]), "held-out labels changed held-out predictions"
    return "fold-fitted blocks receive one held-out fold; held-out labels never change held-out predictions"


def check_common_cohort_pairing() -> str:
    d = _frame().with_columns(pl.when(pl.col("nfl_id") % 7 == 0).then(None).otherwise(pl.col("x")).alias("x"))
    res = ae.oof_ladder(d, "y", {"M1": ae.Model(list(ae.M1_COLS)), "M2": ae.Model([*ae.M1_COLS, "x"])}, C, 0)
    assert all(np.isfinite(p).sum() == res.y.size for p in res.preds.values()), "every model must be scored on the same athletes"
    cmp = ae.compare(res, [("M2", "M1")], C, 0)
    assert cmp["n_athletes"][0] == res.y.size
    return "every model of a ladder is scored on the identical athletes (paired)"


def check_bootstrap_unit_and_seed() -> str:
    a, b = np.random.default_rng(1).random(50), np.random.default_rng(2).random(50)
    r1, r2 = ae.paired_metric_diff(a, b, 300, 7), ae.paired_metric_diff(a, b, 300, 7)
    assert r1 == r2, "same seed must reproduce the bootstrap"
    assert r1["lo"] <= r1["diff"] <= r1["hi"]
    _close(ae.spearman_brown(0.5, 2), 2 / 3, 1e-12, "Spearman–Brown")
    p = ae.holm([0.01, 0.04, 0.03])
    _close(p[0], 0.03, 1e-12, "Holm smallest")
    q = ae.bh([0.01, 0.04, 0.03])
    _close(q[1], 0.04, 1e-12, "BH largest")
    return "athlete bootstrap deterministic by seed; Spearman–Brown, Holm and Benjamini–Hochberg exact"


def check_checkpoint_resume_safety() -> str:
    with tempfile.TemporaryDirectory() as d:
        ck = ShardCheckpoint(Path(d), "x", {"a": 1}, "smoke")
        ck.save("k1", pl.DataFrame({"v": [1]}))
        try:
            ae.require_complete(ck, ["k1", "k2"], "test")
        except Exception as e:                                       # noqa: BLE001
            assert "missing" in str(e)
        else:
            raise AssertionError("an incomplete run must not pass the completion gate")
        ck.save("k2", pl.DataFrame({"v": [2]}))
        ae.require_complete(ck, ["k1", "k2"], "test")
        cfg = Path(d) / "x" / ck.dir.name / "config.json"
        cfg.write_text(cfg.read_text().replace(ck.hash, "deadbeef"), encoding="utf-8")
        try:
            ShardCheckpoint(Path(d), "x", {"a": 1}, "smoke")
        except CheckpointMismatch:
            pass
        else:
            raise AssertionError("a checkpoint of another definition must be refused")
    return "completion gate blocks partial runs; a checkpoint of another definition is refused"


def check_evidence_tier_rules() -> str:
    assert ae.evidence_row("a", "12C", delta_lo=0.01, delta_hi=0.02)["evidence_tier"] == "E4_heldout_physical_gain"
    assert ae.evidence_row("b", "12C", cross_domain_lo=-0.1, cross_domain_hi=0.2, reliability=0.9, reliability_lo=0.8)["evidence_tier"] == "E2_reliable"
    assert ae.evidence_row("c", "12C", delta_rmse=0.5, delta_lo=-0.1, delta_hi=0.9, n_athletes=10)["evidence_tier"] == "E1_measurable"
    assert ae.reading(-0.1, 0.2) == "inconclusive" and ae.reading(0.1, 0.2) == "supported" and ae.reading(-0.3, -0.1) == "reversed"
    return "evidence tiers never promote on a point estimate; interval readings exact"


# ============================================================================== functional
def check_spline_basis_and_support() -> str:
    x = np.linspace(1, 9, 50)
    B = sm.bspline_basis(x, 1, 9, 5)
    _close(float(np.max(np.abs(B.sum(1) - 1))), 0.0, 1e-9, "partition of unity inside the range")
    assert np.all(sm.bspline_basis(np.array([0.5, 9.5]), 1, 9, 5) == 0), "no basis value outside the supported range"
    y = np.sin(x)
    beta = sm.pspline_fit(B, y, None, 1e-3)
    _close(float(np.max(np.abs(B @ beta - y))), 0.0, 0.02, "P-spline reproduces a smooth curve")
    row, n = fm._bridge(np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0, np.nan]), 1)
    _close(row[1], 2.0, 1e-12, "one-bin interior gap bridged")
    assert np.isnan(row[3]) and np.isnan(row[4]) and np.isnan(row[6]) and n == 1, "longer gaps and ends are never filled"
    return "B-spline partition of unity, zero outside support; P-spline fit; only short interior gaps bridged"


def check_fpca_recovery_and_isolation() -> str:
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 12)
    phi1, phi2 = np.sin(np.pi * x), np.cos(np.pi * x)
    S = rng.standard_normal((200, 2)) * [2.0, 0.7]
    M = 1 + S[:, :1] * phi1 / np.linalg.norm(phi1) + S[:, 1:] * phi2 / np.linalg.norm(phi2) + 0.01 * rng.standard_normal((200, 12))
    f = fm.FPCA.fit(M, 0.95, 4)
    _close(abs(float(f.phi[:, 0] @ (phi1 / np.linalg.norm(phi1)))), 1.0, 0.02, "first component [draw]")
    Mp = M.copy()
    Mp[5, 3:] = np.nan
    s = f.scores(Mp)
    _close(float(np.corrcoef(s[:, 0], S[:, 0])[0, 1]), 1.0, 0.02, "scores with a partial curve [draw]")
    ids = np.arange(200)
    blk = fm.fpca_block(ids, ids, M, M12Config(fpca_var=0.95))
    tr = ids % 5 != 0
    a = blk(tr).to_numpy()
    M2 = M.copy()
    M2[~tr] += 5.0
    b = fm.fpca_block(ids, ids, M2, M12Config(fpca_var=0.95))(tr).to_numpy()
    assert np.allclose(a[tr], b[tr]), "training scores must not depend on held-out curves"
    return "FPCA recovers planted components; partial curves scored; fold-fitted FPCA ignores held-out curves"


def check_combine_curve_interpolation() -> str:
    from . import foundation as fd
    t = np.arange(0, 3.0, 0.1)
    v = np.minimum(0.5 + 5.0 * t, 9.0)                    # 5 yd/s² ramp: one frame per 0.5 yd/s bin at 10 Hz
    a = np.where(v < 9.0, 5.0, 0.0)
    rows = fd.attempt_curve(t, v, a, M12Config())
    bins = [r["speed_bin"] for r in rows]
    assert bins == list(range(min(bins), max(bins) + 1)) and min(bins) <= 2 and max(bins) >= 16, f"every bin crossed must be supported, got {bins}"
    assert all(abs(r["a_mean"] - 5.0) < 1e-9 for r in rows if r["speed_mid"] < 8.75), "interpolated a(v) of a constant ramp"
    assert all(r["speed_mid"] <= 9.0 for r in rows), "no bin above the peak speed (no extrapolation)"
    v2 = v.copy()
    v2[10:] += 2.0                                       # a 2 yd/s jump between two frames
    rows2 = fd.attempt_curve(t, v2, a, M12Config())
    assert not any(5.5 < r["speed_mid"] < 7.5 for r in rows2), "a speed gap > 1 yd/s between frames is not bridged"
    return "Combine a(v) at bin mid-speeds: every crossed bin supported at 10 Hz, exact on a ramp, no extrapolation, gaps > 1 yd/s not bridged"


def check_integrated_error_weighting() -> str:
    Y = np.array([[1.0, 1.0, np.nan, np.nan], [1.0, 1.0, 1.0, 1.0]])
    P = np.array([[0.0, 0.0, 9.0, 9.0], [1.0, 1.0, 1.0, 3.0]])
    e = fm.athlete_mse(Y, P)
    _close(e[0], 1.0, 1e-12, "only the athlete's own supported bins")
    _close(e[1], 1.0, 1e-12, "mean over bins, not sum (wider range does not weigh more)")
    return "functional error averages each athlete's own supported bins (equal athlete weight)"


def check_registration() -> str:
    g = np.linspace(0, 3, 31)
    v = 9 * (1 - np.exp(-g / 1.0))
    L = fm.landmarks(v[None, :], g)
    R = fm.landmark_register(v[None, :], g, L, L[0])
    _close(float(np.nanmax(np.abs(R[0] - v))), 0.0, 1e-9, "identity warp leaves the curve unchanged")
    w = g ** 1.3 / g[-1] ** 0.3
    v2 = np.interp(w, g, v)
    gam = fm.dp_align(fm.srsf(v, g), fm.srsf(v2, g), g)
    assert np.all(np.diff(gam) >= -1e-12) and abs(gam[0]) < 1e-9 and abs(gam[-1] - g[-1]) < 1e-9, "warp must be monotone and endpoint-preserving"
    aligned = np.interp(gam, g, v2)
    assert np.mean((aligned - v) ** 2) < np.mean((v2 - v) ** 2), "alignment must reduce the amplitude distance to the template"
    return "landmark registration is exact at identity; SRSF DP warps are monotone, endpoint-preserving and reduce misalignment"


# ============================================================================== measurement error
def check_icc_and_regression_calibration() -> str:
    rng = np.random.default_rng(4)
    n, k, s2t, s2e = 400, 3, 1.0, 1.0
    th = rng.standard_normal(n) * np.sqrt(s2t)
    x = np.repeat(th, k) + rng.standard_normal(n * k) * np.sqrt(s2e)
    ids = np.repeat(np.arange(n), k)
    comp = me.components(x, ids)
    _close(comp["icc"], 0.5, 0.06, "ICC [draw]")
    one = me.components(x[::k], ids[::k])
    assert not one["identifiable"], "one attempt per athlete must be flagged not identifiable"
    long = pl.DataFrame({"seq_id": [f"{i}-{j}" for i, j in zip(ids, np.tile(np.arange(k), n))], "nfl_id": ids, "feature": "f", "value": x})
    st = me.athlete_stats(long)
    v = me.rc_values(st, "f", {i: "A" for i in range(n)}, list(range(n)), long)
    lam = comp["sigma2_theta"] / (comp["sigma2_theta"] + comp["sigma2_eps"] / k)
    r = st.filter(pl.col("nfl_id") == 0).row(0, named=True)
    _close(v[0], comp["mean"] + lam * (r["xbar"] - comp["mean"]), 1e-9, "RC shrinkage formula")
    return "method-of-moments ICC recovers the planted value; single attempts are not identifiable; RC = μ + λ(x̄ − μ)"


def check_simex_and_em_recovery() -> str:
    s = me.Scenario("t", 0.4, 2, 0.4, n=500)
    long, ath, bt = me.simulate(s, np.random.default_rng(5))
    st = me.athlete_stats(long)
    d = ath.join(st.select("nfl_id", "xbar", "n_att"), on="nfl_id")
    comp = me.components(long["value"].to_numpy(), long["nfl_id"].to_numpy())
    e = me.estimands(d, "xbar", "y", "n_att", comp["sigma2_eps"], ["draft_year"], M12Config(n_boot=100, simex_reps=20), 0)
    assert abs(e["beta_rc"] - bt) < abs(e["beta_obs"] - bt), "RC must be less biased than the naive slope [draw]"
    assert abs(e["beta_simex"] - bt) < abs(e["beta_obs"] - bt), "SIMEX must be less biased than the naive slope [draw]"
    em = me.em_latent([g["value"].to_numpy() for _, g in sorted(long.partition_by(["nfl_id"], as_dict=True).items())], d.sort("nfl_id")["y"].to_numpy(), np.zeros((d.height, 0)), 300)
    _close(em["beta_std"], bt, 0.12, "EM standardized β [draw]")
    return "RC and SIMEX reduce attenuation bias; EM recovers the latent slope"


# ============================================================================== distributional
def check_quantiles_and_crossing() -> str:
    rng = np.random.default_rng(6)
    ab = np.arange(-8, 8.25, 0.25)
    H = np.stack([np.histogram(rng.normal(m, 1, 5000), ab)[0] for m in (0.0, 1.0, 2.0)]).astype(float)
    for t in (0.5, 0.9):
        q = dc.q_rows(H, t, ab, 10)
        ref = np.array([ph.hist_quantile(ab, H[k], t) for k in range(3)])
        _close(float(np.max(np.abs(q - ref))), 0.0, 1e-9, f"vectorised quantile τ={t}")
    Q = np.array([[1.0, 2.0], [0.5, 2.5], [3.0, 3.0]])
    R = sm.rearrange(Q)
    assert sm.crossing_rate(Q) > 0 and sm.crossing_rate(R) == 0, "rearrangement removes crossing"
    return "vectorised histogram quantiles equal the Milestone 11 rule; rearrangement removes quantile crossing"


def check_game_splits_and_heldout() -> str:
    c = M12Config(min_frames_bin_nfl=10)
    rng = np.random.default_rng(7)
    ns, na = ae.speed_edges(c).size - 1, ae.accel_edges(c).size - 1
    H = rng.integers(0, 3, (20, ns, na)).astype(np.int32)
    cb = dc.Cube(1, "A", np.arange(20), np.r_[np.full(10, 2022), np.full(10, 2023)], H)
    for kind in ("odd_even", "random", "season"):
        a, b = dc.game_splits(cb, kind, 1)
        assert not (a & b).any() and (a | b).all(), f"{kind} split must partition the games"
    amid = ae.mids(ae.accel_edges(c))
    Hb = np.zeros((ns, na))
    Hb[4] = np.histogram(rng.normal(1.0, 1.0, 20000), ae.accel_edges(c))[0]
    losses = {q: dc.held_out_pinball(np.full(ns, q), Hb, amid, 0.9, [4], c) for q in (1.0, 1.0 + 1.2816, 3.5)}
    assert losses[1.0 + 1.2816] < losses[1.0] and losses[1.0 + 1.2816] < losses[3.5], "held-out pinball is minimised at the true τ-quantile"
    _close(dc.held_out_coverage(np.full(ns, 1.0 + 1.2816), Hb, amid, [4], c), 0.9, 0.02, "coverage of the true quantile [draw]")
    return "game splits partition games (odd/even, random, season); held-out pinball is minimised and calibrated at the true quantile"


def check_pooling_and_trait() -> str:
    Q = {i: np.array([float(i), np.nan]) for i in range(10)}
    SE = {i: np.array([1.0, np.nan]) for i in range(10)}
    P = dc.pooled(Q, SE, {i: "A" for i in range(10)})
    mu = np.mean(range(10))
    for i in range(10):
        assert min(Q[i][0], mu) - 1e-9 <= P[i][0] <= max(Q[i][0], mu) + 1e-9, "pooled value lies between athlete and position mean"
        assert np.isnan(P[i][1]), "unsupported bins stay unsupported"
    assert np.isnan(dc.trait(np.array([1.0, np.nan, 3.0]), [0, 1, 2])) and dc.trait(np.array([1.0, 2.0, 3.0]), [0, 1, 2]) == 2.0
    return "partial pooling shrinks toward the position mean, keeps unsupported bins empty; traits need every shared bin"


# ============================================================================== multi-view
def check_missing_view_factor_model() -> str:
    rng = np.random.default_rng(8)
    n = 400
    f = rng.standard_normal(n)
    W = np.array([0.9, 0.8, 0.7, 0.8, 0.6, 0.7])
    X = f[:, None] * W + rng.standard_normal((n, 6)) * 0.5
    views = ["a", "a", "b", "b", "c", "c"]
    X[rng.random(n) < 0.4, 2:4] = np.nan                       # view b missing for 40 %
    m = mv.MissingFA(*mv.model_spec("D2", 1, views)).fit(X, 200, 0)
    _close(abs(mv.tucker(m.W[:, 0], W)), 1.0, 0.02, "loading congruence [draw]")
    s = m.scores(X)
    _close(abs(float(np.corrcoef(s[:, 0], f)[0, 1])), 0.9, 0.08, "factor score recovery [draw]")
    Xs = X.copy()
    Xs[0, 2:] = 100.0
    s1 = m.scores(np.where(np.isfinite(X), X, np.nan)[:1])
    Xo = X[:1].copy()
    Xo[0, 2:4] = np.nan
    assert np.isfinite(m.scores(Xo)).all(), "scores computed from observed views only"
    pos = np.array(["A"] * 200 + ["B"] * 200)
    tr = np.arange(n) < 300
    Xc = mv.position_center(X, pos, tr)
    X2 = X.copy()
    X2[~tr] += 10
    assert np.allclose(mv.position_center(X2, pos, tr)[tr], Xc[tr], equal_nan=True), "position centring uses training athletes only"
    return "missing-view EM factor model recovers loadings and scores; scores use observed views; centring is training-only"


def check_structured_mask_and_alignment() -> str:
    mask, names = mv.structure_mask(["a", "a", "b"], 1, True)
    assert mask.tolist() == [[True, True, False], [True, True, False], [True, False, True]] and names == ["shared0", "view:a", "view:b"]
    A = np.random.default_rng(9).standard_normal((6, 2))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    _close(float(np.max(np.abs(mv.procrustes(A, A @ R) - A))), 0.0, 1e-9, "Procrustes undoes a rotation")
    return "structured multi-view mask and Procrustes alignment are exact"


# ============================================================================== optimal transport
def check_transport_properties() -> str:
    c = M12Config()
    g = mt.Grid(c, list(range(4, 12)))
    rng = np.random.default_rng(10)
    P = rng.random(g.points.shape[0]).reshape(len(g.speed_bins), -1)
    Q = rng.random(P.shape)
    _close(mt.sinkhorn_divergence(P, P, g, 0.05, 300), 0.0, 1e-6, "Sinkhorn divergence of identical distributions")
    _close(mt.sliced_w2(P, P, g, 16), 0.0, 1e-9, "sliced W2 of identical distributions")
    _close(mt.conditional_w1(P * 100, P * 100, g), 0.0, 1e-12, "conditional W1 of identical distributions")
    _close(mt.sinkhorn_divergence(P, Q, g, 0.05, 300), mt.sinkhorn_divergence(Q, P, g, 0.05, 300), 1e-5, "symmetry")
    S = np.zeros_like(P)
    S[:, 2] = 1.0
    T = np.zeros_like(P)
    T[:, 6] = 1.0
    _close(mt.conditional_w1(S, T, g), 4 * g.da, 1e-9, "conditional W1 of a shifted point mass = shift")
    E = np.zeros_like(P)
    assert np.isnan(mt.conditional_w1(S, E, g)), "no shared supported speed bin → NaN, never 0"
    d1, d2 = mt.sinkhorn_divergence(P, Q, g, 0.01, 500), mt.sinkhorn_divergence(P, Q, g, 0.5, 500)
    assert np.isfinite(d1) and np.isfinite(d2), "regularisation grid runs"
    u = mt.unbalanced(P / P.sum(), 0.5 * Q / Q.sum(), g, 0.05, 1.0, 300)
    assert np.isfinite(u) and u > 0, "unbalanced OT handles unequal mass"
    return "OT distances vanish on identical distributions, are symmetric, respect common support and handle unequal mass"


# ============================================================================== encoder (torch)
def check_encoder_shapes_masks_gradients() -> str:
    from . import shared_movement_encoder as se
    if not se.TORCH:
        raise Skip("PyTorch not installed")
    import torch
    e = se.EncConfig("t", embed_dim=8, width=8, layers=2)
    m = se.SharedEncoder(e)
    B, L = 6, se.STORE_LEN
    x = torch.randn(B, L, 6)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i in range(B):
        mask[i, : 10 + 5 * i] = True
    emb = m.embed(x, mask)
    assert emb.shape == (B, 8)
    x2 = x.clone()
    x2[~mask] = 1e3
    assert torch.allclose(m.embed(x2, mask), emb, atol=1e-4), "padded frames must not change the embedding"
    phys = torch.randn(B, len(se.PHYS))
    ok = torch.ones(B, len(se.PHYS), dtype=torch.bool)
    st = torch.tensor([0, 0, 1, 1, 2, 2])
    dom = torch.tensor([0, 1, 0, 1, 0, 1])
    loss, parts = se.batch_loss(m, x, mask, phys, ok, st, st, dom, torch.Generator().manual_seed(0))
    loss.backward()
    assert set(parts) == {"mask", "phys", "temp", "align"}, f"loss decomposition {set(parts)}"
    assert all(p.grad is not None for p in m.encoder.parameters() if p.requires_grad), "gradients reach every encoder parameter"
    meta = pl.DataFrame({"nfl_id": [1, 1, 2, 2, 3, 3], "kind": ["forty"] * 6})
    tr, va, ho = se.split_indices(meta, {1: 0, 2: 1, 3: 2}, 0, 0.5, 0)
    assert set(meta["nfl_id"].to_numpy()[ho]) == {1} and not (set(meta["nfl_id"].to_numpy()[tr]) & set(meta["nfl_id"].to_numpy()[va])), "grouped splits"
    return "variable-length masks respected, batch shapes, loss decomposition, gradient flow, athlete-grouped splits"


CHECKS_GENERAL: tuple[Callable[[], str], ...] = (check_grouped_folds_and_leakage, check_block_sees_training_only, check_common_cohort_pairing,
                                                 check_bootstrap_unit_and_seed, check_checkpoint_resume_safety, check_evidence_tier_rules)
CHECKS_FUNCTIONAL = (check_spline_basis_and_support, check_fpca_recovery_and_isolation, check_combine_curve_interpolation, check_integrated_error_weighting,
                     check_registration)
CHECKS_MEASUREMENT = (check_icc_and_regression_calibration, check_simex_and_em_recovery)
CHECKS_DISTRIBUTIONAL = (check_quantiles_and_crossing, check_game_splits_and_heldout, check_pooling_and_trait)
CHECKS_MULTIVIEW = (check_missing_view_factor_model, check_structured_mask_and_alignment)
CHECKS_TRANSPORT = (check_transport_properties,)
CHECKS_ENCODER = (check_encoder_shapes_masks_gradients,)
ALL_CHECKS = (*CHECKS_GENERAL, *CHECKS_FUNCTIONAL, *CHECKS_MEASUREMENT, *CHECKS_DISTRIBUTIONAL, *CHECKS_MULTIVIEW, *CHECKS_TRANSPORT, *CHECKS_ENCODER)


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    rows = []
    for fn in checks:
        name = fn.__name__.removeprefix("check_")
        try:
            rows.append({"check": name, "status": "PASS", "detail": fn()})
        except Skip as s:
            rows.append({"check": name, "status": "SKIP", "detail": str(s)})
        except Exception as e:                                        # noqa: BLE001 — the table lists every failure
            rows.append({"check": name, "status": "FAIL", "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


def assert_all_passed(df: pl.DataFrame) -> None:
    bad = df.filter(pl.col("status") == "FAIL")
    if bad.height:
        raise AssertionError(f"{bad.height} Milestone 12 check(s) failed: {bad['check'].to_list()} — fix before running the experiment")


__all__ = ["ALL_CHECKS", "CHECKS_DISTRIBUTIONAL", "CHECKS_ENCODER", "CHECKS_FUNCTIONAL", "CHECKS_GENERAL", "CHECKS_MEASUREMENT", "CHECKS_MULTIVIEW",
           "CHECKS_TRANSPORT", "Skip", "assert_all_passed", "run_all"]
