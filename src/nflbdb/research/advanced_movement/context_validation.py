"""Synthetic validation of Milestone 07 (notebooks 07A–07D), callable from a notebook cell. NumPy / Polars; no PyTorch,
no project data. Every fixture is generated here with a planted, known answer; tolerances were set by reasoning (the
author executed nothing) and each assertion states what it compares.

    from nflbdb.research.advanced_movement import context_validation as cv
    report = cv.run_all(cv.CHECKS_07A); display(report); cv.assert_all_passed(report)

A failing check is reported with its file, line and source text; it is never weakened to pass.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from . import context_outcomes as co
from . import context_reliability as cr
from . import physical_transfer as pt
from . import transfer_models as tm
from . import transfer_power as tp
from .scoring import fingerprint as fg
from .scoring import percentiles as pc
from .scoring import scorecards as sc
from .scoring import uncertainty as un
from .transfer import TransferConfig
from .validation import _raises, assert_all_passed, failure_location

CC = co.ContextConfig(min_windows=20, min_half_windows=3, min_cell_windows=5, min_cell_athletes=2, n_boot=40, n_game_splits=8,
                      first_games_grid=(2,), first_windows_grid=(20,), holdout_min_windows=10, min_athletes=8, min_athletes_position=4)
TC = replace(TransferConfig(), inner_folds=3, alpha_grid=(-1.0, 3.0, 5), n_boot=200, min_athletes=8, min_athletes_group=4)


# ============================================================================== fixtures
def windows_fixture(n_athletes: int = 60, n_games: int = 8, per_game: int = 10, s2a: float = 1.0, s2g: float = 0.0, s2w: float = 4.0,
                    band_effect: float = 0.0, seed: int = 0) -> tuple[co.Windows, np.ndarray]:
    """Balanced nested layout with planted components; returns (windows, true athlete effect). Athletes alternate
    between two positions (position means 5 and 8). With ``band_effect`` the window value depends on its
    starting-speed band and athletes differ in how often they start fast — a planted opportunity confound."""
    rng = np.random.default_rng(seed)
    ath = np.repeat(np.arange(n_athletes), n_games * per_game)
    game = np.tile(np.repeat(np.arange(n_games), per_game), n_athletes) + 100 * ath
    pos = np.where(ath % 2 == 0, "DB", "WR")
    eff = rng.normal(0, np.sqrt(s2a), n_athletes)
    geff = rng.normal(0, np.sqrt(s2g), n_athletes * n_games) if s2g > 0 else np.zeros(n_athletes * n_games)
    fast_share = np.linspace(0.1, 0.9, n_athletes)[rng.permutation(n_athletes)]
    fast = rng.random(ath.size) < fast_share[ath]
    v0 = np.where(fast, 3.0, 0.5)
    y = (np.where(ath % 2 == 0, 5.0, 8.0) + eff[ath] + geff[ath * n_games + (game - 100 * ath)] + band_effect * fast
         + rng.normal(0, np.sqrt(s2w), ath.size))
    w = co.Windows.make(ath + 1, y, game, pos, v0=v0, align=np.where(ath % 4 < 2, "L", "R"), unit="yd/s²", measure="accel_fixture", outcome_id="O.fixture")
    return w, eff


def design_fixture(n: int = 160, k: int = 4, seed: int = 0) -> tm.Design:
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    return tm.Design(1000 + i, np.where(i % 2 == 0, "DB", "WR"), np.where((i // 4) % 2 == 0, "2023", "2024"),
                     {"cmb_combine_weight": rng.normal(200, 10, n), "cmb_forty": rng.normal(4.6, 0.1, n)}, ("cmb_combine_weight",),
                     ("cmb_combine_weight", "cmb_forty"), (), (i // 2) % k, (i // (2 * k)) % 3)


def blocks_fixture(d: tm.Design, seed: int = 1) -> tm.ArrayBlocks:
    rng = np.random.default_rng(seed)
    return tm.ArrayBlocks(d.ids, {"ENG": rng.normal(size=(d.n, 1)), "SPEED2": rng.normal(size=(d.n, 2)), "FPCA": rng.normal(size=(d.n, 2)),
                                  "TCN": rng.normal(size=(d.n, 3)), "RAND": rng.normal(size=(d.n, 3))})


def oof_fixture(d: tm.Design, y: np.ndarray, preds: dict[str, np.ndarray], target: str = "T", analysis: str = "primary", cohort: str = "primary") -> pl.DataFrame:
    res = {"y": y, "y_ref": np.full(d.n, float(y.mean())), "n_train": np.full(d.n, d.n - 1, dtype=np.int64), "pred": preds,
           "alpha": {m: np.full(d.n, 1.0) for m in preds}}
    return tm.oof_frame(analysis, cohort, target, d, res)


# ============================================================================== 07A
def check_context_group_consistency() -> str:
    w, _ = windows_fixture(12, 4, 5)
    assert w.n_athletes == 12 and w.n_windows == 240
    assert w.ag_athlete.size == 48, "one athlete-game code per (athlete, game)"
    assert np.array_equal(w.ag_athlete[w.ag], w.a), "every window's athlete-game belongs to its athlete"
    assert w.g_order.min() == 1 and w.g_order.max() == 4 and w.w_order.max() == 20
    lab = co.ContextLabels(w, CC)
    h = cr.game_halves(lab, np.random.default_rng(0))
    per_game = np.array([np.unique(h[w.ag == g]).size for g in range(48)])
    assert (per_game == 1).all(), "a game is never split between the two halves"
    n1 = np.bincount(w.a[h == 1], minlength=12) / 5
    assert (n1 == 2).all(), "4 games per athlete → 2 games in each half"
    return "athlete-game codes, time orders and whole-game halves are consistent"


def check_game_athlete_boundaries() -> str:
    w, _ = windows_fixture(30, 6, 8, band_effect=2.0)
    lab = co.ContextLabels(w, CC)
    h = cr.game_halves(lab, None)
    base = lab.values("C2", fit=h == 1, score=h == 0)["value"].copy()
    y0 = w.y.copy()
    w.y[h == 0] += 50.0 * (w.a[h == 0] == 3)                 # change evaluation windows of athlete 3 only
    moved = co.ContextLabels(w, CC).values("C2", fit=h == 1, score=h == 0)["value"]
    w.y[:] = y0
    others = np.arange(30) != 3
    assert np.allclose(base[others], moved[others]), "parameters come from the OTHER half: other athletes' values must not move"
    assert abs(moved[3] - base[3] - 50.0) < 1e-6, "the athlete's own evaluation windows carry his value"
    return "cross-fitted halves: evaluation games never inform the adjustment applied to them"


def check_physical_units() -> str:
    assert np.allclose(pt.v4_v8_mean_accel(np.array([1.0, 2.0, 0.5])), [4.0, 2.0, 8.0]), "mean acceleration over 4 → 8 yd/s = 4 / time"
    assert np.isnan(pt.v4_v8_mean_accel(np.array([0.0]))[0])

    def frames(dt: float) -> pl.DataFrame:
        return pl.DataFrame({"seq_id": ["a"] * 20 + ["b"] * 20, "frame": list(range(20)) * 2, "t_rel": [i * dt for i in range(20)] * 2})

    cu, nu = {pt.P95: "yd/s", pt.V4_V8: "s"}, {"O1.speed_p90_mean": "yd/s", "O3.band_acc_v4_6_mean": "yd/s²"}
    ok = pt.unit_audit(frames(0.1), 100.0, cu, nu, "sg7p2", "sg7p2")
    assert ok["passed"].all(), "10 Hz on both sides, same speed unit"
    bad = pt.unit_audit(frames(0.04), 100.0, cu, {**nu, "O1.speed_p90_mean": "m/s"}, "sg7p2", "sg7p2")
    failed = set(bad.filter(~pl.col("passed"))["quantity"].to_list())
    assert {"sampling frequency", "speed"} <= failed, f"a 25 Hz source and a different speed unit must fail, got {failed}"
    assert abs(tp.attenuation(0.5, 0.81, 1.0) - 0.45) < 1e-12
    return "4 / time conversion, sampling-rate and unit checks, attenuation formula"


def check_speed_band_conditions() -> str:
    w = co.Windows.make([1] * 6, [0.0] * 6, [1] * 6, ["DB"] * 6, v0=[0.0, 0.999, 1.0, 3.99, 4.0, np.nan])
    cc = co.ContextConfig()
    assert w.band(cc).tolist() == [0, 0, 1, 2, 3, -1], f"half-open bands [lo, hi), missing → −1, got {w.band(cc).tolist()}"
    assert w.frame(cc)["start_band"].to_list()[-1] == co.NA
    lab = co.ContextLabels(co.Windows.make([1, 2], [1.0, 2.0], [1, 2], ["DB", "DB"]), CC)
    assert not lab.available("C2")[0] and _raises(co.ContextError, lambda: lab.adjust("C2")), "no starting speed → C2 abstains"
    return "starting-speed bands are half-open, missing values are not assigned, C2 abstains without the field"


def check_athlete_game_resampling() -> str:
    w, _ = windows_fixture(40, 6, 10, s2g=0.5)
    st = co.athlete_stats(w, w.y, np.ones(w.n_windows, dtype=bool))
    a = co.components(st, w.pos_a)
    b = co.components(st, w.pos_a, np.ones(40))
    assert all(abs(a[k] - b[k]) < 1e-12 for k in ("s2_window", "s2_game", "s2_athlete")), "unit weights = no resampling"
    w2 = co.Windows.make(np.r_[w.ids[w.a], w.ids[w.a] + 1000], np.r_[w.y, w.y], np.r_[w.game, w.game + 10 ** 6], np.r_[w.position[w.a], w.position[w.a]])
    c = co.components(co.athlete_stats(w2, w2.y, np.ones(w2.n_windows, dtype=bool)), w2.pos_a)
    d = co.components(st, w.pos_a, np.full(40, 2.0))
    assert abs(c["s2_window"] - d["s2_window"]) < 1e-9 and abs(c["s2_game"] - d["s2_game"]) < 1e-9 and abs(c["s2_athlete"] - d["s2_athlete"]) < 1e-9, \
        "an athlete drawn twice counts as two athletes with identical windows"
    assert d["n_athletes"] == 80 and d["n_windows"] == 2 * w.n_windows
    return "bootstrap multiplicities reproduce an explicitly duplicated cohort (athletes are the resampled unit)"


def check_context_standardization() -> str:
    w, eff = windows_fixture(80, 8, 12, s2w=1.0, band_effect=3.0, seed=3)
    lab = co.ContextLabels(w, CC)
    c1 = lab.values("C1")["value"]
    for p in ("DB", "WR"):
        k = w.position == p
        assert abs(c1[k].mean()) < 1e-9 and abs(c1[k].std(ddof=1) - 1) < 1e-9, "C1 is a z-score within position"
    within = lambda v: np.corrcoef(cr._centre(v, w.pos_a), cr._centre(eff, w.pos_a))[0, 1]
    r0, r2, r4 = within(lab.values("C0")["value"]), within(lab.values("C2")["value"]), within(lab.values("C4")["value"])
    assert r2 > r0 + 0.05 and r4 > r0 + 0.05, f"removing a planted starting-speed confound must recover the athlete effect better: C0 {r0:.3f}, C2 {r2:.3f}, C4 {r4:.3f}"
    assert r2 > 0.9, f"C2 should track the planted athlete effect, r = {r2:.3f}"
    return f"C1 z-scores; planted opportunity confound: within-position r with the true effect C0 {r0:.2f} → C2 {r2:.2f}, C4 {r4:.2f}"


def check_hierarchical_model() -> str:
    w, eff = windows_fixture(120, 10, 20, s2a=1.0, s2g=0.25, s2w=4.0, seed=5)
    st = co.athlete_stats(w, w.y, np.ones(w.n_windows, dtype=bool))
    c = co.components(st, w.pos_a)
    assert abs(c["s2_window"] - 4.0) < 0.3, f"window variance {c['s2_window']:.3f}, planted 4"
    assert abs(c["s2_game"] - 0.25) < 0.12, f"game variance {c['s2_game']:.3f}, planted 0.25"
    assert abs(c["s2_athlete"] - 1.0) < 0.4, f"athlete variance {c['s2_athlete']:.3f}, planted 1"
    lam = co.shrink_factor(c, st["n"], st["G"])
    assert ((lam > 0) & (lam < 1)).all()
    lab = co.ContextLabels(w, CC)
    v4 = lab.values("C4")["value"]
    assert np.corrcoef(cr._centre(v4, w.pos_a), cr._centre(eff, w.pos_a))[0, 1] > 0.9, "the shrunk athlete effect recovers the planted one"
    assert v4[w.position == "DB"].std() < lab.values("C0")["value"][w.position == "DB"].std(), "shrinkage pulls towards the position mean"
    w0, _ = windows_fixture(60, 6, 10, s2a=0.0, s2g=0.0, s2w=4.0, seed=6)
    c0 = co.components(co.athlete_stats(w0, w0.y, np.ones(w0.n_windows, dtype=bool)), w0.pos_a)
    assert c0["s2_athlete"] < 0.15, f"no planted athlete variance → estimate near 0, got {c0['s2_athlete']:.3f}"
    return f"planted (1, 0.25, 4) recovered as ({c['s2_athlete']:.2f}, {c['s2_game']:.2f}, {c['s2_window']:.2f}); athlete effects recovered"


def check_fold_safe_outcome() -> str:
    w, _ = windows_fixture(40, 6, 10, band_effect=2.0, seed=7)
    train = w.ids[:30]
    held = np.isin(w.ids[w.a], w.ids[30:])
    out = {}
    for d in ("C1", "C2", "C3", "C4"):
        base = co.ContextLabels(w, CC).label_fn(d, w.ids)(train).copy()
        y0 = w.y.copy()
        w.y[held] += 7.0                                   # alter every window of the held-out athletes
        moved = co.ContextLabels(w, CC).label_fn(d, w.ids)(train)
        w.y[:] = y0
        assert np.allclose(base[:30], moved[:30], atol=1e-9), f"{d}: training labels changed when held-out windows changed — label leakage"
        assert not np.allclose(base[30:], moved[30:]), f"{d}: held-out labels must follow their own windows"
        out[d] = float(np.abs(moved[:30] - base[:30]).max())
    return "context parameters come from the training athletes only for C1–C4 (training labels invariant to held-out windows)"


def check_outcome_reliability() -> str:
    w, _ = windows_fixture(120, 8, 10, s2a=1.0, s2g=0.0, s2w=4.0, seed=9)
    lab = co.ContextLabels(w, CC)
    rel = cr.reliability_table(lab, ["C0", "C1", "C4"], seed=1, progress=False)
    r = rel.filter((pl.col("definition") == "C0") & (pl.col("method") == "random_game_halves_crossfit") & (pl.col("scope") == "ALL"))
    assert r.height == 1 and r["n_athletes"][0] == 120
    expected = 2 * (1 / 1.1) / (1 + 1 / 1.1)                # halves of 40 windows: r = 1 / (1 + 4 / 40); Spearman–Brown
    assert abs(r["reliability"][0] - expected) < 0.04, f"reliability {r['reliability'][0]:.3f}, closed form {expected:.3f}"
    assert r["ci_lower"][0] < r["reliability"][0] + 0.03 and r["ci_upper"][0] > r["reliability"][0] - 0.03
    ex = cr.exposure_table(lab, ["C0"], seed=1)
    assert ex.filter(pl.col("method") == "first_games:2").height == 1
    ov = co.overlap_table(lab, ["C0", "C1", "C2", "C3", "C4"])
    tr = cr.tradeoff_table(lab, ov, rel, ["C0", "C1", "C4"])
    assert tr.height == 3 and tr.filter(pl.col("definition") == "C1")["spearman_with_c0"][0] > 0.97, "C1 is monotone in C0 within position"
    c0, c1 = lab.values("C0")["value"], lab.values("C1")["value"]
    assert all(np.array_equal(np.argsort(c0[w.position == p]), np.argsort(c1[w.position == p])) for p in ("DB", "WR")), "same order inside a position"
    return f"random game halves reproduce the closed-form reliability ({r['reliability'][0]:.3f} vs {expected:.3f})"


# ============================================================================== 07B
def _transfer_fixture(beta: float = 1.0, seed: int = 0) -> tuple[tm.Design, tm.ArrayBlocks, co.ContextLabels, np.ndarray]:
    d = design_fixture(160, 4, seed)
    blocks = blocks_fixture(d, seed + 1)
    rng = np.random.default_rng(seed + 2)
    truth = np.where(d.position == "DB", 5.0, 8.0) + beta * blocks.blocks["ENG"][:, 0] + rng.normal(0, 0.5, d.n)
    ath = np.repeat(np.arange(d.n), 40)
    w = co.Windows.make(d.ids[ath], truth[ath] + rng.normal(0, 1.0, ath.size), np.tile(np.repeat(np.arange(4), 10), d.n) + 10 * ath, d.position[ath],
                        outcome_id="O.fixture", unit="yd/s²")
    return d, blocks, co.ContextLabels(w, CC), truth


def check_fold_safe_transfer_fitting() -> str:
    d, blocks, lab, _ = _transfer_fixture()
    o = pt.OutcomeDef("Y", "O.fixture", "C0", None, "fixture", "synthetic")
    models = [m for m in pt.models_for(TC, ["physical", "control"]) if m.model in ("MEAN", "POS", "M0", "M1", "M2", "M6")]
    oof, cov = pt.run_outcome(o, lab, d, blocks, models, TC)
    assert cov["evaluated"] and cov["n_eligible"] == d.n
    assert oof.height == d.n * len(models) and oof.select(pl.struct("model", "nfl_id").n_unique()).item() == oof.height, "one prediction per athlete and model"
    assert oof["status"].unique().to_list() == [pt.STATUS], "every prediction row carries the exploratory status"
    ev = tm.evaluate(oof.drop("status"), [("M2", "M1"), ("M6", "M1")], TC, seed=1)
    p = {r["model"]: r for r in ev[tm.PAIRED.name].filter(pl.col("metric") == "delta_rmse").iter_rows(named=True)}
    assert p["M2"]["estimate"] > 0 and p["M2"]["established"], f"a planted engineered signal must be recovered: {p['M2']['estimate']:.3f}"
    assert p["M6"]["estimate"] < p["M2"]["estimate"] / 4, "an unrelated block must not reproduce the gain"
    small = d.subset(np.arange(d.n) < 6)
    assert not pt.run_outcome(o, lab, small, blocks, models, TC)[1]["evaluated"], "too few athletes → not evaluated, with a reason"
    return f"planted signal recovered out of fold (Δ RMSE {p['M2']['estimate']:+.3f}); unrelated block {p['M6']['estimate']:+.3f}"


def check_oof_uniqueness_and_sign() -> str:
    d = design_fixture(40, 4)
    y = np.linspace(0, 1, d.n)
    oof = oof_fixture(d, y, {"GOOD": y + 0.1, "BAD": y + 1.0})
    ev = tm.evaluate(oof, [("GOOD", "BAD"), ("BAD", "GOOD")], TC, seed=0)[tm.PAIRED.name].filter(pl.col("metric") == "delta_rmse")
    g = {(r["model"], r["reference"]): r["estimate"] for r in ev.iter_rows(named=True)}
    assert abs(g[("GOOD", "BAD")] - 0.9) < 1e-9 and abs(g[("BAD", "GOOD")] + 0.9) < 1e-9, "Δ RMSE = RMSE(reference) − RMSE(model): positive = better"
    dup = pl.concat([oof, oof.head(1)])
    assert _raises(tm.CohortError, lambda: tm.evaluate(dup, [("GOOD", "BAD")], TC, seed=0)), "a duplicated athlete must be refused"
    part = oof.filter(~((pl.col("model") == "BAD") & (pl.col("nfl_id") == int(d.ids[0]))))
    assert _raises(tm.CohortError, lambda: tm.evaluate(part, [("GOOD", "BAD")], TC, seed=0)), "models on different athletes must be refused"
    return "sign convention of Δ RMSE, one prediction per athlete, identical cohorts enforced"


def check_matched_athlete_selection() -> str:
    d = design_fixture(40, 4)
    y = np.arange(d.n, dtype=float)
    oof = oof_fixture(d, y, {"M1": np.zeros(d.n), "RIGHT": y.copy(), "WRONG": -y}).with_columns(pl.lit(pt.STATUS).alias("status"))
    ids = d.ids
    pairs = pl.DataFrame({"pair_id": list(range(12)), "nfl_id_a": [int(ids[0])] * 6 + [int(ids[i]) for i in range(10, 16)],
                          "nfl_id_b": [int(ids[i]) for i in range(1, 7)] + [int(ids[i]) for i in range(20, 26)]})
    extra = pl.DataFrame({"pair_id": [99], "nfl_id_a": [int(ids[0])], "nfl_id_b": [10 ** 9]})          # partner without a prediction
    m = {r["model"]: r for r in pt.pair_concordance(pl.concat([pairs, extra]), oof, TC, seed=0).iter_rows(named=True)}
    assert m["RIGHT"]["n_pairs"] == 12 and m["RIGHT"]["n_athletes"] == 19 and m["RIGHT"]["max_pairs_per_athlete"] == 6
    assert m["RIGHT"]["concordance"] == 1.0 and m["WRONG"]["concordance"] == 0.0 and m["M1"]["concordance"] == 0.5, "ties count one half"
    assert abs(m["RIGHT"]["delta_vs_reference"] - 0.5) < 1e-12 and m["RIGHT"]["delta_ci_lower"] > 0
    return "pair concordance, tie handling, distinct athletes and pair reuse reported; pairs without both predictions dropped"


def check_hypothesis_registry() -> str:
    reg = pt.hypothesis_table()
    key = ["hypothesis", "target_id", "model", "reference", "scope"]
    assert reg.select(pl.struct(key).n_unique()).item() == reg.height, "a comparison is registered once"
    assert reg.filter((pl.col("tier") == "primary") & (pl.col("scope") == "pooled")).height == 3, "one primary per hypothesis H1–H3"
    known = {m.model for m in (*tm.model_ladder(TC), *pt.EXTRA_MODELS)}
    assert set(reg["model"]) | set(reg["reference"]) <= known and set(reg["status"]) == {pt.STATUS}
    h = pt.holm([0.01, 0.04, 0.03, None])
    assert h[3] is None and np.allclose(h[:3], [0.03, 0.06, 0.06]), f"Holm step-down, got {h}"
    assert not pt.hypothesis_table(("H1",)).filter(pl.col("hypothesis") == "H3")["active"].any()
    models = pt.models_for(TC, ["physical"])
    assert ("M2", "M1") in pt.pairs_for("Y_BAND46", ("H2",), models) and ("M5", "M1") not in pt.pairs_for("Y_CTX", ("H3",), models)
    return f"{reg.height} registered comparisons, 3 pooled primaries, Holm adjustment, inactive hypotheses excluded"


# ============================================================================== 07C
def _power_fixture(seed: int = 0) -> tuple[tp.Structure, tm.ArrayBlocks, tp.PowerConfig]:
    d = design_fixture(200, 4, seed)
    blocks = blocks_fixture(d, seed + 1)
    rng = np.random.default_rng(seed + 2)
    mu = np.where(d.position == "DB", 5.0, 8.0)
    y = mu + rng.normal(0, 0.5, d.n)
    oof = oof_fixture(d, y, {"M1": mu})
    pc_ = tp.PowerConfig(effects=(0.0, 0.6), replicates=5, n_boot=120, modes=("conditional",), candidates=("M5",), sample_sizes=(120,),
                         sample_effects=(0.6,), sample_replicates=2, min_position=20, progress_interval=2)
    return tp.build_structure(d, oof, blocks, TC, pc_, "T"), blocks, pc_


def check_planted_signal_recovery() -> str:
    st, blocks, pc_ = _power_fixture()
    X = tp._official_matrix(st.design, {m.model: m for m in tm.model_ladder(TC)}["M1"])
    z = st.signal[("M5", "conditional")]
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9 and np.abs(X.T @ z).max() / st.design.n < 1e-8, "conditional signal ⟂ official design"
    rng = np.random.default_rng(0)
    ys = np.stack([tp.synthetic_target(st, "M5", "conditional", 0.3, rng) for _ in range(200)])
    e = (ys - st.mu) / st.sigma
    assert abs(e.var() - 1.0) < 0.1, f"standardised synthetic residual variance {e.var():.3f}, planted 1"
    assert abs(np.mean([np.corrcoef(r, z)[0, 1] for r in e]) - np.sqrt(0.3)) < 0.05, "correlation with the signal = √f"
    ladder = {m.model: m for m in tm.model_ladder(TC)}
    models = [ladder["M1"], ladder["M5"]]
    hit = [tp.replicate_rows(tp.synthetic_target(st, "M5", "conditional", 0.6, np.random.default_rng(b)), st.design, blocks, models, "M5", TC, pc_,
                             np.random.default_rng(100 + b))[0] for b in range(5)]
    null = [tp.replicate_rows(tp.synthetic_target(st, "M5", "conditional", 0.0, np.random.default_rng(b)), st.design, blocks, models, "M5", TC, pc_,
                              np.random.default_rng(200 + b))[0] for b in range(6)]
    assert np.mean([r["delta_rmse"] for r in hit]) > 0.05 and sum(r["detected"] for r in hit) >= 4, "a large planted effect is detected"
    assert sum(r["detected"] for r in null) <= 2 and abs(np.mean([r["delta_rmse"] for r in null])) < 0.05, "no planted effect → no systematic gain"
    return f"planted f = 0.6 → mean Δ RMSE {np.mean([r['delta_rmse'] for r in hit]):+.3f}; null {np.mean([r['delta_rmse'] for r in null]):+.4f}"


def check_power_summaries() -> str:
    assert tp.wilson(0, 100)[0] < 1e-12 and 0.03 < tp.wilson(0, 100)[1] < 0.05 and tp.wilson(5, 0) == (None, None)
    assert abs(tp._crossing(np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.6, 1.0]), 0.8) - 0.15) < 1e-12 and tp._crossing(np.array([0.0, 0.1]), np.array([0.0, 0.5]), 0.8) is None
    f = tp.deal_folds(np.array(["A"] * 40 + ["B"] * 20), 4, np.random.default_rng(0))
    assert np.bincount(f[:40]).tolist() == [10] * 4 and np.bincount(f[40:]).tolist() == [5] * 4, "folds balanced within position"
    rows = [{"config_hash": "h", "study": "power", "candidate": "M5", "mode": "conditional", "effect": e, "n_athletes": 100, "replicate": b, "scope": "ALL",
             "n_scope": 100, "delta_rmse": 0.01 * (e > 0) + 0.001 * b, "ci_lower": (0.005 if e > 0 else -0.01), "ci_upper": 0.02, "relative": 0.05 * (e > 0),
             "delta_r2": 0.1 * (e > 0), "rmse_reference": 0.2, "detected": e > 0, "worse": False, "seconds": 0.1} for e in (0.0, 0.2) for b in range(4)]
    reps = pl.DataFrame(rows, schema=tp.REPLICATES.columns)
    plan = pl.DataFrame({"study": ["power"] * 2, "candidate": ["M5"] * 2, "mode": ["conditional"] * 2, "effect": [0.0, 0.2], "n_athletes": [100] * 2, "replicates": [4, 5]})
    sc_ = tp.scenario_table(reps, plan)
    assert sc_["power"].to_list() == [0.0, 1.0] and sc_["complete"].to_list() == [True, False], "a scenario short of its budget is not complete"
    assert tp.null_table(reps)["false_positive_rate"][0] == 0.0
    m = tp.mde_table(sc_, tp.PowerConfig())
    assert abs(m["mde_effect"][0] - 0.16) < 1e-9, "power 0 → 1 over f 0 → 0.2 reaches 0.8 at f = 0.16"
    return "Wilson interval, power-curve crossing, balanced fold deal, scenario completeness, MDE interpolation"


def check_checkpoint_resume_and_hash() -> str:
    st, blocks, pc_ = _power_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "power.parquet"                     # the progress lines printed below belong to this check
        _, s1 = tp.run_simulation(st, blocks, TC, pc_, ck, "resume", max_seconds=0.0)
        assert not s1["complete"] and s1["computed_now"] == 1 and ck.is_file(), "a time-limited run stops after one replicate and keeps its checkpoint"
        assert _raises(tp.PowerError, lambda: tp.assert_complete(s1)), "an incomplete simulation may not be reported"
        assert _raises(tp.PowerError, lambda: tp.run_simulation(st, blocks, TC, pc_, ck, "require_complete")), "require_complete refuses missing replicates"
        reps, s2 = tp.run_simulation(st, blocks, TC, pc_, ck, "resume")
        assert s2["complete"] and s2["computed_now"] == s2["planned_replicates"] - 1, "resume computes only what is missing"
        _, s3 = tp.run_simulation(st, blocks, TC, pc_, ck, "require_complete")
        assert s3["computed_now"] == 0 and s3["config_hash"] == s2["config_hash"]
        other = replace(pc_, n_boot=pc_.n_boot + 1)
        assert tp.config_hash(other, TC, st) != s2["config_hash"], "a changed configuration has another identity"
        assert tp.load_checkpoint(ck, tp.config_hash(other, TC, st), "resume").height == 0, "rows of another configuration are never reused"
    allrep = reps.filter(pl.col("scope") == "ALL")
    assert allrep.select(pl.struct("study", "candidate", "mode", "effect", "n_athletes", "replicate").n_unique()).item() == allrep.height
    return f"stop → resume → complete ({s2['planned_replicates']} replicates); configuration hash isolates checkpoints"


# ============================================================================== 07D
def check_percentile_orientation() -> str:
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pc.percentile_of(3.0, ref) == 50.0 and pc.percentile_of(0.0, ref) == 0.0 and pc.percentile_of(9.0, ref) == 100.0 and np.isnan(pc.percentile_of(np.nan, ref))
    forty = np.array([4.30, 4.40, 4.50, 4.60, 4.70, 4.80])
    pos = np.array(["WR"] * 6)
    p, n = pc.position_percentiles(forty, pos, np.ones(6, dtype=bool), higher_is_better=False, min_reference=3)
    assert p[0] == 100.0 and p[-1] == 0.0 and (np.diff(p) < 0).all(), "the shortest time has the highest percentile"
    s, _ = pc.position_percentiles(-forty, pos, np.ones(6, dtype=bool), higher_is_better=True, min_reference=3)
    assert np.allclose(p, s) and (n == 5).all(), "orientation mirrors the scale; a reference member is scored against the other five"
    assert not pc.FEATURE[pc.FORTY].higher_is_better and not pc.FEATURE[pc.V4_V8].higher_is_better and pc.FEATURE[pc.P95].higher_is_better
    return "mid-rank percentile, orientation of times against speeds, leave-one-out reference size"


def check_position_reference_isolation() -> str:
    rng = np.random.default_rng(0)
    v = rng.normal(size=80)
    pos = np.array(["DB", "WR"] * 40)
    fold = np.arange(80) // 2 % 4
    ref = np.ones(80, dtype=bool)
    base, _ = pc.position_percentiles(v, pos, ref, True, 10)
    v2 = v.copy()
    v2[pos == "WR"] += 100
    assert np.allclose(pc.position_percentiles(v2, pos, ref, True, 10)[0][pos == "DB"], base[pos == "DB"]), "another position never moves a percentile"
    v3 = v.copy()
    v3[0] = 1e6
    hi = pc.position_percentiles(v3, pos, ref, True, 10)[0]
    assert hi[0] == 100.0, "leave-one-out: an athlete's own value is not in the scale he is scored against"
    fb = pc.fold_percentiles(v, pos, ref, fold, True, 10)
    v4 = v.copy()
    v4[(fold == fold[0]) & (np.arange(80) != 0)] += 50
    assert pc.fold_percentiles(v4, pos, ref, fold, True, 10)[0] == fb[0], "athletes of his own fold do not set a held-out athlete's scale"
    few, _ = pc.position_percentiles(v[:8], pos[:8], ref[:8], True, 30)
    assert np.isnan(few).all(), "below the minimum reference size no percentile is given"
    return "percentiles are isolated by position, leave-one-out for reference members, fold-isolated for held-out scoring"


def check_measurement_uncertainty() -> str:
    rng = np.random.default_rng(1)
    A = 400
    true = rng.normal(10, 1.0, A)
    ath = np.repeat(np.arange(A), 2)
    val = true[ath] + rng.normal(0, 0.5, ath.size)
    c = un.attempt_components(val, ath)
    assert abs(c["s2_within"] - 0.25) < 0.05 and abs(c["s2_between"] - 1.0) < 0.2 and c["n_repeat"] == A, f"components {c}"
    r1, r2 = un.reliability_of_mean(c, np.array([1, 2]))
    assert 0 < r1 < r2 < 1, "two attempts are more reliable than one"
    m = un.shrunk_estimate(np.array([12.0]), np.array([1]), c)[0]
    assert c["mu"] < m < 12.0, "the reliability-aware estimate lies between the reference mean and the observed value"
    ref = rng.normal(10, 1.1, 150)
    lo0, hi0 = un.percentile_interval(10.5, 0.0, ref, True, np.random.default_rng(0), 600)
    lo1, hi1 = un.percentile_interval(10.5, 0.5, ref, True, np.random.default_rng(0), 600)
    p = pc.percentile_of(10.5, ref)
    assert lo0 <= p <= hi0 and lo1 <= p <= hi1 and (hi1 - lo1) > (hi0 - lo0) + 5, "measurement error widens the percentile interval"
    assert abs(r2 * p - p) > 1e-6, "sanity: percentile × reliability is a different number and is not what is reported"
    spec = pc.ReferenceSpec(min_reference=30, min_repeat=20)
    assert un.availability(c, 200, 2.0, spec, True)[0] == "available" and un.availability(c, 10, 2.0, spec, True)[0] == "unavailable"
    noisy = un.attempt_components(true[ath] * 0 + rng.normal(0, 1, ath.size), ath)
    assert un.availability(noisy, 200, 2.0, spec, True)[0] == "unavailable", "a measure that is all attempt noise abstains"
    assert un.availability({"n_repeat": 3, "s2_between": float("nan"), "s2_within": float("nan")}, 200, 1.0, spec, True)[0] == "experimental"
    return "variance components recovered; reliability rises with attempts; interval widens with measurement error; noise-only measure abstains"


def _rep_fixture(seed: int = 0) -> tuple[pl.DataFrame, dict[str, int], dict[int, str]]:
    """Two folds, 12 athletes (6 per position), 2 attempts each, 3 dimensions; athletes come in tight pairs."""
    rng = np.random.default_rng(seed)
    centre = np.repeat(rng.normal(0, 5, size=(6, 3)), 2, axis=0)          # athletes (0,1), (2,3), … share a centre
    rows, order, position = [], {}, {}
    for a in range(12):
        position[a + 1] = "DB" if a % 4 < 2 else "WR"                       # each pair shares a position
        for k in range(2):
            seq = f"s{a}_{k}"
            order[seq] = k + 1
            z = centre[a] + 0.2 * (a % 2) + rng.normal(0, 0.05, 3)
            for f in (0, 1):
                role = "heldout" if a % 2 == f else "train"
                rows += [{"scheme": "outer", "family": "TCN", "fold": f, "seq_id": seq, "nfl_id": a + 1, "role": role, "dim": j, "value": float(z[j] * (1 + f))}
                         for j in range(3)]
    return pl.DataFrame(rows), order, position


def check_fingerprint_neighbor_selection() -> str:
    att, order, position = _rep_fixture()
    emb = fg.fold_embeddings(att, order)
    qf = fg.query_fold(emb)
    assert all(ind for _, ind in qf.values()) and {f for f, _ in qf.values()} == {0, 1}, "every athlete is compared under the encoder that held him out"
    official = {a: (4.5 + 0.2 * ((a - 1) // 4 % 2), 1.5) for a in range(1, 13)}     # pairs 0–1: 4.5 s · 2–3: 4.7 s · 4–5: 4.5 s
    nb = fg.neighbor_table(emb, position, official, k=2, caliper=(0.05, 0.03))
    first = nb.filter((pl.col("kind") == "position") & (pl.col("rank") == 1))
    mate = {a: a + 1 if a % 2 else a - 1 for a in range(1, 13)}
    assert all(mate[q] == n for q, n in zip(first["nfl_id"], first["neighbor_id"])), "the nearest athlete is the one sharing the planted centre"
    assert all(position[q] == position[n] for q, n in zip(nb["nfl_id"], nb["neighbor_id"])), "neighbours are of the same position"
    sim = nb.filter(pl.col("kind") == "similar_official")
    assert sim.height > 0 and (sim["d_forty"].abs() <= 0.05).all(), "similar-official neighbours respect the caliper"
    stab = fg.stability_table(emb, position)
    assert (stab["self_auc"] >= 0.75).all() and (stab["n_attempts"] == 2).all(), "a repeat is closer than other athletes"
    assert fg.neighbor_overlap(nb, nb)[1] == 1.0
    d = fg.distance_lookup(emb)
    assert d(1, 2) < d(1, 5) and np.isnan(d(1, 999))
    cols = set(nb.columns) | set(stab.columns)
    assert not any(w in c for c in cols for w in ("score", "explos", "unique", "ready")), "similarity only: no rating column"
    return "inductive query encoder, same-position neighbours, official-time caliper, repeat stability; no rating is produced"


def check_no_nfl_leakage_and_abstention() -> str:
    ok = pl.DataFrame({"nfl_id": [1], "nfl_id_a": [1], "nfl_predictive_status": ["x"], "speed_capacity_pct": [50.0]})
    sc.assert_no_nfl_columns(ok, "fixture")
    for bad in ("nfl_speed", "accel_sust05_mean", "o2.value", "y_pred", "target"):
        assert _raises(sc.LeakageError, lambda b=bad: sc.assert_no_nfl_columns(ok.with_columns(pl.lit(1.0).alias(b)), "fixture")), f"{bad} must be refused"
    rng = np.random.default_rng(0)
    n = 80
    ids, pos, yr = np.arange(1, n + 1), np.array(["DB", "WR"] * (n // 2)), np.full(n, 2024)
    val = rng.normal(4.6, 0.1, n)
    val[:3] = np.nan
    t = sc.percentile_table(pc.FEATURE[pc.FORTY], ids, pos, yr, val, np.ones(n, dtype=np.int64), np.ones(n, dtype=bool), np.arange(n) % 4,
                            pc.ReferenceSpec(min_reference=20), seed=0, n_draws=200)
    miss = t.filter(pl.col("value").is_null())
    assert miss.height == 3 and miss["percentile"].null_count() == 3 and (miss["availability"] == "unavailable").all(), "a missing value gets NO percentile"
    shown = t.filter(pl.col("percentile").is_not_null())
    assert shown.height == n - 3 and shown["pct_lower"].null_count() == 0 and (shown["pct_lower"] <= shown["percentile"] + 1e-9).all()
    assert (shown["measurement_uncertainty"].str.contains("single official value")).all() and shown["standard_error"].null_count() == shown.height
    fast = t.filter(pl.col("value") == t["value"].min())
    assert fast["percentile"][0] > 95, "the fastest forty of the position is at the top"
    roster = pl.DataFrame({"nfl_id": [1, 2, 3], "position_group": ["DB"] * 3, "draft_year": [2024] * 3})
    wf = pl.DataFrame({"nfl_id": [1, 1, 2], "S2_signal": [True, True, True], "start_above_ref_speed": [False, False, True], "first_speed": [0.2, 0.3, 3.0],
                       "last_frame_censored": [False, False, False], "pos_resid_rms": [0.01, 0.02, 0.05]})
    conf = sc.confidence_table(roster, wf, {1: 2, 2: 1}, fg.STABILITY.empty(), {"DB": 100}, 30)
    assert conf.sort("nfl_id")["evidence_tier"].to_list() == ["high", "medium", "official_only"], f"tiers {conf['evidence_tier'].to_list()}"
    return "NFL-named columns are refused; a missing value abstains; evidence tiers follow the stated rule"


CHECKS_07A: tuple[Callable[[], str], ...] = (
    check_context_group_consistency, check_game_athlete_boundaries, check_physical_units, check_speed_band_conditions,
    check_athlete_game_resampling, check_context_standardization, check_hierarchical_model, check_fold_safe_outcome, check_outcome_reliability)
CHECKS_07B: tuple[Callable[[], str], ...] = (
    check_physical_units, check_fold_safe_outcome, check_fold_safe_transfer_fitting, check_oof_uniqueness_and_sign,
    check_matched_athlete_selection, check_hypothesis_registry)
CHECKS_07C: tuple[Callable[[], str], ...] = (check_planted_signal_recovery, check_power_summaries, check_checkpoint_resume_and_hash)
CHECKS_07D: tuple[Callable[[], str], ...] = (
    check_percentile_orientation, check_position_reference_isolation, check_measurement_uncertainty, check_fingerprint_neighbor_selection,
    check_no_nfl_leakage_and_abstention)
ALL_CHECKS: tuple[Callable[[], str], ...] = tuple(dict.fromkeys((*CHECKS_07A, *CHECKS_07B, *CHECKS_07C, *CHECKS_07D)))


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    """Run the synthetic checks in-process. A failing check is reported, not raised, so one table shows them all."""
    rows = []
    for fn in checks:
        try:
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "PASS", "detail": fn()})
        except Exception as e:                                   # noqa: BLE001 — the table must list every failure
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "FAIL", "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


__all__ = ["ALL_CHECKS", "CHECKS_07A", "CHECKS_07B", "CHECKS_07C", "CHECKS_07D", "assert_all_passed", "run_all"]
