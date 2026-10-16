"""Synthetic validation of the FPCA milestone — callable from a notebook cell, no subprocess, no warehouse.

    from nflbdb.research.advanced_movement import fpca_validation
    report = fpca_validation.run_all()          # table: check, status, detail
    fpca_validation.assert_all_passed(report)

Two kinds of synthetic data, both with a known answer:
    curves with two known orthonormal modes (a level and a cosine) and known athlete scores, to test the FPCA
    itself, its treatment of partly recorded curves and its isolation from held-out athletes;
    the analytic sprint sequences of ``validation`` (v = vmax·(1 − exp(−u/τ)) recorded from a late start), to test
    the representations, the preprocessing audit and the whole fold loop.
``tests/test_advanced_movement_fpca.py`` calls the same functions.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from ..contracts import conform
from . import checkpoint as ck
from . import data as dt
from . import eda
from . import fpca as fp
from . import fpca_eval as fe
from . import validation as base
from .sequences import SequenceSet

ACFG, HALF = base.ACFG, base.HALF
FCFG = fp.FpcaConfig(k_grid=(2, 3), n_boot=60, n_boot_r2=30, min_athletes=5)
SYN_SPEC = fp.RepresentationSpec("SYNTHETIC", "synthetic curves", "primary", None, "distance", "speed", "recording",
                                 False, 12, "x", "y", "u")
_raises = base._raises


# ============================================================================== synthetic data
def synthetic_curves(n_athletes: int = 60, seed: int = 0, noise: float = 0.02) -> tuple[fp.CurveSet, dict]:
    """Two attempts per athlete: y = mean + ξ1·φ1 + ξ2·φ2 + noise on [0, 10], with φ1 a level and φ2 a cosine (both
    of unit integrated square, orthogonal). Athlete scores have SD 3 and 1; attempts add SD 0.2 to each score."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 10.0, 51)
    phi = np.column_stack([np.full(grid.size, 1 / np.sqrt(10.0)), np.sqrt(0.2) * np.cos(np.pi * grid / 10.0)])
    mean = 5.0 + 0.3 * grid
    athlete = rng.normal(size=(n_athletes, 2)) * np.array([3.0, 1.0])
    nfl_ids = np.repeat(np.arange(5000, 5000 + n_athletes), 2)
    xi = np.repeat(athlete, 2, axis=0) + 0.2 * rng.normal(size=(2 * n_athletes, 2))
    Y = mean + xi @ phi.T + noise * rng.normal(size=(2 * n_athletes, grid.size))
    seq_ids = np.array([f"S|{a}|{j % 2 + 1}" for j, a in enumerate(nfl_ids)])
    return (fp.CurveSet(SYN_SPEC, grid, Y, np.tile(grid, (Y.shape[0], 1)), seq_ids, nfl_ids),
            {"phi": phi, "xi": xi, "mean": mean})


def synthetic_fpca_bundle(n_athletes: int = 40, seed: int = 0) -> dict:
    """The sprint fixture of ``validation`` with the flags the sprint rescue provides in production, plus the
    per-sequence window table."""
    b = base.synthetic_bundle(n_athletes, seed)
    moving = b["seq_frames"].filter(pl.col("frame") == 0).select("seq_id", (pl.col("speed") > 0.3).alias("_moving"))
    manifest = (b["manifest"].join(moving, on="seq_id", how="left")
                .with_columns(pl.col("_moving").fill_null(False).alias("onset_censored"), pl.lit(False).alias("signal_flag"),
                              pl.lit(False).alias("structural_flag"), pl.lit(False).alias("boundary_peak"),
                              pl.lit(True).alias("S2_signal"), pl.lit(True).alias("S3_no_boundary")).drop("_moving"))
    manifest = manifest.select([pl.col(c).cast(t) for c, t in dt.MANIFEST.columns.items()])
    ss = SequenceSet.from_frames(b["seq_frames"], manifest)
    summary = eda.sequence_summary(ss)
    return {**b, "raw_manifest": b["manifest"], "manifest": manifest, "ss": ss, "summary": summary,
            "windows": ck.sequence_windows(ss, summary, manifest, ACFG, FCFG)}


def _congruence(a: np.ndarray, b: np.ndarray, q: np.ndarray) -> float:
    return float(abs((q * a * b).sum()) / np.sqrt((q * a * a).sum() * (q * b * b).sum()))


# ============================================================================== checks: basis and model
def check_basis_and_small_helpers() -> str:
    knots = fp.clamped_knots(0.0, 10.0, 8, 3)
    x = np.linspace(0.0, 10.0, 101)
    Bm = fp.bspline_design(x, knots, 3)
    assert knots.size == 8 + 3 + 1 and Bm.shape == (101, 8)
    assert np.allclose(Bm.sum(1), 1.0) and (Bm >= -1e-12).all()                  # partition of unity, right end included
    assert abs(fp.quadrature(x).sum() - 10.0) < 1e-12
    assert _raises(fp.FpcaError, lambda: fp.clamped_knots(0.0, 1.0, 3, 3))
    assert fp.longest_run(np.array([True, False, True, True, True, False, True])) == (2, 5)
    assert fp.longest_run(np.array([False, False])) == (0, 0)
    w = fp.athlete_weights(np.array([1, 1, 2, 3, 3, 3]))
    assert abs(w.sum() - 1) < 1e-12 and np.allclose([w[:2].sum(), w[2], w[3:].sum()], 1 / 3)
    t, v = np.array([0.0, 0.1, 0.2, 0.3]), np.array([1.0, 1.5, 2.5, 3.0])
    assert abs(fp.reference_crossing(t, v, 2.0) - 0.15) < 1e-12
    assert np.isnan(fp.reference_crossing(t, v + 2.0, 2.0)) and np.isnan(fp.reference_crossing(t, v, 9.0))
    ids, mean, n = fe.athlete_means(np.array([7, 7, 8]), np.array([[1.0, np.nan], [3.0, 4.0], [5.0, np.nan]]))
    assert ids.tolist() == [7, 8] and n.tolist() == [2, 1] and np.allclose(mean[0], [2.0, 4.0]) and np.isnan(mean[1, 1])
    two, i1, i2 = fe.first_two(np.array([9, 4, 9, 4, 5]), np.array([2, 1, 1, 2, 1]))
    assert two.tolist() == [4, 9] and i1.tolist() == [1, 2] and i2.tolist() == [3, 0]
    assert _raises(fp.FpcaError, lambda: fp.FpcaConfig(score_method="pca"))
    return "B-splines sum to one, quadrature integrates the grid, athlete weights and attempt pairing are correct"


def check_fpca_recovers_known_modes() -> str:
    cs, truth = synthetic_curves()
    m = fp.fit_fpca(cs, np.unique(cs.nfl_ids), FCFG)
    assert m.support.all() and m.k_max == 3 and abs((m.q * m.phi[:, 0] ** 2).sum() - 1) < 1e-8
    c1, c2 = _congruence(m.phi[:, 0], truth["phi"][:, 0], m.q), _congruence(m.phi[:, 1], truth["phi"][:, 1], m.q)
    assert c1 > 0.98 and c2 > 0.95, (c1, c2)
    assert 0.5 * 9.04 < m.eigenvalues[0] < 1.7 * 9.04 and 0.5 * 1.04 < m.eigenvalues[1] < 1.7 * 1.04, m.eigenvalues
    assert m.explained()[:2].sum() > 0.97 and np.abs(m.mean - truth["mean"]).max() < 1.5
    Sc, frac = m.scores(cs.Y, 2)
    assert np.allclose(frac, 1.0) and np.isfinite(Sc).all()
    r1 = abs(np.corrcoef(Sc[:, 0], truth["xi"][:, 0])[0, 1])
    r2 = abs(np.corrcoef(Sc[:, 1], truth["xi"][:, 1])[0, 1])
    assert r1 > 0.98 and r2 > 0.85, (r1, r2)
    res = m.residuals(cs.Y, Sc, 2)
    assert 1 - res["sse"].sum() / res["sst"].sum() > 0.97
    ls = replace(m, score_method="ls").scores(cs.Y, 2)[0]
    assert np.abs(ls - Sc).max() < 0.05 * np.abs(Sc).max()                        # fully observed: shrinkage is negligible
    assert _raises(fp.FpcaError, lambda: m.scores(cs.Y[:, :10], 2)) and _raises(fp.FpcaError, lambda: m.scores(cs.Y, 9))
    return f"both known modes are recovered (congruence {c1:.3f}, {c2:.3f}); scores track the true scores"


def check_partial_coverage_is_never_filled() -> str:
    cs, truth = synthetic_curves()
    Y = cs.Y.copy()
    cut = np.arange(0, Y.shape[0], 9)                           # about 11 % of the curves lose their last 40 %
    Y[cut, 31:] = np.nan
    Y[[1, 2], 10:] = np.nan                                     # two curves keep only the first 20 %
    part = fp.CurveSet(cs.spec, cs.grid, Y, cs.t_orig, cs.seq_ids, cs.nfl_ids)
    m = fp.fit_fpca(part, np.unique(cs.nfl_ids), FCFG)
    assert m.support.all()                                      # the tail is still observed for > 80 % of the weight
    Sc, frac = m.scores(Y, 2)
    assert np.isnan(Sc[[1, 2]]).all() and frac[1] < FCFG.min_observed_fraction        # too little recorded: no score
    assert np.isfinite(Sc[cut[cut > 2]]).all()
    assert abs(np.corrcoef(Sc[cut[cut > 2], 0], truth["xi"][cut[cut > 2], 0])[0, 1]) > 0.95
    res = m.residuals(Y, Sc, 2)
    assert (res["n_points"][cut[cut > 2]] == 31).all() and res["n_points"][1] == 0 and np.isnan(res["sse"][1])
    assert m.reconstruct(np.nan_to_num(Sc), 2).shape == (Y.shape[0], int(m.support.sum()))
    Y2 = cs.Y.copy()
    Y2[np.arange(0, Y2.shape[0], 2), 41:] = np.nan              # half of the curves stop early → tail below 80 %
    m2 = fp.fit_fpca(fp.CurveSet(cs.spec, cs.grid, Y2, cs.t_orig, cs.seq_ids, cs.nfl_ids), np.unique(cs.nfl_ids), FCFG)
    assert int(m2.support.sum()) == 41 and not m2.support[41:].any() and m2.mean.size == 41
    empty = fp.CurveSet(cs.spec, cs.grid, np.full_like(cs.Y, np.nan), cs.t_orig, cs.seq_ids, cs.nfl_ids)
    assert _raises(fp.FpcaError, lambda: fp.fit_fpca(empty, np.unique(cs.nfl_ids), FCFG))
    return "unrecorded points never enter a fit, a score or an error; a poorly covered tail leaves the support"


def check_fit_uses_training_athletes_only() -> str:
    cs, _ = synthetic_curves()
    ids = np.unique(cs.nfl_ids)
    train, held = ids[:45].tolist(), ids[45:].tolist()
    m = fp.fit_fpca(cs, train, FCFG)
    moved = cs.Y.copy()
    moved[np.isin(cs.nfl_ids, held)] = moved[np.isin(cs.nfl_ids, held)] * 10.0 + 3.0      # change held-out athletes only
    m2 = fp.fit_fpca(fp.CurveSet(cs.spec, cs.grid, moved, cs.t_orig, cs.seq_ids, cs.nfl_ids), train, FCFG)
    assert np.allclose(m.mean, m2.mean) and np.allclose(m.phi, m2.phi) and np.allclose(m.eigenvalues, m2.eigenvalues)
    assert np.array_equal(m.support, m2.support) and m.fit_athletes == tuple(train) and m.n_fit_sequences == 90
    m.assert_unseen(held)
    assert _raises(dt.LeakageError, lambda: m.assert_unseen(train[:1]))
    rows = np.isin(cs.nfl_ids, train[:1])
    dup = fp.CurveSet(cs.spec, cs.grid, np.vstack([cs.Y, cs.Y[rows], cs.Y[rows], cs.Y[rows]]),
                      np.vstack([cs.t_orig] + [cs.t_orig[rows]] * 3), np.r_[cs.seq_ids, ["d1", "d2", "d3", "d4", "d5", "d6"]],
                      np.r_[cs.nfl_ids, np.repeat(cs.nfl_ids[rows], 3)])
    m3 = fp.fit_fpca(dup, train, FCFG)
    assert np.allclose(m.mean, m3.mean) and np.allclose(m.eigenvalues, m3.eigenvalues)   # an athlete weighs 1, not his attempts
    return "held-out athletes cannot move a fit; repeating an athlete's attempts does not change his weight"


def check_sign_order_and_persistence() -> str:
    cs, _ = synthetic_curves()
    m = fp.fit_fpca(cs, np.unique(cs.nfl_ids), FCFG)
    perm, sign = np.array([2, 0, 1]), np.array([-1.0, 1.0, -1.0])
    shuffled = replace(m, phi=m.phi[:, perm] * sign, eigenvalues=m.eigenvalues[perm], source_rank=m.source_rank[perm])
    back, cong = fp.align_to_reference(shuffled, m)
    assert np.allclose(back.phi, m.phi) and np.allclose(back.eigenvalues, m.eigenvalues) and (cong > 0.999).all()
    assert back.source_rank.tolist() == [1, 2, 3]
    with tempfile.TemporaryDirectory() as tmp:
        path = fp.save_models(Path(tmp) / "fpca" / "synthetic.npz", {"reference": m, "r0_f0": shuffled})
        loaded = fp.load_models(path)
    assert set(loaded) == {"reference", "r0_f0"} and loaded["reference"].fit_athletes == m.fit_athletes
    assert np.allclose(loaded["reference"].scores(cs.Y, 2)[0], m.scores(cs.Y, 2)[0])
    assert loaded["r0_f0"].source_rank.tolist() == [3, 1, 2] and loaded["reference"].score_method == "blup"
    return "fold fits take the sign and order of the reference without changing a value; models reload identically"


# ============================================================================== checks: representations and audit
def check_representations_from_sequences() -> str:
    b = synthetic_fpca_bundle()
    ss = b["ss"]
    curves = {s.name: fp.build_curves(ss, ACFG, s) for s in fp.REPRESENTATIONS}
    A, Bm, Bk = curves["A_speed_by_distance"], curves["B_accel_by_speed"], curves["B_alt_boundary_kept"]
    path = np.array([np.nanmax(ss.series(r, "dist")) for r in range(ss.n)])
    assert np.isnan(A.Y[A.grid[None, :] > path[:, None] + 1e-3]).all()            # nothing beyond the recorded distance
    assert np.isfinite(A.Y[A.grid[None, :] < path[:, None] - 1e-3]).all()
    for r in range(ss.n):
        vmax, tau = base.athlete_params(int(ss.nfl_ids[r]) - 1000)
        ok = np.isfinite(Bm.Y[r])
        assert ok.any() and np.abs(Bm.Y[r, ok] - (vmax - Bm.grid[ok]) / tau).max() < 0.08       # a(v) = (vmax − v)/τ
        assert not ok[Bm.grid < ss.series(r, "speed")[HALF] - 1e-3].any()         # boundary frames carry no acceleration
    seen_m, seen_k = np.isfinite(Bm.Y), np.isfinite(Bk.Y)
    assert (seen_k | ~seen_m).all() and seen_k.sum() > seen_m.sum()               # the mask only removes values
    first = np.array([ss.series(r, "speed")[0] for r in range(ss.n)])
    above = first >= ACFG.phase_ref_speed
    assert above.any() and (~above).any()
    for name in ("A_alt_origin_at_ref_speed", "C_alt_origin_at_ref_speed"):
        alt = curves[name]
        assert np.isnan(alt.Y[above]).all()                                       # the reference crossing was not recorded
        assert np.abs(alt.Y[~above, 0] - ACFG.phase_ref_speed).max() < 1e-3       # origin = the moment that speed is reached
    C = curves["C_speed_by_time"]
    k = int(ss.lengths[0])
    assert np.allclose(C.Y[0, :k], ss.series(0, "speed"), atol=1e-4) and np.isnan(C.Y[0, k:]).all()
    assert _raises(fp.FpcaError, lambda: fp.build_curves(ss, ACFG, replace(fp.SPEC["B_accel_by_speed"], origin="reference_speed")))
    return "six representations are built; values stop where the recording stops and at masked boundary frames"


def check_preprocessing_audit() -> str:
    b = synthetic_fpca_bundle()
    ss, w = b["ss"], conform(b["windows"], ck.WINDOWS)
    assert w.height == ss.n and (w["n_boundary_frames"] == 2 * HALF).all() and (w["t_last_s"] > w["t_first_s"]).all()
    assert w["first_frame_censored"].null_count() == 0 and w["last_frame_censored"].all()   # the fixture ends at top speed
    curves = {s.name: fp.build_curves(ss, ACFG, s) for s in fp.REPRESENTATIONS}
    audit = conform(ck.audit_preprocessing(ss, b["seq_frames"], b["manifest"], b["splits"], w, curves, ACFG, HALF), ck.AUDIT)
    assert audit["passed"].all(), audit.filter(~pl.col("passed")).to_dicts()
    ck.assert_blocking_passed(audit)
    C = curves["C_speed_by_time"]
    r, g = np.argwhere(~np.isfinite(C.Y))[0]
    planted = fp.CurveSet(C.spec, C.grid, C.Y.copy(), C.t_orig, C.seq_ids, C.nfl_ids)
    planted.Y[r, g] = 9.0                                                         # a value where nothing was recorded
    bad = ck.audit_preprocessing(ss, b["seq_frames"], b["manifest"], b["splits"], w, {"C_speed_by_time": planted}, ACFG, HALF)
    assert not bad.filter(pl.col("check") == "no_value_outside_the_recording:C_speed_by_time")["passed"][0]
    assert _raises(ck.PreprocessingError, lambda: ck.assert_blocking_passed(bad))
    fallback = ck.audit_preprocessing(ss, b["seq_frames"], b["raw_manifest"], b["splits"], w, {}, ACFG, HALF)
    assert not fallback.filter(pl.col("check") == "sprint_rescue_flags_present")["passed"][0]   # production guard
    dec = conform(ck.final_decisions({"representations": "x"}), ck.DECISIONS)
    assert dec.height >= 10 and dec.filter(pl.col("decision") == "representations")["evidence"][0] == "x"
    assert ck.extreme_recordings(w, b["splits"], 3).height == 6 and ck.window_summary(w).height >= 2
    return "the audit passes on clean data and reports a planted extrapolation and a manifest built on the fallback"


# ============================================================================== checks: fold loop and gate
def check_fold_loop_and_evaluation() -> str:
    b = synthetic_fpca_bundle()
    ss, splits = b["ss"], b["splits"]
    names = ("A_speed_by_distance", "B_accel_by_speed", "C_speed_by_time", "A_alt_origin_at_ref_speed")
    curves = {n: fp.build_curves(ss, ACFG, fp.SPEC[n]) for n in names}
    runs = fe.fit_all(curves, splits, FCFG)
    fit = fe.fit_tables(runs)
    for c in fe.FIT_CONTRACTS:
        conform(fit[c.name], c)
    n_fits = ACFG.split_repeats * ACFG.outer_folds
    for name, run in runs.items():
        assert len(run.models) == n_fits + 1
        for r in range(ACFG.split_repeats):
            for f in range(ACFG.outer_folds):
                train, test = dt.fold_athletes(splits, r, f)
                m = run.models[fe.fold_tag(r, f)]
                assert not set(m.fit_athletes) & set(test) and set(m.fit_athletes) <= set(train)
    sc = fit[fe.SCORES.name]
    per = sc.filter(pl.col("component") == 1).group_by("representation", "k", "repeat", "seq_id").len()
    assert per["len"].max() == 1                                                  # every attempt is held out exactly once
    held = sc.join(splits.select("nfl_id", "repeat", pl.col("outer_fold").alias("_fold")), on=["nfl_id", "repeat"])
    assert (held["outer_fold"] == held["_fold"]).all()                            # scored by the fold that excluded the athlete
    a = sc.filter((pl.col("representation") == names[0]) & (pl.col("k") == 2) & (pl.col("repeat") == 0))
    assert a["seq_id"].n_unique() == ss.n and a["score"].is_finite().all()
    rec = fit[fe.SEQ_RECON.name]
    assert rec.filter(pl.col("scored"))["sse"].null_count() == 0 and (rec.filter(pl.col("scored"))["n_points"] > 0).all()
    ev = fe.evaluate(fit, b["windows"], b["athletes"], FCFG, seed=3)
    for c in fe.EVAL_CONTRACTS:
        conform(ev[c.name], c)
    rel = ev[fe.RELIABILITY.name].filter((pl.col("representation") == names[0]) & (pl.col("scope") == "pooled")
                                         & (pl.col("repeat") == 0) & (pl.col("k") == 2))
    assert rel.height == 2 and rel["icc"].null_count() == 0 and rel["ci_lower"].null_count() == 0
    assert (rel["n_athletes_repeat"] == int((b["athletes"]["n_primary_eligible"] >= 2).sum())).all()
    assert ev[fe.RELIABILITY.name].filter(pl.col("repeat") == 1)["ci_lower"].null_count() == \
        ev[fe.RELIABILITY.name].filter(pl.col("repeat") == 1).height                # intervals on split repeat 0 only
    recon = ev[fe.RECON.name]
    two, three = (recon.filter((pl.col("representation") == names[0]) & (pl.col("repeat") == 0) & (pl.col("k") == k))
                  ["fve_heldout"][0] for k in (2, 3))
    assert three >= two - 0.02 and two > 0.5                                      # a third component does not hurt held-out fit
    assert ev[fe.SENSITIVITY.name]["representation"].unique().to_list() == [names[0]]
    assert ev[fe.STABILITY.name].height > 0 and ev[fe.WINDOW.name]["repeat"].unique().to_list() == [0]
    gate = fe.gate_tables(ev, fit, b["windows"], b["athletes"], FCFG, seed=3)
    g, s = conform(gate[fe.GATE.name], fe.GATE), conform(gate[fe.SUMMARY.name], fe.SUMMARY)
    assert s.height == len(names) and s["proposed_benchmark"].sum() <= 1
    assert not s.filter(pl.col("role") == "alternate")["proposed_benchmark"].any()
    assert g.group_by("representation").len()["len"].min() >= 2
    text = fe.conclusions_markdown(s, g, FCFG, "smoke")
    assert "Research gate" in text and "must not be interpreted" in text
    with tempfile.TemporaryDirectory() as tmp:
        paths = fe.save_runs(Path(tmp), runs)
        assert len(paths) == len(names) and all(p.is_file() for p in paths)
        loaded = fe.load_or_refit(Path(tmp), curves, splits, FCFG)
        assert set(loaded[names[0]]) == set(runs[names[0]].models)
        assert set(fe.load_or_refit(Path(tmp) / "absent", curves, splits, FCFG)[names[1]]) == set(runs[names[1]].models)
    return (f"{len(names)} representations × {n_fits} inductive fits: every attempt is scored once by a fit that excluded "
            "its athlete; all evaluation tables satisfy their contracts")


def check_gate_rules() -> str:
    c = FCFG
    assert fe.gate_labels(0.6, 0.8, 0.1, 0.3, 0.0, 0.1, c) == ("yes", "not_redundant", "not_window_driven", "candidate")
    assert fe.gate_labels(0.1, 0.4, 0.1, 0.3, 0.0, 0.1, c)[3] == "not_repeatable"
    assert fe.gate_labels(0.6, 0.8, 0.8, 0.9, 0.0, 0.1, c)[3] == "redundant_with_stopwatch"
    assert fe.gate_labels(0.6, 0.8, 0.1, 0.3, 0.3, 0.6, c)[3] == "window_artifact"
    assert fe.gate_labels(0.4, 0.8, 0.1, 0.3, 0.0, 0.1, c)[3] == "inconclusive"          # interval straddles the ICC line
    assert fe.gate_labels(None, None, None, None, None, None, c) == ("uncertain", "uncertain", "uncertain", "inconclusive")
    assert fe.gate_labels(0.6, 0.8, float("nan"), float("nan"), 0.0, 0.1, c)[1] == "uncertain"
    recon = pl.DataFrame({"representation": ["X"] * 4 + ["Y"] * 4, "k": [2, 3, 2, 3] * 2,
                          "fve_heldout": [0.90, 0.96, 0.92, 0.96, 0.5, 0.6, 0.5, 0.6]})
    chosen = fe.select_k(recon, c)
    assert chosen["X"][0] == 3 and chosen["Y"][0] == 3 and "largest" in chosen["Y"][1] and "smallest" in chosen["X"][1]
    assert fe._abs_interval(-0.2, 0.3) == (0.0, 0.3) and fe._abs_interval(-0.6, -0.4) == (0.4, 0.6)
    return "labels follow the intervals, a straddling interval stays inconclusive, K is chosen without any outcome"


FPCA_CHECKS: tuple[Callable[[], str], ...] = (
    check_basis_and_small_helpers, check_fpca_recovers_known_modes, check_partial_coverage_is_never_filled,
    check_fit_uses_training_athletes_only, check_sign_order_and_persistence, check_representations_from_sequences,
    check_preprocessing_audit, check_fold_loop_and_evaluation, check_gate_rules)


def run_all(checks: tuple[Callable[[], str], ...] = FPCA_CHECKS) -> pl.DataFrame:
    """Run every FPCA check in-process. A failing check is reported, not raised, so one table shows them all."""
    return base.run_all(checks)


assert_all_passed = base.assert_all_passed
