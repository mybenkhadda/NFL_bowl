"""12C — distributional NFL movement capacity. NumPy / Polars.

Data: ``m11_envelope_hist`` (athlete × game × 0.5 yd/s speed bin × 0.25 yd/s² acceleration bin frame counts of eligible
post-snap NFL frames). An athlete's games are the independent replication unit: every split, bootstrap and held-out
evaluation resamples GAMES, never frames (frames of one play are dependent).

Estimators of the speed-conditioned acceleration quantile Q_τ(a | v, athlete) per speed bin
    C0a  observed maximum           upper edge of the highest occupied acceleration bin (≥ 2 frames)        — an extreme
    C0b  linear envelope            Milestone 11 rule: q95 per bin → weighted line a = A0 (1 − v/V0) (refitted per split)
    C1   empirical bin quantile     histogram quantile per supported bin (≥ ``min_frames_bin_nfl`` frames)
    C2   smooth quantile P-spline   penalised quantile regression in speed on the histogram cells (λ tuned once on a
                                    separate game split, never on a Combine value)
    C3   hierarchical (pooled)      C1 shrunk toward the position curve per bin, weight = between-athlete variance /
                                    (between + game-bootstrap sampling variance) — empirical Bayes partial pooling
    C4   shape-constrained          C1 with a non-increasing restriction ABOVE ``shape_vmin`` for τ ≥ 0.9 (a physically
                                    motivated force–velocity decline); no restriction at low speed
    C5   distributional forest      declared optional; NOT implemented in this milestone (reported as not_run)

Context: the histograms carry game and season, not down / alignment / role, so "context" here = position (pooling),
season (split), exposure (frames, games) and speed (conditioning). This limitation is written into every conclusion.
An unobserved theoretical upper envelope is never presented as a measured physiological maximum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ....data.utils import get_logger, utc_now
from ..m11 import physics as ph
from . import advanced_evaluation as ae
from . import smoothing as sm
from .config import M12Config, config_hash

log = get_logger(__name__)
NAN = float("nan")
ESTIMATORS = ("C0a_observed_max", "C0b_linear_envelope", "C1_empirical", "C2_smooth_qr", "C3_pooled", "C4_shape")
QUANTILE_ESTIMATORS = ("C1_empirical", "C2_smooth_qr", "C3_pooled", "C4_shape")
NOT_IMPLEMENTED = {"C5_distributional_forest": "declared optional; not implemented in Milestone 12 (no forest code in the project)"}


# ============================================================================== athlete game cubes
@dataclass
class Cube:
    """One athlete: per-game histograms H[g] (speed bins × acceleration bins, int32) with game and season labels."""
    nfl_id: int
    position: str
    games: np.ndarray
    seasons: np.ndarray
    H: np.ndarray

    def total(self, sel: np.ndarray | None = None) -> np.ndarray:
        return (self.H if sel is None else self.H[sel]).sum(0).astype(float)


def build_cubes(hist: pl.DataFrame, positions: dict[int, str], c: M12Config) -> dict[int, Cube]:
    ns, na = ae.speed_edges(c).size - 1, ae.accel_edges(c).size - 1
    out: dict[int, Cube] = {}
    for (i,), g in hist.partition_by(["nfl_id"], as_dict=True).items():
        games = np.sort(g["game_id"].unique().to_numpy())
        gi = {int(x): k for k, x in enumerate(games)}
        H = np.zeros((games.size, ns, na), np.int32)
        np.add.at(H, (np.array([gi[int(x)] for x in g["game_id"].to_list()]), g["speed_bin"].to_numpy(), g["accel_bin"].to_numpy()), g["n"].to_numpy())
        seas = g.group_by("game_id").agg(pl.col("season").first()).sort("game_id")["season"].to_numpy()
        out[int(i)] = Cube(int(i), positions.get(int(i), "?"), games, seas, H)
    return out


def game_splits(cube: Cube, kind: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """Two disjoint game sets of one athlete. ``odd_even`` (by game order), ``random`` (seeded halves), ``season`` (first
    season vs later seasons; None when one season only)."""
    G = cube.games.size
    if G < 2:
        return None
    if kind == "odd_even":
        a = np.arange(G) % 2 == 0
    elif kind == "season":
        s0 = cube.seasons.min()
        a = cube.seasons == s0
        if a.all() or not a.any():
            return None
    else:
        rng = np.random.default_rng(seed * 1_000_003 + cube.nfl_id % 1_000_003)
        a = np.zeros(G, bool)
        a[rng.permutation(G)[: G // 2]] = True
    return a, ~a


# ============================================================================== estimators on one histogram
def _edges(c: M12Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    se, ab = ae.speed_edges(c), ae.accel_edges(c)
    return se, ab, ae.mids(se), ae.mids(ab)


def q_rows(H: np.ndarray, tau: float, ab: np.ndarray, min_frames: float) -> np.ndarray:
    """Vectorised histogram quantile of every row of ``H`` (rows × acceleration bins), linear inside the bin — the same
    rule as ``m11.physics.hist_quantile``. Rows with fewer than ``min_frames`` frames → NaN."""
    H = np.asarray(H, float)
    n = H.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cdf = np.cumsum(H, -1) / n[..., None]
    j = np.minimum((cdf < tau - 1e-12).sum(-1), H.shape[-1] - 1)
    cj = np.take_along_axis(cdf, j[..., None], -1)[..., 0]
    prev = np.where(j > 0, np.take_along_axis(cdf, np.maximum(j - 1, 0)[..., None], -1)[..., 0], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(cj > prev, (tau - prev) / (cj - prev), 0.5)
    out = ab[j] + frac * (ab[j + 1] - ab[j])
    return np.where(n >= min_frames, out, np.nan)


def q_empirical(H: np.ndarray, tau: float, c: M12Config) -> np.ndarray:
    _, ab, _, _ = _edges(c)
    return q_rows(H, tau, ab, c.min_frames_bin_nfl)


def observed_max(H: np.ndarray, c: M12Config, min_count: int = 2) -> np.ndarray:
    _, ab, _, _ = _edges(c)
    n = H.sum(1)
    out = np.full(H.shape[0], NAN)
    for k in range(H.shape[0]):
        if n[k] >= c.min_frames_bin_nfl:
            occ = np.flatnonzero(H[k] >= min_count)
            out[k] = float(ab[occ[-1] + 1]) if occ.size else NAN
    return out


def linear_envelope_curve(H: np.ndarray, c: M12Config, q: float = 0.95) -> tuple[np.ndarray, dict[str, float]]:
    """Milestone 11 envelope rule refitted on this histogram; evaluated at supported bin mids only."""
    se, ab, sm_, _ = _edges(c)
    v, aq, n = ph.envelope_points(H, se, ab, q, c.min_frames_bin_nfl, 1.0)
    out = np.full(H.shape[0], NAN)
    if v.size < 5:
        return out, {"A0": NAN, "V0": NAN}
    f = ph.fit_linear_envelope(v, aq, n)
    sup = H.sum(1) >= c.min_frames_bin_nfl
    out[sup] = f["A0"] + f["slope"] * sm_[sup]
    return out, {"A0": f["A0"], "V0": f["V0"]}


def q_smooth(H: np.ndarray, tau: float, lam: float, c: M12Config) -> np.ndarray:
    """Penalised quantile P-spline over the athlete's supported speed range (fit on histogram cells, weights = counts)."""
    se, _, smid, amid = _edges(c)
    n = H.sum(1)
    sup = np.flatnonzero(n >= c.min_frames_bin_nfl)
    out = np.full(H.shape[0], NAN)
    if sup.size < 4:
        return out
    lo, hi = float(se[sup[0]]), float(se[sup[-1] + 1])
    ks, js = np.nonzero(H[sup] > 0)
    x, y, w = smid[sup][ks], amid[js], H[sup][ks, js].astype(float)
    nk = max(2, min(c.spline_knots, sup.size - 2))
    B = sm.bspline_basis(x, lo, hi, nk)
    beta = sm.quantile_pspline(B, y, w, tau, lam, c.qr_iters)
    out[sup] = sm.bspline_basis(smid[sup], lo, hi, nk) @ beta
    # bins inside the range but unsupported stay NaN (no interpolation across gaps of the athlete's own support)
    return out


def q_shape(q: np.ndarray, n: np.ndarray, tau: float, c: M12Config) -> np.ndarray:
    if tau < 0.9:
        return q.copy()
    _, _, smid, _ = _edges(c)
    out = q.copy()
    m = np.isfinite(q) & (smid >= c.shape_vmin)
    if m.sum() >= 2:
        out[m] = ph.isotonic_decreasing(q[m], n[m])
    return out


def bootstrap_hists(cube: Cube, sel: np.ndarray, B: int, seed: int) -> np.ndarray | None:
    """B game-bootstrap replicate histograms (B, speed bins, acceleration bins): games resampled with replacement through
    multinomial game weights (one matrix product, no copy of the game stack). None below 3 games or 10 replicates."""
    idx = np.flatnonzero(sel)
    if idx.size < 3 or B < 10:
        return None
    rng = np.random.default_rng(seed + cube.nfl_id)
    W = np.stack([np.bincount(rng.integers(0, idx.size, idx.size), minlength=idx.size) for _ in range(B)]).astype(float)
    G = cube.H[idx].reshape(idx.size, -1).astype(float)
    return (W @ G).reshape(B, *cube.H.shape[1:])


def bootstrap_se(reps: np.ndarray | None, tau: float, c: M12Config) -> np.ndarray:
    """Game-bootstrap SE of the C1 quantile per speed bin."""
    if reps is None:
        return np.full(len(ae.speed_edges(c)) - 1, NAN)
    _, ab, _, _ = _edges(c)
    Q = q_rows(reps, tau, ab, c.min_frames_bin_nfl)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(Q).sum(0) >= 10
        return np.where(ok, np.nanstd(Q, axis=0), np.nan)


def pooled(Q: dict[int, np.ndarray], SE: dict[int, np.ndarray], positions: dict[int, str], ref_ids: Sequence[int] | None = None) -> dict[int, np.ndarray]:
    """C3 partial pooling per position × speed bin. Position mean and between-athlete variance from ``ref_ids`` (default:
    every athlete in ``Q``); shrinkage weight w = τ²/(τ² + se²), τ² = var(q) − mean(se²) floored at 0."""
    ref = list(Q) if ref_ids is None else [i for i in ref_ids if i in Q]
    out = {}
    for p in sorted(set(positions.get(i, "?") for i in Q)):
        r = [i for i in ref if positions.get(i, "?") == p]
        if len(r) < 3:
            for i in Q:
                if positions.get(i, "?") == p:
                    out[i] = Q[i].copy()
            continue
        M = np.array([Q[i] for i in r])
        S2 = np.array([SE[i] ** 2 for i in r])
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(M, 0)
            tau2 = np.maximum(np.nanvar(M, 0) - np.nanmean(S2, 0), 0.0)
        for i in Q:
            if positions.get(i, "?") != p:
                continue
            se2 = SE[i] ** 2
            w = np.where(np.isfinite(se2), tau2 / np.maximum(tau2 + se2, 1e-12), 1.0)
            out[i] = np.where(np.isfinite(Q[i]), w * Q[i] + (1 - w) * mu, np.nan)
    return out


# ============================================================================== curves for a set of athletes and games
def curves(cubes: dict[int, Cube], sel: dict[int, np.ndarray] | None, taus: Sequence[float], c: M12Config, lam: float, boot: int, seed: int,
           pool_ref: Sequence[int] | None = None) -> dict[float, dict[str, dict[int, np.ndarray]]]:
    """{τ: {estimator: {athlete: curve over speed bins}}} from the games selected per athlete (all games when ``sel`` is
    None). The game-bootstrap replicates are drawn ONCE per athlete and reused for every τ. C0a / C0b do not depend on τ."""
    taus = list(taus)
    out = {t: {e: {} for e in ESTIMATORS} for t in taus}
    se_ = {t: {} for t in taus}
    pos = {i: cb.position for i, cb in cubes.items()}
    for i, cb in cubes.items():
        s = np.ones(cb.games.size, bool) if sel is None else sel[i]
        H = cb.total(s)
        n = H.sum(1)
        omax, lin = observed_max(H, c), linear_envelope_curve(H, c)[0]
        reps = bootstrap_hists(cb, s, boot, seed)
        for t in taus:
            q1 = q_empirical(H, t, c)
            o = out[t]
            o["C1_empirical"][i], o["C0a_observed_max"][i], o["C0b_linear_envelope"][i] = q1, omax, lin
            o["C2_smooth_qr"][i] = q_smooth(H, t, lam, c)
            o["C4_shape"][i] = q_shape(q1, n, t, c)
            se_[t][i] = bootstrap_se(reps, t, c)
    for t in taus:
        out[t]["C3_pooled"] = pooled(out[t]["C1_empirical"], se_[t], pos, pool_ref)
    return out


def trait(curve: np.ndarray, bins: Sequence[int], require_all: bool = True) -> float:
    """Athlete-level scalar: mean of the curve over the shared speed bins (equal bin weight). ``require_all``: NaN unless
    every shared bin is supported (no composition bias from differing speed coverage)."""
    v = curve[list(bins)]
    if require_all and not np.isfinite(v).all():
        return NAN
    return float(np.nanmean(v)) if np.isfinite(v).any() else NAN


# ============================================================================== λ tuning on a separate game split (NFL only)
def tune_lambda(cubes: dict[int, Cube], c: M12Config, bins: Sequence[int], tau: float, seed: int = 999, max_athletes: int = 150) -> pl.DataFrame:
    """Held-out-game pinball loss of C2 per λ on a dedicated random split (seed ≠ every evaluation split). No Combine value."""
    ids = sorted(cubes)[:max_athletes] if len(cubes) > max_athletes else sorted(cubes)
    _, _, _, amid = _edges(c)
    rows = []
    for lam in c.spline_lambdas:
        losses = []
        for i in ids:
            sp = game_splits(cubes[i], "random", seed)
            if sp is None:
                continue
            q = q_smooth(cubes[i].total(sp[0]), tau, lam, c)
            Hb = cubes[i].total(sp[1])
            l = held_out_pinball(q, Hb, amid, tau, bins, c)
            if np.isfinite(l):
                losses.append(l)
        rows.append({"lambda": lam, "tau": tau, "n_athletes": len(losses), "pinball": float(np.mean(losses)) if losses else NAN})
    return pl.DataFrame(rows)


# ============================================================================== forecast metrics on held-out games
def held_out_pinball(q: np.ndarray, Hb: np.ndarray, amid: np.ndarray, tau: float, bins: Sequence[int], c: M12Config) -> float:
    """Mean over the athlete's supported shared bins of the frame-averaged pinball loss of the held-out frames."""
    vals = []
    for k in bins:
        n = Hb[k].sum()
        if n < c.min_frames_bin_nfl or not np.isfinite(q[k]):
            continue
        vals.append(float((Hb[k] * sm.pinball(amid, q[k], tau)).sum() / n))
    return float(np.mean(vals)) if vals else NAN


def held_out_coverage(q: np.ndarray, Hb: np.ndarray, amid: np.ndarray, bins: Sequence[int], c: M12Config) -> float:
    vals = []
    for k in bins:
        n = Hb[k].sum()
        if n < c.min_frames_bin_nfl or not np.isfinite(q[k]):
            continue
        vals.append(float(Hb[k][amid <= q[k]].sum() / n))
    return float(np.mean(vals)) if vals else NAN


def held_out_crps(Qs: dict[float, np.ndarray], Hb: np.ndarray, amid: np.ndarray, bins: Sequence[int], c: M12Config) -> float:
    """CRPS ≈ 2 · mean over the τ grid of pinball losses (quantile decomposition on a finite grid; comparable across
    estimators evaluated on the same grid only)."""
    taus = sorted(Qs)
    per = [held_out_pinball(Qs[t], Hb, amid, t, bins, c) for t in taus]
    per = [p for p in per if np.isfinite(p)]
    return float(2.0 * np.mean(per)) if len(per) == len(taus) else NAN


def forecast_table(cubes: dict[int, Cube], c: M12Config, bins: Sequence[int], lam: float, split: str = "odd_even", seed: int = 0) -> pl.DataFrame:
    """Per athlete × estimator × τ: held-out pinball and coverage (train half → test half); CRPS for C1 / C2 / C3 on the τ
    grid; quantile-crossing rate of C2 before rearrangement. Athletes weigh equally."""
    _, _, _, amid = _edges(c)
    sel_a, sel_b = {}, {}
    for i, cb in cubes.items():
        sp = game_splits(cb, split, seed)
        if sp is not None:
            sel_a[i], sel_b[i] = sp
    sub = {i: cubes[i] for i in sel_a}
    rows = []
    taus = sorted(set(c.taus) | set(c.crps_taus))
    fits = curves(sub, sel_a, taus, c, lam, c.env_boot, seed)
    for i in sub:
        Hb = sub[i].total(sel_b[i])
        for t in c.taus:
            for e in ESTIMATORS:
                q = fits[t][e][i]
                rows.append({"nfl_id": i, "position_group": sub[i].position, "estimator": e, "tau": t,
                             "pinball": held_out_pinball(q, Hb, amid, t, bins, c), "coverage": held_out_coverage(q, Hb, amid, bins, c)})
        for e in ("C1_empirical", "C2_smooth_qr", "C3_pooled"):
            rows.append({"nfl_id": i, "position_group": sub[i].position, "estimator": e, "tau": -1.0,
                         "crps": held_out_crps({t: fits[t][e][i] for t in c.crps_taus}, Hb, amid, bins, c),
                         "crossing_rate": sm.crossing_rate(np.array([fits[t][e][i][list(bins)] for t in c.crps_taus]))})
    return pl.DataFrame(rows, infer_schema_length=None)


def forecast_summary(ft: pl.DataFrame, c: M12Config, seed: int, reference: str = "C1_empirical") -> pl.DataFrame:
    """Per estimator × τ: mean pinball / coverage, and the paired athlete-bootstrap difference against the reference
    (positive diff = estimator has LOWER loss, i.e. better)."""
    rows = []
    q = ft.filter(pl.col("tau") > 0)
    for (e, t), g in sorted(q.partition_by(["estimator", "tau"], as_dict=True).items()):
        ref = q.filter((pl.col("estimator") == reference) & (pl.col("tau") == t)).select("nfl_id", pl.col("pinball").alias("_r"))
        j = g.join(ref, on="nfl_id", how="inner")
        d = ae.paired_metric_diff(j["_r"].to_numpy(), j["pinball"].to_numpy(), max(200, c.n_boot // 5), seed)
        rows.append({"estimator": e, "tau": t, "n_athletes": g["pinball"].is_finite().sum(), "pinball": g["pinball"].mean(),
                     "coverage": g["coverage"].mean(), "coverage_gap": (g["coverage"].mean() or NAN) - t,
                     "gain_vs_ref": d["diff"], "gain_lo": d["lo"], "gain_hi": d["hi"]})
    cr = ft.filter(pl.col("tau") < 0).group_by("estimator").agg(pl.col("crps").mean(), pl.col("crossing_rate").mean())
    return pl.DataFrame(rows, infer_schema_length=None).join(cr, on="estimator", how="left")


# ============================================================================== reliability (game splits)
def reliability_tables(cubes: dict[int, Cube], c: M12Config, bins: Sequence[int], lam: float, splits: Sequence[tuple[str, int]], seed: int,
                       progress: Callable[[str], None] | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(trait reliability, per-bin reliability, per-athlete split traits). For every split × τ × estimator: athlete-level trait
    in half A and B → split-half Pearson / Spearman, Spearman–Brown reliability with athlete bootstrap CI, position-adjusted
    reliability (traits residualised on position), functional-shape agreement (within-athlete correlation of the two
    centred curves, median over athletes)."""
    trows, brows, arows = [], [], []
    for kind, s in splits:
        A, B = {}, {}
        for i, cb in cubes.items():
            sp = game_splits(cb, kind, s)
            if sp is not None:
                A[i], B[i] = sp
        sub = {i: cubes[i] for i in A}
        if len(sub) < 10:
            continue
        FA = curves(sub, A, c.taus, c, lam, c.env_boot, seed)
        FB = curves(sub, B, c.taus, c, lam, c.env_boot, seed + 1)
        for t in c.taus:
            fa, fb = FA[t], FB[t]
            for e in ESTIMATORS:
                ids = sorted(sub)
                ta = np.array([trait(fa[e][i], bins) for i in ids])
                tb = np.array([trait(fb[e][i], bins) for i in ids])
                pos = np.array([sub[i].position for i in ids])
                r = ae.split_reliability(ta, tb, max(200, c.n_boot // 5), seed)
                ra, rb = _pos_resid(ta, pos), _pos_resid(tb, pos)
                rp = ae.split_reliability(ra, rb, 0, seed)
                shapes = []
                for i in ids:
                    x, y = fa[e][i][list(bins)], fb[e][i][list(bins)]
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.sum() >= 4 and np.std(x[m]) > 1e-9 and np.std(y[m]) > 1e-9:
                        shapes.append(float(np.corrcoef(x[m], y[m])[0, 1]))
                trows.append({"split": f"{kind}:{s}", "tau": t, "estimator": e, **r, "reliability_pos_adj": rp["reliability"],
                              "shape_agreement": float(np.median(shapes)) if shapes else NAN, "n_shape": len(shapes)})
                arows += [{"split": f"{kind}:{s}", "tau": t, "estimator": e, "nfl_id": i, "trait_a": a_, "trait_b": b_} for i, a_, b_ in zip(ids, ta, tb)]
                for k in bins:
                    xa = np.array([fa[e][i][k] for i in ids])
                    xb = np.array([fb[e][i][k] for i in ids])
                    rb_ = ae.split_reliability(xa, xb, 0, seed)
                    brows.append({"split": f"{kind}:{s}", "tau": t, "estimator": e, "speed_bin": int(k), "n_athletes": rb_["n_athletes"],
                                  "reliability": rb_["reliability"]})
            if progress:
                progress(f"{kind}:{s} τ={t}")
    return pl.DataFrame(trows, infer_schema_length=None), pl.DataFrame(brows, infer_schema_length=None), pl.DataFrame(arows, infer_schema_length=None)


def reliability_difference(splitvals: pl.DataFrame, est_a: str, est_b: str, tau: float, split: str, n_boot: int, seed: int) -> dict[str, Any]:
    """Spearman–Brown split-half reliability of estimator A minus that of B on the SAME athletes (both traits finite in both
    halves for both estimators), athlete bootstrap. Positive = A more reliable."""
    s = splitvals.filter((pl.col("tau") == tau) & (pl.col("split") == split))
    a = s.filter(pl.col("estimator") == est_a).select("nfl_id", pl.col("trait_a").alias("a1"), pl.col("trait_b").alias("a2"))
    b = s.filter(pl.col("estimator") == est_b).select("nfl_id", pl.col("trait_a").alias("b1"), pl.col("trait_b").alias("b2"))
    j = a.join(b, on="nfl_id").drop_nulls()
    j = j.filter(pl.all_horizontal([pl.col(k).is_finite() for k in ("a1", "a2", "b1", "b2")]))
    n = j.height
    out = {"estimator": est_a, "reference": est_b, "tau": tau, "split": split, "n_athletes": n, "rel_a": NAN, "rel_b": NAN, "diff": NAN, "lo": NAN, "hi": NAN}
    if n < 10:
        return out
    X = j.select("a1", "a2", "b1", "b2").to_numpy()
    sb = lambda u, v: ae.spearman_brown(float(np.corrcoef(u, v)[0, 1]), 2)
    out["rel_a"], out["rel_b"] = sb(X[:, 0], X[:, 1]), sb(X[:, 2], X[:, 3])
    out["diff"] = out["rel_a"] - out["rel_b"]
    if n >= 20:
        idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
        d = np.array([sb(X[i, 0], X[i, 1]) - sb(X[i, 2], X[i, 3]) for i in idx])
        d = d[np.isfinite(d)]
        if d.size >= 20:
            out["lo"], out["hi"] = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    return out


def _pos_resid(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.full(x.size, NAN)
    for p in np.unique(pos):
        m = (pos == p) & np.isfinite(x)
        if m.sum() >= 3:
            out[m] = x[m] - x[m].mean()
    return out


def speed_reliability(gts: pl.DataFrame, c: M12Config, seed: int, qs: Sequence[float] = (0.5, 0.75, 0.9)) -> pl.DataFrame:
    """Game-split reliability of per-athlete quantiles of game top speed (odd / even game order), athletes with ≥ min games."""
    rows = []
    d = gts.sort("nfl_id", "game_id").with_columns(pl.int_range(pl.len()).over("nfl_id").alias("_k"), pl.len().over("nfl_id").alias("_g"))
    d = d.filter(pl.col("_g") >= c.min_games_athlete)
    for q in qs:
        h = d.group_by("nfl_id", (pl.col("_k") % 2).alias("_h")).agg(pl.col("top_speed").quantile(q, "linear").alias("_v"))
        w = h.pivot(on="_h", index="nfl_id", values="_v").drop_nulls()
        r = ae.split_reliability(w[w.columns[1]].to_numpy(), w[w.columns[2]].to_numpy(), max(200, c.n_boot // 5), seed) if w.width == 3 else {"n_athletes": 0}
        rows.append({"target": f"nfl.game_top_speed_q{int(q * 100)}", "q": q, **r})
    return pl.DataFrame(rows, infer_schema_length=None)


# ============================================================================== target freeze (declared rule, before any Combine value)
FREEZE_NAME = "m12c_target_freeze"


def freeze_rule(rel: pl.DataFrame, c: M12Config) -> dict[str, Any]:
    """Among the QUANTILE estimators at τ = primary_tau on the odd / even split, the highest trait reliability (ties →
    higher lower bound, then simpler estimator order C1 < C4 < C3 < C2). Requires reliability ≥ 0.5; otherwise C1 with
    flag ``below_reliability_floor``. Written once; a different later freeze is refused."""
    r = rel.filter((pl.col("split") == "odd_even:0") & (pl.col("tau") == c.primary_tau) & pl.col("estimator").is_in(list(QUANTILE_ESTIMATORS)))
    order = {"C1_empirical": 0, "C4_shape": 1, "C3_pooled": 2, "C2_smooth_qr": 3}
    rows = sorted(r.iter_rows(named=True), key=lambda x: (-(x["reliability"] if x["reliability"] is not None and np.isfinite(x["reliability"]) else -9),
                                                          -(x["rel_lo"] if x["rel_lo"] is not None and np.isfinite(x["rel_lo"]) else -9), order[x["estimator"]]))
    if not rows:
        return {"estimator": "C1_empirical", "tau": c.primary_tau, "flag": "no_reliability_rows"}
    best = rows[0]
    flag = "" if (best["reliability"] or 0) >= 0.5 else "below_reliability_floor"
    return {"estimator": best["estimator"] if not flag else "C1_empirical", "tau": c.primary_tau, "reliability": best["reliability"],
            "rel_lo": best["rel_lo"], "flag": flag, "candidates": [{k: x[k] for k in ("estimator", "reliability", "rel_lo", "n_athletes")} for x in rows]}


def write_freeze(store: Any, decision: dict[str, Any], bins: Sequence[int], lam: float, c: M12Config) -> dict[str, Any]:
    p = store.data_dir / f"{FREEZE_NAME}.json"
    body = {"primary_target": "nfl.q90_shared", "estimator": decision["estimator"], "tau": decision["tau"], "shared_bins": list(bins), "lambda_c2": lam,
            "secondary_target": "nfl.game_top_speed_q75", "rule": freeze_rule.__doc__.strip().split("\n")[0], "decision": decision,
            "config_hash": c.hash(), "frozen_at": utc_now()}
    body["freeze_hash"] = config_hash({k: body[k] for k in ("estimator", "tau", "shared_bins", "lambda_c2", "config_hash")})
    if p.is_file():
        have = json.loads(p.read_text(encoding="utf-8"))
        if have.get("freeze_hash") != body["freeze_hash"]:
            raise RuntimeError(f"a different 12C target freeze exists ({have.get('estimator')}, {have.get('frozen_at')}). The freeze is immutable: "
                               "delete it by hand ONLY if the reliability inputs were wrong, and record why in docs/RESEARCH_DECISIONS.md.")
        return have
    if store.controls.save_tables:
        store.write_json(FREEZE_NAME, body)
    return body


def read_freeze(store: Any) -> dict[str, Any]:
    return store.read_json(FREEZE_NAME, "12c_distributional_nfl_movement_capacity.ipynb")


# ============================================================================== athlete targets (all games)
def athlete_targets(cubes: dict[int, Cube], gts: pl.DataFrame, envelopes: pl.DataFrame, freeze: dict[str, Any], c: M12Config, seed: int) -> pl.DataFrame:
    bins, est, lam = freeze["shared_bins"], freeze["estimator"], freeze["lambda_c2"]
    rows = {i: {"nfl_id": i, "position_group": cb.position, "n_games": int(cb.games.size), "n_frames_shared": int(cb.total()[bins].sum())} for i, cb in cubes.items()}
    F = curves(cubes, None, c.taus, c, lam, c.env_boot, seed)
    for t in c.taus:
        f = F[t]
        for i in cubes:
            rows[i][f"nfl.q{int(t * 100)}_shared"] = trait(f[est][i], bins)
            if t == c.primary_tau:
                for e in ESTIMATORS:
                    rows[i][f"est.{e}_shared"] = trait(f[e][i], bins)
    g = (gts.group_by("nfl_id").agg(pl.len().alias("_g"), *[pl.col("top_speed").quantile(q, "linear").alias(f"nfl.game_top_speed_q{int(q * 100)}") for q in (0.5, 0.75, 0.9)])
         .filter(pl.col("_g") >= c.min_games_athlete).drop("_g"))
    out = pl.DataFrame(list(rows.values()), infer_schema_length=None).join(g, on="nfl_id", how="left")
    if envelopes.height:
        keep = [pl.col(k).alias(f"env.{k}") for k in ("lin_V0", "lin_A0", "lin_V0_se", "lin_V0_q90", "lin_V0_q99", "upper_tail_accel") if k in envelopes.columns]
        out = out.join(envelopes.filter(pl.col("supported")).select("nfl_id", *keep), on="nfl_id", how="left")
    return out.sort("nfl_id")


# ============================================================================== transfer ladder (M0–M6)
LADDER_FEATURES = {"M2": ("forty.v_max_obs",), "M3": ("forty.sprint_vmax",), "M4": ("forty.phase_V0",)}
PRIMARY_COMPARISONS = (("M3", "M1", "nfl.q90_shared"), ("M2", "M1", "nfl.game_top_speed_q75"))


def ladder(m5_factory: Callable[[np.ndarray], ae.BlockFn] | None = None, m6_factory: Callable[[np.ndarray], ae.BlockFn] | None = None,
           m1_cols: Sequence[str] = ae.M1_COLS) -> dict[str, ae.Model]:
    """M0–M6. M5 / M6 carry FOLD-FITTED blocks through factories (aligned with the final analysis rows by nfl_id)."""
    m = {"M0": ae.Model(list(ae.M0_COLS), note="position + draft year"), "M1": ae.Model(list(m1_cols), note="M0 + height, weight, official forty, 10-yd split")}
    for k, cols in LADDER_FEATURES.items():
        m[k] = ae.Model([*m1_cols, *cols], note=f"M1 + {', '.join(cols)}")
    if m5_factory is not None:
        m["M5"] = ae.Model(list(m1_cols), factory=m5_factory, note="M1 + functional speed profile (fold-fitted FPCA of the Combine a(v) curve)")
    if m6_factory is not None:
        m["M6"] = ae.Model(list(m1_cols), factory=m6_factory, note="M1 + selected multi-view latent representation (12D)")
    return m


def transfer(data: pl.DataFrame, targets: Sequence[str], models: dict[str, ae.Model], c: M12Config, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every model on the SAME athletes per target (rows with the target AND every static column of every model non-null:
    the common cohort). Comparisons: each model vs M1, M1 vs M0. Primary comparisons Holm-adjusted together."""
    static = sorted({k for m in models.values() for k in m.cols} - set(ae.M1_COLS) - set(ae.M0_COLS))
    rows, oofs = [], []
    for tg in targets:
        if tg not in data.columns:
            continue
        res = ae.oof_ladder(data, tg, models, c, seed, require=static)
        comps = [("M1", "M0")] + [(k, "M1") for k in models if k not in ("M0", "M1")]
        rows.append(ae.compare(res, comps, c, seed, target=tg))
        oofs.append(res.oof_frame(target=tg))
    out = pl.concat([r for r in rows if r.height], how="diagonal_relaxed") if rows else pl.DataFrame()
    if out.height:
        prim = [any(m == r["model"] and ref == r["reference"] and t == r["target"] for m, ref, t in PRIMARY_COMPARISONS) for r in out.iter_rows(named=True)]
        out = out.with_columns(pl.Series("primary", prim))
        pp = out.filter(pl.col("primary"))
        adj = dict(zip(zip(pp["model"], pp["target"]), ae.holm(pp["p_boot_le0"].to_list())))
        out = out.with_columns(pl.Series("p_holm_primary", [adj.get((r["model"], r["target"])) for r in out.iter_rows(named=True)], dtype=pl.Float64))
        ex = out.filter(~pl.col("primary"))
        bh = dict(zip(zip(ex["model"], ex["target"], ex["reference"]), ae.bh(ex["p_boot_le0"].to_list())))
        out = out.with_columns(pl.Series("p_bh_exploratory", [bh.get((r["model"], r["target"], r["reference"])) for r in out.iter_rows(named=True)], dtype=pl.Float64))
    return out, (pl.concat(oofs, how="diagonal_relaxed") if oofs else pl.DataFrame())


# ============================================================================== C7 high-speed replication
REPLICATION_PAIRS = (("forty.sprint_vmax", "env.lin_V0", "11B: sprint Vmax → NFL linear-envelope V0"),
                     ("forty.v_max_obs", "nfl.vmax_p90", "11B: observed Combine top speed → NFL observed high-speed peak (p90)"))


def replication_variants(base: pl.DataFrame, extra_targets: dict[str, pl.DataFrame]) -> dict[str, tuple[pl.DataFrame, str]]:
    """Declared variants (all reported; none selected afterwards). Each value: (analysis frame, target column)."""
    v = {}
    for feat, tgt, _ in REPLICATION_PAIRS:
        v[f"{feat}|{tgt}|as_11B"] = (base, tgt)
    alt = {"forty.sprint_vmax": ("nfl.q90_shared", "nfl.q95_shared", "env.lin_V0_q90", "env.lin_V0_q99"),
           "forty.v_max_obs": ("nfl.game_top_speed_q50", "nfl.game_top_speed_q75", "nfl.game_top_speed_q90")}
    for feat, tgts in alt.items():
        for t in tgts:
            v[f"{feat}|{t}|alt_estimator"] = (base, t)
    for name, d in extra_targets.items():
        for feat, tgt, _ in REPLICATION_PAIRS:
            v[f"{feat}|{tgt}|{name}"] = (d, tgt)
    return v


def replication_table(variants: dict[str, tuple[pl.DataFrame, str]], c: M12Config, seed: int) -> pl.DataFrame:
    """M1 vs M1 + feature per variant, each on its own complete cohort, PLUS the common cohort of every variant of the
    same feature (identical athletes) — read the common-cohort rows for comparisons between variants."""
    rows = []
    for key, (d, tgt) in variants.items():
        feat, _, variant = key.split("|")
        if tgt not in d.columns or feat not in d.columns:
            rows.append({"feature": feat, "target": tgt, "variant": variant, "status": "not_available"})
            continue
        mods = {"M1": ae.Model(list(ae.M1_COLS)), "M1+f": ae.Model([*ae.M1_COLS, feat])}
        res = ae.oof_ladder(d, tgt, mods, c, seed, require=[feat])
        r = ae.compare(res, [("M1+f", "M1")], c, seed, feature=feat, target=tgt, variant=variant)
        rows += [{**x, "status": "computed"} for x in r.iter_rows(named=True)]
    return pl.DataFrame(rows, infer_schema_length=None)


def replication_summary(rep: pl.DataFrame) -> pl.DataFrame:
    """Per feature: variants computed, share with Δ CI > 0, share with Δ > 0 (sign consistency), median Δ%."""
    r = rep.filter(pl.col("status") == "computed")
    if r.is_empty():
        return pl.DataFrame()
    return (r.group_by("feature").agg(pl.len().alias("n_variants"), (pl.col("delta_lo") > 0).mean().alias("share_ci_above_0"),
                                      (pl.col("delta_rmse") > 0).mean().alias("share_positive"), pl.col("delta_pct").median().alias("median_delta_pct"),
                                      pl.col("n_athletes").min().alias("min_n"), pl.col("n_athletes").max().alias("max_n")).sort("feature"))


__all__ = ["Cube", "ESTIMATORS", "FREEZE_NAME", "LADDER_FEATURES", "NOT_IMPLEMENTED", "PRIMARY_COMPARISONS", "QUANTILE_ESTIMATORS", "REPLICATION_PAIRS",
           "athlete_targets", "bootstrap_hists", "bootstrap_se", "build_cubes", "q_rows", "curves", "forecast_summary", "forecast_table", "freeze_rule", "game_splits", "held_out_coverage",
           "held_out_crps", "held_out_pinball", "ladder", "linear_envelope_curve", "observed_max", "pooled", "q_empirical", "q_shape", "q_smooth", "read_freeze",
           "reliability_difference", "reliability_tables", "replication_summary", "replication_table", "replication_variants", "speed_reliability", "trait", "transfer", "tune_lambda",
           "write_freeze"]
