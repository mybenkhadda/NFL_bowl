"""12A — functional movement transfer. NumPy / Polars.

Functions
    Combine a_C(v)   mean tangential acceleration per 0.5 yd/s speed bin on the RISING phase of each forty attempt
                     (``m12_combine_curves``); athlete curve = frame-weighted mean over attempts per bin
    NFL a_N(v)       the 12C frozen estimator of the τ-quantile per speed bin (default τ = 0.90) from the athlete's games
    Both are used ONLY on the shared speed range of 12.0 and only on bins the athlete supports. One interior gap of
    ≤ ``max_interior_gap_bins`` bins may be bridged linearly (counted and flagged); nothing is extrapolated.

Representations (every fitted quantity from TRAINING athletes of the fold)
    A0  scalar band features (Milestone 11 forty.band_2_4 / 4_6 / 6_8)
    A1  P-spline coefficients; λ chosen by REPEAT-ATTEMPT agreement (fit attempt 1, predict attempt 2) on training athletes
    A2  FPCA scores; K = smallest number reaching ``fpca_var`` training variance (≤ ``fpca_max_k``); athletes with partial
        support scored by penalised conditional projection on their observed bins
    A3  function-on-function regression: NFL FPC scores (NFL FPCA on training athletes) ← Combine predictors (ridge), curve
        reconstructed; β(v, u) = Φ_N B Φ_Cᵀ is available for display
    A4  hierarchical functional model: athlete curve = position mean + shrunk athlete deviation per bin (between-athlete
        variance vs within-athlete attempt variance / n attempts) — a penalised mixed model, not a sampler
Elastic analysis (exploratory): landmark registration (times to 4 / 6 / 8 yd/s) and SRSF dynamic-programming alignment
of speed–time curves. Phase (timing) is REPORTED, never discarded: the timing feature under study is not registered away.

Evaluation: per-athlete mean squared error over the athlete's supported shared bins (athletes weigh equally — a wider
speed range does not count more), paired athlete bootstrap of Δ, pointwise Δ per speed bin, Spearman of athlete levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from ..m11 import evaluation as ev
from ..transfer_models import ridge_solve
from . import advanced_evaluation as ae
from . import smoothing as sm
from .config import M12Config

NAN = float("nan")


# ============================================================================== matrices on the shared grid
def _bridge(row: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    """Linear bridge of interior NaN runs of length ≤ max_gap (ends are never filled). Returns (row, bridged count)."""
    x = row.copy()
    ok = np.flatnonzero(np.isfinite(x))
    if ok.size < 2:
        return x, 0
    bridged = 0
    for a, b in zip(ok[:-1], ok[1:]):
        if 1 < b - a <= max_gap + 1:
            x[a + 1: b] = np.interp(np.arange(a + 1, b), [a, b], [x[a], x[b]])
            bridged += b - a - 1
    return x, bridged


def combine_athlete_matrix(curves: pl.DataFrame, bins: Sequence[int], c: M12Config, ids: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ids, M [athletes × bins] mean over attempts of the interpolated a at the bin mid-speed, bridged count per athlete)."""
    d = curves.filter(pl.col("speed_bin").is_in(list(bins)) & (pl.col("n_frames") >= c.min_frames_bin_combine))
    if ids is not None:
        d = d.filter(pl.col("nfl_id").is_in(list(ids)))
    a = d.group_by("nfl_id", "speed_bin").agg(pl.col("a_mean").mean().alias("a"))
    uid = np.sort(a["nfl_id"].unique().to_numpy()) if ids is None else np.asarray(sorted(set(int(i) for i in ids)))
    pos = {int(i): k for k, i in enumerate(uid)}
    col = {int(b): k for k, b in enumerate(bins)}
    M = np.full((uid.size, len(bins)), np.nan)
    for r in a.iter_rows(named=True):
        if int(r["nfl_id"]) in pos:
            M[pos[int(r["nfl_id"])], col[int(r["speed_bin"])]] = r["a"]
    br = np.zeros(uid.size, int)
    for i in range(uid.size):
        M[i], br[i] = _bridge(M[i], c.max_interior_gap_bins)
    return uid, M, br


def combine_attempt_matrix(curves: pl.DataFrame, bins: Sequence[int], c: M12Config) -> tuple[pl.DataFrame, np.ndarray]:
    """(attempt meta [seq_id, nfl_id, attempt_order], M [attempts × bins])."""
    d = curves.filter(pl.col("speed_bin").is_in(list(bins)) & (pl.col("n_frames") >= c.min_frames_bin_combine))
    meta = d.select("seq_id", "nfl_id", "attempt_order").unique().sort("nfl_id", "attempt_order", "seq_id")
    row = {s: k for k, s in enumerate(meta["seq_id"].to_list())}
    col = {int(b): k for k, b in enumerate(bins)}
    M = np.full((meta.height, len(bins)), np.nan)
    for r in d.iter_rows(named=True):
        M[row[r["seq_id"]], col[int(r["speed_bin"])]] = r["a_mean"]
    for i in range(M.shape[0]):
        M[i] = _bridge(M[i], c.max_interior_gap_bins)[0]
    return meta, M


# ============================================================================== FPCA with partial curves
@dataclass
class FPCA:
    mean: np.ndarray
    phi: np.ndarray            # (bins, K)
    eig: np.ndarray
    var_share: np.ndarray
    n_fit: int

    @staticmethod
    def fit(M: np.ndarray, var: float, max_k: int, min_complete: int = 10) -> "FPCA | None":
        full = M[np.isfinite(M).all(1)]
        if full.shape[0] < max(min_complete, M.shape[1] // 2):
            return None
        mu = full.mean(0)
        U, s, Vt = np.linalg.svd(full - mu, full_matrices=False)
        ev_ = s ** 2 / max(full.shape[0] - 1, 1)
        share = np.cumsum(ev_) / ev_.sum() if ev_.sum() > 0 else np.ones_like(ev_)
        K = int(min(max_k, np.searchsorted(share, var) + 1, Vt.shape[0]))
        phi = Vt[:K].T.copy()
        for k in range(K):                                # sign convention: largest |loading| positive
            if phi[np.argmax(np.abs(phi[:, k])), k] < 0:
                phi[:, k] *= -1
        return FPCA(mu, phi, ev_[:K], share[:K], int(full.shape[0]))

    def scores(self, M: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
        """Conditional projection on the observed bins of each row (penalised by the inverse eigenvalues)."""
        out = np.full((M.shape[0], self.phi.shape[1]), np.nan)
        P = np.diag(ridge / np.maximum(self.eig, 1e-9))
        for i in range(M.shape[0]):
            o = np.isfinite(M[i])
            if o.sum() < self.phi.shape[1] + 1:
                continue
            F = self.phi[o]
            out[i] = np.linalg.solve(F.T @ F + P, F.T @ (M[i, o] - self.mean[o]))
        return out

    def reconstruct(self, S: np.ndarray) -> np.ndarray:
        return self.mean + S @ self.phi.T


def fpca_block(analysis_ids: np.ndarray, comb_ids: np.ndarray, M: np.ndarray, c: M12Config, prefix: str = "fpc") -> ae.BlockFn:
    """Fold-fitted FPCA scores of the Combine curve for the analysis rows (``analysis_ids`` in row order). The FPCA is fitted
    on the TRAINING rows' curves only; athletes without a curve get NaN (imputed by the train-only design + indicator)."""
    row_of = {int(i): k for k, i in enumerate(comb_ids)}
    idx = np.array([row_of.get(int(i), -1) for i in analysis_ids])

    def block(train: np.ndarray) -> pl.DataFrame:
        tr_rows = idx[train & (idx >= 0)]
        f = FPCA.fit(M[tr_rows], c.fpca_var, c.fpca_max_k)
        K = c.fpca_max_k if f is None else f.phi.shape[1]
        S = np.full((idx.size, K), np.nan)
        if f is not None:
            have = idx >= 0
            S[have] = f.scores(M[idx[have]])
        return pl.DataFrame({f"{prefix}{k}": S[:, k] for k in range(K)})
    return block


# ============================================================================== P-spline coefficients (A1)
def spline_basis_for(bins: Sequence[int], c: M12Config, n_interior: int = 4) -> tuple[np.ndarray, float, float]:
    se = ae.speed_edges(c)
    lo, hi = float(se[bins[0]]), float(se[bins[-1] + 1])
    x = ae.mids(se)[list(bins)]
    return sm.bspline_basis(x, lo, hi, n_interior), lo, hi


def spline_coefs(M: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    out = np.full((M.shape[0], B.shape[1]), np.nan)
    for i in range(M.shape[0]):
        o = np.isfinite(M[i])
        if o.sum() >= 4:
            out[i] = sm.pspline_fit(B[o], M[i, o], None, lam)
    return out


def choose_lambda_repeat(att_meta: pl.DataFrame, A: np.ndarray, B: np.ndarray, train_ids: set[int], lams: Sequence[float]) -> tuple[float, pl.DataFrame]:
    """λ minimising the error of a curve fitted on an athlete's first attempt in predicting the second attempt's observed
    bins (training athletes with ≥ 2 attempts only): smoothing judged against an INDEPENDENT repeat, not the same data."""
    m = att_meta.with_row_index("_r")
    pairs = []
    for (i,), g in m.partition_by(["nfl_id"], as_dict=True).items():
        if int(i) in train_ids and g.height >= 2:
            r = g.sort("attempt_order")["_r"].to_list()
            pairs.append((r[0], r[1]))
    rows = []
    for lam in lams:
        err = []
        for a, b in pairs:
            oa, ob = np.isfinite(A[a]), np.isfinite(A[b])
            if oa.sum() < 4 or ob.sum() < 2:
                continue
            beta = sm.pspline_fit(B[oa], A[a, oa], None, lam)
            err.append(float(np.mean((B[ob] @ beta - A[b, ob]) ** 2)))
        rows.append({"lambda": lam, "n_pairs": len(err), "repeat_mse": float(np.mean(err)) if err else NAN})
    t = pl.DataFrame(rows)
    ok = t.filter(pl.col("repeat_mse").is_finite())
    return (float(ok.sort("repeat_mse", "lambda")["lambda"][0]) if ok.height else float(lams[len(lams) // 2])), t


def spline_block(analysis_ids: np.ndarray, comb_ids: np.ndarray, M: np.ndarray, B: np.ndarray, lam: float, prefix: str = "spl") -> ae.BlockFn:
    """P-spline coefficients (λ fixed beforehand by ``choose_lambda_repeat`` on training athletes). No fitted cross-athlete
    quantity: the block is per-athlete and identical in every fold."""
    row_of = {int(i): k for k, i in enumerate(comb_ids)}
    C = spline_coefs(M, B, lam)
    S = np.array([C[row_of[int(i)]] if int(i) in row_of else np.full(B.shape[1], np.nan) for i in analysis_ids])

    def block(train: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({f"{prefix}{k}": S[:, k] for k in range(S.shape[1])})
    return block


# ============================================================================== function-on-function regression (A3) and pointwise
@dataclass
class FunctionalResult:
    ids: np.ndarray
    positions: np.ndarray
    Y: np.ndarray
    preds: dict[str, np.ndarray]


def functional_oof(df: pl.DataFrame, Y: np.ndarray, models: dict[str, ae.Model], c: M12Config, seed: int, mode: str = "fpc") -> FunctionalResult:
    """Out-of-fold prediction of the NFL curve of each athlete. ``df`` rows align with ``Y`` rows and hold nfl_id, fold,
    position_group and the static columns. ``mode``: ``fpc`` = ridge on NFL FPC scores (NFL FPCA fitted on training rows),
    curve reconstructed (function-on-function regression); ``pointwise`` = one ridge per speed bin (training rows that
    observe that bin). The penalty is chosen per fold by inner athlete folds on the curve error."""
    folds = df["fold"].to_numpy()
    ev.assert_grouped(df["nfl_id"].to_numpy(), folds)
    preds = {m: np.full(Y.shape, np.nan) for m in models}
    blocks = {name: m.resolve(df["nfl_id"].to_numpy()) for name, m in models.items()}
    for f in sorted(set(folds.tolist())):
        test, train = folds == f, folds != f
        if test.sum() == 0 or train.sum() < 15:
            continue
        nf = FPCA.fit(Y[train], c.fpca_var, c.fpca_max_k) if mode == "fpc" else None
        if mode == "fpc" and nf is None:
            continue
        T = nf.scores(Y) if nf is not None else None
        for name, m in models.items():
            extra = blocks[name](train) if blocks[name] is not None else None
            alpha = _choose_alpha_curve(df, Y, T, nf, m.cols, extra, train, c, seed + 17 * int(f), mode)
            preds[name][test] = _predict_curve(df, Y, T, nf, m.cols, extra, train, test, alpha, mode)
    return FunctionalResult(df["nfl_id"].to_numpy(), df["position_group"].fill_null("?").to_numpy(), Y, preds)


def _design(df: pl.DataFrame, cols: Sequence[str], extra: pl.DataFrame | None, train: np.ndarray) -> ev.Design:
    d = df if extra is None or extra.width == 0 else pl.concat([df, extra.rename({k: f"__b_{k}" for k in extra.columns})], how="horizontal")
    return ev.design(d, [*cols, *([f"__b_{k}" for k in extra.columns] if extra is not None else [])], train)


def _predict_curve(df, Y, T, nf, cols, extra, train, test, alpha, mode):
    d = _design(df, cols, extra, train)
    if mode == "fpc":
        S = np.zeros((int(test.sum()), T.shape[1]))
        for k in range(T.shape[1]):
            ok = train & np.isfinite(T[:, k])
            b0, beta = ridge_solve(d.X[ok], T[ok, k], alpha, d.pen)
            S[:, k] = b0 + d.X[test] @ beta
        return nf.reconstruct(S)
    out = np.full((int(test.sum()), Y.shape[1]), np.nan)
    for k in range(Y.shape[1]):
        ok = train & np.isfinite(Y[:, k])
        if ok.sum() >= 10:
            b0, beta = ridge_solve(d.X[ok], Y[ok, k], alpha, d.pen)
            out[:, k] = b0 + d.X[test] @ beta
    return out


def _choose_alpha_curve(df, Y, T, nf, cols, extra, train, c, seed, mode) -> float:
    idx = np.flatnonzero(train)
    if idx.size < 2 * c.inner_folds:
        return float(max(c.alphas))
    inner = np.random.default_rng(seed).permutation(idx.size) % c.inner_folds
    sse = np.zeros(len(c.alphas))
    for j in range(c.inner_folds):
        tr, va = np.zeros(df.height, bool), np.zeros(df.height, bool)
        tr[idx[inner != j]], va[idx[inner == j]] = True, True
        for a_i, a in enumerate(c.alphas):
            P = _predict_curve(df, Y, T, nf, cols, extra, tr, va, a, mode)
            e = (P - Y[va]) ** 2
            sse[a_i] += float(np.nanmean(e)) if np.isfinite(e).any() else 0.0
    best = np.flatnonzero(sse <= sse.min() * (1 + 1e-9))
    return float(c.alphas[int(best.max())])


def athlete_mse(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Mean squared error over each athlete's OWN observed bins (equal athlete weight)."""
    e = (Y - P) ** 2
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(e).any(1), np.nanmean(np.where(np.isfinite(e), e, np.nan), 1), np.nan)


def compare_functional(res: FunctionalResult, comparisons: Sequence[tuple[str, str]], c: M12Config, seed: int) -> pl.DataFrame:
    """ΔMSE = MSE(ref) − MSE(model) per athlete (positive = model better), paired athlete bootstrap; Spearman of athlete
    mean levels (observed vs predicted)."""
    rows = []
    for mod, ref in comparisons:
        if mod not in res.preds or ref not in res.preds:
            continue
        a, b = athlete_mse(res.Y, res.preds[ref]), athlete_mse(res.Y, res.preds[mod])
        d = ae.paired_metric_diff(a, b, c.n_boot, seed)
        lvl_y = np.nanmean(res.Y, 1)
        rows.append({"model": mod, "reference": ref, "n_athletes": d["n_athletes"], "mse_ref": d["a"], "mse_model": d["b"], "delta_mse": d["diff"],
                     "delta_lo": d["lo"], "delta_hi": d["hi"], "delta_pct": 100 * d["diff"] / d["a"] if d["a"] else NAN,
                     "spearman_level_model": ev.spearman(lvl_y, np.nanmean(res.preds[mod], 1)), "spearman_level_ref": ev.spearman(lvl_y, np.nanmean(res.preds[ref], 1)),
                     "verdict": ae.reading(d["lo"], d["hi"])})
    return pl.DataFrame(rows, infer_schema_length=None)


def pointwise_effect(res: FunctionalResult, mod: str, ref: str, speed_mids: np.ndarray, c: M12Config, seed: int) -> pl.DataFrame:
    """Per speed bin: Δ squared error (ref − model) with an athlete bootstrap — where along the a(v) function transfer sits."""
    rows = []
    for k in range(res.Y.shape[1]):
        a = (res.Y[:, k] - res.preds[ref][:, k]) ** 2
        b = (res.Y[:, k] - res.preds[mod][:, k]) ** 2
        d = ae.paired_metric_diff(a, b, max(200, c.n_boot // 5), seed + k)
        rows.append({"speed_mid": float(speed_mids[k]), "n_athletes": d["n_athletes"], "delta_se": d["diff"], "lo": d["lo"], "hi": d["hi"]})
    return pl.DataFrame(rows)


def fof_surface(Phi_N: np.ndarray, Bcoef: np.ndarray, Phi_C: np.ndarray) -> np.ndarray:
    """β(v, u) = Φ_N · B · Φ_Cᵀ (display only)."""
    return Phi_N @ Bcoef @ Phi_C.T


# ============================================================================== hierarchical functional model (A4)
def hierarchical_curves(meta: pl.DataFrame, A: np.ndarray, positions: dict[int, str]) -> tuple[dict[int, np.ndarray], pl.DataFrame]:
    """Per bin: position mean μ_p(v), between-athlete variance τ²_p(v) and pooled within-athlete attempt variance σ²(v)
    (method of moments); athlete curve = μ_p + w (ȳ_i − μ_p), w = τ²/(τ² + σ²/n_i). Returns (curves, components)."""
    ids = meta["nfl_id"].to_numpy()
    uid = np.unique(ids)
    K = A.shape[1]
    mean_i, n_i = {}, {}
    ss, dof = np.zeros(K), np.zeros(K)
    for i in uid:
        rows = A[ids == i]
        with np.errstate(invalid="ignore"):
            mean_i[int(i)] = np.nanmean(rows, 0)
        n_i[int(i)] = np.isfinite(rows).sum(0)
        r = rows - mean_i[int(i)]
        ss += np.nansum(r ** 2, 0)
        dof += np.maximum(n_i[int(i)] - 1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma2 = np.where(dof > 0, ss / np.maximum(dof, 1), np.nan)
    comps, out = [], {}
    for p in sorted(set(positions.get(int(i), "?") for i in uid)):
        mem = [int(i) for i in uid if positions.get(int(i), "?") == p]
        M = np.array([mean_i[i] for i in mem])
        Nn = np.array([n_i[i] for i in mem], float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.nanmean(M, 0)
            tau2 = np.maximum(np.nanvar(M, 0) - np.nanmean(sigma2 / np.where(Nn > 0, Nn, np.nan), 0), 0.0)
        for i, m_, nn in zip(mem, M, Nn):
            with np.errstate(invalid="ignore", divide="ignore"):
                w = tau2 / (tau2 + sigma2 / np.where(nn > 0, nn, np.nan))
            out[i] = np.where(np.isfinite(m_), np.where(np.isfinite(w), w, 1.0) * m_ + (1 - np.where(np.isfinite(w), w, 1.0)) * mu, np.nan)
        comps += [{"position_group": p, "bin_index": k, "mu": float(mu[k]), "tau2_between": float(tau2[k]), "sigma2_within": float(sigma2[k]),
                   "icc_bin": float(tau2[k] / (tau2[k] + sigma2[k])) if np.isfinite(tau2[k] + sigma2[k]) and tau2[k] + sigma2[k] > 0 else NAN,
                   "n_athletes": len(mem)} for k in range(K)]
    return out, pl.DataFrame(comps)


def hierarchical_apply(meta: pl.DataFrame, A: np.ndarray, positions: dict[int, str], ref_ids: set[int]) -> dict[int, np.ndarray]:
    """Shrunk curves for EVERY athlete of ``meta`` with μ_p, τ²_p, σ² estimated from ``ref_ids`` only (training athletes)."""
    ids = meta["nfl_id"].to_numpy()
    ref_rows = np.isin(ids, list(ref_ids))
    _, comps = hierarchical_curves(meta.filter(pl.Series(ref_rows)), A[ref_rows], positions)
    par = {(r["position_group"], r["bin_index"]): r for r in comps.iter_rows(named=True)}
    sig = comps.group_by("bin_index").agg(pl.col("sigma2_within").first()).sort("bin_index")["sigma2_within"].to_numpy()
    pooled_mu = comps.group_by("bin_index").agg(pl.col("mu").mean()).sort("bin_index")["mu"].to_numpy()
    out = {}
    for i in np.unique(ids):
        rows = A[ids == i]
        with np.errstate(invalid="ignore"):
            m_ = np.nanmean(rows, 0)
        n_ = np.isfinite(rows).sum(0).astype(float)
        p = positions.get(int(i), "?")
        cur = np.full(A.shape[1], np.nan)
        for k in range(A.shape[1]):
            if not np.isfinite(m_[k]):
                continue
            r = par.get((p, k))
            mu, tau2 = (r["mu"], r["tau2_between"]) if r is not None else (pooled_mu[k] if k < pooled_mu.size else m_[k], np.nan)
            s2 = sig[k] if k < sig.size else np.nan
            w = tau2 / (tau2 + s2 / n_[k]) if np.isfinite(tau2) and np.isfinite(s2) and n_[k] > 0 and tau2 + s2 / n_[k] > 0 else 1.0
            cur[k] = w * m_[k] + (1 - w) * mu
        out[int(i)] = cur
    return out


def hierarchical_block(analysis_ids: np.ndarray, meta: pl.DataFrame, A: np.ndarray, positions: dict[int, str], c: M12Config, prefix: str = "hfpc") -> ae.BlockFn:
    """A4 for the ladder: shrinkage parameters AND the FPCA of the shrunk curves from training athletes only."""
    def block(train: np.ndarray) -> pl.DataFrame:
        ref = set(int(i) for i in analysis_ids[train])
        cur = hierarchical_apply(meta, A, positions, ref)
        M = np.array([cur.get(int(i), np.full(A.shape[1], np.nan)) for i in analysis_ids])
        f = FPCA.fit(M[train], c.fpca_var, c.fpca_max_k)
        K = c.fpca_max_k if f is None else f.phi.shape[1]
        S = np.full((analysis_ids.size, K), np.nan) if f is None else f.scores(M)
        return pl.DataFrame({f"{prefix}{k}": S[:, k] for k in range(K)})
    return block


def repeat_prediction(meta: pl.DataFrame, A: np.ndarray, positions: dict[int, str]) -> pl.DataFrame:
    """Athletes with ≥ 2 attempts: curve from attempt 1 alone (raw) vs pooled toward the position mean (components fitted
    WITHOUT the athlete's second attempt) as a predictor of attempt 2. Per-athlete MSE over shared observed bins."""
    m = meta.with_row_index("_r")
    first, second = {}, {}
    for (i,), g in m.partition_by(["nfl_id"], as_dict=True).items():
        r = g.sort("attempt_order")["_r"].to_list()
        if len(r) >= 2:
            first[int(i)], second[int(i)] = r[0], r[1]
    keep = m.filter(~pl.col("_r").is_in(list(second.values())))
    pooled, _ = hierarchical_curves(keep.drop("_r"), A[keep["_r"].to_numpy()], positions)
    rows = []
    for i in first:
        a1, a2 = A[first[i]], A[second[i]]
        o = np.isfinite(a1) & np.isfinite(a2) & np.isfinite(pooled.get(i, np.full(A.shape[1], np.nan)))
        if o.sum() < 3:
            continue
        rows.append({"nfl_id": i, "position_group": positions.get(i, "?"), "mse_raw": float(np.mean((a1[o] - a2[o]) ** 2)),
                     "mse_pooled": float(np.mean((pooled[i][o] - a2[o]) ** 2)), "n_bins": int(o.sum())})
    return pl.DataFrame(rows)


# ============================================================================== elastic analysis of speed–time curves (exploratory)
def speed_time_matrix(st: pl.DataFrame, window_s: float, c: M12Config) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """(meta, V [attempts × grid], grid) on [0, window_s] from onset; attempts not covering the window are excluded."""
    g = ae.time_grid(c)
    g = g[g <= window_s + 1e-9]
    d = st.filter(pl.col("t_rel") <= window_s + 1e-9)
    cov = d.group_by("seq_id").agg(pl.col("t_rel").max().alias("_t"), pl.len().alias("_n")).filter((pl.col("_t") >= window_s - 1e-6) & (pl.col("_n") == g.size))
    d = d.join(cov.select("seq_id"), on="seq_id", how="semi").sort("seq_id", "t_rel")
    meta = d.group_by("seq_id", maintain_order=True).agg(pl.col("nfl_id").first(), pl.col("attempt_order").first())
    V = d["speed"].to_numpy().reshape(meta.height, g.size) if meta.height else np.zeros((0, g.size))
    return meta, V, g


def landmarks(V: np.ndarray, g: np.ndarray, levels: Sequence[float] = (4.0, 6.0, 8.0)) -> np.ndarray:
    """First-crossing times of each level (NaN when not reached in the window)."""
    out = np.full((V.shape[0], len(levels)), np.nan)
    for i in range(V.shape[0]):
        for j, L in enumerate(levels):
            k = np.flatnonzero(V[i] >= L)
            if k.size and k[0] > 0:
                k0 = k[0]
                out[i, j] = g[k0 - 1] + (L - V[i, k0 - 1]) / max(V[i, k0] - V[i, k0 - 1], 1e-9) * (g[k0] - g[k0 - 1])
    return out


def landmark_register(V: np.ndarray, g: np.ndarray, L: np.ndarray, Lref: np.ndarray) -> np.ndarray:
    """Piecewise-linear monotone warps h_i with h_i(Lref_j) = L_ij (and h(0) = 0, h(end) = end); registered v_i(h_i(t))."""
    out = np.full_like(V, np.nan)
    T = g[-1]
    for i in range(V.shape[0]):
        ok = np.isfinite(L[i]) & np.isfinite(Lref)
        xs = np.r_[0.0, Lref[ok], T]
        ys = np.r_[0.0, L[i][ok], T]
        if np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0):
            continue
        out[i] = np.interp(np.interp(g, xs, ys), g, V[i])
    return out


def srsf(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    d = np.gradient(v, g)
    return np.sign(d) * np.sqrt(np.abs(d))


_STEPS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))


def dp_align(q1: np.ndarray, q2: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Warp γ (values on g) minimising ‖q1 − (q2∘γ)√γ'‖² by dynamic programming over a restricted slope set (monotone,
    endpoint-preserving). Returns γ evaluated on the grid."""
    n = g.size
    E = np.full((n, n), np.inf)
    E[0, 0] = 0.0
    P = np.zeros((n, n, 2), int)
    for i in range(1, n):
        for j in range(1, n):
            for di, dj in _STEPS:
                a, b = i - di, j - dj
                if a < 0 or b < 0 or not np.isfinite(E[a, b]):
                    continue
                slope = (g[j] - g[b]) / (g[i] - g[a])
                ii = np.arange(a, i + 1)
                jj = g[b] + slope * (g[ii] - g[a])
                cost = float(np.sum((q1[ii] - np.interp(jj, g, q2) * np.sqrt(slope)) ** 2) * (g[1] - g[0]))
                if E[a, b] + cost < E[i, j]:
                    E[i, j], P[i, j] = E[a, b] + cost, (a, b)
    path = [(n - 1, n - 1)]
    while path[-1] != (0, 0):
        path.append(tuple(P[path[-1]]))
    path = path[::-1]
    xs, ys = np.array([g[p[0]] for p in path]), np.array([g[p[1]] for p in path])
    return np.interp(g, xs, ys)


def elastic_align(V: np.ndarray, g: np.ndarray, iters: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SRSF alignment to an iterated mean template (simplified Karcher mean). Returns (aligned curves, warps, template SRSF).
    Phase distance d_p = ‖√γ' − 1‖, amplitude distance d_a = ‖q_aligned − template‖ (computed by ``phase_amplitude``)."""
    Q = np.array([srsf(v, g) for v in V])
    tmpl = Q.mean(0)
    G = np.tile(g, (V.shape[0], 1))
    for _ in range(iters):
        for i in range(V.shape[0]):
            G[i] = dp_align(tmpl, Q[i], g)
        Qa = np.array([np.interp(G[i], g, Q[i]) * np.sqrt(np.maximum(np.gradient(G[i], g), 0)) for i in range(V.shape[0])])
        tmpl = Qa.mean(0)
    Va = np.array([np.interp(G[i], g, V[i]) for i in range(V.shape[0])])
    return Va, G, tmpl


def phase_amplitude(V: np.ndarray, Va: np.ndarray, G: np.ndarray, tmpl: np.ndarray, g: np.ndarray) -> pl.DataFrame:
    dt = g[1] - g[0]
    rows = []
    for i in range(V.shape[0]):
        gd = np.maximum(np.gradient(G[i], g), 0)
        qa = srsf(Va[i], g)
        rows.append({"phase_distance": float(np.sqrt(np.sum((np.sqrt(gd) - 1) ** 2) * dt)), "amplitude_distance": float(np.sqrt(np.sum((qa - tmpl) ** 2) * dt)),
                     "warp_mid_shift_s": float(G[i][g.size // 2] - g[g.size // 2])})
    return pl.DataFrame(rows)


def pair_decomposition(Va: np.ndarray, G: np.ndarray, g: np.ndarray, a: int, b: int) -> dict[str, float]:
    """Two curves (e.g. a stopwatch-twin pair's mean curves): phase difference = ‖γ_a − γ_b‖, amplitude difference =
    ‖v_a∘γ_a − v_b∘γ_b‖, and the phase share of the raw L2 distance (descriptive)."""
    dt = g[1] - g[0]
    ph_ = float(np.sqrt(np.sum((G[a] - G[b]) ** 2) * dt))
    am = float(np.sqrt(np.sum((Va[a] - Va[b]) ** 2) * dt))
    return {"phase_diff_s": ph_, "amplitude_diff": am}


__all__ = ["FPCA", "FunctionalResult", "athlete_mse", "choose_lambda_repeat", "combine_athlete_matrix", "combine_attempt_matrix", "compare_functional",
           "dp_align", "elastic_align", "fof_surface", "fpca_block", "functional_oof", "hierarchical_apply", "hierarchical_block", "hierarchical_curves",
           "landmark_register", "landmarks",
           "pair_decomposition", "phase_amplitude", "pointwise_effect", "repeat_prediction", "speed_time_matrix", "spline_basis_for", "spline_block",
           "spline_coefs", "srsf"]
