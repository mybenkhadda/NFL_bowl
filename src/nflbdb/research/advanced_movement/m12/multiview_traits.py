"""12D — multi-view latent athletic traits. NumPy / Polars.

Views = tracked Combine drill FAMILIES (Milestone 11 drill mapping; ``forty``, ``short_shuttle``, ``three_cone``,
``long_shuttle``, ``backpedal_transition``, ``pass_rush``, ``gauntlet``) with ≤ 4 physics-preselected features each
(``m11.combine_drills.PRESELECTED`` + forty speed features). Official stopwatch tests are a SEPARATE block (D0), never a view.

Model (factor analysis with missing views; EM, no imputation):
    x_i = μ + W f_i + e_i,  f_i ~ N(0, I_K),  e_i ~ N(0, Ψ) diagonal; only the athlete's OBSERVED features enter its
    E-step and the likelihood. A structural mask on W gives:
        D2  probabilistic FA         all loadings free, K factors
        D3  multi-view FA            K_s shared factors (free) + one view-specific factor per view (nonzero on that view only)
        D4  hierarchical multi-view  D3 on features centred within position (position means from training athletes)
    D1 = concatenated standardized features in ridge (static); D0 = officials only. D5 (nonlinear variational multi-view
    model) runs only when complete-view athletes ≥ ``vae_min_athletes`` — a declared rule, evaluated, not assumed.

Identification: sign convention (largest |loading| positive), varimax among shared factors for display, Procrustes-aligned
bootstrap refits (Tucker congruence), position-specific refits, K by held-out (athlete-fold) log-likelihood.
Missing drills are NOT missing at random (participation follows position and invitation); every claim is conditional
on participation and is checked on complete cases. A factor score is never imputed ability for a drill not performed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import polars as pl

from ..m11.combine_drills import PRESELECTED
from . import advanced_evaluation as ae
from .config import M12Config

NAN = float("nan")
VIEW_FAMILIES = ("forty", "short_shuttle", "three_cone", "long_shuttle", "backpedal_transition", "pass_rush", "gauntlet")
EXTRA = {"forty": ("forty.v_max_obs", "forty.band_2_4_mean_accel")}


def view_features(available: Sequence[str]) -> dict[str, list[str]]:
    av = set(available)
    out = {}
    for v in VIEW_FAMILIES:
        fs = [f for f in (*PRESELECTED.get(v, ()), *EXTRA.get(v, ())) if f in av]
        if fs:
            out[v] = list(dict.fromkeys(fs))
    return out


# ============================================================================== data
def athlete_matrix(athf: pl.DataFrame, views: dict[str, list[str]], ids: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """(ids, X [athletes × features] athlete means with NaN where the drill was not tracked, feature names, view of each)."""
    feats = [f for v in views.values() for f in v]
    w = athf.filter(pl.col("feature").is_in(feats)).pivot(on="feature", index="nfl_id", values="value")
    if ids is not None:
        w = pl.DataFrame({"nfl_id": sorted(set(int(i) for i in ids))}).join(w, on="nfl_id", how="left")
    w = w.sort("nfl_id")
    X = np.column_stack([w[f].cast(pl.Float64).fill_null(np.nan).to_numpy() if f in w.columns else np.full(w.height, np.nan) for f in feats])
    return w["nfl_id"].to_numpy(), X, feats, [v for v, fs in views.items() for _ in fs]


def attempt_split_matrices(attempts: pl.DataFrame, ids: np.ndarray, feats: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """X1 / X2: the athlete's first and second attempt value per feature (by attempt order within the drill). NaN when the
    athlete has fewer than two attempts of that drill."""
    a = attempts.filter(pl.col("feature").is_in(list(feats)) & pl.col("value").is_not_null() & pl.col("value").is_finite())
    a = a.with_columns(pl.col("attempt_order").rank("ordinal").over("nfl_id", "feature").alias("_k"))
    out = []
    for k in (1, 2):
        w = a.filter(pl.col("_k") == k).pivot(on="feature", index="nfl_id", values="value")
        row = {int(i): r for r, i in enumerate(w["nfl_id"].to_list())}
        idx = np.array([row.get(int(i), -1) for i in ids])
        cols = []
        for f in feats:
            v = w[f].cast(pl.Float64).fill_null(np.nan).to_numpy() if f in w.columns else np.full(w.height, np.nan)
            cols.append(np.where(idx >= 0, v[np.maximum(idx, 0)] if v.size else np.nan, np.nan))
        out.append(np.column_stack(cols))
    X1, X2 = out
    both = np.isfinite(X1) & np.isfinite(X2)
    return np.where(both, X1, np.nan), np.where(both, X2, np.nan)


def availability(ids: np.ndarray, X: np.ndarray, views_of: Sequence[str], positions: dict[int, str]) -> pl.DataFrame:
    vs = list(dict.fromkeys(views_of))
    rows = []
    for k, i in enumerate(ids):
        obs = {v: bool(np.isfinite(X[k, [j for j, vv in enumerate(views_of) if vv == v]]).any()) for v in vs}
        rows.append({"nfl_id": int(i), "position_group": positions.get(int(i), "?"), **obs, "n_views": sum(obs.values())})
    return pl.DataFrame(rows)


def missingness_diagnostics(avail: pl.DataFrame, officials: pl.DataFrame, views: Sequence[str]) -> pl.DataFrame:
    """Within position: participation rate per view and standardized mean differences (participants − non-participants) of
    official forty, weight and height. |SMD| ≥ 0.2 is evidence that participation is related to measured athleticism —
    i.e. the views are NOT missing completely at random. Descriptive."""
    d = avail.join(officials.select("nfl_id", "forty", "weight", "height"), on="nfl_id", how="left")
    rows = []
    for (p,), g in sorted(d.partition_by(["position_group"], as_dict=True).items()):
        for v in views:
            if v not in g.columns:
                continue
            row = {"position_group": p, "view": v, "n": g.height, "participation": float(g[v].mean())}
            for k in ("forty", "weight", "height"):
                a, b = g.filter(pl.col(v))[k].drop_nulls().to_numpy(), g.filter(~pl.col(v))[k].drop_nulls().to_numpy()
                sd = math.sqrt((a.var() + b.var()) / 2) if a.size > 2 and b.size > 2 else NAN
                row[f"smd_{k}"] = float((a.mean() - b.mean()) / sd) if np.isfinite(sd) and sd > 0 else NAN
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


# ============================================================================== missing-data factor analysis
def structure_mask(views_of: Sequence[str], k_shared: int, view_specific: bool) -> tuple[np.ndarray, list[str]]:
    vs = list(dict.fromkeys(views_of))
    cols = [f"shared{k}" for k in range(k_shared)] + ([f"view:{v}" for v in vs] if view_specific else [])
    mask = np.zeros((len(views_of), len(cols)), bool)
    mask[:, :k_shared] = True
    if view_specific:
        for j, v in enumerate(views_of):
            mask[j, k_shared + vs.index(v)] = True
    return mask, cols


@dataclass
class MissingFA:
    mask: np.ndarray
    names: list[str]
    mu: np.ndarray = field(default=None)
    W: np.ndarray = field(default=None)
    psi: np.ndarray = field(default=None)
    center: np.ndarray = field(default=None)
    scale: np.ndarray = field(default=None)
    loglik_trace: list[float] = field(default_factory=list)

    def _std(self, X: np.ndarray) -> np.ndarray:
        return (X - self.center) / self.scale

    def fit(self, X: np.ndarray, iters: int = 300, seed: int = 0, tol: float = 1e-6) -> "MissingFA":
        """EM on the observed entries. Standardisation constants from THIS (training) matrix."""
        self.center = np.nanmean(X, 0)
        sd = np.nanstd(X, 0)
        self.scale = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        Z = self._std(X)
        n, p = Z.shape
        K = self.mask.shape[1]
        rng = np.random.default_rng(seed)
        self.mu = np.zeros(p)
        self.W = np.where(self.mask, 0.3 + 0.1 * rng.standard_normal((p, K)), 0.0)
        self.psi = np.full(p, 0.5)
        O = np.isfinite(Z)
        prev = -np.inf
        for it in range(iters):
            Ef, Eff = np.zeros((n, K)), np.zeros((n, K, K))
            ll = 0.0
            for i in range(n):
                o = O[i]
                if not o.any():
                    Eff[i] = np.eye(K)
                    continue
                Wo, po = self.W[o], self.psi[o]
                Minv = np.linalg.inv(np.eye(K) + (Wo.T / po) @ Wo)
                r = Z[i, o] - self.mu[o]
                Ef[i] = Minv @ (Wo.T / po) @ r
                Eff[i] = Minv + np.outer(Ef[i], Ef[i])
                C = Wo @ Wo.T + np.diag(po)
                sign, logdet = np.linalg.slogdet(C)
                ll += -0.5 * (logdet + r @ np.linalg.solve(C, r) + o.sum() * math.log(2 * math.pi))
            self.loglik_trace.append(ll)
            for j in range(p):
                rows = O[:, j]
                if rows.sum() < 3:
                    continue
                cols = self.mask[j]
                S = Eff[rows][:, cols][:, :, cols].sum(0)
                b = ((Z[rows, j] - self.mu[j])[:, None] * Ef[rows][:, cols]).sum(0)
                self.W[j, cols] = np.linalg.solve(S + 1e-6 * np.eye(cols.sum()), b)
                self.mu[j] = float(np.mean(Z[rows, j] - Ef[rows] @ self.W[j]))
                r = Z[rows, j] - self.mu[j]
                quad = np.einsum("k,ikl,l->i", self.W[j], Eff[rows], self.W[j])
                self.psi[j] = max(float(np.mean(r ** 2 - 2 * r * (Ef[rows] @ self.W[j]) + quad)), 1e-3)
            if abs(ll - prev) < tol * max(1.0, abs(prev)):
                break
            prev = ll
        self._sign()
        return self

    def _sign(self) -> None:
        for k in range(self.W.shape[1]):
            j = int(np.argmax(np.abs(self.W[:, k])))
            if self.W[j, k] < 0:
                self.W[:, k] *= -1

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Posterior mean factor scores from the OBSERVED features of each row (NaN row when nothing is observed)."""
        Z = self._std(X)
        K = self.W.shape[1]
        out = np.full((Z.shape[0], K), np.nan)
        for i in range(Z.shape[0]):
            o = np.isfinite(Z[i])
            if not o.any():
                continue
            Wo, po = self.W[o], self.psi[o]
            out[i] = np.linalg.solve(np.eye(K) + (Wo.T / po) @ Wo, (Wo.T / po) @ (Z[i, o] - self.mu[o]))
        return out

    def loglik(self, X: np.ndarray) -> np.ndarray:
        Z = self._std(X)
        out = np.full(Z.shape[0], np.nan)
        for i in range(Z.shape[0]):
            o = np.isfinite(Z[i])
            if not o.any():
                continue
            C = self.W[o] @ self.W[o].T + np.diag(self.psi[o])
            r = Z[i, o] - self.mu[o]
            out[i] = -0.5 * (np.linalg.slogdet(C)[1] + r @ np.linalg.solve(C, r) + o.sum() * math.log(2 * math.pi)) - math.log(float(np.prod(self.scale[o])))
        return out

    def shared_variance_share(self) -> np.ndarray:
        """Per feature: share of (standardized) variance from the shared columns, the view-specific columns and the unique part."""
        ks = np.array([n.startswith("shared") for n in self.names])
        sh, vw = (self.W[:, ks] ** 2).sum(1), (self.W[:, ~ks] ** 2).sum(1)
        tot = sh + vw + self.psi
        return np.c_[sh / tot, vw / tot, self.psi / tot]


def model_spec(model: str, k: int, views_of: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    if model == "D2":
        return np.ones((len(views_of), k), bool), [f"shared{j}" for j in range(k)]
    return structure_mask(views_of, k, view_specific=True)


def position_center(X: np.ndarray, pos: np.ndarray, train: np.ndarray) -> np.ndarray:
    """D4: subtract training-athlete position means per feature (pooled mean for positions unseen in training)."""
    out = X.copy()
    pooled = np.nanmean(X[train], 0)
    for p in np.unique(pos):
        m = pos == p
        ref = X[train & m]
        mu = np.nanmean(ref, 0) if ref.shape[0] >= 5 else pooled
        mu = np.where(np.isfinite(mu), mu, pooled)
        out[m] = X[m] - mu
    return out


def fit_model(model: str, k: int, X: np.ndarray, views_of: Sequence[str], pos: np.ndarray, train: np.ndarray, c: M12Config, seed: int) -> tuple[MissingFA, np.ndarray]:
    """Fit on the training rows; returns (model, the matrix in the model's input space for EVERY row)."""
    Xin = position_center(X, pos, train) if model == "D4" else X
    mask, names = model_spec(model, k, views_of)
    return MissingFA(mask, names).fit(Xin[train], c.fa_iters, seed), Xin


def factor_block(model: str, k: int, ids: np.ndarray, X: np.ndarray, views_of: Sequence[str], pos_of: dict[int, str], analysis_ids: np.ndarray,
                 c: M12Config, seed: int, shared_only: bool = True) -> ae.BlockFn:
    """Fold-fitted factor scores for the analysis rows (ladder block). The model sees only training athletes' Combine
    features (unsupervised; no NFL value enters)."""
    row = {int(i): r for r, i in enumerate(ids)}
    idx = np.array([row.get(int(i), -1) for i in analysis_ids])
    pos = np.array([pos_of.get(int(i), "?") for i in ids])

    def block(train: np.ndarray) -> pl.DataFrame:
        tr = np.zeros(ids.size, bool)
        tr[idx[train & (idx >= 0)]] = True
        m, Xin = fit_model(model, k, X, views_of, pos, tr, c, seed)
        S = m.scores(Xin)
        keep = [j for j, n in enumerate(m.names) if (n.startswith("shared") or not shared_only)]
        out = np.full((idx.size, len(keep)), np.nan)
        out[idx >= 0] = S[idx[idx >= 0]][:, keep]
        return pl.DataFrame({f"f{j}": out[:, q] for q, j in enumerate(keep)})
    return block


# ============================================================================== dimensionality, stability, reliability
def heldout_loglik(model: str, ks: Sequence[int], X: np.ndarray, views_of: Sequence[str], pos: np.ndarray, folds: np.ndarray, c: M12Config, seed: int) -> pl.DataFrame:
    rows = []
    for k in ks:
        ll = np.full(X.shape[0], np.nan)
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            m, Xin = fit_model(model, k, X, views_of, pos, tr, c, seed)
            ll[te] = m.loglik(Xin[te])
        nobs = np.isfinite(X).sum(1)
        ok = np.isfinite(ll) & (nobs > 0)
        rows.append({"model": model, "k": k, "n_athletes": int(ok.sum()), "heldout_loglik_per_value": float(ll[ok].sum() / nobs[ok].sum()) if ok.any() else NAN})
    return pl.DataFrame(rows)


def varimax(L: np.ndarray, iters: int = 100, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(iters):
        Lr = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p))
        R = u @ vt
        if s.sum() < d * (1 + tol):
            break
        d = s.sum()
    return L @ R, R


def procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal rotation of B that best matches A (columns)."""
    u, _, vt = np.linalg.svd(B.T @ A)
    return B @ (u @ vt)


def tucker(a: np.ndarray, b: np.ndarray) -> float:
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else NAN


def bootstrap_stability(model: str, k: int, X: np.ndarray, views_of: Sequence[str], pos: np.ndarray, c: M12Config, seed: int, B: int | None = None) -> pl.DataFrame:
    """Athlete bootstrap refits; shared loadings Procrustes-aligned to the full-data fit; Tucker congruence per factor."""
    ref, _ = fit_model(model, k, X, views_of, pos, np.ones(X.shape[0], bool), c, seed)
    ks = [j for j, n in enumerate(ref.names) if n.startswith("shared")]
    A = ref.W[:, ks]
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(c.fa_boot if B is None else B):
        ix = rng.integers(0, X.shape[0], X.shape[0])
        m = MissingFA(ref.mask, ref.names).fit((position_center(X, pos, np.ones(X.shape[0], bool)) if model == "D4" else X)[ix], max(60, c.fa_iters // 3), seed + b)
        Bm = procrustes(A, m.W[:, ks]) if len(ks) > 1 else m.W[:, ks] * np.sign(tucker(A[:, 0], m.W[:, ks][:, 0]) or 1)
        rows += [{"model": model, "k": k, "replicate": b, "factor": f"shared{q}", "tucker": tucker(A[:, q], Bm[:, q])} for q in range(len(ks))]
    return pl.DataFrame(rows)


def position_loadings(model: str, k: int, X: np.ndarray, views_of: Sequence[str], pos: np.ndarray, c: M12Config, seed: int, min_n: int = 60) -> pl.DataFrame:
    """Refit inside each position with ≥ ``min_n`` athletes; congruence of its shared loadings with the pooled fit
    (features the position never performs are excluded from that comparison)."""
    ref, _ = fit_model(model, k, X, views_of, pos, np.ones(X.shape[0], bool), c, seed)
    ks = [j for j, n in enumerate(ref.names) if n.startswith("shared")]
    rows = []
    for p in np.unique(pos):
        m = pos == p
        if m.sum() < min_n:
            continue
        obs = np.isfinite(X[m]).sum(0) >= 10
        if obs.sum() < 3:
            continue
        sub = MissingFA(ref.mask[obs], ref.names).fit(X[m][:, obs], c.fa_iters, seed)
        A, Bm = ref.W[obs][:, ks], sub.W[:, ks]
        Bm = procrustes(A, Bm) if len(ks) > 1 else Bm
        rows += [{"position_group": str(p), "factor": f"shared{q}", "n_athletes": int(m.sum()), "n_features": int(obs.sum()), "tucker": tucker(A[:, q], Bm[:, q])}
                 for q in range(len(ks))]
    return pl.DataFrame(rows)


def latent_test_retest(model: str, k: int, ids: np.ndarray, X: np.ndarray, X1: np.ndarray, X2: np.ndarray, views_of: Sequence[str], feats: Sequence[str],
                       pos: np.ndarray, folds: np.ndarray, c: M12Config, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cross-fitted: the model is fitted on the athlete means of OTHER folds; the held-out athletes' first-attempt and
    second-attempt matrices are scored separately; test–retest r of each shared factor vs the best single feature among
    those loading |λ| ≥ 0.4 on it (same athletes), paired athlete bootstrap of the difference. Also per-feature retest r."""
    S1, S2 = np.full((ids.size, k), np.nan), np.full((ids.size, k), np.nan)
    Wref = None
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m, _ = fit_model(model, k, X, views_of, pos, tr, c, seed)
        ks = [j for j, n in enumerate(m.names) if n.startswith("shared")]
        A1 = position_center(X1, pos, tr) if model == "D4" else X1
        A2 = position_center(X2, pos, tr) if model == "D4" else X2
        if Wref is None:
            Wref = m.W[:, ks]
            R = np.eye(len(ks))
        else:
            u, _, vt = np.linalg.svd(m.W[:, ks].T @ Wref)
            R = u @ vt
        S1[te] = m.scores(A1[te])[:, ks] @ R
        S2[te] = m.scores(A2[te])[:, ks] @ R
    feat_rows = []
    for j, fname in enumerate(feats):
        a, b = X1[:, j], X2[:, j]
        mm = np.isfinite(a) & np.isfinite(b)
        feat_rows.append({"feature": fname, "view": views_of[j], "n_athletes": int(mm.sum()), "retest_r": float(np.corrcoef(a[mm], b[mm])[0, 1]) if mm.sum() >= 10 else NAN})
    ft = pl.DataFrame(feat_rows)
    rows = []
    rng = np.random.default_rng(seed)
    for q in range(k):
        a, b = S1[:, q], S2[:, q]
        lo_feats = [j for j in range(len(feats)) if Wref is not None and abs(Wref[j, q]) >= 0.4]
        best_j, best_r = None, -np.inf
        for j in lo_feats:
            r = ft["retest_r"][j]
            if r is not None and np.isfinite(r) and r > best_r:
                best_j, best_r = j, r
        mm = np.isfinite(a) & np.isfinite(b)
        if best_j is not None:
            mm &= np.isfinite(X1[:, best_j]) & np.isfinite(X2[:, best_j])
        n = int(mm.sum())
        row = {"model": model, "k": k, "factor": f"shared{q}", "n_athletes": n, "retest_factor": NAN, "best_feature": feats[best_j] if best_j is not None else None,
               "retest_best_feature": NAN, "gain": NAN, "gain_lo": NAN, "gain_hi": NAN}
        if n >= 15:
            rf = float(np.corrcoef(a[mm], b[mm])[0, 1])
            row["retest_factor"] = rf
            if best_j is not None:
                xa, xb = X1[mm, best_j], X2[mm, best_j]
                rb = float(np.corrcoef(xa, xb)[0, 1])
                row["retest_best_feature"], row["gain"] = rb, rf - rb
                bs = []
                for _ in range(max(200, c.n_boot // 5)):
                    ix = rng.integers(0, n, n)
                    bs.append(np.corrcoef(a[mm][ix], b[mm][ix])[0, 1] - np.corrcoef(xa[ix], xb[ix])[0, 1])
                bs = np.asarray(bs)
                bs = bs[np.isfinite(bs)]
                if bs.size >= 20:
                    row["gain_lo"], row["gain_hi"] = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None), ft


def selection_rule(retest: pl.DataFrame, ll: pl.DataFrame) -> dict[str, Any]:
    """Declared rule for the representation carried to 12C (M6) and 12G: K per model = best held-out log-likelihood;
    model = highest mean factor test–retest r at its K (ties → simpler D2 < D3 < D4). No NFL value is used."""
    best_k = {m: int(g.sort("heldout_loglik_per_value", descending=True)["k"][0]) for (m,), g in ll.partition_by(["model"], as_dict=True).items()
              if g["heldout_loglik_per_value"].is_finite().any()}
    cand = []
    for m, k in best_k.items():
        r = retest.filter((pl.col("model") == m) & (pl.col("k") == k)) if "k" in retest.columns else retest.filter(pl.col("model") == m)
        v = r["retest_factor"].drop_nulls()
        cand.append((m, k, float(v.mean()) if v.len() else -9.0))
    order = {"D2": 0, "D3": 1, "D4": 2}
    cand.sort(key=lambda t: (-t[2], order.get(t[0], 9)))
    return {"model": cand[0][0], "k": cand[0][1], "mean_retest": cand[0][2], "candidates": cand} if cand else {"model": None}


def vae_feasible(avail: pl.DataFrame, c: M12Config) -> dict[str, Any]:
    vs = [v for v in VIEW_FAMILIES if v in avail.columns]
    complete = int(avail.filter(pl.all_horizontal([pl.col(v) for v in vs])).height) if vs else 0
    return {"complete_view_athletes": complete, "threshold": c.vae_min_athletes, "run": complete >= c.vae_min_athletes,
            "reason": "data support a nonlinear multi-view model" if complete >= c.vae_min_athletes else
                      f"only {complete} athletes have every view (< {c.vae_min_athletes}): D5 is not run — a declared rule, not a result"}


__all__ = ["MissingFA", "VIEW_FAMILIES", "athlete_matrix", "attempt_split_matrices", "availability", "bootstrap_stability", "factor_block", "fit_model",
           "heldout_loglik", "latent_test_retest", "missingness_diagnostics", "model_spec", "position_center", "position_loadings", "procrustes",
           "selection_rule", "structure_mask", "tucker", "vae_feasible", "varimax", "view_features"]
