"""Regularised regression, athlete-grouped cross-validation and paired evaluation (NumPy only).

Unit            one row = one athlete. A fold therefore never splits an athlete, and no play-level row exists here.
Preprocessing   fitted on the training athletes of each fold only: median imputation, standardisation, dummy
                levels. A level unseen in training gets all-zero dummies (the reference level).
Ridge           all non-intercept columns penalised with one λ. λ is chosen INSIDE the training fold by exact
                leave-one-out error (hat-matrix identity on the SVD), so tuning never sees the test athletes.
                The inner step reuses the training-fold centring and scaling, as scikit-learn's RidgeCV does.
Huber           iteratively reweighted ridge at the λ chosen for the squared loss (sensitivity only).
Target          a :class:`Target` returns the outcome vector GIVEN the training athletes, because the
                context-adjusted target removes alignment × season means that must not be estimated on test athletes.
Metrics         computed per repeat on out-of-fold predictions, then averaged over repeats.
                R²_oos = 1 − Σ(y − ŷ)² / Σ(y − ȳ_train)², i.e. skill against the training-fold mean.
Uncertainty     athlete bootstrap of the out-of-fold table (models are not refitted). It reflects which athletes
                were sampled, not the variability of the fitting procedure; differences between models are paired
                because every model is scored on the same resampled athletes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..studies.stats import rankdata

NAN = float("nan")
METRICS = ("mae", "rmse", "r2_oos", "pearson", "calibration_slope", "calibration_intercept")
LOWER_IS_BETTER = {"mae": True, "rmse": True, "r2_oos": False, "pearson": False}


# ============================================================================== target
@dataclass
class Target:
    """Outcome of n athletes. ``fixed``: values that do not depend on the training set. Otherwise ``S`` and ``K``
    (n × cells) hold each athlete's sum and count of the endpoint per context cell."""
    fixed: np.ndarray | None = None
    S: np.ndarray | None = None
    K: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.fixed.size if self.fixed is not None else self.K.shape[0])

    def take(self, idx: np.ndarray) -> "Target":
        if self.fixed is not None:
            return Target(fixed=self.fixed[idx])
        return Target(S=self.S[idx], K=self.K[idx])

    def y(self, train: np.ndarray | None = None) -> np.ndarray:
        """Outcome of every athlete; context means come from the athletes in ``train`` (all athletes when None)."""
        if self.fixed is not None:
            return self.fixed
        S, K = (self.S, self.K) if train is None else (self.S[train], self.K[train])
        tot = K.sum()
        grand = S.sum() / tot if tot > 0 else NAN
        with np.errstate(divide="ignore", invalid="ignore"):
            cell = np.where(K.sum(0) > 0, S.sum(0) / K.sum(0), grand)       # unseen cell → grand training mean
            return (self.S - self.K * cell).sum(1) / self.K.sum(1) + grand


# ============================================================================== preprocessing
@dataclass
class Prep:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    median: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    levels: tuple[np.ndarray, ...]

    @property
    def terms(self) -> list[str]:
        return [*self.numeric, *[f"{c}={v}" for c, lev in zip(self.categorical, self.levels) for v in lev[1:]]]


def fit_prep(data: dict[str, np.ndarray], numeric: Sequence[str], categorical: Sequence[str], rows: np.ndarray) -> Prep:
    X = np.column_stack([data[c][rows] for c in numeric]) if numeric else np.empty((int(rows.sum()), 0))
    with np.errstate(all="ignore"):
        med = np.nanmedian(X, axis=0) if X.shape[1] else np.empty(0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med)
    sd = Xi.std(0) if X.shape[1] else np.empty(0)
    return Prep(tuple(numeric), tuple(categorical), med, Xi.mean(0) if X.shape[1] else np.empty(0),
                np.where(sd > 1e-12, sd, 1.0), tuple(np.unique(data[c][rows].astype(str)) for c in categorical))


def transform(data: dict[str, np.ndarray], prep: Prep, rows: np.ndarray) -> np.ndarray:
    n = int(rows.sum()) if rows.dtype == bool else int(rows.size)
    cols = []
    if prep.numeric:
        X = np.column_stack([data[c][rows] for c in prep.numeric])
        cols.append((np.where(np.isfinite(X), X, prep.median) - prep.mean) / prep.sd)
    for c, lev in zip(prep.categorical, prep.levels):
        v = data[c][rows].astype(str)
        cols += [(v == level).astype(float)[:, None] for level in lev[1:]]
    return np.column_stack(cols) if cols else np.empty((n, 0))


# ============================================================================== ridge
def lambda_grid(spec: tuple[float, float, int]) -> np.ndarray:
    return np.logspace(spec[0], spec[1], int(spec[2]))


def ridge_loo(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> dict[str, Any]:
    """Ridge with an unpenalised intercept; λ minimises the exact leave-one-out squared error.

    Returns intercept, beta (for X as given), the chosen λ and the LOO mean squared error per λ."""
    n, p = X.shape
    ybar = float(y.mean())
    if p == 0 or n < 3:
        return {"intercept": ybar, "beta": np.zeros(p), "lambda": NAN, "loo_mse": np.full(lambdas.size, NAN)}
    xbar = X.mean(0)
    Xc, yc = X - xbar, y - ybar
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uty = U.T @ yc
    d = s[None, :] ** 2 / (s[None, :] ** 2 + lambdas[:, None])                 # (L, r) shrinkage factors
    fitted = (d * Uty[None, :]) @ U.T                                          # (L, n)
    h = d @ (U ** 2).T + 1.0 / n                                               # (L, n) leverages incl. intercept
    loo = (yc[None, :] - fitted) / np.clip(1.0 - h, 1e-8, None)
    mse = (loo ** 2).mean(1)
    best = int(np.nanargmin(mse))
    lam = float(lambdas[best])
    beta = Vt.T @ (s / (s ** 2 + lam) * Uty)
    return {"intercept": ybar - float(xbar @ beta), "beta": beta, "lambda": lam, "loo_mse": mse}


def ridge_fixed(X: np.ndarray, y: np.ndarray, lam: float, w: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """(Weighted) ridge at a given λ, intercept unpenalised."""
    n, p = X.shape
    w = np.ones(n) if w is None else w
    sw = w.sum()
    ybar, xbar = float((w * y).sum() / sw), (w[:, None] * X).sum(0) / sw
    if p == 0 or not np.isfinite(lam):
        return ybar, np.zeros(p)
    Xc, yc = X - xbar, y - ybar
    A = Xc.T @ (w[:, None] * Xc) + lam * np.eye(p)
    beta = np.linalg.solve(A, Xc.T @ (w * yc))
    return ybar - float(xbar @ beta), beta


def huber_ridge(X: np.ndarray, y: np.ndarray, lam: float, k: float = 1.345, n_iter: int = 30) -> tuple[float, np.ndarray]:
    """Huber M-estimate with a ridge penalty (IRLS; scale = normalised MAD of the residuals, re-estimated)."""
    b0, beta = ridge_fixed(X, y, lam)
    for _ in range(n_iter):
        r = y - b0 - X @ beta
        scale = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        if scale <= 1e-12:
            break
        w = np.minimum(1.0, k * scale / np.maximum(np.abs(r), 1e-12))
        nb0, nbeta = ridge_fixed(X, y, lam, w)
        done = abs(nb0 - b0) + float(np.abs(nbeta - beta).sum()) < 1e-9
        b0, beta = nb0, nbeta
        if done:
            break
    return b0, beta


def fit_predict(data: dict[str, np.ndarray], numeric: Sequence[str], categorical: Sequence[str], y: np.ndarray,
                train: np.ndarray, test: np.ndarray, lambdas: np.ndarray, robust: bool = False, huber_k: float = 1.345
                ) -> tuple[np.ndarray, float]:
    """Predictions for ``test`` from a model fitted on ``train`` (boolean masks). Returns (predictions, λ)."""
    if not numeric and not categorical:
        return np.full(int(test.sum()), float(y[train].mean())), NAN
    prep = fit_prep(data, numeric, categorical, train)
    Xtr, Xte = transform(data, prep, train), transform(data, prep, test)
    fit = ridge_loo(Xtr, y[train], lambdas)
    b0, beta = (huber_ridge(Xtr, y[train], fit["lambda"], huber_k) if robust and np.isfinite(fit["lambda"])
                else (fit["intercept"], fit["beta"]))
    return b0 + Xte @ beta, fit["lambda"]


# ============================================================================== folds
def make_folds(strata: np.ndarray, k: int, repeats: int, seed: int) -> np.ndarray:
    """(repeats, n) fold index 0..k−1; athletes are dealt to folds within each stratum (draft cohort), so every
    fold holds every cohort in proportion."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata).astype(str)
    out = np.empty((repeats, strata.size), dtype=int)
    for r in range(repeats):
        offset = 0
        for s in np.unique(strata):
            idx = rng.permutation(np.flatnonzero(strata == s))
            out[r, idx] = (np.arange(idx.size) + offset) % k
            offset += idx.size
    return out


def group_folds(groups: np.ndarray) -> np.ndarray:
    """(1, n) fold index = group index (leave one group out)."""
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    return inv[None, :].astype(int)


# ============================================================================== cross-validation
def cross_validate(data: dict[str, np.ndarray], target: Target, models: dict[str, tuple[Sequence[str], Sequence[str]]],
                   folds: np.ndarray, lambdas: np.ndarray, robust: bool = False, huber_k: float = 1.345
                   ) -> dict[str, Any]:
    """Out-of-fold predictions of every model on identical folds.

    ``models``: name → (numeric columns, categorical columns). Returns ``y`` (R, n) — the outcome each athlete was
    scored against (fold-specific when the target is context-adjusted) —, ``y_ref`` (R, n) the training-fold mean,
    and per model ``pred`` (R, n) and ``lam`` (R, n) (the λ of the fold that predicted the athlete)."""
    R, n = folds.shape
    y_obs, y_ref = np.full((R, n), NAN), np.full((R, n), NAN)
    pred = {m: np.full((R, n), NAN) for m in models}
    lam = {m: np.full((R, n), NAN) for m in models}
    for r in range(R):
        for k in np.unique(folds[r]):
            test = folds[r] == k
            train = ~test
            if train.sum() < 3:
                continue
            y = target.y(train)
            y_obs[r, test], y_ref[r, test] = y[test], float(y[train].mean())
            for m, (num, cat) in models.items():
                pred[m][r, test], lam[m][r, test] = fit_predict(data, num, cat, y, train, test, lambdas, robust, huber_k)
    return {"y": y_obs, "y_ref": y_ref, "pred": pred, "lam": lam}


# ============================================================================== metrics
def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    den = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-12, (a * b).sum(-1) / den, np.nan)


def metric_arrays(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics along the last axis (athletes); leading axes (repeat, bootstrap replicate) are kept."""
    e = y - p
    sse, sst = (e ** 2).sum(-1), ((y - ref) ** 2).sum(-1)
    pc = p - p.mean(-1, keepdims=True)
    varp = (pc ** 2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(varp > 1e-12, (pc * (y - y.mean(-1, keepdims=True))).sum(-1) / varp, np.nan)
        r2 = np.where(sst > 0, 1.0 - sse / sst, np.nan)
    return {"mae": np.abs(e).mean(-1), "rmse": np.sqrt((e ** 2).mean(-1)), "r2_oos": r2, "pearson": _rowwise_corr(y, p),
            "calibration_slope": slope, "calibration_intercept": y.mean(-1) - slope * p.mean(-1)}


def spearman_oof(y: np.ndarray, p: np.ndarray) -> float:
    vals = [_rowwise_corr(rankdata(y[r])[None, :], rankdata(p[r])[None, :])[0] for r in range(y.shape[0])]
    return float(np.nanmean(vals)) if vals else NAN


def _mean_over_repeats(a: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=0)


def _interval(reps: np.ndarray) -> tuple[float, float]:
    ok = reps[np.isfinite(reps)]
    return (float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))) if ok.size >= 20 else (NAN, NAN)


def evaluate(cv: dict[str, Any], reference: str, n_boot: int, seed: int, subset: np.ndarray | None = None
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-model metrics with athlete-bootstrap intervals, and paired differences against ``reference``.

    ``subset``: boolean mask of athletes to score (evaluation stratum of an otherwise unchanged fit).
    Difference = model − reference. ``improved`` = share of replicates in which the model beats the reference."""
    keep = np.ones(cv["y"].shape[1], bool) if subset is None else subset
    keep = keep & np.isfinite(cv["y"]).all(0)
    y, ref = cv["y"][:, keep], cv["y_ref"][:, keep]
    n = int(keep.sum())
    rows, diffs = [], []
    if n < 3:
        return rows, diffs
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n)) if n_boot > 0 else np.empty((0, n), int)
    point, boot = {}, {}
    for m, p_all in cv["pred"].items():
        p = p_all[:, keep]
        point[m] = {k: float(_mean_over_repeats(v)) for k, v in metric_arrays(y, p, ref).items()}
        point[m]["spearman"] = spearman_oof(y, p)
        boot[m] = ({k: _mean_over_repeats(v) for k, v in metric_arrays(y[:, idx], p[:, idx], ref[:, idx]).items()}
                   if n_boot > 0 else {})
        for k, est in point[m].items():
            lo, hi = _interval(boot[m][k]) if k in boot[m] else (NAN, NAN)
            rows.append({"model": m, "metric": k, "estimate": est, "ci_lower": lo, "ci_upper": hi, "n_athletes": n})
    if reference in point:
        for m in point:
            if m == reference:
                continue
            for k, lower_better in LOWER_IS_BETTER.items():
                d = point[m][k] - point[reference][k]
                reps = boot[m][k] - boot[reference][k] if k in boot[m] else np.empty(0)
                lo, hi = _interval(reps)
                ok = reps[np.isfinite(reps)]
                better = float(((ok < 0) if lower_better else (ok > 0)).mean()) if ok.size else NAN
                diffs.append({"model": m, "reference": reference, "metric": k, "estimate": d, "ci_lower": lo,
                              "ci_upper": hi, "share_boot_improved": better, "n_athletes": n})
    return rows, diffs


# ============================================================================== in-sample coefficients
def coefficient_table(data: dict[str, np.ndarray], numeric: Sequence[str], categorical: Sequence[str], y: np.ndarray,
                      lambdas: np.ndarray, n_boot: int, seed: int) -> list[dict[str, Any]]:
    """Standardised ridge coefficients on ALL athletes (λ by LOO) with an athlete-bootstrap percentile interval
    at that λ. In-sample and descriptive: it describes the fitted model, it is not a held-out result."""
    n = y.size
    rows_all = np.ones(n, bool)
    prep = fit_prep(data, numeric, categorical, rows_all)
    X = transform(data, prep, rows_all)
    fit = ridge_loo(X, y, lambdas)
    terms = prep.terms
    reps = np.full((n_boot, len(terms)), NAN)
    rng = np.random.default_rng(seed)
    for b in range(n_boot if np.isfinite(fit["lambda"]) else 0):
        i = rng.integers(0, n, n)
        sub = {c: v[i] for c, v in data.items()}
        pb = fit_prep(sub, numeric, categorical, rows_all)
        if pb.terms != terms:               # a cohort level missing from the resample → skip the replicate
            continue
        reps[b] = ridge_fixed(transform(sub, pb, rows_all), y[i], fit["lambda"])[1]
    out = []
    for j, term in enumerate(terms):
        lo, hi = _interval(reps[:, j])
        out.append({"term": term, "estimate_std": float(fit["beta"][j]), "ci_lower": lo, "ci_upper": hi,
                    "lambda": fit["lambda"], "n_athletes": int(n)})
    return out
