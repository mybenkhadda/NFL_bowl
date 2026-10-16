"""B-spline bases and penalised (P-spline) mean / quantile smoothers shared by 12A and 12C. NumPy only.

    bspline_basis(x, lo, hi, n_interior, degree)   Cox–de Boor recursion, clamped knots; rows sum to 1 inside [lo, hi]
    difference_penalty(k, order)                   DᵀD of order-``order`` differences (Eilers & Marx P-splines)
    pspline_fit(x, y, w, basis, lam)               weighted penalised least squares → coefficients
    quantile_pspline(x, y, w, tau, basis, lam)     penalised quantile regression by IRLS on the pinball loss
    rearrange(Q)                                   monotone rearrangement across τ (Chernozhukov et al. 2010) — removes
                                                   quantile crossing without changing the set of fitted values

The basis is evaluated ONLY where the caller says the data support it; nothing here extrapolates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

NAN = float("nan")


def knots(lo: float, hi: float, n_interior: int, degree: int = 3) -> np.ndarray:
    inner = np.linspace(lo, hi, n_interior + 2)
    return np.r_[np.full(degree, lo), inner, np.full(degree, hi)]


def bspline_basis(x: np.ndarray, lo: float, hi: float, n_interior: int, degree: int = 3) -> np.ndarray:
    """(len(x), n_interior + degree + 1) basis; x outside [lo, hi] gives a zero row (no extrapolation)."""
    x = np.asarray(x, float)
    t = knots(lo, hi, n_interior, degree)
    k = t.size - degree - 1
    B = np.zeros((x.size, t.size - 1))
    xc = np.where(x >= hi, hi - 1e-12, x)
    for j in range(t.size - 1):
        B[:, j] = (t[j] <= xc) & (xc < t[j + 1])
    for d in range(1, degree + 1):
        Bn = np.zeros((x.size, t.size - 1 - d))
        for j in range(t.size - 1 - d):
            a = (xc - t[j]) / (t[j + d] - t[j]) if t[j + d] > t[j] else 0.0
            b = (t[j + d + 1] - xc) / (t[j + d + 1] - t[j + 1]) if t[j + d + 1] > t[j + 1] else 0.0
            Bn[:, j] = a * B[:, j] + b * B[:, j + 1]
        B = Bn
    B = B[:, :k]
    B[(x < lo) | (x > hi) | ~np.isfinite(x)] = 0.0
    return B


def difference_penalty(k: int, order: int = 2) -> np.ndarray:
    D = np.diff(np.eye(k), n=order, axis=0)
    return D.T @ D


def pspline_fit(B: np.ndarray, y: np.ndarray, w: np.ndarray | None, lam: float, order: int = 2) -> np.ndarray:
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    ok = np.isfinite(y) & (w > 0)
    P = difference_penalty(B.shape[1], order)
    A = (B[ok] * w[ok, None]).T @ B[ok] + lam * P + 1e-8 * np.eye(B.shape[1])
    return np.linalg.solve(A, (B[ok] * w[ok, None]).T @ y[ok])


def pspline_gcv(B: np.ndarray, y: np.ndarray, w: np.ndarray | None, lams: Sequence[float], order: int = 2) -> tuple[float, np.ndarray]:
    """Penalty by generalised cross-validation (used only when no independent repeat exists)."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    ok = np.isfinite(y) & (w > 0)
    Bo, yo, wo = B[ok], y[ok], w[ok]
    P = difference_penalty(B.shape[1], order)
    best, out = np.inf, (float(lams[0]), np.zeros(B.shape[1]))
    for lam in lams:
        A = (Bo * wo[:, None]).T @ Bo + lam * P + 1e-8 * np.eye(B.shape[1])
        Ainv = np.linalg.inv(A)
        beta = Ainv @ (Bo * wo[:, None]).T @ yo
        H_tr = float(np.trace(Bo @ Ainv @ (Bo * wo[:, None]).T))
        rss = float((wo * (yo - Bo @ beta) ** 2).sum())
        n = float(wo.sum())
        gcv = n * rss / max((n - H_tr) ** 2, 1e-9)
        if gcv < best:
            best, out = gcv, (float(lam), beta)
    return out


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    r = np.asarray(y, float) - np.asarray(q, float)
    return np.where(r >= 0, tau * r, (tau - 1.0) * r)


def quantile_pspline(B: np.ndarray, y: np.ndarray, w: np.ndarray, tau: float, lam: float, iters: int = 60, eps: float = 1e-4,
                     order: int = 2) -> np.ndarray:
    """Penalised τ-quantile regression: minimise Σ w ρ_τ(y − Bβ) + λ‖Dβ‖² by IRLS (each iteration a weighted ridge with
    weights w·|τ − 1(r < 0)| / max(|r|, eps)). ``w`` = frame counts of histogram cells."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    ok = np.isfinite(y) & (w > 0)
    Bo, yo, wo = B[ok], y[ok], w[ok]
    P = difference_penalty(B.shape[1], order)
    beta = np.linalg.solve((Bo * wo[:, None]).T @ Bo + lam * P + 1e-6 * np.eye(B.shape[1]), (Bo * wo[:, None]).T @ yo)
    for _ in range(iters):
        r = yo - Bo @ beta
        ww = wo * np.where(r >= 0, tau, 1.0 - tau) / np.maximum(np.abs(r), eps)
        new = np.linalg.solve((Bo * ww[:, None]).T @ Bo + lam * P + 1e-6 * np.eye(B.shape[1]), (Bo * ww[:, None]).T @ yo)
        if np.max(np.abs(new - beta)) < 1e-6:
            beta = new
            break
        beta = new
    return beta


def rearrange(Q: np.ndarray) -> np.ndarray:
    """Sort quantile curves across τ at each x (rows = τ ascending). NaN columns stay NaN."""
    Q = np.asarray(Q, float).copy()
    for j in range(Q.shape[1]):
        col = Q[:, j]
        m = np.isfinite(col)
        if m.sum() > 1:
            col[m] = np.sort(col[m])
            Q[:, j] = col
    return Q


def crossing_rate(Q: np.ndarray) -> float:
    """Share of supported (x, adjacent τ pair) cells where a higher τ curve lies below a lower one."""
    Q = np.asarray(Q, float)
    d = np.diff(Q, axis=0)
    m = np.isfinite(d)
    return float((d[m] < -1e-9).mean()) if m.any() else NAN


__all__ = ["bspline_basis", "crossing_rate", "difference_penalty", "knots", "pinball", "pspline_fit", "pspline_gcv", "quantile_pspline", "rearrange"]
