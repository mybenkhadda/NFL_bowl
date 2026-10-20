"""Fold-safe alternative learners next to the official ridge ladder (M16 E16.A4, M18 Study A / E).

The official engine (``ae.oof_ladder``: ridge, inner-CV α, position dummies) stays the reference and the primary. This
module scores OTHER learners on exactly the same rows (finite target), folds and predictor sets so their predictions can
be compared pairwise with ``transfer_fairness``:

* ``design_train`` — train-fold medians impute missing values (+ one missing indicator per column with any gap), train-fold
  means / SDs standardise, position one-hot (unpenalised in the linear learners);
* ``ElasticNet`` — coordinate descent on standardised columns (NumPy), λ chosen by inner athlete-grouped CV on training
  athletes only, l1_ratio fixed per call;
* ``lightgbm`` / ``catboost`` — optional (``BLOCKED_MISSING_INPUT`` when not installed), shallow trees (depth ≤ 3), inner
  early stopping on a training-athlete split; position enters as a categorical / one-hot column;
* ``oof_generic`` — one OOF vector per learner, with a ``factory`` hook identical to ``ae.Model.factory`` (fold-fitted blocks
  such as embedding PCA or train-only quality selection are reused unchanged).

Nothing here sees a held-out athlete's target, and every fitted transform uses training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .status import BLOCKED_MISSING_INPUT, ResearchBlocked

NAN = float("nan")


@dataclass
class Design:
    X: np.ndarray
    pen: np.ndarray          # True = penalised column
    names: list[str]


def _frame_matrix(df: Any, cols: Sequence[str]) -> np.ndarray:
    import polars as pl

    if not cols:
        return np.zeros((df.height, 0))
    return df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy().astype(float)


def design_train(df: Any, cols: Sequence[str], extra: Any, train: np.ndarray, position: str = "position_group") -> Design:
    """Design matrix for all rows from TRAIN-fold statistics only (see module docstring)."""
    A = _frame_matrix(df, cols)
    names = list(cols)
    if extra is not None and getattr(extra, "width", 0):
        B = extra.to_numpy().astype(float)
        A = np.concatenate([A, B], axis=1) if A.size else B
        names += [f"__b_{c}" for c in extra.columns]
    blocks, pen, out_names = [], [], []
    for j, n in enumerate(names):
        x = A[:, j]
        tr = x[train]
        ok = np.isfinite(tr)
        med = float(np.median(tr[ok])) if ok.any() else 0.0
        miss = ~np.isfinite(x)
        xi = np.where(miss, med, x)
        mu, sd = float(xi[train].mean()), float(xi[train].std())
        blocks.append((xi - mu) / sd if sd > 1e-12 else np.zeros_like(xi))
        pen.append(True)
        out_names.append(n)
        if miss[train].any() and not miss[train].all():
            ind = miss.astype(float)
            m2, s2 = float(ind[train].mean()), float(ind[train].std())
            blocks.append((ind - m2) / s2 if s2 > 1e-12 else np.zeros_like(ind))
            pen.append(True)
            out_names.append(f"{n}__missing")
    pos = df[position].fill_null("?").to_numpy() if position in df.columns else np.array(["?"] * df.height)
    levels = sorted(set(pos[train].tolist()))
    for lv in levels[1:]:
        blocks.append((pos == lv).astype(float))
        pen.append(False)
        out_names.append(f"pos_{lv}")
    X = np.column_stack(blocks) if blocks else np.zeros((df.height, 0))
    return Design(X, np.asarray(pen, bool), out_names)


# ====================================================================== elastic net
def enet_fit(X: np.ndarray, y: np.ndarray, lam: float, l1_ratio: float, pen: np.ndarray, n_iter: int = 300, tol: float = 1e-7) -> tuple[float, np.ndarray]:
    """Coordinate descent for 1/(2n)·||y − b0 − Xβ||² + λ·(α||β_pen||₁ + (1−α)/2·||β_pen||²); unpenalised columns free."""
    n, p = X.shape
    b0 = float(y.mean())
    beta = np.zeros(p)
    if p == 0:
        return b0, beta
    r = y - b0
    col_ss = (X ** 2).sum(0) / n
    for _ in range(n_iter):
        max_d = 0.0
        for j in range(p):
            if col_ss[j] <= 0:
                continue
            old = beta[j]
            rho = (X[:, j] @ r) / n + col_ss[j] * old
            if pen[j]:
                a1, a2 = lam * l1_ratio, lam * (1 - l1_ratio)
                new = np.sign(rho) * max(abs(rho) - a1, 0.0) / (col_ss[j] + a2)
            else:
                new = rho / col_ss[j]
            if new != old:
                r -= X[:, j] * (new - old)
                beta[j] = new
                max_d = max(max_d, abs(new - old))
        shift = float(r.mean())
        b0 += shift
        r -= shift
        if max_d < tol:
            break
    return b0, beta


def _inner_folds(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n) % k


def enet_learner(lams: Sequence[float] = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0), l1_ratio: float = 0.5, inner: int = 5) -> Callable:
    """Learner(df, y, cols, extra, train, test, seed) → predictions on ``test``; λ by inner CV on training athletes."""
    def fit_predict(df: Any, y: np.ndarray, cols: Sequence[str], extra: Any, train: np.ndarray, test: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        idx = np.flatnonzero(train)
        fo = _inner_folds(idx.size, inner, seed)
        sse = np.zeros(len(lams))
        for j in range(inner):
            tr, va = np.zeros(df.height, bool), np.zeros(df.height, bool)
            tr[idx[fo != j]], va[idx[fo == j]] = True, True
            d = design_train(df, cols, extra, tr)
            for k, lam in enumerate(lams):
                b0, b = enet_fit(d.X[tr], y[tr], lam, l1_ratio, d.pen)
                sse[k] += float(((y[va] - b0 - d.X[va] @ b) ** 2).sum())
        lam = float(lams[int(np.argmin(sse))])
        d = design_train(df, cols, extra, train)
        b0, b = enet_fit(d.X[train], y[train], lam, l1_ratio, d.pen)
        return b0 + d.X[test] @ b, {"lambda": lam, "l1_ratio": l1_ratio, "n_nonzero": int((np.abs(b[d.pen]) > 1e-10).sum())}
    return fit_predict


# ====================================================================== optional trees
def _tree_inputs(df: Any, cols: Sequence[str], extra: Any, train: np.ndarray) -> np.ndarray:
    d = design_train(df, cols, extra, train)
    return d.X


def lightgbm_learner(max_depth: int = 3, n_estimators: int = 600, learning_rate: float = 0.03) -> Callable:
    def fit_predict(df: Any, y: np.ndarray, cols: Sequence[str], extra: Any, train: np.ndarray, test: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ResearchBlocked(BLOCKED_MISSING_INPUT, "LightGBM learner needs the optional package 'lightgbm'", missing=["lightgbm"]) from e
        X = _tree_inputs(df, cols, extra, train)
        idx = np.flatnonzero(train)
        fo = _inner_folds(idx.size, 5, seed)
        tr, va = idx[fo != 0], idx[fo == 0]
        m = lgb.LGBMRegressor(max_depth=max_depth, num_leaves=2 ** max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
                              min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        best = int(m.best_iteration_ or n_estimators)
        m2 = lgb.LGBMRegressor(max_depth=max_depth, num_leaves=2 ** max_depth, n_estimators=best, learning_rate=learning_rate, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m2.fit(X[train], y[train])
        return m2.predict(X[test]), {"n_estimators": best}
    return fit_predict


def catboost_learner(depth: int = 3, iterations: int = 800, learning_rate: float = 0.03) -> Callable:
    def fit_predict(df: Any, y: np.ndarray, cols: Sequence[str], extra: Any, train: np.ndarray, test: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ResearchBlocked(BLOCKED_MISSING_INPUT, "CatBoost learner needs the optional package 'catboost'", missing=["catboost"]) from e
        X = _tree_inputs(df, cols, extra, train)
        idx = np.flatnonzero(train)
        fo = _inner_folds(idx.size, 5, seed)
        tr, va = idx[fo != 0], idx[fo == 0]
        m = CatBoostRegressor(depth=depth, iterations=iterations, learning_rate=learning_rate, random_seed=seed, verbose=False, od_type="Iter", od_wait=50)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
        best = int(m.get_best_iteration() or iterations) + 1
        m2 = CatBoostRegressor(depth=depth, iterations=best, learning_rate=learning_rate, random_seed=seed, verbose=False)
        m2.fit(X[train], y[train])
        return m2.predict(X[test]), {"iterations": best}
    return fit_predict


LEARNERS: dict[str, Callable[[], Callable]] = {"enet": enet_learner, "lightgbm": lightgbm_learner, "catboost": catboost_learner}


# ====================================================================== OOF engine
def oof_generic(df: Any, target: str, cols: Sequence[str], learner: Callable, seed: int, factory: Callable | None = None, fold_col: str = "fold") -> tuple[Any, list[dict]]:
    """(frame nfl_id, position_group, fold, y, pred) on the rows with a finite target — the same row filter as
    ``ae.oof_ladder`` — and the per-fold log. ``factory(ids) → block(train)`` = the ``ae.Model.factory`` contract."""
    import polars as pl

    keep = df[target].cast(pl.Float64).is_finite().fill_null(False)
    d = df.filter(keep)
    y = d[target].cast(pl.Float64).to_numpy()
    folds = d[fold_col].to_numpy()
    ids = d["nfl_id"].to_numpy()
    for i in np.unique(ids):
        if np.unique(folds[ids == i]).size > 1:
            raise AssertionError(f"athlete {i} spans several folds")
    block = factory(ids) if factory is not None else None
    pred = np.full(d.height, NAN)
    log = []
    for f in sorted(set(folds.tolist())):
        test, train = folds == f, folds != f
        if test.sum() == 0 or train.sum() < 10:
            continue
        extra = block(train) if block is not None else None
        p, info = learner(d, y, cols, extra, train, test, seed + 31 * int(f))
        pred[test] = p
        log.append({"fold": int(f), "n_train": int(train.sum()), "n_test": int(test.sum()), **info})
    out = pl.DataFrame({"nfl_id": ids, "position_group": d["position_group"].fill_null("?").to_numpy(), "fold": folds, "y": y, "pred": pred})
    return out, log


__all__ = ["Design", "LEARNERS", "catboost_learner", "design_train", "enet_fit", "enet_learner", "lightgbm_learner", "oof_generic"]
