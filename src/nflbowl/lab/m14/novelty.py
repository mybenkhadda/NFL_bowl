"""Phase 3 — novelty beyond official Combine testing, redundancy, and train-fold-only residualisation.

Novelty of a metric = how much of it the official measurements do NOT already explain, estimated out of fold:
for each outer fold a ridge of the metric on M1 (± position dummies) is fitted on the TRAINING athletes (train medians
for imputation, train mean / SD for scaling) and scored on the held-out athletes; OOF R² = 1 − SSE / Σ(x − x̄_train)².
novelty = 1 − max(0, OOF R²). No NFL outcome is involved anywhere in this module.

Redundancy: pairwise Spearman (pairwise-complete, ≥ 30 athletes) among the M14 columns and against the existing M12 /
M13 features (biomech ``c_*``, F0 ``eng*``); single-linkage clusters at |ρ| ≥ ``redundancy_rho``; the representative
of a cluster is the most RELIABLE member (never the most predictive).

Train-fold-only residual blocks (rugby / soccer metrics): ``ae.Model`` factories whose OLS is refitted inside every
outer fold on the training athletes only; held-out athletes are only scored.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ...data.validation import M1_COLS
from .. import reliability as REL

NAN = float("nan")


# ====================================================================== helpers
def _matrix(data: Any, cols: Sequence[str]) -> np.ndarray:
    import polars as pl

    if not cols:
        return np.zeros((data.height, 0))
    return data.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy().astype(float)


def _standardise(X: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Train-median imputation, train mean / SD scaling (constant train columns → 0)."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        tr = col[train & np.isfinite(col)]
        med = float(np.median(tr)) if tr.size else 0.0
        col = np.where(np.isfinite(col), col, med)
        mu, sd = float(col[train].mean()) if train.any() else 0.0, float(col[train].std()) if train.sum() > 1 else 0.0
        X[:, j] = (col - mu) / sd if sd > 1e-12 else 0.0
    return X


def _dummies(pos: np.ndarray, train: np.ndarray) -> np.ndarray:
    levels = sorted(set(pos[train].tolist()))[1:]                       # first training level = reference
    return np.stack([(pos == lv).astype(float) for lv in levels], 1) if levels else np.zeros((pos.size, 0))


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """Ridge with an unpenalised intercept (X centred on the fitting rows)."""
    mu = float(y.mean())
    if X.shape[1] == 0:
        return mu, np.zeros(0)
    xm = X.mean(0)
    Xc = X - xm
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - mu))
    return mu - float(xm @ beta), beta


# ====================================================================== novelty
def oof_explained(data: Any, col: str, covs: Sequence[str], positions: bool, alpha: float = 1.0, min_train: int = 30) -> dict[str, float]:
    """Out-of-fold R² of ``col`` from ``covs`` (+ position dummies), folds from ``data['fold']``."""
    x = data[col].cast(float).to_numpy()
    folds = data["fold"].to_numpy()
    pos = data["position_group"].fill_null("?").to_numpy() if "position_group" in data.columns else np.array(["?"] * data.height)
    Z = _matrix(data, [c for c in covs if c in data.columns])
    ok = np.isfinite(x)
    pred = np.full(x.size, NAN)
    ref = np.full(x.size, NAN)
    for f in np.unique(folds[ok]):
        tr, te = ok & (folds != f), ok & (folds == f)
        if tr.sum() < min_train or not te.any():
            continue
        Xs = _standardise(Z, tr)
        if positions:
            Xs = np.concatenate([Xs, _dummies(pos, tr)], 1)
        b0, beta = _ridge(Xs[tr], x[tr], alpha)
        pred[te] = b0 + Xs[te] @ beta
        ref[te] = float(x[tr].mean())
    m = np.isfinite(pred)
    if m.sum() < min_train:
        return {"n": int(m.sum()), "oof_r2": NAN}
    sse, sst = float(((x[m] - pred[m]) ** 2).sum()), float(((x[m] - ref[m]) ** 2).sum())
    return {"n": int(m.sum()), "oof_r2": 1 - sse / sst if sst > 0 else NAN}


def novelty_table(data: Any, cols: Sequence[str], existing: Sequence[str], max_r2: float) -> Any:
    """Per column: OOF R² from M1 and from M1 + position, novelty, Spearman with each official, strongest |ρ| with an
    existing (M12 / M13) feature."""
    import polars as pl

    from ..stats import spearman

    m1 = [c for c in M1_COLS if c in data.columns]
    E = {c: data[c].cast(float).to_numpy() for c in existing if c in data.columns}
    rows = []
    for c in cols:
        if c not in data.columns:
            continue
        x = data[c].cast(float).to_numpy()
        a = oof_explained(data, c, m1, positions=False)
        b = oof_explained(data, c, m1, positions=True)
        rec: dict[str, Any] = {"column": c, "n": b["n"], "oof_r2_m1": a["oof_r2"], "oof_r2_m1_position": b["oof_r2"]}
        r2 = b["oof_r2"]
        rec["novelty"] = 1 - max(0.0, r2) if np.isfinite(r2) else NAN
        rec["novel"] = bool(np.isfinite(r2) and r2 <= max_r2)
        for o in m1:
            y = data[o].cast(float).to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            rec[f"rho_{o}"] = spearman(x[ok], y[ok]) if ok.sum() >= 30 else NAN
        best, which = NAN, None
        for k, y in E.items():
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 30:
                r = spearman(x[ok], y[ok])
                if np.isfinite(r) and (not np.isfinite(best) or abs(r) > abs(best)):
                    best, which = r, k
        rec["max_rho_existing"], rec["max_rho_existing_feature"] = best, which
        rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== redundancy
def spearman_matrix(data: Any, cols: Sequence[str], min_n: int = 30) -> Any:
    import polars as pl

    from ..stats import spearman

    V = {c: data[c].cast(float).to_numpy() for c in cols if c in data.columns}
    keys = list(V)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i:]:
            ok = np.isfinite(V[a]) & np.isfinite(V[b])
            r = 1.0 if a == b else (spearman(V[a][ok], V[b][ok]) if ok.sum() >= min_n else NAN)
            rows.append({"a": a, "b": b, "rho": r, "n": int(ok.sum())})
            if a != b:
                rows.append({"a": b, "b": a, "rho": r, "n": int(ok.sum())})
    return pl.DataFrame(rows, infer_schema_length=None)


def redundancy_clusters(mat: Any, rho: float, priority: dict[str, float]) -> Any:
    """Single-linkage clusters at |ρ| ≥ rho (union–find); representative = highest ``priority`` (reliability)."""
    import polars as pl

    cols = sorted(set(mat["a"].to_list()))
    parent = {c: c for c in cols}

    def find(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for r in mat.filter((pl.col("a") < pl.col("b")) & pl.col("rho").is_not_null() & pl.col("rho").is_not_nan()).iter_rows(named=True):
        if abs(r["rho"]) >= rho:
            ra, rb = find(r["a"]), find(r["b"])
            if ra != rb:
                parent[rb] = ra
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)
    rows = []
    for k, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        rep = max(members, key=lambda c: (priority.get(c, NAN) if np.isfinite(priority.get(c, NAN)) else -np.inf, c))
        for c in members:
            rows.append({"column": c, "cluster": k, "cluster_size": len(members), "representative": rep, "is_representative": c == rep})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== train-fold-only residualisation
def _ols_fit(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.concatenate([np.ones((Z.shape[0], 1)), Z], 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _ols_resid(Z: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return y - np.concatenate([np.ones((Z.shape[0], 1)), Z], 1) @ beta


def residual_specs(columns: Sequence[str]) -> list[tuple[str, str, tuple[str, ...]]]:
    """(output name, residualised column, covariates) for the train-fitted rugby / soccer metrics present in ``columns``."""
    out = []
    for off in ("three_cone", "short_shuttle"):
        if off in columns and "forty" in columns:
            out.append((f"cod_deficit_residual__{off}", off, tuple(c for c in ("forty", "weight") if c in columns)))
    if {"m14c_agility__turn_speed_retention", "m14c_agility__ctx_v_entry"} <= set(columns):
        out.append(("speed_adjusted_cutting", "m14c_agility__turn_speed_retention", ("m14c_agility__ctx_v_entry",)))
    return out


def train_residual_factory(data: Any, specs: Sequence[tuple[str, str, tuple[str, ...]]], log: list | None = None, min_train: int = 30,
                           model: str = "residual") -> Callable:
    """``ae.Model`` factory: per outer fold, OLS of each spec's column on its covariates fitted on TRAINING rows with
    finite values; residuals for every row with finite inputs (NaN otherwise → imputed inside the design)."""
    def factory(ids: np.ndarray) -> Callable:
        need = sorted({c for _, y, z in specs for c in (y, *z)})
        X = REL._aligned(data, ids, need)
        col = {c: j for j, c in enumerate(need)}
        folds = REL._aligned(data, ids, ["fold"])[:, 0]

        def block(train: np.ndarray) -> Any:
            import polars as pl

            f = REL._fold_of(train, folds)
            out = {}
            for name, y, z in specs:
                yy = X[:, col[y]]
                Z = X[:, [col[c] for c in z]]
                ok = np.isfinite(yy) & np.isfinite(Z).all(1)
                r = np.full(yy.size, np.nan)
                tr = ok & train
                if tr.sum() >= min_train:
                    beta = _ols_fit(Z[tr], yy[tr])
                    r[ok] = _ols_resid(Z[ok], yy[ok], beta)
                    if log is not None:
                        log.append({"model": model, "fold": f, "feature": name, "n_train_fit": int(tr.sum()), "coef": [float(b) for b in beta]})
                out[name] = r
            return pl.DataFrame(out) if out else pl.DataFrame({"none": np.zeros(ids.size)})
        return block
    return factory


def oof_residual(data: Any, y: str, covs: Sequence[str], min_train: int = 30) -> np.ndarray:
    """Descriptive athlete-level residual: each athlete scored by the OLS fitted on the OTHER folds (aligned to data rows)."""
    yy = data[y].cast(float).to_numpy()
    Z = _matrix(data, list(covs))
    folds = data["fold"].to_numpy()
    ok = np.isfinite(yy) & np.isfinite(Z).all(1)
    out = np.full(yy.size, np.nan)
    for f in np.unique(folds):
        tr, te = ok & (folds != f), ok & (folds == f)
        if tr.sum() >= min_train and te.any():
            out[te] = _ols_resid(Z[te], yy[te], _ols_fit(Z[tr], yy[tr]))
    return out


__all__ = ["novelty_table", "oof_explained", "oof_residual", "redundancy_clusters", "residual_specs", "spearman_matrix", "train_residual_factory"]
