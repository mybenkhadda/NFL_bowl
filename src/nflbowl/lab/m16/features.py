"""E16.C0 — fold-specific, Combine-only athlete feature tables for the incremental ladder (Track B2 / C).

For outer fold f, every athlete's features come from the fold-f encoder (never trained on fold-f athletes) and from
COMBINE rows only — no NFL-derived representation of any athlete is ever an input (scouting setting). The fold-f table
holds training athletes (fold ≠ f, used to fit the downstream model) and held-out athletes (fold f, scored), exactly the
official ladder convention (``se.embedding_block``).

Aggregations of an athlete's Combine attempts within a drill scope (all attempts, forty, agility, three_cone, short_shuttle):

    mean          official convention (primary, pre-declared)
    median        robust centre
    trimmed_mean  mean after dropping ``trim_share`` of attempts at each end (per dimension; plain mean below 5 attempts)
    quantiles     25th and 75th percentile per dimension (2·d columns)
    icc_weighted  mean × sqrt(ICC_d) per latent dimension, ICC(1,1) of repeated attempts estimated on the TRAINING athletes
                  of fold f only (unreliable dimensions are shrunk towards 0)

Missingness is preserved: an athlete without attempts in a scope has no row (NaN in the ladder → the official design adds a
missing indicator fitted on training rows); ``n_attempts`` per athlete and scope is kept in the inventory.

Fusion (STFT ⊕ LSTM): per fold, each block is standardised with the mean / SD of the TRAINING athletes of that fold, then
concatenated (``fuse``). Handcrafted drill kinematics (F0-like controls): athlete medians of the ``m15.signals`` features
per drill (``drill_kinematics``).

Ladder factories: ``block_factory(fold_of, by_fold, k)`` — k given: the official ``se.embedding_block`` (train-fold PCA to
≤ k dims); k None: the aligned columns as they are. Both return one row per analysis row in analysis order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

NAN = float("nan")
SCOPES = {"all": None, "forty": ("forty",), "agility": ("three_cone", "short_shuttle", "long_shuttle"), "three_cone": ("three_cone",),
          "short_shuttle": ("short_shuttle",)}
KINEMATICS = ("max_speed", "q90_speed", "entry_speed", "max_atan", "min_atan", "accel_frac", "braking_frac", "jerk_rms", "max_abs_an")


_COMBINE_CACHE: dict[tuple[str, int, int], Any] = {}


def load_combine(m15_run_dir: str, arch: str, d: int, seed: int, fold: int) -> Any:
    """Combine rows of the fold-f embedding table (idx, nfl_id, kind, role, position_group, z…), or None when absent.
    Memoised per (path, size, mtime) — a rewritten file is re-read; the cache is never keyed on the file name alone."""
    import polars as pl

    p = Path(m15_run_dir) / "embeddings" / "inductive" / f"{arch}_d{d}_s{seed}__fold{fold}.parquet"
    if not p.is_file():
        return None
    st = p.stat()
    key = (str(p), int(st.st_size), int(st.st_mtime_ns))
    if key not in _COMBINE_CACHE:
        _COMBINE_CACHE[key] = pl.read_parquet(p).filter(pl.col("domain") == "combine").with_columns(pl.col("nfl_id").cast(pl.Int64))
    return _COMBINE_CACHE[key]


def zcols(e: Any) -> list[str]:
    return [c for c in e.columns if c.startswith("z") and c[1:].isdigit()]


def scope_rows(e: Any, scope: str) -> Any:
    import polars as pl

    kinds = SCOPES[scope]
    return e if kinds is None else e.filter(pl.col("kind").is_in(list(kinds)))


def icc_weights(e: Any, zc: Sequence[str], train_ids: set[int]) -> np.ndarray:
    """Per latent dimension ICC(1,1) of repeated attempts (kind-centred with training means), TRAINING athletes only; clipped
    to [0, 1]. Dimensions without ≥ 10 repeated athletes get weight 0."""
    import polars as pl

    from ..reliability import icc_components

    t = e.filter(pl.col("nfl_id").is_in(list(train_ids)))
    if not t.height:
        return np.zeros(len(zc))
    t = t.with_columns([(pl.col(c) - pl.col(c).mean().over("kind")).alias(c) for c in zc])
    ids = t["nfl_id"].to_numpy()
    w = []
    for c in zc:
        comp = icc_components(t[c].to_numpy().astype(float), ids)
        w.append(comp["icc1"] if comp["n_athletes"] >= 10 and np.isfinite(comp["icc1"]) else 0.0)
    return np.clip(np.asarray(w, float), 0.0, 1.0)


def _grouped(e: Any, zc: Sequence[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    ids = e["nfl_id"].to_numpy()
    Z = e.select(zc).to_numpy().astype(float)
    order = np.argsort(ids, kind="stable")
    ids, Z = ids[order], Z[order]
    u, start = np.unique(ids, return_index=True)
    return u, np.split(Z, start[1:])


def aggregate(e: Any, agg: str, weights: np.ndarray | None = None, trim: float = 0.1, prefix: str = "e") -> Any:
    """One row per athlete: nfl_id, n_attempts, <prefix>0 … (see module docstring)."""
    import polars as pl

    zc = zcols(e)
    if not e.height or not zc:
        return pl.DataFrame(schema={"nfl_id": pl.Int64, "n_attempts": pl.Int64})
    u, groups = _grouped(e, zc)
    if agg == "mean":
        M = np.stack([g.mean(0) for g in groups])
    elif agg == "median":
        M = np.stack([np.median(g, 0) for g in groups])
    elif agg == "trimmed_mean":
        def tm(g: np.ndarray) -> np.ndarray:
            if g.shape[0] < 5:
                return g.mean(0)
            k = int(np.floor(trim * g.shape[0]))
            s = np.sort(g, axis=0)
            return s[k: g.shape[0] - k].mean(0)
        M = np.stack([tm(g) for g in groups])
    elif agg == "quantiles":
        M = np.concatenate([np.stack([np.quantile(g, 0.25, axis=0) for g in groups]), np.stack([np.quantile(g, 0.75, axis=0) for g in groups])], axis=1)
    elif agg == "icc_weighted":
        if weights is None:
            raise ValueError("icc_weighted needs training-fold ICC weights")
        M = np.stack([g.mean(0) for g in groups]) * np.sqrt(weights)[None, :]
    else:
        raise ValueError(f"unknown aggregation {agg!r}")
    return pl.DataFrame({"nfl_id": u.astype(np.int64), "n_attempts": np.array([g.shape[0] for g in groups], np.int64),
                         **{f"{prefix}{j}": M[:, j] for j in range(M.shape[1])}})


def by_fold(m15_run_dir: str, rep: dict[str, Any], scope: str, agg: str, fold_of: dict[int, int], outer_folds: int, trim: float = 0.1) -> tuple[dict[int, Any], Any]:
    """({fold: nfl_id + e… table}, inventory rows). A missing fold file leaves that fold absent (the ladder then sees NaN
    for every athlete of that fold → its block is all-missing: reported in the inventory, never silently filled)."""
    import polars as pl

    out, inv = {}, []
    for f in range(outer_folds):
        e = load_combine(m15_run_dir, rep["arch"], int(rep["d"]), int(rep["seed"]), f)
        if e is None:
            inv.append({"rep": rep["rep"], "scope": scope, "aggregation": agg, "fold": f, "status": "missing fold file", "n_athletes": 0})
            continue
        s = scope_rows(e, scope)
        train_ids = {i for i, k in fold_of.items() if k != f}
        w = icc_weights(s, zcols(s), train_ids) if agg == "icc_weighted" else None
        t = aggregate(s, agg, w, trim)
        out[f] = t.select("nfl_id", *[c for c in t.columns if c.startswith("e") and c[1:].isdigit()])
        held = t.filter(pl.col("nfl_id").replace_strict(fold_of, default=-1, return_dtype=pl.Int64) == f)
        inv.append({"rep": rep["rep"], "scope": scope, "aggregation": agg, "fold": f, "status": "ok", "n_athletes": t.height, "n_heldout_athletes": held.height,
                    "n_dims": out[f].width - 1, "median_attempts": float(t["n_attempts"].median()) if t.height else None,
                    "mean_icc_weight": float(np.mean(w)) if w is not None else None})
    return out, pl.DataFrame(inv, infer_schema_length=None)


def fuse(a: dict[int, Any], b: dict[int, Any], fold_of: dict[int, int]) -> dict[int, Any]:
    """Per fold: each block standardised with its TRAINING athletes (fold ≠ f), then concatenated (athletes in both)."""
    import polars as pl

    out = {}
    for f in sorted(set(a) & set(b)):
        parts = []
        for k, t in enumerate((a[f], b[f])):
            cols = [c for c in t.columns if c != "nfl_id"]
            tr = t.filter(pl.col("nfl_id").replace_strict(fold_of, default=-1, return_dtype=pl.Int64) != f)
            mu = {c: float(tr[c].mean()) for c in cols}
            sd = {c: (float(tr[c].std()) or 1.0) for c in cols}
            parts.append(t.select("nfl_id", *[((pl.col(c) - mu[c]) / sd[c]).alias(f"b{k}_{c}") for c in cols]))
        j = parts[0].join(parts[1], on="nfl_id", how="inner")
        new = [c for c in j.columns if c != "nfl_id"]
        out[f] = j.rename({c: f"e{k}" for k, c in enumerate(new)})
    return out


def drill_kinematics(feat: Any, scopes: Sequence[str] = ("forty", "agility", "three_cone", "short_shuttle")) -> Any:
    """Athlete medians of the per-sequence kinematics per Combine drill scope: ``kin_<scope>__<feature>`` (+ attempt counts)."""
    import polars as pl

    out = None
    for s in scopes:
        kinds = SCOPES[s]
        g = feat.filter((pl.col("domain") == "combine") & pl.col("kind").is_in(list(kinds))).group_by("nfl_id").agg(
            pl.len().alias(f"kin_{s}__n_attempts"), *[pl.col(c).median().alias(f"kin_{s}__{c}") for c in KINEMATICS if c in feat.columns])
        out = g if out is None else out.join(g, on="nfl_id", how="full", coalesce=True)
    return out.with_columns(pl.col("nfl_id").cast(pl.Int64))


def kinematic_columns(frame: Any, scope: str) -> list[str]:
    return [c for c in frame.columns if c.startswith(f"kin_{scope}__") and not c.endswith("__n_attempts")]


def block_factory(fold_of: dict[int, int], by_fold_tables: dict[int, Any], k: int | None) -> Callable[[np.ndarray], Callable[[np.ndarray], Any]]:
    """``ae.Model.factory``: k → official train-fold PCA block (``se.embedding_block``); None → raw aligned columns."""
    import polars as pl

    if k is not None:
        from ... import _backend

        se = _backend.se()
        return lambda ids, _b=by_fold_tables, _k=k: se.embedding_block(ids, fold_of, _b, _k)

    def factory(ids: np.ndarray) -> Callable[[np.ndarray], Any]:
        folds = np.array([fold_of.get(int(i), -1) for i in ids])

        def block(train: np.ndarray) -> Any:
            f = np.unique(folds[~train])
            if f.size != 1:
                raise ValueError(f"held-out rows span folds {f.tolist()}")
            t = by_fold_tables.get(int(f[0]))
            if t is None:
                return pl.DataFrame({"none": np.zeros(ids.size)})
            cols = [c for c in t.columns if c != "nfl_id"]
            row = {int(i): r for r, i in enumerate(t["nfl_id"].to_list())}
            T = t.select(cols).to_numpy().astype(float)
            Z = np.array([T[row[int(i)]] if int(i) in row else np.full(len(cols), NAN) for i in ids])
            return pl.DataFrame({c: Z[:, j] for j, c in enumerate(cols)})
        return block
    return factory


def coverage(by_fold_tables: dict[int, Any], ids: Sequence[int], fold_of: dict[int, int]) -> dict[str, float]:
    """Share of the given athletes that have a feature row in their OWN held-out fold table (what the ladder scores)."""
    present = {f: set(int(i) for i in t["nfl_id"].to_list()) for f, t in by_fold_tables.items()}
    have = [int(i) in present.get(fold_of.get(int(i), -1), set()) for i in ids]
    return {"n": len(have), "covered": int(sum(have)), "share": float(np.mean(have)) if have else NAN}


def m1_relation(by_fold_tables: dict[int, Any], athletes: Any, m1_cols: Sequence[str], fold_of: dict[int, int], alpha: float = 1.0) -> Any:
    """How much of the official Combine measurements each fold's features already encode: per M1 column, ridge R² of
    (features → M1 column) fitted on TRAINING athletes and scored on held-out athletes, and the max |Spearman| of a single
    dimension on held-out athletes. High values = the representation largely re-encodes the stopwatch / body size."""
    import polars as pl

    from .target_audit import _spearman

    rows = []
    a = athletes.select(pl.col("nfl_id").cast(pl.Int64), *[pl.col(c).cast(pl.Float64) for c in m1_cols if c in athletes.columns])
    for f, t in by_fold_tables.items():
        d = t.join(a, on="nfl_id", how="inner")
        cols = [c for c in t.columns if c != "nfl_id"]
        held = d["nfl_id"].replace_strict(fold_of, default=-1, return_dtype=pl.Int64).to_numpy() == f
        X = d.select(cols).to_numpy().astype(float)
        for c in m1_cols:
            if c not in d.columns:
                continue
            y = d[c].fill_null(NAN).to_numpy()
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            tr, te = ok & ~held, ok & held
            if tr.sum() < len(cols) + 10 or te.sum() < 10:
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
            beta = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(Xt.shape[1]), Xt.T @ (y[tr] - y[tr].mean()))
            pred = y[tr].mean() + Xe @ beta
            ss = float(((y[te] - y[te].mean()) ** 2).sum())
            r2 = 1 - float(((y[te] - pred) ** 2).sum()) / ss if ss > 0 else NAN
            rho = [abs(_spearman(X[te, j], y[te])) for j in range(X.shape[1])]
            rows.append({"fold": int(f), "m1_column": c, "n_train": int(tr.sum()), "n_heldout": int(te.sum()), "ridge_r2_heldout": r2,
                         "max_abs_spearman_dim": float(np.nanmax(rho)) if rho else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["KINEMATICS", "SCOPES", "aggregate", "block_factory", "by_fold", "coverage", "drill_kinematics", "fuse", "icc_weights", "kinematic_columns",
           "load_combine", "m1_relation", "scope_rows", "zcols"]
