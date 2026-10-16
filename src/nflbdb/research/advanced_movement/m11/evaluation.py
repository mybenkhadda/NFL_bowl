"""Athlete-held-out evaluation shared by every predictive track (brief §15). NumPy only.

Rules enforced in code
    * folds are over ATHLETES; every Combine attempt and every NFL season of an athlete follows its fold
      (``make_folds`` / ``assign_folds``; Milestone 06 folds are reused for the athletes they cover)
    * imputation medians, missing indicators, centring, scaling, dummy levels and the ridge penalty are taken from the
      TRAINING athletes of the fold only (``design``); the penalty is chosen by inner athlete-grouped folds
    * models of one comparison are scored on exactly the same athletes and labels; differences are PAIRED (same athlete
      resamples for both models)
    * Δ RMSE = RMSE(reference) − RMSE(model): positive = the model is better
    * an interval is reported only with ≥ ``MIN_CI_ATHLETES`` athletes; an interval that includes 0 is "inconclusive at
      this n", never "no effect"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ...studies.stats import rankdata
from ..transfer_models import ridge_solve
from .config import M11Config

NAN = float("nan")
MIN_CI_ATHLETES = 20
CONTEXT = ("draft_year", "height", "weight")
OFFICIAL_CORE = ("forty", "ten")


class LeakageError(RuntimeError):
    """A held-out athlete influenced a training-side quantity."""


# ============================================================================== correlation helpers
def _finite(*xs: np.ndarray) -> np.ndarray:
    m = np.ones(np.asarray(xs[0]).shape[0], bool)
    for x in xs:
        x = np.asarray(x, float)
        m &= np.isfinite(x) if x.ndim == 1 else np.isfinite(x).all(1)
    return m


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = _finite(x, y)
    if m.sum() < 4:
        return NAN
    a, b = x[m] - x[m].mean(), y[m] - y[m].mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else NAN


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = _finite(x, y)
    return pearson(rankdata(x[m]), rankdata(y[m])) if m.sum() >= 4 else NAN


def residualize(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X = np.c_[np.ones(len(y)), Z]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Rank correlation of x and y after removing the linear rank effect of the columns of Z (e.g. official forty)."""
    x, y, Z = np.asarray(x, float), np.asarray(y, float), np.asarray(Z, float).reshape(len(x), -1)
    m = _finite(x, y, Z)
    if m.sum() < Z.shape[1] + 5:
        return NAN
    rz = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    return pearson(residualize(rankdata(x[m]), rz), residualize(rankdata(y[m]), rz))


def fisher_ci(r: float, n: int, k: int = 0) -> tuple[float, float]:
    if not np.isfinite(r) or n - k - 3 <= 0 or abs(r) >= 1:
        return NAN, NAN
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - k - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def mde_correlation(n: int, k: int = 0, power: float = 0.8) -> float:
    """Smallest (partial) correlation detectable at two-sided α = 0.05 with the given power, by the Fisher z approximation,
    conditioned on the ACTUAL cohort size n and k partialled covariates."""
    if n - k - 3 <= 0:
        return NAN
    z_a, z_b = 1.959964, {0.8: 0.841621, 0.9: 1.281552}.get(power, 0.841621)
    return float(np.tanh((z_a + z_b) / np.sqrt(n - k - 3)))


def holm(pvals: Sequence[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (p.size - r) * p[i]))
        adj[i] = run
    return adj.tolist()


# ============================================================================== folds
def make_folds(ids: np.ndarray, strata: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Stratified athlete folds: athletes of each stratum are shuffled and dealt round-robin (offset carried across strata)."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(ids), dtype=np.int64)
    offset = 0
    for s in sorted(set(strata.tolist())):
        idx = rng.permutation(np.flatnonzero(strata == s))
        fold[idx] = (np.arange(idx.size) + offset) % k
        offset += idx.size
    return fold


def assign_folds(athletes: pl.DataFrame, transfer_folds: pl.DataFrame | None, k: int, seed: int) -> pl.DataFrame:
    """(nfl_id, fold, fold_source). Athletes in the Milestone 06 folds keep them (``m06``); the others are dealt into the
    same k folds by position × cohort (``m11``). Folds never depend on a measurement or an NFL value."""
    a = athletes.select("nfl_id", "position_group", "draft_year").unique(subset="nfl_id").sort("nfl_id")
    out = a.with_columns(pl.lit(None, dtype=pl.Int64).alias("fold"), pl.lit("m11").alias("fold_source"))
    if transfer_folds is not None and transfer_folds.height:
        tf = transfer_folds.select("nfl_id", pl.col("outer_fold").cast(pl.Int64).alias("_f"))
        k = max(k, int(tf["_f"].max()) + 1)
        out = out.join(tf, on="nfl_id", how="left").with_columns(pl.col("_f").alias("fold"),
                                                                  pl.when(pl.col("_f").is_not_null()).then(pl.lit("m06")).otherwise(pl.lit("m11")).alias("fold_source")).drop("_f")
    rest = out.filter(pl.col("fold").is_null())
    if rest.height:
        strata = (rest["position_group"].fill_null("?") + "|" + rest["draft_year"].cast(pl.String).fill_null("?")).to_numpy()
        f = make_folds(rest["nfl_id"].to_numpy(), strata, k, seed)
        out = out.join(pl.DataFrame({"nfl_id": rest["nfl_id"], "_g": f}), on="nfl_id", how="left").with_columns(pl.coalesce("fold", "_g").alias("fold")).drop("_g")
    return out.select("nfl_id", pl.col("fold").cast(pl.Int64), "fold_source")


def assert_grouped(ids: np.ndarray, folds: np.ndarray) -> None:
    """Every row of one athlete must carry the same fold."""
    d: dict[int, int] = {}
    for i, f in zip(ids.tolist(), folds.tolist()):
        if d.setdefault(i, f) != f:
            raise LeakageError(f"athlete {i} appears in folds {d[i]} and {f}")


# ============================================================================== train-only design
@dataclass
class Design:
    X: np.ndarray
    names: list[str]
    pen: np.ndarray


def design(df: pl.DataFrame, cols: Sequence[str], train: np.ndarray, position: str | None = "position_group") -> Design:
    """Design matrix for EVERY row with every statistic from the ``train`` rows: medians (imputation), missing indicators
    (only when the training rows have missing values), mean / SD, position dummies (unpenalised, levels seen in training)."""
    blocks, names, pen = [], [], []
    for k in cols:
        x = df[k].cast(pl.Float64).fill_null(NAN).to_numpy().astype(float)
        tr = x[train]
        ok = np.isfinite(tr)
        if ok.sum() < 3:
            continue
        med = float(np.median(tr[ok]))
        miss = ~np.isfinite(x)
        xi = np.where(miss, med, x)
        mu, sd = float(xi[train].mean()), float(xi[train].std())
        if sd < 1e-12:
            continue
        blocks.append((xi - mu) / sd)
        names.append(k)
        pen.append(1.0)
        if (~ok).any():
            blocks.append(miss.astype(float) - miss[train].mean())
            names.append(f"{k}__missing")
            pen.append(1.0)
    if position and position in df.columns:
        pos = df[position].fill_null("?").to_numpy()
        levels = sorted(set(pos[train].tolist()))
        for lv in levels[1:]:
            blocks.append((pos == lv).astype(float))
            names.append(f"pos={lv}")
            pen.append(0.0)
    X = np.column_stack(blocks) if blocks else np.empty((df.height, 0))
    return Design(X, names, np.asarray(pen, float))


def _fit(df: pl.DataFrame, y: np.ndarray, cols: Sequence[str], train: np.ndarray, test: np.ndarray, alpha: float) -> np.ndarray:
    d = design(df, cols, train)
    b0, beta = ridge_solve(d.X[train], y[train], alpha, d.pen)
    return b0 + d.X[test] @ beta


def choose_alpha(df: pl.DataFrame, y: np.ndarray, cols: Sequence[str], train: np.ndarray, alphas: Sequence[float], inner_k: int, seed: int) -> float:
    """Penalty with the lowest inner-fold squared error, inner folds dealt among the TRAINING athletes only (ties → larger)."""
    idx = np.flatnonzero(train)
    if idx.size < 2 * inner_k:
        return float(max(alphas))
    inner = np.random.default_rng(seed).permutation(idx.size) % inner_k
    sse = np.zeros(len(alphas))
    for j in range(inner_k):
        tr, va = np.zeros(df.height, bool), np.zeros(df.height, bool)
        tr[idx[inner != j]], va[idx[inner == j]] = True, True
        for a_i, a in enumerate(alphas):
            p = _fit(df, y, cols, tr, va, a)
            sse[a_i] += float(((y[va] - p) ** 2).sum())
    best = np.flatnonzero(sse <= sse.min() * (1 + 1e-9))
    return float(alphas[int(best.max())])


def oof_predictions(df: pl.DataFrame, y: np.ndarray, models: dict[str, Sequence[str]], folds: np.ndarray, c: M11Config, seed: int
                    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """One out-of-fold prediction per athlete per model. ``models``: name → predictor columns (position always added)."""
    assert_grouped(df["nfl_id"].to_numpy(), folds)
    preds = {m: np.full(df.height, NAN) for m in models}
    log = []
    for f in sorted(set(folds.tolist())):
        test, train = folds == f, folds != f
        if test.sum() == 0 or train.sum() < 10:
            continue
        for m, cols in models.items():
            a = choose_alpha(df, y, cols, train, c.alphas, c.inner_folds, seed + 31 * f)
            preds[m][test] = _fit(df, y, cols, train, test, a)
            log.append({"model": m, "fold": int(f), "alpha": a, "n_train": int(train.sum()), "n_test": int(test.sum())})
    return preds, log


# ============================================================================== paired comparison
def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2))) if y.size else NAN


def paired_compare(y: np.ndarray, p_ref: np.ndarray, p_mod: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    """Δ RMSE = RMSE(ref) − RMSE(model) with a paired athlete bootstrap; Spearman of each model; R² of each."""
    m = _finite(y, p_ref, p_mod)
    y, a, b = y[m], p_ref[m], p_mod[m]
    n = int(y.size)
    out = {"n_athletes": n, "rmse_ref": rmse(y, a), "rmse_model": rmse(y, b), "delta_rmse": NAN, "delta_lo": NAN, "delta_hi": NAN, "delta_pct": NAN,
           "spearman_ref": spearman(y, a), "spearman_model": spearman(y, b), "r2_ref": NAN, "r2_model": NAN, "p_boot_le0": NAN}
    if n < 5:
        return out
    var = float(np.var(y))
    out["delta_rmse"] = out["rmse_ref"] - out["rmse_model"]
    out["delta_pct"] = 100.0 * out["delta_rmse"] / out["rmse_ref"] if out["rmse_ref"] > 0 else NAN
    out["r2_ref"], out["r2_model"] = (1 - np.mean((y - a) ** 2) / var, 1 - np.mean((y - b) ** 2) / var) if var > 0 else (NAN, NAN)
    if n >= MIN_CI_ATHLETES:
        idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
        d = np.sqrt(((y[idx] - a[idx]) ** 2).mean(1)) - np.sqrt(((y[idx] - b[idx]) ** 2).mean(1))
        out["delta_lo"], out["delta_hi"] = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
        out["p_boot_le0"] = float((d <= 0).mean())
    return out


def verdict(row: dict[str, Any]) -> str:
    """Reading of one comparison (never 'no effect' from an interval that includes 0)."""
    lo, hi = row.get("delta_lo"), row.get("delta_hi")
    if lo is None or not np.isfinite(lo):
        return "insufficient_athletes_for_interval"
    if lo > 0:
        return "model_better"
    if hi < 0:
        return "model_worse"
    return "inconclusive"


# ============================================================================== the standard ladder (M0 → M3)
def ladder_columns(context: Sequence[str], official: Sequence[str], tracking: Sequence[str], alternative: Sequence[str] = ()) -> dict[str, list[str]]:
    lad = {"M0": list(context), "M1": [*context, *official], "M2": [*context, *official, *tracking]}
    if alternative:
        lad["M3"] = [*context, *official, *alternative]
    return lad


def run_ladder(df: pl.DataFrame, target: str, ladder: dict[str, list[str]], folds: np.ndarray, c: M11Config, seed: int,
               comparisons: Sequence[tuple[str, str]] = (("M1", "M0"), ("M2", "M1"), ("M3", "M1")), labels: dict[str, Any] | None = None
               ) -> tuple[list[dict[str, Any]], pl.DataFrame]:
    """Out-of-fold predictions of every ladder model on the rows with a finite target, then the declared paired comparisons."""
    keep = df[target].cast(pl.Float64).is_finite().fill_null(False)
    d = df.filter(keep)
    f = folds[keep.to_numpy()]
    y = d[target].cast(pl.Float64).to_numpy()
    preds, _ = oof_predictions(d, y, ladder, f, c, seed)
    rows = []
    for mod, ref in comparisons:
        if mod in preds and ref in preds:
            r = paired_compare(y, preds[ref], preds[mod], c.n_boot, seed)
            rows.append({**(labels or {}), "target": target, "model": mod, "reference": ref, **r, "verdict": verdict(r)})
    oof = pl.DataFrame({"nfl_id": d["nfl_id"], "position_group": d["position_group"], "fold": f, "y": y, **{f"pred_{m}": p for m, p in preds.items()}})
    if labels:
        oof = oof.with_columns([pl.lit(str(v)).alias(k) for k, v in labels.items()])
    return rows, oof


# ============================================================================== negative controls
def permutation_null(df: pl.DataFrame, target: str, ref_cols: Sequence[str], mod_cols: Sequence[str], folds: np.ndarray, c: M11Config, seed: int,
                     n_perm: int | None = None) -> dict[str, Any]:
    """Athlete-label permutation WITHIN position: the tracking block is permuted between athletes of the same position
    (official results and labels stay), both models refitted. Null of Δ RMSE; one-sided p (model better)."""
    keep = df[target].cast(pl.Float64).is_finite().fill_null(False)
    d, f = df.filter(keep), folds[keep.to_numpy()]
    y = d[target].cast(pl.Float64).to_numpy()
    extra = [k for k in mod_cols if k not in ref_cols]
    obs_p, _ = oof_predictions(d, y, {"ref": ref_cols, "mod": mod_cols}, f, c, seed)
    observed = rmse(y, obs_p["ref"]) - rmse(y, obs_p["mod"])
    rng = np.random.default_rng(seed + 7)
    pos = d["position_group"].fill_null("?").to_numpy()
    null = []
    B = c.n_perm if n_perm is None else n_perm
    for _ in range(B):
        perm = np.arange(d.height)
        for g in np.unique(pos):
            ix = np.flatnonzero(pos == g)
            perm[ix] = rng.permutation(ix)
        dp = d.with_columns([pl.Series(k, d[k].to_numpy()[perm]) for k in extra])
        pp, _ = oof_predictions(dp, y, {"mod": mod_cols}, f, c, seed)
        null.append(rmse(y, obs_p["ref"]) - rmse(y, pp["mod"]))
    null = np.asarray(null)
    return {"observed": float(observed), "null_mean": float(null.mean()) if null.size else NAN, "null_q95": float(np.quantile(null, 0.95)) if null.size else NAN,
            "p_value": float((1 + (null >= observed).sum()) / (1 + null.size)) if null.size else NAN, "n_perm": int(null.size)}


def random_projection_control(df: pl.DataFrame, target: str, ref_cols: Sequence[str], k: int, folds: np.ndarray, c: M11Config, seed: int,
                              n_rep: int | None = None) -> np.ndarray:
    """Δ RMSE of M1 + k Gaussian noise columns (as many as the tracking block) over ``n_rep`` draws."""
    keep = df[target].cast(pl.Float64).is_finite().fill_null(False)
    d, f = df.filter(keep), folds[keep.to_numpy()]
    y = d[target].cast(pl.Float64).to_numpy()
    base, _ = oof_predictions(d, y, {"ref": ref_cols}, f, c, seed)
    rng = np.random.default_rng(seed + 11)
    out = []
    for r in range(c.n_random_projections if n_rep is None else n_rep):
        noise = {f"_rp{j}": rng.standard_normal(d.height) for j in range(k)}
        dn = d.with_columns([pl.Series(n_, v) for n_, v in noise.items()])
        p, _ = oof_predictions(dn, y, {"mod": [*ref_cols, *noise]}, f, c, seed)
        out.append(rmse(y, base["ref"]) - rmse(y, p["mod"]))
    return np.asarray(out)


def control_rows(df: pl.DataFrame, target: str, ref_cols: Sequence[str], controls: dict[str, Sequence[str]], folds: np.ndarray, c: M11Config, seed: int,
                 labels: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Δ RMSE (with CI) of M1 + each control block (wrong-drill features, noncomparable interval …)."""
    rows = []
    for name, cols in controls.items():
        if not cols:
            continue
        r, _ = run_ladder(df, target, {"M1": list(ref_cols), "CTRL": [*ref_cols, *cols]}, folds, c, seed, comparisons=(("CTRL", "M1"),), labels=labels)
        rows += [{**x, "control": name} for x in r]
    return rows


# ============================================================================== reliability
def icc_oneway(values: np.ndarray, ids: np.ndarray) -> tuple[float, int, int]:
    """ICC(1) by the one-way method of moments (athlete share of attempt variance), athletes, athletes with a repeat."""
    from ..movement_demands import variance_components

    m = np.isfinite(values)
    v, g = values[m], ids[m]
    if v.size < 4:
        return NAN, 0, 0
    _, counts = np.unique(g, return_counts=True)
    return variance_components(v, g)[2], int(counts.size), int((counts >= 2).sum())


def repeatability_table(long: pl.DataFrame, value: str = "value", by: Sequence[str] = ("feature",), n_boot: int = 200, seed: int = 0) -> pl.DataFrame:
    """ICC(1) per feature with an athlete-bootstrap interval. ``long``: nfl_id, <by…>, value (one row per attempt)."""
    rows = []
    rng = np.random.default_rng(seed)
    for key, g in sorted(long.partition_by(list(by), as_dict=True).items()):
        ids, v = g["nfl_id"].to_numpy(), g[value].cast(pl.Float64).fill_null(NAN).to_numpy()
        icc, n_ath, n_rep = icc_oneway(v, ids)
        lo = hi = NAN
        if n_rep >= MIN_CI_ATHLETES:
            uniq = np.unique(ids)
            pos = {u: np.flatnonzero(ids == u) for u in uniq}
            bs = []
            for _ in range(n_boot):
                pick = rng.choice(uniq, uniq.size)
                ix = np.concatenate([pos[u] for u in pick])
                lab = np.concatenate([np.full(pos[u].size, j) for j, u in enumerate(pick)])
                bs.append(icc_oneway(v[ix], lab)[0])
            bs = np.asarray(bs, float)
            bs = bs[np.isfinite(bs)]
            if bs.size >= 20:
                lo, hi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        rows.append({**dict(zip(by, key)), "n_athletes": n_ath, "n_repeat": n_rep, "icc": icc, "icc_lo": lo, "icc_hi": hi})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def split_half(df: pl.DataFrame, value: str, game: str = "game_id", min_per_half: int = 3) -> dict[str, Any]:
    """Game-parity split-half reliability of an athlete-level mean, Spearman–Brown corrected."""
    g = (df.filter(pl.col(value).is_finite()).with_columns((pl.col(game).rank("dense").over("nfl_id") % 2).alias("_half"))
         .group_by("nfl_id", "_half").agg(pl.col(value).mean().alias("_m"), pl.len().alias("_n")).filter(pl.col("_n") >= min_per_half))
    w = g.pivot(on="_half", index="nfl_id", values="_m").drop_nulls()
    if w.width < 3 or w.height < 10:
        return {"n_athletes": w.height, "r_half": NAN, "reliability": NAN}
    r = pearson(w[w.columns[1]].to_numpy(), w[w.columns[2]].to_numpy())
    return {"n_athletes": w.height, "r_half": r, "reliability": float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else NAN}


def stopwatch_redundancy(df: pl.DataFrame, feature: str, officials: Sequence[str] = OFFICIAL_CORE) -> dict[str, Any]:
    """Within-position R² of a tracking feature from the official tests (descriptive, in-sample). Near 1 = the stopwatch
    already carries it."""
    cols = [k for k in officials if k in df.columns]
    d = df.select("position_group", feature, *cols).drop_nulls()
    d = d.filter(pl.col(feature).is_finite())
    if d.height < 10 or not cols:
        return {"n_athletes": d.height, "r2_official": NAN}
    y = d[feature].to_numpy().astype(float)
    pos = d["position_group"].to_numpy()
    Z = [d[k].to_numpy().astype(float) for k in cols] + [(pos == p).astype(float) for p in sorted(set(pos.tolist()))[1:]]
    X = np.column_stack(Z)
    full = residualize(y, X)
    pos_only = residualize(y, np.column_stack(Z[len(cols):])) if len(Z) > len(cols) else y - y.mean()
    den = float((pos_only ** 2).sum())
    return {"n_athletes": d.height, "r2_official": float(1 - (full ** 2).sum() / den) if den > 0 else NAN}


# ============================================================================== clustered resampling (pairs sharing athletes)
def cluster_bootstrap_pairs(a: np.ndarray, b: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int, seed: int) -> tuple[float, float, np.ndarray]:
    """Bootstrap over ATHLETES for a statistic of PAIRS (a_i, b_i). Each replicate resamples athletes with replacement;
    a pair enters with weight m(a)·m(b) (multiplicities). ``stat(weights)`` returns the weighted statistic."""
    uniq = np.unique(np.concatenate([a, b]))
    pos = {u: i for i, u in enumerate(uniq.tolist())}
    ia, ib = np.array([pos[x] for x in a.tolist()]), np.array([pos[x] for x in b.tolist()])
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size).astype(float)
        w = mult[ia] * mult[ib]
        if w.sum() > 0:
            reps.append(stat(w))
    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    if reps.size < 20:
        return NAN, NAN, reps
    return float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975)), reps


# ============================================================================== descriptive association (athlete level)
def association(x: np.ndarray, y: np.ndarray, Z: np.ndarray | None, n_boot: int, seed: int, rel_x: float = NAN, rel_y: float = NAN,
                floor: float = 0.3, positions: np.ndarray | None = None, min_pos: int = 15) -> dict[str, Any]:
    """Pearson, Spearman (athlete-bootstrap CI), partial Spearman given Z (e.g. official forty, ten; CI), attenuation-corrected
    Pearson, the median within-position Spearman, and the minimum detectable correlation at this n. DESCRIPTIVE."""
    from .physics import attenuation_correct

    x, y = np.asarray(x, float), np.asarray(y, float)
    Zm = None if Z is None else np.asarray(Z, float).reshape(len(x), -1)
    m = _finite(x, y) if Zm is None else _finite(x, y, Zm)
    n = int(m.sum())
    out = {"n_athletes": n, "pearson": NAN, "spearman": NAN, "spearman_lo": NAN, "spearman_hi": NAN, "partial_spearman": NAN, "partial_lo": NAN,
           "partial_hi": NAN, "pearson_disattenuated": NAN, "within_position_spearman": NAN, "mde_r": mde_correlation(n, 0 if Zm is None else Zm.shape[1])}
    if n < 5:
        return out
    xs, ys = x[m], y[m]
    Zs = None if Zm is None else Zm[m]
    out["pearson"], out["spearman"] = pearson(xs, ys), spearman(xs, ys)
    out["pearson_disattenuated"] = attenuation_correct(out["pearson"], rel_x, rel_y, floor)
    if Zs is not None:
        out["partial_spearman"] = partial_spearman(xs, ys, Zs)
    if n >= MIN_CI_ATHLETES:
        idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
        rs = np.array([spearman(xs[i], ys[i]) for i in idx])
        rs = rs[np.isfinite(rs)]
        if rs.size >= 20:
            out["spearman_lo"], out["spearman_hi"] = float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))
        if Zs is not None:
            ps = np.array([partial_spearman(xs[i], ys[i], Zs[i]) for i in idx[: max(200, n_boot // 5)]])
            ps = ps[np.isfinite(ps)]
            if ps.size >= 20:
                out["partial_lo"], out["partial_hi"] = float(np.quantile(ps, 0.025)), float(np.quantile(ps, 0.975))
    if positions is not None:
        p = np.asarray(positions)[m]
        rr = [spearman(xs[p == g], ys[p == g]) for g in np.unique(p) if (p == g).sum() >= min_pos]
        rr = [r for r in rr if np.isfinite(r)]
        out["within_position_spearman"] = float(np.median(rr)) if rr else NAN
    return out


def association_reading(row: dict[str, Any]) -> str:
    lo, hi = row.get("spearman_lo"), row.get("spearman_hi")
    if lo is None or not np.isfinite(lo):
        return "too few athletes for an interval"
    if lo > 0 or hi < 0:
        return "association (interval excludes 0) — descriptive"
    return f"inconclusive (|r| below ≈ {row.get('mde_r', NAN):.2f} is not detectable at this n)"


__all__ = ["CONTEXT", "Design", "LeakageError", "MIN_CI_ATHLETES", "OFFICIAL_CORE", "assert_grouped", "assign_folds", "association", "association_reading",
           "choose_alpha", "cluster_bootstrap_pairs",
           "control_rows", "design", "fisher_ci", "holm", "icc_oneway", "ladder_columns", "make_folds", "mde_correlation", "oof_predictions",
           "paired_compare", "partial_spearman", "pearson", "permutation_null", "random_projection_control", "repeatability_table", "residualize",
           "rmse", "run_ladder", "spearman", "split_half", "stopwatch_redundancy", "verdict"]
