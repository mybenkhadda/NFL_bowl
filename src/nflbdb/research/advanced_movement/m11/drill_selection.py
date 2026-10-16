"""11G — optimal Combine drill selection.

Objectives are reported SEPARATELY (no scouting utility function is invented):
    prediction      held-out Δ RMSE vs official tests only, on a COMMON cohort of athletes having every compared drill
    repeatability   median ICC of the family's features (11A / 11C repeatability tables)
    redundancy      median within-position R² of the family's features from the official tests
    coverage        NFL demands with declared overlap (≥ medium) whose 11.0 cell is SUPPORTED for the position
    resilience      share of the position's Combine athletes who have the drill tracked (participation)

Complexity control: every family enters a model as ``family_components`` principal components fitted on the TRAINING
athletes of the outer fold (an 80-feature drill and a 3-feature drill cost the same number of terms). Selection of the
order in which families are added happens INSIDE each outer fold (inner CV on training athletes): the held-out curve
"official + k drills" is honest.

Non-forty participation is nonrandom (position, invitation, injury, choice). ``participation_model`` quantifies it; a
complete-case gain is never generalised to all prospects.

Drill costs are DECLARED HYPOTHETICAL minutes (``HYPOTHETICAL_COST_MIN``), not measurements.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np
import polars as pl

from . import evaluation as ev
from .config import M11Config, SUPPORTED

NAN = float("nan")
HYPOTHETICAL_COST_MIN = {"forty": 2.0, "short_shuttle": 3.0, "three_cone": 3.0, "long_shuttle": 4.0, "gauntlet": 3.0, "backpedal_transition": 4.0,
                         "pass_rush": 5.0, "other": 4.0}


def family_columns(columns: Sequence[str], families: Sequence[str]) -> dict[str, list[str]]:
    skip = ("_censored", "_identifiable", "_grid_edge", "censored_share", "sprint_rmse", "phase_rmse", "sprint_r2")
    return {f: sorted(k for k in columns if k.startswith(f"{f}.") and not k.endswith(skip)) for f in families}


def pca_scores(X: np.ndarray, train: np.ndarray, k: int) -> np.ndarray:
    """Scores of every row on the first k principal components of the TRAINING rows (median imputation, z-scaling, all
    from training rows). Rows with every value missing get 0 (the training mean)."""
    Xt = X[train]
    med = np.nanmedian(Xt, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med)
    mu, sd = Xi[train].mean(0), Xi[train].std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (Xi - mu) / sd
    _, _, vt = np.linalg.svd(Z[train], full_matrices=False)
    return Z @ vt[: min(k, vt.shape[0])].T


def _fold_frame(data: pl.DataFrame, fams: dict[str, list[str]], chosen: Sequence[str], train: np.ndarray, k: int, rng_noise: np.ndarray | None = None
                ) -> tuple[pl.DataFrame, list[str]]:
    cols, extra = [], {}
    for f in chosen:
        if f == "_random":
            for j in range(rng_noise.shape[1]):
                extra[f"_rnd{j}"] = rng_noise[:, j]
                cols.append(f"_rnd{j}")
            continue
        X = np.column_stack([data[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in fams[f]]) if fams[f] else np.empty((data.height, 0))
        if X.shape[1] == 0:
            continue
        S = pca_scores(X, train, k)
        for j in range(S.shape[1]):
            extra[f"_pc_{f}_{j}"] = S[:, j]
            cols.append(f"_pc_{f}_{j}")
    return data.with_columns([pl.Series(n, v) for n, v in extra.items()]), cols


def oof_families(data: pl.DataFrame, target: str, base: Sequence[str], fams: dict[str, list[str]], chosen_by_fold: dict[int, Sequence[str]], folds: np.ndarray,
                 c: M11Config, seed: int, noise_k: int = 0) -> np.ndarray:
    """Held-out predictions of base + the families chosen FOR EACH FOLD (PCA per fold)."""
    y = data[target].cast(pl.Float64).to_numpy()
    pred = np.full(data.height, NAN)
    noise = np.random.default_rng(seed + 99).standard_normal((data.height, noise_k)) if noise_k else None
    for f in sorted(set(folds.tolist())):
        test, train = folds == f, folds != f
        d, cols = _fold_frame(data, fams, chosen_by_fold.get(f, ()), train, c.family_components, noise)
        a = ev.choose_alpha(d, y, [*base, *cols], train, c.alphas, c.inner_folds, seed + f)
        pred[test] = ev._fit(d, y, [*base, *cols], train, test, a)
    return pred


def inner_forward_order(data: pl.DataFrame, target: str, base: Sequence[str], fams: dict[str, list[str]], train: np.ndarray, c: M11Config, seed: int) -> list[str]:
    """Greedy order of families by inner-CV error on the TRAINING athletes only."""
    idx = np.flatnonzero(train)
    sub = data[idx.tolist()]
    inner = np.random.default_rng(seed).permutation(idx.size) % c.inner_folds
    y = sub[target].cast(pl.Float64).to_numpy()
    order, left = [], [f for f in fams if fams[f]]
    while left:
        best, best_e = None, np.inf
        for f in left:
            p = oof_families(sub, target, base, fams, {k: [*order, f] for k in range(c.inner_folds)}, inner, c, seed)
            e = float(np.nanmean((y - p) ** 2))
            if e < best_e:
                best, best_e = f, e
        order.append(best)
        left.remove(best)
    return order


def common_cohort(data: pl.DataFrame, target: str, fams: dict[str, list[str]]) -> pl.DataFrame:
    conds = [pl.col(target).is_not_null()] + [pl.any_horizontal([pl.col(k).is_not_null() for k in cols]) for cols in fams.values() if cols]
    return data.filter(pl.all_horizontal(conds))


def selection_experiment(data: pl.DataFrame, target: str, fams: dict[str, list[str]], folds: np.ndarray, c: M11Config, seed: int) -> dict[str, pl.DataFrame]:
    """On the common cohort: marginal value of each family, a random family of equal size, the nested forward-selection curve."""
    base = [*ev.CONTEXT, *[k for k in ev.OFFICIAL_CORE if k in data.columns]]
    y = data[target].cast(pl.Float64).to_numpy()
    folds_u = sorted(set(folds.tolist()))
    p0 = oof_families(data, target, base, fams, {}, folds, c, seed)
    marg = []
    for f in [*fams, "_random"]:
        chosen = {k: [f] for k in folds_u}
        p = oof_families(data, target, base, fams, chosen, folds, c, seed, noise_k=c.family_components if f == "_random" else 0)
        r = ev.paired_compare(y, p0, p, c.n_boot, seed)
        marg.append({"target": target, "family": f, **r, "verdict": ev.verdict(r)})
    orders = {}
    for k in folds_u:
        orders[k] = inner_forward_order(data, target, base, fams, folds != k, c, seed + k)
    curve = []
    for m in range(1, len(fams) + 1):
        p = oof_families(data, target, base, fams, {k: orders[k][:m] for k in folds_u}, folds, c, seed)
        r = ev.paired_compare(y, p0, p, c.n_boot, seed)
        curve.append({"target": target, "n_added_families": m, **r, "verdict": ev.verdict(r)})
    freq = pl.DataFrame([{"target": target, "fold": k, "rank": i + 1, "family": f} for k, o in orders.items() for i, f in enumerate(o)])
    return {"marginal": pl.DataFrame(marg, infer_schema_length=None), "curve": pl.DataFrame(curve, infer_schema_length=None), "orders": freq}


# ============================================================================== missingness / participation
def logistic_irls(X: np.ndarray, y: np.ndarray, iters: int = 50, ridge: float = 1e-4) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * beta)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def participation_model(registry: pl.DataFrame, attempt_athletes: pl.DataFrame, families: Sequence[str]) -> dict[str, pl.DataFrame]:
    """Who has each drill tracked: rates by position and by official-forty quartile within position, and a logistic model
    (position + standardised official forty + draft year). Descriptive: it shows HOW nonrandom participation is."""
    a = registry.select("nfl_id", "position_group", "draft_year", pl.col("cmb_forty").alias("forty") if "cmb_forty" in registry.columns else pl.col("forty"))
    a = a.with_columns(((pl.col("forty") - pl.col("forty").mean().over("position_group")) / pl.col("forty").std().over("position_group")).alias("forty_z"))
    a = a.with_columns(pl.col("forty_z").qcut(4, labels=["Q1 fastest", "Q2", "Q3", "Q4 slowest"], allow_duplicates=True).cast(pl.String).alias("forty_quartile"))
    rates, coefs = [], []
    for f in families:
        has = set(attempt_athletes.filter(pl.col("family") == f)["nfl_id"].to_list())
        d = a.with_columns(pl.col("nfl_id").is_in(list(has)).alias("has"))
        for col in ("position_group", "forty_quartile"):
            rates.append(d.group_by(col).agg(pl.len().alias("n"), pl.col("has").mean().alias("rate")).rename({col: "level"})
                         .with_columns(pl.lit(f).alias("family"), pl.lit(col).alias("by")))
        dd = d.drop_nulls(["forty_z", "draft_year", "position_group"])
        if dd.height < 30 or dd["has"].sum() < 5 or dd["has"].sum() > dd.height - 5:
            coefs.append({"family": f, "term": "(not estimable)", "coef": NAN, "n": dd.height})
            continue
        pos = dd["position_group"].to_numpy()
        lv = sorted(set(pos.tolist()))
        X = np.column_stack([np.ones(dd.height), dd["forty_z"].to_numpy(), (dd["draft_year"].to_numpy() - dd["draft_year"].mean()),
                             *[(pos == l).astype(float) for l in lv[1:]]])
        b = logistic_irls(X, dd["has"].cast(pl.Float64).to_numpy())
        names = ["intercept", "forty_z (within position)", "draft_year (centred)", *[f"pos={l}" for l in lv[1:]]]
        coefs += [{"family": f, "term": n, "coef": float(v), "n": dd.height} for n, v in zip(names, b)]
    return {"rates": pl.concat(rates, how="diagonal_relaxed") if rates else pl.DataFrame(), "coefficients": pl.DataFrame(coefs)}


# ============================================================================== objectives and frontier
def objectives_table(families: Sequence[str], marginal: pl.DataFrame, repeat: pl.DataFrame, redundancy: pl.DataFrame, feas: pl.DataFrame,
                     participation: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for f in families:
        m = marginal.filter(pl.col("family") == f) if marginal.height else marginal
        rep = repeat.filter(pl.col("feature").str.starts_with(f"{f}.")) if repeat.height else repeat
        red = redundancy.filter(pl.col("metric").str.starts_with(f"{f}.")) if redundancy.height and "metric" in redundancy.columns else pl.DataFrame()
        cov = feas.filter((pl.col("family") == f) & (pl.col("feasibility") == SUPPORTED) & pl.col("overlap_confidence").is_in(["high", "medium"])) if feas.height else feas
        part = participation.filter((pl.col("family") == f) & (pl.col("by") == "position_group")) if participation.height else participation
        rows.append({"family": f, "prediction_delta_rmse_median": float(m["delta_rmse"].median()) if m.height else NAN,
                     "prediction_n_targets_model_better": int((m["verdict"] == "model_better").sum()) if m.height else 0,
                     "repeatability_median_icc": float(rep["icc"].median()) if rep.height else NAN,
                     "redundancy_median_r2_official": float(red["r2_official"].median()) if red.height else NAN,
                     "coverage_supported_demand_position_cells": cov.height if cov is not None else 0,
                     "participation_min_position_rate": float(part["rate"].min()) if part.height else NAN,
                     "hypothetical_cost_min": HYPOTHETICAL_COST_MIN.get(f, NAN)})
    return pl.DataFrame(rows)


def coverage_frontier(feas: pl.DataFrame, families: Sequence[str]) -> pl.DataFrame:
    """Every subset (always containing the forty when available): hypothetical cost vs distinct SUPPORTED (demand × position)
    cells covered with ≥ medium overlap; ``pareto`` marks subsets no cheaper subset matches."""
    cells = {f: set(map(tuple, feas.filter((pl.col("family") == f) & (pl.col("feasibility") == SUPPORTED) & pl.col("overlap_confidence").is_in(["high", "medium"]))
                        .select("demand_id", "position_group").rows())) for f in families}
    others = [f for f in families if f != "forty"]
    rows = []
    for r in range(0, len(others) + 1):
        for sub in combinations(others, r):
            s = (("forty",) if "forty" in families else ()) + sub
            cov = set().union(*[cells[f] for f in s]) if s else set()
            rows.append({"subset": " + ".join(s) or "(none)", "n_drills": len(s), "hypothetical_cost_min": sum(HYPOTHETICAL_COST_MIN.get(f, 4.0) for f in s),
                         "covered_cells": len(cov)})
    out = pl.DataFrame(rows).sort("hypothetical_cost_min", "covered_cells", descending=[False, True])
    best, pareto = -1, []
    for r in out.iter_rows(named=True):
        pareto.append(r["covered_cells"] > best)
        best = max(best, r["covered_cells"])
    return out.with_columns(pl.Series("pareto", pareto))


def position_recommendations(objectives_by_pos: pl.DataFrame) -> pl.DataFrame:
    """Per position: families ordered by each objective separately (no combined score)."""
    rows = []
    for (pos,), g in sorted(objectives_by_pos.partition_by(["position_group"], as_dict=True).items()):
        for obj, desc in (("prediction_delta_rmse_median", True), ("repeatability_median_icc", True), ("coverage_supported_demand_position_cells", True)):
            if obj in g.columns:
                o = g.drop_nulls(obj).sort(obj, descending=desc)
                rows.append({"position_group": pos, "objective": obj, "order": " > ".join(o["family"].to_list())})
    return pl.DataFrame(rows)


__all__ = ["HYPOTHETICAL_COST_MIN", "common_cohort", "coverage_frontier", "family_columns", "inner_forward_order", "logistic_irls", "objectives_table",
           "oof_families", "participation_model", "pca_scores", "position_recommendations", "selection_experiment"]
