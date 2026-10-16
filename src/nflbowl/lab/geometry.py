"""Part C — what the representations encode: basic geometry, collapse, domain geometry, exploratory projections.

All quantities are computed on the HELD-OUT rows of an outer fold, embedded by the encoder that never saw those
athletes (inductive regime; F0 and the random encoder have nothing fitted). Domain metrics reuse
``evaluation.domain_diagnostics`` (athlete-grouped domain AUC, mean gap, alignment, collapse) and the Milestone 11
``rep.domain_shift`` (ridge domain AUC + RBF MMD² with the median heuristic, the estimator behind the transductive
MMD² values) so numbers are comparable with earlier tables.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..models import registry as reg

NAN = float("nan")


_CACHE: dict[tuple[str, str, int, str], np.ndarray] = {}
_CACHE_MAX = 32


def inductive_matrix(exp: Any, name: str, fold: int) -> np.ndarray:
    """Embedding matrix (N × d, rows aligned with the bundle meta) of the fold-``fold`` inductive representation.
    F0 = the cached engineered features; random / encoders = saved inductive embeddings (computed once if missing,
    which needs the trained checkpoint for encoders). Kept in a small in-memory cache keyed by experiment, model, fold
    and the registered weights checksum (a retrained encoder gets a new key)."""
    n = reg.resolve(name)
    if n == reg.ENGINEERED:
        return exp.features()
    rec = exp.manifest.find("training", f"{n}__inductive_f{fold}")
    key = (str(exp.root), n, int(fold), str((rec or {}).get("sha256")))
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = exp.embed(n, regime="inductive", folds=[fold], reuse=True)[fold]
    return _CACHE[key]


def held_out_rows(exp: Any, fold: int) -> np.ndarray:
    return np.sort(exp.splits(fold)[2])


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def basic_geometry(E: np.ndarray, meta: Any, rows: np.ndarray, seed: int, max_rows: int = 3000, k_nn: int = 10) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Scalar summaries + arrays for distributions (spectrum shares, per-dim variance, pairwise cosines, NN distances,
    same-domain share of k nearest neighbours)."""
    from ..evaluation import domain_diagnostics as dd

    rows = rows[np.isfinite(E[rows]).all(1)]
    Z = E[rows].astype(float)
    dom = meta["domain"].to_numpy()[rows]
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"dims": int(Z.shape[1]), "n_rows": int(Z.shape[0])}
    out.update(dd.collapse_indicators(E, rows, seed=seed))
    var = Z.var(0)
    out["max_abs_dim_corr"] = NAN
    if Z.shape[0] > 3 and Z.shape[1] > 1:
        sd = Z.std(0)
        live = sd > 1e-9
        if live.sum() > 1:
            Cc = np.corrcoef(Z[:, live].T)
            np.fill_diagonal(Cc, 0.0)
            out["max_abs_dim_corr"] = float(np.nanmax(np.abs(Cc)))
            out["mean_abs_dim_corr"] = float(np.nanmean(np.abs(Cc[np.triu_indices(Cc.shape[0], 1)])))
    lam = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    share = lam / lam.sum() if lam.sum() > 0 else lam
    nrm = np.linalg.norm(Z, axis=1)
    out.update({"norm_median": float(np.median(nrm)), "norm_cv": float(nrm.std() / nrm.mean()) if nrm.mean() > 0 else NAN})
    pick = np.sort(rng.choice(Z.shape[0], min(max_rows, Z.shape[0]), replace=False))
    U = _unit(Z[pick])
    S = U @ U.T
    iu = np.triu_indices(S.shape[0], 1)
    cosines = S[iu]
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)[:, :k_nn]
    nn_dist = 1.0 - S[np.arange(S.shape[0]), order[:, 0]]
    dp = dom[pick]
    same_share = (dp[order] == dp[:, None]).mean(1)
    base = np.array([(dp == d).mean() for d in dp])
    out.update({"knn_same_domain_share": float(same_share.mean()), "knn_same_domain_base_rate": float(base.mean()),
                "knn_domain_mixing_excess": float(same_share.mean() - base.mean()), "cosine_mean": float(cosines.mean()), "cosine_sd": float(cosines.std()),
                "nn_cosine_distance_median": float(np.median(nn_dist))})
    arrays = {"spectrum_share": share[: min(64, share.size)], "dim_variance": var, "cosines": cosines[rng.choice(cosines.size, min(20000, cosines.size), replace=False)],
              "nn_distance": nn_dist, "norms": nrm, "norm_domain": dom}
    return out, arrays


def domain_geometry(name: str, E: np.ndarray, meta: Any, rows: np.ndarray, fold: int, seed: int) -> tuple[Any, dict[str, Any]]:
    """(long diagnostics table from ``domain_diagnostics``, scalar dict with MMD² (rep.domain_shift on held-out rows) and
    the position-adjusted domain AUC = athlete-weighted mean of within-position domain AUCs)."""
    import polars as pl

    from .. import _backend
    from ..evaluation import domain_diagnostics as dd

    long = dd.domain_diagnostics(name, E, meta, rows, fold, seed=seed)
    sub = meta[rows]
    ds = _backend.rep().domain_shift(E[rows], sub, seed=seed)
    pos = long.filter((pl.col("metric") == "domain_auc") & (pl.col("position_group") != "ALL") & pl.col("value").is_not_nan())
    nat = long.filter((pl.col("metric") == "domain_auc_n_athletes") & (pl.col("position_group") != "ALL")).select("position_group", pl.col("value").alias("w"))
    j = pos.join(nat, on="position_group", how="inner")
    padj = float(np.average(j["value"].to_numpy(), weights=j["w"].to_numpy())) if j.height and j["w"].sum() > 0 else NAN
    overall = long.filter((pl.col("metric") == "domain_auc") & (pl.col("position_group") == "ALL"))["value"]
    return long, {"domain_auc_grouped": float(overall[0]) if overall.len() else NAN, "domain_auc_m11": ds["domain_auc"], "mmd2": ds["mmd2"],
                  "position_adjusted_domain_auc": padj}


def athlete_pair_similarity(E: np.ndarray, meta: Any, rows: np.ndarray) -> Any:
    """Per held-out athlete with both domains: cosine(own Combine mean, own NFL mean) and the mean cosine to OTHER
    same-position athletes' NFL means (the retrieval contrast, as a distribution)."""
    import polars as pl

    from ..evaluation.retrieval import CrossDomainIndex

    rows = rows[np.isfinite(E[rows]).all(1)]
    sub = meta[rows]
    ix = CrossDomainIndex(E[rows], sub)
    if len(ix.both) < 3:
        return pl.DataFrame()
    S = _unit(ix.combine_means()) @ _unit(ix.nfl_means(ix.both)).T
    P = ix.positions()
    out = []
    for i, a in enumerate(ix.both):
        same = (P == P[i])
        same[i] = False
        out.append({"nfl_id": int(a), "position_group": P[i], "own_cosine": float(S[i, i]),
                    "other_same_position_cosine": float(S[i, same].mean()) if same.any() else NAN, "n_other": int(same.sum())})
    return pl.DataFrame(out, infer_schema_length=None)


def run_geometry(exp: Any, models_folds: dict[str, list[int]], seed: int) -> dict[str, Any]:
    """Every (representation, fold) available: summaries (long), spectra, pair similarities, domain diagnostics."""
    import polars as pl

    meta = exp.bundle.meta
    summ, spec, pairs, doms, norms = [], [], [], [], []
    for name, fl in models_folds.items():
        for f in fl:
            E = inductive_matrix(exp, name, f)
            rows = held_out_rows(exp, f)
            g, arr = basic_geometry(E, meta, rows, seed)
            long, dg = domain_geometry(name, E, meta, rows, f, seed)
            for k, v in {**g, **dg}.items():
                summ.append({"representation": name, "fold": int(f), "metric": k, "value": NAN if v is None else float(v)})
            spec.append(pl.DataFrame({"representation": name, "fold": int(f), "component": np.arange(1, arr["spectrum_share"].size + 1),
                                      "variance_share": arr["spectrum_share"]}))
            norms.append(pl.DataFrame({"representation": name, "fold": int(f), "domain": arr["norm_domain"].astype(str), "norm": arr["norms"]}))
            pp = athlete_pair_similarity(E, meta, rows)
            if pp.height:
                pairs.append(pp.with_columns(pl.lit(name).alias("representation"), pl.lit(int(f)).alias("fold")))
            doms.append(long)
    cat = (lambda xs: pl.concat(xs, how="diagonal_relaxed") if xs else pl.DataFrame())   # noqa: E731
    summary = pl.DataFrame(summ, infer_schema_length=None) if summ else pl.DataFrame()
    wide = (summary.group_by("representation", "metric").agg(pl.col("value").mean().alias("mean"), pl.col("value").min().alias("min"),
                                                              pl.col("value").max().alias("max"), pl.len().alias("n_folds")).sort("representation", "metric")
            if summary.height else summary)
    return {"summary_long": summary, "summary": wide, "spectrum": cat(spec), "norms": cat(norms), "pairs": cat(pairs), "domain_long": cat(doms)}


def collapse_flags(summary: Any) -> Any:
    """Declared collapse rule per representation (mean over folds): effective rank < 2, or > 20 % near-constant
    dimensions, or mean pairwise cosine > 0.9."""
    import polars as pl

    if summary is None or not summary.height:
        return pl.DataFrame()
    w = summary.pivot(on="metric", index="representation", values="mean")
    get = (lambda c: pl.col(c) if c in w.columns else pl.lit(None, dtype=pl.Float64))  # noqa: E731
    return w.select("representation", get("effective_rank").alias("effective_rank"), get("n_dead_dims").alias("n_dead_dims"), get("dims").alias("dims"),
                    get("mean_pairwise_cosine").alias("mean_pairwise_cosine")).with_columns(
        ((pl.col("effective_rank") < 2) | (pl.col("n_dead_dims") > 0.2 * pl.col("dims")) | (pl.col("mean_pairwise_cosine") > 0.9)).alias("collapse_flag"))


def projection_coordinates(exp: Any, name: str, fold: int, seed: int, max_points: int = 6000, use_umap: bool = True) -> Any:
    """2-D PCA (and UMAP when importable) coordinates of a stratified (domain-balanced) seeded sample of the held-out rows
    of ``fold``; the transform is fitted on exactly the plotted sample (stated in the table)."""
    import polars as pl

    meta = exp.bundle.meta
    E = inductive_matrix(exp, name, fold)
    rows = held_out_rows(exp, fold)
    rows = rows[np.isfinite(E[rows]).all(1)]
    dom = meta["domain"].to_numpy()[rows]
    rng = np.random.default_rng(seed)
    pick = np.concatenate([rng.choice(rows[dom == d], min(max_points // 2, int((dom == d).sum())), replace=False) for d in ("combine", "nfl") if (dom == d).any()])
    pick = np.sort(pick)
    Z = E[pick].astype(float)
    Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-9)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    P = Z @ vt[:2].T
    sub = meta[pick].select("idx", "nfl_id", "domain", "kind", "position_group")
    out = [sub.with_columns(pl.Series("x", P[:, 0]), pl.Series("y", P[:, 1]), pl.lit("pca").alias("method"))]
    if use_umap:
        try:
            import umap  # type: ignore[import-not-found]

            U = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.2).fit_transform(Z)
            out.append(sub.with_columns(pl.Series("x", U[:, 0]), pl.Series("y", U[:, 1]), pl.lit("umap").alias("method")))
        except ImportError:
            pass
    return pl.concat(out, how="vertical").with_columns(pl.lit(name).alias("representation"), pl.lit(int(fold)).alias("fold"),
                                                       pl.lit(f"fit on this seeded domain-balanced sample of fold-{fold} held-out rows (seed {seed}), z-scored").alias("fit_scope"))


__all__ = ["athlete_pair_similarity", "basic_geometry", "collapse_flags", "domain_geometry", "held_out_rows", "inductive_matrix", "projection_coordinates",
           "run_geometry"]
