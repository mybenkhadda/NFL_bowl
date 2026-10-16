"""Domain-leakage and representation-health diagnostics (label-free, reusable on any embedding matrix).

Every function takes an embedding matrix ``E`` aligned with ``meta`` and a row subset ``rows`` (for the inductive
comparison: the held-out rows of one outer fold, embedded by the encoder that never saw those athletes).

``grouped_domain_auc``     ridge domain discriminant, cross-validated over ATHLETE groups (an athlete's Combine and NFL
                           rows always fall in the same inner fold, so the classifier cannot exploit identity); the
                           standardisation is fitted on the inner training rows only. AUC with average-rank ties.
                           0.5 = domains not linearly separable, 1 = fully separable. Overall and within position.
``embedding_summary``      per-domain norms (mean / sd / quantiles), per-dimension spread, non-finite share.
``domain_mean_gap``        ‖μ_combine − μ_nfl‖, cosine(μ_combine, μ_nfl), RMS standardised mean difference, norm ratio.
``cross_domain_alignment`` athletes with both domains: mean cosine(own Combine mean, own NFL mean) minus mean cosine to
                           other same-position athletes' NFL means — raw and after removing each domain's mean.
``collapse_indicators``    effective rank (exp spectral entropy), participation ratio, PC1 variance share, near-constant
                           dimensions, mean pairwise cosine.
``positive_pair_coverage`` share of TRAINING sequences that have at least one other-domain training sequence with the same
                           alignment label (state / category) — the anchors the supervised-contrastive alignment term can
                           use (a set-level upper bound; per-batch coverage is lower).

``domain_diagnostics`` stacks them into one long table: representation · fold · position_group · domain · metric · value · n.
Domain separability is not failure on its own — Combine drills and NFL windows are different tasks.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")
ALL = "ALL"


def rank_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC of ``score`` for ``y == True`` with average ranks for ties."""
    s, y = np.asarray(score, float), np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return NAN
    order = np.argsort(s, kind="mergesort")
    _, first, counts = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.empty(s.size)
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _finite_rows(E: np.ndarray, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, np.int64)
    return rows[np.isfinite(E[rows]).all(1)] if rows.size else rows


def grouped_domain_auc(E: np.ndarray, meta: Any, rows: np.ndarray, k: int = 5, ridge: float = 1.0, seed: int = 0, max_per_domain: int = 2000,
                       min_per_domain: int = 20) -> dict[str, Any]:
    rows = _finite_rows(E, rows)
    dom = meta["domain"].to_numpy()[rows]
    rng = np.random.default_rng(seed)
    a, b = rows[dom == "combine"], rows[dom == "nfl"]
    if a.size < min_per_domain or b.size < min_per_domain:
        return {"domain_auc": NAN, "n_combine": int(a.size), "n_nfl": int(b.size), "n_athletes": 0}
    a = np.sort(rng.choice(a, min(max_per_domain, a.size), replace=False))
    b = np.sort(rng.choice(b, min(max_per_domain, b.size), replace=False))
    r = np.r_[a, b]
    y = np.r_[np.zeros(a.size), np.ones(b.size)]
    ids = meta["nfl_id"].to_numpy()[r]
    ath = np.unique(ids)
    if ath.size < k:
        return {"domain_auc": NAN, "n_combine": int(a.size), "n_nfl": int(b.size), "n_athletes": int(ath.size)}
    inner = dict(zip(rng.permutation(ath).tolist(), (np.arange(ath.size) % k).tolist()))
    f = np.array([inner[i] for i in ids.tolist()])
    Z = E[r].astype(float)
    score = np.full(y.size, NAN)
    for g in range(k):
        tr, te = f != g, f == g
        if not te.any() or len(np.unique(y[tr])) < 2:
            continue
        mu, sd = Z[tr].mean(0), Z[tr].std(0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        Ztr, Zte = (Z[tr] - mu) / sd, (Z[te] - mu) / sd
        w = np.linalg.solve(Ztr.T @ Ztr + ridge * np.eye(Z.shape[1]), Ztr.T @ (y[tr] - y[tr].mean()))
        score[te] = Zte @ w
    ok = np.isfinite(score)
    return {"domain_auc": rank_auc(score[ok], y[ok].astype(bool)), "n_combine": int(a.size), "n_nfl": int(b.size), "n_athletes": int(ath.size)}


def embedding_summary(E: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(rows, np.int64)
    if not rows.size:
        return {"n": 0}
    Z = E[rows].astype(float)
    fin = np.isfinite(Z).all(1)
    Z = Z[fin]
    if not Z.shape[0]:
        return {"n": int(rows.size), "nonfinite_share": 1.0}
    nrm = np.linalg.norm(Z, axis=1)
    q05, q50, q95 = np.quantile(nrm, [0.05, 0.5, 0.95])
    return {"n": int(rows.size), "nonfinite_share": float(1 - fin.mean()), "norm_mean": float(nrm.mean()), "norm_sd": float(nrm.std()),
            "norm_q05": float(q05), "norm_q50": float(q50), "norm_q95": float(q95), "dim_sd_mean": float(Z.std(0).mean())}


def domain_mean_gap(E: np.ndarray, meta: Any, rows: np.ndarray) -> dict[str, Any]:
    rows = _finite_rows(E, rows)
    dom = meta["domain"].to_numpy()[rows]
    A, B = E[rows[dom == "combine"]].astype(float), E[rows[dom == "nfl"]].astype(float)
    if A.shape[0] < 2 or B.shape[0] < 2:
        return {}
    ma, mb = A.mean(0), B.mean(0)
    pooled = np.sqrt(((A.shape[0] - 1) * A.var(0, ddof=1) + (B.shape[0] - 1) * B.var(0, ddof=1)) / (A.shape[0] + B.shape[0] - 2))
    pooled = np.where(pooled > 1e-12, pooled, np.nan)
    den = np.linalg.norm(ma) * np.linalg.norm(mb)
    na, nb = np.linalg.norm(A, axis=1).mean(), np.linalg.norm(B, axis=1).mean()
    return {"mean_gap_norm": float(np.linalg.norm(ma - mb)), "mean_cosine": float(ma @ mb / den) if den > 0 else NAN,
            "standardized_mean_gap_rms": float(np.sqrt(np.nanmean(((ma - mb) / pooled) ** 2))) if np.isfinite(pooled).any() else NAN,
            "norm_ratio_nfl_over_combine": float(nb / na) if na > 0 else NAN}


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def cross_domain_alignment(E: np.ndarray, meta: Any, rows: np.ndarray) -> dict[str, Any]:
    from .retrieval import CrossDomainIndex

    rows = _finite_rows(E, rows)
    if not rows.size:
        return {"n_athletes": 0}
    sub = meta[rows]
    Z = E[rows].astype(float)
    dom = sub["domain"].to_numpy()
    out: dict[str, Any] = {}
    for tag, Zc in (("raw", Z), ("centered", Z - np.where((dom == "combine")[:, None], Z[dom == "combine"].mean(0) if (dom == "combine").any() else 0.0,
                                                          Z[dom == "nfl"].mean(0) if (dom == "nfl").any() else 0.0))):
        ix = CrossDomainIndex(Zc, sub)
        if len(ix.both) < 3:
            return {"n_athletes": len(ix.both)}
        S = _unit(ix.combine_means()) @ _unit(ix.nfl_means(ix.both)).T
        P = ix.positions()
        same = (P[:, None] == P[None, :]) & ~np.eye(P.size, dtype=bool)
        own = np.diagonal(S)
        other = np.array([S[i, same[i]].mean() if same[i].any() else np.nan for i in range(P.size)])
        ok = np.isfinite(other)
        out.update({f"own_cosine_{tag}": float(own.mean()), f"other_same_position_cosine_{tag}": float(other[ok].mean()) if ok.any() else NAN,
                    f"alignment_gap_{tag}": float((own[ok] - other[ok]).mean()) if ok.any() else NAN})
        out["n_athletes"] = len(ix.both)
    return out


def collapse_indicators(E: np.ndarray, rows: np.ndarray, max_pairs_rows: int = 1000, seed: int = 0) -> dict[str, Any]:
    rows = _finite_rows(E, rows)
    if rows.size < 3:
        return {}
    Z = E[rows].astype(float)
    sd = Z.std(0)
    lam = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    tot = lam.sum()
    if tot <= 0:
        return {"effective_rank": 0.0, "participation_ratio": 0.0, "pc1_share": NAN, "n_dead_dims": int(Z.shape[1]), "dims": int(Z.shape[1]),
                "mean_pairwise_cosine": 1.0}
    p = lam / tot
    p = p[p > 0]
    rng = np.random.default_rng(seed)
    U = _unit(Z[np.sort(rng.choice(Z.shape[0], min(max_pairs_rows, Z.shape[0]), replace=False))])
    C = U @ U.T
    iu = np.triu_indices(C.shape[0], 1)
    return {"effective_rank": float(np.exp(-(p * np.log(p)).sum())), "participation_ratio": float(tot ** 2 / (lam ** 2).sum()), "pc1_share": float(lam[0] / tot),
            "n_dead_dims": int((sd < 1e-6).sum()), "dims": int(Z.shape[1]), "mean_pairwise_cosine": float(C[iu].mean()) if iu[0].size else NAN}


def positive_pair_coverage(labels: np.ndarray, meta: Any, rows: np.ndarray, label_name: str) -> list[dict[str, Any]]:
    """Rows: domain × position_group (+ ALL) with the share of ``rows`` (training rows) that have ≥ 1 other-domain row in
    ``rows`` with the same non-negative label."""
    rows = np.asarray(rows, np.int64)
    lab = np.asarray(labels)[rows]
    dom = meta["domain"].to_numpy()[rows]
    pos = np.array(meta["position_group"].to_list(), dtype=object)[rows]
    sets = {d: set(lab[(dom == d) & (lab >= 0)].tolist()) for d in ("combine", "nfl")}
    other = {"combine": "nfl", "nfl": "combine"}
    covered = np.array([(l_ >= 0) and (l_ in sets[other[d]]) for l_, d in zip(lab.tolist(), dom.tolist())], dtype=bool) if rows.size else np.zeros(0, bool)
    out = []
    for p in [ALL, *sorted({str(v) for v in pos.tolist()})]:
        pm = np.ones(rows.size, bool) if p == ALL else np.array([str(v) == p for v in pos.tolist()], dtype=bool)
        for d in ("combine", "nfl"):
            m = pm & (dom == d)
            out.append({"position_group": p, "domain": d, "metric": f"positive_pair_coverage_{label_name}", "value": float(covered[m].mean()) if m.any() else NAN,
                        "n": int(m.sum())})
    return out


def domain_diagnostics(representation: str, E: np.ndarray, meta: Any, rows: np.ndarray, fold: int | None = None, seed: int = 0,
                       by_position: bool = True, min_per_domain: int = 20) -> Any:
    """Long table of every diagnostic above for ``rows`` (overall and, for domain AUC / mean gap / alignment, per position)."""
    import polars as pl

    rows = np.asarray(rows, np.int64)
    dom = meta["domain"].to_numpy()
    pos = np.array(meta["position_group"].to_list(), dtype=object)
    out: list[dict[str, Any]] = []

    def put(p: str, d: str, metrics: dict[str, Any], n_key: str | None = None, n: int | None = None) -> None:
        nn = int(metrics.get(n_key, n or 0)) if n_key else int(n or 0)
        for k_, v in metrics.items():
            if k_ not in ("n", "n_combine", "n_nfl", "n_athletes"):
                out.append({"position_group": p, "domain": d, "metric": k_, "value": NAN if v is None else float(v), "n": nn})

    scopes: list[tuple[str, np.ndarray]] = [(ALL, rows)]
    if by_position:
        scopes += [(str(p), rows[pos[rows] == p]) for p in sorted({v for v in pos[rows].tolist() if v is not None}, key=str)]
    for p, r in scopes:
        da = grouped_domain_auc(E, meta, r, seed=seed, min_per_domain=min_per_domain)
        put(p, "both", {"domain_auc": da["domain_auc"]}, n=da["n_combine"] + da["n_nfl"])
        out.append({"position_group": p, "domain": "both", "metric": "domain_auc_n_athletes", "value": float(da["n_athletes"]), "n": da["n_athletes"]})
        put(p, "both", domain_mean_gap(E, meta, r), n=r.size)
        al = cross_domain_alignment(E, meta, r)
        put(p, "both", al, n=al.get("n_athletes", 0))
    for d in ("combine", "nfl"):
        rd = rows[dom[rows] == d]
        es = embedding_summary(E, rd)
        put(ALL, d, es, n=es.get("n", 0))
        put(ALL, d, collapse_indicators(E, rd, seed=seed), n=rd.size)
    put(ALL, "both", collapse_indicators(E, rows, seed=seed), n=rows.size)
    df = pl.DataFrame(out, schema={"position_group": pl.String, "domain": pl.String, "metric": pl.String, "value": pl.Float64, "n": pl.Int64})
    return df.with_columns(pl.lit(representation).alias("representation"), pl.lit(fold, dtype=pl.Int64).alias("fold")).select(
        "representation", "fold", "position_group", "domain", "metric", "value", "n")


__all__ = ["collapse_indicators", "cross_domain_alignment", "domain_diagnostics", "domain_mean_gap", "embedding_summary", "grouped_domain_auc",
           "positive_pair_coverage", "rank_auc"]
