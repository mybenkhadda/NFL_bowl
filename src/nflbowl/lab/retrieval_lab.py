"""Part D (cross-domain retrieval) and the retrieval parts of Part G (pool difficulty, sequence quality, heterogeneity).

The primary paired analysis is ``Experiment.compare_inductive`` (athlete-level, held-out athletes only, identical
candidate pools, fold-stratified paired bootstrap). This module derives everything else from its saved athlete
scores or from the same construction:

* rank metrics with POOL-SIZE-AWARE chance baselines (recall@K, MRR, median rank);
* paired comparisons for any pair (F4 vs F2, F4 vs random …) and any athlete-level metric (auc / top1 / top5 / rr);
* similarity margins (own vs hardest / mean negative) and the top-1 retrieved athlete, for case studies;
* alternative candidate pools defined from metadata / official measurements only, so they are identical for every
  representation: random sub-pools of size k, "stopwatch twins" (k nearest same-position athletes in official forty
  + weight) and the cross-position pool;
* quality-filtered held-out rows (pools change with the filter, but identically for every representation).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

import numpy as np

from ..evaluation import paired_inductive as pi
from ..evaluation.retrieval import CrossDomainIndex
from .geometry import held_out_rows, inductive_matrix
from .stats import chance_median_rank, chance_mrr, chance_recall_at_k

NAN = float("nan")
K_GRID = (1, 2, 3, 5, 10, 20)


# ====================================================================== rank metrics
def extend_scores(scores: Any) -> Any:
    """Adds rr = 1 / rank, normalized rank (rank − 1) / n_candidates and recall@K flags for K in ``K_GRID``."""
    import polars as pl

    s = scores.with_columns((1.0 / pl.col("rank").cast(pl.Float64)).alias("rr"),
                            ((pl.col("rank") - 1).cast(pl.Float64) / pl.col("n_candidates").cast(pl.Float64)).alias("normalized_rank"))
    return s.with_columns([(pl.col("rank") <= k).alias(f"recall_at_{k}") for k in K_GRID])


def rank_summary(scores: Any) -> Any:
    """Per representation: athlete means with the chance value of the SAME pools next to each metric."""
    import polars as pl

    s = extend_scores(scores).filter(pl.col("status") == pi.SCORED)
    rows = []
    for rep in s["representation"].unique(maintain_order=True).to_list():
        g = s.filter(pl.col("representation") == rep)
        nc = g["n_candidates"].to_numpy()
        rows.append({"representation": rep, "n_athletes": g.height, "median_pool_size": float(np.median(nc)) if nc.size else NAN,
                     "auc": float(g["auc"].mean()), "auc_chance": 0.5, "top1": float(g["top1"].cast(float).mean()), "top1_chance": chance_recall_at_k(1, nc),
                     "recall_at_5": float(g["recall_at_5"].cast(float).mean()), "recall_at_5_chance": chance_recall_at_k(5, nc),
                     "recall_at_10": float(g["recall_at_10"].cast(float).mean()), "recall_at_10_chance": chance_recall_at_k(10, nc),
                     "mrr": float(g["rr"].mean()), "mrr_chance": chance_mrr(nc), "median_rank": float(g["rank"].median()),
                     "median_rank_chance": chance_median_rank(nc), "mean_own_similarity": float(g["own_similarity"].mean())})
    return pl.DataFrame(rows, infer_schema_length=None)


def recall_curve(scores: Any) -> Any:
    import polars as pl

    s = extend_scores(scores).filter(pl.col("status") == pi.SCORED)
    rows = []
    for rep in s["representation"].unique(maintain_order=True).to_list():
        g = s.filter(pl.col("representation") == rep)
        nc = g["n_candidates"].to_numpy()
        for k in K_GRID:
            rows.append({"representation": rep, "k": k, "recall": float(g[f"recall_at_{k}"].cast(float).mean()), "chance": chance_recall_at_k(k, nc)})
    return pl.DataFrame(rows)


def by_fold(scores: Any) -> Any:
    import polars as pl

    s = extend_scores(scores).filter(pl.col("status") == pi.SCORED)
    return (s.group_by("representation", "fold").agg(pl.len().alias("n_athletes"), pl.col("auc").mean().alias("auc"), pl.col("rr").mean().alias("mrr"),
                                                     pl.col("top1").cast(pl.Float64).mean().alias("top1"), pl.col("n_candidates").median().alias("median_pool"))
            .sort("representation", "fold"))


def paired_pairs(scores: Any, pairs: Sequence[tuple[str, str]], metrics: Sequence[str] = ("auc", "rr", "top1"), n_boot: int = 1000, seed: int = 0,
                 min_n: int = 10, adequate_n: int = 30) -> Any:
    """Paired Δ (model − reference) for each (model, reference) pair and metric, on the athletes scored by BOTH."""
    import polars as pl

    s = extend_scores(scores)
    present = set(s.filter(pl.col("status") == pi.SCORED)["representation"].unique().to_list())
    out = []
    for m, r in pairs:
        if m not in present or r not in present:
            out.append(pl.DataFrame({"representation": [m], "reference": [r], "status": ["missing"]}))
            continue
        sub = s.filter(pl.col("representation").is_in([m, r]))
        for met in metrics:
            res = pi.paired_bootstrap(sub, [r, m], r, metric=met, n_boot=n_boot, seed=seed, min_n=min_n, adequate_n=adequate_n)
            out.append(res.filter(pl.col("representation") == m))
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


# ====================================================================== similarity margins / case studies
def similarity_margins(name: str, E: np.ndarray, meta: Any, rows: np.ndarray, fold: int) -> Any:
    """Per held-out athlete (both domains, same-position pool ≥ 2): own similarity, mean / max negative similarity,
    margin = own − max negative, z = (own − mean neg) / sd neg, and the top-1 retrieved candidate."""
    import polars as pl

    from ..evaluation.retrieval import cos

    ho = np.sort(np.asarray(rows, np.int64))
    ix = CrossDomainIndex(E[ho], meta[ho])
    both = ix.both
    if not both:
        return pl.DataFrame()
    S = cos(ix.combine_means(), ix.nfl_means(both))
    P = ix.positions()
    out = []
    for i, a in enumerate(both):
        cand = np.flatnonzero((P == P[i]) & (np.arange(P.size) != i))
        if cand.size < 2 or not np.isfinite(S[i, i]):
            continue
        neg = S[i, cand]
        top = cand[int(np.argmax(neg))]
        out.append({"representation": name, "fold": int(fold), "nfl_id": int(a), "position_group": P[i], "own_similarity": float(S[i, i]),
                    "mean_negative": float(neg.mean()), "max_negative": float(neg.max()), "margin": float(S[i, i] - neg.max()),
                    "z_own": float((S[i, i] - neg.mean()) / neg.std()) if neg.std() > 0 else NAN, "top_candidate": int(both[top]),
                    "own_is_top1": bool(S[i, i] > neg.max()), "n_candidates": int(cand.size)})
    return pl.DataFrame(out, infer_schema_length=None)


def margins_all(exp: Any, models_folds: dict[str, list[int]]) -> Any:
    import polars as pl

    meta = exp.bundle.meta
    parts = [similarity_margins(n, inductive_matrix(exp, n, f), meta, held_out_rows(exp, f), f) for n, fl in models_folds.items() for f in fl]
    parts = [p for p in parts if p.height]
    return pl.concat(parts, how="vertical") if parts else pl.DataFrame()


def select_case_studies(scores: Any, margins: Any, model: str, reference: str, n: int = 4) -> Any:
    """Explicit, reproducible selection rules (ties by nfl_id): strong successes (model rank 1, largest margin), strong
    failures (largest normalized rank), high uncertainty (|margin| smallest), model–reference disagreement (largest
    |Δ normalized rank|), unexpected neighbour (model top-1 is not the athlete, highest z of the wrong match)."""
    import polars as pl

    s = extend_scores(scores).filter(pl.col("status") == pi.SCORED)
    a = s.filter(pl.col("representation") == model).select("fold", "nfl_id", "position_group", "rank", "normalized_rank", "n_candidates")
    b = s.filter(pl.col("representation") == reference).select("fold", "nfl_id", pl.col("normalized_rank").alias("ref_normalized_rank"), pl.col("rank").alias("ref_rank"))
    m = margins.filter(pl.col("representation") == model).select("fold", "nfl_id", "margin", "z_own", "top_candidate", "own_is_top1")
    d = a.join(b, on=["fold", "nfl_id"], how="inner").join(m, on=["fold", "nfl_id"], how="inner").with_columns(
        (pl.col("normalized_rank") - pl.col("ref_normalized_rank")).alias("delta_normalized_rank"))
    if not d.height:
        return d
    picks = [("strong_success", d.filter(pl.col("rank") == 1).sort(["margin", "nfl_id"], descending=[True, False]).head(n)),
             ("strong_failure", d.sort(["normalized_rank", "nfl_id"], descending=[True, False]).head(n)),
             ("high_uncertainty", d.with_columns(pl.col("margin").abs().alias("_a")).sort(["_a", "nfl_id"]).head(n).drop("_a")),
             ("model_reference_disagreement", d.with_columns(pl.col("delta_normalized_rank").abs().alias("_a")).sort(["_a", "nfl_id"], descending=[True, False]).head(n).drop("_a")),
             ("unexpected_neighbour", d.filter(~pl.col("own_is_top1")).sort(["margin", "nfl_id"], descending=[False, False]).head(n))]
    return pl.concat([p.with_columns(pl.lit(k).alias("case_type")) for k, p in picks if p.height], how="vertical")


def speed_profiles(exp: Any, athletes: Sequence[int], labels: dict[int, str] | None = None) -> Any:
    """Mean ± SD speed (yd/s) per frame (0.1 s) of each athlete's Combine forty attempts and NFL windows."""
    import polars as pl

    b = exp.bundle
    meta = b.meta
    ids = meta["nfl_id"].to_numpy()
    dom = meta["domain"].to_numpy()
    kind = meta["kind"].to_numpy()
    out = []
    for a in athletes:
        for d in ("combine", "nfl"):
            sel = np.flatnonzero((ids == a) & (dom == d) & ((kind == "forty") if d == "combine" else True))
            if not sel.size:
                continue
            x = np.asarray(b.X[sel][:, :, 0], np.float32) * 10.0
            m = np.asarray(b.M[sel])
            v = np.where(m, x, np.nan)
            with np.errstate(invalid="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mu, sd = np.nanmean(v, 0), np.nanstd(v, 0)
            k = np.flatnonzero(np.isfinite(mu))
            out.append(pl.DataFrame({"athlete": int(a), "label": (labels or {}).get(int(a), str(a)), "domain": d, "t": k * 0.1, "mean": mu[k], "sd": sd[k],
                                     "n_sequences": int(sel.size)}))
    return pl.concat(out, how="vertical") if out else pl.DataFrame()


# ====================================================================== alternative candidate pools (G3)
def _seeded(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng([seed, *[int(k) & 0x7FFFFFFF for k in keys]])


def pool_definitions(athletes: Any, seed: int, ks: Sequence[int] = (3, 5, 10), twin_k: Sequence[int] = (3, 5)) -> dict[str, Callable[[int, int, list[int], list[int]], list[int]]]:
    """{pool name: fn(fold, athlete, same_position_candidates, all_candidates) → candidate list}. Every function
    depends only on metadata / official measurements and the seed — never on a representation."""
    out: dict[str, Callable[..., list[int]]] = {"full_same_position": lambda f, a, same, allc: list(same),
                                                  "cross_position": lambda f, a, same, allc: list(allc)}
    for k in ks:
        def rnd(f: int, a: int, same: list[int], allc: list[int], _k: int = k) -> list[int]:
            if len(same) < _k:
                return []
            return sorted(_seeded(seed, f, a, _k).choice(sorted(same), _k, replace=False).tolist())
        out[f"random_{k}_same_position"] = rnd
    if athletes is not None and {"forty", "weight"} <= set(athletes.columns):
        z = athletes.select("nfl_id", "forty", "weight").drop_nulls()
        mu_f, sd_f = float(z["forty"].mean()), float(z["forty"].std() or 1.0)
        mu_w, sd_w = float(z["weight"].mean()), float(z["weight"].std() or 1.0)
        coord = {int(i): ((f - mu_f) / sd_f, (w - mu_w) / sd_w) for i, f, w in zip(z["nfl_id"].to_list(), z["forty"].to_list(), z["weight"].to_list())}
        for k in twin_k:
            def twins(f: int, a: int, same: list[int], allc: list[int], _k: int = k) -> list[int]:
                if a not in coord:
                    return []
                c = [x for x in same if x in coord]
                if len(c) < _k:
                    return []
                d = sorted(c, key=lambda x: ((coord[x][0] - coord[a][0]) ** 2 + (coord[x][1] - coord[a][1]) ** 2, x))
                return sorted(d[:_k])
            out[f"stopwatch_twins_{k}"] = twins
    return out


def pool_scores(name: str, E: np.ndarray, meta: Any, rows: np.ndarray, fold: int, pools: dict[str, Callable[..., list[int]]]) -> Any:
    """Athlete-level AUC / top-1 / rank under each pool definition (schema compatible with ``paired_bootstrap``)."""
    import polars as pl

    from ..evaluation.retrieval import cos

    ho = np.sort(np.asarray(rows, np.int64))
    ix = CrossDomainIndex(E[ho], meta[ho])
    both = ix.both
    if not both:
        return pl.DataFrame()
    S = cos(ix.combine_means(), ix.nfl_means(both))
    P = ix.positions()
    pos_of = {a: P[i] for i, a in enumerate(both)}
    col = {a: i for i, a in enumerate(both)}
    out = []
    for pname, fn in pools.items():
        for i, a in enumerate(both):
            same = [x for x in both if x != a and pos_of[x] == P[i]]
            allc = [x for x in both if x != a]
            cand = fn(int(fold), int(a), same, allc)
            base = {"pool": pname, "representation": name, "fold": int(fold), "nfl_id": int(a), "position_group": P[i]}
            if len(cand) < 2 or not np.isfinite(S[i, i]):
                out.append({**base, "status": "pool_too_small", "auc": NAN, "top1": None, "top5": None, "rank": None, "n_candidates": len(cand), "pool_hash": None})
                continue
            sims = S[i, [col[c] for c in cand]]
            below, above = int((sims < S[i, i]).sum()), int((sims > S[i, i]).sum())
            out.append({**base, "status": pi.SCORED, "auc": below / len(cand), "top1": above < 1, "top5": above < 5, "rank": above + 1, "n_candidates": len(cand),
                        "pool_hash": hashlib.sha1(",".join(map(str, sorted(cand))).encode()).hexdigest()[:16]})
    return pl.DataFrame(out, infer_schema_length=None)


def pool_difficulty(exp: Any, models_folds: dict[str, list[int]], reference: str, seed: int, n_boot: int) -> tuple[Any, Any]:
    """(athlete scores under every pool, paired comparison per pool vs ``reference``) — chance AUC is 0.5 under every
    pool; chance top-1 = mean 1 / (k + 1) for the realised pool sizes (reported)."""
    import polars as pl

    b = exp.bundle
    pools = pool_definitions(b.athletes, seed)
    meta = b.meta
    parts = [pool_scores(n, inductive_matrix(exp, n, f), meta, held_out_rows(exp, f), f, pools) for n, fl in models_folds.items() for f in fl]
    parts = [p for p in parts if p.height]
    if not parts:
        return pl.DataFrame(), pl.DataFrame()
    sc = pl.concat(parts, how="diagonal_relaxed")
    names = list(models_folds)
    comp = []
    for pname in sc["pool"].unique(maintain_order=True).to_list():
        sub = sc.filter(pl.col("pool") == pname).drop("pool")
        if reference not in sub.filter(pl.col("status") == pi.SCORED)["representation"].unique().to_list():
            continue
        for met in ("auc", "top1"):
            r = pi.paired_bootstrap(sub, names, reference, metric=met, n_boot=n_boot, seed=seed)
            nc = sub.filter((pl.col("status") == pi.SCORED) & (pl.col("representation") == reference))["n_candidates"].to_numpy()
            comp.append(r.with_columns(pl.lit(pname).alias("pool"), pl.lit(0.5 if met == "auc" else chance_recall_at_k(1, nc)).alias("chance")))
    return sc, (pl.concat(comp, how="diagonal_relaxed") if comp else pl.DataFrame())


# ====================================================================== sequence quality (G4)
QUALITY_FILTERS = {
    "all_rows": "no filter",
    "valid_frames_ge_30": "sequences with ≥ 30 valid frames (≥ 3 s)",
    "heading_valid_ge_0_9": "sequences with heading valid on ≥ 90 % of frames",
    "no_extreme_speed": "drop sequences with peak speed > 12.5 yd/s (≈ 25.6 mph; tracking artefacts)",
    "no_extreme_accel": "drop sequences with |a_tan| peak > 12 yd/s²",
    "nfl_duration_ge_median": "NFL windows at least as long as the median NFL window; Combine unchanged",
}


def quality_masks(seq: Any) -> dict[str, np.ndarray]:
    import polars as pl

    n = seq.height
    med = float(seq.filter(pl.col("domain") == "nfl")["duration_s"].median() or 0.0)
    f = lambda e: seq.select(e.fill_null(False).alias("_m"))["_m"].to_numpy().astype(bool)   # noqa: E731
    return {"all_rows": np.ones(n, bool), "valid_frames_ge_30": f(pl.col("n_valid") >= 30),
            "heading_valid_ge_0_9": f(pl.col("heading_valid_share") >= 0.9), "no_extreme_speed": f(~(pl.col("peak_speed") > 12.5)),
            "no_extreme_accel": f(~((pl.col("peak_accel") > 12) | (pl.col("peak_decel") > 12))),
            "nfl_duration_ge_median": f((pl.col("domain") == "combine") | (pl.col("duration_s") >= med))}


def quality_sensitivity(exp: Any, seq: Any, models_folds: dict[str, list[int]], reference: str, seed: int, n_boot: int) -> tuple[Any, Any]:
    """(athlete scores per filter, paired Δ AUC vs reference per filter). Rows of ``seq`` align with the bundle meta."""
    import polars as pl

    meta = exp.bundle.meta
    masks = quality_masks(seq)
    scs, comp = [], []
    names = list(models_folds)
    for fname, mk in masks.items():
        parts = []
        for n, fl in models_folds.items():
            for f in fl:
                rows = held_out_rows(exp, f)
                parts.append(pi.athlete_retrieval_scores(n, inductive_matrix(exp, n, f), meta, rows[mk[rows]], f))
        sc = pl.concat(parts, how="vertical").with_columns(pl.lit(fname).alias("filter"))
        scs.append(sc)
        try:
            r = pi.paired_bootstrap(sc.drop("filter"), names, reference, metric="auc", n_boot=n_boot, seed=seed)
            comp.append(r.with_columns(pl.lit(fname).alias("filter"), pl.lit(QUALITY_FILTERS[fname]).alias("filter_description"),
                                       pl.lit(float(mk.mean())).alias("share_rows_kept")))
        except ValueError:
            continue
    return pl.concat(scs, how="diagonal_relaxed"), (pl.concat(comp, how="diagonal_relaxed") if comp else pl.DataFrame())


# ====================================================================== heterogeneity (G5)
def heterogeneity_groups(ath_table: Any, ath_features: Any | None = None) -> Any:
    """nfl_id, fold → subgroup columns: draft_year, m1_complete, nfl_coverage (tercile of NFL windows), combine_speed
    (within-position tercile of median forty peak speed)."""
    import polars as pl

    g = ath_table.select("nfl_id", "fold", "position_group", *[c for c in ("draft_year", "m1_complete") if c in ath_table.columns], "n_nfl_seq")
    g = g.with_columns(((pl.col("n_nfl_seq").rank("ordinal") - 1) * 3 // pl.len()).cast(pl.Int64)
                       .replace_strict({0: "low", 1: "mid", 2: "high"}, default=None, return_dtype=pl.String).alias("nfl_coverage"))
    if "draft_year" in g.columns:
        g = g.with_columns(pl.col("draft_year").cast(pl.String))
    if "m1_complete" in g.columns:
        g = g.with_columns(pl.col("m1_complete").cast(pl.String))
    if ath_features is not None and "c_forty__peak_speed" in ath_features.columns:
        f = ath_features.select("nfl_id", "c_forty__peak_speed").drop_nulls()
        g = g.join(f, on="nfl_id", how="left").with_columns(
            pl.when(pl.col("c_forty__peak_speed").is_null()).then(None).otherwise(
                ((pl.col("c_forty__peak_speed").rank("ordinal").over("position_group") - 1) * 3 // pl.col("c_forty__peak_speed").count().over("position_group"))
                .cast(pl.Int64)).alias("_t")).with_columns(
            pl.col("_t").replace_strict({0: "slow", 1: "mid", 2: "fast"}, default=None, return_dtype=pl.String).alias("combine_speed")).drop("_t", "c_forty__peak_speed")
    return g


__all__ = ["K_GRID", "QUALITY_FILTERS", "by_fold", "extend_scores", "heterogeneity_groups", "margins_all", "paired_pairs", "pool_definitions", "pool_difficulty",
           "pool_scores", "quality_masks", "quality_sensitivity", "rank_summary", "recall_curve", "select_case_studies", "similarity_margins", "speed_profiles"]
