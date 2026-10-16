"""Paired, athlete-level comparison of inductive representations (F0 · F2 · F4 …) on athletes excluded from encoder training.

Estimand
--------
For outer fold ``f`` the inductive encoder of fold ``f`` was trained without any sequence (Combine or NFL) of the
athletes in fold ``f``. Among those held-out athletes, every athlete with sequences in BOTH domains is a query:

* query vector      mean embedding of the athlete's Combine sequences (all drill kinds, as ``inductive_fold_retrieval``)
* candidate pool    NFL mean embeddings of the OTHER held-out athletes of fold ``f`` with the same ``position_group``
                    and both domains (the original position constraint of ``rank_rows``)
* athlete AUC       share of candidates whose cosine similarity is strictly below the athlete's own NFL mean
                    (0.5 = chance, 1 = own NFL movement always closest; HIGHER IS BETTER)
* top-1 / top-5     fewer than 1 / 5 candidates strictly above the own match

The per-athlete AUCs averaged over queries reproduce ``cross_domain_retrieval`` on the same rows exactly (tested).

Candidate pools depend only on metadata and the held-out rows, never on the representation, so every representation
is scored on the SAME queries against the SAME pools. Each athlete row carries ``pool_hash`` (hash of the sorted
candidate ids) and the pooled comparison refuses to run if any athlete's pool differs between representations.

Uncertainty
-----------
``paired_bootstrap`` resamples ATHLETES (query units) with replacement, stratified by outer fold (each fold keeps its
number of athletes), and applies the SAME resampled index vector to every representation, so each replicate yields a
paired difference ``mean AUC(model) - mean AUC(reference)``. Candidate pools and per-athlete AUCs are held fixed: the
interval is conditional on the realised pools (it does not re-rank against resampled candidates). The interval is the
percentile interval of the replicate differences (default 95 %, ``n_boot`` replicates). Only athletes scored by every
representation enter (complete cases); the others are counted as exclusions, never imputed.

No fitting happens here: retrieval is cosine similarity with no tuned parameter, and no NFL outcome (athletes.parquet)
is read, so held-out outcomes cannot enter encoder fitting or retrieval-model selection.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..data.splits import HELD_OUT, SplitLeakError
from .retrieval import CrossDomainIndex, cos

NAN = float("nan")
SCORED = "scored"
HIGHER_IS_BETTER = "higher_is_better"
RESAMPLING = "athletes with replacement, stratified by outer fold; same draws for every representation; candidate pools fixed"
ALL = "ALL"


class CandidatePoolMismatch(ValueError):
    """The same held-out athlete was ranked against different candidate pools by two representations."""


# ====================================================================== isolation checks
def assert_fold_isolation(meta: Any, folds_by_athlete: dict[int, int], held_out: np.ndarray, fold: int,
                          train: np.ndarray | None = None, validation: np.ndarray | None = None) -> dict[str, int]:
    """Every held-out row belongs to an athlete of ``fold``; no held-out athlete appears among training / validation rows."""
    ids = meta["nfl_id"].to_numpy()
    ho_ath = set(ids[np.asarray(held_out, np.int64)].tolist())
    wrong = sorted(i for i in ho_ath if folds_by_athlete.get(int(i), -1) != int(fold))
    if wrong:
        raise SplitLeakError(f"held-out rows of fold {fold} include athletes of other folds: {wrong[:5]}")
    for name, rows in (("training", train), ("validation", validation)):
        if rows is not None:
            shared = ho_ath & set(ids[np.asarray(rows, np.int64)].tolist())
            if shared:
                raise SplitLeakError(f"held-out athlete(s) of fold {fold} also in the {name} rows: {sorted(shared)[:5]}")
    return {"fold": int(fold), "held_out_athletes": len(ho_ath), "held_out_rows": int(np.asarray(held_out).size)}


def verify_held_out_roles(split_role: Sequence[str], held_out: np.ndarray, fold: int | None = None) -> None:
    """The saved embedding table marks exactly the held-out rows as ``held_out`` (i.e. it comes from the inductive
    encoder of this fold, not from a transductive or other-fold encoder)."""
    roles = np.asarray(split_role, dtype=object)
    ho = np.asarray(held_out, np.int64)
    if ho.size and not np.all(roles[ho] == HELD_OUT):
        raise SplitLeakError(f"embedding table of fold {fold}: some held-out rows are not marked '{HELD_OUT}' (transductive or other-fold embeddings?)")
    if int((roles == HELD_OUT).sum()) != ho.size:
        raise SplitLeakError(f"embedding table of fold {fold}: held-out role count differs from the held-out split")


# ====================================================================== athlete-level scores
def _pool_hash(ids: Sequence[int]) -> str:
    return hashlib.sha1(",".join(str(int(i)) for i in sorted(ids)).encode()).hexdigest()[:16]


def athlete_retrieval_scores(representation: str, E: np.ndarray, meta: Any, held_out: np.ndarray, fold: int) -> Any:
    """One row per held-out athlete of ``fold`` (scored or excluded with a reason). Rows of ``E`` align with ``meta``."""
    import polars as pl

    ho = np.sort(np.asarray(held_out, np.int64))
    mh = meta[ho]
    ix = CrossDomainIndex(E[ho], mh)
    both = ix.both
    out: list[dict[str, Any]] = []

    def base(i: int, status: str) -> dict[str, Any]:
        return {"representation": representation, "fold": int(fold), "nfl_id": int(i), "position_group": ix.pos.get(i), "status": status,
                "n_combine_seq": int(len(ix.cg.get(i, ()))), "n_nfl_seq": int(len(ix.ng.get(i, ()))), "n_candidates": 0, "n_nonfinite_candidates": 0,
                "own_similarity": NAN, "auc": NAN, "rank": None, "top1": None, "top5": None, "pool_hash": None, "candidate_ids": []}

    for i in sorted(set(mh["nfl_id"].to_list())):
        if i not in ix.cg or i not in ix.ng:
            out.append(base(i, "missing_combine" if i not in ix.cg else "missing_nfl"))
    if both:
        S = cos(ix.combine_means(), ix.nfl_means(both))
        groups: dict[Any, list[int]] = {}
        for k, p in enumerate(ix.positions().tolist()):
            groups.setdefault(p, []).append(k)
        for g in groups.values():
            ids_g = [both[k] for k in g]
            if len(g) - 1 < 2:                                          # rank_rows skip rule: < 2 same-position candidates
                for i in ids_g:
                    r = base(i, "position_pool_too_small")
                    r["n_candidates"] = len(g) - 1
                    out.append(r)
                continue
            g_ = np.asarray(g)
            sub = S[np.ix_(g_, g_)]
            d = np.diagonal(sub).copy()
            below = (sub < d[:, None]).sum(1)                           # self-comparison d < d is False
            above = (sub > d[:, None]).sum(1)
            nonfin = (~np.isfinite(sub)).sum(1) - (~np.isfinite(d)).astype(int)
            n = g_.size - 1
            for r_, i in enumerate(ids_g):
                cand = ids_g[:r_] + ids_g[r_ + 1:]
                r = base(i, SCORED if np.isfinite(d[r_]) else "nonfinite_embedding")
                r.update({"n_candidates": n, "n_nonfinite_candidates": int(nonfin[r_]), "pool_hash": _pool_hash(cand), "candidate_ids": [int(c) for c in cand]})
                if r["status"] == SCORED:
                    r.update({"own_similarity": float(d[r_]), "auc": float(below[r_] / n), "rank": int(above[r_]) + 1,
                              "top1": bool(above[r_] < 1), "top5": bool(above[r_] < 5)})
                out.append(r)
    schema = {"representation": pl.String, "fold": pl.Int64, "nfl_id": pl.Int64, "position_group": pl.String, "status": pl.String,
              "n_combine_seq": pl.Int64, "n_nfl_seq": pl.Int64, "n_candidates": pl.Int64, "n_nonfinite_candidates": pl.Int64,
              "own_similarity": pl.Float64, "auc": pl.Float64, "rank": pl.Int64, "top1": pl.Boolean, "top5": pl.Boolean, "pool_hash": pl.String,
              "candidate_ids": pl.List(pl.Int64)}
    return pl.DataFrame(out, schema=schema).sort("fold", "nfl_id")


def check_identical_pools(scores: Any) -> None:
    """Raise ``CandidatePoolMismatch`` when an athlete (fold, nfl_id) has different candidate pools across representations."""
    import polars as pl

    s = scores.filter(pl.col("pool_hash").is_not_null())
    if not s.height:
        return
    bad = s.group_by("fold", "nfl_id").agg(pl.col("pool_hash").n_unique().alias("k")).filter(pl.col("k") > 1)
    if bad.height:
        raise CandidatePoolMismatch(f"{bad.height} athlete(s) were ranked against different candidate pools across representations "
                                    f"(e.g. {bad.head(3).to_dicts()}); representations must share held-out splits")


# ====================================================================== paired bootstrap
def reliability(n: int, min_n: int = 10, adequate_n: int = 30) -> str:
    """``insufficient`` (no interval reported) · ``exploratory`` (interval shown, not a finding) · ``adequate``."""
    return "insufficient" if n < min_n else ("exploratory" if n < adequate_n else "adequate")


def bootstrap_indices(strata: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot × n) row indices; each stratum is resampled within itself with replacement (sizes preserved)."""
    strata = np.asarray(strata)
    parts = []
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        parts.append(members[rng.integers(0, members.size, (n_boot, members.size))])
    return np.concatenate(parts, axis=1) if parts else np.empty((n_boot, 0), np.int64)


def _complete_cases(scores: Any, models: Sequence[str], metric: str) -> Any:
    import polars as pl

    s = scores.filter(pl.col("status") == SCORED)
    keys = ["fold", "nfl_id", "position_group"]
    wide = None
    for m in models:
        part = s.filter(pl.col("representation") == m).select(*keys, pl.col(metric).cast(pl.Float64).alias(m))
        wide = part if wide is None else wide.join(part, on=keys, how="inner")
    return wide.sort("fold", "nfl_id")


def paired_bootstrap(scores: Any, models: Sequence[str], reference: str, metric: str = "auc", n_boot: int = 2000, seed: int = 0,
                     by: str | None = None, stratify: str | None = "fold", min_n: int = 10, adequate_n: int = 30, ci: float = 0.95) -> Any:
    """Paired comparison of ``models`` against ``reference`` on athlete-level ``metric`` (``auc`` / ``top1`` / ``top5``).

    One row per (group, representation): mean with its bootstrap interval, and for non-reference rows the paired
    difference ``mean(model) - mean(reference)`` with its percentile interval, bootstrap SE and the share of replicates
    with difference ≤ 0. ``by='position_group'`` repeats the analysis inside each position (subgroups below ``min_n``
    athletes get no interval and are labelled ``insufficient``)."""
    import polars as pl

    check_identical_pools(scores)
    s = scores.filter(pl.col("status") == SCORED)
    present = set(s["representation"].unique().to_list())
    if reference not in present:
        raise ValueError(f"reference representation {reference!r} has no scored athletes; cannot form paired differences")
    avail = [m for m in models if m in present]
    if reference not in avail:
        avail = [reference, *avail]
    missing = [m for m in models if m not in present]
    wide = _complete_cases(scores, avail, metric)
    any_scored = s.select("fold", "nfl_id", "position_group").unique()
    groups: list[tuple[str, Any]] = [(ALL, None)]
    if by:
        groups = [(str(g), g) for g in sorted(any_scored[by].unique().to_list(), key=lambda v: (v is None, str(v)))]
    alpha = (1 - ci) / 2
    rows: list[dict[str, Any]] = []
    def in_group(df: Any, gval: Any) -> Any:
        if by is None:
            return df
        return df.filter(pl.col(by).is_null()) if gval is None else df.filter(pl.col(by) == gval)

    for gname, gval in groups:
        w, a = in_group(wide, gval), in_group(any_scored, gval)
        n = w.height
        rel = reliability(n, min_n, adequate_n)
        common = {"scope": "overall" if by is None else by, "position_group": gname if by else ALL, "reference": reference, "metric": metric,
                  "direction": HIGHER_IS_BETTER, "n_athletes": n, "n_folds": int(w["fold"].n_unique()) if n else 0,
                  "n_excluded_incomplete": int(a.height - n), "n_boot": int(n_boot), "ci_level": ci, "resampling": RESAMPLING, "reliability": rel}
        X = w.select(avail).to_numpy().astype(float) if n else np.empty((0, len(avail)))
        Mb = None
        if rel != "insufficient" and n_boot > 0:
            rng = np.random.default_rng(seed if by is None else [seed, zlib.crc32(gname.encode())])
            strata = w[stratify].to_numpy() if stratify else np.zeros(n, int)
            draws = bootstrap_indices(strata, n_boot, rng)
            Mb = np.stack([X[:, j][draws].mean(1) for j in range(len(avail))], axis=1)      # (n_boot × models), paired draws
        jr = avail.index(reference)
        for j, m in enumerate(avail):
            mean = float(X[:, j].mean()) if n else NAN
            r = {**common, "representation": m, "status": "ok", "mean": mean, "mean_lo": NAN, "mean_hi": NAN, "delta": NAN, "delta_lo": NAN,
                 "delta_hi": NAN, "delta_se": NAN, "p_boot_delta_le_0": NAN, "ci_excludes_zero": None}
            if Mb is not None:
                r["mean_lo"], r["mean_hi"] = (float(v) for v in np.quantile(Mb[:, j], [alpha, 1 - alpha]))
            if m != reference and n:
                r["delta"] = float((X[:, j] - X[:, jr]).mean())
                if Mb is not None:
                    D = Mb[:, j] - Mb[:, jr]
                    lo, hi = (float(v) for v in np.quantile(D, [alpha, 1 - alpha]))
                    r.update({"delta_lo": lo, "delta_hi": hi, "delta_se": float(D.std(ddof=1)) if D.size > 1 else NAN,
                              "p_boot_delta_le_0": float((D <= 0).mean()), "ci_excludes_zero": bool(lo > 0 or hi < 0)})
            rows.append(r)
        for m in missing:
            rows.append({**common, "representation": m, "status": "missing", "n_athletes": 0, "mean": NAN, "mean_lo": NAN, "mean_hi": NAN, "delta": NAN,
                         "delta_lo": NAN, "delta_hi": NAN, "delta_se": NAN, "p_boot_delta_le_0": NAN, "ci_excludes_zero": None})
    return pl.DataFrame(rows, infer_schema_length=None)


def coverage_table(scores: Any, models: Sequence[str], folds: Sequence[int], unavailable: dict[str, list[int]] | None = None) -> Any:
    """Per representation × fold: held-out athletes, scored athletes and exclusions by reason; ``available`` is False
    when the inductive encoder of that fold was missing."""
    import polars as pl

    unavailable = unavailable or {}
    rows = []
    for m in models:
        for f in folds:
            s = scores.filter((pl.col("representation") == m) & (pl.col("fold") == f))
            counts = dict(zip(*s.group_by("status").len().to_dict(as_series=False).values())) if s.height else {}
            rows.append({"representation": m, "fold": int(f), "available": int(f) not in unavailable.get(m, []), "n_held_out_athletes": int(s.height),
                         "n_scored": int(counts.get(SCORED, 0)), "n_missing_combine": int(counts.get("missing_combine", 0)),
                         "n_missing_nfl": int(counts.get("missing_nfl", 0)), "n_position_pool_too_small": int(counts.get("position_pool_too_small", 0)),
                         "n_nonfinite_embedding": int(counts.get("nonfinite_embedding", 0)),
                         "median_candidates": float(s.filter(pl.col("status") == SCORED)["n_candidates"].median() or NAN) if s.height else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


@dataclass
class InductiveComparison:
    """Result of ``Experiment.compare_inductive`` (also rebuilt by ``Experiment.load_inductive_comparison``)."""
    comparison: Any                 # paired_inductive_comparison.csv  (overall, AUC + top-5)
    subgroups: Any                  # position_subgroup_results.csv
    scores: Any                     # athlete_retrieval_scores.parquet
    domain: Any                     # domain_diagnostics.csv (long)
    coverage: Any                   # inductive_coverage.csv
    meta: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    def summary(self) -> Any:
        import polars as pl

        c = self.comparison
        if not c.height:
            return c
        cols = ["representation", "reference", "metric", "n_athletes", "mean", "mean_lo", "mean_hi", "delta", "delta_lo", "delta_hi", "reliability"]
        return c.filter(pl.col("metric") == "auc").select([k for k in cols if k in c.columns])


__all__ = ["ALL", "CandidatePoolMismatch", "HIGHER_IS_BETTER", "InductiveComparison", "RESAMPLING", "SCORED", "assert_fold_isolation",
           "athlete_retrieval_scores", "bootstrap_indices", "check_identical_pools", "coverage_table", "paired_bootstrap", "reliability",
           "verify_held_out_roles"]
