"""Phase 3 — measurement reliability, robustness, coverage and the measurement status of every M14 metric.

Trials. A Combine trial is one SEQUENCE (one attempt of one drill); its value is the sequence-level metric (event
metrics: median over the sequence's events). Agility columns pool three drills, so values are centred on each drill's
mean first (kind-centred), exactly as M13 E1 (``lab.reliability.kind_centre``).

Estimators (reused from ``lab.reliability``, nothing clamped):
* ICC(1,1) one-way random effects with variance components; athlete-cluster bootstrap interval;
* ICC(1,k̄) — reliability of the athlete median of k̄ trials (k̄ = mean trials per athlete);
* split-half: athlete means of odd vs even trials, Pearson r and Spearman–Brown 2r / (1 + r) (reported raw: for r < 0
  it is not a reliability and the status never relies on it alone);
* matched-k curves: a FIXED athlete set with ≥ k_max trials, k trials drawn without replacement per athlete, repeated →
  ICC(1,1) at equal trial counts across metrics;
* perturbation sensitivity: athlete-level Spearman between the baseline extraction and re-extractions under speed /
  a_tan noise and moving-average smoothing (same athletes, same sequences).

Measurement status (declared; a measurement verdict, NEVER a predictive or scouting claim):
    blocked     required inputs absent (registry.availability)
    invalid     coverage < min_coverage · < min athletes with ≥ 2 trials · ICC(1,1) < marginal  → rejected
    validated   ICC(1,1) ≥ threshold · ICC lower bound ≥ marginal · perturbation Spearman ≥ robust · coverage ok
    exploratory everything in between (marginal reliability, robustness not met / not evaluated)
Higher reliability does not imply predictive value; that is tested separately (Phase 5).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .. import reliability as REL
from . import registry as RG

NAN = float("nan")


# ====================================================================== trials
def feature_map(columns: Sequence[str]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Athlete column → (sequence-table metric, Combine kinds of its trials). Athlete-level metrics have no trials."""
    out = {}
    for c in columns:
        p = RG.parse_column(c)
        if p is None or p[1] not in RG.BY_NAME or RG.BY_NAME[p[1]].level not in ("event", "sequence"):
            continue
        out[c] = (p[1], RG.SCOPES[p[0]])
    return out


def trial_cache(seqtab: Any, columns: Sequence[str]) -> dict[str, Any]:
    """{column: long trials (nfl_id, kind, order, value)} in the format of ``lab.reliability.trials`` (so
    ``lab.reliability.train_icc`` works unchanged on M14 columns)."""
    return {c: REL.trials(seqtab, m, kinds, "combine") for c, (m, kinds) in feature_map(columns).items() if m in seqtab.columns}


def _values(t: Any) -> np.ndarray:
    return REL.kind_centre(t) if t["kind"].n_unique() > 1 else t["value"].to_numpy().astype(float)


def split_half(t: Any) -> dict[str, float]:
    """Odd vs even trials (athlete-level order across kinds) → athlete means → Pearson r, Spearman–Brown."""
    import polars as pl

    from ..stats import spearman

    if t is None or not t.height:
        return {"split_n": 0, "split_r": NAN, "split_rho": NAN, "split_sb": NAN}
    d = t.with_columns(pl.Series("_v", _values(t))).with_columns(pl.int_range(pl.len()).over("nfl_id").alias("_o"))
    g = d.group_by("nfl_id", (pl.col("_o") % 2).alias("_h")).agg(pl.col("_v").mean())
    w = g.pivot(on="_h", index="nfl_id", values="_v").drop_nulls()
    cols = [c for c in w.columns if c != "nfl_id"]
    if len(cols) < 2 or w.height < 10:
        return {"split_n": int(w.height), "split_r": NAN, "split_rho": NAN, "split_sb": NAN}
    a, b = w[cols[0]].to_numpy().astype(float), w[cols[1]].to_numpy().astype(float)
    ok = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 10 else NAN
    return {"split_n": int(ok.sum()), "split_r": r, "split_rho": spearman(a[ok], b[ok]) if ok.sum() >= 10 else NAN,
            "split_sb": REL.spearman_brown_raw(r)}


def reliability_table(cache: dict[str, Any], n_boot: int, seed: int, min_athletes: int) -> Any:
    """One row per column: ICC(1,1) + components + bootstrap CI, k̄, ICC(1,k̄), split-half."""
    import polars as pl

    rows = []
    for col, t in cache.items():
        p = RG.parse_column(col)
        base = {"column": col, "scope": p[0] if p else None, "metric": p[1] if p else None, "family": RG.BY_NAME[p[1]].family if p else None,
                "n_trials": int(t.height) if t is not None else 0}
        if t is None or not t.height:
            rows.append({**base, "n_athletes": 0, "icc1": NAN})
            continue
        v, ids = _values(t), t["nfl_id"].to_numpy()
        comp = REL.icc_components(v, ids)
        lo, hi = REL.icc_bootstrap(v, ids, n_boot, seed) if comp["n_athletes"] >= min_athletes else (NAN, NAN)
        cnt = t.group_by("nfl_id").len()
        kbar = float(cnt.filter(pl.col("len") >= 2)["len"].mean()) if cnt.filter(pl.col("len") >= 2).height else NAN
        rows.append({**base, **comp, "icc1_lo": lo, "icc1_hi": hi, "k_bar": kbar, "icc_k": REL.icc_average(comp["icc1"], kbar) if np.isfinite(kbar) else NAN,
                     "centred_by_kind": bool(t["kind"].n_unique() > 1), "estimable": comp["n_athletes"] >= min_athletes, **split_half(t)})
    return pl.DataFrame(rows, infer_schema_length=None)


def athlete_level_split_half(seqtab: Any, events: Any, columns: Sequence[str]) -> Any:
    """Split-half reliability of athlete-level metrics (asymmetries): recomputed on odd vs even Combine sequences of
    each athlete (store order), Spearman + Spearman–Brown."""
    import polars as pl

    from ..stats import spearman
    from .metrics import athlete_table

    c = seqtab.filter(pl.col("domain") == "combine").sort("row").with_columns(pl.int_range(pl.len()).over("nfl_id").alias("_k"))
    odd = c.filter(pl.col("_k") % 2 == 1)["row"].to_list()
    even = c.filter(pl.col("_k") % 2 == 0)["row"].to_list()
    A, B = athlete_table(seqtab, events, odd), athlete_table(seqtab, events, even)
    rows = []
    for col in columns:
        if col not in A.columns or col not in B.columns:
            continue
        j = A.select("nfl_id", pl.col(col).alias("a")).join(B.select("nfl_id", pl.col(col).alias("b")), on="nfl_id").drop_nulls()
        a, b = j["a"].to_numpy().astype(float), j["b"].to_numpy().astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        rho = spearman(a[ok], b[ok]) if ok.sum() >= 10 else NAN
        p = RG.parse_column(col)
        rows.append({"column": col, "scope": p[0] if p else None, "metric": p[1] if p else None, "family": RG.BY_NAME[p[1]].family if p else None,
                     "n_athletes": int(ok.sum()), "split_rho": rho, "split_sb": REL.spearman_brown_raw(rho), "design": "odd vs even Combine sequences"})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== equal-trial-count curves
def matched_k_curves(cache: dict[str, Any], k_grid: Sequence[int], repeats: int, seed: int, min_athletes: int) -> Any:
    """ICC(1,1) at exactly k trials per athlete on a FIXED athlete set (athletes with ≥ k_max trials, k_max = largest
    grid value reachable by ≥ min_athletes athletes), mean ± SD over ``repeats`` draws; plus the ICC(1,k) of a k-trial mean."""
    import polars as pl

    rows = []
    for col, t in cache.items():
        if t is None or not t.height:
            continue
        cnt = t.group_by("nfl_id").len()
        feasible = [k for k in sorted(k_grid) if cnt.filter(pl.col("len") >= k).height >= min_athletes]
        if not feasible:
            continue
        kmax = feasible[-1]
        S = set(int(i) for i in cnt.filter(pl.col("len") >= kmax)["nfl_id"].to_list())
        tt = t.filter(pl.col("nfl_id").is_in(list(S)))
        v = _values(tt)
        ids = tt["nfl_id"].to_numpy()
        groups = {int(i): np.flatnonzero(ids == i) for i in S}
        rng = np.random.default_rng(seed)
        for k in feasible:
            est = []
            for _ in range(repeats):
                pick = np.concatenate([rng.choice(g, size=k, replace=False) for g in groups.values()])
                est.append(REL.icc_components(v[pick], ids[pick])["icc1"])
            est = np.asarray(est, float)
            est = est[np.isfinite(est)]
            m = float(est.mean()) if est.size else NAN
            rows.append({"column": col, "k": int(k), "n_athletes": len(S), "k_max": int(kmax), "icc1_mean": m, "icc1_sd": float(est.std()) if est.size else NAN,
                         "icc_k_of_mean": REL.icc_average(m, k), "repeats": int(est.size)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def projected_curve(rel: Any, k_max: int = 10) -> Any:
    """ICC(1,k) projected from the full-sample ICC(1,1) for k = 1 … k_max (reliability versus observation count)."""
    import polars as pl

    rows = [{"column": r["column"], "k": k, "icc_k_projected": REL.icc_average(r["icc1"], k)} for r in rel.iter_rows(named=True)
            if r.get("icc1") is not None and np.isfinite(r["icc1"]) for k in range(1, k_max + 1)]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


# ====================================================================== perturbation sensitivity
def sensitivity(base: Any, perturbed: dict[str, Any], columns: Sequence[str]) -> tuple[Any, Any]:
    """(long rows per column × perturbation: n, Spearman with baseline, relative median shift; summary per column:
    min Spearman, worst perturbation). ``base`` / ``perturbed`` are athlete tables on the same athletes."""
    import polars as pl

    from ..stats import spearman

    rows = []
    for name, P in perturbed.items():
        for col in columns:
            if col not in base.columns or col not in P.columns:
                continue
            j = base.select("nfl_id", pl.col(col).alias("a")).join(P.select("nfl_id", pl.col(col).alias("b")), on="nfl_id").drop_nulls()
            a, b = j["a"].to_numpy().astype(float), j["b"].to_numpy().astype(float)
            ok = np.isfinite(a) & np.isfinite(b)
            med = float(np.median(a[ok])) if ok.any() else NAN
            rows.append({"column": col, "perturbation": name, "n_athletes": int(ok.sum()), "spearman": spearman(a[ok], b[ok]) if ok.sum() >= 10 else NAN,
                         "median_shift_rel": float((np.median(b[ok]) - med) / abs(med)) if ok.any() and med != 0 else NAN})
    long = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(schema={"column": pl.String, "perturbation": pl.String, "spearman": pl.Float64})
    if not long.height:
        return long, pl.DataFrame(schema={"column": pl.String, "min_spearman": pl.Float64})
    summ = (long.filter(pl.col("spearman").is_not_null() & pl.col("spearman").is_not_nan()).sort("spearman")
            .group_by("column", maintain_order=True).agg(pl.col("spearman").first().alias("min_spearman"), pl.col("perturbation").first().alias("worst_perturbation"),
                                                         pl.col("spearman").mean().alias("mean_spearman")))
    return long, summ


# ====================================================================== coverage
def coverage(ath: Any, columns: Sequence[str], transfer_ids: set[int] | None = None, min_group: int = 10) -> Any:
    """Per column: coverage over all athletes, over the transfer athletes, and the lowest coverage among positions with ≥ min_group athletes."""
    import polars as pl

    sub = ath.filter(pl.col("nfl_id").is_in(list(transfer_ids))) if transfer_ids else ath
    rows = []
    for c in columns:
        if c not in ath.columns:
            continue
        cov_pos = (sub.group_by("position_group").agg(pl.len().alias("n"), pl.col(c).is_not_null().mean().alias("cov")).filter(pl.col("n") >= min_group))
        lowest = cov_pos.sort("cov").head(1)
        rows.append({"column": c, "n_athletes": ath.height, "coverage_all": float(ath[c].is_not_null().mean()), "n_transfer": sub.height,
                     "coverage_transfer": float(sub[c].is_not_null().mean()) if sub.height else NAN,
                     "lowest_position": lowest["position_group"][0] if lowest.height else None,
                     "lowest_position_coverage": float(lowest["cov"][0]) if lowest.height else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== measurement status
def measurement_status(rel: Any, athlete_rel: Any, robust: Any, cov: Any, cfg: Any) -> Any:
    """Declared measurement status per predictor column (see module docstring) with the reasons."""
    import polars as pl

    R = {r["column"]: r for r in rel.iter_rows(named=True)} if rel is not None and rel.height else {}
    A = {r["column"]: r for r in athlete_rel.iter_rows(named=True)} if athlete_rel is not None and athlete_rel.height else {}
    S = {r["column"]: r for r in robust.iter_rows(named=True)} if robust is not None and robust.height else {}
    C = {r["column"]: r for r in cov.iter_rows(named=True)}
    thr, marg, rob, mincov, mina = cfg.reliability_threshold, cfg.reliability_marginal, cfg.robust_spearman, cfg.min_coverage, cfg.min_reliability_athletes
    rows = []
    for col in C:
        c = C[col]
        p = RG.parse_column(col)
        why = []
        cv = c.get("coverage_transfer")
        cv = c.get("coverage_all") if cv is None or not np.isfinite(cv) else cv
        if col in R:
            r = R[col]
            rel_v, rel_lo, n = r.get("icc1"), r.get("icc1_lo"), r.get("n_athletes") or 0
            est = "ICC(1,1)"
        elif col in A:
            r = A[col]
            rel_v, rel_lo, n = r.get("split_sb"), NAN, r.get("n_athletes") or 0
            est = "split-half SB"
        else:
            rel_v, rel_lo, n, est = NAN, NAN, 0, "none"
        rel_v = NAN if rel_v is None else float(rel_v)
        rel_lo = NAN if rel_lo is None else float(rel_lo)
        s = S.get(col, {}).get("min_spearman")
        s = NAN if s is None else float(s)
        if cv < mincov:
            why.append(f"coverage {cv:.2f} < {mincov}")
        if n < mina:
            why.append(f"{n} athletes with ≥ 2 trials < {mina}")
        if np.isfinite(rel_v) and rel_v < marg:
            why.append(f"{est} {rel_v:.2f} < marginal {marg}")
        if not np.isfinite(rel_v):
            why.append("reliability not estimable")
        if why:
            status = "invalid"
        elif rel_v >= thr and (not np.isfinite(rel_lo) and est != "ICC(1,1)" or (np.isfinite(rel_lo) and rel_lo >= marg)) and np.isfinite(s) and s >= rob:
            status = "validated"
        else:
            status = "exploratory"
            if rel_v < thr:
                why.append(f"{est} {rel_v:.2f} below threshold {thr}")
            if np.isfinite(rel_lo) and rel_lo < marg:
                why.append(f"ICC lower bound {rel_lo:.2f} < {marg}")
            if not np.isfinite(s):
                why.append("robustness not evaluated")
            elif s < rob:
                why.append(f"perturbation Spearman {s:.2f} < {rob}")
        rows.append({"column": col, "scope": p[0] if p else None, "metric": p[1] if p else None, "family": RG.BY_NAME[p[1]].family if p else None,
                     "coverage": cv, "reliability_estimator": est, "reliability": rel_v, "reliability_lo": rel_lo, "n_reliability_athletes": int(n),
                     "min_perturbation_spearman": s, "measurement_status": status, "reasons": "; ".join(why)})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["athlete_level_split_half", "coverage", "feature_map", "matched_k_curves", "measurement_status", "projected_curve", "reliability_table",
           "sensitivity", "split_half", "trial_cache"]
