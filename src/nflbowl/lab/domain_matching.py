"""Part G — domain-shift robustness of the FROZEN inductive embeddings (sensitivity analyses; no encoder is trained or
re-embedded here).

Why is Combine vs NFL almost perfectly separable? Candidate explanations are compositional: sequence length, duration,
position mix, speed range, acceleration / turning state. Each is tested by coarsened exact matching (CEM) of the
held-out sequences of an outer fold on label-free SEQUENCE covariates (valid frames, duration, peak speed, peak
acceleration, peak |a_n|): within each covariate stratum both domains keep min(n_combine, n_nfl) sequences drawn with a
seeded generator. Bin edges are quantiles of the pooled held-out covariate (both domains) — no outcome, no target, no
fold-external data. If domain AUC stays ≈ 1 after matching, the separation is not explained by that covariate.

Metrics per (representation, fold, condition): grouped domain AUC (athlete-grouped inner CV), M11 domain-shift AUC +
RBF-MMD², mean embedding norm per domain, covariate balance (standardized mean difference after matching), and
cross-domain identity retrieval AUC among the matched sequences.

Distinguish: this module only re-reads embeddings. Models that would need re-embedding or retraining (domain-adversarial,
length-normalized encoders) are listed in ``RETRAINING_REQUIRED`` and never run here.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")
COVARIATES = ("n_valid", "duration_s", "peak_speed", "peak_accel", "max_abs_a_n")
CONDITIONS = {"original": (), "matched_valid_frames": ("n_valid",), "matched_duration": ("duration_s",), "speed_matched": ("peak_speed",),
              "state_matched": ("peak_accel", "max_abs_a_n"), "length_and_speed_matched": ("n_valid", "peak_speed")}
RETRAINING_REQUIRED = ("domain-adversarial encoder (F4_adv) evaluated inductively", "length-normalized re-embedding (crop / resample to equal duration)",
                       "position-conditioned encoder", "NFL-window-only encoder trained on matched states")


def quantile_bins(x: np.ndarray, k: int) -> np.ndarray:
    """Integer bin per value from pooled quantile edges (unique edges; NaN → −1)."""
    x = np.asarray(x, float)
    fin = np.isfinite(x)
    out = np.full(x.size, -1, np.int64)
    if fin.sum() < 2:
        return out
    edges = np.unique(np.quantile(x[fin], np.linspace(0, 1, k + 1)))
    if edges.size < 2:
        out[fin] = 0
        return out
    out[fin] = np.clip(np.searchsorted(edges, x[fin], side="right") - 1, 0, edges.size - 2)
    return out


def cem(domain: np.ndarray, strata: np.ndarray, seed: int) -> np.ndarray:
    """Positions (into the input arrays) kept by coarsened exact matching: per stratum min(n_combine, n_nfl) of each
    domain, seeded draw without replacement; strata coded −1 (missing covariate) are dropped."""
    rng = np.random.default_rng(seed)
    keep = []
    for s in np.unique(strata):
        if s < 0:
            continue
        a = np.flatnonzero((strata == s) & (domain == "combine"))
        b = np.flatnonzero((strata == s) & (domain == "nfl"))
        n = min(a.size, b.size)
        if n:
            keep += [rng.choice(a, n, replace=False), rng.choice(b, n, replace=False)]
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, np.int64)


def strata_codes(cov: dict[str, np.ndarray], names: Sequence[str], k: int) -> np.ndarray:
    if not names:
        return np.zeros(next(iter(cov.values())).size, np.int64)
    codes = [quantile_bins(cov[n], k) for n in names]
    out = np.zeros(codes[0].size, np.int64)
    bad = np.zeros(codes[0].size, bool)
    for c in codes:
        bad |= c < 0
        out = out * (k + 1) + np.maximum(c, 0)
    out[bad] = -1
    return out


def smd(x: np.ndarray, domain: np.ndarray) -> float:
    a, b = x[(domain == "combine") & np.isfinite(x)], x[(domain == "nfl") & np.isfinite(x)]
    if a.size < 2 or b.size < 2:
        return NAN
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / s) if s > 0 else NAN


def condition_rows(exp: Any, seq: Any, fold: int, k: int, seed: int, positions_min: int = 200) -> dict[str, np.ndarray]:
    """{condition: held-out sequence indices (sorted)}. Matching uses only sequence covariates of fold-``fold`` held-out rows."""
    import polars as pl

    from .geometry import held_out_rows

    rows = held_out_rows(exp, fold)
    sub = seq.filter(pl.col("idx").is_in(rows.tolist())).sort("idx")
    if sub.height != rows.size:
        raise ValueError(f"sequence-metric table lacks {rows.size - sub.height} held-out rows of fold {fold}")
    dom = sub["domain"].to_numpy()
    cov = {c: sub[c].cast(pl.Float64).to_numpy() for c in COVARIATES if c in sub.columns}
    out: dict[str, np.ndarray] = {}
    for name, cols in CONDITIONS.items():
        if any(c not in cov for c in cols):
            continue
        keep = np.arange(rows.size) if not cols else cem(dom, strata_codes(cov, cols, k), seed + 17 * int(fold))
        out[name] = rows[keep]
    pos = sub["position_group"].fill_null("?").to_numpy()
    for p in np.unique(pos):
        m = pos == p
        if (m & (dom == "combine")).sum() >= positions_min // 4 and (m & (dom == "nfl")).sum() >= positions_min // 4:
            out[f"position:{p}"] = rows[m]
    return out


def balance(seq: Any, rows: np.ndarray) -> dict[str, float]:
    import polars as pl

    sub = seq.filter(pl.col("idx").is_in(rows.tolist()))
    dom = sub["domain"].to_numpy()
    return {f"smd_{c}": smd(sub[c].cast(pl.Float64).to_numpy(), dom) for c in COVARIATES if c in sub.columns}


def evaluate(name: str, E: np.ndarray, meta: Any, rows: np.ndarray, seed: int, n_boot: int, max_per_domain: int) -> dict[str, Any]:
    from .. import _backend
    from ..evaluation.domain_diagnostics import grouped_domain_auc
    from ..evaluation.retrieval import cross_domain_retrieval

    rows = np.asarray(rows, np.int64)
    rows = rows[np.isfinite(E[rows]).all(1)]
    dom = meta["domain"].to_numpy()[rows]
    g = grouped_domain_auc(E, meta, rows, seed=seed, max_per_domain=max_per_domain)
    ds = _backend.rep().domain_shift(E[rows], meta[rows], max_n=max_per_domain, seed=seed) if rows.size else {"domain_auc": NAN, "mmd2": NAN}
    nrm = np.linalg.norm(E[rows].astype(float), axis=1) if rows.size else np.zeros(0)
    nc, nn = nrm[dom == "combine"], nrm[dom == "nfl"]
    try:
        xr = cross_domain_retrieval(E[rows], meta[rows], max(200, n_boot // 5), seed)
    except Exception:                                               # noqa: BLE001 — too few paired athletes after matching
        xr = {"auc": NAN, "n_athletes": 0}
    return {"n_rows": int(rows.size), "n_combine": int((dom == "combine").sum()), "n_nfl": int((dom == "nfl").sum()), "domain_auc_grouped": g["domain_auc"],
            "domain_auc_m11": ds["domain_auc"], "mmd2": ds["mmd2"], "norm_mean_combine": float(nc.mean()) if nc.size else NAN,
            "norm_mean_nfl": float(nn.mean()) if nn.size else NAN, "norm_ratio_nfl_over_combine": float(nn.mean() / nc.mean()) if nc.size and nn.size and nc.mean() > 0 else NAN,
            "retrieval_auc": xr.get("auc", NAN), "retrieval_n_athletes": xr.get("n_athletes", 0)}


def run(exp: Any, seq: Any, models: Sequence[str], folds: Sequence[int], k: int, seed: int, n_boot: int, max_per_domain: int = 2000) -> tuple[Any, Any]:
    """(long results: representation, fold, condition, metrics…, balance…; per-condition summary over folds)."""
    import polars as pl

    from .geometry import inductive_matrix

    meta = exp.bundle.meta
    out = []
    for f in folds:
        conds = condition_rows(exp, seq, f, k, seed)
        bal = {c: balance(seq, r) for c, r in conds.items()}
        for m in models:
            E = inductive_matrix(exp, m, f)
            for c, r in conds.items():
                out.append({"representation": m, "fold": int(f), "condition": c, **evaluate(m, E, meta, r, seed + int(f), n_boot, max_per_domain), **bal[c]})
    long = pl.DataFrame(out, infer_schema_length=None)
    if not long.height:
        return long, long
    summ = (long.group_by("representation", "condition").agg(*[pl.col(c).mean().alias(c) for c in ("domain_auc_grouped", "mmd2", "norm_ratio_nfl_over_combine",
                                                                                                   "retrieval_auc", "n_rows")],
                                                             pl.col("domain_auc_grouped").min().alias("domain_auc_min"), pl.col("domain_auc_grouped").max().alias("domain_auc_max"),
                                                             pl.len().alias("n_folds"))
            .sort("representation", "condition"))
    orig = summ.filter(pl.col("condition") == "original").select("representation", pl.col("domain_auc_grouped").alias("auc_original"),
                                                                 pl.col("mmd2").alias("mmd2_original"))
    summ = summ.join(orig, on="representation", how="left").with_columns((pl.col("domain_auc_grouped") - pl.col("auc_original")).alias("delta_auc_vs_original"),
                                                                         (pl.col("mmd2") - pl.col("mmd2_original")).alias("delta_mmd2_vs_original"))
    return long, summ


def conditional_distributions(seq: Any, covariates: Sequence[str] = COVARIATES) -> Any:
    """Long (domain, covariate, value) of the sequence covariates — the distributions matching acts on."""
    import polars as pl

    have = [c for c in covariates if c in seq.columns]
    return (seq.select("domain", *[pl.col(c).cast(pl.Float64) for c in have]).unpivot(index="domain", on=have, variable_name="covariate", value_name="value")
            .filter(pl.col("value").is_finite()))


__all__ = ["CONDITIONS", "COVARIATES", "RETRAINING_REQUIRED", "balance", "cem", "condition_rows", "conditional_distributions", "evaluate", "quantile_bins", "run", "smd",
           "strata_codes"]
