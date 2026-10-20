"""E17.B1 / E17.B2 — context-adjusted athlete effects (simple statistics before any neural model).

Model per (event metric, domain):  y_ij = μ + context_ij + u_i + e_ij
* context = stratum (entry-speed × turn class, ``contexts.assign_strata``) + drill / demand kind, as fixed effects;
* u_i ~ N(0, σ²_u) athlete effect, e_ij ~ N(0, σ²_e) event noise.

Estimation (fold-safe): for every outer fold f, context means and variance components are estimated from athletes of the
OTHER folds only; athletes of fold f receive BLUP-shrunk effects û_i = σ²_u / (σ²_u + σ²_e / n_i) · mean_j(y_ij − context_ij).
Variance components: ``statsmodels`` MixedLM (REML, random intercept) when installed and enabled; otherwise the one-way
method of moments on the context residuals (``reliability.icc_components``) — the method used is recorded per row.

Read-outs: σ²_u, σ²_e, the context variance, athlete share of residual variance (= ICC of one event after adjustment),
athlete share of TOTAL variance, reliability of a k-event athlete mean (σ²_u / (σ²_u + σ²_e / k)), and the disattenuated
correlation of Combine and NFL athlete effects of the same metric within position (r / √(rel_C · rel_N); values above 1
are reported, never clipped).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")


def _long(events: Any, metric: str, domain: str) -> Any:
    import polars as pl

    return events.filter((pl.col("domain") == domain) & pl.col(metric).cast(pl.Float64).fill_nan(None).is_not_null()).select(
        "nfl_id", "position_group", "kind", "stratum", "row", pl.col(metric).cast(pl.Float64).alias("y"))


def context_means(train: Any) -> tuple[dict[tuple[str, str], float], float]:
    """Additive-free cell means of (kind, stratum) on training rows, with the training grand mean as fallback."""
    import polars as pl

    g = train.group_by("kind", "stratum").agg(pl.col("y").mean().alias("m"), pl.len().alias("n")).filter(pl.col("n") >= 5)
    return {(k, s): m for k, s, m in zip(g["kind"].to_list(), g["stratum"].to_list(), g["m"].to_list(), strict=True)}, float(train["y"].mean())


def residualise(d: Any, means: dict[tuple[str, str], float], grand: float) -> np.ndarray:
    return d["y"].to_numpy() - np.array([means.get((k, s), grand) for k, s in zip(d["kind"].to_list(), d["stratum"].to_list(), strict=True)])


def components(resid: np.ndarray, ids: np.ndarray, frame: Any | None = None, use_statsmodels: bool = True) -> dict[str, Any]:
    """σ²_u, σ²_e (MixedLM REML on the residuals when available, else method of moments)."""
    from ..reliability import icc_components

    if use_statsmodels and frame is not None:
        try:
            import pandas as pd
            import statsmodels.formula.api as smf

            df = pd.DataFrame({"r": resid, "g": ids})
            fit = smf.mixedlm("r ~ 1", df, groups=df["g"]).fit(reml=True, method="lbfgs", disp=False)
            su, se = float(np.asarray(fit.cov_re)[0, 0]), float(fit.scale)
            return {"var_u": su, "var_e": se, "method": "mixedlm_reml", "n_athletes": int(np.unique(ids).size), "n_obs": int(ids.size)}
        except Exception:                                           # noqa: BLE001 — fall back to MoM, recorded
            pass
    c = icc_components(resid, ids)
    return {"var_u": c["var_between_raw"], "var_e": c["var_within"], "method": "method_of_moments", "n_athletes": c["n_athletes"], "n_obs": c["n_obs"]}


def athlete_effects(events: Any, metric: str, domain: str, fold_of: dict[int, int], use_statsmodels: bool = True, min_athletes: int = 30) -> tuple[Any, list[dict]]:
    """(BLUPs per athlete computed with parameters from the other folds, per-fold variance records)."""
    import polars as pl

    d = _long(events, metric, domain).with_columns(pl.col("nfl_id").replace_strict(fold_of, default=-1, return_dtype=pl.Int64).alias("fold"))
    out, comps = [], []
    for f in sorted(set(d["fold"].to_list()) - {-1}):
        tr, te = d.filter(pl.col("fold") != f), d.filter(pl.col("fold") == f)
        if tr["nfl_id"].n_unique() < min_athletes or not te.height:
            continue
        means, grand = context_means(tr)
        rtr = residualise(tr, means, grand)
        c = components(rtr, tr["nfl_id"].to_numpy(), tr, use_statsmodels)
        vu, ve = c["var_u"], c["var_e"]
        comps.append({"metric": metric, "domain": domain, "fold": f, **c, "var_context": float(np.var(tr["y"].to_numpy() - rtr)), "var_total": float(np.var(tr["y"].to_numpy()))})
        rte = residualise(te, means, grand)
        te = te.with_columns(pl.Series("r", rte))
        g = te.group_by("nfl_id", "position_group").agg(pl.col("r").mean().alias("mean_resid"), pl.len().alias("n_events"))
        n = g["n_events"].to_numpy().astype(float)
        shrink = vu / (vu + ve / n) if (np.isfinite(vu) and vu > 0 and np.isfinite(ve)) else np.zeros(n.size)
        out.append(g.with_columns(pl.Series("shrink", shrink), (pl.col("mean_resid") * pl.Series(shrink)).alias("blup"), pl.lit(metric).alias("metric"),
                                  pl.lit(domain).alias("domain"), pl.lit(f).alias("fold")))
    return (pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()), comps


def variance_summary(comps: Sequence[dict[str, Any]], k_grid: Sequence[int] = (1, 2, 4, 8, 16)) -> Any:
    """Median over folds per (metric, domain): components, shares and k-event reliability."""
    import polars as pl

    if not comps:
        return pl.DataFrame()
    t = pl.DataFrame(list(comps), infer_schema_length=None)
    s = t.group_by("metric", "domain").agg(pl.col("var_u").median(), pl.col("var_e").median(), pl.col("var_context").median(), pl.col("var_total").median(),
                                           pl.col("method").first(), pl.col("n_athletes").median(), pl.len().alias("n_folds"))
    s = s.with_columns((pl.col("var_u") / (pl.col("var_u") + pl.col("var_e"))).alias("athlete_share_residual"),
                       (pl.col("var_u") / pl.col("var_total")).alias("athlete_share_total"))
    return s.with_columns([(pl.col("var_u") / (pl.col("var_u") + pl.col("var_e") / k)).alias(f"reliability_k{k}") for k in k_grid]).sort("metric", "domain")


def cross_domain_r(effects: Any, n_boot: int, seed: int, min_n: int = 30) -> Any:
    """Within-position residualised Pearson r of Combine vs NFL BLUPs per metric, disattenuated by mean shrinkage."""
    import polars as pl

    from ..m16.target_reliability import _pos_resid

    rows = []
    rng = np.random.default_rng(seed)
    for m in effects["metric"].unique().to_list() if effects.height else []:
        c = effects.filter((pl.col("metric") == m) & (pl.col("domain") == "combine")).select("nfl_id", "position_group", pl.col("blup").alias("c"), pl.col("shrink").alias("rc"))
        n = effects.filter((pl.col("metric") == m) & (pl.col("domain") == "nfl")).select("nfl_id", pl.col("blup").alias("n"), pl.col("shrink").alias("rn"))
        j = c.join(n, on="nfl_id").drop_nulls()
        if j.height < min_n:
            rows.append({"metric": m, "n_athletes": j.height})
            continue
        pos = j["position_group"].fill_null("?").to_numpy()
        a, b = _pos_resid(j["c"].to_numpy(), pos), _pos_resid(j["n"].to_numpy(), pos)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        r = float(np.corrcoef(a, b)[0, 1])
        relc, reln = float(j["rc"].mean()), float(j["rn"].mean())
        den = np.sqrt(relc * reln) if relc > 0 and reln > 0 else NAN
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, a.size, a.size)
            if a[i].std() > 0 and b[i].std() > 0:
                bs.append(np.corrcoef(a[i], b[i])[0, 1])
        bs = np.asarray(bs)
        rows.append({"metric": m, "n_athletes": int(a.size), "r": r, "r_lo": float(np.quantile(bs, 0.025)) if bs.size else NAN,
                     "r_hi": float(np.quantile(bs, 0.975)) if bs.size else NAN, "reliability_combine": relc, "reliability_nfl": reln,
                     "r_disattenuated": r / den if np.isfinite(den) else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


def run_all(events: Any, metrics: Sequence[str], fold_of: dict[int, int], use_statsmodels: bool, min_athletes: int, n_boot: int, seed: int) -> dict[str, Any]:
    import polars as pl

    effs, comps = [], []
    for m in metrics:
        if m not in events.columns:
            continue
        for dom in ("combine", "nfl"):
            e, c = athlete_effects(events, m, dom, fold_of, use_statsmodels, min_athletes)
            if e.height:
                effs.append(e)
            comps += c
    eff = pl.concat(effs, how="diagonal_relaxed") if effs else pl.DataFrame()
    return {"effects": eff, "components": pl.DataFrame(comps, infer_schema_length=None) if comps else pl.DataFrame(),
            "summary": variance_summary(comps), "cross_domain": cross_domain_r(eff, n_boot, seed) if eff.height else pl.DataFrame()}


__all__ = ["athlete_effects", "components", "context_means", "cross_domain_r", "residualise", "run_all", "variance_summary"]
