"""Part 11 — unified scientific evaluation: the comparison rows and validation tables of every executed stage, read
back from RUN_DIR (so reused and freshly executed stages are treated alike), CLASSIFIED but never pooled.

Evidence classes (one per row):

    diagnostic          software checks, synthetic recovery, nulls / negative controls (properties of the method, not of athletes)
    measurement         measurement reliability (ICC, split-half, test–retest, within-domain consistency, held-out forecast quality)
    transductive        label-free representation diagnostics on all athletes (12F transductive encoders, 12E cross-domain retrieval)
    in_sample_estimand  association estimands on all linked athletes (12B observed / corrected coefficients) — no held-out claim
    inductive           cross-fitted held-out comparisons on identical athletes (oof ladders of 12A–12G, inductive 12F)

``outcome_linked`` = the row involves an NFL target (Δ RMSE / Δ MSE ladders, 12B coefficients). Rows with the same
estimand (e.g. "retrieval AUC within position") are put side by side with an explicit ``estimand`` column; nothing is
averaged across tracks, targets or cohorts. Only the 12G tournament compares methods on identical athletes.
"""

from __future__ import annotations

from typing import Any

PREFIXES = ("m12c", "m12b", "m12a", "m12d", "m12e", "m12f")


def _t(run: Any, name: str) -> Any:
    try:
        return run.load_table(name) if run.has_table(name) else None
    except Exception:                                               # noqa: BLE001
        return None


def evidence_class(row: dict[str, Any]) -> str:
    reg, tgt, metric, src = row.get("training_regime") or "", row.get("evaluation_target") or "", row.get("metric") or "", row.get("source_table") or ""
    if reg == "transductive" or src in ("m12e_retrieval_summary", "m12e_auc_diff"):
        return "transductive"
    if tgt == "measurement_reliability":
        return "measurement"
    if metric.startswith("beta_"):
        return "in_sample_estimand"
    if "RMSE" in metric or "MSE" in metric:
        return "inductive"
    return "diagnostic"


def collect_comparisons(run: Any) -> Any:
    import polars as pl

    parts = [t for p in PREFIXES if (t := _t(run, f"{p}_comparisons")) is not None and t.height]
    if not parts:
        return pl.DataFrame()
    c = pl.concat(parts, how="diagonal_relaxed").unique(maintain_order=True)
    rows = list(c.iter_rows(named=True))
    cls = [evidence_class(r) for r in rows]
    linked = [r.get("evaluation_target") == "cross_domain_physical_transfer" and k in ("inductive", "in_sample_estimand") for r, k in zip(rows, cls)]
    return c.with_columns(pl.Series("evidence_class", cls, dtype=pl.String), pl.Series("outcome_linked", linked, dtype=pl.Boolean),
                          pl.lit("paired / athlete bootstrap, 95 % percentile interval (each track's own resampling unit)").alias("interval_method"))


def cross_domain_retrieval(run: Any) -> Any:
    """12E distribution retrieval and 12F embedding retrieval side by side (same estimand family; different query units)."""
    import polars as pl

    rows = []
    e = _t(run, "m12e_retrieval_summary")
    for r in (e.iter_rows(named=True) if e is not None else []):
        rows.append({"track": "12E", "representation": r["method"], "regime": "transductive (label-free; no fitting except the E1 metric)",
                     "estimand": "AUC of own NFL distribution among same-position athletes (athlete-level distributions)", "auc": r.get("auc"), "lo": r.get("auc_lo"),
                     "hi": r.get("auc_hi"), "n_athletes": r.get("n_athletes")})
    f = _t(run, "12f_metrics")
    for r in (f.iter_rows(named=True) if f is not None else []):
        rows.append({"track": "12F", "representation": r.get("representation"), "regime": f"transductive ({r.get('level')})",
                     "estimand": "AUC of own NFL embeddings among same-position athletes (Combine forty query, sequence embeddings)", "auc": r.get("cross_auc"),
                     "lo": r.get("cross_auc_lo"), "hi": r.get("cross_auc_hi"), "n_athletes": r.get("cross_n_athletes")})
    ind = _t(run, "12f_inductive_retrieval")
    if ind is not None and ind.height:
        s = ind.group_by("representation").agg(pl.col("auc").mean().alias("auc"), pl.col("auc").min().alias("lo"), pl.col("auc").max().alias("hi"), pl.col("n_athletes").sum())
        rows += [{"track": "12F", "representation": r["representation"], "regime": "INDUCTIVE (held-out folds)", "estimand": "mean over folds of held-out retrieval AUC (lo/hi = min/max fold)",
                  "auc": r["auc"], "lo": r["lo"], "hi": r["hi"], "n_athletes": r["n_athletes"]} for r in s.iter_rows(named=True)]
    return pl.DataFrame(rows, infer_schema_length=None)


def identity_controls(run: Any) -> Any:
    import polars as pl

    rows = []
    c = _t(run, "m12e_controls")
    for r in (c.iter_rows(named=True) if c is not None else []):
        rows.append({"track": "12E", "item": f"{r.get('method')} · {r.get('control')}", "observed": r.get("auc_observed"), "control": r.get("auc_control"), "p_value": r.get("p_value")})
    f = _t(run, "12f_metrics")
    for r in (f.iter_rows(named=True) if f is not None else []):
        if r.get("null_mean") is not None:
            rows.append({"track": "12F", "item": f"{r.get('representation')} · shuffled identity", "observed": r.get("cross_auc"), "control": r.get("null_mean"),
                         "p_value": r.get("null_exceed_share")})
    for tr, name in (("12C", "m12c_permutation"), ("12G", "m12g_tournament_permutation")):
        p = _t(run, name)
        for r in (p.iter_rows(named=True) if p is not None and p.height else []):
            rows.append({"track": tr, "item": "within-position athlete-label permutation of the block", "observed": r.get("observed"), "control": r.get("null_mean"), "p_value": r.get("p_value")})
    return pl.DataFrame(rows, infer_schema_length=None)


def domain_shift(run: Any) -> Any:
    """12F domain-classifier AUC / MMD² and the exact total-variation distance between Combine and NFL frame distributions (EDA)."""
    import polars as pl

    rows = []
    f = _t(run, "12f_metrics")
    for r in (f.iter_rows(named=True) if f is not None else []):
        rows += [{"source": "12F", "item": r.get("representation"), "metric": "domain_auc", "value": r.get("domain_auc")},
                 {"source": "12F", "item": r.get("representation"), "metric": "mmd2", "value": r.get("mmd2")}]
    h = _t(run, "eda_channel_histograms")
    if h is not None and h.height:
        for (ch,), g in h.partition_by(["channel"], as_dict=True).items():
            a = g.filter(pl.col("domain") == "combine").sort("lo")["share"].to_numpy()
            b = g.filter(pl.col("domain") == "nfl").sort("lo")["share"].to_numpy()
            if a.size and a.size == b.size:
                rows.append({"source": "EDA", "item": ch, "metric": "total_variation_distance (frames)", "value": float(0.5 * abs(a - b).sum())})
    return pl.DataFrame(rows, infer_schema_length=None)


def latent_consistency(run: Any) -> Any:
    """Reliability-type evidence of every track in one long table (each row keeps its own estimand)."""
    import polars as pl

    rows = []
    b = _t(run, "m12b_reliability")
    for r in (b.filter(pl.col("group") == "ALL").iter_rows(named=True) if b is not None else []):
        rows.append({"track": "12B", "item": r["feature"], "estimand": "ICC(1) of one attempt", "estimate": r.get("icc"), "lo": r.get("icc_lo"), "hi": r.get("icc_hi"), "n_athletes": r.get("n_athletes")})
    c = _t(run, "m12c_reliability")
    if c is not None:                                               # primary τ = M12Config.primary_tau = 0.90
        for r in c.filter((pl.col("split") == "odd_even:0") & ((pl.col("tau") - 0.9).abs() < 1e-9)).iter_rows(named=True):
            rows.append({"track": "12C", "item": r["estimator"], "estimand": "Spearman–Brown split-half (odd / even games), τ = 0.90 trait", "estimate": r.get("reliability"),
                         "lo": r.get("rel_lo"), "hi": r.get("rel_hi"), "n_athletes": r.get("n_athletes")})
    a = _t(run, "m12a_fpc_reliability")
    for r in (a.iter_rows(named=True) if a is not None else []):
        rows.append({"track": "12A", "item": f"FPC{r['component']}", "estimand": "attempt test–retest (Spearman–Brown)", "estimate": r.get("reliability"), "lo": r.get("rel_lo"),
                     "hi": r.get("rel_hi"), "n_athletes": r.get("n_athletes")})
    d = _t(run, "m12d_retest")
    for r in (d.iter_rows(named=True) if d is not None else []):
        rows.append({"track": "12D", "item": f"{r['model']} {r['factor']}", "estimand": "cross-fitted attempt test–retest r of the factor score", "estimate": r.get("retest_factor"),
                     "lo": None, "hi": None, "n_athletes": r.get("n_athletes")})
    e = _t(run, "m12e_consistency")
    for r in (e.iter_rows(named=True) if e is not None else []):
        rows.append({"track": "12E", "item": f"{r['method']} ({r['domain']})", "estimand": "within-domain retrieval AUC (identification)", "estimate": r.get("auc"), "lo": r.get("auc_lo"),
                     "hi": r.get("auc_hi"), "n_athletes": r.get("n_athletes")})
    return pl.DataFrame(rows, infer_schema_length=None)


def calibration(run: Any) -> Any:
    import polars as pl

    parts = []
    c = _t(run, "m12c_forecast_summary")
    if c is not None and c.height:
        parts.append(c.with_columns(pl.lit("12C").alias("track"), pl.lit("held-out-game quantile coverage (fit odd games, score even games)").alias("estimand")))
    b = _t(run, "m12b_interval_calibration")
    if b is not None and b.height:
        parts.append(b.with_columns(pl.lit("12B").alias("track"), pl.lit("cross-conformal predictive interval coverage").alias("estimand")))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def ablations(run: Any) -> Any:
    import polars as pl

    from ..models import registry as reg

    f = _t(run, "12f_metrics")
    if f is None or not f.height:
        return pl.DataFrame()
    abl = set(reg.ablations())
    return f.filter(pl.col("representation").is_in(list(abl) + [reg.resolve("F4"), reg.ENGINEERED, reg.RANDOM]))


def cross_validation(run: Any) -> Any:
    fb = _t(run, "eda_fold_balance")
    return fb if fb is not None else None


def build(run: Any) -> dict[str, Any]:
    """Every Part 11 table (saved under tables/ with an ``unified_`` prefix) + returns them for display."""
    import polars as pl

    comps = collect_comparisons(run)
    out = {"unified_comparisons": comps}
    if comps.height:
        for k in ("diagnostic", "measurement", "transductive", "in_sample_estimand", "inductive"):
            out[f"unified_{k}"] = comps.filter(pl.col("evidence_class") == k)
        out["unified_outcome_linked"] = comps.filter(pl.col("outcome_linked"))
        out["bootstrap_confidence_intervals"] = comps.filter(pl.col("lo").is_not_null() & pl.col("hi").is_not_null()).select(
            "track", "comparison", "metric", "estimate", "lo", "hi", "n_athletes", "evidence_class", "verdict", "interval_method")
        out["model_comparison"] = comps.filter(pl.col("primary")).select("track", "comparison", "metric", "estimate", "lo", "hi", "n_athletes", "evidence_class", "verdict")
    sc = _t(run, "software_checks")
    out["unified_software_checks"] = sc
    out["cross_domain_retrieval"] = cross_domain_retrieval(run)
    out["athlete_identity_controls"] = identity_controls(run)
    out["domain_shift_diagnostics"] = domain_shift(run)
    out["physical_property_probes"] = _t(run, "12f_probes")
    out["latent_trait_consistency"] = latent_consistency(run)
    out["distributional_calibration"] = calibration(run)
    out["model_ablations"] = ablations(run)
    out["cross_validation_folds"] = cross_validation(run)
    for k, v in out.items():
        if v is not None and getattr(v, "height", 0) and not k.startswith("unified_software"):
            run.save_table(v, k, "evaluation")
    return out


def figures(run: Any, tabs: dict[str, Any]) -> None:
    """Primary comparisons by evidence class; retrieval side by side; reliability evidence."""
    import matplotlib.pyplot as plt
    import numpy as np

    from .eda import SERIES, _fig, style

    def forest(df: Any, label_col: str, est: str, lo: str, hi: str, title: str, sub: str, name: str, ref: float | None = None, color_col: str | None = None) -> None:
        if df is None or not df.height:
            return
        style()
        d = df.head(40)
        fig, ax = _fig(title, sub)
        y = np.arange(d.height)[::-1]
        groups = sorted(set(d[color_col].to_list())) if color_col else [None]
        cmap = {g: SERIES[k % len(SERIES)] for k, g in enumerate(groups)}
        for k, r in enumerate(d.iter_rows(named=True)):
            col = cmap[r[color_col]] if color_col else SERIES[0]
            e, l, h = r.get(est), r.get(lo), r.get(hi)
            if l is not None and h is not None:
                ax.plot([l, h], [y[k], y[k]], color=col, linewidth=2)
            if e is not None:
                ax.plot([e], [y[k]], "o", color=col, markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.set_yticks(y, [str(s)[:90] for s in d[label_col].to_list()], fontsize=8.5)
        if ref is not None:
            ax.axvline(ref, color="#52514e", linewidth=1, linestyle="--")
        if color_col:
            import matplotlib.lines as ml

            ax.legend(handles=[ml.Line2D([], [], color=cmap[g], marker="o", linewidth=2, label=str(g)) for g in groups], loc="lower right")
        run.show(fig, name, "evaluation", {"rows": d})

    mc = tabs.get("model_comparison")
    if mc is not None and mc.height:
        mc2 = mc.with_columns((mc["track"] + " · " + mc["comparison"]).alias("label"))
        forest(mc2.filter(mc2["evidence_class"] == "inductive"), "label", "estimate", "lo", "hi", "Declared primary comparisons — inductive (held-out, identical athletes per comparison)",
               "Estimate with 95 % interval in each comparison's own units (Δ RMSE / Δ MSE: positive = better). Rows are not pooled across tracks.", "unified_primary_inductive", ref=0.0, color_col="track")
        forest(mc2.filter(mc2["evidence_class"] != "inductive"), "label", "estimate", "lo", "hi", "Declared primary comparisons — measurement and transductive evidence",
               "Reliability differences, Δ test–retest r, Δ AUC and AUC − 0.5 in their own units; 0 = no difference / chance.", "unified_primary_measurement_transductive", ref=0.0, color_col="track")
    rr = tabs.get("cross_domain_retrieval")
    if rr is not None and rr.height:
        rr2 = rr.with_columns((rr["track"] + " · " + rr["representation"] + " · " + rr["regime"]).alias("label"))
        forest(rr2, "label", "auc", "lo", "hi", "Cross-domain athlete retrieval: distributions (12E) and embeddings (12F)",
               "Same estimand family (own NFL movement among same-position athletes), different query units — compare within a track, not across. 0.5 = chance.",
               "unified_cross_domain_retrieval", ref=0.5, color_col="track")
    lc = tabs.get("latent_trait_consistency")
    if lc is not None and lc.height:
        lc2 = lc.with_columns((lc["track"] + " · " + lc["item"]).alias("label"))
        forest(lc2, "label", "estimate", "lo", "hi", "Measurement reliability evidence across tracks",
               "ICC, split-half, test–retest and within-domain identification, each in its own estimand (see the table). Higher = more reproducible.", "unified_reliability_evidence",
               color_col="track")
    plt.close("all")


__all__ = ["build", "collect_comparisons", "cross_domain_retrieval", "domain_shift", "evidence_class", "figures", "identity_controls", "latent_consistency"]
