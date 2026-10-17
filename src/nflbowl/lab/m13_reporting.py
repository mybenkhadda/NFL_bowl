"""Parts J + K — decision-relevant synthesis and reporting of M13.

Everything here is computed from saved / in-memory result TABLES; no number is typed into a narrative. The outcome
rule is declared before any result is read:

    embedding evidence  = some M1+<encoder> has Δ RMSE vs M1 with 95 % lower bound > 0 AND Holm-adjusted p < 0.05
                          (primary family: every evaluated encoder vs M1, strict common cohort)
    biomech evidence    = some M1+BIO_* has lower bound > 0 AND BH-adjusted p < 0.05 within the BIO family
    precise null        = every interval includes 0 AND every half-width ≤ ``precision_share`` × RMSE(M1)

    A  embeddings only · B  biomechanics only · C  both · D  neither, with precise nulls · otherwise INCONCLUSIVE.
    If a family was not evaluated (prerequisite missing), the outcome cannot be A / C (or B / C) and says so.

Reports: ``reports/markdown/M13_FINAL_RESEARCH_REPORT.md`` and ``reports/html/M13_FINAL_RESEARCH_REPORT.html`` (navy
theme; figures linked from ``figures/png``).
"""

from __future__ import annotations

import html as _html
import math
from typing import Any, Sequence

import numpy as np

from ..utils.reproducibility import utc_now
from .synthesis import md_table

NAN = float("nan")
OUTCOMES = {
    "A": ("Learned embeddings improve NFL capacity prediction", "Proceed to robustness checks and independent validation of the embedding signal."),
    "B": ("Interpretable biomechanics improve prediction; embeddings do not", "Prioritize reliable, physically interpretable biomechanical features."),
    "C": ("Both improve, with complementary signal", "Test a combined representation-plus-biomechanics model on the same folds."),
    "D": ("Neither improves beyond M1 (precise null)", "Revisit target design, domain matching, feature definitions and the competition's actionable outcome."),
    "INCONCLUSIVE": ("Inconclusive", "Intervals or diagnostics are inadequate for a decision: increase evaluable athletes / repeat with more data before choosing."),
}


def _num(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else NAN
    except (TypeError, ValueError):
        return NAN


# ====================================================================== J — unified comparison
def unified_comparison(paired: Any, metrics: Any, retrieval: Any | None, domain: Any | None, reliability: dict[str, str] | None, cohort: str = "strict_common") -> Any:
    """Model · RMSE · Δ vs M1 [CI] · within-position R² · retrieval AUC · domain AUC · reliability evidence · verdict."""
    import polars as pl

    from .figures import short

    p = paired.filter((pl.col("cohort") == cohort) & (pl.col("reference") == "M1")).select(
        "model", "delta_rmse", "delta_lo", "delta_hi", "p_boot_le0", *[c for c in ("p_holm_primary", "p_bh_all") if c in paired.columns])
    pf = paired.filter((pl.col("cohort") == cohort) & (pl.col("reference") == "M1+F0")).select(
        "model", pl.col("delta_rmse").alias("delta_vs_F0"), pl.col("delta_lo").alias("delta_vs_F0_lo"), pl.col("delta_hi").alias("delta_vs_F0_hi"))
    m = metrics.filter(pl.col("cohort") == cohort).select("model", "n_athletes", "rmse", "r2", "r2_within_position", "skill_vs_position_oof", "fold_rmse_sd")
    d = m.join(p, on="model", how="left").join(pf, on="model", how="left")
    d = d.with_columns(pl.col("model").map_elements(lambda s: short(s).removeprefix("M1+"), return_dtype=pl.String).alias("_rep"))
    if retrieval is not None and retrieval.height:
        r = retrieval.group_by("representation").agg(pl.col("retrieval_auc").mean().alias("retrieval_auc")).with_columns(
            pl.col("representation").map_elements(short, return_dtype=pl.String).alias("_rep"))
        d = d.join(r.select("_rep", "retrieval_auc"), on="_rep", how="left")
    else:
        d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("retrieval_auc"))
    if domain is not None and domain.height:
        o = domain.filter(pl.col("condition") == "original").select(pl.col("representation").map_elements(short, return_dtype=pl.String).alias("_rep"),
                                                                    pl.col("domain_auc_grouped").alias("domain_auc"))
        d = d.join(o, on="_rep", how="left")
    else:
        d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("domain_auc"))
    rel = reliability or {}
    d = d.with_columns(pl.col("model").map_elements(lambda s: rel.get(s, "—"), return_dtype=pl.String).alias("reliability_evidence"))
    d = d.with_columns(pl.when(pl.col("model") == "M1").then(pl.lit("reference"))
                       .when(pl.col("delta_lo").is_null()).then(pl.lit("not compared"))
                       .when(pl.col("delta_lo") > 0).then(pl.lit("improves on M1 (CI > 0)"))
                       .when(pl.col("delta_hi") < 0).then(pl.lit("worse than M1 (CI < 0)"))
                       .otherwise(pl.lit("inconclusive (CI includes 0)")).alias("verdict"))
    return d.drop("_rep").sort("rmse")


def determine_outcome(paired: Any, cohort: str, encoder_models: Sequence[str], bio_models: Sequence[str], precision_share: float = 0.02,
                      alpha: float = 0.05) -> dict[str, Any]:
    import polars as pl

    from .. import _backend

    ae = _backend.ae()
    d = paired.filter((pl.col("cohort") == cohort) & (pl.col("reference") == "M1"))
    m1_rmse = _num(d["rmse_ref"].max()) if d.height else NAN
    enc = d.filter(pl.col("model").is_in(list(encoder_models)))
    bio = d.filter(pl.col("model").is_in(list(bio_models)))
    lines = []
    emb_eval, bio_eval = enc.height > 0, bio.height > 0
    emb_pos = False
    if emb_eval:
        holm = ae.holm(enc["p_boot_le0"].to_list())
        enc = enc.with_columns(pl.Series("p_holm", holm))
        hits = enc.filter((pl.col("delta_lo") > 0) & (pl.col("p_holm") < alpha))
        emb_pos = hits.height > 0
        lines.append(f"Embedding family ({enc.height} encoders, Holm): {hits.height} with a positive interval and p_holm < {alpha}"
                     + (f" — {', '.join(hits['model'].to_list())}" if hits.height else ""))
    else:
        lines.append("Embedding family NOT evaluated (no encoder ready on every fold).")
    bio_pos = False
    if bio_eval:
        bh = ae.bh(bio["p_boot_le0"].to_list())
        bio = bio.with_columns(pl.Series("p_bh_bio", bh))
        hits = bio.filter((pl.col("delta_lo") > 0) & (pl.col("p_bh_bio") < alpha))
        bio_pos = hits.height > 0
        lines.append(f"Biomechanics family ({bio.height} models, BH): {hits.height} with a positive interval and p_bh < {alpha}"
                     + (f" — {', '.join(hits['model'].to_list())}" if hits.height else ""))
    else:
        lines.append("Biomechanics family NOT evaluated.")
    fam = pl.concat([x.select("model", "delta_lo", "delta_hi") for x in (enc, bio) if x.height], how="vertical") if (emb_eval or bio_eval) else pl.DataFrame()
    half = ((fam["delta_hi"] - fam["delta_lo"]) / 2).to_numpy() if fam.height else np.array([])
    thr = precision_share * m1_rmse if np.isfinite(m1_rmse) else NAN
    precise = bool(half.size and np.isfinite(thr) and np.all(half <= thr))
    lines.append(f"Precision: max CI half-width {np.max(half) if half.size else NAN:.4f} vs threshold {thr:.4f} (= {precision_share:.0%} of RMSE(M1) {m1_rmse:.4f})")
    if emb_pos and bio_pos:
        key = "C"
    elif emb_pos:
        key = "A" if bio_eval else "INCONCLUSIVE"
    elif bio_pos:
        key = "B" if emb_eval else "INCONCLUSIVE"
    elif precise and emb_eval and bio_eval:
        key = "D"
    else:
        key = "INCONCLUSIVE"
    if key == "INCONCLUSIVE" and emb_pos and not bio_eval:
        lines.append("Embeddings improve but biomechanics were not evaluated: A vs C cannot be distinguished.")
    if key == "INCONCLUSIVE" and bio_pos and not emb_eval:
        lines.append("Biomechanics improve but embeddings were not evaluated: B vs C cannot be distinguished.")
    title, rec = OUTCOMES[key]
    status = {"A": "good", "B": "good", "C": "good", "D": "warning", "INCONCLUSIVE": "neutral"}[key]
    return {"outcome": key, "title": title, "recommendation": rec, "evidence": lines, "status": status, "cohort": cohort, "m1_rmse": m1_rmse,
            "precision_threshold": thr, "embedding_evaluated": emb_eval, "biomech_evaluated": bio_eval, "decided_at": utc_now()}


def evidence_rows(unified: Any, fold_tab: Any | None, domain: Any | None) -> Any:
    """Long evidence grid for the evidence-matrix figure: model × criterion → status + short label (measured values)."""
    import polars as pl

    rows = []
    for r in unified.iter_rows(named=True):
        m = r["model"]
        if m in ("MEAN",):
            continue

        def add(crit: str, status: str, label: str) -> None:
            rows.append({"model": m, "criterion": crit, "status": status, "label": label})

        lo, hi, dl = _num(r.get("delta_lo")), _num(r.get("delta_hi")), _num(r.get("delta_rmse"))
        if m == "M1":
            add("Δ vs M1", "neutral", "reference")
        elif np.isfinite(lo):
            add("Δ vs M1", "good" if lo > 0 else ("critical" if hi < 0 else "warning"), f"{dl:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        else:
            add("Δ vs M1", "neutral", "—")
        lo2, hi2 = _num(r.get("delta_vs_F0_lo")), _num(r.get("delta_vs_F0_hi"))
        add("Δ vs M1+F0", "neutral" if not np.isfinite(lo2) else ("good" if lo2 > 0 else ("critical" if hi2 < 0 else "warning")),
            "—" if not np.isfinite(lo2) else f"{_num(r.get('delta_vs_F0')):+.3f}")
        sk = _num(r.get("skill_vs_position_oof"))
        add("skill beyond position", "neutral" if not np.isfinite(sk) else ("good" if sk > 0.02 else ("warning" if sk > 0 else "critical")),
            "—" if not np.isfinite(sk) else f"{sk:+.3f}")
        if fold_tab is not None and fold_tab.height and m != "M1" and "delta_rmse_vs_M1" in fold_tab.columns:
            g = fold_tab.filter(pl.col("model") == m)
            k = int((g["delta_rmse_vs_M1"] > 0).sum()) if g.height else 0
            add("folds better than M1", "good" if g.height and k == g.height else ("warning" if k > 0 else "critical"), f"{k}/{g.height}")
        ra = _num(r.get("retrieval_auc"))
        add("retrieval AUC", "neutral" if not np.isfinite(ra) else ("good" if ra >= 0.6 else ("warning" if ra >= 0.52 else "critical")),
            "—" if not np.isfinite(ra) else f"{ra:.3f}")
        da = _num(r.get("domain_auc"))
        add("domain AUC", "neutral" if not np.isfinite(da) else ("good" if da <= 0.7 else ("warning" if da <= 0.9 else "critical")),
            "—" if not np.isfinite(da) else f"{da:.3f}")
        add("reliability", "neutral", str(r.get("reliability_evidence") or "—")[:24])
    return pl.DataFrame(rows)


def read_paired(paired: Any, cohort: str, reference: str, max_rows: int = 12) -> str:
    """Plain-language reading of measured paired comparisons (one line per model; interval wording never claims 'no effect')."""
    import polars as pl

    from .figures import short

    if paired is None or not paired.height:
        return "No paired comparisons were produced."
    d = paired.filter((pl.col("cohort") == cohort) & (pl.col("reference") == reference)).sort("delta_rmse", descending=True).head(max_rows)
    out = [f"Cohort {cohort} · reference {short(reference)} · Δ RMSE = RMSE(reference) − RMSE(model), positive = model better (yd/s²):"]
    for r in d.iter_rows(named=True):
        lo, hi, est = _num(r["delta_lo"]), _num(r["delta_hi"]), _num(r["delta_rmse"])
        if lo > 0:
            word = "better than the reference (interval excludes 0)"
        elif hi < 0:
            word = "worse than the reference (interval excludes 0)"
        else:
            word = "not distinguishable from the reference (interval includes 0; not evidence of no effect)"
        out.append(f"• {short(r['model'])}: {est:+.4f} [{lo:+.4f}, {hi:+.4f}], n = {r['n_athletes']} — {word}")
    return "\n".join(out)


def caveats(R: dict[str, Any]) -> list[str]:
    """Automated caveats from the measured state (never generic boilerplate only)."""
    out = ["The target nfl.q90_shared is a constructed NFL movement-capacity quantity (speed-conditioned upper acceleration), not an NFL success outcome.",
           "Prediction validation is athlete-disjoint; target construction (estimator choice, shared bins, C3 pooling if frozen) used all athletes once "
           "— the end-to-end pipeline is not strictly inductive."]
    c = R.get("cohorts_desc")
    if c is not None and c.height:
        d = dict(zip(c["cohort"].to_list(), c["n_athletes"].to_list()))
        out.append(f"Performance claims use the strict common cohort ({d.get('strict_common')} athletes) out of {d.get('ladder_all')} ladder athletes; "
                   "native-coverage numbers are descriptive and not compared across models.")
    if R.get("missing_encoders"):
        out.append(f"Encoders excluded (missing inductive checkpoints on some folds): {R['missing_encoders']} — no transductive fallback was used.")
    f1 = R.get("f1_validity") or {}
    if f1.get("verdict") not in (None, "valid", "absent"):
        out.append(f"F1 TCN fold validity: {f1.get('verdict')} ({f1.get('detail')}) — F1 is excluded from primary comparisons.")
    rc = R.get("repro_check") or {}
    if rc and rc.get("status") != "pass":
        out.append(f"Reproducibility check FAILED: the baseline run's M1 differs from the ladder's M1 ({rc}). Position-only comparisons are suspect.")
    iv = R.get("interval_check") or {}
    if iv and not iv.get("valid"):
        out.append(f"OOF prediction intervals miss nominal coverage ({iv.get('coverage'):.2f} vs {iv.get('level')}): intervals are not shown in profiles.")
    pt = R.get("position_table")
    if pt is not None and pt.height and "small_group" in pt.columns:
        small = pt.filter(pt["small_group"])["position_group"].to_list()
        if small:
            out.append(f"Position groups with fewer than the minimum athletes (no interval, no conclusion): {', '.join(map(str, small))}.")
    out.append("Interactions, help/hurt subgroups and athlete profiles are exploratory and non-causal; subgroup intervals are not multiplicity-controlled "
               "beyond the BH adjustment stated in each table.")
    return out


def stage_summary(run: Any) -> Any:
    import polars as pl

    t = run.stage_table()
    if not t.height:
        return t
    return t.with_columns(pl.col("status").replace({"executed": "ran", "reused": "reused cached outputs", "skipped": "skipped", "blocked": "blocked", "failed": "FAILED",
                                                    "running": "interrupted"}).alias("execution"))


# ====================================================================== K — reports
def _fig_md(name: str, caption: str, rel_prefix: str = "../../figures/png/") -> str:
    return f"![{_html.escape(caption)}]({rel_prefix}{name}.png)\n\n_{caption}_\n"


def final_report(run: Any, cfg: Any, env: dict[str, Any], R: dict[str, Any]) -> str:
    """Markdown report from measured tables in ``R`` (keys documented in the notebook). Missing sections say so."""
    md = [f"# M13 — NFL Outcome Transfer, Reliability-Aware Biomechanics & Scouting Value", "",
          f"_Generated {utc_now()} from executed or verified cached results in `{run.root}`. Every number below is read from a saved table._", "",
          "## Provenance", "", md_table(R.get("provenance_table"), digits=4), "",
          "## Research question", "", "Can 10 Hz NFL Combine movement tracking improve prediction of future NFL movement capacity beyond traditional Combine "
          "measurements, under rigorous athlete-disjoint validation?", ""]
    o = R.get("outcome")
    if o:
        md += [f"## Decision: Outcome {o['outcome']} — {o['title']}", "", f"**Recommendation.** {o['recommendation']}", ""]
        md += [f"- {line}" for line in o["evidence"]] + [""]
    else:
        md += ["## Decision", "", "_Not evaluated: the synthesis stage did not run._", ""]
    md += ["## Unified comparison (strict common cohort)", "", md_table(R.get("unified"), ["model", "n_athletes", "rmse", "delta_rmse", "delta_lo", "delta_hi",
                                                                                          "r2_within_position", "skill_vs_position_oof", "retrieval_auc", "domain_auc",
                                                                                          "reliability_evidence", "verdict"], digits=4), ""]
    for key, title, cols in (("handoff_inventory_summary", "A — M12 checkpoint hand-off", None), ("reconciliation", "A3 — M12 reconciliation", None),
                             ("prerequisites", "A4 — prerequisites", ["requirement", "status", "detail"]),
                             ("target_provenance", "B1 — frozen target provenance", ["aspect", "finding", "implication"]),
                             ("attrition", "B2 — sample attrition", None), ("variance", "B3 — position structure of the target", None),
                             ("cohorts_desc", "D — cohorts", None), ("common_cohort_audit", "D — common-cohort audit", None),
                             ("metrics_strict", "C — ladder metrics (strict common cohort)", ["model", "n_athletes", "rmse", "mae", "r2", "r2_within_position",
                                                                                              "skill_vs_position_oof", "mean_error", "fold_rmse_sd"]),
                             ("paired_strict", "C — paired comparisons (strict common cohort)", ["model", "reference", "n_athletes", "delta_rmse", "delta_lo",
                                                                                                 "delta_hi", "p_boot_le0", "p_holm_primary", "verdict"]),
                             ("metrics_native", "C — native-coverage metrics (descriptive, not comparable)", ["model", "n_athletes", "rmse", "r2"]),
                             ("reliability_audit_summary", "E1 — reliability audit (summary)", None),
                             ("selection_paired", "E3 — reliability-aware selection", ["model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict"]),
                             ("interactions", "E5 — interactions (exploratory)", ["interaction", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "p_bh_family"]),
                             ("position_paired", "F — position-conditioned models", ["model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict"]),
                             ("domain_summary", "G — domain robustness (frozen embeddings)", ["representation", "condition", "domain_auc_grouped", "delta_auc_vs_original",
                                                                                               "mmd2", "retrieval_auc", "n_folds"]),
                             ("associations", "H — retrieval vs transfer", None), ("best_model", "I — model explained", None),
                             ("gain_by_group", "I — where tracking helps or hurts", ["variable", "level", "n", "mean_abs_error_gain", "lo", "hi", "note"])):
        v = R.get(key)
        md += [f"## {title}", ""]
        if v is None:
            md += ["_not evaluated in this run (see the stage table)._", ""]
        elif isinstance(v, dict):
            md += [f"- **{k}**: {val}" for k, val in v.items()] + [""]
        else:
            md += [md_table(v, cols, digits=4), ""]
    md += ["## Figures", ""]
    for f in (R.get("figures") or [])[:40]:
        md.append(_fig_md(f["figure"], f.get("caption") or f.get("question") or f["figure"]))
    md += ["## Automated caveats", ""] + [f"- {c}" for c in caveats(R)] + ["", "## Execution status", "", md_table(R.get("stages"), ["stage", "execution", "note"]), ""]
    md += ["## Environment", "", f"- nflbowl {env.get('versions', {}).get('nflbowl')} · revision {((env.get('revision') or {}).get('commit') or 'unknown')}",
           f"- GPU: {(env.get('gpu') or {}).get('device')} · precision {cfg.precision} · M13 seed {cfg.seed}", ""]
    return "\n".join(md)


CSS = """
:root{--bg:#07101d;--surface:#0b1626;--ink:#eaf2fb;--ink2:#b8c7d9;--muted:#7f93ab;--grid:#1a2a42;--accent:#22d3ee}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--accent:#0e7490}}
:root[data-theme="dark"]{--bg:#07101d;--surface:#0b1626;--ink:#eaf2fb;--ink2:#b8c7d9;--muted:#7f93ab;--grid:#1a2a42;--accent:#22d3ee}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.9rem;margin:.2rem 0 1rem;border-left:4px solid var(--accent);padding-left:.7rem}
h2{font-size:1.2rem;margin:2.2rem 0 .6rem;color:var(--ink);border-bottom:1px solid var(--grid);padding-bottom:.3rem}
p,li{color:var(--ink2)}strong{color:var(--ink)}em{color:var(--muted)}a{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:.82rem;display:block;overflow-x:auto;background:var(--surface);border-radius:8px}
th,td{padding:5px 9px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}th{color:var(--ink);position:sticky;top:0;background:var(--surface)}
td{color:var(--ink2);font-variant-numeric:tabular-nums}img{max-width:100%;border-radius:8px;border:1px solid var(--grid)}
code,pre{background:var(--surface);color:var(--ink);border-radius:4px;padding:0 .3em}
@media print{body{background:#fff;color:#000}table{display:table}}
"""


def report_html(md_text: str, title: str) -> str:
    try:
        import markdown

        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except Exception:                                               # noqa: BLE001 — readable fallback without the markdown package
        body = "<pre>" + _html.escape(md_text) + "</pre>"
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_html.escape(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")


def write_reports(run: Any, md_text: str, name: str = "M13_FINAL_RESEARCH_REPORT") -> dict[str, str]:
    """Markdown + navy HTML under reports/; registered in the artifact manifest."""
    p_md = run.write_text(md_text, f"reports/markdown/{name}.md")
    p_html = run.write_text(report_html(md_text, name.replace("_", " ")), f"reports/html/{name}.html")
    run.register("report", name, [p_md, p_html])
    return {"markdown": run.rel(p_md), "html": run.rel(p_html)}


__all__ = ["CSS", "OUTCOMES", "caveats", "determine_outcome", "evidence_rows", "final_report", "read_paired", "report_html", "stage_summary", "unified_comparison",
           "write_reports"]
