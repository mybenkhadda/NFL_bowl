"""White-background report layer of M14: notebook HTML helpers (banner, callout, styled tables, status badges),
Markdown tables, Markdown → HTML with a print-friendly white stylesheet, and the final scientific report.

Design: white surfaces, dark ink text, colour only as thin borders / small badges (no blue or navy filled panels).
Every number in a report comes from a saved table passed in; nothing is hard-coded.
"""

from __future__ import annotations

import html as _html
from typing import Any, Sequence

import numpy as np

INK, INK2, BORDER, SURFACE, ACCENT = "#22211f", "#5f5e5a", "#e4e2dc", "#ffffff", "#2a78d6"
STATUS_COLORS = {"validated": "#0f8a5f", "exploratory": "#b07800", "inconclusive": "#6f6e69", "blocked": "#5a48c8", "invalid": "#c8322f",
                 "computable": "#0f8a5f", "pass": "#0f8a5f", "FAIL": "#c8322f", "model_better": "#0f8a5f", "model_worse": "#c8322f"}
KIND_COLORS = {"note": ACCENT, "finding": "#0f8a5f", "warning": "#b07800", "caveat": "#c8322f", "method": INK2}

CSS = f"""
body {{ background: {SURFACE}; color: {INK}; font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.55;
        max-width: 1080px; margin: 32px auto; padding: 0 16px; }}
h1 {{ font-size: 1.7em; border-bottom: 2px solid {INK}; padding-bottom: 6px; }}
h2 {{ font-size: 1.25em; margin-top: 1.8em; border-bottom: 1px solid {BORDER}; padding-bottom: 4px; }}
h3 {{ font-size: 1.05em; color: {INK2}; }}
table {{ border-collapse: collapse; margin: 12px 0; font-size: 0.88em; width: 100%; }}
th {{ text-align: left; border-bottom: 2px solid {INK}; padding: 5px 8px; background: {SURFACE}; }}
td {{ border-bottom: 1px solid {BORDER}; padding: 4px 8px; vertical-align: top; }}
code {{ background: #f6f5f2; padding: 1px 4px; border-radius: 3px; }}
img {{ max-width: 100%; border: 1px solid {BORDER}; }}
blockquote {{ border-left: 3px solid {ACCENT}; margin: 12px 0; padding: 4px 14px; color: {INK2}; background: {SURFACE}; }}
.small {{ color: {INK2}; font-size: 0.85em; }}
"""


# ====================================================================== notebook helpers
def banner_html(part: str, title: str, question: str) -> str:
    return (f"<div style='background:{SURFACE};border-left:4px solid {INK};padding:10px 16px;margin:14px 0 8px 0;font-family:sans-serif'>"
            f"<div style='color:{INK2};font-size:11px;letter-spacing:.08em;text-transform:uppercase'>Section {_html.escape(part)}</div>"
            f"<div style='color:{INK};font-size:19px;font-weight:600;margin:2px 0'>{_html.escape(title)}</div>"
            f"<div style='color:{INK2};font-size:13px'><b>Question.</b> {_html.escape(question)}</div></div>")


def callout_html(text: str, kind: str = "note") -> str:
    c = KIND_COLORS.get(kind, ACCENT)
    return (f"<div style='background:{SURFACE};border:1px solid {BORDER};border-left:4px solid {c};padding:8px 14px;margin:8px 0;"
            f"color:{INK};font-family:sans-serif;font-size:13px'><b style='color:{c};text-transform:uppercase;font-size:11px'>{_html.escape(kind)}</b>"
            f"<div>{_html.escape(text)}</div></div>")


def styled(df: Any, digits: int = 4, max_rows: int = 40, status: str | Sequence[str] | None = None) -> Any:
    """pandas Styler of the first ``max_rows`` rows on a white table, status columns coloured as text (None when empty)."""
    if df is None or not getattr(df, "height", 0):
        return None
    try:
        pdf = df.head(max_rows).to_pandas()
    except Exception:                                                  # noqa: BLE001 — pyarrow missing → plain repr
        return None
    s = pdf.style.format(precision=digits, na_rep="—")
    s = s.set_table_styles([{"selector": "th", "props": f"text-align:left;border-bottom:2px solid {INK};background:{SURFACE};color:{INK}"},
                            {"selector": "td", "props": f"border-bottom:1px solid {BORDER};background:{SURFACE};color:{INK}"}])
    cols = [status] if isinstance(status, str) else list(status or [])
    for c in cols:
        if c in pdf.columns:
            s = s.map(lambda v: f"color:{STATUS_COLORS.get(str(v), INK)};font-weight:600", subset=[c])
    return s


# ====================================================================== markdown
def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return "—" if not np.isfinite(v) else f"{float(v):.{digits}f}"
    return str(v).replace("|", "/")


def md_table(df: Any, cols: Sequence[str] | None = None, digits: int = 4, max_rows: int = 30) -> str:
    if df is None or not getattr(df, "height", 0):
        return "_no rows_\n"
    cols = [c for c in (cols or df.columns) if c in df.columns]
    L = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.head(max_rows).iter_rows(named=True):
        L.append("| " + " | ".join(fmt(r[c], digits) for c in cols) + " |")
    if df.height > max_rows:
        L.append(f"\n_{df.height - max_rows} more rows in the saved table._")
    return "\n".join(L) + "\n"


def md_to_html(md: str, title: str) -> str:
    try:
        import markdown

        body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    except ImportError:
        body = f"<pre>{_html.escape(md)}</pre>"
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{_html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"


def write_report(run: Any, md: str, name: str, subdir: str = "") -> list[Any]:
    """``reports/markdown/<subdir>/<name>.md`` + ``reports/html/<subdir>/<name>.html`` (white stylesheet), registered."""
    from ...research.artifacts import safe_name

    nm = safe_name(name)
    sd = f"{subdir.strip('/')}/" if subdir else ""
    for top in ("markdown", "html"):
        (run.root / "reports" / top / sd).mkdir(parents=True, exist_ok=True)
    a = run.write_text(md, f"reports/markdown/{sd}{nm}.md")
    b = run.write_text(md_to_html(md, name), f"reports/html/{sd}{nm}.html")
    run.register("report", nm, [a, b])
    return [a, b]


def write_profile(run: Any, md: str, name: str) -> list[Any]:
    """Athlete / pair case study → ``reports/athlete_profiles/<name>.md|.html``."""
    from ...research.artifacts import safe_name

    nm = safe_name(name)
    a = run.write_text(md, f"reports/athlete_profiles/{nm}.md")
    b = run.write_text(md_to_html(md, name), f"reports/athlete_profiles/{nm}.html")
    run.register("profile", nm, [a, b])
    return [a, b]


# ====================================================================== final report
CAVEATS = (
    "All predictive claims are made on one common held-out cohort (strict_common, athlete-disjoint outer folds); RMSE is never compared across unmatched populations.",
    "Feature selection inside every model is outcome-free and fitted on training athletes only; no held-out outcome or held-out trial enters a selection.",
    "Movement metrics are kinematic proxies from 10 Hz tracking: no muscle force, power, tendon elasticity or neural drive is inferred.",
    "Phase timings (braking release → propulsion) are movement transitions, not reaction times: there are no stimulus timestamps.",
    "Directional asymmetry is a movement description and is never interpreted as injury risk.",
    "Movement smoothness (SPARC) is descriptive; it is not a causal mechanism.",
    "NFL-window metrics (m14n__*) are outcomes only; future NFL tracking never enters a Combine-time predictor.",
    "High reliability does not imply predictive value; a reliable metric can still add nothing beyond M1.",
    "Multi-player contextual metrics (pursuit, separation, defensive recovery) are blocked: the bundle has no synchronised x / y tracking.",
    "Position explains most of the target variance; within-position R² and position-specific results are reported to separate athletic signal from position.",
    "Status 'validated' means supported on this cohort under the declared rule; it is not validated scouting guidance until replicated on independent athletes.",
)


def final_report(R: dict[str, Any]) -> str:
    """Markdown final report from the measured tables collected in ``R`` (keys documented in the notebook Part O)."""
    g = R.get
    L = ["# M14 — Beyond the Stopwatch: cross-sport movement intelligence and the Hidden Braking Advantage", "",
         "**Question.** Do reliable, interpretable movement features inspired by other sports reveal hidden athletic capabilities that predict NFL "
         "movement performance (frozen target `nfl.q90_shared`) beyond the official Combine measurements (M1), on the same held-out athletes?", "",
         f"<p class='small'>Run {fmt(g('run_dir'))} · bundle {fmt(g('bundle'))} · science {fmt(g('science'))} · nflbowl {fmt(g('version'))} · "
         f"generated {fmt(g('generated_at'))}</p>", "", "## 1. Outcome of the primary comparisons", ""]
    L.append(md_table(g("primary_status"), ["model", "reference", "delta_rmse", "delta_lo", "delta_hi", "p_adjusted", "adjustment", "folds_better", "status"], 4))
    L += ["Δ RMSE = RMSE(reference) − RMSE(model): positive = the M14 model predicts better. Primary family adjusted by Holm.", ""]
    L += ["## 2. Claim cohort and model metrics", "", md_table(g("cohort_desc"), ["cohort", "n_athletes", "definition"]),
          md_table(g("metrics"), ["model", "n_athletes", "rmse", "mae", "r2", "r2_within_position", "fold_rmse_sd"], 4, 40)]
    L += ["## 3. Metric feasibility and registry", "", md_table(g("availability_summary"), None, 0, 20),
          md_table(g("feasibility"), None, 3, 20)]
    L += ["## 4. Measurement reliability and robustness", "",
          md_table(g("mstatus"), ["column", "measurement_status", "reliability_estimator", "reliability", "reliability_lo", "min_perturbation_spearman",
                                  "coverage", "reasons"], 3, 60)]
    L += ["## 5. Novelty beyond official testing", "",
          md_table(g("novelty"), ["column", "oof_r2_m1", "oof_r2_m1_position", "novelty", "novel", "max_rho_existing", "max_rho_existing_feature"], 3, 60)]
    L += ["## 6. Hidden Braking Advantage", "", md_table(g("twin_summary"), None, 4, 20), md_table(g("hidden_variation"), None, 3, 20)]
    L += ["## 7. Family, single-metric and position results", "",
          md_table(g("family_status"), ["model", "reference", "delta_rmse", "delta_lo", "delta_hi", "p_adjusted", "status"], 4, 30),
          md_table(g("single_status"), ["model", "delta_rmse", "delta_lo", "delta_hi", "p_adjusted", "status"], 4, 40),
          md_table(g("position_summary"), None, 4, 40)]
    L += ["## 8. Robustness", "", md_table(g("robustness"), None, 4, 30)]
    L += ["## 9. Evidence synthesis", "", md_table(g("answers"), ["question", "answer"], 3, 20),
          md_table(g("evidence"), ["metric", "scope", "family", "measurement_status", "single_metric_status", "family_status", "final_status",
                                   "scouting_hypothesis"], 3, 80)]
    L += ["## 10. Scientific limitations and safeguards", ""] + [f"- {c}" for c in CAVEATS] + [""]
    L += ["## 11. Provenance", "", md_table(g("provenance"), None, 3, 40)]
    return "\n".join(L)


__all__ = ["CAVEATS", "CSS", "STATUS_COLORS", "banner_html", "callout_html", "final_report", "fmt", "md_table", "md_to_html", "styled", "write_profile",
           "write_report"]
