"""M13 visual identity: a dark navy research theme with a cyan accent, applied to the shared lab figure library, plus the
figure types M13 adds (case dumbbells, reliability curves, evidence matrix, executive dashboard).

``apply_theme('navy')`` re-points the colour globals of ``nflbowl.visualization.plots`` and ``nflbowl.lab.figures``
(read at call time by every figure function) to the navy surface and the reference palette's DARK-mode categorical
steps, in the same fixed slot order — identity colours keep their entity across every M13 figure.
``apply_theme('light')`` restores the validated light reference. The dark steps were validated by the palette authors
against a #1a1a19 surface; against the navy surface they are NOT re-validated in this session (no execution allowed):
run ``node scripts/validate_palette.js "<hex,…>" --mode dark --surface #0b1626`` before publication and switch to the
light theme for print if a pair fails.

Rules kept: one axis per chart, text in ink colours (never series colours), recessive grid, status colours only with a
text label, every figure built from a polars table that is saved next to it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..visualization import plots as pv
from . import figures as F

LIGHT = {"SURFACE": "#fcfcfb", "INK": "#0b0b0b", "INK2": "#52514e", "MUTED": "#898781", "GRID": "#e1e0d9", "AXIS": "#c3c2b7", "REF": "#898781",
         "SERIES": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")}
NAVY = {"SURFACE": "#0b1626", "INK": "#eaf2fb", "INK2": "#b8c7d9", "MUTED": "#7f93ab", "GRID": "#1a2a42", "AXIS": "#2c4060", "REF": "#7f93ab",
        "SERIES": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")}
CYAN = "#22d3ee"
STATUS = {"good": "#0ca30c", "warning": "#eda100", "serious": "#eb6834", "critical": "#d03b3b", "neutral": "#7f93ab"}
_ACTIVE = {"name": "light", "accent": CYAN}
_SEQ_LIGHT = tuple(pv.SEQ_BLUE)


def _model_colors(S: Sequence[str], muted: str) -> dict[str, str]:
    m = {"M1": muted, "POS": "#5d6f86", "MEAN": "#45566c", "F0_engineered": S[0], "M1+F0": S[0], "F4_physics_contrastive": S[1], "M1+F4_physics_contrastive": S[1],
         "F2_cnn_physics": S[2], "M1+F2_cnn_physics": S[2], "F3_masked_ae": S[3], "M1+F3_masked_ae": S[3], "F5_jepa_physics": S[4], "M1+F5_jepa_physics": S[4],
         "F6_transformer": S[5], "M1+F6_transformer": S[5], "F1_TCN": S[6], "M1+F1_TCN": S[6], "random_frozen_F4_architecture": S[7]}
    for b in ("M1+BIO_all", "M1+BIO_reliable", "M1+BIO_trainsel", "M1+BIO_poscond"):      # one family → one hue (labels carry identity)
        m[b] = S[7]
    return m


def apply_theme(name: str = "navy") -> dict[str, Any]:
    t = NAVY if name == "navy" else LIGHT
    for k in ("SURFACE", "INK", "INK2", "MUTED", "GRID", "AXIS", "REF"):
        setattr(pv, k, t[k])
        setattr(F, k, t[k])
    pv.SERIES = t["SERIES"]
    # sequential ramp: light → dark on the light surface; dark → light on navy (low values recede into the surface and the
    # heatmap's ink / surface text switch stays legible)
    pv.SEQ_BLUE = tuple(reversed(_SEQ_LIGHT)) if name == "navy" else _SEQ_LIGHT
    F.S = t["SERIES"]
    F.DOMAIN_COLORS = {"combine": t["SERIES"][0], "nfl": t["SERIES"][1]}
    F.MODEL_COLORS = _model_colors(t["SERIES"], t["MUTED"])
    F.DIVERGING = ("#3987e5", "#383835", "#e66767") if name == "navy" else ("#2a78d6", "#f0efec", "#e34948")
    _ACTIVE["name"] = name
    try:
        import matplotlib as mpl

        mpl.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": t["SURFACE"], "figure.facecolor": t["SURFACE"], "text.color": t["INK"],
                             "axes.labelcolor": t["INK2"], "xtick.color": t["MUTED"], "ytick.color": t["MUTED"]})
    except ImportError:
        pass
    return {"theme": name, **{k: v for k, v in t.items() if k != "SERIES"}, "series": list(t["SERIES"]), "accent": CYAN}


def short(name: str) -> str:
    return F.short(str(name)).replace("M1+BIO_", "BIO·")


def _accent(fig: Any) -> None:
    """Thin cyan rule above the title: the M13 identity mark (decorative, carries no data)."""
    from matplotlib.lines import Line2D

    fig.add_artist(Line2D([0.01, 0.07], [0.995, 0.995], transform=fig.transFigure, color=CYAN, linewidth=2.5))


def finish(fig: Any, title: str, subtitle: str = "") -> Any:
    F._suptitle(fig, title, subtitle)
    _accent(fig)
    return fig


# ====================================================================== new figure types
def case_dumbbell(bw: Any, title: str, subtitle: str = "", label: str = "athlete") -> Any:
    """Per athlete: |error| of the reference (open) → |error| of the model (filled); improvements and deteriorations."""
    import polars as pl

    if bw is None or not bw.height:
        return F.empty(title, "no cases")
    d = bw.with_columns((pl.col("y") - pl.col("pred_ref")).abs().alias("e_ref"), (pl.col("y") - pl.col("pred_model")).abs().alias("e_mod"))
    fig, axes = F._fig(F.WIDE, ncols=2, sharex=True)
    for ax, case in zip(axes[0], ("largest improvement", "largest deterioration")):
        g = d.filter(pl.col("case") == case)
        F._style(ax, case, "absolute error (yd/s²)", "", grid="x")
        y = np.arange(g.height)[::-1]
        c = F.color(str(g["model"][0])) if g.height else F.S[0]
        for yi, a, b in zip(y, g["e_ref"].to_numpy(), g["e_mod"].to_numpy()):
            ax.plot([a, b], [yi, yi], color=F.MUTED, linewidth=1.5, zorder=1)
        ax.scatter(g["e_ref"].to_numpy(), y, s=46, facecolors=F.SURFACE, edgecolors=F.INK2, linewidths=1.5, zorder=2, label=short(str(g["reference"][0])) if g.height else None)
        ax.scatter(g["e_mod"].to_numpy(), y, s=46, color=c, edgecolors=F.SURFACE, linewidths=1.5, zorder=3, label=short(str(g["model"][0])) if g.height else None)
        ax.set_yticks(y, [f"{a} · {p}" for a, p in zip(g[label].to_list() if label in g.columns else g["nfl_id"].to_list(), g["position_group"].to_list())])
        F._legend(ax, "lower right")
    return finish(fig, title, subtitle)


def reliability_curves(curves: Any, projected: Any | None, title: str, subtitle: str = "", ncols: int = 3, max_facets: int = 9) -> Any:
    """Small multiples per feature: ICC(1,1) vs matched trial count k, both domains, 2.5–97.5 % band over repeats;
    dashed = Spearman–Brown projection ICC(1,k) of the full-sample Combine ICC."""
    import polars as pl

    if curves is None or not curves.height or "icc1_mean" not in curves.columns:
        return F.empty(title, "no evaluated curves (too few athletes with matched trial counts)")
    c = curves.filter(pl.col("status") == "evaluated")
    feats = c["feature"].unique(maintain_order=True).to_list()[:max_facets]
    nrows = int(np.ceil(len(feats) / ncols)) or 1
    fig, axes = F._fig((12.0, max(3.4, 2.7 * nrows + 1.0)), ncols=ncols, nrows=nrows, sharey=True)
    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= len(feats):
            ax.axis("off")
            continue
        f = feats[k]
        F._style(ax, short(f.replace("c_forty__", "forty·").replace("c_agility__", "agility·")), "matched trials k" if k // ncols == nrows - 1 else "",
                 "ICC(1,1)" if k % ncols == 0 else "")
        for dom in ("combine", "nfl"):
            g = c.filter((pl.col("feature") == f) & (pl.col("domain") == dom)).sort("k")
            if not g.height:
                continue
            x = g["k"].to_numpy()
            col = F.DOMAIN_COLORS[dom]
            ax.fill_between(x, g["icc1_q025"].to_numpy(), g["icc1_q975"].to_numpy(), color=col, alpha=0.18, linewidth=0)
            ax.plot(x, g["icc1_mean"].to_numpy(), color=col, linewidth=2, marker="o", markersize=4, markeredgecolor=F.SURFACE, label=dom)
        if projected is not None and projected.height:
            pj = projected.filter((pl.col("feature") == f) & (pl.col("domain") == "combine")).sort("k")
            if pj.height:
                ax.plot(pj["k"].to_numpy(), pj["icc_k_projected"].to_numpy(), color=F.REF, linestyle="--", linewidth=1, label="ICC(1,k) projected")
        ax.axhline(0, color=F.AXIS, linewidth=0.8)
        if k == 0:
            F._legend(ax, "lower right")
    return finish(fig, title, subtitle)


def evidence_matrix(df: Any, row: str, col: str, status: str, label: str, title: str, subtitle: str = "") -> Any:
    """Categorical evidence grid: cell colour = status (good / warning / critical / neutral) AND a text label in ink."""
    if df is None or not df.height:
        return F.empty(title, "no evidence rows")
    from matplotlib.patches import FancyBboxPatch

    rows = df[row].cast(str).unique(maintain_order=True).to_list()
    cols = df[col].cast(str).unique(maintain_order=True).to_list()
    fig, axes = F._fig((12.0, max(3.2, 0.5 * len(rows) + 1.8)))
    ax = axes[0, 0]
    F._style(ax, "", grid="")
    for r, c, s, t in zip(df[row].cast(str).to_list(), df[col].cast(str).to_list(), df[status].cast(str).to_list(), df[label].cast(str).to_list()):
        i, j = rows.index(r), cols.index(c)
        ax.add_patch(FancyBboxPatch((j - 0.46, i - 0.40), 0.92, 0.80, boxstyle="round,pad=0,rounding_size=0.06",
                                                                       facecolor=STATUS.get(s, STATUS["neutral"]), alpha=0.28, edgecolor=STATUS.get(s, STATUS["neutral"]), linewidth=1.2))
        ax.text(j, i, t, ha="center", va="center", fontsize=7.5, color=F.INK, wrap=True)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)), cols, rotation=0)
    ax.set_yticks(range(len(rows)), [short(r) for r in rows])
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_visible(False)
    return finish(fig, title, subtitle)


def dashboard(tiles: Sequence[tuple[str, str, str]], forest_df: Any, title: str, subtitle: str = "", outcome: str = "", outcome_status: str = "neutral") -> Any:
    """Executive dashboard (16:9): KPI tiles (label, value, note) on top, the primary Δ RMSE forest below, outcome banner."""
    import polars as pl
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    fig = Figure(figsize=F.WIDE, facecolor=F.SURFACE)
    n = max(1, len(tiles))
    w = 0.94 / n
    for i, (lab, val, note) in enumerate(tiles):
        x0 = 0.03 + i * w
        fig.add_artist(FancyBboxPatch((x0, 0.70), w - 0.012, 0.17, boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                      facecolor=F.GRID, edgecolor=F.AXIS, linewidth=0.8))
        fig.text(x0 + 0.012, 0.835, lab, color=F.INK2, fontsize=8.5)
        fig.text(x0 + 0.012, 0.755, val, color=F.INK, fontsize=17, fontweight="semibold")
        fig.text(x0 + 0.012, 0.715, note, color=F.MUTED, fontsize=7)
    ax = fig.add_axes([0.24, 0.10, 0.72, 0.52])
    F._style(ax, "Δ RMSE vs M1 on the strict common cohort (positive = better than M1; 95 % paired bootstrap)", "Δ RMSE (yd/s²)", "", grid="x")
    if forest_df is not None and forest_df.height:
        d = forest_df.filter(pl.col("delta_rmse").is_finite())
        y = np.arange(d.height)[::-1]
        for yi, lo, hi, m in zip(y, d["delta_lo"].to_numpy(), d["delta_hi"].to_numpy(), d["model"].to_list()):
            ax.plot([lo, hi], [yi, yi], color=F.color(m), linewidth=2.2, solid_capstyle="round")
        ax.scatter(d["delta_rmse"].to_numpy(), y, s=44, color=[F.color(m) for m in d["model"].to_list()], edgecolors=F.SURFACE, linewidths=1.5, zorder=3)
        ax.axvline(0, color=F.REF, linestyle="--", linewidth=1)
        ax.set_yticks(y, [short(m) for m in d["model"].to_list()])
    if outcome:
        fig.text(0.03, 0.035, f"■ {outcome}", color=F.INK, fontsize=9.5, fontweight="semibold")
        fig.add_artist(Line2D([0.03, 0.03], [0.02, 0.06], transform=fig.transFigure, color=STATUS.get(outcome_status, STATUS["neutral"]), linewidth=5))
    fig.suptitle(title, x=0.03, y=0.97, ha="left", color=F.INK, fontsize=14, fontweight="semibold")
    if subtitle:
        fig.text(0.03, 0.915, subtitle, color=F.INK2, fontsize=9)
    _accent(fig)
    return fig


UNITS = {"n_valid": "valid frames", "duration_s": "s", "peak_speed": "yd/s", "peak_accel": "yd/s²", "max_abs_a_n": "yd/s²"}


def covariate_ecdfs(long: Any, title: str, subtitle: str = "", ncols: int = 3, max_points: int = 20000, seed: int = 0) -> Any:
    """Small multiples (one per covariate, its own unit): ECDF of the sequence covariate per domain."""
    import polars as pl

    if long is None or not long.height:
        return F.empty(title, "no covariates")
    covs = long["covariate"].unique(maintain_order=True).to_list()
    nrows = int(np.ceil(len(covs) / ncols))
    fig, axes = F._fig((12.0, max(3.4, 2.8 * nrows + 1.0)), ncols=ncols, nrows=nrows, sharey=True)
    rng = np.random.default_rng(seed)
    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= len(covs):
            ax.axis("off")
            continue
        c = covs[k]
        F._style(ax, c, UNITS.get(c, ""), "cumulative share" if k % ncols == 0 else "")
        for dom in ("combine", "nfl"):
            v = long.filter((pl.col("covariate") == c) & (pl.col("domain") == dom))["value"].to_numpy()
            if v.size > max_points:
                v = rng.choice(v, max_points, replace=False)
            F.ecdf_ax(ax, v, dom, F.DOMAIN_COLORS[dom])
        if k == 0:
            F._legend(ax, "lower right")
    return finish(fig, title, subtitle)


def selection_heatmap(sel: Any, title: str, subtitle: str = "") -> Any:
    """Feature × (model, fold) selection indicator from the E3 selection log."""
    import polars as pl

    if sel is None or not sel.height:
        return F.empty(title, "no selection log")
    rows = []
    for r in sel.iter_rows(named=True):
        chosen = [s for s in (r.get("selected") or "").split(", ") if s]
        for c in [k.removeprefix("score:") for k in r if k.startswith("score:")] or chosen:
            rows.append({"feature": c.replace("c_forty__", "forty·").replace("c_agility__", "agility·"), "slot": f"{short(r['model'])} f{r['fold']}",
                         "selected": 1.0 if c in chosen else 0.0})
    d = pl.DataFrame(rows)
    return F.heatmap(d, "feature", "slot", "selected", title, fmt="{:.0f}", vmin=0.0, vmax=1.0, subtitle=subtitle)


# ====================================================================== notebook display helpers (HTML)
def banner_html(part: str, title: str, question: str) -> str:
    import html

    return (f"<div style='background:linear-gradient(135deg,#07101d 0%,#0b1626 65%,#0e2a3d 100%);padding:18px 22px;border-radius:12px;"
            f"border-left:5px solid {CYAN};margin:8px 0 6px'>"
            f"<div style='color:{CYAN};font:600 11px/1.4 system-ui,sans-serif;letter-spacing:.14em;text-transform:uppercase'>Part {html.escape(part)}</div>"
            f"<div style='color:#eaf2fb;font:700 21px/1.3 system-ui,sans-serif;margin:4px 0 6px'>{html.escape(title)}</div>"
            f"<div style='color:#b8c7d9;font:400 13.5px/1.55 system-ui,sans-serif'><b style='color:#eaf2fb'>Research question.</b> {html.escape(question)}</div></div>")


def callout_html(text: str, kind: str = "note") -> str:
    """Interpretation / caveat box; ``kind`` ∈ note · good · warning · critical (status colour on the rule, text in ink)."""
    import html

    col = {"note": CYAN, "good": STATUS["good"], "warning": STATUS["warning"], "critical": STATUS["critical"]}.get(kind, CYAN)
    lab = {"note": "Interpretation", "good": "Supported", "warning": "Caveat", "critical": "Blocked / invalid"}.get(kind, "Note")
    body = "<br>".join(html.escape(t) for t in str(text).split("\n"))
    return (f"<div style='background:#0b1626;border-left:4px solid {col};border-radius:8px;padding:10px 14px;margin:6px 0;color:#b8c7d9;"
            f"font:400 13px/1.55 system-ui,sans-serif'><b style='color:#eaf2fb'>{lab}.</b> {body}</div>")


def styled(df: Any, digits: int = 4, max_rows: int = 40, signed: Sequence[str] = (), status_col: str | None = None) -> Any:
    """polars → pandas Styler in the navy theme: tabular numbers, signed columns tinted (green > 0, red < 0, ink text),
    a status column tinted by keyword (pass / fail / missing / warn)."""
    if df is None or not getattr(df, "height", 0):
        return None
    p = df.head(max_rows).to_pandas()
    sty = p.style.format(precision=digits, na_rep="—")
    try:
        sty = sty.hide(axis="index")
    except Exception:                                               # noqa: BLE001 — older pandas
        pass
    sty = sty.set_table_styles([{"selector": "th", "props": [("background", "#0b1626"), ("color", "#eaf2fb"), ("font-weight", "600"), ("text-align", "left"),
                                                             ("border-bottom", "1px solid #2c4060"), ("padding", "5px 9px")]},
                                {"selector": "td", "props": [("background", "#0f1d31"), ("color", "#b8c7d9"), ("padding", "4px 9px"),
                                                             ("border-bottom", "1px solid #1a2a42"), ("font-variant-numeric", "tabular-nums")]},
                                {"selector": "", "props": [("border-collapse", "collapse"), ("font-size", "12.5px")]}])

    def tint(v: Any) -> str:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(x) or x == 0:
            return ""
        return "background-color: rgba(12,163,12,0.22); color: #eaf2fb" if x > 0 else "background-color: rgba(208,59,59,0.22); color: #eaf2fb"

    def stat(v: Any) -> str:
        s = str(v).lower()
        if any(k in s for k in ("fail", "missing", "invalid", "leaky", "differs", "blocked", "worse")):
            return "background-color: rgba(208,59,59,0.25); color: #eaf2fb"
        if any(k in s for k in ("warn", "inconclusive", "unverifiable", "exploratory", "not ")):
            return "background-color: rgba(237,161,0,0.22); color: #eaf2fb"
        if any(k in s for k in ("pass", "valid", "consistent", "adopted", "better", "improves", "ready", "present")):
            return "background-color: rgba(12,163,12,0.22); color: #eaf2fb"
        return ""

    apply = getattr(sty, "map", None) or sty.applymap                # Styler.map: pandas ≥ 2.1; applymap before
    cols = [c for c in signed if c in p.columns]
    if cols:
        sty = apply(tint, subset=cols)
        apply = getattr(sty, "map", None) or sty.applymap
    if status_col and status_col in p.columns:
        sty = apply(stat, subset=[status_col])
    return sty


__all__ = ["CYAN", "LIGHT", "NAVY", "STATUS", "UNITS", "apply_theme", "banner_html", "callout_html", "case_dumbbell", "covariate_ecdfs", "dashboard",
           "evidence_matrix", "finish", "reliability_curves", "selection_heatmap", "short", "styled"]
