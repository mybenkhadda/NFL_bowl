"""Figures of Milestone 08B (notebooks 08B.0–08B.6). Matplotlib only; every function returns a figure drawn from tables
the notebook computed. An estimate is always drawn with its interval and its sample size; an empty input gives an
explicit "no data" panel, never a blank axis. Names: ``AM08B<notebook>_<nn>_<slug>`` (given by the notebook).
"""

from __future__ import annotations

import textwrap
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS
from ..figkit import empty_panel, figure, finish

CLASS_COLOR = {"CONFIRMED": STATUS["good"], "DERIVED_VERIFIED": SEQUENTIAL_BLUE[4], "PROVISIONAL": STATUS["warning"], "BLOCKED": STATUS["critical"]}
STATUS_COLOR = {"repeatable_beyond_official": STATUS["good"], "weak_residual_signal": STATUS["warning"], "redundant_with_official": INK["axis"],
                "unstable": STATUS["critical"], "insufficient_support": INK["muted"]}
VERDICT_COLOR = {"distinguishable": STATUS["good"], "not_distinguishable": INK["axis"], "insufficient_evidence": STATUS["warning"], "not_observed": INK["muted"]}


# ============================================================================== primitives
def _arr(x: Any) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in (x.to_list() if isinstance(x, pl.Series) else list(x))], dtype=float)


def _color(i: int) -> str:
    return CATEGORICAL[i % len(CATEGORICAL)]


def forest(ax, labels: Sequence[str], est: Any, lo: Any, hi: Any, colors: Sequence[str] | str | None = None, zero: float | None = 0.0,
           xlabel: str = "", notes: Sequence[str] | None = None) -> None:
    """Horizontal point estimates with intervals, first row on top. Missing bounds are drawn without whiskers."""
    est, lo, hi = _arr(est), _arr(lo), _arr(hi)
    if not est.size or not np.isfinite(est).any():
        empty_panel(ax, "no estimate available")
        return
    y = np.arange(est.size)[::-1]
    cols = [colors] * est.size if isinstance(colors, str) or colors is None else list(colors)
    for i in range(est.size):
        if not np.isfinite(est[i]):
            continue
        c = cols[i] or _color(0)
        if np.isfinite(lo[i]) and np.isfinite(hi[i]):
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color=c, lw=2, solid_capstyle="round")
        ax.plot(est[i], y[i], "o", color=c, ms=6)
        if notes:
            ax.annotate(notes[i], (1.01, y[i]), xycoords=("axes fraction", "data"), fontsize=8, color=INK["muted"], va="center")
    if zero is not None:
        ax.axvline(zero, color=INK["axis"], lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.shorten(str(s), 46) for s in labels], fontsize=9)
    ax.set_ylim(-0.7, est.size - 0.3)
    ax.set_xlabel(xlabel)


def hist_groups(ax, groups: dict[str, np.ndarray], bins: int = 40, xlabel: str = "", clip: tuple[float, float] | None = None) -> None:
    data = {k: np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for k, v in groups.items()}
    data = {k: v for k, v in data.items() if v.size}
    if not data:
        empty_panel(ax, "no observation")
        return
    allv = np.concatenate(list(data.values()))
    lo, hi = clip or (float(np.quantile(allv, 0.001)), float(np.quantile(allv, 0.999)))
    edges = np.linspace(lo, hi, bins + 1)
    for i, (k, v) in enumerate(data.items()):
        ax.hist(np.clip(v, lo, hi), bins=edges, density=True, histtype="step", lw=1.8, color=_color(i), label=f"{k} (n = {v.size:,}; median {np.median(v):.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8, frameon=False)


def table_panel(ax, columns: Sequence[str], rows: Sequence[Sequence[Any]], widths: Sequence[float] | None = None, size: float = 8.0,
                wrap: int = 38) -> None:
    ax.axis("off")
    if not rows:
        empty_panel(ax, "nothing to show")
        return
    cells = [["\n".join(textwrap.wrap(str("" if c is None else c), wrap)) or "" for c in r] for r in rows]
    t = ax.table(cellText=cells, colLabels=list(columns), colWidths=widths, loc="upper left", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    lines = [max(c.count("\n") + 1 for c in r) for r in cells]
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor(INK["axis"])
        cell.set_linewidth(0.4)
        cell.set_height(0.055 * (1 if i == 0 else lines[i - 1]))
        if i == 0:
            cell.set_text_props(fontweight="semibold")


def _f(x: Any, d: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


# ============================================================================== 08B.0
def fig_definition_classes(definitions: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1, "wide")
    d = definitions
    forest(axes[0, 0], [f"{r['definition_id']} · {r['role_allowed']}" for r in d.iter_rows(named=True)], d["rel_sb"], d["rel_lo"], d["rel_hi"],
           [CLASS_COLOR[c] for c in d["source_class"]], None, "odd / even game reliability (Spearman–Brown) [95 %]",
           [f"{c} · linked {n}" for c, n in zip(d["source_class"], d["n_linked_all_models"])])
    axes[0, 0].axvline(0.70, color=INK["axis"], lw=1, ls=":")
    finish(fig, "Colour = source class (green confirmed · blue derived-verified · amber provisional · red blocked). The class comes from recorded source "
                "evidence and the 08A.2 audit; a decision never changes it. Dotted line: reliability reference 0.70 (heuristic).",
           "Football-outcome definitions: source class, admissible role, reliability")
    return fig


def fig_decision(definitions: pl.DataFrame, branch: dict[str, str], doc: dict[str, Any] | None) -> Any:
    fig, axes = figure(1, 1, "wide")
    rows = [[r["definition_id"], r["source_class"], r["role_allowed"] + (f" (= {r['redundant_with']})" if r["redundant_with"] else ""),
             r["interpretation_label"]] for r in definitions.iter_rows(named=True)]
    table_panel(axes[0, 0], ["definition", "source class", "role allowed", "may be called"], rows, [0.09, 0.13, 0.2, 0.58], 7.5, 92)
    sel = "no decision recorded" if doc is None else f"{doc['decision']} → {doc['approved_status']}: " + (", ".join(f"{t['target_id']} ({t['role']})" for t in doc["targets"]) or "no target")
    finish(fig, f"Registry decision: {sel}. Branch: {branch['label']}. Strongest admissible claim: {branch['claim']}.", "Outcome registry: what each definition may be called")
    return fig


# ============================================================================== 08B.1
def fig_distribution(rows: pl.DataFrame, by: str, title: str, xlabel: str = "separation at pass forward [yd]") -> Any:
    fig, axes = figure(1, 1, "wide")
    g = {str(k[0]): v["value"].to_numpy() for k, v in sorted(rows.drop_nulls(by).partition_by([by], as_dict=True).items())} if rows.height else {}
    hist_groups(axes[0, 0], g, xlabel=xlabel)
    finish(fig, "One value per recorded player-play; extremes are clipped to the 0.1–99.9 % range for display only. The recorded value is an interaction "
                "outcome (receiver, nearest defender, coverage, route, throw timing).", title)
    return fig


def fig_athlete_intervals(athletes: pl.DataFrame, definition_id: str) -> Any:
    a = athletes.filter((pl.col("denominator_id") == definition_id) & pl.col("supported")).sort("position_group", "value")
    pos = a["position_group"].unique().sort().to_list()
    fig, axes = figure(1, max(1, len(pos)), "wide")
    for j, p in enumerate(pos):
        t, ax = a.filter(pl.col("position_group") == p), axes[0, j]
        x, v, lo, hi = np.arange(t.height), _arr(t["value"]), _arr(t["lo"]), _arr(t["hi"])
        ax.vlines(x, lo, hi, color=SEQUENTIAL_BLUE[2], lw=1)
        ax.plot(x, v, ".", color=SEQUENTIAL_BLUE[5], ms=4)
        ax.set_title(f"{p} (n = {t.height})", fontsize=10)
        ax.set_xlabel("athletes, ordered by value")
        ax.set_ylabel("athlete value with game-bootstrap 95 % interval")
    if not pos:
        empty_panel(axes[0, 0], "no supported athlete")
    finish(fig, "Interval = the athlete's value when his GAMES are resampled whole: game-to-game variability of one athlete against the spread between athletes.",
           f"{definition_id}: game-to-game variability")
    return fig


def fig_cells(cells: pl.DataFrame, context_id: str, top: int = 18) -> Any:
    fig, axes = figure(1, 1, "wide")
    c = cells.filter(pl.col("context_id") == context_id).sort("n_rows", descending=True).head(top)
    se = [None if s is None or n < 2 else s / np.sqrt(n) for s, n in zip(c["sd_value"], c["n_rows"])]
    lo = [None if s is None else m - 1.96 * s for m, s in zip(c["mean_value"], se)]
    hi = [None if s is None else m + 1.96 * s for m, s in zip(c["mean_value"], se)]
    forest(axes[0, 0], c["cell"].to_list(), c["mean_value"], lo, hi, [STATUS["good"] if s else INK["muted"] for s in c["supported"]], None,
           "mean recorded value of the cell [95 % of the mean, rows as units]", [f"{n:,} rows · {a} athletes" for n, a in zip(c["n_rows"], c["n_athletes"])])
    finish(fig, "Grey cells do not meet the count rule and are not used for conditioning. Rows of one athlete are not independent: the intervals are descriptive. "
                "Conditioning on route or alignment changes the estimand.", f"Recorded separation by context cell ({context_id})")
    return fig


def fig_reliability_vs_obs(thresh: pl.DataFrame, definition_id: str, ref: float) -> Any:
    fig, axes = figure(1, 1, "wide")
    ax, t = axes[0, 0], thresh.filter(pl.col("definition_id") == definition_id)
    for i, (k, g) in enumerate(sorted(t.partition_by(["position_group"], as_dict=True).items())):
        g = g.sort("min_half_obs")
        x, y = _arr(g["min_half_obs"]), _arr(g["rel_sb"])
        ax.plot(x, y, "o-", color=_color(i), label=str(k[0]))
        ax.fill_between(x, _arr(g["rel_lo"]), _arr(g["rel_hi"]), color=_color(i), alpha=0.12)
        for xi, yi, n in zip(x, y, g["n_athletes"]):
            if np.isfinite(yi):
                ax.annotate(str(n), (xi, yi), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center", color=INK["muted"])
    ax.axhline(ref, color=INK["axis"], ls=":", lw=1)
    ax.set_xlabel("observations required in EACH game half")
    ax.set_ylabel("odd / even reliability (Spearman–Brown)")
    ax.legend(fontsize=8, frameon=False)
    finish(fig, "Numbers above the points = athletes kept. A stricter threshold keeps fewer, more exposed athletes: reliability and coverage trade off.",
           f"{definition_id}: reliability against opportunities")
    return fig


def fig_reliability_by_position(rel: pl.DataFrame, ids: Sequence[str], ref: float, floor: float) -> Any:
    fig, axes = figure(1, 1, "wide")
    r = rel.filter(pl.col("denominator_id").is_in(list(ids))).sort("denominator_id", "position_group", "split")
    forest(axes[0, 0], [f"{a} · {p} · {s}" for a, p, s in zip(r["denominator_id"], r["position_group"], r["split"])], r["rel_sb"], r["rel_lo"], r["rel_hi"],
           [_color(0) if s == "odd_even" else _color(1) for s in r["split"]], None, "game-split reliability [95 %]", [f"n = {n}" for n in r["n_athletes"]])
    axes[0, 0].axvline(ref, color=INK["axis"], ls=":", lw=1)
    axes[0, 0].axvline(floor, color=STATUS["critical"], ls=":", lw=1)
    finish(fig, "odd_even: chronological parity of the athlete's games, athlete-bootstrap interval. random: random game halves, interval over the splits. "
                "Halves are whole games. Position rows with few athletes are descriptive.", "Game-split reliability by position")
    return fig


def fig_context_ranking(values: pl.DataFrame, context: pl.DataFrame, definition_id: str, other: str = "R2") -> Any:
    fig, axes = figure(1, 2, "wide")
    v = values.filter(pl.col("definition_id") == definition_id)
    a = v.filter(pl.col("context_id") == "R0").select("nfl_id", "position_group", pl.col("pct_position").alias("x"))
    j = a.join(v.filter(pl.col("context_id") == other).select("nfl_id", pl.col("pct_position").alias("y")), on="nfl_id", how="inner")
    ax = axes[0, 0]
    if j.height:
        for i, (k, g) in enumerate(sorted(j.partition_by(["position_group"], as_dict=True).items())):
            ax.scatter(g["x"], g["y"], s=18, color=_color(i), label=f"{k[0]} (n = {g.height})", alpha=0.8)
        ax.plot([0, 100], [0, 100], color=INK["axis"], lw=1, ls="--")
        ax.legend(fontsize=8, frameon=False)
    else:
        empty_panel(ax, f"context {other} not available")
    ax.set_xlabel("position percentile, raw")
    ax.set_ylabel(f"position percentile, conditioned ({other})")
    c = context.filter(pl.col("definition_id") == definition_id)
    forest(axes[0, 1], [f"{r['context_id']} · {r['context']}" for r in c.iter_rows(named=True)], c["rel_sb"], c["rel_lo"], c["rel_hi"], None, None,
           "odd / even reliability of the conditioned value [95 %]", [f"ρ vs raw {_f(s)}" for s in c["spearman_vs_raw"]])
    finish(fig, "Conditioning parameters come from OTHER athletes (cross-fitted). A conditioned value answers a different question (route mix and NFL role removed) "
                "and is a sensitivity: it never replaces the approved definition because it looks better or worse.", f"{definition_id}: raw versus context-conditioned ranking")
    return fig


def fig_attribution(attr: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1, "panel")
    ax = axes[0, 0]
    if attr.is_empty():
        empty_panel(ax, "tackle fields absent")
    else:
        x = np.arange(attr.height)
        solo, asst = _arr(attr["n_solo"]) / _arr(attr["n_snaps"]), (_arr(attr["n_assist"]) - _arr(attr["n_both"])) / _arr(attr["n_snaps"])
        ax.bar(x, solo, color=_color(0), label="solo tackle credit")
        ax.bar(x, asst, bottom=solo, color=_color(1), label="assist credit only")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n{n:,} snaps" for p, n in zip(attr["position_group"], attr["n_snaps"])], fontsize=8)
        ax.set_ylabel("credited plays per eligible snap")
        ax.legend(fontsize=8, frameon=False)
    finish(fig, "A snap on the field is exposure, not a tackle opportunity; a credit is a scorer's attribution. FO12.T2 is a participation-credit rate, "
                "never a tackle success probability.", "Tackle and assist attribution")
    return fig


def fig_rate_vs_exposure(athletes: pl.DataFrame, definition_id: str, sack: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1, "panel")
    ax, a = axes[0, 0], athletes.filter((pl.col("denominator_id") == definition_id) & pl.col("supported"))
    if a.is_empty():
        empty_panel(ax, "no supported athlete")
    else:
        ax.scatter(a["n_obs"], a["value"], s=16, color=_color(0), alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("observations in the denominator (log scale)")
        ax.set_ylabel("athlete rate")
    r = sack.filter(pl.col("definition_id") == definition_id)
    rho = _f(r["rho_value_log_obs"][0]) if r.height else "—"
    finish(fig, f"Spearman ρ(rate, log exposure) within position = {rho}. A rate that follows playing time ranks usage; it is not a pass-rush win rate "
                "(that the player rushed the passer is not recorded).", f"{definition_id}: rate against its own exposure")
    return fig


def fig_funnel(funnel: pl.DataFrame, definition_id: str) -> Any:
    fig, axes = figure(1, 1, "wide")
    f = funnel.filter((pl.col("denominator_id") == definition_id) & pl.col("cumulative")).sort("position_group", "stage_order")
    ax, groups = axes[0, 0], sorted(f["position_group"].unique().to_list())
    stages = f.filter(pl.col("position_group") == "ALL").sort("stage_order")["stage"].to_list() if f.height else []
    w = 0.8 / max(1, len(groups))
    for i, g in enumerate(groups):
        t = f.filter(pl.col("position_group") == g).sort("stage_order")
        ax.bar(np.arange(t.height) + i * w, t["n_athletes"], w, color=_color(i), label=g)
        for x, n in zip(np.arange(t.height) + i * w, t["n_athletes"]):
            ax.annotate(str(n), (x, n), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(stages)) + 0.4 - w / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in stages], fontsize=8)
    ax.set_ylabel("athletes")
    ax.legend(fontsize=8, frameon=False)
    finish(fig, "Each stage is a subset of the previous one. The last stage is the cohort of the transfer benchmark; linkage FLAGS only were read to build it.",
           f"{definition_id}: NFL-to-Combine linked cohort funnel")
    return fig


# ============================================================================== 08B.2
def _metric(perf: pl.DataFrame, target_id: str, metric: str, analysis: str = "primary") -> pl.DataFrame:
    return perf.filter((pl.col("analysis") == analysis) & (pl.col("target_id") == target_id) & (pl.col("metric") == metric))


def fig_model_metric(perf: pl.DataFrame, target_id: str, metric: str, xlabel: str, title: str, order: Sequence[str]) -> Any:
    fig, axes = figure(1, 1, "wide")
    p = _metric(perf, target_id, metric)
    p = p.with_columns(pl.col("model").replace_strict({m: i for i, m in enumerate(order)}, default=99, return_dtype=pl.Int64).alias("_o")).sort("_o")
    ref = p.filter(pl.col("model") == "M1")
    forest(axes[0, 0], p["model"].to_list(), p["estimate"], p["ci_lower"], p["ci_upper"],
           [_color(1) if m == "M1" else INK["axis"] if m in ("MEAN", "POS", "M0") else _color(0) for m in p["model"]],
           float(ref["estimate"][0]) if ref.height and ref["estimate"][0] is not None else None, xlabel)
    n = p["n_athletes"][0] if p.height else 0
    finish(fig, f"Out-of-fold predictions of {n} athletes; athlete-bootstrap 95 % intervals; dashed line = official-testing model M1. Individual intervals overlap "
                "by construction: model DIFFERENCES are read from the paired figure, never from this one.", title)
    return fig


def fig_paired_forest(table: pl.DataFrame, title: str, label_cols: Sequence[str] = ("role", "model", "reference"), est: str = "delta_rmse",
                      caption: str = "") -> Any:
    fig, axes = figure(1, 1, "wide")
    labels = [" · ".join(str(r[c]) for c in label_cols) for r in table.iter_rows(named=True)]
    cols = [STATUS["good"] if e else STATUS["critical"] if w else INK["axis"] for e, w in zip(table["established"], table["worse"])] if table.height else None
    forest(axes[0, 0], labels, table[est] if table.height else [], table["ci_lower"] if table.height else [], table["ci_upper"] if table.height else [],
           cols, 0.0, "Δ RMSE = RMSE(reference) − RMSE(model)  [95 % paired athlete bootstrap] — positive = the model is better",
           [f"n = {n}" for n in table["n_athletes"]] if table.height else None)
    finish(fig, caption or "Green: interval above 0 · red: interval below 0 · grey: includes 0. Models are compared on IDENTICAL held-out athletes.", title)
    return fig


def fig_pred_vs_obs(oof: pl.DataFrame, target_id: str, models: Sequence[str], unit: str) -> Any:
    fig, axes = figure(1, len(models), "wide")
    o = oof.filter((pl.col("analysis") == "primary") & (pl.col("target_id") == target_id))
    for j, m in enumerate(models):
        ax, g = axes[0, j], o.filter(pl.col("model") == m)
        if g.is_empty():
            empty_panel(ax, f"{m} not evaluated")
            continue
        for i, (k, t) in enumerate(sorted(g.partition_by(["position_group"], as_dict=True).items())):
            ax.scatter(t["pred"], t["y"], s=18, color=_color(i), alpha=0.8, label=f"{k[0]} (n = {t.height})")
        lim = [float(min(g["pred"].min(), g["y"].min())), float(max(g["pred"].max(), g["y"].max()))]
        ax.plot(lim, lim, color=INK["axis"], lw=1, ls="--")
        ax.set_title(m, fontsize=10)
        ax.set_xlabel(f"out-of-fold prediction [{unit}]")
        ax.set_ylabel(f"observed athlete value [{unit}]")
        ax.legend(fontsize=8, frameon=False)
    finish(fig, "Each point is one athlete predicted by a model that never saw him. The spread around the diagonal includes the measurement error of the outcome itself.",
           f"{target_id}: predicted versus observed")
    return fig


def fig_calibration(calib: pl.DataFrame, target_id: str, models: Sequence[str]) -> Any:
    fig, axes = figure(1, 1, "panel")
    ax, c = axes[0, 0], calib.filter((pl.col("target_id") == target_id) & pl.col("model").is_in(list(models)))
    if c.is_empty():
        empty_panel(ax, "no calibration table")
    else:
        for i, m in enumerate(models):
            g = c.filter(pl.col("model") == m).sort("bin")
            if g.height:
                ax.errorbar(g["mean_pred"], g["mean_observed"], yerr=1.96 * _arr(g["se_observed"]), fmt="o-", color=_color(i), capsize=3, label=m)
        lim = [float(c["mean_pred"].min()), float(c["mean_pred"].max())]
        ax.plot(lim, lim, color=INK["axis"], lw=1, ls="--")
        ax.set_xlabel("mean out-of-fold prediction of the bin")
        ax.set_ylabel("mean observed value of the bin [± 1.96 SE]")
        ax.legend(fontsize=8, frameon=False)
    finish(fig, "Equal-count bins of the prediction (about twenty athletes each): a coarse check, read with its error bars.", f"{target_id}: calibration")
    return fig


def fig_subgroups(sub: pl.DataFrame, target_id: str, kind: str, pair: Sequence[str], title: str) -> Any:
    s = sub.filter((pl.col("target_id") == target_id) & pl.col("analysis").str.starts_with(kind + ":") & (pl.col("metric") == "delta_rmse")
                   & (pl.col("model") == pair[0]) & (pl.col("reference") == pair[1])).sort("analysis")
    return fig_paired_forest(s.rename({"estimate": "delta_rmse"}), title, ("analysis",),
                             caption=f"{pair[0]} versus {pair[1]}; pooled fit scored inside each subgroup. Exploratory: the subgroup with the largest estimate "
                                     "is not a finding, and small subgroups (about 24 TE) do not support an independent model.")


def fig_matched(matched: pl.DataFrame, counts: dict[str, int]) -> Any:
    fig, axes = figure(1, 1, "wide")
    forest(axes[0, 0], [f"{t} · {m}" for t, m in zip(matched["target_id"], matched["model"])] if matched.height else [],
           matched["concordance"] if matched.height else [], matched["ci_lower"] if matched.height else [], matched["ci_upper"] if matched.height else [],
           None, 0.5, "concordance of predicted and observed difference inside matched pairs [95 % athlete-cluster bootstrap]",
           [f"{n} pairs" for n in matched["n_pairs"]] if matched.height else None)
    finish(fig, f"{counts.get('pairs_in_cohort', 0)} of the {counts.get('pairs_frozen', 0)} matched stopwatch pairs frozen in 06C lie inside the football cohort "
                f"({counts.get('athletes_in_pairs', 0)} athletes). 0.5 = chance. Pairs were defined from pre-draft data only; none is selected by its outcome.",
           "Matched stopwatch prospects: do models order their football-outcome difference?")
    return fig


def fig_tcn_vs_physical(paired: pl.DataFrame, target_id: str, models: Sequence[str] = ("M2", "M3", "M4", "M5", "M6")) -> Any:
    fig, axes = figure(1, 2, "wide")
    for j, (metric, lab) in enumerate((("delta_rmse", "Δ RMSE versus M1 (positive = better)"), ("delta_spearman", "Δ Spearman versus M1 (positive = better)"))):
        p = paired.filter((pl.col("analysis") == "primary") & (pl.col("target_id") == target_id) & (pl.col("metric") == metric) & (pl.col("reference") == "M1")
                          & pl.col("model").is_in(list(models))).sort("model")
        forest(axes[0, j], p["model"].to_list(), p["estimate"], p["ci_lower"], p["ci_upper"], [_color(2) if m == "M5" else _color(4) if m == "M6" else _color(0) for m in p["model"]],
               0.0, lab)
    finish(fig, "M2 engineered speed development · M3 tracked speed (SPEED2) · M4 FPCA · M5 TCN-B · M6 untrained encoder (control). All add ONE block to the same "
                "official-testing model on the same athletes.", f"{target_id}: learned fingerprint versus physical features")
    return fig


def fig_null(draws: pl.DataFrame, perm: pl.DataFrame, title: str) -> Any:
    modes = perm["mode"].unique().sort().to_list() if perm.height else []
    fig, axes = figure(1, max(1, len(modes)), "wide")
    for j, mode in enumerate(modes):
        ax, r = axes[0, j], perm.filter(pl.col("mode") == mode).row(0, named=True)
        null = draws.filter((pl.col("signature") == r["signature"]) & (pl.col("iteration") >= 0))["delta_rmse"].to_numpy()
        ax.hist(null, bins=30, color=SEQUENTIAL_BLUE[2])
        ax.axvline(r["observed"], color=STATUS["critical"], lw=2, label=f"observed {r['observed']:+.4f}")
        ax.axvline(r["null_q95"], color=INK["axis"], ls=":", lw=1, label="95th percentile of the null")
        ax.set_title(f"{mode} permutation · p = {r['p_value']:.3f} · {r['n_perm']} refits" + ("" if r["complete"] else " · INCOMPLETE"), fontsize=10)
        ax.set_xlabel("Δ RMSE under the null")
        ax.legend(fontsize=8, frameon=False)
    if not modes:
        empty_panel(axes[0, 0], "no permutation computed")
    finish(fig, "Every permutation refits both models in every fold. linkage: tracking rows shuffled between athletes within position (does the athlete's OWN "
                "tracking add?). outcome: labels shuffled within position (pipeline-leak check).", title)
    return fig


# ============================================================================== 08B.3
def fig_power(scen: pl.DataFrame, target: float = 0.80) -> Any:
    fig, axes = figure(1, 2, "wide")
    ax, s = axes[0, 0], scen.filter((pl.col("study") == "power") & (pl.col("scope") == "ALL")) if scen.height else scen
    for i, (k, g) in enumerate(sorted(s.partition_by(["candidate", "mode"], as_dict=True).items())) if s.height else []:
        g = g.sort("effect")
        ax.plot(g["effect"], g["power"], "o-", color=_color(i), label=f"{k[0]} · {k[1]}")
        ax.fill_between(_arr(g["effect"]), _arr(g["power_lower"]), _arr(g["power_upper"]), color=_color(i), alpha=0.12)
    ax.axhline(target, color=INK["axis"], ls=":", lw=1)
    ax.set_xlabel("planted share of the residual outcome variance")
    ax.set_ylabel("detection rate (interval above 0)")
    ax.set_title("real football cohort", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax, s = axes[0, 1], scen.filter((pl.col("study") == "sample_size") & (pl.col("scope") == "ALL") & (pl.col("effect") > 0)) if scen.height else scen
    for i, (k, g) in enumerate(sorted(s.partition_by(["effect"], as_dict=True).items())) if s.height else []:
        g = g.sort("n_athletes")
        ax.plot(g["n_athletes"], g["power"], "o-", color=_color(i), label=f"planted share {k[0]:.2f}")
    ax.axhline(target, color=INK["axis"], ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("linked athletes (resampled cohort)")
    ax.set_ylabel("detection rate")
    ax.set_title("cohort-size scenarios", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    finish(fig, "Synthetic targets on the REAL athletes, positions, folds, collinearity and residual noise; the benchmark pipeline and its criterion unchanged. "
                "Effect 0 gives the false-positive rate. Bands: Wilson 95 % of the Monte-Carlo rate.", "Detectability of an incremental effect in the football cohort")
    return fig


def fig_reading(table: pl.DataFrame, columns: Sequence[str], title: str, caption: str, widths: Sequence[float] | None = None) -> Any:
    fig, axes = figure(1, 1, "wide")
    table_panel(axes[0, 0], list(columns), [[r[c] for c in columns] for r in table.iter_rows(named=True)], widths, 8.0, 70)
    finish(fig, caption, title)
    return fig


# ============================================================================== 08B.4
def fig_reliability_redundancy(info: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1, "square")
    ax = axes[0, 0]
    for r in info.iter_rows(named=True):
        if r["raw_value"] is None or r["r2_official"] is None:
            continue
        ax.scatter(r["r2_official"], r["raw_value"], s=70, color=STATUS_COLOR[r["status"]], marker="o" if r["kind"] == "scalar" else "s", zorder=3)
        ax.annotate(r["label"], (r["r2_official"], r["raw_value"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("share of athlete-mean variance explained by position + official tests (cross-fitted R²)")
    ax.set_ylabel("raw repeatability (ICC for scalars ●, retrieval AUC for embeddings ■)")
    finish(fig, "Top-left = repeatable and little explained by official testing. Colour = status after the adjustment (green repeatable beyond official · "
                "amber weak · grey redundant · red unstable). ICC and AUC share an axis for display only; they are different scales.",
           "Reliability versus stopwatch redundancy")
    return fig


def fig_residual_reliability(info: pl.DataFrame, kind: str | None = None) -> Any:
    fig, axes = figure(1, 1, "wide")
    d = info if kind is None else info.filter(pl.col("kind") == kind)
    labels, est, lo, hi, cols = [], [], [], [], []
    for r in d.iter_rows(named=True):
        labels += [f"{r['label']} · raw", f"{r['label']} · after official tests"]
        est += [r["raw_value"], r["residual_value"]]
        lo += [r["raw_lower"], r["residual_lower"]]
        hi += [r["raw_upper"], r["residual_upper"]]
        cols += [INK["axis"], STATUS_COLOR[r["status"]]]
    forest(axes[0, 0], labels, est, lo, hi, cols, 0.5 if kind == "embedding" else 0.0, "repeatability [95 % athlete bootstrap]")
    finish(fig, "Grey = raw. Coloured = after subtracting the cross-fitted expectation from position, forty, 10-yard split, weight and height. "
                + ("Dashed line: chance (0.5). No embedding dimension is given a physical meaning." if kind == "embedding" else "Dashed line: 0."),
           "What stays repeatable once official testing is accounted for" + (" — learned and functional representations" if kind == "embedding" else ""))
    return fig


def fig_curve_groups(groups: dict[str, dict[str, np.ndarray]], position: str, feature_label: str) -> Any:
    fig, axes = figure(1, 1, "wide")
    ax = axes[0, 0]
    for i, (name, b) in enumerate(groups.items()):
        ax.plot(b["grid"], b["q50"], color=_color(i), lw=2, label=f"{name} ({b['n']} runs)")
        ax.fill_between(b["grid"], b["q10"], b["q90"], color=_color(i), alpha=0.10)
    if not groups:
        empty_panel(ax, "no curve available")
    ax.set_xlabel("distance [yd]")
    ax.set_ylabel("tracked speed [yd/s]")
    ax.legend(fontsize=8, frameon=False)
    finish(fig, f"Median recorded speed-by-distance curve (10th–90th percentile band) of {position} athletes grouped by the stopwatch-ADJUSTED value of "
                f"{feature_label}: athletes whose official results predict the same value, with different measured runs.", "Speed-development curves at equal official testing")
    return fig


def fig_matched_dist(match: pl.DataFrame, labels: dict[str, str]) -> Any:
    fig, axes = figure(1, 1, "panel")
    ax, m = axes[0, 0], match.filter(pl.col("share_distinguishable").is_not_null())
    if m.is_empty():
        empty_panel(ax, "too few matched pairs")
    else:
        y = np.arange(m.height)[::-1]
        ax.barh(y, 100 * _arr(m["share_distinguishable"]), color=_color(0))
        ax.set_yticks(y)
        ax.set_yticklabels([labels.get(f, f) for f in m["feature_id"]], fontsize=9)
        for yi, s, n in zip(y, m["share_distinguishable"], m["n_pairs"]):
            ax.annotate(f"{100 * s:.0f} % of {n} pairs", (100 * s, yi), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8)
        ax.set_xlim(0, 110)
        ax.set_xlabel("matched stopwatch pairs whose difference exceeds 1.96 × SE of the difference [%]")
    finish(fig, "Pairs: same position, near-identical official forty, 10-yard split and body (06C rule, pre-draft). The test is on the DIFFERENCE and its standard "
                "error from repeat attempts. 5 % is expected if the pairs did not differ at all.", "Do matched stopwatch prospects differ measurably?")
    return fig


def fig_info_subsets(sens: pl.DataFrame, kinds: Sequence[str], title: str, caption: str) -> Any:
    fig, axes = figure(1, 1, "wide")
    s = sens.filter(pl.col("subset_kind").is_in(list(kinds))).sort("feature_id", "subset_kind", "subset")
    forest(axes[0, 0], [f"{f} · {k}" for f, k in zip(s["feature_id"], s["subset"])], s["residual_value"], s["residual_lower"], s["residual_upper"], None, None,
           "repeatability after official tests [95 %]", [f"{n} with a repeat" for n in s["n_repeat"]])
    finish(fig, caption, title)
    return fig


def fig_information_matrix(info: pl.DataFrame) -> Any:
    rows = [[r["label"], r["metric"].split(" within")[0], f"{_f(r['raw_value'])} [{_f(r['raw_lower'])}, {_f(r['raw_upper'])}]", _f(r["r2_official"]),
             f"{_f(r['residual_value'])} [{_f(r['residual_lower'])}, {_f(r['residual_upper'])}]", f"{r['n_repeat']} / {r['n_athletes']}", r["status"].replace("_", " ")]
            for r in info.iter_rows(named=True)]
    fig, axes = figure(1, 1, "wide")
    table_panel(axes[0, 0], ["characteristic", "metric", "raw [95 %]", "R² official", "after official tests [95 %]", "repeat / athletes", "status"], rows,
                [0.24, 0.13, 0.15, 0.08, 0.17, 0.09, 0.14], 8.0, 30)
    finish(fig, "Status follows an ordered rule on support, raw and adjusted repeatability (reference lines are heuristics). 'Repeatable beyond official' is new "
                "MEASURED information about the run; it carries no NFL claim.", "Additional-information matrix")
    return fig


# ============================================================================== 08B.5
def _bars(ax, rows: Sequence[dict[str, Any]], color: str) -> None:
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        if r["state"] in ("not_observed", "insufficient_evidence") or r["percentile"] is None:
            ax.text(2, yi, r["state"].replace("_", " "), va="center", fontsize=9, color=INK["muted"], style="italic")
            continue
        ax.barh(yi, r["percentile"], color=color, alpha=0.35 if r["state"] == "experimental" else 0.85, height=0.55)
        if r["pct_lower"] is not None and r["pct_upper"] is not None:
            ax.plot([r["pct_lower"], r["pct_upper"]], [yi, yi], color=INK["axis"], lw=2)
        ax.text(101, yi, f"{_f(r['value'])} {r['unit']}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.shorten(r["feature_label"], 34) for r in rows], fontsize=9)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.axvline(50, color=INK["axis"], lw=0.8, ls=":")


def _text(ax, title: str, lines: Sequence[str], width: int = 60) -> None:
    ax.axis("off")
    ax.text(0, 1, title, fontsize=11, fontweight="semibold", va="top", transform=ax.transAxes)
    body = "\n".join("\n".join(textwrap.wrap(str(s), width)) for s in lines)
    ax.text(0, 0.9, body, fontsize=8.5, va="top", transform=ax.transAxes, color=INK["muted"] if not lines else None)


def _cells(pct: pl.DataFrame, states: pl.DataFrame, nfl_id: int, section: str) -> list[dict[str, Any]]:
    st = {r["feature_id"]: r["state"] for r in states.filter(pl.col("nfl_id") == nfl_id).iter_rows(named=True)}
    return [{**r, "state": st.get(r["feature_id"], "not_observed")} for r in pct.filter((pl.col("nfl_id") == nfl_id) & (pl.col("section") == section)).iter_rows(named=True)]


def fig_scorecard(profile: dict[str, Any], pct: pl.DataFrame, states: pl.DataFrame, neighbors: Sequence[str], panel: dict[str, Any]) -> Any:
    """Four sections, never combined: speed capacity · speed development · movement fingerprint (similarity) · evidence confidence."""
    fig, axes = figure(2, 2, "wide")
    i = profile["nfl_id"]
    for ax, section, title, color in ((axes[0, 0], "speed_capacity", "A · Speed capacity", _color(0)), (axes[0, 1], "speed_development", "B · Speed development", _color(2))):
        rows = _cells(pct, states, i, section)
        if rows:
            _bars(ax, rows, color)
        else:
            empty_panel(ax, "not observed")
        ax.set_title(title + " — position percentile with 95 % interval", fontsize=10, loc="left")
    _text(axes[1, 0], "C · Movement fingerprint (similarity only)",
          (["nearest athletes of the position in the TCN-B representation:"] + [f"  {k + 1}. {n}" for k, n in enumerate(neighbors)]
           + [f"repeat-attempt self-recognition: {_f(profile.get('fingerprint_self_auc'))} (1 = his repeat is closer than everybody; 0.5 = not recognisable)"])
          if profile.get("fingerprint_available") else ["not observed: no inductive fingerprint for this athlete"])
    _text(axes[1, 1], f"D · Evidence confidence: {profile.get('evidence_tier', 'n/a')}",
          [f"reasons: {profile.get('evidence_reasons') or 'none recorded'}", f"reference: {profile.get('reference_n')} athletes of the position", "",
           panel.get("title", ""), panel.get("text", "")])
    finish(fig, "A descriptive pre-draft profile. No overall score: no section has an established NFL predictive value, so the sections are never combined. "
                "A missing measurement is shown as not observed, never imputed. Faded bar = experimental (reliability below the 'available' line).",
           f"{profile.get('display_name') or i} · {profile.get('position_group')} · draft {profile.get('draft_year')}")
    return fig


def fig_two_players(case: dict[str, Any], pairwise: pl.DataFrame) -> Any:
    fig, axes = figure(1, 2, "wide")
    p = pairwise.filter((pl.col("case") == case["case"]) & (pl.col("nfl_id_a") == case["nfl_id_a"]) & (pl.col("nfl_id_b") == case["nfl_id_b"]))
    rows = [[r["feature_label"], f"{_f(r['value_a'], 3)} {r['unit']}", f"{_f(r['value_b'], 3)} {r['unit']}", f"{_f(r['difference'], 3)} ± {_f(r['se_difference'], 3)}",
             r["verdict"].replace("_", " ")] for r in p.iter_rows(named=True)]
    table_panel(axes[0, 0], ["measurement", case["name_a"], case["name_b"], "difference ± SE", "verdict"], rows, [0.32, 0.17, 0.17, 0.18, 0.16], 8.0, 24)
    z = p.filter(pl.col("z").is_not_null())
    forest(axes[0, 1], z["feature_label"].to_list(), z["z"], _arr(z["z"]) - 1.96, _arr(z["z"]) + 1.96, [VERDICT_COLOR[v] for v in z["verdict"]], 0.0,
           "difference / its standard error (b − a) [95 %]")
    finish(fig, f"Matched on: {case['matched_on']}. Contrast: {case['contrast']}. Official results carry no repeat-attempt standard error: their difference is shown "
                "without a verdict. Selected by a pre-draft rule, not by an NFL label; an illustration, not evidence.",
           f"{case['criterion']} — {case['name_a']} versus {case['name_b']} ({case['position_group']})")
    return fig


def fig_curve_overlay(curves: dict[str, list[tuple[np.ndarray, np.ndarray]]], band: dict[str, np.ndarray] | None, title: str) -> Any:
    fig, axes = figure(1, 1, "wide")
    ax = axes[0, 0]
    if band is not None and band.get("n"):
        ax.fill_between(band["grid"], band["q10"], band["q90"], color=INK["axis"], alpha=0.18, label=f"position 10th–90th percentile ({band['n']} runs)")
    for i, (name, runs) in enumerate(curves.items()):
        for k, (d, s) in enumerate(runs):
            ax.plot(d, s, color=_color(i), lw=1.8, alpha=0.9, ls="-" if k == 0 else "--", label=f"{name} · attempt {k + 1}")
    if not any(curves.values()):
        empty_panel(ax, "no recorded curve")
    ax.set_xlabel("distance [yd]")
    ax.set_ylabel("tracked speed [yd/s]")
    ax.legend(fontsize=8, frameon=False)
    finish(fig, "Curves as recorded (no extrapolation outside the recorded range). Two attempts of one athlete show how repeatable his curve is.", title)
    return fig


def fig_pairwise(pairwise: pl.DataFrame, feature_id: str) -> Any:
    fig, axes = figure(1, 1, "wide")
    p = pairwise.filter((pl.col("feature_id") == feature_id) & pl.col("difference").is_not_null()).sort("case")
    forest(axes[0, 0], [f"{c} · {a} vs {b}" for c, a, b in zip(p["case"], p["nfl_id_a"], p["nfl_id_b"])], p["difference"], p["ci_lower"], p["ci_upper"],
           [VERDICT_COLOR[v] for v in p["verdict"]], 0.0, f"measured difference (b − a) [{p['unit'][0] if p.height else ''}] with 95 % interval of the DIFFERENCE")
    finish(fig, "Green: the interval of the difference excludes 0 (the two athletes are measurably different). Grey: not distinguishable. Amber: measurement error "
                "not estimable. Two overlapping individual intervals would NOT be a valid test.", f"Pairwise measurement uncertainty — {feature_id}")
    return fig


def fig_confidence(confidence: pl.DataFrame, states: pl.DataFrame) -> Any:
    fig, axes = figure(1, 2, "wide")
    for ax, df, col, order, title in ((axes[0, 0], confidence, "evidence_tier", ("high", "medium", "low", "official_only"), "evidence tier by position"),
                                      (axes[0, 1], states, "state", ("supported", "experimental", "insufficient_evidence", "not_observed"), "scorecard cells by state")):
        t = df.group_by("position_group", col).len()
        pos = sorted(t["position_group"].unique().to_list())
        bottom = np.zeros(len(pos))
        for i, level in enumerate(order):
            n = np.array([int(t.filter((pl.col("position_group") == p) & (pl.col(col) == level))["len"].sum()) for p in pos], float)
            ax.bar(pos, n, bottom=bottom, color=_color(i), label=level.replace("_", " "))
            bottom += n
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    finish(fig, "Confidence describes the MEASUREMENT (attempts, signal, window, repeat), never the athlete's ability. Cells that abstain are counted, not hidden.",
           "Evidence confidence and abstention")
    return fig


def fig_reference_sensitivity(ref: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1, "wide")
    r = ref.sort("feature_id", "reference_kind", "position_group", "alternative")
    forest(axes[0, 0], [f"{f.split('.')[-1]} · {p} · {a}" for f, p, a in zip(r["feature_id"], r["position_group"], r["alternative"])], r["mean_abs_shift"],
           [0.0] * r.height, r["max_abs_shift"], [_color(0) if k == "position" else _color(1) for k in r["reference_kind"]], None,
           "percentile shift when the reference changes: mean (point) and largest (whisker)", [f"n = {n}" for n in r["n_scored"]])
    finish(fig, "A percentile is a statement about a comparison group. Blue: own position versus all positions pooled. Orange: all cohorts versus one draft cohort.",
           "Reference-population sensitivity")
    return fig


def fig_attempt_stability(shifts: pl.DataFrame, labels: dict[str, str]) -> Any:
    fids = shifts["feature_id"].unique().sort().to_list() if shifts.height else []
    fig, axes = figure(1, max(1, len(fids)), "wide")
    for j, f in enumerate(fids):
        s = shifts.filter(pl.col("feature_id") == f)
        ax = axes[0, j]
        ax.scatter(s["pct_first"], s["pct_second"], s=14, color=_color(j), alpha=0.7)
        ax.plot([0, 100], [0, 100], color=INK["axis"], lw=1, ls="--")
        ax.set_title(f"{labels.get(f, f)} · n = {s.height} · median |shift| {float(s['abs_shift'].median()):.0f}", fontsize=10)
        ax.set_xlabel("position percentile of attempt 1")
        ax.set_ylabel("position percentile of attempt 2")
    if not fids:
        empty_panel(axes[0, 0], "no athlete with a repeat")
    finish(fig, "Each point is one athlete with two valid attempts, both scored against the same reference. The distance from the diagonal is what a scout would "
                "not know from a single run.", "Repeat-attempt stability of a percentile")
    return fig


# ============================================================================== 08B.6
def fig_hierarchy(evidence: pl.DataFrame, levels: Sequence[tuple[str, str]]) -> Any:
    fig, axes = figure(1, 1, "wide")
    rows = []
    for level, _ in levels:
        e = evidence.filter(pl.col("evidence_level") == level)
        done, wait = e.filter(pl.col("executed")), e.filter(~pl.col("executed"))
        rows.append([level.replace("_", " "), "; ".join(f"{r['claim_id']} {r['claim']}" for r in done.iter_rows(named=True)) or "—",
                     ", ".join(wait["claim_id"].to_list()) or "—"])
    table_panel(axes[0, 0], ["evidence level", "claims read from an executed artifact", "pending"], rows, [0.2, 0.68, 0.12], 7.5, 110)
    finish(fig, "Six levels, never collapsed into one success label. A pending claim has no executed artifact and is excluded from the executive narrative.",
           "Evidence hierarchy of the research programme")
    return fig


def fig_intelligence_diagram(boxes: Sequence[tuple[str, str, str]]) -> Any:
    """``boxes``: (title, text, status colour key: good · warning · critical · neutral) — the final flow, left to right."""
    from matplotlib.patches import FancyBboxPatch

    fig, axes = figure(1, 1, "wide")
    ax = axes[0, 0]
    ax.axis("off")
    n = len(boxes)
    w = 0.92 / max(n, 1)
    for i, (title, text, key) in enumerate(boxes):
        x = 0.04 + i * w
        col = STATUS.get(key, INK["axis"])
        ax.add_patch(FancyBboxPatch((x, 0.2), w * 0.86, 0.6, boxstyle="round,pad=0.01", transform=ax.transAxes, facecolor="white",
                                    edgecolor=col, lw=2))
        ax.text(x + 0.01, 0.76, "\n".join(textwrap.wrap(title, 20)), transform=ax.transAxes, fontsize=9.5, fontweight="semibold", va="top")
        ax.text(x + 0.01, 0.62, "\n".join(textwrap.wrap(text, 24)), transform=ax.transAxes, fontsize=8, va="top", color=INK["muted"])
        if i < n - 1:
            ax.annotate("", xy=(x + w, 0.5), xytext=(x + w * 0.88, 0.5), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "color": INK["axis"]})
    finish(fig, "Green: supported by an executed artifact. Amber: exploratory or provisional. Red: tested and not established, or unresolved. Grey: pending.",
           "Scouting intelligence: from 10 Hz Combine tracking to what a scout may use")
    return fig


__all__ = [n for n in dir() if n.startswith("fig_")] + ["forest", "hist_groups", "table_panel"]
