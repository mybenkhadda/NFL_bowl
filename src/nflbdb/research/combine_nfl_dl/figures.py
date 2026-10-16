"""Figures of the Combine → NFL get-off experiment — drawn only from persisted tables.

Eight figures, one question each. Every caption states the unit of analysis. While the vendor definition of the
endpoint is unresolved, every figure that relates the Combine to the endpoint says so in its caption, and no axis
ever implies a direction ("faster", "better").
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, STATUS
from ..figkit import ci_errorbar, empty_panel, figure, finish
from ..sprint_rescue import redundancy as rescue_rd
from ..sprint_rescue import reliability as rescue_rl
from ..sprint_rescue.stats import partial_corr
from ..sprint_rescue.store import RescueStore
from . import cohort as co
from . import endpoint as ep
from . import evaluate as ev
from . import targets as tg
from .config import COHORT, TRACKING, TRACKING_MODELS, GetOffConfig, model_specs

Tables = dict[str, pl.DataFrame]
GCFG = GetOffConfig()
Y_LABEL = "player_get_off (s, vendor value)"
MODEL_COLORS = {"MEAN": INK["muted"], "M0": INK["secondary"], "M1": CATEGORICAL[0], "M2a": CATEGORICAL[1],
                "M2b": CATEGORICAL[2], "M2c": CATEGORICAL[3], "M2bc": CATEGORICAL[6]}


def _num(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].cast(pl.Float64).fill_null(float("nan")).to_numpy()


def _gate_note(store: RescueStore) -> str:
    ok, _ = ep.definition_status(ep.load_definition(store.cfg))
    return "" if ok else " VENDOR DEFINITION UNRESOLVED: the endpoint is an opaque number; no direction or football meaning is implied."


def _pick(df: pl.DataFrame, **eq) -> pl.DataFrame:
    for k, v in eq.items():
        df = df.filter(pl.col(k) == v)
    return df


# ------------------------------------------------------------------------------ 1 funnel
def fig_funnel(store: RescueStore, t: Tables) -> None:
    f = t[co.FUNNEL.name]
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    stages = f.filter(pl.col("cohort") == "ALL").sort("stage")
    if not stages.height:
        empty_panel(a, "no funnel rows")
    else:
        names = stages["stage"].to_list()
        y = np.arange(len(names))[::-1]
        left = np.zeros(len(names))
        for k, c in enumerate(sorted(x for x in f["cohort"].unique().to_list() if x != "ALL")):
            n = dict(_pick(f, cohort=c).select("stage", "n_athletes").iter_rows())
            w = np.array([n.get(s, 0) for s in names], float)
            a.barh(y, w, left=left, color=CATEGORICAL[k % 8], label=c)
            left += w
        for yi, tot in zip(y, stages["n_athletes"].to_list()):
            a.text(tot, yi, f"  {tot}", va="center", fontsize=9, color=INK["primary"])
        a.set_yticks(y, [f"{s[3:].replace('_', ' ')}" for s in names], fontsize=9)
        a.set_xlabel("athletes")
        a.legend(title="draft cohort", frameon=False)
    finish(fig, "Unit: athlete. Each bar applies its condition on top of the previous ones where the definition says "
                "'cumulative'; definitions are in cohort_funnel.csv. Combine eligibility never uses an NFL outcome.",
           "How many defensive linemen reach the model?")
    store.save_figure(fig, "CN_01_cohort_funnel", {"funnel": f}, source=co.FUNNEL.name, n=f"{stages.height} stages")


# ------------------------------------------------------------------------------ 2 endpoint distribution
def fig_distribution(store: RescueStore, t: Tables) -> None:
    h, audit = t[ep.HIST.name], t[ep.AUDIT.name]
    fig, ax = figure(1, 2)
    a = ax[0, 0]
    if not h.height:
        empty_panel(a, "no valid observations")
    else:
        for k, name in enumerate(sorted(h["alignment"].unique().to_list())):
            g = _pick(h, alignment=name).sort("bin_lower")
            share = _num(g, "n") / max(float(g["n"].sum()), 1.0)
            a.step(_num(g, "bin_lower"), share, where="post", color=CATEGORICAL[k], label=f"{name} (n = {int(g['n'].sum()):,})")
        a.set_xlabel(Y_LABEL)
        a.set_ylabel("share of the alignment's observations per 0.05 s")
        a.legend(frameon=False)
    b = ax[0, 1]
    av = audit.filter((pl.col("section") == "availability") & (pl.col("metric") == "share_non_null")
                      & pl.col("stratum").str.contains(":") & (pl.col("n") >= 200)).sort("stratum")
    if not av.height:
        empty_panel(b, "no availability rows")
    else:
        y = np.arange(av.height)[::-1]
        b.barh(y, _num(av, "value"), color=CATEGORICAL[0])
        b.set_yticks(y, [f"{s}  (n = {n:,})" for s, n in zip(av["stratum"].to_list(), av["n"].to_list())], fontsize=8)
        b.set_xlim(0, 1)
        b.set_xlabel("share of tracked DL player-plays with a get-off value")
    finish(fig, "Unit: regular-season player-play of a defensive-line athlete. Left: valid observations (EDGE or "
                "INTERIOR_LINE, play not nullified). Right: where the vendor delivers a value at all; strata with fewer "
                "than 200 player-plays are omitted. Descriptive — no Combine value is used.",
           "What does the get-off value look like, and when does it exist?")
    store.save_figure(fig, "CN_02_getoff_distribution", {"histogram": h, "availability": av}, source=ep.AUDIT.name,
                      n=f"{int(h['n'].sum()) if h.height else 0:,} observations")


# ------------------------------------------------------------------------------ 3 outcome reliability
def fig_reliability(store: RescueStore, t: Tables) -> None:
    r = t[tg.RELIABILITY.name]
    fig, ax = figure(1, 2)
    a = ax[0, 0]
    lo_thr = min(GCFG.min_obs_grid)
    drawn = False
    for k, target in enumerate(("adj_mean", "mean")):
        sb = r.filter((pl.col("target") == target) & (pl.col("window") == "all") & (pl.col("min_obs") == lo_thr)
                      & pl.col("method").str.starts_with("sb_projected:") & pl.col("estimate").is_not_null())
        if sb.height:
            n = np.array([int(m.split(":")[1]) for m in sb["method"].to_list()])
            o = np.argsort(n)
            a.plot(n[o], _num(sb, "estimate")[o], color=CATEGORICAL[k], label=f"{target}: projected from single plays")
            a.fill_between(n[o], _num(sb, "ci_lower")[o], _num(sb, "ci_upper")[o], color=CATEGORICAL[k], alpha=0.15)
            drawn = True
        emp = r.filter((pl.col("target") == target) & pl.col("window").str.starts_with("first_")
                       & (pl.col("method") == "game_random_halves_sb") & pl.col("estimate").is_not_null())
        if emp.height:
            ci_errorbar(a, _num(emp, "min_obs"), _num(emp, "estimate"), _num(emp, "ci_lower"), _num(emp, "ci_upper"),
                        color=CATEGORICAL[k], fmt="s", label=f"{target}: measured on game halves (first n plays)")
            drawn = True
    if not drawn:
        empty_panel(a, "reliability not estimable")
    else:
        a.set_xscale("log")
        a.set_xlabel("valid observations per athlete (log scale)")
        a.set_ylabel("reliability of the athlete value")
        a.set_ylim(0, 1)
        a.legend(frameon=False, fontsize=8)
    b = ax[0, 1]
    thr = r.filter((pl.col("window") == "all") & (pl.col("method") == "game_random_halves_sb") & pl.col("estimate").is_not_null())
    if not thr.height:
        empty_panel(b, "reliability not estimable")
    else:
        for k, target in enumerate(sorted(thr["target"].unique().to_list())):
            g = _pick(thr, target=target).sort("min_obs")
            b.plot(_num(g, "min_obs"), _num(g, "estimate"), marker="o", color=CATEGORICAL[k % 8], label=target)
        g = _pick(thr, target=GCFG.primary_target).sort("min_obs")
        for x, y, n in zip(_num(g, "min_obs"), _num(g, "estimate"), g["n_athletes"].to_list()):
            b.annotate(f"n = {n}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK["muted"])
        b.set_xscale("log")
        b.set_xlabel("eligibility threshold: minimum observations (log scale)")
        b.set_ylabel("reliability across random halves of an athlete's games")
        b.set_ylim(0, 1)
        b.legend(title="target", frameon=False, fontsize=8)
    finish(fig, "Unit: athlete. Left: line = Spearman–Brown projection from the single-play ICC (assumes exchangeable "
                "plays; band = athlete bootstrap); squares = split of each athlete's GAMES into halves using only his "
                "first n observations (bars = spread over random splits). Right: every observation, athletes above the "
                "threshold; n = athletes for the primary target. Reliability bounds the variance a predictor can explain.",
           "How many plays make a get-off value an athlete trait?")
    store.save_figure(fig, "CN_03_getoff_reliability", {"reliability": r}, source=tg.RELIABILITY.name, n=f"{r.height} estimates")


# ------------------------------------------------------------------------------ 4 Combine reliability vs redundancy
def fig_combine_screen(store: RescueStore, t: Tables) -> None:
    rescue = RescueStore(store.cfg)
    scope = f"position:{GCFG.position}"
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    rows = pl.DataFrame()
    if rescue.exists(rescue_rl.RELIABILITY) and rescue.exists(rescue_rd.REDUNDANCY):
        icc = _pick(rescue.read(rescue_rl.RELIABILITY), scenario="S0_all", scope=scope).select(
            "feature", "icc1", "ci_lower", "ci_upper", "n_athletes")
        red = _pick(rescue.read(rescue_rd.REDUNDANCY), aggregation="mean", scope=scope, official=rescue_rd.FORTY).select(
            "feature", pl.col("spearman").abs().alias("abs_rho_forty"))
        rows = icc.join(red, on="feature", how="inner").filter(pl.col("icc1").is_not_null() & pl.col("abs_rho_forty").is_not_null())
    if not rows.height:
        empty_panel(a, "sprint rescue tables (within_position_reliability, combine_timing_redundancy) not found")
    else:
        cand = rows.filter(pl.col("feature").is_in(list(TRACKING)))
        rest = rows.filter(~pl.col("feature").is_in(list(TRACKING)))
        a.scatter(_num(rest, "abs_rho_forty"), _num(rest, "icc1"), s=30, color=INK["axis"], label="other tracked features")
        for k, r in enumerate(cand.iter_rows(named=True)):
            lo = r["ci_lower"] if r["ci_lower"] is not None else np.nan
            hi = r["ci_upper"] if r["ci_upper"] is not None else np.nan
            ci_errorbar(a, [r["abs_rho_forty"]], [r["icc1"]], [lo], [hi], color=CATEGORICAL[k], label=r["feature"], markersize=9)
            a.annotate(r["feature"], (r["abs_rho_forty"], r["icc1"]), textcoords="offset points", xytext=(8, 6), fontsize=10)
        a.axhline(0.75, color=INK["axis"], linewidth=0.8, linestyle="--")
        a.axvline(0.70, color=INK["axis"], linewidth=0.8, linestyle="--")
        a.set_xlim(0, 1)
        a.set_ylim(min(0.0, float(np.nanmin(_num(rows, "icc1"))) - 0.05), 1)
        a.set_xlabel("|Spearman| with the official forty, defensive line (redundancy)")
        a.set_ylabel("ICC(1) between an athlete's attempts, defensive line (repeatability)")
        a.legend(frameon=False, fontsize=9, loc="lower left")
    finish(fig, "Unit: defensive-line athlete (sprint reliability rescue, executed). Upper left = repeatable and not "
                "explained by the stopwatch. Bars: 95 % athlete bootstrap of the ICC. Dashed lines are heuristic "
                "references, not thresholds. Position in this plot says nothing about NFL relevance.",
           "Which Combine tracking features are worth testing on the defensive line?")
    store.save_figure(fig, "CN_04_combine_reliability_vs_redundancy", {"features": rows} if rows.height else {},
                      source="sprint_rescue within_position_reliability + combine_timing_redundancy", n=f"{rows.height} features")


# ------------------------------------------------------------------------------ 5 official forty vs get-off
def fig_forty(store: RescueStore, t: Tables) -> None:
    base = t[co.BASE.name].filter((pl.col("n_obs") >= GCFG.primary_min_obs) & pl.col("cmb_forty").is_not_null()
                                  & pl.col(f"nfl_{GCFG.primary_target}").is_not_null())
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    x, y = _num(base, "cmb_forty"), _num(base, f"nfl_{GCFG.primary_target}")
    if x.size < 10:
        empty_panel(a, "fewer than 10 eligible athletes")
    else:
        for k, name in enumerate(sorted(base["alignment_group"].drop_nulls().unique().to_list())):
            g = base.filter(pl.col("alignment_group") == name)
            a.scatter(_num(g, "cmb_forty"), _num(g, f"nfl_{GCFG.primary_target}"), s=28, alpha=0.8, color=CATEGORICAL[k], label=name)
        grid = np.linspace(x.min(), x.max(), 50)
        rng = np.random.default_rng(GCFG.seed)
        lines = np.array([np.polyval(np.polyfit(x[i], y[i], 1), grid) for i in rng.integers(0, x.size, (1000, x.size))])
        a.plot(grid, np.polyval(np.polyfit(x, y, 1), grid), color=INK["primary"], linewidth=1.5)
        a.fill_between(grid, np.quantile(lines, 0.025, axis=0), np.quantile(lines, 0.975, axis=0), color=INK["muted"], alpha=0.2)
        r = partial_corr(y, x, categorical=[base[COHORT].cast(pl.String).fill_null("?").to_numpy()])
        if np.isfinite(r["r"]):
            a.set_title(f"cohort-adjusted Spearman = {r['r']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}], n = {r['n']}", fontsize=11)
        a.set_xlabel("official forty-yard dash (s)")
        a.set_ylabel(f"context-adjusted mean {Y_LABEL}")
        a.legend(title="modal NFL alignment", frameon=False)
    finish(fig, f"Unit: athlete with ≥ {GCFG.primary_min_obs} valid observations. Line = least squares, band = 95 % athlete "
                "bootstrap of the line. In-sample description (context means from all athletes); the held-out comparison "
                "is in CN_06 and CN_07." + _gate_note(store),
           "Does the stopwatch relate to NFL get-off at all?")
    store.save_figure(fig, "CN_05_forty_vs_getoff", {"athletes": base.select("nfl_id", COHORT, "alignment_group", "cmb_forty",
                                                                          f"nfl_{GCFG.primary_target}", "n_obs")},
                      publication=True, source=co.BASE.name, n=f"{x.size} athletes")


# ------------------------------------------------------------------------------ 6 model performance
def fig_models(store: RescueStore, t: Tables) -> None:
    c = t[ev.COMPARISON.name]
    fig, ax = figure(1, 2)
    order = [m for m in model_specs() if m in c["model"].unique().to_list()] if c.height else []
    for j, (metric, label) in enumerate((("rmse", "out-of-fold RMSE (s)"), ("r2_oos", "out-of-fold R² (skill vs training mean)"))):
        a = ax[0, j]
        g = {r["model"]: r for r in _pick(c, metric=metric).iter_rows(named=True)}
        if not g:
            empty_panel(a, "model comparison not built")
            continue
        for i, m in enumerate(order):
            r = g.get(m)
            if r is None or r["estimate"] is None:
                continue
            lo = r["ci_lower"] if r["ci_lower"] is not None else np.nan
            hi = r["ci_upper"] if r["ci_upper"] is not None else np.nan
            ci_errorbar(a, [i], [r["estimate"]], [lo], [hi], color=MODEL_COLORS.get(m, INK["muted"]),
                        markerfacecolor="white" if r["exploratory"] else MODEL_COLORS.get(m, INK["muted"]), markersize=9)
        if metric == "r2_oos":
            a.axhline(0, color=INK["axis"], linewidth=0.8)
        a.set_xticks(range(len(order)), order)
        a.set_ylabel(label)
    n = int(c["n_athletes"].max()) if c.height else 0
    finish(fig, f"Unit: athlete (n = {n}, identical for every model). Repeated athlete-grouped K-fold; preprocessing and "
                "the ridge penalty are fitted inside each training fold. Points = mean over repeats, bars = 95 % athlete "
                "bootstrap of the out-of-fold predictions. Hollow marker = exploratory model. Overlapping bars do not "
                "test a difference: the paired comparison is CN_07." + _gate_note(store),
           "How well does each model predict held-out athletes?")
    store.save_figure(fig, "CN_06_model_performance", {"comparison": c}, publication=True, source=ev.COMPARISON.name, n=f"{n} athletes")


# ------------------------------------------------------------------------------ 7 incremental gain
def fig_gain(store: RescueStore, t: Tables) -> None:
    boot, sens = t[ev.BOOT.name], t[ev.SENSITIVITY.name]
    fig, ax = figure(1, 2, size=(12.8, 8.4))
    a = ax[0, 0]
    d = _pick(boot, metric="rmse")
    order = [m for m in model_specs() if m in d["model"].to_list()] if d.height else []
    if not order:
        empty_panel(a, "paired comparison not built")
    else:
        g = {r["model"]: r for r in d.iter_rows(named=True)}
        for i, m in enumerate(order):
            r = g[m]
            lo = r["ci_lower"] if r["ci_lower"] is not None else np.nan
            hi = r["ci_upper"] if r["ci_upper"] is not None else np.nan
            ci_errorbar(a, [i], [1000 * r["estimate"]], [1000 * lo], [1000 * hi], color=MODEL_COLORS.get(m, INK["muted"]),
                        markerfacecolor="white" if r["exploratory"] else MODEL_COLORS.get(m, INK["muted"]), markersize=9)
        a.axhline(0, color=INK["primary"], linewidth=1)
        a.set_xticks(range(len(order)), order)
        a.set_ylabel("RMSE(model) − RMSE(M1), milliseconds   (below 0 = better than M1)")
        a.set_title("primary analysis", fontsize=11)
    b = ax[0, 1]
    s = sens.filter(pl.col("model").is_in(list(TRACKING_MODELS)) & pl.col("d_rmse_vs_m1").is_not_null())
    names = list(dict.fromkeys(s["analysis"].to_list())) if s.height else []
    if not names:
        empty_panel(b, "sensitivity analyses not built")
    else:
        pos = {n: len(names) - 1 - i for i, n in enumerate(names)}
        for k, m in enumerate(TRACKING_MODELS):
            g = _pick(s, model=m)
            if not g.height:
                continue
            y = np.array([pos[n] for n in g["analysis"].to_list()], float) + (1 - k) * 0.22
            est, lo, hi = 1000 * _num(g, "d_rmse_vs_m1"), 1000 * _num(g, "d_rmse_lower"), 1000 * _num(g, "d_rmse_upper")
            b.errorbar(est, y, xerr=np.vstack([np.where(np.isfinite(lo), est - lo, 0), np.where(np.isfinite(hi), hi - est, 0)]),
                       fmt="o", markersize=4, capsize=2, color=MODEL_COLORS[m], label=m)
        b.axvline(0, color=INK["primary"], linewidth=1)
        b.set_yticks(list(pos.values()), [f"{n}" for n in pos], fontsize=7)
        b.set_xlabel("RMSE(model) − RMSE(M1), milliseconds")
        b.set_title("sensitivity analyses", fontsize=11)
        b.legend(frameon=False, fontsize=8)
    finish(fig, "Unit: athlete. Paired difference in out-of-fold RMSE on identical athletes and folds; bars = 95 % athlete "
                "bootstrap. An interval that contains 0 is no evidence of a gain. Right: each row refits every model under "
                "one changed condition (or scores the primary fit on a subset, 'scored_on'); sample sizes differ between "
                "rows and are in sensitivity_analyses.csv. Hollow marker = exploratory." + _gate_note(store),
           "Does Combine tracking add anything beyond the official tests?")
    store.save_figure(fig, "CN_07_incremental_gain", {"primary": d, "sensitivity": s}, publication=True,
                      source=f"{ev.BOOT.name} + {ev.SENSITIVITY.name}", n=f"{len(names)} analyses")


# ------------------------------------------------------------------------------ 8 matched athletes
def fig_matched(store: RescueStore, t: Tables) -> None:
    p = t[ev.MATCHED.name]
    fig, ax = figure(1, 2)
    a, b = ax[0, 0], ax[0, 1]
    if not p.height:
        empty_panel(a, "no pair inside the calipers")
        empty_panel(b, "no pair inside the calipers")
    else:
        a.scatter(_num(p, "d_feature"), 1000 * _num(p, "d_target"), s=30, color=CATEGORICAL[2])
        a.axhline(0, color=INK["primary"], linewidth=1)
        s = ev.matched_summary(p)
        if s["spearman"] is not None and np.isfinite(s["spearman"]):
            a.set_title(f"Spearman = {s['spearman']:.2f} [{s['ci_lower']:.2f}, {s['ci_upper']:.2f}], {s['n_pairs']} pairs", fontsize=11)
        a.set_xlabel(f"difference in {p['feature'][0]} within the pair (higher − lower)")
        a.set_ylabel("difference in context-adjusted get-off, same order (ms)")
        top = p.sort("d_feature", descending=True).head(GCFG.n_case_pairs)
        for i, r in enumerate(top.iter_rows(named=True)):
            y = top.height - 1 - i
            b.plot([r["target_b"], r["target_a"]], [y, y], color=INK["axis"], linewidth=1.5)
            b.scatter([r["target_a"]], [y], s=60, color=CATEGORICAL[2], zorder=3, label="higher tracked acceleration" if i == 0 else None)
            b.scatter([r["target_b"]], [y], s=60, color=INK["secondary"], zorder=3, label="lower tracked acceleration" if i == 0 else None)
            b.text(0.01, y + 0.28, f"{r['name_a']} ({r['forty_a']:.2f} s, {r['feature_a']:.2f}) vs {r['name_b']} "
                                   f"({r['forty_b']:.2f} s, {r['feature_b']:.2f}) · {r['alignment_group']}",
                   transform=b.get_yaxis_transform(), fontsize=8, color=INK["secondary"])
        b.set_yticks([])
        b.set_ylim(-0.6, top.height - 0.2)
        b.set_xlabel(f"context-adjusted mean {Y_LABEL}")
        b.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, f"Unit: pair of athletes of the same alignment group whose official forty differs by ≤ {GCFG.match_forty_s:g} s "
                f"and 10-yard split by ≤ {GCFG.match_ten_s:g} s (each athlete in one pair). Left: all pairs. Right: the "
                f"{GCFG.n_case_pairs} pairs with the largest difference in tracked acceleration; labels give official forty "
                "and the tracked value. An in-sample illustration, not a held-out result." + _gate_note(store),
           "Same stopwatch, different acceleration: do they differ in the NFL?")
    store.save_figure(fig, "CN_08_matched_athletes", {"pairs": p}, source=ev.MATCHED.name, n=f"{p.height} pairs")


FIGURES: list[tuple[str, tuple[str, ...], Callable[[RescueStore, Tables], None]]] = [
    ("CN_01_cohort_funnel", (co.FUNNEL.name,), fig_funnel),
    ("CN_02_getoff_distribution", (ep.HIST.name, ep.AUDIT.name), fig_distribution),
    ("CN_03_getoff_reliability", (tg.RELIABILITY.name,), fig_reliability),
    ("CN_04_combine_reliability_vs_redundancy", (), fig_combine_screen),
    ("CN_05_forty_vs_getoff", (co.BASE.name,), fig_forty),
    ("CN_06_model_performance", (ev.COMPARISON.name,), fig_models),
    ("CN_07_incremental_gain", (ev.BOOT.name, ev.SENSITIVITY.name), fig_gain),
    ("CN_08_matched_athletes", (ev.MATCHED.name,), fig_matched),
]
