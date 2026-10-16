"""Stage P0 figures — drawn from the persisted P0 tables only (``--figures-only`` redraws without recomputation).

Authored 2026-10-11, NOT executed. Every figure is a census view: counts and shares, no inference. Captions end
with "PENDING interpretation" until the user has run the stage and the outputs have been reviewed.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..studies.store import FigureRecord, StudyStore
from .decision_audit import ALL, COVERAGE, EXPOSURE, FEAS, PAIRS, PER_PLAY, ROLES, SIDE, STUDY

VERDICT_STATUS = {"FEASIBLE": "good", "LIMITED": "warning", "INFEASIBLE": "critical"}
STREAM_NAMES = {"A": "A · capability twin", "B": "B · counterfactual engine", "C": "C · offline RL",
                "D": "D · multi-agent", "E": "E · decision metrics"}


def _kit():
    from ...data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS
    from ..figkit import empty_panel, figure, finish

    return CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS, empty_panel, figure, finish


def _heatmap(a, values: np.ndarray, labels: np.ndarray, rows: list[str], cols: list[str], seq: list[str], ink: dict,
             vmax: float | None = None, fontsize: int = 8):
    """Sequential heatmap with a text label in every cell (the label carries the value, colour the pattern)."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("p0_blue", [ink["surface"], *seq])
    cmap.set_bad(ink["grid"])
    top = vmax if vmax is not None else (float(np.nanmax(values)) if np.isfinite(values).any() else 1.0)
    im = a.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=top or 1.0, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            dark = np.isfinite(values[i, j]) and values[i, j] > 0.55 * (top or 1.0)
            a.text(j, i, labels[i, j], ha="center", va="center", fontsize=fontsize,
                   color="white" if dark else ink["primary"])
    a.set_xticks(np.arange(len(cols)), cols)
    a.set_yticks(np.arange(len(rows)), rows, fontsize=fontsize)
    a.grid(False)
    return im


def fig_players_per_play(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    cat, ink, _, _, empty_panel, figure, finish = _kit()
    d, side = out[PER_PLAY.name], out[SIDE.name]
    fig, ax = figure(1, 2)
    a, b = ax[0, 0], ax[0, 1]
    tot = d.filter(pl.col("season") == ALL).sort("n_tracked")
    if not tot.height:
        empty_panel(a, "no tracked plays")
    else:
        a.bar(tot["n_tracked"].to_numpy(), tot["share_plays"].to_numpy(), color=cat[0], width=0.8, label="all seasons")
        for k, season in enumerate(s for s in sorted(d["season"].unique().to_list()) if s != ALL):
            g = d.filter(pl.col("season") == season).sort("n_tracked")
            a.plot(g["n_tracked"].to_numpy(), g["share_plays"].to_numpy(), "o-", color=cat[1 + k % 7], ms=4, lw=1,
                   label=str(season))
        cap = tot.filter(pl.col("is_capped"))
        ticks = tot["n_tracked"].to_list()
        a.set_xticks(ticks, [f"≥ {t}" if cap.height and t == cap["n_tracked"].item() else str(t) for t in ticks])
        a.set_xlabel("tracked athletes on the play")
        a.set_ylabel("share of plays")
        a.legend(frameon=False, fontsize=8)
    s = side.sort("season")
    if not s.height:
        empty_panel(b, "no side information")
    else:
        y = np.arange(s.height)
        left = np.zeros(s.height)
        for k, (col, lab) in enumerate((("n_both_sides", "offense and defense"), ("n_offense_only", "offense only"),
                                        ("n_defense_only", "defense only"), ("n_unknown_side", "side unknown"))):
            v = s[col].to_numpy() / np.maximum(s["n_plays"].to_numpy(), 1)
            b.barh(y, v, left=left, color=cat[k], label=lab, edgecolor=ink["surface"], lw=0.8)
            left += v
        b.set_yticks(y, [f"{x} (n={n:,})" for x, n in zip(s["season"].to_list(), s["n_plays"].to_list())])
        b.invert_yaxis()
        b.set_xlim(0, 1)
        b.set_xlabel("share of plays, by side(s) with ≥ 1 tracked athlete")
        b.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, "Left: how many Combine-cohort athletes carry tracking on one play (a full play has 22 players). "
                "Right: whether those athletes are on one or both sides of the ball; side = player_play.team_abbr vs "
                "possession_team, unknown = no player_play row. Census, no inference. PENDING interpretation.",
           "P0-1 · How much of a play do we actually see?")
    store.save_figure(fig, rec, {"per_play": d, "side": side},
                      n=f"{int(tot['n_plays'].sum()) if tot.height else 0:,} plays")


def fig_opponent_pairs(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    _, ink, seq, _, empty_panel, figure, finish = _kit()
    p = out[PAIRS.name]
    fig, ax = figure(1, 2)
    if not p.height:
        for a in ax[0]:
            empty_panel(a, "no play has tracked athletes on both sides")
    else:
        offs, defs = sorted(p["off_position"].unique().to_list()), sorted(p["def_position"].unique().to_list())
        for a, col, title in ((ax[0, 0], "n_plays", "plays with both position groups tracked"),
                              (ax[0, 1], "n_distinct_pairs", "distinct athlete pairs")):
            m = np.full((len(offs), len(defs)), np.nan)
            lab = np.full(m.shape, "–", dtype=object)
            for r in p.iter_rows(named=True):
                i, j = offs.index(r["off_position"]), defs.index(r["def_position"])
                m[i, j], lab[i, j] = r[col], f"{r[col]:,}"
            _heatmap(a, m, lab, offs, defs, seq, ink, fontsize=9)
            a.set_xlabel("defensive position group")
            a.set_ylabel("offensive position group")
            a.set_title(title)
    finish(fig, "Co-tracking is availability, NOT a matchup: two athletes on the same play are not assumed to cover or "
                "block each other. Position group = gold.dim_athlete (Combine position). A matchup study needs many "
                "plays AND many distinct pairs. PENDING interpretation.",
           "P0-2 · Which opposing position groups are tracked together?")
    store.save_figure(fig, rec, {"pairs": p}, n=f"{int(p['n_pair_plays'].sum()) if p.height else 0:,} pair-plays")


def fig_field_coverage(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    _, ink, seq, _, empty_panel, figure, finish = _kit()
    c = out[COVERAGE.name]
    fig, ax = figure(1, len(ROLES), size=(12.8, 9.0))
    cols = [x for x in sorted(c["position_group"].unique().to_list()) if x != ALL] + [ALL] if c.height else []
    for k, role in enumerate(ROLES):
        a = ax[0, k]
        d = c.filter(pl.col("role") == role)
        if not d.height:
            empty_panel(a, f"no {role} field")
            continue
        fields = sorted(d["field"].unique().to_list())
        m = np.full((len(fields), len(cols)), np.nan)
        lab = np.full(m.shape, "", dtype=object)
        for r in d.iter_rows(named=True):
            i, j = fields.index(r["field"]), cols.index(r["position_group"])
            absent = r["source_table"] == "absent"
            m[i, j] = np.nan if absent or r["share_non_null"] is None else r["share_non_null"]
            lab[i, j] = "absent" if absent else "–" if r["share_non_null"] is None else f"{100 * r['share_non_null']:.0f}"
        _heatmap(a, m, lab, fields, cols, seq, ink, vmax=1.0, fontsize=7)
        a.set_title(f"{role} fields")
        a.set_xlabel("position group")
    finish(fig, "Cell = % of tracked player-plays with a non-null vendor value. Non-null is not 'applicable': boolean "
                "fields (target, blitzing, tackle) are filled on most rows, see n_true in the table. Fields that the "
                "silver split moved to the play level are constant across positions. PENDING interpretation.",
           "P0-3 · Which vendor states, actions and outcomes exist for which position?")
    store.save_figure(fig, rec, {"coverage": c},
                      n=f"{int(c.filter(pl.col('position_group') == ALL)['n_player_plays'].max() or 0) if c.height else 0:,} player-plays")


def fig_exposure(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    cat, ink, _, _, empty_panel, figure, finish = _kit()
    e = out[EXPOSURE.name]
    fig, ax = figure(1, 2)
    a, b = ax[0, 0], ax[0, 1]
    groups = [g for g in sorted(e["position_group"].unique().to_list()) if g != ALL] if e.height else []
    if not groups:
        empty_panel(a, "no athlete exposure")
        empty_panel(b, "no athlete exposure")
    else:
        for k, g in enumerate(groups):
            d = e.filter(pl.col("position_group") == g).sort("threshold")
            a.plot(np.arange(d.height), d["n_athletes_ge"].to_numpy(), "o-", color=cat[k % 8], ms=5, lw=1.5,
                   label=f"{g} (n={d['n_athletes'][0]})")
        thr = e.filter(pl.col("position_group") == groups[0]).sort("threshold")["threshold"].to_list()
        a.set_xticks(np.arange(len(thr)), [f"≥ {t}" for t in thr])
        a.set_xlabel("tracked plays per athlete")
        a.set_ylabel("athletes")
        a.set_ylim(bottom=0)
        a.legend(frameon=False, fontsize=8)
        q = e.unique(subset=["position_group"], keep="first").sort("position_group")
        y = np.arange(q.height)
        med = q["median_plays"].to_numpy()
        b.errorbar(med, y, xerr=np.vstack([med - q["p25_plays"].to_numpy(), q["p75_plays"].to_numpy() - med]).clip(0),
                   fmt="o", color=cat[0], ms=6, capsize=3)
        b.set_yticks(y, [f"{g} (n={n})" for g, n in zip(q["position_group"].to_list(), q["n_athletes"].to_list())])
        b.invert_yaxis()
        b.set_xlim(left=0)
        b.set_xlabel("tracked plays per athlete (median, whiskers = p25–p75)")
    finish(fig, "Exposure decides which athletes can carry a per-athlete NFL movement envelope. The minimum number of "
                "plays is NOT chosen here: Phase 04 M2 (reliability vs number of plays) sets it. Opportunity is a "
                "selection mechanism (role, depth chart, cohort), not athletic quality. PENDING interpretation.",
           "P0-4 · How many athletes have enough NFL plays?")
    tot = e.filter(pl.col("position_group") == ALL)
    store.save_figure(fig, rec, {"exposure": e}, n=f"{int(tot['n_athletes'][0]) if tot.height else 0:,} athletes")


def fig_stream_feasibility(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    _, ink, _, status, empty_panel, figure, finish = _kit()
    f = out[FEAS.name]
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    if not f.height:
        empty_panel(a, "no feasibility rows")
    else:
        cap = 2.5
        labels, y = [], 0
        for stream in sorted(f["stream"].unique().to_list()):
            d = f.filter(pl.col("stream") == stream)
            verdict = d["verdict"][0]
            a.text(0.0, y - 0.8, f"{STREAM_NAMES.get(stream, stream)} — {verdict}", ha="left", va="center",
                   fontsize=10, fontweight="semibold", color=status[VERDICT_STATUS.get(verdict, "critical")])
            for r in d.iter_rows(named=True):
                ratio = None if r["observed"] is None or not r["threshold"] else r["observed"] / r["threshold"]
                x = 0.0 if ratio is None else min(ratio, cap)
                a.plot([0, x], [y, y], color=ink["grid"], lw=3, solid_capstyle="round", zorder=1)
                a.plot(x, y, "o" if r["passed"] else "X", ms=9, zorder=2,
                       color=status["good"] if r["passed"] else status["critical"])
                obs = "n/a" if r["observed"] is None else f"{r['observed']:,.3g}"
                a.text(min(x, cap) + 0.06, y, f"{obs} vs ≥ {r['threshold']:,.3g}" + ("  (off scale)" if ratio and ratio > cap else ""),
                       va="center", fontsize=8, color=ink["secondary"])
                labels.append((y, r["criterion"].replace("_", " ")))
                y += 1
            y += 1.2
        a.axvline(1.0, color=ink["muted"], ls="--", lw=1)
        a.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=8)
        a.set_ylim(y - 0.8, -1.4)
        a.set_xlim(0, cap + 0.9)
        a.set_xlabel("observed ÷ threshold (dashed line = threshold; ● pass, ✕ fail)")
        a.grid(axis="y", visible=False)
    finish(fig, "Screening only: every threshold is PROPOSED (P05_thresholds.csv) and each observed value is exported, "
                "so a cut can change without recomputation. LIMITED means a restricted version of the stream is "
                "supportable (see `rule`), never that the full method is. PENDING interpretation and user review.",
           "P0-5 · Which frontier streams can the data support?")
    store.save_figure(fig, rec, {"feasibility": f}, n=f"{f.height} criteria")


def _rec(fid: str, name: str, q: str, insight: str, stat: str, ds: str, grain: str) -> FigureRecord:
    return FigureRecord(figure_id=fid, name=name, study=STUDY, research_question=q, expected_insight=insight,
                        statistic=stat, dataset=ds, grain=grain, filters="season types of DecisionAuditConfig",
                        method="exact census")


FIGURES = [
    (_rec("P0-01", "P0_01_tracked_athletes_per_play", "How many athletes are tracked per play, and on which sides?",
          "whether interaction / multi-agent modelling is possible at all", "share of plays",
          f"{PER_PLAY.name}, {SIDE.name}", "play"), fig_players_per_play),
    (_rec("P0-02", "P0_02_opponent_co_tracking", "Which opposing position groups are tracked on the same play?",
          "position pairs that could support a matchup study", "plays and distinct athlete pairs", PAIRS.name,
          "play × athlete pair"), fig_opponent_pairs),
    (_rec("P0-03", "P0_03_vendor_field_coverage", "Which vendor state / action / outcome fields exist per position?",
          "identifiable outcomes and action labels per position group", "share non-null", COVERAGE.name,
          "player-play"), fig_field_coverage),
    (_rec("P0-04", "P0_04_athlete_exposure", "How many athletes have enough tracked NFL plays?",
          "athlete sample for a per-athlete movement envelope", "athletes ≥ N plays; quartiles", EXPOSURE.name,
          "athlete"), fig_exposure),
    (_rec("P0-05", "P0_05_stream_feasibility", "Which frontier streams does the census support?",
          "streams to pursue, restrict or drop", "observed ÷ proposed threshold", FEAS.name, "criterion"),
     fig_stream_feasibility),
]
