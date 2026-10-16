"""Figures of notebook 08A.2 (``AM08A2_01`` … ``AM08A2_14``). Every figure is drawn from a persisted table of the
notebook (observed data only); an empty table gives a labelled empty panel, never a placeholder curve."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from . import opportunity_validation as ov
from . import outcome_semantics as osm

BLUE, ORANGE, GREEN, RED, GREY = "#4c78a8", "#f58518", "#54a24b", "#e45756", "#9d9d9d"
STATUS_COLOUR = {"eligible": GREEN, "provisional": ORANGE, "rejected": RED, "sensitivity_only": GREY, "unavailable": "#d9d9d9"}


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _empty(ax, text: str = "no data") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=GREY)
    ax.set_axis_off()


def fig_semantic_status(checks: pl.DataFrame, candidates: pl.DataFrame):
    """01 — consistency checks per outcome (green pass · red fail · grey informative) and the resulting status."""
    plt = _plt()
    ids = candidates["outcome_id"].to_list()
    fig, axes = plt.subplots(1, max(len(ids), 1), figsize=(5.2 * max(len(ids), 1), 5.2), squeeze=False)
    status = dict(zip(ids, candidates["definition_status"].to_list()))
    for ax, oid in zip(axes[0], ids):
        c = checks.filter(pl.col("outcome_id") == oid)
        if c.is_empty():
            _empty(ax)
            continue
        col = [GREY if p is None else (GREEN if p else RED) for p in c["passed"].to_list()]
        y = np.arange(c.height)[::-1]
        ax.barh(y, [1] * c.height, color=col, edgecolor="white")
        for yi, name, obs, blk in zip(y, c["check"].to_list(), c["observed"].to_list(), c["blocking"].to_list()):
            ax.text(0.02, yi, f"{'■ ' if blk else ''}{name}: {obs[:70]}", va="center", fontsize=6.5, color="white")
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{oid}\nstatus: {status[oid]}", fontsize=9)
    fig.suptitle("Outcome source and semantic status (■ = blocking check; grey = measurement without a pass / fail rule)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_missing_zero(zero: pl.DataFrame, fields: Sequence[str] = ("separation_at_pass_forward", "route_ran", "target", "tackle", "assist",
                                                                   "sack", "coverage_assignment")):
    """02 — null, zero and positive shares of the audited fields by position group."""
    plt = _plt()
    z = zero.filter(pl.col("field").is_in(list(fields)) & (pl.col("position_group") != osm.ALL))
    present = [f for f in fields if f in z["field"].unique().to_list()]
    fig, axes = plt.subplots(1, max(len(present), 1), figsize=(2.6 * max(len(present), 1) + 1, 3.8), squeeze=False, sharey=True)
    if not present:
        _empty(axes[0][0])
        return fig
    for ax, f in zip(axes[0], present):
        t = z.filter(pl.col("field") == f).sort("position_group")
        n = t["n_rows"].to_numpy().astype(float)
        nul, zer, pos = t["n_null"].to_numpy() / n, t["n_zero"].to_numpy() / n, t["n_positive"].to_numpy() / n
        x = np.arange(t.height)
        ax.bar(x, nul, color="#d9d9d9", label="null (not recorded)")
        ax.bar(x, zer, bottom=nul, color=BLUE, label="recorded 0")
        ax.bar(x, pos, bottom=nul + zer, color=ORANGE, label="recorded > 0")
        ax.set_xticks(x, t["position_group"].to_list(), fontsize=8)
        ax.set_title(f, fontsize=8)
    axes[0][0].set_ylabel("share of eligible player-plays")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Missing versus zero: a null is never read as 0", fontsize=10)
    fig.tight_layout()
    return fig


def _hist(ax, groups: dict[str, np.ndarray], xmax: float, xlabel: str) -> None:
    bins = np.linspace(0, xmax, 61)
    for (name, v), col in zip(groups.items(), (BLUE, ORANGE, GREEN, RED)):
        if len(v):
            ax.hist(np.clip(v, 0, xmax), bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=col,
                    label=f"{name} (n = {len(v):,}; median {np.median(v):.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)


def fig_separation(rows: pl.DataFrame, extreme_yd: float):
    """03 — distribution of every recorded separation; values above the extreme line are counted, not removed."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    if rows.is_empty():
        _empty(ax)
        return fig
    v = rows["value"].to_numpy()
    _hist(ax, {"all recorded (S0)": v}, extreme_yd * 1.2, "separation at pass forward (yd); last bin = clipped")
    ax.axvline(extreme_yd, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"FO01 separation: {100 * float((v > extreme_yd).mean()):.2f} % above {extreme_yd:g} yd · min {v.min():.2f} · max {v.max():.1f}", fontsize=10)
    fig.tight_layout()
    return fig


def fig_separation_by(rows: pl.DataFrame, by: str, extreme_yd: float, title: str):
    """04 / 05 — separation by target status or by position group."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    if rows.is_empty():
        _empty(ax)
        return fig
    groups = {str(k): rows.filter(pl.col(by) == k)["value"].to_numpy() for k in sorted(rows[by].drop_nulls().unique().to_list())}
    _hist(ax, groups, extreme_yd * 1.2, "separation at pass forward (yd)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def fig_context_cells(cells: pl.DataFrame):
    """06 — route × alignment coverage of the recorded separations; a frame marks the supported cells."""
    plt = _plt()
    if cells.is_empty():
        fig, ax = plt.subplots(figsize=(6, 3))
        _empty(ax, "no route label recorded")
        return fig
    routes = cells.group_by("route_ran").agg(pl.col("n_obs").sum()).sort("n_obs", descending=True)["route_ran"].to_list()
    aligns = cells.group_by("lined_up_position").agg(pl.col("n_obs").sum()).sort("n_obs", descending=True)["lined_up_position"].to_list()
    lut = {(r["route_ran"], r["lined_up_position"]): r for r in cells.iter_rows(named=True)}
    m = np.array([[float(lut.get((r, a), {}).get("n_obs", 0)) for a in aligns] for r in routes])
    fig, ax = plt.subplots(figsize=(1.0 * len(aligns) + 3, 0.42 * len(routes) + 1.8))
    im = ax.imshow(np.log10(m + 1), aspect="auto", cmap="Blues")
    for i, r in enumerate(routes):
        for j, a in enumerate(aligns):
            if m[i, j] > 0:
                ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center", fontsize=7, color="white" if m[i, j] > 0.3 * m.max() else "black")
            if lut.get((r, a), {}).get("supported", False):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ORANGE, linewidth=1.6))
    ax.set_xticks(range(len(aligns)), aligns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(routes)), routes, fontsize=8)
    ax.set_title("Recorded separations by route × alignment (orange frame = supported cell of S2)", fontsize=10)
    fig.colorbar(im, ax=ax, label="log10(observations + 1)")
    fig.tight_layout()
    return fig


def fig_structure(structure: pl.DataFrame, outcome_id: str, fields: Sequence[str], title: str):
    """07 / 08 — where the outcome is recorded: rows, recorded zeros and positives by value of a conditioning field."""
    plt = _plt()
    s = structure.filter(pl.col("outcome_id") == outcome_id)
    fields = [f for f in fields if f in s["group_field"].unique().to_list()]
    fig, axes = plt.subplots(1, max(len(fields), 1), figsize=(4.4 * max(len(fields), 1), 4.2), squeeze=False)
    if not fields:
        _empty(axes[0][0])
        return fig
    for ax, f in zip(axes[0], fields):
        t = s.filter(pl.col("group_field") == f).sort("n_rows", descending=True).head(10)
        y = np.arange(t.height)[::-1]
        n = t["n_rows"].to_numpy().astype(float)
        zer, pos = t["n_zero"].to_numpy() / n, t["n_positive"].to_numpy() / n
        ax.barh(y, zer, color=BLUE, label="recorded 0")
        ax.barh(y, pos, left=zer, color=ORANGE, label="recorded > 0")
        ax.barh(y, 1 - zer - pos, left=zer + pos, color="#d9d9d9", label="null")
        ax.set_yticks(y, [f"{v[:22]} (n = {int(k):,})" for v, k in zip(t["group_value"].to_list(), n)], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of eligible player-plays")
        ax.set_title(f, fontsize=9)
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def fig_reliability(reliability: pl.DataFrame, cfg: ov.ValidationConfig):
    """09 — odd / even-game reliability (athlete-bootstrap interval) and the mean over random game halves."""
    plt = _plt()
    r = reliability.filter(pl.col("position_group") == ov.ALL)
    ids = [d.denominator_id for d in ov.DENOMINATORS if d.denominator_id in r["denominator_id"].unique().to_list()]
    fig, ax = plt.subplots(figsize=(8, 0.45 * max(len(ids), 1) + 1.8))
    if not ids:
        _empty(ax)
        return fig
    lut = {(x["denominator_id"], x["split"]): x for x in r.iter_rows(named=True)}
    for i, k in enumerate(ids[::-1]):
        oe, rd = lut.get((k, "odd_even"), {}), lut.get((k, "random"), {})
        if oe.get("rel_sb") is not None:
            ax.plot([oe["rel_lo"], oe["rel_hi"]], [i, i], color=BLUE, linewidth=2)
            ax.plot(oe["rel_sb"], i, "o", color=BLUE, label="odd / even games [95 % athlete bootstrap]" if i == 0 else None)
        if rd.get("rel_sb") is not None:
            ax.plot(rd["rel_sb"], i + 0.22, "D", color=ORANGE, markersize=5, label="mean over random game halves" if i == 0 else None)
        ax.text(1.02, i, f"n = {oe.get('n_athletes', 0)} · {ov.BY_ID[k].legitimacy}", va="center", fontsize=7)
    ax.axvline(cfg.reliability_ref, color="black", linestyle="--", linewidth=1)
    ax.axvline(cfg.reliability_floor, color=GREY, linestyle=":", linewidth=1)
    ax.set_yticks(range(len(ids)), ids[::-1], fontsize=8)
    ax.set_xlim(-0.2, 1.25)
    ax.set_xlabel("Spearman–Brown reliability of the athlete value (halves are whole games)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Outcome reliability by opportunity definition", fontsize=10)
    fig.tight_layout()
    return fig


def fig_reliability_exposure(exposure: pl.DataFrame, show: Sequence[str]):
    """10 — reliability against game exposure."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode, col, lab in ((axes[0], "first_n_vs_later", "r", "correlation: first N games vs all later games"),
                               (axes[1], "min_games", "rel_sb", "odd / even reliability, athletes with ≥ N games")):
        e = exposure.filter((pl.col("mode") == mode) & pl.col("denominator_id").is_in(list(show)))
        if e.is_empty():
            _empty(ax)
            continue
        for k in show:
            t = e.filter(pl.col("denominator_id") == k).sort("level")
            if t.height:
                ax.plot(t["level"].to_numpy(), t[col].to_numpy().astype(float), "o-", label=k)
                for x, y, n in zip(t["level"].to_list(), t[col].to_list(), t["n_athletes"].to_list()):
                    if y is not None:
                        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 5), fontsize=6, ha="center")
        ax.set_xlabel("N games")
        ax.set_ylabel(lab, fontsize=8)
        ax.set_ylim(-0.2, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("Reliability versus game exposure (labels = athletes)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_funnel(funnel: pl.DataFrame, show: Sequence[str]):
    """11 — linked-cohort funnel of each denominator (all positions together)."""
    plt = _plt()
    f = funnel.filter((pl.col("position_group") == ov.ALL) & pl.col("cumulative") & pl.col("denominator_id").is_in(list(show)))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if f.is_empty():
        _empty(ax)
        return fig
    stages = f.select("stage", "stage_order").unique().sort("stage_order")["stage"].to_list()
    present = [k for k in show if k in f["denominator_id"].unique().to_list()]
    w = 0.8 / max(len(present), 1)
    for i, k in enumerate(present):
        t = dict(f.filter(pl.col("denominator_id") == k).select("stage", "n_athletes").iter_rows())
        y = [t.get(s, 0) for s in stages]
        x = np.arange(len(stages)) + i * w
        ax.bar(x, y, width=w, label=k)
        for xi, yi in zip(x, y):
            ax.text(xi, yi, str(yi), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(np.arange(len(stages)) + 0.4 - w / 2, [s.replace("_", "\n") for s in stages], fontsize=8)
    ax.set_ylabel("athletes")
    ax.legend(fontsize=7)
    ax.set_title("Linked-cohort funnel: NFL support → Combine link → every model family", fontsize=10)
    fig.tight_layout()
    return fig


def fig_support_heatmap(support: pl.DataFrame, show: Sequence[str]):
    """12 — linked athletes supporting every model family, position group × draft cohort."""
    plt = _plt()
    present = [k for k in show if k in support["denominator_id"].unique().to_list()]
    fig, axes = plt.subplots(1, max(len(present), 1), figsize=(3.6 * max(len(present), 1) + 0.6, 3.2), squeeze=False)
    if not present:
        _empty(axes[0][0])
        return fig
    for ax, k in zip(axes[0], present):
        t = support.filter(pl.col("denominator_id") == k)
        pos, yrs = sorted(t["position_group"].unique().to_list()), sorted(t["draft_year"].unique().to_list())
        lut = {(r["position_group"], r["draft_year"]): r for r in t.iter_rows(named=True)}
        m = np.array([[float(lut.get((p, y), {}).get("n_all_models", 0)) for y in yrs] for p in pos])
        ax.imshow(m, aspect="auto", cmap="Greens", vmin=0)
        for i, p in enumerate(pos):
            for j, y in enumerate(yrs):
                ax.text(j, i, f"{int(m[i, j])}/{int(lut.get((p, y), {}).get('n_supported_linked', 0))}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(yrs)), [str(y) for y in yrs])
        ax.set_yticks(range(len(pos)), pos)
        ax.set_title(k, fontsize=9)
    fig.suptitle("Position × draft-cohort support (all model families / linked with NFL support)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_ranking(ranking: pl.DataFrame, athletes: pl.DataFrame, pair: tuple[str, str] = ("FO01.S0", "FO01.S1")):
    """13 — ranking sensitivity to the opportunity definition: rank agreement of every pair, and one pair in detail."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    r = ranking.filter(pl.col("position_group") == ov.ALL)
    if r.is_empty():
        _empty(axes[0])
    else:
        y = np.arange(r.height)[::-1]
        rho = np.array([np.nan if x is None else x for x in r["spearman"].to_list()], float)
        axes[0].barh(y, rho, color=BLUE)
        for yi, v, s, n in zip(y, rho, r["mean_abs_pct_shift"].to_list(), r["n_common"].to_list()):
            axes[0].text(max(v, 0) + 0.01 if np.isfinite(v) else 0.01, yi, f"shift {s:.1f} pts · n = {n}" if s is not None else f"n = {n}", va="center", fontsize=7)
        axes[0].set_yticks(y, [f"{a} vs {b}" for a, b in zip(r["definition_a"].to_list(), r["definition_b"].to_list())], fontsize=7)
        axes[0].set_xlim(0, 1.35)
        axes[0].set_xlabel("Spearman of within-position percentiles (athletes supported by both)")
    a = athletes.filter((pl.col("denominator_id") == pair[0]) & pl.col("supported")).select("nfl_id", "position_group", pl.col("pct_position").alias("a"))
    b = athletes.filter((pl.col("denominator_id") == pair[1]) & pl.col("supported")).select("nfl_id", pl.col("pct_position").alias("b"))
    j = a.join(b, on="nfl_id", how="inner")
    if j.is_empty():
        _empty(axes[1])
    else:
        for p, col in zip(sorted(j["position_group"].unique().to_list()), (BLUE, ORANGE, GREEN, RED)):
            t = j.filter(pl.col("position_group") == p)
            axes[1].scatter(t["a"].to_numpy(), t["b"].to_numpy(), s=14, color=col, alpha=0.7, label=p)
        axes[1].plot([0, 100], [0, 100], color="black", linewidth=0.8)
        axes[1].set_xlabel(f"percentile under {pair[0]}")
        axes[1].set_ylabel(f"percentile under {pair[1]}")
        axes[1].legend(fontsize=8)
    fig.suptitle("Does the opportunity rule change the athlete ranking?", fontsize=10)
    fig.tight_layout()
    return fig


def fig_halves(athletes: pl.DataFrame, show: Sequence[str], min_half_note: str = ""):
    """Supplement — athlete value in odd versus even games (the raw material of the reliability estimate)."""
    plt = _plt()
    present = [k for k in show if k in athletes["denominator_id"].unique().to_list()]
    fig, axes = plt.subplots(1, max(len(present), 1), figsize=(4.2 * max(len(present), 1), 4), squeeze=False)
    if not present:
        _empty(axes[0][0])
        return fig
    for ax, k in zip(axes[0], present):
        t = athletes.filter((pl.col("denominator_id") == k) & pl.col("supported") & pl.col("half_a").is_not_null() & pl.col("half_b").is_not_null())
        for p, col in zip(sorted(t["position_group"].unique().to_list()), (BLUE, ORANGE, GREEN, RED)):
            u = t.filter(pl.col("position_group") == p)
            ax.scatter(u["half_a"].to_numpy(), u["half_b"].to_numpy(), s=12, alpha=0.7, color=col, label=p)
        ax.set_xlabel("odd games")
        ax.set_ylabel("even games")
        ax.set_title(f"{k} {min_half_note}", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_freeze_summary(screen: pl.DataFrame, gate: tuple[str, str], proposal: dict[str, Any], doc: dict[str, Any] | None):
    """14 — criteria met per denominator, screen status, the rule-based proposal and the recorded decision."""
    plt = _plt()
    crit = ["c_semantics", "c_denominator", "c_support", "c_reliability", "c_exposure"]
    fig, ax = plt.subplots(figsize=(10.5, 0.45 * max(screen.height, 1) + 2.4))
    if screen.is_empty():
        _empty(ax)
        return fig
    frozen = {t["target_id"]: t["role"] for t in (doc or {}).get("targets", [])}
    for i, r in enumerate(screen.iter_rows(named=True)):
        y = screen.height - 1 - i
        for j, c in enumerate(crit):
            ax.add_patch(plt.Rectangle((j, y - 0.4), 0.92, 0.8, color=GREEN if r[c] else "#f2b6b6"))
        ax.add_patch(plt.Rectangle((len(crit) + 0.1, y - 0.4), 1.9, 0.8, color=STATUS_COLOUR.get(r["status"], GREY)))
        ax.text(len(crit) + 1.05, y, r["status"], ha="center", va="center", fontsize=8, color="white")
        tag = ("FROZEN " + frozen[r["denominator_id"]] if r["denominator_id"] in frozen
               else "proposed primary" if r["denominator_id"] == proposal.get("primary")
               else "proposed secondary" if r["denominator_id"] in (proposal.get("secondary") or []) else "")
        ax.text(len(crit) + 2.15, y, tag, va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(range(screen.height), screen["denominator_id"].to_list()[::-1], fontsize=8)
    ax.set_xticks([j + 0.46 for j in range(len(crit))] + [len(crit) + 1.05],
                  ["semantics\nconfirmed", "denominator\nrecorded", "linked\nsupport", "reliability", "exposure\nindependence", "screen status"], fontsize=8)
    ax.set_xlim(0, len(crit) + 4.6)
    ax.set_ylim(-0.6, screen.height - 0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Outcome freeze decision — gate 08A.2: {gate[0]} · rule-based proposal: {proposal.get('decision')}\n{gate[1]}", fontsize=9)
    fig.tight_layout()
    return fig


__all__ = ["fig_context_cells", "fig_freeze_summary", "fig_funnel", "fig_halves", "fig_missing_zero", "fig_ranking", "fig_reliability",
           "fig_reliability_exposure", "fig_semantic_status", "fig_separation", "fig_separation_by", "fig_structure", "fig_support_heatmap"]
