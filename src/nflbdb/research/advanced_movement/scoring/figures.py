"""Figures of notebook 07D: the player scorecard and its supporting views. Matplotlib only; every function returns a
figure drawn from tables the notebook computed (real athletes, no invented value). A percentile bar always carries its
interval, its raw value and its reference size; a section without support says so instead of showing a number.
"""

from __future__ import annotations

import textwrap
from typing import Any, Sequence

import numpy as np
import polars as pl

from ....data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS
from ...figkit import empty_panel, figure, finish
from . import NFL_PREDICTIVE_STATUS
from .percentiles import FEATURE, FORTY, P95, TEN, V4_V8

TIER_COLOR = {"high": STATUS["good"], "medium": STATUS["warning"], "low": STATUS["critical"], "official_only": INK["axis"]}
NAVY, GOLD = "#0b1f3a", "#c9a227"
GRID = np.arange(0.0, 40.01, 0.5)


# ============================================================================== speed-by-distance curves
def athlete_curves(frames: pl.DataFrame, seq_ids: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(distance, speed)] of each sequence, as recorded (no extrapolation)."""
    f = frames.filter(pl.col("seq_id").is_in(list(seq_ids))).sort("seq_id", "frame")
    return [(g["dist"].to_numpy(), g["speed"].to_numpy()) for _, g in f.partition_by(["seq_id"], as_dict=True, maintain_order=True).items()]


def curve_band(frames: pl.DataFrame, seq_ids: Sequence[str], min_curves: int = 10) -> dict[str, np.ndarray]:
    """10th / 50th / 90th percentile of speed at each distance over the given sequences. A sequence contributes only
    inside its recorded distance range; a grid point with fewer than ``min_curves`` sequences is left empty."""
    M = []
    for d, s in athlete_curves(frames, seq_ids):
        ok = np.isfinite(d) & np.isfinite(s)
        if ok.sum() < 5:
            continue
        o = np.argsort(d[ok])
        M.append(np.interp(GRID, d[ok][o], s[ok][o], left=np.nan, right=np.nan))
    if not M:
        return {"grid": GRID, "q10": np.full(GRID.size, np.nan), "q50": np.full(GRID.size, np.nan), "q90": np.full(GRID.size, np.nan), "n": 0}
    A = np.vstack(M)
    n = np.isfinite(A).sum(0)
    with np.errstate(all="ignore"):
        q = np.nanquantile(A, [0.1, 0.5, 0.9], axis=0)
    q[:, n < min_curves] = np.nan
    return {"grid": GRID, "q10": q[0], "q50": q[1], "q90": q[2], "n": int(A.shape[0])}


def _curve_panel(ax, band: dict[str, np.ndarray], curves: dict[str, list[tuple[np.ndarray, np.ndarray]]], colors: dict[str, str], position: str) -> None:
    if np.isfinite(band["q50"]).any():
        ax.fill_between(band["grid"], band["q10"], band["q90"], color=INK["grid"], alpha=0.9, label=f"{position} peers, 10–90 % ({band['n']} runs)")
        ax.plot(band["grid"], band["q50"], color=INK["axis"], lw=1.2)
    for name, cs in curves.items():
        for i, (d, s) in enumerate(cs):
            ax.plot(d, s, color=colors[name], lw=2.0 if i == 0 else 1.2, alpha=1.0 if i == 0 else 0.6, label=name if i == 0 else None)
    ax.axhspan(4, 8, color=GOLD, alpha=0.08)
    ax.set_xlabel("distance covered [yd]")
    ax.set_ylabel("tracked speed [yd/s]")
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def _pct_bars(ax, rows: list[dict[str, Any]], color: str) -> None:
    """Horizontal percentile bars with intervals; an unavailable row is written, not drawn."""
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        spec = FEATURE[r["feature_id"]]
        head = spec.role == "headline"
        if r["percentile"] is None:
            ax.text(2, yi, f"not available — {r['note']}", va="center", fontsize=8, color=INK["muted"], style="italic")
        else:
            ax.barh(yi, r["percentile"], color=color, alpha=0.95 if head else 0.45, height=0.55)
            if r["pct_lower"] is not None and r["pct_upper"] is not None:
                ax.plot([r["pct_lower"], r["pct_upper"]], [yi, yi], color=INK["primary"], lw=1.6, solid_capstyle="butt")
            ax.text(min(r["percentile"] + 2, 86), yi + 0.34, f"{r['percentile']:.0f}th", fontsize=10 if head else 8, fontweight="bold" if head else "normal", va="bottom")
        raw = "—" if r["value"] is None else f"{r['value']:.2f} {r['unit']}"
        flag = "" if r["availability"] == "available" else f" [{r['availability']}]"
        ax.text(-2, yi, f"{spec.label}{flag}\n{raw} · ref. n = {r['n_reference']}", ha="right", va="center", fontsize=8)
    ax.axvline(50, color=INK["axis"], lw=0.8, ls=":")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("position percentile (100 = shortest time / highest speed)", fontsize=8)
    ax.grid(False)


def _text_panel(ax, title: str, lines: Sequence[str], width: int = 58, size: float = 8.5) -> None:
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=11, fontweight="bold", color=NAVY, va="top", transform=ax.transAxes)
    body = "\n".join("\n".join(textwrap.wrap(str(s), width)) or "" for s in lines)
    ax.text(0.0, 0.88, body, fontsize=size, va="top", transform=ax.transAxes, color=INK["primary"], linespacing=1.35)


# ============================================================================== 1 · player scorecard
def plot_scorecard(profile: dict[str, Any], pct_rows: pl.DataFrame, confidence: dict[str, Any], stability: dict[str, Any] | None,
                   band: dict[str, np.ndarray], curves: list[tuple[np.ndarray, np.ndarray]], nfl_panel: Sequence[str] | None = None) -> Any:
    """One athlete. ``pct_rows``: his rows of the percentile table. ``nfl_panel``: optional lines of OBSERVED NFL
    movement, shown in a separate, labelled box — never an input of anything else on the card."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.62, 2.5, 2.0], width_ratios=[1.25, 1.0, 1.15])
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.add_patch(Rectangle((0, 0), 1, 1, transform=head.transAxes, color=NAVY, zorder=0))
    f = lambda v, d=2, u="": "—" if v is None else f"{v:.{d}f}{u}"
    head.text(0.015, 0.62, str(profile.get("display_name") or profile["nfl_id"]), color="white", fontsize=22, fontweight="bold", va="center", transform=head.transAxes)
    head.text(0.015, 0.20, f"{profile['position_group']} · Combine {profile.get('draft_year') or '—'} · PRE-DRAFT ATHLETIC FINGERPRINT", color=GOLD, fontsize=11,
              va="center", transform=head.transAxes)
    head.text(0.60, 0.5, f"Forty {f(profile.get('official_forty'), 2, ' s')}   ·   10-yard {f(profile.get('official_ten'), 2, ' s')}   ·   "
                         f"{f(profile.get('weight'), 0, ' lb')}   ·   {f(profile.get('height'), 1, ' in')}", color="white", fontsize=12, va="center", ha="center",
              transform=head.transAxes)
    tier = confidence.get("evidence_tier", "official_only")
    head.text(0.985, 0.5, f"EVIDENCE: {tier.replace('_', ' ').upper()}", color="white", fontsize=11, fontweight="bold", va="center", ha="right", transform=head.transAxes,
              bbox={"boxstyle": "round,pad=0.4", "fc": TIER_COLOR.get(tier, INK["axis"]), "ec": "none"})

    rows = {r["feature_id"]: r for r in pct_rows.iter_rows(named=True)}
    a = fig.add_subplot(gs[1, 0])
    _pct_bars(a, [rows[k] for k in (FORTY, TEN, P95) if k in rows], CATEGORICAL[0])
    a.set_title("A · SPEED CAPACITY  (headline: official forty; components are not averaged in)", loc="left", fontsize=10, fontweight="bold", color=NAVY)
    b = fig.add_subplot(gs[1, 1])
    _pct_bars(b, [rows[k] for k in (V4_V8,) if k in rows], CATEGORICAL[2])
    b.set_title("B · SPEED DEVELOPMENT  (4 → 8 yd/s)", loc="left", fontsize=10, fontweight="bold", color=NAVY)
    c = fig.add_subplot(gs[1, 2])
    _curve_panel(c, band, {"this athlete": curves}, {"this athlete": NAVY}, profile["position_group"])
    c.set_title("Recorded speed against distance (gold band: 4–8 yd/s)", loc="left", fontsize=10, fontweight="bold", color=NAVY)

    st = stability or {}
    _text_panel(fig.add_subplot(gs[2, 0]), "C · MOVEMENT FINGERPRINT  (similarity, not a rating)", [
        f"Moves most like ({profile['position_group']}): {profile.get('fingerprint_nearest') or 'not available — no tracked curve'}",
        f"Among athletes with near-identical official times: {profile.get('fingerprint_nearest_similar_official') or 'none inside the caliper'}",
        f"Repeat stability: his second run is closer than {f(None if st.get('self_auc') is None else 100 * st['self_auc'], 0, ' %')} of position peers"
        + ("" if st.get("self_auc") is not None else " (needs two runs)"),
        f"Training-seed stability of his neighbours: {f(st.get('seed_overlap'), 2)} overlap (untrained encoder: {f(st.get('random_overlap'), 2)})",
        "Learned representation: its coordinates have no physical name. Distance from other athletes is not a quality."])
    _text_panel(fig.add_subplot(gs[2, 1]), "D · EVIDENCE CONFIDENCE  (about the recording)", [
        f"Tracked attempts: {confidence.get('n_attempts', 0)} · usable signal on {confidence.get('n_signal_ok', 0)} · speed development on {confidence.get('n_development_attempts', 0)}",
        f"Starts in motion: {'yes' if confidence.get('any_start_in_motion') else 'no'} (max first-frame speed {f(confidence.get('max_first_speed'), 1, ' yd/s')})",
        f"Sensor residual (max): {f(confidence.get('max_position_residual'), 3, ' yd')} · position reference: {confidence.get('reference_n', 0)} athletes",
        f"Why this tier: {confidence.get('reasons') or 'every rule of the high tier is met'}"], width=50)
    _text_panel(fig.add_subplot(gs[2, 2]), "HOW TO READ THIS CARD", [
        "Percentiles compare the athlete with Combine athletes of his position; bars carry 95 % intervals (measurement error and reference sampling).",
        "A missing or unsupported measurement shows no number. There is no overall score.",
        f"NFL status: {NFL_PREDICTIVE_STATUS}.",
        *(["", "OBSERVED IN THE NFL (separate, after the fact — not used above):", *nfl_panel] if nfl_panel else [])], width=62, size=8)
    return fig


# ============================================================================== 2 · matched prospects
def plot_matched_prospects(case: dict[str, Any], pct: pl.DataFrame, band: dict[str, np.ndarray], curves: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> Any:
    fig, axes = figure(1, 2)
    names = [case.get("name_a") or str(case["nfl_id_a"]), case.get("name_b") or str(case["nfl_id_b"])]
    cols = {names[0]: NAVY, names[1]: CATEGORICAL[3]}
    feats = [FORTY, TEN, P95, V4_V8]
    ax = axes[0, 0]
    for j, (i, n) in enumerate(zip((case["nfl_id_a"], case["nfl_id_b"]), names)):
        g = {r["feature_id"]: r for r in pct.filter(pl.col("nfl_id") == i).iter_rows(named=True)}
        for k, fid in enumerate(feats):
            r = g.get(fid)
            y = len(feats) - 1 - k + (0.14 if j == 0 else -0.14)
            if r is None or r["percentile"] is None:
                ax.text(1, y, "n/a", fontsize=8, color=cols[n], va="center")
                continue
            ax.plot([r["pct_lower"], r["pct_upper"]], [y, y], color=cols[n], lw=1.6)
            ax.scatter([r["percentile"]], [y], s=60, color=cols[n], zorder=3, label=n if k == 0 else None)
            ax.text(r["percentile"], y + 0.1, f"{r['value']:.2f} {r['unit']}", fontsize=7, ha="center", color=cols[n])
    ax.set_yticks(np.arange(len(feats))[::-1])
    ax.set_yticklabels([FEATURE[f].label for f in feats], fontsize=9)
    ax.set_xlim(0, 100)
    ax.axvline(50, color=INK["axis"], lw=0.8, ls=":")
    ax.set_xlabel("position percentile (100 = shortest time / highest speed)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _curve_panel(axes[0, 1], band, curves, cols, case["position_group"])
    finish(fig, f"Case '{case['case']}': {case['criterion']}. Chosen by this Combine-side rule, fixed in advance; no NFL outcome was consulted. "
                f"Fingerprint distance {case.get('fingerprint_distance')}. An illustration of what the profile shows, not evidence of anything in the NFL.",
           f"Matched prospects — {names[0]} and {names[1]} ({case['position_group']})")
    return fig


# ============================================================================== 3 · reference distributions
def plot_reference_distributions(pct: pl.DataFrame, feature_ids: Sequence[str]) -> Any:
    pos = sorted(pct["position_group"].drop_nulls().unique().to_list())
    fig, axes = figure(len(feature_ids), 1, size="tall" if len(feature_ids) > 2 else "wide")
    for i, fid in enumerate(feature_ids):
        ax = axes[i, 0]
        spec = FEATURE[fid]
        data, labels = [], []
        for p in pos:
            v = pct.filter((pl.col("feature_id") == fid) & (pl.col("position_group") == p) & pl.col("in_reference"))["value"].drop_nulls().to_numpy()
            if v.size >= 5:
                data.append(v)
                labels.append(f"{p}\n(n = {v.size})")
        if data:
            vp = ax.violinplot(data, showmedians=True, widths=0.8)
            for body in vp["bodies"]:
                body.set(facecolor=CATEGORICAL[i % len(CATEGORICAL)], alpha=0.5)
            ax.set_xticks(np.arange(1, len(data) + 1))
            ax.set_xticklabels(labels, fontsize=8)
        else:
            empty_panel(ax, "no reference population")
        ax.set_ylabel(f"{spec.label}\n[{spec.unit}]" + ("" if spec.higher_is_better else "  (lower = higher pct.)"), fontsize=8)
    finish(fig, "Reference populations the percentiles are relative to: Combine athletes of the position meeting the quality rule. A percentile is a place in "
                "one of these distributions — nothing more.", "Position reference distributions")
    return fig


# ============================================================================== 4 · speed-development curves
def plot_development_curves(groups: dict[str, dict[str, np.ndarray]], position: str) -> Any:
    """``groups``: label → curve band (``curve_band``) of the athletes of a speed-development percentile group."""
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    for i, (label, b) in enumerate(groups.items()):
        col = CATEGORICAL[(0, 3, 2)[i % 3]]
        if np.isfinite(b["q50"]).any():
            ax.fill_between(b["grid"], b["q10"], b["q90"], color=col, alpha=0.12)
            ax.plot(b["grid"], b["q50"], color=col, lw=2, label=f"{label} ({b['n']} runs)")
    ax.axhspan(4, 8, color=GOLD, alpha=0.08)
    ax.set_xlabel("distance covered [yd]")
    ax.set_ylabel("tracked speed [yd/s]")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    finish(fig, f"{position}: median and 10–90 % band of recorded speed against distance for athletes grouped by their speed-development percentile. "
                "The gold band is the 4–8 yd/s interval the measure times. Recorded range only — nothing is extrapolated.",
           f"What the speed-development percentile separates — {position}")
    return fig


# ============================================================================== 5 · uncertainty
def plot_uncertainty(pct: pl.DataFrame, feature_id: str) -> Any:
    fig, axes = figure(1, 2)
    g = pct.filter((pl.col("feature_id") == feature_id) & pl.col("percentile").is_not_null()).sort("percentile")
    spec = FEATURE[feature_id]
    if not g.height:
        empty_panel(axes[0, 0], "no percentile available")
        empty_panel(axes[0, 1], "no percentile available")
    else:
        x = np.arange(g.height)
        n = g["n_attempts"].to_numpy()
        col = np.where(n >= 2, CATEGORICAL[0], CATEGORICAL[3])
        axes[0, 0].vlines(x, g["pct_lower"], g["pct_upper"], colors=col, lw=0.8, alpha=0.7)
        axes[0, 0].scatter(x, g["percentile"], s=5, color=INK["primary"], zorder=3)
        axes[0, 0].set_xlabel("athletes, ordered by percentile")
        axes[0, 0].set_ylabel("position percentile with 95 % interval")
        width = (g["pct_upper"] - g["pct_lower"]).to_numpy()
        for k, lab, c in ((n < 2, "one attempt", CATEGORICAL[3]), (n >= 2, "two or more attempts", CATEGORICAL[0])):
            if k.any():
                axes[0, 1].scatter(g["percentile"].to_numpy()[k], width[k], s=10, color=c, alpha=0.7, label=f"{lab} (n = {int(k.sum())}, median width {np.median(width[k]):.0f})")
        axes[0, 1].set_xlabel("position percentile")
        axes[0, 1].set_ylabel("interval width [percentile points]")
        axes[0, 1].legend(frameon=False, fontsize=8)
    finish(fig, f"{spec.label}: {g.height} athletes with a percentile. Intervals combine the athlete's repeat-attempt measurement error and the sampling of the "
                "reference population; they are widest mid-distribution, where athletes are densest, and for athletes with one attempt.",
           f"How certain is a percentile? — {spec.label}")
    return fig


# ============================================================================== 6 · fingerprint similarity
def plot_fingerprint_similarity(stability: pl.DataFrame) -> Any:
    fig, axes = figure(1, 3)
    s = stability.filter(pl.col("self_auc").is_not_null())
    pos = sorted(s["position_group"].drop_nulls().unique().to_list())
    for i, p in enumerate(pos):
        v = s.filter(pl.col("position_group") == p)["self_auc"].to_numpy()
        axes[0, 0].hist(v, bins=np.linspace(0, 1, 21), histtype="step", lw=1.5, color=CATEGORICAL[i % len(CATEGORICAL)], label=f"{p} (n = {v.size}, mean {v.mean():.2f})")
    axes[0, 0].axvline(0.5, color=INK["secondary"], lw=1, ls="--")
    axes[0, 0].set_xlabel("share of position peers farther than the athlete's own repeat")
    axes[0, 0].set_ylabel("athletes")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].scatter(s["nearest_other_distance"], s["self_distance"], s=10, color=CATEGORICAL[6], alpha=0.6)
    lim = float(np.nanmax(np.r_[_arr(s["nearest_other_distance"]), _arr(s["self_distance"]), 0.1]))
    axes[0, 1].plot([0, lim], [0, lim], color=INK["secondary"], lw=1, ls="--")
    axes[0, 1].set_xlabel("distance to the nearest OTHER athlete (first runs)")
    axes[0, 1].set_ylabel("distance between the athlete's own two runs")
    o = stability.filter(pl.col("seed_overlap").is_not_null())
    if o.height:
        axes[0, 2].hist(_arr(o["seed_overlap"]), bins=np.linspace(0, 1, 21), color=CATEGORICAL[0], alpha=0.7, label=f"seed replicates (mean {o['seed_overlap'].mean():.2f})")
        r = _arr(stability["random_overlap"])
        if np.isfinite(r).any():
            axes[0, 2].hist(r[np.isfinite(r)], bins=np.linspace(0, 1, 21), histtype="step", color=INK["secondary"], lw=1.5, label=f"untrained encoder (mean {np.nanmean(r):.2f})")
        axes[0, 2].set_xlabel("overlap of the k nearest neighbours with the primary encoder")
        axes[0, 2].legend(frameon=False, fontsize=7)
    else:
        empty_panel(axes[0, 2], "no seed-replicate embeddings found")
    finish(fig, f"{s.height} athletes with two runs. Left: 1 = the athlete's repeat is closer than every peer, 0.5 = indistinguishable. Middle: below the diagonal, "
                "his own repeat is closer than his nearest neighbour. Right: do his neighbours survive retraining the encoder? Similarity is descriptive.",
           "How stable is a movement fingerprint?")
    return fig


def _arr(s: pl.Series) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in s], dtype=float)


# ============================================================================== 7 · quality against ranking stability
def plot_quality_stability(table: pl.DataFrame) -> Any:
    fig, axes = figure(1, 1)
    ax = axes[0, 0]
    tiers = [t for t in ("ALL", "high", "medium", "low") if t in table["evidence_tier"].to_list()]
    feats = table["feature_id"].unique().to_list()
    for j, fid in enumerate(feats):
        g = {r["evidence_tier"]: r for r in table.filter(pl.col("feature_id") == fid).iter_rows(named=True)}
        for i, t in enumerate(tiers):
            r = g.get(t)
            if r is None or r["rank_agreement"] is None:
                continue
            x = i + (j - (len(feats) - 1) / 2) * 0.22
            if r["ci_lower"] is not None:
                ax.plot([x, x], [r["ci_lower"], r["ci_upper"]], color=CATEGORICAL[(0, 2, 3)[j % 3]], lw=1.6)
            ax.scatter([x], [r["rank_agreement"]], s=60, color=CATEGORICAL[(0, 2, 3)[j % 3]], zorder=3, label=FEATURE[fid].label if i == 0 else None)
            ax.text(x, r["rank_agreement"] + 0.03, f"n={r['n_athletes']}", fontsize=7, ha="center", color=INK["muted"])
    ax.set_xticks(np.arange(len(tiers)))
    ax.set_xticklabels([t.replace("ALL", "all athletes") for t in tiers])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("evidence tier")
    ax.set_ylabel("rank agreement of attempt 1 and attempt 2 (within position)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    finish(fig, "Athletes with a repeat. Lower agreement in a tier means the recording orders its athletes less consistently — the reason confidence is shown "
                "beside every profile and kept apart from the athletic sections.", "Recording quality against ranking stability")
    return fig
