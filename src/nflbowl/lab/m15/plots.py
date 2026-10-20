"""M15 figures (T01 scope: data-audit figures). Every function returns ``(figure, data)`` where ``data`` maps a part
name to the exact plotted numbers (polars DataFrame) — pass both to ``M15Run.figure`` so PNG / SVG / PDF and the
source-data CSV / Parquet are written together. White background, colour-blind-safe fixed categorical order, physical
units on every axis, sample sizes in every title.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# fixed categorical order (dataviz reference palette, light mode): domain / series identity is ALSO given in text labels
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
DOMAIN_COLOR = {"combine": PALETTE[0], "nfl": PALETTE[1]}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
CHANNEL_LABEL = {"speed": "speed (yd/s)", "a_tan": "tangential acceleration (yd/s²)", "a_n": "normal acceleration (yd/s²)", "yaw": "heading rate (rad/s)"}


def _style(ax: Any) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)


def valid_frames_hist(index: Any, max_frames: int) -> tuple[Any, dict[str, Any]]:
    """E15.A1 preview — distribution of valid frames per sequence by domain (sequence counts; share per domain)."""
    import matplotlib.pyplot as plt
    import polars as pl

    bins = np.arange(0, max_frames + 2) - 0.5
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    parts = []
    for dom in ("combine", "nfl"):
        v = index.filter(pl.col("domain") == dom)["n_valid"].to_numpy()
        if not v.size:
            continue
        h, _ = np.histogram(v, bins=bins)
        share = h / max(v.size, 1)
        ax.step(np.arange(max_frames + 1), share, where="mid", color=DOMAIN_COLOR[dom], linewidth=2, label=f"{dom} (n = {v.size:,}; median {np.median(v):.0f})")
        parts.append(pl.DataFrame({"domain": [dom] * h.size, "valid_frames": np.arange(max_frames + 1), "n_sequences": h, "share": share}))
    _style(ax)
    ax.set_xlabel("valid frames per sequence (10 Hz)", color=INK)
    ax.set_ylabel("share of the domain's sequences", color=INK)
    ax.set_title("Valid frames per sequence by domain", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"valid_frames": pl.concat(parts) if parts else pl.DataFrame()}


def signal_panel(bundle: Any, rows: np.ndarray, labels: list[str], channels: tuple[str, ...] = ("speed", "a_tan", "yaw")) -> tuple[Any, dict[str, Any]]:
    """Physical-unit traces of a few sequences (valid frames only; masked frames are gaps, never drawn as zeros).
    One shared x-axis in seconds from the first frame (Σ dt); identical y-axes per channel across sequences."""
    import matplotlib.pyplot as plt
    import polars as pl

    from .data import physical_rows

    rows = np.asarray(rows, np.int64)
    phys = physical_rows(bundle, rows)
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.4 * len(channels)), sharex=True, facecolor="white")
    axes = np.atleast_1d(axes)
    parts = []
    for k, r in enumerate(rows):
        dt = np.where(np.isfinite(phys["dt"][k]), phys["dt"][k], 0.1)
        t = np.concatenate([[0.0], np.cumsum(dt[:-1])])
        color = PALETTE[k % len(PALETTE)]
        rec = {"row": np.full(t.size, int(r)), "label": [labels[k]] * t.size, "t_s": t, "valid": phys["valid"][k]}
        for ax, ch in zip(axes, channels, strict=True):
            y = phys[ch][k].copy()
            if ch == "yaw":
                y = np.where(phys["heading_valid"][k] == 1.0, y, np.nan)        # heading-invalid yaw is not an observation
            ax.plot(t, y, color=color, linewidth=1.8, label=labels[k])
            rec[ch] = y
        parts.append(pl.DataFrame(rec))
    for ax, ch in zip(axes, channels, strict=True):
        _style(ax)
        ax.set_ylabel(CHANNEL_LABEL.get(ch, ch), color=INK, fontsize=8)
    axes[-1].set_xlabel("time from first frame (s)", color=INK)
    axes[0].set_title(f"Sequences in physical units (n = {rows.size}; gaps = invalid frames)", color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig, {"traces": pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame()}


def audit_status_bar(audit: Any) -> tuple[Any, dict[str, Any]]:
    """Counts of data-contract checks per status (pass / warn / fail) — the one-glance audit summary."""
    import matplotlib.pyplot as plt
    import polars as pl

    order = ("pass", "warn", "fail")
    cnt = audit.group_by("status").len().to_dict(as_series=False)
    d = dict(zip(cnt["status"], cnt["len"], strict=True))
    vals = [int(d.get(s, 0)) for s in order]
    colors = {"pass": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b"}             # status palette, always paired with text labels
    fig, ax = plt.subplots(figsize=(6, 2.6), facecolor="white")
    ax.barh(list(order), vals, color=[colors[s] for s in order], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + 0.2, y, f"{v}", va="center", fontsize=9, color=INK)
    _style(ax)
    ax.set_xlabel("number of checks", color=INK)
    ax.set_title(f"Data-contract audit ({audit.height} checks)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig, {"audit_status": pl.DataFrame({"status": list(order), "n_checks": vals})}


# ====================================================================== E15.A2
def contrast_waterfall(steps: Any) -> tuple[Any, dict[str, Any]]:
    """Small multiples, one per metric: NFL − Combine median difference (athlete-cluster 95 % CI) at each matching step.
    A difference that shrinks toward 0 across steps is explained by weighting / length / entry speed, not by domain."""
    import matplotlib.pyplot as plt

    metrics = list(dict.fromkeys(steps["metric"].to_list()))
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.2 * nrow), facecolor="white", squeeze=False)
    for k, m in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        g = steps.filter(steps["metric"] == m).sort("step_order")
        x = np.arange(g.height)
        d, lo, hi = (g[c].to_numpy().astype(float) for c in ("diff_nfl_minus_combine", "lo", "hi"))
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.errorbar(x, d, yerr=[np.where(np.isfinite(lo), d - lo, 0), np.where(np.isfinite(hi), hi - d, 0)], fmt="o", color=PALETTE[0],
                    ecolor=PALETTE[0], capsize=3, markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in g["step"].to_list()], fontsize=7)
        _style(ax)
        ax.set_title(f"{m} ({g['unit'][0]})", fontsize=9, color=INK, loc="left")
    for k in range(len(metrics), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("NFL − Combine median difference under successive matching (95 % athlete-bootstrap CI)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return fig, {"steps": steps}


def kind_heatmap(kc: Any, metric: str) -> tuple[Any, dict[str, Any]]:
    """Combine drill × NFL demand kind matrix of athlete-weighted median differences for one metric (diverging scale,
    neutral at 0; values printed in every cell)."""
    import matplotlib.pyplot as plt

    g = kc.filter(kc["metric"] == metric)
    rows, cols = sorted(set(g["combine_kind"].to_list())), sorted(set(g["nfl_kind"].to_list()))
    M = np.full((len(rows), len(cols)), np.nan)
    for r in g.iter_rows(named=True):
        M[rows.index(r["combine_kind"]), cols.index(r["nfl_kind"])] = r["diff_nfl_minus_combine"]
    lim = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(cols), 1.2 + 0.6 * len(rows)), facecolor="white")
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel("NFL demand kind", color=INK)
    ax.set_ylabel("Combine drill", color=INK)
    ax.set_title(f"NFL − Combine athlete-weighted median: {metric}", fontsize=10, color=INK, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig, {"kind_contrast": g}


# ====================================================================== E15.A3 / A4
def centroid_vs_length(real: Any, domain_of: dict[int, str], fs: float, channel: str = "speed", nperseg: int = 32,
                       detrend: str = "linear", max_points: int = 6000, seed: int = 0) -> tuple[Any, dict[str, Any]]:
    """Spectral centroid against analysed segment length per domain, with the first non-zero bin (fs / nperseg) marked:
    a cloud sitting on that line is a resolution artifact, not a movement property."""
    import matplotlib.pyplot as plt
    import polars as pl

    g = real.filter((pl.col("channel") == channel) & (pl.col("nperseg") == nperseg) & (pl.col("detrend") == detrend) & pl.col("centroid").is_finite())
    g = g.with_columns(pl.col("idx").replace_strict(domain_of, default="?", return_dtype=pl.String).alias("domain"))
    if g.height > max_points:
        g = g.sample(max_points, seed=seed)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    for dom in ("combine", "nfl"):
        s = g.filter(pl.col("domain") == dom)
        if s.height:
            ax.scatter(s["seg_len"].to_numpy() + rng.uniform(-0.3, 0.3, s.height), s["centroid"].to_numpy(), s=6, alpha=0.35,
                       color=DOMAIN_COLOR[dom], edgecolor="none", label=f"{dom} (n = {s.height:,})")
    ax.axhline(fs / nperseg, color=INK_2, linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[1], fs / nperseg, f" first non-zero bin {fs / nperseg:.4f} Hz", va="bottom", ha="right", fontsize=8, color=INK_2)
    _style(ax)
    ax.set_xlabel("analysed gap-free segment length (frames, 10 Hz)", color=INK)
    ax.set_ylabel(f"{channel} spectral centroid (Hz)", color=INK)
    ax.set_title(f"Centroid vs segment length — Welch nperseg {nperseg}, detrend {detrend}", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"points": g.select("idx", "domain", "seg_len", "centroid")}


def psd_by_domain(bundle: Any, index: Any, cfg: Any, channel: str = "speed", length: int = 32, per_domain: int = 1500, seed: int = 0) -> tuple[Any, dict[str, Any]]:
    """Median ± IQR Welch PSD per domain on crops of exactly ``length`` frames (equal resolution in both domains)."""
    import matplotlib.pyplot as plt
    import polars as pl

    from ..m14.events import physical
    from .signals import crop_masks
    from .spectral import welch_psd

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    parts = []
    for dom in ("combine", "nfl"):
        rows = index.filter(pl.col("domain") == dom)["idx"].to_numpy()
        rows = np.sort(rng.choice(rows, min(per_domain, rows.size), replace=False)) if rows.size else rows
        masks, kept = crop_masks(bundle, rows, np.array([length]), cfg, seed)
        P, freqs = [], None
        for j in np.flatnonzero(kept == length):
            p = physical(np.asarray(bundle.X[rows[j]]), masks[j])
            est = welch_psd(p[channel][0][masks[j]], cfg.sampling_hz, length, "linear")
            if est is not None:
                freqs, psd, _ = est
                P.append(psd)
        if not P:
            continue
        P = np.stack(P)
        q25, q50, q75 = np.quantile(P, [0.25, 0.5, 0.75], axis=0)
        ax.plot(freqs, q50, color=DOMAIN_COLOR[dom], linewidth=2, label=f"{dom} (n = {P.shape[0]:,} crops)")
        ax.fill_between(freqs, q25, q75, color=DOMAIN_COLOR[dom], alpha=0.18, linewidth=0)
        parts.append(pl.DataFrame({"domain": [dom] * freqs.size, "freq_hz": freqs, "q25": q25, "median": q50, "q75": q75, "n": [P.shape[0]] * freqs.size}))
    ax.set_yscale("log")
    _style(ax)
    unit = "(yd/s)²/Hz" if channel == "speed" else "(yd/s²)²/Hz"
    ax.set_xlabel("frequency (Hz)", color=INK)
    ax.set_ylabel(f"{channel} PSD {unit}", color=INK)
    ax.set_title(f"Welch PSD at equal resolution ({length}-frame crops, Δf = {cfg.sampling_hz / length:.3f} Hz) — median and IQR", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, {"psd": pl.concat(parts) if parts else pl.DataFrame()}


def icc_forest(rel: Any, channel: str, nperseg: int, detrend: str = "linear") -> tuple[Any, dict[str, Any]]:
    """ICC(1,1) with 95 % CI per feature × scope for one configuration; reference lines at 0.4 and 0.6 (M14 rule)."""
    import matplotlib.pyplot as plt
    import polars as pl

    g = rel.filter((pl.col("channel") == channel) & (pl.col("nperseg") == nperseg) & (pl.col("detrend") == detrend) & pl.col("icc1").is_finite())
    g = g.sort("feature", "domain", "scope")
    labels = [f"{r['feature']} · {r['domain']} · {r['scope']}" for r in g.iter_rows(named=True)]
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.22 * g.height + 1)), facecolor="white")
    y = np.arange(g.height)
    icc, lo, hi = (g[c].to_numpy().astype(float) for c in ("icc1", "icc_lo", "icc_hi"))
    colors = [DOMAIN_COLOR.get(d, PALETTE[2]) for d in g["domain"].to_list()]
    ax.hlines(y, np.where(np.isfinite(lo), lo, icc), np.where(np.isfinite(hi), hi, icc), color=colors, linewidth=1.5)
    ax.scatter(icc, y, color=colors, s=18, zorder=3)
    for ref in (0.4, 0.6):
        ax.axvline(ref, color=INK_2, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    _style(ax)
    ax.set_xlabel("ICC(1,1) with 95 % athlete-bootstrap CI", color=INK)
    ax.set_title(f"Spectral feature reliability — {channel}, nperseg {nperseg}, detrend {detrend} (blue Combine, orange NFL)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    return fig, {"icc": g}


def spectrogram_gallery(bundle: Any, index: Any, cfg: Any, per_kind: int = 1, max_kinds: int = 8, seed: int = 0) -> tuple[Any, dict[str, Any]]:
    """|STFT| of the speed channel for one example sequence per kind (Combine drills and NFL demands), physical units."""
    import matplotlib.pyplot as plt
    import polars as pl

    from ..m14.events import physical
    from .spectral import stft_mag

    rng = np.random.default_rng(seed)
    picks = []
    for (dom, kind), g in index.filter(pl.col("n_valid") >= cfg.stft_nperseg).group_by("domain", "kind"):
        ix = g["idx"].to_numpy()
        picks += [(dom, kind, int(i)) for i in rng.choice(ix, min(per_kind, ix.size), replace=False)]
    picks = sorted(picks)[: max_kinds * 2]
    ncol = 4
    nrow = max(1, int(np.ceil(len(picks) / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.8 * nrow), facecolor="white", squeeze=False)
    parts = []
    for k, (dom, kind, i) in enumerate(picks):
        ax = axes[k // ncol][k % ncol]
        p = physical(np.asarray(bundle.X[i]), np.asarray(bundle.M[i]))
        S = stft_mag(p["speed"][0], p["valid"][0], cfg.stft_nperseg, cfg.stft_hop)
        freqs = np.fft.rfftfreq(cfg.stft_nperseg, 1.0 / cfg.sampling_hz)
        t = np.arange(S.shape[0]) * cfg.stft_hop / cfg.sampling_hz
        ax.pcolormesh(t, freqs, S.T, shading="nearest", cmap="viridis")
        ax.set_title(f"{dom} · {kind} · row {i}", fontsize=8, color=INK, loc="left")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("window start (s)", fontsize=7)
        ax.set_ylabel("Hz", fontsize=7)
        tt, ff = np.meshgrid(t, freqs, indexing="ij")
        parts.append(pl.DataFrame({"row": [i] * tt.size, "domain": [dom] * tt.size, "kind": [kind] * tt.size, "t_s": tt.ravel(), "freq_hz": ff.ravel(),
                                   "magnitude": S.ravel()}))
    for k in range(len(picks), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Speed |STFT| examples (nperseg {cfg.stft_nperseg}, hop {cfg.stft_hop}; magnitude in yd/s units)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return fig, {"spectrograms": pl.concat(parts) if parts else pl.DataFrame()}


__all__ = ["DOMAIN_COLOR", "PALETTE", "audit_status_bar", "centroid_vs_length", "contrast_waterfall", "icc_forest", "kind_heatmap", "psd_by_domain",
           "signal_panel", "spectrogram_gallery", "valid_frames_hist"]
