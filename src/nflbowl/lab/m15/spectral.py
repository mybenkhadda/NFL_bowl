"""E15.A3 / E15.A4 — spectral features at controlled frequency resolution, the resolution-artifact test, reliability
and perturbation sensitivity. NumPy only (no SciPy needed).

Why (Finding 5 / master roadmap § 1.2-6, [EXP]): at 10 Hz with ≤ 64 frames (NFL median ≈ 43) the frequency resolution is
coarse: 0.3125 Hz for a 32-sample Welch segment. The reported speed spectral centroid (≈ 0.312 Hz Combine, 0.319 Hz NFL)
equals the first non-zero bin of a 32-sample segment (10/32 Hz), so it may describe the estimator, not the athlete.
Nothing spectral is promoted until this module shows it is estimable.

Estimation (per sequence, per channel ∈ {speed, a_tan}, per ``nperseg`` ∈ cfg.welch_nperseg, per detrend ∈ cfg.detrend):
* the longest gap-free valid segment (M14 segment rule) — never across gaps or invalid frames; segments shorter than
  ``nperseg`` give no estimate (recorded, not padded);
* Welch: Hann window, 50 % overlap, per-window detrend (``linear`` / ``constant`` / ``none``), one-sided density PSD
  (V²/Hz with V the channel unit), averaged over windows;
* features (the DC bin is excluded from every SHAPE feature): total power, normalised spectral entropy, centroid,
  85 % roll-off, flatness (geometric / arithmetic mean), peak frequency, band-power shares for ``cfg.bands_hz``.

Resolution-artifact test (``artifact_test``), per feature × configuration × domain:
* ``rho_len`` — Spearman of the feature with the analysed segment length (within domain);
* ``pinned_share`` — share of centroids within half a bin of the first non-zero bin fs/nperseg;
* ``auc_vs_ramp`` (detrend ``none`` / ``constant``) — AUC separating real segments from a least-squares RAMP with the
  same endpoints-fit (pure trend): a feature that cannot tell movement from its own trend line is trend-dominated;
* ``auc_vs_noise`` — AUC separating real segments from white noise with the segment's variance (a feature that cannot
  tell movement from noise carries no structure).
Phase-randomised and time-reversed surrogates are NOT used: both preserve the power spectrum exactly, so every PSD
feature would be identical by construction (an uninformative test).
Estimable := |rho_len| ≤ 0.3 AND pinned_share ≤ 0.5 AND (auc_vs_ramp ≥ 0.7 when defined, else auc_vs_noise ≥ 0.7).

Reliability (E15.A4): ICC(1,1) across repeated trials per Combine drill (≥ 30 athletes with ≥ 2 trials) and across NFL
windows within demand kind after kind-centring (``reliability.kind_centre``), with athlete-cluster bootstrap CIs; plus a
season / random-half split of NFL windows (Spearman–Brown). Perturbation: ``m14.metrics.perturbation`` (noise, smoothing)
on a seeded athlete sample → Spearman between perturbed and original values.
Measurement status = the M14 rule: validated (estimable, ICC ≥ 0.6, lower CI ≥ 0.4, min perturbation ρ ≥ 0.8) ·
invalid (not estimable, or ICC < 0.4) · exploratory (otherwise).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..status import BLOCKED_INSUFFICIENT_SAMPLE
from .data import chunks

NAN = float("nan")
CHANNELS = ("speed", "a_tan")
FEATURES = ("total_power", "entropy", "centroid", "rolloff85", "flatness", "peak_freq")
EPS = 1e-12


# ====================================================================== estimators
def _detrend(w: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return w
    if kind == "constant":
        return w - w.mean(axis=-1, keepdims=True)
    t = np.arange(w.shape[-1], dtype=float)
    t = t - t.mean()
    wc = w - w.mean(axis=-1, keepdims=True)
    slope = (wc * t).sum(-1, keepdims=True) / (t ** 2).sum()
    return wc - slope * t


def welch_psd(seg: np.ndarray, fs: float, nperseg: int, detrend: str = "linear") -> tuple[np.ndarray, np.ndarray, int] | None:
    """One-sided Welch PSD of a 1-D gap-free segment (Hann, 50 % overlap). None when len(seg) < nperseg."""
    seg = np.asarray(seg, float)
    if seg.size < nperseg or not np.isfinite(seg).all():
        return None
    step = max(1, nperseg // 2)
    starts = np.arange(0, seg.size - nperseg + 1, step)
    W = np.stack([seg[s:s + nperseg] for s in starts])
    W = _detrend(W, detrend)
    win = np.hanning(nperseg + 2)[1:-1]                      # strictly positive Hann (no zero end-points)
    X = np.fft.rfft(W * win, axis=-1)
    psd = (np.abs(X) ** 2) / (fs * (win ** 2).sum())
    if nperseg % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0
    return np.fft.rfftfreq(nperseg, 1.0 / fs), psd.mean(0), int(starts.size)


def spectral_features(freqs: np.ndarray, psd: np.ndarray, bands: Sequence[tuple[float, float]]) -> dict[str, float]:
    """Shape features exclude the DC bin; band shares use every non-DC bin (the last band includes Nyquist)."""
    f, p = freqs[1:], np.maximum(psd[1:], 0.0)
    tot = float(p.sum())
    out = {"total_power": float(psd.sum() * (freqs[1] - freqs[0])) if freqs.size > 1 else NAN}
    if tot <= EPS or f.size < 2:
        out.update({k: NAN for k in ("entropy", "centroid", "rolloff85", "flatness", "peak_freq")})
        out.update({f"band_{i}": NAN for i in range(len(bands))})
        return out
    q = p / tot
    out["entropy"] = float(-(q * np.log(q + EPS)).sum() / np.log(q.size))
    out["centroid"] = float((f * q).sum())
    out["rolloff85"] = float(f[min(int(np.searchsorted(np.cumsum(q), 0.85)), f.size - 1)])
    out["flatness"] = float(np.exp(np.mean(np.log(p + EPS))) / (p.mean() + EPS))
    out["peak_freq"] = float(f[int(np.argmax(p))])
    for i, (lo, hi) in enumerate(bands):
        last = i == len(bands) - 1
        m = (f >= lo) & ((f <= hi) if last else (f < hi))
        out[f"band_{i}"] = float(q[m].sum())
    return out


def stft_mag(x: np.ndarray, valid: np.ndarray, nperseg: int, hop: int, phase: bool = False) -> np.ndarray:
    """|STFT| (and optionally sin/cos phase) of one fixed-length sequence for encoder inputs (T07). Invalid frames are
    set to the segment-free value 0 AFTER mean removal over valid frames; a validity channel must accompany the input."""
    x = np.where(valid, x, np.nan)
    mu = np.nanmean(x) if valid.any() else 0.0
    z = np.where(valid, x - mu, 0.0)
    win = np.hanning(nperseg + 2)[1:-1]
    starts = np.arange(0, max(1, z.size - nperseg + 1), hop)
    F = np.fft.rfft(np.stack([z[s:s + nperseg] * win for s in starts if s + nperseg <= z.size]), axis=-1)
    mag = np.abs(F)
    return np.concatenate([mag, np.sin(np.angle(F)), np.cos(np.angle(F))], axis=-1) if phase else mag


def cwt_morlet(seg: np.ndarray, fs: float, freqs: Sequence[float], w0: float = 6.0) -> np.ndarray:
    """|CWT|² of a gap-free segment with a complex Morlet wavelet at the given frequencies (NumPy convolution).
    Cone-of-influence edges are set to NaN (half the wavelet support at each end)."""
    seg = np.asarray(seg, float) - np.mean(seg)
    out = np.full((len(freqs), seg.size), np.nan)
    for i, f in enumerate(freqs):
        s = w0 * fs / (2 * np.pi * f)                           # scale in samples
        half = int(np.ceil(3 * s))
        t = np.arange(-half, half + 1)
        wav = np.exp(1j * w0 * t / s) * np.exp(-0.5 * (t / s) ** 2) / np.sqrt(s)
        if 2 * half + 1 > seg.size:
            continue
        c = np.convolve(seg, np.conj(wav)[::-1], mode="same")
        p = np.abs(c) ** 2
        p[:half] = np.nan
        p[seg.size - half:] = np.nan
        out[i] = p
    return out


# ====================================================================== per-sequence table
def _longest_segment(valid: np.ndarray, dt: np.ndarray, max_gap_dt: float) -> tuple[int, int] | None:
    from ..m14.events import segments

    segs = segments(valid, dt, max_gap_dt, 1)
    return max(segs, key=lambda t: t[1] - t[0]) if segs else None


def spectral_rows(phys: dict[str, np.ndarray], idx: np.ndarray, cfg: Any, configs: Sequence[tuple[str, int, str]],
                  transform: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None, seed: int = 0) -> list[dict[str, Any]]:
    """Feature rows for physical arrays of a chunk. ``configs`` = (channel, nperseg, detrend). ``transform(seg, rng)``
    replaces the real segment (surrogates)."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for j, i in enumerate(idx):
        seg_ix = _longest_segment(phys["valid"][j], phys["dt"][j], cfg.max_gap_dt)
        if seg_ix is None:
            continue
        a, b = seg_ix
        for ch, nps, det in configs:
            seg = phys[ch][j, a:b]
            if transform is not None:
                seg = transform(seg, rng)
            est = welch_psd(seg, cfg.sampling_hz, nps, det)
            rec = {"idx": int(i), "channel": ch, "nperseg": int(nps), "detrend": det, "seg_len": int(b - a), "n_windows": 0}
            if est is not None:
                fr, p, nw = est
                rec.update(spectral_features(fr, p, cfg.bands_hz))
                rec["n_windows"] = nw
            rows.append(rec)
    return rows


def all_configs(cfg: Any) -> list[tuple[str, int, str]]:
    return [(ch, int(n), d) for ch in CHANNELS for n in cfg.welch_nperseg for d in cfg.detrend]


def spectral_table(bundle: Any, rows: np.ndarray, cfg: Any, configs: Sequence[tuple[str, int, str]] | None = None, mask: np.ndarray | None = None,
                   transform: Callable | None = None, perturb: Callable | None = None, seed: int = 0) -> Any:
    """Spectral features of bundle ``rows`` (chunked). ``mask`` (len(rows), L) overrides validity (crops);
    ``perturb(arrs, rng)`` is an ``m14.metrics.perturbation`` function applied to the physical arrays first."""
    import polars as pl

    from ..m14.events import physical

    rows = np.asarray(rows, np.int64)
    configs = list(configs or all_configs(cfg))
    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for s, e in chunks(rows.size, cfg.chunk_rows):
        m = np.asarray(bundle.M[rows[s:e]], bool) if mask is None else mask[s:e]
        p = physical(np.asarray(bundle.X[rows[s:e]]), m)
        if perturb is not None:
            p = perturb(p, rng)
        out += spectral_rows(p, rows[s:e], cfg, configs, transform, seed + s)
    if not out:
        return pl.DataFrame(schema={"idx": pl.Int64, "channel": pl.String, "nperseg": pl.Int64, "detrend": pl.String})
    return pl.DataFrame(out, infer_schema_length=None)


# ====================================================================== surrogates
def ramp_surrogate(seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Least-squares straight line through the segment: its trend and nothing else."""
    t = np.arange(seg.size, dtype=float)
    b, a = np.polyfit(t, seg, 1) if seg.size >= 2 else (0.0, float(seg.mean()))
    return a + b * t


def noise_surrogate(seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise with the segment's mean and standard deviation."""
    return rng.normal(float(np.mean(seg)), float(np.std(seg)) + EPS, seg.size)


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + ½ P(a = b) (Mann–Whitney), symmetrised to max(AUC, 1 − AUC): separability regardless of direction."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 10 or b.size < 10:
        return NAN
    allv = np.concatenate([a, b])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - (cnt - 1) / 2.0)[inv]                        # average ranks for ties
    u = avg[: a.size].sum() - a.size * (a.size + 1) / 2
    auc = u / (a.size * b.size)
    return float(max(auc, 1 - auc))


# ====================================================================== E15.A3 artifact test
def feature_columns(n_bands: int) -> list[str]:
    return [*FEATURES, *[f"band_{i}" for i in range(n_bands)]]


def artifact_test(real: Any, ramp: Any, noise: Any, domain_of: dict[int, str], cfg: Any) -> Any:
    """One row per (channel, nperseg, detrend, feature, domain): rho_len, pinned_share (centroid only), auc_vs_ramp,
    auc_vs_noise, n, and the ``estimable`` verdict (module docstring)."""
    import polars as pl

    from ..stats import spearman

    keys = ["idx", "channel", "nperseg", "detrend"]
    rows = []
    feats = feature_columns(len(cfg.bands_hz))
    real = real.with_columns(pl.col("idx").replace_strict(domain_of, default="?", return_dtype=pl.String).alias("domain"))
    for (ch, nps, det, dom), g in real.group_by("channel", "nperseg", "detrend", "domain"):
        rp = ramp.join(g.select(keys), on=keys, how="semi") if ramp is not None and ramp.height else None
        nz = noise.join(g.select(keys), on=keys, how="semi") if noise is not None and noise.height else None
        first_bin = cfg.sampling_hz / nps
        for f in feats:
            if f not in g.columns:
                continue
            v = g[f].cast(pl.Float64).to_numpy()
            ok = np.isfinite(v)
            if ok.sum() < 30:
                continue
            rho = spearman(v[ok], g["seg_len"].to_numpy().astype(float)[ok])
            pinned = float(np.mean(np.abs(v[ok] - first_bin) <= first_bin / 2)) if f == "centroid" else NAN
            a_r = _auc(v, rp[f].cast(pl.Float64).to_numpy()) if (rp is not None and f in rp.columns and det != "linear") else NAN
            a_n = _auc(v, nz[f].cast(pl.Float64).to_numpy()) if (nz is not None and f in nz.columns) else NAN
            sep = a_r if np.isfinite(a_r) else a_n
            est = bool(np.isfinite(rho) and abs(rho) <= 0.3 and (not np.isfinite(pinned) or pinned <= 0.5) and np.isfinite(sep) and sep >= 0.7)
            rows.append({"channel": ch, "nperseg": int(nps), "detrend": det, "domain": dom, "feature": f, "n": int(ok.sum()),
                         "median": float(np.median(v[ok])), "first_bin_hz": first_bin, "rho_len": rho, "pinned_share": pinned,
                         "auc_vs_ramp": a_r, "auc_vs_noise": a_n, "estimable": est})
    return pl.DataFrame(rows, infer_schema_length=None)


def length_matched_spectral(bundle: Any, index: Any, cfg: Any, length: int, seed: int, configs: Sequence[tuple[str, int, str]]) -> Any:
    """Spectral features of BOTH domains on crops of exactly ``length`` frames (rows whose longest gap-free segment is
    shorter are excluded): removes window length as an explanation of domain differences."""
    from .signals import crop_masks

    rows = index["idx"].to_numpy()
    masks, kept = crop_masks(bundle, rows, np.array([length]), cfg, seed)
    full = kept == length
    tab = spectral_table(bundle, rows[full], cfg, configs, mask=masks[full], seed=seed)
    return tab.join(index.select("idx", "domain", "kind", "nfl_id", "position_group"), on="idx", how="left")


def domain_contrast(tab: Any, cfg: Any, n_boot: int, seed: int) -> Any:
    """Athlete-weighted NFL − Combine median difference of every spectral feature per configuration (bootstrap CI)."""
    import polars as pl

    from ..stats import cluster_bootstrap_diff

    rows = []
    for k, ((ch, nps, det), g) in enumerate(tab.group_by("channel", "nperseg", "detrend")):
        for j, f in enumerate(feature_columns(len(cfg.bands_hz))):
            if f not in g.columns:
                continue
            a = g.filter(pl.col(f).is_not_null() & pl.col(f).is_finite()).group_by("domain", "nfl_id").agg(pl.col(f).median())
            n = a.filter(pl.col("domain") == "nfl")
            c = a.filter(pl.col("domain") == "combine")
            r = cluster_bootstrap_diff(n[f].to_numpy(), n["nfl_id"].to_numpy(), c[f].to_numpy(), c["nfl_id"].to_numpy(), np.median, n_boot,
                                       seed + 31 * k + j)
            rows.append({"channel": ch, "nperseg": int(nps), "detrend": det, "feature": f, "nfl_median": r["a"], "combine_median": r["b"],
                         "diff_nfl_minus_combine": r["diff"], "lo": r["lo"], "hi": r["hi"], "athletes_nfl": r["clusters_a"], "athletes_combine": r["clusters_b"]})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== E15.A4 reliability and sensitivity
def reliability(tab: Any, cfg: Any, n_boot: int, seed: int, min_athletes: int = 30) -> Any:
    """ICC(1,1) [bootstrap CI] per (configuration, feature, domain, kind scope). Combine: per drill. NFL: windows of all
    demand kinds after kind-centring (centre means from all athletes: descriptive reliability, no outcome involved)."""
    import polars as pl

    from ..reliability import icc_bootstrap, icc_components, kind_centre

    rows = []
    feats = feature_columns(len(cfg.bands_hz))
    for k, ((ch, nps, det, dom), g) in enumerate(tab.group_by("channel", "nperseg", "detrend", "domain")):
        scopes = [(str(kd), g.filter(pl.col("kind") == kd)) for kd in sorted(g["kind"].unique().to_list())] if dom == "combine" else [("all_kinds_centred", g)]
        for scope, s in scopes:
            for j, f in enumerate(feats):
                if f not in s.columns:
                    continue
                t = s.filter(pl.col(f).is_not_null() & pl.col(f).is_finite()).select("nfl_id", "kind", pl.col(f).cast(pl.Float64).alias("value"))
                if not t.height:
                    continue
                v = kind_centre(t) if scope == "all_kinds_centred" else t["value"].to_numpy()
                ids = t["nfl_id"].to_numpy()
                comp = icc_components(v, ids)
                if comp["n_athletes"] < min_athletes:
                    rows.append({"channel": ch, "nperseg": int(nps), "detrend": det, "domain": dom, "scope": scope, "feature": f, **comp,
                                 "icc_lo": NAN, "icc_hi": NAN, "status_note": BLOCKED_INSUFFICIENT_SAMPLE})
                    continue
                lo, hi = icc_bootstrap(v, ids, n_boot, seed + 13 * k + j)
                rows.append({"channel": ch, "nperseg": int(nps), "detrend": det, "domain": dom, "scope": scope, "feature": f, **comp,
                             "icc_lo": lo, "icc_hi": hi, "status_note": ""})
    return pl.DataFrame(rows, infer_schema_length=None)


def nfl_split_reliability(tab: Any, meta_nfl: Any, split: str, seed: int, min_windows: int = 4) -> Any:
    """Split-half reliability of athlete means over NFL windows: halves by ``season`` (odd / even season rank), by a
    parsed ``game_key`` (alternate games), or ``random``. Pearson r of the two half-means and Spearman–Brown 2r/(1+r)."""
    import polars as pl

    d = tab.join(meta_nfl, on="idx", how="inner")
    if split == "season" and "season" in d.columns:
        rank = {s: i for i, s in enumerate(sorted(d["season"].drop_nulls().unique().to_list()))}
        d = d.with_columns((pl.col("season").replace_strict(rank, default=0) % 2).alias("half"))
    elif split == "game" and "game_key" in d.columns:
        g = d.select("nfl_id", "game_key").unique().sort("nfl_id", "game_key").with_columns(
            (pl.int_range(pl.len()).over("nfl_id") % 2).alias("half"))
        d = d.join(g, on=["nfl_id", "game_key"], how="left")
    else:
        rng = np.random.default_rng(seed)
        d = d.with_columns(pl.Series("half", rng.integers(0, 2, d.height)))
        split = "random"
    rows = []
    for (ch, nps, det), g in d.group_by("channel", "nperseg", "detrend"):
        for f in [c for c in g.columns if c in FEATURES or c.startswith("band_")]:
            h = g.filter(pl.col(f).is_finite()).group_by("nfl_id", "half").agg(pl.col(f).mean().alias("m"), pl.len().alias("n"))
            w = h.pivot(on="half", index="nfl_id", values="m")
            cnt = h.group_by("nfl_id").agg(pl.col("n").min().alias("nmin"))
            w = w.join(cnt, on="nfl_id").filter(pl.col("nmin") >= min_windows // 2)
            if "0" not in w.columns or "1" not in w.columns or w.height < 30:
                continue
            a, b = w["0"].to_numpy().astype(float), w["1"].to_numpy().astype(float)
            ok = np.isfinite(a) & np.isfinite(b)
            r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 30 else NAN
            rows.append({"channel": ch, "nperseg": int(nps), "detrend": det, "feature": f, "split": split, "n_athletes": int(ok.sum()), "r_half": r,
                         "spearman_brown": 2 * r / (1 + r) if np.isfinite(r) and r > -1 else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


def perturbation_sensitivity(bundle: Any, rows: np.ndarray, cfg: Any, configs: Sequence[tuple[str, int, str]], base: Any,
                             noise_sd: Sequence[float] = (0.05, 0.10), smooth: Sequence[int] = (3,), seed: int = 0) -> Any:
    """Spearman between original and perturbed feature values (same sequences) for every M14 perturbation."""
    import polars as pl

    from ..m14.metrics import perturbation, perturbation_grid
    from ..stats import spearman

    out = []
    keys = ["idx", "channel", "nperseg", "detrend"]
    for k, g in enumerate(perturbation_grid(noise_sd, smooth)):
        pt = spectral_table(bundle, rows, cfg, configs, perturb=perturbation(g["noise_sd"], g["smooth"]), seed=seed + 1000 * k)
        j = base.join(pt, on=keys, suffix="_p")
        for (ch, nps, det), s in j.group_by("channel", "nperseg", "detrend"):
            for f in [c for c in base.columns if c in FEATURES or c.startswith("band_")]:
                if f + "_p" not in s.columns:
                    continue
                a, b = s[f].cast(pl.Float64).to_numpy(), s[f + "_p"].cast(pl.Float64).to_numpy()
                ok = np.isfinite(a) & np.isfinite(b)
                out.append({"perturbation": g["name"], "channel": ch, "nperseg": int(nps), "detrend": det, "feature": f, "n": int(ok.sum()),
                            "spearman": spearman(a[ok], b[ok]) if ok.sum() >= 30 else NAN})
    return pl.DataFrame(out, infer_schema_length=None)


def measurement_status(art: Any, rel: Any, sens: Any, cfg: Any) -> Any:
    """Per (configuration, feature): estimable (Combine and NFL), best Combine-drill ICC [CI], min perturbation ρ →
    validated / exploratory / invalid (M14 rule). Missing evidence → ``not_evaluated`` (never validated)."""
    import polars as pl

    keys = ["channel", "nperseg", "detrend", "feature"]
    est = art.group_by(keys).agg(pl.col("estimable").all().alias("estimable_all_domains"), pl.col("rho_len").abs().max().alias("max_abs_rho_len"),
                                 pl.col("pinned_share").max().alias("max_pinned_share"))
    icc = rel.filter((pl.col("domain") == "combine") & pl.col("icc1").is_finite()).sort("icc1", descending=True).group_by(keys, maintain_order=True).agg(
        pl.col("scope").first().alias("best_drill"), pl.col("icc1").first().alias("icc1"), pl.col("icc_lo").first().alias("icc_lo"),
        pl.col("icc_hi").first().alias("icc_hi"), pl.col("n_athletes").first().alias("icc_athletes"))
    sen = sens.group_by(keys).agg(pl.col("spearman").min().alias("min_perturbation_rho")) if sens is not None and sens.height else None
    t = est.join(icc, on=keys, how="left")
    t = t.join(sen, on=keys, how="left") if sen is not None else t.with_columns(pl.lit(None, dtype=pl.Float64).alias("min_perturbation_rho"))
    hi, lo = 0.6, 0.4

    def status(r: dict[str, Any]) -> str:
        icc1, ilo, rho = r.get("icc1"), r.get("icc_lo"), r.get("min_perturbation_rho")
        if not r.get("estimable_all_domains"):
            return "invalid"
        if icc1 is None or not np.isfinite(icc1):
            return "not_evaluated"
        if icc1 < lo:
            return "invalid"
        if icc1 >= hi and ilo is not None and np.isfinite(ilo) and ilo >= lo and rho is not None and np.isfinite(rho) and rho >= 0.8:
            return "validated"
        return "exploratory"

    return t.with_columns(pl.Series("measurement_status", [status(r) for r in t.iter_rows(named=True)], dtype=pl.String)).sort(keys)


# ====================================================================== orchestration (notebook steps)
def run_a3(data: Any, cfg: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    """E15.A3: features (all configurations, cached), surrogates, artifact test, length-matched domain contrast."""
    import polars as pl

    prov = {"bundle_fingerprint": data.fingerprint, "config_fingerprint": cfg.fingerprint(), "spectral": "welch_v1"}
    idx = data.index
    rows = idx["idx"].to_numpy()
    real = run.load_data("features/spectral/welch_all", prov) if cfg.reuse_artifacts else None
    if real is None:
        real = spectral_table(data.bundle, rows, cfg, seed=cfg.seed)
        run.data_file(real, "features/spectral/welch_all", prov)
    else:
        ctx["cache"] = "hit"
    rng = np.random.default_rng(cfg.seed)
    sample = np.sort(np.concatenate([rng.choice(g, min(g.size, 3000), replace=False) for g in
                                     (idx.filter(pl.col("domain") == d)["idx"].to_numpy() for d in ("combine", "nfl")) if g.size]))
    ramp = spectral_table(data.bundle, sample, cfg, transform=ramp_surrogate, seed=cfg.seed + 1)
    noise = spectral_table(data.bundle, sample, cfg, transform=noise_surrogate, seed=cfg.seed + 2)
    dom = dict(zip(idx["idx"].to_list(), idx["domain"].to_list(), strict=True))
    art = artifact_test(real.filter(pl.col("idx").is_in(sample.tolist())), ramp, noise, dom, cfg)
    length = int(max(cfg.welch_nperseg))
    lm = length_matched_spectral(data.bundle, idx, cfg, length, cfg.seed + 3, [(c, length, "linear") for c in CHANNELS])
    contrast = domain_contrast(lm, cfg, cfg.n_boot_screen, cfg.seed)
    for name, df in (("artifact_test", art), ("length_matched_domain_contrast", contrast)):
        ctx["outputs"] += run.table(df, name, "spectral")
    run.data_file(lm, "features/spectral/welch_length_matched", {**prov, "length": length})
    return {"real": real, "artifact": art, "length_matched": lm, "contrast": contrast}


def run_a4(data: Any, real: Any, art: Any, cfg: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    """E15.A4: reliability (Combine drills, NFL kinds), NFL split-half, perturbation sensitivity, measurement status."""
    import polars as pl

    from ..m14.metrics import robustness_rows

    meta = data.index.select("idx", "domain", "kind", "nfl_id", "season")
    tab = real.join(meta, on="idx", how="left")
    rel = reliability(tab, cfg, cfg.n_boot_screen, cfg.seed)
    meta_nfl = data.index.filter(pl.col("domain") == "nfl").select("idx", "season")
    split = "season" if meta_nfl["season"].drop_nulls().n_unique() >= 2 else "random"
    nsplit = nfl_split_reliability(real.join(data.index.select("idx", "nfl_id"), on="idx"), meta_nfl, split, cfg.seed)
    _ids, prow = robustness_rows(data.bundle.meta, int(cfg.extra.get("robustness_athletes", 150)), cfg.seed)
    primary = [(c, int(max(cfg.welch_nperseg)), "linear") for c in CHANNELS]
    base = real.filter(pl.col("idx").is_in(list(prow)) & pl.col("detrend").eq("linear") & pl.col("nperseg").eq(int(max(cfg.welch_nperseg))))
    sens = perturbation_sensitivity(data.bundle, np.asarray(prow, np.int64), cfg, primary, base, seed=cfg.seed)
    status = measurement_status(art, rel, sens, cfg)
    for name, df in (("reliability", rel), ("nfl_split_reliability", nsplit), ("perturbation_sensitivity", sens), ("measurement_status", status)):
        ctx["outputs"] += run.table(df, name, "spectral")
    return {"reliability": rel, "nfl_split": nsplit, "sensitivity": sens, "status": status}


__all__ = ["CHANNELS", "FEATURES", "all_configs", "artifact_test", "cwt_morlet", "domain_contrast", "feature_columns", "length_matched_spectral",
           "measurement_status", "noise_surrogate", "nfl_split_reliability", "perturbation_sensitivity", "ramp_surrogate", "reliability", "run_a3",
           "run_a4", "spectral_features", "spectral_rows", "spectral_table", "stft_mag", "welch_psd"]
