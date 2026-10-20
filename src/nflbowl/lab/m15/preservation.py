"""E15.C1 / E15.C2 — physical information preservation and robustness of the encoders (T08).

Scored on HELD-OUT athletes of each outer fold only (the fold-f model never saw them), on frames that are valid in the
ORIGINAL sequence (yaw only where heading is valid). Ground truth is never a frame the bundle marks invalid.

Per sequence (time-domain models):
    rmse_speed, rmse_atan, rmse_an, rmse_yaw (physical units), mse_scaled (the training-loss scale)
    peak_speed_err, peak_atan_err, min_atan_err (|original − reconstruction| of the extreme, yd/s or yd/s²)
    event_f1, n_events_orig, n_events_recon, brake_onset_err_s, min_time_err_s (M14 detector on both signals; an event
        matches when its speed minimum lies within ``event_match_tol_s``)
    band_err_<k> (relative error of the Welch band-power share of speed, longest gap-free segment)
    deriv_rmse_recon (Δv/Δt vs a_tan consistency of the reconstruction, yd/s²)
STFT models (``domain == 'stft'``) have no time-domain reconstruction: only ``stft_loss`` is reported and every
time-domain column is null (``no_time_domain`` = True) — never imputed.

Robustness (E15.C2): contiguous blocks of 1–3 valid frames are dropped until ``rate`` of the valid frames are gone, or
the sequence is cut to an NFL valid-length quantile; reported: error on the ORIGINAL valid frames, embedding drift
(cosine distance clean ↔ degraded) and self-retrieval top-1 (the degraded embedding's nearest clean embedding is its own).

``paired_vs_reference`` compares every model with PCA at the same (d, seed, fold) on identical sequences: athlete-cluster
bootstrap of the mean per-sequence difference (``stats.cluster_bootstrap``), negative = model better for error metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import training as TR

NAN = float("nan")
ERROR_METRICS = ("rmse_speed", "rmse_atan", "rmse_an", "rmse_yaw", "mse_scaled", "peak_speed_err", "peak_atan_err", "min_atan_err", "brake_onset_err_s",
                 "min_time_err_s", "deriv_rmse_recon")
HIGHER_BETTER = ("event_f1",)


def sample_rows(data: Any, fold: int, n_per_domain: int, seed: int, min_valid: int = 7) -> np.ndarray:
    import polars as pl

    idx = data.index.filter((pl.col("fold") == fold) & (pl.col("n_valid") >= min_valid))
    rng = np.random.default_rng(seed + fold)
    out = []
    for dom in ("combine", "nfl"):
        r = idx.filter(pl.col("domain") == dom)["idx"].to_numpy()
        if r.size:
            out.append(rng.choice(r, min(n_per_domain, r.size), replace=False))
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


def _phys(x6: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
    from ..m14.events import physical

    return physical(x6, m)


def with_recon(x6: np.ndarray, recon4: np.ndarray) -> np.ndarray:
    out = np.array(x6, dtype=np.float32, copy=True)
    out[..., :4] = recon4
    return out


def _match_events(eo: list[dict], er: list[dict], tol_frames: int) -> tuple[int, list[float], list[float]]:
    used, tp, onset, tmin = set(), 0, [], []
    for a in eo:
        best, bj = None, None
        for j, b in enumerate(er):
            if j in used:
                continue
            dmin = abs(a["i_min"] - b["i_min"])
            if dmin <= tol_frames and (best is None or dmin < best):
                best, bj = dmin, j
        if bj is not None:
            used.add(bj)
            tp += 1
            onset.append(abs(a["t_brake"] - er[bj]["t_brake"]))
            tmin.append(abs(a["t_min"] - er[bj]["t_min"]))
    return tp, onset, tmin


def score_rows(x6: np.ndarray, m: np.ndarray, recon4: np.ndarray, cfg: Any) -> list[dict[str, Any]]:
    """Per-sequence preservation metrics (module docstring) for one batch."""
    from ..m14.events import EventParams, detect_sequence
    from .signals import features_from_arrays
    from .spectral import _longest_segment, spectral_features, welch_psd

    P = EventParams(max_gap_dt=cfg.max_gap_dt, min_segment_frames=cfg.min_segment_frames)
    po, pr = _phys(x6, m), _phys(with_recon(x6, recon4), m)
    hv = (po["heading_valid"] == 1.0) & po["valid"]
    xr = with_recon(x6, recon4)
    der = features_from_arrays(xr, m, cfg.max_gap_dt)["deriv_rmse"]
    tol = int(round(cfg.extra.get("event_match_tol_s", 0.3) * cfg.sampling_hz))
    out = []
    for j in range(x6.shape[0]):
        v = po["valid"][j]
        rec: dict[str, Any] = {"no_time_domain": False}
        for ch, key in (("speed", "rmse_speed"), ("a_tan", "rmse_atan"), ("a_n", "rmse_an"), ("yaw", "rmse_yaw")):
            w = hv[j] if ch == "yaw" else v
            e = (pr[ch][j] - po[ch][j])[w]
            rec[key] = float(np.sqrt(np.mean(e ** 2))) if e.size else NAN
        wv = np.stack([v, v, v, hv[j]], -1)
        es = (recon4[j] - x6[j, :, :4]) ** 2
        rec["mse_scaled"] = float(es[wv].mean()) if wv.any() else NAN
        if v.any():
            so, sr = po["speed"][j][v], pr["speed"][j][v]
            ao, ar = po["a_tan"][j][v], pr["a_tan"][j][v]
            rec.update({"peak_speed_err": float(abs(so.max() - sr.max())), "peak_atan_err": float(abs(ao.max() - ar.max())),
                        "min_atan_err": float(abs(ao.min() - ar.min()))})
        row_o = {k: a[j] for k, a in po.items()}
        row_r = {k: a[j] for k, a in pr.items()}
        eo, _, _ = detect_sequence(row_o, P)
        er, _, _ = detect_sequence(row_r, P)
        tp, onset, tmin = _match_events(eo, er, tol)
        rec.update({"n_events_orig": len(eo), "n_events_recon": len(er), "event_f1": (2 * tp / (len(eo) + len(er))) if (eo or er) else NAN,
                    "brake_onset_err_s": float(np.median(onset)) if onset else NAN, "min_time_err_s": float(np.median(tmin)) if tmin else NAN})
        seg = _longest_segment(v, po["dt"][j], cfg.max_gap_dt)
        n = int(max(cfg.welch_nperseg))
        if seg is not None and seg[1] - seg[0] >= n:
            a, b = seg
            fo = welch_psd(po["speed"][j][a:b], cfg.sampling_hz, n, "linear")
            fr = welch_psd(pr["speed"][j][a:b], cfg.sampling_hz, n, "linear")
            if fo is not None and fr is not None:
                so_, sr_ = spectral_features(fo[0], fo[1], cfg.bands_hz), spectral_features(fr[0], fr[1], cfg.bands_hz)
                for k in range(len(cfg.bands_hz)):
                    o = so_[f"band_{k}"]
                    rec[f"band_err_{k}"] = abs(sr_[f"band_{k}"] - o) / o if o and np.isfinite(o) and o > 1e-6 else NAN
        rec["deriv_rmse_recon"] = float(der[j])
        out.append(rec)
    return out


def preservation_for_spec(spec: Any, data: Any, cfg: Any, run: Any, rows: np.ndarray, device: str = "cpu") -> Any:
    """Per-sequence preservation table of one fitted model on held-out ``rows``."""
    import polars as pl

    X, M = TR.arrays(data)
    model = TR.load_model(spec, cfg, run, device)
    out = []
    bs = 512
    for s in range(0, rows.size, bs):
        r = rows[s:s + bs]
        if getattr(model, "domain", "time") == "stft":
            import torch

            with torch.no_grad():
                x, m = torch.from_numpy(X[r]).to(device), torch.from_numpy(M[r]).to(device)
                rec, _ = model(x, m)
                S, cov = model.spectro(x, m)
                w = (cov >= 0.5).float()[:, None, None, :].expand_as(S)
                per = (((rec - S) ** 2) * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1.0)
            out += [{"no_time_domain": True, "stft_loss": float(v)} for v in per.cpu().numpy()]
        else:
            rec = TR.reconstruct_rows(model, X, M, r, device)
            out += score_rows(X[r], M[r], rec, cfg)
    tab = pl.DataFrame(out, infer_schema_length=None)
    meta = data.index.filter(pl.col("idx").is_in(rows.tolist())).sort("idx").select("idx", "nfl_id", "domain", "kind", "n_valid")
    return pl.concat([meta, tab], how="horizontal").with_columns(pl.lit(spec.arch).alias("arch"), pl.lit(spec.d).alias("d"), pl.lit(spec.seed).alias("seed"),
                                                                 pl.lit(spec.fold).alias("fold"))


def summary(tab: Any) -> Any:
    import polars as pl

    num = [c for c in tab.columns if c in (*ERROR_METRICS, *HIGHER_BETTER, "stft_loss") or c.startswith("band_err_")]
    return tab.group_by("arch", "d", "seed", "domain").agg(pl.len().alias("n"), *[pl.col(c).median().alias(c) for c in num]).sort("arch", "d", "seed", "domain")


def paired_vs_reference(tab: Any, n_boot: int, seed: int, reference: str = "pca") -> Any:
    """Per (arch, d, seed, metric): mean per-sequence difference model − reference on the same sequences, athlete-cluster CI."""
    import polars as pl

    from ..stats import cluster_bootstrap

    ref = tab.filter(pl.col("arch") == reference)
    rows = []
    for (arch, d, s), g in tab.filter(pl.col("arch") != reference).group_by("arch", "d", "seed"):
        j = g.join(ref.filter((pl.col("d") == d) & (pl.col("seed") == s)), on=["idx", "fold"], suffix="_ref")
        for k, met in enumerate([c for c in (*ERROR_METRICS, *HIGHER_BETTER) if c in j.columns and f"{c}_ref" in j.columns]):
            diff = (j[met].cast(pl.Float64) - j[f"{met}_ref"].cast(pl.Float64)).to_numpy()
            r = cluster_bootstrap(diff, j["nfl_id"].to_numpy(), np.mean, n_boot, seed + k)
            ref_mean = float(np.nanmean(j[f"{met}_ref"].cast(pl.Float64).to_numpy()))
            rel = float(np.nanmean(diff) / ref_mean) if (met in ERROR_METRICS and ref_mean) else NAN
            rows.append({"arch": arch, "d": d, "seed": s, "metric": met, "reference": reference, "mean_diff": r["estimate"], "lo": r["lo"], "hi": r["hi"],
                         "ref_mean": ref_mean, "relative_diff": rel, "n_sequences": r["n_obs"], "n_athletes": r["n_clusters"],
                         "reading": ("better" if (r["hi"] < 0 if met in ERROR_METRICS else r["lo"] > 0) else
                                     "worse" if (r["lo"] > 0 if met in ERROR_METRICS else r["hi"] < 0) else "inconclusive")})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== E15.C2 robustness
def degrade(m: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Drop contiguous blocks of 1–3 valid frames until ``rate`` of each row's valid frames are removed."""
    out = m.copy()
    for j in range(m.shape[0]):
        v = np.flatnonzero(out[j])
        target = int(round(rate * v.size))
        removed = 0
        guard = 0
        while removed < target and guard < 1000:
            guard += 1
            v = np.flatnonzero(out[j])
            if v.size <= 2:
                break
            k = int(rng.integers(1, 4))
            s = int(rng.choice(v))
            blk = np.arange(s, min(s + k, m.shape[1]))
            blk = blk[out[j, blk]]
            out[j, blk] = False
            removed += blk.size
    return out


def truncate(m: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros_like(m)
    for j in range(m.shape[0]):
        v = np.flatnonzero(m[j])
        out[j, v[:length]] = True
    return out


def robustness(spec: Any, data: Any, cfg: Any, run: Any, rows: np.ndarray, nfl_lengths: np.ndarray, device: str = "cpu", seed: int = 0) -> Any:
    """Error on original valid frames, embedding drift and self-retrieval under dropout rates and NFL-length truncation."""
    import polars as pl

    X, M = TR.arrays(data)
    model = TR.load_model(spec, cfg, run, device)
    x, m = X[rows], M[rows]
    z0 = TR.encode_rows(model, X, M, rows, device)
    conds = [(f"dropout_{r:g}", degrade(m, r, np.random.default_rng(seed + int(100 * r)))) for r in cfg.dropout_rates]
    for q in (0.25, 0.5, 0.75):
        conds.append((f"truncate_q{int(q * 100)}", truncate(m, int(np.quantile(nfl_lengths, q)))))
    out = []
    local = np.arange(rows.size)
    for name, md in conds:
        xd = (x * md[..., None]).astype(np.float32)
        zd = TR.encode_rows(model, xd, md, local, device)
        a = z0 / np.maximum(np.linalg.norm(z0, axis=1, keepdims=True), 1e-12)
        b = zd / np.maximum(np.linalg.norm(zd, axis=1, keepdims=True), 1e-12)
        drift = 1 - (a * b).sum(1)
        top1 = float(np.mean(np.argmax(b @ a.T, axis=1) == np.arange(rows.size)))
        rec = {"arch": spec.arch, "d": spec.d, "seed": spec.seed, "fold": spec.fold, "condition": name, "n": int(rows.size),
               "median_drift": float(np.median(drift)), "self_top1": top1}
        if getattr(model, "domain", "time") == "time":
            rr = TR.reconstruct_rows(model, xd, md, local, device)
            po = _phys(x, m)
            pr = _phys(with_recon(x, rr), m)
            e = (pr["speed"] - po["speed"])[po["valid"]]
            rec["rmse_speed_on_original_valid"] = float(np.sqrt(np.mean(e ** 2))) if e.size else NAN
        out.append(rec)
    return pl.DataFrame(out, infer_schema_length=None)


__all__ = ["ERROR_METRICS", "HIGHER_BETTER", "degrade", "paired_vs_reference", "preservation_for_spec", "robustness", "sample_rows", "score_rows", "summary",
           "truncate", "with_recon"]
