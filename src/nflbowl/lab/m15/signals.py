"""E15.A1 / E15.A2 — time-domain signal analytics on valid frames, and the matched Combine-vs-NFL contrast.

Why (Finding 2, [EXP]): reported Combine / NFL medians (max speed 7.74 vs 5.89 yd/s, braking-frame fraction 0.125 vs
0.345, 62 vs 43 valid frames) mix drill composition, demand selection (NFL ``kind`` = 09A demand rule), window length
(maximum-type statistics grow with observation length) and athlete weighting (NFL athletes contribute many windows).
``domain_contrast_steps`` recomputes each difference under five successively stricter conditions and reports how much
of the raw difference remains:

    raw               sequence-level medians, every sequence counted
    athlete_weighted  median of per-athlete medians (each athlete counts once per domain)
    length_matched    as athlete_weighted, Combine sequences cropped to contiguous gap-free windows whose lengths are
                      drawn from the NFL valid-length distribution (seeded; never across a gap)
    cem               length-matched rows additionally matched by coarsened exact matching on entry speed (mean speed of
                      the first second), valid length and position group (``domain_matching.cem``)
    (within kind)     ``kind_profile`` / ``kind_contrast``: athlete-weighted medians per Combine drill × NFL demand kind

Every statistic is computed on valid frames only (``m14.events.physical``: invalid frames NaN); derivatives use frame
pairs inside one gap-free segment (a frame with dt > ``max_gap_dt`` ends its segment, the M14 rule). Units: yd/s, yd/s²,
yd/s³, s. Intervals: athlete-cluster bootstrap (``stats.cluster_bootstrap_diff``; the two domains are resampled
independently, which is conservative because athletes appear in both).

Definitions fixed here (the brief does not define them): braking frame = a_tan ≤ −``brake_thresh`` (1.0 yd/s², the M14
detector threshold); accelerating frame = a_tan ≥ +``brake_thresh``; entry speed = mean speed over the first
``entry_frames`` (10 = 1 s) valid frames.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np

from ..status import BLOCKED_INSUFFICIENT_SAMPLE, ResearchBlocked
from .data import chunks

NAN = float("nan")
BRAKE_THRESH = 1.0                    # yd/s² (M14 EventParams.brake_thresh)
ENTRY_FRAMES = 10                     # 1 s at 10 Hz
CONTRAST_METRICS = ("max_speed", "q90_speed", "max_atan", "min_atan", "braking_frac", "n_valid")
UNITS = {"max_speed": "yd/s", "q90_speed": "yd/s", "mean_speed": "yd/s", "entry_speed": "yd/s", "max_atan": "yd/s²", "min_atan": "yd/s²",
         "braking_frac": "share of valid frames", "accel_frac": "share of valid frames", "max_abs_an": "yd/s²", "jerk_rms": "yd/s³",
         "deriv_rmse": "yd/s²", "heading_valid_share": "share", "n_valid": "frames", "valid_s": "s"}
STEPS = ("raw", "athlete_weighted", "length_matched", "cem")


# ====================================================================== per-sequence features
def _nan_reduce(fn: Any, a: np.ndarray, **kw: Any) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, **kw)


def features_from_arrays(x: np.ndarray, m: np.ndarray, max_gap_dt: float, brake_thresh: float = BRAKE_THRESH,
                         entry_frames: int = ENTRY_FRAMES) -> dict[str, np.ndarray]:
    """(B, L, 6) scaled channels + (B, L) mask → one value per sequence for every feature in ``UNITS`` (NaN when a
    sequence has too few valid frames for that feature)."""
    from ..m14.events import physical

    p = physical(x, m)
    v, sp, at, an, dt, hv = p["valid"], p["speed"], p["a_tan"], p["a_n"], p["dt"], p["heading_valid"]
    n = v.sum(1).astype(float)
    has = n > 0
    out: dict[str, np.ndarray] = {"n_valid": n, "valid_s": np.where(v, dt, 0.0).sum(1)}
    out["max_speed"] = np.where(has, _nan_reduce(np.nanmax, np.where(has[:, None], sp, 0.0), axis=1), NAN)
    out["q90_speed"] = _nan_reduce(np.nanquantile, sp, q=0.9, axis=1)
    out["mean_speed"] = _nan_reduce(np.nanmean, sp, axis=1)
    out["max_atan"] = np.where(has, _nan_reduce(np.nanmax, np.where(has[:, None], at, 0.0), axis=1), NAN)
    out["min_atan"] = np.where(has, _nan_reduce(np.nanmin, np.where(has[:, None], at, 0.0), axis=1), NAN)
    out["braking_frac"] = np.where(has, (v & (at <= -brake_thresh)).sum(1) / np.maximum(n, 1), NAN)
    out["accel_frac"] = np.where(has, (v & (at >= brake_thresh)).sum(1) / np.maximum(n, 1), NAN)
    out["max_abs_an"] = np.where(has, _nan_reduce(np.nanmax, np.where(v, np.abs(an), -np.inf), axis=1), NAN)
    out["max_abs_an"] = np.where(np.isfinite(out["max_abs_an"]), out["max_abs_an"], NAN)
    out["heading_valid_share"] = np.where(has, np.where(v, hv == 1.0, False).sum(1) / np.maximum(n, 1), NAN)
    order = np.cumsum(v, axis=1)
    first = v & (order <= entry_frames)
    out["entry_speed"] = np.where(first.sum(1) >= min(entry_frames, 5), _nan_reduce(np.nanmean, np.where(first, sp, NAN), axis=1), NAN)
    pair = v[:, 1:] & v[:, :-1] & ~(dt[:, :-1] > max_gap_dt)
    step = np.where(pair, dt[:, :-1], NAN)
    jerk = (at[:, 1:] - at[:, :-1]) / step
    out["jerk_rms"] = np.where(pair.sum(1) >= 3, np.sqrt(_nan_reduce(np.nanmean, np.where(pair, jerk ** 2, NAN), axis=1)), NAN)
    dv = (sp[:, 1:] - sp[:, :-1]) / step
    amid = (at[:, 1:] + at[:, :-1]) / 2
    out["deriv_rmse"] = np.where(pair.sum(1) >= 3, np.sqrt(_nan_reduce(np.nanmean, np.where(pair, (dv - amid) ** 2, NAN), axis=1)), NAN)
    return out


def sequence_features(bundle: Any, rows: np.ndarray, cfg: Any, mask_override: np.ndarray | None = None) -> Any:
    """Per-sequence features for ``rows`` (chunked over the memory map). ``mask_override`` (len(rows), L) replaces the
    bundle mask (used for length-matched crops)."""
    import polars as pl

    rows = np.asarray(rows, np.int64)
    parts: list[dict[str, np.ndarray]] = []
    for s, e in chunks(rows.size, cfg.chunk_rows):
        r = rows[s:e]
        m = np.asarray(bundle.M[r], bool) if mask_override is None else mask_override[s:e]
        parts.append(features_from_arrays(np.asarray(bundle.X[r]), m, cfg.max_gap_dt))
    cols = {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0) for k in UNITS}
    return pl.DataFrame({"idx": rows, **cols})


def feature_table(data: Any, cfg: Any) -> Any:
    """``sequence_features`` for every sequence joined with the sequence index (domain, kind, athlete, position, fold)."""
    feat = sequence_features(data.bundle, data.index["idx"].to_numpy(), cfg)
    return data.index.select("idx", "domain", "kind", "nfl_id", "position_group", "fold", "n_segments").join(
        feat.drop("n_valid", "valid_s").join(data.index.select("idx", "n_valid", "valid_s"), on="idx"), on="idx")


# ====================================================================== E15.A1 validity tables
def validity_tables(feat: Any) -> dict[str, Any]:
    """Validity / gap diagnostics by domain × kind × position (sequence counts, athlete counts, medians, IQRs, shares)."""
    import polars as pl

    agg = [pl.len().alias("n_sequences"), pl.col("nfl_id").n_unique().alias("n_athletes"), pl.col("n_valid").median().alias("median_valid"),
           pl.col("n_valid").quantile(0.25).alias("q25_valid"), pl.col("n_valid").quantile(0.75).alias("q75_valid"),
           (pl.col("n_segments") > 1).mean().alias("share_with_gap"), (pl.col("n_valid") < 20).mean().alias("share_lt_2s"),
           pl.col("heading_valid_share").median().alias("median_heading_valid_share"), pl.col("valid_s").median().alias("median_valid_s")]
    return {"validity_by_domain": feat.group_by("domain").agg(agg).sort("domain"),
            "validity_by_domain_kind": feat.group_by("domain", "kind").agg(agg).sort("domain", "kind"),
            "validity_by_domain_position": feat.group_by("domain", "position_group").agg(agg).sort("domain", "position_group")}


# ====================================================================== length-matched crops (E15.A2)
def crop_masks(bundle: Any, rows: np.ndarray, target_lengths: np.ndarray, cfg: Any, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """For each row, a mask keeping one contiguous gap-free window of a length drawn from ``target_lengths`` (seeded).
    The window lies inside the row's longest gap-free segment; when that segment is shorter than the draw the whole
    segment is kept (and recorded). Returns (masks (len(rows), L), kept lengths)."""
    from ..m14.events import physical, segments

    rng = np.random.default_rng(seed)
    target_lengths = np.asarray(target_lengths, np.int64)
    target_lengths = target_lengths[target_lengths > 0]
    if not target_lengths.size:
        raise ResearchBlocked(BLOCKED_INSUFFICIENT_SAMPLE, "no NFL valid-length distribution to match Combine crops to")
    rows = np.asarray(rows, np.int64)
    L = bundle.X.shape[1]
    out = np.zeros((rows.size, L), bool)
    kept = np.zeros(rows.size, np.int64)
    for s, e in chunks(rows.size, cfg.chunk_rows):
        p = physical(np.asarray(bundle.X[rows[s:e]]), np.asarray(bundle.M[rows[s:e]]))
        for j in range(e - s):
            segs = segments(p["valid"][j], p["dt"][j], cfg.max_gap_dt, 1)
            if not segs:
                continue
            a, b = max(segs, key=lambda t: t[1] - t[0])
            want = int(rng.choice(target_lengths))
            n = min(want, b - a)
            start = a + int(rng.integers(0, b - a - n + 1))
            out[s + j, start:start + n] = True
            kept[s + j] = n
    return out, kept


# ====================================================================== contrasts
def _athlete_values(df: Any, metric: str) -> tuple[np.ndarray, np.ndarray]:
    import polars as pl

    a = df.filter(pl.col(metric).is_not_null() & pl.col(metric).is_finite()).group_by("nfl_id").agg(pl.col(metric).median())
    return a[metric].to_numpy().astype(float), a["nfl_id"].to_numpy()


def _contrast(nfl: Any, comb: Any, metric: str, weighted: bool, n_boot: int, seed: int) -> dict[str, Any]:
    from ..stats import cluster_bootstrap_diff

    if weighted:
        vn, cn = _athlete_values(nfl, metric)
        vc, cc = _athlete_values(comb, metric)
    else:
        vn, cn = nfl[metric].to_numpy().astype(float), nfl["nfl_id"].to_numpy()
        vc, cc = comb[metric].to_numpy().astype(float), comb["nfl_id"].to_numpy()
    r = cluster_bootstrap_diff(vn, cn, vc, cc, np.median, n_boot, seed)
    return {"nfl_median": r["a"], "combine_median": r["b"], "diff_nfl_minus_combine": r["diff"], "lo": r["lo"], "hi": r["hi"],
            "n_nfl": r["n_a"], "n_combine": r["n_b"], "athletes_nfl": r["clusters_a"], "athletes_combine": r["clusters_b"]}


def cem_rows(nfl: Any, comb_crop: Any, k: int, seed: int) -> tuple[Any, Any, dict[str, float]]:
    """CEM on entry speed, valid length (k quantile bins each, pooled edges) and position group. Returns the matched
    NFL rows, matched (cropped) Combine rows, and standardised mean differences before / after."""
    import polars as pl

    from ..domain_matching import cem, smd, strata_codes

    cols = [pl.col("idx").cast(pl.Int64), pl.col("nfl_id").cast(pl.Int64), pl.col("position_group").cast(pl.String),
            pl.col("entry_speed").cast(pl.Float64), pl.col("n_valid").cast(pl.Float64)]
    both = pl.concat([nfl.select(cols).with_columns(pl.lit("nfl").alias("domain")), comb_crop.select(cols).with_columns(pl.lit("combine").alias("domain"))])
    dom = both["domain"].to_numpy()
    cov = {"entry_speed": both["entry_speed"].cast(pl.Float64).fill_null(NAN).to_numpy(), "n_valid": both["n_valid"].cast(pl.Float64).to_numpy()}
    base = strata_codes(cov, ("entry_speed", "n_valid"), k)
    pos = both["position_group"].fill_null("?").to_list()
    pcode = {p: i for i, p in enumerate(sorted(set(pos)))}
    strata = np.where(base < 0, -1, base * (len(pcode) + 1) + np.array([pcode[p] for p in pos]))
    keep = cem(dom, strata, seed)
    kept = both[keep]
    bal = {f"smd_{c}_before": smd(cov[c], dom) for c in cov}
    bal.update({f"smd_{c}_after": smd(cov[c][keep], dom[keep]) for c in cov})
    ids_n = set(kept.filter(pl.col("domain") == "nfl")["idx"].to_list())
    ids_c = set(kept.filter(pl.col("domain") == "combine")["idx"].to_list())
    return nfl.filter(pl.col("idx").is_in(list(ids_n))), comb_crop.filter(pl.col("idx").is_in(list(ids_c))), bal


def domain_contrast_steps(feat: Any, comb_crop: Any, metrics: Sequence[str] = CONTRAST_METRICS, n_boot: int = 500, seed: int = 0,
                          cem_bins: int = 5) -> tuple[Any, dict[str, float]]:
    """One row per (metric, step): NFL − Combine median difference with an athlete-cluster CI and the share of the raw
    difference that remains. ``comb_crop`` = Combine features recomputed on length-matched crops (same ``idx``)."""
    import polars as pl

    nfl = feat.filter(pl.col("domain") == "nfl")
    comb = feat.filter(pl.col("domain") == "combine")
    if nfl.height < 30 or comb.height < 30:
        raise ResearchBlocked(BLOCKED_INSUFFICIENT_SAMPLE, f"too few sequences for a domain contrast (NFL {nfl.height}, Combine {comb.height})")
    nfl_m, comb_m, bal = cem_rows(nfl, comb_crop, cem_bins, seed)
    conds = {"raw": (nfl, comb, False), "athlete_weighted": (nfl, comb, True), "length_matched": (nfl, comb_crop, True), "cem": (nfl_m, comb_m, True)}
    rows = []
    for k, metric in enumerate(metrics):
        raw_diff = NAN
        for j, (step, (a, b, w)) in enumerate(conds.items()):
            r = _contrast(a, b, metric, w, n_boot, seed + 101 * k + j)
            raw_diff = r["diff_nfl_minus_combine"] if step == "raw" else raw_diff
            share = r["diff_nfl_minus_combine"] / raw_diff if (np.isfinite(raw_diff) and raw_diff != 0) else NAN
            rows.append({"metric": metric, "unit": UNITS.get(metric, ""), "step": step, "step_order": j, **r, "share_of_raw_remaining": share})
    return pl.DataFrame(rows, infer_schema_length=None), bal


def kind_profile(feat: Any, metrics: Sequence[str] = CONTRAST_METRICS, min_athletes: int = 10) -> Any:
    """Athlete-weighted medians per (domain, kind) — the 'within kind' view (Combine drills and NFL demands are
    different kinds, so the comparison is a profile, not a matched difference)."""
    import polars as pl

    per = feat.group_by("domain", "kind", "nfl_id").agg([pl.col(m).median() for m in metrics])
    prof = per.group_by("domain", "kind").agg(pl.len().alias("n_athletes"), *[pl.col(m).median() for m in metrics]).sort("domain", "kind")
    return prof.filter(pl.col("n_athletes") >= min_athletes)


def kind_contrast(profile: Any, metric: str) -> Any:
    """Matrix rows: Combine drill × NFL demand kind → NFL median − Combine median of ``metric`` (athlete-weighted)."""
    import polars as pl

    c = profile.filter(pl.col("domain") == "combine").select(pl.col("kind").alias("combine_kind"), pl.col(metric).alias("combine_value"),
                                                               pl.col("n_athletes").alias("n_combine"))
    n = profile.filter(pl.col("domain") == "nfl").select(pl.col("kind").alias("nfl_kind"), pl.col(metric).alias("nfl_value"),
                                                         pl.col("n_athletes").alias("n_nfl"))
    return c.join(n, how="cross").with_columns((pl.col("nfl_value") - pl.col("combine_value")).alias("diff_nfl_minus_combine"),
                                               pl.lit(metric).alias("metric")).sort("combine_kind", "nfl_kind")


def length_matched_combine(data: Any, feat: Any, cfg: Any, seed: int) -> tuple[Any, Any]:
    """Combine features recomputed on crops whose lengths follow the NFL valid-length distribution.
    Returns (crop feature table with the same keys as ``feat``, crop audit: requested vs kept length per row)."""
    import polars as pl

    nfl_len = feat.filter(pl.col("domain") == "nfl")["n_valid"].to_numpy().astype(np.int64)
    comb = feat.filter(pl.col("domain") == "combine")
    rows = comb["idx"].to_numpy()
    masks, kept = crop_masks(data.bundle, rows, nfl_len, cfg, seed)
    cf = sequence_features(data.bundle, rows, cfg, mask_override=masks)
    crop = comb.select("idx", "domain", "kind", "nfl_id", "position_group", "fold").join(cf, on="idx")
    audit = pl.DataFrame({"idx": rows, "original_valid": comb["n_valid"].to_numpy(), "kept_valid": kept})
    return crop, audit


# ====================================================================== orchestration (notebook steps)
def run_a1(data: Any, cfg: Any, run: Any, ctx: dict[str, Any]) -> Any:
    """E15.A1: per-sequence feature table (cached by bundle + config fingerprint) and validity tables."""
    prov = {"bundle_fingerprint": data.fingerprint, "config_fingerprint": cfg.fingerprint(), "features": "signals_v1"}
    feat = run.load_data("features/signal/sequence_time_features", prov) if cfg.reuse_artifacts else None
    if feat is not None:
        ctx["cache"] = "hit"
    else:
        feat = feature_table(data, cfg)
        run.data_file(feat, "features/signal/sequence_time_features", prov)
    for name, df in validity_tables(feat).items():
        ctx["outputs"] += run.table(df, name, "signal")
    return feat


def run_a2(data: Any, feat: Any, cfg: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    """E15.A2: length-matched crops, the four-step contrast, CEM balance and the kind profiles."""
    import polars as pl

    prov = {"bundle_fingerprint": data.fingerprint, "config_fingerprint": cfg.fingerprint(), "features": "signals_v1", "seed": cfg.seed}
    crop = run.load_data("features/signal/combine_length_matched_features", prov) if cfg.reuse_artifacts else None
    if crop is None:
        crop, audit = length_matched_combine(data, feat, cfg, cfg.seed)
        run.data_file(crop, "features/signal/combine_length_matched_features", prov)
        ctx["outputs"] += run.table(audit, "length_crop_audit", "signal")
    steps, bal = domain_contrast_steps(feat, crop, n_boot=cfg.n_boot_screen, seed=cfg.seed)
    prof = kind_profile(feat)
    kc = pl.concat([kind_contrast(prof, m) for m in ("max_speed", "braking_frac", "max_atan")], how="vertical_relaxed")
    ctx["outputs"] += run.table(steps, "domain_contrast_steps", "signal")
    ctx["outputs"] += run.table(pl.DataFrame([{"quantity": k, "value": v} for k, v in bal.items()]), "cem_balance", "signal")
    ctx["outputs"] += run.table(prof, "kind_profile", "signal")
    ctx["outputs"] += run.table(kc, "kind_contrast", "signal")
    return {"steps": steps, "balance": bal, "profile": prof, "kind_contrast": kc, "crop": crop}


__all__ = ["BRAKE_THRESH", "CONTRAST_METRICS", "ENTRY_FRAMES", "STEPS", "UNITS", "cem_rows", "crop_masks", "domain_contrast_steps", "feature_table",
           "features_from_arrays", "kind_contrast", "kind_profile", "length_matched_combine", "run_a1", "run_a2", "sequence_features",
           "validity_tables"]
