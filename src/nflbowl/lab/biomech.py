"""Part H — interpretable movement metrics from the 10 Hz sequence channels, their measurement properties and their
incremental value for the frozen NFL target.

Channels (``m11.representation.CHANNELS`` / ``SCALE``): speed ×10 → yd/s · a_tan ×5 → yd/s² (tangential, signed:
+ accelerating, − braking) · a_n ×5 → yd/s² (normal / centripetal, signed) · yaw ×3 → rad/s (heading rate) ·
dt ×0.1 → s · heading_valid ∈ {0, 1}. Only valid frames (mask M) enter any metric.

Every metric is deterministic (nothing is fitted), so computing it on all athletes leaks nothing across folds. The
``METRICS`` table documents for each metric its expression, physical reading, assumptions and when it is undefined.
Associations reported here are predictive / correlational; none is a validated anatomical trait.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

NAN = float("nan")
VERSION = 2
SPEED, ATAN, AN, YAW, DT, HV = 0, 1, 2, 3, 4, 5
SC = {"speed": 10.0, "a_tan": 5.0, "a_n": 5.0, "yaw": 3.0, "dt": 0.1}
BURST_ACCEL = 3.0           # yd/s²
BURST_FRAMES = 3            # ≥ 0.3 s at 10 Hz
CURVE_MIN_SPEED = 2.0       # yd/s (curvature undefined near standstill)
TURN_MIN_SPEED = 4.0        # yd/s (turning "at speed")
STANDSTILL = 1.0            # yd/s (acceleration-from-rest metrics need a start below this)


@dataclass(frozen=True)
class MetricDef:
    name: str
    unit: str
    expression: str
    interpretation: str
    assumptions: str
    undefined_when: str
    family: str
    higher_is: str


METRICS: tuple[MetricDef, ...] = (
    MetricDef("peak_speed", "yd/s", "max_t v(t)", "top speed reached in the sequence", "tracking speed is smoothed by the provider", "< 3 valid frames",
              "speed", "faster"),
    MetricDef("accel_rate_0_90", "yd/s²", "(0.9·v_max − v_0) / (t_{v≥0.9 v_max} − t_0)", "acceleration efficiency from (near) rest to 90 % of top speed",
              "sequence starts below 1 yd/s and contains the acceleration phase", "start speed ≥ 1 yd/s or the 90 % point is the first frame",
              "acceleration", "quicker build-up"),
    MetricDef("time_to_90", "s", "t_{v≥0.9 v_max} − t_0", "time needed to reach 90 % of the sequence's top speed", "as accel_rate_0_90",
              "as accel_rate_0_90", "acceleration", "slower build-up"),
    MetricDef("peak_accel", "yd/s²", "max_t a_tan(t)", "strongest forward acceleration", "a_tan derived from smoothed speed", "< 3 valid frames",
              "acceleration", "more explosive"),
    MetricDef("peak_decel", "yd/s²", "−min_t a_tan(t)", "strongest braking (positive number)", "as peak_accel", "< 3 valid frames", "braking",
              "harder braking"),
    MetricDef("braking_impulse", "yd/s", "Σ max(−a_tan, 0)·Δt", "speed removed by braking within the sequence", "Δt from the dt channel",
              "< 3 valid frames", "braking", "more braking"),
    MetricDef("max_abs_a_n", "yd/s²", "max_t |a_n(t)|", "largest centripetal (turning) acceleration", "a_n from heading change × speed",
              "< 3 valid frames", "change_of_direction", "sharper turning load"),
    MetricDef("turn_at_speed", "yd/s²", "max |a_n| over frames with v ≥ 4 yd/s", "turning load sustained while moving fast (speed-dependent turning)",
              "heading reliable at speed", "no frame at ≥ 4 yd/s", "change_of_direction", "turns harder at speed"),
    MetricDef("curvature_p90", "1/yd", "q90 of |a_n| / v² over frames with v ≥ 2 yd/s", "path tightness (radius⁻¹) of the sharpest turns",
              "v ≥ 2 yd/s avoids the 1/v² blow-up", "< 5 frames at ≥ 2 yd/s", "change_of_direction", "tighter paths"),
    MetricDef("heading_rate_sd", "rad/s", "SD of yaw over heading-valid frames", "heading variability", "heading valid on ≥ 50 % of frames",
              "heading-valid share < 0.5", "heading", "more variable heading"),
    MetricDef("burst_count", "count", "# runs with a_tan ≥ 3 yd/s² for ≥ 0.3 s", "number of explosive acceleration bursts", "10 Hz frames",
              "< 3 valid frames", "explosiveness", "more bursts"),
    MetricDef("jerk_rms", "yd/s³", "RMS of Δa_tan/Δt over consecutive valid frames", "movement (un)smoothness; lower = smoother",
              "differencing amplifies tracking noise — compare like with like", "< 4 valid frames", "smoothness", "less smooth"),
    MetricDef("turn_asymmetry", "ratio", "(max a_n⁺ − max a_n⁻) / (max a_n⁺ + max a_n⁻)", "left/right turning imbalance within a sequence",
              "SIGN CONVENTION OF a_n NOT VERIFIED across Combine / NFL frames — exploratory only", "no turning in one direction", "asymmetry",
              "imbalance toward + side"),
    MetricDef("duration_s", "s", "Σ Δt", "sequence duration", "—", "never", "context", "longer"),
    MetricDef("heading_valid_share", "share", "mean heading_valid", "heading data quality", "—", "never", "quality", "better quality"),
)
METRIC_NAMES = tuple(m.name for m in METRICS)
ANALYSIS_METRICS = ("peak_speed", "accel_rate_0_90", "time_to_90", "peak_accel", "peak_decel", "braking_impulse", "max_abs_a_n", "turn_at_speed",
                    "curvature_p90", "heading_rate_sd", "burst_count", "jerk_rms", "turn_asymmetry")


def metric_definitions() -> Any:
    import polars as pl

    return pl.DataFrame([asdict(m) for m in METRICS])


# ====================================================================== per-sequence metrics (vectorised, chunked)
def _chunk_metrics(x: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
    B, L = m.shape
    nv = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sp = np.where(m, x[:, :, SPEED] * SC["speed"], np.nan)
        at = np.where(m, x[:, :, ATAN] * SC["a_tan"], np.nan)
        an = np.where(m, x[:, :, AN] * SC["a_n"], np.nan)
        yw = np.where(m, x[:, :, YAW] * SC["yaw"], np.nan)
        dt = np.where(m, x[:, :, DT] * SC["dt"], 0.0)
        hv = np.where(m, x[:, :, HV], np.nan)
        ok3 = nv >= 3
        out: dict[str, np.ndarray] = {}
        out["n_valid"] = nv.astype(np.int64)
        out["duration_s"] = dt.sum(1)
        out["heading_valid_share"] = np.nanmean(hv, 1)
        peak = np.nanmax(sp, 1)
        out["peak_speed"] = np.where(ok3, peak, np.nan)
        out["mean_speed"] = np.where(ok3, np.nanmean(sp, 1), np.nan)
        first = np.argmax(m, 1)
        v0 = sp[np.arange(B), first]
        reach = np.where(np.isfinite(sp), sp >= 0.9 * peak[:, None], False)
        idx90 = np.argmax(reach, 1)
        t = np.cumsum(dt, 1) - dt                                 # time at the start of each frame
        t90 = t[np.arange(B), idx90] - t[np.arange(B), first]
        good = ok3 & (v0 < STANDSTILL) & (idx90 > first) & (t90 > 0)
        out["time_to_90"] = np.where(good, t90, np.nan)
        out["accel_rate_0_90"] = np.where(good, (0.9 * peak - v0) / np.where(t90 > 0, t90, np.nan), np.nan)
        out["peak_accel"] = np.where(ok3, np.nanmax(at, 1), np.nan)
        out["peak_decel"] = np.where(ok3, -np.nanmin(at, 1), np.nan)
        out["mean_a_tan"] = np.where(ok3, np.nanmean(at, 1), np.nan)
        out["braking_impulse"] = np.where(ok3, np.nansum(np.clip(-np.nan_to_num(at), 0, None) * dt, 1), np.nan)
        aan = np.abs(an)
        out["max_abs_a_n"] = np.where(ok3, np.nanmax(aan, 1), np.nan)
        fast = np.isfinite(sp) & (sp >= TURN_MIN_SPEED)
        out["turn_at_speed"] = np.where(fast.any(1), np.nanmax(np.where(fast, aan, np.nan), 1), np.nan)
        mov = np.isfinite(sp) & (sp >= CURVE_MIN_SPEED)
        kap = np.where(mov, aan / np.where(mov, sp, np.nan) ** 2, np.nan)
        out["curvature_p90"] = np.where(mov.sum(1) >= 5, np.nanquantile(np.where(mov.sum(1)[:, None] >= 5, kap, 0.0), 0.9, axis=1), np.nan)
        hvalid = np.isfinite(hv) & (hv > 0.5)
        ysd = np.nanstd(np.where(hvalid, yw, np.nan), 1)
        out["heading_rate_sd"] = np.where(out["heading_valid_share"] >= 0.5, ysd, np.nan)
        b = np.isfinite(at) & (at >= BURST_ACCEL)
        w = b[:, : L - BURST_FRAMES + 1].copy()
        for k in range(1, BURST_FRAMES):
            w &= b[:, k: L - BURST_FRAMES + 1 + k]
        prev = np.zeros_like(w)
        prev[:, 1:] = w[:, :-1]
        out["burst_count"] = np.where(ok3, (w & ~prev).sum(1).astype(float), np.nan)
        pair = np.isfinite(at[:, 1:]) & np.isfinite(at[:, :-1])
        dtm = np.where(pair, 0.5 * (dt[:, 1:] + dt[:, :-1]), np.nan)
        jerk = np.where(pair & (dtm > 0), (at[:, 1:] - at[:, :-1]) / dtm, np.nan)
        out["jerk_rms"] = np.where(nv >= 4, np.sqrt(np.nanmean(jerk ** 2, 1)), np.nan)
        pos_max = np.nanmax(np.where(np.isfinite(an), np.clip(an, 0, None), np.nan), 1)
        neg_max = np.nanmax(np.where(np.isfinite(an), np.clip(-an, 0, None), np.nan), 1)
        den = pos_max + neg_max
        out["turn_asymmetry"] = np.where(ok3 & (pos_max > 0) & (neg_max > 0), (pos_max - neg_max) / np.where(den > 0, den, np.nan), np.nan)
    return out


def sequence_metrics(X: np.ndarray, M: np.ndarray, chunk: int = 4096) -> dict[str, np.ndarray]:
    """{metric: (N,) float array} for every sequence (memory-mapped X / M read in chunks)."""
    import warnings

    N = X.shape[0]
    cols: dict[str, list[np.ndarray]] = {}
    for s in range(0, N, chunk):
        with warnings.catch_warnings():                             # all-NaN rows (undefined metrics) are expected → NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            part = _chunk_metrics(np.asarray(X[s: s + chunk], np.float32), np.asarray(M[s: s + chunk], bool))
        for k, v in part.items():
            cols.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in cols.items()}


def sequence_table(exp: Any, run: Any = None, reuse: bool = True) -> Any:
    """meta (idx, nfl_id, domain, kind, position_group, fold) + derived ids + per-sequence metrics; cached in
    ``datasets/cached/sequence_metrics`` keyed by bundle fingerprint and metric VERSION."""
    import polars as pl

    from .. import _backend

    b = exp.bundle
    prov = {"bundle": b.fingerprint, "version": VERSION, "kind": "sequence_metrics"}
    if run is not None and reuse:
        hit = run.load_data("datasets/cached/sequence_metrics", prov)
        if hit is not None:
            return hit
    mets = sequence_metrics(b.X, b.M)
    d = exp.derived
    fmap = exp.folds_by_athlete
    se = _backend.se()
    cats = sorted(set(se.CATEGORY.values()))
    df = (b.meta.select("idx", "seq_key", "domain", "nfl_id", "kind", "position_group")
          .with_columns(pl.Series("fold", [fmap.get(int(i), -1) for i in b.meta["nfl_id"].to_list()], dtype=pl.Int64),
                        pl.Series("state_id", d.state), pl.Series("category_id", d.cat),
                        pl.Series("category", [cats[c] if c >= 0 else None for c in d.cat.tolist()], dtype=pl.String),
                        *[pl.Series(k, v) for k, v in mets.items()]))
    if run is not None:
        run.data_file(df, "datasets/cached/sequence_metrics", prov)
    return df


# ====================================================================== athlete-level aggregation
COMBINE_GROUPS = {"forty": ("forty",), "agility": ("three_cone", "short_shuttle", "long_shuttle"), "position_drill": ("pass_rush", "gauntlet", "backpedal_transition")}
SPRINT_METRICS = ("peak_speed", "accel_rate_0_90", "time_to_90", "peak_accel", "burst_count", "jerk_rms")
AGILITY_METRICS = ("peak_decel", "braking_impulse", "max_abs_a_n", "turn_at_speed", "curvature_p90", "heading_rate_sd", "jerk_rms", "turn_asymmetry")


def athlete_features(seq: Any) -> Any:
    """One row per athlete: median Combine sprint metrics (forty), median agility metrics (agility drills), NFL medians
    of the same metrics, and within-athlete consistency (SD of forty peak speed across attempts; SD of NFL window peak
    speed). Column names: ``c_forty__<m>``, ``c_agility__<m>``, ``nfl__<m>``, ``c_forty__sd_peak_speed`` …"""
    import polars as pl

    parts = []
    c = seq.filter(pl.col("domain") == "combine")
    for grp, kinds in (("forty", COMBINE_GROUPS["forty"]), ("agility", COMBINE_GROUPS["agility"])):
        mets = SPRINT_METRICS if grp == "forty" else AGILITY_METRICS
        g = c.filter(pl.col("kind").is_in(list(kinds))).group_by("nfl_id").agg(
            *[pl.col(m).median().alias(f"c_{grp}__{m}") for m in mets], pl.len().alias(f"c_{grp}__n_seq"),
            pl.col("peak_speed").std().alias(f"c_{grp}__sd_peak_speed"))
        parts.append(g)
    n = seq.filter(pl.col("domain") == "nfl").group_by("nfl_id").agg(
        *[pl.col(m).median().alias(f"nfl__{m}") for m in ANALYSIS_METRICS], pl.len().alias("nfl__n_seq"), pl.col("peak_speed").std().alias("nfl__sd_peak_speed"))
    parts.append(n)
    base = seq.select("nfl_id", "position_group", "fold").unique(subset=["nfl_id"], keep="first")
    for p in parts:
        base = base.join(p, on="nfl_id", how="left")
    return base.sort("nfl_id")


# ====================================================================== measurement properties
def repeatability(seq: Any, n_boot: int, seed: int) -> Any:
    """Split-half reliability per metric: Combine = first vs second attempt of the same drill family (athletes with ≥ 2
    attempts); NFL = median of odd vs even windows (by store order). Spearman–Brown corrected, athlete bootstrap
    (``ae.split_reliability``)."""
    import polars as pl

    from .. import _backend

    ae = _backend.ae()
    rows = []
    c = seq.filter(pl.col("domain") == "combine").sort("idx")
    for grp, kinds in (("forty", COMBINE_GROUPS["forty"]), ("agility", COMBINE_GROUPS["agility"])):
        mets = SPRINT_METRICS if grp == "forty" else AGILITY_METRICS
        for kind in kinds:                                          # one athlete contributes at most one pair per row
            g = c.filter(pl.col("kind") == kind).with_columns(pl.int_range(pl.len()).over("nfl_id").alias("_k"))
            if not g.height:
                continue
            for mname in mets:
                w = (g.filter(pl.col("_k") < 2).group_by("nfl_id").agg(pl.col(mname).sort_by("_k").alias("_v"))
                     .filter(pl.col("_v").list.len() == 2))
                a = np.array([v[0] for v in w["_v"].to_list()], float) if w.height else np.array([])
                b = np.array([v[1] for v in w["_v"].to_list()], float) if w.height else np.array([])
                r = ae.split_reliability(a, b, n_boot, seed) if a.size else {"n_athletes": 0}
                rows.append({"metric": mname, "domain": "combine", "scope": kind, "design": "attempt 1 vs attempt 2 (same drill)", **r})
    n = seq.filter(pl.col("domain") == "nfl").sort("idx").with_columns(pl.int_range(pl.len()).over("nfl_id").alias("_k"))
    for mname in ANALYSIS_METRICS:
        h = n.group_by("nfl_id", (pl.col("_k") % 2).alias("_h")).agg(pl.col(mname).median().alias("_v"))
        w = h.pivot(on="_h", index="nfl_id", values="_v").drop_nulls()
        cols = [x for x in w.columns if x != "nfl_id"]
        r = ae.split_reliability(w[cols[0]].to_numpy(), w[cols[1]].to_numpy(), n_boot, seed) if len(cols) == 2 else {"n_athletes": 0}
        rows.append({"metric": mname, "domain": "nfl", "scope": "all windows", "design": "odd vs even windows (medians)", **r})
    return pl.DataFrame(rows, infer_schema_length=None)


def cross_domain_stability(ath: Any, n_boot: int, seed: int) -> Any:
    """Spearman between an athlete's Combine metric and the same metric in NFL windows (raw and partial on official
    forty + weight); athlete bootstrap interval of the raw Spearman."""
    import polars as pl

    from .stats import partial_spearman, spearman

    rows = []
    for grp, mets in (("forty", SPRINT_METRICS), ("agility", AGILITY_METRICS)):
        for mname in mets:
            cc, nc = f"c_{grp}__{mname}", f"nfl__{mname}"
            if cc not in ath.columns or nc not in ath.columns:
                continue
            d = ath.select(cc, nc, *[k for k in ("forty", "weight") if k in ath.columns]).drop_nulls(subset=[cc, nc])
            a, b = d[cc].to_numpy().astype(float), d[nc].to_numpy().astype(float)
            ok = np.isfinite(a) & np.isfinite(b)
            r = spearman(a[ok], b[ok]) if ok.sum() >= 10 else NAN
            rng = np.random.default_rng(seed)
            bs = []
            if ok.sum() >= 20:
                aa, bb = a[ok], b[ok]
                for _ in range(n_boot):
                    i = rng.integers(0, aa.size, aa.size)
                    bs.append(spearman(aa[i], bb[i]))
            lo, hi = (float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))) if bs else (NAN, NAN)
            pr = NAN
            if {"forty", "weight"} <= set(d.columns) and ok.sum() >= 20:
                Z = d.select("forty", "weight").to_numpy().astype(float)[ok]
                pr = partial_spearman(a[ok], b[ok], Z)
            rows.append({"metric": mname, "combine_scope": grp, "n_athletes": int(ok.sum()), "spearman": r, "lo": lo, "hi": hi,
                         "partial_spearman_forty_weight": pr})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== incremental value (ladder)
def ladder_frame(exp: Any, ath: Any) -> Any:
    """Transfer analysis frame (same rows as ``Experiment.transfer``: linked athletes with the frozen target and a
    manifest fold) + athlete Combine metrics."""
    import polars as pl

    from ..data.validation import TRANSFER_TARGET

    b = exp.bundle
    if b.athletes is None or TRANSFER_TARGET not in b.athletes.columns:
        return None
    # Combine-side features ONLY: NFL-window metrics (nfl__*) are outcome-side and never enter a predictor frame
    feats = ath.select("nfl_id", *[c for c in ath.columns if c.startswith("c_")])
    return (b.athletes.filter(pl.col("linked").fill_null(False)).filter(pl.col(TRANSFER_TARGET).is_not_null())
            .join(feats, on="nfl_id", how="left").join(b.folds.select("nfl_id", "fold"), on="nfl_id", how="inner").sort("nfl_id"))


def incremental_value(exp: Any, data: Any, metrics: Sequence[str], seed: int, n_boot: int | None = None) -> Any:
    """For each Combine metric column: OOF ladder M1 vs M1 + metric on the transfer rows (fold-fitted design, inner α),
    paired athlete-bootstrap Δ RMSE (positive = metric helps) and BH-adjusted bootstrap shares across the family."""
    import polars as pl

    from dataclasses import replace

    from .. import _backend
    from ..data.validation import TRANSFER_TARGET

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    if n_boot is not None:
        C = replace(C, n_boot=int(n_boot))
    rows = []
    for mcol in metrics:
        if mcol not in data.columns or data[mcol].drop_nulls().len() < 30:
            continue
        mods = {"M1": ae.Model(list(ae.M1_COLS)), f"M1+{mcol}": ae.Model([*ae.M1_COLS, mcol])}
        res = ae.oof_ladder(data, TRANSFER_TARGET, mods, C, seed)
        r = ae.compare(res, [(f"M1+{mcol}", "M1")], C, seed, metric=mcol)
        if r.height:
            cov = float(data[mcol].is_not_null().mean())
            rows.append(r.with_columns(pl.lit(cov).alias("metric_coverage")))
    if not rows:
        return pl.DataFrame()
    out = pl.concat(rows, how="diagonal_relaxed")
    return ae.add_adjusted(out, "p_boot_le0", how="bh")


def hypothesis_tests(data: Any, n_boot: int, seed: int) -> Any:
    """Pre-declared association hypotheses (H-family), each a partial Spearman with the frozen target controlling the
    stated stopwatch / body-size covariates; athlete bootstrap interval; BH across the family. Correlational only."""
    import polars as pl

    from ..data.validation import TRANSFER_TARGET
    from .stats import bh, partial_spearman

    H = [("H1_braking_control", "c_agility__peak_decel", ("forty", "weight"), "Combine braking capacity relates to NFL movement capacity beyond forty + weight"),
         ("H2_curved_acceleration", "c_agility__turn_at_speed", ("forty",), "Turning load at speed adds beyond straight-line speed"),
         ("H3_accel_profile", "c_forty__accel_rate_0_90", ("forty",), "Acceleration build-up differs among athletes with similar forty times and matters"),
         ("H4_explosiveness", "c_forty__peak_accel", ("forty", "ten"), "Peak acceleration adds beyond forty and ten-yard split"),
         ("H5_smoothness", "c_forty__jerk_rms", ("forty",), "Sprint smoothness relates to NFL capacity"),
         ("H6_consistency", "c_forty__sd_peak_speed", ("forty",), "Attempt-to-attempt consistency relates to NFL capacity")]
    rows = []
    for hid, col, covs, text in H:
        if col not in data.columns:
            rows.append({"hypothesis": hid, "statement": text, "metric": col, "status": "metric unavailable"})
            continue
        use = [c for c in covs if c in data.columns]
        d = data.select(col, TRANSFER_TARGET, *use).drop_nulls()
        a, y = d[col].to_numpy().astype(float), d[TRANSFER_TARGET].to_numpy().astype(float)
        Z = d.select(use).to_numpy().astype(float) if use else np.zeros((d.height, 0))
        est = partial_spearman(a, y, Z) if use else NAN
        rng = np.random.default_rng(seed)
        bs = []
        if d.height >= 30:
            for _ in range(n_boot):
                i = rng.integers(0, d.height, d.height)
                bs.append(partial_spearman(a[i], y[i], Z[i]))
        bs = np.asarray(bs, float)
        bs = bs[np.isfinite(bs)]
        lo, hi = (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))) if bs.size >= 20 else (NAN, NAN)
        p2 = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))) if bs.size else NAN
        rows.append({"hypothesis": hid, "statement": text, "metric": col, "controls": ",".join(use), "n_athletes": d.height, "partial_spearman": est,
                     "lo": lo, "hi": hi, "p_boot_two_sided": p2, "status": "evaluated"})
    out = pl.DataFrame(rows, infer_schema_length=None)
    if "p_boot_two_sided" in out.columns:
        p = out["p_boot_two_sided"].to_list()
        out = out.with_columns(pl.Series("p_bh", bh([1.0 if (v is None or not np.isfinite(v)) else v for v in p])))
    return out


def forty_twins_dispersion(data: Any, metric: str = "c_forty__accel_rate_0_90", bins: int = 4) -> Any:
    """'Similar forty, different acceleration profile': within position × forty-quartile cells, the spread (IQR, CV)
    of the acceleration metric — descriptive."""
    import polars as pl

    if metric not in data.columns or "forty" not in data.columns:
        return pl.DataFrame()
    d = data.select("nfl_id", "position_group", "forty", metric).drop_nulls()
    # within-position forty quartile by rank (no categorical window gymnastics)
    d = d.with_columns(((pl.col("forty").rank("ordinal").over("position_group") - 1) * bins // pl.len().over("position_group") + 1)
                       .cast(pl.Int64).alias("_q")).with_columns(("q" + pl.col("_q").cast(pl.String)).alias("forty_bin")).drop("_q")
    return (d.group_by("position_group", "forty_bin").agg(pl.len().alias("n"), pl.col("forty").median().alias("forty_median"),
                                                          pl.col(metric).median().alias("metric_median"),
                                                          (pl.col(metric).quantile(0.75) - pl.col(metric).quantile(0.25)).alias("metric_iqr"),
                                                          (pl.col(metric).std() / pl.col(metric).mean().abs()).alias("metric_cv"))
            .sort("position_group", "forty_bin").filter(pl.col("n") >= 5))


__all__ = ["ANALYSIS_METRICS", "AGILITY_METRICS", "COMBINE_GROUPS", "METRICS", "METRIC_NAMES", "MetricDef", "SPRINT_METRICS", "athlete_features",
           "cross_domain_stability", "forty_twins_dispersion", "hypothesis_tests", "incremental_value", "ladder_frame", "metric_definitions",
           "repeatability", "sequence_metrics", "sequence_table"]
