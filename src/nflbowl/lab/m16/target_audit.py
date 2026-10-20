"""E16.A7 — NFL target laboratory: distributions, variance structure, exposure / context / cohort dependence, stability
and sensitivity of every target, and the selection audit. Descriptive evidence feeding G16-A; no model is fitted here.

Inference unit = athlete. Windows / events of one athlete are never treated as independent: every interval is an athlete
bootstrap (athletes resampled), every association uses one value per athlete, and "within position" means the target is
centred on its position mean first (``_pos_resid``) so that position differences cannot masquerade as athlete signal.

    distributions            per target × position: n, mean, SD, quantiles, skew; KDE grid (Scott bandwidth) for plots
    variance_decomposition   between-position share η² (athlete-bootstrap CI) and within-position SD per position
    position_means           per target × position mean with athlete-bootstrap CI
    relations                within-position Spearman of the target with log exposure, mean window duration, the share of
                             each NFL demand kind in the athlete's windows (max |ρ| reported) and draft year — with CIs
    min_obs_sensitivity      targets rebuilt at each minimum-exposure threshold: coverage and Spearman with the frozen value
    aggregation_sensitivity  alternative statistics (q75 / q90 / q95 / median / mean / max) vs the official definition
    subsample_stability      Spearman between the target on a random fraction of each athlete's observations and the full value
    correlations             raw and within-position Spearman between targets with the common-cohort n of each cell
    selection_bias           linked athletes WITH vs WITHOUT a target value: standardised mean differences of M1 inputs and
                             position composition (who is missing, and is it random?)
    censoring                incomplete / censored shares behind T1, T3, T4 by position
    scorecard                one row per target joining G16-A, measurement reliability and the flags above
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .targets import TARGETS, TargetConfig, event_targets, window_targets

NAN = float("nan")
EXPOSURE = {"T1": "n_windows", "T6": "n_windows", "T2": "n_band_0", "T2b": "n_band_1", "T3": "n_events_t3", "T4": "n_events_t4", "T5": "n_events_t5"}
WINDOW_IDS = ("T1", "T2", "T2b", "T6")
EVENT_IDS = ("T3", "T4", "T5")


def _pos_resid(v: np.ndarray, pos: np.ndarray, min_n: int = 3) -> np.ndarray:
    out = np.full(v.size, NAN)
    for p in np.unique(pos):
        m = (pos == p) & np.isfinite(v)
        if m.sum() >= min_n:
            out[m] = v[m] - v[m].mean()
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size)
    r[o] = np.arange(x.size)
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return NAN
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def _boot_spearman(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float, int]:
    """Spearman with an athlete-bootstrap 95 % CI (one value per athlete in ``a`` / ``b``)."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n < 20:
        return _spearman(a, b), NAN, NAN, n
    rng = np.random.default_rng(seed)
    bs = np.array([_spearman(a[i], b[i]) for i in rng.integers(0, n, (n_boot, n))])
    bs = bs[np.isfinite(bs)]
    return _spearman(a, b), float(np.quantile(bs, 0.025)) if bs.size else NAN, float(np.quantile(bs, 0.975)) if bs.size else NAN, n


def target_columns(ath: Any) -> dict[str, str]:
    return {t: s.column for t, s in TARGETS.items() if s.column in ath.columns}


def long_values(ath: Any) -> Any:
    """(nfl_id, position_group, fold, target_id, unit, value) — finite values only."""
    import polars as pl

    parts = []
    for tid, col in target_columns(ath).items():
        parts.append(ath.select(pl.col("nfl_id").cast(pl.Int64), "position_group", *(["fold"] if "fold" in ath.columns else []),
                                pl.lit(tid).alias("target_id"), pl.lit(TARGETS[tid].unit).alias("unit"), pl.col(col).cast(pl.Float64).alias("value"))
                     .filter(pl.col("value").is_finite()))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


# ====================================================================== distributions
def distributions(long: Any) -> Any:
    import polars as pl

    agg = [pl.len().alias("n_athletes"), pl.col("value").mean().alias("mean"), pl.col("value").std().alias("sd"), pl.col("value").median().alias("median"),
           pl.col("value").quantile(0.05).alias("q05"), pl.col("value").quantile(0.25).alias("q25"), pl.col("value").quantile(0.75).alias("q75"),
           pl.col("value").quantile(0.95).alias("q95"), pl.col("value").skew().alias("skew"), pl.col("unit").first()]
    by = long.group_by("target_id", "position_group").agg(agg)
    allp = long.group_by("target_id").agg(agg).with_columns(pl.lit("ALL").alias("position_group"))
    return pl.concat([allp, by], how="diagonal_relaxed").sort("target_id", "position_group")


def kde_grid(long: Any, target_id: str, n_grid: int = 200) -> Any:
    """Gaussian KDE (Scott bandwidth) of the athlete values of one target, overall and per position (plot data)."""
    import polars as pl

    d = long.filter(pl.col("target_id") == target_id)
    v = d["value"].to_numpy()
    if v.size < 10:
        return pl.DataFrame()
    lo, hi = np.quantile(v, [0.001, 0.999])
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    out = []
    for p in ["ALL", *sorted(set(d["position_group"].drop_nulls().to_list()))]:
        x = v if p == "ALL" else d.filter(pl.col("position_group") == p)["value"].to_numpy()
        if x.size < 10:
            continue
        bw = 1.06 * np.std(x, ddof=1) * x.size ** (-1 / 5) if np.std(x) > 0 else 1.0
        dens = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(1) / (x.size * bw * np.sqrt(2 * np.pi))
        out.append(pl.DataFrame({"target_id": target_id, "position_group": p, "x": grid, "density": dens, "n": x.size, "bandwidth": bw}))
    return pl.concat(out) if out else pl.DataFrame()


def variance_decomposition(long: Any, n_boot: int, seed: int) -> Any:
    """η² = between-position SS / total SS with an athlete-bootstrap CI; within-position SD per position."""
    import polars as pl

    rows = []
    rng = np.random.default_rng(seed)
    for (tid,), g in long.group_by("target_id"):
        v, p = g["value"].to_numpy(), g["position_group"].fill_null("?").to_numpy()
        if v.size < 20:
            continue

        def eta2(vv: np.ndarray, pp: np.ndarray) -> float:
            mu = vv.mean()
            sst = float(((vv - mu) ** 2).sum())
            ssb = float(sum((pp == k).sum() * (vv[pp == k].mean() - mu) ** 2 for k in np.unique(pp)))
            return ssb / sst if sst > 0 else NAN

        bs = np.array([eta2(v[i], p[i]) for i in rng.integers(0, v.size, (n_boot, v.size))])
        sd_w = {f"sd_within_{k}": float(np.std(v[p == k], ddof=1)) for k in np.unique(p) if (p == k).sum() >= 3}
        resid = _pos_resid(v, p)
        rows.append({"target_id": tid, "n_athletes": int(v.size), "total_sd": float(np.std(v, ddof=1)), "eta2_position": eta2(v, p),
                     "eta2_lo": float(np.nanquantile(bs, 0.025)), "eta2_hi": float(np.nanquantile(bs, 0.975)),
                     "pooled_within_sd": float(np.nanstd(resid, ddof=1)), "within_share": 1 - eta2(v, p), **sd_w})
    return pl.DataFrame(rows, infer_schema_length=None).sort("target_id") if rows else pl.DataFrame()


def position_means(long: Any, n_boot: int, seed: int) -> Any:
    import polars as pl

    from ..stats import athlete_bootstrap_ci

    rows = []
    for (tid, p), g in long.group_by("target_id", "position_group"):
        est, lo, hi = athlete_bootstrap_ci(g["value"].to_numpy(), n_boot, seed)
        rows.append({"target_id": tid, "position_group": p, "n_athletes": g.height, "mean": est, "lo": lo, "hi": hi, "small_sample": g.height < 30})
    return pl.DataFrame(rows, infer_schema_length=None).sort("target_id", "position_group")


# ====================================================================== relations
def athlete_context(index: Any, kinds: Sequence[str] | None = None) -> Any:
    """Per athlete (NFL windows): mean valid duration (s) and the share of windows of each demand kind."""
    import polars as pl

    nfl = index.filter(pl.col("domain") == "nfl")
    ks = list(kinds) if kinds else sorted(set(nfl["kind"].drop_nulls().to_list()))
    return nfl.group_by("nfl_id").agg(pl.col("valid_s").mean().alias("mean_window_s"), pl.len().alias("n_nfl_windows_all"),
                                      *[(pl.col("kind") == k).mean().alias(f"share_kind__{k}") for k in ks]).with_columns(pl.col("nfl_id").cast(pl.Int64))


def relations(ath: Any, ctx: Any, athletes: Any, n_boot: int, seed: int) -> Any:
    """Within-position Spearman (athlete bootstrap) of each target with exposure, duration, kind shares and draft year."""
    import polars as pl

    d = ath.with_columns(pl.col("nfl_id").cast(pl.Int64)).join(ctx, on="nfl_id", how="left")
    if athletes is not None and "draft_year" in athletes.columns:
        d = d.join(athletes.select(pl.col("nfl_id").cast(pl.Int64), pl.col("draft_year").cast(pl.Float64)), on="nfl_id", how="left")
    pos = d["position_group"].fill_null("?").to_numpy()
    kind_cols = [c for c in d.columns if c.startswith("share_kind__")]
    rows = []
    for k, (tid, col) in enumerate(target_columns(ath).items()):
        y = _pos_resid(d[col].cast(pl.Float64).fill_null(NAN).to_numpy(), pos)
        covs = {"log_exposure": np.log(np.maximum(d[EXPOSURE[tid]].cast(pl.Float64).fill_null(NAN).to_numpy(), 1e-9)) if tid in EXPOSURE and EXPOSURE[tid] in d.columns else None,
                "log_all_windows": np.log(np.maximum(d["n_nfl_windows_all"].cast(pl.Float64).fill_null(NAN).to_numpy(), 1e-9)) if "n_nfl_windows_all" in d.columns else None,
                "mean_window_s": d["mean_window_s"].cast(pl.Float64).fill_null(NAN).to_numpy() if "mean_window_s" in d.columns else None,
                "draft_year": d["draft_year"].fill_null(NAN).to_numpy() if "draft_year" in d.columns else None}
        for j, (name, x) in enumerate(covs.items()):
            if x is None:
                continue
            r, lo, hi, n = _boot_spearman(y, _pos_resid(x, pos), n_boot, seed + 101 * k + j)
            rows.append({"target_id": tid, "covariate": name, "n": n, "spearman_within_position": r, "lo": lo, "hi": hi,
                         "flag": bool(np.isfinite(r) and abs(r) > 0.3)})
        ks = [(c, _spearman(y, _pos_resid(d[c].cast(pl.Float64).fill_null(NAN).to_numpy(), pos))) for c in kind_cols]
        ks = [(c, r) for c, r in ks if np.isfinite(r)]
        if ks:
            c, r = max(ks, key=lambda t: abs(t[1]))
            rr, lo, hi, n = _boot_spearman(y, _pos_resid(d[c].cast(pl.Float64).fill_null(NAN).to_numpy(), pos), n_boot, seed + 101 * k + 50)
            rows.append({"target_id": tid, "covariate": f"max |ρ| kind share ({c.removeprefix('share_kind__')})", "n": n, "spearman_within_position": rr,
                         "lo": lo, "hi": hi, "flag": bool(np.isfinite(rr) and abs(rr) > 0.3), "note": f"selected from {len(ks)} kinds (max |ρ|: optimistic)"})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== sensitivity and stability
def min_obs_sensitivity(windows: Any, events: Any, base: Any, tcfg: TargetConfig, windows_grid: Sequence[int], events_grid: Sequence[int]) -> Any:
    """Targets rebuilt at each threshold (other settings unchanged): coverage and Spearman with the frozen values."""
    import polars as pl

    rows = []
    ref = base.select(pl.col("nfl_id").cast(pl.Int64), *[pl.col(c) for c in target_columns(base).values()])
    for k in windows_grid:
        c = TargetConfig(min_windows=int(k), min_windows_kind=max(2, min(tcfg.min_windows_kind, int(k))), min_events=tcfg.min_events,
                         band_min_frames=tcfg.band_min_frames, bands=tcfg.bands, q=tcfg.q)
        t = window_targets(windows, c).with_columns(pl.col("nfl_id").cast(pl.Int64))
        rows += _compare(t, ref, WINDOW_IDS, "min_windows", int(k))
    for k in events_grid:
        c = TargetConfig(min_windows=tcfg.min_windows, min_windows_kind=tcfg.min_windows_kind, min_events=int(k), band_min_frames=tcfg.band_min_frames,
                         bands=tcfg.bands, q=tcfg.q)
        t = window_targets(windows, c).with_columns(pl.col("nfl_id").cast(pl.Int64))
        if events is not None and events.height:
            e = event_targets(events, c)
            if e.width > 1:
                t = t.join(e.with_columns(pl.col("nfl_id").cast(pl.Int64)), on="nfl_id", how="full", coalesce=True)
        rows += _compare(t, ref, ("T2", "T2b", *EVENT_IDS), "min_events", int(k))
    return pl.DataFrame(rows, infer_schema_length=None)


def _compare(t: Any, ref: Any, ids: Sequence[str], param: str, value: int) -> list[dict[str, Any]]:
    import polars as pl

    out = []
    for tid in ids:
        col = TARGETS[tid].column
        if col not in t.columns or col not in ref.columns:
            continue
        j = t.select("nfl_id", pl.col(col).alias("a")).join(ref.select("nfl_id", pl.col(col).alias("b")), on="nfl_id", how="inner")
        a, b = j["a"].cast(pl.Float64).fill_null(NAN).to_numpy(), j["b"].cast(pl.Float64).fill_null(NAN).to_numpy()
        out.append({"target_id": tid, "parameter": param, "threshold": value, "n_athletes": int(np.isfinite(a).sum()), "n_common": int((np.isfinite(a) & np.isfinite(b)).sum()),
                    "spearman_vs_frozen": _spearman(a, b)})
    return out


def _athlete_arrays(windows: Any, events: Any) -> dict[str, dict[int, np.ndarray]]:
    """{target: {athlete: underlying observations}} for the window- and event-quantity targets."""
    import polars as pl

    from .target_registry import _source_frame

    out = {}
    for tid in ("T1", "T2", "T2b", "T3", "T4", "T5", "T6"):
        d = _source_frame(tid, windows, events)
        if d is None or not d.height:
            continue
        out[tid] = {int(k[0]): g["v"].to_numpy() for k, g in d.group_by("nfl_id")}
    return out


ALT_STATS = {"q75": lambda v: np.quantile(v, 0.75), "q90": lambda v: np.quantile(v, 0.9), "q95": lambda v: np.quantile(v, 0.95), "median": np.median,
             "mean": np.mean, "max": np.max}


def aggregation_sensitivity(windows: Any, events: Any, base: Any, positions: Any, min_obs: int = 5) -> Any:
    """Raw and within-position Spearman between each alternative athlete statistic and the official target value."""
    import polars as pl

    arrs = _athlete_arrays(windows, events)
    pos = dict(zip(positions["nfl_id"].cast(pl.Int64).to_list(), positions["position_group"].fill_null("?").to_list(), strict=True))
    ref = {tid: dict(zip(base["nfl_id"].cast(pl.Int64).to_list(), base[TARGETS[tid].column].cast(pl.Float64).fill_null(NAN).to_list(), strict=True))
           for tid in arrs if TARGETS[tid].column in base.columns}
    rows = []
    for tid, per in arrs.items():
        if tid not in ref or tid == "T6":
            continue
        ids = [i for i, v in per.items() if v.size >= min_obs and np.isfinite(ref[tid].get(i, NAN))]
        if len(ids) < 20:
            continue
        b = np.array([ref[tid][i] for i in ids])
        p = np.array([pos.get(i, "?") for i in ids])
        for name, f in ALT_STATS.items():
            a = np.array([float(f(per[i])) for i in ids])
            rows.append({"target_id": tid, "statistic": name, "n_athletes": len(ids), "spearman": _spearman(a, b),
                         "spearman_within_position": _spearman(_pos_resid(a, p), _pos_resid(b, p))})
    return pl.DataFrame(rows, infer_schema_length=None)


def subsample_stability(windows: Any, events: Any, fractions: Sequence[float], reps: int, seed: int, min_obs: int = 6) -> Any:
    """Per target × fraction: mean and 2.5 / 97.5 % (over replicates) Spearman between the statistic on a random fraction
    of each athlete's observations and on all of them (athletes with ≥ ``min_obs`` observations)."""
    import polars as pl

    from .target_registry import STAT

    arrs = _athlete_arrays(windows, events)
    rng = np.random.default_rng(seed)
    rows = []
    for tid, per in arrs.items():
        stat = STAT[tid][3]
        ids = [i for i, v in per.items() if v.size >= min_obs]
        if len(ids) < 20:
            continue
        full = np.array([float(stat(per[i][None, :])[0]) for i in ids])
        for f in fractions:
            rs = []
            for _ in range(reps):
                sub = []
                for i in ids:
                    v = per[i]
                    k = max(2 if tid != "T6" else 3, int(round(f * v.size)))
                    sub.append(float(stat(v[rng.permutation(v.size)[:k]][None, :])[0]))
                rs.append(_spearman(np.asarray(sub), full))
            rs = np.asarray([r for r in rs if np.isfinite(r)])
            rows.append({"target_id": tid, "fraction": float(f), "n_athletes": len(ids), "reps": int(rs.size),
                         "spearman_mean": float(rs.mean()) if rs.size else NAN, "spearman_lo": float(np.quantile(rs, 0.025)) if rs.size else NAN,
                         "spearman_hi": float(np.quantile(rs, 0.975)) if rs.size else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== correlations, selection, censoring
def correlations(ath: Any) -> Any:
    """Long matrix: raw and within-position Spearman between every pair of targets, common-cohort n per cell."""
    import polars as pl

    cols = target_columns(ath)
    pos = ath["position_group"].fill_null("?").to_numpy()
    vals = {t: ath[c].cast(pl.Float64).fill_null(NAN).to_numpy() for t, c in cols.items()}
    rows = []
    for a in cols:
        for b in cols:
            ok = np.isfinite(vals[a]) & np.isfinite(vals[b])
            n = int(ok.sum())
            rows.append({"target_a": a, "target_b": b, "n_common": n, "spearman": _spearman(vals[a], vals[b]) if n >= 20 else NAN,
                         "spearman_within_position": _spearman(_pos_resid(vals[a], pos), _pos_resid(vals[b], pos)) if n >= 20 else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


def selection_bias(ath: Any, athletes: Any, m1_cols: Sequence[str]) -> Any:
    """Among LINKED athletes: with vs without each target — SMD of M1 inputs and the position share difference."""
    import polars as pl

    if athletes is None:
        return pl.DataFrame()
    a = athletes.filter(pl.col("linked").fill_null(False)) if "linked" in athletes.columns else athletes
    a = a.select(pl.col("nfl_id").cast(pl.Int64), "position_group", *[pl.col(c).cast(pl.Float64) for c in m1_cols if c in athletes.columns])
    d = a.join(ath.select(pl.col("nfl_id").cast(pl.Int64), *target_columns(ath).values()), on="nfl_id", how="left")
    rows = []
    for tid, col in target_columns(ath).items():
        has = d[col].cast(pl.Float64).fill_null(NAN).is_finite().to_numpy() if col in d.columns else np.zeros(d.height, bool)
        rec = {"target_id": tid, "n_linked": d.height, "n_with": int(has.sum()), "coverage": float(has.mean()) if d.height else NAN}
        smds = {}
        for c in m1_cols:
            if c not in d.columns:
                continue
            x = d[c].fill_null(NAN).to_numpy()
            x1, x0 = x[has & np.isfinite(x)], x[~has & np.isfinite(x)]
            if x1.size >= 5 and x0.size >= 5:
                s = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
                smds[c] = float((x1.mean() - x0.mean()) / s) if s > 0 else NAN
                rec[f"smd_{c}"] = smds[c]
        pos = d["position_group"].fill_null("?").to_numpy()
        for p in np.unique(pos):
            rec[f"coverage_{p}"] = float(has[pos == p].mean()) if (pos == p).any() else NAN
        rec["max_abs_smd"] = float(np.nanmax(np.abs(list(smds.values())))) if smds else NAN
        rec["flag"] = bool(np.isfinite(rec["max_abs_smd"]) and rec["max_abs_smd"] > 0.25)
        rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None)


def censoring(ath: Any) -> Any:
    """Shares behind truncation rules, by position (athlete medians)."""
    import polars as pl

    cols = [c for c in ("share_windows_end_at_max", "share_incomplete", "share_propulsion_censored") if c in ath.columns]
    if not cols:
        return pl.DataFrame()
    agg = [pl.len().alias("n_athletes"), *[pl.col(c).median().alias(f"median_{c}") for c in cols], *[pl.col(c).quantile(0.9).alias(f"q90_{c}") for c in cols]]
    allp = ath.select(*agg).with_columns(pl.lit("ALL").alias("position_group"))
    return pl.concat([allp, ath.group_by("position_group").agg(agg)], how="diagonal_relaxed").sort("position_group")


def scorecard(gate_a: Any, coverage: Any, rel: Any, meas: Any, rel_tab: Any, minobs: Any, aggsens: Any, sel: Any, cens: Any, max_censored: float) -> Any:
    """One row per target: what supports or undermines its validity (descriptive; the decision is G16-A)."""
    import polars as pl

    def pick(df: Any, tid: str, col: str, how: str = "first") -> Any:
        if df is None or not df.height or col not in df.columns or "target_id" not in df.columns:
            return None
        s = df.filter(pl.col("target_id") == tid)[col].drop_nulls()
        if not s.len():
            return None
        return float(s.min()) if how == "min" else (float(s.max()) if how == "max" else s[0])

    cen = cens.filter(pl.col("position_group") == "ALL") if cens is not None and cens.height else None
    rows = []
    for tid in [t for t in TARGETS if t not in ("T0x",)]:
        cov_all = coverage.filter((pl.col("target_id") == tid) & (pl.col("position_group") == "ALL")) if coverage is not None and coverage.height else None
        flags = []
        rr = rel.filter(pl.col("target_id") == tid)["spearman_within_position"].drop_nulls().to_numpy() if rel is not None and rel.height else np.zeros(0)
        rr = rr[np.isfinite(rr)]
        rho = float(np.abs(rr).max()) if rr.size else None
        expo =rel.filter((pl.col("target_id") == tid) & (pl.col("covariate") == "log_exposure")) if rel is not None and rel.height else None
        rho_e = float(expo["spearman_within_position"][0]) if expo is not None and expo.height else None
        if rho_e is not None and np.isfinite(rho_e) and abs(rho_e) > 0.3:
            flags.append("exposure-dependent")
        mo = pick(minobs, tid, "spearman_vs_frozen", "min")
        if mo is not None and np.isfinite(mo) and mo < 0.8:
            flags.append("threshold-sensitive")
        ag = pick(aggsens.filter(pl.col("statistic") != "max") if aggsens is not None and aggsens.height else aggsens, tid, "spearman_within_position", "min")
        if ag is not None and np.isfinite(ag) and ag < 0.7:
            flags.append("aggregation-sensitive")
        sb = pick(sel, tid, "max_abs_smd")
        if sb is not None and np.isfinite(sb) and sb > 0.25:
            flags.append("selective coverage")
        cshare = None
        if cen is not None and cen.height:
            key = {"T3": "median_share_incomplete", "T4": "median_share_propulsion_censored", "T1": "median_share_windows_end_at_max"}.get(tid)
            if key and key in cen.columns:
                cshare = float(cen[key][0]) if cen[key][0] is not None else None
                if cshare is not None and cshare > max_censored:
                    flags.append("heavily censored")
        rows.append({"target_id": tid, "unit": TARGETS[tid].unit, "n_athletes": int(cov_all["n_athletes"][0]) if cov_all is not None and cov_all.height else None,
                     "g16a_status": pick(gate_a, tid, "status"), "split_half_reliability": pick(rel_tab, tid, "reliability"),
                     "split_half_lo": pick(rel_tab, tid, "rel_lo"), "split": pick(rel_tab, tid, "split"),
                     "measurement_reliability": pick(meas, tid, "measurement_reliability"), "rho_log_exposure": rho_e, "max_abs_relation": rho,
                     "min_spearman_threshold_grid": mo, "min_within_position_spearman_aggregation": ag, "max_abs_smd_selection": sb,
                     "censored_share": cshare, "flags": "; ".join(flags) if flags else "none"})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["ALT_STATS", "EXPOSURE", "aggregation_sensitivity", "athlete_context", "censoring", "correlations", "distributions", "kde_grid", "long_values",
           "min_obs_sensitivity", "position_means", "relations", "scorecard", "selection_bias", "subsample_stability", "target_columns",
           "variance_decomposition"]
