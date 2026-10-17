"""Phase 2 — cross-sport metrics: sequence-level sprint / envelope / smoothness metrics, the one-pass extraction driver
(events + sequence records), the per-sequence table, athlete aggregation and perturbation re-extraction.

Aggregation rules (declared, nothing fitted):
* event metric → sequence: median over the sequence's events (``max_controlled_decel``: max over complete events);
* sequence → athlete: median over the athlete's sequences of a scope (``registry.SCOPES``) → ``m14c_<scope>__<metric>``;
* NFL windows → ``m14n__<metric>`` (OUTCOME side only: never a predictor);
* ``directional_asymmetry`` / ``cut_recovery_asymmetry``: athlete level, from agility events with a turn direction.
Context columns (``m14c_<scope>__ctx_*``) are counts / entry speeds, never predictors on their own.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from . import registry as RG
from .events import EVENT_CONTEXT_COLUMNS, EVENT_FLAG_COLUMNS, EVENT_METRIC_COLUMNS, EventParams, detect_sequence, physical, seg_time

NAN = float("nan")
STANDSTILL = 1.0
ENVELOPE_BINS = (2.0, 4.0, 6.0, 8.0, np.inf)
SEQ_METRIC_COLUMNS = ("accel_envelope_a0", "accel_envelope_v0", "accel_reserve_at_vref", "time_to_useful_speed", "high_speed_accel",
                      "curvature_speed_envelope", "sparc_smoothness")
MAX_AGG = ("max_controlled_decel",)


# ====================================================================== sequence-level metrics
def sprint_profile(v: np.ndarray, at: np.ndarray, dt: np.ndarray, P: EventParams) -> dict[str, float]:
    """Acceleration–speed profile of one gap-free segment (the sprint start segment)."""
    out = {"accel_envelope_a0": NAN, "accel_envelope_v0": NAN, "accel_profile_slope": NAN, "accel_profile_r2": NAN, "accel_profile_n": 0,
           "accel_reserve_at_vref": NAN, "high_speed_accel": NAN, "time_to_useful_speed": NAN}
    n = v.size
    if n < 5:
        return out
    t = seg_time(dt)
    pk = int(np.argmax(v))
    vpk = float(v[pk])
    ph = np.arange(pk + 1)
    sel = ph[(v[ph] >= STANDSTILL) & (at[ph] > 0)]
    out["accel_profile_n"] = int(sel.size)
    if sel.size >= 5 and float(np.std(v[sel])) > 1e-6:
        X, Y = v[sel], at[sel]
        s, a0 = np.polyfit(X, Y, 1)
        pred = a0 + s * X
        ss = float(((Y - Y.mean()) ** 2).sum())
        out["accel_profile_slope"] = float(s)
        out["accel_profile_r2"] = 1 - float(((Y - pred) ** 2).sum()) / ss if ss > 0 else NAN
        if s < 0 and a0 > 0:
            out["accel_envelope_a0"] = float(a0)
            out["accel_envelope_v0"] = float(-a0 / s)
            if vpk >= P.v_ref:
                out["accel_reserve_at_vref"] = float(a0 + s * P.v_ref)
    hs = ph[(v[ph] >= 0.8 * vpk) & (ph < pk)]
    if hs.size >= 2:
        out["high_speed_accel"] = float(np.mean(at[hs]))
    if v[0] < STANDSTILL:
        k = next((j for j in range(1, n) if v[j] >= P.useful_speed), None)
        if k is not None:
            a, b = v[k - 1], v[k]
            f = (P.useful_speed - a) / (b - a) if b != a else 1.0
            out["time_to_useful_speed"] = float(t[k - 1] + min(1.0, max(0.0, f)) * (t[k] - t[k - 1]) - t[0])
    return out


def curvature_speed_envelope(v: np.ndarray, an: np.ndarray, min_frames: int = 3) -> dict[str, float]:
    """q90 |a_n| per speed bin and their mean (≥ 2 populated bins)."""
    v, a = np.asarray(v, float), np.abs(np.asarray(an, float))
    ok = np.isfinite(v) & np.isfinite(a)
    out: dict[str, float] = {}
    vals = []
    for lo, hi in zip(ENVELOPE_BINS[:-1], ENVELOPE_BINS[1:]):
        m = ok & (v >= lo) & (v < hi)
        name = f"envelope_q90_{int(lo)}_{'inf' if not np.isfinite(hi) else int(hi)}"
        q = float(np.quantile(a[m], 0.9)) if m.sum() >= min_frames else NAN
        out[name] = q
        if np.isfinite(q):
            vals.append(q)
    out["curvature_speed_envelope"] = float(np.mean(vals)) if len(vals) >= 2 else NAN
    return out


def sparc(v: np.ndarray, fs: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length (Balasubramanian et al. 2015) of a speed profile; the cut-off is capped at Nyquist."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 10 or not np.isfinite(fs) or fs <= 0:
        return NAN
    nfft = int(2 ** (np.ceil(np.log2(v.size)) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(v, nfft))
    if Mf.max() <= 0:
        return NAN
    Mf = Mf / Mf.max()
    fc = min(fc, fs / 2.0)
    sel = f <= fc
    fs_, Ms = f[sel], Mf[sel]
    inx = np.flatnonzero(Ms >= amp_th)
    if inx.size < 2:
        return NAN
    fs_, Ms = fs_[inx[0]: inx[-1] + 1], Ms[inx[0]: inx[-1] + 1]
    span = fs_[-1] - fs_[0]
    if span <= 0:
        return NAN
    return float(-np.sum(np.sqrt((np.diff(fs_) / span) ** 2 + np.diff(Ms) ** 2)))


def sequence_record(arrs: dict[str, np.ndarray], segs: list[tuple[int, int]], P: EventParams) -> dict[str, float]:
    """Sequence-level metrics from 1-D physical arrays: sprint profile on the FIRST segment, envelope over every valid
    frame, SPARC on the longest segment."""
    rec: dict[str, Any] = {}
    if segs:
        s, e = segs[0]
        rec.update(sprint_profile(arrs["speed"][s:e], arrs["a_tan"][s:e], arrs["dt"][s:e], P))
        s2, e2 = max(segs, key=lambda se: se[1] - se[0])
        dts = arrs["dt"][s2:e2]
        rec["sparc_smoothness"] = sparc(arrs["speed"][s2:e2], 1.0 / float(np.median(dts)) if dts.size else NAN)
    else:
        rec.update(sprint_profile(np.array([]), np.array([]), np.array([]), P))
        rec["sparc_smoothness"] = NAN
    rec.update(curvature_speed_envelope(arrs["speed"], arrs["a_n"]))
    return rec


# ====================================================================== extraction driver
def _row(arrs: dict[str, np.ndarray], r: int) -> dict[str, np.ndarray]:
    return {k: v[r] for k, v in arrs.items()}


def extract(X: Any, M: Any, meta: Any, P: EventParams, rows: Sequence[int] | None = None, perturb: Callable | None = None, seed: int = 0,
            chunk: int = 2048) -> tuple[Any, Any]:
    """(events, sequences) for the bundle rows ``rows`` (default all). ``meta`` rows align with X rows (bundle order).
    ``perturb(arrs, rng) → arrs`` modifies the physical chunk (robustness re-extraction)."""
    import polars as pl

    keep = ("idx", "nfl_id", "domain", "kind", "position_group")
    mcols = {c: meta[c].to_list() for c in keep if c in meta.columns}
    N = int(X.shape[0])
    rows_arr = np.arange(N) if rows is None else np.sort(np.asarray(list(rows), np.int64))
    rng = np.random.default_rng(seed)
    ev_rows: list[dict[str, Any]] = []
    sq_rows: list[dict[str, Any]] = []
    import warnings

    for s in range(0, rows_arr.size, chunk):
        rr = rows_arr[s: s + chunk]
        lo, hi = int(rr.min()), int(rr.max()) + 1
        if rows is None:
            xc, mc = np.asarray(X[lo:hi], np.float32), np.asarray(M[lo:hi], bool)
            local = rr - lo
        else:
            xc, mc = np.asarray(X[rr], np.float32), np.asarray(M[rr], bool)
            local = np.arange(rr.size)
        arrs = physical(xc, mc)
        if perturb is not None:
            arrs = perturb(arrs, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j, r in zip(local.tolist(), rr.tolist()):
                a = _row(arrs, j)
                evs, q, segs = detect_sequence(a, P)
                info = {"row": int(r), **{c: mcols[c][r] for c in mcols}}
                for k, ev in enumerate(evs):
                    ev_rows.append({**info, "event_no": k, **ev})
                sq_rows.append({**info, **q, **sequence_record(a, segs, P)})
    events = pl.DataFrame(ev_rows, infer_schema_length=None) if ev_rows else empty_events()
    seqs = pl.DataFrame(sq_rows, infer_schema_length=None)
    return events, seqs


def empty_events() -> Any:
    import polars as pl

    sch = {"row": pl.Int64, "idx": pl.Int64, "nfl_id": pl.Int64, "domain": pl.String, "kind": pl.String, "position_group": pl.String, "event_no": pl.Int64,
           **{c: pl.Float64 for c in (*EVENT_METRIC_COLUMNS, *EVENT_CONTEXT_COLUMNS)}, **{c: pl.Boolean for c in EVENT_FLAG_COLUMNS}}
    return pl.DataFrame(schema=sch)


def extraction_signature(bundle_fingerprint: str, P: EventParams, rows: Sequence[int] | None = None, perturbation: dict[str, Any] | None = None) -> dict[str, Any]:
    from .config import params_hash

    return {"bundle": bundle_fingerprint, "params": params_hash(P.to_dict()), "row_subset": None if rows is None else params_hash(sorted(int(r) for r in rows)),
            "perturbation": perturbation or {}, "kind": "m14_extraction"}


def cached_extract(run: Any, exp: Any, P: EventParams, reuse: bool = True) -> tuple[Any, Any, str]:
    """Full-bundle extraction with Parquet caches ``features/events/events`` and ``features/sequence/sequence_records``
    keyed by (bundle fingerprint, event-parameter hash). Returns (events, sequences, how)."""
    b = exp.bundle
    prov = extraction_signature(b.fingerprint, P)
    if reuse:
        ev = run.load_data("features/events/events", prov)
        sq = run.load_data("features/sequence/sequence_records", prov)
        if ev is not None and sq is not None and sq.height == b.n_sequences:
            return ev, sq, "reused cache"
    events, seqs = extract(b.X, b.M, b.meta, P)                       # meta rows are in bundle (X) order, as in lab.biomech
    run.data_file(events, "features/events/events", prov)
    run.data_file(seqs, "features/sequence/sequence_records", prov)
    return events, seqs, "computed from the bundle"


def cached_perturbed(run: Any, exp: Any, P: EventParams, rows: Sequence[int], pert: dict[str, Any], seed: int, reuse: bool = True) -> tuple[Any, Any, str]:
    """Re-extraction of bundle ``rows`` under one perturbation (``perturbation_grid`` entry), cached in
    ``features/perturbed/<name>__events|__sequences`` keyed by (bundle, parameters, rows, perturbation, seed)."""
    b = exp.bundle
    prov = extraction_signature(b.fingerprint, P, rows, {**pert, "seed": int(seed)})
    nm = pert["name"]
    if reuse:
        ev = run.load_data(f"features/perturbed/{nm}__events", prov)
        sq = run.load_data(f"features/perturbed/{nm}__sequences", prov)
        if ev is not None and sq is not None:
            return ev, sq, "reused cache"
    ev, sq = extract(b.X, b.M, b.meta, P, rows=rows, perturb=perturbation(pert["noise_sd"], pert["smooth"]), seed=seed)
    run.data_file(ev, f"features/perturbed/{nm}__events", prov)
    run.data_file(sq, f"features/perturbed/{nm}__sequences", prov)
    return ev, sq, "computed"


# ====================================================================== sequence table (one row per sequence, metrics wide)
def sequence_table(events: Any, seqs: Any, folds: dict[int, int] | None = None) -> Any:
    """Per sequence: quality + sequence metrics + event medians (max for MAX_AGG) + multi-cut consistency + direction
    agreement (yaw vs a_n sign); ``fold`` from the experiment fold map."""
    import polars as pl

    ev = nan_to_null(events)
    aggs = [pl.len().alias("n_events_all"), pl.col("complete").sum().alias("n_events_complete")]
    for c in EVENT_METRIC_COLUMNS:
        if c in ev.columns:
            aggs.append((pl.col(c).filter(pl.col("complete")).max() if c in MAX_AGG else pl.col(c).median()).alias(c))
    aggs += [pl.col("v_entry").median().alias("ctx_v_entry"), pl.col("corner_exit_efficiency").std().alias("multi_cut_consistency"),
             pl.col("corner_exit_efficiency").drop_nulls().len().alias("_n_cee"),
             ((pl.col("turn_dir") != 0) & (pl.col("an_turn_sign") != 0)).sum().alias("n_dir_events"),
             ((pl.col("turn_dir") != 0) & (pl.col("turn_dir") == pl.col("an_turn_sign"))).sum().alias("n_dir_agree")]
    per = ev.group_by("row").agg(aggs).with_columns(pl.when(pl.col("_n_cee") >= 2).then(pl.col("multi_cut_consistency")).otherwise(None)
                                                    .alias("multi_cut_consistency")).drop("_n_cee") if ev.height else None
    out = seqs
    if per is not None:
        out = out.join(per, on="row", how="left")
    else:
        out = out.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(c) for c in (*EVENT_METRIC_COLUMNS, "ctx_v_entry", "multi_cut_consistency")])
    out = out.with_columns(pl.col(c).fill_null(0) for c in ("n_events_all", "n_events_complete", "n_dir_events", "n_dir_agree") if c in out.columns)
    if folds is not None:
        fm = pl.DataFrame({"nfl_id": [int(k) for k in folds], "fold": [int(v) for v in folds.values()]}, schema={"nfl_id": pl.Int64, "fold": pl.Int64})
        out = out.drop([c for c in ("fold",) if c in out.columns]).with_columns(pl.col("nfl_id").cast(pl.Int64)).join(fm, on="nfl_id", how="left").with_columns(
            pl.col("fold").fill_null(-1))
    return nan_to_null(out).sort("row")


def nan_to_null(df: Any) -> Any:
    """Float NaN → null (polars medians / std treat NaN as a value; "undefined" must be null)."""
    import polars as pl

    num = [c for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
    return df.with_columns([pl.col(c).fill_nan(None) for c in num]) if num else df


def scope_of(domain: str, kind: str) -> str | None:
    if domain == "nfl":
        return "nfl"
    for s, kinds in RG.SCOPES.items():
        if kind in kinds:
            return s
    return None


def scope_expr() -> Any:
    """Vectorised ``scope_of`` (combine scopes from ``registry.SCOPES``; every NFL window → 'nfl')."""
    import polars as pl

    e = pl.when(pl.col("domain") == "nfl").then(pl.lit("nfl"))
    for s, kinds in RG.SCOPES.items():
        e = e.when((pl.col("domain") == "combine") & pl.col("kind").is_in(list(kinds))).then(pl.lit(s))
    return e.otherwise(pl.lit(None, dtype=pl.String)).alias("_scope")


# ====================================================================== athlete aggregation
def _asymmetry(ev: Any, metric: str, min_each: int = 2) -> Any:
    """Per athlete |median(+) − median(−)| / mean of the two medians over events with a turn direction."""
    import polars as pl

    d = ev.filter((pl.col("turn_dir") != 0) & pl.col(metric).is_not_null() & pl.col(metric).is_finite())
    g = d.group_by("nfl_id", "turn_dir").agg(pl.col(metric).median().alias("m"), pl.len().alias("n"))
    w = g.filter(pl.col("n") >= min_each).pivot(on="turn_dir", index="nfl_id", values="m")
    cols = [c for c in w.columns if c != "nfl_id"]
    if len(cols) < 2:
        return pl.DataFrame(schema={"nfl_id": pl.Int64, "value": pl.Float64})
    a, b = pl.col(cols[0]), pl.col(cols[1])
    return w.drop_nulls().select("nfl_id", ((a - b).abs() / ((a + b).abs() / 2)).alias("value")).filter(pl.col("value").is_finite())


def athlete_table(seqtab: Any, events: Any, rows: Sequence[int] | None = None) -> Any:
    """One row per athlete (nfl_id, position_group, fold) with ``m14c_<scope>__<metric>`` medians, context counts,
    athlete-level asymmetries and the outcome-side ``m14n__<metric>`` medians. ``rows`` restricts the sequences used
    (split-half reliability of athlete-level metrics)."""
    import polars as pl

    st = seqtab if rows is None else seqtab.filter(pl.col("row").is_in(list(rows)))
    ev = events if rows is None else events.filter(pl.col("row").is_in(list(rows)))
    base = seqtab.select("nfl_id", "position_group", *(["fold"] if "fold" in seqtab.columns else [])).unique(subset=["nfl_id"], keep="first")
    st = st.with_columns(scope_expr())
    for scope in RG.SCOPES:
        mets = [m.name for m in RG.METRICS if scope in m.scopes and m.level in ("event", "sequence") and m.name in st.columns]
        g = st.filter(pl.col("_scope") == scope).group_by("nfl_id").agg(
            *[pl.col(m).median().alias(RG.combine_column(m, scope)) for m in mets], pl.len().alias(f"m14c_{scope}__ctx_n_seq"),
            pl.col("n_events_all").sum().alias(f"m14c_{scope}__ctx_n_events"), pl.col("ctx_v_entry").median().alias(f"m14c_{scope}__ctx_v_entry"))
        base = base.join(g, on="nfl_id", how="left")
    if ev.height:
        ev_s = nan_to_null(ev).with_columns(scope_expr())
        ag = ev_s.filter(pl.col("_scope") == "agility")
        for name, metric in (("directional_asymmetry", "corner_exit_efficiency"), ("cut_recovery_asymmetry", "cut_recovery_time")):
            a = _asymmetry(ag, metric).rename({"value": RG.combine_column(name, "agility")})
            base = base.join(a, on="nfl_id", how="left")
    nfl_mets = [m.name for m in RG.METRICS if m.level in ("event", "sequence") and m.name in st.columns]
    n = st.filter(pl.col("_scope") == "nfl").group_by("nfl_id").agg(*[pl.col(m).median().alias(RG.nfl_column(m)) for m in nfl_mets],
                                                                     pl.len().alias("m14n__ctx_n_seq"))
    base = base.join(n, on="nfl_id", how="left")
    for c in [RG.combine_column(x, "agility") for x in ("directional_asymmetry", "cut_recovery_asymmetry")]:
        if c not in base.columns:
            base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return base.sort("nfl_id")


def combine_predictors(ath: Any) -> list[str]:
    """Present, declared Combine predictor columns of an athlete table (never ``m14n__`` / ``ctx``)."""
    declared = set(RG.predictor_columns())
    return [c for c in ath.columns if c in declared]


# ====================================================================== perturbations (Phase 3 robustness)
def _moving_average(a: np.ndarray, valid: np.ndarray, w: int) -> np.ndarray:
    """Centred NaN-aware moving average along the last axis (window w, only valid frames contribute)."""
    x = np.where(valid & np.isfinite(a), a, 0.0)
    c = (valid & np.isfinite(a)).astype(float)
    k = np.ones(w)
    pad = w // 2
    xs = np.apply_along_axis(lambda r: np.convolve(np.pad(r, pad), k, "valid"), -1, x)
    cs = np.apply_along_axis(lambda r: np.convolve(np.pad(r, pad), k, "valid"), -1, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = xs / cs
    return np.where(valid, out, np.nan)


def perturbation(noise_sd: float = 0.0, smooth: int = 0) -> Callable:
    """arrs → arrs: optional centred moving average (speed, a_tan, a_n, yaw) then Gaussian noise on speed (σ) and a_tan
    (√2·σ / 0.1 s: the noise a 10 Hz finite difference of the noisy speed would carry)."""
    def fn(arrs: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        out = dict(arrs)
        valid = arrs["valid"]
        if smooth and smooth > 1:
            for k in ("speed", "a_tan", "a_n", "yaw"):
                out[k] = _moving_average(arrs[k], valid, int(smooth))
        if noise_sd > 0:
            sp = out["speed"] + rng.normal(0.0, noise_sd, valid.shape)
            out["speed"] = np.where(valid, np.clip(sp, 0.0, None), np.nan)
            out["a_tan"] = np.where(valid, out["a_tan"] + rng.normal(0.0, np.sqrt(2) * noise_sd / 0.1, valid.shape), np.nan)
        return out
    return fn


def perturbation_grid(noise_sd: Sequence[float], smooth_windows: Sequence[int]) -> list[dict[str, Any]]:
    grid = [{"name": f"noise_{s:g}", "noise_sd": float(s), "smooth": 0} for s in noise_sd]
    grid += [{"name": f"smooth_{w}", "noise_sd": 0.0, "smooth": int(w)} for w in smooth_windows]
    return grid


def robustness_rows(meta: Any, n_athletes: int, seed: int) -> tuple[list[int], list[int]]:
    """(sampled athlete ids, their Combine bundle rows) for the perturbation re-extraction (seeded, outcome-free)."""
    import polars as pl

    c = meta.with_row_index("_row").filter(pl.col("domain") == "combine")
    ids = sorted(set(int(i) for i in c["nfl_id"].to_list()))
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(ids, size=min(n_athletes, len(ids)), replace=False).tolist()) if ids else []
    rows = c.filter(pl.col("nfl_id").is_in(pick))["_row"].to_list()
    return [int(i) for i in pick], [int(r) for r in rows]


__all__ = ["ENVELOPE_BINS", "SEQ_METRIC_COLUMNS", "athlete_table", "cached_extract", "cached_perturbed", "combine_predictors", "curvature_speed_envelope", "empty_events",
           "extract", "extraction_signature", "nan_to_null", "perturbation", "perturbation_grid", "robustness_rows", "scope_expr", "scope_of",
           "sequence_record", "sequence_table", "sparc", "sprint_profile"]
