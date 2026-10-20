"""NFL target registry — every target has its own ID, definition, units, provenance and fingerprint.

T04 needs only the two existing targets; T06 (E16.A2) adds the task targets T1–T6 to ``TARGETS`` with their own specs.
Two targets are NEVER compared on RMSE (different scales and cohorts): ``assert_same_target`` guards every comparison.

    T0   nfl.q90_shared         official 12C frozen target (athletes.parquet), the only target of the official ladder
    T0x  nfl_speed_q90_median   exploratory-notebook target (393 athletes reported [EXP]); audit / reconciliation only
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ...data.validation import TRANSFER_TARGET
from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked, ValidationFailed


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    column: str
    unit: str
    role: str                        # official · exploratory · task (T1–T6, later)
    source: str                      # where the values come from
    aggregation: str
    selection: str
    censoring: str
    min_exposure: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TARGETS: dict[str, TargetSpec] = {
    "T0": TargetSpec("T0", TRANSFER_TARGET, "yd/s", "official", "bundle athletes.parquet (12C freeze)",
                     "frozen 12C estimator over shared speed bins (athlete level)", "linked athletes with the frozen target",
                     "as frozen in 12C", "as frozen in 12C",
                     "12C target freeze; constructed once on all athletes before any Combine join (M13 decision 7: not strictly inductive end-to-end)"),
    "T0x": TargetSpec("T0x", "nfl_speed_q90_median", "yd/s", "exploratory", "exploratory M15 notebook export (exploratory_target_path)",
                      "median over NFL windows of a per-window q90 speed (as described by the exploratory notebook; definition not in the repository)",
                      "exploratory notebook selection (393 athletes reported [EXP])", "unknown", "unknown",
                      "outside the repository; reconciliation only, never a claim"),
}


_TASK = {
    "T1": ("t16_high_speed", "yd/s", "window max speed (valid frames) → athlete q90 within each NFL demand kind (≥ min_windows_kind) → "
           "position-level demand-mix weighted mean over kinds", "windows ending at their max speed are right-censored for the max (flag share reported)"),
    "T2": ("t16_accel_v2_4", "yd/s²", "per window, median positive a_tan over valid frames with speed in [2, 4) yd/s (≥ 3 frames) → athlete q90 "
           "(acceleration capacity at matched entry speed; band [4, 6) as T2b)", "windows never reaching the band contribute nothing"),
    "T2b": ("t16_accel_v4_6", "yd/s²", "as T2 for the [4, 6) yd/s band", "as T2"),
    "T3": ("t16_braking", "yd/s²", "M14 event max_controlled_decel on COMPLETE NFL events → athlete q90", "incomplete events excluded (share reported)"),
    "T4": ("t16_reaccel", "yd/s²", "M14 event first_step_explosiveness (mean a_tan over 3 frames from propulsion onset) on NFL events with "
           "observed propulsion → athlete median", "propulsion-censored events excluded (share reported); time-to-recover handled in M17 (KM)"),
    "T5": ("t16_turn_retention", "ratio", "M14 event turn_speed_retention (v_min / v_entry) on NFL events with a turn direction (|angle| ≥ 20°, "
           "heading coverage ≥ 0.5) → athlete median", "events without a turn direction excluded"),
    "T6": ("t16_consistency", "yd/s", "within-athlete SD of window max speed per demand kind (≥ min_windows_kind) → mean over kinds (lower = "
           "more consistent)", "exposure-dependent by construction: reliability vs exposure reported"),
}
for _k, (_col, _u, _agg, _cens) in _TASK.items():
    TARGETS[_k] = TargetSpec(_k, _col, _u, "task", "bundle NFL windows (+ M14 detector on NFL rows for T3–T5)", _agg,
                             "NFL demand windows of the bundle (09A rules); athletes with the target's minimum exposure", _cens,
                             "see TargetConfig (min_windows / min_events)",
                             "constructed in M16 from NFL data only (no Combine value read); demand-mix weights from all athletes' window counts "
                             "(outcome-free, not strictly inductive)")
TASK_IDS = tuple(_TASK)


@dataclass(frozen=True)
class TargetConfig:
    min_windows: int = 20                    # T1 / T2 / T6: windows per athlete
    min_windows_kind: int = 5                # per demand kind
    min_events: int = 5                      # T3–T5: qualifying events per athlete
    band_min_frames: int = 3
    bands: tuple[tuple[float, float], ...] = ((2.0, 4.0), (4.0, 6.0))
    q: float = 0.9


def band_accel(bundle: Any, rows: Any, cfg: Any, bands: Sequence[tuple[float, float]], min_frames: int) -> dict[str, Any]:
    """Per row and band: median positive a_tan over valid frames with speed in the band (NaN below ``min_frames``)."""
    import numpy as np

    from ..m14.events import physical
    from ..m15.data import chunks

    rows = np.asarray(rows, np.int64)
    out = {f"band_{i}": np.full(rows.size, np.nan) for i in range(len(bands))}
    out["ends_at_max"] = np.zeros(rows.size, bool)
    for s, e in chunks(rows.size, cfg.chunk_rows):
        p = physical(np.asarray(bundle.X[rows[s:e]]), np.asarray(bundle.M[rows[s:e]]))
        v, a, ok = p["speed"], p["a_tan"], p["valid"]
        for i, (lo, hi) in enumerate(bands):
            m = ok & (v >= lo) & (v < hi) & (a > 0)
            cnt = m.sum(1)
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(m, a, np.nan), axis=1)
            out[f"band_{i}"][s:e] = np.where(cnt >= min_frames, med, np.nan)
        last = np.where(ok.any(1), ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1), 0)
        vmax_at = np.nanargmax(np.where(ok, v, -np.inf), axis=1)
        out["ends_at_max"][s:e] = ok.any(1) & (vmax_at == last)
    return out


def nfl_window_table(data: Any, feat: Any, cfg: Any, tcfg: TargetConfig) -> Any:
    """One row per NFL window: idx, nfl_id, position_group, kind, season, fold, n_valid, max_speed, band accelerations,
    ends_at_max (reuses the M15 E15.A1 feature table for max speed)."""
    import polars as pl

    w = feat.filter(pl.col("domain") == "nfl").select("idx", "nfl_id", "position_group", "kind", "fold", "n_valid", "max_speed")
    if "season" in data.index.columns:
        w = w.join(data.index.select("idx", "season"), on="idx", how="left")
    b = band_accel(data.bundle, w["idx"].to_numpy(), cfg, tcfg.bands, tcfg.band_min_frames)
    return w.with_columns(pl.Series("accel_band_0", b["band_0"]), pl.Series("accel_band_1", b["band_1"]), pl.Series("ends_at_max", b["ends_at_max"]))


def nfl_events(data: Any, run: Any, params: Any = None, reuse: bool = True) -> Any:
    """M14 detector on NFL rows only (``m14.metrics.extract``), cached under ``features/targets/nfl_events`` keyed by
    (bundle fingerprint, event-parameter hash)."""
    import polars as pl

    from ..m14.events import EventParams
    from ..m14.metrics import extract, extraction_signature

    P = params or EventParams()
    rows = data.index.filter(pl.col("domain") == "nfl")["idx"].to_numpy()
    prov = extraction_signature(data.fingerprint, P, None, {"domain": "nfl"})
    if reuse and run is not None:
        ev = run.load_data("features/targets/nfl_events", prov)
        if ev is not None:
            return ev
    ev, _seq = extract(data.bundle.X, data.bundle.M, data.bundle.meta, P, rows=rows)
    if run is not None:
        run.data_file(ev, "features/targets/nfl_events", prov)
    return ev


def _kind_weights(windows: Any) -> Any:
    """Position-level share of windows per demand kind (all athletes; outcome-free)."""
    import polars as pl

    c = windows.group_by("position_group", "kind").len()
    return c.with_columns((pl.col("len") / pl.col("len").sum().over("position_group")).alias("w")).select("position_group", "kind", "w")


def window_targets(windows: Any, tcfg: TargetConfig) -> Any:
    """T1, T2, T2b, T6 per athlete with exposure columns (n_windows, n_kinds_used, censored share)."""
    import polars as pl

    w = windows.filter(pl.col("max_speed").is_finite())
    kw = _kind_weights(w)
    per_kind = w.group_by("nfl_id", "position_group", "kind").agg(pl.len().alias("n"), pl.col("max_speed").quantile(tcfg.q).alias("q_max"),
                                                                   pl.col("max_speed").std().alias("sd_max"))
    per_kind = per_kind.filter(pl.col("n") >= tcfg.min_windows_kind).join(kw, on=["position_group", "kind"], how="left")
    t1 = per_kind.group_by("nfl_id").agg(((pl.col("q_max") * pl.col("w")).sum() / pl.col("w").sum()).alias(TARGETS["T1"].column),
                                         pl.col("sd_max").mean().alias(TARGETS["T6"].column), pl.len().alias("n_kinds_used"))
    tot = w.group_by("nfl_id").agg(pl.len().alias("n_windows"), pl.col("ends_at_max").mean().alias("share_windows_end_at_max"))
    acc = windows.group_by("nfl_id").agg(pl.col("accel_band_0").drop_nans().quantile(tcfg.q).alias(TARGETS["T2"].column),
                                         pl.col("accel_band_0").is_finite().sum().alias("n_band_0"),
                                         pl.col("accel_band_1").drop_nans().quantile(tcfg.q).alias(TARGETS["T2b"].column),
                                         pl.col("accel_band_1").is_finite().sum().alias("n_band_1"))
    out = tot.join(t1, on="nfl_id", how="left").join(acc, on="nfl_id", how="left")
    enough = pl.col("n_windows") >= tcfg.min_windows
    return out.with_columns(pl.when(enough).then(pl.col(TARGETS["T1"].column)).otherwise(None).alias(TARGETS["T1"].column),
                            pl.when(enough).then(pl.col(TARGETS["T6"].column)).otherwise(None).alias(TARGETS["T6"].column),
                            pl.when(pl.col("n_band_0") >= tcfg.min_events).then(pl.col(TARGETS["T2"].column)).otherwise(None).alias(TARGETS["T2"].column),
                            pl.when(pl.col("n_band_1") >= tcfg.min_events).then(pl.col(TARGETS["T2b"].column)).otherwise(None).alias(TARGETS["T2b"].column))


def event_targets(events: Any, tcfg: TargetConfig) -> Any:
    """T3, T4, T5 per athlete with qualifying-event counts and exclusion shares."""
    import polars as pl

    if events is None or not events.height:
        return pl.DataFrame(schema={"nfl_id": pl.Int64})
    ev = events.with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in ("max_controlled_decel", "first_step_explosiveness", "turn_speed_retention")
                              if c in events.columns])
    t3 = ev.filter(pl.col("complete") & pl.col("max_controlled_decel").is_not_null()).group_by("nfl_id").agg(
        pl.col("max_controlled_decel").quantile(tcfg.q).alias(TARGETS["T3"].column), pl.len().alias("n_events_t3"))
    t4 = ev.filter(~pl.col("propulsion_censored") & pl.col("first_step_explosiveness").is_not_null()).group_by("nfl_id").agg(
        pl.col("first_step_explosiveness").median().alias(TARGETS["T4"].column), pl.len().alias("n_events_t4"))
    t5 = ev.filter((pl.col("turn_dir") != 0) & pl.col("turn_speed_retention").is_not_null()).group_by("nfl_id").agg(
        pl.col("turn_speed_retention").median().alias(TARGETS["T5"].column), pl.len().alias("n_events_t5"))
    shares = ev.group_by("nfl_id").agg(pl.len().alias("n_events_all"), (~pl.col("complete")).mean().alias("share_incomplete"),
                                       pl.col("propulsion_censored").mean().alias("share_propulsion_censored"))
    out = shares.join(t3, on="nfl_id", how="left").join(t4, on="nfl_id", how="left").join(t5, on="nfl_id", how="left")
    for t, n in (("T3", "n_events_t3"), ("T4", "n_events_t4"), ("T5", "n_events_t5")):
        out = out.with_columns(pl.when(pl.col(n).fill_null(0) >= tcfg.min_events).then(pl.col(TARGETS[t].column)).otherwise(None).alias(TARGETS[t].column))
    return out


def athlete_targets(windows: Any, events: Any, athletes: Any, fold_of: dict[int, int], tcfg: TargetConfig) -> Any:
    """All targets per athlete (T0 from athletes.parquet; T1–T6 from NFL data) with exposure columns and the fold."""
    import polars as pl

    base = athletes.select(pl.col("nfl_id").cast(pl.Int64), "position_group", *([TARGETS["T0"].column] if TARGETS["T0"].column in athletes.columns else []))
    wt = window_targets(windows, tcfg).with_columns(pl.col("nfl_id").cast(pl.Int64))
    et = event_targets(events, tcfg)
    et = et.with_columns(pl.col("nfl_id").cast(pl.Int64)) if et.height else et
    out = base.join(wt, on="nfl_id", how="left")
    if et.width > 1:
        out = out.join(et, on="nfl_id", how="left")
    return out.with_columns(pl.col("nfl_id").replace_strict(fold_of, default=-1, return_dtype=pl.Int64).alias("fold"))


def coverage_table(targets: Any) -> Any:
    """Per target × position: athletes with a value, mean / SD (descriptive), and the total."""
    import polars as pl

    rows = []
    for tid, spec in TARGETS.items():
        if spec.column not in targets.columns:
            continue
        d = targets.filter(pl.col(spec.column).is_not_null() & pl.col(spec.column).is_finite())
        for p, g in [("ALL", d), *[(str(k[0]), g) for k, g in d.group_by("position_group")]]:
            rows.append({"target_id": tid, "column": spec.column, "unit": spec.unit, "role": spec.role, "position_group": p, "n_athletes": g.height,
                         "mean": float(g[spec.column].mean()) if g.height else None, "sd": float(g[spec.column].std()) if g.height > 1 else None})
    return pl.DataFrame(rows, infer_schema_length=None).sort("target_id", "position_group")


def redundancy(targets: Any) -> Any:
    """Spearman between targets (athletes with both), descriptive."""
    import numpy as np
    import polars as pl

    from ..stats import spearman

    ids = [t for t, s in TARGETS.items() if s.column in targets.columns]
    rows = []
    for a in ids:
        for b in ids:
            d = targets.select(TARGETS[a].column, TARGETS[b].column).drop_nulls()
            x, y = d[TARGETS[a].column].to_numpy().astype(float), d[TARGETS[b].column].to_numpy().astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            rows.append({"target_a": a, "target_b": b, "n": int(ok.sum()), "spearman": spearman(x[ok], y[ok]) if ok.sum() >= 20 else None})
    return pl.DataFrame(rows, infer_schema_length=None)


def frame_fingerprint(df: Any, key: str = "nfl_id") -> str:
    """Content fingerprint of a frame (sorted by ``key``; floats at 17 significant digits)."""
    d = df.sort(key) if key in df.columns else df
    return hashlib.sha256(d.write_csv(float_precision=17).encode("utf-8")).hexdigest()[:16]


def target_frame(spec: TargetSpec, athletes: Any, path: str = "") -> Any:
    """(nfl_id, position_group, <column>) for one target, non-null values only. Official: from ``athletes``;
    exploratory: from ``athletes`` if the column exists there, else from ``path``. Missing → BLOCKED_MISSING_INPUT."""
    import polars as pl

    if athletes is not None and spec.column in athletes.columns:
        src = athletes
    elif path and Path(path).is_file():
        p = Path(path)
        src = pl.read_parquet(p) if p.suffix.lower() == ".parquet" else pl.read_csv(p, infer_schema_length=10_000)
        if "nfl_id" not in src.columns or spec.column not in src.columns:
            raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"{p} lacks nfl_id / {spec.column}", missing=[f"{p}:{spec.column}"])
        if "position_group" not in src.columns and athletes is not None:
            src = src.join(athletes.select("nfl_id", "position_group"), on="nfl_id", how="left")
    else:
        where = f"athletes.parquet (column {spec.column})" + (f" or {path}" if path else " (no exploratory_target_path configured)")
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"target {spec.target_id} ({spec.column}) not found in {where}", missing=[spec.column])
    cols = ["nfl_id", *(["position_group"] if "position_group" in src.columns else []), spec.column]
    return src.select(cols).with_columns(pl.col("nfl_id").cast(pl.Int64)).filter(pl.col(spec.column).is_not_null() & pl.col(spec.column).is_finite())


def target_summary(spec: TargetSpec, frame: Any) -> dict[str, Any]:
    """Coverage, distribution and fingerprint of one target frame (descriptive)."""
    import polars as pl

    v = frame[spec.column]
    by = frame.group_by("position_group").len().sort("position_group").to_dicts() if "position_group" in frame.columns else []
    return {**spec.to_dict(), "n_athletes": frame.height, "mean": float(v.mean()) if frame.height else None, "sd": float(v.std()) if frame.height > 1 else None,
            "min": float(v.min()) if frame.height else None, "max": float(v.max()) if frame.height else None,
            "by_position": by, "fingerprint": frame_fingerprint(frame.select("nfl_id", pl.col(spec.column)))}


def assert_same_target(a: str, b: str) -> None:
    """Guard: metrics of two different targets are never compared."""
    if a != b:
        raise ValidationFailed(f"refusing to compare predictions of different targets ({a} vs {b}): RMSEs are on different scales and cohorts")


__all__ = ["TARGETS", "TASK_IDS", "TargetConfig", "TargetSpec", "assert_same_target", "athlete_targets", "band_accel", "coverage_table", "event_targets",
           "frame_fingerprint", "nfl_events", "nfl_window_table", "redundancy", "target_frame", "target_summary", "window_targets"]
