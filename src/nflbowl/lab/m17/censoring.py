"""E17.A3 — censoring-aware recovery time (M14 ``cut_recovery_time`` with ``recovery_censored``).

An event whose speed never returns to ``recovery_frac``·v_entry before the gap-free segment ends is right-censored: its
recovery time is at least the time from the speed minimum to the segment end. Excluding such events (the naive median of
completed recoveries) biases recovery time DOWN for athletes whose clips end early. Here:

* censor time = (seg_end − 1 − i_min) × nominal 0.1 s (the detector's segment indices; the nominal step is a stated
  approximation for segments without long frames);
* athlete-level Kaplan–Meier median of time-to-recovery per domain (NaN when the survival curve never reaches 0.5);
* reliability: events split by sequence parity → KM medians per half → position-residualised split-half (Spearman–Brown),
  compared with the naive completed-only median. The censoring-aware estimate is adopted only if its reliability is higher.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NAN = float("nan")


def km_median(times: np.ndarray, censored: np.ndarray) -> float:
    """Kaplan–Meier median of right-censored times (``censored`` True = no event observed by that time)."""
    t = np.asarray(times, float)
    c = np.asarray(censored, bool)
    ok = np.isfinite(t)
    t, c = t[ok], c[ok]
    if t.size == 0:
        return NAN
    order = np.argsort(t, kind="mergesort")
    t, c = t[order], c[order]
    s, n = 1.0, t.size
    for u in np.unique(t):
        at = t == u
        d = int((at & ~c).sum())
        if d:
            s *= 1 - d / n
            if s <= 0.5:
                return float(u)
        n -= int(at.sum())
    return NAN


def recovery_records(events: Any, dt: float = 0.1) -> Any:
    import polars as pl

    need = ("cut_recovery_time", "recovery_censored", "seg_end", "i_min")
    if events is None or not events.height or any(c not in events.columns for c in need):
        return pl.DataFrame()
    return events.select("row", "nfl_id", "domain", "kind", "position_group", pl.col("cut_recovery_time").cast(pl.Float64).fill_nan(None).alias("t_obs"),
                         pl.col("recovery_censored").cast(pl.Boolean).alias("censored"),
                         ((pl.col("seg_end") - 1 - pl.col("i_min")).cast(pl.Float64) * dt).alias("t_censor")).with_columns(
        pl.when(pl.col("censored")).then(pl.col("t_censor")).otherwise(pl.col("t_obs")).alias("time"))


def athlete_recovery(rec: Any, min_events: int = 3) -> Any:
    """Per athlete × domain: KM median, naive (completed-only) median, event count, censored share."""
    import polars as pl

    rows = []
    for (i, dom, pos), g in rec.group_by("nfl_id", "domain", "position_group"):
        if g.height < min_events:
            continue
        t, c = g["time"].to_numpy().astype(float), g["censored"].to_numpy()
        obs = g.filter(~pl.col("censored"))["t_obs"].drop_nulls().to_numpy()
        rows.append({"nfl_id": int(i), "domain": dom, "position_group": pos, "n_events": g.height, "censored_share": float(np.mean(c)),
                     "km_median": km_median(t, c), "naive_median": float(np.median(obs)) if obs.size else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


def censoring_reliability(rec: Any, n_boot: int, seed: int, min_events: int = 4) -> Any:
    """Split-half reliability (sequence parity) of the KM and naive athlete medians per domain."""
    import polars as pl

    from ..m16.target_reliability import split_half

    r = rec.with_columns((pl.col("row") % 2).alias("half"))
    rows = []
    for dom in ("combine", "nfl"):
        d = r.filter(pl.col("domain") == dom)
        halves = [athlete_recovery(d.filter(pl.col("half") == h), max(2, min_events // 2)) for h in (0, 1)]
        if not all(x.height for x in halves):
            continue
        j = halves[0].join(halves[1], on="nfl_id", suffix="_b")
        for est in ("km_median", "naive_median"):
            out = split_half(j[est].to_numpy().astype(float), j[f"{est}_b"].to_numpy().astype(float), j["position_group"].fill_null("?").to_numpy(), n_boot, seed)
            rows.append({"domain": dom, "estimator": est, **out})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["athlete_recovery", "censoring_reliability", "km_median", "recovery_records"]
