"""E17.A2 — a-priori context strata for events and their coverage in both domains (Findings 2–3).

Strata are fixed in the configuration BEFORE any analysis (never tuned on data):
    entry      entry-speed bin of ``v_entry`` with edges ``entry_speed_bins`` (yd/s): <2, 2–4, 4–6, 6–8, ≥8
    demand     deceleration demand = speed_drop / v_entry with edges ``decel_demand_bins``
    turn       |turn_angle| class (deg) with edges ``turn_bins_deg`` when heading coverage ≥ 0.5, else "turn_unknown"
    complete   detector completeness flag
A stratum is SUPPORTED for cross-domain work when ≥ ``min_stratum_athletes`` athletes have ≥ ``min_events_per_athlete``
events in it in BOTH domains; unsupported strata are excluded from every cross-domain comparison.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _bin_labels(edges: Sequence[float], unit: str) -> list[str]:
    e = list(edges)
    return [f"<{e[0]:g}{unit}", *[f"{a:g}-{b:g}{unit}" for a, b in zip(e[:-1], e[1:], strict=True)], f">={e[-1]:g}{unit}"]


def assign_strata(events: Any, cfg: Any) -> Any:
    import polars as pl

    ev = events.with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in ("v_entry", "speed_drop", "turn_angle", "turn_heading_coverage") if c in events.columns])
    el = _bin_labels(cfg.entry_speed_bins, "yd/s")
    dl = _bin_labels(cfg.decel_demand_bins, "")
    tl = _bin_labels(cfg.turn_bins_deg, "deg")
    ve = ev["v_entry"].to_numpy().astype(float) if "v_entry" in ev.columns else np.full(ev.height, np.nan)
    dem = (ev["speed_drop"].to_numpy().astype(float) / np.where(ve > 0, ve, np.nan)) if "speed_drop" in ev.columns else np.full(ev.height, np.nan)
    ang = np.degrees(np.abs(ev["turn_angle"].to_numpy().astype(float))) if "turn_angle" in ev.columns else np.full(ev.height, np.nan)
    cov = ev["turn_heading_coverage"].to_numpy().astype(float) if "turn_heading_coverage" in ev.columns else np.zeros(ev.height)

    def lab(x: np.ndarray, edges: Sequence[float], labels: list[str], missing: str) -> list[str]:
        idx = np.searchsorted(np.asarray(edges), x, side="right")
        return [labels[i] if np.isfinite(v) else missing for i, v in zip(idx, x, strict=True)]

    turn = lab(ang, cfg.turn_bins_deg, tl, "turn_unknown")
    turn = [t if (np.isfinite(c) and c >= 0.5) else "turn_unknown" for t, c in zip(turn, cov, strict=True)]
    out = ev.with_columns(pl.Series("s_entry", lab(ve, cfg.entry_speed_bins, el, "entry_unknown")), pl.Series("s_demand", lab(dem, cfg.decel_demand_bins, dl, "demand_unknown")),
                          pl.Series("s_turn", turn))
    return out.with_columns(pl.concat_str([pl.col("s_entry"), pl.col("s_turn")], separator=" | ").alias("stratum"))


def coverage(strata: Any, cfg: Any, key: str = "stratum") -> Any:
    """Per stratum: athletes with ≥ min events per domain, and the SUPPORTED flag."""
    import polars as pl

    per = strata.group_by(key, "domain", "nfl_id").len().filter(pl.col("len") >= cfg.min_events_per_athlete)
    c = per.group_by(key, "domain").agg(pl.col("nfl_id").n_unique().alias("athletes")).pivot(on="domain", index=key, values="athletes").fill_null(0)
    for d in ("combine", "nfl"):
        if d not in c.columns:
            c = c.with_columns(pl.lit(0).alias(d))
    tot = strata.group_by(key, "domain").len().pivot(on="domain", index=key, values="len").fill_null(0)
    tot = tot.rename({d: f"events_{d}" for d in ("combine", "nfl") if d in tot.columns})
    return c.rename({"combine": "athletes_combine", "nfl": "athletes_nfl"}).join(tot, on=key, how="left").with_columns(
        ((pl.col("athletes_combine") >= cfg.min_stratum_athletes) & (pl.col("athletes_nfl") >= cfg.min_stratum_athletes)).alias("supported")).sort(key)


__all__ = ["assign_strata", "coverage"]
