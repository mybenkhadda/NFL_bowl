"""E17.A1 — why did M14 event metrics fail their repeatability screen? (Finding 1, [EXP]: 1 of 30 metric instances passed.)

The M14 detector (``m14.events``) is reused unchanged; only its parameters vary. Read-outs:

* ``failure_modes`` — per domain × kind: events per sequence and the share of every detector flag (weak_braking,
  brake_unsustained, propulsion_censored, exit_censored, recovery_censored, complete);
* ``per_drill_icc`` — ICC(1,1) of the per-sequence median of each event metric across an athlete's repeated attempts of ONE
  drill, and pooled over agility drills after kind-centring (the M14 scope), with athlete-bootstrap CIs. A metric whose
  per-drill ICC is clearly higher than the pooled one points to drill heterogeneity, not to an unmeasurable trait;
* ``threshold_grid`` — one-factor-at-a-time variation of ``brake_thresh``, ``sustain_frames`` and ``drop_frac`` around the
  M14 defaults: Combine rows are all re-extracted, NFL rows on a seeded sample (``grid_nfl_sample``); each variant is cached
  under ``features/events_v2/<params hash>`` and summarised by event counts and per-drill ICC.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from ..m14.metrics import EVENT_METRIC_COLUMNS

NAN = float("nan")
FLAGS = ("complete", "weak_braking", "brake_unsustained", "propulsion_censored", "exit_censored", "recovery_censored")
KEY_METRICS = ("braking_point_efficiency", "braking_budget_utilization", "stop_go_elasticity", "turn_speed_retention", "corner_exit_efficiency",
               "brake_propulsion_transition", "cut_recovery_time", "max_controlled_decel", "first_step_explosiveness")
AGILITY = ("three_cone", "short_shuttle", "long_shuttle")


def rows_for_grid(index: Any, nfl_sample: int, seed: int) -> np.ndarray:
    import polars as pl

    c = index.filter(pl.col("domain") == "combine")["idx"].to_numpy()
    n = index.filter(pl.col("domain") == "nfl")["idx"].to_numpy()
    rng = np.random.default_rng(seed)
    n = rng.choice(n, min(nfl_sample, n.size), replace=False) if n.size else n
    return np.sort(np.concatenate([c, n]))


def extract_variant(data: Any, run: Any, P: Any, rows: np.ndarray, reuse: bool = True) -> Any:
    """M14 extraction of ``rows`` under parameters ``P`` (cached by bundle + parameter hash + row subset)."""
    from ..m14.config import params_hash
    from ..m14.metrics import extract, extraction_signature

    prov = extraction_signature(data.fingerprint, P, rows.tolist())
    rel = f"features/events_v2/events_{params_hash(P.to_dict())}"
    if reuse:
        ev = run.load_data(rel, prov)
        if ev is not None:
            return ev
    ev, _ = extract(data.bundle.X, data.bundle.M, data.bundle.meta, P, rows=rows)
    run.data_file(ev, rel, prov)
    return ev


def failure_modes(events: Any) -> Any:
    import polars as pl

    if not events.height:
        return pl.DataFrame()
    ev = events.with_columns([pl.col(f).cast(pl.Boolean) for f in FLAGS if f in events.columns])
    return ev.group_by("domain", "kind").agg(pl.len().alias("n_events"), pl.col("row").n_unique().alias("n_sequences"),
                                             pl.col("nfl_id").n_unique().alias("n_athletes"),
                                             (pl.len() / pl.col("row").n_unique()).alias("events_per_sequence"),
                                             *[pl.col(f).mean().alias(f"share_{f}") for f in FLAGS if f in ev.columns]).sort("domain", "kind")


def sequence_medians(events: Any, metrics: Sequence[str] = KEY_METRICS) -> Any:
    import polars as pl

    ev = events.with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in metrics if c in events.columns])
    return ev.group_by("row", "nfl_id", "domain", "kind", "position_group").agg([pl.col(c).median() for c in metrics if c in ev.columns])


def per_drill_icc(events: Any, n_boot: int, seed: int, min_athletes: int = 30, metrics: Sequence[str] = KEY_METRICS) -> Any:
    """ICC per (metric, drill) and pooled agility (kind-centred), Combine attempts as trials."""
    import polars as pl

    from ..reliability import icc_bootstrap, icc_components, kind_centre

    sm = sequence_medians(events, metrics).filter(pl.col("domain") == "combine")
    rows = []
    for k, m in enumerate([c for c in metrics if c in sm.columns]):
        base = sm.filter(pl.col(m).is_not_null()).select("nfl_id", "kind", pl.col(m).alias("value"))
        scopes = [(kd, base.filter(pl.col("kind") == kd), False) for kd in sorted(base["kind"].unique().to_list())]
        scopes.append(("agility_pooled", base.filter(pl.col("kind").is_in(list(AGILITY))), True))
        for j, (scope, t, centre) in enumerate(scopes):
            if not t.height:
                continue
            v = kind_centre(t) if centre else t["value"].to_numpy().astype(float)
            ids = t["nfl_id"].to_numpy()
            comp = icc_components(v, ids)
            lo = hi = NAN
            if comp["n_athletes"] >= min_athletes and n_boot:
                lo, hi = icc_bootstrap(v, ids, n_boot, seed + 101 * k + j)
            rows.append({"metric": m, "scope": scope, "n_athletes": comp["n_athletes"], "n_obs": comp["n_obs"], "icc1": comp["icc1"], "icc_lo": lo, "icc_hi": hi,
                         "enough": comp["n_athletes"] >= min_athletes})
    return pl.DataFrame(rows, infer_schema_length=None)


def grid_variants(base: Any, cfg: Any) -> list[tuple[str, Any]]:
    out = [("default", base)]
    out += [(f"brake_thresh={v:g}", replace(base, brake_thresh=float(v))) for v in cfg.brake_thresh_grid if v != base.brake_thresh]
    out += [(f"sustain_frames={v}", replace(base, sustain_frames=int(v))) for v in cfg.sustain_grid if v != base.sustain_frames]
    out += [(f"drop_frac={v:g}", replace(base, drop_frac=float(v))) for v in cfg.drop_frac_grid if v != base.drop_frac]
    return out


def threshold_grid(data: Any, run: Any, cfg: Any, base: Any | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """(event counts per variant × domain × kind, per-drill ICC per variant, {variant: events}) — default variant included."""
    import polars as pl

    from ..m14.events import EventParams

    base = base or EventParams()
    rows = rows_for_grid(data.index, cfg.grid_nfl_sample, cfg.seed)
    counts, iccs, evs = [], [], {}
    for name, P in grid_variants(base, cfg):
        ev = extract_variant(data, run, P, rows, cfg.reuse_artifacts)
        evs[name] = ev
        fm = failure_modes(ev)
        if fm.height:
            counts.append(fm.with_columns(pl.lit(name).alias("variant")))
        ic = per_drill_icc(ev, 0, cfg.seed, cfg.min_reliability_athletes)
        if ic.height:
            iccs.append(ic.with_columns(pl.lit(name).alias("variant")))
    return (pl.concat(counts, how="diagonal_relaxed") if counts else pl.DataFrame(), pl.concat(iccs, how="diagonal_relaxed") if iccs else pl.DataFrame(), evs)


def diagnosis(fm: Any, icc_default: Any, grid_icc: Any) -> Any:
    """Per metric: best per-drill ICC vs pooled agility ICC, best variant ICC vs default, and the dominant reading."""
    import polars as pl

    rows = []
    for m in icc_default["metric"].unique().to_list() if icc_default.height else []:
        d = icc_default.filter((pl.col("metric") == m) & pl.col("enough"))
        pooled = d.filter(pl.col("scope") == "agility_pooled")["icc1"]
        drill = d.filter(pl.col("scope") != "agility_pooled")
        best_drill = drill.sort("icc1", descending=True).head(1)
        g = grid_icc.filter((pl.col("metric") == m) & pl.col("enough") & (pl.col("scope") == "agility_pooled")) if grid_icc.height else pl.DataFrame()
        best_var = g.sort("icc1", descending=True).head(1) if g.height else g
        p = float(pooled[0]) if pooled.len() else NAN
        bd = float(best_drill["icc1"][0]) if best_drill.height else NAN
        bv = float(best_var["icc1"][0]) if best_var.height else NAN
        if np.isfinite(bd) and np.isfinite(p) and bd - p >= 0.15:
            reading = "drill heterogeneity (pooling hurts)"
        elif np.isfinite(bv) and np.isfinite(p) and bv - p >= 0.15:
            reading = "definition-sensitive (a threshold variant helps)"
        elif np.isfinite(p) and p < 0.4 and (not np.isfinite(bd) or bd < 0.4):
            reading = "low repeatability under every tested definition"
        else:
            reading = "no dominant failure mode"
        rows.append({"metric": m, "pooled_agility_icc": p, "best_drill": best_drill["scope"][0] if best_drill.height else None, "best_drill_icc": bd,
                     "best_variant": best_var["variant"][0] if best_var.height else None, "best_variant_icc": bv, "reading": reading})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["AGILITY", "EVENT_METRIC_COLUMNS", "FLAGS", "KEY_METRICS", "diagnosis", "extract_variant", "failure_modes", "grid_variants", "per_drill_icc",
           "rows_for_grid", "sequence_medians", "threshold_grid"]
