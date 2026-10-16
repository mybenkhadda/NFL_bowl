"""Machine-readable statistical results — one schema for every study (brief §15 + explicit grain).

Rules enforced at collection time (``ResultCollector.add`` raises on violation):
* ``grain`` is one of ``GRAINS``; frame statistics must carry ``n_frames``; athlete / trajectory
  counts are separate columns and never substituted for each other.
* ``result_status`` is one of ``STATUSES``; nothing is "confirmed" before user review.
* Rows with too few athletes are kept with status ``insufficient_data`` (never silently dropped).
* p-values are adjusted with Benjamini–Hochberg WITHIN ``adjustment_family`` (``finalize``); all
  rows are reported, significant or not.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from ..contracts import TableContract
from .stats import bh_fdr

S, I, F = pl.String(), pl.Int64(), pl.Float64()
GRAINS = ("frame", "segment", "trajectory", "attempt", "event", "maneuver", "athlete", "athlete_drill", "play",
          "season")
STATUSES = ("descriptive", "exploratory", "inconclusive", "insufficient_data", "not_applicable")

RESULTS = TableContract(
    "results", "Long-format statistical results (one row per estimate).",
    {"study_id": S, "analysis_id": S, "domain": S, "population": S, "grain": S, "position": S, "cohort": S,
     "stratum": S, "feature": S, "outcome": S, "n_athletes": I, "n_trajectories": I, "n_frames": I,
     "effect_estimate": F, "effect_size_type": S, "unit": S, "standard_error": F, "ci_lower": F, "ci_upper": F,
     "ci_level": F, "ci_method": S, "p_value": F, "p_adjusted": F, "adjustment_family": S, "method": S,
     "model_adjustments": S, "quality_filter": S, "limitations": S, "result_status": S},
    key=("study_id", "analysis_id", "domain", "population", "position", "cohort", "stratum", "feature", "outcome",
         "effect_size_type"),
    area="statistical",
    docs={"grain": "observational unit of the statistic (frame, trajectory, attempt, athlete, …)",
          "n_frames": "only for frame-grain statistics; null otherwise",
          "ci_method": "e.g. 'athlete-cluster bootstrap (B=1000, percentile)', 'Wilson', 'Fisher z'",
          "p_adjusted": "Benjamini–Hochberg within adjustment_family (null if no p-value)"})

_DEFAULTS: dict[str, Any] = {"domain": "all", "population": "all", "position": "all", "cohort": "all",
                             "stratum": "all", "feature": "", "outcome": "", "n_frames": None, "unit": "",
                             "standard_error": None, "ci_lower": None, "ci_upper": None, "ci_level": 0.95,
                             "ci_method": "", "p_value": None, "adjustment_family": "", "model_adjustments": "none",
                             "quality_filter": "", "limitations": "", "result_status": "descriptive"}


def _num(v: Any) -> float | None:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


class ResultCollector:
    def __init__(self, study_id: str, min_athletes: int = 10):
        self.study_id = study_id
        self.min_athletes = min_athletes
        self.rows: list[dict[str, Any]] = []

    def add(self, *, analysis_id: str, grain: str, n_athletes: int, n_trajectories: int | None,
            effect_estimate: float | None, effect_size_type: str, method: str, **kw: Any) -> None:
        if grain not in GRAINS:
            raise ValueError(f"grain {grain!r} not in {GRAINS}")
        unknown = set(kw) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown result fields {sorted(unknown)}")
        row = {**_DEFAULTS, **kw}
        if grain == "frame" and row["n_frames"] is None:
            raise ValueError(f"{analysis_id}: frame-grain results must report n_frames")
        if grain != "frame" and row["n_frames"] is not None:
            raise ValueError(f"{analysis_id}: n_frames is only reported for frame-grain results")
        if row["result_status"] not in STATUSES:
            raise ValueError(f"result_status {row['result_status']!r} not in {STATUSES}")
        if n_athletes < self.min_athletes and row["result_status"] != "not_applicable":
            row["result_status"] = "insufficient_data"
            row["limitations"] = "; ".join(x for x in (row["limitations"],
                                                       f"< {self.min_athletes} athletes") if x)
        self.rows.append({**row, "study_id": self.study_id, "analysis_id": analysis_id, "grain": grain,
                          "n_athletes": int(n_athletes),
                          "n_trajectories": None if n_trajectories is None else int(n_trajectories),
                          "n_frames": None if row["n_frames"] is None else int(row["n_frames"]),
                          "effect_estimate": _num(effect_estimate), "effect_size_type": effect_size_type,
                          "method": method, "standard_error": _num(row["standard_error"]),
                          "ci_lower": _num(row["ci_lower"]), "ci_upper": _num(row["ci_upper"]),
                          "ci_level": _num(row["ci_level"]), "p_value": _num(row["p_value"])})

    def frame(self) -> pl.DataFrame:
        if not self.rows:
            return RESULTS.empty()
        df = pl.DataFrame([{**r, "p_adjusted": None} for r in self.rows],
                          schema={c: t for c, t in RESULTS.columns.items()})
        return finalize(df)


def finalize(df: pl.DataFrame) -> pl.DataFrame:
    """BH-adjust p-values within each non-empty ``adjustment_family`` (others: p_adjusted = p_value)."""
    if not df.height:
        return df
    p = df["p_value"].to_numpy().astype(float)
    fam = df["adjustment_family"].to_list()
    adj = np.full(p.shape, np.nan)
    for f in set(fam):
        idx = np.array([i for i, x in enumerate(fam) if x == f])
        adj[idx] = bh_fdr(p[idx]) if f else p[idx]
    return df.with_columns(pl.Series("p_adjusted", adj, dtype=pl.Float64).fill_nan(None)).select(list(RESULTS.columns))
