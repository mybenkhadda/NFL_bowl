"""Position-relative percentiles against a documented reference population. NumPy / Polars only.

Percentile      mid-rank: 100 · (#reference < x + ½ · #reference = x) / n. ORIENTED so that 100 is always the
                "more of the quality" end (a shorter time → a higher percentile); the raw value and unit stay beside it.
Reference       athletes of the same position group, of the eligible Combine cohorts, meeting the quality rule of the
                feature. Its size is reported with every percentile; below ``min_reference`` no percentile is given.
Isolation       an athlete's own value never sets the scale he is scored against: a reference member is scored
                LEAVE-ONE-OUT. ``fold_percentiles`` scores every athlete against the athletes of the OTHER folds only —
                the check of how a percentile generalises to a prospect outside the reference.
Missing         a missing measurement gives NO percentile (never the position average) and a stated reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...contracts import TableContract

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
FORTY, TEN = "cmb_forty", "cmb_ten_yd_split"
P95, V4_V8 = "F2.forty.p95_speed", "F6.forty.time_v4_to_v8_s"


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    label: str
    section: str                 # speed_capacity · speed_development
    unit: str
    higher_is_better: bool       # orientation of the RAW value: False = a smaller value is "more" (times)
    source: str                  # official · tracked
    role: str                    # headline · component
    meaning: str


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(FORTY, "official forty-yard dash", "speed_capacity", "s", False, "official", "headline",
                "time over 40 yards from a stationary start: the conventional summary of straight-line speed"),
    FeatureSpec(TEN, "official 10-yard split", "speed_capacity", "s", False, "official", "component",
                "time over the first 10 yards: early acceleration as timed officially"),
    FeatureSpec(P95, "tracked top speed (p95)", "speed_capacity", "yd/s", True, "tracked", "component",
                "95th percentile of tracked speed in the forty: the speed the athlete actually reached and held"),
    FeatureSpec(V4_V8, "speed development 4 → 8 yd/s", "speed_development", "s", False, "tracked", "headline",
                "time for tracked speed to rise from 4 to 8 yd/s: how quickly speed builds through the middle of the run"),
)
FEATURE = {f.feature_id: f for f in FEATURES}
REDUNDANCY_POLICY = (
    "Speed Capacity has ONE headline: the official forty percentile. The 10-yard split and the tracked top speed are shown "
    "beside it as components and are NOT averaged into it: within position they are strongly correlated with the forty "
    "(tracked top speed |ρ| ≈ 0.89, Milestone 04), so adding them would count the same quality several times. Their spread "
    "(`speed_agreement`) is reported instead: a large spread says the three measurements disagree about this athlete.")


@dataclass(frozen=True)
class ReferenceSpec:
    cohorts: tuple[int, ...] | None = None        # eligible draft years; None = every Combine cohort on disk
    positions: tuple[str, ...] = ("DB", "DL", "OL", "TE", "WR")
    min_reference: int = 30                       # athletes of a position needed before a percentile is shown
    min_repeat: int = 20                          # athletes with a repeat needed to estimate measurement error
    available_reliability: float = 0.70           # reliability of the athlete's value: at or above → available
    experimental_reliability: float = 0.50        # between the two → experimental; below → unavailable
    algorithm: str = "mid-rank percentile, oriented, leave-one-out for reference members"

    def params(self) -> dict[str, Any]:
        return asdict(self)


PCT = TableContract(
    "combine_percentiles", "Position-relative percentile of every scorecard feature with its uncertainty and availability.",
    {"nfl_id": I, "position_group": S, "draft_year": I, "section": S, "feature_id": S, "feature_label": S, "role": S, "value": F,
     "unit": S, "higher_is_better": B, "percentile": F, "pct_lower": F, "pct_upper": F, "pct_reliability_aware": F,
     "pct_fold_isolated": F, "n_reference": I, "in_reference": B, "n_attempts": I, "standard_error": F, "reliability": F,
     "measurement_uncertainty": S, "availability": S, "note": S}, key=("nfl_id", "feature_id"), area="athlete × feature",
    docs={"percentile": "oriented mid-rank percentile within position (100 = shortest time / highest speed), leave-one-out",
          "pct_lower": "95 % interval: the athlete's measurement error and the sampling of the reference population",
          "pct_reliability_aware": "percentile of the reliability-shrunk estimate among the shrunk estimates of the reference",
          "pct_fold_isolated": "percentile against the reference athletes of the OTHER folds only (generalisation check)",
          "standard_error": "standard error of the athlete's value from repeat-attempt variation (None: not estimable)",
          "availability": "available · experimental · unavailable (with the reason in note)"})


# ============================================================================== percentile algebra
def percentile_of(x: float, ref: np.ndarray) -> float:
    """Mid-rank percentile of ``x`` among ``ref`` (NaN when ``x`` is missing or the reference is empty)."""
    ref = ref[np.isfinite(ref)]
    if not np.isfinite(x) or ref.size == 0:
        return NAN
    return float(100.0 * ((ref < x).sum() + 0.5 * (ref == x).sum()) / ref.size)


def orient(p: np.ndarray | float, higher_is_better: bool) -> np.ndarray | float:
    """Percentile on the "more of the quality" scale: unchanged when a larger raw value is more, mirrored otherwise."""
    return p if higher_is_better else 100.0 - p


def position_percentiles(values: np.ndarray, position: np.ndarray, reference: np.ndarray, higher_is_better: bool,
                         min_reference: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """(oriented percentile, reference size) for every athlete against the reference athletes of HIS position.
    A reference member is scored leave-one-out; an athlete outside the reference is scored against all of it."""
    pct, n_ref = np.full(values.size, NAN), np.zeros(values.size, dtype=np.int64)
    for p in np.unique(position):
        k = np.flatnonzero(position == p)
        r = k[reference[k] & np.isfinite(values[k])]
        for i in k:
            ref = values[r[r != i]]
            n_ref[i] = ref.size
            if ref.size >= min_reference and np.isfinite(values[i]):
                pct[i] = orient(percentile_of(float(values[i]), ref), higher_is_better)
    return pct, n_ref


def fold_percentiles(values: np.ndarray, position: np.ndarray, reference: np.ndarray, fold: np.ndarray, higher_is_better: bool,
                     min_reference: int = 30) -> np.ndarray:
    """Oriented percentile of every athlete against the reference athletes of the OTHER folds of his position —
    a held-out prospect scored on a scale fitted without him or his fold. ``fold`` < 0 = the athlete has no fold."""
    pct = np.full(values.size, NAN)
    for i in range(values.size):
        if not np.isfinite(values[i]) or fold[i] < 0:
            continue
        ref = values[(position == position[i]) & reference & (fold != fold[i]) & (fold >= 0) & np.isfinite(values)]
        if ref.size >= max(5, int(0.6 * min_reference)):
            pct[i] = orient(percentile_of(float(values[i]), ref), higher_is_better)
    return pct


def reference_table(values: np.ndarray, position: np.ndarray, reference: np.ndarray, spec: FeatureSpec) -> pl.DataFrame:
    """Quantiles of the reference distribution per position (persisted in the manifest; drawn in the figures)."""
    rows = []
    for p in np.unique(position):
        v = values[(position == p) & reference & np.isfinite(values)]
        if v.size:
            q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
            rows.append({"feature_id": spec.feature_id, "position_group": str(p), "n_reference": int(v.size), "mean": float(v.mean()),
                         "sd": float(v.std(ddof=1)) if v.size > 1 else None, "q05": float(q[0]), "q25": float(q[1]), "q50": float(q[2]),
                         "q75": float(q[3]), "q95": float(q[4]), "unit": spec.unit, "higher_is_better": spec.higher_is_better})
    return pl.DataFrame(rows, schema={"feature_id": S, "position_group": S, "n_reference": I, "mean": F, "sd": F, "q05": F, "q25": F,
                                      "q50": F, "q75": F, "q95": F, "unit": S, "higher_is_better": B})


def reference_mask(position: np.ndarray, draft_year: np.ndarray, quality_ok: np.ndarray, spec: ReferenceSpec) -> np.ndarray:
    """Athletes forming the descriptive reference set: in-scope position, eligible cohort, quality rule met."""
    ok = np.isin(position, np.asarray(spec.positions)) & quality_ok
    if spec.cohorts is not None:
        ok &= np.isin(draft_year, np.asarray(spec.cohorts))
    return ok
