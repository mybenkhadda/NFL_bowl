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
from typing import Any

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


__all__ = ["TARGETS", "TargetSpec", "assert_same_target", "frame_fingerprint", "target_frame", "target_summary"]
