"""M2.1 thresholds — every number that changes an M2.1 flag or classification lives here.

Each numeric field is documented in ``THRESHOLD_DOCS`` with its unit, its provenance and a
rationale:

* ``established`` — the value is already used by M1/M2 (``KinematicsConfig`` / ``MovementConfig``)
  and is reused unchanged so that M2.1 judges M2 by its own definitions;
* ``proposed``    — a diagnostic default introduced in M2.1. It is NOT calibrated: every
  proposed threshold is accompanied by a continuous metric in the outputs so that a different
  cut-off can be applied afterwards without recomputation (sensitivity, not pass/fail).

``threshold_table()`` exports the documentation (``m21_thresholds.csv``); a test asserts that
every numeric field is documented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import polars as pl

from ...features.config import KinematicsConfig, MovementConfig
from . import M21_VERSION

ESTABLISHED, PROPOSED, EXECUTION = "established", "proposed", "execution"

_K, _M = KinematicsConfig(), MovementConfig()


@dataclass(frozen=True)
class DuplicateAuditConfig:
    """Workstream A — attempt integrity (exact / near-duplicate / reused trajectories)."""

    coord_quantum_yds: float = 0.01          # raw x, y are compared on this integer grid
    kgram_frames: int = 10                   # length of a raw-displacement window ("k-gram")
    kgram_min_motion_yds: float = 1.0        # windows covering less ground are ignored (standing still ≠ reuse)
    align_speed: float = _M.onset_speed_fast # yd/s, sustained for align_frames → alignment anchor
    align_frames: int = _M.onset_frames
    max_lag_s: float = 1.0                   # residual lag search around the onset anchor (± s)
    grid_dt_s: float = _K.dt_s               # comparison grid = nominal sampling interval
    min_overlap_frac: float = 0.8            # aligned overlap / shorter attempt duration to compare
    exact_shared_frac: float = 0.95          # shared (ts, x, y) frames / shorter attempt → exact duplicate
    reuse_kgram_frac: float = 0.5            # shared moving k-grams / shorter attempt → reused sequence
    reuse_shared_frame_frac: float = 0.2     # shared (ts, x, y) frames / shorter attempt → reused sequence
    near_dup_pos_rmse_yds: float = 0.05      # rigid-aligned raw position RMSE → near duplicate (AND speed)
    near_dup_speed_rmse: float = 0.05        # yd/s, aligned smoothed-speed RMSE → near duplicate (AND position)
    consistent_speed_rmse: float = 0.30      # yd/s, below = "genuinely consistent" (natural, not suspicious)
    cross_athlete_min_frames: int = 20       # identical relative paths across athletes count only if this long
    cross_athlete_min_path_yds: float = 2.0  # … and covering at least this distance
    zero_sd_abs: float = 1e-6                # within-athlete SD ≤ this (or ≤ rel × |median|) → zero variance
    zero_sd_rel: float = 1e-4
    quantized_unique_frac: float = 0.10      # distinct values / values below this → feature on a coarse grid
    max_attempts_per_group: int = 8          # bounded pairwise work per athlete × drill (extra attempts flagged)


@dataclass(frozen=True)
class M21Config:
    version: str = M21_VERSION
    seed: int = _M.seed
    sample_athletes: int | None = None       # None = full run; an int = deterministic athlete sample
    duplicates: DuplicateAuditConfig = field(default_factory=DuplicateAuditConfig)

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def section(self, name: str) -> dict[str, Any]:
        """Parameters of one workstream plus the shared execution settings (run-cache fingerprint)."""
        return {"version": self.version, "seed": self.seed, "sample_athletes": self.sample_athletes,
                name: asdict(getattr(self, name))}


# field → (unit, provenance, rationale)
THRESHOLD_DOCS: dict[str, dict[str, tuple[str, str, str]]] = {
    "duplicates": {
        "coord_quantum_yds": ("yd", PROPOSED, "assumed vendor coordinate precision (2 decimals); verify on raw x, y"),
        "kgram_frames": ("frames", PROPOSED, "1 s at 10 Hz — long enough that coincidental equality is implausible"),
        "kgram_min_motion_yds": ("yd", PROPOSED, "excludes stationary windows whose displacements are all ~0"),
        "align_speed": ("yd/s", ESTABLISHED, "MovementConfig.onset_speed_fast (robust to pre-start drift)"),
        "align_frames": ("frames", ESTABLISHED, "MovementConfig.onset_frames"),
        "max_lag_s": ("s", PROPOSED, "onset anchors of two attempts can differ by a few frames"),
        "grid_dt_s": ("s", ESTABLISHED, "KinematicsConfig.nominal_interval_ms / 1000"),
        "min_overlap_frac": ("fraction", PROPOSED, "pairs with little common aligned time are 'not_comparable'"),
        "exact_shared_frac": ("fraction", PROPOSED, "identical timestamps AND coordinates on ≥ 95 % of frames"),
        "reuse_kgram_frac": ("fraction", PROPOSED, "half of the moving 1-s displacement windows shared"),
        "reuse_shared_frame_frac": ("fraction", PROPOSED, "a fifth of the frames identical in time and position"),
        "near_dup_pos_rmse_yds": ("yd", PROPOSED, "below M1 tier-B residual-noise limit (TierRules.b_max_resid_rms "
                                                    "= 0.06 yd): two real runs should not agree more closely than "
                                                    "the measurement noise"),
        "near_dup_speed_rmse": ("yd/s", PROPOSED, "≈ 1 % of forty peak speed; far below between-attempt variation"),
        "consistent_speed_rmse": ("yd/s", PROPOSED, "example selection only; never used to flag"),
        "cross_athlete_min_frames": ("frames", PROPOSED, "short static recordings can coincide by chance"),
        "cross_athlete_min_path_yds": ("yd", PROPOSED, "as above, for distance"),
        "zero_sd_abs": ("feature unit", PROPOSED, "float32 storage noise floor"),
        "zero_sd_rel": ("fraction", PROPOSED, "relative float32 noise floor"),
        "quantized_unique_frac": ("fraction", PROPOSED, "grid-searched fit parameters (tau step 0.01 s, t0 step "
                                                        "0.02 s) tie by construction; explains zero SD, not flags it"),
        "max_attempts_per_group": ("attempts", EXECUTION, "bounds pairwise work; larger groups are flagged"),
    },
}


def threshold_table(cfg: M21Config | None = None) -> pl.DataFrame:
    """One row per threshold: workstream, name, value, unit, provenance, rationale."""
    cfg = cfg or M21Config()
    rows = []
    for section, docs in THRESHOLD_DOCS.items():
        sub = getattr(cfg, section)
        for f in fields(sub):
            unit, source, why = docs[f.name]
            rows.append({"workstream": section, "name": f.name, "value": str(getattr(sub, f.name)), "unit": unit,
                         "provenance": source, "rationale": why})
    return pl.DataFrame(rows, schema={"workstream": pl.String, "name": pl.String, "value": pl.String,
                                      "unit": pl.String, "provenance": pl.String, "rationale": pl.String})
