"""Study parameters. Each numeric field is documented in ``DOCS`` as
``established`` (reused M1/M2 value), ``proposed`` (uncalibrated study default — continuous metrics
are always exported alongside, so cut-offs can be changed without recomputation) or ``execution``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import polars as pl

from ...features.config import KinematicsConfig

ESTABLISHED, PROPOSED, EXECUTION = "established", "proposed", "execution"
STUDIES_VERSION = "1"
_K = KinematicsConfig()


@dataclass(frozen=True)
class QualityStudyConfig:
    """Study A — quality calibration (A1) and vendor-vs-derived agreement (A2)."""

    screen_quantile: float = 0.99               # dimension flagged above the domain × stratum quantile
    min_group_trajectories: int = 20            # smaller groups → result_status insufficient_data
    min_athletes: int = 10
    n_boot: int = 1000                          # cluster bootstrap for ratio statistics
    n_boot_generic: int = 300                   # cluster bootstrap for rank statistics (Cliff's δ, Spearman)
    duration_bins_s: tuple[float, ...] = (0.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0)
    speed_bands: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 7.0, 9.0)     # yd/s lower edges (Bland–Altman mean)
    rel_error_min_speed: float = 1.0            # yd/s; relative error only above this vendor speed
    extreme_accel_yds2: float = 10.0            # frame |a| above this counts as an acceleration outlier
    min_frames_trajectory: int = 20             # trajectory-level vendor error needs this many valid frames
    hist_speed_max: float = 12.0                # yd/s, 2-D histogram range
    hist_bin: float = 0.1                       # yd/s
    resid_range: float = 1.0                    # yd/s, residual histogram range ±
    nfl_games_per_season: int = 8               # deterministic NFL frame sample (as in M1 diagnostics)
    n_examples: int = 3


@dataclass(frozen=True)
class IntegrityConfig:
    """Study A0 — exact drill-down of failed Phase 01 data-quality checks."""

    layer: str = "bronze"                       # the layer the Phase 01 checks ran on (drill-down re-measures it)
    results: tuple[str, ...] = ("FAIL", "WARN", "ERROR")   # Phase 01 results that are drilled down
    n_examples: int = 10                        # example keys stored per failed check
    blocking_severities: tuple[str, ...] = ("CRITICAL", "HIGH")


@dataclass(frozen=True)
class SignalStudyConfig:
    """Study A3 — derivative-signal reliability, filter-boundary contamination, tier discrimination."""

    unreliable_quantile: float = 0.95           # target "unreliable" = metric above the domain × stratum quantile
    min_group_trajectories: int = 20
    min_athletes: int = 10
    n_boot: int = 300                           # athlete-cluster bootstrap (AUC, shares)
    nfl_games_per_season: int = 8               # = QualityStudyConfig (same deterministic NFL sample)


@dataclass(frozen=True)
class StudiesConfig:
    version: str = STUDIES_VERSION
    seed: int = 20261008
    sample_athletes: int | None = None
    quality: QualityStudyConfig = field(default_factory=QualityStudyConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    signal: SignalStudyConfig = field(default_factory=SignalStudyConfig)

    def section(self, name: str) -> dict[str, Any]:
        return {"version": self.version, "seed": self.seed, "sample_athletes": self.sample_athletes,
                name: asdict(getattr(self, name))}


DOCS: dict[str, dict[str, tuple[str, str, str]]] = {
    "quality": {
        "screen_quantile": ("quantile", PROPOSED, "screening only — flags the extreme 1 % per dimension; the "
                                                  "percentile rank itself is exported"),
        "min_group_trajectories": ("trajectories", PROPOSED, "below this a proportion / effect is not interpreted"),
        "min_athletes": ("athletes", PROPOSED, "clusters needed for a cluster-bootstrap interval"),
        "n_boot": ("resamples", EXECUTION, "ratio statistics (cheap)"),
        "n_boot_generic": ("resamples", EXECUTION, "rank statistics on ≤ 315k trajectories (cost-bounded)"),
        "duration_bins_s": ("s", PROPOSED, "covers Combine drills (3–10 s) and NFL player-plays (≤ 40 s)"),
        "speed_bands": ("yd/s", PROPOSED, "walking / jogging / running / fast / sprint bands"),
        "rel_error_min_speed": ("yd/s", PROPOSED, "relative error is unstable near zero speed"),
        "extreme_accel_yds2": ("yd/s²", PROPOSED, "two-thirds of KinematicsConfig.max_accel_yds2 = 15 (established "
                                                   "plausibility limit)"),
        "min_frames_trajectory": ("frames", PROPOSED, "2 s at 10 Hz"),
        "hist_speed_max": ("yd/s", ESTABLISHED, f"KinematicsConfig.max_speed_yds = {_K.max_speed_yds}"),
        "hist_bin": ("yd/s", EXECUTION, "plot resolution"),
        "resid_range": ("yd/s", PROPOSED, "M1 p95 |Δ| ≤ 0.37 yd/s in every group → ±1 covers the bulk; tails "
                                          "beyond are counted in edge bins"),
        "nfl_games_per_season": ("games", ESTABLISHED, "KinematicDiagnostics default (deterministic spacing)"),
        "n_examples": ("trajectories", EXECUTION, "examples per figure"),
    },
    "integrity": {
        "layer": ("layer", ESTABLISHED, "Phase 01 validation ran on bronze (data_quality_results.csv sql column)"),
        "results": ("labels", ESTABLISHED, "every non-PASS Phase 01 result is drilled down; INFO rows are not checks"),
        "n_examples": ("keys", EXECUTION, "example keys per failed check (inspection only)"),
        "blocking_severities": ("labels", PROPOSED, "a data defect at these severities blocks the research population "
                                                    "unless resolved downstream (report §13 Priority 0)"),
    },
    "signal": {
        "unreliable_quantile": ("quantile", PROPOSED, "defines the 'unreliable signal' target for discrimination "
                                                      "diagnostics; continuous metrics are exported, so the target "
                                                      "can be redefined without recomputation"),
        "min_group_trajectories": ("trajectories", PROPOSED, "below this a share / AUC is not interpreted"),
        "min_athletes": ("athletes", PROPOSED, "clusters needed for a cluster-bootstrap interval"),
        "n_boot": ("resamples", EXECUTION, "AUC bootstrap is O(n log n) per resample on ≤ 6k Combine trajectories"),
        "nfl_games_per_season": ("games", ESTABLISHED, "QualityStudyConfig.nfl_games_per_season (same sample as A2)"),
    },
}


def threshold_table(cfg: StudiesConfig | None = None) -> pl.DataFrame:
    cfg = cfg or StudiesConfig()
    rows = []
    for section, docs in DOCS.items():
        sub = getattr(cfg, section)
        for f in fields(sub):
            unit, source, why = docs[f.name]
            rows.append({"study_section": section, "name": f.name, "value": str(getattr(sub, f.name)), "unit": unit,
                         "provenance": source, "rationale": why})
    return pl.DataFrame(rows, schema={c: pl.String for c in ("study_section", "name", "value", "unit", "provenance",
                                                              "rationale")})
