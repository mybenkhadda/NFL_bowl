"""Explicit, versioned parameters for kinematic feature extraction.

Every number that changes a feature value lives here. :meth:`KinematicsConfig.params` is part
of each persisted dataset's input fingerprint, so changing any threshold invalidates the cached
Parquet outputs (see ``research.artifacts.ArtifactStore.status``).

Thresholds marked PROVISIONAL have no measured justification yet; the M1 diagnostics
(``event_threshold_sensitivity.csv``, ``segment_quality_summary.csv``) exist to calibrate them.
Units: yards, seconds, yd/s, yd/s², degrees, deg/s.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from . import FEATURES_VERSION


@dataclass(frozen=True)
class SmoothingSpec:
    """Savitzky–Golay window (frames, odd) and polynomial order applied to x(t), y(t)."""

    window: int = 7
    polyorder: int = 2

    def __post_init__(self) -> None:
        if self.window % 2 == 0 or self.window < 3:
            raise ValueError(f"window must be odd and >= 3, got {self.window}")
        if not 2 <= self.polyorder < self.window:
            raise ValueError(f"polyorder must be in [2, window), got {self.polyorder}")

    @property
    def label(self) -> str:
        return f"sg{self.window}p{self.polyorder}"


@dataclass(frozen=True)
class EventRule:
    """One threshold set for one event type.

    braking / acceleration: ``threshold`` = |tangential acceleration| in yd/s²,
        ``min_change`` = speed lost / gained over the event (yd/s),
        ``min_speed`` = speed just before onset (braking only; 0 for acceleration).
    turn: ``threshold`` = |yaw rate| in deg/s, ``min_change`` = |cumulative heading change| (deg),
        ``min_speed`` = speed required on every frame of the turn (yd/s).
    reversal (low-speed change of direction): ``threshold`` unused (speed below
        ``min_heading_speed`` defines the run), ``min_change`` = |heading out − heading in| (deg),
        ``max_duration_s`` = longest low-speed window still counted as a cut, not a stop.
    """

    name: str
    threshold: float
    min_duration_s: float
    min_change: float
    min_speed: float = 0.0
    max_duration_s: float | None = None


def _rules(*rows: tuple) -> tuple[EventRule, ...]:
    return tuple(EventRule(*r) for r in rows)


@dataclass(frozen=True)
class TierRules:
    """Trajectory quality tiers (PROVISIONAL — calibrate on segment_quality_summary.csv).

    reject: no smoothable frame, or any smoothed speed / acceleration above plausibility limits
    C:      smoothable coverage < c_min_coverage, or position residual RMS > c_max_resid_rms
    B:      split into > 1 segment, residual RMS > b_max_resid_rms, or median |derived − vendor
            speed| > b_max_speed_mad
    A:      none of the above
    """

    c_min_coverage: float = 0.80
    c_max_resid_rms: float = 0.15       # yd
    b_max_resid_rms: float = 0.06       # yd
    b_max_speed_mad: float = 0.25       # yd/s


@dataclass(frozen=True)
class KinematicsConfig:
    version: str = FEATURES_VERSION
    # sampling / integrity (defaults mirror Phase01Config.tracking; use from_tracking())
    nominal_interval_ms: int = 100
    gap_factor: float = 1.5
    offgrid_tolerance: float = 0.25          # |Δt − nominal| > tol × nominal → off-grid frame
    max_step_yds: float = 1.5                # per nominal interval → coordinate jump → new segment
    max_speed_yds: float = 12.5
    max_accel_yds2: float = 15.0
    # smoothing
    smoothing: SmoothingSpec = field(default_factory=SmoothingSpec)
    sensitivity_smoothing: tuple[SmoothingSpec, ...] = (
        SmoothingSpec(5, 2), SmoothingSpec(7, 2), SmoothingSpec(9, 2), SmoothingSpec(11, 3))
    # validity of direction-dependent quantities
    min_tangent_speed: float = 0.1           # yd/s; below → tangential/lateral accel undefined
    min_heading_speed: float = 1.0           # yd/s; below → heading, yaw rate, curvature undefined
    # event threshold sets (PROVISIONAL); "default" is the primary set, others are sensitivity
    braking_rules: tuple[EventRule, ...] = _rules(
        ("lenient", 1.5, 0.2, 0.75, 1.5), ("default", 2.5, 0.3, 1.0, 2.0), ("strict", 3.5, 0.4, 1.5, 3.0))
    accel_rules: tuple[EventRule, ...] = _rules(
        ("lenient", 1.5, 0.2, 0.75, 0.0), ("default", 2.5, 0.3, 1.0, 0.0), ("strict", 3.5, 0.4, 1.5, 0.0))
    turn_rules: tuple[EventRule, ...] = _rules(
        ("lenient", 45.0, 0.2, 20.0, 1.5), ("default", 75.0, 0.3, 30.0, 2.0), ("strict", 120.0, 0.3, 45.0, 2.5))
    reversal_rules: tuple[EventRule, ...] = _rules(
        ("lenient", 0.0, 0.0, 60.0, 0.0, 2.0), ("default", 0.0, 0.0, 90.0, 0.0, 1.5),
        ("strict", 0.0, 0.0, 135.0, 0.0, 1.0))
    primary_rule: str = "default"
    tiers: TierRules = field(default_factory=TierRules)
    # execution (not part of the scientific definition, but recorded)
    nfl_batch_games: int = 25

    def __post_init__(self) -> None:
        for kind in ("braking_rules", "accel_rules", "turn_rules", "reversal_rules"):
            names = [r.name for r in getattr(self, kind)]
            if self.primary_rule not in names:
                raise ValueError(f"{kind} lacks the primary rule {self.primary_rule!r}")
            if len(set(names)) != len(names):
                raise ValueError(f"{kind} has duplicate rule names {names}")

    # ------------------------------------------------------------------ derived
    @property
    def dt_s(self) -> float:
        return self.nominal_interval_ms / 1000.0

    @property
    def min_segment_frames(self) -> int:
        """Segments shorter than the smoothing window get no derivatives (flagged, not dropped)."""
        return self.smoothing.window

    def rules(self, event_type: str) -> tuple[EventRule, ...]:
        return {"braking": self.braking_rules, "acceleration": self.accel_rules,
                "turn": self.turn_rules, "reversal": self.reversal_rules}[event_type]

    def with_smoothing(self, spec: SmoothingSpec) -> "KinematicsConfig":
        return replace(self, smoothing=spec)

    @classmethod
    def from_tracking(cls, tracking: Any, **overrides: Any) -> "KinematicsConfig":
        """Inherit integrity thresholds from ``Phase01Config.tracking`` so Phase 02 flags and
        Phase 03 segmentation use the same physical limits."""
        base = {k: getattr(tracking, k) for k in
                ("nominal_interval_ms", "gap_factor", "max_step_yds", "max_speed_yds", "max_accel_yds2")
                if hasattr(tracking, k)}
        return cls(**{**base, **overrides})

    def params(self) -> dict[str, Any]:
        """JSON-able parameter record used in dataset fingerprints (execution knobs excluded)."""
        d = asdict(self)
        d.pop("nfl_batch_games", None)
        return d


# ======================================================================================
# Phase 03 · M2 — movement-feature parameters (attempt / event / athlete level).
# Kept separate from KinematicsConfig so that changing an M2 parameter invalidates only the M2
# datasets, never the (expensive) M1 kinematics. Units: yd, s, yd/s, yd/s², deg.
# Values marked PROVISIONAL are conventions, not calibrated optima; the M2 analytics export the
# sensitivity evidence needed to revisit them.
# ======================================================================================
M2_VERSION = "2"   # 2: M2 hardening (sprint fit window, trk_* names, end censoring, active-window
                   #    braking events, reliability caps/labels) — invalidates v1 M2 outputs only


@dataclass(frozen=True)
class MovementConfig:
    version: str = M2_VERSION
    event_rule: str = "default"                  # M1 threshold set used for event-based features
    usable_tiers: tuple[str, ...] = ("A", "B")   # trajectories entering features / analyses
    # movement window ----------------------------------------------------------------
    onset_speed: float = 0.5                     # yd/s, sustained for onset_frames → movement onset
    onset_speed_fast: float = 2.0                # yd/s, alternative onset (robust to pre-start drift)
    onset_frames: int = 3
    # temporal phases (PROVISIONAL) ----------------------------------------------------
    stationary_speed: float = 0.5                # yd/s
    phase_accel: float = 1.0                     # |a_tan| below → "steady"
    min_phase_frames: int = 3                    # shorter phase runs are not counted as transitions
    # speed / acceleration -----------------------------------------------------------
    speed_thresholds: tuple[float, ...] = (4.0, 6.0, 8.0)   # yd/s, time-above features
    high_speed_frac: float = 0.90                # sustained high speed = time ≥ 90 % of attempt peak
    initial_window_s: float = 1.0                # initial acceleration window after onset
    sustain_window_s: float = 0.5                # rolling window for sustained acceleration
    peak_edge_frames: int = 3                    # peak within last k frames → censored (window cut)
    # change of direction (PROVISIONAL) ------------------------------------------------
    cod_min_angle: float = 60.0                  # deg, |heading change| of a turn/reversal to count
    cod_sharp_angle: float = 135.0               # deg
    cod_merge_gap_s: float = 0.3                 # turn/reversal events closer than this = one COD
    cod_approach_s: float = 1.0                  # approach speed = max speed in this window before
    cod_pre_s: float = 0.5                       # approach-deceleration window before the COD
    cod_post_s: float = 0.5                      # exit-acceleration window after the COD
    cod_recover_frac: float = 0.8                # COD time = time below 80 % of approach speed
    cod_recover_max_s: float = 2.0               # search limit on each side of the COD
    # sprint (forty-yard dash) ---------------------------------------------------------
    sprint_drills: tuple[str, ...] = ("FORTY_YARD_DASH",)
    split_distances: tuple[float, ...] = (10.0, 20.0, 40.0)   # yd
    split_tolerance_yds: float = 1.0             # run shorter than split − tol → split undefined
    tau_grid: tuple[float, float, float] = (0.30, 2.50, 0.01)          # s (start, stop, step)
    t0_offset_grid: tuple[float, float, float] = (-1.50, 0.50, 0.02)   # s, relative to first frame
    min_fit_frames: int = 15
    max_model_vmax: float = 12.5                 # yd/s, = KinematicsConfig.max_speed_yds plausibility limit
    # recording truncation -------------------------------------------------------------
    end_censor_speed: float = 2.0                # yd/s; still this fast on the last frame → end truncated
    # reliability --------------------------------------------------------------------
    min_attempts: int = 2                        # per athlete, to enter reliability
    max_attempts: int = 4                        # first k chronological attempts per athlete (no domination)
    min_athletes: int = 10                       # per drill × feature, to report an ICC
    min_athletes_label: int = 20                 # below this an ICC is reported but never labelled
    max_modal_share: float = 0.8                 # ≥ 80 % identical values → degenerate (e.g. counts of 0)
    n_boot: int = 1000
    seed: int = 20261008

    def params(self) -> dict[str, Any]:
        return asdict(self)


# ======================================================================================
# Phase 03 · M1.1 — snap-anchored NFL play windows (Workstream C foundation).
# Separate from KinematicsConfig / MovementConfig: changing a window definition invalidates only
# the window datasets (``research.nfl_windows``), never M1 kinematics or M2 Combine features.
# Every value is PROVISIONAL (a convention, not a calibrated optimum) unless marked otherwise.
# Units: s, yd, yd/s.
# ======================================================================================
PLAY_WINDOWS_VERSION = "1"


@dataclass(frozen=True)
class WindowSpec:
    """A closed time interval [start, end] on one play, both ends given as ``ref + offset_s``.

    refs: ``play_start`` (first tracked frame of the play), ``snap``, ``release`` (first release
    event at/after the snap — pass / handoff / run, see ``PlayWindowConfig.release_events``) and
    ``play_end`` (last tracked frame). A window whose reference is undefined on a play (no snap, no
    release) is absent for that play — never approximated.
    """

    name: str
    start_ref: str
    start_offset_s: float
    end_ref: str
    end_offset_s: float
    phase: str                      # "pre_snap" | "post_snap" — pre-snap analysis is always kept separate

    def __post_init__(self) -> None:
        refs = ("play_start", "snap", "release", "play_end")
        if self.start_ref not in refs or self.end_ref not in refs:
            raise ValueError(f"{self.name}: refs must be in {refs}")
        if self.phase not in ("pre_snap", "post_snap"):
            raise ValueError(f"{self.name}: phase must be pre_snap or post_snap")


@dataclass(frozen=True)
class PlayWindowConfig:
    version: str = PLAY_WINDOWS_VERSION
    # anchors --------------------------------------------------------------------------
    # Tag vocabulary observed in reports/tables/profile_top_values.csv (Phase 01 profile, sampled):
    # ball_snap, pass_forward, handoff, run, qb_sack, play_action, man_in_motion, … The first name
    # in ``snap_events`` that occurs on a play defines the snap; the others are fallbacks that were
    # NOT observed in the profile sample and are listed only so a vocabulary change is caught.
    snap_events: tuple[str, ...] = ("ball_snap", "snap_direct", "autoevent_ballsnap")
    release_events: tuple[str, ...] = ("pass_forward", "pass_shovel", "handoff", "run", "qb_sack", "qb_strip_sack",
                                       "qb_spike", "qb_kneel")
    season_types: tuple[str, ...] = ("REG",)     # PRE / POST excluded by default (different player usage)
    # windows (relative to the anchors above) ----------------------------------------------
    windows: tuple[WindowSpec, ...] = (
        WindowSpec("pre_snap", "play_start", 0.0, "snap", -0.1, "pre_snap"),     # last frame strictly before the snap
        WindowSpec("post_0_1p5", "snap", 0.0, "snap", 1.5, "post_snap"),
        WindowSpec("post_0_3", "snap", 0.0, "snap", 3.0, "post_snap"),
        WindowSpec("snap_to_release", "snap", 0.0, "release", 0.0, "post_snap"),
        WindowSpec("post_snap_all", "snap", 0.0, "play_end", 0.0, "post_snap"),
    )
    frame_tolerance_s: float = 0.05              # half a nominal frame: membership tolerance at window edges
    min_window_frames: int = 5                   # fewer observed frames → window features null (row kept)
    min_coverage: float = 0.8                    # observed / expected frames below this → flagged, not dropped
    # window features ---------------------------------------------------------------------
    sustain_window_s: float = 0.5                # = MovementConfig.sustain_window_s (Combine counterpart)
    speed_thresholds: tuple[float, ...] = (4.0, 6.0, 8.0)   # = MovementConfig.speed_thresholds
    getoff_disp_yds: float = 1.0                 # raw displacement from the snap position → "first-step" time
    getoff_max_s: float = 1.5                    # search limit for the first-step time after the snap

    def __post_init__(self) -> None:
        names = [w.name for w in self.windows]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate window names {names}")
        if not self.snap_events:
            raise ValueError("snap_events must not be empty")

    def window(self, name: str) -> WindowSpec:
        for w in self.windows:
            if w.name == name:
                return w
        raise KeyError(name)

    def params(self) -> dict[str, Any]:
        return asdict(self)
