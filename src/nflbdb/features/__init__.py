"""Phase 03 — kinematic feature engineering (pure computation, no I/O).

Modules
    config             KinematicsConfig, SmoothingSpec, EventRule, TierRules (all thresholds explicit)
    signal_processing  Savitzky–Golay coefficients + segment-aware filtering (NumPy, vectorised)
    trajectory         trajectory keys, drill canonicalisation, ordering / dedup / segmentation
    kinematics         smoothed position → velocity, vector acceleration, tangential / lateral
                       acceleration, yaw rate, curvature, heading
    events             braking, acceleration, turn and low-speed reversal events (threshold sets)
    segment_quality    per-segment and per-trajectory quality metrics + rule-based tiers
    feature_registry   FeatureDef registry → feature_dictionary.csv (M1 + M2_REGISTRY, HIGHER_IS_BETTER)
  M2 (attempt level; parameters in config.MovementConfig, version M2_VERSION)
    temporal           movement window (onset / censoring), phases, thirds          (Family E)
    acceleration       speed and tangential / vector acceleration features          (Family A)
    braking            frame-level deceleration + braking-event summaries           (Family B)
    agility            merged COD events, approach / exit / recovery measures       (Family C)
    efficiency         path efficiency, principal-axis lateral movement             (Family D)
    sprint             mono-exponential sprint model + tracked splits (forty)       (Family A/E)
    reliability        ICC(1) + athlete bootstrap, typical error, rank stability    (Family G)

Persistence and orchestration live in ``nflbdb.research.kinematic_build`` (M1) and
``nflbdb.research.combine_movement`` (M2); diagnostics, CSVs and figures in
``nflbdb.research.kinematic_diagnostics`` (M1) and ``nflbdb.research.combine_analytics`` (M2).

Invariants enforced here (and tested in ``tests/test_phase03_kinematics.py``):

* derivatives are computed strictly inside a *segment* — a run of frames of one trajectory
  (Combine attempt / NFL player-play) with no gap, duplicate timestamp, coordinate jump or
  drill change; nothing is ever filtered across play or attempt boundaries;
* heading changes are obtained from the cross product of velocity and acceleration (yaw rate),
  so they never suffer from 359° → 1° wraparound;
* heading, yaw rate and curvature are undefined (null) below ``min_heading_speed``.
"""

FEATURES_VERSION = "1"
