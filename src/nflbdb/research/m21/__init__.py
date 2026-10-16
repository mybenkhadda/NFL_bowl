"""Phase 03 · M2.1 — validation and research hardening of the M2 Combine movement features.

Reads persisted M1/M2 artifacts only and writes to a dedicated namespace (``phase_03/m2_1``) so no
M1/M2 output is ever overwritten. One module per workstream:

    config.py       thresholds (established vs proposed) + threshold documentation
    io.py           M21Store: table contracts, CSV/Parquet export, run cache, figure manifest
    plotting.py     figure helpers (caption-safe layout, equal-aspect panels without shared axes)
    duplicates.py   Workstream A — duplicate / near-duplicate attempt audit

AUTHORED, NOT EXECUTED.
"""

M21_VERSION = "1"
