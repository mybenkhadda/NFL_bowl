# Active task — Milestone 12 (advanced movement intelligence)

_Updated 2026-10-15. Code-authoring only: Claude executes nothing._

**Status:** all modules, 8 notebooks, 12 reports and the test wrapper AUTHORED. 12.0 executed once by the user → two issues found and fixed in code (not yet re-run):
1. D-12-1 — Combine a(v) frame binning gave 0 shared speed bins at 10 Hz → interpolation at bin mid-speeds (`foundation.CURVE_RULE = rising_phase_interp_v2`, `min_frames_bin_combine = 0`); 12.0 rebuilds the curve tables automatically.
2. Synthetic checks: fixed two test bugs (dtype, NaN comparison) and one estimator bug (12B RC / SIMEX standardized by the observed-mean SD instead of the latent SD).

**User execution order:** 12.0 (re-run) → 12C → 12B → 12A → 12D → 12E → 12F (optional; needs PyTorch) → 12G.
**Send back:** the items printed by the last cell of each notebook; first failures are most likely Polars API details or check tolerances.

Milestone 10 (authored, not executed) is unchanged.
