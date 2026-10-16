"""Study C — audit of the EXISTING mono-exponential sprint model (``features/sprint.py``); no new model.

What the existing implementation does (read from the source, not assumed):

* data        derived speed on valid frames of the longest segment, from one frame before speed first reaches
              ``onset_speed`` (0.5 yd/s) to the peak-speed frame. Times are real timestamps (``t_s``), not frame
              counts.
* estimation  for every (τ, t0) on a grid, vmax is the closed-form least-squares scale; the grid minimum is
              returned. No standard error, no profile, no weighting. Residuals of SG-smoothed speed are strongly
              autocorrelated, so R² close to 1 is expected and is not evidence of identifiability.
* bounds      τ ∈ [0.30, 2.50] s (step 0.01); t0 ∈ [−1.50, +0.50] s around the first fitted frame (step 0.02);
              ``fit_ok`` needs ≥ 15 frames, an interior optimum, 0 < vmax ≤ 12.5 and vmax ≤ 1.10 × observed peak.
* censoring   not part of the model. A recording that starts in motion is absorbed by a t0 before the first
              frame; a recording that ends near peak speed leaves the plateau unobserved, so vmax and τ trade off
              (larger vmax with larger τ fits the same rising curve) — the 1.10 guard limits, not removes, this.
* a0          vmax / τ = model acceleration at zero speed, at the model start t0. When t0 precedes the first
              frame it is an extrapolation outside the recording. It is a curve parameter, not a measured
              acceleration.

What this module adds, reusing ``fit_mono_exponential`` unchanged:

* identifiability   the set of (τ, t0) cells whose SSE is within ``sse_tolerance`` of the minimum. Its τ, a0 and
                    vmax ranges are an INDEX of how flat the fit surface is, not a confidence interval.
* refits            the same model on altered fit windows (initial frames removed, end shortened, window run to
                    the last frame, window started at a common speed) → sensitivity of each parameter to the
                    recording window, per attempt.
* ``a_at_ref``      (vmax − v_ref) / τ: the model acceleration at a speed inside the recorded range.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ...features.config import MovementConfig
from ...features.sprint import MAX_VMAX_EXTRAPOLATION, _grid, fit_mono_exponential
from ..contracts import TableContract
from .config import KEYS, VARIANTS, RescueConfig

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
MIN_WINDOW_FRAMES = 5      # below this the grid fit is not attempted at all (row kept, status says why)

FITS = TableContract(
    "sprint_model_fit_diagnostics", "Existing sprint model refitted on altered fit windows, with identifiability.",
    {"event_id": S, "nfl_id": I, "attempt": I, "variant": S, "status": S, "n_fit_frames": I, "fit_start_rel_s": F,
     "fit_end_rel_s": F, "start_speed": F, "vmax": F, "tau_s": F, "t0_rel_s": F, "a0": F, "a_at_ref": F, "rmse": F,
     "r2": F, "vmax_ratio": F, "at_grid_edge": B, "fit_ok": B, "a0_extrapolated": B, "extrap_gap_s": F, "tau_lo": F,
     "tau_hi": F, "tau_rel_width": F, "a0_rel_width": F, "vmax_rel_width": F, "region_at_edge": B, "tau_t0_corr": F,
     "weakly_identified": B, "d_vmax_rel": F, "d_tau_rel": F, "d_a0_rel": F, "d_a_at_ref_rel": F},
    key=("event_id", "nfl_id", "attempt", "variant"), area="attempt × fit-window variant",
    docs={"variant": "base = existing window · dropK = K initial frames removed · trunc_end = end shortened · to_end "
                     "= window run to the last frame · common_start = window starts at common_start_speed",
          "status": "ok · window_too_short (no fit attempted) · degenerate (no positive scale)",
          "t0_rel_s": "model start relative to the first recorded frame (negative = before the recording)",
          "a0": "vmax / τ: model acceleration at zero speed (extrapolated when a0_extrapolated)",
          "a_at_ref": "(vmax − model_ref_speed) / τ: model acceleration at a speed inside the recorded range",
          "extrap_gap_s": "time between the model start and the first fitted frame",
          "tau_lo": "smallest τ among grid cells with SSE ≤ (1 + sse_tolerance) × minimum (index, not a CI)",
          "tau_rel_width": "(tau_hi − tau_lo) / τ", "tau_t0_corr": "correlation of τ and t0 inside that region",
          "weakly_identified": "region touches the grid edge, or tau_rel_width > weak_tau_rel_width",
          "d_tau_rel": "(variant − base) / base for the same attempt"})


def fit_window(v: np.ndarray, mcfg: MovementConfig) -> tuple[int, int]:
    """Fit window of ``features.sprint.sprint_profile`` (first, last index, inclusive). Kept identical on
    purpose; ``tests/test_sprint_rescue.py`` checks the base refit against ``sprint_profile``."""
    ip = int(np.argmax(v))
    moving = np.nonzero(v >= mcfg.onset_speed)[0]
    i0 = max(0, int(moving[0]) - 1) if moving.size else 0
    return min(i0, ip), ip


def variant_window(t: np.ndarray, v: np.ndarray, i0: int, ip: int, variant: str, rcfg: RescueConfig
                   ) -> tuple[int, int]:
    if variant == "base":
        return i0, ip
    if variant.startswith("drop"):
        return i0 + int(variant[4:]), ip
    if variant == "to_end":
        return i0, t.size - 1
    if variant == "trunc_end":
        keep = np.nonzero(t <= t[ip] - rcfg.truncate_end_s + 1e-9)[0]
        return i0, int(keep[-1]) if keep.size else i0 - 1
    if variant == "common_start":
        fast = np.nonzero(v[i0: ip + 1] >= rcfg.common_start_speed)[0]
        return (i0 + int(fast[0]), ip) if fast.size else (ip + 1, ip)
    raise ValueError(f"unknown variant {variant!r}")


def sse_surface(t: np.ndarray, v: np.ndarray, tau_grid: np.ndarray, t0_grid: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """(SSE, vmax) on the whole τ × t0 grid — the surface whose minimum ``fit_mono_exponential`` returns."""
    g = 1.0 - np.exp(-np.clip(t[None, None, :] - t0_grid[None, :, None], 0.0, None) / tau_grid[:, None, None])
    sgv, sgg = (g * v).sum(-1), (g * g).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vmax = np.where(sgg > 0, sgv / sgg, np.nan)
        sse = np.where(sgg > 0, (v * v).sum() - sgv ** 2 / sgg, np.inf)
    return sse, vmax


def identifiability(sse: np.ndarray, vmax: np.ndarray, tau_grid: np.ndarray, t0_grid: np.ndarray, tol: float
                    ) -> dict[str, Any]:
    ok = np.isfinite(sse) & np.isfinite(vmax) & (vmax > 0)
    out = {"tau_lo": NAN, "tau_hi": NAN, "a0_lo": NAN, "a0_hi": NAN, "vmax_lo": NAN, "vmax_hi": NAN,
           "region_at_edge": None, "tau_t0_corr": NAN}
    if not ok.any():
        return out
    best = max(float(sse[ok].min()), 0.0)
    region = ok & (sse <= best * (1.0 + tol) + 1e-12)
    ti, tj = np.nonzero(region)
    taus, t0s, vm = tau_grid[ti], t0_grid[tj], vmax[region]
    a0 = vm / taus
    out.update(tau_lo=float(taus.min()), tau_hi=float(taus.max()), a0_lo=float(a0.min()), a0_hi=float(a0.max()),
               vmax_lo=float(vm.min()), vmax_hi=float(vm.max()),
               region_at_edge=bool((ti == 0).any() or (ti == tau_grid.size - 1).any() or (tj == 0).any()
                                   or (tj == t0_grid.size - 1).any()))
    if taus.size > 2 and taus.std() > 0 and t0s.std() > 0:
        out["tau_t0_corr"] = float(np.corrcoef(taus, t0s)[0, 1])
    return out


def fit_variant(t: np.ndarray, v: np.ndarray, variant: str, mcfg: MovementConfig, rcfg: RescueConfig
                ) -> dict[str, Any]:
    """One row of FITS (without keys and the vs-base columns) for one attempt and one fit-window variant."""
    row: dict[str, Any] = {c: None for c in FITS.columns if c not in (*KEYS, "variant")}
    i0, ip = fit_window(v, mcfg)
    a, b = variant_window(t, v, i0, ip, variant, rcfg)
    n = b - a + 1
    row.update(variant=variant, n_fit_frames=int(max(n, 0)), fit_ok=False)
    if n < MIN_WINDOW_FRAMES:
        row["status"] = "window_too_short"
        return row
    tf, vf = t[a: b + 1], v[a: b + 1]
    tau_grid, t0_grid = _grid(mcfg.tau_grid), tf[0] + _grid(mcfg.t0_offset_grid)
    fit = fit_mono_exponential(tf, vf, tau_grid, t0_grid)
    vmax, tau, t0, peak = fit["vmax"], fit["tau"], fit["t0"], float(v[ip])
    row.update(fit_start_rel_s=float(tf[0] - t[0]), fit_end_rel_s=float(tf[-1] - t[0]), start_speed=float(vf[0]),
               rmse=float(fit["rmse"]), r2=float(fit["r2"]) if np.isfinite(fit["r2"]) else None,
               at_grid_edge=bool(fit["at_edge"]))
    if not (np.isfinite(vmax) and vmax > 0):
        row["status"] = "degenerate"
        return row
    ratio = vmax / peak if peak > 0 else NAN
    idf = identifiability(*sse_surface(tf, vf, tau_grid, t0_grid), tau_grid, t0_grid, rcfg.sse_tolerance)
    a0 = vmax / tau
    tau_w = (idf["tau_hi"] - idf["tau_lo"]) / tau
    row.update(
        status="ok", vmax=float(vmax), tau_s=float(tau), t0_rel_s=float(t0 - t[0]), a0=float(a0),
        a_at_ref=float((vmax - rcfg.model_ref_speed) / tau) if vmax > rcfg.model_ref_speed else None,
        vmax_ratio=float(ratio) if np.isfinite(ratio) else None,
        fit_ok=bool(n >= mcfg.min_fit_frames and not fit["at_edge"] and vmax <= mcfg.max_model_vmax
                    and np.isfinite(ratio) and ratio <= MAX_VMAX_EXTRAPOLATION),
        a0_extrapolated=bool(t0 < t[0]), extrap_gap_s=float(max(tf[0] - t0, 0.0)),
        tau_lo=idf["tau_lo"], tau_hi=idf["tau_hi"], tau_rel_width=float(tau_w),
        a0_rel_width=float((idf["a0_hi"] - idf["a0_lo"]) / a0), vmax_rel_width=float((idf["vmax_hi"] - idf["vmax_lo"]) / vmax),
        region_at_edge=idf["region_at_edge"],
        tau_t0_corr=idf["tau_t0_corr"] if np.isfinite(idf["tau_t0_corr"]) else None,
        weakly_identified=bool(idf["region_at_edge"] or tau_w > rcfg.weak_tau_rel_width))
    return row


def fit_attempt(t: np.ndarray, v: np.ndarray, mcfg: MovementConfig, rcfg: RescueConfig) -> list[dict[str, Any]]:
    """All fit-window variants of one attempt; a failure of one variant never drops the attempt."""
    if t.size < MIN_WINDOW_FRAMES:
        return [{**{c: None for c in FITS.columns if c not in KEYS}, "variant": var, "status": "window_too_short",
                 "n_fit_frames": 0, "fit_ok": False} for var in VARIANTS]
    return [fit_variant(t, v, var, mcfg, rcfg) for var in VARIANTS]


def finalize_fits(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Typed FITS frame with each variant's relative change against the base fit of the same attempt."""
    if not rows:
        return FITS.empty()
    cols = [c for c in FITS.columns if not c.startswith("d_")]
    df = pl.DataFrame([{c: r.get(c) for c in cols} for r in rows], schema={c: FITS.columns[c] for c in cols})
    base = df.filter(pl.col("variant") == "base").select(*KEYS, *[pl.col(p).alias(f"_b_{p}") for p in
                                                                  ("vmax", "tau_s", "a0", "a_at_ref")])
    df = df.join(base, on=KEYS, how="left").with_columns(
        *[((pl.col(p) - pl.col(f"_b_{p}")) / pl.col(f"_b_{p}")).alias(f"d_{p}_rel")
          for p in ("vmax", "tau_s", "a0", "a_at_ref")])
    return df.rename({"d_tau_s_rel": "d_tau_rel"}).select(list(FITS.columns))


def variant_sensitivity(fits: pl.DataFrame) -> pl.DataFrame:
    """Median |relative change| of each parameter per variant, over attempts where base and variant both fit."""
    d = fits.filter((pl.col("variant") != "base") & (pl.col("status") == "ok") & pl.col("d_tau_rel").is_not_null())
    if not d.height:
        return pl.DataFrame(schema={"variant": S, "n_attempts": pl.UInt32(), "share_fit_ok": F,
                                    **{f"median_abs_{c}": F for c in ("d_vmax_rel", "d_tau_rel", "d_a0_rel",
                                                                      "d_a_at_ref_rel")}})
    return (d.group_by("variant").agg(
        pl.len().alias("n_attempts"), pl.col("fit_ok").mean().alias("share_fit_ok"),
        *[pl.col(c).abs().median().alias(f"median_abs_{c}") for c in ("d_vmax_rel", "d_tau_rel", "d_a0_rel",
                                                                      "d_a_at_ref_rel")]).sort("variant"))
