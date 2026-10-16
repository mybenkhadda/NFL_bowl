"""Segment-aware Savitzky–Golay filtering in NumPy (no SciPy dependency).

Why Savitzky–Golay on positions
    Speed and acceleration are first and second derivatives of position. Differencing 10 Hz
    positions amplifies noise by 1/Δt per derivative (×10, ×100). A local least-squares
    polynomial fit (SG) gives smoothed position and its analytic derivatives from the same fit,
    with a documented bias/variance trade-off controlled by (window, polyorder).

Boundary handling (equivalent to ``scipy.signal.savgol_filter(mode="interp")``)
    Frames closer than ``window // 2`` to a segment edge are evaluated on the polynomial fitted to
    the first / last full window *of the same segment*. Windows never extend past a segment
    boundary, so values from a different play, attempt or a gap are never mixed in.

Zero-phase vs causal
    The default centred fit uses future frames (zero phase lag). That is correct for retrospective
    description but not for a real-time interpretation. ``causal=True`` evaluates the fit at the
    newest frame of a trailing window (pure past information; frames with fewer than ``window``
    past samples in the segment are left undefined). Causal estimates lag the true signal.

All functions are vectorised over many concatenated segments; there is no Python loop per
segment (only a loop over the ≤ 11 window taps).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .config import SmoothingSpec


@lru_cache(maxsize=64)
def _savgol_matrix_cached(window: int, polyorder: int, deriv: int) -> np.ndarray:
    k = window // 2
    t = np.arange(window, dtype=float) - k                      # sample positions, centred
    A = np.vander(t, polyorder + 1, increasing=True)            # W × (p+1): A[i, m] = t_i^m
    pinv = np.linalg.pinv(A)                                    # (p+1) × W: poly coeffs = pinv @ v
    D = np.zeros((window, polyorder + 1))
    for m in range(deriv, polyorder + 1):                       # d^deriv/dt^deriv of t^m at t_r
        D[:, m] = math.factorial(m) / math.factorial(m - deriv) * t ** (m - deriv)
    M = D @ pinv
    M.setflags(write=False)
    return M


def savgol_matrix(window: int, polyorder: int, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    """W × W matrix. Row ``r`` holds the weights that, applied to one window of samples
    (oldest first), give the ``deriv``-th derivative of the fitted polynomial at window position
    ``r``. Row ``window // 2`` is the classic centred SG kernel; row ``window − 1`` is causal."""
    if deriv < 0:
        raise ValueError("deriv must be >= 0")
    if deriv > polyorder:
        return np.zeros((window, window))
    return _savgol_matrix_cached(window, polyorder, deriv) / (delta ** deriv)


def segment_index(pos: np.ndarray, seg_len: np.ndarray, window: int, causal: bool = False
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each frame: (valid mask, first index of the window used, row of the SG matrix).

    ``pos`` is the 0-based position of the frame inside its segment and ``seg_len`` the segment
    length; frames must be stored contiguously and in time order per segment.
    """
    pos = np.asarray(pos, dtype=np.int64)
    seg_len = np.asarray(seg_len, dtype=np.int64)
    i = np.arange(pos.size, dtype=np.int64)
    start = i - pos
    valid = seg_len >= window
    if causal:
        valid &= pos >= window - 1
        wstart = i - (window - 1)
    else:
        k = window // 2
        hi = np.maximum(start + seg_len - window, start)       # guard for invalid (short) segments
        wstart = np.clip(i - k, start, hi)
    return valid, wstart, i - wstart


def segmented_savgol(values: np.ndarray, pos: np.ndarray, seg_len: np.ndarray, spec: SmoothingSpec,
                     deriv: int = 0, delta: float = 1.0, causal: bool = False,
                     index: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """SG-filter ``values`` independently inside every segment; NaN where undefined.

    ``index`` may pass a precomputed :func:`segment_index` result to avoid recomputation when
    several signals / derivatives share the same segmentation.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.size, np.nan)
    valid, wstart, row = index if index is not None else segment_index(pos, seg_len, spec.window, causal)
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return out
    M = savgol_matrix(spec.window, spec.polyorder, deriv, delta)
    ws, r = wstart[idx], row[idx]
    acc = np.zeros(idx.size)
    for j in range(spec.window):                                # ≤ 11 taps
        acc += M[r, j] * v[ws + j]
    out[idx] = acc                                              # NaN input in a window → NaN output
    return out


def wrap_degrees(delta_deg: np.ndarray | float) -> np.ndarray | float:
    """Map an angle difference to (−180, 180]. Use for differences of vendor ``dir`` values."""
    d = (np.asarray(delta_deg, dtype=float) + 180.0) % 360.0 - 180.0
    d = np.where(d == -180.0, 180.0, d)
    return d if np.ndim(d) else float(d)


def circular_abs_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Smallest absolute angle between two headings (degrees, 0–180)."""
    return np.abs(wrap_degrees(np.asarray(a_deg, dtype=float) - np.asarray(b_deg, dtype=float)))
