"""Functional PCA of aligned sprint curves that are only partly recorded (notebook 02).

Representations (one channel against one axis; built with the alignment code of notebook 01)
    A_speed_by_distance   speed against path length since the first recorded frame      primary
    B_accel_by_speed      tangential acceleration against speed, rising phase only      secondary
    C_speed_by_time       speed against time since the first recorded frame             control
    *_alt_*               one alternate alignment per representation (sensitivity only): origin moved to the
                          moment the athlete reaches the reference speed (A, C), boundary frames kept (B)

A curve is a row of ``CurveSet.Y``: NaN wherever the recording holds no data. Nothing here fills a NaN.

Model (fitted on the sequences of the TRAINING athletes only; every athlete weighs 1, shared by his attempts)
    support       the longest run of grid points observed for at least ``min_coverage`` of the training weight
    mean          pointwise weighted mean over the curves observed at each grid point
    covariance    pairwise-complete: cell (g, h) uses the curves observed at both g and h
    basis         clamped cubic B-splines on the support; the covariance is projected on the basis and the
                  eigenproblem is solved in the L2 metric (Gram matrix of the basis), so eigenfunctions are
                  smooth and have unit integrated square
    noise         sigma2(K) = variance per grid point left after K components
    scores        best linear prediction from the OBSERVED grid points of a curve (PACE, Yao et al. 2005):
                      xi = (F'F / sigma2 + diag(1 / lambda))^-1 F'(y - mean) / sigma2,   F = eigenfunctions at
                  those points. A curve observed on less than ``min_observed_fraction`` of the support gets no
                  score. ``score_method = 'ls'`` is the unshrunk weighted least-squares alternative.

A reconstruction (mean + scores · eigenfunctions) is defined on the whole support. Where the curve was not
recorded it is a MODEL-IMPLIED value, never an observation: errors are computed on observed points only, and
figures draw that part with a different line style.

Known limit of the missing-data treatment: in representation B a high speed level is missing precisely for the
athletes who never reach it, so missingness is informative there; the support rule keeps only speed levels most
training athletes reach, and the observed fraction is exported as a diagnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...data.utils import atomic_path
from . import alignment as al
from .config import AdvancedConfig, NotebookControls
from .data import assert_disjoint
from .sequences import SequenceSet

FPCA_VERSION = "1"


class FpcaError(ValueError):
    """A functional PCA cannot be fitted or applied (too few curves, empty support, shape mismatch)."""


@dataclass(frozen=True)
class FpcaConfig:
    version: str = FPCA_VERSION
    # tuning budget: the ONLY quantity compared is the number of components ------------------------------
    k_grid: tuple[int, ...] = (2, 3, 5, 8)        # given by the milestone brief
    # fixed before any fit (proposed, not tuned) ---------------------------------------------------------
    degree: int = 3                               # cubic B-splines
    min_coverage: float = 0.80                    # training weight observed at a grid point to enter the support
    min_observed_fraction: float = 0.50           # share of the support a curve must cover to be scored
    score_method: str = "blup"                    # "blup" | "ls"
    ridge: float = 1e-8                           # "ls" only
    fve_target: float = 0.95                      # benchmark K = smallest K reaching this held-out share of variance
    last_frame_speed_fraction: float = 0.98       # last-frame speed / peak speed above this = still at top speed
    # screening references (heuristic, reused from the sprint rescue; never acceptance thresholds) --------
    icc_floor: float = 0.50                       # established convention: Koo & Li 2016 "moderate"
    icc_reference: float = 0.75                   # established convention: Koo & Li 2016 "good"
    redundancy_reference: float = 0.70            # proposed (RescueConfig.redundancy_reference)
    window_reference: float = 0.25                # proposed: share of attempt-to-attempt change explained by the window
    min_athletes: int = 10                        # established: RescueConfig.min_athletes (below → no estimate)
    # execution --------------------------------------------------------------------------------------------
    n_boot: int = 1000                            # athlete bootstrap replicates (ICC, reconstruction)
    n_boot_r2: int = 500                          # athlete bootstrap replicates of an R²

    def __post_init__(self) -> None:
        if self.score_method not in ("blup", "ls"):
            raise FpcaError(f"score_method must be 'blup' or 'ls', got {self.score_method!r}")
        if not self.k_grid or min(self.k_grid) < 1 or list(self.k_grid) != sorted(set(self.k_grid)):
            raise FpcaError(f"k_grid must be increasing positive integers, got {self.k_grid}")

    @property
    def k_max(self) -> int:
        return max(self.k_grid)

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def for_mode(self, controls: NotebookControls) -> "FpcaConfig":
        """Smoke mode keeps every scientific parameter; it only shortens the bootstraps."""
        return replace(self, n_boot=200, n_boot_r2=100) if controls.smoke else self


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    label: str
    role: str                      # primary · secondary · control · alternate
    alternate_of: str | None
    method: str                    # distance · speed_phase · elapsed_time (alignment.ALIGNERS)
    channel: str                   # the curve's value
    origin: str                    # "recording" = first recorded frame · "reference_speed" = moment speed reaches it
    mask_boundary: bool            # drop the channel on Savitzky–Golay boundary frames before interpolating
    n_basis: int                   # proposed, fixed (not tuned)
    axis_label: str
    value_label: str
    value_unit: str
    what_changes: str = ""         # alternates: the single thing that differs from the main representation


REPRESENTATIONS: tuple[RepresentationSpec, ...] = (
    RepresentationSpec("A_speed_by_distance", "A · speed against recorded distance", "primary", None, "distance",
                       "speed", "recording", False, 16, "path length since the first recorded frame (yd)",
                       "speed (yd/s)", "yd/s"),
    RepresentationSpec("B_accel_by_speed", "B · acceleration against speed (rising phase)", "secondary", None,
                       "speed_phase", "a_tan", "recording", True, 10, "speed level reached (yd/s)",
                       "tangential acceleration (yd/s²)", "yd/s²"),
    RepresentationSpec("C_speed_by_time", "C · speed against elapsed time", "control", None, "elapsed_time", "speed",
                       "recording", False, 16, "time since the first recorded frame (s)", "speed (yd/s)", "yd/s"),
    RepresentationSpec("A_alt_origin_at_ref_speed", "A′ · speed against distance since the reference speed",
                       "alternate", "A_speed_by_distance", "distance", "speed", "reference_speed", False, 16,
                       "path length since the reference speed was reached (yd)", "speed (yd/s)", "yd/s",
                       "origin = the moment speed first reaches the reference speed, not the first recorded frame"),
    RepresentationSpec("B_alt_boundary_kept", "B′ · acceleration against speed, boundary frames kept", "alternate",
                       "B_accel_by_speed", "speed_phase", "a_tan", "recording", False, 10, "speed level reached (yd/s)",
                       "tangential acceleration (yd/s²)", "yd/s²",
                       "acceleration on the smoothing-boundary frames is kept instead of masked"),
    RepresentationSpec("C_alt_origin_at_ref_speed", "C′ · speed against time since the reference speed", "alternate",
                       "C_speed_by_time", "elapsed_time", "speed", "reference_speed", False, 16,
                       "time since the reference speed was reached (s)", "speed (yd/s)", "yd/s",
                       "origin = the moment speed first reaches the reference speed, not the first recorded frame"),
)
MAIN = tuple(s for s in REPRESENTATIONS if s.alternate_of is None)
SPEC = {s.name: s for s in REPRESENTATIONS}


# ============================================================================== curves
@dataclass
class CurveSet:
    """Aligned curves of one representation. ``Y`` (n, G) is NaN where the recording holds no value;
    ``t_orig`` (n, G) is the original clock [s] at each grid point (NaN outside the recording)."""
    spec: RepresentationSpec
    grid: np.ndarray
    Y: np.ndarray
    t_orig: np.ndarray
    seq_ids: np.ndarray
    nfl_ids: np.ndarray

    @property
    def n(self) -> int:
        return int(self.seq_ids.size)

    @property
    def mask(self) -> np.ndarray:
        return np.isfinite(self.Y)

    def subset(self, idx: np.ndarray) -> "CurveSet":
        return CurveSet(self.spec, self.grid, self.Y[idx], self.t_orig[idx], self.seq_ids[idx], self.nfl_ids[idx])

    def index_of(self, seq_id: str) -> int:
        hit = np.flatnonzero(self.seq_ids == seq_id)
        if not hit.size:
            raise KeyError(f"sequence {seq_id!r} has no curve in {self.spec.name}")
        return int(hit[0])


def mask_boundary_frames(ss: SequenceSet, channels: Sequence[str]) -> SequenceSet:
    """Copy of ``ss`` in which ``channels`` are undefined on the smoothing-boundary frames (first and last
    half-window of the segment), where a second derivative is least reliable. Other channels are untouched."""
    X, valid = ss.X.copy(), ss.valid.copy()
    for c in channels:
        j = ss.channel(c)
        valid[:, :, j] &= ~ss.boundary
        X[:, :, j] = np.where(valid[:, :, j], X[:, :, j], 0.0)
    out = SequenceSet(ss.seq_ids, ss.nfl_ids, ss.channels, X, ss.mask, valid, ss.t, ss.boundary, ss.meta)
    out.validate()
    return out


def reference_crossing(x: np.ndarray, v: np.ndarray, level: float) -> float:
    """Value of the axis ``x`` at the moment speed ``v`` first reaches ``level`` (linear interpolation between the
    two frames around it). NaN when the recording starts at or above the level, or never reaches it — the
    crossing was not recorded and is not guessed."""
    idx = np.flatnonzero(v >= level)
    if not idx.size or idx[0] == 0:
        return float("nan")
    k = int(idx[0])
    if not (np.isfinite(v[k - 1]) and np.isfinite(x[k - 1]) and np.isfinite(x[k])) or v[k] <= v[k - 1]:
        return float("nan")
    return float(x[k - 1] + (level - v[k - 1]) / (v[k] - v[k - 1]) * (x[k] - x[k - 1]))


def _align_from_reference(ss: SequenceSet, acfg: AdvancedConfig, spec: RepresentationSpec) -> al.AlignedCurves:
    if spec.method == "distance":
        axis_channel, grid = "dist", np.arange(0.0, acfg.dist_max_yd + 1e-9, acfg.dist_step_yd)
    elif spec.method == "elapsed_time":
        axis_channel, grid = "t_rel", np.arange(0.0, acfg.time_max_s + 1e-9, acfg.frame_dt_s)
    else:
        raise FpcaError(f"origin 'reference_speed' is not defined for the {spec.method!r} alignment")
    al._require(ss, [axis_channel, "speed", spec.channel])

    def axis(i: int):
        x, v = ss.series(i, axis_channel), ss.series(i, "speed")
        x0 = reference_crossing(x, v, acfg.phase_ref_speed)
        if not np.isfinite(x0):
            return np.empty(0, int), np.empty(0)
        idx = al.increasing(x)
        return idx, x[idx] - x0
    return al._align(ss, [spec.channel], grid, axis, f"{spec.method}_from_reference_speed", spec.axis_label, "")


def build_curves(ss: SequenceSet, acfg: AdvancedConfig, spec: RepresentationSpec) -> CurveSet:
    """Curves of one representation: linear interpolation inside each sequence, NaN outside its recorded range."""
    src = mask_boundary_frames(ss, [spec.channel]) if spec.mask_boundary else ss
    if spec.origin == "recording":
        curves = al.ALIGNERS[spec.method](src, acfg, [spec.channel])
    elif spec.origin == "reference_speed":
        curves = _align_from_reference(src, acfg, spec)
    else:
        raise FpcaError(f"unknown origin {spec.origin!r}")
    return CurveSet(spec, curves.grid.astype(float), curves.channel(spec.channel).astype(float),
                    curves.t_orig.astype(float), curves.seq_ids, curves.nfl_ids)


# ============================================================================== basis
def clamped_knots(lo: float, hi: float, n_basis: int, degree: int) -> np.ndarray:
    if n_basis < degree + 1 or hi <= lo:
        raise FpcaError(f"a degree-{degree} B-spline basis needs at least {degree + 1} functions on a non-empty interval")
    inner = np.linspace(lo, hi, n_basis - degree + 1)[1:-1]
    return np.r_[np.full(degree + 1, lo), inner, np.full(degree + 1, hi)]


def bspline_design(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """(len(x), n_basis) values of the B-spline basis (Cox–de Boor recursion; rows sum to 1 inside the knots)."""
    x, knots = np.asarray(x, float), np.asarray(knots, float)
    m = knots.size - 1
    B = np.zeros((x.size, m))
    for j in range(m):
        B[:, j] = (x >= knots[j]) & (x < knots[j + 1])
    last = int(np.flatnonzero(knots[:-1] < knots[1:]).max())        # the right end belongs to the last interval
    B[x == knots[-1], last] = 1.0
    for d in range(1, degree + 1):
        nxt = np.zeros((x.size, m - d))
        for j in range(m - d):
            left, right = knots[j + d] - knots[j], knots[j + d + 1] - knots[j + 1]
            if left > 0:
                nxt[:, j] += (x - knots[j]) / left * B[:, j]
            if right > 0:
                nxt[:, j] += (knots[j + d + 1] - x) / right * B[:, j + 1]
        B = nxt
    return B


def quadrature(grid: np.ndarray) -> np.ndarray:
    """Trapezoid weights of a grid (they sum to its length)."""
    q, d = np.zeros(grid.size), np.diff(grid)
    q[:-1] += d / 2
    q[1:] += d / 2
    return q


def athlete_weights(nfl_ids: np.ndarray) -> np.ndarray:
    """One unit of weight per athlete, shared equally by his curves; the weights sum to 1."""
    _, inv, counts = np.unique(nfl_ids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.sum()


def longest_run(flag: np.ndarray) -> tuple[int, int]:
    """[start, stop) of the longest run of True (the first one on ties); (0, 0) when there is none."""
    best, start = (0, 0), None
    for i, f in enumerate([*flag.tolist(), False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


# ============================================================================== model
@dataclass
class FpcaModel:
    name: str
    grid_full: np.ndarray            # (G,) the representation's whole grid
    support: np.ndarray              # (G,) bool: grid points the model is defined on
    coverage: np.ndarray             # (G,) training weight observed at each grid point
    mean: np.ndarray                 # (Gs,)
    phi: np.ndarray                  # (Gs, K) eigenfunctions, unit integrated square
    eigenvalues: np.ndarray          # (K,) variance of each score
    total_variance: float            # integrated pointwise variance of the training curves on the support
    knots: np.ndarray
    degree: int
    min_observed_fraction: float
    score_method: str
    ridge: float
    fit_athletes: tuple[int, ...]
    n_fit_sequences: int
    source_rank: np.ndarray          # (K,) rank of each component by its own eigenvalue (1 = largest)

    # ------------------------------------------------------------------ structure
    @property
    def grid(self) -> np.ndarray:
        return self.grid_full[self.support]

    @property
    def q(self) -> np.ndarray:
        return quadrature(self.grid)

    @property
    def k_max(self) -> int:
        return int(self.phi.shape[1])

    @property
    def n_basis(self) -> int:
        return int(self.knots.size - self.degree - 1)

    def explained(self) -> np.ndarray:
        """Share of the training curves' integrated variance carried by each component."""
        return self.eigenvalues / self.total_variance

    def sigma2(self, k: int) -> float:
        """Variance per grid point not carried by the first ``k`` components (never below a small floor)."""
        left = self.total_variance - float(self.eigenvalues[:k].sum())
        return max(left, 1e-6 * self.total_variance) / float(self.q.sum())

    def assert_unseen(self, athletes: Sequence[int], where: str = "") -> None:
        """Raise if any of ``athletes`` contributed to this fit (call before an inductive evaluation)."""
        assert_disjoint(self.fit_athletes, list(athletes), where or f"FPCA fit {self.name} vs evaluation athletes")

    def _check(self, Y: np.ndarray, k: int) -> np.ndarray:
        Y = np.atleast_2d(np.asarray(Y, float))
        if Y.shape[1] != self.grid_full.size:
            raise FpcaError(f"{self.name}: curves have {Y.shape[1]} grid points, the model expects {self.grid_full.size}")
        if not 1 <= k <= self.k_max:
            raise FpcaError(f"{self.name}: k = {k} is outside 1 … {self.k_max}")
        return Y[:, self.support]

    # ------------------------------------------------------------------ application
    def scores(self, Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(scores (n, k), observed fraction of the support (n,)). A curve covering less than
        ``min_observed_fraction`` of the support gets NaN scores; nothing is imputed to score it."""
        Ys = self._check(Y, k)
        O, q = np.isfinite(Ys), self.q
        frac = (O * q).sum(1) / q.sum()
        S = np.full((Ys.shape[0], k), np.nan)
        phi = self.phi[:, :k]
        lam = np.maximum(self.eigenvalues[:k], 1e-12 * self.total_variance)
        s2 = self.sigma2(k)
        for i in np.flatnonzero(frac >= self.min_observed_fraction):
            o = O[i]
            F, r = phi[o], Ys[i, o] - self.mean[o]
            if self.score_method == "blup":
                A, b = F.T @ F / s2 + np.diag(1.0 / lam), F.T @ r / s2
            else:
                Fq = F * q[o][:, None]
                A, b = F.T @ Fq + self.ridge * np.eye(k), Fq.T @ r
            S[i] = np.linalg.solve(A, b)
        return S, frac

    def reconstruct(self, S: np.ndarray, k: int) -> np.ndarray:
        """(n, Gs) mean + scores · eigenfunctions on the support. Model-implied wherever the curve was not recorded."""
        return self.mean[None, :] + np.asarray(S, float)[:, :k] @ self.phi[:, :k].T

    def residuals(self, Y: np.ndarray, S: np.ndarray, k: int) -> dict[str, np.ndarray]:
        """Per curve, over its OBSERVED support points only: squared error of the reconstruction (``sse``), squared
        deviation from the training mean (``sst``) and the number of points. NaN / 0 for an unscored curve."""
        Ys = self._check(Y, k)
        O = np.isfinite(Ys) & np.isfinite(S[:, :1])
        err = np.where(O, Ys - self.reconstruct(np.nan_to_num(S), k), 0.0)
        dev = np.where(O, Ys - self.mean[None, :], 0.0)
        n = O.sum(1)
        return {"sse": np.where(n > 0, (err ** 2).sum(1), np.nan), "sst": np.where(n > 0, (dev ** 2).sum(1), np.nan),
                "n_points": n.astype(np.int64)}

    # ------------------------------------------------------------------ persistence
    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"grid_full": self.grid_full, "support": self.support, "coverage": self.coverage, "mean": self.mean,
                "phi": self.phi, "eigenvalues": self.eigenvalues, "total_variance": np.array(self.total_variance),
                "knots": self.knots, "degree": np.array(self.degree),
                "min_observed_fraction": np.array(self.min_observed_fraction), "score_method": np.array(self.score_method),
                "ridge": np.array(self.ridge), "fit_athletes": np.asarray(self.fit_athletes, dtype=np.int64),
                "n_fit_sequences": np.array(self.n_fit_sequences), "source_rank": self.source_rank}

    @classmethod
    def from_arrays(cls, name: str, a: dict[str, np.ndarray]) -> "FpcaModel":
        return cls(name, a["grid_full"], a["support"].astype(bool), a["coverage"], a["mean"], a["phi"], a["eigenvalues"],
                   float(a["total_variance"]), a["knots"], int(a["degree"]), float(a["min_observed_fraction"]),
                   str(a["score_method"]), float(a["ridge"]), tuple(int(i) for i in a["fit_athletes"]),
                   int(a["n_fit_sequences"]), a["source_rank"].astype(np.int64))


def fit_fpca(curves: CurveSet, train_athletes: Sequence[int], fcfg: FpcaConfig, name: str | None = None) -> FpcaModel:
    """Fit on the curves of ``train_athletes`` and nothing else: support, mean, covariance, basis and
    eigenfunctions all come from those athletes."""
    ids = np.asarray(sorted({int(i) for i in train_athletes}), dtype=np.int64)
    rows = np.isin(curves.nfl_ids, ids) & np.isfinite(curves.Y).any(1)
    label = name or curves.spec.name
    if rows.sum() < 5:
        raise FpcaError(f"{label}: only {int(rows.sum())} training curve(s) with data; at least 5 are needed")
    Y, w = curves.Y[rows], athlete_weights(curves.nfl_ids[rows])
    O = np.isfinite(Y)
    coverage = (w[:, None] * O).sum(0)
    lo, hi = longest_run(coverage >= fcfg.min_coverage)
    if hi - lo < fcfg.degree + 3:
        raise FpcaError(f"{label}: only {hi - lo} contiguous grid point(s) are observed for ≥ {fcfg.min_coverage:.0%} of "
                        "the training athletes; the support is too short for a functional basis")
    support = np.zeros(curves.grid.size, dtype=bool)
    support[lo:hi] = True
    g, Ys, Os = curves.grid[support], Y[:, support], O[:, support].astype(float)
    wO = w[:, None] * Os
    mean = (wO * np.where(Os > 0, Ys, 0.0)).sum(0) / wO.sum(0)
    R = np.where(Os > 0, Ys - mean, 0.0)
    den = wO.T @ Os
    C = np.where(den > 1e-12, ((w[:, None] * R).T @ R) / np.maximum(den, 1e-12), 0.0)
    C = (C + C.T) / 2
    q = quadrature(g)
    total = float((q * np.diag(C)).sum())
    if not total > 0:
        raise FpcaError(f"{label}: the training curves have no variance on the support")
    n_basis = int(min(curves.spec.n_basis, g.size - 2))
    knots = clamped_knots(float(g[0]), float(g[-1]), n_basis, fcfg.degree)
    Phi = bspline_design(g, knots, fcfg.degree)
    M = Phi.T @ (q[:, None] * Phi)
    m, U = np.linalg.eigh(M)
    m = np.maximum(m, 1e-12 * m.max())
    M_half, M_inv_half = (U * np.sqrt(m)) @ U.T, (U / np.sqrt(m)) @ U.T
    P = np.linalg.solve(M, Phi.T * q)                       # L2 projection of a function on the basis
    S = M_half @ (P @ C @ P.T) @ M_half
    lam, V = np.linalg.eigh((S + S.T) / 2)
    order = np.argsort(lam)[::-1]
    k = int(min(fcfg.k_max, n_basis))
    lam, V = np.maximum(lam[order][:k], 0.0), V[:, order][:, :k]
    phi = Phi @ (M_inv_half @ V)
    sign = np.sign((q[:, None] * phi).sum(0))               # convention: positive integral (re-aligned to a reference later)
    phi = phi * np.where(sign == 0, 1.0, sign)
    return FpcaModel(label, curves.grid.copy(), support, coverage, mean, phi, lam, total, knots, fcfg.degree,
                     fcfg.min_observed_fraction, fcfg.score_method, fcfg.ridge, tuple(ids.tolist()), int(rows.sum()),
                     np.arange(1, k + 1, dtype=np.int64))


def align_to_reference(model: FpcaModel, ref: FpcaModel) -> tuple[FpcaModel, np.ndarray]:
    """Give ``model`` the sign and order of ``ref`` so that scores of different fits are comparable.

    Each reference component takes, in order, the not-yet-used component of ``model`` with the largest absolute
    congruence on the common support. Returns the re-ordered model and that congruence (1 = same function up to
    sign). Only sign and order are taken from the reference; no value of ``model`` changes."""
    common = model.support & ref.support
    if common.sum() < 3 or model.k_max != ref.k_max:
        raise FpcaError(f"{model.name}: cannot be aligned to the reference (common support {int(common.sum())} points, "
                        f"{model.k_max} vs {ref.k_max} components)")
    a, b = model.phi[common[model.support]], ref.phi[common[ref.support]]
    qc = quadrature(model.grid_full[common])
    cross = a.T @ (qc[:, None] * b)
    na, nb = np.sqrt((qc[:, None] * a ** 2).sum(0)), np.sqrt((qc[:, None] * b ** 2).sum(0))
    cong = cross / np.maximum(np.outer(na, nb), 1e-300)
    K = model.k_max
    order, sign, value, used = np.zeros(K, int), np.ones(K), np.zeros(K), set()
    for k in range(K):
        j = max((j for j in range(K) if j not in used), key=lambda j: abs(cong[j, k]))
        used.add(j)
        order[k], sign[k], value[k] = j, (1.0 if cong[j, k] >= 0 else -1.0), abs(cong[j, k])
    return replace(model, phi=model.phi[:, order] * sign, eigenvalues=model.eigenvalues[order],
                   source_rank=model.source_rank[order]), value


# ============================================================================== persistence of many models
def save_models(path: Path, models: dict[str, FpcaModel]) -> Path:
    """One compressed ``.npz`` holding several fitted models under ``<tag>/<field>`` keys (written atomically)."""
    arrays = {f"{tag}/{field}": value for tag, m in models.items() for field, value in m.to_arrays().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
    return path


def load_models(path: Path) -> dict[str, FpcaModel]:
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            tag, field = key.rsplit("/", 1)
            out.setdefault(tag, {})[field] = z[key]
    return {tag: FpcaModel.from_arrays(tag, arrays) for tag, arrays in out.items()}
