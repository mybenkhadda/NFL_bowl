"""NFL context decomposition (Milestone 07A): what does a frozen NFL movement outcome measure? NumPy / Polars only.

Question        how much of an athlete's NFL movement value is a stable athlete difference, and how much is game-to-game
                variation, role, alignment, starting speed and observed opportunity?
Unit            the ATHLETE. Windows (player-plays) are nested in athlete-games, nested in athletes; no window is ever
                treated as an independent athlete. Every interval resamples athletes.
Definitions     C0 raw athlete aggregate (the Milestone 05 statistic, REFERENCE ONLY — the frozen registry is not touched)
                C1 position-standardised athlete value (z within position)
                C2 starting-speed-conditioned: window − mean of its position × starting-speed band, supported bands only
                C3 position × alignment conditioned: window − mean of its position × alignment cell, supported cells only
                C4 hierarchical: additive context effects (alignment × play class, starting-speed band, season, down)
                   removed by backfitting, then an athlete effect shrunk by nested variance components (athlete / game /
                   window, method of moments). EXPLORATORY.
Fold safety     every context parameter (cell means, position means and SDs, variance components) is estimated on the
                ``fit`` windows only. A transfer fold passes the windows of its TRAINING athletes; a reliability split
                passes the OTHER half of the games (cross-fitting). ``fit = None`` (all windows) is descriptive only.
Limits          a context adjustment removes what the OBSERVED labels explain. It does not turn observed movement into
                athletic capability: assignment, opponent and scheme are unobserved, and role is partly chosen BECAUSE of
                ability, so removing it can remove ability too.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ...data.utils import get_logger
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from .config import NotebookControls

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
CONTEXT_VERSION = "1"
KEYS = ["game_id", "play_id", "nfl_id"]
ALL = "ALL"
NA = "NA"
EXPLORATORY = "exploratory"
DEFINITIONS: dict[str, dict[str, Any]] = {
    "C0": {"label": "raw athlete aggregate (Milestone 05 statistic)", "complexity": 0, "status": "reference_frozen_m05",
           "needs": ()},
    "C1": {"label": "position-standardised athlete value", "complexity": 1, "status": EXPLORATORY, "needs": ()},
    "C2": {"label": "starting-speed-conditioned (position × starting-speed band)", "complexity": 2, "status": EXPLORATORY,
           "needs": ("start_speed",)},
    "C3": {"label": "position × alignment conditioned", "complexity": 3, "status": EXPLORATORY, "needs": ("alignment",)},
    "C4": {"label": "hierarchical context-adjusted athlete effect (shrunk)", "complexity": 4, "status": EXPLORATORY,
           "needs": ()},
}


class ContextError(ValueError):
    """A context definition cannot be computed as requested (missing context field, empty support, bad mask)."""


# ============================================================================== configuration
@dataclass(frozen=True)
class ContextConfig:
    """Provenance: established = reused project value · proposed = fixed here before any Combine value is joined ·
    heuristic = reference line, never a sole decision rule · execution = cost knob."""

    version: str = CONTEXT_VERSION
    seed: int = 20261011
    outcome_id: str = "O2.accel_sust05_mean"       # established: frozen primary target of Milestone 05
    start_col: str = "gain05_v0"                   # established: speed at the start of the best 0.5 s gain window
    min_windows: int = 50                          # established: exposure requirement of the frozen targets
    min_half_windows: int = 5                      # established: OutcomeConfig.min_half_windows
    min_cell_windows: int = 30                     # established: OutcomeConfig.min_cell_windows
    min_cell_athletes: int = 15                    # proposed: athletes behind a context cell before it is "supported"
    start_bands: tuple[tuple[float, float], ...] = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99.0))   # proposed [yd/s]
    max_accel: float = 15.0                        # established: KinematicsConfig.max_accel_yds2 (plausibility)
    max_speed: float = 12.5                        # established: KinematicsConfig.max_speed_yds
    backfit_iter: int = 30                         # execution
    backfit_tol: float = 1e-7                      # execution
    n_boot: int = 1000                             # execution: athlete bootstrap
    n_game_splits: int = 50                        # execution: random game half-splits
    first_games_grid: tuple[int, ...] = (2, 4, 8, 16)          # established: OutcomeConfig.first_games_grid
    first_windows_grid: tuple[int, ...] = (20, 50, 100, 200)   # established: OutcomeConfig.first_n_grid
    holdout_min_windows: int = 50                  # established: OutcomeConfig.holdout_min_windows
    min_athletes: int = 20                         # established: OutcomeConfig.min_athletes
    min_athletes_position: int = 15                # established: OutcomeConfig.min_athletes_position
    ref_reliability: float = 0.70                  # heuristic: OutcomeConfig.ref_reliability
    ref_retention: float = 0.80                    # heuristic: share of the C0 cohort a new definition should keep
    ref_snr: float = 2.0                           # heuristic: between-athlete SD / standard error of an athlete value
    rank_shift_k: int = 8                          # execution: example athletes shown

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def for_mode(self, controls: NotebookControls) -> "ContextConfig":
        """Smoke mode keeps every definition; it lowers the size requirements and shortens the resampling."""
        if not controls.smoke:
            return replace(self, seed=controls.seed)
        return replace(self, seed=controls.seed, min_windows=10, min_cell_windows=10, min_cell_athletes=3, n_boot=100,
                       n_game_splits=6, first_games_grid=(1, 2), first_windows_grid=(5, 10), holdout_min_windows=5,
                       min_athletes=6, min_athletes_position=2, rank_shift_k=3)


# ============================================================================== progress (shared by 07A–07D)
class Progress:
    """Iteration-level progress with a run-time estimate from the iterations already completed."""

    def __init__(self, total: int, label: str, every: int = 1, printer: Callable[[str], None] = print):
        self.total, self.label, self.every, self.printer = int(total), label, max(1, int(every)), printer
        self.t0, self.done = time.perf_counter(), 0
        printer(f"[{label}] {self.total} iteration(s) planned")

    def step(self, k: int = 1, note: str = "") -> None:
        self.done += k
        if self.done % self.every and self.done < self.total:
            return
        el = time.perf_counter() - self.t0
        eta = el / max(self.done, 1) * max(self.total - self.done, 0)
        self.printer(f"[{self.label}] {self.done}/{self.total} · elapsed {el:,.1f} s · about {eta:,.1f} s left"
                     + (f" · {note}" if note else ""))

    def estimate(self, seconds_per_iteration: float) -> str:
        return f"[{self.label}] estimated work: {self.total} × {seconds_per_iteration:.3f} s ≈ {self.total * seconds_per_iteration / 60:.1f} min"


# ============================================================================== windows
def _codes(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels, inv = np.unique(np.asarray(x), return_inverse=True)
    return inv.astype(np.int64), levels


@dataclass
class Windows:
    """Eligible windows of ONE play measure, sorted by athlete then time, with the observed context labels."""

    ids: np.ndarray              # (A,) athlete ids, sorted
    position: np.ndarray         # (A,) position group
    cohort: np.ndarray           # (A,) draft year (NaN unknown)
    a: np.ndarray                # (W,) athlete index
    y: np.ndarray                # (W,) play measure
    game: np.ndarray             # (W,) game id
    g_order: np.ndarray          # (W,) 1 = the athlete's first game
    w_order: np.ndarray          # (W,) 1 = the athlete's first window of this measure
    season: np.ndarray           # (W,)
    align: np.ndarray            # (W,) lined-up position, 'NA' when missing
    play_class: np.ndarray       # (W,)
    down: np.ndarray             # (W,) 0 when missing
    v0: np.ndarray               # (W,) starting speed [yd/s], NaN when the measure has none
    measure: str = "y"
    unit: str = ""
    outcome_id: str = ""

    def __post_init__(self) -> None:
        if np.any(np.diff(self.a) < 0):
            raise ContextError("windows must be sorted by athlete")
        gc, _ = _codes(self.game)
        key = self.a.astype(np.int64) * (int(gc.max()) + 1 if gc.size else 1) + gc
        self.ag, first = np.unique(key, return_inverse=True)[1].astype(np.int64), None
        n_ag = int(self.ag.max()) + 1 if self.ag.size else 0
        self.ag_athlete = np.zeros(n_ag, dtype=np.int64)
        self.ag_athlete[self.ag] = self.a
        self.pos_w, self.pos_levels = _codes(self.position[self.a])
        self.pos_a = np.searchsorted(self.pos_levels, self.position)

    @property
    def n_athletes(self) -> int:
        return int(self.ids.size)

    @property
    def n_windows(self) -> int:
        return int(self.y.size)

    @classmethod
    def make(cls, nfl_id: Sequence[int], y: Sequence[float], game: Sequence[int], position: Sequence[str], **kw: Any) -> "Windows":
        """Windows from per-window arrays (fixtures and tests). Time order = input order within athlete."""
        nfl = np.asarray(nfl_id, dtype=np.int64)
        o = np.argsort(nfl, kind="stable")
        nfl, yy, gg, pp = nfl[o], np.asarray(y, dtype=float)[o], np.asarray(game, dtype=np.int64)[o], np.asarray(position, dtype=str)[o]
        ids, a = np.unique(nfl, return_inverse=True)
        W = nfl.size
        start = np.r_[0, np.flatnonzero(np.diff(a)) + 1]
        w_order = np.arange(W) - start[a] + 1
        g_order = np.empty(W, dtype=np.int64)
        for i in range(ids.size):
            m = a == i
            _, first_idx, inv = np.unique(gg[m], return_index=True, return_inverse=True)
            g_order[m] = np.argsort(np.argsort(first_idx))[inv] + 1
        opt = {k: np.asarray(v)[o] for k, v in kw.items() if k in ("season", "align", "play_class", "down", "v0")}
        first_of = np.r_[True, a[1:] != a[:-1]]
        cohort = np.asarray(kw["cohort"], dtype=float)[o][first_of] if "cohort" in kw else np.full(ids.size, NAN)
        return cls(ids, pp[first_of], cohort, a.astype(np.int64), yy, gg, g_order, w_order.astype(np.int64),
                   opt.get("season", np.full(W, 2024)).astype(np.int64), opt.get("align", np.full(W, "X")).astype(str),
                   opt.get("play_class", np.full(W, "pass")).astype(str), opt.get("down", np.ones(W)).astype(np.int64),
                   opt.get("v0", np.full(W, NAN)).astype(float), kw.get("measure", "y"), kw.get("unit", ""), kw.get("outcome_id", ""))

    @classmethod
    def from_engine(cls, eng: Any, outcome_id: str, start_col: str | None = None) -> "Windows":
        """Windows of a base outcome of ``nfl_outcomes.OutcomeEngine`` — the SAME eligible windows as Milestone 05.
        ``start_col``: a play-table column holding the starting speed of the measured window (None = not available)."""
        spec = eng.by_id[outcome_id]
        d = eng.frame(spec)
        if start_col is not None and start_col in eng.plays.columns:
            d = d.join(eng.plays.select(*KEYS, pl.col(start_col).cast(pl.Float64).alias("_v0")), on=KEYS, how="left")
        else:
            d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("_v0"))
        d = d.sort("nfl_id", "w_order")
        ath = eng.athletes.sort("nfl_id")
        ids = ath["nfl_id"].to_numpy().astype(np.int64)
        nfl = d["nfl_id"].to_numpy().astype(np.int64)
        keep = np.isin(ids, nfl)
        ids = ids[keep]
        return cls(ids, ath["position_group"].fill_null("?").to_numpy().astype(str)[keep],
                   ath["cohort"].cast(pl.Float64).fill_null(NAN).to_numpy()[keep], np.searchsorted(ids, nfl),
                   d["y"].to_numpy().astype(float), d["game_id"].to_numpy().astype(np.int64),
                   d["game_order"].to_numpy().astype(np.int64), d["w_order"].to_numpy().astype(np.int64),
                   d["season"].to_numpy().astype(np.int64), d["lined_up_position"].fill_null(NA).to_numpy().astype(str),
                   d["play_class"].fill_null(NA).to_numpy().astype(str), d["down"].cast(pl.Int64).fill_null(0).to_numpy(),
                   d["_v0"].fill_null(NAN).to_numpy().astype(float), spec.column, spec.unit, outcome_id)

    def band(self, cc: ContextConfig) -> np.ndarray:
        """(W,) starting-speed band index, −1 when the starting speed is missing or outside every band."""
        b = np.full(self.y.size, -1, dtype=np.int64)
        for k, (lo, hi) in enumerate(cc.start_bands):
            b[(self.v0 >= lo) & (self.v0 < hi)] = k
        return b

    def frame(self, cc: ContextConfig) -> pl.DataFrame:
        """Window-level frame for descriptive tables and figures (never a sample of independent athletes)."""
        band = self.band(cc)
        lab = np.array([f"{lo:g}–{hi:g}" if hi < 90 else f"≥ {lo:g}" for lo, hi in cc.start_bands] + [NA])
        return pl.DataFrame({"nfl_id": self.ids[self.a], "position_group": self.position[self.a], "cohort": self.cohort[self.a],
                             "y": self.y, "game_id": self.game, "game_order": self.g_order, "w_order": self.w_order,
                             "season": self.season, "alignment": self.align, "play_class": self.play_class, "down": self.down,
                             "start_speed": self.v0, "start_band": lab[band]}).with_columns(
            pl.col("cohort").fill_nan(None).cast(pl.Int64), pl.col("start_speed").fill_nan(None))


# ============================================================================== nested variance components
def athlete_stats(w: Windows, y: np.ndarray, use: np.ndarray) -> dict[str, np.ndarray]:
    """Per-athlete sufficient statistics of the nested layout athlete ⊃ game ⊃ window on the ``use`` windows:
    n, G (games), sum, ssw (within athlete-game), ssg (game means around the athlete mean, window-weighted),
    r = Σ_g n_g². Variance components and their athlete bootstrap are functions of these sums only."""
    A, nAG = w.n_athletes, w.ag_athlete.size
    ag, yy, a = w.ag[use], y[use], w.a[use]
    n_g = np.bincount(ag, minlength=nAG).astype(float)
    s_g = np.bincount(ag, weights=yy, minlength=nAG)
    m_g = np.where(n_g > 0, s_g / np.maximum(n_g, 1), 0.0)
    ssw_g = np.bincount(ag, weights=(yy - m_g[ag]) ** 2, minlength=nAG)
    has = n_g > 0
    aa = w.ag_athlete
    n = np.bincount(aa, weights=n_g, minlength=A)
    s = np.bincount(aa, weights=s_g, minlength=A)
    mean = np.where(n > 0, s / np.maximum(n, 1), NAN)
    return {"n": n, "G": np.bincount(aa[has], minlength=A).astype(float), "sum": s, "mean": mean,
            "ssw": np.bincount(aa, weights=ssw_g, minlength=A),
            "ssg": np.bincount(aa[has], weights=(n_g * (m_g - np.nan_to_num(mean)[aa]) ** 2)[has], minlength=A),
            "r": np.bincount(aa, weights=n_g ** 2, minlength=A)}


def components(st: dict[str, np.ndarray], pos: np.ndarray, weight: np.ndarray | None = None) -> dict[str, Any]:
    """Method-of-moments variance components of the unbalanced nested layout, POOLED WITHIN position:
    σ²_window (within athlete-game), σ²_game (games within athlete), σ²_athlete (athletes within position).
    ``weight``: athlete multiplicities of a bootstrap resample. Negative estimates are truncated at 0 and flagged —
    a truncation means the layer is not distinguishable from the layer below with these data, not that it is absent."""
    m = (st["n"] > 0).astype(float) if weight is None else weight * (st["n"] > 0)
    n, G, r = st["n"], st["G"], st["r"]
    mean = np.nan_to_num(st["mean"])
    N, n_groups = float((m * n).sum()), float((m * G).sum())
    df_w = N - n_groups
    s2w = float((m * st["ssw"]).sum() / df_w) if df_w > 0 else NAN
    q = np.where(n > 0, r / np.maximum(n, 1), 0.0)
    den_g = float((m * (n - q)).sum())
    raw_g = (float((m * st["ssg"]).sum()) - s2w * float((m * (G - 1)).sum())) / den_g if den_g > 0 else NAN
    s2g = max(raw_g, 0.0) if np.isfinite(raw_g) else NAN
    ssa = c_g = c_a = df_a = 0.0
    for p in np.unique(pos[m > 0]):
        k = (pos == p) & (m > 0)
        Np = float((m[k] * n[k]).sum())
        if Np <= 0 or m[k].sum() < 2:
            continue
        ssa += float((m[k] * n[k] * mean[k] ** 2).sum()) - float((m[k] * n[k] * mean[k]).sum()) ** 2 / Np
        c_g += float((m[k] * q[k]).sum()) - float((m[k] * r[k]).sum()) / Np
        c_a += Np - float((m[k] * n[k] ** 2).sum()) / Np
        df_a += float(m[k].sum()) - 1.0
    raw_a = (ssa - s2w * df_a - (s2g if np.isfinite(s2g) else 0.0) * c_g) / c_a if c_a > 0 else NAN
    s2a = max(raw_a, 0.0) if np.isfinite(raw_a) else NAN
    return {"s2_window": s2w, "s2_game": s2g, "s2_athlete": s2a, "truncated_game": bool(np.isfinite(raw_g) and raw_g < 0),
            "truncated_athlete": bool(np.isfinite(raw_a) and raw_a < 0), "n_athletes": int(round(float(m.sum()))),
            "n_athlete_games": int(round(n_groups)), "n_windows": int(round(N))}


def shrink_factor(c: dict[str, Any], n: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Reliability of an athlete mean under the nested model: σ²_a / (σ²_a + σ²_g / G + σ²_w / n). 0 without windows."""
    s2a, s2g, s2w = (float(np.nan_to_num(c[k])) for k in ("s2_athlete", "s2_game", "s2_window"))
    den = s2a + s2g / np.maximum(G, 1) + s2w / np.maximum(n, 1)
    return np.where((n > 0) & (den > 0), s2a / np.where(den > 0, den, 1.0), 0.0)


# ============================================================================== context adjustment
def _cell_means(code: np.ndarray, y: np.ndarray, fit: np.ndarray, a: np.ndarray, n_levels: int, A: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, windows, distinct athletes) of every level, from the ``fit`` windows only."""
    ok = fit & (code >= 0)
    n = np.bincount(code[ok], minlength=n_levels).astype(float)
    s = np.bincount(code[ok], weights=y[ok], minlength=n_levels)
    pairs = np.unique(code[ok] * np.int64(A) + a[ok])
    ath = np.bincount((pairs // A).astype(np.int64), minlength=n_levels).astype(float)
    return np.where(n > 0, s / np.maximum(n, 1), 0.0), n, ath


def backfit(y: np.ndarray, factors: Sequence[np.ndarray], fit: np.ndarray, cc: ContextConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Additive level effects of several categorical factors by backfitting (Gauss–Seidel on level means of the
    partial residual), estimated on ``fit`` windows. Returns (Σ effects for EVERY window, diagnostics). A level never
    seen among the fit windows, or with fewer than ``min_cell_windows`` of them, gets effect 0 (left unadjusted)."""
    sizes = [int(f.max()) + 1 if f.size else 1 for f in factors]
    eff = [np.zeros(k) for k in sizes]
    cnt = [np.bincount(f[fit & (f >= 0)], minlength=k).astype(float) for f, k in zip(factors, sizes)]
    total = np.zeros(y.size)
    change, it = np.inf, 0
    for it in range(1, cc.backfit_iter + 1):
        change = 0.0
        for j, f in enumerate(factors):
            ok = fit & (f >= 0)
            part = y - total + eff[j][np.maximum(f, 0)] * (f >= 0)
            new = np.bincount(f[ok], weights=part[ok], minlength=sizes[j]) / np.maximum(cnt[j], 1)
            new = np.where(cnt[j] >= cc.min_cell_windows, new, 0.0)
            big = cnt[j] >= cc.min_cell_windows
            if big.any():                                                   # identifiability: weighted mean effect = 0
                new = np.where(big, new - float((new[big] * cnt[j][big]).sum() / cnt[j][big].sum()), 0.0)
            change = max(change, float(np.abs(new - eff[j]).max()) if new.size else 0.0)
            total += (new - eff[j])[np.maximum(f, 0)] * (f >= 0)
            eff[j] = new
        if change < cc.backfit_tol:
            break
    return total, {"iterations": it, "converged": bool(change < cc.backfit_tol), "last_change": float(change),
                   "levels_adjusted": [int((c >= cc.min_cell_windows).sum()) for c in cnt], "levels_total": sizes}


class ContextLabels:
    """Athlete-level values of one play measure under the context definitions C0–C4.

    ``values(definition, fit, score)``: ``fit`` = windows the context parameters are estimated on, ``score`` = windows
    aggregated into the athlete value. ``label_fn`` gives the fold-safe labels of a transfer study (fit = windows of
    the training athletes of the fold)."""

    def __init__(self, w: Windows, cc: ContextConfig):
        self.w, self.cc = w, cc
        A = w.n_athletes
        self._band = w.band(cc)
        nb = len(cc.start_bands)
        self._pos_band = np.where(self._band >= 0, w.pos_w * nb + self._band, -1)
        al, self._align_levels = _codes(w.align)
        missing = w.align == NA
        self._pos_align = np.where(missing, -1, w.pos_w * self._align_levels.size + al)
        cell, self._cell_levels = _codes(np.char.add(np.char.add(w.align, "|"), w.play_class))
        self._pos_cell = w.pos_w * self._cell_levels.size + cell
        se, _ = _codes(w.season)
        dn, _ = _codes(w.down)
        self._season, self._down = se, dn
        self._band_factor = np.where(self._band >= 0, w.pos_w * (nb + 1) + self._band, w.pos_w * (nb + 1) + nb)
        self.has_start = bool(np.isfinite(w.v0).any())
        self.has_alignment = bool((~missing).any())
        self._all = np.ones(w.n_windows, dtype=bool)
        self._A = A
        self.last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ masks
    def train_mask(self, train_ids: Sequence[int] | None) -> np.ndarray:
        if train_ids is None:
            return self._all
        return np.isin(self.w.ids, np.asarray(list(train_ids), dtype=np.int64))[self.w.a]

    def available(self, definition: str) -> tuple[bool, str]:
        if definition not in DEFINITIONS:
            return False, f"unknown definition {definition!r}"
        if definition == "C2" and not self.has_start:
            return False, "the play measure has no starting-speed field"
        if definition == "C3" and not self.has_alignment:
            return False, "no alignment label is recorded"
        return True, ""

    # ------------------------------------------------------------------ window-level adjustment
    def adjust(self, definition: str, fit: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """(adjusted window value, windows kept, diagnostics). Parameters come from the ``fit`` windows only."""
        ok, why = self.available(definition)
        if not ok:
            raise ContextError(f"{definition}: {why}")
        w, cc, A = self.w, self.cc, self._A
        fit = self._all if fit is None else np.asarray(fit, dtype=bool)
        if not fit.any():
            raise ContextError(f"{definition}: no window to estimate the context parameters on")
        P = w.pos_levels.size
        mp, _, _ = _cell_means(w.pos_w, w.y, fit, w.a, P, A)
        if definition in ("C0", "C1"):
            return w.y.copy(), self._all.copy(), {"supported_cells": None, "total_cells": None}
        if definition in ("C2", "C3"):
            code = self._pos_band if definition == "C2" else self._pos_align
            L = int(code.max()) + 1 if (code >= 0).any() else 1
            mc, n_fit, _ = _cell_means(code, w.y, fit, w.a, L, A)
            # SUPPORT is a property of the definition: it uses window and athlete COUNTS of the whole table (no label
            # value), so the set of comparable cells — and each athlete's retained windows — is the same in every
            # fold. The cell MEANS come from the fit windows only.
            _, n, ath = _cell_means(code, w.y, self._all, w.a, L, A)
            sup = (n >= cc.min_cell_windows) & (ath >= cc.min_cell_athletes) & (n_fit > 0)
            keep = (code >= 0) & sup[np.maximum(code, 0)]
            adj = w.y - mc[np.maximum(code, 0)] + mp[w.pos_w]
            return adj, keep, {"supported_cells": int(sup.sum()), "total_cells": int((n > 0).sum()),
                               "missing_context_share": float((code < 0).mean())}
        centred = w.y - mp[w.pos_w]
        factors = [self._pos_cell, self._band_factor, w.pos_w * (int(self._season.max()) + 1) + self._season,
                   w.pos_w * (int(self._down.max()) + 1) + self._down]
        total, info = backfit(centred, factors, fit, cc)
        return w.y - total, self._all.copy(), {**info, "supported_cells": info["levels_adjusted"][0],
                                               "total_cells": int(np.unique(self._pos_cell).size), "missing_context_share": 0.0}

    # ------------------------------------------------------------------ athlete values
    def values(self, definition: str, fit: np.ndarray | None = None, score: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """{ids, value, n_windows, n_games} over ``self.w.ids``; value NaN for an athlete without a scored window."""
        w, cc, A = self.w, self.cc, self._A
        fit = self._all if fit is None else np.asarray(fit, dtype=bool)
        score = self._all if score is None else np.asarray(score, dtype=bool)
        adj, keep, info = self.adjust(definition, fit)
        use = keep & score
        n = np.bincount(w.a[use], minlength=A).astype(float)
        value = np.where(n > 0, np.bincount(w.a[use], weights=adj[use], minlength=A) / np.maximum(n, 1), NAN)
        G = np.bincount(w.ag_athlete[np.unique(w.ag[use])], minlength=A).astype(float)
        if definition in ("C1", "C4"):
            fu = keep & fit
            st = athlete_stats(w, adj, fu)
            enough = st["n"] >= cc.min_half_windows
            mu, sd = np.full(w.pos_levels.size, NAN), np.full(w.pos_levels.size, NAN)
            for p in range(w.pos_levels.size):
                k = enough & (w.pos_a == p)
                if k.sum() >= 2:
                    mu[p], sd[p] = float(st["mean"][k].mean()), float(st["mean"][k].std(ddof=1))
            if definition == "C1":
                value = (value - mu[w.pos_a]) / np.where(sd[w.pos_a] > 1e-12, sd[w.pos_a], NAN)
                info = {**info, "position_mean": mu.tolist(), "position_sd": sd.tolist()}
            else:
                comp = components({k: np.where(enough, v, 0.0) if k != "mean" else v for k, v in st.items()}, w.pos_a)
                lam = shrink_factor(comp, n, G)
                value = np.where(n > 0, mu[w.pos_a] + lam * (value - mu[w.pos_a]), NAN)
                info = {**info, **{k: comp[k] for k in ("s2_window", "s2_game", "s2_athlete", "truncated_game", "truncated_athlete")},
                        "median_shrink": float(np.median(lam[n > 0])) if (n > 0).any() else NAN}
        self.last_info = info
        return {"ids": w.ids, "value": value, "n_windows": n.astype(np.int64), "n_games": G.astype(np.int64)}

    def eligible(self, definition: str) -> np.ndarray:
        """(A,) athletes meeting the exposure requirement under the definition, with ALL athletes fitting (counts only)."""
        return self.values(definition)["n_windows"] >= self.cc.min_windows

    def label_fn(self, definition: str, ids: np.ndarray) -> Callable[[np.ndarray | None], np.ndarray]:
        """train athlete ids → labels of ``ids`` with every context parameter estimated on those training athletes."""
        pos = np.searchsorted(self.w.ids, ids)
        if (pos >= self.w.ids.size).any() or not np.array_equal(self.w.ids[np.minimum(pos, self.w.ids.size - 1)], ids):
            raise ContextError("an athlete of the evaluation cohort has no window of this play measure")
        return lambda train_ids: self.values(definition, fit=self.train_mask(train_ids))["value"][pos]


# ============================================================================== contracts
DIST = TableContract(
    "nfl_context_distribution", "Distribution and scale of the play measure by observed context (windows nested in athletes).",
    {"group_type": S, "group": S, "position_group": S, "n_athletes": I, "n_windows": I, "athlete_mean": F, "athlete_sd": F,
     "window_p01": F, "window_p50": F, "window_p99": F, "window_min": F, "window_max": F, "share_implausible": F,
     "share_at_floor": F, "share_at_ceiling": F}, key=("group_type", "group", "position_group"), area="context group × position",
    docs={"athlete_mean": "mean of athlete means (athletes with ≥ min_half_windows windows in the group)",
          "athlete_sd": "SD of those athlete means — the between-athlete spread a scout would see, noise included",
          "share_implausible": "windows beyond the kinematic plausibility limit",
          "share_at_floor": "windows equal to the smallest observed value (a floor effect shows as a large share)"})
VARIANCE = TableContract(
    "nfl_context_variance_decomposition", "Nested variance components and context shares of the play measure.",
    {"scope": S, "component": S, "variance": F, "share_within_position": F, "share_total": F, "ci_lower": F, "ci_upper": F,
     "n_athletes": I, "n_athlete_games": I, "n_windows": I, "method": S, "truncated": B, "note": S},
    key=("scope", "component"), area="scope × component",
    docs={"component": "position · athlete · game_within_athlete · window_within_game (additive) · context:<factor> "
                       "(marginal η² within position, NOT additive with the nested components)",
          "share_within_position": "share of the within-position window variance", "share_total": "share of all window variance",
          "ci_lower": "95 % athlete bootstrap of share_within_position",
          "truncated": "the moment estimate was negative and set to 0: the layer is not distinguishable here"})
SCALE = TableContract(
    "nfl_context_scale_audit", "Is the between-athlete range large against the uncertainty of one athlete's value?",
    {"scope": S, "definition": S, "n_athletes": I, "between_sd": F, "p10": F, "p90": F, "median_windows": F, "median_games": F,
     "standard_error": F, "snr": F, "model_reliability": F, "unit": S}, key=("scope", "definition"), area="scope × definition",
    docs={"standard_error": "√(σ²_game / G + σ²_window / n) at the median exposure of the scope",
          "snr": "between-athlete SD (noise removed: σ_athlete) / standard error",
          "model_reliability": "σ²_a / (σ²_a + σ²_g / G + σ²_w / n) at the median exposure"})
OVERLAP = TableContract(
    "nfl_context_overlap", "Coverage and context overlap of every candidate outcome definition.",
    {"definition": S, "label": S, "status": S, "n_athletes_any": I, "n_athletes_eligible": I, "n_athlete_games": I, "n_windows": I,
     "share_windows_retained": F, "missing_context_share": F, "n_positions": I, "min_position_athletes": I, "position_counts": S,
     "cells_supported": I, "cells_total": I, "start_speed_overlap": F, "note": S}, key=("definition",), area="definition",
    docs={"n_athletes_eligible": "athletes with ≥ min_windows retained windows",
          "start_speed_overlap": "share of windows whose starting-speed band is supported in EVERY position group",
          "cells_supported": "context cells with enough windows and athletes to be compared"})
ADJUSTED = TableContract(
    "nfl_context_adjusted_outcomes", "Athlete-level value under every context definition (descriptive: all athletes fit the context).",
    {"nfl_id": I, "position_group": S, "cohort": I, "outcome_id": S, "definition": S, "value": F, "n_windows": I, "n_games": I,
     "eligible": B, "rank_in_position": F, "status": S}, key=("nfl_id", "outcome_id", "definition"), kind="dataset",
    area="athlete × outcome × definition",
    docs={"value": "DESCRIPTIVE (context parameters from all athletes); a transfer fold recomputes it from its training athletes",
          "rank_in_position": "0–1 rank among eligible athletes of the position", "status": "exploratory · reference_frozen_m05"})


# ============================================================================== descriptive tables
def distribution_table(w: Windows, cc: ContextConfig) -> pl.DataFrame:
    d = w.frame(cc)
    n_all = d.group_by("nfl_id").agg(pl.len().alias("_n"))
    d = d.join(n_all, on="nfl_id").with_columns(
        pl.when(pl.col("_n") < 100).then(pl.lit("< 100 windows")).when(pl.col("_n") < 400).then(pl.lit("100–399 windows"))
        .otherwise(pl.lit("≥ 400 windows")).alias("exposure_band"))
    lo, hi = float(np.nanmin(w.y)), float(np.nanmax(w.y))
    limit = cc.max_speed if "speed" in w.measure else cc.max_accel
    parts = []
    for gt, col in (("position", None), ("alignment", "alignment"), ("cohort", "cohort"), ("season", "season"),
                    ("exposure", "exposure_band"), ("start_band", "start_band"), ("play_class", "play_class")):
        g = d.with_columns((pl.lit(ALL) if col is None else pl.col(col).cast(pl.String).fill_null(NA)).alias("group"))
        ath = (g.group_by("group", "position_group", "nfl_id").agg(pl.len().alias("n"), pl.col("y").mean().alias("m"))
               .filter(pl.col("n") >= cc.min_half_windows).group_by("group", "position_group")
               .agg(pl.len().cast(pl.Int64).alias("n_athletes"), pl.col("m").mean().alias("athlete_mean"), pl.col("m").std().alias("athlete_sd")))
        win = g.group_by("group", "position_group").agg(
            pl.len().cast(pl.Int64).alias("n_windows"), pl.col("y").quantile(0.01).alias("window_p01"),
            pl.col("y").median().alias("window_p50"), pl.col("y").quantile(0.99).alias("window_p99"),
            pl.col("y").min().alias("window_min"), pl.col("y").max().alias("window_max"),
            (pl.col("y").abs() > limit).mean().alias("share_implausible"), (pl.col("y") <= lo + 1e-9).mean().alias("share_at_floor"),
            (pl.col("y") >= hi - 1e-9).mean().alias("share_at_ceiling"))
        parts.append(win.join(ath, on=["group", "position_group"], how="left").with_columns(
            pl.lit(gt).alias("group_type"), pl.col("n_athletes").fill_null(0)))
    out = pl.concat(parts, how="diagonal")
    return out.select([pl.col(k).cast(t) for k, t in DIST.columns.items()]).sort("group_type", "group", "position_group")


def _eta2(code: np.ndarray, y: np.ndarray, pos: np.ndarray) -> tuple[float, int]:
    """Marginal η² of a factor WITHIN position: SS between position × level cells beyond position / SS within position."""
    ok = code >= 0
    yy, pp = y[ok], pos[ok]
    P = int(pp.max()) + 1
    mp = np.bincount(pp, weights=yy, minlength=P) / np.maximum(np.bincount(pp, minlength=P), 1)
    c = yy - mp[pp]
    L = int(code[ok].max()) + 1
    n = np.bincount(code[ok], minlength=L)
    s = np.bincount(code[ok], weights=c, minlength=L)
    tot = float((c ** 2).sum())
    return (float((s[n > 0] ** 2 / n[n > 0]).sum()) / tot if tot > 0 else NAN), int((n > 0).sum())


def variance_table(lab: ContextLabels, seed: int) -> pl.DataFrame:
    """Nested components (pooled within position, and per position) with athlete-bootstrap intervals of the shares,
    the between-position share, and marginal context shares (η² within position)."""
    w, cc = lab.w, lab.cc
    st = athlete_stats(w, w.y, lab._all)
    rows: list[dict[str, Any]] = []
    grand = float(w.y.mean())
    mp = np.bincount(w.pos_w, weights=w.y) / np.maximum(np.bincount(w.pos_w), 1)
    v_tot = float(((w.y - grand) ** 2).mean())
    v_pos = float(((mp[w.pos_w] - grand) ** 2).mean())
    rng = np.random.default_rng(seed_for(seed, "ctx", "variance"))
    for scope in (ALL, *w.pos_levels.tolist()):
        k = np.ones(w.n_athletes, dtype=bool) if scope == ALL else w.position == scope
        base = np.where(k, 1.0, 0.0)
        c = components(st, w.pos_a, base)
        tot = sum(float(np.nan_to_num(c[x])) for x in ("s2_athlete", "s2_game", "s2_window"))
        idx = np.flatnonzero(k & (st["n"] > 0))
        boot = np.full((cc.n_boot, 3), NAN)
        if idx.size >= cc.min_athletes_position:
            for b in range(cc.n_boot):
                m = np.bincount(rng.choice(idx, idx.size), minlength=w.n_athletes).astype(float)
                cb = components(st, w.pos_a, m)
                tb = sum(float(np.nan_to_num(cb[x])) for x in ("s2_athlete", "s2_game", "s2_window"))
                boot[b] = [float(np.nan_to_num(cb[x])) / tb if tb > 0 else NAN for x in ("s2_athlete", "s2_game", "s2_window")]
        size = {"n_athletes": c["n_athletes"], "n_athlete_games": c["n_athlete_games"], "n_windows": c["n_windows"]}
        within_total = 1.0 - (v_pos / v_tot if v_tot > 0 else NAN)
        if scope == ALL:
            rows.append({"scope": scope, "component": "position", "variance": v_pos, "share_within_position": None,
                         "share_total": v_pos / v_tot if v_tot > 0 else None, "ci_lower": None, "ci_upper": None, **size,
                         "method": "between-position sum of squares / windows", "truncated": False,
                         "note": "position is a fixed, observed label: this share is not an athlete difference"})
        for j, (name, key, trunc) in enumerate((("athlete", "s2_athlete", c["truncated_athlete"]),
                                                ("game_within_athlete", "s2_game", c["truncated_game"]),
                                                ("window_within_game", "s2_window", False))):
            v = float(np.nan_to_num(c[key]))
            ok = boot[:, j][np.isfinite(boot[:, j])]
            share = v / tot if tot > 0 else None
            rows.append({"scope": scope, "component": name, "variance": v, "share_within_position": share,
                         "share_total": share * within_total if share is not None and scope == ALL else None,
                         "ci_lower": float(np.quantile(ok, 0.025)) if ok.size >= 20 else None,
                         "ci_upper": float(np.quantile(ok, 0.975)) if ok.size >= 20 else None, **size,
                         "method": "nested ANOVA, method of moments (unbalanced), athlete bootstrap", "truncated": bool(trunc),
                         "note": "moment estimate truncated at 0" if trunc else ""})
    nb = len(cc.start_bands)
    for name, code in (("alignment", lab._pos_align), ("alignment_x_play_class", lab._pos_cell),
                       ("starting_speed_band", lab._pos_band), ("season", w.pos_w * (int(lab._season.max()) + 1) + lab._season),
                       ("down", w.pos_w * (int(lab._down.max()) + 1) + lab._down)):
        if not (code >= 0).any():
            continue
        e, L = _eta2(code, w.y, w.pos_w)
        rows.append({"scope": ALL, "component": f"context:{name}", "variance": None, "share_within_position": e, "share_total": None,
                     "ci_lower": None, "ci_upper": None, "n_athletes": w.n_athletes, "n_athlete_games": int(w.ag_athlete.size),
                     "n_windows": int((code >= 0).sum()), "method": "marginal η² of position × level cells beyond position",
                     "truncated": False, "note": f"{L} cells · window level, descriptive: not additive with the nested components"
                                                  + (f" · {nb} bands" if name == "starting_speed_band" else "")})
    return pl.DataFrame(rows, schema=VARIANCE.columns).sort("scope", "component")


def scale_table(lab: ContextLabels, definitions: Sequence[str]) -> pl.DataFrame:
    """Between-athlete spread against the standard error of one athlete's value, at the median exposure."""
    w, cc = lab.w, lab.cc
    rows = []
    for d in definitions:
        if not lab.available(d)[0] or d == "C1":
            continue
        adj, keep, _ = lab.adjust(d)
        st = athlete_stats(w, adj, keep)
        elig = st["n"] >= cc.min_windows
        for scope in (ALL, *w.pos_levels.tolist()):
            k = elig & (np.ones(w.n_athletes, dtype=bool) if scope == ALL else w.position == scope)
            if k.sum() < 3:
                continue
            c = components(st, w.pos_a, k.astype(float))
            n_med, g_med = float(np.median(st["n"][k])), float(np.median(st["G"][k]))
            se = float(np.sqrt(np.nan_to_num(c["s2_game"]) / max(g_med, 1) + np.nan_to_num(c["s2_window"]) / max(n_med, 1)))
            centred = st["mean"][k] - np.array([st["mean"][k & (w.pos_a == p)].mean() for p in w.pos_a[k]])
            sa = float(np.sqrt(np.nan_to_num(c["s2_athlete"])))
            rows.append({"scope": scope, "definition": d, "n_athletes": int(k.sum()), "between_sd": float(centred.std(ddof=1)),
                         "p10": float(np.quantile(centred, 0.10)), "p90": float(np.quantile(centred, 0.90)), "median_windows": n_med,
                         "median_games": g_med, "standard_error": se, "snr": sa / se if se > 0 else None,
                         "model_reliability": float(shrink_factor(c, np.array([n_med]), np.array([g_med]))[0]), "unit": w.unit})
    return pl.DataFrame(rows, schema=SCALE.columns) if rows else SCALE.empty()


def overlap_table(lab: ContextLabels, definitions: Sequence[str]) -> pl.DataFrame:
    w, cc = lab.w, lab.cc
    rows = []
    nb = len(cc.start_bands)
    sup_all = NAN
    if lab.has_start:
        _, n, ath = _cell_means(lab._pos_band, w.y, lab._all, w.a, w.pos_levels.size * nb, w.n_athletes)
        sup = ((n >= cc.min_cell_windows) & (ath >= cc.min_cell_athletes)).reshape(w.pos_levels.size, nb).all(0)
        sup_all = float(sup[np.maximum(lab._band, 0)][lab._band >= 0].sum() / max(w.n_windows, 1))
    for d in definitions:
        ok, why = lab.available(d)
        meta = DEFINITIONS[d]
        if not ok:
            rows.append({"definition": d, "label": meta["label"], "status": "unavailable", "n_athletes_any": 0, "n_athletes_eligible": 0,
                         "n_athlete_games": 0, "n_windows": 0, "share_windows_retained": 0.0, "missing_context_share": None,
                         "n_positions": 0, "min_position_athletes": 0, "position_counts": "", "cells_supported": None,
                         "cells_total": None, "start_speed_overlap": None, "note": why})
            continue
        _, keep, info = lab.adjust(d)
        n = np.bincount(w.a[keep], minlength=w.n_athletes)
        elig = n >= cc.min_windows
        counts = {p: int((elig & (w.position == p)).sum()) for p in w.pos_levels.tolist()}
        rows.append({"definition": d, "label": meta["label"], "status": meta["status"], "n_athletes_any": int((n > 0).sum()),
                     "n_athletes_eligible": int(elig.sum()), "n_athlete_games": int(np.unique(w.ag[keep]).size), "n_windows": int(keep.sum()),
                     "share_windows_retained": float(keep.mean()), "missing_context_share": info.get("missing_context_share"),
                     "n_positions": int(sum(v >= cc.min_athletes_position for v in counts.values())),
                     "min_position_athletes": int(min(counts.values())) if counts else 0,
                     "position_counts": " · ".join(f"{p} {v}" for p, v in counts.items()), "cells_supported": info.get("supported_cells"),
                     "cells_total": info.get("total_cells"), "start_speed_overlap": sup_all if d in ("C2", "C4") and lab.has_start else None,
                     "note": ("windows in unsupported cells are DROPPED, not left unadjusted" if d in ("C2", "C3") else
                              f"backfitting converged = {info.get('converged')}, {info.get('iterations')} iteration(s)" if d == "C4" else "")})
    return pl.DataFrame(rows, schema=OVERLAP.columns)


def adjusted_table(lab: ContextLabels, definitions: Sequence[str]) -> pl.DataFrame:
    """Descriptive athlete values under every available definition (context parameters from ALL athletes)."""
    w, cc = lab.w, lab.cc
    parts = []
    for d in definitions:
        if not lab.available(d)[0]:
            continue
        v = lab.values(d)
        elig = v["n_windows"] >= cc.min_windows
        rank = np.full(w.n_athletes, NAN)
        for p in w.pos_levels.tolist():
            k = elig & (w.position == p) & np.isfinite(v["value"])
            if k.sum() >= 2:
                rank[k] = np.argsort(np.argsort(v["value"][k])) / (k.sum() - 1)
        parts.append(pl.DataFrame({"nfl_id": w.ids, "position_group": w.position, "cohort": w.cohort, "value": v["value"],
                                   "n_windows": v["n_windows"], "n_games": v["n_games"], "eligible": elig, "rank_in_position": rank})
                     .with_columns(pl.lit(w.outcome_id).alias("outcome_id"), pl.lit(d).alias("definition"),
                                   pl.lit(DEFINITIONS[d]["status"]).alias("status"), pl.col("cohort").fill_nan(None),
                                   pl.col("value").fill_nan(None), pl.col("rank_in_position").fill_nan(None)))
    if not parts:
        return ADJUSTED.empty()
    return pl.concat(parts).select([pl.col(k).cast(t) for k, t in ADJUSTED.columns.items()]).sort("nfl_id", "definition")


def rank_shift_examples(adjusted: pl.DataFrame, a: str = "C0", b: str = "C4", k: int = 8) -> pl.DataFrame:
    """Athletes whose within-position rank moves most between two definitions (both eligible). Illustration only:
    a large move says the two definitions disagree about the athlete, not which one is right."""
    x = adjusted.filter((pl.col("definition") == a) & pl.col("eligible")).select("nfl_id", "position_group", pl.col("value").alias(f"value_{a}"),
                                                                                pl.col("rank_in_position").alias(f"rank_{a}"), "n_windows")
    y = adjusted.filter((pl.col("definition") == b) & pl.col("eligible")).select("nfl_id", pl.col("value").alias(f"value_{b}"),
                                                                                pl.col("rank_in_position").alias(f"rank_{b}"))
    j = x.join(y, on="nfl_id", how="inner").with_columns((pl.col(f"rank_{b}") - pl.col(f"rank_{a}")).alias("rank_shift"))
    return j.sort(pl.col("rank_shift").abs(), descending=True).head(k)


def target_registry(lab: ContextLabels, overlap: pl.DataFrame, reliability: pl.DataFrame, frozen_version_hash: str,
                    definitions: Sequence[str]) -> dict[str, Any]:
    """Registry of the EXPLORATORY context definitions (``nfl_context_target_registry.json``). It never replaces
    ``nfl_movement_targets_v1.json`` and carries the hash of the frozen registry it was derived beside."""
    rel = {r["definition"]: r for r in reliability.filter((pl.col("method") == "random_game_halves_crossfit") & (pl.col("scope") == ALL)).iter_rows(named=True)} \
        if reliability.height else {}
    ov = {r["definition"]: r for r in overlap.iter_rows(named=True)}
    cc = lab.cc
    return {"artifact": "nfl_context_target_registry", "status": "EXPLORATORY — not a frozen target, never a confirmatory outcome",
            "derived_beside_frozen_registry": frozen_version_hash, "outcome_id": lab.w.outcome_id, "play_measure": lab.w.measure,
            "units": lab.w.unit, "config": cc.params(),
            "definitions": [{"definition": d, **{k: DEFINITIONS[d][k] for k in ("label", "complexity", "status")},
                             "context_parameters_from": "the fit windows only (training athletes of a fold · the other game half of a split)",
                             "n_athletes_eligible": (ov.get(d) or {}).get("n_athletes_eligible"),
                             "reliability": (rel.get(d) or {}).get("reliability"),
                             "reliability_ci": [(rel.get(d) or {}).get("ci_lower"), (rel.get(d) or {}).get("ci_upper")],
                             "interpretation_limit": "observed movement after removing what observed context labels explain; "
                                                     "not athletic capability, not football performance"} for d in definitions]}
