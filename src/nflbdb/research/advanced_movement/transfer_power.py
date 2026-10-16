"""Detectability, power and information limits of the Combine → NFL transfer study (Milestone 07C). NumPy / Polars.

Question        could the linked cohort (about 284 athletes) have detected a practically meaningful incremental
                predictive effect if one existed? An interval that includes 0 says "not established"; it does not say
                how large an effect would have been needed to be established.
Synthetic       REAL cohort structure, SYNTHETIC targets. No real NFL label is modified or written:
                    y* = μ_i + σ_res,pos · ( √(1 − f) · ε_i + √f · z_i )
                μ_i     the athlete's Milestone 06 out-of-fold prediction of the official model M1 (structure explained
                        by position, cohort, body and official tests — a function of pre-NFL covariates)
                σ_res   SD of the real M1 out-of-fold residuals of the athlete's position
                ε_i     residuals of the real fit, resampled within position (keeps their shape, breaks every link)
                z_i     the planted signal: a direction of the candidate representation with unit variance, either
                        'conditional' (orthogonal to the official design: f IS the share of residual variance, i.e. the
                        partial R² given official testing) or 'raw' (the plain direction: part of it is already
                        explained by official tests — the redundancy case)
Pipeline        the SAME athlete-held-out ridge pipeline as Milestone 06 (``transfer_models.run_oof``: folds, inner
                penalty search, train-only preprocessing, fold-fitted representations), then the SAME paired athlete
                bootstrap and the SAME criterion (95 % interval of Δ RMSE above 0).
Cost            one replicate = two ridge models × outer folds; bounded by SIMULATION_REPLICATES. Results are written
                to a checkpoint every PROGRESS_INTERVAL replicates and resumed by configuration hash; an incomplete
                checkpoint is never read as a finished result.
Limits          power is conditional on the planted direction, on the Gaussian-free but homoscedastic-within-position
                noise model and on the Milestone 06 cohort; sample-size scenarios resample the observed athletes and are
                scenario analysis, not a forecast of available data.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import get_logger, stable_hash
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from . import transfer_models as tm
from .config import NotebookControls
from .context_outcomes import Progress
from .context_reliability import GATE
from .transfer import TransferConfig

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
POWER_VERSION = "1"
ALL = "ALL"
RESUME_MODES = ("resume", "restart", "require_complete")
CANDIDATES = {"M2": "ENG", "M3": "SPEED2", "M4": "FPCA", "M5": "TCN"}        # model → representation family carrying the planted signal


class PowerError(RuntimeError):
    """A simulation cannot run, or a checkpoint is incomplete / belongs to another configuration."""


@dataclass(frozen=True)
class PowerConfig:
    version: str = POWER_VERSION
    seed: int = 20261011
    effects: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35)   # f: share of the residual outcome variance
    replicates: int = 200                                             # execution: per scenario
    n_boot: int = 500                                                 # execution: paired athlete bootstrap per replicate
    modes: tuple[str, ...] = ("conditional", "raw")
    candidates: tuple[str, ...] = ("M5", "M2")                         # established: TCN-B (primary of M06), engineered
    reference: str = "M1"
    sample_sizes: tuple[int, ...] = (150, 284, 500, 1000, 2000)        # proposed: linked-cohort scenarios
    sample_effects: tuple[float, ...] = (0.02, 0.05, 0.10)
    sample_replicates: int = 100
    signal_fold: int = 0                                              # proposed: fold encoder defining the planted direction
    target_power: float = 0.80                                        # conventional
    alpha: float = 0.05                                               # conventional (two-sided interval)
    min_position: int = 15                                            # proposed: athletes for a position-level row
    progress_interval: int = 10                                       # execution: replicates between checkpoints

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def for_mode(self, controls: NotebookControls) -> "PowerConfig":
        if not controls.smoke:
            return replace(self, seed=controls.seed)
        return replace(self, seed=controls.seed, effects=(0.0, 0.20), replicates=6, n_boot=60, candidates=self.candidates[:1],
                       modes=self.modes[:1], sample_sizes=(60, 120), sample_effects=(0.10,), sample_replicates=4, min_position=4,
                       progress_interval=2)


# ============================================================================== contracts
REPLICATES = TableContract(
    "transfer_planted_signal_recovery", "One row per synthetic replicate × scope: planted effect and recovered paired gain.",
    {"config_hash": S, "study": S, "candidate": S, "mode": S, "effect": F, "n_athletes": I, "replicate": I, "scope": S, "n_scope": I,
     "delta_rmse": F, "ci_lower": F, "ci_upper": F, "relative": F, "delta_r2": F, "rmse_reference": F, "detected": B, "worse": B,
     "seconds": F}, key=("study", "candidate", "mode", "effect", "n_athletes", "replicate", "scope"), kind="dataset",
    area="scenario × replicate × scope",
    docs={"study": "power = the real cohort and folds · sample_size = resampled cohort of n_athletes with re-dealt folds",
          "effect": "f = planted share of the residual outcome variance (conditional mode: partial R² given official testing)",
          "delta_rmse": "RMSE(reference) − RMSE(candidate) on the synthetic target; positive = the candidate is better",
          "detected": "95 % paired athlete-bootstrap interval above 0 — the Milestone 06 criterion", "scope": "ALL or a position group"})
SCENARIOS = TableContract(
    "transfer_power_scenarios", "Detection rate, recovered gain and interval coverage per scenario.",
    {"study": S, "candidate": S, "mode": S, "effect": F, "n_athletes": I, "scope": S, "n_scope": F, "n_replicates": I, "power": F,
     "power_lower": F, "power_upper": F, "rate_worse": F, "mean_delta_rmse": F, "sd_delta_rmse": F, "mean_relative": F,
     "mean_delta_r2": F, "ci_coverage": F, "mean_ci_width": F, "complete": B},
    key=("study", "candidate", "mode", "effect", "n_athletes", "scope"), area="scenario × scope",
    docs={"power": "share of replicates with the interval above 0 (effect 0: the false-positive rate)",
          "power_lower": "95 % Wilson interval of the rate over replicates (Monte-Carlo error)",
          "ci_coverage": "share of replicate intervals containing the scenario's mean Δ RMSE over replicates",
          "complete": "every planned replicate of the scenario is present"})
NULL = TableContract(
    "transfer_null_calibration", "Behaviour of the pipeline when the planted effect is zero.",
    {"study": S, "candidate": S, "mode": S, "n_athletes": I, "scope": S, "n_replicates": I, "false_positive_rate": F, "fpr_lower": F,
     "fpr_upper": F, "rate_worse": F, "mean_delta_rmse": F, "sd_delta_rmse": F, "q95_delta_rmse": F, "ci_coverage_of_zero": F,
     "nominal_one_sided": F}, key=("study", "candidate", "mode", "n_athletes", "scope"), area="scenario × scope",
    docs={"false_positive_rate": "share of null replicates declared detected; nominal 0.025 for a one-sided reading of a 95 % interval",
          "mean_delta_rmse": "expected Δ RMSE with no signal: usually slightly NEGATIVE (useless predictors cost a little)"})
MDE = TableContract(
    "transfer_mde_estimates", "Smallest planted effect detected with the target power, and the gain it corresponds to.",
    {"study": S, "candidate": S, "mode": S, "n_athletes": I, "scope": S, "target_power": F, "mde_effect": F, "mde_delta_rmse": F,
     "mde_relative": F, "mde_delta_r2": F, "max_power_reached": F, "method": S, "note": S},
    key=("study", "candidate", "mode", "n_athletes", "scope"), area="scenario × scope",
    docs={"mde_effect": "f at which the interpolated power curve reaches the target (None when it never does in the grid)",
          "mde_delta_rmse": "mean recovered Δ RMSE interpolated at that effect"})
LIMITS = TableContract(
    "transfer_information_limits", "What limits an observable Combine → NFL association: precision, attenuation, redundancy.",
    {"topic": S, "item": S, "value": F, "lower": F, "upper": F, "unit": S, "note": S}, key=("topic", "item"), area="topic × item")


# ============================================================================== synthetic targets
@dataclass
class Structure:
    """Real cohort structure the synthetic targets are built on (nothing here is a synthetic value)."""
    design: tm.Design
    mu: np.ndarray                  # (n,) M1 out-of-fold prediction of the real target
    resid: np.ndarray               # (n,) real M1 out-of-fold residual
    sigma: np.ndarray               # (n,) residual SD of the athlete's position
    signal: dict[tuple[str, str], np.ndarray]     # (candidate, mode) → (n,) unit-variance planted direction
    redundancy: dict[str, float]    # candidate → R² of the raw direction on the official design
    target_id: str
    rmse_m1: float


def _official_matrix(d: tm.Design, spec: tm.ModelSpec) -> np.ndarray:
    X, _, _, _ = tm.design_matrix(d, spec, [], np.ones(d.n, dtype=bool))
    return np.column_stack([np.ones(d.n), X])


def planted_direction(Z: np.ndarray, official: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(raw direction, conditional direction, R² of raw on official), both with zero mean and unit variance. The raw
    direction is the first principal axis of the standardised block; the conditional one is its residual after
    least-squares projection on the official design."""
    Zs = (Z - Z.mean(0)) / np.where(Z.std(0) > 1e-12, Z.std(0), 1.0)
    _, _, vt = np.linalg.svd(Zs, full_matrices=False)
    raw = Zs @ vt[0]
    raw = (raw - raw.mean()) / raw.std()
    beta, *_ = np.linalg.lstsq(official, raw, rcond=None)
    res = raw - official @ beta
    r2 = 1.0 - float(res.var() / raw.var())
    if res.std() < 1e-8:
        raise PowerError("the representation direction is fully explained by the official design: no conditional signal can be planted")
    return raw, (res - res.mean()) / res.std(), r2


def build_structure(d: tm.Design, oof: pl.DataFrame, blocks: Any, tc: TransferConfig, pc: PowerConfig, target_id: str) -> Structure:
    """Structure from the Milestone 06 primary analysis: ``oof`` = ``transfer_oof_predictions`` (read only)."""
    o = oof.filter((pl.col("analysis") == "primary") & (pl.col("cohort") == "primary") & (pl.col("target_id") == target_id)
                   & (pl.col("model") == pc.reference)).sort("nfl_id")
    ids = o["nfl_id"].to_numpy().astype(np.int64)
    if not np.array_equal(ids, d.ids):
        raise PowerError(f"the Milestone 06 out-of-fold table holds {ids.size} athletes for {target_id}, the design {d.n}: "
                         "restrict the design to the paired cohort first")
    mu, y = o["pred"].to_numpy().astype(float), o["y"].to_numpy().astype(float)
    resid = y - mu
    sigma = np.empty(d.n)
    for p in np.unique(d.position):
        k = d.position == p
        sigma[k] = resid[k].std(ddof=1) if k.sum() > 2 else resid.std(ddof=1)
    ladder = {m.model: m for m in tm.model_ladder(tc)}
    X = _official_matrix(d, ladder[pc.reference])
    signal, red = {}, {}
    for c in pc.candidates:
        raw, cond, r2 = planted_direction(blocks.matrix(CANDIDATES[c], pc.signal_fold, d.ids), X)
        signal[(c, "raw")], signal[(c, "conditional")], red[c] = raw, cond, r2
    return Structure(d, mu, resid, sigma, signal, red, target_id, float(np.sqrt(np.mean(resid ** 2))))


def synthetic_target(st: Structure, candidate: str, mode: str, effect: float, rng: np.random.Generator, rows: np.ndarray | None = None
                     ) -> np.ndarray:
    """One synthetic target. ``rows``: athlete rows of a resampled cohort (None = the real cohort)."""
    idx = np.arange(st.design.n) if rows is None else rows
    pos = st.design.position[idx]
    eps = np.empty(idx.size)
    std = st.resid / st.sigma
    for p in np.unique(pos):
        k = pos == p
        pool = std[st.design.position == p]
        e = rng.choice(pool, int(k.sum()), replace=True)
        eps[k] = (e - pool.mean()) / (pool.std() if pool.std() > 1e-12 else 1.0)
    f = float(effect)
    return st.mu[idx] + st.sigma[idx] * (np.sqrt(1.0 - f) * eps + np.sqrt(f) * st.signal[(candidate, mode)][idx])


# ============================================================================== one replicate
def _scope_rows(position: np.ndarray, pc: PowerConfig) -> list[tuple[str, np.ndarray]]:
    out = [(ALL, np.ones(position.size, dtype=bool))]
    return out + [(str(p), position == p) for p in np.unique(position) if (position == p).sum() >= pc.min_position]


def replicate_rows(y: np.ndarray, d: tm.Design, blocks: Any, models: Sequence[tm.ModelSpec], candidate: str, tc: TransferConfig,
                   pc: PowerConfig, rng: np.random.Generator) -> list[dict[str, Any]]:
    """The Milestone 06 pipeline on one synthetic target: out-of-fold predictions of reference and candidate, paired
    athlete bootstrap of Δ RMSE, for the whole cohort and for each position group large enough."""
    res, _ = tm.run_oof(d, lambda train_ids: y, blocks, list(models), tc)
    er = (res["y"] - res["pred"][pc.reference]) ** 2
    ec = (res["y"] - res["pred"][candidate]) ** 2
    e0 = (res["y"] - res["y_ref"]) ** 2
    rows = []
    for scope, k in _scope_rows(d.position, pc):
        a, b, z = er[k], ec[k], e0[k]
        n = a.size
        idx = rng.integers(0, n, size=(pc.n_boot, n))
        reps = np.sqrt(a[idx].mean(1)) - np.sqrt(b[idx].mean(1))
        lo, hi = float(np.quantile(reps, pc.alpha / 2)), float(np.quantile(reps, 1 - pc.alpha / 2))
        rr = float(np.sqrt(a.mean()))
        dl = rr - float(np.sqrt(b.mean()))
        rows.append({"scope": scope, "n_scope": int(n), "delta_rmse": dl, "ci_lower": lo, "ci_upper": hi, "relative": dl / rr if rr > 0 else None,
                     "delta_r2": float((a.mean() - b.mean()) / z.mean()) if z.mean() > 0 else None, "rmse_reference": rr,
                     "detected": bool(lo > 0), "worse": bool(hi < 0)})
    return rows


def deal_folds(position: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Athlete folds balanced within position (a random deal: no outcome is involved)."""
    fold = np.empty(position.size, dtype=np.int64)
    for p in np.unique(position):
        idx = rng.permutation(np.flatnonzero(position == p))
        fold[idx] = (np.arange(idx.size) + rng.integers(0, k)) % k
    return fold


def resampled_cohort(st: Structure, blocks: Any, n: int, families: Sequence[str], pc: PowerConfig, tc: TransferConfig,
                     rng: np.random.Generator) -> tuple[tm.Design, tm.ArrayBlocks, np.ndarray]:
    """A cohort of ``n`` athletes drawn with replacement from the observed one within position shares (each draw is a NEW
    athlete with the covariates and representation of the drawn one), with re-dealt outer and inner folds. The
    representation is the one of the ``signal_fold`` encoder for everybody (a fold-independent block): a scenario
    analysis of sampling precision, not an inductive evaluation."""
    d = st.design
    rows = np.sort(rng.choice(d.n, int(n), replace=True))
    k_outer, k_inner = int(np.unique(d.fold).size), int(tc.inner_folds)
    dd = replace(d, ids=np.arange(int(n), dtype=np.int64), position=d.position[rows], cohort=d.cohort[rows],
                 numeric={c: v[rows] for c, v in d.numeric.items()}, fold=deal_folds(d.position[rows], k_outer, rng),
                 inner=deal_folds(d.position[rows], k_inner, rng))
    arr = tm.ArrayBlocks(dd.ids, {f: blocks.matrix(f, pc.signal_fold, d.ids)[rows] for f in families})
    return dd, arr, rows


# ============================================================================== resumable simulation
def scenario_plan(pc: PowerConfig, n_cohort: int) -> pl.DataFrame:
    """Every (study, candidate, mode, effect, n_athletes) scenario with its replicate budget."""
    rows = [{"study": "power", "candidate": c, "mode": m, "effect": float(e), "n_athletes": int(n_cohort), "replicates": pc.replicates}
            for c in pc.candidates for m in pc.modes for e in pc.effects]
    rows += [{"study": "sample_size", "candidate": c, "mode": "conditional", "effect": float(e), "n_athletes": int(n), "replicates": pc.sample_replicates}
             for c in pc.candidates[:1] for n in pc.sample_sizes for e in (0.0, *pc.sample_effects)]
    return pl.DataFrame(rows, schema={"study": S, "candidate": S, "mode": S, "effect": F, "n_athletes": I, "replicates": I}).unique(
        subset=["study", "candidate", "mode", "effect", "n_athletes"], maintain_order=True)


def config_hash(pc: PowerConfig, tc: TransferConfig, st: Structure, extra: dict[str, Any] | None = None) -> str:
    """Identity of a simulation: its configuration, the cohort, the folds and the real structure it is built on."""
    return stable_hash({"power": {k: v for k, v in pc.params().items() if k != "progress_interval"},
                        "transfer": {"alpha_grid": tc.alpha_grid, "inner_folds": tc.inner_folds}, "target": st.target_id,
                        "ids": stable_hash(st.design.ids.tolist()), "folds": stable_hash(st.design.fold.tolist()),
                        "mu": round(float(st.mu.sum()), 8), "resid": round(float((st.resid ** 2).sum()), 8), **(extra or {})})


def _scenario_key(r: dict[str, Any]) -> tuple:
    return (r["study"], r["candidate"], r["mode"], round(float(r["effect"]), 6), int(r["n_athletes"]))


def load_checkpoint(path: Path, chash: str, resume_mode: str) -> pl.DataFrame:
    """Replicate rows already computed for THIS configuration. 'restart' ignores the file; a file of another
    configuration is never reused (it is reported and left in place)."""
    if resume_mode not in RESUME_MODES:
        raise PowerError(f"RESUME_MODE must be one of {RESUME_MODES}, got {resume_mode!r}")
    if resume_mode == "restart" or not path.is_file():
        if resume_mode == "require_complete":
            raise PowerError(f"RESUME_MODE = 'require_complete' but {path.name} does not exist: run the simulation first")
        return REPLICATES.empty()
    have = pl.read_parquet(path)
    mine = have.filter(pl.col("config_hash") == chash)
    if mine.height < have.height:
        log.warning("[advanced_movement:POWER] %s holds %d row(s) of another configuration: ignored, not deleted", path.name, have.height - mine.height)
    return mine.select([pl.col(k).cast(t) for k, t in REPLICATES.columns.items()])


def _write_checkpoint(path: Path, rows: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    rows.write_parquet(tmp)
    tmp.replace(path)


def missing_replicates(plan: pl.DataFrame, done: pl.DataFrame) -> list[tuple[dict[str, Any], list[int]]]:
    """[(scenario, replicate indices still to compute)] — a replicate counts only when its ALL-scope row exists."""
    have: dict[tuple, set[int]] = {}
    for r in done.filter(pl.col("scope") == ALL).iter_rows(named=True):
        have.setdefault(_scenario_key(r), set()).add(int(r["replicate"]))
    out = []
    for s in plan.iter_rows(named=True):
        todo = [b for b in range(int(s["replicates"])) if b not in have.get(_scenario_key(s), set())]
        if todo:
            out.append((s, todo))
    return out


def run_simulation(st: Structure, blocks: Any, tc: TransferConfig, pc: PowerConfig, checkpoint: Path, resume_mode: str = "resume",
                   save: bool = True, max_seconds: float | None = None) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run (or resume) every scenario of ``scenario_plan``. Prints the planned work, a timing from the first replicate,
    then iteration-level progress; writes the checkpoint every ``progress_interval`` replicates. ``max_seconds``
    stops cleanly after the current replicate (the checkpoint is kept and the result is marked incomplete).
    Returns (replicate rows of this configuration, run summary)."""
    chash = config_hash(pc, tc, st)
    plan = scenario_plan(pc, st.design.n)
    done = load_checkpoint(checkpoint, chash, resume_mode)
    todo = missing_replicates(plan, done)
    total = sum(len(t) for _, t in todo)
    planned = int(plan["replicates"].sum())
    print(f"[07C power] configuration {chash} · {plan.height} scenario(s) · {planned} replicate(s) planned · "
          f"{planned - total} found in the checkpoint · {total} to compute")
    if resume_mode == "require_complete" and total:
        raise PowerError(f"RESUME_MODE = 'require_complete' but {total} replicate(s) are missing for configuration {chash}")
    ladder = {m.model: m for m in tm.model_ladder(tc)}
    bar = Progress(total, "07C power", every=pc.progress_interval)
    buffer: list[dict[str, Any]] = []
    t_start, stopped, timed = time.perf_counter(), False, False
    for s, reps in todo:
        cand = s["candidate"]
        models = [ladder[pc.reference], ladder[cand]]
        for b in reps:
            t0 = time.perf_counter()
            rng = np.random.default_rng(seed_for(pc.seed, "power", s["study"], cand, s["mode"], f"{s['effect']:.6f}", str(s["n_athletes"]), str(b)))
            if s["study"] == "power":
                y = synthetic_target(st, cand, s["mode"], s["effect"], rng)
                rows = replicate_rows(y, st.design, blocks, models, cand, tc, pc, rng)
            else:
                dd, arr, pick = resampled_cohort(st, blocks, s["n_athletes"], ladder[cand].families, pc, tc, rng)
                y = synthetic_target(st, cand, s["mode"], s["effect"], rng, pick)
                rows = replicate_rows(y, dd, arr, models, cand, tc, pc, rng)
            sec = time.perf_counter() - t0
            buffer += [{"config_hash": chash, "study": s["study"], "candidate": cand, "mode": s["mode"], "effect": float(s["effect"]),
                        "n_athletes": int(s["n_athletes"]), "replicate": int(b), **r, "seconds": sec} for r in rows]
            if not timed:
                print(bar.estimate(sec) + " (from the first replicate; larger cohorts take longer)")
                timed = True
            bar.step(note=f"{s['study']} {cand} {s['mode']} f = {s['effect']:g} n = {s['n_athletes']}")
            if save and bar.done % pc.progress_interval == 0:
                done = pl.concat([done, pl.DataFrame(buffer, schema=REPLICATES.columns)])
                buffer = []
                _write_checkpoint(checkpoint, done)
            if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
                stopped = True
                break
        if stopped:
            break
    if buffer:
        done = pl.concat([done, pl.DataFrame(buffer, schema=REPLICATES.columns)])
        if save:
            _write_checkpoint(checkpoint, done)
    left = sum(len(t) for _, t in missing_replicates(plan, done))
    summary = {"config_hash": chash, "planned_replicates": planned, "computed_now": bar.done, "missing": left, "complete": left == 0,
               "stopped_by_time_limit": stopped, "seconds": time.perf_counter() - t_start, "checkpoint": checkpoint.name}
    print(f"[07C power] {'COMPLETE' if left == 0 else f'INCOMPLETE — {left} replicate(s) missing; rerun the cell to resume'}")
    return done.sort("study", "candidate", "mode", "effect", "n_athletes", "replicate", "scope"), summary


def assert_complete(summary: dict[str, Any]) -> None:
    if not summary.get("complete"):
        raise PowerError(f"the simulation is incomplete ({summary.get('missing')} replicate(s) missing): no power table may be reported "
                         "from it. Rerun the simulation cell with RESUME_MODE = 'resume'")


# ============================================================================== summaries
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def scenario_table(reps: pl.DataFrame, plan: pl.DataFrame) -> pl.DataFrame:
    want = {_scenario_key(s): int(s["replicates"]) for s in plan.iter_rows(named=True)}
    rows = []
    key = ["study", "candidate", "mode", "effect", "n_athletes", "scope"]
    for k, g in sorted(reps.partition_by(key, as_dict=True).items()):
        d = g["delta_rmse"].to_numpy()
        n = d.size
        det = int(g["detected"].sum())
        lo, hi = wilson(det, n)
        centre = float(d.mean())
        cover = float(((g["ci_lower"].to_numpy() <= centre) & (g["ci_upper"].to_numpy() >= centre)).mean())
        rows.append({**dict(zip(key, k)), "n_scope": float(g["n_scope"].mean()), "n_replicates": n, "power": det / n, "power_lower": lo,
                     "power_upper": hi, "rate_worse": float(g["worse"].mean()), "mean_delta_rmse": centre,
                     "sd_delta_rmse": float(d.std(ddof=1)) if n > 1 else None, "mean_relative": g["relative"].mean(),
                     "mean_delta_r2": g["delta_r2"].mean(), "ci_coverage": cover,
                     "mean_ci_width": float((g["ci_upper"] - g["ci_lower"]).mean()),
                     "complete": n >= want.get((k[0], k[1], k[2], round(float(k[3]), 6), int(k[4])), 10 ** 9)})
    return pl.DataFrame(rows, schema=SCENARIOS.columns) if rows else SCENARIOS.empty()


def null_table(reps: pl.DataFrame) -> pl.DataFrame:
    rows = []
    key = ["study", "candidate", "mode", "n_athletes", "scope"]
    for k, g in sorted(reps.filter(pl.col("effect") == 0.0).partition_by(key, as_dict=True).items()):
        d = g["delta_rmse"].to_numpy()
        det = int(g["detected"].sum())
        lo, hi = wilson(det, d.size)
        rows.append({**dict(zip(key, k)), "n_replicates": int(d.size), "false_positive_rate": det / d.size, "fpr_lower": lo, "fpr_upper": hi,
                     "rate_worse": float(g["worse"].mean()), "mean_delta_rmse": float(d.mean()),
                     "sd_delta_rmse": float(d.std(ddof=1)) if d.size > 1 else None, "q95_delta_rmse": float(np.quantile(d, 0.95)),
                     "ci_coverage_of_zero": float(((g["ci_lower"] <= 0) & (g["ci_upper"] >= 0)).mean()), "nominal_one_sided": 0.025})
    return pl.DataFrame(rows, schema=NULL.columns) if rows else NULL.empty()


def _crossing(x: np.ndarray, p: np.ndarray, target: float) -> float | None:
    """Smallest x at which the piecewise-linear curve through (x, p) reaches ``target``."""
    for i in range(1, x.size):
        if p[i - 1] < target <= p[i]:
            return float(x[i - 1] + (target - p[i - 1]) * (x[i] - x[i - 1]) / (p[i] - p[i - 1]))
    return float(x[0]) if x.size and p[0] >= target else None


def mde_table(scen: pl.DataFrame, pc: PowerConfig) -> pl.DataFrame:
    rows = []
    key = ["study", "candidate", "mode", "n_athletes", "scope"]
    for k, g in sorted(scen.partition_by(key, as_dict=True).items()):
        g = g.sort("effect")
        x, p = g["effect"].to_numpy(), g["power"].to_numpy()
        f = _crossing(x, p, pc.target_power)
        interp = lambda col: None if f is None else float(np.interp(f, x, np.array([np.nan if v is None else v for v in g[col].to_list()], dtype=float)))
        rows.append({**dict(zip(key, k)), "target_power": pc.target_power, "mde_effect": f, "mde_delta_rmse": interp("mean_delta_rmse"),
                     "mde_relative": interp("mean_relative"), "mde_delta_r2": interp("mean_delta_r2"), "max_power_reached": float(p.max()),
                     "method": "linear interpolation of the simulated power curve over the effect grid",
                     "note": "" if f is not None else f"power stays below {pc.target_power:.0%} up to f = {x.max():g}: the effect grid does not reach it"})
    out = pl.DataFrame(rows, schema=MDE.columns) if rows else MDE.empty()
    return out.with_columns([pl.col(c).fill_nan(None) for c in ("mde_delta_rmse", "mde_relative", "mde_delta_r2")])


# ============================================================================== attenuation, redundancy, limits
def attenuation(r_true: float, rel_x: float, rel_y: float) -> float:
    """Classical errors-in-variables approximation of an observed correlation: r_true · √(rel_x · rel_y).
    It assumes independent, non-differential error. It is NOT a bound on predictive performance: a predictor is
    scored against the OBSERVED outcome, a model can pool several noisy measurements, and context variance that is
    reliable (stable across games) is not removed by a reliability coefficient."""
    return float(r_true * np.sqrt(max(rel_x, 0.0) * max(rel_y, 0.0)))


def attenuation_grid(r_true: Sequence[float], rel_x: Sequence[float], rel_y: Sequence[float]) -> pl.DataFrame:
    return pl.DataFrame([{"r_true": float(r), "reliability_x": float(a), "reliability_y": float(b), "r_observed": attenuation(r, a, b),
                          "share_of_r2_kept": float(a * b)} for r in r_true for a in rel_x for b in rel_y])


def redundancy_table(d: tm.Design, blocks: Any, tc: TransferConfig, families: Sequence[str], fold: int = 0) -> pl.DataFrame:
    """How much of each representation the official design already explains (in-sample R², every dimension and the
    leading direction). A repeatable representation with R² near 1 has little left to add: its incremental signal is
    at most the unique share (1 − R²) of whatever it relates to."""
    ladder = {m.model: m for m in tm.model_ladder(tc)}
    X = _official_matrix(d, ladder["M1"])
    P = np.column_stack([np.ones(d.n)] + [(d.position == p).astype(float) for p in np.unique(d.position)[1:]])
    rows = []
    for fam in families:
        Z = blocks.matrix(fam, fold, d.ids)
        cols = [("direction_1", planted_direction(Z, X)[0])] + [(f"dim_{j + 1}", Z[:, j]) for j in range(Z.shape[1])]
        for name, v in cols:
            v = (v - v.mean()) / (v.std() if v.std() > 1e-12 else 1.0)
            r2 = lambda M: 1.0 - float((v - M @ np.linalg.lstsq(M, v, rcond=None)[0]).var() / v.var())
            rows.append({"family": fam, "component": name, "r2_position": r2(P), "r2_official": r2(X), "unique_share": 1.0 - r2(X),
                         "n_athletes": d.n, "n_official_terms": int(X.shape[1] - 1)})
    return pl.DataFrame(rows)


def limits_table(scen: pl.DataFrame, null: pl.DataFrame, mde: pl.DataFrame, st: Structure, observed: dict[str, dict[str, float | None]],
                 reliabilities: dict[str, float | None], pc: PowerConfig) -> pl.DataFrame:
    """One table of what bounds detectability. ``observed``: candidate → {estimate, ci_lower, ci_upper} of the real
    Milestone 06 Δ RMSE. ``reliabilities``: name → reliability estimate read from persisted tables."""
    rows: list[dict[str, Any]] = []

    def add(topic: str, item: str, value: Any, lower: Any = None, upper: Any = None, unit: str = "", note: str = "") -> None:
        rows.append({"topic": topic, "item": item, "value": None if value is None else float(value), "lower": None if lower is None else float(lower),
                     "upper": None if upper is None else float(upper), "unit": unit, "note": note})

    add("cohort", "athletes", st.design.n, unit="athletes", note="paired cohort of Milestone 06")
    add("cohort", "rmse_official_model", st.rmse_m1, unit="target units", note="real M1 out-of-fold RMSE: the residual a candidate can still explain")
    for c in pc.candidates:
        add("redundancy", f"{c}: R² of the planted direction on the official design", st.redundancy.get(c), unit="share",
            note="share of the representation's leading direction already explained by official testing")
        n0 = null.filter((pl.col("study") == "power") & (pl.col("candidate") == c) & (pl.col("mode") == "conditional") & (pl.col("scope") == ALL))
        if n0.height:
            add("null", f"{c}: false-positive rate", n0["false_positive_rate"][0], n0["fpr_lower"][0], n0["fpr_upper"][0], "rate", "nominal 0.025 (one-sided)")
            add("null", f"{c}: SD of Δ RMSE with no signal", n0["sd_delta_rmse"][0], unit="target units", note="sampling noise of the paired gain")
            add("null", f"{c}: mean Δ RMSE with no signal", n0["mean_delta_rmse"][0], unit="target units", note="cost of fitting useless predictors")
        for mode in pc.modes:
            m = mde.filter((pl.col("study") == "power") & (pl.col("candidate") == c) & (pl.col("mode") == mode) & (pl.col("scope") == ALL))
            if m.height:
                add("mde", f"{c} ({mode}): effect f at {pc.target_power:.0%} power", m["mde_effect"][0], unit="share of residual variance", note=m["note"][0])
                add("mde", f"{c} ({mode}): Δ RMSE at {pc.target_power:.0%} power", m["mde_delta_rmse"][0], unit="target units")
                add("mde", f"{c} ({mode}): relative Δ RMSE at {pc.target_power:.0%} power", m["mde_relative"][0], unit="share of M1 RMSE")
        ob = observed.get(c)
        if ob and ob.get("estimate") is not None:
            add("observed_m06", f"{c}: Δ RMSE", ob["estimate"], ob.get("ci_lower"), ob.get("ci_upper"), "target units", "real Milestone 06 comparison (unchanged)")
            if n0.height and n0["sd_delta_rmse"][0]:
                add("observed_m06", f"{c}: Δ RMSE in null SDs", (ob["estimate"] - n0["mean_delta_rmse"][0]) / n0["sd_delta_rmse"][0], unit="SD",
                    note="distance of the observed gain from the simulated no-signal mean")
    rel = {k: v for k, v in reliabilities.items() if v is not None}
    for k, v in rel.items():
        add("reliability", k, v, unit="reliability", note="read from persisted Milestone 04–07A tables")
    for kx in [k for k in rel if k.startswith("combine")]:
        for ky in [k for k in rel if k.startswith("nfl")]:
            add("attenuation", f"{kx} × {ky}: √(rel_x · rel_y)", np.sqrt(rel[kx] * rel[ky]), unit="factor on a correlation",
                note="classical approximation, NOT a ceiling on prediction (see transfer_power.attenuation)")
    return pl.DataFrame(rows, schema=LIMITS.columns)


# ============================================================================== gate and report
def gate_07c(summary: dict[str, Any], scen: pl.DataFrame, null: pl.DataFrame, mde: pl.DataFrame, pc: PowerConfig, assumptions: str) -> pl.DataFrame:
    """Gate 07C — detectability. PASS = the simulation is complete, planted effects are recovered as planted, the null
    is checked and minimum detectable effects are quantified. It says nothing about the real effect."""
    rows = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"criterion": name, "passed": bool(ok), "detail": detail})

    add("assumptions_documented", bool(assumptions.strip()), "synthetic-target model, direction and noise are stated in the report")
    add("simulation_complete", bool(summary.get("complete")), f"{summary.get('planned_replicates')} planned, {summary.get('missing')} missing")
    main = scen.filter((pl.col("study") == "power") & (pl.col("scope") == ALL) & (pl.col("mode") == "conditional")).sort("candidate", "effect")
    mono = True
    for (_,), g in main.partition_by(["candidate"], as_dict=True).items():
        d = g.sort("effect")["mean_delta_rmse"].to_numpy()
        mono &= bool(d.size >= 2 and d[-1] > d[0] and d[-1] > 0)
    add("planted_effects_recovered", main.height > 0 and mono, "mean recovered Δ RMSE rises from the null to the largest planted effect and is positive there")
    n0 = null.filter((pl.col("study") == "power") & (pl.col("scope") == ALL))
    add("null_false_positive_rate_checked", n0.height > 0, "; ".join(
        f"{r['candidate']} {r['mode']}: {r['false_positive_rate']:.3f} [{r['fpr_lower']:.3f}, {r['fpr_upper']:.3f}]" for r in n0.iter_rows(named=True)))
    add("interval_coverage_evaluated", main.height > 0 and main["ci_coverage"].null_count() == 0,
        f"coverage of the scenario mean between {main['ci_coverage'].min() if main.height else None} and {main['ci_coverage'].max() if main.height else None}")
    add("minimum_detectable_effects_quantified", mde.filter(pl.col("scope") == ALL).height > 0,
        f"{mde.filter(pl.col('mde_effect').is_not_null()).height} scenario(s) reach {pc.target_power:.0%} power inside the effect grid")
    return pl.DataFrame(rows, schema=GATE.columns)


ASSUMPTIONS = (
    "Synthetic target y* = μ + σ_res,pos · (√(1 − f) · ε + √f · z). μ = the athlete's real Milestone 06 out-of-fold prediction of the "
    "official model; σ_res,pos = SD of the real residuals of the position; ε = real standardised residuals resampled within position; "
    "z = unit-variance leading direction of the candidate representation (fold-0 encoder), either orthogonal to the official design "
    "('conditional': f is the partial R² given official testing) or as is ('raw'). Folds, athletes, covariates, missingness and "
    "fold-fitted representations are the real ones; the pipeline, bootstrap and detection criterion are those of Milestone 06. "
    "The signal is linear and one-dimensional — favourable to a ridge read-out: real effects that are non-linear, spread over many "
    "weak directions or specific to a position are harder to detect than the planted ones, so the power shown is an upper reading.")


def report_markdown(scen: pl.DataFrame, null: pl.DataFrame, mde: pl.DataFrame, limits: pl.DataFrame, summary: dict[str, Any], pc: PowerConfig) -> str:
    """``transfer_power_report.md`` — written from the executed tables only."""
    f = lambda x, d=4: "n/a" if x is None else f"{x:.{d}f}"
    lines = ["# Detectability of incremental Combine → NFL prediction (Milestone 07C)", "",
             f"Simulation `{summary.get('config_hash')}` — {'COMPLETE' if summary.get('complete') else 'INCOMPLETE: DO NOT REPORT'} "
             f"({summary.get('planned_replicates')} replicates).", "", "## Assumptions", "", ASSUMPTIONS, "", "## Power on the real cohort", "",
             "| Candidate | Mode | f | Power [MC 95 %] | Mean Δ RMSE | Relative | Interval coverage |", "|---|---|---|---|---|---|---|"]
    for r in scen.filter((pl.col("study") == "power") & (pl.col("scope") == ALL)).sort("candidate", "mode", "effect").iter_rows(named=True):
        lines.append(f"| {r['candidate']} | {r['mode']} | {r['effect']:g} | {f(r['power'], 3)} [{f(r['power_lower'], 3)}, {f(r['power_upper'], 3)}] | "
                     f"{f(r['mean_delta_rmse'])} | {f(None if r['mean_relative'] is None else 100 * r['mean_relative'], 2)} % | {f(r['ci_coverage'], 3)} |")
    lines += ["", "## No-signal calibration", ""]
    lines += [f"- {r['study']} · {r['candidate']} · {r['mode']} · n = {r['n_athletes']} · {r['scope']}: false-positive rate {f(r['false_positive_rate'], 3)} "
              f"[{f(r['fpr_lower'], 3)}, {f(r['fpr_upper'], 3)}] (nominal 0.025); mean Δ RMSE {f(r['mean_delta_rmse'], 5)}, SD {f(r['sd_delta_rmse'], 5)}"
              for r in null.filter(pl.col("scope") == ALL).iter_rows(named=True)] or ["Not available."]
    lines += ["", "## Minimum detectable effects", ""]
    lines += [f"- {r['study']} · {r['candidate']} · {r['mode']} · n = {r['n_athletes']} · {r['scope']}: f = {f(r['mde_effect'], 3)}, Δ RMSE {f(r['mde_delta_rmse'])} "
              f"({f(None if r['mde_relative'] is None else 100 * r['mde_relative'], 2)} % of the official error)" + (f" — {r['note']}" if r["note"] else "")
              for r in mde.sort("study", "candidate", "mode", "n_athletes", "scope").iter_rows(named=True)] or ["Not available."]
    lines += ["", "## Information limits", ""]
    lines += [f"- [{r['topic']}] {r['item']}: {f(r['value'])}" + (f" [{f(r['lower'])}, {f(r['upper'])}]" if r["lower"] is not None else "")
              + f" {r['unit']}" + (f" — {r['note']}" if r["note"] else "") for r in limits.iter_rows(named=True)]
    lines += ["", "Reading rules: (1) an observed gain smaller than the minimum detectable gain is INCONCLUSIVE, not a demonstrated absence; "
              "(2) the attenuation factor is a classical approximation for a correlation and is not a ceiling on prediction; "
              "(3) sample-size rows resample the observed athletes — scenario analysis, not a claim about data that will exist; "
              "(4) position-level rows have few athletes and wide Monte-Carlo and sampling error."]
    return "\n".join(lines)
