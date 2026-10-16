"""Milestone 08B.2 / 08B.3 — football-outcome transfer benchmark, resumable permutation inference, robustness and
detectability (NumPy / Polars; the Milestone 06 pipeline of ``transfer_models`` and the 07C simulation, unchanged).

    FootballTransferConfig   small-cohort settings, declared before any NFL outcome is related to a Combine value
    FootballLabels           athlete labels of one definition; a context-conditioned label is refitted on the TRAINING athletes
    football_design          the Milestone 06 paired cohort restricted to the athletes supporting the outcome (folds kept)
    run_benchmark            out-of-fold predictions of the model ladder (``tm.run_oof``)
    evaluate / hypothesis_table / calibration_table / subgroup_scores
    run_permutation          refit-based permutation null with per-iteration seeds, progress, time estimate and checkpoints
    matched_concordance      matched stopwatch prospects (the frozen pre-draft pairs of 06C)
    run_sensitivities        declared robustness analyses — a sensitivity never replaces the primary analysis
    power_config / power_structure      planted-signal simulation on the REAL football cohort (``transfer_power``)
    gate_c                   Gate C — is the transfer experiment valid? (a positive result is NOT a criterion)

Design. Unit = athlete. Folds, inner folds and fold-fitted representations are those of Milestone 06: a representation
of fold f was fitted without the athletes of fold f, which stays true for any subset of the cohort, so no encoder is
trained here and no NFL outcome ever touched a Combine representation. Preprocessing, penalties and context
parameters come from the training athletes of each fold. Sign: Δ RMSE = RMSE(reference) − RMSE(model); positive = the
model is better.

Label kinds. ``observed``: the athlete mean of the recorded outcome — a held-out athlete's label is his own observed
outcome and uses nothing from other athletes. ``cross_fitted``: value − expected value of the context cell, the
expectations estimated on the training athletes of the fold; a held-out athlete's label therefore depends on the
training athletes' OUTCOMES through the cell means (never the reverse). Both are stated in every table.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ...data.utils import get_logger
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from . import football_reliability as frl
from . import opportunity_validation as ov
from . import physical_transfer as pt
from . import transfer_models as tm
from . import transfer_power as tp
from .config import NotebookControls
from .context_outcomes import Progress
from .transfer import TransferConfig

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
ANALYSIS, COHORT = "primary", "primary"      # the labels ``transfer_power.build_structure`` reads
RESUME_MODES = ("resume", "restart", "require_complete")
SANITY_PAIRS = (("POS", "MEAN"), ("M0", "POS"), ("M1", "M0"))


@dataclass(frozen=True)
class FootballTransferConfig:
    """Small-cohort settings (about 90 linked athletes). Provenance: proposed = fixed here before any football outcome
    is related to a Combine value · heuristic = reference line · execution = cost knob."""

    version: str = "1"
    alpha_grid: tuple[float, float, int] = (0.0, 4.0, 5)       # proposed: 5 penalties (1 … 10⁴) instead of 13 — small cohort
    n_boot: int = 2000                                         # execution
    n_perm_primary: int = 500                                  # execution: ONE primary comparison
    n_perm_secondary: int = 200                                # execution: declared secondary comparisons only
    min_athletes: int = 40                                     # proposed: fewer → not evaluated
    min_athletes_group: int = 20                               # proposed: interval of a subgroup
    min_fold_test: int = 8                                     # proposed: held-out athletes in every fold
    min_relative_gain: float = 0.02                            # heuristic: practical margin (Milestone 06)
    perm_alpha: float = 0.05
    models: tuple[str, ...] = ("MEAN", "POS", "M0", "M1", "M2", "M3", "M4", "M5", "M6")
    calibration_bins: int = 4                                  # execution: about 20 athletes per bin
    progress_every: int = 10

    def for_mode(self, controls: NotebookControls) -> "FootballTransferConfig":
        if not controls.smoke:
            return self
        return replace(self, n_boot=200, n_perm_primary=20, n_perm_secondary=10, min_athletes=8, min_athletes_group=4, min_fold_test=2,
                       calibration_bins=2, progress_every=5)

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def transfer_config(self, tc: TransferConfig) -> TransferConfig:
        return replace(tc, alpha_grid=self.alpha_grid, n_boot=self.n_boot, n_perm=self.n_perm_primary, min_athletes=self.min_athletes,
                       min_athletes_group=self.min_athletes_group, min_relative_gain=self.min_relative_gain, perm_alpha=self.perm_alpha)


def ladder(tc: TransferConfig, ftc: FootballTransferConfig) -> list[tm.ModelSpec]:
    return [m for m in tm.model_ladder(tc) if m.model in ftc.models]


def registered_pairs(hyp: dict[str, Any]) -> list[tuple[str, str, str]]:
    """[(role, model, reference)] of the registry: one primary, the declared secondaries, the controls."""
    out = [("primary", *hyp["primary"])] + [("secondary", *p) for p in hyp.get("secondary", [])] + [("control", *p) for p in hyp.get("controls", [])]
    return [(r, str(m), str(ref)) for r, m, ref in out]


def all_pairs(hyp: dict[str, Any]) -> list[tuple[str, str]]:
    return list(dict.fromkeys([*SANITY_PAIRS, *[(m, r) for _, m, r in registered_pairs(hyp)]]))


# ============================================================================== contracts
OOF = replace(tm.OOF, name="football_transfer_oof_predictions",
              description="Out-of-fold predictions of the football-outcome transfer benchmark: one row per analysis × target × model × athlete.")
PERF = replace(tm.PERF, name="football_transfer_model_performance", description="Held-out performance of every model on a football outcome.")
PAIRED = replace(tm.PAIRED, name="football_transfer_paired_comparisons",
                 description="Paired model differences on identical held-out athletes (positive = the model is better).")
FOLD_PERF = replace(tm.FOLD_PERF, name="football_transfer_fold_performance", description="Held-out error by outer fold.")
PREP = replace(tm.PREP, name="football_transfer_fold_preprocessing", description="Fold-specific preprocessing and coefficients.")
OOF_SENS = replace(tm.OOF, name="football_transfer_sensitivity_oof", description="Out-of-fold predictions of the declared sensitivity analyses.")
PERF_SENS = replace(tm.PERF, name="football_transfer_sensitivity_performance", description="Performance in the declared sensitivity analyses.")
PAIRED_SENS = replace(tm.PAIRED, name="football_transfer_sensitivity_comparisons", description="Paired differences in the declared sensitivity analyses.")
SUBGROUP = replace(tm.PAIRED, name="football_transfer_subgroup_comparisons",
                   description="Pooled fit, subgroup scoring: paired differences by position and draft cohort (exploratory).")
SUBGROUP_PERF = replace(tm.PERF, name="football_transfer_subgroup_performance", description="Pooled fit, subgroup scoring: performance.")
COHORT_T = TableContract(
    "football_transfer_cohort", "Athletes of the football transfer benchmark: label, exposure and fold.",
    {"target_id": S, "nfl_id": I, "position_group": S, "draft_year": I, "fold": I, "inner_fold": I, "y_descriptive": F, "n_obs": I,
     "n_games": I, "label_kind": S}, key=("target_id", "nfl_id"), kind="dataset", area="target × athlete",
    docs={"y_descriptive": "label with context parameters of ALL athletes (never used for fitting or scoring)",
          "label_kind": "observed · cross_fitted"})
FOLD_SUPPORT = TableContract(
    "football_transfer_fold_support", "Held-out and training athletes of every outer fold.",
    {"target_id": S, "fold": I, "n_test": I, "n_train": I, "positions_test": S, "min_position_train": I, "viable": B},
    key=("target_id", "fold"), area="target × fold")
HYP = TableContract(
    "football_transfer_hypotheses", "Registered comparisons of the football transfer benchmark with every piece of uncertainty.",
    {"target_id": S, "role": S, "model": S, "reference": S, "n_athletes": I, "delta_rmse": F, "ci_lower": F, "ci_upper": F, "relative": F,
     "p_boot": F, "p_perm_linkage": F, "p_perm_outcome": F, "p_holm": F, "established": B, "worse": B, "reading": S, "evidence_label": S},
    key=("target_id", "model", "reference"), area="target × model pair",
    docs={"p_perm_linkage": "representation rows permuted between athletes within position, full refit — the incremental-value null",
          "p_perm_outcome": "labels permuted between athletes within position, full refit — a pipeline-leak check",
          "p_holm": "Holm over the primary and secondary comparisons of the target (linkage p when computed, else bootstrap p)"})
PERM_DRAWS = TableContract(
    "football_transfer_permutation_draws", "One row per permutation: null Δ RMSE of a refitted model pair (iteration −1 = observed).",
    {"signature": S, "target_id": S, "model": S, "reference": S, "mode": S, "iteration": I, "delta_rmse": F, "seconds": F},
    key=("signature", "iteration"), kind="dataset", area="comparison × permutation")
PERM = TableContract(
    "football_transfer_permutation", "Permutation inference of the registered comparisons.",
    {"target_id": S, "model": S, "reference": S, "mode": S, "observed": F, "null_mean": F, "null_sd": F, "null_q95": F, "p_value": F,
     "n_perm": I, "n_planned": I, "complete": B, "signature": S}, key=("target_id", "model", "reference", "mode"), area="comparison × mode")
CALIB = TableContract(
    "football_transfer_calibration", "Mean observed outcome by bin of the out-of-fold prediction.",
    {"target_id": S, "model": S, "bin": I, "n_athletes": I, "mean_pred": F, "mean_observed": F, "se_observed": F},
    key=("target_id", "model", "bin"), area="target × model × bin")
MATCHED = replace(pt.MATCHED, name="football_transfer_matched_pairs",
                  description="Matched stopwatch prospects (frozen pre-draft pairs of 06C): do models order their football-outcome difference?")
SENS = TableContract(
    "football_transfer_sensitivity_plan", "Declared sensitivity analyses and whether each could be evaluated.",
    {"analysis": S, "kind": S, "target_id": S, "label_kind": S, "n_athletes": I, "position_counts": S, "evaluated": B, "note": S},
    key=("analysis",), area="analysis")
ROBUST = TableContract(
    "football_transfer_robustness_reading", "Six separate readings of the primary comparison (never collapsed into one verdict).",
    {"aspect": S, "finding": S, "evidence": S}, key=("aspect",), area="aspect")
GATE_C = TableContract("football_gate_c_transfer", "Gate C — validity of the transfer experiment.", {"criterion": S, "passed": B, "detail": S},
                       key=("criterion",), area="criterion")


# ============================================================================== labels
class FootballLabels:
    """Athlete labels of one definition from its player-play rows (``football_reliability.outcome_rows``).

    ``adjustment = 'none'``: the athlete mean of the recorded values (label kind ``observed``).
    ``adjustment = 'cell_residual'``: mean of value − expected value of the ``cells`` context, expectations from the
    rows of the TRAINING athletes given to ``values`` (label kind ``cross_fitted``); count-based fallback to the
    coarser cell, then to the training mean, so every athlete keeps every row."""

    def __init__(self, rows: pl.DataFrame, target_id: str, cfg: ov.ValidationConfig, adjustment: str = "none", cells: Sequence[str] = ()):
        if adjustment not in ("none", "cell_residual"):
            raise ValueError(f"adjustment must be 'none' or 'cell_residual', got {adjustment!r}")
        if adjustment == "cell_residual" and not cells:
            raise ValueError("a cell_residual label needs its context cells")
        self.rows, self.target_id, self.cfg, self.adjustment, self.cells = rows.sort("nfl_id"), target_id, cfg, adjustment, tuple(cells)
        g = (self.rows.group_by("nfl_id").agg(pl.len().alias("n"), pl.col("game_id").n_unique().alias("g"), pl.col("value").mean().alias("v"))
             .sort("nfl_id"))
        self.ids = g["nfl_id"].to_numpy().astype(np.int64)
        self.n_obs, self.n_games, self.raw = g["n"].to_numpy().astype(np.int64), g["g"].to_numpy().astype(np.int64), g["v"].to_numpy().astype(float)
        self._cache: dict[int, np.ndarray] = {}

    @property
    def kind(self) -> str:
        return "observed" if self.adjustment == "none" else "cross_fitted"

    def supported(self) -> np.ndarray:
        """Athletes meeting the frozen support rule (a COUNT rule: observations and games)."""
        return self.ids[(self.n_obs >= self.cfg.min_obs) & (self.n_games >= self.cfg.min_games)]

    def values(self, train_ids: np.ndarray | None = None) -> np.ndarray:
        """Label of every athlete of ``ids``; context parameters from ``train_ids`` (None = all athletes: descriptive only)."""
        if self.adjustment == "none":
            return self.raw
        key = hash(None if train_ids is None else tuple(np.sort(np.asarray(train_ids)).tolist()))
        if key not in self._cache:
            fit = self.rows if train_ids is None else self.rows.filter(pl.col("nfl_id").is_in(np.asarray(train_ids).tolist()))
            adj = frl.adjust(self.rows, fit, self.cells, self.cfg)
            self._cache[key] = adj.group_by("nfl_id").agg(pl.col("value").mean()).sort("nfl_id")["value"].to_numpy().astype(float)
        return self._cache[key]

    def label_fn(self, ids: np.ndarray) -> Callable[[np.ndarray | None], np.ndarray]:
        pos = np.searchsorted(self.ids, ids)
        if (pos >= self.ids.size).any() or not np.array_equal(self.ids[np.minimum(pos, self.ids.size - 1)], ids):
            raise tm.CohortError(f"{self.target_id}: an athlete of the evaluation cohort has no outcome row")
        return lambda train_ids: self.values(train_ids)[pos]


def football_design(design: tm.Design, labels: FootballLabels) -> tm.Design:
    """The Milestone 06 paired cohort restricted to the athletes supporting the outcome. Folds are KEPT (the fold-fitted
    representations are inductive for those folds only); nothing is re-dealt and no athlete is chosen by his outcome value."""
    return design.keep_ids(np.intersect1d(design.ids, labels.supported()).tolist())


def fold_support(target_id: str, dd: tm.Design, ftc: FootballTransferConfig) -> pl.DataFrame:
    rows = []
    for f in np.unique(dd.fold):
        te = dd.fold == f
        tr_pos = [int(((dd.position == p) & ~te).sum()) for p in np.unique(dd.position)]
        rows.append({"target_id": target_id, "fold": int(f), "n_test": int(te.sum()), "n_train": int((~te).sum()),
                     "positions_test": " · ".join(f"{p} {int(((dd.position == p) & te).sum())}" for p in np.unique(dd.position)),
                     "min_position_train": min(tr_pos) if tr_pos else 0, "viable": bool(te.sum() >= ftc.min_fold_test and min(tr_pos, default=0) >= 3)})
    return pl.DataFrame(rows, schema=FOLD_SUPPORT.columns)


def cohort_table(dd: tm.Design, labels: FootballLabels) -> pl.DataFrame:
    pos = np.searchsorted(labels.ids, dd.ids)
    return pl.DataFrame({"target_id": [labels.target_id] * dd.n, "nfl_id": dd.ids, "position_group": dd.position.tolist(),
                         "draft_year": [int(c) if str(c).isdigit() else None for c in dd.cohort], "fold": dd.fold, "inner_fold": dd.inner,
                         "y_descriptive": labels.values(None)[pos], "n_obs": labels.n_obs[pos], "n_games": labels.n_games[pos],
                         "label_kind": [labels.kind] * dd.n}, schema=COHORT_T.columns)


def viability(dd: tm.Design, ftc: FootballTransferConfig) -> list[str]:
    """Reasons the cohort cannot carry the benchmark (empty = viable). Count rules only."""
    folds, counts = np.unique(dd.fold, return_counts=True)
    out = []
    if dd.n < ftc.min_athletes:
        out.append(f"{dd.n} athletes (minimum {ftc.min_athletes})")
    if folds.size < 2:
        out.append(f"{folds.size} outer fold(s)")
    elif counts.min() < ftc.min_fold_test:
        out.append(f"smallest held-out fold has {int(counts.min())} athletes (minimum {ftc.min_fold_test})")
    return out


# ============================================================================== benchmark and evaluation
def run_benchmark(analysis: str, dd: tm.Design, labels: FootballLabels, blocks: Any, models: Sequence[tm.ModelSpec], tc: TransferConfig,
                  learner: str = "ridge") -> dict[str, pl.DataFrame]:
    """{OOF, PREP} of one analysis: one out-of-fold prediction per athlete and model, every fit on the training athletes."""
    res, rows = tm.run_oof(dd, labels.label_fn(dd.ids), blocks, list(models), tc, learner)
    return {OOF.name: tm.oof_frame(analysis, COHORT, labels.target_id, dd, res), PREP.name: tm.prep_frame(analysis, COHORT, labels.target_id, rows)}


def evaluate(oof: pl.DataFrame, pairs: Sequence[tuple[str, str]], tc: TransferConfig, seed: int) -> dict[str, pl.DataFrame]:
    ev = tm.evaluate(oof.select(list(tm.OOF.columns)), pairs, tc, seed)
    return {PERF.name: ev[tm.PERF.name], PAIRED.name: ev[tm.PAIRED.name], FOLD_PERF.name: tm.fold_performance(oof)}


def subgroup_scores(oof: pl.DataFrame, groups: pl.DataFrame, kind: str, pairs: Sequence[tuple[str, str]], tc: TransferConfig, seed: int
                    ) -> dict[str, pl.DataFrame]:
    """Pooled fit, subgroup SCORING (``groups``: nfl_id, group) under analysis ``<kind>:<group>``. Exploratory by
    construction: about 24 linked TE do not support an independent TE model, and the best subgroup is not a finding."""
    o = oof.join(groups.select("nfl_id", pl.col("group").cast(pl.String)), on="nfl_id", how="inner")
    parts = [g.drop("group").with_columns(pl.lit(f"{kind}:{name}").alias("analysis"))
             for (name,), g in o.partition_by(["group"], as_dict=True).items() if name is not None]
    if not parts:
        return {SUBGROUP_PERF.name: SUBGROUP_PERF.empty(), SUBGROUP.name: SUBGROUP.empty()}
    ev = tm.evaluate(pl.concat(parts).select(list(tm.OOF.columns)), pairs, tc, seed)
    return {SUBGROUP_PERF.name: ev[tm.PERF.name], SUBGROUP.name: ev[tm.PAIRED.name]}


def calibration_table(oof: pl.DataFrame, bins: int) -> pl.DataFrame:
    """Observed outcome by equal-count bin of the out-of-fold prediction (few athletes per bin: read with its SE)."""
    rows = []
    for (t, m), g in sorted(oof.partition_by(["target_id", "model"], as_dict=True).items()):
        p, y = g["pred"].to_numpy(), g["y"].to_numpy()
        if np.unique(p).size < bins:
            continue
        o = np.argsort(p, kind="stable")
        for b, idx in enumerate(np.array_split(o, bins)):
            rows.append({"target_id": t, "model": m, "bin": b, "n_athletes": int(idx.size), "mean_pred": float(p[idx].mean()),
                         "mean_observed": float(y[idx].mean()), "se_observed": float(y[idx].std(ddof=1) / np.sqrt(idx.size)) if idx.size > 1 else None})
    return pl.DataFrame(rows, schema=CALIB.columns)


def reading(established: bool, worse: bool, relative: float | None, p_perm: float | None, ftc: FootballTransferConfig, branch_name: str) -> str:
    """One sentence per comparison. The wording is capped by the registry branch: an exploratory branch never yields a
    confirmed predictive claim."""
    if worse:
        return "the tracking model is worse (interval below 0)"
    if not established:
        return "not established (interval includes 0)"
    practical = relative is not None and relative >= ftc.min_relative_gain
    perm_ok = p_perm is not None and p_perm < ftc.perm_alpha
    if practical and perm_ok:
        return ("held-out improvement with a practical size in this cohort — requires an independent cohort" if branch_name == "frozen"
                else "exploratory positive association (provisional outcome semantics) — NOT a confirmed predictive claim")
    return "interval above 0 but " + ("below the practical margin" if not practical else "not supported by the permutation null") + " — exploratory"


def hypothesis_table(target_id: str, paired: pl.DataFrame, perm: pl.DataFrame, hyp: dict[str, Any], ftc: FootballTransferConfig,
                     branch: dict[str, str]) -> pl.DataFrame:
    """The registered comparisons of one target with bootstrap interval, bootstrap p, permutation p and Holm adjustment
    over the primary + secondary family."""
    pr = paired.filter((pl.col("analysis") == ANALYSIS) & (pl.col("target_id") == target_id) & (pl.col("metric") == "delta_rmse"))
    by = {(r["model"], r["reference"]): r for r in pr.iter_rows(named=True)}
    pm = {(r["model"], r["reference"], r["mode"]): r["p_value"] for r in perm.filter((pl.col("target_id") == target_id) & pl.col("complete")).iter_rows(named=True)}
    rows = []
    for role, m, ref in registered_pairs(hyp):
        r = by.get((m, ref))
        if r is None:
            continue
        rows.append({"target_id": target_id, "role": role, "model": m, "reference": ref, "n_athletes": r["n_athletes"], "delta_rmse": r["estimate"],
                     "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"], "relative": r["relative"],
                     "p_boot": pt._p_boot(r["share_boot_improved"], ftc.n_boot), "p_perm_linkage": pm.get((m, ref, "linkage")),
                     "p_perm_outcome": pm.get((m, ref, "outcome")), "p_holm": None, "established": r["established"], "worse": r["worse"],
                     "reading": "", "evidence_label": branch["label"]})
    family = [(r["p_perm_linkage"] if r["p_perm_linkage"] is not None else r["p_boot"]) if r["role"] in ("primary", "secondary") else None for r in rows]
    for r, h in zip(rows, pt.holm(family)):
        r["p_holm"] = h
        r["reading"] = reading(bool(r["established"]), bool(r["worse"]), r["relative"], r["p_perm_linkage"], ftc, branch["name"])
    return pl.DataFrame(rows, schema=HYP.columns)


# ============================================================================== resumable permutation inference
def perm_signature(dd: tm.Design, target_id: str, pair: tuple[str, str], mode: str, tc: TransferConfig, seed: int, label_kind: str) -> str:
    """Identity of one permutation experiment: cohort, folds, penalties, pair, mode, seed. Draws of another signature are never reused."""
    return ov._hash({"ids": dd.ids.tolist(), "fold": dd.fold.tolist(), "inner": dd.inner.tolist(), "target": target_id, "pair": list(pair),
                     "mode": mode, "alpha_grid": list(tc.alpha_grid), "inner_folds": tc.inner_folds, "seed": int(seed), "label_kind": label_kind})


def _read_draws(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).select([pl.col(k).cast(t) for k, t in PERM_DRAWS.columns.items()]) if path.is_file() else PERM_DRAWS.empty()


def _write_draws(path: Path, rows: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    rows.write_parquet(tmp)
    tmp.replace(path)


def run_permutation(dd: tm.Design, labels: FootballLabels, blocks: Any, models: Sequence[tm.ModelSpec], pair: tuple[str, str],
                    tc: TransferConfig, seed: int, n_perm: int, mode: str, checkpoint: Path, resume_mode: str = "resume",
                    progress_every: int = 10, max_seconds: float | None = None, save: bool = True,
                    printer: Callable[[str], None] = print) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Refit-based permutation null of Δ RMSE(reference − model) for ONE model pair.

    linkage   the representation rows are permuted between athletes WITHIN position; labels and official results stay.
              The null of "the athlete's own tracking adds nothing to official testing". (It also breaks the link between
              the tracking block and the official tests of the same athlete: stated as a limit of the test.)
    outcome   the labels are permuted between athletes within position: every athlete-level link is broken. A pipeline
              that still shows a gain is leaking.
    Every permutation refits both models in every fold, penalties included — out-of-fold predictions are never shuffled.
    Iteration b uses its own seed, so the result does not depend on the order of computation and an interrupted run
    resumes exactly: rerun the cell. Iteration −1 is the observed statistic. p = (1 + #{null ≥ observed}) / (1 + n)."""
    if mode not in ("linkage", "outcome"):
        raise ValueError(f"mode must be 'linkage' or 'outcome', got {mode!r}")
    if resume_mode not in RESUME_MODES:
        raise ValueError(f"RESUME_MODE must be one of {RESUME_MODES}, got {resume_mode!r}")
    use = [m for m in models if m.model in pair]
    if len(use) != 2:
        raise ValueError(f"both models of {pair} must be in the ladder")
    if mode == "linkage" and not next(m for m in use if m.model == pair[0]).families:
        raise ValueError(f"{pair[0]} has no representation block: a linkage permutation is undefined for it")
    sig = perm_signature(dd, labels.target_id, pair, mode, tc, seed, labels.kind)
    stored = _read_draws(checkpoint)
    others = stored.filter(pl.col("signature") != sig)
    mine = PERM_DRAWS.empty() if resume_mode == "restart" else stored.filter(pl.col("signature") == sig)
    have = set(mine["iteration"].to_list())
    todo = [b for b in range(-1, int(n_perm)) if b not in have]
    name = f"08B permutation · {labels.target_id} · {pair[0]} vs {pair[1]} · {mode}"
    printer(f"[{name}] signature {sig} · {n_perm} permutation(s) planned · {len(have & set(range(n_perm)))} in the checkpoint · "
            f"{len([b for b in todo if b >= 0])} to compute · checkpoint {checkpoint.name}")
    if resume_mode == "require_complete" and todo:
        raise RuntimeError(f"RESUME_MODE = 'require_complete' but {len(todo)} iteration(s) are missing for signature {sig}")
    y_fn = labels.label_fn(dd.ids)
    bar = Progress(len(todo), name, every=progress_every, printer=printer)
    buffer: list[dict[str, Any]] = []
    t_start, stopped = time.perf_counter(), False

    def flush() -> None:
        nonlocal mine, buffer
        if buffer:
            mine = pl.concat([mine, pl.DataFrame(buffer, schema=PERM_DRAWS.columns)])
            buffer = []
            if save:
                _write_draws(checkpoint, pl.concat([others, mine]))

    for b in todo:
        t0 = time.perf_counter()
        if b < 0:
            res = tm.run_oof(dd, y_fn, blocks, use, tc)[0]
        else:
            perm = tm._within(dd.position, np.random.default_rng(seed_for(seed, "fb_perm", mode, pair[0], pair[1], labels.target_id, str(b))))
            if mode == "outcome":
                res = tm.run_oof(dd, lambda t, _p=perm: np.asarray(y_fn(t))[_p], blocks, use, tc)[0]
            else:
                res = tm.run_oof(dd, y_fn, blocks, use, tc, permute_rep=perm)[0]
        e = {m: float(np.sqrt(np.mean((res["y"] - res["pred"][m]) ** 2))) for m in pair}
        buffer.append({"signature": sig, "target_id": labels.target_id, "model": pair[0], "reference": pair[1], "mode": mode,
                       "iteration": int(b), "delta_rmse": e[pair[1]] - e[pair[0]], "seconds": time.perf_counter() - t0})
        if bar.done == 0:
            printer(bar.estimate(time.perf_counter() - t0))
        bar.step()
        if len(buffer) >= progress_every:
            flush()
        if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
            stopped = True
            break
    flush()
    left = len([b for b in range(-1, int(n_perm)) if b not in set(mine["iteration"].to_list())])
    summary = {"signature": sig, "target_id": labels.target_id, "model": pair[0], "reference": pair[1], "mode": mode, "n_planned": int(n_perm),
               "missing": left, "complete": left == 0, "stopped_by_time_limit": stopped, "seconds": time.perf_counter() - t_start,
               "checkpoint": checkpoint.name}
    printer(f"[{name}] {'COMPLETE' if left == 0 else f'INCOMPLETE — {left} iteration(s) missing; rerun the cell to resume'}")
    return mine.sort("iteration"), summary


def permutation_summary(draws: Sequence[pl.DataFrame], summaries: Sequence[dict[str, Any]]) -> pl.DataFrame:
    rows = []
    for d, s in zip(draws, summaries):
        obs = d.filter(pl.col("iteration") < 0)["delta_rmse"]
        null = d.filter((pl.col("iteration") >= 0) & (pl.col("iteration") < s["n_planned"]))["delta_rmse"].to_numpy()
        if not obs.len() or not null.size:
            continue
        o = float(obs[0])
        rows.append({"target_id": s["target_id"], "model": s["model"], "reference": s["reference"], "mode": s["mode"], "observed": o,
                     "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)) if null.size > 1 else None,
                     "null_q95": float(np.quantile(null, 0.95)), "p_value": float((1 + (null >= o).sum()) / (1 + null.size)),
                     "n_perm": int(null.size), "n_planned": int(s["n_planned"]), "complete": bool(s["complete"]), "signature": s["signature"]})
    return pl.DataFrame(rows, schema=PERM.columns)


# ============================================================================== matched stopwatch prospects
def matched_concordance(pairs: pl.DataFrame, oof: pl.DataFrame, tc: TransferConfig, seed: int, label: str) -> tuple[pl.DataFrame, dict[str, int]]:
    """Concordance of predicted and observed football-outcome differences inside the matched pairs frozen in 06C (same
    position, near-identical official forty, 10-yard split and body size — pre-draft information only). A pair is kept
    when both athletes are in the football cohort; none is chosen by its outcome. Fewer than 10 pairs → no estimate."""
    oof = oof.filter(pl.col("analysis") == ANALYSIS)
    ids = set(oof["nfl_id"].unique().to_list())
    kept = pairs.filter(pl.col("nfl_id_a").is_in(list(ids)) & pl.col("nfl_id_b").is_in(list(ids)))
    counts = {"pairs_frozen": pairs.height, "pairs_in_cohort": kept.height,
              "athletes_in_pairs": len(set(kept["nfl_id_a"].to_list()) | set(kept["nfl_id_b"].to_list()))}
    if kept.height < 10:
        return MATCHED.empty(), counts
    out = pt.pair_concordance(kept, oof.select(list(tm.OOF.columns)), tc, seed)
    return out.with_columns(pl.lit(label).alias("status")).select(list(MATCHED.columns)), counts


# ============================================================================== robustness (08B.3)
def run_sensitivities(items: Sequence[dict[str, Any]], blocks: Any, models: Sequence[tm.ModelSpec], pairs: Sequence[tuple[str, str]],
                      tc: TransferConfig, ftc: FootballTransferConfig, seed: int, printer: Callable[[str], None] = print
                      ) -> dict[str, pl.DataFrame]:
    """Declared sensitivity analyses. ``items``: {analysis, kind, labels (FootballLabels), design (tm.Design), learner, note}.
    Each is a full refit on its own cohort with the SAME folds. None of them may replace the primary analysis: they say
    whether the primary reading depends on a choice."""
    plan, oofs, perf, paired = [], [], [], []
    bar = Progress(len(items), "08B.3 sensitivities", printer=printer)
    for it in items:
        lab = it["labels"]
        dd = football_design(it["design"], lab)
        why = viability(dd, ftc)
        row = {"analysis": it["analysis"], "kind": it["kind"], "target_id": lab.target_id, "label_kind": lab.kind, "n_athletes": dd.n,
               "position_counts": " · ".join(f"{p} {int((dd.position == p).sum())}" for p in np.unique(dd.position)), "evaluated": not why,
               "note": it.get("note", "") if not why else "not evaluated: " + "; ".join(why)}
        plan.append(row)
        if not why:
            o = run_benchmark(it["analysis"], dd, lab, blocks, models, tc, it.get("learner", "ridge"))[OOF.name]
            ev = tm.evaluate(o, pairs, tc, seed)
            oofs.append(o)
            perf.append(ev[tm.PERF.name])
            paired.append(ev[tm.PAIRED.name])
        bar.step(note=f"{it['analysis']} · n = {dd.n}" + ("" if not why else " · skipped"))
    cat = lambda parts, c: pl.concat(parts) if parts else c.empty()
    return {SENS.name: pl.DataFrame(plan, schema=SENS.columns), OOF_SENS.name: cat(oofs, OOF_SENS), PERF_SENS.name: cat(perf, PERF_SENS),
            PAIRED_SENS.name: cat(paired, PAIRED_SENS)}


def power_config(controls: NotebookControls, n_cohort: int, candidates: Sequence[str] = ("M2", "M5"), replicates: int = 200, n_boot: int = 300,
                 sample_replicates: int = 100) -> tp.PowerConfig:
    """Bounded budget of the planted-signal simulation on the football cohort (07C machinery, smaller budget)."""
    pc = tp.PowerConfig().for_mode(controls)
    if controls.smoke:
        return replace(pc, candidates=tuple(candidates)[:1], sample_sizes=(int(n_cohort), 2 * int(n_cohort)))
    return replace(pc, candidates=tuple(candidates), replicates=int(replicates), n_boot=int(n_boot), sample_replicates=int(sample_replicates),
                   sample_sizes=tuple(sorted({int(n_cohort), 150, 284, 500, 1000})), sample_effects=(0.05, 0.10, 0.20))


def power_structure(dd: tm.Design, oof: pl.DataFrame, blocks: Any, tc: TransferConfig, pc: tp.PowerConfig, target_id: str) -> tp.Structure:
    """Real structure of the football cohort for the simulation: athlete count, positions, folds, the collinearity of the
    representation with official testing, and the real M1 residual noise by position."""
    o = oof.filter((pl.col("analysis") == ANALYSIS) & (pl.col("target_id") == target_id)).select(list(tm.OOF.columns))
    return tp.build_structure(dd, o, blocks, tc, pc, target_id)


def robustness_reading(h: dict[str, Any] | None, sens: pl.DataFrame, mde: pl.DataFrame | None, reliability: float | None,
                       source_class: str, n: int, position_counts: str, ftc: FootballTransferConfig) -> pl.DataFrame:
    """Six readings of the primary comparison, kept apart. A positive but uncertain point estimate is exploratory."""
    f = lambda x, d=4: "n/a" if x is None else f"{x:+.{d}f}"
    rows = []
    if h is None:
        rows.append({"aspect": "point_estimate", "finding": "not evaluated", "evidence": "no primary comparison in the hypothesis table"})
    else:
        rel = h["relative"]
        rows += [{"aspect": "point_estimate", "finding": f"Δ RMSE {f(h['delta_rmse'])} ({'n/a' if rel is None else f'{100 * rel:+.2f} %'} of the reference error)",
                  "evidence": f"{h['model']} versus {h['reference']} on {h['n_athletes']} held-out athletes"},
                 {"aspect": "statistical_evidence", "finding": ("interval above 0" if h["established"] else "interval below 0" if h["worse"] else "interval includes 0"),
                  "evidence": f"95 % athlete bootstrap [{f(h['ci_lower'])}, {f(h['ci_upper'])}]; linkage permutation p "
                              f"{'not computed' if h['p_perm_linkage'] is None else format(h['p_perm_linkage'], '.3f')}"},
                 {"aspect": "practical_importance", "finding": ("at or above" if rel is not None and rel >= ftc.min_relative_gain else "below") + f" the {ftc.min_relative_gain:.0%} margin",
                  "evidence": "heuristic margin of Milestone 06; a scouting use would need a decision study, not an error ratio"}]
    s = sens.filter((pl.col("metric") == "delta_rmse") & (pl.col("model") == (h or {}).get("model")) & (pl.col("reference") == (h or {}).get("reference"))) if sens.height and h else sens.head(0)
    pos_n = int((s["estimate"] > 0).sum()) if s.height else 0
    rows.append({"aspect": "cohort_limitations", "finding": f"{n} athletes ({position_counts}); {pos_n} of {s.height} sensitivity analyses have a positive point estimate",
                 "evidence": "one linked cohort; subgroup rows are scored from the pooled fit; no independent TE model"})
    m = mde.filter((pl.col("study") == "power") & (pl.col("scope") == "ALL") & (pl.col("mode") == "conditional")) if mde is not None and mde.height else None
    mtxt = "simulation not available" if m is None or not m.height else "; ".join(
        f"{r['candidate']}: smallest detectable planted share {'not reached in the grid' if r['mde_effect'] is None else format(r['mde_effect'], '.2f')} "
        f"(max power {'n/a' if r['max_power_reached'] is None else format(r['max_power_reached'], '.2f')})" for r in m.iter_rows(named=True))
    rows += [{"aspect": "detectability", "finding": mtxt, "evidence": "planted-signal simulation on the real cohort, folds and collinearity (08B.3)"},
             {"aspect": "measurement_reliability", "finding": f"odd / even reliability of the outcome {'n/a' if reliability is None else format(reliability, '.2f')}",
              "evidence": "an unreliable outcome attenuates every association; the ceiling is shown in the attenuation table"},
             {"aspect": "outcome_validity", "finding": f"source class {source_class}",
              "evidence": "an interaction outcome with inferred semantics supports an exploratory association at most"}]
    return pl.DataFrame(rows, schema=ROBUST.columns)


# ============================================================================== Gate C
def gate_c(oof: pl.DataFrame, paired: pl.DataFrame, hyp_table: pl.DataFrame, perm: pl.DataFrame, support: pl.DataFrame, hyp: dict[str, Any],
           design_columns: Sequence[str], branch: dict[str, str], gate_b_status: dict[str, str], primary_id: str) -> pl.DataFrame:
    """Gate C — are the held-out comparisons valid and uncertainty-aware? The experiment can PASS with a null result."""
    rows = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"criterion": name, "passed": bool(ok), "detail": detail})

    o = oof.filter((pl.col("analysis") == ANALYSIS) & (pl.col("target_id") == primary_id))
    sets = {m: frozenset(g["nfl_id"].to_list()) for (m,), g in o.partition_by(["model"], as_dict=True).items()} if o.height else {}
    uniq = o.height > 0 and o.height == o.select(pl.struct("model", "nfl_id").n_unique()).item()
    one_fold = o.height > 0 and o.group_by("nfl_id").agg(pl.col("fold").n_unique().alias("k"))["k"].max() == 1
    add("registry_branch_allows_transfer", branch["name"] in ("frozen", "exploratory"), branch["label"])
    add("gate_b_passed_for_primary", gate_b_status.get(primary_id) == "PASS", f"{primary_id}: Gate B {gate_b_status.get(primary_id, 'not recorded')}")
    add("athlete_held_out_predictions", uniq and one_fold, f"one prediction per athlete and model: {uniq}; every athlete in one fold: {one_fold}")
    add("paired_common_cohort", bool(sets) and len(set(sets.values())) == 1, f"{len(sets)} models on {len(next(iter(sets.values()), ()))} identical athletes")
    add("folds_viable", support.height > 0 and bool(support.filter(pl.col("target_id") == primary_id)["viable"].all()),
        "every outer fold holds enough held-out athletes and ≥ 3 training athletes of each position")
    try:
        tm.assert_pre_nfl_columns(list(design_columns))
        clean = True
    except tm.CohortError:
        clean = False
    add("no_nfl_predictor", clean, f"{len(list(design_columns))} baseline predictors, all pre-draft; representations are the fold-fitted 06B blocks")
    add("baselines_included", {"M0", "M1"} <= set(sets), "M0 and M1 evaluated")
    p = hyp_table.filter((pl.col("target_id") == primary_id) & (pl.col("role") == "primary"))
    add("primary_comparison_is_the_registered_one", p.height == 1 and [p["model"][0], p["reference"][0]] == list(hyp["primary"]),
        f"registered {hyp['primary']}")
    add("paired_uncertainty_calculated", p.height == 1 and p["ci_lower"][0] is not None, "95 % paired athlete bootstrap on the primary comparison")
    pp = perm.filter((pl.col("target_id") == primary_id) & (pl.col("model") == hyp["primary"][0]) & (pl.col("reference") == hyp["primary"][1]))
    add("permutation_complete", pp.height > 0 and bool(pp["complete"].all()), f"modes: {pp['mode'].to_list()}")
    add("evidence_label_explicit", hyp_table.height > 0 and hyp_table["evidence_label"].n_unique() == 1, branch["claim"])
    return pl.DataFrame(rows, schema=GATE_C.columns)


def conclusions_markdown(hyp_table: pl.DataFrame, perf: pl.DataFrame, matched: pl.DataFrame, matched_counts: dict[str, int],
                         gate: pl.DataFrame, branch: dict[str, str], ftc: FootballTransferConfig) -> str:
    f = lambda x, d=4: "n/a" if x is None else f"{x:+.{d}f}"
    g = lambda x: "n/a" if x is None else f"{x:.3f}"
    ok = bool(gate["passed"].all()) if gate.height else False
    lines = ["### 08B.2 — football transfer benchmark", "", f"Evidence label: **{branch['label']}** — {branch['claim']}.",
             f"Gate C: **{'PASS' if ok else 'FAIL'}** (a valid experiment, whatever the sign of the result).", "",
             "| Target | Role | Model vs reference | n | Δ RMSE [95 %] | Relative | p linkage | p outcome | Holm | Reading |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in hyp_table.iter_rows(named=True):
        rel = r["relative"]
        lines.append(f"| `{r['target_id']}` | {r['role']} | {r['model']} vs {r['reference']} | {r['n_athletes']} | {f(r['delta_rmse'])} "
                     f"[{f(r['ci_lower'])}, {f(r['ci_upper'])}] | {'n/a' if rel is None else f'{100 * rel:+.2f} %'} | {g(r['p_perm_linkage'])} | "
                     f"{g(r['p_perm_outcome'])} | {g(r['p_holm'])} | {r['reading']} |")
    base = perf.filter((pl.col("analysis") == ANALYSIS) & pl.col("metric").is_in(["rmse", "r2_oos", "spearman"]) & pl.col("model").is_in(["POS", "M0", "M1"]))
    lines += ["", "Baselines: " + "; ".join(f"`{r['target_id']}` {r['model']} {r['metric']} {g(r['estimate'])}" for r in base.sort("target_id", "model", "metric").iter_rows(named=True))]
    lines += ["", f"Matched stopwatch prospects: {matched_counts.get('pairs_in_cohort', 0)} of {matched_counts.get('pairs_frozen', 0)} frozen 06C pairs lie inside "
                  f"the football cohort ({matched_counts.get('athletes_in_pairs', 0)} athletes)" + (": too few for an estimate." if not matched.height else ".")]
    lines += [f"- `{r['target_id']}` {r['model']}: concordance {g(r['concordance'])} [{g(r['ci_lower'])}, {g(r['ci_upper'])}], versus official "
              f"{f(r['delta_vs_reference'], 3)} [{f(r['delta_ci_lower'], 3)}, {f(r['delta_ci_upper'], 3)}]" for r in matched.sort("target_id", "model").iter_rows(named=True)]
    lines += ["", f"Limits: one linked cohort of about 90 receivers; an interaction outcome with inferred semantics; intervals cover athlete sampling, not "
                  f"model refitting; the practical margin ({ftc.min_relative_gain:.0%}) is a heuristic; the linkage permutation also breaks the "
                  "tracking–official link of the same athlete. Milestone 06 (no validated incremental prediction of NFL movement) is unchanged."]
    return "\n".join(lines)


__all__ = ["ANALYSIS", "CALIB", "COHORT", "COHORT_T", "FOLD_PERF", "FOLD_SUPPORT", "FootballLabels", "FootballTransferConfig", "GATE_C", "HYP",
           "MATCHED", "OOF", "OOF_SENS", "PAIRED", "PAIRED_SENS", "PERF", "PERF_SENS", "PERM", "PERM_DRAWS", "PREP", "ROBUST", "SENS",
           "SUBGROUP", "SUBGROUP_PERF", "all_pairs", "calibration_table", "cohort_table", "conclusions_markdown", "evaluate", "fold_support",
           "football_design", "gate_c", "hypothesis_table", "ladder", "matched_concordance", "perm_signature", "permutation_summary",
           "power_config", "power_structure", "reading", "registered_pairs", "robustness_reading", "run_benchmark", "run_permutation",
           "run_sensitivities", "subgroup_scores", "viability"]
