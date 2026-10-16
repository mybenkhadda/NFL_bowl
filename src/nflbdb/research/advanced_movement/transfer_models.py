"""Regularised transfer models, athlete-grouped out-of-fold prediction and paired evaluation (Milestone 06B). NumPy only.

Unit            one row = one athlete = one label. Folds never split an athlete; attempts and plays are not rows here.
Model ladder    MEAN / POS sanity predictors · M0 context (position, cohort, body) · M1 official Combine (THE reference) ·
                M2 + engineered movement · M3 + SPEED2 · M4 + FPCA · M5 + TCN-B · M6 + random encoder · M7 + compact TCN ·
                M5S = M1 + SPEED2 + TCN-B (does the encoder add anything once a simple speed representation is in?).
Per outer fold  (1) labels of every athlete given the TRAINING athletes (context means from them only) · (2) athlete
                representation = aggregate of his attempts under the fold's encoder · (3) imputation medians, means,
                scales, dummy levels and missingness indicators from training athletes · (4) ridge penalty by inner
                athlete-grouped folds INSIDE the training athletes, preprocessing refitted in each · (5) one prediction
                per held-out athlete. Position effects are additive and unpenalised; everything else shares one penalty.
Evaluation      every model of a comparison is scored on the SAME athletes against the SAME labels (asserted). Metrics:
                RMSE, MAE, R² against the training-fold mean, Spearman, Pearson, calibration slope and intercept.
                Δ = reference − model for errors, model − reference for R² and Spearman: POSITIVE = the model is better.
Uncertainty     paired bootstrap of ATHLETES on the out-of-fold table (models are not refitted): it reflects which
                athletes were sampled, not the variability of the fitting procedure. An interval that includes 0 means
                an incremental benefit is not established.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ...data.utils import get_logger
from ..combine_nfl_dl.models import metric_arrays
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from ..studies.stats import rankdata
from . import transfer as tr
from .transfer import FrozenTarget, TargetSet, TransferConfig

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
PRIMARY = "primary"
REFERENCE = "M1"
METRICS = ("rmse", "mae", "r2_oos", "spearman", "pearson", "calibration_slope", "calibration_intercept")
DELTAS = (("rmse", -1.0), ("mae", -1.0), ("r2_oos", 1.0), ("spearman", 1.0))       # sign making "positive = model better"
FORBIDDEN_PREDICTORS = ("nfl_", "o1.", "o2.", "o3.", "o4.", "o5.", "o6.", "value", "n_windows", "n_games", "target", "season")


class CohortError(ValueError):
    """Models of one comparison were not evaluated on identical athletes and labels, or an athlete lacks an input."""


@dataclass(frozen=True)
class ModelSpec:
    model: str
    label: str
    base: str                       # none · position · M0 · M1
    families: tuple[str, ...] = ()
    role: str = "candidate"         # sanity · baseline · reference · candidate · control · exploratory


def model_ladder(tc: TransferConfig) -> tuple[ModelSpec, ...]:
    return (ModelSpec("MEAN", "training-fold mean", "none", (), "sanity"),
            ModelSpec("POS", "position mean", "position", (), "sanity"),
            ModelSpec("M0", "M0 · context (position, cohort, body)", "M0", (), "baseline"),
            ModelSpec("M1", "M1 · official Combine", "M1", (), "reference"),
            ModelSpec("M2", "M2 · M1 + engineered movement", "M1", ("ENG",), "candidate"),
            ModelSpec("M3", "M3 · M1 + SPEED2", "M1", ("SPEED2",), "control"),
            ModelSpec("M4", "M4 · M1 + FPCA", "M1", ("FPCA",), "candidate"),
            ModelSpec("M5", "M5 · M1 + TCN-B (8-d)", "M1", ("TCN",), "candidate"),
            ModelSpec("M6", "M6 · M1 + random encoder", "M1", ("RAND",), "control"),
            *[ModelSpec(f"M7_PC{k}", f"M7 · M1 + TCN-B on {k} axes", "M1", (f"TCN_PC{k}",), "exploratory") for k in tc.compact_dims],
            ModelSpec("M5S", "M1 + SPEED2 + TCN-B", "M1", ("SPEED2", "TCN"), "exploratory"))


def comparisons(tc: TransferConfig) -> tuple[tuple[str, str], ...]:
    """(model, reference) pairs, declared before any result: everything against M1, then TCN-B against every simpler
    or control representation, and TCN-B on top of SPEED2."""
    return (("POS", "MEAN"), ("M0", "POS"), ("M1", "M0"), ("M2", "M1"), ("M3", "M1"), ("M4", "M1"), ("M5", "M1"), ("M6", "M1"),
            *[(f"M7_PC{k}", "M1") for k in tc.compact_dims], ("M5S", "M1"), ("M5", "M2"), ("M5", "M3"), ("M5", "M4"), ("M5", "M6"),
            ("M5S", "M3"), *[(f"M7_PC{k}", "M5") for k in tc.compact_dims])


# ============================================================================== contracts
_K = {"analysis": S, "cohort": S, "target_id": S}
OOF = TableContract(
    "transfer_oof_predictions", "Out-of-fold predictions: one row per analysis × target × model × athlete.",
    {**_K, "model": S, "nfl_id": I, "position_group": S, "draft_year": I, "fold": I, "y": F, "y_ref": F, "pred": F, "alpha": F,
     "n_train": I}, key=("analysis", "cohort", "target_id", "model", "nfl_id"), kind="dataset",
    area="analysis × target × model × athlete",
    docs={"analysis": "primary, or a labelled sensitivity (agg:… · exposure:… · context:… · loco · seed:… · learner:…)",
          "cohort": "primary = athletes supporting every model · official_wide = official results + target only",
          "y": "label the athlete was scored against (context means, when used, from the fold's training athletes)",
          "y_ref": "mean label of the fold's training athletes (the R² reference)", "alpha": "ridge penalty of the fold"})
OOF_TARGETS = TableContract(
    "transfer_oof_targets", "Labels of the evaluated athletes: fold-specific value, descriptive value and exposure.",
    {**_K, "nfl_id": I, "fold": I, "y": F, "y_descriptive": F, "n_windows": I, "n_games": I, "variant": S, "adjustment": S},
    key=("analysis", "cohort", "target_id", "nfl_id"), kind="dataset", area="analysis × target × athlete",
    docs={"y_descriptive": "same statistic with context means of ALL athletes (never used for fitting or scoring)"})
PERF = TableContract(
    "transfer_model_performance", "Held-out performance of every model with athlete-bootstrap intervals.",
    {**_K, "model": S, "metric": S, "estimate": F, "ci_lower": F, "ci_upper": F, "n_athletes": I, "n_positions": I, "n_cohorts": I},
    key=("analysis", "cohort", "target_id", "model", "metric"), area="analysis × target × model × metric")
PAIRED = TableContract(
    "transfer_paired_comparisons", "Paired differences between models on identical athletes (positive = the model is better).",
    {**_K, "model": S, "reference": S, "metric": S, "estimate": F, "ci_lower": F, "ci_upper": F, "relative": F,
     "share_boot_improved": F, "n_athletes": I, "established": B, "worse": B},
    key=("analysis", "cohort", "target_id", "model", "reference", "metric"), area="analysis × target × model pair × metric",
    docs={"estimate": "delta_rmse / delta_mae = reference − model · delta_r2_oos / delta_spearman = model − reference",
          "relative": "delta / metric of the reference (errors only)",
          "established": "the 95 % interval lies above 0", "worse": "the 95 % interval lies below 0"})
FOLD_PERF = TableContract(
    "transfer_fold_performance", "Held-out error of every model in every outer fold.",
    {**_K, "model": S, "fold": I, "n_athletes": I, "rmse": F, "mae": F, "delta_rmse_vs_m1": F},
    key=("analysis", "cohort", "target_id", "model", "fold"), area="analysis × target × model × fold")
PREP = TableContract(
    "transfer_fold_preprocessing", "Fold-specific preprocessing and fitted coefficients of every model term.",
    {**_K, "model": S, "fold": I, "term": S, "block": S, "penalised": B, "median": F, "mean": F, "sd": F, "coefficient": F, "alpha": F},
    key=("analysis", "cohort", "target_id", "model", "fold", "term"), kind="dataset", area="analysis × target × model × fold × term",
    docs={"median": "training-fold imputation value (numeric predictors with missing values)",
          "mean": "training-fold centre", "sd": "training-fold scale", "coefficient": "per training SD of the term, target units"})
PERM = TableContract(
    "transfer_permutation", "Permutation checks of the pipeline (not a substitute for held-out validation).",
    {**_K, "model": S, "reference": S, "mode": S, "observed": F, "null_mean": F, "null_q95": F, "p_value": F, "n_perm": I},
    key=("analysis", "cohort", "target_id", "model", "reference", "mode"), area="analysis × target × model pair × mode",
    docs={"mode": "outcome = labels permuted between athletes within position · linkage = representation rows permuted "
                  "between athletes within position (labels and official results untouched)"})
SUPPORT = TableContract(
    "transfer_common_support", "Athletes of the 06A primary cohort and whether every representation exists for them in every fold.",
    {"nfl_id": I, "position_group": S, "draft_year": I, "in_paired_cohort": B, "missing": S}, key=("nfl_id",), area="athlete")


# ============================================================================== design
def assert_pre_nfl_columns(names: Sequence[str]) -> None:
    """A predictor may not be, or be named like, an NFL quantity."""
    bad = [n for n in names if any(n.lower().startswith(p) for p in FORBIDDEN_PREDICTORS)]
    if bad:
        raise CohortError(f"predictor(s) named like an NFL outcome: {bad}. Predictors are pre-NFL Combine quantities only")


@dataclass
class Design:
    """Athlete-level frame of one evaluation cohort: identity, baseline predictors (NaN where missing) and folds."""
    ids: np.ndarray
    position: np.ndarray
    cohort: np.ndarray
    numeric: dict[str, np.ndarray]
    m0: tuple[str, ...]
    m1: tuple[str, ...]
    indicators: tuple[str, ...]
    fold: np.ndarray
    inner: np.ndarray

    @classmethod
    def build(cls, folds: pl.DataFrame, official: pl.DataFrame, columns: dict[str, Sequence[str]], primary_only: bool = True) -> "Design":
        need = list(dict.fromkeys(columns["m1_numeric"]))
        assert_pre_nfl_columns(need)
        f = (folds.filter(pl.col("in_primary_cohort")) if primary_only else folds).sort("nfl_id")
        d = f.join(official.select("nfl_id", *need), on="nfl_id", how="left")
        if d.height != f.height:
            raise CohortError("the official table holds several rows for an athlete")
        return cls(d["nfl_id"].to_numpy().astype(np.int64), d["position_group"].fill_null("?").to_numpy().astype(str),
                   d["draft_year"].cast(pl.String).fill_null("?").to_numpy().astype(str),
                   {c: d[c].cast(pl.Float64).fill_null(NAN).to_numpy() for c in need}, tuple(columns["m0_numeric"]),
                   tuple(columns["m1_numeric"]), tuple(columns["indicators"]), d["outer_fold"].to_numpy().astype(np.int64),
                   d["inner_fold"].to_numpy().astype(np.int64))

    @property
    def n(self) -> int:
        return int(self.ids.size)

    def subset(self, keep: np.ndarray) -> "Design":
        return replace(self, ids=self.ids[keep], position=self.position[keep], cohort=self.cohort[keep],
                       numeric={k: v[keep] for k, v in self.numeric.items()}, fold=self.fold[keep], inner=self.inner[keep])

    def keep_ids(self, ids: Sequence[int]) -> "Design":
        return self.subset(np.isin(self.ids, np.asarray(sorted(ids), dtype=np.int64)))

    def with_folds(self, fold: np.ndarray) -> "Design":
        return replace(self, fold=np.asarray(fold, dtype=np.int64))


# ============================================================================== athlete representation from attempts
def aggregate_attempts(Z: np.ndarray, nfl: np.ndarray, order: np.ndarray, train: np.ndarray, how: str = "mean"
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(athlete ids, (A, E) athlete representation, attempts per athlete). No NFL value is an argument.

    mean     mean of the athlete's valid attempts (the frozen rule of the shortlist)
    first    his first valid attempt in time (sensitivity: no averaging advantage for athletes with a repeat)
    shrunk   mean pulled towards the TRAINING athletes' mean by its reliability σ²_between / (σ²_between + σ²_within / n):
             an athlete with one attempt is trusted less than one with two. Variance components come from training
             athletes only (``train``: per-attempt flag)."""
    ids, inv = np.unique(nfl, return_inverse=True)
    A, E = ids.size, Z.shape[1]
    n = np.bincount(inv, minlength=A).astype(float)
    mean = np.column_stack([np.bincount(inv, weights=Z[:, j], minlength=A) for j in range(E)]) / n[:, None]
    if how == "mean":
        return ids, mean, n.astype(np.int64)
    if how == "first":
        o = np.lexsort((order, inv))
        first = o[np.r_[True, inv[o][1:] != inv[o][:-1]]]
        return ids, Z[first], n.astype(np.int64)
    if how != "shrunk":
        raise ValueError(f"unknown aggregation {how!r} (mean · first · shrunk)")
    tr_a = np.bincount(inv, weights=train.astype(float), minlength=A) > 0
    df_w = float((n[tr_a] - 1).sum())
    if tr_a.sum() < 3 or df_w < 1:
        return ids, mean, n.astype(np.int64)
    s2w = ((Z - mean[inv])[tr_a[inv]] ** 2).sum(0) / df_w
    mu = mean[tr_a].mean(0)
    s2b = np.maximum(mean[tr_a].var(0, ddof=1) - s2w * float(np.mean(1.0 / n[tr_a])), 0.0)
    den = s2b[None, :] + s2w[None, :] / n[:, None]
    rel = np.where(den > 0, s2b[None, :] / np.where(den > 0, den, 1.0), 1.0)
    return ids, mu + rel * (mean - mu), n.astype(np.int64)


class RepBlocks:
    """Athlete-level representation blocks of one scheme, built on demand from the attempt table of
    ``transfer_repr`` (``ATTEMPT_REP``). ``order``: seq_id → attempt order (1 = first in time)."""

    def __init__(self, attempts: pl.DataFrame, order: dict[str, int], scheme: str = "outer", how: str = "mean"):
        self.t = attempts.filter(pl.col("scheme") == scheme)
        self.order, self.scheme, self.how = order, scheme, how
        self._cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def families(self) -> list[str]:
        return sorted(self.t["family"].unique().to_list())

    def folds(self) -> list[int]:
        return sorted(self.t["fold"].unique().to_list())

    def athlete(self, family: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (family, int(fold))
        if key not in self._cache:
            g = self.t.filter((pl.col("family") == family) & (pl.col("fold") == int(fold))).sort("seq_id", "dim")
            if not g.height:
                raise CohortError(f"no representation {family!r} for {self.scheme} fold {fold}: run the representation stage of 06B")
            E = g["dim"].n_unique()
            if g.height % E:
                raise CohortError(f"{family}, fold {fold}: {g.height} rows are not a multiple of the {E} dimensions")
            pick = slice(None, None, E)
            seq = g["seq_id"].to_numpy()[pick].astype(str)
            self._cache[key] = aggregate_attempts(
                g["value"].to_numpy().reshape(-1, E), g["nfl_id"].to_numpy()[pick].astype(np.int64),
                np.array([self.order.get(s, 10 ** 6) for s in seq]), g["role"].to_numpy()[pick].astype(str) == "train", self.how)
        return self._cache[key]

    def matrix(self, family: str, fold: int, ids: np.ndarray) -> np.ndarray:
        have, X, _ = self.athlete(family, fold)
        pos = np.searchsorted(have, ids)
        ok = (pos < have.size) & (have[np.minimum(pos, have.size - 1)] == ids)
        if not ok.all():
            raise CohortError(f"{family}, {self.scheme} fold {fold}: {int((~ok).sum())} athlete(s) of the evaluation cohort have no "
                              f"representation ({ids[~ok][:5].tolist()} …). Embeddings are never imputed: restrict the cohort "
                              "with common_support first")
        return X[pos]

    def available(self, families: Sequence[str], folds: Sequence[int]) -> dict[int, list[str]]:
        """athlete → the (family, fold) blocks in which he has NO representation (empty list = fully supported)."""
        everyone = sorted(set(self.t["nfl_id"].unique().to_list()))
        out: dict[int, list[str]] = {int(i): [] for i in everyone}
        for fam in families:
            for f in folds:
                have = set(self.athlete(fam, f)[0].tolist())
                for i in everyone:
                    if i not in have:
                        out[int(i)].append(f"{fam}@{f}")
        return out


class ArrayBlocks:
    """Fold-independent blocks held in memory ({family: (n, E) aligned to ``ids``}) — fixtures and permutation wrappers."""

    def __init__(self, ids: np.ndarray, blocks: dict[str, np.ndarray]):
        self.ids, self.blocks = np.asarray(ids), blocks

    def matrix(self, family: str, fold: int, ids: np.ndarray) -> np.ndarray:
        pos = {int(i): k for k, i in enumerate(self.ids)}
        return self.blocks[family][[pos[int(i)] for i in ids]]


def common_support(design: Design, blocks: RepBlocks, families: Sequence[str]) -> pl.DataFrame:
    """Athletes of the 06A primary cohort that every family represents under EVERY fold model. The others leave the
    paired cohort with the blocks they lack; the folds are kept as assigned (nothing is re-dealt)."""
    gaps = blocks.available(families, sorted(np.unique(design.fold).tolist()))
    miss = [", ".join(gaps.get(int(i), ["no attempt represented"])) for i in design.ids]
    return pl.DataFrame({"nfl_id": design.ids, "position_group": design.position, "draft_year": [int(c) if c.isdigit() else None for c in design.cohort],
                         "in_paired_cohort": [m == "" for m in miss], "missing": miss}, schema=SUPPORT.columns)


# ============================================================================== one fold: preprocessing and fit
def design_matrix(d: Design, spec: ModelSpec, reps: Sequence[tuple[str, np.ndarray]], train: np.ndarray
                  ) -> tuple[np.ndarray, list[str], np.ndarray, list[dict[str, Any]]]:
    """(X for EVERY athlete, term names, penalty factors, preprocessing rows) with every statistic taken from the
    ``train`` rows: imputation medians, centres, scales, dummy levels, which indicators exist. A term that is
    constant among the training athletes is left out. Position dummies are unpenalised (additive position effects)."""
    cols, names, pen, prep = [], [], [], []

    def put(name: str, block: str, x: np.ndarray, penalty: float, median: float = NAN) -> None:
        mu, sd = float(x[train].mean()), float(x[train].std())
        if not sd > 1e-12:
            return
        cols.append((x - mu) / sd)
        names.append(name)
        pen.append(penalty)
        prep.append({"term": name, "block": block, "penalised": penalty > 0, "median": median, "mean": mu, "sd": sd})

    if spec.base in ("position", "M0", "M1"):
        for level in np.unique(d.position[train])[1:]:
            put(f"position={level}", "position", (d.position == level).astype(float), 0.0)
    if spec.base in ("M0", "M1"):
        for level in np.unique(d.cohort[train])[1:]:
            put(f"cohort={level}", "cohort", (d.cohort == level).astype(float), 1.0)
        for c in (d.m0 if spec.base == "M0" else d.m1):
            x = d.numeric[c]
            obs = x[train][np.isfinite(x[train])]
            med = float(np.median(obs)) if obs.size else 0.0
            put(c, "body" if c in d.m0 else "official", np.where(np.isfinite(x), x, med), 1.0, med)
            if c in d.indicators:
                put(f"{c}__missing", "indicator", (~np.isfinite(x)).astype(float), 1.0)
    for family, Z in reps:
        if not np.isfinite(Z).all():
            raise CohortError(f"{family}: a representation value is not finite (embeddings are never imputed)")
        for j in range(Z.shape[1]):
            put(f"{family}_{j + 1}", family, Z[:, j], 1.0)
    X = np.column_stack(cols) if cols else np.empty((d.n, 0))
    return X, names, np.asarray(pen, dtype=float), prep


def ridge_solve(X: np.ndarray, y: np.ndarray, alpha: float, pen: np.ndarray, tc: TransferConfig | None = None) -> tuple[float, np.ndarray]:
    """Ridge with an unpenalised intercept and per-term penalty factors: minimise ‖y − b0 − Xb‖² + α Σ pen_j b_j²."""
    ybar = float(y.mean())
    if X.shape[1] == 0:
        return ybar, np.zeros(0)
    xbar = X.mean(0)
    Xc = X - xbar
    A = Xc.T @ Xc + float(alpha) * np.diag(pen) + 1e-8 * np.eye(X.shape[1])
    beta = np.linalg.solve(A, Xc.T @ (y - ybar))
    return ybar - float(xbar @ beta), beta


def enet_solve(X: np.ndarray, y: np.ndarray, alpha: float, pen: np.ndarray, tc: TransferConfig | None = None,
               n_iter: int = 300, tol: float = 1e-7) -> tuple[float, np.ndarray]:
    """Elastic net by cyclic coordinate descent (exploratory learner): minimise
    ½‖y − b0 − Xb‖² + α Σ pen_j (ρ |b_j| + ½ (1 − ρ) b_j²), ρ = ``enet_l1_ratio``."""
    rho = 0.5 if tc is None else tc.enet_l1_ratio
    ybar = float(y.mean())
    p = X.shape[1]
    if p == 0:
        return ybar, np.zeros(0)
    xbar = X.mean(0)
    Xc, r = X - xbar, y - ybar
    ss = (Xc ** 2).sum(0)
    beta = np.zeros(p)
    for _ in range(n_iter):
        big = 0.0
        for j in range(p):
            old = beta[j]
            z = float(Xc[:, j] @ r) + ss[j] * old
            t = alpha * pen[j] * rho
            new = np.sign(z) * max(abs(z) - t, 0.0) / (ss[j] + alpha * pen[j] * (1.0 - rho) + 1e-12)
            if new != old:
                r -= Xc[:, j] * (new - old)
                beta[j] = new
                big = max(big, abs(new - old))
        if big < tol:
            break
    return ybar - float(xbar @ beta), beta


SOLVERS: dict[str, Callable[..., tuple[float, np.ndarray]]] = {"ridge": ridge_solve, "enet": enet_solve}


def select_alpha(d: Design, spec: ModelSpec, reps: Sequence[tuple[str, np.ndarray]], y: np.ndarray, train: np.ndarray,
                 tc: TransferConfig, learner: str = "ridge") -> float:
    """Penalty minimising the squared error over inner athlete-grouped folds of the TRAINING athletes; preprocessing is
    refitted on each inner training part. Ties go to the larger penalty. (The labels of a context-adjusted target use
    context means of the whole outer training set — the held-out athletes of the outer fold are never involved.)"""
    alphas, solve = tc.alphas, SOLVERS[learner]
    sse = np.zeros(alphas.size)
    used = 0
    for j in np.unique(d.inner[train]):
        va = train & (d.inner == j)
        fit = train & ~va
        if not va.any() or fit.sum() < 5:
            continue
        X, _, pen, _ = design_matrix(d, spec, reps, fit)
        for i, a in enumerate(alphas):
            b0, beta = solve(X[fit], y[fit], float(a), pen, tc)
            sse[i] += float(((y[va] - b0 - X[va] @ beta) ** 2).sum())
        used += 1
    if not used:
        return float(alphas[alphas.size // 2])
    return float(alphas[alphas.size - 1 - int(np.argmin(sse[::-1]))])


def fit_predict(d: Design, spec: ModelSpec, reps: Sequence[tuple[str, np.ndarray]], y: np.ndarray, train: np.ndarray,
                test: np.ndarray, tc: TransferConfig, learner: str = "ridge") -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """(predictions of the ``test`` athletes, penalty, term rows) of one model in one outer fold."""
    if spec.base == "none" and not reps:
        return np.full(int(test.sum()), float(y[train].mean())), NAN, []
    X, _, pen, prep = design_matrix(d, spec, reps, train)
    alpha = select_alpha(d, spec, reps, y, train, tc, learner) if pen.any() else NAN
    b0, beta = SOLVERS[learner](X[train], y[train], alpha if np.isfinite(alpha) else 0.0, pen, tc)
    return b0 + X[test] @ beta, alpha, [{**r, "coefficient": float(b), "alpha": alpha} for r, b in zip(prep, beta)]


# ============================================================================== out-of-fold loop
def label_fn(tset: TargetSet, target: FrozenTarget, ids: np.ndarray, variant: str = "all", adjustment: str | None = None
             ) -> Callable[[np.ndarray | None], np.ndarray]:
    """train athlete ids → labels of ``ids``, with any context adjustment estimated on those training athletes."""
    pos = np.searchsorted(tset.ids, ids)
    if (pos >= tset.ids.size).any() or not np.array_equal(tset.ids[np.minimum(pos, tset.ids.size - 1)], ids):
        raise CohortError("an athlete of the evaluation cohort has no NFL window in the play table")
    return lambda train_ids: tset.athlete_values(target.target_id, train_ids, variant, adjustment)["value"][pos]


def run_oof(d: Design, y_fn: Callable[[np.ndarray | None], np.ndarray], blocks: Any, models: Sequence[ModelSpec], tc: TransferConfig,
            learner: str = "ridge", permute_rep: np.ndarray | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One prediction per athlete and model. ``blocks.matrix(family, fold, ids)`` gives the fold's representation block.
    ``permute_rep``: athlete permutation applied to every representation block (linkage null). Returns
    ({y, y_ref, n_train, pred{model}, alpha{model}}, preprocessing rows)."""
    n = d.n
    res: dict[str, Any] = {"y": np.full(n, NAN), "y_ref": np.full(n, NAN), "n_train": np.zeros(n, dtype=np.int64),
                           "pred": {m.model: np.full(n, NAN) for m in models}, "alpha": {m.model: np.full(n, NAN) for m in models}}
    rows: list[dict[str, Any]] = []
    for f in np.unique(d.fold):
        test = d.fold == f
        train = ~test
        if train.sum() < 5:
            raise CohortError(f"fold {f}: only {int(train.sum())} training athletes")
        y = np.asarray(y_fn(d.ids[train]), dtype=float)
        if not np.isfinite(y).all():
            raise CohortError(f"fold {f}: {int((~np.isfinite(y)).sum())} athlete(s) of the evaluation cohort have no label under "
                              "this exposure rule; restrict the cohort to eligible athletes first")
        res["y"][test], res["y_ref"][test], res["n_train"][test] = y[test], float(y[train].mean()), int(train.sum())
        cache: dict[str, np.ndarray] = {}
        for spec in models:
            for fam in spec.families:
                if fam not in cache:
                    Z = blocks.matrix(fam, int(f), d.ids)
                    cache[fam] = Z if permute_rep is None else Z[permute_rep]
            pred, alpha, terms = fit_predict(d, spec, [(fam, cache[fam]) for fam in spec.families], y, train, test, tc, learner)
            res["pred"][spec.model][test], res["alpha"][spec.model][test] = pred, alpha
            rows += [{"model": spec.model, "fold": int(f), **t} for t in terms]
    return res, rows


def _labelled(df: pl.DataFrame, contract: TableContract, **constants: str) -> pl.DataFrame:
    return df.with_columns([pl.lit(v, dtype=pl.String).alias(k) for k, v in constants.items()]).select(
        [pl.col(k).cast(t) for k, t in contract.columns.items()])


def oof_frame(analysis: str, cohort: str, target_id: str, d: Design, res: dict[str, Any]) -> pl.DataFrame:
    year = pl.Series("draft_year", [int(c) if c.isdigit() else None for c in d.cohort], dtype=pl.Int64)
    parts = [_labelled(pl.DataFrame({"nfl_id": d.ids, "position_group": d.position, "fold": d.fold, "y": res["y"], "y_ref": res["y_ref"],
                                     "pred": p, "alpha": res["alpha"][m], "n_train": res["n_train"]}).with_columns(year),
                       OOF, analysis=analysis, cohort=cohort, target_id=target_id, model=m) for m, p in res["pred"].items()]
    out = pl.concat(parts).with_columns(pl.col("alpha").fill_nan(None))
    if out.select(pl.col("pred").is_nan().any() | pl.col("pred").is_null().any()).item():
        raise CohortError(f"{analysis} / {target_id}: an athlete has no out-of-fold prediction")
    return out


def prep_frame(analysis: str, cohort: str, target_id: str, rows: Sequence[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return PREP.empty()
    df = pl.DataFrame([{**r, "analysis": analysis, "cohort": cohort, "target_id": target_id} for r in rows],
                      schema={**{k: PREP.columns[k] for k in rows[0]}, **_K})
    return df.select(list(PREP.columns)).with_columns(pl.col("median").fill_nan(None), pl.col("alpha").fill_nan(None))


def targets_frame(analysis: str, cohort: str, target: FrozenTarget, d: Design, res: dict[str, Any], tset: TargetSet,
                  variant: str = "all", adjustment: str | None = None) -> pl.DataFrame:
    v = tset.athlete_values(target.target_id, None, variant, adjustment)
    pos = np.searchsorted(tset.ids, d.ids)
    df = pl.DataFrame({"nfl_id": d.ids, "fold": d.fold, "y": res["y"], "y_descriptive": v["value"][pos],
                       "n_windows": v["n_windows"][pos], "n_games": v["n_games"][pos]})
    return _labelled(df, OOF_TARGETS, analysis=analysis, cohort=cohort, target_id=target.target_id, variant=variant,
                     adjustment=target.adjustment if adjustment is None else adjustment)


def run_analysis(analysis: str, target: FrozenTarget, d: Design, tset: TargetSet, blocks: Any, models: Sequence[ModelSpec],
                 tc: TransferConfig, variant: str = "all", adjustment: str | None = None, min_windows: int | None = None,
                 learner: str = "ridge", cohort: str = PRIMARY, min_games: int = 0) -> dict[str, pl.DataFrame]:
    """One labelled analysis of one frozen target: restrict the cohort to athletes meeting the exposure rule, run the
    out-of-fold loop, return {OOF, OOF_TARGETS, PREP} frames. The cohort is restricted BEFORE any model is fitted and
    identically for every model."""
    v = tset.athlete_values(target.target_id, None, variant, "none")
    pos = np.searchsorted(tset.ids, d.ids)
    need = target.min_windows if min_windows is None else int(min_windows)
    dd = d.subset((v["n_windows"][pos] >= need) & (v["n_games"][pos] >= int(min_games)))
    if dd.n < tc.min_athletes or np.unique(dd.fold).size < 2:
        log.warning("[advanced_movement:TRANSFER] %s / %s skipped: %d eligible athletes (minimum %d)", analysis, target.target_id,
                    dd.n, tc.min_athletes)
        return {OOF.name: OOF.empty(), OOF_TARGETS.name: OOF_TARGETS.empty(), PREP.name: PREP.empty()}
    res, rows = run_oof(dd, label_fn(tset, target, dd.ids, variant, adjustment), blocks, models, tc, learner)
    return {OOF.name: oof_frame(analysis, cohort, target.target_id, dd, res),
            OOF_TARGETS.name: targets_frame(analysis, cohort, target, dd, res, tset, variant, adjustment),
            PREP.name: prep_frame(analysis, cohort, target.target_id, rows)}


def concat(parts: Sequence[dict[str, pl.DataFrame]], contracts: Sequence[TableContract] = (OOF, OOF_TARGETS, PREP)) -> dict[str, pl.DataFrame]:
    return {c.name: (pl.concat([p[c.name] for p in parts if p[c.name].height]) if any(p[c.name].height for p in parts) else c.empty())
            for c in contracts}


# ============================================================================== evaluation
def paired_arrays(o: pl.DataFrame) -> dict[str, Any]:
    """Arrays of ONE (analysis, cohort, target): raises unless every model has exactly one prediction for exactly the
    same athletes, scored against exactly the same labels."""
    out: dict[str, Any] = {"pred": {}}
    for (m,), g in o.partition_by(["model"], as_dict=True, maintain_order=True).items():
        g = g.sort("nfl_id")
        ids = g["nfl_id"].to_numpy()
        if np.unique(ids).size != ids.size:
            raise CohortError(f"model {m}: an athlete has several out-of-fold predictions")
        if "ids" not in out:
            out.update(ids=ids, y=g["y"].to_numpy().astype(float), y_ref=g["y_ref"].to_numpy().astype(float),
                       position=g["position_group"].to_numpy().astype(str), cohort=g["draft_year"].cast(pl.String).fill_null("?").to_numpy(),
                       fold=g["fold"].to_numpy())
        elif not np.array_equal(ids, out["ids"]):
            raise CohortError(f"model {m} was evaluated on {ids.size} athletes, the first model on {out['ids'].size}: paired "
                              "comparisons need identical held-out athletes")
        elif not np.allclose(g["y"].to_numpy(), out["y"], equal_nan=False):
            raise CohortError(f"model {m} was scored against different labels than the first model")
        out["pred"][str(m)] = g["pred"].to_numpy().astype(float)
    return out


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    a, b = rankdata(y), rankdata(p)
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else NAN


def point_metrics(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    m = {k: float(v) for k, v in metric_arrays(y, p, ref).items()}
    m["spearman"] = _spearman(y, p)
    return m


def boot_metrics(y: np.ndarray, p: np.ndarray, ref: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics of every athlete resample (``idx``: (replicates, n) athlete indices)."""
    m = {k: np.asarray(v, dtype=float) for k, v in metric_arrays(y[idx], p[idx], ref[idx]).items()}
    m["spearman"] = np.array([_spearman(y[i], p[i]) for i in idx])
    return m


def _interval(reps: np.ndarray) -> tuple[float | None, float | None]:
    ok = reps[np.isfinite(reps)]
    return (float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))) if ok.size >= 20 else (None, None)


def evaluate(oof: pl.DataFrame, pairs: Sequence[tuple[str, str]], tc: TransferConfig, seed: int, n_boot: int | None = None
             ) -> dict[str, pl.DataFrame]:
    """{PERF, PAIRED} for every (analysis, cohort, target) of ``oof``. Every model of a group is resampled with the
    SAME athlete indices, so differences are paired. Intervals are given only with ≥ ``min_athletes_group`` athletes."""
    perf, paired = [], []
    B_ = tc.n_boot if n_boot is None else int(n_boot)
    for key, o in sorted(oof.partition_by(["analysis", "cohort", "target_id"], as_dict=True).items()):
        a = paired_arrays(o)
        n = a["ids"].size
        if n < 3:
            continue
        idx = (np.random.default_rng(seed_for(seed, "boot", *key)).integers(0, n, size=(B_, n))
               if B_ > 0 and n >= tc.min_athletes_group else None)
        point = {m: point_metrics(a["y"], p, a["y_ref"]) for m, p in a["pred"].items()}
        boot = {m: boot_metrics(a["y"], p, a["y_ref"], idx) for m, p in a["pred"].items()} if idx is not None else {}
        head = dict(zip(("analysis", "cohort", "target_id"), key))
        size = {"n_athletes": int(n), "n_positions": int(np.unique(a["position"]).size), "n_cohorts": int(np.unique(a["cohort"]).size)}
        for m in point:
            for k in METRICS:
                lo, hi = _interval(boot[m][k]) if m in boot else (None, None)
                est = point[m][k]
                perf.append({**head, "model": m, "metric": k, "estimate": est if np.isfinite(est) else None, "ci_lower": lo,
                             "ci_upper": hi, **size})
        for m, r in pairs:
            if m not in point or r not in point:
                continue
            for k, sign in DELTAS:
                est = sign * (point[m][k] - point[r][k])
                reps = sign * (boot[m][k] - boot[r][k]) if m in boot else np.empty(0)
                lo, hi = _interval(reps)
                ok = reps[np.isfinite(reps)]
                rel = est / point[r][k] if sign < 0 and point[r][k] > 0 else None
                paired.append({**head, "model": m, "reference": r, "metric": f"delta_{k}", "estimate": est if np.isfinite(est) else None,
                               "ci_lower": lo, "ci_upper": hi, "relative": rel, "share_boot_improved": float((ok > 0).mean()) if ok.size else None,
                               "n_athletes": int(n), "established": bool(lo is not None and lo > 0),
                               "worse": bool(hi is not None and hi < 0)})
    return {PERF.name: pl.DataFrame(perf, schema=PERF.columns) if perf else PERF.empty(),
            PAIRED.name: pl.DataFrame(paired, schema=PAIRED.columns) if paired else PAIRED.empty()}


def fold_performance(oof: pl.DataFrame) -> pl.DataFrame:
    key = ["analysis", "cohort", "target_id"]
    e = oof.with_columns((pl.col("y") - pl.col("pred")).alias("_e"))
    f = e.group_by(*key, "model", "fold").agg(pl.len().cast(pl.Int64).alias("n_athletes"), (pl.col("_e") ** 2).mean().sqrt().alias("rmse"),
                                               pl.col("_e").abs().mean().alias("mae"))
    ref = f.filter(pl.col("model") == REFERENCE).select(*key, "fold", pl.col("rmse").alias("_r"))
    out = f.join(ref, on=[*key, "fold"], how="left").with_columns((pl.col("_r") - pl.col("rmse")).alias("delta_rmse_vs_m1")).drop("_r")
    return out.select(list(FOLD_PERF.columns)).sort(*key, "model", "fold")


# ============================================================================== permutation checks
def _within(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(groups.size)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm[idx] = rng.permutation(idx)
    return perm


def permutation_null(d: Design, y_fn: Callable[[np.ndarray | None], np.ndarray], blocks: Any, models: Sequence[ModelSpec],
                     pair: tuple[str, str], tc: TransferConfig, seed: int, mode: str = "outcome") -> dict[str, Any]:
    """Null distribution of Δ RMSE(reference − model) for one model pair.

    outcome   labels permuted between athletes WITHIN position (position means survive; every athlete-level link
              between predictors and outcome is broken). A pipeline that still shows a gain is leaking.
    linkage   the representation rows permuted between athletes within position; labels and official results stay.
              The gain of a representation model should vanish — if not, it does not come from the athlete's own run.
    p = (1 + #{null ≥ observed}) / (1 + permutations). A check of the pipeline, not a test replacing held-out validation."""
    if mode not in ("outcome", "linkage"):
        raise ValueError(f"mode must be 'outcome' or 'linkage', got {mode!r}")
    use = [m for m in models if m.model in pair]
    if len(use) != 2:
        raise ValueError(f"both models of {pair} must be in the ladder")

    def delta(res: dict[str, Any]) -> float:
        e = {m: float(np.sqrt(np.mean((res["y"] - res["pred"][m]) ** 2))) for m in pair}
        return e[pair[1]] - e[pair[0]]

    observed = delta(run_oof(d, y_fn, blocks, use, tc)[0])
    rng = np.random.default_rng(seed_for(seed, "perm", mode, *pair))
    null = np.empty(tc.n_perm)
    for b in range(tc.n_perm):
        perm = _within(d.position, rng)
        if mode == "outcome":
            res = run_oof(d, lambda t, _p=perm: np.asarray(y_fn(t))[_p], blocks, use, tc)[0]
        else:
            res = run_oof(d, y_fn, blocks, use, tc, permute_rep=perm)[0]
        null[b] = delta(res)
    return {"observed": observed, "null": null, "p_value": float((1 + (null >= observed).sum()) / (1 + null.size)), "mode": mode}


def perm_row(analysis: str, cohort: str, target_id: str, pair: tuple[str, str], out: dict[str, Any]) -> dict[str, Any]:
    return {"analysis": analysis, "cohort": cohort, "target_id": target_id, "model": pair[0], "reference": pair[1], "mode": out["mode"],
            "observed": float(out["observed"]), "null_mean": float(np.mean(out["null"])), "null_q95": float(np.quantile(out["null"], 0.95)),
            "p_value": float(out["p_value"]), "n_perm": int(out["null"].size)}


# ============================================================================== gate and reading
def gate_06b(oof: pl.DataFrame, paired: pl.DataFrame, fits: pl.DataFrame, families: dict[str, Any], dims: dict[str, int],
             primary_target: str) -> pl.DataFrame:
    """Gate 06B — predictive evaluation. A positive result is NOT a criterion: the gate says the experiment is valid."""
    o = oof.filter(pl.col("analysis") == PRIMARY)
    sets: dict[tuple[str, str], dict[str, tuple[int, frozenset]]] = {}
    for (c, t, m), g in o.partition_by(["cohort", "target_id", "model"], as_dict=True).items():
        sets.setdefault((c, t), {})[m] = (g.height, frozenset(g["nfl_id"].to_list()))
    identical = bool(sets) and all(len({v[1] for v in by.values()}) == 1 for by in sets.values())
    unique = bool(sets) and all(v[0] == len(v[1]) for by in sets.values() for v in by.values())
    same = pl.DataFrame({"group": [f"{c} / {t}" for c, t in sets]})
    models = set(o["model"].unique().to_list())
    want = {"ENG": len(families.get("ENG") or []), "SPEED2": len(families.get("SPEED2") or [])}
    frozen_ok = all(dims.get(k) == v for k, v in want.items() if v) and all(k in dims for k in ("FPCA", "TCN"))
    leak = int(fits["n_overlap"].sum()) if fits.height else -1
    main = paired.filter((pl.col("analysis") == PRIMARY) & (pl.col("cohort") == PRIMARY) & (pl.col("target_id") == primary_target)
                         & (pl.col("model") == "M5") & (pl.col("reference") == REFERENCE) & (pl.col("metric") == "delta_rmse"))
    has_ci = bool(main.height and main["ci_lower"][0] is not None)
    rows = [("identical_heldout_athletes", identical, f"{same.height} (cohort, target) group(s): every model on the same athletes"),
            ("official_baseline_included", REFERENCE in models and "M0" in models, f"models evaluated: {sorted(models)}"),
            ("frozen_feature_candidates_used", frozen_ok, f"block sizes {dims}; frozen engineered ids {families.get('ENG')}"),
            ("inductive_representation_fitting", leak == 0, f"{fits.height} fold fits, {max(leak, 0)} athlete(s) shared between a fit and "
             "its held-out fold" if leak >= 0 else "no encoder fit table"),
            ("oof_predictions_unique", unique, "one prediction per athlete, model, target and cohort"),
            ("uncertainty_reported", has_ci, "paired athlete-bootstrap interval of Δ RMSE (M5 − M1) on the primary target"
             if has_ci else "no interval for the primary comparison")]
    return pl.DataFrame([{"criterion": c, "passed": bool(p), "detail": d} for c, p, d in rows], schema=tr.GATE.columns)


def _fmt(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{digits}f}"


def conclusions_06b(perf: pl.DataFrame, paired: pl.DataFrame, targets: Sequence[FrozenTarget], tc: TransferConfig) -> str:
    """Reading generated ONLY from the saved tables of this run."""
    lines = ["### What the tables show (primary analysis, paired cohort)", ""]
    for t in targets:
        p = paired.filter((pl.col("analysis") == PRIMARY) & (pl.col("cohort") == PRIMARY) & (pl.col("target_id") == t.target_id)
                          & (pl.col("metric") == "delta_rmse"))
        if not p.height:
            lines.append(f"- `{t.target_id}` ({t.study_role}): not evaluated (too few eligible athletes).")
            continue
        n = p["n_athletes"][0]
        base = perf.filter((pl.col("analysis") == PRIMARY) & (pl.col("cohort") == PRIMARY) & (pl.col("target_id") == t.target_id)
                           & (pl.col("model") == REFERENCE) & pl.col("metric").is_in(["rmse", "r2_oos"]))
        b = {r["metric"]: r["estimate"] for r in base.iter_rows(named=True)}
        lines.append(f"- **`{t.target_id}`** ({t.study_role}, {n} athletes, {t.unit}): M1 RMSE {_fmt(b.get('rmse')).lstrip('+')}, "
                     f"R² {_fmt(b.get('r2_oos'))}.")
        for r in p.sort("reference", "model").iter_rows(named=True):
            verdict = ("established" if r["established"] else "worse than the reference" if r["worse"] else "not established (interval includes 0)")
            rel = "" if r["relative"] is None else f", {100 * r['relative']:+.1f} %"
            lines.append(f"    - {r['model']} vs {r['reference']}: Δ RMSE {_fmt(r['estimate'])} [{_fmt(r['ci_lower'])}, {_fmt(r['ci_upper'])}]{rel} → {verdict}")
    lines += ["", f"Δ RMSE = RMSE(reference) − RMSE(model); positive = lower error than the reference. Intervals: paired bootstrap of "
              f"athletes ({tc.n_boot} replicates) on out-of-fold predictions. A gain below {100 * tc.min_relative_gain:g} % of the reference "
              "error is not treated as practically relevant (heuristic). Exploratory rows (M7, M5S, secondary targets) cannot replace "
              "the primary comparison M5 vs M1 on the primary target."]
    return "\n".join(lines)
