"""Inductive evaluation of the functional PCA representations (notebook 02). No NFL value is read here.

Fit stage      for every representation, split repeat and outer fold: the model is fitted on the training athletes
               and applied to the held-out athletes, whose scores are therefore out-of-fold (INDUCTIVE). One extra
               fit on all athletes (the REFERENCE, transductive) fixes the sign and the order of the components of
               every fold fit and is used for descriptive figures; no reported statistic uses its scores.
Reliability    stratified ICC(1) of the out-of-fold attempt scores (``sprint_rescue.stats``). The outer fold is
               always part of the stratum, so a difference between two fold fits can never count as a difference
               between athletes. Intervals: athlete bootstrap, on split repeat 0; the other repeats show how much
               an estimate moves with the split.
Redundancy     athlete = mean of his attempt scores. Partial Spearman with the official forty and 10-yard split,
               and the share of a score's variance the two explain together, within position.
               reliable_unique_share = max(reliability of the athlete mean − that share, 0), the sprint-rescue
               lower bound of reproducible variance the stopwatch does not explain.
Window         does a score follow the RECORDING? Between athletes (athlete means, within position) and — the
               sharper test — within athlete: attempt 2 − attempt 1 of the score against attempt 2 − attempt 1 of
               each recording-window variable. An athlete does not change between two attempts; his recording does.
Sensitivity    share of a component's between-athlete variance reproduced by the scores of the alternate
               alignment of the same representation.
Gate           rule-based labels per component from the intervals above. They are screening labels with the
               continuous evidence beside them, and a proposal for review — not a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl

from ...data.utils import atomic_write_json, file_fingerprint, get_logger, read_json, stable_hash, utc_now
from ..contracts import TableContract
from ..sprint_rescue.stats import bootstrap_ci, partial_corr, r_squared, seed_for, spearman_brown, stratified_icc
from .data import fold_athletes
from .fpca import REPRESENTATIONS, SPEC, CurveSet, FpcaConfig, FpcaModel, align_to_reference, fit_fpca
from .fpca import load_models, save_models

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
FORTY, TEN, WEIGHT, HEIGHT = "cmb_forty", "cmb_ten_yd_split", "cmb_combine_weight", "cmb_combine_height"
WINDOW_VARS = ("first_speed", "duration_s", "frames_after_peak", "path_yd", "last_speed_frac_of_peak", "pos_resid_rms",
               "a_tan_roughness", "share_boundary")
FLAG_VARS = ("first_frame_censored", "start_above_ref_speed", "last_frame_censored", "ends_at_peak", "boundary_peak",
             "signal_flag")
JOINT_WINDOW = ("first_speed", "duration_s", "frames_after_peak")
ROLE_ORDER = {"primary": 0, "secondary": 1, "control": 2, "alternate": 3}
_K = {"representation": S, "k": I, "repeat": I}

SCORES = TableContract(
    "fpca_attempt_scores", "Out-of-fold FPCA scores of every scored attempt (inductive).",
    {**_K, "outer_fold": I, "seq_id": S, "nfl_id": I, "component": I, "score": F, "score_z": F, "observed_fraction": F},
    key=("representation", "k", "repeat", "seq_id", "component"), kind="dataset",
    area="representation × K × split repeat × sequence × component",
    docs={"score": "in the units of the curve × √axis; fitted on the training athletes of the outer fold",
          "score_z": "score / √(training eigenvalue of the component)",
          "observed_fraction": "share of the model's support the attempt covers"})
ATHLETE_SCORES = TableContract(
    "fpca_athlete_scores", "Athlete-level FPCA features: mean of the athlete's out-of-fold attempt scores.",
    {**_K, "nfl_id": I, "component": I, "n_attempts": I, "score_mean": F, "score_z_mean": F},
    key=("representation", "k", "repeat", "nfl_id", "component"), kind="dataset",
    area="representation × K × split repeat × athlete × component")
SEQ_RECON = TableContract(
    "fpca_sequence_reconstruction", "Held-out reconstruction error of every attempt, on its observed grid points.",
    {**_K, "outer_fold": I, "seq_id": S, "nfl_id": I, "scored": B, "observed_fraction": F, "n_points": I, "sse": F,
     "sst": F, "rmse": F}, key=("representation", "k", "repeat", "seq_id"), kind="dataset",
    area="representation × K × split repeat × sequence",
    docs={"sse": "Σ (observed − reconstruction)²", "sst": "Σ (observed − training mean)²",
          "scored": "False = the attempt covers too little of the support and has no score"})
FOLD_FITS = TableContract(
    "fpca_fold_fits", "Fold-specific fitted preprocessing and fit quality.",
    {**_K, "outer_fold": I, "n_train_athletes": I, "n_train_sequences": I, "n_test_athletes": I, "n_test_sequences": I,
     "n_test_scored": I, "support_lo": F, "support_hi": F, "n_support": I, "n_basis": I, "sigma2": F, "fve_train": F,
     "rmse_train": F, "fve_heldout": F, "rmse_heldout": F}, key=("representation", "k", "repeat", "outer_fold"),
    area="representation × K × split repeat × outer fold",
    docs={"support_lo": "first axis value of the fold's support (learned on its training athletes)",
          "sigma2": "variance per grid point not carried by the K components",
          "fve_heldout": "1 − Σ sse / Σ sst over the held-out athletes' observed points"})
FOLD_COMPONENTS = TableContract(
    "fpca_fold_components", "Eigenvalue, explained variance and congruence with the reference, per fit and component.",
    {"representation": S, "repeat": I, "outer_fold": I, "component": I, "eigenvalue": F, "explained_ratio": F,
     "source_rank": I, "congruence": F}, key=("representation", "repeat", "outer_fold", "component"),
    area="representation × fit × component",
    docs={"repeat": "−1 = the reference fit on all athletes (transductive; sign and order convention only)",
          "source_rank": "rank of the component by its own eigenvalue in this fit; ≠ component means it was re-ordered",
          "congruence": "|∫ φ_fold φ_reference| / (‖φ_fold‖ ‖φ_reference‖) on the common support"})
FUNCTIONS = TableContract(
    "fpca_functions", "Mean function, eigenfunctions and coverage of every fit, on its support.",
    {"representation": S, "repeat": I, "outer_fold": I, "component": I, "grid": F, "value": F, "coverage": F},
    key=("representation", "repeat", "outer_fold", "component", "grid"), kind="dataset",
    area="representation × fit × component × grid point",
    docs={"component": "0 = mean function, k ≥ 1 = eigenfunction k", "coverage": "training weight observed at the point"})
FIT_CONTRACTS = (SCORES, SEQ_RECON, FOLD_FITS, FOLD_COMPONENTS, FUNCTIONS)

VARIANCE = TableContract(
    "fpca_explained_variance", "Explained variance per component across fold fits, and stability of the eigenfunctions.",
    {"representation": S, "component": I, "n_fits": I, "explained_mean": F, "explained_min": F, "explained_max": F,
     "cumulative_mean": F, "explained_reference": F, "congruence_mean": F, "congruence_min": F, "n_reordered": I},
    key=("representation", "component"), area="representation × component",
    docs={"n_reordered": "fold fits in which the component is not at its own eigenvalue rank"})
RECON = TableContract(
    "fpca_reconstruction_error", "Held-out reconstruction error by number of components.",
    {**_K, "n_athletes": I, "n_sequences": I, "n_unscored": I, "rmse_heldout": F, "rmse_lower": F, "rmse_upper": F,
     "fve_heldout": F, "fve_lower": F, "fve_upper": F, "fve_train": F, "fve_fold_min": F, "fve_fold_max": F,
     "value_unit": S}, key=("representation", "k", "repeat"), area="representation × K × split repeat",
    docs={"rmse_lower": "athlete bootstrap, 95 %", "fve_train": "mean over folds of the training-athlete value"})
RELIABILITY = TableContract(
    "fpca_reliability", "Repeat-attempt reliability of the out-of-fold scores.",
    {**_K, "component": I, "scope": S, "n_athletes_repeat": I, "n_sequences": I, "n_strata": I, "icc": F, "ci_lower": F,
     "ci_upper": F, "var_between": F, "var_within": F, "typical_error": F, "reliability_of_mean": F},
    key=("representation", "k", "repeat", "component", "scope"), area="representation × K × repeat × component × scope",
    docs={"scope": "pooled · position_adjusted · cohort_adjusted · position_cohort_adjusted · position:<group> · "
                   "cohort:<year> (within position) · position_adjusted@<quality scenario>; the outer fold is part of "
                   "every stratum", "ci_lower": "athlete bootstrap within strata, 95 %; split repeat 0 only",
          "typical_error": "√ within-athlete variance", "reliability_of_mean": "Spearman–Brown for the athlete's mean"})
_ASSOC = {**_K, "component": I, "target": S, "scope": S, "n_athletes": I, "spearman": F, "spearman_lower": F,
          "spearman_upper": F, "r2": F, "r2_lower": F, "r2_upper": F}
_ASSOC_DOCS = {"target": "a+b = both jointly (only r2 is filled)", "spearman_lower": "Fisher z, n − 3 − q",
               "r2_lower": "athlete bootstrap, 95 %; split repeat 0, position_adjusted only",
               "scope": "pooled, or position_adjusted (position means removed from both sides)"}
REDUNDANCY = TableContract("fpca_stopwatch_redundancy", "Association of athlete-level scores with official Combine times.",
                           _ASSOC, key=("representation", "k", "repeat", "component", "target", "scope"),
                           area="representation × K × repeat × component × official time × scope", docs=_ASSOC_DOCS)
BODY = TableContract("fpca_body_association", "Association of athlete-level scores with Combine weight and height.",
                     _ASSOC, key=("representation", "k", "repeat", "component", "target", "scope"),
                     area="representation × K × repeat × component × measurement × scope", docs=_ASSOC_DOCS)
WINDOW = TableContract(
    "fpca_window_dependence", "Dependence of the scores on the recording window, censoring flags, noise and cohort.",
    {**_K, "component": I, "variable": S, "level": S, "n_athletes": I, "spearman": F, "spearman_lower": F,
     "spearman_upper": F, "r2": F, "r2_lower": F, "r2_upper": F},
    key=("representation", "k", "repeat", "component", "variable", "level"),
    area="representation × K × component × variable × level",
    docs={"level": "between_athletes = athlete means, within position · within_athlete = attempt 2 − attempt 1 of the "
                   "score against attempt 2 − attempt 1 of the variable",
          "variable": "window_joint = first_speed + duration_s + frames_after_peak together (r2) · draft_cohort = "
                      "cohort within position (r2)", "repeat": "split repeat 0 only"})
SENSITIVITY = TableContract(
    "fpca_alignment_sensitivity", "How much of each component the alternate alignment of the representation reproduces.",
    {"representation": S, "alternate": S, "k": I, "repeat": I, "component": I, "scope": S, "n_athletes": I,
     "r2_from_alternate": F, "r2_lower": F, "r2_upper": F, "best_alt_component": I, "best_abs_spearman": F},
    key=("representation", "k", "repeat", "component", "scope"), area="representation × K × repeat × component × scope",
    docs={"r2_from_alternate": "share of the athlete-level score explained by all K scores of the alternate alignment",
          "n_athletes": "athletes scored under both alignments"})
STABILITY = TableContract(
    "fpca_score_stability", "Agreement of athlete-level scores between split repeats (different training athletes).",
    {"representation": S, "k": I, "component": I, "repeat_a": I, "repeat_b": I, "n_athletes": I, "pearson": F},
    key=("representation", "k", "component", "repeat_a", "repeat_b"), area="representation × K × component × repeat pair")
EVAL_CONTRACTS = (ATHLETE_SCORES, VARIANCE, RECON, RELIABILITY, REDUNDANCY, BODY, WINDOW, SENSITIVITY, STABILITY)

GATE = TableContract(
    "fpca_component_gate", "Research gate: per-component evidence and rule-based labels at the benchmark K.",
    {"representation": S, "role": S, "k": I, "component": I, "explained_ratio": F, "congruence_min": F,
     "n_athletes_repeat": I, "icc_adj": F, "icc_lower": F, "icc_upper": F, "icc_pooled": F, "icc_cohort_min": F,
     "icc_s3": F, "icc_repeat_min": F, "icc_repeat_max": F, "reliability_of_mean": F, "rho_forty": F,
     "rho_forty_lower": F, "rho_forty_upper": F, "rho_ten": F, "redundancy_abs": F, "redundancy_lower": F,
     "redundancy_upper": F, "r2_stopwatch": F, "r2_stopwatch_lower": F, "r2_stopwatch_upper": F,
     "reliable_unique_share": F, "r2_body": F, "window_r2_within": F, "window_r2_lower": F, "window_r2_upper": F,
     "cohort_r2": F, "r2_from_alternate": F, "reproducible": S, "stopwatch": S, "window": S, "status": S},
    key=("representation", "component"), area="representation × component",
    docs={"icc_adj": "position-adjusted ICC(1), all sequences, split repeat 0", "icc_s3": "same on S3_no_boundary sequences",
          "icc_repeat_min": "smallest position-adjusted ICC over the split repeats",
          "redundancy_abs": "max(|ρ forty|, |ρ 10-yard|), position-adjusted", "r2_stopwatch": "forty + 10-yard, within position",
          "window_r2_within": "share of the attempt-to-attempt change of the score explained by the change of the window",
          "status": "candidate · redundant_with_stopwatch · window_artifact · not_repeatable · inconclusive — heuristic, for review"})
SUMMARY = TableContract(
    "fpca_representation_summary", "One row per representation at its benchmark K.",
    {"representation": S, "role": S, "alternate_of": S, "k_selected": I, "k_rule": S, "fve_heldout": F, "fve_lower": F,
     "fve_upper": F, "rmse_heldout": F, "value_unit": S, "n_athletes_scored": I, "n_athletes_repeat": I,
     "n_components_reproducible": I, "n_components_candidate": I, "max_icc_adj": F, "max_icc_component": I,
     "r2_forty_from_scores": F, "r2_forty_lower": F, "r2_forty_upper": F, "best_reliable_unique_share": F,
     "congruence_min": F, "proposed_benchmark": B}, key=("representation",), area="representation",
    docs={"k_rule": "how k_selected was obtained (no outcome is involved)",
          "r2_forty_from_scores": "share of the official forty explained by the K athlete-level scores, within position",
          "proposed_benchmark": "rule-based proposal of the low-complexity benchmark for the sequence models; needs review"})
GATE_CONTRACTS = (GATE, SUMMARY)


# ============================================================================== small helpers
def _cast(df: pl.DataFrame, c: TableContract) -> pl.DataFrame:
    return df.select([pl.col(n).cast(t) for n, t in c.columns.items()]).fill_nan(None)


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        v = v.item() if isinstance(v, np.generic) else v
        out[k] = None if isinstance(v, float) and not np.isfinite(v) else v
    return out


def _table(rows: list[dict[str, Any]], c: TableContract) -> pl.DataFrame:
    return pl.DataFrame([_clean(r) for r in rows], schema=c.columns) if rows else c.empty()


def _join(*parts: np.ndarray) -> np.ndarray:
    out = np.asarray(parts[0]).astype(str)
    for p in parts[1:]:
        out = np.char.add(np.char.add(out, "|"), np.asarray(p).astype(str))
    return out


def athlete_means(nfl_ids: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(athlete ids, mean of the finite rows per athlete (g, p), number of rows per athlete)."""
    ids, inv = np.unique(nfl_ids, return_inverse=True)
    V2 = np.asarray(V, float).reshape(len(nfl_ids), -1)
    ok = np.isfinite(V2)
    sums, cnt = np.zeros((ids.size, V2.shape[1])), np.zeros((ids.size, V2.shape[1]))
    np.add.at(sums, inv, np.where(ok, V2, 0.0))
    np.add.at(cnt, inv, ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return ids, np.where(cnt > 0, sums / cnt, np.nan), np.bincount(inv, minlength=ids.size)


def first_two(nfl_ids: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(athlete ids, row of the first attempt, row of the second attempt) for athletes with at least two rows."""
    idx = np.lexsort((order, nfl_ids))
    ids, start, counts = np.unique(nfl_ids[idx], return_index=True, return_counts=True)
    two = counts >= 2
    return ids[two], idx[start[two]], idx[start[two] + 1]


def _abs_interval(lo: float, hi: float) -> tuple[float, float]:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return NAN, NAN
    return (0.0 if lo < 0 < hi else min(abs(lo), abs(hi))), max(abs(lo), abs(hi))


def _r2_ci(y: np.ndarray, X: list[np.ndarray], cats: list[np.ndarray], n_boot: int, seed: int) -> tuple[float, float, float, int]:
    """(R², lower, upper, n) of y on X within the categorical strata; athlete bootstrap when n_boot > 0."""
    ok = np.isfinite(y)
    for x in X:
        ok &= np.isfinite(x)
    y, X, cats = y[ok], [x[ok] for x in X], [c[ok] for c in cats]
    X = [x for x in X if np.ptp(x) > 0] if y.size else []
    if not X:
        return NAN, NAN, NAN, int(y.size)
    est = r_squared(y, X, categorical=cats)
    lo, hi = NAN, NAN
    if n_boot > 0 and np.isfinite(est["r2"]):
        lo, hi = bootstrap_ci(lambda i: r_squared(y[i], [x[i] for x in X], categorical=[c[i] for c in cats])["r2"],
                              int(y.size), n_boot, seed)
    return est["r2"], lo, hi, est["n"]


# ============================================================================== sequence and athlete covariates
class Meta:
    """Per-sequence recording-window variables (``checkpoint.WINDOWS``) and per-athlete official measurements
    (``data.ATHLETES``) as arrays, looked up by sequence / athlete id."""

    def __init__(self, windows: pl.DataFrame, athletes: pl.DataFrame):
        self.row = {s: i for i, s in enumerate(windows["seq_id"].to_list())}
        self.window = {c: windows[c].cast(pl.Float64).fill_null(NAN).to_numpy() for c in (*WINDOW_VARS, *FLAG_VARS)}
        self.position = windows["position_group"].fill_null("?").to_numpy().astype(str)
        self.cohort = windows["draft_year"].cast(pl.String).fill_null("?").to_numpy().astype(str)
        self.order = windows["attempt_order"].fill_null(0).to_numpy()
        self.scenario = {s: windows[s].fill_null(False).to_numpy() for s in ("S2_signal", "S3_no_boundary")}
        self.athlete = {int(a): i for i, a in enumerate(athletes["nfl_id"].to_list())}
        self.official = {c: athletes[c].cast(pl.Float64).fill_null(NAN).to_numpy() for c in (FORTY, TEN, WEIGHT, HEIGHT)}
        self.athlete_position = athletes["position_group"].fill_null("?").to_numpy().astype(str)
        self.athlete_cohort = athletes["draft_year"].cast(pl.String).fill_null("?").to_numpy().astype(str)

    def rows(self, seq_ids: np.ndarray) -> np.ndarray:
        return np.array([self.row[str(s)] for s in seq_ids], dtype=int)

    def athlete_rows(self, nfl_ids: np.ndarray) -> np.ndarray:
        return np.array([self.athlete[int(a)] for a in nfl_ids], dtype=int)


@dataclass
class Block:
    """Out-of-fold scores of one (representation, K, split repeat): one row per scored attempt."""
    representation: str
    k: int
    repeat: int
    seq_ids: np.ndarray
    nfl_ids: np.ndarray
    fold: np.ndarray
    S: np.ndarray              # (n, k)
    Z: np.ndarray              # (n, k) standardised by the training eigenvalue
    frac: np.ndarray


def blocks(scores: pl.DataFrame) -> Iterator[Block]:
    for (rep, k, repeat), d in sorted(scores.partition_by(["representation", "k", "repeat"], as_dict=True).items()):
        d = d.sort("seq_id", "component")
        if d.height % k:
            raise ValueError(f"{rep}, K = {k}, repeat {repeat}: {d.height} score rows are not a multiple of K")
        pick = slice(None, None, k)
        yield Block(rep, int(k), int(repeat), d["seq_id"].to_numpy()[pick].astype(str), d["nfl_id"].to_numpy()[pick],
                    d["outer_fold"].to_numpy()[pick], d["score"].to_numpy().reshape(-1, k),
                    d["score_z"].to_numpy().reshape(-1, k), d["observed_fraction"].to_numpy()[pick])


# ============================================================================== fit stage
@dataclass
class FoldRun:
    """Everything fitted for one representation. ``models``: 'reference' and 'r<repeat>_f<fold>' → model."""
    curves: CurveSet
    models: dict[str, FpcaModel]
    tables: dict[str, pl.DataFrame]


def fold_tag(repeat: int, fold: int) -> str:
    return f"r{repeat}_f{fold}"


def _component_frame(name: str, repeat: int, fold: int, m: FpcaModel, congruence: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"component": np.arange(1, m.k_max + 1), "eigenvalue": m.eigenvalues, "explained_ratio": m.explained(),
                         "source_rank": m.source_rank, "congruence": congruence}).with_columns(
        pl.lit(name).alias("representation"), pl.lit(repeat).alias("repeat"), pl.lit(fold).alias("outer_fold"))


def _function_frame(name: str, repeat: int, fold: int, m: FpcaModel) -> pl.DataFrame:
    g, cov = m.grid, m.coverage[m.support]
    values = np.column_stack([m.mean, m.phi])
    K1 = values.shape[1]
    return pl.DataFrame({"component": np.repeat(np.arange(K1), g.size), "grid": np.tile(g, K1),
                         "value": values.T.ravel(), "coverage": np.tile(cov, K1)}).with_columns(
        pl.lit(name).alias("representation"), pl.lit(repeat).alias("repeat"), pl.lit(fold).alias("outer_fold"))


def _fve(res: dict[str, np.ndarray]) -> tuple[float, float]:
    sse, sst, n = np.nansum(res["sse"]), np.nansum(res["sst"]), int(res["n_points"].sum())
    return (1.0 - sse / sst if sst > 0 else NAN), (float(np.sqrt(sse / n)) if n else NAN)


def fit_representation(curves: CurveSet, splits: pl.DataFrame, fcfg: FpcaConfig) -> FoldRun:
    """Reference fit plus one inductive fit per split repeat and outer fold, with held-out scores and errors."""
    name = curves.spec.name
    known = set(splits["nfl_id"].to_list())
    everyone = sorted({int(i) for i in curves.nfl_ids} & known)
    ref = fit_fpca(curves, everyone, fcfg, f"{name}:reference")
    models = {"reference": ref}
    parts: dict[str, list[pl.DataFrame]] = {
        SCORES.name: [], SEQ_RECON.name: [], FOLD_FITS.name: [],
        FOLD_COMPONENTS.name: [_component_frame(name, -1, -1, ref, np.ones(ref.k_max))],
        FUNCTIONS.name: [_function_frame(name, -1, -1, ref)]}
    fits: list[dict[str, Any]] = []
    for r in sorted(splits["repeat"].unique().to_list()):
        for f in sorted(splits.filter(pl.col("repeat") == r)["outer_fold"].unique().to_list()):
            train, test = fold_athletes(splits, r, f)
            m, cong = align_to_reference(fit_fpca(curves, train, fcfg, f"{name}:{fold_tag(r, f)}"), ref)
            m.assert_unseen(test, f"{name}, repeat {r}, outer fold {f}: FPCA fit vs held-out athletes")
            models[fold_tag(r, f)] = m
            te, tr = np.isin(curves.nfl_ids, test), np.isin(curves.nfl_ids, train)
            ids = pl.lit(name).alias("representation"), pl.lit(r).alias("repeat"), pl.lit(f).alias("outer_fold")
            parts[FOLD_COMPONENTS.name].append(_component_frame(name, r, f, m, cong))
            parts[FUNCTIONS.name].append(_function_frame(name, r, f, m))
            seq_te, ath_te = curves.seq_ids[te].astype(str), curves.nfl_ids[te]
            for k in (k for k in fcfg.k_grid if k <= m.k_max):
                Sc, frac = m.scores(curves.Y[te], k)
                res = m.residuals(curves.Y[te], Sc, k)
                ok = np.isfinite(Sc[:, 0])
                n_ok = int(ok.sum())
                parts[SCORES.name].append(pl.DataFrame({
                    "seq_id": np.repeat(seq_te[ok], k), "nfl_id": np.repeat(ath_te[ok], k),
                    "component": np.tile(np.arange(1, k + 1), n_ok), "score": Sc[ok].ravel(),
                    "score_z": (Sc[ok] / np.sqrt(np.maximum(m.eigenvalues[:k], 1e-300))).ravel(),
                    "observed_fraction": np.repeat(frac[ok], k)}).with_columns(*ids, pl.lit(k).alias("k")))
                with np.errstate(invalid="ignore", divide="ignore"):
                    rmse = np.sqrt(res["sse"] / np.maximum(res["n_points"], 1))
                parts[SEQ_RECON.name].append(pl.DataFrame({
                    "seq_id": seq_te, "nfl_id": ath_te, "scored": ok, "observed_fraction": frac,
                    "n_points": res["n_points"], "sse": res["sse"], "sst": res["sst"],
                    "rmse": np.where(ok, rmse, np.nan)}).with_columns(*ids, pl.lit(k).alias("k")))
                St, _ = m.scores(curves.Y[tr], k)
                fve_tr, rmse_tr = _fve(m.residuals(curves.Y[tr], St, k))
                fve_te, rmse_te = _fve(res)
                fits.append({"representation": name, "k": k, "repeat": r, "outer_fold": f,
                             "n_train_athletes": int(np.unique(curves.nfl_ids[tr]).size), "n_train_sequences": int(tr.sum()),
                             "n_test_athletes": int(np.unique(ath_te).size), "n_test_sequences": int(te.sum()),
                             "n_test_scored": n_ok, "support_lo": float(m.grid[0]), "support_hi": float(m.grid[-1]),
                             "n_support": int(m.support.sum()), "n_basis": m.n_basis, "sigma2": m.sigma2(k),
                             "fve_train": fve_tr, "rmse_train": rmse_tr, "fve_heldout": fve_te, "rmse_heldout": rmse_te})
    tables = {SCORES.name: _cast(pl.concat(parts[SCORES.name]), SCORES),
              SEQ_RECON.name: _cast(pl.concat(parts[SEQ_RECON.name]), SEQ_RECON), FOLD_FITS.name: _table(fits, FOLD_FITS),
              FOLD_COMPONENTS.name: _cast(pl.concat(parts[FOLD_COMPONENTS.name]), FOLD_COMPONENTS),
              FUNCTIONS.name: _cast(pl.concat(parts[FUNCTIONS.name]), FUNCTIONS)}
    return FoldRun(curves, models, tables)


def fit_all(curves: dict[str, CurveSet], splits: pl.DataFrame, fcfg: FpcaConfig) -> dict[str, FoldRun]:
    runs = {}
    for name, cs in curves.items():
        runs[name] = fit_representation(cs, splits, fcfg)
        log.info("[advanced_movement:FPCA] %s: %d fits, support %d of %d grid points", name, len(runs[name].models),
                 int(runs[name].models["reference"].support.sum()), cs.grid.size)
    return runs


def fit_tables(runs: dict[str, FoldRun]) -> dict[str, pl.DataFrame]:
    return {c.name: pl.concat([r.tables[c.name] for r in runs.values()]) for c in FIT_CONTRACTS}


def model_path(models_dir: Path, name: str) -> Path:
    return models_dir / "fpca" / f"{name}.npz"


def fit_key(params: dict[str, Any], files: Sequence[Path]) -> str:
    """Identity of a fit stage: its parameters and the state of its input files."""
    return stable_hash({"params": params, "inputs": [file_fingerprint(f) if Path(f).is_file() else "missing" for f in files]})


def save_runs(models_dir: Path, runs: dict[str, FoldRun], key: str = "") -> list[Path]:
    """One ``.npz`` per representation: the reference fit and every fold fit (support, coverage, mean, knots,
    noise inputs, eigenfunctions, eigenvalues, training athletes), plus ``_models.json`` holding ``key``."""
    paths = [save_models(model_path(models_dir, name), run.models) for name, run in runs.items()]
    atomic_write_json({"key": key, "representations": sorted(runs), "saved_at": utc_now()},
                      models_dir / "fpca" / "_models.json")
    return paths


def load_or_refit(models_dir: Path, curves: dict[str, CurveSet], splits: pl.DataFrame, fcfg: FpcaConfig, key: str = ""
                  ) -> dict[str, dict[str, FpcaModel]]:
    """Fitted models of every representation: from disk when every file exists AND was saved under the same
    ``key``; otherwise refitted in memory (the fit is deterministic, so a refit equals what the cached tables
    were computed from). A stale model file is never used."""
    paths = {name: model_path(models_dir, name) for name in curves}
    saved = read_json(models_dir / "fpca" / "_models.json", {})
    if all(p.is_file() for p in paths.values()) and saved.get("key") == key:
        return {name: load_models(p) for name, p in paths.items()}
    log.info("[advanced_movement:FPCA] no fitted models on disk for this configuration: refitting in memory")
    return {name: run.models for name, run in fit_all(curves, splits, fcfg).items()}


# ============================================================================== evaluation: descriptive tables
def athlete_score_table(scores: pl.DataFrame) -> pl.DataFrame:
    out = scores.group_by("representation", "k", "repeat", "nfl_id", "component").agg(
        pl.len().alias("n_attempts"), pl.col("score").mean().alias("score_mean"), pl.col("score_z").mean().alias("score_z_mean"))
    return _cast(out, ATHLETE_SCORES).sort("representation", "k", "repeat", "nfl_id", "component")


def variance_table(components: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (rep,), d in sorted(components.partition_by(["representation"], as_dict=True).items()):
        folds, ref = d.filter(pl.col("repeat") >= 0), d.filter(pl.col("repeat") < 0)
        cum = folds.sort("repeat", "outer_fold", "component").with_columns(
            pl.col("explained_ratio").cum_sum().over("repeat", "outer_fold").alias("_cum"))
        for c in sorted(d["component"].unique().to_list()):
            x, r = cum.filter(pl.col("component") == c), ref.filter(pl.col("component") == c)
            rows.append({"representation": rep, "component": c, "n_fits": x.height,
                         "explained_mean": x["explained_ratio"].mean(), "explained_min": x["explained_ratio"].min(),
                         "explained_max": x["explained_ratio"].max(), "cumulative_mean": x["_cum"].mean(),
                         "explained_reference": r["explained_ratio"][0] if r.height else None,
                         "congruence_mean": x["congruence"].mean(), "congruence_min": x["congruence"].min(),
                         "n_reordered": int((x["source_rank"] != c).sum())})
    return _table(rows, VARIANCE)


def reconstruction_table(recon: pl.DataFrame, fits: pl.DataFrame, fcfg: FpcaConfig, seed: int) -> pl.DataFrame:
    rows = []
    for (rep, k, repeat), d in sorted(recon.partition_by(["representation", "k", "repeat"], as_dict=True).items()):
        ok = d.filter(pl.col("scored"))
        if not ok.height:
            rows.append({"representation": rep, "k": k, "repeat": repeat, "n_athletes": 0, "n_sequences": 0,
                         "n_unscored": d.height, "value_unit": SPEC[rep].value_unit if rep in SPEC else ""})
            continue
        ids, sums, _ = athlete_means(ok["nfl_id"].to_numpy(), np.column_stack(
            [ok["sse"].to_numpy(), ok["sst"].to_numpy(), ok["n_points"].to_numpy().astype(float)]))
        _, counts = np.unique(ok["nfl_id"].to_numpy(), return_counts=True)
        tot = sums * counts[:, None]                       # per-athlete sums (means × number of attempts)
        est_fve = 1.0 - tot[:, 0].sum() / tot[:, 1].sum() if tot[:, 1].sum() > 0 else NAN
        est_rmse = float(np.sqrt(tot[:, 0].sum() / tot[:, 2].sum())) if tot[:, 2].sum() > 0 else NAN
        lo_f = hi_f = lo_r = hi_r = NAN
        if ids.size >= fcfg.min_athletes:
            pick = np.random.default_rng(seed_for(seed, "recon", rep, k, repeat)).integers(0, ids.size, (fcfg.n_boot, ids.size))
            b = tot[pick].sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                lo_f, hi_f = np.nanquantile(1.0 - b[:, 0] / b[:, 1], [0.025, 0.975])
                lo_r, hi_r = np.nanquantile(np.sqrt(b[:, 0] / b[:, 2]), [0.025, 0.975])
        ff = fits.filter((pl.col("representation") == rep) & (pl.col("k") == k) & (pl.col("repeat") == repeat))
        rows.append({"representation": rep, "k": k, "repeat": repeat, "n_athletes": int(ids.size), "n_sequences": ok.height,
                     "n_unscored": d.height - ok.height, "rmse_heldout": est_rmse, "rmse_lower": lo_r, "rmse_upper": hi_r,
                     "fve_heldout": est_fve, "fve_lower": lo_f, "fve_upper": hi_f, "fve_train": ff["fve_train"].mean(),
                     "fve_fold_min": ff["fve_heldout"].min(), "fve_fold_max": ff["fve_heldout"].max(),
                     "value_unit": SPEC[rep].value_unit if rep in SPEC else ""})
    return _table(rows, RECON)


# ============================================================================== evaluation: reliability
def _icc(values: np.ndarray, athletes: np.ndarray, strata: np.ndarray, n_boot: int, seed: int, min_athletes: int
         ) -> dict[str, Any]:
    ok = np.isfinite(values)
    _, counts = np.unique(athletes[ok], return_counts=True)
    out = {"n_athletes_repeat": int((counts >= 2).sum()), "n_sequences": int(ok.sum()), "n_strata": None, "icc": None,
           "ci_lower": None, "ci_upper": None, "var_between": None, "var_within": None, "typical_error": None,
           "reliability_of_mean": None}
    if out["n_athletes_repeat"] < min_athletes:
        return out
    r = stratified_icc(values[ok], athletes[ok], strata[ok], n_boot, seed)
    te = float(np.sqrt(r["var_within"])) if np.isfinite(r["var_within"]) else NAN
    out.update(n_strata=r["n_strata"], icc=r["icc"], ci_lower=r["ci_lower"], ci_upper=r["ci_upper"],
               var_between=r["var_between"], var_within=r["var_within"], typical_error=te,
               reliability_of_mean=spearman_brown(r["icc"], r["k_mean"]))
    return out


def reliability_scopes(b: Block, meta: Meta) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(scope, rows kept, stratum of each row). The outer fold is part of every stratum."""
    i = meta.rows(b.seq_ids)
    pos, coh, fold = meta.position[i], meta.cohort[i], b.fold.astype(str)
    everyone = np.ones(b.seq_ids.size, dtype=bool)
    scopes = [("pooled", everyone, fold), ("position_adjusted", everyone, _join(pos, fold)),
              ("cohort_adjusted", everyone, _join(coh, fold)), ("position_cohort_adjusted", everyone, _join(pos, coh, fold))]
    scopes += [(f"position_adjusted@{s}", meta.scenario[s][i], _join(pos, fold)) for s in meta.scenario]
    scopes += [(f"position:{p}", pos == p, fold) for p in sorted(set(pos.tolist()))]
    scopes += [(f"cohort:{y}", coh == y, _join(pos, fold)) for y in sorted(set(coh.tolist()))]
    return scopes


def reliability_table(scores: pl.DataFrame, meta: Meta, fcfg: FpcaConfig, seed: int) -> pl.DataFrame:
    rows = []
    for b in blocks(scores):
        for scope, keep, strata in reliability_scopes(b, meta):
            for c in range(b.k):
                rows.append({"representation": b.representation, "k": b.k, "repeat": b.repeat, "component": c + 1,
                             "scope": scope, **_icc(b.S[keep, c], b.nfl_ids[keep], strata[keep],
                                                    fcfg.n_boot if b.repeat == 0 else 0,
                                                    seed_for(seed, "icc", b.representation, b.k, c, scope),
                                                    fcfg.min_athletes)})
    return _table(rows, RELIABILITY)


# ============================================================================== evaluation: association
def association_table(scores: pl.DataFrame, meta: Meta, fcfg: FpcaConfig, seed: int, targets: Sequence[str],
                      contract: TableContract) -> pl.DataFrame:
    """Athlete-level association of each score with ``targets`` one by one and jointly (``a+b``)."""
    joint = "+".join(targets)
    rows = []
    for b in blocks(scores):
        ids, mean, _ = athlete_means(b.nfl_ids, b.S)
        a = meta.athlete_rows(ids)
        pos = meta.athlete_position[a]
        X = {t: meta.official[t][a] for t in targets}
        for c in range(b.k):
            base = {"representation": b.representation, "k": b.k, "repeat": b.repeat, "component": c + 1}
            for scope, cats in (("pooled", []), ("position_adjusted", [pos])):
                for t in targets:
                    r = partial_corr(mean[:, c], X[t], cats, (), "spearman")
                    rows.append({**base, "target": t, "scope": scope, "n_athletes": r["n"], "spearman": r["r"],
                                 "spearman_lower": r["ci_lower"], "spearman_upper": r["ci_upper"], "r2": None,
                                 "r2_lower": None, "r2_upper": None})
                boot = fcfg.n_boot_r2 if b.repeat == 0 and scope == "position_adjusted" else 0
                r2, lo, hi, n = _r2_ci(mean[:, c], [X[t] for t in targets], cats, boot,
                                       seed_for(seed, contract.name, b.representation, b.k, c, scope))
                rows.append({**base, "target": joint, "scope": scope, "n_athletes": n, "spearman": None, "spearman_lower": None,
                             "spearman_upper": None, "r2": r2, "r2_lower": lo, "r2_upper": hi})
    return _table(rows, contract)


def window_table(scores: pl.DataFrame, meta: Meta, fcfg: FpcaConfig, seed: int) -> pl.DataFrame:
    rows = []
    names = [*WINDOW_VARS, *FLAG_VARS, "observed_fraction"]
    for b in (b for b in blocks(scores) if b.repeat == 0):
        i = meta.rows(b.seq_ids)
        W = np.column_stack([*[meta.window[v][i] for v in (*WINDOW_VARS, *FLAG_VARS)], b.frac])
        ids, mean, _ = athlete_means(b.nfl_ids, np.column_stack([b.S, W]))
        a = meta.athlete_rows(ids)
        pos, coh = meta.athlete_position[a], meta.athlete_cohort[a]
        dummies = [(coh == y).astype(float) for y in sorted(set(coh.tolist()))[1:]]
        two, i1, i2 = first_two(b.nfl_ids, meta.order[i])
        dS, dW = b.S[i2] - b.S[i1], W[i2] - W[i1]
        for c in range(b.k):
            base = {"representation": b.representation, "k": b.k, "repeat": 0, "component": c + 1}
            for j, v in enumerate(names):
                bt = partial_corr(mean[:, c], mean[:, b.k + j], [pos], (), "spearman")
                wi = partial_corr(dS[:, c], dW[:, j], (), (), "spearman")
                for level, r in (("between_athletes", bt), ("within_athlete", wi)):
                    rows.append({**base, "variable": v, "level": level, "n_athletes": r["n"], "spearman": r["r"],
                                 "spearman_lower": r["ci_lower"], "spearman_upper": r["ci_upper"], "r2": None,
                                 "r2_lower": None, "r2_upper": None})
            cols = [names.index(v) for v in JOINT_WINDOW]
            r2, lo, hi, n = _r2_ci(dS[:, c], [dW[:, j] for j in cols], [], fcfg.n_boot_r2,
                                   seed_for(seed, "window", b.representation, b.k, c))
            rows.append({**base, "variable": "window_joint", "level": "within_athlete", "n_athletes": n, "spearman": None,
                         "spearman_lower": None, "spearman_upper": None, "r2": r2, "r2_lower": lo, "r2_upper": hi})
            r2, lo, hi, n = _r2_ci(mean[:, c], dummies, [pos], fcfg.n_boot_r2, seed_for(seed, "cohort", b.representation, b.k, c))
            rows.append({**base, "variable": "draft_cohort", "level": "between_athletes", "n_athletes": n, "spearman": None,
                         "spearman_lower": None, "spearman_upper": None, "r2": r2, "r2_lower": lo, "r2_upper": hi})
    return _table(rows, WINDOW)


def sensitivity_table(scores: pl.DataFrame, meta: Meta, fcfg: FpcaConfig, seed: int) -> pl.DataFrame:
    by = {(b.representation, b.k, b.repeat): b for b in blocks(scores)}
    rows = []
    for alt in (s for s in REPRESENTATIONS if s.alternate_of):
        for (rep, k, repeat), main in by.items():
            other = by.get((alt.name, k, repeat))
            if rep != alt.alternate_of or other is None:
                continue
            ids_m, mean_m, _ = athlete_means(main.nfl_ids, main.S)
            ids_a, mean_a, _ = athlete_means(other.nfl_ids, other.S)
            ids, im, ia = np.intersect1d(ids_m, ids_a, return_indices=True)
            ym, xa = mean_m[im], mean_a[ia]
            pos = meta.athlete_position[meta.athlete_rows(ids)]
            for c in range(k):
                each = [abs(partial_corr(ym[:, c], xa[:, j], (), (), "spearman")["r"]) for j in range(k)]
                best = int(np.nanargmax(each)) + 1 if np.isfinite(each).any() else None
                for scope, cats in (("pooled", []), ("position_adjusted", [pos])):
                    r2, lo, hi, n = _r2_ci(ym[:, c], [xa[:, j] for j in range(k)], cats, fcfg.n_boot_r2 if repeat == 0 else 0,
                                           seed_for(seed, "sensitivity", rep, k, c, scope))
                    rows.append({"representation": rep, "alternate": alt.name, "k": k, "repeat": repeat, "component": c + 1,
                                 "scope": scope, "n_athletes": n, "r2_from_alternate": r2, "r2_lower": lo, "r2_upper": hi,
                                 "best_alt_component": best,
                                 "best_abs_spearman": float(np.nanmax(each)) if best is not None else None})
    return _table(rows, SENSITIVITY)


def stability_table(scores: pl.DataFrame) -> pl.DataFrame:
    by = {(b.representation, b.k, b.repeat): b for b in blocks(scores)}
    rows = []
    for (rep, k, ra), a in by.items():
        for (rep2, k2, rb), b in by.items():
            if (rep2, k2) != (rep, k) or rb <= ra:
                continue
            ids_a, ma, _ = athlete_means(a.nfl_ids, a.S)
            ids_b, mb, _ = athlete_means(b.nfl_ids, b.S)
            ids, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
            for c in range(k):
                x, y = ma[ia, c], mb[ib, c]
                ok = np.isfinite(x) & np.isfinite(y)
                r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 3 and x[ok].std() > 0 and y[ok].std() > 0 else NAN
                rows.append({"representation": rep, "k": k, "component": c + 1, "repeat_a": ra, "repeat_b": rb,
                             "n_athletes": int(ok.sum()), "pearson": r})
    return _table(rows, STABILITY)


def evaluate(fit: dict[str, pl.DataFrame], windows: pl.DataFrame, athletes: pl.DataFrame, fcfg: FpcaConfig, seed: int
             ) -> dict[str, pl.DataFrame]:
    """Every evaluation table from the fit-stage tables. Uses Combine measurements only."""
    scores, meta = fit[SCORES.name], Meta(windows, athletes)
    return {ATHLETE_SCORES.name: athlete_score_table(scores), VARIANCE.name: variance_table(fit[FOLD_COMPONENTS.name]),
            RECON.name: reconstruction_table(fit[SEQ_RECON.name], fit[FOLD_FITS.name], fcfg, seed),
            RELIABILITY.name: reliability_table(scores, meta, fcfg, seed),
            REDUNDANCY.name: association_table(scores, meta, fcfg, seed, (FORTY, TEN), REDUNDANCY),
            BODY.name: association_table(scores, meta, fcfg, seed, (WEIGHT, HEIGHT), BODY),
            WINDOW.name: window_table(scores, meta, fcfg, seed),
            SENSITIVITY.name: sensitivity_table(scores, meta, fcfg, seed), STABILITY.name: stability_table(scores)}


# ============================================================================== research gate
def select_k(recon: pl.DataFrame, fcfg: FpcaConfig) -> dict[str, tuple[int, str]]:
    """representation → (benchmark K, rule). Smallest K of the grid whose held-out share of variance, averaged over
    split repeats, reaches ``fve_target``; the largest K when none does. No outcome and no reliability is used."""
    out = {}
    for (rep,), d in recon.partition_by(["representation"], as_dict=True).items():
        m = d.group_by("k").agg(pl.col("fve_heldout").mean()).sort("k")
        hit = m.filter(pl.col("fve_heldout") >= fcfg.fve_target)
        if hit.height:
            out[rep] = (int(hit["k"][0]), f"smallest K with held-out FVE ≥ {fcfg.fve_target:g}")
        else:
            out[rep] = (int(m["k"][-1]), f"largest K (held-out FVE stays below {fcfg.fve_target:g})")
    return out


def gate_labels(icc_lo: float | None, icc_hi: float | None, red_lo: float | None, red_hi: float | None,
                win_lo: float | None, win_hi: float | None, fcfg: FpcaConfig) -> tuple[str, str, str, str]:
    """(reproducible, stopwatch, window, status) from 95 % intervals against the heuristic reference lines.
    A label is 'uncertain' whenever its interval straddles the line or is missing."""
    def side(lo, hi, line, below, above):
        if lo is None or hi is None:
            return "uncertain"
        return above if lo >= line else below if hi < line else "uncertain"

    rep = side(icc_lo, icc_hi, fcfg.icc_floor, "no", "yes")
    red = side(red_lo, red_hi, fcfg.redundancy_reference, "not_redundant", "redundant")
    win = side(win_lo, win_hi, fcfg.window_reference, "not_window_driven", "window_driven")
    if rep == "no":
        status = "not_repeatable"
    elif win == "window_driven":
        status = "window_artifact"
    elif red == "redundant":
        status = "redundant_with_stopwatch"
    elif rep == "yes" and red == "not_redundant" and win == "not_window_driven":
        status = "candidate"
    else:
        status = "inconclusive"
    return rep, red, win, status


def _one(df: pl.DataFrame, **eq: Any) -> dict[str, Any]:
    for col, value in eq.items():
        df = df.filter(pl.col(col) == value)
    return df.row(0, named=True) if df.height else {}


def gate_tables(ev: dict[str, pl.DataFrame], fit: dict[str, pl.DataFrame], windows: pl.DataFrame, athletes: pl.DataFrame,
                fcfg: FpcaConfig, seed: int) -> dict[str, pl.DataFrame]:
    """Component gate and representation summary at each representation's benchmark K (split repeat 0 for intervals)."""
    rel, red, body, win, sens, var, recon = (ev[c.name] for c in (RELIABILITY, REDUNDANCY, BODY, WINDOW, SENSITIVITY, VARIANCE, RECON))
    chosen, meta = select_k(recon, fcfg), Meta(windows, athletes)
    by = {(b.representation, b.k, b.repeat): b for b in blocks(fit[SCORES.name])}
    gate, summary = [], []
    for spec in REPRESENTATIONS:
        if spec.name not in chosen:
            continue
        k, rule = chosen[spec.name]
        at = {"representation": spec.name, "k": k}
        comp_rows = []
        for c in range(1, k + 1):
            adj = _one(rel, **at, repeat=0, component=c, scope="position_adjusted")
            every = rel.filter((pl.col("representation") == spec.name) & (pl.col("k") == k) & (pl.col("component") == c))
            coh = every.filter((pl.col("repeat") == 0) & pl.col("scope").str.starts_with("cohort:") & pl.col("icc").is_not_null())
            reps = every.filter((pl.col("scope") == "position_adjusted") & pl.col("icc").is_not_null())
            f40 = _one(red, **at, repeat=0, component=c, target=FORTY, scope="position_adjusted")
            t10 = _one(red, **at, repeat=0, component=c, target=TEN, scope="position_adjusted")
            both = _one(red, **at, repeat=0, component=c, target=f"{FORTY}+{TEN}", scope="position_adjusted")
            wj = _one(win, **at, component=c, variable="window_joint", level="within_athlete")
            top = max((r for r in (f40, t10) if r.get("spearman") is not None), key=lambda r: abs(r["spearman"]), default={})
            lo, hi = top.get("spearman_lower"), top.get("spearman_upper")
            red_lo, red_hi = _abs_interval(lo, hi) if lo is not None and hi is not None else (None, None)
            relm, r2 = adj.get("reliability_of_mean"), both.get("r2")
            v = _one(var, representation=spec.name, component=c)
            labels = gate_labels(adj.get("ci_lower"), adj.get("ci_upper"), red_lo, red_hi, wj.get("r2_lower"),
                                 wj.get("r2_upper"), fcfg)
            comp_rows.append({
                "representation": spec.name, "role": spec.role, "k": k, "component": c,
                "explained_ratio": v.get("explained_mean"), "congruence_min": v.get("congruence_min"),
                "n_athletes_repeat": adj.get("n_athletes_repeat"), "icc_adj": adj.get("icc"), "icc_lower": adj.get("ci_lower"),
                "icc_upper": adj.get("ci_upper"),
                "icc_pooled": _one(rel, **at, repeat=0, component=c, scope="pooled").get("icc"),
                "icc_cohort_min": coh["icc"].min() if coh.height else None,
                "icc_s3": _one(rel, **at, repeat=0, component=c, scope="position_adjusted@S3_no_boundary").get("icc"),
                "icc_repeat_min": reps["icc"].min() if reps.height else None,
                "icc_repeat_max": reps["icc"].max() if reps.height else None, "reliability_of_mean": relm,
                "rho_forty": f40.get("spearman"), "rho_forty_lower": f40.get("spearman_lower"),
                "rho_forty_upper": f40.get("spearman_upper"), "rho_ten": t10.get("spearman"),
                "redundancy_abs": abs(top["spearman"]) if top else None, "redundancy_lower": red_lo, "redundancy_upper": red_hi,
                "r2_stopwatch": r2, "r2_stopwatch_lower": both.get("r2_lower"), "r2_stopwatch_upper": both.get("r2_upper"),
                "reliable_unique_share": max(relm - r2, 0.0) if relm is not None and r2 is not None else None,
                "r2_body": _one(body, **at, repeat=0, component=c, target=f"{WEIGHT}+{HEIGHT}", scope="position_adjusted").get("r2"),
                "window_r2_within": wj.get("r2"), "window_r2_lower": wj.get("r2_lower"), "window_r2_upper": wj.get("r2_upper"),
                "cohort_r2": _one(win, **at, component=c, variable="draft_cohort", level="between_athletes").get("r2"),
                "r2_from_alternate": _one(sens, **at, repeat=0, component=c, scope="position_adjusted").get("r2_from_alternate"),
                "reproducible": labels[0], "stopwatch": labels[1], "window": labels[2], "status": labels[3]})
        gate += comp_rows
        rc = _one(recon, **at, repeat=0)
        b = by.get((spec.name, k, 0))
        r2f = lo_f = hi_f = None
        if b is not None:
            ids, mean, _ = athlete_means(b.nfl_ids, b.S)
            a = meta.athlete_rows(ids)
            r2f, lo_f, hi_f, _ = _r2_ci(meta.official[FORTY][a], [mean[:, j] for j in range(k)], [meta.athlete_position[a]],
                                        fcfg.n_boot_r2, seed_for(seed, "forty_from_scores", spec.name, k))
        iccs = [(r["icc_adj"], r["component"]) for r in comp_rows if r["icc_adj"] is not None]
        uniq = [r["reliable_unique_share"] for r in comp_rows if r["reliable_unique_share"] is not None]
        cong = [r["congruence_min"] for r in comp_rows if r["congruence_min"] is not None]
        summary.append({
            "representation": spec.name, "role": spec.role, "alternate_of": spec.alternate_of, "k_selected": k, "k_rule": rule,
            "fve_heldout": rc.get("fve_heldout"), "fve_lower": rc.get("fve_lower"), "fve_upper": rc.get("fve_upper"),
            "rmse_heldout": rc.get("rmse_heldout"), "value_unit": spec.value_unit, "n_athletes_scored": rc.get("n_athletes"),
            "n_athletes_repeat": max((r["n_athletes_repeat"] or 0 for r in comp_rows), default=0),
            "n_components_reproducible": sum(r["reproducible"] == "yes" for r in comp_rows),
            "n_components_candidate": sum(r["status"] == "candidate" for r in comp_rows),
            "max_icc_adj": max(iccs)[0] if iccs else None, "max_icc_component": max(iccs)[1] if iccs else None,
            "r2_forty_from_scores": r2f, "r2_forty_lower": lo_f, "r2_forty_upper": hi_f,
            "best_reliable_unique_share": max(uniq) if uniq else None, "congruence_min": min(cong) if cong else None,
            "proposed_benchmark": False})
    main = [r for r in summary if r["role"] != "alternate" and r["n_components_reproducible"] > 0]
    if main:
        best = min(main, key=lambda r: (-r["n_components_candidate"], -r["n_components_reproducible"], ROLE_ORDER[r["role"]]))
        best["proposed_benchmark"] = True
    return {GATE.name: _table(gate, GATE), SUMMARY.name: _table(summary, SUMMARY)}


# ============================================================================== conclusions from executed metrics
def _f(x: Any, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def conclusions_markdown(summary: pl.DataFrame, gate: pl.DataFrame, fcfg: FpcaConfig, run_mode: str) -> str:
    """Text of the conclusions cell, written from the saved tables only (nothing is asserted from expectation)."""
    lines = [f"_Generated from `fpca_representation_summary` and `fpca_component_gate` of this run (`RUN_MODE = "
             f"{run_mode!r}`). Intervals are 95 %, athlete-level. Labels are heuristic screening labels._", ""]
    if run_mode == "smoke":
        lines += ["**Smoke run: the numbers below check the wiring only and must not be interpreted.**", ""]
    main = summary.filter(pl.col("role") != "alternate").sort("representation")
    for r in main.iter_rows(named=True):
        g = gate.filter(pl.col("representation") == r["representation"]).sort("component")
        lines.append(f"**{SPEC[r['representation']].label}** — K = {r['k_selected']} ({r['k_rule']}); held-out share of "
                     f"variance {_f(r['fve_heldout'], 3)} [{_f(r['fve_lower'], 3)}, {_f(r['fve_upper'], 3)}], RMSE "
                     f"{_f(r['rmse_heldout'], 3)} {r['value_unit']}; {r['n_athletes_scored']} athletes scored, "
                     f"{r['n_athletes_repeat']} with a repeat.")
        for c in g.iter_rows(named=True):
            lines.append(
                f"- Component {c['component']} ({_f(100 * c['explained_ratio'], 1) if c['explained_ratio'] is not None else 'n/a'} % "
                f"of variance): within-position ICC {_f(c['icc_adj'])} [{_f(c['icc_lower'])}, {_f(c['icc_upper'])}]; "
                f"|ρ| with the official times {_f(c['redundancy_abs'])} [{_f(c['redundancy_lower'])}, {_f(c['redundancy_upper'])}]; "
                f"stopwatch R² {_f(c['r2_stopwatch'])}; attempt-to-attempt change explained by the recording window "
                f"{_f(c['window_r2_within'])} [{_f(c['window_r2_lower'])}, {_f(c['window_r2_upper'])}] → **{c['status']}**.")
        lines.append(f"- The K scores together explain {_f(r['r2_forty_from_scores'])} [{_f(r['r2_forty_lower'])}, "
                     f"{_f(r['r2_forty_upper'])}] of the official forty within position; largest reproducible share not "
                     f"explained by the stopwatch: {_f(r['best_reliable_unique_share'])}.")
        lines.append("")
    n_rep = int(main["n_components_reproducible"].sum()) if main.height else 0
    n_cand = int(main["n_components_candidate"].sum()) if main.height else 0
    statuses = gate.filter(pl.col("role") != "alternate")["status"].to_list()
    lines += ["**Research gate**",
              f"1. *Reproducible athlete-level information?* {n_rep} component(s) across the three representations have a "
              f"within-position ICC interval entirely above {fcfg.icc_floor:g}."
              + (" None does: no representation shows reproducible information at this sample size." if n_rep == 0 else ""),
              f"2. *Largely explained by the official forty?* {statuses.count('redundant_with_stopwatch')} component(s) are "
              f"labelled redundant with the stopwatch (|ρ| interval above {fcfg.redundancy_reference:g}); {n_cand} are "
              "reproducible, not redundant and not window-driven.",
              f"3. *Explained by noise and censoring?* {statuses.count('window_artifact')} component(s) are labelled a "
              f"recording-window artifact (window share interval above {fcfg.window_reference:g}); "
              f"{statuses.count('not_repeatable')} are not repeatable; {statuses.count('inconclusive')} are inconclusive."]
    pick = summary.filter(pl.col("proposed_benchmark"))
    if pick.height:
        p = pick.row(0, named=True)
        lines.append(f"4. *Low-complexity benchmark for TCN and GRU/LSTM?* Rule-based proposal: **{SPEC[p['representation']].label}** "
                     f"with K = {p['k_selected']} (most candidate components, then most reproducible components, then the "
                     "pre-declared order primary → secondary → control). Review before adopting it.")
    else:
        lines.append("4. *Low-complexity benchmark for TCN and GRU/LSTM?* No representation has a reproducible component, so "
                     "the rule proposes none. The official forty and 10-yard split remain the benchmark to beat.")
    lines += ["", "FPCA is not assumed to beat the official Combine results, and nothing here uses an NFL outcome."]
    return "\n".join(lines)
