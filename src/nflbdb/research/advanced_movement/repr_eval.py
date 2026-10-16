"""Model-agnostic evaluation of attempt embeddings (notebook 03). NumPy only; no NFL value is read here.

An EMBEDDING is a vector per Combine attempt: the K functional-PCA scores of notebook 02, or the output of a
self-supervised encoder (``tcn``). Every method goes through the same functions, on the same held-out attempts.

Why nothing here compares raw coordinates between fits
    Each outer fold has its own encoder, and the axes of two independently trained encoders are not the same axes.
    Every statistic is therefore computed INSIDE one fold (distances, ranks, regressions) and only then pooled over
    folds, or is invariant to a rotation of the embedding (trace statistics). ``space`` says which metric is used:
    ``standardised`` = each dimension divided by its SD over the fold's TRAINING athletes · ``raw`` = as produced.

Repeatability      attempt 1 against attempt 2 of the held-out athletes of a fold. For athlete i the gallery is
                   the other attempt of every athlete of the same position in the fold (``within_position``).
                       auc   = share of gallery athletes farther from i's attempt than i's own other attempt
                       top1  = own other attempt is the nearest · rr = 1 / rank of the own attempt
                   Values are per athlete; intervals resample ATHLETES (never attempts, pairs or frames).
Trace reliability  Σ_d σ²_between / Σ_d (σ²_between + σ²_within): the stratified ICC(1) of the sprint rescue summed
                   over dimensions (stratum = position × fold), invariant to rotations of the raw embedding.
Probes             ridge regressions from the embedding to official results, body size, position, cohort and
                   recording-window variables, always fitted on athletes other than those they are scored on.
Residual           each embedding dimension minus its cross-fitted linear prediction from official results (and,
                   optionally, the recording window). The residual goes through the SAME repeatability functions:
                   low stopwatch correlation is not evidence of information; a repeatable residual is.
Window             share of the attempt-to-attempt change of the embedding explained by the change of the recording
                   window (an athlete does not change between two attempts; his recording does), and the shift of
                   the embedding when the visible window is truncated on purpose.
Verdict            rule-based reading of the paired intervals against the FPCA reference of the same input — a
                   screening label for review, not a decision.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from ..sprint_rescue.stats import athlete_stats, bootstrap_picks, icc_core, seed_for
from ..studies.stats import rankdata
from .config import NotebookControls
from .fpca import SPEC, CurveSet, FpcaModel
from .fpca_eval import FORTY, HEIGHT, JOINT_WINDOW, TEN, WEIGHT, Meta, _cast, _join, _one
from .fpca_eval import _table as _typed_table
from .fpca_eval import athlete_means, first_two

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
SPACES = ("standardised", "raw")
SCOPES = ("within_position", "pooled", "within_position_cohort")
PAIR_METRICS = ("auc", "top1", "topk", "rr")
EVAL_MASK = {"ratio": 0.30, "n_spans": 2, "p_edge": 0.35, "min_visible": 4}     # the fixed held-out masking task
RESIDUALS = ("official", "official_window")
ATHLETE_TARGETS = ((FORTY, "continuous"), (TEN, "continuous"), (WEIGHT, "continuous"), (HEIGHT, "continuous"),
                   ("position_group", "categorical"), ("draft_cohort", "categorical"))
ATTEMPT_TARGETS = (*[(v, "continuous") for v in ("first_speed", "duration_s", "frames_after_peak", "path_yd",
                                                  "pos_resid_rms", "a_tan_roughness", "observed_fraction")],
                   *[(v, "binary") for v in ("first_frame_censored", "start_above_ref_speed", "last_frame_censored",
                                             "ends_at_peak", "boundary_peak")])


@dataclass(frozen=True)
class EvalConfig:
    version: str = "1"
    top_k: int = 5                                # retrieval: own attempt among the k nearest
    ridge_alpha: float = 1.0                      # proposed, fixed (features are standardised); not tuned
    probe_folds: int = 5                          # athlete-grouped folds of the cross-fitted probes
    truncations: tuple[tuple[str, float], ...] = (("head", 0.15), ("head", 0.30), ("tail", 0.15), ("tail", 0.30))
    near_identical: float = 0.01                  # pair distance below this share of the median distance
    collapse_rank_share: float = 0.25             # proposed: effective rank / dimension below this = collapsed
    collapse_identical_share: float = 0.05        # proposed: share of nearly identical pairs above this = collapsed
    margin_auc: float = 0.02                      # proposed: a gain below this is not a practical improvement
    nuisance_tolerance: float = 0.05              # proposed: allowed excess of window / cohort dependence over FPCA
    min_athletes: int = 10                        # established: RescueConfig.min_athletes (below → no interval)
    n_boot: int = 1000                            # execution
    n_boot_probe: int = 300                       # execution

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def for_mode(self, controls: NotebookControls) -> "EvalConfig":
        """Smoke mode keeps every scientific parameter; it only shortens the bootstraps."""
        return replace(self, n_boot=200, n_boot_probe=60) if controls.smoke else self


@dataclass(frozen=True)
class MethodSpec:
    method: str                    # identifier used in every table
    family: str                    # fpca · tcn
    objective: str                 # fpca · reconstruction · reconstruction+consistency
    representation: str            # fpca.REPRESENTATIONS name
    embedding_dim: int
    label: str
    reference: str                 # the FPCA method of the same input ("" for an FPCA method)
    primary: bool                  # False = sensitivity run


def fpca_method(representation: str, k: int) -> MethodSpec:
    return MethodSpec(f"FPCA_{representation[0]}_K{k}", "fpca", "fpca", representation, k,
                      f"FPCA · {representation[0]} (K = {k})", "", True)


def view_name(where: str, fraction: float) -> str:
    return f"{where}_{int(round(100 * fraction))}"


# ============================================================================== contracts
_M = {"method": S}
METHODS = TableContract(
    "repr_methods", "Methods compared in notebook 03: input, objective, size and cost.",
    {**_M, "family": S, "objective": S, "representation": S, "embedding_dim": I, "label": S, "reference": S, "primary": B,
     "n_parameters": I, "train_seconds": F, "value_unit": S}, key=("method",), area="method",
    docs={"n_parameters": "TCN: trainable weights of one fold model · FPCA: (K + 1) × basis functions (mean + K modes)",
          "train_seconds": "wall time of all fold fits of the method on this machine (null for FPCA: fitted in notebook 02)",
          "reference": "FPCA benchmark of the same input, at the K selected in notebook 02"})
EMBEDDINGS = TableContract(
    "repr_attempt_embeddings", "Attempt embeddings of every fold model (long form).",
    {**_M, "repeat": I, "outer_fold": I, "seq_id": S, "nfl_id": I, "role": S, "view": S, "dim": I, "value": F, "value_z": F},
    key=("method", "repeat", "outer_fold", "seq_id", "view", "dim"), kind="dataset",
    area="method × split repeat × outer fold × sequence × view × dimension",
    docs={"role": "heldout = the fold's model never saw the athlete (INDUCTIVE; the only rows a statistic of "
                  "repeatability uses) · fit / dev = training athletes of the fold (probe fitting, standardisation)",
          "view": "full = every recorded point visible · head_15 … tail_30 = that share of the recorded points hidden",
          "value_z": "value standardised with the fold's TRAINING athletes (FPCA: score / √ training eigenvalue)",
          "dim": "axes of different folds are NOT the same axes: never compare a dimension across folds"})
ATHLETE_EMB = TableContract(
    "repr_athlete_embeddings", "Athlete embeddings: mean of the athlete's held-out attempt embeddings (full view).",
    {**_M, "repeat": I, "outer_fold": I, "nfl_id": I, "dim": I, "n_attempts": I, "value_mean": F, "value_z_mean": F},
    key=("method", "repeat", "nfl_id", "dim"), kind="dataset", area="method × split repeat × athlete × dimension")
RECON_SEQ = TableContract(
    "repr_sequence_reconstruction", "Held-out reconstruction error of every attempt, in physical units.",
    {**_M, "repeat": I, "outer_fold": I, "seq_id": S, "nfl_id": I, "observed_fraction": F, "n_points": I, "sse": F,
     "sst": F, "n_masked": I, "sse_masked": F, "sst_masked": F}, key=("method", "repeat", "seq_id"), kind="dataset",
    area="method × split repeat × sequence",
    docs={"sse": "Σ (observed − reconstruction)² with every recorded point visible, on the evaluation support",
          "sse_masked": "same on the points hidden by the fixed evaluation mask, predicted from the visible ones",
          "sst": "Σ (observed − training mean)²", "observed_fraction": "share of the fold's support the attempt covers"})
RECON = TableContract(
    "repr_reconstruction", "Held-out reconstruction by method, with the paired difference from the FPCA reference.",
    {**_M, "reference": S, "value_unit": S, "n_athletes": I, "n_sequences": I, "rmse": F, "rmse_lower": F, "rmse_upper": F,
     "fve": F, "rmse_masked": F, "rmse_masked_lower": F, "rmse_masked_upper": F, "rmse_diff": F, "rmse_diff_lower": F,
     "rmse_diff_upper": F, "rmse_masked_diff": F, "rmse_masked_diff_lower": F, "rmse_masked_diff_upper": F},
    key=("method",), area="method",
    docs={"rmse": "all recorded points visible (FPCA: scores from all points) — the benchmark of notebook 02",
          "rmse_masked": "hidden points predicted from the visible ones; same mask for every method",
          "rmse_diff": "method − reference on the athletes both reconstruct; athlete bootstrap, 95 %"})
PAIRS = TableContract(
    "repr_pair_statistics", "Per-athlete repeat-attempt statistics of the held-out embeddings.",
    {**_M, "space": S, "scope": S, "repeat": I, "outer_fold": I, "nfl_id": I, "n_candidates": I, "auc": F, "top1": F,
     "topk": F, "rr": F, "d_same": F, "d_diff_median": F, "cos_same": F, "cos_diff_mean": F},
    key=("method", "space", "scope", "repeat", "nfl_id"), kind="dataset", area="method × space × scope × athlete",
    docs={"scope": "gallery = other athletes of the same position in the fold · of the fold · of the same position and cohort",
          "n_candidates": "gallery athletes other than the athlete himself",
          "auc": "share of the gallery farther than the athlete's own other attempt (both directions averaged)"})
REPEAT = TableContract(
    "repr_repeatability", "Repeat-attempt discrimination and retrieval, with athlete-bootstrap intervals.",
    {**_M, "space": S, "scope": S, "subset": S, "n_athletes": I, "gallery_mean": F, "chance_top1": F,
     **{f"{m}{s}": F for m in ("auc", "top1", "topk", "mrr") for s in ("", "_lower", "_upper")}, "distance_ratio": F,
     "cos_same": F, "cos_diff": F}, key=("method", "space", "scope", "subset"), area="method × space × scope × subset",
    docs={"subset": "all · role:confirm (no part in any choice — the confirmation read-out) · role:design · cohort:<year> · "
                    "position:<group>; the gallery is never restricted by the subset",
          "distance_ratio": "median same-athlete distance / median of the athletes' median different-athlete distance",
          "chance_top1": "mean of 1 / (gallery size + 1)"})
PAIRED = TableContract(
    "repr_paired_comparisons", "Paired difference of a method from its reference on the same athletes.",
    {**_M, "reference": S, "space": S, "scope": S, "subset": S, "metric": S, "n_athletes": I, "estimate": F,
     "reference_estimate": F, "difference": F, "diff_lower": F, "diff_upper": F},
    key=("method", "reference", "space", "scope", "subset", "metric"), area="method × reference × subset × metric")
ICC = TableContract(
    "repr_trace_reliability", "Rotation-invariant repeat-attempt reliability of the held-out embeddings.",
    {**_M, "space": S, "n_athletes_repeat": I, "n_sequences": I, "icc_trace": F, "icc_lower": F, "icc_upper": F,
     "icc_dim_min": F, "icc_dim_median": F, "icc_dim_max": F}, key=("method", "space"), area="method × space",
    docs={"icc_trace": "Σ σ²_between / Σ (σ²_between + σ²_within) over dimensions; stratum = position × fold",
          "icc_dim_max": "per-dimension ICC(1): depends on the rotation of the embedding, shown for orientation only"})
COLLAPSE = TableContract(
    "repr_collapse", "Collapse diagnostics of the held-out embeddings of every fold model.",
    {**_M, "repeat": I, "outer_fold": I, "n_sequences": I, "embedding_dim": I, "dim_sd_min": F, "dim_sd_max": F,
     "effective_rank": F, "rank_share": F, "participation_ratio": F, "top_eigen_share": F, "near_identical_share": F,
     "median_distance": F, "collapsed": B}, key=("method", "repeat", "outer_fold"), area="method × fold",
    docs={"effective_rank": "exp(entropy of the normalised covariance spectrum)", "rank_share": "effective rank / dimension",
          "near_identical_share": "pairs closer than 1 % of the median pair distance"})
SPECTRUM = TableContract(
    "repr_covariance_spectrum", "Covariance spectrum and per-dimension variance of the held-out embeddings.",
    {**_M, "repeat": I, "outer_fold": I, "index": I, "eigen_share": F, "dim_variance": F},
    key=("method", "repeat", "outer_fold", "index"), area="method × fold × index")
PROBES = TableContract(
    "repr_probes", "Held-out linear probes from the embedding to official results, body, position, cohort and window.",
    {**_M, "protocol": S, "level": S, "target": S, "metric": S, "n_athletes": I, "n_obs": I, "baseline": F, "value": F,
     "value_lower": F, "value_upper": F, "gain": F, "gain_lower": F, "gain_upper": F},
    key=("method", "protocol", "target"), area="method × protocol × target",
    docs={"protocol": "heldout_crossfit = fitted and scored on disjoint held-out athletes of a fold · train_to_heldout = "
                      "fitted on the fold's training athletes, scored on its held-out athletes",
          "baseline": "same probe with position dummies only (target position_group: the majority class)",
          "value": "r2 / auc / accuracy with position dummies + embedding (position_group: embedding only)",
          "gain": "r2: partial R² beyond position = 1 − SSE(position + embedding) / SSE(position) · otherwise value − baseline"})
WINDOW = TableContract(
    "repr_window_dependence", "Share of the attempt-to-attempt change of the embedding explained by the recording window.",
    {**_M, "space": S, "n_athletes": I, "r2_trace": F, "r2_lower": F, "r2_upper": F}, key=("method", "space"),
    area="method × space",
    docs={"r2_trace": "1 − Σ SSE / Σ SST over dimensions and folds of (attempt 2 − attempt 1) regressed on the change of "
                      "first-frame speed, recorded duration and frames after the peak; in-sample, as in notebook 02"})
TRUNC = TableContract(
    "repr_truncation_stability", "Stability of the held-out embeddings when part of the recording is hidden.",
    {**_M, "view": S, "n_athletes": I, "n_sequences": I, "relative_shift": F, "shift_lower": F, "shift_upper": F,
     "auc": F, "auc_lower": F, "auc_upper": F, "auc_full": F, "top1": F, "top1_full": F}, key=("method", "view"),
    area="method × view",
    docs={"relative_shift": "‖truncated − full‖ / median distance between different athletes of the fold (standardised)",
          "auc": "query = truncated attempt 1, gallery = full attempt 2, within position",
          "auc_full": "same athletes and direction, nothing hidden"})
SUMMARY = TableContract(
    "repr_method_summary", "One row per method: the evidence of notebook 03 side by side.",
    {**_M, "label": S, "family": S, "representation": S, "embedding_dim": I, "n_parameters": I, "train_seconds": F,
     "rmse": F, "rmse_masked": F, "value_unit": S, "n_confirm": I, "chance_top1": F,
     **{f"{m}{s}": F for m in ("auc", "top1", "resid_auc", "resid_window_auc") for s in ("", "_lower", "_upper")},
     "icc_trace": F, "forty_r2": F, "forty_partial_r2": F, "ten_partial_r2": F, "body_partial_r2": F, "window_r2": F,
     "window_r2_lower": F, "window_r2_upper": F, "cohort_gain": F, "cohort_gain_lower": F, "cohort_gain_upper": F,
     "position_accuracy": F, "truncation_shift_max": F, "rank_share_min": F, "collapsed": B}, key=("method",),
    area="method",
    docs={"auc": "within-position discrimination on confirm-role held-out athletes, standardised space",
          "resid_auc": "same after removing the cross-fitted linear prediction from official forty, 10-yard, weight, height",
          "resid_window_auc": "same, also removing the recording-window variables",
          "forty_partial_r2": "held-out share of the official forty the embedding explains beyond position"})
VERDICT = TableContract(
    "repr_verdict", "Rule-based go / no-go reading of each TCN method against its FPCA reference (for review).",
    {**_M, "reference": S, "verdict": S, "repeatability": S, "nonredundant": S, "nuisance": S, "reconstruction": S,
     "detail": S}, key=("method",), area="method",
    docs={"verdict": "A strong positive · B partial positive · C negative · D inconclusive — heuristic, for review"})
EVAL_CONTRACTS = (ATHLETE_EMB, PAIRS, REPEAT, PAIRED, ICC, COLLAPSE, SPECTRUM, PROBES, WINDOW, TRUNC, RECON)
GATE_CONTRACTS = (SUMMARY, VERDICT)


def _table(rows: list[dict[str, Any]], c: TableContract) -> pl.DataFrame:
    """Rows → typed frame. A key a row does not carry is null; a whole number in a float column is made a float."""
    def fix(k: str, v: Any) -> Any:
        whole = isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        return float(v) if whole and c.columns[k] == F else v
    return _typed_table([{k: fix(k, r.get(k)) for k in c.columns} for r in rows], c)


def named(contract: TableContract, prefix: str) -> TableContract:
    """The same contract under a stage-specific name (``repr_x`` → ``<prefix>_x``), so several stages can persist
    tables of one schema without overwriting each other."""
    return replace(contract, name=f"{prefix}_{contract.name.removeprefix('repr_').removeprefix('tcn_')}")


# ============================================================================== views (masks over recorded points)
def span_mask(obs: np.ndarray, rng: np.random.Generator, ratio: float = 0.30, n_spans: int = 2, p_edge: float = 0.35,
              min_visible: int = 4) -> np.ndarray:
    """(n, G) True on deliberately hidden points: about ``ratio`` of each row's RECORDED points, in ``n_spans``
    contiguous runs of the recorded sequence. With probability ``p_edge`` a run touches the first or the last
    recorded point (a shorter recording). Never True where ``obs`` is False; at least ``min_visible`` points stay."""
    out = np.zeros(obs.shape, dtype=bool)
    for i in range(obs.shape[0]):
        idx = np.flatnonzero(obs[i])
        n = idx.size
        m = min(int(round(ratio * n)), n - min_visible)
        left = m
        for s in range(n_spans):
            length = left if s == n_spans - 1 else min(left, max(1, m // n_spans))
            if length <= 0:
                break
            u = rng.random()
            start = 0 if u < p_edge / 2 else (n - length if u < p_edge else int(rng.integers(0, n - length + 1)))
            out[i, idx[start:start + length]] = True          # runs may overlap: the hidden share is then below ratio
            left -= length
    return out


def evaluation_mask(seq_ids: np.ndarray, obs: np.ndarray, seed: int) -> np.ndarray:
    """The fixed held-out masking task: one reproducible mask per SEQUENCE (seeded by its identifier), so every
    method is asked to predict exactly the same hidden points."""
    out = np.zeros(obs.shape, dtype=bool)
    for i, s in enumerate(seq_ids):
        out[i] = span_mask(obs[i:i + 1], np.random.default_rng(seed_for(seed, "evaluation_mask", s)), **EVAL_MASK)[0]
    return out


def truncation_mask(obs: np.ndarray, where: str, fraction: float) -> np.ndarray:
    """(n, G) True on the points that stay VISIBLE when the first (``head``) or last (``tail``) ``fraction`` of
    each row's recorded points is hidden — a controlled change of the recording window."""
    if where not in ("head", "tail"):
        raise ValueError(f"where must be 'head' or 'tail', got {where!r}")
    out = obs.copy()
    for i in range(obs.shape[0]):
        idx = np.flatnonzero(obs[i])
        k = int(np.ceil(fraction * idx.size))
        out[i, idx[:k] if where == "head" else idx[idx.size - k:]] = False
    return out


# ============================================================================== embeddings: long form ↔ blocks
def _lit(df: pl.DataFrame, **values: Any) -> pl.DataFrame:
    return df.with_columns([pl.lit(v).alias(k) for k, v in values.items()])


def embedding_frame(method: str, repeat: int, fold: int, seq_ids: np.ndarray, nfl_ids: np.ndarray, role: Any, view: str,
                    Z: np.ndarray, Zs: np.ndarray) -> pl.DataFrame:
    """Long-form rows of one fold model and view. Attempts without an embedding (NaN) are left out."""
    Z, Zs = np.asarray(Z, float), np.asarray(Zs, float)
    ok = np.isfinite(Z).all(1) & np.isfinite(Zs).all(1)
    n, E = int(ok.sum()), Z.shape[1]
    roles = np.broadcast_to(np.asarray(role, dtype=object), (Z.shape[0],))[ok].astype(str)
    df = pl.DataFrame({"seq_id": np.repeat(np.asarray(seq_ids)[ok].astype(str), E),
                       "nfl_id": np.repeat(np.asarray(nfl_ids)[ok].astype(np.int64), E), "role": np.repeat(roles, E),
                       "dim": np.tile(np.arange(1, E + 1), n), "value": Z[ok].ravel(), "value_z": Zs[ok].ravel()},
                      schema={"seq_id": S, "nfl_id": I, "role": S, "dim": I, "value": F, "value_z": F})
    return _cast(_lit(df, method=method, repeat=repeat, outer_fold=fold, view=view), EMBEDDINGS)


def recon_frame(method: str, repeat: int, fold: int, seq_ids: np.ndarray, nfl_ids: np.ndarray, frac: np.ndarray,
                ev: np.ndarray, err: np.ndarray, dev: np.ndarray, evm: np.ndarray, err_m: np.ndarray, dev_m: np.ndarray
                ) -> pl.DataFrame:
    """Per-sequence error sums. ``ev`` / ``evm`` (n, G): points counted with everything visible / under the mask;
    ``err`` and ``dev``: observed − reconstruction and observed − training mean (any value where not counted)."""
    def sums(keep, a):
        return (np.where(keep, a, 0.0) ** 2).sum(1)
    df = pl.DataFrame({"seq_id": np.asarray(seq_ids).astype(str), "nfl_id": np.asarray(nfl_ids).astype(np.int64),
                       "observed_fraction": np.asarray(frac, float), "n_points": ev.sum(1).astype(np.int64),
                       "sse": sums(ev, err), "sst": sums(ev, dev), "n_masked": evm.sum(1).astype(np.int64),
                       "sse_masked": sums(evm, err_m), "sst_masked": sums(evm, dev_m)})
    return _cast(_lit(df.filter(pl.col("n_points") > 0), method=method, repeat=repeat, outer_fold=fold), RECON_SEQ)


@dataclass
class EmbBlock:
    """Embeddings of ONE fold model in one view: one row per embedded attempt."""
    method: str
    repeat: int
    fold: int
    view: str
    seq_ids: np.ndarray
    nfl_ids: np.ndarray
    role: np.ndarray
    Z: np.ndarray              # (n, E) raw
    Zs: np.ndarray             # (n, E) standardised with the fold's training athletes

    def space(self, name: str) -> np.ndarray:
        if name not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {name!r}")
        return self.Zs if name == "standardised" else self.Z

    @property
    def held(self) -> np.ndarray:
        return self.role == "heldout"


def emb_blocks(emb: pl.DataFrame, view: str = "full") -> Iterator[EmbBlock]:
    d = emb.filter(pl.col("view") == view)
    for (method, repeat, fold), g in sorted(d.partition_by(["method", "repeat", "outer_fold"], as_dict=True).items()):
        g = g.sort("seq_id", "dim")
        E = g["dim"].n_unique()
        if g.height % E:
            raise ValueError(f"{method}, repeat {repeat}, fold {fold}, view {view}: {g.height} rows are not a multiple of "
                             f"the {E} dimensions")
        pick = slice(None, None, E)
        yield EmbBlock(method, int(repeat), int(fold), view, g["seq_id"].to_numpy()[pick].astype(str),
                       g["nfl_id"].to_numpy()[pick], g["role"].to_numpy()[pick].astype(str),
                       g["value"].to_numpy().reshape(-1, E), g["value_z"].to_numpy().reshape(-1, E))


def single_repeat(df: pl.DataFrame) -> int:
    """Every summary is computed on ONE split repeat (an athlete is then held out exactly once and is one unit)."""
    reps = df["repeat"].unique().to_list()
    if len(reps) != 1:
        raise ValueError(f"expected the embeddings of one split repeat, got repeats {sorted(reps)}: filter first")
    return int(reps[0])


def cohort_match(emb: pl.DataFrame) -> pl.DataFrame:
    """Held-out attempts embedded by each method, and how many of them every method of the same input embeds."""
    h = emb.filter((pl.col("role") == "heldout") & (pl.col("view") == "full") & (pl.col("dim") == 1))
    per = h.group_by("method").agg(pl.col("seq_id").n_unique().alias("n_sequences"), pl.col("nfl_id").n_unique().alias("n_athletes"))
    n_methods = h["method"].n_unique()
    common = h.group_by("seq_id").agg(pl.col("method").n_unique().alias("k")).filter(pl.col("k") == n_methods).height
    return per.with_columns(pl.lit(common).alias("n_sequences_common_to_all")).sort("method")


def restrict_to_common(emb: pl.DataFrame, groups: Sequence[Sequence[str]]) -> pl.DataFrame:
    """Keep, for the held-out rows of each group of methods (one input), only the attempts EVERY method of the group
    embeds. Training-side rows are untouched. With identical eligibility rules this removes nothing."""
    parts = []
    for methods in groups:
        d = emb.filter(pl.col("method").is_in(list(methods)))
        h = d.filter((pl.col("role") == "heldout") & (pl.col("view") == "full") & (pl.col("dim") == 1))
        keep = h.group_by("seq_id").agg(pl.col("method").n_unique().alias("k")).filter(pl.col("k") == len(methods))["seq_id"]
        parts.append(d.filter((pl.col("role") != "heldout") | pl.col("seq_id").is_in(keep.to_list())))
    return pl.concat(parts) if parts else emb.clear()


def athlete_embedding_table(emb: pl.DataFrame) -> pl.DataFrame:
    h = emb.filter((pl.col("role") == "heldout") & (pl.col("view") == "full"))
    out = h.group_by("method", "repeat", "outer_fold", "nfl_id", "dim").agg(
        pl.len().alias("n_attempts"), pl.col("value").mean().alias("value_mean"), pl.col("value_z").mean().alias("value_z_mean"))
    return _cast(out, ATHLETE_EMB).sort("method", "repeat", "nfl_id", "dim")


# ============================================================================== distances and pair statistics
def cdist(A: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    d2 = (A ** 2).sum(1)[:, None] + (Bm ** 2).sum(1)[None, :] - 2.0 * A @ Bm.T
    return np.sqrt(np.maximum(d2, 0.0))


def cosine(A: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(A, axis=1, keepdims=True), np.linalg.norm(Bm, axis=1, keepdims=True)
    return (A / np.maximum(na, 1e-12)) @ (Bm / np.maximum(nb, 1e-12)).T


def pair_stats(Z1: np.ndarray, Z2: np.ndarray, groups: np.ndarray, top_k: int = 5, symmetric: bool = True
               ) -> dict[str, np.ndarray]:
    """Row i of ``Z1`` and of ``Z2`` are two attempts of athlete i. The gallery of i is every athlete j ≠ i with
    ``groups[j] == groups[i]``. ``symmetric`` averages the two query directions. Ties count one half."""
    m = Z1.shape[0]
    D, C = cdist(Z1, Z2), cosine(Z1, Z2)
    groups = np.asarray(groups)
    off = (groups[:, None] == groups[None, :]) & ~np.eye(m, dtype=bool)
    n_cand = off.sum(1)
    safe = np.maximum(n_cand, 1)

    def one(Dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        same = np.diag(Dm)[:, None]
        closer = ((Dm < same) & off).sum(1) + 0.5 * ((Dm == same) & off).sum(1)
        return 1.0 - closer / safe, 1.0 + closer
    runs = [one(D), one(D.T)] if symmetric else [one(D)]
    has = n_cand > 0
    mean = lambda f: np.where(has, np.mean([f(a, r) for a, r in runs], axis=0), np.nan)      # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                                   # rows without a gallery
        d_diff = np.nanmedian(np.where(off, D, np.nan), axis=1)
        c_diff = np.nanmean(np.where(off, C, np.nan), axis=1)
    return {"n_candidates": n_cand.astype(np.int64), "auc": mean(lambda a, r: a),
            "top1": mean(lambda a, r: (r <= 1.0 + 1e-9).astype(float)), "topk": mean(lambda a, r: (r <= top_k).astype(float)),
            "rr": mean(lambda a, r: 1.0 / r), "d_same": np.diag(D).copy(), "d_diff_median": d_diff,
            "cos_same": np.diag(C).copy(), "cos_diff_mean": c_diff}


def repeat_pairs(b: EmbBlock, meta: Meta, space: str) -> dict[str, np.ndarray]:
    """First and second held-out attempt of every athlete of the block with at least two embedded attempts."""
    h = b.held
    seq, nfl, Z = b.seq_ids[h], b.nfl_ids[h], b.space(space)[h]
    r = meta.rows(seq)
    ids, i1, i2 = first_two(nfl, meta.order[r])
    return {"ids": ids, "Z1": Z[i1], "Z2": Z[i2], "position": meta.position[r][i1], "cohort": meta.cohort[r][i1],
            "row1": r[i1], "row2": r[i2], "seq1": seq[i1], "seq2": seq[i2]}


def pair_statistics(emb: pl.DataFrame, meta: Meta, ecfg: EvalConfig, spaces: Sequence[str] = SPACES) -> pl.DataFrame:
    parts = []
    for b in emb_blocks(emb):
        for space in spaces:
            p = repeat_pairs(b, meta, space)
            if p["ids"].size < 2:
                continue
            galleries = {"within_position": p["position"], "pooled": np.full(p["ids"].size, "all"),
                         "within_position_cohort": _join(p["position"], p["cohort"])}
            for scope in SCOPES:
                st = pair_stats(p["Z1"], p["Z2"], galleries[scope], ecfg.top_k)
                parts.append(_lit(pl.DataFrame({"nfl_id": p["ids"].astype(np.int64), **st}), method=b.method, space=space,
                                  scope=scope, repeat=b.repeat, outer_fold=b.fold))
    return _cast(pl.concat(parts), PAIRS) if parts else PAIRS.empty()


def boot_mean(x: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column means of ``x`` (rows = athletes) with a percentile interval from resampling rows."""
    x = np.asarray(x, float)
    x = x[:, None] if x.ndim == 1 else x
    n, nan = x.shape[0], np.full(x.shape[1], np.nan)
    if n == 0:
        return nan, nan, nan
    est = x.mean(0)
    if n < 5 or n_boot <= 0:
        return est, nan, nan
    pick = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    lo, hi = np.quantile(x[pick].mean(1), [0.025, 0.975], axis=0)
    return est, lo, hi


def _roles(ids: np.ndarray, roles: dict[int, str]) -> np.ndarray:
    return np.array([roles.get(int(i), "?") for i in ids], dtype=object).astype(str)


def repeatability_table(pairs: pl.DataFrame, meta: Meta, roles: dict[int, str], ecfg: EvalConfig, seed: int) -> pl.DataFrame:
    """``roles``: nfl_id → design / confirm. The seed of a subset does not depend on the method, so two methods
    with the same athletes share their resamples."""
    if pairs.height:
        single_repeat(pairs)
    rows = []
    for (method, space, scope), d in sorted(pairs.partition_by(["method", "space", "scope"], as_dict=True).items()):
        d = d.filter(pl.col("n_candidates") > 0).sort("nfl_id")
        ids = d["nfl_id"].to_numpy()
        if not ids.size:
            continue
        a = meta.athlete_rows(ids)
        pos, coh, role = meta.athlete_position[a], meta.athlete_cohort[a], _roles(ids, roles)
        subsets = [("all", np.ones(ids.size, dtype=bool))]
        if scope == "within_position":
            subsets += [(f"role:{r}", role == r) for r in ("confirm", "design")]
            subsets += [(f"cohort:{y}", coh == y) for y in sorted(set(coh.tolist()))]
            subsets += [(f"position:{p}", pos == p) for p in sorted(set(pos.tolist()))]
        V = np.column_stack([d[c].to_numpy() for c in PAIR_METRICS]).astype(float)
        cand, ds, dd = d["n_candidates"].to_numpy(), d["d_same"].to_numpy(), d["d_diff_median"].to_numpy()
        cs, cd = d["cos_same"].to_numpy(), d["cos_diff_mean"].to_numpy()
        for subset, keep in subsets:
            n = int(keep.sum())
            if not n:
                continue
            est, lo, hi = boot_mean(V[keep], ecfg.n_boot if n >= ecfg.min_athletes else 0,
                                    seed_for(seed, "repeatability", space, scope, subset))
            row = {"method": method, "space": space, "scope": scope, "subset": subset, "n_athletes": n,
                   "gallery_mean": float(cand[keep].mean()), "chance_top1": float((1.0 / (cand[keep] + 1)).mean()),
                   "distance_ratio": float(np.median(ds[keep]) / np.median(dd[keep])) if np.median(dd[keep]) > 0 else NAN,
                   "cos_same": float(np.median(cs[keep])), "cos_diff": float(np.median(cd[keep]))}
            for j, name in enumerate(("auc", "top1", "topk", "mrr")):
                row.update({name: est[j], f"{name}_lower": lo[j], f"{name}_upper": hi[j]})
            rows.append(row)
    return _table(rows, REPEAT)


def paired_comparisons(pairs: pl.DataFrame, reference: dict[str, str], roles: dict[int, str], ecfg: EvalConfig, seed: int,
                       metrics: Sequence[str] = ("auc", "top1", "rr"), scope: str = "within_position") -> pl.DataFrame:
    """method − reference on the athletes both methods score, same resamples for both (paired interval)."""
    d0 = pairs.filter((pl.col("scope") == scope) & (pl.col("n_candidates") > 0))
    by = dict(d0.partition_by(["method", "space"], as_dict=True).items())
    rows = []
    for (method, space), d in sorted(by.items()):
        ref = reference.get(method, "")
        if (ref, space) not in by:
            continue
        other = by[(ref, space)].select("nfl_id", *[pl.col(m).alias(f"{m}_ref") for m in metrics])
        j = d.select("nfl_id", *metrics).join(other, on="nfl_id", how="inner").sort("nfl_id")
        role = _roles(j["nfl_id"].to_numpy(), roles)
        for subset, keep in (("all", np.ones(j.height, dtype=bool)), ("role:confirm", role == "confirm"), ("role:design", role == "design")):
            n = int(keep.sum())
            if not n:
                continue
            for m in metrics:
                x, y = j[m].to_numpy()[keep].astype(float), j[f"{m}_ref"].to_numpy()[keep].astype(float)
                est, lo, hi = boot_mean(np.column_stack([x, y, x - y]), ecfg.n_boot if n >= ecfg.min_athletes else 0,
                                        seed_for(seed, "paired", space, scope, subset, m))
                rows.append({"method": method, "reference": ref, "space": space, "scope": scope, "subset": subset, "metric": m,
                             "n_athletes": n, "estimate": est[0], "reference_estimate": est[1], "difference": est[2],
                             "diff_lower": lo[2], "diff_upper": hi[2]})
    return _table(rows, PAIRED)


# ============================================================================== rotation-invariant reliability
def trace_icc(Z: np.ndarray, athletes: np.ndarray, strata: np.ndarray, n_boot: int = 0, seed: int = 0) -> dict[str, Any]:
    """Σ σ²_between / Σ (σ²_between + σ²_within) over the columns of ``Z`` (rows = attempts). Athletes are resampled
    inside their stratum, all columns together."""
    stats = [athlete_stats(Z[:, d], athletes, strata) for d in range(Z.shape[1])]
    st = stats[0]
    K, g = st["k"], st["k"].size
    out = {"n_athletes_repeat": int((K >= 2).sum()), "n_sequences": int(K.sum()), "icc_trace": None, "icc_lower": None,
           "icc_upper": None, "icc_dim_min": None, "icc_dim_median": None, "icc_dim_max": None}
    if g - st["n_strata"] < 1 or K.sum() <= g:
        return out
    S1, S2 = np.stack([s["s1"] for s in stats]), np.stack([s["s2"] for s in stats])

    def components(Kx, S1x, S2x):
        c = icc_core(Kx, S1x, S2x, st["groups"])
        with np.errstate(divide="ignore", invalid="ignore"):
            vb = np.where(c["n0"] > 0, np.maximum((c["msb"] - c["msw"]) / c["n0"], 0.0), np.nan)
            tot = vb.sum(0) + c["msw"].sum(0)
            return np.where(tot > 0, vb.sum(0) / tot, np.nan), c["icc"]
    trace, icc = components(K[None, :], S1, S2)
    ok = icc[np.isfinite(icc)]
    out.update(icc_trace=float(trace), icc_dim_min=float(ok.min()) if ok.size else None,
               icc_dim_median=float(np.median(ok)) if ok.size else None, icc_dim_max=float(ok.max()) if ok.size else None)
    if n_boot > 0 and g >= 3:
        pick = bootstrap_picks(st["groups"], n_boot, seed)
        reps, _ = components(K[pick][None], S1[:, pick], S2[:, pick])
        reps = reps[np.isfinite(reps)]
        if reps.size >= max(20, n_boot // 10):
            out.update(icc_lower=float(np.quantile(reps, 0.025)), icc_upper=float(np.quantile(reps, 0.975)))
    return out


def trace_icc_table(emb: pl.DataFrame, meta: Meta, ecfg: EvalConfig, seed: int) -> pl.DataFrame:
    by: dict[str, list[EmbBlock]] = {}
    for b in emb_blocks(emb):
        by.setdefault(b.method, []).append(b)
    rows = []
    for method, blocks in sorted(by.items()):
        for space in SPACES:
            Z = np.vstack([b.space(space)[b.held] for b in blocks])
            ath = np.concatenate([b.nfl_ids[b.held] for b in blocks])
            strata = np.concatenate([_join(meta.position[meta.rows(b.seq_ids[b.held])], np.full(int(b.held.sum()), b.fold))
                                     for b in blocks])
            n_boot = ecfg.n_boot if np.unique(ath).size >= ecfg.min_athletes else 0
            rows.append({"method": method, "space": space, **trace_icc(Z, ath, strata, n_boot, seed_for(seed, "trace_icc", space))})
    return _table(rows, ICC)


# ============================================================================== collapse
def collapse_metrics(Z: np.ndarray, near_identical: float = 0.01) -> dict[str, Any]:
    """Spread of a set of embeddings (rows). ``eigen_share`` and ``dim_variance`` are arrays; the rest scalars."""
    Z = np.asarray(Z, float)
    Z = Z[np.isfinite(Z).all(1)]
    n, E = Z.shape
    out = {"n_sequences": n, "embedding_dim": E, "dim_sd_min": NAN, "dim_sd_max": NAN, "effective_rank": NAN,
           "rank_share": NAN, "participation_ratio": NAN, "top_eigen_share": NAN, "near_identical_share": NAN,
           "median_distance": NAN, "eigen_share": np.full(E, np.nan), "dim_variance": np.full(E, np.nan)}
    if n < 3:
        return out
    var = Z.var(0, ddof=1)
    ev = np.clip(np.linalg.eigvalsh(np.atleast_2d(np.cov(Z.T)))[::-1], 0.0, None)
    D = cdist(Z, Z)[np.triu_indices(n, 1)]
    med = float(np.median(D))
    out.update(dim_sd_min=float(np.sqrt(var.min())), dim_sd_max=float(np.sqrt(var.max())), dim_variance=var, median_distance=med,
               near_identical_share=float((D < near_identical * med).mean()) if med > 0 else 1.0)
    if ev.sum() <= 0:
        out.update(effective_rank=0.0, rank_share=0.0, participation_ratio=0.0, top_eigen_share=NAN, eigen_share=np.zeros(E))
        return out
    p = ev / ev.sum()
    eff = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    out.update(effective_rank=eff, rank_share=eff / E, participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()),
               top_eigen_share=float(p[0]), eigen_share=p)
    return out


def is_collapsed(m: dict[str, Any], ecfg: EvalConfig) -> bool:
    """True when the embeddings occupy too few directions or too many attempts are indistinguishable. A set too
    small to judge is not called collapsed."""
    if not np.isfinite(m["rank_share"]):
        return False
    return bool(m["rank_share"] < ecfg.collapse_rank_share or m["near_identical_share"] > ecfg.collapse_identical_share)


def collapse_tables(emb: pl.DataFrame, ecfg: EvalConfig) -> dict[str, pl.DataFrame]:
    rows, spec = [], []
    for b in emb_blocks(emb):
        m = collapse_metrics(b.Z[b.held], ecfg.near_identical)
        key = {"method": b.method, "repeat": b.repeat, "outer_fold": b.fold}
        rows.append({**key, **{k: v for k, v in m.items() if k not in ("eigen_share", "dim_variance")}, "collapsed": is_collapsed(m, ecfg)})
        spec += [{**key, "index": j + 1, "eigen_share": m["eigen_share"][j], "dim_variance": m["dim_variance"][j]}
                 for j in range(m["embedding_dim"])]
    return {COLLAPSE.name: _table(rows, COLLAPSE), SPECTRUM.name: _table(spec, SPECTRUM)}


# ============================================================================== ridge probes
def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Ridge on standardised columns with an unpenalised intercept. ``Y`` (n,) or (n, q). Zero columns are allowed."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    X = X[:, None] if X.ndim == 1 else X
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    b = Y.mean(0)
    if X.shape[1] == 0:
        return {"mu": mu, "sd": sd, "w": np.zeros((0, *Y.shape[1:])), "b": b}
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(X.shape[1]), Xs.T @ (Y - b))
    return {"mu": mu, "sd": sd, "w": w, "b": b}


def ridge_predict(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    return ((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"]


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, float) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return NAN
    return float((rankdata(np.asarray(score, float))[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def athlete_folds(ids: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold label per row; all rows of an athlete share it (deterministic)."""
    uniq = np.unique(ids)
    order = np.random.default_rng(seed).permutation(uniq.size)
    fold_of = {int(a): int(j % k) for j, a in enumerate(uniq[order])}
    return np.array([fold_of[int(i)] for i in ids], dtype=int)


def dummies(labels: np.ndarray, levels: Sequence[str]) -> np.ndarray:
    return np.column_stack([(labels == v).astype(float) for v in levels[1:]]) if len(levels) > 1 else np.zeros((labels.size, 0))


def probe_splits(athletes: np.ndarray, held: np.ndarray, protocol: str, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(fit rows, scoring rows) pairs; an athlete is never on both sides of a pair."""
    h, t = np.flatnonzero(held), np.flatnonzero(~held)
    if protocol == "train_to_heldout":
        return [(t, h)]
    if protocol != "heldout_crossfit":
        raise ValueError(f"unknown probe protocol {protocol!r}")
    f = athlete_folds(athletes[h], k, seed)
    return [(h[f != j], h[f == j]) for j in range(k)]


def _units(b: EmbBlock, meta: Meta, level: str) -> dict[str, Any]:
    if level == "attempt":
        r = meta.rows(b.seq_ids)
        return {"athlete": b.nfl_ids, "X": b.Zs, "held": b.held, "position": meta.position[r], "row": r, "seq": b.seq_ids}
    ids, X, _ = athlete_means(b.nfl_ids, b.Zs)
    held = set(b.nfl_ids[b.held].tolist())
    a = meta.athlete_rows(ids)
    return {"athlete": ids, "X": X, "held": np.array([int(i) in held for i in ids]), "position": meta.athlete_position[a],
            "row": a, "seq": None}


def _target(name: str, kind: str, level: str, u: dict[str, Any], meta: Meta, obs_frac: dict[tuple[str, str], float], method: str
            ) -> tuple[np.ndarray, np.ndarray]:
    if kind == "categorical":
        y = u["position"] if name == "position_group" else meta.athlete_cohort[u["row"]]
        return y, y != "?"
    if level == "athlete":
        y = meta.official[name][u["row"]]
    elif name == "observed_fraction":
        y = np.array([obs_frac.get((method, str(s)), np.nan) for s in u["seq"]], dtype=float)
    else:
        y = meta.window[name][u["row"]]
    return y, np.isfinite(y)


def probe(blocks: Sequence[EmbBlock], meta: Meta, level: str, target: str, kind: str, protocol: str, ecfg: EvalConfig,
          seed: int, obs_frac: dict[tuple[str, str], float] | None = None) -> dict[str, Any] | None:
    """One probe of one method, pooled over its fold models. Returns None when nothing could be scored."""
    cols = []
    for b in blocks:
        u = _units(b, meta, level)
        y, ok = _target(target, kind, level, u, meta, obs_frac or {}, b.method)
        ok = ok & np.isfinite(u["X"]).all(1)
        C = dummies(u["position"], sorted(set(u["position"].tolist())))
        both = np.column_stack([C, u["X"]])
        for fit, ev in probe_splits(u["athlete"], u["held"], protocol, ecfg.probe_folds, seed_for(seed, "probe", b.repeat, b.fold)):
            fit, ev = fit[ok[fit]], ev[ok[ev]]
            if fit.size < 10 or not ev.size:
                continue
            ath = u["athlete"][ev]
            if kind == "categorical":
                classes = sorted(set(y[fit].tolist()))
                if len(classes) < 2:
                    continue
                Yf = np.column_stack([(y[fit] == c).astype(float) for c in classes])
                names = np.array(classes)
                full = u["X"] if target == "position_group" else both
                p1 = names[ridge_predict(ridge_fit(full[fit], Yf, ecfg.ridge_alpha), full[ev]).argmax(1)]
                p0 = (np.full(ev.size, names[int(Yf.sum(0).argmax())]) if target == "position_group"
                      else names[ridge_predict(ridge_fit(C[fit], Yf, ecfg.ridge_alpha), C[ev]).argmax(1)])
                cols.append((ath, (p1 == y[ev]).astype(float), (p0 == y[ev]).astype(float), np.ones(ev.size)))
                continue
            yf = y[fit].astype(float)
            p1 = ridge_predict(ridge_fit(both[fit], yf, ecfg.ridge_alpha), both[ev])
            p0 = ridge_predict(ridge_fit(C[fit], yf, ecfg.ridge_alpha), C[ev])
            cols.append((ath, (y[ev] - p1) ** 2, (y[ev] - p0) ** 2, (y[ev] - yf.mean()) ** 2) if kind == "continuous"
                        else (ath, p1, p0, y[ev].astype(float)))
    if not cols:
        return None
    ath, v1, v0, v2 = (np.concatenate(x) for x in zip(*cols))
    ids, inv = np.unique(ath, return_inverse=True)
    g = ids.size
    pick = (np.random.default_rng(seed_for(seed, "probe_boot", level, target, protocol)).integers(0, g, (ecfg.n_boot_probe, g))
            if g >= ecfg.min_athletes and ecfg.n_boot_probe > 0 else np.empty((0, g), dtype=int))
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "binary":
            if np.unique(v2).size < 2:
                return None
            members = [np.flatnonzero(inv == j) for j in range(g)]

            def stat(rows):
                a1, a0 = auc_score(v2[rows], v1[rows]), auc_score(v2[rows], v0[rows])
                return a1, a0, a1 - a0
            est = stat(np.arange(ath.size))
            reps = np.array([stat(np.concatenate([members[j] for j in p])) for p in pick], float).reshape(-1, 3)
            metric = "auc"
        else:
            s1, s0, s2 = (np.bincount(inv, weights=x, minlength=g) for x in (v1, v0, v2))
            if kind == "continuous":
                f, metric = (lambda a, b_, c: (1 - a / c, 1 - b_ / c, 1 - a / b_)), "r2"
            else:
                f, metric = (lambda a, b_, c: (a / c, b_ / c, a / c - b_ / c)), "accuracy"
            est = f(s1.sum(), s0.sum(), s2.sum())
            reps = np.column_stack(f(s1[pick].sum(1), s0[pick].sum(1), s2[pick].sum(1))) if pick.size else np.empty((0, 3))

    def interval(j):
        r = reps[:, j][np.isfinite(reps[:, j])] if reps.size else np.empty(0)
        return (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))) if r.size >= 20 else (NAN, NAN)
    (vl, vu), (gl, gu) = interval(0), interval(2)
    return {"level": level, "target": target, "metric": metric, "protocol": protocol, "n_athletes": int(g), "n_obs": int(ath.size),
            "value": float(est[0]), "baseline": float(est[1]), "gain": float(est[2]), "value_lower": vl, "value_upper": vu,
            "gain_lower": gl, "gain_upper": gu}


def probe_table(emb: pl.DataFrame, recon_seq: pl.DataFrame, meta: Meta, ecfg: EvalConfig, seed: int) -> pl.DataFrame:
    """Athlete-level probes under both protocols; attempt-level (recording) probes cross-fitted among held-out athletes."""
    obs_frac = {(m, s): v for m, s, v in recon_seq.select("method", "seq_id", "observed_fraction").iter_rows()}
    by: dict[str, list[EmbBlock]] = {}
    for b in emb_blocks(emb):
        by.setdefault(b.method, []).append(b)
    rows = []
    for method, blocks in sorted(by.items()):
        jobs = [("athlete", t, k, p) for t, k in ATHLETE_TARGETS for p in ("heldout_crossfit", "train_to_heldout")]
        jobs += [("attempt", t, k, "heldout_crossfit") for t, k in ATTEMPT_TARGETS]
        for level, target, kind, protocol in jobs:
            r = probe(blocks, meta, level, target, kind, protocol, ecfg, seed, obs_frac)
            if r is not None:
                rows.append({"method": method, **r})
    return _table(rows, PROBES)


# ============================================================================== residual information
def residualise_embeddings(emb: pl.DataFrame, meta: Meta, covariates: str, ecfg: EvalConfig, seed: int) -> pl.DataFrame:
    """Held-out attempt embeddings minus their linear prediction from official results (``official``: position,
    forty, 10-yard, weight, height) and optionally the recording window (``official_window``: + first-frame speed,
    duration, frames after the peak). The regression is cross-fitted among the held-out athletes of each fold, so
    an athlete's residual never uses his own embedding. Athletes without both official times are left out. The
    method is renamed ``<method>|resid_<covariates>`` and the rows go through the same repeatability functions."""
    if covariates not in RESIDUALS:
        raise ValueError(f"covariates must be one of {RESIDUALS}, got {covariates!r}")
    parts = []
    for b in emb_blocks(emb):
        h = np.flatnonzero(b.held)
        if h.size < 12:
            continue
        r, a = meta.rows(b.seq_ids[h]), meta.athlete_rows(b.nfl_ids[h])
        cont = [meta.official[c][a] for c in (FORTY, TEN, WEIGHT, HEIGHT)]
        need = [0, 1]
        if covariates == "official_window":
            cont += [meta.window[v][r] for v in JOINT_WINDOW]
            need += [4, 5, 6]
        ok = np.all([np.isfinite(cont[j]) for j in need], axis=0)
        pos = meta.position[r]
        X = np.column_stack([dummies(pos, sorted(set(pos.tolist()))), *cont])
        fold = athlete_folds(b.nfl_ids[h], ecfg.probe_folds, seed_for(seed, "residual", b.repeat, b.fold))
        R, Rs = np.full((h.size, b.Z.shape[1]), np.nan), np.full((h.size, b.Z.shape[1]), np.nan)
        for j in range(ecfg.probe_folds):
            fit, ev = ok & (fold != j), ok & (fold == j)
            if fit.sum() < X.shape[1] + 5 or not ev.any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(X[fit], axis=0)                   # weight / height: fit-side median when missing
            Xj = np.where(np.isfinite(X), X, np.where(np.isfinite(med), med, 0.0))
            for src, dst in ((b.Z[h], R), (b.Zs[h], Rs)):
                dst[ev] = src[ev] - ridge_predict(ridge_fit(Xj[fit], src[fit], 1e-6), Xj[ev])
        parts.append(embedding_frame(f"{b.method}|resid_{covariates}", b.repeat, b.fold, b.seq_ids[h], b.nfl_ids[h], "heldout",
                                     "full", R, Rs))
    return pl.concat(parts) if parts else EMBEDDINGS.empty()


# ============================================================================== recording-window dependence
def window_dependence_table(emb: pl.DataFrame, meta: Meta, ecfg: EvalConfig, seed: int) -> pl.DataFrame:
    by: dict[str, list[EmbBlock]] = {}
    for b in emb_blocks(emb):
        by.setdefault(b.method, []).append(b)
    rows = []
    for method, blocks in sorted(by.items()):
        for space in SPACES:
            folds = []
            for b in blocks:
                p = repeat_pairs(b, meta, space)
                dW = np.column_stack([meta.window[v][p["row2"]] - meta.window[v][p["row1"]] for v in JOINT_WINDOW])
                ok = np.isfinite(dW).all(1)
                if ok.sum() >= dW.shape[1] + 5:
                    folds.append((p["Z2"][ok] - p["Z1"][ok], np.column_stack([np.ones(int(ok.sum())), dW[ok]])))
            if not folds:
                continue

            def r2(picks):
                sse = sst = 0.0
                for (dZ, X), i in zip(folds, picks):
                    y, x = dZ[i], X[i]
                    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
                    sse += float(((y - x @ beta) ** 2).sum())
                    sst += float(((y - y.mean(0)) ** 2).sum())
                return 1.0 - sse / sst if sst > 0 else NAN
            n = int(sum(len(dZ) for dZ, _ in folds))
            rng = np.random.default_rng(seed_for(seed, "window", space))
            reps = np.array([r2([rng.integers(0, len(dZ), len(dZ)) for dZ, _ in folds]) for _ in range(ecfg.n_boot_probe)]
                            if n >= ecfg.min_athletes else [], float)
            reps = reps[np.isfinite(reps)]
            lo, hi = (np.quantile(reps, [0.025, 0.975]) if reps.size >= 20 else (NAN, NAN))
            rows.append({"method": method, "space": space, "n_athletes": n, "r2_trace": r2([np.arange(len(dZ)) for dZ, _ in folds]),
                         "r2_lower": float(lo), "r2_upper": float(hi)})
    return _table(rows, WINDOW)


def truncation_table(emb: pl.DataFrame, meta: Meta, ecfg: EvalConfig, seed: int) -> pl.DataFrame:
    full = {(b.method, b.repeat, b.fold): b for b in emb_blocks(emb)}
    views = sorted(v for v in emb["view"].unique().to_list() if v != "full")
    rows = []
    for view in views:
        acc: dict[str, dict[str, list[np.ndarray]]] = {}
        for b in emb_blocks(emb, view):
            f = full.get((b.method, b.repeat, b.fold))
            if f is None:
                continue
            fh = np.flatnonzero(f.held)
            seq, iv, jf = np.intersect1d(b.seq_ids, f.seq_ids[fh], return_indices=True)
            if seq.size < 3:
                continue
            Zv, Zf, nfl = b.Zs[iv], f.Zs[fh][jf], f.nfl_ids[fh][jf]
            D = cdist(Zf, Zf)
            scale = float(np.median(D[nfl[:, None] != nfl[None, :]]))
            r = meta.rows(seq)
            ids, i1, i2 = first_two(nfl, meta.order[r])
            a = acc.setdefault(b.method, {k: [] for k in ("ath", "shift", "pid", "auc", "auc_full", "top1", "top1_full")})
            a["ath"].append(nfl)
            a["shift"].append(np.linalg.norm(Zv - Zf, axis=1) / scale if scale > 0 else np.full(seq.size, np.nan))
            if ids.size >= 2:
                st = pair_stats(Zv[i1], Zf[i2], meta.position[r][i1], ecfg.top_k, symmetric=False)
                sf = pair_stats(Zf[i1], Zf[i2], meta.position[r][i1], ecfg.top_k, symmetric=False)
                keep = st["n_candidates"] > 0
                a["pid"].append(ids[keep])
                for k, src in (("auc", st["auc"]), ("auc_full", sf["auc"]), ("top1", st["top1"]), ("top1_full", sf["top1"])):
                    a[k].append(src[keep])
        for method, a in sorted(acc.items()):
            ath, shift = np.concatenate(a["ath"]), np.concatenate(a["shift"])
            ids, mean, _ = athlete_means(ath, shift)
            est, lo, hi = boot_mean(mean, ecfg.n_boot if ids.size >= ecfg.min_athletes else 0, seed_for(seed, "truncation", view))
            row = {"method": method, "view": view, "n_athletes": int(ids.size), "n_sequences": int(ath.size),
                   "relative_shift": est[0], "shift_lower": lo[0], "shift_upper": hi[0]}
            if a["pid"]:
                V = np.column_stack([np.concatenate(a[k]) for k in ("auc", "auc_full", "top1", "top1_full")])
                e, l_, h_ = boot_mean(V, ecfg.n_boot if len(V) >= ecfg.min_athletes else 0, seed_for(seed, "truncation_auc", view))
                row.update(auc=e[0], auc_lower=l_[0], auc_upper=h_[0], auc_full=e[1], top1=e[2], top1_full=e[3])
            rows.append(row)
    return _table(rows, TRUNC)


# ============================================================================== reconstruction
def reconstruction_table(recon_seq: pl.DataFrame, methods: pl.DataFrame, ecfg: EvalConfig, seed: int) -> pl.DataFrame:
    """RMSE per method from per-athlete error sums (athlete bootstrap), and method − reference on common athletes."""
    cols = ("sse", "n_points", "sst", "sse_masked", "n_masked")
    per = {m: d.group_by("nfl_id").agg(*[pl.col(c).sum().cast(pl.Float64) for c in cols], pl.len().alias("n_seq")).sort("nfl_id")
           for (m,), d in recon_seq.partition_by(["method"], as_dict=True).items()}

    def rmse(T: np.ndarray, masked: bool, pick: np.ndarray | None = None) -> Any:
        a, b_ = (T[:, 3], T[:, 4]) if masked else (T[:, 0], T[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(a.sum() / b_.sum()) if pick is None else np.sqrt(a[pick].sum(1) / b_[pick].sum(1))

    def ci(x: np.ndarray) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        return (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))) if x.size >= 20 else (NAN, NAN)
    rows = []
    for spec in methods.sort("method").iter_rows(named=True):
        d = per.get(spec["method"])
        if d is None:
            continue
        T = np.column_stack([d[c].to_numpy() for c in cols])
        g = d.height
        pick = (np.random.default_rng(seed_for(seed, "recon", spec["representation"])).integers(0, g, (ecfg.n_boot, g))
                if g >= ecfg.min_athletes else None)
        row = {"method": spec["method"], "reference": spec["reference"], "value_unit": spec["value_unit"], "n_athletes": g,
               "n_sequences": int(d["n_seq"].sum()), "rmse": float(rmse(T, False)), "rmse_masked": float(rmse(T, True)),
               "fve": float(1.0 - T[:, 0].sum() / T[:, 2].sum()) if T[:, 2].sum() > 0 else NAN}
        if pick is not None:
            (row["rmse_lower"], row["rmse_upper"]) = ci(rmse(T, False, pick))
            (row["rmse_masked_lower"], row["rmse_masked_upper"]) = ci(rmse(T, True, pick))
        ref = per.get(spec["reference"])
        if ref is not None:
            j = d.join(ref, on="nfl_id", how="inner", suffix="_ref").sort("nfl_id")
            A = np.column_stack([j[c].to_numpy() for c in cols])
            R = np.column_stack([j[f"{c}_ref"].to_numpy() for c in cols])
            p2 = (np.random.default_rng(seed_for(seed, "recon_diff", spec["method"])).integers(0, j.height, (ecfg.n_boot, j.height))
                  if j.height >= ecfg.min_athletes else None)
            for masked, name in ((False, "rmse_diff"), (True, "rmse_masked_diff")):
                row[name] = float(rmse(A, masked) - rmse(R, masked))
                if p2 is not None:
                    (row[f"{name}_lower"], row[f"{name}_upper"]) = ci(rmse(A, masked, p2) - rmse(R, masked, p2))
        rows.append(row)
    return _table(rows, RECON)


# ============================================================================== the FPCA benchmark as embeddings
def fpca_fold_outputs(spec: MethodSpec, curves: CurveSet, model: FpcaModel, repeat: int, fold: int, train: Sequence[int],
                      test: Sequence[int], support_eval: np.ndarray, ecfg: EvalConfig, seed: int
                      ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Embeddings (all athletes of the fold, full view; held-out athletes also under truncation) and held-out
    reconstruction sums of ONE fitted FPCA fold model, in the format every method uses. ``support_eval``: grid
    points on which errors are counted (the support shared with the other methods of the same input)."""
    k = spec.embedding_dim
    model.assert_unseen(test, f"{spec.method}, repeat {repeat}, fold {fold}: FPCA fit vs held-out athletes")
    nfl, seq, Y = curves.nfl_ids, curves.seq_ids.astype(str), curves.Y
    te, tr = np.isin(nfl, np.asarray(list(test))), np.isin(nfl, np.asarray(list(train)))
    sd = np.sqrt(np.maximum(model.eigenvalues[:k], 1e-300))
    Sc, frac = model.scores(Y, k)
    role = np.where(te, "heldout", "fit")
    keep = te | tr
    parts = [embedding_frame(spec.method, repeat, fold, seq[keep], nfl[keep], role[keep], "full", Sc[keep], Sc[keep] / sd)]
    Yt = Y[te]
    on_support = np.isfinite(Yt) & model.support[None, :]
    for where, fraction in ecfg.truncations:
        St, _ = model.scores(np.where(truncation_mask(on_support, where, fraction), Yt, np.nan), k)
        parts.append(embedding_frame(spec.method, repeat, fold, seq[te], nfl[te], "heldout", view_name(where, fraction), St, St / sd))

    def on_grid(scores: np.ndarray) -> np.ndarray:
        full = np.full(Yt.shape, np.nan)
        full[:, model.support] = model.reconstruct(np.nan_to_num(scores), k)
        return full
    mean = np.full(Y.shape[1], np.nan)
    mean[model.support] = model.mean
    sup = support_eval & model.support
    scored = np.isfinite(Sc[te][:, :1])
    ev = np.isfinite(Yt) & sup[None, :] & scored
    mask = evaluation_mask(seq[te], np.isfinite(Yt) & sup[None, :], seed)
    Sm, _ = model.scores(np.where(mask, np.nan, Yt), k)
    evm = mask & scored & np.isfinite(Sm[:, :1])
    return pl.concat(parts), recon_frame(spec.method, repeat, fold, seq[te], nfl[te], frac[te], ev, Yt - on_grid(Sc[te]),
                                         Yt - mean, evm, Yt - on_grid(Sm), Yt - mean)


# ============================================================================== summary, verdict, conclusions
def reference_map(methods: pl.DataFrame) -> dict[str, str]:
    """method → reference, including the residualised variants (``X|resid_r`` is compared with ``ref|resid_r``)."""
    out = {}
    for m, ref in methods.select("method", "reference").iter_rows():
        if ref:
            out[m] = ref
            out.update({f"{m}|resid_{r}": f"{ref}|resid_{r}" for r in RESIDUALS})
    return out


def summary_table(methods: pl.DataFrame, ev: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rep, probes, win, trunc, col = (ev[c.name] for c in (REPEAT, PROBES, WINDOW, TRUNC, COLLAPSE))
    rows = []
    for m in methods.sort("method").iter_rows(named=True):
        name = m["method"]

        def conf(method: str) -> dict[str, Any]:
            return _one(rep, method=method, space="standardised", scope="within_position", subset="role:confirm")

        def pr(target: str) -> dict[str, Any]:
            return _one(probes, method=name, protocol="heldout_crossfit", target=target)
        main, res, resw = conf(name), conf(f"{name}|resid_official"), conf(f"{name}|resid_official_window")
        rc = _one(ev[RECON.name], method=name)
        w = _one(win, method=name, space="standardised")
        c = col.filter(pl.col("method") == name)
        t = trunc.filter(pl.col("method") == name)
        coh, body = pr("draft_cohort"), [pr(WEIGHT).get("gain"), pr(HEIGHT).get("gain")]
        body = [x for x in body if x is not None]
        row = {"method": name, **{k: m[k] for k in ("label", "family", "representation", "embedding_dim", "n_parameters",
                                                     "train_seconds", "value_unit")},
               "rmse": rc.get("rmse"), "rmse_masked": rc.get("rmse_masked"), "n_confirm": main.get("n_athletes"),
               "chance_top1": main.get("chance_top1"), "icc_trace": _one(ev[ICC.name], method=name, space="raw").get("icc_trace"),
               "forty_r2": pr(FORTY).get("value"), "forty_partial_r2": pr(FORTY).get("gain"), "ten_partial_r2": pr(TEN).get("gain"),
               "body_partial_r2": max(body) if body else None, "window_r2": w.get("r2_trace"), "window_r2_lower": w.get("r2_lower"),
               "window_r2_upper": w.get("r2_upper"), "cohort_gain": coh.get("gain"), "cohort_gain_lower": coh.get("gain_lower"),
               "cohort_gain_upper": coh.get("gain_upper"), "position_accuracy": pr("position_group").get("value"),
               "truncation_shift_max": t["relative_shift"].max() if t.height else None,
               "rank_share_min": c["rank_share"].min() if c.height else None,
               "collapsed": bool(c["collapsed"].any()) if c.height else False}
        for prefix, src in (("auc", main), ("resid_auc", res), ("resid_window_auc", resw)):
            row.update({prefix: src.get("auc"), f"{prefix}_lower": src.get("auc_lower"), f"{prefix}_upper": src.get("auc_upper")})
        row.update(top1=main.get("top1"), top1_lower=main.get("top1_lower"), top1_upper=main.get("top1_upper"))
        rows.append(row)
    return _table(rows, SUMMARY)


def gain_label(lo: float | None, hi: float | None, margin: float) -> str:
    """better = the whole interval is above 0 · not_better = the whole interval is below the practical margin ·
    uncertain = anything else, including a missing interval."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "uncertain"
    return "better" if lo > 0 else "not_better" if hi < margin else "uncertain"


def verdict_table(summary: pl.DataFrame, paired: pl.DataFrame, recon: pl.DataFrame, methods: pl.DataFrame, ecfg: EvalConfig
                  ) -> pl.DataFrame:
    """A / B / C / D of the milestone brief, read on CONFIRM-role athletes against the FPCA of the same input.

    A  repeatability or non-redundant repeatability is better, the residual embedding discriminates above chance
       and is not worse than the reference's, and window / cohort dependence does not exceed the reference's by
       more than ``nuisance_tolerance``
    B  reconstruction is better but neither repeatability criterion is
    C  both repeatability criteria are not_better (or the representation collapsed)
    D  anything else: the intervals do not decide"""
    rows = []
    for m in methods.filter(pl.col("reference") != "").sort("method").iter_rows(named=True):
        name, ref = m["method"], m["reference"]
        s, r = _one(summary, method=name), _one(summary, method=ref)

        def diff(method: str, other: str) -> dict[str, Any]:
            return _one(paired, method=method, reference=other, space="standardised", subset="role:confirm", metric="auc")
        d_main, d_res = diff(name, ref), diff(f"{name}|resid_official", f"{ref}|resid_official")
        repeatability = gain_label(d_main.get("diff_lower"), d_main.get("diff_upper"), ecfg.margin_auc)
        nonredundant = gain_label(d_res.get("diff_lower"), d_res.get("diff_upper"), ecfg.margin_auc)
        rc = _one(recon, method=name)
        up, lo = rc.get("rmse_diff_upper"), rc.get("rmse_diff_lower")
        reconstruction = "uncertain" if up is None or lo is None else "better" if up < 0 else "worse" if lo > 0 else "uncertain"
        res_lo = s.get("resid_auc_lower")
        above_chance = res_lo is not None and res_lo > 0.5
        not_worse = d_res.get("diff_lower") is not None and d_res["diff_lower"] > -ecfg.margin_auc
        excess = [s.get(k) - r.get(k) for k in ("window_r2", "cohort_gain") if s.get(k) is not None and r.get(k) is not None]
        nuisance = "unknown" if len(excess) < 2 else "controlled" if max(excess) <= ecfg.nuisance_tolerance else "increased"
        if s.get("collapsed"):
            verdict = "C negative (collapsed)"
        elif "better" in (repeatability, nonredundant) and above_chance and not_worse and nuisance == "controlled":
            verdict = "A strong positive"
        elif repeatability == "not_better" and nonredundant == "not_better":
            verdict = "B partial positive" if reconstruction == "better" else "C negative"
        else:
            verdict = "D inconclusive"
        f = lambda x: "n/a" if x is None else f"{x:+.3f}"                                   # noqa: E731
        rows.append({"method": name, "reference": ref, "verdict": verdict, "repeatability": repeatability,
                     "nonredundant": nonredundant, "nuisance": nuisance, "reconstruction": reconstruction,
                     "detail": f"Δ AUC {f(d_main.get('difference'))} [{f(d_main.get('diff_lower'))}, {f(d_main.get('diff_upper'))}]; "
                               f"Δ residual AUC {f(d_res.get('difference'))} [{f(d_res.get('diff_lower'))}, {f(d_res.get('diff_upper'))}]; "
                               f"residual AUC lower bound {f(res_lo)}; Δ RMSE {f(rc.get('rmse_diff'))} [{f(lo)}, {f(up)}]; "
                               f"window / cohort excess over the reference {', '.join(f(x) for x in excess) or 'n/a'}"})
    return _table(rows, VERDICT)


def evaluate(emb: pl.DataFrame, recon_seq: pl.DataFrame, methods: pl.DataFrame, windows: pl.DataFrame, athletes: pl.DataFrame,
             splits: pl.DataFrame, ecfg: EvalConfig, seed: int) -> dict[str, pl.DataFrame]:
    """Every evaluation table from the embeddings and reconstruction sums of ONE split repeat. ``windows``:
    ``checkpoint.WINDOWS`` · ``athletes``: ``data.ATHLETES`` · ``splits``: design / confirm roles. Combine
    measurements only."""
    single_repeat(emb)
    meta = Meta(windows, athletes)
    roles = {int(i): str(r) for i, r in splits.select("nfl_id", "design_role").unique().iter_rows()}
    resid = pl.concat([residualise_embeddings(emb, meta, r, ecfg, seed) for r in RESIDUALS])
    pairs = pl.concat([pair_statistics(emb, meta, ecfg), pair_statistics(resid, meta, ecfg, spaces=("standardised",))])
    return {ATHLETE_EMB.name: athlete_embedding_table(emb), PAIRS.name: pairs,
            REPEAT.name: repeatability_table(pairs, meta, roles, ecfg, seed),
            PAIRED.name: paired_comparisons(pairs, reference_map(methods), roles, ecfg, seed),
            ICC.name: trace_icc_table(emb, meta, ecfg, seed), **collapse_tables(emb, ecfg),
            PROBES.name: probe_table(emb, recon_seq, meta, ecfg, seed),
            WINDOW.name: window_dependence_table(emb, meta, ecfg, seed), TRUNC.name: truncation_table(emb, meta, ecfg, seed),
            RECON.name: reconstruction_table(recon_seq, methods, ecfg, seed)}


def gate(ev: dict[str, pl.DataFrame], methods: pl.DataFrame, ecfg: EvalConfig) -> dict[str, pl.DataFrame]:
    summary = summary_table(methods, ev)
    return {SUMMARY.name: summary, VERDICT.name: verdict_table(summary, ev[PAIRED.name], ev[RECON.name], methods, ecfg)}


def _f(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def conclusions_markdown(summary: pl.DataFrame, verdict: pl.DataFrame, ecfg: EvalConfig, run_mode: str) -> str:
    """Text of the conclusions cell, written from the saved tables only (nothing is asserted from expectation)."""
    lines = [f"_Generated from `repr_method_summary` and `repr_verdict` of this run (`RUN_MODE = {run_mode!r}`). Read-out: "
             "confirm-role held-out athletes, within position, 95 % athlete-bootstrap intervals. Labels are heuristic._", ""]
    if run_mode == "smoke":
        lines += ["**Smoke run: a technical check of the wiring on a few athletes and epochs. The numbers below must not "
                  "be interpreted.**", ""]
    for rep in sorted(set(summary["representation"].to_list())):
        lines.append(f"**{SPEC[rep].label if rep in SPEC else rep}**")
        for s in summary.filter(pl.col("representation") == rep).sort("family", "method").iter_rows(named=True):
            lines.append(
                f"- {s['label']} ({s['embedding_dim']} dimensions, {s['n_parameters']} parameters): reconstruction RMSE "
                f"{_f(s['rmse'])} {s['value_unit']} (masked {_f(s['rmse_masked'])}); repeat-attempt AUC {_f(s['auc'])} "
                f"[{_f(s['auc_lower'])}, {_f(s['auc_upper'])}], top-1 {_f(s['top1'])} (chance {_f(s['chance_top1'])}); official "
                f"forty explained beyond position {_f(s['forty_partial_r2'], 2)}; AUC after removing official results "
                f"{_f(s['resid_auc'])} [{_f(s['resid_auc_lower'])}, {_f(s['resid_auc_upper'])}], and the window "
                f"{_f(s['resid_window_auc'])}; attempt-to-attempt change explained by the window {_f(s['window_r2'], 2)}; cohort "
                f"accuracy gain {_f(s['cohort_gain'], 2)}" + ("; **collapsed**" if s["collapsed"] else "") + ".")
        lines.append("")
    lines.append("**Go / no-go (rule-based, for review)**")
    for v in verdict.sort("method").iter_rows(named=True):
        lines.append(f"- `{v['method']}` against `{v['reference']}`: **{v['verdict']}** — repeatability {v['repeatability']}, "
                     f"non-redundant repeatability {v['nonredundant']}, nuisance {v['nuisance']}, reconstruction "
                     f"{v['reconstruction']}. {v['detail']}.")
    if not verdict.height:
        lines.append("- no method with a reference was evaluated.")
    lines += ["", f"A gain is called an improvement only when its paired interval excludes 0, and its absence only when the "
                  f"interval lies below {ecfg.margin_auc:g}. Reconstruction alone never decides a verdict. Nothing here uses an "
                  "NFL outcome, and no NFL predictive value is claimed."]
    return "\n".join(lines)
