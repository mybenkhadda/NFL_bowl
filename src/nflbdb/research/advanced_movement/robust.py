"""Robust validation of attempt embeddings and the frozen shortlist (notebook 04B). NumPy only; no outcome is read.

Everything here takes embeddings in the long form of ``repr_eval`` (one fold model = one block) and never compares
raw coordinates between two fitted encoders: statistics are computed inside a fold and then pooled, or are invariant
to a rotation.

Matched retrieval   the impostors of an athlete are the held-out athletes of the same fold and position whose official
                    forty, 10-yard split, weight and height are nearest to his (``MatchConfig``: k nearest inside a
                    caliper; scales = within-position SD among DESIGN athletes only). A representation that only
                    re-encodes the stopwatch and body size cannot separate an athlete from such impostors.
                    Official results themselves cannot be evaluated this way: they are one value per athlete, so
                    the two "attempts" coincide and the discrimination is 1 by construction. The honest control is
                    a per-attempt equivalent (tracked top speed), an untrained encoder, and the FPCA benchmark.
                    Uncertainty: athletes are resampled as QUERIES and as IMPOSTORS together (two-way), because one
                    athlete serves in several comparisons.
Residual audit      the embedding minus its linear prediction from official results, fitted four ways (cross-fitted
                    among held-out athletes as in notebook 03 · the same with the official results permuted between
                    athletes · on the fold's TRAINING athletes · in sample), under three distances. With covariates
                    that are constant per athlete and ONE fitted function, the distance between an athlete's two
                    attempts cannot change; cross-fitting applies a different function to each probe fold.
Projections         principal axes fitted on the fold's TRAINING embeddings, applied to every row of the fold.
Cohort probe        class counts, majority share, chance, accuracy and BALANCED accuracy, against position alone.
Seeds               per-seed estimates with their athlete intervals; the spread between seeds and the athlete standard
                    error are reported separately. No seed is selected.
Shortlist           rule-based, from saved Combine evidence; written once (``freeze_shortlist``).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import utc_now
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from ..studies.stats import rankdata
from . import fpca_eval as fe
from . import repr_eval as rv
from .atlas import GRADES, AtlasConfig
from .data import fold_athletes
from .fpca_eval import FORTY, HEIGHT, JOINT_WINDOW, TEN, WEIGHT, Meta, first_two
from .repr_eval import EvalConfig, _table

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
ROBUST_VERSION = "1"
RESID_MODES = ("crossfit", "crossfit_permuted", "train_fit", "heldout_in_sample")
AUDIT_METRICS = ("euclidean", "euclidean_restandardised", "cosine")
SUBSETS = ("all", "role:confirm")
SEED_METRICS = (("auc", True), ("top1", True), ("resid_auc", True), ("resid_window_auc", True), ("icc_trace", False),
                ("forty_partial_r2", False), ("window_r2", True), ("cohort_gain", True), ("truncation_shift_max", False),
                ("rank_share_min", False))


@dataclass(frozen=True)
class MatchConfig:
    version: str = ROBUST_VERSION
    variables: tuple[str, ...] = (FORTY, TEN, WEIGHT, HEIGHT)     # proposed: stopwatch and body size
    required: tuple[str, ...] = (FORTY, TEN, WEIGHT)              # an athlete without these is not matched
    k_nearest: int = 5                                            # proposed
    caliper: float = 0.75                                         # proposed: RMS difference in design-athlete SD units
    scale_role: str = "design"                                    # scales come from these athletes only

    def params(self) -> dict[str, Any]:
        return {**asdict(self), "variables": list(self.variables), "required": list(self.required)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MatchConfig":
        d = dict(d)
        for k in ("variables", "required"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


# ============================================================================== contracts
_M = {"method": S}
MATCHED = TableContract(
    "robust_matched_retrieval", "Per-athlete repeat-attempt discrimination against matched impostors and against the position.",
    {**_M, "repeat": I, "outer_fold": I, "nfl_id": I, "position_group": S, "design_role": S, "n_matched": I, "n_position": I,
     "auc_matched": F, "top1_matched": F, "auc_position": F, "d_same": F, "d_matched_median": F, "d_position_median": F,
     "match_distance_mean": F, "d_forty_mean": F, "d_ten_mean": F}, key=("method", "repeat", "nfl_id"), kind="dataset",
    area="method × athlete",
    docs={"n_matched": "impostors inside the caliper (≤ k_nearest); 0 = the athlete has no match and is not scored",
          "auc_matched": "share of the matched impostors farther than the athlete's own other attempt (both directions)",
          "auc_position": "same against every held-out athlete of the position in the fold",
          "match_distance_mean": "mean matching distance to his impostors (design-athlete SD units)",
          "d_forty_mean": "mean absolute difference of the official forty to his impostors [s]"})
MATCH_PAIRS = TableContract(
    "robust_matched_pairs", "The matched comparisons: query athlete × impostor, with what they share and how far apart they embed.",
    {**_M, "repeat": I, "outer_fold": I, "nfl_id": I, "impostor_id": I, "position_group": S, "rank": I, "match_distance": F,
     "d_forty": F, "d_ten": F, "d_weight": F, "d_height": F, "d_embedding": F, "d_same": F},
    key=("method", "repeat", "nfl_id", "impostor_id"), kind="dataset", area="method × query athlete × impostor",
    docs={"d_embedding": "distance from the query's first attempt to the impostor's second attempt (standardised space)",
          "d_same": "distance between the query's own two attempts", "d_forty": "impostor − query, official forty [s]"})
MATCH_SUMMARY = TableContract(
    "robust_matched_summary", "Matched-stopwatch retrieval by method, with one-way and two-way athlete intervals.",
    {**_M, "subset": S, "n_athletes": I, "n_unmatched": I, "n_comparisons": I, "n_distinct_impostors": I,
     "impostor_reuse_max": I, "match_distance_mean": F, "d_forty_mean": F, "d_ten_mean": F,
     **{f"auc_matched{s}": F for s in ("", "_lower", "_upper", "_lower2", "_upper2")},
     **{f"auc_position{s}": F for s in ("", "_lower", "_upper")}, **{f"matching_penalty{s}": F for s in ("", "_lower", "_upper")},
     "top1_matched": F, "chance_top1": F}, key=("method", "subset"), area="method × subset",
    docs={"auc_matched_lower": "95 %, query athletes resampled", "n_comparisons": "query × impostor comparisons (not independent)",
          "auc_matched_lower2": "95 %, athletes resampled as queries AND as impostors (the interval to read)",
          "impostor_reuse_max": "largest number of queries one athlete serves as an impostor for",
          "matching_penalty": "auc_position − auc_matched on the same athletes: what matching on the stopwatch removes"})
MATCH_PAIRED = TableContract(
    "robust_matched_paired", "Paired difference of matched discrimination between a method and a reference, same athletes.",
    {**_M, "reference": S, "subset": S, "n_athletes": I, "estimate": F, "reference_estimate": F, "difference": F, "diff_lower": F,
     "diff_upper": F}, key=("method", "reference", "subset"), area="method × reference × subset")
RESID_AUDIT = TableContract(
    "robust_residualisation_audit", "Repeat-attempt discrimination of raw and residual embeddings, by fitting mode and distance.",
    {**_M, "covariates": S, "mode": S, "metric": S, "subset": S, "n_athletes": I, "auc": F, "auc_lower": F, "auc_upper": F,
     "auc_minus_raw": F, "diff_lower": F, "diff_upper": F, "auc_same_probe_fold": F, "auc_other_probe_fold": F,
     "d_same_median": F, "d_other_median": F, "distance_ratio": F, "d_same_vs_raw": F, "effective_rank": F, "norm_mean": F,
     "variance_share": F}, key=("method", "covariates", "mode", "metric", "subset"), area="method × variant × distance × subset",
    docs={"mode": "raw · crossfit (notebook 03) · crossfit_permuted (official results shuffled between athletes of a position: "
                  "removes nothing real) · train_fit (fitted on the fold's training athletes) · heldout_in_sample",
          "metric": "euclidean on the standardised embedding · the same after dividing each residual dimension by its SD · cosine",
          "auc_minus_raw": "same athletes, same distance; athlete bootstrap, 95 %",
          "auc_same_probe_fold": "gallery restricted to athletes residualised with the SAME fitted function (cross-fitted modes)",
          "d_same_vs_raw": "median own-repeat distance / its raw value: 1 when the same vector is removed from both attempts",
          "variance_share": "trace of the residual covariance / trace of the raw covariance, same rows"})
COHORT_PROBE = TableContract(
    "robust_cohort_probe", "Held-out draft-cohort probe with class balance, majority baseline and balanced accuracy.",
    {**_M, "protocol": S, "n_athletes": I, "n_classes": I, "class_counts": S, "majority_share": F, "chance": F, "accuracy": F,
     "balanced_accuracy": F, "bal_lower": F, "bal_upper": F, "baseline_accuracy": F, "baseline_balanced_accuracy": F,
     "gain_balanced": F, "gain_lower": F, "gain_upper": F}, key=("method", "protocol"), area="method × protocol",
    docs={"balanced_accuracy": "mean recall over cohorts, position dummies + embedding", "chance": "1 / number of cohorts",
          "baseline_balanced_accuracy": "same probe with position dummies only",
          "gain_balanced": "balanced accuracy − baseline; athlete bootstrap, 95 %"})
SEED_RUNS = TableContract(
    "robust_seed_runs", "Per-seed estimates of the primary encoder with their athlete intervals.",
    {**_M, "training_seed": I, "metric": S, "value": F, "lower": F, "upper": F}, key=("method", "metric"), area="seed × metric")
SEED_SUMMARY = TableContract(
    "robust_seed_summary", "Spread between training seeds against athlete-sampling uncertainty.",
    {"metric": S, "n_seeds": I, "mean": F, "sd_seed": F, "min": F, "max": F, "se_athlete": F, "sd_total": F, "seed_share": F},
    key=("metric",), area="metric",
    docs={"sd_seed": "SD of the estimate between training seeds (same athletes, folds and budget) — few seeds: a rough figure",
          "se_athlete": "mean over seeds of the athlete-bootstrap interval width / 3.92",
          "sd_total": "√(sd_seed² + se_athlete²)", "seed_share": "sd_seed² / (sd_seed² + se_athlete²)"})
GEOMETRY = TableContract(
    "robust_seed_geometry", "Agreement of the geometry of two encoders on the same held-out attempts (rotation-invariant).",
    {"method_a": S, "method_b": S, "outer_fold": I, "n_sequences": I, "cka": F, "distance_spearman": F, "neighbour_overlap": F},
    key=("method_a", "method_b", "outer_fold"), area="encoder pair × fold",
    docs={"cka": "linear centred kernel alignment (1 = same geometry up to rotation and scale)",
          "distance_spearman": "rank correlation of the pairwise distances between attempts",
          "neighbour_overlap": "mean share of an attempt's 5 nearest attempts common to both encoders"})
LOCO = TableContract(
    "robust_loco", "Repeat-attempt discrimination on a cohort the encoder never saw, against the within-fold value of that cohort.",
    {**_M, "cohort": S, "protocol": S, "n_athletes": I, "gallery_mean": F, "auc": F, "auc_lower": F, "auc_upper": F},
    key=("method", "cohort", "protocol"), area="method × cohort × protocol",
    docs={"protocol": "leave_cohort_out = trained on the other cohorts · within_fold = regular folds, gallery of the same cohort"})
QUALITY = TableContract(
    "robust_representation_quality", "One row per representation: the evidence of notebook 04B side by side.",
    {**_M, "label": S, "kind": S, "embedding_dim": I, "n_confirm": I, **{f"auc{s}": F for s in ("", "_lower", "_upper")},
     **{f"matched_auc{s}": F for s in ("", "_lower", "_upper")}, **{f"resid_auc{s}": F for s in ("", "_lower", "_upper")},
     "icc_trace": F, "forty_partial_r2": F, **{f"window_r2{s}": F for s in ("", "_lower", "_upper")},
     **{f"cohort_balanced_gain{s}": F for s in ("", "_lower", "_upper")}, "truncation_shift_max": F, "rank_share_min": F,
     "collapsed": B}, key=("method",), area="method",
    docs={"kind": "primary · seed replicate · baseline · control · projection · sensitivity",
          "matched_auc_lower": "two-way athlete interval (queries and impostors resampled), confirm athletes"})
SHORTLIST = TableContract(
    "transfer_candidates_v1", "Frozen shortlist of Combine representation families for the first confirmatory NFL transfer comparison.",
    {"candidate_id": S, "group": S, "role": S, "selected": B, "readiness": S, "features": S, "definition": S,
     "unit_dimensionality": S, "sample_eligibility": S, "measurement_reliability": S, "nuisance_sensitivity": S,
     "stopwatch_redundancy": S, "selection_rationale": S, "known_limitations": S, "preprocessing_version": S,
     "artifact_paths": S, "config_hash": S, "shortlist_version": S, "frozen_at": S}, key=("candidate_id",), area="candidate family",
    docs={"selected": "enters the confirmatory comparison; an unselected row documents an empty slot and why",
          "readiness": "ready_for_transfer_test · carry_with_caveat · not_ready · reference — Combine evidence only; "
                       "NFL predictive value is untested", "features": "feature ids (dictionary) or method names, ';'-separated"})
EXPLORATORY = TableContract(
    "transfer_exploratory_registry", "Candidates left out of the frozen shortlist, kept for exploratory analyses only.",
    {"feature_id": S, "family": S, "grade": S, "transfer_eligibility": S, "stopwatch_relation": S, "reason": S},
    key=("feature_id",), area="feature")


# ============================================================================== naming of persisted evaluation tables
def prefixed_contracts(prefix: str, contracts: Sequence[TableContract]) -> list[TableContract]:
    return [rv.named(c, prefix) for c in contracts]


def to_prefixed(prefix: str, tables: dict[str, pl.DataFrame], contracts: Sequence[TableContract]) -> dict[str, pl.DataFrame]:
    return {rv.named(c, prefix).name: tables[c.name] for c in contracts}


def from_prefixed(prefix: str, tables: dict[str, pl.DataFrame], contracts: Sequence[TableContract]) -> dict[str, pl.DataFrame]:
    return {c.name: tables[rv.named(c, prefix).name] for c in contracts}


# ============================================================================== embeddings built without an encoder
def scalar_embedding(method: str, values: pl.DataFrame, columns: Sequence[str], splits: pl.DataFrame, repeats: Sequence[int]
                     ) -> pl.DataFrame:
    """A few per-attempt scalars as an 'embedding' (``values``: seq_id, nfl_id, columns). Standardised with the
    training athletes of each fold, like every other method. Attempts lacking a column are left out."""
    v = values.drop_nulls(list(columns)).sort("seq_id")
    seq, nfl = v["seq_id"].to_numpy().astype(str), v["nfl_id"].to_numpy().astype(np.int64)
    Z = np.column_stack([v[c].to_numpy().astype(float) for c in columns])
    parts = []
    for r in repeats:
        for f in sorted(splits.filter(pl.col("repeat") == r)["outer_fold"].unique().to_list()):
            train, test = fold_athletes(splits, r, f)
            te, tr = np.isin(nfl, np.asarray(test)), np.isin(nfl, np.asarray(train))
            if tr.sum() < 3 or not te.any():
                continue
            mu, sd = Z[tr].mean(0), Z[tr].std(0, ddof=1)
            sd = np.where(sd > 1e-12, sd, 1.0)
            keep = te | tr
            parts.append(rv.embedding_frame(method, int(r), int(f), seq[keep], nfl[keep], np.where(te, "heldout", "fit")[keep],
                                            "full", Z[keep], (Z[keep] - mu) / sd))
    return pl.concat(parts) if parts else rv.EMBEDDINGS.empty()


class ProjectionWarning(UserWarning):
    """A requested projection was left out, or some of its axes are not determined by the training embeddings."""


PROJ_OK, PROJ_FEW_ROWS, PROJ_OVER_DIM, PROJ_OVER_RANK = (
    "ok", "skipped: fewer than 3 training rows", "skipped: k exceeds the embedding dimension", "k exceeds the training rank")
RANK_RTOL = 1e-12                                 # a training variance below this share of the largest counts as zero


def training_axes(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(mean, axes, variances, rank) of the training rows ``X`` (n, E): orthonormal principal axes as COLUMNS, by
    decreasing training variance. ``rank`` = number of variances above ``RANK_RTOL`` × the largest; the axes beyond
    it complete the basis but their direction is not determined by the data. The SIGN of every axis is arbitrary:
    never compare a coordinate between two fits — only quantities that ignore it (distances, linear probes)."""
    mu = X.mean(0)
    w, V = np.linalg.eigh(np.atleast_2d(np.cov((X - mu).T)))
    order = np.argsort(w)[::-1]
    w = np.maximum(w[order], 0.0)
    rank = int((w > RANK_RTOL * w[0]).sum()) if w[0] > 0 else 0
    return mu, V[:, order], w, rank


def projection_status(k: int, dim: int, n_train: int, rank: int) -> str:
    """What ``project_embeddings`` does with k axes of a fold model: a k that cannot be honoured is left out, never
    replaced by another k."""
    if n_train < 3:
        return PROJ_FEW_ROWS
    if k > dim:
        return PROJ_OVER_DIM
    return PROJ_OVER_RANK if k > rank else PROJ_OK


def project_embeddings(emb: pl.DataFrame, method: str, dims: Sequence[int], tag: str = "pc") -> pl.DataFrame:
    """Rows of ``method`` projected on the first k principal axes of the fold's TRAINING embeddings (standardised
    space, centred on the training mean), for every k of ``dims`` and every view. The projection is not whitened, so
    k = full dimension reproduces the original distances exactly. New method name: ``<method>|<tag><k>``.
    A k above the embedding dimension, or a fold with fewer than 3 training rows, yields no row; a k above the
    training rank is projected (the basis is still orthonormal). Both are reported by one ``ProjectionWarning``."""
    d = emb.filter(pl.col("method") == method)
    full = {(b.repeat, b.fold): b for b in rv.emb_blocks(d, "full")}
    parts, notes = [], {}
    for view in sorted(d["view"].unique().to_list()):
        for b in rv.emb_blocks(d, view):
            fb = full[(b.repeat, b.fold)]
            X = fb.Zs[~fb.held]
            mu, V, _, rank = training_axes(X) if X.shape[0] >= 3 else (None, None, None, 0)
            for k in dims:
                status = projection_status(int(k), X.shape[1], X.shape[0], rank)
                if status != PROJ_OK:
                    notes.setdefault(status, set()).add(f"k = {k}, repeat {b.repeat}, fold {b.fold} (dimension {X.shape[1]}, "
                                                        f"training rows {X.shape[0]}, training rank {rank})")
                if not status.startswith("skipped"):
                    P = (b.Zs - mu) @ V[:, :k]
                    parts.append(rv.embedding_frame(f"{method}|{tag}{k}", b.repeat, b.fold, b.seq_ids, b.nfl_ids, b.role, view, P, P))
    if notes:
        warnings.warn(f"project_embeddings({method!r}): " + " · ".join(f"{s}: {'; '.join(sorted(v))}" for s, v in sorted(notes.items())),
                      ProjectionWarning, stacklevel=2)
    return pl.concat(parts) if parts else rv.EMBEDDINGS.empty()


def projection_report(emb: pl.DataFrame, method: str, dims: Sequence[int]) -> pl.DataFrame:
    """One row per fold model of ``method`` and k: the rows the axes are fitted on, the training rank, the share of
    the training variance the k axes hold, and what ``project_embeddings`` does with that k (``status``)."""
    rows = []
    for b in rv.emb_blocks(emb.filter(pl.col("method") == method), "full"):
        X = b.Zs[~b.held]
        w, rank = training_axes(X)[2:] if X.shape[0] >= 3 else (np.zeros(X.shape[1]), 0)
        for k in dims:
            status = projection_status(int(k), X.shape[1], X.shape[0], rank)
            share = float(w[:k].sum() / w.sum()) if w.sum() > 0 and not status.startswith("skipped") else NAN
            rows.append({"method": method, "repeat": b.repeat, "outer_fold": b.fold, "k": int(k), "embedding_dim": int(X.shape[1]),
                         "n_train_rows": int(X.shape[0]), "n_heldout_rows": int(b.held.sum()), "training_rank": int(rank),
                         "variance_share": share, "status": status})
    return pl.DataFrame(rows, schema={"method": S, "repeat": I, "outer_fold": I, "k": I, "embedding_dim": I, "n_train_rows": I,
                                      "n_heldout_rows": I, "training_rank": I, "variance_share": F, "status": S})


# ============================================================================== galleries and distances
def distance(Z1: np.ndarray, Z2: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if metric == "cosine":
        return 1.0 - rv.cosine(Z1, Z2)
    if metric != "euclidean":
        raise ValueError(f"unknown distance {metric!r}")
    return rv.cdist(Z1, Z2)


def gallery_stats(D: np.ndarray, off: np.ndarray, weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """``D[i, j]``: distance from attempt 1 of athlete i to attempt 2 of athlete j. ``off[i, j]``: j belongs to the
    gallery of i (any relation, not necessarily symmetric). Both query directions are averaged; a tie counts one half.
    ``weights``: multiplicity of each athlete as a gallery member (bootstrap). An athlete without a gallery gets NaN."""
    w = off.astype(float) if weights is None else off * np.asarray(weights, float)[None, :]
    n = w.sum(1)
    auc, top1 = np.zeros(D.shape[0]), np.zeros(D.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        for Dm in (D, D.T):
            same = np.diag(Dm)[:, None]
            closer = ((Dm < same) * w).sum(1) + 0.5 * ((Dm == same) * w).sum(1)
            auc += 0.5 * (1.0 - closer / n)
            top1 += 0.5 * (closer == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_other = np.nanmedian(np.where(off, D, np.nan), axis=1)
    has = n > 0
    return {"n": off.sum(1).astype(np.int64), "auc": np.where(has, auc, np.nan), "top1": np.where(has, top1, np.nan),
            "d_same": np.diag(D).copy(), "d_other": d_other}


# ============================================================================== matched-stopwatch retrieval
def matching_scales(meta: Meta, roles: dict[int, str], mc: MatchConfig) -> dict[str, float]:
    """Within-position SD of each matching variable among the athletes of ``mc.scale_role`` — the only athletes the
    matching rule is tuned on. Read-out athletes contribute nothing to the scales."""
    ids = np.array(sorted(i for i, r in roles.items() if r == mc.scale_role and i in meta.athlete), dtype=np.int64)
    rows = meta.athlete_rows(ids)
    pos = meta.athlete_position[rows]
    out = {}
    for v in mc.variables:
        x = meta.official[v][rows]
        ok = np.isfinite(x)
        ss, df = 0.0, 0
        for p in np.unique(pos[ok]):
            g = x[ok & (pos == p)]
            ss, df = ss + float(((g - g.mean()) ** 2).sum()), df + g.size - 1
        if df < 5 or ss <= 0:
            raise ValueError(f"matching scale of {v}: too few {mc.scale_role} athletes with a value ({int(ok.sum())})")
        out[v] = float(np.sqrt(ss / df))
    return out


def match_matrix(ids: np.ndarray, position: np.ndarray, meta: Meta, scales: dict[str, float], mc: MatchConfig
                 ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """(M, D, differences). ``M[i, j]``: j is one of the ≤ k nearest athletes to i, of the same position, inside the
    caliper. ``D``: RMS scaled difference over the variables both hold. ``differences[v][i, j]`` = x_j − x_i, raw units."""
    rows, m = meta.athlete_rows(ids), ids.size
    sq, cnt, valid, diffs = np.zeros((m, m)), np.zeros((m, m)), np.ones((m, m), dtype=bool), {}
    for v in mc.variables:
        x = meta.official[v][rows]
        both = np.isfinite(x)[:, None] & np.isfinite(x)[None, :]
        diffs[v] = x[None, :] - x[:, None]
        if v in mc.required:
            valid &= both
        sq += np.where(both, (np.nan_to_num(diffs[v]) / scales[v]) ** 2, 0.0)
        cnt += both
    for v in (FORTY, TEN, WEIGHT, HEIGHT):                       # reported for every pair, matched on or not
        if v not in diffs:
            x = meta.official[v][rows]
            diffs[v] = x[None, :] - x[:, None]
    D = np.sqrt(sq / np.maximum(cnt, 1))
    cand = valid & (position[:, None] == position[None, :]) & ~np.eye(m, dtype=bool) & (cnt > 0) & (D <= mc.caliper)
    order = np.argsort(np.where(cand, D, np.inf), axis=1, kind="stable")[:, :mc.k_nearest]
    M = np.zeros((m, m), dtype=bool)
    M[np.arange(m)[:, None], order] = True
    return M & cand, D, diffs


def matched_retrieval(emb: pl.DataFrame, meta: Meta, roles: dict[int, str], scales: dict[str, float], mc: MatchConfig,
                      ecfg: EvalConfig, seed: int, pairs_for: Sequence[str] = (), space: str = "standardised"
                      ) -> dict[str, pl.DataFrame]:
    """MATCHED, MATCH_PAIRS (methods of ``pairs_for`` only) and MATCH_SUMMARY for every method of ``emb``."""
    per, pairs, folds = [], [], {}
    for b in rv.emb_blocks(emb):
        p = rv.repeat_pairs(b, meta, space)
        ids = p["ids"]
        if ids.size < 3:
            continue
        M, Dm, diffs = match_matrix(ids, p["position"], meta, scales, mc)
        P = (p["position"][:, None] == p["position"][None, :]) & ~np.eye(ids.size, dtype=bool)
        D = rv.cdist(p["Z1"], p["Z2"])
        sm, sp = gallery_stats(D, M), gallery_stats(D, P)
        role = rv._roles(ids, roles)
        folds.setdefault(b.method, []).append({"D": D, "M": M, "confirm": role == "confirm"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            md = np.nanmean(np.where(M, Dm, np.nan), axis=1)
            d40 = np.nanmean(np.where(M, np.abs(diffs[FORTY]), np.nan), axis=1)
            d10 = np.nanmean(np.where(M, np.abs(diffs[TEN]), np.nan), axis=1)
        per.append(rv._lit(pl.DataFrame({"nfl_id": ids.astype(np.int64), "position_group": p["position"], "design_role": role,
                                         "n_matched": sm["n"], "n_position": sp["n"], "auc_matched": sm["auc"],
                                         "top1_matched": sm["top1"], "auc_position": sp["auc"], "d_same": sm["d_same"],
                                         "d_matched_median": sm["d_other"], "d_position_median": sp["d_other"],
                                         "match_distance_mean": md, "d_forty_mean": d40, "d_ten_mean": d10}),
                           method=b.method, repeat=b.repeat, outer_fold=b.fold))
        if b.method in pairs_for:
            for i in range(ids.size):
                js = np.flatnonzero(M[i])
                for rank, j in enumerate(js[np.argsort(Dm[i, js], kind="stable")], start=1):
                    pairs.append({"method": b.method, "repeat": b.repeat, "outer_fold": b.fold, "nfl_id": int(ids[i]),
                                  "impostor_id": int(ids[j]), "position_group": str(p["position"][i]), "rank": rank,
                                  "match_distance": float(Dm[i, j]), "d_forty": diffs[FORTY][i, j], "d_ten": diffs[TEN][i, j],
                                  "d_weight": diffs[WEIGHT][i, j], "d_height": diffs[HEIGHT][i, j], "d_embedding": float(D[i, j]),
                                  "d_same": float(D[i, i])})
    matched = fe._cast(pl.concat(per), MATCHED) if per else MATCHED.empty()
    return {MATCHED.name: matched, MATCH_PAIRS.name: _table(pairs, MATCH_PAIRS),
            MATCH_SUMMARY.name: _matched_summary(matched, folds, ecfg, seed)}


def two_way_intervals(items: Sequence[dict[str, np.ndarray]], n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """95 % interval of the mean matched discrimination when the athletes of every fold are resampled with replacement
    and the resampled athletes act BOTH as queries and as impostors (weights = multiplicities). Keys: ``SUBSETS``."""
    rng = np.random.default_rng(seed)
    reps = {s: [] for s in SUBSETS}
    for _ in range(n_boot):
        num, den = dict.fromkeys(SUBSETS, 0.0), dict.fromkeys(SUBSETS, 0.0)
        for it in items:
            m = it["D"].shape[0]
            w = rng.multinomial(m, np.full(m, 1.0 / m)).astype(float)
            a = gallery_stats(it["D"], it["M"], weights=w)["auc"]
            for s, q in (("all", w), ("role:confirm", w * it["confirm"])):
                ok = np.isfinite(a) & (q > 0)
                num[s] += float((q[ok] * a[ok]).sum())
                den[s] += float(q[ok].sum())
        for s in SUBSETS:
            reps[s].append(num[s] / den[s] if den[s] > 0 else np.nan)
    out = {}
    for s in SUBSETS:
        r = np.asarray(reps[s], float)
        r = r[np.isfinite(r)]
        out[s] = (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))) if r.size >= 20 else (NAN, NAN)
    return out


def _matched_summary(matched: pl.DataFrame, folds: dict[str, list[dict[str, np.ndarray]]], ecfg: EvalConfig, seed: int
                     ) -> pl.DataFrame:
    rows = []
    for (method,), d in sorted(matched.partition_by(["method"], as_dict=True).items()):
        two = two_way_intervals(folds.get(method, []), ecfg.n_boot, seed_for(seed, "matched_two_way"))
        d = d.sort("nfl_id")
        for subset in SUBSETS:
            g = d if subset == "all" else d.filter(pl.col("design_role") == "confirm")
            e = g.filter(pl.col("n_matched") > 0)
            if not e.height:
                continue
            V = np.column_stack([e["auc_matched"].to_numpy(), e["auc_position"].to_numpy(),
                                 e["auc_position"].to_numpy() - e["auc_matched"].to_numpy(), e["top1_matched"].to_numpy()]).astype(float)
            est, lo, hi = rv.boot_mean(V, ecfg.n_boot if e.height >= ecfg.min_athletes else 0, seed_for(seed, "matched", subset))
            confirm = subset == "role:confirm"
            reuse = np.concatenate([it["M"][it["confirm"] if confirm else slice(None)].sum(0) for it in folds.get(method, [])] or [np.zeros(1)])
            rows.append({"method": method, "subset": subset, "n_athletes": e.height, "n_unmatched": g.height - e.height,
                         "n_comparisons": int(e["n_matched"].sum()), "n_distinct_impostors": int((reuse > 0).sum()),
                         "impostor_reuse_max": int(reuse.max()), "match_distance_mean": e["match_distance_mean"].mean(),
                         "d_forty_mean": e["d_forty_mean"].mean(), "d_ten_mean": e["d_ten_mean"].mean(),
                         "auc_matched": est[0], "auc_matched_lower": lo[0], "auc_matched_upper": hi[0],
                         "auc_matched_lower2": two[subset][0], "auc_matched_upper2": two[subset][1],
                         "auc_position": est[1], "auc_position_lower": lo[1], "auc_position_upper": hi[1],
                         "matching_penalty": est[2], "matching_penalty_lower": lo[2], "matching_penalty_upper": hi[2],
                         "top1_matched": est[3], "chance_top1": float((1.0 / (e["n_matched"].to_numpy() + 1)).mean())})
    return _table(rows, MATCH_SUMMARY)


def matched_paired(matched: pl.DataFrame, reference: dict[str, str], ecfg: EvalConfig, seed: int) -> pl.DataFrame:
    """method − reference of the matched discrimination, on the athletes both score (paired athlete bootstrap)."""
    by = {m: d.filter(pl.col("n_matched") > 0).select("nfl_id", "design_role", "auc_matched")
          for (m,), d in matched.partition_by(["method"], as_dict=True).items()}
    rows = []
    for method, ref in sorted(reference.items()):
        if method not in by or ref not in by:
            continue
        j = by[method].join(by[ref].select("nfl_id", pl.col("auc_matched").alias("_ref")), on="nfl_id", how="inner").sort("nfl_id")
        for subset in SUBSETS:
            g = j if subset == "all" else j.filter(pl.col("design_role") == "confirm")
            if not g.height:
                continue
            x, y = g["auc_matched"].to_numpy().astype(float), g["_ref"].to_numpy().astype(float)
            est, lo, hi = rv.boot_mean(np.column_stack([x, y, x - y]), ecfg.n_boot if g.height >= ecfg.min_athletes else 0,
                                       seed_for(seed, "matched_paired", subset))
            rows.append({"method": method, "reference": ref, "subset": subset, "n_athletes": g.height, "estimate": est[0],
                         "reference_estimate": est[1], "difference": est[2], "diff_lower": lo[2], "diff_upper": hi[2]})
    return _table(rows, MATCH_PAIRED)


def matched_examples(pairs: pl.DataFrame, matched: pl.DataFrame, method: str, athletes: pl.DataFrame, k: int = 4
                     ) -> list[dict[str, Any]]:
    """Rule-chosen query athletes of ``method`` with their NEAREST matched impostor: the athletes at evenly spaced
    quantiles of the matched discrimination (lowest to highest). Chosen by rule, not by eye."""
    d = matched.filter((pl.col("method") == method) & (pl.col("n_matched") > 0)).sort("auc_matched", "nfl_id")
    near = pairs.filter((pl.col("method") == method) & (pl.col("rank") == 1))
    name = dict(athletes.select("nfl_id", "display_name").iter_rows())
    forty = dict(athletes.select("nfl_id", "cmb_forty").iter_rows())
    out = []
    for q in np.linspace(0.0, 1.0, k) if d.height else []:
        row = d.row(min(int(round(q * (d.height - 1))), d.height - 1), named=True)
        p = near.filter(pl.col("nfl_id") == row["nfl_id"])
        if not p.height or any(e["nfl_id"] == row["nfl_id"] for e in out):
            continue
        p = p.row(0, named=True)
        out.append({"nfl_id": row["nfl_id"], "impostor_id": p["impostor_id"], "position_group": row["position_group"],
                    "name": name.get(row["nfl_id"], ""), "impostor_name": name.get(p["impostor_id"], ""),
                    "forty": forty.get(row["nfl_id"]), "impostor_forty": forty.get(p["impostor_id"]),
                    "auc_matched": row["auc_matched"], "d_same": p["d_same"], "d_embedding": p["d_embedding"],
                    "match_distance": p["match_distance"]})
    return out


# ============================================================================== residualisation audit
def _covariates(b: rv.EmbBlock, meta: Meta, covariates: str) -> tuple[np.ndarray, np.ndarray]:
    """(design matrix, usable rows) for EVERY row of the block: position dummies + official forty, 10-yard, weight,
    height (+ the recording-window variables). Dummy levels are those of the whole block, so one fitted function can
    be applied to training and held-out rows alike."""
    if covariates not in rv.RESIDUALS:
        raise ValueError(f"covariates must be one of {rv.RESIDUALS}, got {covariates!r}")
    r, a = meta.rows(b.seq_ids), meta.athlete_rows(b.nfl_ids)
    cont, need = [meta.official[c][a] for c in (FORTY, TEN, WEIGHT, HEIGHT)], [0, 1]
    if covariates == "official_window":
        cont += [meta.window[v][r] for v in JOINT_WINDOW]
        need += [4, 5, 6]
    ok = np.all([np.isfinite(cont[j]) for j in need], axis=0)
    pos = meta.position[r]
    return np.column_stack([rv.dummies(pos, sorted(set(pos.tolist()))), *cont]), ok


def _fit_apply(X: np.ndarray, Z: np.ndarray, fit: np.ndarray, ev: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X[fit], axis=0)                       # weight / height: fit-side median when missing
    Xj = np.where(np.isfinite(X), X, np.where(np.isfinite(med), med, 0.0))
    return Z[ev] - rv.ridge_predict(rv.ridge_fit(Xj[fit], Z[fit], 1e-6), Xj[ev])


def residual_variant(b: rv.EmbBlock, meta: Meta, covariates: str, mode: str, ecfg: EvalConfig, seed: int) -> dict[str, np.ndarray]:
    """Residual of the standardised embedding of the HELD-OUT rows of one fold model.

    crossfit            fitted among held-out athletes, athlete-grouped probe folds (the procedure and the folds of
                        ``repr_eval.residualise_embeddings``): a different fitted function per probe fold
    crossfit_permuted   the same, the covariate rows exchanged between athletes of a position ('official' only)
    train_fit           ONE function fitted on the fold's training athletes, applied to every held-out attempt
    heldout_in_sample   ONE function fitted on the held-out attempts themselves (transductive; diagnostic)
    Returns R (held-out rows × dimensions, NaN where undefined), probe_fold, seq, nfl."""
    if mode not in RESID_MODES:
        raise ValueError(f"mode must be one of {RESID_MODES}, got {mode!r}")
    X, ok = _covariates(b, meta, covariates)
    h, tr = np.flatnonzero(b.held), np.flatnonzero(~b.held)
    nfl_h = b.nfl_ids[h]
    R, pf = np.full((h.size, b.Zs.shape[1]), np.nan), np.zeros(h.size, dtype=int)
    if mode.startswith("crossfit"):
        Xh, okh = X[h], ok[h]
        if mode == "crossfit_permuted":
            if covariates != "official":
                raise ValueError("crossfit_permuted is defined for athlete-level covariates ('official') only")
            rng = np.random.default_rng(seed_for(seed, "permute", b.repeat, b.fold))
            pos_h = meta.position[meta.rows(b.seq_ids[h])]
            first = {int(a): i for i, a in reversed(list(enumerate(nfl_h.tolist())))}
            donor: dict[int, int] = {}
            for p in np.unique(pos_h):
                ath = np.unique(nfl_h[pos_h == p])
                donor.update(zip(ath.tolist(), rng.permutation(ath).tolist()))
            take = np.array([first[donor[int(a)]] for a in nfl_h], dtype=int)
            Xh, okh = Xh[take], okh[take]
        pf = rv.athlete_folds(nfl_h, ecfg.probe_folds, seed_for(seed, "residual", b.repeat, b.fold))
        for j in range(ecfg.probe_folds):
            fit, ev = np.flatnonzero(okh & (pf != j)), np.flatnonzero(okh & (pf == j))
            if fit.size >= X.shape[1] + 5 and ev.size:
                R[ev] = _fit_apply(Xh, b.Zs[h], fit, ev)
    else:
        ev = h[ok[h]]
        fit = tr[ok[tr]] if mode == "train_fit" else ev
        if fit.size >= X.shape[1] + 5 and ev.size:
            R[ok[h]] = _fit_apply(X, b.Zs, fit, ev)
    return {"R": R, "probe_fold": pf, "seq": b.seq_ids[h], "nfl": nfl_h}


def residual_audit(emb: pl.DataFrame, meta: Meta, roles: dict[int, str], ecfg: EvalConfig, seed: int,
                   covariate_sets: Sequence[str] = rv.RESIDUALS) -> pl.DataFrame:
    """RESID_AUDIT for every method of ``emb``: raw against each residual variant, on the SAME athletes (those with
    both official times), with the same positive pair (first two attempts) and the same gallery (position, fold)."""
    acc: dict[tuple, dict[str, list]] = {}
    for b in rv.emb_blocks(emb):
        h = np.flatnonzero(b.held)
        if h.size < 12:
            continue
        _, ok0 = _covariates(b, meta, "official")
        raw = {"R": np.where(ok0[h][:, None], b.Zs[h], np.nan), "probe_fold": np.zeros(h.size, dtype=int), "seq": b.seq_ids[h],
               "nfl": b.nfl_ids[h]}
        variants = [("none", "raw", raw)]
        for cov in covariate_sets:
            variants += [(cov, m, residual_variant(b, meta, cov, m, ecfg, seed)) for m in RESID_MODES
                         if not (m == "crossfit_permuted" and cov != "official")]
        raw_var = float(np.nanvar(raw["R"], axis=0, ddof=1).sum())
        for cov, mode, v in variants:
            good = np.isfinite(v["R"]).all(1)
            if good.sum() < 6:
                continue
            Rg, seq, nfl, pf = v["R"][good], v["seq"][good], v["nfl"][good], v["probe_fold"][good]
            r = meta.rows(seq)
            ids, i1, i2 = first_two(nfl, meta.order[r])
            if ids.size < 3:
                continue
            pos, pfa = meta.position[r][i1], pf[i1]
            P = (pos[:, None] == pos[None, :]) & ~np.eye(ids.size, dtype=bool)
            same_fn = P & (pfa[:, None] == pfa[None, :])
            sd = Rg.std(0, ddof=1)
            sd = np.where(sd > 1e-12, sd, 1.0)
            fold_stats = {"effective_rank": rv.collapse_metrics(Rg)["effective_rank"], "norm_mean": float(np.linalg.norm(Rg, axis=1).mean()),
                          "variance_share": float(Rg.var(0, ddof=1).sum() / raw_var) if raw_var > 0 else NAN, "weight": float(len(Rg))}
            for metric in AUDIT_METRICS:
                Z1, Z2 = (Rg[i1] / sd, Rg[i2] / sd) if metric == "euclidean_restandardised" else (Rg[i1], Rg[i2])
                D = distance(Z1, Z2, "cosine" if metric == "cosine" else "euclidean")
                st = gallery_stats(D, P)
                a = acc.setdefault((b.method, cov, mode, metric), {k: [] for k in ("ids", "auc", "same", "other", "d_same", "d_other", "fold")})
                a["ids"].append(ids)
                a["auc"].append(st["auc"])
                a["same"].append(gallery_stats(D, same_fn)["auc"])
                a["other"].append(gallery_stats(D, P & ~same_fn)["auc"])
                a["d_same"].append(st["d_same"])
                a["d_other"].append(st["d_other"])
                a["fold"].append(fold_stats)
    cat = {k: {f: (np.concatenate(v[f]) if f != "fold" else v[f]) for f in v} for k, v in acc.items()}
    rows = []
    for (method, cov, mode, metric), v in sorted(cat.items()):
        base = cat.get((method, "none", "raw", metric))
        role = rv._roles(v["ids"], roles)
        wts = np.array([f["weight"] for f in v["fold"]])
        fold_mean = lambda key: float(np.average([f[key] for f in v["fold"]], weights=wts)) if wts.sum() > 0 else NAN   # noqa: E731
        for subset in SUBSETS:
            keep = np.isfinite(v["auc"]) & (np.ones(role.size, dtype=bool) if subset == "all" else role == "confirm")
            n = int(keep.sum())
            if not n:
                continue
            boots = ecfg.n_boot if n >= ecfg.min_athletes else 0
            est, lo, hi = rv.boot_mean(v["auc"][keep], boots, seed_for(seed, "audit", subset))
            ds, do = float(np.median(v["d_same"][keep])), float(np.nanmedian(v["d_other"][keep]))
            row = {"method": method, "covariates": cov, "mode": mode, "metric": metric, "subset": subset, "n_athletes": n,
                   "auc": est[0], "auc_lower": lo[0], "auc_upper": hi[0],
                   "auc_same_probe_fold": float(np.nanmean(v["same"][keep])) if np.isfinite(v["same"][keep]).any() else None,
                   "auc_other_probe_fold": float(np.nanmean(v["other"][keep])) if np.isfinite(v["other"][keep]).any() else None,
                   "d_same_median": ds, "d_other_median": do, "distance_ratio": ds / do if do > 0 else None,
                   "effective_rank": fold_mean("effective_rank"), "norm_mean": fold_mean("norm_mean"),
                   "variance_share": fold_mean("variance_share")}
            if base is not None:
                _, ia, ib = np.intersect1d(v["ids"][keep], base["ids"], return_indices=True)
                diff = v["auc"][keep][ia] - base["auc"][ib]
                ok = np.isfinite(diff)
                if ok.any():
                    e, l_, h_ = rv.boot_mean(diff[ok], boots, seed_for(seed, "audit_diff", subset))
                    row.update(auc_minus_raw=e[0], diff_lower=l_[0], diff_upper=h_[0])
                    raw_ds = float(np.median(base["d_same"][ib][ok]))
                    row["d_same_vs_raw"] = float(np.median(v["d_same"][keep][ia][ok])) / raw_ds if raw_ds > 0 else None
            rows.append(row)
    return _table(rows, RESID_AUDIT)


def audit_reading(audit: pl.DataFrame, method: str, ecfg: EvalConfig, subset: str = "role:confirm") -> str:
    """Rule-based reading of the audit of one method, from the saved numbers (Euclidean distance, 'official' covariates)."""
    def one(mode: str, cov: str = "official", metric: str = "euclidean") -> dict[str, Any]:
        return fe._one(audit, method=method, covariates=cov, mode=mode, metric=metric, subset=subset)
    raw, cf, perm, tf, ins = one("raw", "none"), one("crossfit"), one("crossfit_permuted"), one("train_fit"), one("heldout_in_sample")
    if not raw or not cf:
        return f"- `{method}`: the audit could not be computed (too few athletes with both official times)."
    f = lambda r, k="auc": "n/a" if not r or r.get(k) is None else f"{r[k]:.3f}"          # noqa: E731
    up = lambda r: bool(r) and r.get("diff_lower") is not None and r["diff_lower"] > 0       # noqa: E731
    flat = lambda r: bool(r) and r.get("diff_upper") is not None and r["diff_upper"] < ecfg.margin_auc   # noqa: E731
    lines = [f"- `{method}` ({subset}, n = {raw['n_athletes']}): raw AUC {f(raw)}; cross-fitted residual {f(cf)} (Δ {f(cf, 'auc_minus_raw')} "
             f"[{f(cf, 'diff_lower')}, {f(cf, 'diff_upper')}]); cross-fitted with PERMUTED official results {f(perm)} (Δ "
             f"{f(perm, 'auc_minus_raw')}); fitted on training athletes {f(tf)} (Δ {f(tf, 'auc_minus_raw')} [{f(tf, 'diff_lower')}, "
             f"{f(tf, 'diff_upper')}]); in sample {f(ins)}."]
    lines.append(f"  - Own-repeat distance relative to raw: cross-fitted {f(cf, 'd_same_vs_raw')}, training fit {f(tf, 'd_same_vs_raw')} "
                 f"(1 = unchanged, as expected when one vector is removed from both attempts); cross-fitted gallery under the same "
                 f"fitted function {f(cf, 'auc_same_probe_fold')} against another function {f(cf, 'auc_other_probe_fold')}; residual "
                 f"variance share {f(cf, 'variance_share')}; effective rank raw {f(raw, 'effective_rank')} → {f(cf, 'effective_rank')}.")
    if up(cf) and up(perm):
        verdict = ("the increase also appears when the official results are permuted, so it is produced by the cross-fitting "
                   "procedure (different fitted functions for different athletes), not by removing stopwatch information")
    elif up(cf) and (flat(tf) or (tf and not up(tf))):
        verdict = ("the increase is specific to cross-fitting: one function fitted on training athletes does not reproduce it. "
                   "Read the training-fit residual as the estimate of non-redundant repeatability")
    elif up(cf) and up(tf):
        verdict = ("the increase survives a single training-fitted function and is absent under permutation: removing the official "
                   "direction changes the geometry in favour of the remaining dimensions. It is a property of the metric, and still "
                   "not evidence of independent NEW information")
    else:
        verdict = "the cross-fitted residual does not exceed the raw embedding on these athletes: nothing to explain"
    lines.append(f"  - Reading (rule-based, for review): {verdict}.")
    return "\n".join(lines)


# ============================================================================== cohort probe with class balance
def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    classes = np.unique(y)
    return float(np.mean([(pred[y == c] == c).mean() for c in classes])) if classes.size else NAN


def cohort_probe(emb: pl.DataFrame, meta: Meta, ecfg: EvalConfig, seed: int,
                 protocols: Sequence[str] = ("heldout_crossfit", "train_to_heldout")) -> pl.DataFrame:
    """Athlete-level ridge probe of the draft cohort (one-hot targets), fitted and scored on disjoint athletes.
    'full' = position dummies + embedding, 'baseline' = position dummies. Balanced accuracy weighs every cohort
    equally, so a probe that answers the majority cohort scores 1 / number of cohorts."""
    by: dict[str, list[rv.EmbBlock]] = {}
    for b in rv.emb_blocks(emb):
        by.setdefault(b.method, []).append(b)
    rows = []
    for method, blocks in sorted(by.items()):
        for protocol in protocols:
            got = []
            for b in blocks:
                u = rv._units(b, meta, "athlete")
                y = meta.athlete_cohort[u["row"]]
                ok = (y != "?") & np.isfinite(u["X"]).all(1)
                C = rv.dummies(u["position"], sorted(set(u["position"].tolist())))
                both = np.column_stack([C, u["X"]])
                for fit, ev in rv.probe_splits(u["athlete"], u["held"], protocol, ecfg.probe_folds, seed_for(seed, "probe", b.repeat, b.fold)):
                    fit, ev = fit[ok[fit]], ev[ok[ev]]
                    classes = sorted(set(y[fit].tolist()))
                    if fit.size < 10 or not ev.size or len(classes) < 2:
                        continue
                    names = np.array(classes)
                    Yf = np.column_stack([(y[fit] == c).astype(float) for c in classes])
                    p1 = names[rv.ridge_predict(rv.ridge_fit(both[fit], Yf, ecfg.ridge_alpha), both[ev]).argmax(1)]
                    p0 = names[rv.ridge_predict(rv.ridge_fit(C[fit], Yf, ecfg.ridge_alpha), C[ev]).argmax(1)]
                    got.append((y[ev], p1, p0))
            if not got:
                continue
            y, p1, p0 = (np.concatenate(x) for x in zip(*got))
            names, counts = np.unique(y, return_counts=True)
            g = y.size
            bal, base = balanced_accuracy(y, p1), balanced_accuracy(y, p0)
            row = {"method": method, "protocol": protocol, "n_athletes": g, "n_classes": int(names.size),
                   "class_counts": "; ".join(f"{n}: {int(k)}" for n, k in zip(names.tolist(), counts.tolist())),
                   "majority_share": float(counts.max() / g), "chance": 1.0 / names.size, "accuracy": float((p1 == y).mean()),
                   "balanced_accuracy": bal, "baseline_accuracy": float((p0 == y).mean()), "baseline_balanced_accuracy": base,
                   "gain_balanced": bal - base}
            if g >= ecfg.min_athletes and ecfg.n_boot_probe > 0:
                pick = np.random.default_rng(seed_for(seed, "cohort_probe", protocol)).integers(0, g, (ecfg.n_boot_probe, g))
                reps = np.array([(balanced_accuracy(y[i], p1[i]), balanced_accuracy(y[i], p0[i])) for i in pick])
                row.update(bal_lower=float(np.quantile(reps[:, 0], 0.025)), bal_upper=float(np.quantile(reps[:, 0], 0.975)),
                           gain_lower=float(np.quantile(reps[:, 0] - reps[:, 1], 0.025)),
                           gain_upper=float(np.quantile(reps[:, 0] - reps[:, 1], 0.975)))
            rows.append(row)
    return _table(rows, COHORT_PROBE)


# ============================================================================== training seeds
def seed_runs(summary: pl.DataFrame, seeds: dict[str, int], trace: pl.DataFrame | None = None,
              matched: pl.DataFrame | None = None, cohort: pl.DataFrame | None = None) -> pl.DataFrame:
    """Long table of the per-seed estimates. ``seeds``: method → training seed. ``summary``: ``repr_eval.SUMMARY``
    rows · ``trace``: trace reliability · ``matched``: MATCH_SUMMARY · ``cohort``: COHORT_PROBE."""
    rows = []
    for method, s in seeds.items():
        r = fe._one(summary, method=method)
        if not r:
            continue
        def add(metric: str, value: Any, lower: Any = None, upper: Any = None) -> None:
            rows.append({"method": method, "training_seed": int(s), "metric": metric, "value": value, "lower": lower, "upper": upper})
        for metric, has_ci in SEED_METRICS:
            add(metric, r.get(metric), r.get(f"{metric}_lower") if has_ci else None, r.get(f"{metric}_upper") if has_ci else None)
        if trace is not None:
            t = fe._one(trace, method=method, space="raw")
            rows[-len(SEED_METRICS) + [m for m, _ in SEED_METRICS].index("icc_trace")].update(lower=t.get("icc_lower"), upper=t.get("icc_upper"))
        if matched is not None:
            m = fe._one(matched, method=method, subset="role:confirm")
            add("matched_auc", m.get("auc_matched"), m.get("auc_matched_lower2"), m.get("auc_matched_upper2"))
        if cohort is not None:
            c = fe._one(cohort, method=method, protocol="heldout_crossfit")
            add("cohort_balanced_gain", c.get("gain_balanced"), c.get("gain_lower"), c.get("gain_upper"))
    return _table(rows, SEED_RUNS)


def aggregate_seeds(runs: pl.DataFrame) -> pl.DataFrame:
    """Mean and dispersion over seeds, with the athlete standard error beside the spread between seeds. Training-seed
    variability (same athletes, different initialisation and batches) and athlete-sampling uncertainty (which
    athletes were measured) are different things and are not merged into one interval."""
    rows = []
    for (metric,), d in sorted(runs.partition_by(["metric"], as_dict=True).items()):
        v = d["value"].drop_nulls().to_numpy().astype(float)
        if not v.size:
            continue
        w = (d["upper"] - d["lower"]).drop_nulls().to_numpy().astype(float)
        sd = float(v.std(ddof=1)) if v.size >= 2 else None
        se = float(w.mean() / 3.92) if w.size else None
        total = float(np.sqrt(sd ** 2 + se ** 2)) if sd is not None and se is not None else None
        rows.append({"metric": metric, "n_seeds": int(v.size), "mean": float(v.mean()), "sd_seed": sd, "min": float(v.min()),
                     "max": float(v.max()), "se_athlete": se, "sd_total": total,
                     "seed_share": sd ** 2 / total ** 2 if total else None})
    return _table(rows, SEED_SUMMARY)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X, Y = X - X.mean(0), Y - Y.mean(0)
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(np.linalg.norm(X.T @ Y) ** 2 / den) if den > 0 else NAN


def cross_geometry(emb: pl.DataFrame, methods: Sequence[str], k: int = 5) -> pl.DataFrame:
    """Geometry agreement between every pair of ``methods`` on the held-out attempts of each fold. Coordinates of two
    encoders are never matched: only quantities that survive a rotation are compared."""
    blocks = {(b.method, b.fold): b for b in rv.emb_blocks(emb.filter(pl.col("method").is_in(list(methods))))}
    rows = []
    for x, ma in enumerate(methods):
        for mb in methods[x + 1:]:
            for fold in sorted({f for m, f in blocks if m == ma}):
                a, b = blocks.get((ma, fold)), blocks.get((mb, fold))
                if a is None or b is None:
                    continue
                _, ia, ib = np.intersect1d(a.seq_ids[a.held], b.seq_ids[b.held], return_indices=True)
                if ia.size < 10:
                    continue
                A, Bm = a.Zs[a.held][ia], b.Zs[b.held][ib]
                Da, Db = rv.cdist(A, A), rv.cdist(Bm, Bm)
                iu = np.triu_indices(ia.size, 1)
                ra, rb = rankdata(Da[iu]), rankdata(Db[iu])
                np.fill_diagonal(Da, np.inf)
                np.fill_diagonal(Db, np.inf)
                na, nb = np.argsort(Da, axis=1)[:, :k], np.argsort(Db, axis=1)[:, :k]
                overlap = float(np.mean([np.intersect1d(na[i], nb[i]).size / k for i in range(ia.size)]))
                rows.append({"method_a": ma, "method_b": mb, "outer_fold": int(fold), "n_sequences": int(ia.size),
                             "cka": linear_cka(A, Bm), "distance_spearman": float(np.corrcoef(ra, rb)[0, 1]),
                             "neighbour_overlap": overlap})
    return _table(rows, GEOMETRY)


# ============================================================================== leave-one-cohort-out (tables)
def cohort_splits(splits: pl.DataFrame, repeat: int = 0) -> tuple[pl.DataFrame, dict[int, str]]:
    """A split table whose outer fold is the DRAFT COHORT (one fold per cohort; inner folds unchanged), and the
    fold → cohort mapping. Passing it to the fold loop trains on the other cohorts and holds one out."""
    s = splits.filter(pl.col("repeat") == repeat).with_columns(pl.col("stratum").str.split("|").list.get(1).alias("_c"))
    cohorts = sorted(s["_c"].unique().to_list())
    code = {c: i for i, c in enumerate(cohorts)}
    out = s.with_columns(pl.col("_c").replace_strict(code, return_dtype=pl.Int64).alias("outer_fold")).drop("_c")
    return out.select(splits.columns), {i: c for c, i in code.items()}


def loco_table(emb_loco: pl.DataFrame, emb_within: pl.DataFrame, meta: Meta, fold_cohort: dict[int, str], base_of: dict[str, str],
               ecfg: EvalConfig, seed: int) -> pl.DataFrame:
    """``base_of``: leave-cohort-out method → its regular-fold method. Gallery: athletes of the same position (and,
    within folds, of the same cohort), so both protocols compare an athlete with others of his own cohort."""
    rows = []

    def add(method: str, cohort: str, protocol: str, d: pl.DataFrame) -> None:
        d = d.filter(pl.col("n_candidates") > 0)
        if not d.height:
            return
        est, lo, hi = rv.boot_mean(d["auc"].to_numpy().astype(float), ecfg.n_boot if d.height >= ecfg.min_athletes else 0,
                                   seed_for(seed, "loco", cohort, protocol))
        rows.append({"method": method, "cohort": cohort, "protocol": protocol, "n_athletes": d.height,
                     "gallery_mean": float(d["n_candidates"].mean()), "auc": est[0], "auc_lower": lo[0], "auc_upper": hi[0]})
    if emb_loco.height:
        pl_ = rv.pair_statistics(emb_loco, meta, ecfg, spaces=("standardised",)).filter(pl.col("scope") == "within_position")
        for (method, fold), d in sorted(pl_.partition_by(["method", "outer_fold"], as_dict=True).items()):
            add(base_of.get(method, method), fold_cohort.get(int(fold), str(fold)), "leave_cohort_out", d)
    within = emb_within.filter(pl.col("method").is_in(list(set(base_of.values()))))
    if within.height:
        pw = rv.pair_statistics(within, meta, ecfg, spaces=("standardised",)).filter(pl.col("scope") == "within_position_cohort")
        coh = meta.athlete_cohort[meta.athlete_rows(pw["nfl_id"].to_numpy())]
        pw = pw.with_columns(pl.Series("_cohort", coh.tolist()))
        for (method, cohort), d in sorted(pw.partition_by(["method", "_cohort"], as_dict=True).items()):
            add(method, str(cohort), "within_fold", d)
    return _table(rows, LOCO)


# ============================================================================== one row per representation
def quality_table(summary: pl.DataFrame, methods: pl.DataFrame, matched: pl.DataFrame, cohort: pl.DataFrame, kinds: dict[str, str]
                  ) -> pl.DataFrame:
    """``kinds``: method → primary / seed replicate / baseline / control / projection / sensitivity."""
    rows = []
    for m in methods.sort("method").iter_rows(named=True):
        name = m["method"]
        s = fe._one(summary, method=name)
        if not s:
            continue
        mt = fe._one(matched, method=name, subset="role:confirm")
        ch = fe._one(cohort, method=name, protocol="heldout_crossfit")
        rows.append({"method": name, "label": m["label"], "kind": kinds.get(name, "other"), "embedding_dim": m["embedding_dim"],
                     "n_confirm": s.get("n_confirm"), **{k: s.get(k) for k in ("auc", "auc_lower", "auc_upper", "resid_auc",
                                                                                "resid_auc_lower", "resid_auc_upper", "icc_trace",
                                                                                "forty_partial_r2", "window_r2", "window_r2_lower",
                                                                                "window_r2_upper", "truncation_shift_max", "rank_share_min")},
                     "matched_auc": mt.get("auc_matched"), "matched_auc_lower": mt.get("auc_matched_lower2"),
                     "matched_auc_upper": mt.get("auc_matched_upper2"), "cohort_balanced_gain": ch.get("gain_balanced"),
                     "cohort_balanced_gain_lower": ch.get("gain_lower"), "cohort_balanced_gain_upper": ch.get("gain_upper"),
                     "collapsed": bool(s.get("collapsed"))})
    return _table(rows, QUALITY)


DIMS = TableContract(
    "robust_dimension_stability", "Evidence by embedding dimension: training-fitted projections of every seed, and references.",
    {"method": S, "base_method": S, "training_seed": I, "k": I, "auc": F, "auc_lower": F, "auc_upper": F, "matched_auc": F,
     "matched_auc_lower": F, "matched_auc_upper": F, "resid_auc": F, "icc_trace": F, "effective_rank": F, "forty_partial_r2": F,
     "window_r2": F, "cohort_gain": F}, key=("method",), area="method (projection)",
    docs={"k": "dimensions kept (projection of the base method) or native dimension (reference rows)",
          "effective_rank": "exp(entropy of the covariance spectrum), mean over fold models — an absolute number of directions",
          "training_seed": "null for a method that was not retrained here"})


def dimension_table(summary: pl.DataFrame, collapse: pl.DataFrame, matched: pl.DataFrame,
                    entries: Sequence[tuple[str, str, int | None, int]]) -> pl.DataFrame:
    """``entries``: (method, base method, training seed or None, k). Rows are read from the evaluation tables."""
    rows = []
    for method, base, seed, k in entries:
        s = fe._one(summary, method=method)
        if not s:
            continue
        mt = fe._one(matched, method=method, subset="role:confirm")
        col = collapse.filter(pl.col("method") == method)
        rows.append({"method": method, "base_method": base, "training_seed": seed, "k": int(k),
                     **{x: s.get(x) for x in ("auc", "auc_lower", "auc_upper", "resid_auc", "icc_trace", "forty_partial_r2",
                                              "window_r2", "cohort_gain")},
                     "matched_auc": mt.get("auc_matched"), "matched_auc_lower": mt.get("auc_matched_lower2"),
                     "matched_auc_upper": mt.get("auc_matched_upper2"),
                     "effective_rank": col["effective_rank"].mean() if col.height else None})
    return _table(rows, DIMS)


def compact_dimension(paired: pl.DataFrame, seed_methods: Sequence[str], dims: Sequence[int], margin: float, tag: str = "pc"
                      ) -> tuple[int | None, str]:
    """Smallest k below the full dimension whose confirm-athlete discrimination is not worse than the full embedding's
    by more than ``margin`` (lower bound of the paired interval above −margin) for EVERY training seed. ``paired``:
    ``repr_eval.PAIRED`` of ``<method>|<tag><k>`` against ``<method>|<tag><full>``."""
    full = max(dims)
    for k in sorted(d for d in dims if d < full):
        lows = []
        for m in seed_methods:
            r = fe._one(paired, method=f"{m}|{tag}{k}", reference=f"{m}|{tag}{full}", space="standardised", subset="role:confirm", metric="auc")
            lows.append(r.get("diff_lower") if r else None)
        if lows and all(x is not None and x > -margin for x in lows):
            return k, f"k = {k}: paired AUC difference from the {full}-dimensional embedding has a lower bound above −{margin:g} for all {len(lows)} seed(s)"
    return None, f"no k below {full} keeps the discrimination within {margin:g} of the full embedding for every seed"


# ============================================================================== shortlist
def _f(x: Any, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _scalar_evidence(ch: dict[str, Any], st: dict[str, Any], unit: str) -> dict[str, str]:
    return {"sample_eligibility": f"{ch.get('n_athletes_value')} of {ch.get('n_athletes')} tracked athletes with a value; "
                                  f"{ch.get('n_athletes_repeat')} with a repeat",
            "measurement_reliability": f"within-position ICC {_f(ch.get('reliability'))} [{_f(ch.get('reliability_lower'))}, "
                                       f"{_f(ch.get('reliability_upper'))}]; SEM {_f(ch.get('sem'), 3)} {unit}; rank agreement of two "
                                       f"attempts {_f(ch.get('repeat_spearman'))}",
            "nuisance_sensitivity": f"window R² {_f(ch.get('window_r2'))} [{_f(ch.get('window_r2_lower'))}, {_f(ch.get('window_r2_upper'))}]; "
                                    f"cohort R² {_f(ch.get('cohort_r2'))}; ICC under the strictest quality filter {_f(ch.get('reliability_s3'))} "
                                    f"→ {st.get('acquisition')}",
            "stopwatch_redundancy": f"max |ρ| {_f(ch.get('redundancy_abs'))} [{_f(ch.get('redundancy_lower'))}, {_f(ch.get('redundancy_upper'))}]; "
                                    f"stopwatch R² {_f(ch.get('r2_stopwatch'))}; reliable share not explained "
                                    f"{_f(ch.get('reliable_unique_share'))} → {st.get('stopwatch_relation')}"}


def _vector_evidence(q: dict[str, Any], seed_summary: pl.DataFrame | None = None) -> dict[str, str]:
    def spread(metric: str) -> str:
        r = fe._one(seed_summary, metric=metric) if seed_summary is not None else {}
        return f"; over {r['n_seeds']} seeds {_f(r['mean'], 3)} ± {_f(r['sd_seed'], 3)} (SD between seeds)" if r else ""
    return {"sample_eligibility": f"{q.get('n_confirm')} confirm-role athletes with a repeat (read-out); every tracked athlete is embedded out of fold",
            "measurement_reliability": f"repeat-attempt AUC {_f(q.get('auc'), 3)} [{_f(q.get('auc_lower'), 3)}, {_f(q.get('auc_upper'), 3)}]"
                                       f"{spread('auc')}; trace ICC {_f(q.get('icc_trace'), 3)}; effective rank share {_f(q.get('rank_share_min'))}",
            "nuisance_sensitivity": f"window R² {_f(q.get('window_r2'))} [{_f(q.get('window_r2_lower'))}, {_f(q.get('window_r2_upper'))}]; "
                                    f"cohort balanced-accuracy gain {_f(q.get('cohort_balanced_gain'))} [{_f(q.get('cohort_balanced_gain_lower'))}, "
                                    f"{_f(q.get('cohort_balanced_gain_upper'))}]; largest shift under truncation {_f(q.get('truncation_shift_max'))}",
            "stopwatch_redundancy": f"official forty explained beyond position {_f(q.get('forty_partial_r2'))}; matched-stopwatch AUC "
                                    f"{_f(q.get('matched_auc'), 3)} [{_f(q.get('matched_auc_lower'), 3)}, {_f(q.get('matched_auc_upper'), 3)}]"
                                    f"{spread('matched_auc')}"}


def encoder_readiness(quality: pl.DataFrame, seed_methods: Sequence[str], control_diff: dict[str, Any], c: AtlasConfig) -> tuple[str, list[str]]:
    """(readiness, failed criteria) of the primary encoder from the rows of its training seeds. Criteria, all seeds:
    discrimination interval above chance · matched-stopwatch interval above chance · not flagged collapsed; and on
    average: window share ≤ window_reference, cohort balanced gain ≤ cohort_gain_reference; and matched discrimination
    above the untrained control (paired interval above 0). Combine evidence only."""
    rows = [fe._one(quality, method=m) for m in seed_methods]
    rows = [r for r in rows if r]
    if not rows:
        return "not_ready", ["no evaluated seed"]
    above = lambda r, k: r.get(k) is not None and r[k] > 0.5          # noqa: E731
    mean = lambda k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None   # noqa: E731
    repeatable = all(above(r, "auc_lower") for r in rows)
    matched = all(above(r, "matched_auc_lower") for r in rows)
    failed = []
    if not repeatable:
        failed.append("repeat-attempt discrimination not above chance for every seed")
    if not matched:
        failed.append("matched-stopwatch discrimination not above chance for every seed")
    if not (control_diff and control_diff.get("diff_lower") is not None and control_diff["diff_lower"] > 0):
        failed.append("matched discrimination not shown above the untrained encoder")
    n_col = sum(bool(r.get("collapsed")) for r in rows)
    if n_col:
        failed.append(f"{n_col} of {len(rows)} seed(s) flagged collapsed (effective rank share below the heuristic line)")
    w, g = mean("window_r2"), mean("cohort_balanced_gain")
    if w is None or w > c.window_reference:
        failed.append(f"mean window share {_f(w)} above {c.window_reference:g}")
    if g is None or g > c.cohort_gain_reference:
        failed.append(f"mean cohort balanced-accuracy gain {_f(g)} above {c.cohort_gain_reference:g}")
    if not failed:
        return "ready_for_transfer_test", failed
    return ("carry_with_caveat" if repeatable and matched else "not_ready"), failed


def build_shortlist(status: pl.DataFrame, chars: pl.DataFrame, dictionary: pl.DataFrame, quality: pl.DataFrame,
                    seed_summary: pl.DataFrame, primary: str, seed_methods: Sequence[str], fpca_method: str,
                    compact: tuple[int | None, str], control_diff: dict[str, Any], c: AtlasConfig, config_hash: str,
                    preprocessing: str, paths: dict[str, str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(shortlist, exploratory registry). Seven pre-declared slots; a slot is filled by RULE from saved Combine
    evidence and stays in the table, unselected, when nothing meets the rule. No NFL value is an input.

    C1 traditional baseline   official forty + 10-yard split (+ weight, height): the comparator, always carried
    C2 speed control          the tracked speed feature with the highest reliability lower bound (graded ≥ Provisional)
    C3 acceleration           the interpretable acceleration feature (F6, F2) with the highest reliability lower bound
    C4 sprint-curve parameter the fitted parameter (not extrapolated, not redundant) with the highest lower bound
    C5 functional PCA         the benchmark K scores of the primary input, as one vector
    C6 primary encoder        TCN-B 8-d, the PRE-DECLARED first training seed (never the best seed); selected unless not_ready
    C7 compact projection     the smallest training-fitted projection that keeps the discrimination (``compact_dimension``)"""
    spec = {r["feature_id"]: r for r in dictionary.iter_rows(named=True)}
    ch = {r["feature_id"]: r for r in chars.filter(pl.col("scope") == "all").iter_rows(named=True)}
    st = {r["feature_id"]: r for r in status.iter_rows(named=True)}
    usable = status.filter(pl.col("grade").is_in(list(GRADES[:2])) & ~pl.col("feature_id").str.contains(":", literal=True))
    now = utc_now()
    common = {"config_hash": config_hash, "shortlist_version": "v1", "frozen_at": now, "preprocessing_version": preprocessing}
    rows: list[dict[str, Any]] = []

    def best(pool: pl.DataFrame) -> str | None:
        p = pool.filter(pl.col("reliability_lower").is_not_null()).sort(["reliability_lower", "feature_id"], descending=[True, False])
        return p["feature_id"][0] if p.height else None

    def scalar_row(cid: str, group: str, role: str, fid: str | None, rationale: str, limits: str, fallback: pl.DataFrame) -> None:
        if fid is None:
            alt = best(fallback)
            rows.append({"candidate_id": cid, "group": group, "role": role, "selected": False, "readiness": "not_ready", "features": "",
                         "selection_rationale": "no feature met the rule (" + rationale + ")"
                                                + (f"; best exploratory alternative: {alt} ({st[alt]['grade']})" if alt else ""),
                         "known_limitations": limits, "artifact_paths": paths.get("athlete_features", ""), **common})
            return
        s = spec[fid]
        rows.append({"candidate_id": cid, "group": group, "role": role, "selected": True,
                     "readiness": "ready_for_transfer_test" if st[fid]["grade"] == GRADES[0] else "carry_with_caveat", "features": fid,
                     "definition": s["definition"], "unit_dimensionality": f"{s['units']} · 1 value per athlete (mean of attempts)",
                     **_scalar_evidence(ch.get(fid, {}), st[fid], s["units"]),
                     "selection_rationale": f"{rationale}; grade {st[fid]['grade']} ({st[fid]['reason']})", "known_limitations": limits,
                     "artifact_paths": paths.get("athlete_features", ""), **common})

    cand = usable.filter(pl.col("role") == "candidate")
    every = status.filter((pl.col("role") == "candidate") & ~pl.col("feature_id").str.contains(":", literal=True))
    is_speed = pl.col("feature_id").str.contains("speed", literal=True) & ~pl.col("feature_id").str.contains("speed_gain", literal=True)
    # C1 -------------------------------------------------------------------------------------------------------------
    base_ids = [f"F1.official.{v}" for v in (FORTY, TEN, WEIGHT, HEIGHT)]
    have = [f for f in base_ids if f in st and st[f]["grade"] != GRADES[3]]
    cov = "; ".join(f"{spec[f]['feature_name']} {_f(st[f]['coverage_athletes'])}" for f in have)
    rows.append({"candidate_id": "C1_traditional_baseline", "group": "traditional Combine baseline", "role": "reference",
                 "selected": all(f in have for f in base_ids[:2]), "readiness": "reference", "features": ";".join(have),
                 "definition": "official forty-yard dash and 10-yard split, with Combine weight and height",
                 "unit_dimensionality": f"s, s, lb, in · {len(have)} values per athlete",
                 "sample_eligibility": f"share of tracked athletes with a value — {cov}",
                 "measurement_reliability": "not estimable here: one official value per athlete",
                 "nuisance_sensitivity": "not applicable (hand or electronic timing, not the tracking recording)",
                 "stopwatch_redundancy": "this IS the stopwatch: every other family is judged by what it adds to it",
                 "selection_rationale": "comparator of the transfer study; carried by definition, not by a screening rule",
                 "known_limitations": "official timing protocol differs from the tracked recording; height may be missing or not numeric",
                 "artifact_paths": paths.get("athlete_features", ""), **common})
    # C2–C4 ----------------------------------------------------------------------------------------------------------
    speed = cand.filter(pl.col("family").is_in(["F2"]) & is_speed)
    scalar_row("C2_speed_control", "reliable speed control", "control", best(speed),
               "tracked speed feature with the highest reliability lower bound among those graded Strong or Provisional",
               "expected to be redundant with the official forty: it controls for 'tracking measures the stopwatch again'",
               every.filter(pl.col("family") == "F2"))
    accel = cand.filter(pl.col("family").is_in(["F6", "F2"]) & ~is_speed & ~pl.col("feature_id").str.contains("decel", literal=True)
                        & pl.col("definition").str.starts_with("interpretable"))
    scalar_row("C3_acceleration", "physically interpretable acceleration", "candidate", best(accel),
               "interpretable acceleration feature (sustained or direct) with the highest reliability lower bound, graded Strong or Provisional",
               "measured on the recorded part of the run only: the first steps are not in the recording; repeatability differs by position and cohort",
               every.filter(pl.col("family").is_in(["F6", "F2"]) & ~is_speed))
    model = cand.filter((pl.col("family") == "F4") & (pl.col("stopwatch_relation") != "redundant")
                        & ~pl.col("definition").str.starts_with("extrapolated"))
    scalar_row("C4_sprint_parameter", "sprint-curve parameter", "candidate", best(model),
               "fitted parameter of the existing exponential sprint model, not extrapolated and not redundant, with the highest reliability lower bound",
               "defined only for attempts whose fit passed its flags; a good curve fit does not make the parameter identifiable; the "
               "recording starts after the first movement", every.filter(pl.col("family") == "F4"))
    # C5–C7 ----------------------------------------------------------------------------------------------------------
    def vector_row(cid: str, group: str, role: str, method: str, definition: str, dims: str, selected: bool, readiness: str,
                   rationale: str, limits: str, path: str, with_seeds: bool = False) -> None:
        q = fe._one(quality, method=method)
        rows.append({"candidate_id": cid, "group": group, "role": role, "selected": bool(selected and q), "readiness": readiness if q else "not_ready",
                     "features": method, "definition": definition, "unit_dimensionality": dims,
                     **(_vector_evidence(q, seed_summary if with_seeds else None) if q else {}),
                     "selection_rationale": rationale if q else f"{method} was not evaluated in this run", "known_limitations": limits,
                     "artifact_paths": path, **common})

    vector_row("C5_fpca", "functional PCA benchmark", "learned benchmark", fpca_method,
               "out-of-fold functional PCA scores of speed against recorded distance, benchmark K", "K scores per attempt; athlete = mean",
               True, "carry_with_caveat", "low-complexity learned benchmark every encoder is compared with; carried by design",
               "its first component is largely the stopwatch; later components are less repeatable or follow the recording window",
               paths.get("fpca", ""))
    readiness, failed = encoder_readiness(quality, seed_methods, control_diff, c)
    vector_row("C6_tcn_b_8d", "TCN-B 8-d embedding", "candidate", primary,
               "encoder of speed against recorded distance trained by masked reconstruction and same-athlete attempt consistency; "
               "8 numbers evaluated as ONE vector", "8-d vector per attempt; athlete = mean of his out-of-fold attempt embeddings",
               readiness != "not_ready", readiness,
               "primary representation of notebook 03, frozen at its PRE-DECLARED first training seed; the other seeds are robustness "
               "replicates on which the transfer result must also be reported",
               ("criteria not met: " + "; ".join(failed) + ". " if failed else "") + "coordinates are not comparable between folds or seeds; "
               "linear residualisation leaves any nonlinear function of the official times; no physical interpretation; NFL value untested",
               paths.get("tcn", ""), with_seeds=True)
    k, why = compact
    name = f"{primary}|pc{k}" if k else ""
    if k:
        vector_row("C7_tcn_compact", "compact TCN projection", "candidate (optional)", name,
                   f"first {k} principal axes of the fold's training embeddings of {primary}, applied to every attempt of the fold",
                   f"{k}-d vector per attempt", readiness != "not_ready", readiness, why,
                   "a projection of C6: not an independent candidate; axes are fold-specific", paths.get("tcn", ""))
    else:
        rows.append({"candidate_id": "C7_tcn_compact", "group": "compact TCN projection", "role": "candidate (optional)",
                     "selected": False, "readiness": "not_ready", "features": "", "selection_rationale": why,
                     "known_limitations": "optional slot left empty", "artifact_paths": "", **common})
    shortlist = _table(rows, SHORTLIST)
    chosen = {f for r in rows if r["selected"] for f in r["features"].split(";") if f}
    chosen |= {f"F5.vector.{f}" for f in chosen}
    rest = status.filter(~pl.col("feature_id").str.contains(":", literal=True) & pl.col("role").is_in(["candidate"])
                         & ~pl.col("feature_id").is_in(sorted(chosen)) & (pl.col("transfer_eligibility") != "not_eligible"))
    return shortlist, _table(rest.select(list(EXPLORATORY.columns)).to_dicts(), EXPLORATORY)


def conclusions_markdown(quality: pl.DataFrame, seed_summary: pl.DataFrame, paired: pl.DataFrame, audit: pl.DataFrame,
                         shortlist: pl.DataFrame, freeze_state: str, primary: str, references: dict[str, str],
                         compact: tuple[int | None, str], ecfg: EvalConfig, run_mode: str) -> str:
    """Text of the conclusions cell of notebook 04B, written from the saved tables only. ``references``: label → method
    the primary encoder is compared with (``paired``: MATCH_PAIRED)."""
    lines = [f"_Generated from the `robust_*` tables and `transfer_candidates_v1` of this run (`RUN_MODE = {run_mode!r}`). Read-out: "
             "confirm-role held-out athletes, within position, 95 % athlete intervals. Labels are heuristic._", ""]
    if run_mode == "smoke":
        lines += ["**Smoke run: a technical check on a few athletes and epochs. The numbers below must not be interpreted, and the "
                  "shortlist written under `_sample` is not the frozen shortlist.**", ""]
    lines.append("**RQ5 — does the TCN-B result survive training seeds and matched impostors?**")
    for metric, label in (("auc", "repeat-attempt AUC"), ("matched_auc", "matched-stopwatch AUC"), ("icc_trace", "trace reliability"),
                          ("rank_share_min", "effective rank / dimension (smallest fold)"), ("resid_auc", "cross-fitted residual AUC"),
                          ("window_r2", "attempt change explained by the window"), ("cohort_balanced_gain", "cohort balanced-accuracy gain")):
        s = fe._one(seed_summary, metric=metric)
        if s:
            lines.append(f"- {label}: mean {_f(s['mean'], 3)} over {s['n_seeds']} seed(s), SD between seeds {_f(s['sd_seed'], 3)}, range "
                         f"[{_f(s['min'], 3)}, {_f(s['max'], 3)}]; athlete standard error {_f(s['se_athlete'], 3)}.")
    for label, ref in references.items():
        r = fe._one(paired, method=primary, reference=ref, subset="role:confirm")
        if r:
            lines.append(f"- Matched discrimination, `{primary}` − {label}: {r['difference']:+.3f} [{_f(r['diff_lower'], 3)}, "
                         f"{_f(r['diff_upper'], 3)}] on {r['n_athletes']} confirm athletes with a matched impostor.")
    col = quality.filter(pl.col("kind").is_in(["primary", "seed replicate"]))
    if col.height:
        lines.append(f"- Collapse flag (effective rank share below {ecfg.collapse_rank_share:g}): {int(col['collapsed'].sum())} of "
                     f"{col.height} seed(s).")
    lines += ["", "**Why did the residual AUC of notebook 03 exceed the raw AUC?**", audit_reading(audit, primary, ecfg), "",
              f"**Effective dimension.** {compact[1]}.", "", f"**RQ6 — frozen shortlist** ({freeze_state})"]
    for r in shortlist.sort("candidate_id").iter_rows(named=True):
        lines.append(f"- `{r['candidate_id']}` · {r['group']}: **{'selected' if r['selected'] else 'not selected'}** ({r['readiness']}) — "
                     f"{r['features'] or 'no feature'}. {r['selection_rationale'] or ''}")
    lines += ["", "**What is and is not established.** A selected family exists, is measurable on the stated athletes, and has the "
                  "repeatability, stopwatch relation and nuisance dependence written in its row. `ready_for_transfer_test` means the "
                  "Combine-side evidence justifies TESTING it against NFL movement; it does not mean it predicts anything. "
                  "**Demonstrated NFL predictive value: NOT TESTED in this milestone** — no NFL outcome was read, and a "
                  "same-athlete retrieval metric does not measure football ability. The eight coordinates of the encoder are one "
                  "vector, not eight validated physiological measurements."]
    return "\n".join(lines)


def freeze_shortlist(store: Any, shortlist: pl.DataFrame, exploratory: pl.DataFrame, provenance: dict[str, Any],
                     refreeze: bool = False) -> tuple[pl.DataFrame, str]:
    """Write the shortlist ONCE. An existing ``transfer_candidates_v1`` of this run mode is returned unchanged unless
    ``refreeze`` is set; a difference from what the current run would produce is reported, never applied silently.
    The exploratory registry is not frozen and is rewritten. ``store``: ``AtlasStore``."""
    if not store.controls.save_tables:
        return shortlist, "NOT WRITTEN (SAVE_TABLES is off)"
    store.write(exploratory, EXPLORATORY, "ATLAS_FREEZE", provenance)
    cols = [c for c in SHORTLIST.columns if c != "frozen_at"]
    if store.exists(SHORTLIST) and not refreeze:
        old = store.read(SHORTLIST)
        same = old.select(cols).sort("candidate_id").equals(shortlist.select(cols).sort("candidate_id"))
        return old, ("already frozen on " + str(old["frozen_at"][0]) + ("; the current run reproduces it" if same else
                     "; THE CURRENT RUN WOULD PRODUCE A DIFFERENT SHORTLIST — kept as frozen. Set REFREEZE_SHORTLIST = True only "
                     "if no transfer analysis has used it yet, and record why"))
    out = store.write(shortlist, SHORTLIST, "ATLAS_FREEZE", provenance)
    store.write_json("transfer_candidates_v1_freeze", {"config_hash": shortlist["config_hash"][0], "frozen_at": shortlist["frozen_at"][0],
                                                       "refrozen": bool(refreeze), "provenance": provenance,
                                                       "selected": shortlist.filter(pl.col("selected"))["candidate_id"].to_list()})
    return out, "frozen now" + (" (REFREEZE)" if refreeze else "")
