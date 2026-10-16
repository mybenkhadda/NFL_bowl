"""Robustness, matched prospects and scoring readiness of the transfer study (Milestone 06C). NumPy / Polars only.

Everything here READS out-of-fold predictions produced by ``transfer_models`` — pooled, regularised models with additive
position effects — or re-runs that same loop under a declared variation (exposure rule, context, cohort held out, seed).
No threshold, subgroup, window or pair is chosen by looking at a result: every grid is in ``TransferConfig`` and every
row of every grid is reported.

    subgroup_tables      pooled fit, subgroup scoring (position, draft cohort) + a permutation test of heterogeneity
    sensitivity_plan     the declared exposure / context variations of a frozen target
    matched_pairs        prospects of the same position with near-identical official forty, 10-yard split and body size
                         (rule and scales of notebook 04B, fixed before any NFL value) → do models order their NFL
                         movement difference correctly? Athlete-cluster bootstrap, because pairs reuse athletes.
    mechanism_table      frozen physical features against frozen NFL targets after official times and position
    evidence_table       one row per scoring dimension: every piece of evidence side by side
    readiness_registry   ``scoring_readiness_registry``: conservative, rule-based status per dimension
    decision_case        case A–E of the pre-declared decision tree
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import get_logger
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from . import robust as rb
from . import transfer as tr
from . import transfer_models as tm
from .fpca_eval import FORTY, HEIGHT, TEN, WEIGHT, Meta
from .robust import MatchConfig
from .transfer import FrozenTarget, TransferConfig
from .transfer_models import PRIMARY, REFERENCE, Design

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
STATUSES = ("descriptive_ready", "exploratory", "predictive_candidate", "predictive_validated", "insufficient_evidence", "rejected")
DIMENSIONS = (
    # name, model, reference, shortlist slot, physical interpretation
    ("official_combine_tests", "M1", "M0", "C1_traditional_baseline",
     "official stopwatch and jump results with body size: what scouting already has"),
    ("tracked_top_speed", "M3", "M1", "C2_speed_control", "top speed reached in the tracked forty (p95 and interior peak speed)"),
    ("sustained_acceleration", "M2", "M1", "C3_acceleration", "speed development over a fixed speed interval of the tracked forty"),
    ("fpca_speed_curve", "M4", "M1", "C5_fpca", "main modes of variation of the speed-by-distance curve (level, shape)"),
    ("tcn_b_movement_fingerprint", "M5", "M1", "C6_tcn_b_8d",
     "learned 8-d summary of the speed-by-distance curve; no physical meaning is attached to a coordinate"),
    ("tcn_b_compact", "M7_PC{k}", "M1", "C7_tcn_compact", "the same summary on its leading training axes"),
)

_K = {"analysis": S, "cohort": S, "target_id": S}
SUB_PERF = TableContract(
    "transfer_subgroup_performance", "Held-out performance by subgroup (pooled fit, subgroup scoring).",
    {**_K, "group_type": S, "group": S, "model": S, "metric": S, "estimate": F, "ci_lower": F, "ci_upper": F, "n_athletes": I,
     "n_positions": I, "n_cohorts": I}, key=("analysis", "cohort", "target_id", "group_type", "group", "model", "metric"),
    area="analysis × target × subgroup × model × metric",
    docs={"estimate": "R² of a subgroup is measured against the pooled training-fold mean: read RMSE / MAE first"})
SUB_PAIRED = TableContract(
    "transfer_subgroup_comparisons", "Paired model differences by subgroup (positive = the model is better).",
    {**_K, "group_type": S, "group": S, "model": S, "reference": S, "metric": S, "estimate": F, "ci_lower": F, "ci_upper": F,
     "relative": F, "share_boot_improved": F, "n_athletes": I, "established": B, "worse": B},
    key=("analysis", "cohort", "target_id", "group_type", "group", "model", "reference", "metric"),
    area="analysis × target × subgroup × model pair × metric")
HETERO = TableContract(
    "transfer_gain_heterogeneity", "Does the gain of a model over its reference differ between subgroups more than chance allows?",
    {**_K, "group_type": S, "model": S, "reference": S, "n_groups": I, "n_athletes": I, "statistic": F, "p_value": F, "n_perm": I},
    key=("analysis", "cohort", "target_id", "group_type", "model", "reference"), area="analysis × target × grouping × model pair",
    docs={"statistic": "size-weighted variance between subgroups of the mean squared-error difference (reference − model)",
          "p_value": "permutation of the subgroup labels between athletes; small = the gain varies between subgroups"})
PAIRS = TableContract(
    "transfer_matched_pairs", "Matched prospects: same position, near-identical official forty, 10-yard split and body size.",
    {"pair_id": I, "nfl_id_a": I, "nfl_id_b": I, "position_group": S, "distance": F, "d_forty": F, "d_ten": F, "d_weight": F,
     "d_height": F, "same_fold": B, "same_cohort": B}, key=("pair_id",), area="athlete pair",
    docs={"distance": "RMS difference over the matching variables, in design-athlete SD units (≤ the caliper)",
          "d_forty": "athlete b − athlete a, raw units"})
PAIR_PRED = TableContract(
    "transfer_matched_pair_predictions", "Actual and out-of-fold predicted NFL movement difference of every matched pair.",
    {"target_id": S, "pair_id": I, "model": S, "d_actual": F, "d_pred": F}, key=("target_id", "pair_id", "model"), kind="dataset",
    area="target × pair × model", docs={"d_actual": "label of b − label of a", "d_pred": "out-of-fold prediction of b − of a"})
PAIR_SUMMARY = TableContract(
    "transfer_matched_pair_summary", "Do models order the NFL movement difference of matched prospects correctly?",
    {"target_id": S, "model": S, "n_pairs": I, "n_athletes": I, "max_reuse": I, "mean_reuse": F, "concordance": F,
     "concordance_lower": F, "concordance_upper": F, "corr": F, "corr_lower": F, "corr_upper": F, "rmse_diff": F,
     "d_concordance": F, "d_concordance_lower": F, "d_concordance_upper": F, "d_corr": F, "d_corr_lower": F, "d_corr_upper": F},
    key=("target_id", "model"), area="target × model",
    docs={"concordance": "share of pairs whose predicted difference has the sign of the actual difference (0.5 = chance)",
          "d_concordance": "concordance − concordance of the official-only model M1, same pairs, athlete-cluster bootstrap",
          "max_reuse": "largest number of pairs one athlete belongs to"})
MECH = TableContract(
    "transfer_physical_mechanism", "Frozen physical Combine features against frozen NFL targets, within position and after official times.",
    {"feature_id": S, "target_id": S, "n_athletes": I, "spearman_within_position": F, "partial_r": F, "partial_lower": F,
     "partial_upper": F, "covariates": S, "note": S}, key=("feature_id", "target_id"), area="feature × target",
    docs={"partial_r": "correlation of the residuals of feature and label after position, draft cohort and the covariates "
                       "(in-sample, descriptive; the held-out answer is model M2 vs M1)"})
STABILITY = TableContract(
    "transfer_ci_stability", "Bootstrap interval of the primary comparison across replicate budgets and bootstrap seeds.",
    {"target_id": S, "model": S, "reference": S, "n_boot": I, "boot_seed": I, "estimate": F, "ci_lower": F, "ci_upper": F},
    key=("target_id", "model", "reference", "n_boot", "boot_seed"), area="target × model pair × budget × seed")
EVIDENCE = TableContract(
    "transfer_evidence_summary", "Every piece of transfer evidence per scoring dimension, side by side (primary target).",
    {"dimension": S, "model": S, "reference": S, "target_id": S, "n_athletes": I, "delta_rmse": F, "ci_lower": F, "ci_upper": F,
     "relative": F, "established": B, "worse": B, "vs_random_lower": F, "loco_delta": F, "loco_lower": F, "loco_upper": F,
     "cohorts_positive": I, "cohorts_total": I, "positions_positive": I, "positions_total": I, "seeds_positive": I, "seeds_total": I,
     "perm_p_outcome": F, "perm_p_linkage": F, "matched_d_concordance": F, "matched_lower": F, "matched_upper": F,
     "heterogeneity_p_position": F}, key=("dimension",), area="scoring dimension")
READINESS = TableContract(
    "scoring_readiness_registry", "Readiness of every potential scoring dimension for the next (scoring) milestone.",
    {"name": S, "candidate_feature_source": S, "physical_interpretation": S, "reliability_evidence": S, "stopwatch_redundancy": S,
     "nfl_transfer_evidence": S, "effect_size": F, "confidence_interval": S, "cohort_robustness": S, "position_applicability": S,
     "measurement_coverage": S, "status": S, "status_reason": S}, key=("name",), area="scoring dimension",
    docs={"effect_size": "Δ RMSE against the reference of the dimension on the PRIMARY frozen target (positive = lower error)",
          "status": " · ".join(STATUSES)})


# ============================================================================== subgroups
def _with_group(df: pl.DataFrame, group_type: str, group: str, contract: TableContract) -> pl.DataFrame:
    return df.with_columns(pl.lit(group_type).alias("group_type"), pl.lit(str(group)).alias("group")).select(list(contract.columns))


def subgroup_tables(oof: pl.DataFrame, by: str, pairs: Sequence[tuple[str, str]], tc: TransferConfig, seed: int) -> dict[str, pl.DataFrame]:
    """{SUB_PERF, SUB_PAIRED}: the POOLED out-of-fold predictions scored inside each level of ``by``. A level with fewer
    than ``min_athletes_group`` athletes is reported without an interval (``evaluate`` rule), never dropped silently."""
    perf, paired = [], []
    for g in sorted(oof[by].drop_nulls().unique().to_list(), key=str):
        ev = tm.evaluate(oof.filter(pl.col(by) == g), pairs, tc, seed_for(seed, "subgroup", by, g))
        perf.append(_with_group(ev[tm.PERF.name], by, g, SUB_PERF))
        paired.append(_with_group(ev[tm.PAIRED.name], by, g, SUB_PAIRED))
    return {SUB_PERF.name: pl.concat(perf) if perf else SUB_PERF.empty(), SUB_PAIRED.name: pl.concat(paired) if paired else SUB_PAIRED.empty()}


def heterogeneity(oof: pl.DataFrame, by: str, pairs: Sequence[tuple[str, str]], tc: TransferConfig, seed: int) -> pl.DataFrame:
    """Permutation test that the squared-error gain of a model over its reference is the same in every subgroup.
    Distinguishes apparent variation between positions from what athlete sampling alone produces."""
    rows = []
    for key, o in sorted(oof.partition_by(["analysis", "cohort", "target_id"], as_dict=True).items()):
        a = tm.paired_arrays(o)
        labels = a["position"] if by == "position_group" else a["cohort"]
        groups, code = np.unique(labels, return_inverse=True)
        n = np.bincount(code).astype(float)
        if groups.size < 2 or n.min() < 2:
            continue
        for m, r in pairs:
            if m not in a["pred"] or r not in a["pred"]:
                continue
            gain = (a["y"] - a["pred"][r]) ** 2 - (a["y"] - a["pred"][m]) ** 2

            def stat(c: np.ndarray) -> float:
                means = np.bincount(c, weights=gain, minlength=groups.size) / n
                return float((n * (means - gain.mean()) ** 2).sum() / n.sum())
            obs = stat(code)
            rng = np.random.default_rng(seed_for(seed, "hetero", by, *key, m, r))
            null = np.array([stat(rng.permutation(code)) for _ in range(tc.n_perm)])
            rows.append({**dict(zip(("analysis", "cohort", "target_id"), key)), "group_type": by, "model": m, "reference": r,
                         "n_groups": int(groups.size), "n_athletes": int(code.size), "statistic": obs,
                         "p_value": float((1 + (null >= obs).sum()) / (1 + null.size)), "n_perm": int(null.size)})
    return pl.DataFrame(rows, schema=HETERO.columns) if rows else HETERO.empty()


# ============================================================================== declared variations
def sensitivity_plan(target: FrozenTarget, tc: TransferConfig) -> list[dict[str, Any]]:
    """Every exposure and context variation of one frozen target, declared before a result exists. The frozen rule
    itself is the row ``exposure:min_windows:<frozen minimum>`` / ``context:<frozen adjustment>``."""
    m = target.min_windows
    plan = [{"analysis": f"exposure:min_windows:{w}", "variant": "all", "adjustment": None, "min_windows": int(w), "min_games": 0,
             "x": float(w), "kind": "min_windows"} for w in sorted({*tc.exposure_min_windows, m})]
    plan += [{"analysis": f"exposure:first_n:{k}", "variant": f"first_n:{k}", "adjustment": None, "min_windows": int(k), "min_games": 0,
              "x": float(k), "kind": "first_n"} for k in tc.first_n_windows]
    plan += [{"analysis": f"exposure:min_games:{g}", "variant": "all", "adjustment": None, "min_windows": m, "min_games": int(g),
              "x": float(g), "kind": "min_games"} for g in tc.min_games]
    plan += [{"analysis": f"exposure:horizon:{h}", "variant": f"horizon:{h}", "adjustment": None, "min_windows": m, "min_games": 0,
              "x": float(h), "kind": "horizon"} for h in tc.horizon_seasons]
    plan += [{"analysis": f"context:{label}", "variant": "all", "adjustment": adj, "min_windows": m, "min_games": 0, "x": NAN,
              "kind": "context"} for label, adj in (("raw", "none"), ("cell_adjusted", "cell_residual"))]
    plan += [{"analysis": f"context:class:{c}", "variant": f"class:{c}", "adjustment": None, "min_windows": m, "min_games": 0, "x": NAN,
              "kind": "context"} for c in tc.play_classes]
    return plan


def loco_design(d: Design, fold_cohort: dict[int, str]) -> Design:
    """The same athletes with the outer fold = draft cohort (codes of ``robust.cohort_splits``, so the fold index is the
    one the leave-one-cohort-out encoders were fitted under)."""
    code = {str(c): int(f) for f, c in fold_cohort.items()}
    missing = sorted(set(d.cohort.tolist()) - set(code))
    if missing:
        raise tm.CohortError(f"draft cohort(s) {missing} of the evaluation cohort are absent from the encoder split table")
    return d.with_folds(np.array([code[c] for c in d.cohort], dtype=np.int64))


# ============================================================================== matched prospects
def matched_pairs(d: Design, meta: Meta, scales: dict[str, float], mc: MatchConfig) -> pl.DataFrame:
    """Unordered pairs (a, b) of the evaluation cohort such that one is among the ≤ k nearest same-position athletes of
    the other inside the caliper (``robust.match_matrix`` — the frozen rule of notebook 04B). No NFL value is read."""
    M, D, diffs = rb.match_matrix(d.ids, d.position, meta, scales, mc)
    either = np.triu(M | M.T, k=1)
    ia, ib = np.nonzero(either)
    order = np.lexsort((d.ids[ib], d.ids[ia]))
    ia, ib = ia[order], ib[order]
    return pl.DataFrame({"pair_id": np.arange(ia.size, dtype=np.int64), "nfl_id_a": d.ids[ia], "nfl_id_b": d.ids[ib],
                         "position_group": d.position[ia], "distance": D[ia, ib], "d_forty": diffs[FORTY][ia, ib],
                         "d_ten": diffs[TEN][ia, ib], "d_weight": diffs[WEIGHT][ia, ib], "d_height": diffs[HEIGHT][ia, ib],
                         "same_fold": d.fold[ia] == d.fold[ib], "same_cohort": d.cohort[ia] == d.cohort[ib]},
                        schema=PAIRS.columns).with_columns(pl.col("d_height").fill_nan(None))


def pair_predictions(pairs: pl.DataFrame, oof: pl.DataFrame) -> pl.DataFrame:
    """Actual and out-of-fold predicted difference (b − a) of every pair, model and target of ``oof`` (one analysis)."""
    v = oof.select("target_id", "model", "nfl_id", "y", "pred")
    j = (pairs.select("pair_id", "nfl_id_a", "nfl_id_b")
         .join(v.rename({"nfl_id": "nfl_id_a", "y": "ya", "pred": "pa"}), on="nfl_id_a", how="inner")
         .join(v.rename({"nfl_id": "nfl_id_b", "y": "yb", "pred": "pb"}), on=["nfl_id_b", "target_id", "model"], how="inner"))
    return j.select("target_id", "pair_id", "model", (pl.col("yb") - pl.col("ya")).alias("d_actual"),
                    (pl.col("pb") - pl.col("pa")).alias("d_pred")).sort("target_id", "pair_id", "model")


def _wcorr(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sw = W.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = (W * x).sum(1) / sw, (W * y).sum(1) / sw
        cx, cy = x[None, :] - mx[:, None], y[None, :] - my[:, None]
        den = np.sqrt((W * cx ** 2).sum(1) * (W * cy ** 2).sum(1))
        return np.where(den > 1e-12, (W * cx * cy).sum(1) / den, np.nan)


def pair_summary(pairs: pl.DataFrame, pred: pl.DataFrame, tc: TransferConfig, seed: int, reference: str = REFERENCE) -> pl.DataFrame:
    """Concordance and correlation of predicted with actual pair differences, per target and model, with an
    athlete-cluster bootstrap (athletes are resampled; a pair counts the product of its two athletes' multiplicities).
    Differences against the official-only model use the same resamples."""
    rows = []
    for (target,), g in sorted(pred.partition_by(["target_id"], as_dict=True).items()):
        ids = g["pair_id"].unique().sort().to_list()
        pp = pairs.filter(pl.col("pair_id").is_in(ids)).sort("pair_id")
        if pp.height < 5:
            continue
        athletes, inv = np.unique(np.r_[pp["nfl_id_a"].to_numpy(), pp["nfl_id_b"].to_numpy()], return_inverse=True)
        ia, ib = inv[:pp.height], inv[pp.height:]
        reuse = np.bincount(inv, minlength=athletes.size)
        counts = np.random.default_rng(seed_for(seed, "pairs", target)).multinomial(athletes.size, np.full(athletes.size, 1.0 / athletes.size),
                                                                                     size=tc.n_boot).astype(float)
        W = counts[:, ia] * counts[:, ib]
        ones = np.ones((1, pp.height))
        stats: dict[str, dict[str, Any]] = {}
        for (model,), gm in g.partition_by(["model"], as_dict=True).items():
            gm = gm.sort("pair_id")
            da, dp = gm["d_actual"].to_numpy().astype(float), gm["d_pred"].to_numpy().astype(float)
            agree = (np.sign(da) == np.sign(dp)).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                conc_b = (W * agree).sum(1) / W.sum(1)
            stats[str(model)] = {"conc": float(agree.mean()), "conc_b": conc_b, "corr": float(_wcorr(ones, da, dp)[0]),
                                 "corr_b": _wcorr(W, da, dp), "rmse": float(np.sqrt(np.mean((da - dp) ** 2)))}
        ref = stats.get(reference)
        for model, s in sorted(stats.items()):
            lo, hi = tm._interval(s["conc_b"])
            clo, chi = tm._interval(s["corr_b"])
            row = {"target_id": target, "model": model, "n_pairs": pp.height, "n_athletes": int(athletes.size), "max_reuse": int(reuse.max()),
                   "mean_reuse": float(reuse.mean()), "concordance": s["conc"], "concordance_lower": lo, "concordance_upper": hi,
                   "corr": s["corr"] if np.isfinite(s["corr"]) else None, "corr_lower": clo, "corr_upper": chi, "rmse_diff": s["rmse"],
                   **{k: None for k in ("d_concordance", "d_concordance_lower", "d_concordance_upper", "d_corr", "d_corr_lower", "d_corr_upper")}}
            if ref is not None and model != reference:
                dlo, dhi = tm._interval(s["conc_b"] - ref["conc_b"])
                rlo, rhi = tm._interval(s["corr_b"] - ref["corr_b"])
                dc = s["corr"] - ref["corr"]
                row.update(d_concordance=s["conc"] - ref["conc"], d_concordance_lower=dlo, d_concordance_upper=dhi,
                           d_corr=dc if np.isfinite(dc) else None, d_corr_lower=rlo, d_corr_upper=rhi)
            rows.append(row)
    return pl.DataFrame(rows, schema=PAIR_SUMMARY.columns) if rows else PAIR_SUMMARY.empty()


def case_studies(pairs: pl.DataFrame, pred: pl.DataFrame, athletes: pl.DataFrame, target_id: str, models: Sequence[str], k: int
                 ) -> pl.DataFrame:
    """The ``k`` pairs with the SMALLEST stopwatch-and-body distance (a rule that reads no NFL value and no prediction),
    with their official results, predicted and actual differences. ILLUSTRATIVE ONLY: a handful of pairs is not evidence."""
    top = pairs.sort("distance", "pair_id").head(k)
    names = athletes.select("nfl_id", "display_name", FORTY, TEN, WEIGHT)
    out = (top.join(names.rename({"nfl_id": "nfl_id_a", "display_name": "athlete_a", FORTY: "forty_a", TEN: "ten_a", WEIGHT: "weight_a"}),
                    on="nfl_id_a", how="left")
           .join(names.rename({"nfl_id": "nfl_id_b", "display_name": "athlete_b", FORTY: "forty_b", TEN: "ten_b", WEIGHT: "weight_b"}),
                 on="nfl_id_b", how="left"))
    p = pred.filter((pl.col("target_id") == target_id) & pl.col("model").is_in(list(models)))
    wide = p.pivot(on="model", index="pair_id", values="d_pred").rename({m: f"predicted_diff_{m}" for m in models if m in p["model"].unique().to_list()})
    actual = p.group_by("pair_id").agg(pl.col("d_actual").first().alias("actual_diff"))
    return (out.join(wide, on="pair_id", how="left").join(actual, on="pair_id", how="left")
            .with_columns(pl.lit("illustrative only").alias("status")).sort("distance", "pair_id")
            .select("pair_id", "position_group", "athlete_a", "athlete_b", "forty_a", "forty_b", "ten_a", "ten_b", "weight_a",
                    "weight_b", "distance", pl.col("^predicted_diff_.*$"), "actual_diff", "status"))


# ============================================================================== physical mechanism
def _residual(v: np.ndarray, X: np.ndarray) -> np.ndarray:
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def mechanism_table(features: dict[str, np.ndarray], labels: dict[str, np.ndarray], d: Design, covariates: Sequence[str],
                    tc: TransferConfig, seed: int, notes: dict[tuple[str, str], str] | None = None) -> pl.DataFrame:
    """``features``: feature id → athlete values aligned to ``d.ids`` (frozen physical features, athlete = mean of
    attempts). ``labels``: target id → DESCRIPTIVE athlete label aligned to ``d.ids``. Position and draft-cohort dummies
    and ``covariates`` (official times, weight) are removed from both by least squares; the bootstrap refits them in
    every athlete resample. In-sample and descriptive — the incremental held-out value is M2 vs M1."""
    dummies = [(d.position == p).astype(float) for p in np.unique(d.position)[1:]] + [(d.cohort == c).astype(float) for c in np.unique(d.cohort)[1:]]
    cov = [np.where(np.isfinite(d.numeric[c]), d.numeric[c], np.nanmedian(d.numeric[c])) for c in covariates]
    X = np.column_stack([np.ones(d.n), *dummies, *cov])
    pos_code = np.unique(d.position, return_inverse=True)[1]
    rows = []
    for fid, f in features.items():
        for tid, y in labels.items():
            ok = np.isfinite(f) & np.isfinite(y)
            n = int(ok.sum())
            if n < tc.min_athletes:
                continue
            fo, yo, Xo, pc = f[ok], y[ok], X[ok], pos_code[ok]
            cnt = np.bincount(pc).astype(float)
            centre = lambda v: v - (np.bincount(pc, weights=v) / np.maximum(cnt, 1))[pc]            # noqa: E731
            rho = tm._spearman(centre(fo), centre(yo))
            partial = float(np.corrcoef(_residual(fo, Xo), _residual(yo, Xo))[0, 1])
            rng = np.random.default_rng(seed_for(seed, "mechanism", fid, tid))
            reps = np.full(tc.n_boot, NAN)
            for b in range(tc.n_boot):
                i = rng.integers(0, n, n)
                rf, ry = _residual(fo[i], Xo[i]), _residual(yo[i], Xo[i])
                if rf.std() > 1e-12 and ry.std() > 1e-12:
                    reps[b] = np.corrcoef(rf, ry)[0, 1]
            lo, hi = tm._interval(reps)
            rows.append({"feature_id": fid, "target_id": tid, "n_athletes": n, "spearman_within_position": rho, "partial_r": partial,
                         "partial_lower": lo, "partial_upper": hi, "covariates": "position, draft cohort, " + ", ".join(covariates),
                         "note": (notes or {}).get((fid, tid), "")})
    return pl.DataFrame(rows, schema=MECH.columns) if rows else MECH.empty()


# ============================================================================== interval stability, ranks, read-out
def ci_stability(oof: pl.DataFrame, pair: tuple[str, str], tc: TransferConfig, seed: int, budgets: Sequence[int] = (200, 500, 1000, 2000, 4000),
                 n_seeds: int = 5) -> pl.DataFrame:
    """Interval of Δ RMSE for one model pair under several bootstrap budgets and seeds: does the conclusion depend on
    the resampling noise? ``oof``: the primary analysis of one target."""
    a = tm.paired_arrays(oof)
    m, r = pair
    if m not in a["pred"] or r not in a["pred"]:
        return STABILITY.empty()
    y, n = a["y"], a["ids"].size
    est = float(np.sqrt(np.mean((y - a["pred"][r]) ** 2)) - np.sqrt(np.mean((y - a["pred"][m]) ** 2)))
    rows = []
    for nb in budgets:
        for s in range(n_seeds):
            idx = np.random.default_rng(seed_for(seed, "stability", nb, s)).integers(0, n, size=(int(nb), n))
            d = np.sqrt(((y[idx] - a["pred"][r][idx]) ** 2).mean(1)) - np.sqrt(((y[idx] - a["pred"][m][idx]) ** 2).mean(1))
            lo, hi = tm._interval(d)
            rows.append({"target_id": oof["target_id"][0], "model": m, "reference": r, "n_boot": int(nb), "boot_seed": s, "estimate": est,
                         "ci_lower": lo, "ci_upper": hi})
    return pl.DataFrame(rows, schema=STABILITY.columns)


def rank_shift(oof: pl.DataFrame, a: str = REFERENCE, b: str = "M5") -> pl.DataFrame:
    """Within-position rank of every athlete under the predictions of models ``a`` and ``b`` (1 = largest predicted
    value) and under the actual label. ``oof``: one analysis and one target."""
    w = (oof.filter(pl.col("model").is_in([a, b])).pivot(on="model", index=["nfl_id", "position_group", "y"], values="pred")
         .drop_nulls([a, b]))
    rank = lambda c: pl.col(c).rank("average", descending=True).over("position_group")                 # noqa: E731
    return w.with_columns(rank(a).alias("rank_a"), rank(b).alias("rank_b"), rank("y").alias("rank_actual"),
                          pl.len().over("position_group").alias("n_position")).with_columns(
        (pl.col("rank_a") - pl.col("rank_b")).alias("rank_gain_b")).sort("position_group", "rank_b")


def tcn_readout(prep: pl.DataFrame, target_id: str, fold: int, model: str = "M5", block: str = "TCN") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(weights, training means, training SDs) of the embedding block of one fold model, in embedding-dimension order
    — the inputs of ``transfer_repr.occlusion_profile``. Dimensions left out of the model (constant) get weight 0."""
    g = prep.filter((pl.col("analysis") == PRIMARY) & (pl.col("cohort") == PRIMARY) & (pl.col("target_id") == target_id)
                    & (pl.col("model") == model) & (pl.col("fold") == int(fold)) & (pl.col("block") == block))
    if not g.height:
        raise tm.CohortError(f"no fitted {block} block for {target_id}, fold {fold}")
    dim = [int(t.rsplit("_", 1)[1]) for t in g["term"].to_list()]
    E = max(dim)
    w, mu, sd = np.zeros(E), np.zeros(E), np.ones(E)
    for j, c, m_, s_ in zip(dim, g["coefficient"].to_list(), g["mean"].to_list(), g["sd"].to_list()):
        w[j - 1], mu[j - 1], sd[j - 1] = c, m_, s_
    return w, mu, sd


# ============================================================================== evidence, readiness, decision
def _row(paired: pl.DataFrame, analysis: str, target_id: str, model: str, reference: str, metric: str = "delta_rmse",
         cohort: str = PRIMARY) -> dict[str, Any] | None:
    r = paired.filter((pl.col("analysis") == analysis) & (pl.col("cohort") == cohort) & (pl.col("target_id") == target_id)
                      & (pl.col("model") == model) & (pl.col("reference") == reference) & (pl.col("metric") == metric))
    return r.row(0, named=True) if r.height else None


def evidence_table(target_id: str, paired: pl.DataFrame, sub_paired: pl.DataFrame, loco_paired: pl.DataFrame, seed_paired: pl.DataFrame,
                   perm: pl.DataFrame, pair_sum: pl.DataFrame, hetero: pl.DataFrame, tc: TransferConfig, compact_k: int | None) -> pl.DataFrame:
    """One row per scoring dimension on the PRIMARY frozen target. Missing pieces stay null — nothing is filled in."""
    rows = []
    for name, model, ref, _, _ in DIMENSIONS:
        if "{k}" in model:
            if compact_k is None:
                continue
            model = model.format(k=compact_k)
        main = _row(paired, PRIMARY, target_id, model, ref)
        if main is None:
            continue
        get = lambda r, k: None if r is None else r[k]                                           # noqa: E731
        rand = _row(paired, PRIMARY, target_id, model, "M6") if model == "M5" else None
        loco = _row(loco_paired, "loco", target_id, model, ref) if loco_paired.height else None

        def positive(group_type: str) -> tuple[int | None, int | None]:
            s = sub_paired.filter((pl.col("analysis") == PRIMARY) & (pl.col("target_id") == target_id) & (pl.col("group_type") == group_type)
                                  & (pl.col("model") == model) & (pl.col("reference") == ref) & (pl.col("metric") == "delta_rmse")
                                  & pl.col("estimate").is_not_null()) if sub_paired.height else sub_paired
            return (int((s["estimate"] > 0).sum()), s.height) if s.height else (None, None)
        seeds = (seed_paired.filter((pl.col("target_id") == target_id) & (pl.col("model") == model) & (pl.col("reference") == ref)
                                    & (pl.col("metric") == "delta_rmse") & pl.col("estimate").is_not_null())
                 if seed_paired.height and model == "M5" else None)

        def pval(mode: str) -> float | None:
            p = perm.filter((pl.col("target_id") == target_id) & (pl.col("model") == model) & (pl.col("reference") == ref)
                            & (pl.col("mode") == mode)) if perm.height else perm
            return float(p["p_value"][0]) if p.height else None
        ms = pair_sum.filter((pl.col("target_id") == target_id) & (pl.col("model") == model)) if pair_sum.height else pair_sum
        het = hetero.filter((pl.col("analysis") == PRIMARY) & (pl.col("target_id") == target_id) & (pl.col("group_type") == "position_group")
                            & (pl.col("model") == model) & (pl.col("reference") == ref)) if hetero.height else hetero
        cp, ct = positive("draft_year")
        pp, pt = positive("position_group")
        rows.append({"dimension": name, "model": model, "reference": ref, "target_id": target_id, "n_athletes": main["n_athletes"],
                     "delta_rmse": main["estimate"], "ci_lower": main["ci_lower"], "ci_upper": main["ci_upper"], "relative": main["relative"],
                     "established": main["established"], "worse": main["worse"], "vs_random_lower": get(rand, "ci_lower"),
                     "loco_delta": get(loco, "estimate"), "loco_lower": get(loco, "ci_lower"), "loco_upper": get(loco, "ci_upper"),
                     "cohorts_positive": cp, "cohorts_total": ct, "positions_positive": pp, "positions_total": pt,
                     "seeds_positive": None if seeds is None else int((seeds["estimate"] > 0).sum()),
                     "seeds_total": None if seeds is None else seeds.height, "perm_p_outcome": pval("outcome"),
                     "perm_p_linkage": pval("linkage"), "matched_d_concordance": ms["d_concordance"][0] if ms.height else None,
                     "matched_lower": ms["d_concordance_lower"][0] if ms.height else None,
                     "matched_upper": ms["d_concordance_upper"][0] if ms.height else None,
                     "heterogeneity_p_position": float(het["p_value"][0]) if het.height else None})
    return pl.DataFrame(rows, schema=EVIDENCE.columns) if rows else EVIDENCE.empty()


def readiness_status(e: dict[str, Any], combine_readiness: str, tc: TransferConfig) -> tuple[str, str]:
    """Conservative rule. ``predictive_validated`` needs the held-out gain over the dimension's reference to be
    established AND practically sized AND to survive every robustness check that applies; one failed or missing check
    leaves ``predictive_candidate``. Repeatability at the Combine never counts as NFL validation."""
    if e["delta_rmse"] is None or e["ci_lower"] is None:
        return "insufficient_evidence", "no held-out interval (too few athletes or not evaluated)"
    if e["worse"]:
        return "rejected", "held-out error is higher than the reference (interval below 0)"
    if not e["established"]:
        if e["delta_rmse"] > 0:
            return "exploratory", "point gain over the reference, interval includes 0: incremental benefit not established"
        ready = combine_readiness in ("ready_for_transfer_test", "carry_with_caveat", "reference")
        return ("descriptive_ready", "reliable Combine description; no held-out gain over the reference") if ready else (
            "insufficient_evidence", "no held-out gain and the Combine measurement itself is not ready")
    checks = {f"relative gain ≥ {100 * tc.min_relative_gain:g} %": None if e["relative"] is None else e["relative"] >= tc.min_relative_gain,
              "gain holds with each draft cohort left out": None if e["loco_delta"] is None else e["loco_delta"] > 0,
              "≥ 3 position groups with a positive estimate": None if e["positions_positive"] is None else e["positions_positive"] >= 3,
              f"outcome permutation p < {tc.perm_alpha:g}": None if e["perm_p_outcome"] is None else e["perm_p_outcome"] < tc.perm_alpha}
    if e["model"] == "M5":
        checks["beats the random encoder"] = None if e["vs_random_lower"] is None else e["vs_random_lower"] > 0
        checks["positive for every training seed"] = None if e["seeds_total"] is None else e["seeds_positive"] == e["seeds_total"]
    failed = [k for k, v in checks.items() if v is False]
    unknown = [k for k, v in checks.items() if v is None]
    if not failed and not unknown:
        return "predictive_validated", "established held-out gain; every robustness check passed: " + "; ".join(checks)
    return "predictive_candidate", "established held-out gain; " + "; ".join(
        ([f"failed: {', '.join(failed)}"] if failed else []) + ([f"not evaluated: {', '.join(unknown)}"] if unknown else []))


def _txt(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{digits}f}"


def readiness_registry(evidence: pl.DataFrame, shortlist: pl.DataFrame, coverage: dict[str, str], tc: TransferConfig,
                       compact_k: int | None) -> pl.DataFrame:
    """``scoring_readiness_registry``: every dimension of ``DIMENSIONS`` with its Combine evidence (frozen shortlist,
    quoted as written there), its transfer evidence (``evidence_table``) and a rule-based status. No score is built."""
    sl = {r["candidate_id"]: r for r in shortlist.iter_rows(named=True)}
    ev = {r["dimension"]: r for r in evidence.iter_rows(named=True)}
    rows = []
    for name, model, ref, slot, meaning in DIMENSIONS:
        s = sl.get(slot, {})
        model = model.format(k=compact_k) if "{k}" in model and compact_k is not None else model
        e = ev.get(name)
        if e is None:
            status, why = "insufficient_evidence", "not evaluated in this run (no frozen candidate or no held-out comparison)"
            transfer, ci, effect, cohort_txt, pos_txt = "not evaluated", "n/a", None, "n/a", "n/a"
        else:
            status, why = readiness_status(e, str(s.get("readiness") or ""), tc)
            effect = e["delta_rmse"]
            ci = f"[{_txt(e['ci_lower'])}, {_txt(e['ci_upper'])}]"
            transfer = (f"{model} vs {ref} on {e['target_id']}, {e['n_athletes']} athletes: Δ RMSE {_txt(effect)} {ci}"
                        + ("" if e["relative"] is None else f" ({100 * e['relative']:+.1f} %)")
                        + f"; outcome permutation p {_txt(e['perm_p_outcome'])}; matched-prospect concordance vs official "
                          f"{_txt(e['matched_d_concordance'])} [{_txt(e['matched_lower'])}, {_txt(e['matched_upper'])}]")
            cohort_txt = (f"leave-one-cohort-out Δ RMSE {_txt(e['loco_delta'])} [{_txt(e['loco_lower'])}, {_txt(e['loco_upper'])}]; "
                          f"{e['cohorts_positive']} of {e['cohorts_total']} cohorts with a positive estimate")
            pos_txt = (f"{e['positions_positive']} of {e['positions_total']} position groups with a positive estimate; "
                       f"heterogeneity p {_txt(e['heterogeneity_p_position'])}")
        rows.append({"name": name, "candidate_feature_source": f"{slot}: {s.get('features') or 'empty slot'} → model {model}",
                     "physical_interpretation": meaning, "reliability_evidence": str(s.get("measurement_reliability") or "n/a"),
                     "stopwatch_redundancy": str(s.get("stopwatch_redundancy") or "n/a"), "nfl_transfer_evidence": transfer,
                     "effect_size": effect, "confidence_interval": ci, "cohort_robustness": cohort_txt, "position_applicability": pos_txt,
                     "measurement_coverage": coverage.get(name, str(s.get("sample_eligibility") or "n/a")), "status": status,
                     "status_reason": why})
    out = pl.DataFrame(rows, schema=READINESS.columns)
    bad = set(out["status"].to_list()) - set(STATUSES)
    if bad:
        raise ValueError(f"status values outside the allowed set: {bad}")
    return out


def decision_case(evidence: pl.DataFrame, paired: pl.DataFrame, sub_paired: pl.DataFrame, target_id: str, tc: TransferConfig) -> tuple[str, str]:
    """Case of the pre-declared decision tree, from the tables only (primary target, paired cohort)."""
    ev = {r["model"]: r for r in evidence.iter_rows(named=True)}
    est = lambda m: bool(ev.get(m) and ev[m]["established"])                                    # noqa: E731
    sized = lambda m: bool(est(m) and ev[m]["relative"] is not None and ev[m]["relative"] >= tc.min_relative_gain)   # noqa: E731
    over_speed = _row(paired, PRIMARY, target_id, "M5", "M3")
    any_pos = (sub_paired.filter((pl.col("analysis") == PRIMARY) & (pl.col("target_id") == target_id) & (pl.col("group_type") == "position_group")
                                 & (pl.col("reference") == REFERENCE) & (pl.col("metric") == "delta_rmse") & pl.col("established"))
               if sub_paired.height else sub_paired)
    if sized("M5") and over_speed is not None and over_speed["established"]:
        return "A", ("TCN-B lowers held-out error against official testing by a practically sized, established margin and also against "
                     "SPEED2: investigate interpretability and calibration, prepare a validated predictive scorecard candidate.")
    if est("M5"):
        return "C", ("TCN-B improves on official testing but not demonstrably on the simple SPEED2 representation (or the gain is below "
                     "the practical margin): prefer the simpler representation unless further evidence supports the encoder.")
    if est("M2") or est("M3") or est("M4"):
        which = [m for m in ("M2", "M3", "M4") if est(m)]
        return "B", (f"an interpretable or simple representation ({', '.join(which)}) improves on official testing and TCN-B does not: "
                     "build scorecards around reliable speed-development measures.")
    if any_pos.height:
        cells = sorted({f"{r['model']} in {r['group']}" for r in any_pos.iter_rows(named=True)})
        return "D", (f"no pooled gain is established; a gain appears only inside position group(s) ({'; '.join(cells)}): investigate "
                     "and require replication, no universal claim.")
    return "E", ("no method is shown to improve on official Combine testing for the frozen primary NFL target: report the negative "
                 "finding, keep the Combine profiles as descriptions, build no Athletic Transfer Score.")


def scoring_conditions(registry: pl.DataFrame, case: str, tc: TransferConfig) -> str:
    """Exact conditions that would justify the scoring milestone, and whether this run meets them."""
    val = registry.filter(pl.col("status") == "predictive_validated")["name"].to_list()
    cand = registry.filter(pl.col("status") == "predictive_candidate")["name"].to_list()
    lines = ["### Conditions for the scoring milestone", "",
             "A dimension may enter a PREDICTIVE score only when, on the frozen primary NFL target and the paired cohort:",
             "1. its model lowers held-out RMSE against its reference with a 95 % athlete-bootstrap interval above 0;",
             f"2. the relative gain is at least {100 * tc.min_relative_gain:g} % of the reference error;",
             "3. the gain stays positive with each draft cohort left out, and in at least three position groups;",
             f"4. the outcome-permutation check gives p < {tc.perm_alpha:g};",
             "5. for the encoder: it beats the random encoder and the gain is positive for every training seed.",
             "A dimension with a reliable Combine measurement and no established gain may enter a DESCRIPTIVE profile only, labelled as such.",
             "", f"- Meeting 1–5 in this run (`predictive_validated`): {val or 'none'}",
             f"- Meeting 1 only (`predictive_candidate`, needs replication or the missing checks): {cand or 'none'}",
             f"- Decision-tree case: **{case}**"]
    return "\n".join(lines)


def gate_06c(evidence: pl.DataFrame, sub_paired: pl.DataFrame, sens_paired: pl.DataFrame, pair_sum: pl.DataFrame,
             pairs: pl.DataFrame, cases: pl.DataFrame, registry: pl.DataFrame) -> pl.DataFrame:
    """Gate 06C — scouting interpretation."""
    with_ci = evidence.filter(pl.col("ci_lower").is_not_null()).height
    exposure = sens_paired.filter(pl.col("analysis").str.starts_with("exposure:"))["analysis"].n_unique() if sens_paired.height else 0
    positions = sub_paired.filter(pl.col("group_type") == "position_group")["group"].n_unique() if sub_paired.height else 0
    illustrative = bool(cases.height == 0 or (cases["status"] == "illustrative only").all())
    consistent = all((r["status"] != "predictive_validated") or (r["effect_size"] is not None and r["effect_size"] > 0)
                     for r in registry.iter_rows(named=True)) and set(registry["status"].to_list()) <= set(STATUSES)
    rows = [("gains_reported_with_uncertainty", evidence.height > 0 and with_ci == evidence.height,
             f"{with_ci} of {evidence.height} dimension(s) carry a paired interval"),
            ("position_and_exposure_limits_documented", positions >= 2 and exposure >= 2,
             f"{positions} position groups and {exposure} exposure rules evaluated and saved"),
            ("matched_prospects_out_of_fold", pair_sum.height > 0 and pairs.height > 0,
             f"{pairs.height} matched pairs scored on out-of-fold predictions" if pairs.height else "no matched pair inside the caliper"),
            ("case_studies_illustrative", illustrative, "case studies are chosen by stopwatch distance only and labelled illustrative"),
            ("readiness_reflects_evidence", bool(registry.height) and consistent, f"{registry.height} dimension(s); statuses "
             f"{dict(registry.group_by('status').len().iter_rows())}")]
    return pl.DataFrame([{"criterion": c, "passed": bool(p), "detail": d} for c, p, d in rows], schema=tr.GATE.columns)


def conclusions_06c(evidence: pl.DataFrame, registry: pl.DataFrame, case: tuple[str, str], pair_sum: pl.DataFrame, target_id: str) -> str:
    """Reading generated ONLY from the saved tables of this run."""
    lines = [f"### What the tables show (primary target `{target_id}`)", ""]
    for e in evidence.iter_rows(named=True):
        lines.append(f"- **{e['dimension']}** ({e['model']} vs {e['reference']}, {e['n_athletes']} athletes): Δ RMSE {_txt(e['delta_rmse'])} "
                     f"[{_txt(e['ci_lower'])}, {_txt(e['ci_upper'])}]; cohort left out {_txt(e['loco_delta'])}; positions with a positive "
                     f"estimate {e['positions_positive']}/{e['positions_total']}; permutation p {_txt(e['perm_p_outcome'])}.")
    ms = pair_sum.filter(pl.col("target_id") == target_id)
    if ms.height:
        lines += ["", f"Matched prospects ({ms['n_pairs'][0]} pairs, {ms['n_athletes'][0]} athletes, an athlete in up to {ms['max_reuse'][0]} pairs):"]
        lines += [f"- {r['model']}: concordance {_txt(r['concordance']).lstrip('+')} [{_txt(r['concordance_lower']).lstrip('+')}, "
                  f"{_txt(r['concordance_upper']).lstrip('+')}]" + ("" if r["d_concordance"] is None else
                                                                     f"; vs official {_txt(r['d_concordance'])} [{_txt(r['d_concordance_lower'])}, "
                                                                     f"{_txt(r['d_concordance_upper'])}]") for r in ms.sort("model").iter_rows(named=True)]
    lines += ["", "Scoring readiness: " + "; ".join(f"`{r['name']}` → **{r['status']}**" for r in registry.iter_rows(named=True)) + ".",
              "", f"Decision-tree case **{case[0]}** — {case[1]}", "",
              "Limits: the labels are observed NFL movement, shaped by role, scheme and usage as well as ability; draft cohorts differ "
              "in sensor generation and in NFL follow-up; intervals cover athlete sampling, not the refitting of the models; subgroup "
              "and matched-pair results are exploratory unless the pooled comparison is established."]
    return "\n".join(lines)
