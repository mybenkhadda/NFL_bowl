"""Physically aligned Combine → NFL transfer hypotheses (Milestone 07B). NumPy / Polars; reuses the Milestone 06 pipeline.

Status          EXPLORATORY. Milestone 06 was the confirmatory study and its conclusion (decision case E: no method shown
                to improve on official Combine testing) is not revisited. Everything here is a NEW hypothesis family,
                declared in ``HYPOTHESES`` before any result, on outcomes labelled exploratory in every table.
Reused          athletes, outer and inner folds, fold-fitted representations (06B ``transfer_attempt_representations_outer``),
                ridge ladder, out-of-fold loop and paired athlete bootstrap of ``transfer_models`` — unchanged.
New             outcomes: NFL speed (positive control), NFL acceleration over the speed bands a Combine sprint crosses,
                context-conditioned sustained acceleration (07A definitions, context parameters from the TRAINING
                athletes of each fold). Models M0 + tracked speed and M0 + speed development answer "does the measured
                Combine quantity predict the comparable NFL quantity beyond position, cohort and body?".
Multiplicity    every registered comparison is evaluated and reported; primaries are Holm-adjusted as a family; a
                subgroup or secondary result is never promoted to a finding.
Sign            Δ RMSE = RMSE(reference) − RMSE(model): POSITIVE = the model is better (Milestone 06 convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import get_logger
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from . import transfer_models as tm
from .context_outcomes import ContextLabels
from .context_reliability import GATE
from .transfer import TransferConfig

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
ANALYSIS = "m07_aligned"
COHORT = "paired"
STATUS = "EXPLORATORY (Milestone 07B)"
V4_V8 = "F6.forty.time_v4_to_v8_s"
P95 = "F2.forty.p95_speed"


# ============================================================================== declared outcomes and hypotheses
@dataclass(frozen=True)
class OutcomeDef:
    key: str
    outcome_id: str                 # base NFL outcome of the Milestone 05 engine (one play measure)
    definition: str                 # 07A context definition applied to it (C0 = the raw athlete aggregate)
    start_col: str | None           # play-table column with the starting speed of the measured window
    family: str                     # frozen_m05 · speed_band_aligned · context_conditioned
    physical: str

    @property
    def target_id(self) -> str:
        return f"{self.definition}:{self.outcome_id}"


OUTCOMES: tuple[OutcomeDef, ...] = (
    OutcomeDef("Y_SPEED", "O1.speed_p90_mean", "C0", None, "frozen_m05",
               "realised NFL speed: mean over windows of the play's 90th-percentile speed [yd/s]"),
    OutcomeDef("Y_BAND46", "O3.band_acc_v4_6_mean", "C0", None, "speed_band_aligned",
               "NFL acceleration from 4–6 yd/s, inside the 4 → 8 yd/s interval of the Combine measure [yd/s²]"),
    OutcomeDef("Y_BAND68", "O3.band_acc_v6_8_mean", "C0", None, "speed_band_aligned",
               "NFL acceleration from 6–8 yd/s, inside the 4 → 8 yd/s interval; fewer athletes reach it [yd/s²]"),
    OutcomeDef("Y_BAND02", "O3.band_acc_v0_2_mean", "C0", None, "frozen_m05",
               "NFL acceleration from a low starting speed (0–2 yd/s); BELOW the Combine 4 → 8 interval [yd/s²]"),
    OutcomeDef("Y_CTX", "O2.accel_sust05_mean", "C4", "gain05_v0", "context_conditioned",
               "sustained 0.5 s acceleration after removing alignment × play class, starting-speed band, season and down [yd/s²]"),
    OutcomeDef("Y_START", "O2.accel_sust05_mean", "C2", "gain05_v0", "context_conditioned",
               "sustained 0.5 s acceleration relative to the position × starting-speed band mean [yd/s²]"),
    OutcomeDef("Y_RAW", "O2.accel_sust05_mean", "C0", "gain05_v0", "frozen_m05",
               "raw sustained 0.5 s acceleration — REFERENCE ONLY (Milestone 06 used its frozen cell-residual form) [yd/s²]"),
)
OUTCOME = {o.key: o for o in OUTCOMES}

# (hypothesis, tier, outcome key, model, reference, question) — fixed before any result of this milestone.
HYPOTHESES: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("H1", "primary", "Y_SPEED", "M1", "M0", "positive control: do official Combine times predict realised NFL speed beyond position, cohort and body?"),
    ("H1", "secondary", "Y_SPEED", "S0", "M0", "does tracked Combine top speed predict realised NFL speed beyond position, cohort and body?"),
    ("H1", "secondary", "Y_SPEED", "M3", "M1", "does tracked Combine top speed add to the official times for NFL speed?"),
    ("H2", "primary", "Y_BAND46", "M2", "M1", "does Combine speed development (4 → 8 yd/s) add to official testing for NFL acceleration from 4–6 yd/s?"),
    ("H2", "secondary", "Y_BAND46", "E0", "M0", "does Combine speed development predict NFL acceleration from 4–6 yd/s beyond position, cohort and body?"),
    ("H2", "secondary", "Y_BAND68", "M2", "M1", "same, NFL acceleration from 6–8 yd/s"),
    ("H2", "secondary", "Y_BAND02", "M2", "M1", "same, NFL acceleration from 0–2 yd/s (a speed range the Combine measure does not cover)"),
    ("H2", "secondary", "Y_CTX", "M2", "M1", "same, context-conditioned sustained acceleration"),
    ("H3", "primary", "Y_CTX", "M5", "M1", "does TCN-B add to official testing for context-conditioned NFL sustained acceleration?"),
    ("H3", "secondary", "Y_START", "M5", "M1", "same, starting-speed-conditioned sustained acceleration"),
    ("H3", "secondary", "Y_BAND46", "M5", "M1", "same, NFL acceleration from 4–6 yd/s"),
    ("H3", "secondary", "Y_SPEED", "M5", "M1", "same, realised NFL speed"),
    ("H3", "secondary", "Y_CTX", "M5", "M3", "TCN-B against the simple tracked-speed representation"),
    ("H3", "secondary", "Y_CTX", "M5", "M2", "TCN-B against engineered speed development"),
    ("H3", "secondary", "Y_CTX", "M5", "M4", "TCN-B against FPCA"),
    ("H3", "secondary", "Y_CTX", "M5", "M6", "TCN-B against an untrained encoder"),
    ("H3", "secondary", "Y_CTX", "M4", "M1", "does FPCA add to official testing for the context-conditioned outcome?"),
    ("H3", "secondary", "Y_CTX", "M3", "M1", "does tracked speed add to official testing for the context-conditioned outcome?"),
)
SUBGROUP_HYPOTHESES = (("H1", "Y_SPEED", "M1", "M0"), ("H2", "Y_BAND46", "M2", "M1"), ("H3", "Y_CTX", "M5", "M1"))   # H4: by position
EXTRA_MODELS = (tm.ModelSpec("S0", "M0 + tracked speed (SPEED2)", "M0", ("SPEED2",), "exploratory"),
                tm.ModelSpec("E0", "M0 + speed development (4 → 8 yd/s)", "M0", ("ENG",), "exploratory"))
MODEL_FAMILIES = {"baselines": ("MEAN", "POS", "M0", "M1"), "physical": ("M2", "E0"), "speed": ("M3", "S0"), "fpca": ("M4",),
                  "tcn": ("M5",), "control": ("M6",)}


def hypothesis_table(active: Sequence[str] = ("H1", "H2", "H3", "H4")) -> pl.DataFrame:
    rows = [{"hypothesis": h, "tier": t, "outcome_key": o, "target_id": OUTCOME[o].target_id, "outcome_family": OUTCOME[o].family,
             "model": m, "reference": r, "scope": "pooled", "question": q, "physical_quantity": OUTCOME[o].physical,
             "active": h in active, "status": STATUS} for h, t, o, m, r, q in HYPOTHESES]
    rows += [{"hypothesis": "H4", "tier": "secondary", "outcome_key": o, "target_id": OUTCOME[o].target_id,
              "outcome_family": OUTCOME[o].family, "model": m, "reference": r, "scope": "by position group",
              "question": f"is the {h} primary comparison different between position groups? (heterogeneity, not a search for the best group)",
              "physical_quantity": OUTCOME[o].physical, "active": "H4" in active and h in active, "status": STATUS}
             for h, o, m, r in SUBGROUP_HYPOTHESES]
    return pl.DataFrame(rows, schema=HYPOTHESIS_REGISTRY.columns)


def models_for(tc: TransferConfig, families: Sequence[str]) -> list[tm.ModelSpec]:
    """Ladder restricted to the chosen model families (``MODEL_FAMILIES``); the baselines are always included."""
    want = set(MODEL_FAMILIES["baselines"]) | {m for f in families for m in MODEL_FAMILIES[f]}
    return [m for m in (*tm.model_ladder(tc), *EXTRA_MODELS) if m.model in want]


def pairs_for(outcome_key: str, active: Sequence[str], models: Sequence[tm.ModelSpec]) -> list[tuple[str, str]]:
    have = {m.model for m in models}
    out = [("POS", "MEAN"), ("M0", "POS")] + [(m, r) for h, _, o, m, r, _ in HYPOTHESES if o == outcome_key and h in active]
    return [p for p in dict.fromkeys(out) if p[0] in have and p[1] in have]


# ============================================================================== contracts
HYPOTHESIS_REGISTRY = TableContract(
    "aligned_transfer_hypotheses", "Registry of every Combine → NFL comparison of Milestone 07B, declared before evaluation.",
    {"hypothesis": S, "tier": S, "outcome_key": S, "target_id": S, "outcome_family": S, "model": S, "reference": S, "scope": S,
     "question": S, "physical_quantity": S, "active": B, "status": S}, key=("hypothesis", "target_id", "model", "reference", "scope"),
    area="hypothesis × outcome × model pair")
ALIGNED_OOF = TableContract(
    "aligned_transfer_oof_predictions", "Out-of-fold predictions of the exploratory aligned-transfer analyses.",
    {**tm.OOF.columns, "status": S}, key=tm.OOF.key, kind="dataset", area=tm.OOF.area,
    docs={**tm.OOF.docs, "target_id": "<context definition>:<NFL outcome>; labels use context parameters of the fold's training athletes"})
ALIGNED_PERF = TableContract(
    "aligned_transfer_model_comparisons", "Held-out performance of every model on every exploratory outcome (and subgroup).",
    {**tm.PERF.columns, "status": S}, key=tm.PERF.key, area=tm.PERF.area,
    docs={"analysis": "m07_aligned (pooled) · position:<group> · cohort:<draft year> · exposure:<band>"})
ALIGNED_PAIRED = TableContract(
    "aligned_transfer_paired_uncertainty", "Paired model differences with athlete-bootstrap intervals and multiplicity-adjusted reading.",
    {**tm.PAIRED.columns, "hypothesis": S, "tier": S, "p_boot": F, "p_holm_primary": F, "p_holm_all": F, "status": S},
    key=tm.PAIRED.key, area=tm.PAIRED.area,
    docs={"p_boot": "two-sided bootstrap p: 2 · min(share of resamples with Δ > 0, share with Δ ≤ 0), floored at 1 / replicates",
          "p_holm_primary": "Holm adjustment over the pooled PRIMARY comparisons (delta_rmse rows only)",
          "p_holm_all": "Holm adjustment over every registered pooled comparison (delta_rmse rows only)"})
COVERAGE = TableContract(
    "aligned_transfer_outcome_coverage", "Athletes of the paired Milestone 06 cohort supporting each exploratory outcome.",
    {"outcome_key": S, "target_id": S, "outcome_family": S, "n_design": I, "n_with_windows": I, "n_eligible": I, "position_counts": S,
     "evaluated": B, "reason": S, "status": S}, key=("outcome_key",), area="outcome")
MATCHED = TableContract(
    "aligned_transfer_matched_pairs", "Matched prospects (frozen pre-NFL rule of 06C): do models order their NFL movement difference?",
    {"target_id": S, "model": S, "n_pairs": I, "n_athletes": I, "max_pairs_per_athlete": I, "concordance": F, "ci_lower": F,
     "ci_upper": F, "delta_vs_reference": F, "delta_ci_lower": F, "delta_ci_upper": F, "diff_correlation": F, "status": S},
    key=("target_id", "model"), area="outcome × model",
    docs={"concordance": "share of pairs whose predicted difference has the sign of the observed difference (ties count ½)",
          "ci_lower": "95 % athlete-cluster bootstrap (an athlete in several pairs is resampled as one unit)",
          "delta_vs_reference": "concordance − concordance of the official model M1, same pairs and resamples"})
ASSOC = TableContract(
    "aligned_transfer_physical_association", "Within-position association of a Combine quantity with an NFL quantity (descriptive).",
    {"hypothesis": S, "feature": S, "target_id": S, "control": S, "n_athletes": I, "r": F, "ci_lower": F, "ci_upper": F,
     "expected_sign": S, "status": S}, key=("feature", "target_id", "control"), area="feature × outcome × control",
    docs={"control": "position = position means removed · official = position, official forty, 10-yard split and weight removed",
          "r": "Pearson correlation of the residuals; DESCRIPTIVE (in-sample), never a transfer claim"})
UNITS = TableContract(
    "aligned_transfer_unit_audit", "Physical-unit and definition consistency between Combine and NFL quantities.",
    {"quantity": S, "combine": S, "nfl": S, "comparable": S, "check": S, "passed": B, "note": S}, key=("quantity",), area="quantity")
EVIDENCE = TableContract(
    "aligned_transfer_evidence_vs_m06", "Milestone 06 confirmatory evidence beside the exploratory Milestone 07 comparisons.",
    {"study": S, "target_id": S, "model": S, "reference": S, "delta_rmse": F, "ci_lower": F, "ci_upper": F, "relative": F,
     "n_athletes": I, "established": B, "role": S}, key=("study", "target_id", "model", "reference"), area="study × outcome × model pair")


# ============================================================================== out-of-fold analysis
def run_outcome(o: OutcomeDef, lab: ContextLabels, d: tm.Design, blocks: Any, models: Sequence[tm.ModelSpec], tc: TransferConfig
                ) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Out-of-fold predictions of every model on one exploratory outcome, and its coverage row. The cohort is the
    athletes of ``d`` meeting the exposure requirement of the outcome definition (a COUNT rule, fixed before fitting,
    identical for every model). Labels are recomputed per fold from the training athletes' windows."""
    ids = lab.w.ids
    pos = np.minimum(np.searchsorted(ids, d.ids), ids.size - 1)
    present = ids[pos] == d.ids
    ok, why = lab.available(o.definition)
    cov = {"outcome_key": o.key, "target_id": o.target_id, "outcome_family": o.family, "n_design": d.n,
           "n_with_windows": int(present.sum()), "n_eligible": 0, "position_counts": "", "evaluated": False, "reason": why, "status": STATUS}
    if not ok:
        return ALIGNED_OOF.empty(), cov
    v = lab.values(o.definition)
    keep = present & (v["n_windows"][pos] >= lab.cc.min_windows) & np.isfinite(v["value"][pos])
    dd = d.subset(keep)
    cov.update(n_eligible=dd.n, position_counts=" · ".join(f"{p} {int((dd.position == p).sum())}" for p in np.unique(dd.position)))
    folds, counts = np.unique(dd.fold, return_counts=True)
    if dd.n < tc.min_athletes or folds.size < 2 or counts.min() < 3:
        cov["reason"] = f"{dd.n} eligible athletes in {folds.size} fold(s) (minimum {tc.min_athletes}): not evaluated"
        return ALIGNED_OOF.empty(), cov
    res, _ = tm.run_oof(dd, lab.label_fn(o.definition, dd.ids), blocks, list(models), tc)
    oof = tm.oof_frame(ANALYSIS, COHORT, o.target_id, dd, res).with_columns(pl.lit(STATUS).alias("status"))
    cov.update(evaluated=True, reason="")
    return oof.select(list(ALIGNED_OOF.columns)), cov


def _p_boot(share: float | None, n_boot: int) -> float | None:
    if share is None or not np.isfinite(share):
        return None
    return float(min(1.0, max(2.0 * min(share, 1.0 - share), 1.0 / max(n_boot, 1))))


def holm(p: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values; None entries stay None and are not counted in the family."""
    idx = [i for i, x in enumerate(p) if x is not None]
    out: list[float | None] = [None] * len(p)
    m, running = len(idx), 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: p[j])):
        running = max(running, min(1.0, (m - rank) * float(p[i])))
        out[i] = running
    return out


def evaluate(oof: pl.DataFrame, active: Sequence[str], models: Sequence[tm.ModelSpec], tc: TransferConfig, seed: int
             ) -> dict[str, pl.DataFrame]:
    """{ALIGNED_PERF, ALIGNED_PAIRED} of the pooled analyses. Each outcome is evaluated on ITS athletes (identical
    for every model of that outcome); different outcomes have different cohorts and are never compared as if paired."""
    by_target = {o.target_id: o for o in OUTCOMES}
    perf, paired = [], []
    for (t,), g in oof.partition_by(["target_id"], as_dict=True).items():
        ev = tm.evaluate(g.drop("status"), pairs_for(by_target[t].key, active, models), tc, seed)
        perf.append(ev[tm.PERF.name])
        paired.append(ev[tm.PAIRED.name])
    if not perf:
        return {ALIGNED_PERF.name: ALIGNED_PERF.empty(), ALIGNED_PAIRED.name: ALIGNED_PAIRED.empty()}
    reg = {(OUTCOME[o].target_id, m, r): (h, tier) for h, tier, o, m, r, _ in HYPOTHESES}
    pr = pl.concat(paired)
    hyp = [reg.get((t, m, r), ("sanity", "sanity")) for t, m, r in zip(pr["target_id"], pr["model"], pr["reference"])]
    pb = [_p_boot(s, tc.n_boot) for s in pr["share_boot_improved"]]
    is_rmse = [x == "delta_rmse" for x in pr["metric"]]
    prim = holm([p if (h[1] == "primary" and k) else None for p, h, k in zip(pb, hyp, is_rmse)])
    allp = holm([p if (h[0] != "sanity" and k) else None for p, h, k in zip(pb, hyp, is_rmse)])
    pr = pr.with_columns(pl.Series("hypothesis", [h[0] for h in hyp], dtype=pl.String), pl.Series("tier", [h[1] for h in hyp], dtype=pl.String),
                         pl.Series("p_boot", pb, dtype=pl.Float64), pl.Series("p_holm_primary", prim, dtype=pl.Float64),
                         pl.Series("p_holm_all", allp, dtype=pl.Float64), pl.lit(STATUS).alias("status"))
    return {ALIGNED_PERF.name: pl.concat(perf).with_columns(pl.lit(STATUS).alias("status")).select(list(ALIGNED_PERF.columns)),
            ALIGNED_PAIRED.name: pr.select(list(ALIGNED_PAIRED.columns))}


def subgroup_eval(oof: pl.DataFrame, groups: pl.DataFrame, kind: str, pairs: Sequence[tuple[str, str]], tc: TransferConfig, seed: int
                  ) -> dict[str, pl.DataFrame]:
    """Pooled fit, subgroup SCORING: the out-of-fold predictions of each subgroup (``groups``: nfl_id, group) are
    evaluated separately under analysis ``<kind>:<group>``. Intervals only with ≥ ``min_athletes_group`` athletes.
    Exploratory by construction: a subgroup with the largest estimate is not a discovery."""
    o = oof.drop("status").join(groups.select("nfl_id", pl.col("group").cast(pl.String)), on="nfl_id", how="inner")
    parts = [g.drop("group").with_columns(pl.lit(f"{kind}:{name}").alias("analysis"))
             for (name,), g in o.partition_by(["group"], as_dict=True).items() if name is not None]
    if not parts:
        return {ALIGNED_PERF.name: ALIGNED_PERF.empty(), ALIGNED_PAIRED.name: ALIGNED_PAIRED.empty()}
    ev = tm.evaluate(pl.concat(parts).select(list(tm.OOF.columns)), pairs, tc, seed)
    pr = ev[tm.PAIRED.name]
    pb = [_p_boot(s, tc.n_boot) for s in pr["share_boot_improved"]]
    pr = pr.with_columns(pl.lit("H4" if kind == "position" else "sensitivity").alias("hypothesis"), pl.lit("secondary").alias("tier"),
                         pl.Series("p_boot", pb, dtype=pl.Float64), pl.lit(None, dtype=pl.Float64).alias("p_holm_primary"),
                         pl.lit(None, dtype=pl.Float64).alias("p_holm_all"), pl.lit(STATUS).alias("status"))
    return {ALIGNED_PERF.name: ev[tm.PERF.name].with_columns(pl.lit(STATUS).alias("status")).select(list(ALIGNED_PERF.columns)),
            ALIGNED_PAIRED.name: pr.select(list(ALIGNED_PAIRED.columns))}


def exposure_groups(lab: ContextLabels, definition: str, ids: Sequence[int]) -> pl.DataFrame:
    """Tertiles of the athletes' window count under the outcome definition (a count: no label value)."""
    v = lab.values(definition)
    d = pl.DataFrame({"nfl_id": v["ids"], "n": v["n_windows"]}).filter(pl.col("nfl_id").is_in(list(ids)))
    q1, q2 = d["n"].quantile(1 / 3), d["n"].quantile(2 / 3)
    return d.with_columns(pl.when(pl.col("n") <= q1).then(pl.lit("low")).when(pl.col("n") <= q2).then(pl.lit("mid"))
                          .otherwise(pl.lit("high")).alias("group")).select("nfl_id", "group")


# ============================================================================== matched prospects
def pair_concordance(pairs: pl.DataFrame, oof: pl.DataFrame, tc: TransferConfig, seed: int, reference: str = tm.REFERENCE) -> pl.DataFrame:
    """Concordance of predicted and observed NFL differences inside the matched pairs frozen in 06C (same position,
    near-identical official forty, 10-yard split and body size — a rule that uses pre-NFL information only). Pairs are
    kept when BOTH athletes have an out-of-fold prediction for the outcome; none is chosen by its outcome."""
    rows: list[dict[str, Any]] = []
    pa, pb = pairs["nfl_id_a"].to_numpy().astype(np.int64), pairs["nfl_id_b"].to_numpy().astype(np.int64)
    for (t,), g in oof.partition_by(["target_id"], as_dict=True).items():
        arr = tm.paired_arrays(g.drop("status") if "status" in g.columns else g)
        ids = arr["ids"]
        ia, ib = np.searchsorted(ids, pa), np.searchsorted(ids, pb)
        ok = (ia < ids.size) & (ib < ids.size)
        ok &= (ids[np.minimum(ia, ids.size - 1)] == pa) & (ids[np.minimum(ib, ids.size - 1)] == pb)
        ia, ib = ia[ok], ib[ok]
        if ia.size < 10:
            continue
        yd = arr["y"][ia] - arr["y"][ib]
        used = np.unique(np.r_[ia, ib])
        reuse = int(np.bincount(np.r_[ia, ib]).max())
        rng = np.random.default_rng(seed_for(seed, "m07", "pairs", t))
        mult = np.stack([np.bincount(rng.choice(used, used.size), minlength=ids.size) for _ in range(tc.n_boot)]).astype(float)
        wts = mult[:, ia] * mult[:, ib]

        def conc(pred: np.ndarray) -> tuple[float, np.ndarray, float]:
            pd_ = pred[ia] - pred[ib]
            hit = np.where(pd_ * yd > 0, 1.0, np.where(pd_ * yd == 0, 0.5, 0.0))
            den = wts.sum(1)
            reps = np.where(den > 0, (wts * hit).sum(1) / np.where(den > 0, den, 1.0), NAN)
            c = np.corrcoef(pd_, yd)[0, 1] if pd_.std() > 1e-12 and yd.std() > 1e-12 else NAN
            return float(hit.mean()), reps, float(c)

        ref = conc(arr["pred"][reference]) if reference in arr["pred"] else None
        for m, pred in arr["pred"].items():
            est, reps, c = conc(pred)
            fin = reps[np.isfinite(reps)]
            row = {"target_id": t, "model": m, "n_pairs": int(ia.size), "n_athletes": int(used.size), "max_pairs_per_athlete": reuse,
                   "concordance": est, "ci_lower": float(np.quantile(fin, 0.025)) if fin.size >= 20 else None,
                   "ci_upper": float(np.quantile(fin, 0.975)) if fin.size >= 20 else None, "delta_vs_reference": None,
                   "delta_ci_lower": None, "delta_ci_upper": None, "diff_correlation": c if np.isfinite(c) else None, "status": STATUS}
            if ref is not None and m != reference:
                dlt = reps - ref[1]
                dlt = dlt[np.isfinite(dlt)]
                row.update(delta_vs_reference=est - ref[0], delta_ci_lower=float(np.quantile(dlt, 0.025)) if dlt.size >= 20 else None,
                           delta_ci_upper=float(np.quantile(dlt, 0.975)) if dlt.size >= 20 else None)
            rows.append(row)
    return pl.DataFrame(rows, schema=MATCHED.columns) if rows else MATCHED.empty()


# ============================================================================== physical quantities (descriptive)
def v4_v8_mean_accel(time_s: np.ndarray) -> np.ndarray:
    """Mean tangential acceleration over the 4 → 8 yd/s interval [yd/s²] = Δv / Δt = 4 / time. A TIME is inverted:
    a shorter time is a larger acceleration."""
    t = np.asarray(time_s, dtype=float)
    return np.where(t > 0, 4.0 / np.where(t > 0, t, 1.0), NAN)


def athlete_features(attempt_values: pl.DataFrame, feature_ids: Sequence[str]) -> pl.DataFrame:
    """Athlete mean of the valid attempts of each feature (wide: nfl_id + one column per feature, n_attempts_<feature>)."""
    d = attempt_values.filter(pl.col("feature_id").is_in(list(feature_ids)) & pl.col("value").is_not_null() & pl.col("value").is_finite())
    g = d.group_by("nfl_id", "feature_id").agg(pl.col("value").mean().alias("v"), pl.len().cast(pl.Int64).alias("n"))
    out = g.select("nfl_id").unique().sort("nfl_id")
    for f in feature_ids:
        out = out.join(g.filter(pl.col("feature_id") == f).select("nfl_id", pl.col("v").alias(f), pl.col("n").alias(f"n_attempts_{f}")),
                       on="nfl_id", how="left")
    return out


def _resid(v: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def partial_association(x: np.ndarray, y: np.ndarray, position: np.ndarray, covariates: np.ndarray | None, n_boot: int,
                        rng: np.random.Generator) -> dict[str, Any]:
    """Pearson correlation of x and y after removing position means (and ``covariates``), with an athlete bootstrap
    that refits the removal in every resample. In-sample and descriptive."""
    ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(covariates).all(1) if covariates is not None else True)
    x, y, pos = x[ok], y[ok], position[ok]
    C = covariates[ok] if covariates is not None else np.empty((x.size, 0))
    if x.size < 10:
        return {"n": int(x.size), "r": None, "lo": None, "hi": None}

    def one(i: np.ndarray) -> float:
        D = np.column_stack([(pos[i] == p).astype(float) for p in np.unique(pos[i])] + [C[i]])
        a, b = _resid(x[i], D), _resid(y[i], D)
        den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
        return float((a * b).sum() / den) if den > 1e-12 else NAN

    reps = np.array([one(i) for i in rng.integers(0, x.size, size=(n_boot, x.size))]) if n_boot > 0 else np.empty(0)
    fin = reps[np.isfinite(reps)]
    return {"n": int(x.size), "r": one(np.arange(x.size)), "lo": float(np.quantile(fin, 0.025)) if fin.size >= 20 else None,
            "hi": float(np.quantile(fin, 0.975)) if fin.size >= 20 else None}


def association_table(features: dict[str, tuple[np.ndarray, np.ndarray]], labels: dict[str, tuple[np.ndarray, np.ndarray]],
                      d: tm.Design, plan: Sequence[tuple[str, str, str, str]], official: Sequence[str], n_boot: int, seed: int) -> pl.DataFrame:
    """``plan``: (hypothesis, feature name, target_id, expected sign). ``features`` / ``labels``: name → (ids, values).
    Each pair is reported with position removed, and with position + the official core removed."""
    def align(pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        ids, v = pair
        pos = np.minimum(np.searchsorted(ids, d.ids), max(ids.size - 1, 0))
        return np.where(ids[pos] == d.ids, np.asarray(v, dtype=float)[pos], NAN) if ids.size else np.full(d.n, NAN)

    off = np.column_stack([d.numeric[c] for c in official]) if official else None
    rows = []
    for h, f, t, sign in plan:
        if f not in features or t not in labels:
            continue
        x, y = align(features[f]), align(labels[t])
        for control, C in (("position", None), ("official", off)):
            a = partial_association(x, y, d.position, C, n_boot, np.random.default_rng(seed_for(seed, "m07", "assoc", f, t, control)))
            rows.append({"hypothesis": h, "feature": f, "target_id": t, "control": control, "n_athletes": a["n"], "r": a["r"],
                         "ci_lower": a["lo"], "ci_upper": a["hi"], "expected_sign": sign, "status": "DESCRIPTIVE, in-sample · " + STATUS})
    out = pl.DataFrame(rows, schema=ASSOC.columns) if rows else ASSOC.empty()
    return out.with_columns(pl.col("r").fill_nan(None))


def unit_audit(frames: pl.DataFrame, nfl_interval_ms: float | None, combine_units: dict[str, str], nfl_units: dict[str, str],
               combine_smoothing: str, nfl_smoothing: str) -> pl.DataFrame:
    """Unit and definition consistency. Measured checks: Combine frame interval (from ``sequence_frames``), declared
    units of the features and outcomes used. The remaining rows state a definitional (mis)match in words."""
    dt = (frames.sort("seq_id", "frame").with_columns((pl.col("t_rel") - pl.col("t_rel").shift(1).over("seq_id")).alias("_dt"))
          .filter(pl.col("_dt").is_not_null())["_dt"])
    med = float(dt.median()) if dt.len() else NAN
    hz_c = 1.0 / med if med and med > 0 else NAN
    hz_n = 1000.0 / nfl_interval_ms if nfl_interval_ms else NAN
    sp_c, sp_n = combine_units.get(P95, "?"), nfl_units.get("O1.speed_p90_mean", "?")
    ac_n = nfl_units.get("O3.band_acc_v4_6_mean", nfl_units.get("O2.accel_sust05_mean", "?"))
    rows = [
        {"quantity": "sampling frequency", "combine": f"{hz_c:.2f} Hz (median frame interval {med:.3f} s)", "nfl": f"{hz_n:.2f} Hz",
         "comparable": "yes", "check": "both 10 Hz within 5 %", "passed": bool(abs(hz_c - 10) < 0.5 and abs(hz_n - 10) < 0.5),
         "note": "window lengths in frames are the same in both sources"},
        {"quantity": "speed", "combine": f"|smoothed velocity| [{sp_c}], {combine_smoothing}", "nfl": f"|smoothed velocity| [{sp_n}], {nfl_smoothing}",
         "comparable": "yes", "check": "same unit", "passed": sp_c == sp_n,
         "note": "Combine p95 speed is reached in a maximal straight sprint; NFL p90-of-play speed depends on play length and assignment"},
        {"quantity": "distance", "combine": "path length of the smoothed position [yd]", "nfl": "field coordinates [yd]", "comparable": "yes",
         "check": "declared", "passed": True, "note": "no distance-based NFL outcome is used in 07B"},
        {"quantity": "speed development / acceleration", "combine": f"time for speed to rise 4 → 8 yd/s [{combine_units.get(V4_V8, '?')}]; "
         "mean acceleration over the interval = 4 / time [yd/s²]", "nfl": f"mean of (speed(t + 0.5 s) − speed(t)) / 0.5 s over start frames in a speed band with a positive change [{ac_n}]",
         "comparable": "partly", "check": "Combine quantity is a time, NFL quantity an acceleration", "passed": combine_units.get(V4_V8, "?") == "s",
         "note": "same speed range for the 4–6 and 6–8 bands; the Combine value is ONE transit at maximal effort, the NFL value an average of "
                 "sub-maximal, opportunity-dependent accelerating moments. Expected sign: shorter Combine time ↔ larger NFL acceleration"},
        {"quantity": "tangential acceleration", "combine": "a_tan = d|v|/dt, signed", "nfl": "accel_sust05 = largest 0.5 s mean of a_tan in the play, signed",
         "comparable": "partly", "check": "declared", "passed": True,
         "note": "same derivative, different statistic: a play maximum over a free window against a fixed speed interval"},
        {"quantity": "starting speed", "combine": "near rest (first-frame speed recorded; some recordings start in motion)",
         "nfl": "any; starting-speed band of the measured window is a context factor (07A)", "comparable": "no", "check": "declared", "passed": True,
         "note": "the 0–2 yd/s NFL band lies below the Combine 4 → 8 interval: H2 on Y_BAND02 compares different speed ranges and is labelled so"},
        {"quantity": "window length", "combine": "4 → 8 yd/s interval: variable duration (about 1 s)", "nfl": "0.5 s forward difference", "comparable": "partly",
         "check": "declared", "passed": True, "note": "no NFL window is resampled to the Combine interval"},
    ]
    return pl.DataFrame(rows, schema=UNITS.columns)


# ============================================================================== evidence, gate, reading
def evidence_vs_m06(m06_paired: pl.DataFrame, m07_paired: pl.DataFrame) -> pl.DataFrame:
    """Milestone 06 primary-analysis Δ RMSE rows (confirmatory, unchanged) beside the pooled Milestone 07 rows."""
    a = (m06_paired.filter((pl.col("analysis") == "primary") & (pl.col("cohort") == "primary") & (pl.col("metric") == "delta_rmse")
                           & (pl.col("reference") == "M1"))
         .select(pl.lit("M06 confirmatory").alias("study"), "target_id", "model", "reference", pl.col("estimate").alias("delta_rmse"),
                 "ci_lower", "ci_upper", "relative", "n_athletes", "established", pl.lit("confirmatory (frozen)").alias("role")))
    b = (m07_paired.filter((pl.col("analysis") == ANALYSIS) & (pl.col("metric") == "delta_rmse") & (pl.col("hypothesis") != "sanity"))
         .select(pl.lit("M07 exploratory").alias("study"), "target_id", "model", "reference", pl.col("estimate").alias("delta_rmse"),
                 "ci_lower", "ci_upper", "relative", "n_athletes", "established",
                 pl.concat_str([pl.col("hypothesis"), pl.lit(" "), pl.col("tier")]).alias("role")))
    return pl.concat([a, b]).select([pl.col(k).cast(t) for k, t in EVIDENCE.columns.items()])


def gate_07b(registry: pl.DataFrame, units: pl.DataFrame, oof: pl.DataFrame, paired: pl.DataFrame, coverage: pl.DataFrame,
             registry_written_first: bool) -> pl.DataFrame:
    """Gate 07B — physical transfer evidence. A positive model gain is NOT a criterion."""
    rows = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"criterion": name, "passed": bool(ok), "detail": detail})

    add("hypotheses_fixed_before_evaluation", registry_written_first and registry.height > 0,
        f"{registry.height} registered comparison(s); the registry was persisted before the first model was fitted" if registry_written_first
        else "the hypothesis registry was not persisted before evaluation")
    bad = units.filter(~pl.col("passed"))["quantity"].to_list()
    add("comparable_physical_quantities", not bad, "unit checks pass; definitional mismatches are stated in the audit" if not bad else f"failed: {bad}")
    sets = {t: {m: frozenset(g["nfl_id"].to_list()) for (m,), g in o.partition_by(["model"], as_dict=True).items()}
            for (t,), o in oof.partition_by(["target_id"], as_dict=True).items()} if oof.height else {}
    same = bool(sets) and all(len(set(by.values())) == 1 for by in sets.values())
    uniq = oof.height > 0 and oof.height == oof.select(pl.struct("target_id", "model", "nfl_id").n_unique()).item()
    add("athlete_held_out_predictions", same and uniq, f"{len(sets)} outcome(s); identical athletes for every model of an outcome: {same}; "
                                                       f"one prediction per athlete and model: {uniq}")
    need = {"M0", "M1"}
    add("baselines_included", bool(sets) and all(need <= set(by) for by in sets.values()), "M0 and M1 evaluated on every outcome")
    pooled = paired.filter((pl.col("analysis") == ANALYSIS) & (pl.col("metric") == "delta_rmse") & (pl.col("hypothesis") != "sanity")) if paired.height else paired
    add("paired_uncertainty_calculated", pooled.height > 0 and pooled["ci_lower"].null_count() == 0, f"{pooled.height} registered pooled comparison(s) with an interval")
    add("exploratory_status_explicit", oof.height > 0 and oof["status"].n_unique() == 1 and paired["status"].n_unique() == 1, STATUS)
    skipped = coverage.filter(~pl.col("evaluated"))["outcome_key"].to_list()
    add("coverage_reported (informative)", True, f"outcomes not evaluated: {skipped or 'none'}")
    return pl.DataFrame(rows, schema=GATE.columns)


def interpretation_markdown(paired: pl.DataFrame, coverage: pl.DataFrame, matched: pl.DataFrame, assoc: pl.DataFrame, tc: TransferConfig) -> str:
    """``aligned_transfer_interpretation.md`` — written from the executed tables only."""
    f = lambda x, d=4: "n/a" if x is None else f"{x:+.{d}f}"
    lines = ["# Physically aligned transfer — interpretation (Milestone 07B)", "",
             f"**Status: {STATUS}.** Milestone 06 remains the confirmatory study (no method shown to improve on official Combine testing). "
             "Nothing below is a confirmatory finding; an interval that excludes 0 here is a hypothesis for an independent cohort.", ""]
    pooled = paired.filter((pl.col("analysis") == ANALYSIS) & (pl.col("metric") == "delta_rmse") & (pl.col("hypothesis") != "sanity"))
    for h, title in (("H1", "H1 — Combine speed → NFL speed (positive control)"), ("H2", "H2 — Combine speed development → NFL acceleration"),
                     ("H3", "H3 — TCN-B beyond official measurements")):
        g = pooled.filter(pl.col("hypothesis") == h).sort("tier", "target_id", "model")
        lines += [f"## {title}", ""]
        if not g.height:
            lines += ["Not evaluated.", ""]
            continue
        lines += ["| Tier | Outcome | Model vs reference | n | Δ RMSE [95 %] | Relative | Holm p (primaries) | Reading |", "|---|---|---|---|---|---|---|---|"]
        for r in g.iter_rows(named=True):
            rel = r["relative"]
            reading = ("lower error, interval above 0" + ("" if rel is not None and rel >= tc.min_relative_gain else f"; below the {tc.min_relative_gain:.0%} practical margin")
                       if r["established"] else "higher error, interval below 0" if r["worse"] else "not established (interval includes 0)")
            lines.append(f"| {r['tier']} | `{r['target_id']}` | {r['model']} vs {r['reference']} | {r['n_athletes']} | {f(r['estimate'])} "
                         f"[{f(r['ci_lower'])}, {f(r['ci_upper'])}] | {f(None if rel is None else 100 * rel, 1)} % | "
                         f"{'n/a' if r['p_holm_primary'] is None else format(r['p_holm_primary'], '.3f')} | {reading} |")
        lines.append("")
    sub = paired.filter((pl.col("hypothesis") == "H4") & (pl.col("metric") == "delta_rmse"))
    lines += ["## H4 — position-conditioned transfer", ""]
    lines += [f"- `{r['target_id']}` {r['model']} vs {r['reference']}, {r['analysis']} (n = {r['n_athletes']}): {f(r['estimate'])} "
              f"[{f(r['ci_lower'])}, {f(r['ci_upper'])}]" for r in sub.sort("target_id", "analysis").iter_rows(named=True)] or ["Not evaluated."]
    lines += ["", "Subgroups are scored from the pooled fit and are exploratory: the group with the largest estimate is not a discovery.", "",
              "## Matched prospects", ""]
    lines += [f"- `{r['target_id']}` {r['model']}: concordance {r['concordance']:.3f} [{f(r['ci_lower'], 3)}, {f(r['ci_upper'], 3)}], "
              f"vs official {f(r['delta_vs_reference'], 3)} [{f(r['delta_ci_lower'], 3)}, {f(r['delta_ci_upper'], 3)}] "
              f"({r['n_pairs']} pairs, {r['n_athletes']} athletes, an athlete in up to {r['max_pairs_per_athlete']} pairs)"
              for r in matched.sort("target_id", "model").iter_rows(named=True)] or ["Not evaluated."]
    lines += ["", "## Descriptive physical associations (in-sample)", ""]
    lines += [f"- {r['feature']} ↔ `{r['target_id']}` after {r['control']} (n = {r['n_athletes']}): r = {f(r['r'], 3)} "
              f"[{f(r['ci_lower'], 3)}, {f(r['ci_upper'], 3)}], expected sign {r['expected_sign']}" for r in assoc.iter_rows(named=True)] or ["Not computed."]
    lines += ["", "## Coverage", ""]
    lines += [f"- {r['outcome_key']} `{r['target_id']}`: {r['n_eligible']} of {r['n_design']} athletes ({r['position_counts']})"
              + ("" if r["evaluated"] else f" — NOT evaluated: {r['reason']}") for r in coverage.iter_rows(named=True)]
    lines += ["", "Limits: outcomes are observed NFL movement shaped by role, scheme and usage; the context-conditioned outcomes remove "
              "observed labels only; intervals cover athlete sampling, not model refitting; outcomes have different cohorts and are not "
              "compared with each other; the comparisons of this notebook were declared after the Milestone 06 result was known."]
    return "\n".join(lines)
