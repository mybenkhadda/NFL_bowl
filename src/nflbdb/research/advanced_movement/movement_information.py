"""Milestone 08B.4 — what does Combine tracking add beyond the stopwatch, whether or not it predicts the NFL?
(NumPy / Polars; Combine side only — no NFL value is read.)

Question      Which measured characteristics remain distinguishable, interpretable and repeatable after the information
              already captured by official tests is accounted for?
Candidates    the frozen shortlist only (``SCALARS``, ``EMBEDDINGS``); the feature atlas is not reopened.
Adjustment    position, official forty, official 10-yard split, weight and height. CROSS-FITTED: the expectation applied to
              an athlete is fitted on the athletes of the other folds (athlete means), so a residual never contains a fit
              to the athlete's own runs.
Repeatability scalars: ICC(1) of repeated attempts within position, athlete bootstrap — raw, then on the residual
              (the attempt minus the athlete's expectation from official tests). Embeddings: repeat-attempt retrieval AUC
              within position on the INDUCTIVE embeddings (the fold encoder that never saw the athlete) — raw, then on the
              residual embedding. An embedding dimension is never given a physical meaning.
Status        an ordered rule on support, raw repeatability and residual repeatability (reference lines declared here,
              before running; heuristics). No weighted novelty score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from .config import NotebookControls
from .scoring.percentiles import FORTY, P95, TEN, V4_V8
from .transfer_power import deal_folds

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
WEIGHT, HEIGHT = "cmb_combine_weight", "cmb_combine_height"
COVARIATES = (FORTY, TEN, WEIGHT, HEIGHT)
ACC_1S, GAIN_10 = "F6.forty.accel_1s_from_v4", "F6.forty.speed_gain_10yd_from_v4"
# (feature id, label, unit, physical definition) — tracked scalars of the frozen shortlist and of the F6 speed-band family
SCALARS: tuple[tuple[str, str, str, str], ...] = (
    (P95, "tracked top speed (p95)", "yd/s", "95th percentile of tracked speed in the forty"),
    (V4_V8, "speed development 4 → 8 yd/s", "s", "time for tracked speed to rise from 4 to 8 yd/s"),
    (ACC_1S, "acceleration over 1 s from 4 yd/s", "yd/s²", "mean tangential acceleration over the second following the 4 yd/s crossing"),
    (GAIN_10, "speed gain over 10 yd from 4 yd/s", "yd/s", "speed gained over the 10 yards following the 4 yd/s crossing"),
)
# (representation family of 06B, label, definition)
EMBEDDINGS: tuple[tuple[str, str, str], ...] = (
    ("FPCA", "FPCA speed-curve scores (K = 2)", "scores of the speed-by-distance curve on two functional principal components"),
    ("TCN", "TCN-B movement fingerprint (8-d)", "self-supervised embedding of the forty sequence; used as a SIMILARITY space only"),
)
M06_MODEL = {P95: "M3", V4_V8: "M2", "FPCA": "M4", "TCN": "M5"}       # Milestone 06 ladder model carrying each candidate
PRACTICAL = {
    P95: ("the speed the athlete actually reached and held; confirms or questions the official forty",
          "largely the same information as the stopwatch"),
    V4_V8: ("how quickly speed builds through the middle of the run; separates athletes with the same forty",
            "needs the 4 → 8 yd/s interval to be recorded; one transit at maximal effort"),
    ACC_1S: ("mid-run acceleration over a fixed second", "sensitive to where the recording starts"),
    GAIN_10: ("speed gained over a fixed distance after 4 yd/s", "close to top speed and to the official times"),
    "FPCA": ("the shape of the whole speed curve in two numbers", "components are statistical axes, not physical quantities"),
    "TCN": ("which athletes MOVE alike; a similarity search among prospects", "a learned similarity: no dimension has a physical meaning; "
            "repeatable is not predictive"),
}
STATUSES = ("repeatable_beyond_official", "weak_residual_signal", "redundant_with_official", "unstable", "insufficient_support")


@dataclass(frozen=True)
class InfoConfig:
    """Reference lines are heuristics declared before running; none is tuned on a result."""

    k_folds: int = 5                    # athlete folds of the cross-fitted adjustment
    n_boot: int = 500                   # athlete bootstrap
    min_repeat: int = 20                # athletes with a repeat needed for a repeatability estimate
    min_subset_repeat: int = 10         # same, inside a sensitivity subset
    raw_icc_ref: float = 0.70           # scalar: raw ICC(1) at or above → repeatable
    resid_icc_ref: float = 0.50         # scalar: residual ICC(1) at or above (lower bound > 0) → repeatable beyond official
    raw_auc_ref: float = 0.75           # embedding: raw repeat-attempt AUC at or above → repeatable
    resid_auc_ref: float = 0.65         # embedding: residual AUC at or above (lower bound > 0.5) → repeatable beyond official
    seed: int = 20261011

    def for_mode(self, controls: NotebookControls) -> "InfoConfig":
        if not controls.smoke:
            return replace(self, seed=controls.seed)
        return replace(self, k_folds=3, n_boot=60, min_repeat=5, min_subset_repeat=3, seed=controls.seed)

    def params(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================== contracts
INFO = TableContract(
    "movement_incremental_information", "Raw and stopwatch-adjusted repeatability of each frozen Combine tracking candidate.",
    {"feature_id": S, "label": S, "kind": S, "metric": S, "unit": S, "n_attempts": I, "n_athletes": I, "n_repeat": I, "n_adjusted": I,
     "raw_value": F, "raw_lower": F, "raw_upper": F, "r2_position": F, "r2_official": F, "residual_value": F, "residual_lower": F,
     "residual_upper": F, "status": S, "reason": S}, key=("feature_id",), area="feature",
    docs={"metric": "ICC(1) within position (scalar) · repeat-attempt retrieval AUC within position (embedding; 0.5 = chance)",
          "r2_official": "cross-fitted share of the variance of athlete means explained by position + official forty, 10-yard split, "
                         "weight, height (can be negative when the adjustment does not generalise)",
          "residual_value": "the same repeatability metric after subtracting the athlete's cross-fitted expectation"})
INFO_SENS = TableContract(
    "movement_information_sensitivity", "Residual repeatability inside recording-quality, position and draft-cohort subsets.",
    {"feature_id": S, "subset_kind": S, "subset": S, "n_athletes": I, "n_repeat": I, "raw_value": F, "residual_value": F,
     "residual_lower": F, "residual_upper": F, "note": S}, key=("feature_id", "subset_kind", "subset"), area="feature × subset")
RESIDUALS = TableContract(
    "movement_official_adjusted_values", "Athlete value of each scalar candidate, its cross-fitted expectation from official tests and the residual.",
    {"feature_id": S, "nfl_id": I, "position_group": S, "n_attempts": I, "value": F, "expected_from_official": F, "residual": F,
     "standard_error": F}, key=("feature_id", "nfl_id"), kind="dataset", area="feature × athlete",
    docs={"standard_error": "√(pooled within-athlete attempt variance / attempts) — measurement error of the athlete value"})
MATCH_DIST = TableContract(
    "movement_matched_distinguishability", "Among matched stopwatch prospects: is the measured difference larger than its measurement error?",
    {"feature_id": S, "unit": S, "n_pairs": I, "n_athletes": I, "median_abs_difference": F, "median_se_difference": F,
     "share_distinguishable": F}, key=("feature_id",), area="feature",
    docs={"share_distinguishable": "share of matched pairs with |difference| > 1.96 × SE of the difference (measurement error of both athletes)"})
REGISTRY = TableContract(
    "scouting_evidence_registry", "Scouting evidence registry: what each Combine tracking characteristic is, how well it is measured and what it may be used for.",
    {"feature_id": S, "label": S, "physical_definition": S, "reference_position": S, "raw_repeatability": S, "residual_repeatability": S,
     "stopwatch_redundancy": S, "measurement_quality": S, "coverage": S, "nfl_predictive_evidence": S, "practical_interpretation": S,
     "limitations": S, "status": S}, key=("feature_id",), area="feature")
GATE_D = TableContract("football_gate_d_information", "Gate D — incremental information, per candidate.",
                       {"feature_id": S, "criterion": S, "value": S, "passed": B}, key=("feature_id", "criterion"), area="feature × criterion")


# ============================================================================== inputs
@dataclass
class Feature:
    feature_id: str
    label: str
    kind: str                # scalar · embedding
    unit: str
    definition: str
    seq: np.ndarray          # (n,) sequence ids
    nfl: np.ndarray          # (n,) athlete ids
    V: np.ndarray            # (n, E) values (E = 1 for a scalar)
    order: np.ndarray        # (n,) attempt order (1 = first in time)


def roster_arrays(roster: pl.DataFrame, covariates: Sequence[str] = COVARIATES) -> dict[str, Any]:
    """{ids (sorted), position, year, X (A, p) with NaN where missing} from a roster holding the official results."""
    r = roster.sort("nfl_id")
    return {"ids": r["nfl_id"].to_numpy().astype(np.int64), "position": np.asarray(r["position_group"].fill_null("?").to_list(), dtype=str),
            "year": r["draft_year"].cast(pl.Float64).fill_null(NAN).to_numpy(),
            "X": np.column_stack([r[c].cast(pl.Float64).fill_null(NAN).to_numpy() for c in covariates]), "covariates": tuple(covariates)}


def inductive_embeddings(rep_attempts: pl.DataFrame, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(seq ids, athlete ids, (n, E) embedding) of the attempts embedded by a fold encoder that did NOT train on the
    athlete (role ≠ 'train' in the 06B outer scheme). An athlete seen by every encoder is left out — never transductive."""
    g = rep_attempts.filter((pl.col("scheme") == "outer") & (pl.col("family") == family) & (pl.col("role") != "train"))
    if g.is_empty():
        return np.empty(0, dtype=str), np.empty(0, dtype=np.int64), np.empty((0, 0))
    g = g.filter(pl.col("fold") == pl.col("fold").min().over("seq_id")).sort("seq_id", "dim")
    E = g["dim"].n_unique()
    if g.height % E:
        raise ValueError(f"{family}: {g.height} rows are not a multiple of the {E} dimensions")
    pick = slice(None, None, E)
    return g["seq_id"].to_numpy()[pick].astype(str), g["nfl_id"].to_numpy()[pick].astype(np.int64), g["value"].to_numpy().reshape(-1, E).astype(float)


def prepare(attempt_values: pl.DataFrame, rep_attempts: pl.DataFrame | None, order: dict[str, int],
            scalars: Sequence[tuple[str, str, str, str]] = SCALARS, embeddings: Sequence[tuple[str, str, str]] = EMBEDDINGS) -> list[Feature]:
    """The frozen candidates as arrays. A candidate absent from the persisted tables is skipped (reported by the caller)."""
    out = []
    for fid, label, unit, definition in scalars:
        d = attempt_values.filter((pl.col("feature_id") == fid) & pl.col("value").is_not_null() & pl.col("value").is_finite()).sort("seq_id")
        if d.is_empty():
            continue
        seq = d["seq_id"].to_numpy().astype(str)
        out.append(Feature(fid, label, "scalar", unit, definition, seq, d["nfl_id"].to_numpy().astype(np.int64),
                           d["value"].to_numpy().astype(float)[:, None], np.array([order.get(s, 10 ** 6) for s in seq])))
    for fam, label, definition in embeddings:
        if rep_attempts is None:
            continue
        seq, nfl, Z = inductive_embeddings(rep_attempts, fam)
        if seq.size:
            out.append(Feature(fam, label, "embedding", "—", definition, seq, nfl, Z, np.array([order.get(s, 10 ** 6) for s in seq])))
    return out


# ============================================================================== cross-fitted adjustment
def _design(pos: np.ndarray, X: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Intercept + position dummies + covariates; levels, medians, centres and scales from the ``train`` athletes."""
    cols = [np.ones(pos.size)]
    for level in np.unique(pos[train])[1:]:
        cols.append((pos == level).astype(float))
    for j in range(X.shape[1]):
        x = X[:, j]
        obs = x[train][np.isfinite(x[train])]
        if obs.size < 3:
            continue
        x = np.where(np.isfinite(x), x, np.median(obs))
        mu, sd = x[train].mean(), x[train].std()
        if sd > 1e-12:
            cols.append((x - mu) / sd)
    return np.column_stack(cols)


def crossfit_predict(Y: np.ndarray, pos: np.ndarray, X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """(A, E) expectation of the athlete means ``Y`` from position and ``X``, each athlete predicted by a least-squares
    fit on the athletes of the other folds (folds balanced within position; no value of ``Y`` decides a fold)."""
    A = Y.shape[0]
    fold = deal_folds(pos, max(2, min(int(k), A)), np.random.default_rng([seed, 84]))
    pred = np.full(Y.shape, NAN)
    for f in np.unique(fold):
        te = fold == f
        tr = ~te
        if tr.sum() < 5:
            continue
        D = _design(pos, X, tr)
        G = D[tr].T @ D[tr] + 1e-8 * np.eye(D.shape[1])
        pred[te] = D[te] @ np.linalg.solve(G, D[tr].T @ Y[tr])
    return pred


# ============================================================================== repeatability
def _icc(n: np.ndarray, s: np.ndarray, ss: np.ndarray) -> float:
    k, N = n.size, float(n.sum())
    if k < 3 or N - k < 1:
        return NAN
    m = s / n
    grand = float(s.sum() / N)
    msb = float((n * (m - grand) ** 2).sum() / (k - 1))
    msw = float((ss - n * m ** 2).sum() / (N - k))
    n0 = (N - float((n ** 2).sum()) / N) / (k - 1)
    den = msb + (n0 - 1.0) * msw
    return (msb - msw) / den if den > 1e-15 else NAN


def icc1(values: np.ndarray, athlete: np.ndarray, position: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    """ICC(1) of repeated attempts after removing position means (one-way ANOVA, unbalanced), on the athletes with a
    repeat; interval = athlete bootstrap. ``position``: per attempt."""
    v = values.astype(float).copy()
    for p in np.unique(position):
        m = position == p
        v[m] -= v[m].mean()
    ids, inv = np.unique(athlete, return_inverse=True)
    n = np.bincount(inv).astype(float)
    s, ss = np.bincount(inv, weights=v), np.bincount(inv, weights=v ** 2)
    rep = n >= 2
    out = {"value": None, "lower": None, "upper": None, "n_athletes": int(ids.size), "n_repeat": int(rep.sum())}
    est = _icc(n[rep], s[rep], ss[rep])
    if not np.isfinite(est):
        return out
    out["value"] = float(est)
    idx = np.flatnonzero(rep)
    if n_boot > 0 and idx.size >= 5:
        reps = np.array([_icc(n[b], s[b], ss[b]) for b in idx[rng.integers(0, idx.size, (n_boot, idx.size))]])
        reps = reps[np.isfinite(reps)]
        if reps.size >= 20:
            out.update(lower=float(np.quantile(reps, 0.025)), upper=float(np.quantile(reps, 0.975)))
    return out


def retrieval_auc(Z: np.ndarray, athlete: np.ndarray, position: np.ndarray, order: np.ndarray, n_boot: int, rng: np.random.Generator,
                  min_gallery: int = 5) -> dict[str, Any]:
    """Repeat-attempt retrieval within position: for an athlete with two attempts, the share of the other athletes of
    his position whose first attempt lies FARTHER from his first attempt than his own second attempt (ties ½). Mean over
    athletes; 0.5 = his repeat is not recognisable. Dimensions are scaled to unit SD; interval = athlete bootstrap."""
    sd = Z.std(0)
    Zs = (Z - Z.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
    o = np.lexsort((order, athlete))
    Zs, a, pos = Zs[o], athlete[o], position[o]
    ids, start, n = np.unique(a, return_index=True, return_counts=True)
    first, apos = Zs[start], pos[start]
    scores = []
    for i in np.flatnonzero(n >= 2):
        gal = (apos == apos[i]) & (np.arange(ids.size) != i)
        if gal.sum() < min_gallery:
            continue
        d_same = float(np.linalg.norm(first[i] - Zs[start[i] + 1]))
        d_other = np.linalg.norm(first[gal] - first[i], axis=1)
        scores.append(float((d_other > d_same).mean() + 0.5 * (d_other == d_same).mean()))
    sc = np.asarray(scores)
    out = {"value": None, "lower": None, "upper": None, "n_athletes": int(ids.size), "n_repeat": int(sc.size)}
    if sc.size < 3:
        return out
    out["value"] = float(sc.mean())
    if n_boot > 0:
        reps = sc[rng.integers(0, sc.size, (n_boot, sc.size))].mean(1)
        out.update(lower=float(np.quantile(reps, 0.025)), upper=float(np.quantile(reps, 0.975)))
    return out


def status_of(kind: str, raw: dict[str, Any], resid: dict[str, Any], cfg: InfoConfig, min_repeat: int | None = None) -> tuple[str, str]:
    """Ordered rule: support → raw repeatability → residual repeatability. Reference lines are heuristics."""
    need = cfg.min_repeat if min_repeat is None else min_repeat
    raw_ref, res_ref, chance = ((cfg.raw_icc_ref, cfg.resid_icc_ref, 0.0) if kind == "scalar" else (cfg.raw_auc_ref, cfg.resid_auc_ref, 0.5))
    if raw["n_repeat"] < need or raw["value"] is None:
        return "insufficient_support", f"{raw['n_repeat']} athletes with a repeat (needed {need})"
    if raw["value"] < raw_ref:
        return "unstable", f"raw repeatability {raw['value']:.2f} below the reference line {raw_ref:.2f}"
    if resid["value"] is None or resid["lower"] is None:
        return "insufficient_support", "the stopwatch-adjusted repeatability is not estimable (official results or repeats missing)"
    if resid["lower"] <= chance:
        return "redundant_with_official", (f"repeatable ({raw['value']:.2f}), but nothing repeatable remains after official tests "
                                           f"(residual {resid['value']:.2f}, lower bound {resid['lower']:.2f} ≤ {chance:.1f})")
    if resid["value"] >= res_ref:
        return "repeatable_beyond_official", f"residual repeatability {resid['value']:.2f} [{resid['lower']:.2f}, {resid['upper']:.2f}] ≥ {res_ref:.2f}"
    return "weak_residual_signal", f"residual repeatability {resid['value']:.2f} [{resid['lower']:.2f}, {resid['upper']:.2f}] below {res_ref:.2f}"


def evaluate_feature(feat: Feature, R: dict[str, Any], cfg: InfoConfig, salt: int, keep: np.ndarray | None = None,
                     min_repeat: int | None = None) -> tuple[dict[str, Any], pl.DataFrame | None]:
    """(information row, athlete table of a scalar) of one candidate, optionally inside a subset of attempts (``keep``)."""
    pos_r = np.searchsorted(R["ids"], feat.nfl)
    known = (pos_r < R["ids"].size) & (R["ids"][np.minimum(pos_r, R["ids"].size - 1)] == feat.nfl)
    ok = np.isfinite(feat.V).all(1) & known & (np.ones(feat.nfl.size, bool) if keep is None else keep)
    nfl, V, order, pr = feat.nfl[ok], feat.V[ok], feat.order[ok], pos_r[ok]
    row = {"feature_id": feat.feature_id, "label": feat.label, "kind": feat.kind, "unit": feat.unit,
           "metric": "ICC(1) within position" if feat.kind == "scalar" else "repeat-attempt retrieval AUC within position",
           "n_attempts": int(nfl.size), "n_athletes": 0, "n_repeat": 0, "n_adjusted": 0, "raw_value": None, "raw_lower": None, "raw_upper": None,
           "r2_position": None, "r2_official": None, "residual_value": None, "residual_lower": None, "residual_upper": None,
           "status": "insufficient_support", "reason": "no attempt with a value"}
    if nfl.size < 6:
        return row, None
    att_pos = R["position"][pr]
    ids, first, inv, n = np.unique(nfl, return_index=True, return_inverse=True, return_counts=True)
    mean = np.column_stack([np.bincount(inv, weights=V[:, j]) for j in range(V.shape[1])]) / n[:, None]
    apos, X = att_pos[first], R["X"][pr[first]]
    measure = (lambda v, a, p, o, g: icc1(v[:, 0], a, p, cfg.n_boot, g)) if feat.kind == "scalar" else (
        lambda v, a, p, o, g: retrieval_auc(v, a, p, o, cfg.n_boot, g))
    raw = measure(V, inv, att_pos, order, np.random.default_rng([cfg.seed, salt, 1]))
    use = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1])                  # official forty and 10-yard split are required to adjust
    resid = {"value": None, "lower": None, "upper": None, "n_athletes": 0, "n_repeat": 0}
    pred = np.full(mean.shape, NAN)
    if use.sum() >= max(10, 2 * cfg.k_folds):
        pred[use] = crossfit_predict(mean[use], apos[use], X[use], cfg.k_folds, cfg.seed + salt)
        pred_pos = crossfit_predict(mean[use], apos[use], X[use][:, :0], cfg.k_folds, cfg.seed + salt)
        fin = np.isfinite(pred[use]).all(1)
        tot = float(((mean[use][fin] - mean[use][fin].mean(0)) ** 2).sum())
        if tot > 0:
            row["r2_official"] = 1.0 - float(((mean[use][fin] - pred[use][fin]) ** 2).sum()) / tot
            row["r2_position"] = 1.0 - float(((mean[use][fin] - pred_pos[fin]) ** 2).sum()) / tot
        a_ok = np.isfinite(pred).all(1)[inv]
        if a_ok.sum() >= 6:
            resid = measure(V[a_ok] - pred[inv[a_ok]], inv[a_ok], att_pos[a_ok], order[a_ok], np.random.default_rng([cfg.seed, salt, 2]))
    status, reason = status_of(feat.kind, raw, resid, cfg, min_repeat)
    row.update(n_athletes=int(ids.size), n_repeat=int(raw["n_repeat"]), n_adjusted=int(np.isfinite(pred).all(1).sum()), raw_value=raw["value"],
               raw_lower=raw["lower"], raw_upper=raw["upper"], residual_value=resid["value"], residual_lower=resid["lower"],
               residual_upper=resid["upper"], status=status, reason=reason)
    if feat.kind != "scalar":
        return row, None
    dev = V[:, 0] - mean[inv, 0]
    df_w = int((n - 1).sum())
    s2w = float((dev ** 2).sum() / df_w) if df_w >= 1 else NAN
    table = pl.DataFrame({"feature_id": [feat.feature_id] * ids.size, "nfl_id": ids, "position_group": apos.tolist(), "n_attempts": n.astype(np.int64),
                          "value": mean[:, 0], "expected_from_official": pred[:, 0], "residual": mean[:, 0] - pred[:, 0],
                          "standard_error": np.sqrt(s2w / n) if np.isfinite(s2w) else np.full(ids.size, NAN)}, schema=RESIDUALS.columns)
    return row, table.with_columns([pl.col(c).fill_nan(None) for c in ("expected_from_official", "residual", "standard_error")])


def information_table(features: Sequence[Feature], R: dict[str, Any], cfg: InfoConfig, progress: Any = None) -> dict[str, pl.DataFrame]:
    rows, tables = [], []
    for salt, feat in enumerate(features):
        row, t = evaluate_feature(feat, R, cfg, salt)
        rows.append(row)
        if t is not None:
            tables.append(t)
        if progress:
            progress(f"{feat.feature_id}: {row['n_attempts']} attempts · {row['n_repeat']} athletes with a repeat · {row['status']}")
    return {INFO.name: pl.DataFrame(rows, schema=INFO.columns), RESIDUALS.name: pl.concat(tables) if tables else RESIDUALS.empty()}


def sensitivity_table(features: Sequence[Feature], R: dict[str, Any], cfg: InfoConfig, quality_ok: set[str] | None = None) -> pl.DataFrame:
    """The same analysis inside subsets: recordings without a signal flag, each position group, each draft cohort.
    Small subsets are reported with their counts and no estimate — never dropped silently."""
    rows = []
    for salt, feat in enumerate(features):
        pr = np.minimum(np.searchsorted(R["ids"], feat.nfl), R["ids"].size - 1)
        known = R["ids"][pr] == feat.nfl
        subsets: list[tuple[str, str, np.ndarray]] = []
        if quality_ok is not None:
            subsets.append(("recording_quality", "no signal flag", np.isin(feat.seq, np.asarray(sorted(quality_ok), dtype=str))))
        subsets += [("position_group", str(p), known & (R["position"][pr] == p)) for p in np.unique(R["position"][pr][known])]
        subsets += [("draft_cohort", str(int(y)), known & (R["year"][pr] == y)) for y in np.unique(R["year"][pr][known & np.isfinite(R["year"][pr])])]
        for kind, name, keep in subsets:
            row, _ = evaluate_feature(feat, R, replace(cfg, n_boot=min(cfg.n_boot, 200)), 100 + salt, keep, cfg.min_subset_repeat)
            small = row["n_repeat"] < cfg.min_subset_repeat
            rows.append({"feature_id": feat.feature_id, "subset_kind": kind, "subset": name, "n_athletes": row["n_athletes"], "n_repeat": row["n_repeat"],
                         "raw_value": None if small else row["raw_value"], "residual_value": None if small else row["residual_value"],
                         "residual_lower": None if small else row["residual_lower"], "residual_upper": None if small else row["residual_upper"],
                         "note": f"fewer than {cfg.min_subset_repeat} athletes with a repeat: no estimate" if small else ""})
    return pl.DataFrame(rows, schema=INFO_SENS.columns)


def matched_distinguishability(residuals: pl.DataFrame, pairs: pl.DataFrame, units: dict[str, str]) -> pl.DataFrame:
    """Among the matched stopwatch prospects of 06C (pre-draft rule): how often do two athletes with near-identical
    official results differ in a tracked characteristic by more than the measurement error of both? A pairwise test on
    the DIFFERENCE and its standard error — not a comparison of two individual intervals."""
    rows = []
    for (fid,), t in residuals.partition_by(["feature_id"], as_dict=True).items():
        a = t.select(pl.col("nfl_id").alias("nfl_id_a"), pl.col("value").alias("va"), pl.col("standard_error").alias("sa"))
        b = t.select(pl.col("nfl_id").alias("nfl_id_b"), pl.col("value").alias("vb"), pl.col("standard_error").alias("sb"))
        j = pairs.select("nfl_id_a", "nfl_id_b").join(a, on="nfl_id_a", how="inner").join(b, on="nfl_id_b", how="inner").drop_nulls(["va", "vb", "sa", "sb"])
        if j.height < 5:
            rows.append({"feature_id": fid, "unit": units.get(fid, ""), "n_pairs": j.height, "n_athletes": 0, "median_abs_difference": None,
                         "median_se_difference": None, "share_distinguishable": None})
            continue
        d = (j["va"] - j["vb"]).abs().to_numpy()
        se = np.sqrt(j["sa"].to_numpy() ** 2 + j["sb"].to_numpy() ** 2)
        rows.append({"feature_id": fid, "unit": units.get(fid, ""), "n_pairs": j.height,
                     "n_athletes": len(set(j["nfl_id_a"].to_list()) | set(j["nfl_id_b"].to_list())), "median_abs_difference": float(np.median(d)),
                     "median_se_difference": float(np.median(se)), "share_distinguishable": float((d > 1.96 * se).mean())})
    return pl.DataFrame(rows, schema=MATCH_DIST.columns).sort("feature_id")


# ============================================================================== registry, gate
def evidence_registry(info: pl.DataFrame, sens: pl.DataFrame, definitions: dict[str, str], nfl_evidence: dict[str, str]) -> pl.DataFrame:
    """One row per candidate with the fields a scout needs. ``nfl_evidence``: feature → sentence read from EXECUTED
    transfer tables (Milestone 06, and 08B.2 when it ran); a missing entry is written as 'not tested'."""
    f = lambda x: "n/a" if x is None else f"{x:.2f}"
    rows = []
    for r in info.iter_rows(named=True):
        fid = r["feature_id"]
        s = sens.filter(pl.col("feature_id") == fid)
        q = s.filter(pl.col("subset_kind") == "recording_quality")
        pos = s.filter((pl.col("subset_kind") == "position_group") & pl.col("residual_value").is_not_null())
        use, limit = PRACTICAL.get(fid, ("", ""))
        rows.append({"feature_id": fid, "label": r["label"], "physical_definition": definitions.get(fid, "") + f" [{r['unit']}]",
                     "reference_position": "within position group; estimable for " + (", ".join(pos["subset"].to_list()) or "no single position"),
                     "raw_repeatability": f"{r['metric']}: {f(r['raw_value'])} [{f(r['raw_lower'])}, {f(r['raw_upper'])}]",
                     "residual_repeatability": f"{f(r['residual_value'])} [{f(r['residual_lower'])}, {f(r['residual_upper'])}] after position, forty, 10-yard split, weight, height",
                     "stopwatch_redundancy": f"cross-fitted R² of athlete means: position {f(r['r2_position'])}, position + official tests {f(r['r2_official'])}",
                     "measurement_quality": ("residual repeatability without flagged recordings " + f(q["residual_value"][0])) if q.height else "quality subset not evaluated",
                     "coverage": f"{r['n_athletes']} athletes, {r['n_repeat']} with a repeat, {r['n_adjusted']} with the official results needed to adjust",
                     "nfl_predictive_evidence": nfl_evidence.get(fid, "not tested as a separate model"),
                     "practical_interpretation": use, "limitations": limit + "; " + r["reason"], "status": r["status"]})
    return pl.DataFrame(rows, schema=REGISTRY.columns)


def gate_d(info: pl.DataFrame) -> pl.DataFrame:
    """Gate D — does the feature retain repeatable information beyond traditional tests? Failing marks the feature as
    redundant or unstable; it does not stop the notebook."""
    rows = []
    for r in info.iter_rows(named=True):
        rows += [{"feature_id": r["feature_id"], "criterion": "support", "value": f"{r['n_repeat']} athletes with a repeat", "passed": r["status"] != "insufficient_support"},
                 {"feature_id": r["feature_id"], "criterion": "raw_repeatability", "value": str(r["raw_value"]), "passed": r["status"] not in ("insufficient_support", "unstable")},
                 {"feature_id": r["feature_id"], "criterion": "repeatable_beyond_official_tests", "value": r["reason"], "passed": r["status"] == "repeatable_beyond_official"}]
    return pl.DataFrame(rows, schema=GATE_D.columns)


def conclusions_markdown(info: pl.DataFrame, match: pl.DataFrame, cfg: InfoConfig) -> str:
    f = lambda x: "n/a" if x is None else f"{x:.2f}"
    lines = ["### 08B.4 — incremental Combine movement information", "",
             "| Candidate | Metric | Raw [95 %] | R² official | After official tests [95 %] | Status |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['label']} | {r['metric']} | {f(r['raw_value'])} [{f(r['raw_lower'])}, {f(r['raw_upper'])}] | {f(r['r2_official'])} | "
              f"{f(r['residual_value'])} [{f(r['residual_lower'])}, {f(r['residual_upper'])}] | {r['status']} |" for r in info.iter_rows(named=True)]
    lines += [""] + [f"- `{r['feature_id']}`: {f(None if r['share_distinguishable'] is None else 100 * r['share_distinguishable'])} % of {r['n_pairs']} matched "
                     f"stopwatch pairs differ by more than their measurement error" for r in match.iter_rows(named=True)]
    lines += ["", f"Reference lines (heuristics): scalar raw ICC ≥ {cfg.raw_icc_ref:.2f}, residual ICC ≥ {cfg.resid_icc_ref:.2f} with a positive lower bound; "
                  f"embedding raw AUC ≥ {cfg.raw_auc_ref:.2f}, residual AUC ≥ {cfg.resid_auc_ref:.2f} with a lower bound above 0.5. Repeatable beyond the "
                  "stopwatch means NEW MEASURED INFORMATION about the athlete's run — it says nothing about NFL performance."]
    return "\n".join(lines)


__all__ = ["COVARIATES", "EMBEDDINGS", "Feature", "GATE_D", "INFO", "INFO_SENS", "InfoConfig", "M06_MODEL", "MATCH_DIST", "PRACTICAL", "REGISTRY",
           "RESIDUALS", "SCALARS", "STATUSES", "conclusions_markdown", "crossfit_predict", "evaluate_feature", "evidence_registry", "gate_d",
           "icc1", "inductive_embeddings", "information_table", "matched_distinguishability", "prepare", "retrieval_auc", "roster_arrays",
           "sensitivity_table", "status_of"]
