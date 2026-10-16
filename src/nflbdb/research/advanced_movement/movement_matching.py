"""Milestone 09B — cross-domain movement matching: a shared PHYSICAL feature space between the Combine forty and NFL
movement-demand episodes, and a descriptive prospect-to-demand correspondence. NumPy / Polars.

Five things are kept apart everywhere (columns, labels, text)
    measured Combine capability     a value of the forty, at maximal effort, from rest, with a repeat-attempt error
    NFL movement demand             the distribution of what athletes of a position were OBSERVED to do in games
    descriptive physical similarity where the first sits inside the second — this module's only output
    observed NFL outcome            never an input here
    validated ability to meet a demand   NOT established by any milestone of this project

Shared dimensions (same unit, same 10 Hz sampling, same Savitzky–Golay kinematics, same first-crossing rule
``movement_demands.band_accel`` on both sides)
    acc_2_4 · acc_4_6 · acc_6_8    mean acceleration through a speed band [yd/s²]
    top_speed                      tracked top speed of the forty (atlas p95) against the peak speed of an NFL episode [yd/s]

Reading of ``coverage``: the share of NFL episodes of that demand and position whose intensity is AT OR BELOW the
prospect's measured value. It is a position of a measured maximal-effort value inside an observed game distribution;
it is not a probability, not a score and not a prediction. A demand without a forty analogue ABSTAINS with its reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from . import movement_demands as md
from .scoring.percentiles import FORTY, P95, TEN, V4_V8, percentile_of

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
DESCRIPTIVE = "descriptive physical similarity — not a prediction of NFL performance"
STATES = ("supported", "abstain_not_recorded", "abstain_no_error_estimate", "abstain_thin_nfl_reference", "unsupported_by_forty")


@dataclass(frozen=True)
class MatchingConfig:
    min_nfl_episodes: int = 30          # proposed: NFL episodes of the position × demand × dimension reference
    min_nfl_athletes: int = 10          # proposed
    min_repeat: int = 20                # established: ReferenceSpec.min_repeat (athletes with a repeat for an error estimate)
    z_crit: float = 1.959964
    icc_reference: float = 0.75         # established convention (AtlasConfig.icc_reference)
    icc_floor: float = 0.50             # established convention
    redundancy_reference: float = 0.85  # proposed: in-sample R² on the official times at or above → "mostly stopwatch"
    distances_yd: tuple[float, ...] = (10.0, 20.0, 30.0)
    wide_interval: float = 25.0         # proposed [coverage points]: an interval wider than this is called wide

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CombineFeature:
    feature: str
    label: str
    unit: str
    higher_is_quicker: bool      # orientation of the RAW value (False = a smaller value means quicker: times)
    source: str                  # "sequence_frames" (computed here) or an atlas feature id (reused as executed)
    phase: str                   # early · middle · late · whole
    meaning: str
    registry_id: str = ""        # row of the 08B.4 scouting evidence registry, when the feature is in it


def combine_features(c: MatchingConfig) -> tuple[CombineFeature, ...]:
    band = lambda lo, hi, phase: CombineFeature(
        md.band_col(lo, hi), f"band acceleration {lo:g} → {hi:g} yd/s", "yd/s²", True, "sequence_frames", phase,
        f"({hi:g} − {lo:g}) / time between the first recorded crossings of {lo:g} and {hi:g} yd/s: how hard speed was built through that band")
    return (band(2.0, 4.0, "early"), band(4.0, 6.0, "middle"), band(6.0, 8.0, "late"),
            CombineFeature("time_4_8", "speed development 4 → 8 yd/s", "s", False, V4_V8, "middle",
                           "time for tracked speed to rise from 4 to 8 yd/s (atlas feature, reused as executed)", "F6.forty.time_v4_to_v8_s"),
            CombineFeature("top_speed", "tracked top speed (p95)", "yd/s", True, P95, "late",
                           "95th percentile of tracked speed in the forty (atlas feature, reused as executed)", "F2.forty.p95_speed"),
            *[CombineFeature(f"speed_{d:g}yd", f"speed at {d:g} yd", "yd/s", True, "sequence_frames", "early" if d <= 10 else "middle" if d <= 20 else "late",
                             f"tracked speed when {d:g} yd of path are covered (inside the recorded range only)") for d in c.distances_yd])


SHARED_DIMS = {md.BAND_COLS[0]: md.BAND_COLS[0], md.BAND_COLS[1]: md.BAND_COLS[1], md.BAND_COLS[2]: md.BAND_COLS[2], "top_speed": "v_max"}

ATTEMPTS = TableContract(
    "combine_curve_attempt_features", "Curve features of every eligible forty attempt (long form): band accelerations, speed at distance, reused atlas values.",
    {"seq_id": S, "nfl_id": I, "attempt_order": I, "feature": S, "value": F}, key=("seq_id", "feature"), kind="dataset", area="sequence × feature")
PROSPECT = TableContract(
    "combine_curve_prospect_features", "Athlete value of each Combine curve feature with its repeat-attempt standard error.",
    {"nfl_id": I, "position_group": S, "feature": S, "label": S, "unit": S, "value": F, "n_attempts": I, "attempt_min": F, "attempt_max": F,
     "standard_error": F, "pct_position": F, "n_reference": I, "state": S}, key=("nfl_id", "feature"), kind="dataset", area="athlete × feature",
    docs={"value": "mean of the athlete's attempts with a value", "standard_error": "pooled within-athlete SD of the position / √n_attempts (null: not estimable)",
          "pct_position": "RAW mid-rank percentile among the other athletes of the position (not oriented: see higher_is_quicker in the evidence table)",
          "state": "supported · not_observed · insufficient_evidence"})
EVIDENCE = TableContract(
    "combine_curve_feature_evidence", "Local evidence screen of each Combine curve feature: repeatability, stopwatch redundancy, evidence label.",
    {"feature": S, "label": S, "unit": S, "phase": S, "higher_is_quicker": B, "meaning": S, "n_athletes": I, "n_repeat": I, "icc_within_position": F,
     "within_athlete_sd": F, "r2_official_times": F, "repeatability_label": S, "redundancy_label": S, "evidence_status": S, "evidence_source": S,
     "football_value": S}, key=("feature",), area="feature",
    docs={"icc_within_position": "one-way ICC of attempts after removing position means (method of moments)",
          "r2_official_times": "IN-SAMPLE R² of the athlete mean on official forty + 10-yard split, within position (not cross-fitted)",
          "evidence_status": "the 08B.4 registry status when the feature is in it; otherwise the local screen of this table",
          "football_value": "always 'not established': repeatability is not football value"})
DICTIONARY = TableContract(
    "movement_shared_feature_dictionary", "The shared Combine / NFL physical dimensions and why each pair is (not) comparable.",
    {"dimension": S, "unit": S, "combine_definition": S, "nfl_definition": S, "demands": S, "sampling": S, "smoothing": S, "window_rule": S,
     "comparable": B, "caveat": S}, key=("dimension",), area="dimension")
SIMILARITY = TableContract(
    "movement_similarity", "Prospect × NFL demand × shared dimension: where the measured Combine value sits in the observed NFL demand distribution.",
    {"nfl_id": I, "position_group": S, "demand_id": S, "dimension": S, "unit": S, "combine_value": F, "standard_error": F, "nfl_q10": F, "nfl_q50": F,
     "nfl_q90": F, "n_nfl_episodes": I, "n_nfl_athletes": I, "coverage": F, "coverage_lo": F, "coverage_hi": F, "robust_z": F, "state": S, "reason": S},
    key=("nfl_id", "demand_id", "dimension"), kind="dataset", area="athlete × demand × dimension",
    docs={"coverage": "100 × share of NFL episodes (same position, same demand) with a value ≤ the prospect's Combine value",
          "coverage_lo": "coverage at the Combine value − 1.96 standard errors (measurement uncertainty of the prospect only)",
          "robust_z": "(Combine value − NFL median) / (NFL IQR / 1.349)", "state": " · ".join(STATES)})
MATRIX = TableContract(
    "movement_similarity_matrix", "Prospect × NFL demand summary of the supported shared dimensions (descriptive).",
    {"nfl_id": I, "position_group": S, "demand_id": S, "n_dimensions": I, "mean_coverage": F, "coverage_lo": F, "coverage_hi": F, "distance": F,
     "state": S, "reading": S}, key=("nfl_id", "demand_id"), kind="dataset", area="athlete × demand",
    docs={"distance": "root mean square of robust_z over the supported dimensions (0 = at the NFL median of every dimension)",
          "reading": DESCRIPTIVE})
DIM_SUMMARY = TableContract(
    "movement_matching_summary", "Matching by position × demand × dimension: support, uncertainty, attempt stability, relation to the official forty.",
    {"position_group": S, "demand_id": S, "dimension": S, "n_prospects": I, "n_abstain": I, "median_coverage": F, "q10_coverage": F, "q90_coverage": F,
     "median_interval_width": F, "share_wide_interval": F, "share_interval_spans_50": F, "n_two_attempts": I, "median_attempt_shift": F,
     "spearman_with_forty": F}, key=("position_group", "demand_id", "dimension"), area="position group × demand × dimension",
    docs={"share_interval_spans_50": "prospects whose coverage interval contains 50: measurement error alone moves them across the NFL median",
          "median_attempt_shift": "median |coverage of attempt 1 − coverage of attempt 2| (athletes with two attempts)",
          "spearman_with_forty": "rank correlation of coverage with the official forty: near −1 = the stopwatch already orders prospects this way"})
CONTRACTS = (ATTEMPTS, PROSPECT, EVIDENCE, DICTIONARY, SIMILARITY, MATRIX, DIM_SUMMARY)


# ============================================================================== Combine side
def trace_features(t: np.ndarray, v: np.ndarray, d: np.ndarray, c: MatchingConfig) -> dict[str, float]:
    """Band accelerations and speed at distance of ONE recorded forty trace (frames in time order). NaN = not recorded."""
    out = {name: md.band_accel(t, v, lo, hi) for (lo, hi), name in zip(md.BANDS, md.BAND_COLS)}
    ok = np.isfinite(d) & np.isfinite(v)
    dd, vv = d[ok], v[ok]
    for x in c.distances_yd:
        inside = dd.size >= 2 and dd[0] <= x <= dd[-1] and bool(np.all(np.diff(dd) >= 0))
        out[f"speed_{x:g}yd"] = float(np.interp(x, dd, vv)) if inside else NAN
    return out


def attempt_table(frames: pl.DataFrame, attempt_values: pl.DataFrame, windows: pl.DataFrame, c: MatchingConfig) -> pl.DataFrame:
    """Long attempt-feature table. ``frames``: ``sequence_frames`` · ``attempt_values``: ``atlas_attempt_features``
    (the two reused atlas features) · ``windows``: ``sequence_window_flags`` (attempt order)."""
    order = dict(zip(windows["seq_id"].to_list(), windows["attempt_order"].to_list()))
    rows = []
    for (sid,), g in frames.sort("seq_id", "frame").partition_by(["seq_id"], as_dict=True, maintain_order=True).items():
        num = lambda k: g[k].cast(pl.Float64).fill_null(NAN).to_numpy().astype(float)
        feats = trace_features(num("t_rel"), num("speed"), num("dist"), c)
        nid = int(g["nfl_id"][0])
        rows += [{"seq_id": sid, "nfl_id": nid, "attempt_order": order.get(sid), "feature": k, "value": v if np.isfinite(v) else None} for k, v in feats.items()]
    reuse = {f.source: f.feature for f in combine_features(c) if f.source != "sequence_frames"}
    a = attempt_values.filter(pl.col("feature_id").is_in(list(reuse)))
    rows += [{"seq_id": r["seq_id"], "nfl_id": r["nfl_id"], "attempt_order": order.get(r["seq_id"]), "feature": reuse[r["feature_id"]],
              "value": r["value"] if r["value"] is not None and np.isfinite(r["value"]) else None} for r in a.iter_rows(named=True)]
    out = pl.DataFrame(rows, schema=ATTEMPTS.columns) if rows else ATTEMPTS.empty()
    return out.filter(pl.col("attempt_order").is_not_null()).unique(subset=list(ATTEMPTS.key), maintain_order=True).sort("seq_id", "feature")


def _components(values: np.ndarray, ids: np.ndarray, pos: np.ndarray) -> tuple[float, float, int]:
    """(within-athlete SD, ICC, athletes with a repeat) of attempt values after removing position means."""
    if values.size < 4:
        return NAN, NAN, 0
    y = values.astype(float).copy()
    for p in np.unique(pos):
        k = pos == p
        y[k] -= y[k].mean()
    w, _, share = md.variance_components(y, ids)
    return w, share, int((np.bincount(np.unique(ids, return_inverse=True)[1]) >= 2).sum())


def _r2_within_position(y: np.ndarray, X: np.ndarray, pos: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X, pos = y[ok].copy(), X[ok].copy(), pos[ok]
    if y.size < X.shape[1] + 5:
        return NAN
    for p in np.unique(pos):
        k = pos == p
        y[k] -= y[k].mean()
        X[k] -= X[k].mean(0)
    tot = float((y ** 2).sum())
    if tot <= 0:
        return NAN
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1.0 - ((y - X @ beta) ** 2).sum() / tot)


def prospect_tables(attempts: pl.DataFrame, roster: pl.DataFrame, c: MatchingConfig, registry: pl.DataFrame | None = None
                    ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(prospect features, feature evidence). ``roster``: nfl_id, position_group, official_forty, official_ten (the 07D
    profile table). ``registry``: the executed 08B.4 scouting evidence registry (optional). Pre-draft information only."""
    roster = roster.drop_nulls(["position_group"])
    pos_of = dict(zip(roster["nfl_id"].to_list(), roster["position_group"].to_list()))
    forty = dict(zip(roster["nfl_id"].to_list(), roster["official_forty"].to_list()))
    ten = dict(zip(roster["nfl_id"].to_list(), roster["official_ten"].to_list()))
    reg = {r["feature_id"]: r["status"] for r in registry.iter_rows(named=True)} if registry is not None else {}
    prows, erows = [], []
    for f in combine_features(c):
        a = attempts.filter((pl.col("feature") == f.feature) & pl.col("value").is_not_null() & pl.col("nfl_id").is_in(list(pos_of))).sort("nfl_id", "attempt_order")
        ids, vals = a["nfl_id"].to_numpy(), a["value"].to_numpy().astype(float)
        pos = np.asarray([pos_of[int(i)] for i in ids], dtype=str)
        sd_by_pos, n_rep_by_pos = {}, {}
        for p in np.unique(pos):
            k = pos == p
            w, _, n_rep = _components(vals[k], ids[k], pos[k])
            sd_by_pos[p], n_rep_by_pos[p] = w, n_rep
        w_all, icc, n_rep = _components(vals, ids, pos)
        g = a.group_by("nfl_id", maintain_order=True).agg(pl.col("value").mean().alias("m"), pl.len().alias("n"), pl.col("value").min().alias("lo"),
                                                          pl.col("value").max().alias("hi"))
        g_ids, g_m = g["nfl_id"].to_numpy(), g["m"].to_numpy().astype(float)
        g_pos = np.asarray([pos_of[int(i)] for i in g_ids], dtype=str)
        have = set()
        for i, m, n, lo, hi, p in zip(g_ids.tolist(), g_m.tolist(), g["n"].to_list(), g["lo"].to_list(), g["hi"].to_list(), g_pos.tolist()):
            ref = g_m[(g_pos == p) & (g_ids != i)]
            sd = sd_by_pos.get(p, NAN)
            se = float(sd / np.sqrt(n)) if np.isfinite(sd) and sd > 0 and n_rep_by_pos.get(p, 0) >= c.min_repeat else None
            have.add(i)
            prows.append({"nfl_id": int(i), "position_group": p, "feature": f.feature, "label": f.label, "unit": f.unit, "value": m, "n_attempts": int(n),
                          "attempt_min": lo, "attempt_max": hi, "standard_error": se, "pct_position": percentile_of(m, ref) if ref.size >= 5 else None,
                          "n_reference": int(ref.size), "state": "supported" if se is not None else "insufficient_evidence"})
        prows += [{"nfl_id": int(i), "position_group": p, "feature": f.feature, "label": f.label, "unit": f.unit, "value": None, "n_attempts": 0,
                   "attempt_min": None, "attempt_max": None, "standard_error": None, "pct_position": None, "n_reference": 0, "state": "not_observed"}
                  for i, p in pos_of.items() if i not in have]
        X = np.array([[forty.get(int(i)) or NAN, ten.get(int(i)) or NAN] for i in g_ids], dtype=float) if g_ids.size else np.empty((0, 2))
        r2 = _r2_within_position(g_m, X, g_pos) if g_ids.size else NAN
        rep = ("insufficient_support" if n_rep < c.min_repeat or not np.isfinite(icc) else "repeatable" if icc >= c.icc_reference
               else "moderately_repeatable" if icc >= c.icc_floor else "unstable")
        red = "not_estimable" if not np.isfinite(r2) else "mostly_stopwatch" if r2 >= c.redundancy_reference else "partly_beyond_stopwatch"
        local = rep if rep in ("insufficient_support", "unstable") else f"{rep}_{red}"
        erows.append({"feature": f.feature, "label": f.label, "unit": f.unit, "phase": f.phase, "higher_is_quicker": f.higher_is_quicker, "meaning": f.meaning,
                      "n_athletes": int(g_ids.size), "n_repeat": int(n_rep), "icc_within_position": icc if np.isfinite(icc) else None,
                      "within_athlete_sd": w_all if np.isfinite(w_all) else None, "r2_official_times": r2 if np.isfinite(r2) else None,
                      "repeatability_label": rep, "redundancy_label": red, "evidence_status": reg.get(f.registry_id, local),
                      "evidence_source": "08B.4 scouting evidence registry (residual repeatability, cross-fitted)" if f.registry_id in reg
                      else "09B local screen (raw ICC and in-sample R²; weaker than the 08B.4 method)", "football_value": "not established"})
    prospect = pl.DataFrame(prows, schema=PROSPECT.columns) if prows else PROSPECT.empty()
    return prospect.with_columns(pl.col("pct_position").fill_nan(None)).sort("nfl_id", "feature"), pl.DataFrame(erows, schema=EVIDENCE.columns)


# ============================================================================== shared dictionary
def dictionary_table(c: md.DemandConfig, smoothing: str) -> pl.DataFrame:
    specs = md.demand_specs(c)
    used = lambda dim: ", ".join(s.demand_id for s in specs if dim in [x.strip() for x in s.combine_analogue.split(",")])
    rows = []
    for (lo, hi), name in zip(md.BANDS, md.BAND_COLS):
        rows.append({"dimension": name, "unit": "yd/s²",
                     "combine_definition": f"forty-yard dash: ({hi:g} − {lo:g}) / time between the first recorded crossings of {lo:g} and {hi:g} yd/s",
                     "nfl_definition": f"accelerating episode: the same quantity between the first crossings INSIDE the episode", "demands": used(name),
                     "sampling": "10 Hz on both sides", "smoothing": smoothing, "window_rule": "movement_demands.band_accel on both sides; null unless both crossings are recorded",
                     "comparable": True, "caveat": "the forty is one maximal-effort transit from rest; an NFL episode is a context-driven, usually submaximal effort"})
    rows.append({"dimension": "top_speed", "unit": "yd/s", "combine_definition": "95th percentile of tracked speed in the forty (atlas F2.forty.p95_speed)",
                 "nfl_definition": "largest speed of a sustained high-speed episode", "demands": used("top_speed"), "sampling": "10 Hz on both sides", "smoothing": smoothing,
                 "window_rule": "a percentile of a 40-yard run against the peak of an episode: both are largest-speed summaries, not the same statistic",
                 "comparable": True, "caveat": "40 yards may end before an athlete's top speed; game speed is not a maximum"})
    for s in specs:
        if not s.combine_analogue:
            rows.append({"dimension": f"({s.demand_id})", "unit": s.unit, "combine_definition": "not measured by the forty-yard dash",
                         "nfl_definition": s.definition, "demands": s.demand_id, "sampling": "10 Hz", "smoothing": smoothing, "window_rule": "—",
                         "comparable": False, "caveat": s.comparability})
    return pl.DataFrame(rows, schema=DICTIONARY.columns)


# ============================================================================== similarity
def nfl_reference(episodes: pl.DataFrame, c: md.DemandConfig) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Sorted NFL episode values of every (position, demand, shared dimension) with a forty analogue."""
    ref = {}
    d = episodes.filter(~pl.col("edge"))
    for s in md.demand_specs(c):
        dims = [x.strip() for x in s.combine_analogue.split(",") if x.strip()]
        g = d.filter(pl.col("demand_id") == s.demand_id)
        for dim in dims:
            col = SHARED_DIMS[dim]
            for (pos,), sub in g.filter(pl.col(col).is_not_null()).partition_by(["position_group"], as_dict=True).items():
                ref[(pos, s.demand_id, dim)] = {"values": np.sort(sub[col].to_numpy().astype(float)), "n_athletes": int(sub["nfl_id"].n_unique())}
    return ref


def coverage_of(x: float, sorted_ref: np.ndarray) -> float:
    return float(100.0 * np.searchsorted(sorted_ref, x, side="right") / sorted_ref.size) if sorted_ref.size and np.isfinite(x) else NAN


def similarity_table(prospect: pl.DataFrame, episodes: pl.DataFrame, dc: md.DemandConfig, c: MatchingConfig) -> pl.DataFrame:
    """One row per prospect × demand × shared dimension, and one abstention row per demand without a forty analogue.
    ``prospect`` may not hold an NFL quantity: only Combine values and their errors are read from it."""
    ref = nfl_reference(episodes, dc)
    val = {(r["nfl_id"], r["feature"]): r for r in prospect.iter_rows(named=True)}
    athletes = prospect.select("nfl_id", "position_group").unique(subset="nfl_id").sort("nfl_id")
    rows = []
    for a in athletes.iter_rows(named=True):
        i, pos = a["nfl_id"], a["position_group"]
        for s in md.demand_specs(dc):
            dims = [x.strip() for x in s.combine_analogue.split(",") if x.strip()]
            base = {"nfl_id": i, "position_group": pos, "demand_id": s.demand_id}
            if not dims:
                rows.append({**base, "dimension": "—", "unit": s.unit, "state": "unsupported_by_forty", "reason": s.comparability})
                continue
            for dim in dims:
                r, rr = val.get((i, dim), {}), ref.get((pos, s.demand_id, dim))
                row = {**base, "dimension": dim, "unit": "yd/s" if dim == "top_speed" else "yd/s²", "combine_value": r.get("value"),
                       "standard_error": r.get("standard_error")}
                if rr is not None:
                    q = np.quantile(rr["values"], [0.1, 0.25, 0.5, 0.75, 0.9])
                    row |= {"nfl_q10": float(q[0]), "nfl_q50": float(q[2]), "nfl_q90": float(q[4]), "n_nfl_episodes": int(rr["values"].size),
                            "n_nfl_athletes": rr["n_athletes"]}
                if r.get("value") is None:
                    row |= {"state": "abstain_not_recorded", "reason": "the band crossing (or the value) was not recorded in this athlete's forty"}
                elif rr is None or rr["values"].size < c.min_nfl_episodes or rr["n_athletes"] < c.min_nfl_athletes:
                    row |= {"state": "abstain_thin_nfl_reference", "reason": f"fewer than {c.min_nfl_episodes} NFL episodes or {c.min_nfl_athletes} athletes of this position"}
                else:
                    x, se = float(r["value"]), r.get("standard_error")
                    iqr = float(q[3] - q[1])
                    row |= {"coverage": coverage_of(x, rr["values"]), "robust_z": (x - float(q[2])) / (iqr / 1.349) if iqr > 0 else None}
                    if se is None:
                        row |= {"state": "abstain_no_error_estimate", "reason": "value shown without an interval: repeat-attempt error not estimable for the position"}
                    else:
                        row |= {"coverage_lo": coverage_of(x - c.z_crit * se, rr["values"]), "coverage_hi": coverage_of(x + c.z_crit * se, rr["values"]),
                                "state": "supported", "reason": DESCRIPTIVE}
                rows.append(row)
    return pl.DataFrame(rows, schema=SIMILARITY.columns) if rows else SIMILARITY.empty()


def matrix_table(sim: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (i, dem), g in sim.partition_by(["nfl_id", "demand_id"], as_dict=True, maintain_order=True).items():
        ok = g.filter(pl.col("state") == "supported")
        row = {"nfl_id": i, "position_group": g["position_group"][0], "demand_id": dem, "n_dimensions": ok.height, "reading": DESCRIPTIVE}
        if ok.height:
            z = ok["robust_z"].drop_nulls().to_numpy()
            row |= {"mean_coverage": float(ok["coverage"].mean()), "coverage_lo": float(ok["coverage_lo"].mean()), "coverage_hi": float(ok["coverage_hi"].mean()),
                    "distance": float(np.sqrt(np.mean(z ** 2))) if z.size else None, "state": "supported"}
        else:
            states = g["state"].unique().to_list()
            row |= {"state": states[0] if len(states) == 1 else "abstain_not_recorded"}
        rows.append(row)
    return pl.DataFrame(rows, schema=MATRIX.columns) if rows else MATRIX.empty()


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return None
    ra, rb = pl.Series(a[ok]).rank("average").to_numpy(), pl.Series(b[ok]).rank("average").to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


def summary_table(sim: pl.DataFrame, attempts: pl.DataFrame, episodes: pl.DataFrame, roster: pl.DataFrame, dc: md.DemandConfig, c: MatchingConfig) -> pl.DataFrame:
    """Support, uncertainty, attempt stability and relation to the official forty, by position × demand × dimension."""
    ref = nfl_reference(episodes, dc)
    forty = dict(zip(roster["nfl_id"].to_list(), roster["official_forty"].to_list()))
    first_two = (attempts.filter(pl.col("value").is_not_null()).sort("nfl_id", "feature", "attempt_order").group_by("nfl_id", "feature", maintain_order=True)
                 .agg(pl.col("value").head(2)).filter(pl.col("value").list.len() == 2))
    pair = {(r["nfl_id"], r["feature"]): r["value"] for r in first_two.iter_rows(named=True)}
    rows = []
    d = sim.filter(pl.col("dimension") != "—")
    for (pos, dem, dim), g in sorted(d.partition_by(["position_group", "demand_id", "dimension"], as_dict=True).items()):
        ok = g.filter(pl.col("state") == "supported")
        row = {"position_group": pos, "demand_id": dem, "dimension": dim, "n_prospects": ok.height, "n_abstain": g.height - ok.height, "n_two_attempts": 0}
        rr = ref.get((pos, dem, dim))
        if ok.height:
            cov, lo, hi = (ok[k].to_numpy().astype(float) for k in ("coverage", "coverage_lo", "coverage_hi"))
            q = np.quantile(cov, [0.1, 0.5, 0.9])
            shifts = [abs(coverage_of(v[0], rr["values"]) - coverage_of(v[1], rr["values"])) for i in ok["nfl_id"].to_list() if (v := pair.get((i, dim))) is not None]
            f = np.array([forty.get(i) if forty.get(i) is not None else NAN for i in ok["nfl_id"].to_list()], dtype=float)
            row |= {"median_coverage": float(q[1]), "q10_coverage": float(q[0]), "q90_coverage": float(q[2]), "median_interval_width": float(np.median(hi - lo)),
                    "share_wide_interval": float(((hi - lo) > c.wide_interval).mean()), "share_interval_spans_50": float(((lo <= 50) & (hi >= 50)).mean()),
                    "n_two_attempts": len(shifts), "median_attempt_shift": float(np.median(shifts)) if shifts else None, "spearman_with_forty": _spearman(cov, f)}
        rows.append(row)
    return pl.DataFrame(rows, schema=DIM_SUMMARY.columns) if rows else DIM_SUMMARY.empty()


def assert_pre_draft(df: pl.DataFrame, what: str) -> None:
    """A Combine-side table may not hold an NFL quantity (same rule as the 07D scorecard)."""
    from .scoring.scorecards import assert_no_nfl_columns

    assert_no_nfl_columns(df, what)


def conclusions(evidence: pl.DataFrame, summary: pl.DataFrame, matrix: pl.DataFrame) -> list[str]:
    out = []
    for r in evidence.iter_rows(named=True):
        icc = "n/a" if r["icc_within_position"] is None else f"{r['icc_within_position']:.2f}"
        r2 = "n/a" if r["r2_official_times"] is None else f"{r['r2_official_times']:.2f}"
        out.append(f"{r['label']}: ICC {icc} ({r['n_repeat']} athletes with a repeat), R² on official times {r2} → {r['evidence_status']}.")
    for r in summary.filter(pl.col("n_prospects") > 0).iter_rows(named=True):
        rho = "n/a" if r["spearman_with_forty"] is None else f"{r['spearman_with_forty']:+.2f}"
        shift = "n/a" if r["median_attempt_shift"] is None else f"{r['median_attempt_shift']:.0f}"
        out.append(f"{r['position_group']} · {r['demand_id']} · {r['dimension']}: {r['n_prospects']} prospects, median coverage {r['median_coverage']:.0f}, "
                   f"interval spans the NFL median for {100 * r['share_interval_spans_50']:.0f} %, attempt shift {shift} points, ρ with the forty {rho}.")
    n_abs = matrix.filter(pl.col("state") == "unsupported_by_forty")["demand_id"].unique().to_list()
    out.append("Demands the forty cannot describe (abstained for every prospect): " + (", ".join(sorted(n_abs)) or "none") + ".")
    return out


__all__ = ["ATTEMPTS", "CONTRACTS", "CombineFeature", "DESCRIPTIVE", "DICTIONARY", "DIM_SUMMARY", "EVIDENCE", "MATRIX", "MatchingConfig", "PROSPECT",
           "SHARED_DIMS", "SIMILARITY", "STATES", "assert_pre_draft", "attempt_table", "combine_features", "conclusions", "coverage_of", "dictionary_table",
           "matrix_table", "nfl_reference", "prospect_tables", "similarity_table", "summary_table", "trace_features"]
