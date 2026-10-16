"""Scorecard assembly (Milestone 07D): percentile tables, evidence confidence, profiles, Combine-side case selection.

Four sections, never merged into one number
    A  Speed Capacity      official forty (headline), official 10-yard split, tracked top speed (components)
    B  Speed Development   tracked time 4 → 8 yd/s — only where the interval was recorded and the position supports it
    C  Movement Fingerprint  similarity in the frozen TCN-B representation (who he moves like, how stable that is)
    D  Evidence Confidence   how much and how good the recording is — SEPARATE from athletic quality
Rules  no overall score · no NFL value as an input · a missing measurement abstains (never the position average) ·
       cases are chosen by Combine-side criteria fixed in advance.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ...contracts import TableContract
from ...sprint_rescue.stats import seed_for
from . import NFL_PREDICTIVE_STATUS
from . import uncertainty as un
from .percentiles import (FEATURE, FEATURES, FORTY, P95, PCT, TEN, V4_V8, FeatureSpec, ReferenceSpec, fold_percentiles, orient,
                          percentile_of, position_percentiles, reference_mask)

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
NFL_MARKERS = ("nfl_", "accel_sust", "band_acc", "speed_p90", "o1.", "o2.", "o3.", "target", "oof", "pred", "n_windows", "n_games")
NFL_ALLOWED = ("nfl_id", "nfl_predictive_status")
TIERS = ("high", "medium", "low", "official_only")


class LeakageError(RuntimeError):
    """An NFL-derived column reached a pre-draft table."""


def assert_no_nfl_columns(df: pl.DataFrame, what: str) -> None:
    """A pre-draft table may not hold, or be named like, an NFL quantity. Raises ``LeakageError``."""
    bad = [c for c in df.columns if c not in NFL_ALLOWED and not c.startswith("nfl_id") and any(m in c.lower() for m in NFL_MARKERS)]
    if bad:
        raise LeakageError(f"{what}: column(s) named like an NFL quantity {bad}. Pre-draft profiles use Combine information only")


# ============================================================================== contracts
CONFIDENCE = TableContract(
    "combine_measurement_confidence", "Evidence confidence of every Combine athlete: how much and how good the recording is.",
    {"nfl_id": I, "position_group": S, "draft_year": I, "n_attempts": I, "n_development_attempts": I, "n_signal_ok": I,
     "any_start_in_motion": B, "max_first_speed": F, "any_end_censored": B, "max_position_residual": F, "repeat_available": B,
     "fingerprint_inductive": B, "fingerprint_self_auc": F, "seed_overlap": F, "reference_n": I, "evidence_tier": S, "reasons": S},
    key=("nfl_id",), area="athlete",
    docs={"evidence_tier": "high · medium · low · official_only — about the MEASUREMENT, never about the athlete's ability",
          "any_start_in_motion": "a recording starts with the athlete already above the reference speed",
          "max_position_residual": "largest RMS distance between raw and smoothed position over his attempts [yd] (sensor noise)"})
PROFILE = TableContract(
    "combine_athletic_scorecard_profiles", "One pre-draft descriptive profile per Combine athlete (four sections, no overall score).",
    {"nfl_id": I, "display_name": S, "position_group": S, "draft_year": I, "official_forty": F, "official_ten": F, "weight": F,
     "height": F, "speed_capacity_pct": F, "speed_capacity_lower": F, "speed_capacity_upper": F, "ten_pct": F, "top_speed_pct": F,
     "top_speed_lower": F, "top_speed_upper": F, "speed_agreement": F, "speed_development_value": F, "speed_development_pct": F,
     "speed_development_lower": F, "speed_development_upper": F, "speed_development_availability": S, "fingerprint_available": B,
     "fingerprint_self_auc": F, "fingerprint_nearest": S, "fingerprint_nearest_similar_official": S, "evidence_tier": S,
     "evidence_reasons": S, "reference_n": I, "nfl_predictive_status": S}, key=("nfl_id",), kind="dataset", area="athlete",
    docs={"speed_capacity_pct": "oriented position percentile of the official forty (the single headline of section A)",
          "speed_agreement": "largest minus smallest of the three oriented speed percentiles (they are not averaged)",
          "speed_development_pct": "None when the 4 → 8 yd/s interval was not recorded or the position does not support the measure",
          "fingerprint_nearest": "nearest athletes of the position in the TCN-B representation (ids, nearest first)"})
EXAMPLES = TableContract(
    "combine_scorecard_examples", "Case studies chosen by Combine-side criteria fixed in advance (illustrations, not evidence).",
    {"case": S, "criterion": S, "position_group": S, "nfl_id_a": I, "nfl_id_b": I, "name_a": S, "name_b": S, "matched_on": S,
     "contrast": S, "value_a": F, "value_b": F, "contrast_size": F, "fingerprint_distance": F, "selected_using_nfl": B},
    key=("case", "nfl_id_a", "nfl_id_b"), area="case × pair")
STABILITY_BY_QUALITY = TableContract(
    "combine_quality_ranking_stability", "Does the rank of an athlete hold between his attempts, by evidence tier?",
    {"feature_id": S, "evidence_tier": S, "n_athletes": I, "rank_agreement": F, "ci_lower": F, "ci_upper": F,
     "mean_abs_percentile_shift": F}, key=("feature_id", "evidence_tier"), area="feature × evidence tier",
    docs={"rank_agreement": "within-position Spearman correlation of attempt 1 and attempt 2 (athletes with a repeat)",
          "mean_abs_percentile_shift": "mean |percentile of attempt 1 − percentile of attempt 2| against the reference"})


# ============================================================================== section tables A and B
def percentile_table(spec: FeatureSpec, ids: np.ndarray, position: np.ndarray, draft_year: np.ndarray, value: np.ndarray,
                     n_attempts: np.ndarray, quality_ok: np.ndarray, fold: np.ndarray, ref: ReferenceSpec, seed: int,
                     attempts: tuple[np.ndarray, np.ndarray] | None = None, n_draws: int = 1000) -> pl.DataFrame:
    """Percentile rows of ONE feature for every athlete. ``attempts``: (athlete id, value) per attempt for a tracked
    feature (measurement model); None for an official single value. Reference statistics (variance components,
    availability) come from the reference athletes only."""
    in_ref = reference_mask(position, draft_year, quality_ok, ref) & np.isfinite(value)
    pct, n_ref = position_percentiles(value, position, in_ref, spec.higher_is_better, ref.min_reference)
    pct_fold = fold_percentiles(value, position, in_ref, fold, spec.higher_is_better, ref.min_reference)
    se, rel, shrunk = np.full(ids.size, NAN), np.full(ids.size, NAN), value.copy()
    avail, note = np.full(ids.size, "unavailable", dtype=object), np.full(ids.size, "", dtype=object)
    for p in np.unique(position):
        k = position == p
        comp = {"s2_within": NAN, "s2_between": NAN, "mu": NAN, "n_repeat": 0}
        if attempts is not None:
            a_id, a_val = attempts
            ref_ids = ids[k & in_ref]
            comp = un.attempt_components(a_val, a_id, np.isin(a_id, ref_ids))
            se[k], rel[k] = un.standard_error(comp, n_attempts[k]), un.reliability_of_mean(comp, n_attempts[k])
            shrunk[k] = un.shrunk_estimate(value[k], n_attempts[k], comp)
        med = float(np.median(n_attempts[k & in_ref])) if (k & in_ref).any() else 0.0
        a, why = un.availability(comp, int((k & in_ref).sum()), med, ref, tracked=attempts is not None)
        avail[k], note[k] = a, why
    lo, hi, pct_s = np.full(ids.size, NAN), np.full(ids.size, NAN), np.full(ids.size, NAN)
    rng = np.random.default_rng(seed_for(seed, "score", spec.feature_id))
    for i in range(ids.size):
        if not np.isfinite(pct[i]):
            continue
        r = (position == position[i]) & in_ref
        r[i] = False
        lo[i], hi[i] = un.percentile_interval(float(value[i]), float(se[i]), value[r], spec.higher_is_better, rng, n_draws)
        pct_s[i] = orient(percentile_of(float(shrunk[i]), shrunk[r]), spec.higher_is_better)
    missing = ~np.isfinite(value)
    note = np.where(missing, "no value recorded: no percentile (never imputed)", note)
    avail = np.where(missing, "unavailable", avail)
    hide = avail == "unavailable"
    for arr in (pct, lo, hi, pct_s, pct_fold):
        arr[hide] = NAN
    unc = ("repeat-attempt standard error + reference sampling" if attempts is not None
           else "reference sampling only (single official value: measurement error not estimable)")
    d = pl.DataFrame({"nfl_id": ids, "position_group": position, "draft_year": draft_year, "value": value, "percentile": pct, "pct_lower": lo,
                      "pct_upper": hi, "pct_reliability_aware": pct_s, "pct_fold_isolated": pct_fold, "n_reference": n_ref, "in_reference": in_ref,
                      "n_attempts": n_attempts, "standard_error": se, "reliability": rel, "availability": avail.astype(str), "note": note.astype(str)})
    d = d.with_columns(pl.lit(spec.section).alias("section"), pl.lit(spec.feature_id).alias("feature_id"), pl.lit(spec.label).alias("feature_label"),
                       pl.lit(spec.role).alias("role"), pl.lit(spec.unit).alias("unit"), pl.lit(spec.higher_is_better).alias("higher_is_better"),
                       pl.lit(unc).alias("measurement_uncertainty"),
                       *[pl.col(c).fill_nan(None) for c in ("value", "percentile", "pct_lower", "pct_upper", "pct_reliability_aware",
                                                            "pct_fold_isolated", "standard_error", "reliability")])
    return d.select([pl.col(c).cast(t) for c, t in PCT.columns.items()])


# ============================================================================== section D
def confidence_table(roster: pl.DataFrame, windows: pl.DataFrame, dev_attempts: dict[int, int], stability: pl.DataFrame,
                     reference_n: dict[str, int], min_reference: int) -> pl.DataFrame:
    """Evidence confidence per athlete from recording facts only. ``roster``: nfl_id, position_group, draft_year.
    ``windows``: sequence window flags of the eligible forty attempts. Rule (fixed, explicit):
        high           ≥ 2 attempts, all with usable signal, none starting in motion, speed development on ≥ 2 of them,
                       position reference ≥ ``min_reference``
        medium         ≥ 1 attempt with usable signal and speed development measured
        low            tracked, but neither of the above
        official_only  no eligible tracked attempt"""
    w = windows.group_by("nfl_id").agg(
        pl.len().cast(pl.Int64).alias("n_attempts"), pl.col("S2_signal").cast(pl.Int64).sum().alias("n_signal_ok"),
        pl.col("start_above_ref_speed").cast(pl.Boolean).any().alias("any_start_in_motion"), pl.col("first_speed").max().alias("max_first_speed"),
        pl.col("last_frame_censored").cast(pl.Boolean).any().alias("any_end_censored"), pl.col("pos_resid_rms").max().alias("max_position_residual"))
    st = stability.select("nfl_id", pl.col("inductive").alias("fingerprint_inductive"), pl.col("self_auc").alias("fingerprint_self_auc"), "seed_overlap")
    d = roster.select("nfl_id", "position_group", "draft_year").join(w, on="nfl_id", how="left").join(st, on="nfl_id", how="left")
    rows = []
    for r in d.iter_rows(named=True):
        n, ok = r["n_attempts"] or 0, r["n_signal_ok"] or 0
        dev = int(dev_attempts.get(int(r["nfl_id"]), 0))
        ref_n = int(reference_n.get(r["position_group"] or "", 0))
        why = []
        if n == 0:
            tier = "official_only"
            why.append("no eligible tracked forty attempt")
        else:
            if n < 2:
                why.append("one attempt: no repeat")
            if ok < n:
                why.append(f"{n - ok} attempt(s) with a signal-quality flag")
            if r["any_start_in_motion"]:
                why.append("a recording starts in motion")
            if dev < min(n, 2):
                why.append(f"speed development measured on {dev} attempt(s)")
            if ref_n < min_reference:
                why.append(f"position reference of {ref_n} athletes")
            tier = "high" if not why else "medium" if ok >= 1 and dev >= 1 else "low"
        rows.append({**{k: r[k] for k in ("nfl_id", "position_group", "draft_year")}, "n_attempts": n, "n_development_attempts": dev,
                     "n_signal_ok": ok, "any_start_in_motion": r["any_start_in_motion"], "max_first_speed": r["max_first_speed"],
                     "any_end_censored": r["any_end_censored"], "max_position_residual": r["max_position_residual"], "repeat_available": n >= 2,
                     "fingerprint_inductive": r["fingerprint_inductive"], "fingerprint_self_auc": r["fingerprint_self_auc"],
                     "seed_overlap": r["seed_overlap"], "reference_n": ref_n, "evidence_tier": tier, "reasons": "; ".join(why)})
    return pl.DataFrame(rows, schema=CONFIDENCE.columns)


# ============================================================================== profiles
def profile_table(roster: pl.DataFrame, pct: pl.DataFrame, neighbors: pl.DataFrame, confidence: pl.DataFrame) -> pl.DataFrame:
    """One row per athlete. ``roster``: nfl_id, display_name, position_group, draft_year, official forty / 10-yard / weight / height."""
    def feat(fid: str, cols: dict[str, str]) -> pl.DataFrame:
        return pct.filter(pl.col("feature_id") == fid).select("nfl_id", *[pl.col(k).alias(v) for k, v in cols.items()])

    names = dict(zip(roster["nfl_id"].to_list(), roster["display_name"].to_list()))

    def near(kind: str, alias: str) -> pl.DataFrame:
        g = (neighbors.filter(pl.col("kind") == kind).sort("nfl_id", "rank").group_by("nfl_id", maintain_order=True).agg(pl.col("neighbor_id")))
        return pl.DataFrame({"nfl_id": g["nfl_id"], alias: [" · ".join(str(names.get(j) or j) for j in v) for v in g["neighbor_id"]]},
                            schema={"nfl_id": I, alias: S})

    d = (roster.rename({FORTY: "official_forty", TEN: "official_ten", "cmb_combine_weight": "weight", "cmb_combine_height": "height"})
         .select("nfl_id", "display_name", "position_group", "draft_year", "official_forty", "official_ten", "weight", "height")
         .join(feat(FORTY, {"percentile": "speed_capacity_pct", "pct_lower": "speed_capacity_lower", "pct_upper": "speed_capacity_upper",
                            "n_reference": "reference_n"}), on="nfl_id", how="left")
         .join(feat(TEN, {"percentile": "ten_pct"}), on="nfl_id", how="left")
         .join(feat(P95, {"percentile": "top_speed_pct", "pct_lower": "top_speed_lower", "pct_upper": "top_speed_upper"}), on="nfl_id", how="left")
         .join(feat(V4_V8, {"value": "speed_development_value", "percentile": "speed_development_pct", "pct_lower": "speed_development_lower",
                            "pct_upper": "speed_development_upper", "availability": "speed_development_availability"}), on="nfl_id", how="left")
         .join(near("position", "fingerprint_nearest"), on="nfl_id", how="left")
         .join(near("similar_official", "fingerprint_nearest_similar_official"), on="nfl_id", how="left")
         .join(confidence.select("nfl_id", "evidence_tier", pl.col("reasons").alias("evidence_reasons"), "fingerprint_self_auc"), on="nfl_id", how="left"))
    three = [pl.col(c) for c in ("speed_capacity_pct", "ten_pct", "top_speed_pct")]
    n_have = pl.sum_horizontal([c.is_not_null().cast(pl.Int64) for c in three])
    d = d.with_columns(pl.when(n_have >= 2).then(pl.max_horizontal(three) - pl.min_horizontal(three)).otherwise(None).alias("speed_agreement"),
                       pl.col("fingerprint_nearest").is_not_null().alias("fingerprint_available"),
                       pl.col("speed_development_availability").fill_null("unavailable"), pl.col("reference_n").fill_null(0),
                       pl.lit(NFL_PREDICTIVE_STATUS).alias("nfl_predictive_status"))
    out = d.select([pl.col(c).cast(t) for c, t in PROFILE.columns.items()]).sort("nfl_id")
    assert_no_nfl_columns(out, PROFILE.name)
    return out


# ============================================================================== Combine-side case selection
def select_pairs(profile: pl.DataFrame, match: dict[str, float], contrast: str | Callable[[int, int], float], k: int,
                 require: pl.Expr | None = None, small: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Pairs of athletes of the same position that are within ``match`` calipers (column → largest |difference|),
    optionally within ``small`` calipers too, ordered by the LARGEST ``contrast`` (a profile column, or a function of
    the two athlete ids). Every column involved is a Combine quantity (asserted); an athlete appears in one pair."""
    assert_no_nfl_columns(profile, "case selection")
    cols = list(match) + list(small or {}) + ([contrast] if isinstance(contrast, str) else [])
    p = (profile if require is None else profile.filter(require)).drop_nulls([c for c in cols if c in profile.columns])
    out: list[dict[str, Any]] = []
    for (pos,), g in sorted(p.partition_by(["position_group"], as_dict=True).items()):
        ids = g["nfl_id"].to_numpy()
        cand = []
        arr = {c: g[c].to_numpy().astype(float) for c in cols}
        for a in range(ids.size):
            for b in range(a + 1, ids.size):
                if any(abs(arr[c][a] - arr[c][b]) > lim for c, lim in {**match, **(small or {})}.items()):
                    continue
                size = abs(arr[contrast][a] - arr[contrast][b]) if isinstance(contrast, str) else contrast(int(ids[a]), int(ids[b]))
                if np.isfinite(size):
                    cand.append((float(size), a, b))
        for size, a, b in sorted(cand, reverse=True):
            out.append({"position_group": pos, "nfl_id_a": int(ids[a]), "nfl_id_b": int(ids[b]), "contrast_size": size,
                        "value_a": float(arr[contrast][a]) if isinstance(contrast, str) else None,
                        "value_b": float(arr[contrast][b]) if isinstance(contrast, str) else None})
    out.sort(key=lambda r: -r["contrast_size"])
    used: set[int] = set()
    kept = []
    for r in out:
        if r["nfl_id_a"] in used or r["nfl_id_b"] in used:
            continue
        kept.append(r)
        used |= {r["nfl_id_a"], r["nfl_id_b"]}
        if len(kept) >= k:
            break
    return kept


def case_studies(profile: pl.DataFrame, dist: Callable[[int, int], float], k: int = 2) -> pl.DataFrame:
    """The five pre-declared case types. Criteria use official times, tracked Combine measures, the fingerprint and the
    evidence tier — nothing from the NFL. A case is an illustration of what the profile shows, never a finding."""
    good = pl.col("evidence_tier").is_in(["high", "medium"])
    times = {"official_forty": 0.03, "official_ten": 0.03}
    plans = [
        ("similar_forty_different_development", "official forty and 10-yard split within 0.03 s; largest difference in speed-development percentile",
         times, "speed_development_pct", good & (pl.col("speed_development_availability") != "unavailable"), None),
        ("similar_top_speed_different_curve", "tracked top-speed percentile within 5 points; largest fingerprint distance",
         {"top_speed_pct": 5.0}, dist, good & pl.col("fingerprint_available"), None),
        ("repeatable_fingerprint_difference", "official times within 0.03 s, both repeats closer than 90 % of the position; largest fingerprint distance",
         times, dist, good & (pl.col("fingerprint_self_auc") >= 0.90), None),
        ("physical_agrees_fingerprint_differs", "official times within 0.03 s and speed-development percentile within 10 points; largest fingerprint distance",
         times, dist, good & pl.col("fingerprint_available"), {"speed_development_pct": 10.0}),
    ]
    names = dict(zip(profile["nfl_id"].to_list(), profile["display_name"].to_list()))
    rows = []
    for case, crit, match, contrast, require, small in plans:
        for r in select_pairs(profile, match, contrast, k, require, small):
            rows.append({"case": case, "criterion": crit, **r, "name_a": names.get(r["nfl_id_a"]), "name_b": names.get(r["nfl_id_b"]),
                         "matched_on": ", ".join({**match, **(small or {})}), "contrast": contrast if isinstance(contrast, str) else "fingerprint distance",
                         "fingerprint_distance": dist(r["nfl_id_a"], r["nfl_id_b"]), "selected_using_nfl": False})
    hi, lo = profile.filter(pl.col("evidence_tier") == "high"), profile.filter(pl.col("evidence_tier") == "low")
    seen: set[int] = set()
    conf = []
    for a in lo.drop_nulls(["official_forty"]).iter_rows(named=True):
        c = hi.filter((pl.col("position_group") == a["position_group"]) & ~pl.col("nfl_id").is_in(list(seen))).drop_nulls(["official_forty"])
        if not c.height:
            continue
        b = c.with_columns((pl.col("official_forty") - a["official_forty"]).abs().alias("_d")).sort("_d").row(0, named=True)
        if b["_d"] <= 0.03:
            conf.append((b["_d"], a, b))
            seen.add(b["nfl_id"])
    for gap, a, b in sorted(conf, key=lambda x: x[0])[:k]:
        rows.append({"case": "high_versus_low_recording_confidence", "criterion": "same position, official forty within 0.03 s, evidence tier low vs high (closest forty first)",
                     "position_group": a["position_group"], "nfl_id_a": a["nfl_id"], "nfl_id_b": b["nfl_id"], "name_a": a["display_name"],
                     "name_b": b["display_name"], "matched_on": "official_forty", "contrast": "evidence_tier", "value_a": None, "value_b": None,
                     "contrast_size": float(gap), "fingerprint_distance": dist(a["nfl_id"], b["nfl_id"]), "selected_using_nfl": False})
    out = pl.DataFrame(rows, schema=EXAMPLES.columns) if rows else EXAMPLES.empty()
    return out.with_columns(pl.col("fingerprint_distance").fill_nan(None))


# ============================================================================== quality against ranking stability
def quality_stability(attempt_values: pl.DataFrame, windows: pl.DataFrame, confidence: pl.DataFrame, pct: pl.DataFrame,
                      feature_ids: Sequence[str], n_boot: int, seed: int) -> pl.DataFrame:
    """Rank agreement of the first two attempts by evidence tier, within position. A tier whose attempts order athletes
    less consistently carries less trustworthy percentiles — a statement about the recording, not the athletes."""
    order = windows.select("seq_id", "attempt_order")
    tier = confidence.select("nfl_id", "position_group", "evidence_tier")
    rows = []
    for fid in feature_ids:
        spec = FEATURE[fid]
        a = (attempt_values.filter((pl.col("feature_id") == fid) & pl.col("value").is_not_null() & pl.col("value").is_finite())
             .join(order, on="seq_id", how="inner").sort("nfl_id", "attempt_order").group_by("nfl_id", maintain_order=True)
             .agg(pl.col("value").head(2)).filter(pl.col("value").list.len() == 2)
             .with_columns(pl.col("value").list.get(0).alias("v1"), pl.col("value").list.get(1).alias("v2")).join(tier, on="nfl_id", how="inner"))
        refs = {p: g["value"].drop_nulls().to_numpy() for (p,), g in
                pct.filter((pl.col("feature_id") == fid) & pl.col("in_reference")).partition_by(["position_group"], as_dict=True).items()}
        for t in ("ALL", *TIERS[:3]):
            g = a if t == "ALL" else a.filter(pl.col("evidence_tier") == t)
            if g.height < 8:
                continue
            x, y, pos = g["v1"].to_numpy(), g["v2"].to_numpy(), g["position_group"].to_numpy().astype(str)
            rho = _within_spearman(x, y, pos)
            rng = np.random.default_rng(seed_for(seed, "score", "quality", fid, t))
            reps = np.array([_within_spearman(x[i], y[i], pos[i]) for i in rng.integers(0, x.size, size=(n_boot, x.size))])
            fin = reps[np.isfinite(reps)]
            shift = [abs(orient(percentile_of(float(u), refs[p]), spec.higher_is_better) - orient(percentile_of(float(v), refs[p]), spec.higher_is_better))
                     for u, v, p in zip(x, y, pos) if p in refs and refs[p].size >= 5]
            rows.append({"feature_id": fid, "evidence_tier": t, "n_athletes": g.height, "rank_agreement": rho if np.isfinite(rho) else None,
                         "ci_lower": float(np.quantile(fin, 0.025)) if fin.size >= 20 else None,
                         "ci_upper": float(np.quantile(fin, 0.975)) if fin.size >= 20 else None,
                         "mean_abs_percentile_shift": float(np.mean(shift)) if shift else None})
    return pl.DataFrame(rows, schema=STABILITY_BY_QUALITY.columns) if rows else STABILITY_BY_QUALITY.empty()


def _within_spearman(x: np.ndarray, y: np.ndarray, pos: np.ndarray) -> float:
    a, b = np.empty(x.size), np.empty(x.size)
    for p in np.unique(pos):
        k = pos == p
        a[k] = np.argsort(np.argsort(x[k])) - (k.sum() - 1) / 2
        b[k] = np.argsort(np.argsort(y[k])) - (k.sum() - 1) / 2
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else NAN


# ============================================================================== manifest and gate
def reference_manifest(ref: ReferenceSpec, reference_tables: pl.DataFrame, pct: pl.DataFrame, quality_rule: str, inputs: dict[str, Any],
                       redundancy_policy: str) -> dict[str, Any]:
    """``combine_score_reference_manifest.json``: who the percentiles are relative to, and how they were computed."""
    counts = (pct.filter(pl.col("in_reference")).group_by("feature_id", "position_group").agg(pl.len().alias("n")).sort("feature_id", "position_group"))
    return {"artifact": "combine_score_reference_manifest", "nfl_predictive_status": NFL_PREDICTIVE_STATUS, "reference": ref.params(),
            "quality_rule": quality_rule, "missingness_policy": "a missing value gives no percentile and a stated reason; it is never imputed",
            "isolation": "reference members are scored leave-one-out; pct_fold_isolated scores every athlete against the other folds only",
            "redundancy_policy": redundancy_policy, "overall_score": "none — the four sections are never combined",
            "features": [{"feature_id": f.feature_id, "label": f.label, "section": f.section, "unit": f.unit, "role": f.role,
                          "orientation": "higher raw value = higher percentile" if f.higher_is_better else "LOWER raw value = higher percentile",
                          "meaning": f.meaning} for f in FEATURES],
            "reference_counts": counts.to_dicts(), "reference_distributions": reference_tables.to_dicts(), "inputs": inputs}


def gate_07d(pct: pl.DataFrame, profile: pl.DataFrame, confidence: pl.DataFrame, neighbors: pl.DataFrame, examples: pl.DataFrame,
             manifest: dict[str, Any], gate_contract: TableContract) -> pl.DataFrame:
    """Gate 07D — scorecard integrity."""
    rows = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"criterion": name, "passed": bool(ok), "detail": detail})

    add("scores_have_defined_meanings", len(manifest.get("features", [])) == len(FEATURES) and all(f.get("meaning") for f in manifest["features"]),
        "every percentile has a definition, a unit and an orientation in the manifest")
    add("reference_populations_known", bool(manifest.get("reference_counts")) and pct["n_reference"].null_count() == 0,
        f"{len(manifest.get('reference_counts', []))} feature × position reference sets recorded")
    shown = pct.filter(pl.col("percentile").is_not_null())
    add("position_specific_uncertainty_reported", shown.height > 0 and shown["pct_lower"].null_count() == 0 and shown["pct_upper"].null_count() == 0,
        f"{shown.height} percentile(s) shown, each with an interval")
    miss = pct.filter(pl.col("value").is_null())
    add("unsupported_characteristics_abstain", miss["percentile"].null_count() == miss.height
        and pct.filter((pl.col("availability") == "unavailable") & pl.col("percentile").is_not_null()).height == 0,
        f"{miss.height} missing value(s) and {pct.filter(pl.col('availability') == 'unavailable').height} unavailable row(s) carry no percentile")
    named = [c for c in (*profile.columns, *neighbors.columns) if any(w in c.lower() for w in ("explosive", "readiness", "talent", "upside", "overall"))]
    add("tcn_coordinates_not_named_as_traits", not named, "no latent coordinate, ability label or overall score in the profile" if not named else f"columns {named}")
    try:
        for t, name in ((pct, "percentiles"), (profile, "profiles"), (confidence, "confidence"), (neighbors, "neighbours"), (examples, "examples")):
            assert_no_nfl_columns(t, name)
        leak = bool(examples.height and examples["selected_using_nfl"].any())
        add("no_nfl_outcome_in_pre_draft_scoring", not leak, "no NFL-named column in any pre-draft table; no case selected with an NFL outcome")
    except LeakageError as e:
        add("no_nfl_outcome_in_pre_draft_scoring", False, str(e))
    return pl.DataFrame(rows, schema=gate_contract.columns)
