"""Milestone 09C — Stopwatch Twins: prospects the official tests cannot tell apart, and what tracking measures between them.

    twin_index        every pair of the same position whose official forty is within the caliper (pre-declared tolerances)
    difference_table  per pair × Combine curve feature: the difference with the standard error OF THE DIFFERENCE
    rank_twins        pairs ordered by the largest measured, distinguishable difference
    twin_summary      how often twins are distinguishable, per feature (read against the 5 % expected by chance)
    select_cases      real case studies of ten declared types, chosen by PRE-DRAFT rules only

Rules
    Matching uses official Combine results and anthropometrics only. No NFL quantity enters a twin table
    (``assert_no_nfl_columns``). One case type needs both athletes to have NFL exposure so that their observed NFL
    movement can be SHOWN afterwards: eligibility uses the exposure COUNT, never an outcome value, and is flagged.
    A difference is reported even when it is not distinguishable — with that verdict. Nothing is forced: a case type
    without a qualifying pair is reported as absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import polars as pl

from ..contracts import TableContract
from . import movement_demands as md
from .scoring.scorecards import assert_no_nfl_columns
from .scouting_intelligence import pairwise_difference

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
LATE = (md.BAND_COLS[2], "speed_30yd")
CASE_TYPES = (
    ("same_forty_different_curve", "Same forty, different acceleration curve"),
    ("same_top_speed_different_development", "Same top speed, different speed development"),
    ("same_first10_different_later", "Same first 10 yards, different later acceleration"),
    ("similar_tests_different_fingerprint", "Similar official tests, distant movement fingerprints"),
    ("high_confidence_difference", "High-confidence movement difference"),
    ("apparent_difference_overlapping_uncertainty", "Apparent difference inside the measurement uncertainty"),
    ("position_specific_comparison", "Position-specific comparison"),
    ("repeat_attempt_inconsistency", "Repeat-attempt inconsistency"),
    ("tracking_quality_limitation", "Tracking-quality limitation"),
    ("similar_movement_nfl_observed", "Similar movement, observed NFL movement shown"),
)


@dataclass(frozen=True)
class TwinConfig:
    """Tolerances fixed before any pair was looked at (proposed; the 07D / 08B.5 calipers where one exists)."""

    forty_tol: float = 0.03          # s — established: the 07D case caliper
    ten_tol: float = 0.03            # s — established: the 07D case caliper
    weight_tol: float = 15.0         # lb
    height_tol: float = 2.0          # in
    top_speed_tol: float = 0.05      # yd/s — established: 08B.5
    first10_tol: float = 0.02        # s
    min_features: int = 3            # curve features both athletes must have for a pair to be ranked
    min_nfl_windows: int = 50        # established: OutcomeConfig.primary_min_windows (exposure eligibility of one case type)
    cases_per_type: int = 2

    def params(self) -> dict[str, Any]:
        return asdict(self)


TWINS = TableContract(
    "stopwatch_twins", "Pairs of same-position prospects with near-identical official forty times, ranked by measured movement differences.",
    {"pair_id": S, "position_group": S, "nfl_id_a": I, "nfl_id_b": I, "name_a": S, "name_b": S, "draft_year_a": I, "draft_year_b": I, "forty_a": F,
     "forty_b": F, "d_forty": F, "ten_a": F, "ten_b": F, "d_ten": F, "ten_matched": B, "d_weight": F, "d_height": F, "body_matched": B, "tier_a": S,
     "tier_b": S, "fingerprint_distance": F, "n_features_compared": I, "n_distinguishable": I, "max_abs_z": F, "top_feature": S, "top_phase": S,
     "rank_in_position": I}, key=("pair_id",), kind="dataset", area="athlete pair",
    docs={"d_forty": "official forty of b − a [s]; the pair exists because |d_forty| ≤ TwinConfig.forty_tol",
          "ten_matched": "official 10-yard splits within the caliper (null: a split is missing)",
          "n_distinguishable": "curve features whose difference exceeds 1.96 standard errors of the difference",
          "max_abs_z": "largest |difference / its standard error| over the features both athletes have with an error estimate",
          "fingerprint_distance": "TCN-B distance (a mathematical similarity, no physical unit); null when not available"})
DIFFS = TableContract(
    "stopwatch_twin_differences", "Measured Combine curve differences of every stopwatch-twin pair, with the uncertainty of the difference.",
    {"pair_id": S, "nfl_id_a": I, "nfl_id_b": I, "feature": S, "label": S, "unit": S, "phase": S, "value_a": F, "value_b": F, "difference": F,
     "se_difference": F, "ci_lower": F, "ci_upper": F, "z": F, "verdict": S, "quicker": S, "evidence_status": S},
    key=("pair_id", "feature"), kind="dataset", area="athlete pair × feature",
    docs={"difference": "athlete b − athlete a, raw units", "quicker": "a · b — which athlete the raw difference favours on this feature ('' when not distinguishable)",
          "verdict": "distinguishable · not_distinguishable · insufficient_evidence · not_observed (a test of the DIFFERENCE)"})
SUMMARY = TableContract(
    "stopwatch_twin_summary", "How often stopwatch twins are measurably different, by position and feature.",
    {"position_group": S, "feature": S, "n_pairs": I, "n_tested": I, "n_distinguishable": I, "share_distinguishable": F, "median_abs_difference": F,
     "expected_by_chance": F}, key=("position_group", "feature"), area="position group × feature",
    docs={"expected_by_chance": "0.05: the share of 'distinguishable' verdicts expected if no pair truly differed",
          "n_pairs": "pairs are NOT independent (an athlete is in many pairs): a share, not a test"})
CASES = TableContract(
    "scouting_cases", "Real athlete case studies of the Combine Intelligence Lab, selected by pre-draft rules.",
    {"case_id": S, "case_type": S, "case_label": S, "rank": I, "pair_id": S, "position_group": S, "nfl_id_a": I, "nfl_id_b": I, "name_a": S, "name_b": S,
     "matched_on": S, "contrast": S, "contrast_size": F, "headline_feature": S, "selected_using_nfl_outcome": B, "selected_using_nfl_exposure": B},
    key=("case_id",), area="case",
    docs={"selected_using_nfl_outcome": "always False: no NFL outcome value enters the selection",
          "selected_using_nfl_exposure": "True only for the case type that SHOWS observed NFL movement: both athletes needed enough tracked NFL windows"})
CONTRACTS = (TWINS, DIFFS, SUMMARY, CASES)


def _gap(a: Any, b: Any) -> float | None:
    return None if a is None or b is None else float(b - a)


def twin_index(profile: pl.DataFrame, c: TwinConfig, dist: Callable[[int, int], float] | None = None) -> pl.DataFrame:
    """Every same-position pair within the forty caliper (unranked). ``profile``: the 07D pre-draft profile table."""
    assert_no_nfl_columns(profile, "twin matching input")
    rows = []
    for (pos,), g in sorted(profile.drop_nulls(["official_forty"]).partition_by(["position_group"], as_dict=True).items()):
        r = g.sort("official_forty", "nfl_id").to_dicts()
        for i, a in enumerate(r):
            for b in r[i + 1:]:
                if b["official_forty"] - a["official_forty"] > c.forty_tol + 1e-9:
                    break                                           # sorted by forty: no later athlete can match
                x, y = (a, b) if a["nfl_id"] < b["nfl_id"] else (b, a)
                d_ten, d_w, d_h = _gap(x["official_ten"], y["official_ten"]), _gap(x["weight"], y["weight"]), _gap(x["height"], y["height"])
                fd = dist(x["nfl_id"], y["nfl_id"]) if dist is not None and x.get("fingerprint_available") and y.get("fingerprint_available") else None
                rows.append({"pair_id": f"{x['nfl_id']}-{y['nfl_id']}", "position_group": pos, "nfl_id_a": x["nfl_id"], "nfl_id_b": y["nfl_id"],
                             "name_a": x["display_name"] or str(x["nfl_id"]), "name_b": y["display_name"] or str(y["nfl_id"]),
                             "draft_year_a": x["draft_year"], "draft_year_b": y["draft_year"], "forty_a": x["official_forty"], "forty_b": y["official_forty"],
                             "d_forty": y["official_forty"] - x["official_forty"], "ten_a": x["official_ten"], "ten_b": y["official_ten"], "d_ten": d_ten,
                             "ten_matched": None if d_ten is None else abs(d_ten) <= c.ten_tol + 1e-9, "d_weight": d_w, "d_height": d_h,
                             "body_matched": d_w is not None and d_h is not None and abs(d_w) <= c.weight_tol and abs(d_h) <= c.height_tol,
                             "tier_a": x.get("evidence_tier"), "tier_b": y.get("evidence_tier"),
                             "fingerprint_distance": fd if fd is not None and np.isfinite(fd) else None})
    base = {k: t for k, t in TWINS.columns.items() if k not in ("n_features_compared", "n_distinguishable", "max_abs_z", "top_feature", "top_phase", "rank_in_position")}
    return pl.DataFrame(rows, schema=base) if rows else pl.DataFrame(schema=base)


def difference_table(pairs: pl.DataFrame, prospect: pl.DataFrame, evidence: pl.DataFrame) -> pl.DataFrame:
    """Pair × feature differences. ``prospect``: ``movement_matching.PROSPECT`` · ``evidence``: ``movement_matching.EVIDENCE``."""
    assert_no_nfl_columns(prospect, "prospect curve features")
    val = {(r["nfl_id"], r["feature"]): r for r in prospect.iter_rows(named=True)}
    ev = {r["feature"]: r for r in evidence.iter_rows(named=True)}
    rows = []
    for p in pairs.iter_rows(named=True):
        for feat, e in ev.items():
            a, b = val.get((p["nfl_id_a"], feat), {}), val.get((p["nfl_id_b"], feat), {})
            d = pairwise_difference(a.get("value"), a.get("standard_error"), b.get("value"), b.get("standard_error"))
            quicker = ""
            if d["verdict"] == "distinguishable":
                quicker = "b" if (d["difference"] > 0) == e["higher_is_quicker"] else "a"
            rows.append({"pair_id": p["pair_id"], "nfl_id_a": p["nfl_id_a"], "nfl_id_b": p["nfl_id_b"], "feature": feat, "label": e["label"], "unit": e["unit"],
                         "phase": e["phase"], "value_a": a.get("value"), "value_b": b.get("value"), **d, "quicker": quicker, "evidence_status": e["evidence_status"]})
    return pl.DataFrame(rows, schema=DIFFS.columns) if rows else DIFFS.empty()


def rank_twins(pairs: pl.DataFrame, diffs: pl.DataFrame, c: TwinConfig) -> pl.DataFrame:
    """Add the measured-difference summary to every pair and rank within position (largest |z| first; ties by pair id).
    Features whose evidence status is ``unstable`` or ``insufficient_support`` never set the rank."""
    usable = diffs.filter(~pl.col("evidence_status").is_in(["unstable", "insufficient_support"]) & pl.col("z").is_not_null())
    top = (usable.with_columns(pl.col("z").abs().alias("_az")).sort("pair_id", "_az", descending=[False, True])
           .group_by("pair_id", maintain_order=True).agg(pl.len().cast(pl.Int64).alias("n_features_compared"),
                                                         (pl.col("verdict") == "distinguishable").sum().cast(pl.Int64).alias("n_distinguishable"),
                                                         pl.col("_az").first().alias("max_abs_z"), pl.col("feature").first().alias("top_feature"),
                                                         pl.col("phase").first().alias("top_phase")))
    out = pairs.join(top, on="pair_id", how="left").with_columns(pl.col("n_features_compared").fill_null(0), pl.col("n_distinguishable").fill_null(0))
    ranked = (out.filter(pl.col("n_features_compared") >= c.min_features).sort("position_group", "max_abs_z", "pair_id", descending=[False, True, False])
              .with_columns((pl.int_range(pl.len(), dtype=pl.Int64).over("position_group") + 1).alias("rank_in_position")).select("pair_id", "rank_in_position"))
    out = out.join(ranked, on="pair_id", how="left")
    out = out.select([pl.col(k).cast(t) for k, t in TWINS.columns.items()]).sort("position_group", "rank_in_position", "pair_id", nulls_last=True)
    assert_no_nfl_columns(out, TWINS.name)
    return out


def twin_summary(twins: pl.DataFrame, diffs: pl.DataFrame) -> pl.DataFrame:
    d = diffs.join(twins.select("pair_id", "position_group"), on="pair_id", how="inner")
    d = pl.concat([d, d.with_columns(pl.lit(md.ALL).alias("position_group"))], how="vertical")
    out = d.group_by("position_group", "feature").agg(
        pl.len().cast(pl.Int64).alias("n_pairs"), pl.col("z").is_not_null().sum().cast(pl.Int64).alias("n_tested"),
        (pl.col("verdict") == "distinguishable").sum().cast(pl.Int64).alias("n_distinguishable"), pl.col("difference").abs().median().alias("median_abs_difference"))
    out = out.with_columns(pl.when(pl.col("n_tested") > 0).then(pl.col("n_distinguishable") / pl.col("n_tested")).otherwise(None).alias("share_distinguishable"),
                           pl.lit(0.05).alias("expected_by_chance"))
    return out.select([pl.col(k).cast(t) for k, t in SUMMARY.columns.items()]).sort("position_group", "feature")


# ============================================================================== case studies
def _greedy(cand: list[tuple[float, dict[str, Any]]], k: int, taken: set[str] | None = None) -> list[tuple[float, dict[str, Any]]]:
    """Largest score first (ties by pair id); an athlete appears once per case type; ``taken`` pair ids are skipped."""
    cand = sorted(cand, key=lambda t: (-t[0], t[1]["pair_id"]))
    used: set[int] = set()
    out = []
    for s, p in cand:
        if p["nfl_id_a"] in used or p["nfl_id_b"] in used or (taken and p["pair_id"] in taken):
            continue
        used |= {p["nfl_id_a"], p["nfl_id_b"]}
        out.append((s, p))
        if len(out) >= k:
            break
    return out


def select_cases(twins: pl.DataFrame, diffs: pl.DataFrame, prospect: pl.DataFrame, c: TwinConfig, exposure: dict[int, int] | None = None) -> pl.DataFrame:
    """Case studies of the ten declared types from the ranked twin tables. Deterministic. ``exposure``: tracked NFL
    windows per athlete (a COUNT — the only NFL quantity read, used by one case type as an eligibility rule)."""
    for name, df in (("twins", twins), ("twin differences", diffs), ("prospect curve features", prospect)):
        assert_no_nfl_columns(df, name)
    D = {(r["pair_id"], r["feature"]): r for r in diffs.iter_rows(named=True)}
    P = {(r["nfl_id"], r["feature"]): r for r in prospect.iter_rows(named=True)}
    pairs = [p for p in twins.to_dicts() if p["rank_in_position"] is not None]
    bands, dev = list(md.BAND_COLS), "time_4_8"
    z_of = lambda p, f: abs(D[(p["pair_id"], f)]["z"]) if (p["pair_id"], f) in D and D[(p["pair_id"], f)]["z"] is not None else None
    verdict = lambda p, f: D.get((p["pair_id"], f), {}).get("verdict")
    usable = lambda p, f: D.get((p["pair_id"], f), {}).get("evidence_status") not in ("unstable", "insufficient_support", None)

    def best(p: dict, feats: list[str]) -> tuple[float, str] | None:
        got = [(z_of(p, f), f) for f in feats if verdict(p, f) == "distinguishable" and usable(p, f)]
        return max(got) if got else None

    def spread(i: int) -> float | None:
        r = P.get((i, dev), {})
        return None if (r.get("n_attempts") or 0) < 2 else float(r["attempt_max"] - r["attempt_min"])

    plans: dict[str, tuple[str, str, list[tuple[float, dict, str]]]] = {}
    add = lambda key, matched, contrast, items: plans.__setitem__(key, (matched, contrast, items))
    forty = f"official forty within {c.forty_tol:g} s"
    add("same_forty_different_curve", forty + f", 10-yard split within {c.ten_tol:g} s when recorded", "band acceleration, |z| of the difference",
        [(b[0], p, b[1]) for p in pairs if p["ten_matched"] is not False and (b := best(p, bands))])
    add("same_top_speed_different_development", forty + f", tracked top speed within {c.top_speed_tol:g} yd/s", "speed development 4 → 8 yd/s, |z| of the difference",
        [(b[0], p, b[1]) for p in pairs if (t := D.get((p["pair_id"], "top_speed"), {}).get("difference")) is not None and abs(t) <= c.top_speed_tol
         and (b := best(p, [dev]))])
    add("same_first10_different_later", forty + f", 10-yard split within {c.first10_tol:g} s, speed at 10 yd not distinguishable", "late-run feature, |z| of the difference",
        [(b[0], p, b[1]) for p in pairs if p["d_ten"] is not None and abs(p["d_ten"]) <= c.first10_tol and verdict(p, "speed_10yd") == "not_distinguishable"
         and (b := best(p, list(LATE)))])
    add("similar_tests_different_fingerprint", forty + f", 10-yard split within {c.ten_tol:g} s, weight within {c.weight_tol:g} lb, height within {c.height_tol:g} in",
        "TCN-B fingerprint distance (mathematical similarity)",
        [(p["fingerprint_distance"], p, "fingerprint") for p in pairs if p["body_matched"] and p["ten_matched"] and p["fingerprint_distance"] is not None])
    add("high_confidence_difference", forty + ", both recordings in the high evidence tier, ≥ 2 distinguishable features", "largest |z| of a difference",
        [(p["max_abs_z"], p, p["top_feature"]) for p in pairs if p["tier_a"] == "high" and p["tier_b"] == "high" and p["n_distinguishable"] >= 2])
    add("apparent_difference_overlapping_uncertainty", forty, "largest speed-development difference that is NOT distinguishable",
        [(abs(D[(p["pair_id"], dev)]["difference"]), p, dev) for p in pairs if verdict(p, dev) == "not_distinguishable"])
    add("repeat_attempt_inconsistency", forty + ", both with two attempts", "difference in the attempt-to-attempt range of speed development",
        [(abs(sa - sb), p, dev) for p in pairs if (sa := spread(p["nfl_id_a"])) is not None and (sb := spread(p["nfl_id_b"])) is not None])
    add("tracking_quality_limitation", forty + ", evidence tier high versus low", "closest official forty first",
        [(-abs(p["d_forty"]), p, "evidence_tier") for p in pairs if {p["tier_a"], p["tier_b"]} == {"high", "low"}])
    ex = exposure or {}
    add("similar_movement_nfl_observed", forty + f", no distinguishable curve feature, both with ≥ {c.min_nfl_windows} tracked NFL windows",
        "smallest largest-|z| (most similar measured movement)",
        [(-(p["max_abs_z"] or 0.0), p, "nfl_movement") for p in pairs if p["n_distinguishable"] == 0 and p["n_features_compared"] >= c.min_features
         and ex.get(p["nfl_id_a"], 0) >= c.min_nfl_windows and ex.get(p["nfl_id_b"], 0) >= c.min_nfl_windows])

    out, labels = [], dict(CASE_TYPES)

    def emit(case: str, picks: list[tuple[float, dict, str]], matched: str, contrast: str) -> None:
        for rank, (s, p, feat) in enumerate(picks, 1):
            out.append({"case_id": f"{case}-{rank}", "case_type": case, "case_label": labels[case], "rank": rank, "pair_id": p["pair_id"],
                        "position_group": p["position_group"], "nfl_id_a": p["nfl_id_a"], "nfl_id_b": p["nfl_id_b"], "name_a": p["name_a"], "name_b": p["name_b"],
                        "matched_on": matched, "contrast": contrast, "contrast_size": abs(float(s)), "headline_feature": feat,
                        "selected_using_nfl_outcome": False, "selected_using_nfl_exposure": case == "similar_movement_nfl_observed"})

    first: set[str] = set()
    for case, _ in CASE_TYPES:
        if case == "position_specific_comparison":
            continue
        matched, contrast, items = plans[case]
        feat_of = {p["pair_id"]: f for _, p, f in items}
        picks = [(s, p, feat_of[p["pair_id"]]) for s, p in _greedy([(s, p) for s, p, _ in items], c.cases_per_type)]
        emit(case, picks, matched, contrast)
        if case == "same_forty_different_curve":
            first = {p["pair_id"] for _, p, _ in picks}
    per_pos = []
    for pos in sorted({p["position_group"] for p in pairs}):
        g = _greedy([(p["max_abs_z"], p) for p in pairs if p["position_group"] == pos and p["n_distinguishable"] >= 1 and p["max_abs_z"] is not None], 1, first)
        per_pos += [(s, p, p["top_feature"]) for s, p in g]
    emit("position_specific_comparison", sorted(per_pos, key=lambda t: t[1]["position_group"]), forty + "; the top-ranked twin pair of each position not already a case",
         "largest |z| of a difference within the position")
    order = {k: i for i, (k, _) in enumerate(CASE_TYPES)}
    out.sort(key=lambda r: (order[r["case_type"]], r["rank"]))
    return pl.DataFrame(out, schema=CASES.columns) if out else CASES.empty()


def missing_case_types(cases: pl.DataFrame) -> list[str]:
    found = set(cases["case_type"].unique().to_list()) if cases.height else set()
    return [k for k, _ in CASE_TYPES if k not in found]


def conclusions(twins: pl.DataFrame, summary: pl.DataFrame, cases: pl.DataFrame) -> list[str]:
    out = [f"{twins.height:,} stopwatch-twin pairs among {pl.concat([twins['nfl_id_a'], twins['nfl_id_b']]).n_unique() if twins.height else 0} athletes; "
           f"{int((twins['n_distinguishable'] >= 1).sum()) if twins.height else 0:,} pairs have at least one distinguishable curve feature."]
    for r in summary.filter(pl.col("position_group") == md.ALL).iter_rows(named=True):
        share = "n/a" if r["share_distinguishable"] is None else f"{100 * r['share_distinguishable']:.0f} %"
        out.append(f"{r['feature']}: {share} of {r['n_tested']:,} tested pairs distinguishable (5 % expected by chance; pairs are not independent).")
    out.append(f"{cases.height} case studies of {cases['case_type'].n_unique() if cases.height else 0} types; absent types: {', '.join(missing_case_types(cases)) or 'none'}.")
    return out


__all__ = ["CASES", "CASE_TYPES", "CONTRACTS", "DIFFS", "SUMMARY", "TWINS", "TwinConfig", "conclusions", "difference_table", "missing_case_types",
           "rank_twins", "select_cases", "twin_index", "twin_summary"]
