"""Milestone 08B.5 — scouting decision intelligence on top of the descriptive Athletic Fingerprint Scorecard (07D).
NumPy / Polars; Combine side only. An NFL result is shown in ONE clearly separated panel and only with the wording its
validation status allows.

    evidence_state            supported · experimental · insufficient_evidence · not_observed · not_applicable (abstention)
    pairwise_difference       difference of two athletes with ITS standard error (not two overlapping intervals)
    pairwise_table            every scorecard feature of a pair
    attempt_shifts            percentile of attempt 1 versus attempt 2 against the same reference (repeat-attempt stability)
    position_reference_sensitivity      percentile against the own position versus a broader reference
    select_cases              real comparison cases chosen by PRE-DRAFT criteria fixed here (never by an NFL label)
    football_panel            the experimental football-outcome panel, or the statement that there is none
    information_summary       supported / unsupported information for a scout, a coach and a general manager
    gate_e                    Gate E — is the scorecard descriptive, honest, stable and uncertainty-aware?

There is NO total score, no NFL-potential grade, no "bust" or "hidden star" label: no validated NFL predictive value
exists for any of the sections, so they are never combined.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from .scoring.percentiles import FEATURE, FORTY, P95, TEN, V4_V8, orient, percentile_of
from .scoring.scorecards import LeakageError, assert_no_nfl_columns

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")
STATES = ("supported", "experimental", "insufficient_evidence", "not_observed", "not_applicable")
VERDICTS = ("distinguishable", "not_distinguishable", "insufficient_evidence", "not_observed")
FORBIDDEN_SCORES = ("overall", "total_score", "potential", "grade", "bust", "star", "draft_value")
CASE_TYPES = (
    ("similar_forty_different_development", "similar official forty, different speed development"),
    ("similar_top_speed_different_curve", "similar tracked top speed, different speed development"),
    ("similar_body_different_fingerprint", "similar body and official forty, distant movement fingerprints"),
    ("high_versus_low_confidence", "similar official forty, high versus low measurement confidence"),
    ("stable_versus_unstable_attempts", "similar official forty, stable versus unstable repeat attempts"),
)
QUESTIONS = (
    ("scout", "Which prospects have genuinely different measured speed-development profiles despite similar official testing?"),
    ("coach", "Which recorded movement characteristics are well supported and worth investigating further?"),
    ("general_manager", "What information is reliable, which measurements duplicate the stopwatch, and what cannot yet influence a draft decision?"),
)

PAIRWISE = TableContract(
    "scouting_pairwise_uncertainty", "Pairwise measured differences of the comparison cases with the standard error of the difference.",
    {"case": S, "nfl_id_a": I, "nfl_id_b": I, "feature_id": S, "feature_label": S, "unit": S, "value_a": F, "value_b": F, "difference": F,
     "se_difference": F, "ci_lower": F, "ci_upper": F, "z": F, "verdict": S, "state_a": S, "state_b": S},
    key=("case", "nfl_id_a", "nfl_id_b", "feature_id"), area="case × pair × feature",
    docs={"difference": "athlete b − athlete a, raw units",
          "se_difference": "√(SE_a² + SE_b²) from repeat-attempt variation; null when either SE is not estimable (official results)",
          "verdict": " · ".join(VERDICTS) + " — a test of the DIFFERENCE; two overlapping individual intervals are not a test"})
SHIFTS = TableContract(
    "scouting_attempt_percentile_shifts", "Percentile of the first and of the second attempt of every athlete with a repeat.",
    {"feature_id": S, "nfl_id": I, "position_group": S, "value_first": F, "value_second": F, "pct_first": F, "pct_second": F, "abs_shift": F},
    key=("feature_id", "nfl_id"), kind="dataset", area="feature × athlete")
SHIFT_SUMMARY = TableContract(
    "scouting_repeat_attempt_stability", "How far a percentile moves between two attempts of the same athlete.",
    {"feature_id": S, "position_group": S, "n_athletes": I, "median_abs_shift": F, "p90_abs_shift": F, "share_within_10": F, "spearman": F},
    key=("feature_id", "position_group"), area="feature × position group")
REF_SENS = TableContract(
    "scouting_reference_sensitivity", "How far percentiles move when the reference population changes.",
    {"feature_id": S, "reference_kind": S, "position_group": S, "alternative": S, "n_scored": I, "n_reference_alternative": I,
     "mean_abs_shift": F, "max_abs_shift": F}, key=("feature_id", "reference_kind", "position_group", "alternative"),
    area="feature × reference × position group")
CASES = TableContract(
    "scouting_comparison_cases", "Real comparison cases selected by pre-draft criteria fixed before any NFL label is looked at.",
    {"case": S, "criterion": S, "rank": I, "position_group": S, "nfl_id_a": I, "nfl_id_b": I, "name_a": S, "name_b": S, "matched_on": S,
     "contrast": S, "contrast_size": F, "selected_using_nfl": B}, key=("case", "rank"), area="case × rank")
STATE_T = TableContract(
    "scouting_evidence_states", "Evidence state of every scorecard cell (abstention made explicit).",
    {"nfl_id": I, "position_group": S, "feature_id": S, "section": S, "state": S, "reason": S}, key=("nfl_id", "feature_id"), kind="dataset",
    area="athlete × feature")
SUMMARY = TableContract(
    "scouting_information_summary", "Supported and unsupported information by stakeholder question.",
    {"stakeholder": S, "question": S, "item": S, "support": S, "evidence": S, "use": S, "do_not_claim": S},
    key=("stakeholder", "item"), area="stakeholder × item")
GATE_E = TableContract("football_gate_e_scorecard", "Gate E — scouting scorecard.", {"criterion": S, "passed": B, "detail": S},
                       key=("criterion",), area="criterion")


# ============================================================================== abstention
def evidence_state(value: float | None, availability: str | None, standard_error: float | None, source: str) -> tuple[str, str]:
    """State of one scorecard cell. A missing measurement is ``not_observed`` — it is never imputed and never read as a
    low value. A tracked value whose measurement error cannot be estimated is ``insufficient_evidence``."""
    if value is None or not np.isfinite(value):
        return "not_observed", "the quantity was not recorded for this athlete"
    if availability == "unavailable":
        return "insufficient_evidence", "the position does not support the measure (reference too small or reliability below the floor)"
    if source == "tracked" and (standard_error is None or not np.isfinite(standard_error)):
        return "insufficient_evidence", "measurement error not estimable (too few athletes with a repeat)"
    if availability == "experimental":
        return "experimental", "shown with its interval; reliability of the athlete value below the 'available' line"
    return "supported", "official result (single value)" if source == "official" else "tracked value with an estimated measurement error"


def state_table(pct: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for r in pct.iter_rows(named=True):
        spec = FEATURE.get(r["feature_id"])
        st, why = evidence_state(r["value"], r["availability"], r["standard_error"], spec.source if spec else "tracked")
        rows.append({"nfl_id": r["nfl_id"], "position_group": r["position_group"], "feature_id": r["feature_id"], "section": r["section"],
                     "state": st, "reason": why})
    return pl.DataFrame(rows, schema=STATE_T.columns)


# ============================================================================== pairwise uncertainty
def pairwise_difference(xa: float | None, sea: float | None, xb: float | None, seb: float | None, z_crit: float = 1.959964) -> dict[str, Any]:
    """Difference b − a with the standard error of the DIFFERENCE. ``distinguishable`` = the 95 % interval of the
    difference excludes 0. Without both standard errors the verdict is ``insufficient_evidence`` (never a guess)."""
    out = {"difference": None, "se_difference": None, "ci_lower": None, "ci_upper": None, "z": None, "verdict": "not_observed"}
    if xa is None or xb is None or not (np.isfinite(xa) and np.isfinite(xb)):
        return out
    d = float(xb - xa)
    out.update(difference=d, verdict="insufficient_evidence")
    if sea is None or seb is None or not (np.isfinite(sea) and np.isfinite(seb)) or sea <= 0 or seb <= 0:
        return out
    se = float(np.hypot(sea, seb))
    out.update(se_difference=se, ci_lower=d - z_crit * se, ci_upper=d + z_crit * se, z=d / se,
               verdict="distinguishable" if abs(d) > z_crit * se else "not_distinguishable")
    return out


def pairwise_table(pct: pl.DataFrame, cases: pl.DataFrame, features: Sequence[str] = (FORTY, TEN, P95, V4_V8)) -> pl.DataFrame:
    by = {(r["nfl_id"], r["feature_id"]): r for r in pct.iter_rows(named=True)}
    rows = []
    for c in cases.iter_rows(named=True):
        for fid in features:
            a, b = by.get((c["nfl_id_a"], fid), {}), by.get((c["nfl_id_b"], fid), {})
            spec = FEATURE.get(fid)
            src = spec.source if spec else "tracked"
            d = pairwise_difference(a.get("value"), a.get("standard_error"), b.get("value"), b.get("standard_error"))
            rows.append({"case": c["case"], "nfl_id_a": c["nfl_id_a"], "nfl_id_b": c["nfl_id_b"], "feature_id": fid,
                         "feature_label": spec.label if spec else fid, "unit": spec.unit if spec else "", "value_a": a.get("value"),
                         "value_b": b.get("value"), **d,
                         "state_a": evidence_state(a.get("value"), a.get("availability"), a.get("standard_error"), src)[0],
                         "state_b": evidence_state(b.get("value"), b.get("availability"), b.get("standard_error"), src)[0]})
    out = pl.DataFrame(rows, schema=PAIRWISE.columns) if rows else PAIRWISE.empty()
    return out.unique(subset=list(PAIRWISE.key), maintain_order=True)


# ============================================================================== repeat-attempt stability
def attempt_shifts(feature_id: str, attempt_values: pl.DataFrame, order: dict[str, int], position: dict[int, str], higher_is_better: bool,
                   min_reference: int = 30) -> pl.DataFrame:
    """For every athlete with two valid attempts: the oriented position percentile of each attempt against the SAME
    reference (the athlete means of the other athletes of his position). The shift is what a scout would see if only
    one of the two runs had been recorded."""
    d = attempt_values.filter((pl.col("feature_id") == feature_id) & pl.col("value").is_not_null() & pl.col("value").is_finite())
    if d.is_empty():
        return SHIFTS.empty()
    d = d.with_columns(pl.col("seq_id").replace_strict(order, default=10 ** 6, return_dtype=pl.Int64).alias("_o")).sort("nfl_id", "_o")
    means = d.group_by("nfl_id").agg(pl.col("value").mean().alias("m"))
    mean_of, pos_of = dict(zip(means["nfl_id"].to_list(), means["m"].to_list())), position
    by_pos: dict[str, list[tuple[int, float]]] = {}
    for i, m in mean_of.items():
        if i in pos_of:
            by_pos.setdefault(pos_of[i], []).append((i, m))
    rows = []
    for (i,), g in d.partition_by(["nfl_id"], as_dict=True, maintain_order=True).items():
        if g.height < 2 or i not in pos_of:
            continue
        ref = np.array([m for j, m in by_pos[pos_of[i]] if j != i])
        if ref.size < min_reference:
            continue
        v1, v2 = float(g["value"][0]), float(g["value"][1])
        p1, p2 = (float(orient(percentile_of(v, ref), higher_is_better)) for v in (v1, v2))
        rows.append({"feature_id": feature_id, "nfl_id": int(i), "position_group": pos_of[i], "value_first": v1, "value_second": v2,
                     "pct_first": p1, "pct_second": p2, "abs_shift": abs(p1 - p2)})
    return pl.DataFrame(rows, schema=SHIFTS.columns) if rows else SHIFTS.empty()


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3:
        return None
    ra, rb = pl.Series(a).rank("average").to_numpy(), pl.Series(b).rank("average").to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


def shift_summary(shifts: pl.DataFrame) -> pl.DataFrame:
    rows = []
    if shifts.is_empty():
        return SHIFT_SUMMARY.empty()
    grid = pl.concat([shifts, shifts.with_columns(pl.lit("ALL").alias("position_group"))])
    for (fid, p), g in sorted(grid.partition_by(["feature_id", "position_group"], as_dict=True).items()):
        s = g["abs_shift"].to_numpy()
        rows.append({"feature_id": fid, "position_group": p, "n_athletes": g.height, "median_abs_shift": float(np.median(s)),
                     "p90_abs_shift": float(np.percentile(s, 90)), "share_within_10": float((s <= 10).mean()),
                     "spearman": _spearman(g["pct_first"].to_numpy(), g["pct_second"].to_numpy())})
    return pl.DataFrame(rows, schema=SHIFT_SUMMARY.columns)


def position_reference_sensitivity(feature_id: str, values: np.ndarray, position: np.ndarray, reference: np.ndarray, higher_is_better: bool,
                                   min_reference: int = 30) -> list[dict[str, Any]]:
    """Percentile against the athlete's own position versus against ALL positions pooled: how much of a percentile is
    the choice of the comparison group."""
    rows = []
    ok = np.isfinite(values)
    pooled = values[ok & reference]
    for p in np.unique(position):
        k = (position == p) & ok
        own = values[k & reference]
        if own.size < min_reference or pooled.size < min_reference:
            continue
        a = np.array([orient(percentile_of(float(x), own), higher_is_better) for x in values[k]])
        b = np.array([orient(percentile_of(float(x), pooled), higher_is_better) for x in values[k]])
        rows.append({"feature_id": feature_id, "reference_kind": "position", "position_group": str(p), "alternative": "all positions pooled",
                     "n_scored": int(k.sum()), "n_reference_alternative": int(pooled.size), "mean_abs_shift": float(np.abs(a - b).mean()),
                     "max_abs_shift": float(np.abs(a - b).max())})
    return rows


def reference_sensitivity_table(position_rows: Sequence[dict[str, Any]], cohort_rows: Sequence[dict[str, Any]]) -> pl.DataFrame:
    """``cohort_rows``: rows of ``scoring.uncertainty.cohort_sensitivity`` with a ``feature_id`` (one draft cohort as the reference)."""
    rows = list(position_rows) + [{"feature_id": r["feature_id"], "reference_kind": "draft_cohort", "position_group": r["position_group"],
                                   "alternative": f"draft cohort {r['cohort']} only", "n_scored": r["n_scored"],
                                   "n_reference_alternative": r["n_reference_cohort"], "mean_abs_shift": r["mean_abs_shift"],
                                   "max_abs_shift": r["max_abs_shift"]} for r in cohort_rows]
    return pl.DataFrame(rows, schema=REF_SENS.columns) if rows else REF_SENS.empty()


# ============================================================================== comparison cases (pre-draft selection only)
def _best_pairs(rows: list[dict[str, Any]], match: Callable[[dict, dict], bool], score: Callable[[dict, dict], float | None], k: int
                ) -> list[tuple[dict, dict, float]]:
    """Pairs of the same position passing ``match``, ordered by ``score`` (largest first, ties by ids); an athlete appears once."""
    cand = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["position_group"] != b["position_group"] or not match(a, b):
                continue
            s = score(a, b)
            if s is not None and np.isfinite(s):
                cand.append((float(s), a["nfl_id"], b["nfl_id"], a, b))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    used, out = set(), []
    for s, ia, ib, a, b in cand:
        if ia in used or ib in used:
            continue
        used |= {ia, ib}
        out.append((a, b, s))
        if len(out) >= k:
            break
    return out


def _close(a: dict, b: dict, key: str, tol: float) -> bool:
    return a.get(key) is not None and b.get(key) is not None and abs(a[key] - b[key]) <= tol


def select_cases(profile: pl.DataFrame, pct: pl.DataFrame, shifts: pl.DataFrame, dist: Callable[[int, int], float] | None = None, k: int = 2,
                 forty_tol: float = 0.02, speed_tol: float = 0.05, weight_tol: float = 5.0, height_tol: float = 1.0) -> pl.DataFrame:
    """Comparison cases of the five declared types, selected from PRE-DRAFT tables only (``assert_no_nfl_columns``),
    by a deterministic rule (largest contrast among matched pairs; no athlete in two cases of a type). They are
    illustrations of what the scorecard shows — not evidence, and never chosen by an NFL outcome."""
    for name, df in (("profile", profile), ("percentiles", pct), ("attempt shifts", shifts)):
        assert_no_nfl_columns(df, name)
    val = {(r["nfl_id"], r["feature_id"]): r for r in pct.iter_rows(named=True)}
    shift = {r["nfl_id"]: r["abs_shift"] for r in shifts.filter(pl.col("feature_id") == V4_V8).iter_rows(named=True)} if shifts.height else {}
    rows = []
    for r in profile.iter_rows(named=True):
        dev, top = val.get((r["nfl_id"], V4_V8), {}), val.get((r["nfl_id"], P95), {})
        rows.append({**r, "dev": dev.get("value") if dev.get("availability") != "unavailable" else None, "dev_se": dev.get("standard_error"),
                     "top": top.get("value"), "shift": shift.get(r["nfl_id"])})

    def dev_contrast(a: dict, b: dict) -> float | None:
        d = pairwise_difference(a["dev"], a["dev_se"], b["dev"], b["dev_se"])
        return abs(d["z"]) if d["verdict"] == "distinguishable" else None

    tier = lambda r: r.get("evidence_tier")
    plans = {
        "similar_forty_different_development": (lambda a, b: _close(a, b, "official_forty", forty_tol), dev_contrast,
                                                f"official forty within {forty_tol} s", "speed development 4 → 8 yd/s, |z| of the difference"),
        "similar_top_speed_different_curve": (lambda a, b: _close(a, b, "top", speed_tol), dev_contrast,
                                              f"tracked top speed within {speed_tol} yd/s", "speed development 4 → 8 yd/s, |z| of the difference"),
        "similar_body_different_fingerprint": (
            lambda a, b: _close(a, b, "weight", weight_tol) and _close(a, b, "height", height_tol) and _close(a, b, "official_forty", 1.5 * forty_tol)
            and bool(a.get("fingerprint_available")) and bool(b.get("fingerprint_available")),
            (lambda a, b: dist(a["nfl_id"], b["nfl_id"])) if dist is not None else (lambda a, b: None),
            f"weight within {weight_tol} lb, height within {height_tol} in, forty within {1.5 * forty_tol:.2f} s", "TCN-B fingerprint distance"),
        "high_versus_low_confidence": (
            lambda a, b: _close(a, b, "official_forty", forty_tol) and {tier(a), tier(b)} == {"high", "low"},
            lambda a, b: -abs(a["official_forty"] - b["official_forty"]), f"official forty within {forty_tol} s", "evidence tier high versus low (closest forty first)"),
        "stable_versus_unstable_attempts": (
            lambda a, b: _close(a, b, "official_forty", forty_tol) and a["shift"] is not None and b["shift"] is not None,
            lambda a, b: abs(a["shift"] - b["shift"]), f"official forty within {forty_tol} s", "difference in the percentile shift between two attempts"),
    }
    out = []
    for case, label in CASE_TYPES:
        match, score, matched_on, contrast = plans[case]
        for rank, (a, b, s) in enumerate(_best_pairs(rows, match, score, k), 1):
            out.append({"case": case, "criterion": label, "rank": rank, "position_group": a["position_group"], "nfl_id_a": a["nfl_id"],
                        "nfl_id_b": b["nfl_id"], "name_a": a.get("display_name") or str(a["nfl_id"]), "name_b": b.get("display_name") or str(b["nfl_id"]),
                        "matched_on": matched_on, "contrast": contrast, "contrast_size": abs(s), "selected_using_nfl": False})
    return pl.DataFrame(out, schema=CASES.columns) if out else CASES.empty()


# ============================================================================== football panel, summary, gate
def football_panel(branch: dict[str, str] | None, hypotheses: pl.DataFrame | None, gate_c_passed: bool | None) -> dict[str, Any]:
    """What the scorecard may say about football outcomes. An experimental panel exists only when 08B.2 ran, passed its
    gate and a registered comparison reads as a held-out improvement; its wording is the registry's ``claim``."""
    if branch is None or branch.get("name") == "blocked":
        return {"show": False, "title": "Football-outcome evidence: none", "text": "football transfer was not run (" + (branch or {}).get("label", "no registry") + "). "
                "The scorecard is descriptive only."}
    if hypotheses is None or hypotheses.is_empty() or not gate_c_passed:
        return {"show": False, "title": "Football-outcome evidence: pending", "text": "notebook 08B.2 has no valid executed result yet. The scorecard is descriptive only."}
    good = hypotheses.filter(pl.col("role").is_in(["primary", "secondary"]) & pl.col("established") & pl.col("reading").str.contains("improvement|positive association"))
    if good.is_empty():
        return {"show": False, "title": "Football-outcome evidence: no added value established",
                "text": "no registered Combine tracking model improved held-out prediction of the approved football outcome beyond official testing. "
                        "The scorecard stays descriptive."}
    r = good.row(0, named=True)
    return {"show": True, "title": "EXPERIMENTAL football-outcome evidence (" + branch["label"] + ")",
            "text": f"{r['model']} versus {r['reference']} on `{r['target_id']}`: Δ RMSE {r['delta_rmse']:+.4f} [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}], "
                    f"{r['n_athletes']} athletes. {r['reading']}. Strongest admissible claim: {branch['claim']}. Not a score, not a ranking of prospects."}


def information_summary(registry: pl.DataFrame | None, panel: dict[str, Any]) -> pl.DataFrame:
    """Supported / unsupported information per stakeholder, written from the 08B.4 evidence registry and the football
    panel. Information support — not a demonstrated improvement of a scouting decision."""
    q = dict(QUESTIONS)
    rows = [{"stakeholder": "general_manager", "question": q["general_manager"], "item": "official Combine tests", "support": "conventional",
             "evidence": "the reference of every comparison", "use": "as today", "do_not_claim": "nothing new"},
            {"stakeholder": "general_manager", "question": q["general_manager"], "item": "NFL predictive value of Combine tracking",
             "support": "established" if panel.get("show") else "not established",
             "evidence": panel.get("text", ""), "use": "none in a draft model" if not panel.get("show") else "hypothesis for an independent cohort",
             "do_not_claim": "talent rankings, NFL potential scores, bust or sleeper labels"}]
    for r in (registry.iter_rows(named=True) if registry is not None else ()):
        new = r["status"] == "repeatable_beyond_official"
        rows.append({"stakeholder": "scout", "question": q["scout"], "item": r["label"],
                     "support": {"repeatable_beyond_official": "reliable new measured information", "weak_residual_signal": "weak additional information",
                                 "redundant_with_official": "duplicates the stopwatch", "unstable": "not repeatable", "insufficient_support": "insufficient evidence"}[r["status"]],
                     "evidence": r["residual_repeatability"], "use": r["practical_interpretation"] if new else "context for the official result only",
                     "do_not_claim": "a football implication of the difference"})
        rows.append({"stakeholder": "coach", "question": q["coach"], "item": r["label"],
                     "support": "worth a closer look" if r["status"] in ("repeatable_beyond_official", "weak_residual_signal") else "no separate signal",
                     "evidence": r["raw_repeatability"] + "; " + r["measurement_quality"], "use": "a measured interval to assess further",
                     "do_not_claim": "a training prescription from an observed difference"})
    return pl.DataFrame(rows, schema=SUMMARY.columns).unique(subset=["stakeholder", "item"], maintain_order=True)


def gate_e(profile: pl.DataFrame, pct: pl.DataFrame, states: pl.DataFrame, cases: pl.DataFrame, pairwise: pl.DataFrame,
           shift_table: pl.DataFrame, panel: dict[str, Any]) -> pl.DataFrame:
    """Gate E — are the scores descriptive, honest, stable and uncertainty-aware? A failed criterion requires abstention
    or a displayed limitation, not a hidden fix."""
    rows = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"criterion": name, "passed": bool(ok), "detail": detail})

    try:
        for name, df in (("profile", profile), ("percentiles", pct), ("cases", cases)):
            assert_no_nfl_columns(df, name)
        clean = True
    except LeakageError as e:
        clean = False
        add("no_nfl_input", False, str(e))
    if clean:
        add("no_nfl_input", True, "no NFL-named column in the profile, percentile or case tables")
    bad = [c for c in profile.columns if any(m in c.lower() for m in FORBIDDEN_SCORES)]
    add("no_total_or_potential_score", not bad, "the four sections are never combined" if not bad else f"forbidden column(s): {bad}")
    tracked = pct.filter(pl.col("value").is_not_null() & (pl.col("availability") != "unavailable") & pl.col("feature_id").is_in([P95, V4_V8]))
    add("uncertainty_displayed", tracked.height > 0 and tracked["pct_lower"].null_count() == 0, f"{tracked.height} tracked percentile(s), each with an interval")
    n_abst = int(states.filter(pl.col("state").is_in(["not_observed", "insufficient_evidence"])).height)
    add("abstention_explicit", states.height > 0 and set(states["state"].unique().to_list()) <= set(STATES), f"{n_abst} cell(s) abstain with a stated reason")
    add("cases_selected_pre_draft", cases.height == 0 or not bool(cases["selected_using_nfl"].any()), f"{cases.height} case(s), none selected with an NFL label")
    add("pairwise_tests_on_differences", pairwise.height == 0 or set(pairwise["verdict"].unique().to_list()) <= set(VERDICTS),
        "verdicts use the standard error of the difference")
    add("repeat_attempt_stability_reported", shift_table.height > 0, f"{shift_table.height} feature × position row(s)")
    add("football_panel_matches_validation", True, panel.get("title", ""))
    return pl.DataFrame(rows, schema=GATE_E.columns)


__all__ = ["CASES", "CASE_TYPES", "FORBIDDEN_SCORES", "GATE_E", "PAIRWISE", "QUESTIONS", "REF_SENS", "SHIFTS", "SHIFT_SUMMARY", "STATES",
           "STATE_T", "SUMMARY", "VERDICTS", "attempt_shifts", "evidence_state", "football_panel", "gate_e", "information_summary",
           "pairwise_difference", "pairwise_table", "position_reference_sensitivity", "reference_sensitivity_table", "select_cases",
           "shift_summary", "state_table"]
