"""Synthetic validation of Milestone 10 (notebooks 10A–10E and the application logic), callable from a notebook cell.
Standard library + Polars; no project data, no warehouse, no Streamlit. Every fixture is built here with a known answer.
Tolerances were set by reasoning — the author executed nothing — and each assertion states what it compares.

    from nflbdb.research.advanced_movement import milestone10_validation as v10
    report = v10.run_all(); display(report); v10.assert_all_passed(report)

Groups: ``CHECKS_KIT`` (phases, replay, evidence labels) · ``CHECKS_GAP`` (10C) · ``CHECKS_SIMULATOR`` (10D) ·
``CHECKS_STORIES`` (10B) · ``CHECKS_SUBMISSION`` (10E). A failing check is reported with its location; never weakened to pass.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import polars as pl

from . import decision_simulator as ds
from . import movement_gap as mg
from . import product_kit as pk
from . import scouting_interpretation as sin
from . import signature_stories as ss
from . import submission_builder as sb
from .validation import _raises, assert_all_passed, failure_location


# ============================================================================== fixtures
def _diff(f: str, va: float | None, vb: float | None, z: float | None, verdict: str, status: str = "repeatable_beyond_official", phase: str = "middle",
          quicker: str = "b") -> dict[str, Any]:
    d = None if va is None or vb is None else vb - va
    se = None if z in (None, 0) or d is None else abs(d / z)
    return {"pair_id": "P1", "feature": f, "label": f, "unit": "yd/s", "phase": phase, "value_a": va, "value_b": vb, "difference": d, "se_difference": se,
            "ci_lower": None if se is None else d - 1.96 * se, "ci_upper": None if se is None else d + 1.96 * se, "z": z, "verdict": verdict, "quicker": quicker,
            "evidence_status": status}


def _pair() -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    twin = {"pair_id": "P1", "position_group": "WR", "nfl_id_a": 1, "nfl_id_b": 2, "name_a": "Alpha One", "name_b": "Beta Two", "forty_a": 4.40, "forty_b": 4.40,
            "d_forty": 0.0, "ten_a": 1.52, "ten_b": 1.51, "d_ten": -0.01, "ten_matched": True, "tier_a": "high", "tier_b": "low", "n_distinguishable": 3}
    diffs = [_diff("acc_4_6", 6.0, 7.5, 4.0, "distinguishable", phase="middle"), _diff("top_speed", 10.9, 11.7, 5.0, "distinguishable", phase="late"),
             _diff("speed_20yd", 10.2, 10.6, 3.0, "distinguishable", "mostly_stopwatch", phase="middle"),
             _diff("acc_6_8", 4.0, 6.5, 6.0, "distinguishable", "unstable", phase="late"), _diff("speed_30yd", 10.8, 10.9, 0.5, "not_distinguishable", phase="late")]
    profiles = {1: {"nfl_id": 1, "display_name": "Alpha One", "position_group": "WR", "draft_year": 2024, "official_forty": 4.40, "official_ten": 1.52, "weight": 190,
                    "evidence_tier": "high", "n_tracked_attempts": 2},
                2: {"nfl_id": 2, "display_name": "Beta Two", "position_group": "WR", "draft_year": 2024, "official_forty": 4.40, "official_ten": None, "weight": 180,
                    "evidence_tier": "low", "n_tracked_attempts": 2}}
    return twin, diffs, profiles


def _registry() -> list[dict[str, Any]]:
    return [{"demand_id": "accel_low", "label": "acceleration from low speed", "combine_analogue": "acc_2_4, acc_4_6", "comparability": "same bands"},
            {"demand_id": "accel_mid", "label": "acceleration from intermediate speed", "combine_analogue": "acc_4_6, acc_6_8", "comparability": "rolling start"},
            {"demand_id": "transition_high", "label": "transition into high-speed running", "combine_analogue": "acc_6_8", "comparability": "same band"},
            {"demand_id": "braking", "label": "braking and deceleration", "combine_analogue": "", "comparability": "none: no braking phase"}]


def _demands() -> list[dict[str, Any]]:
    cell = lambda p, d, rate, n_ath, sup=True, plays=1000: {"position_group": p, "demand_id": d, "label": d, "n_episodes": int(rate * plays / 100), "n_athletes": n_ath,
                                                           "n_eligible_plays": plays, "episodes_per_100_plays": rate, "supported": sup}
    return [cell("WR", "accel_low", 80.0, 60), cell("WR", "accel_mid", 10.0, 55), cell("WR", "transition_high", 1.0, 5, sup=False), cell("WR", "braking", 30.0, 58),
            cell("OL", "accel_low", 80.0, 40, plays=5000), cell("OL", "accel_mid", 3.0, 35, plays=5000), cell("OL", "transition_high", 0.1, 2, sup=False, plays=5000),
            cell("OL", "braking", 10.0, 39, plays=5000), cell("DB", "accel_low", 90.0, 12), cell("DB", "accel_mid", 20.0, 12), cell("DB", "transition_high", 2.0, 11),
            cell("DB", "braking", 200.0, 12)]


def _summary() -> list[dict[str, Any]]:
    return [{"position_group": p, "demand_id": d, "dimension": "x", "n_prospects": 20} for p in ("WR", "OL", "DB") for d in ("accel_low", "accel_mid", "transition_high")]


# ============================================================================== CHECKS_KIT
def check_phase_decomposition_never_invents() -> str:
    twin, diffs, _ = _pair()
    dec = {p["phase_id"]: p for p in pk.phase_decomposition(diffs)}
    assert dec["initial"]["status"] == "not_identified" and not dec["initial"]["features"], "acc_2_4 absent → initial phase not identified"
    assert dec["early_dev"]["status"] == "distinguishable" and dec["early_dev"]["lead"]["feature"] == "acc_4_6", "acc_4_6 leads early development"
    inter = dec["intermediate"]
    assert inter["status"] == "distinguishable" and inter["lead"]["usable"], "the lead of a distinguishable phase is a repeatable feature"
    assert inter["lead"]["feature"] != "acc_6_8", "acc_6_8 (unstable) has the largest |z| but must not lead the phase"
    assert dec["late"]["status"] == "not_distinguishable", "speed_30yd inside its uncertainty"
    s = pk.phase_sentence(dec["initial"], ("A", "B"))
    assert "not identified" in s, "the sentence of an unidentified phase says so"
    return "absent phase stays 'not identified'; unstable feature never leads; verdicts follow the stored rows"


def check_replay_never_extrapolates() -> str:
    att = {"t": [0.3 + 0.1 * k for k in range(11)], "dist": [0.25 * k * k for k in range(11)], "speed": [0.5 * k for k in range(11)], "accel": [5.0] * 11}
    s = pk.sync_series(att, "time", 0.1, "speed")
    assert len(s["grid"]) == 11 and s["grid"][0] == 0.0 and abs(s["grid"][-1] - 1.0) < 1e-9, "time grid spans exactly the recording (re-based to 0)"
    assert all(abs(v - o) < 1e-9 for v, o in zip(s["value"], s["y_obs"])), "on a 10 Hz grid the values ARE the recorded frames"
    assert not any(s["interpolated"]), "no grid point of a 10 Hz grid needs interpolation"
    d = pk.sync_series(att, "distance", 1.0, "speed")
    assert d["grid"][-1] <= 25.0 and all(v is not None for v in d["value"]), "distance grid ends at the last recorded distance"
    assert any(d["interpolated"]), "a distance grid between frames is flagged as interpolated"
    assert pk._interp([0.0, 1.0], [0.0, 1.0], 1.5) is None, "no extrapolation beyond the recording"
    e = pk.sync_series({"t": [0.0], "dist": [0.0], "speed": [0.0]}, "time", 0.1, "speed")
    assert e["grid"] == [] and e["n_frames"] == 1, "one frame → nothing to replay"
    return "grid = recorded frames at 10 Hz; distance mode flags interpolation; nothing beyond the recording"


def check_evidence_labels_and_stability() -> str:
    lab = pk.evidence_label(None)
    assert lab["nfl_transfer"] == "not established" and "not assessed" in lab["repeatability"], "missing evidence row → stated, never assumed"
    row = {"feature": "time_4_8", "icc_within_position": 0.82, "r2_official_times": 0.65, "repeatability_label": "repeatable", "redundancy_label": "partly",
           "evidence_status": "repeatable_beyond_official", "n_athletes": 300, "n_repeat": 200, "football_value": "not established", "evidence_source": "08B.4"}
    lab = pk.evidence_label(row)
    assert "0.82" in lab["repeatability"] and "0.65" in lab["redundancy"] and "not established" in lab["nfl_transfer"], "the five answers carry the stored values"
    st = pk.attempt_stability([{"feature": "f", "label": "f", "unit": "s", "value": 1.0, "n_attempts": 1, "attempt_min": 1.0, "attempt_max": 1.0},
                               {"feature": "g", "label": "g", "unit": "s", "value": 1.0, "n_attempts": 2, "attempt_min": 0.9, "attempt_max": 1.1}])
    assert "single attempt" in st[0]["reading"] and abs(st[1]["spread"] - 0.2) < 1e-9, "single attempt flagged; range of two attempts = max − min"
    return "missing evidence is stated; labels quote stored ICC / R²; single attempts flagged"


# ============================================================================== CHECKS_GAP
def check_coverage_classes_follow_the_registry() -> str:
    reg = {r["demand_id"]: r for r in _registry()}
    sup = {"supported": True, "n_episodes": 500}
    assert mg.coverage_class(reg["braking"], sup, 10)[0] == mg.NOT_MEASURED, "no analogue → not measured, whatever the support"
    assert mg.coverage_class(reg["accel_low"], sup, 10)[0] == mg.DIRECT, "analogue + same start state → direct"
    assert mg.coverage_class(reg["accel_mid"], sup, 10)[0] == mg.PARTIAL, "analogue + different start state → partial"
    assert mg.coverage_class(reg["transition_high"], {"supported": False, "n_episodes": 4}, 10)[0] == mg.UNAVAILABLE, "unsupported cell → unavailable"
    assert mg.coverage_class(reg["accel_low"], sup, 0)[0] == mg.UNAVAILABLE, "no supported prospect comparison → unavailable"
    assert mg.coverage_class(reg["accel_low"], None, 10)[0] == mg.UNAVAILABLE, "absent cell → unavailable"
    return "classes derive from the analogue, the start state, the 09A support and the 09B comparison"


def check_gap_summary_and_focus_rule() -> str:
    M = mg.gap_matrix(_registry(), _demands(), _summary())
    S = {s["position_group"]: s for s in mg.gap_summary(M)}
    wr = S["WR"]
    assert abs(wr["rate_total"] - 120.0) < 1e-9, "WR: unsupported transition_high (1.0) excluded from the index (80 + 10 + 30)"
    assert abs(wr["share_not_measured"] - 30.0 / 120.0) < 1e-9, "WR share = braking / total"
    assert S["DB"]["share_not_measured"] > wr["share_not_measured"], "DB has the largest share in the fixture"
    f = mg.gap_position(list(S.values()))
    assert f is not None and f["position_group"] == "WR", "DB is excluded (12 athletes < 30): the focus is WR (share 0.25 > OL 0.107)"
    return "unsupported cells excluded from rates; the thin position is excluded from the focus rule"


def check_position_comparison_flags() -> str:
    M = mg.gap_matrix(_registry(), _demands(), _summary())
    c = mg.compare_positions(M, "WR", "OL", "braking")
    assert any("differ 5.0-fold" in f for f in c["flags"]), "eligible plays 1,000 vs 5,000 flagged"
    assert abs(c["rate_ratio_b_over_a"] - 10.0 / 30.0) < 1e-9, "rate ratio uses rates, not counts"
    c2 = mg.compare_positions(M, "WR", "DB", "accel_mid")
    assert any("only 12 athletes" in f for f in c2["flags"]), "thin athlete support flagged"
    c3 = mg.compare_positions(M, "WR", "OL", "transition_high")
    assert c3["rate_ratio_b_over_a"] is None and sum("too few" in f for f in c3["flags"]) == 2, "unsupported cells give no ratio"
    return "exposure and support differences are flagged; ratios only between supported rates"


# ============================================================================== CHECKS_SIMULATOR
def check_simulator_sections_and_categories() -> str:
    twin, diffs, P = _pair()
    M = mg.gap_matrix(_registry(), _demands(), _summary())
    ev = [{"feature": f, "label": f, "phase": "middle", "evidence_status": "repeatable_beyond_official"} for f in ("acc_4_6", "top_speed", "time_4_8")]
    r = ds.simulate("twins_differ", P[1], [], b=P[2], twin=twin, diffs=diffs, demands=_demands(), registry=_registry(), evidence=ev, gap=M)
    assert tuple(r["sections"]) == ds.SECTIONS, "six sections in the declared order"
    assert all(r["sections"][s] for s in ds.SECTIONS), "every section has at least one line"
    cats = r["category_counts"]
    assert cats["measured_difference"] >= 1 and cats["missing_observation"] >= 1, "a measured difference and the missing split of Beta Two"
    assert cats["uncertain_difference"] >= 1, "unstable acc_6_8 and the low tier are uncertain"
    no = ds.simulate("twins_differ", P[1], [], b=P[2], twin=None, diffs=(), demands=_demands(), registry=_registry(), evidence=ev, gap=M)
    assert no["category_counts"]["measured_difference"] == 0, "not a stored twin pair → no difference is computed"
    assert any("not a stored stopwatch-twin pair" in x["text"] for x in no["sections"]["Observation"]), "and the reason is stated"
    gm = ds.simulate("not_measured", P[1], [], demands=_demands(), registry=_registry(), evidence=ev, gap=M)
    assert gm["category_counts"]["measurement_mismatch"] >= 1, "braking is reported as a measurement mismatch"
    assert _raises(KeyError, lambda: ds.simulate("rank_them", P[1], [])), "unknown question refused"
    return "six structured sections; categories follow the stored evidence; non-twins get no computed difference"


def check_simulator_audiences_and_wording() -> str:
    twin, diffs, P = _pair()
    r = ds.simulate("twins_differ", P[1], [], b=P[2], twin=twin, diffs=diffs, demands=_demands(), registry=_registry(), evidence=[], gap=[])
    for aud in ("scout", "coach", "gm"):
        sec = ds.audience_sections(r, aud)
        text = " ".join(s for v in sec.values() for s in v).lower()
        assert not any(p in text for p in sin.FORBIDDEN_PHRASES), f"{aud}: no forbidden wording"
        assert "probability" not in text.replace("no success probability", ""), f"{aud}: no probability is produced"
    assert "Decision" in ds.audience_sections(r, "gm"), "the GM summary leads with the decision"
    assert _raises(ValueError, lambda: sin.assert_language("a hidden star")), "the guard refuses grading wording"
    assert _raises(KeyError, lambda: ds.audience_sections(r, "owner")), "unknown audience refused"
    html = ds.render_report(r, "scout", "footer")
    assert html.startswith("<!doctype html>") and "Scout report" in html, "self-contained HTML"
    return "three audiences render; no grading, ranking or probability wording"


# ============================================================================== CHECKS_STORIES
def check_wr_selection_rules() -> str:
    twin, diffs, P = _pair()
    pref = {**twin, "pair_id": "PREF", "name_a": ss.WR_PREFERRED[0], "name_b": ss.WR_PREFERRED[1], "nfl_id_a": 3, "nfl_id_b": 4, "nfl_outcome_value": 99.0}
    P2 = {**P, 3: {**P[1], "nfl_id": 3, "n_tracked_attempts": 1}, 4: {**P[2], "nfl_id": 4}}
    by = {"P1": diffs, "PREF": [{**d, "pair_id": "PREF"} for d in diffs]}
    cand = ss.wr_candidates([twin, pref], by, P2)
    chosen, why = ss.select_wr_pair(cand)
    assert chosen is not None and chosen["pair_id"] == "P1", "the preferred pair has a single attempt → criterion 4 fails → the other eligible pair"
    assert "attempts" in why, "the failed criterion is named"
    cand2 = ss.wr_candidates([twin, {**pref, "nfl_outcome_value": -99.0}], by, P2)
    assert [c["pair_id"] for c in cand2] == [c["pair_id"] for c in cand], "an NFL value on the row never changes the ranking"
    P3 = {**P2, 3: {**P2[3], "n_tracked_attempts": 3}}
    chosen3, why3 = ss.select_wr_pair(ss.wr_candidates([twin, pref], by, P3))
    assert chosen3["pair_id"] == "PREF" and "passes all six" in why3, "the preferred pair is used when it passes every criterion"
    return "preferred pair only when it passes; otherwise largest eligible contrast; NFL values ignored"


def check_story_deliverables() -> str:
    twin, diffs, P = _pair()
    s = ss.pair_story("wr", twin, diffs, P, _demands(), {}, {"selection": "test"})
    assert pk.words(s["summary_100"]) <= ss.SUMMARY_WORDS, "explanation ≤ 100 words"
    assert pk.words(s["narration_30s"]) <= ss.NARRATION_WORDS, "narration ≤ 80 words (about 30 s)"
    for k in ("title", "hero_caption", "card", "technical", "checklist", "provenance", "fallback"):
        assert s.get(k), f"deliverable {k} present"
    assert s["provenance"]["selected_using_nfl_outcome"] is False, "provenance records that no NFL outcome was used"
    M = mg.gap_matrix(_registry(), _demands(), _summary())
    focus = mg.gap_position(mg.gap_summary(M))
    g = ss.gap_story(focus, M, {}, {"run": "test"})
    assert len(g["sections"]) == 5 and pk.words(g["summary_100"]) <= ss.SUMMARY_WORDS, "gap story: five questions, ≤ 100 words"
    md = ss.story_markdown(s, "fig.png")
    assert "In 100 words" in md and "Provenance" in md, "markdown carries every deliverable"
    cases = [{"case_type": "same_forty_different_curve", "pair_id": "P1", "name_a": "x", "name_b": "y", "selected_using_nfl_outcome": False}]
    assert ss.select_same_forty(cases, {"P1": {**twin, "forty_b": 4.45}})[0] is None, "different official forties are not 'same forty'"
    assert ss.select_same_forty(cases, {"P1": twin})[0] is not None, "identical forties with a distinguishable feature validate"
    return "word limits, every deliverable, provenance; same-forty validation"


# ============================================================================== CHECKS_SUBMISSION
def check_token_rendering_and_templates() -> str:
    out, filled, missing = sb.render("a ‹M10:x› b ‹M10:y›", {"x": "1"})
    assert out == "a 1 b ‹M10:y›" and filled == ["x"] and missing == ["y"], "unfilled tokens stay visible and are listed"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "DOC.md").write_text("n = ‹M10:x›", encoding="utf-8")
        a1 = sb.render_documents(p, {"x": "1"}, names=("DOC.md",))
        assert (p / "templates" / "DOC.md").read_text(encoding="utf-8") == "n = ‹M10:x›", "the authored file is kept as the template"
        assert (p / "DOC.md").read_text(encoding="utf-8") == "n = 1", "rendered in place"
        sb.render_documents(p, {"x": "2"}, names=("DOC.md",))
        assert (p / "DOC.md").read_text(encoding="utf-8") == "n = 2", "a rerun refreshes from the template"
        assert a1["missing"][0] == "", "audit: nothing missing"
        T = {"stopwatch_twins": pl.DataFrame({"name_a": ["Anthony Bradford"], "name_b": ["Mason McCormick"], "forty_a": [5.08]})}
        ov = {"n_prospects": 510, "n_tracked_prospects": 418, "n_tracked_attempts": 793, "n_nfl_episodes": 416507, "n_twin_pairs": 3266,
              "n_twin_pairs_distinguishable": 2097, "n_cases": 23}
        (p / "templates" / "DOC.md").write_text("510 prospects, 418, 793, 416,507, 3,266, 2,097, 23, 5.08", encoding="utf-8")
        chk = sb.check_static_numbers(p, ov, T, names=("DOC.md",))
        assert bool(chk["present_in_document"].all()), "every typed anchor is found"
    return "tokens filled from values; authored text kept as template; reruns refresh; typed numbers cross-checked"


CHECKS_KIT: tuple[Callable[[], str], ...] = (check_phase_decomposition_never_invents, check_replay_never_extrapolates, check_evidence_labels_and_stability)
CHECKS_GAP: tuple[Callable[[], str], ...] = (check_coverage_classes_follow_the_registry, check_gap_summary_and_focus_rule, check_position_comparison_flags)
CHECKS_SIMULATOR: tuple[Callable[[], str], ...] = (check_simulator_sections_and_categories, check_simulator_audiences_and_wording)
CHECKS_STORIES: tuple[Callable[[], str], ...] = (check_wr_selection_rules, check_story_deliverables)
CHECKS_SUBMISSION: tuple[Callable[[], str], ...] = (check_token_rendering_and_templates,)
ALL_CHECKS: tuple[Callable[[], str], ...] = (*CHECKS_KIT, *CHECKS_GAP, *CHECKS_SIMULATOR, *CHECKS_STORIES, *CHECKS_SUBMISSION)


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    """Run the synthetic checks in-process. A failing check is reported, not raised, so one table shows them all."""
    rows = []
    for fn in checks:
        try:
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "PASS", "detail": fn()})
        except Exception as e:                                   # noqa: BLE001 — the table must list every failure
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "FAIL", "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


__all__ = ["ALL_CHECKS", "CHECKS_GAP", "CHECKS_KIT", "CHECKS_SIMULATOR", "CHECKS_STORIES", "CHECKS_SUBMISSION", "assert_all_passed", "run_all"]
