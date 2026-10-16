"""Synthetic validation of Milestone 09 (notebooks 09A–09G), callable from a notebook cell. NumPy / Polars; no project
data, no warehouse access. Every fixture is generated here with a planted, known answer. Tolerances were set by
reasoning — the author executed nothing — and each assertion states what it compares.

    from nflbdb.research.advanced_movement import intelligence_validation as iv
    report = iv.run_all(iv.CHECKS_DEMANDS); display(report); iv.assert_all_passed(report)

Groups: ``CHECKS_DEMANDS`` (09A) · ``CHECKS_MATCHING`` (09B) · ``CHECKS_TWINS`` (09C) · ``CHECKS_INTERPRETATION`` (09D) ·
``CHECKS_MART`` (09E – 09G). A failing check is reported with its location; it is never weakened to pass. Assertions that
rest on a fixed-seed random draw are marked ``[draw]``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import polars as pl

from . import milestone09_artifacts as m9
from . import movement_demands as md
from . import movement_matching as mm
from . import scouting_interpretation as sin
from . import stopwatch_twins as stw
from .scoring.scorecards import LeakageError
from .validation import _raises, assert_all_passed, failure_location

DC, MC, TC = md.DemandConfig(), mm.MatchingConfig(min_nfl_episodes=5, min_nfl_athletes=2, min_repeat=3), stw.TwinConfig()


# ============================================================================== fixtures
def _trace() -> dict[str, np.ndarray]:
    """rest → +3 yd/s² to 8 yd/s → hold 2 s → −4 yd/s² to about 3 yd/s → +3 yd/s² to 6 yd/s → hold."""
    t = np.round(np.arange(0.0, 8.01, 0.1), 1)
    v = np.where(t < 0.5, 0.5, np.where(t <= 3.0, 0.5 + 3.0 * (t - 0.5), np.where(t <= 5.0, 8.0, np.where(t <= 6.25, 8.0 - 4.0 * (t - 5.0),
                                                                                                    np.minimum(6.0, 3.0 + 3.0 * (t - 6.25))))))
    n = t.size
    return {"t": t, "v": v, "a": np.gradient(v, 0.1), "yaw": np.zeros(n), "step": np.concatenate([[0.0], (v[1:] + v[:-1]) / 2 * 0.1]),
            "hv": np.ones(n, bool), "edge": np.zeros(n, bool)}


def _episodes(position: str, demand: str, column: str, values: np.ndarray, n_athletes: int = 4) -> pl.DataFrame:
    rows = [{"game_id": 1, "play_id": i + 1, "nfl_id": 9000 + i % n_athletes, "episode": 1, "demand_id": demand, "position_group": position, column: float(x),
             "edge": False} for i, x in enumerate(values)]
    return pl.DataFrame(rows, schema=md.EPISODES.columns)


def _twin_fixture(n: int = 12) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(profile, prospect features, evidence): WR athletes in forty steps of 0.01 s; athletes of odd index are 0.5 units
    'more' on every curve feature (standard error 0.05), so odd-even pairs differ and same-parity pairs do not."""
    ids = list(range(1, n + 1))
    prof = pl.DataFrame({"nfl_id": ids, "display_name": [f"P{i}" for i in ids], "position_group": ["WR"] * (n - 2) + ["DB"] * 2, "draft_year": [2025] * n,
                         "official_forty": [4.40 + 0.01 * (k // 2) for k in range(n)], "official_ten": [1.50] * n, "weight": [200.0] * n, "height": [72.0] * n,
                         "fingerprint_available": [True] * n, "evidence_tier": ["high" if k % 2 == 0 else "low" for k in range(n)]})
    feats = mm.combine_features(MC)
    rows = [{"nfl_id": i, "position_group": "WR" if k < n - 2 else "DB", "feature": f.feature, "label": f.label, "unit": f.unit, "value": 5.0 + 0.5 * (k % 2),
             "n_attempts": 2, "attempt_min": 5.0 + 0.5 * (k % 2) - 0.02 * (k % 3), "attempt_max": 5.0 + 0.5 * (k % 2) + 0.02 * (k % 3), "standard_error": 0.05,
             "pct_position": 50.0, "n_reference": 9, "state": "supported"} for k, i in enumerate(ids) for f in feats]
    ev = pl.DataFrame([{"feature": f.feature, "label": f.label, "unit": f.unit, "phase": f.phase, "higher_is_quicker": f.higher_is_quicker,
                        "evidence_status": "repeatable_partly_beyond_stopwatch", "icc_within_position": 0.8, "r2_official_times": 0.5, "n_repeat": 40,
                        "evidence_source": "synthetic"} for f in feats])
    return prof, pl.DataFrame(rows, schema=mm.PROSPECT.columns), ev


def _twins() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    prof, prospect, ev = _twin_fixture()
    pairs = stw.twin_index(prof, TC, dist=lambda a, b: float(abs(a - b)))
    diffs = stw.difference_table(pairs, prospect, ev)
    return stw.rank_twins(pairs, diffs, TC), diffs, prospect, ev


# ============================================================================== 09A
def check_first_crossing_and_band_acceleration() -> str:
    t = np.arange(0, 3.01, 0.1)
    v = 1.0 + 2.0 * t
    assert abs(md.crossing_time(t, v, 4.0) - 1.5) < 1e-9, "first crossing of 4 yd/s on v = 1 + 2 t is at 1.5 s"
    assert abs(md.band_accel(t, v, 2.0, 4.0) - 2.0) < 1e-9 and abs(md.band_accel(t, v, 4.0, 6.0) - 2.0) < 1e-9, "band acceleration of a 2 yd/s² ramp is 2 in every band"
    assert np.isnan(md.crossing_time(t, v + 5.0, 4.0)) and np.isnan(md.band_accel(t, v + 5.0, 4.0, 6.0)), "a trace that starts above the level has no recorded crossing"
    assert np.isnan(md.band_accel(t, np.minimum(v, 5.0), 4.0, 6.0)), "a band whose upper level is never reached abstains"
    assert md.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)], "runs are [start, end) pairs"
    return "first-crossing times and band accelerations are exact on a ramp; unrecorded crossings give no value"


def check_demand_segmentation_on_a_known_trace() -> str:
    x = _trace()
    eps = md.segment_trace(x["t"], x["v"], x["a"], x["yaw"], x["step"], x["hv"], x["edge"], DC)
    found = {e["demand_id"] for e in eps}
    want = {"accel_low", "sustained_high", "speed_maintenance", "braking", "reaccel_after_brake", "repeated_change"}
    assert want <= found, f"a rest → sprint → hold → brake → re-accelerate trace holds {sorted(want)}; found {sorted(found)}"
    assert "redirection" not in found and "accel_mid" not in found, "no turn without a yaw rate; the run after braking is a re-acceleration, not a second category"
    low = next(e for e in eps if e["demand_id"] == "accel_low")
    assert abs(low[md.BAND_COLS[0]] - 3.0) < 0.05 and abs(low[md.BAND_COLS[1]] - 3.0) < 0.05 and low["v_start"] < DC.low_speed, "band accelerations of the +3 yd/s² ramp"
    brake = next(e for e in eps if e["demand_id"] == "braking")
    assert brake["speed_change"] < -DC.brake_drop and brake["mean_accel"] < 0 and brake["peak_rate_05"] < 0, "braking is signed negative"
    rep = next(e for e in eps if e["demand_id"] == "repeated_change")
    assert rep["n_changes"] == 3, "two accelerating episodes and one braking episode"
    yaw = np.where((x["t"] >= 3.5) & (x["t"] <= 4.5), 90.0, 0.0)
    turn = [e for e in md.segment_trace(x["t"], x["v"], x["a"], yaw, x["step"], x["hv"], x["edge"], DC) if e["demand_id"] == "redirection"]
    assert len(turn) == 1 and abs(turn[0]["heading_change_deg"]) >= DC.turn_min_deg, "a 90 deg/s turn held for 1 s is one redirection"
    none = md.segment_trace(x["t"], x["v"], x["a"], yaw, x["step"], np.zeros(x["t"].size, bool), x["edge"], DC)
    assert not [e for e in none if e["demand_id"] == "redirection"], "no redirection is reported where the heading is not valid"
    flagged = md.segment_trace(x["t"], x["v"], x["a"], x["yaw"], x["step"], x["hv"], np.ones(x["t"].size, bool), DC)
    assert all(e["edge"] for e in flagged), "episodes touching filter-edge frames are flagged (profiles exclude them)"
    return "the declared demand rules recover every phase of a known trace; turns need a valid heading; edge frames are flagged"


def check_variance_components_and_profiles() -> str:
    rng = np.random.default_rng(7)
    ids = np.repeat(np.arange(200), 5)
    y = rng.normal(0, 1, 200)[ids] + rng.normal(0, 1, ids.size)
    w, b, share = md.variance_components(y, ids)
    assert abs(share - 0.5) < 0.1 and abs(w - 1.0) < 0.1, f"[draw] planted athlete share 0.5, within SD 1; got {share:.2f}, {w:.2f}"
    assert np.isnan(md.variance_components(np.array([1.0, 2.0]), np.array([1, 1]))[2]), "one athlete: no between-athlete estimate"
    ep = pl.concat([_episodes("WR", "sustained_high", "v_max", np.linspace(8, 10, 40)), _episodes("DL", "sustained_high", "v_max", np.array([8.0, 8.5]))])
    plays = pl.DataFrame({"game_id": [1] * 50, "play_id": list(range(1, 51)), "nfl_id": [9000] * 50, "position_group": ["WR"] * 40 + ["DL"] * 10, "eligible": [True] * 50})
    freq = md.frequency_table(ep, plays, DC)
    wr = freq.filter((pl.col("position_group") == "WR") & (pl.col("demand_id") == "sustained_high")).row(0, named=True)
    dl = freq.filter((pl.col("position_group") == "DL") & (pl.col("demand_id") == "sustained_high")).row(0, named=True)
    assert wr["n_episodes"] == 40 and abs(wr["episodes_per_100_plays"] - 100.0) < 1e-9 and not dl["supported"], "rates carry their exposure; a thin cell is not supported"
    assert freq.filter(pl.col("demand_id") == "braking")["n_episodes"].sum() == 0, "a demand with no episode is listed with a zero count, not dropped"
    prof = md.profile_table(ep, DC)
    assert not prof.filter(pl.col("position_group") == "DL")["supported"].any(), "no distribution is called supported below the support lines"
    return "variance components recover a planted athlete share; thin position × demand cells are unsupported, absent ones counted as zero"


# ============================================================================== 09B
def check_shared_rule_on_both_domains() -> str:
    t = np.arange(0, 6.01, 0.1)
    v = np.minimum(10.0, 2.5 * t)
    d = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * 0.1)])
    f = mm.trace_features(t, v, d, MC)
    assert abs(f[md.BAND_COLS[1]] - md.band_accel(t, v, 4.0, 6.0)) < 1e-12 and abs(f[md.BAND_COLS[1]] - 2.5) < 1e-9, "the Combine side calls the NFL-side band rule"
    assert abs(f["speed_10yd"] - np.interp(10.0, d, v)) < 1e-12 and np.isnan(mm.trace_features(t[:5], v[:5], d[:5], MC)["speed_30yd"]), "speed at distance only inside the recording"
    started = mm.trace_features(t, v + 3.0, d, MC)
    assert np.isnan(started[md.BAND_COLS[0]]) and np.isfinite(started[md.BAND_COLS[2]]), "a recording that starts above 2 yd/s has no 2 → 4 band but keeps later bands"
    return "Combine band accelerations use the same first-crossing rule as NFL episodes; unrecorded intervals abstain"


def check_similarity_coverage_and_abstention() -> str:
    ep = pl.concat([_episodes("WR", "accel_mid", md.BAND_COLS[1], np.linspace(1.0, 3.0, 21)), _episodes("WR", "transition_high", md.BAND_COLS[2], np.array([1.0, 2.0]))])
    base = {"position_group": "WR", "label": "x", "unit": "yd/s²", "n_attempts": 2, "attempt_min": None, "attempt_max": None, "pct_position": None, "n_reference": 0}
    prospect = pl.DataFrame([{**base, "nfl_id": 1, "feature": md.BAND_COLS[1], "value": 5.0, "standard_error": 0.1, "state": "supported"},
                             {**base, "nfl_id": 1, "feature": md.BAND_COLS[2], "value": 2.0, "standard_error": 0.1, "state": "supported"},
                             {**base, "nfl_id": 2, "feature": md.BAND_COLS[1], "value": 2.0, "standard_error": None, "state": "insufficient_evidence"},
                             {**base, "nfl_id": 3, "feature": md.BAND_COLS[1], "value": None, "standard_error": None, "state": "not_observed"}], schema=mm.PROSPECT.columns)
    sim = mm.similarity_table(prospect, ep, DC, MC)
    get = lambda i, dem, dim: sim.filter((pl.col("nfl_id") == i) & (pl.col("demand_id") == dem) & (pl.col("dimension") == dim)).row(0, named=True)
    top = get(1, "accel_mid", md.BAND_COLS[1])
    assert top["state"] == "supported" and top["coverage"] == 100.0 and top["coverage_lo"] == 100.0, "a Combine value above every NFL episode covers 100 %"
    assert get(1, "transition_high", md.BAND_COLS[2])["state"] == "abstain_thin_nfl_reference", "two NFL episodes are not a reference"
    mid = get(2, "accel_mid", md.BAND_COLS[1])
    assert mid["state"] == "abstain_no_error_estimate" and mid["coverage_lo"] is None and abs(mid["coverage"] - 100 * 11 / 21) < 1e-9, "no error estimate → value without an interval"
    assert get(3, "accel_mid", md.BAND_COLS[1])["state"] == "abstain_not_recorded" and get(3, "accel_mid", md.BAND_COLS[1])["coverage"] is None, "an unrecorded value is never imputed"
    un = sim.filter(pl.col("state") == "unsupported_by_forty")
    assert set(un["demand_id"].unique().to_list()) == {"speed_maintenance", "braking", "reaccel_after_brake", "redirection", "repeated_change"} and un["coverage"].null_count() == un.height, \
        "demands the forty cannot describe abstain for every prospect"
    mat = mm.matrix_table(sim)
    assert mat["reading"].str.contains("not a prediction").all() and mat.filter((pl.col("nfl_id") == 1) & (pl.col("demand_id") == "accel_mid"))["state"][0] == "supported", \
        "every matrix row is labelled descriptive"
    assert _raises(LeakageError, lambda: mm.assert_pre_draft(prospect.with_columns(pl.lit(1.0).alias("nfl_accel_sust05_mean")), "prospect")), "an NFL column in a Combine table is refused"
    return "coverage is a position inside the NFL demand distribution; thin references, missing values and unsupported demands abstain"


# ============================================================================== 09C
def check_twin_matching_uses_pre_draft_information_only() -> str:
    prof, prospect, ev = _twin_fixture()
    pairs = stw.twin_index(prof, TC)
    pos = dict(zip(prof["nfl_id"].to_list(), prof["position_group"].to_list()))
    assert pairs.height > 0 and (pairs["d_forty"].abs() <= TC.forty_tol + 1e-9).all(), "every pair is inside the declared forty caliper"
    assert all(pos[a] == pos[b] for a, b in zip(pairs["nfl_id_a"].to_list(), pairs["nfl_id_b"].to_list())) and (pairs["nfl_id_a"] < pairs["nfl_id_b"]).all(), "same position, one row per pair"
    brute = sum(1 for i in range(prof.height) for j in range(i + 1, prof.height) if prof["position_group"][i] == prof["position_group"][j]
                and abs(prof["official_forty"][i] - prof["official_forty"][j]) <= TC.forty_tol + 1e-9)
    assert pairs.height == brute, "the sorted scan finds exactly the pairs of a brute-force search"
    assert _raises(LeakageError, lambda: stw.twin_index(prof.with_columns(pl.lit(1.0).alias("nfl_speed_p90_mean")), TC)), "an NFL column in the matching table is refused"
    diffs = stw.difference_table(pairs, prospect, ev)
    twins = stw.rank_twins(pairs, diffs, TC)
    odd_even = twins.filter((pl.col("nfl_id_a") == 1) & (pl.col("nfl_id_b") == 2)).row(0, named=True)
    same = twins.filter((pl.col("nfl_id_a") == 1) & (pl.col("nfl_id_b") == 3)).row(0, named=True)
    assert odd_even["n_distinguishable"] == len(mm.combine_features(MC)) and same["n_distinguishable"] == 0, "a planted 0.5 difference (SE 0.05) is distinguishable; no difference is not"
    d = diffs.filter((pl.col("pair_id") == "1-2") & (pl.col("feature") == "time_4_8")).row(0, named=True)
    assert d["quicker"] == "a" and diffs.filter((pl.col("pair_id") == "1-2") & (pl.col("feature") == "top_speed"))["quicker"][0] == "b", "a larger TIME favours the other athlete"
    assert twins.filter(pl.col("position_group") == "WR")["rank_in_position"].min() == 1, "ranks restart within position"
    summ = stw.twin_summary(twins, diffs)
    assert (summ["expected_by_chance"] == 0.05).all() and summ.filter(pl.col("position_group") == md.ALL)["n_tested"].min() == twins.height, "the summary carries the chance line"
    return "twins are matched on official results only, differences are tested on the SE of the difference, time features are oriented"


def check_case_selection_rules() -> str:
    twins, diffs, prospect, _ = _twins()
    cases = stw.select_cases(twins, diffs, prospect, TC)
    assert cases.equals(stw.select_cases(twins, diffs, prospect, TC)), "the selection is deterministic"
    assert not cases["selected_using_nfl_outcome"].any() and "similar_movement_nfl_observed" not in cases["case_type"].to_list(), "without exposure counts the NFL-shown type is absent"
    for (_,), g in cases.partition_by(["case_type"], as_dict=True).items():
        ids = g["nfl_id_a"].to_list() + g["nfl_id_b"].to_list()
        assert len(ids) == len(set(ids)) or g["case_type"][0] == "position_specific_comparison", "no athlete appears twice in a case type"
    assert {"same_forty_different_curve", "apparent_difference_overlapping_uncertainty", "tracking_quality_limitation"} <= set(cases["case_type"].to_list()), "declared types are found"
    assert "high_confidence_difference" in stw.missing_case_types(cases), "a type without a qualifying pair is reported absent, not forced"
    lo, hi = {i: 60 for i in range(1, 13)}, {i: 60 + 100 * i for i in range(1, 13)}
    a, b = stw.select_cases(twins, diffs, prospect, TC, lo), stw.select_cases(twins, diffs, prospect, TC, hi)
    assert a.equals(b) and a.filter(pl.col("case_type") == "similar_movement_nfl_observed")["selected_using_nfl_exposure"].all(), "exposure is an eligibility line: its size never orders pairs"
    few = stw.select_cases(twins, diffs, prospect, TC, {i: 10 for i in range(1, 13)})
    assert "similar_movement_nfl_observed" not in few["case_type"].to_list(), "below the exposure line the case type is absent"
    assert _raises(LeakageError, lambda: stw.select_cases(twins.with_columns(pl.lit(1.0).alias("nfl_separation_mean")), diffs, prospect, TC)), "an NFL outcome column in a selection table is refused"
    return "ten declared case types, deterministic, pre-draft; absent types are reported; NFL exposure is eligibility only"


# ============================================================================== 09D
def check_interpretation_wording_and_views() -> str:
    twins, diffs, _, ev = _twins()
    twin = twins.filter(pl.col("pair_id") == "1-2").row(0, named=True)
    d = diffs.filter(pl.col("pair_id") == "1-2").to_dicts()
    read = sin.pair_reading(twin, d)
    assert list(read) == ["Observation", "Potential football relevance", "Scouting follow-up", "Limitations"], "the four sections of a pair reading"
    assert sin.NOT_ESTABLISHED in read["Potential football relevance"] and any("not demonstrated" in s for s in read["Limitations"]), "relevance is qualified; causation is disclaimed"
    assert any(s.startswith("Film") for s in read["Scouting follow-up"]) and any(s.startswith("Testing") for s in read["Scouting follow-up"]), "a practical follow-up is always given"
    flat = sin.pair_reading(twins.filter(pl.col("pair_id") == "1-3").row(0, named=True), diffs.filter(pl.col("pair_id") == "1-3").to_dicts())
    assert any("not distinguishable" in s for s in flat["Observation"]) and not any(s.startswith("Film —") for s in flat["Scouting follow-up"]), "no difference → no movement phase is named"
    case = {"case_id": "c-1"}
    views = sin.case_views(case, twin, d, [{"demand_id": "accel_mid", "episodes_per_100_plays": 40.0, "n_episodes": 1000, "n_athletes": 50, "supported": True}])
    assert set(views) == {"scout", "coach", "general_manager"} and all(len(v) >= 4 for v in views.values()), "three views, each with its sections"
    assert "Does this justify additional evaluation" in views["general_manager"] and "not an input for ranking" in views["general_manager"]["Does this justify additional evaluation"][0], \
        "the general-manager view answers the evaluation question without a ranking"
    rows = sin.views_rows("c-1", views)
    assert {r["view"] for r in rows} == set(views) and all(r["text"] for r in rows), "views flatten to one row per sentence"
    for bad in ("He projects as a starter", "a draft grade of 90", "a hidden star", "the better receiver"):
        assert _raises(ValueError, lambda bad=bad: sin.assert_language(bad)), f"the wording guard refuses {bad!r}"
    reg = sin.interpretation_registry(ev.to_dicts())
    need = {"what_is_measured", "physical_meaning", "difference_from_official_testing", "reliability", "related_football_movements", "what_to_look_at_next", "what_cannot_be_concluded"}
    assert all(need <= set(r) and all(r[k] for k in need) for r in reg) and len(reg) == len(sin.INTERPRETATIONS), "every displayed metric answers the seven questions"
    page = sin.render_html("P1", "WR", {"A": ["one"], "B": {"sub": ["two"]}}, tables=[("T", [{"x": 1.0, "y": None}], ["x", "y"])])
    assert page.startswith("<!doctype html>") and sin.DISCLAIMER[:40] in page and "<script" not in page, "the report is self-contained and carries the disclaimer"
    return "generated readings are qualified, always give a follow-up, never grade or rank; three views per case; seven answers per metric"


# ============================================================================== 09E – 09G
def _cell(name: str) -> Any:
    if name in ("nfl_id", "nfl_id_a", "nfl_id_b", "frame", "attempt_order", "section_order", "line_order", "bin") or name.startswith("n_"):
        return 1
    return {"selected_using_nfl_outcome": False, "supported": True, "boundary": False, "in_episode": True, "comparable": True, "higher_is_quicker": True,
            "reading": mm.DESCRIPTIVE, "football_value": "not established", "state": "supported", "pair_id": "1-1", "case_id": "c-1", "position_group": "ALL",
            "coverage_lo": 10.0}.get(name, "x")


def _mart_tables() -> dict[str, pl.DataFrame]:
    t = {k: pl.DataFrame([{c: _cell(c) for c in cols}]) for k, cols in m9.MART_TABLES.items()}
    t["case_views"] = pl.concat([t["case_views"].with_columns(pl.lit(v).alias("view")) for v in ("scout", "coach", "general_manager")])
    return t


def check_application_mart_contract() -> str:
    evidence = m9.evidence_document([{"feature_id": "F"}], [{"claim_id": "C01"}], [{"metric": "m"}], [{"feature": "f"}], {"a": "b"})
    with tempfile.TemporaryDirectory() as tmp:
        store = SimpleNamespace(mart_dir=Path(tmp), controls=SimpleNamespace(run_mode="smoke"))
        manifest = m9.write_mart(store, _mart_tables(), evidence, {"notebook": "synthetic"})
        path = m9.mart_path(store)
        rep = m9.validate_mart(path)
        assert rep["passed"].all(), "a complete mart passes: " + "; ".join(f"{r['check']}: {r['detail']}" for r in rep.filter(~pl.col("passed")).iter_rows(named=True))
        assert manifest["mart_version"] == m9.MART_VERSION and manifest["overview"]["n_twin_pairs"] == 1 and manifest["run_mode"] == "smoke", "the manifest declares version and run mode"
        assert not m9.validate_mart(path, version="v999").filter(pl.col("check") == "declared_version")["passed"][0], "another declared version is refused"
        t = pl.read_parquet(path / "prospect_profiles.parquet")
        t.with_columns(pl.lit(90.0).alias("draft_grade")).write_parquet(path / "prospect_profiles.parquet")
        rep = m9.validate_mart(path)
        failed = set(rep.filter(~pl.col("passed"))["check"].to_list())
        assert {"no_grade_or_ranking_column", "fingerprints_match_manifest"} <= failed, f"a grade column and a changed file are both caught; failed = {sorted(failed)}"
        assert _raises(AssertionError, lambda: m9.assert_valid(rep)), "assert_valid raises on a failed check"
        (path / "stopwatch_twins.parquet").unlink()
        assert not m9.validate_mart(path).filter(pl.col("check") == "tables_present")["passed"][0], "a missing table is caught"
        tables = _mart_tables()
        del tables["scouting_cases"]
        assert _raises(KeyError, lambda: m9.write_mart(store, tables, evidence, {})), "an incomplete mart is not written"
    with tempfile.TemporaryDirectory() as tmp:
        assert not m9.validate_mart(Path(tmp))["passed"][0], "a directory without a manifest is not a mart"
    return "the mart declares its version, matches its manifest, holds no grade column and refuses to be written incomplete"


def check_mart_referential_integrity() -> str:
    evidence = m9.evidence_document([{"feature_id": "F"}], [], [{"metric": "m"}], [], {})
    with tempfile.TemporaryDirectory() as tmp:
        store = SimpleNamespace(mart_dir=Path(tmp), controls=SimpleNamespace(run_mode="smoke"))
        t = _mart_tables()
        t["prospect_metrics"] = t["prospect_metrics"].with_columns(pl.lit(777).cast(pl.Int64).alias("nfl_id"))
        t["scouting_cases"] = t["scouting_cases"].with_columns(pl.lit("9-9").alias("pair_id"), pl.lit(True).alias("selected_using_nfl_outcome"))
        t["movement_similarity"] = t["movement_similarity"].with_columns(pl.lit("abstain_not_recorded").alias("state"))
        t["feature_evidence"] = t["feature_evidence"].with_columns(pl.lit("validated").alias("football_value"))
        m9.write_mart(store, t, evidence, {})
        failed = set(m9.validate_mart(m9.mart_path(store)).filter(~pl.col("passed"))["check"].to_list())
    want = {"displayed_athletes_exist_in_source", "cases_backed_by_twin_pairs", "no_case_selected_on_nfl_outcome", "unsupported_comparisons_abstain", "football_value_not_claimed"}
    assert want <= failed, f"an unknown athlete, an unbacked case, an outcome-selected case, an interval on an abstention and a football-value claim are caught; missing {sorted(want - failed)}"
    return "every displayed athlete exists in the profile table; cases, abstentions and football-value wording are enforced"


CHECKS_DEMANDS: tuple[Callable[[], str], ...] = (check_first_crossing_and_band_acceleration, check_demand_segmentation_on_a_known_trace,
                                                 check_variance_components_and_profiles)
CHECKS_MATCHING: tuple[Callable[[], str], ...] = (check_shared_rule_on_both_domains, check_similarity_coverage_and_abstention)
CHECKS_TWINS: tuple[Callable[[], str], ...] = (check_twin_matching_uses_pre_draft_information_only, check_case_selection_rules)
CHECKS_INTERPRETATION: tuple[Callable[[], str], ...] = (check_interpretation_wording_and_views,)
CHECKS_MART: tuple[Callable[[], str], ...] = (check_application_mart_contract, check_mart_referential_integrity)
ALL_CHECKS: tuple[Callable[[], str], ...] = (*CHECKS_DEMANDS, *CHECKS_MATCHING, *CHECKS_TWINS, *CHECKS_INTERPRETATION, *CHECKS_MART)


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    """Run the synthetic checks in-process. A failing check is reported, not raised, so one table shows them all."""
    rows = []
    for fn in checks:
        try:
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "PASS", "detail": fn()})
        except Exception as e:                                   # noqa: BLE001 — the table must list every failure
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "FAIL", "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


__all__ = ["ALL_CHECKS", "CHECKS_DEMANDS", "CHECKS_INTERPRETATION", "CHECKS_MART", "CHECKS_MATCHING", "CHECKS_TWINS", "assert_all_passed", "run_all"]
