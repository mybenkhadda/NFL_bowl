"""Study C2 annotation framework: task selection, label validation, matching, taxonomy, evaluation.

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_event_annotation.py -q
"""

from __future__ import annotations

import polars as pl
import pytest

from nflbdb.research.contracts import ContractError, conform
from nflbdb.research.studies import annotation as A

TOL = 0.3


def test_matching_is_one_to_one_nearest_onset_within_tolerance():
    ref = [(1.0, 1.5, False), (2.0, 2.4, False)]
    det = [(1.1, 1.6), (1.15, 1.7), (3.0, 3.2)]
    pairs, fn, fp = A.match_events(ref, det, TOL)
    assert pairs == [(0, 0)] and fn == [1] and fp == [1, 2]


def test_matching_respects_tolerance_boundary():
    pairs, fn, fp = A.match_events([(1.0, 2.0, False)], [(1.31, 2.0)], TOL)
    assert pairs == [] and fn == [0] and fp == [0]


def test_error_taxonomy():
    rows = A.classify_errors(
        {"braking": [(1.0, 2.0, False), (5.0, 6.0, False)], "turn": [(8.0, 9.0, False)]},
        {"braking": [(1.05, 1.4), (1.5, 1.9), (8.2, 8.6)], "acceleration": [(5.2, 5.8)]}, TOL)
    kinds = {(r["event_class"], r["outcome"], r["error_type"]) for r in rows}
    assert ("braking", "TP", "") in kinds
    assert ("braking", "FP", "fragmentation") in kinds          # second piece of the same braking
    assert ("braking", "FP", "class_confusion") in kinds        # braking detected where a turn was labelled
    assert ("braking", "FN", "class_confusion") in kinds        # braking labelled where acceleration detected
    assert ("turn", "FN", "class_confusion") in kinds


def test_timing_error_category():
    rows = A.classify_errors({"braking": [(1.0, 2.0, False)]}, {"braking": [(1.5, 2.2)]}, TOL)
    assert {(r["outcome"], r["error_type"]) for r in rows} == {("FN", "timing"), ("FP", "spurious")}


def _tasks():
    return pl.DataFrame({"task_id": ["T0001", "T0002"], "domain": ["combine"] * 2, "traj_id": ["E1|1|1", "E2|2|1"],
                         "nfl_id": [1, 2], "drill": ["SHORT_SHUTTLE"] * 2, "cohort": [2024, 2025],
                         "position": ["WR", "DB"], "selection": ["typical", "difficult"], "selection_rule": ["r"] * 2,
                         "duration_s": [6.0, 6.0]}, schema=A.TASKS.columns)


def _labels(rows):
    return pl.DataFrame(rows, schema=A.LABELS.columns, orient="row")


def test_label_validation_catches_logical_errors():
    ok = _labels([("T0001", "ann", 1, "braking", 1.0, 1.5, "certain", ""),
                  ("T0002", "ann", 1, "none", None, None, "certain", "")])
    A.validate_labels(ok, _tasks())
    for bad in ([("T0001", "ann", 1, "brake", 1.0, 1.5, "certain", "")],
                [("T0001", "ann", 1, "braking", 1.5, 1.0, "certain", "")],
                [("T0001", "ann", 1, "braking", 1.0, 9.0, "certain", "")],
                [("T0001", "ann", 1, "none", None, None, "certain", ""),
                 ("T0001", "ann", 2, "braking", 1.0, 1.5, "certain", "")],
                [("T0009", "ann", 1, "braking", 1.0, 1.5, "certain", "")]):
        with pytest.raises(ContractError):
            A.validate_labels(_labels(bad), _tasks())


def test_match_all_and_evaluate_with_uncertain_labels():
    labels = _labels([("T0001", "ann", 1, "braking", 1.0, 1.5, "certain", ""),
                      ("T0001", "ann", 2, "reacceleration", 2.0, 2.6, "uncertain", ""),
                      ("T0002", "ann", 1, "none", None, None, "certain", "")])
    det = pl.DataFrame({"traj_id": ["E1|1|1", "E1|1|1", "E2|2|1"], "rule": ["default"] * 3,
                        "event_type": ["braking", "acceleration", "braking"], "t_start_s": [1.1, 2.1, 3.0],
                        "t_end_s": [1.6, 2.5, 3.3]})
    m = conform(A.match_all(labels, det, _tasks(), TOL), A.MATCHES)
    assert m.filter(pl.col("outcome") == "TP").height == 2      # reacceleration ↦ acceleration
    assert m.filter(pl.col("outcome") == "FP").height == 1      # braking on a task labelled 'none'
    perf = conform(A.evaluate(m, A.AnnotationConfig(n_boot=100)), A.PERFORMANCE)
    allb = perf.filter((pl.col("facet") == "all") & (pl.col("event_class") == "braking")
                       & (pl.col("label_set") == "all_labels")).row(0, named=True)
    assert (allb["tp"], allb["fp"], allb["fn"]) == (1, 1, 0) and allb["precision"] == pytest.approx(0.5)
    certain = perf.filter((pl.col("facet") == "all") & (pl.col("event_class") == "acceleration")
                          & (pl.col("label_set") == "certain_only"))
    assert certain.height == 0 or certain["tp"][0] == 0           # uncertain reference is ignored, not scored


def test_task_selection_is_deterministic_and_blind():
    dims = pl.DataFrame({"domain": ["combine"] * 8, "traj_id": [f"E{i}|{i}|1" for i in range(8)],
                         "nfl_id": list(range(8)), "cohort": [2024] * 8, "stratum": ["SHORT_SHUTTLE"] * 6 + ["SKILL_DRILLS_DB"] * 2,
                         "position": ["WR"] * 8, "quality_tier": ["A"] * 7 + ["reject"], "duration_s": [6.0] * 8,
                         "event_count_rel_range": [0.0, 0.1, 0.2, 0.5, 0.9, 0.3, 0.8, 1.0]})
    t1, t2 = A.select_tasks(dims), A.select_tasks(dims.reverse())
    assert t1.equals(t2) and "E7|7|1" not in t1["traj_id"].to_list()          # reject excluded
    conform(t1, A.TASKS)
    assert set(t1["selection"]) == {"typical", "difficult"}
    assert not any(c.startswith(("det", "rule")) for c in t1.columns)        # no detector output in the task list
