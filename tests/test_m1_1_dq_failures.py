"""M1.1 · Study A0 — exact drill-down of failed Phase 01 checks.

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_m1_1_dq_failures.py -q
"""

from __future__ import annotations

import polars as pl
import pytest

from nflbdb.research.contracts import conform
from nflbdb.research.studies import dq_failures as A0
from nflbdb.research.studies.config import IntegrityConfig

ICFG = IntegrityConfig()


# ----------------------------------------------------------------------------- pure classification
@pytest.mark.parametrize("dup, with_ent, player, expected", [
    (0, None, None, "not_reproduced"),
    (16, 0, 0, "rule_error"),            # BALL rows share the athlete's key: grain incomplete
    (16, 4, 0, "rule_error"),            # non-PLAYER rows only
    (16, 4, 4, "data_defect"),
    (16, None, None, "data_defect"),     # table without entity_type
])
def test_classify_key_duplicates(dup, with_ent, player, expected):
    assert A0.classify_key_duplicates(dup, with_ent, player)[0] == expected


@pytest.mark.parametrize("n_out, n_player, derived, expected", [
    (0, 0, None, "not_reproduced"), (177, 0, None, "rule_error"), (177, 3, 0, "data_defect_handled"),
    (177, 3, 2, "data_defect"), (3, None, None, "data_defect"),
])
def test_classify_range(n_out, n_player, derived, expected):
    assert A0.classify_range(n_out, n_player, derived)[0] == expected


def test_classify_fk_and_trk():
    assert A0.classify_fk(0, "game_tracking", "player_play")[0] == "not_reproduced"
    cls, _, res = A0.classify_fk(150, "game_tracking", "player_play")
    assert cls == "coverage_gap" and res == "exclude_from_context_features"
    assert A0.classify_trk(0)[0] == "not_reproduced" and A0.classify_trk(4)[0] == "data_defect_handled"


def test_blocking_only_for_unresolved_defects_at_blocking_severity():
    assert A0.is_blocking("data_defect", "CRITICAL", "no", ICFG)
    assert not A0.is_blocking("data_defect", "CRITICAL", "yes", ICFG)
    assert not A0.is_blocking("data_defect", "MEDIUM", "no", ICFG)
    assert not A0.is_blocking("rule_error", "CRITICAL", "unknown", ICFG)


def test_parsers():
    assert A0.parse_range("[0, 12.5]") == (0.0, 12.5) and A0.parse_range("[-10.0, 63.3]") == (-10.0, 63.3)
    assert A0.parse_range("<= 0.00%") is None
    assert A0.parse_fk("fk:game_tracking:game_id,play_id->player_play") == ("game_tracking", ["game_id", "play_id"],
                                                                            "player_play")
    assert A0.parse_fk("fk:game_tracking:game_id,play_id,nfl_id->player_play")[1] == ["game_id", "play_id", "nfl_id"]
    assert A0.parse_fk("uniq:players:nfl_id") is None


def test_select_failures_keeps_non_pass_rows_once():
    dq = pl.DataFrame({"check_id": ["a", "b", "c", "c", "d"], "result": ["PASS", "FAIL", "WARN", "WARN", "INFO"]})
    assert A0.select_failures(dq, ICFG)["check_id"].to_list() == ["b", "c"]


# ----------------------------------------------------------------------------- end-to-end (synthetic warehouse)
DQ_COLS = ["check_id", "check", "table", "columns", "result", "severity", "metric", "value", "denominator",
           "threshold", "details", "recommended_action", "mode", "seconds", "sql"]


def _row(check_id, check, table, columns, result, severity, metric, threshold, mode="sample 2.0% (finds issues, cannot prove absence)"):
    return {"check_id": check_id, "check": check, "table": table, "columns": columns, "result": result,
            "severity": severity, "metric": metric, "value": "1.0", "denominator": "10.0", "threshold": threshold,
            "details": "planted", "recommended_action": "phase 01 advice", "mode": mode, "seconds": "0.01", "sql": ""}


def _plant(cfg) -> None:
    """BALL rows that carry the athlete's nfl_id (as in the real Combine file) with an implausible speed."""
    p = cfg.paths.data_root / "combine_tracking.csv"
    df = pl.read_csv(p, infer_schema_length=0, null_values=["NA"])
    ball = df.filter(pl.col("nfl_id") == "101").with_columns(pl.lit("BALL").alias("entity_type"), pl.lit("20.0").alias("s"))
    pl.concat([df, ball]).write_csv(p, null_value="NA")
    rows = [
        _row("uniq:combine_tracking:event_id, nfl_id, attempt, time", "Key uniqueness", "combine_tracking",
             "event_id, nfl_id, attempt, time", "FAIL", "CRITICAL", "duplicate_key_rows", "0"),
        _row("range:combine_tracking:s", "Value range", "combine_tracking", "s", "WARN", "MEDIUM", "out_of_range_values",
             "[0, 12.5]"),
        _row("fk:game_tracking:game_id,play_id->player_play", "Referential integrity", "game_tracking",
             "(game_id,play_id) → player_play(game_id,play_id)", "FAIL", "HIGH", "orphan_distinct_keys", "0"),
        _row("trk:game:n_jumps", "Trajectory: implausible jumps", "game_tracking", "n_jumps", "WARN", "MEDIUM",
             "n_jumps", "> 1.5 yd/frame", mode="sample 2.0% of trajectories"),
        _row("foo:combine_tracking:x", "Unregistered", "combine_tracking", "x", "FAIL", "LOW", "m", "0", mode="exact"),
        _row("uniq:players:nfl_id", "Key uniqueness", "players", "nfl_id", "PASS", "CRITICAL", "duplicate_key_rows", "0",
             mode="exact"),
    ]
    out = cfg.paths.tables / "data_quality_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={c: pl.String for c in DQ_COLS}).write_csv(out)


def test_drilldown_end_to_end(synth):
    from nflbdb.data.bronze import BronzeIngestor
    from nflbdb.data.silver import SilverBuilder
    from nflbdb.research.studies.store import StudyStore

    _plant(synth)
    BronzeIngestor(synth).ingest()
    SilverBuilder(synth).build_all()
    study = A0.DqFailureStudy(synth, store=StudyStore(synth))
    res = study.compute()
    t = conform(res[A0.DQ_FAILURES.name], A0.DQ_FAILURES)
    by = {r["check_id"]: r for r in t.iter_rows(named=True)}
    assert "uniq:players:nfl_id" not in by                                   # PASS rows are not drilled down

    u = by["uniq:combine_tracking:event_id, nfl_id, attempt, time"]
    assert u["classification"] == "rule_error" and u["resolution"] == "repair_rule"
    assert u["n_affected_rows"] == 2 * 16 and u["n_affected_trajectories"] == 2 and u["n_affected_athletes"] == 1
    assert u["phase01_sampled"] and u["drilldown_mode"] == "exact" and u["resolved_downstream"] == "yes"
    assert not u["blocking"]

    r = by["range:combine_tracking:s"]
    assert r["classification"] == "rule_error" and r["n_affected_rows"] == 0           # no PLAYER row out of range
    assert "of which athlete rows=0" in r["evidence"]

    assert by["fk:game_tracking:game_id,play_id->player_play"]["classification"] == "not_reproduced"
    assert by["trk:game:n_jumps"]["drilldown_mode"] == "not_run"                       # M1 not built in this test
    assert by["foo:combine_tracking:x"]["classification"] == "unreviewed"
    assert by["foo:combine_tracking:x"]["recommendation"] == "phase 01 advice"

    bd = conform(res[A0.BREAKDOWN.name], A0.BREAKDOWN)
    kinds = bd.filter(pl.col("check_id") == "range:combine_tracking:s")
    assert kinds.filter(pl.col("level") == "BALL")["n_rows"].item() == 16
    ex = conform(res[A0.EXAMPLES.name], A0.EXAMPLES)
    assert ex.filter(pl.col("check_id").str.starts_with("uniq:")).height > 0
