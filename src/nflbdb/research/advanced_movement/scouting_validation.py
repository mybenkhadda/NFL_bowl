"""Synthetic validation of Milestone 08 (notebook 08A so far), callable from a notebook cell. NumPy / Polars; no PyTorch,
no project data. Every fixture is generated here with a planted, known answer; tolerances were set by reasoning (the
author executed nothing) and each assertion states what it compares.

    from nflbdb.research.advanced_movement import scouting_validation as sv
    report = sv.run_all(sv.CHECKS_08A); display(report); sv.assert_all_passed(report)

A failing check is reported with its file, line and source text; it is never weakened to pass.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import polars as pl

from . import opportunity_validation as ov
from . import outcome_semantics as osm
from . import scouting_feasibility as sf
from .nfl_outcomes import CombineLeakError, assert_no_combine_columns
from .validation import _raises, assert_all_passed, failure_location

POSITIONS = ("WR", "DB", "DL")
FC = sf.FeasibilityConfig(min_obs=20, min_games=4, min_half_obs=8, min_athletes=30, min_athletes_position=10, min_athletes_halves=30,
                          n_boot=100, seed=7)


# ============================================================================== fixtures
def cand(oid: str, field: str, **kw) -> sf.Candidate:
    base = dict(outcome_id=oid, label=oid, family="test", role="candidate", value_field=field, kind="continuous",
                required_fields=(field,) if field else ("x_trait",), positions=POSITIONS, interpretability="movement",
                interpretation_limit="synthetic", leakage_risk="low", leakage_note="synthetic")
    return sf.Candidate(**{**base, **kw})


def fixture(n_athletes: int = 120, n_games: int = 8, per_game: int = 6, s2a: float = 1.0, s2w: float = 4.0,
            seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Balanced player-plays: ``x_trait`` = athlete effect (variance s2a) + position offset + window noise (s2w);
    ``x_noise`` has no athlete effect; ``x_rare`` exists for the first 6 athletes only; ``x_flag`` is a string flag."""
    rng = np.random.default_rng(seed)
    n = n_athletes * n_games * per_game
    nfl_id = np.repeat(np.arange(n_athletes), n_games * per_game)
    game = np.tile(np.repeat(np.arange(n_games), per_game), n_athletes)
    pos = nfl_id % len(POSITIONS)
    effect = rng.normal(0.0, np.sqrt(s2a), n_athletes)
    keys = {"game_id": 1000 + game, "play_id": np.arange(n), "nfl_id": nfl_id}
    plays = pl.DataFrame({**keys, "season": np.full(n, 2024), "week": game + 1, "position_group": np.array(POSITIONS)[pos],
                          "eligible": np.ones(n, bool)}).cast({k: pl.Int64 for k in ("game_id", "play_id", "nfl_id", "season", "week")})
    trait = effect[nfl_id] + 5.0 * pos + rng.normal(0.0, np.sqrt(s2w), n)
    pp = pl.DataFrame({**keys, "x_trait": trait, "x_noise": rng.normal(0.0, 1.0, n),
                       "x_rare": np.where(nfl_id < 6, 1.0, np.nan), "x_flag": np.where(rng.random(n) < 0.3, "TRUE", "false")})
    return plays, pp.with_columns(pl.col("x_rare").fill_nan(None))


def schemas_of(pp: pl.DataFrame) -> dict[str, dict[str, str]]:
    return {"player_play": {k: str(v) for k, v in pp.schema.items()}}


# ============================================================================== checks
def check_candidate_registry_contract() -> str:
    ids = [c.outcome_id for c in sf.CANDIDATES]
    assert len(ids) == len(set(ids)), "candidate identifiers are not unique"
    for c in sf.CANDIDATES:
        assert c.role in sf.ROLES and c.kind in sf.KINDS, f"{c.outcome_id}: role {c.role!r} / kind {c.kind!r}"
        assert c.required_fields and c.positions and c.interpretation_limit and c.leakage_note, f"{c.outcome_id}: an empty declaration"
        assert c.leakage_risk in ("low", "medium", "high"), f"{c.outcome_id}: leakage risk {c.leakage_risk!r}"
        assert not (c.needs_assignment and c.value_field), f"{c.outcome_id}: an outcome that needs a pairing cannot name a value field"
        assert not c.value_field or c.value_field in c.required_fields, f"{c.outcome_id}: value field not among the required fields"
    assert sf.registry_hash() == sf.registry_hash(tuple(sf.CANDIDATES)) != sf.registry_hash(sf.CANDIDATES[:-1]), "hash does not follow the declarations"
    return f"{len(ids)} candidates: unique identifiers, complete declarations, hash follows the declarations"


def check_flag_and_null_parsing() -> str:
    df = pl.DataFrame({"x_flag": ["TRUE", "false", " Yes ", "0", "maybe", None, ""], "x_num": [1.0, 0.0, None, float("nan"), 2.0, 0.0, 1.0],
                       "x_cat": ["GO", "", None, " ", "SLANT", "GO", "OUT"]})
    got = df.select(sf.value_expr(cand("T.f", "x_flag", kind="flag"), pl.String).alias("f"),
                    sf.value_expr(cand("T.n", "x_num", kind="flag"), pl.Float64).alias("n"),
                    sf.value_expr(cand("T.c", "x_num"), pl.Float64).alias("c"),
                    sf.value_expr(cand("T.k", "x_cat", kind="category"), pl.String).alias("k"))
    assert got["f"].to_list() == [1.0, 0.0, 1.0, 0.0, None, None, None], f"string flag parsed as {got['f'].to_list()}"
    assert got["n"].to_list() == [1.0, 0.0, None, None, 1.0, 0.0, 1.0], f"numeric flag parsed as {got['n'].to_list()}"
    assert got["c"].to_list() == [1.0, 0.0, None, None, 2.0, 0.0, 1.0], f"continuous value parsed as {got['c'].to_list()}"
    assert got["k"].to_list() == [1.0, None, None, None, 1.0, 1.0, 1.0], f"category presence parsed as {got['k'].to_list()}"
    return "recognised flag tokens only; null, NaN, empty and unknown tokens stay unobserved (never 0)"


def check_coverage_counts() -> str:
    plays, pp = fixture(n_athletes=30, s2w=1.0)
    plays = plays.with_columns((pl.col("week") > 1).alias("eligible"))                 # the first game is ineligible
    pp = pp.with_columns(pl.when(pl.col("nfl_id") < 24).then(pl.col("x_trait")).otherwise(None).alias("x_trait"))
    c = cand("T.cov", "x_trait", positions=("WR", "DB"))
    base = sf.observation_base(plays, pp, [c])
    assert base.height == plays.height and "x_noise" not in base.columns, "the base must keep every player-play and only the wanted fields"
    cov = {r["position_group"]: r for r in sf.coverage_table(base, [c], FC).iter_rows(named=True)}
    want = {sf.ALL: (30, 1260, 1008, 24, 24, 168), "WR": (10, 420, 336, 8, 8, 56), sf.DECLARED: (20, 840, 672, 16, 16, 112)}
    for g, w in want.items():
        got = tuple(cov[g][k] for k in ("n_athletes_tracked", "n_eligible_plays", "n_obs", "n_athletes_any", "n_athletes_ge_min", "n_athlete_games"))
        assert got == w, f"group {g}: observed {got}, expected {w}"
        assert cov[g]["median_obs"] == 42 and cov[g]["median_games"] == 7 and abs(cov[g]["opportunity_rate"] - 0.8) < 1e-12, f"group {g}: {cov[g]}"
    assert cov["DL"]["declared_position"] is False and cov["WR"]["declared_position"] is True, "declared-position flag"
    strict = sf.coverage_table(base, [c], sf.FeasibilityConfig(min_obs=43, min_games=4))
    assert strict["n_athletes_ge_min"].sum() == 0, "42 observations per athlete cannot satisfy a threshold of 43"
    return "athletes, eligible plays, observations, athlete-games and opportunity rate match the planted counts (ALL / position / DECLARED)"


def check_split_half_reliability() -> str:
    plays, pp = fixture(n_athletes=300, n_games=8, per_game=6, s2a=1.0, s2w=4.0, seed=1)
    cs = [cand("T.trait", "x_trait"), cand("T.noise", "x_noise")]
    rel = {r["outcome_id"]: r for r in sf.reliability_table(sf.observation_base(plays, pp, cs), cs, FC).iter_rows(named=True)}
    expected_r = 1.0 / (1.0 + 4.0 / 24)                       # each game half holds 4 games × 6 windows
    t, z = rel["T.trait"], rel["T.noise"]
    assert t["n_athletes_both_halves"] == 300 and t["n_positions"] == 3, f"split-half cohort {t}"
    assert abs(t["r_half"] - expected_r) < 0.06, f"half correlation {t['r_half']:.3f}, closed form {expected_r:.3f}, tolerance 0.06"
    assert abs(t["rel_sb"] - 2 * expected_r / (1 + expected_r)) < 0.05 and t["rel_lo"] < t["rel_sb"] < t["rel_hi"], f"Spearman–Brown {t}"
    assert abs(z["r_half"]) < 0.20, f"no athlete effect planted, half correlation {z['r_half']:.3f} (|r| < 0.20 expected at n = 300)"
    few = sf.reliability_table(sf.observation_base(plays.filter(pl.col("nfl_id") < 2), pp, cs), cs, FC)
    assert few["rel_sb"].null_count() == 2 and all(few["note"].to_list()), "fewer than 3 athletes must give no estimate and a note"
    return (f"planted reliability recovered (r {t['r_half']:.3f} vs {expected_r:.3f}); position offsets do not inflate it; "
            f"no-trait r {z['r_half']:.3f}; no estimate below 3 athletes")


def check_status_rule_and_gate() -> str:
    plays, pp = fixture(n_athletes=120, s2a=1.0, s2w=1.0, seed=2)
    cs = [cand("T.good", "x_trait"), cand("T.noise", "x_noise"), cand("T.cond", "x_trait", outcome_conditioned=True),
          cand("T.ctx", "x_trait", role="context"), cand("T.pair", "", needs_assignment=True), cand("T.missing", "x_absent"),
          cand("T.blocked", "x_trait"), cand("T.rare", "x_rare"), cand("T.model", "x_trait", interpretability="model_output"),
          cand("T.career", "x_trait", role="career")]
    out = sf.run_audit(sf.observation_base(plays, pp, cs), schemas_of(pp), cs, FC, {"T.blocked": "unresolved"})
    reg = {r["outcome_id"]: r for r in out[sf.REGISTRY.name].iter_rows(named=True)}
    want = {"T.good": "observable", "T.noise": "observable", "T.cond": "observable_conditional", "T.ctx": "context_only",
            "T.pair": "unsupported_requires_inference", "T.missing": "unsupported_missing_field", "T.blocked": "blocked_definition",
            "T.rare": "insufficient_coverage", "T.model": "not_interpretable", "T.career": "not_movement_outcome"}
    got = {k: reg[k]["status"] for k in want}
    assert got == want, f"statuses {got}"
    assert reg["T.missing"]["missing_fields"] == "x_absent" and "x_absent" not in sf.observation_base(plays, pp, cs).columns, "a missing field must stay missing"
    audit = {r["field"]: r["present"] for r in out[sf.FIELD_AUDIT.name].iter_rows(named=True)}
    assert audit["x_absent"] is False and audit["x_trait"] is True, f"field audit {audit}"
    gate = sf.gate_08a(out[sf.REGISTRY.name], cs)
    assert gate["gate"] == "PASS" and gate["defensible"] == ["T.good"] and gate["branch"] == "continue", f"gate {gate}"
    rest = [c for c in cs if c.outcome_id != "T.good"]
    stop = sf.gate_08a(out[sf.REGISTRY.name].filter(pl.col("outcome_id") != "T.good"), rest)
    assert stop["gate"] == "PASS" and stop["branch"] == "stop_branch" and not stop["defensible"], f"gate without a defensible outcome {stop}"
    assert sf.gate_08a(out[sf.REGISTRY.name], rest)["gate"] == "FAIL", "a registry that does not match the declared candidates must fail"
    assert "stops" in sf.conclusions_markdown(out[sf.REGISTRY.name].filter(pl.col("outcome_id") != "T.good"), stop, FC), "the stop must be written"
    return "ordered status rule, missing fields never created, gate PASS with branch continue / stop_branch, incomplete registry FAIL"


def check_no_combine_leak_and_events() -> str:
    plays, pp = fixture(n_athletes=6)
    assert _raises(CombineLeakError, lambda: assert_no_combine_columns(plays.with_columns(pl.lit(4.4).alias("cmb_forty")), "base")), "a Combine column must be refused"
    dup = sf.observation_base(plays, pl.concat([pp, pp]), [cand("T.x", "x_trait")])
    assert dup.height == plays.height and dup["x_trait"].null_count() == 0, "duplicated vendor rows must not multiply the player-plays"
    ev =sf.tracking_event_audit(pl.DataFrame({"game_id": [1, 1, 2, 2], "event": ["ball_snap", None, "ball_snap", "pass_forward"]}).lazy())
    assert dict(zip(ev["event"].to_list(), ev["n_frames"].to_list())) == {"ball_snap": 2, "‹null›": 1, "pass_forward": 1}, f"event inventory {ev}"
    assert sf.tracking_event_audit(pl.DataFrame({"game_id": [1]}).lazy())["n_frames"].to_list() == [0], "an absent event column must be reported, not invented"
    return "Combine-named columns are refused; the event inventory counts frames and reports an absent column"


# ============================================================================== 08A.2 fixtures
FB_POS = ("WR", "TE", "DB", "DL")
VC = ov.ValidationConfig(min_obs=20, min_games=4, min_half_obs=8, min_cell_obs=30, min_cell_athletes=5, n_boot=100, n_random_splits=20,
                         n_boot_games=50, first_games_grid=(2, 4), min_games_grid=(4, 8), min_linked=10, min_linked_position=3, seed=7)


def football_fixture(n_athletes: int = 80, n_games: int = 8, per_game: int = 6, s2a: float = 0.25, s2g: float = 0.0, s2w: float = 0.25,
                     seed: int = 0, full: bool = False, uneven: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, np.ndarray]]:
    """Tracked player-plays + vendor rows with DETERMINISTIC recording patterns (index ``j`` of the row inside the athlete):
    separation of receivers = 6 + athlete effect + game effect + noise, recorded unless j % 5 == 0 (``full``: always);
    route label unless j % 7 == 1; target when j % 4 == 0; tackle statistic of defenders when j % 6 == 0 (1 when
    j % 12 == 0, else 0 with assist 1); sack of DL recorded when j % 2 == 0 (1, 0.5 or 0); ``uneven`` removes the last
    game of half of the athletes. Returns (plays, vendor rows, the planted arrays)."""
    rng = np.random.default_rng(seed)
    m = n_games * per_game
    n = n_athletes * m
    nfl_id, j = np.repeat(np.arange(n_athletes), m), np.tile(np.arange(m), n_athletes)
    game = np.tile(np.repeat(np.arange(n_games), per_game), n_athletes)
    pos = np.array(FB_POS)[nfl_id % 4]
    rec, dfn, dl = np.isin(pos, ("WR", "TE")), np.isin(pos, ("DB", "DL")), pos == "DL"
    geff = rng.normal(0.0, np.sqrt(s2g), (n_athletes, n_games)) if s2g > 0 else np.zeros((n_athletes, n_games))
    sep = 6.0 + rng.normal(0.0, np.sqrt(s2a), n_athletes)[nfl_id] + geff[nfl_id, game] + rng.normal(0.0, np.sqrt(s2w), n)
    eligible = ~((game == n_games - 1) & ((nfl_id // 4) % 2 == 0)) if uneven else np.ones(n, bool)
    a = {"nfl_id": nfl_id, "j": j, "game": game, "pos": pos, "sep": sep, "eligible": eligible, "sep_obs": rec & ((j % 5 != 0) | full),
         "route_obs": rec & (j % 7 != 1), "target": j % 4 == 0, "tkl_obs": dfn & (j % 6 == 0), "sck_obs": dl & (j % 2 == 0),
         "tackle": (j % 12 == 0).astype(np.int64), "sack": np.where(j % 24 == 0, 1.0, np.where(j % 24 == 12, 0.5, 0.0)),
         "cov_obs": dfn & (j % 2 == 0)}
    keys = {"game_id": 1000 + game, "play_id": np.arange(n), "nfl_id": nfl_id}
    ints = {k: pl.Int64 for k in ("game_id", "play_id", "nfl_id", "season", "week", "cohort", "down")}
    plays = pl.DataFrame({**keys, "season": np.full(n, 2024), "week": game + 1, "position_group": pos.tolist(), "cohort": np.full(n, 2022),
                          "side": np.where(rec, "offense", "defense").tolist(),
                          "lined_up_position": np.where(rec, np.where(j % 2 == 1, "WR", "SLOT"), pos).tolist(),
                          "play_class": np.where(j % 2 == 0, "pass", "no_pass").tolist(),
                          "pass_result": np.where(a["sck_obs"] & (a["sack"] > 0), "S", "C").tolist(), "down": np.ones(n, np.int64),
                          "eligible": eligible}).cast(ints)
    none_f, none_s = pl.lit(None, dtype=pl.Float64), pl.lit(None, dtype=pl.String)
    raw = pl.DataFrame({**keys, **{k: a[k] for k in ("j", "sep", "sep_obs", "route_obs", "target", "tkl_obs", "sck_obs", "tackle", "sack", "cov_obs")}})
    pp = raw.select(
        *[pl.col(k).cast(pl.Int64) for k in keys],
        pl.when(pl.col("sep_obs")).then(pl.col("sep")).otherwise(none_f).alias("separation_at_pass_forward"),
        pl.when(pl.col("route_obs")).then(pl.when(pl.col("j") % 2 == 1).then(pl.lit("GO")).otherwise(pl.lit("SLANT"))).otherwise(none_s).alias("route_ran"),
        pl.col("target"),
        pl.when(pl.col("tkl_obs")).then(pl.col("tackle")).otherwise(pl.lit(None, dtype=pl.Int64)).alias("tackle"),
        pl.when(pl.col("tkl_obs")).then((1 - pl.col("tackle")).cast(pl.Float64)).otherwise(none_f).alias("assist"),
        pl.when(pl.col("sck_obs")).then(pl.col("sack")).otherwise(none_f).alias("sack"),
        pl.when(pl.col("cov_obs")).then(pl.when(pl.col("j") % 4 == 0).then(pl.lit("NO_ASSIGNMENT")).otherwise(pl.lit("MAN"))).otherwise(none_s)
        .alias("coverage_assignment"))
    return plays, pp, a


def manifest_fixture(n_athletes: int) -> pl.DataFrame:
    """Combine cohort manifest (flags only): athletes with id % 3 == 0 are NOT in it."""
    i = np.arange(n_athletes)
    i = i[i % 3 != 0]
    return pl.DataFrame({"nfl_id": i, "draft_year": 2022 + (i % 2), "n_attempts": np.where(i % 2 == 0, 2, 1), "n_speed_attempts": np.ones(len(i), np.int64),
                         "st_official": i % 5 != 0, "st_tracked": np.ones(len(i), bool), "st_engineered": np.ones(len(i), bool),
                         "st_fpca": np.ones(len(i), bool), "st_tcn": i % 7 != 0}).cast({"nfl_id": pl.Int64, "draft_year": pl.Int64, "n_attempts": pl.Int64})


def _semantics(plays: pl.DataFrame, pp: pl.DataFrame, evidence: dict | None = None) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    base, dup = osm.semantic_base(plays, pp)
    checks = osm.semantic_checks(base, dup, osm.per_play_credit(pp, ("tackle", "assist", "sack")), osm.tracked_per_play(plays))
    return base, checks, osm.semantic_status(checks, evidence)


def _fake_store(folder: str, run_mode: str = "full") -> SimpleNamespace:
    return SimpleNamespace(data_dir=Path(folder), controls=SimpleNamespace(save_tables=True, run_mode=run_mode))


# ============================================================================== 08A.2 checks
def check_outcome_semantics_status() -> str:
    plays, pp, _ = football_fixture()
    _, checks, status = _semantics(plays, pp)
    assert {v["status"] for v in status.values()} == {"plausible_inferred"}, f"a column name must not confirm a definition: {status}"
    assert checks.filter(pl.col("blocking") & ~pl.col("passed")).is_empty(), f"clean fixture, blocking failures: {checks.filter(pl.col('blocking') & ~pl.col('passed'))}"
    assert _raises(ValueError, lambda: _semantics(plays, pp, {osm.FO01: {"status": "confirmed_source"}})), "confirmation without a source and a quoted definition must be refused"
    ev = {osm.FO01: {"status": "confirmed_source", "source": "data dictionary p. 3", "definition": "distance to the nearest defender"}}
    confirmed = _semantics(plays, pp, ev)[2]
    assert confirmed[osm.FO01]["status"] == "confirmed_source" and confirmed[osm.FO13]["status"] == "plausible_inferred", f"evidence applies to its outcome only: {confirmed}"
    neg = _semantics(plays, pp.with_columns(-pl.col("separation_at_pass_forward")), ev)[2]
    assert neg[osm.FO01]["status"] == "unresolved", "a negative distance must make the definition unresolved, whatever the recorded evidence"
    bad = _semantics(plays, pp.with_columns(pl.when(pl.col("sack") == 0.5).then(2.0).otherwise(pl.col("sack")).alias("sack")))[2]
    assert bad[osm.FO13]["status"] == "unresolved" and bad[osm.FO12]["status"] == "plausible_inferred", f"an undeclared credit value blocks its outcome only: {bad}"
    gone = _semantics(plays, pp.drop("tackle"))[2]
    assert gone[osm.FO12]["status"] == "unresolved", "an absent value field cannot have a meaning"
    return "inferred by default; confirmed only with a named source and quoted definition; blocking failures give unresolved"


def check_zero_versus_missing() -> str:
    plays, pp, a = football_fixture()
    base, _, status = _semantics(plays, pp)
    z = {(r["field"], r["position_group"]): r for r in osm.zero_missing_table(base).iter_rows(named=True)}
    db = a["pos"] == "DB"
    want = (int(db.sum()), int((db & ~a["tkl_obs"]).sum()), int((db & a["tkl_obs"] & (a["tackle"] == 0)).sum()), int((db & a["tkl_obs"] & (a["tackle"] == 1)).sum()))
    got = tuple(z[("tackle", "DB")][k] for k in ("n_rows", "n_null", "n_zero", "n_positive"))
    assert got == want, f"tackle of DB (rows, null, zero, positive): observed {got}, expected {want}"
    assert z[("tackle", "WR")]["n_null"] == z[("tackle", "WR")]["n_rows"], "a field never recorded for a position is all null there, not zero"
    cand = {r["outcome_id"]: r for r in osm.candidates_table(base, {"player_play": {k: str(v) for k, v in pp.schema.items()}}, ["fixture"], status).iter_rows(named=True)}
    dl = a["pos"] == "DL"
    assert cand[osm.FO13]["n_zero"] == int((dl & a["sck_obs"] & (a["sack"] == 0)).sum()) and cand[osm.FO13]["n_observed"] == int((dl & a["sck_obs"]).sum()), f"sack zeros / recorded rows: {cand[osm.FO13]}"
    rows = ov.all_rows(base, VC)
    t0, t1 = float(rows["FO12.T0"]["value"].mean()), float(rows["FO12.T1"]["value"].mean())
    assert abs(t0 - 0.5) < 1e-12 and abs(t1 - 4 / 48) < 1e-12, f"recorded-row share {t0} (planted 0.5) and credits per snap {t1} (planted 4/48) are different quantities"
    assert rows["FO13.K0"].height == int((dl & a["sck_obs"]).sum()) and rows["FO13.K2"].height == int(dl.sum()), "K0 keeps recorded rows only; K2 states its null reading by keeping every snap"
    return "null, recorded zero and positive are counted apart; a null enters a rate only in the denominators that declare that reading"


def check_opportunity_denominators() -> str:
    plays, pp, a = football_fixture()
    base, _, _ = _semantics(plays, pp)
    rows = ov.all_rows(base, VC)
    rec, dl, dfn = np.isin(a["pos"], ("WR", "TE")), a["pos"] == "DL", np.isin(a["pos"], ("DB", "DL"))
    want = {"FO01.S0": a["sep_obs"], "FO01.S1": a["sep_obs"] & a["route_obs"], "FO01.S2": a["sep_obs"] & a["route_obs"],
            "FO01.S0T": a["sep_obs"] & a["target"], "FO01.S0N": a["sep_obs"] & ~a["target"], "FO12.T0": a["tkl_obs"], "FO12.T1": dfn,
            "FO12.T2": dfn, "FO13.K0": a["sck_obs"], "FO13.K1": a["sck_obs"] & (a["j"] % 4 == 0), "FO13.K2": dl}
    got = {k: v.height for k, v in rows.items()}
    assert got == {k: int(v.sum()) for k, v in want.items()}, f"rows per denominator: observed {got}, expected { {k: int(v.sum()) for k, v in want.items()} }"
    assert rec.sum() > 0 and all(set(rows[d.denominator_id]["position_group"].unique().to_list()) <= set(d.positions) for d in ov.DENOMINATORS), "a denominator holds its declared positions only"
    assert abs(float(rows["FO01.S0"]["value"].mean()) - float(a["sep"][a["sep_obs"]].mean())) < 1e-9, "S0 mean differs from the planted values"
    assert abs(float(rows["FO01.S2"]["value"].mean())) < 1e-9, "cell residuals of S2 must average zero over the supported cells"
    assert abs(float(rows["FO13.K0"]["value"].mean()) - 0.125) < 1e-12 and abs(float(rows["FO12.T2"]["value"].mean()) - 8 / 48) < 1e-12, "planted sack and tackle-or-assist rates"
    strict = ov.denominator_rows(base, ov.BY_ID["FO01.S2"], ov.ValidationConfig(min_cell_obs=10**9))
    assert strict is not None and strict.is_empty(), "an unsupported cell contributes no row (support is decided by counts)"
    base2, _, _ = _semantics(plays, pp.drop("assist"))
    out = ov.run_opportunity(base2, VC)
    den = {(r["denominator_id"], r["position_group"]): r for r in out[ov.DENOMS.name].iter_rows(named=True)}
    assert den[("FO12.T2", ov.ALL)]["available"] is False and "FO12.T2" not in out[ov.ATHLETES.name]["denominator_id"].unique().to_list(), "a denominator whose field is absent must not be computed"
    assert den[("FO01.S0", "WR")]["n_athletes_supported"] == 20 and abs(den[("FO01.S0", ov.ALL)]["targeted_share"] - float(a["target"][a["sep_obs"]].mean())) < 1e-12, f"support and targeted share: {den[('FO01.S0', 'WR')]}"
    return f"{len(rows)} denominators match the planted row counts and rates; absent fields give no denominator; positions are respected"


def check_duplication_and_join() -> str:
    plays, pp, a = football_fixture(n_athletes=16)
    base, dup = osm.semantic_base(plays, pl.concat([pp, pp]))
    assert base.height == plays.height and dup["n_duplicate_rows"] == pp.height and dup["n_conflicting_rows"] == 0, f"exact duplicates: {dup}"
    _, dup2 = osm.semantic_base(plays, pl.concat([pp, pp.with_columns(pl.col("separation_at_pass_forward") + 1.0)]))
    assert dup2["n_conflicting_rows"] == int(a["sep_obs"].sum()), f"conflicting duplicates {dup2['n_conflicting_rows']}, planted {int(a['sep_obs'].sum())}"
    assert osm.semantic_checks(base, dup2, osm.per_play_credit(pp, ("sack",)), {"median": 1.0, "p95": 1.0, "max": 1.0}).filter(
        (pl.col("check") == "one_row_per_player_play") & ~pl.col("passed")).height == 3, "conflicting duplicates must fail the blocking check of every outcome"
    assert "x_unknown" not in osm.semantic_base(plays, pp, ("x_unknown", "sack"))[0].columns, "an absent field is not created"
    return "duplicated vendor rows never multiply player-plays; conflicting duplicates are counted and block"


def check_athlete_aggregation() -> str:
    plays, pp, a = football_fixture(n_athletes=120, s2a=1.0, s2w=1.0, seed=3)
    base, _, _ = _semantics(plays, pp)
    rows = ov.denominator_rows(base, ov.BY_ID["FO01.S0"], VC)
    at = ov.athlete_table(ov.game_table(rows), ov.BY_ID["FO01.S0"], VC, 0)
    for i in (0, 1, 4, 117):
        m = (a["nfl_id"] == i) & a["sep_obs"]
        r = at.filter(pl.col("nfl_id") == i).row(0, named=True)
        assert abs(r["value"] - float(a["sep"][m].mean())) < 1e-9 and r["n_obs"] == int(m.sum()) and r["n_games"] == 8, f"athlete {i}: {r}"
    assert at.height == 60 and bool(at["supported"].all()), "every receiver has 38 or 39 rows in 8 games"
    inside = ((at["lo"] <= at["value"]) & (at["value"] <= at["hi"])).mean()
    assert inside >= 0.95 and float(at["se_game_boot"].min()) > 0, f"game-bootstrap interval contains the value for {inside:.2f} of athletes"
    pct = at.group_by("position_group").agg(pl.col("pct_position").mean().alias("m"), pl.col("pct_position").min().alias("lo"), pl.col("pct_position").max().alias("hi"))
    assert all(abs(m - 50) < 1e-9 and lo > 0 and hi < 100 for m, lo, hi in pct.select("m", "lo", "hi").iter_rows()), f"mid-rank percentiles within position: {pct}"
    few = ov.athlete_table(ov.game_table(rows.filter(pl.col("week") <= 2)), ov.BY_ID["FO01.S0"], VC, 0)
    assert not bool(few["supported"].any()) and few["pct_position"].null_count() == few.height, "two games do not support a value, and an unsupported athlete gets no percentile"
    return "athlete value = sum / count over whole games; interval from resampled games; support and percentiles by rule"


def check_game_split_reliability() -> str:
    plays, pp, _ = football_fixture(n_athletes=1200, s2a=1.0, s2g=1.0, s2w=4.0, seed=5, full=True)
    base, _ = osm.semantic_base(plays, pp)
    d = ov.BY_ID["FO01.S0"]
    ga = ov.game_arrays(ov.game_table(ov.denominator_rows(base, d, VC)))
    half = ov.random_half(ga, np.random.default_rng(0))
    sizes = np.bincount(ga.ai, weights=half == 0) - np.bincount(ga.ai, weights=half == 1)
    assert len(half) == len(ga.ai) == 600 * 8 and set(np.unique(half)) == {0, 1} and np.abs(sizes).max() <= 1, "a random split assigns WHOLE games, half to each side"
    rel = {(r["split"], r["position_group"]): r for r in ov.reliability_rows(d, ga, VC, 0)}
    expected = 1.0 / (1.0 + 1.0 / 4 + 4.0 / 24)                   # a half = 4 games × 6 rows: game effect / 4 + row noise / 24
    oe, rd = rel[("odd_even", ov.ALL)], rel[("random", ov.ALL)]
    assert oe["n_athletes"] == 600 and abs(oe["r_half"] - expected) < 0.06, f"odd / even r {oe['r_half']:.3f}, closed form {expected:.3f} (a row-level split would ignore the game effect and give {1 / (1 + 4 / 24):.3f})"
    assert abs(rd["r_half"] - expected) < 0.05 and oe["rel_lo"] < oe["rel_sb"] < oe["rel_hi"], f"random game halves r {rd['r_half']:.3f}; interval {oe}"
    assert rel[("odd_even", "WR")]["n_athletes"] == 300 and abs(rel[("odd_even", "TE")]["r_half"] - expected) < 0.09, "position-level rows"
    plays0, pp0, _ = football_fixture(n_athletes=1200, s2a=0.0, s2g=1.0, s2w=4.0, seed=6, full=True)
    ga0 = ov.game_arrays(ov.game_table(ov.denominator_rows(osm.semantic_base(plays0, pp0)[0], d, VC)))
    r0 = {(r["split"], r["position_group"]): r for r in ov.reliability_rows(d, ga0, VC, 0)}[("odd_even", ov.ALL)]["r_half"]
    assert abs(r0) < 0.15, f"no athlete effect planted: r {r0:.3f} (a game effect alone must not look like reliability)"
    expo = {(r["mode"], r["level"]): r for r in ov.exposure_rows(d, ga, VC)}
    assert expo[("first_n_vs_later", 2)]["n_athletes"] == 600 and expo[("first_n_vs_later", 2)]["r"] < expo[("first_n_vs_later", 4)]["r"] + 0.03, "more early games must not agree worse with the later games (closed form 0.653 vs 0.706; margin 0.03)"
    return f"planted game-split reliability recovered (odd / even {oe['r_half']:.3f}, random {rd['r_half']:.3f}, closed form {expected:.3f}); no-trait r {r0:.3f}"


def check_cohort_linking() -> str:
    plays, pp, _ = football_fixture(n_athletes=120)
    base, _, _ = _semantics(plays, pp)
    athletes = ov.run_opportunity(base, VC)[ov.ATHLETES.name]
    out = ov.run_links(athletes, manifest_fixture(120))
    assert_no_combine_columns(out[ov.LINKS.name], "links")
    fun = {(r["denominator_id"], r["position_group"], r["stage"]): r["n_athletes"] for r in out[ov.FUNNEL.name].iter_rows(named=True)}
    i = np.arange(120)
    rec = i % 4 < 2
    linked, off, tcn = rec & (i % 3 != 0), rec & (i % 3 != 0) & (i % 5 != 0), rec & (i % 3 != 0) & (i % 5 != 0) & (i % 7 != 0)
    want = {"nfl_observed": int(rec.sum()), "nfl_supported": int(rec.sum()), "combine_linked": int(linked.sum()), "official_tests": int(off.sum()),
            "valid_tracking": int(off.sum()), "all_model_families": int(tcn.sum()), "repeat_attempts": int((off & (i % 2 == 0)).sum()),
            "eligible_M5_tcn": int(tcn.sum()), "eligible_M4_fpca": int(off.sum())}
    got = {k: fun[("FO01.S0", ov.ALL, k)] for k in want}
    assert got == want, f"funnel of FO01.S0: observed {got}, expected {want}"
    assert fun[("FO13.K0", ov.ALL, "nfl_observed")] == 30 != fun[("FO01.S0", ov.ALL, "nfl_observed")], "each outcome has its own population"
    assert fun[("FO01.S0", "WR", "combine_linked")] + fun[("FO01.S0", "TE", "combine_linked")] == want["combine_linked"], "positions add up to ALL"
    cum = out[ov.FUNNEL.name].filter(pl.col("cumulative")).sort("denominator_id", "position_group", "stage_order")
    assert all(np.all(np.diff(t["n_athletes"].to_numpy()) <= 0) for _, t in cum.group_by(["denominator_id", "position_group"])), "cumulative stages cannot grow"
    un = out[ov.LINKS.name].filter(~pl.col("linked"))
    assert un.height and un["draft_year"].null_count() == un.height and not bool(un["elig_M1"].any()), "an athlete absent from the manifest is not linked and joins no model cohort"
    sup = out[ov.SUPPORT.name].filter(pl.col("denominator_id") == "FO01.S0")
    assert int(sup["n_supported_linked"].sum()) == want["combine_linked"] and int(sup["n_all_models"].sum()) == want["all_model_families"], "position × cohort support adds up"
    return "linked-cohort funnel, model-family eligibility and position × cohort support match the planted flags"


def _screened(semantic_status: str = "plausible_inferred", cfg: ov.ValidationConfig = VC) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    plays, pp, _ = football_fixture(n_athletes=400, s2a=1.0, s2w=1.0, seed=8, uneven=True)
    base, _, _ = _semantics(plays, pp)
    opp = ov.run_opportunity(base, cfg)
    links = ov.run_links(opp[ov.ATHLETES.name], manifest_fixture(400))
    semantic = {o: {"status": semantic_status, "basis": "fixture"} for o in (osm.FO01, osm.FO12, osm.FO13)}
    screen = ov.screen_table(opp[ov.DENOMS.name], semantic, opp[ov.RELIABILITY.name], opp[ov.DEPENDENCE.name], links[ov.FUNNEL.name], cfg)
    return screen, opp[ov.ATHLETES.name], semantic


def check_screen_and_proposal() -> str:
    def status(screen: pl.DataFrame) -> dict[str, str]:
        return dict(screen.select("denominator_id", "status").iter_rows())

    inferred, confirmed, unresolved = _screened()[0], _screened("confirmed_source")[0], _screened("unresolved")[0]
    si, sc, su = status(inferred), status(confirmed), status(unresolved)
    assert si["FO01.S0"] == "provisional" and sc["FO01.S0"] == "eligible", f"inferred semantics cap the status at provisional: {si['FO01.S0']} / {sc['FO01.S0']}"
    assert all(sc[k] == "sensitivity_only" for k in ("FO01.S0T", "FO01.S0N", "FO12.T0", "FO13.K1")), f"selected / credited-only / inferred denominators are never targets: {sc}"
    assert sc["FO12.T1"] != "eligible" and sc["FO13.K2"] != "eligible", "a structural denominator (null read as no credit) cannot be eligible"
    assert su["FO01.S0"] == "rejected" and set(su.values()) <= {"rejected", "sensitivity_only", "unavailable"}, f"unresolved semantics reject: {su}"
    assert status(_screened("confirmed_source", ov.ValidationConfig(**{**VC.__dict__, "min_linked": 10**6}))[0])["FO01.S0"] == "rejected", "too small a linked cohort must reject"
    pi, pc, pu = ov.propose(inferred), ov.propose(confirmed), ov.propose(unresolved)
    assert (pc["decision"], pc["primary"]) == ("FREEZE", "FO01.S0") and (pi["decision"], pi["primary"]) == ("PROVISIONAL", "FO01.S0"), f"proposals {pc} / {pi}"
    assert pu["decision"] == "STOP" and pu["primary"] is None, f"nothing eligible or provisional must propose STOP: {pu}"
    assert all(inferred.filter(pl.col("denominator_id") == k)["outcome_id"][0] != osm.FO01 for k in pc["secondary"]), "a secondary is another outcome"
    assert _raises(CombineLeakError, lambda: ov.screen_table(pl.DataFrame({"cmb_forty": [4.4]}), {}, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), VC)), "a Combine column in a screen input must be refused"
    return "screen statuses follow semantics, denominator legitimacy, linked support and reliability; the proposal follows the pre-declared order"


def check_explicit_freeze() -> str:
    screen, athletes, semantic = _screened()
    proposal = ov.propose(screen)
    args = dict(screen=screen, proposal=proposal, athletes=athletes, semantic=semantic, cfg=VC, provenance={"source": "fixture"})
    with tempfile.TemporaryDirectory() as tmp:
        store, v1 = _fake_store(tmp), ov.registry_path(_fake_store(tmp), "v1")
        doc, msg = ov.freeze_football_targets(store, None, None, [], "", False, **args)
        assert doc is None and "PENDING" in msg and not v1.is_file() and ov.gate_08a2(doc, None)[0] == "PENDING", "no decision writes nothing"
        assert _raises(ov.FootballTargetFreezeError, lambda: ov.require_frozen(store)), "08B must be refused without a registry"
        assert _raises(ValueError, lambda: ov.freeze_football_targets(store, "FREEZE", "FO01.S0", [], "", True, **args)), "a decision needs a rationale"
        assert _raises(ValueError, lambda: ov.freeze_football_targets(store, "FREEZE", "FO01.S0", [], "reviewed", False, **args)), "a provisional target needs an explicit acknowledgement"
        assert _raises(ValueError, lambda: ov.freeze_football_targets(store, "FREEZE", "FO01.S0T", [], "reviewed", True, **args)), "a sensitivity-only denominator cannot be frozen"
        assert _raises(ValueError, lambda: ov.freeze_football_targets(store, "FREEZE", "FO01.S0", ["FO01.S1"], "reviewed", True, **args)), "two targets of one outcome must be refused"
        assert _raises(ValueError, lambda: ov.freeze_football_targets(store, "AUTO", "FO01.S0", [], "reviewed", True, **args)), "unknown decision"
        for decision in ("PROVISIONAL", "STOP"):
            doc, msg = ov.freeze_football_targets(store, decision, "FO01.S0", [], "reviewed", False, **args)
            assert doc is None and not v1.is_file() and ov.gate_08a2(doc, decision)[0] == decision, f"{decision} must not write a registry"
        assert _raises(ov.FootballTargetFreezeError, lambda: ov.require_frozen(store)), "PROVISIONAL / STOP keep 08B blocked"
        doc, msg = ov.freeze_football_targets(store, "FREEZE", "FO01.S0", [], "reviewed: definition inferred, accepted as exploratory", True, **args)
        assert msg == "frozen now" and v1.is_file() and ov.gate_08a2(doc, "FREEZE")[0] == "PASS", f"freeze: {msg}"
        t = ov.require_frozen(store)["targets"][0]
        need = ("target_id", "formula", "units", "opportunity_denominator", "support_rule", "semantic_status", "reliability", "support",
                "primary_metric", "definition_hash", "cohort_hash")
        assert all(t.get(k) not in (None, "") for k in need) and t["role"] == "primary" and t["semantic_status"] == "plausible_inferred", f"target record {t}"
        assert all(doc.get(k) for k in ("version", "version_hash", "config_hash", "denominators_hash", "semantics_hash", "seed", "run_mode", "provenance")), "provenance fields"
        assert doc["selection"]["combine_values_consulted"] is False and "exploratory" in doc["evidence_family"], "the freeze states what was consulted"
        again, msg2 = ov.freeze_football_targets(store, "FREEZE", "FO01.S0", [], "reviewed again", True, **args)
        assert again["frozen_at"] == doc["frozen_at"] and "already frozen" in msg2, "a reproduced freeze is not rewritten"
        assert _raises(ov.FreezeConflictError, lambda: ov.freeze_football_targets(store, "FREEZE", "FO01.S1", [], "changed", True, **args)), "another definition under the same version must be refused"
        doc2, _ = ov.freeze_football_targets(store, "FREEZE", "FO01.S1", [], "changed", True, version="v2", **args)
        assert ov.require_frozen(store, "v2")["targets"][0]["target_id"] == "FO01.S1" and ov.require_frozen(store)["version_hash"] == doc["version_hash"], "a new version leaves the old registry intact"
        assert doc2["targets"][0]["definition_hash"] != t["definition_hash"], "the definition hash follows the definition"
        smaller = ov.target_definition("FO01.S0", "primary", screen.filter(pl.col("denominator_id") == "FO01.S0").row(0, named=True), semantic,
                                       athletes.filter(pl.col("nfl_id") > 7), VC)
        assert smaller["cohort_hash"] != t["cohort_hash"] and smaller["definition_hash"] == t["definition_hash"], "the cohort hash follows the athletes"
        assert _raises(ov.FootballTargetFreezeError, lambda: ov.require_frozen(_fake_store(tmp, "smoke"))), "a registry of another run mode is refused"
        edited = json.loads(v1.read_text(encoding="utf-8"))
        edited["targets"][0]["cohort_hash"] = "0" * 16
        v1.write_text(json.dumps(edited), encoding="utf-8")
        assert _raises(ov.FootballTargetFreezeError, lambda: ov.require_frozen(store)), "an edited registry must be refused"
    return "nothing is frozen without an explicit, reasoned decision; the registry is immutable and versioned; 08B is gated on it"


def check_no_combine_leak_08a2() -> str:
    plays, pp, _ = football_fixture(n_athletes=16)
    assert _raises(CombineLeakError, lambda: osm.semantic_base(plays, pp.with_columns(pl.lit(4.4).alias("cmb_forty")), ("sack", "cmb_forty"))), "a Combine column in the outcome base must be refused"
    base, _, _ = _semantics(plays, pp)
    out = ov.run_opportunity(base, VC)
    for name in (ov.ATHLETES.name, ov.RELIABILITY.name, ov.DENOMS.name, ov.RANKING.name):
        assert_no_combine_columns(out[name], name)
    rank = out[ov.RANKING.name].filter((pl.col("definition_a") == "FO01.S0") & (pl.col("definition_b") == "FO01.S0|median") & (pl.col("position_group") == ov.ALL))
    assert rank.height == 1 and rank["n_common"][0] == 8 and out[ov.RANKING.name].filter(pl.col("definition_a").str.starts_with("FO01") & pl.col("definition_b").str.starts_with("FO13")).is_empty(), "rankings are compared inside one outcome only"
    assert ov.denominators_hash() == ov.denominators_hash(ov.DENOMINATORS) != ov.denominators_hash(ov.DENOMINATORS[:-1]) and osm.definition_hash() != osm.definition_hash(osm.SEMANTICS[:-1]), "hashes follow the declarations"
    assert set(ov.PRIORITY) <= set(ov.BY_ID) and all(ov.BY_ID[k].legitimacy in ov.FREEZABLE for k in ov.PRIORITY), "only freezable denominators are in the proposal order"
    return "no Combine-named column in any outcome table; ranking comparisons stay inside an outcome; declaration hashes follow the declarations"


CHECKS_08A: tuple[Callable[[], str], ...] = (
    check_candidate_registry_contract, check_flag_and_null_parsing, check_coverage_counts, check_split_half_reliability,
    check_status_rule_and_gate, check_no_combine_leak_and_events)
CHECKS_08A2: tuple[Callable[[], str], ...] = (
    check_outcome_semantics_status, check_zero_versus_missing, check_opportunity_denominators, check_duplication_and_join,
    check_athlete_aggregation, check_game_split_reliability, check_cohort_linking, check_screen_and_proposal, check_explicit_freeze,
    check_no_combine_leak_08a2)
ALL_CHECKS: tuple[Callable[[], str], ...] = (*CHECKS_08A, *CHECKS_08A2)


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    """Run the synthetic checks in-process. A failing check is reported, not raised, so one table shows them all."""
    rows = []
    for fn in checks:
        try:
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "PASS", "detail": fn()})
        except Exception as e:                                   # noqa: BLE001 — the table must list every failure
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "FAIL", "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


__all__ = ["ALL_CHECKS", "CHECKS_08A", "CHECKS_08A2", "assert_all_passed", "run_all"]
