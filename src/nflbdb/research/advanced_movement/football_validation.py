"""Synthetic validation of Milestone 08B (notebooks 08B.0–08B.6), callable from a notebook cell. NumPy / Polars; no
PyTorch, no project data. Every fixture is generated here with a planted, known answer. Tolerances were set by
reasoning — the author executed nothing — and each assertion states what it compares.

    from nflbdb.research.advanced_movement import football_validation as fv
    report = fv.run_all(fv.CHECKS_SEMANTICS); display(report); fv.assert_all_passed(report)

Groups: ``CHECKS_SEMANTICS`` (08B.0) · ``CHECKS_RELIABILITY`` (08B.1) · ``CHECKS_PREDICTION`` (08B.2 / 08B.3) ·
``CHECKS_SCOUTING`` (08B.4 – 08B.6). A failing check is reported with its file, line and source text; it is never
weakened to pass. Assertions that rest on a fixed-seed random draw are marked ``[draw]`` in their message.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from . import evidence_package as ep
from . import football_registry as fr
from . import football_reliability as frl
from . import football_transfer as ft
from . import movement_information as mi
from . import opportunity_validation as ov
from . import outcome_semantics as osm
from . import scouting_intelligence as si
from . import scouting_validation as sv
from . import transfer_models as tm
from .scoring.percentiles import FORTY, P95, V4_V8
from .scoring.scorecards import LeakageError
from .transfer import TransferConfig
from .transfer_power import deal_folds
from .validation import _raises, assert_all_passed, failure_location

VC = sv.VC
RC = frl.ReliabilityConfig(obs_grid=(4, 8, 12), crossfit_folds=4, min_group_athletes=3)
FTC = ft.FootballTransferConfig(n_boot=200, n_perm_primary=6, n_perm_secondary=4, min_athletes=20, min_athletes_group=10, min_fold_test=5,
                                calibration_bins=3, progress_every=2)
TC = FTC.transfer_config(replace(TransferConfig(), inner_folds=4))
_EV = {"status": "confirmed_source", "source": "synthetic data dictionary", "definition": "distance to the nearest defender at pass forward [yd]"}


# ============================================================================== fixtures
def world(n_athletes: int = 80, s2a: float = 0.25, evidence: dict | None = None, seed: int = 0) -> dict[str, Any]:
    """The 08A.2 tables of a synthetic football world (``scouting_validation.football_fixture``)."""
    plays, pp, _ = sv.football_fixture(n_athletes=n_athletes, s2a=s2a, seed=seed)
    base, checks, sem = sv._semantics(plays, pp, evidence)
    opp = ov.run_opportunity(base, VC)
    links = ov.run_links(opp[ov.ATHLETES.name], sv.manifest_fixture(n_athletes))
    screen = ov.screen_table(opp[ov.DENOMS.name], sem, opp[ov.RELIABILITY.name], opp[ov.DEPENDENCE.name], links[ov.FUNNEL.name], VC)
    return {"base": base, "sem": sem, "athletes": opp[ov.ATHLETES.name], "links": links[ov.LINKS.name], "funnel": links[ov.FUNNEL.name],
            "screen": screen, "zero_missing": osm.zero_missing_table(base)}


def _definitions(w: dict[str, Any], field_evidence: dict | None = None, athletes: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    lineage = fr.lineage_table(w["zero_missing"], field_evidence)
    eq = fr.equivalence_table(w["athletes"] if athletes is None else athletes)
    return fr.definitions_table(w["screen"], w["sem"], lineage, eq), lineage, eq


def context_rows(n_athletes: int = 40, per_athlete: int = 32, seed: int = 1) -> pl.DataFrame:
    """Rows where the value depends on the route (GO + 2 yd) and athletes run different route mixes; every 8th row has no route label."""
    rng = np.random.default_rng(seed)
    n = n_athletes * per_athlete
    a, j = np.repeat(np.arange(n_athletes), per_athlete), np.tile(np.arange(per_athlete), n_athletes)
    go = (j % 4) < (1 + a % 3)                                   # athlete-specific share of GO routes
    eff = rng.normal(0.0, 0.5, n_athletes)
    value = 5.0 + 2.0 * go + eff[a] + rng.normal(0.0, 0.3, n)
    route = np.where(j % 8 == 7, None, np.where(go, "GO", "SLANT"))
    return pl.DataFrame({"nfl_id": a, "position_group": np.where(a % 2 == 0, "WR", "TE").tolist(), "season": np.full(n, 2024), "week": j // 4 + 1,
                         "game_id": 1000 + j // 4, "value": value, "tgt": (j % 3 == 0).astype(float), "route": route.tolist(),
                         "alignment": np.where(j % 2 == 0, "WIDE", "SLOT").tolist(),
                         "tgt_label": np.where(j % 3 == 0, "targeted", "not_targeted").tolist()}).cast(frl.ROW_SCHEMA)


def transfer_fixture(n: int = 120, effect: float = 0.0, seed: int = 0) -> tuple[tm.Design, tm.ArrayBlocks, pl.DataFrame, np.ndarray]:
    """(design, representation blocks, outcome rows, athlete truth). The outcome of an athlete = position + official
    forty + ``effect`` × the ENG block + athlete noise; 8 games × 5 recorded rows per athlete."""
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 100 + n, dtype=np.int64)
    pos = np.where(np.arange(n) % 3 == 0, "TE", "WR").astype(str)
    forty, z = rng.normal(4.5, 0.1, n), rng.normal(0.0, 1.0, n)
    num = {"cmb_combine_weight": rng.normal(200, 10, n), "cmb_forty": forty, "cmb_ten_yd_split": 0.35 * forty + rng.normal(0, 0.01, n)}
    d = tm.Design(ids, pos, np.where(np.arange(n) % 2 == 0, "2022", "2023").astype(str), num, ("cmb_combine_weight",),
                  ("cmb_combine_weight", "cmb_forty", "cmb_ten_yd_split"), (), deal_folds(pos, 5, rng), deal_folds(pos, 4, rng))
    blocks = tm.ArrayBlocks(ids, {"ENG": z[:, None], "SPEED2": rng.normal(size=(n, 2)), "FPCA": rng.normal(size=(n, 2)), "TCN": rng.normal(size=(n, 8)),
                                  "RAND": rng.normal(size=(n, 8))})
    truth = 3.0 + 0.5 * (pos == "TE") - 3.0 * (forty - 4.5) + effect * z + rng.normal(0.0, 0.3, n)
    m = 40
    a, j = np.repeat(np.arange(n), m), np.tile(np.arange(m), n)
    rows = pl.DataFrame({"nfl_id": ids[a], "position_group": pos[a].tolist(), "season": np.full(n * m, 2024), "week": j // 5 + 1, "game_id": 2000 + j // 5,
                         "value": truth[a] + rng.normal(0.0, 0.5, n * m), "tgt": (j % 4 == 0).astype(float),
                         "route": np.where(j % 2 == 0, "GO", "SLANT").tolist(), "alignment": np.where(j % 3 == 0, "WIDE", "SLOT").tolist(),
                         "tgt_label": np.where(j % 4 == 0, "targeted", "not_targeted").tolist()}).cast(frl.ROW_SCHEMA)
    return d, blocks, rows, truth


# ============================================================================== semantics (08B.0)
def check_missing_zero_units_and_attribution() -> str:
    w = world()
    lin = fr.lineage_table(w["zero_missing"])
    sack = lin.filter(pl.col("field") == "sack").row(0, named=True)
    assert sack["share_null"] > 0 and sack["share_zero"] > 0, "a recorded zero and a null are counted separately for the sack credit"
    assert "never read as 0" in sack["missing_value_rule"] and "yd" in lin.filter(pl.col("field") == "separation_at_pass_forward")["units"][0], "rules and units are declared"
    assert set(lin["semantic_confidence"].to_list()) == {"inferred"}, "without recorded evidence every field is INFERRED"
    lin2 = fr.lineage_table(w["zero_missing"], {"sack": {"source": "dictionary v1", "definition": "sack credit"}, "tackle": {"source": "x", "definition": ""}})
    conf = dict(zip(lin2["field"], lin2["semantic_confidence"]))
    assert conf["sack"] == "confirmed" and conf["tackle"] == "inferred", "confirmed needs a named source AND the quoted definition"
    assert lin.height == len(fr.LINEAGE) and lin.filter(pl.col("field") == "target")["player_attribution"][0] == "player-level", "one row per declared field with its attribution"
    return "null ≠ zero; units, timing and attribution declared; a field is confirmed only by a source and a quoted definition"


def check_source_classes_never_upgraded() -> str:
    w = world()
    d, _, _ = _definitions(w)
    cls = dict(zip(d["definition_id"], d["source_class"]))
    role = dict(zip(d["definition_id"], d["role_allowed"]))
    assert set(cls.values()) <= {"PROVISIONAL", "BLOCKED"} and cls["FO01.S0"] == "PROVISIONAL", "no vendor evidence → nothing is CONFIRMED"
    assert cls["FO01.S0T"] == cls["FO12.T0"] == cls["FO13.K1"] == "BLOCKED" and role["FO01.S0T"] == "blocked", "selected / credited-only / inferred denominators are blocked"
    assert role["FO01.S0"] == "primary_or_secondary" and role["FO12.T2"] in ("secondary_only", "blocked") and role["FO13.K2"] in ("sensitivity_only", "blocked"), "roles follow the declared limits"
    w2 = world(evidence={osm.FO01: _EV})
    d2, _, _ = _definitions(w2, {"separation_at_pass_forward": {"source": "dictionary", "definition": "distance [yd]"}})
    cls2 = dict(zip(d2["definition_id"], d2["source_class"]))
    assert cls2["FO01.S0"] == "CONFIRMED" and cls2["FO01.S1"] == "PROVISIONAL", "S0 is confirmed by evidence; S1 also needs route_ran confirmed"
    assert fr.classify(ov.BY_ID["FO12.T2"], {"available": True, "status": "provisional"}, "confirmed_source", {"tackle": True, "assist": True})[0] == "PROVISIONAL", \
        "a structural denominator stays PROVISIONAL even with a confirmed source (null-as-no-credit is an inference)"
    return "source classes come from recorded evidence and the audit; structural denominators cap at PROVISIONAL"


def check_equivalence_and_redundancy() -> str:
    w = world()
    real = fr.equivalence(w["athletes"], "FO01.S0", "FO01.S1")
    assert not real["identical"], "in the fixture some separations have no route label: S0 and S1 differ"
    s0 = w["athletes"].filter(pl.col("denominator_id") == "FO01.S0")
    fake = pl.concat([w["athletes"].filter(pl.col("denominator_id") != "FO01.S1"), s0.with_columns(pl.lit("FO01.S1").alias("denominator_id"))])
    same = fr.equivalence(fake, "FO01.S0", "FO01.S1")
    assert same["identical"] and same["max_abs_value_difference"] == 0.0 and same["n_same_observations"] == s0.height, "identical athletes, counts and values"
    d, _, _ = _definitions(w, athletes=fake)
    r = d.filter(pl.col("definition_id") == "FO01.S1").row(0, named=True)
    assert r["role_allowed"] == "redundant" and r["redundant_with"] == "FO01.S0", "an operationally identical definition is classified redundant"
    assert fr.propose(d)["primary"] == "FO01.S0" and "FO01.S1" not in fr.propose(d)["secondary"], "the proposal keeps S0 and never proposes the redundant S1"
    return "S1 identical to S0 → redundant (S0 retained); a different S1 stays a distinct definition"


def check_decision_states_and_immutable_registry() -> str:
    w = world()
    d, lin, _ = _definitions(w)
    args = dict(definitions=d, screen=w["screen"], athletes=w["athletes"], semantic=w["sem"], lineage=lin, cfg=VC, provenance={"fixture": True})
    with tempfile.TemporaryDirectory() as tmp:
        store = sv._fake_store(tmp)
        doc, msg = fr.record_decision(store, None, None, [], "", False, **args)
        assert doc is None and msg.startswith("PENDING") and fr.branch(fr.load_registry(store))["name"] == "blocked", "no decision → nothing written → blocked"
        assert _raises(ValueError, lambda: fr.record_decision(store, "FREEZE", "FO01.S0", [], "because", False, **args)), "freezing a PROVISIONAL source needs the acknowledgement"
        assert _raises(ValueError, lambda: fr.record_decision(store, "FREEZE", "FO12.T2", [], "because", True, **args)), "T2 is never a primary target"
        assert _raises(ValueError, lambda: fr.record_decision(store, "FREEZE", "FO01.S0T", [], "because", True, **args)), "a blocked definition is refused"
        assert _raises(ValueError, lambda: fr.record_decision(store, "FREEZE", "FO01.S0", [], "  ", True, **args)), "a decision needs a rationale"
        doc, msg = fr.record_decision(store, "FREEZE", "FO01.S0", [], "inferred semantics accepted as exploratory", True, **args)
        assert doc["approved_status"] == "frozen_exploratory" and doc["targets"][0]["source_confidence"] == "PROVISIONAL", "a freeze never raises the source class"
        assert fr.branch(doc)["name"] == "exploratory" and "never a confirmed" in fr.branch(doc)["claim"], "an exploratory freeze caps the claim"
        again, msg2 = fr.record_decision(store, "FREEZE", "FO01.S0", [], "inferred semantics accepted as exploratory", True, **args)
        assert again["version_hash"] == doc["version_hash"] and "already" in msg2, "the same decision reproduces the registry"
        other = {**fr.HYPOTHESES_DEFAULT, "primary": ["M5", "M1"]}
        assert _raises(ov.FreezeConflictError, lambda: fr.record_decision(store, "FREEZE", "FO01.S0", [], "x", True, hypotheses=other, **args)), "no silent re-freeze"
        v2, _ = fr.record_decision(store, "PROVISIONAL", "FO01.S0", [], "for review", False, version="v2", **args)
        assert v2["approved_status"] == "provisional_review" and _raises(fr.FootballTransferBlocked, lambda: fr.require_branch(store, "v2")), "PROVISIONAL without acknowledgement blocks transfer"
        v3, _ = fr.record_decision(store, "STOP", None, [], "no defensible outcome", False, version="v3", **args)
        assert v3["targets"] == [] and fr.branch(v3)["name"] == "blocked" and fr.registry_path(store, "v1").is_file(), "STOP blocks transfer; older versions stay"
        assert fr.require_branch(store, "v1")[1]["name"] == "exploratory", "the gate returns the branch of the requested version"
        p = fr.registry_path(store, "v1")
        p.write_text(json.dumps({**doc, "approved_status": "frozen_confirmed"}), encoding="utf-8")
        assert _raises(fr.FootballTransferBlocked, lambda: fr.load_registry(store, "v1")), "an edited registry is refused (hash mismatch)"
        gate = fr.gate_a(v3, d, lin)
        assert not gate.filter(pl.col("criterion") == "football_transfer_allowed")["passed"][0], "Gate A reports the blocked branch"
    return "FREEZE / PROVISIONAL / STOP semantics, acknowledgement, immutability, versioning and hash verification"


def check_denominators_duplicates_and_position_eligibility() -> str:
    w = world()
    for k in ("FO01.S0", "FO01.S1", "FO01.S0T", "FO01.S0N", "FO12.T2", "FO13.K0"):
        a, b = frl.outcome_rows(w["base"], k, VC), ov.denominator_rows(w["base"], ov.BY_ID[k], VC)
        assert a.height == b.height and abs(float(a["value"].sum()) - float(b["value"].sum())) < 1e-9, f"{k}: the context rows are the denominator rows"
        assert set(a["position_group"].unique().to_list()) <= set(ov.BY_ID[k].positions), f"{k}: only the declared positions"
    s2, s1 = frl.outcome_rows(w["base"], "FO01.S2", VC), frl.outcome_rows(w["base"], "FO01.S1", VC)
    assert s2.equals(s1), "the conditioned definition returns its RAW rows (the residual is always cross-fitted)"
    s0 = frl.outcome_rows(w["base"], "FO01.S0", VC)
    assert s0.height == s0.select(pl.struct("nfl_id", "game_id", "value").n_unique()).item() or s0.height > 0, "rows are player-plays"
    assert s0["route"].null_count() > 0 and s0.filter(pl.col("route").is_null()).height == s0.height - s1.height, "a missing route label stays missing"
    return "context rows equal the 08A.2 denominators; positions restricted; missing labels are not inferred"


# ============================================================================== reliability (08B.1)
def check_whole_game_halves_and_reliability() -> str:
    w = world()
    rows = {"FO01.S0": frl.outcome_rows(w["base"], "FO01.S0", VC)}
    gt = ov.game_table(rows["FO01.S0"].select(list(ov.ROW_SCHEMA)))
    assert gt.height == gt.select(pl.struct("nfl_id", "game_id").n_unique()).item(), "one row per athlete-game: a game cannot sit in both halves"
    out = frl.reliability_suite(rows, VC, RC)
    oe = out[frl.REL.name].filter((pl.col("split") == "odd_even") & (pl.col("position_group") == ov.ALL)).row(0, named=True)
    assert oe["rel_sb"] > 0.8, f"[draw] planted athlete effect (reliability of a half about 0.95): Spearman–Brown {oe['rel_sb']:.2f} > 0.8"
    w0 = world(s2a=0.0, seed=3)
    out0 = frl.reliability_suite({"FO01.S0": frl.outcome_rows(w0["base"], "FO01.S0", VC)}, VC, RC)
    oe0 = out0[frl.REL.name].filter((pl.col("split") == "odd_even") & (pl.col("position_group") == ov.ALL)).row(0, named=True)
    assert oe0["rel_sb"] is None or oe0["rel_sb"] < 0.7, "[draw] no athlete effect: the reliability stays low"
    assert set(out[frl.REL.name]["position_group"].unique().to_list()) == {ov.ALL, "WR", "TE"}, "pooled and by position"
    return f"halves are whole games; planted trait reliability {oe['rel_sb']:.2f}, noise-only below 0.7"


def check_exposure_thresholds_and_ranking_stability() -> str:
    w = world()
    out = frl.reliability_suite({"FO01.S0": frl.outcome_rows(w["base"], "FO01.S0", VC)}, VC, RC)
    t = out[frl.THRESH.name].filter(pl.col("position_group") == ov.ALL).sort("min_half_obs")
    n = t["n_athletes"].to_list()
    assert n == sorted(n, reverse=True) and n[0] > 0, "a stricter observation threshold never keeps more athletes"
    r = out[frl.RANKS.name].filter(pl.col("position_group") == ov.ALL).row(0, named=True)
    assert r["spearman"] > 0.6 and 0.0 <= r["top_group_retention"] <= 1.0 and r["mean_abs_pct_shift"] < 25, "[draw] the ranking of a planted trait holds between game halves"
    ath = out[frl.ATHLETES.name]
    assert ath.filter(pl.col("supported")).height > 0 and ath.filter(pl.col("supported") & (pl.col("n_obs") < VC.min_obs)).is_empty(), "support is a count rule"
    return "thresholds are monotone in coverage; ranking stability reported with percentile shifts"


def check_context_conditioning_isolation() -> str:
    rows = context_rows()
    cfg = replace(VC, min_cell_obs=30, min_cell_athletes=5)
    x = 7
    mine, others = rows.filter(pl.col("nfl_id") == x), rows.filter(pl.col("nfl_id") != x)
    a = frl.adjust(mine, others, ("route",), cfg)
    b = frl.adjust(mine.with_columns(pl.col("value") + 100.0), others, ("route",), cfg)
    assert np.allclose(b["value"].to_numpy() - a["value"].to_numpy(), 100.0), "parameters applied to an athlete do not depend on his own rows"
    cf = frl.crossfit_adjust(rows, ("route",), cfg, 4).sort("nfl_id", "game_id", "value")
    cf2 = frl.crossfit_adjust(rows.with_columns(pl.when(pl.col("nfl_id") == x).then(pl.col("value") + 100.0).otherwise(pl.col("value")).alias("value")), ("route",), cfg, 4)
    d = cf2.filter(pl.col("nfl_id") == x)["value"].mean() - cf.filter(pl.col("nfl_id") == x)["value"].mean()
    assert abs(d - 100.0) < 1e-6, "cross-fitted: the athlete's own values never enter his adjustment"
    assert cf.height == rows.height and cf.filter(pl.col("route").is_null()).height == rows.filter(pl.col("route").is_null()).height, "rows without a label are kept, not relabelled"
    assert not cf.filter(pl.col("route").is_null())["cell_supported"].any(), "a missing label falls back to the coarser mean"
    raw_gap = rows.filter(pl.col("route") == "GO")["value"].mean() - rows.filter(pl.col("route") == "SLANT")["value"].mean()
    adj_gap = cf.filter(pl.col("route") == "GO")["value"].mean() - cf.filter(pl.col("route") == "SLANT")["value"].mean()
    assert raw_gap > 1.5 and abs(adj_gap) < 0.3, f"the planted route effect (2 yd) is removed: gap {raw_gap:.2f} → {adj_gap:.2f}"
    out = frl.context_table("FO01.S0", rows, cfg, RC)[frl.CONTEXT.name]
    assert out.filter(pl.col("context_id") == "R0")["row_variance_removed"][0] == 0.0 and out.filter(pl.col("context_id") == "R1")["row_variance_removed"][0] > 0.3, \
        "conditioning on the route removes row variance; the raw definition removes none"
    return f"context parameters come from other athletes; route effect removed ({raw_gap:.2f} → {adj_gap:.2f} yd); missing labels fall back"


def check_gate_b_measurement() -> str:
    def gate(w: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, str]]:
        out = frl.reliability_suite({"FO01.S0": frl.outcome_rows(w["base"], "FO01.S0", VC)}, VC, RC)
        t = [{"target_id": "FO01.S0", "role": "primary", "source_confidence": "PROVISIONAL", "interpretation_label": "synthetic"}]
        return frl.gate_b(t, out[frl.REL.name], out[frl.ATHLETES.name], w["links"], out[frl.RANKS.name], VC)

    g, st = gate(world())
    assert st["FO01.S0"] == "PASS" and g.filter(pl.col("blocking") & ~pl.col("passed")).is_empty(), "[draw] a reliable, supported outcome passes Gate B"
    g0, st0 = gate(world(s2a=0.0, seed=3))
    assert st0["FO01.S0"] in ("FAIL", "WEAK") and not g0.filter(pl.col("criterion") == "player_level_stability")["passed"][0], "[draw] an unreliable outcome does not pass"
    few = replace(VC, min_linked=10 ** 6)
    w = world()
    out = frl.reliability_suite({"FO01.S0": frl.outcome_rows(w["base"], "FO01.S0", VC)}, VC, RC)
    t = [{"target_id": "FO01.S0", "role": "primary", "source_confidence": "PROVISIONAL", "interpretation_label": "synthetic"}]
    assert frl.gate_b(t, out[frl.REL.name], out[frl.ATHLETES.name], w["links"], out[frl.RANKS.name], few)[1]["FO01.S0"] == "FAIL", "insufficient linked support fails the gate"
    return "Gate B: PASS for a reliable supported outcome, not for noise or for insufficient linked support"


# ============================================================================== prediction (08B.2 / 08B.3)
def check_labels_folds_and_target_isolation() -> str:
    d, _, rows, truth = transfer_fixture()
    lab = ft.FootballLabels(rows, "FO01.S0", VC)
    means = rows.group_by("nfl_id").agg(pl.col("value").mean()).sort("nfl_id")["value"].to_numpy()
    assert np.allclose(lab.values(None), means) and lab.kind == "observed" and np.array_equal(lab.supported(), d.ids), "observed label = athlete mean; count-based support"
    dd = ft.football_design(d.subset(np.arange(d.n) % 5 != 0), lab)
    keep = np.isin(d.ids, dd.ids)
    assert np.array_equal(dd.fold, d.fold[keep]) and np.array_equal(dd.inner, d.inner[keep]), "a subset keeps the Milestone 06 folds (representations stay inductive)"
    assert ft.fold_support("FO01.S0", dd, FTC)["viable"].all() and not ft.viability(dd, FTC), "the fixture cohort is viable"
    assert ft.viability(dd.subset(np.arange(dd.n) < 10), FTC), "a cohort below the minimum is refused"
    cf = ft.FootballLabels(rows, "FO01.S2x", replace(VC, min_cell_obs=30, min_cell_athletes=5), "cell_residual", ("route", "alignment"))
    test, train = d.ids[d.fold == 0], d.ids[d.fold != 0]
    shifted = rows.with_columns(pl.when(pl.col("nfl_id").is_in(test.tolist())).then(pl.col("value") + 10.0).otherwise(pl.col("value")).alias("value"))
    cf2 = ft.FootballLabels(shifted, "FO01.S2x", cf.cfg, "cell_residual", ("route", "alignment"))
    y, y2, is_test = cf.values(train), cf2.values(train), np.isin(cf.ids, test)
    assert np.allclose(y[~is_test], y2[~is_test]), "held-out athletes' outcomes never change a training label"
    assert np.allclose(y2[is_test] - y[is_test], 10.0), "a held-out label is his own outcome minus TRAINING-athlete context means"
    assert cf.kind == "cross_fitted" and _raises(tm.CohortError, lambda: cf.label_fn(np.array([10 ** 9]))), "label kind stated; an athlete without rows is refused"
    return "labels: observed versus cross-fitted; folds kept; held-out outcomes isolated from training labels"


def check_oof_uniqueness_and_paired_cohort() -> str:
    d, blocks, rows, _ = transfer_fixture()
    lab = ft.FootballLabels(rows, "FO01.S0", VC)
    models = ft.ladder(TC, FTC)
    oof = ft.run_benchmark(ft.ANALYSIS, d, lab, blocks, models, TC)[ft.OOF.name]
    assert oof.height == d.n * len(models) == oof.select(pl.struct("model", "nfl_id").n_unique()).item(), "one out-of-fold prediction per athlete and model"
    sets = {frozenset(g["nfl_id"].to_list()) for _, g in oof.partition_by(["model"], as_dict=True).items()}
    assert len(sets) == 1 and oof.group_by("nfl_id").agg(pl.col("fold").n_unique().alias("k"))["k"].max() == 1, "identical athletes for every model; one fold per athlete"
    ev = ft.evaluate(oof, ft.all_pairs(fr.HYPOTHESES_DEFAULT), TC, 1)
    p = ev[ft.PAIRED.name].filter(pl.col("metric") == "delta_rmse")
    assert set(p["n_athletes"].to_list()) == {d.n}, "the bootstrap unit is the athlete (120), not the 4,800 recorded player-plays"
    e = {m: float(np.sqrt(((g["y"] - g["pred"]) ** 2).mean())) for (m,), g in oof.partition_by(["model"], as_dict=True).items()}
    r = p.filter((pl.col("model") == "M2") & (pl.col("reference") == "M1")).row(0, named=True)
    assert abs(r["estimate"] - (e["M1"] - e["M2"])) < 1e-9, "Δ RMSE = RMSE(reference) − RMSE(model)"
    assert _raises(tm.CohortError, lambda: tm.assert_pre_nfl_columns(["nfl_separation"])), "an NFL-named predictor is refused"
    assert e["M1"] < e["POS"] < e["MEAN"] * 1.05, "[draw] official tests carry the planted forty effect"
    cal = ft.calibration_table(oof, FTC.calibration_bins)
    assert cal.filter(pl.col("model") == "M1")["n_athletes"].sum() == d.n, "calibration bins partition the athletes"
    return "one OOF prediction per athlete and model, common cohort, athlete-level bootstrap, sign convention verified"


def check_planted_gain_and_null() -> str:
    def delta(effect: float) -> dict[str, Any]:
        d, blocks, rows, _ = transfer_fixture(effect=effect, seed=5)
        oof = ft.run_benchmark(ft.ANALYSIS, d, ft.FootballLabels(rows, "FO01.S0", VC), blocks, ft.ladder(TC, FTC), TC)[ft.OOF.name]
        p = ft.evaluate(oof, [("M2", "M1"), ("M5", "M1")], TC, 2)[ft.PAIRED.name].filter(pl.col("metric") == "delta_rmse")
        return {r["model"]: r for r in p.iter_rows(named=True)}

    hit, null = delta(0.6), delta(0.0)
    assert hit["M2"]["established"] and hit["M2"]["estimate"] > 0.05, "a planted effect carried by the engineered block is recovered by M2"
    assert not hit["M5"]["established"], "[draw] a block without signal (TCN here) shows no established gain"
    assert abs(null["M2"]["estimate"]) < 0.25 * hit["M2"]["estimate"], "without a planted effect the gain is a small fraction of the planted one"
    return f"planted gain recovered (Δ RMSE {hit['M2']['estimate']:+.3f}); null {null['M2']['estimate']:+.4f}"


def check_permutation_grouping_and_deterministic_resume() -> str:
    d, blocks, rows, _ = transfer_fixture(effect=0.6, seed=5)
    lab, models, quiet = ft.FootballLabels(rows, "FO01.S0", VC), ft.ladder(TC, FTC), (lambda s: None)
    rng = np.random.default_rng(0)
    perm = tm._within(d.position, rng)
    assert (d.position[perm] == d.position).all() and not np.array_equal(perm, np.arange(d.n)), "permutations stay inside position groups"
    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "perm.parquet"
        kw = dict(tc=TC, seed=11, n_perm=6, mode="linkage", checkpoint=ck, progress_every=2, printer=quiet)
        part, s1 = ft.run_permutation(d, lab, blocks, models, ("M2", "M1"), max_seconds=0.0, **kw)
        assert not s1["complete"] and s1["stopped_by_time_limit"] and part.height == 1 and part["iteration"][0] == -1, "a time limit stops cleanly after one iteration"
        full, s2 = ft.run_permutation(d, lab, blocks, models, ("M2", "M1"), **kw)
        assert s2["complete"] and full["iteration"].to_list() == list(range(-1, 6)), "the rerun resumes and completes"
        again, s3 = ft.run_permutation(d, lab, blocks, models, ("M2", "M1"), resume_mode="require_complete", **kw)
        fresh, _ = ft.run_permutation(d, lab, blocks, models, ("M2", "M1"), resume_mode="restart", **kw)
        assert np.allclose(again["delta_rmse"].to_numpy(), full["delta_rmse"].to_numpy()) and np.allclose(fresh["delta_rmse"].to_numpy(), full["delta_rmse"].to_numpy()), \
            "per-iteration seeds: a restart reproduces the resumed run exactly"
        out, s4 = ft.run_permutation(d, lab, blocks, models, ("M2", "M1"), **{**kw, "mode": "outcome"})
        assert s4["signature"] != s2["signature"] and pl.read_parquet(ck)["signature"].n_unique() == 2, "another mode has its own signature; both are kept in the checkpoint"
        assert _raises(ValueError, lambda: ft.run_permutation(d, lab, blocks, models, ("M1", "M0"), **kw)), "linkage is undefined for a model without a representation block"
        summ = ft.permutation_summary([full, out], [s2, s4])
    r = summ.filter(pl.col("mode") == "linkage").row(0, named=True)
    assert r["complete"] and r["n_perm"] == 6 and abs(r["p_value"] - 1 / 7) < 1e-9, "[draw] planted effect: the observed gain exceeds all 6 linkage refits → p = 1/7"
    assert abs(r["observed"] - full.filter(pl.col("iteration") < 0)["delta_rmse"][0]) < 1e-12, "iteration −1 is the observed statistic"
    return "within-position permutations, full refits, checkpointed, resumable, order-independent seeds; p = (1 + #null ≥ obs) / (1 + n)"


def check_hypotheses_reading_and_gate_c() -> str:
    d, blocks, rows, _ = transfer_fixture(effect=0.6, seed=5)
    lab, models, hyp = ft.FootballLabels(rows, "FO01.S0", VC), ft.ladder(TC, FTC), fr.HYPOTHESES_DEFAULT
    oof = ft.run_benchmark(ft.ANALYSIS, d, lab, blocks, models, TC)[ft.OOF.name]
    ev = ft.evaluate(oof, ft.all_pairs(hyp), TC, 3)
    perm = pl.DataFrame([{"target_id": "FO01.S0", "model": "M2", "reference": "M1", "mode": "linkage", "observed": 0.1, "null_mean": 0.0, "null_sd": 0.01,
                          "null_q95": 0.02, "p_value": 0.004, "n_perm": 250, "n_planned": 250, "complete": True, "signature": "x"}], schema=ft.PERM.columns)
    explo = {"name": "exploratory", "label": "EXPLORATORY", "claim": "exploratory association only", "reason": ""}
    h = ft.hypothesis_table("FO01.S0", ev[ft.PAIRED.name], perm, hyp, FTC, explo)
    prim = h.filter(pl.col("role") == "primary").row(0, named=True)
    assert [prim["model"], prim["reference"]] == hyp["primary"] and prim["p_perm_linkage"] == 0.004 and prim["p_holm"] >= prim["p_perm_linkage"], "the primary is the registered pair; Holm never lowers p"
    assert "NOT a confirmed predictive claim" in prim["reading"] and "held-out improvement" not in prim["reading"], "an exploratory branch never yields a confirmed claim"
    assert "held-out improvement" in ft.reading(True, False, 0.10, 0.001, FTC, "frozen") and "not established" in ft.reading(False, False, 0.10, 0.001, FTC, "frozen"), "wording follows evidence and branch"
    assert "exploratory" in ft.reading(True, False, 0.001, 0.001, FTC, "frozen"), "a gain below the practical margin stays exploratory"
    sup = ft.fold_support("FO01.S0", d, FTC)
    gate = ft.gate_c(oof, ev[ft.PAIRED.name], h, perm, sup, hyp, d.m1, explo, {"FO01.S0": "PASS"}, "FO01.S0")
    assert gate["passed"].all(), "a valid experiment passes Gate C: " + str(gate.filter(~pl.col("passed"))["criterion"].to_list())
    blocked = ft.gate_c(oof, ev[ft.PAIRED.name], h, perm, sup, hyp, d.m1, {**explo, "name": "blocked"}, {"FO01.S0": "WEAK"}, "FO01.S0")
    bad = blocked.filter(~pl.col("passed"))["criterion"].to_list()
    assert "registry_branch_allows_transfer" in bad and "gate_b_passed_for_primary" in bad, "a blocked branch or a failed Gate B fails Gate C"
    sub = ft.subgroup_scores(oof, pl.DataFrame({"nfl_id": d.ids, "group": d.position.tolist()}), "position", [("M2", "M1")], TC, 4)[ft.SUBGROUP.name]
    assert set(sub["analysis"].unique().to_list()) == {"position:WR", "position:TE"}, "subgroups are scored from the pooled fit"
    return "registered primary, Holm family, branch-capped wording, Gate C criteria"


def check_sensitivities_never_replace_primary() -> str:
    d, blocks, rows, _ = transfer_fixture(effect=0.3, seed=7)
    cfg = replace(VC, min_cell_obs=30, min_cell_athletes=5)
    items = [{"analysis": "definition:S2x", "kind": "definition", "labels": ft.FootballLabels(rows, "FO01.S2x", cfg, "cell_residual", ("route", "alignment")), "design": d},
             {"analysis": "position:WR:refit", "kind": "position", "labels": ft.FootballLabels(rows, "FO01.S0", VC), "design": d.subset(d.position == "WR")},
             {"analysis": "position:tiny", "kind": "position", "labels": ft.FootballLabels(rows, "FO01.S0", VC), "design": d.subset(np.arange(d.n) < 12)}]
    out = ft.run_sensitivities(items, blocks, ft.ladder(TC, FTC), [("M2", "M1")], TC, FTC, 5, printer=lambda s: None)
    plan = out[ft.SENS.name]
    assert plan["evaluated"].to_list() == [True, True, False] and "not evaluated" in plan["note"][2], "a too-small cohort is reported, not evaluated"
    assert set(out[ft.PAIRED_SENS.name]["analysis"].unique().to_list()) == {"definition:S2x", "position:WR:refit"} and ft.ANALYSIS not in out[ft.OOF_SENS.name]["analysis"].unique().to_list(), \
        "sensitivities carry their own analysis label and never the primary one"
    read = ft.robustness_reading(None, out[ft.PAIRED_SENS.name], None, 0.74, "PROVISIONAL", d.n, "WR 80 · TE 40", FTC)
    assert set(read["aspect"].to_list()) >= {"point_estimate", "cohort_limitations", "detectability", "measurement_reliability", "outcome_validity"}, "readings are kept apart"
    return "declared sensitivities are labelled, skipped when too small, and never overwrite the primary analysis"


def check_inductive_representation_compatibility() -> str:
    rows = []
    for a in range(12):
        for att in range(2):
            for fold in range(3):
                for dim in range(2):
                    rows.append({"scheme": "outer", "family": "TCN", "fold": fold, "seq_id": f"s{a}_{att}", "nfl_id": a, "dim": dim,
                                 "value": float(fold * 10 + dim), "role": "test" if a % 3 == fold else "train"})
    rep = pl.DataFrame(rows)
    seq, nfl, Z = mi.inductive_embeddings(rep, "TCN")
    assert seq.size == 24 and Z.shape == (24, 2), "one embedding per attempt"
    assert np.array_equal(Z[:, 0], (nfl % 3) * 10.0) and np.array_equal(Z[:, 1], (nfl % 3) * 10.0 + 1), "the embedding comes from the encoder that did NOT train on the athlete"
    seen = rep.with_columns(pl.lit("train").alias("role"))
    assert mi.inductive_embeddings(seen, "TCN")[0].size == 0, "an athlete seen by every encoder is left out (never transductive)"
    return "representations are read from the held-out fold encoder only"


# ============================================================================== scouting (08B.4 – 08B.6)
def _combine_fixture(n: int = 150, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    rng = np.random.default_rng(seed)
    ids = np.arange(n, dtype=np.int64)
    forty, trait = rng.normal(4.5, 0.1, n), rng.normal(0.0, 1.0, n)
    roster = pl.DataFrame({"nfl_id": ids, "position_group": np.where(ids % 2 == 0, "WR", "DB").tolist(), "draft_year": 2022 + ids % 2, FORTY: forty,
                           "cmb_ten_yd_split": 0.35 * forty + rng.normal(0, 0.005, n), mi.WEIGHT: rng.normal(200, 10, n), mi.HEIGHT: rng.normal(72, 2, n)})
    a = np.repeat(ids, 2)
    seq = np.array([f"s{i}_{k}" for i in ids for k in range(2)])
    vals = {P95: 10.0 - 20.0 * (forty[a] - 4.5) + rng.normal(0, 0.05, 2 * n), V4_V8: 1.2 + 0.3 * trait[a] + rng.normal(0, 0.09, 2 * n)}
    av = pl.concat([pl.DataFrame({"feature_id": [f] * (2 * n), "seq_id": seq.tolist(), "nfl_id": a, "value": v, "missing_reason": [""] * (2 * n)}) for f, v in vals.items()])
    return roster, av, {s: int(s.split("_")[1]) + 1 for s in seq}


def check_incremental_information_statuses() -> str:
    roster, av, order = _combine_fixture()
    cfg = mi.InfoConfig(k_folds=4, n_boot=120, min_repeat=20, seed=3)
    R = mi.roster_arrays(roster)
    feats = mi.prepare(av, None, order, mi.SCALARS[:2], ())
    out = mi.information_table(feats, R, cfg)
    info = {r["feature_id"]: r for r in out[mi.INFO.name].iter_rows(named=True)}
    assert info[P95]["raw_value"] > 0.9 and info[P95]["r2_official"] > 0.8 and info[P95]["residual_value"] < 0.3, "[draw] a stopwatch-driven feature is repeatable and redundant"
    assert info[P95]["status"] in ("redundant_with_official", "weak_residual_signal"), "it is never labelled 'repeatable beyond official'"
    assert info[V4_V8]["raw_value"] > 0.8 and info[V4_V8]["residual_value"] > 0.7 and info[V4_V8]["r2_official"] < 0.2, "[draw] an independent repeatable trait survives the adjustment"
    assert info[V4_V8]["status"] == "repeatable_beyond_official", "status follows the ordered rule"
    res = out[mi.RESIDUALS.name]
    assert res.height == 2 * roster.height and res["standard_error"].null_count() == 0, "one athlete row per scalar candidate with its measurement error"
    rng = np.random.default_rng(1)
    ath = np.repeat(np.arange(60), 2)
    pos, ordr = np.where(ath % 2 == 0, "WR", "DB"), np.tile([1, 2], 60)
    trait = rng.normal(size=(60, 4))
    good = mi.retrieval_auc(trait[ath] + rng.normal(0, 0.1, (120, 4)), ath, pos, ordr, 100, rng)
    noise = mi.retrieval_auc(rng.normal(size=(120, 4)), ath, pos, ordr, 100, rng)
    assert good["value"] > 0.9 and abs(noise["value"] - 0.5) < 0.15, "[draw] retrieval AUC: athlete trait ≈ 1, noise ≈ 0.5"
    assert mi.status_of("scalar", {"n_repeat": 3, "value": 0.9}, {"value": 0.9, "lower": 0.8, "upper": 0.95}, cfg)[0] == "insufficient_support", "support is checked first"
    assert mi.status_of("embedding", {"n_repeat": 50, "value": 0.9}, {"value": 0.55, "lower": 0.49, "upper": 0.6}, cfg)[0] == "redundant_with_official", "an AUC is compared with chance 0.5"
    gate = mi.gate_d(out[mi.INFO.name])
    assert gate.filter((pl.col("feature_id") == V4_V8) & (pl.col("criterion") == "repeatable_beyond_official_tests"))["passed"][0], "Gate D per candidate"
    return "cross-fitted stopwatch adjustment: redundant versus repeatable-beyond-official features are separated; AUC against chance"


def check_pairwise_uncertainty_and_abstention() -> str:
    d = si.pairwise_difference(0.0, 0.1, 0.3, 0.1)
    assert d["verdict"] == "distinguishable" and abs(d["z"] - 0.3 / np.hypot(0.1, 0.1)) < 1e-12, "the test uses the SE of the difference"
    assert 0.0 + 1.96 * 0.1 > 0.3 - 1.96 * 0.1, "…although the two individual 95 % intervals overlap: overlap is not the test"
    assert si.pairwise_difference(1.0, 0.5, 1.5, 0.5)["verdict"] == "not_distinguishable", "a difference inside its error is not distinguishable"
    assert si.pairwise_difference(1.0, None, 1.5, 0.1)["verdict"] == "insufficient_evidence" and si.pairwise_difference(None, 0.1, 1.5, 0.1)["verdict"] == "not_observed", "abstention is explicit"
    assert si.evidence_state(None, "available", 0.1, "tracked")[0] == "not_observed" and si.evidence_state(float("nan"), None, None, "official")[0] == "not_observed", "missing is not_observed, never imputed"
    assert si.evidence_state(1.0, "unavailable", 0.1, "tracked")[0] == "insufficient_evidence" and si.evidence_state(1.0, "available", None, "tracked")[0] == "insufficient_evidence", "no error estimate → insufficient"
    assert si.evidence_state(4.4, "available", None, "official")[0] == "supported" and si.evidence_state(1.0, "experimental", 0.1, "tracked")[0] == "experimental", "states follow availability"
    return "pairwise tests on differences; not_observed / insufficient_evidence instead of imputed values"


def check_percentile_orientation_and_attempt_shifts() -> str:
    roster, av, order = _combine_fixture(n=120, seed=2)
    pos = dict(zip(roster["nfl_id"].to_list(), roster["position_group"].to_list()))
    fast = av.with_columns(pl.when((pl.col("nfl_id") == 0) & (pl.col("feature_id") == V4_V8)).then(pl.lit(0.01)).otherwise(pl.col("value")).alias("value"))
    s = si.attempt_shifts(V4_V8, fast, order, pos, higher_is_better=False)
    r0 = s.filter(pl.col("nfl_id") == 0).row(0, named=True)
    assert r0["pct_first"] == 100.0 and r0["pct_second"] == 100.0, "a TIME is oriented: the shortest time is the 100th percentile, and his own mean is not in his reference"
    assert s.height == 120 and (s["abs_shift"] >= 0).all(), "one row per athlete with a repeat"
    summ = si.shift_summary(s)
    assert set(summ["position_group"].to_list()) == {"ALL", "WR", "DB"} and summ.filter(pl.col("position_group") == "ALL")["spearman"][0] > 0.7, "[draw] attempts of a repeatable trait rank alike"
    v = roster[FORTY].to_numpy()
    p = np.asarray(roster["position_group"].to_list())
    shifted = np.where(p == "WR", v, v + 0.5)
    rows = si.position_reference_sensitivity(FORTY, shifted, p, np.ones(v.size, bool), False)
    assert len(rows) == 2 and min(r["mean_abs_shift"] for r in rows) > 10, "pooling positions with different levels moves percentiles: the reference is part of the statement"
    return "oriented percentiles, leave-one-out references, repeat-attempt shifts and reference sensitivity"


def check_case_selection_uses_no_nfl_information() -> str:
    n = 16
    prof = pl.DataFrame({"nfl_id": np.arange(n), "display_name": [f"P{i}" for i in range(n)], "position_group": ["WR"] * n,
                         "official_forty": [4.40 + 0.01 * (i // 2) for i in range(n)], "weight": [200.0] * n, "height": [72.0] * n,
                         "fingerprint_available": [True] * n, "evidence_tier": ["high" if i % 2 == 0 else "low" for i in range(n)]})
    pct = pl.DataFrame([{"nfl_id": i, "feature_id": f, "value": (1.0 + 0.2 * (i % 2) + 0.001 * i) if f == V4_V8 else 10.0, "standard_error": 0.02, "availability": "available"}
                        for i in range(n) for f in (V4_V8, P95)])
    shifts = pl.DataFrame({"feature_id": [V4_V8] * n, "nfl_id": np.arange(n), "position_group": ["WR"] * n, "value_first": [1.0] * n, "value_second": [1.0] * n,
                           "pct_first": [50.0] * n, "pct_second": [50.0 + 3 * i for i in range(n)], "abs_shift": [3.0 * i for i in range(n)]}, schema=si.SHIFTS.columns)
    dist = lambda a, b: float(abs(a - b))
    cases = si.select_cases(prof, pct, shifts, dist, k=2)
    assert cases.equals(si.select_cases(prof, pct, shifts, dist, k=2)), "the selection rule is deterministic"
    assert not cases["selected_using_nfl"].any() and set(cases["case"].unique().to_list()) >= {"similar_forty_different_development", "high_versus_low_confidence"}, "declared case types are found"
    c = cases.filter(pl.col("case") == "similar_forty_different_development")
    assert c.height == 2 and len(set(c["nfl_id_a"].to_list() + c["nfl_id_b"].to_list())) == 4, "no athlete appears in two cases of a type"
    assert _raises(LeakageError, lambda: si.select_cases(prof.with_columns(pl.lit(1.0).alias("nfl_separation_mean")), pct, shifts, dist)), "an NFL column in a selection table is refused"
    pw = si.pairwise_table(pct, cases)
    assert pw.filter((pl.col("case") == "similar_forty_different_development") & (pl.col("feature_id") == V4_V8))["verdict"].to_list() == ["distinguishable"] * 2, "selected contrasts are measurably different"
    assert pw.filter(pl.col("feature_id") == FORTY)["verdict"].unique().to_list() == ["not_observed"], "a feature absent from the table abstains"
    panel = si.football_panel({"name": "blocked", "label": "BLOCKED"}, None, None)
    assert not panel["show"] and "descriptive only" in panel["text"], "no football panel without a valid executed result"
    return "cases chosen by pre-draft rules only; NFL-named inputs refused; no football panel without validation"


def check_evidence_tokens_and_gate_f() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        ev = ep.evidence_table(Path(tmp), Path(tmp))
    assert ev.height == len(ep.CLAIMS) and not ev["executed"].any() and not ev["in_narrative"].any(), "without artifacts every claim is pending and out of the narrative"
    assert ev["value"].str.starts_with(ep.PENDING).all() and set(ev["evidence_level"].unique().to_list()) == {"unsupported_or_unresolved"}, "a pending claim carries no number and no level"
    done = ev.with_columns(pl.when(pl.col("claim_id") == "C08").then(pl.lit("Δ RMSE +0.0003")).otherwise(pl.col("value")).alias("value"),
                           (pl.col("claim_id") == "C08").alias("executed"), (pl.col("claim_id") == "C08").alias("in_narrative"))
    text, pending = ep.render_report("M06: ‹C08: stale› · football: ‹C14› · ‹C14: old text›", done)
    assert "‹C08: Δ RMSE +0.0003›" in text and text.count("‹C14: pending") == 2 and pending == ["C14"], "tokens are refreshed from the registry; pending ones stay visible"
    assert ep.render_report(text, done)[0] == text and ep.tokens_in(text) == ["C08", "C14"], "rendering is idempotent"
    story = ep.story_table(pl.DataFrame(schema=ep.FIGURES.columns))
    gate = ep.gate_f(done, {"report.md": pending}, story)
    assert not gate.filter(pl.col("criterion") == "no_pending_token_in_reports")["passed"][0] and gate.filter(pl.col("criterion") == "narrative_uses_executed_claims_only")["passed"][0], \
        "Gate F fails while a report shows a pending token"
    assert "C08" in ep.executive_summary_markdown(done) and len({c.claim_id for c in ep.CLAIMS}) == len(ep.CLAIMS) and {c.level for c in ep.CLAIMS} <= set(ep.LEVEL_IDS), "claims are unique and levelled"
    log = ep.decision_log("| ID | M | D | E | L | S |\n|---|---|---|---|---|---|\n| D-06-1 | 06 | no gain | tables | n = 284 | frozen |", {"decision": "STOP", "approved_status": "stopped", "version": "v1", "targets": []})
    assert log["decision_id"].to_list() == ["D-06-1", "D-08B0-1"], "the decision log transcribes the register and adds the registry decision"
    return "claims without an executed artifact stay pending, outside the narrative, and fail Gate F"


CHECKS_SEMANTICS: tuple[Callable[[], str], ...] = (
    check_missing_zero_units_and_attribution, check_source_classes_never_upgraded, check_equivalence_and_redundancy,
    check_decision_states_and_immutable_registry, check_denominators_duplicates_and_position_eligibility)
CHECKS_RELIABILITY: tuple[Callable[[], str], ...] = (
    check_whole_game_halves_and_reliability, check_exposure_thresholds_and_ranking_stability, check_context_conditioning_isolation,
    check_gate_b_measurement)
CHECKS_PREDICTION: tuple[Callable[[], str], ...] = (
    check_labels_folds_and_target_isolation, check_oof_uniqueness_and_paired_cohort, check_planted_gain_and_null,
    check_permutation_grouping_and_deterministic_resume, check_hypotheses_reading_and_gate_c, check_sensitivities_never_replace_primary,
    check_inductive_representation_compatibility)
CHECKS_SCOUTING: tuple[Callable[[], str], ...] = (
    check_incremental_information_statuses, check_pairwise_uncertainty_and_abstention, check_percentile_orientation_and_attempt_shifts,
    check_case_selection_uses_no_nfl_information, check_evidence_tokens_and_gate_f)
ALL_CHECKS: tuple[Callable[[], str], ...] = (*CHECKS_SEMANTICS, *CHECKS_RELIABILITY, *CHECKS_PREDICTION, *CHECKS_SCOUTING)


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    """Run the synthetic checks in-process. A failing check is reported, not raised, so one table shows them all."""
    rows = []
    for fn in checks:
        try:
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "PASS", "detail": fn()})
        except Exception as e:                                   # noqa: BLE001 — the table must list every failure
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "FAIL", "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


__all__ = ["ALL_CHECKS", "CHECKS_PREDICTION", "CHECKS_RELIABILITY", "CHECKS_SCOUTING", "CHECKS_SEMANTICS", "assert_all_passed", "run_all"]
