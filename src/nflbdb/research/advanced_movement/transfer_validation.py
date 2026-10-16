"""Synthetic validation of Milestone 06 (Combine → NFL transfer) — callable from a notebook cell, no subprocess, no
warehouse access.

    from nflbdb.research.advanced_movement import transfer_validation as tv
    tv.assert_all_passed(tv.run_all(tv.CHECKS_06A))      # 06A: linkage, freeze, cohort, folds, compatibility
    tv.assert_all_passed(tv.run_all(tv.CHECKS_06B))      # 06B: preprocessing, targets, OOF, metrics, bootstrap, encoders
    tv.assert_all_passed(tv.run_all(tv.CHECKS_06C))      # 06C: matched pairs, permutations, leakage

Every check builds its own small data with a KNOWN answer, writes only inside a temporary directory and raises
``AssertionError`` when a scientific contract is violated. The three checks that train or load an encoder import
PyTorch inside the function, so the other eighteen run without it. ``tests/test_advanced_movement_transfer.py`` calls
the same functions. AUTHORED, NOT EXECUTED by its author: tolerances were set by reasoning, not by running.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import polars as pl

from ..combine_nfl_dl.models import make_folds
from ..contracts import conform
from . import atlas_validation as av
from . import nfl_outcome_validation as nov
from . import nfl_outcomes as no
from . import outcome_reliability as orl
from . import robust as rb
from . import transfer as tr
from . import transfer_models as tm
from . import transfer_robust as tx
from .config import NotebookControls
from .data import LeakageError
from .fpca_eval import FORTY, HEIGHT, TEN, WEIGHT
from .validation import _raises, failure_location

TC = tr.TransferConfig(outer_folds_grid=(4, 3), min_heldout_fold=10, min_position_train=5, inner_folds=3, alpha_grid=(-1.0, 3.0, 5),
                       n_boot=300, n_perm=40, min_athletes=20, min_athletes_group=10, compact_dims=(2,))
POSITIONS = ("DB", "DL", "WR")


# ============================================================================== fixtures
def synthetic_study(n: int = 240, seed: int = 0, signal: float = 1.0) -> dict[str, Any]:
    """Athlete-level study with a KNOWN structure. Label = position effect − 6·(forty − mean) + ``signal``·trait + noise,
    where ``trait`` is unrelated to the official results. Blocks: TCN (dimension 1 = trait + noise, three noise
    dimensions), ENG (a noisier copy of the trait), SPEED2 (a copy of the forty: redundant), RAND (noise)."""
    rng = np.random.default_rng(seed)
    ids = 1000 + np.arange(n, dtype=np.int64)
    pos = np.array(POSITIONS)[np.arange(n) % 3]
    cohort = (2023 + (np.arange(n) // 3) % 3).astype(str)
    off = np.array([0.0, 0.5, -0.2])[np.arange(n) % 3]
    forty = 4.6 + off + 0.10 * rng.normal(size=n)
    trait = rng.normal(size=n)
    y = 3.0 - 1.5 * off - 6.0 * (forty - 4.6 - off) + signal * trait + 0.5 * rng.normal(size=n)
    height = 74.0 + rng.normal(size=n)
    height[::9] = np.nan
    strata = np.char.add(np.char.add(pos, "|"), cohort)
    d = tm.Design(ids, pos, cohort, {WEIGHT: 230.0 + 30 * off + 8 * rng.normal(size=n), HEIGHT: height, FORTY: forty,
                                     TEN: 1.6 + 0.3 * (forty - 4.6) + 0.02 * rng.normal(size=n)},
                  (WEIGHT, HEIGHT), (WEIGHT, HEIGHT, FORTY, TEN), (HEIGHT,), make_folds(strata, 5, 1, seed + 1)[0].astype(np.int64),
                  make_folds(strata, TC.inner_folds, 1, seed + 2)[0].astype(np.int64))
    blocks = {"TCN": np.column_stack([trait + 0.3 * rng.normal(size=n), rng.normal(size=(n, 3))]), "ENG": (trait + rng.normal(size=n))[:, None],
              "SPEED2": np.column_stack([-forty + 0.01 * rng.normal(size=n), rng.normal(size=n)]), "RAND": rng.normal(size=(n, 4)),
              "FPCA": np.column_stack([-forty + 0.02 * rng.normal(size=n), 0.5 * trait + rng.normal(size=n)]),
              "TCN_PC2": np.column_stack([trait + 0.3 * rng.normal(size=n), rng.normal(size=n)])}
    return {"d": d, "y": y, "trait": trait, "blocks": tm.ArrayBlocks(ids, blocks), "models": tm.model_ladder(TC)}


def _oof(s: dict[str, Any], models: Any = None, analysis: str = tm.PRIMARY) -> pl.DataFrame:
    res, _ = tm.run_oof(s["d"], lambda train_ids: s["y"], s["blocks"], models or s["models"], TC)
    return tm.oof_frame(analysis, tm.PRIMARY, "T", s["d"], res)


def _registry(n: int = 24) -> dict[str, pl.DataFrame]:
    """Minimal registry / mart / NFL exposure / sequence manifest with consistent identities."""
    ids = (1000 + np.arange(n)).tolist()
    pos = [POSITIONS[i % 3] for i in range(n)]
    year = [2023 + i % 3 for i in range(n)]
    athletes = pl.DataFrame({"nfl_id": ids, "display_name": [f"Athlete {i}" for i in range(n)], "position_group": pos, "draft_year": year,
                             "n_primary_eligible": [2] * n}, schema_overrides={"nfl_id": pl.Int64, "draft_year": pl.Int64, "n_primary_eligible": pl.Int64})
    exposure = pl.DataFrame({"nfl_id": ids, "nfl_position_group": pos, "nfl_cohort": year, "n_nfl_windows": [120] * n, "n_nfl_games": [10] * n,
                             "n_nfl_seasons": [1] * n, "first_season": year, "last_season": year},
                            schema_overrides={k: pl.Int64 for k in ("nfl_id", "nfl_cohort", "n_nfl_windows", "n_nfl_games", "n_nfl_seasons", "first_season", "last_season")})
    manifest = pl.DataFrame({"nfl_id": ids, "eligible_primary": [True] * n}, schema_overrides={"nfl_id": pl.Int64})
    return {"athletes": athletes, "mart": athletes.select("nfl_id"), "exposure": exposure, "manifest": manifest}


# ============================================================================== 06A
def check_athlete_linkage() -> str:
    r = _registry()
    table, bad = tr.identity_audit(r["athletes"], r["mart"], r["exposure"], r["manifest"], POSITIONS)
    conform(table, tr.IDENTITY)
    assert not bad and table["passed"].all(), table.filter(~pl.col("passed")).to_dicts()
    exp = r["exposure"].with_columns(
        pl.when(pl.col("nfl_id") == 1001).then(pl.lit("OL")).otherwise(pl.col("nfl_position_group")).alias("nfl_position_group"),
        pl.when(pl.col("nfl_id") == 1002).then(pl.lit(2020, dtype=pl.Int64)).otherwise(pl.col("nfl_cohort")).alias("nfl_cohort"),
        pl.when(pl.col("nfl_id") == 1003).then(pl.lit(2019, dtype=pl.Int64)).otherwise(pl.col("first_season")).alias("first_season"))
    stray = exp.head(1).with_columns(pl.lit(9999, dtype=pl.Int64).alias("nfl_id"))
    table, bad = tr.identity_audit(r["athletes"], r["mart"], pl.concat([exp, stray]), r["manifest"], POSITIONS)
    assert bad == {1001: "same_position_group", 1002: "same_draft_cohort", 1003: "nfl_after_combine"}, bad
    row = {x["check"]: x for x in table.iter_rows(named=True)}
    assert row["nfl_athlete_in_registry"]["n_violations"] == 1 and 9999 not in bad      # reported, cannot be linked, never guessed
    assert not row["same_position_group"]["passed"] and "1001" in row["same_position_group"]["detail"]
    return "a clean linkage passes; a position conflict, a cohort conflict and an NFL window before the draft year are excluded and listed"


def check_duplicate_athlete_detection() -> str:
    r = _registry()
    mart = pl.concat([r["mart"], r["mart"].filter(pl.col("nfl_id") == 1005)])
    athletes = r["athletes"].with_columns(pl.when(pl.col("nfl_id") == 1007).then(pl.lit("Athlete 6")).otherwise(pl.col("display_name")).alias("display_name"))
    table, bad = tr.identity_audit(athletes, mart, r["exposure"], r["manifest"], POSITIONS)
    row = {x["check"]: x for x in table.iter_rows(named=True)}
    assert bad == {1005: "unique_mart_row"} and row["unique_mart_row"]["n_violations"] == 1
    assert row["display_name_unique"]["n_violations"] == 1 and row["display_name_unique"]["severity"] == "info"
    orphan = pl.concat([r["manifest"], pl.DataFrame({"nfl_id": [4242], "eligible_primary": [True]}, schema_overrides={"nfl_id": pl.Int64})])
    _, bad = tr.identity_audit(r["athletes"], r["mart"], r["exposure"], orphan, POSITIONS)
    assert bad == {4242: "attempts_have_athlete"}
    return "a duplicated mart row and an attempt without an athlete are critical; a shared display name is reported and never used to link"


def _cohort_inputs() -> dict[str, Any]:
    """Nine athletes: 1000 complete; 1001…1007 each fail ONE rule, in funnel order; 1008 has no tracking but official + target."""
    r = _registry(9)
    ids = r["athletes"]["nfl_id"].to_list()
    athletes = r["athletes"].with_columns(pl.when(pl.col("nfl_id").is_in([1001, 1008])).then(0).otherwise(2).cast(pl.Int64).alias("n_primary_eligible"))
    official = pl.DataFrame({"nfl_id": ids, FORTY: [None if i == 1002 else 4.6 for i in ids], TEN: [1.6] * 9, WEIGHT: [230.0] * 9,
                             HEIGHT: [None if i == 1000 else 74.0 for i in ids]}, schema_overrides={"nfl_id": pl.Int64, FORTY: pl.Float64})
    tracked = [i for i in ids if i not in (1001, 1008)]
    seqs = [(f"E|{i}|{k}", i) for i in tracked for k in (1, 2)]
    feats = [("F6.x", s, i, None if i == 1003 else 1.0, "start_above_anchor" if i == 1003 else "") for s, i in seqs]
    feats += [(f, s, i, 9.0, "") for f in tr.SPEED2_FEATURES for s, i in seqs]
    attempt_values = pl.DataFrame(feats, orient="row", schema={"feature_id": pl.String, "seq_id": pl.String, "nfl_id": pl.Int64,
                                                                "value": pl.Float64, "missing_reason": pl.String})
    curve_rows = pl.DataFrame({"seq_id": [s for s, _ in seqs], "nfl_id": [i for _, i in seqs], "has_curve": [i != 1004 for _, i in seqs]},
                              schema_overrides={"nfl_id": pl.Int64})
    windows = pl.DataFrame({"nfl_id": [i for _, i in seqs], "S2_signal": [True] * len(seqs)}, schema_overrides={"nfl_id": pl.Int64})
    exposure = r["exposure"].filter(pl.col("nfl_id") != 1005)
    target = tr.FrozenTarget("O2.accel_sust05_mean", "primary", "primary", "O2", "sustained acceleration", "yd/s²", "mean", (), "none", 50, "h")
    linked = [i for i in ids if i != 1005]
    elig = pl.DataFrame({"nfl_id": linked, "target_id": [target.target_id] * 8, "rank": ["primary"] * 8, "study_role": ["primary"] * 8,
                         "n_windows": [10 if i == 1006 else 120 for i in linked], "n_games": [10] * 8, "min_windows": [50] * 8,
                         "eligible": [i != 1006 for i in linked]}, schema=tr.ELIGIBILITY.columns)
    return {"athletes": athletes, "official": official, "attempt_values": attempt_values, "curve_rows": curve_rows, "windows": windows,
            "exposure": exposure, "eligibility": elig, "target": target, "families": {"ENG": ["F6.x"], "SPEED2": list(tr.SPEED2_FEATURES)}}


def check_cohort_eligibility() -> str:
    x = _cohort_inputs()
    cohort = tr.cohort_manifest(x["athletes"], x["official"], x["attempt_values"], x["curve_rows"], x["windows"], x["exposure"],
                                x["eligibility"], x["target"], {1007: "same_position_group"}, x["families"], TC)
    conform(cohort, tr.COHORT)
    why = dict(cohort.select("nfl_id", "exclusion_reason").iter_rows())
    assert why == {1000: "", 1001: "no_tracked_forty", 1002: "no_official_core", 1003: "no_engineered_features", 1004: "no_fpca_input",
                   1005: "no_nfl_tracking", 1006: "no_target_exposure", 1007: "identity_conflict", 1008: "no_tracked_forty"}, why
    assert cohort.filter(pl.col("in_primary_cohort"))["nfl_id"].to_list() == [1000]
    wide = set(cohort.filter(pl.col("in_wide_cohort"))["nfl_id"].to_list())
    assert 1008 in wide and 1002 not in wide and 1006 not in wide and 1007 not in wide           # official + target, tracking not required
    funnel = conform(tr.funnel_table(cohort), tr.FUNNEL).filter(pl.col("group_type") == tr.ALL).sort("stage_order")
    cum = funnel["n_cumulative"].to_list()
    assert cum[0] == 9 and all(a >= b for a, b in zip(cum, cum[1:])) and cum[-1] == 1, cum
    reasons = conform(tr.reasons_table(cohort), tr.REASONS).filter(pl.col("group_type") == tr.ALL)
    assert int(reasons["n_athletes"].sum()) == 9
    conform(tr.breakdown_table(cohort), tr.COUNTS)
    missing = conform(tr.missingness_audit(x["official"], cohort, x["attempt_values"], [FORTY, TEN, WEIGHT, HEIGHT], x["families"], TC), tr.MISSING)
    design = tr.official_design(missing, TC)
    assert design["m0_numeric"] == [WEIGHT, HEIGHT] and FORTY in design["m1_numeric"] and design["indicators"] == [HEIGHT], design
    assert "cmb_vertical" in design["excluded"]                                                   # a column absent from the source is never imputed in
    eng = missing.filter((pl.col("variable") == "F6.x") & (pl.col("population") == "roster") & (pl.col("group_type") == tr.ALL)).row(0, named=True)
    assert eng["n_missing"] == 1 and "start_above_anchor" in eng["missing_class"]
    return ("each athlete leaves the funnel at the first rule he fails; the funnel is monotone and ends at the primary cohort; an athlete "
            "without tracking stays in the wide cohort; mandatory, imputed and excluded predictors follow coverage only")


def check_target_freeze_enforcement() -> str:
    c = nov.CFG
    good, second = "O2.accel_sust05_mean", "O1.speed_p90_mean"
    screen = nov._screen_fixture(c, {good: "eligible", second: "eligible"})
    dictionary, coverage = no.outcome_dictionary(c), nov._coverage_fixture(c, [good, second])
    proposal = {"decision": "FREEZE", "primary": good, "secondary": [second]}
    with tempfile.TemporaryDirectory() as tmp:
        store = SimpleNamespace(data_dir=Path(tmp) / "gold", controls=NotebookControls(run_mode="smoke"))
        freeze = lambda decision, primary=None, secondary=(), refreeze=False: orl.freeze_targets(         # noqa: E731
            store, decision, primary, secondary, "", proposal, screen, dictionary, coverage, c, {"source": "synthetic"}, refreeze)
        doc, table = tr.audit_freeze(store, "smoke")
        assert doc is None and not table["passed"].all() and _raises(tr.TargetFreezeError, lambda: tr.assert_frozen(table))
        freeze(None)                                                                             # pending review: nothing is written
        assert tr.audit_freeze(store, "smoke")[0] is None, "a pending decision must not become a registry"
        assert tr.proposal_for_review(store)["proposal"] is None                                 # a proposal is shown, never promoted
        freeze("STOP")
        doc, table = tr.audit_freeze(store, "smoke")
        assert doc["decision"] == "STOP" and _raises(tr.TargetFreezeError, lambda: tr.assert_frozen(table))
        freeze("FREEZE", good, [second], refreeze=True)
        doc, table = tr.audit_freeze(store, "smoke")
        conform(table, tr.FREEZE_AUDIT)
        assert table["passed"].all(), table.filter(~pl.col("passed")).to_dicts()
        tr.assert_frozen(table)
        targets, ocfg = tr.frozen_targets(doc)
        assert [t.target_id for t in targets] == [good, second] and tr.primary_target(targets).target_id == good
        assert targets[0].study_role == "primary" and targets[1].study_role == "secondary_speed" and ocfg.primary_min_windows == c.primary_min_windows
        assert targets[0].min_windows == c.primary_min_windows and no.definition_hash(ocfg) == no.definition_hash(c)
        failed = lambda t: set(t.filter(~pl.col("passed"))["check"].to_list())                 # noqa: E731
        assert failed(tr.audit_freeze(store, "full")[1]) == {"same_run_mode"}
        assert failed(tr.audit_freeze(store, "smoke", prior_hash="0" * 16)[1]) == {"no_refreeze_after_transfer"}
        assert not failed(tr.audit_freeze(store, "smoke", prior_hash=doc["version_hash"])[1])
        path = tr.targets_path(store)
        raw = json.loads(path.read_text(encoding="utf-8"))
        edited = json.loads(json.dumps(raw))
        edited["targets"][0]["formula"] = "something more favourable"
        path.write_text(json.dumps(edited), encoding="utf-8")
        assert "definition_hashes_reproduce" in failed(tr.audit_freeze(store, "smoke")[1])
        edited = json.loads(json.dumps(raw))
        edited["targets"] = edited["targets"][::-1]                                              # the secondary promoted to first place
        path.write_text(json.dumps(edited), encoding="utf-8")
        assert "version_hash_reproduces" in failed(tr.audit_freeze(store, "smoke")[1])
        edited = json.loads(json.dumps(raw))
        edited["selection"]["combine_data_consulted"] = True
        path.write_text(json.dumps(edited), encoding="utf-8")
        assert "frozen_without_combine_data" in failed(tr.audit_freeze(store, "smoke")[1])
    return ("an absent, pending or STOP registry blocks the study; a FREEZE written by notebook 05B passes; an edited formula, a "
            "re-ordered target list, another run mode or a re-freeze after a transfer run are each detected; nothing is defaulted")


def _fold_cohort(n: int = 180, n_extra: int = 30) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(cohort frame, split registry): ``n`` primary athletes, ``n_extra`` tracked athletes without NFL data, and six
    wide-only athletes that are not in the registry."""
    ids = 1000 + np.arange(n + n_extra + 6)
    pos, year = np.array(POSITIONS)[ids % 3], 2023 + (ids // 3) % 3
    prim = np.arange(ids.size) < n
    wide = prim | (np.arange(ids.size) >= n + n_extra)
    cohort = pl.DataFrame({"nfl_id": ids, "position_group": pos, "draft_year": year, "in_primary_cohort": prim, "in_wide_cohort": wide,
                           "st_target": wide}, schema_overrides={"nfl_id": pl.Int64, "draft_year": pl.Int64})
    tracked = np.arange(ids.size) < n + n_extra
    strata = np.char.add(np.char.add(pos[tracked], "|"), year[tracked].astype(str))
    k = int(tracked.sum())
    registry = pl.DataFrame({"nfl_id": ids[tracked], "repeat": np.zeros(k, dtype=np.int64), "outer_fold": make_folds(strata, 5, 1, 3)[0],
                             "inner_fold": make_folds(strata, 4, 1, 4)[0], "stratum": strata, "design_role": ["confirm"] * k},
                            schema_overrides={"nfl_id": pl.Int64, "repeat": pl.Int64, "outer_fold": pl.Int64, "inner_fold": pl.Int64})
    return cohort, registry


def check_athlete_grouped_splitting() -> str:
    cohort, registry = _fold_cohort()
    manifest = pl.DataFrame({"nfl_id": np.repeat(registry["nfl_id"].to_numpy(), 2), "eligible_primary": np.ones(2 * registry.height, dtype=bool)},
                            schema_overrides={"nfl_id": pl.Int64})
    folds, plan = tr.assign_folds(cohort, registry, TC)
    conform(folds, tr.FOLDS), conform(plan, tr.FOLD_PLAN)
    assert folds["fold_source"][0] == "registry" and plan.filter(pl.col("selected")).height == 1
    assert folds.height == 186 and folds["nfl_id"].n_unique() == 186 and folds["outer_fold"].n_unique() == 5
    reg = dict(registry.select("nfl_id", "outer_fold").iter_rows())
    assert all(reg[i] == f for i, f in folds.select("nfl_id", "outer_fold").iter_rows() if i in reg)     # registry folds kept as they are
    enc = tr.encoder_splits(folds, registry)
    assert enc.equals(registry) and tr.assert_fold_integrity(folds, enc, manifest)
    new, plan2 = tr.assign_folds(cohort, registry, TC, source="new")
    assert new["fold_source"][0] == "new_4" and new["n_folds"][0] == 4 and not plan2.filter(pl.col("candidate") == "registry")["selected"][0]
    prim = new.filter(pl.col("in_primary_cohort"))
    per = prim.group_by("stratum", "outer_fold").len()
    assert per["len"].max() - per["len"].min() <= 1, "folds must be stratified by position × draft cohort"
    enc = tr.encoder_splits(new, registry)
    e = dict(enc.select("nfl_id", "outer_fold").iter_rows())
    assert all(e[i] == f for i, f in new.select("nfl_id", "outer_fold").iter_rows() if i in e)
    outside = set(registry["nfl_id"].to_list()) - set(new["nfl_id"].to_list())
    assert outside and all(e[i] == -1 for i in outside), "tracked athletes without NFL data are never held out under new folds"
    tr.assert_fold_integrity(new, enc, manifest)
    moved = enc.with_columns(pl.when(pl.col("nfl_id") == 1000).then((pl.col("outer_fold") + 1) % 4).otherwise(pl.col("outer_fold")).alias("outer_fold"))
    assert _raises(tr.TransferGateError, lambda: tr.assert_fold_integrity(new, moved, manifest))
    assert tr.folds_hash(folds) != tr.folds_hash(new) and tr.folds_hash(folds) == tr.folds_hash(tr.assign_folds(cohort, registry, TC)[0])
    small = cohort.with_columns((pl.col("in_primary_cohort") & (pl.col("nfl_id") < 1024)).alias("in_primary_cohort"))
    assert _raises(tr.FoldError, lambda: tr.assign_folds(small, registry, TC))
    return ("one outer fold per athlete; registry folds are reused when viable, otherwise the largest viable K is dealt within position × "
            "cohort; unlinked athletes are never held out under new folds; a too-small cohort raises instead of producing folds")


def check_checkpoint_compatibility() -> str:
    cohort, registry = _fold_cohort()
    folds, _ = tr.assign_folds(cohort, registry, TC)
    reg = registry.select("nfl_id", "outer_fold")
    fit_of = lambda f: set(reg.filter(pl.col("outer_fold") != f)["nfl_id"].to_list())             # noqa: E731
    held0 = folds.filter(pl.col("outer_fold") == 0)["nfl_id"].to_list()
    records = [{"artifact": "tcn/A__r0_f0.pt", "family": "tcn", "method": "TCNB_A", "repeat": 0, "fold": 0, "athletes": fit_of(0)},
               {"artifact": "tcn/A__r0_f1.pt", "family": "tcn", "method": "TCNB_A", "repeat": 0, "fold": 1, "athletes": fit_of(1) | set(held0[:3])},
               {"artifact": "tcn/A__r1_f0.pt", "family": "tcn", "method": "TCNB_A", "repeat": 1, "fold": 0, "athletes": fit_of(0)},
               {"artifact": "fpca/A.npz", "family": "fpca", "method": "FPCA:reference", "repeat": -1, "fold": -1, "athletes": set(reg["nfl_id"].to_list())}]
    compat = conform(tr.compatibility_table(records, folds), tr.COMPAT)
    st = {(r["method"], r["repeat"], r["fold"]): r for r in compat.iter_rows(named=True)}
    assert st[("TCNB_A", 0, 0)]["status"] == "compatible" and st[("TCNB_A", 0, 0)]["n_overlap"] == 0
    leaked = fit_of(1) & set(folds.filter(pl.col("outer_fold") == 1)["nfl_id"].to_list())
    assert not leaked and st[("TCNB_A", 0, 1)]["status"] == "compatible"                         # athletes of fold 0 are training athletes of fold 1
    records[1]["athletes"] = fit_of(1) | set(folds.filter(pl.col("outer_fold") == 1)["nfl_id"].to_list()[:3])
    st = {(r["method"], r["repeat"], r["fold"]): r for r in tr.compatibility_table(records, folds).iter_rows(named=True)}
    assert st[("TCNB_A", 0, 1)]["status"] == "descriptive_only" and st[("TCNB_A", 0, 1)]["n_overlap"] == 3
    assert st[("TCNB_A", 1, 0)]["status"] == "descriptive_only" and st[("FPCA:reference", -1, -1)]["status"] == "descriptive_only"
    assert tr.reusable_checkpoints(tr.compatibility_table(records, folds), "TCNB_A") == {0: "tcn/A__r0_f0.pt"}
    new, _ = tr.assign_folds(cohort, registry, TC, source="new")
    renewed = tr.compatibility_table(records, new)
    assert not renewed.filter(pl.col("status") == "compatible").height, "a new fold design makes every saved fit descriptive-only"
    assert renewed.filter(pl.col("status") == "combine_only_selection").height == len(tr.SELECTIONS)
    gate = tr.gate_06a(pl.DataFrame({"check": ["x"], "passed": [True], "detail": [""]}), pl.DataFrame(schema=tr.IDENTITY.columns),
                       cohort.with_columns(pl.lit(True).alias("identity_ok")), folds, tr.assign_folds(cohort, registry, TC)[1], compat, TC)
    conform(gate, tr.GATE)
    assert gate["passed"].all(), gate.to_dicts()
    blocked = tr.gate_06a(pl.DataFrame({"check": ["x"], "passed": [False], "detail": ["pending"]}), pl.DataFrame(schema=tr.IDENTITY.columns),
                          cohort.with_columns(pl.lit(True).alias("identity_ok")), folds, tr.assign_folds(cohort, registry, TC)[1], compat, TC)
    assert _raises(tr.TransferGateError, lambda: tr.assert_gate(blocked, "06A"))
    return ("a saved fit is reusable only for its own fold of split repeat 0 and only if no athlete held out there took part in it; "
            "all-athlete fits, other repeats and every fit under new folds are descriptive-only; an unfrozen registry fails gate 06A")


def check_artifact_version_consistency() -> str:
    cohort, registry = _fold_cohort()
    folds, _ = tr.assign_folds(cohort, registry, TC)
    doc = {"version_hash": "abc"}
    manifest = {"gate_06a": "PASS", "targets_version_hash": "abc", "folds_hash": tr.folds_hash(folds)}
    store = SimpleNamespace(read_json=lambda name, notebook: dict(manifest))
    assert tr.require_audit(store, doc, folds)["gate_06a"] == "PASS"
    assert _raises(tr.TransferGateError, lambda: tr.require_audit(store, {"version_hash": "other"}, folds))        # targets re-frozen since 06A
    other, _ = tr.assign_folds(cohort, registry, TC, source="new")
    assert _raises(tr.TransferGateError, lambda: tr.require_audit(store, doc, other))                               # folds replaced since 06A
    manifest["gate_06a"] = "FAIL"
    assert _raises(tr.TransferGateError, lambda: tr.require_audit(store, doc, folds))
    entry = tr.describe_input("x", None, pl.DataFrame({"a": [1]}), ["a", "b"], "test")
    assert entry["missing_columns"] == ["b"] and _raises(tr.TransferGateError, lambda: tr.assert_inputs([entry]))
    return "06B and 06C refuse to start when the frozen targets, the fold table or the 06A gate differ from what the cohort audit recorded"


# ============================================================================== 06B
def check_repeat_attempt_aggregation() -> str:
    rng = np.random.default_rng(0)
    A = 60
    truth = rng.normal(size=(A, 2)) * 2.0
    n_att = np.where(np.arange(A) % 4 == 0, 1, 2)
    nfl = np.repeat(np.arange(A), n_att)
    order = np.concatenate([np.arange(1, k + 1)[::-1] for k in n_att])                           # rows are NOT in time order
    Z = truth[nfl] + 0.8 * rng.normal(size=(nfl.size, 2))
    train = nfl < 45
    ids, mean, n = tm.aggregate_attempts(Z, nfl, order, train, "mean")
    assert np.array_equal(ids, np.arange(A)) and np.array_equal(n, n_att)
    assert np.allclose(mean[1], Z[nfl == 1].mean(0)) and np.allclose(mean[0], Z[nfl == 0][0])
    _, first, _ = tm.aggregate_attempts(Z, nfl, order, train, "first")
    assert np.allclose(first[1], Z[(nfl == 1) & (order == 1)][0])                                # earliest attempt, whatever the row order
    _, shrunk, _ = tm.aggregate_attempts(Z, nfl, order, train, "shrunk")
    centre = mean[np.unique(nfl[train])].mean(0)
    pull = np.abs(shrunk - centre).sum(1) / np.abs(mean - centre).sum(1)
    assert pull.max() <= 1.0 + 1e-9 and pull[n_att == 1].mean() < pull[n_att == 2].mean()        # one attempt is trusted less than two
    Z2 = Z.copy()
    Z2[~train] += 50.0                                                                            # held-out attempts change …
    _, shrunk2, _ = tm.aggregate_attempts(Z2, nfl, order, train, "shrunk")
    assert np.allclose(shrunk2[:45], shrunk[:45])                                                # … training athletes do not move
    assert _raises(ValueError, lambda: tm.aggregate_attempts(Z, nfl, order, train, "best_attempt"))
    return ("the athlete mean, the first attempt in time and the reliability-weighted mean are exact; an athlete with one attempt is "
            "shrunk more; shrinkage parameters come from training athletes only; no rule can read an NFL value")


def check_train_only_preprocessing() -> str:
    s = synthetic_study()
    d = s["d"]
    spec = next(m for m in s["models"] if m.model == "M5")
    train = d.fold != 0
    rep = [("TCN", s["blocks"].matrix("TCN", 0, d.ids))]
    X, names, pen, prep = tm.design_matrix(d, spec, rep, train)
    assert np.allclose(X[train].mean(0), 0.0, atol=1e-9) and np.allclose(X[train].std(0), 1.0, atol=1e-9)
    assert not np.allclose(X[~train].mean(0), 0.0, atol=1e-6)                                    # held-out rows are scaled, not re-centred
    assert f"{HEIGHT}__missing" in names and "TCN_4" in names and len(names) == X.shape[1] == len(prep)
    assert [p for n_, p in zip(names, pen) if n_.startswith("position=")] == [0.0, 0.0] and pen[names.index(FORTY)] == 1.0
    med = {r["term"]: r["median"] for r in prep}[HEIGHT]
    assert np.isclose(med, np.nanmedian(d.numeric[HEIGHT][train]))
    j = names.index(HEIGHT)
    gap = ~np.isfinite(d.numeric[HEIGHT])
    assert np.allclose(X[gap, j], X[gap & train, j][0])                                          # one training median for every gap
    moved = tm.Design(d.ids, d.position, d.cohort, {k: np.where(train, v, v * 3.0 + 7.0) for k, v in d.numeric.items()}, d.m0, d.m1,
                      d.indicators, d.fold, d.inner)
    rep2 = [("TCN", np.where(train[:, None], rep[0][1], rep[0][1] * 5.0 - 2.0))]
    X2, names2, _, prep2 = tm.design_matrix(moved, spec, rep2, train)
    assert names2 == names and np.allclose(X2[train], X[train]) and prep2 == prep                # held-out values cannot move a statistic
    assert not np.allclose(X2[~train], X[~train])
    y = s["y"]
    a1 = tm.select_alpha(d, spec, rep, y, train, TC)
    y2 = np.where(train, y, y + 100.0)
    assert a1 == tm.select_alpha(d, spec, rep, y2, train, TC), "the penalty must be chosen inside the training athletes"
    assert a1 in TC.alphas
    bad = [("TCN", np.where(np.arange(d.n)[:, None] == 3, np.nan, rep[0][1]))]
    assert _raises(tm.CohortError, lambda: tm.design_matrix(d, spec, bad, train))                # an embedding is never imputed
    assert _raises(tm.CohortError, lambda: tm.assert_pre_nfl_columns(["cmb_forty", "nfl_reg_plays"]))
    return ("medians, centres, scales, dummy levels and the ridge penalty depend on training athletes only: changing every held-out "
            "value leaves them identical; position effects are unpenalised; a missing embedding raises instead of being imputed")


def check_fold_target_handling() -> str:
    c = nov.CFG
    plays, _ = nov.synthetic_plays(c, n_athletes=90, n_games=8, n_windows=8, seed=5, align_effect=2.0)
    adj = tr.FrozenTarget(nov.MEAN_ID, "primary", "primary", "O1", "play p90 speed", "yd/s", "mean", (), "cell_residual", c.primary_min_windows, "h")
    p90 = tr.FrozenTarget("O1.speed_sust05_p90", "secondary", "secondary_speed", "O1", "sustained speed", "yd/s", "p90", (), "none", c.primary_min_windows, "h")
    comp = [f"O5.band_gain_{no.blab(lo, hi)}_p90" for lo, hi in c.speed_bands]
    env = tr.FrozenTarget("O5.envelope_level", "secondary", "exploratory_envelope", "O5", "envelope", "yd/s", "mean", tuple(comp), "none", c.primary_min_windows, "h")
    tset = tr.TargetSet(plays, c, [adj, p90, env])
    ref = no.athlete_outcomes(tset.eng)
    for t, col in ((adj, "value_adj"), (p90, "value"), (env, "value")):
        want = ref.filter(pl.col("outcome_id") == t.target_id).sort("nfl_id")
        got = tset.athlete_values(t.target_id)
        assert np.array_equal(want["nfl_id"].to_numpy(), got["ids"]) and np.allclose(want[col].to_numpy(), got["value"], atol=1e-9), t.target_id
        assert np.array_equal(want["n_windows"].to_numpy(), got["n_windows"])
    raw = tset.athlete_values(adj.target_id, adjustment="none")["value"]
    assert not np.allclose(raw, tset.athlete_values(adj.target_id)["value"])                     # the alignment effect is in the raw label
    ids = tset.ids
    train = ids[: 60]
    base = tset.athlete_values(adj.target_id, train)["value"]
    moved = plays.with_columns(pl.when(pl.col("nfl_id").is_in(ids[60:].tolist())).then(pl.col("speed_p90") + 25.0).otherwise(pl.col("speed_p90")).alias("speed_p90"))
    again = tr.TargetSet(moved, c, [adj]).athlete_values(adj.target_id, train)["value"]
    assert np.allclose(again[:60], base[:60]), "labels of training athletes must not depend on held-out athletes' plays"
    assert np.allclose(again[60:] - base[60:], 25.0)                                             # the held-out athlete keeps his own value
    assert not np.allclose(tset.athlete_values(adj.target_id, ids[30:])["value"], base)          # another training set, other context means
    first = tset.athlete_values(adj.target_id, variant="first_n:10", adjustment="none")
    assert first["n_windows"].max() == 10 and first["n_games"].max() <= 2
    rookie = tset.athlete_values(adj.target_id, variant="horizon:1", adjustment="none")
    assert rookie["n_windows"].max() < tset.athlete_values(adj.target_id)["n_windows"].max()
    assert tset.athlete_values(adj.target_id, variant="class:pass", adjustment="none")["n_windows"].sum() < plays.height
    assert _raises(ValueError, lambda: tset.athlete_values(adj.target_id, variant="best_games"))
    elig = conform(tset.eligibility(), tr.ELIGIBILITY)
    assert elig.height == 3 * 90 and "value" not in elig.columns and elig.filter(pl.col("target_id") == adj.target_id)["eligible"].all()
    conform_cols = set(tset.exposure().columns)
    assert {"n_nfl_windows", "n_nfl_games", "first_season", "nfl_position_group"} <= conform_cols
    one = tr.TargetSet(plays.with_columns(pl.when(pl.col("nfl_id") == int(ids[0])).then(None).otherwise(pl.col(f"band_gain_{no.blab(*c.speed_bands[-1])}"))
                                         .alias(f"band_gain_{no.blab(*c.speed_bands[-1])}")), c, [env]).athlete_values(env.target_id)
    assert not np.isfinite(one["value"][0]) and one["n_windows"][0] == 0 and np.isfinite(one["value"][1:]).all()
    return ("with all athletes the labels equal notebook 05's athlete outcomes (raw, context-adjusted, p90 and composite); with a "
            "training set the context means come from it alone, so held-out plays cannot move a training label; exposure rules keep "
            "time order; an athlete lacking a component has no composite label")


def check_oof_uniqueness() -> str:
    s = synthetic_study()
    res, rows = tm.run_oof(s["d"], lambda train_ids: s["y"], s["blocks"], s["models"], TC)
    oof = conform(tm.oof_frame(tm.PRIMARY, tm.PRIMARY, "T", s["d"], res), tm.OOF)
    assert oof.height == len(s["models"]) * s["d"].n and oof.select(pl.struct("model", "nfl_id").n_unique()).item() == oof.height
    assert np.isfinite(res["y"]).all() and all(np.isfinite(p).all() for p in res["pred"].values())
    m1 = oof.filter(pl.col("model") == "M1").sort("nfl_id")
    assert np.array_equal(m1["fold"].to_numpy(), s["d"].fold) and (m1["n_train"] == (s["d"].n - m1.group_by("fold").len().sort("fold")["len"][0])).any()
    mean = oof.filter(pl.col("model") == "MEAN")
    assert np.allclose(mean["pred"].to_numpy(), mean["y_ref"].to_numpy())                        # the sanity model IS the training-fold mean
    for f in np.unique(s["d"].fold):
        assert np.isclose(res["y_ref"][s["d"].fold == f][0], s["y"][s["d"].fold != f].mean())
    prep = conform(tm.prep_frame(tm.PRIMARY, tm.PRIMARY, "T", rows), tm.PREP)
    assert prep.filter(pl.col("model") == "M5")["block"].n_unique() >= 5 and prep.filter(pl.col("block") == "position")["penalised"].not_().all()
    broken = dict(res, pred={**res["pred"], "M5": np.where(np.arange(s["d"].n) == 0, np.nan, res["pred"]["M5"])})
    assert _raises(tm.CohortError, lambda: tm.oof_frame(tm.PRIMARY, tm.PRIMARY, "T", s["d"], broken))
    dup = pl.concat([oof, oof.filter((pl.col("model") == "M5") & (pl.col("nfl_id") == 1000))])
    assert _raises(tm.CohortError, lambda: tm.paired_arrays(dup))
    assert _raises(tm.CohortError, lambda: tm.run_oof(s["d"], lambda t: np.where(np.arange(s["d"].n) == 5, np.nan, s["y"]), s["blocks"], s["models"], TC))
    return ("every athlete gets exactly one prediction per model, from the fold that held him out; the sanity model equals the "
            "training-fold mean; a missing or duplicated prediction and a missing label raise")


def check_paired_model_cohorts() -> str:
    s = synthetic_study()
    oof = _oof(s)
    a = tm.paired_arrays(oof)
    assert set(a["pred"]) == {m.model for m in s["models"]} and a["ids"].size == s["d"].n
    fewer = oof.filter(~((pl.col("model") == "M5") & (pl.col("nfl_id") < 1010)))
    assert _raises(tm.CohortError, lambda: tm.paired_arrays(fewer)), "a model with fewer missing values must not get its own cohort"
    assert _raises(tm.CohortError, lambda: tm.evaluate(fewer, tm.comparisons(TC), TC, 1))
    shifted = oof.with_columns(pl.when(pl.col("model") == "M5").then(pl.col("y") + 0.5).otherwise(pl.col("y")).alias("y"))
    assert _raises(tm.CohortError, lambda: tm.paired_arrays(shifted)), "models must be scored against identical labels"
    d = s["d"]
    short = tm.ArrayBlocks(d.ids[5:], {"TCN": s["blocks"].blocks["TCN"][5:]})
    assert _raises(KeyError, lambda: short.matrix("TCN", 0, d.ids))
    gate = tm.gate_06b(oof, tm.evaluate(oof, tm.comparisons(TC), TC, 1)[tm.PAIRED.name], pl.DataFrame({"n_overlap": [0, 0]}),
                       {"ENG": ["a"], "SPEED2": ["a", "b"]}, {"ENG": 1, "SPEED2": 2, "FPCA": 2, "TCN": 4}, "T")
    assert gate["passed"].all(), gate.to_dicts()
    leak = tm.gate_06b(fewer, tm.PAIRED.empty(), pl.DataFrame({"n_overlap": [0, 2]}), {"ENG": ["a"], "SPEED2": ["a", "b"]},
                       {"ENG": 3, "SPEED2": 2, "FPCA": 2, "TCN": 4}, "T")
    failed = set(leak.filter(~pl.col("passed"))["criterion"].to_list())
    assert {"identical_heldout_athletes", "frozen_feature_candidates_used", "inductive_representation_fitting", "uncertainty_reported"} <= failed
    return ("paired comparisons require identical athletes AND identical labels for every model; a smaller cohort for one model, an "
            "extra engineered feature or a shared athlete between an encoder fit and its held-out fold each fail gate 06B")


def check_regression_metrics() -> str:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ref = np.full(6, 3.0)
    m = tm.point_metrics(y, y, ref)
    assert m["rmse"] == 0 and m["mae"] == 0 and np.isclose(m["r2_oos"], 1) and np.isclose(m["spearman"], 1) and np.isclose(m["calibration_slope"], 1)
    m = tm.point_metrics(y, ref, ref)
    assert np.isclose(m["r2_oos"], 0) and np.isnan(m["spearman"]) and np.isclose(m["rmse"], np.sqrt(np.mean((y - 3.0) ** 2)))
    p = np.array([2.0, 2.0, 4.0, 4.0, 6.0, 6.0])
    m = tm.point_metrics(y, p, ref)
    assert np.isclose(m["rmse"], np.sqrt(0.5)) and np.isclose(m["mae"], 0.5) and np.isclose(m["r2_oos"], 1 - 3.0 / ((y - 3.0) ** 2).sum())
    m = tm.point_metrics(y, 3.5 + 2.0 * (y - 3.5), ref)                                           # predictions twice too spread out
    assert np.isclose(m["calibration_slope"], 0.5) and np.isclose(m["pearson"], 1.0) and np.isclose(m["calibration_intercept"], 1.75)
    assert np.isclose(tm.point_metrics(y, -y, ref)["spearman"], -1.0)
    assert tm.point_metrics(y, y + 10.0, ref)["r2_oos"] < 0                                       # worse than the training mean is negative
    b0, beta = tm.ridge_solve(np.column_stack([y, y ** 2]), 2.0 + 3.0 * y, 0.0, np.ones(2))
    assert np.allclose([b0, *beta], [2.0, 3.0, 0.0], atol=1e-4)
    _, shrunk = tm.ridge_solve(y[:, None], 3.0 * y, 1e6, np.ones(1))
    _, free = tm.ridge_solve(y[:, None], 3.0 * y, 1e6, np.zeros(1))
    assert abs(shrunk[0]) < 0.01 and np.isclose(free[0], 3.0, atol=1e-4)                           # an unpenalised term is not shrunk
    _, lasso = tm.enet_solve(np.column_stack([y, np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])]), 3.0 * y, 2.0, np.ones(2), TC)
    assert lasso[1] == 0.0 and 2.0 < lasso[0] < 3.0
    return "RMSE, MAE, R² against the training mean, Spearman, Pearson and calibration reproduce hand-computed values; ridge and elastic net recover a known line"


def check_delta_rmse_sign() -> str:
    s = synthetic_study(signal=1.0)
    ev = tm.evaluate(_oof(s), tm.comparisons(TC), TC, seed=3)
    perf, paired = conform(ev[tm.PERF.name], tm.PERF), conform(ev[tm.PAIRED.name], tm.PAIRED)
    rmse = {r["model"]: r["estimate"] for r in perf.filter(pl.col("metric") == "rmse").iter_rows(named=True)}
    row = lambda m, r, k="delta_rmse": paired.filter((pl.col("model") == m) & (pl.col("reference") == r) & (pl.col("metric") == k)).row(0, named=True)   # noqa: E731
    d51 = row("M5", "M1")
    assert np.isclose(d51["estimate"], rmse["M1"] - rmse["M5"]) and d51["estimate"] > 0 and d51["established"] and not d51["worse"]
    assert np.isclose(d51["relative"], d51["estimate"] / rmse["M1"]) and d51["ci_lower"] < d51["estimate"] < d51["ci_upper"]
    assert row("M5", "M1", "delta_r2_oos")["estimate"] > 0 and row("M5", "M1", "delta_spearman")["estimate"] > 0     # same orientation everywhere
    assert rmse["M0"] < rmse["MEAN"] and rmse["M1"] < rmse["M0"]                                  # the planted ladder is recovered
    assert not row("M6", "M1")["established"] and row("M6", "M1")["estimate"] < 0.02              # a random block does not help
    assert not row("M3", "M1")["established"]                                                     # a copy of the forty adds nothing to M1
    assert row("M5", "M6")["established"] and row("M5", "M3")["established"] and row("M2", "M1")["estimate"] > 0
    assert row("M5S", "M3")["established"], "the encoder must still help once SPEED2 is in the model"
    null = synthetic_study(signal=0.0, seed=4)
    d = tm.evaluate(_oof(null), [("M5", "M1")], TC, seed=3)[tm.PAIRED.name].filter(pl.col("metric") == "delta_rmse").row(0, named=True)
    assert not d["established"], "without a planted trait no gain may be established"
    return (f"Δ RMSE = RMSE(reference) − RMSE(model): positive for the planted trait ({d51['estimate']:+.3f}, {100 * d51['relative']:.0f} %), "
            "about zero for a random block and for a copy of the forty, not established when the trait is removed")


def check_athlete_bootstrap() -> str:
    s = synthetic_study()
    oof = _oof(s)
    a = tm.evaluate(oof, [("M5", "M1")], TC, seed=11)
    b = tm.evaluate(oof, [("M5", "M1")], TC, seed=11)
    c = tm.evaluate(oof, [("M5", "M1")], TC, seed=12)
    assert a[tm.PAIRED.name].equals(b[tm.PAIRED.name]) and not a[tm.PAIRED.name].equals(c[tm.PAIRED.name])       # reproducible, seed-dependent
    pa = tm.paired_arrays(oof)
    n = pa["ids"].size
    idx = np.random.default_rng(0).integers(0, n, size=(400, n))
    assert idx.shape == (400, n) and idx.max() < n, "a resample draws ATHLETES, with replacement, as many as the cohort holds"
    bm = tm.boot_metrics(pa["y"], pa["pred"]["M5"], pa["y_ref"], idx)
    assert all(v.shape == (400,) for v in bm.values()) and abs(bm["rmse"].mean() - tm.point_metrics(pa["y"], pa["pred"]["M5"], pa["y_ref"])["rmse"]) < 0.05
    diff = bm["rmse"] - tm.boot_metrics(pa["y"], pa["pred"]["M1"], pa["y_ref"], idx)["rmse"]
    assert diff.std() < bm["rmse"].std() + tm.boot_metrics(pa["y"], pa["pred"]["M1"], pa["y_ref"], idx)["rmse"].std(), "paired resamples share athletes"
    small = oof.filter(pl.col("nfl_id") < 1000 + TC.min_athletes_group - 2)
    few = tm.evaluate(small, [("M5", "M1")], TC, seed=1)[tm.PAIRED.name].filter(pl.col("metric") == "delta_rmse").row(0, named=True)
    assert few["estimate"] is not None and few["ci_lower"] is None and not few["established"]     # too few athletes: an estimate, no interval
    wide = tm.evaluate(oof, [("M5", "M1")], TC, seed=1, n_boot=60)[tm.PAIRED.name].filter(pl.col("metric") == "delta_rmse").row(0, named=True)
    assert wide["ci_lower"] is not None
    stab = conform(tx.ci_stability(oof, ("M5", "M1"), TC, 5, budgets=(100, 400), n_seeds=3), tx.STABILITY)
    assert stab.height == 6 and (stab["ci_lower"] < stab["estimate"]).all() and (stab["estimate"] < stab["ci_upper"]).all()
    sub = tx.subgroup_tables(oof, "position_group", [("M5", "M1")], TC, 5)
    conform(sub[tx.SUB_PERF.name], tx.SUB_PERF), conform(sub[tx.SUB_PAIRED.name], tx.SUB_PAIRED)
    assert sub[tx.SUB_PAIRED.name]["group"].n_unique() == 3 and int(sub[tx.SUB_PAIRED.name].filter(pl.col("metric") == "delta_rmse")["n_athletes"].sum()) == n
    het = conform(tx.heterogeneity(oof, "position_group", [("M5", "M1")], TC, 5), tx.HETERO)
    assert het.height == 1 and 0 < het["p_value"][0] <= 1
    return ("athletes — never plays or attempts — are resampled, with the same indices for both models of a comparison; intervals are "
            "reproducible for a seed and absent below the minimum cohort size; subgroup tables cover every athlete once")


def check_fpca_fold_isolation() -> str:
    from . import fpca_validation as fv
    from . import transfer_repr as trp                                                           # imports PyTorch (the TCN module)

    cs, splits, plan = _repr_fixture(trp, fv)
    scal = {"SPEED2": np.column_stack([cs.Y[:, 10], cs.Y[:, 40]])}
    scal["SPEED2"][3] = np.nan
    fpca_only = trp.RepPlan(plan.tcfg, plan.fcfg, 2, plan.seed, families=("SPEED2", "FPCA"))
    out = trp.fold_representations(cs, splits, [0, 1, 2, 3], fpca_only, scal)
    rep, fits = conform(out[trp.ATTEMPT_REP.name], trp.ATTEMPT_REP), conform(out[trp.ENC_FITS.name], trp.ENC_FITS)
    assert fits["n_overlap"].sum() == 0 and set(fits["source"].to_list()) == {"deterministic", "fitted"}
    assert not rep.filter((pl.col("family") == "SPEED2") & (pl.col("seq_id") == str(cs.seq_ids[3]))).height    # an unmeasured attempt has no row
    held = rep.filter((pl.col("family") == "FPCA") & (pl.col("role") == "heldout"))
    assert held.group_by("seq_id").agg(pl.col("fold").n_unique().alias("k"))["k"].max() == 1 and held["nfl_id"].n_unique() == 60
    test0 = splits.filter(pl.col("outer_fold") == 0)["nfl_id"].to_numpy()
    Y2 = cs.Y.copy()
    Y2[np.isin(cs.nfl_ids, test0)] *= 3.0                                                         # only held-out curves change
    moved = trp.fold_representations(type(cs)(cs.spec, cs.grid, Y2, cs.t_orig, cs.seq_ids, cs.nfl_ids), splits, [0], fpca_only, scal)[trp.ATTEMPT_REP.name]
    pick = lambda t, role: t.filter((pl.col("family") == "FPCA") & (pl.col("fold") == 0) & (pl.col("role") == role)).sort("seq_id", "dim")["value"].to_numpy()   # noqa: E731
    assert np.allclose(pick(moved, "train"), pick(rep, "train")), "training scores must not depend on held-out curves"
    assert not np.allclose(pick(moved, "heldout"), pick(rep, "heldout"))
    blocks = tm.RepBlocks(rep, {str(q): int(str(q).rsplit("|", 1)[1]) for q in cs.seq_ids})
    ids, X, n = blocks.athlete("FPCA", 0)
    assert X.shape == (60, 2) and (n == 2).all() and np.array_equal(ids, np.unique(cs.nfl_ids))
    gaps = blocks.available(["SPEED2", "FPCA"], [0, 1, 2, 3])
    assert all(not v for v in gaps.values())                                                     # the athlete of attempt 3 keeps his other attempt
    assert _raises(tm.CohortError, lambda: blocks.matrix("FPCA", 0, np.array([1, 2, 3])))
    return "FPCA support, mean and eigenfunctions of a fold come from its training athletes: tripling the held-out curves leaves every training score unchanged"


def _repr_fixture(trp: Any, fv: Any) -> tuple[Any, pl.DataFrame, Any]:
    from . import tcn_validation as tv3

    cs, _ = fv.synthetic_curves(n_athletes=60, seed=2)
    ids = np.unique(cs.nfl_ids)
    splits = pl.DataFrame({"nfl_id": ids, "repeat": np.zeros(ids.size, dtype=np.int64), "outer_fold": np.arange(ids.size) % 4,
                           "inner_fold": (np.arange(ids.size) // 4) % 3, "stratum": ["S"] * ids.size, "design_role": ["confirm"] * ids.size},
                          schema_overrides={"nfl_id": pl.Int64, "repeat": pl.Int64, "outer_fold": pl.Int64, "inner_fold": pl.Int64})
    return cs, splits, trp.RepPlan(tv3.TINY_B, fv.FCFG, 2, seed=5, families=("SPEED2", "FPCA", "TCN", "RAND"), compact_dims=(2,))


def check_unseen_athlete_embedding() -> str:
    from . import fpca_validation as fv
    from . import transfer_repr as trp

    cs, splits, plan = _repr_fixture(trp, fv)
    with tempfile.TemporaryDirectory() as tmp:
        out = trp.fold_representations(cs, splits, [0, 1], plan, {}, models_dir=Path(tmp))
        rep, fits = conform(out[trp.ATTEMPT_REP.name], trp.ATTEMPT_REP), conform(out[trp.ENC_FITS.name], trp.ENC_FITS)
        assert set(rep["family"].unique().to_list()) == {"FPCA", "TCN", "TCN_PC2", "RAND"} and fits["n_overlap"].sum() == 0
        tcn_fit = fits.filter(pl.col("family") == "TCN")
        assert tcn_fit["source"].to_list() == ["trained", "trained"] and all(Path(p).is_file() for p in tcn_fit["checkpoint"].to_list())
        for f in (0, 1):
            held = set(splits.filter(pl.col("outer_fold") == f)["nfl_id"].to_list())
            g = rep.filter((pl.col("family") == "TCN") & (pl.col("fold") == f))
            assert set(g.filter(pl.col("role") == "heldout")["nfl_id"].to_list()) == held       # unseen athletes ARE embedded …
            assert not set(g.filter(pl.col("role") == "train")["nfl_id"].to_list()) & held       # … and are not training rows
            assert g["dim"].n_unique() == plan.tcfg.embedding_dim and rep.filter((pl.col("family") == "TCN_PC2") & (pl.col("fold") == f))["dim"].n_unique() == 2
        reuse = {("TCN", 0): tcn_fit.filter(pl.col("fold") == 0)["checkpoint"][0]}
        again = trp.fold_representations(cs, splits, [0], trp.RepPlan(plan.tcfg, plan.fcfg, 2, plan.seed, families=("TCN",), compact_dims=()), {},
                                         reuse=reuse)
        assert again[trp.ENC_FITS.name]["source"].to_list() == ["reused_checkpoint"]
        key = ["seq_id", "dim"]
        a = rep.filter((pl.col("family") == "TCN") & (pl.col("fold") == 0)).sort(key)["value"].to_numpy()
        b = again[trp.ATTEMPT_REP.name].filter(pl.col("family") == "TCN").sort(key)["value"].to_numpy()
        assert np.allclose(a, b, atol=1e-6), "a reloaded encoder must reproduce the embeddings"
        rand = rep.filter((pl.col("family") == "RAND") & (pl.col("fold") == 0)).sort(key)["value"].to_numpy()
        assert rand.shape == a.shape and not np.allclose(rand, a)
        records = trp.checkpoint_records({"transfer": Path(tmp)}, None, cs.spec.name)
        assert len(records) == 2 and all(not r["athletes"] & set(splits.filter(pl.col("outer_fold") == r["fold"])["nfl_id"].to_list()) for r in records)
    return ("a fold encoder is trained without its held-out athletes and still embeds them; its checkpoint reloads to the same "
            "embeddings; the compact projection and the untrained control go through the same path")


def check_tcn_fold_isolation() -> str:
    from . import fpca_validation as fv
    from . import transfer_repr as trp

    cs, splits, plan = _repr_fixture(trp, fv)
    only = trp.RepPlan(plan.tcfg, plan.fcfg, 2, plan.seed, families=("TCN",), compact_dims=())
    with tempfile.TemporaryDirectory() as tmp:
        fits = trp.fold_representations(cs, splits, [0], only, {}, models_dir=Path(tmp))[trp.ENC_FITS.name]
        ckpt = fits["checkpoint"][0]
        wrong = {("TCN", 1): ckpt}                                                                # the fold-0 encoder saw the athletes held out in fold 1
        assert _raises(LeakageError, lambda: trp.fold_representations(cs, splits, [1], only, {}, reuse=wrong)), \
            "a saved encoder whose training athletes are held out must be refused, not silently used"
        reshuffled = splits.with_columns(((pl.col("outer_fold") + 1) % 4).alias("outer_fold"))    # a newly constructed evaluation split
        assert _raises(LeakageError, lambda: trp.fold_representations(cs, reshuffled, [0], only, {}, reuse={("TCN", 0): ckpt}))
        test0 = splits.filter(pl.col("outer_fold") == 0)["nfl_id"].to_numpy()
        Y2 = cs.Y.copy()
        Y2[np.isin(cs.nfl_ids, test0)] += 4.0
        moved = type(cs)(cs.spec, cs.grid, Y2, cs.t_orig, cs.seq_ids, cs.nfl_ids)
        a = trp.fold_representations(cs, splits, [0], only, {})[trp.ATTEMPT_REP.name]
        b = trp.fold_representations(moved, splits, [0], only, {})[trp.ATTEMPT_REP.name]
        pick = lambda t, role: t.filter(pl.col("role") == role).sort("seq_id", "dim")["value"].to_numpy()          # noqa: E731
        assert np.allclose(pick(a, "train"), pick(b, "train"), atol=1e-5), "weights, normaliser and early stopping must ignore held-out curves"
        assert not np.allclose(pick(a, "heldout"), pick(b, "heldout"))
    return ("an encoder called 'out of fold' elsewhere is refused when its training athletes are held out in the transfer fold; "
            "changing held-out curves leaves the trained encoder and every training embedding unchanged")


# ============================================================================== 06C
def check_matched_pair_formation() -> str:
    m = av.synthetic_meta(n=160, seed=3)
    mc = rb.MatchConfig()
    scales = rb.matching_scales(m["meta"], m["roles"], mc)
    d = tm.Design(m["ids"].astype(np.int64), m["position"].astype(str), m["cohort"].astype(str), {}, (), (), (), np.arange(160) % 4, np.arange(160) % 3)
    pairs = conform(tx.matched_pairs(d, m["meta"], scales, mc), tx.PAIRS)
    assert pairs.height > 20 and (pairs["nfl_id_a"] < pairs["nfl_id_b"]).all() and pairs.select(pl.struct("nfl_id_a", "nfl_id_b").n_unique()).item() == pairs.height
    assert (pairs["distance"] <= mc.caliper + 1e-12).all()
    pos = dict(zip(m["ids"].tolist(), m["position"].tolist()))
    assert all(pos[a] == pos[b] == p for a, b, p in pairs.select("nfl_id_a", "nfl_id_b", "position_group").iter_rows())
    off = dict(zip(m["ids"].tolist(), m["official"][:, 0].tolist()))
    assert all(np.isclose(dv, off[b] - off[a]) for a, b, dv in pairs.select("nfl_id_a", "nfl_id_b", "d_forty").iter_rows())
    rng = np.random.default_rng(1)
    y = rng.normal(size=160)

    def oof(pred: dict[str, np.ndarray]) -> pl.DataFrame:
        return pl.concat([pl.DataFrame({"target_id": ["T"] * 160, "model": [k] * 160, "nfl_id": m["ids"], "y": y, "pred": v},
                                       schema_overrides={"nfl_id": pl.Int64}) for k, v in pred.items()])
    pred = conform(tx.pair_predictions(pairs, oof({"M1": rng.normal(size=160), "M5": y + 0.1 * rng.normal(size=160), "M4": -y})), tx.PAIR_PRED)
    assert pred.height == 3 * pairs.height
    summ = conform(tx.pair_summary(pairs, pred, TC, seed=2), tx.PAIR_SUMMARY)
    r = {x["model"]: x for x in summ.iter_rows(named=True)}
    assert r["M5"]["concordance"] > 0.9 and r["M4"]["concordance"] == 0.0 and 0.3 < r["M1"]["concordance"] < 0.7
    assert r["M5"]["d_concordance_lower"] > 0 and r["M4"]["d_concordance_upper"] < 0 and r["M1"]["d_concordance"] is None
    assert r["M5"]["n_pairs"] == pairs.height and r["M5"]["max_reuse"] >= 2 and r["M5"]["n_athletes"] <= 160
    assert r["M5"]["concordance_lower"] < r["M5"]["concordance"] <= r["M5"]["concordance_upper"] + 1e-12
    cases = tx.case_studies(pairs, pred, m["athletes"].with_columns(pl.lit("x").alias("display_name")), "T", ["M1", "M5"], 4)
    assert cases.height == 4 and cases["distance"].to_list() == sorted(pairs["distance"].to_list())[:4] and (cases["status"] == "illustrative only").all()
    flipped = tx.matched_pairs(d, m["meta"], scales, mc)
    assert flipped.equals(pairs), "pairs are formed from official measurements only: NFL values cannot change them"
    return (f"{pairs.height} unordered same-position pairs inside the caliper; a model that knows the outcome orders "
            f"{100 * r['M5']['concordance']:.0f} % of them, a random one about half; intervals resample athletes because pairs reuse them")


def check_grouped_permutation() -> str:
    s = synthetic_study(signal=1.0)
    d = s["d"]
    rng = np.random.default_rng(0)
    perm = tm._within(d.position, rng)
    assert sorted(perm.tolist()) == list(range(d.n)) and (d.position[perm] == d.position).all() and (perm != np.arange(d.n)).mean() > 0.9
    y_fn = lambda train_ids: s["y"]                                                               # noqa: E731
    out = tm.permutation_null(d, y_fn, s["blocks"], s["models"], ("M5", "M1"), TC, seed=1, mode="outcome")
    assert out["null"].shape == (TC.n_perm,) and out["observed"] > np.quantile(out["null"], 0.95) and out["p_value"] <= 2.0 / (1 + TC.n_perm)
    assert abs(out["null"].mean()) < 0.05, "with permuted labels the encoder block must not help"
    link = tm.permutation_null(d, y_fn, s["blocks"], s["models"], ("M5", "M1"), TC, seed=1, mode="linkage")
    assert link["p_value"] <= 2.0 / (1 + TC.n_perm) and abs(link["null"].mean()) < 0.05          # another athlete's embedding is worthless
    base = tm.permutation_null(d, y_fn, s["blocks"], s["models"], ("M1", "M0"), TC, seed=1, mode="outcome")
    assert base["p_value"] <= 2.0 / (1 + TC.n_perm)                                              # the official baseline survives the same check
    null = synthetic_study(signal=0.0, seed=7)
    none = tm.permutation_null(null["d"], lambda t: null["y"], null["blocks"], null["models"], ("M5", "M1"), TC, seed=1)
    assert none["p_value"] > 1.0 / (1 + TC.n_perm), "no planted trait: the observed gain must sit inside the null"
    row = tm.perm_row(tm.PRIMARY, tm.PRIMARY, "T", ("M5", "M1"), out)
    conform(pl.DataFrame([row], schema=tm.PERM.columns), tm.PERM)
    assert _raises(ValueError, lambda: tm.permutation_null(d, y_fn, s["blocks"], s["models"], ("M5", "M1"), TC, 1, mode="plays"))
    return ("labels and representation rows are permuted between ATHLETES inside a position group; the planted gain lies outside both "
            "null distributions and a study without a trait lies inside")


def check_outcome_label_leakage() -> str:
    s = synthetic_study(signal=0.0, seed=9)
    d, y = s["d"], s["y"]
    leak = tm.ArrayBlocks(d.ids, {**s["blocks"].blocks, "TCN": np.column_stack([y + 0.01, s["blocks"].blocks["TCN"][:, 1:]])})
    spec = [m for m in s["models"] if m.model in ("M1", "M5")]
    res, _ = tm.run_oof(d, lambda t: y, leak, spec, TC)
    r2 = tm.point_metrics(res["y"], res["pred"]["M5"], res["y_ref"])["r2_oos"]
    assert r2 > 0.98, "the fixture must be able to show what a leaked label looks like"
    out = tm.permutation_null(d, lambda t: y, leak, s["models"], ("M5", "M1"), TC, seed=2, mode="outcome")
    assert abs(out["null"].mean()) < 0.05 and out["observed"] > out["null"].max()                # a block equal to the label is exposed by its size
    clean, _ = tm.run_oof(d, lambda t: y, s["blocks"], spec, TC)
    gain = np.sqrt(np.mean((y - clean["pred"]["M1"]) ** 2)) - np.sqrt(np.mean((y - clean["pred"]["M5"]) ** 2))
    assert abs(gain) < 0.03, "representations built without the label show no gain when no trait exists"
    seen: list[int] = []

    def y_fn(train_ids: np.ndarray) -> np.ndarray:
        seen.append(len(train_ids))
        return y
    tm.run_oof(d, y_fn, s["blocks"], spec, TC)
    sizes = [int((d.fold != f).sum()) for f in np.unique(d.fold)]
    assert seen == sizes, "labels are requested once per fold, with the TRAINING athletes of that fold only"
    assert _raises(tm.CohortError, lambda: tm.assert_pre_nfl_columns(["O2.accel_sust05_mean"]))
    assert _raises(tm.CohortError, lambda: tm.assert_pre_nfl_columns(["value_adj"]))
    tm.assert_pre_nfl_columns([FORTY, TEN, WEIGHT, HEIGHT, "cmb_vertical"])
    for mod in ("transfer_repr.py",):
        src = (Path(__file__).parent / mod).read_text(encoding="utf-8")
        assert "nfl_outcomes" not in src and "TargetSet" not in src and "athlete_values" not in src, f"{mod} must not be able to read a label"
    ev = pl.DataFrame([{**{k: None for k in tx.EVIDENCE.columns}, "dimension": "tcn_b_movement_fingerprint", "model": "M5", "reference": "M1",
                        "target_id": "T", "n_athletes": 240, "delta_rmse": 0.05, "ci_lower": 0.01, "ci_upper": 0.09, "relative": 0.06,
                        "established": True, "worse": False}], schema=tx.EVIDENCE.columns)
    status, why = tx.readiness_status(ev.row(0, named=True), "ready_for_transfer_test", TC)
    assert status == "predictive_candidate" and "not evaluated" in why, "an established gain without its robustness checks is not validated"
    full = {**ev.row(0, named=True), "loco_delta": 0.03, "positions_positive": 3, "perm_p_outcome": 0.01, "vs_random_lower": 0.01,
            "seeds_positive": 3, "seeds_total": 3}
    assert tx.readiness_status(full, "ready_for_transfer_test", TC)[0] == "predictive_validated"
    assert tx.readiness_status({**full, "seeds_positive": 2}, "ready_for_transfer_test", TC)[0] == "predictive_candidate"
    assert tx.readiness_status({**full, "established": False, "ci_lower": -0.01}, "ready_for_transfer_test", TC)[0] == "exploratory"
    assert tx.readiness_status({**full, "established": False, "delta_rmse": -0.01, "ci_lower": -0.03}, "carry_with_caveat", TC)[0] == "descriptive_ready"
    assert tx.readiness_status({**full, "established": False, "worse": True, "delta_rmse": -0.05, "ci_lower": -0.09, "ci_upper": -0.01}, "x", TC)[0] == "rejected"
    assert tx.readiness_status({**full, "ci_lower": None}, "x", TC)[0] == "insufficient_evidence"
    return ("a block that contains the label is obvious (R² ≈ 1) and unlike any permutation; clean blocks show no gain without a trait; "
            "labels are requested with training athletes only; the representation module cannot read an NFL label; readiness is "
            "'validated' only when the gain is established AND every robustness check passed")


CHECKS_06A: tuple[Callable[[], str], ...] = (
    check_athlete_linkage, check_duplicate_athlete_detection, check_cohort_eligibility, check_target_freeze_enforcement,
    check_athlete_grouped_splitting, check_checkpoint_compatibility, check_artifact_version_consistency)
CHECKS_06B: tuple[Callable[[], str], ...] = (
    check_repeat_attempt_aggregation, check_train_only_preprocessing, check_fold_target_handling, check_oof_uniqueness,
    check_paired_model_cohorts, check_regression_metrics, check_delta_rmse_sign, check_athlete_bootstrap, check_fpca_fold_isolation,
    check_unseen_athlete_embedding, check_tcn_fold_isolation)
CHECKS_06C: tuple[Callable[[], str], ...] = (check_matched_pair_formation, check_grouped_permutation, check_outcome_label_leakage)
ALL_CHECKS: tuple[Callable[[], str], ...] = (*CHECKS_06A, *CHECKS_06B, *CHECKS_06C)
TORCH_CHECKS = ("fpca_fold_isolation", "unseen_athlete_embedding", "tcn_fold_isolation")


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    """Run the synthetic checks in-process. A failing check is reported, not raised, so one table shows them all."""
    rows = []
    for fn in checks:
        try:
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "PASS", "detail": fn()})
        except Exception as e:                                   # noqa: BLE001 — the table must list every failure
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "FAIL",
                         "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


def assert_all_passed(report: pl.DataFrame) -> None:
    failed = report.filter(pl.col("status") != "PASS")
    if failed.height:
        raise AssertionError(f"{failed.height} synthetic validation check(s) failed: "
                             + "; ".join(f"{r['check']} → {r['detail']}" for r in failed.iter_rows(named=True)))
