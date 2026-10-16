"""Regression tests for the Phase 04 contract failures of 2026-10-11 (synthetic data, no warehouse access).

Production failures reproduced here:

* ``Q_athlete_eligibility: key ['domain', 'drill', 'scenario'] has … 80 duplicate rows``
* ``B_smoothing_agreement: key ['domain', 'drill', 'feature', 'smoothing', 'level'] has … 864 duplicate rows``

Cause: five canonical Combine skill drills are run by several position groups (drill_type = ``stratum``), e.g.
GAUNTLET_DRILL under SKILL_DRILLS_DB / _WR / _TE. The producers group by domain × stratum × drill; the keys
omitted ``stratum``. Study Q additionally joined its S0 reference on (domain, drill), multiplying those rows.
The synthetic frames below contain one drill shared by three strata, several cohorts, positions, smoothing
settings and both analytical levels.

Authored 2026-10-11, NOT executed by Claude. Run::

    python -m pytest tests/test_phase04_grain_regression.py tests/test_phase04_m1.py tests/test_polars2_compat.py -q
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nflbdb.features.config import KinematicsConfig
from nflbdb.research.contracts import ContractError, conform
from nflbdb.research.phase04 import __main__ as cli
from nflbdb.research.phase04 import events as c1
from nflbdb.research.phase04 import quality_eligibility as q
from nflbdb.research.phase04 import report as rep
from nflbdb.research.phase04 import smoothing as b
from nflbdb.research.phase04 import validation_registry as v
from nflbdb.research.phase04.config import EligibilityConfig, Phase04Config, SmoothingStudyConfig, ValidationConfig
from nflbdb.research.studies.report import TOKEN, resolve
from nflbdb.research.studies.results import ResultCollector
from nflbdb.research.studies.store import StudyStore

GAUNTLET = "GAUNTLET_DRILL"
# (domain, stratum, drill, position): one measured drill, one skill drill shared by three drill_types, NFL
CELLS = (("combine", "FORTY_YARD_DASH", "FORTY_YARD_DASH", "WR"), ("combine", "SKILL_DRILLS_WR", GAUNTLET, "WR"),
         ("combine", "SKILL_DRILLS_DB", GAUNTLET, "DB"), ("combine", "SKILL_DRILLS_TE", GAUNTLET, "TE"),
         ("nfl_sample", "nfl_play", "nfl_play", "DB"))
FEATS = ("peak_speed", "p95_speed")
ALT = ("sg5p2", "sg11p3")
N_ATH = 12
CFG = SmoothingStudyConfig(n_boot=20, min_athletes=5, min_trajectories=5)
OLD_AGREE_KEY = ["domain", "drill", "feature", "smoothing", "level"]
OLD_ELIG_KEY = ["domain", "drill", "scenario"]
S1 = "S1 structural pass"
TEMPLATE = Path(__file__).resolve().parents[1] / "reports" / "PHASE_04_SCIENTIFIC_RESEARCH_REPORT_TEMPLATE.md"


def _rel_change(cell: int) -> float:
    """sg5p2 inflates values by a stratum-specific factor, so a pooled estimate is distinguishable."""
    return 0.01 * (cell + 1)


def _long(seed: int = 0) -> pl.DataFrame:
    """Study B long table: 5 cells × 12 athletes × 2 attempts × 2 features × 3 SG settings, 2 cohorts."""
    rng = np.random.default_rng(seed)
    rows = []
    for ci, (domain, stratum, drill, pos) in enumerate(CELLS):
        settings = (("sg7p2", True, 1.0), ("sg5p2", False, 1.0 + _rel_change(ci)), ("sg11p3", False, 1.001))
        for a in range(N_ATH):
            nfl_id, trait = 1000 * ci + a, rng.normal(8, 1)
            for k in (1, 2):
                base = trait + rng.normal(0, 0.05)
                for feat in FEATS:
                    for sg, prim, fac in settings:
                        rows.append({"domain": domain, "traj_id": f"{stratum}|{nfl_id}|{k}", "nfl_id": nfl_id,
                                     "cohort": 2023 + a % 2, "stratum": stratum, "drill": drill, "position": pos,
                                     "attempt_order": k if domain == "combine" else None, "smoothing": sg,
                                     "is_primary": prim, "feature": feat,
                                     "value": base * fac * (1.0 if feat == "peak_speed" else 0.9)})
    return pl.DataFrame(rows, schema=b.FEATURES_DS.columns)


def _flags() -> pl.DataFrame:
    """Study Q flag table: forty and WR gauntlet clean; DB gauntlet loses one attempt of 2 of 4 athletes;
    TE gauntlet entirely structurally flagged; NFL clean. Two attempts per athlete."""
    rows = []

    def add(domain, stratum, drill, pos, nfl_id, k, structural):
        rows.append({"domain": domain, "traj_id": f"{stratum}|{nfl_id}|{k}", "nfl_id": nfl_id, "cohort": 2024,
                     "stratum": stratum, "drill": drill, "position": pos, "quality_tier": "A",
                     "structural_flag": structural, "signal_flag": False, "boundary_flag": False,
                     "any_flag": structural, "structural_reasons": "", "signal_reasons": ""})

    for k in (1, 2):
        for a in range(4):
            add("combine", "FORTY_YARD_DASH", "FORTY_YARD_DASH", "WR", a, k, False)
            add("combine", "SKILL_DRILLS_WR", GAUNTLET, "WR", 100 + a, k, False)
            add("combine", "SKILL_DRILLS_DB", GAUNTLET, "DB", 200 + a, k, a < 2 and k == 2)
        for a in range(2):
            add("combine", "SKILL_DRILLS_TE", GAUNTLET, "TE", 300 + a, k, True)
        for a in range(3):
            add("nfl_sample", "nfl_play", "nfl_play", "DB", 400 + a, k, False)
    return pl.DataFrame(rows, schema=q.FLAGS_DS.columns)


def _dups(df: pl.DataFrame, key: list[str]) -> int:
    return df.height - df.select(pl.struct(key).n_unique()).item()


def _cfg(root: Path) -> SimpleNamespace:
    paths = SimpleNamespace(tables=root / "reports" / "tables", figures=root / "reports" / "figures",
                            gold=root / "warehouse" / "gold", project_root=root,
                            rel=lambda p: Path(p).relative_to(root).as_posix())
    return SimpleNamespace(paths=paths, parquet=SimpleNamespace(polars_kwargs=lambda: {}))


def _stores(root: Path) -> tuple[StudyStore, StudyStore]:
    cfg = _cfg(root)
    return StudyStore(cfg, phase="phase_04"), StudyStore(cfg)      # type: ignore[arg-type]


# ============================================================================== Study Q — eligibility grain
def test_eligibility_key_is_unique_when_a_drill_spans_several_strata():
    e = q.eligibility(_flags(), EligibilityConfig())
    conform(e, q.ELIG)                                              # raised ContractError in production
    assert e.height == len(CELLS) * len(q.SCENARIOS)                # one row per stratum × drill × scenario
    # the declared key needs the stratum: under the old key the shared drill still collides
    assert _dups(e, OLD_ELIG_KEY) == 2 * len(q.SCENARIOS)
    with pytest.raises(ContractError, match="duplicate"):
        conform(e, dataclasses.replace(q.ELIG, key=tuple(OLD_ELIG_KEY)))


def test_eligibility_reference_join_does_not_multiply_or_mix_strata():
    e = q.eligibility(_flags(), EligibilityConfig(min_attempts_eligible=2))
    g = {r["stratum"]: r for r in e.filter((pl.col("drill") == GAUNTLET) & (pl.col("scenario") == S1))
         .iter_rows(named=True)}
    assert set(g) == {"SKILL_DRILLS_WR", "SKILL_DRILLS_DB", "SKILL_DRILLS_TE"}       # subgroups preserved
    assert (g["SKILL_DRILLS_WR"]["n_athletes_repeat"], g["SKILL_DRILLS_WR"]["share_athletes_repeat"]) == (4, 1.0)
    db = g["SKILL_DRILLS_DB"]
    assert (db["n_trajectories"], db["n_athletes_any"], db["n_athletes_repeat"]) == (6, 4, 2)
    assert db["share_trajectories"] == pytest.approx(0.75) and db["share_athletes_repeat"] == pytest.approx(0.5)
    te = g["SKILL_DRILLS_TE"]
    assert (te["n_trajectories"], te["n_athletes_any"], te["n_athletes_repeat"]) == (0, 0, 0)
    assert te["share_athletes_repeat"] == 0.0
    s0 = e.filter(pl.col("scenario_order") == 0)
    assert s0["share_athletes_repeat"].to_list() == [1.0] * len(CELLS)


# ============================================================================== Study B — agreement grain
def test_agreement_key_is_unique_when_a_drill_spans_several_strata():
    t = b.agreement_table(_long(), CFG, seed=1)
    conform(t, b.AGREE)                                             # raised ContractError in production
    assert t.height == len(CELLS) * len(FEATS) * len(ALT) * len(b.LEVELS)
    assert set(t["smoothing"].to_list()) == set(ALT) and set(t["level"].to_list()) == set(b.LEVELS)
    # production: 6 extra stratum × drill cells × 24 features × 3 settings × 2 levels = 864; here 2 × 2 × 2 × 2
    assert _dups(t, OLD_AGREE_KEY) == 2 * len(FEATS) * len(ALT) * len(b.LEVELS)
    with pytest.raises(ContractError, match="duplicate"):
        conform(t, dataclasses.replace(b.AGREE, key=tuple(OLD_AGREE_KEY)))


def test_agreement_preserves_stratum_subgroups_at_both_levels():
    t = b.agreement_table(_long(), CFG, seed=1).filter(
        (pl.col("feature") == "peak_speed") & (pl.col("smoothing") == "sg5p2"))
    for ci, (domain, stratum, drill, _) in enumerate(CELLS):
        cell = t.filter((pl.col("domain") == domain) & (pl.col("stratum") == stratum) & (pl.col("drill") == drill))
        by = {r["level"]: r for r in cell.iter_rows(named=True)}
        assert set(by) == set(b.LEVELS)
        assert by["trajectory"]["n_units"] == 2 * N_ATH and by["athlete"]["n_units"] == N_ATH   # never pooled
        assert by["trajectory"]["n_athletes"] == N_ATH
        assert by["trajectory"]["median_rel_diff"] == pytest.approx(_rel_change(ci))            # its own estimate
        assert by["athlete"]["spearman_lower"] is not None and by["athlete"]["top_overlap_lower"] is not None


def test_variance_and_repeatability_tables_share_the_stratum_grain():
    long = _long()
    var, repeat = b.variance_table(long), b.repeatability_table(long, CFG, seed=2)
    conform(var, b.VARSHARE)
    conform(repeat, b.REPEAT)
    assert var.height == len(CELLS) * len(FEATS) * len(b.LEVELS)
    assert repeat.height == 4 * len(FEATS) * 3                      # Combine cells × features × SG settings
    assert set(repeat["n_athletes"].to_list()) == {N_ATH} and set(repeat["n_attempts"].to_list()) == {2 * N_ATH}


def test_repeatability_counts_attempts_within_the_stratum():
    """One attempt under SKILL_DRILLS_WR plus one under SKILL_DRILLS_DB is not a repeated attempt."""
    long = _long()
    one = long.filter(pl.col("traj_id") == "SKILL_DRILLS_WR|1000|1").with_columns(pl.lit(9999, dtype=pl.Int64)
                                                                                   .alias("nfl_id"))
    extra = pl.concat([one.with_columns(pl.lit("SKILL_DRILLS_WR|9999|1").alias("traj_id")),
                       one.with_columns(pl.lit("SKILL_DRILLS_DB|9999|2").alias("traj_id"),
                                        pl.lit("SKILL_DRILLS_DB").alias("stratum"), pl.lit("DB").alias("position"),
                                        pl.lit(2, dtype=pl.Int64).alias("attempt_order"))])
    repeat = b.repeatability_table(pl.concat([long, extra]), CFG, seed=2)
    assert set(repeat.filter(pl.col("drill") == GAUNTLET)["n_athletes"].to_list()) == {N_ATH}


def test_robustness_facets_keep_cohort_and_position_levels():
    rob = b.robustness_table(_long(), CFG)
    conform(rob, b.ROBUST)
    g = rob.filter((pl.col("drill") == GAUNTLET) & (pl.col("feature") == "peak_speed")
                   & (pl.col("smoothing") == "sg5p2"))
    pos = g.filter(pl.col("facet") == "position")
    assert set(pos["level"].to_list()) == {"WR", "DB", "TE"} and set(pos["n_athletes"].to_list()) == {N_ATH}
    assert set(g.filter(pl.col("facet") == "cohort")["level"].to_list()) == {"2023", "2024"}


def test_b_results_rows_are_unique_per_stratum_and_drill():
    long = _long()
    agree, var, repeat = b.agreement_table(long, CFG, 1), b.variance_table(long), b.repeatability_table(long, CFG, 2)
    rc = ResultCollector("B", CFG.min_athletes)
    b.b_results(agree, var, repeat, rc)
    res = conform(rc.frame(), b.STUDY_RESULTS)                      # next production failure after the key fix
    strata = set(res["stratum"].to_list())
    assert {"FORTY_YARD_DASH", "nfl_play", f"SKILL_DRILLS_WR:{GAUNTLET}", f"SKILL_DRILLS_TE:{GAUNTLET}"} <= strata
    assert GAUNTLET not in strata


# ============================================================================== Study V — downstream consumer
def test_registry_follows_the_study_b_grain():
    long, cfg = _long(), ValidationConfig()
    agree, repeat = b.agreement_table(long, CFG, 1), b.repeatability_table(long, CFG, 2)
    ev = v.evidence(agree, repeat, None, None, None, None, cfg, "sg7p2", "default", 0.3)
    reg = conform(v.registry(ev, cfg), v.REGISTRY)
    assert reg.height == len(CELLS) * len(FEATS)
    mag = {r["stratum"]: r["value_smooth_magnitude"] for r in reg.filter(
        (pl.col("drill") == GAUNTLET) & (pl.col("feature") == "peak_speed")).iter_rows(named=True)}
    # each position group keeps its own evidence instead of the worst value across groups
    assert mag == {CELLS[ci][1]: pytest.approx(_rel_change(ci)) for ci in (1, 2, 3)}
    assert set(reg["n_athletes"].to_list()) == {N_ATH}
    conform(v.summary(reg, cfg), v.SUMMARY)


# ============================================================================== persistence, cache, dependencies
def test_contract_roundtrip_is_idempotent(tmp_path: Path):
    store, _ = _stores(tmp_path)
    elig, agree = q.eligibility(_flags(), EligibilityConfig()), b.agreement_table(_long(), CFG, seed=1)
    assert_frame_equal(agree, b.agreement_table(_long(), CFG, seed=1))          # deterministic recomputation
    for df, contract, study in ((elig, q.ELIG, "Q"), (agree, b.AGREE, "B")):
        first = store.write(df, contract, study)
        second = store.write(df, contract, study)                                # rewrite replaces, never appends
        assert_frame_equal(first, second)
        assert_frame_equal(store.read(contract), first)
        assert store._manifest()["tables"][contract.name]["rows"] == df.height
        assert all(p.is_file() for p in store.paths(contract))


def _stub_study(tmp_path: Path, long: pl.DataFrame) -> tuple[b.SmoothingStudy, dict]:
    """SmoothingStudy without a warehouse: the frame stage and both fingerprints are stubbed."""
    study = object.__new__(b.SmoothingStudy)
    study.store, _ = _stores(tmp_path)
    study.pcfg, study.s, study.skip_nfl_sample = Phase04Config(smoothing=CFG), CFG, False
    calls = {"frames": 0, "frames_fp": "f1"}
    catalog = pl.DataFrame([{"feature": f, "unit": u, "family": fam, "definition": b.DEFINITIONS[f]}
                            for f, (u, fam) in b.FEATURES.items()], schema=b.CATALOG.columns)

    def compute_frames():
        calls["frames"] += 1
        return {b.FEATURES_DS.name: long, b.EVENTS_DS.name: b.EVENTS_DS.empty(), b.CATALOG.name: catalog,
                b.VENDOR.name: b.VENDOR.empty(), b.EXAMPLES.name: b.EXAMPLES.empty()}

    study.compute_frames = compute_frames
    study.frames_fingerprint = lambda: calls["frames_fp"]
    study.fingerprint = lambda: "b1"
    return study, calls


def test_frame_stage_survives_an_analysis_failure_and_is_reused(tmp_path: Path):
    study, calls = _stub_study(tmp_path, _long())

    def failing_analysis(long):
        raise ContractError("B_smoothing_agreement: key [...] has 0 null and 864 duplicate rows")

    study.analyse = failing_analysis
    with pytest.raises(ContractError):
        study.run(figures=False)
    runs = study.store._manifest()["runs"]
    assert b.FRAME_STAGE in runs and "B" not in runs               # heavy stage recorded, study still incomplete
    assert not study.store.path(b.AGREE).is_file()

    del study.analyse                                               # the corrected analysis
    out = study.run(figures=False)
    assert calls["frames"] == 1                                     # frames were NOT recomputed
    assert set(out) == {c.name for c in b.CONTRACTS}
    assert "B" in study.store._manifest()["runs"]
    assert all(p.is_file() for c in b.CONTRACTS for p in study.store.paths(c))

    again = study.run(figures=False)                                # idempotent: full reuse
    assert calls["frames"] == 1
    assert_frame_equal(again[b.AGREE.name], out[b.AGREE.name])


def test_frame_stage_is_rebuilt_when_its_fingerprint_changes_or_is_unrecorded(tmp_path: Path):
    study, calls = _stub_study(tmp_path, _long())
    # files on disk without a run record (the state left by the failed production run) are never trusted
    for name, df in study.compute_frames().items():
        study.store.write(df, next(c for c in b.FRAME_CONTRACTS if c.name == name), "B")
    assert b.FRAME_STAGE not in study.store._manifest()["runs"]
    study.frames()
    assert calls["frames"] == 2
    study.frames()
    assert calls["frames"] == 2                                     # recorded → reused
    calls["frames_fp"] = "f2"                                       # kinematics / upstream changed
    study.frames()
    assert calls["frames"] == 3


def test_downstream_studies_refuse_to_run_without_a_completed_study_b(tmp_path: Path):
    store, store03 = _stores(tmp_path)
    store.write(b.agreement_table(_long(), CFG, seed=1), b.AGREE, "B")          # table exists, run never completed
    with pytest.raises(FileNotFoundError, match="completed Study B"):
        c1.EventSensitivityStudy(store.cfg, Phase04Config(), store, kcfg=KinematicsConfig()).fingerprint()
    with pytest.raises(FileNotFoundError, match="--only B"):
        v.ValidationRegistryStudy(store.cfg, Phase04Config(), store, store03, kcfg=KinematicsConfig()).compute()


def test_check_reports_missing_invalid_incomplete_blocked_and_stale(tmp_path: Path):
    store, _ = _stores(tmp_path)
    issues = cli.check_study(store, "Q", (q.ELIG,), [])
    assert sum(i.startswith("missing") for i in issues) == 2 and any(i.startswith("incomplete") for i in issues)

    bad = pl.concat([q.eligibility(_flags(), EligibilityConfig())] * 2)        # duplicated keys on disk
    store.path(q.ELIG).parent.mkdir(parents=True, exist_ok=True)
    bad.write_parquet(store.path(q.ELIG))
    issues = cli.check_study(store, "Q", (q.ELIG,), [])
    assert any(i.startswith("invalid") and "duplicate" in i for i in issues)

    assert any(i.startswith("blocked") for i in cli.check_study(store, "C1", (), []))
    m = store._manifest()
    m["runs"] = {"B": {"fingerprint": "x", "built_at": "2026-01-02T00:00:00Z"},
                 "C1": {"fingerprint": "y", "built_at": "2026-01-01T00:00:00Z"}}
    store._save_manifest(m)
    issues = cli.check_study(store, "C1", (), [], fingerprint=lambda: "y")
    assert [i.split()[0] for i in issues] == ["stale"]
    assert any(i.startswith("stale") for i in cli.check_study(store, "B", (), [], fingerprint=lambda: "changed"))
    assert cli.check_study(store, "B", (), [], fingerprint=lambda: "x") == []


# ============================================================================== report lookup
def test_template_tokens_select_by_the_full_contract_key():
    keyed = {c.name: set(c.key) for c in (q.ELIG, b.AGREE, b.REPEAT)}
    seen = set()
    for kind, body in TOKEN.findall(TEMPLATE.read_text(encoding="utf-8")):
        name, *rest = [p.strip() for p in body.split("|")]
        if kind == "tbl" and name in keyed:
            assert {c.partition("=")[0] for c in rest[0].split(",")} == keyed[name], body
            seen.add(name)
    assert seen == set(keyed)


def test_report_resolves_only_completed_studies_and_names_the_source(tmp_path: Path):
    store, store03 = _stores(tmp_path)
    store.write(q.eligibility(_flags(), EligibilityConfig()), q.ELIG, "Q")
    view = rep.CombinedStore(store, store03)
    body = (f"Q_athlete_eligibility|domain=combine,stratum=SKILL_DRILLS_DB,drill={GAUNTLET},scenario={S1}"
            "|share_athletes_repeat|.1%")
    token = "{{tbl:" + body + "}}"
    assert resolve("tbl", body, view, {}).startswith("⟨pending")                # Q failed part-way: not cited
    dep = rep.source_dependencies([token], store, store03)[0]
    assert dep["source"] == "Phase 04 Study Q" and "no completed run" in dep["state"] and "--only Q" in dep["command"]
    section = rep.unresolved_section([token], store, store03)
    assert token in section and "Source dependencies" in section
    assert "_none_" in rep.unresolved_section([], store, store03)

    store.record_run("Q", "fp", (q.ELIG,))
    assert resolve("tbl", body, view, {}) == "50.0%"
    loose = body.replace("stratum=SKILL_DRILLS_DB,", "")                        # drill alone is not a key
    assert resolve("tbl", loose, view, {}).startswith("⟨ambiguous")
    dep = rep.source_dependencies(["{{tbl:" + loose + "}}"], store, store03)[0]
    assert "several rows match" in dep["state"]
