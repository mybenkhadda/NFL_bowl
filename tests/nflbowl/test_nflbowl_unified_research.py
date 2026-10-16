"""Tests of nflbowl.research (unified Milestone 12 notebook infrastructure). Authored statically; not yet executed.

Covers configuration resolution, the run directory / artifact helpers, figure naming, the input gates without data,
the evidence classification of Part 11 and the decision matrix. No warehouse, bundle or GPU is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")

from nflbowl.research import artifacts, config, data_access, evaluation, reporting  # noqa: E402


# ------------------------------------------------------------------------------ config
def test_resolve_defaults_and_run_dir(tmp_path: Path) -> None:
    u = config.resolve({"drive_root": str(tmp_path)})
    assert u.run_mode == "fast" and u.precision == "fp32" and u.seed == config.RESEARCH_SEED
    assert u.run_dir == tmp_path / "unified_research" / "m12_unified"
    assert u.encoder_preset == "colab_gpu"
    assert u.opt("12C", "permutation") is False and u.opt("12E", "consistency") is True
    assert u.runtime_overrides()["runtime.precision"] == "fp32"


def test_resolve_modes_and_overrides() -> None:
    d = config.resolve({"run_mode": "debug"})
    assert d.encoder_preset == "fast_iteration" and d.opt("12E", "downstream") is False
    f = config.resolve({"run_mode": "full", "options": {"12C": {"permutation": False}}})
    assert f.opt("12C", "replication") is True and f.opt("12C", "permutation") is False
    full = config.resolve({"run_full_research": True})
    assert full.encoder_inductive and full.encoder_ablations and full.encoder_learning_curve
    assert full.opt("12B", "bayes") is False                                  # never switched on implicitly
    assert config.resolve({"precision": "amp"}).runtime_overrides()["runtime.precision"] == "auto"


@pytest.mark.parametrize("bad", [{"nope": 1}, {"run_mode": "x"}, {"precision": "fp8"}, {"stages": {"12Z": True}},
                                 {"options": {"12C": {"unknown": True}}}, {"knobs": {"12E": {"nope": 1}}}, {"experiment_name": "a b"}])
def test_resolve_rejects_invalid(bad: dict) -> None:
    with pytest.raises(config.UnifiedConfigError):
        config.resolve(bad)


def test_config_save_roundtrip(tmp_path: Path) -> None:
    u = config.resolve({"drive_root": str(tmp_path)})
    p = u.save(tmp_path / "c.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["run_dir"] == str(u.run_dir) and d["options"]["12E"]["controls"] is True


# ------------------------------------------------------------------------------ artifacts
def test_descriptive_figure_names() -> None:
    assert artifacts.descriptive_figure_name("AM12_C_03_envelope_vs_quantile_reliability", "12C") == "12c_envelope_vs_quantile_reliability"
    assert artifacts.descriptive_figure_name("AM12_0_01_eligibility_funnel", "12.0") == "12_0_eligibility_funnel"
    assert artifacts.descriptive_figure_name("training_curves", "12F") == "12f_training_curves"
    assert artifacts.descriptive_figure_name("12f_x", "12F") == "12f_x"
    assert artifacts.stage_slug("12.0") == "12_0"


def test_run_tables_processed_and_reports(tmp_path: Path) -> None:
    run = artifacts.ResearchRun(tmp_path / "run", config.resolve({"drive_root": str(tmp_path)}))
    for d in artifacts.LAYOUT:
        assert (run.root / d).is_dir()
    df = pl.DataFrame({"a": [1, 2], "b": [[1], [2, 3]]})
    paths = run.save_table(df, "my table", "12C")
    assert (run.root / "tables" / "parquet" / "my_table.parquet").is_file() and (run.root / "tables" / "csv" / "my_table.csv").is_file()
    assert len(paths) == 2 and run.has_table("my_table") and run.load_table("my_table").height == 2
    run.save_table(df, "diag", "checks", subdir="diagnostics")
    assert run.has_table("diag")
    run.save_processed_dataset(pl.DataFrame({"x": [1.0]}), "proc", "12.0", {"src": "test"})
    assert (run.root / "processed_data" / "proc.provenance.json").is_file()
    assert run.load_processed_dataset("proc").height == 1
    run.save_processed_dataset({"k": 1}, "meta", "12.0")
    assert run.load_processed_dataset("meta")["k"] == 1
    md = run.save_markdown("# t\n\n| a |\n|---|\n| 1 |\n", "12c_distributional_capacity", "12C")
    assert md[0].name == "12c_distributional_capacity.md" and (run.root / "reports" / "html" / "12c_distributional_capacity.html").is_file()
    arts = run.artifacts(stage="12C")
    assert set(arts["kind"].to_list()) >= {"table", "report"}
    index = json.loads((run.root / "manifests" / "artifact_index.json").read_text(encoding="utf-8"))
    assert any(k.startswith("12C:table:my_table") for k in index)


def test_large_table_skips_csv(tmp_path: Path) -> None:
    run = artifacts.ResearchRun(tmp_path / "run", config.resolve({"drive_root": str(tmp_path), "csv_max_rows": 3}))
    run.save_table(pl.DataFrame({"a": list(range(10))}), "big", "x")
    assert (run.root / "tables" / "parquet" / "big.parquet").is_file() and not (run.root / "tables" / "csv" / "big.csv").exists()


def test_save_figure_all_formats(tmp_path: Path) -> None:
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    run = artifacts.ResearchRun(tmp_path / "run", config.resolve({"drive_root": str(tmp_path)}))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = run.save_figure(fig, "eda_test_figure", "eda", {"line": pl.DataFrame({"x": [0, 1], "y": [0, 1]})}, close=True)
    for fmt in ("png", "svg", "pdf"):
        assert (run.root / "figures" / fmt / f"eda_test_figure.{fmt}").is_file()
    assert "data:line" in out and (run.root / "figures" / "data" / "eda_test_figure__line.csv").is_file()


def test_stage_records_and_failure(tmp_path: Path) -> None:
    run = artifacts.ResearchRun(tmp_path / "run")
    run.stage_begin("12C")
    run.stage_end("12C", artifacts.STATUS_DONE, note="ok")
    run.stage_begin("12A")
    try:
        raise ValueError("boom")
    except ValueError as e:
        run.stage_fail("12A", e)
    assert run.stage_status("12C") == "executed" and run.stage_status("12A") == "failed"
    assert (run.root / "logs" / "12a_traceback.txt").is_file()
    reopened = artifacts.ResearchRun(tmp_path / "run")
    assert reopened.stage_status("12A") == "failed"                          # persisted across sessions


# ------------------------------------------------------------------------------ data access
def test_plan_without_any_input(tmp_path: Path) -> None:
    u = config.resolve({"drive_root": str(tmp_path), "bundle_dir": str(tmp_path / "missing"), "research_root": str(tmp_path / "nowhere")})
    src = data_access.discover(u)
    assert src.mode == "none" and src.bundle_dir is None and not src.warehouse
    plan = data_access.plan(u, src)
    assert set(plan["expected"].to_list()) == {"skip"}
    g = data_access.check_stage("12A", None, "full", False)
    assert not g["ok"] and "warehouse" in g["reason"]
    assert data_access.check_stage("12F", None, "full", True)["ok"]


def test_requirements_cover_every_stage() -> None:
    assert set(data_access.REQUIREMENTS) == set(config.STAGES)
    assert ("advanced_v1", "m12c_target_freeze", "json") in data_access.REQUIREMENTS["12A"].required      # 12A hard gate
    assert data_access.REQUIREMENTS["12F"].mode == "bundle"


# ------------------------------------------------------------------------------ evaluation / decisions
def test_evidence_classes() -> None:
    c = evaluation.evidence_class
    assert c({"training_regime": "transductive", "metric": "AUC − 0.5"}) == "transductive"
    assert c({"source_table": "m12e_auc_diff", "metric": "Δ retrieval AUC"}) == "transductive"
    assert c({"evaluation_target": "measurement_reliability", "metric": "reliability difference"}) == "measurement"
    assert c({"evaluation_target": "cross_domain_physical_transfer", "metric": "beta_rc"}) == "in_sample_estimand"
    assert c({"evaluation_target": "cross_domain_physical_transfer", "metric": "Δ RMSE"}) == "inductive"


def test_decision_matrix_without_evidence(tmp_path: Path) -> None:
    run = artifacts.ResearchRun(tmp_path / "run")
    dm = reporting.decision_matrix(run, None)
    assert dm.height and set(dm["decision_stance"].to_list()) == {"no evidence from this run"}


def test_decision_matrix_reads_verdicts(tmp_path: Path) -> None:
    run = artifacts.ResearchRun(tmp_path / "run")
    comps = pl.DataFrame({"track": ["12C", "12C", "12E"], "primary": [True, True, True], "verdict": ["supported", "supported", "inconclusive"]})
    dm = reporting.decision_matrix(run, comps)
    assert set(dm.filter(pl.col("track") == "12C")["decision_stance"].str.slice(0, 13).to_list()) == {"candidate use"}
    assert dm.filter(pl.col("track") == "12E")["decision_stance"].str.starts_with("NOT supported").all()
