"""Phase 02 · artifacts + quality remediation (synthetic data from conftest.py).

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_phase02_quality.py -q
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from nflbdb.data.bronze import BronzeIngestor
from nflbdb.data.gold import GoldBuilder
from nflbdb.data.silver import SilverBuilder
from nflbdb.data.validate import DataQualityValidator
from nflbdb.research.artifacts import ArtifactStore, ArtifactValidationError, DatasetSpec
from nflbdb.research.quality import COMBINE_FLAGS, NFL_FLAGS, QualityRemediation


# ----------------------------------------------------------------------------- artifact store
def _spec(version: str = "1", key=("id",)) -> DatasetSpec:
    return DatasetSpec(name="toy", grain="one row per id", key=key, sources=("bronze.games",),
                       description="toy", columns={"id": "identifier", "v": "value"}, version=version)


def test_build_then_reuse_then_rebuild_on_version_change(synth):
    BronzeIngestor(synth).ingest(["games"])
    store, calls = ArtifactStore(synth), []

    def builder():
        calls.append(1)
        return pl.DataFrame({"id": [1, 2], "v": [0.5, 1.5]})

    store.build(_spec(), builder)
    store.build(_spec(), builder)                       # current → reused
    assert len(calls) == 1
    assert store.status(_spec())[0]
    store.build(_spec(version="2"), builder)            # version change → rebuild + archive
    assert len(calls) == 2
    assert any(store.archive_dir.iterdir())
    sidecar = json.loads((store.data_dir / "toy.schema.json").read_text(encoding="utf-8"))
    assert sidecar["grain"] == "one row per id" and sidecar["rows"] == 2 and "reload" in sidecar


def test_input_change_invalidates(synth):
    BronzeIngestor(synth).ingest(["games"])
    store = ArtifactStore(synth)
    store.build(_spec(), lambda: pl.DataFrame({"id": [1], "v": [1.0]}))
    f = synth.paths.data_root / "games.csv"
    f.write_text(f.read_text(encoding="utf-8") + "2023091001,3,2023,REG,1,BAL,HOU\n", encoding="utf-8")
    BronzeIngestor(synth).ingest(["games"])
    current, reason = store.status(_spec())
    assert not current and "inputs" in reason


def test_contract_violation_raises_and_quarantines(synth):
    BronzeIngestor(synth).ingest(["games"])
    store = ArtifactStore(synth)
    with pytest.raises(ArtifactValidationError, match="not unique"):
        store.build(_spec(), lambda: pl.DataFrame({"id": [1, 1], "v": [1.0, 2.0]}))
    assert not store.dataset_files("toy")


def test_partitioned_dataset_roundtrip(synth):
    BronzeIngestor(synth).ingest(["games"])
    store = ArtifactStore(synth)
    spec = DatasetSpec(name="parts", grain="id", key=("id",), sources=("bronze.games",), description="",
                       columns={"id": "", "season": ""}, partition_by="season")
    lf = store.build(spec, lambda: pl.DataFrame({"id": [1, 2, 3], "season": [2023, 2023, 2024]}))
    df = lf.collect().sort("id")
    assert df["season"].to_list() == [2023, 2023, 2024]
    assert (store.data_dir / "parts" / "season=2024").is_dir()


def test_csv_and_figure_registration(synth):
    import matplotlib
    matplotlib.use("Agg")
    from nflbdb.data.reporting import new_figure

    store = ArtifactStore(synth)
    store.export_csv(pl.DataFrame({"a": [1]}), "tiny")
    fig, ax = new_figure()
    ax.plot([0, 1], [0, 1])
    png = store.save_fig(fig, "tiny_fig", source="test", n=2)
    assert png.exists() and png.with_suffix(".svg").exists()
    kinds = set(store.manifest_frame()["kind"].to_list())
    assert {"tables", "figures"} <= kinds


# ----------------------------------------------------------------------------- remediation
def _phase01(synth):
    BronzeIngestor(synth).ingest()
    v = DataQualityValidator(synth)
    v.export(v.run())
    SilverBuilder(synth).build_all()
    GoldBuilder(synth).build_all()


def test_superseded_definition_is_detected(synth):
    q = QualityRemediation(synth)
    saved = pl.DataFrame({
        "check_id": ["uniq:combine_tracking:event_id, nfl_id, attempt, time",
                     "uniq:combine_tracking:event_id, nfl_id, attempt, entity_type, time",
                     "fk:game_tracking:game_id->games"],
        "severity": ["CRITICAL", "CRITICAL", "CRITICAL"],
    })
    status = q.audit_definitions(saved)["definition_status"].to_list()
    assert status[0].startswith("superseded") and status[1] == "current" and status[2] == "current"


def test_remediation_end_to_end_on_synthetic(synth):
    _phase01(synth)
    out = QualityRemediation(synth).run()
    store = ArtifactStore(synth)
    nfl = store.load(NFL_FLAGS.name).collect()
    cmb = store.load(COMBINE_FLAGS.name).collect()
    # planted gap: player 103 misses one frame in game 2023090700 / play 1
    assert nfl.filter(pl.col("nfl_id") == 103)["n_gaps"].sum() == 1
    # planted exact duplicate was removed in silver → no duplicate timestamps remain
    assert nfl["n_dup_ts"].sum() == 0
    assert nfl.select(pl.struct("game_id", "play_id", "nfl_id").n_unique()).item() == nfl.height
    assert cmb.height == 4 and not cmb["exclude_kinematics"].any()
    for name in ("quality_failures", "quality_remediation_summary", "quality_flags_summary"):
        assert (store.tables_dir / f"{name}.csv").exists()
    assert (synth.paths.reports_root / "09_quality_remediation.md").exists()
    assert set(out["summary"]["status"].to_list()) <= {
        "resolved_superseded_definition", "tracking_artifact", "expected_behaviour", "coverage_gap", "unresolved"}


def test_flags_are_reused_when_inputs_unchanged(synth):
    _phase01(synth)
    q = QualityRemediation(synth)
    q.build_flags()
    built_at = json.loads((q.store.data_dir / f"{NFL_FLAGS.name}.schema.json").read_text(encoding="utf-8"))["built_at"]
    q.build_flags()
    again = json.loads((q.store.data_dir / f"{NFL_FLAGS.name}.schema.json").read_text(encoding="utf-8"))["built_at"]
    assert built_at == again
