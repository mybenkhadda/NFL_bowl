"""Tests for critical transformations on the synthetic dataset (see conftest.py)."""

from __future__ import annotations

import polars as pl
import pytest

from nflbdb.data.bronze import BronzeIngestor, SchemaDriftError
from nflbdb.data.config import duckdb_connect
from nflbdb.data.discover import DataInventory
from nflbdb.data.relationships import JoinPlan, RelationshipAnalyzer
from nflbdb.data.sampling import estimate_csv_rows, offset_sample
from nflbdb.data.silver import SilverBuilder
from nflbdb.data.sources import FOREIGN_KEYS, SOURCES
from nflbdb.data.tables import LayerRefs
from nflbdb.data.tracking_qc import TrackingQC
from nflbdb.data.utils import atomic_write_text
from nflbdb.data.validate import DataQualityValidator


def test_atomic_write_replaces(tmp_path):
    p = tmp_path / "a.txt"
    atomic_write_text("one", p)
    atomic_write_text("two", p)
    assert p.read_text() == "two"
    assert [f.name for f in tmp_path.iterdir()] == ["a.txt"]


def test_row_estimate_exact_for_small_file(synth):
    f = synth.paths.data_root / "games.csv"
    rows, _, exact = estimate_csv_rows(f)
    assert exact and rows == 2


def test_offset_sample_is_deterministic(synth):
    f = synth.paths.data_root / "game_tracking_2023.csv"
    a = offset_sample(f, SOURCES["game_tracking"], n_chunks=3, lines_per_chunk=5, seed=1)
    b = offset_sample(f, SOURCES["game_tracking"], n_chunks=3, lines_per_chunk=5, seed=1)
    assert a.equals(b) and a.height > 0


def test_inventory_matches_registry_and_partitions(synth):
    inv = DataInventory(synth).run()
    assert inv.filter(pl.col("source").is_null()).is_empty()
    gt = inv.filter(pl.col("source") == "game_tracking").row(0, named=True)
    assert gt["header_matches_spec"] and "season=2023" in gt["partition_info"]


def test_bronze_lossless_and_incremental(synth):
    b = BronzeIngestor(synth)
    first = b.ingest()
    assert set(first["action"]) == {"ingested"}
    raw = pl.read_csv(synth.paths.data_root / "game_tracking_2023.csv", null_values=["NA"])
    out = LayerRefs(synth).scan("bronze", "game_tracking").collect()
    assert out.height == raw.height
    assert {"_src_row", "_src_file", "season"} <= set(out.columns)
    second = b.ingest()
    assert set(second["action"]) == {"skipped"}


def test_bronze_schema_drift_raises(synth):
    f = synth.paths.data_root / "games.csv"
    pl.read_csv(f).drop("week").write_csv(f)
    with pytest.raises(SchemaDriftError):
        BronzeIngestor(synth).ingest(["games"])


def test_validator_detects_duplicates_and_orphans(synth):
    BronzeIngestor(synth).ingest()
    v = DataQualityValidator(synth)
    r = v.key_uniqueness("game_tracking", ("game_id", "play_id", "nfl_id", "time"))
    assert r.result == "FAIL" and r.value == 1  # one planted duplicate row (nfl_id 102, first frame)
    fk = next(f for f in FOREIGN_KEYS if f.child == "game_tracking" and f.parent == "players")
    assert v.referential_integrity(fk).result == "PASS"
    # career has 101, 102 → PASS; drop 102 from players → orphan detected
    players = synth.paths.data_root / "players.csv"
    pl.read_csv(players, null_values=["NA"]).filter(pl.col("nfl_id") != 102).write_csv(players, null_value="NA")
    BronzeIngestor(synth).ingest(["players"])
    cr = next(f for f in FOREIGN_KEYS if f.child == "player_career_successes")
    res = v.referential_integrity(cr)
    assert res.result == "FAIL" and res.value == 1


def test_tracking_qc_finds_gap(synth):
    BronzeIngestor(synth).ingest()
    m = TrackingQC(synth).trajectory_metrics("game", sample_pct=0)
    gaps = m.filter(pl.col("nfl_id") == 103)["n_gaps"].sum()
    assert gaps == 1


def test_join_cardinality_matches_actual_join(synth):
    BronzeIngestor(synth).ingest()
    ra = RelationshipAnalyzer(synth)
    plan = JoinPlan("game_tracking", "player_play", ("game_id", "play_id"), "test")
    est = ra.join_cardinality([plan]).row(0, named=True)["est_inner_rows"]
    refs = LayerRefs(synth)
    actual = duckdb_connect(synth).sql(
        f"SELECT count(*) FROM {refs.sql('bronze', 'game_tracking')} t JOIN {refs.sql('bronze', 'player_play')} p "
        f"USING (game_id, play_id)").fetchone()[0]
    assert est == actual


def test_silver_split_dedup_frames_and_standardisation(synth):
    BronzeIngestor(synth).ingest()
    sb = SilverBuilder(synth)
    sb.build_all()
    refs = LayerRefs(synth)
    plays = refs.scan("silver", "plays").collect()
    assert plays.height == 2 and "play_direction" in plays.columns and "game_clock_seconds" in plays.columns
    pp = refs.scan("silver", "player_play").collect()
    assert "target" in pp.columns and pp.schema["target"] == pl.Boolean
    gt = refs.scan("silver", "game_tracking").collect()
    raw_rows = refs.num_rows("bronze", "game_tracking")
    assert gt.height < raw_rows  # exact duplicates dropped
    assert (synth.paths.silver / "_quarantine").exists()
    # frame_id aligned to play start, gap preserved (frame 6 missing for player 103)
    f103 = gt.filter(pl.col("nfl_id") == 103).sort("ts")["frame_id"].to_list()
    assert f103 == [1, 2, 3, 4, 5, 7, 8, 9, 10]
    # left-direction play rotated: x_std = 120 - x
    left = gt.filter((pl.col("play_id") == 1) & pl.col("nfl_id").is_not_null()).head(1).row(0, named=True)
    assert left["x_std"] == pytest.approx(120 - left["x"]) and left["dir_std"] == pytest.approx(270.0)
    assert gt["is_football"].sum() > 0
    ct = refs.scan("silver", "combine_tracking").collect()
    assert ct.group_by("nfl_id", "attempt").agg(pl.col("frame_id").min())["frame_id"].to_list() == [1] * 4


def test_gold_end_to_end(synth):
    from nflbdb.data.gold import GoldBuilder

    BronzeIngestor(synth).ingest()
    SilverBuilder(synth).build_all()
    man = GoldBuilder(synth).build_all()
    assert man["pk_unique"].all()
    refs = LayerRefs(synth)
    dim = refs.scan("gold", "dim_athlete").collect()
    assert dim.filter(pl.col("nfl_id") == 101)["has_nfl_tracking"][0]
    rb = refs.scan("gold", "mart_athlete_research_base").collect()
    assert "career_career_games_active" in rb.columns
