"""Regression tests: DuckDB→Polars conversion (Polars 2.0), mixed-type CSV columns, null handling,
profiling and schema discovery."""

from __future__ import annotations

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from nflbdb.data.bronze import BronzeIngestor
from nflbdb.data.csv_schema import SchemaResolutionError, SchemaResolver, load_resolved_schema, read_csv_strings
from nflbdb.data.frames import query_df, query_row, query_scalar, to_polars
from nflbdb.data.profile import TableProfiler
from nflbdb.data.schema import SchemaCatalog
from nflbdb.data.silver import SilverBuilder
from nflbdb.data.sources import SOURCES, SourceSpec
from nflbdb.data.tables import LayerRefs


# ----------------------------------------------------------------------------- conversions
@pytest.fixture()
def con():
    c = duckdb.connect()
    yield c
    c.close()


def test_from_arrow_on_relation_is_a_struct_series_but_helper_returns_dataframe(con):
    rel = con.sql("SELECT 1 AS a, 'x' AS b")
    raw = pl.from_arrow(rel)  # documents the Polars 2.0 behaviour the helper guards against
    assert isinstance(raw, (pl.Series, pl.DataFrame))
    df = to_polars(rel)
    assert isinstance(df, pl.DataFrame) and df.columns == ["a", "b"] and df.row(0) == (1, "x")


def test_struct_series_is_unnested(con):
    s = pl.Series("s", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    df = to_polars(s)
    assert isinstance(df, pl.DataFrame) and df.columns == ["a", "b"] and df.height == 2
    assert to_polars(pl.Series("v", [1, 2])).columns == ["v"]


def test_arrow_inputs_including_empty_reader(con):
    assert to_polars(pa.table({"a": [1]})).shape == (1, 1)
    reader = con.sql("SELECT 1 AS a WHERE FALSE").to_arrow_reader() if hasattr(
        con.sql("SELECT 1"), "to_arrow_reader") else con.sql("SELECT 1 AS a WHERE FALSE").fetch_record_batch()
    df = to_polars(reader)
    assert df.shape == (0, 1) and df.columns == ["a"]


def test_hugeint_and_decimal_are_normalised(con):
    df = query_df(con, "SELECT sum(range) AS s, count_if(range > 2) AS c, 1.25::DECIMAL(9,2) AS d FROM range(10)")
    assert df.schema["s"] == pl.Int64 and df.schema["c"] == pl.Int64 and df.schema["d"] == pl.Float64
    row = query_row(con, "SELECT sum(range) AS s FROM range(5)")
    assert row["s"] == 10 and type(row["s"]) is int


def test_hugeint_overflow_raises_instead_of_wrapping(con):
    with pytest.raises(pl.exceptions.PolarsError):
        query_df(con, "SELECT CAST('1180591620717411303424' AS HUGEINT) AS big")  # 2**70 > Int64 max


def test_query_helpers_on_empty_results(con):
    assert query_row(con, "SELECT 1 AS a WHERE FALSE") == {"a": None}
    assert query_scalar(con, "SELECT 1 AS a WHERE FALSE") is None
    assert query_scalar(con, "SELECT 42") == 42


# ----------------------------------------------------------------------------- CSV typing
def _spec(**schema) -> SourceSpec:
    return SourceSpec(name="t", pattern="t.csv", domain="test", description="", schema=schema)


def test_raw_strings_never_fail_and_preserve_values(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text('id,flag,amount,label\n1,FALSE,1.5,"a"\n2,NA,,""\n3,TRUE,2,b\n', encoding="utf-8")
    df = read_csv_strings(f)
    assert all(dt == pl.String for dt in df.dtypes)
    assert df["flag"].to_list() == ["FALSE", None, "TRUE"]      # NA → null, literals preserved
    assert df["label"].to_list() == ["a", "", "b"]               # empty string kept, distinct from NA


def test_boolean_literals_in_declared_float_column_resolve_to_boolean_like(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("id,recovered_fumble\n1,FALSE\n2,NA\n3,TRUE\n", encoding="utf-8")
    spec = _spec(id=pl.Int64, recovered_fumble=pl.Float64)
    schema, res = SchemaResolver().resolve_files(spec, [f])
    r = {x.column: x for x in res}["recovered_fumble"]
    assert r.action == "boolean_like" and r.n_cast_fail == 2 and sorted(r.fail_examples) == ["FALSE", "TRUE"]
    assert schema["recovered_fumble"] == pl.String and schema["id"] == pl.Int64


def test_mixed_numeric_and_boolean_is_kept_as_string_and_flagged(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("id,v\n1,0\n2,1\n3,FALSE\n", encoding="utf-8")
    schema, res = SchemaResolver().resolve_files(_spec(id=pl.Int64, v=pl.Float64), [f])
    r = {x.column: x for x in res}["v"]
    assert r.action == "mixed_numeric_boolean" and r.severity == "WARN" and schema["v"] == pl.String


def test_unexpected_value_raises_with_examples(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("id,v\n1,1.0\n2,n/a\n3,oops\n", encoding="utf-8")
    _, res = SchemaResolver().resolve_files(_spec(id=pl.Int64, v=pl.Float64), [f])
    with pytest.raises(SchemaResolutionError, match="oops"):
        SchemaResolver.raise_on_errors(res)


def test_blank_in_numeric_column_is_counted_as_missing(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text('id,v\n1,1.5\n2,""\n3, \n', encoding="utf-8")
    _, res = SchemaResolver().resolve_files(_spec(id=pl.Int64, v=pl.Float64), [f])
    r = {x.column: x for x in res}["v"]
    assert r.action == "keep" and r.n_empty == 2 and r.n_non_null == 1


def test_offset_sample_realigns_multiline_quoted_records(tmp_path):
    from nflbdb.data.sampling import offset_sample

    f = tmp_path / "pp.csv"
    rows = [f'{i},"line one of play {i}.\nline two of play {i}.",{i % 4}' for i in range(2000)]
    f.write_text("game_id,play_description,down\n" + "\n".join(rows) + "\n", encoding="utf-8")
    spec = _spec(game_id=pl.Int64, play_description=pl.String, down=pl.Float64)
    df = offset_sample(f, spec, n_chunks=10, lines_per_chunk=101, seed=3)
    assert df["_sample_chunk"].n_unique() == 10                 # no chunk skipped
    assert df["game_id"].cast(pl.Int64, strict=True).null_count() == 0
    assert df["play_description"].str.contains("\n").all()      # records not split


# ----------------------------------------------------------------------------- pipeline-level
def test_bronze_keeps_boolean_literals_and_silver_casts_them(synth):
    BronzeIngestor(synth).ingest()
    refs = LayerRefs(synth)
    bronze = refs.scan("bronze", "player_play").collect()
    assert bronze.schema["recovered_fumble"] == pl.String
    assert bronze["recovered_fumble"].to_list() == ["FALSE", None, "TRUE", None, None]
    assert bronze["receiver_alignment"].to_list()[-2:] == ["", ""]  # empty strings survive bronze
    assert load_resolved_schema(synth.paths.catalog, "player_play")["recovered_fumble"] == pl.String
    assert (synth.paths.tables / "schema_resolution.csv").exists()

    SilverBuilder(synth).build_all()
    pp = refs.scan("silver", "player_play").collect()
    plays = refs.scan("silver", "plays").collect()
    holder = pp if "recovered_fumble" in pp.columns else plays
    assert holder.schema["recovered_fumble"] == pl.Boolean
    assert holder["recovered_fumble"].drop_nulls().sort().to_list() == [False, True]  # no value lost
    assert "lined_up_position" in pp.columns and "team_abbr" in pp.columns          # role attrs stay player-level


def test_bronze_strict_cast_fails_loudly_on_unknown_value(synth):
    f = synth.paths.data_root / "games.csv"
    df = read_csv_strings(f).with_columns(pl.lit("week-one").alias("week"))
    df.write_csv(f, null_value="NA")
    with pytest.raises(SchemaResolutionError, match="week-one"):
        BronzeIngestor(synth).ingest(["games"])


def test_schema_catalog_reports_resolution(synth):
    cat, samples = SchemaCatalog(synth).build()
    assert all(dt == pl.String for s in samples.values() for c, dt in s.schema.items() if c != "_sample_chunk")
    rf = cat.filter((pl.col("table") == "player_play") & (pl.col("column") == "recovered_fumble")).row(0, named=True)
    assert rf["declared_dtype"] == "Float64" and rf["bronze_dtype"] == "String" and rf["type_resolution"] == "boolean_like"
    SchemaCatalog(synth).write_report(cat)


def test_profiler_summarize_and_sampled_top_values(synth):
    BronzeIngestor(synth).ingest()
    tp = TableProfiler(synth)
    s = tp.summarize("games")
    assert isinstance(s, pl.DataFrame) and {"column_name", "table", "profile_mode"} <= set(s.columns)
    assert s.schema["null_percentage"] == pl.Float64  # DECIMAL(9,2) normalised
    top = tp.top_values("game_tracking", "event", sample_pct=50.0)  # sample clause + GROUP BY (was a parser error)
    assert isinstance(top, pl.DataFrame)
    summary, tops = tp.profile_all()
    assert set(summary["table"].unique()) == set(SOURCES)
    tp.export(summary, tops, tp.combine_test_participation(), tp.tracking_drill_counts())
    assert (synth.paths.tables / "profile_summary.csv").exists()
