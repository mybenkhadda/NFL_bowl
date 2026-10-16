"""Analytics Lab · SQL regressions (UNPIVOT INCLUDE NULLS parser error, NULL preservation, mart schemas).

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_lab_sql_regressions.py -q
The first group needs only an in-memory DuckDB; the last two build the synthetic mini-warehouse.
"""

from __future__ import annotations

import inspect
import re

import duckdb
import polars as pl
import pytest

from nflbdb.data.frames import query_df
from nflbdb.lab import marts
from nflbdb.lab.marts import LabArtifactStore, traditional_test_availability_sql

TA_COLUMNS = ["position_group", "cohort", "test_name", "unit", "n_athletes", "n_with_value",
              "share_available", "median_value"]
CM_COLUMNS = ["position_group", "cohort", "source", "source_order", "n_athletes", "n_with_source",
              "share_with_source"]


@pytest.fixture()
def con():
    c = duckdb.connect()
    # 3 WR athletes with combine results (one NULL forty, all NULL vertical) + 1 without (excluded).
    c.execute("""
        CREATE TABLE lab_athlete AS SELECT * FROM (VALUES
            ('WR', 2023, TRUE,  4.40, NULL::DOUBLE, 10),
            ('WR', 2023, TRUE,  4.50, NULL::DOUBLE, NULL),
            ('WR', 2023, TRUE,  NULL, NULL::DOUBLE, 20),
            ('WR', 2023, FALSE, 4.30, NULL::DOUBLE, 30)
        ) t(position_group, cohort, has_combine_results, forty, vertical, bench_reps)
    """)
    yield c
    c.close()


# ----------------------------------------------------------------------------- static audit
def test_no_unpivot_include_nulls_left_in_lab_sql():
    src = inspect.getsource(marts)
    assert not re.search(r"UNPIVOT\s+INCLUDE\s+NULLS", src, re.IGNORECASE)


# ----------------------------------------------------------------------------- SQL unit tests
def test_test_availability_sql_parses_and_returns_dataframe(con):
    df = query_df(con, traditional_test_availability_sql(["forty", "vertical", "bench_reps"]))
    assert isinstance(df, pl.DataFrame)            # Polars 2.0: never a struct Series
    assert df.columns == TA_COLUMNS
    assert df.height == 3


def test_test_availability_preserves_nulls(con):
    df = query_df(con, traditional_test_availability_sql(["forty", "vertical", "bench_reps"]))
    rows = {r["test_name"]: r for r in df.iter_rows(named=True)}
    # has_combine_results = FALSE row excluded → 3 athletes for every test, NULLs counted.
    assert all(r["n_athletes"] == 3 for r in rows.values())
    assert rows["forty"]["n_with_value"] == 2 and abs(rows["forty"]["share_available"] - 2 / 3) < 1e-9
    assert abs(rows["forty"]["median_value"] - 4.45) < 1e-9
    assert rows["vertical"]["n_with_value"] == 0 and rows["vertical"]["share_available"] == 0.0
    assert rows["vertical"]["median_value"] is None   # all-NULL test kept, not dropped
    assert rows["bench_reps"]["n_with_value"] == 2


def test_test_availability_units_and_dtypes(con):
    df = query_df(con, traditional_test_availability_sql(["forty", "bench_reps"]))
    assert dict(zip(df["test_name"], df["unit"])) == {"forty": "s", "bench_reps": "reps"}
    assert df.schema["n_athletes"] == pl.Int64 and df.schema["n_with_value"] == pl.Int64
    assert df.schema["share_available"] == pl.Float64 and df.schema["median_value"] == pl.Float64
    assert df.schema["test_name"] == pl.String and df.schema["unit"] == pl.String


def test_test_availability_single_test(con):
    df = query_df(con, traditional_test_availability_sql(["forty"]))
    assert df.height == 1 and df["n_athletes"][0] == 3


# ----------------------------------------------------------------------------- built marts (synthetic warehouse)
def _mart(cfg, name: str) -> pl.DataFrame:
    files = LabArtifactStore(cfg).dataset_files(name)
    assert files, f"mart {name} not persisted"
    c = duckdb.connect()
    try:
        listing = ", ".join(f"'{f.as_posix()}'" for f in files)
        return query_df(c, f"SELECT * FROM read_parquet([{listing}])")
    finally:
        c.close()


@pytest.fixture()
def built(synth):
    from nflbdb.data.bronze import BronzeIngestor  # noqa: PLC0415
    from nflbdb.data.gold import GoldBuilder  # noqa: PLC0415
    from nflbdb.data.silver import SilverBuilder  # noqa: PLC0415
    from nflbdb.data.validate import DataQualityValidator  # noqa: PLC0415
    from nflbdb.lab.build import run  # noqa: PLC0415

    BronzeIngestor(synth).ingest()      # same synthetic Phase 01 build as test_analytics_lab._phase01
    v = DataQualityValidator(synth)
    v.export(v.run())
    SilverBuilder(synth).build_all()
    GoldBuilder(synth).build_all()
    run(synth, steps=("marts",))
    return synth


def test_test_availability_mart_schema_and_nulls(built):
    df = _mart(built, "test_availability")
    assert df.columns == TA_COLUMNS
    forty = df.filter(pl.col("test_name") == "forty")
    # Synthetic combine_results: 101, 102 have forty; 103 has NULL → kept as an athlete without a value.
    assert forty["n_athletes"].sum() == 3 and forty["n_with_value"].sum() == 2
    assert df["position_group"].null_count() == 0 and df["cohort"].null_count() == 0


def test_coverage_matrix_counts_every_athlete_per_source(built):
    df = _mart(built, "coverage_matrix")
    assert df.columns == CM_COLUMNS
    per_source = dict(df.group_by("source").agg(pl.col("n_athletes").sum()).iter_rows())
    assert set(per_source) == {"combine_results", "combine_tracking", "nfl_tracking", "player_play", "career",
                               "is_matched"}
    assert set(per_source.values()) == {4}          # synthetic players 101–104; NULL flags not dropped
