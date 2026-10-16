"""Single place where query results become Polars DataFrames.

Why this exists (Polars 2.0 + DuckDB 1.5):

* ``pl.from_arrow(x)`` returns a **struct Series** (not a DataFrame) when ``x`` is any object
  exposing the Arrow PyCapsule stream interface that is not a pyarrow object — e.g. a DuckDB
  relation. ``DataFrame`` methods then fail with ``'Series' object has no attribute …``.
* DuckDB ``sum()`` / ``count_if()`` over integers may return HUGEINT, which arrives in Polars
  as ``Decimal(38, 0)``; DuckDB ``SUMMARIZE`` returns ``DECIMAL(9, 2)`` percentages. Python rows
  then contain ``decimal.Decimal`` objects that break arithmetic/formatting downstream.

Every module calls :func:`query_df` / :func:`query_row` / :func:`query_scalar` /
:func:`to_polars` instead of ``.pl()`` / ``pl.from_arrow()`` directly. Data stays in Arrow
memory end-to-end (no pandas, no Python lists).
"""

from __future__ import annotations

from typing import Any

import polars as pl


def normalize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Decimal(p, 0) / Int128 → Int64 (strict: overflow raises, never wraps);
    Decimal(p, s>0) → Float64. Other dtypes are untouched."""
    exprs = []
    for name, dt in df.schema.items():
        if isinstance(dt, pl.Decimal):
            target = pl.Int64 if (dt.scale or 0) == 0 else pl.Float64
            exprs.append(pl.col(name).cast(target, strict=True))
        elif dt == pl.Int128:
            exprs.append(pl.col(name).cast(pl.Int64, strict=True))
    return df.with_columns(exprs) if exprs else df


def to_polars(obj: Any) -> pl.DataFrame:
    """Coerce a DuckDB relation, pyarrow Table/RecordBatch/RecordBatchReader, Arrow stream
    capsule, Polars Series (struct or plain) or DataFrame into a ``pl.DataFrame``."""
    if isinstance(obj, pl.DataFrame):
        df = obj
    elif isinstance(obj, pl.Series):
        df = obj.struct.unnest() if obj.dtype == pl.Struct else obj.to_frame()
    elif hasattr(obj, "pl") and hasattr(obj, "to_arrow_table"):  # DuckDB relation
        out = obj.pl()
        return to_polars(out)  # .pl() may itself yield a Series in some Polars builds
    else:
        import pyarrow as pa

        if isinstance(obj, pa.RecordBatchReader):
            obj = obj.read_all()  # keeps the schema even for 0 batches
        if isinstance(obj, (pa.Table, pa.RecordBatch)):
            df = pl.DataFrame(obj)
        elif hasattr(obj, "__arrow_c_stream__"):
            df = pl.DataFrame(obj)  # DataFrame constructor, NOT from_arrow (→ struct Series)
        else:
            raise TypeError(f"cannot convert {type(obj).__name__} to a Polars DataFrame")
    if isinstance(df, pl.Series):  # defensive: never leak a Series
        df = df.struct.unnest() if df.dtype == pl.Struct else df.to_frame()
    return normalize_dtypes(df)


def query_df(con, sql: str) -> pl.DataFrame:
    return to_polars(con.sql(sql))


def query_row(con, sql: str) -> dict[str, Any]:
    """First row as a dict (plain Python ints/floats, never Decimal). Empty result → all None."""
    df = query_df(con, sql)
    if df.is_empty():
        return {c: None for c in df.columns}
    return df.row(0, named=True)


def query_scalar(con, sql: str) -> Any:
    df = query_df(con, sql)
    return None if df.is_empty() else df.item(0, 0)


def query_column(con, sql: str) -> list[Any]:
    """First column as a list — only for small results (e.g. distinct partition values)."""
    df = query_df(con, sql)
    return df.to_series(0).to_list() if df.width else []
