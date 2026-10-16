"""Bounded, deterministic access to raw CSVs — never reads a whole file.

* :func:`estimate_csv_rows` — row estimate from the average line length of the first N bytes
  (exact when the file is smaller than N).
* :func:`head_sample` — first N rows (cheap, but biased toward the start of the file:
  first game / first drill session).
* :func:`offset_sample` — reads short runs of lines at evenly spaced, seed-jittered byte
  offsets across the whole file. Deterministic for a given seed and file, and representative
  of the file's full extent (all seasons/weeks/sessions) at a cost of a few MB of IO.
"""

from __future__ import annotations

import csv
import gzip
import io
import random
from pathlib import Path

import polars as pl

from .csv_schema import read_csv_strings
from .sources import SourceSpec
from .utils import get_logger

log = get_logger(__name__)


def _open(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def read_header(path: Path) -> list[str]:
    with _open(path) as fh:
        first = fh.readline().decode("utf-8-sig")
    return next(csv.reader([first]))


def estimate_csv_rows(path: Path, sample_bytes: int = 8 * 1024**2) -> tuple[int, float, bool]:
    """Return ``(rows, avg_line_bytes, is_exact)``. Compressed files return an estimate based
    on the uncompressed sample and the *compressed* size, flagged as not exact."""
    size = path.stat().st_size
    with _open(path) as fh:
        fh.readline()  # header
        chunk = fh.read(sample_bytes)
        at_eof = fh.read(1) == b""
    n_lines = chunk.count(b"\n") + (1 if chunk and not chunk.endswith(b"\n") else 0)
    if at_eof and path.suffix != ".gz":
        return n_lines, (len(chunk) / n_lines if n_lines else 0.0), True
    avg = len(chunk) / max(chunk.count(b"\n"), 1)
    if path.suffix == ".gz":
        return int(size / avg), avg, False  # lower bound-ish: compressed bytes / raw line length
    return int(size / avg), avg, False


def _nulls(spec: SourceSpec | None) -> tuple[str, ...]:
    return tuple(spec.null_values) if spec else ("NA",)


def _read_csv_bytes(buf: bytes, spec: SourceSpec | None) -> pl.DataFrame:
    """All columns as raw strings — typing happens later and explicitly (csv_schema)."""
    return read_csv_strings(io.BytesIO(buf), null_values=_nulls(spec))


def head_sample(path: Path, spec: SourceSpec | None, n_rows: int, infer: bool = False) -> pl.DataFrame:
    """First ``n_rows`` rows as raw strings. ``infer=True`` instead returns Polars' own dtype
    inference over exactly those rows (informational only; it cannot fail because inference
    and parsing see the same rows)."""
    if infer:
        return pl.read_csv(path, n_rows=n_rows, null_values=list(_nulls(spec)), infer_schema_length=n_rows)
    return read_csv_strings(path, n_rows=n_rows, null_values=_nulls(spec))


def _parse_chunk(header: bytes, lines: list[bytes], spec: SourceSpec | None,
                 max_shift: int = 20) -> tuple[pl.DataFrame | None, str]:
    """Parse a run of raw lines that may start/end inside a multi-line quoted record.

    * End: drop trailing lines until the double-quote count is even (no open quoted field).
    * Start: try successive start lines; accept the first parse whose width matches the header
      and whose first declared non-string column (an identifier) casts cleanly on every row —
      a misaligned start shows up as text in the id column.
    """
    n_cols = len(header.decode("utf-8-sig").split(","))  # header names never contain commas here
    first_typed = next((c for c, dt in spec.schema.items() if dt != pl.String), None) if spec else None
    err = "no candidate start line"
    for shift in range(min(max_shift, len(lines))):
        cand = lines[shift:]
        quotes = sum(line.count(b'"') for line in cand)
        while cand and quotes % 2:
            quotes -= cand[-1].count(b'"')
            cand = cand[:-1]
        if not cand:
            continue
        try:
            df = _read_csv_bytes(header + b"".join(cand), spec)
        except Exception as exc:  # noqa: BLE001 - try the next alignment
            err = str(exc).splitlines()[0]
            continue
        if df.width != n_cols:
            err = f"width {df.width} != {n_cols}"
            continue
        if first_typed and first_typed in df.columns:
            bad = df.select(pl.col(first_typed).is_not_null() & pl.col(first_typed).cast(spec.schema[first_typed], strict=False).is_null()).to_series().sum()
            if bad:
                err = f"{bad} rows with non-{spec.schema[first_typed]} {first_typed}"
                continue
        return df, ""
    return None, err


def offset_sample(
    path: Path,
    spec: SourceSpec | None,
    n_chunks: int = 20,
    lines_per_chunk: int = 2_000,
    seed: int = 42,
) -> pl.DataFrame:
    """Deterministic byte-offset sample of raw strings. Adds ``_sample_chunk`` (0..n_chunks-1).

    Quoted fields may contain newlines (e.g. ``player_play.play_description``), so a byte offset
    can land inside a record. Each chunk is realigned by :func:`_parse_chunk`; a chunk that still
    cannot be aligned is skipped and logged rather than silently mis-parsed.
    """
    if path.suffix == ".gz":
        log.warning("%s is compressed: offset sampling not possible, using head sample", path.name)
        return head_sample(path, spec, n_chunks * lines_per_chunk).with_columns(pl.lit(0).alias("_sample_chunk"))

    size = path.stat().st_size
    rng = random.Random(f"{seed}:{path.name}:{size}")
    frames: list[pl.DataFrame] = []
    with path.open("rb") as fh:
        header = fh.readline()
        start = fh.tell()
        span = max(size - start, 1)
        step = span / n_chunks
        for i in range(n_chunks):
            offset = int(start + i * step + rng.random() * step * 0.5)
            fh.seek(offset)
            if offset != start:
                fh.readline()  # discard the partial line we landed in
            lines = []
            for _ in range(lines_per_chunk):
                line = fh.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                continue
            df, err = _parse_chunk(header, lines, spec)
            if df is None:
                log.warning("%s chunk %d skipped (could not align records at offset %d): %s", path.name, i, offset, err)
                continue
            frames.append(df.with_columns(pl.lit(i, dtype=pl.Int32).alias("_sample_chunk")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")
