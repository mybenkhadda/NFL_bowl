"""Evidence-based CSV typing: read raw values as strings, resolve types explicitly, cast strictly.

Raw CSVs are always parsed with ``infer_schema=False`` (every column = String), so parsing can
never fail or silently drop values. Declared types (``sources.SourceSpec.schema``) are then
*tested* per column and resolved with explicit, recorded rules:

==========================  ========================================================  =============
Observation (non-null)       Resolution                                                Severity
==========================  ========================================================  =============
all values cast cleanly      keep declared type                                        —
all values TRUE/FALSE        keep as String in bronze → BOOLEAN in silver              INFO
numbers mixed with T/F       keep as String (no coercion); reported                    WARN
any other unparseable value  raise :class:`SchemaResolutionError` with examples        ERROR
==========================  ========================================================  =============

Empty strings in typed (non-String) columns are treated as missing and **counted** in the
resolution report; in String columns they are preserved as-is (distinct from ``NA``).

Casting uses ``strict=True``: a value that was not seen during resolution (e.g. a tracking
file resolved on a bounded sample) makes the bronze write fail loudly — never a silent null.
``ignore_errors`` is never used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

from .sources import SOURCES, SourceSpec
from .utils import atomic_write_json, get_logger, read_json, utc_now

log = get_logger(__name__)

BOOL_TOKENS = frozenset({"TRUE", "FALSE"})
STRING = pl.String


class SchemaResolutionError(RuntimeError):
    """A declared non-string column contains values that are neither castable nor a known token."""


@dataclass
class ColumnResolution:
    table: str
    column: str
    declared: str
    resolved: str
    n_rows: int
    n_non_null: int
    n_empty: int
    n_cast_fail: int
    fail_examples: list[str] = field(default_factory=list)
    action: str = "keep"          # keep | boolean_like | mixed_numeric_boolean | error
    severity: str = "OK"
    evidence: str = ""            # full scan | bounded sample (n rows)


def read_csv_strings(source, *, n_rows: int | None = None, null_values=("NA",), lazy: bool = False):
    """Parse a CSV (path or bytes buffer) with every column as String. Never fails on values."""
    kwargs = dict(infer_schema=False, null_values=list(null_values))
    if lazy:
        return pl.scan_csv(source, **kwargs)
    return pl.read_csv(source, n_rows=n_rows, **kwargs)


def _is_typed(dt: pl.DataType) -> bool:
    return dt != STRING


def blank_to_null(col: str) -> pl.Expr:
    c = pl.col(col)
    return pl.when(c.str.strip_chars() == "").then(None).otherwise(c)


def strict_cast_exprs(schema: dict[str, pl.DataType], columns: list[str] | None = None) -> list[pl.Expr]:
    """String → declared type with ``strict=True``; blanks in typed columns become null."""
    cols = columns or list(schema)
    return [blank_to_null(c).cast(schema[c], strict=True).alias(c) for c in cols if c in schema and _is_typed(schema[c])]


class SchemaResolver:
    def __init__(self, catalog_dir: Path | None = None, max_examples: int = 10):
        self.catalog_dir = catalog_dir
        self.max_examples = max_examples

    # ------------------------------------------------------------------ diagnosis
    def diagnose(self, frame: pl.DataFrame | pl.LazyFrame, spec: SourceSpec, evidence: str) -> list[ColumnResolution]:
        """``frame`` must be all-String (see :func:`read_csv_strings`). One aggregation pass for
        the counts, then one small pass per failing column for examples."""
        lf = frame.lazy()
        names = lf.collect_schema().names()
        typed = [c for c, dt in spec.schema.items() if _is_typed(dt) and c in names]
        if not typed:
            return []
        aggs = [pl.len().alias("__n")]
        for i, c in enumerate(typed):
            v = blank_to_null(c)
            aggs += [
                v.is_not_null().sum().alias(f"nn{i}"),
                (pl.col(c).str.strip_chars() == "").sum().alias(f"em{i}"),
                (v.is_not_null() & v.cast(spec.schema[c], strict=False).is_null()).sum().alias(f"bad{i}"),
            ]
        stats = _collect(lf.select(aggs)).row(0, named=True)
        out = []
        for i, c in enumerate(typed):
            dt = spec.schema[c]
            res = ColumnResolution(spec.name, c, str(dt), str(dt), int(stats["__n"]), int(stats[f"nn{i}"]),
                                   int(stats[f"em{i}"]), int(stats[f"bad{i}"]), evidence=evidence)
            if res.n_cast_fail:
                v = blank_to_null(c)
                bad = _collect(lf.select(v.alias(c)).filter(pl.col(c).is_not_null()
                                                            & pl.col(c).cast(dt, strict=False).is_null())
                               .unique().sort(c).head(self.max_examples))
                res.fail_examples = bad[c].to_list()
                failing_upper = {str(x).strip().upper() for x in res.fail_examples}
                all_bad_are_bool = res.n_cast_fail == res.n_non_null
                if failing_upper <= BOOL_TOKENS and len(res.fail_examples) < self.max_examples and all_bad_are_bool:
                    res.action, res.resolved, res.severity = "boolean_like", str(STRING), "INFO"
                elif failing_upper <= BOOL_TOKENS and len(res.fail_examples) < self.max_examples:
                    res.action, res.resolved, res.severity = "mixed_numeric_boolean", str(STRING), "WARN"
                else:
                    res.action, res.severity = "error", "ERROR"
            out.append(res)
        return out

    # ------------------------------------------------------------------ resolution
    @staticmethod
    def resolved_schema(spec: SourceSpec, resolutions: list[ColumnResolution],
                        header: list[str] | None = None) -> dict[str, pl.DataType]:
        """Declared schema with resolved overrides; undeclared header columns stay String."""
        override = {r.column: STRING for r in resolutions if r.action in ("boolean_like", "mixed_numeric_boolean")}
        cols = header or spec.columns
        return {c: override.get(c, spec.schema.get(c, STRING)) for c in cols}

    @staticmethod
    def raise_on_errors(resolutions: list[ColumnResolution]) -> None:
        bad = [r for r in resolutions if r.action == "error"]
        if bad:
            detail = "; ".join(f"{r.table}.{r.column} (declared {r.declared}): {r.n_cast_fail:,} unparseable, "
                               f"e.g. {r.fail_examples[:5]}" for r in bad)
            raise SchemaResolutionError(
                f"{len(bad)} column(s) contain values that are neither {bad[0].declared}-castable nor TRUE/FALSE: "
                f"{detail}. Fix the declared type in sources.py or add an explicit null token to null_values."
            )

    def resolve_files(self, spec: SourceSpec, files: list[Path], sample_rows: int | None = None,
                      sample_frames: list[pl.DataFrame] | None = None) -> tuple[dict[str, pl.DataType], list[ColumnResolution]]:
        """Full scan (``sample_rows=None``) or bounded evidence. Multiple files of one source are
        resolved together so they share one schema."""
        if sample_frames is not None:
            frame = pl.concat([f.select([c for c in f.columns if not c.startswith("_")]) for f in sample_frames],
                              how="diagonal")
            evidence = f"bounded sample ({frame.height:,} rows)"
        elif sample_rows:
            frame = pl.concat([read_csv_strings(f, n_rows=sample_rows, null_values=spec.null_values) for f in files],
                              how="diagonal")
            evidence = f"head sample ({frame.height:,} rows)"
        else:
            frame = pl.concat([read_csv_strings(f, null_values=spec.null_values, lazy=True) for f in files],
                              how="diagonal")
            evidence = "full scan"
        res = self.diagnose(frame, spec, evidence)
        return self.resolved_schema(spec, res), res

    # ------------------------------------------------------------------ persistence
    def save(self, spec: SourceSpec, schema: dict[str, pl.DataType], resolutions: list[ColumnResolution]) -> None:
        if not self.catalog_dir:
            return
        path = self.catalog_dir / "resolved_schemas.json"
        state = read_json(path, {})
        state[spec.name] = {
            "resolved_at": utc_now(),
            "schema": {c: str(t) for c, t in schema.items()},
            "resolutions": [asdict(r) for r in resolutions if r.action != "keep" or r.n_empty],
        }
        atomic_write_json(state, path)


def load_resolved_schema(catalog_dir: Path, name: str) -> dict[str, pl.DataType]:
    """Schema actually used for bronze (falls back to the declared registry schema)."""
    state = read_json(catalog_dir / "resolved_schemas.json", {})
    entry = state.get(name)
    if not entry:
        return dict(SOURCES[name].schema)
    return {c: _dtype(t) for c, t in entry["schema"].items()}


def _dtype(name: str) -> pl.DataType:
    return {"Int64": pl.Int64, "Float64": pl.Float64, "String": pl.String, "Utf8": pl.String,
            "Boolean": pl.Boolean, "Int32": pl.Int32}[name]


def resolutions_frame(resolutions: list[ColumnResolution]) -> pl.DataFrame:
    if not resolutions:
        return pl.DataFrame(schema={"table": pl.String, "column": pl.String})
    return pl.DataFrame([asdict(r) for r in resolutions], infer_schema_length=None)


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:  # pragma: no cover - older Polars
        return lf.collect()
