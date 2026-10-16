"""Shared utilities: logging, timing + peak-memory monitoring, atomic writes, checksums."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import psutil

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "nflbdb") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("nflbdb").handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        root = logging.getLogger("nflbdb")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logger


log = get_logger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------- memory
def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


class PeakMemoryMonitor:
    """Samples the RSS of a process on a background thread and keeps the peak.

    Polars and DuckDB allocate natively inside this process, so RSS captures their usage
    (Python-level tracemalloc would not).
    """

    def __init__(self, pid: int | None = None, interval_s: float = 0.05):
        self._proc = psutil.Process(pid or os.getpid())
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_mb = self.peak_mb = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 1024**2)
            except psutil.Error:
                break
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakMemoryMonitor":
        self.start_mb = self.peak_mb = self._proc.memory_info().rss / 1024**2
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


# --------------------------------------------------------------------------- run log
@dataclass
class StepMetrics:
    step: str
    started_at: str = ""
    seconds: float = 0.0
    rss_start_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_end_mb: float = 0.0
    rows_in: int | None = None
    rows_out: int | None = None
    bytes_in: int | None = None
    bytes_out: int | None = None
    artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "RUNNING"
    error: str = ""

    def add_artifact(self, path: Path | str) -> None:
        self.artifacts.append(str(path))

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        log.warning("[%s] %s", self.step, message)


class RunLog:
    """Collects per-step timing, memory, row counts, artifacts and warnings.

    Every significant operation is wrapped in ``with runlog.step("name") as m:`` so that the
    notebook and the CLI report the same evidence; ``save()`` appends to a Parquet + CSV log.
    """

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        self.steps: list[StepMetrics] = []
        self._saved = 0

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[StepMetrics]:
        m = StepMetrics(step=name, started_at=utc_now())
        t0 = time.perf_counter()
        with PeakMemoryMonitor() as mon:
            m.rss_start_mb = mon.start_mb
            try:
                yield m
                m.status = "OK" if not m.warnings else "OK_WITH_WARNINGS"
            except Exception as exc:  # noqa: BLE001 - record then re-raise
                m.status, m.error = "FAILED", f"{type(exc).__name__}: {exc}"
                raise
            finally:
                m.seconds = round(time.perf_counter() - t0, 3)
                m.rss_peak_mb = round(mon.peak_mb, 1)
                m.rss_end_mb = round(rss_mb(), 1)
                self.steps.append(m)
                log.info(
                    "[%s] %s in %.2fs | rows in=%s out=%s | peak RSS %.0f MB | %d artifact(s)%s",
                    name, m.status, m.seconds, _fmt_int(m.rows_in), _fmt_int(m.rows_out),
                    m.rss_peak_mb, len(m.artifacts), f" | {len(m.warnings)} warning(s)" if m.warnings else "",
                )

    def to_frame(self, steps: list[StepMetrics] | None = None):
        import polars as pl

        rows = []
        for s in self.steps if steps is None else steps:
            d = asdict(s)
            d["artifacts"] = "; ".join(d["artifacts"])
            d["warnings"] = "; ".join(d["warnings"])
            d["run_id"] = self.run_id
            rows.append(d)
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()

    def save(self, directory: Path) -> Path | None:
        import polars as pl

        df = self.to_frame(self.steps[self._saved:])  # append only steps not yet persisted
        if df.is_empty():
            return None
        self._saved = len(self.steps)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "pipeline_run_log.parquet"
        if target.exists():
            df = pl.concat([pl.read_parquet(target), df], how="diagonal_relaxed")
        atomic_write_parquet(df, target)
        atomic_write_csv(df, target.with_suffix(".csv"))
        return target


def _fmt_int(v: int | None) -> str:
    return "n/a" if v is None else f"{v:,}"


# --------------------------------------------------------------------------- atomic IO
@contextlib.contextmanager
def atomic_path(target: Path | str) -> Iterator[Path]:
    """Yield a temp path next to ``target``; move it into place only on success.

    ``os.replace`` is atomic on the same filesystem (POSIX and Windows) for files.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            if tmp.is_dir():
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def atomic_directory(target: Path | str) -> Iterator[Path]:
    """Build a directory in a sibling temp location and swap it in on success.

    Directory swaps cannot be fully atomic on Windows; the old directory is renamed aside
    first, so a failure leaves either the old or the new version intact — never a partial one.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.mkdir()
    old = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.old")
    try:
        yield tmp
        if target.exists():
            os.replace(target, old)
        os.replace(tmp, target)
        shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        if old.exists() and not target.exists():
            os.replace(old, target)
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def atomic_write_text(text: str, target: Path | str) -> Path:
    with atomic_path(target) as tmp:
        tmp.write_text(text, encoding="utf-8")
    return Path(target)


def atomic_write_json(obj: Any, target: Path | str) -> Path:
    return atomic_write_text(json.dumps(obj, indent=2, default=str, sort_keys=True), target)


def atomic_write_parquet(df, target: Path | str, **kwargs: Any) -> Path:
    kwargs.setdefault("compression", "zstd")
    with atomic_path(target) as tmp:
        df.write_parquet(tmp, **kwargs)
    return Path(target)


def atomic_write_csv(df, target: Path | str) -> Path:
    """CSV export for human inspection; nested/list columns are flattened to strings."""
    import polars as pl

    flat = df.with_columns(
        [
            pl.col(c).cast(pl.List(pl.Utf8)).list.join(" | ").alias(c)
            for c, dt in df.schema.items()
            if isinstance(dt, pl.List)
        ]
        + [pl.col(c).struct.json_encode().alias(c) for c, dt in df.schema.items() if isinstance(dt, pl.Struct)]
    )
    with atomic_path(target) as tmp:
        flat.write_csv(tmp)
    return Path(target)


def read_json(path: Path | str, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- hashing
def sha256_file(path: Path | str, chunk_bytes: int = 8 * 1024**2) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


def file_fingerprint(path: Path | str, with_sha256: bool = False) -> dict[str, Any]:
    """Cheap change detector (size + mtime); add a SHA-256 for provenance-grade manifests."""
    st = Path(path).stat()
    fp: dict[str, Any] = {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if with_sha256:
        fp["sha256"] = sha256_file(path)
    return fp


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def sql_path(path: Path | str) -> str:
    """Path literal safe for DuckDB SQL strings (forward slashes, quotes escaped)."""
    return Path(path).as_posix().replace("'", "''")


def parquet_glob(directory: Path) -> str:
    return sql_path(directory / "**" / "*.parquet")
