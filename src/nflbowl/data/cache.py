"""Dependency-keyed Parquet cache on top of the experiment manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..artifacts.manifest import Manifest, deps_hash
from ..artifacts.store import atomic_write
from ..utils.logging import get_logger

log = get_logger(__name__)


def cached_frame(manifest: Manifest, kind: str, key: str, deps: dict[str, Any], directory: Path, compute: Callable[[], Any],
                 reuse: bool = True, extra: dict[str, Any] | None = None) -> tuple[Any, Path, bool]:
    """Return ``(frame, path, from_cache)``. The file name carries the deps hash, so a changed input never collides with
    (or overwrites) an older cached file."""
    import polars as pl

    if reuse:
        hit = manifest.find(kind, key, deps)
        if hit is not None:
            p = manifest.resolve(hit)
            log.info("reusing %s/%s (%s)", kind, key, p.name)
            return pl.read_parquet(p), p, True
    df = compute()
    p = Path(directory) / f"{key}__{deps_hash(deps)}.parquet"
    atomic_write(p, lambda t: df.write_parquet(t), overwrite=True)
    manifest.register(kind, key, p, deps, extra)
    return df, p, False


__all__ = ["cached_frame"]
