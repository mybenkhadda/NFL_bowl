"""Immutable, configuration-hashed checkpoint shards (brief §16).

Why a new mechanism. The 07C / 08B.3 simulations rewrote ONE checkpoint file in place; on Windows a reader (antivirus,
indexer, a second kernel) can hold the file open, so ``os.replace`` onto it fails with ``PermissionError`` (WinError 5 /
32). Renaming the checkpoint to another fixed path only moves the collision. Here nothing is ever overwritten:

    <root>/<name>/<mode>_<config hash>/
        config.json                      written once; a later run with another configuration gets ANOTHER directory
        shard_<key>.<ext>                one immutable file per unit of work (scenario × replicate, epoch, fold …)
        shard_<key>.<ext>.meta.json      sidecar written AFTER the data file: its presence commits the shard;
                                         holds sha256, size, rows, key, config hash, time
        COMPLETE.json                    written only by ``mark_complete`` when every expected key is present AND verified

* Temporary files carry a random suffix in the SAME directory (no fixed shared temporary name, same volume for the move).
* Publication uses ``os.replace`` with bounded retries on ``PermissionError``; if the target appeared meanwhile, the
  existing shard is kept (same key, same configuration = same content contract) and the temporary file is removed.
* A shard whose sidecar is missing or whose hash does not match is IGNORED (reported by ``verify``) and recomputed.
* Resume = skip keys that verify. A partial set is never reported as complete: ``status()`` says ``partial``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import polars as pl

from ....data.utils import utc_now
from .config import config_hash

RETRIES = 12
BACKOFF_S = 0.25


class CheckpointMismatch(RuntimeError):
    """The directory holds a different configuration than the one requested."""


class IncompleteRun(RuntimeError):
    """``mark_complete`` was called while expected shards are missing or fail verification."""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _publish(tmp: Path, final: Path) -> bool:
    """Move ``tmp`` onto ``final`` unless ``final`` exists. True when this call published it."""
    for attempt in range(RETRIES):
        if final.exists():
            tmp.unlink(missing_ok=True)
            return False
        try:
            os.replace(tmp, final)
            return True
        except PermissionError:
            time.sleep(BACKOFF_S * (attempt + 1))
    if final.exists():
        tmp.unlink(missing_ok=True)
        return False
    raise PermissionError(f"could not publish {final.name} after {RETRIES} attempts (file locked by another process); the temporary "
                          f"file {tmp.name} was kept for inspection — nothing was overwritten")


def atomic_json(obj: dict[str, Any], final: Path, overwrite: bool = False) -> Path:
    """Write JSON through a uniquely named temporary file. ``overwrite`` only for files that are DESIGNED to be replaced
    (never for shards); it still uses a unique temporary name and retries."""
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    if overwrite:
        for attempt in range(RETRIES):
            try:
                os.replace(tmp, final)
                return final
            except PermissionError:
                time.sleep(BACKOFF_S * (attempt + 1))
        raise PermissionError(f"could not replace {final}")
    _publish(tmp, final)
    return final


def key_text(key: Any) -> str:
    parts = key if isinstance(key, (tuple, list)) else (key,)
    s = "__".join(str(p) for p in parts)
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", s)[:180]


class ShardCheckpoint:
    """One configuration's immutable shards. ``config`` must hold everything that changes a shard's content."""

    def __init__(self, root: Path, name: str, config: dict[str, Any], mode: str):
        self.name, self.mode, self.config = name, mode, config
        self.hash = config_hash({"name": name, "mode": mode, "config": config})
        self.dir = Path(root) / name / f"{mode}_{self.hash}"
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg_file = self.dir / "config.json"
        if cfg_file.is_file():
            have = json.loads(cfg_file.read_text(encoding="utf-8"))
            if have.get("config_hash") != self.hash:
                raise CheckpointMismatch(f"{self.dir} holds configuration {have.get('config_hash')}, not {self.hash}")
        else:
            atomic_json({"name": name, "mode": mode, "config_hash": self.hash, "config": config, "created_at": utc_now()}, cfg_file)

    # ------------------------------------------------------------------ paths
    def _data(self, key: Any, ext: str) -> Path:
        return self.dir / f"shard_{key_text(key)}.{ext}"

    def _meta(self, p: Path) -> Path:
        return p.with_name(p.name + ".meta.json")

    def _find(self, key: Any) -> Path | None:
        for ext in ("parquet", "json", "pt", "npz"):
            p = self._data(key, ext)
            if p.is_file() and self._meta(p).is_file():
                return p
        return None

    # ------------------------------------------------------------------ write
    def save(self, key: Any, payload: Any, writer: Callable[[Any, Path], None] | None = None, ext: str | None = None) -> Path:
        """Persist one unit of work. DataFrame → parquet, dict → json, otherwise ``writer(payload, tmp_path)`` with ``ext``.
        Never overwrites: an existing verified shard of the same key is returned unchanged."""
        have = self._find(key)
        if have is not None and self.verify_one(have)[0]:
            return have
        if isinstance(payload, pl.DataFrame):
            ext, rows = "parquet", payload.height
            write = lambda tmp: payload.write_parquet(tmp)
        elif isinstance(payload, dict) and writer is None:
            ext, rows = "json", None
            write = lambda tmp: tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        else:
            if writer is None or ext is None:
                raise TypeError("a non-table, non-dict payload needs `writer` and `ext`")
            rows = None
            write = lambda tmp: writer(payload, tmp)
        final = self._data(key, ext)
        if final.exists() and not self._meta(final).exists():            # orphan of an interrupted run: never trusted
            orphan = final.with_name(f"{final.name}.orphan-{uuid.uuid4().hex[:8]}")
            try:
                os.replace(final, orphan)
            except PermissionError:
                pass
        tmp = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
        write(tmp)
        _publish(tmp, final)
        atomic_json({"key": key_text(key), "sha256": sha256(final), "bytes": final.stat().st_size,
                     "rows": rows, "config_hash": self.hash, "written_at": utc_now()}, self._meta(final))
        return final

    # ------------------------------------------------------------------ read / verify
    def verify_one(self, p: Path) -> tuple[bool, str]:
        m = self._meta(p)
        if not m.is_file():
            return False, "no sidecar (interrupted write)"
        meta = json.loads(m.read_text(encoding="utf-8"))
        if meta.get("config_hash") != self.hash:
            return False, "configuration hash differs"
        if sha256(p) != meta.get("sha256"):
            return False, "content hash differs"
        return True, "ok"

    def has(self, key: Any) -> bool:
        p = self._find(key)
        return p is not None and self.verify_one(p)[0]

    def load(self, key: Any, reader: Callable[[Path], Any] | None = None) -> Any:
        p = self._find(key)
        if p is None:
            raise KeyError(f"no shard {key_text(key)} in {self.dir}")
        ok, why = self.verify_one(p)
        if not ok:
            raise IOError(f"shard {p.name} failed verification: {why}")
        if reader is not None:
            return reader(p)
        if p.suffix == ".parquet":
            return pl.read_parquet(p)
        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        raise TypeError(f"{p.suffix} shards need a reader")

    def keys(self) -> list[str]:
        out = []
        for m in sorted(self.dir.glob("shard_*.meta.json")):
            data = m.with_name(m.name[: -len(".meta.json")])
            if data.is_file() and self.verify_one(data)[0]:
                out.append(json.loads(m.read_text(encoding="utf-8"))["key"])
        return out

    def verify(self) -> pl.DataFrame:
        rows = []
        for p in sorted(self.dir.glob("shard_*")):
            if p.name.endswith(".meta.json") or ".orphan-" in p.name:
                continue
            ok, why = self.verify_one(p)
            rows.append({"shard": p.name, "ok": ok, "reason": why, "bytes": p.stat().st_size})
        return pl.DataFrame(rows, schema={"shard": pl.String, "ok": pl.Boolean, "reason": pl.String, "bytes": pl.Int64})

    def collect(self, keys: Iterable[Any] | None = None) -> pl.DataFrame:
        want = None if keys is None else {key_text(k) for k in keys}
        parts = [self.load(k) for k in self.keys() if (want is None or k in want)]
        parts = [p for p in parts if isinstance(p, pl.DataFrame) and p.height]
        return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()

    # ------------------------------------------------------------------ completion gate
    def status(self, expected: Sequence[Any]) -> dict[str, Any]:
        have = set(self.keys())
        want = [key_text(k) for k in expected]
        missing = [k for k in want if k not in have]
        return {"directory": str(self.dir), "config_hash": self.hash, "n_expected": len(want), "n_done": len(want) - len(missing),
                "missing_head": missing[:5], "state": "complete" if not missing and (self.dir / "COMPLETE.json").is_file()
                else ("ready_to_mark" if not missing else "partial")}

    def mark_complete(self, expected: Sequence[Any]) -> Path:
        st = self.status(expected)
        if st["n_done"] != st["n_expected"]:
            raise IncompleteRun(f"{st['n_expected'] - st['n_done']} of {st['n_expected']} expected shards missing or unverified "
                                f"(e.g. {st['missing_head']}): the run is PARTIAL and is not published as complete")
        return atomic_json({"config_hash": self.hash, "n_shards": st["n_expected"], "completed_at": utc_now()}, self.dir / "COMPLETE.json")

    def is_complete(self) -> bool:
        return (self.dir / "COMPLETE.json").is_file()


# ------------------------------------------------------------------------------ model checkpoints (11F)
class ModelCheckpoint(ShardCheckpoint):
    """Epoch checkpoints as immutable shards ``shard_epoch=0007.pt``; ``latest`` = highest verified epoch."""

    def save_epoch(self, epoch: int, state: dict[str, Any]) -> Path:
        import torch

        return self.save(("epoch", f"{epoch:04d}"), state, writer=lambda obj, p: torch.save(obj, p), ext="pt")

    def latest(self) -> tuple[int, dict[str, Any]] | None:
        import torch

        eps = sorted(int(k.split("=")[-1].split("__")[-1]) for k in self.keys() if k.startswith("epoch"))
        if not eps:
            return None
        e = eps[-1]
        return e, self.load(("epoch", f"{e:04d}"), reader=lambda p: torch.load(p, map_location="cpu", weights_only=False))


__all__ = ["CheckpointMismatch", "IncompleteRun", "ModelCheckpoint", "ShardCheckpoint", "atomic_json", "key_text", "sha256"]
