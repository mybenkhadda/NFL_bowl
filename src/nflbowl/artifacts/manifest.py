"""Artifact + job manifest with dependency hashing (cache invalidation).

Every artifact is registered under ``(kind, key)`` with a ``deps`` mapping (bundle fingerprint, science-config hash,
checkpoint signature, evaluation options …). ``deps_hash = hash(deps)`` is part of the artifact id and of its file name.
``find(kind, key, deps)`` only returns an artifact whose deps hash matches the CURRENT inputs, so an embedding computed
from another bundle, model or configuration is never reused by accident; it simply is not found and is recomputed into a
new file (the old one stays on disk and in the manifest).

Jobs (one per training run ``model × tag``) record status ``running`` / ``finished`` with their arguments, so
``Experiment.resume()`` can restart every unfinished job after a Colab disconnect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import stable_hash
from ..utils.reproducibility import utc_now
from .store import append_jsonl, atomic_json, sha256_file


def deps_hash(deps: dict[str, Any]) -> str:
    return stable_hash(deps, 12)


class Manifest:
    def __init__(self, root: Path | str, experiment_id: str):
        self.root = Path(root)
        self.file = self.root / "manifests" / "manifest.json"
        self.events = self.root / "manifests" / "events.jsonl"
        self.experiment_id = experiment_id
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if self.file.is_file():
            return json.loads(self.file.read_text(encoding="utf-8"))
        return {"experiment_id": self.experiment_id, "created_at": utc_now(), "artifacts": {}, "jobs": {}}

    def reload(self) -> "Manifest":
        self._data = self._load()
        return self

    def _save(self) -> None:
        self._data["updated_at"] = utc_now()
        atomic_json(self._data, self.file, overwrite=True)

    # ------------------------------------------------------------------ artifacts
    @staticmethod
    def artifact_id(kind: str, key: str, deps: dict[str, Any]) -> str:
        return f"{kind}:{key}:{deps_hash(deps)}"

    def register(self, kind: str, key: str, path: Path | str, deps: dict[str, Any], extra: dict[str, Any] | None = None,
                 checksum: bool = True) -> dict[str, Any]:
        p = Path(path)
        aid = self.artifact_id(kind, key, deps)
        rec = {"id": aid, "kind": kind, "key": key, "path": self._rel(p), "deps": deps, "deps_hash": deps_hash(deps), "created_at": utc_now(),
               "bytes": p.stat().st_size if p.is_file() else None, "sha256": sha256_file(p) if (checksum and p.is_file()) else None, **(extra or {})}
        self._data["artifacts"][aid] = rec
        self._save()
        append_jsonl({"event": "artifact", "id": aid, "at": rec["created_at"]}, self.events)
        return rec

    def find(self, kind: str, key: str, deps: dict[str, Any] | None = None, verify: bool = False) -> dict[str, Any] | None:
        """Latest artifact of ``(kind, key)`` whose deps match ``deps`` (any deps when None) and whose file exists."""
        want = deps_hash(deps) if deps is not None else None
        hits = [r for r in self._data["artifacts"].values() if r["kind"] == kind and r["key"] == key and (want is None or r["deps_hash"] == want)]
        for r in sorted(hits, key=lambda r: r["created_at"], reverse=True):
            p = self.resolve(r)
            if not p.exists():
                continue
            if verify and r.get("sha256") and p.is_file() and sha256_file(p) != r["sha256"]:
                continue
            return r
        return None

    def resolve(self, rec: dict[str, Any]) -> Path:
        p = Path(rec["path"])
        return p if p.is_absolute() else self.root / p

    def artifacts(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self._data["artifacts"].values() if kind is None or r["kind"] == kind]

    def _rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(p)

    # ------------------------------------------------------------------ jobs
    def set_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        rec = {**self._data["jobs"].get(job_id, {"id": job_id, "created_at": utc_now()}), **fields, "updated_at": utc_now()}
        self._data["jobs"][job_id] = rec
        self._save()
        append_jsonl({"event": "job", "id": job_id, "status": rec.get("status"), "at": rec["updated_at"]}, self.events)
        return rec

    def jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        return [j for j in self._data["jobs"].values() if status is None or j.get("status") == status]

    def job(self, job_id: str) -> dict[str, Any] | None:
        return self._data["jobs"].get(job_id)

    # ------------------------------------------------------------------ tables
    def table(self) -> Any:
        import polars as pl

        rows = [{k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in self.artifacts()]
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame({"id": [], "kind": [], "key": [], "path": []},
                                                                                       schema={"id": pl.String, "kind": pl.String, "key": pl.String, "path": pl.String})


__all__ = ["Manifest", "deps_hash"]
