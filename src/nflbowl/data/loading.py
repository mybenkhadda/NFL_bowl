"""Portable data bundle: what a GPU session needs, without the warehouse.

    <data_dir>/
        bundle.json                       format version, source, sha256 + size of every file, store key, channels, scale
        sequence_store/X.npy              (N, L, 6) float32 — the 12F unified sequence store, byte-identical copy
        sequence_store/M.npy              (N, L) bool
        sequence_store/meta.parquet       idx, seq_key, domain, nfl_id, kind, season, n_frames, position_group
        sequence_store/COMPLETE.json      nflbdb SequenceStore completion record
        folds.parquet                     nfl_id, fold — the 12.0 fold manifest (m12_fold_manifest)
        athletes.parquet       optional   nfl_id, position_group, linked, M1 officials, nfl.q90_shared (12C target)
        tcn_representations.parquet  opt.  Milestone 06 fold-fitted TCN-B representations (F1 baseline)

The bundle is written on the research machine by ``nflbowl.data.export.export_research_bundle`` (or
``write_bundle`` from arrays) and copied to Drive / Kaggle datasets. ``open_bundle`` optionally stages it to fast local
disk (Colab ``/content``) before memory-mapping, because random batch reads from a Drive-mounted file are very slow.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import stable_hash
from ..utils.logging import get_logger
from .validation import SchemaError, validate_arrays, validate_athletes, validate_folds, validate_meta

log = get_logger(__name__)
BUNDLE_VERSION = 1
BUNDLE_FILE = "bundle.json"
STORE_DIR = "sequence_store"
OPTIONAL = ("athletes.parquet", "tcn_representations.parquet")


@dataclass
class SequenceBundle:
    root: Path
    X: np.ndarray                       # memory-mapped (N, L, 6) float32
    M: np.ndarray                       # memory-mapped (N, L) bool
    meta: Any                           # polars DataFrame, row i ↔ X[i]
    folds: Any                          # polars DataFrame nfl_id, fold
    info: dict[str, Any]
    athletes: Any = None
    tcn: Any = None
    staged_from: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Content fingerprint: sha256 of every file as recorded at export (verified sizes; ``verify=True`` re-hashes)."""
        return bundle_fingerprint(self.info)

    @property
    def n_sequences(self) -> int:
        return int(self.X.shape[0])

    def summary(self) -> dict[str, Any]:
        dom = self.meta["domain"].to_numpy()
        return {"n_sequences": self.n_sequences, "n_combine": int((dom == "combine").sum()), "n_nfl": int((dom == "nfl").sum()),
                "n_athletes": int(self.meta["nfl_id"].n_unique()), "frames": int(self.X.shape[1]), "channels": int(self.X.shape[2]),
                "has_athletes": self.athletes is not None, "has_tcn": self.tcn is not None, "fingerprint": self.fingerprint,
                "root": str(self.root), "staged_from": str(self.staged_from) if self.staged_from else None}


def bundle_fingerprint(info: dict[str, Any]) -> str:
    files = info.get("files", {})
    return stable_hash({k: v.get("sha256") for k, v in sorted(files.items())}, 16)


def read_info(data_dir: Path) -> dict[str, Any]:
    p = Path(data_dir) / BUNDLE_FILE
    if not p.is_file():
        raw = Path(data_dir) / "COMPLETE.json"
        hint = (" It looks like a raw nflbdb SequenceStore directory: build a bundle with "
                "`nflbowl.data.export.export_research_bundle(store, D, out_dir)` on the research machine.") if raw.is_file() else ""
        raise FileNotFoundError(f"{p} not found: data_dir must be an nflbowl data bundle.{hint} See docs/nflbowl/INSTALL.md § Data bundle.")
    info = json.loads(p.read_text(encoding="utf-8"))
    if int(info.get("format_version", 0)) != BUNDLE_VERSION:
        raise SchemaError(f"bundle format_version {info.get('format_version')} is not supported (expected {BUNDLE_VERSION})")
    return info


def _stage(src: Path, stage_dir: Path, info: dict[str, Any]) -> Path:
    """Copy the bundle to ``stage_dir/<fingerprint>`` unless an identical copy (same sizes, completed marker) exists."""
    dst = Path(stage_dir) / f"bundle_{bundle_fingerprint(info)}"
    done = dst / ".staged.json"
    if done.is_file():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for rel, meta in info["files"].items():
        s, d = src / rel, dst / rel
        if d.is_file() and d.stat().st_size == meta.get("bytes"):
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(f".{d.name}.staging")
        log.info("staging %s (%.1f MB) → %s", rel, (meta.get("bytes") or 0) / 2 ** 20, dst)
        shutil.copyfile(s, tmp)
        tmp.replace(d)
    shutil.copyfile(src / BUNDLE_FILE, dst / BUNDLE_FILE)
    done.write_text(json.dumps({"source": str(src)}), encoding="utf-8")
    return dst


def _check_files(root: Path, info: dict[str, Any], verify: bool) -> None:
    from ..artifacts.store import sha256_file

    for rel, meta in info["files"].items():
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(f"bundle file {rel} is missing under {root}")
        if meta.get("bytes") is not None and p.stat().st_size != meta["bytes"]:
            raise SchemaError(f"bundle file {rel} has {p.stat().st_size} bytes; bundle.json says {meta['bytes']} (incomplete copy?)")
        small = p.suffix in (".parquet", ".json")
        if (verify or small) and meta.get("sha256") and sha256_file(p) != meta["sha256"]:
            raise SchemaError(f"bundle file {rel} fails its sha256 check (corrupted or modified)")


def open_bundle(data_dir: Path | str, stage_dir: Path | str | None = None, verify: bool = False, outer_folds: int | None = None) -> SequenceBundle:
    """Open (and optionally stage) a bundle, memory-map the arrays and validate every schema."""
    import polars as pl

    src = Path(data_dir)
    info = read_info(src)
    root = _stage(src, Path(stage_dir), info) if stage_dir else src
    _check_files(root, info, verify)
    sd = root / STORE_DIR
    X = np.load(sd / "X.npy", mmap_mode="r")
    M = np.load(sd / "M.npy", mmap_mode="r")
    meta = pl.read_parquet(sd / "meta.parquet")
    validate_meta(meta)
    validate_arrays(X, M, meta.height)
    folds = pl.read_parquet(root / "folds.parquet")
    validate_folds(folds, outer_folds)
    athletes = pl.read_parquet(root / "athletes.parquet") if (root / "athletes.parquet").is_file() else None
    if athletes is not None:
        validate_athletes(athletes)
    tcn = pl.read_parquet(root / "tcn_representations.parquet") if (root / "tcn_representations.parquet").is_file() else None
    return SequenceBundle(root, X, M, meta, folds, info, athletes, tcn, staged_from=src if stage_dir else None)


__all__ = ["BUNDLE_FILE", "BUNDLE_VERSION", "STORE_DIR", "SequenceBundle", "bundle_fingerprint", "open_bundle", "read_info"]
