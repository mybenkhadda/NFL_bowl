"""Write a portable data bundle (run on the research machine that holds the nflbdb warehouse and 12.0 / 12F outputs).

    # in a local notebook, AFTER 12F Part 3 has built the sequence store (same objects as the notebook):
    from nflbowl.data.export import export_research_bundle
    export_research_bundle(store, D, "bundles/m12f_v1")          # then upload the folder to Drive / Kaggle

Nothing here recomputes a sequence: the store files are copied byte for byte, the fold manifest and (optionally) the
12C target / officials / Milestone 06 TCN representations are copied as Parquet.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts.store import atomic_json, atomic_write, sha256_file
from ..utils.logging import get_logger
from ..utils.reproducibility import utc_now
from .loading import BUNDLE_FILE, BUNDLE_VERSION, STORE_DIR
from .validation import CHANNELS, M1_COLS, TRANSFER_TARGET, validate_arrays, validate_athletes, validate_folds, validate_meta

log = get_logger(__name__)


def _finish(out: Path, source: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != BUNDLE_FILE and not p.name.startswith("."):
            files[p.relative_to(out).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    info = {"format_version": BUNDLE_VERSION, "created_at": utc_now(), "source": source, "channels": list(CHANNELS), "files": files, **(extra or {})}
    atomic_json(info, out / BUNDLE_FILE, overwrite=True)
    log.info("bundle written: %s (%d files, %.1f MB)", out, len(files), sum(v["bytes"] for v in files.values()) / 2 ** 20)
    return out / BUNDLE_FILE


def _save_npy(path: Path, arr: np.ndarray) -> None:
    """``np.save`` through a closed file handle (a path argument would get '.npy' appended to the temporary name)."""
    with open(path, "wb") as f:
        np.save(f, arr)


def write_bundle(out_dir: Path | str, X: np.ndarray, M: np.ndarray, meta: Any, folds: Any, athletes: Any = None, tcn: Any = None,
                 source: dict[str, Any] | None = None, overwrite: bool = False) -> Path:
    """Bundle from in-memory arrays (tests, other data sources). ``meta`` row i must describe ``X[i]``."""
    out = Path(out_dir)
    if (out / BUNDLE_FILE).is_file() and not overwrite:
        raise FileExistsError(f"{out} already holds a bundle (pass overwrite=True to replace it)")
    validate_meta(meta)
    validate_arrays(X, M, meta.height)
    validate_folds(folds)
    sd = out / STORE_DIR
    sd.mkdir(parents=True, exist_ok=True)
    atomic_write(sd / "X.npy", lambda p: _save_npy(p, np.ascontiguousarray(X, np.float32)))
    atomic_write(sd / "M.npy", lambda p: _save_npy(p, np.ascontiguousarray(M, np.bool_)))
    atomic_write(sd / "meta.parquet", lambda p: meta.write_parquet(p))
    atomic_json({"key": "in_memory", "n": int(X.shape[0]), "L": int(X.shape[1]), "channels": list(CHANNELS)}, sd / "COMPLETE.json")
    atomic_write(out / "folds.parquet", lambda p: folds.write_parquet(p))
    if athletes is not None:
        validate_athletes(athletes)
        atomic_write(out / "athletes.parquet", lambda p: athletes.write_parquet(p))
    if tcn is not None:
        atomic_write(out / "tcn_representations.parquet", lambda p: tcn.write_parquet(p))
    return _finish(out, source or {"kind": "arrays"})


def export_research_bundle(store: Any, D: Any, out_dir: Path | str, include_transfer: bool = True, include_tcn: bool = True,
                           overwrite: bool = False) -> Path:
    """Copy the 12F sequence store + 12.0 fold manifest (+ 12C target, officials, cohort link flag, 06B TCN) into a bundle.

    ``store``: ``nflbdb.research.advanced_movement.artifacts.ResearchStore`` of the run mode to export (full!).
    ``D``: ``nflbdb.research.advanced_movement.m12.foundation.M12Data`` built on that store.
    """
    import polars as pl
    from nflbdb.research.advanced_movement.m11 import representation as rep
    from nflbdb.research.advanced_movement.m12 import foundation as fd

    out = Path(out_dir)
    if (out / BUNDLE_FILE).is_file() and not overwrite:
        raise FileExistsError(f"{out} already holds a bundle (pass overwrite=True to replace it)")
    key = store.read_json("m12f_sequence_store", fd.NB["12F"])["key"]
    st = rep.SequenceStore(store.sequences_dir, key)
    if not (st.dir / "COMPLETE.json").is_file():
        raise FileNotFoundError(f"sequence store {st.dir} is incomplete: run 12F Part 3 first")
    sd = out / STORE_DIR
    sd.mkdir(parents=True, exist_ok=True)
    for name in ("X.npy", "M.npy", "meta.parquet", "COMPLETE.json"):
        log.info("copying %s", name)
        atomic_write(sd / name, lambda p, _n=name: shutil.copyfile(st.dir / _n, p))
    meta = pl.read_parquet(sd / "meta.parquet")
    validate_meta(meta)
    X, M = np.load(sd / "X.npy", mmap_mode="r"), np.load(sd / "M.npy", mmap_mode="r")
    validate_arrays(X, M, meta.height)
    del X, M
    folds = D.m12("m12_fold_manifest").select("nfl_id", "fold")
    validate_folds(folds)
    atomic_write(out / "folds.parquet", lambda p: folds.write_parquet(p))
    extra: dict[str, Any] = {"store_key": key, "run_mode": getattr(store.controls, "run_mode", None)}
    if include_transfer:
        try:
            off = D.officials()
            reg = D.m12("m12_cohort_registry").select("nfl_id", "linked")
            ath = off.select([c for c in ("nfl_id", "position_group", *M1_COLS) if c in off.columns]).join(reg, on="nfl_id", how="left")
            if store.has("m12c_targets"):
                tg = store.read_frame("m12c_targets", fd.NB["12C"]).select("nfl_id", TRANSFER_TARGET)
                ath = ath.join(tg, on="nfl_id", how="left")
                extra["transfer_target"] = TRANSFER_TARGET
            else:
                log.warning("12C targets not found: athletes.parquet is written without %s (transfer ladder unavailable)", TRANSFER_TARGET)
            ath = ath.with_columns(pl.col("linked").fill_null(False))
            atomic_write(out / "athletes.parquet", lambda p: ath.write_parquet(p))
        except Exception as e:                                       # noqa: BLE001 — optional part of the bundle
            log.warning("athletes.parquet not written (%s: %s)", type(e).__name__, e)
    if include_tcn:
        try:
            tcn = store.transfer.require_named("transfer_attempt_representations_outer", "06b")
            atomic_write(out / "tcn_representations.parquet", lambda p: tcn.write_parquet(p))
        except Exception as e:                                       # noqa: BLE001
            log.warning("TCN-B representations not exported (%s)", type(e).__name__)
    return _finish(out, {"kind": "nflbdb_research_store", "sequence_store": str(st.dir), "store_info": json.loads((st.dir / "COMPLETE.json").read_text())},
                   extra)


__all__ = ["export_research_bundle", "write_bundle"]
