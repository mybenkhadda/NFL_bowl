"""Build a validated, portable M12F data bundle from the research files already on this machine — nothing recomputed.

    python -m nflbowl.data.builder --repo-root . --output bundles/m12f_v1                 # build (+ validate)
    python -m nflbowl.data.builder --repo-root . --discover-only                         # what exists, what is missing
    python -m nflbowl.data.builder --repo-root . --output bundles/m12f_v1 --dry-run      # checks + plan, writes nothing
    python -m nflbowl.data.builder --validate bundles/m12f_v1 --verify                   # re-hash and summarise a bundle
    python -m nflbowl.data.builder --zip-existing bundles/m12f_v1                        # m12f_v1.zip for upload

    from nflbowl.data.builder import build_bundle_from_repo
    result = build_bundle_from_repo(repo_root=".", output_dir="bundles/m12f_v1")
    print(result.text())

What it does (``docs/nflbowl/BUNDLE_BUILD_GUIDE.md``):

1. ``discovery.discover_bundle_sources``: the FULL 12F sequence store (explicit choice when there are several) and the
   files of the same research area / run mode (12.0 fold manifest + foundation manifest, cohort registry, 11.0
   officials, 12C targets + freeze record, Milestone 06 TCN-B representations).
2. Compatibility checks (no array is loaded in RAM: ``X`` / ``M`` are memory-mapped and scanned in chunks):
   shapes / dtypes / channels, ``meta`` schema and row order, ``meta`` sha256 against ``COMPLETE.json``, mask padding
   versus ``n_frames``, fold schema and range, fold signature against the 12.0 foundation manifest, configuration
   hash of the fold manifest and of the 12C freeze, run mode of every input, coverage of folds and athletes.
   ``verify=True`` additionally scans ``X`` for non-finite values, non-zero padding and a non-binary heading flag.
3. The bundle is assembled in a hidden sibling directory (``.<name>.building``): store files are COPIED byte for byte
   (sha256 computed while copying), ``folds.parquet`` / ``athletes.parquet`` are written with the same construction
   as ``export.export_research_bundle``; ``bundle.json`` comes from ``export._finish`` (the existing manifest writer:
   size + sha256 of every file actually present). Copied files must hash identically to their sources.
4. The finished directory is renamed into place (an existing bundle is never deleted: with ``overwrite=True`` it is
   moved to ``<name>.replaced-<timestamp>``), opened with ``open_bundle`` and summarised.
5. Reports: ``<output>.reports/`` (build_report.json, checks.csv, plan.csv, discovery.json/.csv, BUILD_REPORT.md) —
   outside the bundle, so they never enter its manifest or its ZIP.

Athlete ids, fold assignments, sequence order, channels and values are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..artifacts.store import atomic_json, atomic_text, atomic_write, sha256_file
from ..config import ScienceConfig, stable_hash
from ..utils.logging import get_logger
from ..utils.reproducibility import code_version, utc_now
from .discovery import BundleSources, discover_bundle_sources
from .loading import BUNDLE_FILE, STORE_DIR, _check_files, bundle_fingerprint, open_bundle, read_info
from .validation import CHANNELS, TRANSFER_TARGET, SchemaError, validate_arrays, validate_athletes, validate_folds, validate_meta

log = get_logger(__name__)
BUILDER_VERSION = 1
MARKER = ".nflbowl_building.json"
STORE_COPY = ("X.npy", "M.npy", "meta.parquet", "COMPLETE.json")
TCN_COLUMNS = ("family", "fold", "nfl_id", "dim", "value")           # se.tcn_by_fold reads these
FOUNDATION_KEYS = ("config_hash", "cohort_signature", "fold_signature", "grid_signature", "n_linked", "run_mode", "written_at")
OUTER_FOLDS = ScienceConfig.__dataclass_fields__["outer_folds"].default
ROWS = 16_384                                                          # rows per chunk when scanning X / M
COPY_CHUNK = 8 << 20


class BundleBuildError(RuntimeError):
    """The bundle cannot be built; the message says what to do."""


class MissingSourcesError(BundleBuildError):
    """A required research artifact is absent (the message names the research step that writes it)."""


class IncompatibleSourcesError(BundleBuildError):
    """The inputs come from different research runs, or fail a schema / integrity check."""


@dataclass
class Check:
    name: str
    status: str                         # pass · warn · fail · skip
    detail: str


@dataclass
class PlannedFile:
    dest: str                           # path inside the bundle
    action: str                         # copy (byte for byte) · derive (selected columns / join of small tables)
    sources: list[str]                  # repo-relative source paths
    bytes: int | None                   # source bytes (copy) or None
    required: bool
    note: str = ""


@dataclass
class BuildResult:
    status: str                         # built · up_to_date · dry_run · discovered · failed
    output_dir: Path | None
    sources: BundleSources
    plan: list[PlannedFile] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    bundle_json: Path | None = None
    fingerprint: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    report_dir: Path | None = None
    replaced: Path | None = None
    zip_path: Path | None = None
    error: str | None = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in ("built", "up_to_date", "dry_run", "discovered") and not any(c.status == "fail" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "ok": self.ok, "output_dir": str(self.output_dir) if self.output_dir else None,
                "bundle_json": str(self.bundle_json) if self.bundle_json else None, "fingerprint": self.fingerprint,
                "summary": self.summary, "error": self.error, "elapsed_s": round(self.elapsed_s, 2),
                "replaced": str(self.replaced) if self.replaced else None, "zip_path": str(self.zip_path) if self.zip_path else None,
                "report_dir": str(self.report_dir) if self.report_dir else None,
                "checks": [asdict(c) for c in self.checks], "plan": [asdict(p) for p in self.plan], "discovery": self.sources.to_dict(),
                "builder_version": BUILDER_VERSION, "written_at": utc_now()}

    def text(self) -> str:
        lines = [f"status: {self.status}" + (f" — {self.error}" if self.error else "")]
        if self.output_dir:
            lines.append(f"bundle: {self.output_dir}")
        if self.plan:
            lines.append("plan:")
            for p in self.plan:
                size = f"{p.bytes / 2 ** 20:9.1f} MB" if p.bytes is not None else " " * 12
                lines.append(f"  {p.action:<6} {p.dest:<30} {size}  ← {', '.join(p.sources)}{('  (' + p.note + ')') if p.note else ''}")
        if self.checks:
            lines.append("checks:")
            for c in self.checks:
                lines.append(f"  [{c.status.upper():<4}] {c.name:<28} {c.detail}")
        if self.summary:
            s = self.summary
            lines.append(f"summary: {s.get('n_sequences', 0):,} sequences ({s.get('n_combine', 0):,} Combine, {s.get('n_nfl', 0):,} NFL), "
                         f"{s.get('n_athletes', 0):,} athletes, {s.get('frames')} frames × {s.get('channels')} channels, "
                         f"{(s.get('total_bytes') or 0) / 2 ** 20:.1f} MB, fingerprint {s.get('fingerprint')}, {s.get('integrity')}")
        if self.replaced:
            lines.append(f"previous bundle moved to: {self.replaced}")
        if self.zip_path:
            lines.append(f"zip: {self.zip_path}")
        if self.report_dir:
            lines.append(f"reports: {self.report_dir}")
        return "\n".join(lines)


# ============================================================================== small helpers
class _Progress:
    """Byte progress on stderr (every ~5 % or 2 s); silent when disabled."""

    def __init__(self, label: str, total: int, enabled: bool = True):
        self.label, self.total, self.enabled = label, max(int(total), 1), enabled
        self.done, self._t, self._pct = 0, 0.0, -100.0

    def update(self, n: int) -> None:
        self.done += n
        if not self.enabled:
            return
        pct = 100.0 * self.done / self.total
        now = time.monotonic()
        if pct - self._pct >= 5.0 or now - self._t >= 2.0 or self.done >= self.total:
            self._pct, self._t = pct, now
            sys.stderr.write(f"\r  {self.label}: {min(pct, 100.0):5.1f}%  ({self.done / 2 ** 20:,.1f} / {self.total / 2 ** 20:,.1f} MB)")
            sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _copy_hashed(src: Path, dst: Path, label: str, progress: bool) -> str:
    """Copy ``src`` → ``dst`` in chunks; returns the sha256 of the bytes written (= the source bytes)."""
    h = hashlib.sha256()
    bar = _Progress(f"copy {label}", src.stat().st_size, progress)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for b in iter(lambda: fi.read(COPY_CHUNK), b""):
            h.update(b)
            fo.write(b)
            bar.update(len(b))
        fo.flush()
        os.fsync(fo.fileno())
    bar.close()
    return h.hexdigest()


def _sha_progress(path: Path, label: str, progress: bool) -> str:
    h = hashlib.sha256()
    bar = _Progress(f"sha256 {label}", path.stat().st_size, progress)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(COPY_CHUNK), b""):
            h.update(b)
            bar.update(len(b))
    bar.close()
    return h.hexdigest()


def _read_json(p: Path | None) -> dict[str, Any] | None:
    if p is None or not Path(p).is_file():
        return None
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rename(src: Path, dst: Path, retries: int = 12) -> None:
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:                            # Windows: indexer / antivirus / sync client holds a handle
            time.sleep(0.25 * (attempt + 1))
    raise PermissionError(f"could not rename {src} → {dst} (a program holds a file open); the complete bundle is at {src}")


def _read_folds(path: Path) -> Any:
    import polars as pl

    if path.suffix.lower() == ".csv":                     # the CSV view of the research table: fixed dtypes (hashes stay strings)
        want = {"nfl_id": pl.Int64, "fold": pl.Int64, "fold_source": pl.String, "fold_signature": pl.String, "config_hash": pl.String}
        cols = pl.read_csv(path, n_rows=0).columns
        return pl.read_csv(path, schema_overrides={c: t for c, t in want.items() if c in cols})
    return pl.read_parquet(path)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    cols: list[str] = []
    for r in rows:
        cols.extend(k for k in r if k not in cols)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols or ["empty"], extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    atomic_text(buf.getvalue(), path)


# ============================================================================== checks
class _Checks:
    def __init__(self) -> None:
        self.items: list[Check] = []

    def add(self, name: str, ok: bool, detail: str, warn: bool = False) -> bool:
        self.items.append(Check(name, "pass" if ok else ("warn" if warn else "fail"), detail))
        return ok

    def skip(self, name: str, detail: str) -> None:
        self.items.append(Check(name, "skip", detail))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.items if c.status == "fail"]


def _check_store(src: BundleSources, ck: _Checks, deep: bool, progress: bool) -> Any:
    """Store-level checks; returns ``meta`` (polars) or None when it cannot be read."""
    import polars as pl

    st = src.selected
    ck.add("store.complete", st.usable, f"{st.rel}: " + ("; ".join(st.problems) or f"COMPLETE.json present, key {st.key}"))
    ck.add("store.run_mode", st.run_mode == "full", f"run mode {st.run_mode!r}" + ("" if st.run_mode == "full" else " (research bundles use FULL)"),
           warn=True)                                      # a smoke store only gets here when explicitly allowed (discovery refuses it)
    ck.add("store.channels", st.channels is None or tuple(st.channels) == CHANNELS, f"{st.channels} vs {list(CHANNELS)}")
    ck.add("store.pointer", st.pointer_key in (None, st.key),
           f"12F pointer m12f_sequence_store.json names {st.pointer_key or '(no pointer)'}", warn=True)
    try:
        X = np.load(st.dir / "X.npy", mmap_mode="r")
        M = np.load(st.dir / "M.npy", mmap_mode="r")
        meta = pl.read_parquet(st.dir / "meta.parquet")
    except (OSError, ValueError) as e:
        ck.add("store.readable", False, f"{type(e).__name__}: {e}")
        return None
    try:
        validate_meta(meta)
        ck.add("store.meta_schema", True, f"{meta.height:,} rows; idx = 0..N-1; domains ⊆ combine/nfl; one position per athlete")
    except SchemaError as e:
        ck.add("store.meta_schema", False, str(e))
    try:
        validate_arrays(X, M, meta.height)
        ck.add("store.arrays", True, f"X {tuple(X.shape)} float32, M {tuple(M.shape)} bool, rows = meta rows")
    except SchemaError as e:
        ck.add("store.arrays", False, str(e))
    ck.add("store.count_vs_complete", st.n is None or int(X.shape[0]) == int(st.n), f"X rows {X.shape[0]:,}, COMPLETE.json n = {st.n}")
    if st.sha_meta:
        got = sha256_file(st.dir / "meta.parquet")
        ck.add("store.meta_sha256", got == st.sha_meta, "meta.parquet matches the sha256 recorded in COMPLETE.json" if got == st.sha_meta
               else f"meta.parquet sha256 {got[:12]}… ≠ COMPLETE.json {st.sha_meta[:12]}… (meta was modified after the store was written)")
    else:
        ck.skip("store.meta_sha256", "COMPLETE.json records no sha_meta")
    if "n_frames" in meta.columns and X.ndim == 3 and M.shape == X.shape[:2] and X.shape[0] == meta.height:
        N, L = int(X.shape[0]), int(X.shape[1])
        k = np.minimum(meta["n_frames"].fill_null(0).to_numpy().astype(np.int64), L)
        cols = np.arange(L)[None, :]
        bad_mask = empty = 0
        for s in range(0, N, ROWS):
            Mc = np.asarray(M[s: s + ROWS])
            pad = cols >= k[s: s + ROWS, None]
            bad_mask += int((Mc & pad).any(1).sum())
            empty += int((~Mc.any(1)).sum())
        ck.add("store.mask_padding", bad_mask == 0, f"{bad_mask:,} sequences mark frames beyond n_frames as valid" if bad_mask
               else "no valid-mask frame beyond n_frames (padding is masked)")
        ck.add("store.mask_nonempty", empty == 0, f"{empty:,} sequences have no valid frame" if empty else "every sequence has ≥ 1 valid frame",
               warn=True)
        if deep:
            nonfinite = pad_nonzero = hv_bad = 0
            bar = _Progress("scan X", N, progress)
            for s in range(0, N, ROWS):
                Xc = np.asarray(X[s: s + ROWS])
                pad = cols >= k[s: s + ROWS, None]
                nonfinite += int((~np.isfinite(Xc)).any((1, 2)).sum())
                pad_nonzero += int(((Xc != 0) & pad[..., None]).any((1, 2)).sum())
                hv = Xc[..., CHANNELS.index("heading_valid")]
                hv_bad += int(((hv != 0) & (hv != 1)).any(1).sum())
                bar.update(Xc.shape[0])
            bar.close()
            ck.add("store.values_finite", nonfinite == 0, f"{nonfinite:,} sequences hold NaN/inf" if nonfinite else "all values finite")
            ck.add("store.padding_zero", pad_nonzero == 0, f"{pad_nonzero:,} sequences have non-zero padding" if pad_nonzero
                   else "padding frames are exactly 0", warn=True)
            ck.add("store.heading_flag", hv_bad == 0, f"{hv_bad:,} sequences have a heading_valid value ∉ {{0, 1}}" if hv_bad
                   else "heading_valid ∈ {0, 1}", warn=True)
        else:
            ck.skip("store.values", "X values not scanned (pass verify=True / --verify)")
    else:
        ck.skip("store.mask_padding", "meta has no n_frames column, or the arrays failed the shape checks above")
    del X, M
    return meta


def _check_folds(src: BundleSources, ck: _Checks, meta: Any, outer_folds: int) -> tuple[Any, dict[str, Any] | None]:
    """Fold checks; returns (folds frame ``nfl_id, fold``, foundation manifest dict)."""
    import polars as pl

    st = src.selected
    fman = _read_json(src.path("foundation_manifest"))
    fp = src.path("fold_manifest")
    try:
        raw = _read_folds(fp)
    except Exception as e:                                   # noqa: BLE001 — reported as a failed check
        ck.add("folds.readable", False, f"{src.rel(fp)}: {type(e).__name__}: {e}")
        return None, fman
    try:
        folds = raw.select("nfl_id", "fold")
        validate_folds(folds, outer_folds)
        nn = folds["fold"].drop_nulls()
        ck.add("folds.schema", True, f"{folds.height:,} athletes, one row each; folds {sorted(set(nn.to_list()))} ⊆ [0, {outer_folds})")
    except Exception as e:                                   # noqa: BLE001 — SchemaError or a missing column
        ck.add("folds.schema", False, f"{type(e).__name__}: {e}")
        return None, fman
    ck.add("folds.ids_integer", folds["nfl_id"].dtype.is_integer() and not folds["nfl_id"].null_count(),
           f"nfl_id dtype {folds['nfl_id'].dtype}, {folds['nfl_id'].null_count()} nulls")
    stored = sorted({str(v) for v in raw["fold_signature"].drop_nulls().to_list()}) if "fold_signature" in raw.columns else []
    recomputed = stable_hash(folds.sort("nfl_id").select("nfl_id", "fold").rows(), 16)    # = m12.advanced_evaluation.fold_signature
    if stored:
        ck.add("folds.signature_unique", len(stored) == 1, f"fold_signature values in the manifest: {stored}")
        ck.add("folds.signature_recomputed", recomputed in stored, f"recomputed {recomputed} vs stored {stored}", warn=True)
    else:
        ck.skip("folds.signature_unique", "the fold manifest has no fold_signature column")
    if fman:
        want = fman.get("fold_signature")
        ck.add("folds.matches_foundation", want is None or want in stored or want == recomputed,
               f"foundation manifest fold_signature {want}; manifest {stored or [recomputed]}")
        if st.run_mode in ("full", "smoke"):
            ck.add("foundation.run_mode", fman.get("run_mode") in (None, st.run_mode), f"foundation run_mode {fman.get('run_mode')!r}, store {st.run_mode!r}")
        if "config_hash" in raw.columns:
            hashes = sorted({str(v) for v in raw["config_hash"].drop_nulls().to_list()})
            ck.add("folds.config_hash", not hashes or hashes == [str(fman.get("config_hash"))],
                   f"fold manifest config_hash {hashes}; foundation {fman.get('config_hash')}")
    else:
        ck.skip("folds.matches_foundation", "m12_foundation_manifest.json absent")
    if meta is not None:
        ids = set(meta["nfl_id"].unique().to_list())
        with_fold = set(folds.filter(pl.col("fold").is_not_null())["nfl_id"].to_list())
        ck.add("folds.coverage", True, f"{len(ids & with_fold):,} / {len(ids):,} sequence athletes have a manifest fold; "
               f"{len(ids - with_fold):,} use the 12F fallback nfl_id % {outer_folds}; {len(with_fold - ids):,} manifest athletes have no sequence")
    return folds, fman


def _check_athletes(src: BundleSources, ck: _Checks, meta: Any, fman: dict[str, Any] | None) -> tuple[Any, bool]:
    """Assemble ``athletes.parquet`` in memory (small tables) and check it; returns (frame or None, has target)."""
    import polars as pl

    from .export import athletes_table

    if not src.transfer_available():
        ck.skip("athletes", "officials / cohort registry absent: athletes.parquet not built (transfer ladder unavailable)")
        return None, False
    try:
        off = pl.read_parquet(src.path("officials"))
        reg = pl.read_parquet(src.path("cohort_registry"), columns=["nfl_id", "linked"])
        tp = src.path("transfer_targets")
        tg = pl.read_parquet(tp, columns=["nfl_id", TRANSFER_TARGET]) if tp is not None else None
        ath = athletes_table(off, reg, tg)
        missing = validate_athletes(ath)
    except Exception as e:                                   # noqa: BLE001 — reported as a failed check
        ck.add("athletes.build", False, f"{type(e).__name__}: {e}")
        return None, False
    ck.add("athletes.schema", not missing, f"{ath.height:,} athletes; missing columns {missing}" if missing else f"{ath.height:,} athletes, all columns",
           warn=True)
    ck.add("athletes.target", tg is not None, f"{TRANSFER_TARGET}: {ath[TRANSFER_TARGET].drop_nulls().len():,} athletes" if tg is not None
           else f"{TRANSFER_TARGET} absent (rerun 12C for the transfer ladder)", warn=True)
    freeze = _read_json(src.path("target_freeze"))
    if tg is not None and freeze and fman:
        ck.add("targets.freeze_config", freeze.get("config_hash") in (None, fman.get("config_hash")),
               f"12C freeze config_hash {freeze.get('config_hash')}; foundation {fman.get('config_hash')}")
    n_linked = int(ath["linked"].sum())
    reg_linked = int(reg["linked"].fill_null(False).sum())
    ck.add("athletes.linked", n_linked == reg_linked and (not fman or fman.get("n_linked") in (None, reg_linked)),
           f"linked athletes: athletes.parquet {n_linked}, cohort registry {reg_linked}, foundation {fman.get('n_linked') if fman else '—'}", warn=True)
    if meta is not None:
        mp = meta.select("nfl_id", "position_group").drop_nulls().unique()
        mism = mp.join(ath.select("nfl_id", pl.col("position_group").alias("_pa")), on="nfl_id", how="inner").filter(
            pl.col("_pa").is_not_null() & (pl.col("_pa") != pl.col("position_group")))
        ck.add("athletes.positions", mism.height == 0, f"{mism.height} athletes have another position_group in meta than in officials "
               f"(e.g. {mism['nfl_id'].head(3).to_list()})" if mism.height else "meta and officials agree on position_group", warn=True)
    return ath, tg is not None


def _check_tcn(src: BundleSources, ck: _Checks, outer_folds: int) -> bool:
    import polars as pl

    p = src.path("tcn_representations")
    if p is None:
        ck.skip("tcn", "Milestone 06 TCN-B representations absent (F1_TCN baseline unavailable)")
        return False
    try:
        names = pl.scan_parquet(p).collect_schema().names()
        miss = [c for c in TCN_COLUMNS if c not in names]
        if miss:
            ck.add("tcn.schema", False, f"missing columns {miss}; TCN representations are left out", warn=True)
            return False
        fo = pl.scan_parquet(p).select(pl.col("fold").unique()).collect()["fold"].drop_nulls().to_list()
        ok = all(0 <= int(f) < outer_folds for f in fo)
        ck.add("tcn.folds", ok, f"TCN folds {sorted(int(f) for f in fo)} ⊆ [0, {outer_folds})", warn=True)
        return ok
    except Exception as e:                                   # noqa: BLE001
        ck.add("tcn.readable", False, f"{type(e).__name__}: {e}; TCN representations are left out", warn=True)
        return False


# ============================================================================== plan / signature
def _plan(src: BundleSources, folds_path: Path, with_athletes: bool, with_tcn: bool) -> list[PlannedFile]:
    st = src.selected
    plan = [PlannedFile(f"{STORE_DIR}/{n}", "copy", [src.rel(st.dir / n)], st.files.get(n), True) for n in STORE_COPY]
    plan.append(PlannedFile("folds.parquet", "derive", [src.rel(folds_path)], None, True, "columns nfl_id, fold (12.0 fold manifest)"))
    if with_athletes:
        srcs = [src.rel(src.path(r)) for r in ("officials", "cohort_registry", "transfer_targets") if src.path(r) is not None]
        plan.append(PlannedFile("athletes.parquet", "derive", srcs, None, False, "officials + linked (+ 12C nfl.q90_shared)"))
    if with_tcn:
        p = src.path("tcn_representations")
        plan.append(PlannedFile("tcn_representations.parquet", "copy", [src.rel(p)], p.stat().st_size, False))
    return plan


def _input_paths(src: BundleSources, folds_path: Path, with_athletes: bool, with_tcn: bool) -> dict[str, Path]:
    st = src.selected
    paths = {f"store/{n}": st.dir / n for n in STORE_COPY}
    paths["fold_manifest"] = folds_path
    if with_athletes:
        for r in ("officials", "cohort_registry", "transfer_targets"):
            if src.path(r) is not None:
                paths[r] = src.path(r)
    if with_tcn:
        paths["tcn_representations"] = src.path("tcn_representations")
    return paths


def _signature(inputs: dict[str, dict[str, Any]], options: dict[str, Any]) -> str:
    """Build identity: sha256 of every input + the resolved options (an unchanged signature ⇒ an identical bundle)."""
    return stable_hash({"inputs": {k: v["sha256"] for k, v in sorted(inputs.items())}, "options": options, "builder": BUILDER_VERSION}, 16)


# ============================================================================== reports
def _write_reports(res: BuildResult, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(res.to_dict(), report_dir / "build_report.json")
    atomic_json(res.sources.to_dict(), report_dir / "discovery.json")
    _write_csv(res.sources.rows(), report_dir / "discovery.csv")
    _write_csv([asdict(c) for c in res.checks], report_dir / "checks.csv")
    _write_csv([{**asdict(p), "sources": "; ".join(p.sources)} for p in res.plan], report_dir / "plan.csv")
    md = ["# Bundle build report", "", f"- written: {utc_now()}", f"- status: **{res.status}**" + (f": {res.error}" if res.error else ""),
          f"- bundle: `{res.output_dir}`", f"- fingerprint: `{res.fingerprint}`", "", "## Discovery", "", "```", res.sources.summary(), "```", "",
          "## Checks", "", "| status | check | detail |", "|---|---|---|"]
    md += [f"| {c.status} | {c.name} | {c.detail.replace('|', '/')} |" for c in res.checks]
    md += ["", "## Plan", "", "| action | file | sources | note |", "|---|---|---|---|"]
    md += [f"| {p.action} | {p.dest} | {', '.join(p.sources)} | {p.note} |" for p in res.plan]
    if res.summary:
        md += ["", "## Bundle summary", "", "```json", json.dumps(res.summary, indent=2, default=str), "```"]
    atomic_text("\n".join(md) + "\n", report_dir / "BUILD_REPORT.md")
    return report_dir


# ============================================================================== build
def build_bundle_from_repo(repo_root: Path | str = ".", output_dir: Path | str = "bundles/m12f_v1", include_transfer: bool = True,
                           include_tcn: bool = True, reuse_existing: bool = True, overwrite: bool = False,
                           sequence_store: Path | str | None = None, fold_manifest: Path | str | None = None, verify: bool = False,
                           dry_run: bool = False, outer_folds: int = OUTER_FOLDS, allow_smoke: bool = False,
                           report_dir: Path | str | None = None, progress: bool = True, make_zip: bool = False) -> BuildResult:
    """Discover → check → copy → ``bundle.json`` → ``open_bundle`` validation → reports (see the module docstring).

    ``reuse_existing``: an existing bundle at ``output_dir`` built from byte-identical inputs with the same options is
    reported ``up_to_date`` (and re-validated) instead of being rebuilt. ``overwrite``: rebuild even if a bundle exists
    (the old directory is moved aside, never deleted). ``verify``: deep scan of ``X`` before the build and a full
    sha256 re-check of the result. ``dry_run``: checks + plan only, nothing written (reports only with ``report_dir``).
    Raises ``MissingSourcesError`` / ``IncompatibleSourcesError`` / ``FileExistsError`` with actionable messages; the
    report directory is written first, except in a dry run.
    """
    t0 = time.monotonic()
    repo = Path(repo_root).resolve()
    out = Path(output_dir)
    out = out if out.is_absolute() else repo / out
    rdir = Path(report_dir) if report_dir else (None if dry_run else out.with_name(out.name + ".reports"))
    src = discover_bundle_sources(repo, sequence_store=sequence_store, fold_manifest=fold_manifest, allow_smoke=allow_smoke)
    res = BuildResult("failed", out, src)

    def fail(exc: BundleBuildError | FileExistsError) -> BuildResult:
        res.status, res.error, res.elapsed_s = "failed", str(exc), time.monotonic() - t0
        if rdir is not None:
            res.report_dir = _write_reports(res, rdir)
        raise exc

    if not src.can_build:
        return fail(MissingSourcesError("cannot build the bundle:\n  - " + "\n  - ".join(src.missing) + "\n\n" + src.summary()))

    # ---- checks (reads meta / small tables; X and M stay memory-mapped)
    ck = _Checks()
    meta = _check_store(src, ck, deep=verify, progress=progress)
    folds, fman = _check_folds(src, ck, meta, outer_folds)
    with_athletes, has_target = False, False
    ath = None
    if include_transfer:
        ath, has_target = _check_athletes(src, ck, meta, fman)
        with_athletes = ath is not None
    with_tcn = _check_tcn(src, ck, outer_folds) if include_tcn else False
    res.checks = ck.items
    folds_path = src.path("fold_manifest")
    res.plan = _plan(src, folds_path, with_athletes, with_tcn)
    if ck.failed:
        names = "\n  - ".join(f"{c.name}: {c.detail}" for c in ck.failed)
        optional = all(c.name.startswith(("athletes.", "targets.")) for c in ck.failed)
        hint = ("\nOnly the optional transfer part failed: rebuild with --no-include-transfer to bundle the sequences and folds without it."
                if optional else "\nThe inputs come from different research runs or are damaged; rerun the research step named above.")
        return fail(IncompatibleSourcesError(f"{len(ck.failed)} check(s) failed (nothing was written to {out}):\n  - {names}{hint}"))
    if dry_run:
        res.status, res.elapsed_s = "dry_run", time.monotonic() - t0
        if rdir is not None:
            res.report_dir = _write_reports(res, rdir)
        return res

    options = {"include_transfer": with_athletes, "transfer_target": has_target, "include_tcn": with_tcn, "outer_folds": outer_folds,
               "run_mode": src.run_mode, "store_key": src.selected.key}
    in_paths = _input_paths(src, folds_path, with_athletes, with_tcn)

    # ---- existing output
    if out.exists() and (out / BUNDLE_FILE).is_file() and not overwrite:
        if not reuse_existing:
            return fail(FileExistsError(f"{out} already holds a bundle; pass overwrite=True (--overwrite) to rebuild it, or another output"))
        prev = read_info(out)
        inputs = {k: {"rel": src.rel(p), "bytes": p.stat().st_size, "sha256": _sha_progress(p, k, progress)} for k, p in in_paths.items()}
        sig = _signature(inputs, options)
        if prev.get("build_signature") != sig:
            why = "it was built by another writer (no build_signature)" if "build_signature" not in prev else "its inputs or options differ"
            return fail(FileExistsError(f"{out} already holds a bundle, and {why}. Pass overwrite=True (--overwrite) to replace it (the old "
                                        "one is moved aside), or choose another --output."))
        log.info("bundle at %s is up to date (signature %s); validating", out, sig)
        b = open_bundle(out, verify=verify, outer_folds=outer_folds)
        res.status, res.bundle_json, res.fingerprint, res.summary = "up_to_date", out / BUNDLE_FILE, b.fingerprint, b.summary()
    else:
        if out.exists() and any(out.iterdir()) and not (out / BUNDLE_FILE).is_file() and not overwrite:
            return fail(FileExistsError(f"{out} exists, is not empty and is not a bundle; refusing to touch it (choose another --output, "
                                        "or pass --overwrite to move it aside)"))
        res = _build(src, res, out, in_paths, folds, ath, options, outer_folds, verify, progress, fail)
    if make_zip:
        res.zip_path = zip_bundle(out, overwrite=True, progress=progress)
    res.elapsed_s = time.monotonic() - t0
    if rdir is not None:
        res.report_dir = _write_reports(res, rdir)
    return res


def _build(src: BundleSources, res: BuildResult, out: Path, in_paths: dict[str, Path], folds: Any, ath: Any, options: dict[str, Any],
           outer_folds: int, verify: bool, progress: bool, fail: Callable[[Any], BuildResult]) -> BuildResult:
    from .export import _finish

    st = src.selected
    staging = out.with_name(f".{out.name}.building")
    if staging.exists():
        if not (staging / MARKER).is_file():
            return fail(BundleBuildError(f"{staging} exists but was not created by the builder; remove it by hand"))
        shutil.rmtree(staging)                             # an interrupted build of ours: start clean
    (staging / STORE_DIR).mkdir(parents=True)
    atomic_json({"started_at": utc_now(), "pid": os.getpid(), "output": str(out)}, staging / MARKER)

    copied: dict[str, str] = {}                            # bundle path → sha256 of the source bytes
    inputs: dict[str, dict[str, Any]] = {}
    for n in STORE_COPY:
        holder: dict[str, str] = {}
        atomic_write(staging / STORE_DIR / n, lambda p, _n=n: holder.__setitem__("sha", _copy_hashed(st.dir / _n, p, _n, progress)))
        copied[f"{STORE_DIR}/{n}"] = holder["sha"]
        inputs[f"store/{n}"] = {"rel": src.rel(st.dir / n), "bytes": (st.dir / n).stat().st_size, "sha256": holder["sha"]}
    atomic_write(staging / "folds.parquet", lambda p: folds.write_parquet(p))
    if ath is not None and options["include_transfer"]:
        atomic_write(staging / "athletes.parquet", lambda p: ath.write_parquet(p))
    if options["include_tcn"]:
        holder = {}
        tp = in_paths["tcn_representations"]
        atomic_write(staging / "tcn_representations.parquet", lambda p: holder.__setitem__("sha", _copy_hashed(tp, p, tp.name, progress)))
        copied["tcn_representations.parquet"] = holder["sha"]
        inputs["tcn_representations"] = {"rel": src.rel(tp), "bytes": tp.stat().st_size, "sha256": holder["sha"]}
    for k, p in in_paths.items():                          # small derived-table inputs
        if k not in inputs:
            inputs[k] = {"rel": src.rel(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}

    fman = _read_json(src.path("foundation_manifest")) or {}
    freeze = _read_json(src.path("target_freeze")) or {}
    store_info = json.loads((st.dir / "COMPLETE.json").read_text(encoding="utf-8"))
    sig = _signature(inputs, options)
    source = {"kind": "nflbdb_research_files", "builder": "nflbowl.data.builder", "builder_version": BUILDER_VERSION,
              "run_mode": src.run_mode, "store_key": st.key, "sequence_store": st.rel, "store_info": store_info, "inputs": inputs,
              "options": options, "foundation": {k: fman.get(k) for k in FOUNDATION_KEYS if k in fman},
              "target_freeze": {k: freeze.get(k) for k in ("config_hash", "written_at", "run_mode") if k in freeze} or None,
              "repo": src.repo_root.name, "code_version": code_version()}
    extra: dict[str, Any] = {"store_key": st.key, "run_mode": src.run_mode, "build_signature": sig, "outer_folds": outer_folds}
    if options["transfer_target"]:
        extra["transfer_target"] = TRANSFER_TARGET
    info_path = _finish(staging, source, extra)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    diff = [rel for rel, sha in copied.items() if info["files"].get(rel, {}).get("sha256") != sha]
    if diff:
        return fail(BundleBuildError(f"copied files differ from their sources after writing: {diff} (disk error?); staging kept at {staging}"))
    res.checks.append(Check("bundle.byte_identical", "pass", f"{len(copied)} copied files hash identically to their sources"))
    (staging / MARKER).unlink()

    # ---- publish: rename the finished directory into place (never delete an existing bundle)
    if out.exists():
        if any(out.iterdir()):
            res.replaced = out.with_name(f"{out.name}.replaced-{time.strftime('%Y%m%dT%H%M%S')}")
            _rename(out, res.replaced)
            log.info("previous content of %s moved to %s", out, res.replaced)
        else:
            out.rmdir()
    out.parent.mkdir(parents=True, exist_ok=True)
    _rename(staging, out)

    b = open_bundle(out, verify=verify, outer_folds=outer_folds)
    res.checks.append(Check("bundle.open_bundle", "pass", "open_bundle(" + ("verify=True: every file re-hashed" if verify else
                                                                            "sizes + parquet/json sha256") + ") succeeded"))
    res.status, res.bundle_json, res.fingerprint, res.summary = "built", out / BUNDLE_FILE, b.fingerprint, b.summary()
    log.info("bundle built: %s (fingerprint %s)", out, b.fingerprint)
    return res


# ============================================================================== validation / zip / extract
def validate_bundle(bundle_dir: Path | str, verify: bool = True, outer_folds: int | None = OUTER_FOLDS) -> dict[str, Any]:
    """Open a bundle exactly as a GPU session will (``verify``: re-hash every file) and return its summary."""
    return open_bundle(bundle_dir, verify=verify, outer_folds=outer_folds).summary()


def zip_bundle(bundle_dir: Path | str, zip_path: Path | str | None = None, verify: bool = False, overwrite: bool = False,
               progress: bool = True) -> Path:
    """``<bundle>.zip`` holding ``<bundle name>/`` with ``bundle.json`` and exactly the files it lists (nothing else).

    The archive is written atomically and CRC-tested (``ZipFile.testzip``) before it is published."""
    d = Path(bundle_dir)
    info = read_info(d)
    if verify:
        open_bundle(d, verify=True)
    else:
        _check_files(d, info, False)
    zp = Path(zip_path) if zip_path else d.with_name(d.name + ".zip")
    if zp.exists() and not overwrite:
        raise FileExistsError(f"{zp} exists (pass overwrite=True)")
    names = [BUNDLE_FILE, *sorted(info["files"])]
    total = sum((d / n).stat().st_size for n in names)

    def writer(tmp: Path) -> None:
        bar = _Progress(f"zip {d.name}", total, progress)
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for n in names:
                z.write(d / n, arcname=f"{d.name}/{n}")
                bar.update((d / n).stat().st_size)
        bar.close()
        with zipfile.ZipFile(tmp) as z:
            bad = z.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"CRC check failed for {bad}")

    atomic_write(zp, writer, overwrite=True)
    log.info("zip written: %s (%.1f MB, fingerprint %s)", zp, zp.stat().st_size / 2 ** 20, bundle_fingerprint(info))
    return zp


def extract_bundle_zip(zip_path: Path | str, dest_dir: Path | str, verify: bool = False, overwrite: bool = False) -> Path:
    """Extract a bundle ZIP into ``dest_dir`` and return the bundle directory (``dest_dir/<bundle name>``).

    Safe: member paths may not escape ``dest_dir``; extraction goes to a temporary directory that is renamed into place
    only when complete. If an identical bundle (same fingerprint) is already there it is returned unchanged."""
    zp, dest = Path(zip_path), Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    root_abs = os.path.abspath(dest)
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.strip("/")]
        for n in names:
            target = os.path.abspath(os.path.join(root_abs, n))
            if os.path.commonpath([root_abs, target]) != root_abs:
                raise SchemaError(f"unsafe path in archive: {n}")
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) == 1 and f"{next(iter(tops))}/{BUNDLE_FILE}" in names:
            top: str | None = next(iter(tops))
            zinfo = json.loads(z.read(f"{top}/{BUNDLE_FILE}"))
        elif BUNDLE_FILE in names:
            top = None
            zinfo = json.loads(z.read(BUNDLE_FILE))
        else:
            raise SchemaError(f"{zp} is not a bundle archive: no bundle.json at its top level or in a single top folder")
        name = top or zp.stem
        final = dest / name
        if (final / BUNDLE_FILE).is_file():
            if bundle_fingerprint(read_info(final)) == bundle_fingerprint(zinfo):
                log.info("bundle %s already extracted (fingerprint %s)", final, bundle_fingerprint(zinfo))
                open_bundle(final, verify=verify)
                return final
            if not overwrite:
                raise FileExistsError(f"{final} holds another bundle (fingerprint {bundle_fingerprint(read_info(final))}); pass overwrite=True")
        tmp = dest / f".nflbowl_extract_{uuid.uuid4().hex}"
        try:
            z.extractall(tmp)
            src = tmp / top if top else tmp
            if final.exists():
                _rename(final, final.with_name(f"{final.name}.replaced-{time.strftime('%Y%m%dT%H%M%S')}"))
            _rename(src, final)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    open_bundle(final, verify=verify)
    return final


# ============================================================================== CLI
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m nflbowl.data.builder",
                                description="Build / validate / zip an nflbowl M12F data bundle from the local research artifacts.")
    p.add_argument("--repo-root", default=".", help="repository root (default: .)")
    p.add_argument("--output", default="bundles/m12f_v1", help="bundle directory to write (default: bundles/m12f_v1)")
    p.add_argument("--discover-only", action="store_true", help="list sources and missing prerequisites; write nothing")
    p.add_argument("--source-sequence-store", metavar="PATH", help="explicit 12F store directory (…/sequence_shards/<key>)")
    p.add_argument("--fold-manifest", metavar="PATH", help="explicit fold manifest (Parquet, or the reports/tables CSV view)")
    p.add_argument("--include-transfer", action=argparse.BooleanOptionalAction, default=True, help="write athletes.parquet (default: yes)")
    p.add_argument("--include-tcn", action=argparse.BooleanOptionalAction, default=True, help="copy the 06B TCN-B representations (default: yes)")
    p.add_argument("--verify", action="store_true", help="deep-scan X before building and re-hash every file after")
    p.add_argument("--overwrite", action="store_true", help="rebuild even if a bundle exists (the old one is moved aside)")
    p.add_argument("--no-reuse", action="store_true", help="do not accept an existing identical bundle as up to date")
    p.add_argument("--dry-run", action="store_true", help="run every check and print the plan; write nothing")
    p.add_argument("--allow-smoke", action="store_true", help="allow an explicitly given smoke-mode store (wiring tests only)")
    p.add_argument("--outer-folds", type=int, default=OUTER_FOLDS, help=f"number of outer folds (default: {OUTER_FOLDS})")
    p.add_argument("--report-dir", metavar="PATH", help="where to write reports (default: <output>.reports)")
    p.add_argument("--zip", action="store_true", help="also write <output>.zip after a successful build")
    p.add_argument("--validate", metavar="BUNDLE", help="only open and summarise an existing bundle (add --verify to re-hash)")
    p.add_argument("--zip-existing", metavar="BUNDLE", help="only write <BUNDLE>.zip from an existing bundle")
    p.add_argument("--no-progress", action="store_true", help="no progress lines on stderr")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    return p


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    progress = not a.no_progress
    try:
        if a.validate:
            s = validate_bundle(a.validate, verify=a.verify, outer_folds=a.outer_folds)
            print(json.dumps(s, indent=2, default=str))
            return 0
        if a.zip_existing:
            zp = zip_bundle(a.zip_existing, verify=a.verify, overwrite=True, progress=progress)
            print(json.dumps({"zip": str(zp), "bytes": zp.stat().st_size}) if a.json else f"zip written: {zp}")
            return 0
        if a.discover_only:
            src = discover_bundle_sources(a.repo_root, sequence_store=a.source_sequence_store, fold_manifest=a.fold_manifest,
                                          allow_smoke=a.allow_smoke)
            if a.report_dir:
                rd = Path(a.report_dir)
                rd.mkdir(parents=True, exist_ok=True)
                atomic_json(src.to_dict(), rd / "discovery.json")
                _write_csv(src.rows(), rd / "discovery.csv")
            print(json.dumps(src.to_dict(), indent=2, default=str) if a.json else src.summary())
            return 0 if src.can_build else 2
        res = build_bundle_from_repo(a.repo_root, a.output, include_transfer=a.include_transfer, include_tcn=a.include_tcn,
                                     reuse_existing=not a.no_reuse, overwrite=a.overwrite, sequence_store=a.source_sequence_store,
                                     fold_manifest=a.fold_manifest, verify=a.verify, dry_run=a.dry_run, outer_folds=a.outer_folds,
                                     allow_smoke=a.allow_smoke, report_dir=a.report_dir, progress=progress, make_zip=a.zip)
        print(json.dumps(res.to_dict(), indent=2, default=str) if a.json else res.text())
        return 0
    except MissingSourcesError as e:
        print(f"MISSING SOURCES\n{e}", file=sys.stderr)
        return 2
    except IncompatibleSourcesError as e:
        print(f"INCOMPATIBLE SOURCES\n{e}", file=sys.stderr)
        return 3
    except FileExistsError as e:
        print(f"OUTPUT EXISTS\n{e}", file=sys.stderr)
        return 4
    except (BundleBuildError, SchemaError, FileNotFoundError) as e:
        print(f"ERROR ({type(e).__name__})\n{e}", file=sys.stderr)
        return 1


__all__ = ["BuildResult", "BundleBuildError", "Check", "IncompatibleSourcesError", "MissingSourcesError", "PlannedFile", "build_bundle_from_repo",
           "extract_bundle_zip", "main", "validate_bundle", "zip_bundle"]


if __name__ == "__main__":
    raise SystemExit(main())
