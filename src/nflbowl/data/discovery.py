"""Find the research artifacts a data bundle is built from, using filesystem metadata only.

    from nflbowl.data.discovery import discover_bundle_sources
    sources = discover_bundle_sources(repo_root=".")
    print(sources.summary())

Where the research code writes them (``nflbdb.research.advanced_movement.artifacts.ResearchStore``; smoke runs nest
everything under ``_sample/`` and are never mixed with full runs):

    <warehouse>/gold/advanced_movement/[_sample/]advanced_v1/
        sequence_shards/<key>/{X.npy, M.npy, meta.parquet, COMPLETE.json}    12F Part 3 sequence store
        m12f_sequence_store.json                                           12F pointer: key of the store it used
        m12_fold_manifest.parquet                                          12.0 athlete folds (+ fold_signature)
        m12_foundation_manifest.json                                       12.0 signatures (config, cohort, folds)
        m12_cohort_registry.parquet                                        12.0 cohort (``linked`` flag)
        m12c_targets.parquet, m12c_target_freeze.json                      12C transfer target + its freeze record
    <warehouse>/gold/advanced_movement/[_sample/]discovery_v1/m11_officials.parquet     11.0 official Combine results
    <warehouse>/gold/advanced_movement/[_sample/]transfer/transfer_attempt_representations_outer.parquet   06B TCN-B

``<warehouse>`` is ``$NFL_BDB_WAREHOUSE``, else ``warehouse_root`` of ``configs/phase01.toml``, else ``warehouse``.
Sequence stores outside that layout (``data/``, ``artifacts/``, ``.tmp/``, …) are found by a bounded directory walk.
Nothing here opens an array: ``.npy`` shapes come from the file header, store facts from ``COMPLETE.json``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .loading import BUNDLE_FILE, STORE_DIR
from .validation import CHANNELS

GOLD_NAMESPACE = ("gold", "advanced_movement")
AREA, DISCOVERY_AREA, TRANSFER_AREA, SMOKE = "advanced_v1", "discovery_v1", "transfer", "_sample"
STORE_FILES = ("X.npy", "M.npy", "meta.parquet", "COMPLETE.json")
SEARCH_DIRS = ("warehouse", "data", "artifacts", ".tmp", "notebooks", "outputs")
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache",
             "bronze", "silver", "catalog"}                   # raw warehouse layers never hold a sequence store
MAX_DEPTH, MAX_DIRS = 8, 50_000

# role → (file name, area, required, notebook that writes it)
AREA_FILES: dict[str, tuple[str, str, bool, str]] = {
    "fold_manifest": ("m12_fold_manifest.parquet", AREA, True, "12_00_advanced_movement_research_foundation.ipynb (12.0)"),
    "foundation_manifest": ("m12_foundation_manifest.json", AREA, False, "12_00_advanced_movement_research_foundation.ipynb (12.0)"),
    "store_pointer": ("m12f_sequence_store.json", AREA, False, "12f_physics_informed_shared_movement_encoder.ipynb (12F Part 3)"),
    "cohort_registry": ("m12_cohort_registry.parquet", AREA, False, "12_00_advanced_movement_research_foundation.ipynb (12.0)"),
    "transfer_targets": ("m12c_targets.parquet", AREA, False, "12c_distributional_nfl_movement_capacity.ipynb (12C)"),
    "target_freeze": ("m12c_target_freeze.json", AREA, False, "12c_distributional_nfl_movement_capacity.ipynb (12C)"),
    "officials": ("m11_officials.parquet", DISCOVERY_AREA, False, "Milestone 11 foundation notebook (11.0)"),
    "tcn_representations": ("transfer_attempt_representations_outer.parquet", TRANSFER_AREA, False, "Milestone 06 notebook 06b"),
}
TRANSFER_ROLES = ("officials", "cohort_registry")             # athletes.parquet needs both; the 12C target is added when present


@dataclass
class StoreCandidate:
    """One 12F / m11 sequence store directory (``<root>/<key>/``)."""

    key: str
    dir: Path
    rel: str
    run_mode: str                       # full · smoke · unknown
    complete: bool
    files: dict[str, int]               # present store file → bytes
    n: int | None = None
    L: int | None = None
    channels: list[str] | None = None
    x_shape: list[int] | None = None
    x_dtype: str | None = None
    m_shape: list[int] | None = None
    m_dtype: str | None = None
    sha_meta: str | None = None
    pointer_key: str | None = None      # key named by m12f_sequence_store.json of the same area
    problems: list[str] = field(default_factory=list)

    @property
    def area_dir(self) -> Path | None:
        return self.dir.parent.parent if self.dir.parent.name == "sequence_shards" else None

    @property
    def is_m12f(self) -> bool:
        """Lives in the Milestone 12 area (12F store); stores of other milestones are listed but never auto-selected."""
        return self.area_dir is not None and self.area_dir.name == AREA

    @property
    def pointer_selected(self) -> bool:
        return self.pointer_key == self.key

    @property
    def usable(self) -> bool:
        return self.complete and not self.problems

    @property
    def bytes(self) -> int:
        return int(sum(self.files.values()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(dir=str(self.dir), area_dir=str(self.area_dir) if self.area_dir else None, pointer_selected=self.pointer_selected,
                 usable=self.usable, is_m12f=self.is_m12f, bytes=self.bytes)
        return d


@dataclass
class SourceRecord:
    role: str
    path: Path
    rel: str
    required: bool
    available: bool
    run_mode: str
    bytes: int | None
    writer: str                         # research step that produces it
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "path": str(self.path)}


@dataclass
class BundleSources:
    """Result of :func:`discover_bundle_sources`."""

    repo_root: Path
    gold_roots: list[Path]
    stores: list[StoreCandidate]
    selected: StoreCandidate | None
    selection: str                      # how the store was chosen, or why none was
    records: dict[str, SourceRecord]
    missing: list[str]                  # actionable messages for REQUIRED inputs
    warnings: list[str]
    existing_bundles: list[Path]

    @property
    def can_build(self) -> bool:
        return self.selected is not None and not self.missing

    @property
    def run_mode(self) -> str | None:
        return self.selected.run_mode if self.selected else None

    def path(self, role: str) -> Path | None:
        r = self.records.get(role)
        return r.path if r is not None and r.available else None

    def transfer_available(self) -> bool:
        return all(self.path(r) is not None for r in TRANSFER_ROLES)

    def rel(self, p: Path | str) -> str:
        return _rel(Path(p), self.repo_root)

    # ------------------------------------------------------------------ reports
    def rows(self) -> list[dict[str, Any]]:
        """One row per discovered source (flat; CSV-ready)."""
        out = []
        for s in self.stores:
            out.append({"role": "sequence_store", "rel": s.rel, "path": str(s.dir), "required": True, "available": s.usable,
                        "run_mode": s.run_mode, "bytes": s.bytes, "store_key": s.key, "selected": s is self.selected,
                        "note": "; ".join(s.problems) or ("12F pointer names this store" if s.pointer_selected else "")})
        for r in self.records.values():
            out.append({"role": r.role, "rel": r.rel, "path": str(r.path), "required": r.required, "available": r.available,
                        "run_mode": r.run_mode, "bytes": r.bytes, "store_key": self.selected.key if self.selected else None,
                        "selected": r.available, "note": r.note or ("" if r.available else f"written by {r.writer}")})
        for b in self.existing_bundles:
            out.append({"role": "existing_bundle", "rel": self.rel(b), "path": str(b), "required": False, "available": True,
                        "run_mode": None, "bytes": None, "store_key": None, "selected": False, "note": "already a bundle"})
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"repo_root": str(self.repo_root), "gold_roots": [str(g) for g in self.gold_roots], "can_build": self.can_build,
                "run_mode": self.run_mode, "selection": self.selection, "selected_store": self.selected.to_dict() if self.selected else None,
                "stores": [s.to_dict() for s in self.stores], "records": {k: v.to_dict() for k, v in self.records.items()},
                "missing": self.missing, "warnings": self.warnings, "existing_bundles": [str(b) for b in self.existing_bundles]}

    def summary(self) -> str:
        lines = [f"nflbowl bundle sources under {self.repo_root}",
                 f"  warehouse gold roots : {', '.join(self.rel(g) for g in self.gold_roots) or '(none found)'}",
                 f"  sequence stores      : {len(self.stores)} found"]
        for s in self.stores:
            mark = "*" if s is self.selected else " "
            shape = f"N={s.n:,} L={s.L}" if s.n is not None and s.L is not None else "shape ?"
            state = "complete" if s.complete else "INCOMPLETE"
            extra = f" — {'; '.join(s.problems)}" if s.problems else ""
            ptr = " [12F pointer]" if s.pointer_selected else ""
            lines.append(f"   {mark} {s.key:<18} {s.run_mode:<7} {state:<10} {shape:<18} {s.bytes / 2 ** 20:8.1f} MB  {s.rel}{ptr}{extra}")
        lines.append(f"  selection            : {self.selection}")
        if self.records:
            lines.append("  inputs of the selected store:")
            for r in self.records.values():
                flag = "ok     " if r.available else ("MISSING" if r.required else "absent ")
                size = f"{r.bytes / 2 ** 20:8.2f} MB" if r.bytes is not None else " " * 11
                kind = "required" if r.required else "optional"
                lines.append(f"     {flag} {r.role:<20} {kind:<8} {size}  {r.rel}{('  (' + r.note + ')') if r.note else ''}")
        if self.existing_bundles:
            lines.append("  existing bundles     : " + ", ".join(self.rel(b) for b in self.existing_bundles))
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        for m in self.missing:
            lines.append(f"  MISSING: {m}")
        lines.append(f"  bundle can be built  : {'YES' if self.can_build else 'NO'}"
                     + ("" if self.can_build or not self.selected else " (see MISSING above)"))
        if self.selected and self.can_build:
            lines.append(f"  transfer data        : {'available' if self.transfer_available() else 'not available (athletes.parquet will be skipped)'}"
                         + ("" if self.path("transfer_targets") else "; no 12C target"))
            lines.append(f"  TCN representations  : {'available' if self.path('tcn_representations') else 'not available'}")
        return "\n".join(lines)


# ============================================================================== helpers
def _rel(p: Path, root: Path) -> str:
    try:
        return Path(os.path.abspath(p)).relative_to(Path(os.path.abspath(root))).as_posix()
    except ValueError:
        return str(p)


def _size(p: Path) -> int | None:
    try:
        return p.stat().st_size if p.is_file() else None
    except OSError:
        return None


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype, bool]:
    """Shape, dtype, Fortran order of a ``.npy`` file from its header (the array itself is never read)."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:                                              # (3, 0): utf-8 field names; same layout as 2.0
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)   # noqa: SLF001
    return tuple(int(s) for s in shape), np.dtype(dtype), bool(fortran)


def run_mode_of(path: Path) -> str:
    parts = Path(path).parts
    if SMOKE in parts:
        return "smoke"
    if AREA in parts or DISCOVERY_AREA in parts or TRANSFER_AREA in parts:
        return "full"
    return "unknown"


def warehouse_roots(repo_root: Path) -> list[Path]:
    """Candidate warehouse roots: ``$NFL_BDB_WAREHOUSE``, ``configs/phase01.toml`` ``warehouse_root``, ``warehouse``."""
    out: list[Path] = []
    env = os.environ.get("NFL_BDB_WAREHOUSE")
    if env:
        out.append(Path(env) if Path(env).is_absolute() else repo_root / env)
    toml = repo_root / "configs" / "phase01.toml"
    if toml.is_file():
        try:
            m = re.search(r'^\s*warehouse_root\s*=\s*"([^"]+)"', toml.read_text(encoding="utf-8"), re.MULTILINE)
        except OSError:
            m = None
        if m:
            p = Path(m.group(1))
            out.append(p if p.is_absolute() else repo_root / p)
    out.append(repo_root / "warehouse")
    seen, uniq = set(), []
    for p in out:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def inspect_store(store_dir: Path, repo_root: Path) -> StoreCandidate:
    """Facts of one sequence-store directory from ``COMPLETE.json`` and the ``.npy`` headers."""
    d = Path(store_dir)
    files = {n: s for n in STORE_FILES if (s := _size(d / n)) is not None}
    complete = "COMPLETE.json" in files
    c = StoreCandidate(key=d.name, dir=d, rel=_rel(d, repo_root), run_mode=run_mode_of(d), complete=complete, files=files)
    missing = [n for n in STORE_FILES if n not in files]
    if missing:
        c.problems.append(f"missing {', '.join(missing)}" + ("" if complete else " (build interrupted: COMPLETE.json absent)"))
    info = _read_json(d / "COMPLETE.json") if complete else None
    if complete and info is None:
        c.problems.append("COMPLETE.json is unreadable")
    if info:
        c.n, c.L = info.get("n"), info.get("L")
        c.channels, c.sha_meta = info.get("channels"), info.get("sha_meta")
        if info.get("key") not in (None, d.name):
            c.problems.append(f"COMPLETE.json key {info.get('key')} does not match directory {d.name}")
        if c.channels is not None and tuple(c.channels) != CHANNELS:
            c.problems.append(f"channels {c.channels} differ from {list(CHANNELS)}")
    for name, attr in (("X.npy", "x"), ("M.npy", "m")):
        if name in files:
            try:
                shape, dtype, fortran = npy_header(d / name)
                setattr(c, f"{attr}_shape", list(shape))
                setattr(c, f"{attr}_dtype", str(dtype))
                if fortran:
                    c.problems.append(f"{name} is Fortran-ordered")
            except (OSError, ValueError) as e:
                c.problems.append(f"{name} header unreadable ({e})")
    if c.x_shape is not None and c.n is not None and c.L is not None and c.x_shape != [c.n, c.L, len(CHANNELS)]:
        c.problems.append(f"X.npy shape {c.x_shape} != COMPLETE.json (n={c.n}, L={c.L}, C={len(CHANNELS)})")
    if c.x_shape is not None and c.m_shape is not None and c.m_shape != c.x_shape[:2]:
        c.problems.append(f"M.npy shape {c.m_shape} != X.npy {c.x_shape[:2]}")
    if c.x_dtype not in (None, "float32"):
        c.problems.append(f"X.npy dtype {c.x_dtype} (expected float32)")
    if c.m_dtype not in (None, "bool"):
        c.problems.append(f"M.npy dtype {c.m_dtype} (expected bool)")
    area = c.area_dir
    if area is not None:
        ptr = _read_json(area / AREA_FILES["store_pointer"][0])
        c.pointer_key = (ptr or {}).get("key")
    return c


def _walk_stores(roots: Iterable[Path], repo_root: Path, warnings: list[str]) -> tuple[list[Path], list[Path]]:
    """Directories holding ``COMPLETE.json`` or ``X.npy`` (sequence stores), and bundle directories, below ``roots``."""
    stores, bundles, seen_dirs = [], [], 0
    for root in roots:
        if not root.is_dir():
            continue
        base = len(root.parts)
        for cur, dirs, names in os.walk(root):
            seen_dirs += 1
            if seen_dirs > MAX_DIRS:
                warnings.append(f"directory walk stopped after {MAX_DIRS:,} directories; pass the store explicitly if it was not found")
                return stores, bundles
            p = Path(cur)
            if BUNDLE_FILE in names:
                bundles.append(p)
                dirs[:] = []                               # a bundle's own sequence_store is not a research source
                continue
            if ("COMPLETE.json" in names or "X.npy" in names) and ("X.npy" in names or "meta.parquet" in names) and p.name != STORE_DIR:
                stores.append(p)
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".") and not d.endswith(".reports")
                       and len(p.parts) - base < MAX_DEPTH]
    return stores, bundles


def _record(role: str, area_root: Path, repo_root: Path, override: Path | None = None) -> SourceRecord:
    name, area, required, writer = AREA_FILES[role]
    p = Path(override) if override is not None else area_root / area / name
    size = _size(p)
    return SourceRecord(role=role, path=p, rel=_rel(p, repo_root), required=required, available=size is not None,
                        run_mode=run_mode_of(p), bytes=size, writer=writer)


# ============================================================================== discovery
def discover_bundle_sources(repo_root: Path | str = ".", sequence_store: Path | str | None = None, fold_manifest: Path | str | None = None,
                            allow_smoke: bool = False, search_dirs: Iterable[str] = SEARCH_DIRS) -> BundleSources:
    """Locate the 12F sequence store and the files that must travel with it.

    ``sequence_store``: explicit store directory (``…/sequence_shards/<key>``); required when several FULL stores exist.
    ``fold_manifest``: explicit fold manifest (Parquet, or the CSV view ``reports/tables/…/m12_fold_manifest.csv``).
    ``allow_smoke``: permit an explicitly given smoke-mode store (never chosen automatically).
    """
    repo = Path(repo_root).resolve()
    warnings: list[str] = []
    missing: list[str] = []
    gold_roots = [w / Path(*GOLD_NAMESPACE) for w in warehouse_roots(repo) if (w / Path(*GOLD_NAMESPACE)).is_dir()]

    # ---- sequence stores: the declared layout first, then a bounded walk of the usual data directories
    dirs: list[Path] = []
    for g in gold_roots:
        for prefix in ("", SMOKE):
            shards = g / prefix / AREA / "sequence_shards"
            if shards.is_dir():
                dirs.extend(sorted(d for d in shards.iterdir() if d.is_dir() and not d.name.startswith(".")))
    walked, bundles = _walk_stores([repo / s for s in search_dirs], repo, warnings)
    dirs.extend(walked)
    uniq: dict[str, Path] = {}
    for d in dirs:
        uniq.setdefault(os.path.normcase(os.path.abspath(d)), d)
    stores = [inspect_store(d, repo) for d in uniq.values()]
    stores.sort(key=lambda s: (s.run_mode != "full", not s.complete, s.rel))

    # ---- selection
    selected: StoreCandidate | None = None
    if sequence_store is not None:
        sp = Path(sequence_store)
        sp = sp if sp.is_absolute() else (repo / sp)
        if sp.name == "COMPLETE.json" or sp.suffix in (".npy", ".parquet"):
            sp = sp.parent
        match = [s for s in stores if os.path.normcase(os.path.abspath(s.dir)) == os.path.normcase(os.path.abspath(sp))]
        cand = match[0] if match else (inspect_store(sp, repo) if sp.is_dir() else None)
        if cand is None:
            selection = f"explicit store {sp} does not exist"
            missing.append(f"--source-sequence-store {sp}: directory not found")
        else:
            if not match:
                stores.append(cand)
            if not cand.usable:
                selection = f"explicit store {cand.rel} is not usable"
                missing.append(f"sequence store {cand.rel}: {'; '.join(cand.problems) or 'incomplete'}")
            elif cand.run_mode == "smoke" and not allow_smoke:
                selection = f"explicit store {cand.rel} is a SMOKE store"
                missing.append(f"{cand.rel} is a smoke-mode store; a research bundle must be built from the FULL store "
                               "(pass allow_smoke=True / --allow-smoke only for wiring tests)")
            else:
                selected, selection = cand, f"explicit: {cand.rel}"
    else:
        full = [s for s in stores if s.run_mode == "full" and s.usable and s.is_m12f]
        pointed = [s for s in full if s.pointer_selected]
        if len(full) == 1 and (full[0].pointer_key in (None, full[0].key)):
            selected = full[0]
            selection = f"only complete FULL store{' (matches the 12F pointer)' if full[0].pointer_selected else ''}: {full[0].rel}"
        elif len(full) > 1 or (len(full) == 1 and not pointed):
            hint = f"; the 12F pointer m12f_sequence_store.json names {pointed[0].key}" if pointed else \
                   (f"; the 12F pointer names {full[0].pointer_key}, which is not among them" if full and full[0].pointer_key else "")
            selection = f"ambiguous: {len(full)} complete FULL store(s){hint}"
            missing.append(f"{len(full)} candidate FULL sequence stores ({', '.join(s.key for s in full)}){hint}. Choose one explicitly with "
                           "--source-sequence-store <dir> (Python: sequence_store=...). Nothing is chosen silently.")
        else:
            incomplete = [s for s in stores if s.run_mode == "full" and s.is_m12f and not s.usable]
            smoke = [s for s in stores if s.run_mode == "smoke" and s.usable]
            other = [s for s in stores if s.usable and not s.is_m12f and s.run_mode != "smoke"]
            selection = "no complete FULL 12F sequence store"
            msg = ("no complete FULL 12F sequence store was found. Rerun notebook 12f_physics_informed_shared_movement_encoder.ipynb "
                   "Part 3 with RUN_MODE='full' (it writes warehouse/gold/advanced_movement/advanced_v1/sequence_shards/<key>/).")
            if incomplete:
                msg += " Unusable FULL store(s): " + ", ".join(f"{s.rel} ({'; '.join(s.problems)})" for s in incomplete) + "."
            if smoke:
                msg += f" Smoke store(s) exist ({', '.join(s.rel for s in smoke)}) but are never used for a research bundle."
            if other:
                msg += (f" Complete store(s) outside the 12F area exist ({', '.join(s.rel for s in other)}); select one explicitly with "
                        "--source-sequence-store only if it is the 12F store moved elsewhere.")
            missing.append(msg)

    # ---- inputs that travel with the selected store (same area, same run mode)
    records: dict[str, SourceRecord] = {}
    if selected is not None:
        area = selected.area_dir
        if area is None:
            warnings.append(f"{selected.rel} is outside the research layout (…/advanced_v1/sequence_shards/<key>); companion files are "
                            "looked up in the default warehouse only")
            prefix_root = next((g / (SMOKE if selected.run_mode == "smoke" else "") for g in gold_roots), repo / "warehouse" / Path(*GOLD_NAMESPACE))
        else:
            prefix_root = area.parent
        for role in AREA_FILES:
            override = Path(fold_manifest) if role == "fold_manifest" and fold_manifest is not None else None
            if override is not None and not override.is_absolute():
                override = repo / override
            records[role] = _record(role, prefix_root, repo, override)
        fm = records["fold_manifest"]
        if fm.available and fold_manifest is not None:
            fm.note = "explicit"
            if fm.path.suffix.lower() == ".csv":
                fm.note = "explicit CSV view: integer columns nfl_id, fold are read with fixed dtypes"
        if not fm.available:
            csv_view = repo / "reports" / "tables" / "advanced_movement" / (SMOKE if selected.run_mode == "smoke" else "") / AREA / "m12_fold_manifest.csv"
            hint = (f" A CSV view exists at {_rel(csv_view, repo)}: pass it explicitly with --fold-manifest to use it"
                    if csv_view.is_file() else "")
            missing.append(f"fold manifest {fm.rel} is missing: rerun notebook {fm.writer} with RUN_MODE='{selected.run_mode}'.{hint}")
        for r in records.values():
            if r.available and r.run_mode not in (selected.run_mode, "unknown") and selected.run_mode != "unknown":
                missing.append(f"{r.role} {r.rel} is a {r.run_mode} artifact but the store is {selected.run_mode}: runs are never mixed")
        if selected.pointer_key and not selected.pointer_selected:
            warnings.append(f"the 12F pointer (m12f_sequence_store.json) names store {selected.pointer_key}, not {selected.key}: the 12F "
                            "evaluation in that area used another store")
        if not records["foundation_manifest"].available:
            warnings.append("m12_foundation_manifest.json absent: the fold-signature cross-check is skipped")
        if not all(records[r].available for r in TRANSFER_ROLES):
            absent = [records[r].rel for r in TRANSFER_ROLES if not records[r].available]
            warnings.append(f"athletes.parquet (transfer ladder) cannot be built: {', '.join(absent)} absent")
        elif not records["transfer_targets"].available:
            warnings.append("m12c_targets.parquet absent: athletes.parquet will lack nfl.q90_shared (rerun 12C for the transfer ladder)")
    existing = sorted({b for b in bundles} | {p.parent for p in (repo / "bundles").glob(f"*/{BUNDLE_FILE}")})
    return BundleSources(repo, gold_roots, stores, selected, selection, records, missing, warnings, existing)


__all__ = ["AREA_FILES", "BundleSources", "SourceRecord", "StoreCandidate", "discover_bundle_sources", "inspect_store", "npy_header",
           "run_mode_of", "warehouse_roots"]
