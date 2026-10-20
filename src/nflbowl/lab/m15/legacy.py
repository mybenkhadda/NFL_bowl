"""E15.00 (legacy) — provenance audit of the exploratory STFT / LSTM notebook artifacts (outside the repository).

The exploratory M15 notebooks wrote embeddings, checkpoints, configurations and split lists to Drive. Their results
(e.g. aligned STFT top-1 8.75 % on one 80-athlete hold-out) are [EXP]. Before any such artifact is used, this module
decides — from the files alone — what it may be used for:

    verified_fold_safe          every embedding row is tied to an official outer fold (``bundle.folds``) and a role; for every
                                fold, no held-out athlete appears in that fold's training set; the split manifest agrees with
                                the official fold map; and the recorded bundle fingerprint (when recorded) equals the current
                                one. → may enter supervised out-of-fold evaluation.
    verified_exploratory_only   a readable split exists and its own train / test athletes are disjoint, but it is not the
                                official five-fold protocol (e.g. one hold-out). → descriptive comparison only.
    unknown_provenance          anything else (no split, unreadable, ids missing, overlap, fingerprint mismatch). → never used
                                in a claim; kept as a historical reference.

Rules: sources are only READ (never moved, rewritten or deleted). ``adopt`` COPIES a file into
``<M15 run>/legacy/adopted/<sha12>/`` with a provenance sidecar. ``assert_fold_safe`` is the single gate for supervised use
(``BLOCKED_PROVENANCE`` otherwise).

Supported inputs (anything else → ``unknown_provenance`` with the reason):
* embedding tables (``.parquet`` / ``.csv``): ``nfl_id`` + numeric latent columns (prefix ``z``, ``e``, ``emb``, ``latent``,
  ``dim``, ``h``); optional ``fold``, ``role`` / ``split``, ``domain``, ``seq_key`` / ``idx``;
* embedding arrays (``.npz``): arrays ``nfl_id`` and ``Z`` / ``embeddings`` (optional ``fold``, ``role``); bare ``.npy``
  arrays have no ids and are always ``unknown_provenance``;
* split manifests: tables with ``nfl_id`` + (``fold`` and ``role``) or ``split``; JSON with ``train_ids`` / ``test_ids`` /
  ``val_ids`` (single split) or ``folds: {"<k>": {"train": [...], "held_out": [...]}}``;
* configurations (``.json`` / ``.yaml`` / ``.yml``): any keys; ``bundle_fingerprint`` and ``seed`` are read when present;
* checkpoints (``.pt`` / ``.pth`` / ``.ckpt``): loaded on CPU with ``weights_only=True`` when torch is available, to
  record parameter names / count; a load failure is recorded and the file is preserved untouched.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...artifacts.store import atomic_json, sha256_file
from ...utils.reproducibility import utc_now
from ..status import BLOCKED_PROVENANCE, ResearchBlocked

VERIFIED_FOLD_SAFE = "verified_fold_safe"
VERIFIED_EXPLORATORY_ONLY = "verified_exploratory_only"
UNKNOWN_PROVENANCE = "unknown_provenance"
PROVENANCE = (VERIFIED_FOLD_SAFE, VERIFIED_EXPLORATORY_ONLY, UNKNOWN_PROVENANCE)
TABLE_EXT, ARRAY_EXT, CKPT_EXT, CONFIG_EXT = (".parquet", ".csv"), (".npz", ".npy"), (".pt", ".pth", ".ckpt"), (".json", ".yaml", ".yml")
LATENT_RE = re.compile(r"^(z|e|emb|embedding|latent|dim|h)_?\d+$", re.IGNORECASE)
TRAIN_ROLES = {"train", "training", "validation", "val", "fit"}
HELD_ROLES = {"held_out", "heldout", "test", "holdout", "eval"}
HASH_MAX_BYTES = 512 * 2 ** 20                                   # sha256 for files up to 512 MB; larger: size + mtime identity


@dataclass
class Split:
    """A parsed split manifest. ``folds``: fold → (train ids, held-out ids). ``single`` = one train / test split."""

    path: str
    folds: dict[int, tuple[set[int], set[int]]] = field(default_factory=dict)
    single: bool = False
    fold_of: dict[int, int] = field(default_factory=dict)        # athlete → fold, when the manifest names folds per athlete


@dataclass
class LegacyRecord:
    path: str
    kind: str                                                      # embedding_table · embedding_array · split_manifest · config · checkpoint · other
    bytes: int
    identity: str                                                  # sha256 or size:mtime
    readable: bool
    provenance: str = UNKNOWN_PROVENANCE
    reasons: list[str] = field(default_factory=list)
    n_rows: int | None = None
    n_athletes: int | None = None
    latent_dim: int | None = None
    has_fold: bool = False
    has_role: bool = False
    split_manifest: str | None = None
    recorded_bundle: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "bytes": self.bytes, "identity": self.identity, "readable": self.readable,
                "provenance": self.provenance, "reasons": "; ".join(self.reasons), "n_rows": self.n_rows, "n_athletes": self.n_athletes,
                "latent_dim": self.latent_dim, "has_fold": self.has_fold, "has_role": self.has_role, "split_manifest": self.split_manifest,
                "recorded_bundle": self.recorded_bundle, "detail": json.dumps(self.detail, default=str)[:2000]}


# ====================================================================== discovery
def find(dirs: Sequence[str | Path], max_files: int = 5000) -> list[Path]:
    """Every supported file under the legacy directories (sorted, de-duplicated; hidden files skipped)."""
    exts = (*TABLE_EXT, *ARRAY_EXT, *CKPT_EXT, *CONFIG_EXT)
    out: set[Path] = set()
    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue
        it = [root] if root.is_file() else root.rglob("*")
        for p in it:
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
                out.add(p)
                if len(out) >= max_files:
                    break
    return sorted(out)


def _identity(p: Path) -> str:
    st = p.stat()
    return sha256_file(p) if st.st_size <= HASH_MAX_BYTES else f"size:{st.st_size}:mtime:{int(st.st_mtime)}"


def _read_table(p: Path) -> Any:
    import polars as pl

    return pl.read_parquet(p) if p.suffix.lower() == ".parquet" else pl.read_csv(p, infer_schema_length=10_000)


def _read_config(p: Path) -> dict[str, Any]:
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as e:                                    # pyyaml is a core dependency; kept explicit
            raise RuntimeError(f"cannot read {p.name}: pyyaml missing") from e
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {"_value": obj}


def _role_col(cols: Sequence[str]) -> str | None:
    for c in ("role", "split", "split_role", "set"):
        if c in cols:
            return c
    return None


def _norm_role(v: Any) -> str | None:
    s = str(v).strip().lower()
    return "train" if s in TRAIN_ROLES else ("held_out" if s in HELD_ROLES else None)


def _ids(v: Any) -> set[int]:
    return {int(x) for x in (v or [])}


# ====================================================================== split manifests
def parse_split(p: Path) -> Split | None:
    """A split manifest from a table or JSON file, or None when the file is not one (see module docstring)."""
    try:
        if p.suffix.lower() in TABLE_EXT:
            t = _read_table(p)
            if "nfl_id" not in t.columns:
                return None
            rc = _role_col(t.columns)
            ids = [int(i) for i in t["nfl_id"].to_list()]
            if "fold" in t.columns and rc is not None:
                sp = Split(str(p))
                for i, f, r in zip(ids, t["fold"].to_list(), t[rc].to_list(), strict=True):
                    role = _norm_role(r)
                    if f is None or role is None:
                        continue
                    tr, ho = sp.folds.setdefault(int(f), (set(), set()))
                    (tr if role == "train" else ho).add(i)
                    if role == "held_out":
                        sp.fold_of[i] = int(f)
                return sp if sp.folds else None
            if "fold" in t.columns and rc is None and len(set(ids)) == len(ids):
                fold_of = {i: int(f) for i, f in zip(ids, t["fold"].to_list(), strict=True) if f is not None}
                ks = sorted(set(fold_of.values()))
                return Split(str(p), {k: ({i for i, f in fold_of.items() if f != k}, {i for i, f in fold_of.items() if f == k}) for k in ks},
                             fold_of=fold_of)
            if rc is not None:
                tr = {i for i, r in zip(ids, t[rc].to_list(), strict=True) if _norm_role(r) == "train"}
                ho = {i for i, r in zip(ids, t[rc].to_list(), strict=True) if _norm_role(r) == "held_out"}
                return Split(str(p), {0: (tr, ho)}, single=True) if (tr or ho) else None
            return None
        if p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return None
            if isinstance(obj.get("folds"), dict):
                sp = Split(str(p))
                for k, v in obj["folds"].items():
                    tr = _ids(v.get("train")) | _ids(v.get("validation")) | _ids(v.get("val"))
                    ho = _ids(v.get("held_out")) | _ids(v.get("test"))
                    sp.folds[int(k)] = (tr, ho)
                    sp.fold_of.update({i: int(k) for i in ho})
                return sp if sp.folds else None
            if any(k in obj for k in ("train_ids", "test_ids", "held_out_ids")):
                tr = _ids(obj.get("train_ids")) | _ids(obj.get("val_ids"))
                ho = _ids(obj.get("test_ids")) | _ids(obj.get("held_out_ids"))
                return Split(str(p), {0: (tr, ho)}, single=True)
    except Exception:                                               # noqa: BLE001 — an unreadable file is simply not a manifest
        return None
    return None


def split_verdict(sp: Split, official_fold_of: dict[int, int], outer_folds: int) -> tuple[str, list[str]]:
    """(provenance the split alone allows, reasons). Fold-safe needs the official folds; a disjoint single split is
    exploratory-only."""
    reasons: list[str] = []
    overlap = {k: len(tr & ho) for k, (tr, ho) in sp.folds.items() if tr & ho}
    if overlap:
        return UNKNOWN_PROVENANCE, [f"train ∩ held-out not empty in split(s) {overlap}"]
    if sp.single or len(sp.folds) != outer_folds:
        reasons.append(f"split has {len(sp.folds)} fold(s), not the official {outer_folds}-fold protocol")
        return VERIFIED_EXPLORATORY_ONLY, reasons
    bad_fold = {i: (f, official_fold_of.get(i)) for i, f in sp.fold_of.items() if official_fold_of.get(i) != f}
    if bad_fold:
        reasons.append(f"{len(bad_fold)} athlete(s) held out in a fold different from bundle.folds, e.g. {list(bad_fold.items())[:3]}")
        return VERIFIED_EXPLORATORY_ONLY, reasons
    leak = {k: len({i for i in tr if official_fold_of.get(i) == k}) for k, (tr, _ho) in sp.folds.items()}
    leak = {k: n for k, n in leak.items() if n}
    if leak:
        return UNKNOWN_PROVENANCE, [f"training sets contain athletes of their own held-out fold: {leak}"]
    return VERIFIED_FOLD_SAFE, ["split equals the official athlete-disjoint folds"]


# ====================================================================== artifacts
def _latent_cols(cols: Sequence[str]) -> list[str]:
    return [c for c in cols if LATENT_RE.match(str(c))]


def inspect_checkpoint(p: Path) -> dict[str, Any]:
    """Parameter names / count of a checkpoint (CPU, ``weights_only=True``). Never raises: failures are returned."""
    try:
        import torch
    except ImportError:
        return {"loaded": False, "reason": "torch not installed in this runtime"}
    try:
        obj = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as e:                                          # noqa: BLE001 — incompatibility is recorded, file preserved
        return {"loaded": False, "reason": f"{type(e).__name__}: {e}"[:500]}
    sd = obj
    for k in ("state_dict", "model_state_dict", "model", "weights"):
        if isinstance(obj, dict) and isinstance(obj.get(k), dict):
            sd = obj[k]
            break
    if not isinstance(sd, dict):
        return {"loaded": True, "reason": f"unrecognised checkpoint object {type(obj).__name__}", "n_tensors": 0}
    tens = {k: v for k, v in sd.items() if hasattr(v, "shape")}
    meta = {k: obj[k] for k in obj if k not in ("state_dict", "model_state_dict", "model", "weights", "optimizer", "optimizer_state_dict")} \
        if isinstance(obj, dict) else {}
    return {"loaded": True, "n_tensors": len(tens), "n_params": int(sum(int(np.prod(v.shape)) for v in tens.values())),
            "first_keys": list(tens)[:12], "shapes": {k: list(v.shape) for k, v in list(tens.items())[:12]},
            "metadata": {k: (v if isinstance(v, (int, float, str, bool)) else type(v).__name__) for k, v in list(meta.items())[:20]}}


def _embedding_from_table(t: Any) -> dict[str, Any]:
    cols = list(t.columns)
    lat = _latent_cols(cols)
    rc = _role_col(cols)
    return {"has_ids": "nfl_id" in cols, "latent": lat, "fold": "fold" in cols, "role_col": rc, "n_rows": t.height,
            "athletes": set(int(i) for i in t["nfl_id"].to_list()) if "nfl_id" in cols else set(),
            "rows": (t.select("nfl_id", "fold", rc).to_dicts() if ("nfl_id" in cols and "fold" in cols and rc) else None)}


def _embedding_rows_verdict(rows: list[dict[str, Any]], role_col: str, official: dict[int, int], outer_folds: int) -> tuple[str, list[str]]:
    """Per-row check for tables that carry (nfl_id, fold, role): held-out rows must sit in the athlete's official fold;
    training rows must never come from an athlete whose official fold is the row's fold."""
    held_bad = train_bad = unknown_role = 0
    folds = set()
    for r in rows:
        f, role, i = r.get("fold"), _norm_role(r.get(role_col)), r.get("nfl_id")
        if f is None or i is None or role is None:
            unknown_role += 1
            continue
        folds.add(int(f))
        if role == "held_out" and official.get(int(i)) != int(f):
            held_bad += 1
        if role == "train" and official.get(int(i)) == int(f):
            train_bad += 1
    if train_bad:
        return UNKNOWN_PROVENANCE, [f"{train_bad} training row(s) from athletes of the same official fold (leakage)"]
    if held_bad:
        return VERIFIED_EXPLORATORY_ONLY, [f"{held_bad} held-out row(s) outside the athlete's official fold"]
    if len(folds) != outer_folds:
        return VERIFIED_EXPLORATORY_ONLY, [f"rows cover folds {sorted(folds)}, not all {outer_folds} official folds"]
    reasons = ["rows tagged with official fold and role; no leakage"]
    if unknown_role:
        reasons.append(f"{unknown_role} row(s) with an unrecognised role ignored")
    return VERIFIED_FOLD_SAFE, reasons


def audit(dirs: Sequence[str | Path], bundle: Any, outer_folds: int = 5, split_paths: Sequence[str | Path] = (),
          official_fold_of: dict[int, int] | None = None) -> tuple[Any, list[LegacyRecord]]:
    """Audit every legacy artifact. Returns (table ``legacy/prototype_audit``, records). No file is modified.

    Pairing: an embedding is checked against (a) its own fold / role columns, else (b) the explicit ``split_paths``, else
    (c) the single split manifest found in its own directory (two or more there → ambiguous → unknown_provenance)."""
    import polars as pl

    from ...data.splits import fold_map

    official = official_fold_of or fold_map(bundle.meta, bundle.folds, outer_folds)
    current_fp = bundle.fingerprint
    files = find(dirs)
    explicit = [s for s in (parse_split(Path(p)) for p in split_paths) if s is not None]
    splits_by_dir: dict[Path, list[Split]] = {}
    configs_by_dir: dict[Path, list[dict[str, Any]]] = {}
    records: list[LegacyRecord] = []
    for p in files:
        rec = LegacyRecord(str(p), "other", p.stat().st_size, _identity(p), True)
        ext = p.suffix.lower()
        try:
            table = _read_table(p) if ext in TABLE_EXT else None
            is_embedding = table is not None and "nfl_id" in table.columns and bool(_latent_cols(table.columns))
            sp = parse_split(p) if (ext in (*TABLE_EXT, ".json") and not is_embedding) else None
            if sp is not None:
                rec.kind, rec.split_manifest = "split_manifest", str(p)
                rec.provenance, rec.reasons = split_verdict(sp, official, outer_folds)
                rec.n_athletes = len(set().union(*[a | b for a, b in sp.folds.values()]))
                splits_by_dir.setdefault(p.parent, []).append(sp)
            elif table is not None:
                t = table
                info = _embedding_from_table(t)
                rec.kind = "embedding_table" if (info["has_ids"] and info["latent"]) else "table"
                rec.n_rows, rec.n_athletes, rec.latent_dim = info["n_rows"], len(info["athletes"]), len(info["latent"]) or None
                rec.has_fold, rec.has_role = info["fold"], info["role_col"] is not None
                rec.detail = {"columns": list(t.columns)[:40], "_rows": info["rows"], "_athletes": sorted(info["athletes"])[:100000]}
            elif ext == ".npz":
                with np.load(p, allow_pickle=False) as z:
                    keys = list(z.keys())
                    zkey = next((k for k in ("Z", "embeddings", "emb", "z") if k in keys), None)
                    rec.kind = "embedding_array"
                    rec.detail = {"keys": keys}
                    if zkey is not None:
                        rec.n_rows, rec.latent_dim = int(z[zkey].shape[0]), int(z[zkey].shape[-1]) if z[zkey].ndim > 1 else 1
                    if "nfl_id" in keys:
                        ids = z["nfl_id"].astype(np.int64)
                        rec.n_athletes = int(np.unique(ids).size)
                        rec.detail["_athletes"] = sorted(set(ids.tolist()))
                        if "fold" in keys and "role" in keys:
                            rec.has_fold = rec.has_role = True
                            rec.detail["_rows"] = [{"nfl_id": int(i), "fold": int(f), "role": str(r)} for i, f, r in
                                                   zip(ids.tolist(), z["fold"].tolist(), z["role"].tolist(), strict=True)]
            elif ext == ".npy":
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
                rec.kind, rec.n_rows = "embedding_array", int(arr.shape[0])
                rec.latent_dim = int(arr.shape[-1]) if arr.ndim > 1 else 1
                rec.reasons.append("bare .npy array has no athlete ids: provenance cannot be verified")
            elif ext in CONFIG_EXT:
                obj = _read_config(p)
                rec.kind = "config"
                fp = obj.get("bundle_fingerprint")
                if fp is None and isinstance(obj.get("bundle"), dict):
                    fp = obj["bundle"].get("fingerprint")
                rec.recorded_bundle = str(fp) if fp else None
                rec.detail = {"keys": list(obj)[:40], "seed": obj.get("seed")}
                configs_by_dir.setdefault(p.parent, []).append({"path": str(p), "bundle": rec.recorded_bundle})
            elif ext in CKPT_EXT:
                rec.kind = "checkpoint"
                rec.detail = inspect_checkpoint(p)
                rec.readable = bool(rec.detail.get("loaded"))
                if not rec.readable:
                    rec.reasons.append(f"checkpoint not loadable here: {rec.detail.get('reason')} (artifact preserved)")
        except Exception as e:                                      # noqa: BLE001 — recorded per file, audit continues
            rec.readable = False
            rec.reasons.append(f"unreadable: {type(e).__name__}: {e}"[:400])
        records.append(rec)

    # ---- second pass: decide embedding provenance
    for rec in records:
        if rec.kind not in ("embedding_table", "embedding_array") or not rec.readable:
            continue
        here = Path(rec.path).parent
        cfgs = [c for c in configs_by_dir.get(here, []) if c["bundle"]]
        if cfgs:
            rec.recorded_bundle = cfgs[0]["bundle"]
        if rec.recorded_bundle and rec.recorded_bundle != current_fp:
            rec.provenance = UNKNOWN_PROVENANCE
            rec.reasons.append(f"recorded bundle {rec.recorded_bundle} ≠ current {current_fp}")
            continue
        rows = rec.detail.get("_rows")
        if rows:
            role_col = "role" if "role" in rows[0] else next(k for k in rows[0] if k not in ("nfl_id", "fold"))
            rec.provenance, why = _embedding_rows_verdict(rows, role_col, official, outer_folds)
            rec.reasons += why
        else:
            cands = explicit or splits_by_dir.get(here, [])
            if len(cands) == 1:
                rec.split_manifest = cands[0].path
                rec.provenance, why = split_verdict(cands[0], official, outer_folds)
                rec.reasons += why
                if rec.provenance == VERIFIED_FOLD_SAFE:
                    rec.provenance = VERIFIED_EXPLORATORY_ONLY       # rows are not fold-tagged: per-fold embeddings cannot be told apart
                    rec.reasons.append("embedding rows carry no (fold, role): cannot show which fold model produced each row")
            elif len(cands) > 1:
                rec.reasons.append(f"{len(cands)} candidate split manifests: ambiguous pairing (pass legacy_split_manifests)")
            else:
                rec.reasons.append("no split manifest found for this embedding")
        if not rec.recorded_bundle and rec.provenance == VERIFIED_FOLD_SAFE:
            rec.provenance = VERIFIED_EXPLORATORY_ONLY
            rec.reasons.append("no recorded bundle fingerprint next to the embedding: the data version cannot be confirmed")
        ath = set(rec.detail.get("_athletes", []))
        if ath:
            unknown = ath - set(official)
            if unknown:
                rec.provenance = UNKNOWN_PROVENANCE
                rec.reasons.append(f"{len(unknown)} athlete id(s) absent from the current bundle")
    for rec in records:
        rec.detail.pop("_rows", None)
        rec.detail.pop("_athletes", None)
    tab = pl.DataFrame([r.row() for r in records], infer_schema_length=None) if records else pl.DataFrame(
        schema={"path": pl.String, "kind": pl.String, "provenance": pl.String, "reasons": pl.String})
    return tab, records


def summary(records: Sequence[LegacyRecord]) -> Any:
    """Counts per (kind, provenance) — the line shown in the notebook."""
    import polars as pl

    if not records:
        return pl.DataFrame({"kind": [], "provenance": [], "n": []}, schema={"kind": pl.String, "provenance": pl.String, "n": pl.Int64})
    return pl.DataFrame([{"kind": r.kind, "provenance": r.provenance} for r in records]).group_by("kind", "provenance").len("n").sort("kind", "provenance")


def assert_fold_safe(record: LegacyRecord | dict[str, Any], purpose: str = "supervised out-of-fold evaluation") -> None:
    """The single gate for supervised use of a legacy artifact."""
    prov = record.provenance if isinstance(record, LegacyRecord) else record.get("provenance")
    path = record.path if isinstance(record, LegacyRecord) else record.get("path")
    if prov != VERIFIED_FOLD_SAFE:
        raise ResearchBlocked(BLOCKED_PROVENANCE, f"{path} is {prov}: not allowed in {purpose}", detail={"path": path, "provenance": prov})


def adopt(record: LegacyRecord, run: Any) -> Path:
    """COPY one audited artifact into ``<run>/legacy/adopted/<identity12>/`` with a provenance sidecar (sources untouched).
    Idempotent: an existing copy with the same identity is reused."""
    src = Path(record.path)
    key = (record.identity.replace(":", "_"))[:12]
    dst = run.guard(run.root / "legacy" / "adopted" / key / src.name)
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.copying")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
    atomic_json({**record.row(), "adopted_at": utc_now(), "source": str(src)}, dst.with_name(dst.name + ".provenance.json"), overwrite=True)
    return dst


__all__ = ["LegacyRecord", "PROVENANCE", "Split", "UNKNOWN_PROVENANCE", "VERIFIED_EXPLORATORY_ONLY", "VERIFIED_FOLD_SAFE", "adopt", "assert_fold_safe",
           "audit", "find", "inspect_checkpoint", "parse_split", "split_verdict", "summary"]
