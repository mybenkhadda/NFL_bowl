"""E16.B4 — provenance registry of every M15 representation artifact (read only) before any Track B / C use.

Discovered per (architecture, d, seed, outer fold f) from the M15 run: ``embeddings/inductive/<arch>_d<d>_s<seed>__fold<f>
.parquet`` (+ ``.provenance.json``), ``checkpoints/<tag>/fold<f>/{done.json, train_athletes.json, config.json, best.pt |
pca.npz, history.parquet}`` and the M15 run configuration (channel order, scaling, STFT transform, training domains).

Per fold artifact the audit records: source bundle fingerprint, code / config fingerprint, architecture, latent dimension,
seed, fold, training-athlete set (hash, size), evaluation athletes (held-out rows), normaliser fitting cohort, channel
order and scaling, input transform, checkpoint identity (sha256), export schema, and whether held-out athletes' data
(Combine or NFL) influenced the representation. Status:

    INVALID_LEAKAGE      a fold-f athlete is in the fold-f training / validation set, or a training-role row belongs to a
                         fold-f athlete (the representation saw a held-out athlete)
    UNKNOWN_PROVENANCE   a required record is missing or inconsistent (no done.json / train_athletes.json, hash mismatch,
                         bundle mismatch, schema violation, channel scaling differs from the M15 contract)
    EXPLORATORY_ONLY     every present fold is clean but the five-fold protocol is incomplete (missing folds), or the
                         artifact is a legacy file verified only for descriptive use
    VERIFIED_FOLD_SAFE   every check passes on all outer folds

A representation (arch, d, seed) takes the worst status of its folds. Only VERIFIED_FOLD_SAFE representations may enter
supervised out-of-fold comparisons (``assert_supervised``); everything else is BLOCKED_PROVENANCE there.

The G15 decision (M15 gate) and the presence of E15.C preservation evidence are attached, never inferred: M15 E15.C2
(robustness) reads ``cfg.dropout_rates``, a key ``M15Config`` does not define — that blocker is reported here as missing
preservation evidence; it is not worked around and reconstruction is never declared validated by M16.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from ..status import BLOCKED_PROVENANCE, ResearchBlocked

VERIFIED_FOLD_SAFE = "VERIFIED_FOLD_SAFE"
EXPLORATORY_ONLY = "EXPLORATORY_ONLY"
UNKNOWN_PROVENANCE = "UNKNOWN_PROVENANCE"
INVALID_LEAKAGE = "INVALID_LEAKAGE"
ORDER = (INVALID_LEAKAGE, UNKNOWN_PROVENANCE, EXPLORATORY_ONLY, VERIFIED_FOLD_SAFE)          # worst first
EXPECTED_COLUMNS = ("idx", "nfl_id", "domain", "kind", "position_group", "fold", "role")
ROLES = {"train", "validation", "held_out"}
TAG_RE = re.compile(r"^(?P<arch>[a-z0-9_]+)_d(?P<d>\d+)_s(?P<seed>\d+)__fold(?P<fold>\d+)$")


def worst(statuses: Sequence[str]) -> str:
    for s in ORDER:
        if s in statuses:
            return s
    return UNKNOWN_PROVENANCE


def discover(m15_run_dir: str) -> list[dict[str, Any]]:
    """Every inductive embedding file of the M15 run, parsed from its name."""
    root = Path(m15_run_dir) / "embeddings" / "inductive"
    out = []
    for p in sorted(root.glob("*__fold*.parquet")) if root.is_dir() else []:
        m = TAG_RE.match(p.stem)
        if not m:
            out.append({"path": str(p), "parse_error": "file name does not follow <arch>_d<d>_s<seed>__fold<f>"})
            continue
        out.append({"path": str(p), "arch": m["arch"], "d": int(m["d"]), "seed": int(m["seed"]), "fold": int(m["fold"]), "tag": p.stem.split("__fold")[0]})
    return out


def m15_snapshot(m15_run_dir: str) -> dict[str, Any] | None:
    """The M15 run configuration (``config/run_config.json``) or None."""
    p = Path(m15_run_dir) / "config" / "run_config.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def g15_decisions(m15_run_dir: str) -> dict[tuple[str, int], str]:
    p = Path(m15_run_dir) / "tables" / "synthesis" / "m15_gate.parquet"
    if not p.is_file():
        return {}
    import polars as pl

    g = pl.read_parquet(p)
    if not {"arch", "d", "criterion", "status"} <= set(g.columns):
        return {}
    g = g.filter(pl.col("criterion") == "G15 decision")
    return {(r["arch"], int(r["d"])): r["status"] for r in g.iter_rows(named=True)}


def preservation_evidence(m15_run_dir: str) -> dict[str, Any]:
    """Which E15.C preservation tables exist, plus the known configuration blocker of E15.C2."""
    from ..m15.config import M15Config

    root = Path(m15_run_dir) / "tables" / "preservation"
    files = {n: (root / f"{n}.parquet").is_file() for n in ("metrics", "paired_vs_pca", "robustness")}
    blocker = None if hasattr(M15Config(), "dropout_rates") else ("E15.C2 robustness cannot run: m15.preservation.robustness reads cfg.dropout_rates, which "
                                                                  "M15Config does not define (configuration / API disagreement); G15.f stays NOT_EVALUATED")
    return {"tables": files, "complete": all(files.values()), "blocker": blocker}


def _sha(p: Path) -> str | None:
    from ...artifacts.store import sha256_file

    try:
        return sha256_file(p) if p.is_file() else None
    except OSError:
        return None


def audit_fold(rec: dict[str, Any], m15_run_dir: str, fold_of: dict[int, int], bundle_fp: str, snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Checks of one (representation, fold) artifact → a record with ``status`` and every reason."""
    import polars as pl

    from ..m15.config import CHANNELS, SCALE
    from ..m15.training import athlete_hash

    f = rec["fold"]
    ck = Path(m15_run_dir) / "checkpoints" / rec["tag"] / f"fold{f}"
    out = {**{k: rec.get(k) for k in ("arch", "d", "seed", "fold", "tag", "path")}, "checkpoint_dir": str(ck)}
    bad: dict[str, list[str]] = {INVALID_LEAKAGE: [], UNKNOWN_PROVENANCE: [], EXPLORATORY_ONLY: []}
    done_p, ta_p = ck / "done.json", ck / "train_athletes.json"
    weights = ck / ("pca.npz" if rec["arch"] == "pca" else "best.pt")
    out["checkpoint_sha256"] = _sha(weights)
    if out["checkpoint_sha256"] is None:
        bad[UNKNOWN_PROVENANCE].append(f"weights file missing ({weights.name})")
    done = json.loads(done_p.read_text(encoding="utf-8")) if done_p.is_file() else None
    ta = json.loads(ta_p.read_text(encoding="utf-8")) if ta_p.is_file() else None
    if done is None:
        bad[UNKNOWN_PROVENANCE].append("done.json missing: the fit is not recorded as finished")
    if ta is None:
        bad[UNKNOWN_PROVENANCE].append("train_athletes.json missing: the training cohort is unknown")
    side_p = Path(rec["path"]).with_suffix(".provenance.json")
    side = json.loads(side_p.read_text(encoding="utf-8")) if side_p.is_file() else None
    out["has_sidecar"] = side is not None
    # schema and roles
    try:
        emb = pl.read_parquet(rec["path"])
    except Exception as e:                                           # noqa: BLE001 — recorded, never raised
        bad[UNKNOWN_PROVENANCE].append(f"embedding unreadable: {type(e).__name__}: {e}"[:200])
        emb = None
    if emb is not None:
        miss = [c for c in EXPECTED_COLUMNS if c not in emb.columns]
        zc = [c for c in emb.columns if c.startswith("z") and c[1:].isdigit()]
        out.update({"n_rows": emb.height, "latent_columns": len(zc), "schema_ok": not miss and len(zc) == rec["d"]})
        if miss or len(zc) != rec["d"]:
            bad[UNKNOWN_PROVENANCE].append(f"schema: missing {miss}, {len(zc)} latent columns for d = {rec['d']}")
        if not miss:
            roles = set(emb["role"].unique().to_list())
            if roles - ROLES:
                bad[UNKNOWN_PROVENANCE].append(f"unknown role values {sorted(roles - ROLES)}")
            off = emb.with_columns(pl.col("nfl_id").cast(pl.Int64).replace_strict(fold_of, default=-1, return_dtype=pl.Int64).alias("_of"))
            leak_rows = off.filter((pl.col("role") != "held_out") & (pl.col("_of") == f)).height
            wrong_held = off.filter((pl.col("role") == "held_out") & (pl.col("_of") != f)).height
            unknown_ath = off.filter(pl.col("_of") == -1)["nfl_id"].n_unique()
            out.update({"training_role_rows_of_fold_f": leak_rows, "held_out_rows_outside_fold_f": wrong_held, "athletes_absent_from_bundle": int(unknown_ath),
                        "n_eval_athletes": int(off.filter(pl.col("role") == "held_out")["nfl_id"].n_unique()),
                        "nan_latent_rows": int(emb.select(pl.any_horizontal([pl.col(c).is_nan() | pl.col(c).is_null() for c in zc]).sum()).item()) if zc else None})
            if leak_rows:
                bad[INVALID_LEAKAGE].append(f"{leak_rows} training/validation-role row(s) of fold-{f} athletes")
            if wrong_held:
                bad[EXPLORATORY_ONLY].append(f"{wrong_held} held-out row(s) of athletes outside official fold {f} (different fold map)")
            if unknown_ath:
                bad[UNKNOWN_PROVENANCE].append(f"{unknown_ath} athlete(s) absent from the current bundle")
    # training cohort
    if ta is not None:
        tr = [int(i) for i in ta.get("train_and_validation_athletes", [])]
        leak = [i for i in tr if fold_of.get(i) == f]
        out.update({"n_train_athletes": len(tr), "train_athlete_hash": athlete_hash(tr), "held_out_in_training": len(leak)})
        if leak:
            bad[INVALID_LEAKAGE].append(f"{len(leak)} fold-{f} athlete(s) in the training set (e.g. {leak[:5]})")
        expected = {i for i, k in fold_of.items() if k != f}
        out["training_set_vs_other_folds"] = "equal" if set(tr) == expected else f"{len(expected - set(tr))} other-fold athlete(s) not used"
        if done is not None and done.get("train_athlete_hash") != athlete_hash(tr):
            bad[UNKNOWN_PROVENANCE].append("done.json training-athlete hash ≠ hash of train_athletes.json")
        if side is not None and side.get("train_athlete_hash") not in (None, athlete_hash(tr)):
            bad[UNKNOWN_PROVENANCE].append("embedding sidecar training-athlete hash ≠ train_athletes.json")
    if done is not None:
        out.update({"bundle_fingerprint": done.get("bundle_fingerprint"), "config_fingerprint": done.get("config_fingerprint"),
                    "finished_at": done.get("finished_at"), "train_rows": done.get("train_rows"), "params": done.get("params")})
        if done.get("bundle_fingerprint") != bundle_fp:
            bad[UNKNOWN_PROVENANCE].append(f"bundle {done.get('bundle_fingerprint')} ≠ current {bundle_fp}")
    if side is not None and side.get("model_fold") not in (None, f):
        bad[UNKNOWN_PROVENANCE].append(f"sidecar model_fold {side.get('model_fold')} ≠ {f}")
    # input contract
    chans = tuple(snapshot.get("channels", ())) if snapshot else ()
    scale = tuple(float(x) for x in snapshot.get("scale", ())) if snapshot else ()
    out["channels"] = ", ".join(chans) if chans else "unknown (no M15 run_config.json)"
    out["scale"] = ", ".join(f"{x:g}" for x in scale) if scale else "unknown"
    if snapshot is None:
        bad[UNKNOWN_PROVENANCE].append("M15 run configuration missing: channel order / scaling not recorded")
    elif chans != tuple(CHANNELS) or tuple(round(x, 9) for x in scale) != tuple(round(float(x), 9) for x in SCALE):
        bad[UNKNOWN_PROVENANCE].append(f"channel contract differs from M15 constants ({chans} / {scale})")
    doms = tuple(snapshot.get("train_domains", ())) if snapshot else ()
    if rec["arch"] == "stft_ae" and snapshot:
        out["input_transform"] = f"STFT magnitude nperseg={snapshot.get('stft_nperseg')} hop={snapshot.get('stft_hop')} phase={snapshot.get('stft_phase')}"
    else:
        out["input_transform"] = "time domain (scaled channels, mask)" if rec["arch"] != "pca" else "time domain, masked closed-form PCA"
    out["normaliser_cohort"] = ("closed-form PCA fitted on training + validation athletes' rows of fold-f (domains " + ", ".join(doms) + ")" if rec["arch"] == "pca"
                                else "fixed per-channel scaling (no fitted normaliser); weights fitted on training-role rows, early stopping on validation athletes")
    out["held_out_influence"] = ("none: fold-f athletes absent from training and early stopping; their rows are only embedded" if not bad[INVALID_LEAKAGE]
                                 else "LEAKAGE: see reasons")
    out["nfl_influence"] = ("NFL windows of TRAINING athletes shaped the encoder (unsupervised); no target value and no held-out athlete's NFL data"
                            if "nfl" in doms else "Combine-only training" if doms else "unknown")
    status = INVALID_LEAKAGE if bad[INVALID_LEAKAGE] else (UNKNOWN_PROVENANCE if bad[UNKNOWN_PROVENANCE] else
                                                          (EXPLORATORY_ONLY if bad[EXPLORATORY_ONLY] else VERIFIED_FOLD_SAFE))
    out["status"] = status
    out["reasons"] = "; ".join(sum(bad.values(), [])) or "all checks passed"
    return out


def legacy_rows(cfg: Any, bundle: Any, fold_of: dict[int, int], outer_folds: int) -> Any:
    """``m15.legacy.audit`` of the exploratory folders, mapped to the four statuses (descriptive use only in M16)."""
    import polars as pl

    from ..m15 import legacy as LG

    if not cfg.legacy_m15_dirs:
        return pl.DataFrame()
    _tab, recs = LG.audit(cfg.legacy_m15_dirs, bundle, outer_folds, cfg.legacy_split_manifests, fold_of)
    m = {LG.VERIFIED_FOLD_SAFE: VERIFIED_FOLD_SAFE, LG.VERIFIED_EXPLORATORY_ONLY: EXPLORATORY_ONLY, LG.UNKNOWN_PROVENANCE: UNKNOWN_PROVENANCE}
    rows = []
    for r in recs:
        if r.kind not in ("embedding_table", "embedding_array"):
            continue
        reasons = "; ".join(r.reasons)
        st = INVALID_LEAKAGE if "leakage" in reasons.lower() or "own held-out fold" in reasons.lower() else m.get(r.provenance, UNKNOWN_PROVENANCE)
        rows.append({"source": "legacy", "path": r.path, "kind": r.kind, "latent_dim": r.latent_dim, "n_athletes": r.n_athletes, "status": st,
                     "reasons": reasons, "usable_for": "descriptive comparison only (not loaded by the M16 ladder)"})
    return pl.DataFrame(rows, infer_schema_length=None)


def registry(cfg: Any, fold_of: dict[int, int], bundle_fp: str, outer_folds: int) -> tuple[Any, Any, dict[str, Any]]:
    """(per-fold audit table, per-representation registry, preservation evidence)."""
    import polars as pl

    snap = m15_snapshot(cfg.m15_run_dir)
    found = discover(cfg.m15_run_dir)
    rows = [{**r, "status": UNKNOWN_PROVENANCE, "reasons": r["parse_error"]} if "parse_error" in r else audit_fold(r, cfg.m15_run_dir, fold_of, bundle_fp, snap)
            for r in found]
    folds = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(schema={"arch": pl.String, "d": pl.Int64, "seed": pl.Int64, "fold": pl.Int64,
                                                                                        "status": pl.String, "reasons": pl.String})
    g15 = g15_decisions(cfg.m15_run_dir)
    pres = preservation_evidence(cfg.m15_run_dir)
    reg = []
    if folds.height and "arch" in folds.columns:
        for (arch, d, seed), g in folds.filter(pl.col("arch").is_not_null()).group_by("arch", "d", "seed"):
            present = sorted(int(x) for x in g["fold"].to_list())
            st = worst(g["status"].to_list())
            reasons = [x for x in g["reasons"].to_list() if x and x != "all checks passed"]
            if st == VERIFIED_FOLD_SAFE and present != list(range(outer_folds)):
                st = EXPLORATORY_ONLY
                reasons.append(f"folds present {present}: the five-fold protocol is incomplete")
            reg.append({"rep": f"{arch}_d{int(d)}", "arch": arch, "d": int(d), "seed": int(seed), "n_folds": len(present), "folds": json.dumps(present),
                        "status": st, "usable_supervised": st == VERIFIED_FOLD_SAFE, "g15_decision": g15.get((arch, int(d)), "NOT_EVALUATED (no M15 gate row)"),
                        "preservation_tables_complete": pres["complete"], "preservation_blocker": pres["blocker"] or "",
                        "reasons": " | ".join(sorted(set(reasons)))[:2000] or "all checks passed on every fold"})
    out = pl.DataFrame(reg, infer_schema_length=None).sort("arch", "d", "seed") if reg else pl.DataFrame(
        schema={"rep": pl.String, "arch": pl.String, "d": pl.Int64, "seed": pl.Int64, "status": pl.String, "usable_supervised": pl.Boolean})
    return folds, out, pres


def choose(reg: Any, arch: str, d: int) -> tuple[dict[str, Any] | None, str]:
    """The pre-declared representation of ``arch`` at dimension ``d`` (smallest seed among VERIFIED_FOLD_SAFE ones) and a
    reason. Never chosen on retrieval or outcome results."""
    import polars as pl

    if reg is None or not reg.height:
        return None, "no M15 representation found"
    cand = reg.filter((pl.col("arch") == arch) & (pl.col("d") == d))
    if not cand.height:
        others = sorted(set(reg.filter(pl.col("arch") == arch)["d"].to_list()))
        return None, f"no {arch} representation at the pre-declared d = {d} (available d: {others or 'none'})"
    ok = cand.filter(pl.col("status") == VERIFIED_FOLD_SAFE).sort("seed")
    if not ok.height:
        return None, f"{arch}_d{d}: provenance {cand['status'].to_list()} — not VERIFIED_FOLD_SAFE ({cand['reasons'][0][:300]})"
    return ok.row(0, named=True), f"{arch}_d{d} seed {ok['seed'][0]} VERIFIED_FOLD_SAFE (G15: {ok['g15_decision'][0]})"


def assert_supervised(reg: Any, rep: str) -> None:
    """BLOCKED_PROVENANCE unless the representation is VERIFIED_FOLD_SAFE."""
    import polars as pl

    r = reg.filter(pl.col("rep") == rep) if reg is not None and reg.height else None
    st = r["status"][0] if r is not None and r.height else "not in registry"
    if st != VERIFIED_FOLD_SAFE:
        raise ResearchBlocked(BLOCKED_PROVENANCE, f"{rep} is {st}: not allowed in supervised out-of-fold comparisons", detail={"rep": rep, "status": st})


def summary_counts(reg: Any) -> Any:
    import polars as pl

    if reg is None or not reg.height:
        return pl.DataFrame(schema={"status": pl.String, "n": pl.UInt32})
    return reg.group_by("status").len("n").sort("status")


__all__ = ["EXPECTED_COLUMNS", "EXPLORATORY_ONLY", "INVALID_LEAKAGE", "ORDER", "UNKNOWN_PROVENANCE", "VERIFIED_FOLD_SAFE", "assert_supervised", "audit_fold",
           "choose", "discover", "g15_decisions", "legacy_rows", "m15_snapshot", "preservation_evidence", "registry", "summary_counts", "worst"]
