"""Part A — M12 → M13 hand-off: environment record, read-only discovery and adoption of the M12 inductive encoders and
their embeddings, the (model × fold) readiness inventory, reconciliation of saved M12 artifacts with the published M12
summaries, research-gate prerequisites, and the F1 (Milestone 06 TCN) fold-validity check.

Rules
* Sources (M12 experiment directories) are only READ. Adoption copies ``best.pt`` (and, optionally, embedding Parquet
  files) into the M13 experiment and registers them with the SOURCE dependency record.
* A checkpoint is valid for M13 only when: bundle fingerprint equal, science hash equal, regime ``inductive``, tag equal
  to ``inductive_f<fold>`` with the same fold, the file exists, the saved ``EncConfig`` equals the current registry
  architecture, and (optionally) the file loads with a strict state-dict match.
* An embedding file is adopted only when its dependency record names the SAME weights checksum as the adopted
  checkpoint and the same embedding precision — so ``Experiment._embed_one(reuse=True)`` finds it by exact deps.
* Nothing falls back to transductive representations.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from ..artifacts.store import atomic_json, atomic_write
from ..utils.reproducibility import code_version, utc_now
from .m13_config import EXPECTED, M12_REPORTED, M13Config

INDUCTIVE = "inductive"
M12_TABLES = (("retrieval", "paired_inductive_comparison"), ("retrieval", "inductive_coverage"), ("retrieval", "domain_diagnostics_comparison"),
              ("geometry", "geometry_summary"), ("geometry", "geometry_summary_by_fold"), ("geometry", "collapse_flags"), ("ablations", "inductive_probes"),
              ("ablations", "training_runs"), ("biomechanics", "repeatability"), ("biomechanics", "cross_domain_stability"),
              ("biomechanics", "incremental_value"), ("prediction", "prediction_metrics"), ("prediction", "paired_comparisons"), ("retrieval", "retrieval_by_fold"),
              ("prediction", "oof_predictions"), ("synthesis", "research_gates"), ("synthesis", "integrated_comparison"), ("dataset", "attrition"))


# ====================================================================== environment
def installed_revision() -> dict[str, Any]:
    """Version + VCS commit of the installed distribution (pip ``direct_url.json``; no git call)."""
    out: dict[str, Any] = {"distribution": "nflbdb", "version": None, "commit": None, "url": None, "requested_revision": None}
    try:
        from importlib import metadata

        d = metadata.distribution("nflbdb")
        out["version"] = d.version
        raw = d.read_text("direct_url.json")
        if raw:
            j = json.loads(raw)
            out["url"] = j.get("url")
            vcs = j.get("vcs_info") or {}
            out["commit"], out["requested_revision"] = vcs.get("commit_id"), vcs.get("requested_revision")
    except Exception as e:                                          # noqa: BLE001 — informational record only
        out["error"] = repr(e)
    return out


def environment(cfg: M13Config) -> dict[str, Any]:
    from .environment import environment_record

    env = environment_record(cfg.lab())
    env["revision"] = installed_revision()
    env["m13_config"] = cfg.to_dict()
    return env


def open_experiment(cfg: M13Config, run: Any) -> Any:
    """The M13 Experiment (new directory under RUN_DIR/checkpoints) with the preset science; refuses another science hash."""
    from .environment import open_experiment as open_lab

    exp = open_lab(cfg.lab(), run)
    sh = exp.cfg.science_hash()
    if cfg.expected_science_hash and sh != cfg.expected_science_hash:
        raise RuntimeError(f"science hash {sh} ≠ expected {cfg.expected_science_hash}: the M12 encoders cannot be reused under another science "
                           "configuration (keep base_config='colab_gpu' and no science_seed override)")
    return exp


# ====================================================================== sources (read-only)
def source_dirs(cfg: M13Config) -> list[Path]:
    out = [cfg.m12_experiment_dir, *[Path(p) for p in cfg.extra_source_experiments]]
    return [p for p in out if (p / "config" / "experiment.json").is_file()]


def _read_manifest(d: Path) -> dict[str, Any]:
    p = d / "manifests" / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"artifacts": {}}


def _norm(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str, sort_keys=True))


def expected_architecture(exp: Any, name: str) -> dict[str, Any] | None:
    from dataclasses import asdict

    from ..models import registry as reg

    try:
        return _norm(asdict(reg.enc_config(name, exp.cfg.science)))
    except Exception:                                               # noqa: BLE001 — baselines have no encoder config
        return None


def saved_architecture(best: Path) -> dict[str, Any] | None:
    side = best.with_suffix(".json")
    if not side.is_file():
        return None
    try:
        return _norm(json.loads(side.read_text(encoding="utf-8")).get("enc_config"))
    except (OSError, json.JSONDecodeError):
        return None


def discover(cfg: M13Config, exp: Any) -> Any:
    """Every training record of the source experiments with an explicit compatibility verdict (``compatible`` or the list
    of failed rules). Same columns as ``environment.discover_sources`` so ``environment.adopt_checkpoints`` is reused."""
    import polars as pl

    fp, sh = exp.bundle.fingerprint, exp.cfg.science_hash()
    rows = []
    for d in source_dirs(cfg):
        if d.resolve() == exp.root.resolve():
            continue
        saved = json.loads((d / "config" / "experiment.json").read_text(encoding="utf-8"))
        for a in _read_manifest(d).get("artifacts", {}).values():
            if a.get("kind") != "training":
                continue
            deps = a.get("deps", {})
            name = a["key"].split("__")[0]
            path = Path(a["path"]) if Path(a["path"]).is_absolute() else d / a["path"]
            tag = deps.get("tag") or ""
            fold = a.get("fold")
            reasons = []
            if deps.get("bundle") != fp:
                reasons.append(f"bundle {deps.get('bundle')}")
            if deps.get("science") != sh:
                reasons.append(f"science {deps.get('science')}")
            if deps.get("regime") != INDUCTIVE:
                reasons.append(f"regime {deps.get('regime')}")
            if fold is None or tag != f"inductive_f{fold}" or a["key"] != f"{name}__{tag}":
                reasons.append(f"fold identity (key {a['key']}, tag {tag}, fold {fold})")
            if not path.is_file():
                reasons.append("best.pt missing")
            else:
                want, have = expected_architecture(exp, name), saved_architecture(path)
                if want is not None and have is not None and want != have:
                    reasons.append("architecture differs from the registry EncConfig")
                if have is None:
                    reasons.append("best.json (EncConfig) missing")
            rows.append({"source": str(d), "experiment": saved.get("experiment_id"), "key": a["key"], "model": name, "tag": tag, "regime": deps.get("regime"),
                         "fold": fold, "path": str(path), "sha256": a.get("sha256"), "created_at": a.get("created_at"),
                         "precision": (a.get("signature") or {}).get("precision"), "source_bundle": deps.get("bundle"), "source_science": deps.get("science"),
                         "status": "compatible" if not reasons else "incompatible: " + "; ".join(reasons)})
    if not rows:
        return pl.DataFrame(schema={"source": pl.String, "key": pl.String, "model": pl.String, "fold": pl.Int64, "status": pl.String})
    return pl.DataFrame(rows, infer_schema_length=None)


def adopt(exp: Any, sources: Any, models: Sequence[str]) -> Any:
    from .environment import adopt_checkpoints

    return adopt_checkpoints(exp, sources, models=list(models), regime=INDUCTIVE)


def adopt_embeddings(cfg: M13Config, exp: Any) -> Any:
    """Copy source embedding files whose deps match the M13 experiment exactly (same bundle, science, model, tag, weights
    checksum of the ADOPTED checkpoint, embedding precision). Returns one row per considered record."""
    import polars as pl

    rows = []
    fp, sh = exp.bundle.fingerprint, exp.cfg.science_hash()
    eprec = exp.cfg.runtime.embed_precision
    for d in source_dirs(cfg):
        for a in _read_manifest(d).get("artifacts", {}).values():
            if a.get("kind") != "embeddings" or "__inductive_f" not in a.get("key", ""):
                continue
            deps = a.get("deps", {})
            key = a["key"]
            tr = exp.manifest.find("training", key)
            src = Path(a["path"]) if Path(a["path"]).is_absolute() else d / a["path"]
            why = []
            if tr is None:
                why.append("no adopted checkpoint")
            elif deps.get("weights") != tr.get("sha256"):
                why.append("weights checksum differs from the adopted checkpoint")
            if deps.get("bundle") != fp or deps.get("science") != sh:
                why.append("bundle / science differ")
            if deps.get("embed_precision") != eprec:
                why.append(f"embed precision {deps.get('embed_precision')} ≠ {eprec}")
            if not src.is_file():
                why.append("file missing")
            if why:
                rows.append({"key": key, "action": "not adopted: " + "; ".join(why), "from": str(d)})
                continue
            if exp.manifest.find("embeddings", key, deps) is not None:
                rows.append({"key": key, "action": "already present", "from": str(d)})
                continue
            dst = exp.store.embeddings / src.name
            atomic_write(dst, lambda t, _s=src: shutil.copyfile(_s, t), overwrite=True)
            if src.with_suffix(".json").is_file():
                atomic_write(dst.with_suffix(".json"), lambda t, _s=src.with_suffix(".json"): shutil.copyfile(_s, t), overwrite=True)
            new = exp.manifest.register("embeddings", key, dst, deps, {"fold": a.get("fold"), "dims": a.get("dims"), "imported_from": str(d),
                                                                        "imported_at": utc_now()})
            rows.append({"key": key, "action": "adopted" if new.get("sha256") == a.get("sha256") else "adopted (sha256 differs — check copy)",
                         "from": str(d)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(schema={"key": pl.String, "action": pl.String, "from": pl.String})


# ====================================================================== inventory
def _held_out_roles_ok(exp: Any, rec: dict[str, Any], fold: int) -> tuple[bool, str]:
    import polars as pl

    from ..evaluation.paired_inductive import verify_held_out_roles

    try:
        roles = pl.read_parquet(exp.manifest.resolve(rec), columns=["split_role"])["split_role"].to_list()
        verify_held_out_roles(roles, exp.splits(fold)[2], fold)
        return True, ""
    except Exception as e:                                          # noqa: BLE001 — reported, never silently accepted
        return False, f"{type(e).__name__}: {e}"


def checkpoint_loads(path: Path) -> tuple[bool, str]:
    try:
        from ..training.checkpoints import load_best

        model, _sig = load_best(path, "cpu")
        del model
        return True, ""
    except Exception as e:                                          # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def inventory(cfg: M13Config, exp: Any, models: Sequence[str], folds: Sequence[int], verify_load: bool = True) -> Any:
    """Model × fold readiness matrix of the M13 experiment (after adoption). Never trains or embeds."""
    import polars as pl

    from ..models import registry as reg

    fp, sh = exp.bundle.fingerprint, exp.cfg.science_hash()
    rows = []
    for m in models:
        n = reg.resolve(m)
        s = reg.spec(n)
        for f in folds:
            tag = f"inductive_f{f}"
            if not s.trainable:
                rows.append({"model": n, "fold": int(f), "training_regime": "fixed features (no training)", "checkpoint_found": None, "checkpoint_valid": None,
                             "science_hash": sh, "bundle_fingerprint": fp, "embedding_found": True, "embedding_roles_ok": True,
                             "evaluation_found": exp.manifest.find("evaluation_inductive", f"{n}__{tag}") is not None, "ready_for_transfer": True,
                             "imported_from": None, "reason": ""})
                continue
            tr = exp.manifest.find("training", f"{n}__{tag}")
            reasons = []
            valid = tr is not None
            if tr is None:
                reasons.append("no inductive checkpoint (not trained in M12 and not adopted)")
            else:
                d = tr.get("deps", {})
                p = exp.manifest.resolve(tr)
                if d.get("bundle") != fp or d.get("science") != sh:
                    valid = False
                    reasons.append("bundle / science mismatch")
                if d.get("regime") != INDUCTIVE or d.get("tag") != tag or (tr.get("fold") is not None and int(tr["fold"]) != int(f)):
                    valid = False
                    reasons.append("regime / fold identity mismatch")
                if not p.is_file():
                    valid = False
                    reasons.append("best.pt missing")
                else:
                    want, have = expected_architecture(exp, n), saved_architecture(p)
                    if want is not None and have is not None and want != have:
                        valid = False
                        reasons.append("architecture mismatch")
                    if verify_load and valid:
                        ok, msg = checkpoint_loads(p)
                        if not ok:
                            valid = False
                            reasons.append(f"does not load: {msg}")
            emb = exp.manifest.find("embeddings", f"{n}__{tag}")
            emb_ok, emb_msg = (_held_out_roles_ok(exp, emb, f) if emb is not None else (False, ""))
            if emb is not None and not emb_ok:
                reasons.append(f"embedding split roles inconsistent ({emb_msg})")
            ready = bool(valid and ((emb is not None and emb_ok) or (emb is None and cfg.allow_inference)))
            if valid and emb is None:
                reasons.append("embedding will be computed from the adopted checkpoint (deterministic inference)" if cfg.allow_inference
                               else "no embedding file and allow_inference=False")
            rows.append({"model": n, "fold": int(f), "training_regime": (tr or {}).get("deps", {}).get("regime") or "—", "checkpoint_found": tr is not None,
                         "checkpoint_valid": bool(valid), "science_hash": (tr or {}).get("deps", {}).get("science"),
                         "bundle_fingerprint": (tr or {}).get("deps", {}).get("bundle"), "embedding_found": emb is not None, "embedding_roles_ok": emb_ok,
                         "evaluation_found": exp.manifest.find("evaluation_inductive", f"{n}__{tag}") is not None, "ready_for_transfer": ready,
                         "imported_from": (tr or {}).get("imported_from"), "reason": "; ".join(reasons)})
    return pl.DataFrame(rows, infer_schema_length=None)


def ready_models(inv: Any, folds: Sequence[int]) -> tuple[list[str], dict[str, list[int]]]:
    """(encoders ready on EVERY fold, {encoder: missing folds}). The ladder needs all folds of an encoder; a partially
    available encoder is reported and excluded (never mixed with another regime)."""
    import polars as pl

    ready, missing = [], {}
    enc = inv.filter(pl.col("checkpoint_found").is_not_null())
    for m in enc["model"].unique(maintain_order=True).to_list():
        g = enc.filter(pl.col("model") == m)
        ok = set(g.filter(pl.col("ready_for_transfer"))["fold"].to_list())
        miss = sorted(set(int(f) for f in folds) - ok)
        if miss:
            missing[m] = miss
        else:
            ready.append(m)
    return ready, missing


# ====================================================================== M12 artifacts and reconciliation
def load_m12_tables(cfg: M13Config) -> tuple[dict[str, Any], Any]:
    """{family/name: frame} of the saved M12 lab tables (read-only) + a found / missing listing."""
    import polars as pl

    root = Path(cfg.m12_run_dir)
    out, rows = {}, []
    for fam, name in M12_TABLES:
        p = root / "tables" / fam / f"{name}.parquet"
        ok = p.is_file()
        if ok:
            try:
                out[f"{fam}/{name}"] = pl.read_parquet(p)
            except Exception as e:                                  # noqa: BLE001
                ok = False
                rows.append({"table": f"{fam}/{name}", "found": False, "rows": None, "path": str(p), "note": f"unreadable: {e}"})
                continue
        rows.append({"table": f"{fam}/{name}", "found": ok, "rows": out[f"{fam}/{name}"].height if ok else None, "path": str(p), "note": ""})
    ev = cfg.m12_experiment_dir / "evaluation" / "inductive_retrieval.csv"
    if ev.is_file():
        out["experiment/inductive_retrieval"] = pl.read_csv(ev, infer_schema_length=None)
    rows.append({"table": "experiment/inductive_retrieval", "found": ev.is_file(), "rows": out["experiment/inductive_retrieval"].height if ev.is_file() else None,
                 "path": str(ev), "note": "per (representation, held-out fold) inductive retrieval records"})
    return out, pl.DataFrame(rows, infer_schema_length=None)


def _mean_metric(geo: Any, rep: str, metric: str) -> float:
    import polars as pl

    if geo is None or not {"representation", "metric"} <= set(geo.columns):
        return float("nan")
    col = "mean" if "mean" in geo.columns else ("value" if "value" in geo.columns else None)
    if col is None:
        return float("nan")
    g = geo.filter((pl.col("representation") == rep) & (pl.col("metric") == metric))
    return float(g[col].cast(float).mean()) if g.height else float("nan")


def reconcile(m12: dict[str, Any], tol: dict[str, float] | None = None) -> Any:
    """Observed values in the SAVED M12 artifacts vs the values published in the reviewed M12 notebook. A difference is
    reported, never resolved by substituting the published number."""
    import polars as pl

    tol = {"auc": 0.0015, "rank": 0.15, "domain_auc": 0.003, "mmd2": 0.005, **(tol or {})}
    rows = []
    ir = m12.get("experiment/inductive_retrieval")
    for rep, reported in M12_REPORTED["inductive_retrieval_auc"].items():
        obs, n = float("nan"), 0
        if ir is not None and {"representation", "auc"} <= set(ir.columns):
            g = ir.filter(pl.col("representation") == rep)
            n = g.height
            obs = float(g["auc"].cast(float).mean()) if n else float("nan")
        rows.append({"quantity": "inductive retrieval AUC (mean over folds)", "model": rep, "observed_m12_artifact": obs, "reported_m12_summary": reported,
                     "n_records": n, "tolerance": tol["auc"], "source": "M12 experiment evaluation/inductive_retrieval.csv"})
    geo = m12.get("geometry/geometry_summary")
    for q, metric, key, t in (("effective rank (F4)", "effective_rank", "F4_effective_rank", tol["rank"]),
                              ("domain AUC grouped (F4)", "domain_auc_grouped", "F4_domain_auc", tol["domain_auc"]),
                              ("MMD² (F4)", "mmd2", "F4_mmd2", tol["mmd2"])):
        rows.append({"quantity": q, "model": "F4_physics_contrastive", "observed_m12_artifact": _mean_metric(geo, "F4_physics_contrastive", metric),
                     "reported_m12_summary": M12_REPORTED[key], "n_records": None, "tolerance": t, "source": "M12 tables/geometry/geometry_summary"})
    d = pl.DataFrame(rows, infer_schema_length=None).with_columns((pl.col("observed_m12_artifact") - pl.col("reported_m12_summary")).abs().alias("abs_diff"))
    return d.with_columns(pl.when(pl.col("observed_m12_artifact").is_nan()).then(pl.lit("artifact missing"))
                          .when(pl.col("abs_diff") <= pl.col("tolerance")).then(pl.lit("consistent")).otherwise(pl.lit("DIFFERS — investigate"))
                          .alias("status"))


# ====================================================================== F1 validity (Milestone 06 TCN)
def f1_validity(exp: Any) -> tuple[dict[str, Any], Any]:
    """Is the bundle's Milestone 06 TCN representation inductive w.r.t. the 12.0 outer folds? For each TCN fold f, no
    athlete of manifest fold f may carry a training role. Verdict: ``valid`` / ``leaky`` / ``unverifiable`` / ``absent``."""
    import polars as pl

    b = exp.bundle
    if b.tcn is None:
        return {"verdict": "absent", "detail": "bundle has no tcn_representations.parquet"}, pl.DataFrame()
    t = b.tcn
    if "family" in t.columns:
        t = t.filter(pl.col("family") == "TCN")
    if not {"fold", "nfl_id"} <= set(t.columns):
        return {"verdict": "unverifiable", "detail": "TCN table lacks fold / nfl_id"}, pl.DataFrame()
    fmap = exp.folds_by_athlete
    roles_present = "role" in t.columns
    vocab = sorted(str(v) for v in t["role"].unique().to_list()) if roles_present else []
    trainish = [v for v in vocab if "train" in v.lower() or "fit" in v.lower()]
    rows, leak_total = [], 0
    tf = sorted(int(v) for v in t["fold"].unique().to_list())
    for f in tf:
        g = t.filter(pl.col("fold") == f)
        held = {i for i, ff in fmap.items() if int(ff) == f}
        ids = set(int(i) for i in g["nfl_id"].unique().to_list())
        tr_ids = set(int(i) for i in g.filter(pl.col("role").cast(pl.String).is_in(trainish))["nfl_id"].unique().to_list()) if trainish else set()
        leak = len(held & tr_ids)
        leak_total += leak
        rows.append({"tcn_fold": f, "athletes_in_tcn_fold": len(ids), "manifest_fold_athletes": len(held), "held_out_with_training_role": leak,
                     "training_role_values": ",".join(trainish), "schemes": int(g["scheme"].n_unique()) if "scheme" in g.columns else None})
    manifest_folds = sorted(set(int(v) for v in fmap.values()))
    if not trainish:
        verdict, detail = "unverifiable", f"role vocabulary {vocab} has no training label: inductive status cannot be checked"
    elif tf != manifest_folds:
        verdict, detail = "leaky", f"TCN folds {tf} ≠ manifest folds {manifest_folds}"
    elif leak_total:
        verdict, detail = "leaky", f"{leak_total} held-out athlete(s) carry a training role in their own TCN fold"
    else:
        verdict, detail = "valid", "no held-out athlete has a training role in its own fold"
    return {"verdict": verdict, "detail": detail, "role_vocabulary": vocab, "tcn_folds": tf}, pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== cached sequence metrics (M13 → M12 → compute)
def sequence_metrics(exp: Any, run: Any, cfg: M13Config) -> tuple[Any, str]:
    """Per-sequence biomechanical metrics: the M13 cache, else the M12 cache when its provenance (bundle, metric version)
    matches (copied in, read-only on M12), else computed from the bundle."""
    import polars as pl

    from . import biomech

    prov = {"bundle": exp.bundle.fingerprint, "version": biomech.VERSION, "kind": "sequence_metrics"}
    hit = run.load_data("datasets/cached/sequence_metrics", prov)
    if hit is not None:
        return hit, "reused M13 cache"
    p = Path(cfg.m12_run_dir) / "datasets" / "cached" / "sequence_metrics.parquet"
    side = p.with_suffix(".provenance.json")
    if p.is_file() and side.is_file():
        try:
            have = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            have = {}
        if all(have.get(k) == v for k, v in prov.items()):
            df = pl.read_parquet(p)
            if df.height == exp.bundle.n_sequences:
                run.data_file(df, "datasets/cached/sequence_metrics", {**prov, "copied_from": str(p)})
                return df, "copied from the M12 cache (provenance matched)"
    return biomech.sequence_table(exp, run, reuse=False), "computed from the bundle"


# ====================================================================== prerequisites
def prerequisites(exp: Any, inv: Any, ready: Sequence[str], missing: dict[str, list[int]], f1: dict[str, Any], m12_found: Any) -> Any:
    import polars as pl

    from ..data.validation import M1_COLS, TRANSFER_TARGET

    b = exp.bundle
    ath = b.athletes
    rows = []

    def add(req: str, ok: bool | None, why: str, consequence: str, detail: str = "") -> None:
        rows.append({"requirement": req, "status": "pass" if ok else ("not applicable" if ok is None else "MISSING"), "why_required": why,
                     "consequence_if_missing": consequence, "detail": detail})

    add("bundle fingerprint = declared", b.fingerprint == EXPECTED["fingerprint"], "M12 encoders and folds are bound to this bundle",
        "nothing can be reused; stop", b.fingerprint)
    add("science hash = declared", exp.cfg.science_hash() == EXPECTED["science_hash"], "checkpoint adoption requires the identical science config",
        "encoders unusable; stop", exp.cfg.science_hash())
    add("athletes.parquet present", ath is not None, "holds positions, M1 officials and the frozen target", "no transfer analysis (Parts B–J blocked)")
    add(f"frozen target {TRANSFER_TARGET}", ath is not None and TRANSFER_TARGET in ath.columns, "the single declared outcome of every comparison",
        "Parts B–J blocked")
    add("M1 official columns", ath is not None and all(c in ath.columns for c in M1_COLS), "M1 is the reference of every Δ RMSE", "ladder unavailable",
        ", ".join(c for c in M1_COLS if ath is None or c not in ath.columns))
    add("fold manifest", b.folds.height > 0, "athlete-disjoint outer folds", "no OOF evaluation")
    add("≥ 1 encoder ready on all folds", len(ready) > 0, "learned-representation rungs of the ladder (M1 + F2…F6)",
        "only Position / M1 / M1+F0 / biomechanics rungs run", ", ".join(ready) or "none")
    for m, f in missing.items():
        add(f"{m}: inductive checkpoints on every fold", False, "the ladder fits the fold-f block from the encoder trained without fold f",
            f"{m} excluded from the ladder (no transductive fallback)", f"missing folds {f}")
    add("F1 TCN fold validity", None if f1.get("verdict") == "absent" else f1.get("verdict") == "valid",
        "F1 enters the primary ladder only when it is inductive w.r.t. the outer folds", "F1 excluded from primary comparisons", f1.get("detail", ""))
    found = int(m12_found.filter(pl.col("found")).height) if m12_found is not None and m12_found.height else 0
    add("M12 saved tables (reconciliation)", found > 0, "A3 reconciliation and Part H retrieval / geometry records", "A3 / H use only M13-computed records",
        f"{found} of {m12_found.height if m12_found is not None else 0} found")
    return pl.DataFrame(rows, infer_schema_length=None)


def handoff_record(run: Any, cfg: M13Config, exp: Any, env: dict[str, Any], inv: Any, ready: Sequence[str]) -> dict[str, Any]:
    rec = {"written_at": utc_now(), "bundle_fingerprint": exp.bundle.fingerprint, "science_hash": exp.cfg.science_hash(), "science_seed": exp.cfg.science.seed,
           "m13_seed": cfg.seed, "precision": cfg.precision, "gpu": env.get("gpu"), "revision": env.get("revision"), "code_version": code_version(),
           "m12_experiment_dir": str(cfg.m12_experiment_dir), "m13_experiment_dir": str(exp.root), "ready_encoders": list(ready),
           "inventory_rows": int(inv.height) if inv is not None else 0}
    atomic_json(rec, run.root / "manifests" / "handoff.json", overwrite=True)
    return rec


__all__ = ["M12_TABLES", "adopt", "adopt_embeddings", "checkpoint_loads", "discover", "environment", "expected_architecture", "f1_validity", "handoff_record",
           "installed_revision", "inventory", "load_m12_tables", "open_experiment", "prerequisites", "ready_models", "reconcile", "saved_architecture",
           "sequence_metrics", "source_dirs"]
