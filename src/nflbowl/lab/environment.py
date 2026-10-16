"""Setup of the lab: environment record, bundle validation against the declared provenance, the nflbowl
``Experiment`` of the lab, precision lock, discovery / adoption of compatible checkpoints from earlier experiments,
the training plan, and the transductive reference records.

Earlier experiments are only READ (their JSON manifests and files). Adoption copies a ``best.pt`` into the lab
experiment and registers it with the source's own dependency record, and only when the bundle fingerprint AND the
science hash are identical — the same rule ``Experiment._training_record`` enforces before any embedding.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from ..artifacts.store import atomic_json, atomic_write
from ..utils.reproducibility import code_version, utc_now
from .config import EXPECTED_BUNDLE, LabConfig

INDUCTIVE = "inductive"


# ====================================================================== environment
def environment_record(cfg: LabConfig) -> dict[str, Any]:
    from ..research import colab

    info = colab.runtime_report(verbose=False)
    gpu = colab.gpu_memory_summary()
    try:
        fix = colab.fixed_losses_check()
    except Exception as e:                                          # noqa: BLE001
        fix = {"ok": None, "error": repr(e)}
    return {"recorded_at": utc_now(), "code_version": code_version(), "git_ref_requested": cfg.git_ref, "precision": cfg.precision, "device": cfg.device,
            "gpu": gpu, "runtime": {k: v for k, v in info.items() if k != "versions"}, "versions": info.get("versions", {}), "fp16_loss_fix": fix}


def setup_summary(env: dict[str, Any], bundle_checks: Any, exp: Any, cfg: LabConfig) -> Any:
    """One visible table: what the session runs on and with what."""
    import polars as pl

    gpu = env.get("gpu", {})
    ok = bundle_checks.filter(pl.col("status") == "fail").height == 0 if bundle_checks is not None and bundle_checks.height else None
    rows = [("nflbowl", str(env.get("versions", {}).get("nflbowl"))), ("torch", str(env.get("versions", {}).get("torch"))),
            ("GPU", f"{gpu.get('device')} · free {gpu.get('free_gb')} / {gpu.get('total_gb')} GB" if gpu.get("cuda") else "none (CPU)"),
            ("precision", cfg.precision), ("fp16 loss fix present", str(env.get("fp16_loss_fix", {}).get("ok"))),
            ("run mode", cfg.run_mode), ("bundle checks", "all pass" if ok else ("FAILURES — see table" if ok is False else "not run")),
            ("experiment", f"{exp.id} · science {exp.cfg.science_hash()}" if exp is not None else "not opened"),
            ("lab seed / science seed", f"{cfg.seed} / {exp.cfg.science.seed if exp is not None else '—'}"),
            ("training enabled", str(cfg.run_training)), ("transfer enabled", str(cfg.run_transfer)), ("ablations enabled", str(cfg.run_ablations))]
    return pl.DataFrame({"item": [r[0] for r in rows], "value": [r[1] for r in rows]})


# ====================================================================== bundle
def validate_bundle(cfg: LabConfig) -> tuple[Any, dict[str, Any]]:
    """(checks table, bundle summary). Opens the cleaned bundle READ-ONLY (staged copy when ``stage_dir`` is set) and
    compares it with the declared provenance. Never rebuilds or modifies the bundle."""
    import polars as pl

    from ..data.loading import open_bundle

    src = Path(cfg.bundle_dir)
    b = open_bundle(src, cfg.stage_dir or None, verify=cfg.verify_bundle_hashes, outer_folds=EXPECTED_BUNDLE["folds"])
    s = b.summary()
    complete = (Path(b.root) / "sequence_store" / "COMPLETE.json")
    comp = json.loads(complete.read_text(encoding="utf-8")) if complete.is_file() else {}
    n_folds = int(b.folds["fold"].n_unique())
    rows = [("fingerprint", cfg.expected_fingerprint, s["fingerprint"]), ("n_sequences", EXPECTED_BUNDLE["n_sequences"], s["n_sequences"]),
            ("frames", EXPECTED_BUNDLE["frames"], s["frames"]), ("channels", EXPECTED_BUNDLE["channels"], s["channels"]),
            ("n_athletes", EXPECTED_BUNDLE["n_athletes"], s["n_athletes"]), ("outer folds in folds.parquet", EXPECTED_BUNDLE["folds"], n_folds),
            ("zero-frame sequences", 0, s["valid_frames"]["n_empty"]), ("athletes.parquet with frozen target", True, s["has_transfer_target"]),
            ("COMPLETE.json present", True, complete.is_file())]
    out = []
    for name, exp_v, obs in rows:
        status = "info" if exp_v is None else ("pass" if exp_v == obs else "fail")
        out.append({"check": name, "expected": str(exp_v), "observed": str(obs), "status": status})
    out.append({"check": "athletes with both domains (declared ≈399)", "expected": "≈399", "observed": str(s["n_athletes_both_domains"]),
                "status": "pass" if abs(s["n_athletes_both_domains"] - 399) <= 5 else "warn"})
    out.append({"check": "COMPLETE.json n (original store, before cleaning)", "expected": "info", "observed": str(comp.get("n")), "status": "info"})
    out.append({"check": "sha256 integrity", "expected": "info", "observed": s["integrity"], "status": "info"})
    return pl.DataFrame(out), s


# ====================================================================== the lab experiment
def open_experiment(cfg: LabConfig, run: Any) -> Any:
    """Create or reopen the nflbowl Experiment at ``RUN_DIR/checkpoints/experiments/<experiment_name>`` (same science as
    the base preset unless ``science_seed`` is set)."""
    from ..api import Experiment

    exp = Experiment(data_dir=cfg.bundle_dir, output_dir=str(run.root / "checkpoints"), config=cfg.base_config, name=cfg.experiment_name,
                     overrides=cfg.experiment_overrides())
    exp.prepare(verify=False)
    if cfg.expected_fingerprint and exp.bundle.fingerprint != cfg.expected_fingerprint:
        raise RuntimeError(f"bundle fingerprint {exp.bundle.fingerprint} ≠ expected {cfg.expected_fingerprint}: refusing to run on other data")
    return exp


def precision_lock(run: Any, cfg: LabConfig, exp: Any) -> dict[str, Any]:
    """The first session that may train writes ``config/precision_lock.json``; a later session with another precision
    is refused while training records exist (precision is never switched mid-experiment)."""
    p = run.root / "config" / "precision_lock.json"
    trained = len(exp.manifest.artifacts("training"))
    if p.is_file():
        lock = json.loads(p.read_text(encoding="utf-8"))
        if lock.get("precision") != cfg.precision and trained:
            raise RuntimeError(f"this lab was locked to precision {lock.get('precision')!r} ({trained} training records exist); "
                               f"CFG precision {cfg.precision!r} would mix numerics. Use a new experiment_name or restore the precision.")
        return lock
    lock = {"precision": cfg.precision, "locked_at": utc_now(), "experiment": exp.id, "science_hash": exp.cfg.science_hash()}
    atomic_json(lock, p, overwrite=False)
    return lock


# ====================================================================== earlier experiments (read-only)
def candidate_experiment_dirs(cfg: LabConfig) -> list[Path]:
    root = Path(cfg.drive_root)
    out = [root / "experiments" / n for n in cfg.reference_experiments]
    if cfg.search_previous_runs:
        out += sorted(root.glob("experiments/*"))
        out += sorted(root.glob("unified_research/*/checkpoints/experiments/*"))
        out += sorted(p for p in root.glob("experimental_lab/*/checkpoints/experiments/*") if p.parent.parent.parent.name != cfg.experiment_name)
    seen, uniq = set(), []
    for p in out:
        k = str(p)
        if k not in seen and (p / "config" / "experiment.json").is_file():
            seen.add(k)
            uniq.append(p)
    return uniq


def discover_sources(cfg: LabConfig, exp: Any) -> Any:
    """Every training record of earlier experiments with its compatibility verdict for the lab experiment."""
    import polars as pl

    fp, sh = exp.bundle.fingerprint, exp.cfg.science_hash()
    rows = []
    for d in candidate_experiment_dirs(cfg):
        if d.resolve() == exp.root.resolve():
            continue
        try:
            saved = json.loads((d / "config" / "experiment.json").read_text(encoding="utf-8"))
            man = json.loads((d / "manifests" / "manifest.json").read_text(encoding="utf-8")) if (d / "manifests" / "manifest.json").is_file() else {"artifacts": {}}
        except (OSError, json.JSONDecodeError) as e:
            rows.append({"source": str(d), "key": None, "status": f"unreadable: {e}"})
            continue
        for a in man.get("artifacts", {}).values():
            if a.get("kind") != "training":
                continue
            deps = a.get("deps", {})
            path = Path(a["path"]) if Path(a["path"]).is_absolute() else d / a["path"]
            reasons = []
            if deps.get("bundle") != fp:
                reasons.append(f"bundle {deps.get('bundle')}")
            if deps.get("science") != sh:
                reasons.append(f"science {deps.get('science')}")
            if not path.is_file():
                reasons.append("best.pt missing")
            rows.append({"source": str(d), "experiment": saved.get("experiment_id"), "key": a["key"], "model": a["key"].split("__")[0],
                         "tag": deps.get("tag"), "regime": deps.get("regime"), "fold": a.get("fold"), "path": str(path), "sha256": a.get("sha256"),
                         "created_at": a.get("created_at"), "precision": (a.get("signature") or {}).get("precision"),
                         "status": "compatible" if not reasons else "incompatible: " + ", ".join(reasons)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(schema={"source": pl.String, "key": pl.String, "status": pl.String})


def adopt_checkpoints(exp: Any, sources: Any, models: Sequence[str] | None = None, regime: str = INDUCTIVE) -> Any:
    """Copy compatible best checkpoints the lab experiment does not have yet (newest compatible source first) and
    register them under the source's dependency record. Sources are never modified."""
    import polars as pl

    from ..models import registry as reg

    if sources is None or not sources.height or "model" not in sources.columns:
        return pl.DataFrame(schema={"key": pl.String, "action": pl.String})
    want = {reg.resolve(m) for m in models} if models else None
    s = sources.filter(pl.col("status") == "compatible")
    if regime:
        s = s.filter(pl.col("regime") == regime)
    out = []
    for key in sorted(set(s["key"].to_list())):
        name = key.split("__")[0]
        if want is not None and name not in want:
            continue
        if exp.manifest.find("training", key) is not None:
            out.append({"key": key, "action": "already present"})
            continue
        cand = s.filter(pl.col("key") == key).sort("created_at", descending=True).row(0, named=True)
        src_dir = Path(cand["source"])
        man = json.loads((src_dir / "manifests" / "manifest.json").read_text(encoding="utf-8"))
        rec = next(a for a in man["artifacts"].values() if a["kind"] == "training" and a["key"] == key and a.get("sha256") == cand["sha256"])
        tag = rec["deps"]["tag"]
        dst = exp.store.checkpoints / name / tag / "best.pt"
        src = Path(cand["path"])
        atomic_write(dst, lambda t, _s=src: shutil.copyfile(_s, t), overwrite=True)
        for extra in (src.with_suffix(".json"), src.parent / "curve.parquet"):
            if extra.is_file():
                atomic_write(dst.parent / extra.name, lambda t, _s=extra: shutil.copyfile(_s, t), overwrite=True)
        keep = {k: v for k, v in rec.items() if k not in ("id", "kind", "key", "path", "deps", "deps_hash", "created_at", "bytes", "sha256")}
        keep.update({"imported_from": str(src_dir), "imported_at": utc_now(), "curve": exp.store.rel(dst.parent / "curve.parquet")})
        new = exp.manifest.register("training", key, dst, rec["deps"], keep)
        out.append({"key": key, "action": "adopted" if new.get("sha256") == rec.get("sha256") else "adopted (sha256 differs — check copy)",
                    "from": str(src_dir)})
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame(schema={"key": pl.String, "action": pl.String})


# ====================================================================== training plan
def inductive_status(exp: Any, models: Sequence[str], folds: Sequence[int]) -> Any:
    """(model, fold) → trained? embedded? Without training anything."""
    import polars as pl

    from ..models import registry as reg

    rows = []
    for m in models:
        n = reg.resolve(m)
        s = reg.spec(n)
        for f in folds:
            tag = f"inductive_f{f}"
            tr = exp.manifest.find("training", f"{n}__{tag}") if s.trainable else None
            em = exp.manifest.find("embeddings", f"{n}__{tag}")
            rows.append({"model": n, "fold": int(f), "trainable": s.trainable, "trained": (tr is not None) if s.trainable else None,
                         "imported_from": (tr or {}).get("imported_from"), "embedded": em is not None,
                         "precision": ((tr or {}).get("signature") or {}).get("precision")})
    return pl.DataFrame(rows, infer_schema_length=None)


def train_missing(exp: Any, models: Sequence[str], folds: Sequence[int], resume: bool = True) -> Any:
    """Train (or resume) only the (model, fold) inductive encoders that have no training record."""
    import polars as pl

    from ..models import registry as reg

    out = []
    for m in models:
        n = reg.resolve(m)
        if not reg.spec(n).trainable:
            continue
        todo = [f for f in folds if exp.manifest.find("training", f"{n}__inductive_f{f}") is None]
        if todo:
            out.append(exp.train(n, regime=INDUCTIVE, resume=resume, folds=todo))
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def available_pairs(exp: Any, models: Sequence[str], folds: Sequence[int]) -> dict[str, list[int]]:
    """{model: folds with a usable inductive representation} (F0 / random: every fold; encoders: trained folds)."""
    from ..models import registry as reg

    out: dict[str, list[int]] = {}
    for m in models:
        n = reg.resolve(m)
        s = reg.spec(n)
        if s.kind == "external":
            continue
        out[n] = [int(f) for f in folds if (not s.trainable) or exp.manifest.find("training", f"{n}__inductive_f{f}") is not None]
    return out


# ====================================================================== transductive reference
def transductive_reference(cfg: LabConfig, exp: Any) -> Any:
    """Saved transductive metrics of the reference experiments (read-only), with comparability flags."""
    import polars as pl

    rows = []
    for d in candidate_experiment_dirs(cfg):
        f = d / "evaluation" / "metrics.csv"
        if not f.is_file():
            continue
        saved = json.loads((d / "config" / "experiment.json").read_text(encoding="utf-8"))
        m = pl.read_csv(f, infer_schema_length=None)
        same_bundle = saved.get("bundle_fingerprint") == exp.bundle.fingerprint
        rows.append(m.with_columns(pl.lit(d.name).alias("source_experiment"), pl.lit(str(saved.get("bundle_fingerprint"))).alias("source_bundle"),
                                   pl.lit(str(saved.get("science_hash"))).alias("source_science"), pl.lit(same_bundle).alias("same_bundle"),
                                   pl.lit(saved.get("science_hash") == exp.cfg.science_hash()).alias("same_science")))
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows, how="diagonal_relaxed")


__all__ = ["adopt_checkpoints", "available_pairs", "candidate_experiment_dirs", "discover_sources", "environment_record", "inductive_status",
           "open_experiment", "precision_lock", "setup_summary", "train_missing", "transductive_reference", "validate_bundle"]
