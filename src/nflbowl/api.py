"""``Experiment`` — the high-level API.

Phases are independent and artifact-backed; each can run in a fresh session from what the previous ones saved:

    prepare()   open / stage / validate the data bundle; derive per-sequence targets and F0 features (cached Parquet)
    train()     train or resume one encoder (transductive, inductive per fold, or learning-curve fraction)
    embed()     embeddings from the saved best weights (or a baseline) → Parquet with ids, domain, fold, split role
    evaluate()  quick / standard / full label-free evaluation → immutable JSON records + CSV views
    transfer()  inductive NFL-transfer ladder on the 12C frozen target (needs athletes.parquet with the target)
    plot() · report() · export() · compare() · list_models() · list_artifacts() · resume()

An experiment directory is bound to ONE data bundle (fingerprint) and ONE science configuration (hash). Runtime
settings (device, precision, placement, batch sizes for embedding) may change between sessions and are logged.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import _backend
from .artifacts.manifest import Manifest, deps_hash
from .artifacts.store import ExperimentStore, append_jsonl, atomic_json, atomic_text, atomic_write, sha256_file
from .config import ConfigError, ExperimentConfig, load_config
from .data import splits as sp
from .devices import apply_backend_flags, cleanup, device_info, resolve_device
from .evaluation.evaluator import evaluate_representation, inductive_fold_retrieval
from .models import registry as reg
from .utils.logging import attach_file_log, get_logger
from .utils.reproducibility import code_version, seed_everything, utc_now

log = get_logger(__name__)
TRANSDUCTIVE, INDUCTIVE = "transductive", "inductive"
_ID = re.compile(r"^experiment_(\d{3,})$")


def _tag(regime: str, nfl_fraction: float = 1.0, fold: int | None = None) -> str:
    if regime == INDUCTIVE:
        return f"inductive_f{fold}"
    return TRANSDUCTIVE if nfl_fraction >= 1.0 else f"lc{int(round(nfl_fraction * 100))}"


def _rep_name(model: str, nfl_fraction: float = 1.0) -> str:
    return model if nfl_fraction >= 1.0 else f"{model}@nfl{int(round(nfl_fraction * 100))}%"     # 12F notebook naming


def _md_table(df: Any, cols: Sequence[str] | None = None, digits: int = 3) -> str:
    if df is None or not getattr(df, "height", 0):
        return "_none_\n"
    cols = [c for c in (cols or df.columns) if c in df.columns]

    def f(v: Any) -> str:
        if isinstance(v, float):
            return "—" if not math.isfinite(v) else f"{v:.{digits}f}"
        return "—" if v is None else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(r[c]) for c in cols) + " |" for r in df.select(cols).iter_rows(named=True)]
    return "\n".join(lines) + "\n"


class Experiment:
    """One experiment directory under ``<output_dir>/experiments/<name>``.

    ``name=None`` creates the next free ``experiment_NNN``; an existing name is REOPENED (science config must match),
    which is what you want after a Colab runtime reset. ``Experiment.open(path)`` reopens from the directory alone.
    """

    def __init__(self, data_dir: str | Path | None = None, output_dir: str | Path = ".", device: str | None = None, config: Any = "default",
                 seed: int | None = None, name: str | None = None, overrides: dict[str, Any] | None = None, _root: Path | None = None):
        ov = dict(overrides or {})
        if device is not None:
            ov["runtime.device"] = device
        if seed is not None:
            ov["science.seed"] = int(seed)
        cfg = load_config(config, ov)
        root = Path(_root) if _root is not None else self._choose_root(Path(output_dir) / "experiments", name)
        self.store = ExperimentStore(root)
        if self.store.exists():
            saved = json.loads((root / "config" / "experiment.json").read_text(encoding="utf-8"))
            if saved["science_hash"] != cfg.science_hash():
                raise ConfigError(f"experiment {root.name} was created with science config {saved['science_hash']}; the requested config hashes to "
                                  f"{cfg.science_hash()}. Reopen it with Experiment.open(path) (keeps its science config) or use a new name.")
            data_dir = data_dir or saved.get("data_dir")
            self._saved = saved
        else:
            self.store.create()
            self._saved = {"experiment_id": root.name, "created_at": utc_now(), "config": cfg.to_dict(), "science_hash": cfg.science_hash(),
                           "data_dir": str(data_dir) if data_dir else None, "bundle_fingerprint": None, "code_version": code_version()}
            atomic_json(self._saved, root / "config" / "experiment.json", overwrite=True)
        self.cfg: ExperimentConfig = cfg
        self.data_dir = Path(data_dir) if data_dir else None
        self.id = root.name
        self.manifest = Manifest(root, self.id)
        attach_file_log(self.store.logs / "experiment.log")
        append_jsonl({"at": utc_now(), "event": "session", "runtime": cfg.to_dict()["runtime"], "evaluation": cfg.to_dict()["evaluation"],
                      "code_version": code_version()}, root / "config" / "runtime_history.jsonl")
        seed_everything(cfg.science.seed)
        self._bundle = None
        self._derived = None
        self._E0 = None
        self._folds: dict[int, int] | None = None
        self._device: str | None = None
        self._data = None
        log.info("experiment %s at %s (science %s, preset %s)", self.id, root, cfg.science_hash(), cfg.preset)

    # ================================================================== construction helpers
    @staticmethod
    def _choose_root(base: Path, name: str | None) -> Path:
        if name:
            if not re.match(r"^[A-Za-z0-9_.-]+$", name):
                raise ValueError("experiment name may only contain letters, digits, '_', '-', '.'")
            return base / name
        nums = [int(m.group(1)) for p in (base.iterdir() if base.is_dir() else []) if (m := _ID.match(p.name))]
        return base / f"experiment_{(max(nums) + 1) if nums else 1:03d}"

    @classmethod
    def open(cls, path: str | Path, data_dir: str | Path | None = None, device: str | None = None, runtime: dict[str, Any] | None = None,
             evaluation: dict[str, Any] | None = None) -> "Experiment":
        """Reopen an experiment directory with ITS saved science configuration. ``device`` / ``runtime`` / ``evaluation``
        override session settings only (e.g. ``runtime={"stage_dir": "/content/stage"}``)."""
        root = Path(path)
        f = root / "config" / "experiment.json"
        if not f.is_file():
            raise FileNotFoundError(f"{root} is not an nflbowl experiment (missing config/experiment.json)")
        saved = json.loads(f.read_text(encoding="utf-8"))
        raw = dict(saved["config"])
        ov: dict[str, Any] = {f"runtime.{k}": v for k, v in (runtime or {}).items()}
        ov.update({f"evaluation.{k}": v for k, v in (evaluation or {}).items()})
        return cls(data_dir=data_dir, device=device, config=raw, overrides=ov, _root=root)

    @staticmethod
    def device_info(verbose: bool = True) -> dict[str, Any]:
        return device_info(verbose)

    @property
    def root(self) -> Path:
        return self.store.root

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self.cfg.runtime.device)
            flags = apply_backend_flags(self.cfg.runtime)
            append_jsonl({"at": utc_now(), "event": "device", "device": self._device, "flags": flags}, self.root / "config" / "runtime_history.jsonl")
        return self._device

    # ================================================================== prepare / features
    def prepare(self, verify: bool = False, reuse: bool = True) -> dict[str, Any]:
        """Open (stage) and validate the bundle, derive per-sequence targets, compute F0 features. Idempotent."""
        from .data.loading import open_bundle
        from .data.preprocessing import derive
        from .features.engineering import engineered_cached

        if self.data_dir is None:
            raise ValueError("data_dir is not set: pass Experiment(data_dir=...) or Experiment.open(path, data_dir=...)")
        b = open_bundle(self.data_dir, self.cfg.runtime.stage_dir, verify=verify, outer_folds=self.cfg.science.outer_folds)
        fp = self._saved.get("bundle_fingerprint")
        if fp and fp != b.fingerprint:
            raise ConfigError(f"experiment {self.id} is bound to bundle {fp}, but {self.data_dir} is bundle {b.fingerprint}. Use a new experiment "
                              f"for different data (artifacts would otherwise mix inputs).")
        if not fp:
            self._saved.update({"bundle_fingerprint": b.fingerprint, "data_dir": str(self.data_dir), "bundle_info": {k: v for k, v in b.info.items() if k != "files"}})
            atomic_json(self._saved, self.root / "config" / "experiment.json", overwrite=True)
        self._bundle = b
        self._folds = sp.fold_map(b.meta, b.folds, self.cfg.science.outer_folds)
        self._derived, dinfo = derive(b, self.manifest, self.store.processed, reuse)
        self._E0, finfo = engineered_cached(b, self.manifest, self.store.features, reuse)
        meta_path = self.store.processed / f"sequence_meta__{b.fingerprint}.parquet"
        if not meta_path.is_file():
            fold_col = [self._folds.get(int(i), -1) for i in b.meta["nfl_id"].to_list()]
            import polars as pl

            self.store.write_parquet(b.meta.with_columns(pl.Series("fold", fold_col, dtype=pl.Int64)), meta_path)
        summary = {**b.summary(), "derived": dinfo, "features": finfo, "folds": sorted(set(self._folds.values()))}
        atomic_json(summary, self.store.processed / "bundle_summary.json", overwrite=True)
        log.info("prepared: %s sequences (%s Combine, %s NFL), %s athletes, bundle %s", f"{summary['n_sequences']:,}", f"{summary['n_combine']:,}",
                 f"{summary['n_nfl']:,}", summary["n_athletes"], b.fingerprint)
        return summary

    def _require(self) -> Any:
        if self._bundle is None:
            self.prepare()
        return self._bundle

    def features(self, reuse: bool = True) -> np.ndarray:
        """F0 engineered features (N × 17), cached under data/features/."""
        self._require()
        return self._E0

    def _device_data(self) -> Any:
        from .training.trainer import DeviceData

        if self._data is None:
            b, d = self._require(), self._derived
            self._data = DeviceData(b.X, b.M, d.state, d.cat, d.dom, self.device, self.cfg.runtime.data_placement, self.cfg.runtime.gpu_memory_fraction)
        return self._data

    def _deps(self, **extra: Any) -> dict[str, Any]:
        return {"bundle": self._require().fingerprint, "science": self.cfg.science_hash(), **extra}

    # ================================================================== training
    def list_models(self) -> Any:
        """Registry (families, ablations, baselines) with what this experiment already holds for each."""
        import polars as pl

        t = reg.list_models()
        trained = {a["key"].split("__")[0] for a in self.manifest.artifacts("training")}
        embedded = {a["key"].split("__")[0] for a in self.manifest.artifacts("embeddings")}
        evaluated = {a["key"].split("__")[0] for a in self.manifest.artifacts("evaluation")}
        return t.with_columns(pl.col("name").is_in(list(trained)).alias("trained"), pl.col("name").is_in(list(embedded)).alias("embedded"),
                              pl.col("name").is_in(list(evaluated)).alias("evaluated"))

    def _splits(self, held_out: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = self._require()
        return sp.split_indices(b.meta, self._folds, held_out, self.cfg.science.val_share, self.cfg.science.seed)

    def _fold_list(self, folds: Iterable[int] | None) -> list[int]:
        self._require()
        all_f = sorted(set(self._folds.values()))
        return all_f if folds is None else [int(f) for f in folds]

    def train(self, model: str, regime: str = TRANSDUCTIVE, resume: bool = True, nfl_fraction: float = 1.0, folds: Iterable[int] | None = None) -> Any:
        """Train (or resume / restore) one encoder. ``regime='inductive'`` trains one encoder per outer fold without that
        fold's athletes (both domains). ``nfl_fraction < 1`` = learning-curve run (transductive only). Returns a polars
        summary table (one row per trained run)."""
        import polars as pl

        from .training.checkpoints import best_path
        from .training.trainer import train_encoder

        name = reg.resolve(model)
        s = reg.spec(name)
        if not s.trainable:
            raise ValueError(f"{name} is a {s.kind} ({s.description}); nothing to train — call embed('{model}') / evaluate('{model}')")
        if regime not in (TRANSDUCTIVE, INDUCTIVE):
            raise ValueError("regime must be 'transductive' or 'inductive'")
        if regime == INDUCTIVE and nfl_fraction < 1.0:
            raise ValueError("learning-curve fractions are transductive diagnostics (12F Part 7a)")
        sci = self.cfg.science
        e = reg.enc_config(name, sci)
        data = self._device_data()
        P = self._derived.P
        runs = [(None, _tag(regime, nfl_fraction))] if regime == TRANSDUCTIVE else [(f, _tag(regime, fold=f)) for f in self._fold_list(folds)]
        rows = []
        for held, tag in runs:
            tr, va, ho = self._splits(held)
            job = f"train:{name}:{tag}"
            self.manifest.set_job(job, status="running", model=name, regime=regime, tag=tag, nfl_fraction=nfl_fraction, fold=held, device=self.device)
            res = train_encoder(e, data, P, tr, va, sci, self.cfg.runtime, self.store.checkpoints, tag, sci.seed, nfl_fraction,
                                best_file=best_path(self.store.checkpoints, name, tag), resume=resume)
            curve = res.curve.with_columns(pl.lit(name).alias("model"), pl.lit(tag).alias("tag")) if res.curve.height else res.curve
            cpath = self.store.checkpoints / name / tag / "curve.parquet"
            self.store.write_parquet(curve, cpath, overwrite=True)
            deps = self._deps(checkpoint=Path(res.checkpoint_dir).name, regime=regime, tag=tag, nfl_fraction=nfl_fraction)   # <mode>_<config hash>
            if res.best_path:
                self.manifest.register("training", f"{name}__{tag}", res.best_path, deps,
                                       {"curve": self.store.rel(cpath), "checkpoint_dir": res.checkpoint_dir, "signature": res.signature, "fold": held})
            self.manifest.set_job(job, status="finished", best=res.best_path, epochs_run=len(res.curve), resumed_from=res.resumed_from_epoch,
                                  rng_restored=res.rng_restored)
            rows.append({"model": name, "tag": tag, "regime": regime, "fold": held, "epochs_run": res.curve.height,
                         "best_val": res.signature.get("best_val"), "restored": res.restored_finished, "best_path": res.best_path})
            del res
            cleanup(self.device)
        self._rebuild_curves()
        return pl.DataFrame(rows, infer_schema_length=None)

    def train_all(self, families: bool = True, ablations: bool = False, regime: str = TRANSDUCTIVE, resume: bool = True,
                  models: Sequence[str] | None = None) -> Any:
        """Train several encoders sequentially (never all by default: ablations are opt-in)."""
        import polars as pl

        names = list(models) if models else ([*reg.families()] if families else []) + ([*reg.ablations()] if ablations else [])
        if regime == INDUCTIVE and models is None:
            names = list(self.cfg.science.inductive_configs)
        out = [self.train(n, regime=regime, resume=resume) for n in names]
        return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()

    def resume(self) -> Any:
        """Continue every training job left ``running`` (e.g. by a Colab disconnect)."""
        import polars as pl

        pending = self.manifest.jobs("running")
        if not pending:
            log.info("no unfinished training jobs")
            return pl.DataFrame()
        out = []
        for j in pending:
            folds = [j["fold"]] if j.get("fold") is not None else None
            out.append(self.train(j["model"], regime=j["regime"], resume=True, nfl_fraction=j.get("nfl_fraction", 1.0), folds=folds))
        return pl.concat(out, how="diagonal_relaxed")

    def _rebuild_curves(self) -> None:
        import polars as pl

        parts = [pl.read_parquet(p) for p in sorted(self.store.checkpoints.glob("*/*/curve.parquet"))]
        parts = [p for p in parts if p.height]
        if parts:
            self.store.write_csv(pl.concat(parts, how="diagonal_relaxed"), self.store.evaluation / "training_curves.csv")

    # ================================================================== embeddings
    def _training_record(self, name: str, tag: str) -> dict[str, Any]:
        rec = self.manifest.find("training", f"{name}__{tag}")
        if rec is None:
            raise FileNotFoundError(f"{name} ({tag}) has not been trained in experiment {self.id}: call train('{name}'"
                                    + (", regime='inductive')" if tag.startswith("inductive") else ")") + " first")
        d = rec["deps"]
        if d.get("bundle") != self._require().fingerprint or d.get("science") != self.cfg.science_hash():
            raise ConfigError(f"the {name} ({tag}) checkpoint was trained on other inputs (bundle / science hash differ); retrain it")
        return rec

    def embed(self, model: str, regime: str = TRANSDUCTIVE, reuse: bool = True, nfl_fraction: float = 1.0, folds: Iterable[int] | None = None) -> Any:
        """Embeddings of every sequence. Transductive → ``np.ndarray``; inductive → ``{fold: np.ndarray}``. Saved to
        ``embeddings/<model>__<tag>__<hash>.parquet`` with ids, domain, kind, position, fold and split role."""
        name = reg.resolve(model)
        if regime == INDUCTIVE:
            return {f: self._embed_one(name, _tag(INDUCTIVE, fold=f), f, reuse) for f in self._fold_list(folds)}
        return self._embed_one(name, _tag(TRANSDUCTIVE, nfl_fraction), None, reuse)

    def _embed_one(self, name: str, tag: str, fold: int | None, reuse: bool) -> np.ndarray:
        import polars as pl

        b = self._require()
        rt, sci = self.cfg.runtime, self.cfg.science
        if name == reg.TCN:
            raise ValueError("F1_TCN is an external athlete-level representation (tcn_representations.parquet); it enters transfer() only")
        if name == reg.ENGINEERED:
            deps = self._deps(model=name, kind="features")
        elif name == reg.RANDOM:
            deps = self._deps(model=name, init_seed=sci.seed, embed_precision=rt.embed_precision)
        else:
            rec = self._training_record(name, tag)
            deps = self._deps(model=name, tag=tag, weights=rec.get("sha256"), embed_precision=rt.embed_precision)
        key = f"{name}__{tag}"
        if reuse:
            hit = self.manifest.find("embeddings", key, deps)
            if hit is not None:
                df = pl.read_parquet(self.manifest.resolve(hit))
                return df.select([c for c in df.columns if re.fullmatch(r"e\d+", c)]).to_numpy().astype(np.float32)
        if name == reg.ENGINEERED:
            E = self._E0
        else:
            from .training.trainer import embed, random_frozen

            if name == reg.RANDOM:
                model = random_frozen(reg.enc_config(name, sci), self.device, sci.seed)
            else:
                from .training.checkpoints import load_best

                model, _sig = load_best(self.manifest.resolve(rec), self.device)
            E = embed(model, self._device_data(), rt.embed_batch, rt.embed_precision)
            del model
            cleanup(self.device)
        tr, va, ho = self._splits(fold)
        roles = sp.split_roles(b.n_sequences, tr, va, ho)
        fold_col = [self._folds.get(int(i), -1) for i in b.meta["nfl_id"].to_list()]
        df = (b.meta.select("idx", "seq_key", "domain", "nfl_id", "kind", "position_group")
              .with_columns(pl.Series("fold", fold_col, dtype=pl.Int64), pl.Series("split_role", roles.tolist(), dtype=pl.String),
                            *[pl.Series(f"e{j}", E[:, j]) for j in range(E.shape[1])]))
        p = self.store.embeddings / f"{key}__{deps_hash(deps)}.parquet"
        self.store.write_parquet(df, p, overwrite=True)
        atomic_json({"model": name, "tag": tag, "fold": fold, "deps": deps, "dims": int(E.shape[1]), "rows": int(E.shape[0]), "written_at": utc_now()},
                    p.with_suffix(".json"), overwrite=True)
        self.manifest.register("embeddings", key, p, deps, {"fold": fold, "dims": int(E.shape[1])})
        return E

    @property
    def bundle(self) -> Any:
        """The opened (staged, validated) data bundle; ``prepare()`` is called on first access."""
        return self._require()

    def inductive_athlete_embeddings(self, model: str = "F4", folds: Iterable[int] | None = None) -> dict[int, Any]:
        """{fold: athlete-mean Combine forty embedding table} of the inductive encoder of each outer fold (``se.athlete_mean_embedding``,
        exactly the 12F notebook's ``BYF[name][f]``) — the input of ``se.embedding_block`` in the 12F / 12G ladders. Uses the saved
        embeddings (``embed(..., regime='inductive')`` runs first when they are missing)."""
        se = _backend.se()
        b = self._require()
        name = reg.resolve(model)
        return {f: se.athlete_mean_embedding(self._embed_one(name, _tag(INDUCTIVE, fold=f), f, reuse=True), b.meta) for f in self._fold_list(folds)}

    def load_embeddings(self, model: str, regime: str = TRANSDUCTIVE, fold: int | None = None, nfl_fraction: float = 1.0) -> Any:
        """The saved embedding table (polars) — ids, domain, fold, split role, e0…; no recomputation."""
        import polars as pl

        name = reg.resolve(model)
        tag = _tag(regime, nfl_fraction, fold)
        rec = self.manifest.find("embeddings", f"{name}__{tag}")
        if rec is None:
            raise FileNotFoundError(f"no saved embeddings for {name} ({tag}); call embed('{name}', regime='{regime}')")
        return pl.read_parquet(self.manifest.resolve(rec))

    # ================================================================== evaluation
    def evaluate(self, model: str, level: str | None = None, regime: str = TRANSDUCTIVE, reuse: bool = True, nfl_fraction: float = 1.0,
                 folds: Iterable[int] | None = None) -> Any:
        """Label-free evaluation; returns the result row(s) as polars. Quick-level rows are DIAGNOSTIC (subsampled
        candidate universe) and flagged ``comparable_to_full = False``."""
        import polars as pl

        name = reg.resolve(model)
        level = level or self.cfg.evaluation.default_level
        ev, sci = self.cfg.evaluation, self.cfg.science
        b = self._require()
        if regime == INDUCTIVE:
            return self._evaluate_inductive(name, reuse, folds)
        E = self._embed_one(name, _tag(TRANSDUCTIVE, nfl_fraction), None, reuse=True)
        emb = self.manifest.find("embeddings", f"{name}__{_tag(TRANSDUCTIVE, nfl_fraction)}")
        rep_name = _rep_name(name, nfl_fraction)
        deps = self._deps(embedding=emb["id"] if emb else name, level=level, evaluation=asdict(ev), n_boot=sci.n_boot, outer_folds=sci.outer_folds,
                          seed=sci.seed)
        key = f"{rep_name}__{level}"
        if reuse and (hit := self.manifest.find("evaluation", key, deps)) is not None:
            rec = json.loads(self.manifest.resolve(hit).read_text(encoding="utf-8"))
            return pl.DataFrame([rec["row"]], infer_schema_length=None)
        res = evaluate_representation(rep_name, E, b.meta, self._derived.phys, self._folds, sci, ev, level=level, regime=TRANSDUCTIVE)
        res.row.update({"experiment_id": self.id, "model": name, "nfl_fraction": nfl_fraction})
        rec = {"kind": "transductive", "row": res.row, "probes": res.probes.to_dicts() if res.probes.height else [],
               "null": None if res.null is None else [float(v) for v in res.null], "deps": deps, "written_at": utc_now()}
        p = self.store.records / f"transductive__{key.replace('%', 'pct').replace('@', '_')}__{deps_hash(deps)}.json"
        atomic_json(rec, p, overwrite=True)
        self.manifest.register("evaluation", key, p, deps, {"level": level})
        self._rebuild_views()
        return pl.DataFrame([res.row], infer_schema_length=None)

    def _evaluate_inductive(self, name: str, reuse: bool, folds: Iterable[int] | None) -> Any:
        import polars as pl

        sci, ev = self.cfg.science, self.cfg.evaluation
        b = self._require()
        rows = []
        for f in self._fold_list(folds):
            tag = _tag(INDUCTIVE, fold=f)
            E = self._E0 if name == reg.ENGINEERED else self._embed_one(name, tag, f, reuse=True)
            emb = self.manifest.find("embeddings", f"{name}__{tag}") if name != reg.ENGINEERED else None
            deps = self._deps(embedding=emb["id"] if emb else name, fold=f, n_boot=sci.n_boot, seed=sci.seed, backend=ev.backend)
            key = f"{name}__{tag}"
            hit = self.manifest.find("evaluation_inductive", key, deps) if reuse else None
            if hit is not None:
                rows.append(json.loads(self.manifest.resolve(hit).read_text(encoding="utf-8"))["row"])
                continue
            _, _, ho = self._splits(f)
            row = {**inductive_fold_retrieval(name, E, b.meta, ho, f, sci.n_boot, sci.seed, ev.backend), "experiment_id": self.id, "level": "standard"}
            p = self.store.records / f"inductive__{key}__{deps_hash(deps)}.json"
            atomic_json({"kind": "inductive", "row": row, "deps": deps, "written_at": utc_now()}, p, overwrite=True)
            self.manifest.register("evaluation_inductive", key, p, deps)
            rows.append(row)
        self._rebuild_views()
        return pl.DataFrame(rows, infer_schema_length=None)

    def _records(self, kind: str) -> list[dict[str, Any]]:
        """Latest record per key (manifest order), only those matching the current bundle / science."""
        out = {}
        for a in sorted(self.manifest.artifacts({"transductive": "evaluation", "inductive": "evaluation_inductive", "transfer": "transfer"}[kind]),
                        key=lambda r: r["created_at"]):
            p = self.manifest.resolve(a)
            if p.is_file() and a["deps"].get("science") == self.cfg.science_hash():
                out[a["key"]] = json.loads(p.read_text(encoding="utf-8"))
        return list(out.values())

    def _rebuild_views(self) -> None:
        import polars as pl

        st = self.store
        tr = self._records("transductive")
        if tr:
            metrics = pl.DataFrame([r["row"] for r in tr], infer_schema_length=None)
            st.write_csv(metrics, st.evaluation / "metrics.csv")
            probes = [pl.DataFrame(r["probes"]) for r in tr if r.get("probes")]
            if probes:
                st.write_csv(pl.concat(probes, how="diagonal_relaxed"), st.evaluation / "probes.csv")
            nulls = [pl.DataFrame({"representation": [r["row"]["representation"]] * len(r["null"]), "replicate": list(range(len(r["null"]))),
                                   "cross_auc_null": [float(v) for v in r["null"]], "observed_cross_auc": [r["row"]["cross_auc"]] * len(r["null"])})
                     for r in tr if r.get("null")]
            if nulls:
                st.write_csv(pl.concat(nulls, how="diagonal_relaxed"), st.evaluation / "identity_null.csv")
        ind = self._records("inductive")
        if ind:
            st.write_csv(pl.DataFrame([r["row"] for r in ind], infer_schema_length=None), st.evaluation / "inductive_retrieval.csv")
        parts = []
        if tr:
            parts.append(pl.DataFrame([r["row"] for r in tr], infer_schema_length=None).with_columns(pl.lit("transductive").alias("table")))
        if ind:
            parts.append(pl.DataFrame([r["row"] for r in ind], infer_schema_length=None).with_columns(pl.lit("inductive").alias("table")))
        if parts:
            st.write_parquet(pl.concat(parts, how="diagonal_relaxed"), st.evaluation / "retrieval_results.parquet", overwrite=True)

    # ================================================================== NFL transfer (inductive ladder)
    def transfer(self, models: Sequence[str] | None = None, reuse: bool = True) -> Any:
        """12F Part 7b ladder on the 12C frozen target: M1 · M1+F0 · M1+F1 TCN (if in the bundle) · M1+<encoder> (inductive
        fold embeddings, ≤ 8 training-fold PCA dims). Paired athlete-bootstrap Δ RMSE; never mixed with transductive tables."""
        import polars as pl

        from .data.validation import TRANSFER_TARGET, validate_athletes

        b = self._require()
        if b.athletes is None:
            raise FileNotFoundError("the bundle has no athletes.parquet: the NFL-transfer ladder is unavailable (re-export with include_transfer=True)")
        validate_athletes(b.athletes, require_target=True)
        ae, rep, se = _backend.ae(), _backend.rep(), _backend.se()
        sci = self.cfg.science
        C = sci.to_m12config()
        names = [reg.resolve(m) for m in (models or sci.inductive_configs)]
        byf: dict[str, dict[int, Any]] = {}
        emb_ids = []
        for n in names:
            byf[n] = {}
            for f in self._fold_list(None):
                tag = _tag(INDUCTIVE, fold=f)
                E = self._embed_one(n, tag, f, reuse=True)
                emb_ids.append(self.manifest.find("embeddings", f"{n}__{tag}")["id"])
                byf[n][f] = se.athlete_mean_embedding(E, b.meta)
        deps = self._deps(models=names, embeddings=sorted(emb_ids), tcn=b.tcn is not None, n_boot=sci.n_boot, seed=sci.seed)
        if reuse and (hit := self.manifest.find("transfer", "ladder", deps)) is not None:
            return pl.DataFrame(json.loads(self.manifest.resolve(hit).read_text(encoding="utf-8"))["rows"], infer_schema_length=None)
        e0a = rep.athlete_embedding_table(self._E0, b.meta, "combine", ["forty"], prefix="eng")
        data = (b.athletes.filter(pl.col("linked").fill_null(False)).filter(pl.col(TRANSFER_TARGET).is_not_null())
                .join(e0a, on="nfl_id", how="left").join(b.folds.select("nfl_id", "fold"), on="nfl_id", how="inner").sort("nfl_id"))
        mods = {"M1": ae.Model(list(ae.M1_COLS)), "M1+F0": ae.Model([*ae.M1_COLS, *[c for c in e0a.columns if c != "nfl_id"]])}
        for n, bf in byf.items():
            mods[f"M1+{n}"] = ae.Model(list(ae.M1_COLS), factory=lambda a, _b=bf: se.embedding_block(a, self._folds, _b))
        if b.tcn is not None:
            mods["M1+F1_TCN"] = ae.Model(list(ae.M1_COLS), factory=lambda a, _b=se.tcn_by_fold(b.tcn): se.embedding_block(a, self._folds, _b))
        res = ae.oof_ladder(data, TRANSFER_TARGET, mods, C, C.seed)
        comps = [(m, "M1") for m in mods if m != "M1"] + [(f"M1+{n}", "M1+F0") for n in byf]
        trf = ae.compare(res, comps, C, C.seed, target=TRANSFER_TARGET, training_regime=INDUCTIVE)
        trf = trf.with_columns(pl.lit(self.id).alias("experiment_id")) if trf.height else trf
        p = self.store.records / f"transfer__ladder__{deps_hash(deps)}.json"
        atomic_json({"kind": "transfer", "rows": trf.to_dicts(), "ladder_log": res.log, "deps": deps, "written_at": utc_now()}, p, overwrite=True)
        self.manifest.register("transfer", "ladder", p, deps)
        if trf.height:
            self.store.write_csv(trf, self.store.evaluation / "transfer.csv")
        return trf

    # ================================================================== comparison / tables
    def metrics(self, include_quick: bool = True) -> Any:
        import polars as pl

        p = self.store.evaluation / "metrics.csv"
        if not p.is_file():
            return pl.DataFrame()
        m = pl.read_csv(p, infer_schema_length=None)
        return m if include_quick else m.filter(pl.col("level") != "quick")

    def compare(self, models: Sequence[str] | None = None, level: str | None = None) -> Any:
        """Side-by-side transductive metrics of this experiment (one level at a time: quick rows are never mixed with
        standard / full)."""
        import polars as pl

        m = self.metrics()
        if not m.height:
            return m
        level = level or self.cfg.evaluation.default_level
        m = m.filter(pl.col("level") == level)
        if models:
            names = [reg.resolve(x) for x in models]
            m = m.filter(pl.col("model").is_in(names))
        cols = ["representation", "level", "cross_auc", "cross_auc_lo", "cross_auc_hi", "cross_n_athletes", "within_top1", "within_auc", "domain_auc", "mmd2",
                "probe_r2_median", "null_mean", "null_exceed_share", "neighbour_dspeed", "random_dspeed"]
        return m.select([c for c in cols if c in m.columns]).sort("cross_auc", descending=True, nulls_last=True)

    @staticmethod
    def compare_experiments(paths: Sequence[str | Path], level: str = "standard") -> Any:
        """Stack the metrics of several experiment directories (no recomputation)."""
        import polars as pl

        out = []
        for p in paths:
            f = Path(p) / "evaluation" / "metrics.csv"
            if f.is_file():
                out.append(pl.read_csv(f, infer_schema_length=None).with_columns(pl.lit(Path(p).name).alias("experiment")))
        if not out:
            return pl.DataFrame()
        m = pl.concat(out, how="diagonal_relaxed")
        return m.filter(pl.col("level") == level) if "level" in m.columns else m

    def list_artifacts(self, kind: str | None = None) -> Any:
        import polars as pl

        t = self.manifest.table()
        return t.filter(pl.col("kind") == kind) if (kind and t.height) else t

    # ================================================================== figures / report / export
    def plot(self, kind: str = "all", save: bool = True, formats: Sequence[str] = ("png", "svg", "pdf")) -> dict[str, Any]:
        """Build figures from the SAVED tables (works in a new session) and optionally save them with their data.
        kinds: training_curves, retrieval, probes, domain, identity_null, embedding_map, inductive, transfer, learning_curve."""
        import polars as pl

        from .visualization import plots as pv

        ev = self.store.evaluation
        read = lambda n: pl.read_csv(ev / n, infer_schema_length=None) if (ev / n).is_file() else pl.DataFrame()     # noqa: E731
        figs: dict[str, tuple[Any, dict[str, Any]]] = {}
        want = (lambda k: kind in ("all", k))
        metrics = read("metrics.csv")
        main = metrics.filter(pl.col("level") != "quick") if metrics.height and "level" in metrics.columns else metrics
        if want("training_curves"):
            cur = read("training_curves.csv")
            figs["training_curves"] = (pv.training_curves(cur), {"curves": cur})
        if want("retrieval") and main.height:
            d = main.with_columns((pl.col("representation") + " · " + pl.col("level")).alias("label"))
            nm = float(d["null_mean"].drop_nulls().mean()) if "null_mean" in d.columns and d["null_mean"].drop_nulls().len() else None
            figs["retrieval_transductive"] = (pv.forest(d, "label", "cross_auc", "cross_auc_lo", "cross_auc_hi", "Cross-domain athlete retrieval (transductive)",
                                                        "retrieval AUC (0.5 = chance)", ref=0.5, ref_label="chance", extra_ref=nm,
                                                        extra_ref_label="identity-null mean" if nm else None), {"metrics": d})
        if want("probes"):
            pr = read("probes.csv")
            if pr.height:
                pr = pr.filter(pl.col("level") != "quick") if "level" in pr.columns else pr
                pr = pr.with_columns((pl.col("domain") + " · " + pl.col("probe")).alias("target"))
                figs["probes"] = (pv.heatmap(pr, "representation", "target", "r2_cv", "Physical probe quality (athlete-grouped CV R²)"), {"probes": pr})
        if want("domain") and main.height and "domain_auc" in main.columns:
            dd = main.select("representation", "domain_auc", "mmd2").unpivot(index="representation", variable_name="metric", value_name="value")
            figs["domain_distance"] = (pv.heatmap(dd, "representation", "metric", "value", "Domain separability (AUC) and MMD²"), {"domain": dd})
        if want("identity_null"):
            nl = read("identity_null.csv")
            for r in (nl["representation"].unique(maintain_order=True).to_list() if nl.height else []):
                g = nl.filter(pl.col("representation") == r)
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", r)
                figs[f"identity_null__{safe}"] = (pv.null_histogram(g["cross_auc_null"].to_numpy(), float(g["observed_cross_auc"][0]),
                                                                    f"Shuffled-identity null · {r}"), {"null": g})
        if want("embedding_map"):
            se = _backend.se()
            for n in (se.PRIMARY,):
                rec = self.manifest.find("embeddings", f"{n}__{TRANSDUCTIVE}")
                if rec:
                    df = pl.read_parquet(self.manifest.resolve(rec))
                    E = df.select([c for c in df.columns if re.fullmatch(r"e\d+", c)]).to_numpy()
                    figs[f"embedding_map__{n}"] = (pv.embedding_map(E, df, f"Embedding map · {n} (transductive, PCA)", seed=self.cfg.science.seed), {})
        if want("inductive"):
            ind = read("inductive_retrieval.csv")
            if ind.height:
                d = ind.with_columns((pl.col("representation") + " · fold " + pl.col("held_out_fold").cast(pl.String)).alias("label"))
                figs["retrieval_inductive"] = (pv.forest(d, "label", "auc", "auc_lo", "auc_hi", "Held-out cross-domain retrieval (inductive)",
                                                         "retrieval AUC (0.5 = chance)", ref=0.5, ref_label="chance", color_by="representation"), {"inductive": d})
        if want("transfer"):
            t = read("transfer.csv")
            if t.height:
                d = t.with_columns((pl.col("model") + " vs " + pl.col("reference")).alias("label"))
                figs["transfer"] = (pv.forest(d, "label", "delta_rmse", "delta_lo", "delta_hi", "NFL transfer: Δ RMSE vs reference (inductive)",
                                              "Δ RMSE (positive = model better)", ref=0.0, ref_label="no change"), {"transfer": d})
        if want("learning_curve") and main.height and "nfl_fraction" in main.columns:
            lc = main.filter(pl.col("model") == _backend.se().PRIMARY)
            if lc.select(pl.col("nfl_fraction").n_unique()).item() > 1:
                figs["learning_curve"] = (pv.lines(lc, "nfl_fraction", "cross_auc", "cross_auc_lo", "cross_auc_hi", "Retrieval vs share of unlabeled NFL windows",
                                                   "share of NFL windows", "retrieval AUC", ref=0.5), {"learning_curve": lc})
        if save:
            index = []
            for name, (fig, src) in figs.items():
                paths = self.store.save_figure(fig, name, tuple(formats), src)
                index.append({"figure": name, **paths})
            if index:
                self.store.write_csv(pl.DataFrame(index, infer_schema_length=None), self.store.figures / "index.csv")
        return {k: v[0] for k, v in figs.items()}

    def report(self) -> Path:
        """``reports/REPORT.md`` from the saved tables (quick results in a separate, labelled section)."""
        import polars as pl

        m = self.metrics()
        main = m.filter(pl.col("level") != "quick") if m.height else m
        quick = m.filter(pl.col("level") == "quick") if m.height else m
        ev = self.store.evaluation
        ind = pl.read_csv(ev / "inductive_retrieval.csv") if (ev / "inductive_retrieval.csv").is_file() else pl.DataFrame()
        trf = pl.read_csv(ev / "transfer.csv") if (ev / "transfer.csv").is_file() else pl.DataFrame()
        ind_sum = (ind.group_by("representation").agg(pl.col("auc").mean().alias("cross_auc_mean_over_folds"), pl.col("auc").min().alias("min_fold_auc"),
                                                      pl.col("n_athletes").sum().alias("n_athletes")).sort("representation") if ind.height else ind)
        saved = json.loads((self.root / "config" / "experiment.json").read_text(encoding="utf-8"))
        figs = sorted(p.name for p in (self.store.figures / "png").glob("*.png"))
        md = [f"# Experiment {self.id}", "", f"_Generated {utc_now()} from saved artifacts. Nothing in this file is computed by hand._", "",
              "## Setup", "", f"- Preset: `{self.cfg.preset}` · science hash `{self.cfg.science_hash()}` · seed {self.cfg.science.seed} · run mode `{self.cfg.science.run_mode}`",
              f"- Bundle: `{saved.get('bundle_fingerprint')}` ({saved.get('data_dir')})", f"- Code: `{json.dumps(saved.get('code_version'))}`",
              f"- Evaluation backend `{self.cfg.evaluation.backend}`; exact options: {self.cfg.evaluation.is_exact()}", "",
              "## Transductive label-free diagnostics (standard / full)", "",
              "Encoders trained on every sequence without identities or NFL outcomes: diagnostics only, never NFL prediction.", "",
              _md_table(main, ["representation", "level", "cross_auc", "cross_auc_lo", "cross_auc_hi", "cross_n_athletes", "within_auc", "domain_auc",
                               "probe_r2_median", "null_mean", "null_exceed_share"]),
              "## Inductive held-out retrieval (separate table)", "", _md_table(ind_sum),
              "## NFL transfer (inductive ladder, 12C target)", "", _md_table(trf, ["model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict"], 4),
              "## Quick-level results — DIAGNOSTIC ONLY", "", "Seeded athlete subsample; candidate sets are smaller, so AUCs are NOT comparable with the tables above.", "",
              _md_table(quick, ["representation", "cross_auc", "quick_athletes", "domain_auc", "probe_r2_median"]),
              "## Figures", "", *[f"- figures/png/{f}" for f in figs], "",
              "## Caveats", "", "- Domain separability is not failure: Combine drills and NFL movement are different tasks.",
              "- Wall-clock caps can stop training early (`time_capped` in each checkpoint signature).",
              "- fast backend: same estimands and random draws as nflbdb; cosine similarities may differ in the last bit (see REPRODUCIBILITY.md).", ""]
        p = self.store.reports / "REPORT.md"
        atomic_text("\n".join(md), p)
        return p

    def export(self, dest: str | Path | None = None, include_embeddings: bool = True, include_checkpoints: bool = False) -> Path:
        """Zip config, manifests, evaluation, figures, reports, logs (+ embeddings; + best.pt files if requested)."""
        self.store.write_csv(self.manifest.table(), self.root / "manifests" / "artifacts.csv")
        stamp = utc_now().replace(":", "").replace("-", "")
        out = Path(dest) if dest else self.root / "exports" / f"{self.id}_{stamp}.zip"
        dirs = ["config", "manifests", "evaluation", "figures", "reports", "logs", "data/processed"] + (["embeddings"] if include_embeddings else [])

        def write(tmp: Path) -> None:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for d in dirs:
                    for f in sorted((self.root / d).rglob("*")):
                        if f.is_file() and not f.name.startswith("."):
                            z.write(f, f"{self.id}/{f.relative_to(self.root).as_posix()}")
                if include_checkpoints:
                    for f in sorted(self.store.checkpoints.glob("*/*/best.*")):
                        z.write(f, f"{self.id}/{f.relative_to(self.root).as_posix()}")

        atomic_write(out, write, overwrite=False)
        log.info("exported %s (%.1f MB)", out, out.stat().st_size / 2 ** 20)
        return out

    def __repr__(self) -> str:
        return f"Experiment(id={self.id!r}, root={str(self.root)!r}, preset={self.cfg.preset!r}, science={self.cfg.science_hash()})"


__all__ = ["Experiment", "INDUCTIVE", "TRANSDUCTIVE"]
