"""E15.B1 — budget-matched, fold-safe, resumable training and embedding export (T07).

Per (architecture, d, seed, outer fold f):
* training athletes = athletes whose official fold ≠ f (``M15Data.fold_of``); their sequences of ``cfg.train_domains``
  only. A seeded ``inner_val_share`` of TRAINING athletes is the early-stopping set. Held-out athletes (fold f) are
  never seen in training or early stopping — their rows are only embedded with the frozen best weights;
* identical budget for every learned architecture: ``train_steps`` optimiser steps, batch ``batch_size``, AdamW
  (``lr``, ``weight_decay``), 5 % warm-up then cosine decay, gradient clipping ``grad_clip``, evaluation every
  ``eval_every`` steps, early stop after ``patience_evals`` evaluations without improvement;
* resumable: ``last.pt`` (model, optimiser, scheduler, step, history, RNG states) every evaluation; ``done.json`` marks a
  finished fit; a finished fit is never retrained (its embeddings are re-exported only when missing);
* PCA is fitted in closed form on the training rows (``pca.npz``).

Outputs (under the M15 run):
    checkpoints/<arch>_d<d>_s<seed>/fold<f>/{best.pt | pca.npz, last.pt, config.json, history.parquet, train_athletes.json, done.json}
    embeddings/inductive/<arch>_d<d>_s<seed>__fold<f>.parquet   idx, nfl_id, domain, kind, position_group, fold, role, z0…z{d-1}
    (+ ``.provenance.json``: bundle / config fingerprints, architecture, width, parameters, training-athlete hash)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ...artifacts.store import atomic_json
from ...utils.reproducibility import utc_now
from ..status import NOT_RUN, ResearchBlocked, BLOCKED_INSUFFICIENT_SAMPLE
from . import models as MD

TRAIN, VAL, HELD = "train", "validation", "held_out"


@dataclass(frozen=True)
class FitSpec:
    arch: str
    d: int
    seed: int
    fold: int

    @property
    def tag(self) -> str:
        return f"{self.arch}_d{self.d}_s{self.seed}"


def grid(cfg: Any) -> list[FitSpec]:
    return [FitSpec(a, int(d), int(s), int(f)) for a in cfg.architectures for d in cfg.latent_dims for s in cfg.train_seeds for f in cfg.selected_folds()]


def ckpt_dir(run: Any, spec: FitSpec) -> Path:
    return run.root / "checkpoints" / spec.tag / f"fold{spec.fold}"


def emb_path(run: Any, spec: FitSpec) -> Path:
    return run.root / "embeddings" / "inductive" / f"{spec.tag}__fold{spec.fold}.parquet"


# ====================================================================== data
_ARRAYS: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def arrays(data: Any) -> tuple[np.ndarray, np.ndarray]:
    """The bundle arrays in RAM (≈ 120 MB for 77k × 64 × 6 float32), loaded once per process."""
    key = data.fingerprint
    if key not in _ARRAYS:
        _ARRAYS[key] = (np.ascontiguousarray(np.asarray(data.bundle.X), np.float32), np.ascontiguousarray(np.asarray(data.bundle.M), bool))
    return _ARRAYS[key]


def roles(data: Any, fold: int, cfg: Any, seed: int) -> np.ndarray:
    """Per sequence: train / validation / held_out (athlete-level; validation drawn from training athletes)."""
    idx = data.index
    fo = idx["fold"].to_numpy()
    ids = idx["nfl_id"].to_numpy()
    role = np.where(fo == fold, HELD, TRAIN).astype(object)
    tr_ath = np.unique(ids[fo != fold])
    rng = np.random.default_rng(seed + 1009 * fold)
    val_ath = set(rng.choice(tr_ath, max(1, int(round(cfg.inner_val_share * tr_ath.size))), replace=False).tolist()) if tr_ath.size else set()
    role[np.isin(ids, list(val_ath)) & (fo != fold)] = VAL
    if set(ids[role == HELD].tolist()) & set(ids[role != HELD].tolist()):
        raise AssertionError("an athlete appears both held out and in training / validation")
    return role


def train_rows(data: Any, role: np.ndarray, which: str, cfg: Any) -> np.ndarray:
    dom = data.index["domain"].to_numpy()
    nv = data.index["n_valid"].to_numpy()
    return np.flatnonzero((role == which) & np.isin(dom, list(cfg.train_domains)) & (nv >= cfg.min_segment_frames))


def athlete_hash(ids: Any) -> str:
    return hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()[:16]


# ====================================================================== torch loop
def _batches(rows: np.ndarray, bs: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    while True:
        perm = rng.permutation(rows)
        for s in range(0, perm.size - bs + 1, bs):
            yield np.sort(perm[s:s + bs])


def _lr(step: int, total: int, base: float) -> float:
    warm = max(1, int(0.05 * total))
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * 0.5 * (1 + np.cos(np.pi * min(1.0, p)))


def _val_loss(model: Any, X: np.ndarray, M: np.ndarray, rows: np.ndarray, device: str, bs: int = 1024) -> float:
    import torch

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, rows.size, bs):
            r = rows[s:s + bs]
            x, m = torch.from_numpy(X[r]).to(device), torch.from_numpy(M[r]).to(device)
            rec, _ = model(x, m)
            tot += float(model.loss(x, m, rec)) * r.size
            n += r.size
    model.train()
    return tot / max(n, 1)


def fit_torch(spec: FitSpec, data: Any, cfg: Any, run: Any, device: str) -> dict[str, Any]:
    import polars as pl
    import torch

    d = ckpt_dir(run, spec)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "done.json").is_file():
        return {**json.loads((d / "done.json").read_text(encoding="utf-8")), "status": "reused"}
    X, M = arrays(data)
    role = roles(data, spec.fold, cfg, spec.seed)
    tr, va = train_rows(data, role, TRAIN, cfg), train_rows(data, role, VAL, cfg)
    if tr.size < cfg.batch_size or va.size < 32:
        raise ResearchBlocked(BLOCKED_INSUFFICIENT_SAMPLE, f"fold {spec.fold}: {tr.size} training / {va.size} validation rows")
    width = MD.choose_width(spec.arch, spec.d, cfg)
    torch.manual_seed(spec.seed + 7919 * spec.fold)
    rng = np.random.default_rng(spec.seed + 104729 * spec.fold)
    model = MD.build(spec.arch, spec.d, cfg, width["width"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step, best, bad, hist = 0, float("inf"), 0, []
    last = d / "last.pt"
    if last.is_file() and cfg.resume:
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, best, bad, hist = int(st["step"]), float(st["best"]), int(st["bad"]), list(st["history"])
        rng = np.random.default_rng(st["rng_seed"] + step)
    tr_ids = sorted(set(int(i) for i in data.index["nfl_id"].to_numpy()[role != HELD].tolist()))
    atomic_json({"train_and_validation_athletes": tr_ids, "held_out_fold": spec.fold, "hash": athlete_hash(tr_ids)}, d / "train_athletes.json", overwrite=True)
    meta = {**asdict(spec), "width": width, "train_rows": int(tr.size), "val_rows": int(va.size), "train_domains": list(cfg.train_domains),
            "bundle_fingerprint": data.fingerprint, "config_fingerprint": cfg.fingerprint(), "train_athlete_hash": athlete_hash(tr_ids), "device": device}
    atomic_json(meta, d / "config.json", overwrite=True)
    it = _batches(tr, cfg.batch_size, rng)
    model.train()
    t0 = time.monotonic()
    while step < cfg.train_steps and bad < cfg.patience_evals:
        for g in opt.param_groups:
            g["lr"] = _lr(step, cfg.train_steps, cfg.lr)
        r = next(it)
        x, m = torch.from_numpy(X[r]).to(device), torch.from_numpy(M[r]).to(device)
        rec, _ = model(x, m)
        loss = model.loss(x, m, rec)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        if step % cfg.eval_every == 0 or step == cfg.train_steps:
            vl = _val_loss(model, X, M, va, device)
            hist.append({"step": step, "train_loss": float(loss), "val_loss": vl, "lr": opt.param_groups[0]["lr"], "seconds": time.monotonic() - t0})
            if vl < best - 1e-6:
                best, bad = vl, 0
                torch.save({"model": model.state_dict(), "spec": asdict(spec), "width": width}, d / "best.pt")
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "bad": bad, "history": hist,
                        "rng_seed": spec.seed + 104729 * spec.fold}, last)
    pl.DataFrame(hist).write_parquet(d / "history.parquet") if hist else None
    done = {**meta, "steps": step, "best_val_loss": best, "stopped_early": bad >= cfg.patience_evals, "params": width["params"], "finished_at": utc_now()}
    atomic_json(done, d / "done.json", overwrite=True)
    return {**done, "status": "trained"}


def fit_pca(spec: FitSpec, data: Any, cfg: Any, run: Any) -> dict[str, Any]:
    d = ckpt_dir(run, spec)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "done.json").is_file() and (d / "pca.npz").is_file():
        return {**json.loads((d / "done.json").read_text(encoding="utf-8")), "status": "reused"}
    X, M = arrays(data)
    role = roles(data, spec.fold, cfg, spec.seed)
    rows = np.flatnonzero(role != HELD)
    rows = rows[np.isin(data.index["domain"].to_numpy()[rows], list(cfg.train_domains))]
    p = MD.PCAEncoder(spec.d).fit(X[rows], M[rows])
    np.savez(d / "pca.npz", **{k: np.asarray(v) for k, v in p.state().items()})
    tr_ids = sorted(set(int(i) for i in data.index["nfl_id"].to_numpy()[role != HELD].tolist()))
    atomic_json({"train_and_validation_athletes": tr_ids, "held_out_fold": spec.fold, "hash": athlete_hash(tr_ids)}, d / "train_athletes.json", overwrite=True)
    done = {**asdict(spec), "params": p.n_params(), "train_rows": int(rows.size), "explained_variance": float(p.explained.sum()), "bundle_fingerprint": data.fingerprint,
            "config_fingerprint": cfg.fingerprint(), "train_athlete_hash": athlete_hash(tr_ids), "finished_at": utc_now()}
    atomic_json(done, d / "done.json", overwrite=True)
    return {**done, "status": "fitted"}


# ====================================================================== loading and embedding
def load_model(spec: FitSpec, cfg: Any, run: Any, device: str = "cpu") -> Any:
    d = ckpt_dir(run, spec)
    if spec.arch == "pca":
        z = np.load(d / "pca.npz", allow_pickle=False)
        return MD.PCAEncoder.from_state({k: z[k] for k in z.files})
    import torch

    st = torch.load(d / "best.pt", map_location=device, weights_only=False)
    model = MD.build(spec.arch, spec.d, cfg, st["width"]["width"]).to(device)
    model.load_state_dict(st["model"])
    model.eval()
    return model


def encode_rows(model: Any, X: np.ndarray, M: np.ndarray, rows: np.ndarray, device: str = "cpu", bs: int = 2048) -> np.ndarray:
    if isinstance(model, MD.PCAEncoder):
        return np.concatenate([model.encode(X[rows[s:s + bs]], M[rows[s:s + bs]]) for s in range(0, rows.size, bs)]) if rows.size else np.zeros((0, model.d))
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, rows.size, bs):
            r = rows[s:s + bs]
            out.append(model.encode(torch.from_numpy(X[r]).to(device), torch.from_numpy(M[r]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def reconstruct_rows(model: Any, X: np.ndarray, M: np.ndarray, rows: np.ndarray, device: str = "cpu", bs: int = 2048) -> np.ndarray:
    if isinstance(model, MD.PCAEncoder):
        return np.concatenate([model.reconstruct(X[rows[s:s + bs]], M[rows[s:s + bs]]) for s in range(0, rows.size, bs)])
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, rows.size, bs):
            r = rows[s:s + bs]
            rec, _ = model(torch.from_numpy(X[r]).to(device), torch.from_numpy(M[r]).to(device))
            out.append(rec.cpu().numpy())
    return np.concatenate(out)


def export_embeddings(spec: FitSpec, data: Any, cfg: Any, run: Any, device: str = "cpu") -> Path:
    """Embed every sequence with the frozen fold-f model; role per row (train / validation / held_out)."""
    import polars as pl

    p = emb_path(run, spec)
    if p.is_file():
        return p
    X, M = arrays(data)
    model = load_model(spec, cfg, run, device)
    rows = data.index["idx"].to_numpy()
    Z = encode_rows(model, X, M, rows, device)
    role = roles(data, spec.fold, cfg, spec.seed)
    tab = data.index.select("idx", "nfl_id", "domain", "kind", "position_group", "fold").with_columns(pl.Series("role", role.astype(str)),
                                                                                                    *[pl.Series(f"z{j}", Z[:, j]) for j in range(Z.shape[1])])
    done = json.loads((ckpt_dir(run, spec) / "done.json").read_text(encoding="utf-8"))
    run.data_file(tab, str(p.relative_to(run.root)), {"spec": asdict(spec), "train_athlete_hash": done.get("train_athlete_hash"),
                                                       "bundle_fingerprint": data.fingerprint, "model_fold": spec.fold})
    return p


def load_embeddings(run: Any, spec: FitSpec) -> Any:
    import polars as pl

    p = emb_path(run, spec)
    return pl.read_parquet(p) if p.is_file() else None


def run_grid(cfg: Any, data: Any, run: Any, device: str, specs: list[FitSpec] | None = None) -> Any:
    """Fit (or reuse) and export every spec; one status row per spec. Learned fits need both training opt-ins."""
    import polars as pl

    rows = []
    for spec in specs or grid(cfg):
        t0 = time.monotonic()
        try:
            if spec.arch == "pca":
                info = fit_pca(spec, data, cfg, run)
            elif not cfg.training_enabled():
                done = ckpt_dir(run, spec) / "done.json"
                if not done.is_file():
                    rows.append({**asdict(spec), "status": NOT_RUN, "reason": "training disabled (run_training + allow_new_training) and no finished checkpoint"})
                    continue
                info = {**json.loads(done.read_text(encoding="utf-8")), "status": "reused"}
            else:
                info = fit_torch(spec, data, cfg, run, device)
            export_embeddings(spec, data, cfg, run, device)
            rows.append({**asdict(spec), "status": info.get("status"), "reason": "", "params": info.get("params"), "steps": info.get("steps"),
                         "best_val_loss": info.get("best_val_loss"), "seconds": round(time.monotonic() - t0, 1)})
        except ResearchBlocked as e:
            rows.append({**asdict(spec), "status": e.status, "reason": e.reason})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["FitSpec", "HELD", "TRAIN", "VAL", "arrays", "ckpt_dir", "emb_path", "encode_rows", "export_embeddings", "fit_pca", "fit_torch", "grid",
           "load_embeddings", "load_model", "reconstruct_rows", "roles", "run_grid"]
