"""GPU-efficient trainer for the 12F shared movement encoders.

Scientific behaviour is ``se.train`` line for line: same seeding order, same domain-balanced batches
(``se._balanced_batches`` with the same NumPy generator), same loss (``se.batch_loss`` with the same CPU torch generator
for masks / crops), physics targets standardised with TRAINING-sequence constants, same shuffled-physics control, AdamW
(weight decay 1e-4), gradient accumulation with the same step rule, EMA update, validation on held-out ATHLETES with
fixed generators, early stopping on validation (patience, 1e-5 margin), wall-clock cap, best-state restore.

What changes is only WHERE the data lives and what is saved:
* ``DeviceData`` places the whole sequence store once per session — on the GPU when it fits (no per-batch host→device
  copy and no memory-mapped random reads), else in pinned host memory with non-blocking copies, else plain host memory.
  ``index_select`` on the placed tensors returns the same rows / dtypes as ``np.asarray(X[ix])``.
* Precision is explicit (``runtime.precision``; ``auto`` = the notebook's fp16 autocast + GradScaler on CUDA) and is
  part of the checkpoint signature together with the device type.
* Each epoch shard also stores the GradScaler state and every RNG stream → resume continues the same random streams
  (the original restarted the batch generator from the seed after an interruption).
* ``torch.compile`` is opt-in (``Module.compile`` in place, so state-dict keys and checkpoints are unchanged).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import _backend
from ..config import stable_hash
from ..devices import autocast, device_type, free_gpu_bytes, grad_scaler, require_torch, resolve_precision
from ..utils.logging import get_logger
from ..utils.reproducibility import capture_rng, restore_rng
from .checkpoints import epoch_checkpoint, export_best

log = get_logger(__name__)


def resolve_placement(requested: str, device: str, nbytes: int, fraction: float) -> str:
    dt = device_type(device)
    if dt == "cpu":
        return "cpu"
    if requested == "auto":
        free = free_gpu_bytes(device)
        if dt == "mps":
            return "gpu"
        if free is not None and nbytes <= fraction * free:
            return "gpu"
        return "pinned" if dt == "cuda" else "cpu"
    if requested == "pinned" and dt != "cuda":
        return "cpu"
    return requested


class DeviceData:
    """The sequence store (X, M) and per-sequence ids (state, category, domain) placed ONCE for all models of a session."""

    def __init__(self, X: np.ndarray, M: np.ndarray, state: np.ndarray, cat: np.ndarray, dom: np.ndarray, device: str, placement: str = "auto",
                 gpu_fraction: float = 0.5):
        torch = require_torch("training / embedding")
        self.device = device
        self.n, self.L = int(X.shape[0]), int(X.shape[1])
        self.dom_np = np.asarray(dom, np.int64)
        nbytes = X.shape[0] * X.shape[1] * X.shape[2] * 4 + M.size + 3 * 8 * X.shape[0]
        self.placement = resolve_placement(placement, device, nbytes, gpu_fraction)
        t0 = time.perf_counter()
        host = [torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(M, dtype=np.bool_)),
                torch.from_numpy(np.ascontiguousarray(state, np.int64)), torch.from_numpy(np.ascontiguousarray(cat, np.int64)),
                torch.from_numpy(self.dom_np.copy())]
        if self.placement == "gpu":
            self.t = [h.to(device) for h in host]
        elif self.placement == "pinned":
            self.t = [h.pin_memory() for h in host]
        else:
            self.t = host
        self.nbytes = nbytes
        log.info("sequence data placed: %s on %s (%.0f MB, %.1f s)", self.placement, device, nbytes / 2 ** 20, time.perf_counter() - t0)

    # ------------------------------------------------------------------ helpers
    def place(self, *arrays: np.ndarray) -> list[Any]:
        """Per-run arrays (standardised physics targets, validity) with the same placement."""
        torch = require_torch()
        out = [torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]
        if self.placement == "gpu":
            return [o.to(self.device) for o in out]
        if self.placement == "pinned":
            return [o.pin_memory() for o in out]
        return out

    def _move(self, t: Any) -> Any:
        if self.placement == "gpu":
            return t
        if self.placement == "pinned":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def take(self, ix: np.ndarray, extra: list[Any]) -> tuple[Any, ...]:
        """Batch rows ``ix`` → (x, m, p, p_ok, state, cat, dom): the argument order of ``se.batch_loss``."""
        torch = require_torch()
        it = torch.as_tensor(ix, dtype=torch.long)
        if self.placement == "gpu":
            it = it.to(self.device)
        x, m, s, c, d = (self._move(t.index_select(0, it)) for t in self.t)
        p, pk = (self._move(t.index_select(0, it)) for t in extra)
        return x, m, p, pk, s, c, d

    def slice(self, b: int, e: int) -> tuple[Any, Any]:
        x, m = self.t[0][b:e], self.t[1][b:e]
        if self.placement == "gpu":
            return x, m
        nb = self.placement == "pinned"
        return x.to(self.device, non_blocking=nb), m.to(self.device, non_blocking=nb)

    def release(self) -> None:
        self.t = []


def check_finite_loss(parts: dict[str, float], total: float | None, where: str) -> None:
    """Raise on a nonfinite loss instead of letting the GradScaler silently skip steps (a run that keeps going on NaN
    losses would still write curves and checkpoints that look valid). ``parts`` are the already-synchronised floats from
    ``se.batch_loss``, so this adds no extra device sync. Nonfinite GRADIENTS with a finite loss are the GradScaler's
    normal fp16 overflow handling and are left to it. Earlier epoch checkpoints stay intact for ``resume=True``."""
    bad = {k: v for k, v in parts.items() if not math.isfinite(v)}
    if total is not None and not math.isfinite(total):
        bad.setdefault("total", total)
    if bad:
        raise FloatingPointError(f"nonfinite loss at {where}: {bad} (all terms: {parts}). The last saved epoch checkpoint is untouched; "
                                 f"rerun with overrides={{'runtime.precision': 'fp32'}} to tell a mixed-precision overflow from a data / model problem.")


@dataclass
class TrainResult:
    model: Any
    curve: Any                     # polars DataFrame (epoch, val_loss, seconds, train_<part>…)
    signature: dict[str, Any]
    checkpoint_dir: str
    best_path: str | None
    restored_finished: bool
    resumed_from_epoch: int | None
    rng_restored: bool


def train_encoder(e: Any, data: DeviceData, P: np.ndarray, tr: np.ndarray, va: np.ndarray, science: Any, runtime: Any, ckpt_root: Path,
                  tag: str, seed: int, nfl_fraction: float = 1.0, best_file: Path | None = None, resume: bool = True,
                  on_epoch: Callable[[dict[str, Any]], None] | None = None) -> TrainResult:
    """Train or resume one encoder (see module docstring for the equivalence with ``se.train``)."""
    torch = require_torch("training")
    import polars as pl

    se = _backend.se()
    device = data.device
    prec = resolve_precision(runtime.precision, device, training=True)
    c = science
    dom = data.dom_np
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    if nfl_fraction < 1.0:
        keep = np.random.default_rng(seed).random(tr.size) < nfl_fraction
        tr = tr[(dom[tr] == 0) | keep]
    sig_cfg = {"enc": asdict(e), "train_hash": stable_hash(tr.tolist()), "epochs": c.epochs, "batch": c.batch, "accum": c.accum_steps, "lr": c.lr,
               "seed": seed, "L": data.L, "nfl_fraction": nfl_fraction,
               "numerics": {"device_type": device_type(device), **prec.signature(), "patience": c.patience, "max_minutes": c.max_minutes,
                            "max_train_batches": c.max_train_batches, "max_val_batches": c.max_val_batches, "val_hash": stable_hash(va.tolist())}}
    ck = epoch_checkpoint(ckpt_root, f"enc_{e.name}_{tag}", sig_cfg, c.run_mode)
    have = ck.latest()
    if have is not None and not resume:
        raise FileExistsError(f"{ck.dir} already holds checkpoints of this exact configuration; call with resume=True to continue / restore "
                              f"them, or start a new experiment to retrain from scratch")
    model = se.SharedEncoder(e).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c.lr, weight_decay=1e-4)
    scaler = grad_scaler(device, prec)
    # physics targets standardised with TRAINING-sequence constants (se.train)
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(P[tr], 0), np.nanstd(P[tr], 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    Pz = np.nan_to_num((P - mu) / sd, nan=0.0).astype(np.float32)
    Pok = np.isfinite(P)
    if e.shuffle_phys:
        perm = np.random.default_rng(seed + 99).permutation(P.shape[0])
        Pz, Pok = Pz[perm], Pok[perm]
    extra = data.place(Pz, Pok)
    curve: list[dict[str, Any]] = []
    start, best, bad, best_state = 0, np.inf, 0, None
    rng = np.random.default_rng(seed)
    resumed_from, rng_restored = None, False
    if have is not None:
        ep, st = have
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        curve, best, bad, best_state = st["curve"], st["best"], st["bad"], st["best_state"]
        start, resumed_from = ep + 1, ep
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        rng_restored = restore_rng(st.get("rng"), rng, gen)
        if st.get("finished"):
            model.load_state_dict(best_state)
            bp = None
            if best_file is not None:          # keep an existing export: re-saving could change its sha256 and invalidate embeddings
                bp = best_file if Path(best_file).is_file() else export_best(best_file, best_state, e, st["signature"])
            log.info("[%s %s] finished checkpoint restored (%d epochs) — not retrained", e.name, tag, len(curve))
            return TrainResult(model.eval(), pl.DataFrame(curve), st["signature"], str(ck.dir), str(bp) if bp else None, True, ep, rng_restored)
        log.info("[%s %s] resuming after epoch %d (rng streams %s)", e.name, tag, ep, "restored" if rng_restored else "re-seeded: notebook checkpoint")
    if runtime.compile:
        if hasattr(model.encoder, "compile"):
            model.encoder.compile(mode=runtime.compile_mode)          # in place: state-dict keys unchanged
        else:
            log.warning("runtime.compile requested but this PyTorch has no nn.Module.compile; continuing uncompiled")
    t_start = time.perf_counter()
    max_b = c.max_train_batches
    finished = False
    for epoch in range(start, c.epochs):
        t0 = time.perf_counter()
        model.train()
        sums: dict[str, float] = {}
        nb = 0
        batches = se._balanced_batches(tr, dom, c.batch, rng)
        opt.zero_grad(set_to_none=True)
        for k, ix in enumerate(batches):
            if max_b and nb >= max_b:
                break
            x, m, p, pk, s_, ca, d_ = data.take(ix, extra)
            with autocast(device, prec):
                loss, parts = se.batch_loss(model, x, m, p, pk, s_, ca, d_, gen)
                loss = loss / c.accum_steps
            check_finite_loss(parts, None, f"{e.name} {tag} epoch {epoch} train batch {k} ({prec.name})")
            (scaler.scale(loss) if scaler else loss).backward()
            if (k + 1) % c.accum_steps == 0:
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                model.ema_update()
            for kk, v in parts.items():
                sums[kk] = sums.get(kk, 0.0) + v
            nb += 1
        model.eval()
        vl, vn = 0.0, 0
        vgen = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for ix in se._balanced_batches(va, dom, c.batch, np.random.default_rng(seed + 2))[: c.max_val_batches]:
                x, m, p, pk, s_, ca, d_ = data.take(ix, extra)
                l_, vparts = se.batch_loss(model, x, m, p, pk, s_, ca, d_, vgen)
                lv = float(l_)
                check_finite_loss(vparts, lv, f"{e.name} {tag} epoch {epoch} validation batch {vn}")
                vl += lv
                vn += 1
        val = vl / max(vn, 1)
        elapsed = (time.perf_counter() - t_start) / 60
        row = {"epoch": epoch, "val_loss": val, "seconds": time.perf_counter() - t0, **{f"train_{k}": v / max(nb, 1) for k, v in sums.items()}}
        curve.append(row)
        if val < best - 1e-5:
            best, bad, best_state = val, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        finished = bad >= c.patience or epoch == c.epochs - 1 or elapsed >= c.max_minutes
        sig = {"config": e.name, "tag": tag, "mode": c.run_mode, "checkpoint_dir": str(ck.dir), "best_val": best, "epochs_run": epoch + 1, "finished": finished,
               "early_stopped": bad >= c.patience, "time_capped": elapsed >= c.max_minutes, "nfl_fraction": nfl_fraction, **prec.signature(),
               "device_type": device_type(device), "placement": data.placement}
        if finished or (epoch + 1 - start) % runtime.checkpoint_every == 0:
            ck.save_epoch(epoch, {"model": model.state_dict(), "opt": opt.state_dict(), "curve": curve, "best": best, "bad": bad, "best_state": best_state,
                                  "finished": finished, "signature": sig, "scaler": scaler.state_dict() if scaler else None, "rng": capture_rng(rng, gen)})
        eta = (time.perf_counter() - t_start) / (epoch - start + 1) * (c.epochs - epoch - 1) / 60
        if runtime.progress:
            print(f"[{e.name} {tag}] epoch {epoch + 1}/{c.epochs} · val {val:.4f} · {row['seconds']:.1f} s · elapsed {elapsed:.1f} min · "
                  f"ETA ≤ {eta:.1f} min · {prec.name} · {data.placement}", flush=True)
        if on_epoch is not None:
            on_epoch({**row, "finished": finished})
        if finished:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    signature = {"config": e.name, "tag": tag, "mode": c.run_mode, "checkpoint_dir": str(ck.dir), "best_val": best, "epochs_run": len(curve), "finished": True,
                 "nfl_fraction": nfl_fraction, **prec.signature(), "device_type": device_type(device), "placement": data.placement,
                 "signature_hash": ck.hash}
    bp = export_best(best_file, best_state, e, signature) if (best_file is not None and best_state is not None) else None
    return TrainResult(model.eval(), pl.DataFrame(curve), signature, str(ck.dir), str(bp) if bp else None, False, resumed_from, rng_restored)


def embed(model: Any, data: DeviceData, batch: int = 512, precision: str = "fp32") -> np.ndarray:
    """Pooled embeddings of every sequence, in store order (``se.embed`` semantics: eval mode, no grad, fp32 output).
    Batches are written into one device buffer and copied to the host once."""
    torch = require_torch("embedding")
    prec = resolve_precision(precision, data.device, training=False)
    model.eval()
    out = None
    with torch.no_grad():
        for b in range(0, data.n, batch):
            x, m = data.slice(b, b + batch)
            with autocast(data.device, prec):
                z = model.embed(x, m)
            z = z.float()
            if out is None:
                out = torch.empty((data.n, z.shape[1]), dtype=torch.float32, device=z.device)
            out[b: b + z.shape[0]] = z
    return out.cpu().numpy() if out is not None else np.zeros((0, 0), np.float32)


def random_frozen(e: Any, device: str, seed: int) -> Any:
    """The random-encoder baseline exactly as ``se.random_frozen`` (seeded CPU init, then moved)."""
    return _backend.se().random_frozen(e, device, seed)


__all__ = ["DeviceData", "TrainResult", "check_finite_loss", "embed", "random_frozen", "resolve_placement", "train_encoder"]
