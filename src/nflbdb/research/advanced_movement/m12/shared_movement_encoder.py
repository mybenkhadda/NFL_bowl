"""12F — physics-informed shared movement encoder (PyTorch; exploratory, compute-heavy).

Reused from Milestone 11 (``m11.representation``, ``m11.nfl_physics``): the unified 6-channel sequence format with FIXED
physical scaling (speed/10, a_tan/5, a_n/5, yaw[rad/s]/3 where the heading is valid else 0, Δt/0.1, heading-valid flag),
the immutable memory-mapped ``SequenceStore``, the deterministic capped NFL episode sample and the resumable window
extractor (one tracking scan, checkpoint shards per season × batch), the conv / transformer encoders, and the label-free
evaluations (within-domain retrieval, cross-domain athlete retrieval AUC, matched retrieval, probes, domain AUC / MMD).

New in 12F
    physics targets per sequence computed from the sequence itself (max speed, speed gain, mean a_tan, max |a_n|,
    |heading change| only when ≥ 50 % of frames have a valid heading, valid duration) — no athlete identity, no NFL outcome
    loss  L = λ_mask L_masked + λ_phys L_physics + λ_temp L_temporal + λ_align L_cross_domain
          L_temporal = InfoNCE between two temporal crops of one sequence (or JEPA latent prediction for F5)
          L_cross_domain ∈ {none, matched_state (supervised contrastive across domains on start-speed × accelerating
          state), adversarial (gradient reversal domain classifier), mmd (RBF), category (drill family ↔ NFL demand map)}
    families  F0 engineered · F1 TCN-B (Milestone 06, Combine forty only) · F2 small CNN, physics loss only · F3 masked AE ·
              F4 physics-informed contrastive + matched-state alignment (PRIMARY) · F5 JEPA-style + physics · F6 small
              transformer with the F4 objective · ablations F4_nophys, F4_noalign, F4_adv, F4_mmd, F4_category · negative
              control F4_shuffled_phys (physics targets permuted across sequences)
    regimes   TRANSDUCTIVE (all sequences, label-free diagnostics only) and INDUCTIVE (one encoder per outer fold without
              the held-out athletes' sequences of EITHER domain; the only regime used for NFL prediction). Never one table.
Domain differences may be real task differences: alignment that collapses domains is not a success criterion.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ....data.utils import get_logger
from ..m11 import representation as rep
from ..m11.checkpoint import ModelCheckpoint
from ..m11.config import M11Config
from . import advanced_evaluation as ae
from .config import INDUCTIVE, TRANSDUCTIVE, M12Config, config_hash

log = get_logger(__name__)
NAN = float("nan")
STORE_LEN = 64                   # proposed: 6.4 s per sequence in the store (NFL windows are 4.8 s; longer drills are truncated)
PHYS = ("max_speed", "speed_gain", "mean_a_tan", "max_abs_a_n", "abs_heading_change", "duration_s")
CATEGORY = {"forty": "accelerate", "pass_rush": "accelerate", "gauntlet": "sustain", "short_shuttle": "brake_turn", "three_cone": "brake_turn",
            "long_shuttle": "brake_turn", "backpedal_transition": "brake_turn", "accel_low": "accelerate", "accel_mid": "accelerate",
            "transition_high": "accelerate", "sustained_high": "sustain", "braking": "brake_turn", "reaccel_after_brake": "brake_turn",
            "redirection": "brake_turn"}
TORCH = rep.TORCH
if TORCH:
    import torch
    from torch import nn
    import torch.nn.functional as F


@dataclass(frozen=True)
class EncConfig:
    name: str
    encoder: str = "conv"            # conv · transformer
    lambda_mask: float = 1.0
    lambda_phys: float = 1.0
    lambda_temp: float = 0.5
    lambda_align: float = 0.5
    temporal: str = "contrastive"    # contrastive · jepa · none
    align: str = "matched_state"     # none · matched_state · adversarial · mmd · category
    shuffle_phys: bool = False
    embed_dim: int = 32
    width: int = 48
    layers: int = 4
    temperature: float = 0.1
    mask_ratio: float = 0.15
    span: int = 4
    ema: float = 0.99


FAMILIES: tuple[EncConfig, ...] = (
    EncConfig("F2_cnn_physics", lambda_mask=0.0, lambda_temp=0.0, lambda_align=0.0, temporal="none", align="none"),
    EncConfig("F3_masked_ae", lambda_phys=0.0, lambda_temp=0.0, lambda_align=0.0, temporal="none", align="none"),
    EncConfig("F4_physics_contrastive"),
    EncConfig("F5_jepa_physics", lambda_mask=0.0, temporal="jepa", align="none", lambda_align=0.0),
    EncConfig("F6_transformer", encoder="transformer"),
)
ABLATIONS: tuple[EncConfig, ...] = (
    EncConfig("F4_nophys", lambda_phys=0.0), EncConfig("F4_noalign", lambda_align=0.0, align="none"), EncConfig("F4_adv", align="adversarial"),
    EncConfig("F4_mmd", align="mmd"), EncConfig("F4_category", align="category"), EncConfig("F4_shuffled_phys", shuffle_phys=True),
)
PRIMARY = "F4_physics_contrastive"
INDUCTIVE_CONFIGS = ("F2_cnn_physics", "F4_physics_contrastive")


def by_name(name: str) -> EncConfig:
    return next(e for e in (*FAMILIES, *ABLATIONS) if e.name == name)


def scale_config(e: EncConfig, c: M12Config) -> EncConfig:
    return replace(e, embed_dim=c.embed_dim, width=c.width, lambda_mask=e.lambda_mask * c.lambda_mask, lambda_phys=e.lambda_phys * c.lambda_phys,
                   lambda_temp=e.lambda_temp * c.lambda_temp, lambda_align=e.lambda_align * c.lambda_align)


# ============================================================================== sequences (NumPy)
def m11_window_config(c: M12Config) -> M11Config:
    return replace(M11Config(), seq_len=c.seq_len, nfl_windows_per_athlete_demand=c.nfl_windows_per_athlete_demand, nfl_windows_max=c.nfl_windows_max,
                   seed=c.seed)


def build_store(root: Path, combine_frames_by_family: Sequence[pl.DataFrame], windows: pl.DataFrame, positions: dict[int, str]) -> rep.SequenceStore:
    Xs, Ms, meta = [], [], []
    for fr in combine_frames_by_family:
        x, m, mt = rep.combine_sequence_arrays(fr, STORE_LEN)
        Xs += x
        Ms += m
        meta += mt
    x, m, mt = rep.nfl_sequence_arrays(windows)
    Xs += x
    Ms += m
    meta += mt
    return rep.SequenceStore.write(root, Xs, Ms, meta, positions, STORE_LEN)


def physics_targets(X: np.ndarray, M: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """(N, 6) physical targets in physical units, NaN where not supported (heading change needs ≥ 50 % valid heading)."""
    out = np.full((X.shape[0], len(PHYS)), np.nan, np.float32)
    S = rep.SCALE
    for b in range(0, X.shape[0], chunk):
        x, m = np.asarray(X[b: b + chunk], np.float32), np.asarray(M[b: b + chunk])
        for i in range(x.shape[0]):
            mi = m[i]
            if mi.sum() < 3:
                continue
            xi = x[i][mi]
            sp, at, an, yaw, dt, hv = (xi[:, k] for k in range(6))
            out[b + i] = [sp.max() * S["speed"], (sp[-1] - sp[0]) * S["speed"], at.mean() * S["a_tan"], np.abs(an).max() * S["a_n"],
                          abs(float((yaw * S["yaw"] * dt * S["dt"]).sum())) if hv.mean() >= 0.5 else np.nan, float((dt * S["dt"]).sum())]
    return out


def state_ids(X: np.ndarray, M: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Physical state of a sequence for matched-state alignment: start-speed bin (1 yd/s, capped at 8) × accelerating
    (mean a_tan ≥ 1 yd/s² over the first second). −1 when undefined."""
    out = np.full(X.shape[0], -1, np.int64)
    for b in range(0, X.shape[0], chunk):
        x, m = np.asarray(X[b: b + chunk], np.float32), np.asarray(M[b: b + chunk])
        for i in range(x.shape[0]):
            idx = np.flatnonzero(m[i])
            if idx.size < 10:
                continue
            v0 = x[i, idx[0], 0] * rep.SCALE["speed"]
            acc = x[i, idx[:10], 1].mean() * rep.SCALE["a_tan"] >= 1.0
            out[b + i] = int(min(int(v0), 8)) * 2 + int(acc)
    return out


def category_ids(meta: pl.DataFrame) -> np.ndarray:
    cats = sorted(set(CATEGORY.values()))
    return np.array([cats.index(CATEGORY[k]) if k in CATEGORY else -1 for k in meta["kind"].to_list()], np.int64)


def split_indices(meta: pl.DataFrame, folds_by_athlete: dict[int, int], held_out: int | None, val_share: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, validation, held-out) sequence indices. Inductive: held-out athletes' sequences of BOTH domains are removed
    before training; validation = sequences of a seeded share of the remaining ATHLETES (grouped, never random frames)."""
    ids = meta["nfl_id"].to_numpy()
    f = np.array([folds_by_athlete.get(int(i), -1) for i in ids])
    held = (f == held_out) if held_out is not None else np.zeros(ids.size, bool)
    rest = np.unique(ids[~held])
    rng = np.random.default_rng(seed)
    val_ath = set(rng.choice(rest, max(1, int(val_share * rest.size)), replace=False).tolist()) if rest.size else set()
    val = ~held & np.isin(ids, list(val_ath))
    return np.flatnonzero(~held & ~val), np.flatnonzero(val), np.flatnonzero(held)


# ============================================================================== torch model
if TORCH:
    class _GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, k):
            ctx.k = k
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.k * g, None

    class SharedEncoder(nn.Module):
        def __init__(self, e: EncConfig, n_phys: int = len(PHYS)):
            super().__init__()
            self.e = e
            mk = (lambda: rep.ConvEncoder(len(rep.CHANNELS), e.width, e.embed_dim, e.layers)) if e.encoder == "conv" else \
                 (lambda: rep.TransformerEncoder(len(rep.CHANNELS), e.width, e.embed_dim, 2, 4, max_len=STORE_LEN))
            self.encoder = mk()
            self.recon = nn.Linear(e.embed_dim, len(rep.CHANNELS))
            self.phys = nn.Sequential(nn.Linear(e.embed_dim, e.embed_dim), nn.GELU(), nn.Linear(e.embed_dim, n_phys))
            self.proj = nn.Sequential(nn.Linear(e.embed_dim, e.embed_dim), nn.GELU(), nn.Linear(e.embed_dim, e.embed_dim))
            self.pred = nn.Sequential(nn.Linear(e.embed_dim, e.embed_dim), nn.GELU(), nn.Linear(e.embed_dim, e.embed_dim))
            self.domain = nn.Sequential(nn.Linear(e.embed_dim, e.embed_dim), nn.GELU(), nn.Linear(e.embed_dim, 1))
            self.target = mk() if e.temporal == "jepa" else None
            if self.target is not None:
                self.target.load_state_dict(self.encoder.state_dict())
                for p in self.target.parameters():
                    p.requires_grad_(False)

        def enc(self, x, m, target: bool = False):
            """Encoder call with padded / invalid frames zeroed first: a convolution's receptive field would otherwise let
            padding values reach valid positions."""
            net = self.target if target else self.encoder
            return net(x.masked_fill(~m.unsqueeze(-1), 0.0), m)

        def embed(self, x, m):
            return self.enc(x, m)[1]

        @torch.no_grad()
        def ema_update(self):
            if self.target is not None:
                for pt, po in zip(self.target.parameters(), self.encoder.parameters()):
                    pt.mul_(self.e.ema).add_(po.detach(), alpha=1 - self.e.ema)


def _supcon(z: "torch.Tensor", labels: "torch.Tensor", domain: "torch.Tensor", temperature: float) -> "torch.Tensor":
    """Supervised contrastive loss whose POSITIVES are sequences of the OTHER domain with the same label (state or
    category). Anchors without a cross-domain positive in the batch are skipped."""
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & (domain[:, None] != domain[None, :]) & (labels[:, None] >= 0) & ~eye
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0.0
    logp = sim.masked_fill(eye, -1e9).log_softmax(1)
    return -(logp * pos.float()).sum(1)[has].div(pos.float().sum(1)[has]).mean()


def _mmd(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    if a.shape[0] < 2 or b.shape[0] < 2:
        return a.sum() * 0.0
    x = torch.cat([a, b])
    d = torch.cdist(x, x) ** 2
    h = d.detach().median().clamp_min(1e-6)
    k = torch.exp(-d / h)
    n = a.shape[0]
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def batch_loss(model: Any, x, m, phys, phys_ok, state, cat, dom, gen, grl: float = 1.0) -> tuple["torch.Tensor", dict[str, float]]:
    e = model.e
    parts: dict[str, "torch.Tensor"] = {}
    seq, emb = model.enc(x, m)
    if e.lambda_mask > 0:
        hid = rep._span_mask(m, e.mask_ratio, e.span, gen)
        sq, _ = model.enc(x, m & ~hid)
        w = hid.float().unsqueeze(-1)
        parts["mask"] = ((model.recon(sq) - x) ** 2 * w).sum() / (w.sum() * x.shape[-1]).clamp_min(1.0)
    if e.lambda_phys > 0:
        p = model.phys(emb)
        okf = phys_ok.float()
        parts["phys"] = (((p - phys) ** 2) * okf).sum() / okf.sum().clamp_min(1.0)
    if e.lambda_temp > 0 and e.temporal == "contrastive":
        (x1, m1), (x2, m2) = rep._crop(x, m, 0.8, gen), rep._crop(x, m, 0.8, gen)
        z1, z2 = F.normalize(model.proj(model.embed(x1, m1)), dim=-1), F.normalize(model.proj(model.embed(x2, m2)), dim=-1)
        lg = z1 @ z2.T / e.temperature
        lab = torch.arange(x.shape[0], device=x.device)
        parts["temp"] = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))
    elif e.lambda_temp > 0 and e.temporal == "jepa":
        hid = rep._span_mask(m, 0.3, e.span, gen)
        ctx = model.pred(model.embed(x, m & ~hid))
        with torch.no_grad():
            tgt = model.enc(x, m, target=True)[1]
        parts["temp"] = (2 - 2 * F.cosine_similarity(ctx, tgt, dim=-1)).mean()
    if e.lambda_align > 0 and e.align != "none":
        if e.align == "matched_state":
            parts["align"] = _supcon(model.proj(emb), state, dom, e.temperature)
        elif e.align == "category":
            parts["align"] = _supcon(model.proj(emb), cat, dom, e.temperature)
        elif e.align == "mmd":
            parts["align"] = _mmd(emb[dom == 0], emb[dom == 1])
        elif e.align == "adversarial":
            logit = model.domain(_GradReverse.apply(emb, grl)).squeeze(-1)
            parts["align"] = F.binary_cross_entropy_with_logits(logit, dom.float())
    lam = {"mask": e.lambda_mask, "phys": e.lambda_phys, "temp": e.lambda_temp, "align": e.lambda_align}
    total = sum(lam[k] * v for k, v in parts.items()) if parts else emb.sum() * 0.0
    return total, {k: float(v.detach()) for k, v in parts.items()}


# ============================================================================== training (resumable, bounded)
def _balanced_batches(idx: np.ndarray, dom: np.ndarray, batch: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Domain-balanced batches: each batch draws half Combine, half NFL (the smaller domain is cycled)."""
    a, b = rng.permutation(idx[dom[idx] == 0]), rng.permutation(idx[dom[idx] == 1])
    if a.size == 0 or b.size == 0:
        p = rng.permutation(idx)
        return [np.sort(p[k: k + batch]) for k in range(0, p.size, batch)]
    half = batch // 2
    nb = int(math.ceil(max(a.size, b.size) / half))
    return [np.sort(np.r_[a[np.arange(k * half, (k + 1) * half) % a.size], b[np.arange(k * half, (k + 1) * half) % b.size]]) for k in range(nb)]


def train(e: EncConfig, X: np.ndarray, M: np.ndarray, P: np.ndarray, state: np.ndarray, cat: np.ndarray, dom: np.ndarray, tr: np.ndarray, va: np.ndarray,
          c: M12Config, device: str, ckpt_root: Path, mode: str, tag: str, smoke: bool, seed: int, nfl_fraction: float = 1.0) -> tuple[Any, pl.DataFrame, dict[str, Any]]:
    """Train or resume one encoder. Epoch checkpoints are immutable shards keyed by (config, train index hash, budget,
    seed, tag); a finished run is restored, never retrained. Early stopping on the validation loss of held-out ATHLETES
    (grouped); hard wall-clock cap ``max_minutes``; gradient accumulation; mixed precision on CUDA; CPU fallback."""
    if not TORCH:
        raise ImportError("PyTorch is not installed: 12F is not feasible in this environment")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    if nfl_fraction < 1.0:
        keep = np.random.default_rng(seed).random(tr.size) < nfl_fraction
        tr = tr[(dom[tr] == 0) | keep]
    sig_cfg = {"enc": asdict(e), "train_hash": config_hash(tr.tolist()), "epochs": c.epochs, "batch": c.batch, "accum": c.accum_steps, "lr": c.lr,
               "seed": seed, "L": int(X.shape[1]), "nfl_fraction": nfl_fraction}
    ck = ModelCheckpoint(ckpt_root, f"enc_{e.name}_{tag}", sig_cfg, mode)
    model = SharedEncoder(e).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c.lr, weight_decay=1e-4)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    # physics targets standardized with TRAINING-sequence constants
    mu, sd = np.nanmean(P[tr], 0), np.nanstd(P[tr], 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    Pz = np.nan_to_num((P - mu) / sd, nan=0.0).astype(np.float32)
    Pok = np.isfinite(P)
    if e.shuffle_phys:
        perm = np.random.default_rng(seed + 99).permutation(P.shape[0])
        Pz, Pok = Pz[perm], Pok[perm]
    curve, start, best, bad, best_state = [], 0, np.inf, 0, None
    have = ck.latest()
    if have is not None:
        ep, st = have
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        curve, best, bad, best_state = st["curve"], st["best"], st["bad"], st["best_state"]
        start = ep + 1
        if st.get("finished"):
            model.load_state_dict(best_state)
            return model.eval(), pl.DataFrame(curve), st["signature"]
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)

    def to(ix):
        t = lambda a, dt=None: torch.from_numpy(np.asarray(a[ix], dt) if dt else np.asarray(a[ix])).to(device)
        return (t(X, np.float32), t(M), t(Pz), t(Pok), t(state), t(cat), t(dom))

    max_batches = 4 if smoke else None
    finished = False
    for epoch in range(start, c.epochs):
        t0 = time.perf_counter()
        model.train()
        sums: dict[str, float] = {}
        nb = 0
        batches = _balanced_batches(tr, dom, c.batch, rng)
        opt.zero_grad(set_to_none=True)
        for k, ix in enumerate(batches):
            if max_batches and nb >= max_batches:
                break
            x, m, p, pk, s_, ca, d_ = to(ix)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss, parts = batch_loss(model, x, m, p, pk, s_, ca, d_, gen)
                loss = loss / c.accum_steps
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
            for ix in _balanced_batches(va, dom, c.batch, np.random.default_rng(seed + 2))[: (4 if smoke else 50)]:
                x, m, p, pk, s_, ca, d_ = to(ix)
                l_, _ = batch_loss(model, x, m, p, pk, s_, ca, d_, vgen)
                vl += float(l_)
                vn += 1
        val = vl / max(vn, 1)
        elapsed = (time.perf_counter() - t_start) / 60
        curve.append({"epoch": epoch, "val_loss": val, "seconds": time.perf_counter() - t0, **{f"train_{k}": v / max(nb, 1) for k, v in sums.items()}})
        if val < best - 1e-5:
            best, bad, best_state = val, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        finished = bad >= c.patience or epoch == c.epochs - 1 or elapsed >= c.max_minutes
        sig = {"config": e.name, "tag": tag, "mode": mode, "checkpoint_dir": str(ck.dir), "best_val": best, "epochs_run": epoch + 1, "finished": finished,
               "early_stopped": bad >= c.patience, "time_capped": elapsed >= c.max_minutes, "nfl_fraction": nfl_fraction}
        ck.save_epoch(epoch, {"model": model.state_dict(), "opt": opt.state_dict(), "curve": curve, "best": best, "bad": bad, "best_state": best_state,
                              "finished": finished, "signature": sig})
        eta = (time.perf_counter() - t_start) / (epoch - start + 1) * (c.epochs - epoch - 1) / 60
        log.info("[12F %s %s] epoch %d/%d val %.4f · elapsed %.1f min · ETA ≤ %.1f min", e.name, tag, epoch + 1, c.epochs, val, elapsed, eta)
        print(f"[12F {e.name} {tag}] epoch {epoch + 1}/{c.epochs} · val {val:.4f} · elapsed {elapsed:.1f} min · ETA ≤ {eta:.1f} min · checkpoint {ck.dir.name}", flush=True)
        if finished:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), pl.DataFrame(curve), {"config": e.name, "tag": tag, "mode": mode, "checkpoint_dir": str(ck.dir), "best_val": best,
                                               "epochs_run": len(curve), "finished": True, "nfl_fraction": nfl_fraction}


def embed(model: Any, X: np.ndarray, M: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for b in range(0, X.shape[0], batch):
            out.append(model.embed(torch.from_numpy(np.asarray(X[b: b + batch], np.float32)).to(device), torch.from_numpy(np.asarray(M[b: b + batch])).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def random_frozen(e: EncConfig, device: str, seed: int) -> Any:
    torch.manual_seed(seed)
    return SharedEncoder(e).to(device).eval()


# ============================================================================== evaluation helpers (NumPy)
def evaluate(name: str, regime: str, E: np.ndarray, meta: pl.DataFrame, phys: pl.DataFrame, folds_by_athlete: dict[int, int], c: M12Config, seed: int
             ) -> tuple[dict[str, Any], pl.DataFrame]:
    """Milestone 11 label-free evaluation row (within / cross-domain retrieval, matched retrieval, domain AUC / MMD,
    athlete-grouped probes) tagged with the training regime."""
    m11c = replace(M11Config(), n_boot=c.n_boot, outer_folds=c.outer_folds)
    row, pr = rep.evaluation_row(name, regime, E, meta, phys, folds_by_athlete, m11c, seed)
    return row, pr


def shuffled_identity_null(E: np.ndarray, meta: pl.DataFrame, n_perm: int, seed: int) -> np.ndarray:
    """Cross-domain retrieval AUC when NFL athlete identities are permuted within position (chance reference)."""
    rng = np.random.default_rng(seed)
    out = []
    nfl = meta["domain"].to_numpy() == "nfl"
    ids, pos = meta["nfl_id"].to_numpy().copy(), meta["position_group"].fill_null("?").to_numpy()
    uniq = {}
    for i, p in zip(ids[nfl], pos[nfl]):
        uniq.setdefault(p, set()).add(int(i))
    for _ in range(n_perm):
        mp = {}
        for p, s in uniq.items():
            s = sorted(s)
            mp |= dict(zip(s, rng.permutation(s).tolist()))
        new = ids.copy()
        new[nfl] = [mp[int(i)] for i in ids[nfl]]
        r = rep.cross_domain_retrieval(E, meta.with_columns(pl.Series("nfl_id", new)), 0, seed)
        out.append(r["auc"])
    return np.asarray(out, float)


def athlete_mean_embedding(E: np.ndarray, meta: pl.DataFrame, kinds: Sequence[str] = ("forty",), domain: str = "combine") -> pl.DataFrame:
    return rep.athlete_embedding_table(E, meta, domain, kinds, prefix="e")


def embedding_block(analysis_ids: np.ndarray, fold_of: dict[int, int], by_fold: dict[int, pl.DataFrame], k: int = 8) -> ae.BlockFn:
    """INDUCTIVE embeddings for the ladder: for the held-out fold f (inferred from the train mask), every athlete's mean
    Combine embedding from the encoder trained WITHOUT fold f, reduced by a PCA fitted on the training rows (≤ k dims)."""
    folds = np.array([fold_of.get(int(i), -1) for i in analysis_ids])

    def block(train: np.ndarray) -> pl.DataFrame:
        f = int(np.unique(folds[~train])[0])
        t = by_fold.get(f)
        if t is None:
            return pl.DataFrame({f"emb{j}": np.full(analysis_ids.size, np.nan) for j in range(k)})
        cols = [x for x in t.columns if x != "nfl_id"]
        row = {int(i): r for r, i in enumerate(t["nfl_id"].to_list())}
        T = t.select(cols).to_numpy().astype(float)
        Z = np.array([T[row[int(i)]] if int(i) in row else np.full(len(cols), np.nan) for i in analysis_ids])
        ok = train & np.isfinite(Z).all(1)
        if ok.sum() < k + 5:
            return pl.DataFrame({f"emb{j}": np.full(analysis_ids.size, np.nan) for j in range(k)})
        mu = Z[ok].mean(0)
        _, _, vt = np.linalg.svd(Z[ok] - mu, full_matrices=False)
        S = (Z - mu) @ vt[:k].T
        return pl.DataFrame({f"emb{j}": S[:, j] for j in range(S.shape[1])})
    return block


def tcn_by_fold(rep_table: pl.DataFrame, family: str = "TCN") -> dict[int, pl.DataFrame]:
    """Milestone 06 fold-fitted TCN-B representations (``transfer_attempt_representations_outer``: scheme, family, fold,
    seq_id, nfl_id, dim, value, role) → per fold, athlete mean (all roles: the fold-f encoder embeds every athlete)."""
    t = rep_table.filter(pl.col("family") == family)
    out = {}
    for (f,), g in t.partition_by(["fold"], as_dict=True).items():
        w = g.group_by("nfl_id", "dim").agg(pl.col("value").mean()).pivot(on="dim", index="nfl_id", values="value")
        out[int(f)] = w.rename({k: f"t{k}" for k in w.columns if k != "nfl_id"})
    return out


__all__ = ["ABLATIONS", "CATEGORY", "EncConfig", "FAMILIES", "INDUCTIVE_CONFIGS", "PHYS", "PRIMARY", "STORE_LEN", "TORCH", "athlete_mean_embedding",
           "batch_loss", "build_store", "by_name", "category_ids", "embed", "embedding_block", "evaluate", "m11_window_config", "physics_targets",
           "random_frozen", "scale_config", "shuffled_identity_null", "split_indices", "state_ids", "tcn_by_fold", "train"]
