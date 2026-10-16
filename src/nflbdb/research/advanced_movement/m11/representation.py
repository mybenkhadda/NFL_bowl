"""11F — shared Combine–NFL movement representation (high-risk, high-novelty).

Input (``build_sequences``): Combine attempts of EVERY drill and NFL episode windows in ONE format, written once to an
immutable on-disk store (memory-mapped ``.npy`` + metadata Parquet; directory named by a content hash):

    channels  speed/10 · a_tan/5 · a_n/5 · yaw_rate[rad/s]/3 (0 where the heading is invalid) · Δt/0.1 · heading-valid flag
    mask      True on real, valid frames
    meta      seq index, domain (combine · nfl), nfl_id, kind (drill family / NFL demand), position, season

Scaling constants are FIXED physical units (no statistic of any athlete is used), so the input carries no leakage.

Encoders: ``conv`` (dilated residual 1-D CNN, masked mean pooling) · ``transformer`` (2 layers, learned positions,
padding mask). Objectives: ``masked`` (span masking, reconstruct masked valid frames) · ``contrastive`` (two augmented
crops of the same sequence, InfoNCE) · ``jepa`` (context view → predictor → EMA target encoder's pooled embedding of the
full view; a JEPA-style latent prediction, simplified). Hyperparameter budget: the 6 encoder × objective
configurations at ONE size each (``CONFIGS``); no further search.

Training modes (declared per experiment, stored with every embedding):
    transductive   unlabeled pretraining on every sequence (athlete identity and NFL labels never used). Allowed for the
                   label-free diagnostics (retrieval, probes, domain shift) — reported as TRANSDUCTIVE.
    inductive      one encoder per outer fold, trained without the held-out athletes' sequences (both domains); the only
                   mode used for held-out NFL transfer.

Evaluation never trusts a picture: within-domain retrieval (repeat attempts), cross-domain athlete retrieval AUC,
physically matched retrieval, physical probes (athlete-grouped CV R²), domain-classifier AUC and MMD, against a random
(untrained) encoder, an engineered summary baseline and — for the forty — the original TCN-B representations.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ....data.utils import get_logger, utc_now
from .checkpoint import ModelCheckpoint, atomic_json, sha256
from .combine_drills import longest_segment
from .config import M11Config, Progress, config_hash

log = get_logger(__name__)
NAN = float("nan")
CHANNELS = ("speed", "a_tan", "a_n", "yaw", "dt", "heading_valid")
SCALE = {"speed": 10.0, "a_tan": 5.0, "a_n": 5.0, "yaw": 3.0, "dt": 0.1}
COMBINE_MAX_LEN = 120

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:                                            # the notebook reports NOT_FEASIBLE
    torch, nn, F, TORCH = None, None, None, False


@dataclass(frozen=True)
class RepConfig:
    encoder: str            # conv · transformer
    objective: str          # masked · contrastive · jepa
    embed_dim: int = 64
    width: int = 64
    layers: int = 4
    heads: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mask_ratio: float = 0.15
    span: int = 4
    temperature: float = 0.1
    ema: float = 0.99
    crop: float = 0.8

    @property
    def name(self) -> str:
        return f"{self.encoder}_{self.objective}"


CONFIGS = tuple(RepConfig(e, o) for e in ("conv", "transformer") for o in ("masked", "contrastive", "jepa"))


# ============================================================================== sequences (NumPy, no torch)
def _channels(sp: np.ndarray, at: np.ndarray, an: np.ndarray, yaw_deg: np.ndarray, dt: np.ndarray, hv: np.ndarray) -> np.ndarray:
    yaw = np.where(hv, np.radians(np.nan_to_num(yaw_deg)), 0.0)
    X = np.column_stack([sp / SCALE["speed"], at / SCALE["a_tan"], an / SCALE["a_n"], yaw / SCALE["yaw"], dt / SCALE["dt"], hv.astype(float)])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def combine_sequence_arrays(frames: pl.DataFrame, max_len: int = COMBINE_MAX_LEN) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """Every Combine attempt (longest valid segment, from 0.5 s before onset when recorded, ≤ ``max_len`` frames)."""
    Xs, Ms, meta = [], [], []
    for (sid,), g in frames.partition_by(["seq_id"], as_dict=True, maintain_order=True).items():
        f = longest_segment(g)
        if f.height < 10:
            continue
        sp = f["speed"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        on = np.flatnonzero(sp >= 1.0)
        start = max(0, int(on[0]) - 5) if on.size else 0
        f = f[start: start + max_len]
        sp = f["speed"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        at = f["a_tan"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        yaw = f["yaw_rate"].cast(pl.Float64).fill_null(np.nan).to_numpy() if "yaw_rate" in f.columns else np.zeros(f.height)
        an = f["a_lat"].cast(pl.Float64).fill_null(np.nan).to_numpy() if "a_lat" in f.columns else sp * np.radians(np.nan_to_num(yaw))
        hv = f["heading_valid"].fill_null(False).to_numpy() if "heading_valid" in f.columns else np.isfinite(yaw)
        t = f["t_s"].to_numpy().astype(float)
        dt = np.r_[0.1, np.diff(t)]
        Xs.append(_channels(sp, at, an, yaw, dt, hv))
        Ms.append(np.isfinite(sp) & np.isfinite(at))
        meta.append({"seq_key": sid, "domain": "combine", "nfl_id": int(g["nfl_id"][0]), "kind": str(g["family"][0]), "season": None, "n_frames": f.height})
    return Xs, Ms, meta


def nfl_sequence_arrays(windows: pl.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    Xs, Ms, meta = [], [], []
    for (wid,), g in windows.sort("window_id", "frame").partition_by(["window_id"], as_dict=True, maintain_order=True).items():
        hv = g["heading_valid"].to_numpy()
        Xs.append(_channels(g["speed"].to_numpy().astype(float), g["a_tan"].to_numpy().astype(float), g["a_n"].to_numpy().astype(float),
                            g["yaw_rate"].to_numpy().astype(float), g["dt"].to_numpy().astype(float), hv))      # yaw_rate in deg/s, as in the frames
        Ms.append(g["valid"].to_numpy() & np.isfinite(g["speed"].to_numpy()))
        meta.append({"seq_key": wid, "domain": "nfl", "nfl_id": int(g["nfl_id"][0]), "kind": str(g["demand_id"][0]), "season": int(g["season"][0]), "n_frames": g.height})
    return Xs, Ms, meta


class SequenceStore:
    """Immutable on-disk store: ``X.npy`` (N, L, C) float32, ``M.npy`` (N, L) bool, ``meta.parquet``; opened memory-mapped."""

    def __init__(self, root: Path, key: str):
        self.dir = Path(root) / key

    @staticmethod
    def write(root: Path, Xs: Sequence[np.ndarray], Ms: Sequence[np.ndarray], meta: Sequence[dict[str, Any]], positions: dict[int, str], L: int) -> "SequenceStore":
        key = config_hash({"n": len(Xs), "L": L, "keys": [m["seq_key"] for m in meta][:2000], "n_frames": int(sum(x.shape[0] for x in Xs))})
        st = SequenceStore(root, key)
        if (st.dir / "COMPLETE.json").is_file():
            return st
        st.dir.mkdir(parents=True, exist_ok=True)
        N, C = len(Xs), len(CHANNELS)
        X = np.lib.format.open_memmap(st.dir / f".X.{key}.tmp.npy", mode="w+", dtype=np.float32, shape=(N, L, C))
        M = np.lib.format.open_memmap(st.dir / f".M.{key}.tmp.npy", mode="w+", dtype=np.bool_, shape=(N, L))
        for i, (x, m) in enumerate(zip(Xs, Ms)):
            k = min(L, x.shape[0])
            X[i, :k], M[i, :k] = x[:k], m[:k]
            X[i, k:], M[i, k:] = 0.0, False
        X.flush()
        M.flush()
        del X, M
        import os

        os.replace(st.dir / f".X.{key}.tmp.npy", st.dir / "X.npy")
        os.replace(st.dir / f".M.{key}.tmp.npy", st.dir / "M.npy")
        pl.DataFrame([{**m, "idx": i, "position_group": positions.get(m["nfl_id"])} for i, m in enumerate(meta)],
                     schema_overrides={"season": pl.Int64}).write_parquet(st.dir / "meta.parquet")
        atomic_json({"key": key, "n": N, "L": L, "channels": list(CHANNELS), "scale": SCALE, "written_at": utc_now(),
                     "sha_meta": sha256(st.dir / "meta.parquet")}, st.dir / "COMPLETE.json")
        return st

    def open(self) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        if not (self.dir / "COMPLETE.json").is_file():
            raise FileNotFoundError(f"sequence store {self.dir} is incomplete")
        return np.load(self.dir / "X.npy", mmap_mode="r"), np.load(self.dir / "M.npy", mmap_mode="r"), pl.read_parquet(self.dir / "meta.parquet")


def engineered_features(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Baseline representation: per-channel masked mean, SD, max, min of the 4 kinematic channels + valid length (17 dims)."""
    out = []
    for i in range(X.shape[0]):
        m = M[i]
        x = X[i][m][:, :4] if m.any() else np.zeros((1, 4), np.float32)
        out.append(np.r_[x.mean(0), x.std(0), x.max(0), x.min(0), m.sum() / X.shape[1]])
    return np.asarray(out, np.float32)


def physical_scalars(X: np.ndarray, M: np.ndarray) -> pl.DataFrame:
    """Probe targets computed from the raw channels (physical units restored)."""
    rows = []
    for i in range(X.shape[0]):
        m = M[i]
        x = X[i][m] if m.any() else np.zeros((1, X.shape[2]), np.float32)
        rows.append({"idx": i, "max_speed": float(x[:, 0].max() * SCALE["speed"]), "mean_a_tan": float(x[:, 1].mean() * SCALE["a_tan"]),
                     "max_abs_a_n": float(np.abs(x[:, 2]).max() * SCALE["a_n"]), "abs_heading_change_rad": float(np.abs((x[:, 3] * SCALE["yaw"]).sum() * 0.1))})
    return pl.DataFrame(rows)


# ============================================================================== torch models
if TORCH:
    class ConvEncoder(nn.Module):
        def __init__(self, c_in: int, width: int, d: int, layers: int = 4):
            super().__init__()
            self.inp = nn.Conv1d(c_in + 1, width, 1)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, 5, padding=2 * 2 ** i, dilation=2 ** i), nn.GELU(),
                                                       nn.Conv1d(width, width, 1)) for i in range(layers)])
            self.norms = nn.ModuleList([nn.GroupNorm(1, width) for _ in range(layers)])
            self.out = nn.Conv1d(width, d, 1)

        def forward(self, x: "torch.Tensor", m: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            h = self.inp(torch.cat([x, m.unsqueeze(-1).float()], -1).transpose(1, 2))
            for b, n in zip(self.blocks, self.norms):
                h = h + b(n(h))
            seq = self.out(h).transpose(1, 2)                      # (B, L, d)
            w = m.float().unsqueeze(-1)
            pooled = (seq * w).sum(1) / w.sum(1).clamp_min(1.0)
            return seq, pooled

    class TransformerEncoder(nn.Module):
        def __init__(self, c_in: int, width: int, d: int, layers: int = 2, heads: int = 4, max_len: int = 256):
            super().__init__()
            self.inp = nn.Linear(c_in + 1, width)
            self.pos = nn.Parameter(torch.zeros(1, max_len, width))
            layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.out = nn.Linear(width, d)

        def forward(self, x: "torch.Tensor", m: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            h = self.inp(torch.cat([x, m.unsqueeze(-1).float()], -1)) + self.pos[:, : x.shape[1]]
            pad = ~m
            pad = pad & ~(pad.all(1, keepdim=True))                # a fully padded row keeps its first position
            seq = self.out(self.enc(h, src_key_padding_mask=pad))
            w = m.float().unsqueeze(-1)
            return seq, (seq * w).sum(1) / w.sum(1).clamp_min(1.0)

    class RepModel(nn.Module):
        def __init__(self, rc: RepConfig, c_in: int = len(CHANNELS)):
            super().__init__()
            self.rc = rc
            mk = (lambda: ConvEncoder(c_in, rc.width, rc.embed_dim, rc.layers)) if rc.encoder == "conv" else \
                 (lambda: TransformerEncoder(c_in, rc.width, rc.embed_dim, max(2, rc.layers // 2), rc.heads))
            self.encoder = mk()
            self.recon = nn.Linear(rc.embed_dim, c_in)
            self.proj = nn.Sequential(nn.Linear(rc.embed_dim, rc.embed_dim), nn.GELU(), nn.Linear(rc.embed_dim, rc.embed_dim))
            self.pred = nn.Sequential(nn.Linear(rc.embed_dim, rc.embed_dim), nn.GELU(), nn.Linear(rc.embed_dim, rc.embed_dim))
            self.target = mk() if rc.objective == "jepa" else None
            if self.target is not None:
                self.target.load_state_dict(self.encoder.state_dict())
                for p in self.target.parameters():
                    p.requires_grad_(False)

        def embed(self, x, m):
            return self.encoder(x, m)[1]

        @torch.no_grad()
        def ema_update(self):
            if self.target is not None:
                for pt, po in zip(self.target.parameters(), self.encoder.parameters()):
                    pt.mul_(self.rc.ema).add_(po.detach(), alpha=1 - self.rc.ema)


def _span_mask(m: "torch.Tensor", ratio: float, span: int, gen: "torch.Generator") -> "torch.Tensor":
    B, L = m.shape
    starts = torch.rand(B, L, generator=gen, device="cpu") < ratio / max(1, span)
    hid = torch.zeros(B, L, dtype=torch.bool)
    for s in range(span):
        hid[:, s:] |= starts[:, : L - s]
    return hid.to(m.device) & m


def _crop(x: "torch.Tensor", m: "torch.Tensor", frac: float, gen: "torch.Generator") -> tuple["torch.Tensor", "torch.Tensor"]:
    B, L, C = x.shape
    k = max(4, int(L * frac))
    off = torch.randint(0, L - k + 1, (B,), generator=gen)
    idx = (off.unsqueeze(1) + torch.arange(k)).to(x.device)
    xc = torch.gather(x, 1, idx.unsqueeze(-1).expand(B, k, C))
    mc = torch.gather(m, 1, idx)
    noise = 0.02 * torch.randn(xc.shape, generator=gen).to(x.device)
    return F.pad(xc + noise * mc.unsqueeze(-1), (0, 0, 0, L - k)), F.pad(mc, (0, L - k), value=False)


def loss_fn(model: "RepModel", x: "torch.Tensor", m: "torch.Tensor", gen: "torch.Generator") -> "torch.Tensor":
    rc = model.rc
    if rc.objective == "masked":
        hid = _span_mask(m, rc.mask_ratio, rc.span, gen)
        xin = x.masked_fill(hid.unsqueeze(-1), 0.0)
        seq, _ = model.encoder(xin, m & ~hid)
        rec = model.recon(seq)
        w = hid.float().unsqueeze(-1)
        return ((rec - x) ** 2 * w).sum() / (w.sum() * x.shape[-1]).clamp_min(1.0)
    if rc.objective == "contrastive":
        (x1, m1), (x2, m2) = _crop(x, m, rc.crop, gen), _crop(x, m, rc.crop, gen)
        z1 = F.normalize(model.proj(model.embed(x1, m1)), dim=-1)
        z2 = F.normalize(model.proj(model.embed(x2, m2)), dim=-1)
        logits = z1 @ z2.T / rc.temperature
        lab = torch.arange(x.shape[0], device=x.device)
        return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
    hid = _span_mask(m, max(rc.mask_ratio, 0.3), rc.span, gen)
    ctx = model.pred(model.embed(x.masked_fill(hid.unsqueeze(-1), 0.0), m & ~hid))
    with torch.no_grad():
        tgt = model.target(x, m)[1]
    return (2 - 2 * F.cosine_similarity(ctx, tgt, dim=-1)).mean()


# ============================================================================== training
def train_encoder(rc: RepConfig, X: np.ndarray, M: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, c: M11Config, device: str, ckpt_root: Path,
                  mode: str, tag: str, smoke: bool, seed: int) -> tuple[Any, pl.DataFrame, dict[str, Any]]:
    """Train (or resume) one encoder. Checkpoint shards are keyed by configuration, training indices and tag; a finished run
    is restored, never retrained. Returns (model, curve, signature)."""
    if not TORCH:
        raise ImportError("PyTorch is not installed: 11F is NOT_FEASIBLE in this environment")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    ck = ModelCheckpoint(ckpt_root, f"rep_{rc.name}_{tag}", {"rc": asdict(rc), "train_hash": config_hash(train_idx.tolist()), "epochs": c.rep_epochs,
                                                             "batch": c.rep_batch, "lr": c.rep_lr, "seed": seed, "L": int(X.shape[1])}, mode)
    model = RepModel(rc).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rc.lr, weight_decay=rc.weight_decay)
    use_amp = device == "cuda"
    scaler = (torch.amp.GradScaler("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()) if use_amp else None
    curve, start, best, bad, best_state = [], 0, np.inf, 0, None
    have = ck.latest()
    if have is not None:
        e, st = have
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        curve, best, bad, best_state = st["curve"], st["best"], st["bad"], st["best_state"]
        start = e + 1
        log.info("[m11.rep] %s resumed after epoch %d", rc.name, e)
        if st.get("finished"):
            model.load_state_dict(best_state)
            return model.eval(), pl.DataFrame(curve), st["signature"]
    max_batches = 5 if smoke else None
    prog = Progress(c.rep_epochs - start, f"11F {rc.name} {tag}", every_s=20)
    finished = False
    for epoch in range(start, c.rep_epochs):
        t0 = time.perf_counter()
        model.train()
        perm = np.random.default_rng(seed + epoch).permutation(train_idx)
        tr_loss, nb = 0.0, 0
        for b in range(0, perm.size, c.rep_batch):
            if max_batches and nb >= max_batches:
                break
            ix = np.sort(perm[b: b + c.rep_batch])
            if ix.size < 4:
                continue
            x, m = torch.from_numpy(np.asarray(X[ix])).to(device), torch.from_numpy(np.asarray(M[ix])).to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model, x, m, gen)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            model.ema_update()
            tr_loss += float(loss.detach())
            nb += 1
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            vgen = torch.Generator().manual_seed(seed + 1)
            for b in range(0, val_idx.size, c.rep_batch):
                if max_batches and vn >= max_batches:
                    break
                ix = val_idx[b: b + c.rep_batch]
                if ix.size < 4:
                    continue
                vl += float(loss_fn(model, torch.from_numpy(np.asarray(X[ix])).to(device), torch.from_numpy(np.asarray(M[ix])).to(device), vgen))
                vn += 1
        val = vl / max(vn, 1)
        curve.append({"epoch": epoch, "train_loss": tr_loss / max(nb, 1), "val_loss": val, "seconds": time.perf_counter() - t0})
        if val < best - 1e-5:
            best, bad, best_state = val, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        finished = bad >= c.rep_patience or epoch == c.rep_epochs - 1
        sig = {"config": rc.name, "tag": tag, "mode": mode, "checkpoint_dir": str(ck.dir), "best_val": best, "epochs_run": epoch + 1, "finished": finished,
               "early_stopped": bad >= c.rep_patience}
        ck.save_epoch(epoch, {"model": model.state_dict(), "opt": opt.state_dict(), "curve": curve, "best": best, "bad": bad, "best_state": best_state,
                              "finished": finished, "signature": sig})
        prog.step(note=f"val {val:.4f}")
        if finished:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), pl.DataFrame(curve), {"config": rc.name, "tag": tag, "mode": mode, "checkpoint_dir": str(ck.dir), "best_val": best,
                                               "epochs_run": len(curve), "finished": True}


def embed_all(model: Any, X: np.ndarray, M: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for b in range(0, X.shape[0], batch):
            x, m = torch.from_numpy(np.asarray(X[b: b + batch])).to(device), torch.from_numpy(np.asarray(M[b: b + batch])).to(device)
            out.append(model.embed(x, m).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def random_encoder(rc: RepConfig, device: str, seed: int) -> Any:
    torch.manual_seed(seed)
    return RepModel(rc).to(device).eval()


# ============================================================================== evaluation (NumPy)
def _cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def _zscore(E: np.ndarray, train: np.ndarray | None = None) -> np.ndarray:
    ref = E if train is None else E[train]
    return (E - ref.mean(0)) / np.maximum(ref.std(0), 1e-9)


def within_domain_retrieval(E: np.ndarray, meta: pl.DataFrame) -> dict[str, Any]:
    """Combine attempts of athletes with ≥ 2 attempts of the same drill family: top-1 accuracy and AUC of the athlete's
    other attempt among all other attempts of that family."""
    c = meta.with_row_index("_r").filter(pl.col("domain") == "combine")
    top1, aucs = [], []
    for (fam,), g in c.partition_by(["kind"], as_dict=True).items():
        ids, rows = g["nfl_id"].to_numpy(), g["_r"].to_numpy()
        if rows.size < 5:
            continue
        S = _cos(E[rows], E[rows])
        np.fill_diagonal(S, -np.inf)
        for i in range(rows.size):
            same = (ids == ids[i])
            same[i] = False
            if not same.any():
                continue
            top1.append(bool(same[int(np.argmax(S[i]))]))
            pos, neg = S[i][same], S[i][~same & np.isfinite(S[i])]
            if neg.size:
                aucs.append(float((pos.max() > neg).mean()))
    return {"n_queries": len(top1), "top1": float(np.mean(top1)) if top1 else NAN, "auc": float(np.mean(aucs)) if aucs else NAN}


def cross_domain_retrieval(E: np.ndarray, meta: pl.DataFrame, n_boot: int, seed: int, combine_kinds: Sequence[str] | None = None) -> dict[str, Any]:
    """Athletes in BOTH domains: mean Combine embedding queried against mean NFL embeddings of athletes of the same position;
    AUC = share of other athletes ranked below the athlete's own NFL embedding (chance 0.5). Athlete-bootstrap CI."""
    m = meta.with_row_index("_r")
    cm = m.filter(pl.col("domain") == "combine")
    if combine_kinds is not None:
        cm = cm.filter(pl.col("kind").is_in(list(combine_kinds)))
    nm = m.filter(pl.col("domain") == "nfl")
    both = sorted(set(cm["nfl_id"].to_list()) & set(nm["nfl_id"].to_list()))
    pos = dict(zip(meta["nfl_id"].to_list(), meta["position_group"].to_list()))
    if len(both) < 10:
        return {"n_athletes": len(both), "auc": NAN, "auc_lo": NAN, "auc_hi": NAN, "top5": NAN}
    mean = lambda d, i: E[d.filter(pl.col("nfl_id") == i)["_r"].to_numpy()].mean(0)
    C = np.stack([mean(cm, i) for i in both])
    N = np.stack([mean(nm, i) for i in both])
    S = _cos(C, N)
    P = np.array([pos.get(i) for i in both])
    aucs, top5 = [], []
    for k in range(len(both)):
        cand = np.flatnonzero(P == P[k])
        others = cand[cand != k]
        if others.size < 2:
            continue
        aucs.append(float((S[k, k] > S[k, others]).mean()))
        top5.append(bool((S[k, others] > S[k, k]).sum() < 5))
    a = np.asarray(aucs)
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, a.size, a.size)].mean() for _ in range(n_boot)] if a.size >= 10 else []
    return {"n_athletes": int(a.size), "auc": float(a.mean()) if a.size else NAN, "auc_lo": float(np.quantile(bs, 0.025)) if bs else NAN,
            "auc_hi": float(np.quantile(bs, 0.975)) if bs else NAN, "top5": float(np.mean(top5)) if top5 else NAN}


def matched_retrieval(E: np.ndarray, meta: pl.DataFrame, phys: pl.DataFrame, k: int = 20, max_queries: int = 2000, seed: int = 0) -> dict[str, Any]:
    """For Combine forty attempts: do the k nearest NFL windows share the query's physics more than random NFL windows?
    Reported: median |Δ max speed| of neighbours vs random, and the share of accelerating demands among neighbours."""
    m = meta.with_row_index("_r").join(phys, left_on="_r", right_on="idx", how="left")
    q = m.filter((pl.col("domain") == "combine") & (pl.col("kind") == "forty"))
    n = m.filter(pl.col("domain") == "nfl")
    if q.height < 5 or n.height < k * 2:
        return {"n_queries": q.height, "neighbour_dspeed": NAN, "random_dspeed": NAN, "neighbour_accel_share": NAN, "base_accel_share": NAN}
    rng = np.random.default_rng(seed)
    q = q.sample(min(max_queries, q.height), seed=seed)
    S = _cos(E[q["_r"].to_numpy()], E[n["_r"].to_numpy()])
    vq, vn = q["max_speed"].to_numpy(), n["max_speed"].to_numpy()
    acc = n["kind"].is_in(["accel_low", "accel_mid", "transition_high", "reaccel_after_brake"]).to_numpy()
    nb = np.argsort(-S, axis=1)[:, :k]
    rnd = rng.integers(0, n.height, nb.shape)
    return {"n_queries": q.height, "neighbour_dspeed": float(np.median(np.abs(vn[nb] - vq[:, None]))), "random_dspeed": float(np.median(np.abs(vn[rnd] - vq[:, None]))),
            "neighbour_accel_share": float(acc[nb].mean()), "base_accel_share": float(acc.mean())}


def probe_r2(E: np.ndarray, meta: pl.DataFrame, phys: pl.DataFrame, folds_by_athlete: dict[int, int], c: M11Config) -> pl.DataFrame:
    """Athlete-grouped cross-validated R² of a ridge probe from the embedding to each physical scalar, per domain."""
    m = meta.with_row_index("_r").join(phys, left_on="_r", right_on="idx", how="left")
    rows = []
    for dom in ("combine", "nfl"):
        d = m.filter(pl.col("domain") == dom)
        if d.height < 50:
            continue
        Z = E[d["_r"].to_numpy()]
        f = np.array([folds_by_athlete.get(i, i % c.outer_folds) for i in d["nfl_id"].to_list()])
        for tgt in ("max_speed", "mean_a_tan", "max_abs_a_n", "abs_heading_change_rad"):
            y = d[tgt].to_numpy().astype(float)
            pred = np.full(y.size, NAN)
            for k in np.unique(f):
                tr, te = f != k, f == k
                Zs = _zscore(Z, tr)
                A = Zs[tr].T @ Zs[tr] + 10.0 * np.eye(Z.shape[1])
                beta = np.linalg.solve(A, Zs[tr].T @ (y[tr] - y[tr].mean()))
                pred[te] = y[tr].mean() + Zs[te] @ beta
            ok = np.isfinite(pred) & np.isfinite(y)
            r2 = 1 - np.mean((y[ok] - pred[ok]) ** 2) / np.var(y[ok]) if ok.sum() > 10 else NAN
            rows.append({"domain": dom, "probe": tgt, "n": int(ok.sum()), "r2_cv": float(r2)})
    return pl.DataFrame(rows)


def domain_shift(E: np.ndarray, meta: pl.DataFrame, max_n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Linear domain-classifier AUC (ridge discriminant, 5-fold by athlete) and RBF-MMD² (median heuristic) on subsamples.
    AUC near 0.5 = domains not separable by a linear read-out; near 1 = fully separable."""
    rng = np.random.default_rng(seed)
    m = meta.with_row_index("_r")
    a = m.filter(pl.col("domain") == "combine")["_r"].to_numpy()
    b = m.filter(pl.col("domain") == "nfl")["_r"].to_numpy()
    if a.size < 20 or b.size < 20:
        return {"domain_auc": NAN, "mmd2": NAN}
    a, b = rng.choice(a, min(max_n, a.size), replace=False), rng.choice(b, min(max_n, b.size), replace=False)
    Z = _zscore(E[np.r_[a, b]])
    y = np.r_[np.zeros(a.size), np.ones(b.size)]
    ids = meta["nfl_id"].to_numpy()[np.r_[a, b]]
    f = np.asarray(ids) % 5
    score = np.zeros(y.size)
    for k in range(5):
        tr, te = f != k, f == k
        A = Z[tr].T @ Z[tr] + 1.0 * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z[tr].T @ (y[tr] - y[tr].mean()))
        score[te] = Z[te] @ w
    order = np.argsort(score)
    ranks = np.empty(y.size)
    ranks[order] = np.arange(1, y.size + 1)
    auc = (ranks[y == 1].sum() - b.size * (b.size + 1) / 2) / (a.size * b.size)
    Za, Zb = Z[: a.size][:500], Z[a.size:][:500]
    D = lambda P, Q: ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    Dab = D(Za, Zb)
    h = np.median(Dab) or 1.0
    k_ = lambda d: np.exp(-d / h)
    mmd = k_(D(Za, Za)).mean() + k_(D(Zb, Zb)).mean() - 2 * k_(Dab).mean()
    return {"domain_auc": float(auc), "mmd2": float(mmd)}


def athlete_embedding_table(E: np.ndarray, meta: pl.DataFrame, domain: str = "combine", kinds: Sequence[str] | None = None, prefix: str = "emb") -> pl.DataFrame:
    """Mean embedding per athlete (wide) for the held-out transfer ladder."""
    m = meta.with_row_index("_r").filter(pl.col("domain") == domain)
    if kinds is not None:
        m = m.filter(pl.col("kind").is_in(list(kinds)))
    rows = []
    for (i,), g in m.partition_by(["nfl_id"], as_dict=True).items():
        v = E[g["_r"].to_numpy()].mean(0)
        rows.append({"nfl_id": int(i), **{f"{prefix}{j}": float(x) for j, x in enumerate(v)}})
    return pl.DataFrame(rows)


def evaluation_row(name: str, mode: str, E: np.ndarray, meta: pl.DataFrame, phys: pl.DataFrame, folds_by_athlete: dict[int, int], c: M11Config, seed: int
                   ) -> tuple[dict[str, Any], pl.DataFrame]:
    w = within_domain_retrieval(E, meta)
    x = cross_domain_retrieval(E, meta, max(200, c.n_boot // 5), seed)
    mr = matched_retrieval(E, meta, phys, seed=seed)
    ds = domain_shift(E, meta, seed=seed)
    pr = probe_r2(E, meta, phys, folds_by_athlete, c)
    row = {"representation": name, "training_mode": mode, "within_top1": w["top1"], "within_auc": w["auc"], "cross_auc": x["auc"], "cross_auc_lo": x["auc_lo"],
           "cross_auc_hi": x["auc_hi"], "cross_top5": x["top5"], "cross_n_athletes": x["n_athletes"], **mr, **ds,
           "probe_r2_median": float(pr["r2_cv"].median()) if pr.height else NAN}
    return row, pr.with_columns(pl.lit(name).alias("representation"))


def save_embeddings(store_dir: Path, name: str, E: np.ndarray, meta: pl.DataFrame, signature: dict[str, Any]) -> Path:
    """Embedding cache: Parquet with the model signature in a sidecar (immutable name = representation + signature hash)."""
    h = config_hash(signature)
    p = Path(store_dir) / f"m11_embeddings_{name}_{h}.parquet"
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        df = meta.select("idx", "seq_key", "domain", "nfl_id", "kind").with_columns([pl.Series(f"e{j}", E[:, j]) for j in range(E.shape[1])])
        tmp = p.with_name(f".{p.name}.{h}.tmp")
        df.write_parquet(tmp)
        import os

        os.replace(tmp, p)
        atomic_json({**signature, "path": p.name, "sha256": sha256(p), "written_at": utc_now()}, p.with_name(p.name + ".meta.json"))
    return p


__all__ = ["CHANNELS", "CONFIGS", "RepConfig", "SCALE", "SequenceStore", "TORCH", "athlete_embedding_table", "combine_sequence_arrays",
           "cross_domain_retrieval", "domain_shift", "embed_all", "engineered_features", "evaluation_row", "matched_retrieval", "nfl_sequence_arrays",
           "physical_scalars", "probe_r2", "random_encoder", "save_embeddings", "train_encoder", "within_domain_retrieval"]
