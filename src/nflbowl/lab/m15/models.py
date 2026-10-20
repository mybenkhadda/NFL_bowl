"""E15.B1 — encoder architectures under one interface (T07). PyTorch is imported lazily (``TORCH`` flag, as in m11).

Common contract (every learned model, ``nn.Module``):
    forward(x, m) → (recon, z)      x: (B, L, 6) bundle-scaled channels, m: (B, L) bool validity
    encode(x, m) → z                (B, d) — the representation that is exported
    domain                          "time" (recon = (B, L, 4) scaled speed, a_tan, a_n, yaw) or "stft" (recon = log-magnitude)
    loss(x, m, recon) → scalar      masked: only valid frames count; yaw only where heading is valid (heading_valid channel)

Inputs never contain an invalid frame's value: signals are multiplied by the validity mask and the mask (and the
heading mask) are given as explicit channels, so a zero is never mistaken for an observation.

Architectures (``ARCHITECTURES`` in config): pca (NumPy, ``PCAEncoder``) · stft_ae (2-D CNN on per-channel |STFT|; phase
sin/cos channels with ``stft_phase``) · lstm_ae / gru_ae (bidirectional recurrent encoder, masked mean pooling;
recurrent decoder from z + learned positions) · tcn_ae (the m11 dilated ``ConvEncoder`` + a dilated conv decoder) ·
dtcn_ae (same encoder, deeper: 6 dilation levels) · masked_tae (tcn_ae trained with random frame masking).
Optional ``wavelet_cnn`` is not built (P2): ``build`` raises a clear error naming it.

Equal budget: ``choose_width`` picks, per architecture, the width whose parameter count is closest to the reference
budget of ``tcn_ae`` at the same d; a count outside ±``param_band`` is reported (``fits_band`` False), never silently used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn

    TORCH = True
except ImportError:                                                  # CPU-only analysis sessions without torch
    TORCH = False

SIGNALS = 4                                                         # speed, a_tan, a_n, yaw
WIDTH_GRID = (16, 24, 32, 48, 64, 96, 128)


def _require_torch() -> None:
    if not TORCH:
        from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked

        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "learned encoders need PyTorch (use the Colab GPU runtime)", missing=["torch"])


# ====================================================================== PCA (NumPy)
class PCAEncoder:
    """Masked-aware PCA baseline: invalid frames imputed with the TRAINING per-frame channel means, flattened (L × 4),
    standardised with training statistics, truncated SVD on training rows."""

    domain = "time"

    def __init__(self, d: int):
        self.d = int(d)

    @staticmethod
    def _flat(x: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(x, float)[..., :SIGNALS].copy()
        hv = (np.asarray(x)[..., 5] > 0.5) & m
        w = np.repeat(m[..., None], SIGNALS, -1)
        w[..., 3] = hv
        return s, w

    def fit(self, x: np.ndarray, m: np.ndarray) -> "PCAEncoder":
        s, w = self._flat(x, m)
        cnt = w.sum(0)
        self.fill = np.where(cnt > 0, (s * w).sum(0) / np.maximum(cnt, 1), 0.0)
        F = np.where(w, s, self.fill).reshape(s.shape[0], -1)
        self.mu, sd = F.mean(0), F.std(0)
        self.sd = np.where(sd > 1e-8, sd, 1.0)
        _, sv, vt = np.linalg.svd((F - self.mu) / self.sd, full_matrices=False)
        self.components = vt[: self.d]
        self.explained = (sv[: self.d] ** 2) / max(float((sv ** 2).sum()), 1e-12)
        self.shape = s.shape[1:]
        return self

    def encode(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        s, w = self._flat(x, m)
        F = np.where(w, s, self.fill).reshape(s.shape[0], -1)
        return ((F - self.mu) / self.sd) @ self.components.T

    def reconstruct(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        z = self.encode(x, m)
        F = (z @ self.components) * self.sd + self.mu
        return F.reshape(-1, *self.shape)

    def n_params(self) -> int:
        return int(self.components.size + self.mu.size + self.sd.size + self.fill.size)

    def state(self) -> dict[str, Any]:
        return {"d": self.d, "fill": self.fill, "mu": self.mu, "sd": self.sd, "components": self.components, "explained": self.explained, "shape": self.shape}

    @classmethod
    def from_state(cls, st: dict[str, Any]) -> "PCAEncoder":
        p = cls(int(st["d"]))
        p.fill, p.mu, p.sd, p.components, p.explained, p.shape = st["fill"], st["mu"], st["sd"], st["components"], st["explained"], tuple(st["shape"])
        return p


# ====================================================================== torch models
if TORCH:
    def _inputs(x: "torch.Tensor", m: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """(signals masked (B, L, 4), validity (B, L) float, heading mask (B, L) float)."""
        mf = m.float()
        hv = (x[..., 5] > 0.5).float() * mf
        s = x[..., :SIGNALS] * mf.unsqueeze(-1)
        s = torch.cat([s[..., :3], (s[..., 3] * hv).unsqueeze(-1)], -1)
        return s, mf, hv

    def time_loss(x: "torch.Tensor", m: "torch.Tensor", recon: "torch.Tensor") -> "torch.Tensor":
        _, mf, hv = _inputs(x, m)
        w = torch.stack([mf, mf, mf, hv], -1)
        err = (recon - x[..., :SIGNALS]) ** 2 * w
        return err.sum() / w.sum().clamp_min(1.0)

    class RNNAE(nn.Module):
        domain = "time"

        def __init__(self, cell: str, d: int, width: int, L: int):
            super().__init__()
            R = nn.LSTM if cell == "lstm" else nn.GRU
            self.enc = R(SIGNALS + 2, width, batch_first=True, bidirectional=True)
            self.to_z = nn.Linear(2 * width, d)
            self.pos = nn.Parameter(torch.zeros(1, L, 8))
            self.dec = R(d + 8, width, batch_first=True)
            self.out = nn.Linear(width, SIGNALS)

        def encode(self, x: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
            s, mf, hv = _inputs(x, m)
            h, _ = self.enc(torch.cat([s, mf.unsqueeze(-1), hv.unsqueeze(-1)], -1))
            w = mf.unsqueeze(-1)
            return self.to_z((h * w).sum(1) / w.sum(1).clamp_min(1.0))

        def forward(self, x: "torch.Tensor", m: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            z = self.encode(x, m)
            L = x.shape[1]
            h, _ = self.dec(torch.cat([z.unsqueeze(1).expand(-1, L, -1), self.pos[:, :L].expand(x.shape[0], -1, -1)], -1))
            return self.out(h), z

        def loss(self, x: "torch.Tensor", m: "torch.Tensor", recon: "torch.Tensor") -> "torch.Tensor":
            return time_loss(x, m, recon)

    class TCNAE(nn.Module):
        """Encoder = m11 ``ConvEncoder`` (dilated residual 1-D conv, masked pooling); decoder = dilated conv stack on
        z broadcast over time + learned positions."""
        domain = "time"

        def __init__(self, d: int, width: int, L: int, layers: int = 4, mask_ratio: float = 0.0):
            super().__init__()
            from nflbdb.research.advanced_movement.m11.representation import ConvEncoder

            self.enc = ConvEncoder(SIGNALS + 1, width, d, layers)
            self.pos = nn.Parameter(torch.zeros(1, L, 8))
            self.dec_in = nn.Conv1d(d + 8, width, 1)
            self.dec = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, 5, padding=2 * 2 ** i, dilation=2 ** i), nn.GELU(), nn.Conv1d(width, width, 1))
                                      for i in range(layers)])
            self.out = nn.Conv1d(width, SIGNALS, 1)
            self.mask_ratio = float(mask_ratio)

        def encode(self, x: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
            s, mf, hv = _inputs(x, m)
            _, pooled = self.enc(torch.cat([s, hv.unsqueeze(-1)], -1), m)
            return pooled

        def forward(self, x: "torch.Tensor", m: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            m_in = m
            if self.training and self.mask_ratio > 0:
                drop = torch.rand(m.shape, device=m.device) < self.mask_ratio
                m_in = m & ~drop
            z = self.encode(x, m_in)
            L = x.shape[1]
            h = self.dec_in(torch.cat([z.unsqueeze(1).expand(-1, L, -1), self.pos[:, :L].expand(x.shape[0], -1, -1)], -1).transpose(1, 2))
            for b in self.dec:
                h = h + b(h)
            return self.out(h).transpose(1, 2), z

        def loss(self, x: "torch.Tensor", m: "torch.Tensor", recon: "torch.Tensor") -> "torch.Tensor":
            return time_loss(x, m, recon)

    class STFTCNNAE(nn.Module):
        """2-D CNN autoencoder on per-channel log|STFT| of speed and a_tan (zero-filled after removing the mean of the valid
        frames), with a frame-coverage channel; optional sin/cos phase channels. Loss on windows with coverage ≥ 0.5."""
        domain = "stft"

        def __init__(self, d: int, width: int, L: int, nperseg: int, hop: int, phase: bool = False):
            super().__init__()
            self.nperseg, self.hop, self.phase = int(nperseg), int(hop), bool(phase)
            self.register_buffer("win", torch.hann_window(self.nperseg, periodic=False))
            self.F = self.nperseg // 2 + 1
            self.T = 1 + (L - self.nperseg) // self.hop
            self.c_sig = 2 * (3 if phase else 1)
            self.enc = nn.Sequential(nn.Conv2d(self.c_sig + 1, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 3, padding=1, stride=2), nn.GELU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((2, 2)))
            self.to_z = nn.Linear(4 * width, d)
            self.dec = nn.Sequential(nn.Linear(d, 4 * width), nn.GELU(), nn.Linear(4 * width, self.c_sig * self.F * self.T))

        def spectro(self, x: "torch.Tensor", m: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            """(features (B, c_sig, F, T), coverage (B, T))."""
            mf = m.float()
            chans = []
            for c in (0, 1):
                s = x[..., c]
                mu = (s * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp_min(1.0)
                z = (s - mu) * mf
                S = torch.stft(z, self.nperseg, self.hop, window=self.win, center=False, return_complex=True)        # (B, F, T)
                chans.append(torch.log1p(S.abs()))
                if self.phase:
                    ang = torch.angle(S)
                    chans += [torch.sin(ang), torch.cos(ang)]
            cov = mf.unfold(1, self.nperseg, self.hop).mean(-1)                                                       # (B, T)
            return torch.stack(chans, 1), cov

        def encode(self, x: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
            S, cov = self.spectro(x, m)
            inp = torch.cat([S, cov[:, None, None, :].expand(-1, 1, self.F, -1)], 1)
            return self.to_z(self.enc(inp).flatten(1))

        def forward(self, x: "torch.Tensor", m: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            z = self.encode(x, m)
            return self.dec(z).view(-1, self.c_sig, self.F, self.T), z

        def loss(self, x: "torch.Tensor", m: "torch.Tensor", recon: "torch.Tensor") -> "torch.Tensor":
            S, cov = self.spectro(x, m)
            w = (cov >= 0.5).float()[:, None, None, :].expand_as(S)
            return (((recon - S) ** 2) * w).sum() / w.sum().clamp_min(1.0)


def _make(arch: str, d: int, width: int, cfg: Any) -> Any:
    L = int(cfg.max_frames)
    if arch == "lstm_ae":
        return RNNAE("lstm", d, width, L)
    if arch == "gru_ae":
        return RNNAE("gru", d, width, L)
    if arch == "tcn_ae":
        return TCNAE(d, width, L, layers=4)
    if arch == "dtcn_ae":
        return TCNAE(d, width, L, layers=6)
    if arch == "masked_tae":
        return TCNAE(d, width, L, layers=4, mask_ratio=float(cfg.extra.get("mask_ratio", 0.3)))
    if arch == "stft_ae":
        return STFTCNNAE(d, width, L, cfg.stft_nperseg, cfg.stft_hop, cfg.stft_phase)
    raise ValueError(f"architecture {arch!r} is not built in this release (wavelet_cnn is optional P2)")


def n_params(model: Any) -> int:
    if isinstance(model, PCAEncoder):
        return model.n_params()
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def choose_width(arch: str, d: int, cfg: Any) -> dict[str, Any]:
    """Width whose parameter count is closest to the reference budget (``tcn_ae`` at width 32 and the same d)."""
    _require_torch()
    ref = n_params(_make("tcn_ae", d, int(cfg.extra.get("reference_width", 32)), cfg))
    if arch in ("tcn_ae",):
        w = int(cfg.extra.get("reference_width", 32))
        return {"arch": arch, "d": d, "width": w, "params": ref, "reference_params": ref, "ratio": 1.0, "fits_band": True}
    best = min(((abs(n_params(_make(arch, d, w, cfg)) / ref - 1), w) for w in WIDTH_GRID))
    w = best[1]
    p = n_params(_make(arch, d, w, cfg))
    return {"arch": arch, "d": d, "width": w, "params": p, "reference_params": ref, "ratio": p / ref, "fits_band": abs(p / ref - 1) <= cfg.param_band}


def build(arch: str, d: int, cfg: Any, width: int | None = None) -> Any:
    """A fresh model (PCA → ``PCAEncoder``)."""
    if arch == "pca":
        return PCAEncoder(d)
    _require_torch()
    return _make(arch, d, int(width if width is not None else choose_width(arch, d, cfg)["width"]), cfg)


__all__ = ["PCAEncoder", "SIGNALS", "TORCH", "WIDTH_GRID", "build", "choose_width", "n_params"]
