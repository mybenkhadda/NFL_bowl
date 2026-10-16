"""Self-supervised convolutional encoder of aligned sprint curves (notebook 03). Needs PyTorch.

Input      one curve of a notebook-02 representation (``fpca.build_curves``): a value on a fixed grid, NaN where
           the recording holds nothing. The encoder never sees a NaN and never a filled value. Per grid point it
           receives three channels:
               value      (curve − training mean function) / training scale, set to 0 wherever it is not VISIBLE
               visible    1 where the point was recorded AND not hidden on purpose
               position   the grid coordinate scaled to [−1, 1] (a constant of the representation)
           A hidden point and a never-recorded point look the same to the encoder, so the input carries no trace of
           where the reconstruction targets are.
Masks      support   grid points recorded for ≥ ``min_coverage`` of the training weight — the rule and the value
                     of the FPCA fit of the same fold, so both models live on the same axis range
           observed  recorded AND on the support
           hidden    observed points hidden on purpose (``repr_eval.span_mask``): the ONLY reconstruction targets
           visible   observed AND not hidden: what the encoder may use
           An attempt covering less than ``min_observed_fraction`` of the support is neither trained on nor
           embedded (the FPCA scoring rule), and the same rule applies to a truncated view.
Model      dilated residual 1-D convolutions over the grid (non-causal: the task is a representation of a whole
           run, not a forecast) → mean and max over VISIBLE points → linear → embedding z (8 numbers) → small
           decoder → the whole curve. The decoder sees z only, so every reconstruction passes the bottleneck and
           is comparable with an FPCA reconstruction from K scores.
TCN-A      loss = mean squared error on hidden points.
TCN-B      TCN-A + λ · consistency between the embeddings of two attempts of the same TRAINING athlete, both
           encoded on the grid points the two recordings share (no comparison of regions only one of them holds).
           ``vicreg``: squared distance + a hinge keeping every dimension's spread + a decorrelation term — no
           negative pair, so two different athletes are never assumed dissimilar. ``mse``: squared distance divided
           by the batch variance (scale-free, no explicit protection against collapse) — the simple comparator.
Leakage    normaliser, support, noise level, weights and early stopping use the training athletes of the fold;
           the early-stopping athletes are an inner development fold of those. ``assert_unseen`` guards it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl
import torch
import torch.nn.functional as Fn
from torch import nn

from ...data.utils import atomic_path, atomic_write_json, get_logger, read_json, utc_now
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from . import repr_eval as rv
from .config import NotebookControls
from .data import assert_disjoint, fold_athletes, inner_athletes
from .fpca import SPEC, CurveSet, FpcaModel, athlete_weights, longest_run, quadrature
from .fpca_eval import Meta, first_two, fit_key, fold_tag
from .repr_eval import EvalConfig, MethodSpec

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
TCN_VERSION = "1"
RECON, CONSISTENCY = "reconstruction", "reconstruction+consistency"
N_INPUT = 3
AUGMENTATIONS = (
    ("span masking of recorded points", "used (the task)",
     "Hides about 30 % of an attempt's recorded points in two contiguous runs; the values stay those measured. "
     "The athlete's curve is unchanged, only what the encoder may look at."),
    ("edge masking (a shorter recording)", "used (35 % of the runs touch the first or last recorded point)",
     "Recordings start and stop at different moments of the run; hiding an end reproduces that difference without "
     "inventing a start. The hidden end remains a target, so the embedding must still describe the whole curve."),
    ("additive noise at the measured noise level", "used (input only, never the target)",
     "SD = robust SD of the second differences of the training curves / √6, i.e. the point-to-point irregularity "
     "actually present in the training fold. Smaller than any difference between athletes by construction."),
    ("pair views on the shared recorded range", "used (TCN-B only)",
     "Two attempts of one athlete are compared on the grid points both recordings hold. No region present in only "
     "one recording is treated as the same measurement."),
    ("time warping", "excluded", "Changes how fast speed is gained — the acceleration profile is the trait under study."),
    ("speed or amplitude scaling", "excluded", "Speed level is the main difference between athletes (and the stopwatch)."),
    ("shifts of acceleration magnitude", "excluded", "Same reason: it is the signal, not a nuisance."),
    ("mixing curves of different athletes or positions", "excluded", "Creates runs nobody produced and blurs athlete identity."),
    ("simulated starts or extrapolated ends", "excluded", "Would be treated as measurements; nothing unrecorded is ever a target."),
)


class TcnError(ValueError):
    """The encoder cannot be configured, trained or applied (too few curves, empty support, bad checkpoint)."""


@dataclass(frozen=True)
class TcnConfig:
    version: str = TCN_VERSION
    objective: str = RECON                        # RECON = TCN-A · CONSISTENCY = TCN-B
    # architecture (proposed, fixed before any fit; only embedding_dim has a declared sensitivity: 16) ----------
    embedding_dim: int = 8
    hidden: int = 32
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4)        # one residual block per dilation (receptive field 57 grid points)
    dropout: float = 0.10
    decoder_hidden: int = 64
    # optimisation (proposed, fixed) -----------------------------------------------------------------------------
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_athletes: int = 32                      # athletes per batch (≤ 2 attempts each) — notebook BATCH_SIZE
    max_epochs: int = 200                         # notebook MAX_EPOCHS
    min_epochs: int = 15
    patience: int = 25                            # epochs without improvement of the development loss
    min_delta: float = 1e-4
    inner_dev_fold: int = 0                       # inner fold of the split registry used for early stopping
    # masking and augmentation (proposed, fixed) ---------------------------------------------------------------
    mask_ratio: float = 0.30
    mask_spans: int = 2
    p_edge: float = 0.35
    min_visible: int = 4
    noise_scale: float = 1.0                      # × the noise SD estimated on the training fold
    # consistency (TCN-B): kind and weight are the ONLY tuned quantities, on design athletes ----------------------
    consistency: str = "vicreg"                   # "vicreg" | "mse"
    consistency_weight: float = 0.3
    variance_weight: float = 1.0
    covariance_weight: float = 0.04
    min_pair_overlap: float = 0.50                # share of the support two attempts must both hold to be a pair
    # eligibility: the FPCA rules of notebook 02 (FpcaConfig) ---------------------------------------------------
    min_coverage: float = 0.80
    min_observed_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.objective not in (RECON, CONSISTENCY):
            raise TcnError(f"objective must be {RECON!r} or {CONSISTENCY!r}, got {self.objective!r}")
        if self.consistency not in ("vicreg", "mse"):
            raise TcnError(f"consistency must be 'vicreg' or 'mse', got {self.consistency!r}")
        if self.kernel_size % 2 != 1 or not self.dilations or min(self.dilations) < 1:
            raise TcnError("kernel_size must be odd and dilations positive (the convolutions keep the grid length)")
        if self.embedding_dim < 1 or self.hidden < 1 or not 0 < self.mask_ratio < 1:
            raise TcnError("embedding_dim and hidden must be ≥ 1 and mask_ratio inside (0, 1)")

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * int(sum(self.dilations))

    def params(self) -> dict[str, Any]:
        return {**asdict(self), "dilations": list(self.dilations)}

    @classmethod
    def from_params(cls, d: dict[str, Any]) -> "TcnConfig":
        return cls(**{**d, "dilations": tuple(d["dilations"])})

    def for_mode(self, controls: NotebookControls) -> "TcnConfig":
        """Notebook budget controls. Smoke mode keeps the model and the losses; it caps the epochs."""
        if controls.smoke:
            return replace(self, batch_athletes=controls.batch_size, max_epochs=min(controls.max_epochs, 12), min_epochs=3, patience=4)
        return replace(self, batch_athletes=controls.batch_size, max_epochs=controls.max_epochs)


HISTORY = TableContract(
    "tcn_training_history", "Loss per epoch of every fold model.",
    {"method": S, "repeat": I, "outer_fold": I, "epoch": I, "train_reconstruction": F, "train_consistency": F,
     "dev_reconstruction": F, "dev_consistency": F, "dev_total": F, "dev_rank_share": F, "is_best": B},
    key=("method", "repeat", "outer_fold", "epoch"), area="method × fold × epoch",
    docs={"train_reconstruction": "mean squared error on hidden points, normalised units, random masks",
          "dev_reconstruction": "same on the early-stopping athletes with a fixed mask",
          "dev_total": "development reconstruction + weight × consistency: the early-stopping criterion",
          "dev_rank_share": "effective rank / dimension of the development embeddings (collapse monitor)"})
FOLDS = TableContract(
    "tcn_fold_fits", "Fold-specific normalisation, sample sizes, stopping epoch and cost of every fold model.",
    {"method": S, "representation": S, "repeat": I, "outer_fold": I, "objective": S, "embedding_dim": I, "n_parameters": I,
     "n_fit_athletes": I, "n_dev_athletes": I, "n_fit_sequences": I, "n_dev_sequences": I, "n_heldout_athletes": I,
     "n_heldout_sequences": I, "n_heldout_embedded": I, "support_lo": F, "support_hi": F, "n_support": I, "n_eval_support": I,
     "scale": F, "noise_sd": F, "epochs_run": I, "best_epoch": I, "best_dev_loss": F, "stopped_early": B, "train_seconds": F,
     "device": S}, key=("method", "repeat", "outer_fold"), area="method × fold",
    docs={"scale": "training-fold RMS deviation from the mean function, in the unit of the curve",
          "noise_sd": "training-fold noise SD used for the input noise, same unit",
          "n_eval_support": "grid points shared with the FPCA fit of the fold, on which errors are compared",
          "n_fit_athletes": "athletes whose curves update the weights (training athletes minus the development fold)"})
DESIGN = TableContract(
    "tcn_design_sweep", "Consistency kind and weight compared on DESIGN athletes only (cross-fitted inside them).",
    {"consistency": S, "consistency_weight": F, "n_folds": I, "n_athletes_repeat": I, "auc": F, "auc_fold_min": F,
     "auc_fold_max": F, "top1": F, "rmse_masked": F, "rank_share_min": F, "near_identical_max": F, "collapsed": B,
     "selected": B}, key=("consistency", "consistency_weight"), area="consistency kind × weight",
    docs={"consistency": "none = TCN-A (reference row, never selectable)",
          "auc": "within-position repeat-attempt discrimination of design athletes the encoder did not see",
          "rmse_masked": "hidden-point error on those athletes, in the unit of the curve"})


# ============================================================================== fold-safe normalisation
def _ids(athletes: Sequence[int]) -> np.ndarray:
    return np.asarray(sorted({int(i) for i in athletes}), dtype=np.int64)


@dataclass
class CurveNormalizer:
    """Support, mean function, scale and noise level of one representation, learned on the training athletes."""
    name: str
    grid: np.ndarray                 # (G,) the representation's whole grid
    support: np.ndarray              # (G,) bool
    coverage: np.ndarray             # (G,) training weight recorded at each grid point
    mean: np.ndarray                 # (G,) training mean function (0 outside the support)
    scale: float                     # RMS deviation from the mean, unit of the curve
    noise_sd: float                  # unit of the curve
    min_observed_fraction: float
    fit_athletes: tuple[int, ...]

    @classmethod
    def fit(cls, curves: CurveSet, train_athletes: Sequence[int], min_coverage: float, min_observed_fraction: float
            ) -> "CurveNormalizer":
        ids = _ids(train_athletes)
        rows = np.isin(curves.nfl_ids, ids) & np.isfinite(curves.Y).any(1)
        if rows.sum() < 5:
            raise TcnError(f"{curves.spec.name}: only {int(rows.sum())} training curve(s) with data")
        Y, w = curves.Y[rows], athlete_weights(curves.nfl_ids[rows])
        O = np.isfinite(Y)
        coverage = (w[:, None] * O).sum(0)
        lo, hi = longest_run(coverage >= min_coverage)
        if hi - lo < 8:
            raise TcnError(f"{curves.spec.name}: only {hi - lo} contiguous grid point(s) are recorded for ≥ {min_coverage:.0%} "
                           "of the training athletes")
        support = np.zeros(curves.grid.size, dtype=bool)
        support[lo:hi] = True
        Os = O[:, support]
        wO = w[:, None] * Os
        mean_s = (wO * np.where(Os, Y[:, support], 0.0)).sum(0) / wO.sum(0)
        resid = np.where(Os, Y[:, support] - mean_s, 0.0)
        scale = float(np.sqrt((wO * resid ** 2).sum() / wO.sum()))
        if not scale > 0:
            raise TcnError(f"{curves.spec.name}: the training curves have no variance on the support")
        Ys = Y[:, support]
        d2 = Ys[:, 2:] - 2.0 * Ys[:, 1:-1] + Ys[:, :-2]
        d2 = d2[np.isfinite(d2)]
        noise = float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0)) if d2.size >= 10 else 0.0
        mean = np.zeros(curves.grid.size)
        mean[support] = mean_s
        return cls(curves.spec.name, curves.grid.astype(float), support, coverage, mean, scale, noise,
                   float(min_observed_fraction), tuple(ids.tolist()))

    @property
    def q(self) -> np.ndarray:
        """(G,) quadrature weights of the support (0 outside), as used by the FPCA coverage rule."""
        q = np.zeros(self.grid.size)
        q[self.support] = quadrature(self.grid[self.support])
        return q

    def observed(self, Y: np.ndarray) -> np.ndarray:
        return np.isfinite(np.atleast_2d(Y)) & self.support[None, :]

    def fraction(self, visible: np.ndarray) -> np.ndarray:
        """Share of the support the visible points cover."""
        q = self.q
        return (visible & self.support[None, :]).astype(float) @ q / q.sum()

    def transform(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(normalised values (n, G) float32 with 0 where nothing is observed, observed (n, G) bool)."""
        Y = np.atleast_2d(np.asarray(Y, float))
        if Y.shape[1] != self.grid.size:
            raise TcnError(f"{self.name}: curves have {Y.shape[1]} grid points, the normaliser expects {self.grid.size}")
        obs = self.observed(Y)
        return np.where(obs, (np.nan_to_num(Y) - self.mean) / self.scale, 0.0).astype(np.float32), obs

    def inverse(self, X: np.ndarray) -> np.ndarray:
        """Normalised values → the unit of the curve; NaN outside the support."""
        out = np.asarray(X, float) * self.scale + self.mean
        out[:, ~self.support] = np.nan
        return out

    def assert_unseen(self, athletes: Sequence[int], where: str = "") -> None:
        assert_disjoint(self.fit_athletes, list(athletes), where or f"normaliser {self.name} vs evaluation athletes")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "grid": self.grid.tolist(), "support": self.support.astype(int).tolist(),
                "coverage": self.coverage.tolist(), "mean": self.mean.tolist(), "scale": self.scale, "noise_sd": self.noise_sd,
                "min_observed_fraction": self.min_observed_fraction, "fit_athletes": list(self.fit_athletes)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CurveNormalizer":
        return cls(str(d["name"]), np.asarray(d["grid"], float), np.asarray(d["support"]).astype(bool),
                   np.asarray(d["coverage"], float), np.asarray(d["mean"], float), float(d["scale"]), float(d["noise_sd"]),
                   float(d["min_observed_fraction"]), tuple(int(i) for i in d["fit_athletes"]))


# ============================================================================== model
class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        y = self.drop(Fn.gelu(self.conv1(h))) * domain
        y = self.drop(Fn.gelu(self.conv2(y))) * domain
        return (y + self.skip(h)) * domain


class CurveTCN(nn.Module):
    """``encode``: (values, visible mask, support) → embedding. ``decode``: embedding → the curve on the whole grid."""

    def __init__(self, tcfg: TcnConfig, n_grid: int):
        super().__init__()
        self.n_grid, self.embedding_dim = int(n_grid), int(tcfg.embedding_dim)
        blocks, c = [], N_INPUT
        for d in tcfg.dilations:
            blocks.append(ResidualBlock(c, tcfg.hidden, tcfg.kernel_size, d, tcfg.dropout))
            c = tcfg.hidden
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(2 * tcfg.hidden, tcfg.embedding_dim)
        self.decoder = nn.Sequential(nn.Linear(tcfg.embedding_dim, tcfg.decoder_hidden), nn.GELU(),
                                     nn.Linear(tcfg.decoder_hidden, n_grid))
        self.register_buffer("position", torch.linspace(-1.0, 1.0, n_grid).view(1, 1, n_grid))

    def encode(self, x: torch.Tensor, visible: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """``x`` (B, G) normalised values · ``visible`` (B, G) bool · ``domain`` (G,) bool support. A value that is
        not visible is multiplied by zero here, whatever ``x`` holds."""
        dom = domain.to(x.dtype).view(1, 1, -1)
        v = (visible & domain.view(1, -1)).to(x.dtype).unsqueeze(1)
        h = torch.cat([x.unsqueeze(1) * v, v, (self.position * dom).expand(x.shape[0], -1, -1)], dim=1)
        for block in self.blocks:
            h = block(h, dom)
        mean = (h * v).sum(-1) / v.sum(-1).clamp_min(1.0)
        top = h.masked_fill(v == 0, float("-inf")).max(-1).values
        top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
        return self.head(torch.cat([mean, top], dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, visible: torch.Tensor, domain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, visible, domain)
        return z, self.decode(z)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ============================================================================== losses
def masked_mse(pred: torch.Tensor, target: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the points of ``hidden`` only (recorded points hidden on purpose). Zero, with a
    graph, when nothing is hidden."""
    m = hidden.to(pred.dtype)
    n = m.sum()
    if n <= 0:
        return (pred * 0.0).sum()
    return (((pred - target) ** 2) * m).sum() / n


def consistency_loss(za: torch.Tensor, zb: torch.Tensor, tcfg: TcnConfig) -> torch.Tensor:
    """``za[i]`` and ``zb[i]`` embed two attempts of the same athlete (≥ 2 pairs). See the module docstring."""
    z = torch.cat([za, zb], dim=0)
    var = z.var(dim=0)
    gap = ((za - zb) ** 2).mean()
    if tcfg.consistency == "mse":
        return gap / var.mean().detach().clamp_min(1e-8)
    hinge = Fn.relu(1.0 - torch.sqrt(var + 1e-4)).mean()
    zc = z - z.mean(dim=0)
    cov = zc.T @ zc / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return gap + tcfg.variance_weight * hinge + tcfg.covariance_weight * (off ** 2).sum() / z.shape[1]


# ============================================================================== batches of athletes
class AttemptPairSampler:
    """Batches of ATHLETES among the given rows. An athlete brings at most two attempts; two attempts of one athlete
    form a positive pair. Pairs are built inside the rows passed in, so a pair can never join two sets of athletes."""

    def __init__(self, nfl_ids: np.ndarray, rows: np.ndarray, batch_athletes: int):
        self.rows = np.asarray(rows, dtype=int)
        ids = np.asarray(nfl_ids)[self.rows]
        self.athletes = np.unique(ids)
        self.members = {int(a): self.rows[ids == a] for a in self.athletes}
        self.batch_athletes = int(batch_athletes)

    def _collect(self, athletes: np.ndarray, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray]:
        rows: list[int] = []
        pairs: list[tuple[int, int]] = []
        for a in athletes:
            m = self.members[int(a)]
            pick = m[:2] if rng is None or m.size <= 2 else rng.choice(m, 2, replace=False)
            if pick.size == 2:
                pairs.append((len(rows), len(rows) + 1))
            rows.extend(int(i) for i in pick)
        return np.asarray(rows, dtype=int), np.asarray(pairs, dtype=int).reshape(-1, 2)

    def epoch(self, rng: np.random.Generator) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """(rows of the batch, (P, 2) positions of the positive pairs inside the batch), over a shuffle of athletes."""
        order = rng.permutation(self.athletes)
        for s in range(0, order.size, self.batch_athletes):
            rows, pairs = self._collect(order[s:s + self.batch_athletes], rng)
            if rows.size >= 2:
                yield rows, pairs

    def all_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic single batch: the first two rows of every athlete (fixed development batch)."""
        return self._collect(self.athletes, None)


def assert_pairs_same_athlete(nfl_ids: np.ndarray, rows: np.ndarray, pairs: np.ndarray) -> None:
    ids = np.asarray(nfl_ids)[np.asarray(rows, dtype=int)]
    if len(pairs) and (ids[pairs[:, 0]] != ids[pairs[:, 1]]).any():
        raise TcnError("a positive pair joins attempts of two different athletes")


# ============================================================================== training
@dataclass
class Trained:
    """One fitted encoder with everything needed to apply it again."""
    model: CurveTCN
    normalizer: CurveNormalizer
    tcfg: TcnConfig
    fit_athletes: tuple[int, ...]
    dev_athletes: tuple[int, ...]
    history: list[dict[str, Any]]
    info: dict[str, Any]
    method: str = ""
    representation: str = ""
    repeat: int = -1
    outer_fold: int = -1


@dataclass
class _Tensors:
    X: torch.Tensor                  # (n, G) normalised values
    obs: torch.Tensor                # (n, G) bool
    domain: torch.Tensor             # (G,) bool
    q: torch.Tensor                  # (G,) quadrature weights of the support


def _losses(model: CurveTCN, data: _Tensors, rows: np.ndarray, pairs: np.ndarray, hidden: np.ndarray, noise: float,
            tcfg: TcnConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """(total, reconstruction, consistency, pairs used) of one batch. Targets are the clean recorded values."""
    idx = torch.as_tensor(rows, dtype=torch.long, device=data.X.device)
    x, o = data.X[idx], data.obs[idx]
    m = torch.as_tensor(hidden, dtype=torch.bool, device=x.device) & o
    xin = x + noise * torch.randn_like(x) if noise > 0 else x
    _, rec = model(xin, o & ~m, data.domain)
    recon = masked_mse(rec, x, m)
    cons, used = x.new_zeros(()), 0
    if tcfg.objective == CONSISTENCY and len(pairs) >= 2:
        pr = torch.as_tensor(pairs, dtype=torch.long, device=x.device)
        a, b = idx[pr[:, 0]], idx[pr[:, 1]]
        common = data.obs[a] & data.obs[b]
        keep = (common.to(x.dtype) * data.q).sum(1) / data.q.sum() >= tcfg.min_pair_overlap
        used = int(keep.sum())
        if used >= 2:
            xa, xb, v = data.X[a][keep], data.X[b][keep], common[keep]
            if noise > 0:
                xa, xb = xa + noise * torch.randn_like(xa), xb + noise * torch.randn_like(xb)
            cons = consistency_loss(model.encode(xa, v, data.domain), model.encode(xb, v, data.domain), tcfg)
    return recon + tcfg.consistency_weight * cons, recon, cons, used


def train_encoder(curves: CurveSet, norm_athletes: Sequence[int], fit_athletes: Sequence[int], dev_athletes: Sequence[int],
                  tcfg: TcnConfig, seed: int, device: str = "cpu", tag: str = "") -> Trained:
    """Normaliser on ``norm_athletes``; weights on ``fit_athletes``; early stopping on ``dev_athletes``. The three
    sets are training-side athletes; the caller keeps the evaluated athletes out of all of them."""
    fit_ids, dev_ids = _ids(fit_athletes), _ids(dev_athletes)
    assert_disjoint(fit_ids.tolist(), dev_ids.tolist(), f"{tag}: encoder fit vs early-stopping athletes")
    norm = CurveNormalizer.fit(curves, norm_athletes, tcfg.min_coverage, tcfg.min_observed_fraction)
    X, obs = norm.transform(curves.Y)
    eligible = norm.fraction(obs) >= tcfg.min_observed_fraction
    fit_rows = np.flatnonzero(np.isin(curves.nfl_ids, fit_ids) & eligible)
    dev_rows = np.flatnonzero(np.isin(curves.nfl_ids, dev_ids) & eligible)
    if fit_rows.size < 8 or dev_rows.size < 4:
        raise TcnError(f"{tag}: {fit_rows.size} fit and {dev_rows.size} development curve(s) are too few to train an encoder")
    where = torch.device(device)
    torch.manual_seed(seed_for(seed, "torch", tag))
    rng = np.random.default_rng(seed_for(seed, "numpy", tag))
    model = CurveTCN(tcfg, curves.grid.size).to(where)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    data = _Tensors(torch.from_numpy(X).to(where), torch.from_numpy(obs).to(where), torch.from_numpy(norm.support).to(where),
                    torch.from_numpy(norm.q.astype(np.float32)).to(where))
    noise = float(tcfg.noise_scale * norm.noise_sd / norm.scale)
    sampler = AttemptPairSampler(curves.nfl_ids, fit_rows, tcfg.batch_athletes)
    d_rows, d_pairs = AttemptPairSampler(curves.nfl_ids, dev_rows, tcfg.batch_athletes).all_pairs()
    d_hidden = rv.evaluation_mask(curves.seq_ids[d_rows], obs[d_rows], seed_for(seed, "dev_mask", tag))
    d_idx = torch.as_tensor(d_rows, dtype=torch.long, device=where)
    best_loss, best_epoch, best_state, history = float("inf"), 0, None, []
    t0 = time.perf_counter()
    for epoch in range(1, tcfg.max_epochs + 1):
        model.train()
        tot = np.zeros(3)
        for rows, pairs in sampler.epoch(rng):
            hidden = rv.span_mask(obs[rows], rng, tcfg.mask_ratio, tcfg.mask_spans, tcfg.p_edge, tcfg.min_visible)
            loss, rec, cons, _ = _losses(model, data, rows, pairs, hidden, noise, tcfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            tot += (float(rec), float(cons), 1.0)
        model.eval()
        with torch.no_grad():
            d_total, d_rec, d_cons, _ = _losses(model, data, d_rows, d_pairs, d_hidden, 0.0, tcfg)
            zd = model.encode(data.X[d_idx], data.obs[d_idx], data.domain).double().cpu().numpy()
        d_total = float(d_total)
        if not np.isfinite(d_total):
            raise TcnError(f"{tag}: the development loss is not finite at epoch {epoch}; lower the learning rate")
        improved = d_total < best_loss - tcfg.min_delta
        if improved:
            best_loss, best_epoch = d_total, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_reconstruction": tot[0] / max(tot[2], 1.0), "train_consistency": tot[1] / max(tot[2], 1.0),
                        "dev_reconstruction": float(d_rec), "dev_consistency": float(d_cons), "dev_total": d_total,
                        "dev_rank_share": rv.collapse_metrics(zd)["rank_share"], "is_best": False})
        if not improved and epoch >= tcfg.min_epochs and epoch - best_epoch >= tcfg.patience:
            break
    seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.cpu().eval()
    history[best_epoch - 1]["is_best"] = True
    g = norm.grid[norm.support]
    info = {"n_parameters": count_parameters(model), "n_fit_athletes": int(np.unique(curves.nfl_ids[fit_rows]).size),
            "n_dev_athletes": int(np.unique(curves.nfl_ids[dev_rows]).size), "n_fit_sequences": int(fit_rows.size),
            "n_dev_sequences": int(dev_rows.size), "support_lo": float(g[0]), "support_hi": float(g[-1]),
            "n_support": int(norm.support.sum()), "scale": norm.scale, "noise_sd": norm.noise_sd, "epochs_run": len(history),
            "best_epoch": int(best_epoch), "best_dev_loss": float(best_loss), "stopped_early": len(history) < tcfg.max_epochs,
            "train_seconds": float(seconds), "device": str(where)}
    return Trained(model, norm, tcfg, tuple(fit_ids.tolist()), tuple(dev_ids.tolist()), history, info)


@torch.no_grad()
def embed(model: CurveTCN, norm: CurveNormalizer, Y: np.ndarray, device: str = "cpu", visible: np.ndarray | None = None,
          batch: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(embeddings (n, E), reconstruction (n, G) in the unit of the curve, visible share of the support (n,)).
    ``visible`` restricts what the encoder may use (evaluation mask, truncation). A row whose visible points cover
    less than ``min_observed_fraction`` of the support gets NaN: it is not embedded."""
    model = model.to(device).eval()
    X, obs = norm.transform(Y)
    vis = obs if visible is None else obs & np.asarray(visible, dtype=bool)
    frac = norm.fraction(vis)
    ok = np.flatnonzero(frac >= norm.min_observed_fraction)
    Z, R = np.full((X.shape[0], model.embedding_dim), np.nan), np.full(X.shape, np.nan)
    domain = torch.from_numpy(norm.support).to(device)
    for s in range(0, ok.size, batch):
        rows = ok[s:s + batch]
        z, rec = model(torch.from_numpy(X[rows]).to(device), torch.from_numpy(vis[rows]).to(device), domain)
        Z[rows], R[rows] = z.double().cpu().numpy(), norm.inverse(rec.double().cpu().numpy())
    return Z, R, frac


# ============================================================================== checkpoints
def checkpoint_path(models_dir: Path, method: str, repeat: int, fold: int) -> Path:
    return Path(models_dir) / "tcn" / f"{method}__{fold_tag(repeat, fold)}.pt"


def save_checkpoint(path: Path, t: Trained) -> Path:
    """Weights, configuration, fold normalisation and provenance in one file (plain containers and tensors only,
    so it reloads with ``weights_only=True``)."""
    payload = {"format": 1, "method": t.method, "representation": t.representation, "repeat": int(t.repeat),
               "outer_fold": int(t.outer_fold), "n_grid": int(t.model.n_grid), "tcfg": t.tcfg.params(),
               "normalizer": t.normalizer.to_dict(), "fit_athletes": list(t.fit_athletes), "dev_athletes": list(t.dev_athletes),
               "info": dict(t.info), "state_dict": {k: v.detach().cpu() for k, v in t.model.state_dict().items()}}
    with atomic_path(Path(path)) as tmp:
        torch.save(payload, tmp)
    return Path(path)


def load_checkpoint(path: Path) -> Trained:
    p = torch.load(Path(path), map_location="cpu", weights_only=True)
    if p.get("format") != 1:
        raise TcnError(f"{path}: unknown checkpoint format {p.get('format')!r}")
    tcfg = TcnConfig.from_params(p["tcfg"])
    model = CurveTCN(tcfg, int(p["n_grid"]))
    model.load_state_dict(p["state_dict"])
    return Trained(model.eval(), CurveNormalizer.from_dict(p["normalizer"]), tcfg, tuple(p["fit_athletes"]),
                   tuple(p["dev_athletes"]), [], dict(p["info"]), p["method"], p["representation"], int(p["repeat"]),
                   int(p["outer_fold"]))


def record_models(models_dir: Path, stage: str, key: str, methods: Sequence[str]) -> None:
    path = Path(models_dir) / "tcn" / "_models.json"
    state = read_json(path, {"stages": {}})
    state["stages"][stage] = {"key": key, "methods": sorted(methods), "saved_at": utc_now()}
    atomic_write_json(state, path)


def load_models(models_dir: Path, stage: str, key: str, plan: Sequence[tuple[MethodSpec, TcnConfig]], splits: pl.DataFrame,
                repeats: Sequence[int]) -> dict[tuple[str, int, int], Trained]:
    """Fold models of a stage from disk — only when they were saved under the same ``key`` (parameters and input
    files). Returns {} otherwise: a stale checkpoint is never used."""
    saved = read_json(Path(models_dir) / "tcn" / "_models.json", {"stages": {}})["stages"].get(stage, {})
    if saved.get("key") != key:
        return {}
    out = {}
    for spec, _ in plan:
        for r in repeats:
            for f in sorted(splits.filter(pl.col("repeat") == r)["outer_fold"].unique().to_list()):
                path = checkpoint_path(models_dir, spec.method, r, f)
                if not path.is_file():
                    return {}
                out[(spec.method, int(r), int(f))] = load_checkpoint(path)
    return out


# ============================================================================== fold loop
def train_fold(curves: CurveSet, splits: pl.DataFrame, repeat: int, fold: int, spec: MethodSpec, tcfg: TcnConfig, seed: int,
               device: str = "cpu") -> Trained:
    """Encoder of one outer fold: normaliser on its training athletes, weights on those outside the inner
    development fold, early stopping on that development fold. The held-out athletes take no part."""
    train, test = fold_athletes(splits, repeat, fold)
    fit, dev = inner_athletes(splits, repeat, fold, tcfg.inner_dev_fold)
    t = train_encoder(curves, train, fit, dev, tcfg, seed, device, tag=f"{spec.method}:{fold_tag(repeat, fold)}")
    t.normalizer.assert_unseen(test, f"{spec.method}, repeat {repeat}, fold {fold}: normaliser vs held-out athletes")
    assert_disjoint([*t.fit_athletes, *t.dev_athletes], test, f"{spec.method}, repeat {repeat}, fold {fold}: encoder vs held-out athletes")
    t.method, t.representation, t.repeat, t.outer_fold = spec.method, spec.representation, int(repeat), int(fold)
    return t


def tcn_fold_outputs(t: Trained, curves: CurveSet, splits: pl.DataFrame, support_eval: np.ndarray | None, ecfg: EvalConfig,
                     seed: int, device: str = "cpu") -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Embeddings (every athlete of the fold in the full view; held-out athletes also truncated) and held-out
    reconstruction sums of one fold model, in the format of ``repr_eval``. The encoder is frozen here."""
    _, test = fold_athletes(splits, t.repeat, t.outer_fold)
    assert_disjoint([*t.fit_athletes, *t.dev_athletes], test, f"{t.method}: frozen encoder vs held-out athletes")
    nfl, seq, norm = curves.nfl_ids, curves.seq_ids.astype(str), t.normalizer
    te = np.isin(nfl, np.asarray(test))
    role = np.where(te, "heldout", np.where(np.isin(nfl, np.asarray(t.dev_athletes)), "dev",
                                            np.where(np.isin(nfl, np.asarray(t.fit_athletes)), "fit", "other")))
    Z, R, frac = embed(t.model, norm, curves.Y, device)
    train_rows = np.isin(role, ("fit", "dev")) & np.isfinite(Z).all(1)
    mu, sd = Z[train_rows].mean(0), Z[train_rows].std(0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    keep = role != "other"
    key = (t.method, t.repeat, t.outer_fold)
    parts = [rv.embedding_frame(*key, seq[keep], nfl[keep], role[keep], "full", Z[keep], (Z[keep] - mu) / sd)]
    Yt = curves.Y[te]
    obs_t = norm.observed(Yt)
    for where, fraction in ecfg.truncations:
        Zt, _, _ = embed(t.model, norm, Yt, device, visible=rv.truncation_mask(obs_t, where, fraction))
        parts.append(rv.embedding_frame(*key, seq[te], nfl[te], "heldout", rv.view_name(where, fraction), Zt, (Zt - mu) / sd))
    sup = norm.support if support_eval is None else norm.support & support_eval
    on_sup = np.isfinite(Yt) & sup[None, :]
    embedded = np.isfinite(Z[te][:, :1])
    hidden = rv.evaluation_mask(seq[te], on_sup, seed)
    Zm, Rm, _ = embed(t.model, norm, Yt, device, visible=~hidden)
    recon = rv.recon_frame(*key, seq[te], nfl[te], frac[te], on_sup & embedded, Yt - R[te], Yt - norm.mean,
                           hidden & embedded & np.isfinite(Zm[:, :1]), Yt - Rm, Yt - norm.mean)
    counts = {"n_heldout_athletes": int(np.unique(nfl[te]).size), "n_heldout_sequences": int(te.sum()),
              "n_heldout_embedded": int(embedded.sum()), "n_eval_support": int(sup.sum())}
    return pl.concat(parts), recon, counts


def evaluation_supports(curves: dict[str, CurveSet], splits: pl.DataFrame, fpca_models: dict[str, dict[str, FpcaModel]],
                        repeats: Sequence[int], tcfg: TcnConfig) -> dict[tuple[str, int, int], np.ndarray]:
    """(representation, repeat, fold) → grid points on which reconstruction errors are counted for EVERY method:
    the support of the fold's normaliser AND of the fold's FPCA fit (both learned on the same training athletes
    with the same rule, so they normally coincide)."""
    out = {}
    for name, cs in curves.items():
        for r in repeats:
            for f in sorted(splits.filter(pl.col("repeat") == r)["outer_fold"].unique().to_list()):
                train, _ = fold_athletes(splits, r, f)
                sup = CurveNormalizer.fit(cs, train, tcfg.min_coverage, tcfg.min_observed_fraction).support
                m = fpca_models.get(name, {}).get(fold_tag(r, f))
                out[(name, int(r), int(f))] = sup & m.support if m is not None else sup
    return out


def fit_methods(curves: dict[str, CurveSet], splits: pl.DataFrame, plan: Sequence[tuple[MethodSpec, TcnConfig]],
                repeats: Sequence[int], ecfg: EvalConfig, seed: int, device: str = "cpu",
                supports: dict[tuple[str, int, int], np.ndarray] | None = None, models_dir: Path | None = None
                ) -> tuple[dict[str, pl.DataFrame], dict[tuple[str, int, int], Trained]]:
    """Train every (method, split repeat, outer fold) of ``plan``, freeze, embed and score the held-out athletes.
    Returns the tables keyed by base contract name and the fitted fold models. Checkpoints are written when
    ``models_dir`` is given."""
    emb, rec, hist, fits = [], [], [], []
    models: dict[tuple[str, int, int], Trained] = {}
    for spec, tcfg in plan:
        cs = curves[spec.representation]
        for r in repeats:
            for f in sorted(splits.filter(pl.col("repeat") == r)["outer_fold"].unique().to_list()):
                t = train_fold(cs, splits, r, f, spec, tcfg, seed, device)
                e, q, counts = tcn_fold_outputs(t, cs, splits, (supports or {}).get((spec.representation, int(r), int(f))),
                                                ecfg, seed, device)
                t.info.update(counts)
                key = {"method": spec.method, "repeat": int(r), "outer_fold": int(f)}
                emb.append(e)
                rec.append(q)
                hist += [{**key, **h} for h in t.history]
                fits.append({**key, "representation": spec.representation, "objective": tcfg.objective,
                             "embedding_dim": tcfg.embedding_dim, **t.info})
                models[(spec.method, int(r), int(f))] = t
                if models_dir is not None:
                    save_checkpoint(checkpoint_path(models_dir, spec.method, r, f), t)
                log.info("[advanced_movement:TCN] %s %s: %d epochs (best %d), %.1f s, %d held-out attempts embedded",
                         spec.method, fold_tag(r, f), t.info["epochs_run"], t.info["best_epoch"], t.info["train_seconds"],
                         counts["n_heldout_embedded"])
    return ({rv.EMBEDDINGS.name: pl.concat(emb), rv.RECON_SEQ.name: pl.concat(rec), HISTORY.name: rv._table(hist, HISTORY),
             FOLDS.name: rv._table(fits, FOLDS)}, models)


def fpca_reference(curves: dict[str, CurveSet], splits: pl.DataFrame, specs: Sequence[MethodSpec],
                   fpca_models: dict[str, dict[str, FpcaModel]], supports: dict[tuple[str, int, int], np.ndarray],
                   repeats: Sequence[int], ecfg: EvalConfig, seed: int) -> dict[str, pl.DataFrame]:
    """The fitted FPCA fold models of notebook 02 re-expressed as embeddings and reconstruction sums on the same
    held-out attempts, views, evaluation mask and support as the encoders. Nothing is refitted or re-tuned."""
    emb, rec = [], []
    for spec in specs:
        cs = curves[spec.representation]
        for r in repeats:
            for f in sorted(splits.filter(pl.col("repeat") == r)["outer_fold"].unique().to_list()):
                train, test = fold_athletes(splits, r, f)
                e, q = rv.fpca_fold_outputs(spec, cs, fpca_models[spec.representation][fold_tag(r, f)], int(r), int(f), train,
                                            test, supports[(spec.representation, int(r), int(f))], ecfg, seed)
                emb.append(e)
                rec.append(q)
    return {rv.EMBEDDINGS.name: pl.concat(emb), rv.RECON_SEQ.name: pl.concat(rec)}


def stage_contracts(prefix: str, training: bool = True) -> list[TableContract]:
    base = (rv.EMBEDDINGS, rv.RECON_SEQ, HISTORY, FOLDS) if training else (rv.EMBEDDINGS, rv.RECON_SEQ)
    return [rv.named(c, prefix) for c in base]


def stage_tables(prefix: str, tables: dict[str, pl.DataFrame], training: bool = True) -> dict[str, pl.DataFrame]:
    base = (rv.EMBEDDINGS, rv.RECON_SEQ, HISTORY, FOLDS) if training else (rv.EMBEDDINGS, rv.RECON_SEQ)
    return {rv.named(c, prefix).name: tables[c.name] for c in base}


def unstage(prefix: str, tables: dict[str, pl.DataFrame], training: bool = True) -> dict[str, pl.DataFrame]:
    """Inverse of :func:`stage_tables`: stage-specific names back to the base contract names."""
    base = (rv.EMBEDDINGS, rv.RECON_SEQ, HISTORY, FOLDS) if training else (rv.EMBEDDINGS, rv.RECON_SEQ)
    return {c.name: tables[rv.named(c, prefix).name] for c in base}


def run_stage(store: Any, stage: str, prefix: str, curves: dict[str, CurveSet], splits: pl.DataFrame,
              plan: Sequence[tuple[MethodSpec, TcnConfig]], repeats: Sequence[int], ecfg: EvalConfig, seed: int, device: str,
              supports: dict[tuple[str, int, int], np.ndarray], params: dict[str, Any], files: Sequence[Path],
              models: dict[tuple[str, int, int], Trained]) -> dict[str, pl.DataFrame]:
    """One cached training stage of a notebook (``store``: ``AdvStore``). Trains every fold model of ``plan`` unless
    the stage's tables are current for these parameters and input files, in which case they are restored and the
    fold models are reloaded from their checkpoints (when saved under the same key). ``models`` is filled in place;
    tables come back under the base contract names."""
    if not plan:
        raise TcnError(f"stage {stage}: nothing to train")
    full = {**params, "plan": [{"method": s.method, "representation": s.representation, **t.params()} for s, t in plan]}
    key = fit_key(full, files)
    save = bool(store.controls.save_models)

    def builder() -> dict[str, pl.DataFrame]:
        tables, fitted = fit_methods(curves, splits, plan, repeats, ecfg, seed, device, supports, store.models_dir if save else None)
        models.update(fitted)
        if save:
            record_models(store.models_dir, stage, key, [s.method for s, _ in plan])
        return stage_tables(prefix, tables)
    out = store.build(stage, stage_contracts(prefix), builder, params=full, files=files)
    if not any(k[0] == plan[0][0].method for k in models):
        models.update(load_models(store.models_dir, stage, key, plan, splits, repeats))
    return unstage(prefix, out)


def masked_examples(curves: CurveSet, recon_seq: pl.DataFrame, method: str, tcn_methods: dict[str, tuple[str, str]],
                    models: dict[tuple[str, int, int], Trained], fpca_spec: MethodSpec, fpca_models: dict[str, FpcaModel],
                    supports: dict[tuple[str, int, int], np.ndarray], seed: int, device: str = "cpu") -> list[dict[str, Any]]:
    """Held-out attempts at the 10th, 50th, 90th percentile and at the maximum of ``method``'s hidden-point error,
    each with the reconstruction of every available model from the visible points (same evaluation mask as the
    tables). ``tcn_methods``: label → (objective, method). Chosen by rule; a model missing from ``models`` is skipped."""
    d = (recon_seq.filter((pl.col("method") == method) & (pl.col("n_masked") > 0))
         .with_columns((pl.col("sse_masked") / pl.col("n_masked")).sqrt().alias("_rmse")).sort("_rmse", "seq_id"))
    out = []
    for q, note in ((0.1, "10th percentile"), (0.5, "median"), (0.9, "90th percentile"), (1.0, "largest")):
        if not d.height:
            break
        row = d.row(min(int(round(q * (d.height - 1))), d.height - 1), named=True)
        r, f, i = int(row["repeat"]), int(row["outer_fold"]), curves.index_of(row["seq_id"])
        Y = curves.Y[i:i + 1]
        sup = supports[(curves.spec.name, r, f)]
        hidden = rv.evaluation_mask(curves.seq_ids[i:i + 1].astype(str), np.isfinite(Y) & sup[None, :], seed)
        recon: dict[str, tuple[str, np.ndarray]] = {}
        m, k = fpca_models[fold_tag(r, f)], fpca_spec.embedding_dim
        Sc, _ = m.scores(np.where(hidden, np.nan, Y), k)
        if np.isfinite(Sc).all():
            full = np.full(Y.shape[1], np.nan)
            full[m.support] = m.reconstruct(Sc, k)[0]
            recon[fpca_spec.label] = ("fpca", full)
        for label, (objective, name) in tcn_methods.items():
            t = models.get((name, r, f))
            if t is not None:
                _, R, _ = embed(t.model, t.normalizer, Y, device, visible=~hidden)
                recon[label] = (objective, R[0])
        out.append({"title": f"{note} of the {method} hidden-point error · RMSE {row['_rmse']:.3f} {SPEC[curves.spec.name].value_unit}",
                    "grid": curves.grid, "y": np.where(sup, Y[0], np.nan), "hidden": hidden[0], "recon": recon})
    return out


# ============================================================================== methods
def tcn_plan(tcfg_a: TcnConfig | None, tcfg_b: TcnConfig | None, representations: Sequence[str], fpca_k: dict[str, int],
             suffix: str = "", primary: bool = True) -> list[tuple[MethodSpec, TcnConfig]]:
    """One single-input encoder per representation and objective. Inputs are never combined."""
    out = []
    for rep in representations:
        ref = rv.fpca_method(rep, fpca_k[rep]).method
        for letter, tcfg in (("A", tcfg_a), ("B", tcfg_b)):
            if tcfg is None:
                continue
            want = RECON if letter == "A" else CONSISTENCY
            if tcfg.objective != want:
                raise TcnError(f"TCN-{letter} needs objective {want!r}, got {tcfg.objective!r}")
            out.append((MethodSpec(f"TCN{letter}_{rep[0]}{suffix}", "tcn", tcfg.objective, rep, tcfg.embedding_dim,
                                   f"TCN-{letter} · {rep[0]} ({tcfg.embedding_dim}-d)", ref, primary), tcfg))
    return out


def methods_table(specs: Sequence[MethodSpec], fold_fits: pl.DataFrame, fpca_models: dict[str, dict[str, FpcaModel]]
                  ) -> pl.DataFrame:
    rows = []
    for s in specs:
        row = {**asdict(s), "value_unit": SPEC[s.representation].value_unit, "n_parameters": None, "train_seconds": None}
        if s.family == "fpca":
            row["n_parameters"] = (s.embedding_dim + 1) * fpca_models[s.representation]["reference"].n_basis
        else:
            ff = fold_fits.filter(pl.col("method") == s.method)
            if ff.height:
                row.update(n_parameters=int(ff["n_parameters"][0]), train_seconds=float(ff["train_seconds"].sum()))
        rows.append(row)
    return rv._table(rows, rv.METHODS)


# ============================================================================== design-only selection of the consistency
def design_sweep(curves: CurveSet, splits: pl.DataFrame, meta: Meta, base: TcnConfig, grid: Sequence[tuple[str, float]],
                 ecfg: EvalConfig, seed: int, device: str = "cpu") -> pl.DataFrame:
    """Compare consistency kinds and weights using DESIGN-role athletes only. Inside them, for every inner fold k:
    evaluated = fold k · early stopping = the next fold · weights = the remaining design athletes. Confirm-role
    athletes are never loaded into a model here, so nothing chosen from this table has seen them."""
    s = splits.filter((pl.col("repeat") == splits["repeat"].min()) & (pl.col("design_role") == "design")).sort("nfl_id")
    ids, inner = s["nfl_id"].to_numpy(), s["inner_fold"].to_numpy()
    folds = sorted(set(inner.tolist()))
    if len(folds) < 3:
        raise TcnError(f"the design subset has {len(folds)} inner fold(s); three are needed (weights, early stopping, evaluation)")
    configs = [("none", 0.0, replace(base, objective=RECON))]
    configs += [(kind, float(w), replace(base, objective=CONSISTENCY, consistency=kind, consistency_weight=float(w))) for kind, w in grid]
    rows = []
    for kind, weight, tcfg in configs:
        auc, top1, fold_auc, rank, near = [], [], [], [], []
        sse, n_pts, n_folds, collapsed = 0.0, 0, 0, False
        for j, k in enumerate(folds):
            nxt = folds[(j + 1) % len(folds)]
            ev, dv, ft = ids[inner == k], ids[inner == nxt], ids[~np.isin(inner, [k, nxt])]
            try:
                t = train_encoder(curves, np.r_[ft, dv], ft, dv, tcfg, seed, device, tag=f"design:{kind}:{weight:g}:{k}")
            except TcnError as e:
                log.warning("[advanced_movement:TCN] design sweep, %s weight %g, fold %s skipped: %s", kind, weight, k, e)
                continue
            t.normalizer.assert_unseen(ev.tolist(), f"design sweep {kind} {weight:g}, fold {k}")
            rows_ev = np.flatnonzero(np.isin(curves.nfl_ids, ev))
            Yev = curves.Y[rows_ev]
            Z, _, _ = embed(t.model, t.normalizer, Yev, device)
            Zt, _, _ = embed(t.model, t.normalizer, curves.Y[np.isin(curves.nfl_ids, np.r_[ft, dv])], device)
            sd = Zt[np.isfinite(Zt).all(1)].std(0, ddof=1)
            ok = np.isfinite(Z).all(1)
            Zs = Z[ok] / np.where(sd > 1e-12, sd, 1.0)
            m = rv.collapse_metrics(Z[ok], ecfg.near_identical)
            rank.append(m["rank_share"])
            near.append(m["near_identical_share"])
            collapsed |= rv.is_collapsed(m, ecfg)
            r = meta.rows(curves.seq_ids[rows_ev][ok].astype(str))
            a_ids, i1, i2 = first_two(curves.nfl_ids[rows_ev][ok], meta.order[r])
            if a_ids.size >= 3:
                st = rv.pair_stats(Zs[i1], Zs[i2], meta.position[r][i1], ecfg.top_k)
                has = st["n_candidates"] > 0
                if has.any():
                    auc.append(st["auc"][has])
                    top1.append(st["top1"][has])
                    fold_auc.append(float(st["auc"][has].mean()))
            hidden = rv.evaluation_mask(curves.seq_ids[rows_ev].astype(str), t.normalizer.observed(Yev), seed)
            Zm, Rm, _ = embed(t.model, t.normalizer, Yev, device, visible=~hidden)
            counted = hidden & np.isfinite(Zm[:, :1])
            sse += float((np.where(counted, Yev - Rm, 0.0) ** 2).sum())
            n_pts += int(counted.sum())
            n_folds += 1
        a = np.concatenate(auc) if auc else np.empty(0)
        rows.append({"consistency": kind, "consistency_weight": weight, "n_folds": n_folds, "n_athletes_repeat": int(a.size),
                     "auc": float(a.mean()) if a.size else None, "auc_fold_min": min(fold_auc) if fold_auc else None,
                     "auc_fold_max": max(fold_auc) if fold_auc else None,
                     "top1": float(np.concatenate(top1).mean()) if top1 else None,
                     "rmse_masked": float(np.sqrt(sse / n_pts)) if n_pts else None,
                     "rank_share_min": float(np.nanmin(rank)) if rank and np.isfinite(rank).any() else None,
                     "near_identical_max": float(np.nanmax(near)) if near and np.isfinite(near).any() else None,
                     "collapsed": bool(collapsed), "selected": False})
    return rv._table(rows, DESIGN)


def select_consistency(sweep: pl.DataFrame, default: tuple[str, float], tolerance: float = 0.005
                       ) -> tuple[pl.DataFrame, str, float, str]:
    """(sweep with ``selected`` set, kind, weight, reason). Rule fixed before running: among the candidates that did
    not collapse, the largest design AUC; candidates within ``tolerance`` of it are tied and the smallest weight
    wins, then 'vicreg'. If nothing can be ranked, the pre-declared default is kept and the reason says so."""
    c = sweep.filter((pl.col("consistency") != "none") & ~pl.col("collapsed") & pl.col("auc").is_not_null())
    if not c.height:
        kind, weight = default
        reason = "no candidate could be ranked on the design athletes (too few pairs or every candidate collapsed): pre-declared default kept"
    else:
        tied = c.filter(pl.col("auc") >= c["auc"].max() - tolerance).sort(["consistency_weight", "consistency"], descending=[False, True])
        kind, weight = tied["consistency"][0], float(tied["consistency_weight"][0])
        reason = (f"largest design AUC among {c.height} non-collapsed candidate(s); {tied.height} within {tolerance:g} of it; "
                  "smallest weight, then vicreg")
    marked = sweep.with_columns(((pl.col("consistency") == kind) & ((pl.col("consistency_weight") - weight).abs() < 1e-12)).alias("selected"))
    return marked, kind, weight, reason
