"""Fold-safe Combine representations for the transfer study (Milestone 06). Needs PyTorch (the TCN of notebook 03).

For every outer fold of the encoder split table (``transfer.encoder_splits``) each family is produced for EVERY tracked
attempt, by something fitted on the training athletes of that fold only:

    ENG        frozen engineered features of the shortlist      deterministic per attempt — nothing is fitted
    SPEED2     p95 speed + interior peak speed (04B control)    deterministic per attempt — nothing is fitted
    FPCA       K scores of ``fpca.fit_fpca``                    support, mean and eigenfunctions from training athletes
    TCN        8-d embedding of the frozen TCN-B                normaliser, weights and early stopping from training athletes;
                                                                a saved checkpoint is reused only when the 06A audit cleared it
                                                                AND the disjointness is re-asserted here
    RAND       same architecture, random weights, never trained normaliser from training athletes
    TCN_PC<k>  TCN projected on k principal axes                axes of the TRAINING attempts' standardised embeddings
    TCN_s<i>   TCN-B seed replicates                            as TCN, another training seed

No NFL value is an argument of any function here: representations cannot be tuned on a label. An attempt a family
cannot represent (too little of the support recorded, a feature not measured) has NO row — nothing is imputed, and
the athlete then drops out of the paired cohort in notebook 06B with a recorded reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

from ...data.utils import get_logger
from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from . import fpca as fp
from . import robust as rb
from . import tcn
from .data import assert_disjoint, fold_athletes
from .fpca import CurveSet, FpcaConfig
from .repr_eval import MethodSpec
from .tcn import TcnConfig, Trained

log = get_logger(__name__)
S, I, F = pl.String(), pl.Int64(), pl.Float64()
SCALAR_FAMILIES = ("ENG", "SPEED2")

ATTEMPT_REP = TableContract(
    "transfer_attempt_representations", "Attempt-level Combine representations of every fold model (long form).",
    {"scheme": S, "family": S, "fold": I, "seq_id": S, "nfl_id": I, "role": S, "dim": I, "value": F},
    key=("scheme", "family", "fold", "seq_id", "dim"), kind="dataset", area="scheme × family × fold × sequence × dimension",
    docs={"scheme": "outer = folds of the primary benchmark · loco = fold is a draft cohort",
          "role": "heldout = the fold's fit never saw the athlete · train = the athlete took part in it",
          "value": "raw value; scaling happens inside the prediction model, on training athletes",
          "dim": "axes of different folds are NOT the same axes: never compare a dimension across folds"})
ENC_FITS = TableContract(
    "transfer_encoder_fits", "How each representation of each fold was obtained, and its isolation from the held-out athletes.",
    {"scheme": S, "family": S, "fold": I, "source": S, "n_train_athletes": I, "n_heldout_athletes": I, "n_overlap": I,
     "n_attempts_valid": I, "n_heldout_valid": I, "embedding_dim": I, "epochs_run": I, "best_epoch": I, "train_seconds": F,
     "seed": I, "checkpoint": S}, key=("scheme", "family", "fold"), area="scheme × family × fold",
    docs={"source": "deterministic · fitted (FPCA) · trained · reused_checkpoint · untrained · projection",
          "n_overlap": "athletes that took part in the fit AND are held out in the fold — must be 0"})


def named(contract: TableContract, suffix: str) -> TableContract:
    return replace(contract, name=f"{contract.name}_{suffix}") if suffix else contract


@dataclass(frozen=True)
class RepPlan:
    tcfg: TcnConfig
    fcfg: FpcaConfig
    fpca_k: int
    seed: int                                              # training seed of the primary encoder (notebook 03 / 04B)
    families: tuple[str, ...] = ("ENG", "SPEED2", "FPCA", "TCN", "RAND")
    compact_dims: tuple[int, ...] = (2, 4)
    replicates: tuple[tuple[str, int], ...] = ()           # (("TCN_s1", seed), …): extra TCN-B training seeds

    def params(self) -> dict[str, Any]:
        return {"tcfg": self.tcfg.params(), "fcfg": self.fcfg.params(), "fpca_k": self.fpca_k, "seed": self.seed,
                "families": list(self.families), "compact_dims": list(self.compact_dims), "replicates": [list(r) for r in self.replicates]}


# ============================================================================== inputs
def curve_rows(cs: CurveSet) -> pl.DataFrame:
    """(seq_id, nfl_id, has_curve): attempts whose curve holds at least one recorded point."""
    return pl.DataFrame({"seq_id": cs.seq_ids.astype(str), "nfl_id": cs.nfl_ids.astype(np.int64),
                         "has_curve": np.isfinite(cs.Y).any(1)}, schema={"seq_id": S, "nfl_id": I, "has_curve": pl.Boolean()})


def scalar_matrix(attempt_values: pl.DataFrame, feature_ids: Sequence[str], seq_ids: np.ndarray) -> np.ndarray:
    """(n, len(feature_ids)) values of per-attempt scalar features aligned to ``seq_ids``; NaN where not measured."""
    M = np.full((len(seq_ids), len(feature_ids)), np.nan)
    row, col = {str(s): i for i, s in enumerate(seq_ids)}, {f: j for j, f in enumerate(feature_ids)}
    w = attempt_values.filter(pl.col("feature_id").is_in(list(feature_ids)) & pl.col("value").is_not_null())
    for f, s, v in w.select("feature_id", "seq_id", "value").iter_rows():
        if s in row:
            M[row[s], col[f]] = v
    return M


def checkpoint_records(tcn_dirs: dict[str, Path], fpca_file: Path | None, representation: str, rel: Any = str
                       ) -> list[dict[str, Any]]:
    """Every saved fit of ``representation`` with the athletes that touched it — the input of
    ``transfer.compatibility_table``. ``tcn_dirs``: label → model directory holding ``tcn/*.pt``."""
    out: list[dict[str, Any]] = []
    for label, d in tcn_dirs.items():
        for p in sorted((Path(d) / "tcn").glob("*.pt")):
            t = tcn.load_checkpoint(p)
            if t.representation != representation:
                continue
            out.append({"artifact": rel(p), "family": f"tcn ({label})", "method": t.method, "repeat": t.repeat, "fold": t.outer_fold,
                        "athletes": {*t.fit_athletes, *t.dev_athletes, *t.normalizer.fit_athletes}})
    if fpca_file is not None and Path(fpca_file).is_file():
        for tag, m in sorted(fp.load_models(Path(fpca_file)).items()):
            rep, fold = (-1, -1) if tag == "reference" else (int(tag.split("_f")[0][1:]), int(tag.split("_f")[1]))
            out.append({"artifact": rel(fpca_file), "family": "fpca", "method": f"FPCA:{tag}", "repeat": rep, "fold": fold,
                        "athletes": set(m.fit_athletes)})
    return out


# ============================================================================== one fold
def _long(scheme: str, family: str, fold: int, seq: np.ndarray, nfl: np.ndarray, role: np.ndarray, Z: np.ndarray) -> pl.DataFrame:
    ok = np.isfinite(Z).all(1) & (role != "other")
    n, E = int(ok.sum()), Z.shape[1]
    df = pl.DataFrame({"seq_id": np.repeat(seq[ok], E), "nfl_id": np.repeat(nfl[ok], E), "role": np.repeat(role[ok], E),
                       "dim": np.tile(np.arange(1, E + 1), n), "value": Z[ok].ravel()},
                      schema={"seq_id": S, "nfl_id": I, "role": S, "dim": I, "value": F})
    return df.with_columns(pl.lit(scheme).alias("scheme"), pl.lit(family).alias("family"),
                           pl.lit(int(fold), dtype=pl.Int64).alias("fold")).select(list(ATTEMPT_REP.columns))


def _encoder(cs: CurveSet, enc_splits: pl.DataFrame, fold: int, family: str, seed: int, plan: RepPlan, device: str,
             models_dir: Path | None, reuse: dict[tuple[str, int], str] | None, scheme: str) -> tuple[Trained, str, str]:
    """(frozen encoder, source, checkpoint path). A reusable checkpoint is still re-checked against the held-out
    athletes of THIS fold; an overlap raises ``LeakageError`` instead of falling back silently."""
    _, test = fold_athletes(enc_splits, 0, fold)
    path = (reuse or {}).get((family, int(fold)))
    if path is not None:
        t = tcn.load_checkpoint(Path(path))
        assert_disjoint(sorted({*t.fit_athletes, *t.dev_athletes, *t.normalizer.fit_athletes}), test,
                        f"saved encoder {Path(path).name} vs held-out athletes of {scheme} fold {fold}")
        if t.representation != cs.spec.name or t.model.n_grid != cs.grid.size or t.tcfg.embedding_dim != plan.tcfg.embedding_dim:
            raise tcn.TcnError(f"{Path(path).name} was trained on another input or size than the frozen TCN-B")
        return t, "reused_checkpoint", str(path)
    spec = MethodSpec(f"{family}__{scheme}", "tcn", plan.tcfg.objective, cs.spec.name, plan.tcfg.embedding_dim,
                      f"{family} · {scheme} fold model", "", True)
    t = tcn.train_fold(cs, enc_splits, 0, int(fold), spec, plan.tcfg, seed, device)
    saved = str(tcn.save_checkpoint(tcn.checkpoint_path(models_dir, spec.method, 0, int(fold)), t)) if models_dir is not None else ""
    log.info("[advanced_movement:TRANSFER] %s %s fold %s: trained, %d epochs (best %d), %.1f s", family, scheme, fold,
             t.info["epochs_run"], t.info["best_epoch"], t.info["train_seconds"])
    return t, "trained", saved


def compact_projection(Z: np.ndarray, train_rows: np.ndarray, dims: Sequence[int]) -> dict[int, np.ndarray]:
    """k → (n, k) coordinates on the first k principal axes of the TRAINING rows' standardised embeddings. Rows
    without an embedding stay NaN. Axis signs are arbitrary (a linear model does not depend on them)."""
    tr = train_rows & np.isfinite(Z).all(1)
    if tr.sum() < 3:
        return {}
    mu, sd = Z[tr].mean(0), Z[tr].std(0, ddof=1)
    Zs = (Z - mu) / np.where(sd > 1e-12, sd, 1.0)
    m, V, _, _ = rb.training_axes(Zs[tr])
    return {int(k): (Zs - m) @ V[:, :k] for k in dims if 1 <= k <= Z.shape[1]}


def fold_representations(cs: CurveSet, enc_splits: pl.DataFrame, folds: Sequence[int], plan: RepPlan,
                         scalars: dict[str, np.ndarray], device: str = "cpu", models_dir: Path | None = None,
                         reuse: dict[tuple[str, int], str] | None = None, scheme: str = "outer") -> dict[str, pl.DataFrame]:
    """Attempt-level representations of every family of ``plan`` for every fold of ``folds``.

    ``scalars``: family → (n attempts, d) matrix aligned to ``cs.seq_ids`` (``scalar_matrix``). ``reuse``:
    (family, fold) → checkpoint cleared by the 06A audit. Returns {ATTEMPT_REP.name, ENC_FITS.name}."""
    seq, nfl = cs.seq_ids.astype(str), cs.nfl_ids.astype(np.int64)
    parts, fits = [], []
    for f in folds:
        train, test = fold_athletes(enc_splits, 0, int(f))
        tr, te = np.isin(nfl, np.asarray(train, dtype=np.int64)), np.isin(nfl, np.asarray(test, dtype=np.int64))
        role = np.where(te, "heldout", np.where(tr, "train", "other"))
        base = {"scheme": scheme, "fold": int(f), "n_train_athletes": len(train), "n_heldout_athletes": len(test), "n_overlap": 0,
                "epochs_run": None, "best_epoch": None, "train_seconds": None, "seed": None, "checkpoint": ""}

        def add(family: str, Z: np.ndarray, source: str, **info: Any) -> None:
            ok = np.isfinite(Z).all(1) & (role != "other")
            parts.append(_long(scheme, family, int(f), seq, nfl, role, Z))
            fits.append({**base, "family": family, "source": source, "n_attempts_valid": int(ok.sum()),
                         "n_heldout_valid": int((ok & te).sum()), "embedding_dim": int(Z.shape[1]), **info})

        for family in SCALAR_FAMILIES:
            if family in plan.families and family in scalars and scalars[family].shape[1]:
                add(family, scalars[family], "deterministic")
        if "FPCA" in plan.families:
            m = fp.fit_fpca(cs, train, plan.fcfg, f"{cs.spec.name}:{scheme}{f}")
            m.assert_unseen(test, f"FPCA of {scheme} fold {f} vs its held-out athletes")
            add("FPCA", m.scores(cs.Y, plan.fpca_k)[0], "fitted")
        encoders = ([("TCN", plan.seed)] if "TCN" in plan.families else []) + [r for r in plan.replicates if r[0] in plan.families]
        for family, seed in encoders:
            t, source, saved = _encoder(cs, enc_splits, int(f), family, int(seed), plan, device, models_dir, reuse, scheme)
            touched = {*t.fit_athletes, *t.dev_athletes, *t.normalizer.fit_athletes}
            Z = tcn.embed(t.model, t.normalizer, cs.Y, device)[0]
            add(family, Z, source, n_overlap=len(touched & set(test)), epochs_run=t.info.get("epochs_run"),
                best_epoch=t.info.get("best_epoch"), train_seconds=t.info.get("train_seconds"), seed=int(seed), checkpoint=saved)
            if family == "TCN":
                for k, P in compact_projection(Z, tr, plan.compact_dims).items():
                    add(f"TCN_PC{k}", P, "projection", seed=int(seed))
        if "RAND" in plan.families:
            norm = tcn.CurveNormalizer.fit(cs, train, plan.tcfg.min_coverage, plan.tcfg.min_observed_fraction)
            norm.assert_unseen(test, f"normaliser of the random encoder, {scheme} fold {f}")
            torch.manual_seed(seed_for(plan.seed, "untrained", scheme, int(f)))
            add("RAND", tcn.embed(tcn.CurveTCN(plan.tcfg, cs.grid.size).eval(), norm, cs.Y, device)[0], "untrained", seed=int(plan.seed))
    if not parts:
        return {ATTEMPT_REP.name: ATTEMPT_REP.empty(), ENC_FITS.name: ENC_FITS.empty()}
    fit_table = pl.DataFrame(fits, schema={k: ENC_FITS.columns[k] for k in fits[0]}).select(list(ENC_FITS.columns))
    bad = fit_table.filter(pl.col("n_overlap") > 0)
    if bad.height:
        raise RuntimeError(f"representation leakage: {bad.select('family', 'fold', 'n_overlap').to_dicts()}")
    return {ATTEMPT_REP.name: pl.concat(parts), ENC_FITS.name: fit_table}


def build_representations(store: Any, stage: str, suffix: str, cs: CurveSet, enc_splits: pl.DataFrame, folds: Sequence[int],
                          plan: RepPlan, scalars: dict[str, np.ndarray], device: str, reuse: dict[tuple[str, int], str] | None,
                          scheme: str, params: dict[str, Any], files: Sequence[Path]) -> dict[str, pl.DataFrame]:
    """Cached stage of ``fold_representations`` (``store``: ``TransferStore``). The cache key holds the fold manifest
    hash, the encoder configuration, the seeds, the reuse map and the state of the input files (``params``, ``files``).
    Tables come back under the base contract names."""
    contracts = [named(ATTEMPT_REP, suffix), named(ENC_FITS, suffix)]
    save = store.models_dir if store.controls.save_models else None

    def builder() -> dict[str, pl.DataFrame]:
        out = fold_representations(cs, enc_splits, folds, plan, scalars, device, save, reuse, scheme)
        return {contracts[0].name: out[ATTEMPT_REP.name], contracts[1].name: out[ENC_FITS.name]}
    full = {**params, "plan": plan.params(), "scheme": scheme, "folds": [int(f) for f in folds],
            "reuse": sorted(f"{k[0]}:{k[1]}:{Path(v).name}" for k, v in (reuse or {}).items())}
    out = store.build(stage, contracts, builder, params=full, files=files)
    return {ATTEMPT_REP.name: out[contracts[0].name], ENC_FITS.name: out[contracts[1].name]}


# ============================================================================== attribution (only read after a gain)
def occlusion_profile(t: Trained, cs: CurveSet, athletes: Sequence[int], weights: np.ndarray, mean: np.ndarray, sd: np.ndarray,
                      n_bins: int = 8, device: str = "cpu") -> pl.DataFrame:
    """Which part of the curve the fitted read-out of a fold relies on, measured on TRAINING athletes only.

    ``weights`` / ``mean`` / ``sd``: ridge coefficients and training standardisation of the embedding block of the
    fold's prediction model. For each of ``n_bins`` equal slices of the support, the recorded points of the slice are
    hidden from the encoder and the change of the read-out w · (z − mean) / sd is recorded (target units). Descriptive:
    a large value says the prediction depends on that slice, not that the slice causes the NFL outcome."""
    rows = np.flatnonzero(np.isin(cs.nfl_ids, np.asarray(list(athletes), dtype=np.int64)))
    norm, Y = t.normalizer, cs.Y[rows]
    full = tcn.embed(t.model, norm, Y, device)[0]
    ok = np.isfinite(full).all(1)
    read = lambda Z: ((Z - mean) / sd) @ weights                                       # noqa: E731
    sup = np.flatnonzero(norm.support)
    out = []
    for b, idx in enumerate(np.array_split(sup, n_bins)):
        if not idx.size:
            continue
        visible = np.ones(Y.shape, dtype=bool)
        visible[:, idx] = False
        Zb = tcn.embed(t.model, norm, Y, device, visible=visible)[0]
        both = ok & np.isfinite(Zb).all(1)
        shift = read(Zb[both]) - read(full[both])
        out.append({"bin": b, "axis_lo": float(norm.grid[idx[0]]), "axis_hi": float(norm.grid[idx[-1]]), "n_attempts": int(both.sum()),
                    "rms_shift": float(np.sqrt(np.mean(shift ** 2))) if both.any() else None,
                    "mean_shift": float(shift.mean()) if both.any() else None})
    return pl.DataFrame(out, schema={"bin": I, "axis_lo": F, "axis_hi": F, "n_attempts": I, "rms_shift": F, "mean_shift": F})
