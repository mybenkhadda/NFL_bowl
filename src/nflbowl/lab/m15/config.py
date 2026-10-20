"""Configuration and run directory of Milestone 15 — spectral & temporal representation science.

``CFG`` in the notebook is a plain dict; ``resolve(CFG)`` validates it (unknown keys raise, booleans type-checked, every
range checked) and returns a frozen ``M15Config``. ``M15Config.m13()`` maps the shared keys onto an ``M13Config`` so the
M13 hand-off helpers (environment record, experiment opening, checkpoint adoption) are reused unchanged.

Execution policy encoded here:

* M12 / M13 / M14 run directories and the exploratory (legacy) M15 folders are READ ONLY: ``resolve`` refuses a run
  directory inside any of them and an experiment name that reuses an earlier milestone prefix; ``M15Run.guard`` refuses
  writes outside the M15 run directory.
* Neural training needs TWO explicit opt-ins (``run_training`` and ``allow_new_training``).
* ``fingerprint()`` hashes every science-relevant key (not the I/O switches), so caches are invalidated when — and only
  when — the science changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from ...artifacts.store import append_jsonl, atomic_json
from ...config import stable_hash
from ...data.validation import CHANNELS
from ...research.artifacts import ResearchRun, safe_name
from ...utils.reproducibility import code_version, utc_now
from ..m13_config import EXPECTED, M13Run
from ..status import StatusLedger

TRANSFORM_VERSION = "m15_v1"
SAMPLING_HZ = 10.0
SCALE = (10.0, 5.0, 5.0, 3.0, 0.1, 1.0)                      # = m11.representation.SCALE in CHANNELS order (heading_valid unscaled)
UNITS = ("yd/s", "yd/s^2", "yd/s^2", "rad/s", "s", "flag")
ARCHITECTURES = ("pca", "stft_ae", "lstm_ae", "gru_ae", "tcn_ae", "dtcn_ae", "masked_tae", "wavelet_cnn")
MVE_ARCHITECTURES = ("pca", "stft_ae", "lstm_ae")
READ_ONLY_PREFIXES = ("m12_", "m13_", "m14_")

M15_LAYOUT = ("config", "manifests", "logs", "datasets/cached", "datasets/audits", "features/signal", "features/spectral", "checkpoints",
              "embeddings/inductive", "predictions/reconstruction", "legacy/adopted", "tables/dataset", "tables/signal", "tables/spectral",
              "tables/preservation", "tables/latent", "tables/reliability", "tables/retrieval", "tables/legacy", "tables/synthesis",
              "figures/png", "figures/svg", "figures/pdf", "figures/data", "reports/markdown", "reports/html", "model_cards", "diagnostics", "exports")
M15_FAMILIES = ("dataset", "signal", "spectral", "preservation", "latent", "reliability", "retrieval", "legacy", "synthesis")
# keys that never change a scientific result (excluded from the fingerprint)
IO_KEYS = frozenset({"experiment_name", "bundle_dir", "drive_root", "stage_dir", "m12_run_dir", "m13_run_dir", "m14_run_dir", "legacy_m15_dirs",
                     "legacy_split_manifests", "device", "resume", "reuse_artifacts", "save_tables", "save_figures", "save_processed_data", "save_reports",
                     "save_models", "overwrite_figures", "csv_max_rows", "figure_formats", "figure_dpi", "git_ref", "notes", "chunk_rows",
                     "verify_bundle_hashes", "run_training", "allow_new_training"})


@dataclass(frozen=True)
class M15Config:
    # ---- identity and paths
    experiment_name: str = "m15_spectral_temporal_v1"
    bundle_dir: str = "/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean"
    drive_root: str = "/content/drive/MyDrive/NFLBowl"
    stage_dir: str = "/content/nflbowl_stage"                   # fast local copy of the bundle (Colab /content)
    m12_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m12_comprehensive_lab_v1"
    m13_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m13_outcome_transfer_v1"
    m14_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m14_cross_sport_athletic_intelligence_v1"
    m12_experiment_name: str = "m12_comprehensive_lab_v1"
    legacy_m15_dirs: tuple[str, ...] = ()                       # Drive folders of the exploratory STFT / LSTM notebooks (read only)
    legacy_split_manifests: tuple[str, ...] = ()                # explicit split-manifest files for the legacy artifacts (optional)
    # ---- runtime
    device: str = "auto"
    precision: str = "fp32"
    seed: int = 42                                              # analysis seed (sampling, bootstrap); training seeds below
    base_config: str = "colab_gpu"
    expected_fingerprint: str = EXPECTED["fingerprint"]
    expected_science_hash: str = EXPECTED["science_hash"]
    verify_bundle_hashes: bool = False                          # True: re-hash every bundle file (slow on Drive; sizes always checked)
    chunk_rows: int = 8192                                      # rows per chunk when scanning the memory-mapped arrays
    # ---- data contract
    channels: tuple[str, ...] = CHANNELS
    scale: tuple[float, ...] = SCALE
    units: tuple[str, ...] = UNITS
    sampling_hz: float = SAMPLING_HZ
    max_frames: int = 64
    outer_folds: int = 5
    folds: tuple[int, ...] | None = None                        # fold selector for training / evaluation (None = all)
    max_gap_dt: float = 0.15                                    # s — a frame longer than this ends a gap-free segment (M14 rule)
    min_segment_frames: int = 7
    min_linked_per_fold: int = 20                               # below → the fold is BLOCKED_INSUFFICIENT_SAMPLE for cross-domain work
    plausible_speed: tuple[float, float] = (0.0, 13.0)          # yd/s on valid frames
    plausible_atan: tuple[float, float] = (-15.0, 15.0)         # yd/s²
    plausible_dt: tuple[float, float] = (0.0, 0.5)              # s
    nfl_windows_per_athlete_plot: int = 20
    # ---- models (Track B)
    run_training: bool = False
    allow_new_training: bool = False
    architectures: tuple[str, ...] = MVE_ARCHITECTURES
    latent_dims: tuple[int, ...] = (32,)
    train_seeds: tuple[int, ...] = (20261020,)
    train_domains: tuple[str, ...] = ("combine", "nfl")
    train_steps: int = 6000
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience_evals: int = 10
    eval_every: int = 200
    inner_val_share: float = 0.1
    param_band: float = 0.25
    # ---- spectral (Track A)
    stft_nperseg: int = 16
    stft_hop: int = 4
    stft_phase: bool = False
    welch_nperseg: tuple[int, ...] = (16, 32)
    detrend: tuple[str, ...] = ("linear", "none")
    bands_hz: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 5.0))
    # ---- statistics
    n_boot: int = 2000
    n_boot_screen: int = 500
    # ---- caching and outputs
    resume: bool = True
    reuse_artifacts: bool = True
    save_tables: bool = True
    save_figures: bool = True
    save_processed_data: bool = True
    save_reports: bool = True
    save_models: bool = True
    overwrite_figures: bool = True
    figure_formats: tuple[str, ...] = ("png", "svg", "pdf")
    figure_dpi: int = 300
    csv_max_rows: int = 10_000_000
    git_ref: str = "main"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ paths
    @property
    def run_dir(self) -> Path:
        return Path(self.drive_root) / "experimental_lab" / self.experiment_name

    @property
    def m12_experiment_dir(self) -> Path:
        return Path(self.m12_run_dir) / "checkpoints" / "experiments" / self.m12_experiment_name

    def read_only_dirs(self) -> list[Path]:
        return [Path(p) for p in (self.m12_run_dir, self.m13_run_dir, self.m14_run_dir, *self.legacy_m15_dirs) if p]

    def training_enabled(self) -> bool:
        return bool(self.run_training and self.allow_new_training)

    def selected_folds(self) -> tuple[int, ...]:
        return tuple(range(self.outer_folds)) if self.folds is None else tuple(self.folds)

    def scale_map(self) -> dict[str, float]:
        return dict(zip(self.channels, self.scale, strict=True))

    def event_params(self) -> dict[str, Any]:
        """The M14 event-detector parameters this milestone uses (M14 defaults except the shared segment rule)."""
        return {"max_gap_dt": self.max_gap_dt, "min_segment_frames": self.min_segment_frames}

    # ------------------------------------------------------------------ provenance
    def science_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if k not in IO_KEYS}

    def fingerprint(self) -> str:
        """Hash of every science-relevant key (``IO_KEYS`` excluded) + the transform version."""
        return stable_hash({"transform_version": TRANSFORM_VERSION, **self.science_dict()}, 16)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def m13(self) -> Any:
        """The equivalent ``M13Config`` for the reused hand-off helpers (no training through M13 helpers, ever)."""
        from ..m13_config import resolve as m13_resolve

        return m13_resolve({"experiment_name": self.experiment_name, "bundle_dir": self.bundle_dir, "drive_root": self.drive_root,
                            "m12_run_dir": self.m12_run_dir, "m12_experiment_name": self.m12_experiment_name, "stage_dir": self.stage_dir,
                            "device": self.device, "precision": self.precision, "seed": self.seed, "base_config": self.base_config,
                            "expected_fingerprint": self.expected_fingerprint, "expected_science_hash": self.expected_science_hash,
                            "resume": self.resume, "reuse_artifacts": self.reuse_artifacts, "run_training": False, "allow_new_training": False,
                            "n_boot": self.n_boot, "figure_theme": "light"})


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _coerce(default: Any, v: Any, key: str) -> Any:
    if isinstance(default, bool) or (default is None and isinstance(v, bool)):
        if not isinstance(v, bool):
            raise TypeError(f"CFG[{key!r}] must be True/False, got {v!r}")
        return v
    if isinstance(v, list):
        return tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return v


def resolve(cfg: Mapping[str, Any] | None = None) -> M15Config:
    """Validated, frozen configuration. Unknown keys and out-of-range values raise with the allowed values."""
    cfg = dict(cfg or {})
    known = {f.name for f in fields(M15Config)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise KeyError(f"unknown CFG key(s) {unknown}; allowed: {sorted(known)}")
    base = M15Config()
    c = replace(base, **{k: _coerce(getattr(base, k), v, k) for k, v in cfg.items()})
    if c.precision != "fp32":
        raise ValueError("M15 runs in FP32 only (comparability with the M12–M14 FP32 evaluations)")
    if c.experiment_name.lower().startswith(READ_ONLY_PREFIXES):
        raise ValueError(f"experiment_name {c.experiment_name!r} uses a read-only milestone prefix {READ_ONLY_PREFIXES}: choose an m15_* name")
    for ro in c.read_only_dirs():
        if _inside(c.run_dir, ro) or _inside(ro, c.run_dir):
            raise ValueError(f"run directory {c.run_dir} overlaps the read-only directory {ro}")
    if c.run_training and not c.allow_new_training:
        raise ValueError("run_training=True needs allow_new_training=True as a second explicit opt-in")
    if tuple(c.channels) != tuple(CHANNELS):
        raise ValueError(f"channels must equal the bundle contract {CHANNELS}; got {c.channels}")
    if len(c.scale) != len(c.channels) or len(c.units) != len(c.channels):
        raise ValueError("scale and units need one entry per channel")
    if tuple(c.scale) != SCALE:
        raise ValueError(f"scale {c.scale} differs from the bundle's physical scaling {SCALE} (m11.representation.SCALE)")
    if c.folds is not None and (not c.folds or any(not 0 <= int(f) < c.outer_folds for f in c.folds)):
        raise ValueError(f"folds must be None or a non-empty subset of 0..{c.outer_folds - 1}")
    bad = [a for a in c.architectures if a not in ARCHITECTURES]
    if bad or not c.architectures:
        raise ValueError(f"architectures must be a non-empty subset of {ARCHITECTURES}; unknown {bad}")
    if not c.latent_dims or any(int(d) < 2 for d in c.latent_dims):
        raise ValueError("latent_dims must be non-empty integers ≥ 2")
    if not c.train_seeds:
        raise ValueError("train_seeds must not be empty")
    if not set(c.train_domains) <= {"combine", "nfl"} or "combine" not in c.train_domains:
        raise ValueError("train_domains must include 'combine' and may add 'nfl'")
    if not 0.0 < c.inner_val_share < 0.5 or not 0.0 < c.param_band < 1.0:
        raise ValueError("inner_val_share must lie in (0, 0.5) and param_band in (0, 1)")
    if c.train_steps < 1 or c.batch_size < 1 or c.eval_every < 1 or c.patience_evals < 1 or c.lr <= 0 or c.grad_clip <= 0:
        raise ValueError("train_steps, batch_size, eval_every, patience_evals ≥ 1 and lr, grad_clip > 0")
    if not 0.0 < c.max_gap_dt < 1.0 or c.min_segment_frames < 3:
        raise ValueError("max_gap_dt must lie in (0, 1) s and min_segment_frames ≥ 3")
    nyq = c.sampling_hz / 2.0
    if any(not (0.0 <= lo < hi <= nyq) for lo, hi in c.bands_hz):
        raise ValueError(f"bands_hz must be increasing pairs inside [0, {nyq}] Hz (Nyquist at {c.sampling_hz} Hz)")
    if any(not 4 <= int(n) <= c.max_frames for n in (*c.welch_nperseg, c.stft_nperseg)):
        raise ValueError(f"welch_nperseg / stft_nperseg must lie in [4, {c.max_frames}]")
    if not 1 <= c.stft_hop <= c.stft_nperseg:
        raise ValueError("stft_hop must lie in [1, stft_nperseg]")
    if not set(c.detrend) <= {"linear", "constant", "none"}:
        raise ValueError("detrend values must be among 'linear', 'constant', 'none'")
    if c.n_boot < 100 or c.n_boot_screen < 100 or c.chunk_rows < 256:
        raise ValueError("n_boot / n_boot_screen ≥ 100 and chunk_rows ≥ 256")
    return c


class M15Run(M13Run):
    """RUN_DIR tree of M15 (``M15_LAYOUT``): tables per M15 family with provenance, status ledger, config snapshots."""

    def __init__(self, root: Path | str, cfg: M15Config | None = None, local_tmp: Path | str | None = None):
        root = Path(root)
        if cfg is not None:
            for ro in cfg.read_only_dirs():
                if _inside(root, ro):
                    raise ValueError(f"refusing to open an M15 run inside the read-only directory {ro}")
        for d in M15_LAYOUT:
            (root / d).mkdir(parents=True, exist_ok=True)
        ResearchRun.__init__(self, root, cfg, local_tmp)
        self.results: dict[str, Any] = {}
        self.provenance: dict[str, Any] = {"transform_version": TRANSFORM_VERSION, "config_fingerprint": cfg.fingerprint() if cfg else None}
        self.ledger = StatusLedger.open(root, "m15")

    def guard(self, path: Path | str) -> Path:
        """``path`` if it lies inside this run directory; raise otherwise (never write into M12–M14 / legacy folders)."""
        p = Path(path)
        if not _inside(p, self.root):
            raise PermissionError(f"M15 writes only under {self.root}; refused {p}")
        return p

    def table(self, df: Any, name: str, family: str, note: str = "", provenance: dict[str, Any] | None = None) -> list[Path]:
        if family not in M15_FAMILIES:
            raise ValueError(f"unknown table family {family!r}; one of {M15_FAMILIES}")
        if df is None or not getattr(df, "width", 0):
            return []
        out = self.save_table(df, name, csv=None, note=note, subdir=f"tables/{family}")
        if out:
            append_jsonl({"table": f"{family}/{safe_name(name)}", "rows": int(getattr(df, "height", 0)), "written_at": utc_now(), **self.provenance,
                          **(provenance or {})}, self.root / "manifests" / "table_provenance.jsonl")
        return out

    def snapshot_config(self, cfg: M15Config, environment: dict[str, Any] | None = None) -> list[Path]:
        """``config/run_config.json`` (+ history) and ``config/config_<fingerprint>.json`` (immutable per science config)."""
        out = self.write_config(cfg, environment or {})
        snap = {"fingerprint": cfg.fingerprint(), "transform_version": TRANSFORM_VERSION, "config": cfg.to_dict(), "code_version": code_version(),
                "written_at": utc_now()}
        out.append(atomic_json(snap, self.root / "config" / f"config_{cfg.fingerprint()}.json", overwrite=False))
        return out

    def write_status(self, extra: dict[str, Any] | None = None) -> Path:
        """``manifests/m15_run_status.json``: stage ledger + research status per step (resumption overview)."""
        rec = {"updated_at": utc_now(), "transform_version": TRANSFORM_VERSION, "config_fingerprint": self.provenance.get("config_fingerprint"),
               "stages": {k: {kk: v.get(kk) for kk in ("status", "note", "seconds", "cache", "finished_at")} for k, v in self.stage_records().items()},
               "steps": self.ledger.records, **(extra or {})}
        return atomic_json(rec, self.root / "manifests" / "m15_run_status.json", overwrite=True)

    def handoff_markdown(self) -> str:
        """Short Drive-side hand-off: per step status, reason and outputs (written by the notebook's last cell)."""
        lines = [f"# M15 run hand-off — {self.root.name}", "", f"Updated {utc_now()} · config fingerprint `{self.provenance.get('config_fingerprint')}`",
                 "", "| step | status | reason |", "|---|---|---|"]
        for k, v in sorted(self.ledger.records.items()):
            lines.append(f"| {k} | {v.get('status')} | {str(v.get('reason', '')).replace('|', '/')[:160]} |")
        lines += ["", "Steps with status COMPLETED / CACHED are reused on the next run (resume=True); BLOCKED_* steps name the missing input."]
        return "\n".join(lines)


def load_snapshot(run_dir: Path | str, fingerprint: str) -> dict[str, Any]:
    p = Path(run_dir) / "config" / f"config_{fingerprint}.json"
    return json.loads(p.read_text(encoding="utf-8"))


__all__ = ["ARCHITECTURES", "IO_KEYS", "M15Config", "M15Run", "M15_FAMILIES", "M15_LAYOUT", "MVE_ARCHITECTURES", "SAMPLING_HZ", "SCALE",
           "TRANSFORM_VERSION", "UNITS", "load_snapshot", "resolve"]
