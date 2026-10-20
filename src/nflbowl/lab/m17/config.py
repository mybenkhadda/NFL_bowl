"""Configuration and run directory of Milestone 17 — context-aware athletic capability learning.

Same pattern as M15 / M16 (``resolve`` → frozen ``M17Config``; ``M17Run`` with read-only guards, status ledger, config
snapshot). M12–M16 run directories are read only. Neural alignment (Track D InfoNCE) needs both training opt-ins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from ...artifacts.store import append_jsonl, atomic_json
from ...config import stable_hash
from ...research.artifacts import ResearchRun, safe_name
from ...utils.reproducibility import code_version, utc_now
from ..m13_config import EXPECTED, M13Run
from ..status import StatusLedger

TRANSFORM_VERSION = "m17_v1"
READ_ONLY_PREFIXES = ("m12_", "m13_", "m14_", "m15_", "m16_")
M17_LAYOUT = ("config", "manifests", "logs", "datasets/cached", "features/events_v2", "checkpoints/align", "tables/events_v2", "tables/context",
              "tables/variance", "tables/signatures", "tables/fusion", "tables/alignment", "tables/synthesis", "figures/png", "figures/svg", "figures/pdf",
              "figures/data", "reports/markdown", "reports/html", "model_cards", "diagnostics", "exports")
M17_FAMILIES = ("events_v2", "context", "variance", "signatures", "fusion", "alignment", "synthesis")
IO_KEYS = frozenset({"experiment_name", "bundle_dir", "drive_root", "stage_dir", "m12_run_dir", "m13_run_dir", "m14_run_dir", "m15_run_dir", "m16_run_dir",
                     "device", "resume", "reuse_artifacts", "save_tables", "save_figures", "save_processed_data", "save_reports", "save_models",
                     "overwrite_figures", "csv_max_rows", "figure_formats", "figure_dpi", "git_ref", "notes", "run_training", "allow_new_training"})


@dataclass(frozen=True)
class M17Config:
    experiment_name: str = "m17_context_athletic_signatures_v1"
    bundle_dir: str = "/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean"
    drive_root: str = "/content/drive/MyDrive/NFLBowl"
    stage_dir: str = "/content/nflbowl_stage"
    m12_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m12_comprehensive_lab_v1"
    m12_experiment_name: str = "m12_comprehensive_lab_v1"
    m13_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m13_outcome_transfer_v1"
    m14_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m14_cross_sport_athletic_intelligence_v1"
    m15_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m15_spectral_temporal_v1"
    m16_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m16_verified_baselines_v1"
    device: str = "auto"
    precision: str = "fp32"
    seed: int = 42
    base_config: str = "colab_gpu"
    expected_fingerprint: str = EXPECTED["fingerprint"]
    expected_science_hash: str = EXPECTED["science_hash"]
    # ---- Track A: event diagnosis / context
    brake_thresh_grid: tuple[float, ...] = (0.5, 1.0, 1.5)
    sustain_grid: tuple[int, ...] = (1, 2, 3)
    drop_frac_grid: tuple[float, ...] = (0.15, 0.25, 0.35)
    grid_nfl_sample: int = 6000                              # NFL rows re-extracted per grid variant (Combine rows: all)
    entry_speed_bins: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
    decel_demand_bins: tuple[float, ...] = (0.25, 0.5, 0.75)
    turn_bins_deg: tuple[float, ...] = (20.0, 90.0)
    min_stratum_athletes: int = 30
    min_events_per_athlete: int = 2
    # ---- Track B: hierarchical
    min_reliability_athletes: int = 30
    use_statsmodels: bool = True
    # ---- Track C / D
    run_fusion: bool = True
    run_alignment: bool = True
    run_training: bool = False
    allow_new_training: bool = False
    allow_descriptive: bool = False
    # ---- signatures / gate
    icc_hi: float = 0.6
    icc_lo: float = 0.4
    athlete_share_min: float = 0.4
    novelty_max_r2: float = 0.5
    split_half_min: float = 0.5
    label_min_abs_rho: float = 0.5
    n_boot: int = 1000
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

    @property
    def run_dir(self) -> Path:
        return Path(self.drive_root) / "experimental_lab" / self.experiment_name

    def read_only_dirs(self) -> list[Path]:
        return [Path(p) for p in (self.m12_run_dir, self.m13_run_dir, self.m14_run_dir, self.m15_run_dir, self.m16_run_dir) if p]

    def training_enabled(self) -> bool:
        return bool(self.run_training and self.allow_new_training)

    def fingerprint(self) -> str:
        return stable_hash({"transform_version": TRANSFORM_VERSION, **{k: v for k, v in asdict(self).items() if k not in IO_KEYS}}, 16)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def m13(self) -> Any:
        from ..m13_config import resolve as m13_resolve

        return m13_resolve({"experiment_name": self.experiment_name, "bundle_dir": self.bundle_dir, "drive_root": self.drive_root, "m12_run_dir": self.m12_run_dir,
                            "m12_experiment_name": self.m12_experiment_name, "stage_dir": self.stage_dir, "device": self.device, "precision": self.precision,
                            "seed": self.seed, "base_config": self.base_config, "expected_fingerprint": self.expected_fingerprint,
                            "expected_science_hash": self.expected_science_hash, "resume": self.resume, "reuse_artifacts": self.reuse_artifacts,
                            "run_training": False, "allow_new_training": False, "n_boot": self.n_boot, "figure_theme": "light"})

    def m15(self) -> Any:
        from ..m15.config import resolve

        return resolve({"bundle_dir": self.bundle_dir, "drive_root": self.drive_root, "stage_dir": self.stage_dir, "expected_fingerprint": self.expected_fingerprint,
                        "expected_science_hash": self.expected_science_hash, "m12_run_dir": self.m12_run_dir, "m13_run_dir": self.m13_run_dir,
                        "m14_run_dir": self.m14_run_dir, "experiment_name": "m15_view_for_m17", "seed": self.seed})


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def resolve(cfg: Mapping[str, Any] | None = None) -> M17Config:
    cfg = dict(cfg or {})
    known = {f.name for f in fields(M17Config)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise KeyError(f"unknown CFG key(s) {unknown}; allowed: {sorted(known)}")
    base = M17Config()
    vals: dict[str, Any] = {}
    for k, v in cfg.items():
        d = getattr(base, k)
        if isinstance(d, bool) and not isinstance(v, bool):
            raise TypeError(f"CFG[{k!r}] must be True/False, got {v!r}")
        vals[k] = tuple(v) if (isinstance(d, tuple) and isinstance(v, list)) else v
    c = replace(base, **vals)
    if c.precision != "fp32":
        raise ValueError("M17 runs in FP32 only")
    if c.experiment_name.lower().startswith(READ_ONLY_PREFIXES):
        raise ValueError(f"experiment_name {c.experiment_name!r} uses a read-only milestone prefix: choose an m17_* name")
    for ro in c.read_only_dirs():
        if _inside(c.run_dir, ro) or _inside(ro, c.run_dir):
            raise ValueError(f"run directory {c.run_dir} overlaps the read-only directory {ro}")
    if c.run_training and not c.allow_new_training:
        raise ValueError("run_training=True needs allow_new_training=True as a second explicit opt-in")
    if list(c.entry_speed_bins) != sorted(c.entry_speed_bins) or list(c.turn_bins_deg) != sorted(c.turn_bins_deg):
        raise ValueError("stratum boundaries must be increasing (fixed a priori, never tuned on data)")
    if not 0 < c.icc_lo < c.icc_hi < 1:
        raise ValueError("need 0 < icc_lo < icc_hi < 1")
    return c


class M17Run(M13Run):
    def __init__(self, root: Path | str, cfg: M17Config | None = None, local_tmp: Path | str | None = None):
        root = Path(root)
        if cfg is not None:
            for ro in cfg.read_only_dirs():
                if _inside(root, ro):
                    raise ValueError(f"refusing to open an M17 run inside the read-only directory {ro}")
        for d in M17_LAYOUT:
            (root / d).mkdir(parents=True, exist_ok=True)
        ResearchRun.__init__(self, root, cfg, local_tmp)
        self.results: dict[str, Any] = {}
        self.provenance: dict[str, Any] = {"transform_version": TRANSFORM_VERSION, "config_fingerprint": cfg.fingerprint() if cfg else None}
        self.ledger = StatusLedger.open(root, "m17")

    def table(self, df: Any, name: str, family: str, note: str = "", provenance: dict[str, Any] | None = None) -> list[Path]:
        if family not in M17_FAMILIES:
            raise ValueError(f"unknown table family {family!r}; one of {M17_FAMILIES}")
        if df is None or not getattr(df, "width", 0):
            return []
        out = self.save_table(df, name, csv=None, note=note, subdir=f"tables/{family}")
        if out:
            append_jsonl({"table": f"{family}/{safe_name(name)}", "rows": int(getattr(df, "height", 0)), "written_at": utc_now(), **self.provenance,
                          **(provenance or {})}, self.root / "manifests" / "table_provenance.jsonl")
        return out

    def snapshot_config(self, cfg: M17Config, environment: dict[str, Any] | None = None) -> list[Path]:
        out = self.write_config(cfg, environment or {})
        out.append(atomic_json({"fingerprint": cfg.fingerprint(), "config": cfg.to_dict(), "code_version": code_version(), "written_at": utc_now()},
                               self.root / "config" / f"config_{cfg.fingerprint()}.json", overwrite=False))
        return out

    def write_status(self, extra: dict[str, Any] | None = None) -> Path:
        return atomic_json({"updated_at": utc_now(), "transform_version": TRANSFORM_VERSION, "steps": self.ledger.records, **(extra or {})},
                           self.root / "manifests" / "m17_run_status.json", overwrite=True)

    def handoff_markdown(self) -> str:
        lines = [f"# M17 run hand-off — {self.root.name}", "", f"Updated {utc_now()}", "", "| step | status | reason |", "|---|---|---|"]
        lines += [f"| {k} | {v.get('status')} | {str(v.get('reason', '')).replace('|', '/')[:160]} |" for k, v in sorted(self.ledger.records.items())]
        return "\n".join(lines)


__all__ = ["M17Config", "M17Run", "M17_FAMILIES", "M17_LAYOUT", "TRANSFORM_VERSION", "resolve"]
