"""Configuration and run directory of Milestone 16 — verified baselines & cross-domain validation.

Same pattern as M13–M15: ``resolve(CFG)`` → frozen ``M16Config``; ``M16Config.m13()`` → ``M13Config`` for the reused
hand-off helpers (``handoff.environment`` / ``handoff.open_experiment``). M16 NEVER trains an encoder (no training keys).

The M1 verification (task T04) recomputes the official ladder M1 in a NEW experiment under the M16 run directory and
compares it with (a) the baseline-engine refit of the same session and (b) every persisted official OOF record found in
the read-only M12 / M13 / M14 experiment directories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from ...artifacts.store import append_jsonl, atomic_json
from ...config import stable_hash
from ...data.validation import TRANSFER_TARGET
from ...research.artifacts import ResearchRun, safe_name
from ...utils.reproducibility import code_version, utc_now
from ..m13_config import EXPECTED, M13Run
from ..status import StatusLedger

TRANSFORM_VERSION = "m16_v1"
READ_ONLY_PREFIXES = ("m12_", "m13_", "m14_", "m15_")
M16_LAYOUT = ("config", "manifests", "logs", "datasets/cached", "features/targets", "predictions", "checkpoints", "tables/cohort", "tables/baselines",
              "tables/target", "tables/registry", "tables/retrieval", "tables/detectability", "tables/synthesis", "figures/png", "figures/svg",
              "figures/pdf", "figures/data", "reports/markdown", "reports/html", "diagnostics", "exports")
M16_FAMILIES = ("cohort", "baselines", "target", "registry", "retrieval", "detectability", "synthesis")
IO_KEYS = frozenset({"experiment_name", "bundle_dir", "drive_root", "stage_dir", "m12_run_dir", "m13_run_dir", "m14_run_dir", "m15_run_dir",
                     "extra_oof_sources", "exploratory_target_path", "device", "resume", "reuse_artifacts", "save_tables", "save_figures",
                     "save_processed_data", "save_reports", "save_models", "overwrite_figures", "csv_max_rows", "figure_formats", "figure_dpi", "git_ref",
                     "notes"})


@dataclass(frozen=True)
class M16Config:
    experiment_name: str = "m16_verified_baselines_v1"
    bundle_dir: str = "/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean"
    drive_root: str = "/content/drive/MyDrive/NFLBowl"
    stage_dir: str = "/content/nflbowl_stage"
    m12_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m12_comprehensive_lab_v1"
    m12_experiment_name: str = "m12_comprehensive_lab_v1"
    m13_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m13_outcome_transfer_v1"
    m13_experiment_name: str = "m13_outcome_transfer_v1"
    m14_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m14_cross_sport_athletic_intelligence_v1"
    m14_experiment_name: str = "m14_cross_sport_athletic_intelligence_v1"
    m15_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m15_spectral_temporal_v1"
    extra_oof_sources: tuple[str, ...] = ()                 # further experiment directories holding evaluation/records/transfer_oof__*.parquet
    device: str = "auto"
    precision: str = "fp32"
    seed: int = 42
    base_config: str = "colab_gpu"
    expected_fingerprint: str = EXPECTED["fingerprint"]
    expected_science_hash: str = EXPECTED["science_hash"]
    # ---- T04 — M1 verification
    official_target: str = TRANSFER_TARGET                  # "nfl.q90_shared" (12C frozen) — the only target of the official ladder
    exploratory_target: str = "nfl_speed_q90_median"        # exploratory notebook target: audited separately, never compared with the official one
    exploratory_target_path: str = ""                       # parquet / csv with nfl_id + exploratory_target (optional)
    reproduction_tol: float = 1e-8                          # max |Δ prediction| (and |Δ y|) for "reproduced"
    require_persisted_match: bool = True                    # FAILED_VALIDATION when a compatible persisted record disagrees
    min_athletes_per_fold: int = 20
    plausible_ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "height": (64.0, 84.0), "weight": (150.0, 400.0), "forty": (4.0, 6.2), "ten": (1.3, 2.2), "draft_year": (2000.0, 2030.0)})
    n_boot: int = 2000
    # ---- T06 targets / reliability / detectability
    target_min_windows: int = 20
    target_min_windows_kind: int = 5
    target_min_events: int = 5
    reliability_split: str = "season"                       # season → game → random (labelled)
    detect_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)
    detect_reliability: float = 0.7
    detect_reps: int = 50
    # ---- T09 baselines
    learners: tuple[str, ...] = ("enet",)                   # + "lightgbm", "catboost" (optional packages)
    min_coverage: float = 0.5
    task_targets_in_ladder: str = "eligible"                # "eligible" (G16-A PASS) or "all"
    # ---- T10 retrieval
    allow_descriptive: bool = False                         # also test GO_DESCRIPTIVE representations (labelled)
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

    def experiment_dirs(self) -> list[Path]:
        """Read-only experiment directories searched for persisted official OOF records (existing ones only)."""
        cands = [Path(self.m12_run_dir) / "checkpoints" / "experiments" / self.m12_experiment_name,
                 Path(self.m13_run_dir) / "checkpoints" / "experiments" / self.m13_experiment_name,
                 Path(self.m14_run_dir) / "checkpoints" / "experiments" / self.m14_experiment_name, *[Path(p) for p in self.extra_oof_sources]]
        return [p for p in cands if (p / "config" / "experiment.json").is_file()]

    def read_only_dirs(self) -> list[Path]:
        return [Path(p) for p in (self.m12_run_dir, self.m13_run_dir, self.m14_run_dir, self.m15_run_dir, *self.extra_oof_sources) if p]

    def fingerprint(self) -> str:
        return stable_hash({"transform_version": TRANSFORM_VERSION, **{k: v for k, v in asdict(self).items() if k not in IO_KEYS}}, 16)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def m13(self) -> Any:
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


def resolve(cfg: Mapping[str, Any] | None = None) -> M16Config:
    cfg = dict(cfg or {})
    for forbidden in ("run_training", "allow_new_training"):
        if forbidden in cfg:
            raise KeyError(f"CFG[{forbidden!r}]: M16 never trains encoders (representations come from M15 / M12, read only)")
    known = {f.name for f in fields(M16Config)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise KeyError(f"unknown CFG key(s) {unknown}; allowed: {sorted(known)}")
    base = M16Config()
    vals: dict[str, Any] = {}
    for k, v in cfg.items():
        d = getattr(base, k)
        if isinstance(d, bool) and not isinstance(v, bool):
            raise TypeError(f"CFG[{k!r}] must be True/False, got {v!r}")
        vals[k] = tuple(v) if (isinstance(d, tuple) and isinstance(v, list)) else v
    c = replace(base, **vals)
    if c.precision != "fp32":
        raise ValueError("M16 runs in FP32 only")
    if c.experiment_name.lower().startswith(READ_ONLY_PREFIXES):
        raise ValueError(f"experiment_name {c.experiment_name!r} uses a read-only milestone prefix {READ_ONLY_PREFIXES}: choose an m16_* name")
    for ro in c.read_only_dirs():
        if _inside(c.run_dir, ro) or _inside(ro, c.run_dir):
            raise ValueError(f"run directory {c.run_dir} overlaps the read-only directory {ro}")
    if c.official_target != TRANSFER_TARGET:
        raise ValueError(f"official_target must stay {TRANSFER_TARGET!r} (the frozen ladder target); other targets get their own IDs in m16.targets")
    if c.exploratory_target == c.official_target:
        raise ValueError("the exploratory target must be distinct from the official target")
    if c.reliability_split not in ("season", "game", "random"):
        raise ValueError("reliability_split must be 'season', 'game' or 'random'")
    if c.task_targets_in_ladder not in ("eligible", "all"):
        raise ValueError("task_targets_in_ladder must be 'eligible' or 'all'")
    bad_l = [x for x in c.learners if x not in ("enet", "lightgbm", "catboost")]
    if bad_l:
        raise ValueError(f"unknown learner(s) {bad_l}; allowed enet, lightgbm, catboost")
    if not 0.0 < c.detect_reliability <= 1.0 or c.detect_reps < 5:
        raise ValueError("detect_reliability in (0, 1] and detect_reps ≥ 5")
    if not 0.0 < c.reproduction_tol <= 1e-4:
        raise ValueError("reproduction_tol must lie in (0, 1e-4]: a looser tolerance is not a reproduction (record a mismatch instead)")
    return c


class M16Run(M13Run):
    """RUN_DIR tree of M16 (``M16_LAYOUT``), tables per family with provenance, status ledger ``manifests/m16_status.json``."""

    def __init__(self, root: Path | str, cfg: M16Config | None = None, local_tmp: Path | str | None = None):
        root = Path(root)
        if cfg is not None:
            for ro in cfg.read_only_dirs():
                if _inside(root, ro):
                    raise ValueError(f"refusing to open an M16 run inside the read-only directory {ro}")
        for d in M16_LAYOUT:
            (root / d).mkdir(parents=True, exist_ok=True)
        ResearchRun.__init__(self, root, cfg, local_tmp)
        self.results: dict[str, Any] = {}
        self.provenance: dict[str, Any] = {"transform_version": TRANSFORM_VERSION, "config_fingerprint": cfg.fingerprint() if cfg else None}
        self.ledger = StatusLedger.open(root, "m16")

    def guard(self, path: Path | str) -> Path:
        p = Path(path)
        if not _inside(p, self.root):
            raise PermissionError(f"M16 writes only under {self.root}; refused {p}")
        return p

    def table(self, df: Any, name: str, family: str, note: str = "", provenance: dict[str, Any] | None = None) -> list[Path]:
        if family not in M16_FAMILIES:
            raise ValueError(f"unknown table family {family!r}; one of {M16_FAMILIES}")
        if df is None or not getattr(df, "width", 0):
            return []
        out = self.save_table(df, name, csv=None, note=note, subdir=f"tables/{family}")
        if out:
            append_jsonl({"table": f"{family}/{safe_name(name)}", "rows": int(getattr(df, "height", 0)), "written_at": utc_now(), **self.provenance,
                          **(provenance or {})}, self.root / "manifests" / "table_provenance.jsonl")
        return out

    def snapshot_config(self, cfg: M16Config, environment: dict[str, Any] | None = None) -> list[Path]:
        out = self.write_config(cfg, environment or {})
        out.append(atomic_json({"fingerprint": cfg.fingerprint(), "transform_version": TRANSFORM_VERSION, "config": cfg.to_dict(),
                                "code_version": code_version(), "written_at": utc_now()}, self.root / "config" / f"config_{cfg.fingerprint()}.json",
                               overwrite=False))
        return out

    def write_status(self, extra: dict[str, Any] | None = None) -> Path:
        rec = {"updated_at": utc_now(), "transform_version": TRANSFORM_VERSION, "config_fingerprint": self.provenance.get("config_fingerprint"),
               "stages": {k: {kk: v.get(kk) for kk in ("status", "note", "seconds", "cache", "finished_at")} for k, v in self.stage_records().items()},
               "steps": self.ledger.records, **(extra or {})}
        return atomic_json(rec, self.root / "manifests" / "m16_run_status.json", overwrite=True)

    def handoff_markdown(self) -> str:
        lines = [f"# M16 run hand-off — {self.root.name}", "", f"Updated {utc_now()} · config fingerprint `{self.provenance.get('config_fingerprint')}`", "",
                 "| step | status | reason |", "|---|---|---|"]
        for k, v in sorted(self.ledger.records.items()):
            lines.append(f"| {k} | {v.get('status')} | {str(v.get('reason', '')).replace('|', '/')[:160]} |")
        return "\n".join(lines)


__all__ = ["M16Config", "M16Run", "M16_FAMILIES", "M16_LAYOUT", "TRANSFORM_VERSION", "resolve"]
