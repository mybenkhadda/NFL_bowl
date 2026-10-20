"""Configuration and run directory of Milestone 18 — NFL performance transfer and scouting value.

``resolve`` → frozen ``M18Config``; ``M18Run`` (read-only guards for M12–M17, status ledger, immutable pre-registration).
M18 never trains an encoder (no training keys): representations come from M15 / M17, read only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from ...artifacts.store import append_jsonl, atomic_json
from ...config import stable_hash
from ...research.artifacts import ResearchRun, safe_name
from ...utils.reproducibility import code_version, utc_now
from ..m13_config import EXPECTED, M13Run
from ..status import StatusLedger

TRANSFORM_VERSION = "m18_v1"
READ_ONLY_PREFIXES = ("m12_", "m13_", "m14_", "m15_", "m16_", "m17_")
RUNGS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")
M18_LAYOUT = ("config", "manifests", "logs", "datasets/cached", "predictions", "tables/ladder", "tables/position", "tables/specificity", "tables/scouting",
              "tables/robustness", "tables/controls", "tables/synthesis", "figures/png", "figures/svg", "figures/pdf", "figures/data", "figures/publication",
              "reports/markdown", "reports/html", "reports/athlete_profiles", "model_cards", "diagnostics", "exports")
M18_FAMILIES = ("ladder", "position", "specificity", "scouting", "robustness", "controls", "synthesis")
IO_KEYS = frozenset({"experiment_name", "bundle_dir", "drive_root", "stage_dir", "m12_run_dir", "m13_run_dir", "m14_run_dir", "m15_run_dir", "m16_run_dir",
                     "m17_run_dir", "device", "resume", "reuse_artifacts", "save_tables", "save_figures", "save_processed_data", "save_reports", "save_models",
                     "overwrite_figures", "csv_max_rows", "figure_formats", "figure_dpi", "git_ref", "notes", "pseudonym_salt"})


@dataclass(frozen=True)
class M18Config:
    experiment_name: str = "m18_scouting_transfer_v1"
    bundle_dir: str = "/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean"
    drive_root: str = "/content/drive/MyDrive/NFLBowl"
    stage_dir: str = "/content/nflbowl_stage"
    m12_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m12_comprehensive_lab_v1"
    m12_experiment_name: str = "m12_comprehensive_lab_v1"
    m13_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m13_outcome_transfer_v1"
    m14_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m14_cross_sport_athletic_intelligence_v1"
    m15_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m15_spectral_temporal_v1"
    m16_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m16_verified_baselines_v1"
    m17_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m17_context_athletic_signatures_v1"
    device: str = "auto"
    precision: str = "fp32"
    seed: int = 42
    base_config: str = "colab_gpu"
    expected_fingerprint: str = EXPECTED["fingerprint"]
    expected_science_hash: str = EXPECTED["science_hash"]
    primary_target: str = "T0"                               # changed only by a recorded G16-A decision (pre-registration)
    secondary_targets: str = "eligible"                      # "eligible" (M16 freeze) or "none"
    embedding_kinds: tuple[str, ...] = ("forty",)            # Combine drills averaged into the athlete embedding (official ladder: forty)
    embedding_dims: int = 8                                  # train-fold PCA dims of an embedding block (official ladder convention)
    allow_descriptive: bool = False                          # admit GO_DESCRIPTIVE representations to B4–B6 (labelled exploratory)
    learners: tuple[str, ...] = ("enet",)
    n_boot: int = 2000
    n_perm: int = 200
    margin_pct: float = 2.0
    min_position_n: int = 30
    twin_caliper: float = 0.75
    twin_covariates: tuple[str, ...] = ("forty", "ten", "weight", "height")
    n_case_studies: int = 6
    pseudonym_salt: str = "m18"
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
        return [Path(p) for p in (self.m12_run_dir, self.m13_run_dir, self.m14_run_dir, self.m15_run_dir, self.m16_run_dir, self.m17_run_dir) if p]

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


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def resolve(cfg: Mapping[str, Any] | None = None) -> M18Config:
    cfg = dict(cfg or {})
    for forbidden in ("run_training", "allow_new_training"):
        if forbidden in cfg:
            raise KeyError(f"CFG[{forbidden!r}]: M18 never trains encoders (representations are read from M15 / M17)")
    known = {f.name for f in fields(M18Config)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise KeyError(f"unknown CFG key(s) {unknown}; allowed: {sorted(known)}")
    base = M18Config()
    vals: dict[str, Any] = {}
    for k, v in cfg.items():
        d = getattr(base, k)
        if isinstance(d, bool) and not isinstance(v, bool):
            raise TypeError(f"CFG[{k!r}] must be True/False, got {v!r}")
        vals[k] = tuple(v) if (isinstance(d, tuple) and isinstance(v, list)) else v
    c = replace(base, **vals)
    if c.precision != "fp32":
        raise ValueError("M18 runs in FP32 only")
    if c.experiment_name.lower().startswith(READ_ONLY_PREFIXES):
        raise ValueError(f"experiment_name {c.experiment_name!r} uses a read-only milestone prefix: choose an m18_* name")
    for ro in c.read_only_dirs():
        if _inside(c.run_dir, ro) or _inside(ro, c.run_dir):
            raise ValueError(f"run directory {c.run_dir} overlaps the read-only directory {ro}")
    if c.secondary_targets not in ("eligible", "none"):
        raise ValueError("secondary_targets must be 'eligible' or 'none'")
    if c.n_perm < 20 or c.n_boot < 200 or c.margin_pct <= 0:
        raise ValueError("n_perm ≥ 20, n_boot ≥ 200, margin_pct > 0")
    return c


class M18Run(M13Run):
    def __init__(self, root: Path | str, cfg: M18Config | None = None, local_tmp: Path | str | None = None):
        root = Path(root)
        if cfg is not None:
            for ro in cfg.read_only_dirs():
                if _inside(root, ro):
                    raise ValueError(f"refusing to open an M18 run inside the read-only directory {ro}")
        for d in M18_LAYOUT:
            (root / d).mkdir(parents=True, exist_ok=True)
        ResearchRun.__init__(self, root, cfg, local_tmp)
        self.results: dict[str, Any] = {}
        self.provenance: dict[str, Any] = {"transform_version": TRANSFORM_VERSION, "config_fingerprint": cfg.fingerprint() if cfg else None}
        self.ledger = StatusLedger.open(root, "m18")

    def table(self, df: Any, name: str, family: str, note: str = "", provenance: dict[str, Any] | None = None) -> list[Path]:
        if family not in M18_FAMILIES:
            raise ValueError(f"unknown table family {family!r}; one of {M18_FAMILIES}")
        if df is None or not getattr(df, "width", 0):
            return []
        out = self.save_table(df, name, csv=None, note=note, subdir=f"tables/{family}")
        if out:
            append_jsonl({"table": f"{family}/{safe_name(name)}", "rows": int(getattr(df, "height", 0)), "written_at": utc_now(), **self.provenance,
                          **(provenance or {})}, self.root / "manifests" / "table_provenance.jsonl")
        return out

    def snapshot_config(self, cfg: M18Config, environment: dict[str, Any] | None = None) -> list[Path]:
        out = self.write_config(cfg, environment or {})
        out.append(atomic_json({"fingerprint": cfg.fingerprint(), "config": cfg.to_dict(), "code_version": code_version(), "written_at": utc_now()},
                               self.root / "config" / f"config_{cfg.fingerprint()}.json", overwrite=False))
        return out

    def preregister(self, record: dict[str, Any]) -> dict[str, Any]:
        """``manifests/preregistration.json`` written ONCE, before any outcome-bearing step. A later call with a different
        primary family does not overwrite it and is reported as ``conflict`` (analyses then label themselves post hoc)."""
        p = self.root / "manifests" / "preregistration.json"
        rec = {**record, "written_at": utc_now()}
        if p.is_file():
            old = json.loads(p.read_text(encoding="utf-8"))
            same = {k: old.get(k) for k in ("primary_target", "primary_family")} == {k: rec.get(k) for k in ("primary_target", "primary_family")}
            return {**old, "conflict": not same, "requested": None if same else rec}
        atomic_json(rec, p, overwrite=False)
        return {**rec, "conflict": False}

    def write_status(self, extra: dict[str, Any] | None = None) -> Path:
        return atomic_json({"updated_at": utc_now(), "transform_version": TRANSFORM_VERSION, "steps": self.ledger.records, **(extra or {})},
                           self.root / "manifests" / "m18_run_status.json", overwrite=True)

    def handoff_markdown(self) -> str:
        lines = [f"# M18 run hand-off — {self.root.name}", "", f"Updated {utc_now()}", "", "| step | status | reason |", "|---|---|---|"]
        lines += [f"| {k} | {v.get('status')} | {str(v.get('reason', '')).replace('|', '/')[:160]} |" for k, v in sorted(self.ledger.records.items())]
        return "\n".join(lines)


__all__ = ["M18Config", "M18Run", "M18_FAMILIES", "M18_LAYOUT", "RUNGS", "TRANSFORM_VERSION", "resolve"]
