"""Configuration and output store of Milestone 13 — NFL outcome transfer, reliability-aware biomechanics and scouting value
(notebook ``M13_NFL_OUTCOME_TRANSFER_RESEARCH_COLAB.ipynb``).

``CFG`` in the notebook is a plain dict; ``resolve(CFG)`` validates it (unknown keys raise) and returns a frozen
``M13Config``. ``M13Config.lab()`` maps the shared keys onto an M12 ``LabConfig`` so the M12 environment helpers
(bundle validation, experiment opening) are reused unchanged.

Execution policy encoded here (never switched on implicitly):

* no neural training: ``run_training`` must stay False unless ``allow_new_training`` is ALSO True (two explicit opt-ins);
* M12 inductive checkpoints are adopted read-only (copied into the M13 experiment, sources never written);
* deterministic inference (embedding sequences with an adopted checkpoint) is allowed, because it is required when the
  M12 embedding files cannot be adopted; it is recorded per (model, fold);
* FP32 everywhere.

``M12_REPORTED`` holds the values PUBLISHED in the reviewed M12 notebook. They are used ONLY in the A3 reconciliation
table (observed vs reported); no analysis reads them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from ..artifacts.store import append_jsonl
from ..research.artifacts import ResearchRun, safe_name
from ..utils.reproducibility import utc_now
from .store import LabRun

TRANSFORM_VERSION = "m13_v1"
EXPECTED = {"fingerprint": "4867be08f8191b3a", "science_hash": "0281883accece2f0", "n_sequences": 77_125, "n_athletes": 510, "folds": 5}
ENCODERS = ("F2", "F3", "F4", "F5", "F6")
# Reported in the reviewed M12 notebook (M12_COMPREHENSIVE_EXPERIMENTAL_LAB_REVIEWED.ipynb). Reconciliation only.
M12_REPORTED = {
    "inductive_retrieval_auc": {"F0_engineered": 0.5024, "F2_cnn_physics": 0.5081, "F3_masked_ae": 0.5124, "F4_physics_contrastive": 0.5126,
                                "F5_jepa_physics": 0.5078, "F6_transformer": 0.4962},
    "F4_vs_F0_delta_auc": 0.0102,
    "F4_effective_rank": 13.26, "F4_domain_auc": 0.997, "F4_mmd2": 0.134,
    "n_sequences": 77_125, "n_athletes": 510, "n_athletes_both_domains": 399, "n_inductive_jobs": 25,
}
M13_LAYOUT = ("config", "manifests", "logs", "datasets/cached", "datasets/audits", "features/sequence", "features/athlete", "embeddings/inductive",
              "checkpoints", "tables/handoff", "tables/target", "tables/transfer", "tables/reliability", "tables/position", "tables/domain",
              "tables/retrieval_transfer", "tables/scouting", "tables/synthesis", "figures/png", "figures/svg", "figures/pdf", "figures/data",
              "reports/markdown", "reports/html", "reports/athlete_profiles", "diagnostics", "exports")
M13_FAMILIES = ("handoff", "target", "transfer", "reliability", "position", "domain", "retrieval_transfer", "scouting", "synthesis")


@dataclass(frozen=True)
class M13Config:
    experiment_name: str = "m13_outcome_transfer_v1"
    bundle_dir: str = "/content/drive/MyDrive/nfl-big-data-bowl-2027/bundles/m12f_v1_clean"
    drive_root: str = "/content/drive/MyDrive/NFLBowl"
    m12_run_dir: str = "/content/drive/MyDrive/NFLBowl/experimental_lab/m12_comprehensive_lab_v1"
    m12_experiment_name: str = "m12_comprehensive_lab_v1"
    extra_source_experiments: tuple[str, ...] = ()       # further experiment dirs to search for compatible checkpoints (absolute)
    stage_dir: str = "/content/nflbowl_stage"
    device: str = "auto"
    precision: str = "fp32"
    seed: int = 42                                       # M13 analysis seed (bootstrap, sampling); the science seed is the preset's
    base_config: str = "colab_gpu"
    expected_fingerprint: str = EXPECTED["fingerprint"]
    expected_science_hash: str = EXPECTED["science_hash"]
    encoders: tuple[str, ...] = ENCODERS
    include_f1: bool = True                              # M1 + F1 TCN only when its fold validity is verified (C0)
    resume: bool = True
    reuse_artifacts: bool = True
    adopt_embeddings: bool = True                        # copy M12 embedding files whose weights checksum matches the adopted checkpoint
    allow_inference: bool = True                         # embed sequences with an adopted checkpoint when no adoptable file exists
    verify_checkpoint_load: bool = True                  # A2: torch.load every adopted best.pt on CPU (strict state-dict match)
    run_training: bool = False
    allow_new_training: bool = False                     # second opt-in required for ANY neural training
    run_transfer: bool = True
    run_reliability: bool = True
    run_position_models: bool = True
    run_domain: bool = True
    run_scouting: bool = True
    run_interactions: bool = True
    n_boot: int = 2000
    reliability_repeats: int = 20
    reliability_k_grid: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    reliability_threshold: float = 0.6                   # train-fold ICC(1,1) a feature must reach to enter M1 + reliability-filtered
    train_selected_k: int = 4                            # features kept by train-fold partial-association selection
    min_reliability_athletes: int = 30
    min_position_n: int = 10
    adequate_n: int = 30
    partial_pooling_kappas: tuple[float, ...] = (5.0, 15.0, 40.0, 120.0)
    domain_max_per_domain: int = 2000
    domain_bins: int = 6
    domain_folds: tuple[int, ...] | None = None          # None = all folds
    n_profiles: int = 12
    interval_level: float = 0.8
    anonymize_ids: bool = True
    figure_theme: str = "navy"                           # navy (dark research theme) or light (print)
    figure_formats: tuple[str, ...] = ("png", "svg", "pdf")
    figure_dpi: int = 300
    save_tables: bool = True
    save_figures: bool = True
    save_processed_data: bool = True
    save_reports: bool = True
    save_models: bool = True
    overwrite_figures: bool = True
    csv_max_rows: int = 10_000_000
    git_ref: str = "main"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.drive_root) / "experimental_lab" / self.experiment_name

    @property
    def m12_experiment_dir(self) -> Path:
        return Path(self.m12_run_dir) / "checkpoints" / "experiments" / self.m12_experiment_name

    def training_enabled(self) -> bool:
        return bool(self.run_training and self.allow_new_training)

    def lab(self) -> Any:
        """The M12 ``LabConfig`` equivalent (shared keys only) for the reused environment helpers."""
        from .config import resolve as lab_resolve

        return lab_resolve({"experiment_name": self.experiment_name, "bundle_dir": self.bundle_dir, "drive_root": self.drive_root, "stage_dir": self.stage_dir,
                            "device": self.device, "precision": self.precision, "seed": self.seed, "base_config": self.base_config,
                            "expected_fingerprint": self.expected_fingerprint, "run_mode": "full", "run_training": False, "import_checkpoints": True,
                            "reference_experiments": [self.m12_experiment_name], "search_previous_runs": False})

    def experiment_overrides(self) -> dict[str, Any]:
        return {"runtime.precision": self.precision, "runtime.device": self.device, "runtime.stage_dir": self.stage_dir}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve(cfg: Mapping[str, Any] | None = None) -> M13Config:
    cfg = dict(cfg or {})
    known = set(M13Config.__dataclass_fields__)
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise KeyError(f"unknown CFG key(s) {unknown}; allowed: {sorted(known)}")
    base = M13Config()
    vals: dict[str, Any] = {}
    for k, v in cfg.items():
        d = getattr(base, k)
        if isinstance(d, tuple) and isinstance(v, list):
            v = tuple(v)
        if isinstance(d, bool) and not isinstance(v, bool):
            raise TypeError(f"CFG[{k!r}] must be True/False, got {v!r}")
        vals[k] = v
    c = replace(base, **vals)
    if c.precision != "fp32":
        raise ValueError("M13 runs in FP32 only (the reused M12 encoders were evaluated in FP32)")
    if c.figure_theme not in ("navy", "light"):
        raise ValueError("figure_theme must be 'navy' or 'light'")
    if c.run_training and not c.allow_new_training:
        raise ValueError("run_training=True needs allow_new_training=True as a second explicit opt-in (M13 reuses the M12 encoders)")
    if not 0.0 < c.reliability_threshold < 1.0:
        raise ValueError("reliability_threshold must lie in (0, 1)")
    if not 0.5 <= c.interval_level < 1.0:
        raise ValueError("interval_level must lie in [0.5, 1)")
    bad = [m for m in c.encoders if m not in ENCODERS]
    if bad:
        raise ValueError(f"encoders must be among {ENCODERS}; got {bad}")
    return c


class M13Run(LabRun):
    """RUN_DIR tree of M13 (``M13_LAYOUT``), tables per M13 family, provenance on every table and data file."""

    def __init__(self, root: Path | str, cfg: Any = None, local_tmp: Path | str | None = None):
        root = Path(root)
        for d in M13_LAYOUT:
            (root / d).mkdir(parents=True, exist_ok=True)
        ResearchRun.__init__(self, root, cfg, local_tmp)
        self.results: dict[str, Any] = {}
        self.provenance: dict[str, Any] = {"transform_version": TRANSFORM_VERSION}

    def set_provenance(self, **kw: Any) -> None:
        self.provenance.update({k: v for k, v in kw.items() if v is not None})
        p = self.root / "manifests" / "provenance.json"
        p.write_text(json.dumps({**self.provenance, "updated_at": utc_now()}, indent=2, default=str), encoding="utf-8")

    def table(self, df: Any, name: str, family: str, note: str = "", provenance: dict[str, Any] | None = None) -> list[Path]:
        if family not in M13_FAMILIES:
            raise ValueError(f"unknown table family {family!r}; one of {M13_FAMILIES}")
        out = self.save_table(df, name, csv=None, note=note, subdir=f"tables/{family}")
        if out:
            append_jsonl({"table": f"{family}/{safe_name(name)}", "rows": int(getattr(df, "height", 0)), "written_at": utc_now(), **self.provenance,
                          **(provenance or {})}, self.root / "manifests" / "table_provenance.jsonl")
        return out

    def data_file(self, df: Any, rel: str, provenance: dict[str, Any] | None = None) -> Path | None:
        return super().data_file(df, rel, {**self.provenance, **(provenance or {})})

    # ------------------------------------------------------------------ signature-checked cache
    def _sig_path(self) -> Path:
        return self.root / "manifests" / "stage_signatures.json"

    def _sigs(self) -> dict[str, Any]:
        p = self._sig_path()
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def set_signature(self, key: str, signature: Any) -> None:
        s = self._sigs()
        s[key] = {"signature": signature, "at": utc_now()}
        self._sig_path().write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")

    def cached(self, family: str, names: tuple[str, ...] | list[str], signature: Any = None, key: str | None = None) -> dict[str, Any] | None:
        """Every named table of a family reloaded from disk — only when reuse is on, every table exists and (when given)
        the stored input signature of ``key`` equals ``signature``. None → recompute."""
        if not (self.cfg is None or getattr(self.cfg, "reuse_artifacts", True)):
            return None
        if not all(self.has(n, family) for n in names):
            return None
        if signature is not None:
            have = self._sigs().get(key or f"{family}/{names[0]}", {}).get("signature")
            if json.dumps(have, default=str, sort_keys=True) != json.dumps(signature, default=str, sort_keys=True):
                return None
        return {n: self.frame(n, family) for n in names}

    def save_all(self, tables: dict[str, Any], family: str, signature: Any = None, key: str | None = None) -> None:
        """Save several tables of one family and record the input signature that produced them."""
        for n, df in tables.items():
            if df is not None and getattr(df, "width", 0):          # zero-column frames are not written (→ recomputed next session)
                self.table(df, n, family)
        if signature is not None and tables:
            self.set_signature(key or f"{family}/{next(iter(tables))}", signature)


__all__ = ["ENCODERS", "EXPECTED", "M12_REPORTED", "M13Config", "M13Run", "M13_FAMILIES", "M13_LAYOUT", "TRANSFORM_VERSION", "resolve"]
