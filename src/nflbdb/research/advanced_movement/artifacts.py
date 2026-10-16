"""Shared artifact store of the advanced-movement notebooks (tables, datasets, figures, model directory).

    reports/tables/advanced_movement/<name>.csv            small tables
    warehouse/gold/advanced_movement/<name>.parquet        every output, typed (+ <name>.schema.json: grain, key,
                                                           columns, provenance, reload snippet)
    warehouse/gold/advanced_movement/_manifest.json        fingerprints of the cached builds
    reports/figures/advanced_movement/<name>.png|svg|pdf   figures (+ <name>.meta.json; plot data under tables/)
    artifacts/models/advanced_movement/                    checkpoints and fitted preprocessing (from notebook 02)

Smoke mode nests ALL of the above under ``…/advanced_movement/_sample`` so a smoke run can never replace or be
mistaken for a full run: a full-mode notebook reads the full paths only and fails with an actionable message when
a prerequisite is absent.

Built on ``sprint_rescue.store.RescueStore`` (contracts, atomic writes, sidecars, fingerprints) — no second
persistence implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import polars as pl

from ...data.utils import atomic_write_json, file_fingerprint, get_logger, read_json, utc_now
from ..contracts import TableContract, conform
from ..sprint_rescue.store import RescueStore
from .config import NAMESPACE, VERSION, NotebookControls

log = get_logger(__name__)


class MissingArtifactError(FileNotFoundError):
    """A prerequisite artifact of an earlier notebook is absent for the current run mode."""


class AdvStore(RescueStore):
    def __init__(self, cfg: Any, controls: NotebookControls):
        controls.validate()
        super().__init__(cfg, sampled=controls.smoke, namespace=NAMESPACE)
        self.controls = controls
        self.models_dir = cfg.paths.project_root / "artifacts" / "models" / NAMESPACE / ("_sample" if controls.smoke else "")

    # ------------------------------------------------------------------ description
    def describe(self) -> pl.DataFrame:
        rel = self.cfg.paths.rel
        return pl.DataFrame([{"what": "run mode", "where": self.controls.run_mode},
                             {"what": "tables (CSV)", "where": rel(self.tables_dir)},
                             {"what": "datasets (Parquet + schema sidecar)", "where": rel(self.data_dir)},
                             {"what": "figures (PNG 300 DPI, SVG, PDF)", "where": rel(self.figures_dir)},
                             {"what": "models and fitted preprocessing", "where": rel(self.models_dir)}])

    # ------------------------------------------------------------------ prerequisites
    def require(self, c: TableContract, notebook: str) -> pl.DataFrame:
        """Read an artifact written by an earlier notebook IN THE SAME RUN MODE, or explain how to produce it."""
        if not self.exists(c):
            raise MissingArtifactError(
                f"`{c.name}` is missing for RUN_MODE = {self.controls.run_mode!r} "
                f"({self.cfg.paths.rel(self.parquet(c))}). Run notebook `{notebook}` with the same RUN_MODE and "
                "SAVE_TABLES = True first. Smoke and full artifacts are kept apart on purpose and are never substituted.")
        return self.read(c)

    def require_named(self, name: str, notebook: str) -> pl.DataFrame:
        """Read a persisted dataset by NAME (no contract import — e.g. a table whose module needs PyTorch), in the
        same run mode, or explain how to produce it. The frame is returned as stored."""
        p = self.data_dir / f"{name}.parquet"
        if not p.is_file():
            raise MissingArtifactError(
                f"`{name}` is missing for RUN_MODE = {self.controls.run_mode!r} ({self.cfg.paths.rel(p)}). Run notebook "
                f"`{notebook}` with the same RUN_MODE and SAVE_TABLES = True first.")
        return pl.read_parquet(p)

    # ------------------------------------------------------------------ cached builds
    def build(self, stage: str, contracts: Sequence[TableContract], builder: Callable[[], dict[str, pl.DataFrame]],
              params: dict[str, Any] | None = None, files: Iterable[Path] = ()) -> dict[str, pl.DataFrame]:
        """Run ``builder`` and persist its outputs, or restore them when parameters and input files are unchanged.

        ``builder`` returns ``{contract name: frame}`` for every contract. With FORCE_REBUILD the cache is ignored.
        With SAVE_TABLES = False the outputs are validated and returned but not written."""
        upstream = {self.cfg.paths.rel(f): (file_fingerprint(f) if Path(f).is_file() else "missing") for f in files}
        fp = self.fingerprint(stage, {"params": params or {}, "version": VERSION, "run_mode": self.controls.run_mode},
                              upstream)
        current, reason = self.is_current(stage, fp, contracts)
        if current and not self.controls.force_rebuild:
            log.info("[%s:%s] restored from cache (%s)", NAMESPACE, stage, reason)
            return {c.name: self.read(c) for c in contracts}
        log.info("[%s:%s] computing (%s)", NAMESPACE, stage, "FORCE_REBUILD" if current else reason)
        out = builder()
        absent = [c.name for c in contracts if c.name not in out]
        if absent:
            raise KeyError(f"builder of stage {stage!r} did not return {absent}")
        if not self.controls.save_tables:
            log.warning("[%s:%s] SAVE_TABLES is False: outputs are not persisted; later notebooks will not find them",
                        NAMESPACE, stage)
            return {c.name: conform(out[c.name], c) for c in contracts}
        provenance = {"inputs": list(upstream), "params": params or {}, "run_mode": self.controls.run_mode,
                      "experiment_version": VERSION, "seed": self.controls.seed}
        written = {c.name: self.write(out[c.name], c, stage, provenance) for c in contracts}
        self.record(stage, fp, contracts)              # recorded only after EVERY output passed its contract
        return written

    # ------------------------------------------------------------------ figures
    def show(self, fig, name: str, plot_data: dict[str, pl.DataFrame] | None = None, publication: bool = False,
             source: str = "", n: str = "", note: str = "") -> None:
        """Display the figure inline, then save it (PNG 300 DPI + SVG, + PDF when ``publication``) with its plot
        data and a metadata sidecar. With SAVE_FIGURES = False it is displayed only."""
        try:
            from IPython.display import display

            display(fig)
        except ImportError:
            pass
        if not self.controls.save_figures:
            import matplotlib.pyplot as plt

            plt.close(fig)
            return
        self.save_figure(fig, name, plot_data or {}, publication=publication, source=source, n=n)
        atomic_write_json({"figure": name, "experiment": NAMESPACE, "experiment_version": VERSION,
                           "run_mode": self.controls.run_mode, "seed": self.controls.seed, "source": source, "n": n,
                           "note": note, "saved_at": utc_now(),
                           "formats": ["png", "svg", "pdf"] if publication else ["png", "svg"]},
                          self.figures_dir / f"{name}.meta.json")

    # ------------------------------------------------------------------ small metadata
    def write_json(self, name: str, obj: dict[str, Any]) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return atomic_write_json({**obj, "run_mode": self.controls.run_mode, "experiment_version": VERSION,
                                  "written_at": utc_now()}, self.data_dir / f"{name}.json")

    def read_json(self, name: str, notebook: str) -> dict[str, Any]:
        p = self.data_dir / f"{name}.json"
        if not p.is_file():
            raise MissingArtifactError(f"{self.cfg.paths.rel(p)} is missing for RUN_MODE = {self.controls.run_mode!r}. "
                                       f"Run notebook `{notebook}` first.")
        return read_json(p, {})


class AtlasStore(AdvStore):
    """Store of Milestone 04 (feature atlas, TCN robustness, frozen shortlist): a sub-area of the namespace.

        reports/tables/advanced_movement/atlas/        warehouse/gold/advanced_movement/atlas/ (+ its own _manifest.json)
        reports/figures/advanced_movement/atlas/       artifacts/models/advanced_movement/atlas/

    Smoke mode: ``…/advanced_movement/_sample/atlas``. Nothing written here can replace an artifact of notebooks
    00–03 (tables, figures, checkpoints and the run cache live in other directories). ``upstream`` is the plain
    ``AdvStore`` of the same run mode, used to READ what those notebooks wrote."""

    AREA = "atlas"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.upstream = AdvStore(cfg, controls)
        self.tables_dir, self.figures_dir, self.data_dir = (d / self.AREA for d in (self.tables_dir, self.figures_dir, self.data_dir))
        self.manifest_path = self.data_dir / "_manifest.json"
        self.models_dir = self.models_dir / self.AREA


class NflOutcomeStore(AtlasStore):
    """Store of Milestone 05 (NFL movement outcomes, game-grouped reliability, frozen targets): sub-area
    ``nfl_outcomes`` with its own manifest, so nothing of notebooks 00–04 can be replaced. Smoke mode:
    ``…/advanced_movement/_sample/nfl_outcomes``. ``upstream`` reads what the earlier notebooks wrote."""

    AREA = "nfl_outcomes"


class TransferStore(AtlasStore):
    """Store of Milestone 06 (Combine → NFL transfer validation): sub-area ``transfer`` with its own manifest and
    model directory, so nothing of notebooks 00–05 can be replaced. Smoke mode: ``…/advanced_movement/_sample/transfer``.
    ``upstream`` reads notebooks 00–03, ``atlas`` Milestone 04 and ``nfl`` Milestone 05, all in the same run mode."""

    AREA = "transfer"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.atlas = AtlasStore(cfg, controls)
        self.nfl = NflOutcomeStore(cfg, controls)


class ContextTransferStore(AtlasStore):
    """Store of Milestone 07A–07C (NFL context decomposition, physically aligned transfer, detectability): sub-area
    ``context_transfer`` with its own manifest, so nothing of notebooks 00–06 can be replaced. Smoke mode:
    ``…/advanced_movement/_sample/context_transfer``. ``upstream`` reads notebooks 00–03, ``atlas`` Milestone 04,
    ``nfl`` Milestone 05 and ``transfer`` Milestone 06 — all READ ONLY, all in the same run mode."""

    AREA = "context_transfer"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.atlas = AtlasStore(cfg, controls)
        self.nfl = NflOutcomeStore(cfg, controls)
        self.transfer = TransferStore(cfg, controls)
        self.checkpoints_dir = self.data_dir / "checkpoints"


class ScoringStore(ContextTransferStore):
    """Store of Milestone 07D (pre-draft athletic fingerprint scorecard): sub-area ``scoring``. ``context`` reads the
    07A–07C outputs (only for the clearly separated post-draft panel; never an input of a pre-draft score)."""

    AREA = "scoring"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.context = ContextTransferStore(cfg, controls)


class ScoutingStore(ScoringStore):
    """Store of Milestone 08 (scouting decision intelligence, notebooks 08A–08D): versioned sub-area ``scouting_v1``
    with its own manifest, so nothing of notebooks 00–07 can be replaced. Smoke mode:
    ``…/advanced_movement/_sample/scouting_v1``. ``scoring`` reads the 07D scorecard outputs — READ ONLY."""

    AREA = "scouting_v1"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.scoring = ScoringStore(cfg, controls)


class FootballStore(ScoutingStore):
    """Store of Milestone 08B (football-outcome registry, reliability, transfer, incremental information, scouting
    decision intelligence, evidence package; notebooks 08B.0–08B.6): versioned sub-area ``football_v1`` with its own
    manifest and checkpoint directory, so nothing of notebooks 00–08A.2 can be replaced. Smoke mode:
    ``…/advanced_movement/_sample/football_v1``. ``scouting`` reads the 08A / 08A.2 outputs — READ ONLY."""

    AREA = "football_v1"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.scouting = ScoutingStore(cfg, controls)


class IntelligenceStore(FootballStore):
    """Store of Milestone 09 (Combine Intelligence Lab; notebooks 09A–09G): versioned sub-area ``intelligence_v1`` with
    its own manifest, so nothing of notebooks 00–08B can be replaced. Smoke mode:
    ``…/advanced_movement/_sample/intelligence_v1``. ``football`` reads the 08B outputs — READ ONLY. ``mart_dir`` holds
    the versioned application data mart written by 09E (the only thing the interactive application reads)."""

    AREA = "intelligence_v1"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.football = FootballStore(cfg, controls)
        self.mart_dir = self.data_dir / "app_mart"
        self.reports_dir = cfg.paths.project_root / "reports" / "milestone09" / ("_sample" if controls.smoke else "")


class ProductStore(IntelligenceStore):
    """Store of Milestone 10 (final product and submission; notebooks 10A–10E): versioned sub-area ``product_v1`` with its
    own manifest, so nothing of Milestones 00–09 can be replaced. Smoke mode: ``…/advanced_movement/_sample/product_v1``.
    ``intelligence`` reads the Milestone 09 outputs and its application mart — READ ONLY. ``mart_dir`` points at the 09E
    mart (Milestone 10 adds no mart table). ``reports_dir`` = ``reports/milestone10``."""

    AREA = "product_v1"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.intelligence = IntelligenceStore(cfg, controls)
        self.mart_dir = self.intelligence.mart_dir
        self.reports_dir = cfg.paths.project_root / "reports" / "milestone10" / ("_sample" if controls.smoke else "")


class DiscoveryStore(ProductStore):
    """Store of Milestone 11 (physics of Combine → NFL movement transfer; notebooks 11.0, 11A–11I): versioned sub-area
    ``discovery_v1`` with its own manifest, checkpoint and model directories, so nothing of Milestones 00–10 can be
    replaced. Smoke mode: ``…/advanced_movement/_sample/discovery_v1``. ``product`` reads the Milestone 10 outputs,
    ``intelligence`` the Milestone 09 outputs (09A episodes, 09B curves, 09C twins) — READ ONLY.
    ``reports_dir`` = ``reports/milestone11``; ``shards_dir`` holds immutable checkpoint shards (``m11.checkpoint``)."""

    AREA = "discovery_v1"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.product = ProductStore(cfg, controls)
        self.mart_dir = self.intelligence.mart_dir
        self.reports_dir = cfg.paths.project_root / "reports" / "milestone11" / ("_sample" if controls.smoke else "")
        self.shards_dir = self.data_dir / "checkpoints"
        self.sequences_dir = self.data_dir / "sequence_shards"

    def write_frame(self, df: pl.DataFrame, name: str, csv: bool = True) -> pl.DataFrame:
        """Persist a table WITHOUT a static contract (result tables whose columns follow the executed design): Parquet in
        the data directory and, when ``csv``, a CSV copy in the tables directory. Atomic: written to a unique temporary
        name, then moved into place. Honours SAVE_TABLES."""
        if not self.controls.save_tables:
            return df
        import os
        import uuid

        self.data_dir.mkdir(parents=True, exist_ok=True)
        p = self.data_dir / f"{name}.parquet"
        tmp = p.with_name(f".{p.stem}.{uuid.uuid4().hex}.tmp")
        df.write_parquet(tmp)
        os.replace(tmp, p)
        if csv:
            self.tables_dir.mkdir(parents=True, exist_ok=True)
            flat = df.select([c for c, t in df.schema.items() if not isinstance(t, (pl.List, pl.Struct, pl.Array))])
            q = self.tables_dir / f"{name}.csv"
            tq = q.with_name(f".{q.stem}.{uuid.uuid4().hex}.tmp")
            flat.write_csv(tq)
            os.replace(tq, q)
        return df

    def read_frame(self, name: str, notebook: str) -> pl.DataFrame:
        return self.require_named(name, notebook)

    def has(self, name: str) -> bool:
        return (self.data_dir / f"{name}.parquet").is_file()


class ResearchStore(DiscoveryStore):
    """Store of Milestone 12 (advanced movement intelligence; notebooks 12.0, 12A–12G): versioned sub-area ``advanced_v1``
    with its own manifest, checkpoint shards and model directory, so nothing of Milestones 00–11 can be replaced. Smoke
    mode: ``…/advanced_movement/_sample/advanced_v1``. ``discovery`` reads the Milestone 11 outputs (11.0 foundation,
    envelope histograms, envelopes, targets) — READ ONLY. ``reports_dir`` = ``reports/milestone12``.
    ``write_frame`` / ``read_frame`` / ``has`` are inherited (atomic, unique temporary names)."""

    AREA = "advanced_v1"

    def __init__(self, cfg: Any, controls: NotebookControls):
        super().__init__(cfg, controls)
        self.discovery = DiscoveryStore(cfg, controls)
        self.reports_dir = cfg.paths.project_root / "reports" / "milestone12" / ("_sample" if controls.smoke else "")
        self.shards_dir = self.data_dir / "checkpoints"
        self.sequences_dir = self.data_dir / "sequence_shards"
        self.models_dir = self.models_dir / "m12"

    def scan(self, name: str, notebook: str, discovery: bool = False) -> pl.LazyFrame:
        """Lazy scan (column projection / predicate pushdown) of a persisted dataset of this area or, with ``discovery``,
        of the Milestone 11 area — for large tables (``m11_envelope_hist``)."""
        d = (self.discovery if discovery else self).data_dir
        p = d / f"{name}.parquet"
        if not p.is_file():
            raise MissingArtifactError(f"`{name}` is missing for RUN_MODE = {self.controls.run_mode!r} ({self.cfg.paths.rel(p)}). "
                                       f"Run notebook `{notebook}` with the same RUN_MODE first.")
        return pl.scan_parquet(p)
