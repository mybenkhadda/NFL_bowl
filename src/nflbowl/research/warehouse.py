"""The research-warehouse context of the unified notebook (Mode B), built ONCE for every Milestone 12 stage.

The eight source notebooks each built::

    cfg = Phase01Config.load(); configure_runtime(cfg)
    controls = amc.NotebookControls(run_mode=..., force_rebuild=..., save_figures=..., save_tables=..., seed=..., num_workers=0).validate()
    store = ResearchStore(cfg, controls); amc.seed_everything(controls.seed)
    C = m12c.M12Config().for_mode(controls); S = src.M11Sources(cfg, store, m11c.M11Config().for_mode(controls)); D = fd.M12Data(store, C, S)

``WarehouseContext`` builds exactly these objects (same classes, same arguments) with ``project_root`` set to the
research root on Drive. ``store`` is a ``MirroredResearchStore``: a ``ResearchStore`` whose ``write_frame`` / ``show`` /
``write_json`` behave identically (same warehouse paths, same atomic writes, same figure files) and additionally copy
each output into the unified run directory (tables → ``tables/``, foundation datasets → ``processed_data/``, figures →
``figures/png|svg|pdf`` with descriptive names, JSON → ``manifests/research/``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ResearchRun, descriptive_figure_name

# result tables that are reusable intermediate datasets rather than reported results
PROCESSED = {"m12_combine_curves", "m12_combine_speed_time", "m12_combine_hist", "m12_combine_quality", "m12_nfl_bin_support", "m12_game_top_speed",
             "m12_cohort_registry", "m12_movement_grids", "m12_shared_speed_support", "m12_fold_manifest", "m12c_targets", "m12c_split_traits",
             "m12c_forecast_athletes", "m12c_trait_uncertainty", "m12b_heldout_oof", "m12b_synthetic_replicates", "m12a_phase_amplitude",
             "m12d_availability", "m12d_trait_uncertainty", "m12e_retrieval_athletes", "m12c_transfer_oof"}


def _mirror_class() -> type:
    from nflbdb.research.advanced_movement.artifacts import ResearchStore

    class MirroredResearchStore(ResearchStore):
        """``ResearchStore`` + a copy of every output in the unified run directory (see the module docstring)."""

        def __init__(self, cfg: Any, controls: Any, run: ResearchRun | None = None):
            super().__init__(cfg, controls)
            self.run = run
            self.stage = "m12"

        # ---------------------------------------------------------------- tables
        def write_frame(self, df: Any, name: str, csv: bool = True) -> Any:
            out = super().write_frame(df, name, csv)
            if self.run is not None and self.controls.save_tables and df is not None:
                if name in PROCESSED or name.startswith("m12f_inductive_"):
                    self.run.save_processed_dataset(df, name, self.stage, {"warehouse_table": name, "warehouse_dir": str(self.data_dir)})
                else:
                    self.run.save_table(df, name, self.stage, note=f"warehouse table {name}")
            return out

        # ---------------------------------------------------------------- figures
        def show(self, fig: Any, name: str, plot_data: dict[str, Any] | None = None, publication: bool = False, source: str = "", n: str = "",
                 note: str = "") -> None:
            if self.run is not None and self.controls.save_figures:
                self.run.save_figure(fig, descriptive_figure_name(name, self.stage), self.stage, plot_data or {},
                                     {"research_id": name, "source": source, "n": n, "note": note, "publication": publication})
            super().show(fig, name, plot_data, publication, source, n, note)        # display + warehouse files (closes the figure)

        # ---------------------------------------------------------------- small JSON records
        def write_json(self, name: str, obj: dict[str, Any]) -> Path:
            p = super().write_json(name, obj)
            if self.run is not None:
                self.run.copy_in(p, f"manifests/research/{Path(p).name}", self.stage, kind="research_json")
            return p

    return MirroredResearchStore


@dataclass
class WarehouseContext:
    """Everything a Milestone 12 stage needs from the warehouse (built once)."""

    cfg: Any                    # nflbdb Phase01Config
    controls: Any               # amc.NotebookControls
    store: Any                  # MirroredResearchStore
    C: Any                      # m12c.M12Config().for_mode(controls)
    C11: Any                    # m11c.M11Config().for_mode(controls)
    S: Any                      # m11 sources
    D: Any                      # fd.M12Data
    research_root: Path

    @classmethod
    def build(cls, research_root: Path | str, run: ResearchRun, run_mode: str = "full", seed: int = 20261015, save_tables: bool = True,
              save_figures: bool = True, force_rebuild: bool = False) -> "WarehouseContext":
        from nflbdb.data.config import Phase01Config, configure_runtime

        cfg = Phase01Config.load(project_root=research_root)
        configure_runtime(cfg)
        from nflbdb.research.advanced_movement import config as amc
        from nflbdb.research.advanced_movement.m11 import config as m11c
        from nflbdb.research.advanced_movement.m11 import sources as src
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import foundation as fd

        controls = amc.NotebookControls(run_mode=run_mode, force_rebuild=force_rebuild, save_figures=save_figures, save_tables=save_tables,
                                        seed=seed, num_workers=0).validate()
        store = _mirror_class()(cfg, controls, run)
        amc.seed_everything(controls.seed)
        C = m12c.M12Config().for_mode(controls)
        C11 = m11c.M11Config().for_mode(controls)
        S = src.M11Sources(cfg, store, C11)
        D = fd.M12Data(store, C, S)
        return cls(cfg, controls, store, C, C11, S, D, Path(research_root))

    def set_stage(self, stage: str) -> None:
        self.store.stage = stage

    def provenance(self, track: str, **extra: Any) -> dict[str, Any]:
        from nflbdb.research.advanced_movement.m12 import config as m12c

        return m12c.provenance(track, self.controls, self.C, **extra)

    def describe(self) -> Any:
        return self.store.describe()


__all__ = ["PROCESSED", "WarehouseContext"]
