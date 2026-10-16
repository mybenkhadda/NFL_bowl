"""``LabRun`` — the Drive output tree, artifact manifest and stage ledger of the comprehensive lab.

    RUN_DIR = <drive_root>/experimental_lab/<experiment_name>/
        config/        run_config.json, environment.json, experiment_registry.csv|md
        datasets/      audits/ (dataset audit tables) · cached/ (reusable per-sequence / per-athlete derived data)
        features/      sequence/ · athlete/ (biomechanical feature tables)
        embeddings/    transductive/ · inductive/ (athlete-level embedding tables exported from the nflbowl experiment)
        checkpoints/   experiments/<experiment_name>/ — the nflbowl Experiment (epoch shards, best.pt, optimizer state)
        tables/        dataset/ retrieval/ prediction/ ablations/ robustness/ interpretability/ geometry/ biomechanics/ synthesis/
        figures/       png/ svg/ pdf/ · data/ (the plotted numbers, CSV + Parquet, one file per figure part)
        reports/       markdown/ · html/ · athlete_profiles/
        diagnostics/   leakage audit, validation checks, gate inputs
        logs/          run.log, <stage>_traceback.txt
        manifests/     artifacts.jsonl, artifact_index.json, stages.json (the experiment manifest), cache records
        exports/       ZIP archives

Built on ``nflbowl.research.artifacts.ResearchRun`` (atomic writes, append-only artifact log, stage records); this class
adds the lab layout, per-family table folders, cache provenance records and a guarded ``stage`` context.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from ..artifacts.store import atomic_json, atomic_write
from ..research.artifacts import STATUS_DONE, STATUS_FAILED, STATUS_REUSED, STATUS_RUNNING, STATUS_SKIPPED, ResearchRun, safe_name
from ..utils.reproducibility import code_version, utc_now

LAB_LAYOUT = ("config", "datasets/audits", "datasets/cached", "features/sequence", "features/athlete", "embeddings/transductive",
              "embeddings/inductive", "checkpoints", "tables/dataset", "tables/retrieval", "tables/prediction", "tables/ablations",
              "tables/robustness", "tables/interpretability", "tables/geometry", "tables/biomechanics", "tables/synthesis", "figures/png",
              "figures/svg", "figures/pdf", "figures/data", "reports/markdown", "reports/html", "reports/athlete_profiles", "diagnostics",
              "logs", "manifests", "exports")
FAMILIES = ("dataset", "retrieval", "prediction", "ablations", "robustness", "interpretability", "geometry", "biomechanics", "synthesis")
STATUS_PENDING, STATUS_BLOCKED = "pending", "blocked"


class StageSkipped(Exception):
    """Raised inside a ``LabRun.stage`` block to record a deliberate skip (prerequisite missing, flag off)."""


class LabRun(ResearchRun):
    def __init__(self, root: Path | str, cfg: Any = None, local_tmp: Path | str | None = None):
        root = Path(root)
        for d in LAB_LAYOUT:
            (root / d).mkdir(parents=True, exist_ok=True)
        super().__init__(root, cfg, local_tmp)
        self.results: dict[str, Any] = {}             # in-memory hand-off between sections (also rebuilt from saved tables)

    # ------------------------------------------------------------------ tables
    def table(self, df: Any, name: str, family: str, note: str = "") -> list[Path]:
        """Result table → ``tables/<family>/<name>.parquet`` + ``.csv`` (never truncated; nested columns JSON-encoded)."""
        if family not in FAMILIES:
            raise ValueError(f"unknown table family {family!r}; one of {FAMILIES}")
        return self.save_table(df, name, csv=None, note=note, subdir=f"tables/{family}")

    def frame(self, name: str, family: str) -> Any:
        """Reload a saved table (Parquet) of a family; FileNotFoundError when it was never produced."""
        import polars as pl

        p = self.root / "tables" / family / f"{safe_name(name)}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"table {family}/{name} has not been produced in this run")
        return pl.read_parquet(p)

    def has(self, name: str, family: str) -> bool:
        return (self.root / "tables" / family / f"{safe_name(name)}.parquet").is_file()

    def data_file(self, df: Any, rel: str, provenance: dict[str, Any] | None = None) -> Path | None:
        """Reusable data (``datasets/…``, ``features/…``, ``embeddings/…``, ``diagnostics/…``) as Parquet + provenance sidecar."""
        if df is None or not self._flag("save_processed_data"):
            return None
        p = self.root / rel
        p = p if p.suffix == ".parquet" else p.with_suffix(".parquet")
        self._publish_local(p, lambda t: df.write_parquet(t))
        prov = {"path": self.rel(p), "rows": int(getattr(df, "height", 0)), "columns": list(getattr(df, "columns", []))[:80], "written_at": utc_now(),
                "code_version": code_version(), **(provenance or {})}
        atomic_json(prov, p.with_suffix(".provenance.json"), overwrite=True)
        self.register("data", safe_name(rel), [p], meta={"rows": prov["rows"]})
        return p

    def load_data(self, rel: str, provenance: dict[str, Any] | None = None) -> Any:
        """A cached data file whose provenance matches every key of ``provenance`` (else None → recompute)."""
        import polars as pl

        p = self.root / rel
        p = p if p.suffix == ".parquet" else p.with_suffix(".parquet")
        side = p.with_suffix(".provenance.json")
        if not (p.is_file() and side.is_file()):
            return None
        try:
            have = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if any(have.get(k) != v for k, v in (provenance or {}).items()):
            return None
        return pl.read_parquet(p)

    # ------------------------------------------------------------------ figures
    def figure(self, fig: Any, name: str, data: dict[str, Any] | None = None, question: str = "", caption: str = "", show: bool = True) -> dict[str, str]:
        """Display, save (PNG 300 dpi · SVG · PDF + plotted data) and close one figure. ``question`` = the research
        question it answers; ``caption`` = what it shows (both go into the sidecar and the figure index)."""
        if fig is None:
            return {}
        if show:
            try:
                from IPython.display import display

                display(fig)
            except ImportError:
                pass
        out = self.save_figure(fig, name, data=data, meta={"question": question, "caption": caption, "source": question})
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:                                           # noqa: BLE001 — Figure objects built without pyplot
            pass
        self.results.setdefault("_figures", []).append({"figure": safe_name(name), "question": question, "caption": caption, **out})
        return out

    # ------------------------------------------------------------------ reports
    def report(self, text: str, name: str, family: str | None = None) -> list[Path]:
        return self.save_markdown(text, name)

    # ------------------------------------------------------------------ stages
    def run_stage(self, stage: str, fn: Callable[[dict[str, Any]], Any], note: str = "", inputs: Sequence[str] = (), enabled: bool = True,
                  reason_disabled: str = "", raise_errors: bool = False) -> Any:
        """Guarded section ``fn(ctx)``: records start / end / duration / status / inputs / outputs / error in
        ``manifests/stages.json`` and returns ``fn``'s result (None when skipped or failed).

        * ``enabled=False`` → status ``skipped`` with ``reason_disabled``; ``fn`` is not called.
        * ``raise StageSkipped(reason)`` inside ``fn`` → ``skipped`` (``blocked`` when the reason starts with 'blocked').
        * any other exception → ``failed`` with the traceback in ``logs/``; not re-raised unless ``raise_errors`` (so later
          independent sections still run). ``fn`` may set ``ctx['status']`` / ``ctx['cache']`` and append to ``ctx['outputs']``.
        """
        ctx: dict[str, Any] = {"outputs": [], "status": None, "cache": None}
        self.stage_begin(stage, note)
        self._stages[stage]["inputs"] = list(inputs)
        if not enabled:
            self.stage_end(stage, STATUS_SKIPPED, note=reason_disabled or "disabled by configuration", seconds=0.0, outputs=[])
            print(f"[{stage}] skipped: {reason_disabled or 'disabled by configuration'}")
            return None
        t0 = time.monotonic()
        try:
            out = fn(ctx)
        except StageSkipped as s:
            msg = str(s)
            self.stage_end(stage, STATUS_BLOCKED if msg.lower().startswith("blocked") else STATUS_SKIPPED, note=msg,
                           seconds=round(time.monotonic() - t0, 2), outputs=ctx["outputs"])
            print(f"[{stage}] skipped: {msg}")
            return None
        except Exception as e:                                       # noqa: BLE001 — isolate failures per section
            self._stages[stage]["outputs"] = ctx["outputs"]
            self.stage_fail(stage, e)
            print(f"[{stage}] FAILED: {type(e).__name__}: {e}  (traceback in logs/{stage.replace('.', '_').lower()}_traceback.txt)")
            if raise_errors:
                raise
            return None
        status = ctx["status"] or (STATUS_REUSED if ctx.get("cache") == "hit" else STATUS_DONE)
        self.stage_end(stage, status, note=note, seconds=round(time.monotonic() - t0, 2), outputs=ctx["outputs"], cache=ctx.get("cache"))
        return out

    def stage_ok(self, stage: str) -> bool:
        return self.stage_status(stage) in (STATUS_DONE, STATUS_REUSED)

    def stage_records(self) -> dict[str, dict[str, Any]]:
        return dict(self._stages)

    def write_config(self, cfg: Any, environment: dict[str, Any]) -> list[Path]:
        a = atomic_json(cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg), self.root / "config" / "run_config.json", overwrite=True)
        b = atomic_json(environment, self.root / "config" / "environment.json", overwrite=True)
        with open(self.root / "config" / "config_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": utc_now(), "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)}, default=str) + "\n")
        return [a, b]

    def write_text(self, text: str, rel: str) -> Path:
        return atomic_write(self.root / rel, lambda p: p.write_text(text, encoding="utf-8"))


__all__ = ["FAMILIES", "LAB_LAYOUT", "LabRun", "STATUS_BLOCKED", "STATUS_DONE", "STATUS_FAILED", "STATUS_PENDING", "STATUS_REUSED", "STATUS_RUNNING",
           "STATUS_SKIPPED", "StageSkipped"]
