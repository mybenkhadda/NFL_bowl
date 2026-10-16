"""Experiment directory layout and atomic, non-destructive writers.

    <output_dir>/experiments/<experiment_id>/
        config/            experiment.json (resolved config, seed, code version, bundle fingerprint), runtime_history.jsonl
        manifests/         manifest.json (artifact + job index), events.jsonl (append-only history)
        checkpoints/       epoch shards (nflbdb ModelCheckpoint layout) + <model>/<tag>/best.pt exports
        data/processed/    per-sequence targets / ids derived from the bundle (Parquet)
        data/features/     engineered (F0) features (Parquet)
        embeddings/        <model>__<tag>__<hash>.parquet + .json sidecar (ids, domain, fold, split role, signature)
        evaluation/        records/*.json (immutable) → metrics.csv, probes.csv, identity_null.csv, inductive_retrieval.csv,
                           transfer.csv, training_curves.csv, retrieval_results.parquet (views rebuilt from records);
                           paired_inductive_comparison.csv, athlete_retrieval_scores.parquet, position_subgroup_results.csv,
                           domain_diagnostics.csv, inductive_coverage.csv (views of records/inductive_comparison__<hash>.*)
        figures/           png/ svg/ pdf/ source/ (the plotted data as CSV); inductive_*.png / domain_separation.png at top level
        reports/           REPORT.md, INDUCTIVE_RETRIEVAL_REVIEW.md
        logs/              experiment.log
        exports/           zip archives from Experiment.export()

Rules: immutable artifacts carry their dependency hash in the file name, so nothing computed is ever overwritten;
only VIEW files (aggregated CSVs, manifest.json, REPORT.md) are designed to be replaced, and they are rebuilt from
the immutable records. Every write goes through a uniquely named temporary file + ``os.replace`` (retried on Windows
locks), so an interrupted session never leaves a half-written file under a final name.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

LAYOUT = ("config", "manifests", "checkpoints", "data/processed", "data/features", "embeddings", "evaluation/records", "figures/png",
          "figures/svg", "figures/pdf", "figures/source", "reports", "logs", "exports")
RETRIES = 12


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write(final: Path, writer: Callable[[Path], None], overwrite: bool = True) -> Path:
    """Write via ``writer(tmp)`` then publish. ``overwrite=False`` keeps an existing file (and drops the temporary)."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(tmp)
        for attempt in range(RETRIES):
            if not overwrite and final.exists():
                tmp.unlink(missing_ok=True)
                return final
            try:
                os.replace(tmp, final)
                return final
            except PermissionError:                         # Windows: a reader (indexer, antivirus, Drive sync) holds the file
                time.sleep(0.25 * (attempt + 1))
        raise PermissionError(f"could not publish {final} after {RETRIES} attempts; temporary kept at {tmp}")
    except BaseException:
        if tmp.exists() and not final.exists():
            tmp.unlink(missing_ok=True)
        raise


def atomic_json(obj: Any, final: Path, overwrite: bool = True) -> Path:
    return atomic_write(final, lambda p: p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8"), overwrite)


def atomic_text(text: str, final: Path, overwrite: bool = True) -> Path:
    return atomic_write(final, lambda p: p.write_text(text, encoding="utf-8"), overwrite)


def append_jsonl(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


class ExperimentStore:
    """Paths and writers of one experiment directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    # ------------------------------------------------------------------ layout
    def create(self) -> "ExperimentStore":
        for d in LAYOUT:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        return self

    def exists(self) -> bool:
        return (self.root / "config" / "experiment.json").is_file()

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def records(self) -> Path:
        return self.root / "evaluation" / "records"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def features(self) -> Path:
        return self.root / "data" / "features"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def rel(self, p: Path | str) -> str:
        try:
            return Path(p).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(p)

    # ------------------------------------------------------------------ tables
    def write_parquet(self, df: Any, path: Path, overwrite: bool = False) -> Path:
        return atomic_write(path, lambda p: df.write_parquet(p), overwrite)

    def write_csv(self, df: Any, path: Path, overwrite: bool = True) -> Path:
        """CSV view of a table. Nested / object columns are JSON-encoded so the CSV always writes."""
        import polars as pl

        flat = df.with_columns([pl.col(c).map_elements(lambda v: json.dumps(v, default=str), return_dtype=pl.String).alias(c)
                                for c, t in df.schema.items() if t.is_nested() or t == pl.Object]) if df.width else df
        return atomic_write(path, lambda p: flat.write_csv(p), overwrite)

    def read_parquet(self, path: Path) -> Any:
        import polars as pl

        return pl.read_parquet(path)

    # ------------------------------------------------------------------ figures
    def save_figure(self, fig: Any, name: str, formats: tuple[str, ...] = ("png", "svg", "pdf"), source: dict[str, Any] | None = None,
                    dpi: int = 200) -> dict[str, str]:
        """Save one matplotlib figure in every requested format (atomic) and its plotted data as CSV."""
        out: dict[str, str] = {}
        for fmt in formats:
            p = self.figures / fmt / f"{name}.{fmt}"
            atomic_write(p, lambda t, _f=fmt: fig.savefig(t, format=_f, dpi=dpi, bbox_inches="tight"))
            out[fmt] = self.rel(p)
        for key, df in (source or {}).items():
            if df is not None and getattr(df, "height", 0):
                p = self.figures / "source" / f"{name}__{key}.csv"
                self.write_csv(df, p)
                out[f"source:{key}"] = self.rel(p)
        return out


__all__ = ["ExperimentStore", "LAYOUT", "append_jsonl", "atomic_json", "atomic_text", "atomic_write", "sha256_file"]
