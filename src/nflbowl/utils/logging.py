"""Package logger (``nflbowl.*``), optionally mirrored to an experiment log file."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "nflbowl") -> logging.Logger:
    root = logging.getLogger("nflbowl")
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        root.addHandler(h)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logging.getLogger(name)


def attach_file_log(path: Path) -> logging.Handler:
    """Append every ``nflbowl`` log record to ``path`` (one handler per file; idempotent)."""
    root = get_logger()
    path = Path(path)
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve():
            return h
    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter(_FORMAT, "%Y-%m-%dT%H:%M:%S"))
    root.addHandler(h)
    return h


__all__ = ["attach_file_log", "get_logger"]
