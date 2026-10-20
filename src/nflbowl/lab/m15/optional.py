"""Lazy optional imports. A missing package blocks only the steps that need it (``BLOCKED_MISSING_INPUT``)."""

from __future__ import annotations

import importlib
from typing import Any

from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked

# import name → pip name (for the message)
PACKAGES = {"scipy": "scipy", "sklearn": "scikit-learn", "statsmodels": "statsmodels", "pywt": "PyWavelets", "umap": "umap-learn",
            "lightgbm": "lightgbm", "catboost": "catboost", "ot": "POT", "torch": "torch", "ipywidgets": "ipywidgets"}


def available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def require(name: str, what: str = "this step") -> Any:
    """The imported module, or ``ResearchBlocked(BLOCKED_MISSING_INPUT)`` naming the pip package to install."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"{what} needs the optional package {PACKAGES.get(name, name)!r} ({e})",
                              missing=[PACKAGES.get(name, name)]) from e


def report() -> Any:
    """Availability table of every optional package (shown in notebook Setup 2)."""
    import polars as pl

    rows = []
    for mod, pip in PACKAGES.items():
        ok, ver = False, None
        try:
            m = importlib.import_module(mod)
            ok, ver = True, str(getattr(m, "__version__", ""))
        except ImportError:
            pass
        rows.append({"import": mod, "pip": pip, "available": ok, "version": ver})
    return pl.DataFrame(rows)


__all__ = ["PACKAGES", "available", "report", "require"]
