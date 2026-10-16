"""Figure / table export.

Figures → ``$LAB_FIGURE_DIR`` (bind mount of reports/figures/analytics_lab):
  PNG at 300 DPI (pixel size = inches × DPI; the PNG carries a 300-DPI pHYs tag) + SVG.
Tables  → ``$LAB_TABLE_DIR`` (reports/tables/analytics_lab) as CSV.
File names: ``<descriptive_stem>__<YYYYmmdd-HHMMSS>.<ext>`` — never overwrites an earlier export.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

FIG_DIR = Path(os.environ.get("LAB_FIGURE_DIR", "/exports/figures"))
TABLE_DIR = Path(os.environ.get("LAB_TABLE_DIR", "/exports/tables"))
DPI = 300


def stem(*parts) -> str:
    text = "_".join(str(p) for p in parts if p not in (None, ""))
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-_").lower()[:150]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _static(fig: go.Figure) -> go.Figure:
    out = go.Figure(fig)
    out.frames = []
    out.update_layout(updatemenus=[], sliders=[])
    return out


def png_bytes(fig: go.Figure, width_in: float = 10, height_in: float = 6, dpi: int = DPI) -> bytes:
    from PIL import Image

    raw = _static(fig).to_image(format="png", width=int(width_in * 100), height=int(height_in * 100), scale=dpi / 100)
    buf = io.BytesIO()
    Image.open(io.BytesIO(raw)).save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


def save_figure(fig: go.Figure, name: str, width_in: float = 10, height_in: float = 6) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / f"{name}__{_stamp()}"
    png = base.with_suffix(".png")
    png.write_bytes(png_bytes(fig, width_in, height_in))
    svg = base.with_suffix(".svg")
    _static(fig).write_image(svg, format="svg", width=int(width_in * 100), height=int(height_in * 100))
    return [png, svg]


def save_csv(df: pd.DataFrame, name: str) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{name}__{_stamp()}.csv"
    df.to_csv(path, index=False)
    return path
