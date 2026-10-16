"""Report helpers: Markdown builder, table export (Parquet + CSV) and figure styling.

Figures are static PNGs for reports. Colours follow a validated colour-blind-safe categorical
order (blue, orange, aqua, ...). Scatter-like plots use at most the first three slots; more
categories are folded into "Other" or faceted. Sequential data uses a single blue ramp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

from .utils import atomic_write_csv, atomic_write_parquet, atomic_write_text, utc_now

# ----------------------------------------------------------------------------- palette
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
SEVERITY_COLORS = {"CRITICAL": STATUS["critical"], "HIGH": STATUS["serious"],
                   "MEDIUM": STATUS["warning"], "LOW": INK["muted"], "INFO": INK["axis"]}

FIG_16x9 = (12.8, 7.2)


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.figsize": FIG_16x9,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "axes.edgecolor": INK["axis"],
        "axes.labelcolor": INK["secondary"],
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "text.color": INK["primary"],
        "legend.frameon": False,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "lines.linewidth": 2.0,
    })


def new_figure(nrows: int = 1, ncols: int = 1, **kwargs: Any):
    import matplotlib.pyplot as plt

    apply_style()
    kwargs.setdefault("figsize", FIG_16x9)
    kwargs.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, **kwargs)


def add_caption(fig, text: str) -> None:
    """Source / sample-size note under the plot (every figure states its n and provenance)."""
    fig.text(0.01, 0.005, text, fontsize=9, color=INK["muted"], ha="left", va="bottom")


def save_figure(fig, directory: Path, stem: str, close: bool = True, dpi: int | None = None,
                formats: Sequence[str] = ("png",), metadata: dict[str, str] | None = None) -> Path:
    """Save ``fig`` as ``<directory>/<stem>.<fmt>`` for every format; returns the first path.

    Defaults keep Phase 01 behaviour (PNG at rcParams dpi). Phase 02 passes ``dpi=300`` and
    ``formats=("png", "svg")``. ``metadata`` is embedded where the backend supports it
    (PNG text chunks, SVG/PDF metadata), e.g. source table and sample size.
    """
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    summary = "; ".join(f"{k}={v}" for k, v in (metadata or {}).items())
    paths = []
    for fmt in formats:
        path = directory / f"{stem}.{fmt}"
        kwargs: dict = {"bbox_inches": "tight"}
        if dpi is not None:  # None keeps rcParams['savefig.dpi'] (Phase 01 behaviour)
            kwargs["dpi"] = dpi
        if metadata:
            if fmt == "png":   # PNG accepts arbitrary text keys
                kwargs["metadata"] = {"Title": stem, **{k: str(v) for k, v in metadata.items()}}
            elif fmt == "svg":
                kwargs["metadata"] = {"Title": stem, "Description": summary}
            elif fmt == "pdf":  # PDF accepts only standard keys
                kwargs["metadata"] = {"Title": stem, "Subject": summary}
        fig.savefig(path, **kwargs)
        paths.append(path)
    if close:
        plt.close(fig)
    return paths[0]


# ----------------------------------------------------------------------------- tables
def export_table(df: pl.DataFrame, directory: Path, stem: str, csv: bool = True) -> list[Path]:
    """Write a result table as Parquet (typed, reloadable) and CSV (human-readable)."""
    directory.mkdir(parents=True, exist_ok=True)
    out = [atomic_write_parquet(df, directory / f"{stem}.parquet")]
    if csv:
        out.append(atomic_write_csv(df, directory / f"{stem}.csv"))
    return out


def df_to_markdown(df: pl.DataFrame, max_rows: int = 60, float_digits: int = 4) -> str:
    if df.is_empty():
        return "_(no rows)_\n"
    shown = df.head(max_rows)

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:,.{float_digits}g}" if abs(v) < 1e6 else f"{v:,.0f}"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        if isinstance(v, (list, tuple)):
            return ", ".join(map(str, v))
        return str(v).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(shown.columns) + " |"
    sep = "|" + "|".join("---" for _ in shown.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in shown.iter_rows()]
    more = f"\n_… {df.height - max_rows:,} more rows in the exported table._\n" if df.height > max_rows else "\n"
    return "\n".join([header, sep, *rows]) + more


# ----------------------------------------------------------------------------- markdown
@dataclass
class MarkdownReport:
    """Incrementally built Markdown report written atomically.

    Evidence labels are first-class: use :meth:`confirmed`, :meth:`inferred` and
    :meth:`unresolved` so readers can always tell measured facts from interpretation.
    """

    title: str
    subtitle: str = ""
    parts: list[str] = field(default_factory=list)

    def h2(self, text: str) -> "MarkdownReport":
        self.parts.append(f"\n## {text}\n")
        return self

    def h3(self, text: str) -> "MarkdownReport":
        self.parts.append(f"\n### {text}\n")
        return self

    def p(self, text: str) -> "MarkdownReport":
        self.parts.append(text.strip() + "\n")
        return self

    def bullets(self, items: Iterable[str]) -> "MarkdownReport":
        self.parts.append("\n".join(f"- {i}" for i in items) + "\n")
        return self

    def table(self, df: pl.DataFrame, max_rows: int = 60) -> "MarkdownReport":
        self.parts.append(df_to_markdown(df, max_rows=max_rows))
        return self

    def kv(self, mapping: dict[str, Any]) -> "MarkdownReport":
        self.parts.append(df_to_markdown(pl.DataFrame({"item": list(mapping), "value": [str(v) for v in mapping.values()]})))
        return self

    def image(self, path: Path, report_dir: Path, alt: str = "") -> "MarkdownReport":
        try:
            rel = Path(path).resolve().relative_to(report_dir.resolve()).as_posix()
        except ValueError:
            rel = Path(path).as_posix()
        self.parts.append(f"\n![{alt}]({rel})\n")
        return self

    def callout(self, kind: str, text: str) -> "MarkdownReport":
        self.parts.append(f"\n> [!{kind.upper()}]\n> " + text.strip().replace("\n", "\n> ") + "\n")
        return self

    def confirmed(self, items: Sequence[str]) -> "MarkdownReport":
        return self.h3("✅ Confirmed (measured)").bullets(items or ["_none recorded_"])

    def inferred(self, items: Sequence[str]) -> "MarkdownReport":
        return self.h3("🔎 Inferred (interpretation — verify)").bullets(items or ["_none recorded_"])

    def unresolved(self, items: Sequence[str]) -> "MarkdownReport":
        return self.h3("❓ Unresolved questions").bullets(items or ["_none recorded_"])

    def render(self) -> str:
        head = f"# {self.title}\n\n"
        if self.subtitle:
            head += f"_{self.subtitle}_\n\n"
        head += f"_Generated {utc_now()} by `nflbdb.data` — numbers below come from executed code only._\n"
        return head + "\n".join(self.parts)

    def save(self, path: Path) -> Path:
        return atomic_write_text(self.render(), path)
