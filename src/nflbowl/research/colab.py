"""Colab runtime helpers: environment facts, Google Drive, local staging, version / fix verification.

Nothing here installs packages (the notebook's first cell does that with ``%pip``) and nothing replaces Colab's CUDA
build of PyTorch.
"""

from __future__ import annotations

import importlib
import inspect
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from ..devices import device_info, environment_kind

THIRD_PARTY = ("numpy", "polars", "pyarrow", "psutil", "yaml", "matplotlib", "duckdb", "torch")
OPTIONAL = ("ot", "pymc", "markdown")


def in_colab() -> bool:
    return environment_kind() == "colab"


def mount_drive(mountpoint: str = "/content/drive", force_remount: bool = False) -> bool:
    """Mount Google Drive in Colab (no-op elsewhere). Returns True when ``mountpoint/MyDrive`` is visible."""
    if not in_colab():
        return Path(mountpoint, "MyDrive").is_dir()
    if not Path(mountpoint, "MyDrive").is_dir() or force_remount:
        from google.colab import drive                       # type: ignore[import-not-found]

        drive.mount(mountpoint, force_remount=force_remount)
    return Path(mountpoint, "MyDrive").is_dir()


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for m in (*THIRD_PARTY, *OPTIONAL, "nflbowl", "nflbdb"):
        try:
            mod = importlib.import_module(m)
            out[m] = str(getattr(mod, "__version__", "installed"))
        except Exception:                                    # noqa: BLE001
            out[m] = None
    return out


def missing_required() -> list[str]:
    return [m for m, v in package_versions().items() if m in THIRD_PARTY and m != "torch" and v is None]


def runtime_report(verbose: bool = True) -> dict[str, Any]:
    """GPU / CUDA / memory / disk facts (``nflbowl.device_info`` + local disk + versions)."""
    info = device_info(verbose=False)
    try:
        du = shutil.disk_usage("/content" if Path("/content").is_dir() else ".")
        info["local_disk_free_gb"] = round(du.free / 2 ** 30, 1)
    except OSError:
        info["local_disk_free_gb"] = None
    info["versions"] = package_versions()
    if verbose:
        flat = {k: v for k, v in info.items() if k != "versions"}
        w = max(len(k) for k in flat)
        print("\n".join(f"{k.ljust(w)} : {v}" for k, v in flat.items()))
        print("versions:", ", ".join(f"{k} {v}" for k, v in info["versions"].items() if v))
    return info


def fixed_losses_check() -> dict[str, Any]:
    """Static check that the installed research code carries the session-3 fp16 fix: ``se._fp32`` exists and the
    contrastive / InfoNCE / MMD terms run inside it. No monkey-patch is applied or needed."""
    from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se

    out: dict[str, Any] = {"has__fp32": hasattr(se, "_fp32"), "has__infonce": hasattr(se, "_infonce")}
    for fn in ("_supcon", "_infonce", "_mmd"):
        f = getattr(se, fn, None)
        try:
            out[f"{fn}_uses__fp32"] = bool(f is not None and "_fp32(" in inspect.getsource(f))
        except (OSError, TypeError):
            out[f"{fn}_uses__fp32"] = None
    try:
        out["check_finite_loss"] = hasattr(importlib.import_module("nflbowl.training.trainer"), "check_finite_loss")
    except Exception:                                        # noqa: BLE001 (torch missing)
        out["check_finite_loss"] = None
    out["ok"] = bool(out["has__fp32"] and out.get("_supcon_uses__fp32") and out.get("_infonce_uses__fp32"))
    out["source_file"] = getattr(se, "__file__", None)
    return out


def assert_version(min_version: str = "0.2.1") -> str:
    import nflbowl

    def key(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())

    if key(nflbowl.__version__) < key(min_version):
        raise RuntimeError(f"nflbowl {nflbowl.__version__} is older than {min_version}: re-run the install cell "
                           "(force-reinstall from @main), then Runtime → Restart session")
    return nflbowl.__version__


def stage_tree(src: Path | str, dst: Path | str, patterns: tuple[str, ...] = ("*",)) -> Path:
    """Copy files from ``src`` to ``dst`` once (same size → skipped). For read-heavy inputs on Drive."""
    src, dst = Path(src), Path(dst)
    for pat in patterns:
        for f in src.rglob(pat):
            if f.is_file():
                t = dst / f.relative_to(src)
                if t.is_file() and t.stat().st_size == f.stat().st_size:
                    continue
                t.parent.mkdir(parents=True, exist_ok=True)
                tmp = t.with_name(f".{t.name}.staging")
                shutil.copyfile(f, tmp)
                os.replace(tmp, t)
    return dst


def gpu_memory_summary() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda": False}
    if not torch.cuda.is_available():
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {"cuda": True, "device": torch.cuda.get_device_name(0), "free_gb": round(free / 2 ** 30, 2), "total_gb": round(total / 2 ** 30, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 2), "reserved_gb": round(torch.cuda.memory_reserved() / 2 ** 30, 2)}


__all__ = ["OPTIONAL", "THIRD_PARTY", "assert_version", "fixed_losses_check", "gpu_memory_summary", "in_colab", "missing_required", "mount_drive",
           "package_versions", "runtime_report", "stage_tree"]
