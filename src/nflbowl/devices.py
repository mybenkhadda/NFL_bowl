"""Device, precision and backend-flag resolution. PyTorch is optional: everything here degrades to CPU without it.

Precision semantics (recorded in every checkpoint signature):
    auto  → CUDA: fp16 autocast + GradScaler (EXACTLY the 12F notebook: ``torch.autocast("cuda")`` defaults to fp16);
            CPU / MPS: fp32.
    fp32  → no autocast.
    fp16  → CUDA only, with GradScaler.
    bf16  → CUDA with ``torch.cuda.is_bf16_supported()`` (Ampere+), or CPU autocast; no GradScaler. NOT the notebook's
            numerics — opt-in only.
Validation losses and embeddings are computed in fp32 unless ``runtime.embed_precision`` says otherwise (notebook).
"""

from __future__ import annotations

import contextlib
import gc
import os
import platform
import sys
from dataclasses import dataclass
from typing import Any

from .utils.logging import get_logger

log = get_logger(__name__)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def require_torch(what: str = "this step") -> Any:
    try:
        import torch
    except ImportError as e:
        raise ImportError(f"{what} needs PyTorch, which is not installed. On Colab / Kaggle it is preinstalled; locally install the "
                          f"build for your platform (https://pytorch.org/get-started/locally/) or `pip install 'nflbdb[torch]'` for CPU.") from e
    return torch


def resolve_device(spec: str = "auto") -> str:
    """'auto' → 'cuda' when available, else 'cpu' (the 12F notebook rule). Explicit 'cuda' without a GPU raises: a
    silent CPU fallback would turn a 10-minute job into hours without telling anyone."""
    spec = str(spec or "auto")
    if spec == "cpu":
        return "cpu"
    if not torch_available():
        if spec == "auto":
            return "cpu"
        raise RuntimeError(f"device {spec!r} requested but PyTorch is not installed")
    import torch

    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {spec!r} requested but torch.cuda.is_available() is False (Colab: Runtime → Change runtime type → GPU)")
        return spec
    if spec == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("device 'mps' requested but MPS is not available")
        return "mps"
    raise RuntimeError(f"unknown device {spec!r}")


def device_type(device: str) -> str:
    return str(device).split(":")[0]


@dataclass(frozen=True)
class Precision:
    name: str                      # fp32 · fp16 · bf16
    autocast_dtype: Any            # torch dtype or None
    use_scaler: bool

    def signature(self) -> dict[str, Any]:
        return {"precision": self.name, "grad_scaler": self.use_scaler}


def resolve_precision(precision: str, device: str, training: bool = True) -> Precision:
    dt = device_type(device)
    if precision == "auto":
        precision = "fp16" if (dt == "cuda" and training) else "fp32"
    if precision == "fp32":
        return Precision("fp32", None, False)
    torch = require_torch("mixed precision")
    if precision == "fp16":
        if dt != "cuda":
            raise RuntimeError("precision 'fp16' is only supported on CUDA here; use 'fp32' (or 'bf16' on CPU)")
        return Precision("fp16", torch.float16, training)
    if precision == "bf16":
        if dt == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support bf16 (e.g. Colab T4); use precision 'auto' / 'fp16'")
        if dt not in ("cuda", "cpu"):
            raise RuntimeError(f"bf16 autocast is not supported on {dt!r}")
        return Precision("bf16", torch.bfloat16, False)
    raise RuntimeError(f"unknown precision {precision!r}")


def autocast(device: str, prec: Precision) -> Any:
    if prec.autocast_dtype is None:
        return contextlib.nullcontext()
    import torch

    return torch.autocast(device_type=device_type(device), dtype=prec.autocast_dtype)


def grad_scaler(device: str, prec: Precision) -> Any:
    if not prec.use_scaler:
        return None
    import torch

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type(device))
        except TypeError:                                  # very old signature
            pass
    return torch.cuda.amp.GradScaler()


def apply_backend_flags(runtime: Any) -> dict[str, Any]:
    """Apply only the flags the user set (None = leave PyTorch's default). Returns the effective values for metadata."""
    out: dict[str, Any] = {}
    if not torch_available():
        return out
    import torch

    if runtime.cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = bool(runtime.cudnn_benchmark)
    if runtime.allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(runtime.allow_tf32)
    if runtime.num_threads:
        torch.set_num_threads(int(runtime.num_threads))
    out.update({"cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "torch_threads": int(torch.get_num_threads())})
    return out


def free_gpu_bytes(device: str) -> int | None:
    if device_type(device) != "cuda" or not torch_available():
        return None
    import torch

    try:
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        return int(free)
    except Exception:                                      # noqa: BLE001 — informational only
        return None


def cleanup(device: str | None = None) -> None:
    """Release Python garbage and cached CUDA blocks between models (does not touch live tensors)."""
    gc.collect()
    if device is not None and device_type(device) == "cuda" and torch_available():
        import torch

        torch.cuda.empty_cache()


def environment_kind() -> str:
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working"):
        return "kaggle"
    return "local"


def device_info(verbose: bool = True) -> dict[str, Any]:
    """Python / PyTorch / CUDA / memory facts of this runtime. Prints a short table when ``verbose``."""
    info: dict[str, Any] = {"environment": environment_kind(), "python": platform.python_version(), "platform": platform.platform(),
                            "cpu_count": os.cpu_count()}
    try:
        import psutil

        vm = psutil.virtual_memory()
        info.update({"ram_total_gb": round(vm.total / 2 ** 30, 1), "ram_available_gb": round(vm.available / 2 ** 30, 1)})
    except ImportError:
        pass
    if torch_available():
        import torch

        info.update({"torch": torch.__version__, "torch_cuda_build": getattr(torch.version, "cuda", None), "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            free = free_gpu_bytes("cuda")
            info.update({"gpu": p.name, "gpu_count": torch.cuda.device_count(), "compute_capability": f"{p.major}.{p.minor}",
                         "gpu_memory_gb": round(p.total_memory / 2 ** 30, 1), "gpu_free_gb": round(free / 2 ** 30, 1) if free else None,
                         "bf16_supported": bool(torch.cuda.is_bf16_supported()), "cudnn": torch.backends.cudnn.version()})
        info["compile_available"] = hasattr(torch, "compile")
    else:
        info.update({"torch": None, "cuda_available": False})
    info["resolved_device"] = resolve_device("auto")
    if verbose:
        width = max(len(k) for k in info)
        print("\n".join(f"{k.ljust(width)} : {v}" for k, v in info.items()))
    return info


__all__ = ["Precision", "apply_backend_flags", "autocast", "cleanup", "device_info", "device_type", "environment_kind", "free_gpu_bytes",
           "grad_scaler", "require_torch", "resolve_device", "resolve_precision", "torch_available"]
