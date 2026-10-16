"""Model registry, discovered from the research code (``se.FAMILIES``, ``se.ABLATIONS``) plus the 12F baselines.

Names are the research names (``F4_physics_contrastive`` …) so every artifact stays traceable to the notebook tables;
short aliases (``F4``) resolve to them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .. import _backend

ENGINEERED = "F0_engineered"
RANDOM = "random_frozen_F4_architecture"           # 12F notebook name of the random encoder baseline
TCN = "F1_TCN"                                     # Milestone 06 TCN-B (external, read from the bundle; never trained here)
ALIASES = {"F0": ENGINEERED, "engineered": ENGINEERED, "random": RANDOM, "random_frozen": RANDOM, "F1": TCN, "TCN": TCN,
           "F2": "F2_cnn_physics", "F3": "F3_masked_ae", "F4": "F4_physics_contrastive", "F5": "F5_jepa_physics", "F6": "F6_transformer"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                    # encoder · baseline · external
    group: str                   # family · ablation · baseline
    trainable: bool
    description: str
    encoder: str | None = None   # conv · transformer
    temporal: str | None = None
    align: str | None = None
    lambdas: str | None = None
    primary: bool = False


def _describe(e: Any) -> str:
    parts = [f"{e.encoder} encoder"]
    if e.lambda_mask > 0:
        parts.append("masked reconstruction")
    if e.lambda_phys > 0:
        parts.append("physics heads" + (" (SHUFFLED targets — negative control)" if e.shuffle_phys else ""))
    if e.lambda_temp > 0 and e.temporal != "none":
        parts.append(f"temporal {e.temporal}")
    if e.lambda_align > 0 and e.align != "none":
        parts.append(f"{e.align} cross-domain alignment")
    return " + ".join(parts)


def specs() -> list[ModelSpec]:
    se = _backend.se()
    out = [ModelSpec(ENGINEERED, "baseline", "baseline", False, "17 masked summary statistics per sequence (rep.engineered_features)"),
           ModelSpec(RANDOM, "baseline", "baseline", False, "untrained F4 architecture, seeded (se.random_frozen)", encoder="conv")]
    for group, items in (("family", se.FAMILIES), ("ablation", se.ABLATIONS)):
        for e in items:
            out.append(ModelSpec(e.name, "encoder", group, True, _describe(e), e.encoder, e.temporal, e.align,
                                 f"mask {e.lambda_mask} · phys {e.lambda_phys} · temp {e.lambda_temp} · align {e.lambda_align}", e.name == se.PRIMARY))
    out.append(ModelSpec(TCN, "external", "baseline", False, "Milestone 06 fold-fitted TCN-B (Combine forty only), from tcn_representations.parquet"))
    return out


def resolve(name: str) -> str:
    n = ALIASES.get(name, name)
    known = {s.name for s in specs()}
    if n not in known:
        raise KeyError(f"unknown model {name!r}; available: {sorted(known)} (aliases: {sorted(ALIASES)})")
    return n


def spec(name: str) -> ModelSpec:
    n = resolve(name)
    return next(s for s in specs() if s.name == n)


def families() -> list[str]:
    return [s.name for s in specs() if s.group == "family"]


def ablations() -> list[str]:
    return [s.name for s in specs() if s.group == "ablation"]


def list_models() -> Any:
    import polars as pl

    return pl.DataFrame([asdict(s) for s in specs()])


def enc_config(name: str, science: Any) -> Any:
    """The scaled ``EncConfig`` used for training (``se.scale_config(se.by_name(name), M12Config)``)."""
    se = _backend.se()
    n = resolve(name)
    base = se.by_name(se.PRIMARY if n == RANDOM else n)
    return se.scale_config(base, science.to_m12config())


def get_model(name: str, science: Any = None) -> Any:
    """A freshly initialised ``SharedEncoder`` for an encoder name (torch required)."""
    from ..config import ScienceConfig
    from ..devices import require_torch

    require_torch("get_model")
    s = spec(name)
    if s.kind == "external" or s.name == ENGINEERED:
        raise ValueError(f"{s.name} is not a neural encoder")
    return _backend.se().SharedEncoder(enc_config(s.name, science or ScienceConfig()))


__all__ = ["ALIASES", "ENGINEERED", "ModelSpec", "RANDOM", "TCN", "ablations", "enc_config", "families", "get_model", "list_models", "resolve", "spec", "specs"]
