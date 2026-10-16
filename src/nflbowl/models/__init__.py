"""Model registry over the 12F encoder families, ablations and baselines.

    from nflbowl.models import list_models, get_model
    list_models()                  # polars table
    model = get_model("F4")        # SharedEncoder (PyTorch), scaled to the default science config
"""

from .registry import ENGINEERED, RANDOM, TCN, ablations, enc_config, families, get_model, list_models, resolve, spec

__all__ = ["ENGINEERED", "RANDOM", "TCN", "ablations", "enc_config", "families", "get_model", "list_models", "resolve", "spec"]
