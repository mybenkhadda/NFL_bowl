"""Stage runners of the unified Milestone 12 notebook (warehouse tracks). 12F lives in ``nflbowl.research.encoder``.

    Foundation (12.0) · DistributionalCapacity (12C) · MeasurementError (12B) · FunctionalTransfer (12A) ·
    MultiviewTraits (12D) · OptimalTransport (12E) · Synthesis (12G)

Each class is constructed with ``(ctx, run, ucfg, sources, checks)``; its steps mirror the Parts of the source notebook.
"""

from __future__ import annotations

from .base import Stage, display_df, md, required_check_names, run_guarded, run_software_checks, step
from .s12_0 import Foundation
from .s12a import FunctionalTransfer
from .s12b import MeasurementError
from .s12c import DistributionalCapacity
from .s12d import MultiviewTraits
from .s12e import OptimalTransport
from .s12g import Synthesis

STAGE_CLASSES = {"12.0": Foundation, "12C": DistributionalCapacity, "12B": MeasurementError, "12A": FunctionalTransfer, "12D": MultiviewTraits,
                 "12E": OptimalTransport, "12G": Synthesis}

__all__ = ["DistributionalCapacity", "Foundation", "FunctionalTransfer", "MeasurementError", "MultiviewTraits", "OptimalTransport", "STAGE_CLASSES", "Stage",
           "Synthesis", "display_df", "md", "required_check_names", "run_guarded", "run_software_checks", "step"]
