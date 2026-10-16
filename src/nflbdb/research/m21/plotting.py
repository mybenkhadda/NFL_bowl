"""M2.1 figure helpers — the generic kit lives in :mod:`nflbdb.research.figkit` (re-exported here);
this module only adds the attempt-integrity class colours used by Workstream A."""

from __future__ import annotations

from ...data.reporting import CATEGORICAL, INK, STATUS
from ..figkit import SIZES, empty_panel, equal_aspect_panels, figure, finish  # noqa: F401  (re-export)

CLASS_COLORS = {  # attempt-integrity classes (Workstream A)
    "exact_duplicate": STATUS["critical"], "translated_copy": STATUS["critical"], "cross_athlete_copy": STATUS["critical"],
    "reused_sequence": STATUS["serious"], "time_overlap": STATUS["serious"], "near_duplicate": STATUS["warning"],
    "consistent": CATEGORICAL[0], "distinct": INK["muted"], "not_comparable": INK["axis"], "unpaired": INK["axis"]}
