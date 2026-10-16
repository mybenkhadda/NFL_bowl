"""Inputs and stage orchestration of the Combine → NFL get-off experiment (reads persisted data; rebuilds nothing).

    DEF   endpoint audit, endpoint histogram, valid get-off observations        (no Combine value used)
    TGT   athlete targets, context cells, outcome reliability      ← DEF       (no Combine value used)
    COH   Combine feature variants, research base, funnel, missingness, predictor redundancy   ← TGT
    MOD   folds, out-of-fold predictions, model comparison, paired bootstrap, coefficients, matched pairs  ← COH
    SENS  sensitivity analyses                                     ← COH

Gate: MOD and SENS relate Combine values to the NFL endpoint. They run only when the vendor definition in
``configs/player_get_off_definition.json`` is confirmed, or when the caller passes ``allow_unresolved=True``
(CLI ``--allow-unresolved-definition``). The definition status is part of their fingerprint and provenance, so a
comparison computed on an unresolved definition is always labelled as such and is recomputed once it is confirmed.

Nothing here scans tracking frames: the NFL side is one row per tracked player-play (the Phase 05 P0 base).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import polars as pl

from ...data.tables import LayerRefs
from ...data.utils import file_fingerprint, get_logger
from ..contracts import TableContract, cast_keys
from ..sprint_rescue import attempts as rescue_attempts
from ..sprint_rescue.store import RescueStore
from . import cohort as co
from . import endpoint as ep
from . import evaluate as ev
from . import targets as tg
from .config import NAMESPACE, GetOffConfig

log = get_logger(__name__)
GATED = ("MOD", "SENS")
STAGES: dict[str, tuple[tuple[TableContract, ...], tuple[str, ...]]] = {
    "DEF": ((ep.AUDIT, ep.HIST, ep.PLAYS), ()),
    "TGT": ((tg.TARGET_TABLE, tg.CELLS, tg.RELIABILITY), ("DEF",)),
    "COH": ((co.VARIANTS, co.BASE, co.FUNNEL, co.MISSING, co.CORRELATION), ("TGT",)),
    "MOD": ((ev.FOLDS, ev.OOF, ev.COMPARISON, ev.BOOT, ev.COEF, ev.MATCHED), ("COH",)),
    "SENS": ((ev.SENSITIVITY,), ("COH",)),
}
ALL_CONTRACTS = tuple(c for contracts, _ in STAGES.values() for c in contracts)


def smoke_config(gcfg: GetOffConfig) -> GetOffConfig:
    """Cheap settings for a wiring check. Results land in ``_sample`` and are not results."""
    return replace(gcfg, n_boot=200, n_boot_sensitivity=100, cv_repeats=3, n_game_splits=20, min_athletes_model=12)


def make_store(cfg: Any, sampled: bool = False) -> RescueStore:
    return RescueStore(cfg, sampled=sampled, namespace=NAMESPACE)


class Inputs:
    """Read-only access to the persisted upstream data."""

    def __init__(self, cfg: Any, gcfg: GetOffConfig):
        self.cfg, self.gcfg = cfg, gcfg
        self.refs = LayerRefs(cfg)
        self.rescue = RescueStore(cfg)                     # sprint rescue outputs (full run, never the sample)

    def definition(self) -> dict[str, Any]:
        return ep.load_definition(self.cfg)

    def catalog(self) -> dict[str, str]:
        p = self.cfg.paths.tables / "schema_catalog.csv"
        if not p.is_file():
            return {}
        cat = pl.read_csv(p, infer_schema_length=0).filter((pl.col("table") == "player_play")
                                                           & (pl.col("column") == self.gcfg.endpoint))
        return {k: (cat[k][0] or "") for k in ("meaning", "unit", "evidence")} if cat.height else {}

    def nfl_base(self) -> pl.DataFrame:
        """Tracked player-plays, assembled exactly as the Phase 05 P0 census does (no frame scan)."""
        from ..phase05 import decision_audit as da
        from ..phase05.config import Phase05Config

        aud = da.DecisionAuditStudy(self.cfg, Phase05Config())
        aud.roles = {"state": (*ep.PLAYER_FIELDS, *ep.PLAY_FIELDS), "action": (), "outcome": (self.gcfg.endpoint,)}
        inp = aud.inputs()
        fields = [f for r in da.ROLES for f in aud.roles[r]]
        base, _ = da.assemble_base(inp["fact"], inp["player_play"], inp["plays"], inp["games"], inp["athletes"], fields)
        return base

    def attempts(self) -> pl.DataFrame:
        c = rescue_attempts.ATTEMPTS
        if not self.rescue.exists(c):
            raise FileNotFoundError(f"{self.cfg.paths.rel(self.rescue.parquet(c))} missing — run "
                                    "`python -m nflbdb.research.sprint_rescue --only ATT`")
        return self.rescue.read(c)

    def mart(self) -> pl.DataFrame:
        return cast_keys(self.refs.scan("gold", "mart_athlete_research_base").collect())

    def upstream_files(self, stage: str) -> list:
        if stage == "DEF":
            return [f for layer, n in (("gold", "fact_nfl_play_participation"), ("silver", "player_play"), ("silver", "plays"),
                                       ("silver", "games"), ("gold", "dim_athlete")) for f in self.refs.files(layer, n)]
        if stage == "COH":
            return [self.rescue.parquet(rescue_attempts.ATTEMPTS), *self.refs.files("gold", "mart_athlete_research_base")]
        return []

    def upstream_state(self, stage: str) -> dict[str, Any]:
        return {self.cfg.paths.rel(f): (file_fingerprint(f) if f.is_file() else "missing") for f in self.upstream_files(stage)}


# ============================================================================== stage computations
def compute_def(inp: Inputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    pop = ep.dl_population(inp.nfl_base(), inp.gcfg)
    plays = ep.valid_plays(pop, inp.gcfg)
    return {ep.AUDIT.name: ep.endpoint_audit(pop, plays, inp.definition(), inp.catalog(), inp.gcfg),
            ep.HIST.name: ep.endpoint_histogram(plays), ep.PLAYS.name: plays}


def compute_tgt(inp: Inputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    plays = store.read(ep.PLAYS)
    return {tg.TARGET_TABLE.name: tg.athlete_targets(plays, inp.gcfg), tg.CELLS.name: tg.context_cells(plays, inp.gcfg),
            tg.RELIABILITY.name: tg.reliability_table(plays, inp.gcfg)}


def compute_coh(inp: Inputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    mart = inp.mart()
    variants = co.combine_variants(inp.attempts(), inp.gcfg)
    base = co.research_base(mart, variants, store.read(tg.TARGET_TABLE), inp.gcfg)
    return {co.VARIANTS.name: variants, co.BASE.name: base, co.FUNNEL.name: co.cohort_funnel(mart, base, inp.gcfg),
            co.MISSING.name: co.cohort_missingness(base, inp.gcfg), co.CORRELATION.name: co.predictor_correlations(base)}


def _model_inputs(store: RescueStore) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return store.read(co.BASE), store.read(co.VARIANTS), store.read(tg.TARGET_TABLE), store.read(tg.CELLS)


def compute_mod(inp: Inputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    return ev.compute_models(*_model_inputs(store), inp.gcfg)


def compute_sens(inp: Inputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    return {ev.SENSITIVITY.name: ev.compute_sensitivity(*_model_inputs(store), inp.gcfg)}


COMPUTE: dict[str, Callable[[Inputs, RescueStore], dict[str, pl.DataFrame]]] = {
    "DEF": compute_def, "TGT": compute_tgt, "COH": compute_coh, "MOD": compute_mod, "SENS": compute_sens}


def stage_fingerprint(stage: str, inp: Inputs, store: RescueStore) -> str:
    deps = {d: store.recorded(d) for d in STAGES[stage][1]}
    params: dict[str, Any] = {"config": inp.gcfg.params()}
    if stage in ("DEF", *GATED):
        params["definition"] = inp.definition()
    return store.fingerprint(stage, params, {"files": inp.upstream_state(stage), "stages": deps})


def run_stage(stage: str, inp: Inputs, store: RescueStore, force: bool = False, allow_unresolved: bool = False) -> str:
    """Returns 'computed', 'reused' or 'blocked'."""
    contracts, deps = STAGES[stage]
    definition = inp.definition()
    confirmed, _ = ep.definition_status(definition)
    if stage in GATED and not confirmed and not allow_unresolved:
        log.warning("[%s:%s] NOT RUN. %s Pass --allow-unresolved-definition to compute the statistical comparison "
                    "anyway; it will be labelled 'definition unresolved' and carries no football interpretation.",
                    NAMESPACE, stage, ep.blocker_text(definition))
        return "blocked"
    missing = [d for d in deps if store.recorded(d) is None]
    if missing:
        raise RuntimeError(f"stage {stage} needs completed stage(s) {missing}: run "
                           f"`python -m nflbdb.research.combine_nfl_dl --only {','.join([*missing, stage])}`")
    fp = stage_fingerprint(stage, inp, store)
    current, reason = store.is_current(stage, fp, contracts)
    if current and not force:
        log.info("[%s:%s] reuse (%s)", NAMESPACE, stage, reason)
        return "reused"
    log.info("[%s:%s] compute (%s%s)", NAMESPACE, stage, "forced; " if force else "", reason)
    out = COMPUTE[stage](inp, store)
    provenance = {"inputs": list(inp.upstream_state(stage)), "depends_on": list(deps), "config": inp.gcfg.params(),
                  "experiment_version": inp.gcfg.version,
                  "definition_status": "confirmed" if confirmed else "unresolved"}
    for c in contracts:
        store.write(out[c.name], c, stage, provenance)
    store.record(stage, fp, contracts)          # recorded only after EVERY output passed its contract
    return "computed"
