"""E16.A3 — reliability, exposure dependence, context (demand-kind) share and the immutable freeze of the NFL targets.

Every task target is rebuilt on two disjoint halves of an athlete's NFL data with the SAME aggregation (thresholds halved):
* split ``season`` (alternate seasons) when ≥ 2 seasons exist, else ``game`` when a game key is available, else ``random``
  (seeded per window — weaker: windows of one game can fall in both halves; labelled in every row);
* event targets (T3–T5) split by the window that contains the event, so an event and its window stay together.

Read-outs per target (athletes with both halves, values residualised on the position means of the athletes used):
``r_half`` (Pearson), ``reliability`` = Spearman–Brown 2r/(1+r) with an athlete-bootstrap 95 % CI; exposure curve
(k windows / events per half); ``rho_exposure`` = Spearman of the position-residualised target with log exposure;
``context_share`` = within-position share of the window / event quantity's variance explained by demand kind.

G16-A rule per target (gates document): reliability ≥ 0.7 with lower CI ≥ 0.5; ≥ 20 athletes in ≥ 3 positions;
|rho_exposure| ≤ 0.3; context_share ≤ 0.25 OR the target is demand-mix adjusted by construction (T1, T6). T0 is the
12C-frozen target: it is not rebuilt here (its reliability lives upstream), so its row is NOT_EVALUATED, never PASS.
``freeze`` writes ``manifests/target_freeze_m16.json`` once (never overwritten; a differing later run is reported).
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from ...artifacts.store import atomic_json
from ...utils.reproducibility import utc_now
from ..status import FAIL, NOT_EVALUATED, PASS
from .targets import TARGETS, TargetConfig, event_targets, window_targets

NAN = float("nan")
WINDOW_TARGETS = ("T1", "T2", "T2b", "T6")
EVENT_TARGETS = ("T3", "T4", "T5")
MIX_ADJUSTED = ("T1", "T6")
EVENT_QUANTITY = {"T3": "max_controlled_decel", "T4": "first_step_explosiveness", "T5": "turn_speed_retention"}
WINDOW_QUANTITY = {"T1": "max_speed", "T6": "max_speed", "T2": "accel_band_0", "T2b": "accel_band_1"}


def half_labels(windows: Any, split: str, seed: int, game_key: Any | None = None) -> tuple[Any, str]:
    """Window → half (0 / 1). Returns (frame idx, half; the split actually used)."""
    import polars as pl

    w = windows.select("idx", "nfl_id", *([c for c in ("season",) if c in windows.columns]))
    if split == "season" and "season" in w.columns and w["season"].drop_nulls().n_unique() >= 2:
        rank = {s: i for i, s in enumerate(sorted(w["season"].drop_nulls().unique().to_list()))}
        return w.select("idx", (pl.col("season").replace_strict(rank, default=0, return_dtype=pl.Int64) % 2).alias("half")), "season"
    if split in ("season", "game") and game_key is not None:
        g = w.join(game_key, on="idx", how="left")
        gm = g.select("nfl_id", "game_key").unique().sort("nfl_id", "game_key").with_columns((pl.int_range(pl.len()).over("nfl_id") % 2).alias("half"))
        return g.join(gm, on=["nfl_id", "game_key"], how="left").select("idx", pl.col("half").fill_null(0)), "game"
    rng = np.random.default_rng(seed)
    return w.select("idx").with_columns(pl.Series("half", rng.integers(0, 2, w.height))), "random"


def _halved(tcfg: TargetConfig) -> TargetConfig:
    return TargetConfig(min_windows=max(2, tcfg.min_windows // 2), min_windows_kind=max(2, tcfg.min_windows_kind // 2),
                        min_events=max(2, tcfg.min_events // 2), band_min_frames=tcfg.band_min_frames, bands=tcfg.bands, q=tcfg.q)


def _pos_resid(v: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.full(v.size, NAN)
    for p in np.unique(pos):
        m = (pos == p) & np.isfinite(v)
        if m.sum() >= 3:
            out[m] = v[m] - v[m].mean()
    return out


def _sb(r: float) -> float:
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else NAN


def split_half(a: np.ndarray, b: np.ndarray, pos: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    """Position-residualised half correlation, Spearman–Brown reliability and its athlete-bootstrap CI."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, pos = a[ok], b[ok], pos[ok]
    ra, rb = _pos_resid(a, pos), _pos_resid(b, pos)
    m = np.isfinite(ra) & np.isfinite(rb)
    n = int(m.sum())
    if n < 20:
        return {"n_athletes": n, "r_half": NAN, "reliability": NAN, "rel_lo": NAN, "rel_hi": NAN}
    ra, rb = ra[m], rb[m]
    r = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if ra[i].std() > 0 and rb[i].std() > 0:
            bs.append(_sb(float(np.corrcoef(ra[i], rb[i])[0, 1])))
    bs = np.asarray([x for x in bs if np.isfinite(x)])
    return {"n_athletes": n, "r_half": r, "reliability": _sb(r), "rel_lo": float(np.quantile(bs, 0.025)) if bs.size >= 20 else NAN,
            "rel_hi": float(np.quantile(bs, 0.975)) if bs.size >= 20 else NAN}


def half_targets(windows: Any, events: Any, halves: Any, tcfg: TargetConfig) -> dict[int, Any]:
    """Targets recomputed on each half (thresholds halved)."""
    import polars as pl

    h = _halved(tcfg)
    out = {}
    for k in (0, 1):
        ids = halves.filter(pl.col("half") == k)["idx"]
        w = windows.filter(pl.col("idx").is_in(ids.to_list()))
        wt = window_targets(w, h)
        ev = events.filter(pl.col("idx").is_in(ids.to_list())) if events is not None and events.height else events
        et = event_targets(ev, h) if ev is not None and ev.height else None
        out[k] = wt.join(et, on="nfl_id", how="full", coalesce=True) if et is not None and et.width > 1 else wt
    return out


def reliability_table(windows: Any, events: Any, positions: Any, split: str, seed: int, n_boot: int, tcfg: TargetConfig,
                      game_key: Any | None = None) -> Any:
    import polars as pl

    halves, used = half_labels(windows, split, seed, game_key)
    hs = half_targets(windows, events, halves, tcfg)
    pos = positions.select(pl.col("nfl_id").cast(pl.Int64), "position_group")
    rows = []
    for k, tid in enumerate((*WINDOW_TARGETS, *EVENT_TARGETS)):
        col = TARGETS[tid].column
        if col not in hs[0].columns or col not in hs[1].columns:
            rows.append({"target_id": tid, "split": used, "n_athletes": 0, "note": "target not computable on halves"})
            continue
        j = hs[0].select(pl.col("nfl_id").cast(pl.Int64), pl.col(col).alias("a")).join(hs[1].select(pl.col("nfl_id").cast(pl.Int64), pl.col(col).alias("b")),
                                                                                     on="nfl_id").join(pos, on="nfl_id", how="left")
        r = split_half(j["a"].cast(pl.Float64).fill_null(NAN).to_numpy(), j["b"].cast(pl.Float64).fill_null(NAN).to_numpy(),
                       j["position_group"].fill_null("?").to_numpy(), n_boot, seed + 17 * k)
        per_pos = j.drop_nulls().group_by("position_group").len()
        rows.append({"target_id": tid, "column": col, "split": used, **r, "positions_ge20": int(per_pos.filter(pl.col("len") >= 20).height), "note": ""})
    return pl.DataFrame(rows, infer_schema_length=None)


def exposure_curve(windows: Any, events: Any, positions: Any, seed: int, grid_windows: Sequence[int] = (5, 10, 20, 40),
                   grid_events: Sequence[int] = (2, 3, 5, 8)) -> Any:
    """Reliability when each half holds exactly k windows (window targets) or k events (event targets), athletes with ≥ 2k."""
    import polars as pl

    rng = np.random.default_rng(seed)
    pos = dict(zip(positions["nfl_id"].cast(pl.Int64).to_list(), positions["position_group"].to_list(), strict=True))
    rows = []
    for tid in (*WINDOW_TARGETS, *EVENT_TARGETS):
        src = windows if tid in WINDOW_TARGETS else events
        qty = WINDOW_QUANTITY.get(tid) or EVENT_QUANTITY[tid]
        if src is None or qty not in src.columns:
            continue
        d = src.select(pl.col("nfl_id").cast(pl.Int64), pl.col(qty).cast(pl.Float64)).filter(pl.col(qty).is_finite())
        if tid == "T3" and "complete" in src.columns:
            d = src.filter(pl.col("complete")).select(pl.col("nfl_id").cast(pl.Int64), pl.col(qty).cast(pl.Float64)).filter(pl.col(qty).is_finite())
        stat = (lambda v: np.std(v, ddof=1)) if tid == "T6" else ((lambda v: np.quantile(v, 0.9)) if tid in ("T1", "T2", "T2b", "T3") else np.median)
        groups = {int(k[0]): g[qty].to_numpy() for k, g in d.group_by("nfl_id")}
        for k in (grid_windows if tid in WINDOW_TARGETS else grid_events):
            a, b, p = [], [], []
            for i, v in groups.items():
                if v.size >= 2 * k and (tid != "T6" or k >= 2):
                    sel = rng.permutation(v.size)
                    a.append(stat(v[sel[:k]]))
                    b.append(stat(v[sel[k:2 * k]]))
                    p.append(pos.get(i, "?"))
            r = split_half(np.asarray(a, float), np.asarray(b, float), np.asarray(p), 200, seed + k)
            rows.append({"target_id": tid, "k_per_half": int(k), **r})
    return pl.DataFrame(rows, infer_schema_length=None)


def exposure_dependence(targets: Any) -> Any:
    """Spearman of each position-residualised target with log(exposure) (windows for T1/T2/T6, events for T3–T5)."""
    import polars as pl

    from ..stats import spearman

    expo = {"T1": "n_windows", "T6": "n_windows", "T2": "n_band_0", "T2b": "n_band_1", "T3": "n_events_t3", "T4": "n_events_t4", "T5": "n_events_t5"}
    rows = []
    for tid, ecol in expo.items():
        col = TARGETS[tid].column
        if col not in targets.columns or ecol not in targets.columns:
            continue
        d = targets.select(pl.col(col).cast(pl.Float64), pl.col(ecol).cast(pl.Float64), "position_group").drop_nulls().filter(pl.col(col).is_finite() & (pl.col(ecol) > 0))
        v = _pos_resid(d[col].to_numpy(), d["position_group"].to_numpy())
        ok = np.isfinite(v)
        rows.append({"target_id": tid, "exposure": ecol, "n": int(ok.sum()), "rho_exposure": spearman(v[ok], np.log(d[ecol].to_numpy()[ok])) if ok.sum() >= 20 else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


def context_share(windows: Any, events: Any) -> Any:
    """Within-position share of variance of the underlying window / event quantity explained by demand kind (η²)."""
    import polars as pl

    rows = []
    for tid in (*WINDOW_TARGETS, *EVENT_TARGETS):
        src = windows if tid in WINDOW_TARGETS else events
        qty = WINDOW_QUANTITY.get(tid) or EVENT_QUANTITY[tid]
        if src is None or qty not in src.columns or "kind" not in src.columns:
            continue
        d = src.select("position_group", "kind", pl.col(qty).cast(pl.Float64)).filter(pl.col(qty).is_finite())
        ss_b = ss_t = 0.0
        for _, g in d.group_by("position_group"):
            v = g[qty].to_numpy()
            mu = v.mean()
            ss_t += float(((v - mu) ** 2).sum())
            km = g.group_by("kind").agg(pl.col(qty).mean().alias("m"), pl.len().alias("n"))
            ss_b += float((km["n"].to_numpy() * (km["m"].to_numpy() - mu) ** 2).sum())
        rows.append({"target_id": tid, "quantity": qty, "n_obs": d.height, "context_share": ss_b / ss_t if ss_t > 0 else NAN,
                     "mix_adjusted": tid in MIX_ADJUSTED})
    return pl.DataFrame(rows, infer_schema_length=None)


def gate_a(rel: Any, expo: Any, ctx: Any) -> Any:
    """G16-A per target from the computed tables (module docstring). T0: NOT_EVALUATED (frozen upstream)."""
    import polars as pl

    e = {r["target_id"]: r for r in expo.iter_rows(named=True)} if expo is not None and expo.height else {}
    c = {r["target_id"]: r for r in ctx.iter_rows(named=True)} if ctx is not None and ctx.height else {}
    rows = [{"target_id": "T0", "status": NOT_EVALUATED, "reliability": None, "rel_lo": None, "rho_exposure": None, "context_share": None,
             "reason": "12C frozen target: reliability established upstream, not rebuildable from the bundle"}]
    for r in rel.iter_rows(named=True):
        tid = r["target_id"]
        rho = e.get(tid, {}).get("rho_exposure")
        cs = c.get(tid, {}).get("context_share")
        checks = {"reliability ≥ 0.7": r.get("reliability") is not None and np.isfinite(r["reliability"]) and r["reliability"] >= 0.7,
                  "lower CI ≥ 0.5": r.get("rel_lo") is not None and np.isfinite(r["rel_lo"]) and r["rel_lo"] >= 0.5,
                  "≥ 3 positions with ≥ 20 athletes": (r.get("positions_ge20") or 0) >= 3,
                  "|rho_exposure| ≤ 0.3": rho is not None and np.isfinite(rho) and abs(rho) <= 0.3,
                  "context ≤ 0.25 or mix-adjusted": tid in MIX_ADJUSTED or (cs is not None and np.isfinite(cs) and cs <= 0.25)}
        computed = r.get("reliability") is not None and np.isfinite(r.get("reliability") or NAN)
        status = (PASS if all(checks.values()) else FAIL) if computed else NOT_EVALUATED
        rows.append({"target_id": tid, "status": status, "reliability": r.get("reliability"), "rel_lo": r.get("rel_lo"), "rho_exposure": rho,
                     "context_share": cs, "reason": "; ".join(k for k, ok in checks.items() if not ok) if computed else "not computable"})
    return pl.DataFrame(rows, infer_schema_length=None)


def freeze(run: Any, gate: Any, definitions: dict[str, Any], primary: str = "T0") -> dict[str, Any]:
    """Write ``manifests/target_freeze_m16.json`` ONCE. A later run with a different decision does not overwrite it: the
    returned record carries ``conflict = True`` and both decisions."""
    rec = {"written_at": utc_now(), "primary": primary, "eligible_task_targets": [r["target_id"] for r in gate.iter_rows(named=True)
                                                                                 if r["status"] == PASS and r["target_id"] != "T0"],
           "gate_a": gate.to_dicts(), "definitions": definitions}
    p = run.root / "manifests" / "target_freeze_m16.json"
    if p.is_file():
        old = json.loads(p.read_text(encoding="utf-8"))
        same = old.get("eligible_task_targets") == rec["eligible_task_targets"] and old.get("primary") == rec["primary"]
        return {**old, "conflict": not same, "this_run": rec if not same else None}
    atomic_json(rec, p, overwrite=False)
    return {**rec, "conflict": False}


__all__ = ["EVENT_TARGETS", "MIX_ADJUSTED", "WINDOW_TARGETS", "context_share", "exposure_curve", "exposure_dependence", "freeze", "gate_a", "half_labels",
           "half_targets", "reliability_table", "split_half"]
