"""E15.00 — the M12 sequence-store adapter: open, validate and index the portable bundle for M15–M18.

Reuses (never re-implements):
* ``nflbowl.data.loading.open_bundle`` — staging, size / sha256 checks, memory-mapping, and the schema validators
  ``validate_meta`` / ``validate_arrays`` / ``validate_folds`` / ``validate_athletes``;
* ``nflbowl.data.splits.fold_map`` — the official athlete → outer-fold map (manifest fold, else ``nfl_id % outer_folds``);
* ``nflbowl.lab.m14.events.physical`` — scaled channels → physical units with invalid frames as NaN.

Added here (the M15 data contract):
* ``audit_bundle`` — one row per check (status pass / warn / fail, observed value, expected value, detail): fingerprint,
  channel definitions and recorded scaling, shapes, row order, athlete identifiers, domain labels, mask validity (masked
  frames must be zero-filled, valid frames finite), physical plausibility on valid frames, heading-validity convention,
  fold completeness / range / athlete-disjointness, linked athletes per fold, athletes-table coverage;
* ``sequence_index`` — per sequence: fold, valid frames, gap-free segment count, leading / trailing invalid frames;
* ``fold_assignments`` — per athlete: official fold and its source (manifest or fallback), domains, linkage;
* ``counts_tables`` — the E15.00 counts (domain × kind × position, trials per Combine drill, linked cohort);
* ``game_key`` — a heuristic NFL game key parsed from ``seq_key`` (None + reason when it cannot be established).

Memory: ``X`` / ``M`` stay memory-mapped; every scan runs over ``cfg.chunk_rows`` rows at a time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ...artifacts.store import atomic_json
from ...data.validation import CHANNELS, DOMAINS, M1_COLS, TRANSFER_TARGET, SchemaError
from ...utils.reproducibility import utc_now
from ..status import BLOCKED_INSUFFICIENT_SAMPLE, BLOCKED_MISSING_INPUT, ResearchBlocked, ValidationFailed
from .config import M15Config

SPEED, ATAN, AN, YAW, DT, HV = range(6)
GAME_KEY_RE = re.compile(r"(?<!\d)(20\d{8})(?!\d)")          # NFL gameId: 10 digits YYYYMMDDgg


@dataclass
class M15Data:
    """What the later M15 steps need: the bundle, its audit, the athlete fold table and the sequence index."""

    bundle: Any
    audit: Any
    folds: Any
    index: Any
    fold_of: dict[int, int]
    fingerprint: str
    game_key_status: dict[str, Any] = field(default_factory=dict)

    @property
    def meta(self) -> Any:
        return self.bundle.meta

    def rows(self, domain: str | None = None, folds: tuple[int, ...] | None = None, role: str | None = None, held_out: int | None = None) -> np.ndarray:
        """Sequence row indices filtered by domain and / or outer fold. ``role='train'|'held_out'`` with ``held_out=f``
        selects the athlete-disjoint training / held-out rows of fold ``f`` (never both)."""
        idx = self.index
        m = np.ones(idx.height, bool)
        if domain is not None:
            m &= idx["domain"].to_numpy() == domain
        fo = idx["fold"].to_numpy()
        if folds is not None:
            m &= np.isin(fo, np.asarray(folds))
        if role is not None:
            if held_out is None:
                raise ValueError("role needs held_out=<fold>")
            m &= (fo == held_out) if role == "held_out" else (fo != held_out)
        return np.flatnonzero(m)


# ====================================================================== opening
def open_m15_bundle(cfg: M15Config) -> Any:
    """``open_bundle`` with the M15 configuration. A missing bundle → BLOCKED_MISSING_INPUT (with the loader's hint); a
    schema violation → FAILED_VALIDATION (with the violated rule)."""
    from ...data.loading import open_bundle

    try:
        return open_bundle(cfg.bundle_dir, cfg.stage_dir or None, verify=cfg.verify_bundle_hashes, outer_folds=cfg.outer_folds)
    except FileNotFoundError as e:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, str(e), missing=[str(Path(cfg.bundle_dir) / "bundle.json")]) from e
    except SchemaError as e:
        raise ValidationFailed(f"bundle schema violation: {e}", {"bundle_dir": cfg.bundle_dir}) from e


def chunks(n: int, size: int) -> Iterator[tuple[int, int]]:
    for s in range(0, n, size):
        yield s, min(n, s + size)


def physical_rows(bundle: Any, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Physical-unit arrays of ``rows`` (bounded slice of the memory map; invalid frames NaN) — ``m14.events.physical``."""
    from ..m14.events import physical

    rows = np.asarray(rows, np.int64)
    return physical(np.asarray(bundle.X[rows]), np.asarray(bundle.M[rows]))


# ====================================================================== indices
def fold_assignments(bundle: Any, cfg: M15Config) -> Any:
    """One row per athlete in ``meta``: nfl_id, fold, fold_source (manifest | fallback_mod), position_group,
    n_combine, n_nfl, has_combine, has_nfl, linked (athletes table, null when absent), selected (fold in cfg.folds)."""
    import polars as pl

    from ...data.splits import fold_map

    fmap = fold_map(bundle.meta, bundle.folds, cfg.outer_folds)
    manifest = set(int(i) for i in bundle.folds.filter(pl.col("fold").is_not_null())["nfl_id"].to_list())
    per = bundle.meta.group_by("nfl_id").agg(pl.col("position_group").first(), (pl.col("domain") == "combine").sum().alias("n_combine"),
                                             (pl.col("domain") == "nfl").sum().alias("n_nfl")).sort("nfl_id")
    ids = [int(i) for i in per["nfl_id"].to_list()]
    out = per.with_columns(pl.Series("fold", [int(fmap[i]) for i in ids], dtype=pl.Int64),
                           pl.Series("fold_source", ["manifest" if i in manifest else "fallback_mod" for i in ids], dtype=pl.String),
                           (pl.col("n_combine") > 0).alias("has_combine"), (pl.col("n_nfl") > 0).alias("has_nfl"))
    if bundle.athletes is not None and "linked" in bundle.athletes.columns:
        out = out.join(bundle.athletes.select(pl.col("nfl_id").cast(pl.Int64), pl.col("linked").cast(pl.Boolean)), on="nfl_id", how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Boolean).alias("linked"))
    sel = set(cfg.selected_folds())
    return out.with_columns(pl.col("fold").is_in(list(sel)).alias("selected")).select(
        "nfl_id", "fold", "fold_source", "position_group", "n_combine", "n_nfl", "has_combine", "has_nfl", "linked", "selected")


def _segment_stats(M: np.ndarray, dt: np.ndarray, max_gap_dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised over rows: (valid frames, segment count, leading invalid, trailing invalid). A segment starts at a valid
    frame whose predecessor is invalid or whose predecessor's duration exceeds ``max_gap_dt`` (``m14.events.segments`` rule;
    the minimum-length filter is applied later by the detectors, not here)."""
    v = M.astype(bool)
    n_valid = v.sum(1)
    prev_ok = np.zeros_like(v)
    prev_ok[:, 1:] = v[:, :-1] & ~(dt[:, :-1] > max_gap_dt)
    n_seg = (v & ~prev_ok).sum(1)
    any_v = v.any(1)
    lead = np.where(any_v, v.argmax(1), v.shape[1])
    trail = np.where(any_v, v[:, ::-1].argmax(1), v.shape[1])
    return n_valid, n_seg, lead, trail


def sequence_index(bundle: Any, fold_of: dict[int, int], cfg: M15Config) -> Any:
    """Per sequence: idx, seq_key, domain, nfl_id, kind, position_group, season, fold, n_valid, n_segments,
    lead_invalid, trail_invalid, valid_s (Σ dt over valid frames, seconds)."""
    import polars as pl

    n = bundle.n_sequences
    nv, ns, ld, tr, vs = (np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.float64))
    sc_dt = cfg.scale[DT]
    for s, e in chunks(n, cfg.chunk_rows):
        Mc = np.asarray(bundle.M[s:e], bool)
        dt = np.asarray(bundle.X[s:e, :, DT], np.float64) * sc_dt
        nv[s:e], ns[s:e], ld[s:e], tr[s:e] = _segment_stats(Mc, dt, cfg.max_gap_dt)
        vs[s:e] = np.where(Mc, np.where(np.isfinite(dt) & (dt > 0), dt, 0.1), 0.0).sum(1)
    meta = bundle.meta
    keep = [c for c in ("idx", "seq_key", "domain", "nfl_id", "kind", "position_group", "season") if c in meta.columns]
    ids = meta["nfl_id"].to_list()
    return meta.select(keep).with_columns(pl.Series("fold", [int(fold_of[int(i)]) for i in ids], dtype=pl.Int64), pl.Series("n_valid", nv),
                                          pl.Series("n_segments", ns), pl.Series("lead_invalid", ld), pl.Series("trail_invalid", tr),
                                          pl.Series("valid_s", vs))


# ====================================================================== audit
def _row(check: str, status: str, observed: Any = None, expected: Any = None, detail: str = "") -> dict[str, Any]:
    return {"check": check, "status": status, "observed": None if observed is None else str(observed),
            "expected": None if expected is None else str(expected), "detail": detail}


def _mask_and_range_scan(bundle: Any, cfg: M15Config) -> dict[str, Any]:
    """One chunked pass: masked-frame fill, non-finite valid frames, plausibility shares, heading convention."""
    n = bundle.n_sequences
    tot = {"valid": 0, "masked": 0, "masked_nonzero": 0, "valid_nonfinite": 0, "speed_out": 0, "atan_out": 0, "dt_out": 0, "hv_not_binary": 0,
           "yaw_nonzero_hv0": 0, "hv0_valid": 0, "empty_rows": 0, "speed_max": -np.inf, "atan_min": np.inf, "atan_max": -np.inf}
    sc = np.asarray(cfg.scale, np.float64)
    lo_s, hi_s = cfg.plausible_speed
    lo_a, hi_a = cfg.plausible_atan
    lo_d, hi_d = cfg.plausible_dt
    for s, e in chunks(n, cfg.chunk_rows):
        x = np.asarray(bundle.X[s:e], np.float64)
        m = np.asarray(bundle.M[s:e], bool)
        tot["empty_rows"] += int((~m.any(1)).sum())
        tot["valid"] += int(m.sum())
        tot["masked"] += int((~m).sum())
        tot["masked_nonzero"] += int((np.abs(x[~m]) > 0).any(axis=-1).sum())
        xv = x[m]
        tot["valid_nonfinite"] += int((~np.isfinite(xv)).any(axis=-1).sum())
        phys = xv * sc
        sp, at, dt, hv, yw = phys[:, SPEED], phys[:, ATAN], phys[:, DT], xv[:, HV], phys[:, YAW]
        tot["speed_out"] += int(((sp < lo_s) | (sp > hi_s)).sum())
        tot["atan_out"] += int(((at < lo_a) | (at > hi_a)).sum())
        tot["dt_out"] += int(((dt <= lo_d) | (dt > hi_d)).sum())
        tot["hv_not_binary"] += int((~np.isin(hv, (0.0, 1.0))).sum())
        hv0 = hv == 0.0
        tot["hv0_valid"] += int(hv0.sum())
        tot["yaw_nonzero_hv0"] += int((hv0 & (yw != 0.0)).sum())
        if sp.size:
            tot["speed_max"] = max(tot["speed_max"], float(np.nanmax(sp)))
            tot["atan_min"] = min(tot["atan_min"], float(np.nanmin(at)))
            tot["atan_max"] = max(tot["atan_max"], float(np.nanmax(at)))
    return tot


def audit_bundle(bundle: Any, cfg: M15Config, fa: Any | None = None) -> Any:
    """The E15.00 data-contract audit (one row per check). Statuses: ``pass``; ``warn`` (reported, not blocking —
    e.g. counts differing from the [EXP] brief); ``fail`` (contract violation). Use ``assert_contract`` to raise."""
    import polars as pl

    rows: list[dict[str, Any]] = []
    fa = fa if fa is not None else fold_assignments(bundle, cfg)
    fp = bundle.fingerprint
    rows.append(_row("bundle fingerprint", "pass" if (not cfg.expected_fingerprint or fp == cfg.expected_fingerprint) else "fail", fp,
                     cfg.expected_fingerprint or "(none set)", "M12–M14 were computed on this bundle; another fingerprint means other data"))
    ch = list(bundle.info.get("channels", []))
    rows.append(_row("channel definitions (bundle.json)", "pass" if ch == list(CHANNELS) else ("warn" if not ch else "fail"), ch, list(CHANNELS),
                     "absent in bundle.json → checked by shape only" if not ch else ""))
    rec_scale = bundle.info.get("scale")
    if rec_scale is None:
        rows.append(_row("physical scaling (bundle.json)", "warn", None, list(cfg.scale), "not recorded by the exporter; verified empirically by the plausibility checks below"))
    else:
        want = cfg.scale_map()
        got = rec_scale if isinstance(rec_scale, dict) else dict(zip(CHANNELS, rec_scale, strict=False))
        ok = all(abs(float(got.get(k, want[k])) - want[k]) < 1e-12 for k in want if k in got)
        rows.append(_row("physical scaling (bundle.json)", "pass" if ok else "fail", got, want))
    N, L, C = (int(s) for s in bundle.X.shape)
    rows.append(_row("array shape (N, L, C)", "pass" if (C == len(CHANNELS) and L == cfg.max_frames) else ("warn" if C == len(CHANNELS) else "fail"),
                     (N, L, C), (bundle.meta.height, cfg.max_frames, len(CHANNELS)), "L ≠ max_frames is reported, not fatal"))
    order_ok = bool(np.array_equal(bundle.meta["idx"].to_numpy(), np.arange(bundle.meta.height)))
    rows.append(_row("meta row order (idx = 0..N-1)", "pass" if order_ok else "fail", order_ok, True))
    rows.append(_row("athlete ids (integer, no nulls)", "pass" if (bundle.meta["nfl_id"].dtype.is_integer() and not bundle.meta["nfl_id"].null_count()) else "fail",
                     str(bundle.meta["nfl_id"].dtype), "integer"))
    doms = sorted(set(bundle.meta["domain"].unique().to_list()))
    rows.append(_row("domain labels", "pass" if set(doms) <= set(DOMAINS) and doms else "fail", doms, list(DOMAINS)))
    dom = bundle.meta["domain"].to_numpy()
    n_c, n_n = int((dom == "combine").sum()), int((dom == "nfl").sum())
    exp_n = cfg.extra.get("expected_counts", {"n_sequences": 77_125, "n_combine": 6_310, "n_nfl": 70_815})
    for k, v in (("n_sequences", N), ("n_combine", n_c), ("n_nfl", n_n)):
        if k in exp_n:
            rows.append(_row(f"count {k}", "pass" if int(exp_n[k]) == v else "warn", v, exp_n[k], "expected values come from the [EXP] brief / M13 EXPECTED"))
    scan = _mask_and_range_scan(bundle, cfg)
    rows.append(_row("masked frames carrying non-zero values", "pass" if scan["masked_nonzero"] == 0 else "warn", scan["masked_nonzero"], 0,
                     "padding is zero, but invalid frames inside an NFL window keep their recorded values (m11 _channels only maps "
                     "nan → 0): every loss / statistic must use M, never the raw array"))
    rows.append(_row("valid frames finite", "pass" if scan["valid_nonfinite"] == 0 else "fail", scan["valid_nonfinite"], 0))
    rows.append(_row("empty sequences (no valid frame)", "pass" if scan["empty_rows"] == 0 else "warn", scan["empty_rows"], 0))
    nv = max(scan["valid"], 1)
    for k, lim, what, can_fail in (("speed_out", cfg.plausible_speed, "speed yd/s", True), ("atan_out", cfg.plausible_atan, "a_tan yd/s²", True),
                                   ("dt_out", cfg.plausible_dt, "dt s", False)):
        share = scan[k] / nv
        status = "pass" if share <= 0.001 else ("warn" if (share <= 0.01 or not can_fail) else "fail")
        rows.append(_row(f"plausible {what} on valid frames", status, round(share, 6), f"share outside {lim} ≤ 0.001",
                         "a large share indicates a scaling error (wrong SCALE) rather than outliers" if can_fail else
                         "non-positive dt on a valid frame is replaced by the nominal 0.1 s (m14.events.physical); reported only"))
    rows.append(_row("speed max / a_tan range (physical)", "pass", f"{scan['speed_max']:.2f} / [{scan['atan_min']:.2f}, {scan['atan_max']:.2f}]", None, "descriptive"))
    rows.append(_row("heading_valid is 0/1", "pass" if scan["hv_not_binary"] == 0 else "fail", scan["hv_not_binary"], 0))
    rows.append(_row("yaw = 0 where heading invalid", "pass" if scan["yaw_nonzero_hv0"] == 0 else "warn", scan["yaw_nonzero_hv0"], 0,
                     "yaw on heading-invalid frames is not an observation and is excluded from every loss"))
    # ---- folds
    fv = fa["fold"].to_numpy()
    in_range = bool(((fv >= 0) & (fv < cfg.outer_folds)).all())
    rows.append(_row("fold range", "pass" if in_range else "fail", f"[{fv.min()}, {fv.max()}]" if fv.size else "empty", f"[0, {cfg.outer_folds - 1}]"))
    present = sorted(set(fv.tolist()))
    rows.append(_row("every outer fold populated", "pass" if present == list(range(cfg.outer_folds)) else "fail", present, list(range(cfg.outer_folds))))
    n_fb = int((fa["fold_source"] == "fallback_mod").sum())
    rows.append(_row("athletes without a manifest fold (fallback nfl_id % K)", "pass" if n_fb == 0 else "warn", n_fb, 0,
                     "fallback folds are deterministic (splits.fold_map) but not part of the 12.0 manifest"))
    fold_of = dict(zip(fa["nfl_id"].to_list(), fa["fold"].to_list(), strict=True))
    unmapped = [int(i) for i in bundle.meta["nfl_id"].unique().to_list() if int(i) not in fold_of]
    per_seq = bundle.meta.select("nfl_id", "domain").filter(pl.col("nfl_id").is_in(list(fold_of))).with_columns(
        pl.col("nfl_id").replace_strict(fold_of, return_dtype=pl.Int64).alias("fold"))
    split = per_seq.group_by("nfl_id").agg(pl.col("fold").n_unique().alias("k")).filter(pl.col("k") > 1).height
    rows.append(_row("athlete-disjoint folds (all sequences of an athlete, both domains, in one fold)", "pass" if (split == 0 and not unmapped) else "fail",
                     {"athletes_in_two_folds": split, "athletes_without_fold": len(unmapped)}, {"athletes_in_two_folds": 0, "athletes_without_fold": 0}))
    if bundle.athletes is not None:
        ath_ids = set(int(i) for i in bundle.athletes["nfl_id"].to_list())
        man = bundle.folds.filter(pl.col("fold").is_not_null())
        man_ids = set(int(i) for i in man["nfl_id"].to_list())
        rows.append(_row("athletes-table athletes covered by the fold manifest", "pass" if ath_ids <= man_ids else "warn", len(ath_ids - man_ids), 0,
                         "uncovered athletes receive the deterministic fallback fold"))
    linked = fa.filter(pl.col("has_combine") & pl.col("has_nfl"))
    per_fold = {int(f): int(linked.filter(pl.col("fold") == f).height) for f in range(cfg.outer_folds)}
    low = {f: n for f, n in per_fold.items() if n < cfg.min_linked_per_fold}
    rows.append(_row("athletes with both domains per fold", "pass" if not low else "warn", per_fold, f"≥ {cfg.min_linked_per_fold} per fold",
                     f"{BLOCKED_INSUFFICIENT_SAMPLE} for cross-domain steps in folds {sorted(low)}" if low else ""))
    # ---- athletes table
    if bundle.athletes is None:
        rows.append(_row("athletes.parquet present", "warn", False, True, "M1 / transfer steps (M16) will be BLOCKED_MISSING_INPUT"))
    else:
        a = bundle.athletes
        miss = [c for c in ("nfl_id", "position_group", "linked", *M1_COLS) if c not in a.columns]
        rows.append(_row("athletes.parquet M1 contract columns", "pass" if not miss else "fail", miss or "all present", list(M1_COLS)))
        rows.append(_row(f"athletes.parquet target {TRANSFER_TARGET}", "pass" if TRANSFER_TARGET in a.columns else "warn",
                         TRANSFER_TARGET in a.columns, True))
        ids_meta = set(int(i) for i in fa["nfl_id"].to_list())
        ids_ath = set(int(i) for i in a["nfl_id"].to_list())
        rows.append(_row("athletes in meta without an athletes row", "pass" if not (ids_meta - ids_ath) else "warn", len(ids_meta - ids_ath), 0))
        if "linked" in a.columns:
            n_link = int(a.filter(pl.col("linked").fill_null(False)).height)
            rows.append(_row("linked athletes (athletes.linked)", "pass" if n_link else "warn", n_link, cfg.extra.get("expected_linked", 399),
                             "brief value [EXP]; reconciliation only"))
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "observed": pl.String, "expected": pl.String, "detail": pl.String})


def assert_contract(audit: Any) -> None:
    """Raise ``ValidationFailed`` when any audit row failed (the audit table is the evidence)."""
    import polars as pl

    bad = audit.filter(pl.col("status") == "fail")
    if bad.height:
        raise ValidationFailed(f"{bad.height} data-contract check(s) failed: {bad['check'].to_list()}", {"failed": bad.to_dicts()})


# ====================================================================== counts (E15.00)
def counts_tables(index: Any, fa: Any) -> dict[str, Any]:
    """counts_by_domain_kind_position · trials_per_drill (Combine: athletes with ≥ 1 / ≥ 2 / ≥ 3 attempts per kind) ·
    linked_cohort (per position: athletes with Combine, NFL, both) · valid_frames_by_domain."""
    import polars as pl

    by = index.group_by("domain", "kind", "position_group").agg(pl.len().alias("n_sequences"), pl.col("nfl_id").n_unique().alias("n_athletes"),
                                                                 pl.col("n_valid").median().alias("median_valid_frames")).sort("domain", "kind", "position_group")
    per = index.filter(pl.col("domain") == "combine").group_by("kind", "nfl_id").len()
    trials = per.group_by("kind").agg(pl.col("nfl_id").n_unique().alias("athletes_ge1"), (pl.col("len") >= 2).sum().alias("athletes_ge2"),
                                      (pl.col("len") >= 3).sum().alias("athletes_ge3"), pl.col("len").median().alias("median_attempts")).sort("kind")
    linked = fa.group_by("position_group").agg(pl.col("has_combine").sum().alias("athletes_combine"), pl.col("has_nfl").sum().alias("athletes_nfl"),
                                               (pl.col("has_combine") & pl.col("has_nfl")).sum().alias("athletes_both"),
                                               pl.col("linked").fill_null(False).sum().alias("athletes_linked_flag")).sort("position_group")
    vf = index.group_by("domain").agg(pl.len().alias("n_sequences"), pl.col("n_valid").median().alias("median_valid"), pl.col("n_valid").quantile(0.25).alias("q25_valid"),
                                      pl.col("n_valid").quantile(0.75).alias("q75_valid"), (pl.col("n_segments") > 1).mean().alias("share_multi_segment"),
                                      pl.col("valid_s").median().alias("median_valid_s")).sort("domain")
    return {"counts_by_domain_kind_position": by, "trials_per_drill": trials, "linked_cohort": linked, "valid_frames_by_domain": vf}


# ====================================================================== NFL game key (heuristic)
def game_key(meta: Any, min_share: float = 0.95) -> tuple[Any | None, dict[str, Any]]:
    """Parse a 10-digit NFL gameId (``20YYMMDDgg``) from NFL ``seq_key``. Returns (Series aligned with NFL rows, status).
    The parse is accepted only when ≥ ``min_share`` of NFL keys yield exactly one id; otherwise (None, reason) — game-split
    reliability then falls back to season splits (documented in the experiment matrix, E15.A4 / E16.A3)."""
    import polars as pl

    nfl = meta.filter(pl.col("domain") == "nfl")
    keys = [str(k) for k in nfl["seq_key"].to_list()]
    if not keys:
        return None, {"status": "no NFL rows", "share": 0.0}
    found = [GAME_KEY_RE.findall(k) for k in keys]
    one = [f[0] if len(f) == 1 else None for f in found]
    share = sum(v is not None for v in one) / len(keys)
    status = {"status": "parsed" if share >= min_share else "not established", "share": round(share, 4), "examples": keys[:3],
              "rule": GAME_KEY_RE.pattern}
    if share < min_share:
        return None, status
    return pl.Series("game_key", one, dtype=pl.String), status


# ====================================================================== orchestration
def prepare_m15_data(cfg: M15Config, run: Any | None = None, strict: bool = True) -> M15Data:
    """Open → index → audit (→ raise on a failed contract when ``strict``) and persist the reusable outputs:
    ``manifests/fold_assignments.parquet``, ``manifests/dataset_fingerprint.json``, ``datasets/cached/sequence_index``
    and the E15.00 tables. Idempotent: the sequence index is reloaded when its provenance (bundle + config fingerprint)
    matches."""
    from ...data.splits import fold_map

    bundle = open_m15_bundle(cfg)
    fold_of = fold_map(bundle.meta, bundle.folds, cfg.outer_folds)
    fa = fold_assignments(bundle, cfg)
    prov = {"bundle_fingerprint": bundle.fingerprint, "config_fingerprint": cfg.fingerprint()}
    index = run.load_data("datasets/cached/sequence_index", prov) if (run is not None and cfg.reuse_artifacts) else None
    if index is None:
        index = sequence_index(bundle, fold_of, cfg)
        if run is not None:
            run.data_file(index, "datasets/cached/sequence_index", prov)
    audit = audit_bundle(bundle, cfg, fa)
    gk, gk_status = game_key(bundle.meta)
    if run is not None:
        run.table(audit, "data_contract_audit", "dataset")
        for name, df in counts_tables(index, fa).items():
            run.table(df, name, "dataset")
        run.data_file(fa, "manifests/fold_assignments", prov)
        atomic_json({"bundle_fingerprint": bundle.fingerprint, "bundle_root": str(bundle.root), "staged_from": str(bundle.staged_from) if bundle.staged_from else None,
                     "expected_fingerprint": cfg.expected_fingerprint, "config_fingerprint": cfg.fingerprint(), "summary": bundle.summary(),
                     "game_key": gk_status, "written_at": utc_now()}, run.root / "manifests" / "dataset_fingerprint.json", overwrite=True)
    if strict:
        assert_contract(audit)
    return M15Data(bundle=bundle, audit=audit, folds=fa, index=index, fold_of=fold_of, fingerprint=bundle.fingerprint, game_key_status=gk_status)


def athlete_weighted_rows(index: Any, per_athlete: int, seed: int, domain: str | None = None) -> np.ndarray:
    """At most ``per_athlete`` sequences per athlete (seeded), so NFL-heavy athletes do not dominate plots."""
    import polars as pl

    d = index if domain is None else index.filter(pl.col("domain") == domain)
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for _, g in d.group_by("nfl_id", maintain_order=True):
        idx = g["idx"].to_numpy()
        out.extend(idx if idx.size <= per_athlete else rng.choice(idx, per_athlete, replace=False))
    return np.sort(np.asarray(out, np.int64))


__all__ = ["M15Data", "assert_contract", "athlete_weighted_rows", "audit_bundle", "chunks", "counts_tables", "fold_assignments", "game_key",
           "open_m15_bundle", "physical_rows", "prepare_m15_data", "sequence_index"]
