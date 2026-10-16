"""Synthetic validation helpers — callable from a notebook cell, no subprocess and no warehouse access.

    from nflbdb.research.advanced_movement import validation
    report = validation.run_all()          # table: check, status, detail
    validation.assert_all_passed(report)

Every check builds its own small synthetic data (analytic sprint curves with a known answer), writes only inside a
temporary directory that it removes, and raises ``AssertionError`` on failure. ``tests/test_advanced_movement.py``
calls the same functions, so the notebook and a test runner exercise identical code.

Synthetic sprint: v(u) = vmax·(1 − exp(−u/τ)), recorded from u = offset (a late recording start), so
a_tan as a function of speed is (vmax − v)/τ whatever the offset — the property the speed-phase alignment relies on.
"""

from __future__ import annotations

import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import polars as pl

from ..contracts import TableContract, conform
from . import alignment as al
from . import data as dt
from . import eda
from .artifacts import AdvStore, MissingArtifactError
from .config import CHANNEL_NAMES, AdvancedConfig, NotebookControls, resolve_device
from .preprocessing import ChannelScaler
from .sequences import SEQ_FRAMES, SequenceError, SequenceSet, build_sequence_frames, pad_batch

ACFG = AdvancedConfig(outer_folds=4, split_repeats=2, inner_folds=3, min_athletes_curve=5, smoke_athletes=8)
DRILL = ACFG.primary_drill
HALF = 3
POSITIONS = ("DL", "WR", "DB")


# ============================================================================== synthetic data
def athlete_params(i: int) -> tuple[float, float]:
    """(vmax [yd/s], τ [s]) of synthetic athlete i — deterministic and different between athletes."""
    return 9.0 + 0.15 * (i % 11), 0.80 + 0.03 * (i % 7)


def synthetic_frames(n_athletes: int = 24, seed: int = 0, rotate_deg: float = 0.0, shift: tuple[float, float] = (0.0, 0.0),
                     ) -> pl.DataFrame:
    """Phase-03-like frame rows: two forty attempts per athlete (one for every fifth athlete), recorded from a
    random late start, plus one attempt of another drill for every third athlete."""
    rng = np.random.default_rng(seed)
    phi = np.deg2rad(rotate_deg)
    rows = []
    for i in range(n_athletes):
        vmax, tau = athlete_params(i)
        attempts = [(DRILL, a) for a in ((1,) if i % 5 == 0 else (1, 2))] + ([("SHORT_SHUTTLE", 1)] if i % 3 == 0 else [])
        for drill, attempt in attempts:
            n = int(rng.integers(45, 56))
            offset = float(rng.uniform(0.05, 0.6))
            u = offset + 0.1 * np.arange(n)
            v = vmax * (1 - np.exp(-u / tau))
            s = vmax * (u - tau * (1 - np.exp(-u / tau)))
            s = s - s[0]
            wig = 0.05 * np.sin(u)
            x = shift[0] + s * np.cos(phi) - wig * np.sin(phi)
            y = shift[1] + s * np.sin(phi) + wig * np.cos(phi)
            step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
            heading = np.rad2deg(phi + np.arctan2(0.05 * np.cos(u), v))
            base = datetime(2023 + i % 3, 3, 1, 10, 0, 0) + timedelta(minutes=10 * attempt + i)
            for k in range(n):
                rows.append({"event_id": f"E{drill[:2]}{attempt}", "nfl_id": 1000 + i, "attempt": attempt, "drill": drill,
                             "draft_year": 2023 + i % 3, "seg_idx": 1, "pos": k, "seg_len": n,
                             "ts": base + timedelta(milliseconds=100 * k), "t_s": 0.1 * k, "x_s": x[k], "y_s": y[k],
                             "speed": v[k], "a_tan": vmax / tau * np.exp(-u[k] / tau), "a_lat": 0.0, "yaw_rate": 0.0,
                             "heading": heading[k], "step_smooth": None if k == 0 else float(step[k]), "pos_resid": 0.01,
                             "valid_kin": True, "heading_valid": k > 0})
    return pl.DataFrame(rows, schema_overrides={"step_smooth": pl.Float64, "seg_idx": pl.Int32})


def synthetic_mart(n_athletes: int = 24) -> pl.DataFrame:
    return pl.DataFrame([{"nfl_id": 1000 + i, "display_name": f"Athlete {i}", "position_group": POSITIONS[i % 3],
                          "cohort": 2023 + i % 3, "has_combine_results": True,
                          "cmb_forty": 4.4 + 0.02 * (i % 9), "cmb_ten_yd_split": 1.5 + 0.01 * (i % 5),
                          "cmb_combine_weight": 200.0 + 5 * i, "cmb_combine_height": None if i == 1 else 72.0 + i % 6}
                         for i in range(n_athletes)], schema_overrides={"cmb_combine_height": pl.Float64})


def synthetic_exposure(n_athletes: int = 24) -> pl.DataFrame:
    rows = [{"nfl_id": 1000 + i, "season_type": t, "n_plays": p, "n_games": g}
            for i in range(n_athletes) if i % 4 for t, p, g in (("REG", 30 * (i % 6), 1 + i % 6), ("PRE", 12, 2))]
    return pl.DataFrame(rows, schema={"nfl_id": pl.Int64, "season_type": pl.String, "n_plays": pl.Int64, "n_games": pl.Int64})


def synthetic_bundle(n_athletes: int = 24, seed: int = 0, **kw) -> dict:
    frames, mart = synthetic_frames(n_athletes, seed, **kw), synthetic_mart(n_athletes)
    manifest = dt.sequence_manifest(frames, None, None, mart, ACFG, rescue_expected=False)   # fixture: fallback intended
    athletes = dt.athlete_registry(mart, manifest, synthetic_exposure(n_athletes), None, ACFG)
    splits = dt.build_split_registry(athletes, ACFG, seed=7)
    seq = build_sequence_frames(frames, manifest, ACFG, HALF)
    return {"frames": frames, "mart": mart, "manifest": manifest, "athletes": athletes, "splits": splits, "seq_frames": seq,
            "ss": SequenceSet.from_frames(seq, manifest)}


def _raises(exc: type[BaseException], fn: Callable[[], object]) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


# ============================================================================== checks
def check_controls_and_device() -> str:
    assert NotebookControls().validate().smoke
    assert _raises(ValueError, lambda: NotebookControls(run_mode="fast").validate())
    assert _raises(ValueError, lambda: NotebookControls(device="gpu").validate())
    assert resolve_device("cpu") == "cpu" and resolve_device("auto") in ("cpu", "cuda")
    assert AdvancedConfig().for_mode(NotebookControls(run_mode="smoke")).split_repeats == 1
    assert AdvancedConfig().for_mode(NotebookControls(run_mode="full")).split_repeats == AdvancedConfig().split_repeats
    return "run mode and device are validated; smoke keeps every scientific parameter"


def check_manifest_and_registry_contracts() -> str:
    b = synthetic_bundle()
    m, a = conform(b["manifest"], dt.MANIFEST), conform(b["athletes"], dt.ATHLETES)
    assert m["seq_id"].n_unique() == m.height
    assert m.filter(pl.col("drill") != DRILL)["exclusion_reason"].unique().to_list() == ["other_drill"]
    assert not m.filter(pl.col("drill") != DRILL)["eligible_primary"].any()
    prim = m.filter(pl.col("drill") == DRILL)
    assert prim["eligible_primary"].all() and prim.height == 24 * 2 - 5
    per = prim.group_by("nfl_id").agg(pl.col("attempt_order").sort().alias("o"))
    assert all(o == list(range(1, len(o) + 1)) for o in per["o"].to_list())
    assert a.height == 24 and a.filter(pl.col("has_repeat")).height == 24 - 5
    assert a.filter(pl.col("nfl_id") == 1000)["nfl_reg_plays"][0] == 0             # no NFL play → 0, not null
    conform(dt.dataset_audit(a, m, ACFG), dt.AUDIT)
    conform(dt.missingness_matrix(a), dt.MISSING)
    flow = dt.dataset_audit(a, m, ACFG).filter(pl.col("section") == "flow").sort("item")["value"].to_list()
    assert len(flow) == 5 + len(ACFG.nfl_min_plays_grid) and all(x >= y for x, y in zip(flow, flow[1:]))   # nested stages
    return f"{m.height} sequences, {a.height} athletes; contracts hold and the flow is nested"


def check_split_integrity() -> str:
    b = synthetic_bundle()
    s = conform(b["splits"], dt.SPLITS)
    audit = conform(dt.leakage_audit(s, b["manifest"], ACFG), dt.LEAKAGE)
    assert audit["passed"].all(), audit.filter(~pl.col("passed")).to_dicts()
    everyone = set(s["nfl_id"].to_list())
    for r in range(ACFG.split_repeats):
        held = []
        for k in range(ACFG.outer_folds):
            train, test = dt.fold_athletes(s, r, k)
            assert not set(train) & set(test) and set(train) | set(test) == everyone
            held += test
            for j in range(ACFG.inner_folds):
                fit, dev = dt.inner_athletes(s, r, k, j)
                assert not set(fit) & set(dev) and set(fit) | set(dev) == set(train) and not set(dev) & set(test)
        assert sorted(held) == sorted(everyone)                                    # held out exactly once per repeat
    seq_fold = b["manifest"].filter(pl.col("eligible_primary")).join(s.filter(pl.col("repeat") == 0), on="nfl_id")
    assert seq_fold.group_by("nfl_id").agg(pl.col("outer_fold").n_unique())["outer_fold"].max() == 1
    again = dt.build_split_registry(b["athletes"], ACFG, seed=7)
    assert again.equals(b["splits"])                                               # persistent: same seed, same folds
    share = s.filter((pl.col("repeat") == 0) & (pl.col("design_role") == "design")).height / len(everyone)
    assert abs(share - ACFG.design_fraction) < 0.1
    return f"{len(everyone)} athletes; outer and inner folds are disjoint, attempts never straddle a fold"


def check_leakage_is_detected() -> str:
    assert _raises(dt.LeakageError, lambda: dt.assert_disjoint([1, 2, 3], [3, 4]))
    b = synthetic_bundle()
    bad = pl.concat([b["splits"], b["splits"].head(1)])
    assert not dt.leakage_audit(bad, b["manifest"], ACFG).filter(
        pl.col("check") == "one_outer_fold_per_athlete_and_repeat")["passed"][0]
    fewer = b["splits"].filter(pl.col("nfl_id") != 1001)
    assert not dt.leakage_audit(fewer, b["manifest"], ACFG).filter(
        pl.col("check") == "every_eligible_sequence_has_a_fold")["passed"][0]
    return "an athlete on both sides, a duplicated key and an unassigned sequence are all reported"


def check_smoke_subset_is_grouped() -> str:
    ids = list(range(1000, 1024))
    sub = dt.smoke_subset(ids, 8, seed=3)
    assert sub == dt.smoke_subset(ids, 8, seed=3) and len(sub) == 8 and set(sub) <= set(ids)
    assert sub == sorted(set(sub)) and dt.smoke_subset(ids, 100, seed=3) == ids    # athletes, never attempts
    assert dt.smoke_subset(ids, 8, seed=4) != sub
    return "the smoke subset is a deterministic set of ATHLETES; all attempts of a chosen athlete follow him"


def check_sequence_channels_and_invariance() -> str:
    a = synthetic_bundle()
    b = synthetic_bundle(rotate_deg=137.0, shift=(53.0, -20.0))
    fa, fb = conform(a["seq_frames"], SEQ_FRAMES), conform(b["seq_frames"], SEQ_FRAMES)
    first = fa.filter(pl.col("frame") == 0)
    assert first["t_rel"].abs().max() == 0.0 and first["dist"].abs().max() == 0.0 and first["head_sin"].null_count() == first.height
    assert (fa.sort("seq_id", "frame").with_columns(pl.col("dist").diff().over("seq_id").alias("d"))["d"].drop_nulls() >= -1e-9).all()
    for c in ("dist", "along", "lat_dev", "speed", "a_tan", "head_sin", "head_cos"):
        x, y = fa[c].fill_null(0.0).to_numpy(), fb[c].fill_null(0.0).to_numpy()
        assert np.abs(x - y).max() < 1e-6, f"channel {c} changes when the field coordinates are rotated and shifted"
    assert fa.filter(pl.col("boundary")).group_by("seq_id").len()["len"].unique().to_list() == [2 * HALF]
    assert _raises(SequenceError, lambda: build_sequence_frames(a["frames"].drop("speed"), a["manifest"], ACFG, HALF))
    return "channels are unchanged by rotating and translating the field; frame 0 anchors time and distance"


def check_padding_and_masks() -> str:
    ss = synthetic_bundle()["ss"]
    assert ss.n == 43 and ss.channels == CHANNEL_NAMES and np.unique(ss.lengths).size > 1
    assert (ss.X[~ss.mask] == 0).all() and not ss.valid[~ss.mask].any()
    j = ss.channel("head_sin")
    assert not ss.valid[:, 0, j].any() and ss.valid[:, 1, j].all()                 # undefined on frame 0 only
    X, mask = pad_batch([np.ones((3, 2)), np.ones((5, 2))])
    assert X.shape == (2, 5, 2) and mask.sum(1).tolist() == [3, 5] and X[0, 3:].sum() == 0
    assert _raises(SequenceError, lambda: pad_batch([np.ones((0, 2))]))
    assert _raises(SequenceError, lambda: pad_batch([np.ones((3, 2)), np.ones((3, 4))]))
    broken = ss.subset(np.arange(ss.n))
    short = int(np.argmin(broken.lengths))
    broken.X[short, -1, 0] = 5.0                                                   # a value written into padding
    assert _raises(SequenceError, broken.validate)
    sub = ss.for_athletes([1001, 1002])
    assert set(sub.nfl_ids.tolist()) == {1001, 1002} and sub.n == 4 and sub.meta.height == 4
    sub.validate()
    assert ss.select_channels(["speed", "a_tan"]).X.shape[2] == 2
    return f"{ss.n} sequences of {int(ss.lengths.min())}–{int(ss.lengths.max())} frames; padding is zero, masked and refused when violated"


def check_scaler_uses_training_athletes_only() -> str:
    b = synthetic_bundle()
    ss, splits = b["ss"], b["splits"]
    train, test = dt.fold_athletes(splits, 0, 0)
    sc = ChannelScaler.fit(ss, train)
    moved = ss.subset(np.arange(ss.n))
    rows = np.isin(moved.nfl_ids, test)
    moved.X[rows] = moved.X[rows] * 100.0 * moved.valid[rows]                      # change held-out athletes only
    assert np.allclose(ChannelScaler.fit(moved, train).mean, sc.mean) and np.allclose(ChannelScaler.fit(moved, train).std, sc.std)
    z = sc.transform(ss)
    z.validate()
    j = ss.channel("speed")
    tr = np.isin(ss.nfl_ids, train)
    vals = z.X[tr][:, :, j][ss.valid[tr][:, :, j]]
    assert abs(vals.mean()) < 1e-4 and abs(vals.std(ddof=1) - 1) < 1e-3
    sc.assert_unseen(test)
    assert _raises(dt.LeakageError, lambda: sc.assert_unseen(train[:1]))
    back = ChannelScaler.from_dict(sc.to_dict())
    assert np.allclose(back.mean, sc.mean) and back.fit_athletes == sc.fit_athletes
    assert _raises(SequenceError, lambda: sc.transform(ss.select_channels(["speed"])))
    return "statistics ignore held-out athletes, padding and undefined values; the scaler knows whom it has seen"


def check_time_alignments() -> str:
    ss = synthetic_bundle()["ss"]
    el = al.align_elapsed_time(ss, ACFG, ["speed"])
    i = 0
    k = int(ss.lengths[i])
    v = el.channel("speed")[i]
    assert np.allclose(v[:k], ss.series(i, "speed"), atol=1e-4)                    # the grid is the frame clock
    assert np.isnan(v[k:]).all()                                                   # nothing after the last frame
    assert np.allclose(el.t_orig[i, :k], ss.t[i, :k], atol=1e-5)                   # original clock preserved
    nt = al.align_normalized_time(ss, ACFG, ["speed"])
    assert np.isfinite(nt.channel("speed")).all()
    assert np.allclose(nt.channel("speed")[:, 0], [ss.series(r, "speed")[0] for r in range(ss.n)], atol=1e-4)
    assert np.allclose(nt.channel("speed")[:, -1], [ss.series(r, "speed")[-1] for r in range(ss.n)], atol=1e-4)
    return "elapsed time reproduces the frames and stops at the last one; normalised time spans first to last frame"


def check_distance_and_speed_phase_alignment() -> str:
    b = synthetic_bundle()
    ss = b["ss"]
    di = al.align_distance(ss, ACFG, ["speed"])
    path = np.array([np.nanmax(ss.series(r, "dist")) for r in range(ss.n)])
    beyond, inside = di.grid[None, :] > path[:, None] + 1e-3, di.grid[None, :] < path[:, None] - 1e-3
    assert np.isnan(di.channel("speed")[beyond]).all() and np.isfinite(di.channel("speed")[inside]).all()
    ph = al.align_speed_phase(ss, ACFG, ["a_tan", "t_rel", "dist"])
    a = ph.channel("a_tan")
    for r in range(ss.n):
        vmax, tau = athlete_params(int(ss.nfl_ids[r]) - 1000)
        ok = np.isfinite(a[r])
        assert ok.sum() > 10 and np.abs(a[r, ok] - (vmax - ph.grid[ok]) / tau).max() < 0.08    # a(v) = (vmax − v)/τ
        assert not ok[ph.grid > np.nanmax(ss.series(r, "speed")) + 1e-3].any()     # nothing beyond the recorded peak
        assert not ok[ph.grid < ss.series(r, "speed")[0] - 1e-3].any()             # nothing before the recording
    ref = int(np.argmin(np.abs(ph.grid - ACFG.phase_ref_speed)))
    t0 = ph.channel("t_rel")[:, ref]
    assert np.nanmax(np.abs(t0)) < 1e-4                                            # cumulative channels re-zeroed
    rep = [np.flatnonzero(ss.nfl_ids == i) for i in np.unique(ss.nfl_ids)]
    gaps_phase, gaps_time = [], []
    el = al.align_elapsed_time(ss, ACFG, ["speed"]).channel("speed")
    for rows in (r for r in rep if r.size == 2):
        both = np.isfinite(a[rows[0]]) & np.isfinite(a[rows[1]])
        gaps_phase.append(np.abs(a[rows[0], both] - a[rows[1], both]).max())
        both = np.isfinite(el[rows[0]]) & np.isfinite(el[rows[1]])
        gaps_time.append(np.abs(el[rows[0], both] - el[rows[1], both]).max())
    assert max(gaps_phase) < 0.15 and np.median(gaps_time) > 0.3                   # a late start shifts time, not phase
    assert _raises(SequenceError, lambda: al.align_speed_phase(ss.select_channels(["a_tan"]), ACFG, ["a_tan"]))
    return "distance and speed-phase never extrapolate; a(v) matches the analytic curve whatever the recording start"


def check_curve_reliability_and_comparison() -> str:
    b = synthetic_bundle(n_athletes=36)
    ss = b["ss"]
    aligned = al.align_all(ss, ACFG, ["speed", "a_tan", "dist"])
    assert "speed" not in aligned["speed_phase"].channels and "dist" not in aligned["distance"].channels
    good = al.curve_reliability(aligned["speed_phase"], "a_tan", None, ACFG.min_athletes_curve)
    assert good["icc_integrated"] > 0.95 and good["grid_points_estimable"] > 5
    shuffled = aligned["speed_phase"].subset(np.arange(ss.n))
    shuffled.nfl_ids = np.random.default_rng(0).permutation(shuffled.nfl_ids)
    assert al.curve_reliability(shuffled, "a_tan", None, ACFG.min_athletes_curve)["icc_integrated"] < 0.5
    pos = dict(b["athletes"].select("nfl_id", "position_group").iter_rows())
    s0 = b["splits"].filter(pl.col("repeat") == 0)
    sets = {"design": s0.filter(pl.col("design_role") == "design")["nfl_id"].to_list(),
            "confirm": s0.filter(pl.col("design_role") == "confirm")["nfl_id"].to_list(), "all": s0["nfl_id"].to_list()}
    cmp_ = conform(al.compare_alignments(aligned, pos, sets, ACFG), al.COMPARISON)
    assert set(cmp_["method"].unique().to_list()) == set(al.ALIGNERS) and set(cmp_["athlete_set"].unique().to_list()) == set(sets)
    return "curve repeatability is high for a state-anchored trait and collapses when athlete labels are shuffled"


def check_eda_helpers() -> str:
    b = synthetic_bundle()
    summ = conform(eda.sequence_summary(b["ss"]), eda.SUMMARY)
    assert summ.height == b["ss"].n and (summ["first_speed"] > 0).all() and (summ["frames_after_peak"] == 0).all()
    assert len(eda.representative_sequences(summ)) == 3
    reps = eda.repeat_athletes(summ, 3, seed=1)
    assert reps == eda.repeat_athletes(summ, 3, seed=1) and len(reps) == 3
    assert all((b["ss"].nfl_ids == i).sum() >= 2 for i in reps)
    pairs = eda.similar_forty_pairs(summ, b["athletes"], 3)
    ids = pairs["seq_id_a"].to_list() + pairs["seq_id_b"].to_list()
    assert len(ids) == len(set(ids)) and ((pairs["forty_a"] - pairs["forty_b"]).abs() <= 0.01 + 1e-9).all()
    dec = conform(eda.decisions_table({"alignment": "x"}, {"alignment": None}, {}), eda.DECISIONS)
    assert dec["status"].unique().to_list() == ["PENDING REVIEW"]
    return "examples are chosen by rule and reproducibly; an unrecorded decision stays PENDING REVIEW"


def _fake_cfg(root: Path) -> SimpleNamespace:
    paths = SimpleNamespace(project_root=root, tables=root / "reports" / "tables", figures=root / "reports" / "figures",
                            gold=root / "warehouse" / "gold", rel=lambda p: Path(p).as_posix())
    return SimpleNamespace(paths=paths, parquet=SimpleNamespace(polars_kwargs=lambda: {}))


def check_store_isolation_cache_and_figures() -> str:
    import matplotlib.pyplot as plt

    c = TableContract("validation_table", "synthetic", {"k": pl.Int64(), "v": pl.Float64()}, key=("k",), area="row")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fake_cfg(Path(tmp))
        smoke, full = AdvStore(cfg, NotebookControls(run_mode="smoke")), AdvStore(cfg, NotebookControls(run_mode="full"))
        assert "_sample" in smoke.data_dir.parts and "_sample" not in full.data_dir.parts
        assert "_sample" in smoke.models_dir.parts and "_sample" in smoke.figures_dir.parts
        calls = []

        def builder() -> dict[str, pl.DataFrame]:
            calls.append(1)
            return {c.name: pl.DataFrame({"k": [1, 2], "v": [0.5, 1.5]})}

        smoke.build("V", [c], builder, params={"a": 1})
        again = smoke.build("V", [c], builder, params={"a": 1})
        assert len(calls) == 1 and again[c.name].height == 2                      # restored, not recomputed
        smoke.build("V", [c], builder, params={"a": 2})
        assert len(calls) == 2                                                    # a changed parameter recomputes
        forced = AdvStore(cfg, NotebookControls(run_mode="smoke", force_rebuild=True))
        forced.build("V", [c], builder, params={"a": 2})
        assert len(calls) == 3
        assert smoke.parquet(c).is_file() and smoke.csv(c).is_file() and smoke.parquet(c).with_suffix(".schema.json").is_file()
        assert not full.exists(c)                                                 # a smoke output is invisible to full
        assert _raises(MissingArtifactError, lambda: full.require(c, "00_research_setup_and_data_audit.ipynb"))
        dry = AdvStore(cfg, NotebookControls(run_mode="full", save_tables=False, save_figures=False))
        assert dry.build("V", [c], builder)[c.name].height == 2 and not dry.exists(c)
        fig, ax = plt.subplots(figsize=(4, 2.25))
        ax.plot([0, 1], [0, 1])
        smoke.show(fig, "validation_figure", {"points": pl.DataFrame({"x": [0, 1]})}, publication=True, source="synthetic", n="2")
        for ext in ("png", "svg", "pdf", "meta.json"):
            assert (smoke.figures_dir / f"validation_figure.{ext}").is_file(), ext
        assert not (full.figures_dir / "validation_figure.png").exists()
    return "smoke and full outputs never mix; builds are cached by parameters; figures are written as PNG, SVG and PDF"


CHECKS: tuple[Callable[[], str], ...] = (
    check_controls_and_device, check_manifest_and_registry_contracts, check_split_integrity, check_leakage_is_detected,
    check_smoke_subset_is_grouped, check_sequence_channels_and_invariance, check_padding_and_masks,
    check_scaler_uses_training_athletes_only, check_time_alignments, check_distance_and_speed_phase_alignment,
    check_curve_reliability_and_comparison, check_eda_helpers, check_store_isolation_cache_and_figures)


def failure_location(e: BaseException) -> str:
    """File, line and source text of the deepest ``check_*`` frame of a failure — a bare ``assert`` carries no message."""
    frames = traceback.extract_tb(e.__traceback__)
    own = [f for f in frames if f.name.startswith("check_")] or list(frames)
    return f" [{Path(own[-1].filename).name}:{own[-1].lineno} in {own[-1].name}: {own[-1].line}]" if own else ""


def run_all(checks: tuple[Callable[[], str], ...] = CHECKS) -> pl.DataFrame:
    """Run every synthetic check in-process. A failing check is reported, not raised, so one table shows them all."""
    rows = []
    for fn in checks:
        try:
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "PASS", "detail": fn()})
        except Exception as e:                                   # noqa: BLE001 — the table must list every failure
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "FAIL",
                         "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


def assert_all_passed(report: pl.DataFrame) -> None:
    failed = report.filter(pl.col("status") != "PASS")
    if failed.height:
        raise AssertionError(f"{failed.height} synthetic validation check(s) failed: "
                             + "; ".join(f"{r['check']} → {r['detail']}" for r in failed.iter_rows(named=True)))
