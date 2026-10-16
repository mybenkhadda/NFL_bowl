"""Study A (quality + vendor agreement) on synthetic M1-shaped tables — no warehouse needed.

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_study_a_quality.py -q
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from nflbdb.research.contracts import conform  # noqa: E402
from nflbdb.research.studies import quality as Q  # noqa: E402
from nflbdb.research.studies.config import QualityStudyConfig  # noqa: E402
from nflbdb.research.studies.quality_figures import FIGURES  # noqa: E402
from nflbdb.research.studies.results import ResultCollector  # noqa: E402

CFG = QualityStudyConfig(n_boot=200, n_boot_generic=30, min_group_trajectories=3, min_athletes=3)
KEYS = ["event_id", "nfl_id", "attempt"]
N_ATH = 40


def combine_inputs():
    rng = np.random.default_rng(0)
    rows, segs, evs = [], [], []
    for a in range(N_ATH):
        for att in (1, 2):
            eid = f"E{a:03d}"
            bad = a == 0 and att == 1                                 # one clearly poor trajectory
            rows.append({"event_id": eid, "nfl_id": 1000 + a, "attempt": att, "draft_year": 2024 + a % 2,
                         "drill_type": "FORTY_YARD_DASH" if a % 3 else "SHORT_SHUTTLE",
                         "quality_tier": "B" if bad or a in (1, 2, 3) else "A",
                         "tier_reasons": "moderate_residual" if bad else "", "n_segments": 2 if bad else 1,
                         "n_gap_breaks": 1 if bad else 0, "n_jump_breaks": 0, "n_null_xy_removed": 0,
                         "n_dup_ts_removed": 0, "n_offgrid": 3 if bad else 0, "n_valid_frames": 60,
                         "resid_rms_yds": 0.5 if bad else float(rng.uniform(0.01, 0.03)),
                         "speed_vendor_mad": 0.4 if bad else float(rng.uniform(0.01, 0.05)),
                         "max_accel_mag": 14.0 if bad else float(rng.uniform(4, 8))})
            segs.append({"event_id": eid, "nfl_id": 1000 + a, "attempt": att, "seg_idx": 1, "n_frames": 60,
                         "duration_s": 5.9 + 0.1 * att, "frac_heading_valid": 0.9})
            for rule, n in (("lenient", 4 if bad else 2), ("default", 2), ("strict", 1)):
                evs += [{"event_id": eid, "nfl_id": 1000 + a, "attempt": att, "rule": rule, "event_type": "braking"}] * n
    pos = pl.DataFrame({"nfl_id": [1000 + a for a in range(N_ATH)], "position_group": ["WR", "OL"] * (N_ATH // 2)})
    return pl.DataFrame(rows), pl.DataFrame(segs), pl.DataFrame(evs), pos


def dims():
    traj, seg, ev, pos = combine_inputs()
    t = Q.normalize_trajectories(traj, seg, ev, "combine", KEYS, "draft_year", "drill_type", pos)
    return Q.quality_dimensions(t, CFG)


def frames(t: pl.DataFrame) -> pl.DataFrame:
    """Vendor s = derived − 0.1 for every frame → residual r ≡ +0.1 yd/s."""
    rows = []
    for r in t.iter_rows(named=True):
        e, n, a = r["traj_id"].split("|")
        sp = np.linspace(0, 9, 30)
        rows.append(pl.DataFrame({"event_id": e, "nfl_id": int(n), "attempt": int(a), "t_s": np.arange(30) * 0.1,
                                  "x_s": np.cumsum(sp) * 0.1, "y_s": np.zeros(30), "s": sp - 0.1, "speed": sp,
                                  "accel_mag": np.full(30, 11.0 if r["traj_id"].startswith("E000|1000|1") else 3.0),
                                  "valid_kin": True}))
    return pl.concat(rows)


# ----------------------------------------------------------------------------- A1
def test_dimensions_metrics_and_flags():
    t, thr = dims()
    conform(t, Q.DIMS)
    bad = t.filter(pl.col("traj_id") == "E000|1000|1").row(0, named=True)
    assert bad["structural_breaks"] == 2.0                     # 1 gap break + 1 extra segment
    assert bad["offgrid_rate"] == pytest.approx(3 / 60)
    assert bad["heading_invalid_frac"] == pytest.approx(0.1)
    assert bad["event_count_rel_range"] == pytest.approx((4 - 1) / 4)
    assert bad["flag_coordinate_validity"] and bad["n_flagged_dims"] >= 3
    assert t["pct_resid_rms_yds"].drop_nulls().is_between(0, 1).all()
    assert t["smoothing_rel_change"].is_null().all() and not t["flag_smoothing_sensitivity"].any()
    conform(thr, Q.THRESHOLDS_TBL)


def test_nfl_without_rule_variants_has_null_event_stability():
    traj, seg, _, pos = combine_inputs()
    t = Q.normalize_trajectories(traj, seg, None, "nfl", KEYS, "draft_year", None, pos)
    assert t["event_count_rel_range"].is_null().all() and set(t["stratum"]) == {"nfl_play"}


def test_proportions_funnel_reasons_contracts():
    t, _ = dims()
    p = conform(Q.proportions(t, CFG, seed=1), Q.PROPORTIONS)
    tiers = p.filter((pl.col("facet") == "all") & pl.col("indicator").str.starts_with("tier_"))
    assert tiers["share"].sum() == pytest.approx(1.0)
    assert (p["wilson_lower"] <= p["share"] + 1e-12).all() and (p["share"] <= p["wilson_upper"] + 1e-12).all()
    f = conform(Q.filter_funnel(t), Q.FUNNEL).filter((pl.col("stratum") == "all")).sort("scenario_order")
    assert f["n_trajectories"].to_list() == sorted(f["n_trajectories"].to_list(), reverse=True)
    r = conform(Q.reason_matrix(t), Q.REASONS)
    assert {"moderate_residual", "flag:coordinate_validity"} <= set(r["reason"])
    conform(Q.metric_by_tier(t), Q.METRIC_BY_TIER)


def test_a1_results_are_trajectory_grain_and_clustered():
    t, _ = dims()
    rc = ResultCollector("A", CFG.min_athletes)
    Q.a1_results(t, CFG, 1, rc)
    res = conform(rc.frame(), Q.STUDY_RESULTS)
    d = res.filter((pl.col("analysis_id") == "A1.tier_contrast") & (pl.col("feature") == "resid_rms_yds")).row(0, named=True)
    assert d["grain"] == "trajectory" and d["n_frames"] is None and d["n_athletes"] == N_ATH
    assert "cluster" in d["ci_method"]


# ----------------------------------------------------------------------------- A2
def test_vendor_error_known_residual():
    t, _ = dims()
    fr = Q.vendor_frames(frames(t), "combine", KEYS, t, CFG)
    rc = ResultCollector("A", CFG.min_athletes)
    v = conform(Q.vendor_error_summary(fr, CFG, 1, rc), Q.VENDOR)
    a = v.filter(pl.col("facet") == "all").row(0, named=True)
    assert a["mae"] == pytest.approx(0.1) and a["bias"] == pytest.approx(0.1) and a["rmse"] == pytest.approx(0.1)
    assert a["sd"] == pytest.approx(0.0, abs=1e-9) and a["loa_lower"] == pytest.approx(0.1)
    assert a["n_frames"] == fr.height and a["n_athletes"] == N_ATH
    res = rc.frame()
    assert (res["grain"] == "frame").all() and res["n_frames"].is_not_null().all()


def test_frame_diagnostics_and_extreme_accel():
    t, _ = dims()
    fr = Q.vendor_frames(frames(t), "combine", KEYS, t, CFG)
    diag = conform(Q.frame_diagnostics(fr, CFG), Q.FRAME_DIAG)
    assert diag.filter(pl.col("traj_id") == "E000|1000|1")["extreme_accel_rate"][0] == 1.0
    x = conform(Q.extreme_accel_by_tier(diag, CFG, 1), Q.EXTREME_ACCEL)
    assert x.filter(pl.col("quality_tier") == "A")["rate"][0] == 0.0
    conform(Q.hist2d(fr, CFG), Q.HIST2D)
    conform(Q.residual_by_speed(fr), Q.BY_SPEED)
    conform(Q.a2_examples(fr, diag, CFG), Q.A2_EXAMPLES)


# ----------------------------------------------------------------------------- figures (from tables only)
def test_every_study_a_figure_renders(tmp_path):
    t, thr = dims()
    fr = Q.vendor_frames(frames(t), "combine", KEYS, t, CFG)
    rc = ResultCollector("A", CFG.min_athletes)
    diag = Q.frame_diagnostics(fr, CFG)
    ex = Q.select_a1_examples(t, {"combine": set(t["traj_id"].to_list())})
    out = {Q.DIMS.name: t, Q.THRESHOLDS_TBL.name: thr, Q.PROPORTIONS.name: Q.proportions(t, CFG, 1),
           Q.METRIC_BY_TIER.name: Q.metric_by_tier(t), Q.REASONS.name: Q.reason_matrix(t),
           Q.FUNNEL.name: Q.filter_funnel(t), Q.EXTREME_ACCEL.name: Q.extreme_accel_by_tier(diag, CFG, 1),
           Q.A1_EXAMPLES.name: ex, Q.A1_SIGNALS.name: Q.a1_signals(fr, ex),
           Q.VENDOR.name: Q.vendor_error_summary(fr, CFG, 1, rc), Q.FRAME_DIAG.name: diag,
           Q.HIST2D.name: Q.hist2d(fr, CFG), Q.BY_SPEED.name: Q.residual_by_speed(fr),
           Q.A2_EXAMPLES.name: Q.a2_examples(fr, diag, CFG)}
    assert set(ex["example"]) == {"combine_good", "combine_poor"}
    ids = [r.figure_id for r, _ in FIGURES]
    assert len(ids) == len(set(ids)) == 20 and all(r.research_question for r, _ in FIGURES)
    for rec, fn in FIGURES:
        fig, data = fn(out, CFG)
        assert all(isinstance(d, pl.DataFrame) for d in data.values())
        fig.savefig(tmp_path / f"{rec.name}.png", dpi=100)
        fig.savefig(tmp_path / f"{rec.name}.svg")
