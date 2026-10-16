"""M2.1 infrastructure: strict contracts, threshold documentation, artifact naming, figure export.

Includes the regression for the M2 ``m2_three_cone_path_extremes`` export failure (equal-aspect
panels must not rely on axes sharing both x and y) and the caption-overlap fix.

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_m21_infra.py -q
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from nflbdb.research.combine_analytics import FIGURES as M2_FIGURES, TABLES as M2_TABLES  # noqa: E402
from nflbdb.research.m21 import plotting as P  # noqa: E402
from nflbdb.research.m21.config import (ESTABLISHED, EXECUTION, PROPOSED, THRESHOLD_DOCS,  # noqa: E402
                                        DuplicateAuditConfig, M21Config, threshold_table)
from nflbdb.research.m21.duplicates import (CONTRACTS, FIGURES, audit_frames, example_signals, finalize,  # noqa: E402
                                            flag_summary, plot_duplicate_examples, plot_flag_prevalence,
                                            plot_similarity, select_examples, similarity_histogram)
from nflbdb.research.m21.io import M21_PHASE, ContractError, TableContract, conform  # noqa: E402

from test_m21_duplicates import run, shifted  # noqa: E402

C = TableContract("t", "test", {"k": pl.Int64(), "v": pl.Float64()}, key=("k",))


# ----------------------------------------------------------------------------- contracts
def test_conform_orders_columns_and_accepts_exact_schema():
    out = conform(pl.DataFrame({"v": [1.0], "k": [1]}), C)
    assert out.columns == ["k", "v"]


@pytest.mark.parametrize("df, msg", [
    (pl.DataFrame({"k": [1]}), "missing"),
    (pl.DataFrame({"k": [1], "v": [1.0], "z": [0]}), "extra"),
    (pl.DataFrame({"k": [1], "v": [1]}), "≠"),                               # Int64 is NOT silently cast
    (pl.DataFrame({"k": [1, 1], "v": [1.0, 2.0]}), "duplicate"),
    (pl.DataFrame({"k": [None], "v": [1.0]}, schema={"k": pl.Int64, "v": pl.Float64}), "null"),
])
def test_conform_rejects_contract_violations(df, msg):
    with pytest.raises(ContractError, match=msg):
        conform(df, C)


def test_contract_declares_its_key_and_kind():
    with pytest.raises(ValueError):
        TableContract("bad", "", {"a": pl.Int64()}, key=("b",))
    with pytest.raises(ValueError):
        TableContract("bad", "", {"a": pl.Int64()}, kind="csv")


def test_m21_artifacts_are_namespaced_and_never_collide_with_m2():
    names = [c.name for c in CONTRACTS]
    assert len(names) == len(set(names))
    assert not set(names) & set(M2_TABLES)
    assert all(f.startswith("m21_") for f in FIGURES) and not set(FIGURES) & set(M2_FIGURES)
    assert M21_PHASE == "phase_03/m2_1"


# ----------------------------------------------------------------------------- thresholds
def test_every_threshold_is_documented_with_provenance():
    for section, docs in THRESHOLD_DOCS.items():
        names = {f.name for f in fields(getattr(M21Config(), section))}
        assert names == set(docs), f"{section}: undocumented {names - set(docs)}, stale {set(docs) - names}"
        assert {d[1] for d in docs.values()} <= {ESTABLISHED, PROPOSED, EXECUTION}
    t = threshold_table()
    assert t.height == sum(len(d) for d in THRESHOLD_DOCS.values()) and t["rationale"].str.len_chars().min() > 0


def test_config_params_are_json_able_and_change_with_thresholds():
    import json

    a = M21Config().section("duplicates")
    b = M21Config(duplicates=DuplicateAuditConfig(near_dup_speed_rmse=0.1)).section("duplicates")
    assert json.dumps(a) != json.dumps(b)


# ----------------------------------------------------------------------------- figures
def test_equal_aspect_panels_export_without_shared_axes(tmp_path):
    """Regression for m2_three_cone_path_extremes (shared x+y with adjustable='datalim')."""
    fig, axes = P.figure(1, 2)
    xs = [np.linspace(0, 10, 20), np.linspace(-2, 4, 20)]
    ys = [np.sin(xs[0]), np.cos(xs[1]) * 5]
    for ax, x, y in zip(axes[0], xs, ys):
        ax.plot(x, y)
    P.equal_aspect_panels(axes[0], xs, ys)
    P.finish(fig, "caption " * 60, title="t")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(tmp_path / f"f.{ext}", dpi=200)
        assert (tmp_path / f"f.{ext}").stat().st_size > 0
    assert axes[0, 0].get_xlim() == axes[0, 1].get_xlim() and axes[0, 0].get_ylim() == axes[0, 1].get_ylim()
    assert fig.get_layout_engine().get()["rect"][1] > 0          # caption band reserved


def test_workstream_a_figures_render_from_tables_only(tmp_path):
    cfg = DuplicateAuditConfig()
    a = run()
    frames = pl.concat([a, a.with_columns(pl.lit("E2").alias("event_id")),
                        shifted(run(seed=2, jitter=0.01), dx=5.0, dt_ms=60_000, event_id="E3"),
                        run(event_id="E4", vmax=9.3, tau=1.15, t0_ms=120_000)])
    att, pairs = audit_frames(frames, cfg)
    _, pairs, flags = finalize(att, pairs, cfg, "test")
    ex = select_examples(pairs, cfg)
    sig = example_signals(frames, ex, cfg)
    assert set(ex["example"]) >= {"raw_duplicate", "consistent"} and sig.height > 0
    figs = [plot_duplicate_examples(sig, ex, pairs, cfg),
            plot_similarity(similarity_histogram(pairs, cfg), pairs, cfg),
            plot_flag_prevalence(flag_summary(flags))]
    for i, fig in enumerate(figs):
        for ext in ("png", "svg"):
            fig.savefig(tmp_path / f"{i}.{ext}", dpi=200)


def test_example_figure_handles_empty_classes(tmp_path):
    from nflbdb.research.m21.duplicates import EXAMPLES, PAIRS, SIGNALS

    fig = plot_duplicate_examples(SIGNALS.empty(), EXAMPLES.empty(), PAIRS.empty(), DuplicateAuditConfig())
    fig.savefig(tmp_path / "empty.png")
