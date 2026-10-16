"""Combine Intelligence Lab — application wiring (Milestone 10). AUTHORED 2026-10-14, NOT EXECUTED by Claude.

Static part (always runs; no Streamlit needed): every page registered in ``app.py`` exists and compiles; every module
the pages import from ``src/`` imports without Streamlit; the app's required mart tables are part of the 09E contract.

Headless part (runs only when ``streamlit.testing`` is installed AND the full mart exists): each page is executed by
Streamlit's ``AppTest`` with a real athlete selected, and must finish without an exception. It reads the mart only.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "combine_intelligence_lab"
MART = Path(os.environ.get("CIL_MART_DIR") or ROOT / "warehouse" / "gold" / "advanced_movement" / "intelligence_v1" / "app_mart" / "v1")


def _registered_pages() -> list[str]:
    tree = ast.parse((APP / "app.py").read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "Page" and node.args and isinstance(node.args[0], ast.Constant):
            out.append(node.args[0].value)
    return out


PAGES = _registered_pages()


def test_ten_pages_registered():
    assert len(PAGES) == 10, PAGES


@pytest.mark.parametrize("page", PAGES)
def test_page_exists_and_compiles(page):
    src = APP / page
    assert src.is_file(), f"{page} registered in app.py but missing"
    compile(src.read_text(encoding="utf-8"), str(src), "exec")


@pytest.mark.parametrize("mod", ["product_kit", "movement_gap", "decision_simulator", "signature_stories", "scouting_interpretation"])
def test_shared_engine_imports_without_streamlit(mod):
    m = importlib.import_module(f"nflbdb.research.advanced_movement.{mod}")
    assert "streamlit" not in getattr(m, "__dict__", {}), f"{mod} must not depend on Streamlit"


def test_required_tables_are_in_the_mart_contract():
    from nflbdb.research.advanced_movement import milestone09_artifacts as m9

    src = (APP / "data_access.py").read_text(encoding="utf-8")
    req = ast.literal_eval(src.split("REQUIRED = ", 1)[1].split("\n\n", 1)[0].replace("\n", " "))
    assert set(req) <= set(m9.MART_TABLES), set(req) - set(m9.MART_TABLES)


headless = pytest.mark.skipif(not (MART / "mart_manifest.json").is_file() or importlib.util.find_spec("streamlit") is None,
                              reason="needs streamlit and the executed application mart")


@headless
@pytest.mark.parametrize("page", PAGES)
def test_page_runs_headless(page):
    import polars as pl
    from streamlit.testing.v1 import AppTest

    prof = pl.read_parquet(MART / "prospect_profiles.parquet").filter(pl.col("n_tracked_attempts") > 0).sort("nfl_id")
    at = AppTest.from_file(str(APP / page), default_timeout=120)
    at.session_state["athlete_id"] = int(prof["nfl_id"][0])
    at.session_state["position_filter"] = "All"
    at.session_state["present"] = False
    cwd = os.getcwd()
    os.chdir(APP)
    sys.path.insert(0, str(APP))
    try:
        at.run()
    finally:
        os.chdir(cwd)
        sys.path.remove(str(APP))
    assert not at.exception, f"{page}: {[e.message for e in at.exception]}"
