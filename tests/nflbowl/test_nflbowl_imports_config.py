"""Package imports, configuration presets and validation (no torch needed)."""

from __future__ import annotations

from dataclasses import asdict, fields

import pytest


def test_import_is_light():
    import subprocess
    import sys

    import nflbowl

    assert nflbowl.__version__
    from nflbowl import Experiment, device_info, load_config  # noqa: F401
    # in a fresh interpreter, `import nflbowl` must not pull in torch or polars (lazy attributes)
    code = "import sys, nflbowl; print(int('torch' in sys.modules), int('polars' in sys.modules))"
    import os

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}       # works without an install (pytest pythonpath)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout.split()
    assert out == ["0", "0"]


def test_submodules_import_without_torch():
    import nflbowl.api  # noqa: F401
    import nflbowl.artifacts  # noqa: F401
    import nflbowl.data  # noqa: F401
    import nflbowl.evaluation  # noqa: F401
    import nflbowl.models  # noqa: F401
    import nflbowl.training  # noqa: F401
    import nflbowl.visualization  # noqa: F401


def test_default_science_equals_m12config():
    from nflbdb.research.advanced_movement.m12.config import M12Config
    from nflbowl.config import load_config

    s = load_config("default").science
    m = M12Config()
    for k in ("seed", "embed_dim", "width", "epochs", "batch", "accum_steps", "lr", "patience", "max_minutes", "lambda_mask", "lambda_phys",
              "lambda_temp", "lambda_align", "outer_folds", "n_boot", "n_perm"):
        assert getattr(s, k) == getattr(m, k), k
    assert s.max_val_batches == 50 and s.max_train_batches is None and s.val_share == 0.1
    c = s.to_m12config()
    assert c.epochs == m.epochs and c.hash() == m.hash()


def test_presets_load_and_colab_is_scientifically_identical():
    from nflbowl.config import PRESETS, load_config

    cfgs = {p: load_config(p) for p in PRESETS}
    assert cfgs["colab_gpu"].science_hash() == cfgs["default"].science_hash()
    assert cfgs["research_full"].science_hash() == cfgs["default"].science_hash()
    assert cfgs["fast_iteration"].science_hash() != cfgs["default"].science_hash()
    assert cfgs["fast_iteration"].science.run_mode == "fast_iteration"
    assert cfgs["colab_gpu"].runtime.stage_dir == "/content/nflbowl_stage"
    assert cfgs["colab_gpu"].preset == "colab_gpu"


def test_smoke_preset_matches_m12_smoke_budgets():
    from nflbdb.research.advanced_movement.config import NotebookControls
    from nflbdb.research.advanced_movement.m12.config import M12Config
    from nflbowl.config import load_config

    s = load_config("fast_iteration").science
    m = M12Config().for_mode(NotebookControls(run_mode="smoke", seed=M12Config().seed))
    assert (s.epochs, s.batch, s.max_minutes, s.n_boot, s.n_perm) == (m.epochs, m.batch, m.max_minutes, m.n_boot, m.n_perm)
    assert s.max_train_batches == 4 and s.max_val_batches == 4          # se.train smoke caps


def test_overrides_dotted_and_nested():
    from nflbowl.config import load_config

    c = load_config("default", {"science.epochs": 3, "runtime": {"precision": "fp32"}})
    assert c.science.epochs == 3 and c.runtime.precision == "fp32"
    assert c.science_hash() != load_config("default").science_hash()


@pytest.mark.parametrize("bad", [{"science.batch": 3}, {"science.batch": 7}, {"science.val_share": 0.7}, {"runtime.precision": "fp8"},
                                 {"runtime.data_placement": "disk"}, {"evaluation.backend": "turbo"}, {"science.nonexistent": 1},
                                 {"runtime.compile": "yes"}])
def test_invalid_config_raises(bad):
    from nflbowl.config import ConfigError, load_config

    with pytest.raises(ConfigError):
        load_config("default", bad)


def test_yaml_file_with_extends(tmp_path):
    from nflbowl.config import load_config

    p = tmp_path / "mine.yaml"
    p.write_text("extends: colab_gpu\nscience:\n  epochs: 7\n", encoding="utf-8")
    c = load_config(p)
    assert c.science.epochs == 7 and c.runtime.embed_batch == 4096 and c.preset == "mine"


def test_config_roundtrip_through_dict():
    from nflbowl.config import from_dict, load_config

    c = load_config("colab_gpu")
    assert from_dict(c.to_dict()).science_hash() == c.science_hash()
    assert {f.name for f in fields(c.science)} == set(asdict(c.science))


def test_evaluation_exact_flag():
    from nflbowl.config import load_config

    assert load_config("default").evaluation.is_exact()
    assert not load_config("default", {"evaluation.approx_topk": True}).evaluation.is_exact()
