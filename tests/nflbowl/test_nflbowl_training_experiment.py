"""Training / checkpoint / resume equivalence and the end-to-end Experiment flow on CPU (PyTorch required)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

TINY = {"science.epochs": 3, "science.batch": 16, "science.embed_dim": 8, "science.width": 8, "science.max_val_batches": 2,
        "science.patience": 10, "science.n_boot": 20, "runtime.device": "cpu", "runtime.progress": False, "runtime.embed_batch": 64}


def _prepared(arrays):
    from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se
    from nflbowl.config import load_config
    from nflbowl.data.splits import fold_map, split_indices
    from nflbowl.models import registry as reg
    from nflbowl.training.trainer import DeviceData

    X, M, meta, folds, _ = arrays
    cfg = load_config("default", TINY)
    P = se.physics_targets(X, M)
    data = DeviceData(X, M, se.state_ids(X, M), se.category_ids(meta), (meta["domain"] == "nfl").to_numpy().astype(np.int64), "cpu", "cpu")
    fm = fold_map(meta, folds, 5)
    tr, va, _ = split_indices(meta, fm, None, 0.1, cfg.science.seed)
    return cfg, reg.enc_config("F4", cfg.science), data, P, tr, va


def test_trainer_matches_se_train(arrays, tmp_path):
    """Same seed, same data, CPU fp32 → the nflbowl trainer and se.train produce the same weights and curve."""
    from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se
    from nflbowl.training.trainer import train_encoder

    from dataclasses import replace

    X, M, meta, _, _ = arrays
    cfg, e, data, P, tr, va = _prepared(arrays)
    sci = replace(cfg.science, max_val_batches=50, max_minutes=1e9)          # se.train (full mode) validates on ≤ 50 batches
    res = train_encoder(e, data, P, tr, va, sci, cfg.runtime, tmp_path / "a", "t", sci.seed)
    ref_model, ref_curve, _ = se.train(e, X, M, P, se.state_ids(X, M), se.category_ids(meta), data.dom_np, tr, va, sci.to_m12config(), "cpu",
                                       tmp_path / "b", "full", "t", False, sci.seed)
    sd_a, sd_b = res.model.state_dict(), ref_model.state_dict()
    assert list(sd_a) == list(sd_b)
    for k in sd_a:
        assert torch.equal(sd_a[k].cpu(), sd_b[k].cpu()), k
    assert np.allclose(res.curve["val_loss"].to_numpy(), ref_curve["val_loss"].to_numpy(), rtol=0, atol=0)


def test_resume_restores_finished_and_refuses_overwrite(arrays, tmp_path):
    from nflbowl.training.checkpoints import load_best
    from nflbowl.training.trainer import embed, train_encoder

    cfg, e, data, P, tr, va = _prepared(arrays)
    best = tmp_path / "best.pt"
    r1 = train_encoder(e, data, P, tr, va, cfg.science, cfg.runtime, tmp_path, "t", cfg.science.seed, best_file=best)
    assert not r1.restored_finished and best.is_file()
    r2 = train_encoder(e, data, P, tr, va, cfg.science, cfg.runtime, tmp_path, "t", cfg.science.seed, best_file=best)
    assert r2.restored_finished
    for a, b in zip(r1.model.state_dict().values(), r2.model.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())
    with pytest.raises(FileExistsError):
        train_encoder(e, data, P, tr, va, cfg.science, cfg.runtime, tmp_path, "t", cfg.science.seed, resume=False)
    m, _sig = load_best(best, "cpu")
    assert np.array_equal(embed(m, data, 64), embed(r1.model, data, 64))


def test_interrupted_run_resumes_to_same_weights(arrays, tmp_path):
    """Stop after epoch 1 (simulated by a 2-epoch budget), then continue to 3 epochs with restored RNG streams; the result
    must equal an uninterrupted 3-epoch run."""
    from nflbowl.training.trainer import train_encoder

    cfg, e, data, P, tr, va = _prepared(arrays)
    full =train_encoder(e, data, P, tr, va, cfg.science, cfg.runtime, tmp_path / "full", "t", cfg.science.seed)
    # interrupted: copy the epoch-0 shard of an identical run into a fresh root, then resume
    part_root = tmp_path / "part"
    train_encoder(e, data, P, tr, va, cfg.science, cfg.runtime, tmp_path / "full2", "t", cfg.science.seed)
    import shutil

    src = next((tmp_path / "full2").glob("enc_*/*"))
    dst = part_root / src.parent.name / src.name
    dst.mkdir(parents=True)
    shutil.copy(src / "config.json", dst / "config.json")
    for f in src.glob("shard_epoch__0000.pt*"):
        shutil.copy(f, dst / f.name)
    resumed = train_encoder(e, data, P, tr, va, cfg.science, cfg.runtime, part_root, "t", cfg.science.seed)
    assert resumed.resumed_from_epoch == 0 and resumed.rng_restored
    for a, b in zip(full.model.state_dict().values(), resumed.model.state_dict().values()):
        assert torch.allclose(a.cpu().float(), b.cpu().float(), atol=1e-6)


def test_rng_capture_restore():
    from nflbowl.utils.reproducibility import capture_rng, restore_rng

    rng = np.random.default_rng(1)
    gen = torch.Generator().manual_seed(2)
    st = capture_rng(rng, gen)
    a = (rng.random(3), torch.rand(3, generator=gen), torch.rand(2))
    restore_rng(st, rng, gen)
    b = (rng.random(3), torch.rand(3, generator=gen), torch.rand(2))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def test_experiment_end_to_end(bundle_dir, tmp_path):
    from nflbowl import Experiment

    out = tmp_path / "out"
    exp = Experiment(data_dir=bundle_dir, output_dir=out, config="default", overrides=TINY, name="e2e")
    s = exp.prepare()
    assert s["n_sequences"] == 280
    assert exp.features().shape == (280, 17)
    r0 = exp.evaluate("F0", level="standard")
    assert r0["level"][0] == "standard" and r0["comparable_to_full"][0]
    rq = exp.evaluate("F0", level="quick")
    assert rq["diagnostic"][0]
    exp.train("F4")
    E = exp.embed("F4")
    assert E.shape == (280, 8)
    exp.evaluate("F4", level="full")
    exp.evaluate("random", level="standard")
    assert (exp.root / "evaluation" / "metrics.csv").is_file() and (exp.root / "evaluation" / "identity_null.csv").is_file()
    figs = exp.plot()
    assert "training_curves" in figs and (exp.root / "figures" / "png" / "training_curves.png").is_file()
    assert (exp.root / "figures" / "svg" / "training_curves.svg").is_file()
    exp.report()
    z = exp.export()
    assert z.is_file()

    # new session: reopen and reuse without retraining or re-embedding
    exp2 = Experiment.open(exp.root, device="cpu")
    exp2.prepare()
    n_emb = len(exp2.list_artifacts("embeddings"))
    assert np.array_equal(exp2.embed("F4", reuse=True), E)
    assert len(exp2.list_artifacts("embeddings")) == n_emb
    assert exp2.evaluate("F4", level="full")["cross_auc"][0] == pytest.approx(
        exp.metrics().filter(exp.metrics()["level"] == "full")["cross_auc"][0], nan_ok=True)
    assert exp2.list_models().filter(exp2.list_models()["name"] == "F4_physics_contrastive")["trained"][0]


def test_experiment_inductive_and_transfer(bundle_dir, tmp_path):
    from nflbowl import Experiment

    exp = Experiment(data_dir=bundle_dir, output_dir=tmp_path, overrides={**TINY, "science.epochs": 1}, name="ind")
    exp.prepare()
    exp.train("F2", regime="inductive", folds=[0, 1])
    rows = exp.evaluate("F2", regime="inductive", folds=[0, 1])
    assert set(rows["held_out_fold"].to_list()) == {0, 1}
    exp.train("F2", regime="inductive")                       # remaining folds; finished folds are restored, not retrained
    t = exp.transfer(models=["F2"])
    assert {"M1+F0", "M1+F2_cnn_physics"} <= set(t["model"].to_list())


def test_new_experiment_refuses_other_science(bundle_dir, tmp_path):
    from nflbowl import Experiment
    from nflbowl.config import ConfigError

    Experiment(data_dir=bundle_dir, output_dir=tmp_path, overrides=TINY, name="x")
    with pytest.raises(ConfigError):
        Experiment(data_dir=bundle_dir, output_dir=tmp_path, overrides={**TINY, "science.epochs": 9}, name="x")
    assert Experiment(data_dir=bundle_dir, output_dir=tmp_path, overrides=TINY).id == "experiment_001"
