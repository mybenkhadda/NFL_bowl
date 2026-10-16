"""Mixed-precision regressions for the 12F loss terms (``se._supcon``, ``se._infonce``, ``se._mmd``, ``se.batch_loss``).

Original failure (Colab T4, fp16 autocast): ``sim.masked_fill(eye, -1e9)`` on an fp16 similarity matrix raised
"value cannot be converted to type c10::Half without overflow". The loss terms now run in autocast-disabled fp32
regions; in fp32 they must equal the original formulation. CUDA-only cases are skipped explicitly without a GPU.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU (fp16 autocast)")
needs_bf16 = pytest.mark.skipif(not (CUDA and torch.cuda.is_bf16_supported()), reason="needs a CUDA GPU with bf16 support")


def _se():
    from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se

    return se


# ------------------------------------------------------------------ the pre-fix formulations (reference)
def _supcon_reference(z, labels, domain, temperature):
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & (domain[:, None] != domain[None, :]) & (labels[:, None] >= 0) & ~eye
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0.0
    logp = sim.masked_fill(eye, -1e9).log_softmax(1)
    return -(logp * pos.float()).sum(1)[has].div(pos.float().sum(1)[has]).mean()


def _infonce_reference(z1, z2, temperature):
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    lg = z1 @ z2.T / temperature
    lab = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))


def _mmd_reference(a, b):
    if a.shape[0] < 2 or b.shape[0] < 2:
        return a.sum() * 0.0
    x = torch.cat([a, b])
    d = torch.cdist(x, x) ** 2
    h = d.detach().median().clamp_min(1e-6)
    k = torch.exp(-d / h)
    n = a.shape[0]
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def _batch(n=64, d=32, n_labels=6, device="cpu", seed=0):
    """Domain-balanced batch (half / half) with repeated labels and ~10 % invalid (-1) labels."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    labels = torch.randint(0, n_labels, (n,), generator=g)
    labels[torch.rand(n, generator=g) < 0.1] = -1
    domain = (torch.arange(n) >= n // 2).long()
    return z.to(device), labels.to(device), domain.to(device)


# ------------------------------------------------------------------ fp32 baseline (runs everywhere)
@pytest.mark.parametrize("temperature", [0.01, 0.05, 0.1, 1.0])
def test_supcon_fp32_matches_original(temperature):
    se = _se()
    z, y, dom = _batch()
    torch.testing.assert_close(se._supcon(z, y, dom, temperature), _supcon_reference(z, y, dom, temperature), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n", [256, 1024])
def test_supcon_fp32_large_batch(n):
    se = _se()
    z, y, dom = _batch(n=n, n_labels=18)
    out = se._supcon(z, y, dom, 0.1)
    assert out.dtype == torch.float32 and math.isfinite(float(out))
    torch.testing.assert_close(out, _supcon_reference(z, y, dom, 0.1), rtol=1e-5, atol=1e-5)


def test_infonce_and_mmd_fp32_match_original():
    se = _se()
    z1, _, _ = _batch(seed=1)
    z2, _, _ = _batch(seed=2)
    torch.testing.assert_close(se._infonce(z1, z2, 0.1), _infonce_reference(z1, z2, 0.1), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(se._mmd(z1[:20], z2[:30]), _mmd_reference(z1[:20], z2[:30]), rtol=1e-6, atol=1e-6)


def test_supcon_gradients_reach_projection():
    se = _se()
    torch.manual_seed(0)
    proj = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 32))
    emb, y, dom = _batch()
    loss = se._supcon(proj(emb), y, dom, 0.1)
    loss.backward()
    grads = [p.grad for p in proj.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


# ------------------------------------------------------------------ edge cases
def _assert_zero_with_graph(se, z, y, dom):
    z = z.clone().requires_grad_(True)
    out = se._supcon(z, y, dom, 0.1)
    assert float(out) == 0.0 and out.requires_grad
    out.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and float(z.grad.abs().sum()) == 0.0


def test_supcon_no_cross_domain_positive():
    se = _se()
    z, _, dom = _batch(n=16)
    y = torch.arange(16)                                       # every label unique → no positive anywhere
    _assert_zero_with_graph(se, z, y, dom)
    y = torch.where(dom == 0, torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long))   # same-domain repeats only
    _assert_zero_with_graph(se, z, y, dom)


def test_supcon_single_domain_batch():
    se = _se()
    z, y, _ = _batch(n=16)
    _assert_zero_with_graph(se, z, y, torch.zeros(16, dtype=torch.long))


def test_supcon_all_labels_invalid():
    se = _se()
    z, _, dom = _batch(n=16)
    _assert_zero_with_graph(se, z, torch.full((16,), -1, dtype=torch.long), dom)


def test_supcon_repeated_labels_and_anchor_eligibility():
    """All rows share one label: every anchor has n/2 cross-domain positives; invalid rows are never anchors or positives."""
    se = _se()
    z, _, dom = _batch(n=12)
    y = torch.zeros(12, dtype=torch.long)
    y[0] = -1
    out = se._supcon(z, y, dom, 0.1)
    torch.testing.assert_close(out, _supcon_reference(z, y, dom, 0.1), rtol=1e-6, atol=1e-6)
    assert math.isfinite(float(out)) and float(out) > 0


def test_supcon_cpu_bf16_autocast_matches_fp32():
    """CPU bf16 autocast (the CPU mixed-precision path): the loss is still computed in fp32 from the given z."""
    se = _se()
    z, y, dom = _batch(n=128)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = se._supcon(z, y, dom, 0.1)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _supcon_reference(z, y, dom, 0.1), rtol=1e-5, atol=1e-5)


def test_check_finite_loss_names_the_term():
    from nflbowl.training.trainer import check_finite_loss

    check_finite_loss({"mask": 0.1, "align": 2.0}, 2.1, "ok")
    with pytest.raises(FloatingPointError, match="align"):
        check_finite_loss({"mask": 0.1, "align": float("nan")}, None, "F4 t epoch 0 train batch 3")
    with pytest.raises(FloatingPointError, match="total"):
        check_finite_loss({}, float("inf"), "validation")


# ------------------------------------------------------------------ CUDA fp16 / bf16 autocast
def _projected(dtype, n=256, d=32, seed=0):
    """z as the trainer produces it: the projection head's output under CUDA autocast (fp16 / bf16)."""
    torch.manual_seed(seed)
    proj = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, d)).cuda()
    emb, y, dom = _batch(n=n, d=d, n_labels=18, device="cuda", seed=seed)
    with torch.autocast(device_type="cuda", dtype=dtype):
        z = proj(emb)
    return proj, emb, z, y, dom


@needs_cuda
def test_original_supcon_overflows_in_fp16():
    """Documents the bug the fix addresses (fails if PyTorch ever starts accepting -1e9 for half)."""
    _, _, z, y, dom = _projected(torch.float16)
    assert z.dtype == torch.float16
    with pytest.raises(RuntimeError, match="overflow"):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _supcon_reference(z, y, dom, 0.1)


@needs_cuda
@pytest.mark.parametrize("n", [64, 256, 1024])
@pytest.mark.parametrize("temperature", [0.01, 0.1])
def test_supcon_fp16_autocast_finite_and_matches_fp32(n, temperature):
    se = _se()
    _, _, z, y, dom = _projected(torch.float16, n=n)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = se._supcon(z, y, dom, temperature)
    assert out.dtype == torch.float32 and math.isfinite(float(out))
    torch.testing.assert_close(out, _supcon_reference(z.float(), y, dom, temperature), rtol=1e-4, atol=1e-4)


@needs_bf16
def test_supcon_bf16_autocast_finite_and_matches_fp32():
    se = _se()
    _, _, z, y, dom = _projected(torch.bfloat16)
    assert z.dtype == torch.bfloat16
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = se._supcon(z, y, dom, 0.1)
    assert out.dtype == torch.float32 and math.isfinite(float(out))
    torch.testing.assert_close(out, _supcon_reference(z.float(), y, dom, 0.1), rtol=1e-4, atol=1e-4)


@needs_cuda
def test_supcon_fp16_gradients_with_grad_scaler():
    """fp16 autocast + GradScaler, as in the trainer: unscaled gradients reach the projection and are finite."""
    se = _se()
    proj, emb, _, y, dom = _projected(torch.float16)
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = se._supcon(proj(emb), y, dom, 0.1)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grads = [p.grad for p in proj.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)
    scaler.step(opt)
    scaler.update()


@needs_cuda
def test_supcon_fp16_edge_cases_finite():
    se = _se()
    _, _, z, _, dom = _projected(torch.float16, n=32)
    cases = [torch.arange(32, device="cuda"), torch.full((32,), -1, device="cuda", dtype=torch.long)]
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outs = [se._supcon(z, y, dom, 0.1) for y in cases] + [se._supcon(z, torch.zeros(32, dtype=torch.long, device="cuda"),
                                                                         torch.zeros(32, dtype=torch.long, device="cuda"), 0.1)]
    assert all(float(o) == 0.0 for o in outs)


# ------------------------------------------------------------------ full batch_loss
def _encoder_batch(device, n=32, seed=0):
    se = _se()
    g = torch.Generator().manual_seed(seed)
    L, C = se.STORE_LEN, 6
    x = torch.randn(n, L, C, generator=g)
    lens = torch.randint(20, L + 1, (n,), generator=g)
    m = torch.arange(L)[None, :] < lens[:, None]
    phys = torch.randn(n, len(se.PHYS), generator=g)
    phys_ok = torch.rand(n, len(se.PHYS), generator=g) > 0.2
    state = torch.randint(0, 6, (n,), generator=g)
    cat = torch.randint(0, 3, (n,), generator=g)
    dom = (torch.arange(n) >= n // 2).long()
    return [t.to(device) for t in (x, m, phys, phys_ok, state, cat, dom)]


CONFIGS = ["F4_physics_contrastive", "F5_jepa_physics", "F6_transformer", "F4_mmd", "F4_adv", "F4_category"]


@pytest.mark.parametrize("name", CONFIGS)
def test_batch_loss_cpu_fp32_finite(name):
    from dataclasses import replace

    se = _se()
    torch.manual_seed(0)
    model = se.SharedEncoder(replace(se.by_name(name), embed_dim=8, width=8))
    total, parts = se.batch_loss(model, *_encoder_batch("cpu"), torch.Generator().manual_seed(0))
    assert math.isfinite(float(total)) and parts and all(math.isfinite(v) for v in parts.values())
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@needs_cuda
@pytest.mark.parametrize("name", CONFIGS)
def test_batch_loss_cuda_fp16_autocast_finite(name):
    """The exact Colab failure path: se.batch_loss inside fp16 autocast with a GradScaler backward."""
    from dataclasses import replace

    se = _se()
    torch.manual_seed(0)
    model = se.SharedEncoder(replace(se.by_name(name), embed_dim=32, width=48)).cuda()
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        total, parts = se.batch_loss(model, *_encoder_batch("cuda", n=256), torch.Generator().manual_seed(0))
    assert total.dtype == torch.float32
    assert math.isfinite(float(total)) and all(math.isfinite(v) for v in parts.values()), parts
    scaler.scale(total).backward()
