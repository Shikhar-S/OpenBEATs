"""Offline unit tests for the vendored helpers in openbeats.nets.beats_utils.

These exercise the numerical helpers the tokenizer/quantizers rely on (no network,
no checkpoint). Seeded so they're deterministic.
"""

import numpy as np
import torch

from openbeats.nets.beats_utils import (
    beats_frontend,
    ema_inplace,
    force_gatherable,
    freeze_conv_module,
    kmeans,
    l2norm,
    norm_ema_inplace,
)


def test_l2norm_unit_rows():
    x = torch.randn(4, 8)
    n = l2norm(x)
    assert torch.allclose(n.norm(dim=-1), torch.ones(4), atol=1e-6)


def test_ema_inplace_math():
    avg = torch.tensor([1.0, 2.0])
    new = torch.tensor([3.0, 4.0])
    ema_inplace(avg, new, decay=0.9)
    # 0.9*old + 0.1*new
    assert torch.allclose(avg, torch.tensor([1.2, 2.2]), atol=1e-6)


def test_norm_ema_inplace_renormalizes():
    avg = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    new = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    norm_ema_inplace(avg, new, decay=0.5)
    # rows must be L2-normalized after the update
    assert torch.allclose(avg.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_kmeans_deterministic_and_shaped():
    torch.manual_seed(0)
    samples = l2norm(torch.randn(256, 16))

    torch.manual_seed(123)
    means_a, bins_a = kmeans(samples, num_clusters=8, num_iters=5, use_cosine_sim=True)
    torch.manual_seed(123)
    means_b, bins_b = kmeans(samples, num_clusters=8, num_iters=5, use_cosine_sim=True)

    assert means_a.shape == (8, 16)
    assert bins_a.shape == (8,)
    assert int(bins_a.sum()) == 256  # every sample assigned to a cluster
    assert torch.equal(means_a, means_b)  # seed-deterministic
    assert torch.equal(bins_a, bins_b)
    # cosine-sim centroids are returned L2-normalized
    assert torch.allclose(means_a.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_beats_frontend_shape_and_determinism():
    # 1 second of 16 kHz audio -> ~100 fbank frames (10 ms shift), 128 mel bins.
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    wav = torch.from_numpy((0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    source = wav.unsqueeze(0)  # (B=1, T)

    fbank = beats_frontend(source, fbank_mean=15.41663, fbank_std=6.55582)
    assert fbank.shape[0] == 1
    assert fbank.shape[2] == 128
    assert 90 <= fbank.shape[1] <= 102  # ~100 frames for 1 s
    # deterministic
    fbank2 = beats_frontend(source, fbank_mean=15.41663, fbank_std=6.55582)
    assert torch.equal(fbank, fbank2)


def test_freeze_conv_module_sets_weight_one_bias_zero():
    conv = torch.nn.Conv2d(1, 1, kernel_size=16, stride=16, bias=True)
    freeze_conv_module(conv)
    assert torch.all(conv.weight == 1)
    assert not conv.weight.requires_grad
    assert torch.all(conv.bias == 0)
    assert not conv.bias.requires_grad


def test_force_gatherable_scalars_to_tensors():
    out = force_gatherable(
        {"f": 1.5, "i": 3, "scalar": torch.tensor(2.0)}, device="cpu"
    )
    assert out["f"].shape == (1,) and out["f"].dtype == torch.float
    assert out["i"].shape == (1,) and out["i"].dtype == torch.long
    assert out["scalar"].shape == (1,)  # 0-dim promoted to 1-dim
    # list/tuple recursion preserved
    lst = force_gatherable([1, 2.0], device="cpu")
    assert isinstance(lst, list) and lst[0].dtype == torch.long
