"""Offline tests for the vendored acoustic tokenizer.

Only the random BestRQ tokenizer is exercised here: it needs no checkpoint and is
seed-deterministic. The real BeatsTokenizer is covered behind OPENBEATS_INTEGRATION
(needs a downloaded {"cfg","model"} checkpoint).
"""

import numpy as np
import torch

from openbeats.tokenizer import BeatsRandomTokenizer, build_tokenizer

SR = 16000


def _tone(seconds, freq=440.0):
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    return torch.from_numpy((0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32))


def test_random_tokenizer_codes_shape_and_range():
    tok = build_tokenizer("random", seed=45)
    wav = _tone(1.0)
    xs = wav.unsqueeze(0)  # (1, T)
    ilens = torch.tensor([wav.numel()])
    out = tok.encode(xs, ilens, waveform_input=True)

    codes, lengths = out["codes"], out["code_lengths"]
    assert codes.dim() == 2 and codes.shape[0] == 1
    assert codes.shape[1] == int(lengths[0])  # no trailing padding for a single clip
    # codes are raw 0..K-1 (the +1 <unk> shift is applied later, in the dump)
    assert int(codes.min()) >= 0
    assert int(codes.max()) < tok.config.quant_n


def test_random_tokenizer_deterministic_across_builds():
    wav = _tone(1.0)
    xs = wav.unsqueeze(0)
    ilens = torch.tensor([wav.numel()])

    a = build_tokenizer("random", seed=45).encode(xs, ilens)["codes"]
    b = build_tokenizer("random", seed=45).encode(xs, ilens)["codes"]
    c = build_tokenizer("random", seed=7).encode(xs, ilens)["codes"]

    assert torch.equal(a, b)  # same seed -> identical codes
    assert not torch.equal(a, c)  # different seed -> different codebook/projection


def test_random_tokenizer_batch_matches_single_in_valid_region():
    """The alignment invariant: padding one clip next to a longer one must not
    change its codes (no cross-example leakage), and lengths track patch counts."""
    tok = build_tokenizer("random", seed=45)
    short, long = _tone(0.7, 440.0), _tone(1.3, 660.0)

    # single-clip references
    s_out = tok.encode(short.unsqueeze(0), torch.tensor([short.numel()]))
    l_out = tok.encode(long.unsqueeze(0), torch.tensor([long.numel()]))

    # batched (short is right-padded with zeros to long's length)
    batch = torch.zeros(2, long.numel())
    batch[0, : short.numel()] = short
    batch[1] = long
    ilens = torch.tensor([short.numel(), long.numel()])
    b_out = tok.encode(batch, ilens)

    s_len = int(s_out["code_lengths"][0])
    l_len = int(l_out["code_lengths"][0])
    assert int(b_out["code_lengths"][0]) == s_len
    assert int(b_out["code_lengths"][1]) == l_len
    # longer clip -> more patches
    assert l_len > s_len
    # codes in each clip's valid region are identical batched vs single
    assert torch.equal(b_out["codes"][0, :s_len], s_out["codes"][0, :s_len])
    assert torch.equal(b_out["codes"][1, :l_len], l_out["codes"][0, :l_len])


def test_build_tokenizer_returns_eval_module():
    tok = build_tokenizer("random", seed=1)
    assert isinstance(tok, BeatsRandomTokenizer)
    assert not tok.training
