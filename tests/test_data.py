"""Offline tests for the stage-B data layer (reader / sampler / collate)."""

import json

import numpy as np
import soundfile as sf
import torch

from openbeats.pretrain.data import (
    PAD_CODE,
    LengthBucketBatchSampler,
    TokenDataset,
    build_dataloader,
    collate,
)
from openbeats.tokenize.dump import dump

SR = 16000


def _make_dataset(tmp_path, specs, seed=45):
    manifest = tmp_path / "corpus.jsonl"
    with open(manifest, "w") as f:
        for i, (sec, freq) in enumerate(specs):
            wav = tmp_path / f"clip{i}.wav"
            t = np.linspace(0, sec, int(sec * SR), endpoint=False)
            sf.write(str(wav), (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)
            f.write(json.dumps({"id": f"clip{i}", "audio": str(wav)}) + "\n")
    out = tmp_path / "tokens"
    dump("random", str(manifest), str(out), seed=seed, batch_size=2)
    return str(out)


def test_collate_shapes_and_padding():
    batch = [
        {"id": "a", "speech": torch.ones(100), "speech_lengths": 100,
         "target": torch.tensor([1, 2, 3]), "target_lengths": 3},
        {"id": "b", "speech": torch.full((60,), 2.0), "speech_lengths": 60,
         "target": torch.tensor([4, 5]), "target_lengths": 2},
    ]
    out = collate(batch)
    assert out["speech"].shape == (2, 100)
    assert out["target"].shape == (2, 3)
    # speech zero-padded past its length
    assert torch.all(out["speech"][1, 60:] == 0)
    # target padded with PAD_CODE past its length
    assert out["target"][1, 2] == PAD_CODE
    assert out["target_lengths"].tolist() == [3, 2]
    assert out["speech_lengths"].tolist() == [100, 60]


def test_length_bucket_sampler_equal_counts_and_disjoint_ranks():
    lengths = list(range(1, 41))  # 40 items
    s0 = LengthBucketBatchSampler(lengths, batch_bins=100, rank=0, world_size=2, seed=0)
    s1 = LengthBucketBatchSampler(lengths, batch_bins=100, rank=1, world_size=2, seed=0)
    # critical for DeepSpeed: every rank runs the SAME number of batches/steps
    assert len(s0) == len(s1)
    b0 = [i for batch in s0 for i in batch]
    b1 = [i for batch in s1 for i in batch]
    assert set(b0).isdisjoint(b1)  # disjoint partition
    assert set(b0) | set(b1) <= set(range(40))  # subset (remainder may be dropped)
    # at most world_size-1 *batches* dropped vs the single-rank batching
    full = LengthBucketBatchSampler(lengths, batch_bins=100, world_size=1, seed=0)
    assert 0 <= len(full) - 2 * len(s0) < 2
    # each batch respects the bin budget: max_len * count <= batch_bins
    for batch in s0:
        m = max(lengths[i] for i in batch)
        assert m * len(batch) <= 100 or len(batch) == 1


def test_single_rank_sampler_covers_all():
    lengths = list(range(1, 41))
    s = LengthBucketBatchSampler(lengths, batch_bins=100, world_size=1, seed=0)
    seen = [i for batch in s for i in batch]
    assert set(seen) == set(range(40))  # no truncation when world_size == 1


def test_dataset_and_dataloader_roundtrip(tmp_path):
    path = _make_dataset(tmp_path, [(0.6, 440), (1.0, 660), (0.8, 330), (0.5, 220)])
    ds = TokenDataset(path)
    assert len(ds) == 4
    assert ds.codebook_size > 0

    _, loader, _ = build_dataloader(
        path, batch_bins=SR * 4, num_workers=0, world_size=1, seed=0
    )
    seen, n_codes_ok = set(), True
    for b in loader:
        B = b["speech"].shape[0]
        assert b["target"].shape[0] == B
        for j in range(B):
            cl = int(b["target_lengths"][j])
            # valid codes are 1..K, padding is PAD_CODE
            assert torch.all(b["target"][j, :cl] >= 1)
            if cl < b["target"].shape[1]:
                assert torch.all(b["target"][j, cl:] == PAD_CODE)
        seen.update(range(len(seen), len(seen) + B))
    # iterate again to count rows
    total = sum(b["speech"].shape[0] for b in loader)
    assert total == 4


def test_data_target_lengths_match_tokenizer_patch_count(tmp_path):
    """The stored target length must equal the tokenizer's code count for the same
    audio (the alignment invariant the offline dump relies on)."""
    from openbeats.tokenizer import build_tokenizer

    path = _make_dataset(tmp_path, [(0.7, 440), (1.1, 550)], seed=45)
    ds = TokenDataset(path)
    tok = build_tokenizer("random", seed=45)

    for i in range(len(ds)):
        item = ds[i]
        wav = item["speech"].unsqueeze(0)
        ilens = torch.tensor([item["speech_lengths"]])
        out = tok.encode(wav, ilens, waveform_input=True)
        assert int(out["code_lengths"][0]) == item["target_lengths"]
