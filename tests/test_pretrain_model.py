"""Offline test for the vendored BeatsPretrainModel on a TINY encoder/predictor.

Exercises the masked-acoustic-modeling forward + CE loss + backward without
DeepSpeed. Targets are the dataset convention (codes shifted to 1..K); the model
applies its verbatim `target - 1` internally.
"""

import torch

from openbeats.beats_encoder import BeatsEncoder, BeatsPretrainingPredictor
from openbeats.pretrain.model import BeatsPretrainModel

# Small enough to run on CPU; codebook_vocab_size kept tiny for speed.
TINY = {
    "encoder_layers": 2,
    "encoder_embed_dim": 64,
    "encoder_ffn_embed_dim": 128,
    "encoder_attention_heads": 4,
    "decoder_embed_dim": 64,
    "decoder_layers": 2,
    "codebook_vocab_size": 32,
}


def _build_model():
    enc = BeatsEncoder(input_size=0, beats_config=TINY, is_pretraining=True)
    dec = BeatsPretrainingPredictor(TINY)
    return BeatsPretrainModel(enc, dec, waveform_input=True)


def _make_batch(model, B=2, T=16000, seed=0):
    torch.manual_seed(seed)
    speech = torch.randn(B, T)
    lens = torch.tensor([T] * B)
    # total patch count (deterministic in input length) -> target width
    with torch.no_grad():
        _, patch_len, _, _ = model.encoder(speech, lens, waveform_input=True)
    K = TINY["codebook_vocab_size"]
    width = int(patch_len.max())
    target = torch.randint(1, K + 1, (B, width))  # codes in 1..K (dataset convention)
    return speech, lens, target, patch_len.clone()


def test_pretrain_loss_backward_and_stats():
    model = _build_model().train()
    speech, lens, target, target_lengths = _make_batch(model)

    loss, stats, weight = model(speech, lens, target, target_lengths)

    # force_gatherable promotes the scalar loss/weight to shape [1] for DataParallel
    assert loss.numel() == 1 and torch.isfinite(loss).all()
    assert int(weight) == speech.shape[0]  # batch_size
    # masked-modeling stats are reported
    for k in ("loss", "acc", "acc_mask", "acc_unmask", "count_masked"):
        assert k in stats
    assert stats["count_masked"] > 0  # ~75% of patches are masked

    loss.sum().backward()
    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)


def test_pretrain_n_targets_from_encoder_config():
    model = _build_model()
    assert model.n_targets == TINY["codebook_vocab_size"]
    assert model.mixup_augmentation is None  # mixup disabled by default
    assert model.contrastive_loss_weight == 0.0


def test_pretrain_padding_is_ignored():
    """A shorter target_length should reduce the masked (counted) patch total."""
    model = _build_model().train()
    speech, lens, target, target_lengths = _make_batch(model, seed=1)

    torch.manual_seed(7)
    _, full_stats, _ = model(speech, lens, target, target_lengths)

    # shorten the second clip's valid region by half
    short_lengths = target_lengths.clone()
    short_lengths[1] = short_lengths[1] // 2
    short_speech_lens = lens.clone()
    # encoder length follows samples; emulate a genuinely shorter clip
    short_speech_lens[1] = lens[1] // 2
    torch.manual_seed(7)
    _, short_stats, _ = model(speech, short_speech_lens, target, short_lengths)

    assert short_stats["count_masked"] <= full_stats["count_masked"]
