"""Offline tests for the classification fine-tuning model (TINY encoder, CPU).

Covers label_to_onehot (both task types, pad=-1), the accuracy/mAP metrics, the
(loss, stats, weight) forward for multi-class and multi-label, frozen-encoder
linear probing, and the epoch-metric buffers.
"""

import torch

from openbeats.nets.classification_model import (
    BeatsClassificationModel,
    _average_precision,
    label_to_onehot,
    macro_mean_average_precision,
    multiclass_accuracy,
)
from openbeats.nets.encoder import BeatsEncoder

SR = 16000
N_CLASSES = 5
TINY = {
    "encoder_layers": 2,
    "encoder_embed_dim": 64,
    "encoder_ffn_embed_dim": 128,
    "encoder_attention_heads": 4,
    "decoder_embed_dim": 64,
    "decoder_layers": 2,
    "codebook_vocab_size": 1024,
}

def _encoder():
    return BeatsEncoder(input_size=0, beats_config=TINY, is_pretraining=False)

def _speech(batch=3, seconds=0.5):
    n = int(seconds * SR)
    torch.manual_seed(0)
    return torch.randn(batch, n), torch.full((batch,), n, dtype=torch.long)

# ---------------------------------------------------------------- label_to_onehot
def test_label_to_onehot_multiclass():
    label = torch.tensor([[0], [4], [2]])
    lengths = torch.ones(3, dtype=torch.long)
    onehot = label_to_onehot(label, lengths, N_CLASSES, "multi-class")
    assert onehot.shape == (3, N_CLASSES)
    assert onehot.argmax(-1).tolist() == [0, 4, 2]
    assert onehot.sum().item() == 3  # exactly one per row

def test_label_to_onehot_multilabel_with_padding():
    # row0: {1,3}, row1: {2}, pad with -1
    label = torch.tensor([[1, 3], [2, -1]])
    lengths = torch.tensor([2, 1])
    onehot = label_to_onehot(label, lengths, N_CLASSES, "multi-label")
    assert onehot.shape == (2, N_CLASSES)
    assert onehot[0].tolist() == [0, 1, 0, 1, 0]
    assert onehot[1].tolist() == [0, 0, 1, 0, 0]

# --------------------------------------------------------------------- metrics
def test_average_precision_perfect_and_empty():
    y = torch.tensor([1.0, 0.0, 1.0]).numpy()
    scores = torch.tensor([0.9, 0.5, 0.8]).numpy()
    assert abs(_average_precision(y, scores) - 1.0) < 1e-9  # perfect ranking
    assert _average_precision((y * 0), scores) is None       # no positives -> skipped

def test_multiclass_accuracy_and_map():
    probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])  # last is wrong
    assert abs(multiclass_accuracy(probs, onehot) - 2 / 3) < 1e-6
    multihot = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert 0.0 <= macro_mean_average_precision(probs, multihot) <= 1.0

# ------------------------------------------------------------ forward contract
def test_forward_multiclass():
    model = BeatsClassificationModel(_encoder(), N_CLASSES, classification_type="multi-class")
    speech, lengths = _speech()
    label = torch.tensor([[0], [1], [2]])
    loss, stats, weight = model(speech, lengths, label, torch.ones(3, dtype=torch.long))
    assert torch.isfinite(loss).all() and loss.numel() == 1  # [1] per force_gatherable
    assert "loss" in stats and "acc" in stats
    assert int(weight) == 3

def test_forward_multilabel():
    model = BeatsClassificationModel(_encoder(), N_CLASSES, classification_type="multi-label")
    speech, lengths = _speech()
    label = torch.tensor([[0, 2], [1, -1], [3, 4]])
    loss, stats, weight = model(speech, lengths, label, torch.tensor([2, 1, 2]))
    assert torch.isfinite(loss).all() and loss.numel() == 1
    assert 0.0 <= float(stats["acc"]) <= 1.0

# ----------------------------------------------------------- linear probing
def test_freeze_encoder_linear_probing():
    model = BeatsClassificationModel(_encoder(), N_CLASSES)
    model.encoder.requires_grad_(False)
    speech, lengths = _speech()
    label = torch.tensor([[0], [1], [2]])
    loss, _, _ = model(speech, lengths, label, torch.ones(3, dtype=torch.long))
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.decoder.linear_out.weight.grad is not None

# ------------------------------------------------------------- epoch metrics
def test_epoch_metric_buffers():
    model = BeatsClassificationModel(_encoder(), N_CLASSES, classification_type="multi-class")
    model.eval()
    speech, lengths = _speech()
    label = torch.tensor([[0], [1], [2]])
    with torch.no_grad():
        model(speech, lengths, label, torch.ones(3, dtype=torch.long))
    preds, targets = model.pop_epoch_buffers()
    assert preds.shape == (3, N_CLASSES) and targets.shape == (3, N_CLASSES)
    metrics = model.compute_epoch_metrics(preds, targets)
    assert "acc" in metrics and 0.0 <= metrics["acc"] <= 1.0
    assert model.pop_epoch_buffers() == (None, None)  # cleared after pop
