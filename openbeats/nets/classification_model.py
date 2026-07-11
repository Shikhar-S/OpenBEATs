"""BEATs classification model for supervised fine-tuning.

A pretrained BeatsEncoder (is_pretraining=False) followed by a masked-pooling linear
head. The head submodule is named decoder.linear_out so the inference path's
build_classifier (utils/checkpoint.py) loads a fine-tuned checkpoint unchanged.

Handles multi-class (softmax + cross-entropy) and multi-label (sigmoid + BCE)
classification and returns the (loss, stats, weight) contract the common
openbeats.train loop drives. Per-batch accuracy is logged in stats; the authoritative
validation metric (accuracy for multi-class, macro mAP for multi-label) is computed
over the whole validation set via the epoch-buffer methods.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .beats_utils import force_gatherable
from .encoder import make_pad_mask
from .pretrain_model import MixupAugment

def label_to_onehot(label, label_lengths, n_classes, classification_type):
    """Convert padded label ids (pad = -1) to float targets.

    multi-class: one id per row -> (B, n_classes) one-hot.
    multi-label: a set of ids per row -> (B, n_classes) multi-hot.
    """
    if classification_type == "multi-class":
        assert int(label_lengths.max()) == 1, "multi-class expects one label per sample"
        return F.one_hot(label.squeeze(-1), n_classes).float()
    # -1 pads map to a dummy column that is then dropped; sum the row's slots to multi-hot.
    safe = label.masked_fill(label < 0, n_classes)
    onehot = F.one_hot(safe.view(-1), n_classes + 1)[:, :n_classes]
    onehot = onehot.view(label.size(0), -1, n_classes).sum(1)
    return (onehot > 0).float()

def multiclass_accuracy(probs, onehot_targets) -> float:
    return float((probs.argmax(-1) == onehot_targets.argmax(-1)).float().mean())

def macro_mean_average_precision(probs, multihot_targets) -> float:
    """Mean over classes of average precision; classes with no positives are skipped."""
    import numpy as np

    p = probs.detach().cpu().numpy()
    y = multihot_targets.detach().cpu().numpy()
    aps = [ap for c in range(p.shape[1])
           if (ap := _average_precision(y[:, c], p[:, c])) is not None]
    return float(np.mean(aps)) if aps else 0.0

def _average_precision(y_true, scores):
    import numpy as np

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return None
    y = y_true[np.argsort(-scores)]
    precision = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / n_pos)

class _PoolHead(nn.Module):
    """Masked pooling over patches + a linear classifier (decoder.linear_out)."""

    def __init__(self, in_dim, n_classes, pooling="mean", dropout=0.0):
        super().__init__()
        assert pooling in ("mean", "max"), f"invalid pooling: {pooling}"
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.linear_out = nn.Linear(in_dim, n_classes)

    def forward(self, hs, hlens):
        mask = make_pad_mask(hlens, maxlen=hs.size(1)).to(hs.device)  # (B, T), True = pad
        if self.pooling == "mean":
            keep = (~mask).unsqueeze(-1).to(hs.dtype)
            pooled = (hs * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        else:
            pooled = hs.masked_fill(mask.unsqueeze(-1), float("-inf")).max(1).values
        if self.dropout is not None:
            pooled = self.dropout(pooled)
        return self.linear_out(pooled)

class BeatsClassificationModel(nn.Module):
    """BeatsEncoder + masked-pool linear head for multi-class / multi-label cls."""

    def __init__(
        self,
        encoder,
        n_classes: int,
        *,
        classification_type: str = "multi-class",
        pooling: str = "mean",
        head_dropout: float = 0.0,
        label_smoothing: float = 0.0,
        mixup_probability: float = 0.0,
        mixup_alpha: float = 0.8,
    ):
        super().__init__()
        assert classification_type in ("multi-class", "multi-label"), classification_type
        assert not getattr(encoder, "is_pretraining", False), \
            "encoder must be built with is_pretraining=False for classification"
        self.encoder = encoder
        self.decoder = _PoolHead(encoder.output_size(), n_classes, pooling, head_dropout)
        self.n_classes = n_classes
        self.classification_type = classification_type
        # best-checkpoint criterion (matches the compute_epoch_metrics key)
        self.selection_metric = "acc" if classification_type == "multi-class" else "mAP"
        self.selection_mode = "max"
        if classification_type == "multi-class":
            self.activation = partial(torch.softmax, dim=-1)
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.activation = torch.sigmoid
            self.loss_fn = nn.BCEWithLogitsLoss()
        self.mixup = (
            MixupAugment(mixup_probability, mixup_alpha) if mixup_probability > 0 else None
        )
        self._eval_preds: list = []
        self._eval_targets: list = []

    def forward(self, speech, speech_lengths, label, label_lengths, **kwargs):
        batch_size = speech.shape[0]
        target = label_to_onehot(
            label, label_lengths, self.n_classes, self.classification_type
        )
        if self.training and self.mixup is not None:
            assert self.classification_type == "multi-label", "mixup is multi-label only"
            speech, target, speech_lengths = self.mixup(speech, target, speech_lengths)

        hs, hlens, _ = self.encoder(speech, speech_lengths, waveform_input=True)
        logits = self.decoder(hs, hlens)
        loss = self.loss_fn(logits, target)
        probs = self.activation(logits)

        stats = {"loss": loss.detach(), "acc": self._batch_acc(probs, label, target)}
        if not self.training:
            self._eval_preds.append(probs.detach())
            self._eval_targets.append(target.detach())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _batch_acc(self, probs, label, target) -> torch.Tensor:
        if self.classification_type == "multi-class":
            return (probs.argmax(-1) == label.squeeze(-1)).float().mean().detach()
        return ((probs > 0.5).float() == target).float().mean().detach()

    def reset_epoch_metrics(self):
        self._eval_preds, self._eval_targets = [], []

    def pop_epoch_buffers(self):
        """Concatenated (preds, targets) collected during eval, then clear; (None, None) if empty."""
        if not self._eval_preds:
            return None, None
        preds = torch.cat(self._eval_preds).cpu()
        targets = torch.cat(self._eval_targets).cpu()
        self.reset_epoch_metrics()
        return preds, targets

    def compute_epoch_metrics(self, preds, targets) -> dict:
        if preds is None:
            return {}
        if self.classification_type == "multi-class":
            return {"acc": multiclass_accuracy(preds, targets)}
        return {"mAP": macro_mean_average_precision(preds, targets)}
