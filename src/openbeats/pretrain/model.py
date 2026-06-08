# Vendored byte-identical from ESPnet.
#
# Provenance:
#   ESPnet fork: branch `audioverse_copy`, commit 0c3c8cab6d30faa4e2bcf7489d5a552f5b040056
#   BeatsPretrainModel   <- espnet2/beats/espnet_model.py
#   MixupAugment         <- espnet2/layers/mixup_augmentation.py
#
# Changes vs upstream (the only ones permitted by the byte-identical rule, §4.1):
#   * imports repointed at this package's vendored copies (..beats_utils,
#     ..beats_encoder), and `force_gatherable` pulled from beats_utils.
#   * the base class `AbsESPnetModel` -> `nn.Module` (an ESPnet abstract base that
#     is itself just an nn.Module with abstract scaffolding).
#   * `AbsEncoder` (used only as a type annotation) aliased to `nn.Module` so the
#     vendored signatures stay byte-identical without importing ESPnet.
# Mixup and the contrastive CLS term are kept verbatim and disabled via config in
# v1 (mixup_probability=0.0, contrastive_loss_weight=0.0). `collect_feats` is an
# ESPnet-harness method never called by our trainer; kept verbatim.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from ..beats_encoder import make_pad_mask
from ..beats_utils import beats_frontend, force_gatherable

logger = logging.getLogger(__name__)

# ESPnet's AbsEncoder is an abstract nn.Module used here only as a type hint;
# alias to nn.Module so the vendored annotations need no ESPnet import.
AbsEncoder = nn.Module


class MixupAugment(torch.nn.Module):
    """Mixup augmentation module for multi-label classification."""

    def __init__(self, mixup_probability: float, mixup_alpha: float = 0.8):
        super().__init__()
        self.mixup_probability = mixup_probability
        self.mixup_alpha = mixup_alpha

    def forward(
        self, speech: torch.Tensor, onehot: torch.Tensor, speech_lengths: torch.Tensor
    ):
        """Applies mixup augmentation.

        Args:
            speech: (Batch, Length)
            onehot: (Batch, n_classes)
            speech_lengths: (Batch,)

        Returns:
            speech: (Batch, Length)
            onehot: (Batch,..., n_classes)
            speech_lengths: (Batch,): Minimum of the two lengths mixed.
        """
        batch_size = speech.size(0)
        assert onehot.size(0) == batch_size
        assert speech_lengths.size(0) == batch_size
        apply_augmentation = (
            torch.rand((batch_size), device=speech.device) < self.mixup_probability
        )
        mix_lambda = (
            torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
            .sample(sample_shape=(batch_size,))
            .to(speech.device)
            .to(dtype=speech.dtype)
        )
        perm = torch.randperm(batch_size).to(speech.device)
        identity_perm = torch.arange(batch_size, device=speech.device)
        perm[~apply_augmentation] = identity_perm[~apply_augmentation]

        speech_shape = (batch_size,) + (1,) * (speech.dim() - 1)
        mix_lambda_ = mix_lambda.view(speech_shape)
        speech = mix_lambda_ * speech + (1 - mix_lambda_) * speech[perm]

        onehot_shape = (batch_size,) + (1,) * (onehot.dim() - 1)
        mix_lambda_ = mix_lambda.view(onehot_shape)
        onehot = mix_lambda_ * onehot + (1 - mix_lambda_) * onehot[perm]

        speech_lengths = torch.minimum(speech_lengths, speech_lengths[perm])
        speech = speech[:, : speech_lengths.max()]
        if onehot.dim() > 2:
            # onehot is (Batch, length, n_classes), must ensure length match
            onehot = onehot[:, : speech_lengths.max()]
        return speech, onehot, speech_lengths


class BeatsPretrainModel(nn.Module):
    """Beats Pretraining model"""

    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        decoder: nn.Module,
        ignore_id: int = -2,
        label_smoothing: float = 0.1,
        waveform_input: bool = False,
        mixup_probability: float = 0.0,
        contrastive_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.ignore_id = ignore_id
        assert (
            encoder.is_pretraining
        ), "Set the encoder to pretraining mode with is_pretraining=True."
        self.encoder = encoder
        self.decoder = decoder
        self.n_targets = getattr(
            getattr(encoder, "config", None), "codebook_vocab_size", None
        )
        self.waveform_input = waveform_input
        self.mixup_probability = mixup_probability
        self.mixup_augmentation = (
            MixupAugment(mixup_probability) if mixup_probability > 0.0 else None
        )
        self.loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_id,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        self.contrastive_loss_weight = contrastive_loss_weight
        logger.info(
            f"Initialized BeatsPretrainModel with ignore_id={ignore_id}, "
            f"label_smoothing={label_smoothing}, encoder={encoder}, decoder={decoder}"
        )

    def patch_contrastive_loss(self, unmasked_patch_emb, temperature=0.1):
        B, _, _ = unmasked_patch_emb.shape
        assert B % 2 == 0, "Using contrastive loss, batch size must be even. "
        "There is some bug. Check encoder."
        z1 = unmasked_patch_emb[: B // 2, 0, :]  # (B//2, D)
        z2 = unmasked_patch_emb[B // 2 :, 0, :]  # (B//2, D)
        z1 = F.normalize(z1, dim=1)  # (B//2, D)
        z2 = F.normalize(z2, dim=1)  # (B//2, D)
        logits = torch.matmul(z1, z2.t()) / temperature  # (B//2, B//2)
        labels = torch.arange(B // 2, device=logits.device)
        loss_1to2 = F.cross_entropy(logits, labels)
        loss_2to1 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_1to2 + loss_2to1)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        target: torch.Tensor,
        target_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Predictor + Calc loss

        Args:
            speech: (Batch, Length, Dim). Either raw speech or features.
                    If raw speech, then should be single channel ie Dim=1.
            speech_lengths: (Batch, )
            target: (Batch, Length)
            target_lengths: (Batch,)
        """
        assert target_lengths.dim() == 1, target_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == target.shape[0]
            == target_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, target.shape, target_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]
        target = target[:, : target_lengths.max()]
        onehot_ = None
        if self.training and self.mixup_augmentation is not None:
            onehot_ = (target - 1).reshape(-1, 1).contiguous()
            onehot_[onehot_ < 0] = 0  # -1 to 0
            onehot_ = F.one_hot(
                onehot_.squeeze(-1),
                num_classes=self.n_targets,
            ).float()
            onehot_ = onehot_.reshape(target.shape[0], target.shape[1], -1).contiguous()
            speech, onehot_, speech_lengths = self.mixup_augmentation(
                speech, onehot_, speech_lengths
            )

        # unmasked_patch_emb (Batch, n_patch*kept_ratio, emb_dim)
        # restore_ids (Batch, n_patch) -- permutation of [0, 1, ..., n_patch-1]
        # kept_mask (Batch, n_patch)
        # patch_len (Batch,)
        unmasked_patch_emb, patch_len, restore_ids, kept_mask = self.encoder(
            speech, speech_lengths, waveform_input=self.waveform_input
        )

        target = target[:, : patch_len.max()]
        onehot_ = onehot_[:, : patch_len.max()] if onehot_ is not None else None

        if self.contrastive_loss_weight != 0:
            contrastive_loss = (
                self.contrastive_loss_weight
                * self.patch_contrastive_loss(unmasked_patch_emb)
            )
            # Remove cls
            unmasked_patch_emb = unmasked_patch_emb[:, 1:, :]
            restore_ids = restore_ids[:, 1:] - 1
            kept_mask = kept_mask[:, 1:]
            patch_len = torch.cat([patch_len, patch_len], dim=0)
            target = torch.cat([target, target], dim=0)
            onehot_ = (
                torch.cat([onehot_, onehot_], dim=0) if onehot_ is not None else None
            )

        logits = self.decoder(unmasked_patch_emb, patch_len, restore_ids, kept_mask)

        loss, stats = self._calc_beats_loss(
            logits, target - 1, ~kept_mask, patch_len, onehot_
        )  # target - 1 because of unk token at 0th position

        if self.contrastive_loss_weight != 0:
            stats["ContraLoss"] = contrastive_loss.detach()
            stats["MAMloss"] = loss.detach()
            loss = loss + contrastive_loss
            stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        target: torch.Tensor,
        target_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        old_is_pretraining = self.encoder.is_pretraining
        self.encoder.is_pretraining = False
        # for data-parallel
        speech = speech[:, : speech_lengths.max()]
        if self.waveform_input:
            feats, feats_lengths = beats_frontend(
                speech.squeeze(-1),
                fbank_mean=self.encoder.fbank_mean,
                fbank_std=self.encoder.fbank_std,
            )
        else:
            feats, feats_lengths = speech, speech_lengths
        self.encoder.is_pretraining = old_is_pretraining
        return {"feats": feats, "feats_lengths": feats_lengths}

    def _calc_beats_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        masked: torch.Tensor,
        speech_lengths: torch.Tensor,
        onehot_: Optional[torch.Tensor] = None,
    ):
        """Compute loss for Beats model.
        Args:
            logits: (Batch, n_patch, codebook_size)
            target: (Batch, n_patch)
            masked: (Batch, n_patch) -- True for masked, False for unmasked
            speech_lengths: (Batch, )
            onehot_: (Batch, n_patch, codebook_size) -- onehot target for mixup
        Returns:
            loss: scalar
            acc_mask: scalar
            acc_unmask: scalar
        """
        logits = logits.transpose(1, 2)  # (Batch, codebook_size, n_patch)
        if onehot_ is not None:
            onehot_ = onehot_.transpose(1, 2)
            loss = self.loss_function(logits, onehot_)  # mixup loss
        else:
            loss = self.loss_function(logits, target)
        loss = loss * masked  # do not count loss for unmasked patches
        loss = loss.sum()

        padding_mask = make_pad_mask(speech_lengths, traceable=False).to(loss.device)
        unmasked = ~masked & ~padding_mask  # not masked and not padded
        masked = masked & ~padding_mask  # masked and not padded, no-op

        weight = masked.sum() + 1e-10
        loss = loss / weight  # normalize by number of masked patches
        preds = logits.argmax(dim=1)
        with torch.no_grad():
            corr_masked = ((preds == target) * masked).sum()
            corr_unmask = ((preds == target) * unmasked).sum()

            count_unmask = unmasked.sum()
            count_masked = masked.sum()
            acc_m = corr_masked / (count_masked + 1e-10)
            acc_u = corr_unmask / (count_unmask + 1e-10)
            n_uniq_pred = torch.unique(preds, return_counts=False).numel()
            vocab_cov_pred = n_uniq_pred / logits.shape[1]
            n_uniq_pred_masked = torch.unique(
                preds[masked], return_counts=False
            ).numel()
            n_uniq_pred_unmask = torch.unique(
                preds[unmasked], return_counts=False
            ).numel()

            n_uniq_tgt_masked = torch.unique(
                target[masked], return_counts=False
            ).numel()
            n_uniq_tgt = torch.unique(target, return_counts=False).numel()
            vocab_cov_tgt = n_uniq_tgt / logits.shape[1]

        stats_dict = dict(
            loss=loss.detach(),
            acc=acc_m,
            # Masked instance metrics
            acc_mask=acc_m,
            count_masked=count_masked * 1.0,
            n_uniq_tgt_msk=n_uniq_tgt_masked * 1.0,
            n_uniq_pred_msk=n_uniq_pred_masked * 1.0,
            # Unmasked instance metrics
            acc_unmask=acc_u,
            count_unmask=count_unmask * 1.0,
            n_uniq_pred_unmsk=n_uniq_pred_unmask * 1.0,
            n_uniq_pred=n_uniq_pred * 1.0,
            n_uniq_tgt=n_uniq_tgt * 1.0,
            # Vocab coverage metrics
            vocab_cov_tgt=vocab_cov_tgt,
            vocab_cov_pred=vocab_cov_pred,
        )
        return loss, stats_dict
