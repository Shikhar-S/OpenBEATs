"""OpenBEATs inference: load a checkpoint into the vendored BeatsEncoder and
extract embeddings (and classification logits when the checkpoint has a head)."""

import logging

import numpy as np
import torch

from ..data.audio import TARGET_SR, load_audio
from ..nets.encoder import BeatsEncoder
from ..utils.checkpoint import (
    build_classifier,
    encoder_state_dict,
    load_checkpoint,
)

logger = logging.getLogger("openbeats")


class OpenBeats:
    """Inference wrapper around the vendored BeatsEncoder."""

    def __init__(self, encoder, classifier=None, labels=None, multi_label=False, device="cpu"):
        self.encoder = encoder.to(device).eval()
        self.classifier = None if classifier is None else classifier.to(device).eval()
        self.labels = labels
        self.multi_label = multi_label
        self.device = device

    @classmethod
    def from_pretrained(cls, checkpoint, device="cpu", max_layer=None, base=None):
        """Build from a HF repo id, a local dir, or a checkpoint file.

        `base`: explicit base SSL repo id, if a fine-tune's architecture source
        can't be auto-derived.
        """
        ckpt = load_checkpoint(checkpoint, base)
        encoder = BeatsEncoder(input_size=0, beats_config=ckpt.cfg, is_pretraining=False,
                               fbank_mean=ckpt.cfg.get("fbank_mean", 15.41663),
                               fbank_std=ckpt.cfg.get("fbank_std", 6.55582), max_layer=max_layer)
        info = encoder.load_state_dict(encoder_state_dict(ckpt.weights), strict=False)
        missing = [k for k in info.missing_keys if "_pad" not in k]
        if missing:
            logger.warning("Missing encoder weights (first 10): %s", missing[:10])

        classifier = build_classifier(ckpt.weights, encoder.output_size())
        return cls(encoder, classifier, ckpt.labels, ckpt.multi_label, device)

    @torch.no_grad()
    def encode(self, waveform, sample_rate=TARGET_SR, chunk_seconds=None) -> dict:
        """Encode a 1-D 16 kHz waveform -> patch_embeddings (num_patches, D), and
        logits/probs/top_label when a classifier head is present.

        chunk_seconds: if set, process the audio in fixed windows of this length
        (e.g. 10) and concatenate patch embeddings; keeps memory bounded for long
        audio. None = encode the whole clip in one pass.
        """
        if sample_rate != TARGET_SR:
            raise ValueError(f"Expected {TARGET_SR} Hz; use utils.load_audio() to resample.")
        wav = torch.as_tensor(np.asarray(waveform), dtype=torch.float32)
        if wav.dim() != 1:
            raise ValueError(f"Expected a 1-D waveform, got {tuple(wav.shape)}")
        wav = wav.to(self.device)

        step = int(chunk_seconds * TARGET_SR) if chunk_seconds else wav.shape[0]
        patches = torch.cat([self._patches(wav[i:i + step])
                             for i in range(0, wav.shape[0], max(step, 1))])
        out = {"patch_embeddings": patches.cpu().numpy()}

        if self.classifier is not None:
            logits = self.classifier(patches.mean(0))
            probs = torch.sigmoid(logits) if self.multi_label else torch.softmax(logits, -1)
            out["logits"], out["probs"] = logits.cpu().numpy(), probs.cpu().numpy()
            if self.labels and len(self.labels) >= probs.shape[-1]:
                out["top_label"] = self.labels[int(probs.argmax())]
        return out

    def _patches(self, wav: "torch.Tensor") -> "torch.Tensor":
        """Encoder patch embeddings (num_patches, D) for one 1-D waveform window."""
        ilens = torch.tensor([wav.shape[0]], device=self.device)
        rep, olens, _ = self.encoder(wav.unsqueeze(0), ilens, waveform_input=True)
        return rep[0, : int(olens[0])]

    def encode_file(self, path: str, chunk_seconds=None) -> dict:
        return self.encode(*load_audio(path), chunk_seconds=chunk_seconds)
