"""Plumbing for OpenBEATs: Hugging Face downloads, audio loading, and the
checkpoint loader that normalizes the two on-disk formats into one spec.

Checkpoint formats:
  - SSL encoder (shikhar7ssu/OpenBEATs-*): self-contained {"cfg", "model"}.
  - ESPnet fine-tune (espnet/OpenBEATS-*-<task>): bare .pth + config.yaml whose
    architecture lives in a base SSL repo -> we fetch that base's config.yaml.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("openbeats")

DEFAULT_REPO = "espnet/OpenBEATS-Large-i2-as20k"
ALLOW_PATTERNS = ["*config.yaml", "*epoch*.pt", "*.pth", "*.ckpt"]
TARGET_SR = 16000

_CLASSIFIER_KEYS = ("classifier.weight", "head.weight", "output_layer.weight",
                    "decoder.linear_out.weight")


# --------------------------------------------------------------------- downloads
def download_checkpoint(repo_id=DEFAULT_REPO, dest=None, patterns=None) -> str:
    """Download (part of) a repo; return the local snapshot directory."""
    from huggingface_hub import snapshot_download

    dest = dest or os.path.join("checkpoints", repo_id.split("/")[-1])
    return snapshot_download(repo_id, allow_patterns=patterns or ALLOW_PATTERNS,
                             local_dir=dest)


def find_artifacts(snapshot_dir):
    """Return (config_path, checkpoint_path), either may be None."""
    config = ckpt = None
    for root, _, files in os.walk(snapshot_dir):
        for f in files:
            if f == "config.yaml":
                config = config or os.path.join(root, f)
            elif f.endswith((".pt", ".pth", ".ckpt")):
                ckpt = ckpt or os.path.join(root, f)
    return config, ckpt


def load_audio(path, target_sr: int = TARGET_SR):
    """Load an audio file as a mono float32 waveform resampled to target_sr."""
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import torch
        import torchaudio  # already required (kaldi fbank); avoids a librosa dep

        wav = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), sr, target_sr
        ).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32), target_sr


# ---------------------------------------------------------------- checkpoint spec
@dataclass
class Checkpoint:
    cfg: dict          # beats_config (architecture)
    weights: dict      # full model state_dict
    labels: Optional[list] = None       # class names, if a classifier is present
    multi_label: bool = False           # sigmoid vs softmax for the head


def encoder_state_dict(weights: dict) -> dict:
    """BeatsEncoder weights; ESPnet models nest them under 'encoder.' -> strip it."""
    if "patch_embedding.weight" in weights:
        return weights
    if "encoder.patch_embedding.weight" in weights:
        return {k[8:]: v for k, v in weights.items() if k.startswith("encoder.")}
    return weights


def build_classifier(weights: dict, in_features: int):
    """Linear head from the state_dict if present (input dim must match encoder)."""
    import torch

    for key in _CLASSIFIER_KEYS:
        full = next((k for k in weights if k.endswith(key)), None)
        if full is None or weights[full].ndim != 2 or weights[full].shape[1] != in_features:
            continue
        w, bias_key = weights[full], full[:-6] + "bias"
        head = torch.nn.Linear(w.shape[1], w.shape[0], bias=bias_key in weights)
        head.load_state_dict({"weight": w, **({"bias": weights[bias_key]} if bias_key in weights else {})})
        logger.info("Classifier head '%s' -> %d classes.", full, w.shape[0])
        return head
    return None


def _derive_base_repo(name: str) -> Optional[str]:
    """Map a fine-tune name/path to its base SSL repo on HF, or None."""
    m = re.search(r"openbeats?-(base|large)-i([123])", name, re.I)
    if m:
        return f"shikhar7ssu/OpenBEATs-{m.group(1).capitalize()}-i{m.group(2)}"
    m = re.search(r"ear_(base|large).*?beats_iter(\d+)", name, re.I)  # ESPnet path
    if m:
        return f"shikhar7ssu/OpenBEATs-{m.group(1).capitalize()}-i{int(m.group(2)) + 1}"
    return None


def _base_beats_config(base_repo: str) -> dict:
    """Fetch a base SSL repo's beats_config (config.yaml is ~13 KB)."""
    import yaml

    config, _ = find_artifacts(download_checkpoint(base_repo, patterns=["*config.yaml"]))
    if config:
        bc = (yaml.safe_load(open(config)).get("encoder_conf") or {}).get("beats_config")
        if bc:
            return bc
    import torch  # fallback: read the bundled cfg from the (large) base checkpoint

    _, ckpt = find_artifacts(download_checkpoint(base_repo))
    return torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]


def load_checkpoint(checkpoint: str, base: Optional[str] = None) -> Checkpoint:
    """Resolve and load any OpenBEATs checkpoint into a normalized Checkpoint."""
    import torch

    config_path, ckpt_path = _resolve(checkpoint)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found from '{checkpoint}'.")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    logger.info("Loaded %s", os.path.basename(ckpt_path))

    if isinstance(obj, dict) and "cfg" in obj:  # self-contained SSL checkpoint
        return Checkpoint(obj["cfg"], obj.get("model", obj), obj.get("token_list"))

    # Bare ESPnet .pth: architecture comes from the base SSL repo, labels from config.
    if not config_path:
        raise FileNotFoundError("Bare ESPnet checkpoint needs its config.yaml.")
    import yaml

    y = yaml.safe_load(open(config_path)) or {}
    base_repo = (base or _derive_base_repo(checkpoint)
                 or _derive_base_repo((y.get("encoder_conf") or {}).get("beats_ckpt_path") or ""))
    if not base_repo:
        raise ValueError("Pass base='shikhar7ssu/OpenBEATs-<Size>-i<N>'.")
    logger.info("Base architecture from %s", base_repo)
    multi_label = (y.get("model_conf") or {}).get("classification_type") == "multi-label"
    return Checkpoint(_base_beats_config(base_repo), obj, y.get("token_list"), multi_label)


def _resolve(checkpoint: str):
    """(config_path, checkpoint_path) from a file, dir, or HF repo id."""
    if os.path.isfile(checkpoint):
        return None, checkpoint
    if os.path.isdir(checkpoint):
        return find_artifacts(checkpoint)
    logger.info("Resolving %s from Hugging Face ...", checkpoint)
    return find_artifacts(download_checkpoint(checkpoint))
