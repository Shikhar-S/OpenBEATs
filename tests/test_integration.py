"""End-to-end test that downloads from Hugging Face.

Opt-in (it pulls a ~1.2 GB checkpoint): run with OPENBEATS_INTEGRATION=1.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENBEATS_INTEGRATION"),
    reason="set OPENBEATS_INTEGRATION=1 to run (downloads from HF)",
)


def _tone(seconds=3, sr=16000):
    t = np.linspace(0, seconds, seconds * sr, endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


def test_ssl_encoder_embeddings():
    from openbeats.model import OpenBeats

    model = OpenBeats.from_pretrained("shikhar7ssu/OpenBEATs-Large-i2")
    out = model.encode(*_tone())
    assert out["patch_embeddings"].shape[1] == 1024
    assert "logits" not in out  # SSL encoder has no classifier head


def test_finetune_classification_logits():
    from openbeats.model import OpenBeats

    model = OpenBeats.from_pretrained("espnet/OpenBEATS-Large-i2-as20k")
    out = model.encode(*_tone())
    assert out["patch_embeddings"].shape[1] == 1024
    assert out["probs"].shape == (527,)
    # a 440 Hz sine should classify as a sine wave
    assert out["top_label"] == "Sine_wave"
