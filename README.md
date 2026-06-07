# OpenBEATs inference

Run inference with [OpenBEATs](https://shikhar-s.github.io/OpenBEATs/), a
general-purpose audio encoder pre-trained on speech, music, environmental sound,
and bioacoustics ([paper](https://arxiv.org/abs/2507.14129)). Given an audio
file, it produces patch-level embeddings.

## Install

```bash
pip install openbeats
```

This installs the `openbeats-infer` / `openbeats-download` commands. Dependencies
are kept lean (torch, torchaudio, numpy, huggingface-hub, pyyaml, soundfile) and
declared with lower bounds, so an existing torch install is reused rather than
reinstalled. For a fully isolated CLI that doesn't touch your environment, use
[`uv tool`](https://docs.astral.sh/uv/) or `pipx`:

```bash
uv tool install openbeats     # or: pipx install openbeats
```

## Usage

### CLI — quick prototyping

```bash
openbeats-infer --checkpoint espnet/OpenBEATS-Large-i1-as20k \
    --audio your_audio.wav --out embeddings.npz
```

`--checkpoint` accepts a Hugging Face repo id (auto-downloaded), a local
directory, or a checkpoint file. The `.npz` holds `patch_embeddings`
`(num_patches, 1024)` (plus `logits`/`probs` for classification checkpoints).
Options: `--device cuda`, `--max-layer N`, `--chunk-seconds 10` (long audio).

### Python — from an audio file

```python
from openbeats.model import OpenBeats

model = OpenBeats.from_pretrained("espnet/OpenBEATS-Large-i1-as20k", device="cuda")
out = model.encode_file("your_audio.wav")          # or chunk_seconds=10 for long audio
print(out["patch_embeddings"].shape)               # (num_patches, 1024)
```

### Python — from your own waveform

Pass a 1-D 16 kHz waveform in `[-1, 1]` (use `load_audio` for other rates):

```python
import numpy as np
from openbeats.utils import load_audio

wav, sr = load_audio("your_audio.wav")             # any rate -> mono 16 kHz
out = model.encode(wav, sr)                         # or pass your own np.ndarray
print(out["patch_embeddings"].shape)               # (num_patches, 1024)
```

## Checkpoints

Browse variants (Base/Large, AudioSet and bioacoustics fine-tunes) in the
[espnet OpenBEATs collection](https://huggingface.co/collections/espnet/openbeats).

## Development

```bash
uv sync                       # install with dev deps (pytest)
uv run pytest                 # unit tests (no downloads)
OPENBEATS_INTEGRATION=1 uv run pytest   # + end-to-end (downloads from HF)
uv build                      # build wheel + sdist into dist/
uv publish                    # publish to PyPI
```
