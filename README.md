# OpenBEATs inference

[OpenBEATs](https://shikhar-s.github.io/OpenBEATs/) is a general-purpose audio
encoder pre-trained on speech, music, environmental sound, and bioacoustics
([paper](https://arxiv.org/abs/2507.14129)). This package runs it on your audio
and gives you patch-level embeddings, plus class probabilities if you point it at
a fine-tuned checkpoint.

## Install

```bash
pip install openbeats
```

You get two commands, `openbeats-infer` and `openbeats-download`. The
dependencies are kept light (torch, torchaudio, numpy, huggingface-hub, pyyaml,
soundfile), and torch is pinned loosely so a build you already have won't be
replaced. If you'd rather not touch your current environment, install it in its
own with [uv](https://docs.astral.sh/uv/) or pipx:

```bash
uv tool install openbeats     # or: pipx install openbeats
```

## Usage

### From the command line

Handy for a quick look:

```bash
openbeats-infer --checkpoint espnet/OpenBEATS-Large-i1-as20k \
    --audio your_audio.wav --out embeddings.npz
```

`--checkpoint` takes a Hugging Face repo id (downloaded for you), a local
directory, or a checkpoint file. The `.npz` holds `patch_embeddings`
`(num_patches, 1024)`, plus `logits` and `probs` when the checkpoint has a
classifier. Other options worth knowing: `--device cuda`, `--max-layer N`, and
`--chunk-seconds 10` for long recordings.

### From Python, on a file

```python
from openbeats.model import OpenBeats

model = OpenBeats.from_pretrained("espnet/OpenBEATS-Large-i1-as20k", device="cuda")
out = model.encode_file("your_audio.wav")          # add chunk_seconds=10 for long audio
print(out["patch_embeddings"].shape)               # (num_patches, 1024)
```

### From Python, on your own waveform

Hand it a 1-D 16 kHz waveform in `[-1, 1]`. If your audio is at another rate,
`load_audio` reads and resamples it for you:

```python
from openbeats.utils import load_audio

wav, sr = load_audio("your_audio.wav")             # mono, 16 kHz
out = model.encode(wav, sr)                         # or pass any numpy array
print(out["patch_embeddings"].shape)
```

## Checkpoints

Browse the variants (Base and Large, plus AudioSet and bioacoustics fine-tunes)
in the [espnet OpenBEATs collection](https://huggingface.co/collections/espnet/openbeats).

## Development

```bash
uv sync                                  # install with dev deps (pytest)
uv run pytest                            # unit tests, no downloads
OPENBEATS_INTEGRATION=1 uv run pytest    # also run the end-to-end tests
uv build                                 # build wheel + sdist into dist/
uv publish                               # publish to PyPI
```
