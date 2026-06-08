# OpenBEATs

[OpenBEATs](https://arxiv.org/abs/2507.14129) is a general-purpose audio
encoder pre-trained on speech, music, environmental sound, and bioacoustics.
This package runs it on audio and
returns patch-level embeddings, plus class probabilities when a fine-tuned
checkpoint is used.

## Install

```bash
pip install openbeats
```

This adds two commands, `openbeats-infer` and `openbeats-download`. The
dependencies are kept light (torch, torchaudio, numpy, huggingface-hub, pyyaml,
soundfile), and torch is pinned loosely so an existing build is not replaced. To
avoid touching an existing environment, install it in its own with
[uv](https://docs.astral.sh/uv/) or pipx:

```bash
uv tool install openbeats     # or: pipx install openbeats
```

## Usage

### From the command line

Handy for a quick look:

```bash
openbeats-infer --checkpoint espnet/OpenBEATS-Large-i2-as20k \
    --audio audio.wav --out embeddings.npz
```

`--checkpoint` takes a Hugging Face repo id (downloaded automatically), a local
directory, or a checkpoint file. The `.npz` holds `patch_embeddings`
`(num_patches, 1024)`, plus `logits` and `probs` when the checkpoint has a
classifier. Other options: `--device cuda`, `--max-layer N`, and
`--chunk-seconds 10` for long recordings.

### From Python

```python
from openbeats.model import OpenBeats
from openbeats.utils import load_audio

# load model
model = OpenBeats.from_pretrained("espnet/OpenBEATS-Large-i2-as20k", device="cuda")

# from a file with any sample rate
out = model.encode_file("audio.wav")              # pass chunk_seconds=10 for long audio

# or load the waveform in 16khz monoaural array with values in [-1,1]
wav, sr = load_audio("audio.wav")
# and pass it
out = model.encode(wav, sr)

print(out["patch_embeddings"].shape)               # (num_patches, 1024)
```

## Checkpoints

The variants (Base and Large, plus AudioSet and bioacoustics fine-tunes) live in
the [espnet OpenBEATs collection](https://huggingface.co/collections/espnet/openbeats).

## Pretraining (optional)

Beyond inference, the package can tokenize a corpus and pretrain the encoder from
scratch — no ESPnet dependency. Install the extras you need:

```bash
pip install "openbeats[tokenize]"   # stage A only (pyarrow, einops)
pip install "openbeats[train]"      # stages A–C (adds deepspeed, wandb, …)
```

```bash
# A. tokenize a corpus -> a Parquet code dataset
#    (use a real BeatsTokenizer checkpoint, or the seed-deterministic random one)
openbeats-tokenize --tokenizer random --seed 45 \
    --manifest corpus.jsonl --out data/tokens_train     # jsonl: {"id","audio"} or a path/line
openbeats-tokens stats data/tokens_train                # inspect / validate

# B. pretrain the encoder (torchrun; single- or multi-node; auto-resumes)
torchrun --standalone --nproc_per_node=8 -m openbeats.pretrain.train \
    --config conf/pretrain_large.yaml \
    --deepspeed_config conf/ds_openbeats_large.json \
    --train_data data/tokens_train --valid_data data/tokens_valid \
    --output_dir exp/openbeats_large

# C. export a lightweight inference checkpoint, then use it
openbeats-convert --train_ckpt exp/openbeats_large --out openbeats_large_pretrained.pt
openbeats-infer --checkpoint openbeats_large_pretrained.pt --audio audio.wav
```

The masked-acoustic-modeling encoder/predictor and the acoustic tokenizer are
vendored byte-for-byte from ESPnet (so pretrained encoders match the published
ones); the dataset, DeepSpeed loop, and CLIs are a small fresh layer.

## Development

```bash
uv sync                                  # install with dev deps (pytest)
uv run pytest                            # unit tests, no downloads
OPENBEATS_INTEGRATION=1 uv run pytest    # also run the end-to-end tests
uv build                                 # build wheel + sdist into dist/
uv publish                               # publish to PyPI
```
