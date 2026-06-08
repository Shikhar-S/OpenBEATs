# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`openbeats` is a package for the OpenBEATs audio encoder
(paper: https://arxiv.org/abs/2507.14129). Its core is **inference**: load a
pre-trained or fine-tuned checkpoint into a vendored ESPnet/Microsoft
`BeatsEncoder` and return patch-level embeddings `(num_patches, 1024)`, plus
classification logits/probs when the checkpoint carries a head. The default
install is deliberately minimal (torch pinned with lower bounds only, so it won't
clobber a user's CUDA-matched build).

It **also** ships an end-to-end pretraining pipeline behind optional extras (no
ESPnet dependency): tokenize a corpus to discrete codes (stage A), pretrain the
encoder with a slim DeepSpeed loop (stage B), and export a `{"model","cfg"}`
inference checkpoint (stage C).

## Commands

```bash
uv sync                                  # install with dev deps (pytest + einops/pyarrow/typeguard)
uv run pytest                            # unit tests — no network/downloads
uv run pytest tests/test_encoder.py::test_chunked_encoding   # single test
OPENBEATS_INTEGRATION=1 uv run pytest    # also runs end-to-end tests (pulls ~1.2 GB from HF)
uv build                                 # wheel + sdist into dist/
uv publish                               # publish to PyPI
```

There is no linter configured. Console entry points (`pyproject.toml
[project.scripts]`): `openbeats-infer`, `openbeats-download` (inference);
`openbeats-tokenize`/`openbeats-tokens` (stage A, `tokenize` extra),
`openbeats-pretrain` (stage B, `train` extra), `openbeats-convert` (stage C).
Extras: `tokenize = [pyarrow, einops]`, `train = [deepspeed, wandb, typeguard, …]`.

Stage B launches under torchrun, e.g.:
```bash
torchrun --standalone --nproc_per_node=8 -m openbeats.pretrain.train \
    --config src/openbeats/conf/pretrain_large.yaml \
    --deepspeed_config src/openbeats/conf/ds_openbeats_large.json \
    --train_data data/tokens_train --valid_data data/tokens_valid \
    --output_dir exp/openbeats_large
# `--no-deepspeed` uses the PlainEngine fallback (CPU / no DeepSpeed; for smoke tests).
```

## Architecture

The flow is: **checkpoint resolution → normalized `Checkpoint` → `BeatsEncoder` +
optional classifier head → `OpenBeats` inference wrapper**.

- `utils.py` — all the plumbing. The key abstraction is `load_checkpoint()`, which
  normalizes **two on-disk checkpoint formats** into one `Checkpoint` dataclass
  (`cfg`, `weights`, `labels`, `multi_label`):
  - **Self-contained SSL encoder** (`shikhar7ssu/OpenBEATs-*`): a single dict with
    `{"cfg", "model"}` — architecture travels with the weights.
  - **Bare ESPnet fine-tune** (`espnet/OpenBEATS-*-<task>`): a `.pth` of weights +
    a `config.yaml`, but the architecture is **not** in either. It must be fetched
    from the corresponding *base* SSL repo, mapped by `_derive_base_repo()` (regex
    on the repo/path name; `--base` overrides if it can't be derived). Labels come
    from the config's `token_list`; `multi-label` vs single-label selects
    sigmoid vs softmax for the head.
  - `encoder_state_dict()` strips ESPnet's `encoder.` prefix; `build_classifier()`
    pulls a linear head from the state dict only if its input dim matches the
    encoder output (guards against e.g. an MLM decoder being mistaken for a head).
- `model.py` — `OpenBeats`. `from_pretrained()` builds the encoder from the
  resolved `Checkpoint` (loads with `strict=False`, warns on non-`_pad` missing
  keys). `encode()` takes a **1-D 16 kHz** waveform and refuses other sample rates
  (resample via `utils.load_audio()` first). `chunk_seconds` windows long audio and
  concatenates patches to bound memory.
- `cli.py` — thin argparse wrappers; imports are deferred into the functions to
  keep `--help` fast.
- `beats_encoder.py` — **vendored, 2000+ lines, adapted from Microsoft BEATs /
  ESPnet. Treat as third-party; do not refactor casually.** It already contains the
  full pretraining stack (`BeatsEncoder(is_pretraining=True)`, `mask_sequence`,
  `BeatsPretrainingPredictor`). SpecAug and the wav2vec2-conformer /
  learned-positional-embedding adapter paths are training- or variant-only and not
  exercised by standard OpenBEATs inference (the `transformers` dependency is
  optional, gated behind the `adapter` extra).

### Pretraining pipeline (stages A→C)

- **Vendored modeling code is byte-identical to ESPnet.** `beats_utils.py`
  (VQ/k-means/`force_gatherable` helpers), `tokenizer.py` (`BeatsTokenizer` +
  `BeatsRandomTokenizer` + quantizers), and `pretrain/model.py` (`BeatsPretrainModel`
  + `MixupAugment`) are copied verbatim from the ESPnet source named in each file's
  provenance header; the only permitted changes are repointed imports and swapping
  ESPnet base classes for `nn.Module`. **When editing these files, keep them in
  lockstep with upstream** (diff against the source/commit in the header). Glue code
  (`tokenize/`, `pretrain/{data,trainer,train}.py`, `convert_checkpoint.py`, configs)
  is fresh, not vendored.
- `tokenize/` — stage A. `schema.py` is the Parquet token-dataset format (one row =
  one utterance: `id, audio, n_samples, n_codes, codes:list<int16>`; dataset config
  in Parquet key-value metadata + a `dataset.json` mirror). `dump.py` drives
  corpus→codes→shards. **Codes are stored shifted to `1..K`** (the `<unk>`+1 shift;
  index 0 reserved) so the verbatim model's `target-1`/`ignore_id=-2` works and the
  data layer does no arithmetic.
- `pretrain/data.py` — `TokenDataset` loads the waveform from each row's `audio`
  path (fbank recomputed at train time), `collate` pads speech with `0.0` and
  targets with `-1`, and `LengthBucketBatchSampler` buckets by length and shards
  batches `[rank::world_size]`.
- `pretrain/trainer.py` — slim loop around a DeepSpeed engine (DeepSpeed owns
  optimizer/LR/bf16/clip/accum/checkpointing/logging from the JSON). Keeps the one
  numerical detail `loss/weight*world_size` and an `iterator_stop` all-reduce. A
  `PlainEngine` fallback mirrors DeepSpeed's checkpoint layout (`global_step{N}/
  mp_rank_00_model_states.pt` with a `module` key + `latest`) for CPU/no-DeepSpeed.
- `convert_checkpoint.py` — stage C. Reads `["module"]` (ZeRO-1 keeps full weights),
  keeps `encoder.`-prefixed keys (strip + float32), averages if multiple, attaches
  `beats_config`, writes `{"model","cfg"}` — the same format `OpenBeats.from_pretrained`
  loads. So stage C output flows straight back into the inference path.

## Conventions / gotchas

- Embedding dim is **1024** for the real checkpoints; tests use a `TINY` 64-dim
  config so the encoder can run without any download.
- Unit tests must stay **offline**. Anything that hits Hugging Face goes behind the
  `OPENBEATS_INTEGRATION` env guard in `test_integration.py`.
- Downloads default into `./checkpoints/<repo-name>/` and are filtered by
  `ALLOW_PATTERNS` to avoid pulling unrelated repo files.
- `DEFAULT_REPO` (`espnet/OpenBEATS-Large-i2-as20k`) is the default for
  `openbeats-download` and the documented inference checkpoint.
- The pretraining tests stay offline too: they use the **random BestRQ tokenizer**
  (`build_tokenizer("random", seed=…)`, seed-deterministic, no checkpoint) + the
  `TINY` encoder, and the end-to-end gate (`tests/test_e2e.py`) runs
  dump→pretrain→convert→infer on a few synthetic clips via the `PlainEngine`.
- Patch-count alignment (tokenizer codes ↔ encoder patches) is structural — both
  share the BEATs fbank+patch frontend — and is checked in
  `tests/test_tokenizer.py` / `tests/test_data.py`. Keep `waveform_input` and fbank
  params single-sourced from dataset metadata.
- **Next milestone: acoustic tokenizer *training*** (out of scope for v1). The EMA-VQ
  / k-means code in `tokenizer.py`/`beats_utils.py` is dead at inference but kept
  verbatim for it.
