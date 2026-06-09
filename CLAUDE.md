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
[project.scripts]`): `openbeats-infer` (inference), `openbeats-download` (utility);
`openbeats-tokenize` (stage A, `tokenize` extra), `openbeats-tokens` (utility, dataset
inspect), `openbeats-train-encoder` (stage B, `train` extra), `openbeats-convert`
(utility, stage C). Extras: `tokenize = [pyarrow, einops]`, `train = [deepspeed,
wandb, typeguard, …]`.

Stage B launches under torchrun, e.g.:
```bash
torchrun --standalone --nproc_per_node=8 -m openbeats.train \
    --config openbeats/conf/pretrain_large.yaml \
    --deepspeed_config openbeats/conf/ds_openbeats_large.json \
    --train_data data/tokens_train --valid_data data/tokens_valid \
    --output_dir exp/openbeats_large
# `--no-deepspeed` uses the PlainEngine fallback (CPU / no DeepSpeed; for smoke tests).
```

## Architecture

**Flat package layout** (no `src/`): the package dir `openbeats/` sits at the repo
root. Subpackages: `nets/` (all `nn.Module` code: `encoder.py`←BEATs encoder,
`tokenizer.py`, `pretrain_model.py`, `beats_utils.py`), `data/` (shared data layer:
`manifest.py` JSONL segment manifest, `audio.py` segment-aware load+resample,
`dataset.py` Parquet token schema, `loader.py` torch Dataset/sampler/collate),
`utils/` (cross-cutting + utility CLIs: `hub.py` HF
download, `checkpoint.py` loader, `convert.py` stage C, `tokens.py` inspect),
`inference/` (`model.py` `OpenBeats`, `run_inference.py` CLI), `pretraining/`
(`engine.py` encoder-train build), `tokenization/` (`dump.py` stage A). A **common
`train.py`** at the package root owns the objective-agnostic loop + engine setup +
`PlainEngine`, driving a pluggable `engine` (`build_model`/`build_dataloaders`).

The inference flow is: **checkpoint resolution → normalized `Checkpoint` →
`BeatsEncoder` + optional classifier head → `OpenBeats` inference wrapper**.

- `utils/checkpoint.py` + `utils/hub.py` — the plumbing. The key abstraction is
  `load_checkpoint()` (in `checkpoint.py`), which
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
- `inference/model.py` — `OpenBeats`. `from_pretrained()` builds the encoder from the
  resolved `Checkpoint` (loads with `strict=False`, warns on non-`_pad` missing
  keys). `encode()` takes a **1-D 16 kHz** waveform and refuses other sample rates
  (resample via `data.audio.load_audio()` first). `chunk_seconds` windows long audio
  and concatenates patches to bound memory.
- CLI entry points (`inference/run_inference.py`, `tokenization/dump.py`,
  `utils/{hub,convert,tokens}.py`, `train.py`) — thin argparse wrappers; imports are
  deferred into the functions to keep `--help` fast.
- `nets/encoder.py` — **vendored, 2000+ lines, adapted from Microsoft BEATs /
  ESPnet. Treat as third-party; do not refactor casually.** It already contains the
  full pretraining stack (`BeatsEncoder(is_pretraining=True)`, `mask_sequence`,
  `BeatsPretrainingPredictor`). SpecAug and the wav2vec2-conformer /
  learned-positional-embedding adapter paths are training- or variant-only and not
  exercised by standard OpenBEATs inference (the `transformers` dependency is
  optional, gated behind the `adapter` extra).

### Pretraining pipeline (stages A→C)

- **Vendored modeling code is byte-identical to ESPnet.** `nets/beats_utils.py`
  (VQ/k-means/`force_gatherable` helpers), `nets/tokenizer.py` (`BeatsTokenizer` +
  `BeatsRandomTokenizer` + quantizers), and `nets/pretrain_model.py`
  (`BeatsPretrainModel` + `MixupAugment`) are copied verbatim from the user's ESPnet
  fork (branch `audioverse_copy`, commit `0c3c8ca`); the files carry no provenance
  header. The only permitted changes are repointed imports and swapping ESPnet base
  classes for `nn.Module`. **When editing these files, keep their function/class
  bodies in lockstep with that upstream.** Glue code (`tokenization/dump.py`,
  `data/{dataset,loader}.py`, `train.py`,
  `pretraining/engine.py`, `utils/convert.py`, configs) is fresh, not vendored.
- `data/manifest.py` — the corpus-agnostic input contract: JSONL of audio segments
  `{id, audio, start?, end?}` (omit span ⇒ whole file). `read_manifest`/`write_manifest`/
  `normalize_entry`. Corpus adapters (e.g. `recipes/watkins/prepare_manifest.py`,
  Kaldi `wav.scp`+`segments`) emit this; tokenize and the loader both consume it.
- `tokenization/dump.py` — stage A driver. `data/dataset.py` is the Parquet
  token-dataset format v2 (one row = one segment: `id, audio, start?, end?, n_samples,
  n_codes, codes:list<int16>`; dataset config incl. **fbank stats** in Parquet
  key-value metadata + `dataset.json` + a `manifest.jsonl` mirror, so the dir is
  self-contained). `dump.py` reads only each segment's span (native-rate seek →
  resample), takes fbank stats from `--config` (`encoder_conf.fbank_mean/std`), and
  records them in the metadata. **Codes are stored shifted to `1..K`** (the `<unk>`+1
  shift; index 0 reserved) so the verbatim model's `target-1`/`ignore_id=-2` works and
  the data layer does no arithmetic. (`start`/`end` are nullable; a v1 dataset without
  them reads as whole-file.)
- `data/loader.py` — `TokenDataset` loads each row's **span** (`start`/`end` → the
  segment, else the whole file) from its `audio` path (fbank recomputed at train
  time), `collate` pads speech with `0.0` and
  targets with `-1`, and `LengthBucketBatchSampler` buckets by length and shards
  batches `[rank::world_size]` (dropping the remainder to keep per-rank batch counts
  equal — DeepSpeed deadlocks otherwise).
- `train.py` + `pretraining/engine.py` — `train.py` is the slim, objective-agnostic
  loop around a DeepSpeed engine (DeepSpeed owns optimizer/LR/bf16/clip/accum/
  checkpointing/logging from the JSON); `engine.py` supplies `build_model` +
  `build_dataloaders`. `train.py` keeps the one numerical detail
  `loss/weight*world_size` and an `iterator_stop` all-reduce. Its `PlainEngine`
  fallback mirrors DeepSpeed's checkpoint layout (`global_step{N}/
  mp_rank_00_model_states.pt` with a `module` key + `latest`) for CPU/no-DeepSpeed.
  **bf16 is via DeepSpeed `torch_autocast`, not pure `"bf16"`** (the JSON configs):
  `waveform_input: true` means the frontend runs `ta_kaldi` in fp32 (it can't do
  bf16), so params stay fp32 and the forward autocasts — the model's
  `with autocast(False)` around the fbank then keeps fbank fp32 and the rest bf16.
  Pure-bf16 would make the `patch_embedding` conv weights bf16 and crash on the
  fp32 fbank. The configs also set `torch_adam: true` so the optimizer needs no
  FusedAdam JIT (the cluster's nvcc can lag torch's CUDA). Verified on a GH200.
- `utils/convert.py` — stage C. Reads `["module"]` (ZeRO-1 keeps full weights),
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
  `tests/test_tokenizer.py` / `tests/test_data.py` (and per-segment in
  `tests/test_segments.py`). **fbank stats are single-sourced from the run config**:
  `openbeats-tokenize --config` records `encoder_conf.fbank_mean/std` in the dataset
  metadata, and `pretraining/engine._check_dataset_compat` fails fast if the training
  run config's stats (or codebook size) don't match the dataset.
- Corpus recipes live in `recipes/<corpus>/` (**gitignored**): a `prepare_manifest.py`
  adapter → `conf/{encoder_*.yaml, ds_*.json}` → `*.slurm` launchers. `recipes/watkins/`
  is the reference (Kaldi `wav.scp`+`segments` → manifest → 4-GPU tokenize/train on
  `ghx4`, account `bbjs-dtai-gh`). The package ships no run configs.
- **Next milestone: acoustic tokenizer *training*** (out of scope for v1). The EMA-VQ
  / k-means code in `nets/tokenizer.py`/`nets/beats_utils.py` is dead at inference but
  kept verbatim for it; it will gain a `tokenization/engine.py` driven by the same
  common `train.py`.
