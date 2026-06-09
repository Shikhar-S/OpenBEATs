"""Stage A driver: corpus -> discrete codes -> Parquet shards.

Console script ``openbeats-tokenize`` (``tokenize_main``) -- dump codes for a
manifest of audio segments. (Dataset inspection lives in ``openbeats.utils.tokens``.)

The manifest may carry ``start``/``end`` per entry; only that span is read (at the
file's native rate) and tokenized, so long recordings need no pre-cutting. fbank
normalization stats come from the run config (``encoder_conf.fbank_mean/std``) and
are recorded in the dataset metadata so training can assert they match.

Heavy imports (torch, the tokenizer) are deferred into the functions so ``--help``
stays fast.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

from ..data.manifest import read_manifest

logger = logging.getLogger("openbeats.tokenize")

_RANDOM = (None, "random", "beats_random")
DEFAULT_FBANK_MEAN = 15.41663
DEFAULT_FBANK_STD = 6.55582


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# ------------------------------------------------------------------------- dump
def dump(
    tokenizer_spec,
    manifest,
    out_dir,
    *,
    seed=45,
    device="cpu",
    batch_size=16,
    num_shards=1,
    shard_id=0,
    tokenizer_config=None,
    fbank_mean=DEFAULT_FBANK_MEAN,
    fbank_std=DEFAULT_FBANK_STD,
    finalize=True,
):
    import numpy as np
    import torch

    from ..data.audio import load_audio
    from ..data.dataset import (
        CODES_OFFSET,
        TokenDatasetWriter,
        build_meta,
        write_dataset_json,
    )
    from ..nets.tokenizer import build_tokenizer

    items = read_manifest(manifest)
    # Disjoint round-robin slice so parallel workers balance long/short clips.
    items = items[shard_id::num_shards]
    logger.info(
        "shard %d/%d: %d segments -> %s", shard_id, num_shards, len(items), out_dir
    )

    tok = build_tokenizer(
        tokenizer_spec,
        device=device,
        seed=seed,
        tokenizer_config=tokenizer_config,
        fbank_mean=fbank_mean,
        fbank_std=fbank_std,
    )
    is_random = tokenizer_spec in _RANDOM
    codebook_size = int(tok.config.quant_n) if is_random else int(tok.quantize.num_tokens)

    meta = build_meta(
        codebook_size=codebook_size,
        tokenizer={
            "type": "beats_random" if is_random else "beats",
            "checkpoint": None if is_random else str(tokenizer_spec),
            "seed": seed,
        },
        codes_offset=CODES_OFFSET,
        fbank_mean=fbank_mean,
        fbank_std=fbank_std,
    )
    writer = TokenDatasetWriter(out_dir, meta, shard_id=shard_id)

    n_done = 0
    for batch in _batched(items, batch_size):
        wavs, ilens, keep = [], [], []
        for it in batch:
            try:
                wav, sr = load_audio(it["audio"], start=it.get("start"), end=it.get("end"))
            except Exception as e:  # noqa: BLE001 - skip unreadable files, keep going
                logger.warning("skip %s (%s)", it["audio"], e)
                continue
            wavs.append(torch.from_numpy(wav))
            ilens.append(len(wav))
            keep.append(it)
        if not wavs:
            continue

        tmax = max(ilens)
        xs = torch.zeros(len(wavs), tmax, dtype=torch.float32)
        for j, w in enumerate(wavs):
            xs[j, : w.numel()] = w
        xs = xs.to(device)
        lens = torch.tensor(ilens, dtype=torch.long, device=device)

        with torch.no_grad():
            out = tok.encode(xs, lens, waveform_input=True)
        codes = out["codes"].cpu().numpy()
        clens = out["code_lengths"].cpu().numpy()

        for j, it in enumerate(keep):
            seq = codes[j, : int(clens[j])].astype(np.int64) + CODES_OFFSET
            writer.add(it["id"], it["audio"], ilens[j], seq,
                       start=it.get("start"), end=it.get("end"))
        n_done += len(keep)
        logger.info("  %d/%d", n_done, len(items))

    path = writer.flush()
    logger.info("wrote %s (%d rows)", path, n_done)
    if finalize and shard_id == 0:
        write_dataset_json(out_dir, meta)
        shutil.copyfile(manifest, os.path.join(out_dir, "manifest.jsonl"))
    return path


# -------------------------------------------------------------------- CLI: dump
def _resolve_fbank(config_path, cli_mean, cli_std):
    """fbank stats: explicit CLI flag > run config's encoder_conf > package default."""
    mean, std = DEFAULT_FBANK_MEAN, DEFAULT_FBANK_STD
    if config_path:
        import yaml

        with open(config_path) as f:
            enc = (yaml.safe_load(f) or {}).get("encoder_conf") or {}
        mean = enc.get("fbank_mean", mean)
        std = enc.get("fbank_std", std)
    if cli_mean is not None:
        mean = cli_mean
    if cli_std is not None:
        std = cli_std
    return float(mean), float(std)


def tokenize_main(argv=None):
    p = argparse.ArgumentParser(
        prog="openbeats-tokenize",
        description="Tokenize an audio corpus into a Parquet code dataset (stage A).",
    )
    p.add_argument(
        "--tokenizer",
        default="random",
        help="'random' (BestRQ, no checkpoint) or a BeatsTokenizer "
        "checkpoint path / local dir / HF repo id.",
    )
    p.add_argument("--manifest", required=True,
                   help="jsonl {id,audio,start?,end?} or one path/line")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--config", default=None,
                   help="run config YAML supplying encoder_conf.fbank_mean/std")
    p.add_argument("--fbank-mean", type=float, default=None, help="override fbank mean")
    p.add_argument("--fbank-std", type=float, default=None, help="override fbank std")
    p.add_argument("--seed", type=int, default=45, help="random tokenizer seed")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument(
        "--no-finalize",
        action="store_true",
        help="do not write dataset.json/manifest.jsonl (each shard still self-describes)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    fbank_mean, fbank_std = _resolve_fbank(args.config, args.fbank_mean, args.fbank_std)
    logger.info("fbank stats: mean=%.5f std=%.5f", fbank_mean, fbank_std)
    dump(
        args.tokenizer,
        args.manifest,
        args.out,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        fbank_mean=fbank_mean,
        fbank_std=fbank_std,
        finalize=not args.no_finalize,
    )


if __name__ == "__main__":
    tokenize_main()
