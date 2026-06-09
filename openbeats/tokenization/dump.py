"""Stage A driver: corpus -> discrete codes -> Parquet shards.

Console script ``openbeats-tokenize`` (``tokenize_main``) -- dump codes for a
manifest of audio. (Dataset inspection lives in ``openbeats.utils.tokens``.)

Heavy imports (torch, the tokenizer) are deferred into the functions so ``--help``
stays fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger("openbeats.tokenize")


# ------------------------------------------------------------------- manifest I/O
def read_manifest(path: str) -> list:
    """Parse a manifest into a list of {"id", "audio"} dicts.

    Each line is either a JSON object with at least ``audio`` (and optionally
    ``id``), or a bare audio path. Missing ids default to the filename stem.
    """
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":
                obj = json.loads(line)
                audio = obj["audio"]
                id_ = str(obj.get("id") or _stem(audio))
            else:
                audio = line
                id_ = _stem(audio)
            items.append({"id": id_, "audio": os.path.abspath(audio)})
    return items


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


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
        "shard %d/%d: %d utterances -> %s", shard_id, num_shards, len(items), out_dir
    )

    tok = build_tokenizer(
        tokenizer_spec,
        device=device,
        seed=seed,
        tokenizer_config=tokenizer_config,
    )
    codebook_size = int(tok.config.quant_n) if tokenizer_spec in (
        None,
        "random",
        "beats_random",
    ) else int(tok.quantize.num_tokens)

    meta = build_meta(
        codebook_size=codebook_size,
        tokenizer={
            "type": "beats_random"
            if tokenizer_spec in (None, "random", "beats_random")
            else "beats",
            "checkpoint": None
            if tokenizer_spec in (None, "random", "beats_random")
            else str(tokenizer_spec),
            "seed": seed,
        },
        codes_offset=CODES_OFFSET,
    )
    writer = TokenDatasetWriter(out_dir, meta, shard_id=shard_id)

    n_done = 0
    for batch in _batched(items, batch_size):
        wavs, ilens, keep = [], [], []
        for it in batch:
            try:
                wav, sr = load_audio(it["audio"])  # mono float32 @ 16 kHz
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
            writer.add(it["id"], it["audio"], ilens[j], seq)
        n_done += len(keep)
        logger.info("  %d/%d", n_done, len(items))

    path = writer.flush()
    logger.info("wrote %s (%d rows)", path, n_done)
    if finalize and shard_id == 0:
        write_dataset_json(out_dir, meta)
    return path


# -------------------------------------------------------------------- CLI: dump
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
    p.add_argument("--manifest", required=True, help="jsonl {id,audio} or one path/line")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--seed", type=int, default=45, help="random tokenizer seed")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument(
        "--no-finalize",
        action="store_true",
        help="do not write dataset.json (each shard still self-describes)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    dump(
        args.tokenizer,
        args.manifest,
        args.out,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        finalize=not args.no_finalize,
    )


if __name__ == "__main__":
    tokenize_main()
