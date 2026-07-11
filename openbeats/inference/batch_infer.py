"""Console entry point openbeats-batch-infer: run OpenBEATs inference over a
manifest of audio segments and persist per-segment outputs to Parquet.

Reads a JSONL manifest ({id, audio, start?, end?}; any 'label' is ignored), runs
the encoder in length-bucketed batches on one device, and writes a single
predictions.parquet (+ predict.json) under --out: per row the mean-pooled probs
and logits when the checkpoint carries a classifier head, and the pooled
embedding when --embeddings is set (or when there is no head). Persisting the
outputs keeps scoring a separate, cheap read step instead of re-running the
encoder each time.
"""

import argparse
import json
import logging
import os

logger = logging.getLogger("openbeats")

def batch_infer(checkpoint, manifest, out, *, device="cpu", batch_bins=3_200_000,
                num_workers=4, max_batch_size=None, embeddings=False,
                base=None, max_layer=None) -> str:
    import torch

    from ..data.labeled import build_infer_dataloader
    from .model import OpenBeats

    model = OpenBeats.from_pretrained(checkpoint, device=device,
                                      max_layer=max_layer, base=base)
    has_head = model.classifier is not None
    want_emb = embeddings or not has_head

    _, loader, _ = build_infer_dataloader(manifest, batch_bins,
                                          num_workers=num_workers,
                                          max_batch_size=max_batch_size)

    rows = []  # eval sets are small; accumulate then write one shard
    with torch.no_grad():
        for batch in loader:
            speech = batch["speech"].to(device)
            ilens = batch["speech_lengths"].to(device)
            rep, olens, _ = model.encoder(speech, ilens, waveform_input=True)
            # masked mean over each clip's valid patches
            mask = torch.arange(rep.size(1), device=device)[None, :] < olens[:, None]
            pooled = (rep * mask.unsqueeze(-1)).sum(1) / olens.clamp(min=1).unsqueeze(-1)

            probs = logits = None
            if has_head:
                logit = model.classifier(pooled)
                prob = torch.sigmoid(logit) if model.multi_label else torch.softmax(logit, -1)
                logits, probs = logit.cpu().numpy(), prob.cpu().numpy()
            emb = pooled.cpu().numpy() if want_emb else None

            for k, rid in enumerate(batch["id"]):
                row = {"id": rid}
                if probs is not None:
                    row["logits"] = logits[k].tolist()
                    row["probs"] = probs[k].tolist()
                    if model.labels and len(model.labels) >= probs.shape[-1]:
                        row["pred_label"] = model.labels[int(probs[k].argmax())]
                if emb is not None:
                    row["embedding"] = emb[k].tolist()
                rows.append(row)

    _write(out, rows, model, has_head, want_emb, checkpoint, manifest)
    logger.info("wrote %d predictions -> %s", len(rows), out)
    return out

def _write(out, rows, model, has_head, want_emb, checkpoint, manifest) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(
            "openbeats-batch-infer needs pyarrow: pip install 'openbeats[tokenize]'"
        )
    os.makedirs(out, exist_ok=True)
    meta = {
        "format": "openbeats-predictions/v1",
        "checkpoint": str(checkpoint),
        "manifest": os.path.abspath(manifest),
        "n_rows": len(rows),
        "has_head": has_head,
        "multi_label": bool(model.multi_label),
        "labels": model.labels,
        "embeddings": want_emb,
    }
    table = pa.Table.from_pylist(rows)
    table = table.replace_schema_metadata({b"openbeats_meta": json.dumps(meta).encode()})
    pq.write_table(table, os.path.join(out, "predictions.parquet"))
    with open(os.path.join(out, "predict.json"), "w") as f:
        json.dump(meta, f, indent=2)

def batch_infer_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="openbeats-batch-infer",
        description="Batch OpenBEATs inference over a manifest -> predictions.parquet.")
    ap.add_argument("--checkpoint", required=True, help="HF repo id, local dir, or file.")
    ap.add_argument("--manifest", required=True,
                    help="JSONL of {id, audio, start?, end?} segments.")
    ap.add_argument("--out", required=True,
                    help="Output directory (predictions.parquet + predict.json).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-bins", type=int, default=3_200_000,
                    help="Summed-sample budget per length-bucketed batch.")
    ap.add_argument("--max-batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--embeddings", action="store_true",
                    help="Also store the pooled encoder embedding per segment.")
    ap.add_argument("--max-layer", type=int, default=None)
    ap.add_argument("--base", default=None, help="Override base SSL repo for a fine-tune.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    batch_infer(args.checkpoint, args.manifest, args.out, device=args.device,
                batch_bins=args.batch_bins, num_workers=args.num_workers,
                max_batch_size=args.max_batch_size, embeddings=args.embeddings,
                base=args.base, max_layer=args.max_layer)
