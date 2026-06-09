"""Console entry point openbeats-infer: run OpenBEATs inference on an audio
file and print (optionally save) patch embeddings + classification probs."""

import argparse
import logging

def infer_main(argv=None):
    import numpy as np

    from .model import OpenBeats

    ap = argparse.ArgumentParser(prog="openbeats-infer",
                                 description="Run OpenBEATs inference on an audio file.")
    ap.add_argument("--checkpoint", required=True, help="HF repo id, local dir, or file.")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default=None, help="Optional .npz output path.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-layer", type=int, default=None)
    ap.add_argument("--chunk-seconds", type=float, default=None,
                    help="Process long audio in fixed windows of this many seconds.")
    ap.add_argument("--base", default=None, help="Override base SSL repo for a fine-tune.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    model = OpenBeats.from_pretrained(args.checkpoint, device=args.device,
                                      max_layer=args.max_layer, base=args.base)
    out = model.encode_file(args.audio, chunk_seconds=args.chunk_seconds)

    print(f"patch_embeddings: {out['patch_embeddings'].shape}")
    if "probs" in out:
        i = int(np.argmax(out["probs"]))
        print(f"top: {out.get('top_label', i)}  p={out['probs'][i]:.4f}")
    else:
        print("(no classifier head; embeddings only)")

    if args.out:
        np.savez(args.out, **out)
        print(f"saved -> {args.out}")
