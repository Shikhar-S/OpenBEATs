"""Console script ``openbeats-tokens``: inspect / stats / validate a token dataset."""

from __future__ import annotations

import argparse
import json


def tokens_main(argv=None):
    p = argparse.ArgumentParser(
        prog="openbeats-tokens",
        description="Inspect / validate an OpenBEATs token dataset.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("inspect", help="print one row and its codes")
    sp.add_argument("dataset")
    sp.add_argument("--id", default=None, help="utterance id (default: first row)")

    ss = sub.add_parser("stats", help="dataset-level statistics")
    ss.add_argument("dataset")

    sv = sub.add_parser("validate", help="check dtypes, code range, lengths")
    sv.add_argument("dataset")
    sv.add_argument("--check-audio", action="store_true", help="also check paths exist")

    args = p.parse_args(argv)
    from ..data import dataset as schema

    if args.cmd == "inspect":
        row = None
        for r in schema.iter_rows(args.dataset):
            if args.id is None or r["id"] == args.id:
                row = r
                break
        if row is None:
            print(f"id {args.id!r} not found")
            return 1
        codes = row["codes"]
        head = codes[:32]
        print(json.dumps({k: row[k] for k in ("id", "audio", "n_samples", "n_codes")}, indent=2))
        print(f"codes[:32] = {head}")
    elif args.cmd == "stats":
        meta = schema.load_meta(args.dataset)
        st = schema.validate(args.dataset)
        used = _coverage(args.dataset)
        print(json.dumps({**meta, **st, "codes_used": used}, indent=2))
    elif args.cmd == "validate":
        st = schema.validate(args.dataset, check_audio_exists=args.check_audio)
        print("OK", json.dumps(st))
    return 0


def _coverage(path):
    """Number of distinct code values used (codebook coverage)."""
    import pyarrow.compute as pc

    from ..data import dataset as schema

    table = schema.read_table(path, columns=["codes"])
    flat = pc.list_flatten(table["codes"].combine_chunks())
    return len(pc.unique(flat))
