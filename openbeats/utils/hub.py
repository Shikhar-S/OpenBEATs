"""Hugging Face downloads for OpenBEATs checkpoints, plus the
openbeats-download console script."""

import argparse
import os

DEFAULT_REPO = "espnet/OpenBEATS-Large-i2-as20k"
ALLOW_PATTERNS = ["*config.yaml", "*epoch*.pt", "*.pth", "*.ckpt"]

def download_checkpoint(repo_id=DEFAULT_REPO, dest=None, patterns=None) -> str:
    """Download (part of) a repo; return the local snapshot directory."""
    from huggingface_hub import snapshot_download

    dest = dest or os.path.join("checkpoints", repo_id.split("/")[-1])
    return snapshot_download(repo_id, allow_patterns=patterns or ALLOW_PATTERNS,
                             local_dir=dest)

def find_artifacts(snapshot_dir):
    """Return (config_path, checkpoint_path), either may be None."""
    config = ckpt = None
    for root, _, files in os.walk(snapshot_dir):
        for f in files:
            if f == "config.yaml":
                config = config or os.path.join(root, f)
            elif f.endswith((".pt", ".pth", ".ckpt")):
                ckpt = ckpt or os.path.join(root, f)
    return config, ckpt

def download_main(argv=None):
    ap = argparse.ArgumentParser(prog="openbeats-download",
                                 description="Download an OpenBEATs checkpoint from HF.")
    ap.add_argument("repo_id", nargs="?", default=DEFAULT_REPO)
    ap.add_argument("--dest", default=None)
    args = ap.parse_args(argv)

    config, ckpt = find_artifacts(download_checkpoint(args.repo_id, args.dest))
    print(f"Downloaded {args.repo_id}\n  config:     {config}\n  checkpoint: {ckpt}")
