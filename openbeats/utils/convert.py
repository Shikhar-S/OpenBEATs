"""Stage C: export a lightweight inference checkpoint from a training checkpoint.

Console script ``openbeats-convert``. Reads a DeepSpeed (or PlainEngine) training
checkpoint -- the full model lives under ``["module"]`` (ZeRO-1 keeps it whole, so
no fp32 consolidation is needed) -- keeps the ``encoder.``-prefixed weights (strip
prefix, cast float32), attaches the run's ``beats_config``, and saves the
``{"model", "cfg"}`` file that ``OpenBeats.from_pretrained`` loads. Multiple
checkpoints are averaged.

    openbeats-convert \\
        --train_ckpt exp/openbeats_large \\        # dir (resolves `latest`) or a
                                                    # .../global_stepN/mp_rank_00_model_states.pt
        --config exp/openbeats_large/config.yaml \\ # default: <rundir>/config.yaml
        --out openbeats_large_pretrained.pt
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger("openbeats.convert")

MODEL_STATES = "mp_rank_00_model_states.pt"


def resolve_ckpt(path: str) -> str:
    """Resolve a training-checkpoint path to a concrete model-states file.

    Accepts the states file itself, a ``global_step{N}`` directory, or a run
    directory containing a DeepSpeed ``latest`` file.
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        direct = os.path.join(path, MODEL_STATES)
        if os.path.isfile(direct):
            return direct
        latest = os.path.join(path, "latest")
        if os.path.isfile(latest):
            tag = open(latest).read().strip()
            cand = os.path.join(path, tag, MODEL_STATES)
            if os.path.isfile(cand):
                return cand
    raise FileNotFoundError(f"No training checkpoint resolved from '{path}'.")


def extract_encoder_state_dict(obj: dict, key_prefix: str = "encoder.") -> dict:
    """Encoder weights from a training checkpoint, stripped + float32."""
    import torch

    sd = obj["module"] if isinstance(obj, dict) and "module" in obj else obj
    return {
        k[len(key_prefix) :]: v.to(dtype=torch.float32)
        for k, v in sd.items()
        if k.startswith(key_prefix)
    }


def average_state_dicts(paths: list, key_prefix: str = "encoder.") -> dict:
    import torch

    avg = None
    keys = None
    for p in paths:
        obj = torch.load(resolve_ckpt(p), map_location="cpu", weights_only=False)
        sd = extract_encoder_state_dict(obj, key_prefix)
        if avg is None:
            avg = {k: v.clone() for k, v in sd.items()}
            keys = set(sd)
        else:
            if set(sd) != keys:
                raise ValueError(f"checkpoint {p} has mismatched keys")
            for k in avg:
                avg[k] += sd[k]
    for k in avg:
        avg[k] /= len(paths)
    return avg


def load_beats_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    cfg = (config.get("encoder_conf") or {}).get("beats_config") or {}
    if not cfg:
        raise ValueError(
            f"No encoder_conf.beats_config in {config_path}; pass the run config."
        )
    return cfg


def convert(train_ckpts: list, config_path: str, out_path: str) -> str:
    import torch

    state_dict = average_state_dicts(train_ckpts)
    cfg = load_beats_config(config_path)
    logger.info("encoder keys: %d | exporting to %s", len(state_dict), out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"model": state_dict, "cfg": cfg}, out_path)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="openbeats-convert")
    p.add_argument(
        "--train_ckpt",
        nargs="+",
        required=True,
        help="run dir / global_step dir / model-states file (multiple -> averaged)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="run config YAML (default: <first train_ckpt dir>/config.yaml)",
    )
    p.add_argument("--out", required=True, help="output {model,cfg} checkpoint path")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = args.config
    if config_path is None:
        first = args.train_ckpt[0]
        rundir = first if os.path.isdir(first) else os.path.dirname(os.path.dirname(first))
        config_path = os.path.join(rundir, "config.yaml")
    convert(args.train_ckpt, config_path, args.out)


if __name__ == "__main__":
    main()
