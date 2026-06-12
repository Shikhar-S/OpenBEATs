"""Stage C: export a lightweight inference checkpoint from a training checkpoint.

Console script openbeats-convert. Reads a DeepSpeed (or PlainEngine) training
checkpoint -- the full model lives under ["module"] (ZeRO-1 keeps it whole, so
no fp32 consolidation is needed) -- keeps the encoder.-prefixed weights (strip
prefix, cast float32), attaches the run's beats_config, and saves the
{"model", "cfg"} file that OpenBeats.from_pretrained loads. Multiple
checkpoints are averaged.

    openbeats-convert \\
        --train_ckpt exp/openbeats_large \\        # dir (resolves latest) or a
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

    Accepts the states file itself, a global_step{N} directory, or a run
    directory containing a DeepSpeed latest file.
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

def _module_sd(obj: dict) -> dict:
    return obj["module"] if isinstance(obj, dict) and "module" in obj else obj

def extract_encoder_state_dict(obj: dict, key_prefix: str = "encoder.") -> dict:
    """Encoder weights from a training checkpoint, stripped + float32."""
    import torch

    return {
        k[len(key_prefix) :]: v.to(dtype=torch.float32)
        for k, v in _module_sd(obj).items()
        if k.startswith(key_prefix)
    }

def extract_cls_state_dict(obj: dict) -> dict:
    """Encoder + classification head, prefixes kept (encoder./decoder.), float32.

    Prefixes are preserved so the inference loader's encoder_state_dict strips
    encoder. and build_classifier finds decoder.linear_out.weight.
    """
    import torch

    return {
        k: v.to(dtype=torch.float32)
        for k, v in _module_sd(obj).items()
        if k.startswith("encoder.") or k.startswith("decoder.")
    }

def average_state_dicts(paths: list, extract=extract_encoder_state_dict) -> dict:
    import torch

    avg = None
    keys = None
    for p in paths:
        obj = torch.load(resolve_ckpt(p), map_location="cpu", weights_only=False)
        sd = extract(obj)
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
    """beats_config from a run config, with fbank stats folded in so the exported
    model normalizes its input correctly at inference."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    enc = config.get("encoder_conf") or {}
    cfg = dict(enc.get("beats_config") or {})
    if not cfg:
        raise ValueError(
            f"No encoder_conf.beats_config in {config_path}; pass the run config."
        )
    for k in ("fbank_mean", "fbank_std"):
        if enc.get(k) is not None:
            cfg[k] = enc[k]
    return cfg

def convert(train_ckpts: list, config_path: str, out_path: str) -> str:
    import torch

    state_dict = average_state_dicts(train_ckpts)
    cfg = load_beats_config(config_path)
    logger.info("encoder keys: %d | exporting to %s", len(state_dict), out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"model": state_dict, "cfg": cfg}, out_path)
    return out_path

def _finetune_cfg(config: dict) -> dict:
    """Full beats_config of a fine-tune, assembled exactly as finetuning.engine does:
    the init_ckpt's architecture under the FT-time overrides, plus fbank stats.

    The FT run config's encoder_conf.beats_config holds only overrides (dropout etc.);
    the real architecture (deep_norm, relative_position_embedding, dims, ...) lives in
    the pretrained init_ckpt, so it must be merged in or inference rebuilds the encoder
    from BEATs defaults and the trained weights become meaningless.
    """
    enc = config.get("encoder_conf") or {}
    overrides = dict(enc.get("beats_config") or {})
    init = (config.get("finetune_conf") or {}).get("init_ckpt")
    base = {}
    if init:
        from .checkpoint import load_checkpoint

        base = dict(load_checkpoint(init).cfg)
    cfg = {**base, **overrides}
    if not cfg:
        raise ValueError(
            "No beats_config: set finetune_conf.init_ckpt or encoder_conf.beats_config."
        )
    for k in ("fbank_mean", "fbank_std"):
        if enc.get(k) is not None:
            cfg[k] = enc[k]
    return cfg

def convert_cls(train_ckpts: list, config_path: str, out_path: str) -> str:
    """Export a fine-tuned classifier ({model, cfg, token_list, multi_label}).

    Keeps encoder + head; labels and the multi-label flag come from the run config,
    so OpenBeats.from_pretrained loads it directly (head + class names + sigmoid/softmax).
    """
    import torch
    import yaml

    state_dict = average_state_dicts(train_ckpts, extract=extract_cls_state_dict)
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    cfg = _finetune_cfg(config)
    label_list_path = (config.get("data_conf") or {}).get("label_list")
    if not label_list_path:
        raise ValueError(f"No data_conf.label_list in {config_path}.")
    from ..data.labeled import read_label_list

    labels = read_label_list(label_list_path)
    multi_label = (config.get("model_conf") or {}).get("classification_type") == "multi-label"
    logger.info("cls keys: %d | %d classes | multi_label=%s -> %s",
                len(state_dict), len(labels), multi_label, out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"model": state_dict, "cfg": cfg, "token_list": labels,
                "multi_label": multi_label}, out_path)
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

    convert(args.train_ckpt, _resolve_config(args.config, args.train_ckpt), args.out)

def cls_main(argv=None):
    p = argparse.ArgumentParser(prog="openbeats-convert-cls")
    p.add_argument("--train_ckpt", nargs="+", required=True,
                   help="run dir / global_step dir / model-states file (multiple -> averaged)")
    p.add_argument("--config", default=None,
                   help="run config YAML (default: <first train_ckpt dir>/config.yaml)")
    p.add_argument("--out", required=True, help="output {model,cfg,token_list,multi_label} checkpoint")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    convert_cls(args.train_ckpt, _resolve_config(args.config, args.train_ckpt), args.out)

def _resolve_config(config_path, train_ckpt):
    if config_path is not None:
        return config_path
    first = train_ckpt[0]
    rundir = first if os.path.isdir(first) else os.path.dirname(os.path.dirname(first))
    return os.path.join(rundir, "config.yaml")

if __name__ == "__main__":
    main()
