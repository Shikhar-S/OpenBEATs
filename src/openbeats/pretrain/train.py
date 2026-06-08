"""Stage B entry point: build the BEATs pretraining model + data and launch the
DeepSpeed loop. Console script ``openbeats-pretrain``.

Launch with torchrun (single- or multi-node):

    torchrun --standalone --nnodes=1 --nproc_per_node=8 -m openbeats.pretrain.train \\
        --config conf/pretrain_large.yaml \\
        --deepspeed_config conf/ds_openbeats_large.json \\
        --train_data data/tokens_train --valid_data data/tokens_valid \\
        --output_dir exp/openbeats_large

Auto-resumes from ``--output_dir`` (DeepSpeed ``latest``). Heavy imports are
deferred into ``main`` so ``--help`` stays fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger("openbeats.pretrain")


def build_model(config: dict):
    """Construct BeatsPretrainModel from a run config's ``encoder_conf``/``model_conf``."""
    from ..beats_encoder import BeatsEncoder, BeatsPretrainingPredictor
    from .model import BeatsPretrainModel

    encoder_conf = config["encoder_conf"]
    beats_config = encoder_conf["beats_config"]
    encoder = BeatsEncoder(
        input_size=0,
        beats_config=beats_config,
        is_pretraining=True,
        fbank_mean=encoder_conf.get("fbank_mean", 15.41663),
        fbank_std=encoder_conf.get("fbank_std", 6.55582),
    )
    decoder = BeatsPretrainingPredictor(beats_config)
    model = BeatsPretrainModel(encoder, decoder, **config.get("model_conf", {}))
    return model


def _check_dataset_compat(config: dict, meta: dict):
    """Fail fast if the run config disagrees with the dataset it will train on."""
    K = config["encoder_conf"]["beats_config"].get("codebook_vocab_size", 1024)
    if int(K) != int(meta["codebook_size"]):
        raise ValueError(
            f"codebook mismatch: config codebook_vocab_size={K} but dataset "
            f"codebook_size={meta['codebook_size']}."
        )
    if not config.get("model_conf", {}).get("waveform_input", False) and meta.get(
        "waveform_input", True
    ):
        logger.warning(
            "dataset is waveform_input but model_conf.waveform_input is false."
        )


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def main(argv=None):
    p = argparse.ArgumentParser(prog="openbeats-pretrain")
    p.add_argument("--config", required=True, help="run config YAML")
    p.add_argument("--deepspeed_config", default=None, help="DeepSpeed JSON config")
    p.add_argument("--train_data", required=True, help="train token dataset dir")
    p.add_argument("--valid_data", default=None, help="valid token dataset dir")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--no-deepspeed",
        action="store_true",
        help="use the plain torch fallback engine (CPU / no DeepSpeed)",
    )
    p.add_argument("--device", default=None, help="override device (default: auto)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    import yaml

    from ..tokenize import schema
    from . import trainer
    from .data import build_dataloader

    with open(args.config) as f:
        config = yaml.safe_load(f)

    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    use_deepspeed = not args.no_deepspeed
    if use_deepspeed:
        try:
            import deepspeed
        except ImportError:
            logger.warning("deepspeed not installed; falling back to plain engine.")
            use_deepspeed = False

    if args.device:
        device = args.device
    elif use_deepspeed or os.environ.get("LOCAL_RANK") is not None:
        import torch

        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- distributed init -----
    if use_deepspeed and world_size > 1:
        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl")

    # persist the run config next to checkpoints so openbeats-convert finds it
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    # ----- model -----
    model = build_model(config)
    meta = schema.load_meta(args.train_data)
    _check_dataset_compat(config, meta)

    data_conf = config.get("data_conf", {})
    _, train_loader, train_sampler = build_dataloader(
        args.train_data,
        batch_bins=data_conf.get("batch_bins", 3_200_000),
        bin_by=data_conf.get("bin_by", "samples"),
        num_workers=data_conf.get("num_workers", 4),
        rank=rank,
        world_size=world_size,
        seed=data_conf.get("seed", 0),
        max_batch_size=data_conf.get("max_batch_size"),
    )
    valid_loader = None
    if args.valid_data:
        _, valid_loader, _ = build_dataloader(
            args.valid_data,
            batch_bins=data_conf.get("batch_bins", 3_200_000),
            bin_by=data_conf.get("bin_by", "samples"),
            num_workers=data_conf.get("num_workers", 4),
            rank=rank,
            world_size=world_size,
            shuffle=False,
        )

    # ----- engine -----
    if use_deepspeed:
        import deepspeed

        with open(args.deepspeed_config) as f:
            ds_config = json.load(f)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
    else:
        train_conf = config.get("train_conf", {})
        engine = trainer.PlainEngine(
            model,
            lr=train_conf.get("lr", 1e-4),
            weight_decay=train_conf.get("weight_decay", 0.01),
            grad_clip=train_conf.get("grad_clip", 1.0),
            device=device,
        )

    train_conf = config.get("train_conf", {})
    trainer.run(
        engine,
        train_loader,
        train_sampler,
        valid_loader=valid_loader,
        max_steps=train_conf.get("max_steps", 400_000),
        max_epochs=train_conf.get("max_epochs", 1000),
        save_dir=args.output_dir,
        save_interval=train_conf.get("save_interval", 10_000),
        valid_interval=train_conf.get("valid_interval", 10_000),
        log_interval=train_conf.get("log_interval", 50),
        device=device,
        resume=True,
    )


if __name__ == "__main__":
    main()
