"""Encoder-pretraining engine: the objective-specific half of training.

Implements the engine contract that the common ``openbeats.train`` runner drives:

    build_model(config) -> nn.Module                          # from nets/
    build_dataloaders(config, rank, world_size) -> (train, sampler, valid)

The model's ``forward(**batch) -> (loss, stats, weight)`` is the shared contract.
Data paths live in ``config["data_conf"]`` (``train_data`` / ``valid_data``),
injected by the runner from its CLI args.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("openbeats.pretrain")


def build_model(config: dict):
    """Construct BeatsPretrainModel from a run config's ``encoder_conf``/``model_conf``."""
    from ..nets.encoder import BeatsEncoder, BeatsPretrainingPredictor
    from ..nets.pretrain_model import BeatsPretrainModel

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


def build_dataloaders(config: dict, rank: int, world_size: int):
    """Build (train_loader, train_sampler, valid_loader) for encoder pretraining.

    Reads a token dataset (audio + codes) from ``config["data_conf"]["train_data"]``
    (and optional ``valid_data``); guards the run config against the dataset metadata.
    """
    from ..data import dataset as schema
    from ..data.loader import build_dataloader

    data_conf = config.get("data_conf", {})
    train_data = data_conf["train_data"]
    valid_data = data_conf.get("valid_data")

    meta = schema.load_meta(train_data)
    _check_dataset_compat(config, meta)

    _, train_loader, train_sampler = build_dataloader(
        train_data,
        batch_bins=data_conf.get("batch_bins", 3_200_000),
        bin_by=data_conf.get("bin_by", "samples"),
        num_workers=data_conf.get("num_workers", 4),
        rank=rank,
        world_size=world_size,
        seed=data_conf.get("seed", 0),
        max_batch_size=data_conf.get("max_batch_size"),
    )
    valid_loader = None
    if valid_data:
        _, valid_loader, _ = build_dataloader(
            valid_data,
            batch_bins=data_conf.get("batch_bins", 3_200_000),
            bin_by=data_conf.get("bin_by", "samples"),
            num_workers=data_conf.get("num_workers", 4),
            rank=rank,
            world_size=world_size,
            shuffle=False,
        )
    return train_loader, train_sampler, valid_loader
