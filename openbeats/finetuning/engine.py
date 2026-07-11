"""Classification fine-tuning engine: the objective-specific half of training.

Implements the engine contract the common openbeats.train runner drives:

    build_model(config) -> nn.Module                          # nets/classification_model
    build_dataloaders(config, rank, world_size) -> (train, sampler, valid)

build_model loads a pretrained encoder from finetune_conf.init_ckpt (reusing the
inference checkpoint loader), attaches a classification head sized to the label
vocabulary, and optionally freezes the encoder for linear probing. The data half
reads labeled manifests (config["data_conf"]["train_data"]/valid_data, injected by
the runner) and a labels.txt vocabulary.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("openbeats.finetune")

def build_model(config: dict):
    """Build a BeatsClassificationModel: pretrained encoder + linear head (+ optional freeze)."""
    from ..data.labeled import read_label_list
    from ..nets.classification_model import BeatsClassificationModel
    from ..nets.encoder import BeatsEncoder
    from ..utils.checkpoint import encoder_state_dict, load_checkpoint

    encoder_conf = config.get("encoder_conf", {})
    finetune_conf = config.get("finetune_conf", {})
    data_conf = config.get("data_conf", {})

    label_list = read_label_list(data_conf["label_list"])
    n_classes = len(label_list)

    # Architecture + pretrained weights travel with the SSL checkpoint; the FT config
    # may override fields (e.g. dropout/layerdrop) via encoder_conf.beats_config.
    cfg, weights = {}, None
    if finetune_conf.get("init_ckpt"):
        ckpt = load_checkpoint(finetune_conf["init_ckpt"], finetune_conf.get("base"))
        cfg, weights = dict(ckpt.cfg), encoder_state_dict(ckpt.weights)
    beats_config = {**cfg, **(encoder_conf.get("beats_config") or {})}
    if not beats_config:
        raise ValueError(
            "No beats_config: set finetune_conf.init_ckpt or encoder_conf.beats_config."
        )

    encoder = BeatsEncoder(
        input_size=0,
        beats_config=beats_config,
        is_pretraining=False,
        fbank_mean=encoder_conf.get("fbank_mean", cfg.get("fbank_mean", 15.41663)),
        fbank_std=encoder_conf.get("fbank_std", cfg.get("fbank_std", 6.55582)),
        specaug_config=encoder_conf.get("specaug_config"),
        roll_augment=encoder_conf.get("roll_augment", False),
        roll_interval=encoder_conf.get("roll_interval", 1600),
        use_weighted_representation=encoder_conf.get("use_weighted_representation", False),
    )
    if weights is not None:
        info = encoder.load_state_dict(weights, strict=False)
        missing = [k for k in info.missing_keys if "_pad" not in k]
        if missing:
            logger.warning("init_ckpt missing encoder keys (first 10): %s", missing[:10])

    model = BeatsClassificationModel(encoder, n_classes, **config.get("model_conf", {}))
    if finetune_conf.get("freeze_encoder", False):
        model.encoder.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("froze encoder (linear probing): %d trainable params", trainable)
    return model

def build_dataloaders(config: dict, rank: int, world_size: int):
    """Build (train_loader, train_sampler, valid_loader) for classification fine-tuning."""
    from ..data.labeled import build_labeled_dataloader, read_label_list

    data_conf = config["data_conf"]
    label_list = read_label_list(data_conf["label_list"])
    multi_label = config.get("model_conf", {}).get("classification_type") == "multi-label"

    _, train_loader, train_sampler = build_labeled_dataloader(
        data_conf["train_data"],
        label_list,
        batch_bins=data_conf.get("batch_bins", 3_200_000),
        multi_label=multi_label,
        num_workers=data_conf.get("num_workers", 4),
        rank=rank,
        world_size=world_size,
        seed=data_conf.get("seed", 0),
        max_batch_size=data_conf.get("max_batch_size"),
    )
    valid_loader = None
    if data_conf.get("valid_data"):
        _, valid_loader, _ = build_labeled_dataloader(
            data_conf["valid_data"],
            label_list,
            batch_bins=data_conf.get("batch_bins", 3_200_000),
            multi_label=multi_label,
            num_workers=data_conf.get("num_workers", 4),
            rank=rank,
            world_size=world_size,
            shuffle=False,
            max_batch_size=data_conf.get("max_batch_size"),
        )
    return train_loader, train_sampler, valid_loader
