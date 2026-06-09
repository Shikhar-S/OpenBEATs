"""The shipped pretraining configs parse and wire up a model correctly."""

import json
from pathlib import Path

import yaml

import openbeats
from openbeats.pretraining.engine import build_model

CONF = Path(openbeats.__file__).parent / "conf"


def _load_yaml(name):
    with open(CONF / name) as f:
        return yaml.safe_load(f)


def _load_json(name):
    with open(CONF / name) as f:
        return json.load(f)


def test_run_configs_present_and_consistent():
    for size in ("base", "large"):
        cfg = _load_yaml(f"pretrain_{size}.yaml")
        bc = cfg["encoder_conf"]["beats_config"]
        assert bc["codebook_vocab_size"] == 1024
        assert cfg["model_conf"]["waveform_input"] is True
        assert cfg["model_conf"]["ignore_id"] == -2
        # the DeepSpeed scheduler horizon should match the loop's max_steps
        ds = _load_json(f"ds_openbeats_{size}.json")
        # bf16 via torch_autocast (NOT pure bf16): the waveform frontend computes
        # fbank in fp32 (ta_kaldi can't do bf16), so params stay fp32 and the
        # forward autocasts -- verified on a GH200. See CLAUDE.md.
        assert ds["torch_autocast"]["enabled"] is True
        assert ds["torch_autocast"]["dtype"] == "bfloat16"
        assert "bf16" not in ds  # pure-bf16 would break the fp32 fbank -> conv
        assert ds["zero_optimization"]["stage"] == 1
        assert (
            ds["scheduler"]["params"]["total_num_steps"]
            == cfg["train_conf"]["max_steps"]
        )


def test_build_model_from_base_config():
    cfg = _load_yaml("pretrain_base.yaml")
    model = build_model(cfg)
    bc = cfg["encoder_conf"]["beats_config"]
    assert model.encoder.config.encoder_embed_dim == bc["encoder_embed_dim"]
    assert model.n_targets == bc["codebook_vocab_size"]
    # mixup / contrastive disabled by config
    assert model.mixup_augmentation is None
    assert model.contrastive_loss_weight == 0.0
