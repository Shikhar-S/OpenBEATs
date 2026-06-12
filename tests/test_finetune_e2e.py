"""End-to-end fine-tuning gate (offline, CPU): TINY encoder -> classification
fine-tune (PlainEngine, with validation) -> cls export -> inference predicts.
"""

import json

import numpy as np
import soundfile as sf
import yaml

from openbeats import train as trainer
from openbeats.finetuning import engine as ft_engine
from openbeats.inference.model import OpenBeats
from openbeats.train import PlainEngine
from openbeats.utils.convert import convert_cls

SR = 16000
TINY = {
    "encoder_layers": 2,
    "encoder_embed_dim": 64,
    "encoder_ffn_embed_dim": 128,
    "encoder_attention_heads": 4,
    "decoder_embed_dim": 64,
    "decoder_layers": 2,
    "codebook_vocab_size": 1024,
}

def _write_corpus(path, manifest):
    with open(manifest, "w") as f:
        for i in range(6):
            lab, freq = ("lo", 200) if i % 2 == 0 else ("hi", 2000)
            wav = path / f"c{i}.wav"
            t = np.linspace(0, 0.6, int(0.6 * SR), endpoint=False)
            sf.write(str(wav), (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)
            f.write(json.dumps({"id": f"c{i}", "audio": str(wav), "label": lab}) + "\n")

def test_finetune_cfg_merges_init_ckpt_arch(tmp_path):
    # The FT config carries only overrides; the architecture must come from init_ckpt,
    # else inference rebuilds the encoder from BEATs defaults and the weights are useless.
    import torch

    from openbeats.utils.convert import _finetune_cfg

    init = tmp_path / "pretrained.pt"
    torch.save({"model": {}, "cfg": {**TINY, "deep_norm": True}}, init)
    config = {
        "encoder_conf": {"beats_config": {"dropout": 0.3}, "fbank_mean": 1.0, "fbank_std": 2.0},
        "finetune_conf": {"init_ckpt": str(init)},
    }
    cfg = _finetune_cfg(config)
    assert cfg["encoder_layers"] == TINY["encoder_layers"]  # architecture from init_ckpt
    assert cfg["deep_norm"] is True
    assert cfg["dropout"] == 0.3                            # FT override wins
    assert cfg["fbank_mean"] == 1.0 and cfg["fbank_std"] == 2.0

def test_finetune_convert_infer(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_corpus(tmp_path, manifest)
    labels_txt = tmp_path / "labels.txt"
    labels_txt.write_text("lo\nhi\n")

    config = {
        "encoder_conf": {"beats_config": TINY, "fbank_mean": 13.0, "fbank_std": 5.0},
        "finetune_conf": {"freeze_encoder": False},
        "model_conf": {"classification_type": "multi-class", "label_smoothing": 0.1},
        "data_conf": {
            "train_data": str(manifest),
            "valid_data": str(manifest),
            "label_list": str(labels_txt),
            "batch_bins": SR * 2,
            "num_workers": 0,
        },
    }

    model = ft_engine.build_model(config)
    train_loader, sampler, valid_loader = ft_engine.build_dataloaders(config, 0, 1)

    exp = tmp_path / "exp"
    engine = PlainEngine(model, lr=1e-3, device="cpu")
    trainer.run(
        engine, train_loader, sampler, valid_loader=valid_loader,
        max_steps=6, max_epochs=50, save_dir=str(exp), save_interval=6,
        valid_interval=6, log_interval=3, device="cpu", resume=False,
    )

    with open(exp / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    out = tmp_path / "openbeats_cls.pt"
    convert_cls([str(exp)], str(exp / "config.yaml"), str(out))
    assert out.is_file()

    # ---- inference: the fine-tuned checkpoint loads with head + labels ----
    model = OpenBeats.from_pretrained(str(out))
    assert model.classifier is not None
    assert model.labels == ["lo", "hi"] and model.multi_label is False
    t = np.linspace(0, 0.6, int(0.6 * SR), endpoint=False)
    tone = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    out_dict = model.encode(tone, SR)
    assert out_dict["probs"].shape[-1] == 2
    assert out_dict["top_label"] in ("lo", "hi")
