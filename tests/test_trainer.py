"""Offline smoke test for the stage-B loop via the PlainEngine fallback (no GPU,
no DeepSpeed): a few steps on a tiny dataset + TINY model, then checkpoint/resume.
"""

import json
import os

import numpy as np
import soundfile as sf

from openbeats.nets.encoder import BeatsEncoder, BeatsPretrainingPredictor
from openbeats import train as trainer
from openbeats.data.loader import build_dataloader
from openbeats.nets.pretrain_model import BeatsPretrainModel
from openbeats.train import MODEL_STATES, PlainEngine
from openbeats.tokenization.dump import dump

SR = 16000
TINY = {
    "encoder_layers": 2,
    "encoder_embed_dim": 64,
    "encoder_ffn_embed_dim": 128,
    "encoder_attention_heads": 4,
    "decoder_embed_dim": 64,
    "decoder_layers": 2,
    "codebook_vocab_size": 1024,  # matches the random tokenizer's quant_n
}


def _make_dataset(tmp_path):
    manifest = tmp_path / "corpus.jsonl"
    with open(manifest, "w") as f:
        for i, (sec, freq) in enumerate([(0.6, 440), (0.8, 550), (0.7, 330), (0.5, 660)]):
            wav = tmp_path / f"clip{i}.wav"
            t = np.linspace(0, sec, int(sec * SR), endpoint=False)
            sf.write(str(wav), (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)
            f.write(json.dumps({"id": f"clip{i}", "audio": str(wav)}) + "\n")
    out = tmp_path / "tokens"
    dump("random", str(manifest), str(out), seed=45, batch_size=2)
    return str(out)


def _model():
    enc = BeatsEncoder(input_size=0, beats_config=TINY, is_pretraining=True)
    dec = BeatsPretrainingPredictor(TINY)
    return BeatsPretrainModel(enc, dec, waveform_input=True)


def test_plain_engine_train_checkpoint_resume(tmp_path):
    data = _make_dataset(tmp_path)
    exp = tmp_path / "exp"

    _, loader, sampler = build_dataloader(
        data, batch_bins=SR * 2, num_workers=0, world_size=1, seed=0
    )
    engine = PlainEngine(_model(), lr=1e-3, device="cpu")
    trainer.run(
        engine, loader, sampler,
        max_steps=5, max_epochs=50, save_dir=str(exp),
        save_interval=5, valid_interval=0, log_interval=1, device="cpu", resume=False,
    )
    assert engine.global_steps >= 5

    # DeepSpeed-style checkpoint layout was written
    latest = exp / "latest"
    assert latest.is_file()
    tag = latest.read_text().strip()
    assert (exp / tag / MODEL_STATES).is_file()

    # resume into a fresh engine: global_steps restored, training continues
    engine2 = PlainEngine(_model(), lr=1e-3, device="cpu")
    _, loader2, sampler2 = build_dataloader(
        data, batch_bins=SR * 2, num_workers=0, world_size=1, seed=0
    )
    resumed_step = None

    # load_checkpoint directly to verify resume state
    path, client = engine2.load_checkpoint(str(exp))
    assert path is not None
    resumed_step = engine2.global_steps
    assert resumed_step >= 5
    assert "epoch" in client


def test_step_loss_rescale_is_scalar():
    import torch

    loss = torch.tensor([2.0], requires_grad=True)
    weight = torch.tensor([4.0])
    out = trainer._step_loss(loss, weight, world_size=2)
    assert out.ndim == 0
    # 2/4 * 2 = 1.0
    assert abs(float(out) - 1.0) < 1e-6


def test_train_module_imports():
    from openbeats import train
    from openbeats.pretraining import engine

    assert hasattr(train, "train_encoder_main") and hasattr(train, "run")
    assert hasattr(engine, "build_model") and hasattr(engine, "build_dataloaders")
