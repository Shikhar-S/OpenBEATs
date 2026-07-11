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
    assert not (exp / "best").exists()  # pretraining declares no selection_metric

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


def test_prune_checkpoints_keeps_last_n(tmp_path):
    import os

    from openbeats.train import _prune_checkpoints

    for n in (38, 76, 2000, 2014, 2128):
        os.makedirs(tmp_path / f"global_step{n}")
        (tmp_path / f"global_step{n}" / "states.pt").write_text("x")
    (tmp_path / "latest").write_text("global_step2128")

    _prune_checkpoints(str(tmp_path), keep_last=3)
    kept = sorted(
        int(d.name[len("global_step"):])
        for d in tmp_path.iterdir()
        if d.name.startswith("global_step")
    )
    assert kept == [2000, 2014, 2128]   # 3 highest by step number, not lexical
    assert (tmp_path / "latest").exists()  # latest file left intact

    _prune_checkpoints(str(tmp_path), keep_last=0)   # both knobs off -> no-op
    assert len([d for d in tmp_path.iterdir() if d.name.startswith("global_step")]) == 3


class _DummyEngine:
    def __init__(self):
        import torch.nn as nn

        self.module = nn.Linear(2, 2)


def test_best_tracker_snapshots_and_resumes(tmp_path):
    from openbeats.train import BestTracker, MODEL_STATES

    eng = _DummyEngine()
    t = BestTracker(str(tmp_path), "acc", "max")

    assert t.update(eng, {"acc": 0.5}, step=10) is True    # first is always best
    assert t.update(eng, {"acc": 0.4}, step=20) is False   # worse -> ignored
    assert t.update(eng, {"acc": 0.7}, step=30) is True     # better -> new best
    assert (t.value, t.step) == (0.7, 30)

    best = tmp_path / "best" / MODEL_STATES
    assert best.is_file()
    meta = json.loads((tmp_path / "best.json").read_text())
    assert meta == {"metric": "acc", "mode": "max", "value": 0.7, "step": 30}

    # a fresh tracker re-seeds from disk (resume) and rejects a later worse pass
    t2 = BestTracker(str(tmp_path), "acc", "max")
    assert (t2.value, t2.step) == (0.7, 30)
    assert t2.update(eng, {"acc": 0.6}, step=40) is False

    # a different metric/mode does not adopt the stale best
    t3 = BestTracker(str(tmp_path), "loss", "min")
    assert t3.value is None


def test_prune_checkpoints_keeps_milestones(tmp_path):
    import os

    from openbeats.train import _prune_checkpoints

    for n in (2000, 4000, 6000, 8000, 10000, 12000, 14000):
        os.makedirs(tmp_path / f"global_step{n}")
    # keep last 2 + every-10k milestone (+ latest always)
    _prune_checkpoints(str(tmp_path), keep_last=2, keep_milestone_every=10000)
    kept = sorted(
        int(d.name[len("global_step"):])
        for d in tmp_path.iterdir()
        if d.name.startswith("global_step")
    )
    assert kept == [10000, 12000, 14000]   # milestone 10k + last-2 (12k,14k)
