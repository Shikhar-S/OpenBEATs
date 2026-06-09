"""End-to-end gate (offline, CPU): random tokenizer + TINY encoder exercises every
stage -- dump -> pretrain (PlainEngine) -> convert -> infer -- with no downloads.
"""

import json

import numpy as np
import soundfile as sf
import yaml

from openbeats.nets.encoder import BeatsEncoder, BeatsPretrainingPredictor
from openbeats.utils.convert import convert
from openbeats.inference.model import OpenBeats
from openbeats import train as trainer
from openbeats.data.loader import build_dataloader
from openbeats.nets.pretrain_model import BeatsPretrainModel
from openbeats.train import PlainEngine
from openbeats.tokenization.dump import dump

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


def test_dump_train_convert_infer(tmp_path):
    # ---- stage A: corpus -> codes ----
    manifest = tmp_path / "corpus.jsonl"
    with open(manifest, "w") as f:
        for i, (sec, freq) in enumerate([(0.6, 440), (0.8, 550), (0.7, 330), (0.5, 660)]):
            wav = tmp_path / f"clip{i}.wav"
            t = np.linspace(0, sec, int(sec * SR), endpoint=False)
            sf.write(str(wav), (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)
            f.write(json.dumps({"id": f"clip{i}", "audio": str(wav)}) + "\n")
    data = tmp_path / "tokens"
    dump("random", str(manifest), str(data), seed=45, batch_size=2)

    # ---- stage B: pretrain a few steps ----
    exp = tmp_path / "exp"
    _, loader, sampler = build_dataloader(
        str(data), batch_bins=SR * 2, num_workers=0, world_size=1, seed=0
    )
    enc = BeatsEncoder(input_size=0, beats_config=TINY, is_pretraining=True)
    dec = BeatsPretrainingPredictor(TINY)
    engine = PlainEngine(BeatsPretrainModel(enc, dec, waveform_input=True), lr=1e-3)
    trainer.run(
        engine, loader, sampler, max_steps=5, max_epochs=50, save_dir=str(exp),
        save_interval=5, valid_interval=0, log_interval=5, device="cpu", resume=False,
    )
    # train.py would write this; emulate it for the convert step
    with open(exp / "config.yaml", "w") as f:
        yaml.safe_dump({"encoder_conf": {"beats_config": TINY}}, f)

    # ---- stage C: export {model, cfg} ----
    out = tmp_path / "openbeats_pretrained.pt"
    convert([str(exp)], str(exp / "config.yaml"), str(out))
    assert out.is_file()

    # ---- use it via the inference path ----
    model = OpenBeats.from_pretrained(str(out))
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    emb = model.encode(tone, SR)["patch_embeddings"]
    assert emb.ndim == 2 and emb.shape[1] == TINY["encoder_embed_dim"]
    assert emb.shape[0] > 0
