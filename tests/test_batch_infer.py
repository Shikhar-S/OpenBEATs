"""Offline batch-inference gate: TINY encoder cls checkpoint -> batch_infer over a
manifest -> predictions.parquet with one row per segment, right-width probs, labels.
"""

import json

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from openbeats.inference.batch_infer import batch_infer
from openbeats.inference.model import OpenBeats
from openbeats.nets.classification_model import BeatsClassificationModel
from openbeats.nets.encoder import BeatsEncoder

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

def _make_checkpoint(path, labels):
    encoder = BeatsEncoder(input_size=0, beats_config=TINY, is_pretraining=False,
                           fbank_mean=13.0, fbank_std=5.0)
    model = BeatsClassificationModel(encoder, n_classes=len(labels),
                                     classification_type="multi-class")
    state = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
    state["decoder.linear_out.weight"] = model.decoder.linear_out.weight.detach()
    state["decoder.linear_out.bias"] = model.decoder.linear_out.bias.detach()
    cfg = {**TINY, "fbank_mean": 13.0, "fbank_std": 5.0}
    torch.save({"model": state, "cfg": cfg, "token_list": labels, "multi_label": False}, path)

def _write_manifest(tmp_path, n=5):
    manifest = tmp_path / "test.jsonl"
    with open(manifest, "w") as f:
        for i in range(n):
            wav = tmp_path / f"c{i}.wav"
            t = np.linspace(0, 0.6, int(0.6 * SR), endpoint=False)
            freq = 200 if i % 2 == 0 else 2000
            sf.write(str(wav), (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)
            f.write(json.dumps({"id": f"c{i}", "audio": str(wav)}) + "\n")
    return manifest

def test_batch_infer_predictions(tmp_path):
    labels = ["lo", "hi"]
    ckpt = tmp_path / "cls.pt"
    _make_checkpoint(ckpt, labels)
    manifest = _write_manifest(tmp_path, n=5)
    out = tmp_path / "preds"

    batch_infer(str(ckpt), str(manifest), str(out), device="cpu",
                batch_bins=SR * 2, num_workers=0)

    table = pq.read_table(str(out / "predictions.parquet"))
    assert table.num_rows == 5
    assert set(table.column_names) >= {"id", "probs", "logits", "pred_label"}
    d = table.to_pydict()
    assert set(d["id"]) == {f"c{i}" for i in range(5)}
    for probs, lab in zip(d["probs"], d["pred_label"]):
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-4  # softmax
        assert lab in labels

    meta = json.loads(open(out / "predict.json").read())
    assert meta["n_rows"] == 5 and meta["labels"] == labels and meta["multi_label"] is False

def test_batch_infer_matches_single_encode(tmp_path):
    labels = ["lo", "hi"]
    ckpt = tmp_path / "cls.pt"
    _make_checkpoint(ckpt, labels)
    manifest = _write_manifest(tmp_path, n=3)
    out = tmp_path / "preds"

    batch_infer(str(ckpt), str(manifest), str(out), device="cpu",
                batch_bins=SR * 2, num_workers=0)
    d = pq.read_table(str(out / "predictions.parquet")).to_pydict()
    batched = dict(zip(d["id"], d["probs"]))

    model = OpenBeats.from_pretrained(str(ckpt))
    for line in open(manifest):
        r = json.loads(line)
        wav, _ = sf.read(r["audio"], dtype="float32")
        ref = model.encode(wav, SR)["probs"]
        np.testing.assert_allclose(batched[r["id"]], ref, atol=1e-4)
