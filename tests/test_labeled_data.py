"""Offline tests for the labeled-audio data layer (manifest label field, dataset,
collate)."""

import json

import numpy as np
import soundfile as sf

from openbeats.data.labeled import (
    MIN_SAMPLES,
    LabeledAudioDataset,
    cls_collate,
    read_label_list,
)
from openbeats.data.manifest import normalize_entry, read_manifest, write_manifest

SR = 16000

def _clip(path, sec=0.5, freq=440):
    t = np.linspace(0, sec, int(sec * SR), endpoint=False)
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)

def _labels_file(tmp_path, names):
    p = tmp_path / "labels.txt"
    p.write_text("\n".join(names) + "\n")
    return p

def test_manifest_carries_label_roundtrip(tmp_path):
    e = normalize_entry("/d/a.wav", label="dog")
    assert e["label"] == "dog"
    out = tmp_path / "m.jsonl"
    write_manifest(out, [e, normalize_entry("/d/b.wav", label=["a", "b"])])
    back = read_manifest(out)
    assert back[0]["label"] == "dog" and back[1]["label"] == ["a", "b"]

def test_labeled_dataset_multiclass(tmp_path):
    manifest = tmp_path / "train.jsonl"
    with open(manifest, "w") as f:
        for i, (freq, lab) in enumerate([(200, "lo"), (2000, "hi"), (200, "lo")]):
            wav = tmp_path / f"c{i}.wav"
            _clip(wav, freq=freq)
            f.write(json.dumps({"id": f"c{i}", "audio": str(wav), "label": lab}) + "\n")
    labels = read_label_list(_labels_file(tmp_path, ["lo", "hi"]))

    ds = LabeledAudioDataset(str(manifest), labels, multi_label=False)
    assert ds.n_classes == 2 and len(ds) == 3
    assert ds.n_samples.tolist() == [int(0.5 * SR)] * 3
    item = ds[1]
    assert item["label"].tolist() == [1]  # "hi"
    assert item["label_lengths"] == 1

def test_labeled_dataset_multilabel_and_collate(tmp_path):
    manifest = tmp_path / "train.jsonl"
    with open(manifest, "w") as f:
        for i, labs in enumerate([["a", "c"], ["b"]]):
            wav = tmp_path / f"c{i}.wav"
            _clip(wav, sec=0.4 + 0.2 * i)
            f.write(json.dumps({"id": f"c{i}", "audio": str(wav), "label": labs}) + "\n")
    labels = read_label_list(_labels_file(tmp_path, ["a", "b", "c"]))

    ds = LabeledAudioDataset(str(manifest), labels, multi_label=True)
    batch = cls_collate([ds[0], ds[1]])
    assert batch["speech"].shape[0] == 2
    assert batch["label"].shape == (2, 2)  # padded to the longest label set
    assert batch["label"][1, 1].item() == -1  # pad
    assert batch["label_lengths"].tolist() == [2, 1]
    # speech padded to the longer clip
    assert int(batch["speech_lengths"].max()) == batch["speech"].shape[1]

def test_short_clip_padded_to_minimum(tmp_path):
    # clips shorter than one BEATs patch (<~3200 samples) must be padded, else the
    # encoder emits zero patches -> NaN. Keep the clip, pad up to MIN_SAMPLES.
    manifest = tmp_path / "train.jsonl"
    wav = tmp_path / "tiny.wav"
    _clip(wav, sec=0.05)  # 800 samples << MIN_SAMPLES
    with open(manifest, "w") as f:
        f.write(json.dumps({"id": "t", "audio": str(wav), "label": "a"}) + "\n")
    labels = read_label_list(_labels_file(tmp_path, ["a", "b"]))
    ds = LabeledAudioDataset(str(manifest), labels)
    item = ds[0]
    assert item["speech_lengths"] == MIN_SAMPLES
    assert item["speech"].shape[0] == MIN_SAMPLES

def test_multiclass_rejects_multiple_labels(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    wav = tmp_path / "c.wav"
    _clip(wav)
    with open(manifest, "w") as f:
        f.write(json.dumps({"id": "c", "audio": str(wav), "label": ["a", "b"]}) + "\n")
    labels = read_label_list(_labels_file(tmp_path, ["a", "b"]))
    try:
        LabeledAudioDataset(str(manifest), labels, multi_label=False)
        assert False, "expected a multi-class single-label error"
    except ValueError:
        pass
