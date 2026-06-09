"""Offline tests for the generic data features: JSONL manifest with segments,
segment-aware audio loading, schema v2 start/end + fbank-stats metadata, and the
training-time fbank match guard."""

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from openbeats.data import dataset as schema
from openbeats.data.audio import load_audio
from openbeats.data.loader import TokenDataset
from openbeats.data.manifest import normalize_entry, read_manifest, write_manifest
from openbeats.tokenization.dump import dump

SR = 16000


def _ramp_wav(path, seconds, sr=SR):
    """A deterministic, position-dependent signal so spans are distinguishable."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sig = 0.1 * np.sin(2 * np.pi * (200 + 80 * t) * t)  # chirp: content varies with t
    # FLOAT subtype so the round-trip is exact (WAV defaults to lossy PCM_16)
    sf.write(str(path), sig.astype(np.float32), sr, subtype="FLOAT")
    return sig.astype(np.float32)


# ------------------------------------------------------------------ manifest
def test_manifest_parses_segments_and_bare_paths(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text(
        json.dumps({"id": "seg0", "audio": "/d/rec.wav", "start": 1.0, "end": 2.5}) + "\n"
        + json.dumps({"audio": "/d/clip.flac"}) + "\n"
        + "/d/bare.wav\n"
    )
    items = read_manifest(str(m))
    assert items[0] == {"id": "seg0", "audio": "/d/rec.wav", "start": 1.0, "end": 2.5}
    assert items[1]["id"] == "clip" and items[1]["start"] is None and items[1]["end"] is None
    assert items[2]["id"] == "bare"


def test_manifest_auto_id_has_span_suffix_and_roundtrips(tmp_path):
    e = normalize_entry("/d/rec.wav", start=3.0, end=4.0)
    assert e["id"].startswith("rec_") and e["id"] != "rec"   # span-suffixed, unique
    out = tmp_path / "w.jsonl"
    write_manifest(str(out), [e, normalize_entry("/d/whole.wav")])
    back = read_manifest(str(out))
    assert back[0]["start"] == 3.0 and back[0]["end"] == 4.0
    assert "start" not in {k for k, v in back[1].items() if v is not None}  # whole-file


# --------------------------------------------------------------- segment audio
def test_load_audio_segment_reads_exact_span(tmp_path):
    wav = tmp_path / "rec.wav"
    full = _ramp_wav(wav, 2.0)
    span, sr = load_audio(str(wav), start=0.5, end=1.0)
    assert sr == SR
    assert span.shape[0] == int(0.5 * SR)
    # native rate == target rate, so the span must equal the slice exactly
    np.testing.assert_allclose(span, full[int(0.5 * SR): int(1.0 * SR)], atol=1e-6)


def test_load_audio_segment_resamples_from_native_rate(tmp_path):
    wav = tmp_path / "hi.wav"
    _ramp_wav(wav, 2.0, sr=32000)            # native 32 kHz
    span, sr = load_audio(str(wav), start=0.5, end=1.0)  # 0.5 s -> 8000 @ 16 kHz
    assert sr == SR
    assert abs(span.shape[0] - int(0.5 * SR)) <= 2


def test_load_audio_whole_file_unchanged(tmp_path):
    wav = tmp_path / "rec.wav"
    full = _ramp_wav(wav, 1.0)
    out, sr = load_audio(str(wav))
    assert sr == SR and out.shape[0] == full.shape[0]


# --------------------------------------------------------- schema v2 + dump
def test_dump_segments_roundtrip_and_metadata(tmp_path):
    rec = tmp_path / "rec.wav"
    _ramp_wav(rec, 3.0)
    manifest = tmp_path / "m.jsonl"
    write_manifest(str(manifest), [
        normalize_entry(str(rec), start=0.0, end=1.5, id="a"),
        normalize_entry(str(rec), start=1.5, end=3.0, id="b"),
    ])
    out = tmp_path / "tokens"
    dump("random", str(manifest), str(out), seed=45, batch_size=2,
         fbank_mean=13.68, fbank_std=6.19)

    meta = schema.load_meta(str(out))
    assert meta["format"] == "openbeats-tokens/v2"
    assert meta["frontend"]["fbank_mean"] == pytest.approx(13.68)
    assert meta["frontend"]["fbank_std"] == pytest.approx(6.19)
    assert (out / "manifest.jsonl").is_file()     # self-contained dir mirror
    assert (out / "dataset.json").is_file()

    rows = {r["id"]: r for r in schema.iter_rows(str(out))}
    assert set(rows) == {"a", "b"}
    assert rows["a"]["start"] == 0.0 and rows["a"]["end"] == 1.5
    assert rows["b"]["start"] == 1.5 and rows["b"]["end"] == 3.0
    assert rows["a"]["n_codes"] > 0


def test_tokendataset_loads_the_segment_span(tmp_path):
    """The dataset must load each row's *span*, and its code length must equal the
    tokenizer's code count for that exact span (alignment invariant, with segments)."""
    from openbeats.nets.tokenizer import build_tokenizer

    rec = tmp_path / "rec.wav"
    _ramp_wav(rec, 3.0)
    manifest = tmp_path / "m.jsonl"
    write_manifest(str(manifest), [
        normalize_entry(str(rec), start=0.0, end=1.5, id="a"),
        normalize_entry(str(rec), start=1.5, end=3.0, id="b"),
    ])
    out = tmp_path / "tokens"
    dump("random", str(manifest), str(out), seed=45, batch_size=2)

    ds = TokenDataset(str(out))
    tok = build_tokenizer("random", seed=45)
    for i in range(len(ds)):
        item = ds[i]
        assert abs(item["speech_lengths"] - int(1.5 * SR)) <= 2  # span, not whole file
        wav = item["speech"].unsqueeze(0)
        ilens = torch.tensor([item["speech_lengths"]])
        enc = tok.encode(wav, ilens, waveform_input=True)
        assert int(enc["code_lengths"][0]) == item["target_lengths"]


# ---------------------------------------------------------------- fbank guard
def _cfg(fbank_mean):
    return {"encoder_conf": {"fbank_mean": fbank_mean, "fbank_std": 6.19,
                             "beats_config": {"codebook_vocab_size": 1024}}}


def test_fbank_guard_matches_and_mismatches():
    from openbeats.pretraining.engine import _check_dataset_compat

    meta = {"codebook_size": 1024,
            "frontend": {"fbank_mean": 13.68, "fbank_std": 6.19}}
    _check_dataset_compat(_cfg(13.68), meta)        # match -> ok
    with pytest.raises(ValueError, match="fbank_mean mismatch"):
        _check_dataset_compat(_cfg(15.41663), meta)  # mismatch -> raises
