"""Offline tests for stage A: the Parquet schema and the tokenize-dump driver.

Synthesizes a few short WAVs, dumps codes with the random tokenizer, and checks
the round trip (metadata, row count, code range 1..K, n_codes == len(codes)).
"""

import json

import numpy as np
import pytest
import soundfile as sf

from openbeats.tokenize import schema
from openbeats.tokenize.dump import dump, read_manifest

SR = 16000


def _write_wav(path, seconds, freq):
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    sf.write(path, (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)


def _make_corpus(tmp_path, specs):
    manifest = tmp_path / "corpus.jsonl"
    with open(manifest, "w") as f:
        for i, (sec, freq) in enumerate(specs):
            wav = tmp_path / f"clip{i}.wav"
            _write_wav(str(wav), sec, freq)
            f.write(json.dumps({"id": f"clip{i}", "audio": str(wav)}) + "\n")
    return str(manifest)


def test_schema_writer_reader_roundtrip(tmp_path):
    meta = schema.build_meta(codebook_size=1024, tokenizer={"type": "beats_random"})
    w = schema.TokenDatasetWriter(str(tmp_path), meta, shard_id=0)
    w.add("a", "/x/a.wav", 16000, [1, 2, 3, 1024])
    w.add("b", "/x/b.wav", 8000, [5, 5, 5])
    path = w.flush()
    assert path is not None

    assert schema.load_meta(str(tmp_path))["codebook_size"] == 1024
    rows = list(schema.iter_rows(str(tmp_path)))
    assert {r["id"] for r in rows} == {"a", "b"}
    a = next(r for r in rows if r["id"] == "a")
    assert a["n_codes"] == 4 and list(a["codes"]) == [1, 2, 3, 1024]
    stats = schema.validate(str(tmp_path))
    assert stats["rows"] == 2 and stats["code_range"] == [1, 1024]


def test_dump_random_tokenizer_roundtrip(tmp_path):
    manifest = _make_corpus(tmp_path, [(0.6, 440), (1.0, 660), (0.8, 330)])
    out = tmp_path / "tokens"

    dump("random", manifest, str(out), seed=45, batch_size=2)

    meta = schema.load_meta(str(out))
    assert meta["format"] == "openbeats-tokens/v1"
    assert meta["codes_offset"] == 1
    K = meta["codebook_size"]

    stats = schema.validate(str(out))
    assert stats["rows"] == 3
    lo, hi = stats["code_range"]
    assert lo >= 1 and hi <= K  # the +1 shift is applied -> values in 1..K
    # dataset.json mirror was written (finalize, shard 0)
    assert (out / "dataset.json").is_file()


def test_dump_sharding_disjoint(tmp_path):
    manifest = _make_corpus(tmp_path, [(0.5, f) for f in (300, 400, 500, 600)])
    out = tmp_path / "tokens"

    dump("random", manifest, str(out), seed=1, batch_size=2, num_shards=2, shard_id=0,
         finalize=False)
    dump("random", manifest, str(out), seed=1, batch_size=2, num_shards=2, shard_id=1,
         finalize=False)

    assert len(schema.shard_paths(str(out))) == 2
    ids = {r["id"] for r in schema.iter_rows(str(out))}
    assert ids == {"clip0", "clip1", "clip2", "clip3"}  # union covers corpus, no dup


def test_read_manifest_bare_paths(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text("/data/foo.wav\n/data/bar.flac\n")
    items = read_manifest(str(manifest))
    assert [it["id"] for it in items] == ["foo", "bar"]
    assert items[0]["audio"].endswith("/data/foo.wav")
