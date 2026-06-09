"""The OpenBEATs token-dataset format: a directory of self-describing Parquet shards.

A dataset is <dir>/shard-*.parquet (plus a manifest.jsonl + dataset.json
mirror). Each row is one utterance (or a segment of a longer recording):

    id         : string        utterance id
    audio      : string        absolute audio path (waveform recomputed at train time)
    start      : float64?      segment start in seconds (null => from file start)
    end        : float64?      segment end in seconds   (null => to file end)
    n_samples  : int64         span length at 16 kHz (for length-bucketing)
    n_codes    : int32         number of patch codes (== patch count)
    codes      : list<int16>   the code sequence, values 1..K (the dump applies the
                               <unk>+1 shift; index 0 reserved)

Dataset-level config (codebook size, frontend incl. fbank stats, tokenizer, code
offset) is stored in every shard's Parquet key-value metadata, so each shard is
self-describing; a top-level dataset.json mirrors it for eyeballing. This module
has no torch dependency on purpose (pure pyarrow/numpy) so it stays light.

start/end (v2) are nullable; a v1 dataset without them reads as whole-file.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = "openbeats-tokens/v2"
META_KEY = b"openbeats_meta"
CODES_OFFSET = 1  # stored codes are 1..K; 0 reserved for <unk>

DEFAULT_FRONTEND = {
    "sample_rate": 16000,
    "n_mels": 128,
    "frame_length_ms": 25,
    "frame_shift_ms": 10,
    "patch_size": 16,
}

_ARROW_CODE_DTYPE = {"int16": pa.int16(), "int32": pa.int32()}
_NUMPY_CODE_DTYPE = {"int16": np.int16, "int32": np.int32}

def arrow_schema(meta: dict, code_dtype: str = "int16") -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("start", pa.float64()),
            pa.field("end", pa.float64()),
            pa.field("n_samples", pa.int64()),
            pa.field("n_codes", pa.int32()),
            pa.field("codes", pa.list_(_ARROW_CODE_DTYPE[code_dtype])),
        ]
    )
    return schema.with_metadata({META_KEY: json.dumps(meta).encode("utf-8")})

def build_meta(
    codebook_size: int,
    tokenizer: dict,
    frontend: Optional[dict] = None,
    waveform_input: bool = True,
    code_dtype: str = "int16",
    codes_offset: int = CODES_OFFSET,
    fbank_mean: Optional[float] = None,
    fbank_std: Optional[float] = None,
) -> dict:
    fe = dict(frontend or DEFAULT_FRONTEND)
    # fbank normalization is part of the frontend contract: the tokenizer that
    # produced the codes and the encoder that trains on them must use the same
    # stats, so record them here and let training assert a match.
    if fbank_mean is not None:
        fe["fbank_mean"] = float(fbank_mean)
    if fbank_std is not None:
        fe["fbank_std"] = float(fbank_std)
    return {
        "format": FORMAT,
        "codebook_size": int(codebook_size),
        "codes_offset": int(codes_offset),
        "waveform_input": bool(waveform_input),
        "frontend": fe,
        "tokenizer": dict(tokenizer),
        "code_dtype": code_dtype,
    }

@dataclass
class TokenDatasetWriter:
    """Buffer rows and flush them to one shard-{shard_id:05d}.parquet.

    The writer is "dumb": it stores whatever code values it is handed. The dump
    driver applies the codes_offset (+1) shift before calling :meth:add, so
    on-disk values are 1..K.
    """

    out_dir: str
    meta: dict
    shard_id: int = 0
    _ids: list = field(default_factory=list)
    _audios: list = field(default_factory=list)
    _starts: list = field(default_factory=list)
    _ends: list = field(default_factory=list)
    _n_samples: list = field(default_factory=list)
    _codes: list = field(default_factory=list)

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._code_dtype = self.meta.get("code_dtype", "int16")

    def add(self, id: str, audio: str, n_samples: int, codes,
            start=None, end=None) -> None:
        codes = np.asarray(codes, dtype=_NUMPY_CODE_DTYPE[self._code_dtype])
        assert codes.ndim == 1
        self._ids.append(str(id))
        self._audios.append(str(audio))
        self._starts.append(None if start is None else float(start))
        self._ends.append(None if end is None else float(end))
        self._n_samples.append(int(n_samples))
        self._codes.append(codes)

    def __len__(self) -> int:
        return len(self._ids)

    def flush(self) -> Optional[str]:
        """Write the buffered rows to a shard file; return its path (or None)."""
        if not self._ids:
            return None
        schema = arrow_schema(self.meta, self._code_dtype)
        table = pa.table(
            {
                "id": pa.array(self._ids, pa.string()),
                "audio": pa.array(self._audios, pa.string()),
                "start": pa.array(self._starts, pa.float64()),
                "end": pa.array(self._ends, pa.float64()),
                "n_samples": pa.array(self._n_samples, pa.int64()),
                "n_codes": pa.array([len(c) for c in self._codes], pa.int32()),
                "codes": pa.array(
                    [c.tolist() for c in self._codes],
                    pa.list_(_ARROW_CODE_DTYPE[self._code_dtype]),
                ),
            },
            schema=schema,
        )
        path = os.path.join(self.out_dir, f"shard-{self.shard_id:05d}.parquet")
        pq.write_table(table, path)
        self._ids, self._audios, self._starts, self._ends = [], [], [], []
        self._n_samples, self._codes = [], []
        return path

def write_dataset_json(out_dir: str, meta: dict) -> str:
    path = os.path.join(out_dir, "dataset.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path

def shard_paths(path: str) -> list:
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "shard-*.parquet")))

def load_meta(path: str) -> dict:
    """Dataset metadata, from a shard's Parquet kv-metadata (fallback dataset.json)."""
    shards = shard_paths(path)
    if shards:
        kv = pq.read_schema(shards[0]).metadata or {}
        if META_KEY in kv:
            return json.loads(kv[META_KEY].decode("utf-8"))
    dj = os.path.join(path, "dataset.json")
    if os.path.isfile(dj):
        with open(dj) as f:
            return json.load(f)
    raise FileNotFoundError(f"No dataset metadata found under '{path}'.")

def read_table(path: str, columns: Optional[list] = None) -> pa.Table:
    """Read all shards (optionally projecting columns) into one Arrow table."""
    shards = shard_paths(path)
    if not shards:
        raise FileNotFoundError(f"No shards (shard-*.parquet) found under '{path}'.")
    return pa.concat_tables([pq.read_table(p, columns=columns) for p in shards])

def iter_rows(path: str) -> Iterator[dict]:
    table = read_table(path)
    for batch in table.to_batches():
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            yield {k: d[k][i] for k in d}

def validate(path: str, check_audio_exists: bool = False) -> dict:
    """Sanity-check a dataset; returns a small stats dict, raises on a hard error."""
    meta = load_meta(path)
    K = meta["codebook_size"]
    offset = meta.get("codes_offset", CODES_OFFSET)
    lo, hi = offset, offset + K - 1

    n_rows = 0
    n_codes_total = 0
    min_code = None
    max_code = None
    for row in iter_rows(path):
        codes = row["codes"]
        if len(codes) != row["n_codes"]:
            raise ValueError(
                f"id={row['id']}: n_codes={row['n_codes']} != len(codes)={len(codes)}"
            )
        if codes:
            cmin, cmax = min(codes), max(codes)
            if cmin < lo or cmax > hi:
                raise ValueError(
                    f"id={row['id']}: code out of range [{lo},{hi}] "
                    f"(got [{cmin},{cmax}])"
                )
            min_code = cmin if min_code is None else min(min_code, cmin)
            max_code = cmax if max_code is None else max(max_code, cmax)
        if check_audio_exists and not os.path.isfile(row["audio"]):
            raise FileNotFoundError(f"id={row['id']}: audio not found: {row['audio']}")
        n_rows += 1
        n_codes_total += len(codes)

    return {
        "rows": n_rows,
        "total_codes": n_codes_total,
        "avg_codes": (n_codes_total / n_rows) if n_rows else 0,
        "code_range": [min_code, max_code],
        "codebook_size": K,
        "codes_offset": offset,
    }
