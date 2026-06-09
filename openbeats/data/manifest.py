"""JSONL manifest of audio segments — the corpus-agnostic data interface.

One JSON object per line:

    {"id": "rec1_000", "audio": "/abs/rec1.wav", "start": 0.0, "end": 10.0}
    {"id": "clipA",    "audio": "/abs/clipA.flac"}

``audio`` is required; ``start``/``end`` are optional seconds (omit => whole file —
long recordings are windowed without pre-cutting). ``id`` is optional (defaults to
the filename stem, plus a start-offset suffix when a span is given). Extra keys are
ignored. Blank lines are skipped; a line that is a bare path (no leading ``{``) is
treated as a whole-file entry. The loader/tokenizer read only the requested span.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Optional


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def normalize_entry(audio, start=None, end=None, id=None) -> dict:
    """Canonical manifest entry: abs ``audio`` path, float|None ``start``/``end``,
    str ``id`` (auto-derived from the stem + start offset when omitted)."""
    audio = os.path.abspath(os.path.expanduser(str(audio)))
    start = None if start is None else float(start)
    end = None if end is None else float(end)
    if id is None or id == "":
        id = _stem(audio)
        if start is not None or end is not None:
            id = f"{id}_{int(round((start or 0.0) * 1000)):09d}"
    return {"id": str(id), "audio": audio, "start": start, "end": end}


def _parse_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    if line[0] == "{":
        obj = json.loads(line)
        return normalize_entry(obj["audio"], obj.get("start"), obj.get("end"), obj.get("id"))
    return normalize_entry(line)


def iter_manifest(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            entry = _parse_line(line)
            if entry is not None:
                yield entry


def read_manifest(path: str) -> list:
    """Parse a manifest into a list of normalized ``{id, audio, start, end}`` dicts."""
    return list(iter_manifest(path))


def write_manifest(path: str, entries) -> str:
    """Write normalized entries as JSONL (omitting null ``start``/``end``)."""
    with open(path, "w") as f:
        for e in entries:
            obj = {"id": e["id"], "audio": e["audio"]}
            if e.get("start") is not None:
                obj["start"] = e["start"]
            if e.get("end") is not None:
                obj["end"] = e["end"]
            f.write(json.dumps(obj) + "\n")
    return path
