"""Labeled-audio data layer for classification fine-tuning.

Reads a JSONL manifest whose entries carry a label (a class name for multi-class,
or a list of names for multi-label) plus a labels.txt vocabulary (one class per
line). Loads each segment's span at 16 kHz via data.audio.load_audio and maps the
label name(s) to ids. Reuses LengthBucketBatchSampler from the SSL loader for
length-bucketed, rank-sharded batches; collate pads speech with 0.0 and label ids
with -1 (so the model's label_to_onehot treats -1 as padding).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import load_audio
from .loader import LengthBucketBatchSampler

LABEL_PAD = -1
# BEATs' 16x16 patch frontend needs >= ~3200 samples (0.2 s) to emit a single patch;
# a shorter clip yields zero patches -> a fully-masked row -> NaN. Pad up to this so
# every clip (incl. short calls in the eval sets) survives without being dropped.
MIN_SAMPLES = 4000

def read_label_list(path: str) -> list:
    """Read a labels.txt (one class name per line) into an ordered list."""
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]

def probe_length(it) -> int:
    """Span length in samples after resampling to 16 kHz (for length bucketing)."""
    import soundfile as sf

    info = sf.info(it["audio"])
    sr = info.samplerate
    s = 0 if it.get("start") is None else int(round(it["start"] * sr))
    e = info.frames if it.get("end") is None else int(round(it["end"] * sr))
    return int(round(max(0, e - s) * 16000 / sr))

def load_padded_span(it) -> np.ndarray:
    """Load an entry's 16 kHz span, padding short clips up to MIN_SAMPLES."""
    wav, _ = load_audio(it["audio"], start=it.get("start"), end=it.get("end"))
    if wav.shape[0] < MIN_SAMPLES:
        wav = np.pad(wav, (0, MIN_SAMPLES - wav.shape[0]))
    return wav

class LabeledAudioDataset(Dataset):
    """A labeled manifest as a torch Dataset of (speech, label ids) items.

    label_list fixes the name->id order (id = index in the list). Each manifest
    entry's label is a single name (multi-class) or a list of names (multi-label);
    both become a list of ids. n_samples is read once per item (soundfile.info) so
    the batch sampler can bucket by length.
    """

    def __init__(self, manifest: str, label_list, *, multi_label: bool = False):
        from .manifest import read_manifest

        self.label_to_id = {name: i for i, name in enumerate(label_list)}
        self.n_classes = len(label_list)
        self.multi_label = multi_label
        self.items = []
        self.n_samples = []
        for it in read_manifest(manifest):
            if "label" not in it:
                raise ValueError(f"manifest entry {it['id']!r} has no 'label'")
            ids = self._encode_label(it["label"])
            if not multi_label and len(ids) != 1:
                raise ValueError(
                    f"multi-class entry {it['id']!r} must have exactly one label, got {ids}"
                )
            self.items.append((it, ids))
            self.n_samples.append(probe_length(it))
        self.n_samples = np.asarray(self.n_samples, dtype=np.int64)

    def _encode_label(self, label) -> list:
        names = [label] if isinstance(label, str) else list(label)
        try:
            return [self.label_to_id[n] for n in names]
        except KeyError as e:
            raise KeyError(f"label {e.args[0]!r} not in label_list") from None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it, ids = self.items[i]
        wav = load_padded_span(it)
        return {
            "id": it["id"],
            "speech": torch.from_numpy(np.ascontiguousarray(wav)),
            "speech_lengths": int(wav.shape[0]),
            "label": torch.tensor(ids, dtype=torch.long),
            "label_lengths": len(ids),
        }

def cls_collate(batch: list) -> dict:
    """Pad speech with 0.0 and label ids with LABEL_PAD (-1); return model kwargs."""
    B = len(batch)
    tmax = max(b["speech_lengths"] for b in batch)
    lmax = max(b["label_lengths"] for b in batch)

    speech = torch.zeros(B, tmax, dtype=torch.float32)
    label = torch.full((B, lmax), LABEL_PAD, dtype=torch.long)
    speech_lengths = torch.empty(B, dtype=torch.long)
    label_lengths = torch.empty(B, dtype=torch.long)

    for j, b in enumerate(batch):
        sl, ll = b["speech_lengths"], b["label_lengths"]
        speech[j, :sl] = b["speech"]
        label[j, :ll] = b["label"]
        speech_lengths[j] = sl
        label_lengths[j] = ll

    return {
        "speech": speech,
        "speech_lengths": speech_lengths,
        "label": label,
        "label_lengths": label_lengths,
    }

def build_labeled_dataloader(
    manifest: str,
    label_list,
    batch_bins: int,
    *,
    multi_label: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    max_batch_size: int | None = None,
):
    """Build (dataset, dataloader, sampler) for a labeled manifest."""
    dataset = LabeledAudioDataset(manifest, label_list, multi_label=multi_label)
    sampler = LengthBucketBatchSampler(
        dataset.n_samples,
        batch_bins,
        max_batch_size=max_batch_size,
        shuffle=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=cls_collate,
        num_workers=num_workers,
        # Same fork-after-CUDA rationale as the SSL loader: spawn fresh workers.
        **(
            {"multiprocessing_context": "spawn", "persistent_workers": True,
             "pin_memory": False}
            if num_workers > 0
            else {}
        ),
    )
    return dataset, loader, sampler

class AudioManifestDataset(Dataset):
    """A manifest as a torch Dataset of (id, speech) items for batched inference.

    Same audio handling as LabeledAudioDataset (16 kHz span load, MIN_SAMPLES
    short-clip pad) but labels are not required and are ignored; n_samples is
    probed for length bucketing.
    """

    def __init__(self, manifest: str):
        from .manifest import read_manifest

        self.items = list(read_manifest(manifest))
        self.n_samples = np.asarray([probe_length(it) for it in self.items], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        wav = load_padded_span(it)
        return {
            "id": it["id"],
            "speech": torch.from_numpy(np.ascontiguousarray(wav)),
            "speech_lengths": int(wav.shape[0]),
        }

def infer_collate(batch: list) -> dict:
    """Pad speech with 0.0; carry ids and lengths (no labels)."""
    B = len(batch)
    tmax = max(b["speech_lengths"] for b in batch)
    speech = torch.zeros(B, tmax, dtype=torch.float32)
    speech_lengths = torch.empty(B, dtype=torch.long)
    ids = []
    for j, b in enumerate(batch):
        sl = b["speech_lengths"]
        speech[j, :sl] = b["speech"]
        speech_lengths[j] = sl
        ids.append(b["id"])
    return {"id": ids, "speech": speech, "speech_lengths": speech_lengths}

def build_infer_dataloader(
    manifest: str,
    batch_bins: int,
    *,
    num_workers: int = 4,
    max_batch_size: int | None = None,
):
    """Build (dataset, dataloader, sampler) for batched inference over a manifest.

    Single-process: length-bucketed but unshuffled and not rank-sharded, so every
    segment is emitted exactly once (rows carry their id; bucket order != manifest
    order).
    """
    dataset = AudioManifestDataset(manifest)
    sampler = LengthBucketBatchSampler(
        dataset.n_samples,
        batch_bins,
        max_batch_size=max_batch_size,
        shuffle=False,
        world_size=1,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=infer_collate,
        num_workers=num_workers,
        **(
            {"multiprocessing_context": "spawn", "persistent_workers": True,
             "pin_memory": False}
            if num_workers > 0
            else {}
        ),
    )
    return dataset, loader, sampler
