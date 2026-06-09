"""Stage B data layer: read a Parquet token dataset and batch it for pretraining.

The dataset stores codes already shifted to 1..K (the dump applied the <unk>+1
shift, design §3.1), so this layer does **no** arithmetic on codes: it loads the
waveform from the row's ``audio`` path, returns the code sequence as-is, and the
collate pads codes with ``-1`` (the verbatim model maps ``target-1 -> -2 ==
ignore_id``). A length-bucket batch sampler groups similar-length clips to bound
padding; for distributed runs each rank takes a disjoint slice of the batches.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from . import dataset as schema
from .audio import load_audio

logger = logging.getLogger("openbeats.data")

PAD_CODE = -1  # model does target-1 -> -2 == ignore_id (CrossEntropyLoss ignore_index)


class TokenDataset(Dataset):
    """A Parquet token dataset as a torch Dataset of pretraining items.

    Degenerate clips are filtered out: any row with ``n_codes == 0`` (ultra-short
    audio -> 0 patches -> masking nothing -> NaN loss) is always dropped, and
    ``min_samples``/``max_samples`` (16 kHz sample counts) bound clip length — mirrors
    ESPnet's ``min/max_wav_duration``.
    """

    def __init__(self, path: str, *, min_samples: int = 0, max_samples: int | None = None):
        self.path = path
        self.meta = schema.load_meta(path)
        table = schema.read_table(path)
        n_samples = np.asarray(table["n_samples"].to_pylist(), dtype=np.int64)
        n_codes = np.asarray(table["n_codes"].to_pylist(), dtype=np.int64)

        keep = (n_codes > 0) & (n_samples >= int(min_samples))
        if max_samples:
            keep &= n_samples <= int(max_samples)
        if not keep.all():
            import pyarrow as pa

            kept = np.nonzero(keep)[0]
            logger.info(
                "TokenDataset: kept %d/%d clips (dropped %d zero-code/out-of-range)",
                len(kept), len(keep), len(keep) - len(kept),
            )
            table = table.take(pa.array(kept))
            n_samples, n_codes = n_samples[kept], n_codes[kept]

        cols = table.column_names
        self.ids = table["id"].to_pylist()
        self.audios = table["audio"].to_pylist()
        # start/end are v2; a v1 dataset without them reads as whole-file (None).
        self.starts = table["start"].to_pylist() if "start" in cols else [None] * len(self.ids)
        self.ends = table["end"].to_pylist() if "end" in cols else [None] * len(self.ids)
        self.n_samples = n_samples
        self.n_codes = n_codes
        # Materialize codes to plain numpy arrays. Keeping an arrow column here makes
        # the dataset slow/deadlock-prone to pickle into spawn DataLoader workers
        # (especially after a .take() filter); a list of small int arrays pickles
        # trivially. Codes are tiny (~sum(n_codes) int16, a few MB).
        self._codes = [np.asarray(c, dtype=np.int64) for c in table["codes"].to_pylist()]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def codebook_size(self) -> int:
        return int(self.meta["codebook_size"])

    def __getitem__(self, i: int) -> dict:
        # read only this segment's span (None/None => whole file); mono float32 @ 16 kHz
        wav, _ = load_audio(self.audios[i], start=self.starts[i], end=self.ends[i])
        codes = self._codes[i]
        return {
            "id": self.ids[i],
            "speech": torch.from_numpy(np.ascontiguousarray(wav)),
            "speech_lengths": int(wav.shape[0]),
            "target": torch.from_numpy(codes),
            "target_lengths": int(codes.shape[0]),
        }


def collate(batch: list) -> dict:
    """Pad speech with 0.0 and targets with PAD_CODE (-1); return model kwargs."""
    B = len(batch)
    tmax = max(b["speech_lengths"] for b in batch)
    cmax = max(b["target_lengths"] for b in batch)

    speech = torch.zeros(B, tmax, dtype=torch.float32)
    target = torch.full((B, cmax), PAD_CODE, dtype=torch.long)
    speech_lengths = torch.empty(B, dtype=torch.long)
    target_lengths = torch.empty(B, dtype=torch.long)

    for j, b in enumerate(batch):
        sl, cl = b["speech_lengths"], b["target_lengths"]
        speech[j, :sl] = b["speech"]
        target[j, :cl] = b["target"]
        speech_lengths[j] = sl
        target_lengths[j] = cl

    return {
        "speech": speech,
        "speech_lengths": speech_lengths,
        "target": target,
        "target_lengths": target_lengths,
    }


class LengthBucketBatchSampler(Sampler):
    """Yield batches of indices grouped by length to ~``batch_bins`` per batch.

    ``batch_bins`` is a budget on the summed length of a batch (``bin_by`` selects
    samples or codes). Clips are sorted by length, packed greedily into batches,
    the batch order is shuffled (seed+epoch), and for distributed training each
    rank takes ``batches[rank::world_size]``. ``set_epoch`` reshuffles.
    """

    def __init__(
        self,
        lengths,
        batch_bins: int,
        *,
        max_batch_size: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_bins = int(batch_bins)
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0
        self._batches = self._build_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if self.shuffle:
            self._batches = self._build_batches()

    def _build_batches(self) -> list:
        order = np.argsort(self.lengths, kind="stable")
        batches, cur, cur_max = [], [], 0
        for idx in order:
            ln = int(self.lengths[idx])
            new_max = max(cur_max, ln)
            over_bins = cur and new_max * (len(cur) + 1) > self.batch_bins
            over_cap = self.max_batch_size and len(cur) >= self.max_batch_size
            if over_bins or over_cap:
                batches.append(cur)
                cur, cur_max = [], 0
                new_max = ln
            cur.append(int(idx))
            cur_max = new_max
        if cur and not (self.drop_last and len(batches) > 0):
            batches.append(cur)

        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(batches)
        # Every rank must run the SAME number of optimizer steps per epoch, or the
        # per-step gradient all-reduce desyncs and NCCL deadlocks. All ranks build
        # the identical `batches` list (same seed), so drop the remainder to a
        # multiple of world_size, then take a disjoint, equal-sized slice. Up to
        # world_size-1 batches are dropped per epoch (different ones each epoch
        # because the order is reshuffled in set_epoch).
        if self.world_size > 1:
            usable = (len(batches) // self.world_size) * self.world_size
            batches = batches[:usable]
        batches = batches[self.rank :: self.world_size]
        return batches

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def build_dataloader(
    path: str,
    batch_bins: int,
    *,
    bin_by: str = "samples",
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    max_batch_size: int | None = None,
    drop_last: bool = False,
    min_samples: int = 0,
    max_samples: int | None = None,
):
    """Build (dataset, dataloader, sampler) for a token dataset."""
    dataset = TokenDataset(path, min_samples=min_samples, max_samples=max_samples)
    lengths = dataset.n_samples if bin_by == "samples" else dataset.n_codes
    sampler = LengthBucketBatchSampler(
        lengths,
        batch_bins,
        max_batch_size=max_batch_size,
        shuffle=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
        drop_last=drop_last,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        # Workers run torchaudio resample (torch CPU/OpenMP ops) when audio isn't
        # already 16 kHz. Forking after the parent has initialized CUDA (DeepSpeed)
        # deadlocks on the inherited threadpool state, so spawn fresh worker procs
        # instead; persistent_workers amortizes the spawn over epochs. pin_memory is
        # off: pinned-memory + spawn is a known startup-hang source and the win is
        # marginal next to audio I/O.
        **(
            {"multiprocessing_context": "spawn", "persistent_workers": True,
             "pin_memory": False}
            if num_workers > 0
            else {}
        ),
    )
    return dataset, loader, sampler
