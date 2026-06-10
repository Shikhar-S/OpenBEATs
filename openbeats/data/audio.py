"""Audio loading: read a file (or a segment span) as a mono float32 waveform at the
target rate. For segments only the requested span is read at native rate (cheap for
long recordings) and then resampled — whole tapes are never loaded into memory."""

from functools import lru_cache

import numpy as np

TARGET_SR = 16000

@lru_cache(maxsize=8)
def _resampler(orig_sr: int, target_sr: int):
    # A transforms.Resample reuses its filter kernel across calls; functional.resample
    # rebuilds it every time. Cached per process (one per worker under a DataLoader).
    import torchaudio

    return torchaudio.transforms.Resample(orig_sr, target_sr)

def load_audio(path, target_sr: int = TARGET_SR, *, start=None, end=None):
    """Load path as a mono float32 waveform resampled to target_sr.

    start/end (seconds, optional) select a span: only that slice is read at
    the file's native rate via a seek, then resampled. Both None => whole file.
    """
    import soundfile as sf

    if start is None and end is None:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        info = sf.info(path)
        sr = info.samplerate
        s = 0 if start is None else max(0, int(round(start * sr)))
        e = info.frames if end is None else min(info.frames, int(round(end * sr)))
        wav, sr = sf.read(path, dtype="float32", always_2d=False,
                          start=s, frames=max(0, e - s))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import torch

        wav = _resampler(sr, target_sr)(
            torch.from_numpy(np.ascontiguousarray(wav))
        ).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32), target_sr
