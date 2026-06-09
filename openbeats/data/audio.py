"""Audio loading: read a file (or a segment span) as a mono float32 waveform at the
target rate. For segments only the requested span is read at native rate (cheap for
long recordings) and then resampled — whole tapes are never loaded into memory."""

import numpy as np

TARGET_SR = 16000


def load_audio(path, target_sr: int = TARGET_SR, *, start=None, end=None):
    """Load ``path`` as a mono float32 waveform resampled to ``target_sr``.

    ``start``/``end`` (seconds, optional) select a span: only that slice is read at
    the file's native rate via a seek, then resampled. Both ``None`` => whole file.
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
        import torchaudio  # already required (kaldi fbank); avoids a librosa dep

        wav = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), sr, target_sr
        ).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32), target_sr
