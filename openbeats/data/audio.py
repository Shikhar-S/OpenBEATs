"""Audio loading: read a file as a mono float32 waveform at the target rate."""

import numpy as np

TARGET_SR = 16000


def load_audio(path, target_sr: int = TARGET_SR):
    """Load an audio file as a mono float32 waveform resampled to target_sr."""
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import torch
        import torchaudio  # already required (kaldi fbank); avoids a librosa dep

        wav = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), sr, target_sr
        ).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32), target_sr
