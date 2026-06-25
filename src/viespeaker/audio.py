"""Cached audio loading shared by the embedding backends.

The old code called ``torchaudio.load(audio_path)`` (which decodes the WHOLE
file) once per segment, then cropped — re-reading a multi-minute WAV hundreds of
times. :func:`load_mono` loads + resamples + downmixes a file ONCE and caches the
result, so per-segment extraction just slices an in-memory tensor.

The math is identical to the previous inline load (same resample, same mono
take), so cached extraction is numerically equivalent — only faster.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=2)
def load_mono(audio_path: str, sample_rate: int):
    """Return a mono ``(1, T)`` float tensor at ``sample_rate`` (cached per file)."""
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]
    return waveform


def clear_cache() -> None:
    load_mono.cache_clear()
