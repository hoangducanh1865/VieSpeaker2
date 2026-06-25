"""Shared helpers for embedding backends (audio cropping + kaldi fbank)."""

import numpy as np
import torch

SAMPLE_RATE = 16000


def load_segment_waveform(audio_path, start, end, sample_rate=SAMPLE_RATE):
    """Load a mono [start, end] crop as a (1, T) float tensor at `sample_rate`.

    The full file is loaded + resampled once and cached (see viespeaker.audio),
    so repeated per-segment calls only slice an in-memory tensor.

    Returns None if the crop is shorter than 10 ms (too short for a frame).
    """
    from viespeaker.audio import load_mono

    waveform = load_mono(audio_path, sample_rate)
    s = max(0, int(round(start * sample_rate)))
    e = int(round(end * sample_rate))
    chunk = waveform[:, s:e]
    if chunk.shape[1] < int(0.01 * sample_rate):
        return None
    return chunk


def compute_kaldi_fbank(waveform, num_mel_bins=80, dither=0.0, sample_rate=SAMPLE_RATE):
    """Kaldi-compatible log-mel fbank used by WeSpeaker / CAM++ / ReDimNet.

    `waveform` is a (1, T) tensor in [-1, 1]. Returns a (frames, num_mel_bins)
    tensor with per-utterance mean normalisation (CMN), matching the WeSpeaker
    feature pipeline (see asv_model/ecapa/get_model.py).
    """
    import torchaudio.compliance.kaldi as kaldi

    feat = kaldi.fbank(
        waveform * (1 << 15),
        num_mel_bins=num_mel_bins,
        frame_length=25,
        frame_shift=10,
        dither=dither,
        sample_frequency=sample_rate,
        window_type="hamming",
        use_energy=False,
    )
    feat = feat - torch.mean(feat, dim=0, keepdim=True)  # CMN
    return feat


def l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    return vec if n < 1e-8 else vec / n
