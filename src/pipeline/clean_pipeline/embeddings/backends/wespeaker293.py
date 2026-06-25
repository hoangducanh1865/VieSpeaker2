"""WeSpeaker ResNet293-LM backend via the bundled ONNX exporter.

Input  : feats  [B, T, 80]  (kaldi log-mel fbank, CMN)
Output : embs   [B, 256]

This is the most portable backend — no architecture code needed, just
onnxruntime + a kaldi fbank front-end. Weight (gitignored, prepared by
scripts/prepare_embeddings.py):
    embeddings/weights/wespeaker293/speaker-embedding.onnx
"""

import os
import numpy as np

from ._common import load_segment_waveform, compute_kaldi_fbank, l2_normalize
from ..registry import WEIGHTS_DIR

_ONNX_PATH = os.path.join(WEIGHTS_DIR, "wespeaker293", "speaker-embedding.onnx")


class WeSpeaker293Embedder:
    def __init__(self, device="cuda"):
        import onnxruntime as ort

        if not os.path.exists(_ONNX_PATH):
            raise FileNotFoundError(
                f"WeSpeaker293 ONNX not found: {_ONNX_PATH}. "
                "Run scripts/prepare_embeddings.py to copy it from stuff/."
            )
        avail = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if (device == "cuda" and "CUDAExecutionProvider" in avail)
            else ["CPUExecutionProvider"]
        )
        self.sess = ort.InferenceSession(_ONNX_PATH, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.dim = 256

    def extract(self, audio_path, start, end):
        wav = load_segment_waveform(audio_path, start, end)
        if wav is None:
            return None
        feat = compute_kaldi_fbank(wav, num_mel_bins=80, dither=0.0)  # (T, 80)
        if feat.shape[0] < 5:
            return None
        feats = feat.numpy().astype(np.float32)[None, :, :]  # (1, T, 80)
        emb = self.sess.run([self.out_name], {self.in_name: feats})[0]
        return l2_normalize(emb.squeeze())
