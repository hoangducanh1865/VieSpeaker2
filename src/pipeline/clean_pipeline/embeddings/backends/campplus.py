"""CAM++ backend (3D-Speaker, zh-cn, 192-d) — best-effort.

Mandarin-trained model; phonetically closer to Vietnamese than the English
VoxCeleb ECAPA, so worth comparing. Weight (gitignored, prepared by
scripts/prepare_embeddings.py):
    embeddings/weights/campplus/campplus_cn_common.bin
"""

import os
import torch

from ._common import load_segment_waveform, compute_kaldi_fbank, l2_normalize
from .arch.campplus import CAMPPlus
from ..registry import WEIGHTS_DIR

_BIN = os.path.join(WEIGHTS_DIR, "campplus", "campplus_cn_common.bin")


class CamPPlusEmbedder:
    def __init__(self, device="cuda"):
        if not os.path.exists(_BIN):
            raise FileNotFoundError(
                f"CAM++ weight not found: {_BIN}. Run scripts/prepare_embeddings.py."
            )
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model = CAMPPlus(feat_dim=80, embedding_size=192)
        state = torch.load(_BIN, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if len(missing) > 20:  # a real architecture mismatch, not just a head
            raise RuntimeError(
                f"CAM++ state_dict mismatch: {len(missing)} missing keys (e.g. {missing[:3]})."
            )
        self.model.to(self.device).eval()
        self.dim = 192

    def extract(self, audio_path, start, end):
        wav = load_segment_waveform(audio_path, start, end)
        if wav is None:
            return None
        feat = compute_kaldi_fbank(wav, num_mel_bins=80, dither=0.0)  # (T, 80)
        if feat.shape[0] < 5:
            return None
        x = feat.unsqueeze(0).to(self.device)  # (1, T, 80)
        with torch.no_grad():
            emb = self.model(x).squeeze(0).cpu().numpy()
        return l2_normalize(emb)
