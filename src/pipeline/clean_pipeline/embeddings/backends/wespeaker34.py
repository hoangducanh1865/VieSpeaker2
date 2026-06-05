"""WeSpeaker ResNet34-LM backend via pyannote.audio (256-d).

pyannote.audio is already a project dependency and ships the WeSpeakerResNet34
model class, so we load the checkpoint with `Model.from_pretrained` and pull a
single embedding per segment with `Inference(window="whole").crop(...)`.

Weight dir (gitignored, prepared by scripts/prepare_embeddings.py):
    embeddings/weights/wespeaker34/{pytorch_model.bin, config.yaml}

Falls back to the public HF id `pyannote/wespeaker-voxceleb-resnet34-LM`
(needs HUGGINGFACE_ACCESS_TOKEN + network) if the local weight is absent.
"""

import os
import numpy as np

from ._common import l2_normalize
from ..registry import WEIGHTS_DIR

_DIR = os.path.join(WEIGHTS_DIR, "wespeaker34")
_BIN = os.path.join(_DIR, "pytorch_model.bin")
_HF_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"


class WeSpeaker34Embedder:
    def __init__(self, device="cuda"):
        import torch
        from pyannote.audio import Model, Inference

        token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
        last_err = None
        model = None
        for target in (_DIR, _BIN, _HF_ID):
            if target in (_DIR, _BIN) and not os.path.exists(target):
                continue
            try:
                kwargs = {}
                if target == _HF_ID and token:
                    kwargs["use_auth_token"] = token
                model = Model.from_pretrained(target, **kwargs)
                break
            except Exception as e:  # try the next candidate
                last_err = e
        if model is None:
            raise RuntimeError(
                f"Could not load WeSpeaker34 from {_DIR}, {_BIN}, or HF '{_HF_ID}'. "
                f"Last error: {last_err}"
            )

        dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.inference = Inference(model, window="whole", device=torch.device(dev))
        self.dim = 256

    def extract(self, audio_path, start, end):
        from pyannote.core import Segment

        if end - start < 0.1:
            return None
        try:
            emb = self.inference.crop(audio_path, Segment(start, end))
        except Exception:
            return None
        return l2_normalize(np.asarray(emb).squeeze())
