"""ReDimNet-B6 backend (best-effort).

ReDimNet-B6 is a strong recent speaker-embedding architecture. Faithfully loading
the bundled WeSpeaker-format checkpoint (embeddings/weights/redimnet/model_120.pt)
would require copying the full WeSpeaker ReDimNet implementation; instead this
backend loads the official IDRnD ReDimNet-B6 via torch.hub (needs network on first
use; cached afterwards) and uses its built-in waveform front-end.

If torch.hub is unavailable the backend raises and the scenario runner skips it.
"""

import torch

from ._common import load_segment_waveform, l2_normalize


class ReDimNetEmbedder:
    def __init__(self, device="cuda"):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        try:
            # trust_repo=True avoids the interactive "do you trust this repo?"
            # prompt that would otherwise hang a non-interactive SLURM job.
            model = torch.hub.load(
                "IDRnD/ReDimNet", "b6", pretrained=True, finetuned=False,
                verbose=False, trust_repo=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load ReDimNet-B6 from torch.hub (needs network on first run): {e}"
            )
        self.model = model.to(self.device).eval()
        self.dim = 192

    def extract(self, audio_path, start, end):
        wav = load_segment_waveform(audio_path, start, end)
        if wav is None:
            return None
        x = wav.to(self.device)  # (1, T) waveform; ReDimNet has its own features
        with torch.no_grad():
            emb = self.model(x)
        return l2_normalize(emb.squeeze().cpu().numpy())
