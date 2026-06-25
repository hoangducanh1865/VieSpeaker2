"""Registry of speaker-embedding backends.

Each backend exposes a single method:

    extract(audio_path: str, start: float, end: float) -> Optional[np.ndarray]

returning an L2-normalised 1-D embedding (or None if the segment is too short /
extraction failed). Backends are imported lazily so importing the registry never
pulls heavy model code (and avoids a circular import with clean.py).

Available backends:
    ecapa        ECAPA-TDNN (TalkNet/clovaai, VoxCeleb EN, 192-d)  [bundled]
    wespeaker34  WeSpeaker ResNet34 via pyannote.audio (256-d)
    wespeaker293 WeSpeaker ResNet293 via bundled ONNX (256-d)
    campplus     CAM++ (3D-Speaker, zh-cn, 192-d)                  [best-effort]
    redimnet     ReDimNet-B6 (wespeaker, 192-d)                    [best-effort]
"""

import os

from viespeaker import paths

_HERE = os.path.dirname(os.path.abspath(__file__))
# Embedding weights now live in the external assets dir (see viespeaker.paths),
# not in-repo. Override via VIESPEAKER2_ASSETS.
WEIGHTS_DIR = str(paths.EMBEDDINGS_WEIGHTS)

KNOWN_BACKENDS = ["ecapa", "wespeaker34", "wespeaker293", "campplus", "redimnet"]

# process-level cache: (name, device) -> embedder instance
_CACHE = {}


def get_embedder(name: str, device: str = "cuda"):
    """Return a cached embedder instance for `name`.

    Raises ValueError for unknown names and propagates the backend's own load
    error (FileNotFoundError / ImportError / RuntimeError) so the caller can
    decide whether to skip the scenario.
    """
    name = (name or "ecapa").lower()
    key = (name, device)
    if key in _CACHE:
        return _CACHE[key]

    if name == "ecapa":
        from .backends.ecapa import EcapaEmbedder
        emb = EcapaEmbedder(device=device)
    elif name == "wespeaker34":
        from .backends.wespeaker34 import WeSpeaker34Embedder
        emb = WeSpeaker34Embedder(device=device)
    elif name == "wespeaker293":
        from .backends.wespeaker293 import WeSpeaker293Embedder
        emb = WeSpeaker293Embedder(device=device)
    elif name == "campplus":
        from .backends.campplus import CamPPlusEmbedder
        emb = CamPPlusEmbedder(device=device)
    elif name == "redimnet":
        from .backends.redimnet import ReDimNetEmbedder
        emb = ReDimNetEmbedder(device=device)
    else:
        raise ValueError(f"Unknown embedding backend '{name}'. Known: {KNOWN_BACKENDS}")

    _CACHE[key] = emb
    return emb
