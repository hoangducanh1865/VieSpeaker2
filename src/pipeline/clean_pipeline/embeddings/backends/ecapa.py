"""ECAPA-TDNN backend (TalkNet/clovaai, VoxCeleb EN, 192-d).

Wraps the existing, validated loader/extractor in clean.py so behaviour is
identical to the original default path. Weight: models/ecapa_tdnn/exps/pretrain.model.
"""


class EcapaEmbedder:
    def __init__(self, device="cuda"):
        # Imported here (not at module top) to avoid a circular import:
        # clean.py imports the registry, which lazily imports this backend.
        from clean import _load_ecapa_model, _extract_ecapa_embedding
        self._extract = _extract_ecapa_embedding
        self.model = _load_ecapa_model()  # CPU; segments are short
        self.dim = 192

    def extract(self, audio_path, start, end):
        return self._extract(self.model, audio_path, start, end)
