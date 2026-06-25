"""Central path resolution for VieSpeaker2.

All large model weights and the audio/video test data live **outside** the git
repo, under an *assets directory*. By default this is a sibling of the repo:

    <repo_parent>/VieSpeaker2_assets

so on the server (repo at ``~/anhhd/sv/VieSpeaker2``) it resolves to
``~/anhhd/sv/VieSpeaker2_assets`` with no configuration. Override with the
``VIESPEAKER2_ASSETS`` environment variable. The audio/video test data may be
overridden separately with ``VIESPEAKER2_DATA`` (defaults to ``<assets>/data``).

Ground-truth label ``.txt`` files stay in the repo (small, version-controlled),
exposed here as :data:`LABEL_DIR`.

Import this module instead of building ``_HERE``-relative paths by hand:

    from viespeaker import paths
    embedder_weights = paths.EMBEDDINGS_WEIGHTS / "wespeaker34"
"""

from __future__ import annotations

import os
from pathlib import Path

# src/viespeaker/paths.py -> parents[0]=viespeaker, [1]=src, [2]=repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT: Path = REPO_ROOT / "src"


def _env_path(var: str) -> Path | None:
    val = os.environ.get(var)
    return Path(val).expanduser().resolve() if val else None


# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #
ASSETS_ROOT: Path = _env_path("VIESPEAKER2_ASSETS") or (REPO_ROOT.parent / "VieSpeaker2_assets")
MODELS_ROOT: Path = ASSETS_ROOT / "models"
DATA_ROOT: Path = _env_path("VIESPEAKER2_DATA") or (ASSETS_ROOT / "data")

# --------------------------------------------------------------------------- #
# Test data — large media live in the assets dir; labels stay in the repo
# --------------------------------------------------------------------------- #
TEST_SET: Path = DATA_ROOT / "diarization_test_set"
AUDIO_DIR: Path = TEST_SET / "audio"
VIDEO_DIR: Path = TEST_SET / "video"
LABEL_AUDIO_DIR: Path = TEST_SET / "label_audio"
LABEL_DIR: Path = REPO_ROOT / "data" / "diarization_test_set" / "label"

# --------------------------------------------------------------------------- #
# Pipeline 3 — ECAPA-TDNN (default embedding)
# --------------------------------------------------------------------------- #
ECAPA_ROOT: Path = MODELS_ROOT / "ecapa_tdnn"
ECAPA_MODEL: Path = ECAPA_ROOT / "exps" / "pretrain.model"

# --------------------------------------------------------------------------- #
# Pipeline 3 — pluggable embedding backends (wespeaker34/293, campplus, redimnet)
# --------------------------------------------------------------------------- #
EMBEDDINGS_WEIGHTS: Path = MODELS_ROOT / "embeddings"

# --------------------------------------------------------------------------- #
# Pipeline 3 — VBx ResNet101 x-vector + PLDA
# --------------------------------------------------------------------------- #
VBX_MODEL_DIR: Path = MODELS_ROOT / "vbx" / "ResNet101_16kHz"
VBX_ONNX: Path = VBX_MODEL_DIR / "nnet" / "final.onnx"
VBX_PLDA: Path = VBX_MODEL_DIR / "plda"
VBX_XFORM: Path = VBX_MODEL_DIR / "transform.h5"

# --------------------------------------------------------------------------- #
# Pipeline 2 — face detection (SCRFD)
# --------------------------------------------------------------------------- #
SCRFD_WEIGHTS: Path = MODELS_ROOT / "face_detection" / "SCRFD" / "weights"
SCRFD_DEFAULT: Path = SCRFD_WEIGHTS / "model_3_kps.onnx"

# --------------------------------------------------------------------------- #
# Pipeline 2 — face embedding (ArcFace / insightface)
# --------------------------------------------------------------------------- #
FACE_EMB_WEIGHTS: Path = MODELS_ROOT / "face_embedding" / "insightface"
ARCFACE_MODEL: Path = FACE_EMB_WEIGHTS / "glintr100.onnx"

# --------------------------------------------------------------------------- #
# Pipeline 2 — Active Speaker Detection (LR-ASD / LoCoNet)
# --------------------------------------------------------------------------- #
LRASD_WEIGHT_DIR: Path = MODELS_ROOT / "asd" / "LR-ASD" / "weight"
LRASD_MODEL: Path = LRASD_WEIGHT_DIR / "finetuning_TalkSet.model"
LRASD_PRETRAIN_AVA: Path = LRASD_WEIGHT_DIR / "pretrain_AVA.model"

LOCONET_PRETRAINED: Path = MODELS_ROOT / "asd" / "LoCoNet" / "pretrained_model"
LOCONET_MODEL: Path = LOCONET_PRETRAINED / "loconet_ava_best.model"
LOCONET_S3FD: Path = LOCONET_PRETRAINED / "s3fd.pth"


def assets_configured() -> bool:
    """True if the assets directory exists (cheap check used by selfcheck/CLIs)."""
    return ASSETS_ROOT.is_dir()


def describe() -> str:
    """Human-readable summary of where everything resolves (for diagnostics)."""
    return (
        f"REPO_ROOT   = {REPO_ROOT}\n"
        f"ASSETS_ROOT = {ASSETS_ROOT}  (set VIESPEAKER2_ASSETS to override)\n"
        f"MODELS_ROOT = {MODELS_ROOT}\n"
        f"DATA_ROOT   = {DATA_ROOT}  (set VIESPEAKER2_DATA to override)\n"
        f"LABEL_DIR   = {LABEL_DIR}  (in repo)\n"
    )
