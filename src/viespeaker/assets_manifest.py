"""Declarative manifest of every external asset (weights + test media).

Each entry records:
  * ``dest``       — absolute target path under the assets dir (from :mod:`viespeaker.paths`)
  * ``old``        — path RELATIVE TO THE REPO ROOT where the file used to live
                     (committed in git or downloaded from Drive). Used to generate
                     the one-time "copy into assets dir" commands (server CP1).
  * ``severity``   — "core" (selfcheck FAILS if missing) or "soft" (WARN only;
                     best-effort / optional model).
  * ``note``       — short description.

This is the single source of truth consumed by ``scripts/selfcheck.py`` and by
``scripts/migrate_assets.py`` (which prints the exact ``cp`` commands).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Asset:
    dest: Path
    old: str
    severity: str  # "core" | "soft"
    note: str


_AV = "src/pipeline/audio_visual_pipeline"
_CLEAN = "src/pipeline/clean_pipeline"

# --------------------------------------------------------------------------- #
# Model weights
# --------------------------------------------------------------------------- #
WEIGHTS: list[Asset] = [
    # ---- Pipeline 3: ECAPA ----
    Asset(paths.ECAPA_MODEL,
          f"{_CLEAN}/models/ecapa_tdnn/exps/pretrain.model", "core", "ECAPA-TDNN (P3 default embedding)"),
    # ---- Pipeline 3: VBx ----
    Asset(paths.VBX_ONNX,
          f"{_CLEAN}/vbx/models/ResNet101_16kHz/nnet/final.onnx", "core", "VBx ResNet101 x-vector ONNX"),
    Asset(paths.VBX_PLDA,
          f"{_CLEAN}/vbx/models/ResNet101_16kHz/plda", "core", "VBx PLDA"),
    Asset(paths.VBX_XFORM,
          f"{_CLEAN}/vbx/models/ResNet101_16kHz/transform.h5", "core", "VBx LDA transform"),
    # ---- Pipeline 3: embedding backends ----
    Asset(paths.EMBEDDINGS_WEIGHTS / "wespeaker34" / "pytorch_model.bin",
          f"{_CLEAN}/embeddings/weights/wespeaker34/pytorch_model.bin", "soft", "WeSpeaker ResNet34"),
    Asset(paths.EMBEDDINGS_WEIGHTS / "wespeaker34" / "config.yaml",
          f"{_CLEAN}/embeddings/weights/wespeaker34/config.yaml", "soft", "WeSpeaker34 config"),
    Asset(paths.EMBEDDINGS_WEIGHTS / "wespeaker293" / "speaker-embedding.onnx",
          f"{_CLEAN}/embeddings/weights/wespeaker293/speaker-embedding.onnx", "soft", "WeSpeaker ResNet293 (>100MB)"),
    Asset(paths.EMBEDDINGS_WEIGHTS / "wespeaker293" / "config.yaml",
          f"{_CLEAN}/embeddings/weights/wespeaker293/config.yaml", "soft", "WeSpeaker293 config"),
    Asset(paths.EMBEDDINGS_WEIGHTS / "campplus" / "campplus_cn_common.bin",
          f"{_CLEAN}/embeddings/weights/campplus/campplus_cn_common.bin", "soft", "CAM++ (zh-cn)"),
    Asset(paths.EMBEDDINGS_WEIGHTS / "campplus" / "config.yaml",
          f"{_CLEAN}/embeddings/weights/campplus/config.yaml", "soft", "CAM++ config"),
    # redimnet: loaded via torch.hub; a local model_120.pt is optional
    Asset(paths.EMBEDDINGS_WEIGHTS / "redimnet" / "config.yaml",
          f"{_CLEAN}/embeddings/weights/redimnet/config.yaml", "soft", "ReDimNet config (model via torch.hub)"),
    # ---- Pipeline 2: SCRFD face detection (all variants) ----
    *[Asset(paths.SCRFD_WEIGHTS / name,
            f"{_AV}/face_detection_model/SCRFD/weights/{name}",
            "core" if name == "model_3_kps.onnx" else "soft", f"SCRFD {name}")
      for name in ("model_1.onnx", "model_1_kps.onnx", "model_2.onnx", "model_3.onnx",
                   "model_3_kps.onnx", "model_4.onnx", "model_4_kps.onnx")],
    # ---- Pipeline 2: ArcFace + insightface utility models ----
    Asset(paths.ARCFACE_MODEL,
          f"{_AV}/face_embedding_model/weights/glintr100.onnx", "soft", "ArcFace glintr100 (>100MB)"),
    *[Asset(paths.FACE_EMB_WEIGHTS / name,
            f"{_AV}/face_embedding_model/weights/{name}", "soft", f"insightface {name}")
      for name in ("1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx")],
    # ---- Pipeline 2: ASD weights ----
    Asset(paths.LRASD_MODEL,
          f"{_AV}/audio_visual_model/LR-ASD/weight/finetuning_TalkSet.model", "soft", "LR-ASD (TalkSet)"),
    Asset(paths.LRASD_PRETRAIN_AVA,
          f"{_AV}/audio_visual_model/LR-ASD/weight/pretrain_AVA.model", "soft", "LR-ASD (AVA pretrain)"),
    Asset(paths.LOCONET_MODEL,
          f"{_AV}/audio_visual_model/LoCoNet_ASD/pretrained_model/loconet_ava_best.model", "soft", "LoCoNet (>100MB)"),
    # NOTE: LoCoNet's tiny s3fd.pth (256KB) and its Caffe S3FD files stay in-repo
    # (not large, hardcoded inside the vendored LoCoNet code).
]

# --------------------------------------------------------------------------- #
# Test data (audio/video/label_audio). Labels (.txt) stay in the repo.
# --------------------------------------------------------------------------- #
SAMPLES = ["drama", "interview_clean", "interview_noise", "movie", "sample_0", "singing"]

DATA: list[Asset] = []
for _s in SAMPLES:
    DATA.append(Asset(paths.AUDIO_DIR / f"{_s}.wav",
                      f"data/diarization_test_set/audio/{_s}.wav", "core", f"audio {_s}"))
    DATA.append(Asset(paths.VIDEO_DIR / f"{_s}.mp4",
                      f"data/diarization_test_set/video/{_s}.mp4", "soft", f"video {_s} (P2 only)"))
for _s in [s for s in SAMPLES if s != "sample_0"]:
    DATA.append(Asset(paths.LABEL_AUDIO_DIR / f"{_s}.wav",
                      f"data/diarization_test_set/label_audio/{_s}.wav", "soft", f"label_audio {_s}"))

ALL: list[Asset] = WEIGHTS + DATA


def missing(severity: str | None = None) -> list[Asset]:
    """Return assets whose destination does not exist, optionally filtered by severity."""
    return [a for a in ALL if (severity is None or a.severity == severity) and not a.dest.exists()]
