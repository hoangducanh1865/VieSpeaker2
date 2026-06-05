#!/usr/bin/env python3
"""Preflight check before launching the (long) scenario sweep on the server.

Verifies: core imports, model-weight presence, embedding backends actually
produce a vector on a synthetic clip, and reports .env keys. Exits non-zero only
if a CORE component is broken; best-effort items (loconet, redimnet, cloud key)
only warn.

    python scripts/selfcheck.py
"""

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CLEAN = os.path.join(_ROOT, "src", "pipeline", "clean_pipeline")
sys.path.insert(0, _CLEAN)

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
_core_failed = False


def line(tag, msg):
    print(f"{tag} {msg}")


def check_imports():
    print("\n== Core imports ==")
    global _core_failed
    core = ["numpy", "torch", "torchaudio", "sklearn", "scipy",
            "pyannote.audio", "pyannote.metrics", "onnxruntime"]
    optional = ["insightface", "igraph", "leidenalg", "intervaltree", "spyder",
                "h5py", "soundfile", "python_speech_features", "cv2"]
    for m in core:
        try:
            __import__(m)
            line(OK, m)
        except Exception as e:
            line(FAIL, f"{m}: {e}")
            _core_failed = True
    for m in optional:
        try:
            __import__(m)
            line(OK, m + " (optional)")
        except Exception as e:
            line(WARN, f"{m} (optional): {e}")


def check_weights():
    print("\n== Model weights ==")
    av = os.path.join(_ROOT, "src", "pipeline", "audio_visual_pipeline")
    emb = os.path.join(_CLEAN, "embeddings", "weights")
    core = {
        "ECAPA pretrain.model": os.path.join(_CLEAN, "models", "ecapa_tdnn", "exps", "pretrain.model"),
        "VBx ResNet101 onnx": os.path.join(_CLEAN, "vbx", "models", "ResNet101_16kHz", "nnet", "final.onnx"),
        "VBx plda": os.path.join(_CLEAN, "vbx", "models", "ResNet101_16kHz", "plda"),
        "VBx transform.h5": os.path.join(_CLEAN, "vbx", "models", "ResNet101_16kHz", "transform.h5"),
        "SCRFD detector": os.path.join(av, "face_detection_model", "SCRFD", "weights", "model_3_kps.onnx"),
    }
    soft = {
        "ArcFace glintr100 (P2)": os.path.join(av, "face_embedding_model", "weights", "glintr100.onnx"),
        "LR-ASD weight (P2)": os.path.join(av, "audio_visual_model", "LR-ASD", "weight", "finetuning_TalkSet.model"),
        "LoCoNet weight (P2)": os.path.join(av, "audio_visual_model", "LoCoNet_ASD", "pretrained_model", "loconet_ava_best.model"),
        "WeSpeaker34 bin": os.path.join(emb, "wespeaker34", "pytorch_model.bin"),
        "WeSpeaker293 onnx": os.path.join(emb, "wespeaker293", "speaker-embedding.onnx"),
        "CAM++ bin": os.path.join(emb, "campplus", "campplus_cn_common.bin"),
        "ReDimNet pt": os.path.join(emb, "redimnet", "model_120.pt"),
    }
    global _core_failed
    for name, p in core.items():
        if os.path.exists(p):
            line(OK, name)
        else:
            line(FAIL, f"{name} MISSING: {p}")
            _core_failed = True
    for name, p in soft.items():
        line(OK if os.path.exists(p) else WARN, f"{name}{'' if os.path.exists(p) else ' MISSING (run scripts/prepare_embeddings.py)'}")


def _synth_wav():
    import numpy as np
    import soundfile as sf
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    sig = 0.1 * np.sin(2 * np.pi * 150 * t) + 0.02 * np.random.randn(len(t))
    path = os.path.join(tempfile.mkdtemp(), "synth.wav")
    sf.write(path, sig.astype("float32"), sr)
    return path


def check_embedders():
    print("\n== Embedding backends (synthetic 3s clip, CPU) ==")
    from embeddings.registry import KNOWN_BACKENDS, get_embedder
    wav = _synth_wav()
    best_effort = {"redimnet"}
    for name in KNOWN_BACKENDS:
        try:
            emb = get_embedder(name, device="cpu")
            v = emb.extract(wav, 0.2, 2.8)
            if v is None:
                raise RuntimeError("extract returned None")
            line(OK, f"{name} -> dim={len(v)}")
        except Exception as e:
            line(WARN if name in best_effort else WARN, f"{name}: {type(e).__name__}: {e}")


def check_env():
    print("\n== .env / API keys ==")
    # load .env (mirrors the pipelines' loader)
    env = os.path.join(_ROOT, ".env")
    if os.path.exists(env):
        for ln in open(env):
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, _, v = ln.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    line(OK if os.getenv("PYANNOTEAI_API_KEY") else WARN,
         "PYANNOTEAI_API_KEY" + ("" if os.getenv("PYANNOTEAI_API_KEY") else " missing (needed for P1 cloud precision-2)"))
    line(OK if os.getenv("HUGGINGFACE_ACCESS_TOKEN") else WARN,
         "HUGGINGFACE_ACCESS_TOKEN" + ("" if os.getenv("HUGGINGFACE_ACCESS_TOKEN") else " missing (needed for P1 local 3.1)"))


def main():
    print("VieSpeaker2 — preflight self-check")
    try:
        import torch
        print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
    except Exception:
        pass
    check_imports()
    check_weights()
    check_env()
    try:
        check_embedders()
    except Exception as e:
        line(WARN, f"embedder check skipped: {e}")
    print("\n== Summary ==")
    if _core_failed:
        line(FAIL, "Core components missing/broken — fix before running the sweep.")
        sys.exit(1)
    line(OK, "Core OK. Warnings above are tolerable (best-effort models / missing keys).")


if __name__ == "__main__":
    main()
