#!/usr/bin/env python3
"""Preflight check before launching the (long) scenario sweep on the server.

Verifies: core imports, that the external assets dir is populated (weights +
test data), embedding backends actually produce a vector, and reports .env keys.
Exits non-zero only if a CORE component is broken; best-effort / soft items only
warn and print exactly where the missing file should go.

    python scripts/selfcheck.py
"""

import os
import sys
import tempfile

# Make `viespeaker` importable from a fresh checkout, then centralize sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_d = _HERE
while _d != os.path.dirname(_d):
    if os.path.isdir(os.path.join(_d, "src", "viespeaker")):
        sys.path.insert(0, os.path.join(_d, "src"))
        break
    _d = os.path.dirname(_d)
from viespeaker import bootstrap, paths  # noqa: E402
from viespeaker import assets_manifest as M  # noqa: E402

bootstrap.setup()

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


def check_assets():
    """Verify the external assets (weights + test media) via the manifest."""
    print("\n== Assets (weights + test data) ==")
    print(paths.describe())
    global _core_failed
    if not paths.ASSETS_ROOT.is_dir():
        line(FAIL, f"Assets dir does not exist: {paths.ASSETS_ROOT}")
        line("", "      Populate it: run `python scripts/migrate_assets.py` and follow the commands,")
        line("", "      or set VIESPEAKER2_ASSETS to point at your assets dir.")
        _core_failed = True
        return
    for a in M.ALL:
        if a.dest.exists():
            line(OK, f"{a.note}")
        elif a.severity == "core":
            line(FAIL, f"{a.note} MISSING -> {a.dest}")
            _core_failed = True
        else:
            line(WARN, f"{a.note} missing (best-effort) -> {a.dest}")


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
    for name in KNOWN_BACKENDS:
        try:
            emb = get_embedder(name, device="cpu")
            v = emb.extract(wav, 0.2, 2.8)
            if v is None:
                raise RuntimeError("extract returned None")
            line(OK, f"{name} -> dim={len(v)}")
        except Exception as e:
            line(WARN, f"{name}: {type(e).__name__}: {e}")


def check_env():
    print("\n== .env / API keys ==")
    env = os.path.join(paths.REPO_ROOT, ".env")
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
    check_assets()
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
