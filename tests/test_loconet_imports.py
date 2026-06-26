import importlib
import sys
from pathlib import Path


def test_loconet_model_package_does_not_require_unused_transformer():
    root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pipeline"
        / "audio_visual_pipeline"
        / "audio_visual_model"
        / "LoCoNet_ASD"
    )
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("model")
        assert module.__doc__
    finally:
        sys.path.remove(str(root))
        sys.modules.pop("model", None)
