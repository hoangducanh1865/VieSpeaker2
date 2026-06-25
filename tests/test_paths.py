import importlib


def test_default_assets_root_is_repo_sibling(monkeypatch):
    monkeypatch.delenv("VIESPEAKER2_ASSETS", raising=False)
    monkeypatch.delenv("VIESPEAKER2_DATA", raising=False)
    import viespeaker.paths as P
    importlib.reload(P)
    assert P.ASSETS_ROOT.name == "VieSpeaker2_assets"
    assert P.ASSETS_ROOT.parent == P.REPO_ROOT.parent
    assert P.MODELS_ROOT == P.ASSETS_ROOT / "models"
    # Labels stay inside the repo (version-controlled ground truth).
    assert str(P.REPO_ROOT) in str(P.LABEL_DIR)
    # Audio/video come from the external data dir.
    assert str(P.ASSETS_ROOT) in str(P.AUDIO_DIR)


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VIESPEAKER2_ASSETS", str(tmp_path / "assets"))
    monkeypatch.delenv("VIESPEAKER2_DATA", raising=False)
    import viespeaker.paths as P
    importlib.reload(P)
    assert P.ASSETS_ROOT == (tmp_path / "assets").resolve()
    assert P.ECAPA_MODEL == P.ASSETS_ROOT / "models" / "ecapa_tdnn" / "exps" / "pretrain.model"
    assert P.VBX_ONNX == P.ASSETS_ROOT / "models" / "vbx" / "ResNet101_16kHz" / "nnet" / "final.onnx"
    importlib.reload(P)  # leave a clean module for other tests
