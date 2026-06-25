"""Integration tests that need the heavy runtime stack. They skip cleanly when
deps (numpy/scipy/pyannote) are absent, so the lightweight CI still passes."""

import pytest


def test_fusion_single_input(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from viespeaker import bootstrap
    bootstrap.setup()
    try:
        from fuse import run_fusion
    except Exception as e:  # dover_lap or its deps unavailable
        pytest.skip(f"fuse import failed: {e}")

    hyp = tmp_path / "h.txt"
    hyp.write_text("0.000 1.000 A\n1.000 2.000 B\n")
    out = tmp_path / "fused.txt"
    run_fusion([str(hyp)], str(out), file_id="h")
    assert out.exists()
    assert len(out.read_text().strip().splitlines()) == 2


def test_eval_unique_tracks(tmp_path):
    pytest.importorskip("pyannote.core")
    from viespeaker import bootstrap
    bootstrap.setup()
    try:
        from evaluation import load_annotation_from_file
    except Exception as e:
        pytest.skip(f"evaluation import failed: {e}")

    ref = tmp_path / "ref.txt"
    # Two identical [start,end] spans with different speakers must both survive
    # (previously the second overwrote the first via the default track).
    ref.write_text("0.0 1.0 A\n0.0 1.0 B\n")
    ann = load_annotation_from_file(str(ref))
    assert len(list(ann.itertracks())) == 2


def test_redimnet_uses_upstream_torch_hub_api(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")

    from viespeaker import bootstrap

    bootstrap.setup()
    from embeddings.backends.redimnet import ReDimNetEmbedder

    calls = []

    class FakeModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    def fake_load(repo, entrypoint, **kwargs):
        calls.append((repo, entrypoint, kwargs))
        return FakeModel()

    monkeypatch.setattr(torch.hub, "load", fake_load)

    embedder = ReDimNetEmbedder(device="cpu")

    assert embedder.dim == 192
    assert calls == [
        (
            "IDRnD/ReDimNet",
            "ReDimNet",
            {
                "model_name": "b6",
                "train_type": "ptn",
                "dataset": "vox2",
                "verbose": False,
                "trust_repo": True,
            },
        )
    ]
