"""Compatibility helpers for pyannote.audio 3.x with newer huggingface_hub.

pyannote.audio 3.x calls ``hf_hub_download(..., use_auth_token=...)`` while
huggingface_hub 1.x only accepts ``token=...``. Keep the translation scoped to
the pyannote loading call instead of globally monkeypatching Hugging Face Hub.
"""

import contextlib
import functools
import importlib
import inspect


def pretrained_auth_kwargs(loader, token):
    """Return the auth keyword supported by a pyannote pretrained loader."""
    if token is None:
        return {}
    parameters = inspect.signature(loader).parameters
    if "token" in parameters:
        return {"token": token}
    return {"use_auth_token": token}


def translate_legacy_hf_auth(download):
    """Wrap ``hf_hub_download`` and translate its removed auth keyword."""

    @functools.wraps(download)
    def wrapped(*args, **kwargs):
        legacy_token = kwargs.pop("use_auth_token", None)
        if legacy_token is not None and "token" not in kwargs:
            kwargs["token"] = legacy_token
        return download(*args, **kwargs)

    return wrapped


@contextlib.contextmanager
def pyannote_hf_hub_compat():
    """Patch pyannote's imported Hub functions only for the loading operation."""
    patched = []
    for module_name in (
        "pyannote.audio.core.pipeline",
        "pyannote.audio.core.model",
        "pyannote.audio.pipelines.speaker_verification",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        download = getattr(module, "hf_hub_download", None)
        if download is None:
            continue
        patched.append((module, download))
        module.hf_hub_download = translate_legacy_hf_auth(download)

    try:
        yield
    finally:
        for module, download in patched:
            module.hf_hub_download = download
