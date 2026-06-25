"""VieSpeaker2 — first-party support package.

Holds the cross-cutting infrastructure that the diarization pipelines share:

* :mod:`viespeaker.paths`     — single source of truth for every weight / data path
* :mod:`viespeaker.bootstrap` — one place that puts the vendored sub-trees on sys.path
* :mod:`viespeaker.logging_setup` — `get_logger()` used across first-party modules
* :mod:`viespeaker.assets_manifest` — declarative list of external assets (for selfcheck)

The heavy pipeline code still lives under ``src/pipeline`` and ``src/evaluation``;
this package only provides glue so those modules stop hard-coding paths and
sys.path hacks.
"""

__version__ = "1.0.0"

from . import paths  # noqa: F401  (re-export for `from viespeaker import paths`)

__all__ = ["paths", "__version__"]
