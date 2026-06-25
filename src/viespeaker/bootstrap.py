"""Single, centralized place that puts the vendored sub-trees on ``sys.path``.

The pipelines vendor third-party code (LR-ASD, LoCoNet, SCRFD, VBx, the ECAPA
model, dover_lap, …) that uses *bare* top-level imports (``from ASD import ASD``,
``from features import ...``, ``from clean import ...``). Rewriting those
third-party import statements is out of scope, so instead of every module doing
its own scattered ``sys.path.insert(...)``, they all call :func:`setup` once.

This replaces ~15 duplicated ``sys.path.insert`` blocks with one idempotent call.
"""

from __future__ import annotations

import sys

from . import paths

_DONE = False

# Directories needed for the *first-party* bare imports (`from clean import`,
# `from face_pipeline import`, `import sort`, `from evaluation import`, …).
# These have unique top-level module names, so adding them globally is safe.
#
# Collision-prone DEEP vendored trees (ecapa_tdnn/model.py, vbx/features.py,
# LR-ASD/ASD.py — all expose generic names like `model`, `features`, `utils`)
# are intentionally NOT added here; their consumers insert them just-in-time at
# sys.path[0] right before importing, to guarantee precedence.
_VENDOR_DIRS = [
    paths.SRC_ROOT,
    paths.SRC_ROOT / "evaluation",
    paths.SRC_ROOT / "pipeline" / "clean_pipeline",
    paths.SRC_ROOT / "pipeline" / "fusion_pipeline",
    paths.SRC_ROOT / "pipeline" / "audio_visual_pipeline",
    paths.SRC_ROOT / "pipeline" / "audio_visual_pipeline" / "sort",
]


def setup() -> None:
    """Idempotently prepend the vendored directories to ``sys.path``."""
    global _DONE
    if _DONE:
        return
    for d in _VENDOR_DIRS:
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    _DONE = True
