"""Locate and load the `.env` file from a configurable location.

The `.env` (API keys) no longer has to live inside the repo. Search order:

  1. ``$VIESPEAKER2_ENV_FILE``                         (explicit override)
  2. ``<repo>/.env``                                   (in-repo, backward compat)
  3. ``<repo_parent>/env/<repo_name>/.env``            (sibling, recommended)

So on the server, putting the file at ``~/anhhd/sv/env/VieSpeaker2/.env`` is
picked up automatically (the repo at ``~/anhhd/sv/VieSpeaker2`` stays clean).
"""

from __future__ import annotations

import os
from pathlib import Path

from . import paths


def env_path() -> Path | None:
    """Return the first existing .env path in priority order, or None."""
    candidates = []
    override = os.environ.get("VIESPEAKER2_ENV_FILE")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(paths.REPO_ROOT / ".env")
    candidates.append(paths.REPO_ROOT.parent / "env" / paths.REPO_ROOT.name / ".env")
    for c in candidates:
        if c.is_file():
            return c
    return None


def load() -> Path | None:
    """Load KEY=VALUE lines from the resolved .env into os.environ (setdefault)."""
    p = env_path()
    if p is None:
        return None
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return p
