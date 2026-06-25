"""Tiny logging helper shared by first-party modules.

Replaces ad-hoc ``print()`` calls. Verbosity is controlled by the
``VIESPEAKER_LOG_LEVEL`` environment variable (DEBUG/INFO/WARNING/...), default
INFO. Logs go to stderr so they don't pollute stdout pipelines.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("VIESPEAKER_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)
