"""LoCoNet model components.

The active LoCoNet encoder imports its concrete modules directly.  Older vendor
code re-exported a separate ``model.transformer`` package here, but that package
is not part of this LoCoNet implementation and none of the runtime code uses it.
Keeping those stale imports made every ``model.*`` import fail before the actual
encoder could be loaded.
"""
