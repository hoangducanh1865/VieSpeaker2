import os
import sys

# Make `viespeaker` and the vendored first-party modules importable without an
# editable install (CI runs the pure tests with a minimal dependency set).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
