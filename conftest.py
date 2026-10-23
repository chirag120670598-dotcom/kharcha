"""Make the package importable when pytest runs from the repo root.

Without this, ``pytest tests/`` fails with ``ModuleNotFoundError`` unless the
caller remembers ``PYTHONPATH=.`` or uses ``python -m pytest``. Putting it in
``conftest.py`` means the test suite works however it is invoked.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
