"""Package bridge so `python -m trading_r1` can load `src/trading_r1/*`."""

from __future__ import annotations

import sys
from pathlib import Path
from pkgutil import extend_path


SRC_PARENT = Path(__file__).resolve().parent / "src"
if str(SRC_PARENT) not in sys.path:
    sys.path.insert(0, str(SRC_PARENT))

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

