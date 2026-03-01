"""Entry point for `python -m trading_r1` from repo root."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_r1.cli import main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
