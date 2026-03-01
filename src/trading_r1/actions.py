"""Action constants and mappings."""

from __future__ import annotations

ACTIONS = ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]

ACTION_TO_POSITION = {
    "STRONG_SELL": -1.0,
    "SELL": -0.5,
    "HOLD": 0.0,
    "BUY": 0.5,
    "STRONG_BUY": 1.0,
}

CANONICAL_ACTION_MAP = {
    "STRONG SELL": "STRONG_SELL",
    "STRONG_SELL": "STRONG_SELL",
    "STRONG-SELL": "STRONG_SELL",
    "SELL": "SELL",
    "HOLD": "HOLD",
    "BUY": "BUY",
    "STRONG BUY": "STRONG_BUY",
    "STRONG_BUY": "STRONG_BUY",
    "STRONG-BUY": "STRONG_BUY",
}
