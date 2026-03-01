"""Stage III asymmetric decision reward matrix."""

from __future__ import annotations

from trading_r1.actions import ACTIONS
from trading_r1.parsing.decision_parser import normalize_action


# Rows are predictions, columns are ground truth.
MATRIX: dict[str, dict[str, float]] = {
    "STRONG_SELL": {
        "STRONG_SELL": 1.00,
        "SELL": 0.75,
        "HOLD": -1.25,
        "BUY": -2.00,
        "STRONG_BUY": -2.25,
    },
    "SELL": {
        "STRONG_SELL": 0.75,
        "SELL": 1.00,
        "HOLD": -0.75,
        "BUY": -1.50,
        "STRONG_BUY": -2.00,
    },
    "HOLD": {
        "STRONG_SELL": -1.50,
        "SELL": -1.00,
        "HOLD": 1.00,
        "BUY": -1.00,
        "STRONG_BUY": -1.50,
    },
    "BUY": {
        "STRONG_SELL": -1.75,
        "SELL": -1.25,
        "HOLD": -0.75,
        "BUY": 1.00,
        "STRONG_BUY": 0.75,
    },
    "STRONG_BUY": {
        "STRONG_SELL": -2.00,
        "SELL": -1.50,
        "HOLD": -1.25,
        "BUY": 0.75,
        "STRONG_BUY": 1.00,
    },
}


def decision_reward(pred_action: str | None, true_action: str | None, scale: float = 1.0) -> float:
    pred = normalize_action(pred_action)
    truth = normalize_action(true_action)
    if pred is None or truth is None:
        return -1.5 * scale
    if pred not in ACTIONS or truth not in ACTIONS:
        return -1.5 * scale
    return MATRIX[pred][truth] * scale
