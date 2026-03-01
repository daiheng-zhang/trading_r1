"""Finance evaluation metrics: CR, SR, HR, MDD."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def cumulative_return(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    eq = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    return float(eq[-1] - 1.0)


def sharpe_ratio_annualized(
    returns: Sequence[float], rf_annual: float = 0.04, periods_per_year: int = 252
) -> float:
    if not returns:
        return 0.0
    r = np.asarray(returns, dtype=float)
    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    ex = r - rf_period
    std = ex.std(ddof=1) if len(ex) > 1 else 0.0
    if std == 0:
        return 0.0
    return float((ex.mean() / std) * math.sqrt(periods_per_year))


def hit_rate(signals: Sequence[float], realized_returns: Sequence[float]) -> float:
    if not signals or not realized_returns:
        return 0.0
    n = min(len(signals), len(realized_returns))
    hits = 0
    total = 0
    for i in range(n):
        s = signals[i]
        r = realized_returns[i]
        if s == 0:
            continue
        total += 1
        if (s > 0 and r > 0) or (s < 0 and r < 0):
            hits += 1
    if total == 0:
        return 0.0
    return float(hits / total)


def max_drawdown(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    eq = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(eq)
    drawdowns = 1.0 - (eq / peaks)
    return float(drawdowns.max(initial=0.0))


def evaluate_all(
    returns: Sequence[float],
    signals: Sequence[float],
    realized_returns: Sequence[float],
    rf_annual: float = 0.04,
    periods_per_year: int = 252,
) -> dict[str, float]:
    return {
        "CR": cumulative_return(returns),
        "SR": sharpe_ratio_annualized(returns, rf_annual=rf_annual, periods_per_year=periods_per_year),
        "HR": hit_rate(signals, realized_returns),
        "MDD": max_drawdown(returns),
    }
