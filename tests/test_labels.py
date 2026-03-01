from __future__ import annotations

import numpy as np
import pandas as pd

from trading_r1.labels.volatility_labels import _label_from_thresholds, compute_weighted_signal, label_series


def test_weighted_signal_shape() -> None:
    close = pd.Series(np.linspace(100, 140, 120))
    sig = compute_weighted_signal(close)
    assert len(sig) == 120


def test_quantile_boundary_mapping() -> None:
    thresholds = (-1.0, -0.5, 0.2, 0.8)
    assert _label_from_thresholds(0.8, thresholds) == "STRONG_BUY"
    assert _label_from_thresholds(0.2, thresholds) == "BUY"
    assert _label_from_thresholds(-0.5, thresholds) == "HOLD"
    assert _label_from_thresholds(-1.0, thresholds) == "SELL"
    assert _label_from_thresholds(-1.1, thresholds) == "STRONG_SELL"


def test_label_series_non_empty() -> None:
    s = pd.Series(np.random.normal(size=200))
    labels, thresholds = label_series(s, (0.03, 0.15, 0.53, 0.85))
    assert labels.notna().sum() > 0
    assert len(thresholds) == 4
