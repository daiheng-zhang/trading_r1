"""Paper-faithful volatility-adjusted label generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trading_r1.actions import ACTIONS
from trading_r1.data.collect_price import available_tickers, load_price_frame
from trading_r1.utils.io import write_jsonl


@dataclass
class LabelConfig:
    price_dir: str
    output_path: str
    horizons: tuple[int, int, int] = (3, 7, 15)
    weights: tuple[float, float, float] = (0.3, 0.5, 0.2)
    quantiles: tuple[float, float, float, float] = (0.03, 0.15, 0.53, 0.85)
    ema_span: int = 3
    vol_window: int = 20


def compute_weighted_signal(
    close: pd.Series,
    horizons: tuple[int, int, int] = (3, 7, 15),
    weights: tuple[float, float, float] = (0.3, 0.5, 0.2),
    ema_span: int = 3,
    vol_window: int = 20,
) -> pd.Series:
    ema = close.ewm(span=ema_span, adjust=False).mean()
    comps: list[pd.Series] = []
    for h, w in zip(horizons, weights):
        ret = (ema - ema.shift(h)) / ema.shift(h)
        vol = ret.rolling(vol_window).std()
        sig = ret / vol
        comps.append(w * sig)
    weighted = sum(comps)
    return weighted


def _label_from_thresholds(value: float, thresholds: tuple[float, float, float, float]) -> str:
    q1, q2, q3, q4 = thresholds
    if value >= q4:
        return "STRONG_BUY"
    if value >= q3:
        return "BUY"
    if value >= q2:
        return "HOLD"
    if value >= q1:
        return "SELL"
    return "STRONG_SELL"


def label_series(signal: pd.Series, quantiles: tuple[float, float, float, float]) -> tuple[pd.Series, tuple[float, float, float, float]]:
    valid = signal.dropna()
    if valid.empty:
        return pd.Series(index=signal.index, dtype="object"), (np.nan, np.nan, np.nan, np.nan)

    thresholds = tuple(float(valid.quantile(q)) for q in quantiles)
    labels = signal.apply(lambda x: _label_from_thresholds(float(x), thresholds) if pd.notna(x) else None)
    return labels, thresholds


def generate_labels_for_frame(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("Date").reset_index(drop=True)

    if "Close" not in out.columns:
        raise ValueError("Price frame requires Close column")

    weighted = compute_weighted_signal(
        out["Close"].astype(float),
        horizons=cfg.horizons,
        weights=cfg.weights,
        ema_span=cfg.ema_span,
        vol_window=cfg.vol_window,
    )
    labels, thresholds = label_series(weighted, cfg.quantiles)

    out["weighted_signal"] = weighted
    out["label_action"] = labels
    out["q03"] = thresholds[0]
    out["q15"] = thresholds[1]
    out["q53"] = thresholds[2]
    out["q85"] = thresholds[3]
    return out


def make_labels(cfg: LabelConfig, tickers: Iterable[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    symbols = list(tickers) if tickers is not None else list(available_tickers(cfg.price_dir))
    for symbol in symbols:
        df = load_price_frame(cfg.price_dir, symbol)
        if df.empty:
            continue
        labeled = generate_labels_for_frame(df, cfg)
        for _, rec in labeled.iterrows():
            action = rec.get("label_action")
            if action not in ACTIONS:
                continue
            rows.append(
                {
                    "ticker": symbol,
                    "trade_date": pd.to_datetime(rec["Date"]).date().isoformat(),
                    "label_action": action,
                    "weighted_signal": float(rec["weighted_signal"]),
                    "quantiles": {
                        "q03": float(rec["q03"]),
                        "q15": float(rec["q15"]),
                        "q53": float(rec["q53"]),
                        "q85": float(rec["q85"]),
                    },
                }
            )
    write_jsonl(cfg.output_path, rows)
    return rows


def make_labels_from_config(config: dict) -> list[dict]:
    c = config.get("labels", config)
    cfg = LabelConfig(
        price_dir=c["price_dir"],
        output_path=c["output_path"],
        horizons=tuple(c.get("horizons", [3, 7, 15])),
        weights=tuple(c.get("weights", [0.3, 0.5, 0.2])),
        quantiles=tuple(c.get("quantiles", [0.03, 0.15, 0.53, 0.85])),
        ema_span=int(c.get("ema_span", 3)),
        vol_window=int(c.get("vol_window", 20)),
    )
    return make_labels(cfg)
