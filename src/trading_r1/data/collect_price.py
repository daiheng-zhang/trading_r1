"""Daily price + technical indicator collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_UNIVERSE = [
    "BRK.B",
    "JPM",
    "LLY",
    "JNJ",
    "XOM",
    "CVX",
    "AAPL",
    "NVDA",
    "AMZN",
    "META",
    "MSFT",
    "TSLA",
    "QQQ",
    "SPY",
]


@dataclass
class PriceCollectConfig:
    output_dir: str
    start_date: str
    end_date: str
    interval: str = "1d"
    universe: list[str] | None = None
    source_cache_dir: str | None = None


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["SMA_50"] = close.rolling(50).mean()
    out["SMA_200"] = close.rolling(200).mean()
    out["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA_10"] = close.ewm(span=10, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    out["RSI_14"] = _rsi(close, 14)
    out["ROC_10"] = close.pct_change(10)
    out["ATR_14"] = _atr(out, 14)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["BB_MID"] = bb_mid
    out["BB_UPPER"] = bb_mid + 2 * bb_std
    out["BB_LOWER"] = bb_mid - 2 * bb_std

    out["Z_75"] = (close - close.rolling(75).mean()) / close.rolling(75).std()

    return out


def _load_cached(path: Path, start_date: str, end_date: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    if not mask.any():
        return None
    return df.loc[mask].copy()


def _download_yfinance(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for collect-price. Install `pip install yfinance`.") from exc

    df = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    return df


def collect_price_data(cfg: PriceCollectConfig) -> dict[str, Path]:
    universe = cfg.universe or DEFAULT_UNIVERSE
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for symbol in universe:
        normalized_symbol = symbol.replace("/", "_")
        out_path = out_dir / f"{normalized_symbol}.csv"

        df: pd.DataFrame | None = None
        if cfg.source_cache_dir:
            cached = Path(cfg.source_cache_dir) / f"{normalized_symbol}.csv"
            df = _load_cached(cached, cfg.start_date, cfg.end_date)

        if df is None:
            df = _download_yfinance(symbol, cfg.start_date, cfg.end_date, cfg.interval)

        if df.empty:
            continue

        needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            continue

        out = add_technicals(df[needed].copy())
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(out_path, index=False)
        saved[symbol] = out_path

    return saved


def collect_price_data_from_config(config: dict) -> dict[str, Path]:
    c = config.get("data", config)
    cfg = PriceCollectConfig(
        output_dir=c["price_output_dir"],
        start_date=c["start_date"],
        end_date=c["end_date"],
        interval=c.get("interval", "1d"),
        universe=list(c.get("universe", DEFAULT_UNIVERSE)),
        source_cache_dir=c.get("price_source_cache_dir"),
    )
    return collect_price_data(cfg)


def load_price_frame(price_dir: str | Path, ticker: str) -> pd.DataFrame:
    p = Path(price_dir) / f"{ticker.replace('/', '_')}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    return df


def available_tickers(price_dir: str | Path) -> Iterable[str]:
    for p in sorted(Path(price_dir).glob("*.csv")):
        yield p.stem
