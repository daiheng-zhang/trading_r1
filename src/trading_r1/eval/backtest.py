"""Backtest runtime with CR/SR/HR/MDD and leakage controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_r1.actions import ACTION_TO_POSITION
from trading_r1.eval.inference import infer_action_for_ticker_date
from trading_r1.eval.metrics import evaluate_all
from trading_r1.utils.io import read_jsonl, write_jsonl


@dataclass
class BacktestConfig:
    infer_config_path: str
    samples_path: str
    price_dir: str
    output_dir: str
    start_date: str
    end_date: str
    transaction_cost_bps: float = 5.0
    rf_annual: float = 0.04
    periods_per_year: int = 252


def _load_prices(price_dir: str, ticker: str) -> pd.DataFrame:
    path = Path(price_dir) / f"{ticker.replace('/', '_')}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df.sort_values("Date").reset_index(drop=True)


def _build_next_open_returns(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or "Open" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for i in range(len(df) - 1):
        d = df.iloc[i]["Date"].isoformat()
        o0 = float(df.iloc[i]["Open"])
        o1 = float(df.iloc[i + 1]["Open"])
        if o0 == 0:
            continue
        out[d] = (o1 / o0) - 1.0
    return out


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _leakage_violation(sample: dict[str, Any]) -> bool:
    trade_date = str(sample.get("trade_date", ""))
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    features_end = str(meta.get("features_window_end", ""))
    if not trade_date or not features_end:
        return False
    return features_end > trade_date


def run_backtest(config: dict[str, Any], infer_config: dict[str, Any]) -> dict[str, Any]:
    c = config.get("backtest", config)
    cfg = BacktestConfig(
        infer_config_path=c.get("infer_config_path", "configs/infer.yaml"),
        samples_path=c["samples_path"],
        price_dir=c["price_dir"],
        output_dir=c["output_dir"],
        start_date=c["start_date"],
        end_date=c["end_date"],
        transaction_cost_bps=float(c.get("transaction_cost_bps", 5.0)),
        rf_annual=float(c.get("rf_annual", 0.04)),
        periods_per_year=int(c.get("periods_per_year", 252)),
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(cfg.samples_path)
    filtered = [
        s
        for s in samples
        if cfg.start_date <= str(s.get("trade_date", "")) <= cfg.end_date
    ]

    leaks = [s.get("sample_id") for s in filtered if _leakage_violation(s)]
    if leaks:
        raise RuntimeError(f"Leakage guard failed for samples: {leaks[:10]}")

    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for s in filtered:
        by_ticker.setdefault(str(s.get("ticker")), []).append(s)

    portfolio_rows: list[dict[str, Any]] = []
    for ticker, ticker_samples in by_ticker.items():
        price_df = _load_prices(cfg.price_dir, ticker)
        next_open_ret = _build_next_open_returns(price_df)
        prev_pos = 0.0

        for s in sorted(ticker_samples, key=lambda x: x["trade_date"]):
            date = str(s["trade_date"])
            if date not in next_open_ret:
                continue

            pred = infer_action_for_ticker_date(infer_config, ticker=ticker, date=date)
            action = pred["action"]
            pos = ACTION_TO_POSITION.get(action, 0.0)

            gross_ret = pos * float(next_open_ret[date])
            turnover = abs(pos - prev_pos)
            tc = turnover * (cfg.transaction_cost_bps / 10000.0)
            net_ret = gross_ret - tc

            portfolio_rows.append(
                {
                    "ticker": ticker,
                    "trade_date": date,
                    "action": action,
                    "position": pos,
                    "gross_return": gross_ret,
                    "transaction_cost": tc,
                    "net_return": net_ret,
                    "realized_return": float(next_open_ret[date]),
                }
            )
            prev_pos = pos

    if not portfolio_rows:
        raise RuntimeError("No backtest rows generated; check samples and price overlap.")

    df = pd.DataFrame(portfolio_rows)
    daily = (
        df.groupby("trade_date")
        .agg(
            net_return=("net_return", "mean"),
            signal=("position", "mean"),
            realized_return=("realized_return", "mean"),
        )
        .reset_index()
        .sort_values("trade_date")
    )

    metrics = evaluate_all(
        returns=daily["net_return"].tolist(),
        signals=daily["signal"].tolist(),
        realized_returns=daily["realized_return"].tolist(),
        rf_annual=cfg.rf_annual,
        periods_per_year=cfg.periods_per_year,
    )

    result = {
        "config_hash": _config_hash(config),
        "rows": len(df),
        "days": len(daily),
        "metrics": metrics,
    }

    df.to_csv(Path(cfg.output_dir) / "trades.csv", index=False)
    daily.to_csv(Path(cfg.output_dir) / "daily_portfolio.csv", index=False)
    write_jsonl(Path(cfg.output_dir) / "summary.jsonl", [result])
    return result
