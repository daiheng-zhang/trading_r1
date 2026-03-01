"""Fundamentals collection with point-in-time report dates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_r1.data.collect_price import DEFAULT_UNIVERSE
from trading_r1.utils.io import write_jsonl


@dataclass
class FundamentalsCollectConfig:
    output_dir: str
    universe: list[str] | None = None
    simfin_csv_dir: str | None = None


def _from_yfinance(ticker: str) -> list[dict[str, Any]]:
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for fundamentals fallback.") from exc

    tk = yf.Ticker(ticker)
    rows: list[dict[str, Any]] = []

    datasets = {
        "quarterly_balance_sheet": tk.quarterly_balance_sheet,
        "quarterly_financials": tk.quarterly_financials,
        "quarterly_cashflow": tk.quarterly_cashflow,
    }

    merged_by_date: dict[str, dict[str, Any]] = {}
    for name, frame in datasets.items():
        if frame is None or frame.empty:
            continue
        df = frame.T.reset_index().rename(columns={"index": "report_date"})
        for _, rec in df.iterrows():
            d = pd.to_datetime(rec["report_date"]).date().isoformat()
            entry = merged_by_date.setdefault(d, {"report_date": d, "ticker": ticker, "source": "yfinance"})
            payload = {k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float)) else v) for k, v in rec.items() if k != "report_date"}
            entry[name] = payload

    rows.extend(sorted(merged_by_date.values(), key=lambda x: x["report_date"]))
    return rows


def _load_simfin_rows(simfin_csv_dir: Path, ticker: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not simfin_csv_dir.exists():
        return rows

    for csv_path in sorted(simfin_csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "Ticker" not in df.columns:
            continue
        sdf = df[df["Ticker"].astype(str).str.upper() == ticker.upper()].copy()
        if sdf.empty:
            continue
        date_col = "Publish Date" if "Publish Date" in sdf.columns else None
        if date_col is None:
            for cand in ["Report Date", "Date", "Fiscal Date"]:
                if cand in sdf.columns:
                    date_col = cand
                    break
        if date_col is None:
            continue

        for _, r in sdf.iterrows():
            d = pd.to_datetime(r[date_col], errors="coerce")
            if pd.isna(d):
                continue
            rec = {
                "report_date": d.date().isoformat(),
                "ticker": ticker,
                "source": f"simfin:{csv_path.stem}",
                "fields": {
                    str(k): (None if pd.isna(v) else v)
                    for k, v in r.items()
                    if k not in {"Ticker", date_col}
                },
            }
            rows.append(rec)
    rows.sort(key=lambda x: x["report_date"])
    return rows


def collect_fundamentals(cfg: FundamentalsCollectConfig) -> dict[str, Path]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = cfg.universe or DEFAULT_UNIVERSE
    simfin_dir = Path(cfg.simfin_csv_dir) if cfg.simfin_csv_dir else None

    saved: dict[str, Path] = {}
    for ticker in universe:
        rows: list[dict[str, Any]] = []
        if simfin_dir is not None:
            rows.extend(_load_simfin_rows(simfin_dir, ticker))

        if not rows:
            try:
                rows = _from_yfinance(ticker)
            except Exception:
                rows = []

        if not rows:
            continue

        rows.sort(key=lambda x: x["report_date"])
        out_path = out_dir / f"{ticker.replace('/', '_')}.jsonl"
        write_jsonl(out_path, rows)
        saved[ticker] = out_path

    return saved


def collect_fundamentals_from_config(config: dict) -> dict[str, Path]:
    c = config.get("data", config)
    cfg = FundamentalsCollectConfig(
        output_dir=c["fundamentals_output_dir"],
        universe=list(c.get("universe", DEFAULT_UNIVERSE)),
        simfin_csv_dir=c.get("simfin_csv_dir"),
    )
    return collect_fundamentals(cfg)


def load_fundamental_rows(fundamentals_dir: str | Path, ticker: str) -> list[dict[str, Any]]:
    p = Path(fundamentals_dir) / f"{ticker.replace('/', '_')}.jsonl"
    if not p.exists():
        return []

    import json

    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
