"""Build PromptSample JSONL from collected modalities."""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_r1.data.collect_fundamentals import load_fundamental_rows
from trading_r1.data.collect_news import bucket_news, load_news_rows
from trading_r1.data.collect_price import DEFAULT_UNIVERSE, load_price_frame
from trading_r1.schemas import PromptSample
from trading_r1.utils.io import read_jsonl, write_jsonl


@dataclass
class BuildSamplesConfig:
    price_dir: str
    news_dir: str
    fundamentals_dir: str
    output_path: str
    labels_path: str | None
    start_date: str
    end_date: str
    universe: list[str]
    variants_per_day: int = 20
    random_seed: int = 42


def _d(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _fmt_num(v: Any) -> str:
    if pd.isna(v):
        return "NA"
    if isinstance(v, (int, float)):
        return f"{v:.6g}"
    return str(v)


def _price_section(df: pd.DataFrame, trade_date: dt.date, window_days: int = 15) -> tuple[str, str]:
    sdf = df[df["Date"].dt.date <= trade_date].tail(window_days)
    if sdf.empty:
        return "<price>No price data available.</price>", trade_date.isoformat()

    latest = sdf.iloc[-1]
    rows = []
    for _, r in sdf.iterrows():
        rows.append(
            f"- {r['Date'].date().isoformat()}: O={_fmt_num(r.get('Open'))} H={_fmt_num(r.get('High'))} "
            f"L={_fmt_num(r.get('Low'))} C={_fmt_num(r.get('Close'))} V={_fmt_num(r.get('Volume'))}"
        )

    indicator_fields = [
        "SMA_50",
        "SMA_200",
        "EMA_50",
        "EMA_10",
        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        "RSI_14",
        "ROC_10",
        "ATR_14",
        "BB_MID",
        "BB_UPPER",
        "BB_LOWER",
        "Z_75",
    ]
    ind_lines = [f"- {k}: {_fmt_num(latest.get(k))}" for k in indicator_fields if k in sdf.columns]

    section = (
        "<price>\n"
        "### Last 15 Daily Bars\n"
        + "\n".join(rows)
        + "\n\n### Technical Indicators (latest)\n"
        + "\n".join(ind_lines)
        + "\n</price>"
    )
    start_window = sdf.iloc[0]["Date"].date().isoformat()
    return section, start_window


def _news_section(news_rows: list[dict[str, Any]], trade_date: dt.date) -> str:
    buckets = bucket_news(news_rows, trade_date)

    def _fmt_items(items: list[dict[str, Any]]) -> str:
        if not items:
            return "- None"
        out: list[str] = []
        for item in items:
            out.append(
                "- "
                f"[{item.get('published_at', 'NA')}] "
                f"{item.get('title', '')} "
                f"(source: {item.get('source', 'NA')}, provider: {item.get('provider', 'NA')})"
            )
        return "\n".join(out)

    return (
        "<news>\n"
        "### Last 3 days (max 10)\n"
        + _fmt_items(buckets["last_3_days"])
        + "\n\n### Last 4-10 days (max 20)\n"
        + _fmt_items(buckets["last_4_10_days"])
        + "\n\n### Last 11-30 days (max 20)\n"
        + _fmt_items(buckets["last_11_30_days"])
        + "\n</news>"
    )


def _fundamentals_section(fund_rows: list[dict[str, Any]], trade_date: dt.date) -> str:
    valid = []
    for row in fund_rows:
        try:
            d = _d(row["report_date"])
        except Exception:
            continue
        if d <= trade_date:
            valid.append(row)

    if not valid:
        return "<fundamentals>\n- No fundamentals available up to trade date.\n</fundamentals>"

    latest = sorted(valid, key=lambda x: x["report_date"])[-1]
    lines = [
        f"- report_date: {latest.get('report_date', 'NA')}",
        f"- source: {latest.get('source', 'NA')}",
    ]

    for key in ["quarterly_financials", "quarterly_balance_sheet", "quarterly_cashflow", "fields"]:
        payload = latest.get(key)
        if isinstance(payload, dict) and payload:
            lines.append(f"\n### {key}")
            for k, v in list(payload.items())[:60]:
                lines.append(f"- {k}: {_fmt_num(v)}")

    return "<fundamentals>\n" + "\n".join(lines) + "\n</fundamentals>"


def _load_label_lookup(labels_path: str | None) -> dict[tuple[str, str], str]:
    if not labels_path:
        return {}
    rows = read_jsonl(labels_path)
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        k = (str(row.get("ticker", "")).upper(), str(row.get("trade_date", "")))
        v = str(row.get("label_action", ""))
        if k[0] and k[1] and v:
            out[k] = v
    return out


def build_prompt_samples(cfg: BuildSamplesConfig) -> list[dict[str, Any]]:
    label_lookup = _load_label_lookup(cfg.labels_path)
    start = _d(cfg.start_date)
    end = _d(cfg.end_date)

    records: list[dict[str, Any]] = []

    for ticker in cfg.universe:
        price_df = load_price_frame(cfg.price_dir, ticker)
        if price_df.empty:
            continue
        news_rows = load_news_rows(cfg.news_dir, ticker)
        fund_rows = load_fundamental_rows(cfg.fundamentals_dir, ticker)

        valid_days = price_df[
            (price_df["Date"].dt.date >= start) & (price_df["Date"].dt.date <= end)
        ]["Date"].dt.date.tolist()

        for trade_date in valid_days:
            price_block, features_window_start = _price_section(price_df, trade_date, window_days=15)
            news_block = _news_section(news_rows, trade_date)
            fund_block = _fundamentals_section(fund_rows, trade_date)

            sections = [price_block, news_block, fund_block]
            label = label_lookup.get((ticker.upper(), trade_date.isoformat()), "HOLD")

            for i in range(cfg.variants_per_day):
                rng = random.Random(f"{cfg.random_seed}:{ticker}:{trade_date}:{i}")
                shuffled = sections.copy()
                rng.shuffle(shuffled)

                sample = PromptSample(
                    sample_id=f"{ticker}_{trade_date.isoformat()}_v{i:02d}",
                    ticker=ticker,
                    trade_date=trade_date.isoformat(),
                    input_text="\n\n".join(shuffled),
                    label_action=label,
                    meta={
                        "features_window_start": features_window_start,
                        "features_window_end": trade_date.isoformat(),
                        "sources": ["yahoo", "finnhub", "google_news", "simfin"],
                    },
                )
                records.append(sample.to_dict())

    write_jsonl(cfg.output_path, records)
    return records


def build_prompt_samples_from_config(config: dict) -> list[dict[str, Any]]:
    c = config.get("data", config)
    cfg = BuildSamplesConfig(
        price_dir=c["price_output_dir"],
        news_dir=c["news_output_dir"],
        fundamentals_dir=c["fundamentals_output_dir"],
        output_path=c["samples_output_path"],
        labels_path=c.get("labels_path"),
        start_date=c["start_date"],
        end_date=c["end_date"],
        universe=list(c.get("universe", DEFAULT_UNIVERSE)),
        variants_per_day=int(c.get("variants_per_day", 20)),
        random_seed=int(c.get("random_seed", 42)),
    )
    return build_prompt_samples(cfg)
