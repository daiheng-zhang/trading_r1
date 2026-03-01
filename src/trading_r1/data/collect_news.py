"""News collection from Finnhub and Google News RSS."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests

from trading_r1.data.collect_price import DEFAULT_UNIVERSE
from trading_r1.utils.io import write_jsonl


@dataclass
class NewsCollectConfig:
    output_dir: str
    start_date: str
    end_date: str
    universe: list[str] | None = None
    finnhub_api_key: str | None = None
    include_google_news: bool = True


def _date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _parse_google_rss_datetime(text: str) -> str | None:
    for fmt in ["%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"]:
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def fetch_google_news(ticker: str, max_items: int = 100) -> list[dict[str, Any]]:
    q = quote_plus(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    items = root.findall(".//item")
    rows: list[dict[str, Any]] = []
    for item in items[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()
        published_at = _parse_google_rss_datetime(pub)
        if not published_at:
            continue
        rows.append(
            {
                "provider": "google_news",
                "ticker": ticker,
                "published_at": published_at,
                "title": title,
                "summary": description,
                "source": "Google News RSS",
                "url": link,
            }
        )
    return rows


def fetch_finnhub_news(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> list[dict[str, Any]]:
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": start_date, "to": end_date, "token": api_key}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    rows: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        ts = item.get("datetime")
        if ts is None:
            continue
        published = dt.datetime.utcfromtimestamp(int(ts)).date().isoformat()
        rows.append(
            {
                "provider": "finnhub",
                "ticker": ticker,
                "published_at": published,
                "title": item.get("headline") or "",
                "summary": item.get("summary") or "",
                "source": item.get("source") or "Finnhub",
                "url": item.get("url") or "",
            }
        )
    return rows


def _dedup_news(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("published_at", ""), row.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.sort(key=lambda x: (x.get("published_at", ""), x.get("title", "")), reverse=True)
    return out


def collect_news(cfg: NewsCollectConfig) -> dict[str, Path]:
    universe = cfg.universe or DEFAULT_UNIVERSE
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for ticker in universe:
        rows: list[dict[str, Any]] = []
        if cfg.finnhub_api_key:
            try:
                rows.extend(fetch_finnhub_news(ticker, cfg.start_date, cfg.end_date, cfg.finnhub_api_key))
            except Exception:
                pass
        if cfg.include_google_news:
            try:
                rows.extend(fetch_google_news(ticker))
            except Exception:
                pass

        rows = _dedup_news(rows)
        if not rows:
            continue

        out_path = out_dir / f"{ticker.replace('/', '_')}.jsonl"
        write_jsonl(out_path, rows)
        saved[ticker] = out_path

    return saved


def collect_news_from_config(config: dict) -> dict[str, Path]:
    c = config.get("data", config)
    cfg = NewsCollectConfig(
        output_dir=c["news_output_dir"],
        start_date=c["start_date"],
        end_date=c["end_date"],
        universe=list(c.get("universe", DEFAULT_UNIVERSE)),
        finnhub_api_key=c.get("finnhub_api_key"),
        include_google_news=bool(c.get("include_google_news", True)),
    )
    return collect_news(cfg)


def load_news_rows(news_dir: str | Path, ticker: str) -> list[dict[str, Any]]:
    path = Path(news_dir) / f"{ticker.replace('/', '_')}.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json

            rows.append(json.loads(line))
    return rows


def bucket_news(
    rows: list[dict[str, Any]], trade_date: dt.date
) -> dict[str, list[dict[str, Any]]]:
    b1: list[dict[str, Any]] = []  # last 3 days
    b2: list[dict[str, Any]] = []  # 4-10
    b3: list[dict[str, Any]] = []  # 11-30
    for row in rows:
        try:
            d = _date(row["published_at"])
        except Exception:
            continue
        delta = (trade_date - d).days
        if delta < 0 or delta > 30:
            continue
        if delta <= 3:
            b1.append(row)
        elif 4 <= delta <= 10:
            b2.append(row)
        elif 11 <= delta <= 30:
            b3.append(row)

    b1 = b1[:10]
    b2 = b2[:20]
    b3 = b3[:20]
    return {
        "last_3_days": b1,
        "last_4_10_days": b2,
        "last_11_30_days": b3,
    }
