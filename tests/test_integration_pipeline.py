from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from trading_r1.data.build_samples import build_prompt_samples
from trading_r1.data.build_samples import BuildSamplesConfig
from trading_r1.eval.backtest import run_backtest
from trading_r1.labels.volatility_labels import LabelConfig, make_labels
from trading_r1.train.grpo import train_grpo
from trading_r1.train.grpo import GRPOConfig
from trading_r1.train.sft import train_sft
from trading_r1.train.sft import SFTConfig
from trading_r1.utils.io import write_jsonl


def _make_price_csv(path: Path, ticker: str, n: int = 40) -> None:
    dates = pd.date_range("2024-06-01", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), dtype=float)
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close + 0.5,
            "Volume": 1_000_000,
            "SMA_50": close.rolling(5, min_periods=1).mean(),
            "SMA_200": close.rolling(10, min_periods=1).mean(),
            "EMA_50": close.ewm(span=5).mean(),
            "EMA_10": close.ewm(span=3).mean(),
            "MACD": 0.1,
            "MACD_SIGNAL": 0.05,
            "MACD_HIST": 0.05,
            "RSI_14": 55.0,
            "ROC_10": 0.01,
            "ATR_14": 1.2,
            "BB_MID": close,
            "BB_UPPER": close + 2,
            "BB_LOWER": close - 2,
            "Z_75": 0.0,
        }
    )
    df.to_csv(path / f"{ticker}.csv", index=False)


def _make_news_jsonl(path: Path, ticker: str) -> None:
    rows = [
        {
            "provider": "finnhub",
            "ticker": ticker,
            "published_at": "2024-06-20",
            "title": f"{ticker} positive catalyst",
            "summary": "summary",
            "source": "src",
            "url": "u",
        },
        {
            "provider": "google_news",
            "ticker": ticker,
            "published_at": "2024-06-10",
            "title": f"{ticker} neutral",
            "summary": "summary",
            "source": "src",
            "url": "u",
        },
    ]
    write_jsonl(path / f"{ticker}.jsonl", rows)


def _make_fund_jsonl(path: Path, ticker: str) -> None:
    rows = [
        {
            "report_date": "2024-06-01",
            "ticker": ticker,
            "source": "simfin",
            "fields": {"Revenue": 1000, "NetIncome": 120},
        }
    ]
    write_jsonl(path / f"{ticker}.jsonl", rows)


def test_end_to_end_pipeline_and_training_smoke(tmp_path: Path) -> None:
    price_dir = tmp_path / "price"
    news_dir = tmp_path / "news"
    fund_dir = tmp_path / "fund"
    out_dir = tmp_path / "out"
    price_dir.mkdir()
    news_dir.mkdir()
    fund_dir.mkdir()
    out_dir.mkdir()

    tickers = ["AAPL", "MSFT"]
    for t in tickers:
        _make_price_csv(price_dir, t)
        _make_news_jsonl(news_dir, t)
        _make_fund_jsonl(fund_dir, t)

    labels_path = out_dir / "labels.jsonl"
    label_cfg = LabelConfig(price_dir=str(price_dir), output_path=str(labels_path))
    label_rows = make_labels(label_cfg, tickers=tickers)
    assert len(label_rows) > 0

    samples_path = out_dir / "samples.jsonl"
    sample_cfg = BuildSamplesConfig(
        price_dir=str(price_dir),
        news_dir=str(news_dir),
        fundamentals_dir=str(fund_dir),
        output_path=str(samples_path),
        labels_path=str(labels_path),
        start_date="2024-06-10",
        end_date="2024-07-10",
        universe=tickers,
        variants_per_day=2,
        random_seed=7,
    )
    samples = build_prompt_samples(sample_cfg)
    assert len(samples) > 0

    # Build simple SFT and GRPO artifacts.
    sft_path = out_dir / "sft.jsonl"
    grpo_path = out_dir / "grpo.jsonl"
    sft_rows = [
        {
            "sample_id": r["sample_id"],
            "input_text": r["input_text"],
            "target_text": "<think>p</think><fundamentals>- a</fundamentals><technical>- b</technical><news>- c</news><valuation>- d</valuation><risk>- e</risk><macro>- f</macro><conclusion>- g</conclusion>\nDECISION: [[[BUY]]]",
        }
        for r in samples[:12]
    ]
    write_jsonl(sft_path, sft_rows)

    grpo_rows = [
        {
            "sample_id": r["sample_id"],
            "prompt": r["input_text"],
            "ground_truth_action": r["label_action"],
            "reward_context": {"allowed_tags": ["fundamentals", "technical", "news", "valuation", "risk", "macro", "conclusion"]},
        }
        for r in samples[:12]
    ]
    write_jsonl(grpo_path, grpo_rows)

    sft_metrics = train_sft(
        SFTConfig(
            mode="mock",
            stage=1,
            train_path=str(sft_path),
            val_path=None,
            output_dir=str(out_dir / "sft_ckpt"),
            model_name="Qwen/Qwen3-4B-Instruct",
            max_seq_len=32768,
            num_train_epochs=3,
            learning_rate=2e-5,
            batch_size=1,
            grad_accum=1,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_target_modules=["q", "k", "v", "o", "up", "down", "gate"],
        )
    )
    losses = sft_metrics["train_loss"]
    assert losses[0] > losses[-1]

    grpo_metrics = train_grpo(
        GRPOConfig(
            mode="mock",
            stage=3,
            train_path=str(grpo_path),
            output_dir=str(out_dir / "grpo_ckpt"),
            model_name_or_path="mock",
            group_size=8,
            clip_eps=0.2,
            kl_beta=0.03,
            invalid_decision_reward=-1.5,
        )
    )
    assert math.isfinite(grpo_metrics["mean_reward"])
    assert math.isfinite(grpo_metrics["policy_loss"])

    infer_cfg = {
        "infer": {
            "mode": "mock",
            "model_name_or_path": "mock",
            "samples_path": str(samples_path),
            "max_new_tokens": 128,
        }
    }
    backtest_cfg = {
        "backtest": {
            "infer_config_path": "",
            "samples_path": str(samples_path),
            "price_dir": str(price_dir),
            "output_dir": str(out_dir / "backtest"),
            "start_date": "2024-06-17",
            "end_date": "2024-07-05",
            "transaction_cost_bps": 5.0,
            "rf_annual": 0.04,
            "periods_per_year": 252,
        }
    }

    result = run_backtest(backtest_cfg, infer_cfg)
    assert "metrics" in result
    for k in ["CR", "SR", "HR", "MDD"]:
        assert k in result["metrics"]
        assert math.isfinite(float(result["metrics"][k]))
