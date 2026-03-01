"""CLI entrypoint for Trading-R1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_r1.config import load_config
from trading_r1.data.build_samples import build_prompt_samples_from_config
from trading_r1.data.collect_fundamentals import collect_fundamentals_from_config
from trading_r1.data.collect_news import collect_news_from_config
from trading_r1.data.collect_price import collect_price_data_from_config
from trading_r1.distill.trace_stitcher import distill_sft_and_grpo_from_config
from trading_r1.eval.backtest import run_backtest
from trading_r1.eval.inference import infer_action_for_ticker_date
from trading_r1.labels.volatility_labels import make_labels_from_config
from trading_r1.train.grpo import train_grpo_from_config
from trading_r1.train.sft import train_sft_from_config


def _cmd_collect_data(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    prices = collect_price_data_from_config(cfg)
    news = collect_news_from_config(cfg)
    funds = collect_fundamentals_from_config(cfg)
    print(
        json.dumps(
            {
                "price_tickers": len(prices),
                "news_tickers": len(news),
                "fundamentals_tickers": len(funds),
            },
            indent=2,
        )
    )
    return 0


def _cmd_build_samples(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    rows = build_prompt_samples_from_config(cfg)
    print(json.dumps({"samples": len(rows)}, indent=2))
    return 0


def _cmd_make_labels(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    rows = make_labels_from_config(cfg)
    print(json.dumps({"labels": len(rows)}, indent=2))
    return 0


def _cmd_distill_sft(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    sft, grpo = distill_sft_and_grpo_from_config(cfg)
    print(json.dumps({"sft_targets": len(sft), "grpo_items": len(grpo)}, indent=2))
    return 0


def _cmd_train_sft(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    metrics = train_sft_from_config(cfg)
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_train_grpo(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    metrics = train_grpo_from_config(cfg)
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out = infer_action_for_ticker_date(cfg, ticker=args.ticker, date=args.date)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    infer_cfg_path = cfg.get("backtest", cfg).get("infer_config_path", "configs/infer.yaml")
    infer_cfg = load_config(infer_cfg_path) if Path(infer_cfg_path).exists() else cfg
    result = run_backtest(cfg, infer_cfg)
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading_r1", description="Trading-R1 research CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect-data", help="Collect price/news/fundamentals")
    p_collect.add_argument("--config", required=True)
    p_collect.set_defaults(func=_cmd_collect_data)

    p_samples = sub.add_parser("build-samples", help="Build PromptSample JSONL")
    p_samples.add_argument("--config", required=True)
    p_samples.set_defaults(func=_cmd_build_samples)

    p_labels = sub.add_parser("make-labels", help="Generate volatility labels")
    p_labels.add_argument("--config", required=True)
    p_labels.set_defaults(func=_cmd_make_labels)

    p_distill = sub.add_parser("distill-sft", help="Generate SFTTarget and GRPOBatchItem")
    p_distill.add_argument("--config", required=True)
    p_distill.set_defaults(func=_cmd_distill_sft)

    p_sft = sub.add_parser("train-sft", help="Run SFT stage")
    p_sft.add_argument("--config", required=True)
    p_sft.set_defaults(func=_cmd_train_sft)

    p_grpo = sub.add_parser("train-grpo", help="Run GRPO stage")
    p_grpo.add_argument("--config", required=True)
    p_grpo.set_defaults(func=_cmd_train_grpo)

    p_infer = sub.add_parser("infer", help="Infer action for ticker/date")
    p_infer.add_argument("--config", required=True)
    p_infer.add_argument("--date", required=True)
    p_infer.add_argument("--ticker", required=True)
    p_infer.add_argument("--output")
    p_infer.set_defaults(func=_cmd_infer)

    p_backtest = sub.add_parser("backtest", help="Run backtest on sample window")
    p_backtest.add_argument("--config", required=True)
    p_backtest.set_defaults(func=_cmd_backtest)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))
