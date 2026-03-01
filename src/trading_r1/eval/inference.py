"""Inference runtime for daily action generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading_r1.actions import ACTIONS
from trading_r1.parsing.decision_parser import extract_decision
from trading_r1.utils.io import read_jsonl


@dataclass
class InferenceConfig:
    mode: str
    model_name_or_path: str
    samples_path: str
    max_new_tokens: int = 1024


def _find_sample(samples: list[dict[str, Any]], ticker: str, date: str) -> dict[str, Any] | None:
    ticker_u = ticker.upper()
    for row in samples:
        if str(row.get("ticker", "")).upper() == ticker_u and str(row.get("trade_date", "")) == date:
            return row
    return None


def _mock_predict(sample: dict[str, Any]) -> tuple[str, str]:
    label = str(sample.get("label_action", "HOLD"))
    action = label if label in ACTIONS else "HOLD"
    completion = (
        "<think>\n- mock inference\n</think>\n"
        "<fundamentals>\n- placeholder\n</fundamentals>\n"
        "<technical>\n- placeholder\n</technical>\n"
        "<news>\n- placeholder\n</news>\n"
        "<valuation>\n- placeholder\n</valuation>\n"
        "<risk>\n- placeholder\n</risk>\n"
        "<macro>\n- placeholder\n</macro>\n"
        "<conclusion>\n- placeholder\n</conclusion>\n"
        f"DECISION: [[[{action}]]]"
    )
    return action, completion


def _hf_predict(cfg: InferenceConfig, sample: dict[str, Any]) -> tuple[str, str]:  # pragma: no cover
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:
        raise RuntimeError("Inference mode=hf requires transformers and torch") from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, torch_dtype=torch.bfloat16)
    prompt = str(sample.get("input_text", ""))
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    action = extract_decision(text) or "HOLD"
    return action, text


def infer_action_for_ticker_date(config: dict[str, Any], ticker: str, date: str) -> dict[str, Any]:
    c = config.get("infer", config)
    cfg = InferenceConfig(
        mode=str(c.get("mode", "mock")),
        model_name_or_path=str(c.get("model_name_or_path", "Qwen/Qwen3-4B-Instruct")),
        samples_path=c["samples_path"],
        max_new_tokens=int(c.get("max_new_tokens", 1024)),
    )

    samples = read_jsonl(cfg.samples_path)
    sample = _find_sample(samples, ticker=ticker, date=date)
    if sample is None:
        raise RuntimeError(f"No sample found for ticker={ticker}, date={date}")

    if cfg.mode == "hf":
        action, completion = _hf_predict(cfg, sample)
    else:
        action, completion = _mock_predict(sample)

    return {
        "sample_id": sample.get("sample_id"),
        "ticker": ticker,
        "trade_date": date,
        "action": action,
        "completion": completion,
    }
