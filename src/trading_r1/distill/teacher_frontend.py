"""Teacher front-end recommendation generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from trading_r1.actions import ACTIONS


@dataclass
class TeacherConfig:
    provider: str = "mock"  # mock | openai
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    temperature: float = 0.2


def _mock_recommendation(input_text: str, label_hint: str | None = None) -> str:
    if label_hint in ACTIONS:
        return str(label_hint)

    low = input_text.lower()
    bullish = low.count("upgrade") + low.count("beat") + low.count("growth") + low.count("bullish")
    bearish = low.count("downgrade") + low.count("miss") + low.count("decline") + low.count("bearish")
    if bullish - bearish >= 2:
        return "BUY"
    if bearish - bullish >= 2:
        return "SELL"
    return "HOLD"


def _openai_recommendation(cfg: TeacherConfig, input_text: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package required for provider=openai") from exc

    if not cfg.api_key:
        raise RuntimeError("OpenAI API key missing for provider=openai")

    client = OpenAI(api_key=cfg.api_key)
    prompt = (
        "You are a trading teacher model. Given the market context, output only one token from this set: "
        "STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY. No explanation.\n\n"
        f"Context:\n{input_text}"
    )

    resp = client.responses.create(
        model=cfg.model,
        temperature=cfg.temperature,
        input=prompt,
    )
    text = resp.output_text.strip().upper().replace("-", "_").replace(" ", "_")
    if text in ACTIONS:
        return text
    return "HOLD"


def generate_frontend_recommendation(
    input_text: str,
    config: dict[str, Any],
    label_hint: str | None = None,
) -> str:
    cfg = TeacherConfig(
        provider=str(config.get("provider", "mock")),
        model=str(config.get("model", "gpt-4.1-mini")),
        api_key=config.get("api_key"),
        temperature=float(config.get("temperature", 0.2)),
    )

    if cfg.provider == "openai":
        return _openai_recommendation(cfg, input_text)
    return _mock_recommendation(input_text, label_hint=label_hint)


def write_frontend_decisions(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
