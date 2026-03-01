"""Dataclass schemas for JSONL artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PromptSample:
    sample_id: str
    ticker: str
    trade_date: str
    input_text: str
    label_action: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SFTTarget:
    sample_id: str
    input_text: str
    target_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GRPOBatchItem:
    sample_id: str
    prompt: str
    ground_truth_action: str
    reward_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
