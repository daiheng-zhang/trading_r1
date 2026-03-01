"""Reverse reasoning reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlannerConfig:
    max_steps: int = 6


def reconstruct_reasoning_steps(
    input_text: str,
    frontend_decision: str,
    config: dict | None = None,
) -> list[str]:
    _ = input_text
    cfg = PlannerConfig(max_steps=int((config or {}).get("max_steps", 6)))

    # Deterministic scaffold for reproducibility in offline dataset generation.
    steps = [
        "Parse and prioritize evidence from price, news, and fundamentals.",
        "Evaluate trend and momentum signals from recent bars and indicators.",
        "Assess fundamental quality, profitability, leverage, and balance-sheet resilience.",
        "Cross-check narrative catalysts and risks from time-bucketed news.",
        "Synthesize contradictions and produce a risk-adjusted thesis.",
        f"Map thesis conviction to final action: {frontend_decision}.",
    ]
    return steps[: cfg.max_steps]
