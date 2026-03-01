"""Decision parser for `DECISION: [[[...]]]` tails."""

from __future__ import annotations

import re

from trading_r1.actions import CANONICAL_ACTION_MAP


_DECISION_PATTERN = re.compile(
    r"DECISION\s*:\s*\[\[\[\s*([A-Z_\- ]+)\s*\]\]\]",
    re.IGNORECASE,
)


def normalize_action(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper().replace("  ", " ")
    return CANONICAL_ACTION_MAP.get(normalized)


def extract_decision(text: str, strict_last_three_lines: bool = False) -> str | None:
    scope = text
    if strict_last_three_lines:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        scope = "\n".join(lines[-3:]) if lines else ""

    matches = list(_DECISION_PATTERN.finditer(scope))
    if not matches:
        return None
    raw = matches[-1].group(1)
    return normalize_action(raw)


def decision_format_valid(text: str) -> bool:
    return extract_decision(text, strict_last_three_lines=True) is not None
