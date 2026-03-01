"""Fixed split policy and leakage helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DateWindow:
    start: str
    end: str


TRAIN_WINDOWS = [
    DateWindow("2024-01-02", "2024-05-31"),
    DateWindow("2024-09-03", "2025-03-31"),
]
VALID_WINDOW = DateWindow("2025-04-01", "2025-05-31")
HOLDOUT_WINDOW = DateWindow("2024-06-03", "2024-08-30")


def in_window(date: str, window: DateWindow) -> bool:
    return window.start <= date <= window.end


def split_of(date: str) -> str:
    if any(in_window(date, w) for w in TRAIN_WINDOWS):
        return "train"
    if in_window(date, VALID_WINDOW):
        return "validation"
    if in_window(date, HOLDOUT_WINDOW):
        return "holdout"
    return "outside"
