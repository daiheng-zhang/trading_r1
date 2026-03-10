from __future__ import annotations

import json
from pathlib import Path

from trading_r1.train.checkpointing import (
    BEST_CHECKPOINT_METADATA,
    coerce_scalar_metric,
    extract_logged_metric,
    is_metric_improved,
    load_best_metric,
)


def test_coerce_scalar_metric_accepts_finite_values() -> None:
    assert coerce_scalar_metric(1) == 1.0
    assert coerce_scalar_metric("2.5") == 2.5


def test_coerce_scalar_metric_rejects_invalid_values() -> None:
    assert coerce_scalar_metric(None) is None
    assert coerce_scalar_metric("nan") is None
    assert coerce_scalar_metric(float("inf")) is None


def test_extract_logged_metric_prefers_first_usable_metric() -> None:
    payload = {"loss": "bad", "eval_loss": "0.75", "reward": 1.2}
    assert extract_logged_metric(payload, ("loss", "eval_loss", "reward")) == ("eval_loss", 0.75)


def test_extract_logged_metric_returns_none_when_missing() -> None:
    assert extract_logged_metric({"learning_rate": 1e-5}, ("loss", "reward")) is None


def test_is_metric_improved_handles_min_and_max_modes() -> None:
    assert is_metric_improved(0.5, None, greater_is_better=False) is True
    assert is_metric_improved(0.5, 0.8, greater_is_better=False) is True
    assert is_metric_improved(0.9, 0.8, greater_is_better=False) is False
    assert is_metric_improved(1.1, 0.8, greater_is_better=True) is True
    assert is_metric_improved(0.7, 0.8, greater_is_better=True) is False


def test_load_best_metric_reads_metadata(tmp_path: Path) -> None:
    best_dir = tmp_path / "best-loss"
    best_dir.mkdir(parents=True)
    (best_dir / BEST_CHECKPOINT_METADATA).write_text(
        json.dumps({"metric_value": 0.42}),
        encoding="utf-8",
    )
    assert load_best_metric(best_dir) == 0.42


def test_load_best_metric_returns_none_for_missing_or_invalid_metadata(tmp_path: Path) -> None:
    best_dir = tmp_path / "best-loss"
    best_dir.mkdir(parents=True)
    assert load_best_metric(best_dir) is None

    (best_dir / BEST_CHECKPOINT_METADATA).write_text("{bad json", encoding="utf-8")
    assert load_best_metric(best_dir) is None
