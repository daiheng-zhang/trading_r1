"""Checkpoint helpers shared across training entrypoints."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

BEST_CHECKPOINT_METADATA = "best_metric.json"


def coerce_scalar_metric(value: Any) -> float | None:
    """Return a finite float metric value, or None when the payload is unusable."""
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(metric) or math.isinf(metric):
        return None
    return metric


def extract_logged_metric(
    payload: dict[str, Any] | None,
    metric_names: Iterable[str],
) -> tuple[str, float] | None:
    """Pick the first usable metric from a callback payload."""
    if not payload:
        return None

    for metric_name in metric_names:
        if metric_name not in payload:
            continue
        metric_value = coerce_scalar_metric(payload[metric_name])
        if metric_value is not None:
            return metric_name, metric_value
    return None


def is_metric_improved(
    current_metric: float,
    best_metric: float | None,
    *,
    greater_is_better: bool,
) -> bool:
    """Return True when the latest metric beats the previous best."""
    if best_metric is None:
        return True
    if greater_is_better:
        return current_metric > best_metric
    return current_metric < best_metric


def load_best_metric(best_checkpoint_dir: str | Path) -> float | None:
    """Load persisted best-metric metadata when resuming training."""
    metadata_path = Path(best_checkpoint_dir) / BEST_CHECKPOINT_METADATA
    if not metadata_path.is_file():
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return coerce_scalar_metric(payload.get("metric_value"))


def build_best_checkpoint_callback(
    *,
    best_checkpoint_dirname: str,
    metric_names: Iterable[str],
    greater_is_better: bool,
):
    """Create a transformers callback that mirrors the best `checkpoint-N` directory."""
    try:
        from transformers import TrainerCallback  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-gated
        raise RuntimeError(
            "Best-checkpoint tracking requires transformers to be installed."
        ) from exc

    metric_names_tuple = tuple(metric_names)
    if not metric_names_tuple:
        raise ValueError("metric_names must contain at least one metric key.")

    class _BestCheckpointCallback(TrainerCallback):
        def __init__(self) -> None:
            self._best_metric: float | None = None
            self._latest_metric_name: str | None = None
            self._latest_metric_value: float | None = None

        def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
            if not state.is_world_process_zero:
                return control

            best_checkpoint_dir = Path(args.output_dir) / best_checkpoint_dirname
            self._best_metric = load_best_metric(best_checkpoint_dir)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            selected = extract_logged_metric(logs, metric_names_tuple)
            if selected is not None:
                self._latest_metric_name, self._latest_metric_value = selected
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
            selected = extract_logged_metric(metrics, metric_names_tuple)
            if selected is not None:
                self._latest_metric_name, self._latest_metric_value = selected
            return control

        def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
            if not state.is_world_process_zero or self._latest_metric_value is None:
                return control

            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not checkpoint_dir.is_dir():
                return control

            if not is_metric_improved(
                self._latest_metric_value,
                self._best_metric,
                greater_is_better=greater_is_better,
            ):
                return control

            best_checkpoint_dir = Path(args.output_dir) / best_checkpoint_dirname
            if best_checkpoint_dir.exists():
                shutil.rmtree(best_checkpoint_dir)
            shutil.copytree(checkpoint_dir, best_checkpoint_dir)

            metadata = {
                "metric_name": self._latest_metric_name,
                "metric_value": self._latest_metric_value,
                "greater_is_better": greater_is_better,
                "source_checkpoint": str(checkpoint_dir),
                "global_step": state.global_step,
            }
            (best_checkpoint_dir / BEST_CHECKPOINT_METADATA).write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            self._best_metric = self._latest_metric_value
            return control

    return _BestCheckpointCallback()
