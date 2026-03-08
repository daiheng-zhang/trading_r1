from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_r1.train.grpo import (
    GRPOConfig,
    _is_peft_adapter_checkpoint,
    _resolve_generation_batching,
    _resolve_model_name_or_path,
    _resolve_peft_base_model_name_or_path,
    _resolve_tokenizer_name_or_path,
)


def _base_cfg(tmp_path: Path) -> GRPOConfig:
    return GRPOConfig(
        mode="trl",
        stage=1,
        train_path=str(tmp_path / "grpo.jsonl"),
        output_dir=str(tmp_path / "out"),
        model_name_or_path=str(tmp_path / "sft_out"),
        group_size=8,
        clip_eps=0.2,
        kl_beta=0.03,
        invalid_decision_reward=-1.5,
    )


def test_resolve_model_path_none(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg.load_from_checkpoint = None
    assert _resolve_model_name_or_path(cfg) == cfg.model_name_or_path


def test_resolve_model_path_explicit_checkpoint(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True)
    cfg.load_from_checkpoint = str(ckpt)
    assert _resolve_model_name_or_path(cfg) == str(ckpt)


def test_resolve_model_path_missing_checkpoint_raises(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg.load_from_checkpoint = str(tmp_path / "checkpoint-missing")
    with pytest.raises(RuntimeError):
        _resolve_model_name_or_path(cfg)


def test_generation_batching_auto_adjusts_to_valid_lcm(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg.per_device_train_batch_size = 1
    cfg.gradient_accumulation_steps = 1
    cfg.group_size = 8
    overrides, auto_adjusted = _resolve_generation_batching(cfg, world_size=4)
    assert auto_adjusted is True
    assert overrides == {"generation_batch_size": 8}


def test_generation_batching_no_adjust_when_divisible(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg.per_device_train_batch_size = 1
    cfg.gradient_accumulation_steps = 2
    cfg.group_size = 8
    overrides, auto_adjusted = _resolve_generation_batching(cfg, world_size=4)
    assert auto_adjusted is False
    assert overrides == {}


def test_generation_batching_prefers_explicit_generation_batch_size(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg.generation_batch_size = 24
    overrides, auto_adjusted = _resolve_generation_batching(cfg, world_size=4)
    assert auto_adjusted is False
    assert overrides == {"generation_batch_size": 24}


def test_generation_batching_rejects_conflicting_overrides(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg.generation_batch_size = 16
    cfg.steps_per_generation = 2
    with pytest.raises(ValueError):
        _resolve_generation_batching(cfg, world_size=4)


def test_detects_peft_adapter_checkpoint_without_config_json(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507"}),
        encoding="utf-8",
    )
    assert _is_peft_adapter_checkpoint(ckpt) is True


def test_resolve_peft_base_model_name_or_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507"}),
        encoding="utf-8",
    )
    assert _resolve_peft_base_model_name_or_path(ckpt) == "Qwen/Qwen3-4B-Instruct-2507"


def test_resolve_peft_base_model_name_or_path_raises_when_missing(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        _resolve_peft_base_model_name_or_path(ckpt)


def test_resolve_tokenizer_name_or_path_prefers_local_tokenizer(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507"}),
        encoding="utf-8",
    )
    (ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert _resolve_tokenizer_name_or_path(ckpt) == str(ckpt)


def test_resolve_tokenizer_name_or_path_falls_back_to_peft_base_model(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507"}),
        encoding="utf-8",
    )
    assert _resolve_tokenizer_name_or_path(ckpt) == "Qwen/Qwen3-4B-Instruct-2507"
