from __future__ import annotations

from pathlib import Path

import pytest

from trading_r1.train.grpo import GRPOConfig, _resolve_model_name_or_path


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
