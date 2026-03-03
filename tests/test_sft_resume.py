from __future__ import annotations

from pathlib import Path

import pytest

from trading_r1.train.sft import SFTConfig, _resolve_resume_checkpoint


def _base_cfg(tmp_path: Path) -> SFTConfig:
    return SFTConfig(
        mode="hf",
        stage=1,
        train_path=str(tmp_path / "train.jsonl"),
        val_path=None,
        output_dir=str(tmp_path / "out"),
        model_name="Qwen/Qwen3-4B-Instruct",
        max_seq_len=32768,
        num_train_epochs=1,
        learning_rate=2e-5,
        batch_size=1,
        grad_accum=1,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )


def test_resolve_resume_none(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg.resume_from_checkpoint = None
    assert _resolve_resume_checkpoint(cfg) is None


def test_resolve_resume_explicit_path(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir(parents=True)
    cfg.resume_from_checkpoint = str(ckpt)
    assert _resolve_resume_checkpoint(cfg) == str(ckpt)


def test_resolve_resume_missing_path_raises(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg.resume_from_checkpoint = str(tmp_path / "checkpoint-missing")
    with pytest.raises(RuntimeError):
        _resolve_resume_checkpoint(cfg)
