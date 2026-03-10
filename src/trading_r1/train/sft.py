"""Stage-wise SFT training entrypoints."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trading_r1.train.checkpointing import build_best_checkpoint_callback
from trading_r1.train.runtime import resolve_training_runtime
from trading_r1.utils.chat_format import build_chat_prompt
from trading_r1.utils.io import read_jsonl


@dataclass
class SFTConfig:
    mode: str
    stage: int
    train_path: str
    val_path: str | None
    output_dir: str
    model_name: str
    max_seq_len: int
    num_train_epochs: int
    learning_rate: float
    batch_size: int
    grad_accum: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 2
    report_to: list[str] = field(default_factory=list)
    run_name: str | None = None
    wandb_project: str = "trading_r1"
    deepspeed_config: str | None = None
    resume_from_checkpoint: str | None = None


def _resolve_resume_checkpoint(cfg: SFTConfig) -> str | None:
    value = cfg.resume_from_checkpoint
    if not value:
        return None

    lowered = str(value).strip().lower()
    if lowered in {"", "none", "false", "no"}:
        return None

    if lowered == "latest":
        try:
            from transformers.trainer_utils import get_last_checkpoint  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "resume_from_checkpoint=latest requires transformers to be installed."
            ) from exc

        ckpt = get_last_checkpoint(cfg.output_dir)
        if not ckpt:
            raise RuntimeError(
                f"resume_from_checkpoint=latest but no checkpoint found in output_dir: {cfg.output_dir}"
            )
        return ckpt

    ckpt_path = Path(value)
    if not ckpt_path.exists():
        raise RuntimeError(f"resume checkpoint path does not exist: {ckpt_path}")
    return str(ckpt_path)


def _run_mock_sft(cfg: SFTConfig) -> dict[str, Any]:
    rows = read_jsonl(cfg.train_path)
    out_dir = Path(cfg.output_dir)
    ckpt = out_dir / "checkpoint-mock"
    ckpt.mkdir(parents=True, exist_ok=True)

    base = 1.2
    losses = [round(base / (1 + i * 0.4), 6) for i in range(max(1, cfg.num_train_epochs))]
    metrics = {
        "mode": "mock",
        "stage": cfg.stage,
        "samples": len(rows),
        "train_loss": losses,
        "final_loss": losses[-1],
    }
    (ckpt / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ckpt / "model_card.json").write_text(
        json.dumps(
            {
                "model": cfg.model_name,
                "stage": cfg.stage,
                "lora": {
                    "r": cfg.lora_r,
                    "alpha": cfg.lora_alpha,
                    "dropout": cfg.lora_dropout,
                    "targets": cfg.lora_target_modules,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def _run_hf_sft(cfg: SFTConfig) -> dict[str, Any]:  # pragma: no cover - heavy path
    try:
        import datasets  # type: ignore
        import torch
        from peft import LoraConfig, get_peft_model  # type: ignore
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "SFT mode=hf requires transformers/datasets/peft/torch dependencies."
        ) from exc

    if "wandb" in cfg.report_to:
        try:
            import wandb  # type: ignore  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "report_to includes 'wandb' but wandb is not installed. "
                "Install with `python -m pip install wandb`."
            ) from exc
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    train_rows = read_jsonl(cfg.train_path)
    if not train_rows:
        raise RuntimeError(f"No training rows found: {cfg.train_path}")
    val_rows = read_jsonl(cfg.val_path) if cfg.val_path else []
    runtime = resolve_training_runtime(torch_module=torch)

    def _fmt(row: dict[str, Any]) -> dict[str, str]:
        return {
            "text": build_chat_prompt(
                user_text=str(row["input_text"]),
                assistant_text=str(row["target_text"]),
            )
        }

    train_ds = datasets.Dataset.from_list([_fmt(r) for r in train_rows])
    eval_ds = datasets.Dataset.from_list([_fmt(r) for r in val_rows]) if val_rows else None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=runtime.torch_dtype)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_len,
            padding=False,
        )

    tokenized_train = train_ds.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized_eval = (
        eval_ds.map(_tokenize, batched=True, remove_columns=["text"]) if eval_ds is not None else None
    )

    training_args_init_params = inspect.signature(TrainingArguments.__init__).parameters
    training_args_kwargs: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "bf16": runtime.bf16,
        "fp16": runtime.fp16,
        "use_cpu": runtime.use_cpu,
        "logging_strategy": "steps",
        "logging_steps": cfg.logging_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "report_to": cfg.report_to,
        "run_name": cfg.run_name,
        "deepspeed": cfg.deepspeed_config,
    }
    if tokenized_eval is not None:
        if "eval_strategy" in training_args_init_params:
            training_args_kwargs["eval_strategy"] = "steps"
        else:
            training_args_kwargs["evaluation_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = cfg.save_steps
        training_args_kwargs["load_best_model_at_end"] = True
        training_args_kwargs["metric_for_best_model"] = "eval_loss"
        training_args_kwargs["greater_is_better"] = False

    filtered_training_args_kwargs = {
        key: value for key, value in training_args_kwargs.items() if key in training_args_init_params
    }
    args = TrainingArguments(**filtered_training_args_kwargs)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if runtime.use_cpu:
        print("[train-sft] No accelerator detected; using CPU training with float32.")

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": tokenized_train,
        "data_collator": collator,
        "callbacks": [
            build_best_checkpoint_callback(
                best_checkpoint_dirname="best-loss",
                metric_names=("eval_loss", "loss") if tokenized_eval is not None else ("loss",),
                greater_is_better=False,
            )
        ],
    }
    if tokenized_eval is not None:
        trainer_kwargs["eval_dataset"] = tokenized_eval
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    resume_ckpt = _resolve_resume_checkpoint(cfg)
    out = trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(cfg.output_dir)

    metrics = {
        "mode": "hf",
        "stage": cfg.stage,
        "samples": len(train_rows),
        "validation_samples": len(val_rows),
        "train_loss": float(out.training_loss),
        "device": runtime.device,
        "precision": runtime.precision,
        "use_cpu": runtime.use_cpu,
        "resumed_from_checkpoint": resume_ckpt,
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_loss_checkpoint_dir": (
            str(Path(cfg.output_dir) / "best-loss") if (Path(cfg.output_dir) / "best-loss").is_dir() else None
        ),
    }
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_sft(cfg: SFTConfig) -> dict[str, Any]:
    if cfg.mode == "mock":
        return _run_mock_sft(cfg)
    if cfg.mode == "hf":
        return _run_hf_sft(cfg)
    raise ValueError(f"Unsupported SFT mode: {cfg.mode}")


def train_sft_from_config(config: dict[str, Any]) -> dict[str, Any]:
    c = config.get("train_sft", config)
    cfg = SFTConfig(
        mode=str(c.get("mode", "mock")),
        stage=int(c.get("stage", 1)),
        train_path=c["train_path"],
        val_path=c.get("val_path"),
        output_dir=c["output_dir"],
        model_name=str(c.get("model_name", "Qwen/Qwen3-4B-Instruct")),
        max_seq_len=int(c.get("max_seq_len", 32768)),
        num_train_epochs=int(c.get("num_train_epochs", 1)),
        learning_rate=float(c.get("learning_rate", 2e-5)),
        batch_size=int(c.get("batch_size", 1)),
        grad_accum=int(c.get("grad_accum", 8)),
        lora_r=int(c.get("lora_r", 64)),
        lora_alpha=int(c.get("lora_alpha", 128)),
        lora_dropout=float(c.get("lora_dropout", 0.05)),
        lora_target_modules=list(
            c.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            )
        ),
        logging_steps=int(c.get("logging_steps", 5)),
        save_steps=int(c.get("save_steps", 100)),
        save_total_limit=int(c.get("save_total_limit", 2)),
        report_to=[str(x) for x in c.get("report_to", [])],
        run_name=str(c["run_name"]) if c.get("run_name") else None,
        wandb_project=str(c.get("wandb_project", "trading_r1")),
        deepspeed_config=c.get("deepspeed_config"),
        resume_from_checkpoint=c.get("resume_from_checkpoint"),
    )
    return train_sft(cfg)
