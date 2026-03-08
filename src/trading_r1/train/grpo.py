"""Stage-wise GRPO training entrypoints."""

from __future__ import annotations

import inspect
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trading_r1.reward.aggregate import aggregate_reward
from trading_r1.train.runtime import resolve_training_runtime
from trading_r1.utils.chat_format import append_instruction_to_user_turn
from trading_r1.utils.io import read_jsonl


@dataclass
class GRPOConfig:
    mode: str
    stage: int
    train_path: str
    output_dir: str
    model_name_or_path: str
    group_size: int
    clip_eps: float
    kl_beta: float
    invalid_decision_reward: float
    load_from_checkpoint: str | None = None
    num_train_epochs: int = 1
    learning_rate: float = 1e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    generation_batch_size: int | None = None
    steps_per_generation: int | None = None
    logging_steps: int = 1
    max_prompt_length: int = 32768
    max_completion_length: int = 1024
    temperature: float = 1.0
    format_instruction: str | None = None
    log_completions: bool = False
    num_completions_to_print: int | None = None
    report_to: list[str] = field(default_factory=list)
    run_name: str | None = None
    wandb_project: str = "trading_r1"


def _prepare_grpo_prompt(prompt: str, format_instruction: str | None = None) -> str:
    return append_instruction_to_user_turn(prompt, format_instruction)


def _resolve_world_size() -> int:
    raw = os.getenv("WORLD_SIZE", "1").strip()
    try:
        value = int(raw)
    except ValueError:
        return 1
    return value if value > 0 else 1


def _resolve_generation_batching(
    cfg: GRPOConfig,
    world_size: int | None = None,
) -> tuple[dict[str, int], bool]:
    if cfg.group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {cfg.group_size}.")
    if cfg.per_device_train_batch_size < 1:
        raise ValueError(
            f"per_device_train_batch_size must be >= 1, got {cfg.per_device_train_batch_size}."
        )
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, got {cfg.gradient_accumulation_steps}."
        )
    if cfg.generation_batch_size is not None and cfg.generation_batch_size < 1:
        raise ValueError(f"generation_batch_size must be >= 1, got {cfg.generation_batch_size}.")
    if cfg.steps_per_generation is not None and cfg.steps_per_generation < 1:
        raise ValueError(f"steps_per_generation must be >= 1, got {cfg.steps_per_generation}.")

    if cfg.generation_batch_size is not None and cfg.steps_per_generation is not None:
        raise ValueError(
            "generation_batch_size and steps_per_generation are mutually exclusive. Set only one."
        )
    if cfg.generation_batch_size is not None:
        return {"generation_batch_size": int(cfg.generation_batch_size)}, False
    if cfg.steps_per_generation is not None:
        return {"steps_per_generation": int(cfg.steps_per_generation)}, False

    effective_world_size = world_size if world_size is not None else _resolve_world_size()
    if effective_world_size < 1:
        effective_world_size = 1

    default_generation_batch_size = (
        cfg.per_device_train_batch_size * effective_world_size * cfg.gradient_accumulation_steps
    )
    if default_generation_batch_size % cfg.group_size == 0:
        return {}, False

    adjusted_generation_batch_size = math.lcm(default_generation_batch_size, cfg.group_size)
    return {"generation_batch_size": int(adjusted_generation_batch_size)}, True


def _resolve_model_name_or_path(cfg: GRPOConfig) -> str:
    value = cfg.load_from_checkpoint
    if not value:
        return cfg.model_name_or_path

    lowered = str(value).strip().lower()
    if lowered in {"", "none", "false", "no"}:
        return cfg.model_name_or_path

    if lowered == "latest":
        try:
            from transformers.trainer_utils import get_last_checkpoint  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "load_from_checkpoint=latest requires transformers to be installed."
            ) from exc

        ckpt = get_last_checkpoint(cfg.model_name_or_path)
        if not ckpt:
            raise RuntimeError(
                "load_from_checkpoint=latest but no checkpoint found under model_name_or_path: "
                f"{cfg.model_name_or_path}"
            )
        return ckpt

    ckpt_path = Path(value)
    if not ckpt_path.exists():
        raise RuntimeError(f"checkpoint path does not exist: {ckpt_path}")
    return str(ckpt_path)


def _is_peft_adapter_checkpoint(model_name_or_path: str | Path) -> bool:
    path = Path(model_name_or_path)
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and not (path / "config.json").exists()
    )


def _resolve_peft_base_model_name_or_path(model_name_or_path: str | Path) -> str:
    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    try:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read PEFT adapter config from {adapter_config_path}."
        ) from exc

    base_model_name_or_path = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model_name_or_path, str) or not base_model_name_or_path.strip():
        raise RuntimeError(
            "PEFT adapter checkpoint is missing `base_model_name_or_path` in adapter_config.json: "
            f"{adapter_config_path}"
        )
    return base_model_name_or_path


def _resolve_tokenizer_name_or_path(model_name_or_path: str | Path) -> str:
    path = Path(model_name_or_path)
    if path.is_dir() and (path / "tokenizer_config.json").is_file():
        return str(path)
    if _is_peft_adapter_checkpoint(path):
        return _resolve_peft_base_model_name_or_path(path)
    return str(path)


def _run_mock_grpo(cfg: GRPOConfig) -> dict[str, Any]:
    rows = read_jsonl(cfg.train_path)

    rewards: list[float] = []
    for row in rows[:500]:
        completion = (
            "<think>\n- mock reasoning\n</think>\n"
            "<fundamentals>\n- Opinion *quote* `src`\n- Opinion *quote* `src`\n"
            "- Opinion *quote* `src`\n- Opinion *quote* `src`\n</fundamentals>\n"
            "<technical>\n- Opinion *quote* `src`\n- Opinion *quote* `src`\n"
            "- Opinion *quote* `src`\n- Opinion *quote* `src`\n</technical>\n"
            "<news>\n- Opinion *quote* `src`\n- Opinion *quote* `src`\n"
            "- Opinion *quote* `src`\n- Opinion *quote* `src`\n</news>\n"
            "<valuation>\n- Opinion *quote* `src`\n- Opinion *quote* `src`\n"
            "- Opinion *quote* `src`\n- Opinion *quote* `src`\n</valuation>\n"
            "<risk>\n- Opinion *quote* `src`\n- Opinion *quote* `src`\n"
            "- Opinion *quote* `src`\n- Opinion *quote* `src`\n</risk>\n"
            "<macro>\n- Opinion *quote* `src`\n- Opinion *quote* `src`\n"
            "- Opinion *quote* `src`\n- Opinion *quote* `src`\n</macro>\n"
            "<conclusion>\n- Done\n</conclusion>\n"
            f"DECISION: [[[{row.get('ground_truth_action', 'HOLD')}]]]"
        )
        scored = aggregate_reward(completion, str(row.get("ground_truth_action", "HOLD")), stage=cfg.stage)
        rewards.append(float(scored["reward_total"]))

    mean_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    policy_loss = float(1.0 / (1.0 + max(mean_reward, -0.99)))

    ckpt = Path(cfg.output_dir) / "checkpoint-mock"
    ckpt.mkdir(parents=True, exist_ok=True)

    metrics = {
        "mode": "mock",
        "stage": cfg.stage,
        "samples": len(rows),
        "group_size": cfg.group_size,
        "clip_eps": cfg.clip_eps,
        "kl_beta": cfg.kl_beta,
        "mean_reward": mean_reward,
        "policy_loss": policy_loss,
    }
    (ckpt / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _run_trl_grpo(cfg: GRPOConfig) -> dict[str, Any]:  # pragma: no cover - heavy path
    try:
        import datasets  # type: ignore
        import torch
        from trl import GRPOConfig as TRLGRPOConfig  # type: ignore
        from trl import GRPOTrainer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "GRPO mode=trl requires `trl`, `datasets`, and `torch`. "
            "Install optional train dependencies."
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

    rows = read_jsonl(cfg.train_path)
    if not rows:
        raise RuntimeError(f"No GRPO rows found at {cfg.train_path}")

    dataset = datasets.Dataset.from_list(
        [
            {
                "prompt": _prepare_grpo_prompt(
                    str(r.get("prompt", r.get("input_text", ""))),
                    cfg.format_instruction,
                ),
                "ground_truth_action": str(r.get("ground_truth_action", "HOLD")),
            }
            for r in rows
        ]
    )

    def reward_func(prompts, completions, ground_truth_action=None, **kwargs):  # type: ignore
        gt = ground_truth_action
        if gt is None:
            gt = kwargs.get("ground_truth_action")
        if gt is None:
            gt = ["HOLD"] * len(completions)

        rewards = []
        for completion, truth in zip(completions, gt):
            scored = aggregate_reward(str(completion), str(truth), stage=cfg.stage)
            rewards.append(float(scored["reward_total"]))
        return rewards

    world_size = _resolve_world_size()
    runtime = resolve_training_runtime(torch_module=torch)
    generation_overrides, auto_adjusted_generation_batch_size = _resolve_generation_batching(
        cfg, world_size=world_size
    )
    trl_cfg_kwargs: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "bf16": runtime.bf16,
        "fp16": runtime.fp16,
        "use_cpu": runtime.use_cpu,
        "logging_steps": cfg.logging_steps,
        "save_strategy": "epoch",
        "report_to": cfg.report_to,
        "run_name": cfg.run_name,
        "max_prompt_length": cfg.max_prompt_length,
        "max_completion_length": cfg.max_completion_length,
        "temperature": cfg.temperature,
        "log_completions": cfg.log_completions,
        "num_completions_to_print": cfg.num_completions_to_print,
        "num_generations": cfg.group_size,
        "model_init_kwargs": {"torch_dtype": runtime.torch_dtype},
        "beta": cfg.kl_beta,
        "epsilon": cfg.clip_eps,
    }
    trl_cfg_kwargs.update(generation_overrides)
    trl_cfg_init_params = inspect.signature(TRLGRPOConfig.__init__).parameters
    filtered_trl_cfg_kwargs = {k: v for k, v in trl_cfg_kwargs.items() if k in trl_cfg_init_params}
    missing_generation_keys = sorted(set(generation_overrides) - set(filtered_trl_cfg_kwargs))
    if missing_generation_keys:
        missing = ", ".join(missing_generation_keys)
        raise RuntimeError(
            "Installed trl.GRPOConfig does not support auto-adjustment keys "
            f"({missing}). Upgrade `trl` or set `group_size` so "
            "(per_device_train_batch_size * WORLD_SIZE * gradient_accumulation_steps) is divisible by group_size."
        )
    trl_cfg = TRLGRPOConfig(**filtered_trl_cfg_kwargs)
    model_name_or_path = _resolve_model_name_or_path(cfg)

    if runtime.use_cpu and os.getenv("LOCAL_RANK", "0") == "0":
        print("[train-grpo] No accelerator detected; using CPU training with float32.")

    if auto_adjusted_generation_batch_size and os.getenv("LOCAL_RANK", "0") == "0":
        print(
            "[train-grpo] Auto-adjusted generation_batch_size to "
            f"{trl_cfg.generation_batch_size} so it is divisible by num_generations={cfg.group_size}."
        )

    loaded_peft_adapter_checkpoint = _is_peft_adapter_checkpoint(model_name_or_path)
    trainer_model: Any = model_name_or_path
    processing_class = None
    tokenizer_name_or_path: str | None = None
    if loaded_peft_adapter_checkpoint:
        try:
            from peft import AutoPeftModelForCausalLM  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Loading PEFT adapter checkpoints for GRPO requires `peft`, `transformers`, and `torch`."
            ) from exc

        tokenizer_name_or_path = _resolve_tokenizer_name_or_path(model_name_or_path)
        trainer_model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            is_trainable=True,
            torch_dtype=runtime.torch_dtype,
        )
        processing_class = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

    trainer_kwargs: dict[str, Any] = {
        "model": trainer_model,
        "reward_funcs": reward_func,
        "args": trl_cfg,
        "train_dataset": dataset,
    }
    if processing_class is not None:
        trainer_kwargs["processing_class"] = processing_class

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(cfg.output_dir)

    metrics = {
        "mode": "trl",
        "stage": cfg.stage,
        "samples": len(rows),
        "group_size": cfg.group_size,
        "clip_eps": cfg.clip_eps,
        "kl_beta": cfg.kl_beta,
        "loaded_model_name_or_path": model_name_or_path,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "generation_batch_size": getattr(trl_cfg, "generation_batch_size", None),
        "steps_per_generation": getattr(trl_cfg, "steps_per_generation", None),
        "world_size": world_size,
        "device": runtime.device,
        "precision": runtime.precision,
        "use_cpu": runtime.use_cpu,
        "auto_adjusted_generation_batch_size": auto_adjusted_generation_batch_size,
        "loaded_peft_adapter_checkpoint": loaded_peft_adapter_checkpoint,
        "tokenizer_name_or_path": tokenizer_name_or_path,
    }
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_grpo(cfg: GRPOConfig) -> dict[str, Any]:
    if cfg.mode == "mock":
        return _run_mock_grpo(cfg)
    if cfg.mode == "trl":
        return _run_trl_grpo(cfg)
    raise ValueError(f"Unsupported GRPO mode: {cfg.mode}")


def train_grpo_from_config(config: dict[str, Any]) -> dict[str, Any]:
    c = config.get("train_grpo", config)
    cfg = GRPOConfig(
        mode=str(c.get("mode", "mock")),
        stage=int(c.get("stage", 1)),
        train_path=c["train_path"],
        output_dir=c["output_dir"],
        model_name_or_path=c.get("model_name_or_path", "Qwen/Qwen3-4B-Instruct"),
        load_from_checkpoint=c.get("load_from_checkpoint"),
        group_size=int(c.get("group_size", 8)),
        clip_eps=float(c.get("clip_eps", 0.2)),
        kl_beta=float(c.get("kl_beta", 0.03)),
        invalid_decision_reward=float(c.get("invalid_decision_reward", -1.5)),
        num_train_epochs=int(c.get("num_train_epochs", 1)),
        learning_rate=float(c.get("learning_rate", 1e-6)),
        per_device_train_batch_size=int(c.get("per_device_train_batch_size", c.get("batch_size", 1))),
        gradient_accumulation_steps=int(c.get("gradient_accumulation_steps", c.get("grad_accum", 1))),
        generation_batch_size=(
            int(c["generation_batch_size"]) if c.get("generation_batch_size") is not None else None
        ),
        steps_per_generation=(
            int(c["steps_per_generation"]) if c.get("steps_per_generation") is not None else None
        ),
        logging_steps=int(c.get("logging_steps", 1)),
        max_prompt_length=int(c.get("max_prompt_length", 32768)),
        max_completion_length=int(c.get("max_completion_length", 1024)),
        temperature=float(c.get("temperature", 1.0)),
        format_instruction=(
            str(c["format_instruction"]) if c.get("format_instruction") is not None else None
        ),
        log_completions=bool(c.get("log_completions", False)),
        num_completions_to_print=(
            int(c["num_completions_to_print"])
            if c.get("num_completions_to_print") is not None
            else None
        ),
        report_to=[str(x) for x in c.get("report_to", [])],
        run_name=str(c["run_name"]) if c.get("run_name") else None,
        wandb_project=str(c.get("wandb_project", "trading_r1")),
    )
    return train_grpo(cfg)
