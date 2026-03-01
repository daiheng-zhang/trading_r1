"""Stage-wise GRPO training entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_r1.reward.aggregate import aggregate_reward
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
        from trl import GRPOConfig as TRLGRPOConfig  # type: ignore
        from trl import GRPOTrainer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "GRPO mode=trl requires `trl` and `datasets`. Install optional train dependencies."
        ) from exc

    rows = read_jsonl(cfg.train_path)
    if not rows:
        raise RuntimeError(f"No GRPO rows found at {cfg.train_path}")

    dataset = datasets.Dataset.from_list(
        [
            {
                "prompt": str(r.get("prompt", "")),
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

    trl_cfg = TRLGRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=1,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        max_prompt_length=32768,
        max_completion_length=1024,
        num_generations=cfg.group_size,
        beta=cfg.kl_beta,
        epsilon=cfg.clip_eps,
    )

    trainer = GRPOTrainer(
        model=cfg.model_name_or_path,
        reward_funcs=reward_func,
        args=trl_cfg,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)

    metrics = {
        "mode": "trl",
        "stage": cfg.stage,
        "samples": len(rows),
        "group_size": cfg.group_size,
        "clip_eps": cfg.clip_eps,
        "kl_beta": cfg.kl_beta,
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
        group_size=int(c.get("group_size", 8)),
        clip_eps=float(c.get("clip_eps", 0.2)),
        kl_beta=float(c.get("kl_beta", 0.03)),
        invalid_decision_reward=float(c.get("invalid_decision_reward", -1.5)),
    )
    return train_grpo(cfg)
