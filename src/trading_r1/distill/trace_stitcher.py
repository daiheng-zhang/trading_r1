"""Stitch reverse-planned reasoning into SFT and GRPO artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_r1.actions import ACTIONS
from trading_r1.distill.reverse_planner import reconstruct_reasoning_steps
from trading_r1.distill.teacher_frontend import generate_frontend_recommendation
from trading_r1.schemas import GRPOBatchItem, SFTTarget
from trading_r1.utils.chat_format import build_chat_prompt
from trading_r1.utils.io import read_jsonl, write_jsonl


DEFAULT_ALLOWED_TAGS = [
    "fundamentals",
    "technical",
    "news",
    "valuation",
    "risk",
    "macro",
    "conclusion",
]


@dataclass
class DistillConfig:
    samples_path: str
    sft_output_path: str
    grpo_output_path: str
    frontend_config: dict[str, Any]
    planner_config: dict[str, Any]


def _analysis_sections() -> list[tuple[str, str]]:
    return [
        ("fundamentals", "Assess balance sheet, profitability, cash flow, and valuation anchors."),
        ("technical", "Review trend, momentum, and volatility indicators with near-term context."),
        ("news", "Summarize recent catalysts from 3/10/30-day buckets and likely impact."),
        ("valuation", "Compare growth expectations and risk premium implications."),
        ("risk", "Enumerate downside scenarios, invalidation triggers, and asymmetric risks."),
        ("macro", "Account for rates, inflation, and regime sensitivity."),
    ]


def build_target_text(input_text: str, decision: str, steps: list[str]) -> str:
    if decision not in ACTIONS:
        decision = "HOLD"

    think_lines = [f"- {s}" for s in steps]
    think = "<think>\n### Plan\n" + "\n".join(think_lines) + "\n</think>"

    sections = []
    for tag, desc in _analysis_sections():
        section = (
            f"<{tag}>\n"
            f"### {tag.title()}\n"
            f"- Opinion: {desc} *Evidence placeholder from context.* `source:{tag}`\n"
            f"- Opinion: Signal reliability and data quality are reviewed. *Quoted context snippet.* `source:{tag}`\n"
            f"- Opinion: Uncertainty is explicitly tracked. *Potential counter-signal noted.* `source:{tag}`\n"
            f"- Opinion: Position sizing should follow conviction and volatility. *Risk budget note.* `source:{tag}`\n"
            f"</{tag}>"
        )
        sections.append(section)

    conclusion = (
        "<conclusion>\n"
        "### Conclusion\n"
        "- The thesis balances structure, evidence, and risk control.\n"
        f"- Final recommendation reflects current conviction: {decision}.\n"
        "</conclusion>"
    )

    return "\n\n".join([think, *sections, conclusion]) + f"\nDECISION: [[[{decision}]]]"


def distill_sft_and_grpo(cfg: DistillConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = read_jsonl(cfg.samples_path)

    sft_targets: list[dict[str, Any]] = []
    grpo_items: list[dict[str, Any]] = []

    for sample in samples:
        input_text = str(sample.get("input_text", ""))
        label_hint = str(sample.get("label_action", "HOLD"))

        decision = generate_frontend_recommendation(
            input_text=input_text,
            config=cfg.frontend_config,
            label_hint=label_hint,
        )
        steps = reconstruct_reasoning_steps(
            input_text=input_text,
            frontend_decision=decision,
            config=cfg.planner_config,
        )
        target_text = build_target_text(input_text, decision, steps)

        sft_targets.append(
            SFTTarget(
                sample_id=str(sample.get("sample_id", "")),
                input_text=input_text,
                target_text=target_text,
            ).to_dict()
        )

        grpo_items.append(
            GRPOBatchItem(
                sample_id=str(sample.get("sample_id", "")),
                prompt=build_chat_prompt(input_text),
                ground_truth_action=label_hint if label_hint in ACTIONS else decision,
                reward_context={"allowed_tags": DEFAULT_ALLOWED_TAGS},
            ).to_dict()
        )

    write_jsonl(cfg.sft_output_path, sft_targets)
    write_jsonl(cfg.grpo_output_path, grpo_items)
    return sft_targets, grpo_items


def distill_sft_and_grpo_from_config(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    c = config.get("distill", config)
    cfg = DistillConfig(
        samples_path=c["samples_path"],
        sft_output_path=c["sft_output_path"],
        grpo_output_path=c["grpo_output_path"],
        frontend_config=dict(c.get("frontend", {"provider": "mock"})),
        planner_config=dict(c.get("planner", {"max_steps": 6})),
    )
    return distill_sft_and_grpo(cfg)
