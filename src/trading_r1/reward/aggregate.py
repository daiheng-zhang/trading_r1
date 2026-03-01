"""Aggregate reward helpers across Stage I/II/III."""

from __future__ import annotations

from trading_r1.parsing.decision_parser import extract_decision
from trading_r1.reward.decision_reward import decision_reward
from trading_r1.reward.evidence_reward import evidence_reward
from trading_r1.reward.structure_reward import structure_reward


STAGE_WEIGHTS: dict[int, tuple[float, float, float]] = {
    1: (1.0, 0.0, 0.0),
    2: (0.4, 0.6, 0.0),
    3: (0.3, 0.5, 0.2),
}


def aggregate_reward(
    completion: str,
    ground_truth_action: str,
    stage: int,
    custom_weights: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    weights = custom_weights or STAGE_WEIGHTS.get(stage, STAGE_WEIGHTS[3])
    w_struct, w_evid, w_dec = weights

    r_struct = structure_reward(completion)
    r_evid = evidence_reward(completion)
    pred = extract_decision(completion, strict_last_three_lines=False)
    r_dec = decision_reward(pred, ground_truth_action, scale=1.0)

    total = w_struct * r_struct + w_evid * r_evid + w_dec * r_dec
    return {
        "reward_total": float(total),
        "reward_structure": float(r_struct),
        "reward_evidence": float(r_evid),
        "reward_decision": float(r_dec),
    }
