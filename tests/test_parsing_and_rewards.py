from __future__ import annotations

import math

from trading_r1.parsing.decision_parser import extract_decision
from trading_r1.reward.decision_reward import MATRIX, decision_reward
from trading_r1.reward.evidence_reward import evidence_reward
from trading_r1.reward.structure_reward import structure_reward


def _completion_with_n_sections(n: int) -> str:
    parts = ["<think>\n- plan\n</think>"]
    for i in range(n):
        parts.append(
            f"<sec{i}>\n### H{i}\n"
            "- Bullet one with enough words to pass threshold.\n"
            "- Bullet two with enough words to pass threshold.\n"
            "- Bullet three with enough words to pass threshold.\n"
            "- Bullet four with enough words to pass threshold.\n"
            f"</sec{i}>"
        )
    parts.append("<conclusion>\n### Conclusion\n- done\n</conclusion>")
    parts.append("DECISION: [[[BUY]]]")
    return "\n".join(parts)


def test_decision_parser_from_noisy_tail() -> None:
    text = "abc\nnoise\nDECISION: [[[strong buy]]]\ntrailing"
    # parser takes last match, not necessarily strict last line.
    assert extract_decision(text) == "STRONG_BUY"


def test_structure_reward_penalizes_section_count_outside_range() -> None:
    r4 = structure_reward(_completion_with_n_sections(4))
    r6 = structure_reward(_completion_with_n_sections(6))
    r8 = structure_reward(_completion_with_n_sections(8))
    assert r6 > r4
    assert r6 > r8


def test_evidence_reward_drops_without_quote_and_source() -> None:
    with_evidence = (
        "<think>ok</think>"
        "<fundamentals>\n"
        "- Opinion text with enough words here *quoted evidence* `source:a`\n"
        "- Opinion text with enough words here *quoted evidence* `source:b`\n"
        "- Opinion text with enough words here *quoted evidence* `source:c`\n"
        "- Opinion text with enough words here *quoted evidence* `source:d`\n"
        "</fundamentals>"
        "<conclusion>done</conclusion>\n"
        "DECISION: [[[BUY]]]"
    )
    without_evidence = with_evidence.replace("*quoted evidence*", "quoted evidence").replace("`source:a`", "").replace("`source:b`", "").replace("`source:c`", "").replace("`source:d`", "")
    assert evidence_reward(with_evidence) > evidence_reward(without_evidence)


def test_decision_reward_matrix_constants() -> None:
    assert math.isclose(MATRIX["STRONG_SELL"]["STRONG_BUY"], -2.25)
    assert math.isclose(MATRIX["STRONG_BUY"]["STRONG_SELL"], -2.00)
    assert math.isclose(decision_reward("BUY", "BUY"), 1.0)
