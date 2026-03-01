"""Stage I structure reward from Trading-R1 appendix."""

from __future__ import annotations

import re

from trading_r1.parsing.xml_parser import get_analysis_sections, has_conclusion, has_table_markdown


_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def section_count_reward(s_count: int) -> float:
    if 5 <= s_count <= 7:
        return 1.0
    if s_count < 5:
        return max(0.3, (s_count / 5.0) * 0.7)
    return max(0.3, 1.0 - 0.15 * (s_count - 7))


def section_structural_reward(content: str, min_words: int = 50) -> float:
    if _word_count(content) < min_words:
        return 0.2

    has_headers = 1.0 if _HEADER_RE.search(content) else 0.0
    has_bullets = 1.0 if _BULLET_RE.search(content) else 0.0
    has_bold = 1.0 if _BOLD_RE.search(content) else 0.0
    has_table = 1.0 if has_table_markdown(content) else 0.0

    return 0.3 * has_headers + 0.4 * has_bullets + 0.2 * has_bold + 0.1 * has_table


def structure_reward(completion: str) -> float:
    sections = get_analysis_sections(completion)
    if not sections or not has_conclusion(completion):
        return 0.0

    analysis_sections = [s for s in sections if s.tag != "conclusion"]
    s_count = len(analysis_sections)
    r_count = section_count_reward(s_count)

    r_struct_mean = 0.0
    if sections:
        r_struct_mean = sum(section_structural_reward(s.content) for s in sections) / len(sections)

    return 0.6 * r_count + 0.4 * r_struct_mean
