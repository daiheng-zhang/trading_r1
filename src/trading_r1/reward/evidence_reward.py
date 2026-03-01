"""Stage II evidence reward from Trading-R1 appendix."""

from __future__ import annotations

import re

from trading_r1.parsing.xml_parser import get_non_conclusion_analysis_sections


_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)(.+)$", re.MULTILINE)
_ITALIC_QUOTE_RE = re.compile(r"\*[^*]+\*")
_CODE_SOURCE_RE = re.compile(r"`[^`]+`")
_FIRST_CITATION_RE = re.compile(r"(\*[^*]+\*|`[^`]+`)" )


def _harmonic_mean(values: list[float], floor: float = 0.01) -> float:
    if not values:
        return 0.0
    denom = sum(1.0 / max(v, floor) for v in values)
    return len(values) / denom if denom else 0.0


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def extract_bullets(section_content: str) -> list[str]:
    return [m.group(1).strip() for m in _BULLET_RE.finditer(section_content)]


def opinion_text_before_citation(bullet: str) -> str:
    match = _FIRST_CITATION_RE.search(bullet)
    if match:
        return bullet[: match.start()].strip()
    return bullet.strip()


def opinion_quality_score(bullet: str) -> float:
    w_min = 15
    w_max = 90

    opinion = opinion_text_before_citation(bullet)
    w_op = _word_count(opinion)

    has_quote = len(_ITALIC_QUOTE_RE.findall(bullet)) > 0
    has_source = len(_CODE_SOURCE_RE.findall(bullet)) > 0
    has_citation_pair = has_quote and has_source

    if has_citation_pair:
        if w_min <= w_op <= w_max:
            return 1.0
        if w_op < w_min:
            return w_op / w_min
        return max(0.5, 1.0 - 0.02 * (w_op - w_max))

    return min(0.3, (w_op / w_min) * 0.3)


def bullet_evidence_score(bullet: str) -> float:
    has_quote = 1.0 if _ITALIC_QUOTE_RE.search(bullet) else 0.0
    has_source = 1.0 if _CODE_SOURCE_RE.search(bullet) else 0.0
    r_opinion = opinion_quality_score(bullet)
    return 0.4 * r_opinion + 0.35 * has_quote + 0.25 * has_source


def bullet_count_reward(bullets: list[str]) -> float:
    n = len(bullets)
    if 4 <= n <= 7:
        return 1.0
    if n < 4:
        return n / 4.0
    return max(0.3, 1.0 - 0.1 * (n - 7))


def section_evidence_reward(section_content: str) -> float:
    bullets = extract_bullets(section_content)
    if not bullets:
        return 0.0

    r_count = bullet_count_reward(bullets)
    bullet_scores = [bullet_evidence_score(b) for b in bullets]
    hmean = _harmonic_mean(bullet_scores)
    return 0.3 * r_count + 0.7 * hmean


def evidence_reward(completion: str) -> float:
    sections = get_non_conclusion_analysis_sections(completion)
    if not sections:
        return 0.0

    section_scores = [section_evidence_reward(s.content) for s in sections]
    return _harmonic_mean(section_scores)
