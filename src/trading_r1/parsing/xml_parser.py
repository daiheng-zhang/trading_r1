"""Parsing helpers for XML-tagged investment theses."""

from __future__ import annotations

import re
from dataclasses import dataclass


TAG_PATTERN = re.compile(r"<([a-zA-Z_][\w\-]*)>(.*?)</\1>", re.DOTALL)


@dataclass
class XMLSection:
    tag: str
    content: str
    start: int
    end: int


def extract_xml_sections(text: str) -> list[XMLSection]:
    sections: list[XMLSection] = []
    for match in TAG_PATTERN.finditer(text):
        sections.append(
            XMLSection(
                tag=match.group(1).strip().lower(),
                content=match.group(2).strip(),
                start=match.start(),
                end=match.end(),
            )
        )
    return sections


def get_analysis_sections(text: str) -> list[XMLSection]:
    """Returns all sections except `<think>`."""
    return [s for s in extract_xml_sections(text) if s.tag != "think"]


def get_non_conclusion_analysis_sections(text: str) -> list[XMLSection]:
    return [s for s in get_analysis_sections(text) if s.tag != "conclusion"]


def has_single_think_block(text: str) -> bool:
    think_sections = [s for s in extract_xml_sections(text) if s.tag == "think"]
    return len(think_sections) == 1


def has_conclusion(text: str) -> bool:
    return any(s.tag == "conclusion" for s in get_analysis_sections(text))


def count_analysis_sections(text: str) -> int:
    """Number of analysis sections excluding `conclusion` and `think`."""
    return len([s for s in get_analysis_sections(text) if s.tag != "conclusion"])


def has_table_markdown(text: str) -> bool:
    return "|" in text and "\n" in text and ("---" in text or ":-" in text)
