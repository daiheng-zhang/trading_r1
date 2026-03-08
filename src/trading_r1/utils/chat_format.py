"""Utilities for stable chat-style prompt formatting."""

from __future__ import annotations


def build_chat_prompt(user_text: str, assistant_text: str | None = None) -> str:
    """Format prompt text as `<|user|> ... <|assistant|>` conversation."""
    user = str(user_text).rstrip("\n")
    prefix = f"<|user|>\n{user}\n<|assistant|>\n"
    if assistant_text is None:
        return prefix
    return prefix + str(assistant_text).lstrip("\n")


def ensure_chat_prompt_has_assistant_turn(prompt: str) -> str:
    """Normalize arbitrary prompt text into a generation-ready assistant turn."""
    text = str(prompt)

    if "<|assistant|>" in text:
        if text.rstrip().endswith("<|assistant|>"):
            return text.rstrip() + "\n"
        return text

    if "<|user|>" in text:
        return text.rstrip("\n") + "\n<|assistant|>\n"

    return build_chat_prompt(text, assistant_text=None)
