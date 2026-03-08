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


def append_instruction_to_user_turn(prompt: str, instruction: str | None) -> str:
    """Append a formatting instruction to the user turn without breaking chat layout."""
    normalized = ensure_chat_prompt_has_assistant_turn(prompt)
    text = str(instruction or "").strip()
    if not text:
        return normalized

    marker = "\n<|assistant|>\n"
    if marker not in normalized:
        return normalized

    user_block, assistant_block = normalized.split(marker, 1)
    if not user_block.startswith("<|user|>\n"):
        return normalized

    user_text = user_block[len("<|user|>\n") :]
    if text in user_text:
        return normalized

    combined_user_text = user_text.rstrip("\n") + "\n\n" + text
    if assistant_block:
        return build_chat_prompt(combined_user_text, assistant_block)
    return build_chat_prompt(combined_user_text)
