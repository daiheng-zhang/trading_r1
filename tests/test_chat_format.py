from __future__ import annotations

from trading_r1.utils.chat_format import build_chat_prompt, ensure_chat_prompt_has_assistant_turn


def test_build_chat_prompt_with_assistant_text() -> None:
    text = build_chat_prompt("market context", "analysis")
    assert text == "<|user|>\nmarket context\n<|assistant|>\nanalysis"


def test_ensure_chat_prompt_wraps_plain_text() -> None:
    text = ensure_chat_prompt_has_assistant_turn("plain prompt")
    assert text == "<|user|>\nplain prompt\n<|assistant|>\n"


def test_ensure_chat_prompt_adds_assistant_turn_after_user_block() -> None:
    text = ensure_chat_prompt_has_assistant_turn("<|user|>\nctx")
    assert text == "<|user|>\nctx\n<|assistant|>\n"


def test_ensure_chat_prompt_keeps_existing_assistant_generation_prefix() -> None:
    text = ensure_chat_prompt_has_assistant_turn("<|user|>\nctx\n<|assistant|>")
    assert text == "<|user|>\nctx\n<|assistant|>\n"


def test_ensure_chat_prompt_keeps_existing_completion_text() -> None:
    prompt = "<|user|>\nctx\n<|assistant|>\nexisting output"
    assert ensure_chat_prompt_has_assistant_turn(prompt) == prompt
