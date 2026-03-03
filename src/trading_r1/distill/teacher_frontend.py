"""Teacher front-end recommendation generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from trading_r1.actions import ACTIONS
from trading_r1.parsing.decision_parser import normalize_action


_HF_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _extract_action_from_free_text(text: str) -> str | None:
    direct = normalize_action(text)
    if direct in ACTIONS:
        return direct

    cleaned = re.sub(r"[^A-Za-z_ -]+", " ", text.upper())
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("_", " ").replace("-", " ")

    patterns = [
        ("STRONG SELL", "STRONG_SELL"),
        ("STRONG BUY", "STRONG_BUY"),
        ("SELL", "SELL"),
        ("BUY", "BUY"),
        ("HOLD", "HOLD"),
    ]
    for pat, action in patterns:
        if re.search(rf"\\b{pat}\\b", cleaned):
            return action
    return None


@dataclass
class TeacherConfig:
    provider: str = "mock"  # mock | openai | hf
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    temperature: float = 0.2
    hf_device_map: str = "auto"
    hf_load_in_4bit: bool = True
    hf_max_new_tokens: int = 16
    hf_top_p: float = 0.9
    hf_do_sample: bool = False
    hf_trust_remote_code: bool = True


def _mock_recommendation(input_text: str, label_hint: str | None = None) -> str:
    if label_hint in ACTIONS:
        return str(label_hint)

    low = input_text.lower()
    bullish = low.count("upgrade") + low.count("beat") + low.count("growth") + low.count("bullish")
    bearish = low.count("downgrade") + low.count("miss") + low.count("decline") + low.count("bearish")
    if bullish - bearish >= 2:
        return "BUY"
    if bearish - bullish >= 2:
        return "SELL"
    return "HOLD"


def _openai_recommendation(cfg: TeacherConfig, input_text: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package required for provider=openai") from exc

    if not cfg.api_key:
        raise RuntimeError("OpenAI API key missing for provider=openai")

    client = OpenAI(api_key=cfg.api_key)
    prompt = (
        "You are a trading teacher model. Given the market context, output only one token from this set: "
        "STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY. No explanation.\n\n"
        f"Context:\n{input_text}"
    )

    resp = client.responses.create(
        model=cfg.model,
        temperature=cfg.temperature,
        input=prompt,
    )
    action = _extract_action_from_free_text(resp.output_text.strip())
    if action in ACTIONS:
        return action
    return "HOLD"


def _build_hf_cache_key(cfg: TeacherConfig) -> str:
    return json.dumps(
        {
            "model": cfg.model,
            "hf_device_map": cfg.hf_device_map,
            "hf_load_in_4bit": cfg.hf_load_in_4bit,
            "hf_trust_remote_code": cfg.hf_trust_remote_code,
        },
        sort_keys=True,
    )


def _load_hf_model_and_tokenizer(cfg: TeacherConfig) -> tuple[Any, Any]:
    cache_key = _build_hf_cache_key(cfg)
    cached = _HF_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "provider=hf requires torch + transformers. Install train extras first."
        ) from exc

    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        else torch.float16
    )

    model_kwargs: dict[str, Any] = {
        "device_map": cfg.hf_device_map,
        "trust_remote_code": cfg.hf_trust_remote_code,
    }

    if cfg.hf_load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "provider=hf with hf_load_in_4bit=true requires a CUDA GPU. "
                "Set hf_load_in_4bit=false for non-CUDA fallback."
            )
        try:
            import bitsandbytes  # type: ignore  # noqa: F401
            from transformers import BitsAndBytesConfig  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "hf_load_in_4bit=true requires bitsandbytes. Install with "
                "`python -m pip install bitsandbytes` or set hf_load_in_4bit=false."
            ) from exc

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model,
        trust_remote_code=cfg.hf_trust_remote_code,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model, **model_kwargs)
    model.eval()

    _HF_MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _token_length(input_ids: Any) -> int:
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    return 0


def _first_sequence_ids(generated: Any) -> Any:
    if hasattr(generated, "tolist"):
        generated = generated.tolist()
    if isinstance(generated, list) and generated:
        first = generated[0]
        if hasattr(first, "tolist"):
            first = first.tolist()
        return first
    return generated


def _hf_recommendation(cfg: TeacherConfig, input_text: str) -> str:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("provider=hf requires torch") from exc

    tokenizer, model = _load_hf_model_and_tokenizer(cfg)

    prompt = (
        "You are a trading teacher model.\n"
        "Return exactly one action token from this set and nothing else:\n"
        "STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY.\n\n"
        f"Context:\n{input_text}"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = getattr(model, "device", None)
    if model_device is not None and str(model_device) != "meta":
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(cfg.hf_max_new_tokens),
        "do_sample": bool(cfg.hf_do_sample),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if cfg.hf_do_sample:
        gen_kwargs["temperature"] = float(cfg.temperature)
        gen_kwargs["top_p"] = float(cfg.hf_top_p)

    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)

    prompt_len = _token_length(inputs.get("input_ids"))
    first_ids = _first_sequence_ids(generated)
    new_ids = first_ids[prompt_len:] if isinstance(first_ids, list) and prompt_len >= 0 else first_ids

    decoded_new = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    action = _extract_action_from_free_text(decoded_new)
    if action in ACTIONS:
        return action

    decoded_full = tokenizer.decode(first_ids, skip_special_tokens=True).strip()
    action = _extract_action_from_free_text(decoded_full)
    if action in ACTIONS:
        return action

    return "HOLD"


def generate_frontend_recommendation(
    input_text: str,
    config: dict[str, Any],
    label_hint: str | None = None,
) -> str:
    cfg = TeacherConfig(
        provider=str(config.get("provider", "mock")),
        model=str(config.get("model", "gpt-4.1-mini")),
        api_key=config.get("api_key"),
        temperature=float(config.get("temperature", 0.2)),
        hf_device_map=str(config.get("hf_device_map", "auto")),
        hf_load_in_4bit=bool(config.get("hf_load_in_4bit", True)),
        hf_max_new_tokens=int(config.get("hf_max_new_tokens", 16)),
        hf_top_p=float(config.get("hf_top_p", 0.9)),
        hf_do_sample=bool(config.get("hf_do_sample", False)),
        hf_trust_remote_code=bool(config.get("hf_trust_remote_code", True)),
    )

    if cfg.provider == "openai":
        return _openai_recommendation(cfg, input_text)
    if cfg.provider == "hf":
        return _hf_recommendation(cfg, input_text)
    return _mock_recommendation(input_text, label_hint=label_hint)


def write_frontend_decisions(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
