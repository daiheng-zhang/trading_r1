from __future__ import annotations

import sys
import types
from pathlib import Path

from trading_r1.distill.teacher_frontend import _HF_MODEL_CACHE, generate_frontend_recommendation
from trading_r1.distill.trace_stitcher import DistillConfig, distill_sft_and_grpo
from trading_r1.utils.io import write_jsonl


def _install_fake_hf_stack(monkeypatch, decoded_text: str = "Strong Buy") -> dict:
    counters = {"tokenizer_loads": 0, "model_loads": 0, "model_kwargs": None}

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    fake_torch = types.SimpleNamespace(
        cuda=_FakeCuda(),
        bfloat16="bf16",
        float16="fp16",
        no_grad=lambda: _NoGrad(),
    )

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, prompt, return_tensors="pt"):
            _ = prompt
            _ = return_tensors
            return {"input_ids": [[101, 102, 103]], "attention_mask": [[1, 1, 1]]}

        def decode(self, token_ids, skip_special_tokens=True):
            _ = token_ids
            _ = skip_special_tokens
            return decoded_text

    class FakeModel:
        device = None

        def eval(self):
            return self

        def generate(self, **kwargs):
            _ = kwargs
            return [[101, 102, 103, 201, 202]]

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            _ = args, kwargs
            counters["tokenizer_loads"] += 1
            return FakeTokenizer()

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            _ = args
            counters["model_loads"] += 1
            counters["model_kwargs"] = kwargs
            return FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModelForCausalLM,
        BitsAndBytesConfig=BitsAndBytesConfig,
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.SimpleNamespace(__version__="0.44.0"))
    _HF_MODEL_CACHE.clear()
    return counters


def test_hf_provider_dispatch_and_normalization(monkeypatch) -> None:
    counters = _install_fake_hf_stack(monkeypatch, decoded_text="STRONG-BUY")
    action = generate_frontend_recommendation(
        input_text="example",
        config={"provider": "hf", "model": "Qwen/Qwen3-4B-Instruct-2507"},
        label_hint=None,
    )
    assert action == "STRONG_BUY"
    assert counters["model_loads"] == 1
    assert counters["tokenizer_loads"] == 1
    assert counters["model_kwargs"]["device_map"] == "auto"
    assert "quantization_config" in counters["model_kwargs"]


def test_hf_provider_invalid_output_falls_back_to_hold(monkeypatch) -> None:
    _install_fake_hf_stack(monkeypatch, decoded_text="No clear answer provided.")
    action = generate_frontend_recommendation(
        input_text="example",
        config={"provider": "hf", "model": "Qwen/Qwen3-4B-Instruct-2507"},
        label_hint=None,
    )
    assert action == "HOLD"


def test_hf_provider_uses_model_cache(monkeypatch) -> None:
    counters = _install_fake_hf_stack(monkeypatch, decoded_text="BUY")
    cfg = {"provider": "hf", "model": "Qwen/Qwen3-4B-Instruct-2507"}
    _ = generate_frontend_recommendation("ctx-1", cfg)
    _ = generate_frontend_recommendation("ctx-2", cfg)
    assert counters["model_loads"] == 1
    assert counters["tokenizer_loads"] == 1


def test_mock_provider_regression_with_label_hint() -> None:
    action = generate_frontend_recommendation(
        input_text="any",
        config={"provider": "mock"},
        label_hint="SELL",
    )
    assert action == "SELL"


def test_distill_with_hf_provider_monkeypatched(tmp_path: Path, monkeypatch) -> None:
    _install_fake_hf_stack(monkeypatch, decoded_text="Strong Buy")

    samples_path = tmp_path / "samples.jsonl"
    sft_out = tmp_path / "sft.jsonl"
    grpo_out = tmp_path / "grpo.jsonl"

    write_jsonl(
        samples_path,
        [
            {
                "sample_id": "AAPL_2024-07-15_v00",
                "ticker": "AAPL",
                "trade_date": "2024-07-15",
                "input_text": "<price>...</price><news>...</news><fundamentals>...</fundamentals>",
                "label_action": "HOLD",
                "meta": {},
            },
            {
                "sample_id": "AAPL_2024-07-16_v00",
                "ticker": "AAPL",
                "trade_date": "2024-07-16",
                "input_text": "<price>...</price><news>...</news><fundamentals>...</fundamentals>",
                "label_action": "HOLD",
                "meta": {},
            },
        ],
    )

    sft, grpo = distill_sft_and_grpo(
        DistillConfig(
            samples_path=str(samples_path),
            sft_output_path=str(sft_out),
            grpo_output_path=str(grpo_out),
            frontend_config={"provider": "hf", "model": "Qwen/Qwen3-4B-Instruct-2507"},
            planner_config={"max_steps": 6},
        )
    )

    assert len(sft) == 2
    assert len(grpo) == 2
    assert "DECISION: [[[STRONG_BUY]]]" in sft[0]["target_text"]
    assert sft_out.exists()
    assert grpo_out.exists()
