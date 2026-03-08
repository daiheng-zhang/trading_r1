from __future__ import annotations

import types

import pytest

from trading_r1.train.runtime import resolve_training_runtime


def _fake_torch(*, cuda: bool, bf16: bool, mps: bool = False) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        float16="fp16",
        float32="fp32",
        bfloat16="bf16",
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda,
            is_bf16_supported=lambda: bf16,
        ),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps),
        ),
    )


def test_runtime_defaults_to_cpu_fp32_without_accelerator() -> None:
    runtime = resolve_training_runtime(torch_module=_fake_torch(cuda=False, bf16=False))
    assert runtime.device == "cpu"
    assert runtime.precision == "fp32"
    assert runtime.use_cpu is True
    assert runtime.bf16 is False
    assert runtime.fp16 is False
    assert runtime.torch_dtype == "fp32"


def test_runtime_prefers_bf16_on_supported_cuda() -> None:
    runtime = resolve_training_runtime(torch_module=_fake_torch(cuda=True, bf16=True))
    assert runtime.device == "cuda"
    assert runtime.precision == "bf16"
    assert runtime.use_cpu is False
    assert runtime.bf16 is True
    assert runtime.fp16 is False
    assert runtime.torch_dtype == "bf16"


def test_runtime_falls_back_to_fp16_on_cuda_without_bf16() -> None:
    runtime = resolve_training_runtime(torch_module=_fake_torch(cuda=True, bf16=False))
    assert runtime.device == "cuda"
    assert runtime.precision == "fp16"
    assert runtime.use_cpu is False
    assert runtime.bf16 is False
    assert runtime.fp16 is True
    assert runtime.torch_dtype == "fp16"


def test_runtime_use_cpu_override_forces_fp32() -> None:
    runtime = resolve_training_runtime(
        use_cpu=True,
        torch_module=_fake_torch(cuda=True, bf16=True),
    )
    assert runtime.device == "cpu"
    assert runtime.precision == "fp32"
    assert runtime.use_cpu is True
    assert runtime.torch_dtype == "fp32"


def test_runtime_rejects_half_precision_on_cpu() -> None:
    with pytest.raises(RuntimeError):
        resolve_training_runtime(
            precision="bf16",
            torch_module=_fake_torch(cuda=False, bf16=False),
        )
