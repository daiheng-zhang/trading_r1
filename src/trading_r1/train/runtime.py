"""Runtime hardware resolution for training backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingRuntime:
    device: str
    precision: str
    bf16: bool
    fp16: bool
    use_cpu: bool
    torch_dtype: Any


def _mps_is_available(torch_module: Any) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps is not None and getattr(mps, "is_available", lambda: False)())


def resolve_training_runtime(
    precision: str = "auto",
    *,
    use_cpu: bool | None = None,
    torch_module: Any | None = None,
) -> TrainingRuntime:
    if torch_module is None:
        import torch as torch_module  # type: ignore

    normalized_precision = str(precision).strip().lower() or "auto"
    if normalized_precision not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(
            "precision must be one of: auto, bf16, fp16, fp32. "
            f"Received {precision!r}."
        )

    if use_cpu is True:
        if normalized_precision in {"bf16", "fp16"}:
            raise RuntimeError(
                f"precision={normalized_precision} is not supported with use_cpu=true. "
                "Use precision=fp32 or precision=auto for CPU training."
            )
        return TrainingRuntime(
            device="cpu",
            precision="fp32",
            bf16=False,
            fp16=False,
            use_cpu=True,
            torch_dtype=torch_module.float32,
        )

    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda is not None and getattr(cuda, "is_available", lambda: False)())
    bf16_supported = bool(
        cuda_available and getattr(cuda, "is_bf16_supported", lambda: False)()
    )
    mps_available = _mps_is_available(torch_module)

    if normalized_precision == "auto":
        if cuda_available:
            if bf16_supported:
                return TrainingRuntime(
                    device="cuda",
                    precision="bf16",
                    bf16=True,
                    fp16=False,
                    use_cpu=False,
                    torch_dtype=torch_module.bfloat16,
                )
            return TrainingRuntime(
                device="cuda",
                precision="fp16",
                bf16=False,
                fp16=True,
                use_cpu=False,
                torch_dtype=torch_module.float16,
            )
        if mps_available:
            return TrainingRuntime(
                device="mps",
                precision="fp32",
                bf16=False,
                fp16=False,
                use_cpu=False,
                torch_dtype=torch_module.float32,
            )
        return TrainingRuntime(
            device="cpu",
            precision="fp32",
            bf16=False,
            fp16=False,
            use_cpu=True,
            torch_dtype=torch_module.float32,
        )

    if normalized_precision == "bf16":
        if not cuda_available or not bf16_supported:
            raise RuntimeError("precision=bf16 requires a CUDA GPU with bf16 support.")
        return TrainingRuntime(
            device="cuda",
            precision="bf16",
            bf16=True,
            fp16=False,
            use_cpu=False,
            torch_dtype=torch_module.bfloat16,
        )

    if normalized_precision == "fp16":
        if not cuda_available:
            raise RuntimeError("precision=fp16 requires a CUDA GPU.")
        return TrainingRuntime(
            device="cuda",
            precision="fp16",
            bf16=False,
            fp16=True,
            use_cpu=False,
            torch_dtype=torch_module.float16,
        )

    if cuda_available:
        return TrainingRuntime(
            device="cuda",
            precision="fp32",
            bf16=False,
            fp16=False,
            use_cpu=False,
            torch_dtype=torch_module.float32,
        )
    if mps_available:
        return TrainingRuntime(
            device="mps",
            precision="fp32",
            bf16=False,
            fp16=False,
            use_cpu=False,
            torch_dtype=torch_module.float32,
        )
    return TrainingRuntime(
        device="cpu",
        precision="fp32",
        bf16=False,
        fp16=False,
        use_cpu=True,
        torch_dtype=torch_module.float32,
    )
